use arrow::array::OffsetSizeTrait;
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use datafusion_common::utils::proxy::VecAllocExt;
use datafusion_expr_common::groups_accumulator::BlocksIndex;
use std::collections::VecDeque;
use std::ops::Index;

/// When `FIXED_BLOCK_SIZING` is true, the block size is the `Self::block_size` otherwise,
/// the callers control the block size
pub struct BlockedOffsetBufferBuilder<const FIXED_BLOCK_SIZING: bool, O: OffsetSizeTrait>
{
    /// Using `VecDeque` so we can remove the first block and reclaim memory
    blocks: VecDeque<Vec<O>>,

    /// The size of each block
    block_size: usize,

    /// The total number of items, not the number of offset since in each block there is the initial offset
    len: usize,

    /// The index of the current block
    current_block_index: usize,

    /// The last offset in the current block
    last_offset: O,

    /// This will equal the number of blocks can be emitted
    ///
    /// We have this rather than using `blocks.len()` since we don't want to add a condition
    /// on every push that checks if we have a block, and if not adds 0 as the initial offset
    ///
    /// And we can't always have a block with 0 since then how could we know when all blocks were emitted
    /// if we keep adding empty block on finish
    number_of_blocks: usize,

    pending_block: bool,

    memory: usize,
}

impl<const FIXED_BLOCK_SIZING: bool, O: OffsetSizeTrait>
    BlockedOffsetBufferBuilder<FIXED_BLOCK_SIZING, O>
{
    pub fn new(mut block_size: usize) -> Self {
        assert_ne!(block_size, 0, "block size must be greater than 0");

        // Add 1 to the block size to account for the initial offset
        block_size += 1;

        let last_offset = O::zero();
        let blocks = VecDeque::from(vec![vec![last_offset]]);
        let memory = blocks.capacity() * size_of::<Vec<O>>() + blocks[0].allocated_size();
        BlockedOffsetBufferBuilder {
            blocks,
            block_size,
            len: 0,
            current_block_index: 0,
            last_offset,
            number_of_blocks: 1,
            pending_block: false,
            memory,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn allocated_size(&self) -> usize {
        self.memory
    }

    /// Get the number of elements in the current block (not the number of offsets since the first offset is always 0)
    pub(crate) fn current_block_len(&self) -> usize {
        self.blocks[self.current_block_index].len() - 1
    }

    pub fn last_offset(&self) -> O {
        self.last_offset
    }

    pub fn start_new_block(&mut self) {
        // Don't add to number of blocks since we might not insert into it
        self.current_block_index += 1;
        self.last_offset = O::zero();
        let prev_capacity = self.blocks.capacity();
        let new_block = vec![self.last_offset];
        self.memory += new_block.allocated_size();
        self.blocks.push_back(new_block);
        let new_capacity = self.blocks.capacity();
        self.memory += (new_capacity - prev_capacity) * size_of::<Vec<O>>();
    }

    fn mark_as_having_value_in_block(&mut self) {
        if self.pending_block {
            self.number_of_blocks += 1;
            self.pending_block = false;
        }
    }

    pub(crate) fn reserve_blocks(&mut self, n: usize) {
        let prev_capacity = self.blocks.capacity();
        self.blocks.reserve(n);

        self.memory += (self.blocks.capacity() - prev_capacity) * size_of::<Vec<O>>();
    }

    /// Push length and return if the current block is now full
    pub fn push_length(&mut self, length: usize) -> bool {
        let mut block = &mut self.blocks[self.current_block_index];

        self.last_offset += O::usize_as(length);
        block.push_accounted(self.last_offset, &mut self.memory);
        self.len += 1;

        let finished_block = FIXED_BLOCK_SIZING && block.len() == self.block_size;

        self.mark_as_having_value_in_block();

        if finished_block {
            self.start_new_block();
            true
        } else {
            false
        }
    }

    /// Extends iterator of lengths within current block
    /// Returns if the current block has finished
    ///
    /// # Panics
    /// Panics if the iterator length exceeds the remaining size of the current block
    pub(super) fn extend_length_in_block(
        &mut self,
        iter: impl Iterator<Item = usize>,
        should_mark_empty: bool,
    ) -> bool {
        let mut block = &mut self.blocks[self.current_block_index];

        let prev_block_len = block.len();
        let prev_block_capacity = block.capacity();

        for item_len in iter {
            self.last_offset += O::usize_as(item_len);
            block.push(self.last_offset);
        }

        if FIXED_BLOCK_SIZING {
            assert!(
                block.len() <= self.block_size,
                "overflow from block new block length: {}, block size: {}",
                block.len(),
                self.block_size
            );
        }

        self.memory += (block.capacity() - prev_block_capacity) * size_of::<O>();

        let added_items = block.len() - prev_block_len;
        self.len += added_items;

        let finished_block = FIXED_BLOCK_SIZING && block.len() == self.block_size;

        if added_items > 0 || should_mark_empty {
            // mark as having value even when if no values were added since we intended to add value,
            // so take_block() should return empty block rather than None
            self.mark_as_having_value_in_block();
        }

        if finished_block {
            self.start_new_block();
            true
        } else {
            false
        }
    }

    /// Extend the length from the current offsets
    /// when it is guaranteed that len is less than the remaining block size
    pub(super) fn extends_length_from_offsets_in_current_block(
        &mut self,
        offset_buffer_slice: &[O],
    ) -> bool {
        assert_ne!(offset_buffer_slice.len(), 0);

        if FIXED_BLOCK_SIZING {
            // - 1 since we don't insert the first offset
            assert!(
                self.current_block_remaining_len() >= offset_buffer_slice.len() - 1,
                "the amount to add exceed the current block size"
            );
        }

        let block = &mut self.blocks[self.current_block_index];
        let prev_block_size = block.len();

        let prev_block_capacity = block.capacity();
        // Do fast large copy
        block.extend_from_slice(&offset_buffer_slice[1..]);
        self.memory += (block.capacity() - prev_block_capacity) * size_of::<O>();

        // Adjust the offset - can be easily SIMD.
        if offset_buffer_slice[0] > self.last_offset {
            // In case we cannot concat, shift the offsets
            // if we currently have [0, 2, 3] so the last_offset is 3
            // and offset buffer is: [6, 9, 10] so the final output should be
            // [0, 2, 3, 6, 7]
            // because we subtract offset_buffer[0] - self.last_offset

            let shift = offset_buffer_slice[0] - self.last_offset;

            block[prev_block_size..]
                .iter_mut()
                .for_each(|offset| *offset = *offset - shift);
        } else if offset_buffer_slice[0] < self.last_offset {
            let shift = self.last_offset - offset_buffer_slice[0];

            block[prev_block_size..]
                .iter_mut()
                .for_each(|offset| *offset = *offset + shift);
        }

        self.last_offset = block[block.len() - 1];

        let added_items = block.len() - prev_block_size;

        self.len += added_items;

        let finished_block = FIXED_BLOCK_SIZING && block.len() == self.block_size;

        // mark as having value even when if no values were added since we intended to add value,
        // so take_block() should return empty block rather than None
        self.mark_as_having_value_in_block();

        if finished_block {
            self.start_new_block();

            true
        } else {
            false
        }
    }

    /// Extend the length from the current offsets
    pub fn extends_length_from_offsets(&mut self, mut offset_buffer_slice: &[O]) {
        // If not fixed, then treat all offsets as single block
        if !FIXED_BLOCK_SIZING {
            self.extends_length_from_offsets_in_current_block(offset_buffer_slice);

            return;
        }

        let number_of_blocks_to_reserve = offset_buffer_slice
            .len()
            .saturating_sub(self.current_block_remaining_len())
            .div_ceil(self.block_size);
        self.reserve_blocks(number_of_blocks_to_reserve);

        let mut len = offset_buffer_slice.len() - 1;

        while len > 0 {
            let remaining_in_current_block = self.current_block_remaining_len();
            let to_add = remaining_in_current_block.min(len);

            let offsets_in_block = &offset_buffer_slice[..to_add + 1];
            offset_buffer_slice = &offset_buffer_slice[to_add..];
            len -= to_add;

            self.extends_length_from_offsets_in_current_block(offsets_in_block);
        }
    }

    /// Extend the length from the current offsets in the indexes
    /// when it is guaranteed that len is less than the remaining block size
    fn extends_length_from_offsets_indexes_in_current_block(
        &mut self,
        offset_buffer_slice: &[O],
        indexes: &[usize],
    ) -> usize {
        assert_ne!(offset_buffer_slice.len(), 0);
        assert_ne!(indexes.len(), 0);
        if FIXED_BLOCK_SIZING {
            assert!(
                self.current_block_remaining_len() >= indexes.len(),
                "the amount to add exceed the current block size"
            );
        }

        let block = &mut self.blocks[self.current_block_index];
        let prev_block_size = block.len();
        let prev_block_capacity = block.capacity();
        for &index_to_copy in indexes {
            let length = offset_buffer_slice[index_to_copy]
                - offset_buffer_slice[index_to_copy - 1];
            self.last_offset += length;

            block.push(self.last_offset);
        }

        self.memory += (block.capacity() - prev_block_capacity) * size_of::<O>();

        let added_items = block.len() - prev_block_size;

        self.len += added_items;

        let finished_block = FIXED_BLOCK_SIZING && block.len() == self.block_size;

        // mark as having value even when if no values were added since we intended to add value,
        // so take_block() should return empty block rather than None
        self.mark_as_having_value_in_block();

        if finished_block {
            self.start_new_block();
        }

        added_items
    }

    /// Extend the length from the current offsets
    pub fn extends_length_from_offsets_in_indexes(
        &mut self,
        offset_buffer_slice: &[O],
        mut indexes: &[usize],
    ) {
        // If not fixed, then treat all offsets as single block
        if !FIXED_BLOCK_SIZING {
            self.extends_length_from_offsets_indexes_in_current_block(
                offset_buffer_slice,
                indexes,
            );

            return;
        }
        let number_of_blocks_to_reserve = indexes
            .len()
            .saturating_sub(self.current_block_remaining_len())
            .div_ceil(self.block_size);
        self.reserve_blocks(number_of_blocks_to_reserve);

        while indexes.len() > 0 {
            let remaining_in_current_block = self.current_block_remaining_len();

            let to_add = remaining_in_current_block.min(indexes.len());
            let (to_copy, left) = indexes.split_at(to_add);
            indexes = left;

            self.extends_length_from_offsets_indexes_in_current_block(
                offset_buffer_slice,
                to_copy,
            );
        }
    }

    pub(super) fn current_block_remaining_len(&self) -> usize {
        assert!(
            FIXED_BLOCK_SIZING,
            "current block remaining length is only relevant for manual block size"
        );
        self.block_size - self.blocks[self.current_block_index].len()
    }

    pub(crate) fn push_empty_within_block(&mut self, n: usize) -> bool {
        self.len += n;
        let mut block = &mut self.blocks[self.current_block_index];
        let new_len = block.len() + n;

        let prev_capacity = block.capacity();

        if FIXED_BLOCK_SIZING {
            assert!(
                new_len <= self.block_size,
                "overflow from block new block length: {new_len}, block size: {}",
                self.block_size
            );
        }
        block.resize(new_len, self.last_offset);

        self.memory += (block.capacity() - prev_capacity) * size_of::<O>();

        let finished_block = FIXED_BLOCK_SIZING && block.len() == self.block_size;

        // mark as having value even when if no values were added since we intended to add value,
        // so take_block() should return empty block rather than None
        self.mark_as_having_value_in_block();

        if finished_block {
            self.start_new_block();
            true
        } else {
            false
        }
    }

    /// Push length 0
    pub fn push_empty_n(&mut self, mut n: usize) {
        // If not fixed, then treat all offsets as single block
        if !FIXED_BLOCK_SIZING {
            self.push_empty_within_block(n);
            return;
        }

        let number_of_blocks_to_reserve = n
            .saturating_sub(self.current_block_remaining_len())
            .div_ceil(self.block_size);
        self.reserve_blocks(number_of_blocks_to_reserve);

        while n > 0 {
            let remaining_in_current_block = self.current_block_remaining_len();

            let to_add = remaining_in_current_block.min(n);
            n -= to_add;

            self.push_empty_within_block(to_add);
        }
    }

    fn push_length_within_block(&mut self, len: usize, n: usize) {
        self.len += n;
        let mut block = &mut self.blocks[self.current_block_index];
        let new_len = block.len() + n;

        if FIXED_BLOCK_SIZING {
            assert!(
                new_len <= self.block_size,
                "overflow from block new block length: {new_len}, block size: {}",
                self.block_size
            );
        }
        let offset_to_add = O::usize_as(len);

        let prev_capacity = block.capacity();
        block.resize_with(new_len, || {
            self.last_offset += offset_to_add;

            self.last_offset
        });

        self.memory += (block.capacity() - prev_capacity) * size_of::<O>();

        let finished_block = FIXED_BLOCK_SIZING && block.len() == self.block_size;

        // mark as having value even when if no values were added since we intended to add value,
        // so take_block() should return empty block rather than None
        self.mark_as_having_value_in_block();

        if finished_block {
            self.start_new_block();
        }
    }

    /// Extend with length 0
    pub fn push_length_n(&mut self, len: usize, mut n: usize) {
        // Optimized
        if len == 0 {
            self.push_empty_n(n);
            return;
        }

        // If not fixed, then treat all offsets as single block
        if !FIXED_BLOCK_SIZING {
            self.push_length_within_block(len, n);
            return;
        }

        let number_of_blocks_to_reserve = n
            .saturating_sub(self.current_block_remaining_len())
            .div_ceil(self.block_size);
        self.reserve_blocks(number_of_blocks_to_reserve);

        while n > 0 {
            let remaining_in_current_block = self.current_block_remaining_len();

            let to_add = remaining_in_current_block.min(n);
            n -= to_add;

            self.push_length_within_block(len, to_add);
        }
    }

    pub fn take_block(&mut self) -> Option<Vec<O>> {
        if self.number_of_blocks == 0 {
            assert_eq!(
                self.blocks.len(),
                1,
                "even if number of blocks is 1 we must have at least one block for future insert"
            );
            assert_eq!(self.len, 0);
            assert_eq!(self.current_block_index, 0);
            assert_eq!(self.last_offset, O::zero());
            return None;
        }
        if self.pending_block {
            assert_eq!(self.number_of_blocks + 1, self.blocks.len());
        } else {
            assert_eq!(self.number_of_blocks, self.blocks.len());
        }

        let prev_blocks_capacity = self.blocks.capacity();

        // TODO - set last offset, add empty block if now finished,
        // but avoid adding it if in last block so we won't get into infinite loop that we always insert one and we never have empty blocks to indicate end
        let block = self
            .blocks
            .pop_front()
            .expect("we verified that we have at least 1 block");

        self.memory -= block.capacity() * size_of::<O>();
        self.memory -=
            (self.blocks.capacity() - prev_blocks_capacity) * size_of::<Vec<O>>();

        self.number_of_blocks -= 1;

        if self.blocks.is_empty() {
            self.current_block_index = 0;
            let prev_blocks_capacity = self.blocks.capacity();

            let block = vec![O::zero()];
            self.memory += block.capacity() * size_of::<O>();
            self.blocks.push_back(block);
            self.memory +=
                (self.blocks.capacity() - prev_blocks_capacity) * size_of::<Vec<O>>();
        } else {
            self.current_block_index -= 1;
        }

        self.last_offset = *self.blocks.back().unwrap().last().unwrap();

        let number_of_items = block.len() - 1;
        self.len -= number_of_items;

        Some(block)
    }

    pub fn take_block_finished(&mut self) -> Option<OffsetBuffer<O>> {
        let block = self.take_block()?;

        let inner = ScalarBuffer::from(block);

        // SAFETY: this is safe as we are the one that control the offsets
        let offsets = unsafe { OffsetBuffer::new_unchecked(inner) };

        Some(offsets)
    }
}

impl<const FIXED_BLOCK_SIZING: bool, O: OffsetSizeTrait> Extend<usize>
    for BlockedOffsetBufferBuilder<FIXED_BLOCK_SIZING, O>
{
    fn extend<T: IntoIterator<Item = usize>>(&mut self, iter: T) {
        if !FIXED_BLOCK_SIZING {
            self.extend_length_in_block(iter.into_iter(), true);

            return;
        }

        let mut iter = iter.into_iter();

        let mut is_first = true;
        loop {
            let remaining_in_current_block = self.current_block_remaining_len();
            let block_finished = self.extend_length_in_block(
                iter.by_ref().take(remaining_in_current_block),
                is_first,
            );

            is_first = false;
            if !block_finished {
                break;
            }
        }
    }
}

impl<O: OffsetSizeTrait> Index<usize> for BlockedOffsetBufferBuilder<true, O> {
    type Output = O;

    fn index(&self, index: usize) -> &Self::Output {
        self.index(BlocksIndex::from_index_in_fixed_block_size(
            index,
            self.block_size,
        ))
    }
}

impl<const FIXED_BLOCK_SIZING: bool, O: OffsetSizeTrait> Index<BlocksIndex>
    for BlockedOffsetBufferBuilder<FIXED_BLOCK_SIZING, O>
{
    type Output = O;

    fn index(&self, index: BlocksIndex) -> &Self::Output {
        &self.blocks[index.block_index()][index.index_in_block()]
    }
}

impl<const MANUAL_BLOCK_SIZE: bool, O: OffsetSizeTrait> IntoIterator
    for BlockedOffsetBufferBuilder<MANUAL_BLOCK_SIZE, O>
{
    type Item = OffsetBuffer<O>;
    type IntoIter = BlockedOffsetBufferIter<O>;

    fn into_iter(self) -> Self::IntoIter {
        if self.number_of_blocks == 0 {
            // Empty
            BlockedOffsetBufferIter {
                blocks: VecDeque::new(),
            }
        } else {
            BlockedOffsetBufferIter {
                blocks: self.blocks,
            }
        }
    }
}

pub struct BlockedOffsetBufferIter<O: OffsetSizeTrait> {
    blocks: VecDeque<Vec<O>>,
}

impl<O: OffsetSizeTrait> Iterator for BlockedOffsetBufferIter<O> {
    type Item = OffsetBuffer<O>;

    fn next(&mut self) -> Option<Self::Item> {
        let block = self.blocks.pop_front()?;
        let inner = ScalarBuffer::from(block);

        // SAFETY: this is safe as we are the one that control the offsets
        let offsets = unsafe { OffsetBuffer::new_unchecked(inner) };

        Some(offsets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::OffsetBuffer;

    #[test]
    fn newly_created_builder_should_return_empty_and_then_none() {
        let mut builder = BlockedOffsetBufferBuilder::<true, i32>::new(10);
        assert_eq!(builder.take_block(), Some(vec![0]));
        assert_eq!(builder.take_block(), None);
    }

    #[test]
    fn newly_created_builder_with_exactly_1_block_should_return_1_block_and_then_none() {
        let block_size = 6;
        let lengths_to_add = vec![3; block_size];
        run_on_all_ways_to_add::<i32>(block_size, &lengths_to_add, |builder, source| {
            let expected_offsets =
                OffsetBuffer::<i32>::from_lengths(lengths_to_add.clone());
            assert_eq!(
                builder.take_block().as_deref(),
                Some(expected_offsets.as_ref()),
                "failed when source is {source}"
            );
            assert_eq!(builder.take_block(), None, "failed when source is {source}");
        });
    }

    #[test]
    fn newly_created_builder_with_exactly_n_block_should_return_n_block_and_then_none() {
        let block_size = 6;
        let number_of_blocks = 3;
        let lengths_blocked = vec![vec![3; block_size]; number_of_blocks];
        let lengths_to_add = lengths_blocked
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        run_on_all_ways_to_add::<i32>(block_size, &lengths_to_add, |builder, source| {
            for length_blocked in &lengths_blocked {
                let expected_offsets =
                    OffsetBuffer::<i32>::from_lengths(length_blocked.clone());
                assert_eq!(
                    builder.take_block().as_deref(),
                    Some(expected_offsets.as_ref()),
                    "failed when source is {source}"
                );
            }

            assert_eq!(builder.take_block(), None, "failed when source is {source}");
        });
    }

    fn run_on_all_ways_to_add<O: OffsetSizeTrait>(
        block_size: usize,
        lengths_to_add: &[usize],
        on_added: impl Fn(&mut BlockedOffsetBufferBuilder<true, O>, &'static str),
    ) {
        {
            let mut builder = BlockedOffsetBufferBuilder::<true, O>::new(block_size);

            for _ in 0..2 {
                for &len in lengths_to_add {
                    builder.push_length(len);
                }
                on_added(&mut builder, "push_length");
            }
        }

        {
            let mut builder = BlockedOffsetBufferBuilder::<true, O>::new(block_size);

            for _ in 0..2 {
                builder.extend(lengths_to_add.iter().copied());
                on_added(&mut builder, "extend");
            }
        }

        if !lengths_to_add.is_empty() {
            let mut builder = BlockedOffsetBufferBuilder::<true, O>::new(block_size);

            for _ in 0..2 {
                let mut current_len: usize = lengths_to_add[0];
                let mut repeat: usize = 1;

                for &len in &lengths_to_add[1..] {
                    if len == current_len {
                        repeat += 1;
                    } else {
                        builder.push_length_n(current_len, repeat);
                        current_len = len;
                        repeat = 1;
                    }
                }

                builder.push_length_n(current_len, repeat);

                on_added(&mut builder, "push_length_n");
            }
        }

        {
            let mut builder = BlockedOffsetBufferBuilder::<true, O>::new(block_size);

            for _ in 0..2 {
                let offset_buffer_input =
                    OffsetBuffer::<O>::from_lengths(lengths_to_add.iter().copied());
                builder.extends_length_from_offsets(&offset_buffer_input);
                on_added(&mut builder, "extends_length_from_offsets");
            }
        }

        {
            let mut builder = BlockedOffsetBufferBuilder::<true, O>::new(block_size);

            let mut lengths_to_add_modified = lengths_to_add.to_vec();
            let mut indices = (0..lengths_to_add.len() + 2).collect::<Vec<_>>();
            lengths_to_add_modified.insert(lengths_to_add.len() - 1, 10);
            indices.remove(lengths_to_add.len());

            lengths_to_add_modified.insert(0, 40);
            indices.remove(0);

            let cleared_length = indices
                .iter()
                .map(|index| lengths_to_add_modified[*index])
                .collect::<Vec<_>>();
            let offset_buffer_input =
                OffsetBuffer::<O>::from_lengths(lengths_to_add_modified);

            for _ in 0..2 {
                builder.extends_length_from_offsets_in_indexes(
                    &offset_buffer_input,
                    &indices,
                );
                on_added(&mut builder, "extends_length_from_offsets_in_indexes");
            }
        }
    }
}
