use crate::groups_accumulator::BlocksIndex;
use arrow::array::OffsetSizeTrait;
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use datafusion_common::utils::proxy::{VecAllocExt, VecDequeAllocExt};
use itertools::Itertools;
use std::collections::VecDeque;
use std::collections::vec_deque::Iter;
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

    finished_memory: usize,
}

impl<const FIXED_BLOCK_SIZING: bool, O: OffsetSizeTrait>
    BlockedOffsetBufferBuilder<FIXED_BLOCK_SIZING, O>
{
    pub fn new(mut block_size: usize) -> Self {
        if FIXED_BLOCK_SIZING {
            assert_ne!(block_size, 0, "block size must be greater than 0");
        }

        // Add 1 to the block size to account for the initial offset
        block_size += 1;

        let last_offset = O::zero();
        let blocks = VecDeque::from(vec![vec![last_offset]]);
        BlockedOffsetBufferBuilder {
            blocks,
            block_size,
            len: 0,
            current_block_index: 0,
            last_offset,
            number_of_blocks: 1,
            pending_block: false,
            finished_memory: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn allocated_size(&self) -> usize {
        self.finished_memory
            + self.blocks.allocated_size()
            + self.blocks.back().map_or_else(0, |b| b.allocated_size())
    }

    /// Get the number of elements in the current block (not the number of offsets since the first offset is always 0)
    pub fn current_block_len(&self) -> usize {
        self.blocks[self.current_block_index].len() - 1
    }

    pub fn current_block_index(&self) -> usize {
        self.blocks.len() - 1
    }

    pub fn last_offset(&self) -> O {
        self.last_offset
    }

    pub fn start_new_block(&mut self) {
        // Don't add to number of blocks since we might not insert into it
        self.current_block_index += 1;
        self.last_offset = O::zero();
        let new_block = vec![self.last_offset];
        self.finished_memory += self
            .blocks
            .back()
            .as_ref()
            .map_or_else(0, |b| b.allocated_size());
        self.blocks.push_back(new_block);
    }

    fn mark_as_having_value_in_block(&mut self) {
        if self.pending_block {
            self.number_of_blocks += 1;
            self.pending_block = false;
        }
    }

    pub(crate) fn reserve_blocks(&mut self, n: usize) {
        self.blocks.reserve(n);
    }

    pub fn push_next_offset_in_block(&mut self, next_offset_in_block: O) -> bool {
        let mut block = &mut self.blocks[self.current_block_index];

        assert!(
            next_offset_in_block >= self.last_offset,
            "offsets must be monotonically increasing"
        );
        self.last_offset = next_offset_in_block;
        block.push(self.last_offset);
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

    /// Push length and return if the current block is now full
    pub fn push_length(&mut self, length: usize) -> bool {
        let mut block = &mut self.blocks[self.current_block_index];

        self.last_offset += O::usize_as(length);
        block.push(self.last_offset);
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

        // Do fast large copy
        block.extend_from_slice(&offset_buffer_slice[1..]);

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
        for &index_to_copy in indexes {
            let length = offset_buffer_slice[index_to_copy]
                - offset_buffer_slice[index_to_copy - 1];
            self.last_offset += length;

            block.push(self.last_offset);
        }

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

        if FIXED_BLOCK_SIZING {
            assert!(
                new_len <= self.block_size,
                "overflow from block new block length: {new_len}, block size: {}",
                self.block_size
            );
        }
        block.resize(new_len, self.last_offset);

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

        block.resize_with(new_len, || {
            self.last_offset += offset_to_add;

            self.last_offset
        });

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

        // TODO - set last offset, add empty block if now finished,
        // but avoid adding it if in last block so we won't get into infinite loop that we always insert one and we never have empty blocks to indicate end
        let block = self
            .blocks
            .pop_front()
            .expect("we verified that we have at least 1 block");

        self.number_of_blocks -= 1;

        if self.blocks.is_empty() {
            self.current_block_index = 0;
            let block = vec![O::zero()];
            self.blocks.push_back(block);
        } else {
            self.current_block_index -= 1;

            // Only if not the last block since the last block is calculated in allocated_size
            self.finished_memory -= block.allocated_size();
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

    pub fn take_all(&mut self) -> Vec<Vec<O>> {
        let blocks = std::mem::take(&mut self.blocks);
        assert_ne!(blocks.len(), 0);

        assert_eq!(self.current_block_index, blocks.len() - 1);

        // TODO - should preallocate? can be expensive for large schema
        self.blocks.push_back(vec![O::zero()]);
        self.len = 0;
        self.current_block_index = 0;
        self.finished_memory = 0;
        self.number_of_blocks = 0;
        self.pending_block = true;
        self.last_offset = O::zero();

        blocks.into_iter().map(|b| b.build()).collect()
    }

    /// Take the first `n` values
    ///
    /// `block_size_iterator` is iterator over the number of items in each block **after** emitting `n`
    ///
    /// this is `None` when `FIXED_BLOCK_SIZING` is true
    ///
    /// The adjusted iterator must meet this requirement:
    /// ```
    /// assert_eq!(n + adjusted_block_size_iter.sum(), self.len);
    /// ```
    ///
    /// TODO - shrink to fit
    ///
    ///
    pub fn take_n(
        &mut self,
        n: usize,
        adjusted_block_size_iter: Option<impl Iterator<Item = usize> + Clone>,
    ) -> Vec<O> {
        assert_eq!(FIXED_BLOCK_SIZING, adjusted_block_size_iter.is_none());
        if let Some(adjusted_block_size_iter) = adjusted_block_size_iter {
            self.take_n_dynamic(n, adjusted_block_size_iter)
        } else {
            self.take_n_fixed(n)
        }
    }

    fn take_n_dynamic(
        &mut self,
        n: usize,
        mut adjusted_block_size_iter: impl Iterator<Item = usize> + Clone,
    ) -> Vec<O> {
        let first_block_items = self.blocks[0].len() - 1;

        assert!(n <= self.len, "n ({n}) must be <= len ({}) than", self.len);
        assert!(
            n <= first_block_items,
            "n ({n}) must be lower than the first block ({first_block_items}), instead use `take_block` and take_n with the remainder"
        );

        let prev_len = self.len;

        if n == first_block_items {
            adjusted_block_size_iter
              .zip_eq(self.blocks.iter().skip(1))
              .for_each(|(len, current_block)| assert_eq!(len, current_block.len() - 1));

            return self.take_block().expect("must have block");
        }

        assert_ne!(
            n, self.len,
            "n must ne smaller than the first block which is smaller that len"
        );

        // Not moving anything
        if n == 0 {
            adjusted_block_size_iter
              .zip_eq(self.blocks.iter())
              .for_each(|(len, current_block)| {
                  assert_eq!(
                      len,
                      current_block.len() - 1,
                      "when n is 0 and not equal the first block size, we should keep the length as is"
                  )
              });

            return vec![O::zero()];
        }

        // The emitted items are always fully contained in the first block, its initial
        // offset is already zero so the emitted offsets need no rebasing
        let mut taken = Vec::with_capacity(n + 1);
        taken.extend_from_slice(&self.blocks[0][..=n]);

        // Read cursor into the old layout, starts right after the emitted items
        let mut src_index = 0;
        let mut src_offset = n;

        // Write cursor into the new layout
        let mut dst_index = 0;

        let mut sum = 0;

        // Reused for swapping blocks out of the deque, an empty vec holds no buffer
        let mut placeholder = Vec::new();

        while let Some(new_block_size) = adjusted_block_size_iter.next() {
            sum += new_block_size;

            // Skip over source blocks that were fully read
            while src_index < self.blocks.len()
              && src_offset >= self.blocks[src_index].len() - 1
            {
                src_index += 1;
                src_offset = 0;
            }

            assert!(
                src_index < self.blocks.len(),
                "sum of adjusted block sizes + n ({n}) is larger than the length ({prev_len})"
            );

            // Invariant, the destination never runs ahead of the read cursor
            // so writing into `dst_index` can never clobber items that were not read yet
            debug_assert!(dst_index <= src_index);

            if dst_index == src_index {
                let remaining_in_src = self.blocks[src_index].len() - 1 - src_offset;

                if new_block_size < remaining_in_src {
                    // The old block is being split, its tail is still needed by later
                    // destinations so it cannot be shifted down in place
                    // Give the split off part its own slot and push the old block one to the right
                    let mut split = Vec::with_capacity(new_block_size + 1);
                    let base = self.blocks[src_index][src_offset];

                    split.extend(
                        self.blocks[src_index][src_offset..=src_offset + new_block_size]
                          .iter()
                          .map(|offset| *offset - base),
                    );

                    self.blocks.insert(dst_index, split);

                    src_index += 1;
                    src_offset += new_block_size;
                    dst_index += 1;
                    continue;
                }

                // The whole tail of this block belongs to the destination, shift it down
                // over the items that were consumed and reuse the same allocation
                Self::shift_offsets_down_in_place(&mut self.blocks[dst_index], src_offset);

                src_index += 1;
                src_offset = 0;

                if remaining_in_src == new_block_size {
                    dst_index += 1;
                    continue;
                }
            } else {
                // This slot held a block that is already fully read, reuse it as an empty destination
                let block = &mut self.blocks[dst_index];
                block.truncate(1);
                block[0] = O::zero();
            }

            let mut remaining = new_block_size - (self.blocks[dst_index].len() - 1);

            self.blocks[dst_index].reserve(remaining);

            while remaining > 0 {
                while src_index < self.blocks.len()
                  && src_offset >= self.blocks[src_index].len() - 1
                {
                    src_index += 1;
                    src_offset = 0;
                }

                assert!(
                    src_index < self.blocks.len(),
                    "sum of adjusted block sizes + n ({n}) is larger than the length ({prev_len}), missing {remaining} items"
                );

                // Move the source block aside so the destination can be borrowed mutably
                std::mem::swap(&mut self.blocks[src_index], &mut placeholder);

                let to_copy = (placeholder.len() - 1 - src_offset).min(remaining);

                // The source block is relative to its own start, so the moved items are
                // rebased onto the last offset of the destination
                let base = placeholder[src_offset];
                let block = &mut self.blocks[dst_index];
                let last_offset = block[block.len() - 1];

                block.extend(
                    placeholder[src_offset + 1..=src_offset + to_copy]
                      .iter()
                      .map(|offset| last_offset + (*offset - base)),
                );

                std::mem::swap(&mut self.blocks[src_index], &mut placeholder);

                src_offset += to_copy;
                remaining -= to_copy;
            }

            dst_index += 1;
        }

        assert_eq!(
            prev_len,
            sum + n,
            "sum of adjusted block sizes ({sum}) + n ({n}) must equal the length {prev_len}"
        );

        // Drop the old blocks that the new layout did not need
        self.blocks.truncate(dst_index);

        // Never have zero blocks since we won't be able to add more items
        if self.blocks.is_empty() {
            self.blocks.push_back(vec![O::zero()]);
        }

        // The back block is the one still being written to and is measured separately
        self.finished_memory = self
          .blocks
          .iter()
          .take(self.blocks.len() - 1)
          .map(|b| b.allocated_size())
          .sum();

        self.current_block_index = self.blocks.len() - 1;
        self.number_of_blocks = self.blocks.len();
        self.pending_block = false;
        self.len = sum;
        self.last_offset = *self.blocks[self.current_block_index]
          .last()
          .expect("every block holds at least the initial offset");

        taken
    }

    /// Drops the first `items` items of the block and rebases the remaining offsets to zero
    /// reusing the same allocation
    fn shift_offsets_down_in_place(block: &mut Vec<O>, items: usize) {
        if items == 0 {
            return;
        }

        let block_len = block.len();

        block.copy_within(items.., 0);
        block.truncate(block_len - items);

        let base = block[0];
        if base > O::zero() {
            block.iter_mut().for_each(|offset| *offset = *offset - base);
        }
    }

    fn take_n_fixed(&mut self, n: usize) -> Vec<O> {
        assert!(n <= self.len, "n ({n}) must be <= len ({}) than", self.len);
        assert!(
            n <= self.block_size - 1,
            "n ({n}) must be lower than block size ({}), instead use `take_block` and take_n with the remainder",
            self.block_size
        );

        if n == self.len || n == self.block_size - 1 {
            return self.take_block().expect("must have block");
        }

        assert_ne!(
            n, self.len,
            "n must ne smaller than the first block which is smaller that len"
        );

        // Every block other than the last one holds exactly `block_size - 1` items and `n`
        // is smaller than that, so the emitted items are always fully contained in the first block
        let mut taken = Vec::with_capacity(n + 1);
        taken.extend_from_slice(&self.blocks[0][..=n]);

        // Reused for swapping blocks out of the deque, an empty vec holds no buffer
        let mut placeholder = Vec::new();

        // Shift every block down by `n` items and refill it from the front of the next one
        // so that all blocks but the last keep holding exactly `block_size - 1` items
        for index in 0..self.blocks.len() {
            {
                let block = &mut self.blocks[index];
                let block_len = block.len();
                let items_in_block = block_len - 1;

                if items_in_block <= n {
                    // Only reachable for the last block, everything it held was already
                    // pulled into the previous block
                    block.truncate(1);
                    block[0] = O::zero();
                } else {
                    // Drop the first `n` items, the offset that starts the remaining items
                    // becomes the new initial offset of the block
                    block.copy_within(n.., 0);
                    block.truncate(block_len - n);

                    // Blocks are self relative so the offsets have to be rebased to zero
                    let base = block[0];
                    if base > O::zero() {
                        block.iter_mut().for_each(|offset| *offset = *offset - base);
                    }
                }
            }

            let next_index = index + 1;

            if next_index < self.blocks.len() {
                // Move the next block aside so the current one can be borrowed mutably
                // it has not been shifted yet, so its first `n` items are the ones we want
                std::mem::swap(&mut self.blocks[next_index], &mut placeholder);

                let items_in_next = placeholder.len() - 1;
                let to_copy = n.min(items_in_next);

                let base = placeholder[0];
                let block = &mut self.blocks[index];
                let mut last_offset = block[block.len() - 1];

                for moved in &placeholder[1..=to_copy] {
                    last_offset += *moved - base;
                    block.push(last_offset);
                }

                std::mem::swap(&mut self.blocks[next_index], &mut placeholder);
            }
        }

        self.len -= n;

        // The last block is allowed to be empty, which is the state `start_new_block` leaves
        // behind when a block fills up exactly
        let items_per_block = self.block_size - 1;
        let new_blocks_count = self.len / items_per_block + 1;

        while self.blocks.len() > new_blocks_count {
            // The back block is measured separately so dropping it needs no adjustment
            self.blocks.pop_back();

            // Whatever is now at the back stopped being a finished block
            self.finished_memory -= self.blocks.back().map_or(0, |b| b.allocated_size());

            self.number_of_blocks = self.number_of_blocks.saturating_sub(1);
        }

        self.current_block_index = self.blocks.len() - 1;
        self.last_offset = *self.blocks[self.current_block_index]
            .last()
            .expect("every block holds at least the initial offset");

        taken
    }

    pub fn blocks_iter(&self) -> Iter<Vec<O>> {
        self.blocks.iter()
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
