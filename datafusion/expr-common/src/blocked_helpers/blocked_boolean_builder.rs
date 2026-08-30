use arrow::array::{new_empty_array, BooleanBufferBuilder, AsArray};
use arrow::buffer::{BooleanBuffer};
use crate::groups_accumulator::BlocksIndex;
use std::collections::VecDeque;
use std::ops::Index;
use arrow::datatypes::DataType;
use itertools::Itertools;
use datafusion_common::utils::proxy::VecDequeAllocExt;

#[derive(Debug)]
pub struct BlockedBooleanBuilder<const FIXED_BLOCK_SIZING: bool> {
    /// Using `VecDeque` so we can remove the first block and reclaim memory
    blocks: VecDeque<BooleanBufferBuilder>,

    /// The size of each block
    block_size: usize,

    /// The index of the current block
    current_block_index: usize,

    len: usize,

    finished_blocks_allocated_size: usize,
}

impl<const FIXED_BLOCK_SIZING: bool> BlockedBooleanBuilder<FIXED_BLOCK_SIZING> {
    pub fn new(block_size: usize) -> Self {
        if FIXED_BLOCK_SIZING {
            assert_ne!(block_size, 0, "block size must be greater than 0");
        }

        let blocks = VecDeque::from(vec![BooleanBufferBuilder::new(block_size)]);

        BlockedBooleanBuilder {
            blocks,
            block_size,
            current_block_index: 0,
            len: 0,
            finished_blocks_allocated_size: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn allocated_size(&self) -> usize {
        self.finished_blocks_allocated_size + self.blocks.allocated_size() + self.blocks.back().map_or(0, |b| allocated_size_for_builder(b))
    }

    pub fn block_size(&self) -> usize {
        assert!(
            FIXED_BLOCK_SIZING,
            "block size is only available for manual block"
        );
        self.block_size
    }

    pub fn start_new_block(&mut self) {
        self.current_block_index += 1;
        self.finished_blocks_allocated_size += self.blocks.back().map_or(0, |b| allocated_size_for_builder(b));
        let new_block = BooleanBufferBuilder::new(self.block_size);
        self.blocks.push_back(new_block);
    }

    pub(crate) fn reserve_blocks(&mut self, n: usize) {
        self.blocks.reserve(n);
    }

    fn current_block_remaining_len(&self) -> usize {
        assert!(
            FIXED_BLOCK_SIZING,
            "remaining block only available for manual block"
        );
        self.block_size - self.blocks[self.current_block_index].len()
    }

    fn push_n_within_block(&mut self, n: usize, is_set: bool) {
        self.len += n;
        let mut block = &mut self.blocks[self.current_block_index];

        block.append_n(n, is_set);

        assert!(
            block.len() <= self.block_size,
            "overflow from block new block length: {}, block size: {}",
            block.len(),
            self.block_size
        );

        if FIXED_BLOCK_SIZING && block.len() == self.block_size {
            self.start_new_block();
        }
    }

    pub fn append_n(&mut self, mut n: usize, is_set: bool) {
        if !FIXED_BLOCK_SIZING {
            self.push_n_within_block(n, is_set);
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

            self.push_n_within_block(to_add, is_set);
        }
    }

    pub fn append(&mut self, is_set: bool) {
        let block = &mut self.blocks[self.current_block_index];

        block.append(is_set);
        self.len += 1;

        if block.len() == self.block_size {
            self.start_new_block();
        }
    }

    pub fn get_bit(&self, blocked_index: BlocksIndex) -> bool {
        self.blocks[blocked_index.block_index()].get_bit(blocked_index.index_in_block())
    }

    pub fn set_bit(&mut self, blocked_index: BlocksIndex, is_set: bool) {
        self.blocks[blocked_index.block_index()]
            .set_bit(blocked_index.index_in_block(), is_set)
    }

    /// Extends iterator of validity within current block
    /// Returns how many items were added
    ///
    /// # Panics
    /// Panics if the iterator length exceeds the remaining size of the current block
    pub(super) fn extend_validity_in_block(
        &mut self,
        iter: impl Iterator<Item = bool>,
    ) -> usize {
        let block = &mut self.blocks[self.current_block_index];

        let prev_block_len = block.len();

        for is_valid in iter {
            block.append(is_valid);
        }

        assert!(
            block.len() <= self.block_size,
            "overflow from block new block length: {}, block size: {}",
            block.len(),
            self.block_size
        );

        let added_items = block.len() - prev_block_len;
        self.len += added_items;

        if FIXED_BLOCK_SIZING && block.len() == self.block_size {
            self.start_new_block();
        }

        added_items
    }

    pub fn take_block(&mut self) -> Option<BooleanBuffer> {
        let mut block = self.blocks.pop_front()?;
        let number_of_items = block.len() - 1;
        self.len -= number_of_items;

        // Never have empty blocks since we won't be able to add more items
        if self.blocks.is_empty() {
            let empty_block = BooleanBufferBuilder::new(self.block_size);
            self.blocks.push_back(empty_block);
        } else {
            // Only if not the current block reduce the memory since current block is calculated separately
            self.finished_blocks_allocated_size -= allocated_size_for_builder(&block);
        }

        Some(block.build())
    }

    fn new_empty_buffer() -> BooleanBuffer {
        let empty_array = new_empty_array(&DataType::Boolean);

        empty_array.as_boolean().clone().into_parts().0
    }

    /// Moves the bits in `[offset, offset + len)` down to the start of the buffer
    /// and shrinks the builder to `len`
    fn shift_down_in_place(block: &mut BooleanBufferBuilder, offset: usize, len: usize) {
        if offset == 0 {
            block.truncate(len);
            return;
        }

        let byte_offset = offset / 8;
        let bit_offset = offset % 8;
        let dst_bytes = len.div_ceil(8);

        {
            let bytes = block.as_slice_mut();
            let src_bytes = bytes.len();

            if bit_offset == 0 {
                bytes.copy_within(byte_offset..byte_offset + dst_bytes, 0);
            } else {
                // Destination byte is always at or below the source byte
                // so a forward pass never reads a byte that was already overwritten
                for dst in 0..dst_bytes {
                    let src = dst + byte_offset;

                    let low = bytes[src] >> bit_offset;
                    let high = if src + 1 < src_bytes {
                        bytes[src + 1] << (8 - bit_offset)
                    } else {
                        0
                    };

                    bytes[dst] = low | high;
                }
            }

            // Clear the stale bits in the last byte so later appends see zeroed padding
            let trailing = len % 8;
            if trailing != 0 {
                bytes[dst_bytes - 1] &= (1u8 << trailing) - 1;
            }
        }

        block.truncate(len);
    }
}

impl BlockedBooleanBuilder<true> {

    ///
    pub fn take_n(&mut self, n: usize) -> BooleanBuffer {
        assert!(n <= self.len, "n ({n}) must be <= len ({}) than", self.len);
        assert!(n <= self.block_size, "n ({n}) must be lower than block size ({}), instead use `take_block` and take_n with the remainder", self.block_size);

        if n == self.len || n == self.block_size {
            return self.take_block().expect("must have block");
        }

        assert_ne!(n, self.len, "n must ne smaller than the first block which is smaller that len");

        // Not moving anything
        if n == 0 {
            return Self::new_empty_buffer();
        }

        // Every block other than the last one is exactly `block_size` long and `n` is smaller
        // than that, so the emitted values are always fully contained in the first block
        let mut taken = BooleanBufferBuilder::new(n);
        taken.append_packed_range(0..n, self.blocks[0].as_slice());

        // Reused for swapping blocks out of the deque, `new(0)` holds no buffer
        let mut placeholder = BooleanBufferBuilder::new(0);

        // Shift every block down by `n` and refill it from the front of the next one
        // so that all blocks but the last stay exactly `block_size` long
        for index in 0..self.blocks.len() {
            let block_len = self.blocks[index].len();

            if block_len <= n {
                // Only reachable for the last block, everything it held was already
                // pulled into the previous block
                self.blocks[index].truncate(0);
            } else {
                Self::shift_down_in_place(&mut self.blocks[index], n, block_len - n);
            }

            let next_index = index + 1;

            if next_index < self.blocks.len() {
                // Move the next block aside so the current one can be borrowed mutably
                // it has not been shifted yet, so its first `n` values are the ones we want
                std::mem::swap(&mut self.blocks[next_index], &mut placeholder);

                let to_copy = n.min(placeholder.len());

                self.blocks[index].append_packed_range(0..to_copy, placeholder.as_slice());

                std::mem::swap(&mut self.blocks[next_index], &mut placeholder);
            }
        }

        self.len -= n;

        // The last block is allowed to be empty, which is the state `start_new_block` leaves
        // behind when a block fills up exactly
        let new_blocks_count = self.len / self.block_size + 1;

        while self.blocks.len() > new_blocks_count {
            // The back block is measured separately so dropping it needs no adjustment
            self.blocks.pop_back();

            // Whatever is now at the back stopped being a finished block
            self.finished_blocks_allocated_size -= self
              .blocks
              .back()
              .map_or(0, |b| allocated_size_for_builder(b));
        }

        self.current_block_index = self.blocks.len() - 1;

        taken.build()
    }
}

impl BlockedBooleanBuilder<false> {

    /// Take the first `n` values
    ///
    /// `block_size_iterator` is iterator over the number of items in each block **after** emitting `n`
    ///
    /// The adjusted iterator must meet this requirement:
    /// ```
    /// assert_eq!(n + adjusted_block_size_iter.sum(), self.len);
    /// ```
    ///
    /// TODO - shrink to fit
    ///
    ///
    pub fn take_n(&mut self, n: usize, mut adjusted_block_size_iter: impl Iterator<Item=usize> + Clone) -> BooleanBuffer {
        assert!(n <= self.len, "n ({n}) must be <= len ({}) than", self.len);
        assert!(n <= self.blocks[0].len(), "n ({n}) must be lower than the first block ({}), instead use `take_block` and take_n with the remainder", self.blocks[0].len());

        let prev_len = self.len;

        if n == self.blocks[0].len() {
            adjusted_block_size_iter.zip_eq(self.blocks.iter().skip(1)).for_each(|(len, current_block)| assert_eq!(len, current_block.len()));

            return self.take_block().expect("must have block");
        }

        assert_ne!(n, self.len, "n must ne smaller than the first block which is smaller that len");

        // Not moving anything
        if n == 0 {
            adjusted_block_size_iter.zip_eq(self.blocks.iter()).for_each(|(len, current_block)| assert_eq!(len, current_block.len(), "when n is 0 and not equal the first block size, we should keep the length as is"));

            return Self::new_empty_buffer();
        }

        // The emitted values are always fully contained in the first block
        let mut taken = BooleanBufferBuilder::new(n);
        taken.append_packed_range(0..n, self.blocks[0].as_slice());

        // Read cursor into the old layout, starts right after the emitted values
        let mut src_index = 0;
        let mut src_offset = n;

        // Write cursor into the new layout
        let mut dst_index = 0;

        let mut sum = 0;

        // Reused for swapping blocks out of the deque, `new(0)` holds no buffer
        let mut placeholder = BooleanBufferBuilder::new(0);

        while let Some(new_block_size) = adjusted_block_size_iter.next() {
            sum += new_block_size;

            // Skip over source blocks that were fully read
            while src_index < self.blocks.len() && src_offset >= self.blocks[src_index].len() {
                src_index += 1;
                src_offset = 0;
            }

            assert!(
                src_index < self.blocks.len(),
                "sum of adjusted block sizes + n ({n}) is larger than the length ({prev_len})"
            );

            // Invariant, the destination never runs ahead of the read cursor
            // so writing into `dst_index` can never clobber values that were not read yet
            debug_assert!(dst_index <= src_index);

            if dst_index == src_index {
                let remaining_in_src = self.blocks[src_index].len() - src_offset;

                if new_block_size < remaining_in_src {
                    // The old block is being split, its tail is still needed by later
                    // destinations so it cannot be shifted down in place
                    // Give the split off part its own slot and push the old block one to the right
                    let mut split = BooleanBufferBuilder::new(new_block_size);
                    split.append_packed_range(
                        src_offset..src_offset + new_block_size,
                        self.blocks[src_index].as_slice(),
                    );

                    self.blocks.insert(dst_index, split);

                    src_index += 1;
                    src_offset += new_block_size;
                    dst_index += 1;
                    continue;
                }

                // The whole tail of this block belongs to the destination, shift it down
                // over the values that were consumed and reuse the same allocation
                Self::shift_down_in_place(&mut self.blocks[dst_index], src_offset, remaining_in_src);

                src_index += 1;
                src_offset = 0;

                if remaining_in_src == new_block_size {
                    dst_index += 1;
                    continue;
                }
            } else {
                // This slot held a block that is already fully read, reuse it as an empty destination
                self.blocks[dst_index].truncate(0);
            }

            let mut remaining = new_block_size - self.blocks[dst_index].len();

            while remaining > 0 {
                while src_index < self.blocks.len() && src_offset >= self.blocks[src_index].len() {
                    src_index += 1;
                    src_offset = 0;
                }

                assert!(
                    src_index < self.blocks.len(),
                    "sum of adjusted block sizes + n ({n}) is larger than the length ({prev_len}), missing {remaining} items"
                );

                // Move the source block aside so the destination can be borrowed mutably
                std::mem::swap(&mut self.blocks[src_index], &mut placeholder);

                let to_copy = (placeholder.len() - src_offset).min(remaining);

                self.blocks[dst_index].append_packed_range(
                    src_offset..src_offset + to_copy,
                    placeholder.as_slice(),
                );

                std::mem::swap(&mut self.blocks[src_index], &mut placeholder);

                src_offset += to_copy;
                remaining -= to_copy;
            }

            dst_index += 1;
        }

        assert_eq!(prev_len, sum + n, "sum of adjusted block sizes ({sum}) + n ({n}) must equal the length {prev_len}");

        // Drop the old blocks that the new layout did not need
        self.blocks.truncate(dst_index);

        // Never have empty blocks since we won't be able to add more items
        if self.blocks.is_empty() {
            self.blocks.push_back(BooleanBufferBuilder::new(self.block_size));
        }

        // The back block is the one still being written to and is measured separately
        self.finished_blocks_allocated_size = self
          .blocks
          .iter()
          .take(self.blocks.len() - 1)
          .map(|b| allocated_size_for_builder(b))
          .sum();

        self.current_block_index = self.blocks.len() - 1;
        self.len = sum;

        taken.build()
    }
}


fn allocated_size_for_builder(builder: &BooleanBufferBuilder) -> usize {
    // capacity returns in bits
    // once we upgrade arrow to have the allocated_size function, we can remove this function
    builder.capacity() / 8
}

// Only when we control the blocking since otherwise each block is not the same size
impl Index<usize> for BlockedBooleanBuilder<true> {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        self.index(BlocksIndex::from_index_in_fixed_block_size(
            index,
            self.block_size,
        ))
    }
}

impl<const FIXED_BLOCK_SIZING: bool> Index<BlocksIndex>
    for BlockedBooleanBuilder<FIXED_BLOCK_SIZING>
{
    type Output = bool;

    fn index(&self, blocked_index: BlocksIndex) -> &Self::Output {
        if self.blocks[blocked_index.block_index()]
            .get_bit(blocked_index.index_in_block())
        {
            &true
        } else {
            &false
        }
    }
}
