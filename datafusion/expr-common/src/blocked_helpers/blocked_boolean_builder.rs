use crate::blocked_helpers::take_n_helpers::{BlockBuilder, take_n_from_blocks};
use crate::groups_accumulator::BlocksIndex;
use arrow::array::{AsArray, BooleanBufferBuilder, new_empty_array};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::DataType;
use datafusion_common::utils::proxy::VecDequeAllocExt;
use itertools::Itertools;
use std::collections::VecDeque;
use std::ops::{Index, Range};

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
        self.finished_blocks_allocated_size
            + self.blocks.allocated_size()
            + self
                .blocks
                .back()
                .map_or(0, |b| allocated_size_for_builder(b))
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
        self.finished_blocks_allocated_size += self
            .blocks
            .back()
            .map_or(0, |b| allocated_size_for_builder(b));
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

    pub fn take_all(&mut self) -> Vec<BooleanBuffer> {
        let blocks = std::mem::take(&mut self.blocks);
        assert_ne!(blocks.len(), 0);

        assert_eq!(self.current_block_index, blocks.len() - 1);

        // TODO - should preallocate? can be expensive for large schema
        self.blocks
            .push_back(BooleanBufferBuilder::new(self.block_size));
        self.len = 0;
        self.current_block_index = 0;
        self.finished_blocks_allocated_size = 0;

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
    ) -> BooleanBuffer {
        assert_eq!(FIXED_BLOCK_SIZING, adjusted_block_size_iter.is_none());
        if let Some(adjusted_block_size_iter) = adjusted_block_size_iter {
            let (taken, layout) = take_n_from_blocks(
                &mut self.blocks,
                self.len,
                n,
                None,
                adjusted_block_size_iter,
            );

            self.len = layout.len;
            self.current_block_index = layout.current_block_index;
            self.finished_blocks_allocated_size = layout.finished_blocks_allocated_size;

            taken
        } else {
            assert!(n <= self.len, "n ({n}) must be <= len ({}) than", self.len);
            assert!(
                n <= self.block_size,
                "n ({n}) must be lower than block size ({}), instead use `take_block` and take_n with the remainder",
                self.block_size
            );

            if n == self.len || n == self.block_size {
                return self.take_block().expect("must have block");
            }

            assert_ne!(
                n, self.len,
                "n must ne smaller than the first block which is smaller that len"
            );

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

                    self.blocks[index]
                        .append_packed_range(0..to_copy, placeholder.as_slice());

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

impl BlockBuilder for BooleanBufferBuilder {
    type Output = BooleanBuffer;

    fn with_capacity(capacity: usize) -> Self {
        BooleanBufferBuilder::new(capacity)
    }

    fn len(&self) -> usize {
        BooleanBufferBuilder::len(self)
    }

    fn truncate(&mut self, len: usize) {
        BooleanBufferBuilder::truncate(self, len)
    }

    fn append_range(&mut self, src: &Self, range: Range<usize>) {
        self.append_packed_range(range, src.as_slice())
    }

    fn shift_down(&mut self, offset: usize, len: usize) {
        if offset == 0 {
            BooleanBufferBuilder::truncate(self, len);
            return;
        }

        let byte_offset = offset / 8;
        let bit_offset = offset % 8;
        let dst_bytes = len.div_ceil(8);

        {
            let bytes = self.as_slice_mut();
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

        BooleanBufferBuilder::truncate(self, len);
    }

    fn allocated_size(&self) -> usize {
        allocated_size_for_builder(self)
    }

    fn finish(mut self) -> BooleanBuffer {
        self.build()
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
