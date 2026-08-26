use arrow::array::BooleanBufferBuilder;
use arrow::buffer::{BooleanBuffer, NullBuffer};
use crate::groups_accumulator::BlocksIndex;
use std::collections::VecDeque;
use std::ops::Index;
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
}

fn allocated_size_for_builder(builder: &BooleanBufferBuilder) -> usize {
    // capacity returns in bits
    // once we upgrade arrow to have the allocated_size function, we can remove this function
    builder.capacity() / 8
}

impl<const MANUAL_BLOCK_SIZE: bool> Extend<bool>
    for BlockedBooleanBuilder<MANUAL_BLOCK_SIZE>
{
    fn extend<T: IntoIterator<Item = bool>>(&mut self, iter: T) {
        let mut iter = iter.into_iter();

        if !MANUAL_BLOCK_SIZE {
            self.extend_validity_in_block(iter);
            return;
        }

        loop {
            let remaining_in_current_block = self.current_block_remaining_len();
            let added_items = self
                .extend_validity_in_block(iter.by_ref().take(remaining_in_current_block));

            if added_items == 0 {
                break;
            }
        }
    }
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
