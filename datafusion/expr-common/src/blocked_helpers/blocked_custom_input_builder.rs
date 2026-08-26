use crate::groups_accumulator::BlocksIndex;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::{Index, IndexMut};

pub trait BlockProvider {
    type Block: Block;

    fn new_block(&self) -> Self::Block;
}

pub trait BlockProviderFinish: BlockProvider {
    type FinishedBlock;

    fn finish(&self, block: Self::Block) -> Self::FinishedBlock;
}

pub trait Block {
    type Item;

    /// Get allocated bytes on heap (not including `size_of::<Self>()`)
    fn allocated_size(&self) -> usize;

    fn push(&mut self, item: Self::Item);

    fn extend(&mut self, iter: impl Iterator<Item = Self::Item>);

    /// Number of items in the block
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;
}

pub trait BlockWithSlice: Block {
    fn extend_from_slice(&mut self, slice: &[Self::Item]);
    fn append_n(&mut self, item: Self::Item, n: usize);
}

/// When `FIXED_BLOCK_SIZING` is true, the block size is the `Self::block_size` otherwise,
/// the callers control the block size
#[derive(Debug)]
pub struct BlockedCustomInputBuilder<
    const FIXED_BLOCK_SIZING: bool,
    CustomBlockProvider: BlockProvider,
> {
    blocks_provider: CustomBlockProvider,
    /// Using `VecDeque` so we can remove the first block and reclaim memory
    blocks: VecDeque<CustomBlockProvider::Block>,

    /// The size of each block
    block_size: usize,

    /// The total number of items, not the number of offset since in each block there is the initial offset
    len: usize,

    /// The index of the current block
    current_block_index: usize,

    /// This will equal the number of blocks can be emitted
    ///
    /// We have this rather than using `blocks.len()` since we don't want to add a condition
    /// on every push that checks if we have a block, and if not adds 0 as the initial offset
    ///
    /// And we can't always have a block with 0 since then how could we know when all blocks were emitted
    /// if we keep adding empty block on finish
    number_of_blocks: usize,

    pending_block: bool,

    /// TODO - only have allocated size for the finished blocks and compute the last block,
    ///        to avoid code complexity and cost
    memory: usize,
}

impl<const FIXED_BLOCK_SIZING: bool, CustomBlockProvider: BlockProvider>
    BlockedCustomInputBuilder<FIXED_BLOCK_SIZING, CustomBlockProvider>
{
    // TODO - some want to preallocate the blocks and some don't,
    //        there should be a way while avoiding having a lot of memory used if all are prealocatting
    pub fn new(block_size: usize, blocks_provider: CustomBlockProvider) -> Self {
        if FIXED_BLOCK_SIZING {
            assert_ne!(block_size, 0, "block size must be greater than 0");
        }

        let blocks = VecDeque::from(vec![blocks_provider.new_block()]);
        let memory = blocks.capacity() * size_of::<CustomBlockProvider::Block>()
            + blocks[0].allocated_size();
        BlockedCustomInputBuilder {
            blocks_provider,
            blocks,
            block_size,
            len: 0,
            current_block_index: 0,
            number_of_blocks: 1,
            pending_block: false,
            memory,
        }
    }

    pub fn blocks_provider(&self) -> &CustomBlockProvider {
        &self.blocks_provider
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn allocated_size(&self) -> usize {
        self.memory
    }

    /// Get the number of elements in the current block (not the number of offsets since the first offset is always 0)
    pub fn current_block_len(&self) -> usize {
        self.blocks[self.current_block_index].len() - 1
    }

    pub fn start_new_block(&mut self) {
        // Don't add to number of blocks since we might not insert into it
        self.current_block_index += 1;
        let prev_capacity = self.blocks.capacity();
        let new_block = self.blocks_provider.new_block();
        self.memory += new_block.allocated_size();
        self.blocks.push_back(new_block);
        let new_capacity = self.blocks.capacity();
        self.memory +=
            (new_capacity - prev_capacity) * size_of::<CustomBlockProvider::Block>();
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

        self.memory += (self.blocks.capacity() - prev_capacity)
            * size_of::<CustomBlockProvider::Block>();
    }

    /// Push length and return if the current block is now full
    pub fn push(&mut self, value: <CustomBlockProvider::Block as Block>::Item) -> bool {
        let mut block = &mut self.blocks[self.current_block_index];

        let before = block.allocated_size();
        block.push(value);
        self.memory += (block.allocated_size() - before);
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
    pub(super) fn extend_in_block(
        &mut self,
        iter: impl Iterator<Item = <CustomBlockProvider::Block as Block>::Item>,
        should_mark_empty: bool,
    ) -> bool {
        let mut block = &mut self.blocks[self.current_block_index];

        let prev_block_len = block.len();
        let prev_block_capacity = block.allocated_size();
        block.extend(iter);

        if FIXED_BLOCK_SIZING {
            assert!(
                block.len() <= self.block_size,
                "overflow from block new block length: {}, block size: {}",
                block.len(),
                self.block_size
            );
        }

        self.memory += (block.allocated_size() - prev_block_capacity);

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

    /// Extends from slice within current block
    /// Returns if the current block has finished
    ///
    /// # Panics
    /// Panics if the iterator length exceeds the remaining size of the current block
    pub(super) fn extend_from_slice_in_block(
        &mut self,
        slice: &[<CustomBlockProvider::Block as Block>::Item],
    ) -> bool
    where
        CustomBlockProvider::Block: BlockWithSlice,
    {
        let mut block = &mut self.blocks[self.current_block_index];

        let prev_block_len = block.len();
        let prev_block_size = block.allocated_size();
        block.extend_from_slice(slice);

        if FIXED_BLOCK_SIZING {
            assert!(
                block.len() <= self.block_size,
                "overflow from block new block length: {}, block size: {}",
                block.len(),
                self.block_size
            );
        }

        self.memory += (block.allocated_size() - prev_block_size);

        let added_items = block.len() - prev_block_len;
        self.len += added_items;

        let finished_block = FIXED_BLOCK_SIZING && block.len() == self.block_size;

        if added_items > 0 {
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
    pub fn extend_from_slice(
        &mut self,
        mut buffer: &[<CustomBlockProvider::Block as Block>::Item],
    ) where
        CustomBlockProvider::Block: BlockWithSlice,
    {
        // If not fixed, then treat all offsets as single block
        if !FIXED_BLOCK_SIZING {
            self.extend_from_slice_in_block(buffer);

            return;
        }

        let number_of_blocks_to_reserve = buffer
            .len()
            .saturating_sub(self.current_block_remaining_len())
            .div_ceil(self.block_size);
        self.reserve_blocks(number_of_blocks_to_reserve);

        while !buffer.is_empty() {
            let remaining_in_current_block = self.current_block_remaining_len();
            let to_add = remaining_in_current_block.min(buffer.len());

            let (to_copy, rest) = buffer.split_at(to_add);
            buffer = rest;

            self.extend_from_slice_in_block(to_copy);
        }
    }

    pub(super) fn current_block_remaining_len(&self) -> usize {
        assert!(
            FIXED_BLOCK_SIZING,
            "current block remaining length is only relevant for manual block size"
        );
        self.block_size - self.blocks[self.current_block_index].len()
    }

    pub(crate) fn push_value_n_within_block(
        &mut self,
        value: <CustomBlockProvider::Block as Block>::Item,
        n: usize,
    ) -> bool
    where
        CustomBlockProvider::Block: BlockWithSlice,
    {
        self.len += n;
        let mut block = &mut self.blocks[self.current_block_index];

        let prev_capacity = block.allocated_size();

        if FIXED_BLOCK_SIZING {
            let new_len = block.len() + n;
            assert!(
                new_len <= self.block_size,
                "overflow from block new block length: {new_len}, block size: {}",
                self.block_size
            );
        }
        block.append_n(value, n);

        self.memory += (block.allocated_size() - prev_capacity);

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

    /// Push default
    pub fn push_default_n(&mut self, n: usize)
    where
        CustomBlockProvider::Block: BlockWithSlice,
        <CustomBlockProvider::Block as Block>::Item: Default + Clone,
    {
        self.push_value_n(<CustomBlockProvider::Block as Block>::Item::default(), n);
    }

    pub fn push_value_n(
        &mut self,
        value: <CustomBlockProvider::Block as Block>::Item,
        mut n: usize,
    ) where
        CustomBlockProvider::Block: BlockWithSlice,
        <CustomBlockProvider::Block as Block>::Item: Clone,
    {
        // If not fixed, then treat all offsets as single block
        if !FIXED_BLOCK_SIZING {
            self.push_value_n_within_block(value, n);
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

            self.push_value_n_within_block(value.clone(), to_add);
        }
    }

    pub fn take_block(&mut self) -> Option<CustomBlockProvider::Block> {
        if self.number_of_blocks == 0 {
            assert_eq!(
                self.blocks.len(),
                1,
                "even if number of blocks is 1 we must have at least one block for future insert"
            );
            assert_eq!(self.len, 0);
            assert_eq!(self.current_block_index, 0);
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

        self.memory -= block.allocated_size();
        self.memory -= (self.blocks.capacity() - prev_blocks_capacity)
            * size_of::<CustomBlockProvider::Block>();

        self.number_of_blocks -= 1;

        if self.blocks.is_empty() {
            self.current_block_index = 0;
            let prev_blocks_capacity = self.blocks.capacity();

            let block = self.blocks_provider.new_block();
            self.memory += block.allocated_size();
            self.blocks.push_back(block);
            self.memory += (self.blocks.capacity() - prev_blocks_capacity)
                * size_of::<CustomBlockProvider::Block>();
        } else {
            self.current_block_index -= 1;
        }

        let number_of_items = block.len() - 1;
        self.len -= number_of_items;

        Some(block)
    }

    pub fn take_block_finished(&mut self) -> Option<CustomBlockProvider::FinishedBlock>
    where
        CustomBlockProvider: BlockProviderFinish,
    {
        let block = self.take_block()?;

        let finished = self.blocks_provider.finish(block);

        Some(finished)
    }

    pub fn reset(&mut self) {
        self.blocks = VecDeque::from(vec![self.blocks_provider.new_block()]);
        self.len = 0;
        self.current_block_index = 0;
        self.number_of_blocks = 1;
        self.pending_block = false;
        self.memory = self.blocks.capacity() * size_of::<CustomBlockProvider::Block>()
            + self.blocks[0].allocated_size();
    }
}

impl<const FIXED_BLOCK_SIZING: bool, CustomBlockProvider: BlockProvider>
    Extend<<CustomBlockProvider::Block as Block>::Item>
    for BlockedCustomInputBuilder<FIXED_BLOCK_SIZING, CustomBlockProvider>
{
    fn extend<T: IntoIterator<Item = <CustomBlockProvider::Block as Block>::Item>>(
        &mut self,
        iter: T,
    ) {
        if !FIXED_BLOCK_SIZING {
            self.extend_in_block(iter.into_iter(), true);

            return;
        }

        let mut iter = iter.into_iter();

        let mut is_first = true;
        loop {
            let remaining_in_current_block = self.current_block_remaining_len();
            let block_finished = self.extend_in_block(
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

impl<CustomBlockProvider> Index<usize>
    for BlockedCustomInputBuilder<true, CustomBlockProvider>
where
    CustomBlockProvider: BlockProvider,
    CustomBlockProvider::Block:
        Index<usize, Output = <CustomBlockProvider::Block as Block>::Item>,
{
    type Output = <CustomBlockProvider::Block as Block>::Item;

    fn index(&self, index: usize) -> &Self::Output {
        self.index(BlocksIndex::from_index_in_fixed_block_size(
            index,
            self.block_size,
        ))
    }
}

impl<CustomBlockProvider> IndexMut<usize>
    for BlockedCustomInputBuilder<true, CustomBlockProvider>
where
    CustomBlockProvider: BlockProvider,
    CustomBlockProvider::Block:
        IndexMut<usize, Output = <CustomBlockProvider::Block as Block>::Item>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.index_mut(BlocksIndex::from_index_in_fixed_block_size(
            index,
            self.block_size,
        ))
    }
}

impl<const FIXED_BLOCK_SIZING: bool, CustomBlockProvider> Index<BlocksIndex>
    for BlockedCustomInputBuilder<FIXED_BLOCK_SIZING, CustomBlockProvider>
where
    CustomBlockProvider: BlockProvider,
    CustomBlockProvider::Block:
        Index<usize, Output = <CustomBlockProvider::Block as Block>::Item>,
{
    type Output = <CustomBlockProvider::Block as Block>::Item;

    fn index(&self, index: BlocksIndex) -> &Self::Output {
        &self.blocks[index.block_index()][index.index_in_block()]
    }
}

impl<const FIXED_BLOCK_SIZING: bool, CustomBlockProvider> IndexMut<BlocksIndex>
    for BlockedCustomInputBuilder<FIXED_BLOCK_SIZING, CustomBlockProvider>
where
    CustomBlockProvider: BlockProvider,
    CustomBlockProvider::Block:
        Index<usize, Output = <CustomBlockProvider::Block as Block>::Item>,
    CustomBlockProvider::Block:
        IndexMut<usize, Output = <CustomBlockProvider::Block as Block>::Item>,
{
    fn index_mut(&mut self, index: BlocksIndex) -> &mut Self::Output {
        &mut self.blocks[index.block_index()][index.index_in_block()]
    }
}
