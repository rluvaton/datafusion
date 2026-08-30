use arrow::array::{BooleanBufferBuilder, OffsetSizeTrait};
use arrow::buffer::{BooleanBuffer, Buffer};
use std::collections::VecDeque;
use std::ops::Range;
use itertools::Itertools;
use datafusion_common::utils::proxy::{VecAllocExt, VecDequeAllocExt};
use crate::blocked_helpers::take_n_helpers::{take_n_from_blocks, BlockBuilder};

#[derive(Debug)]
pub struct BlockedBytesBufferBuilder {
    /// Using `VecDeque` so we can remove the first block and reclaim memory
    blocks: VecDeque<Vec<u8>>,

    len: usize,

    finished_blocks_mem: usize,
}

impl BlockedBytesBufferBuilder {
    pub fn new() -> Self {
        let blocks = VecDeque::from(vec![vec![]]);
        BlockedBytesBufferBuilder { blocks, finished_blocks_mem: 0, len: 0 }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn allocated_size(&self) -> usize {
        self.finished_blocks_mem +
          self.blocks.allocated_size() + self.blocks.back().map_or(0, |b| b.allocated_size())
    }

    pub fn current_block_len(&self) -> usize {
        self.blocks[self.blocks.len() - 1].len()
    }

    pub fn block(&self, block_index: usize) -> &Vec<u8> {
        &self.blocks[block_index]
    }

    pub fn reserve_bytes_in_current_block(&mut self, capacity: usize) {
        let block = self.blocks.back_mut().unwrap();

        // Not adding to finished blocks mem since it does not contain the last block
        block.reserve(capacity);
    }

    pub(crate) fn reserve_blocks(&mut self, n: usize) {
        self.blocks.reserve(n);
    }

    pub fn start_new_block(&mut self) {
        self.finished_blocks_mem += self.blocks.back().map_or(0, |b| b.allocated_size());
        self.blocks.push_back(vec![]);
    }

    pub fn extend_from_slice(&mut self, slice: &[u8]) {
        let block = &mut self.blocks.back_mut().unwrap();
        self.len += slice.len();

        block.extend_from_slice(slice);
    }

    pub(crate) fn reserve_capacity_in_current_block(&mut self, capacity: usize) {
        let block = self.blocks.back_mut().unwrap();

        block.reserve(capacity);
    }

    /// Extend the bytes at the provided offsets
    pub(crate) fn extends_bytes_from_offsets_indexes_in_current_block<
        O: OffsetSizeTrait,
    >(
        &mut self,
        bytes: &[u8],
        offset_buffer_slice: &[O],
        indexes: &[usize],
    ) {
        let block = self.blocks.back_mut().unwrap();

        for &index_to_copy in indexes {
            let from = offset_buffer_slice[index_to_copy - 1].as_usize();
            let to = offset_buffer_slice[index_to_copy].as_usize();

            block.extend_from_slice(&bytes[from..to]);
            self.len += to - from;
        }
    }

    pub fn take_block(&mut self) -> Option<Vec<u8>> {
        let current_block = self.blocks.pop_front();
        self.len -= current_block.as_ref().map(|b| b.len()).unwrap_or(0);

        // TODO - this will create infinite loop that take_block will never return None
        //        but we still need to have a new empty block for next emit
        if self.blocks.is_empty() {
            // Add a new empty block for next emit
            self.blocks.push_back(vec![]);
        } else {
            // Only if not the last block since the current block is being calculated separately
            self.finished_blocks_mem -= current_block
              .as_ref()
              .map(|block| block.capacity())
              .unwrap_or(0);
        }

        current_block
    }

    pub fn take_all(&mut self) -> Vec<Vec<u8>> {
        let blocks = std::mem::take(&mut self.blocks);
        assert_ne!(blocks.len(), 0);

        // TODO - should preallocate? can be expensive for large schema
        self.blocks.push_back(vec![]);
        self.finished_blocks_mem = 0;
        self.len = 0;

        blocks.into()
    }

    pub fn take_block_finished(&mut self) -> Option<Buffer> {
        let block = self.take_block()?;
        Some(Buffer::from(block))
    }

    pub fn take_n(&mut self, n: usize, adjusted_block_size_iter: impl Iterator<Item=usize> + Clone) -> Vec<u8> {
        let (taken, layout) = take_n_from_blocks(
            &mut self.blocks,
            self.len,
            n,
            None,
            adjusted_block_size_iter,
        );

        self.len = layout.len;
        self.finished_blocks_mem = layout.finished_blocks_allocated_size;

        taken
    }
}


impl BlockBuilder for Vec<u8> {
    type Output = Vec<u8>;

    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn truncate(&mut self, len: usize) {
        Vec::truncate(self, len)
    }

    fn append_range(&mut self, src: &Self, range: Range<usize>) {
        self.extend_from_slice(&src[range])
    }

    fn shift_down(&mut self, offset: usize, len: usize) {
        if offset > 0 {
            self.copy_within(offset..offset + len, 0);
        }

        Vec::truncate(self, len)
    }

    fn allocated_size(&self) -> usize {
        self.allocated_size()
    }

    fn finish(self) -> Vec<u8> {
        self
    }
}

impl<'a> Extend<&'a [u8]> for BlockedBytesBufferBuilder {
    fn extend<T: IntoIterator<Item = &'a [u8]>>(&mut self, iter: T) {
        let block = &mut self.blocks.back_mut().unwrap();
        let before = block.capacity();
        for slice in iter {
            block.extend_from_slice(slice);
        }

        self.finished_blocks_mem += (block.capacity() - before);
    }
}
