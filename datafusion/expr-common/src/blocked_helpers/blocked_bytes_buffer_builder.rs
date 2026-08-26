use arrow::array::OffsetSizeTrait;
use arrow::buffer::Buffer;
use std::collections::VecDeque;
use datafusion_common::utils::proxy::{VecAllocExt, VecDequeAllocExt};

#[derive(Debug)]
pub struct BlockedBytesBufferBuilder {
    /// Using `VecDeque` so we can remove the first block and reclaim memory
    blocks: VecDeque<Vec<u8>>,

    finished_blocks_mem: usize,
}

impl BlockedBytesBufferBuilder {
    pub fn new() -> Self {
        let blocks = VecDeque::from(vec![vec![]]);
        BlockedBytesBufferBuilder { blocks, finished_blocks_mem: 0 }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn allocated_size(&self) -> usize {
        self.finished_blocks_mem +
          self.blocks.allocated_size() + self.blocks.back().map_or(0, |b| b.allocated_size())
    }

    pub fn current_block_len(&self) -> Option<usize> {
        self.blocks.back().map(|block| block.len())
    }

    pub fn block(&self, block_index: usize) -> &Vec<u8> {
        &self.blocks[block_index]
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
        }
    }

    pub fn take_block(&mut self) -> Option<Vec<u8>> {
        let current_block = self.blocks.pop_front();

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

    pub fn take_block_finished(&mut self) -> Option<Buffer> {
        let block = self.take_block()?;
        Some(Buffer::from(block))
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
