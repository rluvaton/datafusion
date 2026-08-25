use super::blocked_bytes_buffer_builder::BlockedBytesBufferBuilder;
use super::blocked_nulls_builder::BlockedNullsBuilder;
use super::blocked_offset_buffer_builder::BlockedOffsetBufferBuilder;
use arrow::array::{Array, GenericByteArray, OffsetSizeTrait};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{ArrowNativeType, ByteArrayType};
use std::collections::VecDeque;
use std::ops::{Deref, Index};
use datafusion_expr_common::groups_accumulator::BlocksIndex;

pub struct BlockedByteArrayBuilder<const FIXED_BLOCK_SIZING: bool, B: ByteArrayType> {
    blocked_offsets: BlockedOffsetBufferBuilder<FIXED_BLOCK_SIZING, B::Offset>,
    blocked_bytes: BlockedBytesBufferBuilder,
    blocked_nulls: BlockedNullsBuilder<FIXED_BLOCK_SIZING>,
}

impl<const FIXED_BLOCK_SIZING: bool, B: ByteArrayType> BlockedByteArrayBuilder<FIXED_BLOCK_SIZING, B> {
    pub fn new(block_size: usize) -> Self {
        assert_ne!(block_size, 0, "block size must be greater than 0");

        Self {
            blocked_offsets: BlockedOffsetBufferBuilder::new(block_size),
            blocked_bytes: BlockedBytesBufferBuilder::new(),
            blocked_nulls: BlockedNullsBuilder::new(block_size),
        }
    }

    pub fn len(&self) -> usize {
        self.blocked_offsets.len()
    }

    pub fn allocated_size(&self) -> usize {
        self.blocked_offsets.allocated_size()
            + self.blocked_bytes.allocated_size()
            + self.blocked_nulls.allocated_size()
    }

    fn block_size(&self) -> usize {
        assert!(FIXED_BLOCK_SIZING, "block size is only available for manual block");
        self.blocked_nulls.block_size()
    }

    pub fn push_null(&mut self) {
        self.blocked_nulls.push_null();
        let should_open_new_block = self.blocked_offsets.push_length(0);
        if should_open_new_block {
            self.blocked_bytes.start_new_block();
        }
    }

    /// Append n valids to the builder with the bytes being later appended
    pub fn append_n_valids(&mut self, n: usize) {
        self.blocked_nulls.push_n(n, true);
    }

    /// Append `n` nulls to the builder
    pub fn append_n_nulls(&mut self, mut n: usize) {
        self.blocked_nulls.push_n(n, true);

        if !FIXED_BLOCK_SIZING {
            self.blocked_offsets.push_empty_within_block(n);
            return;
        }

        while n > 0 {
            let remaining_in_current_block = self.current_block_remaining_len();
            let to_add = remaining_in_current_block.min(n);
            n -= to_add;

            let should_create_new_block = self.blocked_offsets.push_empty_within_block(to_add);
            if should_create_new_block {
                self.blocked_bytes.start_new_block();
            }
        }
    }

    pub fn push(&mut self, item: Option<&[u8]>) {
        let Some(bytes) = item else {
            self.push_null();

            return;
        };

        self.append_valid();
        self.append_valid_slice(bytes);
    }

    pub fn append_valid(&mut self) {
        self.blocked_nulls.push_non_null();
    }

    pub fn append_valid_slice(&mut self, bytes: &[u8]) {
        self.blocked_bytes.extend_from_slice(bytes);

        let should_open_new_block = self.blocked_offsets.push_length(bytes.len());
        if should_open_new_block {
            self.blocked_bytes.start_new_block();
        }
    }

    pub fn start_new_block(&mut self) {
        assert!(!FIXED_BLOCK_SIZING, "only valid when FIXED_BLOCK_SIZING is false");
        self.blocked_bytes.start_new_block();
        self.blocked_offsets.start_new_block();
        self.blocked_nulls.start_new_block();
    }

    pub fn current_block_bytes_len(&self) -> usize {
        // TODO - should always exists
        self.blocked_bytes.current_block_len().unwrap_or(0)
    }

    /// Extends iterator of lengths within current block
    /// Returns if the current block finished
    ///
    /// # Panics
    /// Panics if the iterator length exceeds the remaining size of the current block
    fn extend_items_in_block<'a>(
        &'a mut self,
        iter: impl Iterator<Item = Option<&'a [u8]>> + Clone,
        should_mark_empty: bool,
    ) -> bool {
        self.blocked_nulls
            .extend_validity_in_block(iter.clone().map(|item| item.is_some()));
        self.blocked_bytes
            .extend(iter.clone().filter_map(|item| item));
        let should_start_new_block = self.blocked_offsets.extend_length_in_block(
            iter.clone().map(|item| item.map_or(0, |bytes| bytes.len())),
            should_mark_empty,
        );
        if should_start_new_block {
            self.blocked_bytes.start_new_block();
        }

        should_start_new_block
    }

    fn current_block_remaining_len(&self) -> usize {
        self.blocked_offsets.current_block_remaining_len()
    }

    fn reserve_blocks(&mut self, n: usize) {
        self.blocked_nulls.reserve_blocks(n);
        self.blocked_offsets.reserve_blocks(n);
        self.blocked_bytes.reserve_blocks(n);
    }

    /// Extend the length from the current offsets
    pub fn extends_from_array(&mut self, array: &GenericByteArray<B>) {
        if !FIXED_BLOCK_SIZING {
            unimplemented!()
        }
        let number_of_blocks_to_reserve = array
            .len()
            .saturating_sub(self.current_block_remaining_len())
            .div_ceil(self.block_size());
        self.reserve_blocks(number_of_blocks_to_reserve);

        if let Some(null_buffer) = array.nulls().filter(|nulls| nulls.null_count() > 0) {
            let (offsets, buffer, _) = array.clone().into_parts();
            let mut offsets = offsets.deref();
            let bytes = buffer.deref();
            let mut null_buffer = null_buffer.clone();

            let mut len = array.len();
            let mut index = 0;

            while len > 0 {
                let remaining_in_current_block = self.current_block_remaining_len();
                let to_add = remaining_in_current_block.min(len);

                let offsets_in_block = &offsets[..to_add + 1];
                offsets = &offsets[to_add..];
                let bytes_in_block = &bytes[offsets_in_block[0].as_usize()..offsets_in_block[to_add].as_usize()];
                let null_buffer_block = null_buffer.slice(index, to_add);
                index += to_add;

                self.blocked_nulls
                    .extends_from_null_buffer_in_current_block(&null_buffer_block);
                if null_buffer_block.null_count() == 0 {
                    self.blocked_bytes.extend_from_slice(bytes_in_block);
                    let block_finished = self
                        .blocked_offsets
                        .extends_length_from_offsets_in_current_block(offsets_in_block);

                    if block_finished {
                        self.blocked_bytes.start_new_block();
                    }
                } else {
                    // Only add bytes where for valid
                    todo!()
                }
            }
        } else {
            // push n valid since no null buffer
            self.blocked_nulls.push_n(array.len(), true);

            let (offsets, buffer, _) = array.clone().into_parts();
            let mut offsets = offsets.deref();
            let bytes = buffer.deref();

            let mut len = array.len();

            while len > 0 {
                let remaining_in_current_block = self.current_block_remaining_len();
                let to_add = remaining_in_current_block.min(len);

                let offsets_in_block = &offsets[..to_add + 1];
                offsets = &offsets[to_add..];
                let bytes_in_block = &bytes[offsets_in_block[0].as_usize()..offsets_in_block[to_add + 1].as_usize()];

                self.blocked_bytes.extend_from_slice(bytes_in_block);
                let block_finished = self
                    .blocked_offsets
                    .extends_length_from_offsets_in_current_block(offsets_in_block);

                if block_finished {
                    self.blocked_bytes.start_new_block();
                }
            }
        }
    }

    pub fn value(&self, index: BlocksIndex) -> Option<&[u8]> {
        if !self.blocked_nulls[index] {
            return None;
        }

        Some(self.value_bytes(index))
    }

    /// return the current value of the specified row irrespective of null
    pub fn value_bytes(&self, index: BlocksIndex) -> &[u8] {
        let start_in_block = self.blocked_offsets[index].as_usize();

        // Offset in block + 1 always exists for offsets since block size is + 1
        let end_in_block = self.blocked_offsets[index.next_index_in_block()].as_usize();
        let bytes_block = self.blocked_bytes.block(index.block_index());

        // Safety: the offsets are constructed correctly and never decrease
        unsafe {
            bytes_block
                .as_slice()
                .get_unchecked(start_in_block..end_in_block)
        }
    }

    pub fn is_valid(&self, index: BlocksIndex) -> bool {
        self.blocked_nulls[index]
    }

    pub fn is_null(&self, index: BlocksIndex) -> bool {
        !self.is_valid(index)
    }

    pub fn value_len(&self, index: BlocksIndex) -> usize {
        let start_in_block = self.blocked_offsets[index];

        // Offset in block + 1 always exists for offsets since block size is + 1
        let end_in_block = self.blocked_offsets[index.next_index_in_block()];

        end_in_block.as_usize() - start_in_block.as_usize()
    }

    pub fn take_block(&mut self) -> Option<GenericByteArray<B>> {
        let offsets = self.blocked_offsets.take_block_finished()?;
        let blocked_nulls = self.blocked_nulls.take_block()?;
        let bytes = self.blocked_bytes.take_block_finished()?;

        Some(GenericByteArray::new(offsets, bytes, blocked_nulls))
    }

    /// Take a block but build it unchecked
    pub unsafe fn take_block_unchecked(&mut self) -> Option<GenericByteArray<B>> {
        let offsets = self.blocked_offsets.take_block_finished()?;
        let blocked_nulls = self.blocked_nulls.take_block()?;
        let bytes = self.blocked_bytes.take_block_finished()?;

        Some(GenericByteArray::new_unchecked(
            offsets,
            bytes,
            blocked_nulls,
        ))
    }
}

// impl<B: ByteArrayType> Index<usize> for BlockedByteArrayBuilder<B> {
//     type Output = Option<[u8]>;
//
//     fn index(&self, index: usize) -> &Self::Output {
//         let block_index = index / self.block_size();
//         let offset_in_block = index % self.block_size();
//         self.index((block_index, offset_in_block))
//     }
// }
//
// impl<B: ByteArrayType> Index<(usize, usize)> for BlockedByteArrayBuilder<B> {
//     type Output = Option<[u8]>;
//
//     fn index(&self, index: (usize, usize)) -> &Self::Output {
//         if !self.blocked_nulls[index] {
//             return &None;
//         }
//
//         let (block_index, offset_in_block) = index;
//
//         let start_in_block = self.blocked_offsets[index];
//
//         // Offset in block + 1 always exists for offsets since block size is + 1
//         let end_in_block = self.blocked_offsets[(block_index, offset_in_block + 1)];
//         let bytes_block = self.blocked_bytes.block(block_index);
//
//         &Some(bytes_block[start_in_block.as_usize()..end_in_block.as_usize()])
//     }
// }
