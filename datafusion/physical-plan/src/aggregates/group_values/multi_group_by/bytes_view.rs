// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::aggregates::group_values::multi_group_by::{
    GroupColumn, Nulls, nulls_equal_to,
};
use arrow::array::{
    Array, ArrayRef, AsArray, BooleanBufferBuilder, ByteView, GenericByteViewArray,
    make_view,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::ByteViewType;
use datafusion_common::Result;
use datafusion_common::utils::split_vec_min_alloc;
use datafusion_expr_common::groups_accumulator::BlocksIndex;
use datafusion_functions_aggregate_common::blocked_helpers::{
    BlockedBytesBufferBuilder, BlockedNullsBuilder, BlockedVecBuilder,
};
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::mem::{replace, size_of};
use std::sync::Arc;

const BYTE_VIEW_MAX_BLOCK_SIZE: usize = 2 * 1024 * 1024;

/// An implementation of [`GroupColumn`] for binary view and utf8 view types.
///
/// Stores a collection of binary view or utf8 view group values in a buffer
/// whose structure is similar to `GenericByteViewArray`, and we can get benefits:
///
/// 1. Efficient comparison of incoming rows to existing rows
/// 2. Efficient construction of the final output array
/// 3. Efficient to perform `take_n` comparing to use `GenericByteViewBuilder`
pub struct ByteViewGroupValueBuilder<const FIXED_BLOCK_SIZING: bool, B: ByteViewType> {
    /// The views of string values
    ///
    /// If string len <= 12, the view's format will be:
    ///   string(12B) | len(4B)
    ///
    /// If string len > 12, its format will be:
    ///     offset(4B) | buffer_index(4B) | prefix(4B) | len(4B)
    views: BlockedVecBuilder<FIXED_BLOCK_SIZING, u128>,

    /// The progressing block
    ///
    /// New values will be inserted into it until its capacity
    /// is not enough(detail can see `max_block_size`).
    /// this is why it is marked as managed blocks
    data_blocks: BlockedBytesBufferBuilder,

    /// For each block, how many blocks are there
    view_blocks_per_block: VecDeque<usize>,

    /// The max size of `in_progress`
    ///
    /// `in_progress` will be flushed into `completed`, and create new `in_progress`
    /// when found its remaining capacity(`max_block_size` - `len(in_progress)`),
    /// is no enough to store the appended value.
    ///
    /// Currently it is fixed at 2MB.
    max_block_size: usize,

    /// Nulls
    nulls: BlockedNullsBuilder<FIXED_BLOCK_SIZING>,

    /// phantom data so the type requires `<B>`
    _phantom: PhantomData<B>,
}

impl<const FIXED_BLOCK_SIZING: bool, B: ByteViewType>
    ByteViewGroupValueBuilder<FIXED_BLOCK_SIZING, B>
{
    pub fn new(block_size: usize, max_buffer_block_size: Option<usize>) -> Self {
        if FIXED_BLOCK_SIZING {
            assert_ne!(block_size, 0);
        }

        Self {
            views: BlockedVecBuilder::new(block_size),
            data_blocks: BlockedBytesBufferBuilder::new(),
            view_blocks_per_block: VecDeque::from(vec![1]),
            max_block_size: max_buffer_block_size.unwrap_or(BYTE_VIEW_MAX_BLOCK_SIZE),
            nulls: BlockedNullsBuilder::new(block_size),
            _phantom: PhantomData {},
        }
    }

    /// Set the max block size
    fn with_max_block_size(mut self, max_block_size: usize) -> Self {
        self.max_block_size = max_block_size;
        self
    }

    fn equal_to_inner(
        &self,
        lhs_row: BlocksIndex,
        array: &ArrayRef,
        rhs_row: usize,
    ) -> bool {
        let array = array.as_byte_view::<B>();
        // since this is a single row comparison, don't bother specializing for nulls/buffers
        self.do_equal_to_inner::<true, true>(lhs_row, array, rhs_row)
    }

    fn append_val_inner(&mut self, array: &ArrayRef, row: usize) {
        let arr = array.as_byte_view::<B>();

        // Null row case, set and return
        if arr.is_null(row) {
            self.nulls.push_null();
            self.views.push(0);
            return;
        }

        // Not null row case
        self.nulls.push_non_null();
        self.do_append_val_inner(arr, row);
    }

    // Don't inline to keep the code small and give LLVM the best chance of
    // vectorizing the inner loop
    #[inline(never)]
    fn vectorized_equal_to_inner<const HAS_NULLS: bool, const HAS_BUFFERS: bool>(
        &self,
        lhs_rows: &[BlocksIndex],
        array: &GenericByteViewArray<B>,
        rhs_rows: &[usize],
        equal_to_results: &mut BooleanBufferBuilder,
    ) {
        for (idx, (&lhs_row, &rhs_row)) in
            lhs_rows.iter().zip(rhs_rows.iter()).enumerate()
        {
            if !equal_to_results.get_bit(idx) {
                continue;
            }

            if !self.do_equal_to_inner::<HAS_NULLS, HAS_BUFFERS>(lhs_row, array, rhs_row)
            {
                equal_to_results.set_bit(idx, false);
            }
        }
    }

    fn vectorized_append_inner(&mut self, array: &ArrayRef, rows: &[usize]) {
        let arr = array.as_byte_view::<B>();
        let null_count = array.null_count();
        let num_rows = array.len();
        let all_null_or_non_null = if null_count == 0 {
            Nulls::None
        } else if null_count == num_rows {
            Nulls::All
        } else {
            Nulls::Some
        };

        match all_null_or_non_null {
            Nulls::Some => {
                for &row in rows {
                    self.append_val_inner(array, row);
                }
            }

            Nulls::None => {
                self.nulls.push_n_non_nulls(rows.len());
                for &row in rows {
                    self.do_append_val_inner(arr, row);
                }
            }

            Nulls::All => {
                self.nulls.push_n_nulls(rows.len());
                self.views.push_value_n(0, rows.len());
            }
        }
    }

    fn do_append_val_inner(&mut self, array: &GenericByteViewArray<B>, row: usize)
    where
        B: ByteViewType,
    {
        let value: &[u8] = array.value(row).as_ref();

        let value_len = value.len();
        let view = if value_len <= 12 {
            make_view(value, 0, 0)
        } else {
            // Ensure big enough block to hold the value firstly
            self.ensure_in_progress_big_enough(value_len);

            // Append value

            // Make buffer index relative to the start of the current block
            let buffer_index = self.view_blocks_per_block.back().unwrap() - 1;
            let offset = self
                .data_blocks
                .current_block_len()
                .expect("must have current block");
            self.data_blocks.extend_from_slice(value);

            make_view(value, buffer_index as u32, offset as u32)
        };

        // Append view
        self.views.push(view);
    }

    fn ensure_in_progress_big_enough(&mut self, value_len: usize) {
        debug_assert!(value_len > 12);
        let require_cap = self
            .data_blocks
            .current_block_len()
            .expect("must have current block")
            + value_len;

        // If current block isn't big enough, flush it and create a new in progress block
        if require_cap > self.max_block_size {
            self.data_blocks.start_new_block();
            // Increase number of blocks
            *self.view_blocks_per_block.back_mut().unwrap() += 1;
        }
    }

    /// Compare the value at `lhs_row` in this builder with
    /// the value at `rhs_row` in input `array`
    ///
    /// Templated so that the inner compare loop can be
    /// specialized based on the input array
    #[inline(always)]
    fn do_equal_to_inner<const HAS_NULLS: bool, const HAS_BUFFERS: bool>(
        &self,
        lhs_row: BlocksIndex,
        array: &GenericByteViewArray<B>,
        rhs_row: usize,
    ) -> bool {
        // Check if nulls equal firstly
        if HAS_NULLS {
            let exist_null = self.nulls.is_null(lhs_row);
            let input_null = array.is_null(rhs_row);
            if let Some(result) = nulls_equal_to(exist_null, input_null) {
                return result;
            }
        }

        // Otherwise, we need to check their values

        // TODO - add back the get unchecked
        let exist_view = self.views[lhs_row];
        let exist_view_len = exist_view as u32;

        let input_view = unsafe { *array.views().get_unchecked(rhs_row) };
        let input_view_len = input_view as u32;

        // fast path, if we know there are no buffers, then the view must be inlined
        // so we can simply compare the u128 views
        if !HAS_BUFFERS {
            return exist_view == input_view;
        }

        // The check logic
        //   - Check len equality
        //   - If inlined, check inlined value
        //   - If non-inlined, check prefix and then check value in buffer
        //     when needed
        if exist_view_len != input_view_len {
            return false;
        }

        if exist_view_len <= 12 {
            // both inlined, so compare inlined value
            exist_view == input_view
        } else {
            let exist_prefix =
                unsafe { GenericByteViewArray::<B>::inline_value(&exist_view, 4) };
            let input_prefix =
                unsafe { GenericByteViewArray::<B>::inline_value(&input_view, 4) };

            if exist_prefix != input_prefix {
                return false;
            }

            // get the full values and compare
            let exist_full = {
                let byte_view = ByteView::from(exist_view);
                let buffer_index = byte_view.buffer_index as usize;
                let offset = byte_view.offset as usize;
                let length = byte_view.length as usize;
                let current_block_num_data_blocks =
                    *self.view_blocks_per_block.back().unwrap() - 1;
                debug_assert!(buffer_index <= current_block_num_data_blocks);

                let actual_buffer_index = self.data_blocks.num_blocks()
                    - current_block_num_data_blocks
                    + buffer_index;

                let block = self.data_blocks.block(actual_buffer_index);

                unsafe { block.as_slice().get_unchecked(offset..offset + length) }
            };
            let input_full: &[u8] = unsafe { array.value_unchecked(rhs_row).as_ref() };
            exist_full == input_full
        }
    }
}

impl<const FIXED_BLOCK_SIZING: bool, B: ByteViewType> GroupColumn<FIXED_BLOCK_SIZING>
    for ByteViewGroupValueBuilder<FIXED_BLOCK_SIZING, B>
{
    fn equal_to(&self, lhs_row: BlocksIndex, array: &ArrayRef, rhs_row: usize) -> bool {
        self.equal_to_inner(lhs_row, array, rhs_row)
    }

    fn append_val(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        self.append_val_inner(array, row);
        Ok(())
    }

    fn vectorized_equal_to(
        &self,
        group_indices: &[BlocksIndex],
        array: &ArrayRef,
        rows: &[usize],
        equal_to_results: &mut BooleanBufferBuilder,
    ) {
        let has_nulls = array.null_count() != 0;
        let array = array.as_byte_view::<B>();
        let has_buffers = !array.data_buffers().is_empty();
        // call specialized version based on nulls and buffers presence
        match (has_nulls, has_buffers) {
            (true, true) => self.vectorized_equal_to_inner::<true, true>(
                group_indices,
                array,
                rows,
                equal_to_results,
            ),
            (true, false) => self.vectorized_equal_to_inner::<true, false>(
                group_indices,
                array,
                rows,
                equal_to_results,
            ),
            (false, true) => self.vectorized_equal_to_inner::<false, true>(
                group_indices,
                array,
                rows,
                equal_to_results,
            ),
            (false, false) => self.vectorized_equal_to_inner::<false, false>(
                group_indices,
                array,
                rows,
                equal_to_results,
            ),
        }
    }

    fn vectorized_append(&mut self, array: &ArrayRef, rows: &[usize]) -> Result<()> {
        self.vectorized_append_inner(array, rows);
        Ok(())
    }

    fn len(&self) -> usize {
        self.views.len()
    }

    fn size(&self) -> usize {
        self.nulls.allocated_size()
            + self.views.allocated_size()
            + self.data_blocks.allocated_size()
            + size_of::<Self>()
    }

    fn take_block(&mut self) -> Option<ArrayRef> {
        let views = self.views.take_block_finished();

        let null_buffer = self.nulls.take_block();

        assert!(
            self.view_blocks_per_block.len() >= 1,
            "must have only 1 views per block since no blocks"
        );

        let (views, null_buffer) = match (views, null_buffer) {
            (Some(views), Some(null_buffer)) => (views, null_buffer),
            (None, None) => {
                assert_eq!(
                    self.view_blocks_per_block.back(),
                    Some(&1_usize),
                    "must have only 1 block"
                );
                assert_eq!(
                    self.data_blocks.current_block_len(),
                    Some(0),
                    "Must have empty block"
                );
                return None;
            }
            (Some(_), None) => unreachable!("must have null buffer if views are present"),
            (None, Some(_)) => unreachable!("must have views if null buffer is present"),
        };

        let current_block_count = {
            let number_of_views_per_block =
                self.view_blocks_per_block.back_mut().unwrap();

            std::mem::replace(number_of_views_per_block, 1)
        };

        let buffers = (0..current_block_count)
            .map(|_| {
                self.data_blocks
                    .take_block_finished()
                    .expect("must have the block")
            })
            .collect::<Vec<_>>();

        // Take n for values:
        //   - Take first n `view`s from `views`
        //
        //   - Find the last non-inlined `view`, if all inlined,
        //     we can build array and return happily, otherwise we
        //     we need to continue to process related buffers
        //
        //   - Get the last related `buffer index`(let's name it `buffer index n`)
        //     from last non-inlined `view`
        //
        //   - Take buffers, the key is that we need to know if we need to take
        //     the whole last related buffer. The logic is a bit complex, you can
        //     detail in `take_buffers_with_whole_last`, `take_buffers_with_partial_last`
        //     and other related steps in following
        //
        //   - Shift the `buffer index` of remaining non-inlined `views`
        //

        let last_non_inlined_view =
            views.iter().rev().find(|view| ((**view) as u32) > 12);

        // All taken views inlined
        if last_non_inlined_view.is_none() {
            // Safety:
            // * all views were correctly made
            // * (if utf8): Input was valid Utf8 so buffer contents are
            // valid utf8 as well
            unsafe {
                return Some(Arc::new(GenericByteViewArray::<B>::new_unchecked(
                    views,
                    Vec::new(),
                    null_buffer,
                )));
            }
        };

        // Safety:
        // * all views were correctly made
        // * (if utf8): Input was valid Utf8 so buffer contents are
        // valid utf8 as well
        Some(unsafe {
            Arc::new(GenericByteViewArray::<B>::new_unchecked(
                views,
                buffers,
                null_buffer,
            ))
        })
    }

    fn start_new_block(&mut self) {
        // TODO - should it be 0 or 1
        self.view_blocks_per_block.push_back(1);
        self.data_blocks.start_new_block();
        self.views.start_new_block();
        self.nulls.start_new_block();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow::array::{
        ArrayRef, AsArray, BooleanBufferBuilder, NullBufferBuilder, StringViewArray,
    };
    use arrow::datatypes::StringViewType;

    use super::GroupColumn;

    fn make_true_buffer(n: usize) -> BooleanBufferBuilder {
        let mut buf = BooleanBufferBuilder::new(n);
        buf.append_n(n, true);
        buf
    }

    fn to_vec(buf: &BooleanBufferBuilder) -> Vec<bool> {
        (0..buf.len()).map(|i| buf.get_bit(i)).collect()
    }

    #[test]
    fn test_byte_view_append_val() {
        let mut builder =
            ByteViewGroupValueBuilder::<false, StringViewType>::new(0, None)
                .with_max_block_size(60);
        let builder_array = StringViewArray::from(vec![
            Some("this string is quite long"), // in buffer 0
            Some("foo"),
            None,
            Some("bar"),
            Some("this string is also quite long"), // buffer 0
            Some("this string is quite long"),      // buffer 1
            Some("bar"),
        ]);
        let builder_array: ArrayRef = Arc::new(builder_array);
        for row in 0..builder_array.len() {
            builder.append_val(&builder_array, row).unwrap();
        }

        // let output = Box::new(builder).build();
        let output = Box::new(builder).take_block().unwrap();
        // should be 2 output buffers to hold all the data
        assert_eq!(output.as_string_view().data_buffers().len(), 2);
        assert_eq!(&output, &builder_array)
    }

    #[test]
    fn test_byte_view_equal_to() {
        let append = |builder: &mut ByteViewGroupValueBuilder<false, StringViewType>,
                      builder_array: &ArrayRef,
                      append_rows: &[usize]| {
            for &index in append_rows {
                builder.append_val(builder_array, index).unwrap();
            }
        };

        let equal_to =
            |builder: &ByteViewGroupValueBuilder<false, StringViewType>,
             lhs_rows: &[BlocksIndex],
             input_array: &ArrayRef,
             rhs_rows: &[usize],
             equal_to_results: &mut BooleanBufferBuilder| {
                let iter = lhs_rows.iter().zip(rhs_rows.iter());
                for (idx, (&lhs_row, &rhs_row)) in iter.enumerate() {
                    equal_to_results
                        .set_bit(idx, builder.equal_to(lhs_row, input_array, rhs_row));
                }
            };

        test_byte_view_equal_to_internal(append, equal_to);
    }

    #[test]
    fn test_byte_view_vectorized_equal_to() {
        let append = |builder: &mut ByteViewGroupValueBuilder<false, StringViewType>,
                      builder_array: &ArrayRef,
                      append_rows: &[usize]| {
            builder
                .vectorized_append(builder_array, append_rows)
                .unwrap();
        };

        let equal_to =
            |builder: &ByteViewGroupValueBuilder<false, StringViewType>,
             lhs_rows: &[BlocksIndex],
             input_array: &ArrayRef,
             rhs_rows: &[usize],
             equal_to_results: &mut BooleanBufferBuilder| {
                builder.vectorized_equal_to(
                    lhs_rows,
                    input_array,
                    rhs_rows,
                    equal_to_results,
                );
            };

        test_byte_view_equal_to_internal(append, equal_to);
    }

    #[test]
    fn test_byte_view_vectorized_operation_special_case() {
        // Test the special `all nulls` or `not nulls` input array case
        // for vectorized append and equal to

        let mut builder =
            ByteViewGroupValueBuilder::<false, StringViewType>::new(0, None)
                .with_max_block_size(60);

        // All nulls input array
        let all_nulls_input_array = Arc::new(StringViewArray::from(vec![
            Option::<&str>::None,
            None,
            None,
            None,
            None,
        ])) as _;
        builder
            .vectorized_append(&all_nulls_input_array, &[0, 1, 2, 3, 4])
            .unwrap();

        let mut equal_to_results = make_true_buffer(all_nulls_input_array.len());
        builder.vectorized_equal_to(
            &[0, 1, 2, 3, 4].map(BlocksIndex::new_in_first_block),
            &all_nulls_input_array,
            &[0, 1, 2, 3, 4],
            &mut equal_to_results,
        );
        let results = to_vec(&equal_to_results);

        assert!(results[0]);
        assert!(results[1]);
        assert!(results[2]);
        assert!(results[3]);
        assert!(results[4]);

        // All not nulls input array
        let all_not_nulls_input_array = Arc::new(StringViewArray::from(vec![
            Some("stringview1"),
            Some("stringview2"),
            Some("stringview3"),
            Some("stringview4"),
            Some("stringview5"),
        ])) as _;
        builder
            .vectorized_append(&all_not_nulls_input_array, &[0, 1, 2, 3, 4])
            .unwrap();

        let mut equal_to_results = make_true_buffer(all_not_nulls_input_array.len());
        builder.vectorized_equal_to(
            &[5, 6, 7, 8, 9].map(BlocksIndex::new_in_first_block),
            &all_not_nulls_input_array,
            &[0, 1, 2, 3, 4],
            &mut equal_to_results,
        );
        let results = to_vec(&equal_to_results);

        assert!(results[0]);
        assert!(results[1]);
        assert!(results[2]);
        assert!(results[3]);
        assert!(results[4]);
    }

    fn test_byte_view_equal_to_internal<A, E>(mut append: A, mut equal_to: E)
    where
        A: FnMut(
            &mut ByteViewGroupValueBuilder<false, StringViewType>,
            &ArrayRef,
            &[usize],
        ),
        E: FnMut(
            &ByteViewGroupValueBuilder<false, StringViewType>,
            &[BlocksIndex],
            &ArrayRef,
            &[usize],
            &mut BooleanBufferBuilder,
        ),
    {
        // Will cover such cases:
        //   - exist null, input not null
        //   - exist null, input null; values not equal
        //   - exist null, input null; values equal
        //   - exist not null, input null
        //   - exist not null, input not null; value lens not equal
        //   - exist not null, input not null; value not equal(inlined case)
        //   - exist not null, input not null; value equal(inlined case)
        //
        //   - exist not null, input not null; value not equal
        //     (non-inlined case + prefix not equal)
        //
        //   - exist not null, input not null; value not equal
        //     (non-inlined case + value in `completed`)
        //
        //   - exist not null, input not null; value equal
        //     (non-inlined case + value in `completed`)
        //
        //   - exist not null, input not null; value not equal
        //     (non-inlined case + value in `in_progress`)
        //
        //   - exist not null, input not null; value equal
        //     (non-inlined case + value in `in_progress`)

        // Set the block size to 40 for ensuring some unlined values are in `in_progress`,
        // and some are in `completed`, so both two branches in `value` function can be covered.
        let mut builder =
            ByteViewGroupValueBuilder::<false, StringViewType>::new(0, Some(60));
        let builder_array = Arc::new(StringViewArray::from(vec![
            None,
            None,
            None,
            Some("foo"),
            Some("bazz"),
            Some("foo"),
            Some("bar"),
            Some("I am a long string for test eq in completed"),
            Some("I am a long string for test eq in progress"),
        ])) as ArrayRef;
        append(&mut builder, &builder_array, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);

        // Define input array
        let (views, buffer, _nulls) = StringViewArray::from(vec![
            Some("foo"),
            Some("bar"),
            None,
            None,
            Some("baz"),
            Some("oof"),
            Some("bar"),
            Some("i am a long string for test eq in completed"),
            Some("I am a long string for test eq in COMPLETED"),
            Some("I am a long string for test eq in completed"),
            Some("I am a long string for test eq in PROGRESS"),
            Some("I am a long string for test eq in progress"),
        ])
        .into_parts();

        // explicitly build a null buffer where one of the null values also happens to match
        let mut nulls = NullBufferBuilder::new(9);
        nulls.append_non_null();
        nulls.append_null();
        nulls.append_null();
        nulls.append_null();
        nulls.append_non_null();
        nulls.append_non_null();
        nulls.append_non_null();
        nulls.append_non_null();
        nulls.append_non_null();
        nulls.append_non_null();
        nulls.append_non_null();
        nulls.append_non_null();
        let input_array =
            Arc::new(StringViewArray::new(views, buffer, nulls.finish())) as ArrayRef;

        // Check
        let mut equal_to_results = make_true_buffer(input_array.len());
        equal_to(
            &builder,
            &[0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 8].map(BlocksIndex::new_in_first_block),
            &input_array,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            &mut equal_to_results,
        );
        let results = to_vec(&equal_to_results);

        assert!(!results[0]);
        assert!(results[1]);
        assert!(results[2]);
        assert!(!results[3]);
        assert!(!results[4]);
        assert!(!results[5]);
        assert!(results[6]);
        assert!(!results[7]);
        assert!(!results[8]);
        assert!(results[9]);
        assert!(!results[10]);
        assert!(results[11]);
    }

    #[test]
    fn test_byte_view_take_n() {
        // ####### Define cases and init #######

        // `take_n` is really complex, we should consider and test following situations:
        //   1. Take nulls
        //   2. Take all `inlined`s
        //   3. Take non-inlined + partial last buffer in `completed`
        //   4. Take non-inlined + whole last buffer in `completed`
        //   5. Take non-inlined + partial last `in_progress`
        //   6. Take non-inlined + whole last buffer in `in_progress`
        //   7. Take all views at once

        let mut builder =
            ByteViewGroupValueBuilder::<false, StringViewType>::new(0, Some(60));
        let input_array = StringViewArray::from(vec![
            //  Test situation 1
            None,
            None,
            // Test situation 2 (also test take null together)
            None,
            Some("foo"),
            Some("bar"),
            // Test situation 3 (also test take null + inlined)
            None,
            Some("foo"),
            Some("this string is quite long"),
            Some("this string is also quite long"),
            // Test situation 4 (also test take null + inlined)
            None,
            Some("bar"),
            Some("this string is quite long"),
            // Test situation 5 (also test take null + inlined)
            None,
            Some("foo"),
            Some("another string that is is quite long"),
            Some("this string not so long"),
            // Test situation 6 (also test take null + inlined + insert again after taking)
            None,
            Some("bar"),
            Some("this string is quite long"),
            // Insert 4 and just take 3 to ensure it will go the path of situation 6
            None,
            // Finally, we create a new builder,  insert the whole array and then
            // take whole at once for testing situation 7
        ]);

        let input_array: ArrayRef = Arc::new(input_array);
        let first_ones_to_append = 16; // For testing situation 1~5
        let second_ones_to_append = 4; // For testing situation 6
        let final_ones_to_append = input_array.len(); // For testing situation 7

        // ####### Test situation 1~5 #######
        for row in 0..first_ones_to_append {
            builder.append_val(&input_array, row).unwrap();
        }

        assert_eq!(builder.completed.len(), 2);
        assert_eq!(builder.in_progress.len(), 59);

        // Situation 1
        let taken_array = builder.take_n(2);
        assert_eq!(&taken_array, &input_array.slice(0, 2));

        // Situation 2
        let taken_array = builder.take_n(3);
        assert_eq!(&taken_array, &input_array.slice(2, 3));

        // Situation 3
        let taken_array = builder.take_n(3);
        assert_eq!(&taken_array, &input_array.slice(5, 3));

        let taken_array = builder.take_n(1);
        assert_eq!(&taken_array, &input_array.slice(8, 1));

        // Situation 4
        let taken_array = builder.take_n(3);
        assert_eq!(&taken_array, &input_array.slice(9, 3));

        // Situation 5
        let taken_array = builder.take_n(3);
        assert_eq!(&taken_array, &input_array.slice(12, 3));

        let taken_array = builder.take_n(1);
        assert_eq!(&taken_array, &input_array.slice(15, 1));

        // ####### Test situation 6 #######
        assert!(builder.completed.is_empty());
        assert!(builder.in_progress.is_empty());
        assert!(builder.views.is_empty());

        for row in first_ones_to_append..first_ones_to_append + second_ones_to_append {
            builder.append_val(&input_array, row).unwrap();
        }

        assert!(builder.completed.is_empty());
        assert_eq!(builder.in_progress.len(), 25);

        let taken_array = builder.take_n(3);
        assert_eq!(&taken_array, &input_array.slice(16, 3));

        // ####### Test situation 7 #######
        // Create a new builder
        let mut builder =
            ByteViewGroupValueBuilder::<false, StringViewType>::new(0, Some(60));

        for row in 0..final_ones_to_append {
            builder.append_val(&input_array, row).unwrap();
        }

        assert_eq!(builder.completed.len(), 3);
        assert_eq!(builder.in_progress.len(), 25);

        let taken_array = builder.take_n(final_ones_to_append);
        assert_eq!(&taken_array, &input_array);
    }

    #[test]
    fn test_byte_view_take_n_partial_completed_nonzero_index() {
        let mut builder =
            ByteViewGroupValueBuilder::<false, StringViewType>::new(0, Some(30));
        let input_array = StringViewArray::from(vec![
            Some("aaaaaaaaaaaaaa"),
            Some("bbbbbbbbbbbbbb"),
            Some("cccccccccccccc"),
            Some("dddddddddddddd"),
            Some("eeeeeeeeeeeeee"),
        ]);
        let input_array: ArrayRef = Arc::new(input_array);

        for row in 0..input_array.len() {
            builder.append_val(&input_array, row).unwrap();
        }

        assert_eq!(builder.completed.len(), 2);
        assert_eq!(builder.in_progress.len(), 14);

        let taken_array = builder.take_n(3);
        assert_eq!(&taken_array, &input_array.slice(0, 3));
    }
}
