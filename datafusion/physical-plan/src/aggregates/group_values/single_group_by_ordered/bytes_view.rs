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

use crate::aggregates::group_values::GroupValues;
use std::mem::size_of;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BinaryViewArray, ByteView, NullBufferBuilder};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{BinaryViewType, ByteViewType, StringViewType};
use datafusion_common::utils::proxy::VecAllocExt;
use datafusion_common::{Result, internal_err, not_impl_err};
use datafusion_expr::{EmitTo, GroupSelection};
use datafusion_physical_expr_common::binary_map::OutputType;

/// Max size of the in-progress buffer before flushing to completed buffers,
/// same as in [`ArrowBytesViewMap`]
///
/// [`ArrowBytesViewMap`]: datafusion_physical_expr_common::binary_view_map::ArrowBytesViewMap
const BYTE_VIEW_MAX_BLOCK_SIZE: usize = 2 * 1024 * 1024;

/// A [`GroupValues`] storing single column of Utf8View/BinaryView values when
/// the input is fully ordered by that column
///
/// Stores the values the same way as [`ArrowBytesViewMap`] but without the
/// hash table: since the input is ordered, equal values are adjacent, so a new
/// group starts whenever a value differs from the previous one.
///
/// [`ArrowBytesViewMap`]: datafusion_physical_expr_common::binary_view_map::ArrowBytesViewMap
pub struct FullyOrderedGroupValuesBytesView {
    /// Should the output be StringView or BinaryView?
    output_type: OutputType,
    /// Views for all stored values (in insertion order)
    views: Vec<u128>,
    /// In-progress buffer for out-of-line string data
    in_progress: Vec<u8>,
    /// Completed buffers containing string data
    completed: Vec<Buffer>,

    /// The group index of the null value if any
    null_group: Option<usize>,
}

impl FullyOrderedGroupValuesBytesView {
    pub fn new(output_type: OutputType) -> Self {
        assert!(
            matches!(output_type, OutputType::Utf8View | OutputType::BinaryView),
            "FullyOrderedGroupValuesBytesView only supports Utf8View or BinaryView output types"
        );
        Self {
            output_type,
            views: Vec::new(),
            in_progress: Vec::new(),
            completed: Vec::new(),
            null_group: None,
        }
    }

    fn intern_inner<B: ByteViewType>(&mut self, col: &ArrayRef, groups: &mut Vec<usize>) {
        let values = col.as_byte_view::<B>();
        let input_views: &[u128] = values.views();
        let input_buffers = values.data_buffers();
        let num_rows = values.len();
        let null_count = values.null_count();

        // The input is ordered, so the nulls are either all at the start or
        // all at the end of the column
        let (valid_start, valid_end) = if null_count > 0 && values.is_null(0) {
            (null_count, num_rows)
        } else {
            (0, num_rows - null_count)
        };
        debug_assert_eq!(
            values
                .slice(valid_start, valid_end - valid_start)
                .null_count(),
            0,
            "input is ordered, so all nulls should be at the start or at the end of the column"
        );

        // Leading nulls
        if valid_start > 0 {
            let null_group = self.null_group_for_next_rows();
            groups.resize(valid_start, null_group);
        }

        if valid_start < valid_end {
            // The first valid value either continues the current group or starts a new one
            let first_view = input_views[valid_start];
            let mut current_group = match self.current_valid_group() {
                // SAFETY: the views of a valid array point into its buffers
                Some(group)
                    if unsafe { self.group_equal_to_input(group, first_view, input_buffers) } =>
                {
                    group
                }
                // SAFETY: the views of a valid array point into its buffers
                _ => unsafe { self.append_input_view(first_view, input_buffers) },
            };

            // `groups` is filled in row order, so a run of rows in the same
            // group is filled at once when the next group starts
            let valid_views = &input_views[valid_start..valid_end];
            if input_buffers.is_empty() {
                // All the values are inlined, so equal values have equal views
                for (i, pair) in valid_views.windows(2).enumerate() {
                    if pair[0] != pair[1] {
                        groups.resize(valid_start + i + 1, current_group);
                        current_group = self.views.len();
                        self.views.push(pair[1]);
                    }
                }
            } else {
                for (i, pair) in valid_views.windows(2).enumerate() {
                    // SAFETY: the views of a valid array point into its buffers
                    if !unsafe { input_views_equal(pair[0], pair[1], input_buffers) } {
                        groups.resize(valid_start + i + 1, current_group);
                        current_group = unsafe { self.append_input_view(pair[1], input_buffers) };
                    }
                }
            }
            groups.resize(valid_end, current_group);
        }

        // Trailing nulls
        if valid_end < num_rows {
            let null_group = self.null_group_for_next_rows();
            groups.resize(num_rows, null_group);
        }
    }

    /// Returns true if the group at `group` is equal to the valid input `view`
    ///
    /// # Safety
    ///
    /// If not inlined, `view` must point into `input_buffers`
    #[inline]
    unsafe fn group_equal_to_input(
        &self,
        group: usize,
        view: u128,
        input_buffers: &[Buffer],
    ) -> bool {
        let group_view = self.views[group];

        // The length and the 4 bytes prefix are the lower 64 bits
        if group_view as u64 != view as u64 {
            return false;
        }

        // Fast path: inline values can be compared directly
        if view as u32 <= 12 {
            return group_view == view;
        }

        // Length and prefix matched - compare the rest of the bytes
        self.value(group)[4..] == unsafe { non_inlined_value(view, input_buffers) }[4..]
    }

    /// Append the valid input `view` as a new group and return its index
    ///
    /// # Safety
    ///
    /// If not inlined, `view` must point into `input_buffers`
    #[inline]
    unsafe fn append_input_view(&mut self, view: u128, input_buffers: &[Buffer]) -> usize {
        let group = self.views.len();

        // For inline values, the stored view is identical to the input view
        if view as u32 <= 12 {
            self.views.push(view);
            return group;
        }

        let value = unsafe { non_inlined_value(view, input_buffers) };

        // Ensure buffer is big enough
        if self.in_progress.len() + value.len() > BYTE_VIEW_MAX_BLOCK_SIZE {
            let flushed = std::mem::replace(
                &mut self.in_progress,
                Vec::with_capacity(BYTE_VIEW_MAX_BLOCK_SIZE),
            );
            self.completed.push(Buffer::from_vec(flushed));
        }

        // Reuse the length and prefix of the input view
        let stored_view = ByteView {
            buffer_index: self.completed.len() as u32,
            offset: self.in_progress.len() as u32,
            ..ByteView::from(view)
        };
        self.in_progress.extend_from_slice(value);
        self.views.push(stored_view.as_u128());

        group
    }

    /// Return the index of the last group if it is not the null group
    fn current_valid_group(&self) -> Option<usize> {
        let current_group = self.views.len().checked_sub(1)?;
        if self.null_group == Some(current_group) {
            None
        } else {
            Some(current_group)
        }
    }

    /// Return the null group, creating it if this is the first null
    fn null_group_for_next_rows(&mut self) -> usize {
        if let Some(null_group) = self.null_group {
            debug_assert_eq!(
                null_group,
                self.views.len() - 1,
                "input is ordered, so the null group must be the current group"
            );
            return null_group;
        }

        let null_group = self.views.len();
        self.views.push(0);
        self.null_group = Some(null_group);
        null_group
    }

    /// Returns the bytes for the group at `index`, irrespective of nullness.
    fn value(&self, index: usize) -> &[u8] {
        let view = &self.views[index];
        let byte_view = ByteView::from(*view);
        let length = byte_view.length as usize;
        if length <= 12 {
            // SAFETY: `view` is a valid inline view with `length` bytes.
            unsafe { BinaryViewArray::inline_value(view, length) }
        } else {
            let buffer_index = byte_view.buffer_index as usize;
            let offset = byte_view.offset as usize;
            if buffer_index < self.completed.len() {
                &self.completed[buffer_index][offset..offset + length]
            } else {
                &self.in_progress[offset..offset + length]
            }
        }
    }

    /// Take all the groups as a single array, leaving `self` empty
    ///
    /// When `keep_capacity` is true, `self` is left with the same capacity,
    /// so the next batches do not grow the buffers again from scratch
    fn take_state(&mut self, keep_capacity: bool) -> ArrayRef {
        let (views_capacity, in_progress_capacity) = if keep_capacity {
            (self.views.capacity(), self.in_progress.capacity())
        } else {
            (0, 0)
        };
        let views = std::mem::replace(&mut self.views, Vec::with_capacity(views_capacity));
        let in_progress = std::mem::replace(
            &mut self.in_progress,
            Vec::with_capacity(in_progress_capacity),
        );
        let mut completed = std::mem::take(&mut self.completed);
        let null_group = self.null_group.take();

        // Flush any remaining in-progress buffer
        if !in_progress.is_empty() {
            completed.push(Buffer::from_vec(in_progress));
        }

        let output_len = views.len();
        let nulls = null_group.map(|null_idx| {
            let mut buffer = NullBufferBuilder::new(output_len);
            buffer.append_n_non_nulls(null_idx);
            buffer.append_null();
            buffer.append_n_non_nulls(output_len - null_idx - 1);
            // NOTE: The inner builder must be constructed as there is at least one null
            buffer.finish().unwrap()
        });

        let views = ScalarBuffer::from(views);
        // SAFETY: the views were built from valid views and our own buffers
        let array = unsafe { BinaryViewArray::new_unchecked(views, completed.into(), nulls) };

        match self.output_type {
            OutputType::BinaryView => Arc::new(array),
            OutputType::Utf8View => {
                // SAFETY: all input was valid utf8
                let array = unsafe { array.to_string_view_unchecked() };
                Arc::new(array)
            }
            OutputType::Utf8 | OutputType::Binary => unreachable!(),
        }
    }
}

/// Returns true if the two valid views of the same input array are equal
///
/// # Safety
///
/// Views that are not inlined must point into `buffers`
#[inline(always)]
unsafe fn input_views_equal(lhs: u128, rhs: u128, buffers: &[Buffer]) -> bool {
    // Same value if inlined, or the same location in the buffers if not
    if lhs == rhs {
        return true;
    }

    // The length and the 4 bytes prefix are the lower 64 bits
    if lhs as u64 != rhs as u64 {
        return false;
    }

    // Inline values with the same length and prefix but different views differ
    if lhs as u32 <= 12 {
        return false;
    }

    // Length and prefix matched - compare the rest of the bytes
    unsafe { non_inlined_value(lhs, buffers)[4..] == non_inlined_value(rhs, buffers)[4..] }
}

/// Returns the bytes `view` points to
///
/// # Safety
///
/// `view` must be a non inlined view that points into `buffers`
#[inline(always)]
unsafe fn non_inlined_value(view: u128, buffers: &[Buffer]) -> &[u8] {
    let view = ByteView::from(view);
    let start = view.offset as usize;
    let end = start + view.length as usize;
    unsafe {
        buffers
            .get_unchecked(view.buffer_index as usize)
            .as_slice()
            .get_unchecked(start..end)
    }
}

impl GroupValues for FullyOrderedGroupValuesBytesView {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);

        let col = &cols[0];

        groups.clear();

        if col.is_empty() {
            return Ok(());
        }

        match self.output_type {
            OutputType::Utf8View => self.intern_inner::<StringViewType>(col, groups),
            OutputType::BinaryView => self.intern_inner::<BinaryViewType>(col, groups),
            OutputType::Utf8 | OutputType::Binary => {
                return internal_err!("wrong output type for FullyOrderedGroupValuesBytesView");
            }
        }

        // ensure we assigned a group to for each row
        assert_eq!(groups.len(), col.len());
        Ok(())
    }

    fn size(&self) -> usize {
        let completed_size = self.completed.allocated_size()
            + self.completed.iter().map(Buffer::capacity).sum::<usize>();

        self.views.allocated_size()
            + self.in_progress.allocated_size()
            + completed_size
            + size_of::<Self>()
    }

    fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    fn len(&self) -> usize {
        self.views.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        if let EmitTo::First(n) = emit_to
            && n > self.len()
        {
            return internal_err!(
                "emit_to::first({n}) is larger than the number of groups {}",
                self.len()
            );
        }

        // Partial emit is followed by more input, so keep the capacity for it
        let state = self.take_state(matches!(emit_to, EmitTo::First(_)));

        let group_values = match emit_to {
            EmitTo::All => state,
            EmitTo::First(n) if n == state.len() => state,
            EmitTo::First(n) => {
                // Insert the rest back, like `GroupValuesBytesView` does.
                // Adjacent groups are distinct, so each remaining value is
                // its own group again and the group indices shift down by `n`
                let emit_group_values = state.slice(0, n);
                let remaining_group_values = state.slice(n, state.len() - n);

                let mut group_indexes = vec![];
                self.intern(&[remaining_group_values], &mut group_indexes)?;

                // Verify that the group indexes were assigned in the correct order
                assert_eq!(self.len(), state.len() - n);

                emit_group_values
            }
        };

        Ok(vec![group_values])
    }

    fn values_preserving(&mut self, _selection: GroupSelection<'_>) -> Result<Vec<ArrayRef>> {
        not_impl_err!("FullyOrderedGroupValuesBytesView does not support values_preserving")
    }

    fn supports_values_preserving(&self) -> bool {
        false
    }

    fn clear_shrink(&mut self, _num_rows: usize) {
        // Callers use this to hand memory back before spilling or sorting, so
        // release every allocation
        self.views = Vec::new();
        self.in_progress = Vec::new();
        self.completed = Vec::new();
        self.null_group = None;
    }
}
