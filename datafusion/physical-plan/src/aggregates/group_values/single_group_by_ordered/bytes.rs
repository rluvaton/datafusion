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

use crate::aggregates::group_values::{GroupValues, HashValue};
use std::mem::size_of;
use std::ops::SubAssign;
use std::sync::Arc;

use arrow::array::{
  Array, ArrayRef, AsArray, GenericBinaryArray, GenericStringArray,
  NullBufferBuilder, OffsetSizeTrait,
};
use arrow::buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use datafusion_common::utils::proxy::VecAllocExt;
use datafusion_common::utils::split_vec_min_alloc;
use datafusion_common::{Result, internal_err, not_impl_err};
use datafusion_expr::{EmitTo, GroupSelection};
use datafusion_physical_expr_common::binary_map::{
  INITIAL_BUFFER_CAPACITY, OutputType,
};

/// A [`GroupValues`] storing single column of Utf8/LargeUtf8/Binary/LargeBinary values
///
/// This specialization is significantly faster than using the more general
/// purpose `Row`s format
pub struct FullyOrderedGroupValuesBytes<O: OffsetSizeTrait> {
    /// Should the output be String or Binary?
    output_type: OutputType,
    /// In progress buffer containing all values
    buffer: Vec<u8>,
    /// Offsets into `buffer` for each distinct  value. These offsets as used
    /// directly to create the final `GenericBinaryArray`. The `i`th string is
    /// stored in the range `offsets[i]..offsets[i+1]` in `buffer`. Null values
    /// are stored as a zero length string.
    offsets: Vec<O>,
    last_offset: O,

    /// The group index of the null value if any
    null_group: Option<usize>,
}

impl<O: OffsetSizeTrait + SubAssign> FullyOrderedGroupValuesBytes<O> {
    pub fn new(output_type: OutputType) -> Self {
        assert!(
            matches!(output_type, OutputType::Utf8 | OutputType::Binary),
            "FullyOrderedGroupValuesBytes only supports Utf8 or Binary output types"
        );
        Self {
            // One map holds every group value for the whole query, so it is
            // worth pre-allocating the hash table and the value buffer.
            buffer: Vec::with_capacity(INITIAL_BUFFER_CAPACITY),
            offsets: vec![O::zero()],
            last_offset: O::zero(),
            null_group: None,
            output_type,
        }
    }

    fn current_group(&self) -> Option<(Option<&[u8]>, usize)> {
        if self.is_empty() {
            return None;
        }

        let current_group_index = self.len() - 1;
        let current_value = if self.null_group.is_some_and(|n| n == current_group_index) {
            None
        } else {
            let start = self.offsets[current_group_index];
            let end = self.offsets[current_group_index + 1];
            Some(&self.buffer[start.as_usize()..end.as_usize()])
        };
        Some((current_value, current_group_index))
    }

    fn handle_valid<'a>(
        &mut self,
        groups: &mut Vec<usize>,
        mut current_value_valid: &'a [u8],
        mut current_group: usize,
        offsets_slice: &[O],
        buffer_slice: &'a [u8],
    ) -> usize {
        for window in offsets_slice.windows(2) {
            let start = window[0];
            let end = window[1];
            let v: &[u8] = &buffer_slice[start.as_usize()..end.as_usize()];

            // If new group, save the current group and start a new one
            if v != current_value_valid {
                current_group += 1;
                current_value_valid = v;
                // TODO - CHECK IF OVERFLOW
                self.last_offset += O::usize_as(current_value_valid.len());
                self.offsets.push(self.last_offset);
                self.buffer.extend_from_slice(current_value_valid);
            }

            groups.push(current_group);
        }

        current_group
    }

    fn build_array_from_state(
        &self,
        offsets: Vec<O>,
        buffer: Vec<u8>,
        null_group: Option<usize>,
    ) -> ArrayRef {
        let output_len = offsets.len() - 1;

        let offsets = ScalarBuffer::from(offsets);
        // SAFETY: The offsets are guaranteed to be valid because they were built from the buffer
        let offsets = unsafe { OffsetBuffer::new_unchecked(offsets) };

        let buffer = Buffer::from_vec(buffer);
        let nulls = null_group.map(|null_idx| {
            let mut buffer = NullBufferBuilder::new(output_len);
            buffer.append_n_non_nulls(null_idx);
            buffer.append_null();
            buffer.append_n_non_nulls(output_len - null_idx - 1);
            // NOTE: The inner builder must be constructed as there is at least one null
            buffer.finish().unwrap()
        });
        let array: ArrayRef = match self.output_type {
            OutputType::Utf8 => {
                let arr = unsafe {
                    // SAFETY: the input is valid since it came from valid arrays
                    GenericStringArray::<O>::new_unchecked(offsets, buffer, nulls)
                };

                Arc::new(arr) as ArrayRef
            }
            OutputType::Binary => {
                let arr = unsafe {
                    // SAFETY: the input is valid since it came from valid arrays
                    GenericBinaryArray::<O>::new_unchecked(offsets, buffer, nulls)
                };

                Arc::new(arr) as ArrayRef
            }
            OutputType::Utf8View | OutputType::BinaryView => unreachable!(),
        };
        array
    }
}

impl<O: OffsetSizeTrait + SubAssign> GroupValues for FullyOrderedGroupValuesBytes<O> {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);

        // look up / add entries in the table
        let col = &cols[0];

        groups.clear();

        if col.is_empty() {
            return Ok(());
        }

        let (col_offsets, col_buffer) = match self.output_type {
            OutputType::Utf8 => {
                let col = col.as_string::<O>();
                (
                    col.offsets(),
                    col.values().as_slice(),
                )
            }
            OutputType::Binary => {
                let col = col.as_binary::<O>();
                (
                    col.offsets(),
                    col.values().as_slice(),
                )
            }
            OutputType::Utf8View | OutputType::BinaryView => {
                return internal_err!(
                    "wrong output type for FullyOrderedGroupValuesBytes"
                );
            }
        };

        let mut current_value: Option<&[u8]> = if col.is_null(0) {
            None
        } else {
            let start = col_offsets[0];
            let end = col_offsets[1];
            Some(&col_buffer[start.as_usize()..end.as_usize()])
        };

        // Handle the first value in the column
        if let Some((last_group_value, last_index)) = self.current_group() {
            match (last_group_value, current_value) {
                (last_value, current_value) if last_value == current_value => {
                    // Nothing
                }
                (Some(_), None) => {
                    self.null_group = Some(last_index + 1);
                    self.offsets.push(self.last_offset);
                }
                (_, Some(val)) => {
                    // TODO - CHECK IF OVERFLOW
                    self.last_offset += O::usize_as(val.len());
                    self.offsets.push(self.last_offset);
                    self.buffer.extend_from_slice(val);
                }
                _ => unreachable!(),
            }
        } else {
            // On first batch
            match current_value {
                Some(value) => {
                    // TODO - CHECK IF OVERFLOW
                    self.last_offset = O::usize_as(value.len());
                    self.offsets.push(self.last_offset);
                    self.buffer.extend_from_slice(&value);
                }
                None => {
                    self.null_group = Some(0);
                    self.offsets.push(self.last_offset);
                }
            }
        }

        let mut current_group = self.len() - 1;

        match (current_value, col.null_count()) {
            // If current group is null and the entire column is null
            (None, null_count) if null_count == col.len() => {
                // All groups with the same current group
                groups.resize(col.len(), current_group);
            }

            // If current group is null and there may be nulls or not, but not all are nulls
            (None, null_count) => {
                // Add the nulls to the current group
                groups.resize(null_count, current_group);

                assert!(
                    !col.is_null(null_count),
                    "input is ordered, so once null was seen all nulls should be at the beginning of the column"
                );
                debug_assert_eq!(
                    col.slice(0, null_count).null_count(),
                    null_count,
                    "input is ordered, so once null was seen all nulls should be at the beginning of the column"
                );

                current_group += 1;

                let current_value_valid: &[u8] = {
                    let start = col_offsets[null_count];
                    let end = col_offsets[null_count + 1];
                    &col_buffer[start.as_usize()..end.as_usize()]
                };

                // TODO - CHECK IF OVERFLOW
                self.last_offset += O::usize_as(current_value_valid.len());
                self.offsets.push(self.last_offset);
                self.buffer.extend_from_slice(current_value_valid);

                let offsets_slice = &col_offsets[null_count..];
                let buffers_slice = &col_buffer[offsets_slice[0].as_usize()..];

                self.handle_valid(
                    groups,
                    current_value_valid,
                    current_group,
                    offsets_slice,
                    buffers_slice,
                );
            }

            // If current value is valid
            (Some(current_value_valid), null_count) => {
                let values_without_nulls = col.len() - null_count;

                debug_assert_eq!(
                    col.slice(values_without_nulls, null_count).null_count(),
                    null_count,
                    "input is ordered, so once null was seen after non nulls, all nulls should be at the end of the column"
                );

                let offsets_slice = &col_offsets[..values_without_nulls + 1];

                self.handle_valid(
                    groups,
                    current_value_valid,
                    current_group,
                    offsets_slice,
                    col_buffer,
                );

                // If there are nulls
                if null_count > 0 {
                    current_group += 1;
                    self.offsets.push(self.last_offset);
                    self.null_group = Some(current_group);

                    groups.resize(col.len(), current_group);
                }
            }
        }

        // ensure we assigned a group to for each row
        assert_eq!(groups.len(), col.len());
        Ok(())
    }

    fn size(&self) -> usize {
        self.offsets.allocated_size() + self.buffer.allocated_size() + size_of::<Self>()
    }

    fn is_empty(&self) -> bool {
        self.offsets.len() == 1
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let (offsets, buffer, null_group) = match emit_to {
            EmitTo::All => {
                self.last_offset = O::zero();
                let offsets =
                    std::mem::replace(&mut self.offsets, vec![self.last_offset]);
                let buffer = std::mem::replace(
                    &mut self.buffer,
                    Vec::with_capacity(INITIAL_BUFFER_CAPACITY),
                );
                let null_group = self.null_group.take();

                (offsets, buffer, null_group)
            }
            EmitTo::First(n) => {
                if n + 1 > self.offsets.len() {
                    return internal_err!(
                        "emit_to::first({n}) is larger than the number of groups {}",
                        self.offsets.len() - 1
                    );
                }
                // TODO - null group
                let output_offsets = {
                    let mut output_offsets = Vec::with_capacity(n + 1);
                    // SAFETY: this is safe as we just allocated with this capacity
                    // We validated we have enough data here
                    unsafe {
                        output_offsets.set_len(n + 1);
                    };

                    output_offsets.copy_from_slice(&self.offsets[..=n]);
                    // Move the offsets to the start
                    self.offsets.copy_within(n.., 0);
                    // Shrink capacity to the new length, so we don't hold on to memory we don't need
                    self.offsets.shrink_to_fit();

                    // Shift the offset to start from 0
                    let start_offset = output_offsets[0];
                    self.offsets.iter_mut().for_each(|o| *o -= start_offset);
                    self.last_offset = self.offsets.last().copied().unwrap();

                    output_offsets
                };

                let output_buffer = split_vec_min_alloc(
                    &mut self.buffer,
                    output_offsets.last().copied().unwrap().as_usize(),
                );
                let null_group = match &mut self.null_group {
                    Some(v) if *v >= n => {
                        *v -= n;
                        None
                    }
                    Some(_) => self.null_group.take(),
                    None => None,
                };

                (output_offsets, output_buffer, null_group)
            }
        };

        let output = self.build_array_from_state(offsets, buffer, null_group);

        Ok(vec![output])
    }

    fn values_preserving(
        &mut self,
        _selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        not_impl_err!(
            "FullyOrderedGroupValuesBytes does not support values_preserving since there are no guarantees that indices are not duplicate and not sorted in order or something"
        )
    }

    fn supports_values_preserving(&self) -> bool {
        false
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.last_offset = O::zero();

        self.offsets.clear();
        self.offsets.shrink_to(num_rows + 1);
        self.offsets.push(self.last_offset);

        self.buffer.clear();
        self.buffer.shrink_to(INITIAL_BUFFER_CAPACITY);
    }
}

#[cfg(test)]
mod tests {
  use super::*;

  use std::sync::Arc;

  use arrow::array::StringArray;
  use datafusion_physical_expr_common::binary_map::INITIAL_BUFFER_CAPACITY;

  /// `clear_shrink` is how the aggregate stream hands memory back before it
    /// spills and before the spilled batch is sorted, so the memory it releases
    /// has to actually show up in the size it reports afterwards.
    #[test]
    fn clear_shrink_releases_the_reported_memory() {
        let mut group_values = FullyOrderedGroupValuesBytes::<i32>::new(OutputType::Utf8);
        let empty = size_of::<FullyOrderedGroupValuesBytes<i32>>();

        // The map is pre-allocated at construction, so it is already well above
        // its own struct size before a single row is interned.
        let warm_size = group_values.size();
        assert!(
            warm_size > empty + INITIAL_BUFFER_CAPACITY,
            "expected the pre-allocated map to report more than {} bytes, got {warm_size}",
            empty + INITIAL_BUFFER_CAPACITY
        );

        let values: ArrayRef = Arc::new(StringArray::from_iter_values(
            (0..1_000).map(|i| format!("group value number {i}")),
        ));
        let mut groups = vec![];
        group_values
            .intern(&[Arc::clone(&values)], &mut groups)
            .unwrap();
        let populated_size = group_values.size();
        assert!(populated_size > warm_size);

        group_values.clear_shrink(0);

        // Everything the map held is gone: what remains is the struct itself
        // plus the single leading zero offset.
        let released_size = group_values.size();
        assert!(
            released_size < empty + 128,
            "expected clear_shrink to release the map, got {released_size} with a struct size of {empty}"
        );
        assert!(
            released_size * 10 < populated_size,
            "expected {released_size} to be far below {populated_size}"
        );

        // The map still works, and warms back up on the next emit.
        group_values.intern(&[values], &mut groups).unwrap();
        assert!(group_values.size() > released_size);
        group_values.emit(EmitTo::All).unwrap();
        assert!(group_values.size() > empty + INITIAL_BUFFER_CAPACITY);
    }
}
