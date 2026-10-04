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

//! [`GroupsAccumulator`] helpers: [`OrderedNullState`] and [`accumulate_indices`]
//!
//! [`GroupsAccumulator`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator

use arrow::array::{Array, BooleanArray, BooleanBufferBuilder, PrimitiveArray};
use arrow::buffer::NullBuffer;
use arrow::datatypes::ArrowPrimitiveType;

use crate::aggregate::groups_accumulator::nulls::filter_to_validity;
use datafusion_common::Result;
use datafusion_expr_common::groups_accumulator::{EmitTo, GroupSelection};
use datafusion_expr_common::ordered_groups_accumulator::{GroupsInfo, PartitionRange, ProcessGroups};

/// If the input has nulls, then the accumulator must potentially
/// handle each input null value specially (e.g. for `SUM` to mark the
/// corresponding sum as null)
///
/// `NullState` tracks if it has seen *any* value for each group when filters or
/// sparse group indices may omit input for a registered group.
#[derive(Debug)]
pub enum SeenValues {
    /// All groups seen so far have seen at least one non-null value
    All {
        num_values: usize,
    },
    // Some groups have not yet seen a non-null value
    Some {
        values: BooleanBufferBuilder,
    },
}

impl Default for SeenValues {
    fn default() -> Self {
        SeenValues::All { num_values: 0 }
    }
}

impl SeenValues {
    /// Return a mutable reference to the `BooleanBufferBuilder` in `SeenValues::Some`.
    ///
    /// If `self` is `SeenValues::All`, it is transitioned to `SeenValues::Some`
    /// by creating a new `BooleanBufferBuilder` where the first `num_values` are true.
    ///
    /// The builder is then ensured to have at least `total_num_groups` length,
    /// with any new entries initialized to false.
    fn get_builder(&mut self, total_num_groups: usize) -> &mut BooleanBufferBuilder {
       match self {
            SeenValues::All { num_values } => {
                let mut builder = BooleanBufferBuilder::new(total_num_groups);
                builder.append_n(*num_values, true);
                if total_num_groups > *num_values {
                    builder.append_n(total_num_groups - *num_values, false);
                }
                *self = SeenValues::Some { values: builder };
                match self {
                    SeenValues::Some { values } => values,
                    _ => unreachable!(),
                }
            }
            SeenValues::Some { values } => {
                if values.len() < total_num_groups {
                    values.append_n(total_num_groups - values.len(), false);
                }
                values
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            SeenValues::All { num_values } => *num_values,
            SeenValues::Some { values } => values.len(),
        }
    }
}
//
// /// Returns true when all newly registered groups are present in `group_indices`.
// ///
// /// Group indices are assigned in first-seen order, so an unfiltered batch visits
// /// new groups in ascending order. Pre-filtered input can omit a new group, making
// /// the indices sparse even though the accumulator no longer receives a filter.
// fn new_groups_are_dense(
//     groups: &GroupsInfo,
//     first_new_group: usize,
//     total_num_groups: usize,
// ) -> bool {
//     if groups.groups()[0].is_same_as_before {
//         return true;
//     }
//     if first_new_group == total_num_groups {
//         return true;
//     }
//
//     let mut next_new_group = first_new_group;
//     for &group_index in group_indices {
//         if group_index == next_new_group {
//             next_new_group += 1;
//         } else if group_index > next_new_group {
//             return false;
//         }
//     }
//     next_new_group == total_num_groups
// }

/// Track the accumulator null state per row: if any values for that
/// group were null and if any values have been seen at all for that group.
///
/// This is part of the inner loop for many [`GroupsAccumulator`]s,
/// and thus the performance is critical and so there are multiple
/// specialized implementations, invoked depending on the specific
/// combinations of the input.
///
/// Typically there are 4 potential combinations of inputs must be
/// special cased for performance:
///
/// * With / Without filter
/// * With / Without nulls in the input
///
/// If the input has nulls, then the accumulator must potentially
/// handle each input null value specially (e.g. for `SUM` to mark the
/// corresponding sum as null)
///
/// `NullState` tracks if it has seen *any* value for each group when filters or
/// sparse group indices may omit input for a registered group.
///
/// [`GroupsAccumulator`]: datafusion_expr_common::groups_accumulator::GroupsAccumulator
#[derive(Debug)]
pub struct OrderedNullState {
    /// Have we seen any non-filtered input values for `group_index`?
    ///
    /// If `seen_values` is `SeenValues::Some(buffer)` and buffer\[i\] is true, have seen at least one non null
    /// value for group `i`
    ///
    /// If `seen_values` is `SeenValues::Some(buffer)` and buffer\[i\] is false, have not seen any values that
    /// pass the filter yet for group `i`
    ///
    /// If `seen_values` is `SeenValues::All`, all groups have seen at least one non null value
    seen_values: SeenValues,

    /// If the current group saw any non-null values that pass the filter
    current_group_have_non_null: bool
}

impl Default for OrderedNullState {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderedNullState {
    pub fn new() -> Self {
        Self {
            seen_values: SeenValues::All { num_values: 0 },
            current_group_have_non_null: false,
        }
    }

    /// return the size of all buffers allocated by this null state, not including self
    pub fn size(&self) -> usize {
        match &self.seen_values {
            SeenValues::All { .. } => 0,
            SeenValues::Some { values } => values.capacity() / 8,
        }
    }


    /// Invokes `value_fn(group_index, value)` for each non null, non
    /// filtered value of `value`, while tracking which groups have
    /// seen null inputs and which groups have seen any inputs if necessary
    //
    /// # Arguments:
    ///
    /// * `values`: the input arguments to the accumulator
    /// * `group_indices`:  To which groups do the rows in `values` belong, (aka group_index)
    /// * `opt_filter`: if present, only rows for which is Some(true) are included
    /// * `value_fn`: function invoked for  (group_index, value) where value is non null
    ///
    /// See [`crate::aggregate::groups_accumulator::accumulate::accumulate`], for more details on how value_fn is called
    ///
    /// When value_fn is called it also sets
    ///
    /// 1. `self.seen_values[group_index]` to true for all rows that had a non null value
    pub fn accumulate<T, F, Processor: ProcessGroups>(
        &mut self,
        group_indices: &GroupsInfo,
        values: &PrimitiveArray<T>,
        opt_filter: Option<&BooleanArray>,
        processor: &mut Processor
    ) where
      T: ArrowPrimitiveType + Send,
    {
        struct NullProcessorWrapper<'a, Processor: ProcessGroups> {
            seen_values: &'a mut BooleanBufferBuilder,
            inner: &'a mut Processor,
        }

        impl<'a, Processor: ProcessGroups> ProcessGroups for NullProcessorWrapper<'a, Processor> {
            fn reserve_for_n_new_groups(&mut self, n: usize) {
                self.inner.reserve_for_n_new_groups(n);
            }

            fn on_new_standalone_group(&mut self, group: &PartitionRange, group_index: usize) {
                self.seen_values.set_bit(group_index, true);
                self.inner.on_new_standalone_group(group, group_index);
            }

            fn on_new_single_item_group(&mut self, row_index: usize, group_index: usize) {
                self.seen_values.set_bit(group_index, true);
                self.inner.on_new_single_item_group(row_index, group_index);
            }

            fn flush_in_progress_group(&mut self, opt_group: Option<&PartitionRange>, group_index: usize) {
                // Only If adding more values
                if let Some(group) = opt_group {
                    self.seen_values.set_bit(group_index, true);
                }
                self.inner.flush_in_progress_group(opt_group, group_index);
            }

            fn add_last_group(&mut self, last_group: &PartitionRange, group_index: usize) {
                self.seen_values.set_bit(group_index, true);
                self.inner.add_last_group(last_group, group_index);
            }

            fn fallback(&mut self, row_index: usize, group_index: usize) {
                self.seen_values.set_bit(group_index, true);
                self.inner.fallback(row_index, group_index);
            }
        }

        let new_groups = group_indices.num_new_groups();
        // Skip per-value null handling when every input value is valid and all
        // newly registered groups are represented. Pre-filtered inputs can have
        // sparse group indices despite not passing a filter to the accumulator.
        if opt_filter.is_none()
          && values.null_count() == 0
          && let SeenValues::All { num_values } = &mut self.seen_values
          // TODO - add back?
          // && new_groups_are_dense(group_indices, *num_values, total_num_groups)
        {
            group_indices.process(None, None, processor).unwrap();
            *num_values += new_groups;
            return;
        }

        let seen_values = self.seen_values.get_builder(new_groups);
        group_indices.process(values.nulls(), opt_filter, &mut NullProcessorWrapper {
            seen_values,
            inner: processor,
        }).unwrap();
    }

    /// Invokes `value_fn(group_index, value)` for each non null, non
    /// filtered value in `values`, while tracking which groups have
    /// seen null inputs and which groups have seen any inputs, for
    /// [`BooleanArray`]s.
    ///
    /// Since `BooleanArray` is not a [`PrimitiveArray`] it must be
    /// handled specially.
    ///
    /// See [`Self::accumulate`], which handles `PrimitiveArray`s, for
    /// more details on other arguments.
    pub fn accumulate_boolean<F>(
        &mut self,
        group_indices: &[usize],
        values: &BooleanArray,
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
        mut value_fn: F,
    ) where
        F: FnMut(usize, bool) + Send,
    {
        let data = values.values();
        assert_eq!(data.len(), group_indices.len());

        // Skip per-value null handling when every input value is valid and all
        // newly registered groups are represented. Pre-filtered inputs can have
        // sparse group indices despite not passing a filter to the accumulator.
        if opt_filter.is_none()
            && values.null_count() == 0
            && let SeenValues::All { num_values } = &mut self.seen_values
          // TODO - add back?
            // && new_groups_are_dense(group_indices, *num_values, total_num_groups)
        {
            group_indices
                .iter()
                .zip(data.iter())
                .for_each(|(&group_index, new_value)| value_fn(group_index, new_value));
            *num_values = total_num_groups;

            return;
        }

        let seen_values = self.seen_values.get_builder(total_num_groups);

        // These could be made more performant by iterating in chunks of 64 bits at a time
        match (values.null_count() > 0, opt_filter) {
            // no nulls, no filter,
            (false, None) => {
                // if we have previously seen nulls, ensure the null
                // buffer is big enough (start everything at valid)
                group_indices.iter().zip(data.iter()).for_each(
                    |(&group_index, new_value)| {
                        seen_values.set_bit(group_index, true);
                        value_fn(group_index, new_value)
                    },
                )
            }
            // nulls, no filter
            (true, None) => {
                let nulls = values.nulls().unwrap();
                group_indices
                    .iter()
                    .zip(data.iter())
                    .zip(nulls.iter())
                    .for_each(|((&group_index, new_value), is_valid)| {
                        if is_valid {
                            seen_values.set_bit(group_index, true);
                            value_fn(group_index, new_value);
                        }
                    })
            }
            // no nulls, but a filter
            (false, Some(filter)) => {
                assert_eq!(filter.len(), group_indices.len());

                group_indices
                    .iter()
                    .zip(data.iter())
                    .zip(filter.iter())
                    .for_each(|((&group_index, new_value), filter_value)| {
                        if filter_value == Some(true) {
                            seen_values.set_bit(group_index, true);
                            value_fn(group_index, new_value);
                        }
                    })
            }
            // both null values and filters
            (true, Some(filter)) => {
                assert_eq!(filter.len(), group_indices.len());
                filter
                    .iter()
                    .zip(group_indices.iter())
                    .zip(values.iter())
                    .for_each(|((filter_value, &group_index), new_value)| {
                        if filter_value == Some(true)
                            && let Some(new_value) = new_value
                        {
                            seen_values.set_bit(group_index, true);
                            value_fn(group_index, new_value)
                        }
                    })
            }
        }
    }

    /// Creates the a [`NullBuffer`] representing which group_indices
    /// should have null values (because they never saw any values)
    /// for the `emit_to` rows.
    ///
    /// resets the internal state appropriately
    pub fn build(&mut self, emit_to: EmitTo) -> Option<NullBuffer> {
        let emit_to = match emit_to {
            EmitTo::All => EmitTo::All,
            EmitTo::First(n) if n == self.seen_values.len() + 1 => EmitTo::All,
            EmitTo::First(n) => EmitTo::First(n),
        };
        let seen_values_len = self.seen_values.len();
        match emit_to {
            EmitTo::All => {
                let mut old_seen = std::mem::take(&mut self.seen_values);
                let current_value_have_non_null = self.current_group_have_non_null;
                self.current_group_have_non_null = false;
                match (old_seen, current_value_have_non_null) {
                    (SeenValues::All { .. }, false) => None,
                    (SeenValues::All { .. }, true) => {
                        let mut builder = BooleanBufferBuilder::new(seen_values_len + 1);
                        // add valid
                        builder.append_n(seen_values_len, true);
                        // append the last null
                        builder.append(false);
                        Some(NullBuffer::new(builder.build()))
                    },
                    (SeenValues::Some { mut values }, have_non_null) => {
                        values.append(have_non_null);
                        Some(NullBuffer::new(values.finish()))
                    }
                }
            }
            EmitTo::First(n) => match &mut self.seen_values {
                SeenValues::All { num_values } => {
                    *num_values = num_values.saturating_sub(n);
                    None
                }

                SeenValues::Some { .. } if n == seen_values_len => {
                    let SeenValues::Some {
                        values: mut old_values,
                    } = std::mem::take(&mut self.seen_values)
                    else {
                        unreachable!()
                    };
                    let nulls = old_values.build();
                    Some(NullBuffer::new(nulls))
                }
                SeenValues::Some { .. } => {
                    assert!(n < seen_values_len);
                    let SeenValues::Some {
                        values: mut old_values,
                    } = std::mem::take(&mut self.seen_values)
                    else {
                        unreachable!()
                    };
                    let nulls = old_values.finish();
                    let first_n_null = nulls.slice(0, n);
                    let remainder = nulls.slice(n, nulls.len() - n);
                    let mut new_builder = BooleanBufferBuilder::new(remainder.len());
                    new_builder.append_buffer(&remainder);
                    self.seen_values = SeenValues::Some {
                        values: new_builder,
                    };
                    Some(NullBuffer::new(first_n_null))
                }
            },
        }
    }
}
