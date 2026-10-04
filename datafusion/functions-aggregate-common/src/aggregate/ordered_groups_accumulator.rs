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

//! Utilities for implementing [`OrderedGroupsAccumulator`]
//! Adapter that makes [`OrderedGroupsAccumulator`] out of [`Accumulator`]

pub mod accumulate;
pub mod bool_op;
pub mod nulls;
pub mod prim_op;

use std::mem::size_of;

use arrow::array::new_empty_array;
use arrow::{
    array::{ArrayRef, BooleanArray},
    compute,
};
use datafusion_common::{Result, ScalarValue, arrow_datafusion_err, internal_err};
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_expr_common::groups_accumulator::EmitTo;
use datafusion_expr_common::ordered_groups_accumulator::{
    GroupsInfo, OrderedGroupsAccumulator,
};

/// Factory creating a new [`Accumulator`] for each group
pub type AccumulatorFactory = dyn Fn() -> Result<Box<dyn Accumulator>> + Send;

/// An adapter that implements [`OrderedGroupsAccumulator`] for any [`Accumulator`]
///
/// While [`Accumulator`] are simpler to implement and can support
/// more general calculations (like retractable window functions),
/// they are not as fast as a specialized `OrderedGroupsAccumulator`. This
/// interface bridges the gap so the group by operator only operates
/// in terms of [`Accumulator`].
///
/// Since the input is sorted by the group keys, the rows of each group are
/// contiguous, so every group is passed to [`Accumulator::update_batch`] as a
/// single slice of the input (without the `take` the unordered adapter needs).
///
/// Only the last group of the input can still receive values, it is kept as
/// the in progress accumulator. The rest of the groups are done and kept
/// until emitted.
pub struct OrderedGroupsAccumulatorAdapter {
    /// Creates the accumulator for each new group
    factory: Box<AccumulatorFactory>,

    /// Accumulators of the groups that are done, stored in group_index order
    ready_groups: Vec<Box<dyn Accumulator>>,

    /// Accumulator of the last group, `None` when there are no groups
    in_progress: Option<Box<dyn Accumulator>>,

    /// Memory used by the accumulators in `ready_groups`, in bytes.
    ///
    /// Note this is incrementally updated with deltas to avoid the
    /// call to size() being a bottleneck when there are many groups.
    allocation_bytes: usize,
}

impl OrderedGroupsAccumulatorAdapter {
    /// Create a new adapter that will create a new [`Accumulator`]
    /// for each group, using the specified factory function
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> Result<Box<dyn Accumulator>> + Send + 'static,
    {
        Self {
            factory: Box::new(factory),
            ready_groups: vec![],
            in_progress: None,
            allocation_bytes: 0,
        }
    }

    /// Move the in progress group (if any) to the ready groups
    fn flush_in_progress(&mut self) {
        if let Some(accumulator) = self.in_progress.take() {
            self.allocation_bytes += accumulator.size();
            self.ready_groups.push(accumulator);
        }
    }

    /// invokes `f(accumulator, values)` for each group in `groups` with the
    /// rows of that group (after applying `opt_filter`)
    fn invoke_per_group<F>(
        &mut self,
        values: &[ArrayRef],
        groups: &GroupsInfo,
        opt_filter: Option<&BooleanArray>,
        f: F,
    ) -> Result<()>
    where
        F: Fn(&mut dyn Accumulator, &[ArrayRef]) -> Result<()>,
    {
        assert_eq!(values[0].len(), groups.total_number_of_rows());

        for group in groups.groups() {
            if !group.is_same_as_before {
                self.flush_in_progress();
                self.in_progress = Some((self.factory)()?);
            }

            let Some(accumulator) = self.in_progress.as_mut() else {
                return internal_err!(
                    "first group continues the previous group but there is no group in progress"
                );
            };

            let values_to_accumulate =
                slice_and_maybe_filter(values, opt_filter, &[group.start, group.end])?;
            f(accumulator.as_mut(), &values_to_accumulate)?;
        }

        Ok(())
    }

    /// Removes the groups to emit
    fn take_groups(&mut self, emit_to: EmitTo) -> Vec<Box<dyn Accumulator>> {
        let emit_in_progress = match emit_to {
            EmitTo::All => true,
            EmitTo::First(n) => n > self.ready_groups.len(),
        };
        if emit_in_progress {
            self.flush_in_progress();
        }

        let groups = emit_to.take_needed(&mut self.ready_groups);
        let emitted_bytes: usize = groups.iter().map(|acc| acc.size()).sum();
        self.allocation_bytes = self.allocation_bytes.saturating_sub(emitted_bytes);
        groups
    }
}

impl OrderedGroupsAccumulator for OrderedGroupsAccumulatorAdapter {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        groups: &GroupsInfo,
        opt_filter: Option<&BooleanArray>,
    ) -> Result<()> {
        self.invoke_per_group(values, groups, opt_filter, |accumulator, values| {
            accumulator.update_batch(values)
        })
    }

    fn merge_batch(&mut self, values: &[ArrayRef], groups: &GroupsInfo) -> Result<()> {
        self.invoke_per_group(values, groups, None, |accumulator, values| {
            accumulator.merge_batch(values)
        })
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let groups = self.take_groups(emit_to);

        if groups.is_empty() {
            // create empty accumulator to get the state types
            let empty_state = (self.factory)()?.state()?;
            return Ok(empty_state
                .into_iter()
                .map(|state_val| new_empty_array(&state_val.data_type()))
                .collect());
        }

        // each accumulator produces a potential vector of values
        // which we need to form into columns
        let mut results: Vec<Vec<ScalarValue>> = vec![];
        for mut accumulator in groups {
            let accumulator_state = accumulator.state()?;
            results.resize_with(accumulator_state.len(), Vec::new);
            for (idx, state_val) in accumulator_state.into_iter().enumerate() {
                results[idx].push(state_val);
            }
        }

        // create an array for each intermediate column
        let arrays = results
            .into_iter()
            .map(ScalarValue::iter_to_array)
            .collect::<Result<Vec<_>>>()?;

        // double check each array has the same length (aka the
        // accumulator was implemented correctly
        if let Some(first_col) = arrays.first() {
            for arr in &arrays {
                assert_eq!(arr.len(), first_col.len())
            }
        }

        Ok(arrays)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let groups = self.take_groups(emit_to);

        if groups.is_empty() {
            // ScalarValue::iter_to_array needs at least one value to infer the
            // output type, so evaluate a temporary empty accumulator.
            let mut accumulator = (self.factory)()?;
            return Ok(ScalarValue::iter_to_array([accumulator.evaluate()?])?.slice(0, 0));
        }

        let results: Vec<ScalarValue> = groups
            .into_iter()
            .map(|mut accumulator| accumulator.evaluate())
            .collect::<Result<_>>()?;

        ScalarValue::iter_to_array(results)
    }

    fn size(&self) -> usize {
        self.allocation_bytes
            + self.ready_groups.capacity() * size_of::<Box<dyn Accumulator>>()
            + self.in_progress.as_ref().map_or(0, |acc| acc.size())
    }
}

// Copied from physical-plan
pub(crate) fn slice_and_maybe_filter(
    aggr_array: &[ArrayRef],
    filter_opt: Option<&BooleanArray>,
    offsets: &[usize],
) -> Result<Vec<ArrayRef>> {
    let (offset, length) = (offsets[0], offsets[1] - offsets[0]);
    let sliced_arrays: Vec<ArrayRef> = aggr_array
        .iter()
        .map(|array| array.slice(offset, length))
        .collect();

    if let Some(f) = filter_opt {
        let filter = f.slice(offset, length);

        sliced_arrays
            .iter()
            .map(|array| {
                compute::filter(&array, &filter).map_err(|e| arrow_datafusion_err!(e))
            })
            .collect()
    } else {
        Ok(sliced_arrays)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::min_max::MaxAccumulator;
    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::{DataType, Int64Type};
    use datafusion_expr_common::ordered_groups_accumulator::{
        GroupsProperties, PartitionRange,
    };

    fn groups_info(ranges: &[(usize, usize, bool)], start_group_index: usize) -> GroupsInfo {
        GroupsInfo::new(
            ranges
                .iter()
                .map(|&(start, end, is_same_as_before)| PartitionRange {
                    start,
                    end,
                    is_same_as_before,
                })
                .collect(),
            GroupsProperties {
                range_of_single_item_groups: 0..0,
            },
            start_group_index,
        )
    }

    #[test]
    fn adapter_accumulates_contiguous_groups_across_batches() -> Result<()> {
        let mut accumulator = OrderedGroupsAccumulatorAdapter::new(|| {
            Ok(Box::new(MaxAccumulator::try_new(&DataType::Int64)?)
                as Box<dyn Accumulator>)
        });

        // groups: 0 = [1, 5], 1 = [2], 2 = [null] (in progress)
        let values = Arc::new(Int64Array::from(vec![Some(1), Some(5), Some(2), None]));
        accumulator.update_batch(
            &[values],
            &groups_info(&[(0, 2, false), (2, 3, false), (3, 4, false)], 0),
            None,
        )?;

        // group 2 continues with [7, 4] (4 is filtered), group 3 = [9]
        let values = Arc::new(Int64Array::from(vec![7, 4, 9]));
        let filter = BooleanArray::from(vec![true, false, true]);
        accumulator.update_batch(
            &[values],
            &groups_info(&[(0, 2, true), (2, 3, false)], 2),
            Some(&filter),
        )?;

        assert_eq!(
            accumulator
                .evaluate(EmitTo::First(3))?
                .as_primitive::<Int64Type>(),
            &Int64Array::from(vec![5, 2, 7])
        );
        assert_eq!(
            accumulator
                .evaluate(EmitTo::All)?
                .as_primitive::<Int64Type>(),
            &Int64Array::from(vec![9])
        );

        let empty = accumulator.evaluate(EmitTo::All)?;
        assert_eq!(empty.data_type(), &DataType::Int64);
        assert!(empty.is_empty());
        Ok(())
    }
}
