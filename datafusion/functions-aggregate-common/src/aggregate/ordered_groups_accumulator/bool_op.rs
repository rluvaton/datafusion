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

use std::sync::Arc;

use crate::aggregate::groups_accumulator::accumulate::NullState;
use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, BooleanBufferBuilder};
use arrow::buffer::BooleanBuffer;
use datafusion_common::Result;
use datafusion_expr_common::groups_accumulator::EmitTo;
use datafusion_expr_common::ordered_groups_accumulator::{
    GroupsInfo, OrderedGroupsAccumulator, PartitionRange, ProcessGroups,
};

/// An [`OrderedGroupsAccumulator`] that implements a single operation over a
/// [`BooleanArray`] where the accumulated state is also boolean (such
/// as [`BitAndAssign`])
///
/// F: The function to apply to two elements. The first argument is
/// the existing value and the second is the new value, it returns the
/// combined value (e.g. [`BitAndAssign`] style).
///
/// [`BitAndAssign`]: std::ops::BitAndAssign
#[derive(Debug)]
pub struct BooleanOrderedGroupsAccumulator<F>
where
    F: Fn(bool, bool) -> bool + Send + Sync + 'static,
{
    /// values of the groups that are done and ready to be emitted
    ready_values: BooleanBufferBuilder,

    /// value of the current in progress group
    in_progress: bool,

    /// Track nulls in the input / filters
    null_state: NullState,

    /// Function that computes the output
    bool_fn: F,

    /// The identity element for the boolean operation.
    /// Any value combined with this returns the original value.
    identity: bool,
}

impl<F> BooleanOrderedGroupsAccumulator<F>
where
    F: Fn(bool, bool) -> bool + Send + Sync + 'static,
{
    pub fn new(bool_fn: F, identity: bool) -> Self {
        Self {
            ready_values: BooleanBufferBuilder::new(0),
            in_progress: identity,
            null_state: NullState::new(),
            bool_fn,
            identity,
        }
    }

    fn fold(&self, current: bool, values: &BooleanBuffer, group: &PartitionRange) -> bool {
        group
            .as_range()
            .fold(current, |acc, row_index| (self.bool_fn)(acc, values.value(row_index)))
    }
}

impl<F> OrderedGroupsAccumulator for BooleanOrderedGroupsAccumulator<F>
where
    F: Fn(bool, bool) -> bool + Send + Sync + 'static,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        groups: &GroupsInfo,
        opt_filter: Option<&BooleanArray>,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to update_batch");
        let values = values[0].as_boolean();

        struct BoolOpProcessor<'a, F>
        where
            F: Fn(bool, bool) -> bool + Send + Sync + 'static,
        {
            acc: &'a mut BooleanOrderedGroupsAccumulator<F>,
            values: &'a BooleanBuffer,
        }

        impl<F> ProcessGroups for BoolOpProcessor<'_, F>
        where
            F: Fn(bool, bool) -> bool + Send + Sync + 'static,
        {
            fn on_new_standalone_group(&mut self, group: &PartitionRange, _group_index: usize) {
                let value = self.acc.fold(self.acc.identity, self.values, group);
                self.acc.ready_values.append(value);
            }

            fn flush_in_progress_group(
                &mut self,
                opt_group: Option<&PartitionRange>,
                _group_index: usize,
            ) {
                let mut value = self.acc.in_progress;
                if let Some(group) = opt_group {
                    value = self.acc.fold(value, self.values, group);
                }
                self.acc.ready_values.append(value);
                self.acc.in_progress = self.acc.identity;
            }

            fn add_last_group(&mut self, last_group: &PartitionRange, _group_index: usize) {
                self.acc.in_progress =
                    self.acc.fold(self.acc.in_progress, self.values, last_group);
            }

            fn fallback(&mut self, row_index: usize, _group_index: usize) {
                self.acc.in_progress =
                    (self.acc.bool_fn)(self.acc.in_progress, self.values.value(row_index));
            }
        }

        if opt_filter.is_some() || values.null_count() > 0 {
            let group_indices = groups.as_iter_of_group_indices().collect::<Vec<_>>();
            self.null_state.accumulate_boolean(
                &group_indices,
                values,
                opt_filter,
                groups.total_number_of_groups(),
                |_group_index, _new_value| {
                    // values are processed below
                },
            );
        } else {
            self.null_state
                .mark_addition_not_nulls(groups.total_number_of_groups());
        }

        groups.process(
            values.nulls(),
            opt_filter,
            &mut BoolOpProcessor {
                acc: self,
                values: values.values(),
            },
        )
    }

    fn merge_batch(&mut self, values: &[ArrayRef], groups: &GroupsInfo) -> Result<()> {
        // update / merge are the same
        self.update_batch(values, groups, None)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.evaluate(emit_to).map(|arr| vec![arr])
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let nulls = self.null_state.build(emit_to);
        let ready_len = self.ready_values.len();
        let values = match emit_to {
            EmitTo::First(n) if n <= ready_len => {
                let ready = self.ready_values.finish();
                self.ready_values.append_buffer(&ready.slice(n, ready_len - n));
                ready.slice(0, n)
            }
            // Include the in progress group
            EmitTo::All | EmitTo::First(_) => {
                self.ready_values.append(self.in_progress);
                self.in_progress = self.identity;
                self.ready_values.finish()
            }
        };

        Ok(Arc::new(BooleanArray::new(values, nulls)))
    }

    fn size(&self) -> usize {
        // capacity is in bits, so convert to bytes
        self.ready_values.capacity() / 8 + self.null_state.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr_common::ordered_groups_accumulator::GroupsProperties;

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
    fn bool_and_across_batches() -> Result<()> {
        let mut accumulator =
            BooleanOrderedGroupsAccumulator::new(|current, value| current && value, true);

        // groups: 0 = [true, false], 1 = [true] (in progress)
        let values = Arc::new(BooleanArray::from(vec![true, false, true]));
        accumulator.update_batch(&[values], &groups_info(&[(0, 2, false), (2, 3, false)], 0), None)?;

        // group 1 continues with [true, null], group 2 = [false]
        let values = Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)]));
        accumulator.update_batch(&[values], &groups_info(&[(0, 2, true), (2, 3, false)], 1), None)?;

        assert_eq!(
            accumulator.evaluate(EmitTo::First(2))?.as_boolean(),
            &BooleanArray::from(vec![false, true])
        );
        assert_eq!(
            accumulator.evaluate(EmitTo::All)?.as_boolean(),
            &BooleanArray::from(vec![false])
        );
        Ok(())
    }
}
