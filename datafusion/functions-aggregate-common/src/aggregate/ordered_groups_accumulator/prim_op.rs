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

use std::mem::size_of;
use std::sync::Arc;

use super::accumulate::OrderedNullState;
use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, PrimitiveArray};
use arrow::buffer::NullBuffer;
use arrow::compute;
use arrow::datatypes::ArrowPrimitiveType;
use arrow::datatypes::DataType;
use datafusion_common::{DataFusionError, Result, internal_datafusion_err};
use datafusion_expr_common::ordered_groups_accumulator::{
    AllSingleItemType, GroupsInfo, OrderedGroupsAccumulator,
};

/// An accumulator that implements a single operation over
/// [`ArrowPrimitiveType`] where the accumulated state is the same as
/// the input type (such as `Sum`)
///
/// F: The function to apply to two elements. The first argument is
/// the existing value and should be updated with the second value
/// (e.g. [`BitAndAssign`] style).
///
/// [`BitAndAssign`]: std::ops::BitAndAssign
#[derive(Debug)]
pub struct PrimitiveOrderedGroupsAccumulator<T, F>
where
    T: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, T::Native) + Send + Sync + 'static,
{
    /// the values of the groups already ready to be emitted, stored as the native type
    ready_values: Vec<T::Native>,

    /// The current in progress group
    in_progress: T::Native,

    /// The output type (needed for Decimal precision and scale)
    data_type: DataType,

    /// The starting value for new groups
    starting_value: T::Native,

    /// Track nulls in the input / filters
    null_state: OrderedNullState,

    /// Function that computes the primitive result
    prim_fn: F,
}

impl<T, F> PrimitiveOrderedGroupsAccumulator<T, F>
where
    T: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, T::Native) + Send + Sync + 'static,
{
    pub fn new(data_type: &DataType, prim_fn: F) -> Self {
        Self {
            ready_values: vec![],
            in_progress: T::default_value(),
            data_type: data_type.clone(),
            null_state: OrderedNullState::new(),
            starting_value: T::default_value(),
            prim_fn,
        }
    }

    /// Set the starting values for new groups
    pub fn with_starting_value(mut self, starting_value: T::Native) -> Self {
        self.starting_value = starting_value;
        self.in_progress = starting_value;
        self
    }

    fn update_batch_in_single_item(&mut self, groups: &GroupsInfo, values: &[T::Native]) -> Result<()> {
        assert!(!groups.is_single_group(), "not supporting same group here");
        // Assert that the range of single item groups are all single items while allowing the first and last group to not be
        assert!(groups.are_all_single_item_ignoring_edges());

        // If we started a new group, move the last group to the ready groups
        if !groups.is_first_group_same_as_before() {
            // Push the finished in progress value from last groups
            self.ready_values.push(self.in_progress);
            self.in_progress = self.starting_value;
        }

        let prev_len = self.ready_values.len();

        // Add space for the new groups except the last in progress one
        {
            // - 1 for the last group to be in progress
            self.ready_values.resize(prev_len + groups.groups().len() - 1, self.starting_value);

            // If the first group is the same as the in progress one change the starting value to be the in progress value and not the starting value
            if groups.is_first_group_same_as_before() {
                self.ready_values[prev_len] = self.in_progress;
                self.in_progress = self.starting_value;
            }
        };

        let mut start_process_group = prev_len;

        let should_process_first_group_independently = groups.properties().range_of_single_item_groups.start != 0;
        if should_process_first_group_independently {
            let first_group_value = &mut self.ready_values[prev_len];
            // Update the first group independently
            values[groups.groups()[0].as_range()].iter().for_each(|&new_value| {
                (self.prim_fn)(first_group_value, new_value);
            });

            // Mark the we processed the first group
            start_process_group += 1;
        }

        // If the last group is
        if groups.properties().range_of_single_item_groups.end != groups.groups().len() {
            assert_eq!(groups.properties().range_of_single_item_groups.end, groups.groups().len() - 1, "this function only handle when the last group is either a single item group or not from the end");
        }


        let range_of_values_to_process = {
            let last_group = groups.groups().last().unwrap().as_range();
            let first_group = groups.groups()[0].as_range();

            let starting = if should_process_first_group_independently {
                first_group.end
            } else {
                first_group.start
            };

            starting..last_group.start
        };

        // Process the rest of the items except the last group
        {
            // The range of values to process will limit the ready values slice iterator to not include the last value which is for the in progress group
            values[range_of_values_to_process]
              .iter()
              .zip(
                  &mut self.ready_values.as_mut_slice()[start_process_group..]
                    .iter_mut(),
              )
              .for_each(|(&new_value, current_value)| {
                  (self.prim_fn)(current_value, new_value);
              });
        }

        // Process the last group
        self.process_last_group(groups, values);

        Ok(())
    }

    fn process_last_group(&mut self, groups: &GroupsInfo, values: &[T::Native]) {
        let last_group = &groups.groups().last().expect("must have at least one group");
        values[last_group.as_range()]
          .iter()
          .for_each(|&new_value| {
              (self.prim_fn)(&mut self.in_progress, new_value);
          });
    }
}

// 3 cases
// All with the same group

impl<T, F> OrderedGroupsAccumulator for PrimitiveOrderedGroupsAccumulator<T, F>
where
    T: ArrowPrimitiveType + Send,
    F: Fn(&mut T::Native, T::Native) + Send + Sync + 'static,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        groups: &GroupsInfo,
        opt_filter: Option<&BooleanArray>,
    ) -> Result<()> {
        assert_eq!(values.len(), 1, "single argument to update_batch");
        let values = values[0].as_primitive::<T>();

        if opt_filter.is_some() {
            panic!("PrimitiveOrderedGroupsAccumulator does not support filters yet");
        }

        if values.null_count() > 0 {
            panic!("PrimitiveOrderedGroupsAccumulator does not support nulls yet");
        }

        let values_as_slice = values.values().as_ref();

        // update values
        // self.ready_values.resize(groups.total_number_of_groups(), self.starting_value);

        // Optimization: if all the values are in the same group
        if groups.is_single_group() {
            // If we are starting a new group, save the previous group value and reset the in_progress value
            if !groups.is_first_group_same_as_before() {
                self.ready_values.push(self.in_progress);
                self.in_progress = self.starting_value;
            }

            self.process_last_group(groups, values_as_slice);

            return Ok(());
        }

        // Optimization: if all the values are in single item groups we can fast process them
        if groups.are_all_single_item_ignoring_edges() {
            // TODO - handle the case where group does not start from 0 and not ends at the last group? also can it have gaps?
            assert_ne!(
                groups.total_number_of_groups(),
                1,
                "must not have only one group and same as before here since dont want to push the in progress value yet"
            );

            self.update_batch_in_single_item(groups, values_as_slice)?;

            return Ok(());
        }

        // If there are only 2 groups we can optimize by processing 2 slices separately
        if groups.groups().len() == 2 {

            // Save the last group if finished
            if !groups.is_first_group_same_as_before() {
                // if the first group is not the same as before, we want to save the as ready
                // if the first group is the same as before, we want to to make it the base of the in progress value and not the starting value
                self.ready_values.push(self.in_progress);

                // Reset the in progress one
                self.in_progress = self.starting_value;
            }

            // Process the first group
            {
                let first_group = &groups.groups()[0];

                values_as_slice[first_group.start..first_group.end]
                  .iter()
                  .for_each(|&new_value| {
                      (self.prim_fn)(&mut self.in_progress, new_value);
                  });

                self.ready_values.push(self.in_progress);
                // Reset the in progress for the next group
                self.in_progress = self.starting_value;
            }
            self.process_last_group(groups, values_as_slice);

            return Ok(());
        }

        // If starting a new group, save the previous group value and reset the in_progress value
        if !groups.is_first_group_same_as_before() {
            self.ready_values.push(self.in_progress);
            self.in_progress = self.starting_value;
        }

        // Fallback implementation
        // TODO - maybe have distribution in the groups property whether there are more

        let prev_len = self.ready_values.len();
        // - 1 for the last group to be in progress
        self.ready_values.resize(prev_len + groups.groups().len() - 1, self.starting_value);

        // If the first group is not a new group, change the starting value to be the in progress one
        if groups.is_first_group_same_as_before() {
            self.ready_values[prev_len] = self.in_progress;
        };

        groups.as_iter_of_group_indices_with_match(
            values_as_slice,
            // Don't include last group as we process it into the in progress value
            false
        ).for_each(|(group_index, new_value)| {
            // TODO - can optimize to get the next mutable value as it is running instead of using get_unchecked
            // SAFETY: group_index is guaranteed to be in bounds
            let value = unsafe { self.ready_values.get_unchecked_mut(prev_len + group_index) };

            (self.prim_fn)(value, new_value);
        });


        self.process_last_group(groups, values_as_slice);

        Ok(())
    }

    fn evaluate(&mut self, take_in_progress: bool) -> Result<ArrayRef> {
        if take_in_progress {
            // Avoid reserving double by push in case the capacity needed to be increased
            self.ready_values.reserve_exact(1);
            self.ready_values.push(self.in_progress);
            self.in_progress = self.starting_value;
        }

        let values = std::mem::take(&mut self.ready_values);

        let nulls = self.null_state.build(take_in_progress);
        let values = PrimitiveArray::<T>::new(values.into(), nulls) // no copy
            .with_data_type(self.data_type.clone());
        Ok(Arc::new(values))
    }

    fn state(&mut self, take_in_progress: bool) -> Result<Vec<ArrayRef>> {
        self.evaluate(take_in_progress).map(|arr| vec![arr])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        groups: &GroupsInfo,
    ) -> Result<()> {
        // update / merge are the same
        self.update_batch(values, groups, None)
    }

    fn size(&self) -> usize {
        self.ready_values.capacity() * size_of::<T::Native>() + self.null_state.size()
    }
}
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use arrow::array::Int64Array;
//     use arrow::datatypes::Int64Type;
//
//     #[test]
//     fn preserving_reads_do_not_change_accumulator_state() -> Result<()> {
//         let mut accumulator = PrimitiveGroupsAccumulator::<Int64Type, _>::new(
//             &DataType::Int64,
//             |current, value| *current += value,
//         );
//         let values = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
//         accumulator.update_batch(&[values], &[0, 1, 2], None, 4)?;
//
//         let selection = GroupSelection::try_from_indices(&[3, 0, 1, 1], 4)?;
//         let expected = Int64Array::from(vec![None, Some(1), None, None]);
//         for _ in 0..2 {
//             let actual = accumulator.evaluate_preserving(selection)?;
//             assert_eq!(actual.as_primitive::<Int64Type>(), &expected);
//             let state = accumulator.state_preserving(selection)?;
//             assert_eq!(state[0].as_primitive::<Int64Type>(), &expected);
//         }
//
//         // Group indices and unselected state remain valid after repeated reads.
//         let values = Arc::new(Int64Array::from(vec![5, 7]));
//         accumulator.update_batch(&[values], &[1, 3], None, 4)?;
//         let expected = Int64Array::from(vec![Some(1), Some(5), Some(3), Some(7)]);
//         let actual = accumulator.evaluate_preserving(GroupSelection::all(4))?;
//         assert_eq!(actual.as_primitive::<Int64Type>(), &expected);
//
//         // A destructive read still sees all state after preserving reads.
//         let actual = accumulator.evaluate(EmitTo::All)?;
//         assert_eq!(actual.as_primitive::<Int64Type>(), &expected);
//         assert!(accumulator.supports_evaluate_preserving());
//         assert!(accumulator.supports_state_preserving());
//         Ok(())
//     }
// }
