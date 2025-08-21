use arrow::array::{
  new_empty_array, Array, ArrayRef, AsArray, BooleanArray, GenericListArray, ListArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{ArrowNativeType, FieldRef};
use std::sync::Arc;
use datafusion_common::exec_err;
use datafusion_common::utils::proxy::VecAllocExt;
use datafusion_expr_common::groups_accumulator::GroupsAccumulator;

/// The same type for [`arrow::compute::interleave`]
///
/// Tuple of (input index, index inside that input array)
type StateEntry = (usize, usize);

/// All the indices of inputs array for this group
type AccumulatorState = Vec<StateEntry>;

struct Input {
  array: ArrayRef,

  /// The index of the latest group that using this vector as it's input
  ///
  /// If the value is `negative` it means that the input is not used anymore
  max_group_index: usize,
}

/// Collect List group accumulator
///
/// It will save all the inputs that got from `update_batch` or `merge_batch` after filtering nulls
/// and for `merge_batch` also removing empty lists and unpacking the list to get the underlying values
///
/// and saves in the `states` vector the index of the input and the index of the value in the input
///
/// than on `state`/`evaluate` it will use `arrow::compute::interleave` to build the list
/// from the `states` vector and the `inputs` vector
pub(super) struct CollectListGroupsAccumulator {
  states: Vec<AccumulatorState>,

  /// The sum of each state allocated size (without the states vector itself)
  states_size: usize,

  /// All the inputs that are used in the groups
  ///
  /// If no group point to this input this will be a placeholder (an empty list)
  inputs: Vec<Option<Input>>,

  ready: Vec<ArrayRef>,
  current_group: Option<ArrayAggAccumulator

  /// The size of the arrays in the inputs vector
  /// this is not very accurate as we might count the same underlying buffer multiple times
  /// and not account for the placeholder size
  estimated_inputs_size: usize,

  /// The output list field
  return_field: FieldRef,

  /// Value to put in the inputs vector to avoid shifting the array/holding on input
  /// array more than needed and to be able to release the actual array memory
  input_placeholder: ArrayRef,
}

const DEFAULT_CAPACITY: usize = 128;

impl CollectListGroupsAccumulator {
  pub fn new(return_field: FieldRef) -> Self {
    let input_placeholder = new_empty_array(return_field.data_type());

    Self {
      states: Vec::with_capacity(DEFAULT_CAPACITY),
      states_size: 0,
      inputs: vec![],
      estimated_inputs_size: 0,
      return_field,
      input_placeholder,
    }
  }

  fn add_new_batch(&mut self, array: ArrayRef, max_group_index: usize) {
    // This is not accurate when the multiple inputs point to the same underlying buffer
    self.estimated_inputs_size += array.get_array_memory_size();
    self.inputs.push(Some(Input {
      array,
      max_group_index,
    }));
  }

  /// Convert all groups to a single ListArray
  ///
  /// this is faster than get N groups as we can avoid trying to claim unused inputs
  fn get_all_groups(
    &mut self,
    state_used: &[AccumulatorState],
  ) -> datafusion::common::Result<ListArray> {
    let list = self.build_list(state_used)?;

    // Cleanup:
    self.states.clear();
    self.states.shrink_to(DEFAULT_CAPACITY);
    self.states_size = 0;

    self.reset_inputs();

    Ok(list)
  }

  fn reset_inputs(&mut self) {
    self.inputs.clear();
    self.inputs.shrink_to(DEFAULT_CAPACITY);
    self.estimated_inputs_size = 0;
  }

  /// Get some of the groups
  fn get_n_groups(
    &mut self,
    state_used: &[AccumulatorState],
  ) -> datafusion::common::Result<ListArray> {
    if self.states.is_empty() {
      return self.get_all_groups(state_used);
    }

    self.states_size = self.states_size.saturating_sub(
      state_used.iter().map(|s| s.capacity()).sum::<usize>() * size_of::<StateEntry>(),
    );

    let list = self.build_list(state_used)?;
    self.reclaim_unused_inputs(state_used.len());

    Ok(list)
  }

  #[tracy_gizmos::instrument]
  fn build_list(
    &mut self,
    state_used: &[AccumulatorState],
  ) -> datafusion::common::Result<ListArray> {
    let state_indices = state_used.concat();

    let inputs_dyn = self
      .inputs
      .iter()
      .map(|input| {
        input
          .as_ref()
          .map(|input| input.array.as_ref())
          .unwrap_or(self.input_placeholder.as_ref())
      })
      .collect::<Vec<_>>();

    let values = if inputs_dyn.is_empty() || state_indices.is_empty() {
      new_empty_array(self.return_field.data_type())
    } else {
      arrow::compute::interleave(inputs_dyn.as_slice(), &state_indices)?
    };

    let list_offsets =
      OffsetBuffer::<i32>::from_lengths(state_used.iter().map(|state| state.len()));

    ListArray::try_new(
      Arc::clone(&self.return_field),
      list_offsets,
      values,
      // CollectList is not nullable
      None,
    )
      .map_err(Into::into)
  }

  fn update_inputs_on_new_update_batch(
    &mut self,
    array: ArrayRef,
    mut value_index_and_group_index_iter: impl Iterator<Item = (usize, usize)>,
  ) {
    // The index of the current input in the inputs vector
    let array_index = self.inputs.len();

    let (value_index, mut max_group_index) = value_index_and_group_index_iter.next().unwrap();

    self.states[max_group_index]
      .push_accounted((array_index, value_index), &mut self.states_size);

    for (value_index, group_index) in value_index_and_group_index_iter {
      self.states[group_index]
        .push_accounted((array_index, value_index), &mut self.states_size);
      max_group_index = max_group_index.max(group_index);
    }

    self.add_new_batch(array, max_group_index);
  }

  fn update_inputs_on_new_merge_batch(
    &mut self,
    input_list: &GenericListArray<i32>,
    mut group_index_with_start_and_end_offset_iter: impl Iterator<Item = (usize, (usize, usize))>,
  ) {
    // The index of the current input in the inputs vector
    let array_index = self.inputs.len();

    let (mut max_group_index, (start, end)) =
      group_index_with_start_and_end_offset_iter.next().unwrap();

    {
      let state = &mut self.states[max_group_index];
      let prev_capacity = state.capacity();

      state.extend((start..end).map(|value_index| (array_index, value_index)));

      let changed_capacity = state.capacity() - prev_capacity;
      self.states_size += changed_capacity * size_of::<StateEntry>();
    }

    for (group_index, (start, end)) in group_index_with_start_and_end_offset_iter {
      let state = &mut self.states[group_index];
      let prev_capacity = state.capacity();
      state.extend((start..end).map(|value_index| (array_index, value_index)));
      let changed_capacity = state.capacity() - prev_capacity;
      self.states_size += changed_capacity * size_of::<StateEntry>();

      max_group_index = max_group_index.max(group_index);
    }

    self.add_new_batch(input_list.get_values_sliced(), max_group_index);
  }
}

impl GroupsAccumulator for CollectListGroupsAccumulator {
  fn update_batch(
    &mut self,
    values: &[ArrayRef],
    group_indices: &[usize],
    opt_filter: Option<&BooleanArray>,
    total_num_groups: usize,
  ) -> datafusion_common::Result<()> {

    assert_eq!(values.len(), 1);
    assert_eq!(values[0].len(), group_indices.len());

    if group_indices.is_empty() {
      return Ok(());
    }

    // If filter is none, then we can use fast path
    let Some(filter) = opt_filter else {

      self.update_inputs_on_new_update_batch(
        Arc::clone(&values[0]),
        group_indices.iter().copied().enumerate(),
      );

      return Ok(());
    };

    assert_eq!(filter.null_count(), 0);

    let filter_values = filter.values();
    let filtered = arrow::compute::filter(&values[0], &filter)?;

    if filtered.is_empty() {
      // All values filtered out
      return Ok(());
    }

    let valid_group_indices_iter = group_indices
      .iter()
      .zip(filter_values.iter())
      .filter(|(_, is_valid)| *is_valid)
      .enumerate()
      .map(|(index, (group_index, _))| (index, *group_index));

    self.update_inputs_on_new_update_batch(filtered, valid_group_indices_iter);

    Ok(())
  }

  #[tracy_gizmos::instrument]
  fn evaluate(&mut self, emit_to: EmitTo) -> datafusion::common::Result<ArrayRef> {
    let states_used = emit_to.take_needed(&mut self.states);

    match emit_to {
      EmitTo::All => {
        let result = self.get_all_groups(states_used.as_slice())?;
        Ok(Arc::new(result))
      }
      EmitTo::First(_n) => {
        let result = self.get_n_groups(states_used.as_slice())?;
        Ok(Arc::new(result))
      }
    }
  }

  #[tracy_gizmos::instrument]
  fn state(&mut self, emit_to: EmitTo) -> datafusion::common::Result<Vec<ArrayRef>> {
    let states_used = emit_to.take_needed(&mut self.states);

    match emit_to {
      EmitTo::All => {
        let result = self.get_all_groups(states_used.as_slice())?;
        Ok(vec![Arc::new(result)])
      }
      EmitTo::First(_n) => {
        let result = self.get_n_groups(states_used.as_slice())?;
        Ok(vec![Arc::new(result)])
      }
    }
  }

  #[tracy_gizmos::instrument]
  fn merge_batch(
    &mut self,
    values: &[ArrayRef],
    group_indices: &[usize],
    opt_filter: Option<&BooleanArray>,
    total_num_groups: usize,
  ) -> datafusion::common::Result<()> {
    make_accumulators_if_needed(&mut self.states, total_num_groups);

    assert_eq!(values.len(), 1);
    assert_eq!(values[0].len(), group_indices.len());

    if group_indices.is_empty() {
      return Ok(());
    }

    // Create new filter to filter nulls and merge with the optional filter if it exists
    // When the array is a list, the filter will not filter the nulls in the values
    let opt_filter =
      merge_filter_and_null_removal_and_empty_lists(opt_filter, values[0].as_list());

    let list = values[0].as_list::<i32>();
    let initial_offset = list.offsets()[0].as_usize();

    let non_empty_group_indices_iter =
      group_indices
        .iter()
        .copied()
        .zip(list.offsets().windows(2).map(|window| {
          (
            window[0].as_usize() - initial_offset,
            window[1].as_usize() - initial_offset,
          )
        }));

    // If filter is none, then we can use fast path
    let Some(filter) = opt_filter else {
      self.update_inputs_on_new_merge_batch(list, non_empty_group_indices_iter);

      return Ok(());
    };

    assert_eq!(filter.null_count(), 0);

    let filtered_input = arrow::compute::filter(&values[0], &filter)?;

    // If filtering out all values
    if filter.false_count() == values[0].len() {
      return Ok(());
    }

    let filter_values = filter.values();

    let valid_group_indices_iter = non_empty_group_indices_iter
      .zip(filter_values.iter())
      .filter(|(_, is_valid)| *is_valid)
      // Remove the zipped iterator
      .map(|((group_index, (start, end)), _)| (group_index, (start, end)));

    self.update_inputs_on_new_merge_batch(filtered_input.as_list(), valid_group_indices_iter);

    Ok(())
  }

  fn size(&self) -> usize {
    size_of_val(self)
      + self.states.allocated_size()
      + self.states_size
      + self.inputs.allocated_size()
      + self.estimated_inputs_size
      + self.return_field.size()
  }
}
