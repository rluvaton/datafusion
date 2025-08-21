use super::helpers::{
    make_accumulators_if_needed, merge_filter_and_null_removal,
    CollectSetGroupAccumulatorValues, SingleGroupUniqueIndices,
};
use ahash::RandomState;
use arrow::array::types::UInt64Type;
use arrow::array::AsArray;
use arrow::array::{
    Array, ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, BooleanArray, ListArray,
    PrimitiveArray, UInt64Array,
};
use arrow::buffer::{Buffer, OffsetBuffer};
use arrow::compute;
use arrow::datatypes::FieldRef;
use std::sync::Arc;
use datafusion_common::exec_err;
use datafusion_common::hash_utils::HashValue;

// Using macro to get away with Rust ownership rules
macro_rules! get_hasher {
    ($self: expr, $state: ident) => {
        // Using hash_one as it's the function used by the `create_hashes` function
        |&value_index| unsafe { $self.values.get_unchecked(value_index).hash_one($state) }
    };
}

const DEFAULT_CAPACITY: usize = 128;

pub(crate) struct CountDistinctPrimitiveGroupsAccumulator<
    T: ArrowPrimitiveType<Native: ArrowNativeTypeOp + HashValue + Copy>,
> {
    /// Logically maps values to index in [`Self::values`]
    ///
    /// Uses the raw API of hashbrown to avoid actually storing the
    /// keys (primitive) in the table
    ///
    /// We don't store the hashes as hashing fixed width primitives
    /// is fast enough for this not to benefit performance
    ///
    /// keys: u64 hashes of the primitive values
    /// values: value_index
    map: hashbrown::HashTable<usize>,

    /// All unique values
    values: CollectSetGroupAccumulatorValues<T::Native>,

    /// All groups, each group store the indices of the values that are in that group
    states: Vec<SingleGroupUniqueIndices>,

    /// The sum of all states capacity
    ///
    /// Note this is incrementally updated with deltas to avoid the
    /// call to size() being a bottleneck. DataFusion saw size() being a
    /// bottleneck in earlier implementations when there were many
    /// distinct groups.
    states_capacities: usize,

    /// random state used to generate hashes
    random_state: RandomState,

    /// buffer that stores hash values (reused across batches to save allocations)
    hashes_buffer: Vec<u64>,
}

impl<T: ArrowPrimitiveType<Native: ArrowNativeTypeOp + HashValue + Copy>>
CountDistinctPrimitiveGroupsAccumulator<T>
{
    pub fn new(return_field: FieldRef) -> Self {
        Self {
            map: hashbrown::HashTable::with_capacity(DEFAULT_CAPACITY),
            values: CollectSetGroupAccumulatorValues::with_capacity(DEFAULT_CAPACITY),
            states: Vec::with_capacity(DEFAULT_CAPACITY),
            states_capacities: 0,
            random_state: Default::default(),
            hashes_buffer: Default::default(),
        }
    }

    fn apply_batch_for_all<BatchAdder: AddBatch>(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        make_accumulators_if_needed(&mut self.states, total_num_groups);

        assert_eq!(values.len(), 1);
        assert_eq!(values[0].len(), group_indices.len());

        // Create new filter to filter nulls and merge with the optional filter if it exists
        // When the array is a list, the filter will not filter the nulls in the values
        let opt_filter = if opt_filter.is_none() && values[0].null_count() == 0 {
            opt_filter.cloned()
        } else {
            Some(merge_filter_and_null_removal(opt_filter, &values[0]))
        };

        // If filter is none, then we can use fast path
        if opt_filter.is_none() {
            let underlying_values = BatchAdder::unwarp_values(&values[0]);

            if underlying_values.null_count() != 0 {
                return exec_err!("nulls should be filtered before updating hashes");
            }

            let mappings = self.update_hashes::<BatchAdder>(&values[0])?;

            BatchAdder::add_batch_for_ready_array(
                self,
                BatchAdder::get_offsets(&values[0]),
                // Map current string indexes to the index in the values vector
                mappings.into_iter(),
                group_indices.iter().copied(),
            );

            return Ok(());
        }

        let filter = opt_filter.unwrap();

        // Convert the group indices to u64 so we can use the filter kernel
        let group_indices: UInt64Array = group_indices.iter().map(|&idx| idx as u64).collect();
        let filtered_group_indices = compute::filter(&group_indices, &filter)?;
        let filtered_array = compute::filter(&values[0], &filter)?;

        let underlying_values_filtered = BatchAdder::unwarp_values(&filtered_array);

        if underlying_values_filtered.null_count() != 0 {
            return exec_err!("nulls should be filtered before updating hashes");
        }

        let mappings = self.update_hashes::<BatchAdder>(&filtered_array)?;

        BatchAdder::add_batch_for_ready_array(
            self,
            BatchAdder::get_offsets(&filtered_array),
            // Map current string indexes to the index in the values vector
            mappings.into_iter(),
            filtered_group_indices
              .as_primitive::<UInt64Type>()
              .values()
              .iter()
              .map(|&idx| idx as usize),
        );

        Ok(())
    }

    fn update(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        make_accumulators_if_needed(&mut self.states, total_num_groups);

        assert_eq!(values.len(), 1);
        assert_eq!(values[0].len(), group_indices.len());

        // Create new filter to filter nulls and merge with the optional filter if it exists
        // When the array is a list, the filter will not filter the nulls in the values
        let opt_filter = if opt_filter.is_none() && values[0].null_count() == 0 {
            opt_filter.cloned()
        } else {
            Some(merge_filter_and_null_removal(opt_filter, &values[0]))
        };

        // If filter is none, then we can use fast path
        if opt_filter.is_none() {
            let underlying_values = BatchAdder::unwarp_values(&values[0]);

            if underlying_values.null_count() != 0 {
                return exec_err!("nulls should be filtered before updating hashes");
            }

            let mappings = self.update_hashes::<BatchAdder>(&values[0])?;

            BatchAdder::add_batch_for_ready_array(
                self,
                BatchAdder::get_offsets(&values[0]),
                // Map current string indexes to the index in the values vector
                mappings.into_iter(),
                group_indices.iter().copied(),
            );

            return Ok(());
        }

        let filter = opt_filter.unwrap();

        // Convert the group indices to u64 so we can use the filter kernel
        let group_indices: UInt64Array = group_indices.iter().map(|&idx| idx as u64).collect();
        let filtered_group_indices = compute::filter(&group_indices, &filter)?;
        let filtered_array = compute::filter(&values[0], &filter)?;

        let underlying_values_filtered = BatchAdder::unwarp_values(&filtered_array);

        if underlying_values_filtered.null_count() != 0 {
            return exec_err!("nulls should be filtered before updating hashes");
        }

        let mappings = self.update_hashes::<BatchAdder>(&filtered_array)?;

        BatchAdder::add_batch_for_ready_array(
            self,
            BatchAdder::get_offsets(&filtered_array),
            // Map current string indexes to the index in the values vector
            mappings.into_iter(),
            filtered_group_indices
              .as_primitive::<UInt64Type>()
              .values()
              .iter()
              .map(|&idx| idx as usize),
        );

        Ok(())
    }

    /// Update the hashes and get the mapping from the input array to the index in [`Self::values`]
    fn update_hashes<BatchAdder: AddBatch>(&mut self, value: &ArrayRef) -> Result<Vec<usize>> {
        let unwrapped_value = BatchAdder::unwarp_values(value);

        // Calculate the hashes for the values
        {
            let batch_hashes = &mut self.hashes_buffer;
            batch_hashes.clear();
            batch_hashes.resize(unwrapped_value.len(), 0);

            // If changing this change the single hash function used in `get_hasher!` macro
            create_hashes(
                &[Arc::clone(&unwrapped_value)],
                &self.random_state,
                batch_hashes,
            )?;
        }

        let primitive_values = unwrapped_value.as_primitive::<T>();
        assert_eq!(
            primitive_values.null_count(),
            0,
            "nulls should be filtered before updating hashes"
        );
        let state = &self.random_state;

        // Mapping between index in the current input array  to the index in the values vector
        // mapping[i] = j means that the value at index i in the input array is the same as the value at index j in [`Self::values`]
        let mut mappings: Vec<usize> = vec![0; primitive_values.len()];

        // Fill the map with the new values
        for (array_index, &target_hash) in self.hashes_buffer.iter().enumerate() {
            let value = primitive_values.value(array_index);
            let insert = self.map.entry(
                target_hash,
                |&value_index| unsafe { self.values.get_unchecked(value_index).is_eq(value) },
                get_hasher!(self, state),
            );

            let value_index = match insert {
                // Existing value_index for this value
                hashbrown::hash_table::Entry::Occupied(o) => *o.get(),
                hashbrown::hash_table::Entry::Vacant(v) => {
                    let value_index = self.values.insert_new_unique(value);

                    v.insert(value_index);

                    value_index
                }
            };

            mappings[array_index] = value_index;
        }

        Ok(mappings)
    }

    /// Convert all groups to a single ListArray
    ///
    /// this is faster than get N groups as we can avoid updating ref count
    fn get_all_groups(&mut self, state_used: &[SingleGroupUniqueIndices]) -> Result<ListArray> {
        let list = self.build_list(state_used)?;

        // Cleanup:
        self.values.reset(DEFAULT_CAPACITY);
        self.states.clear();
        self.states_capacities = 0;
        self.states.shrink_to(DEFAULT_CAPACITY);
        self.map.clear();
        let state = &self.random_state;
        self.map
          .shrink_to(DEFAULT_CAPACITY, get_hasher!(self, state));

        Ok(list)
    }

    /// Get some of the groups
    fn get_n_groups(&mut self, state_used: &[SingleGroupUniqueIndices]) -> Result<ListArray> {
        if self.states.is_empty() {
            return self.get_all_groups(state_used);
        }

        self.states_capacities -= state_used
          .iter()
          .map(|state| state.capacity())
          .sum::<usize>();

        let list = self.build_list(state_used)?;

        let available_indices_length = self.values.get_number_of_available_indices();

        // Update the ref count for the values that are used in the group
        for value_index in state_used.iter().flat_map(|state| state.indices()) {
            self.values.decrement_ref_count(value_index);
        }

        // This means that we now have more available indices, so we should remove those values from the map
        if available_indices_length != self.values.get_number_of_available_indices() {
            // Keep only the items that are still in use
            self.map.retain(|index| self.values.is_used(*index));
        }

        Ok(list)
    }

    fn build_list(&mut self, state_used: &[SingleGroupUniqueIndices]) -> Result<ListArray> {
        let total_number_of_items = state_used.iter().map(|state| state.len()).sum::<usize>();

        let primitive_values_buffer = unsafe {
            let buffer = Buffer::from_trusted_len_iter::<T::Native, _>(
                state_used
                  .iter()
                  .flat_map(|state| state.indices())
                  // Safe as all indices are valid
                  .map(|index| *self.values.get_unchecked(index))
                  // This is what make this iterator trusted length (have upper bound)
                  .take(total_number_of_items),
            );

            buffer.into()
        };

        let list_offsets =
          OffsetBuffer::<i32>::from_lengths(state_used.iter().map(|index| index.len()));

        let primitive_values = PrimitiveArray::<T>::new(
            primitive_values_buffer,
            // CollectSet is not nullable
            None,
        )
          .with_data_type(self.return_field.data_type().clone());

        ListArray::try_new(
            Arc::clone(&self.return_field),
            list_offsets,
            Arc::new(primitive_values),
            // CollectSet is not nullable
            None,
        )
          .map_err(Into::into)
    }
}

impl<T: ArrowPrimitiveType<Native: ArrowNativeTypeOp + HashValue + Copy>> GroupsAccumulator
for CollectSetPrimitiveGroupsAccumulator<T>
{
    #[tracy_gizmos::instrument]
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.apply_batch_for_all::<UpdateBatchCall>(
            values,
            group_indices,
            opt_filter,
            total_num_groups,
        )
    }

    #[tracy_gizmos::instrument]
    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let state_used = emit_to.take_needed(&mut self.states);

        match emit_to {
            EmitTo::All => {
                let result = self.get_all_groups(state_used.as_slice())?;
                Ok(Arc::new(result))
            }
            EmitTo::First(_n) => {
                let result = self.get_n_groups(state_used.as_slice())?;
                Ok(Arc::new(result))
            }
        }
    }

    #[tracy_gizmos::instrument]
    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let state_used = emit_to.take_needed(&mut self.states);

        match emit_to {
            EmitTo::All => {
                let result = self.get_all_groups(state_used.as_slice())?;
                Ok(vec![Arc::new(result)])
            }
            EmitTo::First(_n) => {
                let result = self.get_n_groups(state_used.as_slice())?;
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
    ) -> Result<()> {
        self.apply_batch_for_all::<MergeBatchCall>(
            values,
            group_indices,
            opt_filter,
            total_num_groups,
        )
    }

    fn size(&self) -> usize {
        size_of_val(self)
          + self.states.allocated_size()
          + self.states_capacities * size_of::<usize>()
          + self.map.capacity() * size_of::<usize>()
          + self.hashes_buffer.allocated_size()
          + self.values.size()
    }

    #[tracy_gizmos::instrument]
    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        convert_to_state(&self.return_field, values, opt_filter)
    }

    fn supports_convert_to_state(&self) -> bool {
        true
    }
}

trait AddBatch {
    /// This function should be called when the values do not need any filtering
    fn add_batch_for_ready_array<
        T: ArrowPrimitiveType<Native: ArrowNativeTypeOp + HashValue + Copy>,
    >(
        group_accumulator: &mut CollectSetPrimitiveGroupsAccumulator<T>,
        offsets: Option<&OffsetBuffer<i32>>,
        value_indices: impl Iterator<Item = usize>,
        group_indices: impl Iterator<Item = usize>,
    );

    fn get_offsets(value: &ArrayRef) -> Option<&OffsetBuffer<i32>>;

    fn unwarp_values(value: &ArrayRef) -> ArrayRef;
}

struct UpdateBatchCall;

impl AddBatch for UpdateBatchCall {
    #[tracy_gizmos::instrument]
    fn add_batch_for_ready_array<
        T: ArrowPrimitiveType<Native: ArrowNativeTypeOp + HashValue + Copy>,
    >(
        group_accumulator: &mut CollectSetPrimitiveGroupsAccumulator<T>,
        offsets: Option<&OffsetBuffer<i32>>,
        value_indices: impl Iterator<Item = usize>,
        group_indices: impl Iterator<Item = usize>,
    ) {
        assert!(offsets.is_none(), "offsets should be None for update batch");
        for (group_index, value_index) in group_indices.zip(value_indices) {
            // Add to the appropriate group
            let inserted = group_accumulator.states[group_index]
              .insert_accounted(value_index, &mut group_accumulator.states_capacities)
              as usize;

            // Add 1 to the ref count if the value is new in that group
            group_accumulator
              .values
              .increment_ref_count_by(value_index, inserted)
        }
    }

    fn get_offsets(_value: &ArrayRef) -> Option<&OffsetBuffer<i32>> {
        None
    }

    fn unwarp_values(value: &ArrayRef) -> ArrayRef {
        Arc::clone(value)
    }
}

struct MergeBatchCall;

impl AddBatch for MergeBatchCall {
    #[tracy_gizmos::instrument]
    fn add_batch_for_ready_array<
        T: ArrowPrimitiveType<Native: ArrowNativeTypeOp + HashValue + Copy>,
    >(
        group_accumulator: &mut CollectSetPrimitiveGroupsAccumulator<T>,
        offsets: Option<&OffsetBuffer<i32>>,
        mut value_indices: impl Iterator<Item = usize>,
        group_indices: impl Iterator<Item = usize>,
    ) {
        let offsets = offsets.expect("offsets should exists for merge batch");

        let list_lengths = offsets.windows(2).map(|offsets| offsets[1] - offsets[0]);

        // Add to the appropriate group the row
        for (group_index, length) in group_indices.zip(list_lengths) {
            let state = &mut group_accumulator.states[group_index];
            let prev_capacity = state.capacity();
            for value_index in value_indices.by_ref().take(length as usize) {
                // Add to the appropriate group
                let inserted = state.insert(value_index);

                // Add 1 to the ref count if the value is new in that group
                group_accumulator
                  .values
                  .increment_ref_count_by(value_index, inserted as usize);
            }

            group_accumulator.states_capacities += state.capacity() - prev_capacity;
        }
    }

    fn get_offsets(value: &ArrayRef) -> Option<&OffsetBuffer<i32>> {
        Some(value.as_list::<i32>().offsets())
    }

    fn unwarp_values(value: &ArrayRef) -> ArrayRef {
        let list = value.as_list::<i32>();
        list.get_values_sliced()
    }
}
