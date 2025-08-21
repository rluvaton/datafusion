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

//! [`BytesDistinctCountAccumulator`] for Utf8/LargeUtf8/Binary/LargeBinary values

use arrow::array::{ArrayRef, OffsetSizeTrait};
use datafusion_common::cast::as_list_array;
use datafusion_common::utils::SingleRowListArrayBuilder;
use datafusion_common::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_physical_expr_common::binary_map::{ArrowBytesSet, OutputType};
use datafusion_physical_expr_common::binary_view_map::ArrowBytesViewSet;
use std::fmt::Debug;
use std::mem::size_of_val;

use ahash::RandomState;
use arrow::array::cast::AsArray;
use arrow::array::types::{ByteArrayType, UInt64Type};
use arrow::array::{
    Array, ArrayRef, BooleanArray, GenericByteArray, ListArray, OffsetBufferBuilder, UInt64Array,
};
use arrow::buffer::{Buffer, MutableBuffer, OffsetBuffer};
use arrow::compute;
use arrow::datatypes::{ArrowNativeType, FieldRef};
use std::marker::PhantomData;
use std::sync::Arc;

/// State of a single group
#[derive(Default)]
struct AccumulatorState {
    unique_indices: SingleGroupUniqueIndices,

    /// This is the size of all the values that are pointed by the indices in the unique_indices
    size: usize,
}

impl AccumulatorState {
    fn len(&self) -> usize {
        self.unique_indices.len()
    }

    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.unique_indices.indices()
    }

    /// Insert a new index to the unique indices, if the index is already in the unique indices it will return 0
    /// otherwise it will return 1
    ///
    /// It will also update the size of the accumulator in case the index is inserted
    fn insert(&mut self, index: usize, size: usize) -> usize {
        let inserted = self.unique_indices.insert(index) as usize;

        // Add the size of the value to the total size if inserted
        self.size += inserted * size;

        inserted
    }

    /// Insert a new index to the unique indices, if the index is already in the unique indices it will return 0
    /// otherwise it will return 1
    ///
    /// It will also update the size of the accumulator, and updating the memory usage in case the index is inserted
    fn insert_accounted(
        &mut self,
        index: usize,
        size: usize,
        accounting_capacities: &mut usize,
    ) -> usize {
        let prev_capacity = self.capacity();
        let inserted = self.unique_indices.insert(index) as usize;

        *accounting_capacities += self.capacity() - prev_capacity;

        // Add the size of the value to the total size if inserted
        self.size += inserted * size;

        inserted
    }

    fn underlying_size(&self) -> usize {
        self.size
    }

    fn capacity(&self) -> usize {
        self.unique_indices.capacity()
    }
}

const DEFAULT_CAPACITY: usize = 128;

/// Specialized group accumulator for collect_set on bytes types
/// This is highly influenced by the implementation of [`GroupValuesBytes`] and [`GroupValuesRow`] in DataFusion
///
/// This is the internal representation:
///
/// For the following groups:
/// 1. "hello", "you"
/// 2. "hello", "world"
/// 3. "how", "are", "you", "?"
///
/// This is the unique values in the order they are added:
/// "hello", "you", "world", "how", "are", "?"
///
/// so we will have the following map:
/// "hello" -> 0
/// "you" -> 1
/// "world" -> 2
/// "how" -> 3
/// "are" -> 4
/// "?" -> 5
///
/// And it will be laid out in the following way in the values vector
/// And the [Self::unique_values_ref_count] will be the following:
/// Index:                           |     0     |    1  |   2     |   3   |   4   |  5  |
/// [Self::values]:                  |  "hello"  | "you" | "world" | "how" | "are" | "?" |
/// [Self::unique_values_ref_count]: |    2      |   2   |    1    |   1   |   1   |  1  |
///
/// and each index will have an index for the values in the values vector that in that group
/// So the groups are:
/// 1 -> `[0, 1]`
/// 2 -> `[0, 2]`
/// 3 -> `[3, 4, 1, 5]`
///
/// When Emit all is used we just clear the entire values and unique_values_ref_count
///
/// And then when we need to emit group 1 values we will go over the indices and get the values from the values vector
/// and decrement the unique_values_ref_count by 1 and if it is 0 will add it to the [Self::available_indices] to be reused
///
pub(crate) struct CollectSetBytesGroupsAccumulator<T: ByteArrayType> {
    /// Logically maps bytes values to index in [`Self::values`]
    ///
    /// Uses the raw API of hashbrown to avoid actually storing the
    /// keys (bytes) in the table
    ///
    /// keys: u64 hashes of the byte value
    /// values: (hash, value index)
    map: hashbrown::HashTable<(u64, usize)>,

    /// All unique values
    values: CollectSetGroupAccumulatorValues<Vec<u8>, GetSizeBytes>,

    /// All groups, each group store the indices of the values that are in that group
    states: Vec<AccumulatorState>,

    /// The sum of all states capacity
    ///
    /// Note this is incrementally updated with deltas to avoid the
    /// call to size() being a bottleneck. DataFusion saw size() being a
    /// bottleneck in earlier implementations when there were many
    /// distinct groups.
    states_capacities: usize,

    /// The Output list field
    return_field: FieldRef,

    /// random state used to generate hashes
    random_state: RandomState,

    /// buffer that stores hash values (reused across batches to save allocations)
    hashes_buffer: Vec<u64>,

    phantom_data: PhantomData<T>,
}

impl<T: ByteArrayType> CollectSetBytesGroupsAccumulator<T> {
    pub fn new(return_field: FieldRef) -> Self {
        Self {
            map: hashbrown::HashTable::with_capacity(DEFAULT_CAPACITY),
            values: CollectSetGroupAccumulatorValues::with_capacity(DEFAULT_CAPACITY),
            states: Vec::with_capacity(DEFAULT_CAPACITY),
            states_capacities: 0,
            random_state: Default::default(),
            hashes_buffer: Default::default(),
            return_field,
            phantom_data: PhantomData,
        }
    }

    #[tracy_gizmos::instrument]
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

    /// Update the hashes and get the mapping from the input array to the index in [`Self::values`] and the size of the value
    fn update_hashes<BatchAdder: AddBatch>(
        &mut self,
        value: &ArrayRef,
    ) -> Result<Vec<(usize, usize)>> {
        let unwrapped_value = BatchAdder::unwarp_values(value);

        // Calculate the hashes for the values
        {
            let batch_hashes = &mut self.hashes_buffer;
            batch_hashes.clear();
            batch_hashes.resize(unwrapped_value.len(), 0);

            create_hashes(
                &[Arc::clone(&unwrapped_value)],
                &self.random_state,
                batch_hashes,
            )?;
        }

        let bytes_values = unwrapped_value.as_bytes::<T>();
        assert_eq!(
            bytes_values.null_count(),
            0,
            "nulls should be filtered before updating hashes"
        );

        // Mapping between index in the current input array to the index in the values vector
        // mapping[i] = j means that the value at index i in the input array is the same as the value at index j in [`Self::values`]
        let mut mappings: Vec<(usize, usize)> = vec![(0, 0); bytes_values.len()];

        // Fill the map with the new values
        for (array_index, &target_hash) in self.hashes_buffer.iter().enumerate() {
            let entry = self.map.find(target_hash, |(exist_hash, value_index)| {
                // Somewhat surprisingly, this closure can be called even if the
                // hash doesn't match, so check the hash first with an integer
                // comparison first avoid the more expensive comparison with
                // group value. https://github.com/apache/datafusion/pull/11718
                target_hash == *exist_hash
                  && unsafe { self.values.get_unchecked(*value_index) }
                  == bytes_values.value_as_bytes(array_index)
            });

            let value_index = match entry {
                // Existing value_index for this value
                Some((_hash, value_index)) => *value_index,
                // Need to create new entry for the group
                None => {
                    let value_index = {
                        let new_value = Vec::from(bytes_values.value_as_bytes(array_index));
                        self.values.insert_new_unique(new_value)
                    };

                    self.map.insert_unique(
                        target_hash,
                        (target_hash, value_index),
                        // for hasher function, use precomputed hash value
                        |(hash, _value_index)| *hash,
                    );

                    value_index
                }
            };

            mappings[array_index] = (value_index, self.values[value_index].len());
        }

        Ok(mappings)
    }

    /// Convert all groups to a single ListArray
    ///
    /// this is faster than get N groups as we can avoid updating ref count
    fn get_all_groups(&mut self, state_used: &[AccumulatorState]) -> Result<ListArray> {
        let list = self.build_list(state_used)?;

        // Cleanup:
        self.values.reset(DEFAULT_CAPACITY);
        self.states.clear();
        self.states.shrink_to(DEFAULT_CAPACITY);
        self.states_capacities = 0;
        self.map.clear();
        self.map
          .shrink_to(DEFAULT_CAPACITY, |(hash, _value_index)| *hash);

        Ok(list)
    }

    /// Get some of the groups
    fn get_n_groups(&mut self, state_used: &[AccumulatorState]) -> Result<ListArray> {
        if self.states.is_empty() {
            return self.get_all_groups(state_used);
        }

        self.states_capacities -= state_used
          .iter()
          .map(|state| state.capacity())
          .sum::<usize>();

        let list = self.build_list(state_used)?;

        let initial_number_of_available_indices = self.values.get_number_of_available_indices();

        // Update the ref count for the values that are used in the group
        for value_index in state_used.iter().flat_map(|state| state.indices()) {
            self.values.decrement_ref_count(value_index);
        }

        // This means that we now have more available indices, so we should remove those values from the map and add empty vector for the value
        if initial_number_of_available_indices != self.values.get_number_of_available_indices() {
            // Keep only the items that are still in use
            self.map.retain(|(_, index)| self.values.is_used(*index));

            self.values.claim_data_from_added_available_indices(
                initial_number_of_available_indices,
                &Vec::new(),
            );
        }

        Ok(list)
    }

    #[tracy_gizmos::instrument]
    fn build_list(&mut self, state_used: &[AccumulatorState]) -> Result<ListArray> {
        let list_offsets = self.build_list_offsets(state_used);
        let total_number_of_items = {
            assert_eq!(list_offsets[0], 0);

            list_offsets[list_offsets.len() - 1].as_usize()
        };

        let bytes_values_buffer = self.build_byte_values(state_used);
        let bytes_offsets = self.build_bytes_offsets(state_used, total_number_of_items);

        // SAFETY: We know the offsets are valid as they are derived from the input
        // and using unchecked to skip the offsets assertions on the input string, and that each offset is a valid character boundary
        let bytes_values = unsafe {
            GenericByteArray::<T>::new_unchecked(
                bytes_offsets,
                bytes_values_buffer,
                // CollectSet is not nullable
                None,
            )
        };

        ListArray::try_new(
            Arc::clone(&self.return_field),
            list_offsets,
            Arc::new(bytes_values),
            // CollectSet is not nullable
            None,
        )
          .map_err(Into::into)
    }

    // #[tracy_gizmos::instrument]
    fn build_list_offsets(&self, state_used: &[AccumulatorState]) -> OffsetBuffer<i32> {
        OffsetBuffer::<i32>::from_lengths(state_used.iter().map(|state| state.len()))
    }

    // #[tracy_gizmos::instrument]
    fn build_bytes_offsets(
        &mut self,
        state_used: &[AccumulatorState],
        total_number_of_items: usize,
    ) -> OffsetBuffer<<T as ByteArrayType>::Offset> {
        let mut offset_builder = OffsetBufferBuilder::<T::Offset>::new(total_number_of_items);
        for state in state_used {
            for index in state.indices() {
                offset_builder.push_length(self.values[index].len());
            }
        }
        offset_builder.finish()
    }

    // #[tracy_gizmos::instrument]
    fn build_byte_values(&mut self, state_used: &[AccumulatorState]) -> Buffer {
        let total_number_of_bytes = state_used.iter().map(|state| state.underlying_size()).sum();

        let mut bytes_values_buffer = MutableBuffer::new(total_number_of_bytes);

        for state in state_used {
            for index in state.indices() {
                bytes_values_buffer.extend_from_slice(&self.values[index]);
            }
        }

        bytes_values_buffer.into()
    }
}

impl<T: ByteArrayType> GroupsAccumulator for CollectSetBytesGroupsAccumulator<T> {
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
          + self.map.capacity() * size_of::<(u64, usize)>()
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
    fn add_batch_for_ready_array<T: ByteArrayType>(
        group_accumulator: &mut CollectSetBytesGroupsAccumulator<T>,
        offsets: Option<&OffsetBuffer<i32>>,
        value_index_and_size: impl Iterator<Item = (usize, usize)>,
        group_indices: impl Iterator<Item = usize>,
    );

    fn get_offsets(value: &ArrayRef) -> Option<&OffsetBuffer<i32>>;

    fn unwarp_values(value: &ArrayRef) -> ArrayRef;
}

struct UpdateBatchCall;

impl AddBatch for UpdateBatchCall {
    #[tracy_gizmos::instrument]
    fn add_batch_for_ready_array<T: ByteArrayType>(
        group_accumulator: &mut CollectSetBytesGroupsAccumulator<T>,
        offsets: Option<&OffsetBuffer<i32>>,
        value_index_and_size: impl Iterator<Item = (usize, usize)>,
        group_indices: impl Iterator<Item = usize>,
    ) {
        assert!(offsets.is_none(), "offsets should be None for update batch");
        for (group_index, (value_index, size)) in group_indices.zip(value_index_and_size) {
            // Add to the appropriate group
            let inserted = group_accumulator.states[group_index].insert_accounted(
                value_index,
                size,
                &mut group_accumulator.states_capacities,
            );

            // Add 1 to the ref count if the value is new in that group
            group_accumulator
              .values
              .increment_ref_count_by(value_index, inserted);
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
    fn add_batch_for_ready_array<T: ByteArrayType>(
        group_accumulator: &mut CollectSetBytesGroupsAccumulator<T>,
        offsets: Option<&OffsetBuffer<i32>>,
        mut value_index_and_size: impl Iterator<Item = (usize, usize)>,
        group_indices: impl Iterator<Item = usize>,
    ) {
        let offsets = offsets.expect("offsets should exists for merge batch");

        let list_lengths = offsets.windows(2).map(|offsets| offsets[1] - offsets[0]);

        // Add to the appropriate group the row
        for (group_index, length) in group_indices.zip(list_lengths) {
            let state = &mut group_accumulator.states[group_index];
            let prev_capacity = state.capacity();

            for (value_index, size) in value_index_and_size.by_ref().take(length as usize) {
                // Add to the appropriate group
                let inserted = state.insert(value_index, size);

                // Add 1 to the ref count if the value is new in that group
                group_accumulator
                  .values
                  .increment_ref_count_by(value_index, inserted);
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

