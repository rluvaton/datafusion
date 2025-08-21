mod convert_to_state;
mod group_accumulator_unique_values;
mod track_single_group_unique_indices;

pub(crate) use convert_to_state::convert_to_state;
pub(crate) use group_accumulator_unique_values::{
    CollectSetGroupAccumulatorValues, GetHeapAllocatedSize,
};
pub(crate) use track_single_group_unique_indices::SingleGroupUniqueIndices;

use arrow::array::{Array, ArrayRef, BooleanArray};
use std::ops::BitAnd;

/// Create a [`BooleanArray`] that will be used to filter null values in the array and merge it with the optional filter
///
/// This functions guarantees:
/// 1. The returned array will have the same length as the input array
/// 2. The returned array will not have any nulls in it
///
/// # Panic
/// If the `opt_filter` is `None` and the array does not have nulls this function will panic
///
pub(crate) fn merge_filter_and_null_removal(
    opt_filter: Option<&BooleanArray>,
    array: &ArrayRef,
) -> BooleanArray {
    let opt_filter = opt_filter.map(|filter| {
        if filter.null_count() > 0 {
            // Make sure all values behind nulls are 0
            filter.values().bitand(filter.nulls().unwrap().inner())
        } else {
            // Clone is zero-copy due to the way it is implemented
            filter.values().clone()
        }
    });

    // If we have filter or nulls we merge them to 1 using
    // & to only keep the rows that are not null and pass the filter
    let merged_filter = match (opt_filter, array.nulls()) {
        (Some(filter), Some(nulls)) => (&filter) & nulls.inner(),
        (Some(filter), None) => filter,
        (None, Some(nulls)) => nulls.inner().clone(),
        (None, None) => {
            unreachable!("merge_filter_and_null_removal called with no filter and no nulls")
        }
    };

    BooleanArray::new(
        merged_filter,
        // No nulls
        None,
    )
}

/// Ensure that states has total_num_groups
pub(super) fn make_accumulators_if_needed<T: Default>(
    states: &mut Vec<T>,
    total_num_groups: usize,
) {
    // can't shrink
    assert!(total_num_groups >= states.len());

    // instantiate new accumulators
    let new_accumulators = total_num_groups - states.len();
    states.extend((0..new_accumulators).map(|_| T::default()));
}
