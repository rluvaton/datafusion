use datafusion::common::utils::proxy::VecAllocExt;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::ops::Deref;

/// Wrapper to track unique values and their ref count for a group accumulator
///
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
pub(crate) struct CollectSetGroupAccumulatorValues<
    T,
    GetSizeV: GetHeapAllocatedSize<T> = OnlyOnStackSize,
> {
    /// Unique T values
    values: Vec<T>,

    /// The number of bytes the values vector has for heap allocated values in `T`
    heap_allocated_values_size: usize,

    /// Vector of ref count for each unique value, when it reaches 0 it will be added to the [Self::available_indices]
    /// to avoid shifting and updating all indices
    unique_values_ref_count: Vec<usize>,

    /// Queue of available indices in the values vector that can be reused
    available_indices: VecDeque<usize>,

    phantom_data: PhantomData<GetSizeV>,
}

impl<T, GetSizeV: GetHeapAllocatedSize<T>> CollectSetGroupAccumulatorValues<T, GetSizeV> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            heap_allocated_values_size: 0,
            unique_values_ref_count: Vec::with_capacity(capacity),
            available_indices: VecDeque::with_capacity(capacity),
            phantom_data: PhantomData,
        }
    }

    /// Insert new value that is known to be unique to the first available place or extend the values vector
    ///
    /// Returns the index of the newly inserted value
    #[inline]
    pub(crate) fn insert_new_unique(&mut self, value: T) -> usize {
        if GetSizeV::HAS_HEAP_ALLOCATION {
            self.heap_allocated_values_size += GetSizeV::get_heap_allocated_size(&value);
        }

        if let Some(index) = self.available_indices.pop_front() {
            if GetSizeV::HAS_HEAP_ALLOCATION {
                self.heap_allocated_values_size -=
                    GetSizeV::get_heap_allocated_size(&self.values[index]);
            }
            self.values[index] = value;
            index
        } else {
            let index = self.values.len();
            self.values.push(value);

            // Initialize it with 0 ref count (it will be incremented later by 1 for each group that uses it)
            self.unique_values_ref_count.push(0);

            index
        }
    }

    pub(crate) fn reset(&mut self, capacity: usize) {
        self.values.clear();
        self.values.shrink_to(capacity);

        self.unique_values_ref_count.clear();
        self.unique_values_ref_count.shrink_to(capacity);

        self.available_indices.clear();
        self.available_indices.shrink_to(capacity);

        self.heap_allocated_values_size = 0;
    }

    /// Claim the data from the values that are now available to be reused
    pub(crate) fn claim_data_from_added_available_indices(
        &mut self,
        initial_number_of_available_indices: usize,
        initial_value: &T,
    ) where
        T: Clone,
    {
        let newly_added_available_indices = self
            .available_indices
            .range((self.available_indices.len() - initial_number_of_available_indices)..);

        let mut cleaned_bytes = 0;

        for &value_index in newly_added_available_indices {
            if GetSizeV::HAS_HEAP_ALLOCATION {
                cleaned_bytes += GetSizeV::get_heap_allocated_size(&self.values[value_index]);
            }
            // Drop the last value - maybe for small values we can just keep it
            self.values[value_index] = initial_value.clone();
        }

        if GetSizeV::HAS_HEAP_ALLOCATION {
            self.heap_allocated_values_size -= cleaned_bytes;
            self.heap_allocated_values_size += (self.available_indices.len()
                - initial_number_of_available_indices)
                * GetSizeV::get_heap_allocated_size(initial_value);
        }
    }

    #[inline]
    pub(crate) fn increment_ref_count_by(&mut self, index: usize, count: usize) {
        self.unique_values_ref_count[index] += count;
    }

    #[inline]
    pub(crate) fn decrement_ref_count(&mut self, index: usize) {
        self.unique_values_ref_count[index] -= 1;
        if self.unique_values_ref_count[index] == 0 {
            self.available_indices.push_back(index);
        }
    }

    /// Return whether the value at the index is used by any group (ref count > 0)
    #[inline]
    pub(crate) fn is_used(&self, index: usize) -> bool {
        self.unique_values_ref_count[index] > 0
    }

    #[inline]
    pub(crate) fn get_number_of_available_indices(&self) -> usize {
        self.available_indices.len()
    }

    pub(crate) fn size(&self) -> usize {
        self.heap_allocated_values_size
            + self.values.allocated_size()
            + self.unique_values_ref_count.allocated_size()
            + self.available_indices.capacity() * size_of::<usize>()
    }
}

impl<T, GetSizeV: GetHeapAllocatedSize<T>> Deref for CollectSetGroupAccumulatorValues<T, GetSizeV> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T, GetSizeV: GetHeapAllocatedSize<T>> AsRef<[T]>
    for CollectSetGroupAccumulatorValues<T, GetSizeV>
{
    #[inline]
    fn as_ref(&self) -> &[T] {
        self
    }
}

/// Get size of value T
pub(crate) trait GetHeapAllocatedSize<T> {
    /// Whether the value is on the stack or heap, if on the stack the allocated size function will not be called
    const HAS_HEAP_ALLOCATION: bool;

    /// Get the size of the value, this should not return size for the stack allocated values as it is already
    /// accounted for in [`CollectSetGroupAccumulatorValues::size`] function
    fn get_heap_allocated_size(value: &T) -> usize;
}

pub(crate) struct OnlyOnStackSize;

impl<T> GetHeapAllocatedSize<T> for OnlyOnStackSize {
    const HAS_HEAP_ALLOCATION: bool = false;

    fn get_heap_allocated_size(_value: &T) -> usize {
        unreachable!("This should not be called if the value is on the stack")
    }
}
