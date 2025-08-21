use std::collections::HashSet;

/// State of a single group
#[derive(Default)]
pub(crate) struct SingleGroupUniqueIndices {
    unique_indices: HashSet<usize>,
}

impl SingleGroupUniqueIndices {
    pub(crate) fn len(&self) -> usize {
        self.unique_indices.len()
    }

    pub(crate) fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.unique_indices.iter()
    }

    /// Insert a new index to the unique indices, return true if it was inserted
    pub(crate) fn insert(&mut self, index: usize) -> bool {
        self.unique_indices.insert(index)
    }

    /// Insert a new index to the unique indices, and updating the memory usage
    pub(crate) fn insert_accounted(
        &mut self,
        index: usize,
        accounting_capacities: &mut usize,
    ) -> bool {
        let prev_capacity = self.capacity();
        let result = self.unique_indices.insert(index);
        *accounting_capacities += self.capacity() - prev_capacity;

        result
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.unique_indices.capacity()
    }
}
