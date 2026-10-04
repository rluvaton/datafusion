use std::fmt::Debug;
use std::ops::{BitAnd, Range};
use arrow::array::{Array, ArrayRef, ArrowPrimitiveType, BooleanArray, PrimitiveArray};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::compute::prep_null_mask_filter;
use datafusion_common::Result;
use crate::groups_accumulator::{EmitTo, GroupsAccumulator};

pub trait ProcessGroups {
  fn reserve_for_n_new_groups(&mut self, n: usize) {
    // default implementation does nothing, but can be overridden to reserve space for new groups
  }

  /// Called when a new group is found in the input and it is not the same as the previous group and it is contained in the current batch
  fn on_new_standalone_group(&mut self, group: &PartitionRange, group_index: usize) {
    // TODO - is it correct to call flush in progress group here?
    self.flush_in_progress_group(Some(group), group_index);
  }

  fn on_new_single_item_group(&mut self, row_index: usize, group_index: usize) {
    self.on_new_standalone_group(&PartitionRange { start: row_index, end: row_index + 1, is_same_as_before: false }, group_index);
  }

  /// Finish the current group and flush it to the output, if any.
  /// If `opt_group` is `Some`, add the opt group values to the in progress and flush it
  fn flush_in_progress_group(&mut self, opt_group: Option<&PartitionRange>, group_index: usize);

  /// Add the last group to the in progress group
  fn add_last_group(&mut self, last_group: &PartitionRange, group_index: usize) {
    last_group
      .as_range()
      .for_each(|row_index| self.fallback(row_index, group_index));
  }

  fn fallback(&mut self, row_index: usize, group_index: usize);
}

/// Hold a range for a partition in a batch (start..end) and whether this partition is the same as the previous one
/// (e.g. between batches)
#[derive(Debug, Clone)]
pub struct PartitionRange {
  /// The start index of the slice (inclusive)
  /// When limit is used, the start might not begin from the last end
  pub start: usize,

  /// The end index of the slice (exclusive)
  pub end: usize,

  /// Whether this slice point to the same partition as the previous one (between batches)
  pub is_same_as_before: bool,
}

impl PartitionRange {
  pub fn len(&self) -> usize {
    self.end - self.start
  }

  pub fn as_range(&self) -> Range<usize> {
    self.start..self.end
  }
}

#[derive(Debug, Clone)]
pub struct GroupsProperties {
  /// when this is 0..num_groups it means that every group here have single item (regardless of if the first group is the same as before or not)
  pub range_of_single_item_groups: Range<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllSingleItemType {
  All {
    first_group_is_same_as_before: bool,
  },
  AllExceptFirst {
    first_group_is_same_as_before: bool,
  },
  AllExceptLast,
  AllExceptFirstAndLast {
    first_group_is_same_as_before: bool,
  }
}

#[derive(Debug, Clone)]
pub struct GroupsInfo {
  // TODO - THIS IS a problem with the Boolean since the range would not be continues
  groups: Vec<PartitionRange>,
  start_group_index: usize,
  properties: GroupsProperties,
  total_number_of_rows: usize,
}

impl GroupsInfo {
  pub fn new(groups: Vec<PartitionRange>, properties: GroupsProperties, start_group_index: usize) -> Self {
    let total_number_of_rows: usize = groups.iter().map(|g| g.len()).sum();

    // assert the groups are sorted, dense and non overlapping
    for i in 0..groups.len() {
      assert_ne!(groups[i].len(), 0, "group length must be non zero");
      if i > 0 {
        assert_eq!(groups[i].start, groups[i - 1].end, "groups are overlapping or not sorted");
      } else {
        assert_eq!(groups[0].start, 0, "first group must start at 0");
      }
        assert!(groups[i].end > groups[i].start, "group end must be greater than start");
    }
    Self {
      groups,
      properties,
      total_number_of_rows,
      start_group_index,
    }
  }

  pub fn properties(&self) -> &GroupsProperties {
    &self.properties
  }

  pub fn groups(&self) -> &[PartitionRange] {
    &self.groups
  }

  pub fn groups_without_last(&self) -> (&[PartitionRange], Option<&PartitionRange>) {
    if self.groups.is_empty() {
      (&[], None)
    } else {
      let last = self.groups.last();
      let without_last = &self.groups[0..self.groups.len() - 1];
      (without_last, last)
    }
  }

  pub fn groups_without_first_and_last(&self) -> &[PartitionRange] {
    if self.groups.len() <= 2 {
      &[]
    } else {
      &self.groups[1..self.groups.len() - 1]
    }
  }

  fn process_groups_without_filter_and_nulls(&self, processor: &mut impl ProcessGroups) -> Result<()> {
    let mut group_index = self.start_group_index;

    let (groups_iter, last_group) = self.groups_without_last();
    let mut groups_iter = groups_iter.iter();
    let Some(group) = groups_iter.next() else {
      let Some(last_group) = last_group else {
        return Ok(());
      };

      // This is the only group
      if !last_group.is_same_as_before {
        // the prev group index is the last group index - 1
        processor.flush_in_progress_group(None, self.start_group_index - 1);
      }
      processor.add_last_group(last_group, self.start_group_index);

      return Ok(())
    };

    // TODO - reserve needed groups

    if !group.is_same_as_before && self.start_group_index > 0 {
      // Flush the prev group if not the first one
      processor.flush_in_progress_group(None, self.start_group_index - 1);
    }


    // Finish the first group
    processor.flush_in_progress_group(Some(group), group_index);
    group_index += 1;


    for group in groups_iter {
      processor.on_new_standalone_group(group, group_index);
      group_index += 1;
    }


    processor.add_last_group(last_group.expect("must have last group if have groups before last"), group_index);

    Ok(())
  }

  fn process_with_mask(&self, mask: BooleanBuffer, processor: &mut impl ProcessGroups) -> Result<()> {
    assert_eq!(mask.len(), self.total_number_of_rows);

    if !self.is_first_group_same_as_before() && self.start_group_index > 0 {
      processor.flush_in_progress_group(None, self.start_group_index - 1);
    }

    let mut current_group_index = self.start_group_index;
    let mut processed_rows_in_group = 0;

    for ((row_index, should_process), group_index) in mask.iter().enumerate().zip(self.as_iter_of_group_indices()) {
      // Fully filtered groups
      if processed_rows_in_group > 0 && group_index != current_group_index {
        // Flush the last group index
        processor.flush_in_progress_group(None, current_group_index);
        current_group_index = group_index;
        processed_rows_in_group = 0;
      }

      if should_process {
        processor.fallback(row_index, group_index);
        processed_rows_in_group += 1;
      }
    }

    Ok(())
  }

  pub fn process(&self, nulls: Option<&NullBuffer>, opt_filter: Option<&BooleanArray>, processor: &mut impl ProcessGroups) -> Result<()> {
    let mask = match (nulls.filter(|n| n.null_count() > 0), opt_filter) {
      (None, None) => None,
      (None, Some(filter)) => Some(prep_null_mask_filter(filter).into_parts().0),
      (Some(valids), None) => Some(valids.into_inner()),
      (Some(valids), Some(filter)) => {
        debug_assert_eq!(filter.len(), self.total_number_of_rows);
        debug_assert_eq!(valids.len(), self.total_number_of_rows);

        let filter_mask = prep_null_mask_filter(filter).into_parts().0;
        Some(&filter_mask & valids.inner())
      }
    };

    if let Some(mask) = mask {
      self.process_with_mask(mask, processor)
    } else {
      self.process_groups_without_filter_and_nulls(processor)
    }
  }


  /// Return iterator of group_index for each row
  pub fn as_iter_of_group_indices(&self) -> impl Iterator<Item = usize> + '_ {
    self.groups.iter().enumerate().flat_map(|(group_index, group)| std::iter::repeat_n(self.start_group_index + group_index, group.len()))
  }

  /// Return iterator of (group_index, T)
  pub fn as_iter_of_group_indices_with_match<T>(&self, slice: &[T], include_last_group: bool) -> impl Iterator<Item = (usize, T)> + '_ {
    todo!()
    // self.groups.iter().enumerate().flat_map(|(group_index, group)| std::iter::repeat_n(group_index, group.len()))
  }

  pub fn total_number_of_rows(&self) -> usize {
    self.total_number_of_rows
  }

  pub fn total_number_of_groups(&self) -> usize {
    let total = self.start_group_index + self.groups.len();
    if self.is_first_group_same_as_before() {
       total - 1
    } else {
      total
    }
  }

  pub fn num_new_groups(&self) -> usize {
    if self.is_first_group_same_as_before() {
      self.groups.len() - 1
    } else {
      self.groups.len()
    }
  }

  pub fn is_first_group_same_as_before(&self) -> bool {
    self.groups.first().unwrap().is_same_as_before
  }

  /// If the entire input match to a single group
  pub fn is_single_group(&self) -> bool {
    self.groups.len() == 1
  }

  pub fn are_all_single_item(&self) -> bool {
    self.properties.range_of_single_item_groups.start == 0 && self.properties.range_of_single_item_groups.end == self.groups.len()
  }

  pub fn are_all_single_item_except_first_group(&self) -> bool {
    self.properties.range_of_single_item_groups.start == 1 && self.properties.range_of_single_item_groups.end == self.groups.len()
  }

  // TODO - Check this with 1 groups, 2 groups, since it might have a bug there
  pub fn are_all_single_item_ignoring_edges(&self) -> bool {
    let range_of_single_item_groups = self.properties().range_of_single_item_groups;
    range_of_single_item_groups.start <= 1 && range_of_single_item_groups.end >= self.groups().len() - 1
  }

  /// When a single batch contain the end of the last partition and the start of a new one
  pub fn is_boundry(&self) -> bool {
    self.groups.len() == 2
  }
}

pub struct GroupsInfoIterator<'a> {
  groups_info: &'a GroupsInfo,
  current_group_index: usize,
}

/// Evaluator for a window function that **does not need the entire partition** to emit value
///
/// and when the input is sorted on the partition keys and order by keys
///
/// Examples of such cases are:
/// - `row_number`
/// - `count` where window frame is `range unbounded preceding current row`
/// - `sum` where window frame is `range unbounded preceding current row`
/// etc..
pub trait OrderedGroupsAccumulator: Send + std::any::Any {
  // TODO - add filtering
  fn update_batch(
    &mut self,
    input: &[ArrayRef],
    groups: &GroupsInfo,
    opt_filter: Option<&BooleanArray>,
  ) -> Result<()>;

  fn merge_batch(&mut self, input: &[ArrayRef], groups: &GroupsInfo) -> Result<()>;

  fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>>;

  fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef>;

  // /// Convert every state to evaluate, this is for when every row is a single partition so we can skip
  // fn state_to_evaluate(&self,
  //                      values: &[ArrayRef],
  //                      opt_filter: Option<&BooleanArray>) -> Result<ArrayRef>;

  //
  // /// num_rows in case the partition does not get any input
  // fn evaluate(&mut self, input: &[ArrayRef], partitions: &[PartitionRange], num_rows: usize) -> Result<ArrayRef>;
  //
  // /// When the entire input is a single partition
  // fn evaluate_single_partition(&mut self, input: &[ArrayRef], partition: &PartitionRange, num_rows: usize) -> Result<ArrayRef> {
  //   assert_eq!(partition.start, 0);
  //   assert_eq!(partition.end, num_rows);
  //
  //   self.evaluate(input, std::slice::from_ref(partition), num_rows)
  // }
  //
  // /// When a single batch contain the end of the last partition and the start of a new one
  // fn evaluate_boundry_partition(&mut self, input: &[ArrayRef], two_partitions: &[PartitionRange; 2], num_rows: usize) -> Result<ArrayRef> {
  //   assert_eq!(two_partitions[0].start, 0);
  //   assert_eq!(two_partitions[0].end, num_rows);
  //
  //   assert!(two_partitions[0].is_same_as_before);
  //   assert!(!two_partitions[1].is_same_as_before);
  //
  //   self.evaluate(input, two_partitions, num_rows)
  // }
  //
  // /// When each row is a single partition (except the first partition which may not be - e.g. boundary)
  // ///
  // ///
  // fn evaluate_every_row_is_single_partition(&mut self, input: &[ArrayRef], first_partition: &PartitionRange, num_rows: usize) -> Result<ArrayRef> {
  //   assert_eq!(first_partition.start, 0);
  //   let mut partitions = Vec::with_capacity(num_rows - first_partition.len() + 1);
  //
  //   partitions.push(first_partition.clone());
  //
  //   for i in first_partition.end..num_rows {
  //     partitions.push(PartitionRange {
  //       start: i,
  //       end: i + 1,
  //       is_same_as_before: false,
  //     });
  //   }
  //
  //   self.evaluate(input, &partitions, num_rows)
  // }

  /// TODO - Size on the heap? or including stack? since size_of will get the size I think but not sure for Box<dyn>
  fn size(&self) -> usize;
}

pub struct OrderedGroupsAccumulatorWrapper {
  inner: Box<dyn GroupsAccumulator>,
  total_number_of_groups: usize,
}

impl From<Box<dyn GroupsAccumulator>> for OrderedGroupsAccumulatorWrapper
{
  fn from(inner: Box<dyn GroupsAccumulator>) -> Self {
    Self {
      inner,
      total_number_of_groups: 0,
    }
  }
}

impl OrderedGroupsAccumulator for OrderedGroupsAccumulatorWrapper {
  fn update_batch(&mut self, input: &[ArrayRef], groups: &GroupsInfo, opt_filter: Option<&BooleanArray>) -> Result<()> {
    self.total_number_of_groups += groups.num_new_groups();
    assert_eq!(self.total_number_of_groups, groups.total_number_of_groups(), "total_number_of_groups must match");
    let group_indices = groups.as_iter_of_group_indices().collect::<Vec<_>>();
    self.inner.update_batch(input, &group_indices, opt_filter, self.total_number_of_groups)
  }

  fn merge_batch(&mut self, input: &[ArrayRef], groups: &GroupsInfo) -> Result<()> {
    self.total_number_of_groups += groups.num_new_groups();
    assert_eq!(self.total_number_of_groups, groups.total_number_of_groups(), "total_number_of_groups must match");
    let group_indices = groups.as_iter_of_group_indices().collect::<Vec<_>>();
    self.inner.merge_batch(input, &group_indices, self.total_number_of_groups)
  }

  fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
    match emit_to {
      EmitTo::All => self.total_number_of_groups = 0,
      EmitTo::First(n) => self.total_number_of_groups -= n,
    }
    self.inner.state(emit_to)
  }

  fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
    match emit_to {
      EmitTo::All => self.total_number_of_groups = 0,
      EmitTo::First(n) => self.total_number_of_groups -= n,
    }
    self.inner.evaluate(emit_to)
  }

  fn size(&self) -> usize {
    self.inner.size()
  }
}
//
//
// /// Super optimized row number for sorted input
// #[derive(Debug)]
// struct RowNumberSorted {
//   current_row: u64,
// }
//
// impl SortedPartitionEvaluatorThatCanEmitRightAway for RowNumberSorted {
//   fn evaluate(&mut self, input: &[ArrayRef], partitions: &[PartitionRange], num_rows: usize) -> Result<ArrayRef> {
//     assert_eq!(input.len(), 0);
//     assert_ne!(num_rows, 0);
//     assert_ne!(partitions.len(), 0);
//     debug_assert_eq!(partitions.iter().map(|p| p.len()).sum(), num_rows);
//
//     let mut output = vec![0; num_rows];
//
//     if !partitions[0].is_same_as_before {
//       self.current_row = 0;
//     }
//
//     {
//       let mut output_slice = output.as_mut_slice();
//
//       for partition in partitions {
//         for i in partition.start..partition.end {
//           self.current_row += 1;
//           output_slice[i] = self.current_row;
//         }
//         self.current_row = 0;
//       }
//     }
//
//     self.current_row = output[output.len() - 1];
//
//     Ok(Arc::new(UInt64Array::from(output)))
//   }
//
//   fn evaluate_single_partition(&mut self, input: &[ArrayRef], partition: &PartitionRange, num_rows: usize) -> Result<ArrayRef> {
//     assert_eq!(input.len(), 0);
//     assert_ne!(num_rows, 0);
//     assert_eq!(partition.len(), num_rows);
//
//     if !partition.is_same_as_before {
//       self.current_row = 0;
//     }
//
//     let start = self.current_row + 1;
//
//     let output = UInt64Array::from((start..start + (num_rows as u64)).collect::<Vec<u64>>());
//
//     self.current_row += num_rows as u64;
//
//     Ok(Arc::new(output))
//   }
//
//   fn evaluate_boundry_partition(&mut self, input: &[ArrayRef], two_partitions: &[PartitionRange; 2], num_rows: usize) -> Result<ArrayRef> {
//     assert_eq!(input.len(), 0);
//     assert_ne!(num_rows, 0);
//     assert_eq!(two_partitions[0].len() + two_partitions[1].len(), num_rows);
//
//     assert!(two_partitions[0].is_same_as_before);
//     assert!(!two_partitions[1].is_same_as_before);
//
//     let start = self.current_row + 1;
//
//     let partition_one_range = (start..start + (two_partitions[0].len() as u64));
//     let partition_two_range = 1..(1 + two_partitions[1].len() as u64);
//
//
//     let result_vec =  partition_one_range.chain(partition_two_range).collect::<Vec<_>>();
//     self.current_row = result_vec[result_vec.len() - 1];
//
//     let output = UInt64Array::from(result_vec);
//
//     Ok(Arc::new(output))
//   }
//
//   fn evaluate_every_row_is_single_partition(&mut self, input: &[ArrayRef], first_partition: &PartitionRange, num_rows: usize) -> Result<ArrayRef> {
//     assert_eq!(input.len(), 0);
//     assert_ne!(num_rows, 0);
//
//     let values = if !first_partition.is_same_as_before && first_partition.len() == 1 {
//       vec![1; num_rows]
//     } else if first_partition.len() == 1 {
//       let mut values = vec![1; num_rows];
//       values[0] = self.current_row + 1;
//       values
//     } else {
//       let start = self.current_row + 1;
//
//       let first_partition_range = (start..start + (first_partition.len() as u64));
//       let result = first_partition_range.chain(std::iter::repeat_n(1, num_rows - first_partition.len() + 1)).collect::<Vec<u64>>();
//
//       assert_eq!(result.len(), num_rows);
//
//       result
//     };
//
//     self.current_row = 1;
//     Ok(Arc::new(UInt64Array::from(values)))
//   }
//
//   fn size(&self) -> usize {
//     0
//   }
// }
//
// /// Super optimized sum for sorted input
// #[derive(Debug)]
// struct SumSortedGrowingWindow {
//   current_sum: u64,
// }
//
// impl SortedPartitionEvaluatorThatCanEmitRightAway for SumSortedGrowingWindow {
//   fn evaluate(&mut self, input: &[ArrayRef], partitions: &[PartitionRange], num_rows: usize) -> Result<ArrayRef> {
//     assert_eq!(input.len(), 1);
//     assert_ne!(num_rows, 0);
//     assert_ne!(partitions.len(), 0);
//     debug_assert_eq!(partitions.iter().map(|p| p.len()).sum(), num_rows);
//
//     let input = input[0].as_primitive::<UInt64Type>();
//
//     let mut output = vec![0; num_rows];
//     let mut input_iter = input.iter();
//
//     if !partitions[0].is_same_as_before {
//       self.current_sum = 0;
//     }
//
//     {
//       let mut shift = 0;
//       let mut output_slice = output.as_mut_slice();
//       for partition in partitions {
//         for (i, value) in input_iter.by_ref().take(partition.len()).enumerate() {
//           self.current_sum += value.unwrap_or_default();
//           output_slice[i + shift] = self.current_sum;
//         }
//         self.current_sum = 0;
//         shift += partition.len();
//       }
//     }
//
//     self.current_sum = output[output.len() - 1];
//
//     Ok(Arc::new(UInt64Array::from(output)))
//   }
//
//   fn evaluate_single_partition(&mut self, input: &[ArrayRef], partition: &PartitionRange, num_rows: usize) -> Result<ArrayRef> {
//     assert_eq!(input.len(), 1);
//     assert_ne!(num_rows, 0);
//     assert_eq!(partition.len(), num_rows);
//     let input = input[0].as_primitive::<UInt64Type>();
//
//     if !partition.is_same_as_before {
//       self.current_sum = 0;
//     }
//
//     let output: UInt64Array = input.iter().map(|x| {
//       // null will be 0
//       self.current_sum += x.unwrap_or_default();
//       self.current_sum
//     }).collect();
//
//     Ok(Arc::new(output))
//   }
//
//   fn evaluate_every_row_is_single_partition(&mut self, input: &[ArrayRef], first_partition: &PartitionRange, num_rows: usize) -> Result<ArrayRef> {
//     assert_eq!(input.len(), 1);
//     assert_ne!(num_rows, 0);
//     let input = input[0].as_primitive::<UInt64Type>();
//
//     let values = if !first_partition.is_same_as_before && first_partition.len() == 1 {
//       input.clone()
//     } else if first_partition.len() == 1 {
//       todo!()
//     } else {
//       todo!()
//     };
//
//     // TODO - is this the right thing? what about nulls
//     self.current_sum = values.iter().last().copied().unwrap_or_default();
//     Ok(Arc::new(UInt64Array::from(values)))
//   }
//
//   fn size(&self) -> usize {
//     0
//   }
// }
