use std::fmt::Debug;
use std::ops::Range;
use arrow::array::{ArrayRef, BooleanArray};
use datafusion_common::Result;
use crate::groups_accumulator::EmitTo;

pub trait ProcessGroups {
  type A: OrderedGroupsAccumulator;

  fn reserve_for_n_new_groups(t: &mut Self::A, n: usize) {
    // default implementation does nothing, but can be overridden to reserve space for new groups
  }

  /// Called when a new group is found in the input and it is not the same as the previous group and it is contained in the current batch
  fn on_new_standalone_group(t: &mut Self::A, group: &PartitionRange);

  fn on_new_single_item_group(t: &mut Self::A, group: usize) {
    Self::on_new_standalone_group(t, &PartitionRange { start: group, end: group + 1, is_same_as_before: false });
  }

  /// Finish the current group and flush it to the output, if any.
  /// If `opt_group` is `Some`, add the opt group values to the in progress and flush it
  fn flush_in_progress_group(t: &mut Self::A, opt_group: Option<&PartitionRange>);

  /// Add the last group to the in progress group
  fn add_last_group(t: &mut Self::A, last_group: &PartitionRange);
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
  properties: GroupsProperties,
  total_number_of_groups: usize,
}

impl GroupsInfo {

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

  pub fn process<Processor: ProcessGroups>(&self, input: &[ArrayRef], opt_filter: Option<&BooleanArray>, groups_accu: &mut Processor::A) -> Result<()> {
    // TODO - this is only if input is non null
    assert_eq!(input.len(), 1, "single argument to update_batch");
    let values = &input[0];
    assert_eq!(values.logical_null_count(), 0, "nulls are not supported (need to implement to not count non nulls)");
    assert!(opt_filter.is_none(), "filter is not supported");

    let (groups_iter, last_group) = self.groups_without_last();
    let mut groups_iter = groups_iter.iter();
    let Some(group) = groups_iter.next() else {
      let Some(last_group) = last_group else {
        return Ok(());
      };

      if !last_group.is_same_as_before {
        Processor::flush_in_progress_group(groups_accu, None);
      }
      Processor::add_last_group(groups_accu, last_group);

      return Ok(())
    };

    // TODO - reserve needed groups

    if !group.is_same_as_before {
      Processor::flush_in_progress_group(groups_accu, None);
      // // Flush prev group
      // self.ready_counts.push(self.current_count);
      // self.current_count = 0;
    } else {
      Processor::flush_in_progress_group(groups_accu, Some(group));
    }
    //
    //
    // // Add the first group
    // self.ready_counts.push(self.current_count + group.len() as i64);
    // self.current_count = 0;


    for group in groups_iter {
      Processor::on_new_standalone_group(groups_accu, group);
    }


    Processor::add_last_group(groups_accu, last_group.expect("must have last group if have groups before last"));

    Ok(())
  }

  /// Return iterator of (group_index, row_index) for each row
  pub fn as_iter_of_group_indices(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
    todo!()
    // self.groups.iter().enumerate().flat_map(|(group_index, group)| std::iter::repeat_n(group_index, group.len()))
  }

  /// Return iterator of (group_index, T)
  pub fn as_iter_of_group_indices_with_match<T>(&self, slice: &[T], include_last_group: bool) -> impl Iterator<Item = (usize, T)> + '_ {
    todo!()
    // self.groups.iter().enumerate().flat_map(|(group_index, group)| std::iter::repeat_n(group_index, group.len()))
  }

  pub fn total_number_of_groups(&self) -> usize {
    self.total_number_of_groups
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
