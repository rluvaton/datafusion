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

//! Building [`GroupsInfo`] for input sorted by the group keys

use datafusion_common::{Result, internal_err};
use datafusion_expr::GroupsInfo;
use datafusion_expr::ordered_groups_accumulator::{GroupsProperties, PartitionRange};

/// Builds the [`GroupsInfo`] of a batch from its contiguous, non empty group
/// runs, where `start_group_index` is the group index of the first run.
pub(crate) fn groups_info_from_ranges(
    groups: Vec<PartitionRange>,
    start_group_index: usize,
) -> GroupsInfo {
    debug_assert!(!groups.is_empty());

    // Leading run of single item groups, allowing the first group (which may
    // continue the previous batch) to have more than one item.
    let single_item_start = usize::from(groups[0].len() != 1);
    let single_item_end = single_item_start
        + groups[single_item_start..]
            .iter()
            .take_while(|group| group.len() == 1)
            .count();

    GroupsInfo::new(
        groups,
        GroupsProperties {
            range_of_single_item_groups: single_item_start..single_item_end,
        },
        start_group_index,
    )
}

/// Builds the [`GroupsInfo`] of one input batch from its interned group
/// indices when the input is fully sorted by the group keys.
///
/// `starting_num_groups` is the number of groups before interning the batch.
/// The first run continues the previous batch's last group when its index is
/// `starting_num_groups - 1`, and every following run must use the next index.
pub(crate) fn groups_info_from_group_indices(
    group_indices: &[usize],
    starting_num_groups: usize,
) -> Result<GroupsInfo> {
    debug_assert!(!group_indices.is_empty());

    let mut groups = Vec::new();
    let mut start = 0;
    let mut expected_group_index = None;
    for run in group_indices.chunk_by(|a, b| a == b) {
        let group_index = run[0];
        let is_same_as_before = match expected_group_index {
            None if group_index + 1 == starting_num_groups => true,
            None if group_index == starting_num_groups => false,
            Some(expected) if group_index == expected => false,
            _ => {
                return internal_err!(
                    "ordered aggregation expects contiguous group indices, got group {group_index} \
                     after {expected_group_index:?} (starting with {starting_num_groups} groups)"
                );
            }
        };
        groups.push(PartitionRange {
            start,
            end: start + run.len(),
            is_same_as_before,
        });
        start += run.len();
        expected_group_index = Some(group_index + 1);
    }

    Ok(groups_info_from_ranges(groups, group_indices[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges(info: &GroupsInfo) -> Vec<(usize, usize, bool)> {
        info.groups()
            .iter()
            .map(|g| (g.start, g.end, g.is_same_as_before))
            .collect()
    }

    #[test]
    fn groups_info_first_batch() -> Result<()> {
        let info = groups_info_from_group_indices(&[0, 0, 1, 2, 3, 3], 0)?;
        assert_eq!(
            ranges(&info),
            vec![(0, 2, false), (2, 3, false), (3, 4, false), (4, 6, false)]
        );
        assert_eq!(info.properties().range_of_single_item_groups, 1..3);
        assert_eq!(info.total_number_of_groups(), 4);
        assert_eq!(info.num_new_groups(), 4);
        Ok(())
    }

    #[test]
    fn groups_info_continues_previous_group() -> Result<()> {
        // 3 groups already interned, the batch starts with the last one (index 2)
        let info = groups_info_from_group_indices(&[2, 2, 3], 3)?;
        assert_eq!(ranges(&info), vec![(0, 2, true), (2, 3, false)]);
        assert_eq!(info.total_number_of_groups(), 4);
        assert_eq!(info.num_new_groups(), 1);

        // the batch starts a new group
        let info = groups_info_from_group_indices(&[3, 4, 5], 3)?;
        assert_eq!(
            ranges(&info),
            vec![(0, 1, false), (1, 2, false), (2, 3, false)]
        );
        assert_eq!(info.properties().range_of_single_item_groups, 0..3);
        assert!(info.are_all_single_item());
        assert_eq!(info.total_number_of_groups(), 6);
        Ok(())
    }

    #[test]
    fn groups_info_rejects_non_contiguous_groups() {
        // revisits a group
        assert!(groups_info_from_group_indices(&[0, 1, 0], 0).is_err());
        // skips a group index
        assert!(groups_info_from_group_indices(&[0, 2], 0).is_err());
        // starts with a group that is neither the last one nor a new one
        assert!(groups_info_from_group_indices(&[0, 1], 2).is_err());
    }
}
