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

//! Common utilities for aggregate tables used in aggregations that inputs are ordered
//! by the groups.

use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::ordered_groups_accumulator::{
    GroupsProperties, OrderedGroupsAccumulatorWrapper, PartitionRange,
};
use datafusion_expr::{AggregateMetrics, EmitTo, GroupsInfo, OrderedGroupsAccumulator};
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;

use crate::InputOrderMode;
use crate::PhysicalExpr;
use crate::aggregates::group_values::{
    AccumulatorPhase, AggregateAccumulatorMetrics, AggregateArgumentMetrics,
    GroupByMetrics, GroupValues, new_group_values,
};
use crate::aggregates::order::GroupOrdering;
use crate::aggregates::{
    AggregateExec, AggregateMode, PhysicalGroupBy, aggregate_expressions,
    evaluate_group_by,
};

use super::AggregateTableMetrics;
use super::common::{
    AggregateHashTable, CompactedAccumulatorArgs, HashAggregateAccumulator,
    RowAlignedAccumulatorArgs, create_group_accumulator,
};

#[derive(Clone)]
pub(in crate::aggregates) struct OrderedAggregateTableMetrics {
    pub(super) group_by: GroupByMetrics,
    pub(super) aggregate_arguments: AggregateArgumentMetrics,
    pub(super) accumulator: Arc<AggregateAccumulatorMetrics>,
    pub(super) submetrics: Vec<Arc<dyn AggregateMetrics>>,
}

impl OrderedAggregateTableMetrics {
    pub(in crate::aggregates) fn new(agg: &AggregateExec, partition: usize) -> Self {
        let metrics = AggregateTableMetrics::new(agg, partition);
        Self {
            group_by: metrics.group_by,
            aggregate_arguments: metrics.aggregate_arguments,
            accumulator: metrics.accumulator,
            submetrics: metrics.submetrics,
        }
    }

    pub(in crate::aggregates) fn from_hash_table<AggrMode>(
        table: &AggregateHashTable<AggrMode>,
    ) -> Self {
        Self {
            group_by: table.group_by_metrics.clone(),
            aggregate_arguments: table.aggregate_argument_metrics.clone(),
            accumulator: Arc::clone(&table.aggregate_accumulator_metrics),
            submetrics: table.aggregate_submetrics.clone(),
        }
    }
}

/// Create an accumulator for `agg_expr` -- an [`OrderedGroupsAccumulator`] if
/// that is supported by the aggregate, or an [`OrderedGroupsAccumulatorWrapper`]
/// around the [`GroupsAccumulator`](datafusion_expr::GroupsAccumulator) if not.
pub(in crate::aggregates) fn create_ordered_group_accumulator(
    agg_expr: &Arc<AggregateFunctionExpr>,
    metrics: Arc<dyn AggregateMetrics>,
) -> Result<Box<dyn OrderedGroupsAccumulator>> {
    if agg_expr.ordered_groups_accumulator_supported() {
        agg_expr.create_ordered_groups_accumulator_with_metrics(metrics)
    } else {
        create_group_accumulator(agg_expr, metrics).map(|acc| {
            Box::new(OrderedGroupsAccumulatorWrapper::from(acc))
                as Box<dyn OrderedGroupsAccumulator>
        })
    }
}

/// State and argument information for a single aggregate in the ordered table.
pub(super) enum OrderedAggregateAccumulator {
    /// The input is fully sorted by the group keys (and there is a single
    /// grouping set), so the groups of each batch are contiguous runs of
    /// consecutive group indices and can be described by [`GroupsInfo`].
    Ordered(HashAggregateAccumulator<dyn OrderedGroupsAccumulator>),

    /// The input is only partially sorted or there are multiple grouping sets,
    /// so the group indices of a batch are not contiguous runs.
    Unordered(HashAggregateAccumulator),
}

/// Evaluated arguments for one [`OrderedAggregateAccumulator`].
pub(super) enum OrderedAccumulatorArgs {
    /// Used by [`OrderedAggregateAccumulator::Ordered`].
    ///
    /// The arguments keep one row per input row and the `FILTER` is passed to
    /// the accumulator, so groups whose rows are all filtered out are still
    /// part of the dense [`GroupsInfo`].
    RowAligned(RowAlignedAccumulatorArgs),

    /// Used by [`OrderedAggregateAccumulator::Unordered`].
    Compacted(CompactedAccumulatorArgs),
}

/// Group assignment of the rows of one input batch.
pub(super) enum BatchGroups<'a> {
    /// Used by [`OrderedAggregateAccumulator::Ordered`].
    Ordered(&'a GroupsInfo),

    /// Used by [`OrderedAggregateAccumulator::Unordered`].
    Indices {
        /// One group index per input row.
        group_indices: &'a [usize],
        /// Total number of groups currently interned, including new groups.
        total_num_groups: usize,
    },
}

/// Evaluated group keys and accumulator arguments for one input batch.
pub(super) struct OrderedEvaluatedAggregateBatch {
    /// One entry per grouping set; each entry contains all evaluated group key
    /// arrays for the current input batch.
    pub(super) grouping_set_args: Vec<Vec<ArrayRef>>,

    /// One entry per aggregate expression.
    pub(super) accumulator_args: Vec<OrderedAccumulatorArgs>,
}

/// Function used by [`OrderedAggregateTable::aggregate_evaluated_batch`] to
/// update one accumulator with one evaluated input batch.
pub(super) type OrderedAggregateBatchFn = fn(
    &mut OrderedAggregateAccumulator,
    &OrderedAccumulatorArgs,
    &BatchGroups<'_>,
) -> Result<()>;

/// Function used by [`OrderedAggregateTable::materialize_groups`] to
/// materialize one accumulator's output columns.
pub(super) type OrderedMaterializeAccumulatorFn =
    fn(&mut OrderedAggregateAccumulator, EmitTo) -> Result<Vec<ArrayRef>>;

impl OrderedAggregateAccumulator {
    fn evaluate_args(&self, batch: &RecordBatch) -> Result<OrderedAccumulatorArgs> {
        match self {
            Self::Ordered(acc) => acc
                .evaluate_row_aligned_args(batch)
                .map(OrderedAccumulatorArgs::RowAligned),
            Self::Unordered(acc) => acc
                .evaluate_compacted_args(batch)
                .map(OrderedAccumulatorArgs::Compacted),
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Ordered(acc) => acc.accumulator.size(),
            Self::Unordered(acc) => acc.size(),
        }
    }

    pub(super) fn update_batch(
        &mut self,
        values: &OrderedAccumulatorArgs,
        groups: &BatchGroups<'_>,
    ) -> Result<()> {
        match (self, values, groups) {
            (
                Self::Ordered(acc),
                OrderedAccumulatorArgs::RowAligned(values),
                BatchGroups::Ordered(groups),
            ) => acc.accumulator.update_batch(
                &values.arguments,
                groups,
                values.filter.as_ref(),
            ),
            (
                Self::Unordered(acc),
                OrderedAccumulatorArgs::Compacted(values),
                BatchGroups::Indices {
                    group_indices,
                    total_num_groups,
                },
            ) => acc.update_batch(values, group_indices, *total_num_groups),
            _ => internal_err!("mismatched ordered aggregate accumulator inputs"),
        }
    }

    pub(super) fn merge_batch(
        &mut self,
        values: &OrderedAccumulatorArgs,
        groups: &BatchGroups<'_>,
    ) -> Result<()> {
        match (self, values, groups) {
            (
                Self::Ordered(acc),
                OrderedAccumulatorArgs::RowAligned(values),
                BatchGroups::Ordered(groups),
            ) => {
                debug_assert!(values.filter.is_none());
                acc.accumulator.merge_batch(&values.arguments, groups)
            }
            (
                Self::Unordered(acc),
                OrderedAccumulatorArgs::Compacted(values),
                BatchGroups::Indices {
                    group_indices,
                    total_num_groups,
                },
            ) => acc.merge_batch(values, group_indices, *total_num_groups),
            _ => internal_err!("mismatched ordered aggregate accumulator inputs"),
        }
    }

    pub(super) fn evaluate_to_columns(
        &mut self,
        emit_to: EmitTo,
    ) -> Result<Vec<ArrayRef>> {
        match self {
            Self::Ordered(acc) => Ok(vec![acc.accumulator.evaluate(emit_to)?]),
            Self::Unordered(acc) => acc.evaluate_to_columns(emit_to),
        }
    }

    pub(super) fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        match self {
            Self::Ordered(acc) => acc.accumulator.state(emit_to),
            Self::Unordered(acc) => acc.state(emit_to),
        }
    }
}

/// Builds the [`GroupsInfo`] of one input batch from its interned group
/// indices when the input is fully sorted by the group keys.
///
/// `starting_num_groups` is the number of groups before interning the batch.
/// The first run continues the previous batch's last group when its index is
/// `starting_num_groups - 1`, and every following run must use the next index.
fn build_groups_info(
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

    // Leading run of single item groups, allowing the first group (which may
    // continue the previous batch) to have more than one item.
    let single_item_start = usize::from(groups[0].len() != 1);
    let single_item_end = single_item_start
        + groups[single_item_start..]
            .iter()
            .take_while(|group| group.len() == 1)
            .count();

    Ok(GroupsInfo::new(
        groups,
        GroupsProperties {
            range_of_single_item_groups: single_item_start..single_item_end,
        },
        group_indices[0],
    ))
}

/// Aggregate table shared by the ordered single, partial and final paths.
///
/// # Ordering optimization
///
/// The table consumes input batches while `GroupOrdering` tracks which groups
/// are proven complete. Completed groups can be emitted before the input stream
/// ends, which keeps memory bounded by the active ordered key range.
///
/// # Single, partial and final variant difference
///
/// The partial and final aggregate tables implement the two stages of grouped
/// aggregation, while the single aggregate table implements both stages in one
/// table. See
/// [`OrderedPartialAggregateStream`](crate::aggregates::ordered_partial_stream::OrderedPartialAggregateStream)
/// for the high-level plan shape.
///
/// Example: `AVG(v) FILTER (WHERE v>0) GROUP BY k`
///
/// Partial table ([`AggregateMode::Partial`], with optional filter from query):
/// - Input rows: `k, v`
/// - Table stores: `k, sum(v), count(v)`
/// - Output schema: `k, sum(v), count(v)`
///
/// Final table ([`AggregateMode::Final`], no filters):
/// - Input rows: `k, sum(v), count(v)`
/// - Table stores: `k, sum(v), count(v)`
/// - Output schema: `k, avg(v)`
///
/// Single table ([`AggregateMode::Single`], with optional filter from query):
/// - Input rows: `k, v`
/// - Table stores: `k, sum(v), count(v)`
/// - Output schema: `k, avg(v)`
///
/// # Marker Type
///
/// `OrderedAggrMode` selects the aggregate semantics. For example,
/// `OrderedAggregateTable::<PartialMarker>::new(...)` consumes raw rows
/// and emits partial states, while
/// `OrderedAggregateTable::<FinalMarker>::new_with_input_order(...)`
/// consumes partial states and emits final values.
///
/// Shared methods live on `impl<T>`; single/partial/final behavior lives on
/// marker-specific impls.
pub(in crate::aggregates) struct OrderedAggregateTable<OrderedAggrMode> {
    /// Output schema: group columns followed by aggregate state or final values.
    pub(super) output_schema: SchemaRef,

    /// Intermediate-state schema used when memory pressure requires the table
    /// to pass through or spill its current state.
    pub(super) state_schema: SchemaRef,

    /// Grouping and accumulator-specific timing metrics.
    pub(super) group_by_metrics: GroupByMetrics,

    /// Per-aggregate timing metrics for evaluating aggregate arguments.
    pub(super) aggregate_argument_metrics: AggregateArgumentMetrics,

    /// Per-aggregate timing metrics for accumulator operations.
    pub(super) aggregate_accumulator_metrics: Arc<AggregateAccumulatorMetrics>,

    /// Optional internal metrics owned by each aggregate expression.
    pub(super) aggregate_submetrics: Vec<Arc<dyn AggregateMetrics>>,

    /// Group keys, ordering state, and accumulator states.
    pub(super) buffer: OrderedAggregateTableBuffer,

    _mode: PhantomData<OrderedAggrMode>,
}

/// Buffer for the ordered aggregate table's group keys and accumulator states.
///
/// It accumulates input during aggregation and emits output rows as soon as the
/// input ordering proves those groups are complete.
///
/// [`GroupOrdering`] tracks when and how to do early emit.
/// [`GroupValues`] stores the physical group-key layout, while
/// [`datafusion_expr::GroupsAccumulator`] stores per-group aggregate state.
pub(super) struct OrderedAggregateTableBuffer {
    /// GROUP BY expressions evaluated against input batches.
    pub(super) group_by: Arc<PhysicalGroupBy>,

    /// Tracks how far ordered input allows this table to drain safely.
    pub(super) group_ordering: GroupOrdering,

    /// Interned group keys, in the same group-id order used by accumulators.
    pub(super) group_values: Box<dyn GroupValues>,

    /// Scratch group id vector for the current input batch.
    pub(super) group_indices: Vec<usize>,

    /// Whether the accumulators are [`OrderedAggregateAccumulator::Ordered`].
    ///
    /// True when the input is fully sorted by the group keys and there is a
    /// single grouping set.
    pub(super) use_ordered_accumulators: bool,

    /// One item per aggregate expression.
    ///
    /// Example: `COUNT(x), SUM(y)` creates two items. Each item owns the input
    /// expressions, optional filter, and accumulator state for all groups.
    pub(super) accumulators: Vec<OrderedAggregateAccumulator>,
}

/// Methods shared by all aggregate modes
impl<AggrMode> OrderedAggregateTable<AggrMode> {
    #[expect(
        clippy::too_many_arguments,
        reason = "keeps ordered single, partial and final table construction explicit"
    )]
    pub(super) fn new_for_mode(
        agg: &AggregateExec,
        input_schema: &SchemaRef,
        output_schema: SchemaRef,
        state_schema: SchemaRef,
        input_order_mode: &InputOrderMode,
        aggregate_mode: &AggregateMode,
        filters: Vec<Option<Arc<dyn PhysicalExpr>>>,
        metrics: OrderedAggregateTableMetrics,
    ) -> Result<Self> {
        let group_ordering = GroupOrdering::try_new(input_order_mode)?;
        let use_ordered_accumulators = matches!(group_ordering, GroupOrdering::Full(_))
            && agg.group_by().groups().len() == 1;
        let group_schema = agg.group_by().group_schema(input_schema)?;
        let group_values = new_group_values(group_schema, &group_ordering)?;
        let aggregate_arguments = aggregate_expressions(
            agg.aggr_expr(),
            aggregate_mode,
            agg.group_by().num_group_exprs(),
        )?;
        let accumulators = agg
            .aggr_expr()
            .iter()
            .zip(aggregate_arguments)
            .zip(filters)
            .zip(metrics.submetrics.iter())
            .map(|(((agg_expr, arguments), filter), submetrics)| {
                if use_ordered_accumulators {
                    let accumulator = create_ordered_group_accumulator(
                        agg_expr,
                        Arc::clone(submetrics),
                    )?;
                    Ok(OrderedAggregateAccumulator::Ordered(
                        HashAggregateAccumulator::new(
                            Arc::clone(agg_expr),
                            arguments,
                            filter,
                            accumulator,
                            Arc::clone(submetrics),
                        ),
                    ))
                } else {
                    let accumulator =
                        create_group_accumulator(agg_expr, Arc::clone(submetrics))?;
                    Ok(OrderedAggregateAccumulator::Unordered(
                        HashAggregateAccumulator::new(
                            Arc::clone(agg_expr),
                            arguments,
                            filter,
                            accumulator,
                            Arc::clone(submetrics),
                        ),
                    ))
                }
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            output_schema,
            state_schema,
            group_by_metrics: metrics.group_by,
            aggregate_argument_metrics: metrics.aggregate_arguments,
            aggregate_accumulator_metrics: metrics.accumulator,
            aggregate_submetrics: metrics.submetrics,
            buffer: OrderedAggregateTableBuffer {
                group_by: Arc::clone(agg.group_by()),
                group_ordering,
                group_values,
                group_indices: vec![],
                use_ordered_accumulators,
                accumulators,
            },
            _mode: PhantomData,
        })
    }

    /// Evaluates all group by keys and accumulator args.
    ///
    /// e.g., `select k+1, sum(v*v) from t group by (k+1)`, this function
    /// evaluates `k+1`, `v*v`.
    pub(super) fn evaluate_batch(
        &self,
        batch: &RecordBatch,
    ) -> Result<OrderedEvaluatedAggregateBatch> {
        let grouping_set_args =
            self.group_by_metrics.time_group_key_preparation(|| {
                evaluate_group_by(&self.buffer.group_by, batch)
            })?;

        let accumulator_args = self.group_by_metrics.time_aggregate_arguments(|| {
            self.buffer
                .accumulators
                .iter()
                .enumerate()
                .map(|(idx, acc)| {
                    self.aggregate_argument_metrics
                        .time(idx, || acc.evaluate_args(batch))
                })
                .collect::<Result<Vec<_>>>()
        })?;

        Ok(OrderedEvaluatedAggregateBatch {
            grouping_set_args,
            accumulator_args,
        })
    }

    /// Called after the input stream is exhausted and the last batch has been
    /// aggregated.
    ///
    /// Updates the internal `GroupOrdering` so it can continue emitting until
    /// the buffer is empty.
    pub(in crate::aggregates) fn input_done(&mut self) {
        self.buffer.group_ordering.input_done();
    }

    /// Returns the ordering state used to decide how memory pressure is handled.
    pub(in crate::aggregates) fn group_ordering(&self) -> &GroupOrdering {
        &self.buffer.group_ordering
    }

    /// Number of groups currently buffered.
    pub(in crate::aggregates) fn num_groups(&self) -> usize {
        self.buffer.group_values.len()
    }

    /// Check if there is zero groups accumulated so far.
    pub(in crate::aggregates) fn is_empty(&self) -> bool {
        self.num_groups() == 0
    }

    /// All internal buffer's memory size.
    pub(in crate::aggregates) fn memory_size(&self) -> usize {
        self.buffer
            .accumulators
            .iter()
            .map(|acc| acc.size())
            .sum::<usize>()
            + self.buffer.group_values.size()
            + self.buffer.group_ordering.size()
            + self.buffer.group_indices.allocated_size()
    }

    pub(in crate::aggregates) fn metrics(&self) -> OrderedAggregateTableMetrics {
        OrderedAggregateTableMetrics {
            group_by: self.group_by_metrics.clone(),
            aggregate_arguments: self.aggregate_argument_metrics.clone(),
            accumulator: Arc::clone(&self.aggregate_accumulator_metrics),
            submetrics: self.aggregate_submetrics.clone(),
        }
    }

    /// Takes every intermediate aggregate state and resets the table so it can
    /// continue with a new ordered input segment.
    ///
    /// Unlike normal ordered emission, this operation is allowed to take the
    /// active (incomplete) groups. Partial aggregation can pass those states to
    /// its final stage, while single and final aggregation sort and spill them
    /// before replay.
    pub(in crate::aggregates) fn take_state_batch(
        &mut self,
    ) -> Result<Option<RecordBatch>> {
        if self.buffer.group_values.is_empty() {
            return Ok(None);
        }

        let accumulator_metrics = Arc::clone(&self.aggregate_accumulator_metrics);
        let output = self.group_by_metrics.time_emitting(|| {
            let mut output = self.buffer.group_values.emit(EmitTo::All)?;
            for (idx, acc) in self.buffer.accumulators.iter_mut().enumerate() {
                output.extend(accumulator_metrics.time(
                    idx,
                    AccumulatorPhase::State,
                    || acc.state(EmitTo::All),
                )?);
            }
            Ok::<_, datafusion_common::DataFusionError>(output)
        })?;

        let batch = RecordBatch::try_new(Arc::clone(&self.state_schema), output)?;
        debug_assert!(batch.num_rows() > 0);

        // `emit(EmitTo::All)` resets accumulator state. Explicitly shrink the
        // key/index buffers too so the memory reservation can be released
        // before the batch is passed downstream or sorted for spilling.
        self.buffer.group_values.clear_shrink(0);
        self.buffer.group_indices.clear();
        self.buffer.group_indices.shrink_to_fit();
        self.buffer.group_ordering.reset();

        Ok(Some(batch))
    }

    /// Aggregates one evaluated input batch after selecting the mode-specific
    /// accumulator operation.
    ///
    /// Each aggregation mode chooses a different `aggregate_fn` according to its
    /// semantics. For example, partial aggregation takes raw inputs and updates
    /// stored partial states, so it uses
    /// [`OrderedAggregateAccumulator::update_batch`].
    pub(super) fn aggregate_evaluated_batch(
        &mut self,
        evaluated_batch: &OrderedEvaluatedAggregateBatch,
        aggregate_fn: OrderedAggregateBatchFn,
        accumulator_phase: AccumulatorPhase,
    ) -> Result<()> {
        let accumulator_metrics = Arc::clone(&self.aggregate_accumulator_metrics);
        let group_by_metrics = self.group_by_metrics.clone();
        for group_values in &evaluated_batch.grouping_set_args {
            let (starting_num_groups, total_num_groups) = group_by_metrics
                .time_group_key_preparation(|| {
                    let starting_num_groups = self.buffer.group_values.len();
                    self.buffer
                        .group_values
                        .intern(group_values, &mut self.buffer.group_indices)?;
                    let total_num_groups = self.buffer.group_values.len();
                    if total_num_groups > starting_num_groups {
                        self.buffer.group_ordering.new_groups(
                            group_values,
                            &self.buffer.group_indices,
                            total_num_groups,
                        )?;
                    }
                    Ok::<_, datafusion_common::DataFusionError>((
                        starting_num_groups,
                        total_num_groups,
                    ))
                })?;

            let groups_info;
            let batch_groups = if self.buffer.use_ordered_accumulators {
                if self.buffer.group_indices.is_empty() {
                    continue;
                }
                groups_info = group_by_metrics.time_group_key_preparation(|| {
                    build_groups_info(&self.buffer.group_indices, starting_num_groups)
                })?;
                BatchGroups::Ordered(&groups_info)
            } else {
                BatchGroups::Indices {
                    group_indices: &self.buffer.group_indices,
                    total_num_groups,
                }
            };

            group_by_metrics.time_aggregation(|| {
                for (idx, (acc, values)) in self
                    .buffer
                    .accumulators
                    .iter_mut()
                    .zip(evaluated_batch.accumulator_args.iter())
                    .enumerate()
                {
                    accumulator_metrics.time(idx, accumulator_phase, || {
                        aggregate_fn(acc, values, &batch_groups)
                    })?;
                }
                Ok::<(), datafusion_common::DataFusionError>(())
            })?;
        }

        Ok(())
    }

    /// Removes the selected groups once and materializes their output columns.
    /// The caller chooses the completed prefix.
    pub(super) fn materialize_groups(
        &mut self,
        emit_to: EmitTo,
        materialize_accumulator_fn: OrderedMaterializeAccumulatorFn,
        accumulator_phase: AccumulatorPhase,
    ) -> Result<RecordBatch> {
        let accumulator_metrics = Arc::clone(&self.aggregate_accumulator_metrics);
        let output = self.group_by_metrics.time_emitting(|| {
            let mut output = self.buffer.group_values.emit(emit_to)?;
            // `EmitTo::All` is only used after `input_done`, when the ordering
            // state no longer tracks group indexes.
            if let EmitTo::First(n) = emit_to {
                self.buffer.group_ordering.remove_groups(n);
            }

            for (idx, acc) in self.buffer.accumulators.iter_mut().enumerate() {
                output.extend(accumulator_metrics.time(
                    idx,
                    accumulator_phase,
                    || materialize_accumulator_fn(acc, emit_to),
                )?);
            }
            Ok::<_, datafusion_common::DataFusionError>(output)
        })?;

        let batch = RecordBatch::try_new(Arc::clone(&self.output_schema), output)?;
        debug_assert!(batch.num_rows() > 0);

        Ok(batch)
    }
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
    fn build_groups_info_first_batch() -> Result<()> {
        let info = build_groups_info(&[0, 0, 1, 2, 3, 3], 0)?;
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
    fn build_groups_info_continues_previous_group() -> Result<()> {
        // 3 groups already interned, the batch starts with the last one (index 2)
        let info = build_groups_info(&[2, 2, 3], 3)?;
        assert_eq!(ranges(&info), vec![(0, 2, true), (2, 3, false)]);
        assert_eq!(info.total_number_of_groups(), 4);
        assert_eq!(info.num_new_groups(), 1);

        // the batch starts a new group
        let info = build_groups_info(&[3, 4, 5], 3)?;
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
    fn build_groups_info_rejects_non_contiguous_groups() {
        // revisits a group
        assert!(build_groups_info(&[0, 1, 0], 0).is_err());
        // skips a group index
        assert!(build_groups_info(&[0, 2], 0).is_err());
        // starts with a group that is neither the last one nor a new one
        assert!(build_groups_info(&[0, 1], 2).is_err());
    }
}
