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

#[macro_use]
extern crate criterion;
extern crate arrow;
extern crate datafusion;

mod data_utils;

use std::any::Any;
use std::fmt::{Display, Formatter};
use crate::criterion::Criterion;
use data_utils::create_table_provider;
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use parking_lot::Mutex;
use std::hint::black_box;
use std::sync::Arc;
use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::datatypes::{Int32Type, Int64Type, Int8Type};
use arrow::record_batch::RecordBatch;
use arrow::util::bench_util::{create_primitive_array_with_seed, create_string_array_with_len_range_and_prefix_and_seed};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use itertools::Itertools;
use tokio::runtime::Runtime;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::memory::LazyMemoryExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::test::TestMemoryExec;

#[expect(clippy::needless_pass_by_value)]
fn query(ctx: Arc<Mutex<SessionContext>>, rt: &Runtime, sql: &str) {
    let df = rt.block_on(ctx.lock().sql(sql)).unwrap();
    black_box(rt.block_on(df.collect()).unwrap());
}

fn create_context(
    partitions_len: usize,
    array_len: usize,
    batch_size: usize,
) -> Result<Arc<Mutex<SessionContext>>> {
    let ctx = SessionContext::new();
    let provider = create_table_provider(partitions_len, array_len, batch_size)?;
    ctx.register_table("t", provider)?;
    Ok(Arc::new(Mutex::new(ctx)))
}

enum InputSchemaType<'a> {
    Provided {
        schema: &'a SchemaRef,
    },
    FromOutput {
        all_nullable: bool,
    },
}

fn get_medium_amount_and_types_of_batch_without_nesting(
    batch_size: usize,
    seed: usize,
    input_schema_type: InputSchemaType<'_>,
) -> RecordBatch {
    let mut seed = seed as u64;
    let mut cols: Vec<ArrayRef> = vec![];

    for nulls in [0.0, 0.1, 0.2, 0.5] {
        seed += 1;
        cols.push(Arc::new(create_primitive_array_with_seed::<Int8Type>(
            batch_size, nulls, seed,
        )) as ArrayRef);
    }

    for nulls in [0.0, 0.1, 0.2, 0.5] {
        seed += 1;
        cols.push(Arc::new(create_primitive_array_with_seed::<Int32Type>(
            batch_size, nulls, seed,
        )) as ArrayRef);
    }

    for nulls in [0.0, 0.1, 0.2, 0.5] {
        seed += 1;
        cols.push(Arc::new(create_primitive_array_with_seed::<Int64Type>(
            batch_size, nulls, seed,
        )) as ArrayRef);
    }

    for _ in 0..10 {
        seed += 1;
        cols.push(Arc::new(create_primitive_array_with_seed::<Int64Type>(
            batch_size, 0.0, seed,
        )) as ArrayRef);
    }

    for nulls in [0.0, 0.1, 0.2, 0.5] {
        seed += 1;
        cols.push(Arc::new(
            create_string_array_with_len_range_and_prefix_and_seed::<i32>(
                batch_size, nulls, 0, 50, "", seed,
            ),
        ));
    }

    for _ in 0..3 {
        seed += 1;
        cols.push(Arc::new(
            create_string_array_with_len_range_and_prefix_and_seed::<i32>(
                batch_size, 0.0, 0, 10, "", seed,
            ),
        ));
    }
    for _ in 0..3 {
        seed += 1;
        cols.push(Arc::new(
            create_string_array_with_len_range_and_prefix_and_seed::<i32>(
                batch_size, 0.0, 10, 20, "", seed,
            ),
        ));
    }
    for _ in 0..3 {
        seed += 1;
        cols.push(Arc::new(
            create_string_array_with_len_range_and_prefix_and_seed::<i32>(
                batch_size, 0.0, 20, 30, "", seed,
            ),
        ));
    }

    for _ in 0..10 {
        seed += 1;
        cols.push(Arc::new(create_primitive_array_with_seed::<Int64Type>(
            batch_size, 0.0, seed,
        )) as ArrayRef);
    }

    let schema = match input_schema_type {
        InputSchemaType::Provided { schema } => {
            Arc::clone(schema)
        }
        InputSchemaType::FromOutput { all_nullable } => {
            let schema = Schema::new(cols.iter().enumerate().map(|(index, c)| {
                let nullable = if all_nullable { true } else { c.logical_null_count() > 0 };
                Field::new(&format!("col_{}", index), c.data_type().clone(), nullable)
            }).collect());
            Arc::new(schema)
        }
    };

    RecordBatch::try_new(
        schema,
        cols
    ).unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    // Pure repartition benchmark

    {
        let batch_size = 8192;
        let number_of_batches = 10;
        let batches = {
            let batch1 = get_medium_amount_and_types_of_batch_without_nesting(batch_size, 0, InputSchemaType::FromOutput { all_nullable: false });
            let schema = batch1.schema();
            let mut batches = vec![batch1];
            for index in 1..number_of_batches {
                let seed = index * 1000;
                let batch = get_medium_amount_and_types_of_batch_without_nesting(batch_size, seed, InputSchemaType::Provided { schema: &schema });
                batches.push(batch);
            }

            batches
        };

        run_benchmarks(
            c,
            format!("repartition {number_of_batches} batches of {batch_size} rows with ~50 non nested columns | 1 input partitions").as_str(),
            // Single input
            &[batches]
        )
    }
}

fn run_benchmarks(criterion: &mut Criterion, name: &str, partitions: &[Vec<RecordBatch>]) {
    for num_of_partitions in [2, 4, 8, 50, 500, 1000, 1500] {
        do_bench(DoBenchArgs {
            criterion,
            name: format!("{name} {num_of_partitions} output partitions").as_str(),
            input_partitions: partitions,
            number_of_output_partitions: num_of_partitions,
            partitioning: Partitioning::Hash(vec![Arc::new(DummyHashExpression::new(partitions))], num_of_partitions),
        })
    }

}

struct DoBenchArgs<'a> {
    criterion: &'a mut Criterion,
    name: &'a str,
    input_partitions: &'a [Vec<RecordBatch>],
    number_of_output_partitions: usize,
    partitioning: Partitioning,
}

fn do_bench(args: DoBenchArgs<'_>) {
    let first_schema = args.input_partitions.iter().flatten().next().expect("No batches").schema();
    let source = TestMemoryExec::try_new_exec(args.input_partitions, first_schema, None).unwrap();

    let repartition_exec = RepartitionExec::try_new(
        Arc::clone(&source),
        args.partitioning.clone(),
    )
      .unwrap();

    let mut rt = Runtime::new().unwrap();
    let ctx = SessionContext::new();
    let task_ctx = ctx.task_ctx();

    args.criterion.bench_function(args.name, |b| {
        b.iter(|| {
            let mut handles = Vec::with_capacity(args.number_of_output_partitions);
            for partition in 0..args.number_of_output_partitions {
                let exec = repartition_exec.clone();
                let task_ctx = task_ctx.clone();
                handles.push(rt.spawn(async move {
                    let mut stream = exec.execute(partition, task_ctx).unwrap();
                    while let Some(_batch) = stream.next().await.transpose().unwrap() {
                        // consume the batch
                    }
                }));
            }
            rt.block_on(async {
                for handle in handles {
                    handle.await.unwrap();
                }
            });
        })
    });
}

/// Expression that it's sole purpose is to output an array with no duplicates
/// the fastest way possible to avoid counting
#[derive(Debug, Hash, PartialEq, Eq)]
struct DummyHashExpression {
    output: ArrayRef,
}

impl DummyHashExpression {
    fn new(partitions: &[Vec<RecordBatch>]) -> Self {
        let largest_batch_size = partitions.iter().flatten().map(|p| p.num_rows()).max().unwrap();

        let array = UInt32Array::from_iter_values(
            (0..largest_batch_size as u32)
        );

        Self {
            output:Arc::new(array)
        }
    }
}

impl Display for DummyHashExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "DummyHashExpression")
    }
}

impl PhysicalExpr for DummyHashExpression {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> DataType {
        DataType::UInt32
    }

    fn nullable(&self, input_schema: &Schema) -> bool {
        false
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        assert!(batch.num_rows() <= self.output.len());

        Ok(self.output.slice(0, batch.num_rows()))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![]
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Arc<dyn PhysicalExpr>>) -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::clone(&self))
    }

    fn fmt_sql(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
