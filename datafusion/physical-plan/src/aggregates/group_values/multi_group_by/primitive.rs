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

use crate::aggregates::group_values::multi_group_by::{
    GroupColumn, Nulls, nulls_equal_to,
};
use arrow::array::ArrowNativeTypeOp;
use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, BooleanBufferBuilder, PrimitiveArray,
    cast::AsArray,
};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::DataType;
use arrow::util::bit_util::apply_bitwise_binary_op;
use datafusion_common::Result;
use datafusion_common::utils::split_vec_min_alloc;
use datafusion_expr_common::groups_accumulator::BlocksIndex;
use datafusion_functions_aggregate_common::blocked_helpers::{
    BlockedNullsBuilder, BlockedVecBuilder,
};
use std::iter;
use std::sync::Arc;

/// An implementation of [`GroupColumn`] for primitive values
///
/// Optimized to skip null buffer construction if the input is known to be non nullable
///
/// # Template parameters
///
/// `T`: the native Rust type that stores the data
/// `NULLABLE`: if the data can contain any nulls
#[derive(Debug)]
pub struct PrimitiveGroupValueBuilder<
    const FIXED_BLOCK_SIZING: bool,
    T: ArrowPrimitiveType,
    const NULLABLE: bool,
> {
    data_type: DataType,
    group_values: BlockedVecBuilder<FIXED_BLOCK_SIZING, T::Native>,
    nulls: BlockedNullsBuilder<FIXED_BLOCK_SIZING>,
}

impl<const FIXED_BLOCK_SIZING: bool, T, const NULLABLE: bool>
    PrimitiveGroupValueBuilder<FIXED_BLOCK_SIZING, T, NULLABLE>
where
    T: ArrowPrimitiveType,
{
    /// Create a new `PrimitiveGroupValueBuilder`
    pub fn new(data_type: DataType, block_size: usize) -> Self {
        Self {
            data_type,
            group_values: BlockedVecBuilder::new(block_size),
            nulls: BlockedNullsBuilder::new(block_size),
        }
    }

    fn vectorized_equal_to_non_nullable(
        &self,
        lhs_rows: &[BlocksIndex],
        array: &ArrayRef,
        rhs_rows: &[usize],
        equal_to_results: &mut BooleanBufferBuilder,
    ) {
        assert!(
            !NULLABLE || (array.null_count() == 0 && !self.nulls.might_have_nulls()),
            "called with nullable input"
        );
        let array_values = array.as_primitive::<T>().values();
        let n = lhs_rows.len();

        // Build a packed comparison bitmask, then AND it into equal_to_results
        let num_bytes = n.div_ceil(8);
        let mut cmp_buf = vec![0u8; num_bytes];

        for (i, (&lhs_row, &rhs_row)) in lhs_rows.iter().zip(rhs_rows.iter()).enumerate()
        {
            let left = self.group_values[lhs_row];
            let right = if cfg!(debug_assertions) {
                array_values[rhs_row]
            } else {
                unsafe { *array_values.get_unchecked(rhs_row) }
            };
            if left.is_eq(right) {
                cmp_buf[i / 8] |= 1 << (i % 8);
            }
        }

        // AND the comparison result into the existing equal_to_results bitmask
        apply_bitwise_binary_op(
            equal_to_results.as_slice_mut(),
            0,
            &cmp_buf,
            0,
            n,
            |a, b| a & b,
        );
    }

    pub fn vectorized_equal_nullable(
        &self,
        lhs_rows: &[BlocksIndex],
        array: &ArrayRef,
        rhs_rows: &[usize],
        equal_to_results: &mut BooleanBufferBuilder,
    ) {
        assert!(NULLABLE, "called with non-nullable input");
        let array = array.as_primitive::<T>();

        for (idx, (&lhs_row, &rhs_row)) in
            lhs_rows.iter().zip(rhs_rows.iter()).enumerate()
        {
            if !equal_to_results.get_bit(idx) {
                continue;
            }

            let exist_null = self.nulls.is_null(lhs_row);
            let input_null = array.is_null(rhs_row);
            if let Some(result) = nulls_equal_to(exist_null, input_null) {
                if !result {
                    equal_to_results.set_bit(idx, false);
                }
                continue;
            }

            if !self.group_values[lhs_row].is_eq(array.value(rhs_row)) {
                equal_to_results.set_bit(idx, false);
            }
        }
    }
}

impl<const FIXED_BLOCK_SIZING: bool, T: ArrowPrimitiveType, const NULLABLE: bool>
    GroupColumn<FIXED_BLOCK_SIZING>
    for PrimitiveGroupValueBuilder<FIXED_BLOCK_SIZING, T, NULLABLE>
{
    fn equal_to(&self, lhs_row: BlocksIndex, array: &ArrayRef, rhs_row: usize) -> bool {
        // Perf: skip null check (by short circuit) if input is not nullable
        if NULLABLE {
            let exist_null = self.nulls.is_null(lhs_row);
            let input_null = array.is_null(rhs_row);
            if let Some(result) = nulls_equal_to(exist_null, input_null) {
                return result;
            }
            // Otherwise, we need to check their values
        }

        self.group_values[lhs_row]
            .is_eq(array.as_primitive::<T>().value(rhs_row).canonicalize())
    }

    fn append_val(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        // Perf: skip null check if input can't have nulls
        if NULLABLE {
            if array.is_null(row) {
                self.nulls.push_null();
                self.group_values.push(T::default_value());
            } else {
                self.nulls.push_non_null();
                self.group_values
                    .push(array.as_primitive::<T>().value(row).canonicalize());
            }
        } else {
            self.group_values
                .push(array.as_primitive::<T>().value(row).canonicalize());
        }

        Ok(())
    }

    fn vectorized_equal_to(
        &self,
        lhs_rows: &[BlocksIndex],
        array: &ArrayRef,
        rhs_rows: &[usize],
        equal_to_results: &mut BooleanBufferBuilder,
    ) {
        if !NULLABLE || (array.null_count() == 0 && !self.nulls.might_have_nulls()) {
            self.vectorized_equal_to_non_nullable(
                lhs_rows,
                array,
                rhs_rows,
                equal_to_results,
            );
        } else {
            self.vectorized_equal_nullable(lhs_rows, array, rhs_rows, equal_to_results);
        }
    }

    fn vectorized_append(&mut self, array: &ArrayRef, rows: &[usize]) -> Result<()> {
        let arr = array.as_primitive::<T>();

        let null_count = array.null_count();
        let num_rows = array.len();
        let all_null_or_non_null = if null_count == 0 {
            Nulls::None
        } else if null_count == num_rows {
            Nulls::All
        } else {
            Nulls::Some
        };

        match (NULLABLE, all_null_or_non_null) {
            (true, Nulls::Some) => {
                for &row in rows {
                    if array.is_null(row) {
                        self.nulls.push_null();
                        self.group_values.push(T::default_value());
                    } else {
                        self.nulls.push_non_null();
                        self.group_values.push(arr.value(row).canonicalize());
                    }
                }
            }

            (true, Nulls::None) => {
                self.nulls.push_n(rows.len(), true);
                for &row in rows {
                    self.group_values.push(arr.value(row).canonicalize());
                }
            }

            (true, Nulls::All) => {
                self.nulls.push_n(rows.len(), false);
                self.group_values.push_default_n(rows.len());
            }

            (false, _) => {
                for &row in rows {
                    self.group_values.push(arr.value(row).canonicalize());
                }
            }
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.group_values.len()
    }

    fn size(&self) -> usize {
        self.group_values.allocated_size() + self.nulls.allocated_size()
    }

    fn take_block(&mut self) -> Option<ArrayRef> {
        let values_block = self.group_values.take_block_finished()?;
        let nulls = if NULLABLE {
            self.nulls.take_block()?
        } else {
            None
        };

        Some(Arc::new(
            PrimitiveArray::<T>::new(values_block, nulls)
                .with_data_type(self.data_type.clone()),
        ))
    }

    fn start_new_block(&mut self) {
        self.group_values.start_new_block();
        if NULLABLE {
            self.nulls.start_new_block();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow::array::{
        ArrayRef, BooleanBufferBuilder, Float32Array, Int64Array, NullBufferBuilder,
    };
    use arrow::datatypes::{DataType, Float32Type, Int64Type};

    fn make_true_buffer(n: usize) -> BooleanBufferBuilder {
        let mut buf = BooleanBufferBuilder::new(n);
        buf.append_n(n, true);
        buf
    }

    fn to_vec(buf: &BooleanBufferBuilder) -> Vec<bool> {
        (0..buf.len()).map(|i| buf.get_bit(i)).collect()
    }

    #[test]
    fn test_nullable_primitive_equal_to() {
        let append =
            |builder: &mut PrimitiveGroupValueBuilder<false, Float32Type, true>,
             builder_array: &ArrayRef,
             append_rows: &[usize]| {
                for &index in append_rows {
                    builder.append_val(builder_array, index).unwrap();
                }
            };

        let equal_to =
            |builder: &PrimitiveGroupValueBuilder<false, Float32Type, true>,
             lhs_rows: &[BlocksIndex],
             input_array: &ArrayRef,
             rhs_rows: &[usize],
             equal_to_results: &mut BooleanBufferBuilder| {
                let iter = lhs_rows.iter().zip(rhs_rows.iter());
                for (idx, (&lhs_row, &rhs_row)) in iter.enumerate() {
                    equal_to_results
                        .set_bit(idx, builder.equal_to(lhs_row, input_array, rhs_row));
                }
            };

        test_nullable_primitive_equal_to_internal(append, equal_to);
    }

    #[test]
    fn test_nullable_primitive_vectorized_equal_to() {
        let append =
            |builder: &mut PrimitiveGroupValueBuilder<false, Float32Type, true>,
             builder_array: &ArrayRef,
             append_rows: &[usize]| {
                builder
                    .vectorized_append(builder_array, append_rows)
                    .unwrap();
            };

        let equal_to =
            |builder: &PrimitiveGroupValueBuilder<false, Float32Type, true>,
             lhs_rows: &[BlocksIndex],
             input_array: &ArrayRef,
             rhs_rows: &[usize],
             equal_to_results: &mut BooleanBufferBuilder| {
                builder.vectorized_equal_to(
                    lhs_rows,
                    input_array,
                    rhs_rows,
                    equal_to_results,
                );
            };

        test_nullable_primitive_equal_to_internal(append, equal_to);
    }

    fn test_nullable_primitive_equal_to_internal<A, E>(mut append: A, mut equal_to: E)
    where
        A: FnMut(
            &mut PrimitiveGroupValueBuilder<false, Float32Type, true>,
            &ArrayRef,
            &[usize],
        ),
        E: FnMut(
            &PrimitiveGroupValueBuilder<false, Float32Type, true>,
            &[BlocksIndex],
            &ArrayRef,
            &[usize],
            &mut BooleanBufferBuilder,
        ),
    {
        // Will cover such cases:
        //   - exist null, input not null
        //   - exist null, input null; values not equal
        //   - exist null, input null; values equal
        //   - exist not null, input null
        //   - exist not null, input not null; values not equal
        //   - exist not null, input not null; values equal

        // Define PrimitiveGroupValueBuilder
        let mut builder = PrimitiveGroupValueBuilder::<false, Float32Type, true>::new(
            DataType::Float32,
            0,
        );
        let builder_array = Arc::new(Float32Array::from(vec![
            None,
            None,
            None,
            Some(1.0),
            Some(2.0),
            Some(f32::NAN),
            Some(3.0),
        ])) as ArrayRef;
        append(&mut builder, &builder_array, &[0, 1, 2, 3, 4, 5, 6]);

        // Define input array
        let (_, values, _nulls) = Float32Array::from(vec![
            Some(1.0),
            Some(2.0),
            None,
            Some(1.0),
            None,
            Some(f32::NAN),
            None,
        ])
        .into_parts();

        // explicitly build a null buffer where one of the null values also happens to match
        let mut nulls = NullBufferBuilder::new(6);
        nulls.append_non_null();
        nulls.append_null();
        nulls.append_null();
        nulls.append_non_null();
        nulls.append_null();
        nulls.append_non_null();
        nulls.append_null();
        let input_array = Arc::new(Float32Array::new(values, nulls.finish())) as ArrayRef;

        // Check
        let mut equal_to_results = make_true_buffer(builder.len());
        equal_to(
            &builder,
            &[0, 1, 2, 3, 4, 5, 6].map(BlocksIndex::new_in_first_block),
            &input_array,
            &[0, 1, 2, 3, 4, 5, 6],
            &mut equal_to_results,
        );
        let results = to_vec(&equal_to_results);

        assert!(!results[0]);
        assert!(results[1]);
        assert!(results[2]);
        assert!(results[3]);
        assert!(!results[4]);
        assert!(results[5]);
        assert!(!results[6]);
    }

    #[test]
    fn test_not_nullable_primitive_equal_to() {
        let append =
            |builder: &mut PrimitiveGroupValueBuilder<false, Int64Type, false>,
             builder_array: &ArrayRef,
             append_rows: &[usize]| {
                for &index in append_rows {
                    builder.append_val(builder_array, index).unwrap();
                }
            };

        let equal_to =
            |builder: &PrimitiveGroupValueBuilder<false, Int64Type, false>,
             lhs_rows: &[BlocksIndex],
             input_array: &ArrayRef,
             rhs_rows: &[usize],
             equal_to_results: &mut BooleanBufferBuilder| {
                let iter = lhs_rows.iter().zip(rhs_rows.iter());
                for (idx, (&lhs_row, &rhs_row)) in iter.enumerate() {
                    equal_to_results
                        .set_bit(idx, builder.equal_to(lhs_row, input_array, rhs_row));
                }
            };

        test_not_nullable_primitive_equal_to_internal(append, equal_to);
    }

    #[test]
    fn test_not_nullable_primitive_vectorized_equal_to() {
        let append =
            |builder: &mut PrimitiveGroupValueBuilder<false, Int64Type, false>,
             builder_array: &ArrayRef,
             append_rows: &[usize]| {
                builder
                    .vectorized_append(builder_array, append_rows)
                    .unwrap();
            };

        let equal_to =
            |builder: &PrimitiveGroupValueBuilder<false, Int64Type, false>,
             lhs_rows: &[BlocksIndex],
             input_array: &ArrayRef,
             rhs_rows: &[usize],
             equal_to_results: &mut BooleanBufferBuilder| {
                builder.vectorized_equal_to(
                    lhs_rows,
                    input_array,
                    rhs_rows,
                    equal_to_results,
                );
            };

        test_not_nullable_primitive_equal_to_internal(append, equal_to);
    }

    fn test_not_nullable_primitive_equal_to_internal<A, E>(mut append: A, mut equal_to: E)
    where
        A: FnMut(
            &mut PrimitiveGroupValueBuilder<false, Int64Type, false>,
            &ArrayRef,
            &[usize],
        ),
        E: FnMut(
            &PrimitiveGroupValueBuilder<false, Int64Type, false>,
            &[BlocksIndex],
            &ArrayRef,
            &[usize],
            &mut BooleanBufferBuilder,
        ),
    {
        // Will cover such cases:
        //   - values equal
        //   - values not equal

        // Define PrimitiveGroupValueBuilder
        let mut builder = PrimitiveGroupValueBuilder::<false, Int64Type, false>::new(
            DataType::Int64,
            0,
        );
        let builder_array =
            Arc::new(Int64Array::from(vec![Some(0), Some(1)])) as ArrayRef;
        append(&mut builder, &builder_array, &[0, 1]);

        // Define input array
        let input_array = Arc::new(Int64Array::from(vec![Some(0), Some(2)])) as ArrayRef;

        // Check
        let mut equal_to_results = make_true_buffer(builder.len());
        equal_to(
            &builder,
            &[0, 1].map(BlocksIndex::new_in_first_block),
            &input_array,
            &[0, 1],
            &mut equal_to_results,
        );
        let results = to_vec(&equal_to_results);

        assert!(results[0]);
        assert!(!results[1]);
    }

    #[test]
    fn test_nullable_primitive_vectorized_operation_special_case() {
        // Test the special `all nulls` or `not nulls` input array case
        // for vectorized append and equal to

        let mut builder =
            PrimitiveGroupValueBuilder::<false, Int64Type, true>::new(DataType::Int64, 0);

        // All nulls input array
        let all_nulls_input_array = Arc::new(Int64Array::from(vec![
            Option::<i64>::None,
            None,
            None,
            None,
            None,
        ])) as _;
        builder
            .vectorized_append(&all_nulls_input_array, &[0, 1, 2, 3, 4])
            .unwrap();

        let mut equal_to_results = make_true_buffer(all_nulls_input_array.len());
        builder.vectorized_equal_to(
            &[0, 1, 2, 3, 4].map(BlocksIndex::new_in_first_block),
            &all_nulls_input_array,
            &[0, 1, 2, 3, 4],
            &mut equal_to_results,
        );
        let results = to_vec(&equal_to_results);

        assert!(results[0]);
        assert!(results[1]);
        assert!(results[2]);
        assert!(results[3]);
        assert!(results[4]);

        // All not nulls input array
        let all_not_nulls_input_array = Arc::new(Int64Array::from(vec![
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
        ])) as _;
        builder
            .vectorized_append(&all_not_nulls_input_array, &[0, 1, 2, 3, 4])
            .unwrap();

        let mut equal_to_results = make_true_buffer(all_not_nulls_input_array.len());
        builder.vectorized_equal_to(
            &[5, 6, 7, 8, 9].map(BlocksIndex::new_in_first_block),
            &all_not_nulls_input_array,
            &[0, 1, 2, 3, 4],
            &mut equal_to_results,
        );
        let results = to_vec(&equal_to_results);

        assert!(results[0]);
        assert!(results[1]);
        assert!(results[2]);
        assert!(results[3]);
        assert!(results[4]);
    }

    #[test]
    fn test_primitive_take_n() {
        // drain branch: n * 2 <= len
        let mut builder =
            PrimitiveGroupValueBuilder::<false, Int64Type, true>::new(DataType::Int64, 0);
        let array = Arc::new(Int64Array::from(vec![
            Some(10),
            None,
            Some(30),
            Some(40),
            Some(50),
        ])) as ArrayRef;
        for i in 0..5 {
            builder.append_val(&array, i).unwrap();
        }
        // len=5, n=2, n*2=4 <= 5  →  drain branch
        let out = builder.take_n(2);
        let expected = Arc::new(Int64Array::from(vec![Some(10), None])) as ArrayRef;
        assert_eq!(&out, &expected);
        // remaining: [30, 40, 50]
        assert_eq!(builder.len(), 3);

        // split_off branch: remaining < n  (len=3, n=2, n*2=4 > 3)
        let out2 = builder.take_n(2);
        let expected2 = Arc::new(Int64Array::from(vec![Some(30), Some(40)])) as ArrayRef;
        assert_eq!(&out2, &expected2);
        // remaining: [50]
        assert_eq!(builder.len(), 1);

        // take the last element
        let out3 = builder.take_n(1);
        let expected3 = Arc::new(Int64Array::from(vec![Some(50)])) as ArrayRef;
        assert_eq!(&out3, &expected3);
        assert_eq!(builder.len(), 0);
    }
}
