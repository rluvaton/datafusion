use super::blocked_custom_input_builder::{
    Block, BlockProvider, BlockProviderFinish, BlockWithSlice, BlockedCustomInputBuilder,
};
use arrow::buffer::ScalarBuffer;
use arrow::datatypes::ArrowNativeType;
use datafusion_common::utils::proxy::VecAllocExt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct BlockedVecBuilder<const FIXED_BLOCK_SIZING: bool, T>(
    BlockedCustomInputBuilder<FIXED_BLOCK_SIZING, VecBlockProvider<T>>,
);

impl<const FIXED_BLOCK_SIZING: bool, T> BlockedVecBuilder<FIXED_BLOCK_SIZING, T> {
    pub fn new(block_size: usize) -> Self {
        BlockedVecBuilder(BlockedCustomInputBuilder::new(
            block_size,
            VecBlockProvider::<T>::default(),
        ))
    }
}

impl<const FIXED_BLOCK_SIZING: bool, T> Deref
    for BlockedVecBuilder<FIXED_BLOCK_SIZING, T>
{
    type Target = BlockedCustomInputBuilder<FIXED_BLOCK_SIZING, VecBlockProvider<T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const FIXED_BLOCK_SIZING: bool, T> DerefMut
    for BlockedVecBuilder<FIXED_BLOCK_SIZING, T>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug)]
pub struct VecBlockProvider<T>(PhantomData<T>);

impl<T> Default for VecBlockProvider<T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<T> BlockProvider for VecBlockProvider<T> {
    type Block = Vec<T>;

    fn new_block(&self) -> Self::Block {
        vec![]
    }

    fn allocated_size(&self) -> usize {
        0
    }
}

impl<T: ArrowNativeType> BlockProviderFinish for VecBlockProvider<T> {
    type FinishedBlock = ScalarBuffer<T>;

    fn finish(&self, block: Self::Block) -> Self::FinishedBlock {
        ScalarBuffer::from(block)
    }
}

impl<T> Block for Vec<T> {
    type Item = T;

    fn allocated_size(&self) -> usize {
        VecAllocExt::allocated_size(self)
    }

    fn push(&mut self, item: Self::Item) {
        Self::push(self, item)
    }

    fn extend(&mut self, iter: impl Iterator<Item = Self::Item>) {
        Extend::extend(self, iter)
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn is_empty(&self) -> bool {
        Vec::is_empty(self)
    }
}

impl<T: Clone> BlockWithSlice for Vec<T> {
    fn extend_from_slice(&mut self, slice: &[Self::Item]) {
        Vec::extend_from_slice(self, slice)
    }

    fn append_n(&mut self, item: Self::Item, n: usize) {
        self.resize(self.len() + n, item)
    }
}
