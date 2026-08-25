use std::ops::{Deref, DerefMut};
use super::blocked_custom_input_builder_with_lifetime::{BlockWithLifetime, BlockWithLifetimeProvider, BlockedCustomInputBuilderWithLifetime};
use arrow::row::{RowConverter, Rows};

#[derive(Debug)]
pub struct BlockedRowsBuilder<const FIXED_BLOCK_SIZING: bool>(
  BlockedCustomInputBuilderWithLifetime<FIXED_BLOCK_SIZING, RowsBlockProvider>
);

impl<const FIXED_BLOCK_SIZING: bool> BlockedRowsBuilder<FIXED_BLOCK_SIZING> {
  pub fn new(block_size: usize, row_converter: RowConverter) -> Self {
    let block_provider = RowsBlockProvider {
      row_converter,
    };
    
    Self(BlockedCustomInputBuilderWithLifetime::new(block_size, block_provider))
  }
}

impl<const FIXED_BLOCK_SIZING: bool> Deref for BlockedRowsBuilder<FIXED_BLOCK_SIZING> {
  type Target = BlockedCustomInputBuilderWithLifetime<FIXED_BLOCK_SIZING, RowsBlockProvider>;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl<const FIXED_BLOCK_SIZING: bool> DerefMut for BlockedRowsBuilder<FIXED_BLOCK_SIZING> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.0
  }
}

#[derive(Debug)]
pub struct RowsBlockProvider {
  row_converter: arrow::row::RowConverter,
}

impl RowsBlockProvider {
  pub fn new(row_converter: RowConverter) -> Self {
    Self {
      row_converter
    }
  }

  pub fn row_converter(&self) -> &RowConverter {
    &self.row_converter
  }
}

impl BlockWithLifetimeProvider for RowsBlockProvider {
  type Block = Rows;

  fn new_block(&self) -> Self::Block {
    self.row_converter.empty_rows(0, 0)
  }
}

impl BlockWithLifetime for Rows {
  type Item<'a> = arrow::row::Row<'a>;

  fn allocated_size(&self) -> usize {
    self.size()
  }

  fn push(&mut self, item: Self::Item<'_>) {
    self.push(item)
  }

  fn extend<'a>(&mut self, iter: impl Iterator<Item=Self::Item<'a>>) {
    for item in iter {
      self.push(item)
    }
  }

  fn len(&self) -> usize {
    self.num_rows()
  }

  fn is_empty(&self) -> bool {
    self.num_rows() == 0
  }

  fn index<'a>(&'a self, index: usize) -> Self::Item<'a> {
    self.row(index)
  }
}


