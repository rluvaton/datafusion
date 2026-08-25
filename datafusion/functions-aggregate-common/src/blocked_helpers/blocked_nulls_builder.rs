use arrow::array::NullBufferBuilder;
use arrow::buffer::NullBuffer;
use std::collections::VecDeque;
use std::ops::Index;
use datafusion_expr_common::groups_accumulator::BlocksIndex;

#[derive(Debug)]
pub struct BlockedNullsBuilder<const FIXED_BLOCK_SIZING: bool> {
  /// Using `VecDeque` so we can remove the first block and reclaim memory
  blocks: VecDeque<NullBufferBuilder>,

  /// The size of each block
  block_size: usize,

  /// The index of the current block
  current_block_index: usize,

  len: usize,

  allocated_size: usize,
}

impl<const FIXED_BLOCK_SIZING: bool> BlockedNullsBuilder<FIXED_BLOCK_SIZING> {
  pub fn new(block_size: usize) -> Self {
    if FIXED_BLOCK_SIZING {
      assert_ne!(block_size, 0, "block size must be greater than 0");
    }

    let blocks = VecDeque::from(vec![NullBufferBuilder::new(block_size)]);

    let allocated_size = blocks.capacity() * size_of::<NullBufferBuilder>() + blocks[0].allocated_size();

    BlockedNullsBuilder {
      blocks,
      block_size,
      current_block_index: 0,
      len: 0,
      allocated_size,
    }
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn allocated_size(&self) -> usize {
    self.allocated_size
  }

  pub(super) fn block_size(&self) -> usize {
    assert!(FIXED_BLOCK_SIZING, "block size is only available for manual block");
    self.block_size
  }

  pub fn start_new_block(&mut self) {
    self.current_block_index += 1;
    let capacity_before = self.blocks.capacity();
    let new_block = NullBufferBuilder::new(self.block_size);
    self.allocated_size += new_block.allocated_size();
    self.blocks.push_back(new_block);
    self.allocated_size += (self.blocks.capacity() - capacity_before) * size_of::<NullBufferBuilder>();
  }

  pub(crate) fn reserve_blocks(&mut self, n: usize) {
    let capacity_before = self.blocks.capacity();
    self.blocks.reserve(n);
    self.allocated_size += (self.blocks.capacity() - capacity_before) * size_of::<NullBufferBuilder>();
  }

  /// Extend the null buffer
  /// when it is guaranteed that len is less than the remaining block size
  pub(super) fn extends_from_null_buffer_in_current_block(&mut self, null_buffer: &NullBuffer) {
    assert_ne!(null_buffer.len(), 0);
    if FIXED_BLOCK_SIZING {
      assert!(self.current_block_remaining_len() >= null_buffer.len(), "the amount to add exceed the current block size");
    }
    if null_buffer.null_count() == 0 {
      self.push_n_within_block(null_buffer.len(), true);
      return;
    }

    let block = &mut self.blocks[self.current_block_index];
    let prev_block_size = block.len();

    let size_before = block.allocated_size();

    // Do fast large copy
    block.append_buffer(null_buffer);

    self.allocated_size += (block.allocated_size() - size_before);

    let added_items = block.len() - prev_block_size;

    self.len += added_items;

    if FIXED_BLOCK_SIZING && block.len() == self.block_size {
      self.start_new_block();
    }
  }

  /// Extend the length from the current offsets
  pub fn extends_from_null_buffer(&mut self, null_buffer: &NullBuffer) {
    if null_buffer.null_count() == 0 {
      self.push_n(null_buffer.len(), true);
      return;
    }

    if !FIXED_BLOCK_SIZING {
      self.extends_from_null_buffer_in_current_block(null_buffer);
      return;
    }

    let number_of_blocks_to_reserve = null_buffer.len().saturating_sub(self.current_block_remaining_len()).div_ceil(self.block_size);
    self.reserve_blocks(number_of_blocks_to_reserve);

    let mut len = null_buffer.len();
    let mut index = 0;

    while len > 0 {
      let remaining_in_current_block = self.current_block_remaining_len();

      let to_add = remaining_in_current_block.min(len);
      if to_add == len {

        // Avoid slice which does null counting
        if index == 0 {
          self.extends_from_null_buffer_in_current_block(null_buffer);
        } else {
          self.extends_from_null_buffer_in_current_block(&null_buffer.slice(index, len));
        }

        break;
      }

      let null_section = null_buffer.slice(index, to_add);
      index += to_add;

      self.extends_from_null_buffer_in_current_block(&null_section);
    }
  }

  /// Extend the length from the current offsets in the indexes
  /// when it is guaranteed that len is less than the remaining block size
  fn extends_from_null_buffer_in_indexes_in_current_block(&mut self, null_buffer: &NullBuffer, indexes: &[usize]) -> usize {
    assert_ne!(null_buffer.len(), 0);
    assert_ne!(indexes.len(), 0);

    if FIXED_BLOCK_SIZING {
      assert!(self.current_block_remaining_len() >= indexes.len(), "the amount to add exceed the current block size");
    }

    let block = &mut self.blocks[self.current_block_index];
    let prev_block_size = block.len();


    let size_before = block.allocated_size();

    // TODO - reserve in block and set each byte without extra checks
    for &index_to_copy in indexes {
      block.append(null_buffer.is_valid(index_to_copy));
    }

    self.allocated_size += (block.allocated_size() - size_before);
    let added_items = block.len() - prev_block_size;

    self.len += added_items;

    if FIXED_BLOCK_SIZING && block.len() == self.block_size {
      self.start_new_block();
    }

    added_items
  }

  /// Extend the length from the current offsets
  pub fn extends_from_null_buffer_in_indexes(&mut self, null_buffer: &NullBuffer, mut indexes: &[usize]) {
    if !FIXED_BLOCK_SIZING {
      self.extends_from_null_buffer_in_indexes_in_current_block(null_buffer, indexes);
      return;
    }

    let number_of_blocks_to_reserve = indexes.len().saturating_sub(self.current_block_remaining_len()).div_ceil(self.block_size);
    self.reserve_blocks(number_of_blocks_to_reserve);

    while indexes.len() > 0 {
      let remaining_in_current_block = self.current_block_remaining_len();

      let to_add = remaining_in_current_block.min(indexes.len());
      let (to_copy, left) = indexes.split_at(to_add);
      indexes = left;

      self.extends_from_null_buffer_in_indexes_in_current_block(null_buffer, to_copy);
    }
  }

  fn current_block_remaining_len(&self) -> usize {
    assert!(FIXED_BLOCK_SIZING, "remaining block only available for manual block");
    self.block_size - self.blocks[self.current_block_index].len()
  }

  fn push_n_within_block(&mut self, n: usize, is_valid: bool) {
    self.len += n;
    let mut block = &mut self.blocks[self.current_block_index];

    let size_before = block.allocated_size();

    if is_valid {
      block.append_n_non_nulls(n)
    } else {
      block.append_n_nulls(n)
    }

    self.allocated_size += (block.allocated_size() - size_before);

    assert!(block.len() <= self.block_size, "overflow from block new block length: {}, block size: {}", block.len(), self.block_size);

    if FIXED_BLOCK_SIZING && block.len() == self.block_size {
      self.start_new_block();
    }
  }

  pub fn push_n(&mut self, mut n: usize, is_valid: bool) {
    if !FIXED_BLOCK_SIZING {
      self.push_n_within_block(n, is_valid);
      return;
    }

    let number_of_blocks_to_reserve = n.saturating_sub(self.current_block_remaining_len()).div_ceil(self.block_size);
    self.reserve_blocks(number_of_blocks_to_reserve);

    while n > 0 {
      let remaining_in_current_block = self.current_block_remaining_len();

      let to_add = remaining_in_current_block.min(n);
      n -= to_add;

      self.push_n_within_block(to_add, is_valid);
    }
  }

  pub fn push_n_nulls(&mut self, n: usize) {
    self.push_n(n, false);
  }

  pub fn push_n_non_nulls(&mut self, n: usize) {
    self.push_n(n, true);
  }
  

  pub fn push_non_null(&mut self) {
    let mut block = &mut self.blocks[self.current_block_index];

    let mem_before = block.allocated_size();
    block.append_non_null();
    self.allocated_size += (block.allocated_size() - mem_before);
    self.len += 1;

    if FIXED_BLOCK_SIZING && block.len() == self.block_size {
      self.start_new_block();
    }
  }

  pub fn push_null(&mut self) {
    let mut block = &mut self.blocks[self.current_block_index];

    let mem_before = block.allocated_size();
    block.append_null();
    self.allocated_size += (block.allocated_size() - mem_before);
    self.len += 1;

    if block.len() == self.block_size {
      self.start_new_block();
    }
  }

  pub fn is_null(&self, blocked_index: BlockedIndex) -> bool {
    !self.blocks[blocked_index.block_index()].is_valid(blocked_index.index_in_block())
  }

  /// Extends iterator of validity within current block
  /// Returns how many items were added
  ///
  /// # Panics
  /// Panics if the iterator length exceeds the remaining size of the current block
  pub(super) fn extend_validity_in_block(&mut self, iter: impl Iterator<Item = bool>) -> usize {
    let block = &mut self.blocks[self.current_block_index];

    let prev_block_len = block.len();

    let mem_before = block.allocated_size();

    for is_valid in iter {
      block.append(is_valid);
    }

    self.allocated_size += (block.allocated_size() - mem_before);
    assert!(block.len() <= self.block_size, "overflow from block new block length: {}, block size: {}", block.len(), self.block_size);

    let added_items = block.len() - prev_block_len;
    self.len += added_items;

    if FIXED_BLOCK_SIZING && block.len() == self.block_size {
      self.start_new_block();
    }

    added_items
  }

  pub fn take_block(&mut self) -> Option<Option<NullBuffer>> {
    let capacity_before = self.blocks.capacity();
    let mut block = self.blocks.pop_front()?;
    self.allocated_size -= (capacity_before - self.blocks.capacity()) * size_of::<NullBufferBuilder>();
    self.allocated_size -= block.allocated_size();
    let number_of_items = block.len() - 1;
    self.len -= number_of_items;

    // Never have empty blocks since we won't be able to add more items
    if self.blocks.is_empty() {
      let empty_block = NullBufferBuilder::new(self.block_size);
      self.allocated_size += empty_block.allocated_size();
      let capacity_before = self.blocks.capacity();
      self.blocks.push_back(empty_block);

      self.allocated_size += (self.blocks.capacity() - capacity_before) * size_of::<NullBufferBuilder>();
    }

    Some(block.build().filter(|b| b.null_count() > 0))
  }
}

impl<const MANUAL_BLOCK_SIZE: bool> Extend<bool> for BlockedNullsBuilder<MANUAL_BLOCK_SIZE> {
  fn extend<T: IntoIterator<Item=bool>>(&mut self, iter: T) {
    let mut iter = iter.into_iter();

    if !MANUAL_BLOCK_SIZE {
      self.extend_validity_in_block(iter);
      return;
    }

    loop {
      let remaining_in_current_block = self.current_block_remaining_len();
      let added_items = self.extend_validity_in_block(iter.by_ref().take(remaining_in_current_block));

      if added_items == 0 {
        break;
      }
    }
  }
}


// Only when we control the blocking since otherwise each block is not the same size
impl Index<usize> for BlockedNullsBuilder<true> {
  type Output = bool;

  fn index(&self, index: usize) -> &Self::Output {
    self.index(BlocksIndex::from_index_in_fixed_block_size(index, self.block_size))
  }
}

impl<const FIXED_BLOCK_SIZING: bool> Index<BlocksIndex> for BlockedNullsBuilder<FIXED_BLOCK_SIZING> {
  type Output = bool;

  fn index(&self, blocked_index: BlocksIndex) -> &Self::Output {
    if self.blocks[blocked_index.block_index()].is_valid(blocked_index.index_in_block()) {
      &true
    } else {
      &false
    }
  }
}
