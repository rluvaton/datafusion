
/// Suppose to provide the same API as [`Vec`] but with the functionality to take owned blocks
///
/// But because this is a blocked implementation, some underline implementations might not allow a slice
/// over the entire underlying data, so providing that API in case the current underlying implementation allows that
/// will limit us from changing the underlying implementation without breaking the API.
pub struct BlockedVec<T> {
  /// # Implementations considerations
  ///
  /// ## `Vec<Vec<T>>`
  /// The naive approach is doing `Vec<Vec<T>>`
  ///
  /// ### Advantages:
  /// 1. Easy to implement
  /// 2. Easy to reason about from a high level
  ///
  /// ### Disadvantages:
  /// 1. The vecs are not contiguous in memory, so you will have:
  ///   1. More cache misses
  ///   2. Less prefetching optimizations
  /// 2. Taking the first `block_size / 2` will:
  ///   1. leave the first block with half the elements
  ///   2. will not free the memory for the taken elements
  ///   3. Any next taking `block_size` will require copy to combine into a single `Vec`
  /// 3. You can't have a slice over the entire underlying data
  ///
  /// ## `Vec<T, MMAP>`
  /// The more complex approach is using [`Vec`] backed by mmap
  ///
  /// ### Advantages:
  /// 1. Can hold on a slice of the entire underlying data (I think)
  /// 2. The data between blocks are contiguous in memory (I think)
  /// 3. Allows you to take owned blocks which will free when dropped
  ///    (unless the block is partial or span 2 pages but not the entire page)
  /// 4. This will make it easier to implement bytes for `StringArray` where we don't want each byte
  ///    to count toward the block but instead by the number of items (`offsets.len() - 1`)
  ///    since we are not really manage blocks - but pages
  ///
  /// ### Disadvantages:
  /// 1. Harder to implement
  /// 2. Not tracked by the global allocator
  ///    (if you have a custom one that acts like a cgroup for memory limit, it will not count that)
  ///
  ///
  ///
  blocks: Vec<Vec<T>>,

  block_size: usize,
}

// We also need an implementation of `MutableBuffer`
// We also need an implementation for each block but for specifying the provider that decide the actual length
//  For example if I want to create blocks of `StringArray`,
//  I need to have block of offsets and block of bytes,
//  but the block size of the bytes is not determined by the number of bytes
//  but instead by the number of offsets - i.e. the number of items.
//
//  If the block contains both offsets and bytes (since for `StringArray` case when you add offsets you also need to add bytes probably)
//  But we want to model each block so we can add offsets in bytes in large copy and that we can add
//
//
// But for `StringArray` each emitted blocks the offsets should start from 0, so the notion of blocks should still exist
// so if we say we don't have `Emit::First(n)` anymore and only have `Emit::Block` then we won't have
// an issue with `Vec<Vec<T>>` regarding emitting `First(n)` and then emitting blocks which required to copy data
// however for each emit, having to shift the offset is quite inexpensive so the cost is ok, BUT the next offset you add you must account for the last offset + len
// and not `buffer.len()`
