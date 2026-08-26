use crate::groups_accumulator::BlocksIndex;

pub trait EnumerateBlockedIteratorExt: Iterator {
  fn enumerate_blocked(self, block_size: usize) -> EnumerateBlocked<Self>
  where
    Self: Sized {
    EnumerateBlocked::new(self, block_size)
  }
}

impl<T: Iterator> EnumerateBlockedIteratorExt for T {}


/// An iterator that yields the current count in blocked index and the element during iteration.
///
/// This `struct` is created by the [`enumerate_blocked`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`enumerate_blocked`]: EnumerateBlocked::enumerate_blocked
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct EnumerateBlocked<I> {
  block_size: usize,
  iter: I,
  index: BlocksIndex,
}

impl<I> EnumerateBlocked<I> {
  pub(crate) const fn new(iter: I, block_size: usize) -> Self<I> {
    assert_ne!(block_size, 0);
    Self {
      iter,
      index: BlocksIndex::default(),
      block_size,
    }
  }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> Iterator for EnumerateBlocked<I>
where
  I: Iterator,
{
  type Item = (BlocksIndex, <I as Iterator>::Item);

  /// # Overflow Behavior
  ///
  /// The method does no guarding against overflows, so enumerating more than
  /// `usize::MAX` elements either produces the wrong result or panics. If
  /// overflow checks are enabled, a panic is guaranteed.
  ///
  /// # Panics
  ///
  /// Might panic if the index of the element overflows a `usize`.
  #[inline]
  fn next(&mut self) -> Option<(BlocksIndex, <I as Iterator>::Item)> {
    let a = self.iter.next()?;
    let i = self.index;
    self.index.next_mut_fixed(self.block_size);

    Some((i, a))
  }

  #[inline]
  fn size_hint(&self) -> (usize, Option<usize>) {
    self.iter.size_hint()
  }

  #[inline]
  fn nth(&mut self, n: usize) -> Option<(BlocksIndex, I::Item)> {
    let a = self.iter.nth(n)?;
    let i = self.index.add_fixed(n, self.block_size);
    self.index = i.next_fixed(self.block_size);

    Some((i, a))
  }

  #[inline]
  fn count(self) -> usize {
    self.iter.count()
  }

  #[inline]
  fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
  where
    Fold: FnMut(Acc, Self::Item) -> Acc,
  {
    #[inline]
    fn enumerate<T, Acc>(
      mut index: BlocksIndex,
      block_size: usize,
      mut fold: impl FnMut(Acc, (BlocksIndex, T)) -> Acc,
    ) -> impl FnMut(Acc, T) -> Acc {
      #[rustc_inherit_overflow_checks]
      move |acc, item| {
        let acc = fold(acc, (index, item));
        index.next_mut_fixed(block_size);
        acc
      }
    }

    self.iter.fold(init, enumerate(self.index, self.block_size, fold))
  }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I> DoubleEndedIterator for EnumerateBlocked<I>
where
  I: ExactSizeIterator + DoubleEndedIterator,
{
  #[inline]
  fn next_back(&mut self) -> Option<(BlocksIndex, <I as Iterator>::Item)> {
    let a = self.iter.next_back()?;
    let len = self.iter.len();
    // Can safely add, `ExactSizeIterator` promises that the number of
    // elements fits into a `usize`.
    Some((self.index.add_fixed(len, self.block_size), a))
  }

  #[inline]
  fn nth_back(&mut self, n: usize) -> Option<(BlocksIndex, <I as Iterator>::Item)> {
    let a = self.iter.nth_back(n)?;
    let len = self.iter.len();
    // Can safely add, `ExactSizeIterator` promises that the number of
    // elements fits into a `usize`.
    Some((self.index.add_fixed(len, self.block_size), a))
  }

  #[inline]
  fn rfold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
  where
    Fold: FnMut(Acc, Self::Item) -> Acc,
  {
    // Can safely add and subtract the count, as `ExactSizeIterator` promises
    // that the number of elements fits into a `usize`.
    fn enumerate<T, Acc>(
      mut index: BlocksIndex,
      block_size: usize,
      mut fold: impl FnMut(Acc, (BlocksIndex, T)) -> Acc,
    ) -> impl FnMut(Acc, T) -> Acc {
      move |acc, item| {
        index.prev_mut_fixed(block_size);
        fold(acc, (index, item))
      }
    }

    let index = self.index.add_fixed(self.iter.len(), self.block_size);
    self.iter.rfold(init, enumerate(index, self.block_size, fold))
  }
}

impl<I> ExactSizeIterator for EnumerateBlocked<I>
where
  I: ExactSizeIterator,
{
  fn len(&self) -> usize {
    self.iter.len()
  }
}

