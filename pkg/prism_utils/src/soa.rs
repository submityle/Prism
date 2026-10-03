//! Structure-of-arrays (`SoA`) columnar storage: [`SoaVec`], a derive-free,
//! cache-friendly container that stores each field of a tuple in its own
//! contiguous column.
//!
//! # Why `SoA`
//! Array-of-structs (`AoS`) interleaves fields, so a hot loop that only reads
//! one field still drags the cold fields through the cache. Structure-of-arrays
//! stores each field contiguously, so a batch pass over one column is dense and
//! vectorisable. This is the layout ECS column storage and `prism_math` batch
//! kernels want.
//!
//! # Derive-free
//! Rather than a proc-macro deriving per-struct storage, [`SoaVec`] is generic
//! over the [`Soa`] trait, which is implemented here for tuples of arity 1
//! through 8. `SoaVec<(A, B, C)>` therefore stores three columns (`Vec<A>`,
//! `Vec<B>`, `Vec<C>`) with no user boilerplate. The whole module is safe code.
//!
//! ```
//! use prism_utils::soa::SoaVec;
//!
//! let mut v: SoaVec<(u32, f32)> = SoaVec::new();
//! v.push((1, 1.5));
//! v.push((2, 2.5));
//! assert_eq!(v.get(1), Some((&2, &2.5)));
//! // Each field is a dense column:
//! let (ids, weights) = v.columns();
//! assert_eq!(ids, &[1, 2]);
//! assert_eq!(weights, &[1.5, 2.5]);
//! ```

extern crate alloc;

use alloc::vec::Vec;

/// A tuple whose fields can be split into parallel columns.
///
/// Implemented for tuples of arity 1 through 8. The associated [`Columns`](Soa::Columns)
/// type is a tuple of `Vec`s, one per field; [`Ref`](Soa::Ref) and
/// [`RefMut`](Soa::RefMut) are the matching tuples of (mutable) references.
pub trait Soa: Sized {
    /// The backing storage: one `Vec` per field.
    type Columns: Default;
    /// A tuple of shared references, one per field.
    type Ref<'a>
    where
        Self: 'a;
    /// A tuple of exclusive references, one per field.
    type RefMut<'a>
    where
        Self: 'a;

    /// Build empty columns with capacity for `cap` rows each.
    fn columns_with_capacity(cap: usize) -> Self::Columns;
    /// Append `value`, pushing one element onto each column.
    fn push(cols: &mut Self::Columns, value: Self);
    /// Swap-remove row `index` from every column and reconstitute the tuple.
    fn swap_remove(cols: &mut Self::Columns, index: usize) -> Self;
    /// Borrow row `index` from every column as a tuple of shared references.
    fn get(cols: &Self::Columns, index: usize) -> Self::Ref<'_>;
    /// Borrow row `index` from every column as a tuple of exclusive references.
    fn get_mut(cols: &mut Self::Columns, index: usize) -> Self::RefMut<'_>;
    /// The number of rows (length of column 0).
    fn len(cols: &Self::Columns) -> usize;
    /// Drop every row, leaving the columns empty.
    fn clear(cols: &mut Self::Columns);
}

/// A growable structure-of-arrays vector: a logical sequence of `T` tuples
/// stored as one dense column per field.
///
/// `push` / `swap_remove` / `get` keep all columns the same length, so the
/// container behaves like a `Vec<T>` whose fields happen to live in separate
/// cache-friendly arrays.
#[derive(Debug, Default)]
pub struct SoaVec<T: Soa> {
    columns: T::Columns,
}

impl<T: Soa> SoaVec<T> {
    /// Create an empty `SoaVec`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: T::Columns::default(),
        }
    }

    /// Create an empty `SoaVec` with room for `cap` rows per column.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            columns: T::columns_with_capacity(cap),
        }
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        T::len(&self.columns)
    }

    /// Returns `true` if there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a row, pushing one element onto each column.
    pub fn push(&mut self, value: T) {
        T::push(&mut self.columns, value);
    }

    /// Remove row `index` by swapping the last row into its place and return
    /// the removed tuple, or `None` if `index` is out of bounds.
    pub fn swap_remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }
        Some(T::swap_remove(&mut self.columns, index))
    }

    /// Borrow row `index` as a tuple of shared references, or `None` if out of
    /// bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<T::Ref<'_>> {
        if index >= self.len() {
            return None;
        }
        Some(T::get(&self.columns, index))
    }

    /// Borrow row `index` as a tuple of exclusive references, or `None` if out
    /// of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<T::RefMut<'_>> {
        if index >= self.len() {
            return None;
        }
        Some(T::get_mut(&mut self.columns, index))
    }

    /// Remove every row.
    pub fn clear(&mut self) {
        T::clear(&mut self.columns);
    }

    /// Borrow the raw column storage (a tuple of `&[Field]` slices via the
    /// column `Vec`s). This is the whole point of `SoA`: a dense, contiguous,
    /// vectorisable view of a single field.
    #[must_use]
    pub fn columns(&self) -> &T::Columns {
        &self.columns
    }

    /// Exclusively borrow the raw column storage.
    pub fn columns_mut(&mut self) -> &mut T::Columns {
        &mut self.columns
    }

    /// Iterate the rows as tuples of shared references, in order.
    #[must_use]
    pub fn iter(&self) -> SoaIter<'_, T> {
        SoaIter {
            vec: self,
            index: 0,
        }
    }
}

/// Row iterator over a [`SoaVec`], yielding `T::Ref` in order. Produced by
/// [`SoaVec::iter`].
pub struct SoaIter<'a, T: Soa> {
    vec: &'a SoaVec<T>,
    index: usize,
}

impl<'a, T: Soa> Iterator for SoaIter<'a, T> {
    type Item = T::Ref<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.vec.len() {
            return None;
        }
        let row = T::get(&self.vec.columns, self.index);
        self.index += 1;
        Some(row)
    }
}

/// Implement [`Soa`] for one tuple arity.
macro_rules! impl_soa_tuple {
    ($($t:ident => $idx:tt),+ $(,)?) => {
        impl<$($t),+> Soa for ($($t,)+) {
            type Columns = ($(Vec<$t>,)+);
            type Ref<'a> = ($(&'a $t,)+) where Self: 'a;
            type RefMut<'a> = ($(&'a mut $t,)+) where Self: 'a;

            #[inline]
            fn columns_with_capacity(cap: usize) -> Self::Columns {
                ($(Vec::<$t>::with_capacity(cap),)+)
            }

            #[inline]
            fn push(cols: &mut Self::Columns, value: Self) {
                $( cols.$idx.push(value.$idx); )+
            }

            #[inline]
            fn swap_remove(cols: &mut Self::Columns, index: usize) -> Self {
                ($( cols.$idx.swap_remove(index), )+)
            }

            #[inline]
            fn get(cols: &Self::Columns, index: usize) -> Self::Ref<'_> {
                ($( &cols.$idx[index], )+)
            }

            #[inline]
            fn get_mut(cols: &mut Self::Columns, index: usize) -> Self::RefMut<'_> {
                ($( &mut cols.$idx[index], )+)
            }

            #[inline]
            fn len(cols: &Self::Columns) -> usize {
                cols.0.len()
            }

            #[inline]
            fn clear(cols: &mut Self::Columns) {
                $( cols.$idx.clear(); )+
            }
        }
    };
}

impl_soa_tuple!(A => 0);
impl_soa_tuple!(A => 0, B => 1);
impl_soa_tuple!(A => 0, B => 1, C => 2);
impl_soa_tuple!(A => 0, B => 1, C => 2, D => 3);
impl_soa_tuple!(A => 0, B => 1, C => 2, D => 3, E => 4);
impl_soa_tuple!(A => 0, B => 1, C => 2, D => 3, E => 4, F => 5);
impl_soa_tuple!(A => 0, B => 1, C => 2, D => 3, E => 4, F => 5, G => 6);
impl_soa_tuple!(A => 0, B => 1, C => 2, D => 3, E => 4, F => 5, G => 6, H => 7);
