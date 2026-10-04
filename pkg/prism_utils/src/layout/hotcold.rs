//! [`HotCold`]: a hot / cold field-separated structure-of-arrays container.
//!
//! A hot loop that only reads an entity's position should not drag its name,
//! debug tag, or editor metadata through the cache. [`HotCold`] stores the
//! *hot* fields and the *cold* fields in two completely separate
//! [`SoaVec`](crate::soa::SoaVec)s, kept in lockstep by row index. Iterating
//! the hot columns therefore never touches a single byte of cold memory, so
//! the hot working set stays dense and vectorisable; the cold columns are
//! still one `O(1)` index away when a rare path needs them.
//!
//! Both halves are ordinary tuples driving the derive-free
//! [`Soa`](crate::soa::Soa) trait, so no proc-macro or `unsafe` is involved:
//!
//! ```
//! use prism_utils::layout::HotCold;
//!
//! // Hot = (position, velocity); Cold = (name_id, spawn_tick).
//! let mut world: HotCold<([f32; 3], [f32; 3]), (u64, u64)> = HotCold::new();
//! world.push(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), (7, 100));
//! world.push(([5.0, 0.0, 0.0], [0.0, 1.0, 0.0]), (9, 101));
//! assert_eq!(world.len(), 2);
//!
//! // Hot batch pass over dense columns — no cold bytes loaded.
//! let (positions, velocities) = world.hot_columns();
//! let mut sum = 0.0f32;
//! for (p, v) in positions.iter().zip(velocities) {
//!     sum += p[0] + v[0];
//! }
//! assert_eq!(sum, 6.0);
//!
//! // Cold data is still one index away.
//! assert_eq!(world.get_cold(0), Some((&7, &100)));
//! ```

use crate::soa::{Soa, SoaIter, SoaVec};

use super::plan::{ColumnShapes, LayoutPlan};

/// A hot / cold field-separated columnar container.
///
/// `H` is the tuple of hot fields and `C` the tuple of cold fields; row `i`'s
/// hot fields live at index `i` of the hot columns and its cold fields at the
/// same index of the cold columns. [`push`](Self::push),
/// [`swap_remove`](Self::swap_remove) and [`clear`](Self::clear) keep the two
/// halves exactly the same length, so the shared length is always well defined.
pub struct HotCold<H: Soa, C: Soa> {
    hot: SoaVec<H>,
    cold: SoaVec<C>,
}

impl<H: Soa, C: Soa> core::fmt::Debug for HotCold<H, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HotCold")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl<H: Soa, C: Soa> Default for HotCold<H, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Soa, C: Soa> HotCold<H, C> {
    /// Create an empty container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hot: SoaVec::new(),
            cold: SoaVec::new(),
        }
    }

    /// Create an empty container with room for `cap` rows in both halves.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            hot: SoaVec::with_capacity(cap),
            cold: SoaVec::with_capacity(cap),
        }
    }

    /// The number of rows (shared by both halves).
    #[must_use]
    pub fn len(&self) -> usize {
        self.hot.len()
    }

    /// Whether the container has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hot.is_empty()
    }

    /// Append a row: its hot fields onto the hot columns and its cold fields
    /// onto the cold columns.
    pub fn push(&mut self, hot: H, cold: C) {
        self.hot.push(hot);
        self.cold.push(cold);
    }

    /// Remove row `index` from both halves by swapping the last row into its
    /// place, returning the removed `(hot, cold)` pair, or `None` if `index`
    /// is out of bounds.
    pub fn swap_remove(&mut self, index: usize) -> Option<(H, C)> {
        if index >= self.len() {
            return None;
        }
        let hot = self
            .hot
            .swap_remove(index)
            .expect("index checked against the shared length");
        let cold = self
            .cold
            .swap_remove(index)
            .expect("hot and cold columns stay in lockstep");
        Some((hot, cold))
    }

    /// Borrow row `index`'s hot fields as a tuple of shared references, or
    /// `None` if out of bounds.
    #[must_use]
    pub fn get_hot(&self, index: usize) -> Option<H::Ref<'_>> {
        self.hot.get(index)
    }

    /// Borrow row `index`'s cold fields as a tuple of shared references, or
    /// `None` if out of bounds.
    #[must_use]
    pub fn get_cold(&self, index: usize) -> Option<C::Ref<'_>> {
        self.cold.get(index)
    }

    /// Borrow row `index`'s hot fields as a tuple of exclusive references.
    pub fn get_hot_mut(&mut self, index: usize) -> Option<H::RefMut<'_>> {
        self.hot.get_mut(index)
    }

    /// Borrow row `index`'s cold fields as a tuple of exclusive references.
    pub fn get_cold_mut(&mut self, index: usize) -> Option<C::RefMut<'_>> {
        self.cold.get_mut(index)
    }

    /// Borrow both halves of row `index` at once, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<(H::Ref<'_>, C::Ref<'_>)> {
        match (self.hot.get(index), self.cold.get(index)) {
            (Some(hot), Some(cold)) => Some((hot, cold)),
            _ => None,
        }
    }

    /// Borrow the raw hot column storage (a tuple of `Vec`s), for a dense,
    /// parallel-friendly batch pass that never touches cold memory.
    #[must_use]
    pub fn hot_columns(&self) -> &H::Columns {
        self.hot.columns()
    }

    /// Borrow the raw cold column storage.
    #[must_use]
    pub fn cold_columns(&self) -> &C::Columns {
        self.cold.columns()
    }

    /// Exclusively borrow the raw hot column storage.
    pub fn hot_columns_mut(&mut self) -> &mut H::Columns {
        self.hot.columns_mut()
    }

    /// Exclusively borrow the raw cold column storage.
    pub fn cold_columns_mut(&mut self) -> &mut C::Columns {
        self.cold.columns_mut()
    }

    /// Iterate the hot rows as tuples of shared references, in order.
    #[must_use]
    pub fn iter_hot(&self) -> SoaIter<'_, H> {
        self.hot.iter()
    }

    /// Iterate the cold rows as tuples of shared references, in order.
    #[must_use]
    pub fn iter_cold(&self) -> SoaIter<'_, C> {
        self.cold.iter()
    }

    /// Remove every row from both halves.
    pub fn clear(&mut self) {
        self.hot.clear();
        self.cold.clear();
    }
}

impl<H: Soa + ColumnShapes, C: Soa + ColumnShapes> HotCold<H, C> {
    /// The static hot / cold [`LayoutPlan`] for this container's field types:
    /// per-column strides, each group's recommended alignment, and the hot
    /// working-set width. Independent of the current row count.
    #[must_use]
    pub fn layout_plan() -> LayoutPlan {
        LayoutPlan::of::<H, C>()
    }
}
