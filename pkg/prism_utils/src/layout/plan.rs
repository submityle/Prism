//! `SoA` auto-layout planning: deterministic column alignment / stride maths
//! and a hot / cold group description.
//!
//! Separating the *hot* fields a loop touches every iteration (position,
//! transform) from the *cold* fields it rarely reads (name, debug tag) keeps
//! the hot working set dense in cache. Before a container commits to a physical
//! layout it is useful to be able to *describe* that layout — how wide each
//! column is, what alignment its backing buffer wants, and how many bytes one
//! logical row costs in each temperature group. [`LayoutPlan`] computes exactly
//! that from the field shapes, with no allocation of the stored data and no
//! `unsafe`.
//!
//! ```
//! use prism_utils::layout::{ColumnShapes, LayoutPlan, Temperature};
//!
//! // Hot = (position: [f32; 3], velocity: [f32; 3]); Cold = (name_id: u64,).
//! let plan = LayoutPlan::of::<([f32; 3], [f32; 3]), (u64,)>();
//! assert_eq!(plan.hot.len(), 2);
//! assert_eq!(plan.cold.len(), 1);
//! assert_eq!(plan.hot.temperature, Temperature::Hot);
//! // Two 12-byte columns => 24 hot bytes per row.
//! assert_eq!(plan.hot.lane_bytes(), 24);
//! // The 8-byte cold column wants 8-byte alignment.
//! assert_eq!(plan.cold.group_align, 8);
//! # let _ = <(u64,) as ColumnShapes>::ARITY;
//! ```

extern crate alloc;

use alloc::vec::Vec;

/// A typical x86-64 / `AArch64` cache line, in bytes. Used by
/// [`GroupLayout::fits_cache_line`] to flag a hot group that stays inside one
/// line.
pub const CACHE_LINE: usize = 64;

/// Round `value` up to the next multiple of `align`.
///
/// `align` must be a non-zero power of two (every type's `align_of` is). This
/// is the standard stride / offset rounding used throughout the layout maths.
///
/// ```
/// use prism_utils::layout::align_up;
/// assert_eq!(align_up(0, 16), 0);
/// assert_eq!(align_up(1, 16), 16);
/// assert_eq!(align_up(16, 16), 16);
/// assert_eq!(align_up(17, 16), 32);
/// ```
#[must_use]
pub const fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two(), "align must be a power of two");
    (value + align - 1) & !(align - 1)
}

/// Whether a column holds hot (frequently touched) or cold (rarely touched)
/// data. Drives the hot / cold split that keeps the hot working set dense.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Temperature {
    /// Accessed on the hot path every / most iterations.
    Hot,
    /// Accessed rarely; parked out of the hot cache footprint.
    Cold,
}

/// The physical shape of one column's element type: its size and alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnShape {
    /// `size_of` the element type, in bytes.
    pub size: usize,
    /// `align_of` the element type, in bytes (a power of two).
    pub align: usize,
}

impl ColumnShape {
    /// The shape of column element type `T`, resolved at compile time.
    #[must_use]
    pub const fn of<T>() -> Self {
        Self {
            size: size_of::<T>(),
            align: align_of::<T>(),
        }
    }

    /// The element stride: `size` rounded up to the element's own alignment.
    ///
    /// For a well-formed Rust type this already equals `size`; it is spelled
    /// out so an `AoSoA` lane that over-aligns an element has a correct byte
    /// stride.
    #[must_use]
    pub const fn stride(&self) -> usize {
        align_up(self.size, self.align)
    }

    /// The element stride when the lane is additionally forced to at least
    /// `lane_align` (e.g. a `SIMD` lane width). The effective alignment is the
    /// larger of the element's own alignment and `lane_align`.
    #[must_use]
    pub const fn stride_for(&self, lane_align: usize) -> usize {
        let align = if lane_align > self.align {
            lane_align
        } else {
            self.align
        };
        align_up(self.size, align)
    }
}

/// One planned column: which tuple field it came from and its [`ColumnShape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnPlan {
    /// Zero-based index of the field within its temperature group's tuple.
    pub field_index: usize,
    /// The element shape of this column.
    pub shape: ColumnShape,
}

/// The computed layout of one temperature group (all hot columns, or all cold
/// columns): the per-column plans plus the derived group-wide alignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupLayout {
    /// Whether this group is the hot or the cold half.
    pub temperature: Temperature,
    /// One [`ColumnPlan`] per field, in tuple order.
    pub columns: Vec<ColumnPlan>,
    /// The group's recommended backing-buffer alignment: the max of every
    /// column's element alignment (at least 1, so an empty group is well
    /// defined).
    pub group_align: usize,
}

impl GroupLayout {
    /// Build a group layout from the ordered column shapes of `temperature`.
    #[must_use]
    pub fn from_shapes(temperature: Temperature, shapes: &[ColumnShape]) -> Self {
        let mut group_align = 1usize;
        let mut columns = Vec::with_capacity(shapes.len());
        for (field_index, &shape) in shapes.iter().enumerate() {
            if shape.align > group_align {
                group_align = shape.align;
            }
            columns.push(ColumnPlan { field_index, shape });
        }
        Self {
            temperature,
            columns,
            group_align,
        }
    }

    /// The number of columns in the group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Whether the group has no columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The total bytes one logical row occupies across this group's columns,
    /// summing each column's [`ColumnShape::stride`]. For the hot group this is
    /// the per-element hot working-set width.
    #[must_use]
    pub fn lane_bytes(&self) -> usize {
        self.columns.iter().map(|c| c.shape.stride()).sum()
    }

    /// `lane_bytes` recomputed with every column forced to at least `lane_align`
    /// (a `SIMD` lane width); useful for sizing an `AoSoA` block.
    #[must_use]
    pub fn lane_bytes_for(&self, lane_align: usize) -> usize {
        self.columns.iter().map(|c| c.shape.stride_for(lane_align)).sum()
    }

    /// Whether a whole row of this group fits inside a single [`CACHE_LINE`].
    #[must_use]
    pub fn fits_cache_line(&self) -> bool {
        self.lane_bytes() <= CACHE_LINE
    }
}

/// The full hot / cold layout plan for a hot tuple `H` and a cold tuple `C`:
/// their two [`GroupLayout`]s side by side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutPlan {
    /// Layout of the hot group.
    pub hot: GroupLayout,
    /// Layout of the cold group.
    pub cold: GroupLayout,
}

impl LayoutPlan {
    /// Compute the plan for hot tuple `H` and cold tuple `C` directly from
    /// their field types.
    #[must_use]
    pub fn of<H: ColumnShapes, C: ColumnShapes>() -> Self {
        Self::from_shapes(&H::column_shapes(), &C::column_shapes())
    }

    /// Compute the plan from already-collected hot and cold column shapes.
    #[must_use]
    pub fn from_shapes(hot: &[ColumnShape], cold: &[ColumnShape]) -> Self {
        Self {
            hot: GroupLayout::from_shapes(Temperature::Hot, hot),
            cold: GroupLayout::from_shapes(Temperature::Cold, cold),
        }
    }
}

/// A tuple whose fields' physical [`ColumnShape`]s can be enumerated in order.
///
/// Implemented for tuples of arity 1 through 8, mirroring the
/// [`Soa`](crate::soa::Soa) trait, so the same tuple that drives a
/// [`SoaVec`](crate::soa::SoaVec) column split also drives its layout plan.
pub trait ColumnShapes {
    /// The number of columns (tuple arity).
    const ARITY: usize;
    /// The ordered element shapes, one per field.
    fn column_shapes() -> Vec<ColumnShape>;
}

macro_rules! impl_column_shapes {
    ($arity:literal; $($t:ident),+ $(,)?) => {
        impl<$($t),+> ColumnShapes for ($($t,)+) {
            const ARITY: usize = $arity;

            fn column_shapes() -> Vec<ColumnShape> {
                let mut shapes = Vec::with_capacity($arity);
                $( shapes.push(ColumnShape::of::<$t>()); )+
                shapes
            }
        }
    };
}

impl_column_shapes!(1; A);
impl_column_shapes!(2; A, B);
impl_column_shapes!(3; A, B, C);
impl_column_shapes!(4; A, B, C, D);
impl_column_shapes!(5; A, B, C, D, E);
impl_column_shapes!(6; A, B, C, D, E, F);
impl_column_shapes!(7; A, B, C, D, E, F, G);
impl_column_shapes!(8; A, B, C, D, E, F, G, H);
