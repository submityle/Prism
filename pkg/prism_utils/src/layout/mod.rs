//! §24.4 hot / cold data separation and `SoA` auto-layout.
//!
//! Cache efficiency comes from putting *only the hot fields into the cache
//! line*. This module delivers the two halves of that idea:
//!
//! - [`HotCold`]: a container that stores a type's frequently touched (hot)
//!   fields and its rarely touched (cold) fields in two separate
//!   structure-of-arrays halves, so a hot batch pass never loads cold memory.
//!   It composes the derive-free [`SoaVec`](crate::soa::SoaVec) column storage,
//!   so the whole thing is safe code.
//! - [`LayoutPlan`] / [`GroupLayout`] / [`ColumnShape`]: the `SoA` auto-layout
//!   maths — per-column alignment and stride, each temperature group's
//!   recommended backing-buffer alignment, the hot working-set width, and a
//!   [`SIMD`-lane](ColumnShape::stride_for)-aware stride for sizing `AoSoA`
//!   blocks. [`ColumnShapes`] enumerates a tuple's field shapes, mirroring the
//!   [`Soa`](crate::soa::Soa) trait.
//!
//! Where the base [`soa`](crate::soa) module gives one flat `SoaVec` over a
//! single tuple, this module adds the *temperature split* and the layout
//! description the ECS column store and `prism_math` batch kernels want on top.

pub mod aosoa;
pub mod hotcold;
pub mod plan;

pub use aosoa::AoSoa;
pub use hotcold::HotCold;
pub use plan::{
    align_up, ColumnPlan, ColumnShape, ColumnShapes, GroupLayout, LayoutPlan, Temperature,
    CACHE_LINE,
};
