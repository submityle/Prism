//! Large-world coordinate and spatial-cell contracts.
//!
//! Open worlds are far larger than single-precision floating point can address
//! without visible jitter. This module splits every position into an integer
//! [`WorldCell`] plus a small local offset, quantized by a configurable
//! [`WorldGrid`], and keeps a camera-relative [`WorldOrigin`] so render math
//! stays near zero. When the camera moves far the origin is rebased and all
//! offsets are recomputed (see [`rebase_offset`] / [`rebase_all`]), with an
//! `epoch` marking stale pre-rebase offsets. All world coordinates are `f64`;
//! only basic arithmetic plus `sqrt` is used, so there are no transcendental
//! calls anywhere in the coordinate path.
//!
//! Submodules:
//! - [`cell`] — [`WorldCell`], the [`WorldGrid`] quantizer, and integer
//!   distance/neighbourhood queries.
//! - [`position`] — [`WorldPosition`], [`WorldOrigin`], camera-relative
//!   rebasing, and epoch invalidation.

pub mod cell;
pub mod position;

pub use cell::{WorldCell, WorldGrid};
pub use position::{rebase_all, rebase_offset, WorldOrigin, WorldPosition};
