//! Big-world grid-cell origin rebasing (milestone M3).
//!
//! Large open worlds cannot be rendered directly from `f32` world coordinates:
//! a single-precision value at 100 km has an inter-value spacing of ~12 mm, so
//! static geometry visibly *jitters* as positions snap between representable
//! values. This module eliminates that jitter with the standard floating-origin
//! technique, built on the [`f64`](crate::f64) big-world types.
//!
//! # Model
//! World space is tiled into cubic [`GridCell`]s of
//! [`GridCell::CELL_SIZE`] metres. A world point is stored as a
//! [`GridPosition`]: an integer `cell` plus a small `f32` `offset` within that
//! cell. Because the cell index is an exact integer, the stored position keeps
//! full local precision no matter how far it is from the world origin.
//!
//! # Rebasing
//! To render, every position is expressed *relative to the camera's cell* with
//! [`GridPosition::rebased_offset`] (or, for whole transforms,
//! [`GridCell::relative_transform`]). Nearby objects then have small
//! coordinates and are reconstructed in `f32` with sub-millimetre accuracy,
//! while distant objects — where a larger absolute error is imperceptible —
//! degrade gracefully.
//!
//! # Precision contract
//! With the default `CELL_SIZE` of 1024 m:
//! - **Local offset:** an in-cell `f32` offset resolves to `1024 * 2^-23 ≈
//!   0.12 mm`, independent of absolute world magnitude.
//! - **Round trip:** [`GridPosition::from_dvec3`] → [`GridPosition::to_dvec3`]
//!   reconstructs the original `f64` world point to within one local-offset ULP
//!   (≤ 0.12 mm) anywhere in the representable world.
//! - **Rebased offset:** a point `d` metres from the camera cell is
//!   reconstructed in `f32` with absolute error ≈ `d * 2^-23` — ≤ 0.5 mm for
//!   `d ≤ 4 km`, ≤ 12 mm at `d = 100 km`. This is the ±100 km jitter-free
//!   guarantee: geometry near the camera (visible-jitter range) stays
//!   sub-millimetre even with the scene parked 100 km from the origin.
//!
//! Cell arithmetic uses `i64` intermediates and the exact power-of-two
//! `CELL_SIZE`, so cell-to-cell translations carry no rounding error of their
//! own.

mod grid;

pub use grid::{GridCell, GridPosition};
