//! Double-precision (`f64`) "big-world" math layer (milestone M3).
//!
//! This module provides exact `f64` analogues of the `f32` facade types:
//! [`DVec2`]/[`DVec3`]/[`DVec4`], [`DMat2`]/[`DMat3`]/[`DMat4`], [`DQuat`], and
//! [`DAffine3`]. The API matches the `f32` types one-to-one, so code can be
//! written against either precision with the same method names.
//!
//! Why `f64`: a single-precision `f32` coordinate has only ~24 bits of
//! mantissa, so at a world distance of 100 km the spacing between representable
//! values is about 12 mm — large enough to produce visible positional
//! *jitter*. The big-world path keeps authoritative positions in `f64` (ULP of
//! ~1.5e-8 m at 100 km) and only narrows to `f32` after rebasing them into a
//! small, camera-relative offset via [`crate::bigworld`].
//!
//! There is no SIMD backend here: `f64` math is implemented in scalar form and
//! is its own behavioural reference. Conversions between the precisions are
//! spelled `as_dvec3` (widen) and `as_vec3` (narrow), mirroring the glam-style
//! naming used across the crate.

pub mod daffine;
pub mod dmat;
pub mod dquat;
pub mod dvec;

pub use daffine::DAffine3;
pub use dmat::{DMat2, DMat3, DMat4};
pub use dquat::DQuat;
pub use dvec::{dvec2, dvec3, dvec4, DVec2, DVec3, DVec4};
