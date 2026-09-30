//! Tunables and fixed-point helpers for the `GPU` per-fragment bounds builder.
//!
//! Once the Voronoi classifier has binned a debris point cloud into fragment
//! cells, a destruction system needs each fragment's broad-phase proxy: an
//! axis-aligned bounding box and a bounding sphere. This module holds the two
//! quantisation scales the box and sphere reductions share.
//!
//! # Fixed-point reduction
//!
//! `WGSL` atomics operate only on 32-bit integers, so the box extents are
//! reduced with integer `atomicMin`/`atomicMax` over quantised coordinates and
//! the squared sphere radius with an integer `atomicMax` over quantised squared
//! distances. Quantisation is monotone, so the integer extrema reproduce the
//! real-valued extrema exactly once de-quantised: the box is therefore
//! bit-identical to the [`cpu_bounds_fragments`](super::cpu::cpu_bounds_fragments)
//! golden twin, and only the final radius square root can diverge by low bits.
//!
//! # Overflow budget
//!
//! Each accumulator is an `i32`, so a quantised coordinate must stay within
//! `+/- 2^31`. The squared distance grows with the *square* of the coordinate
//! magnitude, so its scale defaults below the position scale. A scene with
//! larger coordinates should lower the scales; the parity tests deliberately
//! stay inside the documented budget.
//!
//! # Provenance
//!
//! Fixed-point atomic reduction of extrema is a standard `GPU` technique and
//! the box/sphere formulas are elementary geometry. This module contains no
//! Unreal Engine source or derived code.

/// Parameters controlling a per-fragment bounds dispatch.
///
/// The two scales trade fractional precision against the `i32` overflow budget
/// documented on this module; the defaults suit debris binning with coordinates
/// within a few tens of units.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoundsConfig {
    /// Fixed-point scale applied to a coordinate before the box `atomicMin`/
    /// `atomicMax` reduction.
    pub position_scale: f32,
    /// Fixed-point scale applied to a squared distance before the radius
    /// `atomicMax` reduction.
    pub radius_sq_scale: f32,
}

impl BoundsConfig {
    /// Default position scale (`2^16`).
    pub const DEFAULT_POSITION_SCALE: f32 = 65_536.0;
    /// Default squared-distance scale (`2^14`), lower than the position scale
    /// because the squared distance carries the *square* of the coordinate
    /// magnitude.
    pub const DEFAULT_RADIUS_SQ_SCALE: f32 = 16_384.0;
}

impl Default for BoundsConfig {
    fn default() -> BoundsConfig {
        BoundsConfig {
            position_scale: Self::DEFAULT_POSITION_SCALE,
            radius_sq_scale: Self::DEFAULT_RADIUS_SQ_SCALE,
        }
    }
}

/// Quantises `value * scale` to the nearest integer using round-half-to-even,
/// matching the `WGSL` `round` builtin used by the device kernel.
///
/// The `as i32` cast saturates on overflow and truncates the already
/// integer-valued float, agreeing with the in-range `WGSL` `i32(f32)`
/// conversion.
#[must_use]
pub fn quantise(value: f32, scale: f32) -> i32 {
    (value * scale).round_ties_even() as i32
}

/// De-quantises a fixed-point integer back to a float by dividing by `scale`.
#[must_use]
pub fn dequantise(value: i32, scale: f32) -> f32 {
    value as f32 / scale
}
