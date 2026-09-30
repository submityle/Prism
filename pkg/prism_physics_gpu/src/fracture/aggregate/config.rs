//! Tunables and fixed-point helpers for the `GPU` per-fragment aggregator.
//!
//! Once the Voronoi classifier has binned a debris point cloud into fragment
//! cells, a destruction system needs each fragment's rigid-body seed: its total
//! mass, centre of mass, and inertia tensor. This module holds the aggregation
//! knobs and the shared quantisation used to accumulate those moments with
//! integer atomics.
//!
//! # Fixed-point accumulation
//!
//! `WGSL` atomics operate only on 32-bit integers, so the mass, first moment
//! (mass times position), and second moment (mass times the position outer
//! product) are accumulated as fixed-point `i32` values: each contribution is
//! multiplied by a scale, rounded to the nearest integer (ties to even), and
//! added with an integer atomic. Integer addition is exact and order
//! independent, so the massively parallel device sums are bit-identical to the
//! [`cpu_aggregate_fragments`](super::cpu::cpu_aggregate_fragments) golden.
//! The only floating divergence is the final centre-of-mass division and the
//! inertia algebra performed after de-quantisation.
//!
//! # Overflow budget
//!
//! Each accumulator is an `i32`, so every per-fragment sum must stay within
//! `+/- 2^31`. The second moment grows with the *square* of the coordinate
//! magnitude, so its scale defaults well below the mass and first-moment scales.
//! A scene with larger coordinates or more points per fragment should lower the
//! scales; the parity tests deliberately stay inside the documented budget.
//!
//! # Provenance
//!
//! Fixed-point atomic accumulation of mass moments is a standard `GPU`
//! reduction technique; the rigid-body mass, centroid, and inertia formulas are
//! textbook mechanics. This module contains no Unreal Engine source or derived
//! code.

/// Parameters controlling a per-fragment aggregation dispatch.
///
/// The three scales trade fractional precision against the `i32` overflow
/// budget documented on this module; the defaults suit debris binning with
/// unit-order masses and coordinates within a few tens of units.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AggregateConfig {
    /// Fixed-point scale applied to accumulated fragment mass.
    pub mass_scale: f32,
    /// Fixed-point scale applied to the accumulated first moment
    /// (mass times position).
    pub moment_scale: f32,
    /// Fixed-point scale applied to the accumulated second moment
    /// (mass times the position outer product).
    pub second_moment_scale: f32,
}

impl AggregateConfig {
    /// Default mass scale (`2^16`).
    pub const DEFAULT_MASS_SCALE: f32 = 65_536.0;
    /// Default first-moment scale (`2^15`), lower than the mass scale because
    /// the first moment carries the coordinate magnitude.
    pub const DEFAULT_MOMENT_SCALE: f32 = 32_768.0;
    /// Default second-moment scale (`2^12`), lower still because the second
    /// moment carries the *square* of the coordinate magnitude.
    pub const DEFAULT_SECOND_MOMENT_SCALE: f32 = 4_096.0;
}

impl Default for AggregateConfig {
    fn default() -> AggregateConfig {
        AggregateConfig {
            mass_scale: Self::DEFAULT_MASS_SCALE,
            moment_scale: Self::DEFAULT_MOMENT_SCALE,
            second_moment_scale: Self::DEFAULT_SECOND_MOMENT_SCALE,
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
