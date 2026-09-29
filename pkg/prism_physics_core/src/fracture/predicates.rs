//! Epsilon-consistent geometric predicates for robust cell construction.
//!
//! Voronoi/half-space carving is prone to degenerate configurations: four
//! planes meeting at a point, a vertex sitting exactly on a clipping plane, or
//! two seeds landing on top of each other. Rather than exact arithmetic, this
//! module centralises the tolerances and a deterministic *symbolic
//! perturbation* of seed positions, which mimics the tie-breaking role of the
//! Simulation-of-Simplicity technique so that boundary cases resolve
//! consistently instead of oscillating.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The sign
//! predicate and index-based deterministic perturbation follow the standard,
//! publicly documented Simulation-of-Simplicity idea (Edelsbrunner & Mucke
//! 1990).

use glam::Vec3;

use crate::math::scalar::Real;

/// Classifies a signed value into `-1`, `0`, or `+1` using an absolute
/// tolerance so that near-zero magnitudes collapse to a stable `0`.
#[must_use]
pub fn robust_sign(value: Real, eps: Real) -> i32 {
    if value > eps {
        1
    } else if value < -eps {
        -1
    } else {
        0
    }
}

/// Returns `true` when two points coincide within `eps` (squared-distance
/// test, so no square root is taken).
#[must_use]
pub fn points_close(a: Vec3, b: Vec3, eps: Real) -> bool {
    (a - b).length_squared() <= eps * eps
}

/// A monotonic replacement for `atan2` that maps a 2-D direction to a value in
/// `(-2, 2]` increasing with the counter-clockwise angle.
///
/// This orders the vertices of a convex face without evaluating any
/// transcendental function (which the engine forbids for determinism), using
/// only the sign and the L1-normalised component ratio.
#[must_use]
pub fn pseudo_angle(dx: Real, dy: Real) -> Real {
    let denom = dx.abs() + dy.abs();
    if denom <= Real::EPSILON {
        return 0.0;
    }
    let p = dx / denom;
    if dy < 0.0 {
        p - 1.0
    } else {
        1.0 - p
    }
}

/// Deterministically perturbs a seed position by a tiny index-dependent
/// offset to break exact ties between coincident or co-spherical sites.
///
/// The offset is bounded by `magnitude` and is a pure function of `index`, so
/// results stay reproducible while degenerate inputs are nudged into general
/// position.
#[must_use]
pub fn symbolic_perturbation(index: usize, magnitude: Real) -> Vec3 {
    // Three decorrelated integer hashes mapped into [-1, 1], scaled small.
    let i = index as u64;
    let hx = hash_u64(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1);
    let hy = hash_u64(i.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ 0x2);
    let hz = hash_u64(i.wrapping_mul(0x1656_67B1_9E37_79F9) ^ 0x3);
    Vec3::new(
        unit_signed(hx) * magnitude,
        unit_signed(hy) * magnitude,
        unit_signed(hz) * magnitude,
    )
}

/// Finalising integer hash (`splitmix64` mixer) used by the perturbation.
fn hash_u64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Maps a 64-bit word to a [`Real`] in `[-1, 1)`.
fn unit_signed(bits: u64) -> Real {
    let u = ((bits >> 40) as Real) / ((1u32 << 24) as Real);
    2.0 * u - 1.0
}
