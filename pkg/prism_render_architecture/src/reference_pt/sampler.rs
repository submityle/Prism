//! Deterministic sampling for the reference path tracer.
//!
//! This module owns the random source and the direction/area sampling helpers
//! the integrator relies on. Everything is deterministic (same seed, same
//! sequence of draws) so the renderer is fully reproducible, and no `sin`/`cos`
//! is used: 2D disk samples are drawn by rejection and lifted to a
//! cosine-weighted hemisphere with Malley's method, then rotated into world
//! space through a branchless orthonormal basis.

use alloc::vec::Vec;

use super::Vec3;

/// Default `PCG` stream increment (any odd constant selects a distinct stream).
const DEFAULT_STREAM: u64 = 0xda3e_39cb_94b9_5bdb;

/// `PCG` multiplier (`PCG`-`XSH`-`RR` 64/32 constant from O'Neill, 2014).
const PCG_MULT: u64 = 6_364_136_223_846_793_005;

/// Scale that maps a 24-bit integer into the half-open unit interval `[0, 1)`.
const F32_SCALE: f32 = 1.0 / ((1u32 << 24) as f32);

/// Maximum rejection attempts before the disk sampler falls back to the pole,
/// bounding worst-case work so a pathological draw sequence cannot loop forever.
const MAX_REJECT: u32 = 64;

/// A small, fast, deterministic `PCG`-`XSH`-`RR` 64/32 random number generator.
///
/// This is a classical hashing generator (M. E. O'Neill, 2014), chosen for its
/// reproducibility and good statistical quality; it carries no global state, so
/// each path can own an independent, seed-addressed stream.
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    /// Builds a generator from a `seed` and a stream-selecting `sequence`.
    ///
    /// Two generators with the same `seed` but different `sequence` produce
    /// independent, non-overlapping streams.
    #[must_use]
    pub fn with_stream(seed: u64, sequence: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (sequence << 1) | 1,
        };
        let _ = rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        let _ = rng.next_u32();
        rng
    }

    /// Builds a generator from a `seed` on the default stream.
    #[must_use]
    pub fn seed(seed: u64) -> Self {
        Self::with_stream(seed, DEFAULT_STREAM)
    }

    /// Draws the next 32-bit unsigned integer and advances the state.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Draws the next float uniformly in the half-open interval `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Keep the high 24 bits so the result lands on a clean `f32` lattice.
        ((self.next_u32() >> 8) as f32) * F32_SCALE
    }
}

/// A single stratified 2D sample in the unit square `[0, 1)^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample2 {
    /// First coordinate.
    pub x: f32,
    /// Second coordinate.
    pub y: f32,
}

/// Generates `dim * dim` jittered, stratified samples over the unit square.
///
/// Each cell of a `dim * dim` grid receives exactly one sample jittered within
/// the cell, which drives down variance relative to purely random sampling
/// while remaining unbiased. Returns an empty vector when `dim` is zero.
#[must_use]
pub fn stratified_grid(dim: u32, rng: &mut Rng) -> Vec<Sample2> {
    let mut out = Vec::new();
    if dim == 0 {
        return out;
    }
    let inv = 1.0 / (dim as f32);
    for sy in 0..dim {
        for sx in 0..dim {
            let jx = rng.next_f32();
            let jy = rng.next_f32();
            out.push(Sample2 {
                x: ((sx as f32) + jx) * inv,
                y: ((sy as f32) + jy) * inv,
            });
        }
    }
    out
}

/// Builds a right-handed orthonormal basis `(tangent, bitangent)` around a unit
/// `normal`, using the branchless construction from Duff et al. ("Building an
/// Orthonormal Basis, Revisited", 2017). No trigonometry is involved.
#[must_use]
pub fn orthonormal_basis(normal: Vec3) -> (Vec3, Vec3) {
    let sign = 1.0_f32.copysign(normal.z);
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = Vec3::new(
        1.0 + sign * normal.x * normal.x * a,
        sign * b,
        -sign * normal.x,
    );
    let bitangent = Vec3::new(b, sign + normal.y * normal.y * a, -normal.y);
    (tangent, bitangent)
}

/// A uniform point inside the unit disk, drawn by rejection (no trigonometry).
///
/// Returns the `(x, y)` offset and the squared radius `x^2 + y^2 <= 1`. Falls
/// back to the origin after [`MAX_REJECT`] rejected attempts.
fn concentric_disk_rejection(rng: &mut Rng) -> (f32, f32, f32) {
    let mut attempts = 0;
    loop {
        let x = 2.0 * rng.next_f32() - 1.0;
        let y = 2.0 * rng.next_f32() - 1.0;
        let r2 = x * x + y * y;
        if r2 > 0.0 && r2 <= 1.0 {
            return (x, y, r2);
        }
        attempts += 1;
        if attempts >= MAX_REJECT {
            return (0.0, 0.0, 0.0);
        }
    }
}

/// A uniform sample inside the unit disk `x^2 + y^2 <= 1`, drawn by rejection
/// so no trigonometry is involved. Returns the `(x, y)` offset; the point is
/// uniformly distributed by area, exactly the input that visible-normal
/// (`VNDF`) microfacet sampling expects for its disk warp.
///
/// Falls back to the disk centre `(0, 0)` after [`MAX_REJECT`] rejected draws,
/// bounding worst-case work without introducing bias in practice.
#[must_use]
pub fn uniform_disk(rng: &mut Rng) -> (f32, f32) {
    let (x, y, _r2) = concentric_disk_rejection(rng);
    (x, y)
}

/// A direction drawn from a cosine-weighted hemisphere, with its probability
/// density, both expressed in world space around `normal`.
#[derive(Clone, Copy, Debug)]
pub struct HemisphereSample {
    /// The sampled unit direction in world space.
    pub direction: Vec3,
    /// The solid-angle probability density `cos(theta) / pi` of `direction`.
    pub pdf: f32,
}

/// Samples a direction over the hemisphere around unit `normal` with density
/// proportional to the cosine of the angle to `normal`.
///
/// Implemented with Malley's method: a uniform disk sample `(x, y)` is lifted to
/// `(x, y, sqrt(1 - x^2 - y^2))` in the local frame, which is exactly
/// cosine-distributed, then rotated into world space. The returned `pdf` is
/// `cos(theta) / pi`.
#[must_use]
pub fn cosine_sample_hemisphere(normal: Vec3, rng: &mut Rng) -> HemisphereSample {
    let (x, y, r2) = concentric_disk_rejection(rng);
    let z = (1.0 - r2).max(0.0).sqrt();
    let (tangent, bitangent) = orthonormal_basis(normal);
    let direction = tangent
        .scale(x)
        .add(bitangent.scale(y))
        .add(normal.scale(z))
        .normalize_or_zero();
    // `z` is exactly the cosine of the polar angle for a Malley sample.
    HemisphereSample {
        direction,
        pdf: z * super::INV_PI,
    }
}

/// The cosine-weighted solid-angle density of a direction `wi` about `normal`:
/// `max(cos(theta), 0) / pi`.
#[must_use]
pub fn cosine_hemisphere_pdf(normal: Vec3, wi: Vec3) -> f32 {
    normal.dot(wi).max(0.0) * super::INV_PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::seed(42);
        let mut b = Rng::seed(42);
        for _ in 0..1024 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn rng_streams_differ() {
        let mut a = Rng::with_stream(1, 1);
        let mut b = Rng::with_stream(1, 2);
        let mut differ = false;
        for _ in 0..64 {
            if a.next_u32() != b.next_u32() {
                differ = true;
                break;
            }
        }
        assert!(differ, "independent streams must not be identical");
    }

    #[test]
    fn next_f32_in_unit_interval() {
        let mut rng = Rng::seed(7);
        for _ in 0..100_000 {
            let f = rng.next_f32();
            assert!((0.0..1.0).contains(&f), "f32 draw {f} out of [0,1)");
        }
    }

    #[test]
    fn stratified_grid_count_and_bounds() {
        let mut rng = Rng::seed(99);
        let samples = stratified_grid(8, &mut rng);
        assert_eq!(samples.len(), 64);
        for s in &samples {
            assert!((0.0..1.0).contains(&s.x) && (0.0..1.0).contains(&s.y));
        }
        assert!(stratified_grid(0, &mut rng).is_empty());
    }

    #[test]
    fn stratified_grid_covers_every_cell() {
        let mut rng = Rng::seed(5);
        let dim = 4u32;
        let samples = stratified_grid(dim, &mut rng);
        // Each cell index must appear exactly once.
        let mut seen = [false; 16];
        for s in &samples {
            let cx = (s.x * dim as f32) as usize;
            let cy = (s.y * dim as f32) as usize;
            let idx = cy * dim as usize + cx;
            assert!(!seen[idx], "cell {idx} sampled twice");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn orthonormal_basis_is_orthonormal() {
        let mut rng = Rng::seed(123);
        for _ in 0..2048 {
            let n = Vec3::new(
                2.0 * rng.next_f32() - 1.0,
                2.0 * rng.next_f32() - 1.0,
                2.0 * rng.next_f32() - 1.0,
            )
            .normalize_or_zero();
            if n.length_squared() < 0.5 {
                continue;
            }
            let (t, b) = orthonormal_basis(n);
            assert!((t.length() - 1.0).abs() < 1e-4, "tangent not unit");
            assert!((b.length() - 1.0).abs() < 1e-4, "bitangent not unit");
            assert!(t.dot(n).abs() < 1e-4, "tangent not perpendicular to n");
            assert!(b.dot(n).abs() < 1e-4, "bitangent not perpendicular to n");
            assert!(t.dot(b).abs() < 1e-4, "tangent/bitangent not perpendicular");
        }
    }

    #[test]
    fn cosine_samples_stay_in_hemisphere() {
        let mut rng = Rng::seed(321);
        let n = Vec3::new(0.0, 1.0, 0.0);
        for _ in 0..50_000 {
            let s = cosine_sample_hemisphere(n, &mut rng);
            assert!(s.direction.dot(n) > -1e-4, "sample fell below the surface");
            assert!(s.direction.is_finite());
            assert!((s.direction.length() - 1.0).abs() < 1e-3);
            assert!(s.pdf >= 0.0);
        }
    }

    #[test]
    fn cosine_pdf_integrates_to_one() {
        // Monte Carlo estimate of the hemisphere integral of the cosine density
        // using uniform-hemisphere reference samples; it must converge to 1.
        let mut rng = Rng::seed(2024);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let count = 400_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            // Uniform direction in the upper hemisphere by cube rejection.
            let dir = loop {
                let v = Vec3::new(
                    2.0 * rng.next_f32() - 1.0,
                    2.0 * rng.next_f32() - 1.0,
                    2.0 * rng.next_f32() - 1.0,
                );
                let l2 = v.length_squared();
                if l2 > 1e-6 && l2 <= 1.0 {
                    let u = v.normalize_or_zero();
                    break if u.dot(n) < 0.0 { u.negate() } else { u };
                }
            };
            // Uniform-hemisphere pdf is 1/(2*pi); estimator is pdf / (1/(2*pi)).
            let pdf = cosine_hemisphere_pdf(n, dir) as f64;
            sum += pdf * (2.0 * super::super::PI as f64);
        }
        let integral = sum / f64::from(count);
        assert!(
            (integral - 1.0).abs() < 2e-2,
            "cosine pdf integral {integral} should converge to 1"
        );
    }
}
