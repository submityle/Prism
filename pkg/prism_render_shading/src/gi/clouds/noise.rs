//! Cloud-modelling noise primitives: deterministic 3D value noise, fractal
//! Brownian motion (FBM), Worley / cellular noise, and the Perlin-Worley blend.
//!
//! Volumetric clouds are sculpted from procedural noise rather than stored
//! textures. This module is the backend-neutral CPU golden reference for that
//! noise, built entirely from *integer lattice hashing* so it is bit-stable
//! across runs and across the GPU twin:
//!
//! * [`value_noise_3d`] — smooth value noise on the integer lattice, hashed
//!   corner values interpolated with the Perlin quintic fade; range `[0, 1]`.
//! * [`value_noise_signed`] — the same field remapped to `[-1, 1]`.
//! * [`fbm`] — fractal Brownian motion: a weighted octave sum of value noise,
//!   normalised to `[0, 1]`.
//! * [`worley_3d`] — cellular noise returning the distance to the nearest
//!   feature point, clamped to `[0, 1]` (small near points, large in voids).
//! * [`worley_fbm`] — an octave sum of [`worley_3d`], range `[0, 1]`.
//! * [`perlin_worley`] — Schneider's cloud base-shape blend of FBM with
//!   inverted Worley billows, range `[0, 1]`.
//!
//! # Conventions
//! * All randomness comes from hashing *integer* lattice coordinates with
//!   `wrapping_mul` + xor-shift bit mixing (the `lowbias32` finaliser). There
//!   is no RNG, I/O, GPU, or `unsafe`; every function is a deterministic pure
//!   function of its arguments.
//! * Scalar noise fields are clamped to their documented range and are always
//!   finite (never `NaN`); non-finite inputs fall back to the neutral value
//!   `0.5` (or `0.0` for the signed field) so the marcher never diverges.
//! * Transcendental maths goes through [`bevy_math::ops`]; `sqrt` uses the
//!   inherent method. `f32` arithmetic mirrors the GPU twin bit-for-bit.
//! * Octave counts are clamped to `[1, 8]`; `gain` to `[0, 1]` and
//!   `lacunarity` to `[1, 8]` so the FBM amplitude series stays convergent and
//!   bounded.

use bevy_math::{ops, IVec3, Vec3};

/// Maximum octave count honoured by the fractal summations.
const MAX_OCTAVES: u32 = 8;
/// `2^24`, the mantissa range used to turn a hash into a unit float.
const HASH_UNIT_SCALE: f32 = 16_777_216.0;

/// Integer bit-mixing finaliser (`lowbias32`): maps a `u32` to a well-mixed
/// `u32` with low statistical bias.
#[inline]
fn mix_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Hashes an integer lattice coordinate to a well-mixed `u32`.
///
/// Each axis is folded in with a distinct odd multiplier so permutations of the
/// coordinates (e.g. `(1, 2, 3)` vs `(3, 2, 1)`) map to unrelated hashes.
#[inline]
fn hash_cell(cell: IVec3) -> u32 {
    let x = (cell.x as u32).wrapping_mul(0x9e37_79b1);
    let y = (cell.y as u32).wrapping_mul(0x85eb_ca77);
    let z = (cell.z as u32).wrapping_mul(0xc2b2_ae3d);
    mix_u32(mix_u32(mix_u32(x) ^ y) ^ z)
}

/// Converts a hashed `u32` into a float in `[0, 1)` using its top 24 bits.
#[inline]
fn hash_to_unit(h: u32) -> f32 {
    ((h >> 8) as f32) / HASH_UNIT_SCALE
}

/// Perlin quintic fade `t^3 (t (6t - 15) + 10)`, with first and second
/// derivatives vanishing at `0` and `1` for C2 continuity. `t` is clamped to
/// `[0, 1]`.
#[inline]
fn fade(t: f32) -> f32 {
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation `a + (b - a) * t`.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Returns the hashed unit value stored at an integer lattice corner.
#[inline]
fn corner_value(cell: IVec3) -> f32 {
    hash_to_unit(hash_cell(cell))
}

/// Smooth 3D value noise on the integer lattice, in `[0, 1]`.
///
/// The eight hashed corner values of the lattice cell containing `p` are
/// trilinearly interpolated with the quintic [`fade`] weights. The field is
/// deterministic and `C2` continuous. Non-finite components of `p` fall back to
/// the neutral value `0.5`.
#[inline]
pub fn value_noise_3d(p: Vec3) -> f32 {
    if !p.is_finite() {
        return 0.5;
    }
    let fx = ops::floor(p.x);
    let fy = ops::floor(p.y);
    let fz = ops::floor(p.z);
    let base = IVec3::new(fx as i32, fy as i32, fz as i32);
    let frac = Vec3::new(p.x - fx, p.y - fy, p.z - fz);
    let u = Vec3::new(fade(frac.x), fade(frac.y), fade(frac.z));

    let c000 = corner_value(base + IVec3::new(0, 0, 0));
    let c100 = corner_value(base + IVec3::new(1, 0, 0));
    let c010 = corner_value(base + IVec3::new(0, 1, 0));
    let c110 = corner_value(base + IVec3::new(1, 1, 0));
    let c001 = corner_value(base + IVec3::new(0, 0, 1));
    let c101 = corner_value(base + IVec3::new(1, 0, 1));
    let c011 = corner_value(base + IVec3::new(0, 1, 1));
    let c111 = corner_value(base + IVec3::new(1, 1, 1));

    let x00 = lerp(c000, c100, u.x);
    let x10 = lerp(c010, c110, u.x);
    let x01 = lerp(c001, c101, u.x);
    let x11 = lerp(c011, c111, u.x);
    let y0 = lerp(x00, x10, u.y);
    let y1 = lerp(x01, x11, u.y);
    let value = lerp(y0, y1, u.z);
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Signed value noise in `[-1, 1]`: [`value_noise_3d`] remapped about zero.
#[inline]
pub fn value_noise_signed(p: Vec3) -> f32 {
    (value_noise_3d(p) * 2.0 - 1.0).clamp(-1.0, 1.0)
}

/// Clamps an octave count into the honoured `[1, MAX_OCTAVES]` range.
#[inline]
fn clamp_octaves(octaves: u32) -> u32 {
    octaves.clamp(1, MAX_OCTAVES)
}

/// Clamps `lacunarity` into `[1, 8]`, mapping non-finite inputs to `2`.
#[inline]
fn clamp_lacunarity(lacunarity: f32) -> f32 {
    if lacunarity.is_finite() {
        lacunarity.clamp(1.0, 8.0)
    } else {
        2.0
    }
}

/// Clamps `gain` into `[0, 1]`, mapping non-finite inputs to `0.5`.
#[inline]
fn clamp_gain(gain: f32) -> f32 {
    if gain.is_finite() {
        gain.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Fractal Brownian motion: a normalised weighted sum of [`value_noise_3d`]
/// octaves, in `[0, 1]`.
///
/// Starting from unit frequency and amplitude, each octave multiplies the
/// sampling frequency by `lacunarity` and the amplitude by `gain`. Dividing by
/// the summed amplitudes keeps the result in `[0, 1]` regardless of octave
/// count. `octaves` is clamped to `[1, 8]`, `lacunarity` to `[1, 8]`, and
/// `gain` to `[0, 1]` so the amplitude series is bounded and convergent.
#[inline]
pub fn fbm(p: Vec3, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    if !p.is_finite() {
        return 0.5;
    }
    let octaves = clamp_octaves(octaves);
    let lacunarity = clamp_lacunarity(lacunarity);
    let gain = clamp_gain(gain);

    let mut frequency = 1.0f32;
    let mut amplitude = 1.0f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for _ in 0..octaves {
        sum += amplitude * value_noise_3d(p * frequency);
        norm += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }
    if norm > 0.0 {
        (sum / norm).clamp(0.0, 1.0)
    } else {
        value_noise_3d(p)
    }
}

/// Worley / cellular noise: the distance from `p` to the nearest feature point,
/// clamped to `[0, 1]`.
///
/// One pseudo-random feature point is seeded per integer cell (its position is
/// the cell origin plus a hashed `[0, 1)^3` offset); the `3x3x3` neighbourhood
/// around `p` is searched for the closest. The value is small (`-> 0`) at
/// feature points and large (`-> 1`) in the gaps between them — invert it for
/// cloud "billows". Non-finite inputs fall back to `1.0` (empty space).
#[inline]
pub fn worley_3d(p: Vec3) -> f32 {
    if !p.is_finite() {
        return 1.0;
    }
    let base = IVec3::new(
        ops::floor(p.x) as i32,
        ops::floor(p.y) as i32,
        ops::floor(p.z) as i32,
    );
    let mut nearest = f32::INFINITY;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cell = base + IVec3::new(dx, dy, dz);
                let h = hash_cell(cell);
                let offset = Vec3::new(
                    hash_to_unit(h),
                    hash_to_unit(mix_u32(h ^ 0x68bc_21eb)),
                    hash_to_unit(mix_u32(h ^ 0x02e5_be93)),
                );
                let feature = Vec3::new(cell.x as f32, cell.y as f32, cell.z as f32) + offset;
                let dist_sq = (p - feature).length_squared();
                if dist_sq < nearest {
                    nearest = dist_sq;
                }
            }
        }
    }
    if nearest.is_finite() {
        nearest.sqrt().clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Octave sum of [`worley_3d`], normalised to `[0, 1]`.
///
/// Mirrors [`fbm`] but over the cellular field, producing multi-scale billowy
/// detail. Clamping rules match [`fbm`].
#[inline]
pub fn worley_fbm(p: Vec3, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    if !p.is_finite() {
        return 1.0;
    }
    let octaves = clamp_octaves(octaves);
    let lacunarity = clamp_lacunarity(lacunarity);
    let gain = clamp_gain(gain);

    let mut frequency = 1.0f32;
    let mut amplitude = 1.0f32;
    let mut sum = 0.0f32;
    let mut norm = 0.0f32;
    for _ in 0..octaves {
        sum += amplitude * worley_3d(p * frequency);
        norm += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }
    if norm > 0.0 {
        (sum / norm).clamp(0.0, 1.0)
    } else {
        worley_3d(p)
    }
}

/// Remaps `v` from `[lo, hi]` onto `[nlo, nhi]`, clamped to the output range.
///
/// A degenerate input span (`hi <= lo`) collapses to `nlo`.
#[inline]
fn remap01(v: f32, lo: f32, hi: f32, nlo: f32, nhi: f32) -> f32 {
    let span = hi - lo;
    if !(span.abs() > f32::EPSILON) || !span.is_finite() {
        return nlo;
    }
    let t = ((v - lo) / span).clamp(0.0, 1.0);
    let out = nlo + (nhi - nlo) * t;
    if out.is_finite() {
        out
    } else {
        nlo
    }
}

/// Perlin-Worley cloud base-shape blend, in `[0, 1]`.
///
/// Follows Schneider's "Nubis" construction: a Perlin-style [`fbm`] is lifted
/// by inverted Worley billows so dense cores form where both fields agree. The
/// Worley octaves are deliberately coarser (capped at three) to keep the
/// billows large relative to the FBM detail. The result is clamped to
/// `[0, 1]`.
#[inline]
pub fn perlin_worley(p: Vec3, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let perlin = fbm(p, octaves, lacunarity, gain);
    let billow = 1.0 - worley_fbm(p, octaves.min(3), lacunarity, gain);
    // Raise the Perlin field by the billow floor; both already in [0, 1].
    remap01(perlin, billow * 0.5, 1.0, 0.0, 1.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic spread of sample points covering several cells.
    fn sample_points() -> [Vec3; 8] {
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.37, 1.9, -2.4),
            Vec3::new(-5.1, 3.3, 7.2),
            Vec3::new(12.6, -8.8, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.25, -0.75, 2.125),
            Vec3::new(100.4, -50.2, 33.3),
            Vec3::new(3.14159, 2.71828, 1.61803),
        ]
    }

    #[test]
    fn value_noise_is_deterministic_and_in_unit_range() {
        for p in sample_points() {
            let a = value_noise_3d(p);
            let b = value_noise_3d(p);
            assert_eq!(a, b, "value noise not deterministic at {p:?}");
            assert!((0.0..=1.0).contains(&a), "value noise out of range: {a}");
        }
    }

    #[test]
    fn value_noise_matches_corner_hashes_at_integers() {
        for cell in [IVec3::new(0, 0, 0), IVec3::new(2, -3, 5), IVec3::new(-7, 11, 4)] {
            let p = Vec3::new(cell.x as f32, cell.y as f32, cell.z as f32);
            let expected = corner_value(cell);
            assert!((value_noise_3d(p) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn signed_noise_in_symmetric_range() {
        for p in sample_points() {
            let s = value_noise_signed(p);
            assert!((-1.0..=1.0).contains(&s), "signed noise out of range: {s}");
        }
    }

    #[test]
    fn fbm_is_bounded_and_deterministic() {
        for p in sample_points() {
            let a = fbm(p, 5, 2.0, 0.5);
            let b = fbm(p, 5, 2.0, 0.5);
            assert_eq!(a, b);
            assert!((0.0..=1.0).contains(&a), "fbm out of range: {a}");
        }
    }

    #[test]
    fn fbm_clamps_degenerate_parameters() {
        for p in sample_points() {
            let v = fbm(p, 0, f32::NAN, f32::INFINITY);
            assert!((0.0..=1.0).contains(&v), "fbm degenerate out of range: {v}");
            assert!(v.is_finite());
        }
    }

    #[test]
    fn worley_in_unit_range_and_deterministic() {
        for p in sample_points() {
            let a = worley_3d(p);
            let b = worley_3d(p);
            assert_eq!(a, b);
            assert!((0.0..=1.0).contains(&a), "worley out of range: {a}");
        }
    }

    #[test]
    fn worley_is_small_near_a_feature_point() {
        let cell = IVec3::new(0, 0, 0);
        let h = hash_cell(cell);
        let feature = Vec3::new(
            hash_to_unit(h),
            hash_to_unit(mix_u32(h ^ 0x68bc_21eb)),
            hash_to_unit(mix_u32(h ^ 0x02e5_be93)),
        );
        assert!(worley_3d(feature) < 1e-3, "expected near-zero at feature point");
    }

    #[test]
    fn worley_fbm_bounded() {
        for p in sample_points() {
            let v = worley_fbm(p, 4, 2.0, 0.5);
            assert!((0.0..=1.0).contains(&v), "worley fbm out of range: {v}");
        }
    }

    #[test]
    fn perlin_worley_bounded_and_deterministic() {
        for p in sample_points() {
            let a = perlin_worley(p, 5, 2.0, 0.5);
            let b = perlin_worley(p, 5, 2.0, 0.5);
            assert_eq!(a, b);
            assert!((0.0..=1.0).contains(&a), "perlin-worley out of range: {a}");
        }
    }

    #[test]
    fn non_finite_inputs_never_nan() {
        let bad = Vec3::new(f32::NAN, 0.0, f32::INFINITY);
        assert!(value_noise_3d(bad).is_finite());
        assert!(value_noise_signed(bad).is_finite());
        assert!(fbm(bad, 5, 2.0, 0.5).is_finite());
        assert!(worley_3d(bad).is_finite());
        assert!(worley_fbm(bad, 4, 2.0, 0.5).is_finite());
        assert!(perlin_worley(bad, 5, 2.0, 0.5).is_finite());
    }

    #[test]
    fn value_noise_is_continuous_across_cell_boundary() {
        let a = value_noise_3d(Vec3::new(0.999, 0.3, 0.3));
        let b = value_noise_3d(Vec3::new(1.001, 0.3, 0.3));
        assert!((a - b).abs() < 0.05, "discontinuity: a={a} b={b}");
    }
}
