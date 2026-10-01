//! Heitz–Neyret by-example stochastic tiling weights — CPU golden reference.
//!
//! A tiling texture repeated across a large surface betrays itself: the eye
//! locks onto the periodic pattern long before the texels blur out.  Heitz &
//! Neyret's *High-Performance By-Example Noise using a Histogram-Preserving
//! Blending Operator* (2018) hides that repetition by overlaying several
//! randomly offset copies of the same texture and blending them, so the macro
//! pattern never recurs while the micro statistics stay intact.
//!
//! This module is the backend-neutral reference for the *sampling* half of that
//! operator: given a texture coordinate it returns the three offset copies to
//! fetch and the barycentric weights to blend them with.  The histogram
//! correction (variance-preserving remap through the texture's inverse CDF) is
//! a separate texturing concern; here the three samples are combined by a plain
//! partition-of-unity blend, which is exactly the numerical reference the GPU
//! twin reproduces before the optional histogram pass.
//!
//! The construction tiles UV space with a regular triangular lattice.  The UV
//! is mapped into a skewed grid so each unit cell splits cleanly into two
//! triangles; the enclosing triangle's three vertices each own a deterministic
//! pseudo-random offset (an integer hash of the vertex index).  The barycentric
//! coordinates of the point within its triangle become the blend weights, and
//! each offset copy is sampled at `uv + offset`.  Because the weights are
//! barycentric they are non-negative and sum to one, and because the offsets
//! are keyed on integer lattice vertices the result is perfectly deterministic
//! and seamless across triangle boundaries (adjacent triangles share vertices,
//! hence share offsets along the shared edge).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; only `floor` is needed.
//! * Hashing is pure integer bit-mixing (a murmur-style finaliser); no float
//!   hashing, so results are bit-stable across platforms.
//! * Barycentric weights are non-negative and sum to one (within rounding).
//! * Defensive clamping everywhere: non-finite UVs fall back to the origin,
//!   `scale` is clamped to a positive range, and no `NaN`/`inf` ever escapes.

use bevy_math::{ops, IVec2, Vec2};

/// Smallest usable lattice scale.  A non-positive scale would collapse the
/// triangular grid; we keep a small positive floor so a cell always has area.
const MIN_SCALE: f32 = 1.0e-3;

/// Largest lattice scale honoured, bounding integer vertex magnitudes so the
/// hash input stays well away from overflow-sensitive ranges.
const MAX_SCALE: f32 = 4096.0;

/// Linear map from world/UV space into the skewed triangular grid.
///
/// This is the standard simplex-style skew: `x' = x`, `y' = -tan(30°)·x +
/// (1/cos(30°))·y`, which turns the unit square lattice into a lattice of
/// equilateral-ish triangles so every point lands inside a well-defined
/// triangle with two integer-indexed neighbours.
const SKEW_YX: f32 = -0.577_350_27; // -tan(30°)
const SKEW_YY: f32 = 1.154_700_54; //  1 / cos(30°)

/// Tuning for [`stochastic_tiling`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StochasticConfig {
    /// Lattice frequency: triangles per unit of input UV.  Larger values make
    /// finer tiles and break up repetition at a smaller scale.
    pub scale: f32,
    /// Multiplier on the per-vertex random offset.  `1` gives a full unit of
    /// decorrelation between copies; `0` disables the offsetting (every copy
    /// samples the same place, i.e. a plain repeat).
    pub offset_strength: f32,
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl StochasticConfig {
    /// A sensible default: unit lattice frequency, full offset strength.
    pub const DEFAULT: Self = Self {
        scale: 1.0,
        offset_strength: 1.0,
    };

    /// Builds a configuration, clamping every field to a finite, sane range.
    pub fn new(scale: f32, offset_strength: f32) -> Self {
        let scale = if scale.is_finite() {
            scale.clamp(MIN_SCALE, MAX_SCALE)
        } else {
            1.0
        };
        let offset_strength = if offset_strength.is_finite() {
            offset_strength.clamp(0.0, 1.0)
        } else {
            1.0
        };
        Self {
            scale,
            offset_strength,
        }
    }
}

/// The three offset copies to fetch and blend for one stochastic lookup.
///
/// `uv[i]` is the texture coordinate of copy `i` (in the scaled lattice space),
/// `weight[i]` its barycentric blend weight (non-negative, summing to one), and
/// `vertex[i]` the integer lattice vertex that generated the offset — exposed so
/// callers can verify same-cell behaviour or key further randomness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StochasticTiling {
    /// The three sample coordinates.
    pub uv: [Vec2; 3],
    /// The three barycentric blend weights (sum to one).
    pub weight: [f32; 3],
    /// The three generating lattice vertices.
    pub vertex: [IVec2; 3],
}

impl StochasticTiling {
    /// Sum of the three weights (one for any valid result).
    pub fn weight_sum(self) -> f32 {
        self.weight[0] + self.weight[1] + self.weight[2]
    }

    /// Blends three scalar samples by the barycentric weights.
    pub fn blend_scalar(self, samples: [f32; 3]) -> f32 {
        self.weight[0] * samples[0] + self.weight[1] * samples[1] + self.weight[2] * samples[2]
    }
}

/// Computes the three stochastic-tiling samples and weights for a UV.
///
/// `uv` is the (untiled) texture coordinate; it is scaled by `cfg.scale`, the
/// enclosing lattice triangle is found, and the triangle's three vertices
/// supply deterministic offsets and barycentric weights.  The returned `uv`
/// coordinates are in the scaled space and are ready to fetch a tileable
/// texture; the shader blends the three fetches by `weight`.
pub fn stochastic_tiling(uv: Vec2, cfg: StochasticConfig) -> StochasticTiling {
    let cfg = StochasticConfig::new(cfg.scale, cfg.offset_strength);
    let base_uv = sanitize_vec2(uv) * cfg.scale;

    // Map into the skewed lattice.
    let skewed = Vec2::new(base_uv.x, SKEW_YX * base_uv.x + SKEW_YY * base_uv.y);
    let cell = Vec2::new(ops::floor(skewed.x), ops::floor(skewed.y));
    let base_id = IVec2::new(cell.x as i32, cell.y as i32);
    let f = skewed - cell; // fractional position within the cell, in [0, 1)

    // Split the cell into two triangles; pick the one containing `f`.
    let s = 1.0 - f.x - f.y;
    let (w, v0, v1, v2) = if s > 0.0 {
        // Lower triangle: vertices (0,0), (0,1), (1,0).
        (
            [s, f.y, f.x],
            base_id,
            base_id + IVec2::new(0, 1),
            base_id + IVec2::new(1, 0),
        )
    } else {
        // Upper triangle: vertices (1,1), (1,0), (0,1).
        (
            [-s, 1.0 - f.y, 1.0 - f.x],
            base_id + IVec2::new(1, 1),
            base_id + IVec2::new(1, 0),
            base_id + IVec2::new(0, 1),
        )
    };

    // Normalise defensively (the barycentric sum is 1 analytically, but clamp
    // negatives from rounding and renormalise so the contract always holds).
    let w = [w[0].max(0.0), w[1].max(0.0), w[2].max(0.0)];
    let sum = w[0] + w[1] + w[2];
    let weight = if sum > f32::MIN_POSITIVE {
        [w[0] / sum, w[1] / sum, w[2] / sum]
    } else {
        [1.0, 0.0, 0.0]
    };

    let o = cfg.offset_strength;
    let uv0 = base_uv + hash_offset(v0) * o;
    let uv1 = base_uv + hash_offset(v1) * o;
    let uv2 = base_uv + hash_offset(v2) * o;

    StochasticTiling {
        uv: [uv0, uv1, uv2],
        weight,
        vertex: [v0, v1, v2],
    }
}

/// Deterministic pseudo-random offset in `[0, 1)^2` for a lattice vertex.
fn hash_offset(v: IVec2) -> Vec2 {
    let xi = v.x as u32;
    let yi = v.y as u32;
    // Combine the two coordinates, then derive two independent channels with
    // distinct salts so the X and Y offsets are uncorrelated.
    let seed = fmix32(xi.wrapping_mul(0x9e37_79b1) ^ fmix32(yi.wrapping_mul(0x85eb_ca77)));
    let hx = fmix32(seed ^ 0x27d4_eb2f);
    let hy = fmix32(seed ^ 0xb529_7a4d);
    Vec2::new(to_unit_f32(hx), to_unit_f32(hy))
}

/// Murmur3 32-bit finaliser: a bijective avalanche mixer over `u32`.
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

/// Maps a `u32` into `[0, 1)` using its top 24 bits (full float mantissa).
fn to_unit_f32(u: u32) -> f32 {
    (u >> 8) as f32 / 16_777_216.0
}

/// Replaces any non-finite component of a `Vec2` with `0`.
fn sanitize_vec2(v: Vec2) -> Vec2 {
    Vec2::new(finite_or_zero(v.x), finite_or_zero(v.y))
}

/// Returns `x` when finite, otherwise `0`.
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Barycentric weights are always a partition of unity.
    #[test]
    fn weights_sum_to_one() {
        let cfg = StochasticConfig::DEFAULT;
        for i in 0..13 {
            for j in 0..13 {
                let uv = Vec2::new(i as f32 * 0.137, j as f32 * 0.211);
                let t = stochastic_tiling(uv, cfg);
                assert!((t.weight_sum() - 1.0).abs() < 1e-5, "sum {} at {:?}", t.weight_sum(), uv);
                assert!(t.weight.iter().all(|&w| w >= 0.0), "negative weight {:?}", t.weight);
            }
        }
    }

    /// Repeated calls with identical input yield identical output.
    #[test]
    fn deterministic() {
        let cfg = StochasticConfig::new(2.5, 1.0);
        let uv = Vec2::new(3.1415, 2.7182);
        let a = stochastic_tiling(uv, cfg);
        let b = stochastic_tiling(uv, cfg);
        assert_eq!(a, b);
    }

    /// Two points inside the same lattice triangle share the same three
    /// generating vertices (hence the same offsets) — only the weights differ.
    #[test]
    fn same_cell_consistency() {
        let cfg = StochasticConfig::DEFAULT;
        // A reference point and a tiny nudge that stays inside the triangle.
        let base = Vec2::new(0.30, 0.30);
        let near = base + Vec2::new(0.001, 0.0005);
        let a = stochastic_tiling(base, cfg);
        let b = stochastic_tiling(near, cfg);
        assert_eq!(a.vertex, b.vertex, "same triangle must share vertices");
        // The offsets are vertex-keyed, so the per-copy base offset matches.
        for k in 0..3 {
            let off_a = a.uv[k] - base * cfg.scale;
            let off_b = b.uv[k] - near * cfg.scale;
            assert!((off_a - off_b).length() < 1e-5, "offset mismatch at {}", k);
        }
    }

    /// Distinct lattice vertices produce distinct offsets (hash decorrelation).
    #[test]
    fn hash_offsets_decorrelate() {
        let a = hash_offset(IVec2::new(0, 0));
        let b = hash_offset(IVec2::new(1, 0));
        let c = hash_offset(IVec2::new(0, 1));
        assert!((a - b).length() > 1e-3, "neighbours should differ: {:?} {:?}", a, b);
        assert!((a - c).length() > 1e-3, "neighbours should differ: {:?} {:?}", a, c);
        // Every offset stays in the unit square.
        for v in [a, b, c] {
            assert!((0.0..1.0).contains(&v.x) && (0.0..1.0).contains(&v.y), "offset range {:?}", v);
        }
    }

    /// A constant blended across the three samples is preserved exactly
    /// (partition of unity => the operator reproduces constants).
    #[test]
    fn blend_preserves_constant() {
        let cfg = StochasticConfig::new(1.7, 0.8);
        for i in 0..7 {
            let uv = Vec2::new(i as f32 * 0.33, i as f32 * 0.19);
            let t = stochastic_tiling(uv, cfg);
            let blended = t.blend_scalar([0.6, 0.6, 0.6]);
            assert!((blended - 0.6).abs() < 1e-5, "constant drift {} at {:?}", blended, uv);
        }
    }

    /// Zero offset strength degenerates to a plain repeat: all three copies
    /// sample the same coordinate.
    #[test]
    fn zero_offset_is_plain_repeat() {
        let cfg = StochasticConfig::new(1.0, 0.0);
        let uv = Vec2::new(0.42, 0.58);
        let t = stochastic_tiling(uv, cfg);
        assert!((t.uv[0] - t.uv[1]).length() < 1e-6);
        assert!((t.uv[0] - t.uv[2]).length() < 1e-6);
    }

    /// Non-finite UVs are sanitised to a finite, valid result.
    #[test]
    fn non_finite_uv_is_safe() {
        let t = stochastic_tiling(Vec2::new(f32::NAN, f32::INFINITY), StochasticConfig::DEFAULT);
        assert!((t.weight_sum() - 1.0).abs() < 1e-5);
        for uv in t.uv {
            assert!(uv.is_finite(), "sample uv finite: {:?}", uv);
        }
    }
}
