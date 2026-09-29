//! Guide-to-render strand interpolation (clumping, curl, randomization).
//!
//! A groom simulates only a small set of *guide* strands, then interpolates
//! them into the many *render* strands that actually draw (pipeline stage 1 in
//! the hair design doc, mirroring `UE5` Groom, AMD `TressFX`, and NVIDIA
//! `HairWorks` at the algorithm level without reusing their code). This module
//! owns that pure, deterministic expansion:
//!
//! 1. **Weighted blend** — each render strand carries a [`RenderStrandBinding`]
//!    of up to [`GUIDE_INFLUENCE_COUNT`] nearest guides with barycentric-style
//!    weights; every control point is the weighted mix of the corresponding
//!    guide control points.
//! 2. **Clumping** — the strand is pulled toward a representative guide of its
//!    clump, more strongly toward the tip, so render strands gather into locks.
//! 3. **Curl** — a seed-stable helix is added in the plane orthogonal to the
//!    strand tangent, growing from root to tip.
//! 4. **Randomization** — per-strand position jitter and length jitter break up
//!    the regularity, all derived from a fixed integer hash so a given seed
//!    always yields the same strand.
//!
//! Everything is array-in / array-out and free of real randomness: the only
//! entropy source is [`hash_to_unit`], a hand-written integer hash, so results
//! are golden-testable and reproducible frame to frame. The functions here also
//! bind to the LOD ladder in [`super::lod`]: [`resolved_render_count`] reads how
//! many render strands a decision keeps, and [`decimate_bindings`] thins a
//! binding list by that count with a deterministic, order-preserving stride —
//! the same integer-decimation philosophy the LOD tiers already use.

use alloc::vec::Vec;
use core::f32::consts::{FRAC_PI_2, PI, TAU};
use core::ops::{Add, Mul, Sub};

use super::lod::HairLodDecision;

/// Number of guide strands that influence one render strand.
///
/// Four nearest guides give a smooth barycentric-style blend on a surface
/// without paying for a dense influence set; this matches production grooms.
pub const GUIDE_INFLUENCE_COUNT: usize = 4;

/// A minimal 3-component vector, hand-written to keep the crate dependency-free.
#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    /// Cartesian x component.
    pub x: f32,
    /// Cartesian y component.
    pub y: f32,
    /// Cartesian z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot product.
    #[must_use]
    pub fn dot(self, rhs: Vec3) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Linear interpolation from `self` toward `rhs` by `t`.
    #[must_use]
    pub fn lerp(self, rhs: Vec3, t: f32) -> Vec3 {
        self + (rhs - self).scale(t)
    }

    /// Returns the unit vector, or `fallback` when `self` is (near) zero length.
    ///
    /// The zero-length guard keeps curl framing defined even when two guide
    /// control points coincide, so the interpolator never divides by zero.
    #[must_use]
    pub fn normalize_or(self, fallback: Vec3) -> Vec3 {
        let len_sq = self.dot(self);
        if len_sq <= f32::EPSILON {
            fallback
        } else {
            self.scale(1.0 / len_sq.sqrt())
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Vec3 {
        self.scale(rhs)
    }
}

/// A fixed-seed integer hash mapping `(seed, key)` to a unit float in `[0, 1)`.
///
/// This is a `splitmix64`-style finalizer over the two 32-bit inputs packed into
/// one 64-bit word; the top 24 bits become the mantissa of the result. The same
/// inputs always produce the same output, which is what makes clumping, curl
/// phase, and jitter deterministic and golden-testable. It is emphatically not a
/// cryptographic hash and never touches a real random source.
#[must_use]
pub fn hash_to_unit(seed: u32, key: u32) -> f32 {
    // 2^24; the mantissa width we keep, guaranteeing the ratio stays below 1.0.
    const UNIT_SCALE: f32 = 16_777_216.0;
    let bits = hash_bits(seed, key) >> 40;
    // `bits` holds 24 bits, so it is in `0..=16_777_215` and the ratio is `< 1`.
    (bits as f32) / UNIT_SCALE
}

/// Full 64-bit `splitmix64` finalizer over the packed `(seed, key)` word.
fn hash_bits(seed: u32, key: u32) -> u64 {
    let mut z = ((u64::from(seed) << 32) | u64::from(key)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A signed hash in `[-1, 1)`, used for symmetric jitter offsets.
fn hash_to_signed(seed: u32, key: u32) -> f32 {
    hash_to_unit(seed, key).mul_add(2.0, -1.0)
}

/// Derives a stable sub-key from an integer purpose salt plus two coordinates.
///
/// Keeping key derivation in one place ensures per-point and per-axis hashes
/// stay independent yet reproducible.
fn sub_key(salt: u32, a: u32, b: u32) -> u32 {
    salt.wrapping_add(a.wrapping_mul(0x27D4_EB2F))
        .wrapping_add(b.wrapping_mul(0x1656_67B1))
}

/// Deterministic sine, hand-written to avoid `f32::sin`.
///
/// The workspace bans the libm-backed transcendentals (see the `clippy.toml`
/// `disallowed-methods` list) because their results are not bit-reproducible
/// across platforms, which would break golden tests. This range-reduces the
/// argument to `[-PI/2, PI/2]` and evaluates a 9th-order Taylor polynomial in
/// Horner form; the worst-case error over a full period stays well under
/// `1e-4`, ample for curl framing and fully deterministic.
fn sin_turns(x: f32) -> f32 {
    // Reduce modulo 2*PI into `[-PI, PI]`.
    let mut a = x - (x / TAU).round() * TAU;
    // Fold into `[-PI/2, PI/2]` using sin(PI - a) == sin(a).
    if a > FRAC_PI_2 {
        a = PI - a;
    } else if a < -FRAC_PI_2 {
        a = -PI - a;
    }
    let x2 = a * a;
    // sin(a) = a * (1 - x2/6 + x2^2/120 - x2^3/5040 + x2^4/362880).
    let poly = x2
        .mul_add(1.0 / 362_880.0, -1.0 / 5_040.0)
        .mul_add(x2, 1.0 / 120.0)
        .mul_add(x2, -1.0 / 6.0)
        .mul_add(x2, 1.0);
    a * poly
}

/// One render strand's binding to its influencing guide strands.
///
/// `guides[i]` indexes into the guide slice passed to
/// [`interpolate_render_strand`]; `weights[i]` is its non-negative contribution.
/// Weights are expected to be barycentric (non-negative, summing to one) but are
/// re-normalized over the in-range, positive-weight guides at use, so a stale
/// index or a zero weight is skipped rather than trusted blindly. `root_uv` is
/// the strand root's surface parameterization (carried for downstream masks and
/// shading), and `seed` drives all deterministic randomization for this strand.
#[derive(Clone, Copy, Debug)]
pub struct RenderStrandBinding {
    /// Guide indices, nearest first; out-of-range entries are ignored.
    pub guides: [u32; GUIDE_INFLUENCE_COUNT],
    /// Per-guide weights, re-normalized over the valid, positive entries.
    pub weights: [f32; GUIDE_INFLUENCE_COUNT],
    /// Root surface UV, preserved through interpolation for downstream stages.
    pub root_uv: (f32, f32),
    /// Deterministic per-strand seed for clumping, curl phase, and jitter.
    pub seed: u32,
}

/// Tunable parameters shared by every render strand of a groom.
///
/// `clump_strength`, `position_jitter`, and `length_jitter` are normalized to
/// `0..=1` (out-of-range values are clamped by
/// [`InterpolationParams::sanitized`]); `curl_frequency` and `curl_amplitude`
/// are clamped to be non-negative. All clamping happens once up front so the hot
/// loop stays branch-light.
#[derive(Clone, Copy, Debug)]
pub struct InterpolationParams {
    /// Number of clump buckets a strand can be assigned to (`0`/`1` ⇒ one clump).
    pub clump_count: u32,
    /// How strongly the tip is pulled to its clump guide, in `0..=1`.
    pub clump_strength: f32,
    /// Curl oscillations along the strand length (non-negative).
    pub curl_frequency: f32,
    /// Peak curl displacement at the tip in world units (non-negative).
    pub curl_amplitude: f32,
    /// Peak per-point positional jitter in world units, in `0..=1`.
    pub position_jitter: f32,
    /// Fractional strand-length randomization in `0..=1`.
    pub length_jitter: f32,
}

impl InterpolationParams {
    /// Returns a copy with strengths/jitters clamped to `0..=1` and curl terms
    /// clamped to be non-negative, so downstream math never sees stray values.
    #[must_use]
    pub fn sanitized(self) -> InterpolationParams {
        InterpolationParams {
            clump_count: self.clump_count,
            clump_strength: self.clump_strength.clamp(0.0, 1.0),
            curl_frequency: self.curl_frequency.max(0.0),
            curl_amplitude: self.curl_amplitude.max(0.0),
            position_jitter: self.position_jitter.clamp(0.0, 1.0),
            length_jitter: self.length_jitter.clamp(0.0, 1.0),
        }
    }
}

// Purpose salts keep the different deterministic draws from the same strand seed
// statistically independent.
const CLUMP_ID_SALT: u32 = 0x00C1_0107;
const CURL_PHASE_SALT: u32 = 0x00C0_1201;
const LENGTH_JITTER_SALT: u32 = 0x001E_4671;
const POSITION_JITTER_SALT: u32 = 0x0050_5A17;

/// A guide selected as contributing to a render strand.
#[derive(Clone, Copy)]
struct Contribution {
    /// Index into the guide slice.
    index: usize,
    /// Re-normalized weight in `0..=1`.
    weight: f32,
}

/// Interpolates one render strand from its guides into `out`.
///
/// `guides[i]` is the polyline of guide `i` (its control points, root first).
/// The blended strand length is the shortest contributing guide's control-point
/// count, so mismatched guide resolutions never index out of bounds. `out` is
/// cleared first and then filled with the final control points; if the binding
/// has no in-range, positive-weight, non-empty guide, `out` is left empty and
/// nothing panics.
///
/// The transform order is blend → length jitter → clump → curl → position
/// jitter. Curl framing is computed from the pre-curl (clumped) polyline so the
/// helix rides a stable tangent, and jitter/curl both scale from zero at the
/// root to full at the tip, keeping the strand rooted while its tip is free.
pub fn interpolate_render_strand(
    guides: &[&[Vec3]],
    binding: &RenderStrandBinding,
    params: InterpolationParams,
    out: &mut Vec<Vec3>,
) {
    out.clear();
    let params = params.sanitized();

    // Gather in-range, positive-weight, non-empty guides and their total weight.
    let mut contrib: [Contribution; GUIDE_INFLUENCE_COUNT] = [Contribution {
        index: 0,
        weight: 0.0,
    }; GUIDE_INFLUENCE_COUNT];
    let mut contrib_len = 0usize;
    let mut weight_sum = 0.0f32;
    let mut rep = 0usize;
    let mut rep_weight = -1.0f32;
    let mut min_len = usize::MAX;

    for (raw_weight, &guide_index) in binding.weights.iter().zip(binding.guides.iter()) {
        let raw_weight = raw_weight.max(0.0);
        if raw_weight <= 0.0 {
            continue;
        }
        let index = guide_index as usize;
        let Some(points) = guides.get(index) else {
            continue;
        };
        if points.is_empty() {
            continue;
        }
        contrib[contrib_len] = Contribution {
            index,
            weight: raw_weight,
        };
        contrib_len += 1;
        weight_sum += raw_weight;
        if points.len() < min_len {
            min_len = points.len();
        }
        if raw_weight > rep_weight {
            rep_weight = raw_weight;
            rep = index;
        }
    }

    if contrib_len == 0 || weight_sum <= 0.0 {
        return;
    }
    let len = min_len;
    let inv_sum = 1.0 / weight_sum;
    let rep_guide = guides[rep];

    // Pass 1: weighted blend, then length jitter relative to the root.
    let length_factor =
        1.0 + hash_to_signed(binding.seed, LENGTH_JITTER_SALT) * params.length_jitter;
    let mut shaped: Vec<Vec3> = Vec::with_capacity(len);
    for (i, _) in rep_guide.iter().take(len).enumerate() {
        let mut point = Vec3::ZERO;
        for c in &contrib[..contrib_len] {
            point = point + guides[c.index][i] * (c.weight * inv_sum);
        }
        shaped.push(point);
    }
    if let Some(&root) = shaped.first() {
        for point in &mut shaped {
            *point = root + (*point - root) * length_factor;
        }
    }

    // Clump pull toward the representative guide, increasing toward the tip.
    if params.clump_strength > 0.0 {
        for (i, point) in shaped.iter_mut().enumerate() {
            let t = strand_param(i, len);
            let pull = params.clump_strength * t;
            *point = point.lerp(rep_guide[i], pull);
        }
    }

    // Curl phase is stable per strand and per clump bucket.
    let clump_id = clump_bucket(binding.seed, params.clump_count);
    let clump_seed = binding.seed ^ clump_id.wrapping_mul(0x9E37_79B1);
    let phase = hash_to_unit(clump_seed, CURL_PHASE_SALT) * TAU;

    // Pass 2: add curl (from the clumped tangent) and positional jitter.
    for (i, &shaped_point) in shaped.iter().enumerate() {
        let t = strand_param(i, len);
        let mut point = shaped_point;

        if params.curl_amplitude > 0.0 {
            let tangent = strand_tangent(&shaped, i, len);
            let (n1, n2) = orthonormal_basis(tangent);
            let angle = params.curl_frequency * t * TAU + phase;
            let amplitude = params.curl_amplitude * t;
            let offset = (n1 * sin_turns(angle) + n2 * sin_turns(angle + FRAC_PI_2)) * amplitude;
            point = point + offset;
        }

        if params.position_jitter > 0.0 {
            let jx = hash_to_signed(binding.seed, sub_key(POSITION_JITTER_SALT, i as u32, 0));
            let jy = hash_to_signed(binding.seed, sub_key(POSITION_JITTER_SALT, i as u32, 1));
            let jz = hash_to_signed(binding.seed, sub_key(POSITION_JITTER_SALT, i as u32, 2));
            point = point + Vec3::new(jx, jy, jz) * (params.position_jitter * t);
        }

        out.push(point);
    }
}

/// Normalized strand parameter `t` in `0..=1` for control point `i` of `len`.
fn strand_param(i: usize, len: usize) -> f32 {
    if len > 1 {
        (i as f32) / ((len - 1) as f32)
    } else {
        0.0
    }
}

/// Forward-difference tangent at control point `i`, root/tip clamped.
fn strand_tangent(points: &[Vec3], i: usize, len: usize) -> Vec3 {
    let raw = if len <= 1 {
        Vec3::new(0.0, 1.0, 0.0)
    } else if i + 1 < len {
        points[i + 1] - points[i]
    } else {
        points[i] - points[i - 1]
    };
    raw.normalize_or(Vec3::new(0.0, 1.0, 0.0))
}

/// Builds two orthonormal vectors spanning the plane normal to `tangent`.
fn orthonormal_basis(tangent: Vec3) -> (Vec3, Vec3) {
    let helper = if tangent.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let n1 = tangent.cross(helper).normalize_or(Vec3::new(0.0, 0.0, 1.0));
    let n2 = tangent.cross(n1).normalize_or(Vec3::new(1.0, 0.0, 0.0));
    (n1, n2)
}

/// Assigns a strand to a stable clump bucket in `0..clump_count`.
fn clump_bucket(seed: u32, clump_count: u32) -> u32 {
    if clump_count <= 1 {
        return 0;
    }
    let scaled = hash_to_unit(seed, CLUMP_ID_SALT) * (clump_count as f32);
    (scaled.floor() as u32).min(clump_count - 1)
}

/// Returns how many render strands a resolved LOD decision keeps.
///
/// Strand-based tiers report [`HairLodDecision::render_strands`]; card and mesh
/// proxies keep no per-strand geometry and report `0`, matching the LOD
/// contract in [`super::lod`].
#[must_use]
pub fn resolved_render_count(decision: &HairLodDecision) -> u32 {
    if decision.tier.is_strand_based() {
        decision.render_strands
    } else {
        0
    }
}

/// Thins a render-strand binding list down to a resolved LOD's strand count.
///
/// The kept count is [`resolved_render_count`]. Selection delegates to
/// [`super::decimation::decimate_bindings_nested`], a deterministic,
/// order-preserving **nested** decimation: the strands kept at a low count are
/// always a subset of those kept at a higher count, so a groom drops the *same*
/// render strands every frame and never pops — at *any* target count, not only
/// counts sharing integer factors (the failure mode of the old stride sample).
/// `out` is cleared first; a `0` count (proxy tier) yields an empty list, and a
/// count at or above the input length copies every binding in order. When
/// strand geometry is available, prefer
/// [`super::decimation::decimate_bindings_importance`] for importance-weighted
/// survival (long, curly, or artist-prioritized strands persist to the lowest
/// counts).
pub fn decimate_bindings(
    bindings: &[RenderStrandBinding],
    decision: &HairLodDecision,
    out: &mut Vec<RenderStrandBinding>,
) {
    super::decimation::decimate_bindings_nested(bindings, decision, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::lod::HairLodDecision;
    use crate::hair::{HairGroupHandle, HairLodTier};
    use alloc::vec::Vec;

    const EPS: f32 = 1e-5;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a.x - b.x).abs() < EPS && (a.y - b.y).abs() < EPS && (a.z - b.z).abs() < EPS
    }

    fn straight_guide(x: f32, points: usize) -> Vec<Vec3> {
        (0..points).map(|i| Vec3::new(x, i as f32, 0.0)).collect()
    }

    fn quiet_params() -> InterpolationParams {
        InterpolationParams {
            clump_count: 0,
            clump_strength: 0.0,
            curl_frequency: 0.0,
            curl_amplitude: 0.0,
            position_jitter: 0.0,
            length_jitter: 0.0,
        }
    }

    fn binding(
        guides: [u32; GUIDE_INFLUENCE_COUNT],
        weights: [f32; GUIDE_INFLUENCE_COUNT],
        seed: u32,
    ) -> RenderStrandBinding {
        RenderStrandBinding {
            guides,
            weights,
            root_uv: (0.25, 0.75),
            seed,
        }
    }

    fn strand_decision(tier: HairLodTier, render_strands: u32) -> HairLodDecision {
        HairLodDecision {
            handle: HairGroupHandle(0),
            tier,
            render_strands,
            segments_per_strand: 8,
        }
    }

    #[test]
    fn single_guide_full_weight_reproduces_guide() {
        let g0 = straight_guide(0.0, 6);
        let guides: [&[Vec3]; 1] = [&g0];
        let b = binding([0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0], 1);
        let mut out = Vec::new();
        interpolate_render_strand(&guides, &b, quiet_params(), &mut out);
        assert_eq!(out.len(), g0.len());
        for (got, want) in out.iter().zip(g0.iter()) {
            assert!(close(*got, *want));
        }
    }

    #[test]
    fn two_guides_half_each_is_midpoint() {
        let g0 = straight_guide(0.0, 5);
        let g1 = straight_guide(2.0, 5);
        let guides: [&[Vec3]; 2] = [&g0, &g1];
        let b = binding([0, 1, 0, 0], [0.5, 0.5, 0.0, 0.0], 7);
        let mut out = Vec::new();
        interpolate_render_strand(&guides, &b, quiet_params(), &mut out);
        assert_eq!(out.len(), 5);
        for (i, got) in out.iter().enumerate() {
            assert!(close(*got, Vec3::new(1.0, i as f32, 0.0)));
        }
    }

    #[test]
    fn hash_to_unit_is_deterministic_and_in_unit_range() {
        for seed in 0..64u32 {
            for key in 0..64u32 {
                let a = hash_to_unit(seed, key);
                let b = hash_to_unit(seed, key);
                assert_eq!(a.to_bits(), b.to_bits());
                assert!(a >= 0.0);
                assert!(a < 1.0);
            }
        }
        // Different inputs must not collapse to a single value.
        assert_ne!(hash_to_unit(1, 2).to_bits(), hash_to_unit(2, 1).to_bits());
    }

    #[test]
    fn curl_disabled_leaves_the_blended_curve() {
        let g0 = straight_guide(0.0, 6);
        let guides: [&[Vec3]; 1] = [&g0];
        let b = binding([0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0], 3);
        let mut out = Vec::new();
        interpolate_render_strand(&guides, &b, quiet_params(), &mut out);
        for (got, want) in out.iter().zip(g0.iter()) {
            assert!(close(*got, *want));
        }
    }

    #[test]
    fn curl_grows_from_root_to_tip() {
        let g0 = straight_guide(0.0, 8);
        let guides: [&[Vec3]; 1] = [&g0];
        let b = binding([0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0], 11);
        let params = InterpolationParams {
            curl_frequency: 3.0,
            curl_amplitude: 0.5,
            ..quiet_params()
        };
        let mut out = Vec::new();
        interpolate_render_strand(&guides, &b, params, &mut out);
        assert_eq!(out.len(), g0.len());
        let root_offset = (out[0] - g0[0]).length();
        let tip_offset = (out[out.len() - 1] - g0[g0.len() - 1]).length();
        assert!(root_offset < EPS);
        assert!(tip_offset > root_offset + 0.1);
    }

    #[test]
    fn full_clump_pulls_tip_to_representative_guide() {
        let g0 = straight_guide(0.0, 6);
        let g1 = straight_guide(4.0, 6);
        let guides: [&[Vec3]; 2] = [&g0, &g1];
        // Guide 0 is the heavier (representative) influence.
        let b = binding([0, 1, 0, 0], [0.7, 0.3, 0.0, 0.0], 5);
        let params = InterpolationParams {
            clump_count: 1,
            clump_strength: 1.0,
            ..quiet_params()
        };
        let mut out = Vec::new();
        interpolate_render_strand(&guides, &b, params, &mut out);
        let tip = out[out.len() - 1];
        assert!(close(tip, g0[g0.len() - 1]));
        // The un-clumped blend tip would sit at x = 0.7*0 + 0.3*4 = 1.2, so the
        // pull materially moved the strand.
        assert!((tip.x - 1.2).abs() > 0.5);
    }

    #[test]
    fn lod_decimation_keeps_count_order_and_is_deterministic() {
        let bindings: Vec<RenderStrandBinding> = (0..40u32)
            .map(|i| binding([i, 0, 0, 0], [1.0, 0.0, 0.0, 0.0], i))
            .collect();
        // Reduced-strands tier keeps N/4 render strands.
        let decision = strand_decision(HairLodTier::ReducedStrands, 10);
        let mut out = Vec::new();
        decimate_bindings(&bindings, &decision, &mut out);
        assert_eq!(out.len(), 10);
        // Kept strands are emitted in ascending original order (coherent
        // downstream processing), not necessarily contiguous.
        for w in out.windows(2) {
            assert!(w[0].seed < w[1].seed);
        }
        // Determinism: a second pass yields the identical selection.
        let mut again = Vec::new();
        decimate_bindings(&bindings, &decision, &mut again);
        let seeds_a: Vec<u32> = out.iter().map(|b| b.seed).collect();
        let seeds_b: Vec<u32> = again.iter().map(|b| b.seed).collect();
        assert_eq!(seeds_a, seeds_b);
        // Pop-free nesting: the 10 kept strands are a strict subset of the 20
        // kept at the next-higher count. A stride sample (40/10=4 vs 40/20=2)
        // would swap survivors here and pop.
        let mut higher = Vec::new();
        decimate_bindings(
            &bindings,
            &strand_decision(HairLodTier::ReducedStrands, 20),
            &mut higher,
        );
        assert_eq!(higher.len(), 20);
        let higher_seeds: Vec<u32> = higher.iter().map(|b| b.seed).collect();
        for kept in &out {
            assert!(higher_seeds.contains(&kept.seed), "kept set must nest");
        }
    }

    #[test]
    fn resolved_render_count_zeroes_proxy_tiers() {
        assert_eq!(
            resolved_render_count(&strand_decision(HairLodTier::Strands, 40_000)),
            40_000
        );
        assert_eq!(
            resolved_render_count(&strand_decision(HairLodTier::ReducedStrands, 10_000)),
            10_000
        );
        let mut cards = strand_decision(HairLodTier::Cards, 0);
        cards.render_strands = 999;
        assert_eq!(resolved_render_count(&cards), 0);
        let mut mesh = strand_decision(HairLodTier::Mesh, 0);
        mesh.render_strands = 5;
        assert_eq!(resolved_render_count(&mesh), 0);
    }

    #[test]
    fn decimation_edge_counts_do_not_panic() {
        let bindings: Vec<RenderStrandBinding> = (0..3u32)
            .map(|i| binding([i, 0, 0, 0], [1.0, 0.0, 0.0, 0.0], i))
            .collect();
        let mut out = Vec::new();
        // Zero target (proxy tier) yields an empty list.
        decimate_bindings(&bindings, &strand_decision(HairLodTier::Cards, 0), &mut out);
        assert!(out.is_empty());
        // Target above length copies everything in order.
        decimate_bindings(
            &bindings,
            &strand_decision(HairLodTier::Strands, 99),
            &mut out,
        );
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].seed, 0);
        assert_eq!(out[2].seed, 2);
        // Empty input stays empty.
        decimate_bindings(&[], &strand_decision(HairLodTier::Strands, 4), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn empty_and_out_of_range_bindings_do_not_panic() {
        let mut out = Vec::new();
        // No guides at all.
        let empty: [&[Vec3]; 0] = [];
        let b = binding([0, 1, 2, 3], [0.25, 0.25, 0.25, 0.25], 9);
        interpolate_render_strand(&empty, &b, quiet_params(), &mut out);
        assert!(out.is_empty());

        // All indices out of range.
        let g0 = straight_guide(0.0, 4);
        let guides: [&[Vec3]; 1] = [&g0];
        let oob = binding([5, 6, 7, 8], [0.4, 0.3, 0.2, 0.1], 2);
        interpolate_render_strand(&guides, &oob, quiet_params(), &mut out);
        assert!(out.is_empty());

        // An empty guide slot is skipped without panicking.
        let g_empty: Vec<Vec3> = Vec::new();
        let two: [&[Vec3]; 2] = [&g_empty, &g0];
        let b2 = binding([0, 1, 0, 0], [0.5, 0.5, 0.0, 0.0], 4);
        interpolate_render_strand(&two, &b2, quiet_params(), &mut out);
        assert_eq!(out.len(), g0.len());
        for (got, want) in out.iter().zip(g0.iter()) {
            assert!(close(*got, *want));
        }
    }

    #[test]
    fn mismatched_guide_lengths_clamp_to_shortest() {
        let g0 = straight_guide(0.0, 5);
        let g1 = straight_guide(2.0, 3);
        let guides: [&[Vec3]; 2] = [&g0, &g1];
        let b = binding([0, 1, 0, 0], [0.5, 0.5, 0.0, 0.0], 6);
        let mut out = Vec::new();
        interpolate_render_strand(&guides, &b, quiet_params(), &mut out);
        assert_eq!(out.len(), 3);
        for (i, got) in out.iter().enumerate() {
            assert!(close(*got, Vec3::new(1.0, i as f32, 0.0)));
        }
    }

    #[test]
    fn zero_weights_produce_no_strand() {
        let g0 = straight_guide(0.0, 4);
        let guides: [&[Vec3]; 1] = [&g0];
        let b = binding([0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0], 1);
        let mut out = Vec::new();
        interpolate_render_strand(&guides, &b, quiet_params(), &mut out);
        assert!(out.is_empty());
    }
}
