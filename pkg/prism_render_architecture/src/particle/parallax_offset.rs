//! Parallax occlusion mapping (`POM`) `UV`-offset reference: the `CPU`-verifiable
//! contract that turns a tangent-space view ray plus a height field into the
//! offset `UV` a shader should sample, matching the steep-parallax + secant
//! refinement pipeline used by `Unreal`, `CryEngine`, and most `AAA` material
//! stacks (design §16-§21).
//!
//! # What `POM` actually does
//!
//! A flat quad textured with a height field *looks* flat because every fragment
//! samples the albedo at its own `UV`. `POM` fakes real relief by pretending the
//! height field is a stack of parallel layers between a top reference plane
//! (height `1`) and the deepest valley (height `0`). It walks the tangent-space
//! view ray downward through those layers until the ray first passes *below* the
//! sampled surface, then reports the `UV` at that intersection instead of the
//! fragment's own `UV`. Because the intersection drifts against the view
//! direction, near geometry shifts more than far geometry and the flat quad
//! reads as carved stone, brick mortar, or cobblestone.
//!
//! Two stages, exactly as in production:
//!
//! 1. [`ParallaxConfig::steep_parallax`] — *steep parallax*: split the depth
//!    range `[0, 1]` into equidistant layers and step the view ray one layer at
//!    a time until the accumulated ray depth first reaches or exceeds the
//!    sampled surface depth. This localizes the crossing to a single layer but
//!    quantizes it to the layer boundary, which shows up as visible stair steps
//!    on shallow viewing angles.
//! 2. [`secant_refine`] — *secant / linear refinement*: inside the bracketing
//!    layer the ray-minus-surface function changes sign, so one linear (secant)
//!    solve between the last-outside and first-inside samples lands on the true
//!    crossing far more accurately than the raw layer boundary, erasing the
//!    stair steps for a single extra height sample.
//!
//! The layer count is view-dependent: at a head-on angle a few layers suffice,
//! while a grazing angle needs many to keep the step short enough to catch thin
//! features. [`ParallaxConfig::layer_count`] linearly interpolates
//! `max_layers -> min_layers` by the cosine of the view angle (a plain `lerp`,
//! never a `mix` faked with `powf`). The grazing amplification of the offset
//! vector uses the rational `1 / max(cos, eps)` truncation rather than `tan`, so
//! the reference stays free of transcendental calls and reproduces bit for bit
//! on a future `GPU` kernel.
//!
//! An optional [`ParallaxConfig::self_shadow`] marches a second ray toward the
//! light and softens the occlusion boundary with a multiply-only `smoothstep`,
//! the standard cheap contact shadow that sells the relief.
//!
//! # What this module deliberately does *not* do
//!
//! This is strictly the "height-field ray march to a `UV` offset" stage. It does
//! **not** scroll, tile, rotate, or flipbook the `UV` — that animation lives in
//! `uv_animation` and is intentionally untouched here so the two concerns never
//! fight over the same coordinate. It does not pack or address a texture atlas
//! (`atlas_packing` / `billboard_atlas`), and it does not fade translucency
//! against scene depth (`soft_particle`). It imports no sibling particle module
//! except the shared `std430` layout primitives, and its procedural height field
//! draws on a self-contained integer hash rather than the `noise` module.
//!
//! # Determinism
//!
//! The only floating-point primitive beyond ordinary arithmetic is `f32::sqrt`
//! (view-ray normalization); height-field cell lookup uses only integer
//! comparisons and arithmetic. There are no `sin` / `cos` / `tan` / `exp` /
//! `ln` / `pow` calls anywhere; every shaped curve is either a `lerp`, a
//! rational, or the multiply-only `smoothstep` `t^2 (3 - 2 t)`. Bare `==` / `!=`
//! on `f32` is avoided; magnitudes are compared against [`CMP_EPS`].

use crate::particle::gpu_layout::{U32_STRIDE, VEC2_STRIDE};
use alloc::vec::Vec;

/// Absolute tolerance for the degenerate-interval and near-grazing guards, and
/// for the `f32` magnitude comparisons that stand in for forbidden `==` / `!=`.
pub const CMP_EPS: f32 = 1.0e-6;

/// Byte size of the `std430` packing of [`ParallaxConfig`]: two `f32` scalars
/// (`height_scale`, `self_shadow_softness`) followed by two `u32` scalars
/// (`min_layers`, `max_layers`). Four scalars fill exactly one `vec4` slot, laid
/// out as two `vec2`-sized halves so the block honors the 16-byte `std430` base
/// alignment with no padding tail.
pub const PARALLAX_STD430_SIZE: usize = 2 * VEC2_STRIDE;

/// Clamps a scalar into the closed unit interval `[0, 1]`.
///
/// Kept private so the intent — "a normalized fraction never leaves `[0, 1]`" —
/// reads directly at every call site.
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The multiply-only `smoothstep` shaper `t^2 (3 - 2 t)` after clamping `t` into
/// `[0, 1]`.
///
/// This is the soft-edge curve the self-shadow uses; it stays a polynomial so no
/// transcendental (`pow` / `exp`) ever enters the deterministic path.
fn smoothstep01(t: f32) -> f32 {
    let c = clamp01(t);
    c * c * (3.0 - 2.0 * c)
}

/// Normalizes a tangent-space vector, returning `None` for a degenerate
/// (near-zero) input so callers can fall back to "no parallax" instead of
/// dividing by a vanishing length.
///
/// Uses `f32::sqrt` only; the squared-length guard compares against
/// [`CMP_EPS`] squared to avoid a bare `==` on the length.
fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < CMP_EPS * CMP_EPS {
        return None;
    }
    let inv = 1.0 / len_sq.sqrt();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

/// The full tangent-space `UV` displacement vector at maximum depth.
///
/// For a normalized tangent-space view direction `v` (pointing from the surface
/// toward the eye, `z` along the surface normal), the ray reaches the deepest
/// layer after travelling `v.xy / v.z` per unit depth. Scaling by
/// `height_scale` gives the offset at full depth `1`; the per-layer step is this
/// vector divided by the layer count.
///
/// The `1 / max(v.z, eps)` truncation is the rational stand-in for the `tan`
/// that a naive derivation would use: as the view grazes (`v.z -> 0`) the offset
/// grows without a transcendental, and the `eps` floor keeps it finite.
fn full_offset(view_unit: [f32; 3], height_scale: f32) -> [f32; 2] {
    let denom = view_unit[2].max(CMP_EPS);
    let k = height_scale / denom;
    [view_unit[0] * k, view_unit[1] * k]
}

/// One bracketed intersection produced by [`ParallaxConfig::steep_parallax`].
///
/// The steep march leaves the crossing bracketed by two consecutive layer
/// samples: `prev_*` is the last sample whose ray depth was still *above* the
/// surface, and `hit_*` is the first sample whose ray depth reached or passed
/// *below* it. [`secant_refine`] consumes exactly these fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SteepHit {
    /// `UV` at the last outside sample (ray still above the surface).
    pub prev_uv: [f32; 2],
    /// `UV` at the first inside sample (ray at or below the surface).
    pub hit_uv: [f32; 2],
    /// Ray depth at the last outside sample.
    pub prev_layer_depth: f32,
    /// Sampled surface depth (`1 - height`) at the last outside sample.
    pub prev_surface_depth: f32,
    /// Ray depth at the first inside sample.
    pub hit_layer_depth: f32,
    /// Sampled surface depth (`1 - height`) at the first inside sample.
    pub hit_surface_depth: f32,
    /// Whether the march actually crossed the surface. `false` means the ray
    /// stayed above the surface for every layer (a flat top plane, or a view so
    /// shallow the guard cap was reached); callers keep the base `UV`.
    pub penetrated: bool,
}

impl SteepHit {
    /// The ray-minus-surface value at the last outside sample.
    ///
    /// Non-negative when the sample is genuinely outside the surface; feeds the
    /// `before` argument of [`secant_refine`].
    #[must_use]
    pub fn before(self) -> f32 {
        self.prev_surface_depth - self.prev_layer_depth
    }

    /// The ray-minus-surface value at the first inside sample.
    ///
    /// Non-positive when the sample is genuinely inside the surface; feeds the
    /// `after` argument of [`secant_refine`].
    #[must_use]
    pub fn after(self) -> f32 {
        self.hit_surface_depth - self.hit_layer_depth
    }

    /// The secant-refined intersection `UV` for this bracket.
    #[must_use]
    pub fn refined_uv(self) -> [f32; 2] {
        secant_refine(self.prev_uv, self.hit_uv, self.before(), self.after())
    }
}

/// Configuration for a parallax-occlusion `UV` offset.
///
/// `min_layers` / `max_layers` bound the view-dependent layer count (head-on
/// uses `min_layers`, grazing uses `max_layers`); they are `u16` so the
/// conversion to `f32` (for the `lerp`) and to `u32` (for the `std430` pack) are
/// both lossless `From` conversions rather than lossy casts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParallaxConfig {
    /// Depth of the height field in `UV` units: the full offset magnitude when
    /// the view grazes and the surface plunges to height `0`.
    pub height_scale: f32,
    /// Softness of the self-shadow contact edge, in depth units, fed to the
    /// `smoothstep`. A value at or below [`CMP_EPS`] collapses to a hard edge.
    pub self_shadow_softness: f32,
    /// Layer count used at a head-on view (cosine `1`).
    pub min_layers: u16,
    /// Layer count used at a grazing view (cosine `0`).
    pub max_layers: u16,
}

impl ParallaxConfig {
    /// Builds a configuration.
    #[must_use]
    pub fn new(
        height_scale: f32,
        self_shadow_softness: f32,
        min_layers: u16,
        max_layers: u16,
    ) -> Self {
        Self {
            height_scale,
            self_shadow_softness,
            min_layers,
            max_layers,
        }
    }

    /// The view-dependent layer count, as a real number in `[1, inf)`.
    ///
    /// A plain linear interpolation `lerp(max_layers, min_layers, cos)` in the
    /// view cosine: at `cos = 1` (head-on) it returns `min_layers`, at `cos = 0`
    /// (grazing) it returns `max_layers`. The result is bounded to at least `1`
    /// so the per-layer step is always finite. No `powf`-based `mix` is used.
    #[must_use]
    pub fn layer_count(self, cos_view: f32) -> f32 {
        let min_f = f32::from(self.min_layers);
        let max_f = f32::from(self.max_layers);
        let grazing = clamp01(1.0 - clamp01(cos_view));
        let n = min_f + grazing * (max_f - min_f);
        n.max(1.0)
    }

    /// Runs the steep-parallax march for a fragment.
    ///
    /// `base_uv` is the fragment's own `UV`, `view_ts` the tangent-space view
    /// direction (from surface toward eye, `z` along the normal), and `height`
    /// samples the height field in `[0, 1]` (height `1` is the top reference
    /// plane, `0` the deepest valley). Returns `None` for a degenerate view
    /// direction so [`ParallaxConfig::parallax_uv`] can keep the base `UV`.
    ///
    /// The ray depth starts at `0` (the top plane) and grows by one layer per
    /// step while the `UV` slides by the per-layer share of [`full_offset`],
    /// against the view direction. The loop stops at the first layer whose ray
    /// depth reaches or exceeds the sampled surface depth `1 - height`; an
    /// integer guard bounded by `max_layers` caps the iteration count so a
    /// hostile input can never spin forever.
    #[must_use]
    pub fn steep_parallax<F>(
        self,
        base_uv: [f32; 2],
        view_ts: [f32; 3],
        height: &F,
    ) -> Option<SteepHit>
    where
        F: Fn([f32; 2]) -> f32,
    {
        let view_unit = normalize3(view_ts)?;
        let layers = self.layer_count(view_unit[2]);
        let layer_step = 1.0 / layers;
        let full = full_offset(view_unit, self.height_scale);
        let delta_uv = [full[0] * layer_step, full[1] * layer_step];

        let mut cur_uv = base_uv;
        let mut cur_layer_depth = 0.0_f32;
        let mut cur_surface_depth = 1.0 - clamp01(height(cur_uv));

        let mut prev_uv = cur_uv;
        let mut prev_layer_depth = cur_layer_depth;
        let mut prev_surface_depth = cur_surface_depth;

        // The guard is a hard integer cap on iterations, independent of the
        // floating-point layer counter, so the loop always terminates.
        let hard_cap = u32::from(self.max_layers).saturating_add(2);
        let mut steps: u32 = 0;

        while cur_layer_depth < cur_surface_depth && steps < hard_cap {
            prev_uv = cur_uv;
            prev_layer_depth = cur_layer_depth;
            prev_surface_depth = cur_surface_depth;

            cur_uv = [cur_uv[0] - delta_uv[0], cur_uv[1] - delta_uv[1]];
            cur_layer_depth += layer_step;
            cur_surface_depth = 1.0 - clamp01(height(cur_uv));

            steps += 1;
        }

        // Penetration means the ray reached or passed below the surface (the
        // `<` continuation test failed for a reason other than the guard).
        let penetrated = cur_layer_depth >= cur_surface_depth;

        Some(SteepHit {
            prev_uv,
            hit_uv: cur_uv,
            prev_layer_depth,
            prev_surface_depth,
            hit_layer_depth: cur_layer_depth,
            hit_surface_depth: cur_surface_depth,
            penetrated,
        })
    }

    /// The parallax-occlusion offset `UV` for a fragment: steep march followed
    /// by one secant refinement.
    ///
    /// Returns the fragment's own `base_uv` unchanged when the view is
    /// degenerate or the ray never crosses the surface (a flat top plane), so a
    /// caller can always sample the returned coordinate safely.
    #[must_use]
    pub fn parallax_uv<F>(self, base_uv: [f32; 2], view_ts: [f32; 3], height: &F) -> [f32; 2]
    where
        F: Fn([f32; 2]) -> f32,
    {
        match self.steep_parallax(base_uv, view_ts, height) {
            Some(hit) if hit.penetrated => hit.refined_uv(),
            _ => base_uv,
        }
    }

    /// The parallax-occlusion offset `UV` for a batch of fragments sharing this
    /// configuration and view direction.
    ///
    /// A thin convenience over [`ParallaxConfig::parallax_uv`] that keeps the
    /// per-fragment allocation out of hot call sites.
    #[must_use]
    pub fn parallax_uv_batch<F>(
        self,
        base_uvs: &[[f32; 2]],
        view_ts: [f32; 3],
        height: &F,
    ) -> Vec<[f32; 2]>
    where
        F: Fn([f32; 2]) -> f32,
    {
        base_uvs
            .iter()
            .map(|&uv| self.parallax_uv(uv, view_ts, height))
            .collect()
    }

    /// Optional `POM` self-shadow: the fraction of light reaching the resolved
    /// surface point, in `[0, 1]` (`1` fully lit, `0` fully shadowed).
    ///
    /// From the resolved hit (`hit_uv`, `hit_depth`), a second ray marches
    /// *toward* the light in tangent space, rising back toward the top plane.
    /// At each step the sampled surface depth is compared against the shadow
    /// ray's depth: whenever the surface sits *above* the ray it occludes the
    /// light, and the largest such overlap drives the shadow amount through the
    /// multiply-only [`smoothstep01`] softened by `self_shadow_softness`. A
    /// degenerate or grazing light direction (or a softness at or below
    /// [`CMP_EPS`]) yields a hard, well-defined result rather than a divide by a
    /// vanishing quantity.
    #[must_use]
    pub fn self_shadow<F>(
        self,
        hit_uv: [f32; 2],
        hit_depth: f32,
        light_ts: [f32; 3],
        height: &F,
    ) -> f32
    where
        F: Fn([f32; 2]) -> f32,
    {
        let Some(light_unit) = normalize3(light_ts) else {
            return 1.0;
        };
        let start_depth = clamp01(hit_depth);
        if start_depth < CMP_EPS {
            // The point sits on the top plane: nothing can occlude it.
            return 1.0;
        }

        let layers = self.layer_count(light_unit[2]);
        let depth_step = start_depth / layers;
        let full = full_offset(light_unit, self.height_scale);
        let delta_uv = [full[0] * depth_step, full[1] * depth_step];

        let hard_cap = u32::from(self.max_layers).saturating_add(2);
        let mut steps: u32 = 0;

        let mut ray_uv = hit_uv;
        let mut ray_depth = start_depth;
        let mut max_overlap = 0.0_f32;

        while ray_depth > CMP_EPS && steps < hard_cap {
            ray_uv = [ray_uv[0] + delta_uv[0], ray_uv[1] + delta_uv[1]];
            ray_depth -= depth_step;

            let surface_depth = 1.0 - clamp01(height(ray_uv));
            // Surface above the ray (smaller depth) occludes the light.
            let overlap = ray_depth - surface_depth;
            if overlap > max_overlap {
                max_overlap = overlap;
            }

            steps += 1;
        }

        if max_overlap <= 0.0 {
            return 1.0;
        }

        let shadow = if self.self_shadow_softness < CMP_EPS {
            1.0
        } else {
            smoothstep01(max_overlap / self.self_shadow_softness)
        };
        clamp01(1.0 - shadow)
    }

    /// Packs the configuration into its `std430` uniform-block byte layout.
    ///
    /// Layout: `height_scale` and `self_shadow_softness` as two `f32` in the
    /// first `vec2` half, then `min_layers` and `max_layers` as two `u32` in the
    /// second half, filling exactly one `vec4` slot ([`PARALLAX_STD430_SIZE`]
    /// bytes) with no padding tail. `min_layers` / `max_layers` widen to `u32`
    /// through a lossless [`u32::from`].
    #[must_use]
    pub fn to_std430(self) -> [u8; PARALLAX_STD430_SIZE] {
        let mut bytes = [0u8; PARALLAX_STD430_SIZE];
        bytes[0..U32_STRIDE].copy_from_slice(&self.height_scale.to_le_bytes());
        bytes[U32_STRIDE..2 * U32_STRIDE].copy_from_slice(&self.self_shadow_softness.to_le_bytes());
        bytes[2 * U32_STRIDE..3 * U32_STRIDE]
            .copy_from_slice(&u32::from(self.min_layers).to_le_bytes());
        bytes[3 * U32_STRIDE..4 * U32_STRIDE]
            .copy_from_slice(&u32::from(self.max_layers).to_le_bytes());
        bytes
    }
}

/// Solves for the intersection `UV` inside the bracketing layer by a single
/// secant (linear) step.
///
/// `before` is the ray-minus-surface value at the last outside sample (expected
/// non-negative) and `after` the same value at the first inside sample
/// (expected non-positive), so the intersection lies at fraction
/// `before / (before - after)` from the outside `UV` toward the inside `UV`.
/// This is the classic `POM` refinement: it removes the layer-boundary stair
/// step for one extra height sample. A degenerate bracket whose endpoints
/// coincide (`before - after` within [`CMP_EPS`]) returns the inside `UV`
/// unchanged rather than dividing by a vanishing width.
#[must_use]
pub fn secant_refine(prev_uv: [f32; 2], hit_uv: [f32; 2], before: f32, after: f32) -> [f32; 2] {
    let denom = before - after;
    if denom.abs() < CMP_EPS {
        return hit_uv;
    }
    let weight = clamp01(before / denom);
    [
        prev_uv[0] + (hit_uv[0] - prev_uv[0]) * weight,
        prev_uv[1] + (hit_uv[1] - prev_uv[1]) * weight,
    ]
}

/// A self-contained integer hash (a `Wang`-style bit mix) mapping a 32-bit key
/// to a well-distributed 32-bit value.
///
/// Used only to seed the procedural height field below; it exists so this
/// module never reaches into the `noise` sibling for randomness.
#[must_use]
pub fn hash_u32(key: u32) -> u32 {
    let mut x = key;
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_add(x << 3);
    x ^= x >> 4;
    x = x.wrapping_mul(0x27d4_eb2d);
    x ^= x >> 15;
    x
}

/// Hashes a 2D integer lattice cell (plus a seed) to a height in `[0, 1]`.
///
/// The low 16 bits of the mixed hash are taken through a lossless
/// [`u16`]-to-[`f32`] `From` and divided by `u16::MAX`, so no lossy cast is
/// involved and the result is deterministic.
#[must_use]
pub fn lattice_height(ix: u32, iy: u32, seed: u32) -> f32 {
    let mixed = hash_u32(ix.wrapping_mul(0x9e37_79b1) ^ hash_u32(iy).wrapping_add(seed));
    let low = u16::try_from(mixed & 0xffff).expect("value masked into the u16 range");
    f32::from(low) / f32::from(u16::MAX)
}

/// A dense height field stored as a row-major grid of samples in `[0, 1]`, with
/// clamp-to-edge bilinear reconstruction.
///
/// This is the "array" height source complementing the closure-based
/// `impl Fn([f32; 2]) -> f32` samplers the core march accepts: build one and
/// pass `|uv| field.sample(uv)` to [`ParallaxConfig::parallax_uv`].
#[derive(Clone, Debug, PartialEq)]
pub struct HeightField {
    width: usize,
    height: usize,
    samples: Vec<f32>,
}

impl HeightField {
    /// Builds a field from an explicit sample grid.
    ///
    /// Returns `None` when the dimensions are zero or the sample count does not
    /// match `width * height`, so a malformed grid can never be sampled.
    #[must_use]
    pub fn new(width: usize, height: usize, samples: Vec<f32>) -> Option<Self> {
        if width == 0 || height == 0 || samples.len() != width.saturating_mul(height) {
            return None;
        }
        Some(Self {
            width,
            height,
            samples,
        })
    }

    /// A uniform field: every sample equals `value`.
    #[must_use]
    pub fn flat(width: usize, height: usize, value: f32) -> Option<Self> {
        let count = width.saturating_mul(height);
        let samples: Vec<f32> = core::iter::repeat_n(value, count).collect();
        Self::new(width, height, samples)
    }

    /// A field whose samples come from [`lattice_height`], for procedural relief
    /// without importing the `noise` sibling.
    #[must_use]
    pub fn value_noise(width: usize, height: usize, seed: u32) -> Option<Self> {
        let count = width.saturating_mul(height);
        let samples: Vec<f32> = (0..count)
            .map(|index| {
                let ix = index % width;
                let iy = index / width;
                let ix_key = u32::try_from(ix).unwrap_or(u32::MAX);
                let iy_key = u32::try_from(iy).unwrap_or(u32::MAX);
                lattice_height(ix_key, iy_key, seed)
            })
            .collect();
        Self::new(width, height, samples)
    }

    /// The grid width in samples.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The grid height in samples.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Bilinearly samples the field at `uv`, clamping to the grid edges.
    #[must_use]
    pub fn sample(&self, uv: [f32; 2]) -> f32 {
        let (x0, x1, fx) = cell_coord(uv[0], self.width);
        let (y0, y1, fy) = cell_coord(uv[1], self.height);
        let s00 = self.samples[y0 * self.width + x0];
        let s10 = self.samples[y0 * self.width + x1];
        let s01 = self.samples[y1 * self.width + x0];
        let s11 = self.samples[y1 * self.width + x1];
        let top = s00 + (s10 - s00) * fx;
        let bottom = s01 + (s11 - s01) * fx;
        top + (bottom - top) * fy
    }
}

/// Reconstructs the two bracketing integer indices and the interpolation
/// fraction for one `UV` component over a grid of `count` samples.
///
/// The `UV` component is clamped into `[0, 1]` and scaled onto `[0, count - 1]`.
/// The integer floor is found by walking up whole numbers (a bounded loop over
/// small grid dimensions) rather than a lossy float-to-int cast, so the
/// conversion stays exact and cast-free.
fn cell_coord(coord: f32, count: usize) -> (usize, usize, f32) {
    if count <= 1 {
        return (0, 0, 0.0);
    }
    let last = count - 1;
    let last_f = f32::from(u16::try_from(last).unwrap_or(u16::MAX));
    let scaled = clamp01(coord) * last_f;

    let mut i0 = 0usize;
    let mut i0_f = 0.0_f32;
    while i0 < last {
        let next_f = i0_f + 1.0;
        if next_f > scaled {
            break;
        }
        i0 += 1;
        i0_f = next_f;
    }
    let frac = clamp01(scaled - i0_f);
    let i1 = (i0 + 1).min(last);
    (i0, i1, frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::gpu_layout::storage_bytes;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0e-4
    }

    fn cfg() -> ParallaxConfig {
        ParallaxConfig::new(0.1, 0.05, 8, 32)
    }

    #[test]
    fn cmp_eps_is_micro() {
        assert!(approx(CMP_EPS, 1.0e-6));
    }

    #[test]
    fn flat_top_plane_yields_zero_offset() {
        // A field at height 1 everywhere is the top reference plane: no relief,
        // so the offset UV is exactly the base UV.
        let field = HeightField::flat(4, 4, 1.0).unwrap();
        let base = [0.5, 0.5];
        let out = cfg().parallax_uv(base, [0.3, 0.2, 1.0], &|uv| field.sample(uv));
        assert!(approx(out[0], base[0]));
        assert!(approx(out[1], base[1]));
    }

    #[test]
    fn constant_mid_level_offsets_against_view_x() {
        // A uniformly depressed plane still parallaxes: the intersection slides
        // opposite the view's tangential component (we subtract the offset).
        let field = HeightField::flat(4, 4, 0.5).unwrap();
        let base = [0.5, 0.5];
        let out = cfg().parallax_uv(base, [0.5, 0.0, 1.0], &|uv| field.sample(uv));
        assert!(out[0] < base[0]);
        assert!(approx(out[1], base[1]));
    }

    #[test]
    fn degenerate_view_returns_base_uv() {
        let field = HeightField::flat(4, 4, 0.3).unwrap();
        let base = [0.25, 0.75];
        let out = cfg().parallax_uv(base, [0.0, 0.0, 0.0], &|uv| field.sample(uv));
        assert!(approx(out[0], base[0]));
        assert!(approx(out[1], base[1]));
    }

    #[test]
    fn monotonic_ramp_offset_direction_is_correct() {
        // Height decreases as u increases (deeper toward +u). Looking with a +u
        // tangential view component, the resolved point shifts toward -u.
        let ramp = |uv: [f32; 2]| clamp01(1.0 - uv[0]);
        let base = [0.5, 0.5];
        let out = cfg().parallax_uv(base, [0.6, 0.0, 1.0], &ramp);
        assert!(out[0] < base[0]);
    }

    #[test]
    fn steeper_scale_offsets_further() {
        let ramp = |uv: [f32; 2]| clamp01(1.0 - uv[0]);
        let base = [0.5, 0.5];
        let shallow = ParallaxConfig::new(0.05, 0.05, 8, 32);
        let deep = ParallaxConfig::new(0.2, 0.05, 8, 32);
        let a = shallow.parallax_uv(base, [0.6, 0.0, 1.0], &ramp);
        let b = deep.parallax_uv(base, [0.6, 0.0, 1.0], &ramp);
        assert!(b[0] < a[0]);
    }

    #[test]
    fn secant_refine_beats_steep_on_linear_ramp() {
        // A many-layer steep march approximates the true crossing; the refined
        // UV from the coarse march must land at least as close to it as the raw
        // coarse layer boundary.
        let ramp = |uv: [f32; 2]| clamp01(1.0 - uv[0]);
        let base = [0.0, 0.5];
        let view = [0.5, 0.0, 1.0];
        let hit = cfg().steep_parallax(base, view, &ramp).unwrap();
        assert!(hit.penetrated);
        let raw = hit.hit_uv;
        let refined = hit.refined_uv();
        let fine = ParallaxConfig::new(0.1, 0.05, 4096, 4096);
        let reference = fine.steep_parallax(base, view, &ramp).unwrap().hit_uv;
        let err_raw = (raw[0] - reference[0]).abs();
        let err_refined = (refined[0] - reference[0]).abs();
        assert!(err_refined <= err_raw + 1.0e-6);
    }

    #[test]
    fn secant_refine_degenerate_bracket_returns_hit() {
        let prev = [0.1, 0.2];
        let hit = [0.3, 0.4];
        // before - after within CMP_EPS -> degenerate, returns hit unchanged.
        let out = secant_refine(prev, hit, 1.0e-7, 0.0);
        assert!(approx(out[0], hit[0]));
        assert!(approx(out[1], hit[1]));
    }

    #[test]
    fn secant_refine_midpoint_when_symmetric() {
        let prev = [0.0, 0.0];
        let hit = [1.0, 1.0];
        // before = +1, after = -1 -> weight 0.5 -> midpoint.
        let out = secant_refine(prev, hit, 1.0, -1.0);
        assert!(approx(out[0], 0.5));
        assert!(approx(out[1], 0.5));
    }

    #[test]
    fn grazing_view_increases_layer_count() {
        let c = cfg();
        let head_on = c.layer_count(1.0);
        let grazing = c.layer_count(0.05);
        assert!(grazing > head_on);
    }

    #[test]
    fn head_on_uses_min_layers() {
        let c = cfg();
        assert!(approx(c.layer_count(1.0), f32::from(c.min_layers)));
    }

    #[test]
    fn grazing_uses_max_layers() {
        let c = cfg();
        assert!(approx(c.layer_count(0.0), f32::from(c.max_layers)));
    }

    #[test]
    fn layer_count_never_below_one() {
        let c = ParallaxConfig::new(0.1, 0.05, 0, 0);
        assert!(c.layer_count(1.0) >= 1.0);
        assert!(c.layer_count(0.0) >= 1.0);
    }

    #[test]
    fn full_offset_truncates_at_grazing() {
        // As view.z shrinks the offset magnitude grows (1 / max(z, eps)).
        let near_grazing = full_offset([0.9999, 0.0, 0.0141], 0.1);
        let head_on = full_offset([0.0, 0.0, 1.0], 0.1);
        assert!(near_grazing[0].abs() > head_on[0].abs());
        // Even a zero-z direction stays finite thanks to the eps floor.
        let clamped = full_offset([1.0, 0.0, 0.0], 0.1);
        assert!(clamped[0].is_finite());
    }

    #[test]
    fn self_shadow_fully_lit_without_occluder() {
        // Flat top plane: nothing occludes, fully lit.
        let field = HeightField::flat(4, 4, 1.0).unwrap();
        let v = cfg().self_shadow([0.5, 0.5], 0.0, [0.3, 0.0, 1.0], &|uv| field.sample(uv));
        assert!(approx(v, 1.0));
    }

    #[test]
    fn self_shadow_monotonic_in_occluder_height() {
        // A ridge just off the sample point occludes more as it rises (its
        // height grows, so its surface_depth shrinks and overlaps the ray more).
        let c = ParallaxConfig::new(0.3, 0.2, 8, 32);
        let low_ridge = |uv: [f32; 2]| if uv[0] > 0.55 { 0.6 } else { 0.2 };
        let high_ridge = |uv: [f32; 2]| if uv[0] > 0.55 { 0.95 } else { 0.2 };
        let light = [0.7, 0.0, 0.7];
        let v_low = c.self_shadow([0.5, 0.5], 0.8, light, &low_ridge);
        let v_high = c.self_shadow([0.5, 0.5], 0.8, light, &high_ridge);
        assert!(v_high <= v_low);
    }

    #[test]
    fn self_shadow_stays_in_unit_range() {
        let c = ParallaxConfig::new(0.3, 0.2, 8, 32);
        let ridge = |uv: [f32; 2]| if uv[0] > 0.55 { 0.95 } else { 0.1 };
        let v = c.self_shadow([0.5, 0.5], 0.8, [0.7, 0.0, 0.7], &ridge);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn self_shadow_degenerate_light_is_lit() {
        let field = HeightField::flat(4, 4, 0.2).unwrap();
        let v = cfg().self_shadow([0.5, 0.5], 0.5, [0.0, 0.0, 0.0], &|uv| field.sample(uv));
        assert!(approx(v, 1.0));
    }

    #[test]
    fn std430_layout_size_and_storage_bytes() {
        assert_eq!(PARALLAX_STD430_SIZE, 16);
        assert_eq!(storage_bytes(PARALLAX_STD430_SIZE, 1), PARALLAX_STD430_SIZE);
        assert_eq!(storage_bytes(PARALLAX_STD430_SIZE, 0), PARALLAX_STD430_SIZE);
        assert_eq!(storage_bytes(PARALLAX_STD430_SIZE, 3), 48);
    }

    #[test]
    fn std430_roundtrip_fields() {
        let c = ParallaxConfig::new(0.125, 0.0625, 8, 32);
        let bytes = c.to_std430();
        let hs = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let soft = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let min = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let max = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert!(approx(hs, 0.125));
        assert!(approx(soft, 0.0625));
        assert_eq!(min, 8);
        assert_eq!(max, 32);
    }

    #[test]
    fn heightfield_bilinear_interpolates() {
        // 2x1 grid [0, 1]: the midpoint reads 0.5.
        let field = HeightField::new(2, 1, alloc::vec![0.0, 1.0]).unwrap();
        assert!(approx(field.sample([0.0, 0.0]), 0.0));
        assert!(approx(field.sample([1.0, 0.0]), 1.0));
        assert!(approx(field.sample([0.5, 0.0]), 0.5));
    }

    #[test]
    fn heightfield_rejects_bad_dimensions() {
        assert!(HeightField::new(0, 4, Vec::new()).is_none());
        assert!(HeightField::new(2, 2, alloc::vec![0.0, 1.0]).is_none());
        assert!(HeightField::flat(3, 3, 0.5).is_some());
    }

    #[test]
    fn value_noise_is_deterministic_and_in_range() {
        let a = HeightField::value_noise(8, 8, 42).unwrap();
        let b = HeightField::value_noise(8, 8, 42).unwrap();
        assert_eq!(a, b);
        let c = HeightField::value_noise(8, 8, 7).unwrap();
        assert!(a != c);
        for uv in [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.3, 0.7]] {
            let h = a.sample(uv);
            assert!((0.0..=1.0).contains(&h));
        }
    }

    #[test]
    fn parallax_uv_batch_matches_scalar() {
        let field = HeightField::value_noise(16, 16, 3).unwrap();
        let sampler = |uv: [f32; 2]| field.sample(uv);
        let base = [[0.2, 0.2], [0.6, 0.4], [0.9, 0.1]];
        let view = [0.4, 0.3, 1.0];
        let batch = cfg().parallax_uv_batch(&base, view, &sampler);
        for (i, &uv) in base.iter().enumerate() {
            let scalar = cfg().parallax_uv(uv, view, &sampler);
            assert!(approx(batch[i][0], scalar[0]));
            assert!(approx(batch[i][1], scalar[1]));
        }
    }

    #[test]
    fn hash_is_well_distributed_enough() {
        // Distinct keys should not collapse to a single value.
        let a = hash_u32(1);
        let b = hash_u32(2);
        let c = hash_u32(3);
        assert!(a != b || b != c);
    }
}
