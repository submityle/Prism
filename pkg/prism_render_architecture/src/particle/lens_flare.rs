//! Screen-space lens-flare ghost/`halo` layout: the deterministic `CPU` gold
//! standard for the optical-axis flare a compositor draws over bright points
//! (design §16-§21).
//!
//! A physical lens scatters a bright light source into a chain of faint
//! reflections between its elements. Production `VFX` stacks fake that look in
//! screen space: they extract the brightest highlight, project a string of
//! *ghost* sprites along the *optical axis* (the line through the frame center
//! and the highlight), and wrap a soft *halo* ring around the center. This
//! module owns the `CPU`-verifiable geometry of that layout and packs its
//! parameters into the `std430` uniform block a future `GPU` composite kernel
//! binds.
//!
//! It owns five orthogonal pieces the compositor combines:
//!
//! 1. a highlight *threshold* with a soft `knee`, turning a sampled
//!    `luminance` into a `[0, 1]` bright-point weight;
//! 2. the *ghost* positions, each the reflection of the bright `UV` through the
//!    center scaled by `-spacing * i`, so successive ghosts are evenly spaced
//!    along the axis and sit on the far side of the center;
//! 3. a per-ghost *chromatic* split, a pure `UV` offset along the optical axis
//!    that pushes the red channel one way, the blue channel the other, and
//!    leaves the green channel on the ghost center;
//! 4. a *halo* ring whose weight is a `smoothstep` band centered on a fixed
//!    radius from the frame center; and
//! 5. a radial *attenuation* plus an off-screen *fade* that dims ghosts toward
//!    the frame edges and kills any ghost that lands outside the `[0, 1]`
//!    texture domain.
//!
//! # Strict scope
//!
//! This file is *only* the flare layout. It is **not** [`super::bloom_threshold`]
//! /[`super::bloom_upsample`]: it never runs a Gaussian bright-pass blur or a
//! mip-pyramid up/downsample — it lays out discrete ghost/halo geometry. It is
//! **not** [`super::chromatic_aberration`]: that model splits *every* pixel
//! radially across the whole frame, whereas the split here is a small `UV`
//! nudge applied *only* to each ghost's sample position along the axis.
//!
//! # Determinism
//!
//! The determinism-locked contract layer (design §29) forbids transcendental
//! functions (`sin` / `cos` / `tan` / `atan` / `exp` / `ln` / `powf`) and the
//! rounding primitives (`ceil` / `round`). The only floating-point primitive
//! beyond ordinary arithmetic is `f32::sqrt` (the radial distance and the axis
//! normalization); soft edges are the multiply-only `smoothstep` `t^2 (3 - 2 t)`
//! and the falloff is a rational polynomial, so a future `GPU` kernel
//! reproduces the `CPU` result bit for bit.

use crate::particle::gpu_layout::VEC4_STRIDE;
use alloc::vec::Vec;

/// Number of scalar slots packed into the [`LensFlareParams`] `std430` block:
/// the eleven `f32` fields plus the single `u32` `ghost_count`.
const LENS_FIELD_COUNT: usize = 12;

/// Byte size of the `std430` packing of [`LensFlareParams`]: [`LENS_FIELD_COUNT`]
/// scalar slots rounded up to whole `vec4` slots so the block honors the
/// 16-byte `std430` base alignment. Twelve scalars fill exactly three `vec4`
/// slots (48 bytes) with no padding tail.
pub const LENS_FLARE_STD430_SIZE: usize = LENS_FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

/// Denominators with magnitude at or below this are treated as (near) zero so
/// evaluation falls back to a defined result instead of dividing by zero or
/// propagating `NaN`.
const MIN_DENOM: f32 = 1e-6;

/// Squared axis length at or below which the bright point is treated as sitting
/// on the optical center, so the axis direction degenerates to the zero vector
/// instead of normalizing a (near) zero vector.
const EPS_AXIS_SQ: f32 = 1e-12;

/// `Rec. 709` `luminance` weight of the red channel.
const LUMA_R: f32 = 0.2126;

/// `Rec. 709` `luminance` weight of the green channel.
const LUMA_G: f32 = 0.7152;

/// `Rec. 709` `luminance` weight of the blue channel.
const LUMA_B: f32 = 0.0722;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// Clamps a scalar into the closed unit interval `[0, 1]`.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// The `smoothstep` shaper `t^2 (3 - 2 t)` after clamping `t` into `[0, 1]`: a
/// multiply-only Hermite polynomial with zero first derivative at `0` and `1`,
/// monotonically non-decreasing on `[0, 1]`, and free of any transcendental
/// call.
#[must_use]
fn smoothstep01(t: f32) -> f32 {
    let c = clamp01(t);
    c * c * (3.0 - 2.0 * c)
}

/// Euclidean distance between two `UV` points, the only `sqrt` in the module's
/// hot path besides the axis normalization.
#[must_use]
fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

/// The perceptual `luminance` of a linear `RGB` triple, `dot(rgb, [0.2126,
/// 0.7152, 0.0722])`, written as an explicit `Rec. 709` dot product. This is
/// the brightness the flare thresholds against when deciding whether a sampled
/// pixel is a bright point worth flaring.
#[must_use]
pub fn luminance(rgb: [f32; 3]) -> f32 {
    rgb[0] * LUMA_R + rgb[1] * LUMA_G + rgb[2] * LUMA_B
}

/// One laid-out ghost sample: the three per-channel `UV`s the compositor reads
/// (`[red, green, blue]`) and the scalar weight it multiplies the fetched color
/// by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensFlareGhost {
    /// The per-channel sampled `UV`s `[red, green, blue]` after the chromatic
    /// split along the optical axis.
    pub uv_rgb: [[f32; 2]; 3],
    /// The combined ghost weight: the bright-point weight folded with the
    /// radial attenuation and the off-screen fade, floored at `0`.
    pub weight: f32,
}

/// Screen-space lens-flare layout parameters (design §16-§21).
///
/// `center` is the optical center in `UV` space (typically `[0.5, 0.5]`), the
/// point every ghost reflects through. `threshold` / `knee` shape the highlight
/// extraction. `ghost_spacing` is the per-index axis scale, and `ghost_count`
/// is how many ghosts the chain emits. `chroma_offset` is the per-ghost `UV`
/// split distance along the axis. `halo_radius` / `halo_width` / `halo_intensity`
/// describe the halo ring band. `radial_falloff` dims ghosts with distance from
/// the center, and `edge_fade` is the off-screen fade margin near the `[0, 1]`
/// borders.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensFlareParams {
    /// Optical center in `UV` space that every ghost reflects through.
    pub center: [f32; 2],
    /// `Luminance` above which a sampled pixel counts as a bright point.
    pub threshold: f32,
    /// Half-width of the soft `knee` band around the threshold.
    pub knee: f32,
    /// Per-index axis scale: ghost `i` sits at factor `-ghost_spacing * i`.
    pub ghost_spacing: f32,
    /// Number of ghosts the chain emits.
    pub ghost_count: u32,
    /// Per-ghost `UV` split distance along the optical axis.
    pub chroma_offset: f32,
    /// Radius from the center at which the halo ring band peaks.
    pub halo_radius: f32,
    /// Half-width of the halo ring band.
    pub halo_width: f32,
    /// Peak weight of the halo ring at its band center.
    pub halo_intensity: f32,
    /// Rational-polynomial falloff coefficient dimming ghosts with radius.
    pub radial_falloff: f32,
    /// Off-screen fade margin near the `[0, 1]` texture borders.
    pub edge_fade: f32,
}

impl LensFlareParams {
    /// Builds a parameter set from its raw fields, clamping the shaping
    /// coefficients that must stay non-negative (`knee`, `halo_width`,
    /// `halo_intensity`, `radial_falloff`, `edge_fade`) so every curve stays
    /// well defined.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "The flare layout is genuinely parameterized by these independent optical knobs; grouping them into sub-structs would only obscure the flat std430 block they pack into."
    )]
    pub fn new(
        center: [f32; 2],
        threshold: f32,
        knee: f32,
        ghost_spacing: f32,
        ghost_count: u32,
        chroma_offset: f32,
        halo_radius: f32,
        halo_width: f32,
        halo_intensity: f32,
        radial_falloff: f32,
        edge_fade: f32,
    ) -> Self {
        Self {
            center,
            threshold,
            knee: knee.max(0.0),
            ghost_spacing,
            ghost_count,
            chroma_offset,
            halo_radius,
            halo_width: halo_width.max(0.0),
            halo_intensity: halo_intensity.max(0.0),
            radial_falloff: radial_falloff.max(0.0),
            edge_fade: edge_fade.max(0.0),
        }
    }

    /// The `[0, 1]` bright-point weight of a sampled `luminance`.
    ///
    /// Below `threshold - knee` the weight is exactly `0`; above
    /// `threshold + knee` it saturates at `1`; across the band it is the
    /// multiply-only [`smoothstep01`], so the extraction has a soft `knee`
    /// rather than a hard cutoff. At exactly the threshold the weight is `0.5`.
    /// The function is monotonically non-decreasing in `lum`.
    #[must_use]
    pub fn threshold_weight(&self, lum: f32) -> f32 {
        let lo = self.threshold - self.knee;
        let span = (2.0 * self.knee).max(MIN_DENOM);
        smoothstep01((lum - lo) / span)
    }

    /// The unit-length optical-axis direction from the center toward `bright_uv`.
    ///
    /// When the bright point sits on the center the axis degenerates and this
    /// returns the zero vector instead of normalizing a (near) zero vector.
    #[must_use]
    pub fn axis_dir(&self, bright_uv: [f32; 2]) -> [f32; 2] {
        let dx = bright_uv[0] - self.center[0];
        let dy = bright_uv[1] - self.center[1];
        let len_sq = dx * dx + dy * dy;
        if len_sq <= EPS_AXIS_SQ {
            return [0.0, 0.0];
        }
        let inv = 1.0 / len_sq.sqrt();
        [dx * inv, dy * inv]
    }

    /// The `UV` position of ghost `index`, the reflection of `bright_uv` through
    /// the center scaled by `-ghost_spacing * index`.
    ///
    /// This is the closed form `center + (bright_uv - center) * (-spacing * i)`:
    /// the negative factor places ghost `i >= 1` on the far side of the center
    /// from the bright point, and equal index steps produce equally spaced
    /// ghosts along the axis.
    #[must_use]
    pub fn ghost_uv(&self, bright_uv: [f32; 2], index: u32) -> [f32; 2] {
        #[expect(
            clippy::cast_precision_loss,
            reason = "Ghost indices are small counts; f32 represents them exactly for every realistic ghost chain length."
        )]
        let factor = -self.ghost_spacing * index as f32;
        [
            self.center[0] + (bright_uv[0] - self.center[0]) * factor,
            self.center[1] + (bright_uv[1] - self.center[1]) * factor,
        ]
    }

    /// The three per-channel sampled `UV`s `[red, green, blue]` for ghost
    /// `index`.
    ///
    /// Each channel starts at the ghost center and is nudged along the optical
    /// axis by `chroma_offset`: red is pushed in the `+axis` direction, blue in
    /// the `-axis` direction, and green stays exactly on the ghost center. The
    /// offset is a pure `UV` translation, so red and blue separate in opposite
    /// directions along the same axis line.
    #[must_use]
    pub fn ghost_chroma_uvs(&self, bright_uv: [f32; 2], index: u32) -> [[f32; 2]; 3] {
        let g = self.ghost_uv(bright_uv, index);
        let dir = self.axis_dir(bright_uv);
        let ox = dir[0] * self.chroma_offset;
        let oy = dir[1] * self.chroma_offset;
        [[g[0] + ox, g[1] + oy], [g[0], g[1]], [g[0] - ox, g[1] - oy]]
    }

    /// The halo ring weight at a sampled `UV`.
    ///
    /// The weight is a `smoothstep` band centered on `halo_radius`: it peaks at
    /// `halo_intensity` when the distance from the center equals `halo_radius`
    /// and falls off symmetrically to `0` once the distance leaves the band by
    /// more than `halo_width`. The falloff is monotonically non-increasing in
    /// the absolute distance from the ring.
    #[must_use]
    pub fn halo_weight(&self, uv: [f32; 2]) -> f32 {
        let r = distance(uv, self.center);
        let off_band = (r - self.halo_radius).abs();
        let t = off_band / (self.halo_width + MIN_DENOM);
        self.halo_intensity * (1.0 - smoothstep01(t))
    }

    /// The rational-polynomial radial attenuation `1 / (1 + radial_falloff r^2)`
    /// at radius `r` from the center.
    ///
    /// It is exactly `1` at the center (its maximum) and decreases
    /// monotonically toward the edges, staying strictly positive. With a zero
    /// `radial_falloff` it is the constant `1` (no attenuation).
    #[must_use]
    pub fn radial_attenuation(&self, r: f32) -> f32 {
        1.0 / (1.0 + self.radial_falloff * r * r)
    }

    /// The off-screen fade weight at a sampled `UV`.
    ///
    /// Each component's distance to its nearest `[0, 1]` border is fed through
    /// [`smoothstep01`] scaled by `edge_fade`, and the two component weights are
    /// multiplied. Deep in the interior the weight is `1`; a `UV` that lands on
    /// or outside the unit square has a non-positive border distance and so
    /// fades to exactly `0`.
    #[must_use]
    pub fn screen_fade(&self, uv: [f32; 2]) -> f32 {
        self.edge_component(uv[0]) * self.edge_component(uv[1])
    }

    /// The per-component off-screen fade: `smoothstep` of the distance to the
    /// nearest border over the `edge_fade` margin.
    #[must_use]
    fn edge_component(&self, c: f32) -> f32 {
        let border = c.min(1.0 - c);
        smoothstep01(border / (self.edge_fade + MIN_DENOM))
    }

    /// Lays out a single ghost sample for `index`, folding the chromatic split
    /// `UV`s with the combined weight.
    ///
    /// The weight is the incoming `bright_weight` scaled by the ghost's radial
    /// attenuation and its off-screen fade, floored at `0`. A ghost that lands
    /// outside the frame therefore carries zero weight regardless of the bright
    /// point's strength.
    #[must_use]
    pub fn sample_ghost(
        &self,
        bright_uv: [f32; 2],
        bright_weight: f32,
        index: u32,
    ) -> LensFlareGhost {
        let uv_rgb = self.ghost_chroma_uvs(bright_uv, index);
        let g = self.ghost_uv(bright_uv, index);
        let r = distance(g, self.center);
        let weight = bright_weight * self.radial_attenuation(r) * self.screen_fade(g);
        LensFlareGhost {
            uv_rgb,
            weight: weight.max(0.0),
        }
    }

    /// Lays out the whole ghost chain, one [`LensFlareGhost`] per index in
    /// `1..=ghost_count`, preserving order.
    ///
    /// Indexing starts at `1` so the first ghost is already offset from the
    /// center by one `ghost_spacing` step; a zero index would collapse onto the
    /// center and carry no directional information.
    #[must_use]
    pub fn sample_ghosts(&self, bright_uv: [f32; 2], bright_weight: f32) -> Vec<LensFlareGhost> {
        let mut ghosts = Vec::new();
        let mut index = 1u32;
        while index <= self.ghost_count {
            ghosts.push(self.sample_ghost(bright_uv, bright_weight, index));
            index += 1;
        }
        ghosts
    }

    /// Packs the parameters into their `std430` uniform-block byte layout.
    ///
    /// The eleven `f32` fields fill the first eleven scalar slots and the
    /// `u32` `ghost_count` fills the twelfth, so the block is exactly three
    /// `vec4` slots with no padding tail.
    #[must_use]
    pub fn to_std430(&self) -> [u8; LENS_FLARE_STD430_SIZE] {
        let floats = [
            self.center[0],
            self.center[1],
            self.threshold,
            self.knee,
            self.ghost_spacing,
            self.chroma_offset,
            self.halo_radius,
            self.halo_width,
            self.halo_intensity,
            self.radial_falloff,
            self.edge_fade,
        ];
        let mut bytes = [0u8; LENS_FLARE_STD430_SIZE];
        let mut slots = bytes.chunks_exact_mut(4);
        for value in floats {
            if let Some(slot) = slots.next() {
                slot.copy_from_slice(&value.to_le_bytes());
            }
        }
        if let Some(slot) = slots.next() {
            slot.copy_from_slice(&self.ghost_count.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::gpu_layout::storage_bytes;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx2(a: [f32; 2], b: [f32; 2]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1])
    }

    fn sample_params() -> LensFlareParams {
        LensFlareParams::new(
            [0.5, 0.5], // center
            1.0,        // threshold
            0.25,       // knee
            0.35,       // ghost_spacing
            4,          // ghost_count
            0.01,       // chroma_offset
            0.3,        // halo_radius
            0.08,       // halo_width
            0.7,        // halo_intensity
            2.0,        // radial_falloff
            0.05,       // edge_fade
        )
    }

    #[test]
    fn ghost_layout_is_equidistant_along_axis() {
        let p = sample_params();
        let bright = [0.9, 0.7];
        let g1 = p.ghost_uv(bright, 1);
        let g2 = p.ghost_uv(bright, 2);
        let g3 = p.ghost_uv(bright, 3);
        let step_a = [g2[0] - g1[0], g2[1] - g1[1]];
        let step_b = [g3[0] - g2[0], g3[1] - g2[1]];
        assert!(approx2(step_a, step_b));
    }

    #[test]
    fn ghost_lands_on_opposite_side_of_center() {
        let p = sample_params();
        let bright = [0.9, 0.65];
        let bright_rel = [bright[0] - p.center[0], bright[1] - p.center[1]];
        let g = p.ghost_uv(bright, 1);
        let ghost_rel = [g[0] - p.center[0], g[1] - p.center[1]];
        // Opposite side => the relative vectors point in opposite directions.
        assert!(bright_rel[0] * ghost_rel[0] < 0.0);
        assert!(bright_rel[1] * ghost_rel[1] < 0.0);
    }

    #[test]
    fn ghost_uv_matches_closed_form() {
        let p = sample_params();
        let bright = [0.8, 0.2];
        for index in 0u32..4 {
            #[expect(
                clippy::cast_precision_loss,
                reason = "Test indices are tiny and exactly representable in f32."
            )]
            let factor = -p.ghost_spacing * index as f32;
            let expected = [
                p.center[0] + (bright[0] - p.center[0]) * factor,
                p.center[1] + (bright[1] - p.center[1]) * factor,
            ];
            assert!(approx2(p.ghost_uv(bright, index), expected));
        }
    }

    #[test]
    fn ghost_zero_index_collapses_onto_center() {
        let p = sample_params();
        assert!(approx2(p.ghost_uv([0.9, 0.1], 0), p.center));
    }

    #[test]
    fn chroma_splits_red_and_blue_in_opposite_directions() {
        let p = sample_params();
        let bright = [0.95, 0.5]; // axis points in +x
        let uvs = p.ghost_chroma_uvs(bright, 1);
        let g = p.ghost_uv(bright, 1);
        // Red pushed +axis, blue -axis: their x offsets carry opposite signs.
        let red_dx = uvs[0][0] - g[0];
        let blue_dx = uvs[2][0] - g[0];
        assert!(red_dx * blue_dx < 0.0);
        assert!(red_dx.abs() > 0.0);
    }

    #[test]
    fn chroma_green_stays_on_ghost_center() {
        let p = sample_params();
        let bright = [0.7, 0.9];
        let uvs = p.ghost_chroma_uvs(bright, 2);
        let g = p.ghost_uv(bright, 2);
        assert!(approx2(uvs[1], g));
    }

    #[test]
    fn chroma_offset_is_parallel_to_optical_axis() {
        let p = sample_params();
        let bright = [0.82, 0.74];
        let dir = p.axis_dir(bright);
        let uvs = p.ghost_chroma_uvs(bright, 1);
        // Red - green offset must be parallel to the axis direction (zero cross).
        let ox = uvs[0][0] - uvs[1][0];
        let oy = uvs[0][1] - uvs[1][1];
        let cross = ox * dir[1] - oy * dir[0];
        assert!(approx(cross, 0.0));
    }

    #[test]
    fn axis_dir_is_unit_length_off_center() {
        let p = sample_params();
        let dir = p.axis_dir([0.9, 0.3]);
        let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
        assert!(approx(len, 1.0));
    }

    #[test]
    fn halo_weight_peaks_on_the_ring() {
        let p = sample_params();
        // A UV exactly halo_radius away in +x from the center.
        let on_ring = [p.center[0] + p.halo_radius, p.center[1]];
        assert!(approx(p.halo_weight(on_ring), p.halo_intensity));
    }

    #[test]
    fn halo_weight_smoothsteps_to_zero_off_band() {
        let p = sample_params();
        let on_ring = [p.center[0] + p.halo_radius, p.center[1]];
        let mid_band = [
            p.center[0] + p.halo_radius + 0.5 * p.halo_width,
            p.center[1],
        ];
        let off_band = [p.center[0] + p.halo_radius + p.halo_width, p.center[1]];
        let peak = p.halo_weight(on_ring);
        let mid = p.halo_weight(mid_band);
        let edge = p.halo_weight(off_band);
        assert!(peak > mid);
        assert!(mid > edge);
        assert!(approx(edge, 0.0));
    }

    #[test]
    fn halo_weight_is_symmetric_about_ring() {
        let p = sample_params();
        let inner = [p.center[0] + p.halo_radius - 0.03, p.center[1]];
        let outer = [p.center[0] + p.halo_radius + 0.03, p.center[1]];
        assert!(approx(p.halo_weight(inner), p.halo_weight(outer)));
    }

    #[test]
    fn threshold_weight_is_half_at_threshold() {
        let p = sample_params();
        assert!(approx(p.threshold_weight(p.threshold), 0.5));
    }

    #[test]
    fn threshold_weight_saturates_below_and_above_band() {
        let p = sample_params();
        assert!(approx(p.threshold_weight(p.threshold - p.knee - 0.1), 0.0));
        assert!(approx(p.threshold_weight(p.threshold + p.knee + 0.1), 1.0));
    }

    #[test]
    fn threshold_weight_is_monotonic() {
        let p = sample_params();
        let samples = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5];
        let mut prev = p.threshold_weight(samples[0]);
        for &lum in &samples[1..] {
            let cur = p.threshold_weight(lum);
            assert!(cur >= prev);
            prev = cur;
        }
    }

    #[test]
    fn radial_attenuation_peaks_at_center() {
        let p = sample_params();
        assert!(approx(p.radial_attenuation(0.0), 1.0));
    }

    #[test]
    fn radial_attenuation_is_monotonic_decreasing() {
        let p = sample_params();
        let samples = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0];
        let mut prev = p.radial_attenuation(samples[0]);
        for &r in &samples[1..] {
            let cur = p.radial_attenuation(r);
            assert!(cur < prev);
            assert!(cur > 0.0);
            prev = cur;
        }
    }

    #[test]
    fn screen_fade_is_zero_outside_unit_square() {
        let p = sample_params();
        for uv in [
            [-0.1, 0.5],
            [1.2, 0.5],
            [0.5, -0.05],
            [0.5, 1.4],
            [1.5, 1.5],
        ] {
            assert!(approx(p.screen_fade(uv), 0.0));
        }
    }

    #[test]
    fn screen_fade_is_full_in_interior() {
        let p = sample_params();
        assert!(approx(p.screen_fade([0.5, 0.5]), 1.0));
    }

    #[test]
    fn off_screen_ghost_has_zero_weight() {
        // A bright point near an edge with a large spacing throws ghost 3 well
        // outside the frame; its combined weight must vanish.
        let far = LensFlareParams::new(
            [0.5, 0.5],
            1.0,
            0.25,
            0.9,
            4,
            0.01,
            0.3,
            0.08,
            0.7,
            2.0,
            0.05,
        );
        let ghost = far.sample_ghost([0.98, 0.5], 1.0, 3);
        assert!(approx(ghost.weight, 0.0));
    }

    #[test]
    fn sample_ghosts_count_matches_ghost_count() {
        let p = sample_params();
        let ghosts = p.sample_ghosts([0.9, 0.6], 1.0);
        assert_eq!(ghosts.len(), p.ghost_count as usize);
    }

    #[test]
    fn sample_ghost_weight_folds_attenuation_and_fade() {
        let p = sample_params();
        let bright = [0.85, 0.6];
        let bright_weight = 0.8;
        let ghost = p.sample_ghost(bright, bright_weight, 2);
        let g = p.ghost_uv(bright, 2);
        let r = distance(g, p.center);
        let expected = bright_weight * p.radial_attenuation(r) * p.screen_fade(g);
        assert!(approx(ghost.weight, expected.max(0.0)));
        assert_eq!(ghost.uv_rgb, p.ghost_chroma_uvs(bright, 2));
    }

    #[test]
    fn degenerate_bright_at_center_yields_zero_axis() {
        let p = sample_params();
        let dir = p.axis_dir(p.center);
        assert!(approx2(dir, [0.0, 0.0]));
        // Every ghost collapses onto the center and the chroma split vanishes.
        let uvs = p.ghost_chroma_uvs(p.center, 2);
        assert!(approx2(uvs[0], p.center));
        assert!(approx2(uvs[1], p.center));
        assert!(approx2(uvs[2], p.center));
    }

    #[test]
    fn layout_is_deterministic() {
        let p = sample_params();
        let bright = [0.77, 0.41];
        assert_eq!(p.sample_ghosts(bright, 0.9), p.sample_ghosts(bright, 0.9));
        assert_eq!(p.ghost_chroma_uvs(bright, 1), p.ghost_chroma_uvs(bright, 1));
        assert!(approx(p.halo_weight(bright), p.halo_weight(bright)));
    }

    #[test]
    fn std430_size_is_three_vec4_slots() {
        let p = sample_params();
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), LENS_FLARE_STD430_SIZE);
        assert_eq!(LENS_FLARE_STD430_SIZE, 3 * VEC4_STRIDE);
        assert_eq!(LENS_FLARE_STD430_SIZE % VEC4_STRIDE, 0);
        assert_eq!(
            storage_bytes(LENS_FLARE_STD430_SIZE, 1),
            LENS_FLARE_STD430_SIZE
        );
    }

    #[test]
    fn std430_round_trips_fields() {
        let p = LensFlareParams::new(
            [0.25, 0.6],
            1.5,
            0.2,
            0.4,
            5,
            0.02,
            0.35,
            0.09,
            0.6,
            1.5,
            0.04,
        );
        let bytes = p.to_std430();
        let floats = [
            p.center[0],
            p.center[1],
            p.threshold,
            p.knee,
            p.ghost_spacing,
            p.chroma_offset,
            p.halo_radius,
            p.halo_width,
            p.halo_intensity,
            p.radial_falloff,
            p.edge_fade,
        ];
        let mut slots = bytes.chunks_exact(4);
        for value in floats {
            let slot = slots.next().expect("float slot present");
            let mut word = [0u8; 4];
            word.copy_from_slice(slot);
            assert!(approx(f32::from_le_bytes(word), value));
        }
        let slot = slots.next().expect("count slot present");
        let mut word = [0u8; 4];
        word.copy_from_slice(slot);
        assert_eq!(u32::from_le_bytes(word), p.ghost_count);
    }
}
