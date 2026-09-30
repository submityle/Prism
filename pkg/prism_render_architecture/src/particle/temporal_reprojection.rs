//! Temporal reprojection and history sampling for `TAA` / temporal
//! accumulation (design §21), as a pure-`CPU` verifiable reference.
//!
//! This module owns the *history-resolve* half of a temporal anti-aliasing
//! (`TAA`) / temporal-accumulation pipeline, aligned at the algorithm level
//! with Unreal's `TAA` / `TSR` and `Frostbite`'s temporal filter, without
//! reusing any of their code. Given a *screen-space motion vector supplied by
//! an upstream pass*, it reprojects the previous frame's resolved colour into
//! the current pixel, decides whether that history is trustworthy, constrains
//! it against the current frame's local neighbourhood to fight ghosting, and
//! blends the constrained history against the current sample by confidence.
//!
//! The temporal-resolve chain implemented here is:
//!
//! 1. **Reproject** — [`reproject_uv`] maps the current pixel's `UV` back to the
//!    `UV` the same surface occupied last frame by subtracting the *given*
//!    screen-space motion vector. This is the standard `TAA` convention where
//!    the motion vector encodes `current - previous` screen position, so the
//!    history sample lives at `current_uv - motion`.
//! 2. **Validate** — [`history_valid`] rejects history that reprojects
//!    off-screen and history whose depth disagrees with the current surface
//!    (disocclusion), and softens confidence for fast motion. It returns a
//!    confidence in `[0, 1]`, never a hard boolean, so the blend can fade
//!    rather than pop.
//! 3. **Constrain** — [`neighborhood_clamp`] clamps the reprojected history to
//!    the min/max `AABB` box of the current frame's `3x3` neighbourhood, and
//!    [`clip_history_ycocg`] performs the sharper `YCoCg`-space `AABB` line clip
//!    toward the current sample. Both pull stale, out-of-gamut history back into
//!    a plausible range so trailing "ghosts" cannot survive.
//! 4. **Blend** — [`blend_history`] mixes the current sample with the
//!    constrained history by [`history_weight`]. At full confidence the result
//!    approaches the history (the steady-state accumulation that gives `TAA` its
//!    stability); at zero confidence it collapses to the current sample.
//!
//! # Deliberate scope boundary: this module does *not* generate motion vectors
//!
//! Producing the screen-space motion vector — projecting current and previous
//! world positions through the current and previous view-projection matrices,
//! doing the perspective divide, and taking the `NDC`-to-`UV` difference — is
//! the job of the sibling [`super::motion_vectors`] module. This module
//! **consumes** that vector as an input and never touches a `Mat4`, a clip-space
//! position, or the perspective divide. It also does not do ordered temporal
//! dithering ([`super::temporal_dither`]) or motion blur
//! ([`super::motion_blur`]); it only resolves history against the current frame.
//!
//! Everything is built from ordinary arithmetic plus `f32::sqrt` and
//! `f32::floor` only — no transcendental functions — so a future `GPU` compute
//! kernel reproduces the same resolved colour bit for bit. Soft falloffs use the
//! cubic `smoothstep` polynomial rather than any exponential curve, and any
//! self-contained randomness comes from a local integer bit-mixer so this file
//! depends on no other particle sibling except [`super::gpu_layout`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, U32_STRIDE, VEC4_STRIDE};

/// Floating-point comparison tolerance.
///
/// `f32` values are compared against this epsilon instead of `==` / `!=` so a
/// vanishing denominator or a "practically equal" pair is handled without an
/// exact bit compare.
pub const CMP_EPS: f32 = 1e-6;

/// Smallest vector length treated as non-zero when normalizing a direction.
///
/// A shorter vector has no well-defined direction, so it normalizes to the zero
/// vector rather than dividing by a vanishing length.
pub const EPS_LEN: f32 = 1e-12;

/// Floor on the depth denominator when forming a *relative* depth difference.
///
/// A surface at or behind the camera plane can report a near-zero depth; using
/// it directly as a denominator would explode the relative difference, so the
/// denominator is clamped up to this bound first.
pub const DEPTH_EPS: f32 = 1e-6;

/// A minimal two-component screen-space (`UV`) vector.
///
/// Temporal reprojection lives in the two-dimensional screen plane, so this
/// module uses its own small `2D` type rather than the three-dimensional shared
/// vector. It uses only add / subtract / multiply / divide plus `f32::sqrt`,
/// never a transcendental function, and derives only [`PartialEq`] (no `Eq` /
/// `Hash`) because it holds `f32` fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// Horizontal (`U`) component in `UV` space.
    pub x: f32,
    /// Vertical (`V`) component in `UV` space.
    pub y: f32,
}

impl Vec2 {
    /// The zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Constructs a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Component-wise subtraction (named `minus` to avoid shadowing the
    /// `std::ops::Sub` trait method).
    #[must_use]
    pub fn minus(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }

    /// Uniformly scales both components.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }

    /// Squared Euclidean length (no `sqrt`).
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Euclidean length in `UV` units.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the unit-length direction, or [`Vec2::ZERO`] when the vector is
    /// shorter than [`EPS_LEN`] and has no well-defined direction.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len <= EPS_LEN {
            Self::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }
}

/// A linear `RGBA` colour sample resolved by the temporal filter.
///
/// The four channels are premultiplied-agnostic here; the neighbourhood
/// constraint and the history blend operate per channel.
pub type Rgba = [f32; 4];

/// A per-channel axis-aligned bounding box (`AABB`) over `RGBA` colour.
///
/// `min[c]` and `max[c]` bound channel `c` across a set of samples; a history
/// colour is trustworthy when it lies inside this box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbRgba {
    /// Per-channel lower bound.
    pub min: Rgba,
    /// Per-channel upper bound.
    pub max: Rgba,
}

impl AabbRgba {
    /// Widens the box symmetrically about its per-channel centre.
    ///
    /// `extra` is a non-negative fractional expansion: `0.0` leaves the box
    /// unchanged, `1.0` doubles each channel's width. Widening the constraint
    /// box trades a little ghosting resistance for less flicker on high-contrast
    /// edges, mirroring the tunable neighbourhood scale in production `TAA`.
    #[must_use]
    pub fn widened(self, extra: f32) -> Self {
        let factor = 1.0 + extra.max(0.0);
        let mut min = self.min;
        let mut max = self.max;
        for ((lo, hi), (out_lo, out_hi)) in self
            .min
            .iter()
            .zip(self.max.iter())
            .zip(min.iter_mut().zip(max.iter_mut()))
        {
            let centre = (lo + hi) * 0.5;
            let half = (hi - lo) * 0.5 * factor;
            *out_lo = centre - half;
            *out_hi = centre + half;
        }
        Self { min, max }
    }
}

/// `std430`-friendly temporal-reprojection parameters (design §21).
///
/// The block is four 4-byte `f32` scalars, so it packs into a single
/// `vec4`-sized `std430` slot with no interior padding. All fields are
/// validity-clamped by [`ReprojectionParams::new`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReprojectionParams {
    /// Steady-state history weight in `[0, 1]`: the fraction of the previous
    /// frame retained when confidence is full. Values near `0.9` give the slow
    /// convergence that makes `TAA` stable.
    pub max_history_weight: f32,
    /// Relative depth difference at or above which the history is rejected as a
    /// disocclusion. Compared against `|current - history| / max(|current|,
    /// DEPTH_EPS)`.
    pub depth_reject_relative: f32,
    /// Screen-space motion magnitude (`UV` units) at which the velocity
    /// confidence has fully faded to zero. Non-positive disables the velocity
    /// penalty.
    pub velocity_reject_uv: f32,
    /// Fractional widening applied to the neighbourhood constraint box before
    /// clamping / clipping history. Zero clamps to the tight `3x3` box.
    pub clamp_widen: f32,
}

impl ReprojectionParams {
    /// Builds a validity-clamped parameter block.
    ///
    /// `max_history_weight` is clamped to `[0, 1]`, and the remaining fields are
    /// floored at zero so no downstream arithmetic can be driven by a negative
    /// threshold.
    #[must_use]
    pub fn new(
        max_history_weight: f32,
        depth_reject_relative: f32,
        velocity_reject_uv: f32,
        clamp_widen: f32,
    ) -> Self {
        Self {
            max_history_weight: max_history_weight.clamp(0.0, 1.0),
            depth_reject_relative: depth_reject_relative.max(0.0),
            velocity_reject_uv: velocity_reject_uv.max(0.0),
            clamp_widen: clamp_widen.max(0.0),
        }
    }

    /// Byte size of the `std430` parameter block: one `vec4`-sized slot.
    #[must_use]
    pub fn std430_size() -> usize {
        VEC4_STRIDE
    }

    /// Serializes the block to its `std430` byte image (little-endian).
    ///
    /// The four `f32` fields occupy four consecutive 4-byte scalar slots, in
    /// declaration order, filling exactly one `vec4` slot with no padding.
    #[must_use]
    pub fn to_std430(self) -> [u8; VEC4_STRIDE] {
        let mut bytes = [0u8; VEC4_STRIDE];
        bytes[0..U32_STRIDE].copy_from_slice(&self.max_history_weight.to_le_bytes());
        bytes[U32_STRIDE..2 * U32_STRIDE]
            .copy_from_slice(&self.depth_reject_relative.to_le_bytes());
        bytes[2 * U32_STRIDE..3 * U32_STRIDE]
            .copy_from_slice(&self.velocity_reject_uv.to_le_bytes());
        bytes[3 * U32_STRIDE..4 * U32_STRIDE].copy_from_slice(&self.clamp_widen.to_le_bytes());
        bytes
    }
}

/// Packs a parameter array into a contiguous `std430` byte blob.
///
/// The capacity is reserved through [`storage_bytes`], which clamps an empty
/// array up to a single element so a `WebGPU` storage binding is never
/// zero-sized; the returned bytes still mirror the array exactly (empty in,
/// empty out).
#[must_use]
pub fn pack_params_std430(params: &[ReprojectionParams]) -> Vec<u8> {
    let total = storage_bytes(ReprojectionParams::std430_size(), params.len());
    let mut bytes = Vec::with_capacity(total);
    for p in params {
        bytes.extend_from_slice(&p.to_std430());
    }
    bytes
}

/// Reprojects the current pixel's `UV` to where the same surface sat last frame.
///
/// `motion_uv` is the *given* screen-space motion vector (the sibling
/// [`super::motion_vectors`] pass produced it as `current - previous` in `UV`
/// space); this module only consumes it. The history sample lives at
/// `current_uv - motion_uv`. A zero motion vector returns `current_uv`
/// unchanged, which is the correct degenerate behaviour for a static pixel.
#[must_use]
pub fn reproject_uv(current_uv: Vec2, motion_uv: Vec2) -> Vec2 {
    current_uv.minus(motion_uv)
}

/// Whether a reprojected `UV` lands inside the `[0, 1]` screen rectangle.
///
/// History that reprojects outside the frame has no source pixel and must be
/// rejected; this is the screen-boundary half of the validity test.
#[must_use]
pub fn is_on_screen(uv: Vec2) -> bool {
    (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y)
}

/// Depth-based history confidence in `[0, 1]` (disocclusion rejection).
///
/// The *relative* depth difference `|current - history| / max(|current|,
/// DEPTH_EPS)` is compared against `reject_relative`. At or above the threshold
/// the surfaces differ (a disocclusion) and confidence is zero; below it,
/// confidence falls off with a cubic `smoothstep` so a near-match stays close to
/// one and a borderline match fades smoothly rather than popping.
#[must_use]
pub fn depth_confidence(current_depth: f32, history_depth: f32, reject_relative: f32) -> f32 {
    let denom = current_depth.abs().max(DEPTH_EPS);
    let rel = (current_depth - history_depth).abs() / denom;
    let thr = reject_relative.max(CMP_EPS);
    if rel >= thr {
        0.0
    } else {
        1.0 - smoothstep(rel / thr)
    }
}

/// Velocity-based history confidence in `[0, 1]`.
///
/// Fast screen motion makes the single-tap reprojection increasingly
/// inaccurate, so confidence fades as the motion magnitude approaches
/// `reject_uv`. A non-positive `reject_uv` disables the penalty and returns full
/// confidence.
#[must_use]
pub fn velocity_confidence(motion_uv: Vec2, reject_uv: f32) -> f32 {
    let limit = reject_uv.max(0.0);
    if limit <= CMP_EPS {
        return 1.0;
    }
    let speed = motion_uv.length();
    (1.0 - smoothstep(speed / limit)).clamp(0.0, 1.0)
}

/// Combined history confidence in `[0, 1]` for a reprojected sample.
///
/// Rejects history that reprojects off-screen (confidence zero), then multiplies
/// the depth-disocclusion confidence by the velocity confidence. The product is
/// clamped to `[0, 1]`; it is a soft weight, never a hard boolean, so the blend
/// can fade history in and out without visible pops.
#[must_use]
pub fn history_valid(
    history_uv: Vec2,
    current_depth: f32,
    history_depth: f32,
    motion_uv: Vec2,
    params: ReprojectionParams,
) -> f32 {
    if !is_on_screen(history_uv) {
        return 0.0;
    }
    let depth = depth_confidence(current_depth, history_depth, params.depth_reject_relative);
    let velocity = velocity_confidence(motion_uv, params.velocity_reject_uv);
    (depth * velocity).clamp(0.0, 1.0)
}

/// Builds the per-channel min/max `AABB` of a `3x3` colour neighbourhood.
///
/// `samples` is the current frame's `3x3` window around the pixel (centre at
/// index four, row-major); the box bounds every channel across all nine taps and
/// is the plausible-colour range history is expected to fall inside.
#[must_use]
pub fn neighborhood_box(samples: &[Rgba; 9]) -> AabbRgba {
    let mut min = samples[0];
    let mut max = samples[0];
    for sample in &samples[1..] {
        for ((lo, hi), &value) in min.iter_mut().zip(max.iter_mut()).zip(sample.iter()) {
            if value < *lo {
                *lo = value;
            }
            if value > *hi {
                *hi = value;
            }
        }
    }
    AabbRgba { min, max }
}

/// Clamps a reprojected history colour into the current `3x3` neighbourhood box.
///
/// This is the min/max box constraint that keeps stale history from producing
/// trailing ghosts: any channel that drifted outside the plausible range of the
/// current neighbourhood is pulled back to the nearest box face. The box is
/// widened by [`ReprojectionParams::clamp_widen`] first so the constraint can be
/// loosened when flicker matters more than ghosting.
#[must_use]
pub fn neighborhood_clamp(history: Rgba, samples: &[Rgba; 9], params: ReprojectionParams) -> Rgba {
    let bounds = neighborhood_box(samples).widened(params.clamp_widen);
    let mut out = [0.0f32; 4];
    for (((slot, &value), &lo), &hi) in out
        .iter_mut()
        .zip(history.iter())
        .zip(bounds.min.iter())
        .zip(bounds.max.iter())
    {
        *slot = value.clamp(lo, hi);
    }
    out
}

/// Converts a linear `RGBA` colour to `YCoCg` (alpha ignored), returning
/// `[Y, Co, Cg]`.
///
/// `TAA` neighbourhood clipping is done in `YCoCg` because the luma / chroma
/// split makes the constraint box hug the perceptually relevant axis, which
/// suppresses chroma ghosting better than a raw `RGB` box. The transform is a
/// fixed linear combination — only adds, subtracts, and multiplies by constants
/// — so it is exactly invertible by [`ycocg_to_rgb`].
#[must_use]
pub fn rgb_to_ycocg(color: Rgba) -> [f32; 3] {
    let r = color[0];
    let g = color[1];
    let b = color[2];
    let y = r * 0.25 + g * 0.5 + b * 0.25;
    let co = r * 0.5 - b * 0.5;
    let cg = -r * 0.25 + g * 0.5 - b * 0.25;
    [y, co, cg]
}

/// Inverse of [`rgb_to_ycocg`]: reconstructs linear `RGB` from `[Y, Co, Cg]`.
///
/// The `alpha` argument is carried through unchanged, since the `YCoCg`
/// transform never touched it.
#[must_use]
pub fn ycocg_to_rgb(ycocg: [f32; 3], alpha: f32) -> Rgba {
    let y = ycocg[0];
    let co = ycocg[1];
    let cg = ycocg[2];
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    [r, g, b, alpha]
}

/// Clips the reprojected history toward the current sample until it lands inside
/// the current neighbourhood's `YCoCg` `AABB`.
///
/// This is the sharper anti-ghosting constraint used by production `TAA`: rather
/// than clamping each channel independently (which can shift hue), it moves the
/// history colour along the straight segment toward the current sample and stops
/// at the first `AABB` face it would cross. History already inside the box is
/// returned unchanged. The box is built in `YCoCg` from the `3x3` neighbourhood
/// and widened by [`ReprojectionParams::clamp_widen`]; the history's alpha is
/// preserved.
#[must_use]
pub fn clip_history_ycocg(
    current: Rgba,
    history: Rgba,
    samples: &[Rgba; 9],
    params: ReprojectionParams,
) -> Rgba {
    let q = rgb_to_ycocg(current);
    let p = rgb_to_ycocg(history);

    let mut lo = rgb_to_ycocg(samples[0]);
    let mut hi = lo;
    for sample in &samples[1..] {
        let value = rgb_to_ycocg(*sample);
        for ((box_lo, box_hi), &channel) in lo.iter_mut().zip(hi.iter_mut()).zip(value.iter()) {
            if channel < *box_lo {
                *box_lo = channel;
            }
            if channel > *box_hi {
                *box_hi = channel;
            }
        }
    }

    let factor = 1.0 + params.clamp_widen.max(0.0);
    for (box_lo, box_hi) in lo.iter_mut().zip(hi.iter_mut()) {
        let centre = (*box_lo + *box_hi) * 0.5;
        let half = (*box_hi - *box_lo) * 0.5 * factor;
        *box_lo = centre - half;
        *box_hi = centre + half;
    }

    let clipped = clip_toward(lo, hi, q, p);
    ycocg_to_rgb(clipped, history[3])
}

/// Moves point `p` toward point `q` (assumed inside the box) until it lies on or
/// inside the `[box_min, box_max]` `AABB`, returning the clipped point.
///
/// For each axis the crossing fraction is the distance from `q` to the relevant
/// box face divided by the signed extent of `p - q`; the tightest crossing over
/// all axes bounds the retained fraction, clamped to `[0, 1]`. An axis whose
/// component barely moves (within [`CMP_EPS`]) imposes no constraint, so a
/// vanishing extent never divides by zero.
fn clip_toward(box_min: [f32; 3], box_max: [f32; 3], q: [f32; 3], p: [f32; 3]) -> [f32; 3] {
    let mut t = 1.0f32;
    for (((&lo, &hi), &qa), &pa) in box_min
        .iter()
        .zip(box_max.iter())
        .zip(q.iter())
        .zip(p.iter())
    {
        let delta = pa - qa;
        if delta > CMP_EPS {
            t = t.min((hi - qa) / delta);
        } else if delta < -CMP_EPS {
            t = t.min((lo - qa) / delta);
        }
    }
    let t = t.clamp(0.0, 1.0);
    [
        q[0] + t * (p[0] - q[0]),
        q[1] + t * (p[1] - q[1]),
        q[2] + t * (p[2] - q[2]),
    ]
}

/// History blend weight in `[0, 1]` for a given confidence.
///
/// The weight is `confidence * max_history_weight`, clamped to `[0, 1]`. At full
/// confidence it approaches `max_history_weight` (the steady-state accumulation);
/// at zero confidence it is zero, so the blend keeps only the current sample.
#[must_use]
pub fn history_weight(confidence: f32, params: ReprojectionParams) -> f32 {
    (confidence.clamp(0.0, 1.0) * params.max_history_weight).clamp(0.0, 1.0)
}

/// Blends the current sample with the constrained history by confidence.
///
/// `clamped_history` should already have been constrained by
/// [`neighborhood_clamp`] or [`clip_history_ycocg`]. The result is the per-channel
/// linear interpolation `current * (1 - w) + history * w`, where `w` is
/// [`history_weight`]. This is the frame-to-frame exponential accumulation that
/// gives temporal filtering its stability: high confidence trends toward the
/// history, zero confidence returns the current sample untouched.
#[must_use]
pub fn blend_history(
    current: Rgba,
    clamped_history: Rgba,
    confidence: f32,
    params: ReprojectionParams,
) -> Rgba {
    let w = history_weight(confidence, params);
    let mut out = [0.0f32; 4];
    for ((slot, &c), &h) in out
        .iter_mut()
        .zip(current.iter())
        .zip(clamped_history.iter())
    {
        *slot = c * (1.0 - w) + h * w;
    }
    out
}

/// Cubic `smoothstep` `t*t*(3 - 2t)` clamped to `[0, 1]`, for soft confidence
/// falloff.
///
/// Used to fade confidence in and out without a transcendental curve; inputs
/// outside `[0, 1]` saturate to the endpoints.
#[must_use]
pub fn smoothstep(t: f32) -> f32 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn rgba_close(a: Rgba, b: Rgba) -> bool {
        a.iter().zip(b.iter()).all(|(&x, &y)| close(x, y))
    }

    fn flat_neighborhood(color: Rgba) -> [Rgba; 9] {
        [color; 9]
    }

    const PARAMS: ReprojectionParams = ReprojectionParams {
        max_history_weight: 0.9,
        depth_reject_relative: 0.1,
        velocity_reject_uv: 0.2,
        clamp_widen: 0.0,
    };

    #[test]
    fn reproject_subtracts_motion_vector() {
        let uv = Vec2::new(0.5, 0.5);
        let motion = Vec2::new(0.1, -0.2);
        let history = reproject_uv(uv, motion);
        assert!(close(history.x, 0.4));
        assert!(close(history.y, 0.7));
    }

    #[test]
    fn reproject_zero_motion_is_identity() {
        let uv = Vec2::new(0.3, 0.8);
        let history = reproject_uv(uv, Vec2::ZERO);
        assert_eq!(history, uv);
    }

    #[test]
    fn on_screen_detects_bounds() {
        assert!(is_on_screen(Vec2::new(0.0, 1.0)));
        assert!(is_on_screen(Vec2::new(0.5, 0.5)));
        assert!(!is_on_screen(Vec2::new(-0.01, 0.5)));
        assert!(!is_on_screen(Vec2::new(0.5, 1.01)));
    }

    #[test]
    fn off_screen_history_is_rejected() {
        let conf = history_valid(Vec2::new(1.2, 0.5), 10.0, 10.0, Vec2::ZERO, PARAMS);
        assert!(close(conf, 0.0));
    }

    #[test]
    fn disocclusion_depth_gap_is_rejected() {
        // History depth far from current depth: relative gap exceeds threshold.
        let conf = history_valid(Vec2::new(0.5, 0.5), 10.0, 5.0, Vec2::ZERO, PARAMS);
        assert!(close(conf, 0.0));
    }

    #[test]
    fn matching_depth_is_fully_confident() {
        let conf = history_valid(Vec2::new(0.5, 0.5), 10.0, 10.0, Vec2::ZERO, PARAMS);
        assert!(close(conf, 1.0));
    }

    #[test]
    fn depth_confidence_fades_smoothly() {
        // A relative gap at half the threshold sits strictly between 0 and 1.
        let conf = depth_confidence(10.0, 9.5, 0.1);
        assert!(conf > 0.0 && conf < 1.0);
    }

    #[test]
    fn velocity_penalty_reduces_confidence() {
        let slow = velocity_confidence(Vec2::new(0.01, 0.0), 0.2);
        let fast = velocity_confidence(Vec2::new(0.15, 0.0), 0.2);
        assert!(slow > fast);
        assert!((0.0..=1.0).contains(&slow));
        assert!((0.0..=1.0).contains(&fast));
    }

    #[test]
    fn velocity_penalty_disabled_when_limit_nonpositive() {
        let conf = velocity_confidence(Vec2::new(5.0, 5.0), 0.0);
        assert!(close(conf, 1.0));
    }

    #[test]
    fn degenerate_huge_motion_reprojects_off_screen() {
        let uv = Vec2::new(0.5, 0.5);
        let history_uv = reproject_uv(uv, Vec2::new(3.0, 0.0));
        assert!(!is_on_screen(history_uv));
        let conf = history_valid(history_uv, 10.0, 10.0, Vec2::new(3.0, 0.0), PARAMS);
        assert!(close(conf, 0.0));
    }

    #[test]
    fn neighborhood_box_bounds_all_samples() {
        let mut samples = flat_neighborhood([0.5, 0.5, 0.5, 1.0]);
        samples[0] = [0.1, 0.2, 0.3, 1.0];
        samples[8] = [0.9, 0.8, 0.7, 1.0];
        let bounds = neighborhood_box(&samples);
        assert!(close(bounds.min[0], 0.1));
        assert!(close(bounds.max[0], 0.9));
        assert!(close(bounds.min[1], 0.2));
        assert!(close(bounds.max[1], 0.8));
    }

    #[test]
    fn neighborhood_clamp_pulls_out_of_range_history_back() {
        let samples = flat_neighborhood([0.4, 0.4, 0.4, 1.0]);
        // Stale ghost far brighter than the current neighbourhood.
        let ghost: Rgba = [5.0, 5.0, 5.0, 1.0];
        let clamped = neighborhood_clamp(ghost, &samples, PARAMS);
        assert!(rgba_close(clamped, [0.4, 0.4, 0.4, 1.0]));
    }

    #[test]
    fn neighborhood_clamp_leaves_in_range_history_untouched() {
        let mut samples = flat_neighborhood([0.5, 0.5, 0.5, 1.0]);
        samples[0] = [0.2, 0.2, 0.2, 1.0];
        samples[8] = [0.8, 0.8, 0.8, 1.0];
        let history: Rgba = [0.5, 0.5, 0.5, 1.0];
        let clamped = neighborhood_clamp(history, &samples, PARAMS);
        assert!(rgba_close(clamped, history));
    }

    #[test]
    fn widened_box_admits_more_history() {
        let mut samples = flat_neighborhood([0.5, 0.5, 0.5, 1.0]);
        samples[0] = [0.4, 0.4, 0.4, 1.0];
        samples[8] = [0.6, 0.6, 0.6, 1.0];
        let widen = ReprojectionParams::new(0.9, 0.1, 0.2, 2.0);
        let history: Rgba = [0.75, 0.75, 0.75, 1.0];
        let tight = neighborhood_clamp(history, &samples, PARAMS);
        let loose = neighborhood_clamp(history, &samples, widen);
        // Widened box clamps less aggressively, so it keeps a larger value.
        assert!(loose[0] > tight[0]);
    }

    #[test]
    fn ycocg_round_trips() {
        let color: Rgba = [0.2, 0.7, 0.4, 0.8];
        let back = ycocg_to_rgb(rgb_to_ycocg(color), color[3]);
        assert!(rgba_close(back, color));
    }

    #[test]
    fn ycocg_clip_pulls_history_into_box() {
        let mut samples = flat_neighborhood([0.5, 0.5, 0.5, 1.0]);
        samples[0] = [0.3, 0.3, 0.3, 1.0];
        samples[8] = [0.7, 0.7, 0.7, 1.0];
        let current: Rgba = [0.5, 0.5, 0.5, 1.0];
        let ghost: Rgba = [2.0, 0.0, 0.0, 1.0];
        let clipped = clip_history_ycocg(current, ghost, &samples, PARAMS);
        // The clipped colour must fall inside the neighbourhood's YCoCg box.
        let bounds = neighborhood_box(&samples);
        let box_min = rgb_to_ycocg(bounds.min);
        let box_max = rgb_to_ycocg(bounds.max);
        let yc = rgb_to_ycocg(clipped);
        // Luma is the stable axis: clipped luma stays within the box (with eps).
        assert!(yc[0] >= box_min[0] - CMP_EPS && yc[0] <= box_max[0] + CMP_EPS);
    }

    #[test]
    fn ycocg_clip_leaves_inside_history_untouched() {
        let mut samples = flat_neighborhood([0.5, 0.5, 0.5, 1.0]);
        samples[0] = [0.3, 0.3, 0.3, 1.0];
        samples[8] = [0.7, 0.7, 0.7, 1.0];
        let current: Rgba = [0.5, 0.5, 0.5, 1.0];
        let history: Rgba = [0.55, 0.55, 0.55, 1.0];
        let clipped = clip_history_ycocg(current, history, &samples, PARAMS);
        assert!(rgba_close(clipped, history));
    }

    #[test]
    fn blend_full_confidence_trends_to_history() {
        let current: Rgba = [1.0, 0.0, 0.0, 1.0];
        let history: Rgba = [0.0, 0.0, 1.0, 1.0];
        let blended = blend_history(current, history, 1.0, PARAMS);
        // With max_history_weight 0.9 the result is dominated by history.
        assert!(blended[2] > blended[0]);
        assert!(close(blended[0], 0.1));
        assert!(close(blended[2], 0.9));
    }

    #[test]
    fn blend_zero_confidence_returns_current() {
        let current: Rgba = [0.3, 0.6, 0.9, 1.0];
        let history: Rgba = [0.0, 0.0, 0.0, 0.0];
        let blended = blend_history(current, history, 0.0, PARAMS);
        assert!(rgba_close(blended, current));
    }

    #[test]
    fn history_weight_stays_in_unit_range() {
        for &c in &[-1.0f32, 0.0, 0.5, 1.0, 2.0] {
            let w = history_weight(c, PARAMS);
            assert!((0.0..=1.0).contains(&w));
        }
    }

    #[test]
    fn params_new_clamps_fields() {
        let p = ReprojectionParams::new(2.0, -0.5, -1.0, -3.0);
        assert!(close(p.max_history_weight, 1.0));
        assert!(close(p.depth_reject_relative, 0.0));
        assert!(close(p.velocity_reject_uv, 0.0));
        assert!(close(p.clamp_widen, 0.0));
    }

    #[test]
    fn std430_layout_size_and_round_trip() {
        assert_eq!(ReprojectionParams::std430_size(), VEC4_STRIDE);
        let p = ReprojectionParams::new(0.85, 0.12, 0.25, 0.5);
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), 16);
        let f0 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let f1 = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let f2 = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let f3 = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert!(close(f0, 0.85));
        assert!(close(f1, 0.12));
        assert!(close(f2, 0.25));
        assert!(close(f3, 0.5));
    }

    #[test]
    fn pack_params_matches_element_size() {
        let arr = [
            ReprojectionParams::new(0.9, 0.1, 0.2, 0.0),
            ReprojectionParams::new(0.8, 0.2, 0.3, 1.0),
        ];
        let packed = pack_params_std430(&arr);
        assert_eq!(packed.len(), 2 * VEC4_STRIDE);
        // Empty input still yields empty bytes (capacity clamps, length does not).
        assert!(pack_params_std430(&[]).is_empty());
    }

    #[test]
    fn smoothstep_endpoints_and_midpoint() {
        assert!(close(smoothstep(0.0), 0.0));
        assert!(close(smoothstep(1.0), 1.0));
        assert!(close(smoothstep(0.5), 0.5));
        assert!(close(smoothstep(-4.0), 0.0));
        assert!(close(smoothstep(4.0), 1.0));
    }
}
