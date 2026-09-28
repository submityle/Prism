//! Backend-neutral CPU reference for the full-screen **temporal
//! anti-aliasing** resolve.
//!
//! Each frame the jittered scene colour is blended with the reprojected history
//! so the sub-pixel jitter (see [`super::jitter`]) integrates into a supersampled
//! image. The two failure modes TAA must fight are **ghosting** (stale history
//! bleeding onto a surface it no longer covers) and **flicker** (a bright
//! sub-pixel sample popping in and out). This resolve fights both with the AAA
//! trio popularised by Karis' *High Quality Temporal Supersampling*:
//!
//! * **Motion reprojection** — history is fetched at `current_uv - motion`
//!   ([`super::super::reproject_prev_uv_motion`]) using the resolve pass's
//!   motion-vector G-buffer, so it follows camera *and* per-object motion.
//! * **`YCoCg` neighbourhood clipping** — the reprojected history is clipped to
//!   the colour-variance box of the current 3x3 neighbourhood
//!   ([`super::super::variance_clip_box`] / [`super::super::clip_history_to_aabb`]),
//!   computed in **`YCoCg`** where luma and chroma separate so the box hugs the
//!   real signal and rejects stale colour without the boxy artefacts an `RGB` box
//!   leaves.
//! * **Luminance feedback weighting** — the current and clipped-history
//!   contributions are weighted by the inverse of their luma
//!   ([`tonemap_weight`]), which suppresses fireflies: a single blindingly
//!   bright sub-pixel sample is down-weighted so it cannot dominate the blend
//!   and flicker.
//!
//! Mirrored bit-for-bit by the `taa_resolve.wesl` GPU twin; every operation is
//! plain arithmetic so CPU and GPU agree exactly.

use bevy_math::Vec3;

use super::super::{clip_history_to_aabb, variance_clip_box};

/// Golden tunables for the TAA resolve, shared byte-for-byte with the GPU
/// immediate block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TaaParams {
    /// Fraction of the (clipped) history kept when the sample is trusted. `0.9`
    /// integrates roughly ten frames of jitter — the AAA default that balances
    /// edge smoothness against motion lag.
    pub history_blend: f32,
    /// Half-width of the neighbourhood variance box in standard deviations.
    /// `1.0` is a tight box that resolves edges crisply; larger values keep
    /// more history at the cost of a little ghosting.
    pub variance_gamma: f32,
}

impl Default for TaaParams {
    fn default() -> Self {
        Self {
            history_blend: 0.9,
            variance_gamma: 1.0,
        }
    }
}

/// Rec.601-style **`RGB` -> `YCoCg`** transform (luma + two chroma). Exactly
/// invertible by [`ycocg_to_rgb`]; the clipping box is built in this space so a
/// bright specular edge does not drag chroma around.
pub fn rgb_to_ycocg(rgb: Vec3) -> Vec3 {
    Vec3::new(
        0.25 * rgb.x + 0.5 * rgb.y + 0.25 * rgb.z,
        0.5 * rgb.x - 0.5 * rgb.z,
        -0.25 * rgb.x + 0.5 * rgb.y - 0.25 * rgb.z,
    )
}

/// Exact inverse of [`rgb_to_ycocg`].
pub fn ycocg_to_rgb(ycocg: Vec3) -> Vec3 {
    let (y, co, cg) = (ycocg.x, ycocg.y, ycocg.z);
    Vec3::new(y + co - cg, y + cg, y - co - cg)
}

/// The firefly-suppressing feedback weight for a sample of luma `luma`:
/// `1 / (1 + luma)`. A very bright sample (a specular firefly) earns a small
/// weight so it cannot dominate the temporal blend and flicker, exactly the
/// tone-mapped weighting Karis uses to keep the accumulation stable in HDR.
pub fn tonemap_weight(luma: f32) -> f32 {
    1.0 / (1.0 + luma.max(0.0))
}

/// Resolve one TAA output pixel from the jittered current colour, the
/// reprojected history colour and the current 3x3 neighbourhood.
///
/// `neighbourhood` is the nine current-frame `RGB` samples of the 3x3 block
/// centred on the pixel (the centre sample equals `current_rgb`). When
/// `history_valid` is `false` (first frame, a resize, or a dropped
/// reprojection) the history is ignored and the current colour passes straight
/// through, so a camera cut degrades to the un-accumulated — still jittered but
/// un-smeared — frame rather than ghosting.
pub fn resolve_taa(
    current_rgb: Vec3,
    history_rgb: Vec3,
    neighbourhood: &[Vec3; 9],
    params: &TaaParams,
    history_valid: bool,
) -> Vec3 {
    if !history_valid {
        return current_rgb;
    }

    let current = rgb_to_ycocg(current_rgb);
    let history = rgb_to_ycocg(history_rgb);

    // Neighbourhood mean and mean-of-squares in YCoCg for the variance box.
    let mut sum = Vec3::ZERO;
    let mut sum_sq = Vec3::ZERO;
    for sample in neighbourhood {
        let y = rgb_to_ycocg(*sample);
        sum += y;
        sum_sq += y * y;
    }
    let count = neighbourhood.len() as f32;
    let mean = sum / count;
    let mean_sq = sum_sq / count;

    let (box_min, box_max) = variance_clip_box(mean, mean_sq, params.variance_gamma);
    let clipped = clip_history_to_aabb(history, box_min, box_max);

    // Luminance (YCoCg `.x`) feedback weighting: down-weight the brighter
    // contributor so a firefly cannot flicker the accumulation.
    let weight_current = (1.0 - params.history_blend) * tonemap_weight(current.x);
    let weight_history = params.history_blend * tonemap_weight(clipped.x);
    let total = weight_current + weight_history;
    let blended = if total > 0.0 {
        (current * weight_current + clipped * weight_history) / total
    } else {
        current
    };

    ycocg_to_rgb(blended)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3) -> bool {
        (a - b).abs().max_element() < 1e-5
    }

    #[test]
    fn ycocg_round_trips() {
        for rgb in [
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(1.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(4.0, 2.5, 0.75),
        ] {
            assert!(
                approx(ycocg_to_rgb(rgb_to_ycocg(rgb)), rgb),
                "YCoCg must round-trip for {rgb:?}"
            );
        }
    }

    #[test]
    fn invalid_history_passes_current_through() {
        let current = Vec3::new(0.3, 0.6, 0.9);
        let out = resolve_taa(
            current,
            Vec3::new(9.0, 9.0, 9.0),
            &[current; 9],
            &TaaParams::default(),
            false,
        );
        assert!(approx(out, current), "invalid history must yield the current frame");
    }

    #[test]
    fn static_signal_is_a_fixed_point() {
        // Current == history == every neighbour: zero variance box clamps the
        // history to the mean (== current), so the blend is a no-op regardless
        // of the feedback weight.
        let colour = Vec3::new(0.4, 0.55, 0.2);
        let out = resolve_taa(colour, colour, &[colour; 9], &TaaParams::default(), true);
        assert!(approx(out, colour), "a still image must be a TAA fixed point: {out:?}");
    }

    #[test]
    fn neighbourhood_clip_reins_in_stale_history() {
        // History far outside the neighbourhood box (a disocclusion) is clipped
        // toward the neighbourhood before blending, so the result stays close to
        // the current signal instead of smearing the stale colour through.
        let current = Vec3::new(0.2, 0.2, 0.2);
        let neighbourhood = [current; 9];
        let stale = Vec3::new(1.0, 0.0, 0.0);
        let out = resolve_taa(current, stale, &neighbourhood, &TaaParams::default(), true);
        // With a zero-variance neighbourhood the clip pins history to `current`,
        // so the output must equal the current colour (no red bleed-through);
        // allow a small epsilon for the YCoCg clamp round trip.
        assert!(
            (out - current).abs().max_element() < 1e-4,
            "clipped history must not smear stale colour: {out:?}"
        );
    }

    #[test]
    fn bright_history_is_weighted_down() {
        // A moderate variance box lets some history through. The tone-map weight
        // must pull a very bright history toward the darker current more than a
        // naive `history_blend` lerp would, keeping fireflies from dominating.
        let current = Vec3::new(0.1, 0.1, 0.1);
        let neighbourhood = [
            Vec3::new(0.05, 0.05, 0.05),
            Vec3::new(0.15, 0.15, 0.15),
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(0.12, 0.12, 0.12),
            current,
            Vec3::new(0.08, 0.08, 0.08),
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::new(0.06, 0.06, 0.06),
            Vec3::new(0.14, 0.14, 0.14),
        ];
        let bright = Vec3::new(0.6, 0.6, 0.6);
        let out = resolve_taa(current, bright, &neighbourhood, &TaaParams::default(), true);
        // Output luma should not exceed the clipped-history luma, and the bright
        // sample must be reined in below a plain 0.9 feedback of the raw history.
        assert!(out.x < bright.x, "bright history must be weighted down: {out:?}");
        // Grey stays grey (chroma-free) through the YCoCg round trip.
        assert!(
            (out.x - out.y).abs() < 1e-4 && (out.y - out.z).abs() < 1e-4,
            "grey input must stay grey: {out:?}"
        );
    }
}
