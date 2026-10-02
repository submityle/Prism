//! Robustness heuristics for temporal upscaling: thin-feature locking,
//! disocclusion fallback weighting, and the temporal accumulation weight.
//!
//! Reconstructing a high-resolution image from a low-resolution stream is only
//! stable if the accumulation knows *how much* history to trust each frame.
//! Three heuristics — lifted from FSR2 / UE-TSR — decide that:
//!
//! * **Thin-feature lock** — [`thin_feature_strength`] detects a sub-pixel
//!   luminance ridge (a bright wire on a dark background, a spark, a highlight
//!   edge) that the reconstruction would otherwise wash out, so the
//!   accumulation can hold onto it (via [`super::history::HistoryLock`]).
//! * **Disocclusion fallback** — [`disocclusion_history_weight`] sheds history
//!   weight as the reprojection is invalidated (off-screen, depth mismatch, or a
//!   large neighbourhood-clip overshoot), collapsing to the freshly
//!   reconstructed sample on a full disocclusion. It also folds in the FSR2-style
//!   **reactive mask** so particles, transparents and high-frequency stylised
//!   segments can shed history independently of any reprojection failure.
//! * **Temporal accumulation** — [`update_accumulation`] grows a per-pixel
//!   confidence (clamped at [`MAX_ACCUMULATION_FRAMES`]) that reduces the
//!   current sample's blend weight the longer a surface has been stably tracked,
//!   while [`history_blend_alpha`] turns that count into the actual lerp factor.
//!
//! Everything is plain arithmetic, mirrored bit-for-bit by the GPU twin.

use bevy_math::Vec3;

use crate::rgb_to_ycocg;

/// The upper bound on the temporal accumulation count, in frames. Sixteen
/// frames of history is the AAA balance between a clean, well-integrated image
/// and enough responsiveness that a lifted lock or a slow disocclusion does not
/// lag visibly.
pub const MAX_ACCUMULATION_FRAMES: f32 = 16.0;

/// The `YCoCg` luma (`.x`) of an RGB colour — the perceptual channel the
/// thin-feature detector works in, matching the clip box built in
/// [`super::history::clip_history_neighbourhood`].
#[must_use]
pub fn luma(rgb: Vec3) -> f32 {
    rgb_to_ycocg(rgb).x
}

/// The min and max luma over a 3x3 `neighbourhood` (row-major RGB).
#[must_use]
pub fn neighbourhood_luma_range(neighbourhood: &[Vec3; 9]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for sample in neighbourhood {
        let l = luma(*sample);
        lo = lo.min(l);
        hi = hi.max(l);
    }
    (lo, hi)
}

/// The strength in `[0, 1]` with which the centre pixel is a thin luminance
/// feature that should be locked.
///
/// A thin feature is a pixel that sits at a **luminance extreme** of a
/// **high-contrast** neighbourhood: a bright sub-pixel line reads as a local
/// maximum, a dark crack as a local minimum. The strength multiplies two
/// signals:
///
/// * *extremeness* — `|2t - 1|` where `t = (center - min) / contrast` is the
///   centre's normalised position in the neighbourhood luma range; `1` at either
///   extreme, `0` in the middle of the range.
/// * *contrast confidence* — `saturate((contrast - threshold) / threshold)`, so
///   a flat or low-contrast neighbourhood (noise, not a feature) scores `0`.
///
/// `contrast_threshold` is the minimum luma range that counts as a feature.
#[must_use]
pub fn thin_feature_strength(
    center_luma: f32,
    min_luma: f32,
    max_luma: f32,
    contrast_threshold: f32,
) -> f32 {
    let contrast = max_luma - min_luma;
    let threshold = contrast_threshold.max(1.0e-6);
    if contrast <= threshold {
        return 0.0;
    }
    let t = ((center_luma - min_luma) / contrast).clamp(0.0, 1.0);
    let extremeness = (2.0 * t - 1.0).abs();
    let confidence = ((contrast - threshold) / threshold).clamp(0.0, 1.0);
    (extremeness * confidence).clamp(0.0, 1.0)
}

/// Whether the centre pixel is a thin feature strong enough to request a lock
/// (strength above `0.5`).
#[must_use]
pub fn is_thin_feature(
    center_luma: f32,
    min_luma: f32,
    max_luma: f32,
    contrast_threshold: f32,
) -> bool {
    thin_feature_strength(center_luma, min_luma, max_luma, contrast_threshold) > 0.5
}

/// The history weight after disocclusion fallback and reactive shedding:
/// `base_weight · (1 - disocclusion) · (1 - reactive)`.
///
/// `disocclusion` in `[0, 1]` is the fused signal from
/// [`super::reproject::disocclusion_factor`] (optionally maxed with the
/// neighbourhood-clip overshoot). A fully disoccluded pixel (`1`) drops the
/// history entirely and the accumulation falls back to the reconstructed
/// current sample; a fully consistent pixel (`0`) keeps the full `base_weight`.
///
/// `reactive` in `[0, 1]` is the FSR2-style reactive-mask response decoded from
/// the packed motion channel (`PackedMasks::reactive`): particles, transparents
/// and high-frequency stylised segments raise it to say "do not let TAA smear
/// me". It sheds history as an event *independent* of the reprojection test —
/// composed multiplicatively so the two signals never cancel — matching the
/// golden-standard `resolve_taa` weight (`history_blend · (1 - reactive)`). A
/// fully reactive pixel (`1`) drops history even when perfectly reprojected;
/// `reactive == 0` is bit-for-bit the shipping disocclusion-only path.
#[must_use]
pub fn disocclusion_history_weight(base_weight: f32, disocclusion: f32, reactive: f32) -> f32 {
    let d = disocclusion.clamp(0.0, 1.0);
    let r = reactive.clamp(0.0, 1.0);
    base_weight.clamp(0.0, 1.0) * (1.0 - d) * (1.0 - r)
}

/// Advance the per-pixel temporal accumulation count one frame.
///
/// `prev_frames` is last frame's count. The surviving history is melted by the
/// `disocclusion` factor, but a thin-feature `lock_strength` (in `[0, 1]`,
/// from [`super::history::lock_strength`]) resists that melt so a locked feature
/// is not reset by a transient mismatch: `survived = prev · (1 - d·(1 -
/// lock))`. This frame then contributes one sample, clamped to
/// [`MAX_ACCUMULATION_FRAMES`].
///
/// A full disocclusion with no lock resets the count to `1` (this frame only);
/// a fully tracked surface climbs to the cap and stays there.
#[must_use]
pub fn update_accumulation(prev_frames: f32, disocclusion: f32, lock_strength: f32) -> f32 {
    let d = disocclusion.clamp(0.0, 1.0);
    let lock = lock_strength.clamp(0.0, 1.0);
    let melt = d * (1.0 - lock);
    let survived = prev_frames.max(0.0) * (1.0 - melt);
    (survived + 1.0).min(MAX_ACCUMULATION_FRAMES)
}

/// The current-sample blend weight (`alpha`) for an accumulation count:
/// `1 / (accumulation + 1)`.
///
/// With `accumulation == 0` the output is the current sample (`alpha == 1`);
/// as the count climbs, `alpha` falls so each new frame nudges a
/// well-integrated history only slightly. The result is the lerp factor for
/// `mix(history, current, alpha)`.
#[must_use]
pub fn history_blend_alpha(accumulation: f32) -> f32 {
    1.0 / (accumulation.max(0.0) + 1.0)
}

/// [`history_blend_alpha`] with a lower bound so the accumulation never fully
/// stops responding (a `min_alpha` floor of e.g. `1/32` keeps slow lighting
/// changes from being frozen out).
#[must_use]
pub fn history_blend_alpha_clamped(accumulation: f32, min_alpha: f32) -> f32 {
    history_blend_alpha(accumulation).max(min_alpha.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_range_spans_the_neighbourhood() {
        let mut n = [Vec3::splat(0.5); 9];
        n[0] = Vec3::splat(0.1);
        n[8] = Vec3::splat(0.9);
        let (lo, hi) = neighbourhood_luma_range(&n);
        assert!(
            (lo - 0.1).abs() < 1.0e-6 && (hi - 0.9).abs() < 1.0e-6,
            "{lo} {hi}"
        );
    }

    #[test]
    fn flat_neighbourhood_has_no_thin_feature() {
        assert_eq!(thin_feature_strength(0.5, 0.5, 0.5, 0.1), 0.0);
        assert!(!is_thin_feature(0.5, 0.5, 0.5, 0.1));
    }

    #[test]
    fn low_contrast_scores_zero() {
        // Contrast 0.05 below a 0.1 threshold => not a feature.
        assert_eq!(thin_feature_strength(0.55, 0.5, 0.55, 0.1), 0.0);
    }

    #[test]
    fn bright_ridge_is_a_strong_feature() {
        // Centre at the max of a high-contrast range => extremeness 1.
        let s = thin_feature_strength(1.0, 0.0, 1.0, 0.1);
        assert!(s > 0.9, "a bright ridge should lock strongly, got {s}");
        assert!(is_thin_feature(1.0, 0.0, 1.0, 0.1));
    }

    #[test]
    fn dark_crack_is_a_strong_feature() {
        // Centre at the min is just as much an extreme as the max.
        let s = thin_feature_strength(0.0, 0.0, 1.0, 0.1);
        assert!(s > 0.9, "a dark crack should lock strongly, got {s}");
    }

    #[test]
    fn mid_range_centre_is_not_a_feature() {
        // Centre in the middle of the range => extremeness 0.
        let s = thin_feature_strength(0.5, 0.0, 1.0, 0.1);
        assert!(
            s < 1.0e-6,
            "a mid-range pixel is not a thin feature, got {s}"
        );
        assert!(!is_thin_feature(0.5, 0.0, 1.0, 0.1));
    }

    #[test]
    fn disocclusion_weight_sheds_history() {
        // reactive == 0.0 is the shipping disocclusion-only path, unchanged.
        assert!((disocclusion_history_weight(0.9, 0.0, 0.0) - 0.9).abs() < 1.0e-6);
        assert_eq!(disocclusion_history_weight(0.9, 1.0, 0.0), 0.0);
        assert!((disocclusion_history_weight(0.8, 0.5, 0.0) - 0.4).abs() < 1.0e-6);
    }

    #[test]
    fn full_reactive_mask_drops_history_even_when_reprojected() {
        // Perfect reprojection (d = 0) but a fully reactive pixel keeps nothing.
        assert_eq!(disocclusion_history_weight(0.9, 0.0, 1.0), 0.0);
    }

    #[test]
    fn reactive_composes_multiplicatively_with_disocclusion() {
        // base 0.8, half disoccluded, half reactive: 0.8 * 0.5 * 0.5 = 0.2.
        let w = disocclusion_history_weight(0.8, 0.5, 0.5);
        assert!((w - 0.2).abs() < 1.0e-6, "got {w}");
    }

    #[test]
    fn reactive_monotonically_reduces_history_weight() {
        // With a fixed base and disocclusion, more reactive can only shed more.
        let base = 0.9;
        let d = 0.25;
        let mut prev = disocclusion_history_weight(base, d, 0.0);
        for step in 1..=8 {
            let r = step as f32 / 8.0;
            let w = disocclusion_history_weight(base, d, r);
            assert!(
                w <= prev + 1.0e-6,
                "reactive {r} raised weight: {w} > {prev}"
            );
            prev = w;
        }
    }

    #[test]
    fn accumulation_grows_on_a_tracked_surface_and_caps() {
        let mut acc = 0.0;
        for _ in 0..64 {
            acc = update_accumulation(acc, 0.0, 0.0);
        }
        assert_eq!(acc, MAX_ACCUMULATION_FRAMES);
    }

    #[test]
    fn full_disocclusion_resets_accumulation() {
        let acc = update_accumulation(12.0, 1.0, 0.0);
        assert_eq!(
            acc, 1.0,
            "a full disocclusion keeps only the current sample"
        );
    }

    #[test]
    fn a_lock_resists_the_disocclusion_reset() {
        // Full disocclusion but full lock => no melt, keeps climbing.
        let locked = update_accumulation(12.0, 1.0, 1.0);
        assert_eq!(locked, 13.0);
        // A partial lock melts partially: survived = 12*(1 - 1*0.5) = 6, +1 = 7.
        let partial = update_accumulation(12.0, 1.0, 0.5);
        assert_eq!(partial, 7.0);
    }

    #[test]
    fn blend_alpha_falls_as_history_accumulates() {
        assert!((history_blend_alpha(0.0) - 1.0).abs() < 1.0e-6);
        assert!((history_blend_alpha(1.0) - 0.5).abs() < 1.0e-6);
        assert!(history_blend_alpha(MAX_ACCUMULATION_FRAMES) < 0.07);
    }

    #[test]
    fn clamped_alpha_respects_the_floor() {
        // The unclamped alpha at the accumulation cap is 1/(16+1) ~= 0.0588, so a
        // floor above that (here 0.1) must bind and hold the response open.
        let floor = 0.1;
        let unclamped = history_blend_alpha(MAX_ACCUMULATION_FRAMES);
        assert!(
            unclamped < floor,
            "test floor must exceed unclamped alpha, got {unclamped}"
        );
        let a = history_blend_alpha_clamped(MAX_ACCUMULATION_FRAMES, floor);
        assert!(
            (a - floor).abs() < 1.0e-6,
            "alpha should hit the floor, got {a}"
        );
        // When the unclamped value already exceeds the floor it wins unchanged.
        assert!((history_blend_alpha_clamped(0.0, floor) - 1.0).abs() < 1.0e-6);
    }
}
