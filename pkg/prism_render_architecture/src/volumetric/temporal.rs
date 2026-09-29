//! Temporal `reproject`ion / upsampling plan and `history` rectification
//! (design section 10).
//!
//! Volumetric clouds are the most expensive thing on screen, so the `raymarch`
//! runs at a fraction of the output resolution and the full-resolution image is
//! reconstructed over several frames. Two low-resolution update patterns are
//! supported: a `checkerboard` pattern that resolves half the pixels each frame
//! and a `quarter-res` pattern that resolves one pixel of every `2x2` block
//! each frame. Both are deterministic and, over one period, their per-frame
//! active-pixel sets union to the whole grid so no pixel is ever starved.
//!
//! Each frame the newly `raymarch`ed pixels are combined with the previous
//! frame's result `reproject`ed along a `motion vector`. To stop stale
//! `history` from smearing ("ghosting"), the reprojected `history` sample is
//! rectified against the current-frame neighbourhood: a neighbourhood `clamp`
//! ([`clamp_history`]) and a variance box `clip` ([`variance_clip`]). Where the
//! `history` is invalid (disocclusion / large parallax) the pipeline falls back
//! to the current frame ([`should_fallback`]). The cloud `motion vector` itself
//! is the pure sum of the cloud advection velocity and the camera-induced
//! screen motion ([`composite_motion_vector`]).
//!
//! Everything here is pure and deterministic; the only float intrinsic reached
//! is the shared [`super::math::clamp`], so results are bit-reproducible frame
//! to frame.

use super::math::{clamp, Vec2};

/// The low-resolution update pattern used to reconstruct the full-resolution
/// cloud image over time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpscaleMode {
    /// Every output pixel is `raymarch`ed every frame (no upsampling). Period 1.
    Full,
    /// `checkerboard`: half the pixels resolve each frame, alternating by a
    /// per-pixel parity that flips every frame. Period 2.
    Checkerboard,
    /// `quarter-res`: one pixel of every `2x2` block resolves each frame,
    /// cycling through all four block members. Period 4.
    QuarterRes,
}

impl UpscaleMode {
    /// Number of frames after which the active-pixel pattern repeats.
    ///
    /// Over exactly this many consecutive frames the union of active pixels
    /// covers the whole grid.
    #[must_use]
    pub fn period(self) -> u32 {
        match self {
            UpscaleMode::Full => 1,
            UpscaleMode::Checkerboard => 2,
            UpscaleMode::QuarterRes => 4,
        }
    }
}

/// Whether pixel `(x, y)` is `raymarch`ed on frame `frame_index` under `mode`.
///
/// This is the deterministic scheduling core of the low-resolution `raymarch`:
/// for a fixed `(frame_index, x, y, mode)` it always returns the same answer,
/// and over one [`UpscaleMode::period`] every pixel is active on at least one
/// frame (the per-frame active sets partition, then union to, the whole grid).
///
/// - [`UpscaleMode::Full`] is always active.
/// - [`UpscaleMode::Checkerboard`] uses the parity `(x + y + frame_index)`, so
///   the resolved half swaps every frame and two consecutive frames cover all.
/// - [`UpscaleMode::QuarterRes`] selects the `2x2` sub-pixel addressed by the
///   low two bits of `frame_index`, cycling all four members every four frames.
#[must_use]
pub fn active_pixel(frame_index: u32, x: u32, y: u32, mode: UpscaleMode) -> bool {
    match mode {
        UpscaleMode::Full => true,
        UpscaleMode::Checkerboard => ((x ^ y ^ frame_index) & 1) == 0,
        UpscaleMode::QuarterRes => {
            (x & 1) == (frame_index & 1) && (y & 1) == ((frame_index >> 1) & 1)
        }
    }
}

/// Neighbourhood `history` rectification parameters (anti-ghosting).
///
/// `neighborhood_min` / `neighborhood_max` bound the reprojected `history`
/// sample to the range observed in the current-frame spatial neighbourhood, and
/// `variance_gamma` scales the standard-deviation half-width of the variance box
/// `clip`. The parameters are always ordered `neighborhood_min <=
/// neighborhood_max`: [`HistoryClampParams::new`] sorts the bounds on
/// construction, and `variance_gamma` is forced non-negative, so downstream
/// `clamp`ing can never see an inverted range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryClampParams {
    /// Lower neighbourhood bound the `history` sample is `clamp`ed to.
    pub neighborhood_min: f32,
    /// Upper neighbourhood bound the `history` sample is `clamp`ed to.
    pub neighborhood_max: f32,
    /// Non-negative standard-deviation multiplier for the variance box `clip`.
    pub variance_gamma: f32,
}

impl HistoryClampParams {
    /// Builds ordered clamp parameters, swapping the bounds if inverted.
    ///
    /// Guarantees `neighborhood_min <= neighborhood_max` and
    /// `variance_gamma >= 0`, so the invariant holds regardless of caller input.
    #[must_use]
    pub fn new(min: f32, max: f32, variance_gamma: f32) -> Self {
        let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
        let gamma = if variance_gamma < 0.0 {
            0.0
        } else {
            variance_gamma
        };
        Self {
            neighborhood_min: lo,
            neighborhood_max: hi,
            variance_gamma: gamma,
        }
    }

    /// Returns `true` when the bounds are ordered and gamma is non-negative.
    #[must_use]
    pub fn is_ordered(&self) -> bool {
        self.neighborhood_min <= self.neighborhood_max && self.variance_gamma >= 0.0
    }
}

/// The per-frame temporal `reproject`ion / upsampling plan.
///
/// Bundles the low-resolution update `mode`, the `history` rectification
/// parameters, and the maximum number of frames a `history` sample is trusted
/// before it must be refreshed. This is the `CPU` decision the shared temporal
/// upscale service consumes for the cloud buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReprojectionPlan {
    /// Low-resolution update pattern for this frame.
    pub mode: UpscaleMode,
    /// `history` rectification (anti-ghosting) parameters.
    pub clamp: HistoryClampParams,
    /// Maximum frame age a `history` sample is trusted before a forced refresh.
    pub max_history_frames: u32,
}

/// `clamp`s a reprojected `history` sample into the current neighbourhood.
///
/// Bounds `history_sample` to `[history_min, history_max]`, the range observed
/// in the current-frame spatial neighbourhood, which is the cheapest and most
/// robust anti-ghosting rectification. When the bounds are supplied inverted
/// they are ordered first, so the result is always within the ordered range and
/// never `NaN`.
#[must_use]
pub fn clamp_history(history_sample: f32, history_min: f32, history_max: f32) -> f32 {
    let (lo, hi) = if history_min <= history_max {
        (history_min, history_max)
    } else {
        (history_max, history_min)
    };
    clamp(history_sample, lo, hi)
}

/// Variance box `clip` of a `history` sample around the current-frame mean.
///
/// Clips `history_sample` to `[mean - gamma*std_dev, mean + gamma*std_dev]`,
/// the anisotropic `AABB` variance clip that keeps more `history` than a hard
/// neighbourhood `clamp` while still rejecting ghosting. `std_dev` and `gamma`
/// are treated as non-negative (negatives are floored to zero), so the clip
/// window is always well-formed and the result lies inside it.
#[must_use]
pub fn variance_clip(history_sample: f32, mean: f32, std_dev: f32, gamma: f32) -> f32 {
    let sd = if std_dev < 0.0 { 0.0 } else { std_dev };
    let g = if gamma < 0.0 { 0.0 } else { gamma };
    let half = sd * g;
    clamp(history_sample, mean - half, mean + half)
}

/// Whether this pixel must fall back to the current frame instead of `history`.
///
/// The `history` is discarded (fall back to the freshly `raymarch`ed current
/// frame) when it was flagged invalid — for example a `reproject`ion that lands
/// off-screen — or when the parallax disocclusion measure exceeds
/// `max_disocclusion`. This is the deterministic invalidation rule that stops
/// disoccluded or high-parallax regions from dragging stale cloud color.
#[must_use]
pub fn should_fallback(history_valid: bool, disocclusion: f32, max_disocclusion: f32) -> bool {
    !history_valid || disocclusion > max_disocclusion
}

/// Composites the cloud screen-space `motion vector` from its two sources.
///
/// The cloud `motion vector` is the pure sum of the cloud advection velocity
/// (wind-driven cloud drift projected to screen space) and the camera-induced
/// screen motion. Keeping it a pure [`Vec2`] combination makes the
/// `reproject`ion deterministic and trivially testable.
#[must_use]
pub fn composite_motion_vector(cloud_advection: Vec2, camera_motion: Vec2) -> Vec2 {
    cloud_advection.add(camera_motion)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Collects, over one full period, the set of pixels active at least once.
    fn covered(mode: UpscaleMode, w: u32, h: u32) -> Vec<bool> {
        let mut seen = vec![false; (w * h) as usize];
        for frame in 0..mode.period() {
            for y in 0..h {
                for x in 0..w {
                    if active_pixel(frame, x, y, mode) {
                        seen[(y * w + x) as usize] = true;
                    }
                }
            }
        }
        seen
    }

    #[test]
    fn active_pixel_is_deterministic() {
        for mode in [
            UpscaleMode::Full,
            UpscaleMode::Checkerboard,
            UpscaleMode::QuarterRes,
        ] {
            for frame in 0..8 {
                for y in 0..4 {
                    for x in 0..4 {
                        assert_eq!(
                            active_pixel(frame, x, y, mode),
                            active_pixel(frame, x, y, mode)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn one_period_covers_every_pixel() {
        for mode in [
            UpscaleMode::Full,
            UpscaleMode::Checkerboard,
            UpscaleMode::QuarterRes,
        ] {
            let seen = covered(mode, 8, 8);
            assert!(seen.iter().all(|&s| s), "{mode:?} left a pixel unresolved");
        }
    }

    #[test]
    fn checkerboard_resolves_half_each_frame() {
        // Each frame exactly half of an even-sized grid is active, and the two
        // frames of the period are disjoint (a true checkerboard swap).
        let (w, h) = (8u32, 8u32);
        let mut frame0 = 0;
        let mut frame1 = 0;
        for y in 0..h {
            for x in 0..w {
                if active_pixel(0, x, y, UpscaleMode::Checkerboard) {
                    frame0 += 1;
                }
                if active_pixel(1, x, y, UpscaleMode::Checkerboard) {
                    frame1 += 1;
                }
                // Disjoint across the two frames.
                assert_ne!(
                    active_pixel(0, x, y, UpscaleMode::Checkerboard),
                    active_pixel(1, x, y, UpscaleMode::Checkerboard)
                );
            }
        }
        assert_eq!(frame0, (w * h) / 2);
        assert_eq!(frame1, (w * h) / 2);
    }

    #[test]
    fn quarter_res_resolves_one_of_four_each_frame() {
        // Each frame resolves exactly a quarter of the pixels.
        let (w, h) = (8u32, 8u32);
        for frame in 0..4 {
            let mut count = 0;
            for y in 0..h {
                for x in 0..w {
                    if active_pixel(frame, x, y, UpscaleMode::QuarterRes) {
                        count += 1;
                    }
                }
            }
            assert_eq!(
                count,
                (w * h) / 4,
                "frame {frame} did not resolve a quarter"
            );
        }
    }

    #[test]
    fn clamp_params_are_ordered_even_when_inverted() {
        let ordered = HistoryClampParams::new(0.2, 0.8, 1.5);
        assert!(ordered.is_ordered());
        assert_eq!(ordered.neighborhood_min, 0.2);
        assert_eq!(ordered.neighborhood_max, 0.8);

        // Inverted bounds and negative gamma are normalized.
        let fixed = HistoryClampParams::new(0.9, 0.1, -3.0);
        assert!(fixed.is_ordered());
        assert_eq!(fixed.neighborhood_min, 0.1);
        assert_eq!(fixed.neighborhood_max, 0.9);
        assert_eq!(fixed.variance_gamma, 0.0);
    }

    #[test]
    fn clamp_history_result_is_within_bounds() {
        assert_eq!(clamp_history(5.0, 0.0, 1.0), 1.0);
        assert_eq!(clamp_history(-5.0, 0.0, 1.0), 0.0);
        assert_eq!(clamp_history(0.5, 0.0, 1.0), 0.5);
        // Inverted bounds are ordered internally; result stays in [min, max].
        let v = clamp_history(0.5, 1.0, 0.0);
        assert!((0.0..=1.0).contains(&v));
        // Property sweep.
        let (lo, hi) = (-2.0, 3.0);
        let mut x = -10.0;
        while x <= 10.0 {
            let c = clamp_history(x, lo, hi);
            assert!((lo..=hi).contains(&c), "clamp escaped range at x={x}");
            x += 0.5;
        }
    }

    #[test]
    fn variance_clip_result_is_within_window() {
        let (mean, sd, gamma) = (0.5, 0.1, 2.0);
        let half = sd * gamma;
        let mut x = -5.0;
        while x <= 5.0 {
            let c = variance_clip(x, mean, sd, gamma);
            assert!(
                (mean - half - 1e-6..=mean + half + 1e-6).contains(&c),
                "variance clip escaped window at x={x}"
            );
            x += 0.25;
        }
        // Negative std_dev / gamma collapse the window to the mean.
        assert_eq!(variance_clip(9.0, 0.5, -1.0, 2.0), 0.5);
        assert_eq!(variance_clip(9.0, 0.5, 1.0, -2.0), 0.5);
    }

    #[test]
    fn variance_clip_is_deterministic() {
        assert_eq!(
            variance_clip(0.7, 0.5, 0.1, 2.0),
            variance_clip(0.7, 0.5, 0.1, 2.0)
        );
    }

    #[test]
    fn fallback_triggers_on_invalid_or_high_parallax() {
        // Valid history with low disocclusion keeps history.
        assert!(!should_fallback(true, 0.1, 0.5));
        // Invalid history always falls back.
        assert!(should_fallback(false, 0.0, 0.5));
        // High disocclusion falls back even when the flag is valid.
        assert!(should_fallback(true, 0.9, 0.5));
        // Exactly at the threshold does not fall back (strict greater-than).
        assert!(!should_fallback(true, 0.5, 0.5));
    }

    #[test]
    fn motion_vector_is_pure_sum_of_sources() {
        let advection = Vec2::new(1.0, -2.0);
        let camera = Vec2::new(0.5, 3.0);
        let mv = composite_motion_vector(advection, camera);
        assert_eq!(mv, Vec2::new(1.5, 1.0));
        // Commutative and deterministic.
        assert_eq!(mv, composite_motion_vector(camera, advection));
        assert_eq!(mv, composite_motion_vector(advection, camera));
        // Zero camera motion leaves pure advection.
        assert_eq!(composite_motion_vector(advection, Vec2::ZERO), advection);
    }
}
