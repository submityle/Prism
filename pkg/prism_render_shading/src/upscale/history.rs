//! History reconstruction, neighbourhood colour clipping, and the thin-feature
//! **history lock** for temporal upscaling.
//!
//! The history buffer is display-resolution, but the current frame arrives at
//! render resolution, so before the two can be blended the low-resolution input
//! must be *reconstructed* onto the display grid. A naive bilinear upsample
//! blurs away the very detail temporal upscaling exists to recover, so this
//! module uses a **Lanczos-2** windowed-sinc kernel — the FSR2 / UE-TSR choice —
//! which preserves high-frequency edges with a controlled amount of ringing.
//!
//! Two more pieces guard the accumulation:
//!
//! * **`YCoCg` neighbourhood clipping** — [`clip_history_neighbourhood`] clips
//!   the reprojected history to the colour-variance box of the current
//!   neighbourhood in `YCoCg` (reusing [`crate::variance_clip_box`] /
//!   [`crate::clip_history_to_aabb_ex`]), the standard anti-ghosting primitive,
//!   and reports how far the history sat outside the box so the accumulation can
//!   shed weight on a mismatch.
//! * **History lock** — [`HistoryLock`] tracks how long a thin feature (see
//!   [`super::robust`]) has been continuously present so the accumulation can
//!   hold onto its sub-pixel detail instead of letting the reconstruction wash
//!   it out; the lock melts on disocclusion.
//!
//! `sin` routes through [`bevy_math::ops`] so this golden and its GPU twin agree
//! bit-for-bit.

use bevy_math::{ops, Vec2, Vec3};
use core::f32::consts::PI;

use crate::{clip_history_to_aabb_ex, rgb_to_ycocg, variance_clip_box, ycocg_to_rgb};

/// The maximum lifetime, in frames, a [`HistoryLock`] can accumulate. Four
/// frames is enough to carry a thin feature through the reconstruction without
/// letting a stale lock outlive the detail that justified it.
pub const MAX_LOCK_LIFETIME: f32 = 4.0;

/// The normalised **sinc** `sin(pi·x) / (pi·x)`, with the removable singularity
/// at `x == 0` filled in as `1`.
#[must_use]
pub fn sinc(x: f32) -> f32 {
    if x.abs() < 1.0e-6 {
        1.0
    } else {
        let px = PI * x;
        ops::sin(px) / px
    }
}

/// The **Lanczos-2** windowed-sinc kernel value at signed distance `x`:
/// `sinc(x)·sinc(x / 2)` inside the two-lobe support `|x| < 2`, and `0` outside.
///
/// The `a = 2` window is the temporal-upscale sweet spot: sharp enough to keep
/// edges crisp, narrow enough that the negative lobes do not ring visibly.
#[must_use]
pub fn lanczos2_weight(x: f32) -> f32 {
    if x.abs() >= 2.0 {
        0.0
    } else {
        sinc(x) * sinc(x * 0.5)
    }
}

/// The four normalised Lanczos-2 tap weights for a sample sitting a fractional
/// `frac` in `[0, 1)` past the second of four consecutive taps.
///
/// The taps are at integer offsets `-1, 0, 1, 2` from the sample's floor, so the
/// signed distances from the sample are `frac + 1`, `frac`, `frac - 1`,
/// `frac - 2`. Weights are normalised to sum to `1` so a flat input reconstructs
/// exactly (no brightness shift); a degenerate zero sum falls back to the
/// nearest tap.
#[must_use]
pub fn lanczos2_weights(frac: f32) -> [f32; 4] {
    let f = frac.clamp(0.0, 1.0);
    let mut w = [
        lanczos2_weight(f + 1.0),
        lanczos2_weight(f),
        lanczos2_weight(f - 1.0),
        lanczos2_weight(f - 2.0),
    ];
    let sum = w[0] + w[1] + w[2] + w[3];
    if sum.abs() > 1.0e-6 {
        let inv = 1.0 / sum;
        for weight in &mut w {
            *weight *= inv;
        }
    } else {
        w = [0.0, 1.0, 0.0, 0.0];
    }
    w
}

/// Reconstruct a single scalar from four consecutive taps and a fractional
/// position with the Lanczos-2 kernel.
#[must_use]
pub fn reconstruct_lanczos2_1d(taps: [f32; 4], frac: f32) -> f32 {
    let w = lanczos2_weights(frac);
    taps[0] * w[0] + taps[1] * w[1] + taps[2] * w[2] + taps[3] * w[3]
}

/// Reconstruct an RGB sample from a 4x4 tap grid and a 2D fractional position
/// with the separable Lanczos-2 kernel.
///
/// `taps[row][col]` is the low-resolution colour at row-major offsets `-1..=2`
/// on each axis (rows along `y`, columns along `x`); `frac` is the sub-tap
/// position of the reconstructed sample within the central cell. Separability
/// makes this the outer product of the per-axis [`lanczos2_weights`].
#[must_use]
pub fn reconstruct_lanczos2(taps: &[[Vec3; 4]; 4], frac: Vec2) -> Vec3 {
    let wx = lanczos2_weights(frac.x);
    let wy = lanczos2_weights(frac.y);
    let mut acc = Vec3::ZERO;
    for (row, wy_r) in taps.iter().zip(wy.iter()) {
        let mut row_acc = Vec3::ZERO;
        for (tap, wx_c) in row.iter().zip(wx.iter()) {
            row_acc += *tap * *wx_c;
        }
        acc += row_acc * *wy_r;
    }
    acc
}

/// The reprojected history clipped to the current neighbourhood plus how far
/// outside the colour box it began.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClippedHistory {
    /// History colour clamped onto (or left inside) the neighbourhood box, in
    /// RGB.
    pub color: Vec3,
    /// `0` when the history was already inside the box, rising past `1` as it
    /// disagreed — the disocclusion-like signal [`super::robust`] decays the
    /// blend with.
    pub overshoot: f32,
}

/// Clip the reprojected `history_rgb` to the `YCoCg` variance box of the current
/// 3x3 `neighbourhood` (row-major RGB, centre at index 4).
///
/// Builds the `mean ± gamma·sigma` box from the neighbourhood's first two
/// `YCoCg` moments ([`crate::variance_clip_box`]), clips with the Karis AABB
/// method ([`crate::clip_history_to_aabb_ex`]), and converts back to RGB. A
/// tighter `gamma` resolves edges more crisply at the cost of a little more
/// ghosting; `1.0` is the AAA default.
#[must_use]
pub fn clip_history_neighbourhood(
    history_rgb: Vec3,
    neighbourhood: &[Vec3; 9],
    gamma: f32,
) -> ClippedHistory {
    let history = rgb_to_ycocg(history_rgb);
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
    let (box_min, box_max) = variance_clip_box(mean, mean_sq, gamma);
    let clipped = clip_history_to_aabb_ex(history, box_min, box_max);
    ClippedHistory {
        color: ycocg_to_rgb(clipped.clipped),
        overshoot: clipped.overshoot,
    }
}

/// A thin-feature history lock: how many frames a sub-pixel feature has been
/// continuously present and trusted.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HistoryLock {
    /// Remaining lock lifetime in frames, in `[0, MAX_LOCK_LIFETIME]`.
    pub lifetime: f32,
}

/// Advance a [`HistoryLock`] one frame.
///
/// A disocclusion (`disocclusion` in `[0, 1]`) melts the surviving lifetime
/// proportionally — a fully disoccluded pixel clears the lock outright. When the
/// thin-feature detector still `requested` a lock the lifetime is refreshed
/// (incremented, clamped to [`MAX_LOCK_LIFETIME`]); otherwise it decays by one
/// frame toward zero.
#[must_use]
pub fn update_lock(prev: HistoryLock, requested: bool, disocclusion: f32) -> HistoryLock {
    let d = disocclusion.clamp(0.0, 1.0);
    let survived = prev.lifetime.max(0.0) * (1.0 - d);
    let next = if requested {
        (survived + 1.0).min(MAX_LOCK_LIFETIME)
    } else {
        (survived - 1.0).max(0.0)
    };
    HistoryLock { lifetime: next }
}

/// The normalised lock strength in `[0, 1]` (`lifetime / MAX_LOCK_LIFETIME`),
/// the factor the accumulation uses to bias toward preserving locked history.
#[must_use]
pub fn lock_strength(lock: HistoryLock) -> f32 {
    (lock.lifetime / MAX_LOCK_LIFETIME).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_is_one_at_the_origin_and_zero_at_integers() {
        assert!((sinc(0.0) - 1.0).abs() < 1.0e-6);
        assert!(sinc(1.0).abs() < 1.0e-6);
        assert!(sinc(2.0).abs() < 1.0e-6);
        assert!(sinc(-3.0).abs() < 1.0e-6);
    }

    #[test]
    fn lanczos_kernel_has_two_lobe_support() {
        assert!((lanczos2_weight(0.0) - 1.0).abs() < 1.0e-6);
        // Zero crossings at the integers within support.
        assert!(lanczos2_weight(1.0).abs() < 1.0e-6);
        // Compact support: nothing past |x| == 2.
        assert_eq!(lanczos2_weight(2.0), 0.0);
        assert_eq!(lanczos2_weight(2.5), 0.0);
        assert_eq!(lanczos2_weight(-2.1), 0.0);
        // A negative lobe exists between the first and second zero crossings.
        assert!(lanczos2_weight(1.5) < 0.0);
    }

    #[test]
    fn weights_are_normalised() {
        for &frac in &[0.0, 0.25, 0.5, 0.75, 0.999] {
            let w = lanczos2_weights(frac);
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-5, "frac {frac}: sum {sum}");
        }
    }

    #[test]
    fn weights_collapse_to_the_centre_tap_at_zero_frac() {
        let w = lanczos2_weights(0.0);
        assert!((w[1] - 1.0).abs() < 1.0e-6, "centre tap should dominate: {w:?}");
        assert!(w[0].abs() < 1.0e-6 && w[2].abs() < 1.0e-6 && w[3].abs() < 1.0e-6);
    }

    #[test]
    fn reconstruct_is_exact_on_a_flat_signal() {
        // A constant input must reconstruct to that constant for any frac.
        let taps = [3.0, 3.0, 3.0, 3.0];
        for &frac in &[0.0, 0.3, 0.5, 0.85] {
            assert!((reconstruct_lanczos2_1d(taps, frac) - 3.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn reconstruct_passes_through_the_centre_tap_at_zero_frac() {
        let taps = [0.0, 7.0, 0.0, 0.0];
        assert!((reconstruct_lanczos2_1d(taps, 0.0) - 7.0).abs() < 1.0e-5);
    }

    #[test]
    fn reconstruct_2d_is_flat_preserving() {
        let taps = [[Vec3::splat(0.4); 4]; 4];
        let out = reconstruct_lanczos2(&taps, Vec2::new(0.3, 0.7));
        assert!((out - Vec3::splat(0.4)).abs().max_element() < 1.0e-5, "{out:?}");
    }

    #[test]
    fn reconstruct_2d_picks_the_centre_cell_at_zero_frac() {
        let mut taps = [[Vec3::ZERO; 4]; 4];
        taps[1][1] = Vec3::new(0.2, 0.5, 0.9);
        let out = reconstruct_lanczos2(&taps, Vec2::ZERO);
        assert!((out - taps[1][1]).abs().max_element() < 1.0e-5, "{out:?}");
    }

    #[test]
    fn clip_leaves_consistent_history_untouched() {
        let neighbourhood = [Vec3::new(0.5, 0.5, 0.5); 9];
        let clipped = clip_history_neighbourhood(Vec3::new(0.5, 0.5, 0.5), &neighbourhood, 1.0);
        assert!((clipped.color - Vec3::splat(0.5)).abs().max_element() < 1.0e-5);
        assert_eq!(clipped.overshoot, 0.0);
    }

    #[test]
    fn clip_reins_in_stale_history_and_reports_overshoot() {
        let neighbourhood = [Vec3::new(0.2, 0.2, 0.2); 9];
        let clipped = clip_history_neighbourhood(Vec3::new(1.0, 0.0, 0.0), &neighbourhood, 1.0);
        // Zero-variance neighbourhood pins history to the mean (no red bleed).
        assert!((clipped.color - Vec3::splat(0.2)).abs().max_element() < 1.0e-4, "{:?}", clipped.color);
        assert!(clipped.overshoot > 0.0, "a stale sample must report overshoot");
    }

    #[test]
    fn lock_accumulates_while_requested_and_saturates() {
        let mut lock = HistoryLock::default();
        for _ in 0..10 {
            lock = update_lock(lock, true, 0.0);
        }
        assert_eq!(lock.lifetime, MAX_LOCK_LIFETIME);
        assert!((lock_strength(lock) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn lock_decays_when_not_requested() {
        let lock = update_lock(HistoryLock { lifetime: 3.0 }, false, 0.0);
        assert_eq!(lock.lifetime, 2.0);
    }

    #[test]
    fn disocclusion_melts_the_lock() {
        let full = update_lock(HistoryLock { lifetime: 4.0 }, false, 1.0);
        assert_eq!(full.lifetime, 0.0, "a full disocclusion clears the lock");
        let partial = update_lock(HistoryLock { lifetime: 4.0 }, true, 0.5);
        // survived = 4*0.5 = 2, requested => +1 => 3.
        assert_eq!(partial.lifetime, 3.0);
    }
}
