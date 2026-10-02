//! Per-pixel temporal accumulation resolve.
//!
//! This module is the heart of the reconstructor: given the current frame's
//! color at an output pixel, a small window of its neighbors, and the
//! reprojected history color plus how many frames have accumulated into it, it
//! decides the final color and the updated history to store. It composes the
//! sibling modules — the Karis tone-map and `YCoCg` transform
//! ([`super::color`]), the neighborhood statistics and clip box
//! ([`super::neighborhood`]) — into the standard `TAA` / temporal-upsampler
//! resolve, which is:
//!
//! 1. move current, history, and the neighborhood into the bounded tone-mapped
//!    `YCoCg` domain where blending is stable and chroma is decorrelated;
//! 2. build the variance clip box from the neighborhood and clip the history
//!    into it, pulling stale color back toward the current frame;
//! 3. choose a blend weight from the accumulated confidence (an exponential
//!    moving average that converges as more frames land) and raise it toward
//!    "take the current color" in proportion to how far the history had to be
//!    clipped (color-based rejection), which is what suppresses ghosting;
//! 4. blend, map back to linear `HDR`, and update the confidence so a stable
//!    pixel keeps sharpening while a rejected one restarts.
//!
//! A disocclusion (reported by [`super::reproject`]) or absent history short-
//! circuits to the current color with a fresh single-frame confidence. Every
//! operation is `+`, `-`, `*`, `/`, `min`/`max`, so a `GPU` kernel reproduces
//! the resolve bit-for-bit.

use super::color::{luminance, rgb_to_ycocg, tonemap, untonemap, ycocg_to_rgb};
use super::lock::{advance_lock, rejection_scale, LockState};
use super::neighborhood::{clip_to_aabb, NeighborhoodStats};
use alloc::vec::Vec;

/// Default variance-box width in neighborhood standard deviations.
///
/// `1.0` is the common `TAA` default: tight enough to reject ghosts, loose
/// enough not to clip genuine high-frequency detail into oblivion.
pub const DEFAULT_VARIANCE_GAMMA: f32 = 1.0;

/// Default cap on accumulated confidence (effective history frame count).
///
/// Capping the accumulation keeps the reconstructor responsive: without a cap
/// the blend weight would shrink toward zero and the image would stop reacting
/// to real change. Sixteen frames is a typical convergence/latency trade.
pub const DEFAULT_MAX_CONFIDENCE: f32 = 16.0;

/// Tunable weights for the temporal resolve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolveParams {
    /// Variance-box half-width in standard deviations (see
    /// [`DEFAULT_VARIANCE_GAMMA`]). Larger keeps more history; smaller rejects
    /// more aggressively.
    pub variance_gamma: f32,
    /// Maximum accumulated confidence (see [`DEFAULT_MAX_CONFIDENCE`]), which
    /// floors the blend weight and bounds history latency.
    pub max_confidence: f32,
}

impl Default for ResolveParams {
    fn default() -> Self {
        Self {
            variance_gamma: DEFAULT_VARIANCE_GAMMA,
            max_confidence: DEFAULT_MAX_CONFIDENCE,
        }
    }
}

impl ResolveParams {
    /// Returns a copy with every field clamped into a safe working range.
    ///
    /// `variance_gamma` is forced non-negative (a negative box is meaningless;
    /// `0` clips straight to the mean) and `max_confidence` is floored at `1`
    /// so there is always at least the current frame of accumulation. `NaN`
    /// resolves to the default for each field.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let variance_gamma = if self.variance_gamma >= 0.0 {
            self.variance_gamma
        } else {
            DEFAULT_VARIANCE_GAMMA
        };
        let max_confidence = if self.max_confidence >= 1.0 {
            self.max_confidence
        } else {
            DEFAULT_MAX_CONFIDENCE
        };
        Self {
            variance_gamma,
            max_confidence,
        }
    }
}

/// The result of resolving one output pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolveOutput {
    /// Final linear `HDR` color to display for this pixel.
    pub color: [f32; 3],
    /// Updated confidence (effective accumulated frame count) to store as this
    /// pixel's history weight for next frame, in `[1, max_confidence]`.
    pub confidence: f32,
}

/// Resolves one output pixel's temporal accumulation.
///
/// `current_color` is the current frame's resolved linear `HDR` color at this
/// output pixel and `neighborhood` is the window of current-frame colors used
/// to build the clip box (the pixel itself should be included). `history_color`
/// is the already-reprojected, already-resampled history (see
/// [`super::reproject`]) and `history_confidence` is how many frames have
/// accumulated into it. `disoccluded` is the reprojection verdict: when it is
/// `true`, or when there is no usable history, the resolve restarts from the
/// current color.
///
/// See the module docs for the full algorithm. The returned confidence is
/// always in `[1, max_confidence]`.
#[must_use]
pub fn resolve(
    params: ResolveParams,
    current_color: [f32; 3],
    neighborhood: &[[f32; 3]],
    history_color: [f32; 3],
    history_confidence: f32,
    disoccluded: bool,
) -> ResolveOutput {
    resolve_core(
        params,
        current_color,
        neighborhood,
        history_color,
        history_confidence,
        disoccluded,
        1.0,
    )
}

/// Resolves one output pixel, advancing a thin-feature [`LockState`] alongside.
///
/// This is the lock-aware entry point: it detects a shading change / disocclusion,
/// advances `previous_lock` using the current luma and the caller-supplied
/// `thin_strength` (from [`super::lock::thin_feature_strength`]), and uses the
/// resulting lock to suppress color rejection on a thin feature so the clamp
/// cannot eat its accumulated detail (see [`super::lock`]). Returns the resolved
/// pixel together with the lock to store for next frame.
#[must_use]
pub fn resolve_with_lock(
    params: ResolveParams,
    current_color: [f32; 3],
    neighborhood: &[[f32; 3]],
    history_color: [f32; 3],
    history_confidence: f32,
    disoccluded: bool,
    previous_lock: LockState,
    thin_strength: f32,
) -> (ResolveOutput, LockState) {
    let lock = advance_lock(
        previous_lock,
        luminance(sanitize_color(current_color)),
        thin_strength,
        disoccluded,
    );
    let output = resolve_core(
        params,
        current_color,
        neighborhood,
        history_color,
        history_confidence,
        disoccluded,
        rejection_scale(lock),
    );
    (output, lock)
}

/// The shared resolve body, parameterized by a `rejection_scale` in `[0, 1]`
/// that attenuates the color rejection (see [`resolve`] for the full
/// algorithm). A scale of `1` is the plain resolve; a locked thin feature
/// passes a smaller scale so its history survives the clamp.
#[must_use]
fn resolve_core(
    params: ResolveParams,
    current_color: [f32; 3],
    neighborhood: &[[f32; 3]],
    history_color: [f32; 3],
    history_confidence: f32,
    disoccluded: bool,
    rejection_scale: f32,
) -> ResolveOutput {
    let params = params.sanitized();

    // Reset path: a disocclusion, missing/invalid history, or an empty
    // neighborhood all mean there is nothing trustworthy to blend against.
    if disoccluded
        || history_confidence.is_nan()
        || history_confidence <= 0.0
        || neighborhood.is_empty()
    {
        return ResolveOutput {
            color: sanitize_color(current_color),
            confidence: 1.0,
        };
    }

    // Move everything into the bounded tone-mapped YCoCg domain.
    let current_t = rgb_to_ycocg(tonemap(sanitize_color(current_color)));
    let history_t = rgb_to_ycocg(tonemap(sanitize_color(history_color)));
    let mut window = alloc_window(neighborhood.len());
    for (dst, src) in window.iter_mut().zip(neighborhood) {
        *dst = rgb_to_ycocg(tonemap(sanitize_color(*src)));
    }

    // Build the variance clip box and clip the history into it.
    let stats = NeighborhoodStats::from_samples(&window);
    let (lo, hi) = stats.variance_aabb(params.variance_gamma);
    let history_clipped = clip_to_aabb(lo, hi, history_t);

    // Color-based rejection: how far the history's luma had to move during
    // clipping, relative to the box's luma half-width. A history that sat well
    // outside the plausible set is a likely ghost and is rejected toward the
    // current color.
    let box_half_width_y = 0.5 * (hi[0] - lo[0]);
    let clip_dist_y = (history_t[0] - history_clipped[0]).abs();
    let rejection =
        (clip_dist_y / (box_half_width_y + 1e-3)).clamp(0.0, 1.0) * rejection_scale.clamp(0.0, 1.0);

    // Lock influence relaxes the clamp itself: a thin-feature lock
    // (`rejection_scale` below 1) blends the clamped history back toward the
    // raw reprojected history so a one-pixel feature survives a neighborhood
    // too coarse to contain it. Unlocked pixels (scale 1) use the clamped
    // history unchanged.
    let lock_influence = (1.0 - rejection_scale).clamp(0.0, 1.0);
    let mut history_eff = [0.0f32; 3];
    for c in 0..3 {
        history_eff[c] = history_clipped[c] + (history_t[c] - history_clipped[c]) * lock_influence;
    }

    // Exponential-moving-average weight from the accumulated confidence, raised
    // toward 1 (take current) as rejection grows.
    let eff = history_confidence.min(params.max_confidence);
    let base_alpha = 1.0 / (eff + 1.0);
    let alpha = base_alpha + (1.0 - base_alpha) * rejection;

    // Blend in the bounded domain, then map back to linear HDR.
    let mut blended_t = [0.0f32; 3];
    for c in 0..3 {
        blended_t[c] = history_eff[c] + (current_t[c] - history_eff[c]) * alpha;
    }
    let color = untonemap(ycocg_to_rgb(blended_t));

    // Advance the confidence, slowed by rejection so a ghosty pixel reconverges
    // instead of locking onto stale history.
    let confidence = ((eff + 1.0) * (1.0 - rejection)).clamp(1.0, params.max_confidence);

    ResolveOutput {
        color: [color[0].max(0.0), color[1].max(0.0), color[2].max(0.0)],
        confidence,
    }
}

/// Replaces any non-finite channel with `0` and clamps negatives to `0`.
///
/// Keeps a stray `NaN`/`inf`/negative input (from an upstream pass) from
/// poisoning the tone-map and the accumulated history; the resolve stays
/// deterministic and the stored color stays a valid non-negative `HDR` value.
#[must_use]
fn sanitize_color(c: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        out[i] = if c[i].is_finite() { c[i].max(0.0) } else { 0.0 };
    }
    out
}

/// Allocates a scratch window initialized to black.
///
/// Split out so the resolve body reads cleanly; the length matches the caller's
/// neighborhood so the subsequent fill overwrites every entry.
#[must_use]
fn alloc_window(len: usize) -> Vec<[f32; 3]> {
    alloc::vec![[0.0f32; 3]; len]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-error tolerance for the resolve checks.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4
    }

    /// A tight neighborhood centered on `color` (nine near-identical samples).
    fn tight_window(color: [f32; 3]) -> alloc::vec::Vec<[f32; 3]> {
        alloc::vec![color; 9]
    }

    #[test]
    fn disocclusion_takes_current_color() {
        let out = resolve(
            ResolveParams::default(),
            [0.4, 0.5, 0.6],
            &tight_window([0.4, 0.5, 0.6]),
            [10.0, 0.0, 0.0],
            8.0,
            true,
        );
        assert!(
            approx(out.color[0], 0.4) && approx(out.color[1], 0.5) && approx(out.color[2], 0.6)
        );
        assert!(approx(out.confidence, 1.0));
    }

    #[test]
    fn missing_history_takes_current_color() {
        let out = resolve(
            ResolveParams::default(),
            [0.2, 0.2, 0.2],
            &tight_window([0.2, 0.2, 0.2]),
            [9.0, 9.0, 9.0],
            0.0,
            false,
        );
        assert!(approx(out.color[0], 0.2));
        assert!(approx(out.confidence, 1.0));
    }

    #[test]
    fn stable_pixel_converges_and_builds_confidence() {
        // Current and history agree; the blend should stay on that color and
        // the confidence should climb by one frame.
        let color = [0.5, 0.5, 0.5];
        let out = resolve(
            ResolveParams::default(),
            color,
            &tight_window(color),
            color,
            4.0,
            false,
        );
        assert!(
            approx(out.color[0], 0.5) && approx(out.color[1], 0.5) && approx(out.color[2], 0.5)
        );
        assert!(approx(out.confidence, 5.0), "confidence {}", out.confidence);
    }

    #[test]
    fn higher_confidence_keeps_more_history() {
        // History differs slightly but stays inside the (loosened) box; higher
        // accumulated confidence must move the result less toward current.
        let current = [0.6, 0.5, 0.5];
        // A window spanning both colors so neither is clipped away.
        let window = alloc::vec![[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.55, 0.5, 0.5]];
        let history = [0.5, 0.5, 0.5];
        let params = ResolveParams {
            variance_gamma: 4.0,
            ..ResolveParams::default()
        };
        let low = resolve(params, current, &window, history, 1.0, false);
        let high = resolve(params, current, &window, history, 15.0, false);
        // Both move from history (0.5) toward current (0.6); low confidence
        // moves further.
        assert!(
            low.color[0] > high.color[0],
            "{} !> {}",
            low.color[0],
            high.color[0]
        );
        assert!(high.color[0] >= 0.5 && low.color[0] <= 0.6);
    }

    #[test]
    fn out_of_gamut_history_is_rejected_toward_current() {
        // History is a bright red ghost far outside a dim neighborhood; the
        // resolve must snap toward the current color and drop confidence.
        let current = [0.1, 0.1, 0.1];
        let out = resolve(
            ResolveParams::default(),
            current,
            &tight_window(current),
            [20.0, 0.0, 0.0],
            16.0,
            false,
        );
        // Result is close to current, not the bright ghost.
        assert!(out.color[0] < 1.0, "ghost leaked through: {}", out.color[0]);
        // Confidence collapses back toward a restart.
        assert!(
            out.confidence < 2.0,
            "confidence not reset: {}",
            out.confidence
        );
    }

    #[test]
    fn confidence_is_capped() {
        let params = ResolveParams {
            max_confidence: 8.0,
            ..ResolveParams::default()
        };
        let color = [0.3, 0.3, 0.3];
        let out = resolve(params, color, &tight_window(color), color, 100.0, false);
        assert!(out.confidence <= 8.0 + 1e-4, "{}", out.confidence);
    }

    #[test]
    fn non_finite_inputs_are_sanitized() {
        let current = [f32::NAN, 0.5, f32::INFINITY];
        let out = resolve(
            ResolveParams::default(),
            current,
            &tight_window([0.0, 0.5, 0.0]),
            [0.0, 0.5, 0.0],
            4.0,
            false,
        );
        assert!(out.color.iter().all(|c| c.is_finite() && *c >= 0.0));
    }

    #[test]
    fn params_sanitize_bad_values() {
        let p = ResolveParams {
            variance_gamma: -1.0,
            max_confidence: 0.0,
        }
        .sanitized();
        assert!(approx(p.variance_gamma, DEFAULT_VARIANCE_GAMMA));
        assert!(approx(p.max_confidence, DEFAULT_MAX_CONFIDENCE));
    }

    #[test]
    fn hdr_blend_round_trips_through_tonemap() {
        // A bright but in-gamut history and current average correctly in the
        // bounded domain and come back as a sensible linear value.
        let current = [8.0, 8.0, 8.0];
        let history = [8.0, 8.0, 8.0];
        let out = resolve(
            ResolveParams::default(),
            current,
            &tight_window(current),
            history,
            4.0,
            false,
        );
        assert!(approx(out.color[0], 8.0), "{}", out.color[0]);
    }

    #[test]
    fn lock_preserves_thin_feature_history_against_the_clamp() {
        use super::super::lock::LockState;
        // A bright one-pixel feature in history sits outside the (darker)
        // current neighborhood, so the plain resolve clamps it away. A fresh
        // lock (full thin-feature strength) must retain more of that history.
        let params = ResolveParams::default();
        let current = [0.2, 0.2, 0.2];
        let neighborhood = tight_window(current);
        let history = [0.9, 0.9, 0.9];
        let confidence = 8.0;

        let plain = resolve(params, current, &neighborhood, history, confidence, false);
        let (locked, lock) = resolve_with_lock(
            params,
            current,
            &neighborhood,
            history,
            confidence,
            false,
            LockState::default(),
            1.0,
        );

        assert!(lock.is_locked(), "a full-strength thin feature should lock");
        assert!(
            locked.color[0] > plain.color[0],
            "lock should keep more bright history: locked {} vs plain {}",
            locked.color[0],
            plain.color[0]
        );
    }

    #[test]
    fn lock_is_dropped_on_disocclusion() {
        use super::super::lock::LockState;
        // Even a strong thin feature cannot lock across a disocclusion.
        let (out, lock) = resolve_with_lock(
            ResolveParams::default(),
            [0.5, 0.5, 0.5],
            &tight_window([0.5, 0.5, 0.5]),
            [0.9, 0.9, 0.9],
            8.0,
            true,
            LockState {
                lifetime: 3.0,
                luma: 0.5,
            },
            1.0,
        );
        assert!(!lock.is_locked());
        // Disocclusion resets to the current color.
        assert!(approx(out.color[0], 0.5));
    }
}
