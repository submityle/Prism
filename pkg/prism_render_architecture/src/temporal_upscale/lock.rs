//! Thin-feature locks for temporal reconstruction.
//!
//! The neighborhood clamp in [`reconstruct`](super::reconstruct) is what kills
//! ghosting, but it has a well-known failure mode: a feature only one output
//! pixel wide — a power line, a railing, a specular glint — barely registers in
//! the current frame's low-resolution neighborhood, so the variance box is too
//! narrow to contain the detail accumulated in history and the clamp eats it.
//! The result is a thin feature that shimmers or disappears under motion.
//!
//! `FSR2`'s answer is a *lock*: when a pixel is detected to be a thin
//! luminance feature, the reconstructor "locks" it for a few frames, trusting
//! its accumulated history and suppressing the color rejection that would
//! otherwise clamp the detail away. The lock is dropped the moment the pixel is
//! disoccluded or its shading changes (so a lock can never cause a ghost), and
//! it decays on its own after a fixed lifetime.
//!
//! This module owns the `CPU` golden for that lifecycle: the thin-feature
//! detector, the per-pixel [`LockState`] carried across frames, its
//! advance/decay/break rules, and the trust factor the resolve multiplies into
//! its rejection term. Every operation is `+`, `-`, `*`, `/`, `min`/`max`, and
//! `abs`, so a `GPU` kernel reproduces it bit-for-bit.

use super::color::luminance;

/// Lifetime, in frames, granted to a freshly created lock.
///
/// A new lock protects its pixel for this many frames before expiring; it is a
/// small count so a lock that is no longer justified (the feature moved on)
/// clears quickly rather than smearing. Four frames is a typical `FSR2`-class
/// trade between thin-feature stability and responsiveness.
pub const INITIAL_LOCK_LIFETIME: f32 = 4.0;

/// Minimum thin-feature strength that creates a new lock.
///
/// The detector returns a `[0, 1]` strength; below this threshold the local
/// contrast is treated as ordinary texture rather than a one-pixel feature, so
/// no lock is created and the normal clamp applies.
pub const LOCK_CREATION_THRESHOLD: f32 = 0.25;

/// Fractional luma change that breaks an existing lock.
///
/// A lock means "this pixel's history is trustworthy." If the pixel's luma
/// departs from the luma captured when it locked by more than this fraction,
/// the shading genuinely changed and the stored history is stale, so the lock
/// is dropped and the normal rejection resumes. Keeping it relative makes the
/// test scale-invariant across the `HDR` range.
pub const LOCK_BREAK_LUMA_TOLERANCE: f32 = 0.25;

/// Per-pixel lock carried in the history buffer between frames.
///
/// A `lifetime` of `0` (or less) means the pixel is unlocked. A positive
/// `lifetime` is the number of frames the lock still has to live, and `luma` is
/// the Rec. 709 luminance captured when the lock was created, used to detect a
/// shading change that should break it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LockState {
    /// Remaining lock lifetime in frames; `<= 0` means unlocked.
    pub lifetime: f32,
    /// Luminance captured at lock creation, for the shading-change test.
    pub luma: f32,
}

impl Default for LockState {
    /// The unlocked state: no remaining lifetime and no captured luma.
    fn default() -> Self {
        Self {
            lifetime: 0.0,
            luma: 0.0,
        }
    }
}

impl LockState {
    /// Whether this pixel currently holds a live lock.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.lifetime > 0.0
    }
}

/// Detects how strongly the center pixel is a one-pixel luminance feature.
///
/// A thin feature is a local luminance extremum along *both* screen axes: a
/// bright center sitting above both its vertical and both its horizontal
/// neighbors (a bright line or dot), or a dark center sitting below all of
/// them. The returned strength in `[0, 1]` is the smaller of the vertical and
/// horizontal separations from the nearest neighbor on the "feature" side,
/// normalized by the full local luma range, so a crisp isolated line scores
/// near `1` and a gentle gradient scores `0`. A center that is not an extremum
/// on both axes is not a thin feature and scores `0`.
///
/// `center`, `north`, `south`, `west`, `east` are the linear `RGB` colors of
/// the cross around the pixel (the same pattern the sharpener uses).
#[must_use]
pub fn thin_feature_strength(
    center: [f32; 3],
    north: [f32; 3],
    south: [f32; 3],
    west: [f32; 3],
    east: [f32; 3],
) -> f32 {
    let c = luminance(center);
    let n = luminance(north);
    let s = luminance(south);
    let w = luminance(west);
    let e = luminance(east);

    let v_max = n.max(s);
    let v_min = n.min(s);
    let h_max = w.max(e);
    let h_min = w.min(e);

    let ring_max = v_max.max(h_max);
    let ring_min = v_min.min(h_min);
    // Full local luma range including the center; guard the normalizer so a
    // flat cross has no feature.
    let range = ring_max.max(c) - ring_min.min(c);
    if range.is_nan() || range <= 0.0 {
        return 0.0;
    }

    // Bright feature: center above both axis maxima. Dark feature: center below
    // both axis minima. The separation is the weaker of the two axes.
    let bright = (c - v_max).min(c - h_max);
    let dark = (v_min - c).min(h_min - c);
    let separation = bright.max(dark);
    if separation <= 0.0 {
        return 0.0;
    }
    (separation / range).min(1.0)
}

/// Advances a pixel's lock by one frame and returns its new state.
///
/// The rules, in order:
///
/// 1. A disocclusion invalidates everything about the pixel's past, so any lock
///    is dropped immediately (returns the unlocked [`LockState::default`]).
/// 2. An existing lock whose captured luma no longer matches `current_luma`
///    (beyond [`LOCK_BREAK_LUMA_TOLERANCE`], relative) has gone stale and is
///    dropped.
/// 3. A surviving lock loses one frame of lifetime; when it reaches zero it is
///    unlocked.
/// 4. Independently, a strong enough thin feature (`thin_strength` at or above
///    [`LOCK_CREATION_THRESHOLD`]) creates or refreshes the lock to the full
///    [`INITIAL_LOCK_LIFETIME`], capturing the current luma. A fresh lock
///    always wins over a decaying one so a persistent feature stays protected.
#[must_use]
pub fn advance_lock(
    previous: LockState,
    current_luma: f32,
    thin_strength: f32,
    disoccluded: bool,
) -> LockState {
    if disoccluded {
        return LockState::default();
    }

    // Decay / break an existing lock.
    let mut decayed = LockState::default();
    if previous.is_locked() {
        let drift = (current_luma - previous.luma).abs();
        let tolerance = LOCK_BREAK_LUMA_TOLERANCE * (previous.luma.abs() + 1e-3);
        if drift <= tolerance {
            let lifetime = previous.lifetime - 1.0;
            if lifetime > 0.0 {
                decayed = LockState {
                    lifetime,
                    luma: previous.luma,
                };
            }
        }
    }

    // Create / refresh a lock for a detected thin feature; it supersedes the
    // decayed state.
    if thin_strength >= LOCK_CREATION_THRESHOLD {
        return LockState {
            lifetime: INITIAL_LOCK_LIFETIME,
            luma: current_luma,
        };
    }

    decayed
}

/// The fraction of the normal color rejection to keep for a locked pixel.
///
/// A fully fresh lock suppresses the rejection almost entirely so the clamp
/// cannot eat the thin feature's history; as the lock decays toward expiry the
/// suppression eases back so the pixel returns smoothly to normal clamping
/// rather than popping. The result is in `[0, 1]`: `1` for an unlocked pixel
/// (rejection unchanged), approaching `0` for a freshly created lock. The
/// caller multiplies its rejection term by this value.
#[must_use]
pub fn rejection_scale(lock: LockState) -> f32 {
    if !lock.is_locked() {
        return 1.0;
    }
    // Normalized remaining life in [0, 1]; more life -> stronger suppression.
    let life = (lock.lifetime / INITIAL_LOCK_LIFETIME).clamp(0.0, 1.0);
    // Keep at least this fraction of rejection even at full lock so a genuine
    // disocclusion-free ghost can still eventually be corrected.
    const MIN_SCALE: f32 = 0.05;
    1.0 - life * (1.0 - MIN_SCALE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-error tolerance for the value checks.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    /// A gray color of a given intensity.
    fn gray(v: f32) -> [f32; 3] {
        [v, v, v]
    }

    #[test]
    fn flat_cross_has_no_feature() {
        let s = thin_feature_strength(gray(0.5), gray(0.5), gray(0.5), gray(0.5), gray(0.5));
        assert!(approx(s, 0.0));
    }

    #[test]
    fn bright_line_is_a_strong_feature() {
        // Center far above all four neighbors: a crisp bright thin feature.
        let s = thin_feature_strength(gray(1.0), gray(0.0), gray(0.0), gray(0.0), gray(0.0));
        assert!(s > 0.9, "expected near-max strength, got {s}");
    }

    #[test]
    fn dark_line_is_a_strong_feature() {
        let s = thin_feature_strength(gray(0.0), gray(1.0), gray(1.0), gray(1.0), gray(1.0));
        assert!(s > 0.9, "expected near-max strength, got {s}");
    }

    #[test]
    fn edge_is_not_a_thin_feature() {
        // Center matches one axis side (a step edge, not an isolated line): it
        // is not an extremum on both axes, so no lock.
        let s = thin_feature_strength(gray(1.0), gray(1.0), gray(0.0), gray(1.0), gray(0.0));
        assert!(approx(s, 0.0), "edge should not lock, got {s}");
    }

    #[test]
    fn disocclusion_clears_any_lock() {
        let prev = LockState {
            lifetime: 3.0,
            luma: 0.8,
        };
        let out = advance_lock(prev, 0.8, 0.0, true);
        assert!(!out.is_locked());
    }

    #[test]
    fn thin_feature_creates_full_lifetime_lock() {
        let out = advance_lock(LockState::default(), 0.9, 1.0, false);
        assert!(out.is_locked());
        assert!(approx(out.lifetime, INITIAL_LOCK_LIFETIME));
        assert!(approx(out.luma, 0.9));
    }

    #[test]
    fn lock_decays_by_one_frame_when_stable() {
        let prev = LockState {
            lifetime: 3.0,
            luma: 0.5,
        };
        let out = advance_lock(prev, 0.5, 0.0, false);
        assert!(approx(out.lifetime, 2.0));
        assert!(approx(out.luma, 0.5));
    }

    #[test]
    fn lock_expires_after_its_lifetime() {
        let prev = LockState {
            lifetime: 1.0,
            luma: 0.5,
        };
        let out = advance_lock(prev, 0.5, 0.0, false);
        assert!(!out.is_locked());
    }

    #[test]
    fn shading_change_breaks_the_lock() {
        let prev = LockState {
            lifetime: 3.0,
            luma: 0.5,
        };
        // Luma jumps far beyond the relative tolerance: stale history.
        let out = advance_lock(prev, 2.0, 0.0, false);
        assert!(!out.is_locked());
    }

    #[test]
    fn rejection_scale_is_identity_when_unlocked() {
        assert!(approx(rejection_scale(LockState::default()), 1.0));
    }

    #[test]
    fn fresh_lock_strongly_suppresses_rejection() {
        let lock = LockState {
            lifetime: INITIAL_LOCK_LIFETIME,
            luma: 0.5,
        };
        let scale = rejection_scale(lock);
        assert!(
            scale < 0.1,
            "fresh lock should nearly zero rejection: {scale}"
        );
    }

    #[test]
    fn rejection_scale_rises_as_lock_decays() {
        let young = rejection_scale(LockState {
            lifetime: INITIAL_LOCK_LIFETIME,
            luma: 0.5,
        });
        let old = rejection_scale(LockState {
            lifetime: 1.0,
            luma: 0.5,
        });
        assert!(
            old > young,
            "decaying lock should restore rejection: {old} vs {young}"
        );
    }
}
