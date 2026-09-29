//! History-rejection heuristic: depth + normal + surface-id disocclusion.
//!
//! After [`super::reproject`] maps a pixel back to where its surface was last
//! frame, the temporal consumer must decide whether that history is actually
//! the *same surface*. When a foreground object moves it reveals background
//! that was previously hidden (a *disocclusion*): the reprojected history for
//! those newly revealed pixels belongs to the wrong surface and blending it in
//! produces ghosting. This module is the `CPU` reference for that accept /
//! reject decision.
//!
//! Three independent signals are combined, mirroring the checks production
//! `TAA` / temporal-upsampler stacks run per pixel:
//!
//! 1. **Surface identity** — if the current and reprojected samples carry
//!    different surface ids, they are provably different surfaces and the
//!    history is rejected outright. This is the strongest, cheapest signal and
//!    is why the G-buffer carries a stable per-primitive id.
//! 2. **Depth continuity** — a large relative depth gap between the current
//!    pixel and its reprojected history means the reprojection landed on a
//!    different surface along the same ray.
//! 3. **Normal continuity** — even at matching depth, a sharp normal change
//!    (a crease, a different facet) indicates a different surface.
//!
//! The output is a graded confidence in `[0, 1]` plus a hard accept / reject
//! verdict and the reasons, which feed the [`super::encode::flags::DISOCCLUDED`]
//! per-pixel flag. View-global invalidations (camera cut, resolution change,
//! streaming reveal) from [`crate::history::InvalidationMask`] override the
//! per-pixel decision and force rejection everywhere.

use super::{clamp01, encode::flags as motion_flags, EPS};
use crate::history::InvalidationMask;

/// Dot product of two 3-component vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// A single surface sample used by the disocclusion test.
///
/// `depth` is whatever monotone depth the pipeline stores (linear view depth or
/// clip-space `NDC` depth); only relative differences are compared, so the
/// exact convention does not matter as long as current and history agree.
/// `normal` is expected to be unit length in a shared space (world or view).
/// `surface_id` follows the [`super::SurfaceId`] convention where `0` means
/// "no stable id".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfacePoint {
    /// Stored depth (linear or `NDC`); only relative gaps are compared.
    pub depth: f32,
    /// Unit surface normal in a shared space.
    pub normal: [f32; 3],
    /// Stable surface identity; `0` denotes "unknown / none".
    pub surface_id: u64,
}

impl SurfacePoint {
    /// Builds a surface sample.
    #[must_use]
    pub const fn new(depth: f32, normal: [f32; 3], surface_id: u64) -> Self {
        Self {
            depth,
            normal,
            surface_id,
        }
    }
}

/// Tunable thresholds for the disocclusion test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisocclusionParams {
    /// Maximum tolerated relative depth difference before the depth signal
    /// collapses to zero confidence. `0.05` accepts a 5% depth gap.
    pub depth_relative_tolerance: f32,
    /// Minimum `cos(angle)` between current and history normals to retain any
    /// confidence. `0.9` corresponds to roughly a 25-degree crease.
    pub normal_cos_threshold: f32,
    /// When set, mismatched surface ids reject the history unconditionally.
    pub require_surface_match: bool,
    /// Combined confidence at or above which the history is accepted.
    pub accept_threshold: f32,
}

impl Default for DisocclusionParams {
    fn default() -> Self {
        Self {
            depth_relative_tolerance: 0.05,
            normal_cos_threshold: 0.9,
            require_surface_match: true,
            accept_threshold: 0.5,
        }
    }
}

impl DisocclusionParams {
    /// Builds sanitized parameters: the tolerance is floored at [`EPS`], the
    /// normal threshold is clamped to `[-1, 1)`, and the accept threshold to
    /// `[0, 1]`, so downstream math can never divide by zero or clamp against a
    /// degenerate range.
    #[must_use]
    pub fn new(
        depth_relative_tolerance: f32,
        normal_cos_threshold: f32,
        require_surface_match: bool,
        accept_threshold: f32,
    ) -> Self {
        let depth_relative_tolerance = if depth_relative_tolerance.is_nan() {
            EPS
        } else {
            depth_relative_tolerance.max(EPS)
        };
        // Keep the threshold strictly below 1 so the `1 - threshold` normal
        // remap never divides by zero.
        let normal_cos_threshold = if normal_cos_threshold.is_nan() {
            0.0
        } else {
            normal_cos_threshold.clamp(-1.0, 1.0 - EPS)
        };
        Self {
            depth_relative_tolerance,
            normal_cos_threshold,
            require_surface_match,
            accept_threshold: clamp01(accept_threshold),
        }
    }
}

/// Bit set describing *why* a history sample was rejected.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct RejectionReasons(u32);

impl RejectionReasons {
    /// No rejection reason (history is coherent).
    pub const NONE: Self = Self(0);
    /// Current and history surface ids differ.
    pub const SURFACE_MISMATCH: Self = Self(1 << 0);
    /// Relative depth gap exceeds the tolerance.
    pub const DEPTH_DISCONTINUITY: Self = Self(1 << 1);
    /// Normals diverge beyond the cosine threshold.
    pub const NORMAL_DISCONTINUITY: Self = Self(1 << 2);

    /// Raw bit representation.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns `true` when no reason bits are set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` when every bit in `other` is set in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// The outcome of a per-pixel disocclusion test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisocclusionVerdict {
    /// `true` when the reprojected history should be blended in.
    pub accepted: bool,
    /// Graded validity in `[0, 1]`; usable as a history blend weight even when
    /// `accepted` is `true` (a low-but-passing confidence blends more of the
    /// current frame).
    pub confidence: f32,
    /// Why the history was rejected, when it was.
    pub reasons: RejectionReasons,
}

/// Depth-continuity confidence: `1` at a perfect match, falling linearly to `0`
/// as the relative gap reaches `tolerance`, and `0` beyond it.
#[must_use]
pub fn depth_consistency(current_depth: f32, history_depth: f32, tolerance: f32) -> f32 {
    let tol = if tolerance.is_nan() {
        EPS
    } else {
        tolerance.max(EPS)
    };
    let denom = current_depth.abs().max(history_depth.abs()).max(EPS);
    let relative = (current_depth - history_depth).abs() / denom;
    clamp01(1.0 - relative / tol)
}

/// Normal-continuity confidence: `1` when the normals align, falling linearly
/// to `0` as their cosine drops to `cos_threshold`, and `0` below it.
#[must_use]
pub fn normal_consistency(
    current_normal: [f32; 3],
    history_normal: [f32; 3],
    cos_threshold: f32,
) -> f32 {
    let threshold = if cos_threshold.is_nan() {
        0.0
    } else {
        cos_threshold.clamp(-1.0, 1.0 - EPS)
    };
    let cos = dot3(current_normal, history_normal);
    clamp01((cos - threshold) / (1.0 - threshold))
}

/// Classifies a reprojected history sample against the current surface.
///
/// The combined confidence is the minimum of the surface, depth, and normal
/// sub-confidences (a single failing signal collapses the result), and the
/// history is accepted when that confidence reaches `params.accept_threshold`
/// and no hard reason fired.
#[must_use]
pub fn classify(
    current: SurfacePoint,
    history: SurfacePoint,
    params: DisocclusionParams,
) -> DisocclusionVerdict {
    let mut reasons = RejectionReasons::NONE;
    let mut confidence = 1.0_f32;

    if params.require_surface_match && current.surface_id != history.surface_id {
        reasons.insert(RejectionReasons::SURFACE_MISMATCH);
        confidence = 0.0;
    }

    let depth_conf = depth_consistency(
        current.depth,
        history.depth,
        params.depth_relative_tolerance,
    );
    if depth_conf <= 0.0 {
        reasons.insert(RejectionReasons::DEPTH_DISCONTINUITY);
    }
    confidence = confidence.min(depth_conf);

    let normal_conf =
        normal_consistency(current.normal, history.normal, params.normal_cos_threshold);
    if normal_conf <= 0.0 {
        reasons.insert(RejectionReasons::NORMAL_DISCONTINUITY);
    }
    confidence = confidence.min(normal_conf);

    let accepted = reasons.is_empty() && confidence >= params.accept_threshold;

    DisocclusionVerdict {
        accepted,
        confidence,
        reasons,
    }
}

/// Maps a verdict to the per-pixel motion flag byte, setting
/// [`super::encode::flags::DISOCCLUDED`] on rejection.
#[must_use]
pub fn history_flags(verdict: DisocclusionVerdict) -> u8 {
    if verdict.accepted {
        0
    } else {
        motion_flags::DISOCCLUDED
    }
}

/// The view-global invalidations that force per-pixel history rejection
/// regardless of the local depth/normal/surface test.
#[must_use]
pub fn global_rejection_mask() -> InvalidationMask {
    InvalidationMask(
        InvalidationMask::CAMERA_CUT.0
            | InvalidationMask::RESOLUTION.0
            | InvalidationMask::SCENE.0
            | InvalidationMask::STREAMING_REVEAL.0,
    )
}

/// Returns `true` when a view-global invalidation forces the entire history to
/// be rejected this frame, short-circuiting the per-pixel test.
#[must_use]
pub fn view_forces_rejection(mask: InvalidationMask) -> bool {
    mask.intersects(global_rejection_mask())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_T: f32 = 1e-6;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() <= EPS_T, "expected {b}, got {a}");
    }

    const UP: [f32; 3] = [0.0, 1.0, 0.0];
    const RIGHT: [f32; 3] = [1.0, 0.0, 0.0];

    #[test]
    fn params_default_is_sane() {
        let p = DisocclusionParams::default();
        assert!(p.require_surface_match);
        approx(p.depth_relative_tolerance, 0.05);
        approx(p.normal_cos_threshold, 0.9);
        approx(p.accept_threshold, 0.5);
    }

    #[test]
    fn params_new_sanitizes_degenerate_inputs() {
        let p = DisocclusionParams::new(f32::NAN, f32::NAN, false, 5.0);
        approx(p.depth_relative_tolerance, EPS);
        approx(p.normal_cos_threshold, 0.0);
        approx(p.accept_threshold, 1.0);
        assert!(!p.require_surface_match);
        // Threshold stays strictly below 1 to keep the normal remap finite.
        let clamped = DisocclusionParams::new(0.1, 2.0, true, -1.0);
        assert!(clamped.normal_cos_threshold < 1.0);
        approx(clamped.accept_threshold, 0.0);
    }

    #[test]
    fn depth_consistency_grades_linearly() {
        approx(depth_consistency(1.0, 1.0, 0.1), 1.0);
        // 5% gap (denominator 1.0) against a 10% tolerance => half confidence.
        approx(depth_consistency(1.0, 0.95, 0.1), 0.5);
        // Beyond tolerance clamps to zero.
        approx(depth_consistency(1.0, 2.0, 0.1), 0.0);
    }

    #[test]
    fn depth_consistency_handles_zero_depth() {
        // Both near zero: relative gap uses the EPS floor and stays finite.
        let c = depth_consistency(0.0, 0.0, 0.1);
        approx(c, 1.0);
    }

    #[test]
    fn normal_consistency_grades_linearly() {
        approx(normal_consistency(UP, UP, 0.5), 1.0);
        // Orthogonal normals (cos 0) against threshold 0.5 => below, zero.
        approx(normal_consistency(UP, RIGHT, 0.5), 0.0);
        // cos 0.75 halfway between threshold 0.5 and 1.0 => 0.5.
        let half = [0.75, 0.661_437_8, 0.0];
        let c = normal_consistency([1.0, 0.0, 0.0], half, 0.5);
        assert!((c - 0.5).abs() < 1e-3);
    }

    #[test]
    fn classify_accepts_coherent_surface() {
        let p = DisocclusionParams::default();
        let current = SurfacePoint::new(1.0, UP, 42);
        let history = SurfacePoint::new(1.0, UP, 42);
        let v = classify(current, history, p);
        assert!(v.accepted);
        approx(v.confidence, 1.0);
        assert!(v.reasons.is_empty());
        assert_eq!(history_flags(v), 0);
    }

    #[test]
    fn classify_rejects_surface_mismatch() {
        let p = DisocclusionParams::default();
        let current = SurfacePoint::new(1.0, UP, 7);
        let history = SurfacePoint::new(1.0, UP, 8);
        let v = classify(current, history, p);
        assert!(!v.accepted);
        approx(v.confidence, 0.0);
        assert!(v.reasons.contains(RejectionReasons::SURFACE_MISMATCH));
        assert_eq!(history_flags(v), motion_flags::DISOCCLUDED);
    }

    #[test]
    fn classify_ignores_surface_when_disabled() {
        let p = DisocclusionParams::new(0.05, 0.9, false, 0.5);
        let current = SurfacePoint::new(1.0, UP, 7);
        let history = SurfacePoint::new(1.0, UP, 8);
        let v = classify(current, history, p);
        assert!(v.accepted);
        assert!(!v.reasons.contains(RejectionReasons::SURFACE_MISMATCH));
    }

    #[test]
    fn classify_rejects_depth_discontinuity() {
        let p = DisocclusionParams::default();
        let current = SurfacePoint::new(1.0, UP, 1);
        let history = SurfacePoint::new(5.0, UP, 1);
        let v = classify(current, history, p);
        assert!(!v.accepted);
        assert!(v.reasons.contains(RejectionReasons::DEPTH_DISCONTINUITY));
    }

    #[test]
    fn classify_rejects_normal_discontinuity() {
        let p = DisocclusionParams::default();
        let current = SurfacePoint::new(1.0, UP, 1);
        let history = SurfacePoint::new(1.0, RIGHT, 1);
        let v = classify(current, history, p);
        assert!(!v.accepted);
        assert!(v.reasons.contains(RejectionReasons::NORMAL_DISCONTINUITY));
    }

    #[test]
    fn view_forces_rejection_on_global_events() {
        assert!(view_forces_rejection(InvalidationMask::CAMERA_CUT));
        assert!(view_forces_rejection(InvalidationMask::STREAMING_REVEAL));
        assert!(view_forces_rejection(InvalidationMask::RESOLUTION));
        assert!(view_forces_rejection(InvalidationMask::SCENE));
        // Exposure change alone does not invalidate reprojected geometry.
        assert!(!view_forces_rejection(InvalidationMask::EXPOSURE));
    }

    #[test]
    fn rejection_reasons_bit_ops() {
        let mut r = RejectionReasons::NONE;
        assert!(r.is_empty());
        r.insert(RejectionReasons::DEPTH_DISCONTINUITY);
        r.insert(RejectionReasons::NORMAL_DISCONTINUITY);
        assert!(r.contains(RejectionReasons::DEPTH_DISCONTINUITY));
        assert!(r.contains(RejectionReasons::NORMAL_DISCONTINUITY));
        assert!(!r.contains(RejectionReasons::SURFACE_MISMATCH));
        assert_ne!(r.bits(), 0);
    }
}
