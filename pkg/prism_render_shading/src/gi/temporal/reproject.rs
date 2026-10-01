//! Motion-vector reprojection and disocclusion detection — CPU golden.
//!
//! Temporal resolve (TAA and temporal GI accumulation) reuses last frame's
//! shaded result by *reprojecting* each current-frame pixel back into the
//! history buffer along its screen-space motion vector, then validating that
//! the reprojected sample actually corresponds to the same surface.  When the
//! surface was hidden last frame (a *disocclusion*) or the reprojected
//! coordinate falls outside the frame, history must be rejected to avoid
//! dragging stale colour across silhouettes ("ghosting").
//!
//! This module is the backend-neutral reference for that geometry:
//!
//! * [`reproject_uv`] / [`reproject`] — map a current-frame UV to its history
//!   UV via a UV-space motion vector, with bounds classification.
//! * [`SurfaceSample`] + [`is_disoccluded`] — depth / normal / velocity
//!   consistency tests that gate history reuse.
//! * [`bilinear_weights`] + [`sample_bilinear`] / [`sample_nearest`] —
//!   clamp-to-edge history fetch封装 built on a caller-supplied texel reader,
//!   so the sampling math is tested independently of any texture backend.
//!
//! # Conventions
//! * UVs are in `[0, 1]^2` with the origin at the top-left texel edge; integer
//!   texel coordinates index `[0, size-1]`.  A UV `u` maps to the continuous
//!   texel coordinate `u * size - 0.5` (texel *centers* sit at half-integers),
//!   matching the GPU twin's `textureSample` addressing.
//! * The motion vector `velocity` is the screen-space displacement, in UV
//!   units, that the surface underfoot moved **from the previous frame to the
//!   current frame**.  History therefore lives at `current_uv - velocity`.
//! * Every helper is deterministic and allocation-free (no RNG/IO/GPU/unsafe)
//!   and defends against degenerate inputs: non-finite UVs and velocities fall
//!   back to the identity reprojection, sizes are floored to at least one
//!   texel, and no path can emit a NaN.

use bevy_math::{ops, IVec2, Vec2, Vec3};

/// Smallest texture extent (per axis) the sampler will address; guards against
/// divide-by-zero when a caller passes a zero or negative size.
const MIN_TEXTURE_SIZE: f32 = 1.0;

/// Sanitizes a UV, replacing any non-finite component with `0.5` (frame center)
/// so downstream arithmetic can never propagate a NaN.
#[inline]
fn sanitize_uv(uv: Vec2) -> Vec2 {
    Vec2::new(
        if uv.x.is_finite() { uv.x } else { 0.5 },
        if uv.y.is_finite() { uv.y } else { 0.5 },
    )
}

/// Sanitizes a velocity, replacing any non-finite component with `0.0` so a
/// bad motion vector degrades to the identity reprojection.
#[inline]
fn sanitize_velocity(v: Vec2) -> Vec2 {
    Vec2::new(
        if v.x.is_finite() { v.x } else { 0.0 },
        if v.y.is_finite() { v.y } else { 0.0 },
    )
}

/// Clamps a texture size to be at least one texel per axis and finite.
#[inline]
fn sanitize_size(size: Vec2) -> Vec2 {
    Vec2::new(
        if size.x.is_finite() {
            size.x.max(MIN_TEXTURE_SIZE)
        } else {
            MIN_TEXTURE_SIZE
        },
        if size.y.is_finite() {
            size.y.max(MIN_TEXTURE_SIZE)
        } else {
            MIN_TEXTURE_SIZE
        },
    )
}

/// Reprojects a current-frame UV into the history buffer along a UV-space
/// motion vector.
///
/// The surface moved by `velocity` (in UV units) from the previous to the
/// current frame, so its history position is `current_uv - velocity`.  The
/// result is **not** clamped to the frame; use [`uv_in_bounds`] or [`reproject`]
/// to classify out-of-frame reprojections.
#[inline]
pub fn reproject_uv(current_uv: Vec2, velocity: Vec2) -> Vec2 {
    sanitize_uv(current_uv) - sanitize_velocity(velocity)
}

/// Returns `true` when `uv` lies inside the unit square `[0, 1]^2` (inclusive),
/// i.e. the history fetch would land on real data rather than off-frame.
#[inline]
pub fn uv_in_bounds(uv: Vec2) -> bool {
    uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0
}

/// Clamps a UV to the unit square so clamp-to-edge addressing reads the border
/// texel instead of wrapping or reading garbage.
#[inline]
pub fn clamp_uv(uv: Vec2) -> Vec2 {
    let uv = sanitize_uv(uv);
    Vec2::new(uv.x.clamp(0.0, 1.0), uv.y.clamp(0.0, 1.0))
}

/// The outcome of a reprojection: where history lives and whether that location
/// is inside the frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reprojection {
    /// History UV = `current_uv - velocity` (unclamped).
    pub history_uv: Vec2,
    /// `true` when [`history_uv`](Reprojection::history_uv) is within `[0,1]^2`.
    pub in_bounds: bool,
}

/// Reprojects a UV and classifies whether the result stays inside the frame.
///
/// Combines [`reproject_uv`] with an [`uv_in_bounds`] test so callers get the
/// history coordinate and its validity in one deterministic step.
#[inline]
pub fn reproject(current_uv: Vec2, velocity: Vec2) -> Reprojection {
    let history_uv = reproject_uv(current_uv, velocity);
    Reprojection {
        history_uv,
        in_bounds: uv_in_bounds(history_uv),
    }
}

/// A minimal per-pixel G-buffer sample used for disocclusion testing.
///
/// Stores the three quantities whose discontinuities mark a surface change:
/// view/linear depth, world- or view-space normal, and the UV-space motion
/// vector.  The normal is stored raw and re-normalized on use so a slightly
/// non-unit input cannot skew the dot-product test.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSample {
    /// Linear (view-space) depth; expected positive, larger = farther.
    pub depth: f32,
    /// Surface normal; need not be unit length (re-normalized on use).
    pub normal: Vec3,
    /// UV-space motion vector of this sample.
    pub velocity: Vec2,
}

impl SurfaceSample {
    /// Builds a sample, leaving fields untouched (sanitation happens at test
    /// time so the struct stays a plain data carrier).
    #[inline]
    pub const fn new(depth: f32, normal: Vec3, velocity: Vec2) -> Self {
        Self {
            depth,
            normal,
            velocity,
        }
    }
}

/// Thresholds that decide when history must be rejected as disoccluded.
///
/// Each threshold is compared against a normalized discrepancy, so the same
/// struct works across depth ranges and resolutions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisocclusionParams {
    /// Max allowed **relative** depth difference `|d0-d1| / max(d0,d1,eps)`.
    pub depth_relative_threshold: f32,
    /// Min allowed `dot(n0, n1)` between unit normals (e.g. `cos(30°)`).
    pub normal_cos_threshold: f32,
    /// Max allowed motion-vector difference length, in UV units.
    pub velocity_threshold: f32,
}

impl Default for DisocclusionParams {
    fn default() -> Self {
        // Defaults tuned for 1080p-ish temporal reuse: ~5% depth tolerance,
        // ~25° normal cone, and a motion tolerance of a couple of texels.
        Self {
            depth_relative_threshold: 0.05,
            normal_cos_threshold: 0.906_307_8, // cos(25°)
            velocity_threshold: 2.0e-3,
        }
    }
}

/// Returns `true` when two depths agree to within a relative tolerance.
///
/// Uses `|d0 - d1| / max(|d0|, |d1|, eps)` so the test is scale-invariant and
/// robust near the camera.  Non-finite depths are treated as inconsistent.
#[inline]
pub fn depth_consistent(d0: f32, d1: f32, relative_threshold: f32) -> bool {
    if !d0.is_finite() || !d1.is_finite() {
        return false;
    }
    let denom = d0.abs().max(d1.abs()).max(1.0e-6);
    (d0 - d1).abs() / denom <= relative_threshold.max(0.0)
}

/// Returns `true` when two normals point within the given cosine cone.
///
/// Both normals are re-normalized; a degenerate (zero-length) normal fails the
/// test rather than producing a NaN.
#[inline]
pub fn normal_consistent(n0: Vec3, n1: Vec3, cos_threshold: f32) -> bool {
    let l0 = n0.length();
    let l1 = n1.length();
    if !(l0 > 1.0e-8 && l1 > 1.0e-8) {
        return false;
    }
    let c = (n0 / l0).dot(n1 / l1);
    if !c.is_finite() {
        return false;
    }
    c >= cos_threshold.clamp(-1.0, 1.0)
}

/// Returns `true` when two motion vectors agree to within `threshold` (UV
/// units).  A large velocity delta usually signals a different surface slid
/// under the pixel even if depth and normal happen to match.
#[inline]
pub fn velocity_consistent(v0: Vec2, v1: Vec2, threshold: f32) -> bool {
    let d = sanitize_velocity(v0) - sanitize_velocity(v1);
    let len = d.length();
    len.is_finite() && len <= threshold.max(0.0)
}

/// Full disocclusion test: history is rejected when depth, normal, **or**
/// velocity disagree beyond their thresholds.
///
/// Returns `true` when the pixel is considered *disoccluded* (history invalid).
#[inline]
pub fn is_disoccluded(
    current: SurfaceSample,
    history: SurfaceSample,
    params: DisocclusionParams,
) -> bool {
    !(depth_consistent(current.depth, history.depth, params.depth_relative_threshold)
        && normal_consistent(current.normal, history.normal, params.normal_cos_threshold)
        && velocity_consistent(
            current.velocity,
            history.velocity,
            params.velocity_threshold,
        ))
}

/// A continuous confidence in `[0, 1]` that history is still valid, blending
/// the three disocclusion signals instead of a hard accept/reject.
///
/// `1.0` means perfectly consistent; `0.0` means fully disoccluded.  Each
/// factor decays linearly from `1` at zero discrepancy to `0` at its threshold,
/// and the result is their product so any single strong mismatch kills reuse.
#[inline]
pub fn history_confidence(
    current: SurfaceSample,
    history: SurfaceSample,
    params: DisocclusionParams,
) -> f32 {
    // Depth factor.
    let depth_f = if current.depth.is_finite() && history.depth.is_finite() {
        let denom = current.depth.abs().max(history.depth.abs()).max(1.0e-6);
        let rel = (current.depth - history.depth).abs() / denom;
        1.0 - (rel / params.depth_relative_threshold.max(1.0e-6)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Normal factor.
    let l0 = current.normal.length();
    let l1 = history.normal.length();
    let normal_f = if l0 > 1.0e-8 && l1 > 1.0e-8 {
        let c = (current.normal / l0).dot(history.normal / l1).clamp(-1.0, 1.0);
        let thr = params.normal_cos_threshold.clamp(-1.0, 1.0);
        ((c - thr) / (1.0 - thr).max(1.0e-6)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Velocity factor.
    let dv = (sanitize_velocity(current.velocity) - sanitize_velocity(history.velocity)).length();
    let vel_f = 1.0 - (dv / params.velocity_threshold.max(1.0e-6)).clamp(0.0, 1.0);

    (depth_f * normal_f * vel_f).clamp(0.0, 1.0)
}

/// Precomputed bilinear addressing for a UV: the top-left integer texel, the
/// fractional offset within the `2x2` footprint, and the four corner weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilinearWeights {
    /// Top-left integer texel of the `2x2` footprint (clamp-to-edge applied by
    /// the sampler, not here, so this may legally be `-1` or `size-1`).
    pub base: IVec2,
    /// Fractional position within the footprint, each component in `[0, 1)`.
    pub frac: Vec2,
    /// Corner weights for `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)`; sum to `1`.
    pub weights: [f32; 4],
}

/// Computes bilinear addressing for `uv` against a texture of size `size`.
///
/// The continuous texel coordinate is `uv * size - 0.5`; its floor is the
/// top-left texel and its fraction drives the four separable corner weights.
#[inline]
pub fn bilinear_weights(uv: Vec2, size: Vec2) -> BilinearWeights {
    let uv = sanitize_uv(uv);
    let size = sanitize_size(size);
    let coord = uv * size - Vec2::splat(0.5);
    let fx = ops::floor(coord.x);
    let fy = ops::floor(coord.y);
    let frac = Vec2::new((coord.x - fx).clamp(0.0, 1.0), (coord.y - fy).clamp(0.0, 1.0));
    let base = IVec2::new(fx as i32, fy as i32);
    let (tx, ty) = (frac.x, frac.y);
    let weights = [
        (1.0 - tx) * (1.0 - ty),
        tx * (1.0 - ty),
        (1.0 - tx) * ty,
        tx * ty,
    ];
    BilinearWeights {
        base,
        frac,
        weights,
    }
}

/// Clamp-to-edge integer texel addressing: maps an arbitrary `coord` to a valid
/// texel in `[0, size-1]`.
#[inline]
fn clamp_texel(coord: IVec2, size: Vec2) -> IVec2 {
    let max_x = (size.x as i32 - 1).max(0);
    let max_y = (size.y as i32 - 1).max(0);
    IVec2::new(coord.x.clamp(0, max_x), coord.y.clamp(0, max_y))
}

/// Bilinearly samples a history texture through a caller-supplied texel reader.
///
/// `fetch` returns the stored value at an integer texel; this helper applies
/// clamp-to-edge addressing and the four [`bilinear_weights`] so the sampling
/// math can be unit-tested without any GPU texture backend.
#[inline]
pub fn sample_bilinear<F>(uv: Vec2, size: Vec2, mut fetch: F) -> Vec3
where
    F: FnMut(IVec2) -> Vec3,
{
    let bw = bilinear_weights(uv, size);
    let size = sanitize_size(size);
    let c00 = fetch(clamp_texel(bw.base, size));
    let c10 = fetch(clamp_texel(bw.base + IVec2::new(1, 0), size));
    let c01 = fetch(clamp_texel(bw.base + IVec2::new(0, 1), size));
    let c11 = fetch(clamp_texel(bw.base + IVec2::new(1, 1), size));
    c00 * bw.weights[0] + c10 * bw.weights[1] + c01 * bw.weights[2] + c11 * bw.weights[3]
}

/// Nearest-neighbour sample through a caller-supplied texel reader, with
/// clamp-to-edge addressing.  Rounds the continuous texel coordinate to the
/// closest texel center.
#[inline]
pub fn sample_nearest<F>(uv: Vec2, size: Vec2, mut fetch: F) -> Vec3
where
    F: FnMut(IVec2) -> Vec3,
{
    let uv = sanitize_uv(uv);
    let size = sanitize_size(size);
    let coord = uv * size - Vec2::splat(0.5);
    let texel = IVec2::new(
        ops::floor(coord.x + 0.5) as i32,
        ops::floor(coord.y + 0.5) as i32,
    );
    fetch(clamp_texel(texel, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn reproject_subtracts_velocity() {
        let uv = Vec2::new(0.5, 0.5);
        let vel = Vec2::new(0.1, -0.2);
        let r = reproject(uv, vel);
        assert!((r.history_uv - Vec2::new(0.4, 0.7)).length() < EPS);
        assert!(r.in_bounds);
    }

    #[test]
    fn reproject_out_of_bounds_flagged() {
        let r = reproject(Vec2::new(0.05, 0.5), Vec2::new(0.2, 0.0));
        assert!(r.history_uv.x < 0.0);
        assert!(!r.in_bounds);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let r = reproject(Vec2::new(f32::NAN, 0.3), Vec2::new(0.0, f32::INFINITY));
        assert!(r.history_uv.x.is_finite());
        assert!(r.history_uv.y.is_finite());
        // NaN uv -> 0.5 center, inf velocity -> 0.0.
        assert!((r.history_uv.x - 0.5).abs() < EPS);
        assert!((r.history_uv.y - 0.3).abs() < EPS);
    }

    #[test]
    fn clamp_uv_edges() {
        assert_eq!(clamp_uv(Vec2::new(-1.0, 2.0)), Vec2::new(0.0, 1.0));
    }

    #[test]
    fn depth_threshold_behaviour() {
        assert!(depth_consistent(10.0, 10.2, 0.05));
        assert!(!depth_consistent(10.0, 12.0, 0.05));
        assert!(!depth_consistent(f32::NAN, 10.0, 0.05));
    }

    #[test]
    fn normal_cone_behaviour() {
        let n = Vec3::Z;
        // 10 degrees apart -> consistent under cos(25).
        let tilt = Vec3::new(0.173_648, 0.0, 0.984_807_8);
        assert!(normal_consistent(n, tilt, 0.906_307_8));
        // Opposite normals -> inconsistent.
        assert!(!normal_consistent(n, -n, 0.906_307_8));
        // Zero normal -> inconsistent, no NaN.
        assert!(!normal_consistent(n, Vec3::ZERO, 0.5));
    }

    #[test]
    fn velocity_threshold_behaviour() {
        assert!(velocity_consistent(
            Vec2::new(0.001, 0.0),
            Vec2::new(0.0015, 0.0),
            2.0e-3
        ));
        assert!(!velocity_consistent(
            Vec2::new(0.0, 0.0),
            Vec2::new(0.1, 0.0),
            2.0e-3
        ));
    }

    #[test]
    fn disocclusion_rejects_depth_jump() {
        let p = DisocclusionParams::default();
        let cur = SurfaceSample::new(5.0, Vec3::Z, Vec2::ZERO);
        let same = SurfaceSample::new(5.01, Vec3::Z, Vec2::ZERO);
        let far = SurfaceSample::new(50.0, Vec3::Z, Vec2::ZERO);
        assert!(!is_disoccluded(cur, same, p));
        assert!(is_disoccluded(cur, far, p));
    }

    #[test]
    fn confidence_is_monotone_and_bounded() {
        let p = DisocclusionParams::default();
        let cur = SurfaceSample::new(5.0, Vec3::Z, Vec2::ZERO);
        let perfect = history_confidence(cur, cur, p);
        let worse = history_confidence(
            cur,
            SurfaceSample::new(5.2, Vec3::Z, Vec2::ZERO),
            p,
        );
        assert!(perfect >= worse);
        assert!((0.0..=1.0).contains(&perfect));
        assert!((0.0..=1.0).contains(&worse));
    }

    #[test]
    fn bilinear_weights_sum_to_one() {
        let bw = bilinear_weights(Vec2::new(0.37, 0.62), Vec2::new(16.0, 16.0));
        let s: f32 = bw.weights.iter().sum();
        assert!((s - 1.0).abs() < EPS);
    }

    #[test]
    fn bilinear_samples_constant_field() {
        // A constant texture must bilinearly resolve to that constant.
        let size = Vec2::new(8.0, 8.0);
        let got = sample_bilinear(Vec2::new(0.333, 0.777), size, |_| Vec3::splat(2.5));
        assert!((got - Vec3::splat(2.5)).length() < EPS);
    }

    #[test]
    fn bilinear_interpolates_ramp() {
        // Horizontal ramp texel.x -> value; sample at a texel center recovers it.
        let size = Vec2::new(4.0, 4.0);
        let fetch = |t: IVec2| Vec3::splat(t.x as f32);
        // UV for center of texel x=2 is (2+0.5)/4 = 0.625.
        let got = sample_bilinear(Vec2::new(0.625, 0.5), size, fetch);
        assert!((got.x - 2.0).abs() < 1.0e-4);
    }

    #[test]
    fn nearest_clamps_to_edge() {
        let size = Vec2::new(4.0, 4.0);
        // Far off-frame UV must clamp to a valid border texel (value 3).
        let got = sample_nearest(Vec2::new(5.0, 0.5), size, |t: IVec2| Vec3::splat(t.x as f32));
        assert!((got.x - 3.0).abs() < EPS);
    }

    #[test]
    fn deterministic() {
        let a = reproject(Vec2::new(0.4, 0.6), Vec2::new(0.01, 0.02));
        let b = reproject(Vec2::new(0.4, 0.6), Vec2::new(0.01, 0.02));
        assert_eq!(a, b);
    }
}
