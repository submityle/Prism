//! ReBLUR / ReLAX-style spatio-temporal bilateral denoising — CPU golden.
//!
//! The GI ray budget (1–2 spp after ReSTIR) is far too sparse to display
//! directly: the raw per-pixel radiance is a high-variance Monte-Carlo
//! estimate riddled with noise and fireflies.  Production real-time GI closes
//! that gap with a *spatio-temporal denoiser* in the spirit of NVIDIA's NRD
//! (ReBLUR / ReLAX) and SVGF / A-SVGF:
//!
//! 1. **Temporal accumulation** — reproject the previous frame's filtered
//!    history into the current frame and blend it with the new noisy sample
//!    using an exponential moving average.  The blend weight adapts to the
//!    accumulated sample count (fast convergence right after disocclusion,
//!    slow drift once converged) and to a *history clamp* that rejects stale
//!    history when the signal has changed, trading a little bias for a large
//!    variance reduction without ghosting.
//! 2. **Spatial à-trous filtering** — a few edge-stopping wavelet passes with
//!    geometrically increasing tap spacing approximate a wide bilateral blur
//!    at a fraction of the cost.  Edge-stopping weights built from depth
//!    (plane distance), normal, roughness, and luminance / variance keep the
//!    blur from bleeding across geometry, material, or lighting discontinuities
//!    while still cleaning flat, noisy regions.
//!
//! This module is the backend-neutral, deterministic reference the WESL/GPU
//! denoiser twin must reproduce under real-device parity.  It composes the
//! sibling [`super::variance`] (Welford / temporal moments for variance
//! guidance) and [`super::firefly`] (Rec.709 luminance) primitives rather than
//! re-deriving them.
//!
//! # Conventions
//! * All radiance is linear RGB stored as `f32` to match the GPU twin.
//! * Diffuse and specular are denoised *separately* in production; the kernels
//!   here are signal-agnostic and parameterised by their edge-stopping
//!   sensitivities ([`EdgeStoppingParams`]) so the same code drives both, with
//!   specular passing a tighter roughness/normal sensitivity and (optionally)
//!   a virtual-reprojected position via [`specular_virtual_position`].
//! * Weights are always finite and non-negative; degenerate inputs
//!   (zero-length normals, non-finite depths, empty neighbourhoods) fall back
//!   to a safe identity (keep the centre sample) rather than emitting `NaN`.
//! * Transcendentals go through [`bevy_math::ops`] (never `f32::exp`), matching
//!   the crate-wide no-`std` numerical contract.
//! * Every function is pure and side-effect free (no RNG / IO / GPU / global
//!   state); randomness, if any, is supplied by the caller.

use bevy_math::{ops, Vec3};

use super::firefly::luminance;
use super::variance::TemporalMoments;

/// A single denoiser tap: filtered radiance plus the geometry / material
/// attributes used to build its edge-stopping weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseSample {
    /// Linear RGB radiance of the tap.
    pub color: Vec3,
    /// World-space position of the shaded point.
    pub position: Vec3,
    /// Unit surface normal at the shaded point.
    pub normal: Vec3,
    /// Perceptual linear roughness in `[0, 1]`.
    pub roughness: f32,
}

impl DenoiseSample {
    /// Construct a sample, normalising the stored normal defensively.
    ///
    /// A zero-length or non-finite normal collapses to `+Z` so downstream
    /// dot products stay finite; callers that care should pre-validate.
    #[must_use]
    pub fn new(color: Vec3, position: Vec3, normal: Vec3, roughness: f32) -> Self {
        Self {
            color: sanitize_rgb(color),
            position: sanitize_vec(position),
            normal: safe_normalize(normal),
            roughness: roughness.clamp(0.0, 1.0),
        }
    }
}

/// Sensitivities controlling how sharply each edge-stopping term falls off.
///
/// Larger `phi_*` values mean a *looser* weight (more blur across that
/// attribute); smaller values preserve edges more aggressively.  Diffuse
/// typically uses looser normal/roughness terms than specular.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeStoppingParams {
    /// Plane-distance tolerance scale (world units); depth weight falls off as
    /// the tap drifts off the centre tangent plane faster than this.
    pub phi_depth: f32,
    /// Normal weight exponent; higher sharpens the cosine falloff.
    pub phi_normal: f32,
    /// Roughness tolerance; specular passes a small value, diffuse a large one.
    pub phi_roughness: f32,
    /// Luminance tolerance scale, divided by the (sqrt of) local variance so
    /// noisy regions tolerate larger luminance gaps (SVGF-style).
    pub phi_luminance: f32,
}

impl Default for EdgeStoppingParams {
    fn default() -> Self {
        // Balanced diffuse defaults.
        Self {
            phi_depth: 1.0,
            phi_normal: 32.0,
            phi_roughness: 0.5,
            phi_luminance: 4.0,
        }
    }
}

impl EdgeStoppingParams {
    /// Tighter preset for specular / reflection signals: edges in normal and
    /// roughness must be preserved far more aggressively than for diffuse.
    #[must_use]
    pub fn specular() -> Self {
        Self {
            phi_depth: 0.5,
            phi_normal: 128.0,
            phi_roughness: 0.08,
            phi_luminance: 2.0,
        }
    }
}

/// Depth / plane-distance edge-stopping weight in `[0, 1]`.
///
/// Rather than comparing raw depths (which breaks on grazing surfaces), this
/// measures how far `sample_pos` lies off the tangent plane through
/// `center_pos` with normal `center_normal`, normalised by `phi_depth`.  The
/// weight is `exp(-|plane_distance| / (phi_depth + eps))`.
#[must_use]
pub fn depth_edge_weight(
    center_pos: Vec3,
    center_normal: Vec3,
    sample_pos: Vec3,
    phi_depth: f32,
) -> f32 {
    let n = safe_normalize(center_normal);
    let plane_distance = (sample_pos - center_pos).dot(n).abs();
    let phi = phi_depth.max(1.0e-6);
    stable_exp(-plane_distance / phi)
}

/// Normal edge-stopping weight in `[0, 1]`: `max(0, dot(n0, n1))^phi_normal`.
///
/// The clamped cosine raised to `phi_normal` gives a lobe that is wide for
/// diffuse (small exponent) and narrow for specular (large exponent).
#[must_use]
pub fn normal_edge_weight(n0: Vec3, n1: Vec3, phi_normal: f32) -> f32 {
    let a = safe_normalize(n0);
    let b = safe_normalize(n1);
    let cosine = a.dot(b).clamp(0.0, 1.0);
    let exponent = phi_normal.max(0.0);
    ops::powf(cosine, exponent).clamp(0.0, 1.0)
}

/// Roughness edge-stopping weight in `[0, 1]`.
///
/// `exp(-|r0 - r1| / (phi_roughness + eps))`; keeps the filter from mixing
/// sharp and blurry reflections, which is critical for specular stability.
#[must_use]
pub fn roughness_edge_weight(r0: f32, r1: f32, phi_roughness: f32) -> f32 {
    let diff = (r0.clamp(0.0, 1.0) - r1.clamp(0.0, 1.0)).abs();
    let phi = phi_roughness.max(1.0e-6);
    stable_exp(-diff / phi)
}

/// Luminance edge-stopping weight in `[0, 1]`, widened by local variance.
///
/// SVGF-style: `exp(-|l0 - l1| / (phi_luminance * sqrt(variance) + eps))`.
/// High variance (noisy / under-converged) loosens the weight so the filter
/// averages more aggressively; low variance preserves genuine lighting edges.
#[must_use]
pub fn luminance_edge_weight(l0: f32, l1: f32, variance: f32, phi_luminance: f32) -> f32 {
    let diff = (l0 - l1).abs();
    let var = variance.max(0.0);
    let denom = phi_luminance.max(0.0) * var.sqrt() + 1.0e-4;
    stable_exp(-diff / denom)
}

/// Combined edge-stopping weight for one neighbour tap against the centre.
///
/// Product of the depth, normal, roughness, and (variance-guided) luminance
/// terms.  `center_variance` is the centre pixel's luminance variance estimate
/// (e.g. from [`TemporalMoments::variance`]); pass `0.0` to disable the
/// variance widening.  The result is always finite and in `[0, 1]`.
#[must_use]
pub fn edge_stopping_weight(
    center: &DenoiseSample,
    sample: &DenoiseSample,
    center_variance: f32,
    params: &EdgeStoppingParams,
) -> f32 {
    let w_depth = depth_edge_weight(
        center.position,
        center.normal,
        sample.position,
        params.phi_depth,
    );
    let w_normal = normal_edge_weight(center.normal, sample.normal, params.phi_normal);
    let w_rough = roughness_edge_weight(center.roughness, sample.roughness, params.phi_roughness);
    let w_luma = luminance_edge_weight(
        luminance(center.color.to_array()),
        luminance(sample.color.to_array()),
        center_variance,
        params.phi_luminance,
    );
    (w_depth * w_normal * w_rough * w_luma).clamp(0.0, 1.0)
}

/// One edge-stopping weighted average (a single à-trous tap gather).
///
/// `center_kernel` is the centre tap's own kernel weight (typically the à-trous
/// kernel centre, e.g. `1.0`); `neighbours` carry each tap plus its à-trous
/// kernel weight.  Returns the weighted-mean colour.  If every weight collapses
/// to zero (fully disjoint neighbourhood) the centre colour is returned
/// unchanged so the filter degrades to identity rather than producing `NaN`.
#[must_use]
pub fn atrous_filter(
    center: &DenoiseSample,
    center_kernel: f32,
    neighbours: &[(DenoiseSample, f32)],
    center_variance: f32,
    params: &EdgeStoppingParams,
) -> Vec3 {
    let w_center = center_kernel.max(0.0);
    let mut sum = center.color * w_center;
    let mut weight = w_center;

    for (sample, kernel) in neighbours {
        let k = kernel.max(0.0);
        if k == 0.0 {
            continue;
        }
        let w = k * edge_stopping_weight(center, sample, center_variance, params);
        sum += sample.color * w;
        weight += w;
    }

    if weight <= 1.0e-12 {
        center.color
    } else {
        sum / weight
    }
}

/// Standard 5-tap à-trous B-spline kernel weights `[1/16, 1/4, 3/8, 1/4, 1/16]`
/// for a given 1D offset in `{-2, -1, 0, 1, 2}`; other offsets weigh `0`.
///
/// A 2D separable pass multiplies the per-axis weights; this helper exposes the
/// canonical coefficients so the GPU twin and CPU golden share one source of
/// truth.
#[must_use]
pub fn atrous_bspline_weight(offset: i32) -> f32 {
    match offset.abs() {
        0 => 3.0 / 8.0,
        1 => 1.0 / 4.0,
        2 => 1.0 / 16.0,
        _ => 0.0,
    }
}

/// Accumulated temporal state carried per pixel between frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalState {
    /// Filtered, accumulated linear RGB radiance.
    pub color: Vec3,
    /// Luminance moments for variance-guided spatial filtering.
    pub moments: TemporalMoments,
    /// Number of frames accumulated (saturating), driving the blend weight.
    pub age: u32,
}

impl TemporalState {
    /// Seed a fresh history from a single sample (post-disocclusion).
    #[must_use]
    pub fn from_sample(color: Vec3) -> Self {
        let c = sanitize_rgb(color);
        Self {
            color: c,
            moments: TemporalMoments::from_sample(luminance(c.to_array())),
            age: 1,
        }
    }
}

/// Parameters governing temporal accumulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalParams {
    /// Maximum number of frames to accumulate; caps the EMA weight so the
    /// filter keeps adapting (ReBLUR "max accumulated frames").
    pub max_frames: u32,
    /// History-clamp half-width in standard deviations (`gamma`); the blended
    /// result is clamped to `mean ± gamma * std` of the local neighbourhood to
    /// reject ghosting after fast motion.  `0` disables clamping.
    pub clamp_gamma: f32,
}

impl Default for TemporalParams {
    fn default() -> Self {
        Self {
            max_frames: 32,
            clamp_gamma: 2.0,
        }
    }
}

/// Blend weight `alpha` (new-sample weight) for a given accumulated age.
///
/// `alpha = 1 / min(age + 1, max_frames)`: the first frames average quickly,
/// then the weight floors at `1 / max_frames` so the EMA never fully freezes.
#[must_use]
pub fn temporal_alpha(age: u32, max_frames: u32) -> f32 {
    let cap = max_frames.max(1);
    let n = (age + 1).min(cap);
    1.0 / n as f32
}

/// Clamp a colour to an axis-aligned box `mean ± gamma * std` (per channel).
///
/// Used to reject stale reprojected history: `neighbourhood_mean` /
/// `neighbourhood_std` summarise the current frame's local statistics, and the
/// blended history is pulled back inside that box.  Non-positive `gamma`
/// returns the colour unchanged.
#[must_use]
pub fn clamp_to_neighbourhood(
    color: Vec3,
    neighbourhood_mean: Vec3,
    neighbourhood_std: Vec3,
    gamma: f32,
) -> Vec3 {
    if gamma <= 0.0 {
        return sanitize_rgb(color);
    }
    let g = gamma;
    let lo = neighbourhood_mean - neighbourhood_std * g;
    let hi = neighbourhood_mean + neighbourhood_std * g;
    Vec3::new(
        clamp_ordered(color.x, lo.x, hi.x),
        clamp_ordered(color.y, lo.y, hi.y),
        clamp_ordered(color.z, lo.z, hi.z),
    )
}

/// Advance temporal accumulation by one frame.
///
/// `prev` is the reprojected previous state (or `None` on disocclusion);
/// `current_sample` is this frame's noisy radiance; `neighbourhood_mean/std`
/// summarise the current spatial neighbourhood for the history clamp.  Returns
/// the new accumulated [`TemporalState`].
///
/// On disocclusion (`prev == None`) the sample seeds a fresh history.  Otherwise
/// the previous colour is history-clamped, then blended with the new sample via
/// [`temporal_alpha`], and the luminance moments and age are advanced.
#[must_use]
pub fn temporal_accumulate(
    prev: Option<TemporalState>,
    current_sample: Vec3,
    neighbourhood_mean: Vec3,
    neighbourhood_std: Vec3,
    params: &TemporalParams,
) -> TemporalState {
    let sample = sanitize_rgb(current_sample);
    let Some(prev) = prev else {
        return TemporalState::from_sample(sample);
    };

    let clamped_history =
        clamp_to_neighbourhood(prev.color, neighbourhood_mean, neighbourhood_std, params.clamp_gamma);
    let alpha = temporal_alpha(prev.age, params.max_frames);
    let color = clamped_history.lerp(sample, alpha);

    let sample_luma = luminance(sample.to_array());
    let moments = prev.moments.blend(sample_luma, alpha);
    let age = (prev.age + 1).min(params.max_frames.max(1));

    TemporalState {
        color: sanitize_rgb(color),
        moments,
        age,
    }
}

/// Virtual (parallax-corrected) world position for reprojecting a specular
/// highlight, à la ReBLUR specular tracking.
///
/// A mirror reflection does not move with the surface but with the *virtual
/// image* behind it, at `hit_distance` along the reflection ray.  Reprojecting
/// the specular history from this virtual position (instead of the surface
/// position) avoids the smeared "reflection lag" artefact on glossy surfaces.
/// `view_dir` is the unit vector from the surface toward the camera.
#[must_use]
pub fn specular_virtual_position(
    surface_pos: Vec3,
    view_dir: Vec3,
    normal: Vec3,
    hit_distance: f32,
) -> Vec3 {
    let v = safe_normalize(view_dir);
    let n = safe_normalize(normal);
    // Reflect the view ray about the surface normal and march the hit distance.
    let reflected = reflect(-v, n);
    let dist = hit_distance.max(0.0);
    surface_pos + reflected * dist
}

/// Reflect incident direction `i` about unit normal `n` (`i - 2 (i·n) n`).
#[must_use]
fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    i - n * (2.0 * i.dot(n))
}

/// Numerically safe `exp` that never returns `NaN` and saturates for large
/// magnitudes.
#[must_use]
fn stable_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    // Clamp to avoid overflow to +inf for pathological inputs.
    ops::exp(x.clamp(-80.0, 0.0))
}

/// Normalise a vector, falling back to `+Z` for zero-length / non-finite input.
#[must_use]
fn safe_normalize(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > 1.0e-24 {
        v / len_sq.sqrt()
    } else {
        Vec3::Z
    }
}

/// Replace any non-finite component of an RGB triple with `0`.
#[must_use]
fn sanitize_rgb(c: Vec3) -> Vec3 {
    Vec3::new(finite_or_zero(c.x), finite_or_zero(c.y), finite_or_zero(c.z)).max(Vec3::ZERO)
}

/// Replace any non-finite component of a position with `0`.
#[must_use]
fn sanitize_vec(v: Vec3) -> Vec3 {
    Vec3::new(finite_or_zero(v.x), finite_or_zero(v.y), finite_or_zero(v.z))
}

#[must_use]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Clamp `x` to `[lo, hi]`, tolerating `lo > hi` by swapping (defensive).
#[must_use]
fn clamp_ordered(x: f32, lo: f32, hi: f32) -> f32 {
    let (a, b) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    x.clamp(a, b)
}

/// Compute the per-channel mean and standard deviation of a colour
/// neighbourhood, for use with [`clamp_to_neighbourhood`].
///
/// Returns `(mean, std)`.  An empty slice yields `(ZERO, ZERO)`.  Variance is
/// clamped non-negative before the square root so cancellation cannot feed a
/// negative into `sqrt`.
#[must_use]
pub fn neighbourhood_statistics(colors: &[Vec3]) -> (Vec3, Vec3) {
    let n = colors.len();
    if n == 0 {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let inv = 1.0 / n as f32;
    let mut mean = Vec3::ZERO;
    for c in colors {
        mean += sanitize_rgb(*c);
    }
    mean *= inv;

    let mut m2 = Vec3::ZERO;
    for c in colors {
        let d = sanitize_rgb(*c) - mean;
        m2 += d * d;
    }
    let var = (m2 * inv).max(Vec3::ZERO);
    let std = Vec3::new(var.x.sqrt(), var.y.sqrt(), var.z.sqrt());
    (mean, std)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(color: Vec3, pos: Vec3, normal: Vec3, rough: f32) -> DenoiseSample {
        DenoiseSample::new(color, pos, normal, rough)
    }

    #[test]
    fn depth_weight_is_one_on_plane_and_decays_off_plane() {
        let c = Vec3::ZERO;
        let n = Vec3::Z;
        // In-plane shift: zero plane distance -> weight 1.
        let on_plane = depth_edge_weight(c, n, Vec3::new(5.0, -3.0, 0.0), 1.0);
        assert!((on_plane - 1.0).abs() < 1e-6);
        // Off-plane shift decays monotonically.
        let near = depth_edge_weight(c, n, Vec3::new(0.0, 0.0, 0.5), 1.0);
        let far = depth_edge_weight(c, n, Vec3::new(0.0, 0.0, 2.0), 1.0);
        assert!(near > far);
        assert!(far > 0.0 && near < 1.0);
    }

    #[test]
    fn normal_weight_peaks_when_aligned() {
        let aligned = normal_edge_weight(Vec3::Z, Vec3::Z, 32.0);
        assert!((aligned - 1.0).abs() < 1e-6);
        let tilted = normal_edge_weight(Vec3::Z, Vec3::new(0.3, 0.0, 0.954).normalize(), 32.0);
        let orth = normal_edge_weight(Vec3::Z, Vec3::X, 32.0);
        assert!(aligned > tilted && tilted > orth);
        assert!(orth.abs() < 1e-6);
    }

    #[test]
    fn roughness_weight_decays_with_difference() {
        assert!((roughness_edge_weight(0.5, 0.5, 0.1) - 1.0).abs() < 1e-6);
        let small = roughness_edge_weight(0.5, 0.55, 0.1);
        let large = roughness_edge_weight(0.1, 0.9, 0.1);
        assert!(small > large);
    }

    #[test]
    fn luminance_weight_widens_with_variance() {
        // Same luminance gap tolerated more under high variance.
        let low_var = luminance_edge_weight(1.0, 2.0, 0.01, 4.0);
        let high_var = luminance_edge_weight(1.0, 2.0, 4.0, 4.0);
        assert!(high_var > low_var);
    }

    #[test]
    fn atrous_filter_averages_a_flat_noisy_region() {
        // All taps share geometry; only colour differs -> should average.
        let center = s(Vec3::splat(1.0), Vec3::ZERO, Vec3::Z, 0.5);
        let neighbours = [
            (s(Vec3::splat(0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::Z, 0.5), 1.0),
            (s(Vec3::splat(2.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::Z, 0.5), 1.0),
        ];
        let out = atrous_filter(&center, 1.0, &neighbours, 1.0, &EdgeStoppingParams::default());
        // Weighted mean of {1,0,2} with near-equal weights ~ 1.
        assert!((out.x - 1.0).abs() < 0.25);
    }

    #[test]
    fn atrous_filter_rejects_disjoint_neighbour() {
        // Neighbour on a different plane far away: weight collapses, centre kept.
        let center = s(Vec3::splat(1.0), Vec3::ZERO, Vec3::Z, 0.1);
        let bad = s(
            Vec3::splat(100.0),
            Vec3::new(0.0, 0.0, 50.0),
            Vec3::NEG_Z,
            0.9,
        );
        let out = atrous_filter(
            &center,
            1.0,
            &[(bad, 1.0)],
            0.0,
            &EdgeStoppingParams::specular(),
        );
        assert!((out - center.color).length() < 1e-3);
    }

    #[test]
    fn atrous_identity_on_empty_neighbourhood() {
        let center = s(Vec3::new(0.3, 0.6, 0.9), Vec3::ZERO, Vec3::Z, 0.5);
        let out = atrous_filter(&center, 1.0, &[], 0.0, &EdgeStoppingParams::default());
        assert_eq!(out, center.color);
    }

    #[test]
    fn bspline_weights_sum_to_one() {
        let sum: f32 = (-2..=2).map(atrous_bspline_weight).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn temporal_alpha_decreases_and_floors() {
        // `temporal_alpha(age)` is the running-average weight `1/(age+1)`,
        // floored at `1/max_frames` so the EMA never freezes.
        let a0 = temporal_alpha(0, 32); // 1/min(1,32) = 1.0 (no history yet)
        let a1 = temporal_alpha(1, 32); // 1/min(2,32) = 0.5
        let a5 = temporal_alpha(5, 32); // 1/min(6,32) = 1/6
        let afloor = temporal_alpha(1000, 32); // floored at 1/32
        assert!((a0 - 1.0).abs() < 1e-6);
        assert!((a1 - 0.5).abs() < 1e-6);
        assert!((a5 - 1.0 / 6.0).abs() < 1e-6);
        assert!((afloor - 1.0 / 32.0).abs() < 1e-6);
        // Strictly monotone non-increasing, then floors.
        assert!(a0 > a1 && a1 > a5 && a5 > afloor);
    }

    #[test]
    fn temporal_accumulate_seeds_on_disocclusion() {
        let st = temporal_accumulate(None, Vec3::splat(2.0), Vec3::ZERO, Vec3::ZERO, &TemporalParams::default());
        assert_eq!(st.age, 1);
        assert!((st.color - Vec3::splat(2.0)).length() < 1e-6);
    }

    #[test]
    fn temporal_accumulate_converges_toward_constant_signal() {
        let params = TemporalParams::default();
        let target = Vec3::splat(1.0);
        let mut st = TemporalState::from_sample(Vec3::ZERO);
        // Seeded from a (deliberately wrong) ZERO history; the running average
        // then the floored EMA must drag the state back onto the constant signal.
        for _ in 0..96 {
            st = temporal_accumulate(Some(st), target, target, Vec3::splat(0.5), &params);
        }
        assert!((st.color - target).length() < 1e-2);
        assert_eq!(st.age, params.max_frames);
    }

    #[test]
    fn history_clamp_rejects_stale_history() {
        // Stale bright history, current neighbourhood is dark & tight.
        let clamped = clamp_to_neighbourhood(
            Vec3::splat(10.0),
            Vec3::splat(0.2),
            Vec3::splat(0.05),
            2.0,
        );
        assert!(clamped.x <= 0.2 + 2.0 * 0.05 + 1e-6);
        assert!(clamped.x >= 0.2 - 2.0 * 0.05 - 1e-6);
    }

    #[test]
    fn history_clamp_gamma_zero_is_identity() {
        let c = Vec3::new(3.0, 4.0, 5.0);
        assert_eq!(clamp_to_neighbourhood(c, Vec3::ZERO, Vec3::ZERO, 0.0), c);
    }

    #[test]
    fn neighbourhood_statistics_matches_closed_form() {
        let colors = [Vec3::splat(0.0), Vec3::splat(2.0)];
        let (mean, std) = neighbourhood_statistics(&colors);
        assert!((mean.x - 1.0).abs() < 1e-6);
        // Population std of {0,2} = 1.
        assert!((std.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn neighbourhood_statistics_empty_is_zero() {
        let (mean, std) = neighbourhood_statistics(&[]);
        assert_eq!(mean, Vec3::ZERO);
        assert_eq!(std, Vec3::ZERO);
    }

    #[test]
    fn specular_virtual_position_marches_along_reflection() {
        // Camera above a floor looking down; reflection ray points up.
        let surface = Vec3::ZERO;
        let view = Vec3::Z; // toward camera (up)
        let n = Vec3::Z;
        let vp = specular_virtual_position(surface, view, n, 3.0);
        // Mirror reflection of a straight-down view about up-normal is straight up.
        assert!((vp - Vec3::new(0.0, 0.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn weights_are_finite_on_degenerate_inputs() {
        let w = edge_stopping_weight(
            &DenoiseSample::new(Vec3::splat(f32::NAN), Vec3::ZERO, Vec3::ZERO, 2.0),
            &DenoiseSample::new(Vec3::splat(f32::INFINITY), Vec3::splat(f32::NAN), Vec3::ZERO, -1.0),
            f32::NAN,
            &EdgeStoppingParams::default(),
        );
        assert!(w.is_finite());
        assert!((0.0..=1.0).contains(&w));
    }

    #[test]
    fn determinism() {
        let center = s(Vec3::new(0.5, 0.5, 0.5), Vec3::ZERO, Vec3::Z, 0.3);
        let nb = [(s(Vec3::splat(0.7), Vec3::new(0.5, 0.0, 0.0), Vec3::Z, 0.3), 0.25)];
        let a = atrous_filter(&center, 1.0, &nb, 0.2, &EdgeStoppingParams::default());
        let b = atrous_filter(&center, 1.0, &nb, 0.2, &EdgeStoppingParams::default());
        assert_eq!(a, b);
    }
}
