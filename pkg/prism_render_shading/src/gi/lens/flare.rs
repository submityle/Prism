//! Screen-space lens flare — ghosts and halo (CPU golden reference).
//!
//! A bright light hitting the front element of a compound lens bounces between
//! element surfaces before reaching the sensor.  Those internal reflections
//! paint a chain of defocused "ghost" images along the line through the optical
//! centre, plus a soft "halo" ring from the near-circular aperture.  Rendering
//! the real inter-reflection path is far too expensive at runtime, so games use
//! the screen-space approximation popularised by John Chapman's *Pseudo Lens
//! Flare* (after Hullin et al., *Physically-Based Real-Time Lens Flare
//! Rendering*, SIGGRAPH 2011): take the already-resolved HDR frame, keep only
//! its highlights, and re-project them back through the centre.
//!
//! The construction here is:
//!
//! * **Highlight prefilter.**  [`prefilter`] keeps only the energy of a sample
//!   above a threshold, so only genuine highlights seed flares.
//! * **Ghosts.**  The pixel is flipped about the centre and sampled at several
//!   evenly spaced steps back toward (and past) the centre.  Every ghost sample
//!   therefore lies on the line through the pixel and the centre — the defining
//!   property of lens ghosts.  Each sample is weighted by a radial falloff so
//!   ghosts near the frame edge fade out.
//! * **Halo.**  A single sample stepped a fixed distance toward the centre,
//!   weighted by how close the pixel's radius is to a target ring radius, which
//!   paints the characteristic aperture ring.
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * The radial weight exponent uses [`bevy_math::ops::powf`]; no other
//!   transcendental is needed.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit `Vec2`).
//! * Channel order is always `[R, G, B]`; colours are kept non-negative.
//! * Defensive clamping everywhere: counts, radii and exponents are sanitized
//!   so a degenerate configuration never emits `NaN`/`inf`.

use alloc::vec::Vec;
use bevy_math::{Vec2, ops};

/// Hard upper bound on the number of ghost samples synthesized per pixel.
pub const MAX_GHOSTS: u32 = 64;

/// Smallest denominator used when normalising radii, to avoid divide-by-zero.
const EPSILON: f32 = 1.0e-6;

/// One screen-space flare sample: where to fetch and how strongly to add it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GhostSample {
    /// UV coordinate to sample the (prefiltered) HDR frame at.
    pub uv: Vec2,
    /// Non-negative accumulation weight for this sample.
    pub weight: f32,
}

/// Replace a non-finite scalar with a fallback.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Sanitize a `Vec2` component-wise against a fallback.
#[inline]
fn sanitize_vec2(v: Vec2, fallback: Vec2) -> Vec2 {
    Vec2::new(finite_or(v.x, fallback.x), finite_or(v.y, fallback.y))
}

/// Keep only the highlight energy of `color` above `threshold`.
///
/// Returns `max(color - threshold, 0)` per channel: a sample at or below the
/// threshold contributes nothing (no flare), and a bright sample contributes
/// only its surplus so the flare tracks genuine highlights.  A negative or
/// non-finite threshold is treated as `0`, and non-finite colour components
/// collapse to `0`.
#[must_use]
pub fn prefilter(color: [f32; 3], threshold: f32) -> [f32; 3] {
    let t = finite_or(threshold, 0.0).max(0.0);
    [
        (finite_or(color[0], 0.0) - t).max(0.0),
        (finite_or(color[1], 0.0) - t).max(0.0),
        (finite_or(color[2], 0.0) - t).max(0.0),
    ]
}

/// Lens-flare synthesis parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlareConfig {
    /// Optical centre in UV space (usually `(0.5, 0.5)`).
    pub center: Vec2,
    /// Number of ghost samples (clamped to `1..=MAX_GHOSTS`).
    pub ghost_count: u32,
    /// Spacing between successive ghosts as a fraction of the pixel-to-centre
    /// vector.  Larger values spread the ghosts farther apart.
    pub dispersal: f32,
    /// Exponent of the radial `(1 - r)^falloff` weight; higher concentrates
    /// energy toward the centre.
    pub weight_falloff: f32,
    /// Distance (in UV units) the halo sample steps toward the centre.
    pub halo_width: f32,
    /// Target radius of the halo ring (distance from the centre in UV units).
    pub halo_radius: f32,
    /// Half-thickness of the halo ring; the ring weight falls off over this
    /// band around [`FlareConfig::halo_radius`].
    pub halo_thickness: f32,
    /// Overall multiplier applied to the composited flare.
    pub intensity: f32,
}

impl Default for FlareConfig {
    /// A moderate default: 4 ghosts, centred, with a thin halo ring.
    fn default() -> Self {
        Self {
            center: Vec2::splat(0.5),
            ghost_count: 4,
            dispersal: 0.3,
            weight_falloff: 10.0,
            halo_width: 0.45,
            halo_radius: 0.35,
            halo_thickness: 0.1,
            intensity: 1.0,
        }
    }
}

impl FlareConfig {
    /// Sanitize every field: centre forced finite, count clamped to
    /// `1..=MAX_GHOSTS`, and all scalar parameters forced finite and
    /// non-negative (thickness floored to a small positive value).
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            center: sanitize_vec2(self.center, Vec2::splat(0.5)),
            ghost_count: self.ghost_count.clamp(1, MAX_GHOSTS),
            dispersal: finite_or(self.dispersal, 0.0).max(0.0),
            weight_falloff: finite_or(self.weight_falloff, 0.0).max(0.0),
            halo_width: finite_or(self.halo_width, 0.0).max(0.0),
            halo_radius: finite_or(self.halo_radius, 0.0).max(0.0),
            halo_thickness: finite_or(self.halo_thickness, EPSILON).max(EPSILON),
            intensity: finite_or(self.intensity, 0.0).max(0.0),
        }
    }
}

/// Largest distance from `center` to any of the four UV corners.
///
/// Used to normalise radial distances to `[0, 1]` so the radial weight reaches
/// zero at the farthest corner.  Floored to `EPSILON` so a centre pinned to a
/// corner cannot produce a zero divisor.
#[inline]
fn corner_max_dist(center: Vec2) -> f32 {
    let corners = [
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(1.0, 1.0),
    ];
    let mut m = 0.0_f32;
    for c in corners {
        m = m.max((c - center).length());
    }
    m.max(EPSILON)
}

/// Radial weight `(1 - d)^falloff` where `d` is the point's distance from the
/// centre normalised by [`corner_max_dist`].  Peaks at the centre, zero at the
/// farthest corner.
#[inline]
fn radial_weight(center: Vec2, point: Vec2, max_dist: f32, falloff: f32) -> f32 {
    let d = ((center - point).length() / max_dist).clamp(0.0, 1.0);
    ops::powf((1.0 - d).max(0.0), falloff)
}

/// Build the ghost sample points and weights for the pixel at `uv`.
///
/// The pixel is flipped about the centre (`flipped = 2*center - uv`) and sampled
/// at `flipped + i * dispersal * (center - flipped)` for `i` in
/// `0..ghost_count`.  Because every offset is a scalar multiple of
/// `(center - uv)` added to a point on the pixel-centre line, all returned
/// coordinates are collinear with the pixel and the centre.  Each weight is the
/// radial falloff at that coordinate.
#[must_use]
pub fn synthesize_ghosts(uv: Vec2, config: FlareConfig) -> Vec<GhostSample> {
    let cfg = config.sanitized();
    let uv = sanitize_vec2(uv, cfg.center);
    let flipped = 2.0 * cfg.center - uv;
    let ghost_step = (cfg.center - flipped) * cfg.dispersal;
    let max_dist = corner_max_dist(cfg.center);

    let mut out = Vec::with_capacity(cfg.ghost_count as usize);
    for i in 0..cfg.ghost_count {
        let point = flipped + ghost_step * (i as f32);
        let weight = radial_weight(cfg.center, point, max_dist, cfg.weight_falloff);
        out.push(GhostSample { uv: point, weight });
    }
    out
}

/// Build the single halo sample for the pixel at `uv`.
///
/// The sample steps `halo_width` toward the centre along the pixel-centre line
/// (so it, too, is collinear with the pixel and centre), and its weight peaks
/// when the pixel's radius matches [`FlareConfig::halo_radius`], falling off over
/// [`FlareConfig::halo_thickness`] to form a ring.
#[must_use]
pub fn halo_sample(uv: Vec2, config: FlareConfig) -> GhostSample {
    let cfg = config.sanitized();
    let uv = sanitize_vec2(uv, cfg.center);
    let to_center = cfg.center - uv;
    let dir = to_center.normalize_or_zero();
    let point = uv + dir * cfg.halo_width;

    let radius = to_center.length();
    let d = ((radius - cfg.halo_radius).abs() / cfg.halo_thickness).clamp(0.0, 1.0);
    let weight = ops::powf((1.0 - d).max(0.0), cfg.weight_falloff);
    GhostSample { uv: point, weight }
}

/// Composite the full flare (ghosts + halo) for the pixel at `uv`.
///
/// `sampler(uv)` returns the HDR colour of the resolved frame at `uv`; each
/// sample is highlight-prefiltered with `threshold`, scaled by its weight, and
/// accumulated.  The sum is multiplied by [`FlareConfig::intensity`].  The
/// result is non-negative and finite: with no highlights above `threshold` the
/// output is pure black (no flare).
#[must_use]
pub fn synthesize<F>(uv: Vec2, config: FlareConfig, threshold: f32, sampler: F) -> [f32; 3]
where
    F: Fn(Vec2) -> [f32; 3],
{
    let cfg = config.sanitized();
    let mut acc = [0.0_f32; 3];

    for g in synthesize_ghosts(uv, cfg) {
        let c = prefilter(sampler(g.uv), threshold);
        for ch in 0..3 {
            acc[ch] += c[ch] * g.weight;
        }
    }

    let halo = halo_sample(uv, cfg);
    let hc = prefilter(sampler(halo.uv), threshold);
    for ch in 0..3 {
        acc[ch] += hc[ch] * halo.weight;
    }

    [
        finite_or(acc[0] * cfg.intensity, 0.0).max(0.0),
        finite_or(acc[1] * cfg.intensity, 0.0).max(0.0),
        finite_or(acc[2] * cfg.intensity, 0.0).max(0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// 2D cross product; zero means the two vectors are parallel.
    fn cross(a: Vec2, b: Vec2) -> f32 {
        a.x * b.y - a.y * b.x
    }

    #[test]
    fn prefilter_discards_below_threshold() {
        assert_eq!(prefilter([0.2, 0.3, 0.1], 1.0), [0.0, 0.0, 0.0]);
        let out = prefilter([2.0, 1.5, 1.0], 1.0);
        assert!(approx(out[0], 1.0, EPS));
        assert!(approx(out[1], 0.5, EPS));
        assert!(approx(out[2], 0.0, EPS));
    }

    #[test]
    fn no_highlights_means_no_flare() {
        let cfg = FlareConfig::default();
        // Every sample is dim; threshold removes all of it.
        let out = synthesize(Vec2::new(0.8, 0.2), cfg, 1.0, |_| [0.1, 0.1, 0.1]);
        assert_eq!(out, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn ghost_weights_sum_positive() {
        let cfg = FlareConfig::default();
        let ghosts = synthesize_ghosts(Vec2::new(0.55, 0.52), cfg);
        assert_eq!(ghosts.len(), cfg.ghost_count as usize);
        let sum: f32 = ghosts.iter().map(|g| g.weight).sum();
        assert!(sum > 0.0, "ghost weights should sum positive, got {sum}");
        for g in &ghosts {
            assert!(g.weight >= 0.0 && g.weight.is_finite());
        }
    }

    #[test]
    fn ghosts_are_collinear_with_center() {
        let cfg = FlareConfig::default();
        let uv = Vec2::new(0.82, 0.33);
        let ghosts = synthesize_ghosts(uv, cfg);
        let axis = cfg.center - uv; // the pixel -> centre direction
        for g in &ghosts {
            let v = g.uv - cfg.center;
            assert!(cross(v, axis).abs() <= EPS, "ghost off the centre line: {:?}", g.uv);
        }
    }

    #[test]
    fn halo_sample_is_collinear_with_center() {
        let cfg = FlareConfig::default();
        let uv = Vec2::new(0.2, 0.7);
        let halo = halo_sample(uv, cfg);
        let axis = cfg.center - uv;
        let v = halo.uv - cfg.center;
        assert!(cross(v, axis).abs() <= EPS, "halo off the centre line: {:?}", halo.uv);
    }

    #[test]
    fn halo_weight_peaks_near_target_radius() {
        let cfg = FlareConfig::default();
        // A pixel exactly at halo_radius from the centre should weight ~1.
        let on_ring = cfg.center + Vec2::new(cfg.halo_radius, 0.0);
        let w_on = halo_sample(on_ring, cfg).weight;
        // A pixel far off the ring should weight much less.
        let off_ring = cfg.center + Vec2::new(cfg.halo_radius + 3.0 * cfg.halo_thickness, 0.0);
        let w_off = halo_sample(off_ring, cfg).weight;
        assert!(w_on > w_off, "ring weight should peak on the ring: {w_on} vs {w_off}");
        assert!(approx(w_on, 1.0, 1.0e-3), "on-ring weight should be ~1: {w_on}");
    }

    #[test]
    fn synthesis_is_deterministic() {
        let cfg = FlareConfig::default();
        let uv = Vec2::new(0.9, 0.15);
        let sampler = |p: Vec2| [2.0 + p.x, 1.5, 3.0 * p.y];
        let a = synthesize(uv, cfg, 1.0, sampler);
        let b = synthesize(uv, cfg, 1.0, sampler);
        assert_eq!(a, b);
    }

    #[test]
    fn degenerate_config_stays_finite() {
        let cfg = FlareConfig {
            center: Vec2::new(f32::NAN, 0.5),
            ghost_count: 0,
            dispersal: f32::INFINITY,
            weight_falloff: -5.0,
            halo_width: f32::NAN,
            halo_radius: -1.0,
            halo_thickness: 0.0,
            intensity: f32::INFINITY,
        };
        let out = synthesize(Vec2::new(f32::INFINITY, 0.3), cfg, -2.0, |_| [1e9, 2e9, 3e9]);
        for c in out {
            assert!(c.is_finite(), "flare colour not finite: {c}");
        }
        // Count clamps up to at least one ghost.
        assert_eq!(synthesize_ghosts(Vec2::new(0.3, 0.3), cfg).len(), 1);
    }

    #[test]
    fn centre_pixel_ghosts_stay_at_centre() {
        // A pixel at the centre flips to itself; every ghost sits at the centre.
        let cfg = FlareConfig::default();
        let ghosts = synthesize_ghosts(cfg.center, cfg);
        for g in ghosts {
            assert!((g.uv - cfg.center).length() <= EPS, "centre ghost drifted: {:?}", g.uv);
        }
    }
}
