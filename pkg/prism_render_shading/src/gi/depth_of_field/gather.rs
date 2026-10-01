//! Scatter-as-gather bokeh accumulation on a circular lens kernel.
//!
//! A physically scattering lens spreads each source pixel's energy over its
//! circle of confusion.  Implemented literally that is a *scatter* — every
//! pixel writes to many neighbours — which is awkward on a GPU.  The standard
//! reformulation (Abadie, "A Life of a Bokeh", SIGGRAPH 2018; Karis bokeh)
//! inverts it into a *gather*: a destination pixel visits the neighbours on a
//! disk around it and asks, for each neighbour, "does *your* blur disk reach
//! far enough to cover me?"  A neighbour at on-screen distance `d` contributes
//! only when its CoC radius `|r|` is at least `d`; its energy is spread over
//! its disk area, so the contribution weight falls as `1 / area`.  Summing the
//! weighted neighbour colours and dividing by the total weight yields the
//! energy-conserving blurred colour.
//!
//! This module is the deterministic CPU reference for that gather.  Sampling
//! positions come from an analytic golden-angle (Vogel) spiral, so the kernel
//! is reproducible with no RNG and no stored tables.
//!
//! # Conventions
//! * Deterministic pure functions: no RNG, I/O, GPU, or `unsafe`.  The only
//!   allocation is [`golden_angle_disk`], which fills a `Vec` of offsets.
//! * Offsets and CoC radii are in the *same* planar unit (output pixels is the
//!   intended choice); the gather never needs the depth itself beyond what the
//!   caller already folded into each sample's CoC radius.
//! * CoC radii are magnitudes here (`>= 0`): the near/far *sign* handling lives
//!   in [`super::layers`]; this accumulator treats a disk as a disk.
//! * Transcendental math via [`bevy_math::ops`]; `sqrt` via the inherent
//!   method.
//! * Defensive clamping everywhere: a minimum disk radius keeps the `1 / area`
//!   weight finite, zero-weight gathers fall back to the centre colour, and no
//!   path emits `NaN`/`inf`.

use alloc::vec::Vec;
use bevy_math::{ops, Vec2, Vec3};
use core::f32::consts::PI;

/// Golden angle `pi * (3 - sqrt(5))` radians (~2.399963).
///
/// Consecutive Vogel-spiral samples are offset by this angle, which maximally
/// avoids clumping and gives an even, deterministic disk coverage.
const GOLDEN_ANGLE: f32 = 2.399_963_2;

/// Smallest CoC radius (in planar units) that still spreads energy.
///
/// A neighbour with a smaller radius is treated as a point sample contributing
/// only to its own pixel; this also floors the `1 / area` weight so an
/// in-focus neighbour cannot acquire an unbounded weight.
const MIN_SPREAD_RADIUS: f32 = 0.5;

/// One neighbour visited by the gather.
///
/// Laid out as a GPU-twin-friendly plain-data struct.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatherSample {
    /// Linear scene colour of the neighbour (pre-exposure, non-negative).
    pub color: Vec3,
    /// On-screen offset of the neighbour from the destination pixel, in planar
    /// units (pixels).
    pub offset: Vec2,
    /// Signed CoC radius of the neighbour, in the same planar units.  Only the
    /// magnitude is used for coverage; the sign is accepted for convenience.
    pub coc_radius: f32,
}

impl GatherSample {
    /// Builds a sample, sanitising non-finite fields to safe defaults.
    #[inline]
    pub fn new(color: Vec3, offset: Vec2, coc_radius: f32) -> Self {
        Self {
            color: sanitize_color(color),
            offset: if offset.is_finite() { offset } else { Vec2::ZERO },
            coc_radius: if coc_radius.is_finite() { coc_radius } else { 0.0 },
        }
    }
}

/// Accumulated result of a gather: the energy-conserving colour and the total
/// weight that produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatherResult {
    /// Weighted-average colour.  Equals the centre colour when no neighbour
    /// contributed.
    pub color: Vec3,
    /// Sum of all contribution weights (`> 0` whenever the centre itself
    /// contributes, which it always does).
    pub weight: f32,
}

/// Tunable bokeh shaping for [`gather`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatherParams {
    /// Softness of the coverage edge, in planar units.
    ///
    /// The coverage test `|r| >= d` is softened to a linear ramp of this width
    /// just inside the disk rim, removing aliasing on the bokeh boundary.  `0`
    /// gives a hard-edged disk.
    pub edge_softness: f32,
    /// Rim-brightening factor in `[0, inf)`.
    ///
    /// Real bokeh is brighter at the disk edge (cat's-eye / soap-bubble look).
    /// `0` keeps a flat disk; larger values boost the weight of samples landing
    /// near a neighbour's rim.  A gentle default of `0.0` keeps the reference
    /// energy-neutral unless asked otherwise.
    pub edge_boost: f32,
}

impl Default for GatherParams {
    #[inline]
    fn default() -> Self {
        Self {
            edge_softness: 1.0,
            edge_boost: 0.0,
        }
    }
}

impl GatherParams {
    #[inline]
    fn sanitized(self) -> Self {
        Self {
            edge_softness: if self.edge_softness.is_finite() {
                self.edge_softness.max(0.0)
            } else {
                0.0
            },
            edge_boost: if self.edge_boost.is_finite() {
                self.edge_boost.max(0.0)
            } else {
                0.0
            },
        }
    }
}

/// Clamps a colour to the finite non-negative octant.
#[inline]
fn sanitize_color(c: Vec3) -> Vec3 {
    Vec3::new(
        if c.x.is_finite() { c.x.max(0.0) } else { 0.0 },
        if c.y.is_finite() { c.y.max(0.0) } else { 0.0 },
        if c.z.is_finite() { c.z.max(0.0) } else { 0.0 },
    )
}

/// Generates `count` Vogel-spiral sample offsets filling a disk of `radius`.
///
/// Sample `i` sits at radius `radius * sqrt((i + 0.5) / count)` and angle
/// `i * GOLDEN_ANGLE`, which distributes points with equal area per sample and
/// no angular clumping.  Returns an empty `Vec` for a non-positive `count` or a
/// non-finite / non-positive `radius`.
pub fn golden_angle_disk(count: u32, radius: f32) -> Vec<Vec2> {
    let mut out = Vec::new();
    if count == 0 || !radius.is_finite() || radius <= 0.0 {
        return out;
    }
    let inv_count = 1.0 / count as f32;
    out.reserve(count as usize);
    for i in 0..count {
        let r = radius * ((i as f32 + 0.5) * inv_count).sqrt();
        let theta = i as f32 * GOLDEN_ANGLE;
        let (s, c) = ops::sin_cos(theta);
        out.push(Vec2::new(r * c, r * s));
    }
    out
}

/// Coverage weight of one neighbour for the destination at the kernel centre.
///
/// Returns `0` when the neighbour's blur disk does not reach the centre; near
/// the rim the weight ramps smoothly over `edge_softness`.  Inside the disk the
/// base weight is `1 / (pi * r^2)` (energy spread over the disk area), floored
/// by `MIN_SPREAD_RADIUS`, optionally rim-boosted by `edge_boost`.
#[inline]
fn coverage_weight(dist: f32, coc_radius: f32, params: GatherParams) -> f32 {
    let r = coc_radius.abs().max(MIN_SPREAD_RADIUS);
    // How far outside the disk is the centre?  `dist - r > 0` means no cover.
    let overshoot = dist - r;
    let coverage = if params.edge_softness > 0.0 {
        // Linear ramp from full (overshoot <= -softness) to none (overshoot>=0).
        (1.0 - (overshoot + params.edge_softness) / params.edge_softness).clamp(0.0, 1.0)
    } else if overshoot <= 0.0 {
        1.0
    } else {
        0.0
    };
    if coverage <= 0.0 {
        return 0.0;
    }
    let area = PI * r * r;
    let base = coverage / area;
    if params.edge_boost > 0.0 {
        // Rim fraction in [0,1]: 0 at disk centre, 1 at the rim.
        let rim = (dist / r).clamp(0.0, 1.0);
        base * (1.0 + params.edge_boost * rim)
    } else {
        base
    }
}

/// Gathers `samples` into the destination pixel at the kernel centre.
///
/// Each sample's offset length is its on-screen distance to the centre; it
/// contributes when its CoC disk covers the centre (see [`coverage_weight`]).
/// The returned colour is the weight-normalised average; if no sample
/// contributes (every disk too small) the result falls back to `center_color`
/// with a nominal unit weight so callers never divide by zero.
///
/// `center_color` is the destination's own (sharp) colour, used only for the
/// degenerate fallback; include the centre explicitly in `samples` (offset
/// `Vec2::ZERO`) when it should participate in the average.
pub fn gather(center_color: Vec3, samples: &[GatherSample], params: GatherParams) -> GatherResult {
    let params = params.sanitized();
    let center_color = sanitize_color(center_color);
    let mut accum = Vec3::ZERO;
    let mut total = 0.0_f32;
    for s in samples {
        let s = GatherSample::new(s.color, s.offset, s.coc_radius);
        let dist = s.offset.length();
        if !dist.is_finite() {
            continue;
        }
        let w = coverage_weight(dist, s.coc_radius, params);
        if w > 0.0 && w.is_finite() {
            accum += s.color * w;
            total += w;
        }
    }
    if total > 0.0 {
        GatherResult {
            color: sanitize_color(accum / total),
            weight: total,
        }
    } else {
        GatherResult {
            color: center_color,
            weight: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn approx_vec(a: Vec3, b: Vec3, eps: f32) -> bool {
        approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
    }

    /// Builds a gather set from disk offsets, all sharing one colour and CoC.
    fn uniform_samples(offsets: &[Vec2], color: Vec3, coc: f32) -> Vec<GatherSample> {
        offsets
            .iter()
            .map(|&o| GatherSample::new(color, o, coc))
            .collect()
    }

    #[test]
    fn disk_has_requested_count_and_radius_bound() {
        let pts = golden_angle_disk(64, 10.0);
        assert_eq!(pts.len(), 64);
        for p in &pts {
            assert!(p.length() <= 10.0 + 1.0e-5, "point escaped disk: {p:?}");
        }
    }

    #[test]
    fn disk_rejects_degenerate_requests() {
        assert!(golden_angle_disk(0, 10.0).is_empty());
        assert!(golden_angle_disk(32, 0.0).is_empty());
        assert!(golden_angle_disk(32, f32::NAN).is_empty());
    }

    #[test]
    fn disk_is_roughly_centered() {
        // Equal-area Vogel sampling has a near-zero centroid for large counts.
        let pts = golden_angle_disk(2048, 1.0);
        let mut sum = Vec2::ZERO;
        for p in &pts {
            sum += *p;
        }
        let centroid = sum / pts.len() as f32;
        assert!(centroid.length() < 0.05, "centroid too far: {centroid:?}");
    }

    #[test]
    fn constant_field_reproduces_input_color() {
        // A constant-colour, constant-CoC neighbourhood must gather back to the
        // same colour (energy-conserving weighted average of equal colours).
        let offsets = golden_angle_disk(128, 8.0);
        let color = Vec3::new(0.3, 0.6, 0.9);
        let samples = uniform_samples(&offsets, color, 8.0);
        let r = gather(color, &samples, GatherParams::default());
        assert!(r.weight > 0.0, "weight must be positive");
        assert!(approx_vec(r.color, color, 1.0e-5), "got {:?}", r.color);
    }

    #[test]
    fn in_focus_center_is_barely_blurred() {
        // Centre is in focus (tiny CoC); surrounding neighbours are also in
        // focus so their disks cannot reach the centre.  Only the centre
        // sample (offset 0) contributes -> output == centre colour.
        let center = Vec3::new(1.0, 0.2, 0.1);
        let mut samples = alloc::vec![GatherSample::new(center, Vec2::ZERO, 0.0)];
        for p in golden_angle_disk(64, 6.0) {
            // Neighbours have a different colour but near-zero CoC.
            samples.push(GatherSample::new(Vec3::new(0.0, 0.0, 1.0), p, 0.0));
        }
        let r = gather(center, &samples, GatherParams::default());
        assert!(approx_vec(r.color, center, 1.0e-4), "focus leaked: {:?}", r.color);
    }

    #[test]
    fn large_coc_neighbor_bleeds_into_center() {
        // A single bright neighbour with a wide CoC disk should colour an
        // otherwise dark centre: its disk covers the centre, so it contributes.
        let center = Vec3::ZERO;
        let bright = Vec3::splat(4.0);
        let samples = alloc::vec![
            GatherSample::new(center, Vec2::ZERO, 0.0),
            GatherSample::new(bright, Vec2::new(5.0, 0.0), 12.0),
        ];
        let r = gather(center, &samples, GatherParams::default());
        assert!(r.color.length() > 0.0, "bright bokeh should bleed in");
        assert!(r.weight > 0.0);
    }

    #[test]
    fn out_of_reach_neighbor_does_not_contribute() {
        // Neighbour sits at distance 10 but its CoC radius is only 3: its disk
        // cannot cover the centre, so the centre keeps its own colour.
        let center = Vec3::new(0.5, 0.5, 0.5);
        let samples = alloc::vec![
            GatherSample::new(center, Vec2::ZERO, 0.0),
            GatherSample::new(Vec3::splat(9.0), Vec2::new(10.0, 0.0), 3.0),
        ];
        let r = gather(center, &samples, GatherParams::default());
        assert!(approx_vec(r.color, center, 1.0e-4), "unreachable sample leaked: {:?}", r.color);
    }

    #[test]
    fn empty_samples_fall_back_to_center() {
        let center = Vec3::new(0.1, 0.2, 0.3);
        let r = gather(center, &[], GatherParams::default());
        assert!(approx_vec(r.color, center, 0.0));
        assert_eq!(r.weight, 1.0);
    }

    #[test]
    fn larger_disk_has_smaller_per_sample_weight() {
        // Energy conservation: a wider CoC spreads the same energy thinner, so
        // a single covering neighbour contributes less total weight.
        let near = alloc::vec![GatherSample::new(Vec3::ONE, Vec2::ZERO, 4.0)];
        let far = alloc::vec![GatherSample::new(Vec3::ONE, Vec2::ZERO, 16.0)];
        let wn = gather(Vec3::ZERO, &near, GatherParams::default()).weight;
        let wf = gather(Vec3::ZERO, &far, GatherParams::default()).weight;
        assert!(wn > wf, "small disk {wn} should weigh more than large disk {wf}");
    }

    #[test]
    fn edge_boost_increases_rim_sample_weight() {
        // A neighbour whose centre-distance is near its rim should gain weight
        // when rim brightening is enabled.
        let s = alloc::vec![GatherSample::new(Vec3::ONE, Vec2::new(9.5, 0.0), 10.0)];
        let flat = gather(Vec3::ZERO, &s, GatherParams { edge_softness: 0.0, edge_boost: 0.0 });
        let rim = gather(Vec3::ZERO, &s, GatherParams { edge_softness: 0.0, edge_boost: 2.0 });
        assert!(rim.weight > flat.weight, "rim boost should raise weight");
    }

    #[test]
    fn results_are_finite_for_adversarial_input() {
        let samples = alloc::vec![
            GatherSample::new(Vec3::new(f32::NAN, 1.0, 2.0), Vec2::new(f32::INFINITY, 0.0), 5.0),
            GatherSample::new(Vec3::splat(1.0), Vec2::new(1.0, 1.0), f32::NAN),
            GatherSample::new(Vec3::splat(1.0), Vec2::ZERO, -8.0),
        ];
        let r = gather(Vec3::new(f32::NAN, 0.0, 0.0), &samples, GatherParams::default());
        assert!(r.color.is_finite(), "color not finite: {:?}", r.color);
        assert!(r.weight.is_finite() && r.weight > 0.0);
    }
}
