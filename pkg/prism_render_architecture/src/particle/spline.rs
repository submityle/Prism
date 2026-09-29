//! `CPU` reference geometry for the `SampleSpline` `DataInterface` (design §8.3).
//!
//! This module is the `CPU`-verifiable maths behind the fieldless
//! [`super::modules::BuiltinDataInterface::SampleSpline`] catalog entry: it
//! backs the two function names that entry advertises
//! ([`super::modules`]'s `F_SPLINE` = `sample_spline_point` /
//! `sample_spline_tangent`) and the `spline_control_points` buffer binding
//! (`B_SPLINE`). Where that registry only declares *that* a spline is bound and
//! *which* `sample_*` names a module may call, this file provides the actual
//! evaluation a future `GPU` spline kernel must agree with, and the extra
//! spatial tooling a spawn/ribbon stage needs on the `CPU` path.
//!
//! # Relationship to [`super::curves`]
//!
//! [`super::curves`] and this module both say "spline", but they live in
//! different spaces and serve different jobs, and deliberately share no code:
//!
//! * [`super::curves`] is a **one-dimensional, over-life** layer: it evaluates
//!   authored *scalar* and *colour* ramps against a particle's normalized age
//!   (`Hermite`/`Catmull-Rom`/`Bezier` reconstruction of `f32`/`RGBA` values)
//!   and bakes them into 1D look-up tables (`LUT`s). Its domain is *time*; its
//!   codomain is a scalar or a colour.
//! * This module is a **three-dimensional, spatial-path** layer: it evaluates a
//!   [`Spline`] of [`Vec3`] control points into world-space *points*,
//!   *tangents*, *curvature*, arc-length parameterization, and
//!   rotation-minimizing orientation frames. Its domain is a path parameter;
//!   its codomain is a position/orientation in space.
//!
//! The one-dimensional reconstruction logic in [`super::curves`] is **not**
//! reused here; the two evaluators are independent by design so neither
//! constrains the other's numeric contract.
//!
//! # Algorithmic lineage
//!
//! The public surface mirrors production `VFX` spline tooling — Unreal
//! `Niagara`'s "Spawn Along Spline" / "Sample Spline" modules and the
//! `USplineComponent` reparameterization it reads — without reusing any of
//! their code. Arc-length parameterization follows the textbook prefix-sum /
//! inverse-lookup recipe, and the orientation frames use the *double
//! reflection* rotation-minimizing-frame (`RMF`) method of Wang, Jüttler, Zheng
//! and Liu (2008), which is attractive here precisely because it needs only
//! vector reflections.
//!
//! # Determinism
//!
//! Every routine uses ordinary `f32` arithmetic, `sqrt` (the sole permitted
//! transcendental, via [`Vec3`] length/normalize), and `f32::floor` (to locate
//! an arc-length cell). No `sin`/`cos`/`exp`/`ln`/`pow` is ever called, so the
//! `CPU` reference stays bit-reproducible against a future `GPU` kernel and
//! matches the determinism contract of the sibling [`super::simulation`] and
//! [`super::curves`] modules.

use alloc::vec::Vec;

use super::emitter::UnitCursor;
use super::{Vec3, EPS_LEN_SQ};

/// Absolute tolerance for `f32` equality and denominator guards.
///
/// Comparisons never use a bare `==`/`!=` on floating point; a difference
/// smaller than `EPS` is treated as zero so no routine divides by (near) zero
/// or propagates `NaN`.
pub const EPS: f32 = 1e-6;

/// Number of straight-line chords sampled per spline segment when building the
/// arc-length table.
///
/// Higher resolution tightens the arc-length approximation (and therefore the
/// evenness of distance-based sampling) at a linear memory/time cost. `64`
/// keeps the equal-arc error well under a percent for the smooth modes while
/// staying cheap to bake.
const ARC_SAMPLES_PER_SEGMENT: usize = 64;

/// Internal marching resolution for a single [`Spline::frame_at`] query.
///
/// A single frame lookup re-runs the rotation-minimizing recurrence from the
/// spline start to the requested parameter; this is the number of double
/// reflection steps it takes over that sub-interval.
const FRAME_MARCH_STEPS: usize = 48;

/// How the control points of a [`Spline`] are reconstructed into a path.
///
/// The three families cover the spatial-path counterparts of what authoring
/// tools expose: a straight polyline, an interpolating `Catmull-Rom` spline
/// that passes through every control point, and a cubic `Bezier` chain whose
/// intermediate control points shape each segment.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SplineMode {
    /// Straight segments between consecutive control points (a polyline). Needs
    /// at least two points.
    #[default]
    Linear,
    /// Interpolating `Catmull-Rom`: the path passes smoothly through every
    /// control point, with endpoint tangents inferred from neighbours. Needs at
    /// least two points.
    CatmullRom,
    /// A cubic `Bezier` chain: control points are grouped as
    /// `[p0, p1, p2, p3, p4, p5, p6, ...]`, where each segment `s` uses
    /// `[3s, 3s+1, 3s+2, 3s+3]` (adjacent segments share an anchor). An open
    /// chain needs `3k + 1` points (at least four); a closed chain expects a
    /// multiple of three and wraps the final anchor back to the first.
    Bezier,
}

/// An orthonormal orientation frame carried along a [`Spline`].
///
/// The three axes are mutually perpendicular unit vectors forming a
/// right-handed basis (`binormal = tangent × normal`). A sequence of these,
/// produced by [`Spline::rmf_frames`], is what a ribbon/beam renderer sweeps to
/// orient its quads without the twist "flip" a naive Frenet frame suffers at
/// inflection points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// Unit path direction (matches the spline's normalized tangent).
    pub tangent: Vec3,
    /// Unit normal, rotation-minimized along the path.
    pub normal: Vec3,
    /// Unit binormal, equal to `tangent × normal`.
    pub binormal: Vec3,
}

/// One rotation-minimizing frame sampled at a specific point on the spline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FramePoint {
    /// Global path parameter in `0..=1` for this frame.
    pub param: f32,
    /// World-space position on the spline at [`FramePoint::param`].
    pub position: Vec3,
    /// The orthonormal orientation frame at this point.
    pub frame: Frame,
}

/// A full sample produced by [`sample_along_spline`]: everything a spawn stage
/// needs to place and orient one particle on the path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplineSample {
    /// World-space spawn position on the spline.
    pub position: Vec3,
    /// Unit tangent (emission direction) at the spawn point.
    pub tangent: Vec3,
    /// Rotation-minimizing orientation frame at the spawn point.
    pub frame: Frame,
    /// Global path parameter in `0..=1` of the spawn point.
    pub param: f32,
    /// Arc-length distance from the spline start to the spawn point.
    pub distance: f32,
}

/// A control-point spline in 3D space with an arc-length reparameterization.
///
/// Construct with [`Spline::new`]; the arc-length table is baked once at
/// construction so distance queries are `O(log n)` afterwards. The path is
/// evaluated by a global parameter `t` in `0..=1` (spread evenly across the
/// segments) via [`Spline::point_at`]/[`Spline::tangent_at`], or by physical
/// arc length via [`Spline::sample_by_distance`].
#[derive(Clone, Debug, PartialEq)]
pub struct Spline {
    points: Vec<Vec3>,
    mode: SplineMode,
    closed: bool,
    /// Global parameters sampled when baking the arc-length table.
    arc_params: Vec<f32>,
    /// Cumulative arc length aligned with [`Spline::arc_params`].
    arc_lengths: Vec<f32>,
}

/// Clamps `x` into `[lo, hi]` without `f32::clamp`'s `lo <= hi` panic risk.
#[must_use]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Reflects the single vector reused by both halves of the double-reflection
/// `RMF` step: `v` mirrored across the plane with unit-scaled axis `axis`
/// (whose squared length is `axis_sq`). A (near) zero axis leaves `v` intact.
#[must_use]
fn reflect(v: Vec3, axis: Vec3, axis_sq: f32) -> Vec3 {
    if axis_sq > EPS_LEN_SQ {
        v.sub(axis.scale(2.0 * axis.dot(v) / axis_sq))
    } else {
        v
    }
}

/// Builds a unit vector perpendicular to `tangent`, seeded from `up`.
///
/// `up` is projected onto the plane perpendicular to `tangent`; if that
/// degenerates (because `up` is (anti)parallel to `tangent`, or `tangent` is
/// zero), successive world axes are tried so a usable reference always results.
#[must_use]
fn orthonormal_reference(tangent: Vec3, up: Vec3) -> Vec3 {
    let candidates = [
        up,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    for candidate in candidates {
        let projected = candidate.sub(tangent.scale(candidate.dot(tangent)));
        let unit = projected.normalize_or_zero();
        // A successful projection normalizes to (near) unit length; a failed
        // one collapses to zero via `normalize_or_zero`.
        if unit.length_squared() > 0.5 {
            return unit;
        }
    }
    Vec3::ZERO
}

/// One step of the Wang et al. double-reflection `RMF` recurrence: transports
/// the reference normal `r` from `(x0, t0)` to `(x1, t1)` and renormalizes.
#[must_use]
fn rmf_step(r: Vec3, x0: Vec3, t0: Vec3, x1: Vec3, t1: Vec3) -> Vec3 {
    let v1 = x1.sub(x0);
    let c1 = v1.length_squared();
    let r_l = reflect(r, v1, c1);
    let t_l = reflect(t0, v1, c1);
    let v2 = t1.sub(t_l);
    let c2 = v2.length_squared();
    reflect(r_l, v2, c2).normalize_or_zero()
}

impl Spline {
    /// Builds a spline from `points` under `mode`, optionally `closed`, and
    /// bakes its arc-length table.
    ///
    /// Degenerate inputs (too few points for the mode) are accepted and simply
    /// yield a zero-length path: every query then returns the first control
    /// point (or [`Vec3::ZERO`] when there are none) rather than panicking.
    #[must_use]
    pub fn new(points: Vec<Vec3>, mode: SplineMode, closed: bool) -> Self {
        let mut spline = Self {
            points,
            mode,
            closed,
            arc_params: Vec::new(),
            arc_lengths: Vec::new(),
        };
        spline.rebuild_arc_table();
        spline
    }

    /// The control points backing this spline.
    #[must_use]
    pub fn control_points(&self) -> &[Vec3] {
        &self.points
    }

    /// The reconstruction mode.
    #[must_use]
    pub fn mode(&self) -> SplineMode {
        self.mode
    }

    /// Whether the path wraps from its last control point back to its first.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Number of control points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// `true` when the spline has no control points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Number of path segments for the current mode, closure, and point count.
    ///
    /// Returns `0` when there are too few points to form a single segment, which
    /// is how the degenerate-input contract is enforced downstream.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        let n = self.points.len();
        match self.mode {
            SplineMode::Linear | SplineMode::CatmullRom => {
                if n < 2 {
                    0
                } else if self.closed {
                    n
                } else {
                    n - 1
                }
            }
            SplineMode::Bezier => {
                if self.closed {
                    if n < 3 {
                        0
                    } else {
                        n / 3
                    }
                } else if n < 4 {
                    0
                } else {
                    (n - 1) / 3
                }
            }
        }
    }

    /// Maps a global parameter `t` in `0..=1` to a `(segment, local u)` pair.
    ///
    /// The parameter is clamped, then spread evenly across the segments; `t == 1`
    /// lands on the end of the final segment (`u == 1`).
    #[must_use]
    fn locate(&self, t: f32, segments: usize) -> (usize, f32) {
        let clamped = clamp(t, 0.0, 1.0);
        let scaled = clamped * segments as f32;
        let floored = scaled.floor();
        let seg = floored as usize;
        if seg >= segments {
            (segments - 1, 1.0)
        } else {
            (seg, scaled - floored)
        }
    }

    /// Fetches a `Catmull-Rom` control point with end clamping (open) or index
    /// wrapping (closed).
    #[must_use]
    fn cr_ctrl(&self, i: isize) -> Vec3 {
        let n = self.points.len() as isize;
        let idx = if self.closed {
            ((i % n) + n) % n
        } else {
            i.clamp(0, n - 1)
        };
        self.points[idx as usize]
    }

    /// Fetches the `k`-th control point of `Bezier` segment starting at `base`,
    /// wrapping the index when the chain is closed.
    #[must_use]
    fn bez_ctrl(&self, base: usize, k: usize) -> Vec3 {
        let raw = base + k;
        let idx = if self.closed {
            raw % self.points.len()
        } else {
            raw
        };
        self.points[idx]
    }

    /// Evaluates the world-space point at global parameter `t` in `0..=1`.
    ///
    /// Returns the first control point (or [`Vec3::ZERO`]) for a degenerate
    /// spline so it is always safe to call.
    #[must_use]
    pub fn point_at(&self, t: f32) -> Vec3 {
        let segments = self.segment_count();
        if segments == 0 {
            return self.points.first().copied().unwrap_or(Vec3::ZERO);
        }
        let (seg, u) = self.locate(t, segments);
        match self.mode {
            SplineMode::Linear => self.eval_linear(seg, u),
            SplineMode::CatmullRom => self.eval_catmull(seg, u),
            SplineMode::Bezier => self.eval_bezier(seg, u),
        }
    }

    /// Evaluates the segment-local first derivative (`d/du`) at parameter `t`.
    ///
    /// This is the raw, **unnormalized** velocity of the current segment; its
    /// direction is the path tangent and its magnitude carries the segment's
    /// parameter speed (useful for curvature). Use [`Spline::tangent_at`] for a
    /// unit tangent.
    #[must_use]
    pub fn derivative_at(&self, t: f32) -> Vec3 {
        let segments = self.segment_count();
        if segments == 0 {
            return Vec3::ZERO;
        }
        let (seg, u) = self.locate(t, segments);
        match self.mode {
            SplineMode::Linear => self.eval_linear_deriv(seg),
            SplineMode::CatmullRom => self.eval_catmull_deriv(seg, u),
            SplineMode::Bezier => self.eval_bezier_deriv(seg, u),
        }
    }

    /// Evaluates the segment-local second derivative (`d²/du²`) at `t`.
    #[must_use]
    pub fn second_derivative_at(&self, t: f32) -> Vec3 {
        let segments = self.segment_count();
        if segments == 0 {
            return Vec3::ZERO;
        }
        let (seg, u) = self.locate(t, segments);
        match self.mode {
            SplineMode::Linear => Vec3::ZERO,
            SplineMode::CatmullRom => self.eval_catmull_second(seg, u),
            SplineMode::Bezier => self.eval_bezier_second(seg, u),
        }
    }

    /// The **unit** tangent (path direction) at global parameter `t`.
    #[must_use]
    pub fn tangent_at(&self, t: f32) -> Vec3 {
        self.derivative_at(t).normalize_or_zero()
    }

    /// The geometric curvature `κ = |r' × r''| / |r'|³` at `t`.
    ///
    /// Curvature is invariant under the segment-local parameterization, so the
    /// `d/du` derivatives above give the true spatial curvature; a straight path
    /// returns `0`.
    #[must_use]
    pub fn curvature_at(&self, t: f32) -> f32 {
        let d1 = self.derivative_at(t);
        let d2 = self.second_derivative_at(t);
        let speed = d1.length();
        let denom = speed * speed * speed;
        if denom > EPS {
            d1.cross(d2).length() / denom
        } else {
            0.0
        }
    }

    /// Linear segment point: straight blend between the two anchors.
    #[must_use]
    fn eval_linear(&self, seg: usize, u: f32) -> Vec3 {
        let n = self.points.len();
        let a = self.points[seg];
        let b = self.points[(seg + 1) % n];
        a.add(b.sub(a).scale(u))
    }

    /// Linear segment derivative: the constant chord vector.
    #[must_use]
    fn eval_linear_deriv(&self, seg: usize) -> Vec3 {
        let n = self.points.len();
        self.points[(seg + 1) % n].sub(self.points[seg])
    }

    /// `Catmull-Rom` segment point using the standard uniform basis.
    #[must_use]
    fn eval_catmull(&self, seg: usize, u: f32) -> Vec3 {
        let s = seg as isize;
        let p0 = self.cr_ctrl(s - 1);
        let p1 = self.cr_ctrl(s);
        let p2 = self.cr_ctrl(s + 1);
        let p3 = self.cr_ctrl(s + 2);
        let u2 = u * u;
        let u3 = u2 * u;
        // 0.5 * (2P1 + (-P0+P2)u + (2P0-5P1+4P2-P3)u^2 + (-P0+3P1-3P2+P3)u^3)
        let c0 = p1.scale(2.0);
        let c1 = p2.sub(p0);
        let c2 = p0.scale(2.0).sub(p1.scale(5.0)).add(p2.scale(4.0)).sub(p3);
        let c3 = p1.scale(3.0).sub(p0).sub(p2.scale(3.0)).add(p3);
        c0.add(c1.scale(u))
            .add(c2.scale(u2))
            .add(c3.scale(u3))
            .scale(0.5)
    }

    /// `Catmull-Rom` segment first derivative (`d/du`).
    #[must_use]
    fn eval_catmull_deriv(&self, seg: usize, u: f32) -> Vec3 {
        let s = seg as isize;
        let p0 = self.cr_ctrl(s - 1);
        let p1 = self.cr_ctrl(s);
        let p2 = self.cr_ctrl(s + 1);
        let p3 = self.cr_ctrl(s + 2);
        let u2 = u * u;
        let c1 = p2.sub(p0);
        let c2 = p0.scale(2.0).sub(p1.scale(5.0)).add(p2.scale(4.0)).sub(p3);
        let c3 = p1.scale(3.0).sub(p0).sub(p2.scale(3.0)).add(p3);
        c1.add(c2.scale(2.0 * u)).add(c3.scale(3.0 * u2)).scale(0.5)
    }

    /// `Catmull-Rom` segment second derivative (`d²/du²`).
    #[must_use]
    fn eval_catmull_second(&self, seg: usize, u: f32) -> Vec3 {
        // point = 0.5*(2P1 + c1 u + c2 u^2 + c3 u^3), so the exact second
        // derivative is 0.5*(2 c2 + 6 c3 u) = c2 + 3 c3 u (linear in `u`).
        let s = seg as isize;
        let p0 = self.cr_ctrl(s - 1);
        let p1 = self.cr_ctrl(s);
        let p2 = self.cr_ctrl(s + 1);
        let p3 = self.cr_ctrl(s + 2);
        let c2 = p0.scale(2.0).sub(p1.scale(5.0)).add(p2.scale(4.0)).sub(p3);
        let c3 = p1.scale(3.0).sub(p0).sub(p2.scale(3.0)).add(p3);
        c2.add(c3.scale(3.0 * u))
    }

    /// Cubic `Bezier` segment point via the Bernstein basis.
    #[must_use]
    fn eval_bezier(&self, seg: usize, u: f32) -> Vec3 {
        let base = 3 * seg;
        let p0 = self.bez_ctrl(base, 0);
        let p1 = self.bez_ctrl(base, 1);
        let p2 = self.bez_ctrl(base, 2);
        let p3 = self.bez_ctrl(base, 3);
        let mu = 1.0 - u;
        let b0 = mu * mu * mu;
        let b1 = 3.0 * mu * mu * u;
        let b2 = 3.0 * mu * u * u;
        let b3 = u * u * u;
        p0.scale(b0)
            .add(p1.scale(b1))
            .add(p2.scale(b2))
            .add(p3.scale(b3))
    }

    /// Cubic `Bezier` segment first derivative (`d/du`).
    #[must_use]
    fn eval_bezier_deriv(&self, seg: usize, u: f32) -> Vec3 {
        let base = 3 * seg;
        let p0 = self.bez_ctrl(base, 0);
        let p1 = self.bez_ctrl(base, 1);
        let p2 = self.bez_ctrl(base, 2);
        let p3 = self.bez_ctrl(base, 3);
        let mu = 1.0 - u;
        // 3(1-u)^2 (P1-P0) + 6(1-u)u (P2-P1) + 3u^2 (P3-P2)
        p1.sub(p0)
            .scale(3.0 * mu * mu)
            .add(p2.sub(p1).scale(6.0 * mu * u))
            .add(p3.sub(p2).scale(3.0 * u * u))
    }

    /// Cubic `Bezier` segment second derivative (`d²/du²`).
    #[must_use]
    fn eval_bezier_second(&self, seg: usize, u: f32) -> Vec3 {
        let base = 3 * seg;
        let p0 = self.bez_ctrl(base, 0);
        let p1 = self.bez_ctrl(base, 1);
        let p2 = self.bez_ctrl(base, 2);
        let p3 = self.bez_ctrl(base, 3);
        let mu = 1.0 - u;
        // 6(1-u)(P2-2P1+P0) + 6u(P3-2P2+P1)
        let term0 = p2.sub(p1.scale(2.0)).add(p0).scale(6.0 * mu);
        let term1 = p3.sub(p2.scale(2.0)).add(p1).scale(6.0 * u);
        term0.add(term1)
    }

    /// Rebuilds the arc-length prefix-sum table from a dense chord sampling.
    fn rebuild_arc_table(&mut self) {
        let segments = self.segment_count();
        if segments == 0 {
            self.arc_params = Vec::from([0.0]);
            self.arc_lengths = Vec::from([0.0]);
            return;
        }
        let resolution = ARC_SAMPLES_PER_SEGMENT * segments;
        let count = resolution + 1;
        let mut params = Vec::with_capacity(count);
        let mut lengths = Vec::with_capacity(count);
        let mut prev = self.point_at(0.0);
        let mut accumulated = 0.0;
        params.push(0.0);
        lengths.push(0.0);
        for i in 1..count {
            let t = i as f32 / resolution as f32;
            let current = self.point_at(t);
            accumulated += current.distance(prev);
            params.push(t);
            lengths.push(accumulated);
            prev = current;
        }
        self.arc_params = params;
        self.arc_lengths = lengths;
    }

    /// Total arc length of the path (`0` for a degenerate spline).
    #[must_use]
    pub fn total_length(&self) -> f32 {
        self.arc_lengths.last().copied().unwrap_or(0.0)
    }

    /// Inverts the arc-length table: maps a distance `s` (clamped to
    /// `0..=total_length`) to the global parameter `t` reaching that distance.
    #[must_use]
    pub fn arc_to_param(&self, s: f32) -> f32 {
        let total = self.total_length();
        if total <= EPS {
            return 0.0;
        }
        let target = clamp(s, 0.0, total);
        // First index whose cumulative length exceeds `target`.
        let upper = self
            .arc_lengths
            .partition_point(|&length| length <= target)
            .clamp(1, self.arc_lengths.len() - 1);
        let lower = upper - 1;
        let lo_len = self.arc_lengths[lower];
        let hi_len = self.arc_lengths[upper];
        let lo_t = self.arc_params[lower];
        let hi_t = self.arc_params[upper];
        let span = hi_len - lo_len;
        if span > EPS {
            let frac = (target - lo_len) / span;
            lo_t + (hi_t - lo_t) * frac
        } else {
            lo_t
        }
    }

    /// Samples the world-space point a physical distance `s` along the path.
    #[must_use]
    pub fn sample_by_distance(&self, s: f32) -> Vec3 {
        self.point_at(self.arc_to_param(s))
    }

    /// Builds `count` rotation-minimizing frames evenly spaced in the global
    /// parameter, seeded from `up`.
    ///
    /// The frames are transported with the Wang et al. double-reflection `RMF`
    /// recurrence, so the normal does not spin around a straight run and never
    /// "flips" at an inflection — the property ribbon/beam renderers rely on.
    /// `count` is clamped to at least two so a start and end frame always exist.
    #[must_use]
    pub fn rmf_frames(&self, count: usize, up: Vec3) -> Vec<FramePoint> {
        let steps = count.max(2);
        let last = steps - 1;
        let mut out = Vec::with_capacity(steps);
        let mut position = self.point_at(0.0);
        let mut tangent = self.tangent_at(0.0);
        let mut normal = orthonormal_reference(tangent, up);
        out.push(FramePoint {
            param: 0.0,
            position,
            frame: frame_from(tangent, normal),
        });
        for i in 1..=last {
            let t = i as f32 / last as f32;
            let next_pos = self.point_at(t);
            let next_tan = self.tangent_at(t);
            normal = rmf_step(normal, position, tangent, next_pos, next_tan);
            position = next_pos;
            tangent = next_tan;
            out.push(FramePoint {
                param: t,
                position,
                frame: frame_from(tangent, normal),
            });
        }
        out
    }

    /// The rotation-minimizing frame at a single global parameter `t`.
    ///
    /// The recurrence is re-run from the spline start to `t` over a fixed number
    /// of sub-steps, so the result is consistent with the same-`up`
    /// [`Spline::rmf_frames`] sequence up to sampling resolution.
    #[must_use]
    pub fn frame_at(&self, t: f32, up: Vec3) -> Frame {
        let target = clamp(t, 0.0, 1.0);
        let mut position = self.point_at(0.0);
        let mut tangent = self.tangent_at(0.0);
        let mut normal = orthonormal_reference(tangent, up);
        if target <= EPS {
            return frame_from(tangent, normal);
        }
        for i in 1..=FRAME_MARCH_STEPS {
            let t_i = target * (i as f32 / FRAME_MARCH_STEPS as f32);
            let next_pos = self.point_at(t_i);
            let next_tan = self.tangent_at(t_i);
            normal = rmf_step(normal, position, tangent, next_pos, next_tan);
            position = next_pos;
            tangent = next_tan;
        }
        frame_from(tangent, normal)
    }

    /// Projects `p` onto the path, returning `(param, closest point, distance)`.
    ///
    /// A coarse uniform scan seeds a local ternary refinement (arithmetic only),
    /// which is robust for the smooth modes and exact enough for spawn snapping.
    #[must_use]
    pub fn closest_point(&self, p: Vec3) -> (f32, Vec3, f32) {
        let segments = self.segment_count();
        if segments == 0 {
            let point = self.points.first().copied().unwrap_or(Vec3::ZERO);
            return (0.0, point, p.distance(point));
        }
        let scan = (ARC_SAMPLES_PER_SEGMENT * segments).max(2);
        let mut best_t = 0.0;
        let mut best_d2 = f32::INFINITY;
        for i in 0..=scan {
            let t = i as f32 / scan as f32;
            let d2 = self.point_at(t).distance_squared(p);
            if d2 < best_d2 {
                best_d2 = d2;
                best_t = t;
            }
        }
        let half = 1.0 / scan as f32;
        let mut lo = clamp(best_t - half, 0.0, 1.0);
        let mut hi = clamp(best_t + half, 0.0, 1.0);
        for _ in 0..48 {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            if self.point_at(m1).distance_squared(p) < self.point_at(m2).distance_squared(p) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        let param = (lo + hi) * 0.5;
        let point = self.point_at(param);
        (param, point, p.distance(point))
    }
}

/// Assembles a right-handed orthonormal [`Frame`] from a tangent and a
/// (roughly perpendicular) normal, re-orthogonalizing to cancel drift.
#[must_use]
fn frame_from(tangent: Vec3, normal: Vec3) -> Frame {
    let binormal = tangent.cross(normal).normalize_or_zero();
    // Re-derive the normal from the cleaned binormal so the basis stays exactly
    // orthonormal even after accumulated floating-point error.
    let ortho_normal = binormal.cross(tangent).normalize_or_zero();
    Frame {
        tangent,
        normal: ortho_normal,
        binormal,
    }
}

/// Samples one spawn placement uniformly by **arc length** along `spline`,
/// consuming a single unit sample from `cursor`.
///
/// Drawing the placement in arc-length space (rather than raw parameter space)
/// gives an even spatial density along the path — the behaviour Unreal
/// `Niagara`'s "Spawn Along Spline" expects — because the arc-length table
/// straightens out the non-uniform parameter speed of the smooth modes. The
/// result is fully determined by the cursor stream, so it agrees with a `GPU`
/// spawn kernel reading the same hash-RNG samples.
#[must_use]
pub fn sample_along_spline(spline: &Spline, cursor: &mut UnitCursor<'_>) -> SplineSample {
    let unit = cursor.next_unit();
    let total = spline.total_length();
    let distance = unit * total;
    let param = spline.arc_to_param(distance);
    let position = spline.point_at(param);
    let tangent = spline.tangent_at(param);
    let frame = spline.frame_at(param, Vec3::new(0.0, 1.0, 0.0));
    SplineSample {
        position,
        tangent,
        frame,
        param,
        distance,
    }
}

/// `CPU` reference for the `sample_spline_point` function the `SampleSpline`
/// `DataInterface` advertises: the world-space point at parameter `t`.
#[must_use]
pub fn sample_spline_point(spline: &Spline, t: f32) -> Vec3 {
    spline.point_at(t)
}

/// `CPU` reference for the `sample_spline_tangent` function the `SampleSpline`
/// `DataInterface` advertises: the unit tangent at parameter `t`.
#[must_use]
pub fn sample_spline_tangent(spline: &Spline, t: f32) -> Vec3 {
    spline.tangent_at(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for approximate `f32` geometry assertions.
    const TOL: f32 = 1e-4;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_vec(a: Vec3, b: Vec3, eps: f32) -> bool {
        approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
    }

    fn straight_line() -> Spline {
        Spline::new(
            Vec::from([Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)]),
            SplineMode::Linear,
            false,
        )
    }

    #[test]
    fn linear_point_and_tangent() {
        let s = straight_line();
        assert!(approx_vec(s.point_at(0.0), Vec3::new(0.0, 0.0, 0.0), TOL));
        assert!(approx_vec(s.point_at(0.5), Vec3::new(5.0, 0.0, 0.0), TOL));
        assert!(approx_vec(s.point_at(1.0), Vec3::new(10.0, 0.0, 0.0), TOL));
        assert!(approx_vec(s.tangent_at(0.5), Vec3::new(1.0, 0.0, 0.0), TOL));
    }

    #[test]
    fn linear_closed_wraps_back_to_start() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
            ]),
            SplineMode::Linear,
            true,
        );
        assert_eq!(s.segment_count(), 3);
        // Perimeter = 2 + 2 + distance((2,2)->(0,0)) = 4 + sqrt(8).
        let expected = 4.0 + 8.0_f32.sqrt();
        assert!(approx(s.total_length(), expected, 1e-2));
        // The last segment returns to the first control point at t = 1.
        assert!(approx_vec(s.point_at(1.0), Vec3::new(0.0, 0.0, 0.0), TOL));
    }

    #[test]
    fn catmull_passes_through_control_points() {
        let pts = Vec::from([
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(3.0, 2.0, 1.0),
            Vec3::new(4.0, 0.0, 0.0),
        ]);
        let s = Spline::new(pts.clone(), SplineMode::CatmullRom, false);
        let segs = s.segment_count() as f32;
        for (k, expected) in pts.iter().enumerate() {
            let t = k as f32 / segs;
            assert!(
                approx_vec(s.point_at(t), *expected, TOL),
                "catmull did not interpolate control point {k}"
            );
        }
    }

    #[test]
    fn catmull_tangent_matches_finite_difference() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 2.0, 0.0),
                Vec3::new(3.0, 2.0, 1.0),
                Vec3::new(4.0, 0.0, 0.0),
            ]),
            SplineMode::CatmullRom,
            false,
        );
        let t = 0.4;
        let h = 1e-4;
        let fd = s.point_at(t + h).sub(s.point_at(t - h)).normalize_or_zero();
        assert!(approx_vec(s.tangent_at(t), fd, 1e-2));
    }

    #[test]
    fn bezier_endpoints_and_derivative() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
            ]),
            SplineMode::Bezier,
            false,
        );
        assert_eq!(s.segment_count(), 1);
        assert!(approx_vec(s.point_at(0.0), Vec3::new(0.0, 0.0, 0.0), TOL));
        assert!(approx_vec(s.point_at(1.0), Vec3::new(1.0, 0.0, 0.0), TOL));
        // At t=0 the tangent points toward the first handle (+Y).
        assert!(approx_vec(s.tangent_at(0.0), Vec3::new(0.0, 1.0, 0.0), TOL));
        // Symmetric curve: the midpoint sits at x = 0.5.
        assert!(approx(s.point_at(0.5).x, 0.5, TOL));
    }

    #[test]
    fn bezier_multi_segment_shares_anchor() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(4.0, 0.0, 0.0),
                Vec3::new(5.0, 0.0, 0.0),
                Vec3::new(6.0, 0.0, 0.0),
            ]),
            SplineMode::Bezier,
            false,
        );
        assert_eq!(s.segment_count(), 2);
        // Shared anchor (index 3) is reached at the segment boundary t = 0.5.
        assert!(approx_vec(s.point_at(0.5), Vec3::new(3.0, 0.0, 0.0), TOL));
    }

    #[test]
    fn total_length_straight_line_is_exact() {
        assert!(approx(straight_line().total_length(), 10.0, TOL));
    }

    #[test]
    fn total_length_polyline_is_exact() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(3.0, 4.0, 0.0),
            ]),
            SplineMode::Linear,
            false,
        );
        // Collinear chord sampling sums to the exact polyline length 3 + 4 = 7.
        assert!(approx(s.total_length(), 7.0, TOL));
    }

    #[test]
    fn arc_to_param_endpoints_and_monotonic() {
        let s = straight_line();
        assert!(approx(s.arc_to_param(0.0), 0.0, TOL));
        assert!(approx(s.arc_to_param(10.0), 1.0, TOL));
        assert!(approx(s.arc_to_param(-5.0), 0.0, TOL));
        assert!(approx(s.arc_to_param(999.0), 1.0, TOL));
        let mut prev = -1.0;
        for i in 0..=20 {
            let t = s.arc_to_param(i as f32 * 0.5);
            assert!(t >= prev - TOL, "arc_to_param not monotonic");
            prev = t;
        }
    }

    #[test]
    fn sample_by_distance_is_equidistant() {
        let s = straight_line();
        let l = s.total_length();
        let quarter = s.sample_by_distance(l * 0.25);
        let half = s.sample_by_distance(l * 0.5);
        let three_q = s.sample_by_distance(l * 0.75);
        assert!(approx_vec(quarter, Vec3::new(2.5, 0.0, 0.0), TOL));
        assert!(approx_vec(half, Vec3::new(5.0, 0.0, 0.0), TOL));
        assert!(approx_vec(three_q, Vec3::new(7.5, 0.0, 0.0), TOL));
    }

    #[test]
    fn sample_by_distance_equidistant_on_curve() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 2.0, 0.0),
                Vec3::new(3.0, 2.0, 0.0),
                Vec3::new(4.0, 0.0, 0.0),
            ]),
            SplineMode::CatmullRom,
            false,
        );
        let l = s.total_length();
        let steps = 16;
        let expected = l / steps as f32;
        let mut prev = s.sample_by_distance(0.0);
        for i in 1..=steps {
            let cur = s.sample_by_distance(l * (i as f32 / steps as f32));
            let gap = cur.distance(prev);
            // Equal-arc spacing: each chord is within a few percent of L/steps.
            assert!(
                approx(gap, expected, expected * 0.05),
                "arc spacing {gap} deviated from {expected}"
            );
            prev = cur;
        }
    }

    #[test]
    fn rmf_frames_are_orthonormal() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 3.0, 1.0),
                Vec3::new(5.0, 1.0, 4.0),
                Vec3::new(7.0, -2.0, 2.0),
            ]),
            SplineMode::CatmullRom,
            false,
        );
        let frames = s.rmf_frames(24, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(frames.len(), 24);
        for fp in &frames {
            let f = fp.frame;
            assert!(approx(f.tangent.length(), 1.0, 1e-3));
            assert!(approx(f.normal.length(), 1.0, 1e-3));
            assert!(approx(f.binormal.length(), 1.0, 1e-3));
            assert!(approx(f.tangent.dot(f.normal), 0.0, 1e-3));
            assert!(approx(f.tangent.dot(f.binormal), 0.0, 1e-3));
            assert!(approx(f.normal.dot(f.binormal), 0.0, 1e-3));
        }
    }

    #[test]
    fn rmf_tangent_matches_spline_tangent() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(3.0, 0.0, 1.0),
                Vec3::new(4.0, 1.0, 2.0),
            ]),
            SplineMode::CatmullRom,
            false,
        );
        for fp in &s.rmf_frames(12, Vec3::new(0.0, 1.0, 0.0)) {
            let expected = s.tangent_at(fp.param);
            assert!(
                approx_vec(fp.frame.tangent, expected, 1e-3),
                "frame tangent diverged from spline tangent at t = {}",
                fp.param
            );
        }
    }

    #[test]
    fn rmf_does_not_rotate_along_a_straight_line() {
        // On a straight run the rotation-minimizing normal must stay constant.
        let s = Spline::new(
            Vec::from([Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 10.0)]),
            SplineMode::Linear,
            false,
        );
        let frames = s.rmf_frames(16, Vec3::new(0.0, 1.0, 0.0));
        let first = frames[0].frame.normal;
        for fp in &frames {
            assert!(
                approx_vec(fp.frame.normal, first, 1e-4),
                "normal rotated on a straight line at t = {}",
                fp.param
            );
        }
    }

    #[test]
    fn frame_at_is_orthonormal() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 2.0, 0.0),
                Vec3::new(4.0, 0.0, 2.0),
            ]),
            SplineMode::CatmullRom,
            false,
        );
        let f = s.frame_at(0.5, Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(f.tangent.dot(f.normal), 0.0, 1e-3));
        assert!(approx(f.tangent.dot(f.binormal), 0.0, 1e-3));
        assert!(approx(f.normal.dot(f.binormal), 0.0, 1e-3));
        assert!(approx(f.normal.length(), 1.0, 1e-3));
    }

    #[test]
    fn sample_along_spline_is_arc_uniform() {
        let s = straight_line();
        let l = s.total_length();
        let samples = [0.0_f32, 0.25, 0.5, 0.75];
        let mut cursor = UnitCursor::new(&samples);
        for &u in &samples {
            let sample = sample_along_spline(&s, &mut cursor);
            assert!(approx(sample.distance, u * l, 1e-2));
            assert!(approx(sample.position.x, u * l, 1e-2));
            assert!(approx(sample.param, u, 1e-2));
        }
    }

    #[test]
    fn sample_along_spline_carries_orthonormal_frame() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
                Vec3::new(4.0, 0.0, 1.0),
            ]),
            SplineMode::CatmullRom,
            false,
        );
        let samples = [0.3_f32];
        let mut cursor = UnitCursor::new(&samples);
        let out = sample_along_spline(&s, &mut cursor);
        assert!(approx(out.frame.tangent.dot(out.frame.normal), 0.0, 1e-3));
        assert!(approx_vec(out.tangent, out.frame.tangent, 1e-3));
    }

    #[test]
    fn closest_point_projects_onto_segment() {
        let s = straight_line();
        let (param, point, dist) = s.closest_point(Vec3::new(5.0, 3.0, 0.0));
        assert!(approx(param, 0.5, 1e-3));
        assert!(approx_vec(point, Vec3::new(5.0, 0.0, 0.0), 1e-3));
        assert!(approx(dist, 3.0, 1e-3));
    }

    #[test]
    fn closest_point_snaps_to_endpoint() {
        let s = straight_line();
        let (param, point, _) = s.closest_point(Vec3::new(-4.0, 0.0, 0.0));
        assert!(approx(param, 0.0, 1e-3));
        assert!(approx_vec(point, Vec3::new(0.0, 0.0, 0.0), 1e-3));
    }

    #[test]
    fn curvature_is_zero_on_a_straight_line() {
        let s = straight_line();
        assert!(approx(s.curvature_at(0.5), 0.0, 1e-4));
    }

    #[test]
    fn curvature_positive_on_a_bend() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, -1.0, 0.0),
            ]),
            SplineMode::CatmullRom,
            false,
        );
        assert!(s.curvature_at(0.5) > 0.0);
    }

    #[test]
    fn top_level_point_and_tangent_match_methods() {
        let s = Spline::new(
            Vec::from([
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 2.0, 0.0),
                Vec3::new(3.0, 2.0, 1.0),
                Vec3::new(4.0, 0.0, 0.0),
            ]),
            SplineMode::CatmullRom,
            false,
        );
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            assert!(approx_vec(sample_spline_point(&s, t), s.point_at(t), TOL));
            assert!(approx_vec(
                sample_spline_tangent(&s, t),
                s.tangent_at(t),
                TOL
            ));
        }
    }

    #[test]
    fn sample_spline_tangent_is_unit_length() {
        let s = straight_line();
        assert!(approx(sample_spline_tangent(&s, 0.7).length(), 1.0, TOL));
    }

    #[test]
    fn empty_spline_is_safe() {
        let s = Spline::new(Vec::new(), SplineMode::Linear, false);
        assert!(s.is_empty());
        assert_eq!(s.segment_count(), 0);
        assert!(approx(s.total_length(), 0.0, TOL));
        assert!(approx_vec(s.point_at(0.5), Vec3::ZERO, TOL));
        assert!(approx_vec(s.tangent_at(0.5), Vec3::ZERO, TOL));
        let (_, point, _) = s.closest_point(Vec3::new(1.0, 1.0, 1.0));
        assert!(approx_vec(point, Vec3::ZERO, TOL));
    }

    #[test]
    fn single_point_spline_is_safe() {
        let s = Spline::new(
            Vec::from([Vec3::new(2.0, 3.0, 4.0)]),
            SplineMode::Linear,
            false,
        );
        assert_eq!(s.segment_count(), 0);
        assert!(approx_vec(s.point_at(0.5), Vec3::new(2.0, 3.0, 4.0), TOL));
        let (_, point, dist) = s.closest_point(Vec3::new(2.0, 3.0, 5.0));
        assert!(approx_vec(point, Vec3::new(2.0, 3.0, 4.0), TOL));
        assert!(approx(dist, 1.0, TOL));
    }

    #[test]
    fn spline_mode_default_is_linear() {
        assert_eq!(SplineMode::default(), SplineMode::Linear);
    }

    #[test]
    fn accessors_report_construction() {
        let s = Spline::new(
            Vec::from([Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)]),
            SplineMode::CatmullRom,
            true,
        );
        assert_eq!(s.point_count(), 2);
        assert_eq!(s.mode(), SplineMode::CatmullRom);
        assert!(s.is_closed());
        assert_eq!(s.control_points().len(), 2);
    }
}
