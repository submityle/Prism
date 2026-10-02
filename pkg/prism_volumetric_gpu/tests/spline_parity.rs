//! Real-device parity for the per-`t` spline twin:
//! [`GpuSpline`](prism_volumetric_gpu::spline::GpuSpline) must reproduce the
//! `CPU` golden
//! [`spline`](prism_render_architecture::particle::spline) for the five
//! stateless, per-`t` local evaluations it exposes — the world point
//! ([`Spline::point_at`](prism_render_architecture::particle::spline::Spline::point_at)),
//! the segment-local first and second derivatives
//! ([`Spline::derivative_at`](prism_render_architecture::particle::spline::Spline::derivative_at),
//! [`Spline::second_derivative_at`](prism_render_architecture::particle::spline::Spline::second_derivative_at)),
//! the unit tangent
//! ([`Spline::tangent_at`](prism_render_architecture::particle::spline::Spline::tangent_at))
//! and the geometric curvature
//! ([`Spline::curvature_at`](prism_render_architecture::particle::spline::Spline::curvature_at)).
//!
//! The fixtures cover all three
//! [`SplineMode`](prism_render_architecture::particle::spline::SplineMode)
//! families (`Linear`, `Catmull-Rom`, `Bezier`), both open and closed chains,
//! and point counts of `2`, `3`, `4`, `6` and `7`, each sampled at the segment
//! ends and interior (`t` of `0`, `0.17`, `0.5`, `0.83`, `1.0`). Every control
//! point is a distinct, well-spread coordinate so no segment collapses: the
//! first derivative stays far from zero, keeping the unit tangent and the
//! curvature denominator clear of their degenerate guards. A separate test
//! drives the explicit degenerate branch (`segment_count == 0`) where the point
//! pins to the first control point and every derivative, tangent and curvature
//! is zero. All queries are flattened into one shared control-point buffer and
//! one dispatch, exercising the one-thread-per-query path.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The positions, derivatives, tangents and curvatures thread through
//! multiplies, adds, one `sqrt` and one guarded division, so they are compared
//! under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR =
//! 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::spline`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::spline::{Spline, SplineMode};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::spline::{
    GpuSpline, SplineEvalQuery, SplineEvalResult, SPLINE_BEZIER, SPLINE_CATMULL_ROM, SPLINE_LINEAR,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Global parameters sampled on every spline: the two segment ends and three
/// interior points, chosen clear of the curvature denominator's guard.
const T_SAMPLES: [f32; 5] = [0.0, 0.17, 0.5, 0.83, 1.0];

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of a `GPU` `[f32; 3]` lane triple against a golden
/// [`Vec3`].
fn approx3(a: [f32; 3], b: Vec3) -> bool {
    approx(a[0], b.x) && approx(a[1], b.y) && approx(a[2], b.z)
}

/// Deterministic host-side `u64` LCG (numerical-recipes constants) producing a
/// repeatable stream of simple-decimal coordinate jitter; no transcendental
/// math is involved.
struct Lcg {
    /// Current state word.
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// Advances the state and returns a signed `0.1`-step jitter in
    /// `[-0.3, 0.3]`, built from integer arithmetic so the value is an exact
    /// small decimal shared bit-for-bit between the `CPU` and `GPU` inputs.
    fn jitter(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 40) as u32;
        // Integer in 0..=6, centred to -3..=3, then scaled to a 0.1-step grid.
        let step = (bits % 7) as i32 - 3;
        step as f32 / 10.0
    }
}

/// Maps the golden [`SplineMode`] to the kernel's `u32` mode code.
fn mode_code(mode: SplineMode) -> u32 {
    match mode {
        SplineMode::Linear => SPLINE_LINEAR,
        SplineMode::CatmullRom => SPLINE_CATMULL_ROM,
        SplineMode::Bezier => SPLINE_BEZIER,
    }
}

/// One sampled comparison: the golden spline, the global parameter and the
/// index of its query in the flattened batch.
struct Case {
    /// Golden reference spline.
    spline: Spline,
    /// Global path parameter in `0..=1`.
    t: f32,
    /// Index of this `(spline, t)` query in the flattened batch.
    index: usize,
}

/// Accumulates control points, queries and golden cases for a single batched
/// dispatch.
struct Batch {
    /// Shared flattened `[x, y, z]` control points for every query.
    points: Vec<[f32; 3]>,
    /// One query per `(spline, t)` pair.
    queries: Vec<SplineEvalQuery>,
    /// Golden cases parallel to the queries.
    cases: Vec<Case>,
    /// Jitter source for the control-point coordinates.
    rng: Lcg,
}

impl Batch {
    /// Starts an empty batch seeded for deterministic coordinates.
    fn new(seed: u64) -> Batch {
        Batch {
            points: Vec::new(),
            queries: Vec::new(),
            cases: Vec::new(),
            rng: Lcg::new(seed),
        }
    }

    /// Builds `n` well-spread, distinct control points. Each point is a wide
    /// per-axis base plus small `0.1`-step jitter, so consecutive points never
    /// collapse and every segment keeps a non-degenerate first derivative.
    fn make_points(&mut self, n: usize) -> Vec<Vec3> {
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let fi = i as f32;
            let x = fi * 1.7 - 3.0 + self.rng.jitter();
            let y = fi * fi * 0.4 - 2.0 + self.rng.jitter();
            let z = fi * -1.3 + 1.0 + self.rng.jitter();
            pts.push(Vec3::new(x, y, z));
        }
        pts
    }

    /// Adds a spline of `n` control points under `mode`/`closed`, queued at
    /// every parameter in [`T_SAMPLES`].
    fn add(&mut self, n: usize, mode: SplineMode, closed: bool) {
        let pts = self.make_points(n);
        let offset = self.points.len() as u32;
        for p in &pts {
            self.points.push([p.x, p.y, p.z]);
        }
        for &t in &T_SAMPLES {
            let index = self.queries.len();
            self.queries.push(SplineEvalQuery {
                points_offset: offset,
                points_len: n as u32,
                mode: mode_code(mode),
                closed: u32::from(closed),
                t,
            });
            self.cases.push(Case {
                spline: Spline::new(pts.clone(), mode, closed),
                t,
                index,
            });
        }
    }
}

/// Asserts one `GPU` result matches the golden spline at the case's parameter.
fn assert_case(case: &Case, got: &SplineEvalResult) {
    let t = case.t;
    let cpu_point = case.spline.point_at(t);
    let cpu_deriv = case.spline.derivative_at(t);
    let cpu_second = case.spline.second_derivative_at(t);
    let cpu_tangent = case.spline.tangent_at(t);
    let cpu_curv = case.spline.curvature_at(t);
    assert!(
        approx3(got.position, cpu_point),
        "point mismatch at t={t}: gpu {:?} vs cpu {cpu_point:?}",
        got.position
    );
    assert!(
        approx3(got.derivative, cpu_deriv),
        "derivative mismatch at t={t}: gpu {:?} vs cpu {cpu_deriv:?}",
        got.derivative
    );
    assert!(
        approx3(got.second, cpu_second),
        "second-derivative mismatch at t={t}: gpu {:?} vs cpu {cpu_second:?}",
        got.second
    );
    assert!(
        approx3(got.tangent, cpu_tangent),
        "tangent mismatch at t={t}: gpu {:?} vs cpu {cpu_tangent:?}",
        got.tangent
    );
    assert!(
        approx(got.curvature, cpu_curv),
        "curvature mismatch at t={t}: gpu {} vs cpu {cpu_curv}",
        got.curvature
    );
}

#[test]
fn parity_across_modes_closures_and_point_counts() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpline::new(&ctx);
    let mut batch = Batch::new(0x5101_2357_9bdf_1113);
    // Linear polylines, open and closed, covering the minimum and a longer run.
    batch.add(2, SplineMode::Linear, false);
    batch.add(4, SplineMode::Linear, true);
    batch.add(7, SplineMode::Linear, false);
    // Catmull-Rom: open (end clamping) and closed (index wrapping), 3/4/7 pts.
    batch.add(3, SplineMode::CatmullRom, false);
    batch.add(4, SplineMode::CatmullRom, false);
    batch.add(7, SplineMode::CatmullRom, true);
    // Bezier: open chains need 3k+1 points (4, 7); closed chains a multiple of
    // three (3, 6).
    batch.add(4, SplineMode::Bezier, false);
    batch.add(7, SplineMode::Bezier, false);
    batch.add(3, SplineMode::Bezier, true);
    batch.add(6, SplineMode::Bezier, true);

    let results = gpu.evaluate(&batch.queries, &batch.points);
    assert_eq!(results.len(), batch.queries.len());
    for case in &batch.cases {
        assert_case(case, &results[case.index]);
    }
}

#[test]
fn degenerate_point_counts_return_the_contract() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpline::new(&ctx);
    let mut batch = Batch::new(0x0f1e_2d3c_4b5a_6978);
    // Too few points for a single segment under each mode: segment_count == 0.
    // Linear/Catmull-Rom with one point, Bezier open with three (< 4) and
    // Bezier closed with two (< 3).
    batch.add(1, SplineMode::Linear, false);
    batch.add(1, SplineMode::CatmullRom, false);
    batch.add(3, SplineMode::Bezier, false);
    batch.add(2, SplineMode::Bezier, true);

    let results = gpu.evaluate(&batch.queries, &batch.points);
    assert_eq!(results.len(), batch.queries.len());
    for case in &batch.cases {
        let got = &results[case.index];
        // Degenerate point pins to the first control point; the rest are zero.
        assert!(
            approx3(got.position, case.spline.point_at(case.t)),
            "degenerate point mismatch: gpu {:?} vs cpu {:?}",
            got.position,
            case.spline.point_at(case.t)
        );
        assert!(approx3(got.derivative, Vec3::ZERO), "derivative not zero");
        assert!(approx3(got.second, Vec3::ZERO), "second not zero");
        assert!(approx3(got.tangent, Vec3::ZERO), "tangent not zero");
        assert!(approx(got.curvature, 0.0), "curvature not zero");
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpline::new(&ctx);
    // No queries: no dispatch is issued and the result vector is empty. The
    // shared point buffer is non-empty so this isolates the query short-circuit.
    let points = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]];
    let out = gpu.evaluate(&[], &points);
    assert!(out.is_empty());
}
