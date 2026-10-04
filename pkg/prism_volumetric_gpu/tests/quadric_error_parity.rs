//! Real-device parity for the quadric-error twin:
//! [`GpuQuadricError`](prism_volumetric_gpu::quadric_error::GpuQuadricError) must
//! reproduce the `CPU` golden `error` of
//! `prism_physics_core::collider::quadric::Quadric`, which expands the quadratic
//! form `v^T Q v` from the ten distinct symmetric coefficients and clamps the
//! result at zero.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written in pure `f32` with the exact same additive term order as the golden
//! and no `glam` dependency, so the test never imports `prism_physics_core` or
//! `prism_render_architecture`.
//!
//! The fixtures cover the zero quadric (zero error everywhere), a single-plane
//! quadric evaluated on its own plane (near-zero error), the same plane
//! evaluated off the plane (positive squared distance), and a batch of distinct
//! queries that catches any `std430` stride aliasing. A reject-sampled sweep
//! over random coefficients and points follows, plus an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The error is a continuous quantity, compared with an absolute-or-relative
//! tolerance (`abs <= 1e-4 || rel <= 1e-3`, with a relative floor of `1e-6`).
//! The additive chain is emitted in the exact golden term order on both sides so
//! a fused multiply-add on the device cannot reassociate the sum; the sweep
//! reject-samples any query whose unclamped form sits within `1e-2` of the
//! `max(e, 0)` clamp knee so the clamp verdict is never ambiguous. `valid` is
//! total and compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric`；无第三方
//! 引擎源码或衍生代码。

use prism_volumetric_gpu::quadric_error::{GpuQuadricError, QuadricErrorQuery};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor so near-zero magnitudes do not divide by a tiny
/// reference.
const REL_FLOOR: f32 = 1.0e-6;
/// Absolute tolerance for the continuous error.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous error.
const REL_TOL: f32 = 1.0e-3;

/// The resolved oracle answer: the clamped quadratic form and the validity flag.
struct Expected {
    error: f32,
    valid: u32,
}

/// The unclamped quadratic form, in the exact golden term order. Shared by the
/// oracle and the sweep's clamp-knee reject sampler.
fn unclamped_form(q: &QuadricErrorQuery) -> f32 {
    let x = q.vx;
    let y = q.vy;
    let z = q.vz;
    q.a2 * x * x
        + 2.0 * q.ab * x * y
        + 2.0 * q.ac * x * z
        + 2.0 * q.ad * x
        + q.b2 * y * y
        + 2.0 * q.bc * y * z
        + 2.0 * q.bd * y
        + q.c2 * z * z
        + 2.0 * q.cd * z
        + q.d2
}

/// The independent oracle for one query, mirroring the on-device kernel's
/// additive order exactly, then clamping at zero. Written in pure `f32` with no
/// `glam` dependency.
fn oracle(q: &QuadricErrorQuery) -> Expected {
    Expected {
        error: unclamped_form(q).max(0.0),
        valid: 1,
    }
}

/// True when `got` matches `want` within the absolute-or-relative tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    if diff <= ABS_TOL {
        return true;
    }
    let scale = want.abs().max(REL_FLOOR);
    diff / scale <= REL_TOL
}

/// Dispatches one query and asserts the resolved verdict.
fn assert_parity(ctx: &GpuContext, gpu: &GpuQuadricError, q: QuadricErrorQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    assert_result(&got[0], &q);
}

/// Asserts parity for a whole batch, so the shared dispatch exercises the
/// `std430` stride.
fn assert_batch(ctx: &GpuContext, gpu: &GpuQuadricError, queries: &[QuadricErrorQuery]) {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_result(r, q);
    }
}

/// Compares one `GPU` result against the oracle for the same query. `valid` is
/// discrete and compared exactly; `error` is continuous.
fn assert_result(
    r: &prism_volumetric_gpu::quadric_error::QuadricErrorResult,
    q: &QuadricErrorQuery,
) {
    let e = oracle(q);
    assert_eq!(r.valid, e.valid, "valid flag mismatch: query={q:?}");
    assert!(
        close(r.error, e.error),
        "error mismatch: gpu={} cpu={} query={q:?}",
        r.error,
        e.error
    );
}

#[test]
fn zero_quadric_reports_zero_everywhere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricError::new(&ctx);
    // The zero quadric adds nothing and reports zero at an arbitrary point.
    let q = QuadricErrorQuery::new([0.0; 10], [3.0, -2.0, 1.5]);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert!(
        got[0].error.abs() <= ABS_TOL,
        "zero quadric must report zero error"
    );
}

/// Builds the ten coefficients of the plane quadric for a unit normal
/// `(a, b, c)` and offset `d`, matching the golden `Quadric::from_plane`
/// construction. The resulting quadric reports `(a*x + b*y + c*z + d)^2`.
fn plane_quadric(a: f32, b: f32, c: f32, d: f32) -> [f32; 10] {
    [
        a * a,
        a * b,
        a * c,
        a * d,
        b * b,
        b * c,
        b * d,
        c * c,
        c * d,
        d * d,
    ]
}

#[test]
fn point_on_plane_reports_near_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricError::new(&ctx);
    // Plane 0.6x + 0.8y - 1 = 0 (unit normal). The point (1, 0.5, 7) satisfies
    // 0.6 + 0.4 = 1, so it lies on the plane and the squared distance is zero.
    let coeffs = plane_quadric(0.6, 0.8, 0.0, -1.0);
    let q = QuadricErrorQuery::new(coeffs, [1.0, 0.5, 7.0]);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert!(
        got[0].error.abs() <= ABS_TOL,
        "point on the plane must report zero squared distance"
    );
}

#[test]
fn point_off_plane_reports_squared_distance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricError::new(&ctx);
    // Plane z - 3 = 0 (unit normal along z). At z = 5 the squared distance is
    // (5 - 3)^2 = 4, independent of x and y.
    let coeffs = plane_quadric(0.0, 0.0, 1.0, -3.0);
    let q = QuadricErrorQuery::new(coeffs, [2.0, -4.0, 5.0]);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert!(
        close(got[0].error, 4.0),
        "point off the plane must report the squared distance (4.0), got {}",
        got[0].error
    );
}

#[test]
fn batch_stride_mixes_distinct_quadrics() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricError::new(&ctx);
    // A batch of distinct queries exercises the std430 query/result stride:
    // every slot must read and write its own non-aliased data.
    let queries = [
        QuadricErrorQuery::new([0.0; 10], [3.0, -2.0, 1.5]),
        QuadricErrorQuery::new(plane_quadric(0.6, 0.8, 0.0, -1.0), [1.0, 0.5, 7.0]),
        QuadricErrorQuery::new(plane_quadric(0.0, 0.0, 1.0, -3.0), [2.0, -4.0, 5.0]),
        QuadricErrorQuery::new(plane_quadric(0.0, 0.0, 1.0, 0.0), [-1.0, 2.0, -2.5]),
    ];
    assert_batch(&ctx, &gpu, &queries);
}

/// A small deterministic linear-congruential generator so the sweep needs no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricError::new(&ctx);
    let mut rng = Lcg::new(0x0E_11_02_57);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let coeffs = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let point = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let q = QuadricErrorQuery::new(coeffs, point);

        // Reject-sample any query whose unclamped form sits within 1e-2 of the
        // max(e, 0) clamp knee, so a fused multiply-add on the device cannot
        // flip which side of the clamp the result lands on.
        if unclamped_form(&q).abs() <= 1.0e-2 {
            continue;
        }
        queries.push(q);
    }

    assert_batch(&ctx, &gpu, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricError::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
