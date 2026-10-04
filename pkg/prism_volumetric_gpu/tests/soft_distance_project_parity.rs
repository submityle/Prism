//! Real-device parity for the distance-projection twin:
//! [`GpuSoftDistanceProject`](prism_volumetric_gpu::soft_distance_project::GpuSoftDistanceProject)
//! must reproduce the `CPU` golden `project_distance_constraint` of
//! `prism_physics_core::soft::constraint::distance`, the compliant `XPBD`
//! stretch projection that drives the constraint `C = |p_a - p_b| -
//! rest_length` to zero with a warm-started Lagrange multiplier, splitting the
//! correction between the two endpoints in proportion to their inverse masses
//! so a pinned particle never moves.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the two ordered degeneracy guards (non-positive inverse-mass sum, coincident
//! endpoints), the compliant multiplier delta and the mass-weighted correction
//! — written out directly so the test never imports `prism_physics_core` or
//! `prism_render_architecture`. It mirrors the reference branch for branch.
//!
//! The fixtures cover the regimes the kernel must honor: symmetric free masses
//! projected back to rest, a single pinned endpoint, coincident endpoints
//! (degenerate), both endpoints pinned (degenerate), a compliant partial
//! correction with a warm-started multiplier, plus a multi-element mixed batch
//! that validates the `std430` array stride end to end. A sweep over random
//! edges kept away from the coincidence floor follows, plus an empty batch the
//! host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The correction threads through a subtraction, a `sqrt` and guarded
//! divisions, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact. The continuous comparison is `abs_diff <= 1e-4 || rel_diff <=
//! 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//! The sweep keeps every edge length comfortably away from the coincidence
//! floor and uses strictly positive inverse masses so parity never sits on a
//! branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::distance::project_distance_constraint`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_distance_project::{
    GpuSoftDistanceProject, SoftDistanceProjectQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Coincidence floor on the edge length, matching the golden `EPSILON`
/// (`f32::EPSILON`).
const EPSILON: f32 = 1.1920929e-7;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Returns `true` when two vectors agree component-wise within tolerance.
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Independent host re-implementation of the golden `project_distance_constraint`,
/// returning the projected endpoints, the updated multiplier and the `valid`
/// flag without importing the golden crate.
fn oracle(q: &SoftDistanceProjectQuery) -> ([f32; 3], [f32; 3], f32, u32) {
    let pa = q.pa;
    let pb = q.pb;
    let w_sum = q.wa + q.wb;
    // Both endpoints pinned: nothing to project, echo lambda.
    if w_sum <= 0.0 {
        return (pa, pb, q.lambda, 0);
    }
    let delta = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
    let len = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    // Coincident endpoints: direction is undefined, skip.
    if len < EPSILON {
        return (pa, pb, q.lambda, 0);
    }
    let normal = [delta[0] / len, delta[1] / len, delta[2] / len];
    let c = len - q.rest_length;
    let alpha_tilde = q.compliance / (q.dt * q.dt);
    let d_lambda = (-c - alpha_tilde * q.lambda) / (w_sum + alpha_tilde);
    let correction = [
        normal[0] * d_lambda,
        normal[1] * d_lambda,
        normal[2] * d_lambda,
    ];
    let new_pa = [
        pa[0] + correction[0] * q.wa,
        pa[1] + correction[1] * q.wa,
        pa[2] + correction[2] * q.wa,
    ];
    let new_pb = [
        pb[0] - correction[0] * q.wb,
        pb[1] - correction[1] * q.wb,
        pb[2] - correction[2] * q.wb,
    ];
    (new_pa, new_pb, q.lambda + d_lambda, 1)
}

/// Dispatches a single query and asserts the device output matches the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuSoftDistanceProject, q: SoftDistanceProjectQuery) {
    let results = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let (new_pa, new_pb, new_lambda, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close3(r.new_pa, new_pa),
        "new_pa mismatch: gpu={:?} cpu={new_pa:?} query={q:?}",
        r.new_pa
    );
    assert!(
        close3(r.new_pb, new_pb),
        "new_pb mismatch: gpu={:?} cpu={new_pb:?} query={q:?}",
        r.new_pb
    );
    assert!(
        close(r.new_lambda, new_lambda),
        "new_lambda mismatch: gpu={} cpu={new_lambda} query={q:?}",
        r.new_lambda
    );
}

#[test]
fn symmetric_free_masses_return_to_rest() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftDistanceProject::new(&ctx);
    // Equal inverse masses, rigid (compliance 0): one projection restores the
    // rest length, each endpoint moving half the length error toward the other.
    let q = SoftDistanceProjectQuery::new(
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0 / 60.0,
    );
    assert_parity(&ctx, &gpu, q);
    // Explicit check: endpoints collapse symmetrically to the rest separation.
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1);
    assert!(close3(r.new_pa, [0.5, 0.0, 0.0]), "A should move to 0.5");
    assert!(close3(r.new_pb, [1.5, 0.0, 0.0]), "B should move to 1.5");
}

#[test]
fn single_pinned_only_other_moves() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftDistanceProject::new(&ctx);
    // Endpoint A pinned (wa = 0): only B carries the full correction.
    let q = SoftDistanceProjectQuery::new(
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0 / 60.0,
    );
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.new_pa, q.pa, "pinned endpoint A must not move");
    assert!(
        close3(r.new_pb, [1.0, 0.0, 0.0]),
        "B should restore rest length"
    );
}

#[test]
fn coincident_endpoints_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftDistanceProject::new(&ctx);
    // Endpoints coincide: length below EPSILON, direction undefined → invalid,
    // output echoes the input and the incoming multiplier.
    let q = SoftDistanceProjectQuery::new(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        1.0,
        1.0,
        1.0,
        0.0,
        0.42,
        1.0 / 60.0,
    );
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 0);
    assert_eq!(r.new_lambda, 0.42, "degenerate edge echoes lambda");
}

#[test]
fn both_pinned_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftDistanceProject::new(&ctx);
    // Zero inverse-mass sum: both particles pinned, output equals input.
    let q = SoftDistanceProjectQuery::new(
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        0.0,
        0.0,
        1.0,
        0.0,
        0.7,
        1.0 / 60.0,
    );
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 0);
    assert_eq!(r.new_pa, q.pa);
    assert_eq!(r.new_pb, q.pb);
    assert_eq!(r.new_lambda, 0.7, "pinned edge echoes lambda");
}

#[test]
fn compliant_partial_correction_updates_multiplier() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftDistanceProject::new(&ctx);
    // Positive compliance softens the projection: the correction is a fraction
    // of the rigid move and the warm-started multiplier feeds d_lambda.
    let q = SoftDistanceProjectQuery::new(
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        1.0,
        1.0,
        1.0,
        2.0,
        0.1,
        1.0,
    );
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1);
    // alpha_tilde = 2, d_lambda = (-(2-1) - 2*0.1)/(2+2) = -0.3.
    assert!(
        close(r.new_lambda, -0.2),
        "new_lambda should be 0.1 + (-0.3)"
    );
}

#[test]
fn multi_element_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftDistanceProject::new(&ctx);
    // A mixed batch (symmetric, pinned, coincident, both-pinned, compliant,
    // diagonal) exercises the std430 array stride: every slot must decode at
    // the right byte offset.
    let queries = [
        SoftDistanceProjectQuery::new(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
        SoftDistanceProjectQuery::new(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
        SoftDistanceProjectQuery::new(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            1.0,
            1.0,
            1.0,
            0.0,
            0.25,
            1.0 / 60.0,
        ),
        SoftDistanceProjectQuery::new(
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            0.0,
            0.0,
            1.0,
            0.0,
            0.5,
            1.0 / 60.0,
        ),
        SoftDistanceProjectQuery::new(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            1.0,
            1.0,
            1.0,
            2.0,
            0.1,
            1.0,
        ),
        SoftDistanceProjectQuery::new(
            [1.0, 1.0, 1.0],
            [2.0, 3.0, 5.0],
            0.75,
            1.25,
            2.0,
            0.3,
            -0.2,
            1.0 / 120.0,
        ),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (new_pa, new_pb, new_lambda, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close3(r.new_pa, new_pa),
            "batch new_pa mismatch: gpu={:?} cpu={new_pa:?} query={q:?}",
            r.new_pa
        );
        assert!(
            close3(r.new_pb, new_pb),
            "batch new_pb mismatch: gpu={:?} cpu={new_pb:?} query={q:?}",
            r.new_pb
        );
        assert!(
            close(r.new_lambda, new_lambda),
            "batch new_lambda mismatch: gpu={} cpu={new_lambda} query={q:?}",
            r.new_lambda
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftDistanceProject::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
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
    let gpu = GpuSoftDistanceProject::new(&ctx);
    let mut rng = Lcg::new(0x3D_9A_71_55);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // Positive inverse masses: w_sum is always strictly positive.
        let wa = rng.next_range(0.1, 2.0);
        let wb = rng.next_range(0.1, 2.0);
        let rest_length = rng.next_range(0.5, 3.0);
        let compliance = rng.next_range(0.0, 1.0);
        let lambda = rng.next_range(-1.0, 1.0);
        // Timestep strictly positive, in a physically plausible substep range.
        let dt = rng.next_range(1.0 / 240.0, 1.0 / 30.0);

        // A well-conditioned unit direction: components bounded away from zero.
        let sx = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
        let sy = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
        let sz = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
        let dx = sx * rng.next_range(0.3, 1.0);
        let dy = sy * rng.next_range(0.3, 1.0);
        let dz = sz * rng.next_range(0.3, 1.0);
        let inv_norm = 1.0 / (dx * dx + dy * dy + dz * dz).sqrt();
        let ux = dx * inv_norm;
        let uy = dy * inv_norm;
        let uz = dz * inv_norm;

        // Target length kept well clear of the coincidence floor (margin >= 0.1).
        let target_len = rng.next_range(0.1, 4.0);
        let pb = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let pa = [
            pb[0] + ux * target_len,
            pb[1] + uy * target_len,
            pb[2] + uz * target_len,
        ];
        queries.push(SoftDistanceProjectQuery::new(
            pa,
            pb,
            wa,
            wb,
            rest_length,
            compliance,
            lambda,
            dt,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (new_pa, new_pb, new_lambda, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close3(r.new_pa, new_pa),
            "sweep new_pa mismatch: gpu={:?} cpu={new_pa:?} query={q:?}",
            r.new_pa
        );
        assert!(
            close3(r.new_pb, new_pb),
            "sweep new_pb mismatch: gpu={:?} cpu={new_pb:?} query={q:?}",
            r.new_pb
        );
        assert!(
            close(r.new_lambda, new_lambda),
            "sweep new_lambda mismatch: gpu={} cpu={new_lambda} query={q:?}",
            r.new_lambda
        );
    }
}
