//! Real-device parity test for the one-sided long-range-attachment (LRA) leash
//! projection twin.
//!
//! Each case evaluates one or more [`SoftLongRangeProjectQuery`] values on the
//! GPU and pins the returned [`SoftLongRangeProjectResult`] against an
//! independent `f32` reimplementation of
//! `prism_physics_core::soft::constraint::long_range::project_long_range`. The
//! oracle is rebuilt here from first principles; this test never depends on the
//! golden crate.
//!
//! The projected position and the updated Lagrange multiplier are continuous
//! quantities threaded through a `sqrt` and guarded divisions, so they are
//! pinned with an `abs_diff <= 1e-4 || rel_diff <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`) that absorbs a fused multiply-add the scalar reference
//! leaves separate. The discrete `valid` flag is pinned with an exact `==`.
//!
//! The named fixtures and the random sweep keep the stretch a safe margin clear
//! of the `C = 0` leash knee so a few units in the last place cannot flip the
//! branch, use a strictly positive `dt`, and cover the two degenerate inputs
//! (a pinned particle and one coincident with its anchor) that report
//! `valid = 0`.
//!
//! Every case short-circuits to a skip when no headless adapter is available,
//! so the suite is inert on a machine without a GPU and exercises the real
//! device elsewhere.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::long_range::project_long_range`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_long_range_project::{
    GpuSoftLongRangeProject, SoftLongRangeProjectQuery, SoftLongRangeProjectResult,
};
use prism_volumetric_gpu::GpuContext;

/// Coincidence epsilon; mirrors `EPSILON = f32::EPSILON` in the golden crate.
const EPSILON: f32 = 1.192_092_9e-7;

/// Relative-tolerance floor so a near-zero reference magnitude does not demand
/// an impossibly tight absolute match.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent `f32` reimplementation of `project_long_range` for one query, in
/// the same branch order and arithmetic order as the kernel: a pinned particle
/// (`w <= 0`) and one coincident with its anchor (`|p - anchor| < EPSILON`) are
/// inert and invalid; a particle still inside the leash sphere (`C <= 0`) is a
/// valid no-op; otherwise the compliant correction pulls it toward the anchor.
fn oracle(query: &SoftLongRangeProjectQuery) -> SoftLongRangeProjectResult {
    let p = query.position;
    let a = query.anchor;

    // Default: pass position and multiplier through unchanged, invalid.
    let passthrough = SoftLongRangeProjectResult {
        position: p,
        lambda: query.lambda,
        valid: 0,
    };

    if query.w <= 0.0 {
        return passthrough;
    }

    let delta = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let dist = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    if dist < EPSILON {
        return passthrough;
    }

    let c = dist - query.max_distance;
    if c <= 0.0 {
        return SoftLongRangeProjectResult {
            position: p,
            lambda: query.lambda,
            valid: 1,
        };
    }

    let normal = [delta[0] / dist, delta[1] / dist, delta[2] / dist];
    let alpha_tilde = query.compliance / (query.dt * query.dt);
    let denom = query.w + alpha_tilde;
    let delta_lambda = if denom > 0.0 {
        (-c - alpha_tilde * query.lambda) / denom
    } else {
        0.0
    };
    let scale = delta_lambda * query.w;
    SoftLongRangeProjectResult {
        position: [
            p[0] + normal[0] * scale,
            p[1] + normal[1] * scale,
            p[2] + normal[2] * scale,
        ],
        lambda: query.lambda + delta_lambda,
        valid: 1,
    }
}

/// Returns `true` when `a` matches `b` within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= 1.0e-3
}

/// Pins one `GPU` result against the independent host oracle: each continuous
/// channel within tolerance, the discrete `valid` flag exact.
fn pin(idx: usize, query: &SoftLongRangeProjectQuery, result: &SoftLongRangeProjectResult) {
    let want = oracle(query);
    for axis in 0..3 {
        assert!(
            close(result.position[axis], want.position[axis]),
            "query {idx}: position[{axis}] gpu={} oracle={} (p={:?} a={:?} max={} comp={} lambda={} dt={})",
            result.position[axis],
            want.position[axis],
            query.position,
            query.anchor,
            query.max_distance,
            query.compliance,
            query.lambda,
            query.dt
        );
    }
    assert!(
        close(result.lambda, want.lambda),
        "query {idx}: lambda gpu={} oracle={}",
        result.lambda,
        want.lambda
    );
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSoftLongRangeProject, queries: &[SoftLongRangeProjectQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// 64-bit linear-congruential step (`Knuth`/`PCG` constants), returning the
/// high word so the stream has good spread without any transcendental math.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 32) as u32
}

/// A deterministic pseudo-random `f32` in `[0, 1]`.
fn unit01(state: &mut u64) -> f32 {
    lcg(state) as f32 / u32::MAX as f32
}

/// A deterministic pseudo-random `f32` in `[lo, hi]`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + unit01(state) * (hi - lo)
}

/// A deterministic, well-conditioned random query: a strictly positive inverse
/// mass, a strictly positive timestep in `[1/240, 1/30]`, a non-negative
/// compliance, and a position placed along a random unit direction from the
/// anchor so the stretch sits a safe margin (`>= 1e-2`) either above or below
/// the leash radius, never on the `C = 0` knee.
fn rand_query(state: &mut u64) -> SoftLongRangeProjectQuery {
    let anchor = [
        range(state, -5.0, 5.0),
        range(state, -5.0, 5.0),
        range(state, -5.0, 5.0),
    ];
    // A random direction, renormalized; biased away from zero so the unit
    // vector is well defined.
    let mut dir = [
        range(state, -1.0, 1.0),
        range(state, -1.0, 1.0),
        range(state, -1.0, 1.0),
    ];
    let mut len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if len < 1.0e-3 {
        dir = [1.0, 0.0, 0.0];
        len = 1.0;
    }
    let unit = [dir[0] / len, dir[1] / len, dir[2] / len];

    let max_distance = range(state, 0.5, 4.0);
    let margin = 1.0e-2;
    let above = lcg(state) & 1 == 0;
    let dist = if above {
        // Clearly beyond the leash: pulled back onto the sphere.
        max_distance + margin + unit01(state) * 3.0
    } else {
        // Clearly inside the leash: a valid slack no-op. Keep it strictly
        // positive and clear of the anchor-coincidence epsilon.
        (max_distance - margin - unit01(state) * (max_distance * 0.5)).max(0.05)
    };
    let position = [
        anchor[0] + unit[0] * dist,
        anchor[1] + unit[1] * dist,
        anchor[2] + unit[2] * dist,
    ];
    let w = range(state, 0.1, 4.0);
    let compliance = unit01(state) * 0.01;
    let lambda = range(state, -1.0, 1.0);
    let dt = range(state, 1.0 / 240.0, 1.0 / 30.0);
    SoftLongRangeProjectQuery::new(position, w, anchor, max_distance, compliance, lambda, dt)
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftLongRangeProject::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty input must return an empty vector");
}

#[test]
fn rigid_overstretch_pulls_onto_sphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftLongRangeProject::new(&ctx);
    // The particle sits at distance 3 along +x from the origin anchor with a
    // leash radius of 1 and zero compliance (rigid): it should be pulled back
    // essentially onto the sphere.
    let query = SoftLongRangeProjectQuery::new(
        [3.0, 0.0, 0.0],
        1.0,
        [0.0, 0.0, 0.0],
        1.0,
        0.0,
        0.0,
        1.0 / 60.0,
    );
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "an overstretched particle is a valid pull");
    // Rigid leash, lambda=0: delta_lambda = -(3-1)/w = -2, correction = -2
    // along +x, so x -> 1.0 (onto the sphere).
    assert!(
        close(got[0].position[0], 1.0),
        "rigid pull should land on the sphere, got {}",
        got[0].position[0]
    );
    pin(0, &query, &got[0]);
}

#[test]
fn slack_inside_is_valid_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftLongRangeProject::new(&ctx);
    // Distance 0.5 < leash radius 2.0: one-sided constraint is slack.
    let query = SoftLongRangeProjectQuery::new(
        [0.5, 0.0, 0.0],
        1.0,
        [0.0, 0.0, 0.0],
        2.0,
        0.0,
        0.25,
        1.0 / 60.0,
    );
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(
        got[0].valid, 1,
        "a slack particle is still a valid decision"
    );
    assert!(
        close(got[0].position[0], 0.5) && close(got[0].lambda, 0.25),
        "a slack particle must be an exact no-op"
    );
    pin(0, &query, &got[0]);
}

#[test]
fn pinned_particle_is_inert_and_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftLongRangeProject::new(&ctx);
    // w = 0 marks a pinned (infinite-mass) particle: inert and invalid even
    // while grossly overstretched.
    let query = SoftLongRangeProjectQuery::new(
        [10.0, 0.0, 0.0],
        0.0,
        [0.0, 0.0, 0.0],
        1.0,
        0.0,
        0.5,
        1.0 / 60.0,
    );
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(
        got[0].valid, 0,
        "a pinned particle must be reported invalid"
    );
    assert!(
        close(got[0].position[0], 10.0) && close(got[0].lambda, 0.5),
        "a pinned particle must pass position and lambda through unchanged"
    );
    pin(0, &query, &got[0]);
}

#[test]
fn coincident_with_anchor_is_inert_and_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftLongRangeProject::new(&ctx);
    // Particle exactly on the anchor: no defined gradient, inert and invalid.
    let query = SoftLongRangeProjectQuery::new(
        [1.0, 2.0, 3.0],
        1.0,
        [1.0, 2.0, 3.0],
        1.0,
        0.0,
        0.75,
        1.0 / 60.0,
    );
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(
        got[0].valid, 0,
        "an anchor-coincident particle must be reported invalid"
    );
    assert!(
        close(got[0].lambda, 0.75),
        "a coincident particle must pass lambda through unchanged"
    );
    pin(0, &query, &got[0]);
}

#[test]
fn compliant_partial_correction() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftLongRangeProject::new(&ctx);
    // A positive compliance softens the leash: the particle is pulled part of
    // the way back rather than snapping onto the sphere.
    let query = SoftLongRangeProjectQuery::new(
        [0.0, 4.0, 0.0],
        2.0,
        [0.0, 0.0, 0.0],
        1.0,
        0.004,
        0.0,
        1.0 / 60.0,
    );
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "a compliant pull is a valid decision");
    // Still outside the sphere but closer than it started.
    assert!(
        got[0].position[1] < 4.0 && got[0].position[1] > 1.0,
        "a compliant pull should move partway back, got {}",
        got[0].position[1]
    );
    pin(0, &query, &got[0]);
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftLongRangeProject::new(&ctx);
    // An overstretched pull followed by a slack no-op: a wrong per-element
    // stride would cross-contaminate the two decisions.
    let queries = [
        SoftLongRangeProjectQuery::new(
            [3.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            1.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
        SoftLongRangeProjectQuery::new(
            [0.5, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            2.0,
            0.0,
            0.1,
            1.0 / 60.0,
        ),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 2, "both results must be returned");
    assert!(
        close(got[0].position[0], 1.0),
        "first particle pulled onto the sphere"
    );
    assert!(
        close(got[1].position[0], 0.5),
        "second particle is a slack no-op"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftLongRangeProject::new(&ctx);
    let mut queries = vec![
        SoftLongRangeProjectQuery::new(
            [3.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            1.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
        SoftLongRangeProjectQuery::new(
            [0.5, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            2.0,
            0.0,
            0.2,
            1.0 / 60.0,
        ),
        SoftLongRangeProjectQuery::new(
            [10.0, 0.0, 0.0],
            0.0,
            [0.0, 0.0, 0.0],
            1.0,
            0.0,
            0.5,
            1.0 / 60.0,
        ),
        SoftLongRangeProjectQuery::new(
            [1.0, 2.0, 3.0],
            1.0,
            [1.0, 2.0, 3.0],
            1.0,
            0.0,
            0.3,
            1.0 / 60.0,
        ),
        SoftLongRangeProjectQuery::new(
            [0.0, 4.0, 0.0],
            2.0,
            [0.0, 0.0, 0.0],
            1.0,
            0.004,
            0.0,
            1.0 / 60.0,
        ),
    ];
    let mut state: u64 = 0x5151_7EA2_C10B_1234;
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftLongRangeProject::new(&ctx);
    let mut state: u64 = 0x0CEA_1F10_7A6B_9D55;
    let queries: Vec<SoftLongRangeProjectQuery> =
        (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
