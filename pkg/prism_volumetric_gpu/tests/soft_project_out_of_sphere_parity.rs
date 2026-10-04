//! Real-device parity test for the single-point sphere-projection twin.
//!
//! Each case evaluates one or more [`SoftProjectOutOfSphereQuery`] values on the
//! GPU and pins the returned [`SoftProjectOutOfSphereResult`] against an
//! independent `f32` reimplementation of
//! `prism_physics_core::soft::collision::body::project_out_of_sphere`. The
//! oracle is rebuilt here from first principles; this test never depends on the
//! golden crate.
//!
//! The projected position is a continuous quantity threaded through a `sqrt`
//! and a guarded division, so each channel is pinned with an
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` tolerance (`REL_FLOOR = 1e-6`) that
//! absorbs a fused multiply-add the scalar reference leaves separate. The
//! discrete `valid` flag is pinned with an exact `==`.
//!
//! The named fixtures and the random sweep keep the squared distance a safe
//! margin clear of both the `radius^2` knee and the `EPS_LEN_SQ` floor so a few
//! units in the last place cannot flip a branch, use a strictly positive
//! radius for the projecting cases, and cover the degenerate inputs (a
//! non-positive radius and a point already on or outside the sphere) that
//! report `valid = 0`, plus the center-coincident `+Y` fallback that reports
//! `valid = 1`.
//!
//! Every case short-circuits to a skip when no headless adapter is available,
//! so the suite is inert on a machine without a GPU and exercises the real
//! device elsewhere.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body::project_out_of_sphere`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_project_out_of_sphere::{
    GpuSoftProjectOutOfSphere, SoftProjectOutOfSphereQuery, SoftProjectOutOfSphereResult,
};
use prism_volumetric_gpu::GpuContext;

/// Squared-length coincidence floor; mirrors `EPS_LEN_SQ = 1e-12` in the golden
/// crate.
const EPS_LEN_SQ: f32 = 1e-12;

/// Relative-tolerance floor so a near-zero reference magnitude does not demand
/// an impossibly tight absolute match.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent `f32` reimplementation of `project_out_of_sphere` for one query,
/// in the same branch order and arithmetic order as the kernel: a non-positive
/// radius and a point already on or outside the sphere are echoed and invalid;
/// a point coincident with the center is nudged out along `+Y`; otherwise the
/// point is pushed onto the surface along the unit radial direction.
fn oracle(query: &SoftProjectOutOfSphereQuery) -> SoftProjectOutOfSphereResult {
    let pos = query.position;
    let center = query.center;
    let radius = query.radius;

    // Default: echo the position unchanged, invalid.
    let echo = SoftProjectOutOfSphereResult {
        position: pos,
        valid: 0,
    };

    if radius <= 0.0 {
        return echo;
    }

    let delta = [pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]];
    let dist_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

    if dist_sq >= radius * radius {
        return echo;
    }

    if dist_sq <= EPS_LEN_SQ {
        return SoftProjectOutOfSphereResult {
            position: [center[0], center[1] + radius, center[2]],
            valid: 1,
        };
    }

    let d = dist_sq.sqrt();
    let dir = [delta[0] / d, delta[1] / d, delta[2] / d];
    SoftProjectOutOfSphereResult {
        position: [
            center[0] + dir[0] * radius,
            center[1] + dir[1] * radius,
            center[2] + dir[2] * radius,
        ],
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
/// position channel within tolerance, the discrete `valid` flag exact.
fn pin(idx: usize, query: &SoftProjectOutOfSphereQuery, result: &SoftProjectOutOfSphereResult) {
    let want = oracle(query);
    for axis in 0..3 {
        assert!(
            close(result.position[axis], want.position[axis]),
            "query {idx}: position[{axis}] gpu={} oracle={} (pos={:?} center={:?} radius={})",
            result.position[axis],
            want.position[axis],
            query.position,
            query.center,
            query.radius
        );
    }
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={} (pos={:?} center={:?} radius={})",
        result.valid, want.valid, query.position, query.center, query.radius
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuSoftProjectOutOfSphere,
    queries: &[SoftProjectOutOfSphereQuery],
) {
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

/// A deterministic, well-conditioned random query: a strictly positive radius,
/// a random center, and a position placed along a random unit direction from
/// the center so the distance sits a safe margin (`>= 1e-2`) either clearly
/// inside or clearly beyond the sphere, never on the `radius` knee and always
/// well clear of the `EPS_LEN_SQ` coincidence floor.
fn rand_query(state: &mut u64) -> SoftProjectOutOfSphereQuery {
    let center = [
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

    let radius = range(state, 0.5, 4.0);
    let margin = 1.0e-2;
    let inside = lcg(state) & 1 == 0;
    let dist = if inside {
        // Clearly inside the sphere: pushed out onto the surface. Keep it well
        // above the coincidence floor and a margin below the radius knee.
        range(state, 0.05, radius - margin)
    } else {
        // Clearly outside the sphere: echoed unchanged, invalid.
        radius + margin + unit01(state) * 3.0
    };
    let position = [
        center[0] + unit[0] * dist,
        center[1] + unit[1] * dist,
        center[2] + unit[2] * dist,
    ];
    SoftProjectOutOfSphereQuery::new(position, center, radius)
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty input must return an empty vector");
}

#[test]
fn interior_point_pushed_onto_surface() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    // A point at distance 1 along +x inside a radius-2 sphere centered at the
    // origin: pushed out to (2, 0, 0) on the surface.
    let query = SoftProjectOutOfSphereQuery::new([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "an interior point is a valid projection");
    // The projected point must lie on the sphere surface.
    let p = got[0].position;
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    assert!(
        close(r, 2.0),
        "projected point must land on the sphere, got radius {r}"
    );
    assert!(
        close(p[0], 2.0),
        "projected point must land at (2,0,0), got {p:?}"
    );
    pin(0, &query, &got[0]);
}

#[test]
fn offset_center_interior_projection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    // A sphere not at the origin, with an oblique interior point.
    let center = [3.0, -2.0, 1.5];
    let query = SoftProjectOutOfSphereQuery::new([3.6, -1.4, 1.9], center, 1.5);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "an interior point is a valid projection");
    // The distance from the center to the projected point must equal radius.
    let p = got[0].position;
    let d = ((p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) + (p[2] - center[2]).powi(2))
        .sqrt();
    assert!(
        close(d, 1.5),
        "projected point must sit at radius 1.5 from the center, got {d}"
    );
    pin(0, &query, &got[0]);
}

#[test]
fn already_outside_is_echoed_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    // A point at distance 3 outside a radius-1 sphere: echoed unchanged.
    let query = SoftProjectOutOfSphereQuery::new([3.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(
        got[0].valid, 0,
        "a point outside the sphere must be reported invalid"
    );
    assert!(
        close(got[0].position[0], 3.0)
            && close(got[0].position[1], 0.0)
            && close(got[0].position[2], 0.0),
        "an outside point must be echoed unchanged, got {:?}",
        got[0].position
    );
    pin(0, &query, &got[0]);
}

#[test]
fn on_surface_boundary_is_echoed_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    // A point a safe margin beyond the surface so `dist_sq >= radius^2` holds
    // without jitter: echoed unchanged, invalid.
    let query = SoftProjectOutOfSphereQuery::new([2.02, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(
        got[0].valid, 0,
        "a point on/outside the surface must be reported invalid"
    );
    assert!(
        close(got[0].position[0], 2.02),
        "a boundary point must be echoed unchanged, got {:?}",
        got[0].position
    );
    pin(0, &query, &got[0]);
}

#[test]
fn center_coincident_falls_back_to_plus_y() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    // A point exactly on the center has no defined radial direction; it is
    // nudged out along +Y to center + (0, radius, 0), reported valid.
    let center = [1.0, 2.0, 3.0];
    let query = SoftProjectOutOfSphereQuery::new(center, center, 2.5);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(
        got[0].valid, 1,
        "a center-coincident point must still report a valid projection"
    );
    assert!(
        close(got[0].position[0], 1.0)
            && close(got[0].position[1], 4.5)
            && close(got[0].position[2], 3.0),
        "a coincident point must be nudged out along +Y, got {:?}",
        got[0].position
    );
    pin(0, &query, &got[0]);
}

#[test]
fn non_positive_radius_is_echoed_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    // A zero and a negative radius both mean "no collider": echo, invalid.
    let queries = [
        SoftProjectOutOfSphereQuery::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.0),
        SoftProjectOutOfSphereQuery::new([-4.0, 1.0, 2.0], [0.0, 0.0, 0.0], -1.5),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got[0].valid, 0, "a zero radius must be reported invalid");
    assert_eq!(
        got[1].valid, 0,
        "a negative radius must be reported invalid"
    );
    assert!(
        close(got[0].position[0], 1.0)
            && close(got[0].position[1], 2.0)
            && close(got[0].position[2], 3.0),
        "a zero-radius query must echo the position unchanged"
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    // An interior projection followed by an outside echo: a wrong per-element
    // stride would cross-contaminate the two decisions.
    let queries = [
        SoftProjectOutOfSphereQuery::new([0.5, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
        SoftProjectOutOfSphereQuery::new([5.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 2, "both results must be returned");
    assert_eq!(got[0].valid, 1, "first point is an interior projection");
    assert_eq!(got[1].valid, 0, "second point is an outside echo");
    assert!(
        close(got[0].position[0], 1.0),
        "first point pushed onto the sphere, got {:?}",
        got[0].position
    );
    assert!(
        close(got[1].position[0], 5.0),
        "second point echoed unchanged, got {:?}",
        got[1].position
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    let mut queries = vec![
        SoftProjectOutOfSphereQuery::new([0.5, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
        SoftProjectOutOfSphereQuery::new([5.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
        SoftProjectOutOfSphereQuery::new([2.0, 2.0, 3.0], [2.0, 2.0, 3.0], 1.75),
        SoftProjectOutOfSphereQuery::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.0),
        SoftProjectOutOfSphereQuery::new([3.6, -1.4, 1.9], [3.0, -2.0, 1.5], 1.5),
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
    let gpu = GpuSoftProjectOutOfSphere::new(&ctx);
    let mut state: u64 = 0x0CEA_1F10_7A6B_9D55;
    let queries: Vec<SoftProjectOutOfSphereQuery> =
        (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
