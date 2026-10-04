//! Real-device parity test for the sphere-vs-plane contact twin.
//!
//! Each case evaluates one or more [`SpherePlaneContactQuery`] values on the
//! GPU and pins the returned [`SpherePlaneContactResult`] against an
//! independent `f32` reimplementation of
//! `prism_physics_core::collide::primitives::sphere_plane`. The oracle is
//! rebuilt here from first principles; this test never depends on the golden
//! crate, and the crate carries no `glam` dev-dependency, so the dot product
//! and the point arithmetic are all hand-written in `f32`.
//!
//! The contact normal, the two contact points and the penetration depth are
//! continuous quantities threaded through a dot product and a few multiplies,
//! so each is pinned with an `abs_diff <= 1e-4 || rel_diff <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`) that absorbs a fused multiply-add the scalar reference
//! leaves separate. The discrete `valid` flag is pinned with an exact `==`.
//!
//! The rejection boundary sits at `penetration = -CONTACT_TOLERANCE`; the
//! random sweep keeps every sample a safe margin away from that knee so the
//! `valid` flag never flips under a `GPU`-only reassociation.
//!
//! Every case short-circuits to a skip when no headless adapter is available,
//! so the suite is inert on a machine without a GPU and exercises the real
//! device elsewhere.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collide::primitives::sphere_plane`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sphere_plane_contact::{
    GpuSpherePlaneContact, SpherePlaneContactQuery, SpherePlaneContactResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor so a near-zero reference magnitude does not demand
/// an impossibly tight absolute match.
const REL_FLOOR: f32 = 1.0e-6;

/// Contact tolerance mirroring `CONTACT_TOLERANCE = 1e-4` in the golden crate.
const CONTACT_TOLERANCE: f32 = 1.0e-4;

/// Independent `f32` reimplementation of `sphere_plane` for one query, in the
/// same arithmetic order as the kernel: `signed = dot(normal_world, center) -
/// offset_world`, `penetration = radius - signed`, a contact when
/// `penetration >= -CONTACT_TOLERANCE`, and then `normal = -normal_world`,
/// `point_a = center - normal_world * radius`, `point_b = center -
/// normal_world * signed`. A miss zeroes every channel and sets `valid = 0`.
fn oracle(query: &SpherePlaneContactQuery) -> SpherePlaneContactResult {
    let [cx, cy, cz] = query.center;
    let [nx, ny, nz] = query.normal_world;
    let radius = query.radius;

    let signed = nx * cx + ny * cy + nz * cz - query.offset_world;
    let penetration = radius - signed;
    let hit = penetration >= -CONTACT_TOLERANCE;

    if !hit {
        return SpherePlaneContactResult {
            normal: [0.0, 0.0, 0.0],
            point_a: [0.0, 0.0, 0.0],
            point_b: [0.0, 0.0, 0.0],
            penetration: 0.0,
            valid: 0,
        };
    }

    SpherePlaneContactResult {
        normal: [-nx, -ny, -nz],
        point_a: [cx - nx * radius, cy - ny * radius, cz - nz * radius],
        point_b: [cx - nx * signed, cy - ny * signed, cz - nz * signed],
        penetration,
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

/// Pins one `GPU` result against the independent host oracle: the continuous
/// channels within tolerance, the discrete `valid` flag exact.
fn pin(idx: usize, query: &SpherePlaneContactQuery, result: &SpherePlaneContactResult) {
    let want = oracle(query);
    for axis in 0..3 {
        assert!(
            close(result.normal[axis], want.normal[axis]),
            "query {idx}: normal[{axis}] gpu={} oracle={}",
            result.normal[axis],
            want.normal[axis]
        );
        assert!(
            close(result.point_a[axis], want.point_a[axis]),
            "query {idx}: point_a[{axis}] gpu={} oracle={}",
            result.point_a[axis],
            want.point_a[axis]
        );
        assert!(
            close(result.point_b[axis], want.point_b[axis]),
            "query {idx}: point_b[{axis}] gpu={} oracle={}",
            result.point_b[axis],
            want.point_b[axis]
        );
    }
    assert!(
        close(result.penetration, want.penetration),
        "query {idx}: penetration gpu={} oracle={}",
        result.penetration,
        want.penetration
    );
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuSpherePlaneContact, queries: &[SpherePlaneContactQuery]) {
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

/// A deterministic, well-conditioned random query: a random unit plane normal,
/// a positive radius and a signed distance held a safe margin away from the
/// `penetration = -CONTACT_TOLERANCE` rejection knee so the `valid` flag is
/// stable under a `GPU`-only reassociation. The center is reconstructed as
/// `normal * (signed + offset) + tangential`, so the chosen signed distance is
/// reproduced exactly while a nonzero offset and a tangential component keep
/// the dot product fully exercised.
fn rand_query(state: &mut u64) -> SpherePlaneContactQuery {
    // Random unit normal, falling back to +x on a near-zero draw.
    let mut n = [
        range(state, -1.0, 1.0),
        range(state, -1.0, 1.0),
        range(state, -1.0, 1.0),
    ];
    let mut len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1.0e-3 {
        n = [1.0, 0.0, 0.0];
        len = 1.0;
    }
    let normal = [n[0] / len, n[1] / len, n[2] / len];

    let radius = range(state, 0.5, 4.0);

    // Keep the signed distance away from the knee signed = radius + tolerance.
    let knee = radius + CONTACT_TOLERANCE;
    let mut signed_target = range(state, -6.0, 6.0);
    while (signed_target - knee).abs() < 1.0e-2 {
        signed_target = range(state, -6.0, 6.0);
    }

    // Tangential component orthogonal to the normal (does not change `signed`).
    let v = [
        range(state, -5.0, 5.0),
        range(state, -5.0, 5.0),
        range(state, -5.0, 5.0),
    ];
    let proj = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2];
    let tang = [
        v[0] - proj * normal[0],
        v[1] - proj * normal[1],
        v[2] - proj * normal[2],
    ];

    let offset = range(state, -2.0, 2.0);
    let s = signed_target + offset;
    let center = [
        normal[0] * s + tang[0],
        normal[1] * s + tang[1],
        normal[2] * s + tang[2],
    ];

    SpherePlaneContactQuery::new(center, radius, normal, offset)
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpherePlaneContact::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty input must return an empty vector");
}

#[test]
fn separation_rejects_with_zeroed_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpherePlaneContact::new(&ctx);
    // Sphere five units above a y=0 plane with radius 1: signed = 5,
    // penetration = -4 < -tolerance, so the contact is rejected.
    let query = SpherePlaneContactQuery::new([0.0, 5.0, 0.0], 1.0, [0.0, 1.0, 0.0], 0.0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 0, "a separated sphere must be rejected");
    assert_eq!(
        got[0].normal,
        [0.0, 0.0, 0.0],
        "a rejected contact must zero the normal"
    );
    assert_eq!(got[0].point_a, [0.0, 0.0, 0.0], "a miss zeroes point_a");
    assert_eq!(got[0].point_b, [0.0, 0.0, 0.0], "a miss zeroes point_b");
    assert_eq!(got[0].penetration, 0.0, "a miss zeroes penetration");
    pin(0, &query, &got[0]);
}

#[test]
fn tangent_contact_touches_at_the_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpherePlaneContact::new(&ctx);
    // Sphere center one unit above the plane with radius 1: signed = 1,
    // penetration = 0, so the sphere just touches. Both contact points coincide
    // at the origin and the normal flips to -y.
    let query = SpherePlaneContactQuery::new([0.0, 1.0, 0.0], 1.0, [0.0, 1.0, 0.0], 0.0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "a tangent sphere is in contact");
    assert!(
        close(got[0].penetration, 0.0),
        "tangent penetration is zero"
    );
    assert!(close(got[0].normal[1], -1.0), "the contact normal is -y");
    pin(0, &query, &got[0]);
}

#[test]
fn deep_penetration_center_through_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpherePlaneContact::new(&ctx);
    // Center half a unit below the plane with radius 1: signed = -0.5,
    // penetration = 1.5 > radius, exercising the deep branch.
    let query = SpherePlaneContactQuery::new([0.0, -0.5, 0.0], 1.0, [0.0, 1.0, 0.0], 0.0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "a deeply penetrating sphere is in contact");
    assert!(
        close(got[0].penetration, 1.5),
        "deep penetration must be radius - signed = 1.5, got {}",
        got[0].penetration
    );
    pin(0, &query, &got[0]);
}

#[test]
fn non_axis_aligned_normal_contact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpherePlaneContact::new(&ctx);
    // Unit normal normalize([1, 1, 0]); center placed at normal * 0.5 so
    // signed = 0.5 and penetration = 0.5 with radius 1.
    let inv = 1.0 / 2.0_f32.sqrt();
    let normal = [inv, inv, 0.0];
    let center = [normal[0] * 0.5, normal[1] * 0.5, normal[2] * 0.5];
    let query = SpherePlaneContactQuery::new(center, 1.0, normal, 0.0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "the sphere is in contact");
    assert!(
        close(got[0].penetration, 0.5),
        "penetration must be 0.5, got {}",
        got[0].penetration
    );
    pin(0, &query, &got[0]);
}

#[test]
fn offset_plane_contact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpherePlaneContact::new(&ctx);
    // A nonzero world offset shifts the plane: signed = dot(n, center) - offset
    // = 3 - 2 = 1, penetration = 0.5 with radius 1.5.
    let query = SpherePlaneContactQuery::new([0.0, 3.0, 0.0], 1.5, [0.0, 1.0, 0.0], 2.0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(
        got[0].valid, 1,
        "the sphere is in contact with the offset plane"
    );
    assert!(
        close(got[0].penetration, 0.5),
        "penetration must be 0.5, got {}",
        got[0].penetration
    );
    pin(0, &query, &got[0]);
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpherePlaneContact::new(&ctx);
    // Two distinct queries, one hit and one miss: a wrong per-element stride
    // would cross-contaminate the two answers.
    let queries = [
        SpherePlaneContactQuery::new([0.0, 0.5, 0.0], 1.0, [0.0, 1.0, 0.0], 0.0),
        SpherePlaneContactQuery::new([0.0, 9.0, 0.0], 1.0, [0.0, 1.0, 0.0], 0.0),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 2, "both results must be returned");
    assert_eq!(got[0].valid, 1, "the first sphere is in contact");
    assert_eq!(got[1].valid, 0, "the second sphere is separated");
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpherePlaneContact::new(&ctx);
    let mut queries = vec![
        SpherePlaneContactQuery::new([0.0, 0.5, 0.0], 1.0, [0.0, 1.0, 0.0], 0.0),
        SpherePlaneContactQuery::new([0.0, 9.0, 0.0], 1.0, [0.0, 1.0, 0.0], 0.0),
        SpherePlaneContactQuery::new([0.0, -0.5, 0.0], 1.0, [0.0, 1.0, 0.0], 0.0),
        SpherePlaneContactQuery::new([1.2, -0.3, 0.7], 2.0, [0.267_26, 0.534_52, 0.801_78], 0.4),
    ];
    let mut state: u64 = 0x51A7_3C9D_0E12_4455;
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
    let gpu = GpuSpherePlaneContact::new(&ctx);
    let mut state: u64 = 0x0C3A_1F70_7B6E_9D11;
    let queries: Vec<SpherePlaneContactQuery> = (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
