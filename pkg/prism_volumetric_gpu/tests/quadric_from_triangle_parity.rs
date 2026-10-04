//! Real-device parity test for the triangle plane-quadric twin.
//!
//! Each case evaluates one or more [`QuadricFromTriangleQuery`] values on the
//! GPU and pins the returned [`QuadricFromTriangleResult`] against an
//! independent `f32` reimplementation of
//! `prism_physics_core::collider::quadric::Quadric::from_triangle`. The oracle
//! is rebuilt here from first principles; this test never depends on the golden
//! crate, and the crate carries no `glam` dev-dependency, so the cross product,
//! normalisation and ten plane-quadric coefficients are hand-written in `f32`
//! in the same arithmetic order as the kernel.
//!
//! The ten coefficients are continuous quantities threaded through
//! subtractions, a cross product, a square root, a reciprocal and products, so
//! each is pinned with an `abs_diff <= 1e-4 || rel_diff <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`) that absorbs a fused multiply-add the scalar reference
//! leaves separate. The discrete `valid` flag is pinned with an exact `==`; the
//! all-zero degenerate quadric is a legitimate output, so `valid` is always
//! `1`.
//!
//! Every case short-circuits to a skip when no headless adapter is available,
//! so the suite is inert on a machine without a GPU and exercises the real
//! device elsewhere.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric::Quadric::from_triangle`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::quadric_from_triangle::{
    GpuQuadricFromTriangle, QuadricFromTriangleQuery, QuadricFromTriangleResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor so a near-zero reference magnitude does not demand
/// an impossibly tight absolute match.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent `f32` reimplementation of `Quadric::from_triangle` for one
/// query, in the same arithmetic order as the kernel: `normal = (b-a)x(c-a)`,
/// `len = length(normal)`; a `len <= f32::MIN_POSITIVE` triangle returns the
/// all-zero quadric; otherwise `n = normal/len`, `d = -dot(n, a)` and the ten
/// coefficients follow the golden `from_plane` operator order. `valid = 1`.
fn oracle(query: &QuadricFromTriangleQuery) -> QuadricFromTriangleResult {
    let [ax, ay, az] = query.a;
    let [bx, by, bz] = query.b;
    let [cx, cy, cz] = query.c;

    let e1 = [bx - ax, by - ay, bz - az];
    let e2 = [cx - ax, cy - ay, cz - az];
    let normal = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();

    if len <= f32::MIN_POSITIVE {
        return QuadricFromTriangleResult {
            a2: 0.0,
            ab: 0.0,
            ac: 0.0,
            ad: 0.0,
            b2: 0.0,
            bc: 0.0,
            bd: 0.0,
            c2: 0.0,
            cd: 0.0,
            d2: 0.0,
            valid: 1,
        };
    }

    let inv = 1.0 / len;
    let nx = normal[0] * inv;
    let ny = normal[1] * inv;
    let nz = normal[2] * inv;
    let d = -(nx * ax + ny * ay + nz * az);

    QuadricFromTriangleResult {
        a2: nx * nx,
        ab: nx * ny,
        ac: nx * nz,
        ad: nx * d,
        b2: ny * ny,
        bc: ny * nz,
        bd: ny * d,
        c2: nz * nz,
        cd: nz * d,
        d2: d * d,
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

/// Pins one `GPU` result against the independent host oracle: the ten
/// continuous coefficients within tolerance, the discrete `valid` flag exact.
fn pin(idx: usize, query: &QuadricFromTriangleQuery, result: &QuadricFromTriangleResult) {
    let want = oracle(query);
    let pairs = [
        ("a2", result.a2, want.a2),
        ("ab", result.ab, want.ab),
        ("ac", result.ac, want.ac),
        ("ad", result.ad, want.ad),
        ("b2", result.b2, want.b2),
        ("bc", result.bc, want.bc),
        ("bd", result.bd, want.bd),
        ("c2", result.c2, want.c2),
        ("cd", result.cd, want.cd),
        ("d2", result.d2, want.d2),
    ];
    for (name, got, expected) in pairs {
        assert!(
            close(got, expected),
            "query {idx}: coeff {name} gpu={got} oracle={expected}"
        );
    }
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuQuadricFromTriangle, queries: &[QuadricFromTriangleQuery]) {
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

/// Asserts every coefficient of `result` is exactly zero and `valid` is `1`,
/// the signature of a degenerate (zero-area) triangle.
fn assert_zero_quadric(result: &QuadricFromTriangleResult) {
    assert_eq!(result.a2, 0.0, "degenerate a2 must be exactly zero");
    assert_eq!(result.ab, 0.0, "degenerate ab must be exactly zero");
    assert_eq!(result.ac, 0.0, "degenerate ac must be exactly zero");
    assert_eq!(result.ad, 0.0, "degenerate ad must be exactly zero");
    assert_eq!(result.b2, 0.0, "degenerate b2 must be exactly zero");
    assert_eq!(result.bc, 0.0, "degenerate bc must be exactly zero");
    assert_eq!(result.bd, 0.0, "degenerate bd must be exactly zero");
    assert_eq!(result.c2, 0.0, "degenerate c2 must be exactly zero");
    assert_eq!(result.cd, 0.0, "degenerate cd must be exactly zero");
    assert_eq!(result.d2, 0.0, "degenerate d2 must be exactly zero");
    assert_eq!(result.valid, 1, "degenerate quadric is still valid");
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

/// A deterministic random query with each vertex coordinate in `[-5, 5]`.
///
/// Samples whose cross-product length is tiny are rejected and regenerated: a
/// near-degenerate triangle sits on the `len <= f32::MIN_POSITIVE` knee where a
/// fused multiply-add can flip the host and device onto opposite arms of the
/// guard, so the sweep stays well clear of it.
fn rand_query(state: &mut u64) -> QuadricFromTriangleQuery {
    loop {
        let a = [
            range(state, -5.0, 5.0),
            range(state, -5.0, 5.0),
            range(state, -5.0, 5.0),
        ];
        let b = [
            range(state, -5.0, 5.0),
            range(state, -5.0, 5.0),
            range(state, -5.0, 5.0),
        ];
        let c = [
            range(state, -5.0, 5.0),
            range(state, -5.0, 5.0),
            range(state, -5.0, 5.0),
        ];
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len >= 1.0e-2 {
            return QuadricFromTriangleQuery::new(a, b, c);
        }
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromTriangle::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty input must return an empty vector");
}

#[test]
fn axis_aligned_triangle_in_xy_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromTriangle::new(&ctx);
    // Triangle in the z=0 plane: normal = +Z, d = -dot(n, a) = 0, so the only
    // non-zero coefficient is c2 = n.z^2 = 1.
    let query = QuadricFromTriangleQuery::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_eq!(got[0].valid, 1, "a non-degenerate triangle is valid");
    assert!(close(got[0].c2, 1.0), "c2 must be 1, got {}", got[0].c2);
    assert!(close(got[0].a2, 0.0), "a2 must be 0, got {}", got[0].a2);
    assert!(close(got[0].d2, 0.0), "d2 must be 0, got {}", got[0].d2);
    pin(0, &query, &got[0]);
}

#[test]
fn offset_plane_carries_nonzero_d() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromTriangle::new(&ctx);
    // Triangle in the z=2 plane: normal = +Z, d = -2, so c2 = 1, cd = -2,
    // d2 = 4.
    let query = QuadricFromTriangleQuery::new([0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert!(close(got[0].c2, 1.0), "c2 must be 1, got {}", got[0].c2);
    assert!(close(got[0].cd, -2.0), "cd must be -2, got {}", got[0].cd);
    assert!(close(got[0].d2, 4.0), "d2 must be 4, got {}", got[0].d2);
    pin(0, &query, &got[0]);
}

#[test]
fn tilted_triangle_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromTriangle::new(&ctx);
    // A triangle with a normal in a general octant exercises all ten
    // coefficients at once.
    let queries = [
        QuadricFromTriangleQuery::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        QuadricFromTriangleQuery::new([-1.0, 2.0, 3.0], [2.0, -1.0, 0.5], [0.3, 4.0, -2.0]),
        QuadricFromTriangleQuery::new([2.0, 2.0, 2.0], [-3.0, 1.0, 0.0], [1.0, -4.0, 5.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn degenerate_collinear_returns_zero_quadric() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromTriangle::new(&ctx);
    // Three collinear points: cross product is zero, so the quadric is all-zero.
    let query = QuadricFromTriangleQuery::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_zero_quadric(&got[0]);
    pin(0, &query, &got[0]);
}

#[test]
fn degenerate_coincident_returns_zero_quadric() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromTriangle::new(&ctx);
    // Two coincident vertices (b == a): zero-area triangle, all-zero quadric.
    let query = QuadricFromTriangleQuery::new([3.0, -1.0, 2.0], [3.0, -1.0, 2.0], [5.0, 0.0, 1.0]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert_zero_quadric(&got[0]);
    pin(0, &query, &got[0]);
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromTriangle::new(&ctx);
    // Two distinct queries: a wrong per-element stride would cross-contaminate
    // the two answers.
    let queries = [
        QuadricFromTriangleQuery::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        QuadricFromTriangleQuery::new([-1.0, 2.0, 3.0], [2.0, -1.0, 0.5], [0.3, 4.0, -2.0]),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 2, "both results must be returned");
    assert!(close(got[0].c2, 1.0), "first triangle has c2 = 1");
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuQuadricFromTriangle::new(&ctx);
    let mut queries = vec![
        QuadricFromTriangleQuery::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        QuadricFromTriangleQuery::new([0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]),
        // Degenerate collinear case mixed into the batch.
        QuadricFromTriangleQuery::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]),
        QuadricFromTriangleQuery::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
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
    let gpu = GpuQuadricFromTriangle::new(&ctx);
    let mut state: u64 = 0x0C3A_1F70_7B6E_9D11;
    let queries: Vec<QuadricFromTriangleQuery> = (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
