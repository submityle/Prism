//! Real-device parity test for the rigid-body generalized-inverse-mass twin.
//!
//! Each case evaluates one or more [`RigidGeneralizedInverseMassQuery`] values
//! on the GPU and pins the returned [`RigidGeneralizedInverseMassResult`]
//! against an independent `f32` reimplementation of
//! `prism_physics_core::solver::xpbd::rigid::generalized_inverse_mass`. The
//! oracle is rebuilt here from first principles; this test never depends on the
//! golden crate, and the crate carries no `glam` dev-dependency, so the cross
//! product, the column-major matrix-vector product and the dot product are all
//! hand-written in `f32`.
//!
//! The generalized inverse mass `w` is a continuous quantity threaded through a
//! cross product and a dot product, so it is pinned with an
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` tolerance (`REL_FLOOR = 1e-6`) that
//! absorbs a fused multiply-add the scalar reference leaves separate. The
//! discrete `valid` flag is pinned with an exact `==`; the formula has no
//! degenerate branch, so it is always `1`.
//!
//! The random sweep builds a symmetric positive-definite inverse inertia tensor
//! `R * diag(d) * R^T` from a normalized quaternion and strictly positive
//! eigenvalues, so `dot(rn, I * rn) >= 0` and the reference stays
//! well-conditioned.
//!
//! Every case short-circuits to a skip when no headless adapter is available,
//! so the suite is inert on a machine without a GPU and exercises the real
//! device elsewhere.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid::generalized_inverse_mass`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::rigid_generalized_inverse_mass::{
    GpuRigidGeneralizedInverseMass, RigidGeneralizedInverseMassQuery,
    RigidGeneralizedInverseMassResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor so a near-zero reference magnitude does not demand
/// an impossibly tight absolute match.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent `f32` reimplementation of `generalized_inverse_mass` for one
/// query, in the same arithmetic order as the kernel: `rn = cross(r, dir)`,
/// `I * rn` with `I` column-major (`col0, col1, col2`), then
/// `w = inv_mass + dot(rn, I * rn)`. There is no degenerate branch; `valid` is
/// always `1`.
fn oracle(query: &RigidGeneralizedInverseMassQuery) -> RigidGeneralizedInverseMassResult {
    let r = query.r;
    let dir = query.direction;
    let i = query.inv_inertia_world;

    // rn = cross(r, dir).
    let rn = [
        r[1] * dir[2] - r[2] * dir[1],
        r[2] * dir[0] - r[0] * dir[2],
        r[0] * dir[1] - r[1] * dir[0],
    ];

    // Column-major: col0 = (i[0], i[1], i[2]), col1 = (i[3], i[4], i[5]),
    // col2 = (i[6], i[7], i[8]); I * rn = col0*rn.x + col1*rn.y + col2*rn.z.
    let i_rn = [
        i[0] * rn[0] + i[3] * rn[1] + i[6] * rn[2],
        i[1] * rn[0] + i[4] * rn[1] + i[7] * rn[2],
        i[2] * rn[0] + i[5] * rn[1] + i[8] * rn[2],
    ];

    let w = query.inv_mass + (rn[0] * i_rn[0] + rn[1] * i_rn[1] + rn[2] * i_rn[2]);
    RigidGeneralizedInverseMassResult { w, valid: 1 }
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
/// `w` within tolerance, the discrete `valid` flag exact.
fn pin(
    idx: usize,
    query: &RigidGeneralizedInverseMassQuery,
    result: &RigidGeneralizedInverseMassResult,
) {
    let want = oracle(query);
    assert!(
        close(result.w, want.w),
        "query {idx}: w gpu={} oracle={} (inv_mass={} r={:?} dir={:?})",
        result.w,
        want.w,
        query.inv_mass,
        query.r,
        query.direction
    );
    assert_eq!(
        result.valid, want.valid,
        "query {idx}: valid gpu={} oracle={}",
        result.valid, want.valid
    );
}

/// Evaluates `queries` on-device and pins every result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuRigidGeneralizedInverseMass,
    queries: &[RigidGeneralizedInverseMassQuery],
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

/// Builds a column-major rotation matrix from a normalized quaternion
/// `(w, x, y, z)`. Returns the nine entries row-by-row; because the matrix is
/// used only to form a symmetric product, the storage order is irrelevant.
fn rotation_from_quat(mut q: [f32; 4]) -> [f32; 9] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let inv = if len < 1.0e-6 { 1.0 } else { 1.0 / len };
    for v in &mut q {
        *v *= inv;
    }
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// Builds a symmetric positive-definite inverse inertia tensor
/// `R * diag(d) * R^T` with strictly positive eigenvalues `d`, returned
/// column-major. Because the result is symmetric, column-major and row-major
/// storage coincide.
fn spd_inertia(r: &[f32; 9], d: [f32; 3]) -> [f32; 9] {
    // r is laid out row-major as [r00,r01,r02, r10,r11,r12, r20,r21,r22].
    // out[i][j] = sum_k R[i][k] * d[k] * R[j][k].
    let mut out = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0f32;
            for k in 0..3 {
                acc += r[i * 3 + k] * d[k] * r[j * 3 + k];
            }
            out[i * 3 + j] = acc;
        }
    }
    out
}

/// A deterministic, well-conditioned random query: a positive inverse mass, a
/// symmetric positive-definite inverse inertia tensor, a random lever arm and a
/// random unit direction.
fn rand_query(state: &mut u64) -> RigidGeneralizedInverseMassQuery {
    let inv_mass = range(state, 0.1, 5.0);

    let quat = [
        range(state, -1.0, 1.0),
        range(state, -1.0, 1.0),
        range(state, -1.0, 1.0),
        range(state, -1.0, 1.0),
    ];
    let rot = rotation_from_quat(quat);
    let eigen = [
        range(state, 0.2, 3.0),
        range(state, 0.2, 3.0),
        range(state, 0.2, 3.0),
    ];
    let inertia = spd_inertia(&rot, eigen);

    let r = [
        range(state, -3.0, 3.0),
        range(state, -3.0, 3.0),
        range(state, -3.0, 3.0),
    ];

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
    let direction = [dir[0] / len, dir[1] / len, dir[2] / len];

    RigidGeneralizedInverseMassQuery::new(inv_mass, inertia, r, direction)
}

/// The 3x3 identity, column-major.
const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidGeneralizedInverseMass::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty input must return an empty vector");
}

#[test]
fn zero_lever_arm_collapses_to_inv_mass() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidGeneralizedInverseMass::new(&ctx);
    // With r = 0 the cross product is zero, so w collapses to inv_mass by plain
    // arithmetic regardless of the inertia tensor or direction.
    let query = RigidGeneralizedInverseMassQuery::new(
        0.75,
        [2.0, 0.3, -0.1, 0.3, 1.5, 0.2, -0.1, 0.2, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert!(
        close(got[0].w, 0.75),
        "a zero lever arm must give w == inv_mass, got {}",
        got[0].w
    );
    assert_eq!(got[0].valid, 1, "the result is always valid");
    pin(0, &query, &got[0]);
}

#[test]
fn lever_parallel_to_direction_collapses_to_inv_mass() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidGeneralizedInverseMass::new(&ctx);
    // r parallel to direction gives cross(r, dir) = 0, so again w == inv_mass.
    let query = RigidGeneralizedInverseMassQuery::new(
        1.25,
        [2.0, 0.3, -0.1, 0.3, 1.5, 0.2, -0.1, 0.2, 3.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    );
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert!(
        close(got[0].w, 1.25),
        "a lever arm parallel to the direction must give w == inv_mass, got {}",
        got[0].w
    );
    assert_eq!(got[0].valid, 1, "the result is always valid");
    pin(0, &query, &got[0]);
}

#[test]
fn identity_inertia_known_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidGeneralizedInverseMass::new(&ctx);
    // r = +x, dir = +y so rn = cross(x, y) = +z; with identity inertia
    // dot(rn, I*rn) = 1, hence w = inv_mass + 1 = 1.5.
    let query =
        RigidGeneralizedInverseMassQuery::new(0.5, IDENTITY, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&query));
    assert!(
        close(got[0].w, 1.5),
        "identity inertia with orthogonal r/dir must give w = 1.5, got {}",
        got[0].w
    );
    assert_eq!(got[0].valid, 1, "the result is always valid");
    pin(0, &query, &got[0]);
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidGeneralizedInverseMass::new(&ctx);
    // Two distinct queries: a wrong per-element stride would cross-contaminate
    // the two answers. The first has a nonzero rn, the second has rn = 0.
    let queries = [
        RigidGeneralizedInverseMassQuery::new(0.5, IDENTITY, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        RigidGeneralizedInverseMassQuery::new(2.0, IDENTITY, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), 2, "both results must be returned");
    assert!(
        close(got[0].w, 1.5),
        "first query must give w = 1.5, got {}",
        got[0].w
    );
    assert!(
        close(got[1].w, 2.0),
        "second query (zero lever) must give w = inv_mass = 2.0, got {}",
        got[1].w
    );
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidGeneralizedInverseMass::new(&ctx);
    let mut queries = vec![
        RigidGeneralizedInverseMassQuery::new(0.5, IDENTITY, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        RigidGeneralizedInverseMassQuery::new(2.0, IDENTITY, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        RigidGeneralizedInverseMassQuery::new(
            1.0,
            [2.0, 0.3, -0.1, 0.3, 1.5, 0.2, -0.1, 0.2, 3.0],
            [0.5, -1.2, 0.8],
            [0.0, 0.0, 1.0],
        ),
        RigidGeneralizedInverseMassQuery::new(
            0.25,
            [1.4, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 2.1],
            [1.5, 2.0, -0.5],
            [0.267_26, 0.534_52, 0.801_78],
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
    let gpu = GpuRigidGeneralizedInverseMass::new(&ctx);
    let mut state: u64 = 0x0CEA_1F10_7A6B_9D55;
    let queries: Vec<RigidGeneralizedInverseMassQuery> =
        (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
