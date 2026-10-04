//! Real-device parity for the world-space inverse-inertia twin:
//! [`GpuRigidInvInertiaWorld`](prism_volumetric_gpu::rigid_inv_inertia_world::GpuRigidInvInertiaWorld)
//! must reproduce the `CPU` golden `inv_inertia_world` of
//! `prism_physics_core::solver::xpbd::rigid`, which rotates a body's diagonal
//! inverse inertia into world space as the symmetric tensor `R *
//! diag(inv_inertia) * R^T`, where `R = from_quat(q)`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the `glam` column-major quaternion-to-matrix basis, the per-column scaling
//! and the column-major triple product — written out directly in flat `f32`
//! array math so the test never imports `prism_physics_core`,
//! `prism_render_architecture` or `glam`. It mirrors the reference operation
//! for operation and in the same evaluation order.
//!
//! The fixtures cover the regimes the kernel must honor: an identity quaternion
//! with unit inertia (the identity tensor), a `90`-degree rotation about `Z`
//! that swaps the `X`/`Y` diagonal entries (exercising the `R^T` terms), a
//! tilted non-axis-aligned quaternion with distinct inertia, an all-zero
//! inertia (the zero tensor), plus a multi-element mixed batch that validates
//! the `std430` array stride end to end. A sweep over random normalized
//! quaternions and random inverse inertia follows, plus an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The triple product threads through several multiply-adds, so `CPU` and `GPU`
//! evaluate the same closed form but need not be bit-exact. The continuous
//! comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on
//! each of the nine matrix entries; the discrete `valid` flag is compared
//! exactly. The kernel has no degenerate branch, so no comparison sits on a
//! branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid::inv_inertia_world`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::rigid_inv_inertia_world::{
    GpuRigidInvInertiaWorld, RigidInvInertiaWorldQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Returns `true` when two column-major `3x3` matrices agree entry-wise within
/// tolerance.
fn close9(a: [f32; 9], b: [f32; 9]) -> bool {
    (0..9).all(|k| close(a[k], b[k]))
}

/// Independent host re-implementation of the golden `inv_inertia_world`,
/// returning the column-major world-space inverse inertia tensor and the
/// `valid` flag without importing the golden crate or `glam`. The quaternion
/// basis, the per-column scaling and the triple product are evaluated in the
/// same order as the kernel.
fn oracle(q: &RigidInvInertiaWorldQuery) -> ([f32; 9], u32) {
    let ix = q.inv_inertia[0];
    let iy = q.inv_inertia[1];
    let iz = q.inv_inertia[2];
    let x = q.q[0];
    let y = q.q[1];
    let z = q.q[2];
    let w = q.q[3];

    // R = from_quat(q), glam column-major basis.
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    let col0 = [1.0 - (yy + zz), xy + wz, xz - wy];
    let col1 = [xy - wz, 1.0 - (xx + zz), yz + wx];
    let col2 = [xz + wy, yz - wx, 1.0 - (xx + yy)];

    // Scale each column by its inverse-inertia component.
    let s0 = [col0[0] * ix, col0[1] * ix, col0[2] * ix];
    let s1 = [col1[0] * iy, col1[1] * iy, col1[2] * iy];
    let s2 = [col2[0] * iz, col2[1] * iz, col2[2] * iz];

    // result = scaled * R^T; R^T column j = R row j = (col0[j], col1[j], col2[j]).
    // result column j component k = s0[k]*rowj[0] + s1[k]*rowj[1] + s2[k]*rowj[2].
    let col = |rj0: f32, rj1: f32, rj2: f32| -> [f32; 3] {
        [
            s0[0] * rj0 + s1[0] * rj1 + s2[0] * rj2,
            s0[1] * rj0 + s1[1] * rj1 + s2[1] * rj2,
            s0[2] * rj0 + s1[2] * rj1 + s2[2] * rj2,
        ]
    };
    let rc0 = col(col0[0], col1[0], col2[0]);
    let rc1 = col(col0[1], col1[1], col2[1]);
    let rc2 = col(col0[2], col1[2], col2[2]);

    (
        [
            rc0[0], rc0[1], rc0[2], rc1[0], rc1[1], rc1[2], rc2[0], rc2[1], rc2[2],
        ],
        1,
    )
}

/// Dispatches a single query and asserts the device output matches the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuRigidInvInertiaWorld, q: RigidInvInertiaWorldQuery) {
    let results = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let (m, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close9(r.m, m),
        "matrix mismatch: gpu={:?} cpu={m:?} query={q:?}",
        r.m
    );
}

#[test]
fn identity_quat_unit_inertia_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidInvInertiaWorld::new(&ctx);
    // Identity orientation with unit inverse inertia is the identity tensor.
    let q = RigidInvInertiaWorldQuery::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1);
    assert!(
        close9(r.m, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        "identity quaternion with unit inertia is the identity tensor"
    );
}

#[test]
fn ninety_degrees_about_z_swaps_xy_diagonal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidInvInertiaWorld::new(&ctx);
    // A +90-degree rotation about Z sends body-X to world-Y and body-Y to
    // world -X, so the world tensor swaps the X/Y diagonal entries: with
    // inertia (2, 3, 5) the diagonal becomes (3, 2, 5).
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let q = RigidInvInertiaWorldQuery::new([2.0, 3.0, 5.0], [0.0, 0.0, s, s]);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1);
    assert!(
        close9(r.m, [3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0]),
        "90-degree Z rotation swaps the X/Y diagonal entries, got {:?}",
        r.m
    );
}

#[test]
fn tilted_quaternion_distinct_inertia_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidInvInertiaWorld::new(&ctx);
    // A tilted, non-axis-aligned normalized quaternion with distinct inertia:
    // the oracle is the source of truth, and the result stays symmetric.
    let raw = [0.3_f32, -0.5, 0.7, 0.4];
    let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]).sqrt();
    let q = RigidInvInertiaWorldQuery::new(
        [0.8, 1.5, 2.3],
        [raw[0] / norm, raw[1] / norm, raw[2] / norm, raw[3] / norm],
    );
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1);
    // The world inverse inertia tensor is symmetric.
    assert!(
        close(r.m[1], r.m[3]),
        "tensor should be symmetric (m01=m10)"
    );
    assert!(
        close(r.m[2], r.m[6]),
        "tensor should be symmetric (m02=m20)"
    );
    assert!(
        close(r.m[5], r.m[7]),
        "tensor should be symmetric (m12=m21)"
    );
}

#[test]
fn zero_inertia_is_zero_tensor() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidInvInertiaWorld::new(&ctx);
    // A zero inverse inertia yields the zero tensor for any orientation; valid
    // is still 1 because the closed form has no degenerate branch.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let q = RigidInvInertiaWorldQuery::new([0.0, 0.0, 0.0], [s, 0.0, 0.0, s]);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1);
    assert!(
        close9(r.m, [0.0; 9]),
        "zero inertia produces the zero tensor, got {:?}",
        r.m
    );
}

#[test]
fn multi_element_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidInvInertiaWorld::new(&ctx);
    // A mixed batch (identity, Z-rotation, tilted, zero inertia) exercises the
    // std430 array stride: every slot must decode at the right byte offset.
    let s = std::f32::consts::FRAC_1_SQRT_2;
    let raw = [0.3_f32, -0.5, 0.7, 0.4];
    let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]).sqrt();
    let queries = [
        RigidInvInertiaWorldQuery::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]),
        RigidInvInertiaWorldQuery::new([2.0, 3.0, 5.0], [0.0, 0.0, s, s]),
        RigidInvInertiaWorldQuery::new(
            [0.8, 1.5, 2.3],
            [raw[0] / norm, raw[1] / norm, raw[2] / norm, raw[3] / norm],
        ),
        RigidInvInertiaWorldQuery::new([0.0, 0.0, 0.0], [s, 0.0, 0.0, s]),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (m, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close9(r.m, m),
            "batch matrix mismatch: gpu={:?} cpu={m:?} query={q:?}",
            r.m
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidInvInertiaWorld::new(&ctx);
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
    let gpu = GpuRigidInvInertiaWorld::new(&ctx);
    let mut rng = Lcg::new(0x5F_2C_81_A3);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // Random inverse inertia, including occasional zeros via the range
        // reaching 0; all finite and well-conditioned.
        let inv_inertia = [
            rng.next_range(0.0, 4.0),
            rng.next_range(0.0, 4.0),
            rng.next_range(0.0, 4.0),
        ];
        // Normalized quaternion: draw components in [-1, 1] and regenerate
        // until the norm is comfortably away from zero, then normalize.
        let (qx, qy, qz, qw) = loop {
            let a = rng.next_range(-1.0, 1.0);
            let b = rng.next_range(-1.0, 1.0);
            let c = rng.next_range(-1.0, 1.0);
            let d = rng.next_range(-1.0, 1.0);
            let n2 = a * a + b * b + c * c + d * d;
            if n2 > 0.04 {
                let inv = 1.0 / n2.sqrt();
                break (a * inv, b * inv, c * inv, d * inv);
            }
        };
        queries.push(RigidInvInertiaWorldQuery::new(
            inv_inertia,
            [qx, qy, qz, qw],
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (m, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close9(r.m, m),
            "sweep matrix mismatch: gpu={:?} cpu={m:?} query={q:?}",
            r.m
        );
        // The world inverse inertia tensor is symmetric for every input.
        assert!(close(r.m[1], r.m[3]), "sweep tensor asymmetry (m01!=m10)");
        assert!(close(r.m[2], r.m[6]), "sweep tensor asymmetry (m02!=m20)");
        assert!(close(r.m[5], r.m[7]), "sweep tensor asymmetry (m12!=m21)");
    }
}
