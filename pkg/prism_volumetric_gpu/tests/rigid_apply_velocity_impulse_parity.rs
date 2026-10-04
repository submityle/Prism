//! Real-device parity for the rigid-body velocity-impulse twin:
//! [`GpuRigidApplyVelocityImpulse`](prism_volumetric_gpu::rigid_apply_velocity_impulse::GpuRigidApplyVelocityImpulse)
//! must reproduce the `CPU` golden `apply_velocity_impulse` of
//! `prism_physics_core::solver::xpbd::rigid`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the linear update `new_lin = linear_velocity + inv_mass * p`, the torque
//! impulse `rp = cross(r, p)` and the angular update
//! `new_ang = angular_velocity + I * rp`, where `I` is the world
//! inverse-inertia stored column-major and the product expands to
//! `I * rp = col0 * rp.x + col1 * rp.y + col2 * rp.z` — written out directly
//! with scalar `f32` arithmetic so the test never imports `prism_physics_core`,
//! `prism_render_architecture` or `glam`.
//!
//! The fixtures cover a general non-trivial body (non-zero linear and angular
//! velocity, non-zero lever arm and impulse, a symmetric positive-definite
//! inverse-inertia), a zero-inverse-mass zero-lever-arm identity (the velocities
//! pass through unchanged), a mixed `>=2`-element batch that validates the
//! `std430` stride end to end, and a `512`-query `LCG` sweep. An empty batch is
//! short-circuited by the host with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every velocity component is continuous and checked with an
//! absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`,
//! `REL_FLOOR = 1e-6`); `valid` is discrete and compared exactly. The update
//! has no degenerate branch and no division, so there is no conditioning knee —
//! the only subtlety the sweep exercises is the column-major matrix layout and
//! the cross-product sign convention shared by host and device.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::rigid_apply_velocity_impulse::{
    GpuRigidApplyVelocityImpulse, RigidApplyVelocityImpulseQuery, RigidApplyVelocityImpulseResult,
};
use prism_volumetric_gpu::GpuContext;

/// Independent host re-implementation of `apply_velocity_impulse`, flattened
/// into the `(new_lin, new_ang, valid)` record the twin encodes. The matrix is
/// read column-major from the query scalars (`m0..m2` = col0, `m3..m5` = col1,
/// `m6..m8` = col2) and the cross product uses the right-handed convention, so
/// no extra `f32` error is introduced relative to the device. No
/// `prism_physics_core` / `glam` import.
fn oracle(q: &RigidApplyVelocityImpulseQuery) -> RigidApplyVelocityImpulseResult {
    // new_lin = lin + inv_mass * p.
    let new_linx = q.linx + q.inv_mass * q.px;
    let new_liny = q.liny + q.inv_mass * q.py;
    let new_linz = q.linz + q.inv_mass * q.pz;

    // rp = cross(r, p), right-handed.
    let rpx = q.ry * q.pz - q.rz * q.py;
    let rpy = q.rz * q.px - q.rx * q.pz;
    let rpz = q.rx * q.py - q.ry * q.px;

    // I * rp with I column-major: component k = col0[k]*rp.x + col1[k]*rp.y + col2[k]*rp.z.
    let new_angx = q.angx + q.m0 * rpx + q.m3 * rpy + q.m6 * rpz;
    let new_angy = q.angy + q.m1 * rpx + q.m4 * rpy + q.m7 * rpz;
    let new_angz = q.angz + q.m2 * rpx + q.m5 * rpy + q.m8 * rpz;

    RigidApplyVelocityImpulseResult {
        linx: new_linx,
        liny: new_liny,
        linz: new_linz,
        angx: new_angx,
        angy: new_angy,
        angz: new_angz,
        valid: 1,
    }
}

/// Absolute-or-relative closeness for a continuous channel.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    if diff <= 1e-4 {
        return true;
    }
    let rel_floor = 1e-6_f32;
    let denom = want.abs().max(got.abs()).max(rel_floor);
    diff / denom <= 1e-3
}

/// Asserts one GPU result matches the oracle. The six velocity components are
/// continuous (tolerance); `valid` is discrete (exact).
fn assert_result(
    got: RigidApplyVelocityImpulseResult,
    want: RigidApplyVelocityImpulseResult,
    label: &str,
) {
    assert_eq!(got.valid, want.valid, "valid mismatch: {label}");
    assert!(
        close(got.linx, want.linx),
        "linx mismatch: {label}: got {} want {}",
        got.linx,
        want.linx
    );
    assert!(
        close(got.liny, want.liny),
        "liny mismatch: {label}: got {} want {}",
        got.liny,
        want.liny
    );
    assert!(
        close(got.linz, want.linz),
        "linz mismatch: {label}: got {} want {}",
        got.linz,
        want.linz
    );
    assert!(
        close(got.angx, want.angx),
        "angx mismatch: {label}: got {} want {}",
        got.angx,
        want.angx
    );
    assert!(
        close(got.angy, want.angy),
        "angy mismatch: {label}: got {} want {}",
        got.angy,
        want.angy
    );
    assert!(
        close(got.angz, want.angz),
        "angz mismatch: {label}: got {} want {}",
        got.angz,
        want.angz
    );
}

/// Asserts a single-query GPU result matches the oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuRigidApplyVelocityImpulse,
    q: RigidApplyVelocityImpulseQuery,
    label: &str,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query: {label}");
    assert_result(got[0], oracle(&q), label);
}

/// Builds a symmetric positive-definite `3x3` from a seed matrix `a` as
/// `M = A * A^T + 0.1 * I`, returned as three columns `[col0, col1, col2]`.
/// Because `M` is symmetric, column `j` equals row `j`.
fn spd_columns(a: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut m = [[0.0_f32; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for k in 0..3 {
                sum += a[i][k] * a[j][k];
            }
            *cell = sum + if i == j { 0.1 } else { 0.0 };
        }
    }
    // Columns of a symmetric matrix: col_j = (M[0][j], M[1][j], M[2][j]).
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[test]
fn general_non_trivial_body() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyVelocityImpulse::new(&ctx);
    // Non-zero linear and angular velocity, non-zero lever arm and impulse, a
    // symmetric positive-definite inverse-inertia from a generic seed matrix.
    let cols = spd_columns([[1.3, 0.4, -0.2], [0.1, 0.9, 0.5], [-0.3, 0.2, 1.1]]);
    let q = RigidApplyVelocityImpulseQuery::new(
        [0.5, -1.2, 2.0],
        [0.3, 0.7, -0.4],
        0.75,
        cols,
        [0.6, -0.3, 0.9],
        [2.0, 1.5, -1.0],
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: update always valid");
    assert_parity(&ctx, &gpu, q, "general_non_trivial_body");
}

#[test]
fn zero_inv_mass_zero_lever_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyVelocityImpulse::new(&ctx);
    // inv_mass = 0 cancels the linear term and r = 0 makes rp = 0, so both
    // velocities pass through unchanged — the pure identity path.
    let cols = spd_columns([[0.8, 0.2, 0.0], [0.1, 1.4, 0.3], [0.0, 0.2, 0.7]]);
    let q = RigidApplyVelocityImpulseQuery::new(
        [1.1, -0.5, 0.25],
        [-0.6, 0.9, 1.3],
        0.0,
        cols,
        [0.0, 0.0, 0.0],
        [3.0, -2.0, 1.0],
    );
    let want = oracle(&q);
    assert!(
        close(want.linx, 1.1) && close(want.liny, -0.5) && close(want.linz, 0.25),
        "fixture sanity: linear velocity unchanged"
    );
    assert!(
        close(want.angx, -0.6) && close(want.angy, 0.9) && close(want.angz, 1.3),
        "fixture sanity: angular velocity unchanged"
    );
    assert_parity(&ctx, &gpu, q, "zero_inv_mass_zero_lever_is_identity");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyVelocityImpulse::new(&ctx);
    // A >=2-element batch mixing identity, diagonal and dense inverse-inertia
    // validates the std430 body/result stride end to end.
    let diag = spd_columns([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let dense = spd_columns([[0.7, -0.5, 0.3], [0.2, 1.1, -0.4], [-0.1, 0.6, 0.9]]);
    let queries = vec![
        RigidApplyVelocityImpulseQuery::new(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.0,
            diag,
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ),
        RigidApplyVelocityImpulseQuery::new(
            [1.0, -1.0, 0.5],
            [0.2, 0.3, -0.1],
            2.0,
            diag,
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ),
        RigidApplyVelocityImpulseQuery::new(
            [-0.4, 0.8, 1.2],
            [0.5, -0.6, 0.7],
            1.25,
            dense,
            [0.3, 0.9, -0.5],
            [-1.5, 0.4, 2.1],
        ),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("mixed batch index {i}"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyVelocityImpulse::new(&ctx);
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
    let gpu = GpuRigidApplyVelocityImpulse::new(&ctx);
    let mut rng = Lcg::new(0x7F_2C_11_A3);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let lin = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let ang = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let inv_mass = rng.next_range(0.0, 4.0);
        let seed = [
            [
                rng.next_range(-1.5, 1.5),
                rng.next_range(-1.5, 1.5),
                rng.next_range(-1.5, 1.5),
            ],
            [
                rng.next_range(-1.5, 1.5),
                rng.next_range(-1.5, 1.5),
                rng.next_range(-1.5, 1.5),
            ],
            [
                rng.next_range(-1.5, 1.5),
                rng.next_range(-1.5, 1.5),
                rng.next_range(-1.5, 1.5),
            ],
        ];
        let cols = spd_columns(seed);
        let r = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let p = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        queries.push(RigidApplyVelocityImpulseQuery::new(
            lin, ang, inv_mass, cols, r, p,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("sweep index {i}"));
    }
}
