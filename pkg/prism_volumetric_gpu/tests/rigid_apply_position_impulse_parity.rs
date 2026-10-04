//! Real-device parity for the positional-impulse apply twin:
//! [`GpuRigidApplyPositionImpulse`](prism_volumetric_gpu::rigid_apply_position_impulse::GpuRigidApplyPositionImpulse)
//! must reproduce the `CPU` golden
//! `prism_physics_core::solver::xpbd::rigid::apply_position_impulse`, the
//! `XPBD` update that applies a positional impulse `p` at a world lever arm `r`
//! to a rigid body's position and orientation.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! `new_position = position + inv_mass * p` (always committed);
//! `dphi = I (r x p)` with `I` a column-major world inverse-inertia matrix;
//! the first-order quaternion delta `omega = (dphi, 0)`,
//! `dq = omega (x) orientation` (Hamilton product),
//! `updated = orientation + 0.5 * dq`, and
//! `new_orientation = len_sq > 1e-20 ? updated / sqrt(len_sq) : orientation` —
//! written out directly so the test never imports `prism_physics_core` or
//! `prism_render_architecture`, nor `glam`.
//!
//! The fixtures cover the branches the kernel must honor: a general non-trivial
//! pose with a normalized quaternion and a symmetric-positive-definite inverse
//! inertia renormalizes with `valid = 1`; a collapsed orientation update
//! (`len_sq <= 1e-20`, built from a zero quaternion so every `dq` term vanishes)
//! echoes the orientation with `valid = 0` while still committing the linear
//! translation; a multi-element batch validates the `std430` stride; and a
//! `512`-step sweep over random normalized quaternions, positive inverse masses,
//! diagonal positive-definite inverse inertias, lever arms and impulses follows,
//! kept well-conditioned so `len_sq` stays near one, far from the renormalize
//! knee. An empty batch is short-circuited on the host with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The position (`new_px`, `new_py`, `new_pz`) and orientation (`new_qx`,
//! `new_qy`, `new_qz`, `new_qw`) are continuous `f32` outputs, so parity uses an
//! absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`, with a `1e-6`
//! relative floor so near-zero components compare on the absolute leg). `valid`
//! is discrete and compared exactly: `1` when the orientation was renormalized,
//! `0` when the update collapsed and the input orientation was echoed.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::rigid_apply_position_impulse::{
    GpuRigidApplyPositionImpulse, RigidApplyPositionImpulseQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Squared-length guard matching the kernel's orientation renormalize knee.
const LEN_SQ_KNEE: f32 = 1.0e-20;
/// Absolute tolerance leg for the continuous comparisons.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance leg for the continuous comparisons.
const REL_EPS: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero components fall back to the absolute
/// leg instead of demanding an impossible relative match.
const REL_FLOOR: f32 = 1.0e-6;

/// Absolute-or-relative closeness for a single `f32` lane.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff <= REL_EPS * scale
}

/// Independent oracle for one query, returning `([new_px, new_py, new_pz],
/// [new_qx, new_qy, new_qz, new_qw], valid)`.
///
/// Reproduces the body of `apply_position_impulse` operator by operator, in the
/// same order as the kernel: the linear translation is always committed; the
/// angular delta is the half of the Hamilton product `(dphi, 0) (x) orientation`
/// with `dphi = I (r x p)`; the orientation is renormalized when its squared
/// length clears `1e-20` and otherwise echoed with `valid = 0`.
fn oracle(q: &RigidApplyPositionImpulseQuery) -> ([f32; 3], [f32; 4], u32) {
    let new_pos = [
        q.px + q.inv_mass * q.ipx,
        q.py + q.inv_mass * q.ipy,
        q.pz + q.inv_mass * q.ipz,
    ];
    // rp = cross(r, p).
    let rp = [
        q.ry * q.ipz - q.rz * q.ipy,
        q.rz * q.ipx - q.rx * q.ipz,
        q.rx * q.ipy - q.ry * q.ipx,
    ];
    // dphi = col0 * rp.x + col1 * rp.y + col2 * rp.z (column-major).
    let dphi = [
        q.i0 * rp[0] + q.i3 * rp[1] + q.i6 * rp[2],
        q.i1 * rp[0] + q.i4 * rp[1] + q.i7 * rp[2],
        q.i2 * rp[0] + q.i5 * rp[1] + q.i8 * rp[2],
    ];
    let (ax, ay, az) = (dphi[0], dphi[1], dphi[2]);
    let (bx, by, bz, bw) = (q.qx, q.qy, q.qz, q.qw);
    // dq = (ax, ay, az, 0) (x) orientation, Hamilton product.
    let dqx = ax * bw + ay * bz - az * by;
    let dqy = -ax * bz + ay * bw + az * bx;
    let dqz = ax * by - ay * bx + az * bw;
    let dqw = -(ax * bx + ay * by + az * bz);
    let u = [
        bx + 0.5 * dqx,
        by + 0.5 * dqy,
        bz + 0.5 * dqz,
        bw + 0.5 * dqw,
    ];
    let len_sq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3];
    if len_sq > LEN_SQ_KNEE {
        let inv = 1.0 / len_sq.sqrt();
        (new_pos, [u[0] * inv, u[1] * inv, u[2] * inv, u[3] * inv], 1)
    } else {
        (new_pos, [bx, by, bz, bw], 0)
    }
}

/// Dispatches one query and asserts the device result matches the oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuRigidApplyPositionImpulse,
    q: RigidApplyPositionImpulseQuery,
) {
    let results = gpu.evaluate(ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let (pos, quat, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close(r.new_px, pos[0]) && close(r.new_py, pos[1]) && close(r.new_pz, pos[2]),
        "position mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
        r.new_px,
        r.new_py,
        r.new_pz,
        pos[0],
        pos[1],
        pos[2]
    );
    assert!(
        close(r.new_qx, quat[0])
            && close(r.new_qy, quat[1])
            && close(r.new_qz, quat[2])
            && close(r.new_qw, quat[3]),
        "orientation mismatch: query={q:?} gpu=({}, {}, {}, {}) oracle=({}, {}, {}, {})",
        r.new_qx,
        r.new_qy,
        r.new_qz,
        r.new_qw,
        quat[0],
        quat[1],
        quat[2],
        quat[3]
    );
}

/// Builds a query from a (not necessarily normalized) quaternion, normalizing it
/// so the fixture represents a valid rigid-body orientation, with a diagonal
/// positive-definite world inverse inertia.
#[expect(
    clippy::too_many_arguments,
    reason = "the fixture mirrors the flat scalar query layout one group at a time"
)]
fn query(
    pos: [f32; 3],
    quat: [f32; 4],
    inv_mass: f32,
    inertia_diag: [f32; 3],
    lever: [f32; 3],
    impulse: [f32; 3],
) -> RigidApplyPositionImpulseQuery {
    let qlen = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3])
        .sqrt()
        .max(REL_FLOOR);
    let inv = 1.0 / qlen;
    RigidApplyPositionImpulseQuery::new(
        pos[0],
        pos[1],
        pos[2],
        quat[0] * inv,
        quat[1] * inv,
        quat[2] * inv,
        quat[3] * inv,
        inv_mass,
        inertia_diag[0],
        0.0,
        0.0,
        0.0,
        inertia_diag[1],
        0.0,
        0.0,
        0.0,
        inertia_diag[2],
        lever[0],
        lever[1],
        lever[2],
        impulse[0],
        impulse[1],
        impulse[2],
    )
}

#[test]
fn general_pose_renormalizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyPositionImpulse::new(&ctx);
    // A non-trivial normalized orientation, a positive inverse mass, a
    // positive-definite inverse inertia and a non-zero impulse at an off-axis
    // lever arm: the orientation renormalizes (valid = 1) and the position is
    // translated.
    assert_parity(
        &ctx,
        &gpu,
        query(
            [1.5, -0.5, 2.0],
            [0.5, 0.5, 0.5, 0.5],
            0.25,
            [1.2, 0.8, 1.6],
            [0.3, -0.4, 0.2],
            [2.0, -1.0, 0.5],
        ),
    );
}

#[test]
fn off_axis_impulse_renormalizes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyPositionImpulse::new(&ctx);
    // A second well-conditioned pose with a different orientation and inertia so
    // the Hamilton product and the column-major matrix multiply are exercised on
    // non-symmetric data.
    assert_parity(
        &ctx,
        &gpu,
        query(
            [-2.0, 3.0, 0.0],
            [0.1, -0.3, 0.2, 0.9],
            0.5,
            [0.6, 2.0, 1.1],
            [1.0, 0.5, -0.5],
            [-0.8, 1.2, 0.6],
        ),
    );
}

#[test]
fn zero_quaternion_echoes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyPositionImpulse::new(&ctx);
    // A zero orientation makes every Hamilton-product term vanish, so the
    // updated quaternion is zero (len_sq = 0 <= 1e-20): the kernel must echo the
    // orientation with valid = 0 while still committing the linear translation.
    // The query is built directly (not through `query`, which would normalize).
    let q = RigidApplyPositionImpulseQuery::new(
        0.5, -1.0, 2.0, // position
        0.0, 0.0, 0.0, 0.0, // zero orientation
        0.5, // inverse mass
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // identity inverse inertia
        0.4, -0.2, 0.7, // lever
        1.0, 2.0, -1.5, // impulse
    );
    let results = gpu.evaluate(&ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    assert_eq!(r.valid, 0, "collapsed orientation must report valid = 0");
    // Linear translation still committed.
    assert!(
        close(r.new_px, 0.5 + 0.5 * 1.0)
            && close(r.new_py, -1.0 + 0.5 * 2.0)
            && close(r.new_pz, 2.0 + 0.5 * -1.5),
        "collapsed-case position should still translate: gpu=({}, {}, {})",
        r.new_px,
        r.new_py,
        r.new_pz
    );
    // Orientation echoed unchanged (the input zero quaternion).
    assert!(
        close(r.new_qx, 0.0)
            && close(r.new_qy, 0.0)
            && close(r.new_qz, 0.0)
            && close(r.new_qw, 0.0),
        "collapsed-case orientation should echo input: gpu=({}, {}, {}, {})",
        r.new_qx,
        r.new_qy,
        r.new_qz,
        r.new_qw
    );
}

#[test]
fn zero_impulse_identity_angular() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyPositionImpulse::new(&ctx);
    // A zero impulse leaves dphi = 0, so the orientation is unchanged after
    // renormalization (valid = 1) and the position is unchanged too; a clean
    // identity check through the full pipeline.
    assert_parity(
        &ctx,
        &gpu,
        query(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
            0.3,
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
        ),
    );
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyPositionImpulse::new(&ctx);
    // A multi-element batch exercises the std430 stride: adjacent 23-lane slots
    // must decode independently and in order, spanning a general pose, the
    // collapsed zero-orientation echo and a second general pose.
    let general_a = query(
        [1.5, -0.5, 2.0],
        [0.5, 0.5, 0.5, 0.5],
        0.25,
        [1.2, 0.8, 1.6],
        [0.3, -0.4, 0.2],
        [2.0, -1.0, 0.5],
    );
    let collapsed = RigidApplyPositionImpulseQuery::new(
        0.5, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.4,
        -0.2, 0.7, 1.0, 2.0, -1.5,
    );
    let general_b = query(
        [-2.0, 3.0, 0.0],
        [0.1, -0.3, 0.2, 0.9],
        0.5,
        [0.6, 2.0, 1.1],
        [1.0, 0.5, -0.5],
        [-0.8, 1.2, 0.6],
    );
    let queries = [general_a, collapsed, general_b];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (pos, quat, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close(r.new_px, pos[0]) && close(r.new_py, pos[1]) && close(r.new_pz, pos[2]),
            "batch position mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
            r.new_px,
            r.new_py,
            r.new_pz,
            pos[0],
            pos[1],
            pos[2]
        );
        assert!(
            close(r.new_qx, quat[0])
                && close(r.new_qy, quat[1])
                && close(r.new_qz, quat[2])
                && close(r.new_qw, quat[3]),
            "batch orientation mismatch: query={q:?} gpu=({}, {}, {}, {}) oracle=({}, {}, {}, {})",
            r.new_qx,
            r.new_qy,
            r.new_qz,
            r.new_qw,
            quat[0],
            quat[1],
            quat[2],
            quat[3]
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyPositionImpulse::new(&ctx);
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
    let gpu = GpuRigidApplyPositionImpulse::new(&ctx);
    let mut rng = Lcg::new(0x1D_3A_C5_09);
    // Every query uses a normalized quaternion plus a small angular delta, so
    // the updated squared length stays near one — far from the 1e-20
    // renormalize knee — making the valid = 1 branch decisive. The inverse
    // inertia is diagonal with positive entries, a valid symmetric
    // positive-definite world inverse inertia.
    const QUAT_FLOOR: f32 = 0.1;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let qx = rng.next_range(-1.0, 1.0);
        let qy = rng.next_range(-1.0, 1.0);
        let qz = rng.next_range(-1.0, 1.0);
        let qw = rng.next_range(-1.0, 1.0);
        let qlen_sq = qx * qx + qy * qy + qz * qz + qw * qw;
        // Reject near-zero quaternions so the normalize path is well-conditioned.
        if qlen_sq < QUAT_FLOOR {
            continue;
        }
        let pos = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let inv_mass = rng.next_range(0.1, 2.0);
        let inertia_diag = [
            rng.next_range(0.5, 2.0),
            rng.next_range(0.5, 2.0),
            rng.next_range(0.5, 2.0),
        ];
        // Small lever and impulse keep the angular delta small so the updated
        // quaternion stays near unit length.
        let lever = [
            rng.next_range(-0.5, 0.5),
            rng.next_range(-0.5, 0.5),
            rng.next_range(-0.5, 0.5),
        ];
        let impulse = [
            rng.next_range(-0.5, 0.5),
            rng.next_range(-0.5, 0.5),
            rng.next_range(-0.5, 0.5),
        ];
        queries.push(query(
            pos,
            [qx, qy, qz, qw],
            inv_mass,
            inertia_diag,
            lever,
            impulse,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (pos, quat, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close(r.new_px, pos[0]) && close(r.new_py, pos[1]) && close(r.new_pz, pos[2]),
            "sweep position mismatch: query={q:?} gpu=({}, {}, {}) oracle=({}, {}, {})",
            r.new_px,
            r.new_py,
            r.new_pz,
            pos[0],
            pos[1],
            pos[2]
        );
        assert!(
            close(r.new_qx, quat[0])
                && close(r.new_qy, quat[1])
                && close(r.new_qz, quat[2])
                && close(r.new_qw, quat[3]),
            "sweep orientation mismatch: query={q:?} gpu=({}, {}, {}, {}) oracle=({}, {}, {}, {})",
            r.new_qx,
            r.new_qy,
            r.new_qz,
            r.new_qw,
            quat[0],
            quat[1],
            quat[2],
            quat[3]
        );
    }
}
