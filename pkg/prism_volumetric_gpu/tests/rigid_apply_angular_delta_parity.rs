//! Real-device parity for the angular-integration twin:
//! [`GpuRigidApplyAngularDelta`](prism_volumetric_gpu::rigid_apply_angular_delta::GpuRigidApplyAngularDelta)
//! must reproduce the `CPU` golden `apply_angular_delta` of
//! `prism_physics_core::solver::xpbd::rigid`, which advances a rigid body's
//! orientation by a small rotation vector `dphi` using the first-order update
//! `q' = normalize(q + 0.5 * [dphi, 0] q)`.
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! Hamilton product `[dphi, 0] * orientation`, the half-step accumulation, the
//! squared-length degeneracy guard and the `1 / sqrt(len_sq)` renormalization —
//! written in pure `f32` without `glam`, so the test never imports
//! `prism_physics_core` or `prism_render_architecture`.
//!
//! The fixtures cover a zero `dphi` (the orientation is only renormalized, so it
//! passes through unchanged), a single-axis small rotation off the identity
//! (checking the first-order update direction and that the result stays unit
//! length), a non-identity normalized orientation with a small `dphi`, a
//! degenerate collapse (`len_sq <= 1e-20`, echoed with `valid = 0`), and a batch
//! of at least two distinct queries that catches any `std430` stride aliasing. A
//! sweep over random near-unit orientations with small `dphi` follows, plus an
//! empty batch the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every continuous channel threads through a product, a divide and a square
//! root, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a device may fuse a multiply-add the scalar reference leaves
//! separate). The continuous comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::solver::xpbd::rigid`；无第三方
//! 引擎源码或衍生代码。

use prism_volumetric_gpu::rigid_apply_angular_delta::{
    GpuRigidApplyAngularDelta, RigidApplyAngularDeltaQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Squared length below which the accumulated quaternion has no defined
/// direction, matching the golden threshold in `apply_angular_delta`.
const LEN_SQ_EPS: f32 = 1.0e-20;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// The resolved oracle answer: the resolved quaternion and the validity flag.
struct Expected {
    out: [f32; 4],
    valid: u32,
}

/// The independent oracle for one query, mirroring the on-device kernel's branch
/// structure exactly, including the echo pass-through on a degenerate collapse.
/// Written in pure `f32` with no `glam` dependency.
fn oracle(q: &RigidApplyAngularDeltaQuery) -> Expected {
    let (bx, by, bz, bw) = (q.bx, q.by, q.bz, q.bw);
    let (ax, ay, az) = (q.ax, q.ay, q.az);

    // Hamilton product dq = [dphi, 0] * orientation (self = omega, rhs = q).
    let dqx = ax * bw + ay * bz - az * by;
    let dqy = -ax * bz + ay * bw + az * bx;
    let dqz = ax * by - ay * bx + az * bw;
    let dqw = -(ax * bx + ay * by + az * bz);

    // Half-step accumulation.
    let ux = bx + 0.5 * dqx;
    let uy = by + 0.5 * dqy;
    let uz = bz + 0.5 * dqz;
    let uw = bw + 0.5 * dqw;

    let len_sq = ux * ux + uy * uy + uz * uz + uw * uw;
    // A collapsed quaternion has no defined direction: echo the input.
    if len_sq <= LEN_SQ_EPS {
        return Expected {
            out: [bx, by, bz, bw],
            valid: 0,
        };
    }

    let inv_len = 1.0 / len_sq.sqrt();
    Expected {
        out: [ux * inv_len, uy * inv_len, uz * inv_len, uw * inv_len],
        valid: 1,
    }
}

/// Dispatches one query and asserts every resolved channel plus the validity
/// flag.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuRigidApplyAngularDelta,
    q: RigidApplyAngularDeltaQuery,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    assert_result(&got[0], &q);
}

/// Asserts parity for a whole batch, so the shared dispatch exercises the
/// `std430` stride.
fn assert_batch(
    ctx: &GpuContext,
    gpu: &GpuRigidApplyAngularDelta,
    queries: &[RigidApplyAngularDeltaQuery],
) {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_result(r, q);
    }
}

/// Compares one `GPU` result against the oracle for the same query.
fn assert_result(
    r: &prism_volumetric_gpu::rigid_apply_angular_delta::RigidApplyAngularDeltaResult,
    q: &RigidApplyAngularDeltaQuery,
) {
    let e = oracle(q);
    assert_eq!(r.valid, e.valid, "valid flag mismatch: query={q:?}");
    let got = [r.outx, r.outy, r.outz, r.outw];
    let want = e.out;
    let labels = ["outx", "outy", "outz", "outw"];
    for ((g, w), label) in got.iter().zip(want.iter()).zip(labels.iter()) {
        assert!(
            close(*g, *w),
            "{label} mismatch: gpu={g} cpu={w} query={q:?}"
        );
    }
}

/// Returns the Euclidean length of a quaternion, used to confirm an integrated
/// result stays on the unit sphere.
fn quat_len(
    r: &prism_volumetric_gpu::rigid_apply_angular_delta::RigidApplyAngularDeltaResult,
) -> f32 {
    (r.outx * r.outx + r.outy * r.outy + r.outz * r.outz + r.outw * r.outw).sqrt()
}

#[test]
fn zero_delta_passes_orientation_through() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyAngularDelta::new(&ctx);
    // A zero rotation vector leaves dq = 0, so updated = orientation; a unit
    // input is simply renormalized back to itself (valid = 1).
    let q = RigidApplyAngularDeltaQuery::new([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert!(
        close(quat_len(&got[0]), 1.0),
        "renormalized quaternion must stay unit length: len={}",
        quat_len(&got[0])
    );
}

#[test]
fn single_axis_small_rotation_advances_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyAngularDelta::new(&ctx);
    // Identity orientation with a small rotation about +x: the first-order update
    // bumps the x component by ~0.5 * dphi.x and keeps w near 1, then
    // renormalizes to unit length (valid = 1).
    let q = RigidApplyAngularDeltaQuery::new([0.0, 0.0, 0.0, 1.0], [0.02, 0.0, 0.0]);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let r = &got[0];
    assert!(r.outx > 0.0, "rotation about +x must grow outx: {r:?}");
    assert!(
        close(quat_len(r), 1.0),
        "integrated quaternion must stay unit length: len={}",
        quat_len(r)
    );
}

#[test]
fn non_identity_orientation_integrates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyAngularDelta::new(&ctx);
    // A normalized non-axis-aligned orientation advanced by a small dphi: the
    // continuous channels must match the oracle and stay unit length.
    let raw = [0.3_f32, -0.4, 0.5, 0.7];
    let inv = 1.0 / (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]).sqrt();
    let orientation = [raw[0] * inv, raw[1] * inv, raw[2] * inv, raw[3] * inv];
    let q = RigidApplyAngularDeltaQuery::new(orientation, [0.01, -0.015, 0.02]);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert!(
        close(quat_len(&got[0]), 1.0),
        "integrated quaternion must stay unit length: len={}",
        quat_len(&got[0])
    );
}

#[test]
fn degenerate_collapse_is_echoed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyAngularDelta::new(&ctx);
    // A zero orientation with a zero dphi accumulates to the zero quaternion:
    // len_sq <= 1e-20, so the input is echoed unchanged (valid = 0) rather than
    // dividing by zero.
    let q = RigidApplyAngularDeltaQuery::new([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].valid, 0, "degenerate collapse must be invalid");
}

#[test]
fn batch_stride_reads_non_aliased_slots() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyAngularDelta::new(&ctx);
    // A batch of several distinct queries exercises the std430 query/result
    // stride: every slot must read and write its own non-aliased data.
    let queries = [
        RigidApplyAngularDeltaQuery::new([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        RigidApplyAngularDeltaQuery::new([0.0, 0.0, 0.0, 1.0], [0.02, 0.0, 0.0]),
        RigidApplyAngularDeltaQuery::new([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.03]),
        RigidApplyAngularDeltaQuery::new([0.5, 0.5, 0.5, 0.5], [-0.01, 0.02, -0.015]),
        RigidApplyAngularDeltaQuery::new([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
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
    let gpu = GpuRigidApplyAngularDelta::new(&ctx);
    let mut rng = Lcg::new(0x2B_17_93_C5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let raw = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let len_sq = raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3];
        // Reject-sample near-zero quaternions so the normalized orientation is
        // well defined; a unit orientation plus a small dphi then guarantees the
        // accumulated len_sq stays far above the 1e-20 degeneracy knee, keeping
        // CPU and GPU on the same (valid = 1) branch.
        if len_sq < 1.0e-2 {
            continue;
        }
        let inv = 1.0 / len_sq.sqrt();
        let orientation = [raw[0] * inv, raw[1] * inv, raw[2] * inv, raw[3] * inv];
        let dphi = [
            rng.next_range(-0.05, 0.05),
            rng.next_range(-0.05, 0.05),
            rng.next_range(-0.05, 0.05),
        ];
        queries.push(RigidApplyAngularDeltaQuery::new(orientation, dphi));
    }

    assert_batch(&ctx, &gpu, &queries);

    // Every swept query is non-degenerate, so each result must be unit length.
    let results = gpu.evaluate(&ctx, &queries);
    for r in &results {
        assert!(
            close(quat_len(r), 1.0),
            "swept quaternion must stay unit length: len={}",
            quat_len(r)
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRigidApplyAngularDelta::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
