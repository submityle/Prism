//! Real-device parity for the half-space projection twin:
//! [`GpuSoftProjectOutOfHalfSpace`](prism_volumetric_gpu::soft_project_out_of_half_space::GpuSoftProjectOutOfHalfSpace)
//! must reproduce the `CPU` golden `project_out_of_half_space` of
//! `prism_physics_core::soft::collision::body`, which keeps a particle on the
//! feasible side of the plane `normal . x == offset` by pushing an
//! infeasible-side point along `normal` onto the plane.
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! squared-length zero-normal rejection, the signed plane distance, the
//! feasible-side early-out and the `t = -signed / len_sq` projection — so the
//! test never imports `prism_physics_core` or `prism_render_architecture`.
//!
//! The fixtures cover an infeasible-side point projected onto the plane (checked
//! by `normal . new_pos ~= offset`), a non-unit normal (so the `1 / |normal|^2`
//! scale is exercised), a point already on the feasible side (echoed), a point
//! essentially on the plane (echoed via a boundary margin), a zero normal
//! (echoed, no plane), and a batch of at least two distinct queries that catches
//! any `std430` stride aliasing. A sweep over random geometry follows, with
//! knee-point rejection sampling, plus an empty batch the host short-circuits
//! with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every continuous channel threads through a dot product, a divide and a
//! multiply-add, so `CPU` and `GPU` evaluate the same closed form but need not
//! be bit-exact (a device may fuse a multiply-add the scalar reference leaves
//! separate). The continuous comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body`；无第三方
//! 引擎源码或衍生代码。

use prism_volumetric_gpu::soft_project_out_of_half_space::{
    GpuSoftProjectOutOfHalfSpace, SoftProjectOutOfHalfSpaceQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Squared length below which a normal has no defined plane, matching the golden
/// `crate::soft::collision::EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// The resolved oracle answer: the resolved position and the validity flag.
struct Expected {
    pos: [f32; 3],
    valid: u32,
}

/// The independent oracle for one query, mirroring the on-device kernel's branch
/// structure exactly, including the pass-through on either rejection.
fn oracle(q: &SoftProjectOutOfHalfSpaceQuery) -> Expected {
    let pos = [q.posx, q.posy, q.posz];
    let normal = [q.nx, q.ny, q.nz];

    let len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    // Zero normal has no defined plane: echo the position.
    if len_sq <= EPS_LEN_SQ {
        return Expected { pos, valid: 0 };
    }

    let signed = normal[0] * pos[0] + normal[1] * pos[1] + normal[2] * pos[2] - q.offset;
    // Already on the feasible side (including on the plane): echo.
    if signed >= 0.0 {
        return Expected { pos, valid: 0 };
    }

    let t_push = -signed / len_sq;
    Expected {
        pos: [
            pos[0] + normal[0] * t_push,
            pos[1] + normal[1] * t_push,
            pos[2] + normal[2] * t_push,
        ],
        valid: 1,
    }
}

/// Dispatches one query and asserts every resolved channel plus the validity
/// flag.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuSoftProjectOutOfHalfSpace,
    q: SoftProjectOutOfHalfSpaceQuery,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    assert_result(&got[0], &q);
}

/// Asserts parity for a whole batch, so the shared dispatch exercises the
/// `std430` stride.
fn assert_batch(
    ctx: &GpuContext,
    gpu: &GpuSoftProjectOutOfHalfSpace,
    queries: &[SoftProjectOutOfHalfSpaceQuery],
) {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_result(r, q);
    }
}

/// Compares one `GPU` result against the oracle for the same query.
fn assert_result(
    r: &prism_volumetric_gpu::soft_project_out_of_half_space::SoftProjectOutOfHalfSpaceResult,
    q: &SoftProjectOutOfHalfSpaceQuery,
) {
    let e = oracle(q);
    assert_eq!(r.valid, e.valid, "valid flag mismatch: query={q:?}");
    let got = [r.new_posx, r.new_posy, r.new_posz];
    let want = e.pos;
    let labels = ["new_posx", "new_posy", "new_posz"];
    for ((g, w), label) in got.iter().zip(want.iter()).zip(labels.iter()) {
        assert!(
            close(*g, *w),
            "{label} mismatch: gpu={g} cpu={w} query={q:?}"
        );
    }
}

#[test]
fn infeasible_point_is_projected_onto_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfHalfSpace::new(&ctx);
    // Point below the plane z = 0 (unit normal +z, offset 0): signed = -1 < 0, so
    // it is pushed up onto the plane at z = 0 (valid = 1).
    let q = SoftProjectOutOfHalfSpaceQuery::new([0.3, -0.7, -1.0], [0.0, 0.0, 1.0], 0.0);
    assert_parity(&ctx, &gpu, q);
    // Confirm the resolved point lands on the plane: normal . new_pos ~= offset.
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let r = &got[0];
    let dot = q.nx * r.new_posx + q.ny * r.new_posy + q.nz * r.new_posz;
    assert!(
        close(dot, q.offset),
        "projected point must lie on the plane: dot={dot} offset={}",
        q.offset
    );
}

#[test]
fn non_unit_normal_still_lands_on_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfHalfSpace::new(&ctx);
    // Normal of length 3 (|n|^2 = 9): the 1 / |normal|^2 scale must still place
    // the point exactly on the plane 2x + 2y + z = 4.
    let q = SoftProjectOutOfHalfSpaceQuery::new([0.0, 0.0, 0.0], [2.0, 2.0, 1.0], 4.0);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let r = &got[0];
    let dot = q.nx * r.new_posx + q.ny * r.new_posy + q.nz * r.new_posz;
    assert!(
        close(dot, q.offset),
        "non-unit normal must still land on the plane: dot={dot} offset={}",
        q.offset
    );
}

#[test]
fn feasible_side_point_is_echoed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfHalfSpace::new(&ctx);
    // Point above the plane z = 0: signed = +2 >= 0, already feasible, so it is
    // echoed unchanged (valid = 0).
    assert_parity(
        &ctx,
        &gpu,
        SoftProjectOutOfHalfSpaceQuery::new([0.5, -0.5, 2.0], [0.0, 0.0, 1.0], 0.0),
    );
}

#[test]
fn on_plane_boundary_is_echoed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfHalfSpace::new(&ctx);
    // Point a small positive margin above the plane: signed > 0 but tiny, so it
    // is on the feasible side and echoed (valid = 0). The margin keeps CPU and
    // GPU on the same side of the signed >= 0 branch.
    assert_parity(
        &ctx,
        &gpu,
        SoftProjectOutOfHalfSpaceQuery::new([1.0, -2.0, 1.0e-2], [0.0, 0.0, 1.0], 0.0),
    );
}

#[test]
fn zero_normal_is_echoed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfHalfSpace::new(&ctx);
    // A zero normal has no defined plane: len_sq <= EPS_LEN_SQ, so the point is
    // echoed unchanged (valid = 0) rather than dividing by zero.
    assert_parity(
        &ctx,
        &gpu,
        SoftProjectOutOfHalfSpaceQuery::new([0.7, -0.3, 0.9], [0.0, 0.0, 0.0], 0.0),
    );
}

#[test]
fn batch_stride_reads_non_aliased_slots() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfHalfSpace::new(&ctx);
    // A batch of several distinct queries exercises the std430 query/result
    // stride: every slot must read and write its own non-aliased data.
    let queries = [
        SoftProjectOutOfHalfSpaceQuery::new([0.3, -0.7, -1.0], [0.0, 0.0, 1.0], 0.0),
        SoftProjectOutOfHalfSpaceQuery::new([0.0, 0.0, 0.0], [2.0, 2.0, 1.0], 4.0),
        SoftProjectOutOfHalfSpaceQuery::new([0.5, -0.5, 2.0], [0.0, 0.0, 1.0], 0.0),
        SoftProjectOutOfHalfSpaceQuery::new([0.7, -0.3, 0.9], [0.0, 0.0, 0.0], 0.0),
        SoftProjectOutOfHalfSpaceQuery::new([-1.2, 0.4, 0.6], [1.0, -0.5, 0.25], -0.3),
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
    let gpu = GpuSoftProjectOutOfHalfSpace::new(&ctx);
    let mut rng = Lcg::new(0x51_7D_1A_0C);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let pos = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let normal = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let offset = rng.next_range(-2.0, 2.0);

        // Reject-sample so no query sits on a branch knee where a last-bit split
        // could flip which arm the CPU and GPU take: keep len_sq well above
        // EPS_LEN_SQ (away from the zero-normal gate) and |signed| above a margin
        // (away from the feasible-side gate).
        let len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if len_sq < 1.0e-2 {
            continue;
        }
        let signed = normal[0] * pos[0] + normal[1] * pos[1] + normal[2] * pos[2] - offset;
        if signed.abs() < 1.0e-2 {
            continue;
        }

        queries.push(SoftProjectOutOfHalfSpaceQuery::new(pos, normal, offset));
    }

    assert_batch(&ctx, &gpu, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfHalfSpace::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
