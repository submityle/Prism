//! Real-device parity for the three-particle bending-projection twin:
//! [`GpuSoftBendingProject`](prism_volumetric_gpu::soft_bending_project::GpuSoftBendingProject)
//! must reproduce the `CPU` golden `project_bending` of
//! `prism_physics_core::soft::constraint::bending`, which resolves the fold at a
//! `center` particle toward the midpoint of its two neighbours by one compliant
//! `XPBD` Lagrange step and reports the updated positions and multiplier.
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! inverse-mass denominator, the midpoint and offset, the two degeneracy
//! rejections (all-pinned and coincident), the normal, the compliance term, the
//! multiplier delta and the three position updates — so the test never imports
//! `prism_physics_core` or `prism_render_architecture`.
//!
//! The fixtures cover a flat hinge essentially at rest, a tensile fold, a
//! compressive fold, a softened (non-zero compliance) step, a step driven by a
//! non-zero initial multiplier, an asymmetric-mass joint, an all-pinned joint
//! (rejected, pass-through), a coincident center-on-midpoint joint (rejected,
//! pass-through), and a batch of at least two distinct queries that catches any
//! `std430` stride aliasing. A sweep over random geometry and parameters
//! follows, with knee-point rejection sampling, plus an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every continuous channel threads through subtracts, divides and multiplies,
//! so `CPU` and `GPU` evaluate the same closed form but need not be bit-exact (a
//! device may fuse a multiply-add the scalar reference leaves separate). The
//! continuous comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::bending`；无第三方
//! 引擎源码或衍生代码。

use prism_volumetric_gpu::soft_bending_project::{GpuSoftBendingProject, SoftBendingProjectQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Offset length below which the normal is undefined, matching the golden
/// `crate::math::scalar::EPSILON`.
const EPSILON: f32 = 1.192_092_9e-7;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// The resolved oracle answer: three updated positions, the updated multiplier
/// and the validity flag.
struct Expected {
    pa: [f32; 3],
    pc: [f32; 3],
    pb: [f32; 3],
    lambda: f32,
    valid: u32,
}

/// The independent oracle for one query, mirroring the on-device kernel's branch
/// structure exactly, including the pass-through on either rejection.
fn oracle(q: &SoftBendingProjectQuery) -> Expected {
    let pa = [q.pax, q.pay, q.paz];
    let pc = [q.pcx, q.pcy, q.pcz];
    let pb = [q.pbx, q.pby, q.pbz];

    let denom_mass = q.wc + 0.25 * (q.wa + q.wb);
    let mass_ok = denom_mass > 0.0;

    let midpoint = [
        (pa[0] + pb[0]) * 0.5,
        (pa[1] + pb[1]) * 0.5,
        (pa[2] + pb[2]) * 0.5,
    ];
    let delta = [
        pc[0] - midpoint[0],
        pc[1] - midpoint[1],
        pc[2] - midpoint[2],
    ];
    let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    let length_ok = length >= EPSILON;

    let safe_length = if length_ok { length } else { 1.0 };
    let normal = [
        delta[0] / safe_length,
        delta[1] / safe_length,
        delta[2] / safe_length,
    ];

    let c = length - q.rest_offset;
    let alpha_tilde = q.compliance / (q.dt * q.dt);
    let delta_lambda = (-c - alpha_tilde * q.lambda) / (denom_mass + alpha_tilde);

    let new_pc = [
        pc[0] + normal[0] * (delta_lambda * q.wc),
        pc[1] + normal[1] * (delta_lambda * q.wc),
        pc[2] + normal[2] * (delta_lambda * q.wc),
    ];
    let new_pa = [
        pa[0] - normal[0] * (delta_lambda * q.wa * 0.5),
        pa[1] - normal[1] * (delta_lambda * q.wa * 0.5),
        pa[2] - normal[2] * (delta_lambda * q.wa * 0.5),
    ];
    let new_pb = [
        pb[0] - normal[0] * (delta_lambda * q.wb * 0.5),
        pb[1] - normal[1] * (delta_lambda * q.wb * 0.5),
        pb[2] - normal[2] * (delta_lambda * q.wb * 0.5),
    ];
    let new_lambda = q.lambda + delta_lambda;

    let accepted = mass_ok && length_ok;
    if accepted {
        Expected {
            pa: new_pa,
            pc: new_pc,
            pb: new_pb,
            lambda: new_lambda,
            valid: 1,
        }
    } else {
        Expected {
            pa,
            pc,
            pb,
            lambda: q.lambda,
            valid: 0,
        }
    }
}

/// Dispatches one query and asserts every updated channel plus the validity
/// flag.
fn assert_parity(ctx: &GpuContext, gpu: &GpuSoftBendingProject, q: SoftBendingProjectQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    assert_result(&got[0], &q);
}

/// Asserts parity for a whole batch, so the shared dispatch exercises the
/// `std430` stride.
fn assert_batch(
    ctx: &GpuContext,
    gpu: &GpuSoftBendingProject,
    queries: &[SoftBendingProjectQuery],
) {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_result(r, q);
    }
}

/// Compares one `GPU` result against the oracle for the same query.
fn assert_result(
    r: &prism_volumetric_gpu::soft_bending_project::SoftBendingProjectResult,
    q: &SoftBendingProjectQuery,
) {
    let e = oracle(q);
    assert_eq!(r.valid, e.valid, "valid flag mismatch: query={q:?}");
    let got = [
        r.new_pax,
        r.new_pay,
        r.new_paz,
        r.new_pcx,
        r.new_pcy,
        r.new_pcz,
        r.new_pbx,
        r.new_pby,
        r.new_pbz,
        r.new_lambda,
    ];
    let want = [
        e.pa[0], e.pa[1], e.pa[2], e.pc[0], e.pc[1], e.pc[2], e.pb[0], e.pb[1], e.pb[2], e.lambda,
    ];
    let labels = [
        "new_pax",
        "new_pay",
        "new_paz",
        "new_pcx",
        "new_pcy",
        "new_pcz",
        "new_pbx",
        "new_pby",
        "new_pbz",
        "new_lambda",
    ];
    for ((g, w), label) in got.iter().zip(want.iter()).zip(labels.iter()) {
        assert!(
            close(*g, *w),
            "{label} mismatch: gpu={g} cpu={w} query={q:?}"
        );
    }
}

#[test]
fn flat_hinge_at_rest_is_near_inert() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    // Center sits one unit above the midpoint with rest_offset = 1, so c ~= 0 and
    // the correction is tiny; the constraint still projects (valid = 1).
    assert_parity(
        &ctx,
        &gpu,
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            1.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
    );
}

#[test]
fn tensile_fold_pulls_center_back() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    // Center folded two units above the midpoint with zero rest offset: a rigid
    // step drives it back toward the midpoint.
    assert_parity(
        &ctx,
        &gpu,
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            0.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
    );
}

#[test]
fn compressive_fold_pushes_center_out() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    // Center only 0.25 above the midpoint but rest_offset = 1: c < 0 drives the
    // center outward away from the midpoint.
    assert_parity(
        &ctx,
        &gpu,
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 0.25, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            1.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
    );
}

#[test]
fn nonzero_compliance_softens_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    // A non-zero compliance adds alpha_tilde to the denominator, softening the
    // correction relative to the rigid case.
    assert_parity(
        &ctx,
        &gpu,
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            0.0,
            1.0e-3,
            0.0,
            1.0 / 60.0,
        ),
    );
}

#[test]
fn nonzero_initial_lambda_feeds_update() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    // A non-zero initial multiplier feeds the -alpha_tilde * lambda term, so the
    // compliance path exercises both the numerator and the accumulation.
    assert_parity(
        &ctx,
        &gpu,
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            0.5,
            2.0e-3,
            0.3,
            1.0 / 90.0,
        ),
    );
}

#[test]
fn asymmetric_masses_split_correction() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    // Distinct positive inverse masses distribute the correction unevenly across
    // the three particles while the denominator stays positive.
    assert_parity(
        &ctx,
        &gpu,
        SoftBendingProjectQuery::new(
            [[-1.0, 0.2, 0.1], [0.1, 1.5, -0.3], [1.0, -0.1, 0.2]],
            [0.5, 2.0, 1.25],
            0.4,
            0.0,
            0.0,
            1.0 / 120.0,
        ),
    );
}

#[test]
fn all_pinned_is_inert() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    // Every endpoint pinned (zero inverse mass): denom_mass = 0 so the constraint
    // is inert and all channels pass through unchanged (valid = 0).
    assert_parity(
        &ctx,
        &gpu,
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0],
            0.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
    );
}

#[test]
fn coincident_center_is_inert() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    // Center placed exactly on the midpoint: the offset length is below EPSILON,
    // the normal is undefined, so the constraint is inert and passes through
    // unchanged (valid = 0).
    assert_parity(
        &ctx,
        &gpu,
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            0.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
    );
}

#[test]
fn batch_stride_reads_non_aliased_slots() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    // A batch of several distinct queries exercises the std430 query/result
    // stride: every slot must read and write its own non-aliased data.
    let queries = [
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            0.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
        SoftBendingProjectQuery::new(
            [[-1.0, 0.2, 0.1], [0.1, 1.5, -0.3], [1.0, -0.1, 0.2]],
            [0.5, 2.0, 1.25],
            0.4,
            1.0e-3,
            0.2,
            1.0 / 120.0,
        ),
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0],
            0.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
        SoftBendingProjectQuery::new(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [0.0, 0.0, 0.0],
            0.0,
            0.0,
            0.0,
            1.0 / 60.0,
        ),
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
    let gpu = GpuSoftBendingProject::new(&ctx);
    let mut rng = Lcg::new(0x51_7D_1A_0C);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let pa = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let pc = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let pb = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let wa = rng.next_range(0.1, 2.0);
        let wc = rng.next_range(0.1, 2.0);
        let wb = rng.next_range(0.1, 2.0);
        let rest_offset = rng.next_range(0.0, 1.0);
        let compliance = rng.next_range(0.0, 0.01);
        let lambda = rng.next_range(-0.5, 0.5);
        let dt = rng.next_range(1.0 / 240.0, 1.0 / 30.0);

        // Reject-sample so no query sits on a branch knee where a last-bit split
        // could flip which arm the CPU and GPU take. The midpoint offset must
        // stay well above EPSILON, and the positive inverse masses already keep
        // denom_mass well above zero.
        let midpoint = [
            (pa[0] + pb[0]) * 0.5,
            (pa[1] + pb[1]) * 0.5,
            (pa[2] + pb[2]) * 0.5,
        ];
        let delta = [
            pc[0] - midpoint[0],
            pc[1] - midpoint[1],
            pc[2] - midpoint[2],
        ];
        let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if length < 1.0e-2 {
            continue;
        }

        queries.push(SoftBendingProjectQuery::new(
            [pa, pc, pb],
            [wa, wc, wb],
            rest_offset,
            compliance,
            lambda,
            dt,
        ));
    }

    assert_batch(&ctx, &gpu, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftBendingProject::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
