//! Real-device parity for the shape-goal pull-back twin:
//! [`GpuSoftPullToTarget`](prism_volumetric_gpu::soft_pull_to_target::GpuSoftPullToTarget)
//! must reproduce the `CPU` golden
//! `prism_physics_core::soft::constraint::linear_stiffness::project_pull_to_target`,
//! the `TressFX`/groom-style projection that pulls each free strand particle a
//! fraction of the way toward its authored geometric target.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! `moved = (w > 0) && (stiffness > 0)` and
//! `new_p = moved ? p + (target - p) * stiffness : p` — written out directly so
//! the test never imports `prism_physics_core` or `prism_render_architecture`.
//! It mirrors the reference per-particle body: a free particle (inverse mass
//! strictly positive) under a positive stiffness is blended toward its target,
//! while a pinned particle or a non-positive stiffness holds position.
//!
//! The slice iteration, the short-`targets` skip and the sequential sweep
//! bookkeeping are not twinned: this kernel is the pure per-particle geometric
//! pull, one thread per particle.
//!
//! The fixtures cover the branches the kernel must honor: `stiffness == 1` snaps
//! onto the target, `stiffness == 0.5` moves halfway, a pinned particle
//! (`w == 0`) holds, a non-positive stiffness (`0` and negative) holds, and a
//! mixed multi-element batch validates the `std430` stride. A `512`-step sweep
//! over random positions, targets, inverse masses and stiffness follows, kept a
//! safe margin away from the `w == 0` and `stiffness == 0` knees (which the
//! named fixtures already pin), plus an empty batch the host short-circuits with
//! no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The position (`new_px`, `new_py`, `new_pz`) is a continuous `f32` blend, so
//! parity uses an absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`,
//! with a `1e-6` relative floor so near-zero coordinates compare on the absolute
//! leg). `valid` is discrete and compared exactly; it is always `1` because the
//! golden has no rejected input — a pinned particle or a non-positive stiffness
//! is a legal no-op, not a degenerate case.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::linear_stiffness`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_pull_to_target::{GpuSoftPullToTarget, SoftPullToTargetQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance leg for the continuous position comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance leg for the continuous position comparison.
const REL_EPS: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero coordinates fall back to the absolute
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
/// valid)`.
///
/// Reproduces the per-particle body of `project_pull_to_target`: a particle
/// moves only when it is free (`w > 0`) and the stiffness is positive, and a
/// moving particle is blended `stiffness` of the way to its target. `valid` is
/// always `1`.
fn oracle(q: &SoftPullToTargetQuery) -> ([f32; 3], u32) {
    let moved = q.w > 0.0 && q.stiffness > 0.0;
    let new_p = if moved {
        [
            q.px + (q.tx - q.px) * q.stiffness,
            q.py + (q.ty - q.py) * q.stiffness,
            q.pz + (q.tz - q.pz) * q.stiffness,
        ]
    } else {
        [q.px, q.py, q.pz]
    };
    (new_p, 1)
}

/// Dispatches one query and asserts the device result matches the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuSoftPullToTarget, q: SoftPullToTargetQuery) {
    let results = gpu.evaluate(ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let ([nx, ny, nz], valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close(r.new_px, nx) && close(r.new_py, ny) && close(r.new_pz, nz),
        "position mismatch: query={q:?} gpu=({}, {}, {}) oracle=({nx}, {ny}, {nz})",
        r.new_px,
        r.new_py,
        r.new_pz
    );
}

#[test]
fn stiffness_one_snaps_to_target() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftPullToTarget::new(&ctx);
    // A free particle at full stiffness lands exactly on its target.
    assert_parity(
        &ctx,
        &gpu,
        SoftPullToTargetQuery::new(1.0, -2.0, 0.5, 1.0, 4.0, 3.0, -1.5, 1.0),
    );
}

#[test]
fn stiffness_half_moves_halfway() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftPullToTarget::new(&ctx);
    // A free particle at half stiffness moves to the midpoint of p and target.
    assert_parity(
        &ctx,
        &gpu,
        SoftPullToTargetQuery::new(0.0, 0.0, 0.0, 2.0, 2.0, -4.0, 6.0, 0.5),
    );
}

#[test]
fn pinned_particle_holds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftPullToTarget::new(&ctx);
    // Inverse mass 0 means pinned: the particle holds position regardless of the
    // positive stiffness and a distant target.
    assert_parity(
        &ctx,
        &gpu,
        SoftPullToTargetQuery::new(1.0, 2.0, 3.0, 0.0, -5.0, -5.0, -5.0, 1.0),
    );
}

#[test]
fn zero_stiffness_holds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftPullToTarget::new(&ctx);
    // Stiffness 0 is a no-op even for a free particle.
    assert_parity(
        &ctx,
        &gpu,
        SoftPullToTargetQuery::new(-1.0, 0.5, 2.0, 1.0, 7.0, 7.0, 7.0, 0.0),
    );
}

#[test]
fn negative_stiffness_holds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftPullToTarget::new(&ctx);
    // Negative stiffness is non-positive, so the projection is a no-op.
    assert_parity(
        &ctx,
        &gpu,
        SoftPullToTargetQuery::new(3.0, -1.0, 0.0, 1.5, 0.0, 0.0, 0.0, -1.0),
    );
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftPullToTarget::new(&ctx);
    // A multi-element batch exercises the std430 stride: adjacent slots must
    // decode independently and in order, spanning snap, halfway, pinned,
    // zero-stiffness and negative-stiffness cases.
    let queries = [
        SoftPullToTargetQuery::new(1.0, -2.0, 0.5, 1.0, 4.0, 3.0, -1.5, 1.0),
        SoftPullToTargetQuery::new(0.0, 0.0, 0.0, 2.0, 2.0, -4.0, 6.0, 0.5),
        SoftPullToTargetQuery::new(1.0, 2.0, 3.0, 0.0, -5.0, -5.0, -5.0, 1.0),
        SoftPullToTargetQuery::new(-1.0, 0.5, 2.0, 1.0, 7.0, 7.0, 7.0, 0.0),
        SoftPullToTargetQuery::new(3.0, -1.0, 0.0, 1.5, 0.0, 0.0, 0.0, -1.0),
        SoftPullToTargetQuery::new(0.25, 0.75, -0.5, 0.5, -0.25, 1.5, 2.0, 0.3),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let ([nx, ny, nz], valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close(r.new_px, nx) && close(r.new_py, ny) && close(r.new_pz, nz),
            "batch position mismatch: query={q:?} gpu=({}, {}, {}) oracle=({nx}, {ny}, {nz})",
            r.new_px,
            r.new_py,
            r.new_pz
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftPullToTarget::new(&ctx);
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
    let gpu = GpuSoftPullToTarget::new(&ctx);
    let mut rng = Lcg::new(0x50_1F_7A_93);
    // Margin kept well above any f32 noise so the move predicate never sits on
    // the w == 0 or stiffness == 0 knee; those boundaries are pinned by the
    // named fixtures instead.
    const KNEE_MARGIN: f32 = 0.01;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let w = rng.next_range(0.0, 2.0);
        let stiffness = rng.next_range(0.0, 1.0);
        // Reject near-zero inverse mass or stiffness so the branch is decisive.
        if w < KNEE_MARGIN || stiffness < KNEE_MARGIN {
            continue;
        }
        let px = rng.next_range(-2.0, 2.0);
        let py = rng.next_range(-2.0, 2.0);
        let pz = rng.next_range(-2.0, 2.0);
        let tx = rng.next_range(-2.0, 2.0);
        let ty = rng.next_range(-2.0, 2.0);
        let tz = rng.next_range(-2.0, 2.0);
        queries.push(SoftPullToTargetQuery::new(
            px, py, pz, w, tx, ty, tz, stiffness,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let ([nx, ny, nz], valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close(r.new_px, nx) && close(r.new_py, ny) && close(r.new_pz, nz),
            "sweep position mismatch: query={q:?} gpu=({}, {}, {}) oracle=({nx}, {ny}, {nz})",
            r.new_px,
            r.new_py,
            r.new_pz
        );
    }
}
