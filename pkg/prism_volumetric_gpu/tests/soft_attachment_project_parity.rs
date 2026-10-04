//! Real-device parity for the single-particle XPBD attachment-projection twin:
//! [`GpuSoftAttachmentProject`](prism_volumetric_gpu::soft_attachment_project::GpuSoftAttachmentProject)
//! must reproduce the `CPU` golden `AttachmentConstraint::project` of
//! `prism_physics_core::soft::constraint::attachment`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the pinned guard `w <= 0`, the already-at-target guard `|p - target| <
//! EPSILON`, the unit gradient `n = delta / length`, the compliant denominator
//! `alpha_tilde = compliance / (dt * dt)`, the Lagrange delta
//! `delta_lambda = (-length - alpha_tilde * lambda) / (w + alpha_tilde)`, and
//! the position/multiplier updates — written out directly so the test never
//! imports `prism_physics_core` or `prism_render_architecture`.
//!
//! The fixtures cover a rigid snap toward the target (`compliance = 0`), a soft
//! compliant pull (`compliance > 0`), a pinned particle (`w = 0`, inert), a
//! particle already on the target (`|delta| < EPSILON`, inert), a non-axis
//! aligned target, and a mixed batch that validates the `std430` stride end to
//! end. A `512`-query `LCG` sweep follows, keeping samples clear of the
//! `length = EPSILON` and `w = 0` knees, and an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The position (`new_px`, `new_py`, `new_pz`) and the multiplier
//! (`new_lambda`) are continuous and checked with an absolute-or-relative
//! tolerance (`abs <= 1e-4 || rel <= 1e-3`, `REL_FLOOR = 1e-6`); `valid` is
//! discrete and compared exactly. The random sweep keeps samples away from the
//! `length = EPSILON` and `w = 0` knees so a host/device ordering difference on
//! the guards cannot flip the discrete `valid` channel; dedicated named
//! fixtures pin those degenerate cases.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::attachment`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_attachment_project::{
    GpuSoftAttachmentProject, SoftAttachmentProjectQuery, SoftAttachmentProjectResult,
};
use prism_volumetric_gpu::GpuContext;

/// `f32::EPSILON`, the already-at-target cutoff the golden uses.
const EPSILON: f32 = 1.192_092_9e-7;

/// Independent host re-implementation of `AttachmentConstraint::project`,
/// flattened into the `(new_p, new_lambda, valid)` record the twin encodes. The
/// constraint value is the raw distance to the target (zero rest length), so
/// the numerator is `-length`. No `prism_physics_core` import.
fn oracle(q: &SoftAttachmentProjectQuery) -> SoftAttachmentProjectResult {
    let dx = q.px - q.tx;
    let dy = q.py - q.ty;
    let dz = q.pz - q.tz;
    let length = (dx * dx + dy * dy + dz * dz).sqrt();

    // Ordered guards: a NaN input fails these and routes to the inert branch.
    let active = q.w > 0.0 && length >= EPSILON;
    if !active {
        return SoftAttachmentProjectResult {
            new_px: q.px,
            new_py: q.py,
            new_pz: q.pz,
            new_lambda: q.lambda,
            valid: 0,
        };
    }

    // Match the device: normal = delta / length (length >= EPSILON here).
    let nx = dx / length;
    let ny = dy / length;
    let nz = dz / length;

    let alpha_tilde = q.compliance / (q.dt * q.dt);
    let delta_lambda = (-length - alpha_tilde * q.lambda) / (q.w + alpha_tilde);
    let new_lambda = q.lambda + delta_lambda;
    let scale = delta_lambda * q.w;

    SoftAttachmentProjectResult {
        new_px: q.px + nx * scale,
        new_py: q.py + ny * scale,
        new_pz: q.pz + nz * scale,
        new_lambda,
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

/// Asserts one GPU result matches the oracle. Position and multiplier are
/// continuous (tolerance); `valid` is discrete (exact).
fn assert_result(got: SoftAttachmentProjectResult, want: SoftAttachmentProjectResult, label: &str) {
    assert_eq!(got.valid, want.valid, "valid mismatch: {label}");
    assert!(
        close(got.new_px, want.new_px),
        "new_px mismatch: {label}: got {} want {}",
        got.new_px,
        want.new_px
    );
    assert!(
        close(got.new_py, want.new_py),
        "new_py mismatch: {label}: got {} want {}",
        got.new_py,
        want.new_py
    );
    assert!(
        close(got.new_pz, want.new_pz),
        "new_pz mismatch: {label}: got {} want {}",
        got.new_pz,
        want.new_pz
    );
    assert!(
        close(got.new_lambda, want.new_lambda),
        "new_lambda mismatch: {label}: got {} want {}",
        got.new_lambda,
        want.new_lambda
    );
}

/// Asserts a single-query GPU result matches the oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuSoftAttachmentProject,
    q: SoftAttachmentProjectQuery,
    label: &str,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query: {label}");
    assert_result(got[0], oracle(&q), label);
}

#[test]
fn rigid_snap_pulls_toward_target() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftAttachmentProject::new(&ctx);
    // Rigid pin (compliance 0): a unit-mass particle three units out along x is
    // snapped straight onto the origin target in one step.
    let q =
        SoftAttachmentProjectQuery::new(3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 60.0);
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: projection runs");
    assert!(
        want.new_px.abs() < 1e-4,
        "fixture sanity: snapped onto target, got {}",
        want.new_px
    );
    assert_parity(&ctx, &gpu, q, "rigid_snap_pulls_toward_target");
}

#[test]
fn soft_compliance_partial_pull() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftAttachmentProject::new(&ctx);
    // A positive compliance softens the pull: the particle moves toward the
    // target but does not snap all the way in one step.
    let q =
        SoftAttachmentProjectQuery::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-3, 0.0, 1.0 / 60.0);
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: projection runs");
    assert!(
        want.new_px > 0.0 && want.new_px < 2.0,
        "fixture sanity: partial pull, got {}",
        want.new_px
    );
    assert_parity(&ctx, &gpu, q, "soft_compliance_partial_pull");
}

#[test]
fn pinned_particle_is_inert() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftAttachmentProject::new(&ctx);
    // Zero inverse mass marks a pinned particle: the step is inert, position and
    // multiplier are returned unchanged and valid is 0.
    let q =
        SoftAttachmentProjectQuery::new(3.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0 / 60.0);
    let want = oracle(&q);
    assert_eq!(want.valid, 0, "fixture sanity: pinned is inert");
    assert_eq!(want.new_px, 3.0, "fixture sanity: position unchanged");
    assert_eq!(want.new_lambda, 0.5, "fixture sanity: multiplier unchanged");
    assert_parity(&ctx, &gpu, q, "pinned_particle_is_inert");
}

#[test]
fn already_at_target_is_inert() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftAttachmentProject::new(&ctx);
    // A particle already on the target has no defined gradient direction, so the
    // step is inert and valid is 0.
    let q =
        SoftAttachmentProjectQuery::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.25, 1.0 / 60.0);
    let want = oracle(&q);
    assert_eq!(want.valid, 0, "fixture sanity: at-target is inert");
    assert_eq!(want.new_px, 1.0, "fixture sanity: position unchanged");
    assert_eq!(
        want.new_lambda, 0.25,
        "fixture sanity: multiplier unchanged"
    );
    assert_parity(&ctx, &gpu, q, "already_at_target_is_inert");
}

#[test]
fn non_axis_target_projects_along_gradient() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftAttachmentProject::new(&ctx);
    // A non-axis-aligned delta exercises all three gradient components; a small
    // compliance and nonzero incoming lambda keep every term live.
    let q = SoftAttachmentProjectQuery::new(
        1.5,
        -2.0,
        0.75,
        0.5,
        -0.5,
        1.0,
        -0.25,
        5e-4,
        0.1,
        1.0 / 120.0,
    );
    let want = oracle(&q);
    assert_eq!(want.valid, 1, "fixture sanity: projection runs");
    assert_parity(&ctx, &gpu, q, "non_axis_target_projects_along_gradient");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftAttachmentProject::new(&ctx);
    // A >=2-element batch mixing every branch validates the std430 stride end to
    // end: rigid snap, soft pull, pinned, already-at-target and a non-axis case.
    let queries = vec![
        SoftAttachmentProjectQuery::new(3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 60.0),
        SoftAttachmentProjectQuery::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-3, 0.0, 1.0 / 60.0),
        SoftAttachmentProjectQuery::new(3.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0 / 60.0),
        SoftAttachmentProjectQuery::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.25, 1.0 / 60.0),
        SoftAttachmentProjectQuery::new(
            1.5,
            -2.0,
            0.75,
            0.5,
            -0.5,
            1.0,
            -0.25,
            5e-4,
            0.1,
            1.0 / 120.0,
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
    let gpu = GpuSoftAttachmentProject::new(&ctx);
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
    let gpu = GpuSoftAttachmentProject::new(&ctx);
    let mut rng = Lcg::new(0x1F_6B_42_C7);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let px = rng.next_range(-4.0, 4.0);
        let py = rng.next_range(-4.0, 4.0);
        let pz = rng.next_range(-4.0, 4.0);
        let tx = rng.next_range(-4.0, 4.0);
        let ty = rng.next_range(-4.0, 4.0);
        let tz = rng.next_range(-4.0, 4.0);

        let dx = px - tx;
        let dy = py - ty;
        let dz = pz - tz;
        let length = (dx * dx + dy * dy + dz * dz).sqrt();
        // Keep the main body well clear of the already-at-target knee so the
        // discrete valid channel cannot flip on a rounding tie.
        if length < 1e-2 {
            continue;
        }

        // Timestep strictly > 0 in [1/240, 1/30]; compliance >= 0; the body uses
        // a positive inverse mass well away from the w = 0 pin knee.
        let dt = rng.next_range(1.0 / 240.0, 1.0 / 30.0);
        let compliance = rng.next_range(0.0, 2e-3);
        let w = rng.next_range(0.25, 4.0);
        let lambda = rng.next_range(-1.0, 1.0);

        queries.push(SoftAttachmentProjectQuery::new(
            px, py, pz, w, tx, ty, tz, compliance, lambda, dt,
        ));
    }

    // Append a handful of degenerate cases: pinned particles and
    // already-at-target particles, both of which must report valid = 0.
    queries.push(SoftAttachmentProjectQuery::new(
        2.0,
        -1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        1e-3,
        0.3,
        1.0 / 60.0,
    ));
    queries.push(SoftAttachmentProjectQuery::new(
        -1.0,
        2.0,
        3.0,
        1.0,
        -1.0,
        2.0,
        3.0,
        0.0,
        -0.2,
        1.0 / 90.0,
    ));

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (i, (q, r)) in queries.iter().zip(results.iter()).enumerate() {
        assert_result(*r, oracle(q), &format!("sweep index {i}"));
    }
}
