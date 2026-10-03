//! Real-device parity for the curvature-mask twin:
//! [`GpuSdfCurvatureMasks`](prism_volumetric_gpu::mesh_sdf_curvature_masks::GpuSdfCurvatureMasks)
//! must reproduce the `CPU` golden closed form `curvature_masks_from_principals`
//! of `prism_render_architecture::ray_scene::mesh_sdf_curvature_masks`, which
//! turns a pair of principal curvatures (convex positive) and the ramp tuning
//! into a cavity mask, an edge-wear mask and a combined signed-curvature
//! channel.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the `smooth_ramp` cubic, the convex / concave split and the mean-curvature
//! normalization — written out directly so the test does not import
//! `prism_render_architecture`. It mirrors the reference branch for branch,
//! including the degenerate `hi <= lo` hard-step collapse.
//!
//! The fixtures cover the shapes the kernel must honor: a convex ridge whose
//! edge-wear saturates while its cavity stays zero, a concave crevice whose
//! cavity saturates, a flat patch whose masks and signed channel read zero, and
//! a degenerate ramp (`threshold = 1`, so `hi <= lo`) that must take the hard
//! step on both sides. A sweep over random curvatures and tuning follows, plus
//! an empty batch that the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each channel threads through multiplies, adds, one guarded division and a
//! cubic polynomial, so `CPU` and `GPU` evaluate the same closed form but need
//! not be bit-exact (a `GPU` may contract a multiply-add). The continuous
//! comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`).
//! The random sweep keeps `saturation` well above zero and `threshold` well
//! below one, so the ordered `hi <= lo` compare never sits on its equality knife
//! edge and both sides take the smooth branch; the degenerate hard-step branch
//! is checked separately by a named fixture.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_curvature_masks`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mesh_sdf_curvature_masks::{
    GpuSdfCurvatureMasks, SdfCurvatureMasksQuery,
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

/// Independent oracle re-implementing the golden `smooth_ramp`: a smoothstep of
/// `value` from `lo` (maps to zero) to `hi` (maps to one), clamped outside the
/// band, with inverted or zero-width bounds (`hi <= lo`) collapsing to a hard
/// step just above `hi`.
fn smooth_ramp(value: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return if value > hi { 1.0 } else { 0.0 };
    }
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The three channels the golden `curvature_masks_from_principals` returns:
/// `(cavity, edge_wear, signed_curvature)`.
fn oracle(
    principal_max: f32,
    principal_min: f32,
    saturation_curvature: f32,
    threshold: f32,
) -> (f32, f32, f32) {
    let saturation = saturation_curvature;
    let lo = threshold.clamp(0.0, 1.0) * saturation;
    let hi = saturation;
    let convex = principal_max.max(0.0);
    let concave = (-principal_min).max(0.0);
    let edge_wear = smooth_ramp(convex, lo, hi);
    let cavity = smooth_ramp(concave, lo, hi);
    let mean = 0.5 * (principal_max + principal_min);
    let signed_curvature = (mean / saturation.max(f32::MIN_POSITIVE)).clamp(-1.0, 1.0);
    (cavity, edge_wear, signed_curvature)
}

/// Dispatches one query and asserts the three channels against the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuSdfCurvatureMasks, q: SdfCurvatureMasksQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (cavity, edge_wear, signed) = oracle(
        q.principal_max,
        q.principal_min,
        q.saturation_curvature,
        q.threshold,
    );
    let r = got[0];
    assert!(
        close(r.cavity, cavity),
        "cavity mismatch: gpu={} cpu={} query={q:?}",
        r.cavity,
        cavity
    );
    assert!(
        close(r.edge_wear, edge_wear),
        "edge_wear mismatch: gpu={} cpu={} query={q:?}",
        r.edge_wear,
        edge_wear
    );
    assert!(
        close(r.signed_curvature, signed),
        "signed_curvature mismatch: gpu={} cpu={} query={q:?}",
        r.signed_curvature,
        signed
    );
}

#[test]
fn convex_ridge_edge_wears() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCurvatureMasks::new(&ctx);
    // A sharp convex ridge: large positive max, near-zero min. Edge wear should
    // saturate toward one while the cavity reads zero (nothing concave).
    assert_parity(&ctx, &gpu, SdfCurvatureMasksQuery::new(1.2, 0.0, 1.0, 0.1));
}

#[test]
fn concave_crevice_cavities() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCurvatureMasks::new(&ctx);
    // A sharp concave crevice: strongly negative min. Cavity saturates, edge
    // wear reads zero.
    assert_parity(&ctx, &gpu, SdfCurvatureMasksQuery::new(0.0, -1.3, 1.0, 0.1));
}

#[test]
fn flat_patch_reads_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCurvatureMasks::new(&ctx);
    // A flat patch: both principal curvatures zero. All channels read zero.
    assert_parity(&ctx, &gpu, SdfCurvatureMasksQuery::new(0.0, 0.0, 1.0, 0.1));
}

#[test]
fn midband_ramps_smoothly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCurvatureMasks::new(&ctx);
    // A convex curvature sitting partway up the ramp exercises the cubic itself
    // rather than its saturated ends.
    assert_parity(
        &ctx,
        &gpu,
        SdfCurvatureMasksQuery::new(0.55, -0.2, 1.0, 0.1),
    );
}

#[test]
fn degenerate_bounds_hard_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCurvatureMasks::new(&ctx);
    // threshold = 1 makes lo == hi == saturation, so the ramp collapses to a
    // hard step just above hi. Probe both sides of the step on each mask.
    assert_parity(&ctx, &gpu, SdfCurvatureMasksQuery::new(1.5, -1.5, 1.0, 1.0));
    assert_parity(&ctx, &gpu, SdfCurvatureMasksQuery::new(0.5, -0.5, 1.0, 1.0));
    // A non-positive saturation also inverts the bounds (hi <= lo).
    assert_parity(&ctx, &gpu, SdfCurvatureMasksQuery::new(0.8, -0.3, 0.0, 0.1));
}

#[test]
fn signed_curvature_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCurvatureMasks::new(&ctx);
    // Mean curvature well beyond the saturation on each side drives the signed
    // channel into its clamp.
    assert_parity(&ctx, &gpu, SdfCurvatureMasksQuery::new(4.0, 3.0, 1.0, 0.1));
    assert_parity(
        &ctx,
        &gpu,
        SdfCurvatureMasksQuery::new(-3.0, -4.0, 1.0, 0.1),
    );
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfCurvatureMasks::new(&ctx);
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
    let gpu = GpuSdfCurvatureMasks::new(&ctx);
    let mut rng = Lcg::new(0x5C_A7_1E_11);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // Keep saturation comfortably positive and threshold well under one so
        // the ordered `hi <= lo` compare stays off its equality knife edge and
        // both sides pick the smooth branch; the hard-step branch is checked by
        // its own named fixture above.
        let principal_max = rng.next_range(-2.0, 2.0);
        let principal_min = rng.next_range(-2.0, 2.0);
        let saturation_curvature = rng.next_range(0.3, 2.0);
        let threshold = rng.next_range(0.0, 0.8);
        queries.push(SdfCurvatureMasksQuery::new(
            principal_max,
            principal_min,
            saturation_curvature,
            threshold,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (cavity, edge_wear, signed) = oracle(
            q.principal_max,
            q.principal_min,
            q.saturation_curvature,
            q.threshold,
        );
        assert!(
            close(r.cavity, cavity),
            "sweep cavity mismatch: gpu={} cpu={} query={q:?}",
            r.cavity,
            cavity
        );
        assert!(
            close(r.edge_wear, edge_wear),
            "sweep edge_wear mismatch: gpu={} cpu={} query={q:?}",
            r.edge_wear,
            edge_wear
        );
        assert!(
            close(r.signed_curvature, signed),
            "sweep signed_curvature mismatch: gpu={} cpu={} query={q:?}",
            r.signed_curvature,
            signed
        );
    }
}
