//! Real-device parity for the capillary-bridge force-at-gap twin:
//! [`GpuCapillaryForceAtGap`](prism_volumetric_gpu::capillary_force_at_gap::GpuCapillaryForceAtGap)
//! must reproduce the `CPU` golden `CapillaryBridgeModel::force_at_gap` of
//! `prism_physics_core::collider::capillary_bridge`, the Rabinovich et al.
//! (2005) pendular-bridge attractive-force magnitude between two wet grains.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly in flat `f32`/`f64` math so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. It replicates
//! the golden exactly, including the rupture cube root and the contact-angle
//! cosine taken in `f64` (`(V as f64).cbrt() as f32`,
//! `(theta as f64).cos() as f32`) and the golden `force_at_gap` validity gate:
//! `gap` finite, `gap <= rupture`, and both radii finite and strictly positive.
//! The golden `force_at_gap` does *not* re-gate surface tension, contact angle
//! or liquid volume, so neither does the oracle.
//!
//! The fixtures cover the regimes the kernel must honor: contact
//! (`gap <= 0`, clamped to the peak pull `F0`); a small positive gap; a gap
//! safely below rupture; a gap past rupture that must report `valid = 0`;
//! non-finite gaps (`INFINITY`, `NaN`); degenerate radii (non-positive or
//! non-finite); a multi-element mixed valid/invalid batch that validates the
//! `std430` array stride end to end; plus an empty batch the host
//! short-circuits with no dispatch. A `512`-step `LCG` sweep over valid
//! interior inputs follows, spanning both the contact and embracing regimes.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL` plus the `sqrt`, `cos` and `pow`
//! builtins the closed form requires, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The golden evaluates the rupture cube root and the cosine in `f64` while the
//! twin uses `f32` `pow(V, 1/3)` and `cos`, so `CPU` and `GPU` agree only up to
//! a small numerical gap. The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on `magnitude`;
//! the discrete `valid` flag is compared exactly. The sweep keeps the gap well
//! clear of the rupture knee (`|gap - rupture| >= 0.05 * rupture`) so round-off
//! cannot flip the validity decision, and keeps `V` strictly positive so the
//! guarded `pow` base never diverges from the golden `cbrt`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::capillary_bridge`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::capillary_force_at_gap::{
    CapillaryForceAtGapQuery, GpuCapillaryForceAtGap,
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

/// Independent host re-implementation of the golden
/// `CapillaryBridgeModel::force_at_gap`, returning the force magnitude and the
/// `valid` flag without importing the golden crate or `glam`. The rupture cube
/// root and the contact-angle cosine are taken in `f64` exactly as the golden
/// does, and the validity gate mirrors the golden `force_at_gap`: finite
/// `gap`, `gap <= rupture`, both radii finite and strictly positive.
fn oracle(q: &CapillaryForceAtGapQuery) -> (f32, u32) {
    // Non-finite gap is rejected first, matching golden `!gap.is_finite()`.
    if !(q.gap.abs() < 3.0e38) {
        return (0.0, 0);
    }
    let cube_root = (q.liquid_volume as f64).cbrt() as f32;
    let rupture = (1.0 + 0.5 * q.contact_angle) * cube_root;
    if q.gap > rupture {
        return (0.0, 0);
    }
    let radii_ok = q.radius_a.abs() < 3.0e38
        && q.radius_b.abs() < 3.0e38
        && q.radius_a > 0.0
        && q.radius_b > 0.0;
    if !radii_ok {
        return (0.0, 0);
    }
    let reduced = 2.0 * q.radius_a * q.radius_b / (q.radius_a + q.radius_b);
    let cos_theta = (q.contact_angle as f64).cos() as f32;
    let f0 = 2.0 * std::f32::consts::PI * reduced * q.surface_tension * cos_theta;
    let sep = q.gap.max(0.0);
    if sep <= 0.0 {
        return (f0, 1);
    }
    let inner = 1.0 + 2.0 * q.liquid_volume / (std::f32::consts::PI * reduced * sep * sep);
    let embracing = 0.5 * sep * (-1.0 + inner.sqrt());
    (f0 / (1.0 + sep / (2.0 * embracing)), 1)
}

/// Dispatches one query and asserts the `GPU` result matches the oracle on the
/// `magnitude` scalar (within tolerance) and the `valid` flag (exactly).
fn assert_parity(ctx: &GpuContext, gpu: &GpuCapillaryForceAtGap, q: CapillaryForceAtGapQuery) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(&q))[0];
    let (magnitude, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close(r.magnitude, magnitude),
        "magnitude mismatch: gpu={} cpu={} query={q:?}",
        r.magnitude,
        magnitude
    );
}

#[test]
fn contact_returns_peak_pull() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryForceAtGap::new(&ctx);
    // gap <= 0 (overlapping cores) clamps to contact and returns F0.
    // Water-like film: gamma = 0.072, perfectly wetting (theta = 0), V = 1e-9.
    for &gap in &[0.0_f32, -1.0e-4, -5.0e-4] {
        assert_parity(
            &ctx,
            &gpu,
            CapillaryForceAtGapQuery::new(0.5, 0.5, gap, 0.072, 0.0, 1.0e-9),
        );
    }
}

#[test]
fn small_positive_gap_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryForceAtGap::new(&ctx);
    // V = 1e-9 -> rupture = 1e-3; a small positive gap sits well inside.
    for &gap in &[1.0e-5_f32, 1.0e-4, 5.0e-4] {
        assert_parity(
            &ctx,
            &gpu,
            CapillaryForceAtGapQuery::new(1.0, 1.0, gap, 0.072, 0.3, 1.0e-9),
        );
    }
}

#[test]
fn gap_just_below_rupture_is_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryForceAtGap::new(&ctx);
    // theta = 0, V = 1e-9 -> rupture = 1e-3; 0.9e-3 is safely inside.
    let q = CapillaryForceAtGapQuery::new(0.75, 1.25, 0.9e-3, 0.05, 0.0, 1.0e-9);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1, "gap below rupture must be valid");
}

#[test]
fn gap_past_rupture_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryForceAtGap::new(&ctx);
    // theta = 0, V = 1e-9 -> rupture = 1e-3; gap = 2e-3 exceeds it.
    let q = CapillaryForceAtGapQuery::new(1.0, 1.0, 2.0e-3, 0.072, 0.0, 1.0e-9);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 0, "gap past rupture must be invalid: {q:?}");
    assert!(
        close(r.magnitude, 0.0),
        "invalid magnitude must be zero, got {}",
        r.magnitude
    );
    let (magnitude, valid) = oracle(&q);
    assert_eq!(valid, 0, "oracle must agree gap past rupture is invalid");
    assert_eq!(magnitude, 0.0, "oracle invalid magnitude must be zero");
}

#[test]
fn non_finite_gap_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryForceAtGap::new(&ctx);
    for &gap in &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let q = CapillaryForceAtGapQuery::new(1.0, 1.0, gap, 0.072, 0.2, 1.0e-9);
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(r.valid, 0, "non-finite gap must be invalid: {q:?}");
        assert!(
            close(r.magnitude, 0.0),
            "invalid magnitude must be zero, got {} for {q:?}",
            r.magnitude
        );
        let (_, valid) = oracle(&q);
        assert_eq!(
            valid, 0,
            "oracle must agree non-finite gap is invalid: {q:?}"
        );
    }
}

#[test]
fn degenerate_radius_reports_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryForceAtGap::new(&ctx);
    // Non-positive or non-finite radii collapse to valid = 0, magnitude = 0.
    let degenerate = [
        CapillaryForceAtGapQuery::new(0.0, 1.0, 1.0e-4, 0.072, 0.0, 1.0e-9),
        CapillaryForceAtGapQuery::new(1.0, -0.5, 1.0e-4, 0.072, 0.0, 1.0e-9),
        CapillaryForceAtGapQuery::new(f32::INFINITY, 1.0, 1.0e-4, 0.072, 0.0, 1.0e-9),
        CapillaryForceAtGapQuery::new(1.0, f32::NAN, 1.0e-4, 0.072, 0.0, 1.0e-9),
    ];
    for q in degenerate {
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(r.valid, 0, "degenerate radius must be invalid: {q:?}");
        assert!(
            close(r.magnitude, 0.0),
            "invalid magnitude must be zero, got {} for {q:?}",
            r.magnitude
        );
        let (magnitude, valid) = oracle(&q);
        assert_eq!(valid, 0, "oracle must agree radius is degenerate: {q:?}");
        assert_eq!(
            magnitude, 0.0,
            "oracle invalid magnitude must be zero: {q:?}"
        );
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryForceAtGap::new(&ctx);
    // A multi-element mixed batch (contact, embracing, past-rupture, degenerate
    // radius, non-finite gap) exercises the std430 array stride: every slot must
    // decode at the right byte offset and remain independent.
    let queries = [
        CapillaryForceAtGapQuery::new(0.5, 0.5, -1.0e-4, 0.072, 0.0, 1.0e-9),
        CapillaryForceAtGapQuery::new(1.0, 1.0, 2.0e-4, 0.05, 0.4, 1.0e-9),
        CapillaryForceAtGapQuery::new(1.0, 1.0, 5.0e-3, 0.072, 0.0, 1.0e-9),
        CapillaryForceAtGapQuery::new(-1.0, 1.0, 1.0e-4, 0.072, 0.0, 1.0e-9),
        CapillaryForceAtGapQuery::new(1.0, 1.0, f32::NAN, 0.072, 0.0, 1.0e-9),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (magnitude, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close(r.magnitude, magnitude),
            "batch magnitude mismatch: gpu={} cpu={} query={q:?}",
            r.magnitude,
            magnitude
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryForceAtGap::new(&ctx);
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
    let gpu = GpuCapillaryForceAtGap::new(&ctx);
    let mut rng = Lcg::new(0x9E_3B_7F_11);
    let mut queries = Vec::with_capacity(512);
    // Keep every drawn sample valid and well clear of the knees: physically
    // sensible radii and bridge parameters, V strictly positive (so the guarded
    // pow base never diverges from the golden cbrt), and a gap kept at least
    // 5% of rupture away from the rupture knee so the f64/f32 round-off cannot
    // flip the validity decision. Half the samples are contact (gap < 0), half
    // embracing (0 < gap <= 0.9 * rupture).
    let hi_angle = std::f32::consts::FRAC_PI_2 - 0.05;
    while queries.len() < 512 {
        let radius_a = rng.next_range(0.1, 2.0);
        let radius_b = rng.next_range(0.1, 2.0);
        let gamma = rng.next_range(0.01, 0.1);
        let theta = rng.next_range(0.0, hi_angle);
        let vol = rng.next_range(1.0e-9, 1.0e-3);
        // Rupture via the golden f64 cube root to place the gap window.
        let cube_root = (vol as f64).cbrt() as f32;
        let rupture = (1.0 + 0.5 * theta) * cube_root;
        let gap = if rng.next_unit() < 0.5 {
            // Contact regime: always valid, clamps to F0.
            rng.next_range(-0.5 * rupture, -0.02 * rupture)
        } else {
            // Embracing regime: strictly inside rupture with a safety margin.
            rng.next_range(0.02 * rupture, 0.9 * rupture)
        };
        queries.push(CapillaryForceAtGapQuery::new(
            radius_a, radius_b, gap, gamma, theta, vol,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (magnitude, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert_eq!(valid, 1, "sweep samples must be valid: query={q:?}");
        assert!(
            close(r.magnitude, magnitude),
            "sweep magnitude mismatch: gpu={} cpu={} query={q:?}",
            r.magnitude,
            magnitude
        );
    }
}
