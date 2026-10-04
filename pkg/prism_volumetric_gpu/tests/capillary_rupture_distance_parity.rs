//! Real-device parity for the capillary-bridge rupture-distance twin:
//! [`GpuCapillaryRuptureDistance`](prism_volumetric_gpu::capillary_rupture_distance::GpuCapillaryRuptureDistance)
//! must reproduce the `CPU` golden `CapillaryBridgeModel::rupture_distance` of
//! `prism_physics_core::collider::capillary_bridge`, the wet-`DEM` closed form
//! `H_rupture = (1 + 0.5 * theta) * V^(1/3)` for a liquid bridge with contact
//! angle `theta` and liquid volume `V`.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly in flat `f32`/`f64` math so the test never imports
//! `prism_physics_core`, `prism_render_architecture` or `glam`. It replicates
//! the golden exactly, including the cube root taken in `f64`
//! (`(V as f64).cbrt() as f32`) and the admissible gate from the golden
//! `CapillaryBridgeModel::new`: both inputs finite, `theta` in `[0, pi/2]` and
//! `V` strictly positive.
//!
//! The fixtures cover the regimes the kernel must honor: the angle endpoints
//! `theta = 0` and `theta = pi/2`; several volume magnitudes (`1e-9`, `1e-3`,
//! `1.0`); degenerate inputs (negative angle, over-range angle, non-positive
//! volume, non-finite angle or volume) that must report `valid = 0` with
//! `rupture = 0`; a multi-element mixed batch that validates the `std430` array
//! stride end to end; plus an empty batch the host short-circuits with no
//! dispatch. A `512`-step `LCG` sweep over valid interior inputs follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The golden evaluates the cube root in `f64` while the twin uses `f32`
//! `pow(V, 1/3)`, so `CPU` and `GPU` agree only up to a small numerical gap.
//! The continuous comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`) on `rupture`; the discrete `valid` flag is compared
//! exactly. The sweep keeps the validity decision well clear of its knees
//! (`theta` strictly inside `[0, pi/2]`, `V` well above zero) so round-off
//! cannot flip the gate.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::capillary_bridge`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::capillary_rupture_distance::{
    CapillaryRuptureDistanceQuery, GpuCapillaryRuptureDistance,
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
/// `CapillaryBridgeModel::rupture_distance`, returning the rupture distance and
/// the `valid` flag without importing the golden crate or `glam`. The cube root
/// is taken in `f64` exactly as the golden does, and the admissible gate
/// mirrors the golden `new`: finite inputs, `theta` in `[0, pi/2]` and `V > 0`.
fn oracle(q: &CapillaryRuptureDistanceQuery) -> (f32, u32) {
    let theta = q.contact_angle;
    let vol = q.liquid_volume;
    let finite = theta.abs() < 3.0e38 && vol.abs() < 3.0e38;
    let angle_ok = theta >= 0.0 && theta <= std::f32::consts::FRAC_PI_2;
    let positive = vol > 0.0;
    if !(finite && angle_ok && positive) {
        return (0.0, 0);
    }
    let cube_root = (vol as f64).cbrt() as f32;
    let rupture = (1.0 + 0.5 * theta) * cube_root;
    (rupture, 1)
}

/// Dispatches one query and asserts the `GPU` result matches the oracle on the
/// `rupture` scalar (within tolerance) and the `valid` flag (exactly).
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuCapillaryRuptureDistance,
    q: CapillaryRuptureDistanceQuery,
) {
    let r = gpu.evaluate(ctx, std::slice::from_ref(&q))[0];
    let (rupture, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close(r.rupture, rupture),
        "rupture mismatch: gpu={} cpu={} query={q:?}",
        r.rupture,
        rupture
    );
}

#[test]
fn zero_angle_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryRuptureDistance::new(&ctx);
    // theta = 0: rupture reduces to the pure cube root V^(1/3).
    for &v in &[1.0e-9_f32, 1.0e-3, 1.0, 8.0, 1000.0] {
        assert_parity(&ctx, &gpu, CapillaryRuptureDistanceQuery::new(0.0, v));
    }
}

#[test]
fn half_pi_angle_is_valid_endpoint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryRuptureDistance::new(&ctx);
    // theta = pi/2 is the inclusive upper endpoint of the admissible range and
    // must report valid = 1.
    let q = CapillaryRuptureDistanceQuery::new(std::f32::consts::FRAC_PI_2, 2.0);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1, "theta = pi/2 must be a valid endpoint");
}

#[test]
fn volume_magnitudes_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryRuptureDistance::new(&ctx);
    // A mid-range contact angle across several volume magnitudes exercises the
    // f64-cbrt vs f32-pow gap at small and large V.
    let theta = 0.6_f32;
    for &v in &[1.0e-9_f32, 1.0e-6, 1.0e-3, 1.0, 1000.0] {
        assert_parity(&ctx, &gpu, CapillaryRuptureDistanceQuery::new(theta, v));
    }
}

#[test]
fn degenerate_inputs_report_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryRuptureDistance::new(&ctx);
    // Negative angle, over-range angle, zero and negative volume, and non-finite
    // inputs must all collapse to valid = 0 with rupture = 0.
    let degenerate = [
        CapillaryRuptureDistanceQuery::new(-0.1, 1.0),
        CapillaryRuptureDistanceQuery::new(std::f32::consts::FRAC_PI_2 + 0.1, 1.0),
        CapillaryRuptureDistanceQuery::new(0.5, 0.0),
        CapillaryRuptureDistanceQuery::new(0.5, -1.0),
        CapillaryRuptureDistanceQuery::new(f32::INFINITY, 1.0),
        CapillaryRuptureDistanceQuery::new(f32::NAN, 1.0),
        CapillaryRuptureDistanceQuery::new(0.5, f32::INFINITY),
        CapillaryRuptureDistanceQuery::new(0.5, f32::NAN),
    ];
    for q in degenerate {
        let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
        assert_eq!(r.valid, 0, "degenerate query must be invalid: {q:?}");
        assert!(
            close(r.rupture, 0.0),
            "invalid rupture must be zero, got {} for {q:?}",
            r.rupture
        );
        // The oracle must agree the input is degenerate.
        let (rupture, valid) = oracle(&q);
        assert_eq!(valid, 0, "oracle must agree query is invalid: {q:?}");
        assert_eq!(rupture, 0.0, "oracle invalid rupture must be zero: {q:?}");
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryRuptureDistance::new(&ctx);
    // A multi-element mixed batch (valid small/large volume, endpoint angle, and
    // two degenerate entries) exercises the std430 array stride: every slot must
    // decode at the right byte offset and remain independent.
    let queries = [
        CapillaryRuptureDistanceQuery::new(0.0, 1.0e-3),
        CapillaryRuptureDistanceQuery::new(std::f32::consts::FRAC_PI_2, 1000.0),
        CapillaryRuptureDistanceQuery::new(0.75, 2.5),
        CapillaryRuptureDistanceQuery::new(-0.2, 1.0),
        CapillaryRuptureDistanceQuery::new(0.4, -3.0),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (rupture, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close(r.rupture, rupture),
            "batch rupture mismatch: gpu={} cpu={} query={q:?}",
            r.rupture,
            rupture
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCapillaryRuptureDistance::new(&ctx);
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
    let gpu = GpuCapillaryRuptureDistance::new(&ctx);
    let mut rng = Lcg::new(0x7E_41_C3_09);
    let mut queries = Vec::with_capacity(512);
    // Keep every drawn sample well inside the admissible gate so round-off
    // cannot flip the validity decision: theta strictly inside [0, pi/2] with a
    // margin, V comfortably above zero.
    let hi_angle = std::f32::consts::FRAC_PI_2 - 0.01;
    while queries.len() < 512 {
        let theta = rng.next_range(0.01, hi_angle);
        let vol = rng.next_range(1.0e-6, 10.0);
        queries.push(CapillaryRuptureDistanceQuery::new(theta, vol));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (rupture, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert_eq!(valid, 1, "sweep samples must be valid: query={q:?}");
        assert!(
            close(r.rupture, rupture),
            "sweep rupture mismatch: gpu={} cpu={} query={q:?}",
            r.rupture,
            rupture
        );
    }
}
