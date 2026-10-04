//! Real-device parity for the Savage-number twin:
//! [`GpuSavageNumber`](prism_volumetric_gpu::savage_number::GpuSavageNumber)
//! must reproduce the `CPU` golden `SavageNumber::from_state` plus `regime` of
//! `prism_physics_core::collider::savage_number`. The Savage number of a
//! granular state is `N_sav = ρ_s · d² · γ̇² / P` when all inputs are finite
//! and `ρ_s > 0`, `d > 0`, `γ̇ >= 0`, `P > 0` (and the result is finite), and
//! undefined (invalid) otherwise. The regime is `FrictionalQuasiStatic`
//! (code `0`) below the critical value `0.1`, else `Collisional` (code `1`).
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness and range guards, then `ρ_s · d · d · γ̇ · γ̇ / P` in the
//! golden operator order, then the half-open regime split — written out
//! directly so the test never imports `prism_render_architecture`,
//! `prism_physics_core` or `glam`.
//!
//! The fixtures cover a hand-computed frictional value, a hand-computed
//! collisional value, zero shear (`N_sav = 0`, frictional), a batch of
//! invalid states (out-of-range and non-finite), a mixed batch of two or more
//! elements validating the `std430` stride, and an empty batch the host
//! short-circuits with no dispatch. A sweep over random states covering both
//! regimes follows, reject-sampling the `N_sav ≈ 0.1` regime knee so no
//! classification can flip.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through multiplies and a division, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact
//! (a `GPU` may contract a multiply-add). The valid `savage_number` scalar is
//! compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the
//! discrete `regime` and `valid` flags are compared exactly. The fixtures keep
//! the sweep away from the regime knee so the classification agrees on both
//! sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::savage_number`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::savage_number::{GpuSavageNumber, SavageNumberQuery, SavageNumberResult};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Savage-Hutter critical value separating the two regimes.
const SAVAGE_CRITICAL: f32 = 0.1;

/// Independent host oracle: reproduces `from_state` plus `regime` in the golden
/// operator order, returning the Savage number, the regime code and the
/// validity flag.
fn oracle(q: &SavageNumberQuery) -> (f32, u32, u32) {
    let rho = q.solid_density;
    let d = q.grain_diameter;
    let gamma = q.shear_rate;
    let p = q.normal_stress;
    if !rho.is_finite() || !d.is_finite() || !gamma.is_finite() || !p.is_finite() {
        return (0.0, 0, 0);
    }
    if rho <= 0.0 || d <= 0.0 || gamma < 0.0 || p <= 0.0 {
        return (0.0, 0, 0);
    }
    let n_sav = rho * d * d * gamma * gamma / p;
    if !n_sav.is_finite() {
        return (0.0, 0, 0);
    }
    let regime = if n_sav < SAVAGE_CRITICAL { 0u32 } else { 1u32 };
    (n_sav, regime, 1)
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` flag exactly, and when valid the `regime` exactly and the
/// `savage_number` scalar to tolerance.
fn assert_parity(gpu: &SavageNumberResult, q: &SavageNumberQuery, label: &str) {
    let (n_sav, regime, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert_eq!(gpu.regime, regime, "{label}: regime mismatch");
        assert!(
            close(gpu.savage_number, n_sav),
            "{label}: savage_number mismatch gpu={} oracle={}",
            gpu.savage_number,
            n_sav
        );
    }
}

#[test]
fn frictional_hand_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSavageNumber::new(&ctx);
    // rho_s=2500, d=0.002, gamma=10, P=1000 -> N_sav = 1e-3, frictional.
    let q = SavageNumberQuery::new(2500.0, 0.002, 10.0, 1000.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].regime, 0);
    assert!(close(out[0].savage_number, 1.0e-3));
    assert_parity(&out[0], &q, "frictional");
}

#[test]
fn collisional_hand_value() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSavageNumber::new(&ctx);
    // rho_s=2500, d=0.01, gamma=50, P=500 -> N_sav = 1.25, collisional.
    let q = SavageNumberQuery::new(2500.0, 0.01, 50.0, 500.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].regime, 1);
    assert!(close(out[0].savage_number, 1.25));
    assert_parity(&out[0], &q, "collisional");
}

#[test]
fn zero_shear_is_frictional() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSavageNumber::new(&ctx);
    // gamma = 0 -> N_sav = 0, frictional, still valid.
    let q = SavageNumberQuery::new(2000.0, 0.005, 0.0, 1500.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].regime, 0);
    assert!(close(out[0].savage_number, 0.0));
    assert_parity(&out[0], &q, "zero_shear");
}

#[test]
fn invalid_states_are_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSavageNumber::new(&ctx);
    let queries = vec![
        SavageNumberQuery::new(0.0, 0.002, 10.0, 1000.0),
        SavageNumberQuery::new(2500.0, 0.0, 10.0, 1000.0),
        SavageNumberQuery::new(2500.0, 0.002, -1.0, 1000.0),
        SavageNumberQuery::new(2500.0, 0.002, 10.0, 0.0),
        SavageNumberQuery::new(f32::NAN, 0.002, 10.0, 1000.0),
        SavageNumberQuery::new(2500.0, 0.002, 10.0, f32::INFINITY),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "invalid[{i}] should be rejected");
        assert_eq!(res.savage_number, 0.0, "invalid[{i}] savage_number zeroed");
        assert_eq!(res.regime, 0, "invalid[{i}] regime zeroed");
        assert_parity(res, q, &format!("invalid[{i}]"));
    }
}

#[test]
fn batch_mixes_regimes_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSavageNumber::new(&ctx);
    let queries = vec![
        SavageNumberQuery::new(2500.0, 0.002, 10.0, 1000.0), // frictional, valid
        SavageNumberQuery::new(2500.0, 0.01, 50.0, 500.0),   // collisional, valid
        SavageNumberQuery::new(2500.0, 0.002, -1.0, 1000.0), // invalid
        SavageNumberQuery::new(2000.0, 0.005, 0.0, 1500.0),  // zero shear, valid
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[0].regime, 0);
    assert_eq!(out[1].valid, 1);
    assert_eq!(out[1].regime, 1);
    assert_eq!(out[2].valid, 0);
    assert_eq!(out[3].valid, 1);
    assert_eq!(out[3].regime, 0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSavageNumber::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSavageNumber::new(&ctx);
    let mut lcg = Lcg::new(0x5A17_9E42);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Strictly-positive, well-conditioned granular states.
        let rho = lcg.next_range(500.0, 3000.0);
        let d = lcg.next_range(0.001, 0.02);
        let gamma = lcg.next_range(0.0, 80.0);
        let p = lcg.next_range(200.0, 3000.0);
        let q = SavageNumberQuery::new(rho, d, gamma, p);
        // Reject-sample the regime knee so round-off cannot flip the class.
        let (n_sav, _, valid) = oracle(&q);
        if valid == 1 && (n_sav - SAVAGE_CRITICAL).abs() < 0.02 {
            continue;
        }
        queries.push(q);
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A small deterministic linear-congruential generator; the fixture carries no
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
