//! Real-device parity for the cohesive-zone secant-damage twin:
//! [`GpuCohesiveZoneDamage`](prism_volumetric_gpu::cohesive_zone_damage::GpuCohesiveZoneDamage)
//! must reproduce the `CPU` golden `CohesiveModel::damage_at` of
//! `prism_physics_core::collider::cohesive_zone`. The secant damage of a
//! bilinear (Alfano–Crisfield) cohesive law rises from `0` at the onset
//! separation `δ₀` to `1` at the final separation `δ_f` with the largest
//! effective separation ever reached, `κ`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the two ordered branch thresholds then `δ_f·(κ − δ₀) / (κ·(δ_f − δ₀))`
//! clamped to `[0, 1]` in the golden operator order — written out directly so
//! the test never imports `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover `κ` below the onset (damage `0`), above the final
//! separation (damage `1`), several interior points along the monotone
//! softening envelope, a batch of two or more elements that validates the
//! `std430` stride, a mixed batch across all three regimes, and an empty batch
//! the host short-circuits with no dispatch. A sweep over random well-formed
//! models and histories follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The softening branch threads through a subtract, two multiplies, a division
//! and a clamp, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). The `damage` scalar is
//! compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`). The sweep
//! rejects any random `κ` within `0.05·δ_f` of either knee (`δ₀`, `δ_f`) so the
//! branch decision cannot be flipped by round-off on either side.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::cohesive_zone`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cohesive_zone_damage::{
    CohesiveZoneDamageQuery, CohesiveZoneDamageResult, GpuCohesiveZoneDamage,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `damage_at` in the golden operator
/// order. The model guarantees `δ_f > δ₀ > 0`, so the middle divisor is
/// positive whenever that branch is taken.
fn oracle(q: &CohesiveZoneDamageQuery) -> f32 {
    let onset = q.onset_separation;
    let final_sep = q.final_separation;
    let kappa = q.kappa;
    if kappa <= onset {
        return 0.0;
    }
    if kappa >= final_sep {
        return 1.0;
    }
    let num = final_sep * (kappa - onset);
    let den = kappa * (final_sep - onset);
    (num / den).clamp(0.0, 1.0)
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle to tolerance.
fn assert_parity(gpu: &CohesiveZoneDamageResult, q: &CohesiveZoneDamageQuery, label: &str) {
    let want = oracle(q);
    assert!(
        close(gpu.damage, want),
        "{label}: damage mismatch gpu={} oracle={} (onset={} final={} kappa={})",
        gpu.damage,
        want,
        q.onset_separation,
        q.final_separation,
        q.kappa
    );
}

#[test]
fn kappa_below_onset_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveZoneDamage::new(&ctx);
    // kappa strictly below delta0: still in the reversible elastic rise.
    let q = CohesiveZoneDamageQuery::new(0.5, 2.0, 0.2);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].damage, 0.0));
    assert_parity(&out[0], &q, "below_onset");
}

#[test]
fn kappa_at_onset_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveZoneDamage::new(&ctx);
    // kappa == delta0 exactly: the ordered <= branch yields d = 0.
    let q = CohesiveZoneDamageQuery::new(0.5, 2.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].damage, 0.0));
    assert_parity(&out[0], &q, "at_onset");
}

#[test]
fn kappa_above_final_is_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveZoneDamage::new(&ctx);
    // kappa beyond deltaf: fully decohered.
    let q = CohesiveZoneDamageQuery::new(0.5, 2.0, 5.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].damage, 1.0));
    assert_parity(&out[0], &q, "above_final");
}

#[test]
fn kappa_at_final_is_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveZoneDamage::new(&ctx);
    // kappa == deltaf exactly: the ordered >= branch yields d = 1.
    let q = CohesiveZoneDamageQuery::new(0.5, 2.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].damage, 1.0));
    assert_parity(&out[0], &q, "at_final");
}

#[test]
fn interior_points_are_monotone() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveZoneDamage::new(&ctx);
    // Several interior histories under one model; damage must rise with kappa.
    let onset = 0.5_f32;
    let final_sep = 2.0_f32;
    let queries = vec![
        CohesiveZoneDamageQuery::new(onset, final_sep, 0.75),
        CohesiveZoneDamageQuery::new(onset, final_sep, 1.0),
        CohesiveZoneDamageQuery::new(onset, final_sep, 1.25),
        CohesiveZoneDamageQuery::new(onset, final_sep, 1.5),
        CohesiveZoneDamageQuery::new(onset, final_sep, 1.75),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("interior[{i}]"));
    }
    for pair in out.windows(2) {
        assert!(
            pair[1].damage >= pair[0].damage - 1.0e-4,
            "damage should be monotone non-decreasing: {} then {}",
            pair[0].damage,
            pair[1].damage
        );
    }
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveZoneDamage::new(&ctx);
    // Two distinct valid queries validate the std430 stride end to end.
    let queries = vec![
        CohesiveZoneDamageQuery::new(0.3, 1.2, 0.6),
        CohesiveZoneDamageQuery::new(1.0, 4.0, 2.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 2);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("stride[{i}]"));
    }
}

#[test]
fn mixed_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveZoneDamage::new(&ctx);
    // A batch spanning below / middle / above regimes.
    let queries = vec![
        CohesiveZoneDamageQuery::new(0.5, 2.0, 0.1),
        CohesiveZoneDamageQuery::new(0.5, 2.0, 1.0),
        CohesiveZoneDamageQuery::new(0.5, 2.0, 3.0),
        CohesiveZoneDamageQuery::new(0.2, 0.9, 0.55),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(close(out[0].damage, 0.0));
    assert!(close(out[2].damage, 1.0));
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveZoneDamage::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveZoneDamage::new(&ctx);
    let mut lcg = Lcg::new(0x0C0E_51BE);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Well-formed model: delta_f > delta0 > 0.
        let onset = lcg.next_range(0.1, 1.0);
        let final_sep = onset + lcg.next_range(0.5, 3.0);
        // History spanning below / middle / above the band.
        let kappa = lcg.next_range(0.0, final_sep + 1.0);
        // Reject kappa within 0.05*delta_f of either knee so the branch
        // decision cannot flip under round-off.
        let margin = 0.05 * final_sep;
        if (kappa - onset).abs() < margin || (kappa - final_sep).abs() < margin {
            continue;
        }
        queries.push(CohesiveZoneDamageQuery::new(onset, final_sep, kappa));
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
