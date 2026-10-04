//! Real-device parity for the cohesive dissipated-energy twin:
//! [`GpuCohesiveDissipatedEnergy`](prism_volumetric_gpu::cohesive_dissipated_energy::GpuCohesiveDissipatedEnergy)
//! must reproduce the `CPU` golden `CohesiveModel::new` derivation plus
//! `dissipated_energy` of `prism_physics_core::collider::cohesive_zone`. The
//! model derives `onset = strength / stiffness` and
//! `final = 2 * fracture_energy / strength`, then the bilinear dissipation is
//! `0` for `kappa <= onset`, the full fracture energy for `kappa >= final`, and
//! `0.5 * strength * final * (kappa - onset) / (final - onset)` in between.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the derived onset/final then the three-branch dissipation in the golden
//! operator order — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover a mid-segment softening value, the below-onset zero
//! branch, the above-final fracture-energy branch, both knee boundaries, a
//! `NaN` input, an infinite input, a non-positive stiffness/strength/fracture
//! energy, a parameter set whose derived `final <= onset` (no softening branch),
//! a batch of two or more elements mixing valid and invalid queries to validate
//! the `std430` stride, and an empty batch the host short-circuits with no
//! dispatch. A sweep over random finite inputs follows; it reject-samples both
//! the parameters (so `final` stays clearly above `onset`) and `kappa` (so it
//! stays at least `0.05 * (final - onset)` away from either knee), keeping the
//! branch decision stable between host and device.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The softening arithmetic threads through two derived divisions, a multiply
//! chain and a division, so `CPU` and `GPU` evaluate the same closed form but
//! need not be bit-exact (a `GPU` may contract a multiply-add). The valid
//! `energy` scalar is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::cohesive_zone`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cohesive_dissipated_energy::{
    CohesiveDissipatedEnergyQuery, CohesiveDissipatedEnergyResult, GpuCohesiveDissipatedEnergy,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `CohesiveModel::new`'s onset/final
/// derivation plus `dissipated_energy` in the golden operator order, returning
/// the dissipated energy and the validity flag.
fn oracle(q: &CohesiveDissipatedEnergyQuery) -> (f32, bool) {
    if !q.stiffness.is_finite()
        || !q.strength.is_finite()
        || !q.fracture_energy.is_finite()
        || !q.kappa.is_finite()
        || q.stiffness <= 0.0
        || q.strength <= 0.0
        || q.fracture_energy <= 0.0
    {
        return (0.0, false);
    }
    let onset = q.strength / q.stiffness;
    let final_sep = 2.0 * q.fracture_energy / q.strength;
    if final_sep <= onset {
        return (0.0, false);
    }
    let energy = if q.kappa <= onset {
        0.0
    } else if q.kappa >= final_sep {
        q.fracture_energy
    } else {
        0.5 * q.strength * final_sep * (q.kappa - onset) / (final_sep - onset)
    };
    (energy, true)
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
/// `valid` flag exactly, and the `energy` scalar to tolerance when valid.
fn assert_parity(
    gpu: &CohesiveDissipatedEnergyResult,
    q: &CohesiveDissipatedEnergyQuery,
    label: &str,
) {
    let (energy, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid {
        assert!(
            close(gpu.energy, energy),
            "{label}: energy mismatch gpu={} oracle={}",
            gpu.energy,
            energy
        );
    } else {
        assert_eq!(
            gpu.energy, 0.0,
            "{label}: invalid query must yield 0 energy"
        );
    }
}

#[test]
fn mid_segment_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    // K=1, sigma_c=1, G_c=1 -> onset=1, final=2; kappa=1.5 at the segment
    // midpoint dissipates half the fracture energy: 0.5.
    let q = CohesiveDissipatedEnergyQuery::new(1.0, 1.0, 1.0, 1.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].energy, 0.5));
    assert_parity(&out[0], &q, "mid_segment");
}

#[test]
fn below_onset_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    // onset=1; kappa=0.5 is still in the elastic range -> no dissipation.
    let q = CohesiveDissipatedEnergyQuery::new(1.0, 1.0, 1.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].energy, 0.0));
    assert_parity(&out[0], &q, "below_onset");
}

#[test]
fn above_final_is_fracture_energy() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    // final=2; kappa=3 is fully decohered -> the whole fracture energy G_c=1.
    let q = CohesiveDissipatedEnergyQuery::new(1.0, 1.0, 1.0, 3.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].energy, 1.0));
    assert_parity(&out[0], &q, "above_final");
}

#[test]
fn boundary_at_onset_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    // kappa exactly at onset=1 falls on the `kappa <= onset` branch -> 0. Both
    // host and device derive onset the same way, so the exact knee agrees.
    let q = CohesiveDissipatedEnergyQuery::new(1.0, 1.0, 1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].energy, 0.0));
    assert_parity(&out[0], &q, "boundary_onset");
}

#[test]
fn boundary_at_final_is_fracture_energy() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    // kappa exactly at final=2 falls on the `kappa >= final` branch -> G_c=1.
    let q = CohesiveDissipatedEnergyQuery::new(1.0, 1.0, 1.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].energy, 1.0));
    assert_parity(&out[0], &q, "boundary_final");
}

#[test]
fn nan_input_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    let q = CohesiveDissipatedEnergyQuery::new(f32::NAN, 1.0, 1.0, 1.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_eq!(out[0].energy, 0.0);
    assert_parity(&out[0], &q, "nan");
}

#[test]
fn infinite_input_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    let positive = CohesiveDissipatedEnergyQuery::new(1.0, 1.0, f32::INFINITY, 1.5);
    let negative = CohesiveDissipatedEnergyQuery::new(1.0, 1.0, 1.0, f32::NEG_INFINITY);
    let out = gpu.evaluate(&ctx, &[positive, negative]);
    assert_eq!(out.len(), 2);
    assert!(!out[0].valid, "infinite fracture energy should be invalid");
    assert!(!out[1].valid, "infinite kappa should be invalid");
    assert_eq!(out[0].energy, 0.0);
    assert_eq!(out[1].energy, 0.0);
    assert_parity(&out[0], &positive, "inf_pos");
    assert_parity(&out[1], &negative, "inf_neg");
}

#[test]
fn non_positive_parameters_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    let zero_stiffness = CohesiveDissipatedEnergyQuery::new(0.0, 1.0, 1.0, 1.5);
    let neg_strength = CohesiveDissipatedEnergyQuery::new(1.0, -1.0, 1.0, 1.5);
    let zero_energy = CohesiveDissipatedEnergyQuery::new(1.0, 1.0, 0.0, 1.5);
    let out = gpu.evaluate(&ctx, &[zero_stiffness, neg_strength, zero_energy]);
    assert_eq!(out.len(), 3);
    assert!(!out[0].valid, "zero stiffness should be invalid");
    assert!(!out[1].valid, "negative strength should be invalid");
    assert!(!out[2].valid, "zero fracture energy should be invalid");
    assert_parity(&out[0], &zero_stiffness, "zero_stiffness");
    assert_parity(&out[1], &neg_strength, "neg_strength");
    assert_parity(&out[2], &zero_energy, "zero_energy");
}

#[test]
fn final_not_above_onset_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    // K=1, sigma_c=10, G_c=1 -> onset=10, final=0.2; final <= onset means the
    // softening branch does not exist, so the model is invalid.
    let q = CohesiveDissipatedEnergyQuery::new(1.0, 10.0, 1.0, 5.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_eq!(out[0].energy, 0.0);
    assert_parity(&out[0], &q, "final_le_onset");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    let queries = vec![
        CohesiveDissipatedEnergyQuery::new(1.0, 1.0, 1.0, 1.5),
        CohesiveDissipatedEnergyQuery::new(f32::NAN, 1.0, 1.0, 1.5),
        CohesiveDissipatedEnergyQuery::new(1.0, 1.0, 1.0, 0.5),
        CohesiveDissipatedEnergyQuery::new(1.0, 10.0, 1.0, 5.0),
        CohesiveDissipatedEnergyQuery::new(2.0, 1.0, 4.0, 3.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(out[0].valid);
    assert!(!out[1].valid);
    assert!(out[2].valid);
    assert!(!out[3].valid);
    assert!(out[4].valid);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveDissipatedEnergy::new(&ctx);
    let mut lcg = Lcg::new(0x0C0E_51BE);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Parameters chosen so onset = sigma_c/K is tiny while final =
        // 2*G_c/sigma_c is order 1 or larger, keeping final clearly above onset.
        let stiffness = lcg.next_range(1.0e3, 1.0e4);
        let strength = lcg.next_range(1.0, 10.0);
        let fracture_energy = lcg.next_range(1.0, 10.0);
        let onset = strength / stiffness;
        let final_sep = 2.0 * fracture_energy / strength;
        // Guard the softening band is well separated before sampling kappa.
        let span = final_sep - onset;
        if span <= 0.0 {
            continue;
        }
        let margin = 0.05 * span;
        // Sample kappa across all three branches but reject near either knee so
        // the branch decision agrees between host and device.
        let kappa = lcg.next_range(0.0, final_sep * 1.5);
        if (kappa - onset).abs() < margin || (kappa - final_sep).abs() < margin {
            continue;
        }
        queries.push(CohesiveDissipatedEnergyQuery::new(
            stiffness,
            strength,
            fracture_energy,
            kappa,
        ));
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
