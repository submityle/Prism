//! Real-device parity for the cohesive-zone onset/final separation twin:
//! [`GpuCohesiveOnsetFinalSeparation`](prism_volumetric_gpu::cohesive_onset_final_separation::GpuCohesiveOnsetFinalSeparation)
//! must reproduce the `CPU` golden `CohesiveModel::new` of
//! `prism_physics_core::collider::cohesive_zone`. A bilinear cohesive law
//! derives the damage-onset separation `δ₀ = σ_c / K` and the final separation
//! `δ_f = 2 · G_c / σ_c`, and is valid only when every parameter is finite,
//! `K`/`σ_c`/`G_c` are strictly positive, the shear weight `β` is non-negative,
//! and the derived `δ_f` is strictly greater than `δ₀`.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness and sign gates, the two divisions in the golden operator
//! order, then the strict `δ_f > δ₀` check — written out directly so the test
//! never imports `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover a nominal valid model, every degenerate rejection
//! (non-finite, non-positive `K`/`σ_c`/`G_c`, negative `β`, and a derived
//! `δ_f <= δ₀`), a boundary where `δ_f` sits just above `δ₀` with ample margin,
//! a batch of two or more elements mixing valid and invalid parameters to
//! validate the `std430` stride, and an empty batch the host short-circuits
//! with no dispatch. A sweep over random parameters whose derived separations
//! stay well-ordered follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through two divisions and a multiply, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact (a
//! `GPU` may contract a multiply-add). Each valid separation is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag
//! is compared exactly. The fixtures keep the parameters strictly inside the
//! valid region (and `final_sep` comfortably above `onset`) so the validity
//! decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::cohesive_zone`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cohesive_onset_final_separation::{
    CohesiveOnsetFinalSeparationQuery, CohesiveOnsetFinalSeparationResult,
    GpuCohesiveOnsetFinalSeparation,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `CohesiveModel::new` in the golden
/// operator order, returning the two separations and the validity flag.
fn oracle(q: &CohesiveOnsetFinalSeparationQuery) -> (f32, f32, bool) {
    let k = q.stiffness;
    let s = q.strength;
    let g = q.fracture_energy;
    let b = q.shear_weight;
    if !k.is_finite()
        || !s.is_finite()
        || !g.is_finite()
        || !b.is_finite()
        || k <= 0.0
        || s <= 0.0
        || g <= 0.0
        || b < 0.0
    {
        return (0.0, 0.0, false);
    }
    let onset = s / k;
    let final_sep = 2.0 * g / s;
    if final_sep <= onset {
        return (0.0, 0.0, false);
    }
    (onset, final_sep, true)
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
/// `valid` flag exactly, and each separation to tolerance when valid.
fn assert_parity(
    gpu: &CohesiveOnsetFinalSeparationResult,
    q: &CohesiveOnsetFinalSeparationQuery,
    label: &str,
) {
    let (onset, final_sep, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid {
        assert!(
            close(gpu.onset, onset),
            "{label}: onset mismatch gpu={} oracle={}",
            gpu.onset,
            onset
        );
        assert!(
            close(gpu.final_sep, final_sep),
            "{label}: final_sep mismatch gpu={} oracle={}",
            gpu.final_sep,
            final_sep
        );
    }
}

#[test]
fn nominal_model_derives_both_separations() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    // K=1e6, σc=1e3, Gc=1, β=1 → onset=1e-3, final_sep=2e-3.
    let q = CohesiveOnsetFinalSeparationQuery::new(1.0e6, 1.0e3, 1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].onset, 1.0e-3));
    assert!(close(out[0].final_sep, 2.0e-3));
    assert_parity(&out[0], &q, "nominal");
}

#[test]
fn nan_stiffness_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    let q = CohesiveOnsetFinalSeparationQuery::new(f32::NAN, 1.0e3, 1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_eq!(out[0].onset, 0.0);
    assert_eq!(out[0].final_sep, 0.0);
    assert_parity(&out[0], &q, "nan");
}

#[test]
fn infinite_strength_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    let q = CohesiveOnsetFinalSeparationQuery::new(1.0e6, f32::INFINITY, 1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "inf");
}

#[test]
fn zero_stiffness_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    let q = CohesiveOnsetFinalSeparationQuery::new(0.0, 1.0e3, 1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "zero_stiffness");
}

#[test]
fn negative_strength_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    let q = CohesiveOnsetFinalSeparationQuery::new(1.0e6, -1.0e3, 1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "neg_strength");
}

#[test]
fn zero_energy_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    let q = CohesiveOnsetFinalSeparationQuery::new(1.0e6, 1.0e3, 0.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "zero_energy");
}

#[test]
fn negative_shear_weight_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    let q = CohesiveOnsetFinalSeparationQuery::new(1.0e6, 1.0e3, 1.0, -0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "neg_shear");
}

#[test]
fn final_not_above_onset_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    // K=1, σc=10, Gc=1 → onset=10, final_sep=0.2 → final_sep <= onset → invalid.
    let q = CohesiveOnsetFinalSeparationQuery::new(1.0, 10.0, 1.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid);
    assert_parity(&out[0], &q, "final_le_onset");
}

#[test]
fn boundary_final_just_above_onset_is_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    // K=10, σc=1, Gc=1 → onset=0.1, final_sep=2.0: valid with a large margin so
    // the ordering decision cannot be flipped by round-off.
    let q = CohesiveOnsetFinalSeparationQuery::new(10.0, 1.0, 1.0, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(close(out[0].onset, 0.1));
    assert!(close(out[0].final_sep, 2.0));
    assert_parity(&out[0], &q, "boundary");
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    let queries = vec![
        CohesiveOnsetFinalSeparationQuery::new(1.0e6, 1.0e3, 1.0, 1.0),
        CohesiveOnsetFinalSeparationQuery::new(1.0, 10.0, 1.0, 1.0),
        CohesiveOnsetFinalSeparationQuery::new(10.0, 1.0, 1.0, 0.0),
        CohesiveOnsetFinalSeparationQuery::new(1.0e6, 1.0e3, 1.0, -1.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(out[0].valid);
    assert!(!out[1].valid);
    assert!(out[2].valid);
    assert!(!out[3].valid);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveOnsetFinalSeparation::new(&ctx);
    let mut lcg = Lcg::new(0x0C0E_51BE);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // K large so onset = σc/K stays tiny; σc moderate and Gc >= 1 so
        // final_sep = 2*Gc/σc stays >= 0.2 — thus final_sep is always far above
        // onset (margin well over 0.05) and the validity decision is stable.
        let stiffness = lcg.next_range(1.0e4, 1.0e5);
        let strength = lcg.next_range(1.0, 10.0);
        let fracture_energy = lcg.next_range(1.0, 10.0);
        let shear_weight = lcg.next_range(0.0, 2.0);
        queries.push(CohesiveOnsetFinalSeparationQuery::new(
            stiffness,
            strength,
            fracture_energy,
            shear_weight,
        ));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "sweep[{i}] should be valid");
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
