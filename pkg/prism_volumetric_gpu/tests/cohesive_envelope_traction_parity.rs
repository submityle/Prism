//! Real-device parity for the cohesive-envelope traction twin:
//! [`GpuCohesiveEnvelopeTraction`](prism_volumetric_gpu::cohesive_envelope_traction::GpuCohesiveEnvelopeTraction)
//! must reproduce the `CPU` golden `CohesiveZone::envelope_traction` of
//! `prism_physics_core::collider::cohesive_zone`. The bilinear cohesive
//! envelope maps the effective separation `lambda` to a traction magnitude:
//! zero below contact, a linear elastic ramp `K * lambda` up to the onset
//! separation `delta0`, a linear softening branch
//! `sigma_c * (delta_f - lambda) / (delta_f - delta0)` between `delta0` and the
//! final separation `delta_f`, and zero once fully separated.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the exact golden branch order — written out directly so the test never
//! imports `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover the below-contact branch (`lambda <= 0`), the elastic
//! ramp (`0 < lambda <= delta0`), the softening branch
//! (`delta0 < lambda < delta_f`), the fully separated branch
//! (`lambda >= delta_f`), a batch of two or more elements that mixes every
//! branch to validate the `std430` stride, and an empty batch the host
//! short-circuits with no dispatch. A sweep over random envelopes with
//! rejection-sampled separations well away from the three branch knees follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic (a multiply on the ramp, a subtract-multiply-divide
//! on the softening branch) threads through operators a `GPU` may contract, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact. The
//! `traction` scalar is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`). The sweep keeps `lambda` well away from the three
//! branch knees (`0`, `delta0`, `delta_f`) so the piecewise selection agrees on
//! both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::cohesive_zone`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cohesive_envelope_traction::{
    CohesiveEnvelopeTractionQuery, CohesiveEnvelopeTractionResult, GpuCohesiveEnvelopeTraction,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `envelope_traction` in the exact golden
/// branch order.
fn oracle(q: &CohesiveEnvelopeTractionQuery) -> f32 {
    let lambda = q.lambda;
    if lambda <= 0.0 {
        return 0.0;
    }
    if lambda <= q.onset_separation {
        return q.stiffness * lambda;
    }
    if lambda >= q.final_separation {
        return 0.0;
    }
    q.strength * (q.final_separation - lambda) / (q.final_separation - q.onset_separation)
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
fn assert_parity(
    gpu: &CohesiveEnvelopeTractionResult,
    q: &CohesiveEnvelopeTractionQuery,
    label: &str,
) {
    let expected = oracle(q);
    assert!(
        close(gpu.traction, expected),
        "{label}: traction mismatch gpu={} oracle={}",
        gpu.traction,
        expected
    );
}

#[test]
fn below_contact_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveEnvelopeTraction::new(&ctx);
    let at_zero = CohesiveEnvelopeTractionQuery::new(2.0, 3.0, 0.5, 2.0, 0.0);
    let negative = CohesiveEnvelopeTractionQuery::new(2.0, 3.0, 0.5, 2.0, -0.4);
    let out = gpu.evaluate(&ctx, &[at_zero, negative]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].traction, 0.0);
    assert_eq!(out[1].traction, 0.0);
    assert_parity(&out[0], &at_zero, "at_zero");
    assert_parity(&out[1], &negative, "negative");
}

#[test]
fn elastic_ramp_is_linear() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveEnvelopeTraction::new(&ctx);
    // 0 < lambda <= delta0 -> K*lambda = 4*0.25 = 1.0.
    let q = CohesiveEnvelopeTractionQuery::new(4.0, 5.0, 0.5, 2.0, 0.25);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].traction, 1.0));
    assert_parity(&out[0], &q, "elastic");
}

#[test]
fn elastic_at_onset_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveEnvelopeTraction::new(&ctx);
    // lambda == delta0 is still the elastic branch (<=): K*delta0 = 4*0.5 = 2.0.
    let q = CohesiveEnvelopeTractionQuery::new(4.0, 5.0, 0.5, 2.0, 0.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].traction, 2.0));
    assert_parity(&out[0], &q, "onset");
}

#[test]
fn softening_branch_decays_linearly() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveEnvelopeTraction::new(&ctx);
    // delta0 < lambda < delta_f -> sigma_c*(delta_f-lambda)/(delta_f-delta0)
    //   = 6*(2.0-1.0)/(2.0-0.5) = 6/1.5 = 4.0.
    let q = CohesiveEnvelopeTractionQuery::new(4.0, 6.0, 0.5, 2.0, 1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(close(out[0].traction, 4.0));
    assert_parity(&out[0], &q, "softening");
}

#[test]
fn fully_separated_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveEnvelopeTraction::new(&ctx);
    // lambda >= delta_f -> 0 (test both equal and beyond).
    let at_final = CohesiveEnvelopeTractionQuery::new(4.0, 6.0, 0.5, 2.0, 2.0);
    let beyond = CohesiveEnvelopeTractionQuery::new(4.0, 6.0, 0.5, 2.0, 5.0);
    let out = gpu.evaluate(&ctx, &[at_final, beyond]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].traction, 0.0);
    assert_eq!(out[1].traction, 0.0);
    assert_parity(&out[0], &at_final, "at_final");
    assert_parity(&out[1], &beyond, "beyond");
}

#[test]
fn batch_mixes_all_branches_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveEnvelopeTraction::new(&ctx);
    let queries = vec![
        // below contact
        CohesiveEnvelopeTractionQuery::new(2.0, 3.0, 0.5, 2.0, -0.1),
        // elastic ramp
        CohesiveEnvelopeTractionQuery::new(4.0, 5.0, 0.5, 2.0, 0.25),
        // softening
        CohesiveEnvelopeTractionQuery::new(4.0, 6.0, 0.5, 2.0, 1.0),
        // fully separated
        CohesiveEnvelopeTractionQuery::new(4.0, 6.0, 0.5, 2.0, 3.0),
        // another elastic (odd length to stress stride)
        CohesiveEnvelopeTractionQuery::new(1.5, 2.0, 0.8, 2.5, 0.4),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveEnvelopeTraction::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveEnvelopeTraction::new(&ctx);
    let mut lcg = Lcg::new(0x00C0_FFEE);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Valid envelope: K>0, sigma_c>0, delta_f > delta0 > 0.
        let stiffness = lcg.next_range(0.5, 5.0);
        let strength = lcg.next_range(0.5, 5.0);
        let delta0 = lcg.next_range(0.1, 1.0);
        let delta_f = delta0 + lcg.next_range(0.5, 3.0);
        // Rejection-sample lambda across the whole active range but keep it well
        // away from all three branch knees so the piecewise selection cannot be
        // flipped by f32 round-off.
        let lambda = lcg.next_range(-0.5, delta_f + 1.0);
        let margin = 0.05 * delta_f;
        let near_zero = lambda.abs() < margin;
        let near_onset = (lambda - delta0).abs() < margin;
        let near_final = (lambda - delta_f).abs() < margin;
        if near_zero || near_onset || near_final {
            continue;
        }
        queries.push(CohesiveEnvelopeTractionQuery::new(
            stiffness, strength, delta0, delta_f, lambda,
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
