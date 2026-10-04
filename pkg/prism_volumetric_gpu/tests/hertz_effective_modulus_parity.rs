//! Real-device parity for the Hertzian effective-modulus twin:
//! [`GpuHertzEffectiveModulus`](prism_volumetric_gpu::hertz_effective_modulus::GpuHertzEffectiveModulus)
//! must reproduce the `CPU` golden `HertzModel::new` effective modulus of
//! `prism_physics_core::collider::hertz_contact`. For two bodies of the same
//! material the effective modulus is `E* = E / (2 · (1 − ν²))` when the inputs
//! are finite, `E > 0`, `ν ∈ [0, 0.5)`, and the result is itself finite and
//! positive; otherwise the pair is invalid.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness and range guards, then `E / (2·(1 − ν²))` in the golden
//! operator order — written out directly so the test never imports
//! `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover `ν = 0` (where `E* = E/2`), a typical `ν = 0.3`, a `ν`
//! near but below `0.5`, `ν >= 0.5` (invalid), negative `ν` (invalid), a
//! non-positive `young_modulus` (invalid), non-finite inputs (invalid), a batch
//! of two or more elements that mixes valid and invalid pairs to validate the
//! `std430` stride, and an empty batch the host short-circuits with no dispatch.
//! A sweep over random valid materials follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through multiplies, a subtract and a
//! division, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). The valid
//! `effective_modulus` scalar is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly. The
//! sweep keeps `ν` well below `0.5` and `young_modulus` strictly positive so the
//! validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::hertz_contact`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hertz_effective_modulus::{
    GpuHertzEffectiveModulus, HertzEffectiveModulusQuery, HertzEffectiveModulusResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `HertzModel::new`'s effective modulus in
/// the golden operator order, returning the modulus and the validity flag.
fn oracle(q: &HertzEffectiveModulusQuery) -> (f32, u32) {
    let young = q.young_modulus;
    let nu = q.poisson_ratio;
    let finite = young.is_finite() && nu.is_finite();
    let base = finite && young > 0.0 && nu >= 0.0 && nu < 0.5;
    if !base {
        return (0.0, 0);
    }
    let e = young / (2.0 * (1.0 - nu * nu));
    if e.is_finite() && e > 0.0 {
        (e, 1)
    } else {
        (0.0, 0)
    }
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
/// `valid` flag exactly, and the `effective_modulus` scalar to tolerance when
/// valid.
fn assert_parity(gpu: &HertzEffectiveModulusResult, q: &HertzEffectiveModulusQuery, label: &str) {
    let (modulus, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.effective_modulus, modulus),
            "{label}: effective_modulus mismatch gpu={} oracle={}",
            gpu.effective_modulus,
            modulus
        );
    }
}

#[test]
fn zero_poisson_halves_the_modulus() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    // nu = 0 => E* = E / 2.
    let q = HertzEffectiveModulusQuery::new(2.0e6, 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].effective_modulus, 1.0e6));
    assert_parity(&out[0], &q, "zero_nu");
}

#[test]
fn typical_poisson_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    // nu = 0.3 => E* = E / (2*(1-0.09)) = E / 1.82.
    let q = HertzEffectiveModulusQuery::new(1.0e7, 0.3);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].effective_modulus, 1.0e7 / 1.82));
    assert_parity(&out[0], &q, "typical");
}

#[test]
fn high_poisson_below_half_is_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    // nu = 0.45, well away from the 0.5 knee, still valid.
    let q = HertzEffectiveModulusQuery::new(5.0e5, 0.45);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "high_nu");
}

#[test]
fn poisson_at_or_above_half_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    let at_half = HertzEffectiveModulusQuery::new(1.0e6, 0.5);
    let above = HertzEffectiveModulusQuery::new(1.0e6, 0.7);
    let out = gpu.evaluate(&ctx, &[at_half, above]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "nu == 0.5 should be invalid (half-open)");
    assert_eq!(out[1].valid, 0, "nu > 0.5 should be invalid");
    assert_eq!(out[0].effective_modulus, 0.0);
    assert_eq!(out[1].effective_modulus, 0.0);
    assert_parity(&out[0], &at_half, "at_half");
    assert_parity(&out[1], &above, "above_half");
}

#[test]
fn negative_poisson_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    let q = HertzEffectiveModulusQuery::new(1.0e6, -0.1);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_eq!(out[0].effective_modulus, 0.0);
    assert_parity(&out[0], &q, "negative_nu");
}

#[test]
fn non_positive_young_modulus_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    let zero = HertzEffectiveModulusQuery::new(0.0, 0.3);
    let negative = HertzEffectiveModulusQuery::new(-1.0e6, 0.3);
    let out = gpu.evaluate(&ctx, &[zero, negative]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].valid, 0, "young == 0 should be invalid");
    assert_eq!(out[1].valid, 0, "young < 0 should be invalid");
    assert_eq!(out[0].effective_modulus, 0.0);
    assert_eq!(out[1].effective_modulus, 0.0);
    assert_parity(&out[0], &zero, "zero_young");
    assert_parity(&out[1], &negative, "negative_young");
}

#[test]
fn non_finite_inputs_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    let inf_young = HertzEffectiveModulusQuery::new(f32::INFINITY, 0.3);
    let nan_young = HertzEffectiveModulusQuery::new(f32::NAN, 0.3);
    let nan_nu = HertzEffectiveModulusQuery::new(1.0e6, f32::NAN);
    let out = gpu.evaluate(&ctx, &[inf_young, nan_young, nan_nu]);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].valid, 0, "inf young should be invalid");
    assert_eq!(out[1].valid, 0, "nan young should be invalid");
    assert_eq!(out[2].valid, 0, "nan nu should be invalid");
    for (i, (res, q)) in out
        .iter()
        .zip([inf_young, nan_young, nan_nu].iter())
        .enumerate()
    {
        assert_eq!(res.effective_modulus, 0.0);
        assert_parity(res, q, &format!("non_finite[{i}]"));
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    let queries = vec![
        HertzEffectiveModulusQuery::new(2.0e6, 0.0),
        HertzEffectiveModulusQuery::new(1.0e6, 0.6),
        HertzEffectiveModulusQuery::new(3.0e6, 0.25),
        HertzEffectiveModulusQuery::new(-5.0e5, 0.3),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[3].valid, 0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHertzEffectiveModulus::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Young's modulus strictly positive and well away from zero; Poisson
        // ratio in [0, 0.45], kept clear of the 0.5 knee so validity is stable.
        let young_modulus = lcg.next_range(1.0e3, 1.0e7);
        let poisson_ratio = lcg.next_range(0.0, 0.45);
        queries.push(HertzEffectiveModulusQuery::new(
            young_modulus,
            poisson_ratio,
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
