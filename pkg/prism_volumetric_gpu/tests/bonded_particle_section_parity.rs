//! Real-device parity for the bonded-particle section twin:
//! [`GpuBondedParticleSection`](prism_volumetric_gpu::bonded_particle_section::GpuBondedParticleSection)
//! must reproduce the `CPU` golden section quantities that `BondModel::new`
//! derives from a bond radius `R` in
//! `prism_physics_core::collider::bonded_particle`. Those quantities are the
//! cross-sectional `area = PI * R^2`, the second moment of area
//! `inertia = 0.25 * PI * R^4`, and the polar moment `polar = 0.5 * PI * R^4`,
//! all defined only when the radius is finite and strictly positive.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out in the golden left-associative operator order so the test never
//! imports `prism_render_architecture` or `prism_physics_core`. The oracle uses
//! the same decimal `PI` literal (`3.1415927`) as the kernel so neither side
//! drifts on the shared constant.
//!
//! The fixtures cover a unit radius (where the quantities collapse to `PI`,
//! `PI/4`, `PI/2`), degenerate radii (`zero`, `negative`, `NaN`, `infinite`,
//! all invalid), a tiny positive radius, a large radius, a batch of two or more
//! elements that mixes valid and invalid queries to validate the `std430`
//! stride, and an empty batch the host short-circuits with no dispatch. A sweep
//! over random strictly-positive radii follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through multiplies, so `CPU` and `GPU`
//! evaluate the same closed form but need not be bit-exact (a `GPU` may contract
//! a multiply-add). Each valid scalar is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag
//! is compared exactly. The fixtures keep radii strictly positive and well away
//! from zero/non-finite so the validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::bonded_particle`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::bonded_particle_section::{
    BondedParticleSectionQuery, BondedParticleSectionResult, GpuBondedParticleSection,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// The shared circle constant, as the same decimal literal the kernel uses so
/// neither side drifts on the constant's low bits.
const PI: f32 = 3.1415927;

/// Independent host oracle: reproduces the `BondModel::new` section quantities
/// in the golden left-associative operator order, returning
/// `(area, inertia, polar, valid)`.
fn oracle(q: &BondedParticleSectionQuery) -> (f32, f32, f32, u32) {
    let radius = q.radius;
    if !radius.is_finite() || radius <= 0.0 {
        return (0.0, 0.0, 0.0, 0);
    }
    let r2 = radius * radius;
    (PI * r2, 0.25 * PI * r2 * r2, 0.5 * PI * r2 * r2, 1)
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
/// `valid` flag exactly, and the section scalars to tolerance when valid.
fn assert_parity(gpu: &BondedParticleSectionResult, q: &BondedParticleSectionQuery, label: &str) {
    let (area, inertia, polar, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        assert!(
            close(gpu.area, area),
            "{label}: area mismatch gpu={} oracle={}",
            gpu.area,
            area
        );
        assert!(
            close(gpu.inertia, inertia),
            "{label}: inertia mismatch gpu={} oracle={}",
            gpu.inertia,
            inertia
        );
        assert!(
            close(gpu.polar, polar),
            "{label}: polar mismatch gpu={} oracle={}",
            gpu.polar,
            polar
        );
    } else {
        assert_eq!(gpu.area, 0.0, "{label}: invalid area should be zero");
        assert_eq!(gpu.inertia, 0.0, "{label}: invalid inertia should be zero");
        assert_eq!(gpu.polar, 0.0, "{label}: invalid polar should be zero");
    }
}

#[test]
fn unit_radius_section() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBondedParticleSection::new(&ctx);
    // R = 1 => area = PI, inertia = PI/4, polar = PI/2.
    let q = BondedParticleSectionQuery::new(1.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].area, PI));
    assert!(close(out[0].inertia, 0.25 * PI));
    assert!(close(out[0].polar, 0.5 * PI));
    assert_parity(&out[0], &q, "unit");
}

#[test]
fn zero_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBondedParticleSection::new(&ctx);
    let q = BondedParticleSectionQuery::new(0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "zero");
}

#[test]
fn negative_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBondedParticleSection::new(&ctx);
    let q = BondedParticleSectionQuery::new(-2.5);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "negative");
}

#[test]
fn nan_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBondedParticleSection::new(&ctx);
    let q = BondedParticleSectionQuery::new(f32::NAN);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "nan");
}

#[test]
fn infinite_radius_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBondedParticleSection::new(&ctx);
    let q = BondedParticleSectionQuery::new(f32::INFINITY);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0);
    assert_parity(&out[0], &q, "infinite");
}

#[test]
fn tiny_positive_radius() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBondedParticleSection::new(&ctx);
    let q = BondedParticleSectionQuery::new(1.0e-3);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "tiny");
}

#[test]
fn large_radius() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBondedParticleSection::new(&ctx);
    let q = BondedParticleSectionQuery::new(50.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "large");
}

#[test]
fn stride_regression_two_element_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBondedParticleSection::new(&ctx);
    let queries = vec![
        BondedParticleSectionQuery::new(2.0),
        BondedParticleSectionQuery::new(-1.0),
        BondedParticleSectionQuery::new(7.5),
        BondedParticleSectionQuery::new(0.0),
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
    let gpu = GpuBondedParticleSection::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBondedParticleSection::new(&ctx);
    let mut lcg = Lcg::new(0x00B0_11D5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Strictly-positive radii well away from zero so validity is stable.
        let radius = lcg.next_range(0.05, 10.0);
        queries.push(BondedParticleSectionQuery::new(radius));
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
