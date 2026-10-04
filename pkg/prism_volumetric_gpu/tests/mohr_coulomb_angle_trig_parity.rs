//! Real-device parity for the Mohr-Coulomb angle-trig twin:
//! [`GpuMohrCoulombAngleTrig`](prism_volumetric_gpu::mohr_coulomb_angle_trig::GpuMohrCoulombAngleTrig)
//! must reproduce the `CPU` golden `MohrCoulombModel::from_angles` of
//! `prism_physics_core::collider::tet_fem_mohr_coulomb_plasticity`. From a
//! friction angle `φ` and dilation angle `ψ` (both in degrees) the reference
//! derives the trio `sin φ`, `cos φ`, `sin ψ`, valid only when `0 < φ < 90` and
//! `0 <= ψ <= φ` with both angles finite.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finiteness and range guards, then the `f64`-intermediate trig narrowed
//! to `f32` exactly as the golden does — written out directly so the test never
//! imports `prism_render_architecture` or `prism_physics_core`.
//!
//! The fixtures cover normal angle pairs, the inclusive boundaries (`ψ = 0`,
//! `ψ = φ`, `φ` just under `90`), the degenerate rejections (non-finite angles,
//! `φ` outside `(0, 90)`, `ψ` outside `[0, φ]`), a mixed batch that validates the
//! `std430` stride, and an empty batch the host short-circuits. A sweep over
//! random angles well inside the valid band follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The reference evaluates the trig in `f64` then narrows to `f32`; the kernel
//! evaluates `f32` `sin`/`cos` directly, so `CPU` and `GPU` are not bit-exact.
//! Each valid trig scalar is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly. The
//! fixtures keep random angles well inside the valid band so the validity
//! decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_mohr_coulomb_plasticity`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::mohr_coulomb_angle_trig::{
    GpuMohrCoulombAngleTrig, MohrCoulombAngleTrigQuery, MohrCoulombAngleTrigResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces `from_angles`' trig trio with the golden
/// `f64`-intermediate narrowed to `f32`, returning the three scalars and the
/// validity flag.
fn oracle(q: &MohrCoulombAngleTrigQuery) -> (f32, f32, f32, bool) {
    let f = q.friction_degrees;
    let d = q.dilation_degrees;
    if !(f.is_finite() && f > 0.0 && f < 90.0) || !(d.is_finite() && d >= 0.0 && d <= f) {
        return (0.0, 0.0, 0.0, false);
    }
    let phi = f64::from(f).to_radians();
    let psi = f64::from(d).to_radians();
    (phi.sin() as f32, phi.cos() as f32, psi.sin() as f32, true)
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
/// `valid` flag exactly, and the three trig scalars to tolerance when valid.
fn assert_parity(gpu: &MohrCoulombAngleTrigResult, q: &MohrCoulombAngleTrigQuery, label: &str) {
    let (sin_phi, cos_phi, sin_psi, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid {
        assert!(
            close(gpu.sin_phi, sin_phi),
            "{label}: sin_phi mismatch gpu={} oracle={}",
            gpu.sin_phi,
            sin_phi
        );
        assert!(
            close(gpu.cos_phi, cos_phi),
            "{label}: cos_phi mismatch gpu={} oracle={}",
            gpu.cos_phi,
            cos_phi
        );
        assert!(
            close(gpu.sin_psi, sin_psi),
            "{label}: sin_psi mismatch gpu={} oracle={}",
            gpu.sin_psi,
            sin_psi
        );
    } else {
        assert_eq!(gpu.sin_phi, 0.0, "{label}: invalid sin_phi should be 0");
        assert_eq!(gpu.cos_phi, 0.0, "{label}: invalid cos_phi should be 0");
        assert_eq!(gpu.sin_psi, 0.0, "{label}: invalid sin_psi should be 0");
    }
}

#[test]
fn normal_angle_pairs_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombAngleTrig::new(&ctx);
    let queries = vec![
        MohrCoulombAngleTrigQuery::new(30.0, 0.0),
        MohrCoulombAngleTrigQuery::new(30.0, 15.0),
        MohrCoulombAngleTrigQuery::new(45.0, 45.0),
        MohrCoulombAngleTrigQuery::new(60.0, 20.0),
        MohrCoulombAngleTrigQuery::new(80.0, 10.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "normal[{i}] should be valid");
        assert_parity(res, q, &format!("normal[{i}]"));
    }
}

#[test]
fn non_finite_angles_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombAngleTrig::new(&ctx);
    let queries = vec![
        MohrCoulombAngleTrigQuery::new(f32::NAN, 10.0),
        MohrCoulombAngleTrigQuery::new(f32::INFINITY, 10.0),
        MohrCoulombAngleTrigQuery::new(f32::NEG_INFINITY, 10.0),
        MohrCoulombAngleTrigQuery::new(30.0, f32::NAN),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "non_finite[{i}] should be invalid");
        assert_parity(res, q, &format!("non_finite[{i}]"));
    }
}

#[test]
fn out_of_range_friction_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombAngleTrig::new(&ctx);
    // phi <= 0 and phi >= 90 both reject.
    let queries = vec![
        MohrCoulombAngleTrigQuery::new(0.0, 0.0),
        MohrCoulombAngleTrigQuery::new(-5.0, 0.0),
        MohrCoulombAngleTrigQuery::new(90.0, 10.0),
        MohrCoulombAngleTrigQuery::new(120.0, 10.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "friction[{i}] should be invalid");
        assert_parity(res, q, &format!("friction[{i}]"));
    }
}

#[test]
fn out_of_range_dilation_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombAngleTrig::new(&ctx);
    // psi < 0 and psi > phi both reject.
    let queries = vec![
        MohrCoulombAngleTrigQuery::new(30.0, -5.0),
        MohrCoulombAngleTrigQuery::new(30.0, 40.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "dilation[{i}] should be invalid");
        assert_parity(res, q, &format!("dilation[{i}]"));
    }
}

#[test]
fn inclusive_boundaries_are_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombAngleTrig::new(&ctx);
    // psi = 0, psi = phi, and phi just under 90 are all accepted.
    let queries = vec![
        MohrCoulombAngleTrigQuery::new(35.0, 0.0),
        MohrCoulombAngleTrigQuery::new(50.0, 50.0),
        MohrCoulombAngleTrigQuery::new(89.9, 10.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "boundary[{i}] should be valid");
        assert_parity(res, q, &format!("boundary[{i}]"));
    }
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombAngleTrig::new(&ctx);
    let queries = vec![
        MohrCoulombAngleTrigQuery::new(30.0, 15.0),
        MohrCoulombAngleTrigQuery::new(120.0, 10.0),
        MohrCoulombAngleTrigQuery::new(45.0, 45.0),
        MohrCoulombAngleTrigQuery::new(30.0, 40.0),
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
    let gpu = GpuMohrCoulombAngleTrig::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMohrCoulombAngleTrig::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Friction in [1, 89]; dilation in [0, phi] with a small margin so the
        // validity decision stays stable under round-off.
        let friction = lcg.next_range(1.0, 89.0);
        let margin = 0.05_f32;
        let hi = (friction - margin).max(0.0);
        let dilation = lcg.next_range(0.0, hi);
        queries.push(MohrCoulombAngleTrigQuery::new(friction, dilation));
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
