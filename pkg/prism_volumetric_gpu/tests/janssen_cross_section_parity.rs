//! Real-device parity for the Janssen silo cross-section twin:
//! [`GpuJanssenCrossSection`](prism_volumetric_gpu::janssen_cross_section::GpuJanssenCrossSection)
//! must reproduce the `CPU` golden `prism_physics_core::collider::janssen_pressure`'s
//! `SiloCrossSection` plus the depth-independent part of `JanssenProfile::new`.
//!
//! From a silo cross-section (hydraulic, circular or rectangular) and the
//! material parameters the reference derives the hydraulic radius `R`, the
//! characteristic depth `z_c = R / (μ_w K)`, and the saturation stresses
//! `σ_∞ = ρ g z_c`, `σ_h∞ = K σ_∞`, `τ_w∞ = μ_w σ_h∞`, valid only when every
//! shape and material parameter is finite and positive and the derived radius
//! and depth are finite and positive.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`
//! or `prism_physics_core`. The golden constant part is pure `f32`, so the host
//! oracle matches the device bit-closely modulo multiply-add contraction.
//!
//! The fixtures cover the three shapes with sane material parameters, the
//! degenerate rejections (non-finite or non-positive shape or material
//! parameter), small-but-positive boundaries, a mixed batch that validates the
//! `std430` stride, and an empty batch the host short-circuits. A sweep over
//! random parameters well inside the valid band follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The reference evaluates the constants in `f32`; the kernel reproduces the
//! same `f32` operator order, so `CPU` and `GPU` need not be bit-exact (a `GPU`
//! may contract a multiply-add). Each valid scalar is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag
//! is compared exactly. The fixtures keep random parameters well inside the
//! valid band so the validity decision agrees on both sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::janssen_pressure`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::janssen_cross_section::{
    GpuJanssenCrossSection, JanssenCrossSectionQuery, JanssenCrossSectionResult, SHAPE_CIRCULAR,
    SHAPE_HYDRAULIC, SHAPE_RECTANGULAR,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces the Janssen cross-section constants in
/// pure `f32`, returning the five scalars `(R, z_c, σ_v, σ_h, τ_w)` and the
/// validity flag.
fn oracle(q: &JanssenCrossSectionQuery) -> (f32, f32, f32, f32, f32, bool) {
    let (radius, shape_ok) = match q.shape_kind {
        SHAPE_HYDRAULIC => (q.param0, q.param0.is_finite() && q.param0 > 0.0),
        SHAPE_CIRCULAR => (0.5 * q.param0, q.param0.is_finite() && q.param0 > 0.0),
        SHAPE_RECTANGULAR => {
            let ok =
                q.param0.is_finite() && q.param1.is_finite() && q.param0 > 0.0 && q.param1 > 0.0;
            let r = (q.param0 * q.param1) / (2.0 * (q.param0 + q.param1));
            (r, ok)
        }
        _ => (0.0, false),
    };
    let radius_ok = shape_ok && radius.is_finite() && radius > 0.0;
    let mat_ok = [q.bulk_density, q.gravity, q.wall_friction, q.k_ratio]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0);
    if !radius_ok || !mat_ok {
        return (0.0, 0.0, 0.0, 0.0, 0.0, false);
    }
    let z_c = radius / (q.wall_friction * q.k_ratio);
    if !z_c.is_finite() || z_c <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, false);
    }
    let sigma_v = q.bulk_density * q.gravity * z_c;
    if !sigma_v.is_finite() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, false);
    }
    let sigma_h = q.k_ratio * sigma_v;
    let tau_w = q.wall_friction * sigma_h;
    (radius, z_c, sigma_v, sigma_h, tau_w, true)
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
/// `valid` flag exactly, and the five Janssen constants to tolerance when
/// valid, else all zero.
fn assert_parity(gpu: &JanssenCrossSectionResult, q: &JanssenCrossSectionQuery, label: &str) {
    let (r, z_c, sv, sh, tw, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid {
        assert!(
            close(gpu.hydraulic_radius, r),
            "{label}: hydraulic_radius mismatch gpu={} oracle={}",
            gpu.hydraulic_radius,
            r
        );
        assert!(
            close(gpu.characteristic_depth, z_c),
            "{label}: characteristic_depth mismatch gpu={} oracle={}",
            gpu.characteristic_depth,
            z_c
        );
        assert!(
            close(gpu.saturation_vertical, sv),
            "{label}: saturation_vertical mismatch gpu={} oracle={}",
            gpu.saturation_vertical,
            sv
        );
        assert!(
            close(gpu.saturation_horizontal, sh),
            "{label}: saturation_horizontal mismatch gpu={} oracle={}",
            gpu.saturation_horizontal,
            sh
        );
        assert!(
            close(gpu.saturation_wall_shear, tw),
            "{label}: saturation_wall_shear mismatch gpu={} oracle={}",
            gpu.saturation_wall_shear,
            tw
        );
    } else {
        assert_eq!(
            gpu.hydraulic_radius, 0.0,
            "{label}: invalid hydraulic_radius should be 0"
        );
        assert_eq!(
            gpu.characteristic_depth, 0.0,
            "{label}: invalid characteristic_depth should be 0"
        );
        assert_eq!(
            gpu.saturation_vertical, 0.0,
            "{label}: invalid saturation_vertical should be 0"
        );
        assert_eq!(
            gpu.saturation_horizontal, 0.0,
            "{label}: invalid saturation_horizontal should be 0"
        );
        assert_eq!(
            gpu.saturation_wall_shear, 0.0,
            "{label}: invalid saturation_wall_shear should be 0"
        );
    }
}

#[test]
fn normal_shapes_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenCrossSection::new(&ctx);
    // Sane granular material: rho=1500, g=9.81, mu_w=0.5, K=0.5.
    let queries = vec![
        JanssenCrossSectionQuery::hydraulic(1.2, 1500.0, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::circular(2.0, 1500.0, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::rectangular(2.0, 3.0, 1500.0, 9.81, 0.5, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "normal[{i}] should be valid");
        assert_parity(res, q, &format!("normal[{i}]"));
    }
}

#[test]
fn non_finite_shape_param_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenCrossSection::new(&ctx);
    let queries = vec![
        JanssenCrossSectionQuery::hydraulic(f32::NAN, 1500.0, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::circular(f32::INFINITY, 1500.0, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::rectangular(2.0, f32::NEG_INFINITY, 1500.0, 9.81, 0.5, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "non_finite_shape[{i}] should be invalid");
        assert_parity(res, q, &format!("non_finite_shape[{i}]"));
    }
}

#[test]
fn non_positive_shape_param_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenCrossSection::new(&ctx);
    let queries = vec![
        JanssenCrossSectionQuery::hydraulic(0.0, 1500.0, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::circular(-1.0, 1500.0, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::rectangular(2.0, 0.0, 1500.0, 9.81, 0.5, 0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "non_positive_shape[{i}] should be invalid");
        assert_parity(res, q, &format!("non_positive_shape[{i}]"));
    }
}

#[test]
fn non_finite_or_non_positive_material_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenCrossSection::new(&ctx);
    let queries = vec![
        JanssenCrossSectionQuery::hydraulic(1.2, f32::NAN, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::hydraulic(1.2, 1500.0, f32::INFINITY, 0.5, 0.5),
        JanssenCrossSectionQuery::hydraulic(1.2, 1500.0, 9.81, 0.0, 0.5),
        JanssenCrossSectionQuery::hydraulic(1.2, 1500.0, 9.81, 0.5, -0.5),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(!res.valid, "material[{i}] should be invalid");
        assert_parity(res, q, &format!("material[{i}]"));
    }
}

#[test]
fn small_positive_parameters_are_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenCrossSection::new(&ctx);
    let queries = vec![
        JanssenCrossSectionQuery::hydraulic(1.0e-2, 10.0, 1.0, 0.1, 0.1),
        JanssenCrossSectionQuery::circular(1.0e-2, 10.0, 1.0, 0.1, 0.1),
        JanssenCrossSectionQuery::rectangular(1.0e-2, 1.0e-2, 10.0, 1.0, 0.1, 0.1),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert!(res.valid, "small[{i}] should be valid");
        assert_parity(res, q, &format!("small[{i}]"));
    }
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenCrossSection::new(&ctx);
    let queries = vec![
        JanssenCrossSectionQuery::circular(2.0, 1500.0, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::hydraulic(-1.0, 1500.0, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::rectangular(2.0, 3.0, 1500.0, 9.81, 0.5, 0.5),
        JanssenCrossSectionQuery::hydraulic(1.2, 1500.0, 9.81, 0.0, 0.5),
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
    let gpu = GpuJanssenCrossSection::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenCrossSection::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Pick a shape, then draw every parameter from a strictly positive band
        // with margin so the validity decision stays stable under round-off.
        let shape = lcg.next_u32() % 3;
        let param0 = lcg.next_range(0.1, 5.0);
        let param1 = lcg.next_range(0.1, 5.0);
        let rho = lcg.next_range(500.0, 2500.0);
        let g = lcg.next_range(1.0, 20.0);
        let mu = lcg.next_range(0.1, 0.9);
        let k = lcg.next_range(0.1, 0.9);
        let q = match shape {
            0 => JanssenCrossSectionQuery::hydraulic(param0, rho, g, mu, k),
            1 => JanssenCrossSectionQuery::circular(param0, rho, g, mu, k),
            _ => JanssenCrossSectionQuery::rectangular(param0, param1, rho, g, mu, k),
        };
        queries.push(q);
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
