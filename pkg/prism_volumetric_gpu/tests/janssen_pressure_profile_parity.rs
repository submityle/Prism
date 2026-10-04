//! Real-device parity for the Janssen pressure-profile twin:
//! [`GpuJanssenPressureProfile`](prism_volumetric_gpu::janssen_pressure_profile::GpuJanssenPressureProfile)
//! must reproduce the `CPU` golden `prism_physics_core::collider::janssen_pressure`'s
//! `JanssenProfile`.
//!
//! From the hydraulic radius `R` and the material parameters (`ρ`, `g`, `μ_w`,
//! `K`) the reference derives the characteristic depth `z_c = R / (μ_w K)`, the
//! saturation stresses `σ_∞ = ρ g z_c`, `σ_h∞ = K σ_∞`, `τ_w∞ = μ_w σ_h∞`, and
//! for one depth `z` the stresses `σ_v(z) = σ_∞ (1 − exp(−z / z_c))`,
//! `σ_h(z) = K σ_v(z)`, `τ_w(z) = μ_w σ_h(z)` plus the screening fraction
//! `clamp(1 − σ_v(z) / (ρ g z), 0, 1)`. The profile is valid only when every
//! material parameter is finite and positive and the derived `z_c` and `σ_∞`
//! are finite and positive.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`
//! or `prism_physics_core`. The reference evaluates the saturation factor
//! `1 − exp(−z / z_c)` in `f64`; the oracle mirrors that exactly and the
//! tolerance absorbs the `f32`-device versus `f64`-host difference.
//!
//! The fixtures cover sane configurations across depths, the surface and
//! small-depth regimes, the deep saturation regime, degenerate rejections
//! (non-finite or non-positive material parameter, radius or `z_c`), a valid
//! profile with a non-positive depth (saturation stresses non-zero but the
//! depth-dependent stresses and screening zero), a mixed batch that validates
//! the `std430` stride, and an empty batch the host short-circuits. A sweep
//! over random parameters well inside the valid band follows.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Each valid scalar is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly. The
//! sweep keeps `μ_w · K` away from zero and every material parameter well
//! inside the valid band so the validity decision and `z_c` agree on both
//! sides.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::janssen_pressure`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::janssen_pressure_profile::{
    GpuJanssenPressureProfile, JanssenPressureProfileQuery, JanssenPressureProfileResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// The eight continuous outputs plus the discrete validity flag of the oracle.
struct Oracle {
    characteristic_depth: f32,
    saturation_vertical: f32,
    saturation_horizontal: f32,
    saturation_wall_shear: f32,
    vertical_stress: f32,
    horizontal_stress: f32,
    wall_shear_stress: f32,
    screening_fraction: f32,
    valid: bool,
}

/// Independent host oracle: reproduces `JanssenProfile` in pure `f32` (with the
/// `f64` saturation factor the reference uses), returning the eight continuous
/// quantities and the validity flag.
fn oracle(q: &JanssenPressureProfileQuery) -> Oracle {
    let zero = Oracle {
        characteristic_depth: 0.0,
        saturation_vertical: 0.0,
        saturation_horizontal: 0.0,
        saturation_wall_shear: 0.0,
        vertical_stress: 0.0,
        horizontal_stress: 0.0,
        wall_shear_stress: 0.0,
        screening_fraction: 0.0,
        valid: false,
    };
    let mat_ok = [q.bulk_density, q.gravity, q.wall_friction, q.k_ratio]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0);
    if !mat_ok {
        return zero;
    }
    let z_c = q.hydraulic_radius / (q.wall_friction * q.k_ratio);
    if !z_c.is_finite() || z_c <= 0.0 {
        return zero;
    }
    let sat_v = q.bulk_density * q.gravity * z_c;
    if !sat_v.is_finite() {
        return zero;
    }
    let sat_h = q.k_ratio * sat_v;
    let sat_shear = q.wall_friction * sat_h;

    // Depth-dependent vertical stress, with the f64 saturation factor matching
    // the reference exactly.
    let depth = q.depth;
    let sigma_v = if depth.is_finite() && depth > 0.0 {
        let ratio = f64::from(depth / z_c);
        let factor = 1.0 - (-ratio).exp();
        sat_v * factor as f32
    } else {
        0.0
    };
    let sigma_h = q.k_ratio * sigma_v;
    let tau_w = q.wall_friction * sigma_h;

    // Screening fraction against the frictionless hydrostatic stress.
    let screening = if depth.is_finite() && depth > 0.0 {
        let hydrostatic = q.bulk_density * q.gravity * depth;
        if hydrostatic <= 0.0 {
            0.0
        } else {
            (1.0 - sigma_v / hydrostatic).clamp(0.0, 1.0)
        }
    } else {
        0.0
    };

    Oracle {
        characteristic_depth: z_c,
        saturation_vertical: sat_v,
        saturation_horizontal: sat_h,
        saturation_wall_shear: sat_shear,
        vertical_stress: sigma_v,
        horizontal_stress: sigma_h,
        wall_shear_stress: tau_w,
        screening_fraction: screening,
        valid: true,
    }
}

/// Absolute-or-relative closeness: `abs <= 1e-4 || rel <= 1e-3` with a relative
/// floor so the relative test never divides by a magnitude below `REL_FLOOR`.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts one `GPU` result matches the independent oracle: the discrete
/// `valid` flag exactly, the eight continuous scalars to tolerance when valid,
/// else all zero.
fn assert_parity(q: &JanssenPressureProfileQuery, got: &JanssenPressureProfileResult) {
    let want = oracle(q);
    if want.valid {
        assert_eq!(got.valid, 1, "query {q:?} should be valid");
        assert!(
            close(got.characteristic_depth, want.characteristic_depth),
            "z_c mismatch for {q:?}: gpu={} oracle={}",
            got.characteristic_depth,
            want.characteristic_depth
        );
        assert!(
            close(got.saturation_vertical, want.saturation_vertical),
            "sigma_inf mismatch for {q:?}: gpu={} oracle={}",
            got.saturation_vertical,
            want.saturation_vertical
        );
        assert!(
            close(got.saturation_horizontal, want.saturation_horizontal),
            "sigma_h_inf mismatch for {q:?}: gpu={} oracle={}",
            got.saturation_horizontal,
            want.saturation_horizontal
        );
        assert!(
            close(got.saturation_wall_shear, want.saturation_wall_shear),
            "tau_w_inf mismatch for {q:?}: gpu={} oracle={}",
            got.saturation_wall_shear,
            want.saturation_wall_shear
        );
        assert!(
            close(got.vertical_stress, want.vertical_stress),
            "sigma_v mismatch for {q:?}: gpu={} oracle={}",
            got.vertical_stress,
            want.vertical_stress
        );
        assert!(
            close(got.horizontal_stress, want.horizontal_stress),
            "sigma_h mismatch for {q:?}: gpu={} oracle={}",
            got.horizontal_stress,
            want.horizontal_stress
        );
        assert!(
            close(got.wall_shear_stress, want.wall_shear_stress),
            "tau_w mismatch for {q:?}: gpu={} oracle={}",
            got.wall_shear_stress,
            want.wall_shear_stress
        );
        assert!(
            close(got.screening_fraction, want.screening_fraction),
            "screening mismatch for {q:?}: gpu={} oracle={}",
            got.screening_fraction,
            want.screening_fraction
        );
    } else {
        assert_eq!(got.valid, 0, "query {q:?} should be invalid");
        assert_eq!(got.characteristic_depth, 0.0);
        assert_eq!(got.saturation_vertical, 0.0);
        assert_eq!(got.saturation_horizontal, 0.0);
        assert_eq!(got.saturation_wall_shear, 0.0);
        assert_eq!(got.vertical_stress, 0.0);
        assert_eq!(got.horizontal_stress, 0.0);
        assert_eq!(got.wall_shear_stress, 0.0);
        assert_eq!(got.screening_fraction, 0.0);
    }
}

#[test]
fn saturation_constants_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    // R = 0.5, mu = 0.5, K = 0.5 => z_c = 2; sigma_inf = 1000 * 10 * 2 = 20000.
    let q = JanssenPressureProfileQuery::new(0.5, 1000.0, 10.0, 0.5, 0.5, 2.0);
    let got = twin.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    assert_parity(&q, &got[0]);
    // Spot-check the derived constants explicitly.
    assert!(close(got[0].characteristic_depth, 2.0));
    assert!(close(got[0].saturation_vertical, 20_000.0));
    assert!(close(got[0].saturation_horizontal, 10_000.0));
    assert!(close(got[0].saturation_wall_shear, 5_000.0));
}

#[test]
fn vertical_stress_hits_63_percent_at_characteristic_depth() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    // At z = z_c the vertical stress is sigma_inf * (1 - 1/e) ~= 63.2%.
    let q = JanssenPressureProfileQuery::new(0.5, 1000.0, 10.0, 0.5, 0.5, 2.0);
    let got = twin.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    assert_parity(&q, &got[0]);
    let expected = got[0].saturation_vertical * 0.632_120_6;
    assert!(close(got[0].vertical_stress, expected));
}

#[test]
fn small_depth_is_near_hydrostatic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    // z << z_c: sigma_v ~= rho g z, screening ~= 0.
    let q = JanssenPressureProfileQuery::new(0.5, 1000.0, 10.0, 0.5, 0.5, 0.01);
    let got = twin.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    assert_parity(&q, &got[0]);
}

#[test]
fn deep_depth_approaches_saturation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    // z >> z_c: sigma_v -> sigma_inf, screening -> 1.
    let q = JanssenPressureProfileQuery::new(0.5, 1500.0, 9.81, 0.45, 0.5, 40.0);
    let got = twin.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    assert_parity(&q, &got[0]);
    assert!(got[0].vertical_stress <= got[0].saturation_vertical + 1.0);
}

#[test]
fn valid_profile_with_nonpositive_depth_zeroes_depth_terms() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    // Negative depth: profile constructs (saturation stresses non-zero), but the
    // depth-dependent stresses and screening are zero.
    let q = JanssenPressureProfileQuery::new(0.5, 1200.0, 9.81, 0.4, 0.5, -3.0);
    let got = twin.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    assert_parity(&q, &got[0]);
    assert_eq!(got[0].valid, 1);
    assert!(got[0].saturation_vertical > 0.0);
    assert_eq!(got[0].vertical_stress, 0.0);
    assert_eq!(got[0].screening_fraction, 0.0);
}

#[test]
fn zero_depth_zeroes_depth_terms() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    let q = JanssenPressureProfileQuery::new(0.75, 1000.0, 9.81, 0.5, 0.5, 0.0);
    let got = twin.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    assert_parity(&q, &got[0]);
    assert_eq!(got[0].valid, 1);
    assert_eq!(got[0].vertical_stress, 0.0);
}

#[test]
fn degenerate_material_parameters_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    let queries = [
        // bulk_density <= 0.
        JanssenPressureProfileQuery::new(0.5, 0.0, 9.81, 0.5, 0.5, 2.0),
        // gravity negative.
        JanssenPressureProfileQuery::new(0.5, 1000.0, -9.81, 0.5, 0.5, 2.0),
        // wall friction zero.
        JanssenPressureProfileQuery::new(0.5, 1000.0, 9.81, 0.0, 0.5, 2.0),
        // k_ratio NaN.
        JanssenPressureProfileQuery::new(0.5, 1000.0, 9.81, 0.5, f32::NAN, 2.0),
        // gravity infinite.
        JanssenPressureProfileQuery::new(0.5, 1000.0, f32::INFINITY, 0.5, 0.5, 2.0),
        // radius <= 0 => z_c <= 0.
        JanssenPressureProfileQuery::new(0.0, 1000.0, 9.81, 0.5, 0.5, 2.0),
        // radius negative => z_c < 0.
        JanssenPressureProfileQuery::new(-0.5, 1000.0, 9.81, 0.5, 0.5, 2.0),
        // radius non-finite => z_c non-finite.
        JanssenPressureProfileQuery::new(f32::INFINITY, 1000.0, 9.81, 0.5, 0.5, 2.0),
    ];
    let got = twin.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (q, r) in queries.iter().zip(got.iter()) {
        assert_parity(q, r);
        assert_eq!(r.valid, 0);
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    let queries = [
        JanssenPressureProfileQuery::new(0.5, 1000.0, 10.0, 0.5, 0.5, 2.0),
        JanssenPressureProfileQuery::new(0.0, 1000.0, 9.81, 0.5, 0.5, 2.0),
        JanssenPressureProfileQuery::new(1.2, 1500.0, 9.81, 0.45, 0.6, 15.0),
        JanssenPressureProfileQuery::new(0.3, 800.0, 9.81, 0.6, 0.4, 0.5),
        JanssenPressureProfileQuery::new(0.9, 2000.0, 9.81, 0.3, 0.7, -1.0),
    ];
    let got = twin.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (q, r) in queries.iter().zip(got.iter()) {
        assert_parity(q, r);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    let got = twin.evaluate(&ctx, &[]);
    assert!(got.is_empty());
}

#[test]
fn random_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let twin = GpuJanssenPressureProfile::new(&ctx);
    let mut lcg = Lcg::new(0x5A17_C0DE);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // All strictly positive and well inside the valid band so the master
        // validity decision cannot be flipped by round-off. mu*K stays away
        // from zero so z_c does not blow up; depth spans sub-saturation to
        // saturated without gluing everything to the surface.
        let radius = lcg.next_range(0.1, 5.0);
        let rho = lcg.next_range(500.0, 2500.0);
        let g = lcg.next_range(1.0, 20.0);
        let mu = lcg.next_range(0.2, 0.8);
        let k = lcg.next_range(0.3, 0.7);
        let depth = lcg.next_range(0.05, 50.0);
        queries.push(JanssenPressureProfileQuery::new(
            radius, rho, g, mu, k, depth,
        ));
    }
    let got = twin.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len());
    for (q, r) in queries.iter().zip(got.iter()) {
        assert_parity(q, r);
    }
}

/// A small linear-congruential generator, keeping the sweep deterministic and
/// free of transcendental host calls.
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
