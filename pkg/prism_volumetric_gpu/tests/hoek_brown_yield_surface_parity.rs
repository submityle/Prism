//! Real-device parity for the Hoek–Brown yield-surface twin:
//! [`GpuHoekBrownYieldSurface`](prism_volumetric_gpu::hoek_brown_yield_surface::GpuHoekBrownYieldSurface)
//! must reproduce the `CPU` golden `HoekBrownModel` bracket/yield helpers of
//! `prism_physics_core::collider::tet_fem_hoek_brown_plasticity`. For one model
//! `(σ_ci, m_b, s, a, m_g)` and one principal-stress pair `(σ₁, σ₃)` (tension
//! positive) the surface quantities are
//!
//! * `raw_bracket = s − m_b·σ₁/σ_ci`,
//! * `bracket = max(raw_bracket, 1e-9)`,
//! * `yield_value = (σ₁ − σ₃) − σ_ci·bracket^a`,
//! * `dyield_dmajor = 1 + a·m_b·bracket^(a−1)`,
//! * `bracket_g = max(s − m_g·σ₁/σ_ci, 1e-9)`,
//! * `flow_major = a·m_g·bracket_g^(a−1)`,
//!
//! and the query is valid only when the model lies in the golden
//! `HoekBrownModel::new` range.
//!
//! The oracle here is an independent re-implementation of those closed forms —
//! the range checks, then the clamped brackets and powers in the golden
//! operator order — written out directly in `f64` (mirroring the golden) and
//! cast to `f32`, so the test never imports `prism_render_architecture` or
//! `prism_physics_core`.
//!
//! The fixtures cover a typical yield-surface point, a near-apex point that
//! trips the `MIN_BRACKET` clamp, several out-of-range / non-finite models
//! (invalid), a batch mixing valid and invalid rows to validate the `std430`
//! stride, and an empty batch the host short-circuits. A sweep over random
//! in-range models follows, keeping `raw_bracket` comfortably above the clamp
//! knee.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! The golden evaluates the fractional powers in `f64`; the kernel uses `f32`
//! `pow`, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact. Each continuous output is compared with `abs <= 1e-4 ||
//! rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared
//! exactly. The fixtures keep the model well inside its valid range and
//! `raw_bracket` well above `MIN_BRACKET` so neither the validity decision nor
//! the clamp knee can be flipped by round-off.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_fem_hoek_brown_plasticity`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hoek_brown_yield_surface::{
    GpuHoekBrownYieldSurface, HoekBrownYieldSurfaceQuery, HoekBrownYieldSurfaceResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// The golden confinement-bracket clamp floor.
const MIN_BRACKET: f64 = 1.0e-9;

/// Independent host oracle: reproduces the `HoekBrownModel` bracket/yield
/// helpers in `f64` (mirroring the golden), returning the six continuous
/// outputs and the validity flag.
fn oracle(q: &HoekBrownYieldSurfaceQuery) -> ([f32; 6], u32) {
    let sci = q.sigma_ci;
    let mb = q.m_b;
    let s = q.s;
    let a = q.a;
    let mg = q.m_g;
    // Range check mirrors HoekBrownModel::new (minus the hardening term, which
    // this surface twin does not use).
    let in_range = sci.is_finite()
        && sci > 0.0
        && mb.is_finite()
        && mb > 0.0
        && s.is_finite()
        && s > 0.0
        && s <= 1.0
        && a.is_finite()
        && a > 0.0
        && a <= 1.0
        && mg.is_finite()
        && mg >= 0.0
        && mg <= mb;
    if !in_range {
        return ([0.0; 6], 0);
    }

    let sci = f64::from(sci);
    let mb = f64::from(mb);
    let s = f64::from(s);
    let a = f64::from(a);
    let mg = f64::from(mg);
    let s1 = f64::from(q.sigma1);
    let s3 = f64::from(q.sigma3);

    let raw = s - mb * s1 / sci;
    let bracket = raw.max(MIN_BRACKET);
    let raw_g = s - mg * s1 / sci;
    let bracket_g = raw_g.max(MIN_BRACKET);

    let yv = (s1 - s3) - sci * bracket.powf(a);
    let dyv = 1.0 + a * mb * bracket.powf(a - 1.0);
    let fm = a * mg * bracket_g.powf(a - 1.0);

    (
        [
            bracket as f32,
            raw as f32,
            yv as f32,
            dyv as f32,
            bracket_g as f32,
            fm as f32,
        ],
        1,
    )
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
/// `valid` flag exactly, and every continuous output to tolerance when valid.
fn assert_parity(gpu: &HoekBrownYieldSurfaceResult, q: &HoekBrownYieldSurfaceQuery, label: &str) {
    let (expected, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 1 {
        let got = [
            gpu.bracket,
            gpu.raw_bracket,
            gpu.yield_value,
            gpu.dyield_dmajor,
            gpu.bracket_g,
            gpu.flow_major,
        ];
        let names = [
            "bracket",
            "raw_bracket",
            "yield_value",
            "dyield_dmajor",
            "bracket_g",
            "flow_major",
        ];
        for ((g, e), name) in got.iter().zip(expected.iter()).zip(names.iter()) {
            assert!(close(*g, *e), "{label}: {name} mismatch gpu={g} oracle={e}");
        }
    } else {
        // Invalid rows must be all-zero continuous outputs.
        assert_eq!(gpu.bracket, 0.0, "{label}: invalid bracket not zero");
        assert_eq!(
            gpu.raw_bracket, 0.0,
            "{label}: invalid raw_bracket not zero"
        );
        assert_eq!(
            gpu.yield_value, 0.0,
            "{label}: invalid yield_value not zero"
        );
        assert_eq!(
            gpu.dyield_dmajor, 0.0,
            "{label}: invalid dyield_dmajor not zero"
        );
        assert_eq!(gpu.bracket_g, 0.0, "{label}: invalid bracket_g not zero");
        assert_eq!(gpu.flow_major, 0.0, "{label}: invalid flow_major not zero");
    }
}

#[test]
fn typical_yield_surface_point_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHoekBrownYieldSurface::new(&ctx);
    // Original criterion (s = 1, a = 0.5) with associated flow (m_g = m_b),
    // moderate confinement well away from the apex.
    let q = HoekBrownYieldSurfaceQuery::new(50.0, 8.0, 1.0, 0.5, 8.0, -10.0, -30.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "typical");
}

#[test]
fn generalized_exponent_point_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHoekBrownYieldSurface::new(&ctx);
    // Generalized criterion with a != 0.5, reduced dilation (m_g < m_b).
    let q = HoekBrownYieldSurfaceQuery::new(80.0, 5.0, 0.6, 0.55, 2.5, -5.0, -40.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert_parity(&out[0], &q, "generalized");
}

#[test]
fn near_apex_point_trips_the_bracket_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHoekBrownYieldSurface::new(&ctx);
    // At the tensile apex sigma1 = s*sci/m_b the raw bracket is zero; push just
    // past it so raw_bracket < 0 while the clamped bracket pins to MIN_BRACKET.
    let sci = 40.0_f32;
    let mb = 10.0_f32;
    let s = 1.0_f32;
    let apex = s * sci / mb; // raw_bracket == 0 here.
    let sigma1 = apex + 1.0; // raw_bracket < 0, clamp engages.
    let q = HoekBrownYieldSurfaceQuery::new(sci, mb, s, 0.5, mb, sigma1, sigma1 - 5.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    // raw_bracket should be strictly negative while bracket is the floor.
    assert!(
        out[0].raw_bracket < 0.0,
        "raw_bracket should be negative past apex"
    );
    assert_parity(&out[0], &q, "near_apex");
}

#[test]
fn out_of_range_models_are_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHoekBrownYieldSurface::new(&ctx);
    let queries = vec![
        // sigma_ci <= 0.
        HoekBrownYieldSurfaceQuery::new(0.0, 8.0, 1.0, 0.5, 8.0, -10.0, -30.0),
        // m_b <= 0.
        HoekBrownYieldSurfaceQuery::new(50.0, -1.0, 1.0, 0.5, 0.0, -10.0, -30.0),
        // s out of (0, 1].
        HoekBrownYieldSurfaceQuery::new(50.0, 8.0, 1.5, 0.5, 8.0, -10.0, -30.0),
        // a out of (0, 1].
        HoekBrownYieldSurfaceQuery::new(50.0, 8.0, 1.0, 0.0, 8.0, -10.0, -30.0),
        // m_g > m_b.
        HoekBrownYieldSurfaceQuery::new(50.0, 8.0, 1.0, 0.5, 20.0, -10.0, -30.0),
        // Non-finite sigma_ci.
        HoekBrownYieldSurfaceQuery::new(f32::NAN, 8.0, 1.0, 0.5, 8.0, -10.0, -30.0),
        // Non-finite m_b.
        HoekBrownYieldSurfaceQuery::new(50.0, f32::INFINITY, 1.0, 0.5, 8.0, -10.0, -30.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "row {i} should be invalid");
        assert_parity(res, q, &format!("invalid[{i}]"));
    }
}

#[test]
fn batch_mixes_valid_and_invalid_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHoekBrownYieldSurface::new(&ctx);
    let queries = vec![
        HoekBrownYieldSurfaceQuery::new(50.0, 8.0, 1.0, 0.5, 8.0, -10.0, -30.0),
        HoekBrownYieldSurfaceQuery::new(0.0, 8.0, 1.0, 0.5, 8.0, -10.0, -30.0),
        HoekBrownYieldSurfaceQuery::new(80.0, 5.0, 0.6, 0.55, 2.5, -5.0, -40.0),
        HoekBrownYieldSurfaceQuery::new(50.0, 8.0, 1.0, 1.5, 8.0, -10.0, -30.0),
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
    let gpu = GpuHoekBrownYieldSurface::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHoekBrownYieldSurface::new(&ctx);
    let mut lcg = Lcg::new(0x48B5_0E21);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let sigma_ci = lcg.next_range(20.0, 120.0);
        let m_b = lcg.next_range(2.0, 15.0);
        // s and a comfortably inside (0, 1].
        let s = lcg.next_range(0.3, 1.0);
        let a = lcg.next_range(0.4, 0.6);
        // m_g in [0, m_b].
        let m_g = lcg.next_range(0.0, m_b);
        // Keep raw_bracket = s - m_b*sigma1/sigma_ci comfortably above the
        // MIN_BRACKET knee by choosing sigma1 so the bracket stays in a safe
        // band [0.1, s]. raw_bracket = target => sigma1 = (s - target)*sci/m_b.
        let target = lcg.next_range(0.1, s * 0.9);
        let sigma1 = (s - target) * sigma_ci / m_b;
        // sigma3 below sigma1 (tension positive ordering sigma1 >= sigma3).
        let sigma3 = sigma1 - lcg.next_range(1.0, 60.0);
        queries.push(HoekBrownYieldSurfaceQuery::new(
            sigma_ci, m_b, s, a, m_g, sigma1, sigma3,
        ));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be valid");
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
