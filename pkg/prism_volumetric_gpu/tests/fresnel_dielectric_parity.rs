//! Real-device parity for the smooth-dielectric Fresnel reflectance twin:
//! [`GpuFresnelDielectric`](prism_volumetric_gpu::fresnel_dielectric::GpuFresnelDielectric)
//! must reproduce the unpolarized reflectance of the `CPU` golden
//! [`fresnel_dielectric`](prism_render_architecture::reference_pt::dielectric::fresnel_dielectric)
//! across normal incidence, grazing angles, the total internal reflection
//! (`TIR`) regime, index-matched interfaces, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected reflectance comes straight from the public golden
//! [`fresnel_dielectric`](prism_render_architecture::reference_pt::dielectric::fresnel_dielectric).
//! A `GPU` parity pass is therefore direct evidence the ported kernel reflects
//! identically to the reference.
//!
//! # Parity criterion
//!
//! The reflectance is a *continuous* quantity, so each assertion compares with
//! an absolute-or-relative tolerance (`abs <= 1e-5 || rel <= 1e-4`, relative
//! floor `1e-6`). The `TIR` branch (reflectance exactly `1`) folds into that
//! continuous output; the randomized sweep rejects samples within a small
//! margin of the critical angle so the discrete branch cannot tie-flip between
//! the `CPU` and `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::dielectric::fresnel_dielectric`；无第三方引擎源码或衍生代码。

use prism_render_architecture::reference_pt::dielectric::fresnel_dielectric;
use prism_volumetric_gpu::fresnel_dielectric::{
    FresnelDielectricQuery, FresnelDielectricResult, GpuFresnelDielectric,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous reflectance comparison.
const EPS: f32 = 1e-5;
/// Relative tolerance for the continuous reflectance comparison.
const REL: f32 = 1e-4;
/// Relative-tolerance floor, so near-zero magnitudes do not inflate the ratio.
const REL_FLOOR: f32 = 1e-6;

/// Returns whether `a` and `b` agree within the absolute-or-relative tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL
}

/// Computes the expected reflectance from the public golden.
fn oracle(q: &FresnelDielectricQuery) -> f32 {
    fresnel_dielectric(q.cos_i, q.eta_i, q.eta_t)
}

/// Dispatches every query and pins each `GPU` reflectance against the golden
/// oracle with the absolute-or-relative tolerance.
fn check(ctx: &GpuContext, gpu: &GpuFresnelDielectric, queries: &[FresnelDielectricQuery]) {
    let got: Vec<FresnelDielectricResult> = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert!(
            close(result.reflectance, want),
            "query {idx} reflectance: gpu {} vs cpu {} \
             (cos_i {}, eta_i {}, eta_t {})",
            result.reflectance,
            want,
            q.cos_i,
            q.eta_i,
            q.eta_t,
        );
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at milli resolution from `state`, using only
/// integer arithmetic so no transcendental method appears.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    let step = lcg(state) % (span + 1);
    lo + step as f32 / 1000.0
}

/// The deterministic edge fixtures exercising normal incidence, grazing angles,
/// the `TIR` regime, the index-matched interface, and both `eta > 1` and
/// `eta < 1` sides. Each sample sits well clear of the critical angle so the
/// `TIR` branch resolves identically on both sides.
fn edge_fixtures() -> Vec<FresnelDielectricQuery> {
    vec![
        // Normal incidence air -> glass: R ≈ 0.04.
        FresnelDielectricQuery::new(1.0, 1.0, 1.5),
        // Normal incidence air -> water.
        FresnelDielectricQuery::new(1.0, 1.0, 1.33),
        // Normal incidence glass -> air (eta < 1 side), still R ≈ 0.04.
        FresnelDielectricQuery::new(1.0, 1.5, 1.0),
        // Grazing incidence (cos_i = 0): R = 1 for any index pair.
        FresnelDielectricQuery::new(0.0, 1.0, 1.5),
        // Near-grazing but transmitting, air -> glass.
        FresnelDielectricQuery::new(0.1, 1.0, 1.5),
        // Mid-angle air -> glass.
        FresnelDielectricQuery::new(0.5, 1.0, 1.5),
        // Index-matched interface at normal incidence: R = 0.
        FresnelDielectricQuery::new(1.0, 1.4, 1.4),
        // Index-matched interface at an oblique angle: still R = 0.
        FresnelDielectricQuery::new(0.6, 1.4, 1.4),
        // TIR, glass -> air past the critical angle: shallow cos drives
        // sin2_t >= 1, so R = 1.
        FresnelDielectricQuery::new(0.2, 1.5, 1.0),
        // TIR, water -> air past the critical angle.
        FresnelDielectricQuery::new(0.3, 1.33, 1.0),
        // High-index pair, oblique.
        FresnelDielectricQuery::new(0.7, 1.0, 2.4),
        // Dense-to-less-dense, transmitting (just inside the critical angle).
        FresnelDielectricQuery::new(0.95, 1.5, 1.0),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fresnel_dielectric parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuFresnelDielectric::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn normal_incidence_air_glass() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelDielectric::new(&ctx);
    let q = FresnelDielectricQuery::new(1.0, 1.0, 1.5);
    // Classic Fresnel reflectance at normal incidence is ≈ 0.04.
    assert!(
        close(oracle(&q), 0.04),
        "fixture must reflect ~4% at normal"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn grazing_incidence_reflects_fully() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelDielectric::new(&ctx);
    let q = FresnelDielectricQuery::new(0.0, 1.0, 1.5);
    assert!(close(oracle(&q), 1.0), "grazing incidence reflects fully");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn total_internal_reflection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelDielectric::new(&ctx);
    // glass -> air at a shallow angle past the critical angle: R = 1.
    let q = FresnelDielectricQuery::new(0.2, 1.5, 1.0);
    assert!(close(oracle(&q), 1.0), "fixture must be in the TIR regime");
    check(&ctx, &gpu, &[q]);
}

#[test]
fn index_matched_interface_is_transparent() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelDielectric::new(&ctx);
    let q = FresnelDielectricQuery::new(1.0, 1.4, 1.4);
    assert!(
        close(oracle(&q), 0.0),
        "index-matched normal incidence R = 0"
    );
    check(&ctx, &gpu, &[q]);
}

#[test]
fn edge_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelDielectric::new(&ctx);
    check(&ctx, &gpu, &edge_fixtures());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFresnelDielectric::new(&ctx);
    let mut state: u64 = 0x1337_C0DE_F00D_BEEF;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let cos_i = uniform(&mut state, 0.0, 1.0);
        let eta_i = uniform(&mut state, 1.0, 2.5);
        let eta_t = uniform(&mut state, 1.0, 2.5);
        // Compute sin2_t exactly as the kernel does and reject samples within a
        // margin of the critical angle, so the discrete TIR branch cannot
        // tie-flip between the CPU and GPU.
        let cos_c = cos_i.clamp(0.0, 1.0);
        let eta = eta_i / eta_t;
        let sin2_i = (1.0 - cos_c * cos_c).max(0.0);
        let sin2_t = eta * eta * sin2_i;
        if (sin2_t - 1.0).abs() < 0.02 {
            continue;
        }
        queries.push(FresnelDielectricQuery::new(cos_i, eta_i, eta_t));
    }
    check(&ctx, &gpu, &queries);
}
