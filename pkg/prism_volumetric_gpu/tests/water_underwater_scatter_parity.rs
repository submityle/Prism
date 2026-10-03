//! Real-device parity for the underwater single/multiple-scatter twin:
//! [`GpuWaterUnderwaterScatter`](prism_volumetric_gpu::water_underwater_scatter::GpuWaterUnderwaterScatter)
//! must reproduce the three stateless numeric kernels of the `CPU` golden
//! [`underwater`](prism_render_architecture::water::underwater) — the
//! `Henyey-Greenstein` phase function
//! [`henyey_greenstein`](prism_render_architecture::water::underwater::henyey_greenstein),
//! the bounded
//! [`multiple_scatter_boost`](prism_render_architecture::water::underwater::multiple_scatter_boost),
//! and the god-ray
//! [`godray_inscatter`](prism_render_architecture::water::underwater::godray_inscatter)
//! — across hand fixtures, a mixed batch, and a randomized sweep compared
//! query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! All three golden functions are public and pure, so they are called directly
//! as the oracle: the expected [`WaterUnderwaterScatterResult`] is assembled
//! from `henyey_greenstein`, `multiple_scatter_boost` and `godray_inscatter`
//! evaluated on the very same query drivers. A `GPU == golden` pass is therefore
//! direct evidence the ported kernel computes the same scatter response the
//! reference does.
//!
//! # Parity criterion
//!
//! Every output is a continuous value threading through a divide, a `sqrt`
//! (phase) or the shared `exp_approx` squaring envelope (in-scatter), so a `GPU`
//! built-in may land a few units in the last place from the scalar reference.
//! All three scalars are asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`; there are no discrete outputs.
//!
//! # Conditioning
//!
//! Every fixture is kept clear of the lobe and boost singularities: the
//! asymmetry `g` stays within `[-0.9, 0.9]` (far from the `±0.999` clamp where
//! the lobe sharpens), the multiple-scatter `albedo` stays within `[0.0, 0.95]`
//! (far from the `0.999` clamp where `1 / (1 - albedo)` amplifies the last
//! place), and the product `extinction * path_length` stays moderate so the
//! `exp_approx` envelope's small divergence stays well under the tolerance.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::underwater`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::underwater::{
    godray_inscatter, henyey_greenstein, multiple_scatter_boost,
};
use prism_volumetric_gpu::water_underwater_scatter::{
    GpuWaterUnderwaterScatter, WaterUnderwaterScatterQuery, WaterUnderwaterScatterResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a scalar output. A `GPU` `sqrt`, divide or squaring
/// envelope may land a few units in the last place from the scalar reference;
/// `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Evaluates the three `CPU` golden closed forms on one query's drivers and
/// packs them into a [`WaterUnderwaterScatterResult`]: the faithful oracle the
/// `GPU` is pinned against.
fn oracle(q: &WaterUnderwaterScatterQuery) -> WaterUnderwaterScatterResult {
    WaterUnderwaterScatterResult {
        phase: henyey_greenstein(q.cos_theta, q.g),
        scatter_boost: multiple_scatter_boost(q.single, q.albedo),
        inscatter: godray_inscatter(
            q.surface_light,
            q.scatter_albedo,
            q.extinction,
            q.path_length,
        ),
    }
}

/// Pins one `GPU` result against the in-host oracle: all three scalar outputs
/// within tolerance.
fn check_one(idx: usize, got: &WaterUnderwaterScatterResult, want: &WaterUnderwaterScatterResult) {
    assert!(
        close(got.phase, want.phase),
        "query {idx} phase: gpu {} vs cpu {}",
        got.phase,
        want.phase
    );
    assert!(
        close(got.scatter_boost, want.scatter_boost),
        "query {idx} scatter_boost: gpu {} vs cpu {}",
        got.scatter_boost,
        want.scatter_boost
    );
    assert!(
        close(got.inscatter, want.inscatter),
        "query {idx} inscatter: gpu {} vs cpu {}",
        got.inscatter,
        want.inscatter
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterUnderwaterScatter,
    queries: &[WaterUnderwaterScatterQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_one(idx, result, &want);
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

/// Draws a value in `[0.0, 1.0)` at milli resolution from `state`.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) % 1000) as f32 / 1000.0
}

/// Draws a well-conditioned random query: every driver stays clear of the lobe
/// and boost singularities so the `CPU` and `GPU` stay in agreement.
fn random_query(state: &mut u64) -> WaterUnderwaterScatterQuery {
    // cos_theta in [-1, 1].
    let cos_theta = unit(state) * 2.0 - 1.0;
    // g in [-0.9, 0.9], far from the +-0.999 clamp.
    let g = unit(state) * 1.8 - 0.9;
    // single in [0, 2].
    let single = unit(state) * 2.0;
    // albedo in [0, 0.95], far from the 0.999 clamp.
    let albedo = unit(state) * 0.95;
    // surface_light in [0, 2].
    let surface_light = unit(state) * 2.0;
    // scatter_albedo in [0, 1].
    let scatter_albedo = unit(state);
    // extinction in [0, 2] and path_length in [0, 3], so the product stays
    // moderate and the exp_approx envelope's divergence stays tiny.
    let extinction = unit(state) * 2.0;
    let path_length = unit(state) * 3.0;
    WaterUnderwaterScatterQuery {
        cos_theta,
        g,
        single,
        albedo,
        surface_light,
        scatter_albedo,
        extinction,
        path_length,
    }
}

/// The deterministic hand fixtures: a spread of asymmetries, boosts and shafts,
/// all well clear of the singularities.
fn fixture_queries() -> Vec<WaterUnderwaterScatterQuery> {
    vec![
        // Isotropic lobe (g = 0), modest boost, short shaft.
        WaterUnderwaterScatterQuery {
            cos_theta: 0.3,
            g: 0.0,
            single: 1.0,
            albedo: 0.1,
            surface_light: 1.0,
            scatter_albedo: 0.5,
            extinction: 0.3,
            path_length: 1.0,
        },
        // Forward-peaked lobe, strong boost, long shaft.
        WaterUnderwaterScatterQuery {
            cos_theta: 1.0,
            g: 0.6,
            single: 1.0,
            albedo: 0.9,
            surface_light: 1.0,
            scatter_albedo: 0.5,
            extinction: 0.3,
            path_length: 3.0,
        },
        // Backward-peaked lobe, zero light shaft.
        WaterUnderwaterScatterQuery {
            cos_theta: -1.0,
            g: 0.6,
            single: 0.5,
            albedo: 0.4,
            surface_light: 0.0,
            scatter_albedo: 0.7,
            extinction: 0.5,
            path_length: 2.0,
        },
        // Negative asymmetry, zero extinction (transmittance one, inscatter nil).
        WaterUnderwaterScatterQuery {
            cos_theta: 0.2,
            g: -0.5,
            single: 1.5,
            albedo: 0.0,
            surface_light: 2.0,
            scatter_albedo: 1.0,
            extinction: 0.0,
            path_length: 2.5,
        },
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_underwater_scatter parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterUnderwaterScatter::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn isotropic_phase_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterScatter::new(&ctx);
    // g = 0 is the uniform lobe 1 / (4*PI); the GPU must land on the same value.
    let q = WaterUnderwaterScatterQuery {
        cos_theta: 0.3,
        g: 0.0,
        single: 1.0,
        albedo: 0.2,
        surface_light: 1.0,
        scatter_albedo: 0.5,
        extinction: 0.4,
        path_length: 1.5,
    };
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    check_one(0, &got[0], &want);
}

#[test]
fn forward_lobe_exceeds_backward_lobe() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterScatter::new(&ctx);
    // Positive g peaks forward: the cos_theta = 1 phase must exceed cos_theta =
    // -1, and both must match the golden.
    let forward = WaterUnderwaterScatterQuery {
        cos_theta: 1.0,
        g: 0.6,
        single: 1.0,
        albedo: 0.3,
        surface_light: 1.0,
        scatter_albedo: 0.5,
        extinction: 0.3,
        path_length: 1.0,
    };
    let backward = WaterUnderwaterScatterQuery {
        cos_theta: -1.0,
        ..forward
    };
    let got = gpu.evaluate(&ctx, &[forward, backward]);
    assert_eq!(got.len(), 2);
    assert!(
        got[0].phase > got[1].phase,
        "forward lobe {} must exceed backward lobe {}",
        got[0].phase,
        got[1].phase
    );
    check_one(0, &got[0], &oracle(&forward));
    check_one(1, &got[1], &oracle(&backward));
}

#[test]
fn boost_grows_with_albedo() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterScatter::new(&ctx);
    // Milky water (higher albedo) lifts the radiance more than clear water, and
    // both must match the golden.
    let clear = WaterUnderwaterScatterQuery {
        cos_theta: 0.0,
        g: 0.0,
        single: 1.0,
        albedo: 0.1,
        surface_light: 1.0,
        scatter_albedo: 0.5,
        extinction: 0.3,
        path_length: 1.0,
    };
    let milky = WaterUnderwaterScatterQuery {
        albedo: 0.9,
        ..clear
    };
    let got = gpu.evaluate(&ctx, &[clear, milky]);
    assert_eq!(got.len(), 2);
    assert!(
        got[1].scatter_boost > got[0].scatter_boost,
        "milky boost {} must exceed clear boost {}",
        got[1].scatter_boost,
        got[0].scatter_boost
    );
    check_one(0, &got[0], &oracle(&clear));
    check_one(1, &got[1], &oracle(&milky));
}

#[test]
fn godray_rises_with_path_length() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterScatter::new(&ctx);
    // A longer shaft scatters more toward the eye, and both must match golden.
    let short = WaterUnderwaterScatterQuery {
        cos_theta: 0.0,
        g: 0.0,
        single: 1.0,
        albedo: 0.2,
        surface_light: 1.0,
        scatter_albedo: 0.5,
        extinction: 0.3,
        path_length: 1.0,
    };
    let long = WaterUnderwaterScatterQuery {
        path_length: 5.0,
        ..short
    };
    let got = gpu.evaluate(&ctx, &[short, long]);
    assert_eq!(got.len(), 2);
    assert!(
        got[1].inscatter > got[0].inscatter,
        "long shaft inscatter {} must exceed short {}",
        got[1].inscatter,
        got[0].inscatter
    );
    check_one(0, &got[0], &oracle(&short));
    check_one(1, &got[1], &oracle(&long));
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterScatter::new(&ctx);
    // Every hand fixture dispatched together so the per-thread indexing and the
    // contiguous output slots are both exercised.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterUnderwaterScatter::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random well-conditioned queries pin every
    // output across a wide span of drivers.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
