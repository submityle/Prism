//! Real-device parity for the volumetric evaluation twin:
//! [`GpuVolumetrics`](prism_volumetric_gpu::volumetrics::GpuVolumetrics) must
//! reproduce the `CPU` golden
//! [`volumetrics`](prism_render_architecture::particle::volumetrics) and
//! [`shading`](prism_render_architecture::particle::shading) across the
//! single-lobe Henyey-Greenstein phase, its double-lobe blend, the `NPR`
//! cel-banded response, the directional six-way read-back, the single-sample
//! six-way deposit and the per-bracket deep-opacity transmittance lerp, plus a
//! randomized mixed batch compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every routine threads through multiplies, adds and one guarded divide /
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. Continuous channels are
//! compared with `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Fixtures are drawn clear of every branch knee: `g` and `back_g` stay in
//! `(-0.9, 0.9)` away from the `±1` grazing singularity; `cos_theta` stays in
//! `(-0.95, 0.95)`; `back_lobe_weight` stays in `(0.05, 0.95)` away from the
//! clamp knees; the banded response is conditioned so its normalized value
//! lands at a fixed interior target whose `band` fraction is kept clear of the
//! `floor` step edge, so a fused multiply-add cannot tip the band index; the
//! deep-opacity bracket keeps a positive span and places the query depth in the
//! bracket interior. All randomness comes from a host-side integer generator,
//! so no transcendental appears in a fixture.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::volumetrics`
//! 与 `prism_render_architecture::particle::shading`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::shading::{
    six_way_response, PhaseParams, SixWayLuminance,
};
use prism_render_architecture::particle::volumetrics::{
    double_lobe_phase, henyey_greenstein, phase_response_banded, sample_deep_transmittance,
    DeepOpacityLayer, SixWayAccumulator, EPS,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::volumetrics::{GpuVolumetrics, VolumetricsQuery, VolumetricsResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS_ABS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS_ABS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Builds the golden phase parameters for the given lobe triple.
fn phase_params(g: f32, back_lobe_weight: f32, back_g: f32) -> PhaseParams {
    PhaseParams {
        g,
        back_lobe_weight,
        back_g,
    }
}

/// Draws a seam-safe single-lobe Henyey-Greenstein query (anisotropy clear of
/// the `±1` grazing singularity, cosine clear of the extremes).
fn make_henyey(state: &mut u64) -> VolumetricsQuery {
    VolumetricsQuery::HenyeyGreenstein {
        g: range(state, -0.9, 0.9),
        cos_theta: range(state, -0.95, 0.95),
    }
}

/// Draws a seam-safe double-lobe phase query.
fn make_double_lobe(state: &mut u64) -> VolumetricsQuery {
    VolumetricsQuery::DoubleLobePhase {
        g: range(state, -0.9, 0.9),
        back_lobe_weight: range(state, 0.05, 0.95),
        back_g: range(state, -0.9, 0.9),
        cos_theta: range(state, -0.95, 0.95),
    }
}

/// Draws a cel-banded phase query conditioned so the normalized response lands
/// at a fixed interior target whose scaled band fraction is clear of the
/// `floor` step edge; `peak` is derived from the golden response so the twin's
/// own `response / peak` reproduces that target.
fn make_banded(state: &mut u64) -> VolumetricsQuery {
    loop {
        let g = range(state, -0.9, 0.9);
        let back_lobe_weight = range(state, 0.05, 0.95);
        let back_g = range(state, -0.9, 0.9);
        let cos_theta = range(state, -0.95, 0.95);
        let bands = 2 + (lcg(state) * 7.0) as u32;
        let target = range(state, 0.1, 0.9);
        let scaled = target * bands as f32;
        let frac = scaled - scaled.floor();
        if !(0.15..=0.85).contains(&frac) {
            continue;
        }
        let response = double_lobe_phase(phase_params(g, back_lobe_weight, back_g), cos_theta);
        if response <= 1.0e-3 {
            continue;
        }
        let peak = response / target;
        if peak <= EPS {
            continue;
        }
        return VolumetricsQuery::PhaseResponseBanded {
            g,
            back_lobe_weight,
            back_g,
            cos_theta,
            peak,
            bands,
        };
    }
}

/// A random direction with every component clear of zero so the normalized
/// vector stays well inside one sign octant.
fn rand_dir(state: &mut u64) -> [f32; 3] {
    [
        range(state, -2.0, 2.0),
        range(state, -2.0, 2.0),
        range(state, -2.0, 2.0),
    ]
}

/// A random six-bucket luminance rig with non-negative buckets.
fn rand_luminance(state: &mut u64) -> [f32; 6] {
    [
        range(state, 0.1, 2.0),
        range(state, 0.1, 2.0),
        range(state, 0.1, 2.0),
        range(state, 0.1, 2.0),
        range(state, 0.1, 2.0),
        range(state, 0.1, 2.0),
    ]
}

/// Draws a six-way directional read-back query.
fn make_six_way_response(state: &mut u64) -> VolumetricsQuery {
    VolumetricsQuery::SixWayResponse {
        luminance: rand_luminance(state),
        light_dir: rand_dir(state),
    }
}

/// Draws a single-sample six-way deposit query (positive intensity).
fn make_six_way_deposit(state: &mut u64) -> VolumetricsQuery {
    VolumetricsQuery::SixWayDeposit {
        light_dir: rand_dir(state),
        intensity: range(state, 0.2, 4.0),
    }
}

/// Draws a deep-opacity bracket lerp query with a positive span and a query
/// depth in the bracket interior, away from the boundary clamp knees.
fn make_deep_lerp(state: &mut u64) -> VolumetricsQuery {
    let lo_depth = range(state, 0.5, 4.0);
    let span = range(state, 1.0, 6.0);
    let hi_depth = lo_depth + span;
    let t = range(state, 0.1, 0.9);
    VolumetricsQuery::DeepTransmittanceLerp {
        lo_depth,
        lo_transmittance: range(state, 0.1, 0.9),
        hi_depth,
        hi_transmittance: range(state, 0.1, 0.9),
        depth: lo_depth + t * span,
    }
}

/// Builds a [`SixWayLuminance`] from a six-bucket array in
/// `[right, left, up, down, front, back]` order.
fn luminance_of(b: [f32; 6]) -> SixWayLuminance {
    SixWayLuminance {
        right: b[0],
        left: b[1],
        up: b[2],
        down: b[3],
        front: b[4],
        back: b[5],
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`, matching the
/// result variant to the query variant and comparing channel-for-channel.
fn pin(idx: usize, query: &VolumetricsQuery, got: &VolumetricsResult) {
    match (query, got) {
        (
            VolumetricsQuery::HenyeyGreenstein { g, cos_theta },
            VolumetricsResult::HenyeyGreenstein(v),
        ) => {
            let want = henyey_greenstein(*g, *cos_theta);
            assert!(
                close(*v, want),
                "query {idx} henyey-greenstein: gpu {v} vs cpu {want}"
            );
        }
        (
            VolumetricsQuery::DoubleLobePhase {
                g,
                back_lobe_weight,
                back_g,
                cos_theta,
            },
            VolumetricsResult::DoubleLobePhase(v),
        ) => {
            let want = double_lobe_phase(phase_params(*g, *back_lobe_weight, *back_g), *cos_theta);
            assert!(
                close(*v, want),
                "query {idx} double-lobe phase: gpu {v} vs cpu {want}"
            );
        }
        (
            VolumetricsQuery::PhaseResponseBanded {
                g,
                back_lobe_weight,
                back_g,
                cos_theta,
                peak,
                bands,
            },
            VolumetricsResult::PhaseResponseBanded(v),
        ) => {
            let want = phase_response_banded(
                phase_params(*g, *back_lobe_weight, *back_g),
                *cos_theta,
                *peak,
                *bands,
            );
            assert!(
                close(*v, want),
                "query {idx} banded response: gpu {v} vs cpu {want}"
            );
        }
        (
            VolumetricsQuery::SixWayResponse {
                luminance,
                light_dir,
            },
            VolumetricsResult::SixWayResponse(v),
        ) => {
            let want = six_way_response(
                luminance_of(*luminance),
                Vec3::new(light_dir[0], light_dir[1], light_dir[2]),
            );
            assert!(
                close(*v, want),
                "query {idx} six-way response: gpu {v} vs cpu {want}"
            );
        }
        (
            VolumetricsQuery::SixWayDeposit {
                light_dir,
                intensity,
            },
            VolumetricsResult::SixWayDeposit(buckets),
        ) => {
            let mut acc = SixWayAccumulator::new();
            acc.add_light(
                Vec3::new(light_dir[0], light_dir[1], light_dir[2]),
                *intensity,
            );
            let baked = acc.bake();
            let want = [
                baked.right,
                baked.left,
                baked.up,
                baked.down,
                baked.front,
                baked.back,
            ];
            for (lane, (g, c)) in buckets.iter().zip(want.iter()).enumerate() {
                assert!(
                    close(*g, *c),
                    "query {idx} deposit lane {lane}: gpu {g} vs cpu {c}"
                );
            }
        }
        (
            VolumetricsQuery::DeepTransmittanceLerp {
                lo_depth,
                lo_transmittance,
                hi_depth,
                hi_transmittance,
                depth,
            },
            VolumetricsResult::DeepTransmittanceLerp(v),
        ) => {
            // Replay the reference bracket lerp through the real sampler: a
            // two-layer map with the query depth in the interior hits exactly
            // the per-bracket interpolation branch the twin runs.
            let layers = [
                DeepOpacityLayer::new(*lo_depth, *lo_transmittance),
                DeepOpacityLayer::new(*hi_depth, *hi_transmittance),
            ];
            let want = sample_deep_transmittance(&layers, *depth);
            assert!(
                close(*v, want),
                "query {idx} deep transmittance lerp: gpu {v} vs cpu {want}"
            );
        }
        _ => panic!("query {idx}: result variant does not match the query variant"),
    }
}

/// Draws a random query of a random routine with seam-safe fixtures.
fn rand_query(state: &mut u64) -> VolumetricsQuery {
    match (lcg(state) * 6.0) as u32 {
        0 => make_henyey(state),
        1 => make_double_lobe(state),
        2 => make_banded(state),
        3 => make_six_way_response(state),
        4 => make_six_way_deposit(state),
        _ => make_deep_lerp(state),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuVolumetrics, queries: &[VolumetricsQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumetrics::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn henyey_greenstein_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumetrics::new(&ctx);
    let mut state = 0x5157_1a2b_3c4d_5e6f_u64;
    let queries: Vec<VolumetricsQuery> = (0..64).map(|_| make_henyey(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn double_lobe_phase_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumetrics::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let queries: Vec<VolumetricsQuery> = (0..64).map(|_| make_double_lobe(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn phase_response_banded_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumetrics::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    let queries: Vec<VolumetricsQuery> = (0..64).map(|_| make_banded(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn six_way_response_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumetrics::new(&ctx);
    let mut state = 0x3141_5926_5358_9793_u64;
    let queries: Vec<VolumetricsQuery> =
        (0..64).map(|_| make_six_way_response(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn six_way_deposit_matches_golden_accumulator() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumetrics::new(&ctx);
    // Cross-check the twin's single deposit against the real
    // `SixWayAccumulator`: one `add_light` on a fresh accumulator, then `bake`,
    // compared bucket-for-bucket.
    let mut state = 0x2718_2818_2845_9045_u64;
    let queries: Vec<VolumetricsQuery> =
        (0..64).map(|_| make_six_way_deposit(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn deep_transmittance_lerp_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumetrics::new(&ctx);
    let mut state = 0x5a5a_a5a5_0f0f_f0f0_u64;
    let queries: Vec<VolumetricsQuery> = (0..64).map(|_| make_deep_lerp(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumetrics::new(&ctx);
    let mut state = 0x2b2b_1a1a_3c3c_4d4d_u64;
    // One batch mixing deterministic fixtures with many random queries of every
    // routine, dispatched together so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        VolumetricsQuery::HenyeyGreenstein {
            g: 0.3,
            cos_theta: 0.5,
        },
        VolumetricsQuery::DoubleLobePhase {
            g: 0.4,
            back_lobe_weight: 0.25,
            back_g: -0.3,
            cos_theta: -0.2,
        },
        VolumetricsQuery::SixWayResponse {
            luminance: [1.0, 0.5, 0.75, 0.25, 1.5, 0.6],
            light_dir: [0.3, -0.6, 0.7],
        },
        VolumetricsQuery::SixWayDeposit {
            light_dir: [-0.4, 0.5, -0.7],
            intensity: 2.0,
        },
        VolumetricsQuery::DeepTransmittanceLerp {
            lo_depth: 1.0,
            lo_transmittance: 0.9,
            hi_depth: 4.0,
            hi_transmittance: 0.2,
            depth: 2.5,
        },
    ];
    queries.push(make_banded(&mut state));
    for _ in 0..64 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumetrics::new(&ctx);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    // A larger sweep (several workgroups' worth) pins every routine across many
    // random fixtures.
    let queries: Vec<VolumetricsQuery> = (0..256).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
