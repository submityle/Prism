//! Real-device parity for the spectral-dispersion refraction twin:
//! [`GpuWaterDispersionOffsets`](prism_volumetric_gpu::water_dispersion_offsets::GpuWaterDispersionOffsets)
//! must reproduce the per-sample outputs of the `CPU` golden
//! [`dispersion`](prism_render_architecture::water::dispersion) — the `Cauchy`
//! index law, the three reference-wavelength indices, the `Snell` transmitted
//! sine, the red-minus-blue colour spread and the per-channel screen-space
//! offsets — across fixed fixtures and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Each expected value is produced by calling the public golden functions
//! [`cauchy_ior`](prism_render_architecture::water::dispersion::cauchy_ior),
//! [`spectral_iors`](prism_render_architecture::water::dispersion::spectral_iors),
//! [`channel_transmitted_sine`](prism_render_architecture::water::dispersion::channel_transmitted_sine),
//! [`dispersion_spread`](prism_render_architecture::water::dispersion::dispersion_spread)
//! and
//! [`dispersion_offsets`](prism_render_architecture::water::dispersion::dispersion_offsets)
//! with the same inputs, the flattened three indices rebuilt into an
//! [`RgbIor`](prism_render_architecture::water::dispersion::RgbIor).
//!
//! # Parity criterion
//!
//! Every output threads through only subtracts, multiplies, a guarded divide,
//! `max` and `clamp`, so the `CPU` and `GPU` are not bit-exact; each of the nine
//! continuous fields is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (relative floor `1e-6`). The randomized sweep keeps the
//! indices above one and the incidence sine clear of the `0..=1` clamp edges so
//! no `CPU`/`GPU` last-place difference straddles a clamp or the non-negative
//! spread/strength floor.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::dispersion`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::dispersion::{
    cauchy_ior, channel_transmitted_sine, dispersion_offsets, dispersion_spread, spectral_iors,
    RgbIor,
};
use prism_volumetric_gpu::water_dispersion_offsets::{
    GpuWaterDispersionOffsets, WaterDispersionOffsetsQuery, WaterDispersionOffsetsResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute agreement bound for a continuous field.
const EPS: f32 = 1.0e-4;

/// Relative agreement bound for a continuous field.
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

/// Rebuilds the golden outputs for one query by calling the public reference
/// functions directly, the three flattened indices reassembled into an
/// [`RgbIor`].
fn oracle(q: &WaterDispersionOffsetsQuery) -> WaterDispersionOffsetsResult {
    let iors = RgbIor {
        r: q.ior_r,
        g: q.ior_g,
        b: q.ior_b,
    };
    let spectral = spectral_iors(q.cauchy_a, q.cauchy_b);
    let offsets = dispersion_offsets(iors, q.sin_incidence, q.strength);
    WaterDispersionOffsetsResult {
        cauchy: cauchy_ior(q.cauchy_a, q.cauchy_b, q.wavelength_um),
        spectral_r: spectral.r,
        spectral_g: spectral.g,
        spectral_b: spectral.b,
        channel_sine: channel_transmitted_sine(q.sin_incidence, q.channel_ior),
        spread: dispersion_spread(iors, q.sin_incidence),
        offset_r: offsets[0],
        offset_g: offsets[1],
        offset_b: offsets[2],
    }
}

/// Pins one `GPU` sample result against the in-host oracle: every continuous
/// field within tolerance.
fn check_sample(
    idx: usize,
    got: &WaterDispersionOffsetsResult,
    want: &WaterDispersionOffsetsResult,
) {
    let fields: [(&str, f32, f32); 9] = [
        ("cauchy", got.cauchy, want.cauchy),
        ("spectral_r", got.spectral_r, want.spectral_r),
        ("spectral_g", got.spectral_g, want.spectral_g),
        ("spectral_b", got.spectral_b, want.spectral_b),
        ("channel_sine", got.channel_sine, want.channel_sine),
        ("spread", got.spread, want.spread),
        ("offset_r", got.offset_r, want.offset_r),
        ("offset_g", got.offset_g, want.offset_g),
        ("offset_b", got.offset_b, want.offset_b),
    ];
    for (name, g, w) in fields {
        assert!(close(g, w), "sample {idx} {name}: gpu {g} vs cpu {w}");
    }
}

/// Dispatches every sample and pins each result against the oracle.
fn check(
    ctx: &GpuContext,
    gpu: &GpuWaterDispersionOffsets,
    queries: &[WaterDispersionOffsetsQuery],
) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_sample(idx, result, &want);
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

/// Draws a `f32` in `[lo, hi]` at milli resolution from `state`.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let t = (lcg(state) % 1_000_001) as f32 / 1_000_000.0;
    lo + (hi - lo) * t
}

/// The clear open-water tuning used as the fixed fixture base: positive
/// dispersion, water-like indices, a mid incidence sine.
fn base_query() -> WaterDispersionOffsetsQuery {
    let iors = spectral_iors(1.324, 0.003);
    WaterDispersionOffsetsQuery {
        cauchy_a: 1.324,
        cauchy_b: 0.003,
        wavelength_um: 0.546,
        sin_incidence: 0.6,
        strength: 2.0,
        channel_ior: 1.333,
        ior_r: iors.r,
        ior_g: iors.g,
        ior_b: iors.b,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_dispersion_offsets parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterDispersionOffsets::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn fixed_fixtures_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterDispersionOffsets::new(&ctx);

    let mut queries = Vec::new();

    // Canonical clear water.
    queries.push(base_query());

    // Normal incidence (sine 0): transmitted sines, spread and offsets vanish.
    let mut head_on = base_query();
    head_on.sin_incidence = 0.0;
    queries.push(head_on);

    // Near-grazing incidence: maximal transmitted sines and colour spread.
    let mut grazing = base_query();
    grazing.sin_incidence = 0.95;
    queries.push(grazing);

    // Zero dispersion coefficient collapses the three indices together.
    let mut flat = base_query();
    flat.cauchy_b = 0.0;
    let flat_iors = spectral_iors(flat.cauchy_a, flat.cauchy_b);
    flat.ior_r = flat_iors.r;
    flat.ior_g = flat_iors.g;
    flat.ior_b = flat_iors.b;
    queries.push(flat);

    // Strong dispersion widens the fringe.
    let mut strong = base_query();
    strong.cauchy_b = 0.01;
    let strong_iors = spectral_iors(strong.cauchy_a, strong.cauchy_b);
    strong.ior_r = strong_iors.r;
    strong.ior_g = strong_iors.g;
    strong.ior_b = strong_iors.b;
    queries.push(strong);

    // Non-positive strength clamps every offset to zero on both sides.
    let mut no_gain = base_query();
    no_gain.strength = -1.0;
    queries.push(no_gain);

    // A standalone channel probe at a higher index bends closer to the normal.
    let mut dense = base_query();
    dense.channel_ior = 1.4;
    dense.wavelength_um = 0.70;
    queries.push(dense);

    check(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterDispersionOffsets::new(&ctx);

    let mut state = 0x51ed_dcab_1234_f00du64;
    let mut queries = Vec::new();
    while queries.len() < 4096 {
        // Positive dispersion with water-like Cauchy parameters so the derived
        // indices stay above one and ordered r < g < b.
        let cauchy_a = uniform(&mut state, 1.20, 1.45);
        let cauchy_b = uniform(&mut state, 0.0, 0.012);

        // Independent indices for the spread/offset probe, kept above one and
        // ordered so the spread stays a clear margin off its non-negative floor.
        let ior_r = uniform(&mut state, 1.30, 1.34);
        let ior_g = ior_r + uniform(&mut state, 0.004, 0.012);
        let ior_b = ior_g + uniform(&mut state, 0.004, 0.012);

        let q = WaterDispersionOffsetsQuery {
            cauchy_a,
            cauchy_b,
            // Probe wavelength kept positive, clear of the degenerate guard.
            wavelength_um: uniform(&mut state, 0.40, 0.75),
            // Incidence sine kept clear of the 0 and 1 clamp edges.
            sin_incidence: uniform(&mut state, 0.05, 0.9),
            // Strength kept non-negative, clear of the max(., 0) floor.
            strength: uniform(&mut state, 0.2, 5.0),
            // Standalone channel index above one so s / n stays below the clamp.
            channel_ior: uniform(&mut state, 1.30, 1.40),
            ior_r,
            ior_g,
            ior_b,
        };
        queries.push(q);
    }

    check(&ctx, &gpu, &queries);
}
