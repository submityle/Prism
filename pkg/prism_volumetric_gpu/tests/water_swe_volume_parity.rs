//! Real-device parity for the shallow-water conserved-volume twin:
//! [`GpuWaterSweVolume`](prism_volumetric_gpu::water_swe_volume::GpuWaterSweVolume)
//! must reproduce the stateless conserved volume of the `CPU` golden
//! [`total_volume`](prism_render_architecture::water::swe::SweState::total_volume)
//! — the left-to-right depth fold scaled by the square cell area `dx * dx` —
//! across hand-computed, fixed-size and randomized depth arrays compared
//! value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The reference method is public, so it drives the oracle directly: each query
//! feeds its active depth prefix into a
//! [`SweState`](prism_render_architecture::water::swe::SweState) and calls
//! [`total_volume`](prism_render_architecture::water::swe::SweState::total_volume)
//! with a matching
//! [`SweConfig`](prism_render_architecture::water::swe::SweConfig). A passing
//! `GPU == oracle` run is direct evidence the kernel computes the same volume.
//!
//! # Parity criterion
//!
//! Both sides evaluate the identical left-to-right fold and the identical area
//! scale, so the volume matches to within floating-point tolerance
//! (`abs <= 1e-4` or `rel <= 1e-3`, with a `REL_FLOOR` of `1e-6`). There is no
//! branch on an `f32`, so there is no floating-point crossing to flip; the
//! relative bound covers large sums.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::swe`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::swe::{SweConfig, SweState};
use prism_volumetric_gpu::water_swe_volume::{
    GpuWaterSweVolume, WaterSweVolumeQuery, WaterSweVolumeResult, MAX_H,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute closeness floor for the parity comparison.
const ABS: f32 = 1e-4;
/// Relative closeness bound for the parity comparison.
const REL: f32 = 1e-3;
/// Smallest denominator used in the relative comparison, guarding `0 == 0`.
const REL_FLOOR: f32 = 1e-6;

/// Absolute-or-relative closeness: `true` when `a` and `b` agree to within the
/// shared tolerance.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS || rel <= REL
}

/// Computes the reference conserved volume for one query by feeding its active
/// depth prefix to a [`SweState`] and calling the golden
/// [`total_volume`](prism_render_architecture::water::swe::SweState::total_volume).
///
/// The grid shape (`nx`, `nz`) is irrelevant to `total_volume` — it reads only
/// `h` and `cfg.dx` — so a `1 x len` layout with the query's `dx` suffices.
fn oracle(q: &WaterSweVolumeQuery) -> WaterSweVolumeResult {
    let len = q.len as usize;
    let cfg = SweConfig {
        nx: 1,
        nz: len as u32,
        dx: q.dx,
        gravity: 9.81,
        damping: 0.0,
    };
    let state = SweState {
        h: q.h[..len].to_vec(),
        u: vec![0.0; len],
        v: vec![0.0; len],
    };
    WaterSweVolumeResult {
        volume: state.total_volume(cfg),
    }
}

/// Pins one `GPU` result against the oracle: the volume within tolerance.
fn check_result(idx: usize, got: &WaterSweVolumeResult, want: &WaterSweVolumeResult) {
    assert!(
        close(got.volume, want.volume),
        "query {idx}: gpu {} vs cpu {}",
        got.volume,
        want.volume
    );
}

/// Runs `queries` on the `GPU` and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterSweVolume, queries: &[WaterSweVolumeQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_result(idx, result, &want);
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

/// Maps a raw `u32` into `[lo, hi)` as an `f32`, using only integer and
/// floating-point arithmetic (no transcendental), for the random sweep.
fn uniform(raw: u32, lo: f32, hi: f32) -> f32 {
    let unit = (raw as f32) / (u32::MAX as f32);
    lo + unit * (hi - lo)
}

/// Builds a query whose active `len` depth cells are filled from the generator
/// in `[0, 10]` and whose padding stays `0`, with `dx` in `[0.1, 2.0]`.
fn random_query(state: &mut u64, len: u32) -> WaterSweVolumeQuery {
    let mut h = [0.0f32; MAX_H];
    for value in h.iter_mut().take(len as usize) {
        *value = uniform(lcg(state), 0.0, 10.0);
    }
    let dx = uniform(lcg(state), 0.1, 2.0);
    WaterSweVolumeQuery::new(h, len, dx)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_swe_volume parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterSweVolume::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn hand_computed_uniform_depth() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweVolume::new(&ctx);
    // 64 cells of depth 1.0 with dx = 0.5: volume = 64 * 1.0 * (0.5 * 0.5) =
    // 64 * 0.25 = 16.0, matching the reference conserved quantity.
    let mut h = [0.0f32; MAX_H];
    for value in h.iter_mut().take(64) {
        *value = 1.0;
    }
    let queries = [WaterSweVolumeQuery::new(h, 64, 0.5)];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(close(got[0].volume, 16.0), "uniform volume is 16.0");
    check(&ctx, &gpu, &queries);
}

#[test]
fn hand_computed_single_cell() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweVolume::new(&ctx);
    // A single cell of depth 3.0 with dx = 2.0: volume = 3.0 * (2.0 * 2.0) =
    // 12.0.
    let mut h = [0.0f32; MAX_H];
    h[0] = 3.0;
    let queries = [WaterSweVolumeQuery::new(h, 1, 2.0)];
    let got = gpu.evaluate(&ctx, &queries);
    assert!(close(got[0].volume, 12.0), "single-cell volume is 12.0");
    check(&ctx, &gpu, &queries);
}

#[test]
fn fixed_sizes_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweVolume::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // A spread of active lengths from a single cell up to the full cap.
    let queries = [
        random_query(&mut state, 1),
        random_query(&mut state, 16),
        random_query(&mut state, 64),
        random_query(&mut state, 1024),
        random_query(&mut state, MAX_H as u32),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterSweVolume::new(&ctx);
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let lengths = [1u32, 7u32, 64u32, 256u32, 1000u32, 4096u32];
    let mut queries = Vec::new();
    // A broad sweep: random active lengths, random depths in [0, 10] and a
    // square cell size in [0.1, 2.0] kept well away from zero. There is no
    // branch on an f32, so no reject-sampling is needed.
    while queries.len() < 512 {
        let len = lengths[(lcg(&mut state) % lengths.len() as u32) as usize];
        queries.push(random_query(&mut state, len));
    }
    check(&ctx, &gpu, &queries);
}
