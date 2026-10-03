//! Real-device parity for the film-grain luma and hashed-grain twin:
//! [`GpuFilmGrainLuma`](prism_volumetric_gpu::film_grain_luma::GpuFilmGrainLuma)
//! must reproduce an independent host reimplementation of the `CPU` golden
//! `luma709` and `grain_hash01` across hand-picked fixtures plus a randomized
//! batch compared value-for-value.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! Rather than linking the golden crate, this test faithfully reimplements both
//! routines on the host from the published constants and arithmetic, so the
//! `WGSL` kernel and the host oracle are independent transcriptions of the same
//! closed-form contract.
//!
//! # Parity criterion
//!
//! The luma is a continuous `f32` built from multiply-adds and one clamp,
//! compared with `abs <= 1e-6 || rel <= 1e-5`. The grain is derived from an
//! integer hash and an exact power-of-two divisor, so it should be essentially
//! exact; it is compared with `abs <= 1e-5 || rel <= 1e-4`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::film_grain`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::film_grain_luma::{
    FilmGrainLumaQuery, FilmGrainLumaResult, GpuFilmGrainLuma,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-comparison floor so tiny magnitudes do not inflate the relative
/// error.
const REL_FLOOR: f32 = 1.0e-6;

/// `Rec. 709` luma weight for the red channel.
const LUMA_R: f32 = 0.2126;
/// `Rec. 709` luma weight for the green channel.
const LUMA_G: f32 = 0.7152;
/// `Rec. 709` luma weight for the blue channel.
const LUMA_B: f32 = 0.0722;

/// Odd decorrelation multiplier for the pixel `x` coordinate.
const ODD_X: u32 = 0x9E37_79B1;
/// Odd decorrelation multiplier for the pixel `y` coordinate.
const ODD_Y: u32 = 0x85EB_CA77;
/// Odd decorrelation multiplier for the frame index.
const ODD_FRAME: u32 = 0xC2B2_AE3D;
/// First avalanche-finalizer odd multiply constant.
const MIX_A: u32 = 0x7FEB_352D;
/// Second avalanche-finalizer odd multiply constant.
const MIX_B: u32 = 0x846C_A68B;
/// `2^24`, the exact `f32` normalization divisor for the top `24` hash bits.
const NORM_24: f32 = 16_777_216.0;

/// Host reimplementation of the golden `hash_u32` avalanche finalizer.
fn hash_u32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(MIX_A);
    h ^= h >> 15;
    h = h.wrapping_mul(MIX_B);
    h ^= h >> 16;
    h
}

/// Host reimplementation of the golden `grain_hash01`.
fn grain_hash01(x: u32, y: u32, frame: u32) -> f32 {
    let a = x.wrapping_mul(ODD_X);
    let b = y.wrapping_mul(ODD_Y);
    let c = frame.wrapping_mul(ODD_FRAME);
    let h = hash_u32(a ^ b ^ c);
    (h >> 8) as f32 / NORM_24
}

/// Host reimplementation of the golden `luma709`.
fn luma709(r: f32, g: f32, b: f32) -> f32 {
    (LUMA_R * r + LUMA_G * g + LUMA_B * b).clamp(0.0, 1.0)
}

/// Computes the golden reconstruction for one query via the host oracle.
fn expected(q: &FilmGrainLumaQuery) -> FilmGrainLumaResult {
    FilmGrainLumaResult {
        luma: luma709(q.color_r, q.color_g, q.color_b),
        grain: grain_hash01(q.px, q.py, q.frame),
    }
}

/// Returns whether `got` matches `want` within the mixed absolute/relative
/// tolerance `abs <= abs_tol || rel <= rel_tol`.
fn close(got: f32, want: f32, abs_tol: f32, rel_tol: f32) -> bool {
    let diff = (got - want).abs();
    if diff <= abs_tol {
        return true;
    }
    let denom = want.abs().max(REL_FLOOR);
    diff / denom <= rel_tol
}

/// Pins one `GPU` result against the oracle: luma under the multiply-add
/// tolerance and grain under the integer-derived tolerance.
fn assert_result(idx: usize, got: &FilmGrainLumaResult, want: &FilmGrainLumaResult) {
    assert!(
        close(got.luma, want.luma, 1.0e-6, 1.0e-5),
        "result {idx} luma: gpu={} cpu={}",
        got.luma,
        want.luma
    );
    assert!(
        close(got.grain, want.grain, 1.0e-5, 1.0e-4),
        "result {idx} grain: gpu={} cpu={}",
        got.grain,
        want.grain
    );
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[FilmGrainLumaQuery]) {
    let gpu = GpuFilmGrainLuma::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
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

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping film_grain_luma parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuFilmGrainLuma::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn film_grain_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Pure black: luma 0, grain still well defined at origin.
        FilmGrainLumaQuery {
            color_r: 0.0,
            color_g: 0.0,
            color_b: 0.0,
            px: 0,
            py: 0,
            frame: 0,
        },
        // Pure white: luma clamps to 1 (weights sum to 1).
        FilmGrainLumaQuery {
            color_r: 1.0,
            color_g: 1.0,
            color_b: 1.0,
            px: 1,
            py: 1,
            frame: 1,
        },
        // Saturated overflow: components above 1 clamp the luma to 1.
        FilmGrainLumaQuery {
            color_r: 2.0,
            color_g: 3.0,
            color_b: 4.0,
            px: 7,
            py: 11,
            frame: 2,
        },
        // Pure red channel: luma == LUMA_R.
        FilmGrainLumaQuery {
            color_r: 1.0,
            color_g: 0.0,
            color_b: 0.0,
            px: 100,
            py: 200,
            frame: 3,
        },
        // Pure green channel: luma == LUMA_G.
        FilmGrainLumaQuery {
            color_r: 0.0,
            color_g: 1.0,
            color_b: 0.0,
            px: 640,
            py: 480,
            frame: 4,
        },
        // Pure blue channel: luma == LUMA_B.
        FilmGrainLumaQuery {
            color_r: 0.0,
            color_g: 0.0,
            color_b: 1.0,
            px: 1920,
            py: 1080,
            frame: 5,
        },
        // Midtone gray.
        FilmGrainLumaQuery {
            color_r: 0.5,
            color_g: 0.5,
            color_b: 0.5,
            px: 12_345,
            py: 54_321,
            frame: 60,
        },
        // Large pixel/frame indices to exercise the wrapping multiplies.
        FilmGrainLumaQuery {
            color_r: 0.25,
            color_g: 0.75,
            color_b: 0.125,
            px: 4_000_000_000,
            py: 3_000_000_000,
            frame: 2_000_000_000,
        },
        // Mixed asymmetric color.
        FilmGrainLumaQuery {
            color_r: 0.9,
            color_g: 0.1,
            color_b: 0.4,
            px: 65_535,
            py: 32_768,
            frame: 123,
        },
        // Another fixed grain probe with distinct coordinates.
        FilmGrainLumaQuery {
            color_r: 0.33,
            color_g: 0.66,
            color_b: 0.99,
            px: 255,
            py: 256,
            frame: 257,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x51ed_270b_c3f2_a1d9_u64;

    let mut queries: Vec<FilmGrainLumaQuery> = Vec::new();
    while queries.len() < 512 {
        queries.push(FilmGrainLumaQuery {
            // Colors span below 0 and above 1 to exercise both clamp ends.
            color_r: ranged(&mut state, -0.5, 1.5),
            color_g: ranged(&mut state, -0.5, 1.5),
            color_b: ranged(&mut state, -0.5, 1.5),
            px: lcg(&mut state),
            py: lcg(&mut state),
            frame: lcg(&mut state),
        });
    }
    run_and_check(&ctx, &queries);
}
