//! Real-device parity for the `NDC` <-> pixel coordinate twin:
//! [`GpuMotionReprojectNdcPixel`](prism_volumetric_gpu::motion_reproject_ndc_pixel::GpuMotionReprojectNdcPixel)
//! must reproduce the stateless affine pair of the `CPU` golden
//! [`reproject`](prism_render_architecture::motion::reproject) module — the
//! [`ndc_to_pixel`](prism_render_architecture::motion::reproject::ndc_to_pixel)
//! forward map and the
//! [`pixel_to_ndc`](prism_render_architecture::motion::reproject::pixel_to_ndc)
//! inverse — across the `NDC` cube corners, interior points, a spread of
//! render-target extents, and a randomized batch mixing both ops compared
//! coordinate-for-coordinate.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden functions
//! [`ndc_to_pixel`](prism_render_architecture::motion::reproject::ndc_to_pixel)
//! and
//! [`pixel_to_ndc`](prism_render_architecture::motion::reproject::pixel_to_ndc)
//! are `pub`, so each `GPU` result is pinned directly against the golden run on
//! the same input. The extents are passed through
//! [`ScreenDims::new`](prism_render_architecture::motion::ScreenDims::new), which
//! clamps each axis up to `1` exactly as the device-side host packer does, so
//! both sides divide by the same sanitized extent.
//!
//! # Parity criterion
//!
//! Every output of both maps is a continuous `f32` threaded through a
//! multiply-add (or a divide and a multiply-add) only — no `sqrt`, no
//! transcendental — so each coordinate is asserted within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! The maps are affine with no truncation or branch on the input coordinate, so
//! there is no half-step tie or discrete threshold to straddle; ordinary
//! fixtures and random inputs are safe. Extents are kept at `1` or greater, the
//! same floor both the oracle and the host apply.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::reproject`；无第三方引擎源码或衍生代码。

use prism_render_architecture::motion::reproject::{ndc_to_pixel, pixel_to_ndc};
use prism_render_architecture::motion::{ScreenDims, Vec2};
use prism_volumetric_gpu::motion_reproject_ndc_pixel::{
    GpuMotionReprojectNdcPixel, MotionReprojectNdcPixelQuery, MotionReprojectNdcPixelResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a converted coordinate.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes.
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

/// Computes the golden result for one query, so the oracle lives beside the
/// device call and both read the same input.
fn expected(q: &MotionReprojectNdcPixelQuery) -> MotionReprojectNdcPixelResult {
    match *q {
        MotionReprojectNdcPixelQuery::NdcToPixel { ndc, dims } => {
            let p = ndc_to_pixel(Vec2::new(ndc[0], ndc[1]), ScreenDims::new(dims[0], dims[1]));
            MotionReprojectNdcPixelResult::NdcToPixel { pixel: [p.x, p.y] }
        }
        MotionReprojectNdcPixelQuery::PixelToNdc { pixel, dims } => {
            let n = pixel_to_ndc(
                Vec2::new(pixel[0], pixel[1]),
                ScreenDims::new(dims[0], dims[1]),
            );
            MotionReprojectNdcPixelResult::PixelToNdc { ndc: [n.x, n.y] }
        }
    }
}

/// Pins one `GPU` result against the golden oracle: both continuous coordinates
/// within tolerance, with a variant-mismatch guard.
fn assert_result(
    idx: usize,
    got: &MotionReprojectNdcPixelResult,
    want: &MotionReprojectNdcPixelResult,
) {
    match (*got, *want) {
        (
            MotionReprojectNdcPixelResult::NdcToPixel { pixel: g },
            MotionReprojectNdcPixelResult::NdcToPixel { pixel: w },
        ) => assert!(
            close(g[0], w[0]) && close(g[1], w[1]),
            "result {idx} ndc_to_pixel: gpu {g:?} vs cpu {w:?}"
        ),
        (
            MotionReprojectNdcPixelResult::PixelToNdc { ndc: g },
            MotionReprojectNdcPixelResult::PixelToNdc { ndc: w },
        ) => assert!(
            close(g[0], w[0]) && close(g[1], w[1]),
            "result {idx} pixel_to_ndc: gpu {g:?} vs cpu {w:?}"
        ),
        _ => panic!("result {idx} variant mismatch: gpu {got:?} vs cpu {want:?}"),
    }
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[MotionReprojectNdcPixelQuery]) {
    let gpu = GpuMotionReprojectNdcPixel::new(ctx);
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

/// Draws an integer extent in `[lo, hi]` from `state`.
fn ranged_u32(state: &mut u64, lo: u32, hi: u32) -> u32 {
    lo + lcg(state) % (hi - lo + 1)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping motion_reproject_ndc_pixel parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuMotionReprojectNdcPixel::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn ndc_to_pixel_corner_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // The four NDC cube corners, the center, and the two off-diagonal corners,
    // at a 16:9 render target.
    let dims = [1920u32, 1080u32];
    let queries: Vec<MotionReprojectNdcPixelQuery> = [
        [-1.0, -1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [0.5, -0.25],
        [-0.3, 0.7],
    ]
    .into_iter()
    .map(|ndc| MotionReprojectNdcPixelQuery::NdcToPixel { ndc, dims })
    .collect();
    run_and_check(&ctx, &queries);
}

#[test]
fn pixel_to_ndc_corner_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // The pixel-grid corners, center, and interior points at a 1280x720 target.
    let dims = [1280u32, 720u32];
    let queries: Vec<MotionReprojectNdcPixelQuery> = [
        [0.0, 0.0],
        [1280.0, 720.0],
        [640.0, 360.0],
        [0.0, 720.0],
        [1280.0, 0.0],
        [320.5, 540.25],
        [900.0, 120.0],
    ]
    .into_iter()
    .map(|pixel| MotionReprojectNdcPixelQuery::PixelToNdc { pixel, dims })
    .collect();
    run_and_check(&ctx, &queries);
}

#[test]
fn varied_extent_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // A spread of render-target extents, including the sanitized floor of 1.
    let extents: [[u32; 2]; 5] = [[1920, 1080], [1280, 720], [640, 480], [100, 100], [1, 1]];
    let mut queries: Vec<MotionReprojectNdcPixelQuery> = Vec::new();
    for dims in extents {
        queries.push(MotionReprojectNdcPixelQuery::NdcToPixel {
            ndc: [0.37, -0.62],
            dims,
        });
        queries.push(MotionReprojectNdcPixelQuery::PixelToNdc {
            pixel: [0.37 * dims[0] as f32, 0.62 * dims[1] as f32],
            dims,
        });
    }
    run_and_check(&ctx, &queries);
}

#[test]
fn out_of_range_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    // Out-of-frame NDC and out-of-bounds pixels still map affinely; there is no
    // clamp on the coordinate, only on the extent.
    let dims = [800u32, 600u32];
    let queries = vec![
        MotionReprojectNdcPixelQuery::NdcToPixel {
            ndc: [-2.5, 3.1],
            dims,
        },
        MotionReprojectNdcPixelQuery::NdcToPixel {
            ndc: [4.0, -4.0],
            dims,
        },
        MotionReprojectNdcPixelQuery::PixelToNdc {
            pixel: [-120.0, 950.0],
            dims,
        },
        MotionReprojectNdcPixelQuery::PixelToNdc {
            pixel: [2000.0, -300.0],
            dims,
        },
    ];
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x7d4e_31a9_c0f5_1b63_u64;

    let mut queries: Vec<MotionReprojectNdcPixelQuery> = Vec::new();
    while queries.len() < 256 {
        let dims = [
            ranged_u32(&mut state, 1, 4096),
            ranged_u32(&mut state, 1, 4096),
        ];
        let q = if (lcg(&mut state) & 1) == 0 {
            MotionReprojectNdcPixelQuery::NdcToPixel {
                ndc: [ranged(&mut state, -1.5, 1.5), ranged(&mut state, -1.5, 1.5)],
                dims,
            }
        } else {
            let w = dims[0] as f32;
            let h = dims[1] as f32;
            MotionReprojectNdcPixelQuery::PixelToNdc {
                pixel: [
                    ranged(&mut state, -50.0, w + 50.0),
                    ranged(&mut state, -50.0, h + 50.0),
                ],
                dims,
            }
        };
        queries.push(q);
    }
    run_and_check(&ctx, &queries);
}
