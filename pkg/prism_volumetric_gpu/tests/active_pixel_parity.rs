//! Real-device parity for the active-pixel twin:
//! [`GpuActivePixel`] must reproduce the `CPU` golden
//! [`active_pixel`](prism_render_architecture::volumetric::temporal::active_pixel)
//! for every update mode across a grid of pixels and a span of frame indices.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL
//! (integer bit ops and comparisons), so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The predicate is integer bit ops and comparisons, so the decision is exact —
//! there is no floating-point slack. Beyond a bit-for-bit match against the
//! `CPU` golden, the suite asserts the scheduling invariant that makes the
//! predicate correct: over one [`UpscaleMode::period`] every pixel is active on
//! exactly one frame (the per-frame active sets partition, then union to, the
//! whole grid). A degenerate kernel (dropped mode branch, wrong bit) could not
//! satisfy both the parity and coverage checks.
//!
//! Provenance: standard checkerboard / quarter-res upsampling schedule; no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::temporal::{active_pixel, UpscaleMode};
use prism_volumetric_gpu::{ActivePixelQuery, GpuActivePixel, GpuContext};

const GRID: u32 = 8;

/// Every update mode under test.
const MODES: [UpscaleMode; 3] = [
    UpscaleMode::Full,
    UpscaleMode::Checkerboard,
    UpscaleMode::QuarterRes,
];

/// Builds one query per `(mode, frame_index, x, y)` over an `8x8` grid and a
/// sweep of frame indices, in a fixed, deterministic order.
fn build_queries(frames: u32) -> Vec<ActivePixelQuery> {
    let mut queries = Vec::new();
    for &mode in &MODES {
        for frame_index in 0..frames {
            for y in 0..GRID {
                for x in 0..GRID {
                    queries.push(ActivePixelQuery {
                        frame_index,
                        x,
                        y,
                        mode,
                    });
                }
            }
        }
    }
    queries
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_active_pixel_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping active-pixel parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuActivePixel::new(&ctx);

    // Enough frames to cover more than one full period of every mode.
    let frames = 8;
    let queries = build_queries(frames);
    let gpu = gpu_kernel.eval(&ctx, &queries);

    assert_eq!(gpu.len(), queries.len(), "one decision per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = active_pixel(q.frame_index, q.x, q.y, q.mode);
        assert_eq!(
            gpu[i], exp,
            "active-pixel mismatch for query {i} (mode {:?}, frame {}, x {}, y {}): \
             gpu {}, cpu {exp}",
            q.mode, q.frame_index, q.x, q.y, gpu[i]
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_active_pixel_covers_grid_once_per_period() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping active-pixel coverage: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuActivePixel::new(&ctx);

    // For each mode, over exactly one period the active sets must partition the
    // grid: every pixel is active on exactly one frame (union covers all, no
    // pixel starved and none resolved twice).
    for &mode in &MODES {
        let period = mode.period();
        let mut queries = Vec::new();
        for frame_index in 0..period {
            for y in 0..GRID {
                for x in 0..GRID {
                    queries.push(ActivePixelQuery {
                        frame_index,
                        x,
                        y,
                        mode,
                    });
                }
            }
        }
        let gpu = gpu_kernel.eval(&ctx, &queries);
        assert_eq!(gpu.len(), queries.len(), "one decision per query");

        // Count active frames per pixel over the period.
        let mut active_count = vec![0u32; (GRID * GRID) as usize];
        for (i, q) in queries.iter().enumerate() {
            if gpu[i] {
                active_count[(q.y * GRID + q.x) as usize] += 1;
            }
        }
        for (pixel, &count) in active_count.iter().enumerate() {
            let x = pixel as u32 % GRID;
            let y = pixel as u32 / GRID;
            assert_eq!(
                count, 1,
                "mode {mode:?}: pixel ({x}, {y}) active {count} times over period \
                 {period}, expected exactly once"
            );
        }
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuActivePixel::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
