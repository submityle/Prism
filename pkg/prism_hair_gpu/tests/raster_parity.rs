//! Real-device parity for the hair raster-path classification twin:
//! [`GpuHairRaster`] must reproduce the `CPU` golden
//! [`classify_hair_raster`](prism_render_architecture::hair::raster::classify_hair_raster)
//! for every projected segment, including the threshold boundaries and the
//! negative-parameter clamp the reference guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Unlike the geometry twins, the routing decision is pure comparison
//! arithmetic with no fused multiply-add and no transcendental call: the `CPU`
//! and `GPU` compare the identical raw `f32` inputs, so the emitted path code
//! is **bit-identical** and every case asserts exact integer equality. The
//! sweeps also assert that all three paths are actually exercised, so a
//! degenerate constant kernel (always culling, or always software) could not
//! pass.
//!
//! Provenance: standard analytic thin-strand raster routing; no Unreal Engine
//! source or derived code.

use prism_hair_gpu::{path_code, GpuContext, GpuHairRaster, RasterQuery};
use prism_render_architecture::hair::raster::{
    classify_hair_raster, HairRasterParams, HairRasterPath, HairStrandSegmentStats,
};

/// Builds a golden statistic triple.
fn stats(screen_width_px: f32, screen_length_px: f32, coverage: f32) -> HairStrandSegmentStats {
    HairStrandSegmentStats {
        screen_width_px,
        screen_length_px,
        coverage,
    }
}

/// Asserts `gpu` matches the `CPU` golden path code for every segment. The
/// classification is pure comparison arithmetic, so the codes are bit-identical
/// and equality is exact.
fn assert_parity(segments: &[HairStrandSegmentStats], params: HairRasterParams, gpu: &[u32]) {
    assert_eq!(gpu.len(), segments.len(), "one path code per segment");
    for (i, &s) in segments.iter().enumerate() {
        let expected = path_code(classify_hair_raster(s, params));
        assert_eq!(
            gpu[i], expected,
            "raster path mismatch for segment {i}: gpu {}, cpu {expected}",
            gpu[i]
        );
    }
}

/// Number of distinct path codes present in `codes`.
fn distinct_paths(codes: &[u32]) -> usize {
    let mut seen = [false; 3];
    for &c in codes {
        seen[c as usize] = true;
    }
    seen.iter().filter(|b| **b).count()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_default_sweep_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping raster parity: no wgpu adapter on this host");
        return;
    };
    let raster = GpuHairRaster::new(&ctx);
    let params = HairRasterParams::default();

    // A mix that lands in every bucket: thin+covered -> software,
    // thick+covered -> hardware, sub-coverage or zero-length -> culled.
    let segments = [
        stats(0.4, 20.0, 0.9), // thin, covered -> software
        stats(1.5, 12.0, 0.5), // thin, covered -> software
        stats(6.0, 20.0, 0.9), // thick, covered -> hardware
        stats(3.0, 8.0, 0.2),  // thick, covered -> hardware
        stats(0.4, 20.0, 0.0), // zero coverage -> culled
        stats(0.4, 0.0, 1.0),  // zero length -> culled
        stats(0.4, -1.0, 1.0), // negative length -> culled
    ];
    let queries: Vec<RasterQuery> = segments
        .iter()
        .map(|&s| RasterQuery::from_stats(s))
        .collect();

    let gpu = raster.eval(&ctx, &queries, params);
    assert_parity(&segments, params, &gpu);
    assert_eq!(
        distinct_paths(&gpu),
        3,
        "the sweep must exercise all three raster paths"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_width_threshold_boundary_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping raster parity: no wgpu adapter on this host");
        return;
    };
    let raster = GpuHairRaster::new(&ctx);
    let params = HairRasterParams {
        min_coverage: 0.0,
        software_width_threshold: 2.0,
    };

    // Exactly at the threshold resolves to software (`<=`); just past flips to
    // hardware. Both sides must agree bit-for-bit on the boundary.
    let segments = [
        stats(1.999_999, 5.0, 1.0),
        stats(2.0, 5.0, 1.0),
        stats(2.000_001, 5.0, 1.0),
    ];
    let queries: Vec<RasterQuery> = segments
        .iter()
        .map(|&s| RasterQuery::from_stats(s))
        .collect();

    let gpu = raster.eval(&ctx, &queries, params);
    assert_parity(&segments, params, &gpu);
    assert_eq!(gpu[1], path_code(HairRasterPath::SubpixelSoftware));
    assert_eq!(gpu[2], path_code(HairRasterPath::ThickHardware));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_coverage_threshold_boundary_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping raster parity: no wgpu adapter on this host");
        return;
    };
    let raster = GpuHairRaster::new(&ctx);
    let params = HairRasterParams {
        min_coverage: 0.25,
        software_width_threshold: 2.0,
    };

    // Coverage equal to the minimum survives (`<` cull test); just below culls.
    let segments = [
        stats(0.5, 5.0, 0.25),
        stats(0.5, 5.0, 0.249),
        stats(0.5, 5.0, 0.250_001),
    ];
    let queries: Vec<RasterQuery> = segments
        .iter()
        .map(|&s| RasterQuery::from_stats(s))
        .collect();

    let gpu = raster.eval(&ctx, &queries, params);
    assert_parity(&segments, params, &gpu);
    assert_eq!(gpu[0], path_code(HairRasterPath::SubpixelSoftware));
    assert_eq!(gpu[1], path_code(HairRasterPath::Culled));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_negative_params_clamp_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping raster parity: no wgpu adapter on this host");
        return;
    };
    let raster = GpuHairRaster::new(&ctx);
    // Negative min_coverage acts as zero (only truly empty coverage culls);
    // negative width threshold acts as zero (nothing takes software).
    let params = HairRasterParams {
        min_coverage: -1.0,
        software_width_threshold: -1.0,
    };

    let segments = [
        stats(0.0, 5.0, 0.0), // clamped: width 0 <= 0 -> software, coverage 0 not < 0
        stats(0.1, 5.0, 0.5), // width 0.1 > 0 -> hardware
    ];
    let queries: Vec<RasterQuery> = segments
        .iter()
        .map(|&s| RasterQuery::from_stats(s))
        .collect();

    let gpu = raster.eval(&ctx, &queries, params);
    assert_parity(&segments, params, &gpu);
    assert_eq!(gpu[0], path_code(HairRasterPath::SubpixelSoftware));
    assert_eq!(gpu[1], path_code(HairRasterPath::ThickHardware));
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_large_mixed_batch_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping raster parity: no wgpu adapter on this host");
        return;
    };
    let raster = GpuHairRaster::new(&ctx);
    let params = HairRasterParams::default();

    // A batch spanning more than one workgroup, deterministically covering all
    // three paths via a polyline of widths/lengths/coverages (no transcendental
    // calls, so the construction stays exact and portable).
    let mut segments = Vec::new();
    for k in 0..200u32 {
        let width = (k % 10) as f32 * 0.5; // 0.0 .. 4.5
        let length = if k % 17 == 0 {
            0.0
        } else {
            4.0 + (k % 5) as f32
        };
        let coverage = if k % 23 == 0 {
            0.0
        } else {
            0.02 + (k % 7) as f32 * 0.1
        };
        segments.push(stats(width, length, coverage));
    }
    let queries: Vec<RasterQuery> = segments
        .iter()
        .map(|&s| RasterQuery::from_stats(s))
        .collect();

    let gpu = raster.eval(&ctx, &queries, params);
    assert_parity(&segments, params, &gpu);
    assert_eq!(
        distinct_paths(&gpu),
        3,
        "the large batch must exercise all three raster paths"
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_input_yields_empty_output() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping raster parity: no wgpu adapter on this host");
        return;
    };
    let raster = GpuHairRaster::new(&ctx);
    let gpu = raster.eval(&ctx, &[], HairRasterParams::default());
    assert!(gpu.is_empty(), "empty input must yield empty output");
}
