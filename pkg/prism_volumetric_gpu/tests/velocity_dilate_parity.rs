//! Real-device parity for the velocity-dilation twin:
//! [`GpuVelocityDilate`](prism_volumetric_gpu::velocity_dilate::GpuVelocityDilate)
//! must reproduce the `CPU` golden
//! [`particle::velocity_dilate`](prism_render_architecture::particle::velocity_dilate)
//! across uniform, single-`tile`, non-square, random (multi-resolution), empty
//! and border-`tile` velocity fields.
//!
//! The tests skip when the host has no `wgpu` adapter, so the suite stays
//! green everywhere while still exercising the full dispatch-and-readback on
//! any real device such as an Apple `M`-series `GPU`. The kernels are portable core-`WGSL`, so they need no
//! optional device feature.
//!
//! # Parity criterion
//!
//! Both stages are a maximum-selection reduction that copies a surviving input
//! velocity verbatim, with no reorderable accumulation, so `CPU` and `GPU`
//! select the identical winner and copy identical bits in the common case. The
//! comparison allows `abs_diff <= 1e-5` or `rel_diff <= 1e-5` to stay robust
//! against a legal fused multiply-add in the squared-length comparison, yet
//! tight enough to fail a wrong port (a dropped border clamp, a transposed
//! index, a reversed tie rule). Each scenario independently checks `TileMax`,
//! `NeighborMax` (fed the `CPU` `tile`-max grid to isolate the stage) and the
//! composed `dominant_velocity`.
//!
//! Provenance: standard `McGuire` 2012 reconstruction-filter velocity dilation;
//! no Unreal Engine source or derived code.

use prism_render_architecture::particle::velocity_dilate::{
    dominant_velocity, neighbor_max, tile_max, Vel2, VelocityDilateConfig,
};
use prism_volumetric_gpu::velocity_dilate::GpuVelocityDilate;
use prism_volumetric_gpu::GpuContext;

/// Absolute/relative parity bound. A `GPU` may fuse a multiply-add the scalar
/// reference leaves separate in the squared-length comparison, perturbing a
/// near-tie by a few units in the last place; `1e-5` admits that legal slack
/// while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-5;

/// A small floor keeping the relative-error denominator away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= EPS
}

/// Asserts two velocity grids agree element- and component-wise.
fn assert_grid(gpu: &[Vel2], cpu: &[Vel2], what: &str) {
    assert_eq!(
        gpu.len(),
        cpu.len(),
        "{what}: grid length mismatch (gpu {}, cpu {})",
        gpu.len(),
        cpu.len()
    );
    for (idx, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert!(
            close(g.x, c.x) && close(g.y, c.y),
            "{what} mismatch at tile {idx}: gpu ({}, {}), cpu ({}, {})",
            g.x,
            g.y,
            c.x,
            c.y
        );
    }
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[-1, 1)`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1) then onto [-1, 1).
    let unit = (bits & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}

/// Builds a random per-pixel velocity field of `width * height` samples.
fn random_field(width: usize, height: usize, seed: u64) -> Vec<Vel2> {
    let mut state = seed;
    let mut data = Vec::with_capacity(width * height);
    for _ in 0..(width * height) {
        let x = lcg(&mut state);
        let y = lcg(&mut state);
        data.push(Vel2::new(x, y));
    }
    data
}

/// Runs the three `GPU` stages over `field`/`config` and asserts cell-by-cell
/// parity against the `CPU` reference for each stage.
fn check(ctx: &GpuContext, gpu: &GpuVelocityDilate, field: &[Vel2], config: VelocityDilateConfig) {
    let cpu_tiles = tile_max(field, config);
    let gpu_tiles = gpu.tile_max(ctx, field, config);
    assert_grid(&gpu_tiles, &cpu_tiles, "tile_max");

    let cols = config.tile_cols();
    let rows = config.tile_rows();
    // Feed both the identical CPU tile-max grid so the NeighborMax stage is
    // compared in isolation from any upstream difference.
    let cpu_neighbor = neighbor_max(&cpu_tiles, cols, rows);
    let gpu_neighbor = gpu.neighbor_max(ctx, &cpu_tiles, cols, rows);
    assert_grid(&gpu_neighbor, &cpu_neighbor, "neighbor_max");

    let cpu_dom = dominant_velocity(field, config);
    let gpu_dom = gpu.dominant_velocity(ctx, field, config);
    assert_grid(&gpu_dom, &cpu_dom, "dominant_velocity");
}

#[test]
fn gpu_matches_cpu_on_uniform_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVelocityDilate::new(&ctx);
    // A constant field dilates to the same velocity in every tile.
    let config = VelocityDilateConfig::new(40, 40, 20);
    let field = vec![Vel2::new(1.5, -2.0); config.width * config.height];
    check(&ctx, &gpu, &field, config);
}

#[test]
fn gpu_matches_cpu_on_single_tile_with_a_peak() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVelocityDilate::new(&ctx);
    // One 20px tile with a single fast pixel among slow ones: the peak must win.
    let config = VelocityDilateConfig::new(20, 20, 20);
    let mut field = vec![Vel2::new(0.1, 0.0); config.width * config.height];
    field[7 * config.width + 3] = Vel2::new(9.0, 0.0);
    check(&ctx, &gpu, &field, config);
}

#[test]
fn gpu_matches_cpu_on_non_square_non_divisible_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVelocityDilate::new(&ctx);
    // 45x30 with 20px tiles -> a 3x2 tile grid with trailing partial tiles on
    // both axes; a peak placed in the trailing partial column/row must survive.
    let config = VelocityDilateConfig::new(45, 30, 20);
    let mut field = random_field(config.width, config.height, 0x1234_5678_9abc_def0);
    field[29 * config.width + 44] = Vel2::new(0.0, 12.0);
    check(&ctx, &gpu, &field, config);
}

#[test]
fn gpu_matches_cpu_on_random_fields_multiple_resolutions() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVelocityDilate::new(&ctx);
    // A spread of resolutions and tile sizes (including non-square, a thin
    // strip, and a tile size that does not divide either dimension), each with
    // its own LCG-seeded random field.
    let cases: [(usize, usize, usize, u64); 4] = [
        (64, 64, 20, 0x0f0f_0f0f_1234_5678),
        (37, 51, 16, 0xdead_beef_cafe_babe),
        (100, 8, 20, 0x5555_aaaa_3333_cccc),
        (23, 23, 7, 0x9e37_79b9_7f4a_7c15),
    ];
    for (width, height, tile, seed) in cases {
        let config = VelocityDilateConfig::new(width, height, tile);
        let field = random_field(width, height, seed);
        check(&ctx, &gpu, &field, config);
    }
}

#[test]
fn gpu_matches_cpu_on_empty_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVelocityDilate::new(&ctx);
    // A zero-sized field has no tiles: every stage returns an empty grid.
    let config = VelocityDilateConfig::new(0, 0, 20);
    let field: Vec<Vel2> = Vec::new();
    assert_eq!(config.num_tiles(), 0);
    assert!(gpu.tile_max(&ctx, &field, config).is_empty());
    assert!(gpu
        .neighbor_max(&ctx, &[], config.tile_cols(), config.tile_rows())
        .is_empty());
    assert!(gpu.dominant_velocity(&ctx, &field, config).is_empty());
    check(&ctx, &gpu, &field, config);
}

#[test]
fn gpu_matches_cpu_on_border_tiles() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVelocityDilate::new(&ctx);
    // A multi-tile grid with a lone fast pixel near a corner exercises the
    // clamped 3x3 NeighborMax window: the border tile still surveys a full
    // in-bounds neighbourhood and the dilated peak must reach one tile out.
    let config = VelocityDilateConfig::new(60, 40, 20);
    let mut field = vec![Vel2::new(0.2, 0.2); config.width * config.height];
    field[config.width + 1] = Vel2::new(-7.0, 3.0);
    field[39 * config.width + 59] = Vel2::new(5.0, -6.0);
    check(&ctx, &gpu, &field, config);
}
