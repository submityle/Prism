//! Real-device parity for the `McGuire` `TileMax` / `NeighborMax` reductions.
//!
//! Each test builds a velocity field, runs the CPU golden
//! [`tile_max`]/[`neighbor_max`], runs the `GPU` kernels on a real adapter, and
//! asserts the two agree **bit-for-bit** (every component's `f32::to_bits`),
//! because the kernels only compare squared magnitudes (computed in the
//! golden's operation order) and copy whole velocity vectors. The suite skips
//! gracefully when no adapter is available so it still passes on a device-less
//! CI image, while running the full dispatch on a real `GPU`.

use prism_motion_gpu::context::GpuContext;
use prism_motion_gpu::tile::{GpuNeighborMax, GpuTileMax, TileField};
use prism_render_architecture::motion::dilation::{
    neighbor_max, tile_max, TileVelocityField, VelocityField,
};
use prism_render_architecture::motion::Vec2;

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Deterministic LCG mapped to `f32`, no transcendentals.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Lcg {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// Uniform-ish `f32` in `[0, 1)` from the high bits.
    fn unit(&mut self) -> f32 {
        let v = self.next_u32() >> 8; // 24 bits of entropy
        (v as f32) / (16_777_216.0_f32)
    }

    /// `f32` in `[-range, range)`.
    fn signed(&mut self, range: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * range
    }
}

fn bits_eq(a: Vec2, b: Vec2) -> bool {
    a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits()
}

fn field(width: usize, height: usize, values: &[(f32, f32)]) -> VelocityField {
    let data = values.iter().map(|&(x, y)| Vec2::new(x, y)).collect();
    VelocityField::from_pixels(width, height, data).expect("dimensions match")
}

/// Asserts a GPU tile field equals a golden tile field bit-for-bit.
fn assert_tile_parity(gpu: &TileField, golden: &TileVelocityField) {
    assert_eq!(gpu.tiles_x(), golden.tiles_x(), "tiles_x mismatch");
    assert_eq!(gpu.tiles_y(), golden.tiles_y(), "tiles_y mismatch");
    assert_eq!(gpu.len(), golden.len(), "tile count mismatch");
    for (i, (g, c)) in gpu
        .as_slice()
        .iter()
        .zip(golden.as_slice().iter())
        .enumerate()
    {
        assert!(bits_eq(*g, *c), "tile {i}: gpu {g:?} != golden {c:?}");
    }
}

/// Rebuilds the twin's own [`TileField`] from a golden tile field so the
/// `NeighborMax` kernel can consume it (the golden type has no public
/// constructor from parts).
fn tile_field_from_golden(golden: &TileVelocityField) -> TileField {
    TileField::from_parts(
        golden.tile_size(),
        golden.tiles_x(),
        golden.tiles_y(),
        golden.as_slice().to_vec(),
    )
    .expect("dimensions match the golden field")
}

#[test]
fn tile_max_single_tile_picks_largest_magnitude() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuTileMax::new(&ctx);

    // One 2x2 tile: the largest-magnitude vector (3,4 -> 25) must win.
    let vel = field(2, 2, &[(1.0, 1.0), (3.0, 4.0), (-2.0, 0.0), (0.0, -1.0)]);
    let golden = tile_max(&vel, 2).expect("non-empty");
    let gpu = kernel.reduce(&ctx, &vel, 2).expect("non-empty");
    assert_tile_parity(&gpu, &golden);
    // Anti-vacuous: the chosen tile velocity is the (3,4) pixel.
    assert!(bits_eq(gpu.get(0, 0).unwrap(), Vec2::new(3.0, 4.0)));
}

#[test]
fn tile_max_partial_edge_tiles_match_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuTileMax::new(&ctx);

    // 3x1 field, tile_size 2 -> 2 tiles, the second a partial (one pixel) tile.
    let vel = field(3, 1, &[(1.0, 0.0), (5.0, 0.0), (-2.0, 1.0)]);
    let golden = tile_max(&vel, 2).expect("non-empty");
    let gpu = kernel.reduce(&ctx, &vel, 2).expect("non-empty");
    assert_eq!(gpu.tiles_x(), 2);
    assert_eq!(gpu.tiles_y(), 1);
    assert_tile_parity(&gpu, &golden);
    // Anti-vacuous: tile 0 picks (5,0); the partial tile 1 holds (-2,1).
    assert!(bits_eq(gpu.get(0, 0).unwrap(), Vec2::new(5.0, 0.0)));
    assert!(bits_eq(gpu.get(1, 0).unwrap(), Vec2::new(-2.0, 1.0)));
}

#[test]
fn tile_max_zero_or_empty_returns_none() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuTileMax::new(&ctx);

    let vel = field(2, 2, &[(1.0, 1.0), (3.0, 4.0), (-2.0, 0.0), (0.0, -1.0)]);
    assert!(
        kernel.reduce(&ctx, &vel, 0).is_none(),
        "tile_size 0 -> None"
    );

    let empty = VelocityField::zeroed(0, 0);
    assert!(kernel.reduce(&ctx, &empty, 2).is_none(), "empty -> None");
}

#[test]
fn neighbor_max_spreads_across_3x3_and_respects_locality() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let tile_kernel = GpuTileMax::new(&ctx);
    let nbr_kernel = GpuNeighborMax::new(&ctx);

    // A 5x5 field, tile_size 1 -> 5x5 tiles so neighbor locality is exercised
    // directly on single-pixel tiles. One strong mover sits at the center.
    let mut vals = alloc::vec![(0.0f32, 0.0f32); 25];
    vals[2 * 5 + 2] = (9.0, 0.0); // center tile (2,2)
    vals[0] = (1.0, 0.0); // corner (0,0): outside the center's 3x3 reach
    let vel = field(5, 5, &vals);

    let golden_tiles = tile_max(&vel, 1).expect("non-empty");
    let golden_nbr = neighbor_max(&golden_tiles);

    let gpu_tiles = tile_kernel.reduce(&ctx, &vel, 1).expect("non-empty");
    assert_tile_parity(&gpu_tiles, &golden_tiles);

    let gpu_nbr = nbr_kernel.expand(&ctx, &gpu_tiles);
    assert_tile_parity(&gpu_nbr, &golden_nbr);

    // Anti-vacuous locality: the center spreads to its 8 neighbors (e.g. (1,1))
    // but not to the far corner (4,4), which stays zero.
    assert!(bits_eq(gpu_nbr.get(1, 1).unwrap(), Vec2::new(9.0, 0.0)));
    assert!(bits_eq(gpu_nbr.get(4, 4).unwrap(), Vec2::ZERO));
    // The corner (0,0) keeps its own small mover (no stronger neighbor).
    assert!(bits_eq(gpu_nbr.get(0, 0).unwrap(), Vec2::new(1.0, 0.0)));
}

#[test]
fn neighbor_max_from_reconstructed_tile_field() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let nbr_kernel = GpuNeighborMax::new(&ctx);

    // Build a golden tile field independently, reconstruct the twin's TileField
    // from it, and confirm the GPU NeighborMax matches golden on that input.
    let vel = field(6, 4, &{
        let mut rng = Lcg::new(0x00c0_ffee);
        (0..24)
            .map(|_| (rng.signed(8.0), rng.signed(8.0)))
            .collect::<Vec<_>>()
    });
    let golden_tiles = tile_max(&vel, 2).expect("non-empty");
    let golden_nbr = neighbor_max(&golden_tiles);

    let gpu_input = tile_field_from_golden(&golden_tiles);
    let gpu_nbr = nbr_kernel.expand(&ctx, &gpu_input);
    assert_tile_parity(&gpu_nbr, &golden_nbr);
}

#[test]
fn large_multi_workgroup_grid_matches_golden() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let tile_kernel = GpuTileMax::new(&ctx);
    let nbr_kernel = GpuNeighborMax::new(&ctx);

    // 40x24 pixels, tile_size 4 -> 10x6 tiles, spanning several 8x8 workgroups
    // in the full-res TileMax dispatch and exercising the coarse NeighborMax.
    let width = 40;
    let height = 24;
    let mut rng = Lcg::new(0x9e37_79b9);
    let vel_vals: Vec<(f32, f32)> = (0..width * height)
        .map(|_| (rng.signed(12.0), rng.signed(12.0)))
        .collect();
    let vel = field(width, height, &vel_vals);

    let golden_tiles = tile_max(&vel, 4).expect("non-empty");
    let golden_nbr = neighbor_max(&golden_tiles);

    let gpu_tiles = tile_kernel.reduce(&ctx, &vel, 4).expect("non-empty");
    assert_tile_parity(&gpu_tiles, &golden_tiles);

    let gpu_nbr = nbr_kernel.expand(&ctx, &gpu_tiles);
    assert_tile_parity(&gpu_nbr, &golden_nbr);

    // Anti-vacuous: some tile is non-zero, and NeighborMax changes at least one
    // tile relative to TileMax (a stronger neighbor bleeds in somewhere).
    assert!(
        gpu_tiles
            .as_slice()
            .iter()
            .any(|v| !bits_eq(*v, Vec2::ZERO)),
        "tile_max produced an all-zero field"
    );
    assert!(
        gpu_tiles
            .as_slice()
            .iter()
            .zip(gpu_nbr.as_slice().iter())
            .any(|(t, n)| !bits_eq(*t, *n)),
        "neighbor_max never changed any tile"
    );
}

extern crate alloc;
