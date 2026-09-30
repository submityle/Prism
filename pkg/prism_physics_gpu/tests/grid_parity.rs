//! Real-device parity tests for the `GPU` bounded uniform grid.
//!
//! Each test builds the grid on the device and asserts bit-for-bit equality
//! with the [`cpu_grid_sort`] golden twin: the build is a pure integer
//! permutation (stable sort by cell) plus a range scan, identical on host and
//! device, so parity is exact rather than within a tolerance. Every test skips
//! cleanly when no adapter is available (for example inside a sandbox) so the
//! suite never fails for lack of a `GPU`.
//!
//! Provenance: exercises Prism's own hash and cell-ranges kernels (Green,
//! "Particle Simulation using CUDA", NVIDIA 2008) against their `CPU` twin. No
//! Unreal Engine source or derived code.

use std::time::Instant;

use glam::Vec3;
use prism_physics_gpu::{cpu_grid_sort, GpuContext, GpuUniformGrid, GridConfig};

/// A small xorshift generator so the tests are deterministic without pulling in
/// an `RNG` dependency.
struct Rng {
    /// Mutable generator state; never zero.
    state: u64,
}

impl Rng {
    /// Seeds the generator, forcing a non-zero state.
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    /// Advances the state and returns the next 64-bit value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Returns a `f32` in `[0, span)` offset by `lo`, quantised to a small grid
    /// so many particles deliberately share cells.
    fn coord(&mut self, lo: f32, span: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + unit * span
    }
}

/// Asserts a `GPU` grid build of `positions` equals the twin, bit-for-bit.
fn assert_build_matches(
    grid: &GpuUniformGrid,
    ctx: &GpuContext,
    positions: &[Vec3],
    config: &GridConfig,
) {
    let want = cpu_grid_sort(positions, config);
    let got = grid
        .build(ctx, positions, config)
        .expect("valid config builds");
    assert_eq!(
        got.sorted_indices,
        want.sorted_indices,
        "sorted index mismatch for {} particles",
        positions.len()
    );
    assert_eq!(got.cell_start, want.cell_start, "cell_start mismatch");
    assert_eq!(got.cell_end, want.cell_end, "cell_end mismatch");
}

#[test]
fn random_cloud_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let grid = GpuUniformGrid::new(&ctx);
    let config = GridConfig::new(Vec3::new(-4.0, -4.0, -4.0), 0.5, [16, 16, 16]);
    let mut rng = Rng::new(0xC0FF_EE01);
    let positions: Vec<Vec3> = (0..5_000)
        .map(|_| {
            Vec3::new(
                rng.coord(-4.0, 8.0),
                rng.coord(-4.0, 8.0),
                rng.coord(-4.0, 8.0),
            )
        })
        .collect();
    assert_build_matches(&grid, &ctx, &positions, &config);
}

#[test]
fn out_of_bounds_points_clamp_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let grid = GpuUniformGrid::new(&ctx);
    let config = GridConfig::new(Vec3::ZERO, 1.0, [8, 8, 8]);
    // Deliberately spill well outside the lattice on every axis.
    let mut rng = Rng::new(0xBEEF_0002);
    let positions: Vec<Vec3> = (0..2_000)
        .map(|_| {
            Vec3::new(
                rng.coord(-20.0, 40.0),
                rng.coord(-20.0, 40.0),
                rng.coord(-20.0, 40.0),
            )
        })
        .collect();
    assert_build_matches(&grid, &ctx, &positions, &config);
}

#[test]
fn many_empty_cells_keep_sentinels() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let grid = GpuUniformGrid::new(&ctx);
    // A large lattice with only a handful of occupied corner cells.
    let config = GridConfig::new(Vec3::ZERO, 1.0, [12, 12, 12]);
    let positions = [
        Vec3::new(0.1, 0.1, 0.1),
        Vec3::new(11.9, 11.9, 11.9),
        Vec3::new(0.2, 11.5, 0.3),
        Vec3::new(11.5, 0.2, 11.5),
    ];
    assert_build_matches(&grid, &ctx, &positions, &config);
}

#[test]
fn dense_shared_cell_is_stable() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let grid = GpuUniformGrid::new(&ctx);
    // A tiny lattice so thousands of particles collapse into very few cells,
    // stressing the stability of the sort and the range scan.
    let config = GridConfig::new(Vec3::ZERO, 1.0, [2, 2, 1]);
    let mut rng = Rng::new(0x5EED_0003);
    let positions: Vec<Vec3> = (0..4_000)
        .map(|_| Vec3::new(rng.coord(0.0, 2.0), rng.coord(0.0, 2.0), 0.5))
        .collect();
    assert_build_matches(&grid, &ctx, &positions, &config);
}

#[test]
fn single_particle_builds() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let grid = GpuUniformGrid::new(&ctx);
    let config = GridConfig::new(Vec3::ZERO, 1.0, [4, 4, 4]);
    assert_build_matches(&grid, &ctx, &[Vec3::new(2.5, 1.5, 3.5)], &config);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "reports the large-input timing to the test log"
)]
fn large_cloud_is_bit_exact() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let grid = GpuUniformGrid::new(&ctx);
    let config = GridConfig::new(Vec3::new(-16.0, -16.0, -16.0), 0.5, [64, 64, 64]);
    let mut rng = Rng::new(0x1234_ABCD);
    let positions: Vec<Vec3> = (0..100_000)
        .map(|_| {
            Vec3::new(
                rng.coord(-16.0, 32.0),
                rng.coord(-16.0, 32.0),
                rng.coord(-16.0, 32.0),
            )
        })
        .collect();

    let started = Instant::now();
    let got = grid
        .build(&ctx, &positions, &config)
        .expect("valid config builds");
    let elapsed = started.elapsed();

    let want = cpu_grid_sort(&positions, &config);
    assert_eq!(
        got.sorted_indices, want.sorted_indices,
        "large sorted mismatch"
    );
    assert_eq!(got.cell_start, want.cell_start, "large cell_start mismatch");
    assert_eq!(got.cell_end, want.cell_end, "large cell_end mismatch");
    eprintln!("100k particle grid build: {elapsed:?}");
}
