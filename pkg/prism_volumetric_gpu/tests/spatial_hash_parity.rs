//! Real-device parity for the spatial-hash build twin:
//! [`GpuSpatialHash`](prism_volumetric_gpu::spatial_hash::GpuSpatialHash) must
//! reproduce the `CPU` golden
//! [`spatial_hash`](prism_render_architecture::particle::spatial_hash) element
//! for element across both twinned kernels.
//!
//! The `hash_cell` fixtures cover the empty input (no dispatch), the
//! `table_size == 0` early return, negative and `i32::MIN`/`i32::MAX` extreme
//! coordinates (the wrapping-multiply path), a collision-concentrated set
//! (coordinates that fold into a tiny table), a uniform scatter and a large
//! pseudo-random batch. The `count_cells` fixtures cover the empty input, the
//! zero-cell grid, out-of-range indices (skipped on both paths), a hand-checked
//! distribution and a large pseudo-random batch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Both kernels are pure integer arithmetic with no rounding anywhere, so `CPU`
//! and `GPU` must agree bit for bit. The comparison is an exact `==` on every
//! hash slot and every bucket count, with no tolerance: any mismatch is a
//! genuine port bug (a wrong prime, a dropped `xor`, a miscounted bucket). In
//! particular the fixtures exercise the `i32::wrapping_mul` + `from_ne_bytes`
//! wrap semantics against `WGSL`'s native `u32` wrap at `i32::MIN`/`i32::MAX`.
//! `WGSL` has no `u64`, so the host-side `UniformGrid::cell_count` `u64`
//! overflow guard is out of scope here.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::spatial_hash`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::spatial_hash::{count_cells, hash_cell};
use prism_volumetric_gpu::spatial_hash::GpuSpatialHash;
use prism_volumetric_gpu::GpuContext;

/// A deterministic linear-congruential generator so the randomized fixtures are
/// reproducible bit for bit across runs and platforms.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Runs the `GPU` hash kernel and asserts exact per-cell parity against the
/// `CPU` golden [`hash_cell`], returning the slots for extra assertions.
fn check_hash(
    ctx: &GpuContext,
    gpu: &GpuSpatialHash,
    cells: &[[i32; 3]],
    table_size: u32,
) -> Vec<u32> {
    let got = gpu.hash_cells(ctx, cells, table_size);
    assert_eq!(got.len(), cells.len(), "hash output length must match input");
    for (i, (&g, cell)) in got.iter().zip(cells.iter()).enumerate() {
        let want = hash_cell(*cell, table_size);
        assert_eq!(
            g, want,
            "cell {i} {cell:?}: gpu slot {g} vs cpu slot {want} (table {table_size})"
        );
    }
    got
}

/// Runs the `GPU` count kernel and asserts exact per-bucket parity against the
/// `CPU` golden [`count_cells`], returning the counts for extra assertions.
fn check_count(
    ctx: &GpuContext,
    gpu: &GpuSpatialHash,
    indices: &[u32],
    cell_count: u32,
) -> Vec<u32> {
    let got = gpu.count_cells(ctx, indices, cell_count);
    let want = count_cells(indices, cell_count);
    assert_eq!(got.len(), want.len(), "count output length must match cell_count");
    for (cell, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g, w,
            "cell {cell}: gpu count {g} vs cpu count {w} (cells {cell_count})"
        );
    }
    got
}

#[test]
fn hash_empty_input_is_the_empty_vector() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    let got = check_hash(&ctx, &gpu, &[], 1024);
    assert!(got.is_empty(), "an empty cell batch yields no slots");
}

#[test]
fn hash_zero_table_is_all_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    // A zero-sized table matches the golden early return of 0 for every cell.
    let cells = [[9, 9, 9], [-3, 7, -12], [0, 0, 0], [i32::MIN, 0, i32::MAX]];
    let got = check_hash(&ctx, &gpu, &cells, 0);
    assert!(got.iter().all(|&h| h == 0), "a zero table hashes to all zero");
}

#[test]
fn hash_negative_and_extreme_coordinates_wrap_identically() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    // The wrapping-multiply path at the i32 boundaries must fold bit-for-bit the
    // same on both devices.
    let cells = [
        [-1, -1, -1],
        [-7, 3, -12],
        [i32::MIN, i32::MAX, i32::MIN],
        [i32::MAX, i32::MIN, i32::MAX],
        [i32::MIN, i32::MIN, i32::MIN],
        [-2_147_483_648, 123_456, -987_654],
    ];
    let got = check_hash(&ctx, &gpu, &cells, 97);
    assert!(got.iter().all(|&h| h < 97), "every slot is inside the table");
}

#[test]
fn hash_collisions_concentrate_into_a_tiny_table() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    // A tiny table forces heavy aliasing; parity holds slot-for-slot regardless
    // of how many distinct cells collide.
    let mut cells = Vec::new();
    for x in -8..8 {
        for y in -8..8 {
            cells.push([x, y, 0]);
        }
    }
    let got = check_hash(&ctx, &gpu, &cells, 4);
    assert!(got.iter().all(|&h| h < 4), "every slot is inside the tiny table");
}

#[test]
fn hash_uniform_scatter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    // A regular 3-D lattice spread across a mid-sized table.
    let mut cells = Vec::new();
    for z in 0..6 {
        for y in 0..6 {
            for x in 0..6 {
                cells.push([x * 3 - 7, y * 5 + 2, z * 2 - 4]);
            }
        }
    }
    let got = check_hash(&ctx, &gpu, &cells, 1024);
    assert!(got.iter().all(|&h| h < 1024), "every slot is inside the table");
}

#[test]
fn hash_large_random_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    let mut state = 0x0123_4567_89AB_CDEF_u64;
    let mut cells = Vec::with_capacity(4096);
    for _ in 0..4096 {
        let x = lcg_next(&mut state) as u32 as i32;
        let y = lcg_next(&mut state) as u32 as i32;
        let z = lcg_next(&mut state) as u32 as i32;
        cells.push([x, y, z]);
    }
    // An odd, non-power-of-two table so the modulus path is non-trivial.
    let got = check_hash(&ctx, &gpu, &cells, 1_000_003);
    assert!(
        got.iter().all(|&h| h < 1_000_003),
        "every slot is inside the table"
    );
}

#[test]
fn count_zero_cells_is_the_empty_vector() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    let got = check_count(&ctx, &gpu, &[0, 1, 2], 0);
    assert!(got.is_empty(), "a zero-cell grid has no buckets");
}

#[test]
fn count_empty_input_is_the_zero_histogram() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    let got = check_count(&ctx, &gpu, &[], 6);
    assert_eq!(got, vec![0u32; 6], "no entry leaves every bucket empty");
}

#[test]
fn count_skips_out_of_range_indices() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    // Indices at or beyond the cell count are dropped on both paths.
    let got = check_count(&ctx, &gpu, &[0, 7, 2, 3, 100, 2], 3);
    assert_eq!(got, vec![1, 0, 2], "only in-range indices are counted");
}

#[test]
fn count_distributes_hits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    let got = check_count(&ctx, &gpu, &[0, 2, 2, 5], 6);
    assert_eq!(got, vec![1, 0, 2, 0, 0, 1], "hits land in the hand-checked cells");
}

#[test]
fn count_large_random_batch_conserves_totals() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSpatialHash::new(&ctx);
    let cell_count = 257u32;
    let mut state = 0x5EED_4A7D_0BAD_C0DE_u64;
    let mut indices = Vec::with_capacity(8192);
    for _ in 0..8192 {
        // A mix of in-range and (occasionally) out-of-range indices, so the
        // skip path is exercised alongside the atomic accumulation.
        indices.push((lcg_next(&mut state) >> 33) as u32 % (cell_count + 16));
    }
    let got = check_count(&ctx, &gpu, &indices, cell_count);
    let in_range = indices.iter().filter(|&&i| i < cell_count).count() as u64;
    assert_eq!(
        got.iter().map(|&c| u64::from(c)).sum::<u64>(),
        in_range,
        "every in-range entry is counted exactly once"
    );
    assert!(got.iter().any(|&c| c > 0), "the histogram is not vacuously empty");
}
