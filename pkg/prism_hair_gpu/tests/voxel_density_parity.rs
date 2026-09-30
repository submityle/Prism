//! Real-device parity for the froxel voxel density twin:
//! [`GpuHairVoxelDensity`] must reproduce the `CPU` golden
//! [`accumulate_voxel_density`](prism_render_architecture::hair::deep_transmittance::accumulate_voxel_density)
//! for a batch of per-texel strand buckets, covering the uniform equal-width
//! voxel binning, the input-order (sort-free) opacity scatter-add, the
//! out-of-slab skip (`depth < slab_start` or `depth >= slab_end`), the
//! degenerate zero-width slab, the `voxel_count` clamp to one, the last-voxel
//! index clamp, the empty-texel all-zero row, and the multi-texel batch walk.
//! Each accumulated density slab is additionally composited through
//! [`voxel_transmittance`](prism_render_architecture::hair::deep_transmittance::voxel_transmittance)
//! at every voxel index so a wrong per-voxel `sigma` could not pass.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The accumulation is a closed-form clamped sum with no transcendental call,
//! and each thread owns a disjoint density row, so the `CPU` and `GPU` evaluate
//! the same expressions and diverge only through legal fused-multiply-add
//! contraction. Parity is asserted to within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3` — tight enough to fail a genuinely wrong port (a missing
//! skip, a wrong voxel index, a dropped clamp), loose enough to admit the
//! contraction. Samples are built from explicit depth/opacity literals (never
//! `sin`/`cos`), and the shaping cases assert a non-trivial occlusion so a
//! no-op kernel could not pass.
//!
//! Provenance: standard uniform-voxel opacity accumulation plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::voxel_density::GpuHairVoxelDensity;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::deep_transmittance::{
    accumulate_voxel_density, bin_samples, voxel_transmittance, TexelSample, TransmittanceBins,
    TransmittanceSample,
};

/// Asserts a single value matches within the documented tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got} vs cpu {expected} (abs {abs_diff}, rel {rel_diff})",
    );
}

/// Builds a bin set from `(texel, depth, opacity)` triples.
fn bins_from(entries: &[(u32, f32, f32)], texel_count: u32) -> TransmittanceBins {
    let samples: Vec<TexelSample> = entries
        .iter()
        .map(|&(texel, depth, opacity)| TexelSample {
            texel,
            sample: TransmittanceSample::new(depth, opacity),
        })
        .collect();
    bin_samples(&samples, texel_count)
}

/// Runs the twin and the golden on the same `bins`, asserts every per-texel
/// density slab matches value-for-value, then composites both slabs through
/// [`voxel_transmittance`] at every voxel index so a wrong density could not
/// hide. Returns the `GPU` slabs for further case-specific assertions.
fn assert_parity(
    ctx: &GpuContext,
    packer: &GpuHairVoxelDensity,
    bins: &TransmittanceBins,
    slab_start: f32,
    slab_end: f32,
    voxel_count: u32,
) -> Vec<Vec<f32>> {
    let gpu = packer.eval(ctx, bins, slab_start, slab_end, voxel_count);

    assert_eq!(gpu.len(), bins.len(), "texel row count");

    for (texel, row) in gpu.iter().enumerate() {
        let bucket = bins.bucket(texel as u32).unwrap_or(&[]);
        let cpu = accumulate_voxel_density(bucket, slab_start, slab_end, voxel_count);
        assert_eq!(row.len(), cpu.len(), "texel {texel} voxel count");

        for (v, (g, c)) in row.iter().zip(&cpu).enumerate() {
            assert_close(*g, *c, &format!("density texel {texel} voxel {v}"));
        }

        // Composite both slabs at every index: a wrong per-voxel sigma would
        // agree nowhere here even if the raw sum slipped through the tolerance.
        for index in 0..cpu.len() {
            let g = voxel_transmittance(row, index);
            let c = voxel_transmittance(&cpu, index);
            assert_close(g, c, &format!("transmittance texel {texel} index {index}"));
        }
    }

    gpu
}

/// A single texel with three occluders spread across a five-voxel slab: each
/// sample lands in its own voxel, so the density row is a direct histogram and
/// the composited transmittance steps down at each occupied voxel.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_texel_histogram_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel density parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairVoxelDensity::new(&ctx);

    // Slab 0..10, five voxels of width 2: depths 1, 5, 9 land in voxels 0, 2, 4.
    let bins = bins_from(&[(0, 1.0, 0.3), (0, 5.0, 0.5), (0, 9.0, 0.7)], 1);
    let gpu = assert_parity(&ctx, &packer, &bins, 0.0, 10.0, 5);

    let row = &gpu[0];
    assert_close(row[0], 0.3, "voxel 0 holds the shallow occluder");
    assert_close(row[1], 0.0, "voxel 1 is empty");
    assert_close(row[2], 0.5, "voxel 2 holds the mid occluder");
    assert_close(row[4], 0.7, "voxel 4 holds the deep occluder");
}

/// Samples outside the slab (`depth < slab_start` or `depth >= slab_end`) are
/// skipped rather than clamped into the boundary voxels, so a neighbouring
/// slab's occluders never bleed in.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_out_of_slab_samples_are_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel density parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairVoxelDensity::new(&ctx);

    // Slab 2..6, four voxels of width 1. Depth 1.0 is before the slab, depth
    // 6.0 is exactly the exclusive end, both skipped; 3.5 lands in voxel 1.
    let bins = bins_from(&[(0, 1.0, 0.9), (0, 6.0, 0.9), (0, 3.5, 0.4)], 1);
    let gpu = assert_parity(&ctx, &packer, &bins, 2.0, 6.0, 4);

    let row = &gpu[0];
    assert_close(row[0], 0.0, "before-slab sample skipped");
    assert_close(row[1], 0.4, "in-slab sample kept");
    assert_close(row[3], 0.0, "end-boundary sample skipped");
}

/// A degenerate slab (`slab_end <= slab_start`) skips every sample and leaves an
/// all-zero density row, so its transmittance is a full `1.0`.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_degenerate_slab_is_all_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel density parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairVoxelDensity::new(&ctx);

    let bins = bins_from(&[(0, 3.0, 0.8), (0, 4.0, 0.6)], 1);
    let gpu = assert_parity(&ctx, &packer, &bins, 5.0, 5.0, 4);

    for (v, d) in gpu[0].iter().enumerate() {
        assert_close(*d, 0.0, &format!("degenerate slab voxel {v} stays zero"));
    }
    assert_close(voxel_transmittance(&gpu[0], 3), 1.0, "empty slab fully lit");
}

/// A `voxel_count` of zero clamps to a single voxel that absorbs every in-slab
/// occluder.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_voxel_count_clamps_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel density parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairVoxelDensity::new(&ctx);

    let bins = bins_from(&[(0, 1.0, 0.3), (0, 4.0, 0.4)], 1);
    let gpu = assert_parity(&ctx, &packer, &bins, 0.0, 8.0, 0);

    assert_eq!(gpu[0].len(), 1, "zero voxel count clamps to one");
    assert_close(gpu[0][0], 0.7, "single voxel sums both occluders");
}

/// Several samples in the same voxel accumulate their clamped opacity in input
/// order, and a sample at the very end of the slab clamps into the last voxel
/// rather than overflowing.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_same_voxel_accumulates_and_last_voxel_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel density parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairVoxelDensity::new(&ctx);

    // Slab 0..4, four voxels of width 1. Depths 0.1/0.2/0.9 all fall in voxel 0
    // (0.3 + 0.2 + 0.4). Depth 3.999 rounds to voxel 3 (the last voxel).
    let bins = bins_from(
        &[(0, 0.1, 0.3), (0, 0.2, 0.2), (0, 0.9, 0.4), (0, 3.999, 0.5)],
        1,
    );
    let gpu = assert_parity(&ctx, &packer, &bins, 0.0, 4.0, 4);

    let row = &gpu[0];
    assert_close(row[0], 0.9, "voxel 0 sums the three shallow occluders");
    assert_close(row[3], 0.5, "near-end sample clamps into the last voxel");
}

/// A multi-texel batch with an empty middle texel exercises the per-texel base
/// offset, the one-thread-per-texel dispatch bound, and the all-zero empty row,
/// all in one dispatch.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_texel_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping voxel density parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairVoxelDensity::new(&ctx);

    // Texel 0 occupied, texel 1 empty, texel 2 with two occluders, texel 3 one.
    let bins = bins_from(
        &[
            (0, 0.5, 0.3),
            (0, 2.5, 0.4),
            (2, 1.0, 0.6),
            (2, 3.5, 0.5),
            (3, 2.0, 0.9),
        ],
        4,
    );
    let gpu = assert_parity(&ctx, &packer, &bins, 0.0, 4.0, 4);

    assert_eq!(gpu.len(), 4, "batch covers every texel");
    for (v, d) in gpu[1].iter().enumerate() {
        assert_close(*d, 0.0, &format!("empty texel 1 voxel {v} stays zero"));
    }
}
