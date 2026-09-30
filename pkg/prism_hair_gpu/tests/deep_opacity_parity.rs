//! Real-device parity for the deep opacity packing twin:
//! [`GpuHairDeepOpacity`] must reproduce the `CPU` golden
//! [`build_deep_opacity_map`](prism_render_architecture::hair::deep_opacity_layout::build_deep_opacity_map)
//! for a batch of per-texel strand buckets, covering the stable per-texel depth
//! sort (verified with shuffled input), the fixed equal-width layer slicing,
//! the multiplicative `alpha`-composite `T = product(1 - alpha)`, the near-bias
//! `start_offset`, the degenerate zero-width slab, the `layer_count` clamp, the
//! empty-texel fully transmissive row, and the multi-texel batch walk. The
//! packed slab is additionally decoded through
//! [`map_transmittance`](prism_render_architecture::hair::deep_opacity_layout::map_transmittance)
//! at a range of probe depths so a wrong near/step could not pass.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The composite is a closed-form running product with no transcendental call,
//! so the `CPU` and `GPU` evaluate the same expressions and diverge only
//! through legal fused-multiply-add contraction. Parity is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong port (a swapped factor, a wrong boundary, a missing host sort), loose
//! enough to admit the contraction. Samples are built from explicit
//! depth/opacity literals (never `sin`/`cos`), and the shaping cases assert a
//! non-trivial occlusion so a no-op kernel could not pass.
//!
//! Provenance: standard deep opacity map packing plus `wgpu` compute dispatch;
//! no Unreal Engine source or derived code.

use prism_hair_gpu::deep_opacity::GpuHairDeepOpacity;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::deep_opacity_layout::{
    build_deep_opacity_map, map_transmittance, DeepOpacityMap,
};
use prism_render_architecture::hair::deep_transmittance::{
    bin_samples, TexelSample, TransmittanceBins, TransmittanceSample,
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

/// Runs the twin and the golden on the same `bins`, asserts the packed slab
/// (near/step/transmittance) matches value-for-value plus a decode sweep, and
/// returns the `GPU` map for further case-specific assertions.
fn assert_parity(
    ctx: &GpuContext,
    packer: &GpuHairDeepOpacity,
    bins: &TransmittanceBins,
    layer_count: u32,
    start_offset: f32,
) -> DeepOpacityMap {
    let gpu = packer.eval(ctx, bins, layer_count, start_offset);
    let cpu = build_deep_opacity_map(bins, layer_count, start_offset);

    assert_eq!(gpu.texel_count, cpu.texel_count, "texel_count");
    assert_eq!(gpu.layer_count, cpu.layer_count, "layer_count");
    assert_eq!(
        gpu.transmittance.len(),
        cpu.transmittance.len(),
        "transmittance length",
    );
    assert_eq!(gpu.near_depth.len(), cpu.near_depth.len(), "near length");
    assert_eq!(gpu.layer_step.len(), cpu.layer_step.len(), "step length");

    for (t, (g, c)) in gpu.near_depth.iter().zip(&cpu.near_depth).enumerate() {
        assert_close(*g, *c, &format!("near_depth[{t}]"));
    }
    for (t, (g, c)) in gpu.layer_step.iter().zip(&cpu.layer_step).enumerate() {
        assert_close(*g, *c, &format!("layer_step[{t}]"));
    }
    for (i, (g, c)) in gpu.transmittance.iter().zip(&cpu.transmittance).enumerate() {
        assert_close(*g, *c, &format!("transmittance[{i}]"));
    }

    // Decode both maps at a depth sweep on every texel: a wrong near/step would
    // agree on the raw layers but diverge here.
    for texel in 0..gpu.texel_count {
        for step in 0..24 {
            let depth = step as f32 * 0.6;
            let g = map_transmittance(&gpu, texel, depth);
            let c = map_transmittance(&cpu, texel, depth);
            assert_close(
                g,
                c,
                &format!("map_transmittance texel {texel} depth {depth}"),
            );
        }
    }

    gpu
}

/// Single texel, two half-opacity occluders, two layers: the composite must
/// halve then quarter the light (mirrors the golden), proving the running
/// product fans out on device.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_texel_layers_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep opacity parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairDeepOpacity::new(&ctx);

    let bins = bins_from(&[(0, 0.0, 0.5), (0, 10.0, 0.5)], 1);
    let gpu = assert_parity(&ctx, &packer, &bins, 2, 0.0);

    let row = gpu.layers(0).expect("texel 0");
    assert_close(row[0], 0.5, "layer 0 halves the light");
    assert_close(row[1], 0.25, "layer 1 quarters the light");
}

/// The host must stably sort each texel's samples before upload: shuffled input
/// must pack the identical slab as sorted input.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_packing_is_order_independent() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep opacity parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairDeepOpacity::new(&ctx);

    let ordered = bins_from(&[(0, 0.0, 0.2), (0, 4.0, 0.3), (0, 9.0, 0.4)], 1);
    let shuffled = bins_from(&[(0, 9.0, 0.4), (0, 0.0, 0.2), (0, 4.0, 0.3)], 1);

    let a = assert_parity(&ctx, &packer, &ordered, 3, 0.0);
    let b = assert_parity(&ctx, &packer, &shuffled, 3, 0.0);
    for (x, y) in a.transmittance.iter().zip(&b.transmittance) {
        assert_close(*x, *y, "shuffled vs ordered transmittance");
    }
}

/// A texel with no samples packs a fully transmissive row with zero near/step;
/// the batch also carries occupied texels so both paths run in one dispatch.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_texel_is_fully_transmissive() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep opacity parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairDeepOpacity::new(&ctx);

    // Texel 0 occupied, texel 1 empty, texel 2 single opaque occluder.
    let bins = bins_from(&[(0, 0.0, 0.5), (0, 6.0, 0.5), (2, 3.0, 1.0)], 3);
    let gpu = assert_parity(&ctx, &packer, &bins, 2, 0.0);

    let row1 = gpu.layers(1).expect("texel 1");
    assert_close(row1[0], 1.0, "empty texel layer 0 fully lit");
    assert_close(row1[1], 1.0, "empty texel layer 1 fully lit");
    assert_close(gpu.near_depth[1], 0.0, "empty texel near is zero");
    assert_close(gpu.layer_step[1], 0.0, "empty texel step is zero");

    let row2 = gpu.layers(2).expect("texel 2");
    assert_close(row2[1], 0.0, "opaque occluder fully blocks");
}

/// A positive `start_offset` biases the first layer's front off the shallowest
/// sample; the packed near/step and decode must still track the golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_start_offset_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep opacity parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairDeepOpacity::new(&ctx);

    let bins = bins_from(&[(0, 1.0, 0.3), (0, 3.0, 0.4), (0, 7.0, 0.5)], 1);
    let gpu = assert_parity(&ctx, &packer, &bins, 4, 0.5);
    assert_close(gpu.near_depth[0], 1.5, "near = shallowest + start_offset");
}

/// Every sample at one depth is a degenerate zero-width slab: all boundaries
/// pin to `end`, so the whole row collapses to the full product.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_width_slab_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep opacity parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairDeepOpacity::new(&ctx);

    let bins = bins_from(&[(0, 5.0, 0.5), (0, 5.0, 0.5)], 1);
    let gpu = assert_parity(&ctx, &packer, &bins, 3, 0.0);
    assert_close(gpu.layer_step[0], 0.0, "zero-width slab step is zero");
    let row = gpu.layers(0).expect("texel 0");
    // Both occluders composite into every layer: 0.5 * 0.5 = 0.25.
    assert_close(row[2], 0.25, "degenerate slab composites both occluders");
}

/// `layer_count` of zero clamps to one; the single layer holds the full product.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_layer_count_clamps_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep opacity parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairDeepOpacity::new(&ctx);

    let bins = bins_from(&[(0, 2.0, 0.5)], 1);
    let gpu = assert_parity(&ctx, &packer, &bins, 0, 0.0);
    assert_eq!(gpu.layer_count, 1, "zero layer count clamps to one");
    assert_eq!(gpu.transmittance.len(), 1, "one texel, one layer");
    assert_close(gpu.transmittance[0], 0.5, "single layer holds the product");
}

/// A multi-texel batch with distinct depth spans exercises the per-texel base
/// offset and the one-thread-per-texel dispatch bound.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_texel_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping deep opacity parity: no wgpu adapter on this host");
        return;
    };
    let packer = GpuHairDeepOpacity::new(&ctx);

    let bins = bins_from(
        &[
            (0, 0.0, 0.3),
            (0, 2.0, 0.4),
            (1, 5.0, 0.6),
            (1, 8.0, 0.2),
            (1, 11.0, 0.5),
            (3, 1.0, 0.9),
            (3, 1.5, 0.9),
        ],
        4,
    );
    let gpu = assert_parity(&ctx, &packer, &bins, 3, 0.25);
    assert_eq!(gpu.texel_count, 4, "batch covers every texel");
    // Texel 2 had no samples: fully transmissive.
    let row2 = gpu.layers(2).expect("texel 2");
    for (l, v) in row2.iter().enumerate() {
        assert_close(*v, 1.0, &format!("empty texel 2 layer {l}"));
    }
}
