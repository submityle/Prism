//! Real-device parity for the per-texel sample-binning twin:
//! [`GpuHairBinSamples`] must reproduce the `CPU` golden
//! [`bin_samples`](prism_render_architecture::hair::deep_transmittance::bin_samples)
//! for an indexed stream of transmittance samples — routing every sample into
//! its light texel's bucket, preserving input order inside each bucket, and
//! silently skipping out-of-range tags. The golden
//! [`bin_samples`](prism_render_architecture::hair::deep_transmittance::bin_samples)
//! builds the reference buckets and the twin's rows are asserted member-for-
//! member against [`TransmittanceBins::bucket`], so the parity covers the exact
//! routing (which sample lands where) and the stable per-bucket order.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL
//! (compare-and-copy only), so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel performs no arithmetic — it only compares texel tags and copies
//! `(depth, opacity)` payloads — so the `CPU` and `GPU` produce bit-identical
//! bucket contents. Parity is nonetheless asserted through a tight numeric
//! tolerance helper (`abs_diff < 1e-4` or `rel_diff < 1e-3`) to stay uniform
//! with the sibling twins; the ordering and bucket-length checks would fail any
//! mis-routing outright. Sample depths/opacities are explicit literals (never
//! `sin`/`cos`), and the cases assert non-trivial multi-sample buckets so a
//! no-op kernel could not pass.
//!
//! Provenance: standard per-bucket scatter/gather partition plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::bin_samples::GpuHairBinSamples;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::deep_transmittance::{
    bin_samples, TexelSample, TransmittanceSample,
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

/// Builds one indexed sample from `(texel, depth, opacity)` literals.
fn indexed(texel: u32, depth: f32, opacity: f32) -> TexelSample {
    TexelSample {
        texel,
        sample: TransmittanceSample::new(depth, opacity),
    }
}

/// Runs the twin and the golden on the same stream, asserting every emitted
/// bucket equals [`TransmittanceBins::bucket`] member-for-member and in order.
/// Returns the `GPU` buckets for further case-specific assertions.
fn assert_parity(
    ctx: &GpuContext,
    binner: &GpuHairBinSamples,
    samples: &[TexelSample],
    texel_count: u32,
) -> Vec<Vec<[f32; 2]>> {
    let gpu = binner.eval(ctx, samples, texel_count);
    let cpu = bin_samples(samples, texel_count);

    assert_eq!(gpu.len(), cpu.len(), "bucket count");

    for (t, bucket) in gpu.iter().enumerate() {
        let reference = cpu
            .bucket(t as u32)
            .expect("golden bucket must exist for every texel");
        assert_eq!(bucket.len(), reference.len(), "texel {t} bucket length");
        for (k, (row, sample)) in bucket.iter().zip(reference).enumerate() {
            assert_close(row[0], sample.depth, &format!("texel {t} sample {k} depth"));
            assert_close(
                row[1],
                sample.opacity,
                &format!("texel {t} sample {k} opacity"),
            );
        }
    }

    gpu
}

/// Several samples routed to a single texel must land in the same bucket in
/// input order.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_texel_preserves_input_order() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bin_samples parity: no wgpu adapter on this host");
        return;
    };
    let binner = GpuHairBinSamples::new(&ctx);

    let samples = [
        indexed(0, 3.0, 0.5),
        indexed(0, 1.0, 0.25),
        indexed(0, 2.0, 0.75),
    ];
    let gpu = assert_parity(&ctx, &binner, &samples, 1);

    assert_eq!(gpu.len(), 1, "single bucket");
    assert_eq!(gpu[0].len(), 3, "all three samples routed");
    // Bucket order is the input order, not sorted by depth.
    assert_close(gpu[0][0][0], 3.0, "first sample kept first");
    assert_close(gpu[0][1][0], 1.0, "second sample kept second");
    assert_close(gpu[0][2][0], 2.0, "third sample kept third");
}

/// Samples tagged with distinct texels fan out into distinct buckets.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_multi_texel_routing_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bin_samples parity: no wgpu adapter on this host");
        return;
    };
    let binner = GpuHairBinSamples::new(&ctx);

    let samples = [
        indexed(2, 1.0, 0.4),
        indexed(0, 2.0, 0.6),
        indexed(1, 3.0, 0.2),
        indexed(2, 4.0, 0.8),
        indexed(0, 5.0, 0.1),
    ];
    let gpu = assert_parity(&ctx, &binner, &samples, 3);

    assert_eq!(gpu.len(), 3, "three buckets");
    assert_eq!(gpu[0].len(), 2, "texel 0 got two samples");
    assert_eq!(gpu[1].len(), 1, "texel 1 got one sample");
    assert_eq!(gpu[2].len(), 2, "texel 2 got two samples");
    // Texel 2 keeps its two samples in encounter order (depths 1.0 then 4.0).
    assert_close(gpu[2][0][0], 1.0, "texel 2 first in order");
    assert_close(gpu[2][1][0], 4.0, "texel 2 second in order");
}

/// A sample whose texel index is out of range is skipped, not routed anywhere
/// and never crashes.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_out_of_range_texel_is_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bin_samples parity: no wgpu adapter on this host");
        return;
    };
    let binner = GpuHairBinSamples::new(&ctx);

    let samples = [
        indexed(0, 1.0, 0.5),
        indexed(5, 2.0, 0.5), // out of range for texel_count 2
        indexed(1, 3.0, 0.5),
        indexed(9, 4.0, 0.5), // out of range too
    ];
    let gpu = assert_parity(&ctx, &binner, &samples, 2);

    assert_eq!(gpu.len(), 2, "two in-range buckets");
    assert_eq!(gpu[0].len(), 1, "texel 0 kept its sample");
    assert_eq!(gpu[1].len(), 1, "texel 1 kept its sample");
    let total: usize = gpu.iter().map(Vec::len).sum();
    assert_eq!(total, 2, "out-of-range samples dropped");
}

/// A zero texel count is clamped to one bucket (matching `TransmittanceBins::new`),
/// which then collects only the in-range (texel 0) samples.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_texel_count_clamps_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bin_samples parity: no wgpu adapter on this host");
        return;
    };
    let binner = GpuHairBinSamples::new(&ctx);

    let samples = [
        indexed(0, 1.0, 0.3),
        indexed(1, 2.0, 0.3), // out of range once clamped to a single bucket
        indexed(0, 3.0, 0.3),
    ];
    let gpu = assert_parity(&ctx, &binner, &samples, 0);

    assert_eq!(gpu.len(), 1, "clamped to a single bucket");
    assert_eq!(gpu[0].len(), 2, "only texel 0 samples land");
}

/// An empty sample stream yields empty buckets without a panic or a zero-sized
/// buffer.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_empty_stream_yields_empty_buckets() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bin_samples parity: no wgpu adapter on this host");
        return;
    };
    let binner = GpuHairBinSamples::new(&ctx);

    let gpu = assert_parity(&ctx, &binner, &[], 3);

    assert_eq!(gpu.len(), 3, "three empty buckets");
    assert!(gpu.iter().all(Vec::is_empty), "no sample routed");
}

/// Shuffled texel tags must still produce stable per-bucket order (encounter
/// order within each texel), and a stream longer than the bucket count spreads
/// across every texel.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_shuffled_stream_stable_bucket_order() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping bin_samples parity: no wgpu adapter on this host");
        return;
    };
    let binner = GpuHairBinSamples::new(&ctx);

    // 8 samples across 3 texels, interleaved so encounter order != sorted order.
    let samples = [
        indexed(1, 10.0, 0.1),
        indexed(0, 11.0, 0.2),
        indexed(2, 12.0, 0.3),
        indexed(1, 13.0, 0.4),
        indexed(0, 14.0, 0.5),
        indexed(2, 15.0, 0.6),
        indexed(1, 16.0, 0.7),
        indexed(0, 17.0, 0.8),
    ];
    let gpu = assert_parity(&ctx, &binner, &samples, 3);

    assert_eq!(gpu.len(), 3, "three buckets");
    assert_eq!(gpu[0].len(), 3, "texel 0 got three");
    assert_eq!(gpu[1].len(), 3, "texel 1 got three");
    assert_eq!(gpu[2].len(), 2, "texel 2 got two");
    // Texel 1 encounter order: depths 10, 13, 16.
    assert_close(gpu[1][0][0], 10.0, "texel 1 first");
    assert_close(gpu[1][1][0], 13.0, "texel 1 second");
    assert_close(gpu[1][2][0], 16.0, "texel 1 third");
    // Opacity payload rides along with its depth.
    assert_close(gpu[1][1][1], 0.4, "texel 1 second opacity");
}
