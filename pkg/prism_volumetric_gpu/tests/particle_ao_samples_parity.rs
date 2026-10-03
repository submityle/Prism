//! Real-device parity for the ambient-occlusion depth-fold twin:
//! [`GpuParticleAoSamples`](prism_volumetric_gpu::particle_ao_samples::GpuParticleAoSamples)
//! must reproduce the `CPU` golden
//! [`ao_from_samples`](prism_render_architecture::particle::ao_sample::ao_from_samples)
//! — a variable-length set of scene depths folded into one ambient-occlusion
//! term in `0..=1` around a shaded point — across the degenerate branches
//! (empty batch, empty depth set, no occluder, zero `range`, very large
//! `range`), a mixed-length batch resolved in a single dispatch, and a
//! randomized sweep compared term-for-term.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The oracle is the public golden
//! [`ao_from_samples`](prism_render_architecture::particle::ao_sample::ao_from_samples)
//! itself: for each query the reference term is computed on the host from the
//! same `depths`, `center_depth`, and `range`, and the `GPU` is pinned against
//! it.
//!
//! # Parity criterion
//!
//! The fold threads through only `+ - * /`, `clamp`, and comparisons with no
//! `sqrt` and no transcendental, and the per-query accumulation is serial, so
//! the `CPU` and `GPU` evaluate the same expression in the same order. A `GPU`
//! may still fuse a multiply-add the scalar reference leaves separate, so both
//! are asserted within `abs_diff <= 1e-5` or `rel_diff <= 1e-4`. No `f32` `==`
//! is used anywhere.
//!
//! # Conditioning
//!
//! The occlusion gate `delta > 0` is the only sharp branch; the randomized
//! sweep rejection-samples so every `center_depth - d` sits a clear margin away
//! from zero, keeping the `CPU` and `GPU` on the same side of the gate. The
//! `clamp` of `range / delta` is continuous, so no conditioning is needed near
//! its endpoints.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ao_sample`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::ao_sample::ao_from_samples;
use prism_volumetric_gpu::particle_ao_samples::{
    GpuParticleAoSamples, ParticleAoSamplesQuery, ParticleAoSamplesResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous quantity. A `GPU` arithmetic pipeline
/// may land a few units in the last place from the scalar reference; `1e-5`
/// admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-5;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Minimum magnitude required of every `center_depth - d` in the randomized
/// sweep so the `delta > 0` occlusion gate resolves identically on both paths.
const DELTA_MARGIN: f32 = 0.05;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at ten-thousandth resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 10_001) as f32 / 10_000.0 * (hi - lo)
}

/// Builds one query from a depth set, shaded-point depth and soft-range width.
fn query(depths: &[f32], center_depth: f32, range: f32) -> ParticleAoSamplesQuery {
    ParticleAoSamplesQuery {
        depths: depths.to_vec(),
        center_depth,
        range,
    }
}

/// Dispatches `queries` and asserts every term matches the golden fold.
fn check_batch(ctx: &GpuContext, gpu: &GpuParticleAoSamples, queries: &[ParticleAoSamplesQuery]) {
    let got: Vec<ParticleAoSamplesResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden = ao_from_samples(&q.depths, q.center_depth, q.range);
        assert!(
            close(r.ao, golden),
            "ao mismatch: gpu={} golden={} (center={}, range={}, n={})",
            r.ao,
            golden,
            q.center_depth,
            q.range,
            q.depths.len()
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping particle_ao_samples parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuParticleAoSamples::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn empty_depths_is_unoccluded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleAoSamples::new(&ctx);
    // A query with no depths resolves to the unoccluded 1.0.
    check_batch(&ctx, &gpu, &[query(&[], 5.0, 1.0)]);
}

#[test]
fn no_occluder_is_unoccluded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleAoSamples::new(&ctx);
    // Every sample is at or beyond the shaded point (delta <= 0), so nothing
    // occludes and the term stays 1.0.
    check_batch(&ctx, &gpu, &[query(&[5.0, 6.0, 7.0, 10.0], 5.0, 1.0)]);
}

#[test]
fn zero_range_is_unoccluded() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleAoSamples::new(&ctx);
    // range == 0 makes every smoothstep weight zero, so the term stays 1.0 even
    // with nearer occluders present.
    check_batch(&ctx, &gpu, &[query(&[1.0, 2.0, 3.0, 4.0], 5.0, 0.0)]);
}

#[test]
fn large_range_fully_occludes_near_samples() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleAoSamples::new(&ctx);
    // A very large range saturates range/delta to 1 for every nearer sample, so
    // the mean occlusion approaches 1 and the term approaches 0.
    check_batch(&ctx, &gpu, &[query(&[1.0, 2.0, 3.0, 4.0], 5.0, 1000.0)]);
}

#[test]
fn mixed_delta_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleAoSamples::new(&ctx);
    // A mix of nearer and farther samples exercises the partial-occlusion fold.
    check_batch(&ctx, &gpu, &[query(&[2.0, 8.0, 3.5, 11.0, 4.9], 5.0, 2.0)]);
}

#[test]
fn single_sample_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleAoSamples::new(&ctx);
    // One nearer sample: mean equals that single smoothstep weight.
    check_batch(&ctx, &gpu, &[query(&[2.0], 5.0, 1.5)]);
}

#[test]
fn mixed_length_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleAoSamples::new(&ctx);
    // Several queries of different depth-set lengths resolved in one dispatch
    // exercise the per-query offset/count slicing of the shared flattened
    // buffer, including an empty-depths query sharing the batch.
    let queries = vec![
        query(&[], 3.0, 1.0),
        query(&[1.0], 3.0, 1.0),
        query(&[1.0, 2.0, 2.5, 2.9], 3.0, 1.0),
        query(&[10.0, 11.0], 3.0, 1.0),
        query(&[0.5, 1.5, 2.5, 2.8, 2.95, 2.99], 3.0, 0.75),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuParticleAoSamples::new(&ctx);
    let mut state: u64 = 0x5eed_1234_abcd_0001;

    let mut queries: Vec<ParticleAoSamplesQuery> = Vec::new();
    for _ in 0..256 {
        let center = draw(&mut state, 1.0, 20.0);
        let range = draw(&mut state, 0.0, 6.0);
        let n = 1 + (lcg(&mut state) % 12) as usize;
        let mut depths: Vec<f32> = Vec::with_capacity(n);
        while depths.len() < n {
            let d = draw(&mut state, -5.0, 25.0);
            // Keep every delta a clear margin from the gate so both paths agree.
            if (center - d).abs() >= DELTA_MARGIN {
                depths.push(d);
            }
        }
        queries.push(query(&depths, center, range));
    }

    // One dispatch over the whole randomized batch, compared term-for-term.
    check_batch(&ctx, &gpu, &queries);
}
