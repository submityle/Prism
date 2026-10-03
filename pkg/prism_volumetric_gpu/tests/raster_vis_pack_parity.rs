//! Real-device parity for the vis-buffer packing twin:
//! [`GpuRasterVisPack`](prism_volumetric_gpu::raster_vis_pack::GpuRasterVisPack)
//! must reproduce, bit for bit, the two `32`-bit halves of the `CPU` golden
//! packed word
//! [`pack_vis`](prism_render_architecture::virtual_geometry::software_raster::pack_vis)`(`
//! [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth)`(depth), payload)`
//! across clamp boundaries, extreme payloads, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden packing core is `pub`, so the oracle calls it directly: it forms
//! the reference `u64` word `pack_vis(encode_depth(depth), payload)` and splits
//! it into the same `[hi, lo]` pair the twin returns (`hi = word >> 32`,
//! `lo = word & 0xffff_ffff`). The split is also cross-checked against
//! [`vis_depth`](prism_render_architecture::virtual_geometry::software_raster::vis_depth)
//! /
//! [`vis_payload`](prism_render_architecture::virtual_geometry::software_raster::vis_payload)
//! and against `encode_depth`/`payload`, so a `GPU == golden` pass is
//! established directly with no in-host re-derivation of the packing logic.
//!
//! # Parity criterion
//!
//! Both halves are discrete `32`-bit values: `hi` is the raw bit pattern of a
//! clamped depth (a `bitcast`, not an arithmetic result) and `lo` is the payload
//! copied verbatim. The clamp is an exact magnitude selection with no rounding,
//! so there is no floating-point slack to admit — every half is asserted with an
//! exact integer `==`.
//!
//! # Conditioning
//!
//! Fixtures sweep the `clamp` endpoints (`depth = 0.0`, `1.0`), interior depths,
//! and out-of-range depths on both sides (`depth < 0`, `depth > 1`) so the clamp
//! is exercised in every direction, plus the payload extremes `0` and
//! [`u32::MAX`]. The randomized sweep draws depths across `[-1, 2]` and arbitrary
//! payloads, driving both the clamped and unclamped paths.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::virtual_geometry::software_raster`；无第三方引擎源码或衍生代码。

use prism_render_architecture::virtual_geometry::software_raster::{
    encode_depth, pack_vis, vis_depth, vis_payload,
};
use prism_volumetric_gpu::raster_vis_pack::{
    GpuRasterVisPack, RasterVisPackQuery, RasterVisPackResult,
};
use prism_volumetric_gpu::GpuContext;

/// Builds the expected `[hi, lo]` by forming the golden `u64` word and splitting
/// it, cross-checking the split against the field extractors and the inputs.
fn oracle(q: &RasterVisPackQuery) -> RasterVisPackResult {
    let depth_key = encode_depth(q.depth);
    let packed = pack_vis(depth_key, q.payload);
    let hi = (packed >> 32) as u32;
    let lo = (packed & 0xffff_ffff) as u32;
    // The split must agree with the reference field extractors and inputs.
    assert_eq!(hi, depth_key, "high half must equal the depth key");
    assert_eq!(hi, vis_depth(packed), "high half must equal vis_depth");
    assert_eq!(lo, q.payload, "low half must equal the payload");
    assert_eq!(lo, vis_payload(packed), "low half must equal vis_payload");
    RasterVisPackResult { hi, lo }
}

/// Dispatches every query and pins each packed half against the golden with an
/// exact integer `==`, since both halves are discrete bit patterns.
fn check(ctx: &GpuContext, gpu: &GpuRasterVisPack, queries: &[RasterVisPackQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (qi, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        assert_eq!(
            result.hi, want.hi,
            "query {qi} depth key: gpu {result:?} vs cpu {want:?}"
        );
        assert_eq!(
            result.lo, want.lo,
            "query {qi} payload: gpu {result:?} vs cpu {want:?}"
        );
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a depth across `[-1, 2]` at sixteenth resolution, so some draws fall
/// below `0` and some above `1` to exercise the clamp in both directions.
fn rand_depth(state: &mut u64) -> f32 {
    (lcg(state) % 49) as f32 / 16.0 - 1.0
}

/// Draws one random packing query: a depth across the clamp range and an
/// arbitrary payload.
fn rand_query(state: &mut u64) -> RasterVisPackQuery {
    RasterVisPackQuery::new(rand_depth(state), lcg(state))
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterVisPack::new(&ctx);
    // No queries: the host short-circuits and never dispatches.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch must return no results");
}

#[test]
fn clamp_endpoints_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterVisPack::new(&ctx);
    // The two clamp endpoints and the midpoint, with a mix of payloads.
    let queries = [
        RasterVisPackQuery::new(0.0, 0),
        RasterVisPackQuery::new(0.5, 0x0012_3456),
        RasterVisPackQuery::new(1.0, 0x00AB_CDEF),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn out_of_range_depth_is_clamped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterVisPack::new(&ctx);
    // Depths below 0 and above 1 must clamp to the endpoints, so a stray
    // negative depth cannot masquerade as the nearest surface.
    let queries = [
        RasterVisPackQuery::new(-0.25, 1),
        RasterVisPackQuery::new(-5.0, 7),
        RasterVisPackQuery::new(1.5, 42),
        RasterVisPackQuery::new(1_000.0, 0x7FFF_FFFF),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn payload_extremes_round_trip() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterVisPack::new(&ctx);
    // The payload is copied verbatim, so both extremes must survive untouched.
    let queries = [
        RasterVisPackQuery::new(0.3, 0),
        RasterVisPackQuery::new(0.7, u32::MAX),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn interior_depths_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterVisPack::new(&ctx);
    // A spread of interior depths at sixteenth resolution with varied payloads,
    // pinning the per-query indexing of the result grid.
    let queries: Vec<RasterVisPackQuery> = (1..16)
        .map(|k| RasterVisPackQuery::new(k as f32 / 16.0, (k as u32) << 7))
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRasterVisPack::new(&ctx);
    let mut state: u64 = 0x5EED_B10C_1234_0001;
    let queries: Vec<RasterVisPackQuery> = (0..512).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
