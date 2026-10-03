//! Real-device parity for the ocean `clipmap` level-of-detail twin:
//! [`GpuWaterOceanClipmap`](prism_volumetric_gpu::water_ocean_clipmap::GpuWaterOceanClipmap)
//! must reproduce the numeric core of the `CPU` golden
//! [`ocean_lod`](prism_render_architecture::water::ocean_lod) — the owning ring
//! [`select_clipmap_ring`](prism_render_architecture::water::ocean_lod::select_clipmap_ring)
//! and the geomorph blend weight
//! [`clipmap_morph_weight`](prism_render_architecture::water::ocean_lod::clipmap_morph_weight)
//! — across ring interiors, morph bands, out-of-range saturation, degenerate
//! bands and disabled morphing, a mixed batch and a randomized sweep compared
//! sample-for-sample.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values come straight from the public golden
//! [`select_clipmap_ring`](prism_render_architecture::water::ocean_lod::select_clipmap_ring)
//! and
//! [`clipmap_morph_weight`](prism_render_architecture::water::ocean_lod::clipmap_morph_weight)
//! evaluated on an
//! [`OceanClipmapConfig`](prism_render_architecture::water::ocean_lod::OceanClipmapConfig)
//! rebuilt from each query, so a `GPU == golden` pass is direct evidence the
//! ported kernel classifies the same way the reference does.
//!
//! # Parity criterion
//!
//! The owning ring is a discrete classification, asserted bit-exact (`==`); the
//! morph weight threads through subtracts and a guarded divide, so a `GPU`
//! divide may land a few units in the last place from the scalar reference and
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor
//! `1e-6`).
//!
//! # Conditioning
//!
//! Ring boundaries (`distance ~ ring_outer_radius`) are discrete
//! discontinuities and coincide with the `0 -> 1` step of a degenerate band, so
//! the randomized sweep rejects any distance whose golden ring is not stable
//! across a comfortable margin on both sides. The morph-band start is a
//! continuous kink (the weight is `0` on both sides there), so it needs no
//! rejection. Every fixture keeps a wide margin from every ring boundary.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::ocean_lod`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::ocean_lod::{
    clipmap_morph_weight, select_clipmap_ring, OceanClipmapConfig,
};
use prism_volumetric_gpu::water_ocean_clipmap::{
    GpuWaterOceanClipmap, WaterOceanClipmapQuery, WaterOceanClipmapResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the morph weight. A `GPU` divide may land a few
/// units in the last place from the scalar reference; `1e-4` admits that legal
/// slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Rebuilds the golden config from a query.
fn config_of(q: &WaterOceanClipmapQuery) -> OceanClipmapConfig {
    OceanClipmapConfig {
        ring_count: q.ring_count,
        inner_radius: q.inner_radius,
        radius_growth: q.radius_growth,
        morph_fraction: q.morph_fraction,
    }
}

/// Computes the golden result for one query straight from the public
/// [`select_clipmap_ring`](prism_render_architecture::water::ocean_lod::select_clipmap_ring)
/// and
/// [`clipmap_morph_weight`](prism_render_architecture::water::ocean_lod::clipmap_morph_weight).
fn oracle(q: &WaterOceanClipmapQuery) -> WaterOceanClipmapResult {
    let cfg = config_of(q);
    WaterOceanClipmapResult {
        ring: select_clipmap_ring(q.distance, cfg),
        morph_weight: clipmap_morph_weight(q.distance, cfg),
    }
}

/// Pins one `GPU` sample against the golden: the ring exactly, the morph weight
/// within tolerance.
fn check_sample(idx: usize, got: &WaterOceanClipmapResult, want: &WaterOceanClipmapResult) {
    assert_eq!(
        got.ring, want.ring,
        "sample {idx} ring: gpu {} vs cpu {}",
        got.ring, want.ring
    );
    assert!(
        close(got.morph_weight, want.morph_weight),
        "sample {idx} morph_weight: gpu {} vs cpu {}",
        got.morph_weight,
        want.morph_weight
    );
}

/// Dispatches `queries` and pins every `GPU` sample against the golden oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterOceanClipmap, queries: &[WaterOceanClipmapQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query is expected");
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        check_sample(idx, g, &oracle(q));
    }
}

/// A small, fixed set of well-conditioned fixtures covering each ring interior,
/// a morph band, out-of-range saturation and both degenerate branches.
///
/// The reference layout has four rings at outer radii `16, 32, 64, 128`
/// (`inner_radius = 16`, `radius_growth = 2`), with a quarter-band morph.
fn fixture_queries() -> Vec<WaterOceanClipmapQuery> {
    let inner = 16.0;
    let growth = 2.0;
    let frac = 0.25;
    let rings = 4;
    vec![
        // Ring 0 interior, below the morph start (morph_start = 12).
        WaterOceanClipmapQuery::new(8.0, inner, growth, frac, rings),
        // Ring 0 morph band (12 < d <= 16): a partial blend.
        WaterOceanClipmapQuery::new(14.0, inner, growth, frac, rings),
        // Ring 1 interior, below its morph start (morph_start = 28).
        WaterOceanClipmapQuery::new(22.0, inner, growth, frac, rings),
        // Ring 2 morph band (56 < d <= 64): a partial blend.
        WaterOceanClipmapQuery::new(60.0, inner, growth, frac, rings),
        // Out past the outermost ring: ring = last, morph saturates to 1.
        WaterOceanClipmapQuery::new(400.0, inner, growth, frac, rings),
        // Single-ring config: last_ring = 0, interior sample.
        WaterOceanClipmapQuery::new(10.0, 24.0, growth, frac, 1),
        // Disabled morphing (morph_fraction = 0): hard boundary, 0 inside.
        WaterOceanClipmapQuery::new(30.0, inner, growth, 0.0, rings),
        // Disabled morphing, past the outermost ring: saturates to 1.
        WaterOceanClipmapQuery::new(500.0, inner, growth, 0.0, rings),
        // Degenerate band (tiny inner_radius): band <= EPS on ring 0.
        WaterOceanClipmapQuery::new(0.5, 1.0e-7, growth, frac, 1),
        // Deeper ring stack exercising more growth multiplications.
        WaterOceanClipmapQuery::new(300.0, 8.0, 1.7, 0.3, 8),
    ]
}

/// A 64-bit linear-congruential generator for host-side fixtures; no
/// transcendental and no float equality.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws a value in `[0, span)` with milli-resolution from the generator.
fn draw(state: &mut u64, span: f32) -> f32 {
    (lcg(state) % 1000) as f32 / 1000.0 * span
}

/// Whether the golden ring is stable across `+/- margin` meters, so the sample
/// sits well clear of any ring boundary (and the degenerate `0 -> 1` step).
fn ring_stable(q: &WaterOceanClipmapQuery, margin: f32) -> bool {
    let cfg = config_of(q);
    let r0 = select_clipmap_ring(q.distance, cfg);
    let lo = select_clipmap_ring((q.distance - margin).max(0.0), cfg);
    let hi = select_clipmap_ring(q.distance + margin, cfg);
    r0 == lo && r0 == hi
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanClipmap::new(&ctx);
    // An empty batch never dispatches (a storage buffer cannot be zero-sized)
    // and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn ring_interior_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanClipmap::new(&ctx);
    let q = WaterOceanClipmapQuery::new(22.0, 16.0, 2.0, 0.25, 4);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert_eq!(want.ring, 1, "fixture must land in ring 1");
    check_sample(0, &got[0], &want);
}

#[test]
fn morph_band_is_partial() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanClipmap::new(&ctx);
    let q = WaterOceanClipmapQuery::new(14.0, 16.0, 2.0, 0.25, 4);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        want.morph_weight > 0.0 && want.morph_weight < 1.0,
        "fixture must sit inside the morph band"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn out_of_range_saturates_to_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanClipmap::new(&ctx);
    let q = WaterOceanClipmapQuery::new(400.0, 16.0, 2.0, 0.25, 4);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert_eq!(want.ring, 3, "fixture must clamp to the last ring");
    assert!(
        close(want.morph_weight, 1.0),
        "fixture must saturate the morph weight"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn disabled_morph_is_a_hard_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanClipmap::new(&ctx);
    // morph_fraction = 0: inside the ring the weight is 0, past the outermost
    // ring it is 1.
    let inside = WaterOceanClipmapQuery::new(30.0, 16.0, 2.0, 0.0, 4);
    let beyond = WaterOceanClipmapQuery::new(500.0, 16.0, 2.0, 0.0, 4);
    let got = gpu.evaluate(&ctx, &[inside, beyond]);
    assert_eq!(got.len(), 2);
    let want_inside = oracle(&inside);
    let want_beyond = oracle(&beyond);
    assert!(close(want_inside.morph_weight, 0.0));
    assert!(close(want_beyond.morph_weight, 1.0));
    check_sample(0, &got[0], &want_inside);
    check_sample(1, &got[1], &want_beyond);
}

#[test]
fn degenerate_band_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanClipmap::new(&ctx);
    // Tiny inner_radius makes ring 0's band <= EPS; the weight follows the
    // past-outer saturate branch.
    let q = WaterOceanClipmapQuery::new(0.5, 1.0e-7, 2.0, 0.25, 1);
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    check_sample(0, &got[0], &oracle(&q));
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanClipmap::new(&ctx);
    // Every fixture dispatched together exercises per-thread indexing and the
    // contiguous output slots.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanClipmap::new(&ctx);
    let mut state = 0x51ed_2c7a_9b4e_f013_u64;
    let mut queries = fixture_queries();
    let mut accepted = 0u32;
    let mut tries = 0u32;
    // Many well-conditioned random samples (several workgroups' worth) pin both
    // classifications across a wide span of layouts and distances, with
    // rejection sampling keeping each distance clear of every ring boundary.
    while accepted < 256 && tries < 40_000 {
        tries += 1;
        // ring_count in 1..=16, radius_growth > 1, inner_radius > 0.
        let ring_count = 1 + (lcg(&mut state) % 16);
        let inner_radius = 4.0 + draw(&mut state, 36.0);
        let radius_growth = 1.5 + draw(&mut state, 1.0);
        let morph_fraction = 0.1 + draw(&mut state, 0.4);

        // Outermost radius for the chosen layout: inner * growth^(ring_count-1).
        let mut outer = inner_radius;
        let last = ring_count.saturating_sub(1);
        let mut level = 0u32;
        while level < last {
            outer *= radius_growth;
            level += 1;
        }
        // Distances up to 1.5x the outermost radius so beyond-range saturation
        // is covered too.
        let distance = draw(&mut state, outer * 1.5);

        let q = WaterOceanClipmapQuery::new(
            distance,
            inner_radius,
            radius_growth,
            morph_fraction,
            ring_count,
        );
        // Reject near a ring boundary: keep the golden ring stable across a
        // comfortable margin so CPU and GPU cannot straddle it.
        if !ring_stable(&q, 0.25) {
            continue;
        }
        queries.push(q);
        accepted += 1;
    }
    assert!(
        accepted >= 256,
        "expected at least 256 random samples, got {accepted}"
    );
    check(&ctx, &gpu, &queries);
}
