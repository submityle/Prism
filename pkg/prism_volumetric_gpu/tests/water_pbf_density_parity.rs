//! Real-device parity for the `SPH` density-estimate twin:
//! [`GpuWaterPbfDensity`](prism_volumetric_gpu::water_pbf_density::GpuWaterPbfDensity)
//! must reproduce the numeric core of the `CPU` golden
//! [`pbf`](prism_render_architecture::water::pbf) — the `Poly6`-weighted
//! neighbour sum
//! [`estimate_density`](prism_render_architecture::water::pbf::estimate_density)
//! built on the `Poly6` kernel
//! [`poly6`](prism_render_architecture::water::pbf::poly6) — across interior
//! samples, the support breakpoint (`r^2 >= h^2`), a degenerate radius
//! (`h <= EPS`), an empty neighbour list, a mixed batch and a randomized sweep
//! compared sample-for-sample.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The expected values come straight from the public golden
//! [`estimate_density`](prism_render_architecture::water::pbf::estimate_density),
//! so a `GPU == golden` pass is direct evidence the ported kernel sums the same
//! density the reference does.
//!
//! # Parity criterion
//!
//! The density threads through multiplies, subtracts, a guarded divide and a
//! running sum, so a `GPU` divide may land a few units in the last place from
//! the scalar reference and the running sum may reorder rounding; the density
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative floor
//! `1e-6`).
//!
//! # Conditioning
//!
//! The support breakpoint `r^2` crossing `h^2` is a discontinuity. Fixtures and
//! the randomized sweep keep every neighbour's `r^2` in `[0, 0.9 * h^2]`, well
//! clear of it, so `CPU` and `GPU` cannot straddle it. The smoothing radius `h`
//! is kept far above the `1e-6` degeneracy floor except in the dedicated
//! degenerate fixture.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::pbf`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::pbf::estimate_density;
use prism_volumetric_gpu::water_pbf_density::{
    GpuWaterPbfDensity, WaterPbfDensityQuery, WaterPbfDensityResult, MAX_NEIGHBORS,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous field. A `GPU` divide or reordered sum
/// may land a few units in the last place from the scalar reference; `1e-4`
/// admits that legal slack while still failing a wrong port.
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

/// Computes the golden result for one query straight from the public
/// [`estimate_density`](prism_render_architecture::water::pbf::estimate_density).
fn oracle(q: &WaterPbfDensityQuery) -> WaterPbfDensityResult {
    WaterPbfDensityResult {
        density: estimate_density(q.mass, &q.neighbor_r_squared, q.h),
    }
}

/// Pins one `GPU` sample against the golden density within tolerance.
fn check_sample(idx: usize, got: &WaterPbfDensityResult, want: &WaterPbfDensityResult) {
    assert!(
        close(got.density, want.density),
        "sample {idx} density: gpu {} vs cpu {}",
        got.density,
        want.density
    );
}

/// Dispatches `queries` and pins every `GPU` sample against the golden oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterPbfDensity, queries: &[WaterPbfDensityQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query is expected");
    for (idx, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        check_sample(idx, g, &oracle(q));
    }
}

/// The deterministic fixture battery exercising the interior, the support
/// breakpoint, the degenerate radius and an empty neighbour list.
fn fixture_queries() -> Vec<WaterPbfDensityQuery> {
    vec![
        // Interior neighbours well inside the support, plus the self term r=0.
        WaterPbfDensityQuery::new(1.0, 1.0, vec![0.0, 0.1, 0.25, 0.4, 0.6]),
        // Heavier mass, several interior neighbours.
        WaterPbfDensityQuery::new(2.5, 0.8, vec![0.05, 0.1, 0.2, 0.3]),
        // Every neighbour at or beyond h^2 contributes zero -> density 0.
        WaterPbfDensityQuery::new(1.0, 0.5, vec![0.25, 0.3, 1.0]),
        // Degenerate radius h <= EPS zeroes every weight -> density 0.
        WaterPbfDensityQuery::new(1.0, 1.0e-9, vec![0.0, 0.01, 0.02]),
        // Empty neighbour list -> sum 0 -> density 0.
        WaterPbfDensityQuery::new(3.0, 1.0, vec![]),
        // Negative mass clamps up to 0 -> density 0.
        WaterPbfDensityQuery::new(-2.0, 1.0, vec![0.1, 0.2, 0.3]),
    ]
}

/// A 64-bit linear-congruential generator (`SplitMix`/`PCG`-style multiplier)
/// for host-side fixtures; no transcendental and no float equality.
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

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfDensity::new(&ctx);
    // An empty batch never dispatches (a storage buffer cannot be zero-sized)
    // and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn interior_density_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfDensity::new(&ctx);
    let q = WaterPbfDensityQuery::new(1.0, 1.0, vec![0.0, 0.1, 0.25, 0.4, 0.6]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        want.density > 0.0,
        "interior sample must have positive density"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn beyond_support_density_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfDensity::new(&ctx);
    // Every neighbour at or beyond h^2 vanishes, so the density is exactly 0.
    let q = WaterPbfDensityQuery::new(1.0, 0.5, vec![0.25, 0.3, 1.0]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.density, 0.0),
        "fixture must be beyond the support"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn degenerate_radius_density_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfDensity::new(&ctx);
    // h well below EPS is the "no kernel" degenerate case: density 0.
    let q = WaterPbfDensityQuery::new(1.0, 1.0e-9, vec![0.0, 0.01, 0.02]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.density, 0.0),
        "degenerate radius must zero the density"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn empty_neighbours_density_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfDensity::new(&ctx);
    // No neighbours leaves the sum at 0, so the density is 0 regardless of mass.
    let q = WaterPbfDensityQuery::new(3.0, 1.0, vec![]);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        close(want.density, 0.0),
        "no neighbours must give zero density"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn mixed_batch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfDensity::new(&ctx);
    // Every fixture dispatched together exercises per-thread indexing and the
    // contiguous output slots.
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn full_neighbour_block_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfDensity::new(&ctx);
    // A full MAX_NEIGHBORS block exercises the loop bound and the whole slot
    // array, with every neighbour well inside the support.
    let h = 1.5_f32;
    let h2 = h * h;
    let mut neighbours = Vec::with_capacity(MAX_NEIGHBORS);
    let mut state = 0x51ed_2701_c0ff_ee11_u64;
    while neighbours.len() < MAX_NEIGHBORS {
        neighbours.push(draw(&mut state, 0.9 * h2));
    }
    let q = WaterPbfDensityQuery::new(1.25, h, neighbours);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(
        want.density > 0.0,
        "a full support block must sum to positive density"
    );
    check_sample(0, &got[0], &want);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterPbfDensity::new(&ctx);
    let mut state = 0x0bad_c0de_dead_beef_u64;
    let mut queries = fixture_queries();
    // Many well-conditioned random particles (several workgroups' worth) pin the
    // density across a wide span of masses, radii and neighbour counts, with
    // every neighbour kept in [0, 0.9 * h^2] so no sample straddles the support
    // breakpoint.
    let mut built = 0u32;
    while built < 512 {
        built += 1;
        // Smoothing radius well above EPS: [0.5, 2.5).
        let h = 0.5 + draw(&mut state, 2.0);
        let h2 = h * h;
        let mass = draw(&mut state, 4.0);
        // Neighbour count in 0..=MAX_NEIGHBORS.
        let n = (lcg(&mut state) as usize) % (MAX_NEIGHBORS + 1);
        let mut neighbours = Vec::with_capacity(n);
        let mut j = 0;
        while j < n {
            neighbours.push(draw(&mut state, 0.9 * h2));
            j += 1;
        }
        queries.push(WaterPbfDensityQuery::new(mass, h, neighbours));
    }
    check(&ctx, &gpu, &queries);
}
