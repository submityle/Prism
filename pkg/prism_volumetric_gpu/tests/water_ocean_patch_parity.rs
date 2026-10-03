//! Real-device parity for the ocean `clipmap` patch twin:
//! [`GpuWaterOceanPatch`](prism_volumetric_gpu::water_ocean_patch::GpuWaterOceanPatch)
//! must reproduce the ring selection and geomorph weighting of the `CPU` golden
//! [`ocean_lod`](prism_render_architecture::water::ocean_lod) —
//! [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch)
//! and its kernels
//! [`select_clipmap_ring`](prism_render_architecture::water::ocean_lod::select_clipmap_ring)
//! and
//! [`clipmap_morph_weight`](prism_render_architecture::water::ocean_lod::clipmap_morph_weight)
//! — across hand-chosen configs, boundary distances, and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`resolve_ocean_patch`](prism_render_architecture::water::ocean_lod::resolve_ocean_patch)
//! is public and pure, so the expected result is built in-host by calling it
//! directly. A `GPU == oracle` pass is therefore directly a `GPU == golden`
//! pass. The patch's
//! [`WaterBodyHandle`](prism_render_architecture::water::WaterBodyHandle) is a
//! pure pass-through in the golden, so the oracle passes a placeholder handle
//! and only `ring` and `morph` are compared.
//!
//! # Parity criterion
//!
//! The selected `ring` is pure integer classification and is asserted exactly.
//! The `morph` weight threads through a subtract, a divide and a `clamp`, so a
//! `GPU` divide may land a few units in the last place from the scalar
//! reference; it is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Conditioning
//!
//! Because `ring` is a discrete tier, the random sweep rejects any distance
//! within a margin of a ring's outer radius (where the ring selection could
//! flip on a last-place difference) and within a margin of the selected ring's
//! morph band start (where the morph branch could flip). The degenerate
//! `morph_fraction = 0` and `ring_count = 0` cases are exercised separately with
//! wide-margin fixtures.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::ocean_lod`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::ocean_lod::{resolve_ocean_patch, OceanClipmapConfig};
use prism_render_architecture::water::WaterBodyHandle;
use prism_volumetric_gpu::water_ocean_patch::{
    GpuWaterOceanPatch, WaterOceanPatchQuery, WaterOceanPatchResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the geomorph weight. A `GPU` divide may land a few
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

/// Reconstructs the reference [`OceanClipmapConfig`] from a query.
fn cfg_of(q: &WaterOceanPatchQuery) -> OceanClipmapConfig {
    OceanClipmapConfig {
        ring_count: q.ring_count,
        inner_radius: q.inner_radius,
        radius_growth: q.radius_growth,
        morph_fraction: q.morph_fraction,
    }
}

/// Builds the in-host oracle for one query by calling the golden
/// `resolve_ocean_patch` directly. The `GPU` is pinned against this exact closed
/// form. The body handle is a pure pass-through and is dropped from the result.
fn oracle(q: &WaterOceanPatchQuery) -> WaterOceanPatchResult {
    let lod = resolve_ocean_patch(WaterBodyHandle(0), q.distance, cfg_of(q));
    WaterOceanPatchResult {
        ring: lod.ring,
        morph: lod.morph,
    }
}

/// Pins one `GPU` result against the in-host oracle: `ring` exactly, `morph`
/// within tolerance.
fn check_patch(idx: usize, got: &WaterOceanPatchResult, want: &WaterOceanPatchResult) {
    assert_eq!(
        got.ring, want.ring,
        "patch {idx} ring: gpu {} vs cpu {}",
        got.ring, want.ring
    );
    assert!(
        close(got.morph, want.morph),
        "patch {idx} morph: gpu {} vs cpu {}",
        got.morph,
        want.morph
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterOceanPatch, queries: &[WaterOceanPatchQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_patch(idx, result, &want);
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

/// Draws a scalar in `[lo, hi]` at milli resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    lo + (lcg(state) % (span + 1)) as f32 / 1000.0
}

/// Whether `distance` keeps a comfortable margin from every ring's outer radius
/// (so the discrete ring selection is stable) and from the selected ring's
/// morph band start (so the morph branch is stable). Uses only arithmetic and
/// comparisons, never an `f32` `==`.
fn well_conditioned(q: &WaterOceanPatchQuery) -> bool {
    let cfg = cfg_of(q);
    let margin = 1.0_f32;
    // Keep clear of every ring boundary where ring selection could flip.
    for ring in 0..q.ring_count {
        let outer = cfg.ring_outer_radius(ring);
        if (q.distance - outer).abs() < margin {
            return false;
        }
    }
    // Keep clear of the selected ring's morph band start where the morph branch
    // could flip, when morphing is active.
    let ring = cfg.ring_count.saturating_sub(1).min(q.ring_count);
    let selected = {
        let mut r = 0;
        let last = cfg.last_ring();
        while r < last {
            if q.distance <= cfg.ring_outer_radius(r) {
                break;
            }
            r += 1;
        }
        r.min(ring)
    };
    let inner = cfg.ring_inner_radius(selected);
    let outer = cfg.ring_outer_radius(selected);
    let band = outer - inner;
    if band > 1.0e-3 && q.morph_fraction > 1.0e-3 {
        let morph_start = outer - band * q.morph_fraction.min(1.0);
        if (q.distance - morph_start).abs() < margin {
            return false;
        }
    }
    true
}

/// Draws one well-conditioned random query: positive geometry, a growth factor
/// above one, and a distance kept clear of every ring boundary and the morph
/// band start.
fn random_query(state: &mut u64) -> WaterOceanPatchQuery {
    loop {
        let q = WaterOceanPatchQuery {
            distance: draw(state, 0.0, 2000.0),
            inner_radius: draw(state, 8.0, 64.0),
            radius_growth: draw(state, 1.5, 2.5),
            morph_fraction: draw(state, 0.05, 0.6),
            ring_count: 1 + lcg(state) % 5,
        };
        if well_conditioned(&q) {
            return q;
        }
    }
}

/// The deterministic hand-chosen fixtures, each exercising a distinct branch and
/// kept clear of every discrete tie. The reference `CLIPMAP` config used in the
/// golden tests has ring outer radii `32, 64, 128, 256` with a `25%` morph band.
fn fixture_queries() -> Vec<WaterOceanPatchQuery> {
    vec![
        // Near camera, inside ring 0, before its morph band (starts at 24).
        WaterOceanPatchQuery {
            distance: 10.0,
            inner_radius: 32.0,
            radius_growth: 2.0,
            morph_fraction: 0.25,
            ring_count: 4,
        },
        // Inside ring 0's morph band (24..32): a partial geomorph weight.
        WaterOceanPatchQuery {
            distance: 28.0,
            inner_radius: 32.0,
            radius_growth: 2.0,
            morph_fraction: 0.25,
            ring_count: 4,
        },
        // Inside ring 1 (32..64), before its morph band (starts at 56).
        WaterOceanPatchQuery {
            distance: 40.0,
            inner_radius: 32.0,
            radius_growth: 2.0,
            morph_fraction: 0.25,
            ring_count: 4,
        },
        // Inside ring 2 (64..128), before its morph band (starts at 112).
        WaterOceanPatchQuery {
            distance: 90.0,
            inner_radius: 32.0,
            radius_growth: 2.0,
            morph_fraction: 0.25,
            ring_count: 4,
        },
        // Far beyond the outermost ring: clamps to the last ring, morph
        // saturates to 1.
        WaterOceanPatchQuery {
            distance: 900.0,
            inner_radius: 32.0,
            radius_growth: 2.0,
            morph_fraction: 0.25,
            ring_count: 4,
        },
        // Morphing disabled (morph_fraction = 0): hard ring boundaries, morph is
        // 0 inside the ring. Distance 90 is inside ring 2 (64..128).
        WaterOceanPatchQuery {
            distance: 90.0,
            inner_radius: 32.0,
            radius_growth: 2.0,
            morph_fraction: 0.0,
            ring_count: 4,
        },
        // Single ring: everything is ring 0, distance beyond inner_radius
        // saturates the morph to 1 (degenerate band past the ring).
        WaterOceanPatchQuery {
            distance: 50.0,
            inner_radius: 20.0,
            radius_growth: 2.0,
            morph_fraction: 0.25,
            ring_count: 1,
        },
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_ocean_patch parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterOceanPatch::new(&ctx);
    // The host short-circuits an empty batch (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn fixture_patches_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanPatch::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn zero_ring_count_resolves_to_single_ring() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanPatch::new(&ctx);
    // ring_count = 0 is a degenerate single-ring config: last_ring saturates to
    // 0, so every distance resolves to ring 0.
    let q = WaterOceanPatchQuery {
        distance: 150.0,
        inner_radius: 32.0,
        radius_growth: 2.0,
        morph_fraction: 0.25,
        ring_count: 0,
    };
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].ring, 0, "degenerate config selects ring 0");
    check_patch(0, &got[0], &oracle(&q));
}

#[test]
fn far_distance_clamps_to_last_ring() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanPatch::new(&ctx);
    // A distance well past the outermost ring clamps to the last ring and the
    // morph saturates to 1.
    let q = WaterOceanPatchQuery {
        distance: 5000.0,
        inner_radius: 32.0,
        radius_growth: 2.0,
        morph_fraction: 0.25,
        ring_count: 4,
    };
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert_eq!(want.ring, 3, "distance past the horizon clamps to ring 3");
    assert!(
        (want.morph - 1.0).abs() < EPS,
        "morph saturates to 1 past the outermost ring"
    );
    check_patch(0, &got[0], &want);
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterOceanPatch::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random patches pin every output across a wide
    // span of configs and distances.
    for _ in 0..512 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
