//! Real-device parity for the two-way fluid/rigid coupling-force twin:
//! [`GpuWaterCouplingForces`](prism_volumetric_gpu::water_coupling_forces::GpuWaterCouplingForces)
//! must reproduce the three stateless scalar primitives of the `CPU` golden
//! [`coupling`](prism_render_architecture::water::coupling) — the quadratic
//! [`drag_force`](prism_render_architecture::water::coupling::drag_force), the
//! [`added_mass`](prism_render_architecture::water::coupling::added_mass)
//! reaction, and the
//! [`source_writeback_fraction`](prism_render_architecture::water::coupling::source_writeback_fraction)
//! submersion weight — across hand-chosen fixtures and a randomized sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The three golden functions are public and pure, so the expected result is
//! built in-host by calling them directly:
//! [`drag_force`](prism_render_architecture::water::coupling::drag_force),
//! [`added_mass`](prism_render_architecture::water::coupling::added_mass) and
//! [`source_writeback_fraction`](prism_render_architecture::water::coupling::source_writeback_fraction).
//! A `GPU == oracle` pass is therefore directly a `GPU == golden` pass.
//!
//! # Parity criterion
//!
//! All three outputs are continuous magnitudes threading through multiplies, a
//! divide and a `clamp`, so a `GPU` divide may land a few units in the last
//! place from the scalar reference; each is asserted within `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3`. There is no discrete flag to compare.
//!
//! # Conditioning
//!
//! Fixtures keep the writeback fraction a clear margin away from both `clamp`
//! edges and from the `EPS = 1e-6` total-volume threshold, except for the cases
//! that deliberately overshoot a `clamp` edge or undershoot the threshold with a
//! wide margin, so `CPU` and `GPU` agree on every branch. Clamp-edge fixtures
//! push well past the edge (submerged far exceeding total, or a frankly
//! negative input) so both sides land on the same saturated value.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::coupling`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::coupling::{
    added_mass, drag_force, source_writeback_fraction,
};
use prism_volumetric_gpu::water_coupling_forces::{
    GpuWaterCouplingForces, WaterCouplingForcesQuery, WaterCouplingForcesResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on a continuous output. A `GPU` divide may land a few
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

/// Builds the in-host oracle for one query by calling the three golden
/// functions directly. The `GPU` is pinned against this exact closed form.
fn oracle(q: &WaterCouplingForcesQuery) -> WaterCouplingForcesResult {
    WaterCouplingForcesResult {
        drag: drag_force(q.drag_coeff, q.fluid_density, q.area, q.rel_speed),
        added_mass: added_mass(q.added_mass_coeff, q.fluid_density, q.displaced_volume),
        writeback_fraction: source_writeback_fraction(q.submerged_volume, q.total_volume),
    }
}

/// Pins one `GPU` result against the in-host oracle: every continuous output
/// within tolerance.
fn check_body(idx: usize, got: &WaterCouplingForcesResult, want: &WaterCouplingForcesResult) {
    assert!(
        close(got.drag, want.drag),
        "body {idx} drag: gpu {} vs cpu {}",
        got.drag,
        want.drag
    );
    assert!(
        close(got.added_mass, want.added_mass),
        "body {idx} added_mass: gpu {} vs cpu {}",
        got.added_mass,
        want.added_mass
    );
    assert!(
        close(got.writeback_fraction, want.writeback_fraction),
        "body {idx} writeback_fraction: gpu {} vs cpu {}",
        got.writeback_fraction,
        want.writeback_fraction
    );
}

/// Dispatches every query and pins each result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuWaterCouplingForces, queries: &[WaterCouplingForcesQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (q, result)) in queries.iter().zip(got.iter()).enumerate() {
        let want = oracle(q);
        check_body(idx, result, &want);
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

/// Draws a non-negative scalar in `[lo, hi]` at milli resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    let span = ((hi - lo) * 1000.0) as u32;
    lo + (lcg(state) % (span + 1)) as f32 / 1000.0
}

/// Draws one well-conditioned random query: all inputs non-negative and the
/// writeback fraction kept a clear margin from both `clamp` edges and from the
/// `EPS` total-volume threshold, so `CPU` and `GPU` stay on the same branch.
fn random_query(state: &mut u64) -> WaterCouplingForcesQuery {
    let total_volume = draw(state, 0.5, 10.0);
    // Keep the fraction inside `[0.08, 0.92]` so neither clamp edge is near.
    let frac = draw(state, 0.08, 0.92);
    let submerged_volume = frac * total_volume;
    WaterCouplingForcesQuery {
        drag_coeff: draw(state, 0.0, 2.0),
        fluid_density: draw(state, 1.0, 1200.0),
        area: draw(state, 0.0, 5.0),
        rel_speed: draw(state, 0.0, 30.0),
        added_mass_coeff: draw(state, 0.0, 2.0),
        displaced_volume: draw(state, 0.0, 8.0),
        submerged_volume,
        total_volume,
    }
}

/// The deterministic hand-chosen fixtures, each exercising a distinct branch and
/// kept clear of every discrete tie.
fn fixture_queries() -> Vec<WaterCouplingForcesQuery> {
    vec![
        // Typical mid-range body, fraction well inside `(0, 1)`.
        WaterCouplingForcesQuery {
            drag_coeff: 1.2,
            fluid_density: 1000.0,
            area: 0.75,
            rel_speed: 4.5,
            added_mass_coeff: 0.5,
            displaced_volume: 0.3,
            submerged_volume: 0.4,
            total_volume: 1.0,
        },
        // Zero relative speed: drag vanishes exactly on both sides.
        WaterCouplingForcesQuery {
            drag_coeff: 0.9,
            fluid_density: 998.0,
            area: 1.5,
            rel_speed: 0.0,
            added_mass_coeff: 0.8,
            displaced_volume: 1.2,
            submerged_volume: 0.6,
            total_volume: 2.0,
        },
        // Negative inputs: every `max(0)` floor drives drag and added mass to 0.
        WaterCouplingForcesQuery {
            drag_coeff: -3.0,
            fluid_density: -5.0,
            area: -1.0,
            rel_speed: -2.0,
            added_mass_coeff: -0.5,
            displaced_volume: -4.0,
            submerged_volume: 0.5,
            total_volume: 1.0,
        },
        // Submerged far exceeding total: writeback saturates to 1 on both sides.
        WaterCouplingForcesQuery {
            drag_coeff: 0.6,
            fluid_density: 1025.0,
            area: 2.0,
            rel_speed: 7.0,
            added_mass_coeff: 1.0,
            displaced_volume: 3.0,
            submerged_volume: 9.0,
            total_volume: 2.0,
        },
        // Negative submerged volume: writeback floored to 0 on both sides.
        WaterCouplingForcesQuery {
            drag_coeff: 0.4,
            fluid_density: 1000.0,
            area: 1.0,
            rel_speed: 3.0,
            added_mass_coeff: 0.3,
            displaced_volume: 0.5,
            submerged_volume: -2.0,
            total_volume: 4.0,
        },
        // Degenerate total volume (well below `EPS`): writeback short-circuits
        // to 0 on both sides.
        WaterCouplingForcesQuery {
            drag_coeff: 1.0,
            fluid_density: 1000.0,
            area: 0.5,
            rel_speed: 2.0,
            added_mass_coeff: 0.5,
            displaced_volume: 0.2,
            submerged_volume: 0.1,
            total_volume: 0.0,
        },
        // Large dense-water body: exercises larger magnitudes against the
        // relative tolerance.
        WaterCouplingForcesQuery {
            drag_coeff: 1.8,
            fluid_density: 1180.0,
            area: 4.2,
            rel_speed: 22.0,
            added_mass_coeff: 1.6,
            displaced_volume: 6.5,
            submerged_volume: 2.1,
            total_volume: 7.0,
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
        eprintln!("skipping water_coupling_forces parity: no wgpu adapter");
        return;
    };
    let gpu = GpuWaterCouplingForces::new(&ctx);
    // The host short-circuits an empty batch (a storage buffer cannot be
    // zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn fixture_bodies_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCouplingForces::new(&ctx);
    check(&ctx, &gpu, &fixture_queries());
}

#[test]
fn negative_inputs_floor_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCouplingForces::new(&ctx);
    // Every force input negative: drag and added mass must be exactly zero, and
    // the oracle agrees since the golden floors each factor with `max(0)`.
    let q = WaterCouplingForcesQuery {
        drag_coeff: -1.0,
        fluid_density: -1.0,
        area: -1.0,
        rel_speed: -5.0,
        added_mass_coeff: -1.0,
        displaced_volume: -1.0,
        submerged_volume: -1.0,
        total_volume: 3.0,
    };
    let got = gpu.evaluate(&ctx, &[q]);
    assert_eq!(got.len(), 1);
    let want = oracle(&q);
    assert!(want.drag.abs() < EPS, "fixture drag must floor to zero");
    assert!(
        want.added_mass.abs() < EPS,
        "fixture added mass must floor to zero"
    );
    assert!(
        want.writeback_fraction.abs() < EPS,
        "fixture writeback must floor to zero"
    );
    check_body(0, &got[0], &want);
}

#[test]
fn writeback_saturates_at_both_edges() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCouplingForces::new(&ctx);
    // Submerged far above total saturates to 1; submerged below zero floors the
    // numerator to 0, so the fraction is 0. Both edges overshoot with a wide
    // margin so the two sides land on the same saturated value.
    let saturated = WaterCouplingForcesQuery {
        drag_coeff: 1.0,
        fluid_density: 1000.0,
        area: 1.0,
        rel_speed: 1.0,
        added_mass_coeff: 0.5,
        displaced_volume: 1.0,
        submerged_volume: 50.0,
        total_volume: 2.0,
    };
    let floored = WaterCouplingForcesQuery {
        submerged_volume: -10.0,
        ..saturated
    };
    let got = gpu.evaluate(&ctx, &[saturated, floored]);
    assert_eq!(got.len(), 2);
    let want_sat = oracle(&saturated);
    let want_floor = oracle(&floored);
    assert!(
        (want_sat.writeback_fraction - 1.0).abs() < EPS,
        "saturated fixture must clamp to 1"
    );
    assert!(
        want_floor.writeback_fraction.abs() < EPS,
        "floored fixture must clamp to 0"
    );
    check_body(0, &got[0], &want_sat);
    check_body(1, &got[1], &want_floor);
}

#[test]
fn degenerate_total_volume_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCouplingForces::new(&ctx);
    // Total volume below `EPS` (both exactly zero and a tiny negative) short-
    // circuits the writeback to zero on both sides.
    let zero_total = WaterCouplingForcesQuery {
        drag_coeff: 0.8,
        fluid_density: 1000.0,
        area: 1.1,
        rel_speed: 3.3,
        added_mass_coeff: 0.6,
        displaced_volume: 0.9,
        submerged_volume: 0.5,
        total_volume: 0.0,
    };
    let neg_total = WaterCouplingForcesQuery {
        total_volume: -1.0,
        ..zero_total
    };
    let got = gpu.evaluate(&ctx, &[zero_total, neg_total]);
    assert_eq!(got.len(), 2);
    for (idx, q) in [zero_total, neg_total].iter().enumerate() {
        let want = oracle(q);
        assert!(
            want.writeback_fraction.abs() < EPS,
            "degenerate total must yield zero writeback"
        );
        check_body(idx, &got[idx], &want);
    }
}

#[test]
fn random_sweep_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuWaterCouplingForces::new(&ctx);
    let mut state = 0x0f1e_2d3c_4b5a_6978_u64;
    let mut queries = fixture_queries();
    // Several workgroups' worth of random bodies pin every output across a wide
    // span of coefficients, speeds and submersion ratios.
    for _ in 0..256 {
        queries.push(random_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}
