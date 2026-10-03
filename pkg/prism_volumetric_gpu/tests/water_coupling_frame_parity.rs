//! Real-device parity for the per-frame coupling assembly twin:
//! [`GpuWaterCouplingFrame`](prism_volumetric_gpu::water_coupling_frame::GpuWaterCouplingFrame)
//! must reproduce the `CPU` golden
//! [`plan_coupling_frame`](prism_render_architecture::water::coupling_frame::plan_coupling_frame)
//! across hand-picked fixtures plus a randomized batch compared field-for-field.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! The golden
//! [`plan_coupling_frame`](prism_render_architecture::water::coupling_frame::plan_coupling_frame)
//! is `pub`, so each `GPU` frame plan is pinned directly against the golden run
//! on the same `profile`/`inputs`.
//!
//! # Parity criterion
//!
//! The schedule outputs (`substeps`, `readback_batch`) are `u32`, asserted with
//! exact `==`. The four forces/fractions (`buoyancy`, `drag`, `added_mass`,
//! `writeback_fraction`) are `f32`, asserted within an absolute `1e-4` or
//! relative `1e-3` tolerance (`REL_FLOOR = 1e-6`). The randomized batch keeps
//! the intermediate `crossings` quantity small, non-saturating, and away from
//! integer boundaries (or lets the sub-step cap bind it) so the `CPU` `as u32`
//! and the device `u32(floor(...))` truncate to the same integer.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::coupling_frame`；无第三方引擎源码或衍生代码。

use prism_render_architecture::water::coupling_frame::{
    plan_coupling_frame, CouplingFramePlan, CouplingInputs, CouplingProfile,
};
use prism_volumetric_gpu::water_coupling_frame::{
    GpuWaterCouplingFrame, WaterCouplingFrameQuery, WaterCouplingFrameResult,
};
use prism_volumetric_gpu::GpuContext;

/// Rest threshold matching `water::EPS`; the host mirrors the golden `cell`
/// floor when checking the `crossings` boundary.
const EPS: f32 = 1.0e-6;

/// Relative-tolerance floor so a near-zero reference magnitude does not inflate
/// the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether two continuous quantities agree within the crate tolerance:
/// absolute `1e-4` or relative `1e-3`.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff <= 1.0e-4 || diff / scale <= 1.0e-3
}

/// Computes the golden frame plan for one query.
fn expected(q: &WaterCouplingFrameQuery) -> WaterCouplingFrameResult {
    let profile = CouplingProfile {
        fluid_density: q.fluid_density,
        drag_coeff: q.drag_coeff,
        added_mass_coeff: q.added_mass_coeff,
        max_substeps: q.max_substeps,
        max_readback: q.max_readback,
    };
    let inputs = CouplingInputs {
        query_count: q.query_count,
        max_rel_speed: q.max_rel_speed,
        frame_dt: q.frame_dt,
        cell_size: q.cell_size,
        submerged_volume: q.submerged_volume,
        total_volume: q.total_volume,
        cross_section: q.cross_section,
        rel_speed: q.rel_speed,
    };
    let CouplingFramePlan {
        plan,
        buoyancy,
        drag,
        added_mass,
        writeback_fraction,
    } = plan_coupling_frame(profile, inputs);
    WaterCouplingFrameResult {
        substeps: plan.substeps,
        readback_batch: plan.readback_batch,
        buoyancy,
        drag,
        added_mass,
        writeback_fraction,
    }
}

/// Pins one `GPU` result against the golden oracle: discrete schedule fields
/// exact, continuous forces within tolerance.
fn assert_result(idx: usize, got: &WaterCouplingFrameResult, want: &WaterCouplingFrameResult) {
    assert_eq!(got.substeps, want.substeps, "result {idx}: substeps");
    assert_eq!(
        got.readback_batch, want.readback_batch,
        "result {idx}: readback_batch"
    );
    assert!(
        close(got.buoyancy, want.buoyancy),
        "result {idx}: buoyancy gpu={} cpu={}",
        got.buoyancy,
        want.buoyancy
    );
    assert!(
        close(got.drag, want.drag),
        "result {idx}: drag gpu={} cpu={}",
        got.drag,
        want.drag
    );
    assert!(
        close(got.added_mass, want.added_mass),
        "result {idx}: added_mass gpu={} cpu={}",
        got.added_mass,
        want.added_mass
    );
    assert!(
        close(got.writeback_fraction, want.writeback_fraction),
        "result {idx}: writeback_fraction gpu={} cpu={}",
        got.writeback_fraction,
        want.writeback_fraction
    );
}

/// Runs every query on the device and pins each result against the oracle.
fn run_and_check(ctx: &GpuContext, queries: &[WaterCouplingFrameQuery]) {
    let gpu = GpuWaterCouplingFrame::new(ctx);
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (idx, (q, g)) in queries.iter().zip(got.iter()).enumerate() {
        assert_result(idx, g, &expected(q));
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

/// Draws a float in `[0, 1)` from `state` using only integer work.
fn unit(state: &mut u64) -> f32 {
    (lcg(state) >> 8) as f32 / (1u32 << 24) as f32
}

/// Draws a float in `[lo, hi)` from `state`.
fn ranged(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * unit(state)
}

/// Mirrors the golden `crossings` intermediate so the host can reject samples
/// whose fractional part sits too close to an integer boundary (where `CPU`
/// `as u32` and device `u32(floor(..))` could truncate differently).
fn crossings(q: &WaterCouplingFrameQuery) -> f32 {
    let cell = q.cell_size.max(EPS);
    q.max_rel_speed.max(0.0) * q.frame_dt.max(0.0) / cell
}

/// Returns whether `crossings` is comfortably away from an integer boundary.
fn boundary_safe(c: f32) -> bool {
    let frac = c - c.floor();
    frac > 0.05 && frac < 0.95
}

/// Returns whether a fixture is parity-safe regardless of `CPU`/device
/// truncation agreement on `crossings`.
///
/// A fixture is safe when `crossings` sits comfortably away from an integer
/// boundary, when it truncates to zero (`< 0.95`), or when the sub-step `cap`
/// is low enough that it binds the result. In the last case
/// `substeps = min(1 + (crossings as u32), cap)` returns `cap` whenever
/// `cap + 1 <= crossings`, so any floating-point jitter in the truncation
/// cannot change the clamped minimum.
fn fixture_safe(q: &WaterCouplingFrameQuery) -> bool {
    let c = crossings(q);
    let cap = q.max_substeps.max(1) as f32;
    boundary_safe(c) || c < 0.95 || (cap + 1.0) <= c
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping water_coupling_frame parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuWaterCouplingFrame::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn coupling_frame_fixtures() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let queries = vec![
        // Normal frame: a few crossings, caps non-binding, partial submersion.
        WaterCouplingFrameQuery {
            fluid_density: 1000.0,
            drag_coeff: 1.0,
            added_mass_coeff: 0.5,
            max_substeps: 16,
            max_readback: 32,
            query_count: 8,
            max_rel_speed: 6.0,
            frame_dt: 0.1,
            cell_size: 1.0,
            submerged_volume: 0.3,
            total_volume: 1.0,
            cross_section: 2.0,
            rel_speed: 1.0,
        },
        // frame_dt = 0: no crossings, substeps floors at one; drag non-zero.
        WaterCouplingFrameQuery {
            fluid_density: 998.0,
            drag_coeff: 0.8,
            added_mass_coeff: 0.4,
            max_substeps: 8,
            max_readback: 64,
            query_count: 50,
            max_rel_speed: 9.0,
            frame_dt: 0.0,
            cell_size: 0.5,
            submerged_volume: 0.6,
            total_volume: 2.0,
            cross_section: 1.5,
            rel_speed: 3.0,
        },
        // rel_speed = 0: drag collapses to zero; buoyancy and added mass remain.
        WaterCouplingFrameQuery {
            fluid_density: 1025.0,
            drag_coeff: 1.2,
            added_mass_coeff: 0.6,
            max_substeps: 8,
            max_readback: 128,
            query_count: 200,
            max_rel_speed: 0.0,
            frame_dt: 0.05,
            cell_size: 0.25,
            submerged_volume: 0.4,
            total_volume: 1.0,
            cross_section: 3.0,
            rel_speed: 0.0,
        },
        // query_count > max_readback: read-back batch clamps to the cap.
        WaterCouplingFrameQuery {
            fluid_density: 1000.0,
            drag_coeff: 0.9,
            added_mass_coeff: 0.5,
            max_substeps: 32,
            max_readback: 512,
            query_count: 4000,
            max_rel_speed: 3.0,
            frame_dt: 0.02,
            cell_size: 0.5,
            submerged_volume: 0.2,
            total_volume: 1.0,
            cross_section: 2.5,
            rel_speed: 1.5,
        },
        // max_substeps = 0 -> cap = 1: substeps clamps to one though demand is
        // high (crossings = 10 is cap-bound, so truncation jitter is harmless).
        WaterCouplingFrameQuery {
            fluid_density: 1000.0,
            drag_coeff: 1.0,
            added_mass_coeff: 0.5,
            max_substeps: 0,
            max_readback: 16,
            query_count: 10,
            max_rel_speed: 50.0,
            frame_dt: 0.1,
            cell_size: 0.5,
            submerged_volume: 0.5,
            total_volume: 1.0,
            cross_section: 2.0,
            rel_speed: 2.0,
        },
        // total_volume below EPS: write-back fraction short-circuits to zero.
        WaterCouplingFrameQuery {
            fluid_density: 1000.0,
            drag_coeff: 1.0,
            added_mass_coeff: 0.5,
            max_substeps: 16,
            max_readback: 32,
            query_count: 12,
            max_rel_speed: 2.0,
            frame_dt: 0.3,
            cell_size: 1.0,
            submerged_volume: 0.0,
            total_volume: 0.0,
            cross_section: 1.0,
            rel_speed: 1.0,
        },
        // Over-submerged: write-back fraction clamps to one.
        WaterCouplingFrameQuery {
            fluid_density: 1000.0,
            drag_coeff: 1.0,
            added_mass_coeff: 0.5,
            max_substeps: 16,
            max_readback: 32,
            query_count: 5,
            max_rel_speed: 5.0,
            frame_dt: 0.1,
            cell_size: 1.0,
            submerged_volume: 5.0,
            total_volume: 1.0,
            cross_section: 2.0,
            rel_speed: 4.0,
        },
    ];
    // Keep every fixture clear of an integer crossing boundary, unless the
    // sub-step cap binds the result so truncation jitter is irrelevant.
    for q in &queries {
        assert!(
            fixture_safe(q),
            "fixture crossings={} cap={}",
            crossings(q),
            q.max_substeps.max(1)
        );
    }
    run_and_check(&ctx, &queries);
}

#[test]
fn random_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let mut state = 0x2f6e_41b8_5c7d_90a3_u64;

    let mut queries: Vec<WaterCouplingFrameQuery> = Vec::new();
    while queries.len() < 512 {
        let q = WaterCouplingFrameQuery {
            fluid_density: ranged(&mut state, 0.0, 1200.0),
            drag_coeff: ranged(&mut state, 0.0, 2.0),
            added_mass_coeff: ranged(&mut state, 0.0, 1.0),
            max_substeps: lcg(&mut state) % 64,
            max_readback: lcg(&mut state) % 4096,
            query_count: lcg(&mut state) % 4096,
            max_rel_speed: ranged(&mut state, 0.0, 20.0),
            frame_dt: ranged(&mut state, 0.0, 0.05),
            cell_size: ranged(&mut state, 0.1, 2.0),
            submerged_volume: ranged(&mut state, 0.0, 3.0),
            total_volume: ranged(&mut state, 0.2, 3.0),
            cross_section: ranged(&mut state, 0.0, 4.0),
            rel_speed: ranged(&mut state, 0.0, 10.0),
        };
        // Reject samples whose crossing count sits near an integer boundary so
        // CPU/GPU truncation always agrees.
        if fixture_safe(&q) {
            queries.push(q);
        }
    }
    run_and_check(&ctx, &queries);
}
