//! Whole-frame end-to-end golden regression for the water planning pipeline.
//!
//! The sibling per-stage tests in [`super::pipeline`] prove each decision in
//! isolation (a clamp here, a route there), but nothing pins the *aggregate*
//! shape of a realistic multi-body frame. A cross-stage regression — an extract
//! magnitude that silently changes a prepare cost that then flips a queue
//! arbitration, or a routing threshold that quietly re-bins an ocean clipmap —
//! can leave every narrow unit test green while the frame the backend actually
//! dispatches drifts. Shipping oceans (`WaveWorks`, `Crest`, `UE5` Water) guard
//! their frame graphs with exactly this kind of end-to-end snapshot.
//!
//! This module is that guard. It builds one deterministic, representative scene
//! — a near and a far spectral ocean, a shallow-water surface, and two
//! volumetric bodies (`FLIP`/`APIC` and `PBF`) at distinct distances, quality
//! biases, and shading frontends — runs the full [`super::pipeline::plan_frame`]
//! Extract/Prepare/Queue flow against a deliberately *tight* budget (so the
//! deferral path is exercised, not just the all-fits path), and locks the
//! result two ways:
//!
//! * a compact 64-bit `FNV-1a` digest over the canonical [`core::fmt::Debug`]
//!   serialization of the whole [`super::pipeline::WaterFramePlan`], which trips
//!   on *any* field drift anywhere in the plan, and
//! * a set of readable aggregate invariants (body/job counts, the
//!   `NPR`-shares-the-full-base guarantee, budget quotas never exceeded,
//!   blend weights summing to one, ocean clipmap coverage) so a tripped digest
//!   points at the stage that moved.
//!
//! Every planner input is pure arithmetic over `sqrt`-only floats, so the
//! `Debug` text — and therefore the digest — is bit-identical on every platform
//! and every thread, making the pinned constant a legitimate golden rather than
//! a host-specific fingerprint.

use alloc::format;

use crate::deformation::DeformationHandle;

use super::caustics::CausticsThresholds;
use super::flip::PressureSolverThresholds;
use super::ocean_lod::OceanClipmapConfig;
use super::pipeline::{
    plan_frame, WaterDynamics, WaterFrameConfig, WaterFrameInput, WaterFramePlan, WaterViewSample,
};
use super::profile::WaterSimProfile;
use super::reconstruct::ReconstructionThresholds;
use super::transition::TransitionBands;
use super::{
    ShadingFrontend, SharedBaseServices, SolverKind, Vec3, WaterBody, WaterBodyHandle, WaterBudget,
    WaterKind,
};

/// Tolerance for the "blend weights sum to one" invariant. The three weights are
/// a normalized partition, so their sum lands within a few `f32` ulps of one;
/// `1e-4` is far tighter than any real drift yet immune to rounding noise.
const BLEND_SUM_EPS: f32 = 1.0e-4;

/// The pinned digest of the canonical frame plan. Regenerated only by a
/// deliberate, reviewed change to the planning pipeline; a mismatch means the
/// aggregate frame drifted and must be explained before the constant is moved.
const GOLDEN_FRAME_DIGEST: u64 = 0xe258_fb08_efbd_695e;

/// Shared per-frame routing thresholds, budget layout, and clipmap rings for the
/// golden scene. Fixed values so the plan is reproducible frame to frame.
const CONFIG: WaterFrameConfig = WaterFrameConfig {
    reconstruction: ReconstructionThresholds {
        near_max_distance: 12.0,
        mid_max_distance: 60.0,
        max_volumetric_particles: 1_000_000,
    },
    caustics: CausticsThresholds {
        photon_max_distance: 12.0,
        ray_max_distance: 60.0,
    },
    pressure: PressureSolverThresholds {
        jacobi_max_cells: 10_000,
        cg_max_cells: 1_000_000,
    },
    transition: TransitionBands {
        particle_to_swe: 25.0,
        swe_to_spectral: 200.0,
        half_width: 12.0,
    },
    clipmap: OceanClipmapConfig {
        ring_count: 4,
        inner_radius: 32.0,
        radius_growth: 2.0,
        morph_fraction: 0.25,
    },
};

/// A deliberately constrained budget: the solve-step and reconstruct quotas sit
/// below the scene's aggregate demand, so the queue must admit by priority and
/// defer the rest — exercising the arbitration path the golden exists to lock.
const BUDGET: WaterBudget = WaterBudget {
    solve_steps_per_frame: 60_000,
    reconstruct_cells_per_frame: 40_000,
    displacement_vertices_per_frame: 1_000_000,
    foam_cells_per_frame: 1_000_000,
    spray_bursts_per_frame: 1_000_000,
    coupling_queries_per_frame: 1_000_000,
};

/// Builds one body of the golden scene with a distinct handle, geometry class,
/// solver, and shading frontend.
fn body(
    handle: u32,
    kind: WaterKind,
    solver: SolverKind,
    frontend: ShadingFrontend,
    grid_resolution: u32,
) -> WaterBody {
    WaterBody {
        handle: WaterBodyHandle(handle),
        kind,
        solver,
        frontend,
        deformation: DeformationHandle(handle),
        grid_resolution,
        cascade_count: 4,
        domain_half_extent: Vec3::new(16.0, 16.0, 16.0),
        still_water_level: 0.0,
        profile: WaterSimProfile::physical_water(),
    }
}

/// Builds one body's per-frame view sample at a distance, quality bias, and live
/// particle count.
fn sample(distance: f32, quality_bias: f32, particles: u32) -> WaterViewSample {
    WaterViewSample {
        camera_distance: distance,
        quality_bias,
        particle_count: particles,
        dynamics: WaterDynamics::default(),
    }
}

/// Assembles the canonical five-body scene, in a fixed input order.
fn golden_scene() -> [WaterFrameInput; 5] {
    [
        // A near spectral ocean under the physically based frontend.
        WaterFrameInput {
            body: body(
                0,
                WaterKind::Ocean,
                SolverKind::SpectralIfft,
                ShadingFrontend::Pbr,
                64,
            ),
            sample: sample(20.0, 0.6, 0),
        },
        // A far spectral ocean under the hybrid frontend (clipmap outer ring).
        WaterFrameInput {
            body: body(
                1,
                WaterKind::Ocean,
                SolverKind::SpectralIfft,
                ShadingFrontend::Hybrid,
                64,
            ),
            sample: sample(500.0, 0.2, 0),
        },
        // A shallow-water surface under the stylized (NPR) frontend.
        WaterFrameInput {
            body: body(
                2,
                WaterKind::Surface,
                SolverKind::ShallowWater,
                ShadingFrontend::Npr,
                48,
            ),
            sample: sample(15.0, 0.4, 0),
        },
        // A close FLIP/APIC volume under a custom frontend, heavy particle load.
        WaterFrameInput {
            body: body(
                3,
                WaterKind::Volume,
                SolverKind::FlipApic,
                ShadingFrontend::Custom,
                32,
            ),
            sample: sample(8.0, 0.8, 50_000),
        },
        // A mid-distance PBF volume under the physically based frontend.
        WaterFrameInput {
            body: body(
                4,
                WaterKind::Volume,
                SolverKind::Pbf,
                ShadingFrontend::Pbr,
                32,
            ),
            sample: sample(30.0, 0.3, 20_000),
        },
    ]
}

/// 64-bit `FNV-1a` hash of `bytes`. A tiny, dependency-free, order-sensitive
/// digest: identical bytes always hash identically on every platform, and any
/// single-byte change avalanches, so it is a faithful fingerprint of the plan's
/// canonical `Debug` text.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Canonical digest of a whole frame plan: the `FNV-1a` hash of its `Debug`
/// serialization. `Debug` walks every nested field in declaration order, so the
/// digest is sensitive to a change anywhere in the plan.
fn digest_plan(plan: &WaterFramePlan) -> u64 {
    let text = format!("{plan:?}");
    fnv1a64(text.as_bytes())
}

#[test]
fn whole_frame_plan_matches_the_golden_digest() {
    let scene = golden_scene();
    let plan = plan_frame(&scene, &CONFIG, BUDGET);

    // Structural invariants first, so a tripped digest is diagnosable.
    assert_eq!(
        plan.extracts.len(),
        scene.len(),
        "every input body must be extracted, in order"
    );
    assert_eq!(
        plan.prepares.len(),
        scene.len(),
        "every extracted body must be prepared, in order"
    );

    // Design §5 guarantee: every frontend — NPR included — consumes the full
    // shared advanced base, never a reduced one.
    for prepare in &plan.prepares {
        assert_eq!(
            prepare.shared_base,
            SharedBaseServices::ALL,
            "frontend {:?} must share the full advanced base",
            prepare.frontend
        );
        let blend = prepare.blend;
        let sum = blend.particle + blend.shallow_water + blend.spectral;
        assert!(
            (sum - 1.0).abs() < BLEND_SUM_EPS,
            "solver blend weights must sum to one, got {sum}"
        );
    }

    // The two spectral oceans are the only clipmap-binned bodies.
    assert_eq!(
        plan.queue.clipmap.total(),
        2,
        "both ocean bodies must land in the clipmap plan"
    );

    // The tight budget must force arbitration: some jobs admitted, some deferred.
    let scheduled = &plan.queue.plan.scheduled;
    let deferred = &plan.queue.plan.deferred;
    assert!(
        !scheduled.is_empty(),
        "the frame must admit at least one job"
    );
    assert!(
        !deferred.is_empty(),
        "the tight budget must defer at least one job"
    );

    // No quota may be overspent by what was admitted.
    assert!(plan.queue.plan.steps_used <= BUDGET.solve_steps_per_frame);
    assert!(plan.queue.plan.reconstruct_used <= BUDGET.reconstruct_cells_per_frame);
    assert!(plan.queue.plan.displacement_used <= BUDGET.displacement_vertices_per_frame);
    assert!(plan.queue.plan.foam_used <= BUDGET.foam_cells_per_frame);

    // Finally, lock the whole aggregate against the pinned golden.
    let digest = digest_plan(&plan);
    assert_eq!(
        digest, GOLDEN_FRAME_DIGEST,
        "whole-frame plan digest drifted to {digest:#018x}; if this change is intended, re-pin GOLDEN_FRAME_DIGEST"
    );
}

#[test]
fn whole_frame_plan_is_deterministic() {
    let scene = golden_scene();
    let first = plan_frame(&scene, &CONFIG, BUDGET);
    let second = plan_frame(&scene, &CONFIG, BUDGET);
    assert_eq!(
        digest_plan(&first),
        digest_plan(&second),
        "replanning the same scene must produce an identical frame plan"
    );
    assert_eq!(first, second, "the plan itself must be reproducible");
}
