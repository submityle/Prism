//! End-to-end check of the data-driven water authoring path.
//!
//! Exercises the public flow a tool/runtime drives to turn an authored asset
//! into a dispatch-ready frame plan: build a [`WaterBodyAsset`] preset, compile
//! it to a validated [`CompiledWaterParams`] snapshot, assemble that into the
//! [`WaterBody`] routing contract plus its `WESL` specialization key, then run
//! the three-stage [`plan_frame`] pipeline over a mixed-bucket scene. This
//! proves the `asset` -> `assembly` -> `pipeline` modules compose across the
//! crate's public surface, not just in their isolated unit tests, and that a
//! purely authored body produces a coherent, deterministic frame plan.

use prism_render_architecture::deformation::DeformationHandle;
use prism_render_architecture::water::asset::WaterBodyAsset;
use prism_render_architecture::water::caustics::CausticsThresholds;
use prism_render_architecture::water::flip::PressureSolverThresholds;
use prism_render_architecture::water::ocean_lod::OceanClipmapConfig;
use prism_render_architecture::water::pipeline::{
    plan_frame, WaterFrameConfig, WaterFrameInput, WaterViewSample,
};
use prism_render_architecture::water::reconstruct::ReconstructionThresholds;
use prism_render_architecture::water::transition::TransitionBands;
use prism_render_architecture::water::{
    SharedBaseServices, SolverKind, Vec3, WaterBodyHandle, WaterBudget, WaterKind,
};

const CONFIG: WaterFrameConfig = WaterFrameConfig {
    reconstruction: ReconstructionThresholds {
        near_max_distance: 10.0,
        mid_max_distance: 50.0,
        max_volumetric_particles: 1_000_000,
    },
    caustics: CausticsThresholds {
        photon_max_distance: 10.0,
        ray_max_distance: 50.0,
    },
    pressure: PressureSolverThresholds {
        jacobi_max_cells: 10_000,
        cg_max_cells: 1_000_000,
    },
    transition: TransitionBands {
        particle_to_swe: 20.0,
        swe_to_spectral: 200.0,
        half_width: 10.0,
    },
    clipmap: OceanClipmapConfig {
        ring_count: 4,
        inner_radius: 32.0,
        radius_growth: 2.0,
        morph_fraction: 0.25,
    },
};

const BUDGET: WaterBudget = WaterBudget {
    solve_steps_per_frame: 1_000_000,
    reconstruct_cells_per_frame: 1_000_000,
    displacement_vertices_per_frame: 1_000_000,
    foam_cells_per_frame: 1_000_000,
    spray_bursts_per_frame: 1_000_000,
    coupling_queries_per_frame: 1_000_000,
};

fn extent() -> Vec3 {
    Vec3::new(256.0, 32.0, 256.0)
}

fn sample(distance: f32, particles: u32) -> WaterViewSample {
    WaterViewSample {
        camera_distance: distance,
        quality_bias: 0.0,
        particle_count: particles,
        ..WaterViewSample::default()
    }
}

/// Authored ocean -> compile -> assemble -> frame plan, with the specialization
/// key carrying the spectral cascade bit through the whole flow.
#[test]
fn authored_ocean_plans_a_spectral_frame() {
    let compiled = WaterBodyAsset::ocean_default(extent())
        .compile()
        .expect("ocean preset compiles");
    assert_eq!(compiled.kind, WaterKind::Ocean);
    assert_eq!(compiled.solver, SolverKind::SpectralIfft);

    let assembled = compiled.assemble(WaterBodyHandle(10), DeformationHandle(10));
    // The GPU specialization key the dispatch path keys on survives assembly.
    assert_eq!(assembled.spec_key, compiled.spec_key);
    assert!(assembled.spec_key.has_cascades());

    let inputs = [WaterFrameInput {
        body: assembled.body,
        sample: sample(5.0, 0),
    }];
    let plan = plan_frame(&inputs, &CONFIG, BUDGET);

    assert_eq!(plan.extracts.len(), 1);
    assert_eq!(plan.prepares.len(), 1);
    // Spectral oceans run the full shared advanced base regardless of frontend.
    assert_eq!(plan.prepares[0].shared_base, SharedBaseServices::ALL);
    // An ocean body does solver + displacement + foam work, so the budget must
    // admit at least one job.
    assert!(!plan.queue.plan.scheduled.is_empty());
    assert!(plan.queue.plan.steps_used > 0);
    assert!(plan.queue.plan.displacement_used > 0);
    // Ocean patches bin into the clipmap for indirect draw.
    assert!(!plan.queue.clipmap.is_empty());
}

/// Authored volume -> incompressible FLIP body routed to a pressure solve.
#[test]
fn authored_volume_plans_an_incompressible_frame() {
    let compiled = WaterBodyAsset::volume_default(extent(), 50_000)
        .compile()
        .expect("volume preset compiles");
    assert_eq!(compiled.kind, WaterKind::Volume);
    assert!(compiled.solver.is_particle_based());

    let assembled = compiled.assemble(WaterBodyHandle(20), DeformationHandle(20));
    assert!(assembled.spec_key.is_incompressible());
    // A volume body carries no spectral cascade.
    assert!(!assembled.spec_key.has_cascades());

    let inputs = [WaterFrameInput {
        body: assembled.body,
        sample: sample(5.0, 20_000),
    }];
    let plan = plan_frame(&inputs, &CONFIG, BUDGET);

    let prepare = plan.prepares[0];
    // FLIP/APIC volumes reconstruct a surface and run a pressure solver.
    assert!(prepare.reconstruction.is_some());
    assert!(prepare.pressure_solver.is_some());
    // Particle bodies produce no displacement mesh or foam field.
    assert_eq!(prepare.displacement_cost, 0);
    assert_eq!(prepare.foam_cost, 0);
    assert!(plan.queue.plan.reconstruct_used > 0);
}

/// Authored river -> shallow-water height-field body.
#[test]
fn authored_river_plans_a_height_field_frame() {
    let compiled = WaterBodyAsset::river_default(extent())
        .compile()
        .expect("river preset compiles");
    assert_eq!(compiled.kind, WaterKind::Surface);
    assert!(compiled.solver.is_height_field());

    let assembled = compiled.assemble(WaterBodyHandle(30), DeformationHandle(30));
    let inputs = [WaterFrameInput {
        body: assembled.body,
        sample: sample(5.0, 0),
    }];
    let plan = plan_frame(&inputs, &CONFIG, BUDGET);

    let prepare = plan.prepares[0];
    // Height-field bodies do not reconstruct a particle surface.
    assert!(prepare.reconstruction.is_none());
    assert!(prepare.pressure_solver.is_none());
    assert!(prepare.solve_cost > 0);
}

/// A mixed scene (far ocean + near river + local volume) plans in one call,
/// proving the pipeline mixes solver buckets as the design intends, and that
/// the whole authored flow is deterministic.
#[test]
fn mixed_authored_scene_is_deterministic() {
    let ocean = WaterBodyAsset::ocean_default(extent())
        .compile()
        .expect("ocean compiles")
        .assemble(WaterBodyHandle(1), DeformationHandle(1));
    let river = WaterBodyAsset::river_default(extent())
        .compile()
        .expect("river compiles")
        .assemble(WaterBodyHandle(2), DeformationHandle(2));
    let volume = WaterBodyAsset::volume_default(extent(), 50_000)
        .compile()
        .expect("volume compiles")
        .assemble(WaterBodyHandle(3), DeformationHandle(3));

    let inputs = [
        WaterFrameInput {
            body: ocean.body,
            sample: sample(300.0, 0),
        },
        WaterFrameInput {
            body: river.body,
            sample: sample(15.0, 0),
        },
        WaterFrameInput {
            body: volume.body,
            sample: sample(5.0, 20_000),
        },
    ];

    let first = plan_frame(&inputs, &CONFIG, BUDGET);
    let second = plan_frame(&inputs, &CONFIG, BUDGET);
    assert_eq!(first, second);

    // All three bodies extract and prepare in input order.
    assert_eq!(first.extracts.len(), 3);
    assert_eq!(first.prepares.len(), 3);
    assert_eq!(first.extracts[0].body, WaterBodyHandle(1));
    assert_eq!(first.extracts[1].body, WaterBodyHandle(2));
    assert_eq!(first.extracts[2].body, WaterBodyHandle(3));
    // Every frontend-agnostic body shares the full advanced base.
    for prepare in &first.prepares {
        assert_eq!(prepare.shared_base, SharedBaseServices::ALL);
    }
    // The generous budget admits work for the scene.
    assert!(first.queue.plan.steps_used > 0);
}
