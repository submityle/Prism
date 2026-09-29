//! End-to-end water subsystem integration: the Extract, Prepare, and Queue
//! contract layer.
//!
//! This module owns no new simulation math. It is the orchestration layer that
//! wires the sibling solver, reconstruction, caustics, ocean-`LOD`,
//! solver-transition, and budget modules into one per-frame data flow, mirroring
//! the three-stage shape production renderers expose (an `Extract` that snapshots
//! the frame's inputs, a `Prepare` that resolves every routing decision, and a
//! `Queue` that arbitrates the shared budget and bins the work for dispatch).
//!
//! 1. **Extract** ([`extract`]) — snapshots one [`WaterBody`] against a
//!    per-frame [`WaterViewSample`] into a flat, `GPU`-agnostic [`WaterExtract`]:
//!    the geometry class, solver bucket, shading frontend, clamped camera
//!    distance and quality bias, live particle count, and the derived work
//!    magnitudes (solver cells, displacement vertices, foam cells). No decision
//!    is taken here; it only gathers.
//! 2. **Prepare** ([`prepare`]) — resolves every distance- and budget-driven
//!    routing decision into a [`WaterPrepare`]: the solver-blend weights
//!    ([`super::transition`]), the caustics route ([`super::caustics`]), the
//!    surface-reconstruction route for particle bodies ([`super::reconstruct`]),
//!    the pressure solver for `FLIP`/`APIC` bodies ([`super::flip`]), the shared
//!    advanced base the frontend consumes ([`super::SharedBaseServices`]), and
//!    the per-stage work costs charged against the budget.
//! 3. **Queue** ([`queue`]) — turns the prepared bodies into
//!    [`super::budget::WaterJobRequest`]s, arbitrates them through
//!    [`super::budget::plan_water`] against the frame [`WaterBudget`], and bins
//!    the ocean bodies into a [`super::ocean_lod::OceanClipmapPlan`] for indirect
//!    draw submission.
//!
//! Every stage is a pure, deterministic function of its inputs, so a frame plans
//! identically on any thread and the `CPU` reference matches a future `GPU`
//! path. The `GPU` compute/draw kernels these decisions dispatch are described
//! in [`super::kernels`]; they are out of scope for this `CPU`-verifiable layer.

use alloc::vec::Vec;

use super::breaking::BreakingSample;
use super::budget::{plan_water, WaterJobKind, WaterJobRequest, WaterSolvePlan};
use super::caustics::{select_caustics, CausticsMethod, CausticsThresholds};
use super::coupling_frame::{plan_coupling_frame, CouplingFramePlan, CouplingInputs};
use super::flip::{select_pressure_solver, PressureSolver, PressureSolverThresholds};
use super::ocean_lod::{bin_ocean_patches, OceanClipmapConfig, OceanClipmapPlan};
use super::optics::{plan_optics, OpticsInputs, OpticsPlan};
use super::profile::WaterSimProfile;
use super::reconstruct::{
    select_reconstruction, ReconstructionContext, ReconstructionMethod, ReconstructionThresholds,
};
use super::shoreline::{plan_shoreline, ShorelineInputs, ShorelinePlan};
use super::simulation::{plan_sim, SimInputs, SimStepPlan};
use super::surface_fx::{plan_surface_fx, SurfaceFxInputs, SurfaceFxPlan};
use super::transition::{solver_blend_weights, SolverBlendWeights, TransitionBands};
use super::underwater::RgbColor;
use super::wetness::SurfaceMoisture;
use super::{
    ShadingFrontend, SharedBaseServices, SolverKind, Vec3, WaterBody, WaterBodyHandle, WaterBudget,
    WaterKind,
};

/// The live, per-frame dynamic state sampled from the running water solve that
/// the sim, surface, shoreline, optics, and coupling planners consume.
///
/// The renderer fills this each frame from the previous step's fields (crest
/// steepness and folding for breaking, local flow for foam decay, view geometry
/// for optics, submerged volume for coupling). Every field is a plain scalar or
/// `Copy` sub-record so the sample threads through the pipeline without heap
/// traffic; [`WaterDynamics::default`] is the quiescent state (no motion, no
/// breaking, no coupling) used for spectral bodies and tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterDynamics {
    /// Frame duration in seconds, shared by the sim and coupling schedules.
    pub frame_dt: f32,
    /// Grid cell size `dx` in meters; `0` asks the pipeline to derive it from
    /// the body's domain extent and grid resolution.
    pub cell_size: f32,
    /// `SWE` local maximum signal speed `|u| + sqrt(g*h)` in m/s.
    pub max_signal_speed: f32,
    /// Frame minimum surface Jacobian (choppy-displacement fold measure).
    pub min_jacobian: f32,
    /// Breaking metrics (steepness, Jacobian, curvature) at the crest sample.
    pub breaking: BreakingSample,
    /// Surface tangent (crest flow direction) for the spray jet.
    pub crest_tangent: Vec3,
    /// Surface normal (upward jet direction) for the spray jet.
    pub surface_normal: Vec3,
    /// Local surface flow speed driving foam decay.
    pub flow_speed: f32,
    /// World height of the shoreline surface sample, in meters.
    pub sample_y: f32,
    /// World height of the local water surface, in meters.
    pub water_surface_y: f32,
    /// Total water column depth at the shoreline sample, in meters.
    pub water_depth: f32,
    /// Distance of the shoreline sample above the waterline, in meters.
    pub dist_above_water: f32,
    /// Moisture state carried in from the previous frame.
    pub moisture: SurfaceMoisture,
    /// Normalized rain drive for wetting and puddle fill.
    pub rain_rate: f32,
    /// Sine of the incidence angle at the water interface (optics).
    pub sin_incidence: f32,
    /// View-ray depth through the medium, in meters (optics).
    pub view_depth: f32,
    /// Surface `RGB` color prior to depth attenuation (optics).
    pub surface_color: RgbColor,
    /// Incident surface light intensity (optics).
    pub surface_light: f32,
    /// Cosine of the scattering angle for the phase function (optics).
    pub cos_scatter: f32,
    /// Length of the light shaft used for godray inscatter (optics).
    pub shaft_length: f32,
    /// Number of two-way coupling field queries requested this frame.
    pub coupling_query_count: u32,
    /// Fastest relative body/fluid speed this frame, for coupling scheduling.
    pub max_rel_speed: f32,
    /// Volume of the coupled body currently below the surface.
    pub submerged_volume: f32,
    /// Total volume of the coupled body.
    pub total_volume: f32,
    /// Cross-sectional area presented to the flow, for coupling drag.
    pub cross_section: f32,
    /// Relative body/fluid speed used to evaluate coupling drag.
    pub rel_speed: f32,
}

impl Default for WaterDynamics {
    fn default() -> Self {
        Self {
            frame_dt: 0.0,
            cell_size: 0.0,
            max_signal_speed: 0.0,
            min_jacobian: 1.0,
            breaking: BreakingSample {
                steepness: 0.0,
                jacobian: 1.0,
                curvature: 0.0,
            },
            crest_tangent: Vec3::ZERO,
            surface_normal: Vec3::new(0.0, 1.0, 0.0),
            flow_speed: 0.0,
            sample_y: 0.0,
            water_surface_y: 0.0,
            water_depth: 0.0,
            dist_above_water: 0.0,
            moisture: SurfaceMoisture {
                wetness: 0.0,
                puddle_depth: 0.0,
            },
            rain_rate: 0.0,
            sin_incidence: 0.0,
            view_depth: 0.0,
            surface_color: RgbColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
            surface_light: 0.0,
            cos_scatter: 1.0,
            shaft_length: 0.0,
            coupling_query_count: 0,
            max_rel_speed: 0.0,
            submerged_volume: 0.0,
            total_volume: 0.0,
            cross_section: 0.0,
            rel_speed: 0.0,
        }
    }
}

/// The per-frame view- and budget-dependent inputs for one water body that are
/// not part of its static [`WaterBody`] description.
///
/// These come from the renderer each frame: how far the camera is, the active
/// quality preference, and the live particle population of a volumetric domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterViewSample {
    /// Distance from the camera to the body, in world units (clamped at zero on
    /// extract).
    pub camera_distance: f32,
    /// Quality preference in `0..=1` (clamped on extract); higher values push
    /// the expensive routes farther out.
    pub quality_bias: f32,
    /// Live particle count for a volumetric (`PBF` / `FLIP`) domain; `0` for
    /// height-field and spectral bodies.
    pub particle_count: u32,
    /// Live per-frame dynamic state feeding the sim, surface, shoreline, optics,
    /// and coupling planners.
    pub dynamics: WaterDynamics,
}

impl Default for WaterViewSample {
    fn default() -> Self {
        Self {
            camera_distance: 0.0,
            quality_bias: 0.0,
            particle_count: 0,
            dynamics: WaterDynamics::default(),
        }
    }
}

/// One water body paired with its per-frame view sample, the unit of work the
/// pipeline consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFrameInput {
    /// Static description of the body.
    pub body: WaterBody,
    /// This frame's view- and budget-dependent sample for the body.
    pub sample: WaterViewSample,
}

/// The flattened, decision-free snapshot of one body for a frame.
///
/// Produced by [`extract`]. It carries only the quantities the [`prepare`] stage
/// needs, with the raw grid resolution already expanded into concrete work
/// magnitudes so the later stages never re-derive them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterExtract {
    /// Stable identity of the body.
    pub body: WaterBodyHandle,
    /// Geometry class.
    pub kind: WaterKind,
    /// Solver bucket.
    pub solver: SolverKind,
    /// Lighting-response frontend.
    pub frontend: ShadingFrontend,
    /// Camera distance in world units, clamped to be non-negative.
    pub camera_distance: f32,
    /// Quality preference clamped into `0..=1`.
    pub quality_bias: f32,
    /// Live particle count for volumetric domains.
    pub particle_count: u32,
    /// Number of solver grid cells this body iterates: the `MAC` / height-field
    /// grid population, derived as `grid_resolution` squared for height fields
    /// and cubed for volumetric domains. A design-target magnitude, not a
    /// measured cost.
    pub solver_cell_count: u32,
    /// Displacement-mesh vertices generated per frame for a height-field /
    /// spectral body: `grid_resolution` squared, scaled by the cascade count for
    /// an ocean. Zero for volumetric bodies.
    pub displacement_vertex_count: u32,
    /// Foam-advection cells for a body that produces foam (ocean / surface):
    /// `grid_resolution` squared. Zero for volumetric bodies.
    pub foam_cell_count: u32,
    /// Aggregate per-body planner tuning, forwarded verbatim from the body.
    pub profile: WaterSimProfile,
    /// Live per-frame dynamic state forwarded verbatim from the view sample.
    pub dynamics: WaterDynamics,
    /// Resolved grid cell size `dx` in meters: the dynamics value when the
    /// renderer supplies one, otherwise derived from the domain extent and grid
    /// resolution so the sim / coupling schedules never divide by zero.
    pub cell_size: f32,
}

/// Shared per-frame thresholds and layouts every body is resolved against.
///
/// The renderer builds one of these per frame from the active quality profile;
/// [`prepare`] and [`queue`] read it without mutating it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFrameConfig {
    /// Distance / particle thresholds for surface-reconstruction routing.
    pub reconstruction: ReconstructionThresholds,
    /// Distance thresholds for caustics routing.
    pub caustics: CausticsThresholds,
    /// Cell-count thresholds for pressure-solver selection.
    pub pressure: PressureSolverThresholds,
    /// Crossfade bands for the solver-transition blend.
    pub transition: TransitionBands,
    /// Concentric-ring layout for ocean clipmap binning.
    pub clipmap: OceanClipmapConfig,
}

/// The fully resolved per-frame plan for one body.
///
/// Produced by [`prepare`]. Every routing decision the body needs this frame is
/// materialized here; the [`queue`] stage reads it to build budget requests and
/// clipmap bins but takes no further per-body decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterPrepare {
    /// Stable identity of the body.
    pub body: WaterBodyHandle,
    /// Geometry class.
    pub kind: WaterKind,
    /// Solver bucket.
    pub solver: SolverKind,
    /// Lighting-response frontend.
    pub frontend: ShadingFrontend,
    /// The shared advanced base the frontend consumes. Every frontend, `NPR`
    /// included, returns the full base, so this is always
    /// [`SharedBaseServices::ALL`].
    pub shared_base: SharedBaseServices,
    /// Camera distance carried forward for budget-priority ordering.
    pub camera_distance: f32,
    /// Solver-blend weights (particle / shallow-water / spectral) at this
    /// distance; the three always sum to one.
    pub blend: SolverBlendWeights,
    /// Caustics route selected for this body.
    pub caustics: CausticsMethod,
    /// Surface-reconstruction route, present only for particle-based bodies.
    pub reconstruction: Option<ReconstructionMethod>,
    /// Pressure-projection solver, present only for `FLIP`/`APIC` bodies.
    pub pressure_solver: Option<PressureSolver>,
    /// Solver-step work units charged against the solve-step quota.
    pub solve_cost: u32,
    /// Reconstruction work units charged against the reconstruct quota; `0` for
    /// non-particle bodies.
    pub reconstruct_cost: u32,
    /// Displacement work units charged against the displacement quota; `0` for
    /// volumetric bodies.
    pub displacement_cost: u32,
    /// Foam-advection work units charged against the foam quota; `0` for bodies
    /// that produce no foam.
    pub foam_cost: u32,
    /// Per-frame solver-stepping schedule (sub-steps, stable dt, folding).
    pub sim: SimStepPlan,
    /// Breaking / foam / crest-spray surface-effect plan for the sample.
    pub surface_fx: SurfaceFxPlan,
    /// Waterline transition and surface-wetness plan for the shoreline sample.
    pub shoreline: ShorelinePlan,
    /// Spectral refraction / extinction / scattering optics plan.
    pub optics: OpticsPlan,
    /// Two-way rigid-body coupling forces and read-back schedule.
    pub coupling: CouplingFramePlan,
}

/// The arbitrated, dispatch-ready result for a frame.
///
/// Produced by [`queue`]. It bundles the budget-admitted job schedule and the
/// ocean clipmap draw bins so the backend can walk one structure per stage.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WaterQueue {
    /// The budget-arbitrated job schedule (admitted and deferred jobs).
    pub plan: WaterSolvePlan,
    /// Ocean patches binned by clipmap ring for indirect draw submission.
    pub clipmap: OceanClipmapPlan,
}

/// The complete per-frame plan: the intermediate stage outputs plus the final
/// queue, returned together by [`plan_frame`] for inspection and dispatch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WaterFramePlan {
    /// The extracted snapshot of every input body, in input order.
    pub extracts: Vec<WaterExtract>,
    /// The resolved plan for every body, in input order.
    pub prepares: Vec<WaterPrepare>,
    /// The arbitrated, dispatch-ready queue.
    pub queue: WaterQueue,
}

/// Squared, then cubed, saturating powers of the grid resolution.
///
/// Returns `(res^2, res^3)` with saturating multiplication so an extreme
/// resolution can never overflow into a small wrapped magnitude.
fn grid_magnitudes(resolution: u32) -> (u32, u32) {
    let sq = resolution.saturating_mul(resolution);
    let cube = sq.saturating_mul(resolution);
    (sq, cube)
}

/// Snapshots one body against its view sample into a decision-free
/// [`WaterExtract`].
///
/// Pure and deterministic: the camera distance is clamped to be non-negative,
/// the quality bias into `0..=1`, and the grid resolution is expanded into the
/// solver / displacement / foam work magnitudes the later stages consume.
#[must_use]
pub fn extract(body: &WaterBody, sample: WaterViewSample) -> WaterExtract {
    let (sq, cube) = grid_magnitudes(body.grid_resolution);
    let solver_cell_count = match body.kind {
        WaterKind::Volume => cube,
        WaterKind::Ocean | WaterKind::Surface => sq,
    };
    let displacement_vertex_count = match body.kind {
        WaterKind::Ocean => sq.saturating_mul(body.cascade_count.max(1)),
        WaterKind::Surface => sq,
        WaterKind::Volume => 0,
    };
    let foam_cell_count = match body.kind {
        WaterKind::Ocean | WaterKind::Surface => sq,
        WaterKind::Volume => 0,
    };
    let cell_size = if sample.dynamics.cell_size > super::EPS {
        sample.dynamics.cell_size
    } else {
        let span = body.domain_half_extent.x.abs() * 2.0;
        let res = body.grid_resolution.max(1) as f32;
        (span / res).max(super::EPS)
    };
    WaterExtract {
        body: body.handle,
        kind: body.kind,
        solver: body.solver,
        frontend: body.frontend,
        camera_distance: sample.camera_distance.max(0.0),
        quality_bias: sample.quality_bias.clamp(0.0, 1.0),
        particle_count: sample.particle_count,
        solver_cell_count,
        displacement_vertex_count,
        foam_cell_count,
        profile: body.profile,
        dynamics: sample.dynamics,
        cell_size,
    }
}

/// Resolves every routing decision for one extracted body into a
/// [`WaterPrepare`].
///
/// Pure and deterministic. The solver-blend weights, caustics route, and (for
/// particle bodies) the reconstruction route and pressure solver are drawn from
/// the sibling selection functions; the per-stage work costs are gated by the
/// solver family so a body is only charged for the stages it actually runs.
#[must_use]
pub fn prepare(extract: WaterExtract, config: &WaterFrameConfig) -> WaterPrepare {
    let blend = solver_blend_weights(extract.camera_distance, config.transition);
    let caustics = select_caustics(
        extract.camera_distance,
        extract.quality_bias,
        config.caustics,
    );

    let reconstruction = if extract.solver.is_particle_based() {
        let ctx = ReconstructionContext {
            camera_distance: extract.camera_distance,
            particle_count: extract.particle_count,
            quality_bias: extract.quality_bias,
        };
        Some(select_reconstruction(ctx, config.reconstruction))
    } else {
        None
    };

    let pressure_solver = if matches!(extract.solver, SolverKind::FlipApic) {
        Some(select_pressure_solver(
            extract.solver_cell_count,
            extract.quality_bias,
            config.pressure,
        ))
    } else {
        None
    };

    let solve_cost = if extract.solver.is_particle_based() {
        extract.particle_count
    } else {
        extract.solver_cell_count
    };
    let reconstruct_cost = if extract.solver.is_particle_based() {
        extract.solver_cell_count
    } else {
        0
    };
    let displacement_cost = if extract.solver.is_height_field() {
        extract.displacement_vertex_count
    } else {
        0
    };
    let foam_cost = extract.foam_cell_count;

    let dynamics = extract.dynamics;
    let sim = plan_sim(
        extract.profile.sim,
        SimInputs {
            solver: extract.solver,
            frame_dt: dynamics.frame_dt,
            cell_size: extract.cell_size,
            max_signal_speed: dynamics.max_signal_speed,
            min_jacobian: dynamics.min_jacobian,
        },
    );
    let surface_fx = plan_surface_fx(
        extract.profile.surface_fx,
        SurfaceFxInputs {
            sample: dynamics.breaking,
            tangent: dynamics.crest_tangent,
            normal: dynamics.surface_normal,
            flow_speed: dynamics.flow_speed,
        },
    );
    let shoreline = plan_shoreline(
        extract.profile.shoreline,
        ShorelineInputs {
            sample_y: dynamics.sample_y,
            water_surface_y: dynamics.water_surface_y,
            water_depth: dynamics.water_depth,
            dist_above_water: dynamics.dist_above_water,
            moisture: dynamics.moisture,
            rain_rate: dynamics.rain_rate,
            dt: dynamics.frame_dt,
        },
    );
    let optics = plan_optics(
        extract.profile.optics,
        OpticsInputs {
            sin_incidence: dynamics.sin_incidence,
            view_depth: dynamics.view_depth,
            surface_color: dynamics.surface_color,
            surface_light: dynamics.surface_light,
            cos_scatter: dynamics.cos_scatter,
            shaft_length: dynamics.shaft_length,
        },
    );
    let coupling = plan_coupling_frame(
        extract.profile.coupling,
        CouplingInputs {
            query_count: dynamics.coupling_query_count,
            max_rel_speed: dynamics.max_rel_speed,
            frame_dt: dynamics.frame_dt,
            cell_size: extract.cell_size,
            submerged_volume: dynamics.submerged_volume,
            total_volume: dynamics.total_volume,
            cross_section: dynamics.cross_section,
            rel_speed: dynamics.rel_speed,
        },
    );

    WaterPrepare {
        body: extract.body,
        kind: extract.kind,
        solver: extract.solver,
        frontend: extract.frontend,
        shared_base: extract.frontend.shared_base(),
        camera_distance: extract.camera_distance,
        blend,
        caustics,
        reconstruction,
        pressure_solver,
        solve_cost,
        reconstruct_cost,
        displacement_cost,
        foam_cost,
        sim,
        surface_fx,
        shoreline,
        optics,
        coupling,
    }
}

/// Priority ceiling from which a body's distance is subtracted, so a closer body
/// (smaller distance) yields a higher priority and is admitted first.
const DISTANCE_PRIORITY_BASE: u32 = 1_000_000;

/// Maps a camera distance to a descending budget priority.
///
/// Closer bodies (smaller distance) get a higher priority; the subtraction
/// saturates so a distance beyond the base clamps to the lowest priority rather
/// than wrapping.
fn distance_priority(distance: f32) -> u32 {
    let d = distance.max(0.0);
    let steps = if d >= DISTANCE_PRIORITY_BASE as f32 {
        DISTANCE_PRIORITY_BASE
    } else {
        d as u32
    };
    DISTANCE_PRIORITY_BASE.saturating_sub(steps)
}

/// Pushes the budget requests for one prepared body onto `requests`.
///
/// A solve-step job is always emitted; the reconstruct, displacement, and foam
/// jobs are emitted only when their gated cost is non-zero, so a body never
/// competes for a quota it does not use. All jobs share the body's
/// distance-derived priority.
fn push_requests(prepare: &WaterPrepare, requests: &mut Vec<WaterJobRequest>) {
    let priority = distance_priority(prepare.camera_distance);
    requests.push(WaterJobRequest {
        handle: prepare.body,
        kind: WaterJobKind::SolveStep,
        cost: prepare.solve_cost,
        priority,
    });
    if prepare.reconstruct_cost > 0 {
        requests.push(WaterJobRequest {
            handle: prepare.body,
            kind: WaterJobKind::Reconstruct,
            cost: prepare.reconstruct_cost,
            priority,
        });
    }
    if prepare.displacement_cost > 0 {
        requests.push(WaterJobRequest {
            handle: prepare.body,
            kind: WaterJobKind::Displacement,
            cost: prepare.displacement_cost,
            priority,
        });
    }
    if prepare.foam_cost > 0 {
        requests.push(WaterJobRequest {
            handle: prepare.body,
            kind: WaterJobKind::FoamAdvect,
            cost: prepare.foam_cost,
            priority,
        });
    }
    if prepare.surface_fx.spray.count > 0 {
        requests.push(WaterJobRequest {
            handle: prepare.body,
            kind: WaterJobKind::SprayEmit,
            cost: prepare.surface_fx.spray.count,
            priority,
        });
    }
    if prepare.coupling.plan.readback_batch > 0 {
        requests.push(WaterJobRequest {
            handle: prepare.body,
            kind: WaterJobKind::Coupling,
            cost: prepare.coupling.plan.readback_batch,
            priority,
        });
    }
}

/// Arbitrates the prepared bodies against the frame budget and bins the ocean
/// bodies for draw submission.
///
/// Each body contributes one solve-step job plus the reconstruct / displacement
/// / foam jobs its family runs; [`super::budget::plan_water`] then admits and
/// defers them per independent quota. The ocean bodies are separately resolved
/// into clipmap ring buckets. Deterministic in the input order.
#[must_use]
pub fn queue(
    prepares: &[WaterPrepare],
    budget: WaterBudget,
    clipmap: OceanClipmapConfig,
) -> WaterQueue {
    let mut requests: Vec<WaterJobRequest> = Vec::new();
    for prepare in prepares {
        push_requests(prepare, &mut requests);
    }
    let plan = plan_water(&requests, budget);

    let mut ocean_bodies: Vec<WaterBodyHandle> = Vec::new();
    let mut ocean_distances: Vec<f32> = Vec::new();
    for prepare in prepares {
        if matches!(prepare.kind, WaterKind::Ocean) {
            ocean_bodies.push(prepare.body);
            ocean_distances.push(prepare.camera_distance);
        }
    }
    let clipmap_plan = bin_ocean_patches(&ocean_bodies, &ocean_distances, clipmap);

    WaterQueue {
        plan,
        clipmap: clipmap_plan,
    }
}

/// Runs the whole Extract, Prepare, Queue flow over a set of input bodies.
///
/// Convenience orchestrator: extracts every input, prepares each extract against
/// `config`, and queues the prepared bodies against `budget`, returning all
/// three stage outputs. Pure and deterministic in the input order.
#[must_use]
pub fn plan_frame(
    inputs: &[WaterFrameInput],
    config: &WaterFrameConfig,
    budget: WaterBudget,
) -> WaterFramePlan {
    let mut extracts: Vec<WaterExtract> = Vec::with_capacity(inputs.len());
    for input in inputs {
        extracts.push(extract(&input.body, input.sample));
    }
    let mut prepares: Vec<WaterPrepare> = Vec::with_capacity(extracts.len());
    for &snapshot in &extracts {
        prepares.push(prepare(snapshot, config));
    }
    let queue = queue(&prepares, budget, config.clipmap);
    WaterFramePlan {
        extracts,
        prepares,
        queue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deformation::DeformationHandle;

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

    fn body(handle: u32, kind: WaterKind, solver: SolverKind) -> WaterBody {
        WaterBody {
            handle: WaterBodyHandle(handle),
            kind,
            solver,
            frontend: ShadingFrontend::Pbr,
            deformation: DeformationHandle(handle),
            grid_resolution: 16,
            cascade_count: 4,
            domain_half_extent: Vec3::new(10.0, 10.0, 10.0),
            still_water_level: 0.0,
            profile: WaterSimProfile::physical_water(),
        }
    }

    fn sample(distance: f32, particles: u32) -> WaterViewSample {
        WaterViewSample {
            camera_distance: distance,
            quality_bias: 0.0,
            particle_count: particles,
            dynamics: WaterDynamics::default(),
        }
    }

    #[test]
    fn extract_clamps_and_expands_grid() {
        let b = body(0, WaterKind::Volume, SolverKind::FlipApic);
        let e = extract(&b, sample(-5.0, 500));
        // Negative distance is clamped to zero.
        assert_eq!(e.camera_distance, 0.0);
        // Volume solver cells are the cube of the resolution.
        assert_eq!(e.solver_cell_count, 16 * 16 * 16);
        // Volume bodies generate no displacement or foam.
        assert_eq!(e.displacement_vertex_count, 0);
        assert_eq!(e.foam_cell_count, 0);
        assert_eq!(e.particle_count, 500);
    }

    #[test]
    fn extract_ocean_scales_displacement_by_cascade() {
        let b = body(1, WaterKind::Ocean, SolverKind::SpectralIfft);
        let e = extract(&b, sample(0.0, 0));
        assert_eq!(e.solver_cell_count, 16 * 16);
        assert_eq!(e.displacement_vertex_count, 16 * 16 * 4);
        assert_eq!(e.foam_cell_count, 16 * 16);
    }

    #[test]
    fn prepare_reconstruction_only_for_particle_bodies() {
        let ocean = prepare(
            extract(
                &body(0, WaterKind::Ocean, SolverKind::SpectralIfft),
                sample(5.0, 0),
            ),
            &CONFIG,
        );
        assert!(ocean.reconstruction.is_none());
        assert!(ocean.pressure_solver.is_none());

        let volume = prepare(
            extract(
                &body(1, WaterKind::Volume, SolverKind::Pbf),
                sample(5.0, 1000),
            ),
            &CONFIG,
        );
        assert!(volume.reconstruction.is_some());
        // PBF is particle based but not FLIP/APIC, so no pressure solver.
        assert!(volume.pressure_solver.is_none());

        let flip = prepare(
            extract(
                &body(2, WaterKind::Volume, SolverKind::FlipApic),
                sample(5.0, 1000),
            ),
            &CONFIG,
        );
        assert!(flip.reconstruction.is_some());
        assert!(flip.pressure_solver.is_some());
    }

    #[test]
    fn prepare_costs_are_gated_by_solver_family() {
        let ocean = prepare(
            extract(
                &body(0, WaterKind::Ocean, SolverKind::SpectralIfft),
                sample(5.0, 0),
            ),
            &CONFIG,
        );
        // Height-field body: solve + displacement + foam, no reconstruction.
        assert!(ocean.solve_cost > 0);
        assert!(ocean.displacement_cost > 0);
        assert!(ocean.foam_cost > 0);
        assert_eq!(ocean.reconstruct_cost, 0);

        let volume = prepare(
            extract(
                &body(1, WaterKind::Volume, SolverKind::FlipApic),
                sample(5.0, 800),
            ),
            &CONFIG,
        );
        // Particle body: solve (by particle count) + reconstruction, no
        // displacement or foam.
        assert_eq!(volume.solve_cost, 800);
        assert!(volume.reconstruct_cost > 0);
        assert_eq!(volume.displacement_cost, 0);
        assert_eq!(volume.foam_cost, 0);
    }

    #[test]
    fn prepare_every_frontend_shares_full_base() {
        for frontend in [
            ShadingFrontend::Pbr,
            ShadingFrontend::Npr,
            ShadingFrontend::Custom,
            ShadingFrontend::Hybrid,
        ] {
            let mut b = body(0, WaterKind::Ocean, SolverKind::SpectralIfft);
            b.frontend = frontend;
            let p = prepare(extract(&b, sample(5.0, 0)), &CONFIG);
            assert_eq!(p.shared_base, SharedBaseServices::ALL);
        }
    }

    #[test]
    fn prepare_blend_weights_sum_to_one() {
        let mut d = 0.0;
        while d <= 400.0 {
            let p = prepare(
                extract(
                    &body(0, WaterKind::Surface, SolverKind::ShallowWater),
                    sample(d, 0),
                ),
                &CONFIG,
            );
            assert!((p.blend.sum() - 1.0).abs() < super::super::EPS);
            d += 7.0;
        }
    }

    #[test]
    fn queue_closer_body_is_scheduled_first() {
        let near = prepare(
            extract(
                &body(7, WaterKind::Ocean, SolverKind::SpectralIfft),
                sample(1.0, 0),
            ),
            &CONFIG,
        );
        let far = prepare(
            extract(
                &body(3, WaterKind::Ocean, SolverKind::SpectralIfft),
                sample(500.0, 0),
            ),
            &CONFIG,
        );
        // Pass the far body first to prove the schedule reorders by priority.
        let q = queue(&[far, near], BUDGET, CONFIG.clipmap);
        let solve_handles = q.plan.handles_of_kind(WaterJobKind::SolveStep);
        assert_eq!(solve_handles.first(), Some(&WaterBodyHandle(7)));
    }

    #[test]
    fn queue_defers_when_budget_is_tight() {
        // A budget that admits only the first solve step per the forward-progress
        // guarantee, deferring the rest.
        let tight = WaterBudget {
            solve_steps_per_frame: 1,
            reconstruct_cells_per_frame: 1,
            displacement_vertices_per_frame: 1,
            foam_cells_per_frame: 1,
            spray_bursts_per_frame: 1,
            coupling_queries_per_frame: 1,
        };
        let a = prepare(
            extract(
                &body(0, WaterKind::Ocean, SolverKind::SpectralIfft),
                sample(1.0, 0),
            ),
            &CONFIG,
        );
        let b = prepare(
            extract(
                &body(1, WaterKind::Ocean, SolverKind::SpectralIfft),
                sample(2.0, 0),
            ),
            &CONFIG,
        );
        let q = queue(&[a, b], tight, CONFIG.clipmap);
        // First solve step is admitted (forward progress); the second defers.
        assert_eq!(q.plan.count_of_kind(WaterJobKind::SolveStep), 1);
        assert!(!q.plan.deferred.is_empty());
    }

    #[test]
    fn queue_bins_only_ocean_bodies() {
        let ocean = prepare(
            extract(
                &body(0, WaterKind::Ocean, SolverKind::SpectralIfft),
                sample(40.0, 0),
            ),
            &CONFIG,
        );
        let surface = prepare(
            extract(
                &body(1, WaterKind::Surface, SolverKind::ShallowWater),
                sample(5.0, 0),
            ),
            &CONFIG,
        );
        let volume = prepare(
            extract(
                &body(2, WaterKind::Volume, SolverKind::Pbf),
                sample(5.0, 100),
            ),
            &CONFIG,
        );
        let q = queue(&[ocean, surface, volume], BUDGET, CONFIG.clipmap);
        // Only the single ocean body lands in the clipmap plan.
        assert_eq!(q.clipmap.total(), 1);
    }

    #[test]
    fn plan_frame_is_deterministic() {
        let inputs = [
            WaterFrameInput {
                body: body(0, WaterKind::Ocean, SolverKind::SpectralIfft),
                sample: sample(30.0, 0),
            },
            WaterFrameInput {
                body: body(1, WaterKind::Volume, SolverKind::FlipApic),
                sample: sample(5.0, 2000),
            },
        ];
        let a = plan_frame(&inputs, &CONFIG, BUDGET);
        let b = plan_frame(&inputs, &CONFIG, BUDGET);
        assert_eq!(a, b);
        assert_eq!(a.extracts.len(), 2);
        assert_eq!(a.prepares.len(), 2);
    }

    #[test]
    fn plan_frame_empty_input_does_not_panic() {
        let plan = plan_frame(&[], &CONFIG, BUDGET);
        assert!(plan.extracts.is_empty());
        assert!(plan.prepares.is_empty());
        assert_eq!(plan.queue.plan.scheduled_count(), 0);
        assert!(plan.queue.clipmap.is_empty());
    }

    fn sample_with_dynamics(distance: f32, dynamics: WaterDynamics) -> WaterViewSample {
        WaterViewSample {
            camera_distance: distance,
            quality_bias: 0.0,
            particle_count: 0,
            dynamics,
        }
    }

    #[test]
    fn quiescent_dynamics_emit_no_spray_or_coupling_jobs() {
        // The default dynamics describe calm, uncoupled water, so the integrated
        // surface-effect and coupling planners must not enqueue any work.
        let calm = prepare(
            extract(
                &body(0, WaterKind::Surface, SolverKind::ShallowWater),
                sample(5.0, 0),
            ),
            &CONFIG,
        );
        let q = queue(&[calm], BUDGET, CONFIG.clipmap);
        assert_eq!(q.plan.count_of_kind(WaterJobKind::SprayEmit), 0);
        assert_eq!(q.plan.count_of_kind(WaterJobKind::Coupling), 0);
    }

    #[test]
    fn queue_emits_spray_job_for_breaking_crest() {
        // A steep, folded, high-curvature crest classifies as breaking and must
        // schedule a crest-spray emission burst.
        let dynamics = WaterDynamics {
            breaking: BreakingSample {
                steepness: 3.0,
                jacobian: -0.5,
                curvature: 8.0,
            },
            crest_tangent: Vec3::new(1.0, 0.0, 0.0),
            flow_speed: 1.0,
            ..WaterDynamics::default()
        };
        let breaking = prepare(
            extract(
                &body(4, WaterKind::Surface, SolverKind::ShallowWater),
                sample_with_dynamics(5.0, dynamics),
            ),
            &CONFIG,
        );
        assert!(breaking.surface_fx.spray.count > 0);
        let q = queue(&[breaking], BUDGET, CONFIG.clipmap);
        assert_eq!(q.plan.count_of_kind(WaterJobKind::SprayEmit), 1);
        assert_eq!(
            q.plan.handles_of_kind(WaterJobKind::SprayEmit).first(),
            Some(&WaterBodyHandle(4))
        );
    }

    #[test]
    fn queue_emits_coupling_job_when_queries_requested() {
        // Pending two-way coupling queries drive a read-back batch, which must
        // surface as a coupling job bounded by the profile read-back cap.
        let dynamics = WaterDynamics {
            coupling_query_count: 4,
            frame_dt: 1.0 / 60.0,
            submerged_volume: 0.5,
            total_volume: 1.0,
            cross_section: 0.25,
            rel_speed: 2.0,
            max_rel_speed: 2.0,
            ..WaterDynamics::default()
        };
        let coupled = prepare(
            extract(
                &body(6, WaterKind::Volume, SolverKind::Pbf),
                sample_with_dynamics(5.0, dynamics),
            ),
            &CONFIG,
        );
        assert!(coupled.coupling.plan.readback_batch > 0);
        let q = queue(&[coupled], BUDGET, CONFIG.clipmap);
        assert_eq!(q.plan.count_of_kind(WaterJobKind::Coupling), 1);
        assert_eq!(
            q.plan.handles_of_kind(WaterJobKind::Coupling).first(),
            Some(&WaterBodyHandle(6))
        );
    }

    #[test]
    fn prepare_planner_outputs_are_deterministic() {
        // The five integrated planner outputs are pure functions of the extract,
        // so preparing the same body twice yields byte-identical plans.
        let dynamics = WaterDynamics {
            breaking: BreakingSample {
                steepness: 2.0,
                jacobian: -0.2,
                curvature: 6.0,
            },
            crest_tangent: Vec3::new(1.0, 0.0, 0.0),
            flow_speed: 1.5,
            frame_dt: 1.0 / 60.0,
            max_signal_speed: 5.0,
            coupling_query_count: 3,
            total_volume: 1.0,
            submerged_volume: 0.4,
            ..WaterDynamics::default()
        };
        let b = body(2, WaterKind::Surface, SolverKind::ShallowWater);
        let first = prepare(extract(&b, sample_with_dynamics(8.0, dynamics)), &CONFIG);
        let second = prepare(extract(&b, sample_with_dynamics(8.0, dynamics)), &CONFIG);
        assert_eq!(first.sim, second.sim);
        assert_eq!(first.surface_fx, second.surface_fx);
        assert_eq!(first.shoreline, second.shoreline);
        assert_eq!(first.optics, second.optics);
        assert_eq!(first.coupling, second.coupling);
    }
}
