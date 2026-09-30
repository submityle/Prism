//! `GPU`-free assembly of one cloth piece's complete solve plan.
//!
//! This is the device-independent half of the prepare stage: given a garment's
//! authored `CPU` state it produces everything the resident buffers and the
//! dispatch node need, *except* the `wgpu` handles themselves. Keeping the
//! plan device-free means the whole packing-and-scheduling pipeline is
//! deterministic and unit-testable without a `GPU`, and the device stage that
//! lands next only has to turn the plan's byte slices into buffers.
//!
//! The plan chains the three real bridges this subsystem already owns:
//!
//! 1. [`plan_constraint_upload`] partitions the authored constraints into the
//!    distance and long-range classes, colors each independently and lays them
//!    out contiguously by color; [`color_bending`] does the same for the
//!    dihedral hinges.
//! 2. [`super::pack`] packs those planned reorderings into the `#[repr(C)]`
//!    [`super::abi`] records the buffers upload.
//! 3. [`extract`] + [`prepare`] expand the class-separated per-color counts and
//!    the substep/iteration loop into the flat, ordered
//!    [`PlannedDispatch`] schedule the `Core3d` node records.
//!
//! The per-substep uniforms ([`GpuClothSimParams`] and the collision/embed
//! params) are derived here too, so the device stage and the dispatch node get
//! their initial scalars from one documented place.

use prism_render_architecture::cloth::aero_gather::VertexTriangleAdjacency;
use prism_render_architecture::cloth::bending::BendingConstraint;
use prism_render_architecture::cloth::gpu::buffers::BufferCounts;
use prism_render_architecture::cloth::gpu::pipeline::{extract, prepare, PlannedDispatch};
use prism_render_architecture::cloth::gpu::upload::{color_bending, plan_constraint_upload};
use prism_render_architecture::cloth::wind::{AeroParams, WindField};
use prism_render_architecture::cloth::{Constraint, Vec3};

use super::abi::{
    GpuClothAeroParams, GpuClothBackstop, GpuClothBackstopParams, GpuClothBendingConstraint,
    GpuClothBodyParams, GpuClothCollider, GpuClothConstraint, GpuClothEmbedBinding,
    GpuClothEmbedParams, GpuClothSelfParams, GpuClothSimParams,
};
use super::pack::{pack_bending, pack_constraints};

/// The authored `CPU` state of one cloth garment, borrowed for planning.
///
/// This mirrors the data a main-world cloth component carries. Positions pack
/// the inverse mass into `.w` (`<= 0` pins the particle); the constraint list
/// is the authored mixed-kind set (the planner partitions it), and the
/// remaining slices are already in their `#[repr(C)]` device form because they
/// have no `CPU`-golden solver type to reorder.
pub(crate) struct ClothSolveInput<'a> {
    /// Particle positions with inverse mass in `.w`.
    pub(crate) positions: &'a [[f32; 4]],
    /// Particle velocities; `.w` is unused padding.
    pub(crate) velocities: &'a [[f32; 4]],
    /// Authored distance and attachment constraints, mixed kinds.
    pub(crate) constraints: &'a [Constraint],
    /// Authored dihedral bending hinges.
    pub(crate) bending: &'a [BendingConstraint],
    /// Sim-mesh triangles (three particle indices each) the aerodynamic gather
    /// integrates wind over; empty disables the aerodynamic passes.
    pub(crate) triangles: &'a [[u32; 3]],
    /// Steady world-space wind velocity, world units per second.
    pub(crate) wind_velocity: [f32; 3],
    /// Per-triangle turbulence strength; `0` disables the jitter.
    pub(crate) wind_turbulence: f32,
    /// Normal-direction (drag) aerodynamic coefficient.
    pub(crate) aero_drag: f32,
    /// In-plane (lift) aerodynamic coefficient.
    pub(crate) aero_lift: f32,
    /// Analytic body-collision proxies.
    pub(crate) colliders: &'a [GpuClothCollider],
    /// Painted backstop planes, one per constrained particle.
    pub(crate) backstops: &'a [GpuClothBackstop],
    /// Render-vertex embed bindings.
    pub(crate) embed_bindings: &'a [GpuClothEmbedBinding],
    /// Number of render-mesh vertices (sizes the embed output pool).
    pub(crate) render_vertex_count: u32,
    /// Number of self-collision hash cells (`0` disables self-collision).
    pub(crate) hash_cell_count: u32,
    /// Constant external acceleration (gravity), world units per second².
    pub(crate) gravity: [f32; 3],
    /// Full-frame timestep, seconds.
    pub(crate) dt: f32,
    /// Number of XPBD substeps per frame (clamped to at least one).
    pub(crate) substeps: u32,
    /// Constraint-projection iterations per substep (clamped to at least one).
    pub(crate) iterations: u32,
    /// Velocity damping in `[0, 1]`; the retained fraction is `1 - damping`.
    pub(crate) damping: f32,
    /// Strain limit: the maximum fractional stretch a structural edge may keep.
    pub(crate) strain_limit: f32,
    /// Self-collision separation distance (particle pairs closer are pushed
    /// apart).
    pub(crate) self_thickness: f32,
    /// Self-collision uniform grid cell edge, world units.
    pub(crate) self_cell_size: f32,
}

/// The complete device-free solve plan for one cloth piece.
///
/// The `constraints` slice is the shared distance-then-long-range buffer
/// content; `bending` is the independent bending buffer content; `dispatches`
/// is the flat ordered schedule; and the six params structs (the per-substep
/// sim uniform, the body / self / backstop / embed uniforms, and the
/// aerodynamic uniform) are the initial per-pass uniforms. The device stage
/// turns the byte slices into buffers and the dispatch node records
/// `dispatches` in order.
pub(crate) struct ClothSolvePlan {
    /// Packed distance-then-long-range constraint buffer content.
    pub(crate) constraints: Vec<GpuClothConstraint>,
    /// Packed bending buffer content.
    pub(crate) bending: Vec<GpuClothBendingConstraint>,
    /// Flattened `CSR` vertex->triangle offsets (length `particles + 1`) the
    /// aerodynamic gather kernel walks; empty when aerodynamics is disabled.
    pub(crate) csr_offsets: Vec<u32>,
    /// Flattened `CSR` vertex->triangle entries (triangle indices) the gather
    /// reads; empty when aerodynamics is disabled.
    pub(crate) csr_entries: Vec<u32>,
    /// Aerodynamic dispatch uniform (wind, coefficients, full-frame dt, bound).
    pub(crate) aero_params: GpuClothAeroParams,
    /// Resident buffer element counts for this piece.
    pub(crate) counts: BufferCounts,
    /// The flat, ordered dispatch schedule in golden record order.
    pub(crate) dispatches: Vec<PlannedDispatch>,
    /// Initial per-substep solver uniform.
    pub(crate) sim_params: GpuClothSimParams,
    /// Initial body-collision uniform.
    pub(crate) body_params: GpuClothBodyParams,
    /// Initial self-collision uniform.
    pub(crate) self_params: GpuClothSelfParams,
    /// Initial backstop uniform.
    pub(crate) backstop_params: GpuClothBackstopParams,
    /// Initial skin-embed uniform.
    pub(crate) embed_params: GpuClothEmbedParams,
}

/// Builds the complete device-free solve plan for one cloth piece.
///
/// Chains the constraint/bending planning, the `#[repr(C)]` packing and the
/// architecture-layer [`extract`]/[`prepare`] scheduling, then derives the
/// per-pass uniforms. Substeps and iterations are clamped to at least one to
/// match [`extract`]. The optional passes (self-collision, skin-embed and
/// backstop) are enabled only when their inputs are non-empty, so a bare cloth
/// piece schedules just the sim passes and never records fabricated work.
#[must_use]
pub(crate) fn build_solve_plan(input: &ClothSolveInput<'_>) -> ClothSolvePlan {
    let constraint_plan = plan_constraint_upload(input.constraints);
    let bending_plan = color_bending(input.bending);

    let packed_constraints = pack_constraints(&constraint_plan);
    let packed_bending = pack_bending(&bending_plan);

    let particle_count = input.positions.len() as u32;
    let constraint_count = packed_constraints.len() as u32;
    let bending_count = packed_bending.len() as u32;

    let counts = BufferCounts {
        particles: particle_count,
        constraints: constraint_count,
        hash_cells: input.hash_cell_count,
        render_vertices: input.render_vertex_count,
        backstops: input.backstops.len() as u32,
    };

    let substeps = input.substeps.max(1);
    let iterations = input.iterations.max(1);

    // The optional passes only run when their inputs exist; an empty input
    // would otherwise schedule a pass whose bounds check discards every
    // element, wasting a dispatch.
    let self_collision = input.hash_cell_count > 0 && particle_count > 0;
    let embed = input.render_vertex_count > 0 && !input.embed_bindings.is_empty();
    let backstop = !input.backstops.is_empty();
    // The rigid body-collision resolve only projects particles out of colliders
    // when the piece carries at least one; an empty set makes the CPU golden
    // `resolve_body_collisions` a no-op, so the GPU schedule drops the pass to
    // stay in lockstep (and to avoid forcing consumers to compile the collision
    // module for zero work).
    let body = !input.colliders.is_empty() && particle_count > 0;

    // Aerodynamics needs a driving wind (steady or turbulent) *and* a triangle
    // topology to integrate that wind over; with neither there is no force to
    // apply, so both passes stay off and the `CSR` adjacency is never built.
    let has_wind = input.wind_velocity.iter().any(|c| c.abs() > 0.0);
    let has_turbulence = input.wind_turbulence > 0.0;
    let aerodynamics =
        (has_wind || has_turbulence) && !input.triangles.is_empty() && particle_count > 0;

    // The strain limiter mirrors the CPU golden `solve_cloth`, which runs
    // `apply_strain_limit` only when `strain_limit > 0.0`. A non-positive
    // limit must drop the pass so the plan does not clamp every over-stretched
    // edge to rest (a `1 + 0` max-scale) and diverge from the golden.
    let strain = input.strain_limit > 0.0;
    let (csr_offsets, csr_entries) = if aerodynamics {
        // The gather walks one `CSR` row per vertex, so the adjacency is sized
        // to the particle count; disabled aerodynamics keeps both rows empty.
        let adjacency = VertexTriangleAdjacency::build(particle_count as usize, input.triangles);
        (adjacency.offsets().to_vec(), adjacency.entries().to_vec())
    } else {
        (Vec::new(), Vec::new())
    };

    let plan = extract(
        counts,
        constraint_plan.distance_colors.clone(),
        bending_plan.bending_colors.clone(),
        constraint_plan.long_range_colors.clone(),
        substeps,
        iterations,
        self_collision,
        embed,
        backstop,
        aerodynamics,
        strain,
        body,
    );
    let prepared = prepare(&plan);

    let dt_sub = if substeps > 0 {
        input.dt / substeps as f32
    } else {
        input.dt
    };
    let retain = (1.0 - input.damping).clamp(0.0, 1.0);
    let strain_max_scale = 1.0 + input.strain_limit.max(0.0);

    let sim_params = GpuClothSimParams {
        gravity: input.gravity,
        dt_sub,
        retain,
        strain_max_scale,
        particle_count,
        constraint_count,
        bending_count,
        _pad: [0; 3],
    };
    let body_params = GpuClothBodyParams {
        particle_count,
        collider_count: input.colliders.len() as u32,
        _pad: [0; 2],
    };
    let self_params = GpuClothSelfParams {
        particle_count,
        table_size: input.hash_cell_count,
        cell_size: input.self_cell_size,
        thickness: input.self_thickness,
    };
    let backstop_params = GpuClothBackstopParams {
        particle_count,
        _pad: [0; 3],
    };
    let embed_params = GpuClothEmbedParams {
        render_vertex_count: input.render_vertex_count,
        _pad: [0; 3],
    };

    // Sanitize the aerodynamic scalars on the host through the architecture
    // layer's own `WindField::sanitized`/`AeroParams::sanitized` (the single
    // golden source): every wind component is finite-forced (`NaN` -> `0`),
    // turbulence is clamped to `0..=1`, and drag/lift are clamped non-negative.
    // The `WESL` kernels only guard drag/lift with `max(x, 0)`, so the host must
    // supply already-clean wind and turbulence to keep the `CPU`/`GPU` results
    // bit-parallel.
    let wind = WindField::new(
        Vec3::new(
            input.wind_velocity[0],
            input.wind_velocity[1],
            input.wind_velocity[2],
        ),
        input.wind_turbulence,
    )
    .sanitized();
    let aero = AeroParams::new(input.aero_drag, input.aero_lift).sanitized();
    let aero_params = GpuClothAeroParams {
        wind: [wind.velocity.x, wind.velocity.y, wind.velocity.z],
        turbulence: wind.turbulence,
        drag: aero.drag,
        lift: aero.lift,
        dt: input.dt,
        particle_count,
    };

    ClothSolvePlan {
        constraints: packed_constraints,
        bending: packed_bending,
        csr_offsets,
        csr_entries,
        aero_params,
        counts,
        dispatches: prepared.dispatches,
        sim_params,
        body_params,
        self_params,
        backstop_params,
        embed_params,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::cloth::gpu::kernels::ClothKernel;
    use prism_render_architecture::cloth::{Compliance, ConstraintKind};

    /// A two-triangle quad: four particles, structural and one diagonal edge.
    fn quad_input<'a>(
        positions: &'a [[f32; 4]],
        velocities: &'a [[f32; 4]],
        constraints: &'a [Constraint],
    ) -> ClothSolveInput<'a> {
        ClothSolveInput {
            positions,
            velocities,
            constraints,
            bending: &[],
            triangles: &[],
            wind_velocity: [0.0, 0.0, 0.0],
            wind_turbulence: 0.0,
            aero_drag: 0.0,
            aero_lift: 0.0,
            colliders: &[],
            backstops: &[],
            embed_bindings: &[],
            render_vertex_count: 0,
            hash_cell_count: 0,
            gravity: [0.0, -9.81, 0.0],
            dt: 1.0 / 60.0,
            substeps: 4,
            iterations: 2,
            damping: 0.02,
            strain_limit: 0.1,
            self_thickness: 0.01,
            self_cell_size: 0.05,
        }
    }

    /// Builds a rigid structural constraint.
    fn edge(a: u32, b: u32, kind: ConstraintKind) -> Constraint {
        Constraint::new(a, b, 1.0, Compliance::RIGID, kind)
    }

    #[test]
    fn counts_track_the_packed_buffers() {
        let positions = [[0.0; 4], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let velocities = [[0.0; 4]; 3];
        let constraints = [
            edge(0, 1, ConstraintKind::Stretch),
            edge(1, 2, ConstraintKind::Shear),
            edge(0, 2, ConstraintKind::Lra),
        ];
        let input = quad_input(&positions, &velocities, &constraints);
        let plan = build_solve_plan(&input);
        assert_eq!(plan.counts.particles, 3);
        assert_eq!(plan.counts.constraints, plan.constraints.len() as u32);
        assert_eq!(plan.counts.constraints, 3);
        assert_eq!(plan.sim_params.particle_count, 3);
        assert_eq!(plan.sim_params.constraint_count, 3);
    }

    #[test]
    fn sim_params_are_derived_from_scalars() {
        let positions = [[0.0; 4], [1.0, 0.0, 0.0, 0.0]];
        let velocities = [[0.0; 4]; 2];
        let constraints = [edge(0, 1, ConstraintKind::Stretch)];
        let input = quad_input(&positions, &velocities, &constraints);
        let plan = build_solve_plan(&input);
        // dt_sub = dt / substeps
        assert!((plan.sim_params.dt_sub - (1.0 / 60.0) / 4.0).abs() <= 1e-7);
        // retain = 1 - damping
        assert!((plan.sim_params.retain - 0.98).abs() <= 1e-6);
        // strain_max_scale = 1 + strain_limit
        assert!((plan.sim_params.strain_max_scale - 1.1).abs() <= 1e-6);
    }

    #[test]
    fn optional_passes_are_gated_off_when_unused() {
        let positions = [[0.0; 4], [1.0, 0.0, 0.0, 0.0]];
        let velocities = [[0.0; 4]; 2];
        let constraints = [edge(0, 1, ConstraintKind::Stretch)];
        let input = quad_input(&positions, &velocities, &constraints);
        let plan = build_solve_plan(&input);
        // No backstops, no embed, no self-collision cells: those kernels must
        // not appear in the schedule.
        let has = |k: ClothKernel| plan.dispatches.iter().any(|d| d.kernel == k);
        assert!(!has(ClothKernel::Backstop));
        assert!(!has(ClothKernel::SkinEmbed));
        assert!(!has(ClothKernel::SelfCollisionHashBuild));
        assert!(!has(ClothKernel::SelfCollisionResolve));
        // The core sim passes are always present.
        assert!(has(ClothKernel::Predict));
        assert!(has(ClothKernel::ProjectDistanceBatch));
        assert!(has(ClothKernel::VelocityUpdate));
    }

    #[test]
    fn optional_passes_turn_on_with_inputs() {
        let positions = [[0.0; 4], [1.0, 0.0, 0.0, 0.0]];
        let velocities = [[0.0; 4]; 2];
        let constraints = [edge(0, 1, ConstraintKind::Stretch)];
        let backstops = [GpuClothBackstop::default()];
        let embed = [GpuClothEmbedBinding::default()];
        let mut input = quad_input(&positions, &velocities, &constraints);
        input.backstops = &backstops;
        input.embed_bindings = &embed;
        input.render_vertex_count = 1;
        input.hash_cell_count = 8;
        let plan = build_solve_plan(&input);
        let has = |k: ClothKernel| plan.dispatches.iter().any(|d| d.kernel == k);
        assert!(has(ClothKernel::Backstop));
        assert!(has(ClothKernel::SkinEmbed));
        assert!(has(ClothKernel::SelfCollisionHashBuild));
        assert!(has(ClothKernel::SelfCollisionResolve));
        assert_eq!(plan.counts.backstops, 1);
        assert_eq!(plan.counts.render_vertices, 1);
        assert_eq!(plan.counts.hash_cells, 8);
        assert_eq!(plan.self_params.table_size, 8);
    }

    #[test]
    fn substeps_and_iterations_clamp_to_one() {
        let positions = [[0.0; 4], [1.0, 0.0, 0.0, 0.0]];
        let velocities = [[0.0; 4]; 2];
        let constraints = [edge(0, 1, ConstraintKind::Stretch)];
        let mut input = quad_input(&positions, &velocities, &constraints);
        input.substeps = 0;
        input.iterations = 0;
        let plan = build_solve_plan(&input);
        // A single Predict pass (one substep) must be scheduled.
        let predicts = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == ClothKernel::Predict)
            .count();
        assert_eq!(predicts, 1);
        // dt_sub falls back to the full dt when clamped to one substep.
        assert!((plan.sim_params.dt_sub - 1.0 / 60.0).abs() <= 1e-7);
    }

    #[test]
    fn empty_garment_yields_empty_plan() {
        let input = ClothSolveInput {
            positions: &[],
            velocities: &[],
            constraints: &[],
            bending: &[],
            triangles: &[],
            wind_velocity: [0.0, 0.0, 0.0],
            wind_turbulence: 0.0,
            aero_drag: 0.0,
            aero_lift: 0.0,
            colliders: &[],
            backstops: &[],
            embed_bindings: &[],
            render_vertex_count: 0,
            hash_cell_count: 0,
            gravity: [0.0, 0.0, 0.0],
            dt: 0.0,
            substeps: 0,
            iterations: 0,
            damping: 0.0,
            strain_limit: 0.0,
            self_thickness: 0.0,
            self_cell_size: 0.0,
        };
        let plan = build_solve_plan(&input);
        assert!(plan.constraints.is_empty());
        assert!(plan.bending.is_empty());
        assert_eq!(plan.counts.particles, 0);
        // No particles and no constraints: the schedule records no work.
        assert!(plan.dispatches.is_empty());
    }

    #[test]
    fn aerodynamics_turns_on_with_wind_and_triangles() {
        let positions = [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let velocities = [[0.0; 4]; 3];
        let constraints = [edge(0, 1, ConstraintKind::Stretch)];
        let triangles = [[0u32, 1, 2]];
        let mut input = quad_input(&positions, &velocities, &constraints);
        input.triangles = &triangles;
        input.wind_velocity = [3.0, 0.0, 0.0];
        input.aero_drag = 1.0;
        input.aero_lift = 0.5;
        let plan = build_solve_plan(&input);
        // The gather adjacency is built: one `CSR` row per particle plus the
        // trailing sentinel, and one entry per (triangle, vertex) incidence.
        assert_eq!(plan.csr_offsets.len(), positions.len() + 1);
        assert_eq!(plan.csr_entries.len(), triangles.len() * 3);
        // Both aerodynamic passes are scheduled, snapshot before the gather.
        let has = |k: ClothKernel| plan.dispatches.iter().any(|d| d.kernel == k);
        assert!(has(ClothKernel::AerodynamicsSnapshot));
        assert!(has(ClothKernel::Aerodynamics));
        // The uniform carries the sanitized scalars and the full-frame `dt`.
        assert!((plan.aero_params.wind[0] - 3.0).abs() <= 1e-6);
        assert!((plan.aero_params.drag - 1.0).abs() <= 1e-6);
        assert!((plan.aero_params.lift - 0.5).abs() <= 1e-6);
        assert!((plan.aero_params.dt - input.dt).abs() <= 1e-7);
        assert_eq!(plan.aero_params.particle_count, positions.len() as u32);
    }

    #[test]
    fn turbulence_alone_enables_aerodynamics() {
        let positions = [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let velocities = [[0.0; 4]; 3];
        let constraints = [edge(0, 1, ConstraintKind::Stretch)];
        let triangles = [[0u32, 1, 2]];
        let mut input = quad_input(&positions, &velocities, &constraints);
        input.triangles = &triangles;
        // No steady wind, only turbulence — still a driving force.
        input.wind_turbulence = 0.5;
        let plan = build_solve_plan(&input);
        assert!(!plan.csr_offsets.is_empty());
        let has = |k: ClothKernel| plan.dispatches.iter().any(|d| d.kernel == k);
        assert!(has(ClothKernel::Aerodynamics));
        assert!((plan.aero_params.turbulence - 0.5).abs() <= 1e-6);
    }

    #[test]
    fn aerodynamics_stays_off_without_a_driver_or_triangles() {
        let positions = [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let velocities = [[0.0; 4]; 3];
        let constraints = [edge(0, 1, ConstraintKind::Stretch)];
        let triangles = [[0u32, 1, 2]];

        // Triangles present but no wind and no turbulence: nothing to integrate.
        let mut no_wind = quad_input(&positions, &velocities, &constraints);
        no_wind.triangles = &triangles;
        let plan = build_solve_plan(&no_wind);
        assert!(plan.csr_offsets.is_empty());
        assert!(plan.csr_entries.is_empty());
        let has = |k: ClothKernel| plan.dispatches.iter().any(|d| d.kernel == k);
        assert!(!has(ClothKernel::AerodynamicsSnapshot));
        assert!(!has(ClothKernel::Aerodynamics));

        // Wind present but no triangle topology: no faces to gather over.
        let mut no_tris = quad_input(&positions, &velocities, &constraints);
        no_tris.wind_velocity = [5.0, 0.0, 0.0];
        let plan = build_solve_plan(&no_tris);
        assert!(plan.csr_offsets.is_empty());
        assert!(!plan
            .dispatches
            .iter()
            .any(|d| d.kernel == ClothKernel::Aerodynamics));
    }

    #[test]
    fn aero_scalars_are_sanitized_like_the_architecture_layer() {
        let positions = [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let velocities = [[0.0; 4]; 3];
        let constraints = [edge(0, 1, ConstraintKind::Stretch)];
        let triangles = [[0u32, 1, 2]];
        let mut input = quad_input(&positions, &velocities, &constraints);
        input.triangles = &triangles;
        // Poisoned inputs: a NaN wind component, out-of-range turbulence and a
        // negative drag must all be cleaned before reaching the uniform.
        input.wind_velocity = [f32::NAN, 2.0, 0.0];
        input.wind_turbulence = 5.0;
        input.aero_drag = -1.0;
        input.aero_lift = f32::NAN;
        let plan = build_solve_plan(&input);
        assert!((plan.aero_params.wind[0] - 0.0).abs() <= 1e-6);
        assert!((plan.aero_params.wind[1] - 2.0).abs() <= 1e-6);
        assert!((plan.aero_params.turbulence - 1.0).abs() <= 1e-6);
        assert!((plan.aero_params.drag - 0.0).abs() <= 1e-6);
        assert!((plan.aero_params.lift - 0.0).abs() <= 1e-6);
    }
}
