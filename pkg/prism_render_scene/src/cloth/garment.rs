//! The main-world cloth garment component and its per-frame render-world
//! snapshot.
//!
//! A [`ClothGarment`] is the authored `CPU` description a game spawns on an
//! entity: the sim-mesh particles (position with inverse mass in `.w`, plus
//! velocity), the authored constraint set, the dihedral bending hinges, the
//! analytic colliders, the painted backstops, the render-vertex embed bindings
//! and the solver scalars. It is the single main-world source the extract stage
//! snapshots into the render world each frame.
//!
//! [`ExtractedCloth`] is the render-world resource that snapshot lands in. The
//! extract system clears and refills it every frame (mirroring the lighting
//! extract), so the prepare stage always sees the current garments and an
//! entity that despawns simply stops contributing.

use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;

use prism_render_architecture::cloth::bending::BendingConstraint;
use prism_render_architecture::cloth::lod::{cloth_lod_budget, ClothLodDecision};
use prism_render_architecture::cloth::{ClothLodTier, Constraint};

use super::abi::{GpuClothBackstop, GpuClothCollider, GpuClothEmbedBinding};
use super::lod::garment_cloth_piece;
use super::lod_mesh::ClothReducedMesh;
use super::solve_plan::ClothSolveInput;

/// The authored `CPU` state of one cloth garment, spawned on a main-world
/// entity.
///
/// Positions pack the inverse mass into `.w` (`<= 0` pins the particle). The
/// `constraints` list is the authored mixed-kind set; the prepare stage
/// partitions and colors it, so the author never has to pre-sort it. The
/// collider / backstop / embed slices are already in their `#[repr(C)]` device
/// form because they carry no `CPU`-golden solver type to reorder.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct ClothGarment {
    /// Particle positions with inverse mass in `.w`.
    pub(crate) positions: Vec<[f32; 4]>,
    /// Particle velocities; `.w` is unused padding.
    pub(crate) velocities: Vec<[f32; 4]>,
    /// Authored distance and attachment constraints, mixed kinds.
    pub(crate) constraints: Vec<Constraint>,
    /// Authored dihedral bending hinges.
    pub(crate) bending: Vec<BendingConstraint>,
    /// Sim-mesh triangles (three particle indices each) describing the surface
    /// the aerodynamic gather pass integrates wind over. Empty disables the
    /// aerodynamic passes for this garment.
    pub(crate) triangles: Vec<[u32; 3]>,
    /// Steady world-space wind velocity, world units per second.
    pub(crate) wind_velocity: [f32; 3],
    /// Per-triangle turbulence strength; `0` disables the jitter.
    pub(crate) wind_turbulence: f32,
    /// Normal-direction (drag) aerodynamic coefficient.
    pub(crate) aero_drag: f32,
    /// In-plane (lift) aerodynamic coefficient.
    pub(crate) aero_lift: f32,
    /// Fluid (air) density; `0` (the default) keeps the linear aerodynamic
    /// model, a positive value selects the UE5 `Chaos`-style quadratic
    /// (airspeed-squared) drag/lift model.
    pub(crate) aero_air_density: f32,
    /// Cloth-side Coulomb friction coefficient for body collision, sourced from
    /// `FabricMaterial::friction` and clamped to `0..=1` during planning. `0`
    /// (the default) keeps the frictionless projection; higher values grip the
    /// garment against the collider proxies instead of letting it slide.
    pub(crate) friction: f32,
    /// Analytic body-collision proxies.
    pub(crate) colliders: Vec<GpuClothCollider>,
    /// Painted backstop planes, one per constrained particle.
    pub(crate) backstops: Vec<GpuClothBackstop>,
    /// Render-vertex embed bindings driving the skinning pass.
    pub(crate) embed_bindings: Vec<GpuClothEmbedBinding>,
    /// Number of render-mesh vertices (sizes the embed output pool).
    pub(crate) render_vertex_count: u32,
    /// Number of self-collision hash cells (`0` disables self-collision).
    pub(crate) hash_cell_count: u32,
    /// Constant external acceleration (gravity), world units per second².
    pub(crate) gravity: [f32; 3],
    /// Full-frame timestep, seconds.
    pub(crate) dt: f32,
    /// Number of XPBD substeps per frame.
    pub(crate) substeps: u32,
    /// Constraint-projection iterations per substep.
    pub(crate) iterations: u32,
    /// Velocity damping in `[0, 1]`.
    pub(crate) damping: f32,
    /// Strain limit: maximum fractional stretch a structural edge may keep.
    pub(crate) strain_limit: f32,
    /// Self-collision separation distance.
    pub(crate) self_thickness: f32,
    /// Self-collision uniform grid cell edge, world units.
    pub(crate) self_cell_size: f32,

    // -- Level of detail ----------------------------------------------------
    /// The finest representation this garment actually has geometry for. LOD
    /// selection clamps the coverage-chosen tier no finer than this, so a
    /// background outfit authored to only ever skin
    /// ([`ClothLodTier::SkinnedProxy`]) is never promoted to a simulation it
    /// does not own. Defaults to [`ClothLodTier::FullSim`].
    pub(crate) native_form: ClothLodTier,
    /// This frame's projected screen coverage in `0..=1`. A coverage feeding
    /// system updates it per frame; the default `1.0` (fills the screen) keeps
    /// a garment at its finest tier until coverage is actually supplied.
    pub(crate) coverage: f32,
    /// Coverage below which the garment drops from full to reduced simulation.
    /// The default `0.0` (with `lod_skinned_below` also `0.0`) disables LOD:
    /// coverage is always `>= 0`, so the garment stays at full simulation until
    /// an author opts in with real thresholds.
    pub(crate) lod_reduced_sim_below: f32,
    /// Coverage below which the garment collapses to a non-simulated skinned
    /// proxy (no resident GPU piece, no compute pass). Defaults to `0.0`.
    pub(crate) lod_skinned_below: f32,
    /// Stable LOD identity for this piece, surfaced in the resolved
    /// [`ClothLodDecision::handle`] so the renderer can bin pieces by tier.
    /// Defaults to `0`.
    pub(crate) lod_piece_id: u32,
    /// The pre-authored coarser simulation mesh this garment swaps to at the
    /// reduced-simulation LOD tier. `None` (the default) keeps the honest
    /// fallback: the reduced tier re-solves the full mesh because no coarse
    /// geometry was authored. `Some` lands a real per-frame cost reduction —
    /// fewer particles, constraints and dispatched work-items — whenever the
    /// coverage gate selects [`ClothLodTier::ReducedSim`].
    pub(crate) reduced_mesh: Option<ClothReducedMesh>,
    /// Symmetric screen-coverage dead-band applied around each LOD threshold so
    /// a garment hovering on a boundary does not oscillate ("pop") between tiers
    /// frame to frame. The default `0.0` disables hysteresis: the frame-state
    /// tier then tracks the stateless coverage classification exactly, matching
    /// the pre-hysteresis behavior bit for bit.
    pub(crate) lod_hysteresis: f32,
    /// The LOD tier resolved for this garment last frame. The coverage system
    /// advances it through the hysteretic gate each frame (holding it inside the
    /// dead-band), the extract stage snapshots it into the render world, and the
    /// prepare/budget stages solve and charge against it. Seeded at build time
    /// to the stateless coverage tier so the very first frame already matches the
    /// non-hysteretic decision. Defaults to [`ClothLodTier::FullSim`].
    pub(crate) current_tier: ClothLodTier,
}

impl Default for ClothGarment {
    /// An empty garment with LOD disabled: `native_form` is
    /// [`ClothLodTier::FullSim`], coverage is `1.0`, and both LOD thresholds are
    /// `0.0`, so the coverage->tier classification can never trigger a reduction
    /// until an author supplies real thresholds. Every other field is the type
    /// default (empty buffers, zero scalars), matching the pre-LOD behavior.
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            constraints: Vec::new(),
            bending: Vec::new(),
            triangles: Vec::new(),
            wind_velocity: [0.0; 3],
            wind_turbulence: 0.0,
            aero_drag: 0.0,
            aero_lift: 0.0,
            aero_air_density: 0.0,
            friction: 0.0,
            colliders: Vec::new(),
            backstops: Vec::new(),
            embed_bindings: Vec::new(),
            render_vertex_count: 0,
            hash_cell_count: 0,
            gravity: [0.0; 3],
            dt: 0.0,
            substeps: 0,
            iterations: 0,
            damping: 0.0,
            strain_limit: 0.0,
            self_thickness: 0.0,
            self_cell_size: 0.0,
            native_form: ClothLodTier::FullSim,
            coverage: 1.0,
            lod_reduced_sim_below: 0.0,
            lod_skinned_below: 0.0,
            lod_piece_id: 0,
            reduced_mesh: None,
            lod_hysteresis: 0.0,
            current_tier: ClothLodTier::FullSim,
        }
    }
}

impl ClothGarment {
    /// Borrows this garment's fields as a [`ClothSolveInput`] for the prepare
    /// stage, without copying any of the owned buffers.
    #[must_use]
    pub(crate) fn as_solve_input(&self) -> ClothSolveInput<'_> {
        ClothSolveInput {
            positions: &self.positions,
            velocities: &self.velocities,
            constraints: &self.constraints,
            bending: &self.bending,
            triangles: &self.triangles,
            wind_velocity: self.wind_velocity,
            wind_turbulence: self.wind_turbulence,
            aero_drag: self.aero_drag,
            aero_lift: self.aero_lift,
            aero_air_density: self.aero_air_density,
            friction: self.friction,
            colliders: &self.colliders,
            backstops: &self.backstops,
            embed_bindings: &self.embed_bindings,
            render_vertex_count: self.render_vertex_count,
            hash_cell_count: self.hash_cell_count,
            gravity: self.gravity,
            dt: self.dt,
            substeps: self.substeps,
            iterations: self.iterations,
            damping: self.damping,
            strain_limit: self.strain_limit,
            self_thickness: self.self_thickness,
            self_cell_size: self.self_cell_size,
        }
    }

    /// Borrows this garment's fields as a [`ClothSolveInput`] for the prepare
    /// stage at a given resolved LOD tier.
    ///
    /// At [`ClothLodTier::ReducedSim`], when this garment carries a non-empty
    /// pre-authored coarse mesh, the resolution-dependent buffers (particles,
    /// constraints, bending, triangles, backstops, embeds and the self-collision
    /// hash size) come from that coarser mesh, so the solve genuinely runs fewer
    /// particles and constraints — a real per-frame cost reduction, not a budget
    /// annotation. Every material/environment scalar and the analytic colliders
    /// and render-vertex count are shared off this garment because they do not
    /// change with mesh resolution. Any other tier — and the reduced tier with no
    /// authored coarse mesh — borrows the full mesh through [`Self::as_solve_input`],
    /// preserving the honest fallback of re-solving the full mesh.
    #[must_use]
    pub(crate) fn as_solve_input_for_tier(&self, tier: ClothLodTier) -> ClothSolveInput<'_> {
        match (tier, &self.reduced_mesh) {
            (ClothLodTier::ReducedSim, Some(reduced)) if !reduced.positions.is_empty() => {
                self.reduced_solve_input(reduced)
            }
            _ => self.as_solve_input(),
        }
    }

    /// Borrows the reduced-tier solve view: the resolution-dependent buffers from
    /// `reduced` spliced onto this garment's shared material/environment scalars,
    /// analytic colliders and render-vertex count.
    #[must_use]
    fn reduced_solve_input<'a>(&'a self, reduced: &'a ClothReducedMesh) -> ClothSolveInput<'a> {
        ClothSolveInput {
            positions: &reduced.positions,
            velocities: &reduced.velocities,
            constraints: &reduced.constraints,
            bending: &reduced.bending,
            triangles: &reduced.triangles,
            wind_velocity: self.wind_velocity,
            wind_turbulence: self.wind_turbulence,
            aero_drag: self.aero_drag,
            aero_lift: self.aero_lift,
            aero_air_density: self.aero_air_density,
            friction: self.friction,
            colliders: &self.colliders,
            backstops: &reduced.backstops,
            embed_bindings: &reduced.embed_bindings,
            render_vertex_count: self.render_vertex_count,
            hash_cell_count: reduced.hash_cell_count,
            gravity: self.gravity,
            dt: self.dt,
            substeps: self.substeps,
            iterations: self.iterations,
            damping: self.damping,
            strain_limit: self.strain_limit,
            self_thickness: self.self_thickness,
            self_cell_size: self.self_cell_size,
        }
    }

    /// Borrows this garment's world-space particle positions (inverse mass in
    /// `.w`). The coverage estimator reads the `xyz` cloud to bound the garment
    /// on screen without cloning the buffer.
    #[must_use]
    pub(crate) fn positions(&self) -> &[[f32; 4]] {
        &self.positions
    }

    // -- Level-of-detail accessors -----------------------------------------

    /// The number of simulated sim-mesh vertices (one per particle row). This
    /// is the LOD sim-vertex budget and the dominant per-frame solve cost.
    #[must_use]
    pub(crate) fn sim_vertex_count(&self) -> u32 {
        self.positions.len() as u32
    }

    /// The number of authored constraints in this garment's constraint graph.
    #[must_use]
    pub(crate) fn constraint_count(&self) -> u32 {
        self.constraints.len() as u32
    }

    /// The number of embedded render-mesh vertices this garment drives.
    #[must_use]
    pub(crate) fn render_vertex_count(&self) -> u32 {
        self.render_vertex_count
    }

    /// The finest LOD tier this garment has geometry for.
    #[must_use]
    pub(crate) fn native_form(&self) -> ClothLodTier {
        self.native_form
    }

    /// This frame's projected screen coverage in `0..=1`.
    #[must_use]
    pub(crate) fn coverage(&self) -> f32 {
        self.coverage
    }

    /// Records this frame's projected screen coverage, clamped to `0..=1`.
    ///
    /// Written by the coverage estimator ([`update_cloth_coverage`]) each frame
    /// before the extract stage snapshots the garment, so the LOD gate resolves
    /// against a live on-screen size rather than a static authored value.
    ///
    /// [`update_cloth_coverage`]: super::coverage::update_cloth_coverage
    pub(crate) fn set_coverage(&mut self, coverage: f32) {
        self.coverage = coverage.clamp(0.0, 1.0);
    }

    /// Coverage below which the garment drops to reduced simulation.
    #[must_use]
    pub(crate) fn lod_reduced_sim_below(&self) -> f32 {
        self.lod_reduced_sim_below
    }

    /// Coverage below which the garment collapses to a skinned proxy.
    #[must_use]
    pub(crate) fn lod_skinned_below(&self) -> f32 {
        self.lod_skinned_below
    }

    /// This garment's stable LOD identity.
    #[must_use]
    pub(crate) fn lod_piece_id(&self) -> u32 {
        self.lod_piece_id
    }

    /// The symmetric coverage dead-band this garment applies around each LOD
    /// threshold. `0.0` (the default) disables hysteresis.
    #[must_use]
    pub(crate) fn lod_hysteresis(&self) -> f32 {
        self.lod_hysteresis
    }

    /// The LOD tier resolved for this garment as of the last coverage update.
    #[must_use]
    pub(crate) fn current_tier(&self) -> ClothLodTier {
        self.current_tier
    }

    /// Advances this garment's frame-state LOD tier.
    ///
    /// Written by the coverage estimator ([`update_cloth_coverage`]) each frame
    /// through the hysteretic gate, before the extract stage snapshots the
    /// garment into the render world. The prepare and budget stages then solve
    /// and charge against this tier via [`Self::lod_decision`].
    ///
    /// [`update_cloth_coverage`]: super::coverage::update_cloth_coverage
    pub(crate) fn set_current_tier(&mut self, tier: ClothLodTier) {
        self.current_tier = tier;
    }

    /// Resolves this garment's LOD budget at its current frame-state tier.
    ///
    /// The tier itself is chosen by the coverage system through the hysteretic
    /// gate ([`resolve_garment_tier_hysteretic`]) and already clamped no finer
    /// than the garment's native form; this reads that stored [`current_tier`]
    /// and charges its decimated budget through the one authoritative
    /// [`cloth_lod_budget`]. With hysteresis disabled and a build-time seed of
    /// the stateless coverage tier, the decision is bit-identical to the
    /// stateless [`resolve_garment_lod`] gate.
    ///
    /// [`resolve_garment_tier_hysteretic`]: super::lod::resolve_garment_tier_hysteretic
    /// [`current_tier`]: Self::current_tier
    #[must_use]
    pub(crate) fn lod_decision(&self) -> ClothLodDecision {
        cloth_lod_budget(garment_cloth_piece(self), self.current_tier)
    }
}

/// The render-world snapshot of every main-world [`ClothGarment`] this frame.
///
/// Rebuilt each frame by the extract stage; the prepare stage turns each entry
/// into a resident `GPU` piece. Defaults to empty, which makes the whole cloth
/// pass an honest no-op when no garment is spawned.
#[derive(Resource, Default)]
pub(crate) struct ExtractedCloth {
    /// Every extracted garment, in main-world iteration order.
    pub(crate) garments: Vec<ClothGarment>,
    /// The stable main-world [`Entity`] of each garment, kept strictly parallel
    /// to [`Self::garments`] (same length, same order). The prepare stage keys
    /// each garment's resident `GPU` piece by this entity so its device state
    /// persists across frames; a despawned garment drops out of both arrays
    /// together, which is how the prepare stage detects the eviction.
    pub(crate) entities: Vec<Entity>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::cloth::{Compliance, ConstraintKind};

    #[test]
    fn solve_input_borrows_every_field() {
        let garment = ClothGarment {
            positions: vec![[0.0; 4], [1.0, 0.0, 0.0, 0.0]],
            velocities: vec![[0.0; 4]; 2],
            constraints: vec![Constraint::new(
                0,
                1,
                1.0,
                Compliance::RIGID,
                ConstraintKind::Stretch,
            )],
            substeps: 3,
            iterations: 2,
            gravity: [0.0, -9.81, 0.0],
            ..ClothGarment::default()
        };
        let input = garment.as_solve_input();
        assert_eq!(input.positions.len(), 2);
        assert_eq!(input.constraints.len(), 1);
        assert_eq!(input.substeps, 3);
        assert_eq!(input.iterations, 2);
        assert!((input.gravity[1] + 9.81).abs() <= 1e-6);
    }

    #[test]
    fn default_garment_is_empty() {
        let garment = ClothGarment::default();
        assert!(garment.positions.is_empty());
        assert!(garment.constraints.is_empty());
        assert_eq!(garment.render_vertex_count, 0);
    }

    /// A four-particle full mesh with four structural edges, mirrored by a
    /// two-particle coarse mesh with a single edge, so the reduced tier is a
    /// strictly smaller solve.
    fn full_and_reduced_garment(coverage: f32) -> ClothGarment {
        use crate::cloth::{ClothGarmentBuilder, ClothReducedMeshBuilder};
        use prism_render_architecture::cloth::{ClothParticle, Compliance, ConstraintKind, Vec3};

        let full = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(1.0, 1.0, 0.0), 1.0),
        ];
        let full_constraints = vec![
            Constraint::new(0, 1, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
            Constraint::new(1, 3, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
            Constraint::new(3, 2, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
            Constraint::new(2, 0, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
        ];
        let coarse = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 1.0, 0.0), 1.0),
        ];
        let reduced = ClothReducedMeshBuilder::from_particles(&coarse)
            .constraints(vec![Constraint::new(
                0,
                1,
                1.0,
                Compliance::RIGID,
                ConstraintKind::Stretch,
            )])
            .build();
        ClothGarmentBuilder::from_particles(&full)
            .constraints(full_constraints)
            .lod_thresholds(0.5, 0.1)
            .coverage(coverage)
            .reduced_lod_mesh(reduced)
            .build()
    }

    #[test]
    fn reduced_tier_solves_the_coarse_mesh_when_authored() {
        let garment = full_and_reduced_garment(1.0);
        let full = garment.as_solve_input_for_tier(ClothLodTier::FullSim);
        assert_eq!(full.positions.len(), 4);
        assert_eq!(full.constraints.len(), 4);

        let reduced = garment.as_solve_input_for_tier(ClothLodTier::ReducedSim);
        // The reduced tier borrows the coarse mesh: strictly fewer particles and
        // constraints, so the per-frame solve genuinely costs less.
        assert_eq!(reduced.positions.len(), 2);
        assert_eq!(reduced.constraints.len(), 1);
        assert!(reduced.positions.len() < full.positions.len());
        assert!(reduced.constraints.len() < full.constraints.len());
    }

    #[test]
    fn reduced_tier_without_a_coarse_mesh_falls_back_to_the_full_mesh() {
        use crate::cloth::ClothGarmentBuilder;
        use prism_render_architecture::cloth::{ClothParticle, Vec3};

        let particles = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
        ];
        let garment = ClothGarmentBuilder::from_particles(&particles).build();
        // No coarse mesh authored: the reduced tier honestly re-solves the full
        // mesh rather than fabricating a decimation.
        let reduced = garment.as_solve_input_for_tier(ClothLodTier::ReducedSim);
        assert_eq!(reduced.positions.len(), 3);
    }

    #[test]
    fn empty_coarse_mesh_falls_back_to_the_full_mesh() {
        use crate::cloth::{ClothGarmentBuilder, ClothReducedMeshBuilder};
        use prism_render_architecture::cloth::{ClothParticle, Vec3};

        let particles = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
        ];
        let garment = ClothGarmentBuilder::from_particles(&particles)
            .reduced_lod_mesh(ClothReducedMeshBuilder::default().build())
            .build();
        // An attached-but-empty coarse mesh carries no geometry, so the reduced
        // tier still falls back to the full mesh instead of solving nothing.
        let reduced = garment.as_solve_input_for_tier(ClothLodTier::ReducedSim);
        assert_eq!(reduced.positions.len(), 2);
    }

    #[test]
    fn coverage_selected_reduced_tier_picks_the_coarse_solve_input() {
        // End-to-end wiring: at a coverage that the LOD gate classifies as the
        // reduced tier, selecting the solve input by the resolved tier lands on
        // the coarse mesh — the same path the prepare stage takes.
        let garment = full_and_reduced_garment(0.3);
        let decision = garment.lod_decision();
        assert_eq!(decision.tier, ClothLodTier::ReducedSim);
        let input = garment.as_solve_input_for_tier(decision.tier);
        assert_eq!(input.positions.len(), 2);
    }

    #[test]
    fn reduced_tier_produces_a_strictly_smaller_solve_plan() {
        use crate::cloth::solve_plan::build_solve_plan;

        let garment = full_and_reduced_garment(1.0);
        let full_plan = build_solve_plan(&garment.as_solve_input_for_tier(ClothLodTier::FullSim));
        let reduced_plan =
            build_solve_plan(&garment.as_solve_input_for_tier(ClothLodTier::ReducedSim));
        // The reduced tier is not a budget annotation: the actual device plan
        // schedules fewer particles and constraints, so it dispatches less work.
        assert!(reduced_plan.counts.particles < full_plan.counts.particles);
        assert_eq!(reduced_plan.counts.particles, 2);
        assert_eq!(full_plan.counts.particles, 4);
    }

    #[test]
    fn reduced_tier_shares_material_and_environment_scalars() {
        use crate::cloth::{ClothGarmentBuilder, ClothReducedMeshBuilder};
        use prism_render_architecture::cloth::{ClothParticle, Vec3};

        let full = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
        ];
        let coarse = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 1.0, 0.0), 1.0),
        ];
        let garment = ClothGarmentBuilder::from_particles(&full)
            .wind([3.0, 0.0, 0.0], 0.25)
            .aerodynamics(0.8, 0.4)
            .air_density(1.2)
            .gravity([0.0, -9.81, 0.0])
            .solver_iterations(6, 5)
            .reduced_lod_mesh(ClothReducedMeshBuilder::from_particles(&coarse).build())
            .build();
        let reduced = garment.as_solve_input_for_tier(ClothLodTier::ReducedSim);
        // Resolution-independent scalars are shared off the garment, not the mesh.
        assert_eq!(reduced.wind_velocity, [3.0, 0.0, 0.0]);
        assert!((reduced.wind_turbulence - 0.25).abs() <= 1e-6);
        assert!((reduced.aero_drag - 0.8).abs() <= 1e-6);
        assert!((reduced.aero_lift - 0.4).abs() <= 1e-6);
        assert!((reduced.aero_air_density - 1.2).abs() <= 1e-6);
        assert_eq!(reduced.gravity, [0.0, -9.81, 0.0]);
        assert_eq!(reduced.substeps, 6);
        assert_eq!(reduced.iterations, 5);
        // But the mesh itself is the coarse one.
        assert_eq!(reduced.positions.len(), 2);
    }

    #[test]
    fn build_seeds_current_tier_from_the_stateless_coverage_gate() {
        use crate::cloth::ClothGarmentBuilder;
        use prism_render_architecture::cloth::{ClothParticle, Vec3};

        let particles = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(1.0, 1.0, 0.0), 1.0),
        ];
        // Coverage 0.3 with thresholds 0.5/0.1 classifies as reduced sim, so the
        // seed and the decision it drives must both land on the reduced tier.
        let garment = ClothGarmentBuilder::from_particles(&particles)
            .lod_thresholds(0.5, 0.1)
            .coverage(0.3)
            .build();
        assert_eq!(garment.current_tier(), ClothLodTier::ReducedSim);
        assert_eq!(garment.lod_decision().tier, ClothLodTier::ReducedSim);
        assert_eq!(garment.lod_hysteresis(), 0.0);
    }

    #[test]
    fn lod_decision_charges_the_held_frame_state_tier() {
        use crate::cloth::ClothGarmentBuilder;
        use prism_render_architecture::cloth::{ClothParticle, Vec3};

        let particles = [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(1.0, 1.0, 0.0), 1.0),
        ];
        let mut garment = ClothGarmentBuilder::from_particles(&particles)
            .lod_thresholds(0.5, 0.1)
            .coverage(1.0)
            .build();
        assert_eq!(garment.current_tier(), ClothLodTier::FullSim);
        assert_eq!(garment.lod_decision().sim_vertices, 4);

        // The decision reads the held frame-state tier, not the coverage: forcing
        // the tier to reduced decimates the charged budget to a quarter (clamped
        // to at least one) even though coverage still fills the screen.
        garment.set_current_tier(ClothLodTier::ReducedSim);
        let reduced = garment.lod_decision();
        assert_eq!(reduced.tier, ClothLodTier::ReducedSim);
        assert_eq!(reduced.sim_vertices, 1);

        // The skinned proxy drops all sim geometry.
        garment.set_current_tier(ClothLodTier::SkinnedProxy);
        let skinned = garment.lod_decision();
        assert_eq!(skinned.tier, ClothLodTier::SkinnedProxy);
        assert_eq!(skinned.sim_vertices, 0);
    }
}
