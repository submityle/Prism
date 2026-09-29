//! End-to-end cloth pipeline integration.
//!
//! This module wires the sibling cloth modules into one runnable garment:
//! sleep gating ([`super::sleep`]) → aerodynamic wind impulse ([`super::wind`])
//! → XPBD dynamics ([`super::dynamics`]) → pressure/volume projection
//! ([`super::pressure`]) → collision resolution ([`super::collision`]) →
//! render-mesh embedding ([`super::embed`]), built from an asset material and a
//! constraint graph ([`super::constraints`]), gated by the LOD ladder
//! ([`super::lod`]) and metered through the shared deformation budget
//! ([`crate::deformation::schedule`]).
//!
//! It owns no new simulation math; it is the orchestration layer that production
//! engines expose as a "cloth actor" (UE5 `Chaos` Cloth component, `NvCloth`
//! solver instance, Houdini `Vellum` object). A [`Garment`] bundles the sim
//! particles, their colored constraint graph, the body/backstop colliders, the
//! self-collision settings, the solver parameters, and the barycentric bindings
//! that carry a dense render mesh on the coarse sim mesh.
//!
//! ## Collision coupling
//!
//! Body proxies and backstops are one-sided positional projections, so they run
//! **inside** the solver as the per-particle hook of
//! [`super::dynamics::solve_cloth_with_collision`], correcting every substep.
//! Self-collision operates on the whole particle array (a spatial hash), which
//! the per-particle hook cannot express, so it runs **once per frame** after the
//! solve, followed by a final body/backstop pass so a particle pushed by a
//! neighbor never ends a frame inside a collider. Per-frame self-collision is
//! the standard real-time simplification; substep self-collision is a future
//! high-fidelity slot.

use alloc::vec::Vec;

use super::asset::FabricMaterial;
use super::bending::{apply_bending, build_dihedral_bending, BendingConstraint};
use super::ccd::{resolve_ccd, CcdParams};
use super::collision::{
    apply_backstop, resolve_backstops, resolve_body_collisions, resolve_self_collision, Backstop,
    BodyCollider,
};
use super::constraints::{
    build_grid_constraints, color_constraints, ClothGrid, GridConstraintParams,
};
use super::dynamics::{extract_positions, solve_cloth_with_collision, SolverParams};
use super::embed::{embed_render_mesh, BarycentricBinding};
use super::lod::{cloth_deformation_request, resolve_cloth_lod, ClothLodThresholds};
use super::pressure::{apply_pressure, PressureParams};
use super::sleep::{max_kinetic_indicator, should_simulate, SleepParams, SleepState, SleepTracker};
use super::tearing::{apply_plasticity, apply_tearing, PlasticParams, TearingParams};
use super::wind::{apply_aero_forces, AeroParams, WindField};
use super::{ClothLodTier, ClothParticle, ClothPiece, Compliance, ConstraintGraph, Vec3};
use crate::deformation::schedule::DeformationRequest;

/// Self-collision settings for a garment.
///
/// `cell_size` is the spatial-hash cell edge and `thickness` the minimum
/// particle separation the resolver enforces. `enabled` lets a LOD tier or a
/// budget decision switch self-collision off (reduced-sim cloth commonly runs
/// with body collision only), matching the degrade matrix where self-collision
/// is a top-tier-only cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfCollisionParams {
    /// Spatial-hash cell edge length.
    pub cell_size: f32,
    /// Minimum enforced particle separation.
    pub thickness: f32,
    /// Whether self-collision runs this frame.
    pub enabled: bool,
}

impl Default for SelfCollisionParams {
    fn default() -> Self {
        Self {
            cell_size: 0.1,
            thickness: 0.05,
            enabled: false,
        }
    }
}

impl SelfCollisionParams {
    /// Builds enabled self-collision settings.
    #[must_use]
    pub fn new(cell_size: f32, thickness: f32) -> Self {
        Self {
            cell_size,
            thickness,
            enabled: true,
        }
    }
}

/// A simulatable garment: everything one cloth actor needs for a frame step.
///
/// The render mesh is not stored here; [`Garment::render_positions`] evaluates
/// it on demand from the current sim positions and the barycentric
/// [`bindings`](Garment::bindings), so the caller controls where the dense
/// vertices land (a GPU upload buffer, a BLAS refit staging array).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Garment {
    /// Sim-mesh particles (position, velocity, inverse mass).
    pub particles: Vec<ClothParticle>,
    /// Colored constraint graph solved every substep.
    pub graph: ConstraintGraph,
    /// Barycentric bindings carrying render vertices on the sim mesh.
    pub bindings: Vec<BarycentricBinding>,
    /// Body proxies (sphere/capsule/half-space) applied every substep.
    pub colliders: Vec<BodyCollider>,
    /// Backstop planes applied every substep.
    pub backstops: Vec<Backstop>,
    /// XPBD solver parameters.
    pub solver: SolverParams,
    /// Self-collision settings applied once per frame after the solve.
    pub self_collision: SelfCollisionParams,
    /// Sim-mesh triangulation driving the wind and pressure passes; empty
    /// disables both (each iterates faces).
    pub triangles: Vec<[u32; 3]>,
    /// Steady wind field applied as a pre-solve aerodynamic impulse; a calm
    /// default (zero velocity) adds no force.
    pub wind: WindField,
    /// Aerodynamic drag/lift coefficients paired with [`Garment::wind`]; a zero
    /// default adds no force.
    pub aero: AeroParams,
    /// Optional pressure (volume) target; [`None`] disables the pressure pass so
    /// only inflatable garments (down jackets, balloons) pay for it.
    pub pressure: Option<PressureParams>,
    /// Sleep/activation state machine, advanced each simulated frame when
    /// [`Garment::sleep_enabled`] is set.
    pub sleep: SleepTracker,
    /// Hysteresis parameters for the sleep gate.
    pub sleep_params: SleepParams,
    /// Whether the sleep gate runs; when `false` the garment always simulates.
    pub sleep_enabled: bool,
    /// Dihedral bending hinges projected each substep after the distance solve;
    /// empty disables bending. Built from the triangulation by
    /// [`Garment::build_bending`] or supplied directly with
    /// [`Garment::set_bending`].
    pub bending: Vec<BendingConstraint>,
    /// Continuous-collision sweep parameters applied once per frame against the
    /// body colliders using the frame-start positions, so a fast particle cannot
    /// tunnel through a thin collider between substeps.
    pub ccd: CcdParams,
    /// Whether the continuous-collision sweep runs; `false` by default so the
    /// per-substep projection alone governs collision unless CCD is requested.
    pub ccd_enabled: bool,
    /// Frame-start position snapshot reused as the CCD sweep origin; kept as a
    /// field to avoid a per-frame allocation.
    prev_positions: Vec<Vec3>,
}

/// Projects one position out of every body collider, then behind every
/// backstop, in a fixed order. This is the per-particle collision hook shared
/// by the substep solve; running colliders before backstops means a backstop
/// (the "do not pass through the body" plane) always has the final say.
fn project_colliders(pos: Vec3, colliders: &[BodyCollider], backstops: &[Backstop]) -> Vec3 {
    let mut out = pos;
    for collider in colliders {
        out = collider.project(out);
    }
    for backstop in backstops {
        out = apply_backstop(out, *backstop);
    }
    out
}

impl Garment {
    /// Builds a garment from prebuilt parts.
    #[must_use]
    pub fn new(
        particles: Vec<ClothParticle>,
        graph: ConstraintGraph,
        solver: SolverParams,
    ) -> Self {
        Self {
            particles,
            graph,
            bindings: Vec::new(),
            colliders: Vec::new(),
            backstops: Vec::new(),
            solver,
            self_collision: SelfCollisionParams::default(),
            triangles: Vec::new(),
            wind: WindField::default(),
            aero: AeroParams::default(),
            pressure: None,
            sleep: SleepTracker::new(),
            sleep_params: SleepParams::default(),
            sleep_enabled: false,
            bending: Vec::new(),
            ccd: CcdParams::default(),
            ccd_enabled: false,
            prev_positions: Vec::new(),
        }
    }

    /// Number of sim particles.
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Pins the particle at `index` (sets it immovable) if in range; a
    /// convenience for attaching a garment's waistband or shoulders to a body.
    /// Out-of-range indices are ignored so authoring mistakes never panic.
    pub fn pin(&mut self, index: usize) {
        if let Some(particle) = self.particles.get_mut(index) {
            particle.velocity = Vec3::ZERO;
            particle.inverse_mass = 0.0;
        }
    }

    /// Advances the garment by `dt` seconds: substep XPBD solve with body and
    /// backstop collision as the per-substep hook, then one self-collision pass
    /// and a final collider pass. A non-positive `dt` or an empty particle set
    /// is a no-op.
    pub fn step(&mut self, dt: f32) {
        if dt <= 0.0 || self.particles.is_empty() {
            return;
        }
        // Sleep gating: a rested garment skips the whole solve and stays out of
        // the deformation budget until its motion crosses the wake threshold.
        if self.sleep_enabled {
            let indicator = max_kinetic_indicator(&self.particles);
            let state = self.sleep.update(indicator, self.sleep_params);
            if !should_simulate(state) {
                return;
            }
        }
        // Snapshot frame-start positions so the continuous-collision sweep can
        // reconstruct each particle's swept segment after the solve. Done before
        // any force or projection touches the positions.
        if self.ccd_enabled {
            self.prev_positions.clear();
            self.prev_positions
                .extend(self.particles.iter().map(|p| p.position));
        }
        // Aerodynamic wind is a pre-solve velocity impulse, so the substep
        // prediction integrates it. A calm field (the default) adds nothing and
        // an empty triangulation makes the pass a no-op.
        apply_aero_forces(
            &mut self.particles,
            &self.triangles,
            &self.wind,
            self.aero,
            dt,
        );
        let colliders = &self.colliders;
        let backstops = &self.backstops;
        solve_cloth_with_collision(&mut self.particles, &self.graph, self.solver, dt, |p| {
            project_colliders(p.position, colliders, backstops)
        });
        // Dihedral bending is projected after the distance solve as a lower-
        // frequency relaxation pass. The substep timestep keeps the XPBD
        // compliance scaling consistent with the distance solve; an empty hinge
        // set makes this a no-op.
        if !self.bending.is_empty() {
            let substeps = self.solver.substeps.max(1);
            let dt_sub = dt / substeps as f32;
            // Give bending authority comparable to the distance solve by running
            // it for the same total iteration budget (substeps * iterations),
            // since it is a single decoupled pass rather than interleaved per
            // substep.
            let passes = substeps * self.solver.iterations.max(1);
            apply_bending(&mut self.particles, &self.bending, passes, dt_sub);
        }
        // Pressure (volume) is a per-frame positional projection over the closed
        // sim mesh, applied after the distance solve like self-collision. It is
        // opt-in (only inflatable garments carry a target volume); per-frame is
        // the real-time simplification, per-substep is a future high-fidelity
        // slot.
        if let Some(pressure) = self.pressure {
            apply_pressure(&mut self.particles, &self.triangles, pressure, dt);
        }
        if self.self_collision.enabled {
            resolve_self_collision(
                &mut self.particles,
                self.self_collision.cell_size,
                self.self_collision.thickness,
            );
        }
        // Self-collision or pressure can push a particle back into a body;
        // re-project so a frame never ends inside a collider.
        resolve_body_collisions(&mut self.particles, &self.colliders);
        resolve_backstops(&mut self.particles, &self.backstops);
        // Continuous collision: sweep frame-start -> current against the body
        // colliders and snap any tunnelling particle back to the surface. Runs
        // last so it corrects the fully resolved end-of-frame positions.
        if self.ccd_enabled {
            resolve_ccd(
                &mut self.particles,
                &self.prev_positions,
                &self.colliders,
                self.ccd,
                dt,
            );
        }
    }

    /// Advances the garment only when `tier` simulates. A
    /// [`ClothLodTier::SkinnedProxy`] tier skips the solver entirely (the
    /// skinned shell path owns those garments), so distant cloth costs nothing.
    pub fn step_at_tier(&mut self, dt: f32, tier: ClothLodTier) {
        if tier.is_simulated() {
            self.step(dt);
        }
    }

    /// Evaluates the render mesh from the current sim positions into `out`
    /// (cleared first), following the coarse sim mesh through the stored
    /// barycentric bindings. Index `i` of `out` is render vertex `i`.
    pub fn render_positions(&self, out: &mut Vec<Vec3>) {
        let sim = extract_positions(&self.particles);
        embed_render_mesh(&self.bindings, &sim, out);
    }

    /// Builds the deformation-budget request for this garment at a screen
    /// `coverage`, or [`None`] when the resolved LOD tier does not simulate.
    ///
    /// The garment does not own the budget; it emits a request and lets
    /// [`crate::deformation::schedule::plan_deformations`] arbitrate it against
    /// every other deforming subsystem this frame.
    #[must_use]
    pub fn deformation_request(
        &self,
        piece: ClothPiece,
        coverage: f32,
        thresholds: ClothLodThresholds,
        priority: u32,
    ) -> Option<DeformationRequest> {
        // A sleeping garment reserves no budget (design §8), so it emits no
        // request even when its coverage would otherwise simulate.
        if self.sleep_enabled && self.sleep.state.is_sleeping() {
            return None;
        }
        let decision = resolve_cloth_lod(piece, coverage, thresholds);
        cloth_deformation_request(piece, decision, priority)
    }

    /// Sets the closed sim-mesh triangulation used by the wind and pressure
    /// passes. Faces should wind counter-clockwise seen from outside so the
    /// pressure volume reads positive; out-of-range indices are skipped by the
    /// passes and never panic.
    pub fn set_triangles(&mut self, triangles: Vec<[u32; 3]>) {
        self.triangles = triangles;
    }

    /// Replaces the dihedral bending hinge set directly.
    pub fn set_bending(&mut self, bending: Vec<BendingConstraint>) {
        self.bending = bending;
    }

    /// Builds dihedral bending hinges from the current triangulation and rest
    /// pose (the current particle positions are the flat-rest reference) at the
    /// given bending `compliance`, and installs them. Requires a triangulation
    /// ([`Garment::set_triangles`]); with none set this clears the hinge set.
    pub fn build_bending(&mut self, compliance: Compliance) {
        let positions = extract_positions(&self.particles);
        self.bending = build_dihedral_bending(&positions, &self.triangles, compliance);
    }

    /// Enables the continuous-collision sweep with the given parameters. The
    /// sweep runs once per frame after the solve, against the body colliders,
    /// preventing fast particles from tunnelling through thin proxies.
    pub fn enable_ccd(&mut self, params: CcdParams) {
        self.ccd = params;
        self.ccd_enabled = true;
    }

    /// Disables the continuous-collision sweep.
    pub fn disable_ccd(&mut self) {
        self.ccd_enabled = false;
    }

    /// Tears every over-stretched two-sided fabric edge (tensile strain above
    /// `params.break_strain`) from the constraint graph and re-colors the
    /// remaining edges so the next step stays parallel-safe. Returns the number
    /// of edges torn; when none tear the coloring is left untouched.
    pub fn tear(&mut self, params: TearingParams) -> usize {
        let torn = apply_tearing(&mut self.graph.constraints, &self.particles, params);
        if torn > 0 {
            let recolored = color_constraints(&self.graph.constraints);
            self.graph = recolored;
        }
        torn
    }

    /// Applies plastic rest-length creep to edges stretched past
    /// `params.yield_strain`, capturing permanent wrinkles and sag. Topology is
    /// unchanged, so no re-coloring is needed.
    pub fn relax_plastic(&mut self, params: PlasticParams) {
        apply_plasticity(&mut self.graph.constraints, &self.particles, params);
    }

    /// Configures the steady wind field and aerodynamic coefficients. Requires a
    /// triangulation ([`Garment::set_triangles`]) to have any visible effect.
    pub fn set_wind(&mut self, wind: WindField, aero: AeroParams) {
        self.wind = wind;
        self.aero = aero;
    }

    /// Enables pressure (volume preservation / inflation) with the given target
    /// parameters. Requires a triangulation to have any effect.
    pub fn set_pressure(&mut self, params: PressureParams) {
        self.pressure = Some(params);
    }

    /// Disables the pressure pass.
    pub fn clear_pressure(&mut self) {
        self.pressure = None;
    }

    /// Turns on the sleep gate with the given hysteresis parameters, starting
    /// awake. A gated garment that comes to rest stops solving and stops
    /// charging the deformation budget until it is disturbed.
    pub fn enable_sleep(&mut self, params: SleepParams) {
        self.sleep_enabled = true;
        self.sleep_params = params;
        self.sleep = SleepTracker::new();
    }

    /// Turns the sleep gate off; the garment then always simulates.
    pub fn disable_sleep(&mut self) {
        self.sleep_enabled = false;
    }

    /// Forces the garment awake (after a teleport or an external hit) so the
    /// next [`Garment::step`] simulates and the dwell timer restarts.
    pub fn wake(&mut self) {
        self.sleep.wake_on_disturbance();
    }

    /// The current sleep state; [`SleepState::Awake`] whenever the gate is off.
    #[must_use]
    pub fn sleep_state(&self) -> SleepState {
        self.sleep.state
    }
}

/// Builds a woven-grid garment from a draped rest pose.
///
/// `positions` are the authored (draped) particle positions in row-major grid
/// order; they become the constraint rest lengths, so the input pose is the
/// equilibrium the solver relaxes toward. Every particle starts free with unit
/// inverse mass and zero velocity; the caller pins attachment points afterward
/// with [`Garment::pin`]. Constraints are built by [`build_grid_constraints`]
/// and graph-colored by [`color_constraints`], so the returned garment is
/// immediately steppable. The material supplies warp/weft/bend compliance;
/// shear reuses the weft compliance.
///
/// When `positions` is shorter than the grid demands, the missing particles are
/// filled at the origin and their edges are skipped by the constraint builder,
/// so a truncated input never panics.
#[must_use]
pub fn build_grid_garment(
    grid: ClothGrid,
    positions: &[Vec3],
    material: &FabricMaterial,
    solver: SolverParams,
) -> Garment {
    let count = grid.particle_count() as usize;
    let mut particles: Vec<ClothParticle> = Vec::with_capacity(count);
    for i in 0..count {
        let pos = positions.get(i).copied().unwrap_or(Vec3::ZERO);
        particles.push(ClothParticle::new(pos, 1.0));
    }
    let params = GridConstraintParams {
        warp: material.warp_compliance(),
        weft: material.weft_compliance(),
        shear: material.weft_compliance(),
        bend: material.bend_compliance(),
    };
    let constraints = build_grid_constraints(grid, positions, params);
    let graph = color_constraints(&constraints);
    Garment::new(particles, graph, solver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::asset::FabricMaterial;
    use crate::cloth::embed::bind_render_vertex;
    use crate::cloth::lod::ClothLodThresholds;
    use crate::cloth::pressure::mesh_volume;
    use crate::cloth::{ClothPieceHandle, Compliance};
    use crate::deformation::DeformationHandle;

    fn drape_grid(rows: u32, cols: u32, spacing: f32) -> (ClothGrid, Vec<Vec3>) {
        let grid = ClothGrid::new(rows, cols);
        let mut positions: Vec<Vec3> = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                positions.push(Vec3::new(c as f32 * spacing, 0.0, r as f32 * spacing));
            }
        }
        (grid, positions)
    }

    fn stiff_solver() -> SolverParams {
        SolverParams {
            substeps: 8,
            iterations: 4,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.02,
            strain_limit: 0.1,
        }
    }

    fn max_abs_component(v: Vec3) -> f32 {
        v.x.abs().max(v.y.abs()).max(v.z.abs())
    }

    fn assert_finite(particles: &[ClothParticle]) {
        for p in particles {
            assert!(
                p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
                "position not finite: {:?}",
                p.position
            );
        }
    }

    #[test]
    fn pinned_top_row_holds_while_cloth_falls() {
        let (grid, positions) = drape_grid(4, 4, 0.25);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        // Pin the first row.
        for c in 0..grid.cols as usize {
            garment.pin(c);
        }
        let top_before = garment.particles[0].position;
        for _ in 0..30 {
            garment.step(1.0 / 60.0);
        }
        // Pinned corner never moves.
        assert!(garment.particles[0].position.distance(top_before) < 1.0e-6);
        // A bottom particle has fallen (its y is now negative).
        let bottom = garment.particles[grid.index(3, 0) as usize].position;
        assert!(bottom.y < -0.01, "bottom y = {}", bottom.y);
        assert_finite(&garment.particles);
    }

    #[test]
    fn stretch_limiting_bounds_hanging_edge_length() {
        let (grid, positions) = drape_grid(6, 2, 0.2);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        garment.pin(0);
        garment.pin(1);
        for _ in 0..120 {
            garment.step(1.0 / 60.0);
        }
        // No structural edge stretches far beyond rest * (1 + strain_limit).
        let rest = 0.2;
        let limit = rest * (1.0 + garment.solver.strain_limit) + 0.05;
        for r in 0..5u32 {
            let a = grid.index(r, 0) as usize;
            let b = grid.index(r + 1, 0) as usize;
            let d = garment.particles[a]
                .position
                .distance(garment.particles[b].position);
            assert!(d < limit, "edge {r} stretched to {d}, limit {limit}");
        }
        assert_finite(&garment.particles);
    }

    #[test]
    fn sphere_collider_pushes_cloth_out() {
        let (grid, positions) = drape_grid(3, 3, 0.3);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        // Pin the top row so the sheet drapes over a sphere below it.
        for c in 0..grid.cols as usize {
            garment.pin(c);
        }
        garment.colliders.push(BodyCollider::Sphere {
            center: Vec3::new(0.3, -0.4, 0.3),
            radius: 0.35,
        });
        for _ in 0..90 {
            garment.step(1.0 / 60.0);
        }
        // Every non-pinned particle sits on or outside the sphere.
        for p in &garment.particles {
            if p.is_pinned() {
                continue;
            }
            let d = p.position.distance(Vec3::new(0.3, -0.4, 0.3));
            assert!(d >= 0.35 - 1.0e-3, "particle inside sphere, d = {d}");
        }
        assert_finite(&garment.particles);
    }

    #[test]
    fn half_space_backstop_keeps_cloth_above_floor() {
        let (grid, positions) = drape_grid(4, 4, 0.25);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        garment.pin(0);
        // Floor at y = -0.5, normal up; backstop distance 0 keeps particles at
        // or above the plane.
        garment.backstops.push(Backstop {
            origin: Vec3::new(0.0, -0.5, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            distance: 0.0,
        });
        for _ in 0..120 {
            garment.step(1.0 / 60.0);
        }
        for p in &garment.particles {
            assert!(
                p.position.y >= -0.5 - 1.0e-3,
                "cloth fell through floor: {}",
                p.position.y
            );
        }
        assert_finite(&garment.particles);
    }

    #[test]
    fn self_collision_separates_stacked_layers() {
        // Two free particles at the same spot must be pushed apart to thickness.
        let particles = alloc::vec![
            ClothParticle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
            ClothParticle::new(Vec3::new(0.001, 0.0, 0.0), 1.0),
        ];
        let mut garment = Garment::new(particles, ConstraintGraph::default(), stiff_solver());
        garment.self_collision = SelfCollisionParams::new(0.1, 0.05);
        garment.step(1.0 / 60.0);
        let sep = garment.particles[0]
            .position
            .distance(garment.particles[1].position);
        assert!(sep >= 0.05 - 1.0e-3, "layers not separated: {sep}");
        assert_finite(&garment.particles);
    }

    #[test]
    fn render_mesh_follows_sim_mesh() {
        let (grid, positions) = drape_grid(2, 2, 1.0);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        // Bind one render vertex at the centroid of triangle (0,1,2).
        let sim = extract_positions(&garment.particles);
        let centroid = sim[0].add(sim[1]).add(sim[2]).scale(1.0 / 3.0);
        let binding = bind_render_vertex(centroid, [0, 1, 2], &sim).expect("in range");
        garment.bindings.push(binding);
        let mut render = Vec::new();
        garment.render_positions(&mut render);
        assert_eq!(render.len(), 1);
        assert!(render[0].distance(centroid) < 1.0e-5);
        // Translate every particle and confirm the render vertex tracks it.
        for p in &mut garment.particles {
            p.position = p.position.add(Vec3::new(1.0, 2.0, 3.0));
        }
        garment.render_positions(&mut render);
        assert!(render[0].distance(centroid.add(Vec3::new(1.0, 2.0, 3.0))) < 1.0e-5);
    }

    #[test]
    fn skinned_proxy_tier_skips_simulation() {
        let (grid, positions) = drape_grid(3, 3, 0.3);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        let before = extract_positions(&garment.particles);
        garment.step_at_tier(1.0 / 60.0, ClothLodTier::SkinnedProxy);
        let after = extract_positions(&garment.particles);
        for (a, b) in before.iter().zip(after.iter()) {
            assert!(a.distance(*b) < 1.0e-9);
        }
        // A simulated tier does move the cloth.
        garment.step_at_tier(1.0 / 60.0, ClothLodTier::FullSim);
        let moved = extract_positions(&garment.particles);
        let mut any_moved = false;
        for (a, b) in before.iter().zip(moved.iter()) {
            if a.distance(*b) > 1.0e-6 {
                any_moved = true;
            }
        }
        assert!(any_moved, "full-sim tier did not move any particle");
    }

    #[test]
    fn deformation_request_gates_on_lod() {
        let (grid, _positions) = drape_grid(4, 4, 0.25);
        let piece = ClothPiece {
            handle: ClothPieceHandle(7),
            sim_vertex_count: grid.particle_count(),
            render_vertex_count: 64,
            constraint_count: 40,
            deformation: DeformationHandle(3),
            native_form: ClothLodTier::FullSim,
        };
        let garment = Garment::new(Vec::new(), ConstraintGraph::default(), stiff_solver());
        let thresholds = ClothLodThresholds {
            reduced_sim_below: 0.25,
            skinned_below: 0.05,
        };
        // High coverage → full sim → a request charging the full vertex count.
        let full = garment
            .deformation_request(piece, 0.9, thresholds, 100)
            .expect("full sim requests deformation");
        assert_eq!(full.vertex_count, grid.particle_count());
        assert!(full.needs_blas_refit);
        // Tiny coverage → skinned proxy → no request.
        assert!(garment
            .deformation_request(piece, 0.001, thresholds, 100)
            .is_none());
    }

    #[test]
    fn stepping_is_deterministic() {
        let (grid, positions) = drape_grid(4, 4, 0.25);
        let material = FabricMaterial::default();
        let mut a = build_grid_garment(grid, &positions, &material, stiff_solver());
        let mut b = build_grid_garment(grid, &positions, &material, stiff_solver());
        a.pin(0);
        b.pin(0);
        a.colliders.push(BodyCollider::Sphere {
            center: Vec3::new(0.4, -0.5, 0.4),
            radius: 0.3,
        });
        b.colliders.push(BodyCollider::Sphere {
            center: Vec3::new(0.4, -0.5, 0.4),
            radius: 0.3,
        });
        for _ in 0..40 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
        }
        for (pa, pb) in a.particles.iter().zip(b.particles.iter()) {
            assert!(pa.position.distance(pb.position) < 1.0e-9);
        }
    }

    #[test]
    fn zero_dt_and_empty_garment_are_no_ops() {
        let (grid, positions) = drape_grid(2, 2, 0.5);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        let before = extract_positions(&garment.particles);
        garment.step(0.0);
        let after = extract_positions(&garment.particles);
        for (x, y) in before.iter().zip(after.iter()) {
            assert!(max_abs_component(x.sub(*y)) < 1.0e-12);
        }
        let mut empty = Garment::new(Vec::new(), ConstraintGraph::default(), stiff_solver());
        empty.step(1.0 / 60.0);
        assert!(empty.particles.is_empty());
    }

    #[test]
    fn long_run_stays_finite() {
        let (grid, positions) = drape_grid(5, 5, 0.2);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        garment.self_collision = SelfCollisionParams::new(0.1, 0.03);
        for c in 0..grid.cols as usize {
            garment.pin(c);
        }
        garment.colliders.push(BodyCollider::Capsule {
            p0: Vec3::new(0.0, -0.6, 0.4),
            p1: Vec3::new(0.8, -0.6, 0.4),
            radius: 0.2,
        });
        for _ in 0..240 {
            garment.step(1.0 / 60.0);
        }
        assert_finite(&garment.particles);
    }

    /// Builds the two-triangles-per-cell triangulation of a row-major grid.
    fn grid_triangles(grid: ClothGrid) -> Vec<[u32; 3]> {
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for r in 0..grid.rows.saturating_sub(1) {
            for c in 0..grid.cols.saturating_sub(1) {
                let a = grid.index(r, c);
                let b = grid.index(r, c + 1);
                let d = grid.index(r + 1, c);
                let e = grid.index(r + 1, c + 1);
                tris.push([a, b, d]);
                tris.push([b, e, d]);
            }
        }
        tris
    }

    #[test]
    fn wind_pushes_cloth_downwind() {
        let (grid, positions) = drape_grid(4, 4, 0.25);
        let material = FabricMaterial::default();
        let mut calm = build_grid_garment(grid, &positions, &material, stiff_solver());
        let mut windy = build_grid_garment(grid, &positions, &material, stiff_solver());
        // Pin the top row on both so only the free rows can drift.
        for c in 0..grid.cols as usize {
            calm.pin(c);
            windy.pin(c);
        }
        windy.set_triangles(grid_triangles(grid));
        // Equal drag and lift make the aerodynamic force track the relative wind
        // itself, so a strong +z wind pushes the sheet downwind regardless of
        // the exact draped orientation.
        windy.set_wind(
            WindField::new(Vec3::new(0.0, 0.0, 20.0), 0.0),
            AeroParams::new(3.0, 3.0),
        );
        for _ in 0..60 {
            calm.step(1.0 / 60.0);
            windy.step(1.0 / 60.0);
        }
        // A free bottom particle is displaced well past the calm (gravity-only)
        // run once the wind is doing work on it.
        let idx = grid.index(3, 3) as usize;
        let drift = windy.particles[idx]
            .position
            .distance(calm.particles[idx].position);
        assert!(drift > 0.05, "wind drift too small: {drift}");
        assert!(
            windy.particles[idx].position.z > calm.particles[idx].position.z + 0.01,
            "windy z {} vs calm z {}",
            windy.particles[idx].position.z,
            calm.particles[idx].position.z
        );
        assert_finite(&windy.particles);
    }

    #[test]
    fn calm_wind_and_empty_triangulation_change_nothing() {
        let (grid, positions) = drape_grid(3, 3, 0.3);
        let material = FabricMaterial::default();
        let mut a = build_grid_garment(grid, &positions, &material, stiff_solver());
        let mut b = build_grid_garment(grid, &positions, &material, stiff_solver());
        a.pin(0);
        b.pin(0);
        // b carries a wind field but no triangulation, so the pass is inert.
        b.set_wind(
            WindField::new(Vec3::new(5.0, 0.0, 0.0), 0.5),
            AeroParams::new(1.0, 1.0),
        );
        for _ in 0..30 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
        }
        for (pa, pb) in a.particles.iter().zip(b.particles.iter()) {
            assert!(pa.position.distance(pb.position) < 1.0e-9);
        }
    }

    #[test]
    fn pressure_inflates_a_closed_mesh() {
        // A unit cube shell with outward-wound faces (rest volume 1).
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
        ];
        let triangles: Vec<[u32; 3]> = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 6, 2],
            [3, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let mut particles: Vec<ClothParticle> = Vec::new();
        for pos in &positions {
            particles.push(ClothParticle::new(*pos, 1.0));
        }
        // Zero gravity so the only motion is the pressure projection.
        let solver = SolverParams {
            substeps: 1,
            iterations: 1,
            gravity: Vec3::ZERO,
            damping: 0.0,
            strain_limit: 0.1,
        };
        let mut garment = Garment::new(particles, ConstraintGraph::default(), solver);
        garment.set_triangles(triangles.clone());
        let start = mesh_volume(&extract_positions(&garment.particles), &triangles);
        // Target twice the rest volume: the shell should inflate toward it.
        garment.set_pressure(PressureParams::new(start, 2.0, Compliance::RIGID));
        for _ in 0..30 {
            garment.step(1.0 / 60.0);
        }
        let end = mesh_volume(&extract_positions(&garment.particles), &triangles);
        assert!(end > start + 0.1, "start {start} end {end}");
        assert_finite(&garment.particles);
    }

    #[test]
    fn sleep_gate_halts_rested_cloth_and_wakes_on_disturbance() {
        let (grid, positions) = drape_grid(3, 3, 0.3);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        // Pin every particle so the garment has no simulated motion at all.
        for i in 0..grid.particle_count() as usize {
            garment.pin(i);
        }
        let params = SleepParams {
            linear_threshold: 1.0e-4,
            frames_to_sleep: 3,
            wake_threshold: 1.0e-2,
        };
        garment.enable_sleep(params);
        assert_eq!(garment.sleep_state(), SleepState::Awake);
        // After the dwell of quiet frames the garment sleeps.
        for _ in 0..5 {
            garment.step(1.0 / 60.0);
        }
        assert_eq!(garment.sleep_state(), SleepState::Sleeping);
        // A sleeping garment reserves no deformation budget.
        let piece = ClothPiece {
            handle: ClothPieceHandle(1),
            sim_vertex_count: grid.particle_count(),
            render_vertex_count: 9,
            constraint_count: 10,
            deformation: DeformationHandle(0),
            native_form: ClothLodTier::FullSim,
        };
        let thresholds = ClothLodThresholds {
            reduced_sim_below: 0.25,
            skinned_below: 0.05,
        };
        assert!(garment
            .deformation_request(piece, 0.9, thresholds, 10)
            .is_none());
        // Waking re-activates it and restores its budget request.
        garment.wake();
        assert_eq!(garment.sleep_state(), SleepState::Awake);
        assert!(garment
            .deformation_request(piece, 0.9, thresholds, 10)
            .is_some());
    }

    #[test]
    fn bending_hinges_build_and_step_stays_finite() {
        let (grid, positions) = drape_grid(5, 5, 0.2);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        garment.set_triangles(grid_triangles(grid));
        garment.build_bending(Compliance(0.0));
        // Interior edges of a 5x5 grid produce a non-empty hinge set.
        assert!(!garment.bending.is_empty());
        for c in 0..grid.cols as usize {
            garment.pin(c);
        }
        for _ in 0..40 {
            garment.step(1.0 / 60.0);
        }
        assert_finite(&garment.particles);
    }

    #[test]
    fn bending_resists_out_of_plane_fold() {
        // A horizontal cantilever pinned along its first row folds down under
        // gravity. Bending resists *curvature*, so the stiff sheet stays far
        // flatter than an identical sheet with no hinges. The physical signature
        // of that is a much lower total bending energy over the same hinge set;
        // the free-edge height is *not* a valid proxy here, because a flatter
        // sheet rotates like a rigid plate about the pin and its tip actually
        // sweeps lower than a floppy sheet that buckles and bunches up.
        let (grid, positions) = drape_grid(6, 3, 0.2);
        let material = FabricMaterial::default();
        let tris = grid_triangles(grid);

        // The floppy sheet carries the identical triangulation (so aerodynamics
        // and self state match) but no bending hinges; the stiff sheet adds
        // perfectly stiff hinges built from the same rest pose.
        let mut floppy = build_grid_garment(grid, &positions, &material, stiff_solver());
        floppy.set_triangles(tris.clone());
        let mut stiff = build_grid_garment(grid, &positions, &material, stiff_solver());
        stiff.set_triangles(tris);
        stiff.build_bending(Compliance(0.0));
        // A 6x3 interior produces a non-empty hinge stencil to measure against.
        assert!(!stiff.bending.is_empty());

        for c in 0..grid.cols as usize {
            floppy.pin(c);
            stiff.pin(c);
        }
        for _ in 0..30 {
            floppy.step(1.0 / 60.0);
            stiff.step(1.0 / 60.0);
        }

        // Evaluate the same hinge stencils on both configurations. Curvature
        // (bending energy) is the direct measure of how much each sheet folded.
        let stiff_bend: f32 = stiff
            .bending
            .iter()
            .map(|h| h.energy(&stiff.particles))
            .sum();
        let floppy_bend: f32 = stiff
            .bending
            .iter()
            .map(|h| h.energy(&floppy.particles))
            .sum();

        // The stiff sheet must be dramatically flatter: at least an order of
        // magnitude less bending energy than the un-hinged sheet.
        assert!(
            floppy_bend > 1e-3,
            "floppy sheet should have curled measurably: {floppy_bend}"
        );
        assert!(
            stiff_bend < floppy_bend * 0.1,
            "stiff bending energy {stiff_bend} should be far below floppy {floppy_bend}"
        );
        assert_finite(&stiff.particles);
        assert_finite(&floppy.particles);
    }

    #[test]
    fn ccd_keeps_a_fast_sheet_above_the_floor() {
        let (grid, positions) = drape_grid(3, 3, 0.2);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        // A ground plane just below the sheet, and CCD enabled.
        garment.colliders.push(BodyCollider::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: -0.05,
        });
        garment.enable_ccd(CcdParams {
            skin: 0.001,
            restitution: 0.0,
            enabled: true,
        });
        for _ in 0..60 {
            garment.step(1.0 / 60.0);
        }
        // No particle sinks below the floor.
        for p in &garment.particles {
            assert!(p.position.y >= -0.05 - 1e-3, "y = {}", p.position.y);
        }
        assert_finite(&garment.particles);
    }

    #[test]
    fn tearing_removes_edges_and_recolors() {
        let (grid, positions) = drape_grid(6, 2, 0.2);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        garment.pin(0);
        garment.pin(1);
        let before = garment.graph.constraint_count();
        // Pull the free tip far away so several edges exceed the break strain.
        let tip = grid.index(grid.rows - 1, 0) as usize;
        garment.particles[tip].position = Vec3::new(0.0, -5.0, 0.0);
        let torn = garment.tear(TearingParams { break_strain: 0.5 });
        assert!(torn > 0);
        assert_eq!(garment.graph.constraint_count(), before - torn);
        // The recolored graph is still steppable and finite.
        for _ in 0..20 {
            garment.step(1.0 / 60.0);
        }
        assert_finite(&garment.particles);
    }

    #[test]
    fn plasticity_relaxes_an_over_stretched_edge() {
        let (grid, positions) = drape_grid(2, 2, 1.0);
        let material = FabricMaterial::default();
        let mut garment = build_grid_garment(grid, &positions, &material, stiff_solver());
        // Stretch the whole sheet along X by moving the second column far out.
        garment.particles[1].position = Vec3::new(3.0, 0.0, 0.0);
        garment.particles[3].position = Vec3::new(3.0, 0.0, 1.0);
        // Find a horizontal structural edge (0-1) rest length before/after.
        let rest_before = garment
            .graph
            .constraints
            .iter()
            .find(|c| (c.a == 0 && c.b == 1) || (c.a == 1 && c.b == 0))
            .map(|c| c.rest_length)
            .expect("edge 0-1 exists");
        garment.relax_plastic(PlasticParams {
            yield_strain: 0.1,
            creep: 0.5,
            max_strain: 10.0,
        });
        let rest_after = garment
            .graph
            .constraints
            .iter()
            .find(|c| (c.a == 0 && c.b == 1) || (c.a == 1 && c.b == 0))
            .map(|c| c.rest_length)
            .expect("edge 0-1 exists");
        assert!(
            rest_after > rest_before + 1e-4,
            "rest should creep up: {rest_before} -> {rest_after}"
        );
    }
}
