//! End-to-end cloth pipeline integration.
//!
//! This module wires the sibling cloth modules into one runnable garment:
//! asset material → constraint graph ([`super::constraints`]) → XPBD dynamics
//! ([`super::dynamics`]) → collision resolution ([`super::collision`]) →
//! render-mesh embedding ([`super::embed`]), gated by the LOD ladder
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
use super::{ClothLodTier, ClothParticle, ClothPiece, ConstraintGraph, Vec3};
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
        let colliders = &self.colliders;
        let backstops = &self.backstops;
        solve_cloth_with_collision(&mut self.particles, &self.graph, self.solver, dt, |p| {
            project_colliders(p.position, colliders, backstops)
        });
        if self.self_collision.enabled {
            resolve_self_collision(
                &mut self.particles,
                self.self_collision.cell_size,
                self.self_collision.thickness,
            );
        }
        // Self-collision can push a particle back into a body; re-project so a
        // frame never ends inside a collider.
        resolve_body_collisions(&mut self.particles, &self.colliders);
        resolve_backstops(&mut self.particles, &self.backstops);
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
        let decision = resolve_cloth_lod(piece, coverage, thresholds);
        cloth_deformation_request(piece, decision, priority)
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
    use crate::cloth::ClothPieceHandle;
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
}
