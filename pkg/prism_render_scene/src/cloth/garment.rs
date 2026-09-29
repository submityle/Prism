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
use bevy_ecs::resource::Resource;

use prism_render_architecture::cloth::bending::BendingConstraint;
use prism_render_architecture::cloth::Constraint;

use super::abi::{GpuClothBackstop, GpuClothCollider, GpuClothEmbedBinding};
use super::solve_plan::ClothSolveInput;

/// The authored `CPU` state of one cloth garment, spawned on a main-world
/// entity.
///
/// Positions pack the inverse mass into `.w` (`<= 0` pins the particle). The
/// `constraints` list is the authored mixed-kind set; the prepare stage
/// partitions and colors it, so the author never has to pre-sort it. The
/// collider / backstop / embed slices are already in their `#[repr(C)]` device
/// form because they carry no `CPU`-golden solver type to reorder.
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub(crate) struct ClothGarment {
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
}
