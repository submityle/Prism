//! Pre-authored coarse simulation mesh for the reduced LOD tier.
//!
//! The screen-coverage LOD gate ([`super::lod`]) can classify a garment down to
//! [`ClothLodTier::ReducedSim`](prism_render_architecture::cloth::ClothLodTier::ReducedSim),
//! but a tier that only *records a smaller budget* while still solving the full
//! authored mesh would be a lie: the per-frame solve cost would not actually
//! drop. This module lands the honest device consequence of the reduced tier,
//! mirroring UE5 `Chaos Cloth`'s per-LOD sim meshes: a garment carries an
//! optional, pre-authored, coarser-resolution simulation mesh, and the prepare
//! stage solves *that* mesh (fewer particles, fewer constraints, fewer
//! dispatched work-items) whenever the garment's coverage drops it to the
//! reduced tier.
//!
//! Only the *resolution-dependent* buffers live here. The material and
//! environment scalars a garment carries (wind, aerodynamics, friction, the
//! analytic body colliders, gravity, the timestep and the substep/iteration
//! counts, the strain limit and the self-collision thickness/cell size) are
//! properties of the cloth and the world, not of a particular mesh resolution,
//! so both LODs share them straight off the [`ClothGarment`] and this reduced
//! mesh never re-authors them. The render-vertex count is likewise shared (the
//! render mesh is the same at every LOD); the reduced mesh only re-binds how
//! those render vertices embed into its own, coarser sim triangles.
//!
//! A garment with no reduced mesh keeps the honest fallback the reduced tier
//! shipped with: it re-solves the full mesh, which is the correct behavior when
//! no coarse geometry was authored (never a fabricated decimation).
//!
//! [`ClothGarment`]: super::garment::ClothGarment

use prism_render_architecture::cloth::bending::BendingConstraint;
use prism_render_architecture::cloth::collision::Backstop;
use prism_render_architecture::cloth::embed::BarycentricBinding;
use prism_render_architecture::cloth::polygon_garment::PolygonGarmentMesh;
use prism_render_architecture::cloth::{ClothParticle, Constraint};

use super::abi::{GpuClothBackstop, GpuClothEmbedBinding};
use super::pack::{pack_backstops, pack_embed_bindings};

/// A pre-authored, coarser-resolution simulation mesh a garment swaps to when
/// its LOD decision drops to the reduced-simulation tier.
///
/// Holds only the resolution-dependent solve buffers (see the module docs);
/// every material/environment scalar is shared off the parent
/// [`ClothGarment`](super::garment::ClothGarment). Build one with
/// [`ClothReducedMeshBuilder`] and attach it through
/// [`ClothGarmentBuilder::reduced_lod_mesh`](super::authoring::ClothGarmentBuilder::reduced_lod_mesh).
#[derive(Clone, Debug, PartialEq)]
pub struct ClothReducedMesh {
    /// Coarse-mesh particle positions with inverse mass in `.w`.
    pub(crate) positions: Vec<[f32; 4]>,
    /// Coarse-mesh particle velocities; `.w` is unused padding.
    pub(crate) velocities: Vec<[f32; 4]>,
    /// Coarse-mesh distance and attachment constraints, mixed kinds.
    pub(crate) constraints: Vec<Constraint>,
    /// Coarse-mesh dihedral bending hinges.
    pub(crate) bending: Vec<BendingConstraint>,
    /// Coarse-mesh triangles the aerodynamic gather integrates wind over.
    pub(crate) triangles: Vec<[u32; 3]>,
    /// Coarse-mesh painted backstop planes, one per constrained coarse particle.
    pub(crate) backstops: Vec<GpuClothBackstop>,
    /// Render-vertex embed bindings that skin the *shared* render mesh from this
    /// coarse sim mesh. Empty leaves the render mesh unbound at the reduced tier.
    pub(crate) embed_bindings: Vec<GpuClothEmbedBinding>,
    /// Number of self-collision hash cells for the coarse mesh (`0` disables it).
    pub(crate) hash_cell_count: u32,
}

/// Splits a [`ClothParticle`] into the position (inverse mass in `.w`) and
/// velocity rows the resident reduced mesh stores, matching the full-mesh
/// authoring layout.
#[must_use]
fn split_particle(particle: &ClothParticle) -> ([f32; 4], [f32; 4]) {
    let position = [
        particle.position.x,
        particle.position.y,
        particle.position.z,
        particle.inverse_mass,
    ];
    let velocity = [particle.velocity.x, particle.velocity.y, particle.velocity.z, 0.0];
    (position, velocity)
}

/// Fluent builder for a [`ClothReducedMesh`], mirroring the resolution-dependent
/// subset of [`ClothGarmentBuilder`](super::authoring::ClothGarmentBuilder).
///
/// Start from raw coarse particles ([`ClothReducedMeshBuilder::from_particles`])
/// or a welded coarse [`PolygonGarmentMesh`]
/// ([`ClothReducedMeshBuilder::from_mesh`]), chain the optional setters, then
/// call [`ClothReducedMeshBuilder::build`]. Backstops and embed bindings are
/// packed into their device mirrors as they are supplied, so `build` is a cheap
/// move.
#[derive(Clone, Debug, PartialEq)]
pub struct ClothReducedMeshBuilder {
    positions: Vec<[f32; 4]>,
    velocities: Vec<[f32; 4]>,
    constraints: Vec<Constraint>,
    bending: Vec<BendingConstraint>,
    triangles: Vec<[u32; 3]>,
    backstops: Vec<GpuClothBackstop>,
    embed_bindings: Vec<GpuClothEmbedBinding>,
    hash_cell_count: u32,
}

impl Default for ClothReducedMeshBuilder {
    /// An empty coarse mesh: no particles, no constraints, no passes. Building it
    /// yields a zero-particle reduced mesh, which the prepare stage treats as
    /// "no coarse geometry", falling back to the full mesh at the reduced tier.
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            velocities: Vec::new(),
            constraints: Vec::new(),
            bending: Vec::new(),
            triangles: Vec::new(),
            backstops: Vec::new(),
            embed_bindings: Vec::new(),
            hash_cell_count: 0,
        }
    }
}

impl ClothReducedMeshBuilder {
    /// Starts a coarse-mesh builder from a slice of authored coarse particles.
    #[must_use]
    pub fn from_particles(particles: &[ClothParticle]) -> Self {
        let mut positions = Vec::with_capacity(particles.len());
        let mut velocities = Vec::with_capacity(particles.len());
        for particle in particles {
            let (position, velocity) = split_particle(particle);
            positions.push(position);
            velocities.push(velocity);
        }
        Self {
            positions,
            velocities,
            ..Self::default()
        }
    }

    /// Starts a coarse-mesh builder from a welded coarse [`PolygonGarmentMesh`],
    /// seeding the coarse particles, constraint graph and triangle membrane.
    #[must_use]
    pub fn from_mesh(mesh: &PolygonGarmentMesh) -> Self {
        Self {
            constraints: mesh.constraints.clone(),
            triangles: mesh.triangles.clone(),
            ..Self::from_particles(&mesh.particles)
        }
    }

    /// Replaces the coarse-mesh constraint set (mixed distance and attachment
    /// kinds, in any order).
    #[must_use]
    pub fn constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Sets the coarse-mesh dihedral bending hinges.
    #[must_use]
    pub fn bending(mut self, bending: Vec<BendingConstraint>) -> Self {
        self.bending = bending;
        self
    }

    /// Sets the coarse-mesh triangles the aerodynamic gather integrates wind
    /// over. An empty set leaves aerodynamics disabled at the reduced tier.
    #[must_use]
    pub fn triangles(mut self, triangles: Vec<[u32; 3]>) -> Self {
        self.triangles = triangles;
        self
    }

    /// Packs and stores the coarse-mesh painted backstop planes; `backstops[i]`
    /// constrains coarse particle `i`.
    #[must_use]
    pub fn backstops(mut self, backstops: &[Backstop]) -> Self {
        self.backstops = pack_backstops(backstops);
        self
    }

    /// Packs and stores the render-vertex embed bindings that skin the shared
    /// render mesh from this coarse sim mesh; `bindings[i]` drives render vertex
    /// `i` against coarse-mesh triangles.
    #[must_use]
    pub fn embed_bindings(mut self, bindings: &[BarycentricBinding]) -> Self {
        self.embed_bindings = pack_embed_bindings(bindings);
        self
    }

    /// Sets the coarse-mesh self-collision hash cell count. A zero count keeps
    /// self-collision disabled for the reduced tier.
    #[must_use]
    pub fn hash_cell_count(mut self, hash_cell_count: u32) -> Self {
        self.hash_cell_count = hash_cell_count;
        self
    }

    /// Consumes the builder and produces the [`ClothReducedMesh`].
    #[must_use]
    pub fn build(self) -> ClothReducedMesh {
        ClothReducedMesh {
            positions: self.positions,
            velocities: self.velocities,
            constraints: self.constraints,
            bending: self.bending,
            triangles: self.triangles,
            backstops: self.backstops,
            embed_bindings: self.embed_bindings,
            hash_cell_count: self.hash_cell_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::cloth::{Compliance, ConstraintKind, Vec3};

    fn coarse_particles() -> [ClothParticle; 3] {
        [
            ClothParticle::pinned(Vec3::new(0.0, 0.0, 0.0)),
            ClothParticle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
            ClothParticle {
                position: Vec3::new(0.0, 1.0, 0.0),
                velocity: Vec3::new(2.0, 3.0, 4.0),
                inverse_mass: 0.5,
            },
        ]
    }

    #[test]
    fn from_particles_packs_position_inverse_mass_and_velocity() {
        let mesh = ClothReducedMeshBuilder::from_particles(&coarse_particles()).build();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.velocities.len(), 3);
        // Pinned particle: zero inverse mass, zero velocity.
        assert_eq!(mesh.positions[0], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(mesh.velocities[0], [0.0, 0.0, 0.0, 0.0]);
        // Movable particle carries inverse mass in .w and its explicit velocity.
        assert_eq!(mesh.positions[2], [0.0, 1.0, 0.0, 0.5]);
        assert_eq!(mesh.velocities[2], [2.0, 3.0, 4.0, 0.0]);
        assert_eq!(mesh.positions.len(), 3);
    }

    #[test]
    fn from_mesh_seeds_particles_constraints_and_triangles() {
        let welded = PolygonGarmentMesh {
            particles: coarse_particles().to_vec(),
            constraints: vec![Constraint::new(
                0,
                1,
                1.0,
                Compliance::RIGID,
                ConstraintKind::Stretch,
            )],
            triangles: vec![[0, 1, 2]],
            ..PolygonGarmentMesh::default()
        };
        let mesh = ClothReducedMeshBuilder::from_mesh(&welded).build();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.constraints.len(), 1);
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
    }

    #[test]
    fn default_builder_yields_an_empty_reduced_mesh() {
        let mesh = ClothReducedMeshBuilder::default().build();
        assert_eq!(mesh.positions.len(), 0);
        assert_eq!(mesh.constraints.len(), 0);
        assert!(mesh.triangles.is_empty());
        assert!(mesh.embed_bindings.is_empty());
        assert_eq!(mesh.hash_cell_count, 0);
    }
}
