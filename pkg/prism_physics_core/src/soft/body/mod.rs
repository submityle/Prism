//! A self-contained deformable body: particles, constraints, and solver config.
//!
//! [`SoftBody`] is the top-level unit the rest of the engine (and the cloth /
//! rope / soft-body builders) works with. It bundles a [`ParticleStorage`], a
//! [`ConstraintSet`], and a [`SoftSolverConfig`], and advances them with the
//! shared [`SoftSolver`]. Cloth, rope, and volumetric soft bodies are all just
//! a [`SoftBody`] populated with different particles and constraints.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::constraint::{
    build_dihedral_bending, AttachmentConstraint, BendingConstraint, ConstraintSet,
    DistanceConstraint, TetraVolumeConstraint,
};
use crate::soft::particle::{ParticleHandle, ParticleStorage};
use crate::soft::solver::{SoftSolver, SoftSolverConfig};
use crate::soft::topology::{
    TearableMesh, TopologyMeshError, TopologyTearEvent, TopologyTearUpdate, TopologyTearingConfig,
};

/// A deformable body advanced by the substep XPBD [`SoftSolver`].
///
/// Construct one directly and add particles and constraints, or use the
/// builders in [`crate::soft::build`] to author cloth, rope, and soft bodies.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftBody {
    /// The point-mass particles making up this body.
    pub particles: ParticleStorage,
    /// The constraints coupling this body's particles.
    pub constraints: ConstraintSet,
    /// Solver tunables (gravity, substeps, iterations, damping).
    pub config: SoftSolverConfig,
    /// Optional triangle topology used for real seam/vertex tearing.
    pub topology: Option<TearableMesh>,
    /// Structural compliance used when rebuilding distance edges after a tear.
    pub topology_distance_compliance: Real,
    /// Isometric bending compliance used for post-tear triangle hinges.
    pub topology_bending_compliance: Real,
    /// Bounded deterministic tearing policy for this body.
    pub topology_config: TopologyTearingConfig,
    /// Most recent complete render/collision replacement from a topology edit.
    pub topology_update: Option<TopologyTearUpdate>,
}

impl SoftBody {
    /// Creates an empty soft body with the given solver configuration.
    #[must_use]
    pub fn new(config: SoftSolverConfig) -> SoftBody {
        SoftBody {
            particles: ParticleStorage::new(),
            constraints: ConstraintSet::new(),
            config,
            topology: None,
            topology_distance_compliance: 0.0,
            topology_bending_compliance: 0.02,
            topology_config: TopologyTearingConfig::default(),
            topology_update: None,
        }
    }

    /// Creates an empty soft body with the default solver configuration.
    #[must_use]
    pub fn with_default_config() -> SoftBody {
        SoftBody::new(SoftSolverConfig::default())
    }

    /// Spawns a dynamic particle at `position` with the given `mass`.
    pub fn spawn(&mut self, position: Vec3, mass: Real) -> ParticleHandle {
        self.particles.spawn(position, mass)
    }

    /// Spawns a pinned (immovable) particle at `position`.
    pub fn spawn_pinned(&mut self, position: Vec3) -> ParticleHandle {
        self.particles.spawn_pinned(position)
    }

    /// Adds a distance constraint between `a` and `b` whose rest length is the
    /// current separation. Returns `false` (adding nothing) if either handle is
    /// out of range.
    pub fn connect(&mut self, a: ParticleHandle, b: ParticleHandle, compliance: Real) -> bool {
        match DistanceConstraint::from_positions(a, b, self.particles.positions(), compliance) {
            Some(c) => {
                self.constraints.distance.push(c);
                true
            }
            None => false,
        }
    }

    /// Adds a pre-built distance constraint.
    pub fn add_distance(&mut self, constraint: DistanceConstraint) {
        self.constraints.distance.push(constraint);
    }

    /// Adds a pre-built bending constraint.
    pub fn add_bending(&mut self, constraint: BendingConstraint) {
        self.constraints.bending.push(constraint);
    }

    /// Adds a pre-built tetrahedral volume constraint.
    pub fn add_volume(&mut self, constraint: TetraVolumeConstraint) {
        self.constraints.volume.push(constraint);
    }

    /// Adds a pre-built attachment constraint.
    pub fn add_attachment(&mut self, constraint: AttachmentConstraint) {
        self.constraints.attachment.push(constraint);
    }

    /// Attaches a triangle topology to this body's particle columns.
    ///
    /// The topology becomes the authoritative render/tear index buffer. The
    /// particle columns remain the authoritative solver state and are kept in
    /// sync at every topology edit. Existing constraints are preserved until
    /// the first edit; after a split, structural distances and arbitrary-mesh
    /// isometric hinges are rebuilt from the new triangle graph.
    pub fn attach_topology(
        &mut self,
        mut topology: TearableMesh,
        distance_compliance: Real,
        bending_compliance: Real,
    ) -> Result<(), TopologyMeshError> {
        if topology.vertex_count() != self.particles.len() {
            return Err(TopologyMeshError::AttributeLengthMismatch);
        }
        topology
            .positions
            .copy_from_slice(self.particles.positions());
        topology
            .velocities
            .copy_from_slice(self.particles.velocities());
        topology
            .inverse_masses
            .copy_from_slice(self.particles.inverse_masses());
        self.topology_distance_compliance = distance_compliance.max(0.0);
        self.topology_bending_compliance = bending_compliance.max(0.0);
        self.topology = Some(topology);
        self.topology_update = self
            .topology
            .as_ref()
            .map(|mesh| mesh.tear_update(Vec::new()));
        Ok(())
    }

    /// Returns the current topology revision, if topology tearing is enabled.
    #[must_use]
    pub fn topology_version(&self) -> Option<u64> {
        self.topology
            .as_ref()
            .map(|topology| topology.topology_version)
    }

    /// Returns the latest complete topology bridge payload.
    #[must_use]
    pub fn topology_update(&self) -> Option<&TopologyTearUpdate> {
        self.topology_update.as_ref()
    }

    /// Takes the latest topology bridge payload, if it has not been consumed.
    pub(crate) fn take_topology_update(&mut self) -> Option<TopologyTearUpdate> {
        self.topology_update.take()
    }

    /// Advances stress/strain-driven topology tearing and synchronizes all
    /// duplicated particle columns and topology-derived constraints.
    pub fn advance_topology_tearing(&mut self) -> Vec<TopologyTearEvent> {
        self.sync_topology_from_particles();
        let Some(topology) = self.topology.as_mut() else {
            return Vec::new();
        };
        let old_count = topology.vertex_count();
        let events = topology.advance(self.topology_config);
        if events.is_empty() {
            return events;
        }
        self.sync_particles_from_topology(old_count);
        self.rebuild_topology_constraints();
        self.topology_update = self
            .topology
            .as_ref()
            .map(|topology| topology.tear_update(events.clone()));
        events
    }

    /// Applies a point-directed puncture and performs the same particle,
    /// render-topology, and constraint synchronization as front tearing.
    pub fn puncture_topology(
        &mut self,
        point: Vec3,
        direction: Vec3,
        radius: Real,
    ) -> Option<Vec<TopologyTearEvent>> {
        self.sync_topology_from_particles();
        let topology = self.topology.as_mut()?;
        let old_count = topology.vertex_count();
        let events = topology.puncture(point, direction, radius, self.topology_config)?;
        self.sync_particles_from_topology(old_count);
        self.rebuild_topology_constraints();
        self.topology_update = self
            .topology
            .as_ref()
            .map(|topology| topology.tear_update(events.clone()));
        Some(events)
    }

    fn sync_topology_from_particles(&mut self) {
        let Some(topology) = self.topology.as_mut() else {
            return;
        };
        if topology.vertex_count() != self.particles.len() {
            return;
        }
        topology
            .positions
            .copy_from_slice(self.particles.positions());
        topology
            .velocities
            .copy_from_slice(self.particles.velocities());
        topology
            .inverse_masses
            .copy_from_slice(self.particles.inverse_masses());
    }

    fn sync_particles_from_topology(&mut self, old_count: usize) {
        let Some(topology) = self.topology.as_ref() else {
            return;
        };
        for index in old_count..topology.vertex_count() {
            self.particles.append_state(
                topology.positions[index],
                topology.positions[index],
                topology.velocities[index],
                topology.inverse_masses[index],
            );
        }
        for index in 0..topology.vertex_count() {
            let handle = ParticleHandle::from_index(index as u32);
            self.particles
                .set_position(handle, topology.positions[index]);
            self.particles
                .set_velocity(handle, topology.velocities[index]);
            self.particles
                .set_inverse_mass(handle, topology.inverse_masses[index]);
        }
    }

    fn rebuild_topology_constraints(&mut self) {
        let Some(topology) = self.topology.as_ref() else {
            return;
        };
        self.constraints.topology_distance = topology
            .active_edges()
            .into_iter()
            .filter_map(|(a, b)| {
                let rest = (topology.rest_positions[a as usize]
                    - topology.rest_positions[b as usize])
                    .length();
                (rest > 1.0e-8).then(|| {
                    DistanceConstraint::new(
                        ParticleHandle::from_index(a),
                        ParticleHandle::from_index(b),
                        rest,
                        self.topology_distance_compliance,
                    )
                })
            })
            .collect();
        self.constraints.distance.retain(|constraint| {
            !topology.seams.contains(
                &topology.root_edge(constraint.a.index() as u32, constraint.b.index() as u32),
            )
        });
        self.constraints.bending.retain(|constraint| {
            let pairs = [
                (
                    constraint.a.index() as u32,
                    constraint.center.index() as u32,
                ),
                (
                    constraint.center.index() as u32,
                    constraint.b.index() as u32,
                ),
                (constraint.a.index() as u32, constraint.b.index() as u32),
            ];
            !pairs
                .iter()
                .any(|&(a, b)| topology.seams.contains(&topology.root_edge(a, b)))
        });
        self.constraints.isometric_bending = build_dihedral_bending(
            &topology.rest_positions,
            &topology.triangles,
            self.topology_bending_compliance,
        );
    }

    /// Advances the body by `dt` seconds using the substep XPBD solver.
    pub fn step(&mut self, dt: Real) {
        SoftSolver::new().step(&mut self.particles, &mut self.constraints, &self.config, dt);
    }

    /// Returns the axis-aligned bounding box of the current particle positions
    /// as `(min, max)`, or `None` when the body has no particles.
    #[must_use]
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let positions = self.particles.positions();
        let (first, rest) = positions.split_first()?;
        let mut min = *first;
        let mut max = *first;
        for &p in rest {
            min = min.min(p);
            max = max.max(p);
        }
        Some((min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soft::build::ClothGrid;

    #[test]
    fn new_body_is_empty() {
        let body = SoftBody::with_default_config();
        assert!(body.particles.is_empty());
        assert!(body.constraints.is_empty());
        assert!(body.bounds().is_none());
    }

    #[test]
    fn connect_uses_current_separation_as_rest_length() {
        let mut body = SoftBody::with_default_config();
        let a = body.spawn(Vec3::ZERO, 1.0);
        let b = body.spawn(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(body.connect(a, b, 0.0));
        assert_eq!(body.constraints.distance.len(), 1);
        assert!((body.constraints.distance[0].rest_length - 3.0).abs() < 1e-6);
    }

    #[test]
    fn connect_rejects_out_of_range_handle() {
        let mut body = SoftBody::with_default_config();
        let a = body.spawn(Vec3::ZERO, 1.0);
        assert!(!body.connect(a, ParticleHandle::from_index(9), 0.0));
        assert!(body.constraints.distance.is_empty());
    }

    #[test]
    fn bounds_span_all_particles() {
        let mut body = SoftBody::with_default_config();
        body.spawn(Vec3::new(-1.0, 0.0, 2.0), 1.0);
        body.spawn(Vec3::new(3.0, -4.0, 1.0), 1.0);
        let (min, max) = body.bounds().unwrap();
        assert_eq!(min, Vec3::new(-1.0, -4.0, 1.0));
        assert_eq!(max, Vec3::new(3.0, 0.0, 2.0));
    }

    #[test]
    fn step_moves_dynamic_particles() {
        let mut body = SoftBody::new(SoftSolverConfig {
            damping: 0.0,
            ..SoftSolverConfig::default()
        });
        let p = body.spawn(Vec3::ZERO, 1.0);
        body.step(1.0 / 60.0);
        assert!(body.particles.position(p).unwrap().y < 0.0);
    }

    #[test]
    fn topology_tearing_duplicates_particles_and_rebuilds_constraints() {
        let mut cloth = ClothGrid {
            columns: 2,
            rows: 2,
            spacing: 1.0,
            compliance: 0.0,
            ..ClothGrid::default()
        }
        .build_default();
        cloth.body.topology_config = TopologyTearingConfig {
            break_stress: 0.1,
            max_front_steps: 1,
            ..TopologyTearingConfig::default()
        };
        // The two triangles share the diagonal 0-3. Stretching it makes the
        // only interior edge fail while boundary edges remain irrelevant.
        cloth
            .body
            .particles
            .set_position(ParticleHandle::from_index(3), Vec3::new(3.0, 0.0, 3.0));
        let events = cloth.body.advance_topology_tearing();
        assert!(events
            .iter()
            .any(|event| event.kind == crate::soft::topology::TopologyTearEventKind::EdgeBreak));
        assert!(events.iter().any(|event| {
            event.kind == crate::soft::topology::TopologyTearEventKind::VertexSplit
        }));
        assert!(cloth.body.particles.len() > 4);
        assert!(!cloth.body.constraints.topology_distance.is_empty());
        let update = cloth.body.topology_update().expect("topology update");
        assert_eq!(update.topology_version, 1);
        assert_eq!(update.render.vertices.len(), cloth.body.particles.len());
        assert_eq!(
            update.collision.triangle_bounds.len(),
            update.collision.triangles.len()
        );
        assert_eq!(
            cloth.body.topology.as_ref().unwrap().vertex_count(),
            cloth.body.particles.len()
        );
    }
}
