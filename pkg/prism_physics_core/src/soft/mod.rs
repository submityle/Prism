//! Unified soft-body / cloth / rope kernel (milestone M4).
//!
//! Prism's deformable simulation is built on a single idea from the engine
//! design: *everything is particles plus constraints*, solved with substep
//! Extended Position-Based Dynamics (XPBD). Cloth is a triangle mesh of
//! particles, rope and hair are one-dimensional constraint chains, and a
//! volumetric soft body is a tetrahedral lattice of particles. All three reuse
//! the same [`particle`] store and the same constraint projection, differing
//! only in how their particles and constraints are wired together.
//!
//! This module provides the particle foundation ([`particle`]), the constraint
//! primitives ([`constraint`]), the substep solver ([`solver`]), the
//! cloth/rope/soft-body builders ([`build`]), and the position-level contact
//! resolution ([`collision`]) that keeps a garment out of itself.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! substep XPBD kernel and its distance / bending / volume constraints are
//! implemented from standard, publicly documented position-based-dynamics
//! literature (Müller et al., "XPBD", 2016/2020).

pub mod body;
pub mod build;
pub mod collision;
pub mod constraint;
pub mod particle;
pub mod solver;

pub use body::SoftBody;
pub use build::{Cloth, ClothGrid, Rope, RopeGrid, SoftBox, SoftBoxGrid};
pub use collision::{
    apply_backstop, closest_point_on_segment, project_out_of_half_space, project_out_of_sphere,
    resolve_backstops, resolve_body_collisions, resolve_body_collisions_with_friction,
    resolve_self_collision, resolve_self_collision_with_friction, Backstop, BodyCollider,
};
pub use constraint::{
    AttachmentConstraint, BendingConstraint, ConstraintSet, DistanceConstraint, ParticleConstraint,
    SoftConstraintKind, TetraVolumeConstraint,
};
pub use particle::{ParticleHandle, ParticleStorage};
pub use solver::{SelfCollisionParams, SoftSolver, SoftSolverConfig};
