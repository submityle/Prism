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
//! This module currently provides the particle foundation ([`particle`]);
//! subsequent parts of M4 add the constraint primitives, the substep solver,
//! and the cloth/rope/soft-body builders.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! substep XPBD kernel and its distance / bending / volume constraints are
//! implemented from standard, publicly documented position-based-dynamics
//! literature (Müller et al., "XPBD", 2016/2020).

pub mod constraint;
pub mod particle;

pub use constraint::{
    AttachmentConstraint, BendingConstraint, DistanceConstraint, ParticleConstraint,
    SoftConstraintKind, TetraVolumeConstraint,
};
pub use particle::{ParticleHandle, ParticleStorage};
