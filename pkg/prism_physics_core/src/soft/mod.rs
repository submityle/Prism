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

pub mod aero;
pub mod body;
pub mod build;
pub mod collision;
pub mod constraint;
pub mod damage;
pub mod particle;
pub mod rigid_coupling;
pub mod solver;

pub use aero::{
    apply_aero_forces, apply_aero_to_columns, triangle_aero_force, turbulence_offset, AeroParams,
    WindField,
};
pub use body::SoftBody;
pub use build::{Cloth, ClothGrid, Rope, RopeGrid, SoftBox, SoftBoxGrid};
pub use collision::{
    accumulate_vertex_normals, apply_backstop, capsule_toi, closest_point_on_segment,
    couple_particle_against_body, generate_virtual_particles, half_space_toi,
    project_out_of_half_space, project_out_of_sphere, resolve_backstops, resolve_body_collisions,
    resolve_body_collisions_with_friction, resolve_ccd, resolve_layer_coupling,
    resolve_layer_coupling_jacobi, resolve_self_ccd, resolve_self_ccd_jacobi,
    resolve_self_collision, resolve_self_collision_jacobi, resolve_self_collision_virtual,
    resolve_self_collision_virtual_augment, resolve_self_collision_virtual_augment_jacobi,
    resolve_self_collision_virtual_jacobi, resolve_self_collision_with_friction,
    resolve_self_collision_with_friction_jacobi, resolve_two_way_coupling, sphere_toi,
    swept_pair_toi, Backstop, BodyCollider, CcdParams, CouplingBody, CouplingContribution,
    LayerParams, SelfCcdParams, VirtualParticle, VirtualParticlePattern,
};
pub use constraint::{
    mesh_volume, AttachmentConstraint, BendingConstraint, ConstraintSet, DistanceConstraint,
    LongRangeConstraint, ParticleConstraint, PressureConstraint, SoftConstraintKind,
    StrainLimitConstraint, TetraVolumeConstraint,
};
pub use damage::{
    apply_plasticity, apply_tearing, tear_flags, tear_report, PlasticParams, TearReport,
    TearingParams,
};
pub use particle::{ParticleHandle, ParticleStorage};
pub use rigid_coupling::{
    couple_cloth_to_rigid, couple_cloth_to_rigid_angular, couple_cloth_to_rigid_friction,
    gather_rigid_proxies, Aabb as CouplingAabb, AngularCouplingReport, ClothRigidCouplingConfig,
    CouplingReport, FrictionCouplingReport, RigidProxy,
};
pub use solver::{
    SelfCollisionParams, SoftContacts, SoftSolver, SoftSolverConfig, VirtualSelfCollisionParams,
};
