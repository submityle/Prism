//! `GPU` 6-DOF rigid-body integrator.
//!
//! This module advances free rigid bodies — three translational and three
//! rotational degrees of freedom each — under per-body external forces and
//! torques. It is the dynamics base layer the later contact and joint solvers
//! build on; on its own it integrates motion and nothing else.
//!
//! # Layout
//!
//! * [`body`] — the structure-of-arrays [`RigidBodyState`] every stage reads
//!   and writes, including the body-frame diagonal inverse inertia.
//! * [`config`] — the [`IntegratorConfig`] tunables (gravity, substeps,
//!   damping) and the [`RigidError`] type.
//! * [`cpu`] — the authoritative [`cpu_integrate`] golden reference.
//! * [`gpu`] — the [`GpuRigidIntegrator`] device twin that must reproduce the
//!   `CPU` trajectory frame for frame.
//! * [`gyroscopic`] — the optional implicit (backward-Euler) gyroscopic
//!   coupling solver ([`GyroscopicConfig`], [`GyroscopicMode`]) the angular
//!   update can use in place of the default explicit treatment.
//!
//! # Scheme and scope
//!
//! The integrator uses semi-implicit (symplectic) Euler for translation and the
//! explicit-gyroscopic form of Euler's rigid-body equations for rotation, with
//! the orientation advanced by the quaternion kinematic equation. See
//! [`cpu`] for the full derivation and the honest limits of the explicit
//! gyroscopic treatment (robust for gameplay spin rates). The optional
//! implicit-gyroscopic variant in [`gyroscopic`], selected through
//! [`cpu_integrate_gyro`], stays stable in the intermediate-axis
//! (Dzhanibekov) regime where the explicit form can gain energy; the
//! contact/joint solvers that consume this integrator are later slices.
//!
//! Provenance: Euler's rigid-body equations with explicit gyroscopic coupling
//! and the quaternion kinematic equation (Baraff & Witkin). No Unreal Engine
//! source or derived code.

mod body;
mod config;
mod contact;
mod contact_coloring;
mod contact_cpu;
mod contact_gpu;
mod contact_tgs_config;
mod contact_tgs_cpu;
mod contact_tgs_gpu;
mod cpu;
mod gpu;
mod gyroscopic;
mod joint;

pub use body::RigidBodyState;
pub use config::{ContactSolverConfig, IntegratorConfig, RigidError};
pub use contact::RigidContact;
pub use contact_coloring::RigidContactColouring;
pub use contact_cpu::cpu_solve_contacts;
pub use contact_gpu::GpuRigidContactSolver;
pub use contact_tgs_config::TgsContactConfig;
pub use contact_tgs_cpu::cpu_solve_contacts_tgs;
pub use contact_tgs_gpu::GpuRigidTgsContactSolver;
pub use cpu::{cpu_integrate, cpu_integrate_gyro};
pub use gpu::GpuRigidIntegrator;
pub use gyroscopic::{GyroscopicConfig, GyroscopicMode};
pub use joint::{
    cpu_solve_joints_angular_slerp_drive, cpu_solve_joints_cylindrical,
    cpu_solve_joints_cylindrical_drive, cpu_solve_joints_cylindrical_limit,
    cpu_solve_joints_distance, cpu_solve_joints_elliptical_cone_twist, cpu_solve_joints_fixed,
    cpu_solve_joints_gear, cpu_solve_joints_hinge_limit, cpu_solve_joints_prismatic,
    cpu_solve_joints_prismatic_drive, cpu_solve_joints_prismatic_limit,
    cpu_solve_joints_rack_pinion, cpu_solve_joints_revolute, cpu_solve_joints_revolute_drive,
    cpu_solve_joints_revolute_motor, cpu_solve_joints_revolute_servo, cpu_solve_joints_spherical,
    cpu_solve_joints_swing_twist, cpu_solve_joints_universal, AngularSlerpDriveJoint,
    CylindricalDriveJoint, CylindricalJoint, CylindricalLimitJoint, DistanceJoint,
    EllipticalConeTwistJoint, FixedJoint, GearJoint, GpuAngularSlerpDriveJointSolver,
    GpuCylindricalDriveJointSolver, GpuCylindricalJointSolver, GpuCylindricalLimitJointSolver,
    GpuDistanceJointSolver, GpuEllipticalConeTwistJointSolver, GpuFixedJointSolver,
    GpuGearJointSolver, GpuHingeLimitJointSolver, GpuPrismaticDriveJointSolver,
    GpuPrismaticJointSolver, GpuPrismaticLimitJointSolver, GpuRackPinionJointSolver,
    GpuRevoluteDriveJointSolver, GpuRevoluteJointSolver, GpuRevoluteMotorJointSolver,
    GpuRevoluteServoJointSolver, GpuSphericalJointSolver, GpuSwingTwistJointSolver,
    GpuUniversalJointSolver, HingeLimitJoint, JointColouring, JointSolverConfig,
    PrismaticDriveJoint, PrismaticJoint, PrismaticLimitJoint, RackPinionJoint, RevoluteDriveJoint,
    RevoluteJoint, RevoluteMotorJoint, RevoluteServoJoint, SphericalJoint, SwingTwistJoint,
    UniversalJoint, MAX_JOINT_BATCHES,
};
