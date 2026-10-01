//! Position-based (`XPBD`) rigid-body joints.
//!
//! A joint couples two bodies in the shared [`RigidBodyState`](super::RigidBodyState)
//! by removing some of their relative degrees of freedom. The family starts
//! with the [`SphericalJoint`] (ball-and-socket): the point-to-point positional
//! weld that is the base of every articulated mechanism — a hinge, a slider, and
//! a configurable `6`-DOF joint are each this weld plus one or more angular
//! restrictions. Every later joint type layers on top of the same shared
//! stepper, so adding one is writing its constraint projection rather than a new
//! integrator.
//!
//! # Layout
//!
//! The module follows the crate's one-concept-per-file convention and the
//! `*_cpu` / `*_gpu` twin split, with the pieces every joint type shares pulled
//! into their own files:
//!
//! * [`spherical`] — the [`SphericalJoint`] definition and its device-packed
//!   storage representation.
//! * [`config`] — the [`JointSolverConfig`] tunables (the per-substep
//!   projection-sweep count).
//! * [`coloring`] — the static-aware [`JointColouring`] that partitions a joint
//!   set into race-free parallel batches, over the [`JointBodies`](coloring::JointBodies) endpoints
//!   every joint type exposes.
//! * [`math`] — the shared quaternion and world-space inverse-inertia helpers
//!   every joint `CPU` golden projects its constraints with.
//! * [`stepper`] — the shared per-substep body phases (`snapshot`, `predict`,
//!   `recover_velocities`) every joint `CPU` golden runs around its projection.
//! * [`gpu_core`] — the shared [`JointGpuCore`](gpu_core::JointGpuCore) `wgpu`
//!   dispatch skeleton every joint `GPU` twin runs its shader through.
//! * [`spherical_cpu`] — the authoritative [`cpu_solve_joints_spherical`] golden
//!   stepper.
//! * [`spherical_gpu`] — the device-side [`GpuSphericalJointSolver`] that
//!   reproduces the golden stepper on real `wgpu` hardware.
//! * [`revolute`] — the [`RevoluteJoint`] (hinge) definition and its
//!   device-packed storage representation.
//! * [`revolute_cpu`] — the authoritative [`cpu_solve_joints_revolute`] golden
//!   stepper (axis alignment plus point-to-point weld).
//! * [`revolute_gpu`] — the device-side [`GpuRevoluteJointSolver`] twin.
//! * [`distance`] — the [`DistanceJoint`] (limit) definition and its
//!   device-packed storage representation.
//! * [`distance_cpu`] — the authoritative [`cpu_solve_joints_distance`] golden
//!   stepper (one-sided min/max separation limit with a free dead zone).
//! * [`distance_gpu`] — the device-side [`GpuDistanceJointSolver`] twin.
//! * [`fixed`] — the [`FixedJoint`] (weld) definition and its device-packed
//!   storage representation.
//! * [`fixed_cpu`] — the authoritative [`cpu_solve_joints_fixed`] golden
//!   stepper (angular lock plus full point-to-point weld).
//! * [`fixed_gpu`] — the device-side [`GpuFixedJointSolver`] twin.
//! * [`hinge_limit`] — the [`HingeLimitJoint`] (hinge with a swing limit)
//!   definition and its device-packed storage representation.
//! * [`hinge_limit_cpu`] — the authoritative [`cpu_solve_joints_hinge_limit`]
//!   golden stepper (axis alignment, one-sided angular limit, and
//!   point-to-point weld).
//! * [`hinge_limit_gpu`] — the device-side [`GpuHingeLimitJointSolver`] twin.
//! * [`prismatic`] — the [`PrismaticJoint`] (slider) definition and its
//!   device-packed storage representation.
//! * [`prismatic_cpu`] — the authoritative [`cpu_solve_joints_prismatic`] golden
//!   stepper (angular lock plus perpendicular point-to-point weld).
//! * [`prismatic_gpu`] — the device-side [`GpuPrismaticJointSolver`] twin.
//! * [`prismatic_drive`] — the [`PrismaticDriveJoint`] (slider with a linear
//!   position drive/motor) definition and its device-packed storage
//!   representation.
//! * [`prismatic_drive_cpu`] — the authoritative
//!   [`cpu_solve_joints_prismatic_drive`] golden stepper (angular lock plus
//!   perpendicular weld plus an along-axis compliant-and-damped position
//!   drive).
//! * [`prismatic_drive_gpu`] — the device-side
//!   [`GpuPrismaticDriveJointSolver`] twin.
//! * [`prismatic_limit`] — the [`PrismaticLimitJoint`] (slider with a travel
//!   limit) definition and its device-packed storage representation.
//! * [`prismatic_limit_cpu`] — the authoritative
//!   [`cpu_solve_joints_prismatic_limit`] golden stepper (angular lock plus
//!   perpendicular weld plus a one-sided along-axis travel limit).
//! * [`prismatic_limit_gpu`] — the device-side
//!   [`GpuPrismaticLimitJointSolver`] twin.
//!
//! # Scheme and scope
//!
//! The stepper is the substep-`XPBD` scheme of Müller et al., "Detailed Rigid
//! Body Simulation with XPBD": each substep predicts the bodies forward, resets
//! the constraints' Lagrange multipliers, projects the constraints in
//! colour-batch order, then recovers the velocities from the net motion. The
//! `CPU` reference and the `GPU` kernel walk the identical reordered joint list
//! and batch order, which is what keeps them in lock-step.
//!
//! Provenance: the point-to-point (ball-socket) constraint and its substep
//! `XPBD` positional handling (Müller et al.), over the world-space inverse
//! inertia and quaternion kinematics of Baraff & Witkin. No Unreal Engine source
//! or derived code.

mod coloring;
mod config;
mod distance;
mod distance_cpu;
mod distance_gpu;
mod fixed;
mod fixed_cpu;
mod fixed_gpu;
mod gpu_core;
mod hinge_limit;
mod hinge_limit_cpu;
mod hinge_limit_gpu;
mod math;
mod prismatic;
mod prismatic_cpu;
mod prismatic_drive;
mod prismatic_drive_cpu;
mod prismatic_drive_gpu;
mod prismatic_gpu;
mod prismatic_limit;
mod prismatic_limit_cpu;
mod prismatic_limit_gpu;
mod revolute;
mod revolute_cpu;
mod revolute_gpu;
mod spherical;
mod spherical_cpu;
mod spherical_gpu;
mod stepper;

pub use coloring::{JointColouring, MAX_JOINT_BATCHES};
pub use config::JointSolverConfig;
pub use distance::DistanceJoint;
pub use distance_cpu::cpu_solve_joints_distance;
pub use distance_gpu::GpuDistanceJointSolver;
pub use fixed::FixedJoint;
pub use fixed_cpu::cpu_solve_joints_fixed;
pub use fixed_gpu::GpuFixedJointSolver;
pub use hinge_limit::HingeLimitJoint;
pub use hinge_limit_cpu::cpu_solve_joints_hinge_limit;
pub use hinge_limit_gpu::GpuHingeLimitJointSolver;
pub use prismatic::PrismaticJoint;
pub use prismatic_cpu::cpu_solve_joints_prismatic;
pub use prismatic_drive::PrismaticDriveJoint;
pub use prismatic_drive_cpu::cpu_solve_joints_prismatic_drive;
pub use prismatic_drive_gpu::GpuPrismaticDriveJointSolver;
pub use prismatic_gpu::GpuPrismaticJointSolver;
pub use prismatic_limit::PrismaticLimitJoint;
pub use prismatic_limit_cpu::cpu_solve_joints_prismatic_limit;
pub use prismatic_limit_gpu::GpuPrismaticLimitJointSolver;
pub use revolute::RevoluteJoint;
pub use revolute_cpu::cpu_solve_joints_revolute;
pub use revolute_gpu::GpuRevoluteJointSolver;
pub use spherical::SphericalJoint;
pub use spherical_cpu::cpu_solve_joints_spherical;
pub use spherical_gpu::GpuSphericalJointSolver;
