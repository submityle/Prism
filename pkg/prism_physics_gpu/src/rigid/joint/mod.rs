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
//! * [`revolute_drive`] — the [`RevoluteDriveJoint`] (hinge with an angular
//!   position drive/motor) definition and its device-packed storage
//!   representation.
//! * [`revolute_drive_cpu`] — the authoritative
//!   [`cpu_solve_joints_revolute_drive`] golden stepper (axis alignment plus
//!   an about-axis compliant-and-damped angular position drive plus a
//!   point-to-point weld).
//! * [`revolute_drive_gpu`] — the device-side [`GpuRevoluteDriveJointSolver`]
//!   twin.
//! * [`revolute_motor`] — the [`RevoluteMotorJoint`] (hinge with an angular
//!   velocity motor) definition and its device-packed storage representation.
//! * [`revolute_motor_cpu`] — the authoritative
//!   [`cpu_solve_joints_revolute_motor`] golden stepper (axis alignment plus
//!   an about-axis compliant velocity motor plus a point-to-point weld).
//! * [`revolute_motor_gpu`] — the device-side [`GpuRevoluteMotorJointSolver`]
//!   twin.
//! * [`cylindrical`] — the [`CylindricalJoint`] definition and its
//!   device-packed storage representation.
//! * [`cylindrical_cpu`] — the authoritative [`cpu_solve_joints_cylindrical`]
//!   golden stepper (axis alignment plus a point-on-line weld, freeing the
//!   slide along and the spin about a shared axis).
//! * [`cylindrical_gpu`] — the device-side [`GpuCylindricalJointSolver`] twin.
//! * [`cylindrical_drive`] — the [`CylindricalDriveJoint`] (cylindrical
//!   joint with a linear position drive/motor along its slide axis)
//!   definition and its device-packed storage representation.
//! * [`cylindrical_drive_cpu`] — the authoritative
//!   [`cpu_solve_joints_cylindrical_drive`] golden stepper (axis alignment
//!   plus a point-on-line weld plus an along-axis compliant-and-damped
//!   linear position drive, leaving the spin about the axis free).
//! * [`cylindrical_drive_gpu`] — the device-side
//!   [`GpuCylindricalDriveJointSolver`] twin.
//! * [`cylindrical_limit`] — the [`CylindricalLimitJoint`] (cylindrical
//!   joint with a one-sided travel limit along its slide axis)
//!   definition and its device-packed storage representation.
//! * [`cylindrical_limit_cpu`] — the authoritative
//!   [`cpu_solve_joints_cylindrical_limit`] golden stepper (axis alignment
//!   plus a point-on-line weld plus an along-axis one-sided travel limit
//!   with a free dead zone, leaving the spin about the axis free).
//! * [`cylindrical_limit_gpu`] — the device-side
//!   [`GpuCylindricalLimitJointSolver`] twin.
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
//! * [`swing_twist`] — the [`SwingTwistJoint`] (cone-twist/ragdoll)
//!   definition and its device-packed storage representation.
//! * [`swing_twist_cpu`] — the authoritative
//!   [`cpu_solve_joints_swing_twist`] golden stepper (swing cone,
//!   twist limit, and point-to-point weld).
//! * [`swing_twist_gpu`] — the device-side [`GpuSwingTwistJointSolver`]
//!   twin.
//! * [`universal`] — the [`UniversalJoint`] (Cardan/Hooke) definition and its
//!   device-packed storage representation.
//! * [`universal_cpu`] — the authoritative [`cpu_solve_joints_universal`]
//!   golden stepper (perpendicularity plus point-to-point weld).
//! * [`universal_gpu`] — the device-side [`GpuUniversalJointSolver`] twin.
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
mod cylindrical;
mod cylindrical_cpu;
mod cylindrical_drive;
mod cylindrical_drive_cpu;
mod cylindrical_drive_gpu;
mod cylindrical_gpu;
mod cylindrical_limit;
mod cylindrical_limit_cpu;
mod cylindrical_limit_gpu;
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
mod revolute_drive;
mod revolute_drive_cpu;
mod revolute_drive_gpu;
mod revolute_gpu;
mod revolute_motor;
mod revolute_motor_cpu;
mod revolute_motor_gpu;
mod spherical;
mod spherical_cpu;
mod spherical_gpu;
mod stepper;
mod swing_twist;
mod swing_twist_cpu;
mod swing_twist_gpu;
mod universal;
mod universal_cpu;
mod universal_gpu;

pub use coloring::{JointColouring, MAX_JOINT_BATCHES};
pub use config::JointSolverConfig;
pub use cylindrical::CylindricalJoint;
pub use cylindrical_cpu::cpu_solve_joints_cylindrical;
pub use cylindrical_drive::CylindricalDriveJoint;
pub use cylindrical_drive_cpu::cpu_solve_joints_cylindrical_drive;
pub use cylindrical_drive_gpu::GpuCylindricalDriveJointSolver;
pub use cylindrical_gpu::GpuCylindricalJointSolver;
pub use cylindrical_limit::CylindricalLimitJoint;
pub use cylindrical_limit_cpu::cpu_solve_joints_cylindrical_limit;
pub use cylindrical_limit_gpu::GpuCylindricalLimitJointSolver;
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
pub use revolute_drive::RevoluteDriveJoint;
pub use revolute_drive_cpu::cpu_solve_joints_revolute_drive;
pub use revolute_drive_gpu::GpuRevoluteDriveJointSolver;
pub use revolute_gpu::GpuRevoluteJointSolver;
pub use revolute_motor::RevoluteMotorJoint;
pub use revolute_motor_cpu::cpu_solve_joints_revolute_motor;
pub use revolute_motor_gpu::GpuRevoluteMotorJointSolver;
pub use spherical::SphericalJoint;
pub use spherical_cpu::cpu_solve_joints_spherical;
pub use spherical_gpu::GpuSphericalJointSolver;
pub use swing_twist::SwingTwistJoint;
pub use swing_twist_cpu::cpu_solve_joints_swing_twist;
pub use swing_twist_gpu::GpuSwingTwistJointSolver;
pub use universal::UniversalJoint;
pub use universal_cpu::cpu_solve_joints_universal;
pub use universal_gpu::GpuUniversalJointSolver;
