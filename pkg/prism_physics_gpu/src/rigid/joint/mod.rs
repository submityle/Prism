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
mod gpu_core;
mod math;
mod revolute;
mod revolute_cpu;
mod revolute_gpu;
mod spherical;
mod spherical_cpu;
mod spherical_gpu;
mod stepper;

pub use coloring::{JointColouring, MAX_JOINT_BATCHES};
pub use config::JointSolverConfig;
pub use revolute::RevoluteJoint;
pub use revolute_cpu::cpu_solve_joints_revolute;
pub use revolute_gpu::GpuRevoluteJointSolver;
pub use spherical::SphericalJoint;
pub use spherical_cpu::cpu_solve_joints_spherical;
pub use spherical_gpu::GpuSphericalJointSolver;
