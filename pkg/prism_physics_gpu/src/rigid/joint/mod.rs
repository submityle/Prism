//! Position-based (`XPBD`) rigid-body joints.
//!
//! A joint couples two bodies in the shared [`RigidBodyState`](super::RigidBodyState)
//! by removing some of their relative degrees of freedom. This module starts
//! with the [`SphericalJoint`] (ball-and-socket): the point-to-point positional
//! weld that is the base of every articulated mechanism — a hinge, a slider, and
//! a configurable `6`-DOF joint are each this weld plus one or more angular
//! restrictions. The later joint types layer on top of the same stepper.
//!
//! # Layout
//!
//! * [`spherical`] — the [`SphericalJoint`] definition and its device-packed
//!   storage representation.
//! * [`config`] — the [`JointSolverConfig`] tunables (the per-substep
//!   projection-sweep count).
//! * [`coloring`] — the static-aware [`JointColouring`] that partitions a joint
//!   set into race-free parallel batches.
//! * [`cpu`] — the authoritative [`cpu_solve_joints_spherical`] golden stepper.
//! * [`gpu`] — the device-side [`GpuSphericalJointSolver`] that reproduces the
//!   golden stepper on real `wgpu` hardware.
//!
//! # Scheme and scope
//!
//! The stepper is the substep-`XPBD` scheme of Müller et al., "Detailed Rigid
//! Body Simulation with XPBD": each substep predicts the bodies forward, resets
//! the constraints' Lagrange multipliers, projects the positional constraints in
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
mod cpu;
mod gpu;
mod spherical;

pub use coloring::{JointColouring, MAX_JOINT_BATCHES};
pub use config::JointSolverConfig;
pub use cpu::cpu_solve_joints_spherical;
pub use gpu::GpuSphericalJointSolver;
pub use spherical::SphericalJoint;
