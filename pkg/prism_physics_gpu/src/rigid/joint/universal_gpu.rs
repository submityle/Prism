//! Real-device `wgpu` compute implementation of the universal (Cardan) rigid-body
//! joint stepper.
//!
//! [`GpuUniversalJointSolver`] is the device twin of
//! [`cpu_solve_joints_universal`](super::cpu_solve_joints_universal). Like its
//! `CPU` golden it is a *full stepper*: it owns the frame's integration. The host
//! colours the joint graph, uploads the bodies and the colour-reordered joints,
//! then runs the shared [`JointGpuCore`] dispatch skeleton — `snapshot` /
//! `predict` / `reset_lambda` / (`position_iterations` × colour-ordered `solve`)
//! / `recover` per substep — and reads the updated transforms and velocities
//! back into the caller's state.
//!
//! The solver is a thin wrapper: all of the device plumbing (the eleven-binding
//! group-0 layout, the per-batch group-1 uniform, the pipeline set, the upload,
//! and the readback) lives in [`JointGpuCore`], shared with every other joint
//! type. The universal joint contributes only its own shader
//! (`shaders/rigid_joint_universal.wgsl`), its packed [`GpuUniversalJoint`]
//! element, and its two-Lagrange-multipliers-per-joint count (one positional
//! weld, one perpendicularity constraint).
//!
//! Provenance: the point-to-point (ball-socket) constraint and the
//! perpendicularity (orthogonality) constraint over two body-fixed axes with
//! their substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. Standard `wgpu` compute dispatch. No Unreal Engine source or
//! derived code.

use crate::context::GpuContext;

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::gpu_core::{JointGpuCore, Params};
use super::universal::{GpuUniversalJoint, UniversalJoint};

/// Device-side universal-joint stepper. Owns the compiled shared joint core
/// built from the universal shader; one instance may solve many frames.
pub struct GpuUniversalJointSolver {
    /// The shared compiled joint-stepper pipeline set over the universal shader.
    core: JointGpuCore,
}

impl GpuUniversalJointSolver {
    /// Compiles the universal-joint stepper pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuUniversalJointSolver {
        GpuUniversalJointSolver {
            core: JointGpuCore::new(
                ctx,
                include_str!("../../shaders/rigid_joint_universal.wgsl"),
                "prism_rigid_joint_universal",
            ),
        }
    }

    /// Advances `state` by `dt` under the universal joints in `joints` on the
    /// device, reproducing
    /// [`cpu_solve_joints_universal`](super::cpu_solve_joints_universal) frame
    /// for frame.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InconsistentState`] if the per-body arrays disagree
    /// in length or a joint references a body outside the state, and
    /// [`RigidError::TooManyJointBatches`] if the joint graph needs more colour
    /// batches than the colouring supports. Returns `Ok(())` with the state
    /// untouched when there is nothing to do (`dt <= 0`, no bodies, or no
    /// joints).
    pub fn solve_joints_universal(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        joints: &[UniversalJoint],
        integrator: &IntegratorConfig,
        joint_config: &JointSolverConfig,
        dt: f32,
    ) -> Result<(), RigidError> {
        integrator.validate()?;
        joint_config.validate()?;
        if !state.is_consistent() {
            return Err(RigidError::InconsistentState {
                reason: "per-body arrays must have equal length",
            });
        }
        if state.is_empty() || joints.is_empty() || dt <= 0.0 {
            return Ok(());
        }

        let substeps = integrator.effective_substeps();
        let h = dt / substeps as f32;
        if h <= 0.0 {
            return Ok(());
        }

        let movable = movable_mask(state);
        let colouring = JointColouring::build(joints, &movable)?;
        let ordered = colouring.reorder(joints);
        let iterations = joint_config.effective_position_iterations();

        // Two Lagrange multipliers per universal joint — slot `2j` positional
        // weld, slot `2j + 1` perpendicularity — matching the `CPU` golden's
        // `lambda` buffer length of `2 * joint_count`.
        let joint_count = ordered.len() as u32;
        let lambda_count = 2 * joint_count;
        let gpu_joints: Vec<GpuUniversalJoint> = ordered.iter().map(|j| j.to_gpu()).collect();
        let params = Params::new(
            integrator.gravity,
            h,
            integrator.linear_damping,
            integrator.angular_damping,
            joint_count,
            state.len() as u32,
            lambda_count,
        );

        self.core.run(
            ctx,
            state,
            &gpu_joints,
            lambda_count,
            &colouring,
            &params,
            substeps,
            iterations,
        );
        Ok(())
    }
}
