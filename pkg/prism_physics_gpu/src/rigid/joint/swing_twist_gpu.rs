//! Real-device `wgpu` compute implementation of the swing-twist (cone-twist)
//! rigid-body joint stepper.
//!
//! [`GpuSwingTwistJointSolver`] is the device twin of
//! [`cpu_solve_joints_swing_twist`](super::cpu_solve_joints_swing_twist). Like
//! its `CPU` golden it is a *full stepper*: it owns the frame's integration. The
//! host colours the joint graph, uploads the bodies and the colour-reordered
//! joints, then runs the shared [`JointGpuCore`] dispatch skeleton — `snapshot`
//! / `predict` / `reset_lambda` / (`position_iterations` × colour-ordered
//! `solve`) / `recover` per substep — and reads the updated transforms and
//! velocities back into the caller's state.
//!
//! The solver is a thin wrapper: all of the device plumbing (the eleven-binding
//! group-0 layout, the per-batch group-1 uniform, the pipeline set, the upload,
//! and the readback) lives in [`JointGpuCore`], shared with every other joint
//! type. The swing-twist joint contributes only its own shader
//! (`shaders/rigid_joint_swing_twist.wgsl`), its packed [`GpuSwingTwistJoint`]
//! element, and its three-Lagrange-multipliers-per-joint count (one
//! point-to-point weld, one swing cone, one twist limit).
//!
//! Provenance: the point-to-point (ball-socket) constraint, the signed angular
//! limit shared with the hinge limit, and the cone-swing limit with their
//! substep `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation with
//! XPBD"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. Standard `wgpu` compute dispatch. No Unreal Engine source or
//! derived code.

use crate::context::GpuContext;

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::gpu_core::{JointGpuCore, Params};
use super::swing_twist::{GpuSwingTwistJoint, SwingTwistJoint};

/// Device-side swing-twist-joint stepper. Owns the compiled shared joint core
/// built from the swing-twist shader; one instance may solve many frames.
pub struct GpuSwingTwistJointSolver {
    /// The shared compiled joint-stepper pipeline set over the swing-twist
    /// shader.
    core: JointGpuCore,
}

impl GpuSwingTwistJointSolver {
    /// Compiles the swing-twist-joint stepper pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSwingTwistJointSolver {
        GpuSwingTwistJointSolver {
            core: JointGpuCore::new(
                ctx,
                include_str!("../../shaders/rigid_joint_swing_twist.wgsl"),
                "prism_rigid_joint_swing_twist",
            ),
        }
    }

    /// Advances `state` by `dt` under the swing-twist joints in `joints` on the
    /// device, reproducing
    /// [`cpu_solve_joints_swing_twist`](super::cpu_solve_joints_swing_twist)
    /// frame for frame.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InconsistentState`] if the per-body arrays disagree
    /// in length or a joint references a body outside the state, and
    /// [`RigidError::TooManyJointBatches`] if the joint graph needs more colour
    /// batches than the colouring supports. Returns `Ok(())` with the state
    /// untouched when there is nothing to do (`dt <= 0`, no bodies, or no
    /// joints).
    pub fn solve_joints_swing_twist(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        joints: &[SwingTwistJoint],
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

        // Three Lagrange multipliers per swing-twist joint — slot `3j`
        // point-to-point weld, slot `3j + 1` swing cone, slot `3j + 2` twist
        // limit — matching the `CPU` golden's `lambda` buffer length of
        // `3 * joint_count`.
        let joint_count = ordered.len() as u32;
        let lambda_count = 3 * joint_count;
        let gpu_joints: Vec<GpuSwingTwistJoint> = ordered.iter().map(|j| j.to_gpu()).collect();
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
