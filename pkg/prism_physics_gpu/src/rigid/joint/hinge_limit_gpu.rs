//! Real-device `wgpu` compute implementation of the hinge angular-limit
//! rigid-body joint stepper.
//!
//! [`GpuHingeLimitJointSolver`] is the device twin of
//! [`cpu_solve_joints_hinge_limit`](super::cpu_solve_joints_hinge_limit). Like
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
//! type. The hinge-limit joint contributes only its own shader
//! (`shaders/rigid_joint_hinge_limit.wgsl`), its packed [`GpuHingeLimitJoint`]
//! element, and its three-Lagrange-multipliers-per-joint count (one positional
//! weld, one axis alignment, one angular limit).
//!
//! Provenance: the point-to-point (ball-socket) constraint, the hinge
//! axis-alignment constraint, and the one-sided angular limit with their substep
//! `XPBD` handling (Müller et al., "Detailed Rigid Body Simulation with XPBD"),
//! over the world-space inverse inertia and quaternion kinematics of Baraff &
//! Witkin. Standard `wgpu` compute dispatch. No Unreal Engine source or derived
//! code.

use crate::context::GpuContext;

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::gpu_core::{JointGpuCore, Params};
use super::hinge_limit::{GpuHingeLimitJoint, HingeLimitJoint};

/// Device-side hinge-limit-joint stepper. Owns the compiled shared joint core
/// built from the hinge-limit shader; one instance may solve many frames.
pub struct GpuHingeLimitJointSolver {
    /// The shared compiled joint-stepper pipeline set over the hinge-limit
    /// shader.
    core: JointGpuCore,
}

impl GpuHingeLimitJointSolver {
    /// Compiles the hinge-limit-joint stepper pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHingeLimitJointSolver {
        GpuHingeLimitJointSolver {
            core: JointGpuCore::new(
                ctx,
                include_str!("../../shaders/rigid_joint_hinge_limit.wgsl"),
                "prism_rigid_joint_hinge_limit",
            ),
        }
    }

    /// Advances `state` by `dt` under the hinge-limit joints in `joints` on the
    /// device, reproducing
    /// [`cpu_solve_joints_hinge_limit`](super::cpu_solve_joints_hinge_limit)
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
    pub fn solve_joints_hinge_limit(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        joints: &[HingeLimitJoint],
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

        // Three Lagrange multipliers per hinge-limit joint — slot `3j`
        // positional weld, slot `3j + 1` axis alignment, slot `3j + 2` angular
        // limit — matching the `CPU` golden's `lambda` buffer length of
        // `3 * joint_count`.
        let joint_count = ordered.len() as u32;
        let lambda_count = 3 * joint_count;
        let gpu_joints: Vec<GpuHingeLimitJoint> = ordered.iter().map(|j| j.to_gpu()).collect();
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
