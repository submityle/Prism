//! Real-device `wgpu` compute implementation of the gear (angular ratio
//! coupling) rigid-body joint stepper.
//!
//! [`GpuGearJointSolver`] is the device twin of
//! [`cpu_solve_joints_gear`](super::cpu_solve_joints_gear). Like its `CPU`
//! golden it is a *full stepper*: it owns the frame's integration. The host
//! colours the joint graph, uploads the bodies and the colour-reordered joints,
//! then runs the shared [`JointGpuCore`] dispatch skeleton — `snapshot` /
//! `predict` / `reset_lambda` / (`position_iterations` × colour-ordered
//! `solve`) / `recover` per substep — and reads the updated transforms and
//! velocities back into the caller's state.
//!
//! The solver is a thin wrapper: all of the device plumbing (the eleven-binding
//! group-0 layout, the per-batch group-1 uniform, the pipeline set, the upload,
//! and the readback) lives in [`JointGpuCore`], shared with every other joint
//! type. The gear joint contributes only its own shader
//! (`shaders/rigid_joint_gear.wgsl`), its packed [`GpuGearJoint`] element, and
//! its single-Lagrange-multiplier-per-joint count (one ratio coupling). The
//! coupling's relative-displacement term reads the shared per-substep snapshot
//! buffer (`prev_orientations`), which the core already binds for the velocity
//! recovery.
//!
//! Provenance: the angular ratio coupling expressed as a per-substep compliant
//! equality on the relative angular displacement about the two gear axes
//! (Macklin et al., "XPBD: Position-Based Simulation of Compliant Constrained
//! Dynamics"), over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. Standard `wgpu` compute dispatch. No Unreal Engine source or
//! derived code.

use crate::context::GpuContext;

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::gear::{GearJoint, GpuGearJoint};
use super::gpu_core::{JointGpuCore, Params};

/// Device-side gear-joint stepper. Owns the compiled shared joint core built
/// from the gear shader; one instance may solve many frames.
pub struct GpuGearJointSolver {
    /// The shared compiled joint-stepper pipeline set over the gear shader.
    core: JointGpuCore,
}

impl GpuGearJointSolver {
    /// Compiles the gear-joint stepper pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGearJointSolver {
        GpuGearJointSolver {
            core: JointGpuCore::new(
                ctx,
                include_str!("../../shaders/rigid_joint_gear.wgsl"),
                "prism_rigid_joint_gear",
            ),
        }
    }

    /// Advances `state` by `dt` under the gear joints in `joints` on the device,
    /// reproducing [`cpu_solve_joints_gear`](super::cpu_solve_joints_gear) frame
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
    pub fn solve_joints_gear(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        joints: &[GearJoint],
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

        // One Lagrange multiplier per gear joint — slot `j` is the ratio
        // coupling — matching the `CPU` golden's `lambda` buffer length of
        // `joint_count`.
        let joint_count = ordered.len() as u32;
        let lambda_count = joint_count;
        let gpu_joints: Vec<GpuGearJoint> = ordered.iter().map(|j| j.to_gpu()).collect();
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
