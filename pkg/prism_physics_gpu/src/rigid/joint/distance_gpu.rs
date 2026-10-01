//! Real-device `wgpu` compute implementation of the distance (limit) rigid-body
//! joint stepper.
//!
//! [`GpuDistanceJointSolver`] is the device twin of
//! [`cpu_solve_joints_distance`](super::cpu_solve_joints_distance). Like its
//! `CPU` golden it is a *full stepper*: it owns the frame's integration. The
//! host colours the joint graph, uploads the bodies and the colour-reordered
//! joints, then runs the shared [`JointGpuCore`] dispatch skeleton — `snapshot`
//! / `predict` / `reset_lambda` / (`position_iterations` × colour-ordered
//! `solve`) / `recover` per substep — and reads the updated transforms and
//! velocities back into the caller's state.
//!
//! The solver is a thin wrapper: all of the device plumbing (the eleven-binding
//! group-0 layout, the per-batch group-1 uniform, the pipeline set, the upload,
//! and the readback) lives in [`JointGpuCore`], shared with every other joint
//! type. The distance joint contributes only its own shader
//! (`shaders/rigid_joint_distance.wgsl`), its packed [`GpuDistanceJoint`]
//! element, and its one-Lagrange-multiplier-per-joint count.
//!
//! Provenance: the point-to-point distance constraint and its substep `XPBD`
//! positional handling (Müller et al., "Detailed Rigid Body Simulation with
//! XPBD"), with the one-sided limit / dead-zone treatment standard to distance
//! joints, over the world-space inverse inertia and quaternion kinematics of
//! Baraff & Witkin. Standard `wgpu` compute dispatch. No Unreal Engine source or
//! derived code.

use crate::context::GpuContext;

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::distance::{DistanceJoint, GpuDistanceJoint};
use super::gpu_core::{JointGpuCore, Params};

/// Device-side distance-joint stepper. Owns the compiled shared joint core built
/// from the distance shader; one instance may solve many frames.
pub struct GpuDistanceJointSolver {
    /// The shared compiled joint-stepper pipeline set over the distance shader.
    core: JointGpuCore,
}

impl GpuDistanceJointSolver {
    /// Compiles the distance-joint stepper pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDistanceJointSolver {
        GpuDistanceJointSolver {
            core: JointGpuCore::new(
                ctx,
                include_str!("../../shaders/rigid_joint_distance.wgsl"),
                "prism_rigid_joint_distance",
            ),
        }
    }

    /// Advances `state` by `dt` under the distance joints in `joints` on the
    /// device, reproducing [`cpu_solve_joints_distance`](super::cpu_solve_joints_distance)
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
    pub fn solve_joints_distance(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        joints: &[DistanceJoint],
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

        // One Lagrange multiplier per distance joint (a single limit
        // constraint), matching the `CPU` golden's `lambda` buffer length.
        let joint_count = ordered.len() as u32;
        let gpu_joints: Vec<GpuDistanceJoint> = ordered.iter().map(|j| j.to_gpu()).collect();
        let params = Params::new(
            integrator.gravity,
            h,
            integrator.linear_damping,
            integrator.angular_damping,
            joint_count,
            state.len() as u32,
            joint_count,
        );

        self.core.run(
            ctx,
            state,
            &gpu_joints,
            joint_count,
            &colouring,
            &params,
            substeps,
            iterations,
        );
        Ok(())
    }
}
