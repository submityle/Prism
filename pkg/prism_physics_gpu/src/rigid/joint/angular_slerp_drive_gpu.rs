//! Real-device `wgpu` compute implementation of the angular `SLERP` drive
//! rigid-body joint stepper.
//!
//! [`GpuAngularSlerpDriveJointSolver`] is the device twin of
//! [`cpu_solve_joints_angular_slerp_drive`](super::cpu_solve_joints_angular_slerp_drive).
//! Like its `CPU` golden it is a *full stepper*: it owns the frame's
//! integration. The host colours the joint graph, uploads the bodies and the
//! colour-reordered joints, then runs the shared [`JointGpuCore`] dispatch
//! skeleton — `snapshot` / `predict` / `reset_lambda` /
//! (`position_iterations` × colour-ordered `solve`) / `recover` per substep —
//! and reads the updated transforms and velocities back into the caller's
//! state.
//!
//! The solver is a thin wrapper: all of the device plumbing (the eleven-binding
//! group-0 layout, the per-batch group-1 uniform, the pipeline set, the upload,
//! and the readback) lives in [`JointGpuCore`], shared with every other joint
//! type. The angular `SLERP` drive contributes only its own shader
//! (`shaders/rigid_joint_angular_slerp_drive.wgsl`), its packed
//! [`GpuAngularSlerpDriveJoint`] element, and its single-Lagrange-multiplier-per-joint
//! count — one geodesic angular drive, with no positional weld. The drive's
//! damping term reads the shared per-substep snapshot buffer
//! (`prev_orientations`), which the core already binds for the velocity
//! recovery.
//!
//! Provenance: the relative-orientation geodesic measurement and its substep
//! `XPBD` angular correction (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), with the bilateral compliant-and-damped drive and its
//! Macklin-style damping regularisation (Macklin et al., "XPBD: Position-Based
//! Simulation of Compliant Constrained Dynamics") and the box-limited
//! projected-Gauss-Seidel torque cap, over the world-space inverse inertia and
//! quaternion kinematics of Baraff & Witkin. Standard `wgpu` compute dispatch.
//! No Unreal Engine source or derived code.

use crate::context::GpuContext;

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::angular_slerp_drive::{AngularSlerpDriveJoint, GpuAngularSlerpDriveJoint};
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::gpu_core::{JointGpuCore, Params};

/// Device-side angular-`SLERP`-drive-joint stepper. Owns the compiled shared
/// joint core built from the angular-`SLERP`-drive shader; one instance may
/// solve many frames.
pub struct GpuAngularSlerpDriveJointSolver {
    /// The shared compiled joint-stepper pipeline set over the
    /// angular-`SLERP`-drive shader.
    core: JointGpuCore,
}

impl GpuAngularSlerpDriveJointSolver {
    /// Compiles the angular-`SLERP`-drive-joint stepper pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAngularSlerpDriveJointSolver {
        GpuAngularSlerpDriveJointSolver {
            core: JointGpuCore::new(
                ctx,
                include_str!("../../shaders/rigid_joint_angular_slerp_drive.wgsl"),
                "prism_rigid_joint_angular_slerp_drive",
            ),
        }
    }

    /// Advances `state` by `dt` under the angular `SLERP` drives in `joints` on
    /// the device, reproducing
    /// [`cpu_solve_joints_angular_slerp_drive`](super::cpu_solve_joints_angular_slerp_drive)
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
    pub fn solve_joints_angular_slerp_drive(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        joints: &[AngularSlerpDriveJoint],
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
        for joint in joints {
            let (a, b) = joint.bodies();
            if a as usize >= state.len() || b as usize >= state.len() {
                return Err(RigidError::InconsistentState {
                    reason: "joint references a body outside the state",
                });
            }
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

        // One Lagrange multiplier per angular-`SLERP`-drive joint — slot `j` is
        // the geodesic angular drive, with no positional weld — matching the
        // `CPU` golden's `lambda` buffer length of `joint_count`.
        let joint_count = ordered.len() as u32;
        let lambda_count = joint_count;
        let gpu_joints: Vec<GpuAngularSlerpDriveJoint> =
            ordered.iter().map(|j| j.to_gpu()).collect();
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
