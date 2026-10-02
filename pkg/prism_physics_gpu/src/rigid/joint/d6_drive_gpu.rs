//! Real-device `wgpu` compute implementation of the configurable `D6` joint
//! stepper *with per-axis drives*.
//!
//! [`GpuD6DriveJointSolver`] is the device twin of
//! [`cpu_solve_joints_d6_driven`](super::cpu_solve_joints_d6_driven). Like its
//! `CPU` golden it is a *full stepper*: it owns the frame's integration. The
//! host colours the joint graph, reorders both the joints and their matching
//! drive sets into the same colour order, packs each joint and its six drives
//! into one device element, uploads the bodies and that combined joint array,
//! then runs the shared [`JointGpuCore`] dispatch skeleton — `snapshot` /
//! `predict` / `reset_lambda` / (`position_iterations` × colour-ordered
//! `solve`) / `recover` per substep — and reads the updated transforms and
//! velocities back into the caller's state.
//!
//! # Packing the drives alongside the joint
//!
//! The shared [`JointGpuCore`] binds exactly one joint storage buffer, so the
//! driven twin carries its drives *inside* that buffer: each element is a
//! [`GpuD6DrivenJoint`], the `128`-byte [`GpuD6Joint`] followed by the
//! `96`-byte [`GpuD6DriveSet`], a `224`-byte block the shader's `Joint` struct
//! mirrors field for field. This keeps the device plumbing (the eleven-binding
//! group-0 layout, the per-batch group-1 uniform, the pipeline set, the upload,
//! and the readback) entirely shared with every other joint type while giving
//! the drive projections their own data without a second bind group.
//!
//! # Multiplier layout
//!
//! Each joint owns *twelve* multipliers: slots `12j + 0 .. 6` the six passive
//! axes (linear `x` / `y` / `z`, twist, swing1, swing2) laid out exactly as the
//! passive [`GpuD6JointSolver`](super::GpuD6JointSolver), and slots
//! `12j + 6 .. 12` the six drives in the same axis order. The host therefore
//! sizes the shared `lambda` buffer at `12 * joint_count` and the shader's
//! `solve` pass projects the six passive axes first and the six drives second,
//! matching the `CPU` golden's `passive`-then-`drive` sweep.
//!
//! Provenance: the per-axis spring-damper actuator of a general-purpose
//! constraint (Unreal Engine's `FConstraintDrive`, `PhysX`'s `PxD6JointDrive`),
//! realised as a compliant, velocity-damped `XPBD` constraint over the passive
//! `D6` constraints of Müller et al. and the world-space inverse inertia and
//! quaternion kinematics of Baraff & Witkin. Standard `wgpu` compute dispatch.
//! No Unreal Engine source or derived code: only the public spring-damper drive
//! semantics are mirrored.

use bytemuck::{Pod, Zeroable};

use crate::context::GpuContext;

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::d6::{D6Joint, GpuD6Joint};
use super::d6_drive::{D6DriveSet, GpuD6DriveSet};
use super::gpu_core::{JointGpuCore, Params};

/// One device joint element: the packed passive [`D6Joint`] immediately
/// followed by its packed [`D6DriveSet`]. The two blocks are each a multiple of
/// the `16`-byte storage alignment (`128` and `96` bytes), so the `224`-byte
/// pair has no interior or trailing padding and matches the `Joint` struct in
/// `shaders/rigid_joint_d6_drive.wgsl` field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuD6DrivenJoint {
    /// The passive `D6` joint, byte-identical to the undriven twin's element.
    joint: GpuD6Joint,
    /// The six per-axis drives in frame-axis order (linear `x` / `y` / `z`,
    /// twist, swing1, swing2).
    drives: GpuD6DriveSet,
}

/// Device-side driven-`D6`-joint stepper. Owns the compiled shared joint core
/// built from the driven `D6` shader; one instance may solve many frames.
pub struct GpuD6DriveJointSolver {
    /// The shared compiled joint-stepper pipeline set over the driven `D6`
    /// shader.
    core: JointGpuCore,
}

impl GpuD6DriveJointSolver {
    /// Compiles the driven-`D6`-joint stepper pipelines on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuD6DriveJointSolver {
        GpuD6DriveJointSolver {
            core: JointGpuCore::new(
                ctx,
                include_str!("../../shaders/rigid_joint_d6_drive.wgsl"),
                "prism_rigid_joint_d6_drive",
            ),
        }
    }

    /// Advances `state` by `dt` under the configurable `D6` joints in `joints`,
    /// each actuated by the matching drive set in `drives`, on the device,
    /// reproducing
    /// [`cpu_solve_joints_d6_driven`](super::cpu_solve_joints_d6_driven) frame
    /// for frame.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InconsistentState`] if the per-body arrays disagree
    /// in length, if `drives` and `joints` differ in length, or if a joint
    /// references a body outside the state, and
    /// [`RigidError::TooManyJointBatches`] if the joint graph needs more colour
    /// batches than the colouring supports. Returns `Ok(())` with the state
    /// untouched when there is nothing to do (`dt <= 0`, no bodies, or no
    /// joints).
    pub fn solve_joints_d6_driven(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        joints: &[D6Joint],
        drives: &[D6DriveSet],
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
        if drives.len() != joints.len() {
            return Err(RigidError::InconsistentState {
                reason: "each D6 joint needs exactly one drive set",
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
        let ordered_drives = colouring.reorder(drives);
        let iterations = joint_config.effective_position_iterations();

        // Twelve Lagrange multipliers per joint — slots `12j + 0..6` the six
        // passive axes (linear x/y/z, twist, swing1, swing2) and `12j + 6..12`
        // the six drives in the same axis order — matching the `CPU` golden's
        // `lambda` buffer length of `12 * joint_count`.
        let joint_count = ordered.len() as u32;
        let lambda_count = 12 * joint_count;
        let gpu_joints: Vec<GpuD6DrivenJoint> = ordered
            .iter()
            .zip(ordered_drives.iter())
            .map(|(joint, drive)| GpuD6DrivenJoint {
                joint: joint.to_gpu(),
                drives: drive.to_gpu(),
            })
            .collect();
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
