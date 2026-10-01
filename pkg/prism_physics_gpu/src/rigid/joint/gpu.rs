//! Real-device `wgpu` compute implementation of the spherical (ball-and-socket)
//! rigid-body joint stepper.
//!
//! [`GpuSphericalJointSolver`] is the device twin of
//! [`cpu_solve_joints_spherical`](super::cpu_solve_joints_spherical). Like its
//! `CPU` golden it is a *full stepper*: it owns the frame's integration. The
//! host colours the joint graph, uploads the bodies and the colour-reordered
//! joints, then unrolls the substep loop into the identical dispatch sequence as
//! the `CPU` reference — `snapshot` / `predict` / `reset_lambda` /
//! (`position_iterations` × colour-ordered `solve`) / `recover` per substep —
//! and reads the updated transforms and velocities back into the caller's state.
//!
//! The `snapshot`, `predict`, and `recover` passes are per-body global
//! dispatches whose threads each write only their own body, so they carry no
//! race; the `reset_lambda` pass is a per-joint global dispatch, likewise
//! race-free. The `solve` pass is dispatched once per colour batch; the
//! colouring guarantees same-batch joints write disjoint movable bodies, and
//! static bodies may be shared because their corrections scale by zero. Solving
//! the batches in sequence (a barrier between them) reproduces the `CPU`
//! Gauss-Seidel sweep over the identical reordered joint list.
//!
//! The host pre-computes the substep size, its reciprocal, and the per-substep
//! damping scales so the device never re-derives them, matching the `CPU` twin's
//! single computation. The solver compiles its own pipelines from its own shader
//! (`shaders/rigid_joint_spherical.wgsl`) and owns its bind-group layouts,
//! mirroring the crate convention of copying tiny setup rather than coupling
//! features through private internals.
//!
//! Provenance: the point-to-point (ball-socket) constraint and its substep
//! `XPBD` positional handling (Müller et al., "Detailed Rigid Body Simulation
//! with XPBD"), over the world-space inverse inertia and quaternion kinematics
//! of Baraff & Witkin. Standard `wgpu` compute dispatch. No Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoder,
    CommandEncoderDescriptor, ComputePass, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

use super::super::body::RigidBodyState;
use super::super::config::{IntegratorConfig, RigidError};
use super::super::contact_cpu::movable_mask;
use super::coloring::JointColouring;
use super::config::JointSolverConfig;
use super::spherical::{GpuSphericalJoint, SphericalJoint};

/// Global solver parameters. Layout matches `Params` in
/// `shaders/rigid_joint_spherical.wgsl` (`48` bytes, `16`-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Uniform acceleration applied each substep, typically gravity.
    gravity: [f32; 3],
    /// Substep size `dt / substeps`.
    h: f32,
    /// Reciprocal substep size, `1 / h`.
    inv_h: f32,
    /// Linear velocity retention per substep: `(1 - linear_damping * h).max(0)`.
    linear_damping_scale: f32,
    /// Angular velocity retention per substep.
    angular_damping_scale: f32,
    /// Number of (colour-reordered) joints in the joint buffer.
    joint_count: u32,
    /// Number of bodies.
    body_count: u32,
    /// Padding to the `16`-byte `std140` uniform stride.
    _pad0: u32,
    /// Padding to the `16`-byte `std140` uniform stride.
    _pad1: u32,
    /// Padding to the `16`-byte `std140` uniform stride.
    _pad2: u32,
}

/// Per-batch dispatch parameters. Layout matches `ColourParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColourParams {
    /// First joint index of the batch within the ordered buffer.
    start: u32,
    /// Number of joints in the batch.
    count: u32,
    /// Padding to a `16`-byte boundary.
    _pad0: u32,
    /// Padding to a `16`-byte boundary.
    _pad1: u32,
}

/// A compiled, reusable `GPU` spherical-joint stepper pipeline set.
pub struct GpuSphericalJointSolver {
    /// Owns the compiled shader so the pipelines built from it stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// Group-0 layout: the uniform params and the per-body and per-joint buffers.
    global_layout: BindGroupLayout,
    /// Group-1 layout: the per-batch [`ColourParams`] uniform.
    colour_layout: BindGroupLayout,
    /// Copies the transforms into the per-substep snapshot (one dispatch per body).
    snapshot: ComputePipeline,
    /// Predicts every body forward under gravity and damping (per body).
    predict: ComputePipeline,
    /// Resets each joint's `XPBD` multiplier (one dispatch per joint).
    reset_lambda: ComputePipeline,
    /// Projects one colour batch of joints for a single sweep (per batch).
    solve: ComputePipeline,
    /// Recovers velocities from the net substep motion (one dispatch per body).
    recover: ComputePipeline,
}

impl GpuSphericalJointSolver {
    /// Compiles the spherical-joint stepper kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphericalJointSolver {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_rigid_joint_spherical"),
            source: ShaderSource::Wgsl(
                include_str!("../../shaders/rigid_joint_spherical.wgsl").into(),
            ),
        });
        let global_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_rigid_joint_spherical_global_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
                buffer_entry(7, BufferBindingType::Storage { read_only: true }),
                buffer_entry(8, BufferBindingType::Storage { read_only: false }),
                buffer_entry(9, BufferBindingType::Storage { read_only: false }),
                buffer_entry(10, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let colour_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_rigid_joint_spherical_colour_layout"),
            entries: &[buffer_entry(0, BufferBindingType::Uniform)],
        });
        // The per-body and per-joint passes touch only group 0; the per-batch
        // `solve` pass also reads the per-batch uniform in group 1.
        let global_only = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_rigid_joint_spherical_global_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout)],
            immediate_size: 0,
        });
        let with_colour = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_rigid_joint_spherical_colour_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout), Some(&colour_layout)],
            immediate_size: 0,
        });
        let snapshot = make(
            device,
            &module,
            "snapshot",
            "prism_rigid_joint_spherical_snapshot",
            &global_only,
        );
        let predict = make(
            device,
            &module,
            "predict",
            "prism_rigid_joint_spherical_predict",
            &global_only,
        );
        let reset_lambda = make(
            device,
            &module,
            "reset_lambda",
            "prism_rigid_joint_spherical_reset_lambda",
            &global_only,
        );
        let recover = make(
            device,
            &module,
            "recover",
            "prism_rigid_joint_spherical_recover",
            &global_only,
        );
        let solve = make(
            device,
            &module,
            "solve",
            "prism_rigid_joint_spherical_solve",
            &with_colour,
        );
        GpuSphericalJointSolver {
            module,
            global_layout,
            colour_layout,
            snapshot,
            predict,
            reset_lambda,
            solve,
            recover,
        }
    }

    /// Advances `state` by `dt` under the spherical joints in `joints` on the
    /// device, reproducing [`cpu_solve_joints_spherical`](super::cpu_solve_joints_spherical)
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
    pub fn solve_joints_spherical(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        joints: &[SphericalJoint],
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

        let plan = self.upload(ctx, state, &ordered, integrator, h);
        let staging = self.encode_and_run(ctx, &plan, &colouring, substeps, iterations);
        read_back(ctx, &staging, state);
        Ok(())
    }

    /// Uploads all buffers and builds the group-0 bind group. The host computes
    /// the substep size, its reciprocal, and the per-substep damping scales once
    /// so the device never re-derives them, matching the `CPU` twin.
    fn upload(
        &self,
        ctx: &GpuContext,
        state: &RigidBodyState,
        ordered: &[SphericalJoint],
        integrator: &IntegratorConfig,
        h: f32,
    ) -> SolvePlan {
        let device = ctx.device();
        let body_count = state.len();
        let joint_count = ordered.len() as u32;
        let inv_h = 1.0 / h;

        let params = Params {
            gravity: [
                integrator.gravity.x,
                integrator.gravity.y,
                integrator.gravity.z,
            ],
            h,
            inv_h,
            linear_damping_scale: (1.0 - integrator.linear_damping * h).max(0.0),
            angular_damping_scale: (1.0 - integrator.angular_damping * h).max(0.0),
            joint_count,
            body_count: body_count as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        // Per-body uploads in original body order.
        let linear: Vec<[f32; 4]> = state.linear_velocities.iter().map(vec3_to_vec4).collect();
        let angular: Vec<[f32; 4]> = state.angular_velocities.iter().map(vec3_to_vec4).collect();
        let positions: Vec<[f32; 4]> = state.positions.iter().map(vec3_to_vec4).collect();
        let orientations: Vec<[f32; 4]> = state.orientations.iter().map(quat_to_vec4).collect();
        let inverse_inertias: Vec<[f32; 4]> =
            state.inverse_inertias.iter().map(vec3_to_vec4).collect();

        // The snapshot pass overwrites these each substep before they are read,
        // so their initial contents are irrelevant; they only need the per-body
        // length. Seeding them with the current transforms keeps the upload
        // simple and avoids a zeroed-buffer helper.
        let prev_positions = positions.clone();
        let prev_orientations = orientations.clone();

        // Per-joint uploads in colour-reordered order, matching the device joint
        // buffer and the host `cpu_solve_joints_spherical` reordered list.
        let gpu_joints: Vec<GpuSphericalJoint> = ordered.iter().map(|j| j.to_gpu()).collect();
        // One Lagrange multiplier per joint, reset to zero each substep by the
        // `reset_lambda` pass.
        let lambda = vec![0.0f32; ordered.len()];

        let params_buf = buffer::uniform(device, "rigid_joint_spherical_params", &params);
        let linear_buf = buffer::storage_rw_init(device, "rigid_joint_spherical_linear", &linear);
        let angular_buf =
            buffer::storage_rw_init(device, "rigid_joint_spherical_angular", &angular);
        let positions_buf =
            buffer::storage_rw_init(device, "rigid_joint_spherical_positions", &positions);
        let orientations_buf =
            buffer::storage_rw_init(device, "rigid_joint_spherical_orientations", &orientations);
        let inverse_mass_buf = buffer::storage_read(
            device,
            "rigid_joint_spherical_inverse_masses",
            &state.inverse_masses,
        );
        let inverse_inertia_buf = buffer::storage_read(
            device,
            "rigid_joint_spherical_inverse_inertias",
            &inverse_inertias,
        );
        let joints_buf = buffer::storage_read(device, "rigid_joint_spherical_joints", &gpu_joints);
        let lambda_buf = buffer::storage_rw_init(device, "rigid_joint_spherical_lambda", &lambda);
        let prev_positions_buf = buffer::storage_rw_init(
            device,
            "rigid_joint_spherical_prev_positions",
            &prev_positions,
        );
        let prev_orientations_buf = buffer::storage_rw_init(
            device,
            "rigid_joint_spherical_prev_orientations",
            &prev_orientations,
        );

        let global_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_rigid_joint_spherical_global_bind_group"),
            layout: &self.global_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &linear_buf),
                entry(2, &angular_buf),
                entry(3, &positions_buf),
                entry(4, &orientations_buf),
                entry(5, &inverse_mass_buf),
                entry(6, &inverse_inertia_buf),
                entry(7, &joints_buf),
                entry(8, &lambda_buf),
                entry(9, &prev_positions_buf),
                entry(10, &prev_orientations_buf),
            ],
        });

        let body_bytes = (body_count * 16) as u64;
        SolvePlan {
            global_bind,
            linear_buf,
            angular_buf,
            positions_buf,
            orientations_buf,
            body_bytes,
            body_count: body_count as u32,
            joint_count,
        }
    }

    /// Records every substep's passes into one encoder in the identical phase
    /// order as the `CPU` twin, submits it, and returns the staging buffers the
    /// results were copied into.
    fn encode_and_run(
        &self,
        ctx: &GpuContext,
        plan: &SolvePlan,
        colouring: &JointColouring,
        substeps: u32,
        iterations: u32,
    ) -> SolveStaging {
        let device = ctx.device();
        let colour_binds = self.colour_bind_groups(ctx, colouring);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_rigid_joint_spherical_encoder"),
        });
        let body_groups = plan.body_count.div_ceil(64).max(1);
        let joint_groups = plan.joint_count.div_ceil(64).max(1);

        for _ in 0..substeps {
            // 1. Snapshot the transforms the velocity recovery differences against.
            self.pass(
                &mut encoder,
                "snapshot",
                &self.snapshot,
                plan,
                body_groups,
                None,
            );
            // 2. Predict every body forward under gravity and damping.
            self.pass(
                &mut encoder,
                "predict",
                &self.predict,
                plan,
                body_groups,
                None,
            );
            // 3. Reset each joint's XPBD multiplier.
            self.pass(
                &mut encoder,
                "reset_lambda",
                &self.reset_lambda,
                plan,
                joint_groups,
                None,
            );
            // 4. Project the joint constraints in colour-batch order, repeated
            //    `iterations` times, reproducing the CPU Gauss-Seidel sweeps.
            for _ in 0..iterations {
                self.each_batch(&mut encoder, &self.solve, plan, colouring, &colour_binds);
            }
            // 5. Recover the velocities from the net substep motion.
            self.pass(
                &mut encoder,
                "recover",
                &self.recover,
                plan,
                body_groups,
                None,
            );
        }

        let linear_stage = buffer::staging(
            device,
            "rigid_joint_spherical_linear_stage",
            plan.body_bytes,
        );
        let angular_stage = buffer::staging(
            device,
            "rigid_joint_spherical_angular_stage",
            plan.body_bytes,
        );
        let positions_stage = buffer::staging(
            device,
            "rigid_joint_spherical_positions_stage",
            plan.body_bytes,
        );
        let orientations_stage = buffer::staging(
            device,
            "rigid_joint_spherical_orientations_stage",
            plan.body_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.linear_buf,
            &linear_stage,
            plan.body_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.angular_buf,
            &angular_stage,
            plan.body_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.positions_buf,
            &positions_stage,
            plan.body_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.orientations_buf,
            &orientations_stage,
            plan.body_bytes,
        );
        ctx.queue().submit([encoder.finish()]);
        SolveStaging {
            linear: linear_stage,
            angular: angular_stage,
            positions: positions_stage,
            orientations: orientations_stage,
        }
    }

    /// Builds one uniform + bind group per batch describing its joint slice.
    fn colour_bind_groups(&self, ctx: &GpuContext, colouring: &JointColouring) -> Vec<BindGroup> {
        let device = ctx.device();
        colouring
            .ranges()
            .iter()
            .map(|&(start, end)| {
                let params = ColourParams {
                    start,
                    count: end - start,
                    _pad0: 0,
                    _pad1: 0,
                };
                let buf = buffer::uniform(device, "rigid_joint_spherical_colour_params", &params);
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("prism_rigid_joint_spherical_colour_bind_group"),
                    layout: &self.colour_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                })
            })
            .collect()
    }

    /// Records one `solve` dispatch per colour batch, each over the batch's
    /// joints, so same-batch threads write disjoint movable bodies.
    fn each_batch(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: &ComputePipeline,
        plan: &SolvePlan,
        colouring: &JointColouring,
        colour_binds: &[BindGroup],
    ) {
        for (c, &(start, end)) in colouring.ranges().iter().enumerate() {
            let groups = (end - start).div_ceil(64).max(1);
            self.pass(
                encoder,
                "solve",
                pipeline,
                plan,
                groups,
                Some(&colour_binds[c]),
            );
        }
    }

    /// Records one compute pass dispatching `groups` workgroups.
    fn pass(
        &self,
        encoder: &mut CommandEncoder,
        label: &str,
        pipeline: &ComputePipeline,
        plan: &SolvePlan,
        groups: u32,
        colour_bind: Option<&BindGroup>,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &plan.global_bind, &[]);
        if let Some(bind) = colour_bind {
            pass.set_bind_group(1, bind, &[]);
        }
        dispatch(&mut pass, groups);
    }
}

/// The uploaded buffers and dispatch dimensions for one `solve_joints_spherical`
/// call.
struct SolvePlan {
    /// Group-0 bind group shared by every pass.
    global_bind: BindGroup,
    /// Linear velocities (read back into the state).
    linear_buf: Buffer,
    /// Angular velocities (read back into the state).
    angular_buf: Buffer,
    /// Body positions (read back into the state).
    positions_buf: Buffer,
    /// Body orientations (read back into the state).
    orientations_buf: Buffer,
    /// Byte length of each per-body buffer.
    body_bytes: u64,
    /// Number of bodies.
    body_count: u32,
    /// Number of joints in the colour-ordered buffer.
    joint_count: u32,
}

/// The staging buffers the device results were copied into for readback.
struct SolveStaging {
    /// Linear velocities.
    linear: Buffer,
    /// Angular velocities.
    angular: Buffer,
    /// Body positions.
    positions: Buffer,
    /// Body orientations.
    orientations: Buffer,
}

/// Reads the solved transforms and velocities back into `state`. The joints need
/// no readback: their Lagrange multipliers are reset each substep and never
/// persisted.
fn read_back(ctx: &GpuContext, staging: &SolveStaging, state: &mut RigidBodyState) {
    let linear = buffer::read_back::<[f32; 4]>(ctx, &staging.linear);
    let angular = buffer::read_back::<[f32; 4]>(ctx, &staging.angular);
    let positions = buffer::read_back::<[f32; 4]>(ctx, &staging.positions);
    let orientations = buffer::read_back::<[f32; 4]>(ctx, &staging.orientations);
    for i in 0..state.len() {
        state.linear_velocities[i] = vec4_to_vec3(linear[i]);
        state.angular_velocities[i] = vec4_to_vec3(angular[i]);
        state.positions[i] = vec4_to_vec3(positions[i]);
        state.orientations[i] = vec4_to_quat(orientations[i]);
    }
}

/// Compiles one compute pipeline for `entry` under `layout`.
fn make(
    device: &wgpu::Device,
    module: &ShaderModule,
    entry: &str,
    label: &str,
    layout: &PipelineLayout,
) -> ComputePipeline {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Packs a [`Vec3`] into a padded `vec4` upload element.
fn vec3_to_vec4(v: &Vec3) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}

/// Packs a [`Quat`] into a `vec4` upload element as `(x, y, z, w)`.
fn quat_to_vec4(q: &Quat) -> [f32; 4] {
    [q.x, q.y, q.z, q.w]
}

/// Unpacks the `xyz` of a `vec4` readback element.
fn vec4_to_vec3(v: [f32; 4]) -> Vec3 {
    Vec3::new(v[0], v[1], v[2])
}

/// Unpacks a `(x, y, z, w)` readback element into a [`Quat`].
fn vec4_to_quat(v: [f32; 4]) -> Quat {
    Quat::from_xyzw(v[0], v[1], v[2], v[3])
}

/// Dispatches `groups` workgroups on `pass`.
fn dispatch(pass: &mut ComputePass<'_>, groups: u32) {
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Builds a bind-group entry binding `buffer` to `binding`.
fn entry(binding: u32, buffer: &Buffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
