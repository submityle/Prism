//! Real-device `wgpu` compute implementation of the 6-DOF rigid-body
//! integrator.
//!
//! [`GpuRigidIntegrator`] is the device twin of
//! [`cpu_integrate`](super::cpu_integrate): it uploads the body state, external
//! forces, and torques, dispatches a single compute pass in which each
//! invocation advances one body through every substep, and reads the updated
//! positions, orientations, and velocities back into the state. Because bodies
//! are integrated independently, the kernel needs neither colouring nor atomics
//! — one workgroup-per-64-bodies dispatch covers the whole set.
//!
//! The integrator compiles its own pipeline from its own shader
//! (`shaders/rigid_integrate.wgsl`) and owns its bind-group layout, mirroring
//! the crate convention of copying tiny setup rather than coupling features
//! through private internals.
//!
//! Provenance: Euler's rigid-body equations with explicit gyroscopic coupling
//! and the quaternion kinematic equation (Baraff & Witkin). Standard `wgpu`
//! compute dispatch. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

use super::body::RigidBodyState;
use super::config::{IntegratorConfig, RigidError};
use super::gyroscopic::{GyroscopicConfig, GyroscopicMode};

/// Global integrator parameters. Layout matches `Params` in
/// `shaders/rigid_integrate.wgsl` (48 bytes: padded so its size is a multiple
/// of the 16-byte `vec3` alignment the uniform binding requires).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    gravity: [f32; 3],
    h: f32,
    linear_damping_scale: f32,
    angular_damping_scale: f32,
    substeps: u32,
    body_count: u32,
    /// Gyroscopic integration scheme: `1` selects the implicit Newton solve,
    /// anything else the explicit subtraction. Mirrors [`GyroscopicMode`].
    gyroscopic_mode: u32,
    /// Newton iterations per substep when `gyroscopic_mode` is implicit.
    gyroscopic_iterations: u32,
    /// Padding to 48 bytes; the shader declares matching `pad0`/`pad1` fields.
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable `GPU` rigid-body integrator pipeline.
pub struct GpuRigidIntegrator {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    integrate: ComputePipeline,
}

impl GpuRigidIntegrator {
    /// Compiles the integrator kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRigidIntegrator {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_rigid_integrate"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rigid_integrate.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_rigid_integrate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
                buffer_entry(7, BufferBindingType::Storage { read_only: true }),
                buffer_entry(8, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_rigid_integrate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let integrate = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_rigid_integrate"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("integrate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRigidIntegrator {
            module,
            layout,
            integrate,
        }
    }

    /// Advances `state` by `dt` seconds under the per-body external `forces` and
    /// `torques` on device, reading the result back into `state`.
    ///
    /// Carries the identical semantics as the `CPU` twin
    /// [`cpu_integrate`](super::cpu_integrate): `forces` and `torques` may be
    /// empty (treated as all-zero) or hold exactly one world-space load per
    /// body.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InvalidConfig`] when `config` fails validation, or
    /// [`RigidError::InconsistentState`] when the state arrays disagree in
    /// length or a non-empty `forces`/`torques` slice does not match the body
    /// count. Does nothing (returns `Ok`) when there are no bodies or `dt` is
    /// non-positive.
    pub fn integrate(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        forces: &[Vec3],
        torques: &[Vec3],
        config: &IntegratorConfig,
        dt: f32,
    ) -> Result<(), RigidError> {
        self.integrate_inner(ctx, state, forces, torques, config, &GyroscopicConfig::explicit(), dt)
    }

    /// Advances `state` by `dt` seconds like [`integrate`](Self::integrate) but
    /// with the gyroscopic coupling of the angular update integrated according
    /// to `gyro`.
    ///
    /// With [`GyroscopicConfig::explicit`] this is identical to
    /// [`integrate`](Self::integrate). With [`GyroscopicConfig::implicit`] the
    /// angular update solves the backward-Euler gyroscopic equation with Newton
    /// iteration on device, matching the `CPU` twin
    /// [`cpu_integrate_gyro`](super::cpu_integrate_gyro). A body with any locked
    /// (zero-inertia) axis falls back to the explicit path on both backends so
    /// they stay in parity.
    ///
    /// # Errors
    ///
    /// Identical to [`integrate`](Self::integrate).
    pub fn integrate_with_gyroscopic(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        forces: &[Vec3],
        torques: &[Vec3],
        config: &IntegratorConfig,
        gyro: &GyroscopicConfig,
        dt: f32,
    ) -> Result<(), RigidError> {
        self.integrate_inner(ctx, state, forces, torques, config, gyro, dt)
    }

    /// Shared integrate path for the explicit and implicit gyroscopic entry
    /// points. Only the `gyro` configuration they pass differs; every other
    /// step — validation, substep sizing, upload, dispatch, readback — is
    /// identical.
    #[expect(
        clippy::too_many_arguments,
        reason = "shared integrate path threads every per-call input plus the gyroscopic config"
    )]
    fn integrate_inner(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        forces: &[Vec3],
        torques: &[Vec3],
        config: &IntegratorConfig,
        gyro: &GyroscopicConfig,
        dt: f32,
    ) -> Result<(), RigidError> {
        config.validate()?;
        if !state.is_consistent() {
            return Err(RigidError::InconsistentState {
                reason: "per-body arrays must have equal length",
            });
        }
        let n = state.len();
        if !forces.is_empty() && forces.len() != n {
            return Err(RigidError::InconsistentState {
                reason: "forces length must match body count or be empty",
            });
        }
        if !torques.is_empty() && torques.len() != n {
            return Err(RigidError::InconsistentState {
                reason: "torques length must match body count or be empty",
            });
        }
        if state.is_empty() || dt <= 0.0 {
            return Ok(());
        }

        let substeps = config.effective_substeps();
        let h = dt / substeps as f32;
        if h <= 0.0 {
            return Ok(());
        }

        let gyroscopic_mode = match gyro.mode {
            GyroscopicMode::Explicit => 0u32,
            GyroscopicMode::Implicit => 1u32,
        };
        let gyroscopic_iterations = gyro.effective_iterations();
        let plan = self.upload(
            ctx,
            state,
            forces,
            torques,
            config,
            h,
            substeps,
            gyroscopic_mode,
            gyroscopic_iterations,
        );
        let staging = self.encode_and_run(ctx, &plan);
        read_state_back(ctx, &staging, state);
        Ok(())
    }

    /// Uploads every buffer and builds the bind group for one `integrate` call.
    #[expect(
        clippy::too_many_arguments,
        reason = "uploads every per-call input plus the two gyroscopic uniform fields"
    )]
    fn upload(
        &self,
        ctx: &GpuContext,
        state: &RigidBodyState,
        forces: &[Vec3],
        torques: &[Vec3],
        config: &IntegratorConfig,
        h: f32,
        substeps: u32,
        gyroscopic_mode: u32,
        gyroscopic_iterations: u32,
    ) -> IntegratePlan {
        let device = ctx.device();
        let body_count = state.len() as u32;

        let params = Params {
            gravity: [config.gravity.x, config.gravity.y, config.gravity.z],
            h,
            linear_damping_scale: (1.0 - config.linear_damping * h).max(0.0),
            angular_damping_scale: (1.0 - config.angular_damping * h).max(0.0),
            substeps,
            body_count,
            gyroscopic_mode,
            gyroscopic_iterations,
            pad0: 0,
            pad1: 0,
        };

        let positions: Vec<[f32; 4]> = state.positions.iter().map(vec3_to_vec4).collect();
        let orientations: Vec<[f32; 4]> = state.orientations.iter().map(quat_to_vec4).collect();
        let linear_velocities: Vec<[f32; 4]> =
            state.linear_velocities.iter().map(vec3_to_vec4).collect();
        let angular_velocities: Vec<[f32; 4]> =
            state.angular_velocities.iter().map(vec3_to_vec4).collect();
        let inverse_inertias: Vec<[f32; 4]> =
            state.inverse_inertias.iter().map(vec3_to_vec4).collect();
        let force_upload = loads_to_vec4(forces, state.len());
        let torque_upload = loads_to_vec4(torques, state.len());

        let params_buf = buffer::uniform(device, "rigid_params", &params);
        let positions_buf = buffer::storage_rw_init(device, "rigid_positions", &positions);
        let orientations_buf = buffer::storage_rw_init(device, "rigid_orientations", &orientations);
        let linear_velocities_buf =
            buffer::storage_rw_init(device, "rigid_linear_velocities", &linear_velocities);
        let angular_velocities_buf =
            buffer::storage_rw_init(device, "rigid_angular_velocities", &angular_velocities);
        let inverse_masses_buf =
            buffer::storage_read(device, "rigid_inverse_masses", &state.inverse_masses);
        let inverse_inertias_buf =
            buffer::storage_read(device, "rigid_inverse_inertias", &inverse_inertias);
        let forces_buf = buffer::storage_read(device, "rigid_forces", &force_upload);
        let torques_buf = buffer::storage_read(device, "rigid_torques", &torque_upload);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_rigid_integrate_bind_group"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &orientations_buf),
                entry(3, &linear_velocities_buf),
                entry(4, &angular_velocities_buf),
                entry(5, &inverse_masses_buf),
                entry(6, &inverse_inertias_buf),
                entry(7, &forces_buf),
                entry(8, &torques_buf),
            ],
        });

        let vec4_bytes = (state.len() * 16) as u64;
        IntegratePlan {
            bind,
            positions_buf,
            orientations_buf,
            linear_velocities_buf,
            angular_velocities_buf,
            vec4_bytes,
            body_count,
        }
    }

    /// Records the single integrate pass plus the readback copies, submits the
    /// encoder, and returns the staging buffers the results were copied into.
    fn encode_and_run(&self, ctx: &GpuContext, plan: &IntegratePlan) -> IntegrateStaging {
        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_rigid_integrate_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_rigid_integrate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.integrate);
            pass.set_bind_group(0, &plan.bind, &[]);
            let groups = plan.body_count.div_ceil(64).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let positions_stage = buffer::staging(device, "rigid_positions_stage", plan.vec4_bytes);
        let orientations_stage =
            buffer::staging(device, "rigid_orientations_stage", plan.vec4_bytes);
        let linear_velocities_stage =
            buffer::staging(device, "rigid_linear_velocities_stage", plan.vec4_bytes);
        let angular_velocities_stage =
            buffer::staging(device, "rigid_angular_velocities_stage", plan.vec4_bytes);
        buffer::copy(
            &mut encoder,
            &plan.positions_buf,
            &positions_stage,
            plan.vec4_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.orientations_buf,
            &orientations_stage,
            plan.vec4_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.linear_velocities_buf,
            &linear_velocities_stage,
            plan.vec4_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.angular_velocities_buf,
            &angular_velocities_stage,
            plan.vec4_bytes,
        );
        ctx.queue().submit([encoder.finish()]);
        IntegrateStaging {
            positions: positions_stage,
            orientations: orientations_stage,
            linear_velocities: linear_velocities_stage,
            angular_velocities: angular_velocities_stage,
        }
    }
}

/// The uploaded buffers and dispatch dimensions for one `integrate` call.
struct IntegratePlan {
    bind: BindGroup,
    positions_buf: Buffer,
    orientations_buf: Buffer,
    linear_velocities_buf: Buffer,
    angular_velocities_buf: Buffer,
    vec4_bytes: u64,
    body_count: u32,
}

/// The staging buffers the device results were copied into for readback.
struct IntegrateStaging {
    positions: Buffer,
    orientations: Buffer,
    linear_velocities: Buffer,
    angular_velocities: Buffer,
}

/// Reads the integrated positions, orientations, and velocities back into
/// `state`.
fn read_state_back(ctx: &GpuContext, staging: &IntegrateStaging, state: &mut RigidBodyState) {
    let positions = buffer::read_back::<[f32; 4]>(ctx, &staging.positions);
    let orientations = buffer::read_back::<[f32; 4]>(ctx, &staging.orientations);
    let linear_velocities = buffer::read_back::<[f32; 4]>(ctx, &staging.linear_velocities);
    let angular_velocities = buffer::read_back::<[f32; 4]>(ctx, &staging.angular_velocities);
    for i in 0..state.len() {
        state.positions[i] = vec4_to_vec3(positions[i]);
        let q = orientations[i];
        state.orientations[i] = Quat::from_xyzw(q[0], q[1], q[2], q[3]);
        state.linear_velocities[i] = vec4_to_vec3(linear_velocities[i]);
        state.angular_velocities[i] = vec4_to_vec3(angular_velocities[i]);
    }
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

/// Expands a per-body load slice to padded `vec4` uploads, treating an empty
/// slice as all-zero.
fn loads_to_vec4(loads: &[Vec3], len: usize) -> Vec<[f32; 4]> {
    if loads.is_empty() {
        vec![[0.0; 4]; len]
    } else {
        loads.iter().map(vec3_to_vec4).collect()
    }
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
