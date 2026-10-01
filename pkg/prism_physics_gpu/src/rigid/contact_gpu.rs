//! Real-device `wgpu` compute implementation of the 6-DOF rigid-body contact
//! solver.
//!
//! [`GpuRigidContactSolver`] is the device twin of
//! [`cpu_solve_contacts`](super::cpu_solve_contacts): it colours the contact
//! graph on the host, uploads the bodies and the batch-grouped contacts,
//! dispatches the `prepare` / `warm_start` / `solve` passes in the identical
//! phase order as the `CPU` reference, and reads the updated velocities and the
//! converged accumulated impulses back into the caller's state and contacts.
//!
//! The colouring guarantees that the contacts in a batch write disjoint movable
//! bodies, so each batch is a single race-free dispatch; static bodies may be
//! shared across a batch because the shader skips their (zero) writes. The
//! solver compiles its own pipelines from its own shader
//! (`shaders/rigid_contact.wgsl`) and owns its bind-group layouts, mirroring the
//! crate convention of copying tiny setup rather than coupling features through
//! private internals.
//!
//! Provenance: velocity-level sequential impulses with a 2-D friction cone,
//! Baumgarte position bias, restitution, and warm starting (Catto / `Box2D`,
//! Bullet; standard constrained rigid-body dynamics). Standard `wgpu` compute
//! dispatch. No Unreal Engine source or derived code.

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

use super::body::RigidBodyState;
use super::config::{ContactSolverConfig, RigidError};
use super::contact::{GpuRigidContact, RigidContact};
use super::contact_coloring::RigidContactColouring;

/// Global solver parameters. Layout matches `Params` in
/// `shaders/rigid_contact.wgsl` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Baumgarte position-bias factor.
    baumgarte: f32,
    /// Penetration slop the bias leaves uncorrected.
    slop: f32,
    /// Approach speed below which restitution is suppressed.
    restitution_threshold: f32,
    /// Reciprocal time step, `1 / dt`.
    inv_dt: f32,
    /// Number of contacts in the batch-grouped buffer.
    contact_count: u32,
    /// Padding to a 16-byte boundary.
    _pad0: u32,
    /// Padding to a 16-byte boundary.
    _pad1: u32,
    /// Padding to a 16-byte boundary.
    _pad2: u32,
}

/// Per-batch dispatch parameters. Layout matches `ColourParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColourParams {
    /// First contact index of the batch within the ordered buffer.
    start: u32,
    /// Number of contacts in the batch.
    count: u32,
    /// Padding to a 16-byte boundary.
    _pad0: u32,
    /// Padding to a 16-byte boundary.
    _pad1: u32,
}

/// A compiled, reusable `GPU` rigid-body contact-solver pipeline set.
pub struct GpuRigidContactSolver {
    /// Owns the compiled shader so the pipelines built from it stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// Group-0 layout: params, velocities, body data, contacts, approach speed.
    global_layout: BindGroupLayout,
    /// Group-1 layout: the per-batch [`ColourParams`] uniform.
    colour_layout: BindGroupLayout,
    /// Captures each contact's pre-impulse approach speed.
    prepare: ComputePipeline,
    /// Re-applies each contact's accumulated impulse (one dispatch per batch).
    warm_start: ComputePipeline,
    /// Runs one sequential-impulse sweep (one dispatch per batch per iteration).
    solve: ComputePipeline,
}

impl GpuRigidContactSolver {
    /// Compiles the contact-solver kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRigidContactSolver {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_rigid_contact"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rigid_contact.wgsl").into()),
        });
        let global_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_rigid_contact_global_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
                buffer_entry(7, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let colour_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_rigid_contact_colour_layout"),
            entries: &[buffer_entry(0, BufferBindingType::Uniform)],
        });
        // `prepare` only touches group 0; `warm_start` and `solve` also read the
        // per-batch uniform in group 1.
        let global_only = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_rigid_contact_global_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout)],
            immediate_size: 0,
        });
        let with_colour = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_rigid_contact_colour_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout), Some(&colour_layout)],
            immediate_size: 0,
        });
        let prepare = make(
            device,
            &module,
            "prepare",
            "prism_rigid_contact_prepare",
            &global_only,
        );
        let warm_start = make(
            device,
            &module,
            "warm_start",
            "prism_rigid_contact_warm_start",
            &with_colour,
        );
        let solve = make(
            device,
            &module,
            "solve",
            "prism_rigid_contact_solve",
            &with_colour,
        );
        GpuRigidContactSolver {
            module,
            global_layout,
            colour_layout,
            prepare,
            warm_start,
            solve,
        }
    }

    /// Resolves `contacts` between the bodies of `state` on device, carrying the
    /// identical semantics as the `CPU` twin
    /// [`cpu_solve_contacts`](super::cpu_solve_contacts).
    ///
    /// Updates `state`'s linear and angular velocities in place and writes each
    /// contact's converged accumulated impulses back into `contacts` (in the
    /// caller's original order) so the next frame can warm-start from them.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InvalidConfig`] when `config` fails validation,
    /// [`RigidError::InconsistentState`] when the state arrays disagree in
    /// length or a contact indexes a missing body, or
    /// [`RigidError::TooManyContactBatches`] when the contact graph needs more
    /// batches than supported. Does nothing (returns `Ok`) when there are no
    /// bodies, no contacts, or `dt` is non-positive.
    pub fn solve_contacts(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        contacts: &mut [RigidContact],
        config: &ContactSolverConfig,
        dt: f32,
    ) -> Result<(), RigidError> {
        config.validate()?;
        if !state.is_consistent() {
            return Err(RigidError::InconsistentState {
                reason: "per-body arrays must have equal length",
            });
        }
        if state.is_empty() || contacts.is_empty() || dt <= 0.0 {
            return Ok(());
        }

        let movable = movable_mask(state);
        let colouring = RigidContactColouring::build(contacts, &movable)?;
        let ordered = colouring.reorder(contacts);
        let iterations = config.effective_iterations();

        let plan = self.upload(ctx, state, &ordered, config, 1.0 / dt);
        let staging = self.encode_and_run(ctx, &plan, &colouring, iterations);
        read_back(ctx, &staging, state, contacts, &colouring);
        Ok(())
    }

    /// Uploads all buffers and builds the group-0 bind group.
    fn upload(
        &self,
        ctx: &GpuContext,
        state: &RigidBodyState,
        ordered: &[RigidContact],
        config: &ContactSolverConfig,
        inv_dt: f32,
    ) -> SolvePlan {
        let device = ctx.device();
        let body_count = state.len();
        let contact_count = ordered.len() as u32;

        let params = Params {
            baumgarte: config.baumgarte,
            slop: config.slop,
            restitution_threshold: config.restitution_threshold,
            inv_dt,
            contact_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let linear: Vec<[f32; 4]> = state.linear_velocities.iter().map(vec3_to_vec4).collect();
        let angular: Vec<[f32; 4]> = state.angular_velocities.iter().map(vec3_to_vec4).collect();
        let inverse_inertias: Vec<[f32; 4]> =
            state.inverse_inertias.iter().map(vec3_to_vec4).collect();
        let orientations: Vec<[f32; 4]> = state.orientations.iter().map(quat_to_vec4).collect();
        let gpu_contacts: Vec<GpuRigidContact> = ordered.iter().map(|c| c.to_gpu()).collect();

        let params_buf = buffer::uniform(device, "rigid_contact_params", &params);
        let linear_buf = buffer::storage_rw_init(device, "rigid_contact_linear", &linear);
        let angular_buf = buffer::storage_rw_init(device, "rigid_contact_angular", &angular);
        let inverse_mass_buf = buffer::storage_read(
            device,
            "rigid_contact_inverse_masses",
            &state.inverse_masses,
        );
        let inverse_inertia_buf =
            buffer::storage_read(device, "rigid_contact_inverse_inertias", &inverse_inertias);
        let orientation_buf =
            buffer::storage_read(device, "rigid_contact_orientations", &orientations);
        let contact_buf = buffer::storage_rw_init(device, "rigid_contact_contacts", &gpu_contacts);
        let vn_initial_buf = buffer::storage_rw_zeroed(
            device,
            "rigid_contact_vn_initial",
            u64::from(contact_count) * 4,
        );

        let global_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_rigid_contact_global_bind_group"),
            layout: &self.global_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &linear_buf),
                entry(2, &angular_buf),
                entry(3, &inverse_mass_buf),
                entry(4, &inverse_inertia_buf),
                entry(5, &orientation_buf),
                entry(6, &contact_buf),
                entry(7, &vn_initial_buf),
            ],
        });

        let velocity_bytes = (body_count * 16) as u64;
        let contact_bytes = u64::from(contact_count) * CONTACT_BYTES;
        SolvePlan {
            global_bind,
            linear_buf,
            angular_buf,
            contact_buf,
            velocity_bytes,
            contact_bytes,
            contact_count,
        }
    }

    /// Records the `prepare` / `warm_start` / `solve` passes into one encoder in
    /// the same phase order as the `CPU` twin, submits it, and returns the
    /// staging buffers the results were copied into.
    fn encode_and_run(
        &self,
        ctx: &GpuContext,
        plan: &SolvePlan,
        colouring: &RigidContactColouring,
        iterations: u32,
    ) -> SolveStaging {
        let device = ctx.device();
        let colour_binds = self.colour_bind_groups(ctx, colouring);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_rigid_contact_encoder"),
        });

        // 1. Prepare: one thread per contact captures the approach speed.
        let contact_groups = plan.contact_count.div_ceil(64).max(1);
        self.pass(
            &mut encoder,
            "prepare",
            &self.prepare,
            plan,
            contact_groups,
            None,
        );

        // 2. Warm start: one dispatch per batch re-applies the accumulated
        //    impulse (same-batch contacts write disjoint movable bodies).
        for (c, &(start, end)) in colouring.ranges().iter().enumerate() {
            let groups = (end - start).div_ceil(64).max(1);
            self.pass(
                &mut encoder,
                "warm_start",
                &self.warm_start,
                plan,
                groups,
                Some(&colour_binds[c]),
            );
        }

        // 3. Iterate the sequential-impulse sweeps, one dispatch per batch.
        for _ in 0..iterations {
            for (c, &(start, end)) in colouring.ranges().iter().enumerate() {
                let groups = (end - start).div_ceil(64).max(1);
                self.pass(
                    &mut encoder,
                    "solve",
                    &self.solve,
                    plan,
                    groups,
                    Some(&colour_binds[c]),
                );
            }
        }

        let linear_stage =
            buffer::staging(device, "rigid_contact_linear_stage", plan.velocity_bytes);
        let angular_stage =
            buffer::staging(device, "rigid_contact_angular_stage", plan.velocity_bytes);
        let contact_stage =
            buffer::staging(device, "rigid_contact_contact_stage", plan.contact_bytes);
        buffer::copy(
            &mut encoder,
            &plan.linear_buf,
            &linear_stage,
            plan.velocity_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.angular_buf,
            &angular_stage,
            plan.velocity_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.contact_buf,
            &contact_stage,
            plan.contact_bytes,
        );
        ctx.queue().submit([encoder.finish()]);
        SolveStaging {
            linear: linear_stage,
            angular: angular_stage,
            contacts: contact_stage,
        }
    }

    /// Builds one uniform + bind group per batch describing its contact slice.
    fn colour_bind_groups(
        &self,
        ctx: &GpuContext,
        colouring: &RigidContactColouring,
    ) -> Vec<BindGroup> {
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
                let buf = buffer::uniform(device, "rigid_contact_colour_params", &params);
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("prism_rigid_contact_colour_bind_group"),
                    layout: &self.colour_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                })
            })
            .collect()
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

/// Byte size of one [`GpuRigidContact`] element in the device buffer.
const CONTACT_BYTES: u64 = size_of::<GpuRigidContact>() as u64;

/// The uploaded buffers and dispatch dimensions for one `solve_contacts` call.
struct SolvePlan {
    /// Group-0 bind group shared by every pass.
    global_bind: BindGroup,
    /// Linear velocities (read back into the state).
    linear_buf: Buffer,
    /// Angular velocities (read back into the state).
    angular_buf: Buffer,
    /// Batch-ordered contacts (read back for the accumulated impulses).
    contact_buf: Buffer,
    /// Byte length of each velocity buffer.
    velocity_bytes: u64,
    /// Byte length of the contact buffer.
    contact_bytes: u64,
    /// Number of contacts in the batch-grouped buffer.
    contact_count: u32,
}

/// The staging buffers the device results were copied into for readback.
struct SolveStaging {
    /// Linear velocities.
    linear: Buffer,
    /// Angular velocities.
    angular: Buffer,
    /// Batch-ordered contacts.
    contacts: Buffer,
}

/// Reads the solved velocities back into `state` and scatters each ordered
/// contact's converged accumulated impulses back into `contacts` in the
/// caller's original order.
fn read_back(
    ctx: &GpuContext,
    staging: &SolveStaging,
    state: &mut RigidBodyState,
    contacts: &mut [RigidContact],
    colouring: &RigidContactColouring,
) {
    let linear = buffer::read_back::<[f32; 4]>(ctx, &staging.linear);
    let angular = buffer::read_back::<[f32; 4]>(ctx, &staging.angular);
    for i in 0..state.len() {
        state.linear_velocities[i] = vec4_to_vec3(linear[i]);
        state.angular_velocities[i] = vec4_to_vec3(angular[i]);
    }
    let solved = buffer::read_back::<GpuRigidContact>(ctx, &staging.contacts);
    for (ordered_index, &original_index) in colouring.order().iter().enumerate() {
        let gpu = solved[ordered_index];
        let target = &mut contacts[original_index as usize];
        target.normal_impulse = gpu.normal_impulse;
        target.tangent_impulse_0 = gpu.tangent_impulse_0;
        target.tangent_impulse_1 = gpu.tangent_impulse_1;
    }
}

/// Returns a per-body flag that is `true` when the solver may write the body.
fn movable_mask(state: &RigidBodyState) -> Vec<bool> {
    (0..state.len())
        .map(|i| {
            let inv_i = state.inverse_inertias[i];
            state.inverse_masses[i] > 0.0 || inv_i.x > 0.0 || inv_i.y > 0.0 || inv_i.z > 0.0
        })
        .collect()
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
