//! Real-device `wgpu` compute implementation of the soft-constraint Temporal
//! Gauss-Seidel (`TGS`) 6-DOF rigid-body contact stepper.
//!
//! [`GpuRigidTgsContactSolver`] is the device twin of
//! [`cpu_solve_contacts_tgs`](super::cpu_solve_contacts_tgs). Like its `CPU`
//! golden it is a *full stepper*: it owns the frame's integration. The host
//! colours the contact graph, uploads the bodies, their frame-initial snapshot,
//! and the batch-grouped contacts, then unrolls the substep loop into the
//! identical dispatch sequence as the `CPU` reference —
//! `integrate_velocities` / `warm_start` / `relinearise` / `solve_biased` /
//! `integrate_positions` / `solve_relax` per substep, then one
//! `apply_restitution` pass — and reads the updated transforms, velocities, and
//! converged accumulated impulses back into the caller's state and contacts.
//!
//! Per-body passes (`integrate_velocities`, `integrate_positions`) and the
//! per-contact `relinearise` pass are global dispatches whose threads each
//! write only their own body or contact, so they carry no race. The
//! impulse-applying passes (`warm_start`, `solve_biased`, `solve_relax`,
//! `apply_restitution`) are dispatched once per colour batch; the colouring
//! guarantees same-batch contacts write disjoint movable bodies, and static
//! bodies may be shared because the shader skips their (zero) writes. The host
//! pre-computes the soft-constraint coefficients so the device never re-derives
//! them, matching the `CPU` twin's single `SoftParams::from_hertz` call.
//!
//! The solver compiles its own pipelines from its own shader
//! (`shaders/rigid_contact_tgs.wgsl`) and owns its bind-group layouts, mirroring
//! the crate convention of copying tiny setup rather than coupling features
//! through private internals.
//!
//! Provenance: the soft-constraint contact of Catto ("Soft Constraints", GDC
//! 2011) and the substepping solver loop it feeds (`Box2D` TGS Soft), over the
//! world-space inverse inertia and quaternion kinematics of Baraff & Witkin.
//! Standard `wgpu` compute dispatch. No Unreal Engine source or derived code.

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
use crate::xpbd::SoftParams;

use super::body::RigidBodyState;
use super::config::{IntegratorConfig, RigidError};
use super::contact::{GpuRigidContact, RigidContact};
use super::contact_coloring::RigidContactColouring;
use super::contact_tgs_config::TgsContactConfig;

/// Global solver parameters. Layout matches `Params` in
/// `shaders/rigid_contact_tgs.wgsl` (64 bytes, 16-byte aligned).
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
    /// Penetration slop the biased solve leaves uncorrected.
    slop: f32,
    /// Soft-constraint bias coefficient (`bias = bias_rate * (separation+slop)`).
    bias_rate: f32,
    /// Soft-constraint effective-mass scale, in `[0, 1]`.
    mass_scale: f32,
    /// Soft-constraint accumulated-impulse decay, in `[0, 1]`.
    impulse_scale: f32,
    /// Approach speed below which restitution is suppressed.
    restitution_threshold: f32,
    /// Number of (colour-reordered) contacts in the batch buffer.
    contact_count: u32,
    /// Number of bodies.
    body_count: u32,
    /// Padding to the 16-byte `std140` uniform stride.
    _pad0: u32,
    /// Padding to the 16-byte `std140` uniform stride.
    _pad1: u32,
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

/// A compiled, reusable `GPU` soft-constraint `TGS` rigid-body contact-stepper
/// pipeline set.
pub struct GpuRigidTgsContactSolver {
    /// Owns the compiled shader so the pipelines built from it stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// Group-0 layout: params, the bodies, the contacts, and the snapshot.
    global_layout: BindGroupLayout,
    /// Group-1 layout: the per-batch [`ColourParams`] uniform.
    colour_layout: BindGroupLayout,
    /// Integrates velocities under gravity and damping (one dispatch per body).
    integrate_velocities: ComputePipeline,
    /// Re-applies each contact's accumulated impulse (one dispatch per batch).
    warm_start: ComputePipeline,
    /// Re-linearises every contact against the moved geometry (per contact).
    relinearise: ComputePipeline,
    /// One biased Gauss-Seidel sweep (one dispatch per batch).
    solve_biased: ComputePipeline,
    /// Integrates positions and orientations (one dispatch per body).
    integrate_positions: ComputePipeline,
    /// One bias-free relaxation sweep (one dispatch per batch).
    solve_relax: ComputePipeline,
    /// Restores elastic bounce after the substep loop (one dispatch per batch).
    apply_restitution: ComputePipeline,
}

impl GpuRigidTgsContactSolver {
    /// Compiles the `TGS` contact-stepper kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRigidTgsContactSolver {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_rigid_contact_tgs"),
            source: ShaderSource::Wgsl(include_str!("../shaders/rigid_contact_tgs.wgsl").into()),
        });
        let global_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_rigid_contact_tgs_global_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
                buffer_entry(7, BufferBindingType::Storage { read_only: false }),
                buffer_entry(8, BufferBindingType::Storage { read_only: true }),
                buffer_entry(9, BufferBindingType::Storage { read_only: true }),
                buffer_entry(10, BufferBindingType::Storage { read_only: true }),
                buffer_entry(11, BufferBindingType::Storage { read_only: true }),
                buffer_entry(12, BufferBindingType::Storage { read_only: true }),
                buffer_entry(13, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let colour_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_rigid_contact_tgs_colour_layout"),
            entries: &[buffer_entry(0, BufferBindingType::Uniform)],
        });
        // The per-body and per-contact passes touch only group 0; the four
        // impulse-applying passes also read the per-batch uniform in group 1.
        let global_only = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_rigid_contact_tgs_global_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout)],
            immediate_size: 0,
        });
        let with_colour = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_rigid_contact_tgs_colour_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout), Some(&colour_layout)],
            immediate_size: 0,
        });
        let integrate_velocities = make(
            device,
            &module,
            "integrate_velocities",
            "prism_rigid_contact_tgs_integrate_velocities",
            &global_only,
        );
        let relinearise = make(
            device,
            &module,
            "relinearise",
            "prism_rigid_contact_tgs_relinearise",
            &global_only,
        );
        let integrate_positions = make(
            device,
            &module,
            "integrate_positions",
            "prism_rigid_contact_tgs_integrate_positions",
            &global_only,
        );
        let warm_start = make(
            device,
            &module,
            "warm_start",
            "prism_rigid_contact_tgs_warm_start",
            &with_colour,
        );
        let solve_biased = make(
            device,
            &module,
            "solve_biased",
            "prism_rigid_contact_tgs_solve_biased",
            &with_colour,
        );
        let solve_relax = make(
            device,
            &module,
            "solve_relax",
            "prism_rigid_contact_tgs_solve_relax",
            &with_colour,
        );
        let apply_restitution = make(
            device,
            &module,
            "apply_restitution",
            "prism_rigid_contact_tgs_apply_restitution",
            &with_colour,
        );
        GpuRigidTgsContactSolver {
            module,
            global_layout,
            colour_layout,
            integrate_velocities,
            warm_start,
            relinearise,
            solve_biased,
            integrate_positions,
            solve_relax,
            apply_restitution,
        }
    }

    /// Advances `state` by one frame of `dt` seconds on device, resolving
    /// `contacts` with the soft-constraint `TGS` stepper and integrating the
    /// bodies under gravity, carrying the identical semantics as the `CPU` twin
    /// [`cpu_solve_contacts_tgs`](super::cpu_solve_contacts_tgs).
    ///
    /// Because this routine integrates the bodies, the caller must **not** also
    /// integrate them in the same frame. The accumulated-impulse fields of each
    /// [`RigidContact`] are read as the warm-start seed and written with the
    /// converged solution.
    ///
    /// # Errors
    ///
    /// Returns [`RigidError::InvalidConfig`] when either configuration fails
    /// validation, [`RigidError::InconsistentState`] when the body arrays
    /// disagree in length or a contact indexes a missing body, or
    /// [`RigidError::TooManyContactBatches`] when the contact graph needs more
    /// parallel batches than supported. Does nothing (returns `Ok`) when there
    /// are no bodies or `dt` is non-positive; a body with no contacts is still
    /// integrated.
    pub fn solve_contacts_tgs(
        &self,
        ctx: &GpuContext,
        state: &mut RigidBodyState,
        contacts: &mut [RigidContact],
        integrator: &IntegratorConfig,
        tgs: &TgsContactConfig,
        dt: f32,
    ) -> Result<(), RigidError> {
        integrator.validate()?;
        tgs.validate()?;
        if !state.is_consistent() {
            return Err(RigidError::InconsistentState {
                reason: "per-body arrays must have equal length",
            });
        }
        if state.is_empty() || dt <= 0.0 {
            return Ok(());
        }

        let substeps = integrator.effective_substeps();
        let h = dt / substeps as f32;
        if h <= 0.0 {
            return Ok(());
        }

        let movable = movable_mask(state);
        let colouring = RigidContactColouring::build(contacts, &movable)?;
        let ordered = colouring.reorder(contacts);
        let relax_iterations = tgs.effective_relax_iterations();

        let plan = self.upload(ctx, state, &ordered, integrator, tgs, h, substeps);
        let staging = self.encode_and_run(ctx, &plan, &colouring, substeps, relax_iterations);
        read_back(ctx, &staging, state, contacts, &colouring);
        Ok(())
    }

    /// Uploads all buffers and builds the group-0 bind group. The host computes
    /// the per-substep soft-constraint coefficients and damping scales once so
    /// the device never re-derives them.
    #[expect(
        clippy::too_many_arguments,
        reason = "the stepper's uniforms are all independent scalars"
    )]
    fn upload(
        &self,
        ctx: &GpuContext,
        state: &RigidBodyState,
        ordered: &[RigidContact],
        integrator: &IntegratorConfig,
        tgs: &TgsContactConfig,
        h: f32,
        _substeps: u32,
    ) -> SolvePlan {
        let device = ctx.device();
        let body_count = state.len();
        let contact_count = ordered.len() as u32;

        let soft = SoftParams::from_hertz(tgs.contact_hertz, tgs.contact_damping_ratio, h);
        let params = Params {
            gravity: [integrator.gravity.x, integrator.gravity.y, integrator.gravity.z],
            h,
            inv_h: 1.0 / h,
            linear_damping_scale: (1.0 - integrator.linear_damping * h).max(0.0),
            angular_damping_scale: (1.0 - integrator.angular_damping * h).max(0.0),
            slop: tgs.slop,
            bias_rate: soft.bias_rate,
            mass_scale: soft.mass_scale,
            impulse_scale: soft.impulse_scale,
            restitution_threshold: tgs.restitution_threshold,
            contact_count,
            body_count: body_count as u32,
            _pad0: 0,
            _pad1: 0,
        };

        // Per-body uploads in original body order.
        let linear: Vec<[f32; 4]> = state.linear_velocities.iter().map(vec3_to_vec4).collect();
        let angular: Vec<[f32; 4]> = state.angular_velocities.iter().map(vec3_to_vec4).collect();
        let positions: Vec<[f32; 4]> = state.positions.iter().map(vec3_to_vec4).collect();
        let orientations: Vec<[f32; 4]> = state.orientations.iter().map(quat_to_vec4).collect();
        let inverse_inertias: Vec<[f32; 4]> =
            state.inverse_inertias.iter().map(vec3_to_vec4).collect();
        // Frame-initial transform snapshot, also in original body order.
        let initial_positions = positions.clone();
        let initial_orientations = orientations.clone();

        // Per-contact uploads in colour-reordered order, matching the device
        // contact buffer and the host `cpu_solve_contacts_tgs` snapshot.
        let gpu_contacts: Vec<GpuRigidContact> = ordered.iter().map(|c| c.to_gpu()).collect();
        let initial_arm_a: Vec<[f32; 4]> = ordered.iter().map(|c| vec3_to_vec4(&c.anchor_a)).collect();
        let initial_arm_b: Vec<[f32; 4]> = ordered.iter().map(|c| vec3_to_vec4(&c.anchor_b)).collect();
        let base_separation: Vec<f32> = ordered.iter().map(|c| -c.penetration).collect();
        let approach_speed: Vec<f32> = ordered
            .iter()
            .map(|c| relative_velocity(state, c).dot(c.normal))
            .collect();

        // The device needs non-empty storage bindings even when there are no
        // contacts; `contact_count == 0` makes every contact thread guard out.
        let contact_upload = non_empty(gpu_contacts, GpuRigidContact::zeroed());
        let arm_a_upload = non_empty(initial_arm_a, [0.0; 4]);
        let arm_b_upload = non_empty(initial_arm_b, [0.0; 4]);
        let base_sep_upload = non_empty(base_separation, 0.0);
        let approach_upload = non_empty(approach_speed, 0.0);

        let params_buf = buffer::uniform(device, "rigid_contact_tgs_params", &params);
        let linear_buf = buffer::storage_rw_init(device, "rigid_contact_tgs_linear", &linear);
        let angular_buf = buffer::storage_rw_init(device, "rigid_contact_tgs_angular", &angular);
        let positions_buf =
            buffer::storage_rw_init(device, "rigid_contact_tgs_positions", &positions);
        let orientations_buf =
            buffer::storage_rw_init(device, "rigid_contact_tgs_orientations", &orientations);
        let inverse_mass_buf =
            buffer::storage_read(device, "rigid_contact_tgs_inverse_masses", &state.inverse_masses);
        let inverse_inertia_buf = buffer::storage_read(
            device,
            "rigid_contact_tgs_inverse_inertias",
            &inverse_inertias,
        );
        let contact_buf =
            buffer::storage_rw_init(device, "rigid_contact_tgs_contacts", &contact_upload);
        let initial_pos_buf = buffer::storage_read(
            device,
            "rigid_contact_tgs_initial_positions",
            &initial_positions,
        );
        let initial_orient_buf = buffer::storage_read(
            device,
            "rigid_contact_tgs_initial_orientations",
            &initial_orientations,
        );
        let initial_arm_a_buf =
            buffer::storage_read(device, "rigid_contact_tgs_initial_arm_a", &arm_a_upload);
        let initial_arm_b_buf =
            buffer::storage_read(device, "rigid_contact_tgs_initial_arm_b", &arm_b_upload);
        let base_separation_buf = buffer::storage_read(
            device,
            "rigid_contact_tgs_base_separation",
            &base_sep_upload,
        );
        let approach_speed_buf =
            buffer::storage_read(device, "rigid_contact_tgs_approach_speed", &approach_upload);

        let global_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_rigid_contact_tgs_global_bind_group"),
            layout: &self.global_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &linear_buf),
                entry(2, &angular_buf),
                entry(3, &positions_buf),
                entry(4, &orientations_buf),
                entry(5, &inverse_mass_buf),
                entry(6, &inverse_inertia_buf),
                entry(7, &contact_buf),
                entry(8, &initial_pos_buf),
                entry(9, &initial_orient_buf),
                entry(10, &initial_arm_a_buf),
                entry(11, &initial_arm_b_buf),
                entry(12, &base_separation_buf),
                entry(13, &approach_speed_buf),
            ],
        });

        let body_bytes = (body_count * 16) as u64;
        let contact_bytes = u64::from(contact_count) * CONTACT_BYTES;
        SolvePlan {
            global_bind,
            linear_buf,
            angular_buf,
            positions_buf,
            orientations_buf,
            contact_buf,
            body_bytes,
            contact_bytes,
            body_count: body_count as u32,
            contact_count,
        }
    }

    /// Records every substep's passes into one encoder in the identical phase
    /// order as the `CPU` twin, appends the restitution pass, submits it, and
    /// returns the staging buffers the results were copied into.
    fn encode_and_run(
        &self,
        ctx: &GpuContext,
        plan: &SolvePlan,
        colouring: &RigidContactColouring,
        substeps: u32,
        relax_iterations: u32,
    ) -> SolveStaging {
        let device = ctx.device();
        let colour_binds = self.colour_bind_groups(ctx, colouring);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_rigid_contact_tgs_encoder"),
        });
        let body_groups = plan.body_count.div_ceil(64).max(1);
        let contact_groups = plan.contact_count.div_ceil(64).max(1);

        for _ in 0..substeps {
            // 1. Integrate velocities under gravity and damping.
            self.pass(
                &mut encoder,
                "integrate_velocities",
                &self.integrate_velocities,
                plan,
                body_groups,
                None,
            );
            // 2. Warm start: re-apply the accumulated impulse, one dispatch per
            //    batch over disjoint movable bodies.
            self.each_batch(&mut encoder, "warm_start", &self.warm_start, plan, colouring, &colour_binds);
            // 3. Re-linearise every contact against the moved geometry.
            self.pass(
                &mut encoder,
                "relinearise",
                &self.relinearise,
                plan,
                contact_groups,
                None,
            );
            // 4. One biased Gauss-Seidel sweep, per batch.
            self.each_batch(&mut encoder, "solve_biased", &self.solve_biased, plan, colouring, &colour_binds);
            // 5. Integrate positions and orientations.
            self.pass(
                &mut encoder,
                "integrate_positions",
                &self.integrate_positions,
                plan,
                body_groups,
                None,
            );
            // 6. Bias-free relaxation sweeps, per batch.
            for _ in 0..relax_iterations {
                self.each_batch(&mut encoder, "solve_relax", &self.solve_relax, plan, colouring, &colour_binds);
            }
        }

        // Final: restore elastic bounce, colour-ordered to match the CPU sweep.
        self.each_batch(
            &mut encoder,
            "apply_restitution",
            &self.apply_restitution,
            plan,
            colouring,
            &colour_binds,
        );

        let linear_stage =
            buffer::staging(device, "rigid_contact_tgs_linear_stage", plan.body_bytes);
        let angular_stage =
            buffer::staging(device, "rigid_contact_tgs_angular_stage", plan.body_bytes);
        let positions_stage =
            buffer::staging(device, "rigid_contact_tgs_positions_stage", plan.body_bytes);
        let orientations_stage =
            buffer::staging(device, "rigid_contact_tgs_orientations_stage", plan.body_bytes);
        let contact_stage = buffer::staging(
            device,
            "rigid_contact_tgs_contact_stage",
            plan.contact_bytes.max(CONTACT_BYTES),
        );
        buffer::copy(&mut encoder, &plan.linear_buf, &linear_stage, plan.body_bytes);
        buffer::copy(&mut encoder, &plan.angular_buf, &angular_stage, plan.body_bytes);
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
        if plan.contact_count > 0 {
            buffer::copy(
                &mut encoder,
                &plan.contact_buf,
                &contact_stage,
                plan.contact_bytes,
            );
        }
        ctx.queue().submit([encoder.finish()]);
        SolveStaging {
            linear: linear_stage,
            angular: angular_stage,
            positions: positions_stage,
            orientations: orientations_stage,
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
                let buf = buffer::uniform(device, "rigid_contact_tgs_colour_params", &params);
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("prism_rigid_contact_tgs_colour_bind_group"),
                    layout: &self.colour_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                })
            })
            .collect()
    }

    /// Records one `pipeline` dispatch per colour batch, each over the batch's
    /// contacts, so same-batch threads write disjoint movable bodies.
    fn each_batch(
        &self,
        encoder: &mut CommandEncoder,
        label: &str,
        pipeline: &ComputePipeline,
        plan: &SolvePlan,
        colouring: &RigidContactColouring,
        colour_binds: &[BindGroup],
    ) {
        for (c, &(start, end)) in colouring.ranges().iter().enumerate() {
            let groups = (end - start).div_ceil(64).max(1);
            self.pass(encoder, label, pipeline, plan, groups, Some(&colour_binds[c]));
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

/// Byte size of one [`GpuRigidContact`] element in the device buffer.
const CONTACT_BYTES: u64 = size_of::<GpuRigidContact>() as u64;

/// The uploaded buffers and dispatch dimensions for one `solve_contacts_tgs`
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
    /// Batch-ordered contacts (read back for the accumulated impulses).
    contact_buf: Buffer,
    /// Byte length of each per-body buffer.
    body_bytes: u64,
    /// Byte length of the contact buffer.
    contact_bytes: u64,
    /// Number of bodies.
    body_count: u32,
    /// Number of contacts in the batch-grouped buffer.
    contact_count: u32,
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
    /// Batch-ordered contacts.
    contacts: Buffer,
}

/// Reads the solved transforms and velocities back into `state` and scatters
/// each ordered contact's converged accumulated impulses back into `contacts`
/// in the caller's original order.
fn read_back(
    ctx: &GpuContext,
    staging: &SolveStaging,
    state: &mut RigidBodyState,
    contacts: &mut [RigidContact],
    colouring: &RigidContactColouring,
) {
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
    if contacts.is_empty() {
        return;
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

/// Relative velocity of the contact point on body `a` with respect to body `b`,
/// matching the `CPU` twin's `relative_velocity`, used to seed the pre-step
/// approach speed on the host.
fn relative_velocity(state: &RigidBodyState, c: &RigidContact) -> Vec3 {
    let a = c.body_a as usize;
    let b = c.body_b as usize;
    let va = state.linear_velocities[a] + state.angular_velocities[a].cross(c.anchor_a);
    let vb = state.linear_velocities[b] + state.angular_velocities[b].cross(c.anchor_b);
    va - vb
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

/// Returns `values` unchanged, or a single-element vector holding `fallback`
/// when `values` is empty, so a zero-length storage binding is never created.
fn non_empty<T>(values: Vec<T>, fallback: T) -> Vec<T> {
    if values.is_empty() {
        vec![fallback]
    } else {
        values
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
