//! Real-device `wgpu` compute implementation of the colour-ordered `XPBD`
//! distance solver.
//!
//! [`GpuXpbdSolver`] compiles `shaders/xpbd.wgsl` once and exposes
//! [`GpuXpbdSolver::solve`], which uploads the particle state and the
//! colour-reordered constraints, then encodes the full frame — for each
//! substep: `predict`, `reset_lambdas`, `iterations` sweeps of one `project`
//! pass per colour, then `finalize` — as a chain of compute passes in a single
//! submission. Separate passes give the implicit memory barrier that makes each
//! colour's position writes visible to the next colour, so the device
//! reproduces the sequential Gauss-Seidel sweep of the [`cpu_solve`](
//! super::cpu_solve) twin.
//!
//! Provenance: substep `XPBD` (Müller et al.); standard `wgpu` compute
//! dispatch. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoder,
    CommandEncoderDescriptor, ComputePass, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

use super::coloring::Colouring;
use super::config::{XpbdConfig, XpbdError};
use super::constraint::{DistanceConstraint, GpuConstraint};
use super::state::ParticleState;

/// Global solver parameters. Layout matches `Params` in `shaders/xpbd.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    gravity: [f32; 3],
    h: f32,
    damping_scale: f32,
    inv_h: f32,
    particle_count: u32,
    constraint_count: u32,
}

/// Per-colour dispatch parameters. Layout matches `ColourParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColourParams {
    start: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// A compiled, reusable `GPU` `XPBD` distance-solver pipeline set.
pub struct GpuXpbdSolver {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    global_layout: BindGroupLayout,
    colour_layout: BindGroupLayout,
    predict: ComputePipeline,
    reset_lambdas: ComputePipeline,
    project: ComputePipeline,
    finalize: ComputePipeline,
}

impl GpuXpbdSolver {
    /// Compiles the solver kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuXpbdSolver {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_xpbd"),
            source: ShaderSource::Wgsl(include_str!("../shaders/xpbd.wgsl").into()),
        });
        let global_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_xpbd_global_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let colour_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_xpbd_colour_layout"),
            entries: &[buffer_entry(0, BufferBindingType::Uniform)],
        });
        // `predict` / `reset_lambdas` / `finalize` only touch group 0; `project`
        // also reads the per-colour uniform in group 1.
        let global_only = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_xpbd_global_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout)],
            immediate_size: 0,
        });
        let with_colour = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_xpbd_colour_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout), Some(&colour_layout)],
            immediate_size: 0,
        });
        let predict = make(
            device,
            &module,
            "predict",
            "prism_xpbd_predict",
            &global_only,
        );
        let reset_lambdas = make(
            device,
            &module,
            "reset_lambdas",
            "prism_xpbd_reset_lambdas",
            &global_only,
        );
        let project = make(
            device,
            &module,
            "project",
            "prism_xpbd_project",
            &with_colour,
        );
        let finalize = make(
            device,
            &module,
            "finalize",
            "prism_xpbd_finalize",
            &global_only,
        );
        GpuXpbdSolver {
            module,
            global_layout,
            colour_layout,
            predict,
            reset_lambdas,
            project,
            finalize,
        }
    }

    /// Advances `state` under `constraints` by `dt` seconds on device.
    ///
    /// Mutates `state` in place with the positions and velocities read back from
    /// the device; state carries forward to the next call exactly as the `CPU`
    /// twin's does.
    ///
    /// # Errors
    ///
    /// Returns [`XpbdError`] when the config or state is invalid, a constraint
    /// indexes a missing particle, or the constraint graph needs more colours
    /// than supported. Does nothing (returns `Ok`) when there are no particles
    /// or `dt` is non-positive.
    pub fn solve(
        &self,
        ctx: &GpuContext,
        state: &mut ParticleState,
        constraints: &[DistanceConstraint],
        config: &XpbdConfig,
        dt: f32,
    ) -> Result<(), XpbdError> {
        config.validate()?;
        if !state.is_consistent() {
            return Err(XpbdError::InvalidConfig(
                "particle state arrays must have equal length",
            ));
        }
        if state.is_empty() || dt <= 0.0 {
            return Ok(());
        }

        let particle_count = state.len() as u32;
        let colouring = Colouring::build(constraints, particle_count)?;
        let ordered = colouring.reorder(constraints);

        let substeps = config.effective_substeps();
        let iterations = config.effective_iterations();
        let h = dt / substeps as f32;
        if h <= 0.0 {
            return Ok(());
        }
        let plan = self.upload(ctx, state, &ordered, config, h);
        let staging = self.encode_and_run(ctx, &plan, &colouring, substeps, iterations);
        read_state_back(ctx, &staging, state);
        Ok(())
    }

    /// Uploads all buffers and builds the global bind group.
    fn upload(
        &self,
        ctx: &GpuContext,
        state: &ParticleState,
        ordered: &[DistanceConstraint],
        config: &XpbdConfig,
        h: f32,
    ) -> SolvePlan {
        let device = ctx.device();
        let particle_count = state.len() as u32;
        let constraint_count = ordered.len() as u32;

        let params = Params {
            gravity: [config.gravity.x, config.gravity.y, config.gravity.z],
            h,
            damping_scale: (1.0 - config.damping * h).max(0.0),
            inv_h: 1.0 / h,
            particle_count,
            constraint_count,
        };

        let positions: Vec<[f32; 4]> = state.positions.iter().map(vec3_to_vec4).collect();
        let velocities: Vec<[f32; 4]> = state.velocities.iter().map(vec3_to_vec4).collect();
        let prev = vec![[0.0f32; 4]; state.len()];
        let gpu_constraints: Vec<GpuConstraint> = ordered.iter().map(|c| c.to_gpu()).collect();
        let constraint_upload = if gpu_constraints.is_empty() {
            vec![GpuConstraint::zeroed()]
        } else {
            gpu_constraints
        };

        let params_buf = buffer::uniform(device, "xpbd_params", &params);
        let positions_buf = buffer::storage_rw_init(device, "xpbd_positions", &positions);
        let velocities_buf = buffer::storage_rw_init(device, "xpbd_velocities", &velocities);
        let prev_buf = buffer::storage_rw_init(device, "xpbd_prev", &prev);
        let inverse_mass_buf =
            buffer::storage_read(device, "xpbd_inverse_masses", &state.inverse_masses);
        let constraint_buf = buffer::storage_read(device, "xpbd_constraints", &constraint_upload);
        let lambda_buf = buffer::storage_rw_zeroed(
            device,
            "xpbd_lambdas",
            u64::from(constraint_count.max(1)) * 4,
        );

        let global_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_xpbd_global_bind_group"),
            layout: &self.global_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &velocities_buf),
                entry(3, &prev_buf),
                entry(4, &inverse_mass_buf),
                entry(5, &constraint_buf),
                entry(6, &lambda_buf),
            ],
        });

        let position_bytes = (state.len() * 16) as u64;
        SolvePlan {
            global_bind,
            positions_buf,
            velocities_buf,
            position_bytes,
            particle_count,
            constraint_count,
        }
    }

    /// Records every substep's passes into one encoder, submits it, and returns
    /// the staging buffers the results were copied into.
    fn encode_and_run(
        &self,
        ctx: &GpuContext,
        plan: &SolvePlan,
        colouring: &Colouring,
        substeps: u32,
        iterations: u32,
    ) -> Staging {
        let device = ctx.device();
        let colour_binds = self.colour_bind_groups(ctx, colouring);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_xpbd_encoder"),
        });
        let particle_groups = plan.particle_count.div_ceil(64).max(1);
        let constraint_groups = plan.constraint_count.div_ceil(64).max(1);

        for _ in 0..substeps {
            self.pass(
                &mut encoder,
                "predict",
                &self.predict,
                plan,
                particle_groups,
                None,
            );
            self.pass(
                &mut encoder,
                "reset_lambdas",
                &self.reset_lambdas,
                plan,
                constraint_groups,
                None,
            );
            for _ in 0..iterations {
                for (c, &(start, end)) in colouring.ranges().iter().enumerate() {
                    let groups = (end - start).div_ceil(64).max(1);
                    self.pass(
                        &mut encoder,
                        "project",
                        &self.project,
                        plan,
                        groups,
                        Some(&colour_binds[c]),
                    );
                }
            }
            self.pass(
                &mut encoder,
                "finalize",
                &self.finalize,
                plan,
                particle_groups,
                None,
            );
        }

        let position_stage = buffer::staging(device, "xpbd_position_stage", plan.position_bytes);
        let velocity_stage = buffer::staging(device, "xpbd_velocity_stage", plan.position_bytes);
        buffer::copy(
            &mut encoder,
            &plan.positions_buf,
            &position_stage,
            plan.position_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.velocities_buf,
            &velocity_stage,
            plan.position_bytes,
        );
        ctx.queue().submit([encoder.finish()]);
        Staging {
            positions: position_stage,
            velocities: velocity_stage,
        }
    }

    /// Builds one uniform + bind group per colour describing its constraint
    /// slice. `wgpu` retains the uniform buffer through the bind group, so the
    /// local buffer handle can be dropped at the end of each iteration.
    fn colour_bind_groups(&self, ctx: &GpuContext, colouring: &Colouring) -> Vec<BindGroup> {
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
                let buf = buffer::uniform(device, "xpbd_colour_params", &params);
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("prism_xpbd_colour_bind_group"),
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

/// The uploaded buffers and dispatch dimensions for one `solve` call.
struct SolvePlan {
    global_bind: BindGroup,
    positions_buf: Buffer,
    velocities_buf: Buffer,
    position_bytes: u64,
    particle_count: u32,
    constraint_count: u32,
}

/// The staging buffers the device results were copied into for readback.
struct Staging {
    positions: Buffer,
    velocities: Buffer,
}

/// Reads positions and velocities back into `state`.
fn read_state_back(ctx: &GpuContext, staging: &Staging, state: &mut ParticleState) {
    let positions = buffer::read_back::<[f32; 4]>(ctx, &staging.positions);
    let velocities = buffer::read_back::<[f32; 4]>(ctx, &staging.velocities);
    for i in 0..state.len() {
        state.positions[i] = vec4_to_vec3(positions[i]);
        state.velocities[i] = vec4_to_vec3(velocities[i]);
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

/// Packs a [`glam::Vec3`] into a padded `vec4` upload element.
fn vec3_to_vec4(v: &glam::Vec3) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}

/// Unpacks the `xyz` of a `vec4` readback element.
fn vec4_to_vec3(v: [f32; 4]) -> glam::Vec3 {
    glam::Vec3::new(v[0], v[1], v[2])
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
