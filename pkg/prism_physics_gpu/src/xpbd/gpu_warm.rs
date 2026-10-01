//! Real-device `wgpu` compute implementation of the *warm-started* colour-ordered
//! `XPBD` distance solver.
//!
//! [`GpuXpbdWarmSolver`] is the device twin of
//! [`cpu_solve_warm`](super::cpu_solve_warm): it carries a
//! [`DistanceCache`](super::DistanceCache) across frames, seeds each distance
//! constraint's Lagrange multiplier from the previous frame's converged
//! solution, applies the matching position correction up front, solves, and
//! writes the converged multipliers back. Warm-starting lets a loaded structure
//! settle in far fewer iterations than the cold
//! [`GpuXpbdSolver`](super::GpuXpbdSolver), which always starts from rest.
//!
//! The warm solver differs from the cold one in exactly three places, each a
//! byte-for-byte-intent twin of the `CPU` reference:
//!
//! 1. an extra read-only `seed` buffer (binding 8) holds the per-constraint
//!    warm-start multiplier produced once per frame by
//!    [`DistanceCache::seed`](super::DistanceCache::seed);
//! 2. each substep runs one `apply_warm_start` pass per colour — replacing the
//!    cold path's single `reset_lambdas` dispatch — which seeds the multiplier
//!    and applies the cached correction before the projection sweeps; and
//! 3. after the final substep the device multiplier buffer is read back and
//!    handed to [`DistanceCache::store`](super::DistanceCache::store) so the next
//!    frame seeds from it.
//!
//! This solver is intentionally decoupled from [`GpuXpbdSolver`]: it compiles
//! its own pipelines from its own shader (`shaders/xpbd_warm.wgsl`) and owns its
//! own bind-group layouts, mirroring the crate convention of copying tiny setup
//! rather than coupling two features through private internals.
//!
//! Provenance: substep `XPBD` (Müller et al.) with the canonical stretch
//! constraint and the warm-starting of an iterative constraint solver (Müller
//! et al. substep `XPBD`). Standard `wgpu` compute dispatch. No Unreal Engine
//! source or derived code.

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
use super::warm_start::DistanceCache;

/// Global solver parameters. Layout matches `Params` in
/// `shaders/xpbd_warm.wgsl`.
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

/// A compiled, reusable warm-started `GPU` `XPBD` distance-solver pipeline set.
pub struct GpuXpbdWarmSolver {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    global_layout: BindGroupLayout,
    colour_layout: BindGroupLayout,
    predict: ComputePipeline,
    apply_warm_start: ComputePipeline,
    project: ComputePipeline,
    finalize: ComputePipeline,
}

impl GpuXpbdWarmSolver {
    /// Compiles the warm-start solver kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuXpbdWarmSolver {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_xpbd_warm"),
            source: ShaderSource::Wgsl(include_str!("../shaders/xpbd_warm.wgsl").into()),
        });
        // Nine bindings: the eight the cold solver uses plus the read-only
        // warm-start `seed` (binding 8). Entries that never read `seed`
        // (`predict`, `project`, `finalize`) run under this same layout — a
        // layout may declare bindings an entry does not use.
        let global_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_xpbd_warm_global_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
                buffer_entry(7, BufferBindingType::Storage { read_only: true }),
                buffer_entry(8, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let colour_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_xpbd_warm_colour_layout"),
            entries: &[buffer_entry(0, BufferBindingType::Uniform)],
        });
        // `predict` / `finalize` only touch group 0; `apply_warm_start` and
        // `project` also read the per-colour uniform in group 1.
        let global_only = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_xpbd_warm_global_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout)],
            immediate_size: 0,
        });
        let with_colour = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_xpbd_warm_colour_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout), Some(&colour_layout)],
            immediate_size: 0,
        });
        let predict = make(
            device,
            &module,
            "predict",
            "prism_xpbd_warm_predict",
            &global_only,
        );
        let apply_warm_start = make(
            device,
            &module,
            "apply_warm_start",
            "prism_xpbd_warm_apply_warm_start",
            &with_colour,
        );
        let project = make(
            device,
            &module,
            "project",
            "prism_xpbd_warm_project",
            &with_colour,
        );
        let finalize = make(
            device,
            &module,
            "finalize",
            "prism_xpbd_warm_finalize",
            &global_only,
        );
        GpuXpbdWarmSolver {
            module,
            global_layout,
            colour_layout,
            predict,
            apply_warm_start,
            project,
            finalize,
        }
    }

    /// Advances `state` under `constraints` by `dt` seconds on device,
    /// warm-starting each constraint from `cache` and re-storing the converged
    /// multipliers into it.
    ///
    /// Mutates `state` in place with the positions and velocities read back from
    /// the device, and rebuilds `cache` from this frame's constraints so the
    /// next call seeds from the solution computed here. State and cache carry
    /// forward exactly as the `CPU` twin
    /// [`cpu_solve_warm`](super::cpu_solve_warm) does.
    ///
    /// # Errors
    ///
    /// Returns [`XpbdError`] when the config or state is invalid, a constraint
    /// indexes a missing particle, or the constraint graph needs more colours
    /// than supported. Does nothing (returns `Ok`) when there are no particles
    /// or `dt` is non-positive.
    pub fn solve_warm(
        &self,
        ctx: &GpuContext,
        state: &mut ParticleState,
        constraints: &[DistanceConstraint],
        config: &XpbdConfig,
        dt: f32,
        cache: &mut DistanceCache,
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

        // Read the per-constraint seed once (the cache does not change mid-frame)
        // and align it with `ordered`, matching the CPU twin.
        let seed = cache.seed(&ordered);
        let plan = self.upload(ctx, state, &ordered, &seed, config, h);
        let staging = self.encode_and_run(ctx, &plan, &colouring, substeps, iterations);
        let lambda = read_state_back(ctx, &staging, state, plan.constraint_count);
        // Persist the converged multipliers (aligned with `ordered`) so next
        // frame seeds from them; pairs absent this frame are pruned by the
        // rebuild inside `store`.
        cache.store(&ordered, &lambda);
        Ok(())
    }

    /// Uploads all buffers (including the warm-start `seed`) and builds the
    /// global bind group.
    fn upload(
        &self,
        ctx: &GpuContext,
        state: &ParticleState,
        ordered: &[DistanceConstraint],
        seed: &[f32],
        config: &XpbdConfig,
        h: f32,
    ) -> WarmSolvePlan {
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
        // A zero-length storage binding is invalid, so a constraint-free frame
        // uploads one dummy element the shader's bounds guard never dispatches.
        let constraint_upload = if gpu_constraints.is_empty() {
            vec![GpuConstraint::zeroed()]
        } else {
            gpu_constraints
        };
        let seed_upload = if seed.is_empty() {
            vec![0.0f32]
        } else {
            seed.to_vec()
        };

        // The dense warm solve integrates every particle, so the awake mask is
        // all ones; an all-ones mask makes the shader's awake branch a no-op,
        // preserving bit-identical behaviour with the CPU twin's all-true mask.
        let awake = vec![1u32; state.len()];

        let params_buf = buffer::uniform(device, "xpbd_warm_params", &params);
        let positions_buf = buffer::storage_rw_init(device, "xpbd_warm_positions", &positions);
        let velocities_buf = buffer::storage_rw_init(device, "xpbd_warm_velocities", &velocities);
        let prev_buf = buffer::storage_rw_init(device, "xpbd_warm_prev", &prev);
        let inverse_mass_buf =
            buffer::storage_read(device, "xpbd_warm_inverse_masses", &state.inverse_masses);
        let constraint_buf =
            buffer::storage_read(device, "xpbd_warm_constraints", &constraint_upload);
        let lambda_buf = buffer::storage_rw_zeroed(
            device,
            "xpbd_warm_lambdas",
            u64::from(constraint_count.max(1)) * 4,
        );
        let awake_buf = buffer::storage_read(device, "xpbd_warm_awake", &awake);
        let seed_buf = buffer::storage_read(device, "xpbd_warm_seed", &seed_upload);

        let global_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_xpbd_warm_global_bind_group"),
            layout: &self.global_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &velocities_buf),
                entry(3, &prev_buf),
                entry(4, &inverse_mass_buf),
                entry(5, &constraint_buf),
                entry(6, &lambda_buf),
                entry(7, &awake_buf),
                entry(8, &seed_buf),
            ],
        });

        let position_bytes = (state.len() * 16) as u64;
        let lambda_bytes = u64::from(constraint_count.max(1)) * 4;
        WarmSolvePlan {
            global_bind,
            positions_buf,
            velocities_buf,
            lambda_buf,
            position_bytes,
            lambda_bytes,
            particle_count,
            constraint_count,
        }
    }

    /// Records every substep's passes into one encoder, submits it, and returns
    /// the staging buffers the results were copied into.
    fn encode_and_run(
        &self,
        ctx: &GpuContext,
        plan: &WarmSolvePlan,
        colouring: &Colouring,
        substeps: u32,
        iterations: u32,
    ) -> WarmStaging {
        let device = ctx.device();
        let colour_binds = self.colour_bind_groups(ctx, colouring);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_xpbd_warm_encoder"),
        });
        let particle_groups = plan.particle_count.div_ceil(64).max(1);

        for _ in 0..substeps {
            self.pass(
                &mut encoder,
                "predict",
                &self.predict,
                plan,
                particle_groups,
                None,
            );
            // Warm-start replaces the cold path's single `reset_lambdas`
            // dispatch: one pass per colour (same-colour constraints share no
            // particle) seeds the multipliers and applies their cached
            // correction.
            for (c, &(start, end)) in colouring.ranges().iter().enumerate() {
                let groups = (end - start).div_ceil(64).max(1);
                self.pass(
                    &mut encoder,
                    "apply_warm_start",
                    &self.apply_warm_start,
                    plan,
                    groups,
                    Some(&colour_binds[c]),
                );
            }
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

        let position_stage =
            buffer::staging(device, "xpbd_warm_position_stage", plan.position_bytes);
        let velocity_stage =
            buffer::staging(device, "xpbd_warm_velocity_stage", plan.position_bytes);
        let lambda_stage = buffer::staging(device, "xpbd_warm_lambda_stage", plan.lambda_bytes);
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
        buffer::copy(
            &mut encoder,
            &plan.lambda_buf,
            &lambda_stage,
            plan.lambda_bytes,
        );
        ctx.queue().submit([encoder.finish()]);
        WarmStaging {
            positions: position_stage,
            velocities: velocity_stage,
            lambdas: lambda_stage,
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
                let buf = buffer::uniform(device, "xpbd_warm_colour_params", &params);
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("prism_xpbd_warm_colour_bind_group"),
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
        plan: &WarmSolvePlan,
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

/// The uploaded buffers and dispatch dimensions for one `solve_warm` call.
struct WarmSolvePlan {
    global_bind: BindGroup,
    positions_buf: Buffer,
    velocities_buf: Buffer,
    lambda_buf: Buffer,
    position_bytes: u64,
    lambda_bytes: u64,
    particle_count: u32,
    constraint_count: u32,
}

/// The staging buffers the device results were copied into for readback.
struct WarmStaging {
    positions: Buffer,
    velocities: Buffer,
    lambdas: Buffer,
}

/// Reads positions and velocities back into `state` and returns the converged
/// per-constraint multipliers (length `constraint_count`) for the cache store.
fn read_state_back(
    ctx: &GpuContext,
    staging: &WarmStaging,
    state: &mut ParticleState,
    constraint_count: u32,
) -> Vec<f32> {
    let positions = buffer::read_back::<[f32; 4]>(ctx, &staging.positions);
    let velocities = buffer::read_back::<[f32; 4]>(ctx, &staging.velocities);
    for i in 0..state.len() {
        state.positions[i] = vec4_to_vec3(positions[i]);
        state.velocities[i] = vec4_to_vec3(velocities[i]);
    }
    let lambdas = buffer::read_back::<f32>(ctx, &staging.lambdas);
    // The lambda buffer is padded to at least one element; return exactly the
    // `constraint_count` entries aligned with `ordered` for the cache store.
    lambdas
        .into_iter()
        .take(constraint_count as usize)
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
