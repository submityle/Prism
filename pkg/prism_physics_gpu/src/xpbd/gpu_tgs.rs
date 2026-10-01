//! Real-device `wgpu` compute implementation of the colour-ordered Temporal
//! Gauss-Seidel (`TGS`) distance solver.
//!
//! [`GpuTgsSolver`] compiles `shaders/tgs.wgsl` once and exposes
//! [`GpuTgsSolver::solve`], which uploads the particle state and the
//! colour-reordered constraints, then encodes the full frame — for each
//! substep: `integrate_velocities`, a biased velocity sweep group
//! (`reset_impulses` then `iterations` passes of one `solve_biased` per
//! colour), `integrate_positions`, then a bias-free relaxation sweep group
//! (`reset_impulses` then `relax_iterations` passes of one `solve_relax` per
//! colour) — as a chain of compute passes in a single submission.
//!
//! Separate passes give the implicit memory barrier that makes each colour's
//! velocity writes visible to the next colour, so the device reproduces the
//! sequential Gauss-Seidel sweep of the [`tgs_solve`](super::tgs_solve) golden
//! twin. Both run the identical `f32` arithmetic in the identical order; only
//! device floating-point reassociation (fused multiply-add, differing division
//! and square-root rounding) separates them, which the parity test bounds with
//! a tight tolerance rather than exact equality.
//!
//! Unlike the `XPBD` kernel, `TGS` carries no previous-position snapshot: it
//! corrects velocities against a position-error bias and integrates positions
//! from the corrected velocity, so the accumulated impulse (reset before each
//! sweep group) is the only per-constraint scratch the device needs.
//!
//! # Provenance
//!
//! Temporal Gauss-Seidel substepping with soft constraints (`PhysX` 5 / Chaos
//! lineage; Catto, "Soft Constraints", GDC 2011 for the coefficient form);
//! standard `wgpu` compute dispatch. No Unreal Engine source or derived code.

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
use super::config::XpbdError;
use super::constraint::{DistanceConstraint, GpuConstraint};
use super::state::ParticleState;
use super::tgs::TgsConfig;
use super::tgs_soft::SoftParams;

/// Global solver parameters. Layout matches `Params` in `shaders/tgs.wgsl`
/// (48 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Uniform acceleration applied each substep, typically gravity.
    gravity: [f32; 3],
    /// Substep size `dt / substeps`.
    h: f32,
    /// Linear velocity retention per substep: `(1 - damping * h).max(0)`.
    damping_scale: f32,
    /// Soft-constraint bias coefficient (`bias = bias_rate * C`).
    bias_rate: f32,
    /// Soft-constraint effective-mass scale, in `[0, 1]`.
    mass_scale: f32,
    /// Soft-constraint accumulated-impulse decay, in `[0, 1]`.
    impulse_scale: f32,
    /// Number of particles.
    particle_count: u32,
    /// Number of (colour-reordered) constraints.
    constraint_count: u32,
    /// Padding to the 16-byte `std140` uniform stride.
    _pad0: u32,
    /// Padding to the 16-byte `std140` uniform stride.
    _pad1: u32,
}

/// Per-colour dispatch parameters. Layout matches `ColourParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColourParams {
    /// Index of the first constraint in this colour's contiguous range.
    start: u32,
    /// Number of constraints in this colour.
    count: u32,
    /// Padding to the 16-byte `std140` uniform stride.
    _pad0: u32,
    /// Padding to the 16-byte `std140` uniform stride.
    _pad1: u32,
}

/// A compiled, reusable `GPU` `TGS` distance-solver pipeline set.
pub struct GpuTgsSolver {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// Group-0 layout: params, positions, velocities, inverse masses,
    /// constraints, impulses, awake mask.
    global_layout: BindGroupLayout,
    /// Group-1 layout: the per-colour constraint-range uniform.
    colour_layout: BindGroupLayout,
    /// Stage 1: integrate velocity under gravity and damping.
    integrate_velocities: ComputePipeline,
    /// Stage 3: advance positions with the corrected velocity.
    integrate_positions: ComputePipeline,
    /// Clears the per-constraint accumulated impulse before a sweep group.
    reset_impulses: ComputePipeline,
    /// Stage 2: biased velocity solve for one colour (soft coefficients).
    solve_biased: ComputePipeline,
    /// Stage 4: bias-free relaxation solve for one colour.
    solve_relax: ComputePipeline,
}

impl GpuTgsSolver {
    /// Compiles the solver kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTgsSolver {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_tgs"),
            source: ShaderSource::Wgsl(include_str!("../shaders/tgs.wgsl").into()),
        });
        let global_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_tgs_global_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
            ],
        });
        let colour_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_tgs_colour_layout"),
            entries: &[buffer_entry(0, BufferBindingType::Uniform)],
        });
        // `integrate_velocities` / `integrate_positions` / `reset_impulses`
        // only touch group 0; the two colour sweeps also read the per-colour
        // uniform in group 1.
        let global_only = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_tgs_global_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout)],
            immediate_size: 0,
        });
        let with_colour = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_tgs_colour_pipeline_layout"),
            bind_group_layouts: &[Some(&global_layout), Some(&colour_layout)],
            immediate_size: 0,
        });
        let integrate_velocities = make(
            device,
            &module,
            "integrate_velocities",
            "prism_tgs_integrate_velocities",
            &global_only,
        );
        let integrate_positions = make(
            device,
            &module,
            "integrate_positions",
            "prism_tgs_integrate_positions",
            &global_only,
        );
        let reset_impulses = make(
            device,
            &module,
            "reset_impulses",
            "prism_tgs_reset_impulses",
            &global_only,
        );
        let solve_biased = make(
            device,
            &module,
            "solve_biased",
            "prism_tgs_solve_biased",
            &with_colour,
        );
        let solve_relax = make(
            device,
            &module,
            "solve_relax",
            "prism_tgs_solve_relax",
            &with_colour,
        );
        GpuTgsSolver {
            module,
            global_layout,
            colour_layout,
            integrate_velocities,
            integrate_positions,
            reset_impulses,
            solve_biased,
            solve_relax,
        }
    }

    /// Advances `state` under `constraints` by `dt` seconds on device using
    /// Temporal Gauss-Seidel substepping.
    ///
    /// Mutates `state` in place with the positions and velocities read back
    /// from the device; state carries forward to the next call exactly as the
    /// `CPU` twin's does.
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
        config: &TgsConfig,
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
        // The dense solve integrates every particle, so the awake mask is all
        // ones; the island-aware stepper calls [`solve_masked`](Self::solve_masked)
        // with a real mask. An all-ones mask makes the shader's awake branch a
        // no-op, preserving bit-identical behaviour with the pre-mask kernel.
        let awake = vec![1u32; state.len()];
        self.solve_masked(ctx, state, constraints, &awake, config, dt)
    }

    /// Advances `state` under `constraints`, integrating only the particles the
    /// `awake` mask selects, and recovering velocities for only those
    /// particles.
    ///
    /// This is the shared device core of both the dense [`solve`](Self::solve)
    /// (all-ones mask, every constraint) and the island-aware `GPU` `TGS`
    /// stepper (per-particle awake mask, awake-island constraints only).
    /// Factoring it out keeps a single, parity-locked dispatch path: the two
    /// callers differ only in which particles and constraints they feed in,
    /// never in the device arithmetic. `awake` must have one `u32` per particle
    /// (1 = awake, 0 = frozen).
    ///
    /// # Errors
    ///
    /// Returns [`XpbdError`] when the config or state is invalid, a constraint
    /// indexes a missing particle, or the constraint graph needs more colours
    /// than supported. Does nothing (returns `Ok`) when there are no particles
    /// or `dt` is non-positive.
    pub(crate) fn solve_masked(
        &self,
        ctx: &GpuContext,
        state: &mut ParticleState,
        constraints: &[DistanceConstraint],
        awake: &[u32],
        config: &TgsConfig,
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
        debug_assert_eq!(
            awake.len(),
            state.len(),
            "awake mask must cover every particle"
        );

        let particle_count = state.len() as u32;
        let colouring = Colouring::build(constraints, particle_count)?;
        let ordered = colouring.reorder(constraints);

        let substeps = config.effective_substeps();
        let iterations = config.effective_iterations();
        let relax_iterations = config.effective_relax_iterations();
        let h = dt / substeps as f32;
        if h <= 0.0 {
            return Ok(());
        }
        let plan = self.upload(ctx, state, &ordered, awake, config, h);
        let staging = self.encode_and_run(
            ctx,
            &plan,
            &colouring,
            substeps,
            iterations,
            relax_iterations,
        );
        read_state_back(ctx, &staging, state);
        Ok(())
    }

    /// Uploads all buffers and builds the global bind group.
    ///
    /// The soft-constraint coefficients are derived once per solve from the
    /// config's `hertz` / `damping_ratio` at the substep size `h`, matching the
    /// `CPU` twin's single `SoftParams::from_hertz` call.
    fn upload(
        &self,
        ctx: &GpuContext,
        state: &ParticleState,
        ordered: &[DistanceConstraint],
        awake: &[u32],
        config: &TgsConfig,
        h: f32,
    ) -> SolvePlan {
        let device = ctx.device();
        let particle_count = state.len() as u32;
        let constraint_count = ordered.len() as u32;

        let soft = SoftParams::from_hertz(config.hertz, config.damping_ratio, h);
        let params = Params {
            gravity: [config.gravity.x, config.gravity.y, config.gravity.z],
            h,
            damping_scale: (1.0 - config.damping * h).max(0.0),
            bias_rate: soft.bias_rate,
            mass_scale: soft.mass_scale,
            impulse_scale: soft.impulse_scale,
            particle_count,
            constraint_count,
            _pad0: 0,
            _pad1: 0,
        };

        let positions: Vec<[f32; 4]> = state.positions.iter().map(vec3_to_vec4).collect();
        let velocities: Vec<[f32; 4]> = state.velocities.iter().map(vec3_to_vec4).collect();
        let gpu_constraints: Vec<GpuConstraint> = ordered.iter().map(|c| c.to_gpu()).collect();
        let constraint_upload = if gpu_constraints.is_empty() {
            vec![GpuConstraint::zeroed()]
        } else {
            gpu_constraints
        };

        let params_buf = buffer::uniform(device, "tgs_params", &params);
        let positions_buf = buffer::storage_rw_init(device, "tgs_positions", &positions);
        let velocities_buf = buffer::storage_rw_init(device, "tgs_velocities", &velocities);
        let inverse_mass_buf =
            buffer::storage_read(device, "tgs_inverse_masses", &state.inverse_masses);
        let constraint_buf = buffer::storage_read(device, "tgs_constraints", &constraint_upload);
        let impulse_buf = buffer::storage_rw_zeroed(
            device,
            "tgs_impulses",
            u64::from(constraint_count.max(1)) * 4,
        );
        // The device needs a non-empty binding even when there are no
        // particles; `solve_masked` has already returned on the empty state, so
        // `awake` is non-empty here.
        let awake_buf = buffer::storage_read(device, "tgs_awake", awake);

        let global_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_tgs_global_bind_group"),
            layout: &self.global_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &velocities_buf),
                entry(3, &inverse_mass_buf),
                entry(4, &constraint_buf),
                entry(5, &impulse_buf),
                entry(6, &awake_buf),
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
    ///
    /// Each substep: integrate velocities, then the biased sweep group
    /// (`reset_impulses` once, then `iterations` sweeps of one `solve_biased`
    /// per colour), then integrate positions, then — only when
    /// `relax_iterations > 0` — the relaxation sweep group (`reset_impulses`
    /// once, then `relax_iterations` sweeps of one `solve_relax` per colour).
    /// Resetting the impulse once per sweep group (not once per iteration)
    /// matches the `CPU` twin's `sweep`, so the soft `impulse_scale` decay acts
    /// across each group's iterations only.
    fn encode_and_run(
        &self,
        ctx: &GpuContext,
        plan: &SolvePlan,
        colouring: &Colouring,
        substeps: u32,
        iterations: u32,
        relax_iterations: u32,
    ) -> Staging {
        let device = ctx.device();
        let colour_binds = self.colour_bind_groups(ctx, colouring);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_tgs_encoder"),
        });
        let particle_groups = plan.particle_count.div_ceil(64).max(1);
        let constraint_groups = plan.constraint_count.div_ceil(64).max(1);

        for _ in 0..substeps {
            self.pass(
                &mut encoder,
                "integrate_velocities",
                &self.integrate_velocities,
                plan,
                particle_groups,
                None,
            );
            self.pass(
                &mut encoder,
                "reset_impulses",
                &self.reset_impulses,
                plan,
                constraint_groups,
                None,
            );
            for _ in 0..iterations {
                for (c, &(start, end)) in colouring.ranges().iter().enumerate() {
                    let groups = (end - start).div_ceil(64).max(1);
                    self.pass(
                        &mut encoder,
                        "solve_biased",
                        &self.solve_biased,
                        plan,
                        groups,
                        Some(&colour_binds[c]),
                    );
                }
            }
            self.pass(
                &mut encoder,
                "integrate_positions",
                &self.integrate_positions,
                plan,
                particle_groups,
                None,
            );
            if relax_iterations > 0 {
                self.pass(
                    &mut encoder,
                    "reset_impulses",
                    &self.reset_impulses,
                    plan,
                    constraint_groups,
                    None,
                );
                for _ in 0..relax_iterations {
                    for (c, &(start, end)) in colouring.ranges().iter().enumerate() {
                        let groups = (end - start).div_ceil(64).max(1);
                        self.pass(
                            &mut encoder,
                            "solve_relax",
                            &self.solve_relax,
                            plan,
                            groups,
                            Some(&colour_binds[c]),
                        );
                    }
                }
            }
        }

        let position_stage = buffer::staging(device, "tgs_position_stage", plan.position_bytes);
        let velocity_stage = buffer::staging(device, "tgs_velocity_stage", plan.position_bytes);
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
                let buf = buffer::uniform(device, "tgs_colour_params", &params);
                device.create_bind_group(&BindGroupDescriptor {
                    label: Some("prism_tgs_colour_bind_group"),
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
    /// Group-0 bind group covering all seven solver buffers.
    global_bind: BindGroup,
    /// The read-write positions buffer, copied back after the frame.
    positions_buf: Buffer,
    /// The read-write velocities buffer, copied back after the frame.
    velocities_buf: Buffer,
    /// Byte length of the position (and velocity) buffer.
    position_bytes: u64,
    /// Number of particles.
    particle_count: u32,
    /// Number of constraints.
    constraint_count: u32,
}

/// The staging buffers the device results were copied into for readback.
struct Staging {
    /// Positions staged for host readback.
    positions: Buffer,
    /// Velocities staged for host readback.
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
