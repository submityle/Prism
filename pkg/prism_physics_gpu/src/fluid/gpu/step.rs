//! Real-device `wgpu` orchestrator that runs a complete `FLIP`/`APIC` fluid
//! step in a single command submission.
//!
//! [`GpuFluidStep`] compiles `shaders/fluid_step.wgsl` once — a fused module
//! whose entry points reproduce, stage by stage, the arithmetic of the
//! standalone transfer / grid-ops / pressure / extrapolation / advection
//! kernels — and exposes [`GpuFluidStep::step`], which advances the markers by
//! one full step in place. The whole pipeline runs over one resident set of
//! grid buffers, so only the initial particle upload and the final particle
//! read-back cross the bus:
//!
//! 1. `p2g_scatter` — fixed-point atomic splat of particle velocities.
//! 2. `normalize` — momentum / weight per staggered face.
//! 3. `save_field` — snapshot the transferred field for the `FLIP` increment.
//! 4. `add_gravity` — integrate the constant body force on every face.
//! 5. `enforce_u` / `enforce_v` / `enforce_w` — zero faces bordering solids.
//! 6. `sor_red` / `sor_black` — iterated red-black `SOR` pressure projection,
//!    then `subtract_u` / `subtract_v` / `subtract_w` remove the gradient.
//! 7. `init_known` then `extrap_sweep` — grow the known band into the air so
//!    the surface advection stays stable.
//! 8. `g2p` — gather grid velocities back with the `PIC`/`FLIP` blend.
//! 9. `advect` — move the markers through the projected flow with `RK2`.
//!
//! Separate compute passes give the implicit memory barrier that makes each
//! stage's writes visible to the next, reproducing the sequential order of the
//! [`super::super::cpu::step::fluid_step`] golden twin.
//!
//! # Extrapolation ping-pong
//!
//! `extrap_sweep` reads the `*_scratch` (source) buffers and writes the primary
//! `velocity` / `known` (destination) buffers. The host therefore seeds the
//! scratch buffers from the primary field once, then copies the primary field
//! back into the scratch buffers after every sweep so the next sweep sees the
//! grown band. The final field always lands in the primary `velocity` buffer,
//! which `g2p` and `advect` read directly, so no parity bookkeeping is needed.
//!
//! # Provenance
//!
//! The `P2G`/`G2P` `PIC`/`FLIP` transfer (Zhu and Bridson 2005; Bridson), the
//! `MAC` body-force and solid boundary operators (Harlow and Welch 1965;
//! Bridson), the red-black `SOR` pressure projection (Bridson; Foster and
//! Fedkiw 2001), the free-surface velocity extrapolation, and the `RK2`
//! advection are all standard published techniques. No Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayout,
    PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;
use crate::fluid::config::{FluidConfig, FluidError};
use crate::fluid::grid::{CellType, GridDims};
use crate::fluid::particle::FluidParticles;

use super::layout::{buffer_entry, entry};

/// Uniform parameter block. Layout matches `StepParams` in
/// `shaders/fluid_step.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StepParams {
    /// `xyz` = grid origin, `w` = cell size `dx`.
    origin_dx: [f32; 4],
    /// `x`/`y`/`z` = cell counts, `w` = particle count.
    dims: [u32; 4],
    /// `x`/`y`/`z` = concatenated face bases, `w` = total face count.
    bases: [u32; 4],
    /// `x` = cell count, `yzw` = padding.
    cell: [u32; 4],
    /// `x`/`y`/`z` = gravity times `dt` (per-axis face increment), `w` = `dt`.
    grav_dt: [f32; 4],
    /// `x` = right-hand-side scale, `y` = over-relaxation, `z` = subtraction
    /// scale, `w` = cell size.
    coeff: [f32; 4],
    /// `x` = `FLIP` blend, `yzw` = padding.
    blend: [f32; 4],
}

/// A compiled, reusable full-step `GPU` fluid pipeline set.
pub struct GpuFluidStep {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    p2g_scatter: ComputePipeline,
    normalize: ComputePipeline,
    save_field: ComputePipeline,
    add_gravity: ComputePipeline,
    enforce_u: ComputePipeline,
    enforce_v: ComputePipeline,
    enforce_w: ComputePipeline,
    sor_red: ComputePipeline,
    sor_black: ComputePipeline,
    subtract_u: ComputePipeline,
    subtract_v: ComputePipeline,
    subtract_w: ComputePipeline,
    init_known: ComputePipeline,
    extrap_sweep: ComputePipeline,
    g2p: ComputePipeline,
    advect: ComputePipeline,
}

impl GpuFluidStep {
    /// Compiles every full-step kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFluidStep {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_fluid_step"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/fluid_step.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_fluid_step_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
                buffer_entry(7, BufferBindingType::Storage { read_only: false }),
                buffer_entry(8, BufferBindingType::Storage { read_only: false }),
                buffer_entry(9, BufferBindingType::Storage { read_only: false }),
                buffer_entry(10, BufferBindingType::Storage { read_only: false }),
                buffer_entry(11, BufferBindingType::Storage { read_only: true }),
                buffer_entry(12, BufferBindingType::Storage { read_only: false }),
                buffer_entry(13, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_fluid_step_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mk = |entry_point: &str, label: &str| {
            make(device, &module, entry_point, label, &pipeline_layout)
        };
        GpuFluidStep {
            p2g_scatter: mk("p2g_scatter", "prism_fluid_step_p2g_scatter"),
            normalize: mk("normalize", "prism_fluid_step_normalize"),
            save_field: mk("save_field", "prism_fluid_step_save_field"),
            add_gravity: mk("add_gravity", "prism_fluid_step_add_gravity"),
            enforce_u: mk("enforce_u", "prism_fluid_step_enforce_u"),
            enforce_v: mk("enforce_v", "prism_fluid_step_enforce_v"),
            enforce_w: mk("enforce_w", "prism_fluid_step_enforce_w"),
            sor_red: mk("sor_red", "prism_fluid_step_sor_red"),
            sor_black: mk("sor_black", "prism_fluid_step_sor_black"),
            subtract_u: mk("subtract_u", "prism_fluid_step_subtract_u"),
            subtract_v: mk("subtract_v", "prism_fluid_step_subtract_v"),
            subtract_w: mk("subtract_w", "prism_fluid_step_subtract_w"),
            init_known: mk("init_known", "prism_fluid_step_init_known"),
            extrap_sweep: mk("extrap_sweep", "prism_fluid_step_extrap_sweep"),
            g2p: mk("g2p", "prism_fluid_step_g2p"),
            advect: mk("advect", "prism_fluid_step_advect"),
            module,
            layout,
        }
    }

    /// Advances `particles` on the grid described by `dims` by one full fluid
    /// step, writing the new positions and velocities back in place.
    ///
    /// `cell_types` is the row-major cell classification (length
    /// [`GridDims::cell_count`]); it drives the solid boundary condition and the
    /// free-surface Dirichlet condition of the pressure solve. `cfg` supplies
    /// the time step, gravity, blend, and the pressure / extrapolation sweep
    /// counts. The transfer uses the `PIC`/`FLIP` blend from
    /// [`FluidConfig::effective_flip_blend`]; the affine (`APIC`) transfer is a
    /// later slice.
    ///
    /// # Errors
    ///
    /// Returns [`FluidError::EmptyGrid`] if `dims` has a zero axis,
    /// [`FluidError::InconsistentParticles`] if the particle columns have
    /// mismatched lengths, and [`FluidError::InvalidConfig`] if `cell_types`
    /// does not match the grid cell count.
    pub fn step(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        cell_types: &[CellType],
        particles: &mut FluidParticles,
        cfg: &FluidConfig,
    ) -> Result<(), FluidError> {
        if !dims.is_valid() {
            return Err(FluidError::EmptyGrid);
        }
        if !particles.is_consistent() {
            return Err(FluidError::InconsistentParticles);
        }
        if cell_types.len() != dims.cell_count() {
            return Err(FluidError::InvalidConfig(
                "cell_types length must equal the grid cell count",
            ));
        }
        if particles.is_empty() {
            return Ok(());
        }

        let plan = self.upload(ctx, dims, cell_types, particles, cfg);
        let (positions_stage, velocities_stage) = self.encode_and_run(ctx, &plan, cfg);

        let new_positions = buffer::read_back::<[f32; 4]>(ctx, &positions_stage);
        let new_velocities = buffer::read_back::<[f32; 4]>(ctx, &velocities_stage);
        for (dst, src) in particles.positions_mut().iter_mut().zip(&new_positions) {
            *dst = glam::Vec3::new(src[0], src[1], src[2]);
        }
        for (dst, src) in particles.velocities_mut().iter_mut().zip(&new_velocities) {
            *dst = glam::Vec3::new(src[0], src[1], src[2]);
        }
        Ok(())
    }

    /// Uploads the particles and a fresh resident set of grid buffers.
    fn upload(
        &self,
        ctx: &GpuContext,
        dims: GridDims,
        cell_types: &[CellType],
        particles: &FluidParticles,
        cfg: &FluidConfig,
    ) -> StepPlan {
        let device = ctx.device();
        let particle_count = particles.len() as u32;
        let face_total = dims.face_total() as u32;
        let cell_count = dims.cell_count() as u32;

        let rhs_scale = cfg.density * dims.dx * dims.dx / cfg.dt;
        let omega = cfg.effective_over_relaxation();
        let sub_scale = cfg.dt / (cfg.density * dims.dx);

        let params = StepParams {
            origin_dx: [dims.origin.x, dims.origin.y, dims.origin.z, dims.dx],
            dims: [dims.nx, dims.ny, dims.nz, particle_count],
            bases: [
                dims.u_offset() as u32,
                dims.v_offset() as u32,
                dims.w_offset() as u32,
                face_total,
            ],
            cell: [cell_count, 0, 0, 0],
            grav_dt: [
                cfg.gravity.x * cfg.dt,
                cfg.gravity.y * cfg.dt,
                cfg.gravity.z * cfg.dt,
                cfg.dt,
            ],
            coeff: [rhs_scale, omega, sub_scale, dims.dx],
            blend: [cfg.effective_flip_blend(), 0.0, 0.0, 0.0],
        };

        let positions: Vec<[f32; 4]> = particles.positions().iter().map(vec3_to_vec4).collect();
        let velocities: Vec<[f32; 4]> = particles.velocities().iter().map(vec3_to_vec4).collect();
        let out_init = vec![[0.0f32; 4]; particles.len()];
        let codes: Vec<u32> = cell_types.iter().map(|t| t.code()).collect();

        let face_bytes = u64::from(face_total) * 4;
        let params_buf = buffer::uniform(device, "fluid_step_params", &params);
        let positions_buf = buffer::storage_read(device, "fluid_step_positions", &positions);
        let velocities_in_buf =
            buffer::storage_read(device, "fluid_step_velocities_in", &velocities);
        let velocities_out_buf =
            buffer::storage_rw_init(device, "fluid_step_velocities_out", &out_init);
        let positions_out_buf =
            buffer::storage_rw_init(device, "fluid_step_positions_out", &out_init);
        let momentum_buf = buffer::storage_rw_zeroed(device, "fluid_step_momentum", face_bytes);
        let weight_buf = buffer::storage_rw_zeroed(device, "fluid_step_weight", face_bytes);
        let velocity_buf = buffer::storage_rw_zeroed(device, "fluid_step_velocity", face_bytes);
        let velocity_scratch_buf =
            buffer::storage_rw_zeroed(device, "fluid_step_velocity_scratch", face_bytes);
        let saved_buf = buffer::storage_rw_zeroed(device, "fluid_step_saved", face_bytes);
        let pressure_buf =
            buffer::storage_rw_zeroed(device, "fluid_step_pressure", u64::from(cell_count) * 4);
        let cell_buf = buffer::storage_read(device, "fluid_step_cells", &codes);
        let known_buf = buffer::storage_rw_zeroed(device, "fluid_step_known", face_bytes);
        let known_scratch_buf =
            buffer::storage_rw_zeroed(device, "fluid_step_known_scratch", face_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_fluid_step_bind_group"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &velocities_in_buf),
                entry(3, &velocities_out_buf),
                entry(4, &positions_out_buf),
                entry(5, &momentum_buf),
                entry(6, &weight_buf),
                entry(7, &velocity_buf),
                entry(8, &velocity_scratch_buf),
                entry(9, &saved_buf),
                entry(10, &pressure_buf),
                entry(11, &cell_buf),
                entry(12, &known_buf),
                entry(13, &known_scratch_buf),
            ],
        });

        StepPlan {
            bind,
            velocity_buf,
            velocity_scratch_buf,
            known_buf,
            known_scratch_buf,
            positions_out_buf,
            velocities_out_buf,
            particle_bytes: (particles.len() * 16) as u64,
            face_bytes,
            particle_count,
            face_total,
            cell_count,
            u_count: dims.u_count() as u32,
            v_count: dims.v_count() as u32,
            w_count: dims.w_count() as u32,
        }
    }

    /// Records the full step into one encoder, submits, and returns the staging
    /// buffers the new positions and velocities were copied into.
    fn encode_and_run(
        &self,
        ctx: &GpuContext,
        plan: &StepPlan,
        cfg: &FluidConfig,
    ) -> (Buffer, Buffer) {
        let device = ctx.device();
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_fluid_step_encoder"),
        });
        let particle_groups = plan.particle_count.div_ceil(64).max(1);
        let face_groups = plan.face_total.div_ceil(64).max(1);
        let cell_groups = plan.cell_count.div_ceil(64).max(1);
        let u_groups = plan.u_count.div_ceil(64).max(1);
        let v_groups = plan.v_count.div_ceil(64).max(1);
        let w_groups = plan.w_count.div_ceil(64).max(1);

        self.pass(&mut encoder, &self.p2g_scatter, plan, particle_groups);
        self.pass(&mut encoder, &self.normalize, plan, face_groups);
        self.pass(&mut encoder, &self.save_field, plan, face_groups);
        self.pass(&mut encoder, &self.add_gravity, plan, face_groups);
        self.pass(&mut encoder, &self.enforce_u, plan, u_groups);
        self.pass(&mut encoder, &self.enforce_v, plan, v_groups);
        self.pass(&mut encoder, &self.enforce_w, plan, w_groups);
        for _ in 0..cfg.pressure_iterations {
            self.pass(&mut encoder, &self.sor_red, plan, cell_groups);
            self.pass(&mut encoder, &self.sor_black, plan, cell_groups);
        }
        self.pass(&mut encoder, &self.subtract_u, plan, u_groups);
        self.pass(&mut encoder, &self.subtract_v, plan, v_groups);
        self.pass(&mut encoder, &self.subtract_w, plan, w_groups);

        // Seed the extrapolation known band, then grow it: each sweep reads the
        // scratch source and writes the primary destination, so the primary
        // field is copied back into the scratch buffers before the next sweep.
        self.pass(&mut encoder, &self.init_known, plan, face_groups);
        buffer::copy(
            &mut encoder,
            &plan.velocity_buf,
            &plan.velocity_scratch_buf,
            plan.face_bytes,
        );
        buffer::copy(
            &mut encoder,
            &plan.known_buf,
            &plan.known_scratch_buf,
            plan.face_bytes,
        );
        for _ in 0..cfg.extrapolation_iterations {
            self.pass(&mut encoder, &self.extrap_sweep, plan, face_groups);
            buffer::copy(
                &mut encoder,
                &plan.velocity_buf,
                &plan.velocity_scratch_buf,
                plan.face_bytes,
            );
            buffer::copy(
                &mut encoder,
                &plan.known_buf,
                &plan.known_scratch_buf,
                plan.face_bytes,
            );
        }

        self.pass(&mut encoder, &self.g2p, plan, particle_groups);
        self.pass(&mut encoder, &self.advect, plan, particle_groups);

        let positions_stage =
            buffer::staging(device, "fluid_step_positions_stage", plan.particle_bytes);
        buffer::copy(
            &mut encoder,
            &plan.positions_out_buf,
            &positions_stage,
            plan.particle_bytes,
        );
        let velocities_stage =
            buffer::staging(device, "fluid_step_velocities_stage", plan.particle_bytes);
        buffer::copy(
            &mut encoder,
            &plan.velocities_out_buf,
            &velocities_stage,
            plan.particle_bytes,
        );

        ctx.queue().submit([encoder.finish()]);
        (positions_stage, velocities_stage)
    }

    /// Records one compute pass dispatching `groups` workgroups.
    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &ComputePipeline,
        plan: &StepPlan,
        groups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_fluid_step_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &plan.bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// The uploaded buffers and dispatch dimensions for one full step.
struct StepPlan {
    bind: BindGroup,
    velocity_buf: Buffer,
    velocity_scratch_buf: Buffer,
    known_buf: Buffer,
    known_scratch_buf: Buffer,
    positions_out_buf: Buffer,
    velocities_out_buf: Buffer,
    particle_bytes: u64,
    face_bytes: u64,
    particle_count: u32,
    face_total: u32,
    cell_count: u32,
    u_count: u32,
    v_count: u32,
    w_count: u32,
}

/// Compiles one compute pipeline for `entry_point` under `layout`.
fn make(
    device: &wgpu::Device,
    module: &ShaderModule,
    entry_point: &str,
    label: &str,
    layout: &PipelineLayout,
) -> ComputePipeline {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Packs a [`glam::Vec3`] into a padded `vec4` upload element.
fn vec3_to_vec4(v: &glam::Vec3) -> [f32; 4] {
    [v.x, v.y, v.z, 0.0]
}
