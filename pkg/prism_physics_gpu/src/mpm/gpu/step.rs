//! Real-device `wgpu` orchestrator that runs a complete MLS-MPM step — and a
//! multi-step advance — in a single command submission.
//!
//! [`GpuMpmStep`] compiles the shared pure-function library
//! `shaders/mpm_math.wgsl` concatenated with the fused `shaders/mpm_step.wgsl`,
//! whose four entry points reproduce, stage by stage, the arithmetic of the
//! standalone `P2G` scatter, grid update, and `G2P` gather kernels but over one
//! resident set of grid and particle buffers. [`GpuMpmStep::advance`] advances
//! the particle batch by `steps` full steps in place, so only the initial
//! particle upload and the final particle read-back cross the bus regardless of
//! how many steps run:
//!
//! 1. `step_clear`  — zero the fixed-point atomic grid accumulators.
//! 2. `p2g_scatter` — affine scatter of mass/momentum into the atomics.
//! 3. `grid_update` — dequantise, finalise `v = p/m`, add gravity, wall
//!    boundary.
//! 4. `g2p_gather`  — gather velocity and affine `C`, advect, update `F`, apply
//!    the snow return mapping, then clamp the position into the domain interior.
//!
//! Each entry point runs as its own compute pass, so the implicit inter-pass
//! memory barrier makes each stage's writes visible to the next, reproducing the
//! sequential order of the `CPU` golden [`prism_physics_core::mpm`] `MpmSolver`
//! step (clear grid, `particle_to_grid`, `finalize_velocity`, add gravity,
//! `apply_grid_boundary`, `grid_to_particle`, `clamp_particles`). Looping the
//! four passes inside one encoder reproduces `MpmSolver::advance`.
//!
//! # Fixed-point accumulation
//!
//! `f32` atomics are not portable, so mass and each momentum component are
//! quantised to `i32` for the atomic scatter and dequantised in the grid update,
//! using the same `2^22` scale as the standalone `P2G` kernel. The `step_clear`
//! pass resets the accumulators at the start of every step.
//!
//! # Provenance
//!
//! The affine MLS-MPM transfer with the folded stress term (Hu et al. 2018;
//! Jiang et al. 2015), the fixed-corotated / snow plasticity model (Stomakhin
//! et al. 2013), and the standard MPM wall boundary conditions are all standard,
//! publicly documented techniques. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::grid_update::BoundaryMode;
use super::layout::{buffer_entry, entry};
use super::params::{cols_to_mat3, mat3_to_cols, vec3_to_vec4};

/// Uniform parameter block. Layout matches `StepParams` in
/// `shaders/mpm_step.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StepParams {
    /// `xyz` = grid origin, `w` = cell size `dx`.
    origin_dx: [f32; 4],
    /// `x` = shear modulus `μ0`, `y` = first Lamé `λ0`, `z` = hardening `ξ`,
    /// `w` = time step `dt`.
    material: [f32; 4],
    /// `xyz` = gravity acceleration, `w` = time step `dt`.
    gravity_dt: [f32; 4],
    /// `x` = critical compression `θc`, `y` = critical stretch `θs`, `zw` =
    /// padding.
    snow: [f32; 4],
    /// `x` = `nx`, `y` = `ny`, `z` = `nz`, `w` = particle count.
    dims: [u32; 4],
    /// `x` = grid node count, `yzw` = padding.
    grid: [u32; 4],
    /// `x` = boundary thickness (nodes), `y` = boundary mode, `z` = plastic flag
    /// (`0`/`1`), `w` = padding.
    bounds: [u32; 4],
    /// `xyz` = clamp lower corner, `w` = padding.
    clamp_lo: [f32; 4],
    /// `xyz` = clamp upper corner, `w` = padding.
    clamp_hi: [f32; 4],
}

/// The particle state produced by an MLS-MPM advance.
#[derive(Clone, Debug, PartialEq)]
pub struct StepParticles {
    /// Advected particle positions, one entry per particle.
    pub positions: Vec<Vec3>,
    /// `APIC` particle velocities, one entry per particle.
    pub velocities: Vec<Vec3>,
    /// `APIC` affine velocity matrices `C`, one entry per particle.
    pub affine: Vec<Mat3>,
    /// Deformation gradients `F` (elastic part after the return mapping when
    /// plasticity is enabled), one entry per particle.
    pub deformation: Vec<Mat3>,
    /// Plastic determinants `Jp`, one entry per particle.
    pub plastic_det: Vec<f32>,
}

/// Geometry, material, and stepping inputs for one [`GpuMpmStep::advance`] call.
///
/// Grouping the scalar inputs keeps the advance entry point readable and lets
/// the host mirror the `CPU` golden `MpmConfig` / `MpmMaterial` split.
#[derive(Clone, Copy, Debug)]
pub struct StepConfig {
    /// Grid node counts along each axis.
    pub dims: (usize, usize, usize),
    /// Grid origin (the position of node `(0, 0, 0)`).
    pub origin: Vec3,
    /// Grid cell size `dx`.
    pub dx: f32,
    /// Time step `dt`.
    pub dt: f32,
    /// Gravity acceleration added to every node with mass each step.
    pub gravity: Vec3,
    /// Shear modulus `μ0`.
    pub mu0: f32,
    /// First Lamé parameter `λ0`.
    pub lambda0: f32,
    /// Snow hardening coefficient `ξ`.
    pub hardening: f32,
    /// Snow critical compression `θc`.
    pub theta_c: f32,
    /// Snow critical stretch `θs`.
    pub theta_s: f32,
    /// Wall boundary condition applied within `boundary_thickness` nodes of each
    /// face.
    pub boundary: BoundaryMode,
    /// Boundary layer thickness in nodes.
    pub boundary_thickness: usize,
    /// Whether the snow return mapping runs (elastoplastic vs. purely elastic).
    pub plastic: bool,
    /// Number of full steps to advance.
    pub steps: usize,
}

/// The initial particle state uploaded to [`GpuMpmStep::advance`].
///
/// All slices must have the same length (the particle count).
#[derive(Clone, Copy, Debug)]
pub struct StepInputs<'a> {
    /// Initial particle positions.
    pub positions: &'a [Vec3],
    /// Initial particle velocities.
    pub velocities: &'a [Vec3],
    /// Initial affine velocity matrices `C`.
    pub affine: &'a [Mat3],
    /// Initial deformation gradients `F`.
    pub deformation: &'a [Mat3],
    /// Per-particle masses (read only).
    pub masses: &'a [f32],
    /// Per-particle initial volumes (read only).
    pub volumes: &'a [f32],
    /// Initial plastic determinants `Jp`.
    pub plastic_det: &'a [f32],
}

/// A compiled, reusable full-step MLS-MPM `GPU` pipeline set.
pub struct GpuMpmStep {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    clear: ComputePipeline,
    p2g: ComputePipeline,
    grid: ComputePipeline,
    g2p: ComputePipeline,
}

impl GpuMpmStep {
    /// Compiles every full-step kernel on `ctx`.
    ///
    /// The shared math library is concatenated ahead of the fused step module
    /// because WGSL has no include directive.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMpmStep {
        let device = ctx.device();
        let source = format!(
            "{}\n{}",
            include_str!("../../shaders/mpm_math.wgsl"),
            include_str!("../../shaders/mpm_step.wgsl"),
        );
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_mpm_step"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_mpm_step_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                // Particle state updated in place.
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                // Read-only masses and volumes.
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
                // Plastic determinant updated in place.
                buffer_entry(7, BufferBindingType::Storage { read_only: false }),
                // Fixed-point atomic grid accumulators.
                buffer_entry(8, BufferBindingType::Storage { read_only: false }),
                buffer_entry(9, BufferBindingType::Storage { read_only: false }),
                buffer_entry(10, BufferBindingType::Storage { read_only: false }),
                buffer_entry(11, BufferBindingType::Storage { read_only: false }),
                // Finalised node velocity field.
                buffer_entry(12, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_mpm_step_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mk = |entry_point: &str, label: &str| {
            make(device, &module, entry_point, label, &pipeline_layout)
        };
        GpuMpmStep {
            clear: mk("step_clear", "prism_mpm_step_clear"),
            p2g: mk("p2g_scatter", "prism_mpm_step_p2g"),
            grid: mk("grid_update", "prism_mpm_step_grid"),
            g2p: mk("g2p_gather", "prism_mpm_step_g2p"),
            module,
            layout,
        }
    }

    /// Advances `inputs` by `cfg.steps` full MLS-MPM steps and returns the final
    /// particle state.
    ///
    /// All particle buffers are resident on the device for the whole advance, so
    /// only the initial upload and final read-back cross the bus. The advanced
    /// positions are clamped into the domain interior each step, mirroring the
    /// `CPU` golden `clamp_particles`.
    ///
    /// # Panics
    ///
    /// Panics if the [`StepInputs`] slices do not all have the same length.
    #[must_use]
    pub fn advance(
        &self,
        ctx: &GpuContext,
        inputs: &StepInputs,
        cfg: &StepConfig,
    ) -> StepParticles {
        let count = inputs.positions.len();
        assert!(
            inputs.velocities.len() == count
                && inputs.affine.len() == count
                && inputs.deformation.len() == count
                && inputs.masses.len() == count
                && inputs.volumes.len() == count
                && inputs.plastic_det.len() == count,
            "StepInputs slices must all have the particle count length",
        );
        if count == 0 {
            return StepParticles {
                positions: Vec::new(),
                velocities: Vec::new(),
                affine: Vec::new(),
                deformation: Vec::new(),
                plastic_det: Vec::new(),
            };
        }
        let device = ctx.device();
        let (nx, ny, nz) = cfg.dims;
        let node_count = nx * ny * nz;

        // Clamp corners, mirroring `clamp_particles`: keep every particle at
        // least `margin` inside each face so its stencil stays in range.
        let margin = (cfg.boundary_thickness.max(2)) as f32 * cfg.dx;
        let lo = cfg.origin + Vec3::splat(margin);
        let hi = cfg.origin
            + Vec3::new(
                (nx - 1) as f32 * cfg.dx,
                (ny - 1) as f32 * cfg.dx,
                (nz - 1) as f32 * cfg.dx,
            )
            - Vec3::splat(margin);

        let params = StepParams {
            origin_dx: [cfg.origin.x, cfg.origin.y, cfg.origin.z, cfg.dx],
            material: [cfg.mu0, cfg.lambda0, cfg.hardening, cfg.dt],
            gravity_dt: [cfg.gravity.x, cfg.gravity.y, cfg.gravity.z, cfg.dt],
            snow: [cfg.theta_c, cfg.theta_s, 0.0, 0.0],
            dims: [
                u32::try_from(nx).unwrap_or(u32::MAX),
                u32::try_from(ny).unwrap_or(u32::MAX),
                u32::try_from(nz).unwrap_or(u32::MAX),
                u32::try_from(count).unwrap_or(u32::MAX),
            ],
            grid: [u32::try_from(node_count).unwrap_or(u32::MAX), 0, 0, 0],
            bounds: [
                u32::try_from(cfg.boundary_thickness).unwrap_or(u32::MAX),
                cfg.boundary.as_u32(),
                u32::from(cfg.plastic),
                0,
            ],
            clamp_lo: [lo.x, lo.y, lo.z, 0.0],
            clamp_hi: [hi.x, hi.y, hi.z, 0.0],
        };
        let params_buf = buffer::uniform(device, "prism_mpm_step_params", &params);

        // Resident particle buffers. Positions, velocities, affine, deformation,
        // and Jp are read-write (updated in place and copied back); masses and
        // volumes are read-only.
        let pos_packed: Vec<[f32; 4]> = inputs.positions.iter().map(vec3_to_vec4).collect();
        let vel_packed: Vec<[f32; 4]> = inputs.velocities.iter().map(vec3_to_vec4).collect();
        let affine_cols: Vec<[[f32; 4]; 3]> = inputs.affine.iter().map(mat3_to_cols).collect();
        let deform_cols: Vec<[[f32; 4]; 3]> = inputs.deformation.iter().map(mat3_to_cols).collect();

        let pos_buf = buffer::storage_rw_init(device, "prism_mpm_step_positions", &pos_packed);
        let vel_buf = buffer::storage_rw_init(device, "prism_mpm_step_velocities", &vel_packed);
        let affine_buf = buffer::storage_rw_init(device, "prism_mpm_step_affine", &affine_cols);
        let deform_buf =
            buffer::storage_rw_init(device, "prism_mpm_step_deformation", &deform_cols);
        let mass_buf = buffer::storage_read(device, "prism_mpm_step_masses", inputs.masses);
        let vol_buf = buffer::storage_read(device, "prism_mpm_step_volumes", inputs.volumes);
        let jp_buf =
            buffer::storage_rw_init(device, "prism_mpm_step_plastic_det", inputs.plastic_det);

        // Resident grid buffers: four fixed-point atomic accumulators plus the
        // finalised node velocity field.
        let atomic_bytes = (node_count * size_of::<i32>()) as u64;
        let node_vec4_bytes = (node_count * size_of::<[f32; 4]>()) as u64;
        let grid_mass = buffer::storage_rw_zeroed(device, "prism_mpm_step_grid_mass", atomic_bytes);
        let grid_mom_x =
            buffer::storage_rw_zeroed(device, "prism_mpm_step_grid_mom_x", atomic_bytes);
        let grid_mom_y =
            buffer::storage_rw_zeroed(device, "prism_mpm_step_grid_mom_y", atomic_bytes);
        let grid_mom_z =
            buffer::storage_rw_zeroed(device, "prism_mpm_step_grid_mom_z", atomic_bytes);
        let grid_vel =
            buffer::storage_rw_zeroed(device, "prism_mpm_step_grid_velocity", node_vec4_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_mpm_step_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &pos_buf),
                entry(2, &vel_buf),
                entry(3, &affine_buf),
                entry(4, &deform_buf),
                entry(5, &mass_buf),
                entry(6, &vol_buf),
                entry(7, &jp_buf),
                entry(8, &grid_mass),
                entry(9, &grid_mom_x),
                entry(10, &grid_mom_y),
                entry(11, &grid_mom_z),
                entry(12, &grid_vel),
            ],
        });

        let node_groups = u32::try_from(node_count.div_ceil(64)).unwrap_or(u32::MAX);
        let particle_groups = u32::try_from(count.div_ceil(64)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_mpm_step_encoder"),
        });
        for _ in 0..cfg.steps {
            self.pass(&mut encoder, &self.clear, &bind, node_groups);
            self.pass(&mut encoder, &self.p2g, &bind, particle_groups);
            self.pass(&mut encoder, &self.grid, &bind, node_groups);
            self.pass(&mut encoder, &self.g2p, &bind, particle_groups);
        }

        let vec4_bytes = (count * size_of::<[f32; 4]>()) as u64;
        let mat3_bytes = (count * size_of::<[[f32; 4]; 3]>()) as u64;
        let scalar_bytes = (count * size_of::<f32>()) as u64;

        let pos_stage = buffer::staging(device, "prism_mpm_step_positions_stage", vec4_bytes);
        buffer::copy(&mut encoder, &pos_buf, &pos_stage, vec4_bytes);
        let vel_stage = buffer::staging(device, "prism_mpm_step_velocities_stage", vec4_bytes);
        buffer::copy(&mut encoder, &vel_buf, &vel_stage, vec4_bytes);
        let affine_stage = buffer::staging(device, "prism_mpm_step_affine_stage", mat3_bytes);
        buffer::copy(&mut encoder, &affine_buf, &affine_stage, mat3_bytes);
        let deform_stage = buffer::staging(device, "prism_mpm_step_deformation_stage", mat3_bytes);
        buffer::copy(&mut encoder, &deform_buf, &deform_stage, mat3_bytes);
        let jp_stage = buffer::staging(device, "prism_mpm_step_plastic_det_stage", scalar_bytes);
        buffer::copy(&mut encoder, &jp_buf, &jp_stage, scalar_bytes);

        ctx.queue().submit([encoder.finish()]);

        let pos_raw = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        let vel_raw = buffer::read_back::<[f32; 4]>(ctx, &vel_stage);
        let affine_raw = buffer::read_back::<[[f32; 4]; 3]>(ctx, &affine_stage);
        let deform_raw = buffer::read_back::<[[f32; 4]; 3]>(ctx, &deform_stage);
        let jp_raw = buffer::read_back::<f32>(ctx, &jp_stage);

        StepParticles {
            positions: pos_raw
                .iter()
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect(),
            velocities: vel_raw
                .iter()
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect(),
            affine: affine_raw.iter().map(cols_to_mat3).collect(),
            deformation: deform_raw.iter().map(cols_to_mat3).collect(),
            plastic_det: jp_raw,
        }
    }

    /// Records one compute pass dispatching `groups` workgroups over `bind`.
    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &ComputePipeline,
        bind: &wgpu::BindGroup,
        groups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism_mpm_step_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
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
