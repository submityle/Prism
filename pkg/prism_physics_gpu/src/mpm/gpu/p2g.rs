//! Real-device `wgpu` pipeline for the MLS-MPM particle-to-grid (`P2G`)
//! affine scatter, isolated so a parity test can pin the transfer arithmetic
//! before it is fused into the full step.
//!
//! [`GpuMpmP2g`] compiles the shared pure-function library `shaders/mpm_math.wgsl`
//! concatenated with the scatter entry point in `shaders/mpm_p2g.wgsl`, and
//! exposes [`GpuMpmP2g::scatter`], which distributes each particle's mass and
//! affine (`APIC`) momentum, folded with the fixed-corotated internal-force
//! term, to its 27 surrounding grid nodes.
//!
//! This is the device twin of the `CPU` golden
//! [`prism_physics_core::mpm`] `particle_to_grid` transfer, evaluated up to
//! (but not including) [`prism_physics_core::mpm::Grid::finalize_velocity`]:
//! the returned grid momentum is the raw accumulated `Σ w·(m·v + affine·dpos)`,
//! matching the pre-finalize grid state.
//!
//! # Fixed-point accumulation
//!
//! Portable `f32` atomics do not exist in WGSL, so mass and each momentum
//! component are quantised to `i32` (scale, round, `atomicAdd`) exactly as the
//! fluid affine scatter does, and dequantised here on read-back. The scale is
//! chosen so accumulated magnitudes stay well within `i32` range for the tested
//! configurations.
//!
//! # Provenance
//!
//! The affine MLS-MPM scatter with the folded stress term (Hu et al. 2018;
//! Jiang et al. 2015) and the fixed-corotated model (Stomakhin et al. 2013) are
//! standard, publicly documented techniques. No Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::layout::{buffer_entry, entry};
use super::params::{mat3_to_cols, vec3_to_vec4};

/// Fixed-point scale applied to accumulated mass before quantising to `i32`.
/// Must match `MPM_MASS_SCALE` in `shaders/mpm_p2g.wgsl`.
const MASS_SCALE: f32 = 4_194_304.0;
/// Fixed-point scale applied to accumulated momentum components before
/// quantising to `i32`. Must match `MPM_MOMENTUM_SCALE` in the WGSL.
const MOMENTUM_SCALE: f32 = 4_194_304.0;

/// Uniform parameter block. Layout matches `P2gParams` in
/// `shaders/mpm_p2g.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct P2gParams {
    /// `xyz` = grid origin, `w` = cell size `dx`.
    origin_dx: [f32; 4],
    /// `x` = shear modulus `μ0`, `y` = first Lamé `λ0`, `z` = hardening `ξ`,
    /// `w` = time step `dt`.
    material: [f32; 4],
    /// `x` = `nx`, `y` = `ny`, `z` = `nz`, `w` = particle count.
    dims: [u32; 4],
    /// `x` = plastic-enabled flag (`0`/`1`), `yzw` = padding.
    flags: [u32; 4],
}

/// The accumulated grid state produced by one `P2G` scatter, before velocity
/// finalisation.
#[derive(Clone, Debug, PartialEq)]
pub struct P2gGrid {
    /// Accumulated node mass, one entry per grid node in `i + nx·(j + ny·k)`
    /// order.
    pub mass: Vec<f32>,
    /// Accumulated node momentum `Σ w·(m·v + affine·dpos)`, one entry per grid
    /// node in the same order as [`P2gGrid::mass`].
    pub momentum: Vec<Vec3>,
}

/// A compiled, reusable `P2G` affine-scatter `GPU` pipeline.
pub struct GpuMpmP2g {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMpmP2g {
    /// Compiles the `P2G` scatter kernel on `ctx`.
    ///
    /// The shared math library is concatenated ahead of the scatter entry
    /// point because WGSL has no include directive.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMpmP2g {
        let device = ctx.device();
        let source = format!(
            "{}\n{}",
            include_str!("../../shaders/mpm_math.wgsl"),
            include_str!("../../shaders/mpm_p2g.wgsl"),
        );
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_mpm_p2g"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_mpm_p2g_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
                buffer_entry(7, BufferBindingType::Storage { read_only: true }),
                buffer_entry(8, BufferBindingType::Storage { read_only: false }),
                buffer_entry(9, BufferBindingType::Storage { read_only: false }),
                buffer_entry(10, BufferBindingType::Storage { read_only: false }),
                buffer_entry(11, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_mpm_p2g_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_mpm_p2g_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("p2g_scatter"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMpmP2g {
            module,
            layout,
            pipeline,
        }
    }

    /// Scatters a batch of particles to a `nx·ny·nz` background grid and reads
    /// back the accumulated node mass and momentum.
    ///
    /// `positions`/`velocities` are the particle state, `affine` is the `APIC`
    /// affine velocity matrix `C`, `deformation` is the deformation gradient
    /// `F`, and `masses`/`volumes`/`plastic_dets` are the per-particle mass,
    /// initial volume, and plastic determinant `Jp`. `origin`/`dx` place the
    /// grid, `dt` is the time step, `mu0`/`lambda0` are the material Lamé
    /// parameters, and `hardening` is the snow hardening coefficient `ξ`. When
    /// `plastic` is `true` the Lamé parameters are scaled per particle by
    /// `hardening_factor(ξ, Jp)`, mirroring the `CPU` `P2G` path.
    ///
    /// The returned [`P2gGrid`] holds the pre-finalisation accumulated state
    /// (`momentum` is raw momentum, not velocity), matching the `CPU` golden
    /// grid before [`prism_physics_core::mpm::Grid::finalize_velocity`].
    ///
    /// # Panics
    ///
    /// Panics if the per-particle input slices do not all have equal length, or
    /// if any grid dimension is zero.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the scatter mirrors the CPU golden signature: particle state, \
                  grid placement, and material parameters are all independent inputs"
    )]
    pub fn scatter(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        velocities: &[Vec3],
        affine: &[Mat3],
        deformation: &[Mat3],
        masses: &[f32],
        volumes: &[f32],
        plastic_dets: &[f32],
        nx: usize,
        ny: usize,
        nz: usize,
        origin: Vec3,
        dx: f32,
        dt: f32,
        mu0: f32,
        lambda0: f32,
        hardening: f32,
        plastic: bool,
    ) -> P2gGrid {
        let count = positions.len();
        assert_eq!(count, velocities.len(), "velocities length mismatch");
        assert_eq!(count, affine.len(), "affine length mismatch");
        assert_eq!(count, deformation.len(), "deformation length mismatch");
        assert_eq!(count, masses.len(), "masses length mismatch");
        assert_eq!(count, volumes.len(), "volumes length mismatch");
        assert_eq!(
            count,
            plastic_dets.len(),
            "plastic determinant length mismatch"
        );
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "grid dimensions must be non-zero"
        );

        let node_count = nx * ny * nz;
        if count == 0 {
            return P2gGrid {
                mass: vec![0.0; node_count],
                momentum: vec![Vec3::ZERO; node_count],
            };
        }
        let device = ctx.device();

        let params = P2gParams {
            origin_dx: [origin.x, origin.y, origin.z, dx],
            material: [mu0, lambda0, hardening, dt],
            dims: [
                u32::try_from(nx).unwrap_or(u32::MAX),
                u32::try_from(ny).unwrap_or(u32::MAX),
                u32::try_from(nz).unwrap_or(u32::MAX),
                u32::try_from(count).unwrap_or(u32::MAX),
            ],
            flags: [u32::from(plastic), 0, 0, 0],
        };
        let params_buf = buffer::uniform(device, "prism_mpm_p2g_params", &params);

        let pos_packed: Vec<[f32; 4]> = positions.iter().map(vec3_to_vec4).collect();
        let vel_packed: Vec<[f32; 4]> = velocities.iter().map(vec3_to_vec4).collect();
        let affine_cols: Vec<[[f32; 4]; 3]> = affine.iter().map(mat3_to_cols).collect();
        let deform_cols: Vec<[[f32; 4]; 3]> = deformation.iter().map(mat3_to_cols).collect();

        let pos_buf = buffer::storage_read(device, "prism_mpm_p2g_positions", &pos_packed);
        let vel_buf = buffer::storage_read(device, "prism_mpm_p2g_velocities", &vel_packed);
        let affine_buf = buffer::storage_read(device, "prism_mpm_p2g_affine", &affine_cols);
        let deform_buf = buffer::storage_read(device, "prism_mpm_p2g_deformation", &deform_cols);
        let mass_buf = buffer::storage_read(device, "prism_mpm_p2g_masses", masses);
        let volume_buf = buffer::storage_read(device, "prism_mpm_p2g_volumes", volumes);
        let jp_buf = buffer::storage_read(device, "prism_mpm_p2g_plastic_det", plastic_dets);

        let grid_bytes = (node_count * size_of::<i32>()) as u64;
        let grid_mass = buffer::storage_rw_zeroed(device, "prism_mpm_p2g_grid_mass", grid_bytes);
        let grid_mom_x = buffer::storage_rw_zeroed(device, "prism_mpm_p2g_grid_mom_x", grid_bytes);
        let grid_mom_y = buffer::storage_rw_zeroed(device, "prism_mpm_p2g_grid_mom_y", grid_bytes);
        let grid_mom_z = buffer::storage_rw_zeroed(device, "prism_mpm_p2g_grid_mom_z", grid_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_mpm_p2g_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &pos_buf),
                entry(2, &vel_buf),
                entry(3, &affine_buf),
                entry(4, &deform_buf),
                entry(5, &mass_buf),
                entry(6, &volume_buf),
                entry(7, &jp_buf),
                entry(8, &grid_mass),
                entry(9, &grid_mom_x),
                entry(10, &grid_mom_y),
                entry(11, &grid_mom_z),
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_mpm_p2g_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_mpm_p2g_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = u32::try_from(count.div_ceil(64)).unwrap_or(u32::MAX);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let mass_stage = buffer::staging(device, "prism_mpm_p2g_mass_stage", grid_bytes);
        buffer::copy(&mut encoder, &grid_mass, &mass_stage, grid_bytes);
        let mom_x_stage = buffer::staging(device, "prism_mpm_p2g_mom_x_stage", grid_bytes);
        buffer::copy(&mut encoder, &grid_mom_x, &mom_x_stage, grid_bytes);
        let mom_y_stage = buffer::staging(device, "prism_mpm_p2g_mom_y_stage", grid_bytes);
        buffer::copy(&mut encoder, &grid_mom_y, &mom_y_stage, grid_bytes);
        let mom_z_stage = buffer::staging(device, "prism_mpm_p2g_mom_z_stage", grid_bytes);
        buffer::copy(&mut encoder, &grid_mom_z, &mom_z_stage, grid_bytes);

        ctx.queue().submit([encoder.finish()]);

        let mass_raw = buffer::read_back::<i32>(ctx, &mass_stage);
        let mom_x_raw = buffer::read_back::<i32>(ctx, &mom_x_stage);
        let mom_y_raw = buffer::read_back::<i32>(ctx, &mom_y_stage);
        let mom_z_raw = buffer::read_back::<i32>(ctx, &mom_z_stage);

        let mass = mass_raw.iter().map(|&m| m as f32 / MASS_SCALE).collect();
        let momentum = (0..node_count)
            .map(|n| {
                Vec3::new(
                    mom_x_raw[n] as f32 / MOMENTUM_SCALE,
                    mom_y_raw[n] as f32 / MOMENTUM_SCALE,
                    mom_z_raw[n] as f32 / MOMENTUM_SCALE,
                )
            })
            .collect();

        P2gGrid { mass, momentum }
    }
}
