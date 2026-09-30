//! Real-device `wgpu` pipeline for the MLS-MPM grid-to-particle (`G2P`) affine
//! gather, isolated so a parity test can pin the gather arithmetic and the
//! deformation-gradient update before they are fused into the full step.
//!
//! [`GpuMpmG2p`] compiles the shared pure-function library `shaders/mpm_math.wgsl`
//! concatenated with the gather entry point in `shaders/mpm_g2p.wgsl`, and
//! exposes [`GpuMpmG2p::gather`], which for each particle gathers the finalised
//! node velocities from its 27 surrounding grid nodes into an `APIC` velocity
//! and affine matrix `C`, advects the particle, updates the deformation
//! gradient `F`, and (when plasticity is enabled) applies the snow return
//! mapping to produce the new plastic determinant `Jp`.
//!
//! This is the device twin of the `CPU` golden
//! [`prism_physics_core::mpm`] `grid_to_particle` transfer. The gather consumes
//! the *finalised* node velocities (the output of the grid update), matching the
//! solver order `P2G` → grid update → `G2P`.
//!
//! Unlike the `P2G` scatter this is a pure per-particle gather with no
//! cross-invocation contention, so it reads and writes plain `f32` buffers with
//! no fixed-point atomics.
//!
//! # Provenance
//!
//! The `APIC` gather (Jiang et al. 2015), the MLS-MPM deformation update (Hu et
//! al. 2018), and the snow return mapping (Stomakhin et al. 2013) are standard,
//! publicly documented techniques. No Unreal Engine source or derived code.

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
use super::params::{cols_to_mat3, mat3_to_cols, vec3_to_vec4};

/// Uniform parameter block. Layout matches `G2pParams` in
/// `shaders/mpm_g2p.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct G2pParams {
    /// `xyz` = grid origin, `w` = cell size `dx`.
    origin_dx: [f32; 4],
    /// `x` = time step `dt`, `y` = critical compression `θc`, `z` = critical
    /// stretch `θs`, `w` = padding.
    step: [f32; 4],
    /// `x` = `nx`, `y` = `ny`, `z` = `nz`, `w` = particle count.
    dims: [u32; 4],
    /// `x` = plastic-enabled flag (`0`/`1`), `yzw` = padding.
    flags: [u32; 4],
}

/// The updated particle state produced by one `G2P` gather.
#[derive(Clone, Debug, PartialEq)]
pub struct G2pParticles {
    /// Advected particle positions, one entry per particle.
    pub positions: Vec<Vec3>,
    /// Gathered `APIC` particle velocities, one entry per particle.
    pub velocities: Vec<Vec3>,
    /// Gathered `APIC` affine velocity matrices `C`, one entry per particle.
    pub affine: Vec<Mat3>,
    /// Updated deformation gradients `F` (elastic part after the return mapping
    /// when plasticity is enabled), one entry per particle.
    pub deformation: Vec<Mat3>,
    /// Updated plastic determinants `Jp`, one entry per particle (unchanged from
    /// the input when plasticity is disabled).
    pub plastic_det: Vec<f32>,
}

/// A compiled, reusable `G2P` affine-gather `GPU` pipeline.
pub struct GpuMpmG2p {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMpmG2p {
    /// Compiles the `G2P` gather kernel on `ctx`.
    ///
    /// The shared math library is concatenated ahead of the gather entry point
    /// because WGSL has no include directive.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMpmG2p {
        let device = ctx.device();
        let source = format!(
            "{}\n{}",
            include_str!("../../shaders/mpm_math.wgsl"),
            include_str!("../../shaders/mpm_g2p.wgsl"),
        );
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_mpm_g2p"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_mpm_g2p_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
                buffer_entry(7, BufferBindingType::Storage { read_only: false }),
                buffer_entry(8, BufferBindingType::Storage { read_only: false }),
                buffer_entry(9, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_mpm_g2p_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_mpm_g2p_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("g2p_gather"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMpmG2p {
            module,
            layout,
            pipeline,
        }
    }

    /// Gathers the finalised grid velocities back to a batch of particles and
    /// reads back the updated particle state.
    ///
    /// `positions` are the current particle positions, `deformation` the current
    /// deformation gradients `F`, and `plastic_det` the current plastic
    /// determinants `Jp`. `grid_velocity` holds the finalised node velocities in
    /// `i + nx·(j + ny·k)` order. `origin`/`dx` place the grid, `dt` is the time
    /// step, and `theta_c`/`theta_s` are the snow critical compression / stretch.
    /// When `plastic` is `true` the trial deformation is split by the snow return
    /// mapping into an elastic part and a new `Jp`; otherwise the trial
    /// deformation is kept as-is and `Jp` is unchanged, mirroring the `CPU` `G2P`
    /// path.
    ///
    /// # Panics
    ///
    /// Panics if the per-particle input slices do not all have equal length, if
    /// any grid dimension is zero, or if `grid_velocity` does not have `nx·ny·nz`
    /// entries.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the gather mirrors the CPU golden signature: particle state, \
                  grid placement, and plasticity parameters are all independent inputs"
    )]
    pub fn gather(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        deformation: &[Mat3],
        plastic_det: &[f32],
        grid_velocity: &[Vec3],
        nx: usize,
        ny: usize,
        nz: usize,
        origin: Vec3,
        dx: f32,
        dt: f32,
        theta_c: f32,
        theta_s: f32,
        plastic: bool,
    ) -> G2pParticles {
        let count = positions.len();
        assert_eq!(count, deformation.len(), "deformation length mismatch");
        assert_eq!(
            count,
            plastic_det.len(),
            "plastic determinant length mismatch"
        );
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "grid dimensions must be non-zero"
        );
        let node_count = nx * ny * nz;
        assert_eq!(
            grid_velocity.len(),
            node_count,
            "grid velocity length must equal node count"
        );

        if count == 0 {
            return G2pParticles {
                positions: Vec::new(),
                velocities: Vec::new(),
                affine: Vec::new(),
                deformation: Vec::new(),
                plastic_det: Vec::new(),
            };
        }
        let device = ctx.device();

        let params = G2pParams {
            origin_dx: [origin.x, origin.y, origin.z, dx],
            step: [dt, theta_c, theta_s, 0.0],
            dims: [
                u32::try_from(nx).unwrap_or(u32::MAX),
                u32::try_from(ny).unwrap_or(u32::MAX),
                u32::try_from(nz).unwrap_or(u32::MAX),
                u32::try_from(count).unwrap_or(u32::MAX),
            ],
            flags: [u32::from(plastic), 0, 0, 0],
        };
        let params_buf = buffer::uniform(device, "prism_mpm_g2p_params", &params);

        let pos_packed: Vec<[f32; 4]> = positions.iter().map(vec3_to_vec4).collect();
        let deform_cols: Vec<[[f32; 4]; 3]> = deformation.iter().map(mat3_to_cols).collect();
        let vel_packed: Vec<[f32; 4]> = grid_velocity.iter().map(vec3_to_vec4).collect();

        let pos_buf = buffer::storage_read(device, "prism_mpm_g2p_positions", &pos_packed);
        let deform_buf = buffer::storage_read(device, "prism_mpm_g2p_deformation", &deform_cols);
        let jp_buf = buffer::storage_read(device, "prism_mpm_g2p_plastic_det", plastic_det);
        let grid_vel_buf = buffer::storage_read(device, "prism_mpm_g2p_grid_velocity", &vel_packed);

        let vec4_bytes = (count * size_of::<[f32; 4]>()) as u64;
        let mat3_bytes = (count * size_of::<[[f32; 4]; 3]>()) as u64;
        let scalar_bytes = (count * size_of::<f32>()) as u64;

        let out_pos = buffer::storage_rw_zeroed(device, "prism_mpm_g2p_out_positions", vec4_bytes);
        let out_vel = buffer::storage_rw_zeroed(device, "prism_mpm_g2p_out_velocities", vec4_bytes);
        let out_affine = buffer::storage_rw_zeroed(device, "prism_mpm_g2p_out_affine", mat3_bytes);
        let out_deform =
            buffer::storage_rw_zeroed(device, "prism_mpm_g2p_out_deformation", mat3_bytes);
        let out_jp =
            buffer::storage_rw_zeroed(device, "prism_mpm_g2p_out_plastic_det", scalar_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_mpm_g2p_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &pos_buf),
                entry(2, &deform_buf),
                entry(3, &jp_buf),
                entry(4, &grid_vel_buf),
                entry(5, &out_pos),
                entry(6, &out_vel),
                entry(7, &out_affine),
                entry(8, &out_deform),
                entry(9, &out_jp),
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_mpm_g2p_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_mpm_g2p_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = u32::try_from(count.div_ceil(64)).unwrap_or(u32::MAX);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let pos_stage = buffer::staging(device, "prism_mpm_g2p_out_positions_stage", vec4_bytes);
        buffer::copy(&mut encoder, &out_pos, &pos_stage, vec4_bytes);
        let vel_stage = buffer::staging(device, "prism_mpm_g2p_out_velocities_stage", vec4_bytes);
        buffer::copy(&mut encoder, &out_vel, &vel_stage, vec4_bytes);
        let affine_stage = buffer::staging(device, "prism_mpm_g2p_out_affine_stage", mat3_bytes);
        buffer::copy(&mut encoder, &out_affine, &affine_stage, mat3_bytes);
        let deform_stage =
            buffer::staging(device, "prism_mpm_g2p_out_deformation_stage", mat3_bytes);
        buffer::copy(&mut encoder, &out_deform, &deform_stage, mat3_bytes);
        let jp_stage = buffer::staging(device, "prism_mpm_g2p_out_plastic_det_stage", scalar_bytes);
        buffer::copy(&mut encoder, &out_jp, &jp_stage, scalar_bytes);

        ctx.queue().submit([encoder.finish()]);

        let pos_raw = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        let vel_raw = buffer::read_back::<[f32; 4]>(ctx, &vel_stage);
        let affine_raw = buffer::read_back::<[[f32; 4]; 3]>(ctx, &affine_stage);
        let deform_raw = buffer::read_back::<[[f32; 4]; 3]>(ctx, &deform_stage);
        let jp_raw = buffer::read_back::<f32>(ctx, &jp_stage);

        G2pParticles {
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
}
