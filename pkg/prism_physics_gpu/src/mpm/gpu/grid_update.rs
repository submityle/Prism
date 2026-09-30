//! Real-device `wgpu` pipeline for the MLS-MPM grid velocity update, isolated
//! so a parity test can pin the node-wise finalise / gravity / boundary logic
//! before it is fused into the full step.
//!
//! [`GpuMpmGridUpdate`] compiles `shaders/mpm_grid_update.wgsl` and exposes
//! [`GpuMpmGridUpdate::update`], which maps the accumulated `P2G` grid state
//! (node mass and momentum) to the finalised node velocity by, per node:
//!
//! 1. finalising `v = momentum / mass` on nodes with positive mass (empty nodes
//!    stay at zero velocity),
//! 2. adding the gravity increment `gravity · dt` to every node with mass, and
//! 3. enforcing the selected wall [`BoundaryMode`] within `thickness` nodes of
//!    each domain face.
//!
//! This is the device twin of the `CPU` golden
//! [`prism_physics_core::mpm::Grid::finalize_velocity`],
//! `Grid::add_velocity_to_active(gravity · dt)`, and
//! [`prism_physics_core::mpm::apply_grid_boundary`], run in that order.
//!
//! Unlike the `P2G` scatter this is a pure per-node map with no cross-invocation
//! contention, so it reads and writes plain `f32` buffers with no fixed-point
//! atomics.
//!
//! # Provenance
//!
//! The affine PIC grid update and the standard MPM wall boundary conditions
//! (Stomakhin et al. 2013; Jiang et al. 2015) are standard, publicly documented
//! techniques. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::layout::{buffer_entry, entry};
use super::params::vec3_to_vec4;

/// The wall boundary condition applied to grid nodes near a domain face.
///
/// The discriminants match the `MPM_BOUNDARY_*` selectors in
/// `shaders/mpm_grid_update.wgsl` and the variants of
/// [`prism_physics_core::mpm::BoundaryCondition`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Zero the full velocity of boundary nodes (no-slip).
    Sticky,
    /// Zero only the wall-normal velocity component (free tangential slip).
    Slip,
    /// Zero the wall-normal component only when it points into the wall.
    Separate,
}

impl BoundaryMode {
    /// The `u32` selector uploaded to the kernel.
    #[must_use]
    pub(crate) fn as_u32(self) -> u32 {
        match self {
            BoundaryMode::Sticky => 0,
            BoundaryMode::Slip => 1,
            BoundaryMode::Separate => 2,
        }
    }
}

/// Uniform parameter block. Layout matches `GridUpdateParams` in
/// `shaders/mpm_grid_update.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridUpdateParams {
    /// `xyz` = gravity acceleration, `w` = time step `dt`.
    gravity_dt: [f32; 4],
    /// `x` = `nx`, `y` = `ny`, `z` = `nz`, `w` = node count.
    dims: [u32; 4],
    /// `x` = boundary thickness (nodes), `y` = boundary mode, `zw` = padding.
    bounds: [u32; 4],
}

/// A compiled, reusable grid-velocity-update `GPU` pipeline.
pub struct GpuMpmGridUpdate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMpmGridUpdate {
    /// Compiles the grid-update kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMpmGridUpdate {
        let device = ctx.device();
        let source = include_str!("../../shaders/mpm_grid_update.wgsl");
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_mpm_grid_update"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_mpm_grid_update_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_mpm_grid_update_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_mpm_grid_update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("grid_update"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMpmGridUpdate {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps the accumulated grid state to finalised node velocities.
    ///
    /// `mass` and `momentum` are the accumulated `P2G` node state in
    /// `i + nx·(j + ny·k)` order (as produced by
    /// [`super::p2g::GpuMpmP2g::scatter`]). `gravity`/`dt` drive the external
    /// force increment, and `thickness`/`boundary` select the wall condition
    /// applied within `thickness` nodes of each face.
    ///
    /// Returns the finalised velocity per node in the same order; empty nodes
    /// (zero mass) return a zero velocity.
    ///
    /// # Panics
    ///
    /// Panics if `mass` and `momentum` have different lengths, if that length
    /// does not equal `nx · ny · nz`, or if any grid dimension is zero.
    #[must_use]
    pub fn update(
        &self,
        ctx: &GpuContext,
        mass: &[f32],
        momentum: &[Vec3],
        nx: usize,
        ny: usize,
        nz: usize,
        gravity: Vec3,
        dt: f32,
        thickness: usize,
        boundary: BoundaryMode,
    ) -> Vec<Vec3> {
        assert!(
            nx > 0 && ny > 0 && nz > 0,
            "grid dimensions must be non-zero"
        );
        let node_count = nx * ny * nz;
        assert_eq!(mass.len(), node_count, "mass length must equal node count");
        assert_eq!(
            momentum.len(),
            node_count,
            "momentum length must equal node count"
        );
        let device = ctx.device();

        let params = GridUpdateParams {
            gravity_dt: [gravity.x, gravity.y, gravity.z, dt],
            dims: [
                u32::try_from(nx).unwrap_or(u32::MAX),
                u32::try_from(ny).unwrap_or(u32::MAX),
                u32::try_from(nz).unwrap_or(u32::MAX),
                u32::try_from(node_count).unwrap_or(u32::MAX),
            ],
            bounds: [
                u32::try_from(thickness).unwrap_or(u32::MAX),
                boundary.as_u32(),
                0,
                0,
            ],
        };
        let params_buf = buffer::uniform(device, "prism_mpm_grid_update_params", &params);

        let mom_packed: Vec<[f32; 4]> = momentum.iter().map(vec3_to_vec4).collect();
        let mass_buf = buffer::storage_read(device, "prism_mpm_grid_update_mass", mass);
        let mom_buf = buffer::storage_read(device, "prism_mpm_grid_update_momentum", &mom_packed);

        let vel_bytes = (node_count * size_of::<[f32; 4]>()) as u64;
        let vel_buf =
            buffer::storage_rw_zeroed(device, "prism_mpm_grid_update_velocity", vel_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_mpm_grid_update_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &mass_buf),
                entry(2, &mom_buf),
                entry(3, &vel_buf),
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_mpm_grid_update_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_mpm_grid_update_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            let groups = u32::try_from(node_count.div_ceil(64)).unwrap_or(u32::MAX);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let vel_stage = buffer::staging(device, "prism_mpm_grid_update_velocity_stage", vel_bytes);
        buffer::copy(&mut encoder, &vel_buf, &vel_stage, vel_bytes);

        ctx.queue().submit([encoder.finish()]);

        let vel_raw = buffer::read_back::<[f32; 4]>(ctx, &vel_stage);
        vel_raw
            .iter()
            .map(|v| Vec3::new(v[0], v[1], v[2]))
            .collect()
    }
}
