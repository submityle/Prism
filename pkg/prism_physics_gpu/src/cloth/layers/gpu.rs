//! Real-device `wgpu` compute implementation of the multi-layer garment
//! coupling pass, the faithful twin of [`prism_physics_core`]'s
//! [`resolve_layer_coupling_jacobi`](prism_physics_core::resolve_layer_coupling_jacobi).
//!
//! [`GpuClothLayerCoupling`] compiles `shaders/cloth_layers.wgsl` once and
//! exposes a single [`GpuClothLayerCoupling::solve`] that resolves every
//! penetrating cross-layer particle pair with one parallel-safe (Jacobi)
//! iteration and returns the applied particle positions.
//!
//! # Correctness model
//!
//! The per-particle cross-layer adjacency is built on the host by
//! [`prep::build`](super::prep::build) so it is integer-identical to the
//! golden's broad phase (same cell assignment, same ascending traversal, same
//! same-layer filtering). Only the separating-push arithmetic runs on the GPU,
//! in a single own-slot pass: one invocation per particle folds its own half of
//! every incident cross-layer contact in the golden's reduction order from a
//! frozen snapshot and writes `out_positions`. Because the pass is own-slot and
//! reads only frozen inputs, it reproduces the whole Jacobi pass regardless of
//! invocation order; the only float divergence from the `CPU` is a few `ULP` in
//! `sqrt`/division, verified by the parity suite within a tight tolerance.
//!
//! # Provenance
//!
//! The layer-number stacking constraint and inverse-mass-weighted separation
//! are standard position-based dynamics, and the Jacobi own-slot accumulate is
//! standard parallel position-based dynamics. No Unreal Engine source or
//! derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use prism_physics_core::LayerParams;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::prep;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size(64)` in the kernel.
const WORKGROUP: usize = 64;

/// Uniform parameters shared with `Params` in `shaders/cloth_layers.wgsl`
/// (16 bytes / 4 words).
///
/// The field order mirrors the `WGSL` struct exactly; reordering silently
/// corrupts the solve.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of addressable particles (kernel thread count).
    particle_count: u32,
    /// Sanitized minimum inter-layer separation.
    thickness: f32,
    /// `thickness * thickness`, precomputed on the host like the golden.
    thickness_sq: f32,
    /// Padding to a 16-byte uniform.
    _pad: u32,
}

/// A compiled, reusable `GPU` inter-layer coupling pipeline.
pub struct GpuClothLayerCoupling {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the pass binds.
    layout: BindGroupLayout,
    /// Per-particle own-slot gather applying the summed corrections.
    pipeline: ComputePipeline,
}

impl GpuClothLayerCoupling {
    /// Compiles the inter-layer coupling kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothLayerCoupling {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_layers"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_layers.wgsl").into()),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_layers_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, read),
                buffer_entry(2, read),
                buffer_entry(3, read),
                buffer_entry(4, read),
                buffer_entry(5, read),
                buffer_entry(6, read),
                buffer_entry(7, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_layers_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cloth_layers_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("apply_layer_coupling"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothLayerCoupling {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every penetrating cross-layer particle pair with one
    /// parallel-safe (Jacobi) inter-layer coupling iteration and returns the
    /// applied particle positions.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_layer_coupling`](super::cpu::cpu_cloth_layer_coupling): both
    /// reproduce the engine's own parallel-safe inter-layer coupling pass. Any
    /// pass the host broad phase reports as a no-op (non-positive sanitized
    /// `thickness`/`cell_size`, a mismatched `inverse_masses` length, fewer than
    /// two particles, or no cross-layer neighbor) returns `positions` unchanged.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        layer_of: &[u32],
        normals: &[Vec3],
        params: LayerParams,
    ) -> Vec<Vec3> {
        let Some(prep) = prep::build(positions, inverse_masses, layer_of, normals, params) else {
            return positions.to_vec();
        };

        let device = ctx.device();
        let particle_count = prep.particle_count as usize;
        let out_bytes = (particle_count as u64) * 16;

        let uniform = Params {
            particle_count: prep.particle_count,
            thickness: prep.thickness,
            thickness_sq: prep.thickness_sq,
            _pad: 0,
        };
        let params_buf = buffer::uniform(device, "prism_cloth_layers_params", &uniform);

        let positions_buf =
            buffer::storage_read(device, "prism_cloth_layers_pos", &prep.positions);
        let normals_buf = buffer::storage_read(device, "prism_cloth_layers_normals", &prep.normals);
        let inv_mass_buf =
            buffer::storage_read(device, "prism_cloth_layers_invmass", &prep.inverse_masses);
        let layer_buf = buffer::storage_read(device, "prism_cloth_layers_layer", &prep.layer_of);
        let noff_buf =
            buffer::storage_read(device, "prism_cloth_layers_nbr_off", &prep.nbr_offsets);
        let nent_buf =
            buffer::storage_read(device, "prism_cloth_layers_nbr_ent", &prep.nbr_entries);

        // Own-slot output; `apply_layer_coupling` fully overwrites every slot,
        // so zeroed initialisation only guards unreachable lanes.
        let out_pos_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_layers_out_pos", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_layers_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &normals_buf),
                entry(3, &inv_mass_buf),
                entry(4, &layer_buf),
                entry(5, &noff_buf),
                entry(6, &nent_buf),
                entry(7, &out_pos_buf),
            ],
        });

        let groups = u32::try_from(particle_count.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let pos_stage = buffer::staging(device, "prism_cloth_layers_pos_stage", out_bytes);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_layers_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_layers_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        buffer::copy(&mut encoder, &out_pos_buf, &pos_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let pos_read = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);

        // The host broad phase addresses only the first `count` particles; any
        // tail beyond that is left exactly as the caller passed it in.
        let mut out_positions = positions.to_vec();
        for (i, slot) in out_positions.iter_mut().take(particle_count).enumerate() {
            let q = pos_read[i];
            *slot = Vec3::new(q[0], q[1], q[2]);
        }
        out_positions
    }
}
