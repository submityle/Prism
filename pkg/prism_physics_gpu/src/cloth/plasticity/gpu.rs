//! Real-device `wgpu` compute implementation of cloth plasticity.
//!
//! [`GpuClothPlasticity`] compiles `shaders/cloth_plasticity.wgsl` once and
//! exposes a single [`GpuClothPlasticity::solve`] that runs one plasticity pass:
//! every distance edge, in parallel, samples its endpoint separation from a
//! read-only position snapshot and writes its (possibly crept) rest length,
//! while a single atomic counter tallies the edges that crept.
//!
//! This is the real-device twin of
//! [`cpu_cloth_plasticity`](super::cpu::cpu_cloth_plasticity); both delegate the
//! creep arithmetic to the same
//! [`prism_physics_core::plastic_rest_length`](prism_physics_core::plastic_rest_length)
//! kernel, so parity holds within a tight tolerance (`GPU` division rounding
//! perturbs the low bits) and the `modified` count matches exactly.
//!
//! # Provenance
//!
//! Rest-length creep past a yield strain is a standard, publicly documented
//! plastic-set model for position-based cloth. No Unreal Engine source or
//! derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use prism_physics_core::PlasticParams;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::ClothPlasticEdge;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size(64)` in the kernel.
const WORKGROUP: usize = 64;

/// Uniform parameters shared with `Params` in `shaders/cloth_plasticity.wgsl`
/// (32 bytes / 8 words).
///
/// The field order mirrors the `WGSL` struct exactly; reordering silently
/// corrupts the solve. The three strain scalars are uploaded already sanitized
/// so the kernel can use them directly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of addressable particles (positions length).
    particle_count: u32,
    /// Number of edges in the slice.
    edge_count: u32,
    /// Strain magnitude beyond which plastic flow begins (sanitized).
    yield_strain: f32,
    /// Fraction of beyond-yield strain converted to a rest-length change (sanitized).
    creep: f32,
    /// Residual elastic-strain cap left after creep (sanitized).
    max_strain: f32,
    /// Padding to a 16-byte boundary.
    _pad0: f32,
    /// Padding to a 16-byte boundary.
    _pad1: f32,
    /// Padding to a 16-byte boundary.
    _pad2: f32,
}

/// A compiled, reusable `GPU` cloth plasticity pipeline.
pub struct GpuClothPlasticity {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the pass reads and writes.
    layout: BindGroupLayout,
    /// The per-edge rest-length creep pass.
    pipeline: ComputePipeline,
}

impl GpuClothPlasticity {
    /// Compiles the cloth plasticity kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothPlasticity {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_plasticity"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_plasticity.wgsl").into()),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_plasticity_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, read),
                buffer_entry(2, read),
                buffer_entry(3, write),
                buffer_entry(4, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_plasticity_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cloth_plasticity_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothPlasticity {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs one plasticity pass over `edges` and returns the per-edge updated
    /// rest lengths (parallel to `edges`) and the number of edges that crept.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_plasticity`](super::cpu::cpu_cloth_plasticity): each edge is
    /// independent, so one thread owns one edge with no colouring. An edge that
    /// stays within the yield band, has a degenerate rest length, or references
    /// an out-of-range particle keeps its input rest length. An empty `edges`
    /// slice returns an empty vector and a zero count.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        edges: &[ClothPlasticEdge],
        params: PlasticParams,
    ) -> (Vec<Real>, u32) {
        if edges.is_empty() {
            return (Vec::new(), 0);
        }

        let device = ctx.device();
        let params = params.sanitized();
        let uniform = Params {
            particle_count: u32::try_from(positions.len()).unwrap_or(u32::MAX),
            edge_count: u32::try_from(edges.len()).unwrap_or(u32::MAX),
            yield_strain: params.yield_strain,
            creep: params.creep,
            max_strain: params.max_strain,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        // Positions upload as padded `vec4`; an empty position array is legal
        // (every edge then reads out of range and keeps its rest length), but a
        // zero-length storage buffer is invalid, so feed a one-element dummy.
        let packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let packed = if packed.is_empty() {
            Vec::from([[0.0f32; 4]])
        } else {
            packed
        };

        let rest_bytes = (edges.len() as u64) * 4;

        let params_buf = buffer::uniform(device, "prism_cloth_plasticity_params", &uniform);
        let positions_buf = buffer::storage_read(device, "prism_cloth_plasticity_pos", &packed);
        let edges_buf = buffer::storage_read(device, "prism_cloth_plasticity_edges", edges);
        let out_rest_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_plasticity_rest", rest_bytes);
        let modified_buf = buffer::storage_rw_zeroed(device, "prism_cloth_plasticity_modified", 4);
        let rest_stage = buffer::staging(device, "prism_cloth_plasticity_rest_stage", rest_bytes);
        let modified_stage = buffer::staging(device, "prism_cloth_plasticity_modified_stage", 4);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_plasticity_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &edges_buf),
                entry(3, &out_rest_buf),
                entry(4, &modified_buf),
            ],
        });

        let groups = u32::try_from(edges.len().div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_plasticity_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_plasticity_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        buffer::copy(&mut encoder, &out_rest_buf, &rest_stage, rest_bytes);
        buffer::copy(&mut encoder, &modified_buf, &modified_stage, 4);
        ctx.queue().submit([encoder.finish()]);

        let rest_lengths = buffer::read_back::<Real>(ctx, &rest_stage);
        let modified = buffer::read_back::<u32>(ctx, &modified_stage);
        (rest_lengths, modified.first().copied().unwrap_or(0))
    }
}
