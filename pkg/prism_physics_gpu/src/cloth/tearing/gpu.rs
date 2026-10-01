//! Real-device `wgpu` compute implementation of cloth tearing (break flags).
//!
//! [`GpuClothTearing`] compiles `shaders/cloth_tearing.wgsl` once and exposes a
//! single [`GpuClothTearing::solve`] that runs one break-flag pass: every
//! distance edge, in parallel, samples its endpoint separation from a read-only
//! position snapshot and writes its break flag (`1` = tears), while a single
//! atomic counter tallies the torn edges.
//!
//! This is the real-device twin of
//! [`cpu_cloth_tearing`](super::cpu::cpu_cloth_tearing); both delegate the break
//! decision to the same
//! [`prism_physics_core::tear_flag`](prism_physics_core::tear_flag) predicate, so
//! the flags match exactly (the comparison is integer-valued, with no low-bit
//! divergence) and the torn count matches exactly.
//!
//! Compacting the torn edges out of the constraint graph is left to the host:
//! this kernel reports only the flags the host consumes.
//!
//! # Provenance
//!
//! Removing a constraint whose strain exceeds a threshold is a standard,
//! publicly documented position-based-dynamics technique. No Unreal Engine
//! source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use prism_physics_core::TearingParams;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::ClothTearEdge;

/// Lanes per workgroup; must match `@workgroup_size(64)` in the kernel.
const WORKGROUP: usize = 64;

/// Uniform parameters shared with `Params` in `shaders/cloth_tearing.wgsl`
/// (16 bytes / 4 words).
///
/// The field order mirrors the `WGSL` struct exactly; reordering silently
/// corrupts the solve. `break_strain` is uploaded already sanitized so the
/// kernel can use it directly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of addressable particles (positions length).
    particle_count: u32,
    /// Number of edges in the slice.
    edge_count: u32,
    /// Tensile strain above which an edge tears (sanitized; a `NaN`/negative
    /// threshold arrives as `+inf`, so nothing tears).
    break_strain: f32,
    /// Padding to a 16-byte boundary.
    _pad0: f32,
}

/// A compiled, reusable `GPU` cloth tearing pipeline.
pub struct GpuClothTearing {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the pass reads and writes.
    layout: BindGroupLayout,
    /// The per-edge break-flag pass.
    pipeline: ComputePipeline,
}

impl GpuClothTearing {
    /// Compiles the cloth tearing kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothTearing {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_tearing"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_tearing.wgsl").into()),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_tearing_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, read),
                buffer_entry(2, read),
                buffer_entry(3, write),
                buffer_entry(4, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_tearing_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cloth_tearing_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothTearing {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs one tearing (break-flag) pass over `edges` and returns the per-edge
    /// break flags (parallel to `edges`, `1` = tears) and the number of torn
    /// edges.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_tearing`](super::cpu::cpu_cloth_tearing): each edge is
    /// independent, so one thread owns one edge with no colouring. An edge
    /// within the break threshold, with a degenerate rest length, or referencing
    /// an out-of-range particle keeps a `0` flag. An empty `edges` slice returns
    /// an empty vector and a zero count.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        edges: &[ClothTearEdge],
        params: TearingParams,
    ) -> (Vec<u32>, u32) {
        if edges.is_empty() {
            return (Vec::new(), 0);
        }

        let device = ctx.device();
        let uniform = Params {
            particle_count: u32::try_from(positions.len()).unwrap_or(u32::MAX),
            edge_count: u32::try_from(edges.len()).unwrap_or(u32::MAX),
            break_strain: params.sanitized().break_strain,
            _pad0: 0.0,
        };

        // Positions upload as padded `vec4`; an empty position array is legal
        // (every edge then reads out of range and keeps a `0` flag), but a
        // zero-length storage buffer is invalid, so feed a one-element dummy.
        let packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let packed = if packed.is_empty() {
            Vec::from([[0.0f32; 4]])
        } else {
            packed
        };

        let flags_bytes = (edges.len() as u64) * 4;

        let params_buf = buffer::uniform(device, "prism_cloth_tearing_params", &uniform);
        let positions_buf = buffer::storage_read(device, "prism_cloth_tearing_pos", &packed);
        let edges_buf = buffer::storage_read(device, "prism_cloth_tearing_edges", edges);
        let out_flags_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_tearing_flags", flags_bytes);
        let torn_buf = buffer::storage_rw_zeroed(device, "prism_cloth_tearing_torn", 4);
        let flags_stage = buffer::staging(device, "prism_cloth_tearing_flags_stage", flags_bytes);
        let torn_stage = buffer::staging(device, "prism_cloth_tearing_torn_stage", 4);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_tearing_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &edges_buf),
                entry(3, &out_flags_buf),
                entry(4, &torn_buf),
            ],
        });

        let groups = u32::try_from(edges.len().div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_tearing_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_tearing_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        buffer::copy(&mut encoder, &out_flags_buf, &flags_stage, flags_bytes);
        buffer::copy(&mut encoder, &torn_buf, &torn_stage, 4);
        ctx.queue().submit([encoder.finish()]);

        let flags = buffer::read_back::<u32>(ctx, &flags_stage);
        let torn = buffer::read_back::<u32>(ctx, &torn_stage);
        (flags, torn.first().copied().unwrap_or(0))
    }
}
