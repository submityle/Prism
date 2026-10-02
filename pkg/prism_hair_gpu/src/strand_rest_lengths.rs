//! `wgpu` compute twin of Prism's per-segment strand rest lengths
//! ([`strand_rest_lengths`](prism_render_architecture::hair::groom_import::strand_rest_lengths)).
//!
//! The guide XPBD solver stores one *rest length* per edge of a resampled
//! strand - the distance between consecutive control points - as the target its
//! edge-length constraint relaxes toward (design dynamics.rs). The `CPU` golden
//! [`strand_rest_lengths`](prism_render_architecture::hair::groom_import::strand_rest_lengths)
//! emits one length per edge for every strand with at least two control points
//! (`count - 1` entries) and an empty list for a strand shorter than two points.
//! This crate is the on-device twin that precomputes exactly that rest-length
//! buffer, one thread per edge over a flattened batch of strands, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same lengths as the reference - not merely that its shader compiles.
//!
//! # Relationship to the sibling metrics twin
//!
//! This is deliberately a different dispatch from
//! [`GpuStrandMetrics`](crate::strand_metrics::GpuStrandMetrics), which runs one
//! thread per strand and *sums* the segment lengths into a single scalar arc
//! length. Here each thread owns one edge and emits that edge's bare length, so
//! the output has `count - 1` entries per strand (not one) and the batch is
//! embarrassingly parallel over a flattened edge stream with no per-strand
//! reduction; the metrics twin never exposes the per-edge rest-length `Vec`.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairStrandRestLengths::eval`] takes a batch of strands (each a slice of
//! control points) and returns one `Vec<f32>` of edge rest lengths per strand,
//! in input order. The edge index is the invocation id (`@compute
//! @workgroup_size(64)`, a one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the flattened edge count
//! early-return. Each edge reads the global point index of its first endpoint
//! and takes the length of the vector to the next point, exactly as the golden
//! does.
//!
//! # Correctness model
//!
//! The edge vector is an exact subtraction and the length is `sqrt(dot(d, d))`,
//! matching [`Vec3::length`](prism_render_architecture::hair::dynamics::Vec3::length)
//! exactly, so the only `CPU` vs `GPU` divergence is legal fused-multiply-add
//! contraction in the squared length and the `sqrt`. Each length is matched
//! against a tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`), not bit-for-bit.
//! A strand shorter than two points yields no edges (an empty inner `Vec`),
//! reproduced exactly, so a kernel that mishandled the short-strand case still
//! fails the parity test.
//!
//! # Portability
//!
//! The kernel uses only subtract, `dot` and `sqrt` in the portable core-`WGSL`
//! subset - no `exp`, `pow` or optional device feature - so the twin runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard polyline per-segment lengths plus a `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::dynamics::Vec3;
use prism_render_architecture::hair::groom_import::strand_rest_lengths;

use crate::context::GpuContext;

/// Uniform parameters for one rest-length dispatch. Layout matches `Params` in
/// `shaders/strand_rest_lengths.wesl`: the flattened edge count, padded to one
/// `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    edge_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-edge strand rest-length pipeline.
pub struct GpuHairStrandRestLengths {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairStrandRestLengths {
    /// Compiles the rest-length shader and builds its pipeline against `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_strand_rest_lengths"),
            source: ShaderSource::Wgsl(include_str!("../shaders/strand_rest_lengths.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_strand_rest_lengths_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_strand_rest_lengths_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_strand_rest_lengths_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairStrandRestLengths {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes per-segment rest lengths for every strand, returning one
    /// `Vec<f32>` per input strand (same length as `strands`), each with one
    /// length per edge in root -> tip order.
    ///
    /// The lengths for strand `s` equal
    /// [`strand_rest_lengths`](prism_render_architecture::hair::groom_import::strand_rest_lengths)
    /// applied to `strands[s]`, to within the fused-multiply-add tolerance
    /// documented on this module (`abs_diff < 1e-4` or `rel_diff < 1e-3`): a
    /// strand with fewer than two control points yields an empty inner `Vec`,
    /// and a strand of `n` points yields `n - 1` edge lengths. A wholly empty
    /// batch (no strands, or every strand shorter than two points) is handled
    /// without a dispatch - storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, strands: &[&[[f32; 3]]]) -> Vec<Vec<f32>> {
        if strands.is_empty() {
            return Vec::new();
        }

        // Flatten points and build the per-edge first-endpoint index. The edge
        // index of strand `s` edge `k` points at the global index of the k-th
        // control point of that strand; the second endpoint is that index + 1.
        let mut flat_points: Vec<f32> = Vec::new();
        let mut edge_p0: Vec<u32> = Vec::new();
        for strand in strands {
            let base = (flat_points.len() / 3) as u32;
            for p in *strand {
                flat_points.extend_from_slice(p);
            }
            // `strand.len() - 1` edges when the strand has at least two points.
            let edges = strand.len().saturating_sub(1);
            for k in 0..edges {
                edge_p0.push(base + k as u32);
            }
        }

        let total_edges = edge_p0.len();
        if total_edges == 0 {
            // Every strand is too short to have an edge: no dispatch.
            return strands.iter().map(|_| Vec::new()).collect();
        }

        let device = ctx.device();
        let params = Params {
            edge_count: total_edges as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // One f32 (rest length) per edge.
        let out_bytes = (total_edges as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_rest_lengths_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_rest_lengths_points"),
            contents: bytemuck::cast_slice(&flat_points),
            usage: BufferUsages::STORAGE,
        });
        let edge_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_rest_lengths_edges"),
            contents: bytemuck::cast_slice(&edge_p0),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_rest_lengths_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_rest_lengths_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_strand_rest_lengths_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: edge_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_strand_rest_lengths_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_strand_rest_lengths_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (total_edges as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();
        debug_assert_eq!(flat.len(), total_edges);

        // Re-split the flat length stream back into per-strand vectors using the
        // same `n - 1` edge counts the flattening used.
        let mut cursor = 0usize;
        strands
            .iter()
            .map(|strand| {
                let edges = strand.len().saturating_sub(1);
                let mut out = Vec::with_capacity(edges);
                for _ in 0..edges {
                    out.push(flat[cursor]);
                    cursor += 1;
                }
                out
            })
            .collect()
    }
}

/// The `CPU` golden per-segment strand rest lengths, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
///
/// Runs [`strand_rest_lengths`](prism_render_architecture::hair::groom_import::strand_rest_lengths)
/// on `points` and returns the per-edge lengths in order.
#[must_use]
pub fn reference_strand_rest_lengths(points: &[[f32; 3]]) -> Vec<f32> {
    let strand: Vec<Vec3> = points.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
    strand_rest_lengths(&strand)
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
