//! `wgpu` compute twin of Prism's Projective-Dynamics local edge projection
//! ([`local_project_edge`](prism_render_architecture::hair::projective_global::local_project_edge)).
//!
//! Guide-strand dynamics relaxes its distance constraints with a local
//! Gauss-Seidel sweep, but Projective Dynamics (`Bouaziz` 2014) keeps the same
//! per-constraint *local* projection and couples the batch through one global
//! solve. The local step for an edge-length constraint is a pure, independent
//! per-edge map: recenter the edge on its midpoint and place the two endpoints
//! symmetrically about it so their separation equals the rest length. This twin
//! runs that local projection batch-wide on the device, one thread per edge —
//! the array-in/array-out form the global solve's right-hand-side assembly
//! consumes.
//!
//! # What the kernel evaluates
//!
//! [`GpuProjectiveEdge::eval`] takes a batch of `(xi, xj, rest)` edges and
//! returns one projected `(xi', xj')` endpoint pair per edge in input order. The
//! edge index is the invocation id (`@compute @workgroup_size(64)`, a
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! edge count early-return.
//!
//! # Correctness model
//!
//! The midpoint, the `0.5` halving and the endpoint add/sub are exact (`0.5` is
//! a power of two), but the projection scales the edge vector by `rest / length`
//! — a divide the `GPU` may round a few `ULP` differently from the scalar
//! reference. The twin is therefore matched against a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`), not bit-for-bit. Each endpoint and
//! the rest length are sanitised in-shader bit-faithfully to the golden
//! (`sanitize_finite` forces a non-finite coordinate to `0`; `sanitize_nonneg`
//! collapses a negative or non-finite rest length to `0`), and a degenerate
//! near-zero edge is split along `+x`, so a stray `NaN` can never poison a
//! projection.
//!
//! # Portability
//!
//! The kernel uses only `sqrt` (via `length`), a divide and single add/sub/mul
//! in the portable core-`WGSL` subset — no `exp`, `pow` or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Projective Dynamics / position-based distance projection
//! (`Bouaziz` 2014) plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

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

use prism_render_architecture::hair::projective_global::{local_project_edge, Vec3};

use crate::context::GpuContext;

/// Uniform parameters for one edge-projection dispatch. Layout matches `Params`
/// in `shaders/projective_edge.wesl`: the edge count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    edge_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-edge Projective-Dynamics local-projection pipeline.
pub struct GpuProjectiveEdge {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuProjectiveEdge {
    /// Compiles the per-edge local-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuProjectiveEdge {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_projective_edge"),
            source: ShaderSource::Wgsl(include_str!("../shaders/projective_edge.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_projective_edge_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_projective_edge_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_projective_edge_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuProjectiveEdge {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects each edge-length constraint, returning one projected endpoint
    /// pair `(xi', xj')` per input in order.
    ///
    /// The pair for edge `i` matches the `CPU` golden
    /// [`local_project_edge`](prism_render_architecture::hair::projective_global::local_project_edge)
    /// of `edges[i]` within the fma tolerance (`abs_diff < 1e-4` or
    /// `rel_diff < 1e-3`), with non-finite coordinates and negative/non-finite
    /// rest lengths sanitised to the same values as the reference. An empty batch
    /// yields an empty vector without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, edges: &[(Vec3, Vec3, f32)]) -> Vec<(Vec3, Vec3)> {
        let edge_count = edges.len();
        if edge_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            edge_count: edge_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Flatten each edge to 7 f32: xi.xyz, xj.xyz, rest. The raw authored
        // values are uploaded unchanged; the shader sanitises them bit-faithfully
        // to the golden.
        let mut flat: Vec<f32> = Vec::with_capacity(edge_count * 7);
        for &(xi, xj, rest) in edges {
            flat.push(xi.x);
            flat.push(xi.y);
            flat.push(xi.z);
            flat.push(xj.x);
            flat.push(xj.y);
            flat.push(xj.z);
            flat.push(rest);
        }

        // Output is 6 f32 (xi'.xyz, xj'.xyz) per edge.
        let out_len = edge_count * 6;
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_projective_edge_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let edges_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_projective_edge_edges"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_projective_edge_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_projective_edge_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_projective_edge_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: edges_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_projective_edge_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_projective_edge_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (edge_count as u32).div_ceil(64);
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
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out.chunks_exact(6)
            .map(|c| (Vec3::new(c[0], c[1], c[2]), Vec3::new(c[3], c[4], c[5])))
            .collect()
    }
}

/// The `CPU` golden local projection for one edge, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_project_edge(xi: Vec3, xj: Vec3, rest: f32) -> (Vec3, Vec3) {
    local_project_edge(xi, xj, rest)
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
