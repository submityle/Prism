//! `wgpu` compute twin of the virtual-geometry scanline fast path
//! ([`TriangleGradients::row_span`](prism_render_architecture::virtual_geometry::TriangleGradients::row_span)
//! and
//! [`TriangleGradients::depth_from_edges`](prism_render_architecture::virtual_geometry::TriangleGradients::depth_from_edges)).
//!
//! After the per-triangle gradient setup, the vis-buffer software rasterizer
//! walks each row incrementally. On a wide row it stops testing every pixel and
//! instead solves the three affine edge inequalities for the whole row in closed
//! form, yielding one contiguous coverage interval `[k_lo, k_hi]` in `+x` step
//! units, then walks only that span while stepping the depth plane by `z_x` per
//! column. The CPU golden
//! [`TriangleGradients::row_span`](prism_render_architecture::virtual_geometry::TriangleGradients::row_span)
//! owns that closed form and
//! [`TriangleGradients::depth_from_edges`](prism_render_architecture::virtual_geometry::TriangleGradients::depth_from_edges)
//! owns the depth the walk carries; [`GpuRowSpan`] is the on-device twin that
//! runs one thread per query and returns the same interval and depth, so the
//! twin's scanline fast path joins the already-landed vertex-projection,
//! gradient-setup and screen-fill twins into a full device-side vis-buffer
//! chain.
//!
//! # Coverage convention
//!
//! The reference marks a row covered only when the intersection of the three
//! half-lines `w_row[i] + k * w_x[i] >= 0` is non-empty within `[0, steps]`,
//! returning [`None`] otherwise; the returned bounds are clamped to that range.
//! The twin mirrors that: uncovered rows come back as [`None`] and covered rows
//! come back clamped, exactly as the reference derives them. Depth is
//! coverage-independent and always returned.
//!
//! # Portability
//!
//! The kernel is three affine comparisons, at most three reciprocal-free
//! divisions, a `ceil`/`floor` pair and one three-term dot product in the
//! portable core-`WGSL` subset, so it needs no optional device feature and runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! For gradients built on integer / dyadic pixel coordinates each `w_x` and
//! `w_row` value is exactly representable, and a lone division `-w0 / g` is
//! correctly rounded identically on `CPU` and `GPU` (no fused-multiply-add can
//! contract a single divide), so `ceil`/`floor` of the quotient agree and the
//! interval bounds are bit-exact. When the double area is an exact power of two
//! the depth weights `vertices_z` are exact, so each `vertices_z[i] * w[i]`
//! product and their sum are fma-immune and the depth is bit-exact too. The
//! parity test asserts field-for-field equality rather than a tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard affine edge-inequality span solve for software
//! rasterization plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One scanline query: the gradient fields the closed form needs plus the
/// per-row and per-depth edge-function triples.
///
/// `w_x` and `vertices_z` come straight from a
/// [`TriangleGradients`](prism_render_architecture::virtual_geometry::TriangleGradients)
/// setup; `w_row` is the edge triple at the row's first candidate pixel center,
/// `steps` is the number of `+x` unit steps to the last candidate, and
/// `depth_edges` is the edge triple at the pixel whose depth is wanted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScanQuery {
    /// Per-column (`+x`) edge-function increments (`TriangleGradients::w_x`).
    pub w_x: [f32; 3],
    /// Depth weights `(v0.z, v1.z, v2.z) / double_area` (`vertices_z`).
    pub vertices_z: [f32; 3],
    /// Edge-function triple at the row's first candidate pixel center.
    pub w_row: [f32; 3],
    /// Number of `+x` unit steps to the last candidate (`k in 0..=steps`).
    pub steps: u32,
    /// Edge-function triple at the depth query pixel.
    pub depth_edges: [f32; 3],
}

/// One scanline result: the closed-form coverage interval and the depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScanResult {
    /// Inclusive `[k_lo, k_hi]` coverage interval, or [`None`] if uncovered.
    pub span: Option<(u32, u32)>,
    /// `dot(vertices_z, depth_edges)` - the barycentric depth of the walk.
    pub depth: f32,
}

/// Uniform parameters for one scanline dispatch. Layout matches `Params` in
/// `shaders/row_span.wesl`: the query count then three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One scanline query upload. `52`-byte stride, matching `ScanInput` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuScanInput {
    wx0: f32,
    wx1: f32,
    wx2: f32,
    vz0: f32,
    vz1: f32,
    vz2: f32,
    wrow0: f32,
    wrow1: f32,
    wrow2: f32,
    steps: u32,
    dw0: f32,
    dw1: f32,
    dw2: f32,
}

/// One scanline result readback. `16`-byte stride, matching `ScanOutput` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuScanOutput {
    valid: u32,
    k_lo: u32,
    k_hi: u32,
    depth: f32,
}

/// A compiled, reusable scanline-span pipeline.
pub struct GpuRowSpan {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRowSpan {
    /// Compiles the scanline-span kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so it needs no
    /// optional device feature.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_row_span"),
            source: ShaderSource::Wgsl(include_str!("../shaders/row_span.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_row_span_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_row_span_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_row_span_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("scan"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRowSpan {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves each query's coverage interval and depth on-device, returning one
    /// [`ScanResult`] per query in input order.
    ///
    /// Each returned entry equals
    /// [`TriangleGradients::row_span`](prism_render_architecture::virtual_geometry::TriangleGradients::row_span)`(w_row, steps)`
    /// paired with
    /// [`TriangleGradients::depth_from_edges`](prism_render_architecture::virtual_geometry::TriangleGradients::depth_from_edges)`(depth_edges)`
    /// for the gradient the query carries. An empty `queries` slice yields an
    /// empty result - storage buffers cannot be zero-sized, so it is handled by
    /// an early return.
    #[must_use]
    pub fn scan(&self, ctx: &GpuContext, queries: &[ScanQuery]) -> Vec<ScanResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: u32::try_from(queries.len())
                .expect("query count must fit in u32 for the GPU dispatch"),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let gpu_inputs: Vec<GpuScanInput> = queries
            .iter()
            .map(|q| GpuScanInput {
                wx0: q.w_x[0],
                wx1: q.w_x[1],
                wx2: q.w_x[2],
                vz0: q.vertices_z[0],
                vz1: q.vertices_z[1],
                vz2: q.vertices_z[2],
                wrow0: q.w_row[0],
                wrow1: q.w_row[1],
                wrow2: q.w_row[2],
                steps: q.steps,
                dw0: q.depth_edges[0],
                dw1: q.depth_edges[1],
                dw2: q.depth_edges[2],
            })
            .collect();

        let out_bytes = (queries.len() as u64) * (size_of::<GpuScanOutput>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_row_span_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_row_span_inputs"),
            contents: bytemuck::cast_slice(&gpu_inputs),
            usage: BufferUsages::STORAGE,
        });
        let spans_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_row_span_spans"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let spans_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_row_span_spans_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_row_span_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: inputs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: spans_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_row_span_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_row_span_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&spans_buf, 0, &spans_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        spans_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = spans_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_spans = bytemuck::cast_slice::<u8, GpuScanOutput>(&view).to_vec();
        drop(view);
        spans_stage.unmap();
        debug_assert_eq!(gpu_spans.len(), queries.len());
        gpu_spans
            .into_iter()
            .map(|o| ScanResult {
                span: if o.valid == 0 {
                    None
                } else {
                    Some((o.k_lo, o.k_hi))
                },
                depth: o.depth,
            })
            .collect()
    }
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
