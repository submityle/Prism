//! Host orchestration of the tensor-product spline-surface compute kernel
//! (§24.1 twin; terrain detail, procedural geometry, path corridors, and
//! vehicle motion surfaces consume these patch evaluations on the GPU side).
//!
//! [`GpuSurface`] batch-evaluates one surface query over a slice of `(u, v)`
//! parameters against a shared 4x4 control grid **on a real device**, mirroring
//! the CPU reference family [`prism_math::curve::surface`]
//! ([`BezierPatch`](prism_math::curve::surface::BezierPatch) and
//! [`BSplineSurface`](prism_math::curve::surface::BSplineSurface)). The surface
//! math is **not** duplicated here: a single pipeline is composed at runtime by
//! prefixing the single-sourced fragments
//! [`WGSL_SPLINE`](prism_math::shader_mirror::WGSL_SPLINE) (whose Bézier basis
//! the surface reuses) and
//! [`WGSL_SURFACE`](prism_math::shader_mirror::WGSL_SURFACE) ahead of a thin
//! compute wrapper that calls the `prism_surface(op, grid, u, v)` dispatcher,
//! so the device helpers cannot silently drift from the CPU reference. The
//! query is selected per dispatch by the [`Surface`] op code in the uniform.
//!
//! # Parity contract (honest boundary)
//!
//! The `sample` and `tangent_*` queries are bare multiply/add polynomial
//! arithmetic and match the CPU path to a tight `1e-5` FMA tolerance. The
//! `normal` queries additionally take a cross product and normalize with the
//! same `length > 1e-20` guard as the CPU `normalize_or_zero`; the `rsqrt`
//! there is fast-math, so the normal contract is a slightly looser `5e-5`
//! tolerance evaluated on non-degenerate patches, rather than a bit contract.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::{WGSL_SPLINE, WGSL_SURFACE};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Threads per workgroup; a standard 1D batch tiling.
const WORKGROUP: u32 = 64;

/// Number of control points in the 4x4 grid.
const GRID: usize = 16;

/// Surface-query selector, matching the `prism_surface` dispatcher op codes in
/// [`WGSL_SURFACE`] and the CPU methods on
/// [`BezierPatch`](prism_math::curve::surface::BezierPatch) /
/// [`BSplineSurface`](prism_math::curve::surface::BSplineSurface).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Surface {
    /// Bicubic Bézier `S(u, v)`.
    BezierSample = 0,
    /// Bicubic Bézier `dS/du`.
    BezierTangentU = 1,
    /// Bicubic Bézier `dS/dv`.
    BezierTangentV = 2,
    /// Bicubic Bézier unit normal.
    BezierNormal = 3,
    /// Uniform cubic B-spline `S(u, v)`.
    BSplineSample = 4,
    /// Uniform cubic B-spline `dS/du`.
    BSplineTangentU = 5,
    /// Uniform cubic B-spline `dS/dv`.
    BSplineTangentV = 6,
    /// Uniform cubic B-spline unit normal.
    BSplineNormal = 7,
}

/// The 4x4 control grid uniform: 16 control points packed as `vec4` with the
/// value in `xyz`, indexed `p[row * 4 + col]` for `[u_row][v_col]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GridUniform {
    p: [[f32; 4]; GRID],
}

/// Compute wrapper that evaluates the selected surface query at each `(u, v)`.
const WRAP: &str = "\
struct Params { count: u32, op: u32 };\n\
struct Grid { p: array<vec4<f32>, 16> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<uniform> grid: Grid;\n\
@group(0) @binding(2) var<storage, read> src: array<vec2<f32>>;\n\
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let uv = src[i];\n\
    let r = prism_surface(params.op, grid.p, uv.x, uv.y);\n\
    dst[i] = vec4<f32>(r, 0.0);\n\
}\n";

/// Uniform block carrying the valid element count and the surface op code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    op: u32,
    _pad: [u32; 2],
}

/// Real-device twin of the tensor-product spline-surface family.
///
/// Build it once per device with [`GpuSurface::new`]; the single pipeline and
/// bind-group layout are created up front and reused across batches and ops.
pub struct GpuSurface {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuSurface {
    /// Compiles the surface pipeline on `ctx`'s device, embedding the
    /// single-sourced [`WGSL_SPLINE`] and [`WGSL_SURFACE`] fragments verbatim.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_surface_layout"),
            entries: &[
                buffer_layout(
                    0,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    1,
                    BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    3,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_surface_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut source = String::new();
        source.push_str(WGSL_SPLINE);
        source.push('\n');
        source.push_str(WGSL_SURFACE);
        source.push('\n');
        source.push_str(WRAP);
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_math_surface"),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_math_surface"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSurface { pipeline, layout }
    }

    /// Batch-evaluates `query` over `uvs` against the 4x4 `grid` on the device,
    /// mirroring the matching CPU method in [`prism_math::curve::surface`].
    ///
    /// `grid` is row-major (`grid[row * 4 + col]` for `[u_row][v_col]`), each a
    /// `[x, y, z]` control point. The returned vectors carry the result in the
    /// `xyz` lanes with a zero `w`, one entry per `(u, v)` sample.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        query: Surface,
        grid: &[[f32; 3]; GRID],
        uvs: &[[f32; 2]],
    ) -> Vec<[f32; 4]> {
        let n = uvs.len();
        if n == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let grid_uniform = GridUniform {
            p: core::array::from_fn(|k| [grid[k][0], grid[k][1], grid[k][2], 0.0]),
        };
        let grid_buf = buffer::uniform(device, "prism_math_surface_grid", &grid_uniform);
        let in_buf = buffer::storage_read(device, "prism_math_surface_in", uvs);
        let out_bytes = (n * size_of::<[f32; 4]>()) as u64;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_surface_out", out_bytes);
        let bind_group = self.bind(device, n, query, &grid_buf, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_surface_encoder"),
        });
        dispatch(&mut enc, &self.pipeline, &bind_group, n);
        let stage = buffer::staging(device, "prism_math_surface_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        buffer::read_back::<[f32; 4]>(ctx, &stage)
    }

    /// Builds the four-entry bind group (params, grid, input uv, output).
    fn bind(
        &self,
        device: &Device,
        count: usize,
        query: Surface,
        grid: &Buffer,
        input: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        let params = buffer::uniform(
            device,
            "prism_math_surface_params",
            &Params {
                count: count as u32,
                op: query as u32,
                _pad: [0; 2],
            },
        );
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_surface_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: grid.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }
}

/// Records a 1D batch dispatch covering `n` elements at [`WORKGROUP`] threads
/// per group.
fn dispatch(
    enc: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group: &BindGroup,
    n: usize,
) {
    let groups = (n as u32).div_ceil(WORKGROUP);
    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("prism_math_surface_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// One storage/uniform bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
