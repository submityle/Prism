//! Host orchestration of the GPU plane-query kernels (§24.1 twin: point/plane
//! distance, projection, and plane re-orientation on device).
//!
//! [`GpuPlaneQuery`] batch-evaluates the plane query family **on a real
//! device**, mirroring the CPU [`prism_math::Plane`] methods: the signed
//! distance, its magnitude, the orthogonal point projection, the flipped plane,
//! and the normalized plane. The math is **not** duplicated here: each kernel is
//! composed at runtime by prefixing the single-sourced fragment
//! [`WGSL_PLANE`](prism_math::shader_mirror::WGSL_PLANE) ahead of a thin compute
//! wrapper, so the device query cannot silently drift from the CPU reference —
//! it is literally the same text.
//!
//! # Tolerance / parity
//!
//! Signed distance, distance, flipped, and projection are single dot products or
//! sign flips; `normalized` adds a `length` divide. Metal compiles WGSL under
//! fast-math, so all but the exact sign flips are compared within a small
//! absolute+relative epsilon. It skips gracefully when no adapter is available;
//! the device-free WGSL parse gate and a real `GPU` cover the rest.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine or Unity source or
//! derived code.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_math::shader_mirror::WGSL_PLANE;
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

/// Point-query wrapper: one scalar (`prism_plane_signed_distance`) per element.
const WRAP_SIGNED: &str = "\
struct Params { count: u32 };\n\
struct PlaneIn { normal: vec4<f32>, point: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<PlaneIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = src[i];\n\
    dst[i] = prism_plane_signed_distance(s.normal.xyz, s.normal.w, s.point.xyz);\n\
}\n";

/// Distance wrapper: the absolute signed distance per element.
const WRAP_DISTANCE: &str = "\
struct Params { count: u32 };\n\
struct PlaneIn { normal: vec4<f32>, point: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<PlaneIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = src[i];\n\
    dst[i] = prism_plane_distance(s.normal.xyz, s.normal.w, s.point.xyz);\n\
}\n";

/// Projection wrapper: one `vec4` (projected `point.xyz`) per element.
const WRAP_PROJECT: &str = "\
struct Params { count: u32 };\n\
struct PlaneIn { normal: vec4<f32>, point: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: Params;\n\
@group(0) @binding(1) var<storage, read> src: array<PlaneIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = src[i];\n\
    let p = prism_plane_project_point(s.normal.xyz, s.normal.w, s.point.xyz);\n\
    dst[i] = vec4<f32>(p, 0.0);\n\
}\n";

/// Plane-op wrapper: one `vec4` (the re-oriented plane `normal.xyz` + `d`) per
/// element; `mode` selects `normalize` (0) or `flip` (1).
const WRAP_PLANE_OP: &str = "\
struct ModeParams { count: u32, mode: u32 };\n\
struct PlaneIn { normal: vec4<f32>, point: vec4<f32> };\n\
@group(0) @binding(0) var<uniform> params: ModeParams;\n\
@group(0) @binding(1) var<storage, read> src: array<PlaneIn>;\n\
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;\n\
@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    let i = gid.x;\n\
    if (i >= params.count) { return; }\n\
    let s = src[i];\n\
    if (params.mode == 0u) {\n\
        dst[i] = prism_plane_normalized(s.normal.xyz, s.normal.w);\n\
    } else {\n\
        dst[i] = prism_plane_flipped(s.normal.xyz, s.normal.w);\n\
    }\n\
}\n";

/// Uniform block: count only, padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CountParams {
    count: u32,
    _pad: [u32; 3],
}

/// Uniform block: count + mode, padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModeParams {
    count: u32,
    mode: u32,
    _pad: [u32; 2],
}

/// A plane `(normal.xyz, d)`; `normal` need not be unit length for the
/// distance/projection/flip ops, and `normalized` rescales it.
pub type GpuPlane = [f32; 4];

/// A plane/point pair input element matching the WGSL `PlaneIn` layout.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaneIn {
    normal: [f32; 4],
    point: [f32; 4],
}

/// Real-device batched plane-query kernels, composed from the single-sourced
/// [`WGSL_PLANE`] fragment.
pub struct GpuPlaneQuery {
    layout: BindGroupLayout,
    signed: ComputePipeline,
    distance: ComputePipeline,
    project: ComputePipeline,
    plane_op: ComputePipeline,
}

impl GpuPlaneQuery {
    /// Builds the four plane-query pipelines.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPlaneQuery {
        let device = ctx.device();
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_math_plane_layout"),
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                buffer_layout(
                    2,
                    BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_math_plane_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let build = |label: &str, wrapper: &str| -> ComputePipeline {
            let mut source = String::new();
            source.push_str(WGSL_PLANE);
            source.push('\n');
            source.push_str(wrapper);
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: Some(label),
                source: ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        GpuPlaneQuery {
            signed: build("prism_math_plane_signed", WRAP_SIGNED),
            distance: build("prism_math_plane_distance", WRAP_DISTANCE),
            project: build("prism_math_plane_project", WRAP_PROJECT),
            plane_op: build("prism_math_plane_op", WRAP_PLANE_OP),
            layout,
        }
    }

    /// Batch signed distance `dot(normal, point) + d`, mirroring
    /// [`prism_math::Plane::signed_distance`].
    #[must_use]
    pub fn signed_distance(
        &self,
        ctx: &GpuContext,
        planes: &[GpuPlane],
        points: &[[f32; 3]],
    ) -> Vec<f32> {
        self.run_scalar(ctx, &self.signed, &pack(planes, points))
    }

    /// Batch distance `|dot(normal, point) + d|`, mirroring
    /// [`prism_math::Plane::distance`].
    #[must_use]
    pub fn distance(&self, ctx: &GpuContext, planes: &[GpuPlane], points: &[[f32; 3]]) -> Vec<f32> {
        self.run_scalar(ctx, &self.distance, &pack(planes, points))
    }

    /// Batch orthogonal projection of each point onto its plane, mirroring
    /// [`prism_math::Plane::project_point`].
    #[must_use]
    pub fn project_points(
        &self,
        ctx: &GpuContext,
        planes: &[GpuPlane],
        points: &[[f32; 3]],
    ) -> Vec<[f32; 3]> {
        self.run_vec(ctx, &self.project, &pack(planes, points), 0)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }

    /// Batch normalized planes (`normal/|normal|`, `d/|normal|`), mirroring
    /// [`prism_math::Plane::normalized`].
    #[must_use]
    pub fn normalized(&self, ctx: &GpuContext, planes: &[GpuPlane]) -> Vec<GpuPlane> {
        self.run_plane_op(ctx, planes, 0)
    }

    /// Batch flipped planes (`-normal`, `-d`), mirroring
    /// [`prism_math::Plane::flipped`].
    #[must_use]
    pub fn flipped(&self, ctx: &GpuContext, planes: &[GpuPlane]) -> Vec<GpuPlane> {
        self.run_plane_op(ctx, planes, 1)
    }

    /// Runs a unary plane op over `planes` (the query `point` lane is unused).
    fn run_plane_op(&self, ctx: &GpuContext, planes: &[GpuPlane], mode: u32) -> Vec<GpuPlane> {
        let items = pack(planes, &vec![[0.0f32; 3]; planes.len()]);
        self.run_vec(ctx, &self.plane_op, &items, mode)
    }

    /// Uploads a batch, dispatches a scalar-output pipeline, reads `f32` back.
    fn run_scalar(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        items: &[PlaneIn],
    ) -> Vec<f32> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let out_bytes = (n * size_of::<f32>()) as u64;
        self.run(ctx, pipeline, items, 0, out_bytes, |stage| {
            buffer::read_back::<f32>(ctx, stage)
        })
    }

    /// Uploads a batch, dispatches a `vec4`-output pipeline (plane-op `mode`),
    /// reads `[f32; 4]` back.
    fn run_vec(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        items: &[PlaneIn],
        mode: u32,
    ) -> Vec<GpuPlane> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let out_bytes = (n * size_of::<GpuPlane>()) as u64;
        self.run(ctx, pipeline, items, mode, out_bytes, |stage| {
            buffer::read_back::<GpuPlane>(ctx, stage)
        })
    }

    /// Shared upload/dispatch/stage-copy for a batch of plane inputs, reading
    /// `out_bytes` back through `read`.
    fn run<S>(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        items: &[PlaneIn],
        mode: u32,
        out_bytes: u64,
        read: impl FnOnce(&Buffer) -> S,
    ) -> S {
        let device = ctx.device();
        let params_buf = buffer::uniform(
            device,
            "prism_math_plane_params",
            &ModeParams {
                count: items.len() as u32,
                mode,
                _pad: [0; 2],
            },
        );
        let in_buf = buffer::storage_read(device, "prism_math_plane_in", items);
        let out_buf = buffer::storage_rw_zeroed(device, "prism_math_plane_out", out_bytes);
        let bind_group = self.bind(device, &params_buf, &in_buf, &out_buf);

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_math_plane_encoder"),
        });
        dispatch(&mut enc, pipeline, &bind_group, items.len());
        let stage = buffer::staging(device, "prism_math_plane_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        read(&stage)
    }

    /// Builds the three-entry bind group (params uniform, input, output).
    fn bind(&self, device: &Device, params: &Buffer, input: &Buffer, output: &Buffer) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_math_plane_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }
}

/// Packs parallel plane/point slices into the aligned input layout.
fn pack(planes: &[GpuPlane], points: &[[f32; 3]]) -> Vec<PlaneIn> {
    assert_eq!(planes.len(), points.len(), "planes and points must align");
    planes
        .iter()
        .zip(points)
        .map(|(n, p)| PlaneIn {
            normal: *n,
            point: [p[0], p[1], p[2], 0.0],
        })
        .collect()
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
        label: Some("prism_math_plane_pass"),
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
