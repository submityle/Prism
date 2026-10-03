//! `wgpu` compute twin of the homogeneous `4x4` matrix-vector transform
//! [`Mat4::mul_vec4`](prism_render_architecture::motion::Mat4::mul_vec4) that the
//! motion subsystem uses to carry world positions through its
//! previous/current view-projection matrices.
//!
//! Reprojection projects a surface point through two transforms and takes the
//! screen-space difference; the elementary step underneath every such
//! projection is the column-major product `M * v`. This twin reproduces that
//! pure numeric step on the device, so a passing real-device parity test is
//! direct evidence the ported kernel transforms the same homogeneous points the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! One thread performs one transform. For the column-major matrix with columns
//! `c0, c1, c2, c3` and the homogeneous input `v`, the output is
//! [`Mat4::mul_vec4`](prism_render_architecture::motion::Mat4::mul_vec4)'s
//! `c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w`, using only `+` and `*`, so it
//! maps directly onto the portable core-`WGSL` subset.
//!
//! # What stays on the host
//!
//! The surrounding reprojection policy that frames this transform in
//! [`ReprojectionContext`](prism_render_architecture::motion::reproject::ReprojectionContext)
//! — the `4x4` inverse reconstruction, the perspective divide, and the
//! confidence logic — is not twinned here: it is stateful reprojection policy,
//! not the stateless matrix-vector arithmetic this module targets. The
//! [`Mat4::inverse`](prism_render_architecture::motion::Mat4::inverse) cofactor
//! expansion likewise stays on the host.
//!
//! # Correctness model
//!
//! Each output component is a sum of four products — no `sqrt`, no
//! transcendental — so the `CPU` and `GPU` agree to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+` and `*` on
//! `vec4<f32>` — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse
//! trigonometry, no `smoothstep`, and no `sqrt`. Each thread performs a fixed,
//! bounded sequence of arithmetic, so the kernel provably terminates. No
//! optional device feature is required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::Mat4::mul_vec4`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` homogeneous transform kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`Mat4::mul_vec4`](prism_render_architecture::motion::Mat4::mul_vec4)
/// column-major product; see the module documentation for the algorithm.
const MOTION_MAT4_TRANSFORM_WGSL: &str = r#"
// Homogeneous 4x4 matrix-vector transform twin: one thread computes one
// column-major product `M * v`, mirroring the CPU golden `Mat4::mul_vec4` with
// only + and * on vec4<f32>.
//
// Provenance: 孪生自本仓 prism_render_architecture::motion::Mat4::mul_vec4；无第三方引擎源码或衍生代码。

struct Params {
    // Number of transforms in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Column-major matrix columns.
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
    // Homogeneous input vector.
    v: vec4<f32>,
}

struct Result {
    // Transformed homogeneous vector.
    transformed: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Column-major M * v: scale each column by the matching input component and
// sum, matching the CPU golden's `cols[0]*v.x + cols[1]*v.y + ...`.
fn mul_vec4(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, v: vec4<f32>) -> vec4<f32> {
    return c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var out: Result;
    out.transformed = mul_vec4(q.c0, q.c1, q.c2, q.c3, q.v);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the operation count plus three pad words
/// to fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MOTION_MAT4_TRANSFORM_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid transforms in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one transform operation, matching the `WGSL`
/// `Query` struct: the four `16`-byte matrix columns followed by the input
/// vector.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Column `0` of the column-major matrix.
    c0: [f32; 4],
    /// Column `1` of the column-major matrix.
    c1: [f32; 4],
    /// Column `2` of the column-major matrix.
    c2: [f32; 4],
    /// Column `3` of the column-major matrix.
    c3: [f32; 4],
    /// Homogeneous input vector `(x, y, z, w)`.
    v: [f32; 4],
}

/// `repr(C)` `std430` layout of one transform result, matching the `WGSL`
/// `Result` struct: the transformed `16`-byte homogeneous vector.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Transformed homogeneous vector `(x, y, z, w)`.
    transformed: [f32; 4],
}

/// One homogeneous `4x4` matrix-vector transform to run on the device,
/// mirroring the golden
/// [`Mat4::mul_vec4`](prism_render_architecture::motion::Mat4::mul_vec4).
///
/// The `columns` are the column-major matrix columns `c0, c1, c2, c3`, each
/// `(x, y, z, w)`; `vector` is the homogeneous input `(x, y, z, w)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionMat4TransformQuery {
    /// The four column-major matrix columns, each `(x, y, z, w)`.
    pub columns: [[f32; 4]; 4],
    /// The homogeneous input vector `(x, y, z, w)`.
    pub vector: [f32; 4],
}

impl MotionMat4TransformQuery {
    /// Builds a transform query from the four column-major `columns` and the
    /// homogeneous input `vector`.
    #[must_use]
    pub const fn new(columns: [[f32; 4]; 4], vector: [f32; 4]) -> MotionMat4TransformQuery {
        MotionMat4TransformQuery { columns, vector }
    }
}

/// One resolved transform result, mirroring the golden
/// [`Mat4::mul_vec4`](prism_render_architecture::motion::Mat4::mul_vec4) output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionMat4TransformResult {
    /// The transformed homogeneous vector `(x, y, z, w)`.
    pub transformed: [f32; 4],
}

/// Encodes one [`MotionMat4TransformQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MotionMat4TransformQuery) -> GpuQuery {
    GpuQuery {
        c0: q.columns[0],
        c1: q.columns[1],
        c2: q.columns[2],
        c3: q.columns[3],
        v: q.vector,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`MotionMat4TransformResult`].
fn decode_result(raw: &GpuResult) -> MotionMat4TransformResult {
    MotionMat4TransformResult {
        transformed: raw.transformed,
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

/// A compiled, reusable homogeneous `4x4` matrix-vector transform compute
/// pipeline, twinning the numeric core of the `CPU` golden
/// [`Mat4::mul_vec4`](prism_render_architecture::motion::Mat4::mul_vec4).
pub struct GpuMotionMat4Transform {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMotionMat4Transform {
    /// Compiles the homogeneous transform kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMotionMat4Transform {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform"),
            source: ShaderSource::Wgsl(MOTION_MAT4_TRANSFORM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMotionMat4Transform {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every transform in `queries` and returns one
    /// [`MotionMat4TransformResult`] per input, in order.
    ///
    /// The outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MotionMat4TransformQuery],
    ) -> Vec<MotionMat4TransformResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_motion_mat4_transform_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_motion_mat4_transform_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per transform, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
