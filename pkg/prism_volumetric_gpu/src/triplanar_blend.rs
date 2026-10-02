//! `wgpu` compute twin of the triplanar-mapping blend-weight and composite
//! contract
//! ([`triplanar_blend`](prism_render_architecture::particle::triplanar_blend),
//! particle design §16-§21).
//!
//! The `CPU` golden
//! [`triplanar_blend`](prism_render_architecture::particle::triplanar_blend)
//! turns a world-space surface `normal` into three projection-plane weights
//! ([`TriplanarWeights::from_normal`](prism_render_architecture::particle::triplanar_blend::TriplanarWeights::from_normal))
//! and blends three already-fetched plane samples into one shaded value
//! ([`TriplanarWeights::blend_scalar`](prism_render_architecture::particle::triplanar_blend::TriplanarWeights::blend_scalar)
//! and
//! [`TriplanarWeights::blend_vec3`](prism_render_architecture::particle::triplanar_blend::TriplanarWeights::blend_vec3)).
//! [`GpuTriplanarBlend`] is the on-device twin: one thread solves one query, so
//! a passing real-device parity test is direct evidence the ported kernel
//! produces the same weights and the same blended samples the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For every query the kernel reproduces the three normalized weights, the
//! scalar blend of three plane scalars, and the component-wise `vec3` blend of
//! three plane `vec3` samples. The sharpen is the reference's integer-exponent
//! repeated multiply: the kernel loops `sharpness_exp` times accumulating a
//! product, never a `pow` call, so it reproduces the reference branch for
//! branch. The host guarantees `sharpness_exp` stays at or below a small bound
//! (`16`) so the loop is short and bounded.
//!
//! # Correctness model
//!
//! The weights and the two blends thread through multiplies, adds and one
//! guarded reciprocal, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous quantity,
//! tight enough to catch a genuinely wrong port (a dropped axis, a wrong
//! exponent count, a swapped sample) yet loose enough to admit legal fused
//! multiply-add contraction. The `sharpness_exp` loop bound is a `u32`, so the
//! iteration count matches exactly.
//!
//! # Degenerate inputs
//!
//! A degenerate normal (zero length, or whose sharpened magnitudes sum at or
//! below [`CMP_EPS`]) would divide the weights by a near-zero sum; the kernel
//! instead returns the uniform `1/3, 1/3, 1/3` split, matching the reference
//! fallback. `sharpness_exp == 0` makes every sharpened magnitude `1`, which is
//! the same uniform split for any normal. An empty query batch short-circuits
//! on the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`, a
//! bounded `u32` loop and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no inverse trigonometry and no `sqrt`, and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::triplanar_blend`；无第三方引擎源码或衍生代码。
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

/// Magnitude at or below which the sharpened-weight sum is treated as zero and
/// the kernel falls back to the uniform split. Matches the reference `CMP_EPS`;
/// the compare rule used instead of a bare `f32` `==`.
pub const CMP_EPS: f32 = 1.0e-6;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` triplanar-blend kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`triplanar_blend`](prism_render_architecture::particle::triplanar_blend)
/// branch for branch; see the module documentation for the algorithm.
const TRIPLANAR_BLEND_WGSL: &str = r#"
// Triplanar-blend twin: one thread per query reproduces the three normalized
// projection-plane weights, the scalar blend of three plane scalars, and the
// component-wise vec3 blend of three plane vec3 samples. It mirrors the CPU
// golden `particle::triplanar_blend` branch for branch, uses only the portable
// core-WGSL subset (abs, + - * /, a bounded u32 loop and unsigned index math),
// needs no sqrt and no transcendental call and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::triplanar_blend；无第三方
// 引擎源码或衍生代码。

// Magnitude at or below which the sharpened-weight sum is treated as zero and
// the uniform fallback is used. Matches the reference `CMP_EPS`; the compare
// rule used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // World-space surface normal, with the sharpness exponent in the pad lane.
    normal: vec3<f32>,
    sharpness_exp: u32,
    // Three plane scalar samples sx, sy, sz for the scalar blend; a pad lane
    // follows.
    plane_scalar: vec3<f32>,
    pad_s: f32,
    // Three plane vec3 samples (one per projection axis), each with a trailing
    // pad lane so the vec3 stays 16-byte aligned on device.
    plane_x: vec3<f32>,
    pad_x: f32,
    plane_y: vec3<f32>,
    pad_y: f32,
    plane_z: vec3<f32>,
    pad_z: f32,
}

struct Result {
    // Normalized triplanar weights, with the scalar blend packed into w's lane.
    weights: vec3<f32>,
    blend_scalar: f32,
    // Component-wise vec3 blend, with a trailing pad lane.
    blend_vec3: vec3<f32>,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Raises `base` to the non-negative integer power `exp` by repeated multiply;
// mirrors the reference private `int_pow`. `exp == 0` yields 1.0 for any base.
fn int_pow(base: f32, exp: u32) -> f32 {
    var acc: f32 = 1.0;
    for (var i: u32 = 0u; i < exp; i = i + 1u) {
        acc = acc * base;
    }
    return acc;
}

// Weighted sum of three scalars by the weight triple; mirrors the reference
// `blend_scalar`.
fn weighted_sum(w: vec3<f32>, sx: f32, sy: f32, sz: f32) -> f32 {
    return sx * w.x + sy * w.y + sz * w.z;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // from_normal: sharpen each |component| by the integer exponent, then
    // normalize so the three weights sum to one. A degenerate normal whose
    // sharpened sum is at or below CMP_EPS falls back to the uniform split,
    // matching the reference rather than dividing by ~zero.
    let ax = int_pow(abs(q.normal.x), q.sharpness_exp);
    let ay = int_pow(abs(q.normal.y), q.sharpness_exp);
    let az = int_pow(abs(q.normal.z), q.sharpness_exp);
    let sum = ax + ay + az;
    var weights: vec3<f32> = vec3<f32>(ax, ay, az);
    if (sum <= CMP_EPS) {
        let third = 1.0 / 3.0;
        weights = vec3<f32>(third, third, third);
    } else {
        let inv = 1.0 / sum;
        weights = vec3<f32>(ax * inv, ay * inv, az * inv);
    }

    // blend_scalar and blend_vec3: convex combinations under the weights. The
    // vec3 blend is the scalar blend applied per component.
    let blend_s = weighted_sum(
        weights,
        q.plane_scalar.x,
        q.plane_scalar.y,
        q.plane_scalar.z,
    );
    let blend_v = vec3<f32>(
        weighted_sum(weights, q.plane_x.x, q.plane_y.x, q.plane_z.x),
        weighted_sum(weights, q.plane_x.y, q.plane_y.y, q.plane_z.y),
        weighted_sum(weights, q.plane_x.z, q.plane_y.z, q.plane_z.z),
    );

    var out: Result;
    out.weights = weights;
    out.blend_scalar = blend_s;
    out.blend_vec3 = blend_v;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TRIPLANAR_BLEND_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Every `vec3` lane carries a trailing word so each stays `16`-byte aligned on
/// device; `sharpness_exp` reuses the normal's pad lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// World-space surface normal.
    normal: [f32; 3],
    /// Integer sharpen exponent, reusing the normal's pad lane.
    sharpness_exp: u32,
    /// Three plane scalar samples `sx`, `sy`, `sz`.
    plane_scalar: [f32; 3],
    /// Pad lane after `plane_scalar`.
    pad_s: f32,
    /// `X`-plane `vec3` sample.
    plane_x: [f32; 3],
    /// Pad lane after `plane_x`.
    pad_x: f32,
    /// `Y`-plane `vec3` sample.
    plane_y: [f32; 3],
    /// Pad lane after `plane_y`.
    pad_y: f32,
    /// `Z`-plane `vec3` sample.
    plane_z: [f32; 3],
    /// Pad lane after `plane_z`.
    pad_z: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Normalized triplanar weights.
    weights: [f32; 3],
    /// Scalar blend of the three plane scalars.
    blend_scalar: f32,
    /// Component-wise `vec3` blend of the three plane `vec3` samples.
    blend_vec3: [f32; 3],
    /// Padding lane.
    pad0: f32,
}

/// One query for the triplanar twin: a world-space `normal`, the integer
/// `sharpness_exp`, three plane scalars for the scalar blend and three plane
/// `vec3` samples for the component-wise blend.
///
/// The weights derive from the `normal` and `sharpness_exp` alone; the two
/// blends consume those weights, so a single query exercises
/// [`TriplanarWeights::from_normal`](prism_render_architecture::particle::triplanar_blend::TriplanarWeights::from_normal),
/// [`TriplanarWeights::blend_scalar`](prism_render_architecture::particle::triplanar_blend::TriplanarWeights::blend_scalar)
/// and
/// [`TriplanarWeights::blend_vec3`](prism_render_architecture::particle::triplanar_blend::TriplanarWeights::blend_vec3)
/// at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriplanarBlendQuery {
    /// World-space surface normal fed to the weight builder.
    pub normal: [f32; 3],
    /// Integer sharpen exponent (repeated multiply); the host keeps it at or
    /// below a small bound so the device loop is short.
    pub sharpness_exp: u32,
    /// Three plane scalar samples `sx`, `sy`, `sz` for the scalar blend.
    pub plane_scalar: [f32; 3],
    /// Three plane `vec3` samples, one per projection axis, for the
    /// component-wise blend.
    pub plane_vec3: [[f32; 3]; 3],
}

/// One resolved answer for a single query, mirroring every value the reference
/// reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriplanarBlendResult {
    /// Normalized triplanar weights, matching
    /// [`TriplanarWeights::from_normal`](prism_render_architecture::particle::triplanar_blend::TriplanarWeights::from_normal).
    pub weights: [f32; 3],
    /// Scalar blend, matching
    /// [`TriplanarWeights::blend_scalar`](prism_render_architecture::particle::triplanar_blend::TriplanarWeights::blend_scalar).
    pub blend_scalar: f32,
    /// Component-wise `vec3` blend, matching
    /// [`TriplanarWeights::blend_vec3`](prism_render_architecture::particle::triplanar_blend::TriplanarWeights::blend_vec3).
    pub blend_vec3: [f32; 3],
}

/// Encodes one [`TriplanarBlendQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &TriplanarBlendQuery) -> GpuQuery {
    GpuQuery {
        normal: q.normal,
        sharpness_exp: q.sharpness_exp,
        plane_scalar: q.plane_scalar,
        pad_s: 0.0,
        plane_x: q.plane_vec3[0],
        pad_x: 0.0,
        plane_y: q.plane_vec3[1],
        pad_y: 0.0,
        plane_z: q.plane_vec3[2],
        pad_z: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TriplanarBlendResult`].
fn decode_result(raw: &GpuResult) -> TriplanarBlendResult {
    TriplanarBlendResult {
        weights: raw.weights,
        blend_scalar: raw.blend_scalar,
        blend_vec3: raw.blend_vec3,
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

/// A compiled, reusable triplanar-blend compute pipeline, twinning the `CPU`
/// golden
/// [`triplanar_blend`](prism_render_architecture::particle::triplanar_blend).
pub struct GpuTriplanarBlend {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTriplanarBlend {
    /// Compiles the triplanar-blend kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTriplanarBlend {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_triplanar_blend"),
            source: ShaderSource::Wgsl(TRIPLANAR_BLEND_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_triplanar_blend_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_triplanar_blend_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_triplanar_blend_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTriplanarBlend {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`TriplanarBlendResult`]
    /// per input, in order.
    ///
    /// The weights and both blends match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TriplanarBlendQuery],
    ) -> Vec<TriplanarBlendResult> {
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
            label: Some("prism_volumetric_triplanar_blend_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_triplanar_blend_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_triplanar_blend_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_triplanar_blend_bind_group"),
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
            label: Some("prism_volumetric_triplanar_blend_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_triplanar_blend_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_triplanar_blend_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
