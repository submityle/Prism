//! `wgpu` compute twin of the three smooth constructive-solid-geometry
//! operators that additionally report a material blend factor, from the `CPU`
//! golden path
//! ([`smooth_union_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_union_blend),
//! [`smooth_intersection_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_intersection_blend)
//! and
//! [`smooth_subtraction_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_subtraction_blend)).
//!
//! Signed-distance modelling composes shapes with algebra: the `min` of two
//! fields is their union, the `max` their intersection, and a `max` against a
//! negated field a subtraction. The *smooth* variants replace the hard
//! `min`/`max` with Inigo Quilez's quadratic polynomial so neighbouring solids
//! merge with a controllable fillet radius `k` instead of a crease. Unlike the
//! plain smooth operators, each function here also returns a `blend` factor in
//! `[0, 1]` — the weight of the second operand `b` at the seam — so a shader
//! can interpolate per-solid material attributes (albedo, roughness, and so
//! on) across the rounded join. [`GpuSdfCsgBlend`] is the on-device twin: each
//! thread reads one triple `(a, b, k)` and writes all three operators' paired
//! `(distance, blend)` outputs, reproducing the reference closed forms with
//! only `abs`, `min`, `max`, `select`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfCsgBlendQuery`] — two distance values `a`/`b`
//! and the fillet radius `k` — and writes one [`SdfCsgBlendResult`] holding the
//! three `(distance, blend)` pairs `union_distance`/`union_blend`,
//! `intersection_distance`/`intersection_blend` and
//! `subtraction_distance`/`subtraction_blend`.
//!
//! For a positive `k` the shared polynomial is `h = max(k - |a - b|, 0) / k`,
//! `m = h * h * 0.5`, and the fillet offset `h * h * k * 0.25`. The union
//! subtracts that offset from `min(a, b)` and reports `m` when `a < b` else
//! `1 - m`; the intersection adds it to `max(a, b)` and reports `m` when
//! `a > b` else `1 - m`; the subtraction is the intersection evaluated against
//! the complement `-b`. A non-positive `k` falls back to the hard operator
//! with a hard `0`/`1` blend selection, matching the reference tie-break
//! conventions (`a <= b` for the union, `a >= b` for the intersection) exactly.
//!
//! # What stays on the host
//!
//! The planar/volumetric domain, the field-sampling that produces `a` and `b`
//! at a shared point, and the material interpolation that consumes `blend` all
//! stay on the host; the device sees only the three stateless, fixed-width
//! operator evaluations, one query at a time, so a storage buffer is never
//! zero-sized.
//!
//! # Correctness model
//!
//! The smooth-branch distance and blend are continuous polynomials in `a`, `b`
//! and `k`, so the `CPU` and `GPU` are not bit-exact: a `GPU` divide or
//! multiply may land a few units in the last place from the scalar reference.
//! The parity test asserts each distance and blend within `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` (relative error floored at `1e-6`), tight enough to
//! catch a genuinely wrong port yet loose enough to admit a legal last-place
//! difference. The one genuine hazard is a `k` approaching zero, which divides
//! the polynomial by a vanishing radius; fixtures and the randomized sweep keep
//! `k` in a safe positive band, and the hard `k <= 0` fallback is exercised by
//! dedicated fixtures where the result is an exact `min`/`max` with a hard
//! `0`/`1` blend.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `select`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, `sqrt`, no inverse trigonometry, no `round` and
//! no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_csg`；无第三方引擎源码或衍生代码。
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

/// The inline `WGSL` source of the smooth-`CSG` blend twin. The crate ships the
/// kernel as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`smooth_union_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_union_blend),
/// [`smooth_intersection_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_intersection_blend)
/// and
/// [`smooth_subtraction_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_subtraction_blend)
/// closed forms; see the module documentation for the algorithm.
const SDF_CSG_BLEND_WGSL: &str = r#"
// Smooth constructive-solid-geometry blend twin: one thread computes one query
// triple (a, b, k)'s smooth union, intersection and subtraction, each as a
// (distance, blend) pair, using only abs, min, max, select, products and
// quotients. The field sampling and material interpolation stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_csg；无第三方引擎
// 源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First signed-distance value.
    a: f32,
    // Second signed-distance value.
    b: f32,
    // Fillet radius; non-positive falls back to the hard operator.
    k: f32,
    pad0: f32,
}

struct Blend {
    // Smooth union distance and second-operand blend weight.
    union_distance: f32,
    union_blend: f32,
    // Smooth intersection distance and second-operand blend weight.
    intersection_distance: f32,
    intersection_blend: f32,
    // Smooth subtraction distance and carving-tool blend weight.
    subtraction_distance: f32,
    subtraction_blend: f32,
    // Padding words to a 32-byte stride.
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Blend>;

// Smooth union with fillet radius k, reporting the second-operand blend weight.
// Non-positive k falls back to the hard union with the a<=b tie-break.
fn smooth_union_blend(a: f32, b: f32, k: f32) -> vec2<f32> {
    if (k <= 0.0) {
        return vec2<f32>(min(a, b), select(1.0, 0.0, a <= b));
    }
    let h = max(k - abs(a - b), 0.0) / k;
    let m = h * h * 0.5;
    let distance = min(a, b) - h * h * k * 0.25;
    let blend = select(1.0 - m, m, a < b);
    return vec2<f32>(distance, blend);
}

// Smooth intersection with fillet radius k, the dual of the union. Non-positive
// k falls back to the hard intersection with the a>=b tie-break.
fn smooth_intersection_blend(a: f32, b: f32, k: f32) -> vec2<f32> {
    if (k <= 0.0) {
        return vec2<f32>(max(a, b), select(1.0, 0.0, a >= b));
    }
    let h = max(k - abs(a - b), 0.0) / k;
    let m = h * h * 0.5;
    let distance = max(a, b) + h * h * k * 0.25;
    let blend = select(1.0 - m, m, a > b);
    return vec2<f32>(distance, blend);
}

// Smooth subtraction: carve b out of a as a smooth intersection with the
// complement -b.
fn smooth_subtraction_blend(a: f32, b: f32, k: f32) -> vec2<f32> {
    return smooth_intersection_blend(a, -b, k);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let u = smooth_union_blend(q.a, q.b, q.k);
    let i = smooth_intersection_blend(q.a, q.b, q.k);
    let s = smooth_subtraction_blend(q.a, q.b, q.k);

    var out: Blend;
    out.union_distance = u.x;
    out.union_blend = u.y;
    out.intersection_distance = i.x;
    out.intersection_blend = i.y;
    out.subtraction_distance = s.x;
    out.subtraction_blend = s.y;
    out.pad0 = 0.0;
    out.pad1 = 0.0;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_CSG_BLEND_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the two distance values and the fillet radius plus one pad word packing to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First signed-distance value.
    a: f32,
    /// Second signed-distance value.
    b: f32,
    /// Fillet radius.
    k: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Blend` struct:
/// three `(distance, blend)` pairs plus two pad words packing to a `32`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Smooth union distance.
    union_distance: f32,
    /// Smooth union second-operand blend weight.
    union_blend: f32,
    /// Smooth intersection distance.
    intersection_distance: f32,
    /// Smooth intersection second-operand blend weight.
    intersection_blend: f32,
    /// Smooth subtraction distance.
    subtraction_distance: f32,
    /// Smooth subtraction carving-tool blend weight.
    subtraction_blend: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One query for the smooth-`CSG` blend twin: the two signed-distance values
/// `a` and `b` and the fillet radius `k`.
///
/// `a` and `b` are the signed distances of the two solids at a shared sample
/// point; `k` is the fillet radius shared by all three operators. A
/// non-positive `k` makes every operator fall back to its hard form with a
/// hard `0`/`1` blend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCsgBlendQuery {
    /// First signed-distance value.
    pub a: f32,
    /// Second signed-distance value.
    pub b: f32,
    /// Fillet radius shared by the three operators.
    pub k: f32,
}

impl SdfCsgBlendQuery {
    /// Builds a query from the two distance values and the fillet radius.
    #[must_use]
    pub const fn new(a: f32, b: f32, k: f32) -> SdfCsgBlendQuery {
        SdfCsgBlendQuery { a, b, k }
    }
}

/// One resolved query of the smooth-`CSG` blend twin: the three operators'
/// `(distance, blend)` pairs at `(a, b, k)`.
///
/// `union_distance`/`union_blend` are
/// [`smooth_union_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_union_blend);
/// `intersection_distance`/`intersection_blend` are
/// [`smooth_intersection_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_intersection_blend);
/// `subtraction_distance`/`subtraction_blend` are
/// [`smooth_subtraction_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_subtraction_blend).
/// Each `blend` lies in `[0, 1]` and is the weight of the second operand `b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCsgBlendResult {
    /// Smooth union distance.
    pub union_distance: f32,
    /// Smooth union second-operand blend weight.
    pub union_blend: f32,
    /// Smooth intersection distance.
    pub intersection_distance: f32,
    /// Smooth intersection second-operand blend weight.
    pub intersection_blend: f32,
    /// Smooth subtraction distance.
    pub subtraction_distance: f32,
    /// Smooth subtraction carving-tool blend weight.
    pub subtraction_blend: f32,
}

/// Encodes one [`SdfCsgBlendQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfCsgBlendQuery) -> GpuQuery {
    GpuQuery {
        a: q.a,
        b: q.b,
        k: q.k,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfCsgBlendResult`].
fn decode_result(raw: &GpuResult) -> SdfCsgBlendResult {
    SdfCsgBlendResult {
        union_distance: raw.union_distance,
        union_blend: raw.union_blend,
        intersection_distance: raw.intersection_distance,
        intersection_blend: raw.intersection_blend,
        subtraction_distance: raw.subtraction_distance,
        subtraction_blend: raw.subtraction_blend,
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

/// A compiled, reusable smooth-`CSG` blend compute pipeline, twinning the `CPU`
/// golden
/// [`smooth_union_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_union_blend),
/// [`smooth_intersection_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_intersection_blend)
/// and
/// [`smooth_subtraction_blend`](prism_render_architecture::ray_scene::sdf_csg::smooth_subtraction_blend).
pub struct GpuSdfCsgBlend {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfCsgBlend {
    /// Compiles the smooth-`CSG` blend compute pipeline on the given context.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfCsgBlend {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_csg_blend_module"),
            source: ShaderSource::Wgsl(SDF_CSG_BLEND_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_csg_blend_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_csg_blend_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_csg_blend_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfCsgBlend {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfCsgBlendResult`]
    /// per input, in order.
    ///
    /// The distances and blends match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfCsgBlendQuery],
    ) -> Vec<SdfCsgBlendResult> {
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
            label: Some("prism_volumetric_sdf_csg_blend_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_csg_blend_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_csg_blend_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_csg_blend_bind_group"),
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
            label: Some("prism_volumetric_sdf_csg_blend_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_csg_blend_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_csg_blend_pass"),
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
