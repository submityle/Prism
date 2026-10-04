//! `wgpu` compute twin of the closest-point-on-segment projection from the
//! `CPU` golden `prism_physics_core::soft::collision::body::closest_point_on_segment`.
//!
//! A capsule collider stores its axis as a segment `p0`..`p1`; the closest
//! point on that segment to a query position is the heart of capsule distance
//! queries, and clamping the projection parameter to `[0, 1]` turns the end
//! caps into hemispheres. A zero-length segment (`p0 == p1`) degenerates
//! gracefully to `p0`, so a collapsed capsule behaves like a sphere.
//!
//! This module ports that stateless, closed-form projection onto the device:
//! one thread resolves one query. [`GpuSoftClosestPointOnSegment`] is the
//! on-device twin; a passing real-device parity test is direct evidence the
//! kernel takes the same ordered degenerate guard and the same clamped
//! projection arithmetic the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! * `closest_point_on_segment` — the whole closed form: the axis
//!   `axis = p1 - p0`, the squared length `len_sq = dot(axis, axis)`, the
//!   degenerate guard `len_sq <= EPS_LEN_SQ` returning `p0`, the clamped
//!   projection parameter `t = clamp(dot(pos - p0, axis) / len_sq, 0, 1)` and
//!   the result `out = p0 + axis * t`.
//!
//! # Result encoding
//!
//! The reference returns the closest point only. The twin additionally reports
//! a `valid` flag: `1` when the segment had positive length and the projection
//! ran, `0` when the segment was degenerate (`len_sq <= EPS_LEN_SQ`) and the
//! result echoes `p0`.
//!
//! # Correctness model
//!
//! The closest point is continuous and checked with an absolute-or-relative
//! tolerance; `valid` is discrete and checked exactly. The division by `len_sq`
//! is the only conditioning-sensitive operation, so fixtures and the sweep keep
//! samples clear of the `len_sq = EPS_LEN_SQ` degenerate knee and of the
//! `t = 0` / `t = 1` clamp knees where a rounding tie could otherwise disagree;
//! dedicated named fixtures pin the degenerate and both clamp cases.
//!
//! # Degenerate inputs
//!
//! A segment whose squared length is at or below `EPS_LEN_SQ` has no defined
//! axis direction: the result is `p0` and `valid = 0`. The guard is an ordered
//! compare, so a `NaN` input routes to the degenerate branch rather than
//! producing a spurious projection, and the divisor is guarded with `select`
//! so the inert arm never forms an `inf` or `NaN`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, `clamp`, `dot`, division and `vec3` arithmetic — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no float modulo and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no bare float equality: every guard is an ordered compare,
//! so a `Metal` fast-math build that folds `x == x` to `true` cannot change the
//! degenerate decision.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` closest-point-on-segment kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `closest_point_on_segment` branch for branch; see
/// the module documentation for the algorithm.
const SOFT_CLOSEST_POINT_ON_SEGMENT_WGSL: &str = r#"
// Closest-point-on-segment twin: one thread per query projects a position onto
// the segment p0..p1 with the parameter clamped to [0, 1], mirroring
// closest_point_on_segment. A zero-length segment degenerates to p0.
// Provenance: 孪生自本仓 prism_physics_core::soft::collision::body；无第三方引擎源码或衍生代码。

// Squared-length floor below which the segment is treated as a single point;
// matches EPS_LEN_SQ used by the golden.
const EPS_LEN_SQ: f32 = 1e-12;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Segment start p0 (x, y, z).
    p0x: f32,
    p0y: f32,
    p0z: f32,
    // Segment end p1 (x, y, z).
    p1x: f32,
    p1y: f32,
    p1z: f32,
    // Query position pos (x, y, z).
    posx: f32,
    posy: f32,
    posz: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Hit {
    // Closest point on the segment (x, y, z).
    outx: f32,
    outy: f32,
    outz: f32,
    // 1 when the segment had positive length, 0 on the degenerate echo of p0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Hit>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let p0 = vec3<f32>(q.p0x, q.p0y, q.p0z);
    let p1 = vec3<f32>(q.p1x, q.p1y, q.p1z);
    let pos = vec3<f32>(q.posx, q.posy, q.posz);

    let axis = p1 - p0;
    let len_sq = dot(axis, axis);

    // Ordered guard: a NaN len_sq fails this and routes to the degenerate echo.
    let ok = len_sq > EPS_LEN_SQ;

    // Guarded divisor so the inert arm never forms inf/nan.
    let safe_len_sq = select(1.0, len_sq, ok);
    let raw_t = dot(pos - p0, axis) / safe_len_sq;
    let t = clamp(raw_t, 0.0, 1.0);
    let projected = p0 + axis * t;

    // On the degenerate branch the closest point echoes p0.
    let closest = select(p0, projected, ok);

    var hit: Hit;
    hit.outx = closest.x;
    hit.outy = closest.y;
    hit.outz = closest.z;
    hit.valid = select(0u, 1u, ok);

    results[idx] = hit;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
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
/// Nine payload words plus three padding words keep the stride a flat `48`
/// bytes, a multiple of `16` with every `vec3` flattened to scalars so no
/// vector-alignment surprise can appear.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    p0x: f32,
    p0y: f32,
    p0z: f32,
    p1x: f32,
    p1y: f32,
    p1z: f32,
    posx: f32,
    posy: f32,
    posz: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Hit` struct.
/// Three position words plus the `valid` word keep the stride a flat `16`
/// bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    outx: f32,
    outy: f32,
    outz: f32,
    valid: u32,
}

/// One closest-point query: the segment endpoints and the query position,
/// flattened to scalars so the `std430` stride stays an unambiguous flat
/// layout with no `vec3` field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftClosestPointOnSegmentQuery {
    /// Segment start `p0`, x component.
    pub p0x: f32,
    /// Segment start `p0`, y component.
    pub p0y: f32,
    /// Segment start `p0`, z component.
    pub p0z: f32,
    /// Segment end `p1`, x component.
    pub p1x: f32,
    /// Segment end `p1`, y component.
    pub p1y: f32,
    /// Segment end `p1`, z component.
    pub p1z: f32,
    /// Query position `pos`, x component.
    pub posx: f32,
    /// Query position `pos`, y component.
    pub posy: f32,
    /// Query position `pos`, z component.
    pub posz: f32,
}

impl SoftClosestPointOnSegmentQuery {
    /// Builds a closest-point query from the segment endpoints `p0`, `p1` and
    /// the query position `pos`.
    #[must_use]
    pub fn new(p0: [f32; 3], p1: [f32; 3], pos: [f32; 3]) -> SoftClosestPointOnSegmentQuery {
        SoftClosestPointOnSegmentQuery {
            p0x: p0[0],
            p0y: p0[1],
            p0z: p0[2],
            p1x: p1[0],
            p1y: p1[1],
            p1z: p1[2],
            posx: pos[0],
            posy: pos[1],
            posz: pos[2],
        }
    }
}

/// One resolved answer for a single query: the closest point on the segment and
/// whether the segment had positive length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftClosestPointOnSegmentResult {
    /// Closest point, x component.
    pub outx: f32,
    /// Closest point, y component.
    pub outy: f32,
    /// Closest point, z component.
    pub outz: f32,
    /// `1` when the segment had positive length, `0` on the degenerate echo of
    /// `p0`.
    pub valid: u32,
}

/// Encodes one [`SoftClosestPointOnSegmentQuery`] into its `std430`
/// [`GpuQuery`].
fn encode_query(q: &SoftClosestPointOnSegmentQuery) -> GpuQuery {
    GpuQuery {
        p0x: q.p0x,
        p0y: q.p0y,
        p0z: q.p0z,
        p1x: q.p1x,
        p1y: q.p1y,
        p1z: q.p1z,
        posx: q.posx,
        posy: q.posy,
        posz: q.posz,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SoftClosestPointOnSegmentResult`].
fn decode_result(raw: &GpuResult) -> SoftClosestPointOnSegmentResult {
    SoftClosestPointOnSegmentResult {
        outx: raw.outx,
        outy: raw.outy,
        outz: raw.outz,
        valid: raw.valid,
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

/// A compiled, reusable closest-point-on-segment compute pipeline, twinning the
/// `CPU` golden `closest_point_on_segment`.
pub struct GpuSoftClosestPointOnSegment {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftClosestPointOnSegment {
    /// Compiles the closest-point-on-segment kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftClosestPointOnSegment {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_closest_point_on_segment"),
            source: ShaderSource::Wgsl(SOFT_CLOSEST_POINT_ON_SEGMENT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_closest_point_on_segment_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_closest_point_on_segment_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_closest_point_on_segment_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftClosestPointOnSegment {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftClosestPointOnSegmentResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftClosestPointOnSegmentQuery],
    ) -> Vec<SoftClosestPointOnSegmentResult> {
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
            label: Some("prism_volumetric_soft_closest_point_on_segment_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_closest_point_on_segment_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_closest_point_on_segment_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_closest_point_on_segment_bind_group"),
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
            label: Some("prism_volumetric_soft_closest_point_on_segment_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_closest_point_on_segment_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_closest_point_on_segment_pass"),
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
