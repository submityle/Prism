//! `wgpu` compute twin of the `prism_math::geom::Aabb3` point and containment
//! closed forms, from the `CPU` golden `prism_math::geom::aabb`.
//!
//! Each thread resolves one query selected by a `u32` op id against an
//! axis-aligned bounding box (`AABB`), so a passing real-device parity test is
//! direct evidence the ported kernel computes the same answer the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! This module deliberately covers only the `Aabb3` operations not already
//! twinned by the `bounds` module (which handles surface area, center,
//! half-extent, union/merge, longest axis and point expansion). The seven ops
//! here, selected by `op`, are:
//!
//! * `0` volume: `e.x * e.y * e.z` with `e = max - min` (scalar).
//! * `1` contains point: inclusive `min <= p <= max` on every axis (flag).
//! * `2` contains box: the other box's `min` and `max` corners both inside
//!   (flag).
//! * `3` closest point: `clamp(p, min, max)` component-wise (vec3).
//! * `4` distance squared: `|p - closest_point(p)|^2` (scalar).
//! * `5` intersection: `lo = max(a.min, b.min)`, `hi = min(a.max, b.max)`;
//!   valid only when `lo <= hi` on every axis, in which case `lo`/`hi` are the
//!   overlap box, otherwise `valid = 0` and the corners are zero.
//! * `6` expand: `min - splat(amount)` / `max + splat(amount)` (two vec3; a
//!   negative amount shrinks).
//!
//! An op id of `7` or greater yields `valid = 0` and an all-zero result.
//!
//! # Correctness model
//!
//! The golden is entirely `f32`, so the host oracle is a straight `f32`
//! re-implementation. Each continuous output is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`); the discrete
//! `flag` and `valid` outputs are compared exactly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `select`, `+ - *` and unsigned index/op arithmetic — with no
//! `u64`/`i64`/`f64`, no `round`, no `f32` remainder, and no bare `f32`
//! equality (every box test is an ordered comparison; only the `u32` op id is
//! compared with `==`), so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::geom::aabb`；无第三方引擎源码或衍生代码。
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

/// Op id: box volume (scalar output).
pub const OP_VOLUME: u32 = 0;
/// Op id: inclusive point containment (flag output).
pub const OP_CONTAINS_POINT: u32 = 1;
/// Op id: inclusive box containment (flag output).
pub const OP_CONTAINS_AABB: u32 = 2;
/// Op id: closest point on the box (`v0` vec3 output).
pub const OP_CLOSEST_POINT: u32 = 3;
/// Op id: squared distance from a point to the box (scalar output).
pub const OP_DISTANCE_SQUARED: u32 = 4;
/// Op id: box intersection (`v0`/`v1` corners + `valid`).
pub const OP_INTERSECTION: u32 = 5;
/// Op id: uniform expand/shrink (`v0`/`v1` corners output).
pub const OP_EXPAND: u32 = 6;

/// The portable core-`WGSL` `AABB` point-query kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `Aabb3` methods; see the module documentation for the
/// closed forms.
const AABB_POINT_QUERY_WGSL: &str = r#"
// AABB point-query twin: one thread per query dispatches on a u32 op id and
// reproduces one Aabb3 operation. It uses only the portable core-WGSL subset
// (clamp, min, max, select, + - * plus unsigned index/op math) and has no loop,
// so it provably terminates. Every box test is an ordered comparison; only the
// u32 op id is compared with ==. There is no bare f32 equality.

struct Query {
    op: u32,
    a_min_x: f32,
    a_min_y: f32,
    a_min_z: f32,
    a_max_x: f32,
    a_max_y: f32,
    a_max_z: f32,
    b_min_x: f32,
    b_min_y: f32,
    b_min_z: f32,
    b_max_x: f32,
    b_max_y: f32,
    b_max_z: f32,
    p_x: f32,
    p_y: f32,
    p_z: f32,
    amount: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Res {
    scalar: f32,
    flag: u32,
    v0_x: f32,
    v0_y: f32,
    v0_z: f32,
    v1_x: f32,
    v1_y: f32,
    v1_z: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

fn contains(lo: vec3<f32>, hi: vec3<f32>, pt: vec3<f32>) -> bool {
    return (pt.x >= lo.x) && (pt.x <= hi.x)
        && (pt.y >= lo.y) && (pt.y <= hi.y)
        && (pt.z >= lo.z) && (pt.z <= hi.z);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let a_min = vec3<f32>(q.a_min_x, q.a_min_y, q.a_min_z);
    let a_max = vec3<f32>(q.a_max_x, q.a_max_y, q.a_max_z);
    let b_min = vec3<f32>(q.b_min_x, q.b_min_y, q.b_min_z);
    let b_max = vec3<f32>(q.b_max_x, q.b_max_y, q.b_max_z);
    let p = vec3<f32>(q.p_x, q.p_y, q.p_z);

    // Shared closed forms, all evaluated branch-free.
    let e = a_max - a_min;
    let cp = contains(a_min, a_max, p);
    let cb = contains(a_min, a_max, b_min) && contains(a_min, a_max, b_max);
    let cl = clamp(p, a_min, a_max);
    let d = p - cl;
    let dist_sq = d.x * d.x + d.y * d.y + d.z * d.z;
    let lo = max(a_min, b_min);
    let hi = min(a_max, b_max);
    let overlap = (lo.x <= hi.x) && (lo.y <= hi.y) && (lo.z <= hi.z);
    let amt = vec3<f32>(q.amount, q.amount, q.amount);
    let ex_min = a_min - amt;
    let ex_max = a_max + amt;

    var out: Res;
    out.scalar = 0.0;
    out.flag = 0u;
    out.v0_x = 0.0;
    out.v0_y = 0.0;
    out.v0_z = 0.0;
    out.v1_x = 0.0;
    out.v1_y = 0.0;
    out.v1_z = 0.0;
    out.valid = 0u;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    if (q.op == 0u) {
        out.scalar = e.x * e.y * e.z;
        out.valid = 1u;
    } else if (q.op == 1u) {
        out.flag = select(0u, 1u, cp);
        out.valid = 1u;
    } else if (q.op == 2u) {
        out.flag = select(0u, 1u, cb);
        out.valid = 1u;
    } else if (q.op == 3u) {
        out.v0_x = cl.x;
        out.v0_y = cl.y;
        out.v0_z = cl.z;
        out.valid = 1u;
    } else if (q.op == 4u) {
        out.scalar = dist_sq;
        out.valid = 1u;
    } else if (q.op == 5u) {
        out.valid = select(0u, 1u, overlap);
        out.v0_x = select(0.0, lo.x, overlap);
        out.v0_y = select(0.0, lo.y, overlap);
        out.v0_z = select(0.0, lo.z, overlap);
        out.v1_x = select(0.0, hi.x, overlap);
        out.v1_y = select(0.0, hi.y, overlap);
        out.v1_z = select(0.0, hi.z, overlap);
    } else if (q.op == 6u) {
        out.v0_x = ex_min.x;
        out.v0_y = ex_min.y;
        out.v0_z = ex_min.z;
        out.v1_x = ex_max.x;
        out.v1_y = ex_max.y;
        out.v1_z = ex_max.z;
        out.valid = 1u;
    }
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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
/// All vectors are flattened to scalar `f32` words so the layout carries no
/// `vec3` alignment holes; three trailing pad words round the stride to a
/// 16-byte multiple.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op: u32,
    a_min_x: f32,
    a_min_y: f32,
    a_min_z: f32,
    a_max_x: f32,
    a_max_y: f32,
    a_max_z: f32,
    b_min_x: f32,
    b_min_y: f32,
    b_min_z: f32,
    b_max_x: f32,
    b_max_y: f32,
    b_max_z: f32,
    p_x: f32,
    p_y: f32,
    p_z: f32,
    amount: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Res` struct:
/// a scalar, a flag, two flattened vec3 corners and a `valid` word, with three
/// trailing pad words rounding the stride to a 16-byte multiple.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    scalar: f32,
    flag: u32,
    v0_x: f32,
    v0_y: f32,
    v0_z: f32,
    v1_x: f32,
    v1_y: f32,
    v1_z: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One `AABB` point query: an op id selecting the operation, box `a`
/// (`a_min`/`a_max`), an optional second box `b` (`b_min`/`b_max`), a point
/// `p`, and a scalar `amount` for expand. Unused fields for a given op are
/// ignored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbPointQueryQuery {
    /// Operation selector; see the `OP_*` constants.
    pub op: u32,
    /// Minimum corner of box `a`.
    pub a_min: [f32; 3],
    /// Maximum corner of box `a`.
    pub a_max: [f32; 3],
    /// Minimum corner of the second box `b` (containment/intersection ops).
    pub b_min: [f32; 3],
    /// Maximum corner of the second box `b` (containment/intersection ops).
    pub b_max: [f32; 3],
    /// Query point (point/closest/distance ops).
    pub p: [f32; 3],
    /// Expand amount (positive grows, negative shrinks).
    pub amount: f32,
}

impl AabbPointQueryQuery {
    /// Builds a query with every optional field zeroed, carrying only the op id
    /// and box `a`.
    fn base(op: u32, a_min: [f32; 3], a_max: [f32; 3]) -> AabbPointQueryQuery {
        AabbPointQueryQuery {
            op,
            a_min,
            a_max,
            b_min: [0.0; 3],
            b_max: [0.0; 3],
            p: [0.0; 3],
            amount: 0.0,
        }
    }

    /// Box volume `e.x * e.y * e.z`.
    #[must_use]
    pub fn volume(a_min: [f32; 3], a_max: [f32; 3]) -> AabbPointQueryQuery {
        AabbPointQueryQuery::base(OP_VOLUME, a_min, a_max)
    }

    /// Inclusive point containment.
    #[must_use]
    pub fn contains_point(a_min: [f32; 3], a_max: [f32; 3], p: [f32; 3]) -> AabbPointQueryQuery {
        let mut q = AabbPointQueryQuery::base(OP_CONTAINS_POINT, a_min, a_max);
        q.p = p;
        q
    }

    /// Inclusive box containment of box `b` within box `a`.
    #[must_use]
    pub fn contains_aabb(
        a_min: [f32; 3],
        a_max: [f32; 3],
        b_min: [f32; 3],
        b_max: [f32; 3],
    ) -> AabbPointQueryQuery {
        let mut q = AabbPointQueryQuery::base(OP_CONTAINS_AABB, a_min, a_max);
        q.b_min = b_min;
        q.b_max = b_max;
        q
    }

    /// Closest point on box `a` to `p`.
    #[must_use]
    pub fn closest_point(a_min: [f32; 3], a_max: [f32; 3], p: [f32; 3]) -> AabbPointQueryQuery {
        let mut q = AabbPointQueryQuery::base(OP_CLOSEST_POINT, a_min, a_max);
        q.p = p;
        q
    }

    /// Squared distance from `p` to box `a`.
    #[must_use]
    pub fn distance_squared(a_min: [f32; 3], a_max: [f32; 3], p: [f32; 3]) -> AabbPointQueryQuery {
        let mut q = AabbPointQueryQuery::base(OP_DISTANCE_SQUARED, a_min, a_max);
        q.p = p;
        q
    }

    /// Intersection (overlap box) of boxes `a` and `b`.
    #[must_use]
    pub fn intersection(
        a_min: [f32; 3],
        a_max: [f32; 3],
        b_min: [f32; 3],
        b_max: [f32; 3],
    ) -> AabbPointQueryQuery {
        let mut q = AabbPointQueryQuery::base(OP_INTERSECTION, a_min, a_max);
        q.b_min = b_min;
        q.b_max = b_max;
        q
    }

    /// Uniform expand (or shrink, for a negative amount) of box `a`.
    #[must_use]
    pub fn expand(a_min: [f32; 3], a_max: [f32; 3], amount: f32) -> AabbPointQueryQuery {
        let mut q = AabbPointQueryQuery::base(OP_EXPAND, a_min, a_max);
        q.amount = amount;
        q
    }
}

/// One resolved answer for a single query. Which fields are meaningful depends
/// on the op; inactive fields are zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AabbPointQueryResult {
    /// Scalar output (volume, distance squared); `0` for other ops.
    pub scalar: f32,
    /// Discrete flag output (containment ops); `0` for other ops.
    pub flag: u32,
    /// First vector output (closest point, intersection `lo`, expanded `min`).
    pub v0: [f32; 3],
    /// Second vector output (intersection `hi`, expanded `max`).
    pub v1: [f32; 3],
    /// `1` when the op produced a defined answer; `0` for a disjoint
    /// intersection or an out-of-range op id.
    pub valid: u32,
}

/// Encodes one [`AabbPointQueryQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &AabbPointQueryQuery) -> GpuQuery {
    GpuQuery {
        op: q.op,
        a_min_x: q.a_min[0],
        a_min_y: q.a_min[1],
        a_min_z: q.a_min[2],
        a_max_x: q.a_max[0],
        a_max_y: q.a_max[1],
        a_max_z: q.a_max[2],
        b_min_x: q.b_min[0],
        b_min_y: q.b_min[1],
        b_min_z: q.b_min[2],
        b_max_x: q.b_max[0],
        b_max_y: q.b_max[1],
        b_max_z: q.b_max[2],
        p_x: q.p[0],
        p_y: q.p[1],
        p_z: q.p[2],
        amount: q.amount,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`AabbPointQueryResult`].
fn decode_result(raw: &GpuResult) -> AabbPointQueryResult {
    AabbPointQueryResult {
        scalar: raw.scalar,
        flag: raw.flag,
        v0: [raw.v0_x, raw.v0_y, raw.v0_z],
        v1: [raw.v1_x, raw.v1_y, raw.v1_z],
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

/// A compiled, reusable `AABB` point-query compute pipeline, twinning the `CPU`
/// golden `Aabb3` point and containment methods.
pub struct GpuAabbPointQuery {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAabbPointQuery {
    /// Compiles the `AABB` point-query kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAabbPointQuery {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_aabb_point_query_shader"),
            source: ShaderSource::Wgsl(AABB_POINT_QUERY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_aabb_point_query_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_aabb_point_query_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_aabb_point_query_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAabbPointQuery {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`AabbPointQueryResult`]
    /// per input, in order.
    ///
    /// The discrete `flag` and `valid` fields match the reference exactly and
    /// the continuous outputs to the module's tolerance. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[AabbPointQueryQuery],
    ) -> Vec<AabbPointQueryResult> {
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
            label: Some("prism_volumetric_aabb_point_query_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_aabb_point_query_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_aabb_point_query_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_aabb_point_query_bind_group"),
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
            label: Some("prism_volumetric_aabb_point_query_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_aabb_point_query_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_aabb_point_query_pass"),
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
