//! `wgpu` compute twin of the plane construction and projection primitives of
//! the `CPU` golden `prism_math::geom::plane`'s `Plane` (Hessian normal form).
//!
//! A plane is the set of points `p` with `normal · p + d == 0`, the `normal`
//! pointing toward the positive half-space. This module ports the stateless
//! construction and query closed forms of `Plane` onto the device, selected per
//! query by an unsigned `op_id`: one thread resolves one query, so a passing
//! real-device parity test is direct evidence the ported kernel reproduces the
//! reference values, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces one `Plane` operation:
//!
//! * `0` `from_point_normal(point, normal)`: `n = normalize(normal)`,
//!   `normal = n`, `d = -n · point`.
//! * `1` `from_points(a, b, c)`: `n = (b - a) × (c - a)`, `len = |n|`; if
//!   `len <= 1e-20` the points are collinear (`valid = 0`), otherwise
//!   `normal = n / len`, `d = -normal · a`.
//! * `2` `normalized(plane)`: `inv = 1 / |normal|`, `normal *= inv`, `d *= inv`.
//! * `3` `signed_distance(plane, p)`: `normal · p + d` (scalar).
//! * `4` `project_point(plane, p)`: `p - normal * signed_distance(p)` (vector).
//! * `5` `flipped(plane)`: `normal = -normal`, `d = -d`.
//! * `6` `distance(plane, p)`: `|signed_distance(p)|` (scalar).
//!
//! Vector-producing operations (`0`, `1`, `2`, `4`, `5`) write the plane
//! `normal` (or the projected point for `4`) to `vec_out`; the scalar `d` (or
//! the distance for `3`, `6`) goes to `scalar_out`. An `op_id` outside `0..=6`
//! is invalid (`valid = 0`, outputs zero).
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through operators and a square root a `GPU`
//! may evaluate slightly differently, so `CPU` and `GPU` are not bit-exact; the
//! three `vec_out` components and `scalar_out` are compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`). The discrete `valid` flag
//! is compared exactly, and the parity sweep keeps construction normals well
//! clear of zero length so the validity decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! The collinear `from_points` case, and a zero-length normal fed to the
//! normalizing operations (`0`, `2`), yield `valid = 0` with zeroed outputs; the
//! divisor is fed through a `select` guard so the un-taken branch never divides
//! by zero. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`,
//! `dot`, `cross`, `+ - * /`, `select` and unsigned index/`op_id` arithmetic —
//! with no transcendental, no `round`, no `f32` remainder. Length guards are
//! ordered `>` compares fed to `select`; branch selection uses unsigned `op_id`
//! equality; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::geom::plane`；无第三方引擎源码或衍生代码。
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

/// The plane-construction kernel, embedded inline so the twin ships as a single
/// source file. The single entry point `solve` mirrors the `CPU` golden
/// `Plane` operations, selected per query by `op_id`.
const PLANE_CONSTRUCT_WGSL: &str = r#"
// Plane-construction twin: one thread per query reproduces one Plane operation
// selected by an unsigned op_id. Length guards are ordered > compares fed to
// select; branch selection uses unsigned op_id equality; there is no bare f32
// equality anywhere.

const EPS_LEN: f32 = 1.0e-20;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation selector in 0..=6.
    op_id: u32,
    // Plane normal (op 2..=6) or input normal (op 0), stored x, y, z.
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    // Plane offset d (op 2..=6).
    d: f32,
    // Point a: op 0's point, op 1's first vertex.
    ax: f32,
    ay: f32,
    az: f32,
    // Point b: op 1's second vertex.
    bx: f32,
    by: f32,
    bz: f32,
    // Point c: op 1's third vertex.
    cx: f32,
    cy: f32,
    cz: f32,
    // Query point p (op 3, 4, 6).
    px: f32,
    py: f32,
    pz: f32,
    // Padding to an 80-byte stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Vector output (plane normal, or projected point for op 4), else zero.
    vx: f32,
    vy: f32,
    vz: f32,
    // Scalar output (plane d, signed distance or distance), else zero.
    scalar: f32,
    // 1 when the operation produced a defined result, else 0.
    valid: u32,
    // Padding to a 32-byte stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let op = q.op_id;
    let n_in = vec3<f32>(q.normal_x, q.normal_y, q.normal_z);
    let d_in = q.d;
    let a = vec3<f32>(q.ax, q.ay, q.az);
    let b = vec3<f32>(q.bx, q.by, q.bz);
    let c = vec3<f32>(q.cx, q.cy, q.cz);
    let p = vec3<f32>(q.px, q.py, q.pz);

    var vec_out = vec3<f32>(0.0, 0.0, 0.0);
    var scalar_out = 0.0;
    var valid = 0u;

    if (op == 0u) {
        // from_point_normal: normalize the input normal, d = -n . point.
        let len = sqrt(dot(n_in, n_in));
        let ok = len > EPS_LEN;
        let denom = select(1.0, len, ok);
        let n = n_in / denom;
        vec_out = n;
        scalar_out = -dot(n, a);
        valid = select(0u, 1u, ok);
    } else if (op == 1u) {
        // from_points: n = (b-a) x (c-a); collinear when |n| <= 1e-20.
        let e1 = b - a;
        let e2 = c - a;
        let n = cross(e1, e2);
        let len = sqrt(dot(n, n));
        let ok = len > EPS_LEN;
        let denom = select(1.0, len, ok);
        let nn = n / denom;
        vec_out = nn;
        scalar_out = -dot(nn, a);
        valid = select(0u, 1u, ok);
    } else if (op == 2u) {
        // normalized: scale normal and d by 1/|normal|.
        let len = sqrt(dot(n_in, n_in));
        let ok = len > EPS_LEN;
        let denom = select(1.0, len, ok);
        let inv = 1.0 / denom;
        vec_out = n_in * inv;
        scalar_out = d_in * inv;
        valid = select(0u, 1u, ok);
    } else if (op == 3u) {
        // signed_distance = normal . p + d.
        scalar_out = dot(n_in, p) + d_in;
        valid = 1u;
    } else if (op == 4u) {
        // project_point = p - normal * signed_distance.
        let sd = dot(n_in, p) + d_in;
        vec_out = p - n_in * sd;
        valid = 1u;
    } else if (op == 5u) {
        // flipped: negate normal and d.
        vec_out = -n_in;
        scalar_out = -d_in;
        valid = 1u;
    } else if (op == 6u) {
        // distance = |signed_distance|.
        scalar_out = abs(dot(n_in, p) + d_in);
        valid = 1u;
    }

    // Invalid (degenerate or out-of-range op_id) zeroes every output.
    let keep = valid != 0u;
    var out: Result;
    out.vx = select(0.0, vec_out.x, keep);
    out.vy = select(0.0, vec_out.y, keep);
    out.vz = select(0.0, vec_out.z, keep);
    out.scalar = select(0.0, scalar_out, keep);
    out.valid = valid;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the `op_id`, the plane/normal inputs, three construction points, the query
/// point and padding — `20` words (`80` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    op_id: u32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    d: f32,
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    px: f32,
    py: f32,
    pz: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the vector output, the scalar output, the validity flag and
/// padding — `8` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    vx: f32,
    vy: f32,
    vz: f32,
    scalar: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// The plane operation selected by a [`PlaneConstructQuery`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneOp {
    /// `from_point_normal(point, normal)`.
    FromPointNormal,
    /// `from_points(a, b, c)`.
    FromPoints,
    /// `normalized(plane)`.
    Normalized,
    /// `signed_distance(plane, p)`.
    SignedDistance,
    /// `project_point(plane, p)`.
    ProjectPoint,
    /// `flipped(plane)`.
    Flipped,
    /// `distance(plane, p)`.
    Distance,
}

impl PlaneOp {
    /// The unsigned selector matching the `WGSL` `op_id`.
    #[must_use]
    pub fn op_id(self) -> u32 {
        match self {
            PlaneOp::FromPointNormal => 0,
            PlaneOp::FromPoints => 1,
            PlaneOp::Normalized => 2,
            PlaneOp::SignedDistance => 3,
            PlaneOp::ProjectPoint => 4,
            PlaneOp::Flipped => 5,
            PlaneOp::Distance => 6,
        }
    }
}

/// One plane-construction query: the operation selector plus every operand slot
/// (unused slots are ignored by the selected operation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneConstructQuery {
    /// Operation selector `op_id`.
    pub op_id: u32,
    /// Plane normal (ops `2..=6`) or input normal (op `0`), stored `[x, y, z]`.
    pub normal: [f32; 3],
    /// Plane offset `d` (ops `2..=6`).
    pub d: f32,
    /// Point `a`: op `0`'s point, op `1`'s first vertex.
    pub a: [f32; 3],
    /// Point `b`: op `1`'s second vertex.
    pub b: [f32; 3],
    /// Point `c`: op `1`'s third vertex.
    pub c: [f32; 3],
    /// Query point `p` (ops `3`, `4`, `6`).
    pub p: [f32; 3],
}

impl PlaneConstructQuery {
    /// Builds a `from_point_normal` query.
    #[must_use]
    pub fn from_point_normal(point: [f32; 3], normal: [f32; 3]) -> PlaneConstructQuery {
        PlaneConstructQuery {
            op_id: PlaneOp::FromPointNormal.op_id(),
            normal,
            d: 0.0,
            a: point,
            b: [0.0; 3],
            c: [0.0; 3],
            p: [0.0; 3],
        }
    }

    /// Builds a `from_points` query.
    #[must_use]
    pub fn from_points(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> PlaneConstructQuery {
        PlaneConstructQuery {
            op_id: PlaneOp::FromPoints.op_id(),
            normal: [0.0; 3],
            d: 0.0,
            a,
            b,
            c,
            p: [0.0; 3],
        }
    }

    /// Builds a plane-only query (`normalized`, `flipped`) from `normal`/`d`.
    #[must_use]
    pub fn plane_only(op: PlaneOp, normal: [f32; 3], d: f32) -> PlaneConstructQuery {
        PlaneConstructQuery {
            op_id: op.op_id(),
            normal,
            d,
            a: [0.0; 3],
            b: [0.0; 3],
            c: [0.0; 3],
            p: [0.0; 3],
        }
    }

    /// Builds a plane-plus-point query (`signed_distance`, `project_point`,
    /// `distance`) from `normal`/`d` and the query point `p`.
    #[must_use]
    pub fn plane_point(op: PlaneOp, normal: [f32; 3], d: f32, p: [f32; 3]) -> PlaneConstructQuery {
        PlaneConstructQuery {
            op_id: op.op_id(),
            normal,
            d,
            a: [0.0; 3],
            b: [0.0; 3],
            c: [0.0; 3],
            p,
        }
    }
}

/// One resolved answer for a single query, mirroring the selected `Plane`
/// operation for that input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneConstructResult {
    /// Vector output (plane `normal`, or the projected point for
    /// `project_point`) when valid, else all zero.
    pub vec_out: [f32; 3],
    /// Scalar output (plane `d`, signed distance or unsigned distance) when
    /// valid, else zero.
    pub scalar: f32,
    /// `1` when the operation produced a defined result, else `0`.
    pub valid: u32,
}

/// Encodes one [`PlaneConstructQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &PlaneConstructQuery) -> GpuQuery {
    GpuQuery {
        op_id: q.op_id,
        normal_x: q.normal[0],
        normal_y: q.normal[1],
        normal_z: q.normal[2],
        d: q.d,
        ax: q.a[0],
        ay: q.a[1],
        az: q.a[2],
        bx: q.b[0],
        by: q.b[1],
        bz: q.b[2],
        cx: q.c[0],
        cy: q.c[1],
        cz: q.c[2],
        px: q.p[0],
        py: q.p[1],
        pz: q.p[2],
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`PlaneConstructResult`].
fn decode_result(raw: &GpuResult) -> PlaneConstructResult {
    PlaneConstructResult {
        vec_out: [raw.vx, raw.vy, raw.vz],
        scalar: raw.scalar,
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

/// A compiled, reusable plane-construction compute pipeline, twinning the `CPU`
/// golden `prism_math::geom::plane`'s `Plane`.
pub struct GpuPlaneConstruct {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPlaneConstruct {
    /// Compiles the plane-construction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPlaneConstruct {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_plane_construct"),
            source: ShaderSource::Wgsl(PLANE_CONSTRUCT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_plane_construct_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_plane_construct_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_plane_construct_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPlaneConstruct {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`PlaneConstructResult`]
    /// per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `vec_out`/`scalar`
    /// outputs to the module's tolerance. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[PlaneConstructQuery],
    ) -> Vec<PlaneConstructResult> {
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
            label: Some("prism_volumetric_plane_construct_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_plane_construct_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_plane_construct_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_plane_construct_bind_group"),
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
            label: Some("prism_volumetric_plane_construct_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_plane_construct_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_plane_construct_pass"),
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
