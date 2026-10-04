//! `wgpu` compute twin of the triangle plane-quadric constructor from the `CPU`
//! golden `prism_physics_core::collider::quadric::Quadric::from_triangle`.
//!
//! A triangle `abc` defines a plane; the quadric error metric of that plane is
//! the symmetric `4x4` matrix `[a b c d]ᵀ [a b c d]`, stored as its ten unique
//! upper-triangular coefficients. This kernel builds that quadric per triangle:
//!
//! ```text
//! normal = (b - a) x (c - a)
//! len    = length(normal)
//! if len <= f32::MIN_POSITIVE { return ZERO }   // degenerate triangle
//! n      = normal / len
//! d      = -dot(n, a)
//! (a2, ab, ac, ad, b2, bc, bd, c2, cd, d2) = from_plane(n, d)
//! ```
//!
//! # What is twinned
//!
//! The single stateless, no-`RNG` body of `Quadric::from_triangle`, including
//! its degenerate guard: a zero-area triangle (collinear or coincident
//! vertices) has a cross product of length at most `f32::MIN_POSITIVE` and
//! returns the all-zero quadric. The ten coefficients follow the golden
//! operator order of `from_plane`: `a2 = n.x², ab = n.x·n.y, ac = n.x·n.z, ad =
//! n.x·d, b2 = n.y², bc = n.y·n.z, bd = n.y·d, c2 = n.z², cd = n.z·d, d2 = d²`.
//!
//! # Correctness model
//!
//! Each coefficient threads through subtractions, a cross product, a square
//! root, a reciprocal and a product, so `CPU` and `GPU` are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate. The
//! parity test asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on each continuous coefficient; the discrete `valid`
//! flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! A degenerate triangle yields `len <= f32::MIN_POSITIVE`. The kernel guards
//! the reciprocal with an ordered compare so the normalise divisor is never
//! smaller than `f32::MIN_POSITIVE`, then selects the all-zero coefficient arm,
//! so no `inf`/`nan` can escape. `valid` is always `1`: the all-zero quadric is
//! a legitimate output, not a rejection. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — subtractions, a cross
//! product, `sqrt`, a guarded reciprocal and products — with no `pow`, no `sin`,
//! `cos`, `exp`, `log`, no `round`, no `f32` remainder, no `u64`/`i64`, no `f64`
//! and no bare `f32` equality, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric::Quadric::from_triangle`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` triangle plane-quadric kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `evaluate`
/// mirrors the `CPU` golden `Quadric::from_triangle`; see the module
/// documentation for the algorithm.
const QUADRIC_FROM_TRIANGLE_WGSL: &str = r#"
// Triangle plane-quadric twin: one thread per query builds the ten unique
// coefficients of the plane quadric of triangle abc. It uses only the portable
// core-WGSL subset (subtractions, a cross product, sqrt, a guarded reciprocal
// and products) with no u64/i64/f64, no pow and no transcendental, so it runs
// unmodified on Metal, Vulkan and DX12. A degenerate (zero-area) triangle takes
// the guarded all-zero arm, so no inf/nan escapes and valid is always 1.
//
// Provenance: 孪生自本仓 prism_physics_core::collider::quadric::Quadric::from_triangle；无第三方引擎源码或衍生代码。

// Smallest positive normal f32, matching the golden guard len <= f32::MIN_POSITIVE.
const MIN_POSITIVE: f32 = 1.17549435e-38;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the three triangle vertices, flattened to scalar lanes so the
// std430 layout never trips a 16-byte vector-alignment rule.
struct Query {
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
}

// One result: the ten quadric coefficients and a valid flag that is always 1.
struct Res {
    a2: f32,
    ab: f32,
    ac: f32,
    ad: f32,
    b2: f32,
    bc: f32,
    bd: f32,
    c2: f32,
    cd: f32,
    d2: f32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let edge1 = vec3<f32>(q.bx - q.ax, q.by - q.ay, q.bz - q.az);
    let edge2 = vec3<f32>(q.cx - q.ax, q.cy - q.ay, q.cz - q.az);
    let normal = vec3<f32>(
        edge1.y * edge2.z - edge1.z * edge2.y,
        edge1.z * edge2.x - edge1.x * edge2.z,
        edge1.x * edge2.y - edge1.y * edge2.x,
    );
    let len = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

    // Degenerate guard: a zero-area triangle has len <= MIN_POSITIVE and emits
    // the all-zero quadric. The reciprocal divisor is clamped away from zero so
    // the normalise never produces inf/nan even on the discarded arm.
    let degenerate = len <= MIN_POSITIVE;
    let safe_len = select(len, 1.0, degenerate);
    let inv = 1.0 / safe_len;
    let nx = normal.x * inv;
    let ny = normal.y * inv;
    let nz = normal.z * inv;
    let d = -(nx * q.ax + ny * q.ay + nz * q.az);

    var out: Res;
    out.a2 = select(nx * nx, 0.0, degenerate);
    out.ab = select(nx * ny, 0.0, degenerate);
    out.ac = select(nx * nz, 0.0, degenerate);
    out.ad = select(nx * d, 0.0, degenerate);
    out.b2 = select(ny * ny, 0.0, degenerate);
    out.bc = select(ny * nz, 0.0, degenerate);
    out.bd = select(ny * d, 0.0, degenerate);
    out.c2 = select(nz * nz, 0.0, degenerate);
    out.cd = select(nz * d, 0.0, degenerate);
    out.d2 = select(d * d, 0.0, degenerate);
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`QUADRIC_FROM_TRIANGLE_WGSL`].
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
/// The three vertices are flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Res` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    a2: f32,
    ab: f32,
    ac: f32,
    ad: f32,
    b2: f32,
    bc: f32,
    bd: f32,
    c2: f32,
    cd: f32,
    d2: f32,
    valid: u32,
}

/// One query for the triangle plane-quadric twin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricFromTriangleQuery {
    /// First triangle vertex `a`.
    pub a: [f32; 3],
    /// Second triangle vertex `b`.
    pub b: [f32; 3],
    /// Third triangle vertex `c`.
    pub c: [f32; 3],
}

impl QuadricFromTriangleQuery {
    /// Builds a query from the three triangle vertices.
    #[must_use]
    pub fn new(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> QuadricFromTriangleQuery {
        QuadricFromTriangleQuery { a, b, c }
    }
}

/// One resolved triangle plane-quadric: the ten unique coefficients plus a
/// `valid` flag that is always `1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricFromTriangleResult {
    /// The `(1, 1)` entry `n.x²`.
    pub a2: f32,
    /// The `(1, 2)` entry `n.x·n.y`.
    pub ab: f32,
    /// The `(1, 3)` entry `n.x·n.z`.
    pub ac: f32,
    /// The `(1, 4)` entry `n.x·d`.
    pub ad: f32,
    /// The `(2, 2)` entry `n.y²`.
    pub b2: f32,
    /// The `(2, 3)` entry `n.y·n.z`.
    pub bc: f32,
    /// The `(2, 4)` entry `n.y·d`.
    pub bd: f32,
    /// The `(3, 3)` entry `n.z²`.
    pub c2: f32,
    /// The `(3, 4)` entry `n.z·d`.
    pub cd: f32,
    /// The `(4, 4)` entry `d²`.
    pub d2: f32,
    /// Always `1`; the all-zero degenerate quadric is a legitimate output.
    pub valid: u32,
}

/// Encodes one [`QuadricFromTriangleQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QuadricFromTriangleQuery) -> GpuQuery {
    GpuQuery {
        ax: q.a[0],
        ay: q.a[1],
        az: q.a[2],
        bx: q.b[0],
        by: q.b[1],
        bz: q.b[2],
        cx: q.c[0],
        cy: q.c[1],
        cz: q.c[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`QuadricFromTriangleResult`].
fn decode_result(raw: &GpuResult) -> QuadricFromTriangleResult {
    QuadricFromTriangleResult {
        a2: raw.a2,
        ab: raw.ab,
        ac: raw.ac,
        ad: raw.ad,
        b2: raw.b2,
        bc: raw.bc,
        bd: raw.bd,
        c2: raw.c2,
        cd: raw.cd,
        d2: raw.d2,
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

/// A compiled, reusable triangle plane-quadric compute pipeline, twinning the
/// `CPU` golden `Quadric::from_triangle`.
pub struct GpuQuadricFromTriangle {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuadricFromTriangle {
    /// Compiles the triangle plane-quadric kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuadricFromTriangle {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quadric_from_triangle_module"),
            source: ShaderSource::Wgsl(QUADRIC_FROM_TRIANGLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quadric_from_triangle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quadric_from_triangle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quadric_from_triangle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuadricFromTriangle {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`QuadricFromTriangleResult`] per input, in order.
    ///
    /// Each continuous coefficient matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[QuadricFromTriangleQuery],
    ) -> Vec<QuadricFromTriangleResult> {
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
            label: Some("prism_volumetric_quadric_from_triangle_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quadric_from_triangle_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quadric_from_triangle_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quadric_from_triangle_bind_group"),
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
            label: Some("prism_volumetric_quadric_from_triangle_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quadric_from_triangle_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quadric_from_triangle_pass"),
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
