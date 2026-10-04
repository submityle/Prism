//! `wgpu` compute twin of the single-tetrahedron shape metrics from the `CPU`
//! golden `prism_physics_core::collider::tet_quality`'s `tet_quality`.
//!
//! A tetrahedron `(v0, v1, v2, v3)` is graded by its signed volume, inversion
//! and degeneracy flags, the normalised radius ratio `3 * r_in / r_circ`, the
//! smallest and largest of its six dihedral angles, and its shortest and
//! longest edge. This module ports that single stateless closed form onto the
//! device: one thread resolves one tetrahedron, so a passing real-device parity
//! test is direct evidence the ported kernel computes the same descriptors the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `tet_quality` for one tetrahedron:
//!
//! * `volume = (v1-v0) . ((v2-v0) x (v3-v0)) / 6`, and `inverted` is true when
//!   that signed volume is not strictly positive (which also captures `NaN`).
//! * The six edge lengths give `min_edge` and `max_edge`.
//! * The four outward face normals give the six dihedral angles (face pairs
//!   `(0,1) (0,2) (0,3) (1,2) (1,3) (2,3)`); their extremes are
//!   `min_dihedral_deg` and `max_dihedral_deg`, each `pi - acos(n_i . n_j)`
//!   converted to degrees.
//! * The circumradius is solved from the `3x3` edge system; when `|det|` is
//!   below `1e-12` or the circumradius / total face area is non-positive the
//!   element is `degenerate` and `radius_ratio = 0`. Otherwise
//!   `radius_ratio = clamp(3 * r_in / r_circ, 0, 1)` with
//!   `r_in = 3 * |volume| / area_total`.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through operators a `GPU` may contract, so
//! `CPU` and `GPU` are not necessarily bit-exact; the continuous scalars are
//! compared with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`). The dihedral
//! angle is routed through `acos`, so the fixtures keep every angle inside
//! `[25, 150]` degrees where `acos` is well conditioned. The discrete
//! `inverted` and `degenerate` flags are compared exactly; the fixtures keep
//! well-conditioned tetrahedra away from the `radius_ratio` clamp knees and the
//! `|det|` degeneracy edge so neither flag can be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! A numerically flat element (edge-matrix `|det| < 1e-12`, or a non-positive
//! circumradius / face area) is `degenerate` with `radius_ratio = 0`. Any
//! non-finite input coordinate short-circuits — on both the device and the host
//! oracle — to `inverted = 1`, `degenerate = 1`, `radius_ratio = 0` with the
//! continuous fields zeroed, so the two sides agree deterministically rather
//! than racing propagated `NaN`. Every divisor (`det`, `r_circ`, `area_total`,
//! and the normal length) is fed through a `select` guard so an un-taken branch
//! never divides by zero. An empty query batch short-circuits on the host with
//! no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sqrt`
//! (through `length`), `acos`, `cross`, `dot`, `clamp`, `min`, `max`, `select`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no `f32` remainder, and no `u64/i64/f64`, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with
//! the ordered compare `abs(x) < 3.0e38` (which rejects both infinities and
//! `NaN`) rather than a bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_quality`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` tetrahedron-quality kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `tet_quality`; see the module documentation for the closed
/// form.
const TET_QUALITY_METRICS_WGSL: &str = r#"
// Tetrahedron-quality twin: one thread per query reproduces tet_quality using
// only the portable core-WGSL subset. Non-finite inputs are rejected with the
// ordered compare abs < 3.0e38 and short-circuit to the degenerate/inverted
// result; every divisor is fed through a select guard so no un-taken branch
// divides by zero.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    v0: vec3<f32>,
    pad0: f32,
    v1: vec3<f32>,
    pad1: f32,
    v2: vec3<f32>,
    pad2: f32,
    v3: vec3<f32>,
    pad3: f32,
}

struct Result {
    volume: f32,
    radius_ratio: f32,
    min_dihedral_deg: f32,
    max_dihedral_deg: f32,
    min_edge: f32,
    max_edge: f32,
    inverted: u32,
    degenerate: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const DEGENERATE_DET: f32 = 1.0e-12;
const PI: f32 = 3.1415927;

// Outward unit normal of face (a, b, c) pointing away from apex. Returns the
// zero vector for a degenerate (near-zero-area) face.
fn outward_face_normal(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, apex: vec3<f32>) -> vec3<f32> {
    let nvec = cross(b - a, c - a);
    let len = length(nvec);
    let ok = len > 1.0e-20;
    let safe_len = select(1.0, len, ok);
    let unit = nvec / safe_len;
    let outward = dot(unit, apex - a) > 0.0;
    let oriented = select(unit, -unit, outward);
    return select(vec3<f32>(0.0, 0.0, 0.0), oriented, ok);
}

// Dihedral angle in degrees between two outward face normals meeting at an edge.
fn dihedral_deg(n0: vec3<f32>, n1: vec3<f32>) -> f32 {
    let cosv = clamp(dot(n0, n1), -1.0, 1.0);
    let angle = PI - acos(cosv);
    return angle * 180.0 / PI;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let v0 = q.v0;
    let v1 = q.v1;
    let v2 = q.v2;
    let v3 = q.v3;

    // Non-finite detection via ordered compare (rejects +/-inf and NaN since
    // every comparison with NaN is false).
    let finite_in =
        (abs(v0.x) < FINITE_LIMIT) && (abs(v0.y) < FINITE_LIMIT) && (abs(v0.z) < FINITE_LIMIT) &&
        (abs(v1.x) < FINITE_LIMIT) && (abs(v1.y) < FINITE_LIMIT) && (abs(v1.z) < FINITE_LIMIT) &&
        (abs(v2.x) < FINITE_LIMIT) && (abs(v2.y) < FINITE_LIMIT) && (abs(v2.z) < FINITE_LIMIT) &&
        (abs(v3.x) < FINITE_LIMIT) && (abs(v3.y) < FINITE_LIMIT) && (abs(v3.z) < FINITE_LIMIT);

    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let e3 = v3 - v0;

    let vol6 = dot(e1, cross(e2, e3));
    let volume = vol6 / 6.0;
    let inverted = !(volume > 0.0);

    // Six edge lengths.
    let d10 = length(e1);
    let d20 = length(e2);
    let d30 = length(e3);
    let d21 = length(v2 - v1);
    let d31 = length(v3 - v1);
    let d32 = length(v3 - v2);
    let min_edge = min(min(min(d10, d20), min(d30, d21)), min(d31, d32));
    let max_edge = max(max(max(d10, d20), max(d30, d21)), max(d31, d32));

    // Four face areas and their sum.
    let a0 = 0.5 * length(cross(v2 - v1, v3 - v1));
    let a1 = 0.5 * length(cross(e2, e3));
    let a2 = 0.5 * length(cross(e1, e3));
    let a3 = 0.5 * length(cross(e1, e2));
    let area_total = a0 + a1 + a2 + a3;

    // Four outward normals and the six dihedral angles.
    let n0 = outward_face_normal(v1, v2, v3, v0);
    let n1 = outward_face_normal(v0, v2, v3, v1);
    let n2 = outward_face_normal(v0, v1, v3, v2);
    let n3 = outward_face_normal(v0, v1, v2, v3);
    let da = dihedral_deg(n0, n1);
    let db = dihedral_deg(n0, n2);
    let dc = dihedral_deg(n0, n3);
    let dd = dihedral_deg(n1, n2);
    let de = dihedral_deg(n1, n3);
    let df = dihedral_deg(n2, n3);
    let min_dih = min(min(min(da, db), min(dc, dd)), min(de, df));
    let max_dih = max(max(max(da, db), max(dc, dd)), max(de, df));

    // Circumradius via the 3x3 edge system (rows e1,e2,e3), solved with glam's
    // cofactor inverse so the host oracle and the device agree.
    let ax = vec3<f32>(e1.x, e2.x, e3.x);
    let ay = vec3<f32>(e1.y, e2.y, e3.y);
    let az = vec3<f32>(e1.z, e2.z, e3.z);
    let tmp0 = cross(ay, az);
    let tmp1 = cross(az, ax);
    let tmp2 = cross(ax, ay);
    let det = dot(az, tmp2);
    let det_ok = abs(det) >= DEGENERATE_DET;
    let inv_det = 1.0 / select(1.0, det, det_ok);
    let inv_x = vec3<f32>(tmp0.x, tmp1.x, tmp2.x) * inv_det;
    let inv_y = vec3<f32>(tmp0.y, tmp1.y, tmp2.y) * inv_det;
    let inv_z = vec3<f32>(tmp0.z, tmp1.z, tmp2.z) * inv_det;
    let bb = vec3<f32>(0.5 * dot(e1, e1), 0.5 * dot(e2, e2), 0.5 * dot(e3, e3));
    let centre = inv_x * bb.x + inv_y * bb.y + inv_z * bb.z;
    let r_circ = length(centre);

    let nondegen = det_ok && (r_circ > 0.0) && (area_total > 0.0);
    let r_in = 3.0 * abs(volume) / select(1.0, area_total, area_total > 0.0);
    let ratio_raw = 3.0 * r_in / select(1.0, r_circ, r_circ > 0.0);
    let ratio = clamp(ratio_raw, 0.0, 1.0);
    let radius_ratio = select(0.0, ratio, nondegen);

    var out: Result;
    out.volume = select(0.0, volume, finite_in);
    out.radius_ratio = select(0.0, radius_ratio, finite_in);
    out.min_dihedral_deg = select(0.0, min_dih, finite_in);
    out.max_dihedral_deg = select(0.0, max_dih, finite_in);
    out.min_edge = select(0.0, min_edge, finite_in);
    out.max_edge = select(0.0, max_edge, finite_in);
    out.inverted = select(1u, select(0u, 1u, inverted), finite_in);
    out.degenerate = select(1u, select(0u, 1u, !nondegen), finite_in);
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
/// Each vertex is a `vec3<f32>` padded to a 16-byte-aligned slot, so the whole
/// query is `64` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    v0: [f32; 3],
    pad0: f32,
    v1: [f32; 3],
    pad1: f32,
    v2: [f32; 3],
    pad2: f32,
    v3: [f32; 3],
    pad3: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: six continuous scalars plus two `u32` flags — `32` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    volume: f32,
    radius_ratio: f32,
    min_dihedral_deg: f32,
    max_dihedral_deg: f32,
    min_edge: f32,
    max_edge: f32,
    inverted: u32,
    degenerate: u32,
}

/// One tetrahedron-quality query: the four vertices `(v0, v1, v2, v3)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetQualityMetricsQuery {
    /// First vertex.
    pub v0: [f32; 3],
    /// Second vertex.
    pub v1: [f32; 3],
    /// Third vertex.
    pub v2: [f32; 3],
    /// Fourth vertex.
    pub v3: [f32; 3],
}

impl TetQualityMetricsQuery {
    /// Builds a query from the four tetrahedron vertices.
    #[must_use]
    pub fn new(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], v3: [f32; 3]) -> TetQualityMetricsQuery {
        TetQualityMetricsQuery { v0, v1, v2, v3 }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `tet_quality` output for that tetrahedron.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TetQualityMetricsResult {
    /// Signed volume `(v1-v0) . ((v2-v0) x (v3-v0)) / 6`.
    pub volume: f32,
    /// Normalised radius ratio `3 * r_in / r_circ` in `[0, 1]`; `0` when
    /// degenerate.
    pub radius_ratio: f32,
    /// Smallest dihedral angle across the six edges, in degrees.
    pub min_dihedral_deg: f32,
    /// Largest dihedral angle across the six edges, in degrees.
    pub max_dihedral_deg: f32,
    /// Shortest edge length.
    pub min_edge: f32,
    /// Longest edge length.
    pub max_edge: f32,
    /// `true` when the signed volume is not strictly positive (inverted, flat
    /// or non-finite input).
    pub inverted: bool,
    /// `true` when the element is numerically flat (no finite circumsphere) or
    /// the input is non-finite.
    pub degenerate: bool,
}

/// Encodes one [`TetQualityMetricsQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &TetQualityMetricsQuery) -> GpuQuery {
    GpuQuery {
        v0: q.v0,
        pad0: 0.0,
        v1: q.v1,
        pad1: 0.0,
        v2: q.v2,
        pad2: 0.0,
        v3: q.v3,
        pad3: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`TetQualityMetricsResult`].
fn decode_result(raw: &GpuResult) -> TetQualityMetricsResult {
    TetQualityMetricsResult {
        volume: raw.volume,
        radius_ratio: raw.radius_ratio,
        min_dihedral_deg: raw.min_dihedral_deg,
        max_dihedral_deg: raw.max_dihedral_deg,
        min_edge: raw.min_edge,
        max_edge: raw.max_edge,
        inverted: raw.inverted != 0,
        degenerate: raw.degenerate != 0,
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

/// A compiled, reusable tetrahedron-quality compute pipeline, twinning the `CPU`
/// golden `tet_quality`.
pub struct GpuTetQualityMetrics {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTetQualityMetrics {
    /// Compiles the tetrahedron-quality kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTetQualityMetrics {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_tet_quality_metrics"),
            source: ShaderSource::Wgsl(TET_QUALITY_METRICS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_tet_quality_metrics_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_tet_quality_metrics_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_tet_quality_metrics_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTetQualityMetrics {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`TetQualityMetricsResult`] per input, in order.
    ///
    /// The discrete flags match the reference exactly and the continuous
    /// scalars to the module's tolerance. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TetQualityMetricsQuery],
    ) -> Vec<TetQualityMetricsResult> {
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
            label: Some("prism_volumetric_tet_quality_metrics_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_tet_quality_metrics_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_tet_quality_metrics_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_tet_quality_metrics_bind_group"),
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
            label: Some("prism_volumetric_tet_quality_metrics_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_tet_quality_metrics_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_tet_quality_metrics_pass"),
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
