//! `wgpu` compute twin of three exact analytic signed-distance primitives from
//! the `CPU` golden `prism_render_architecture::ray_scene::sdf_primitives`.
//!
//! Implicit modelling needs *analytic* primitives whose exact signed distance
//! is known in closed form rather than sampled on a grid. This module is the
//! on-device twin of three such lens/capsule distances, each returning the
//! signed Euclidean distance to the surface — negative inside, positive
//! outside, zero on the boundary:
//!
//! - `vesica_2d`: the 2D vesica (a symmetric lens), the intersection of two
//!   circles of radius `radius` whose centres sit at `(-offset, 0)` and
//!   `(+offset, 0)`. The query is folded into the first quadrant and the
//!   nearest feature is either a cusp or one of the two arcs, selected by a
//!   single linear test.
//! - `vesica_segment`: the 3D vesica swept along an arbitrary segment from `a`
//!   to `b` with half-thickness `width`. The query is projected onto the
//!   segment axis, reduced to an axial/radial pair, and measured against either
//!   the shared tip or the generating arc.
//! - `uneven_capsule_2d`: the 2D capsule with a `r_bottom` disc at the origin
//!   and a `r_top` disc at height `h`, joined by the exterior tangent flank.
//!   One linear slope test selects the bottom cap, the top cap, or the flank.
//!
//! [`GpuSdfVesica`] evaluates all three for one [`SdfVesicaQuery`] per thread
//! and writes one [`SdfVesicaResult`], reproducing the reference closed forms
//! with only `abs`, `min`, `max`, `sqrt`, dot products, a sign fold and
//! comparisons — no transcendental — so a passing real-device parity test is
//! direct evidence the ported kernel computes the same distances the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfVesicaQuery`] carrying the independent inputs of
//! all three primitives and writes one [`SdfVesicaResult`] holding the three
//! signed distances. The kernel mirrors the scalar reference operation order
//! (the same products, quotients and `sqrt` chains) to keep the last-place
//! difference minimal.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms into complex shapes,
//! the ray-march loop and the field baking all stay on the host; the device
//! sees only the stateless, fixed-width distance evaluation, one query at a
//! time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every distance threads through `sqrt`, products and quotients, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-5` or `rel_diff <= 1e-4` (relative floor
//! `1e-6`), tight enough to catch a genuinely wrong port yet loose enough to
//! admit a legal last-place difference. The branch-selection predicates are
//! kept clear of their switch-over cliffs by the fixtures and the rejection
//! sampling in the sweep.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `sqrt`, `select`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and
//! no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` signed-distance kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `vesica_2d`, `vesica_segment` and `uneven_capsule_2d` closed
/// forms; see the module documentation for the algorithm.
const SDF_VESICA_WGSL: &str = r#"
// Signed-distance twin: one thread evaluates one query's three analytic
// distances — the 2D vesica, the arbitrary-segment 3D vesica and the 2D uneven
// capsule — mirroring the CPU golden
// `ray_scene::sdf_primitives::{vesica_2d, vesica_segment, uneven_capsule_2d}`
// with only abs, min, max, sqrt, a sign fold, products, quotients and
// comparisons. The CSG operators and ray-march loop stay on the host.
//
// Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // vesica_2d inputs: query point, lens radius and centre half-separation.
    vesica_px: f32,
    vesica_py: f32,
    vesica_radius: f32,
    vesica_offset: f32,
    // vesica_segment inputs: query point, endpoints a and b, half-thickness.
    seg_px: f32,
    seg_py: f32,
    seg_pz: f32,
    seg_ax: f32,
    seg_ay: f32,
    seg_az: f32,
    seg_bx: f32,
    seg_by: f32,
    seg_bz: f32,
    seg_width: f32,
    // uneven_capsule_2d inputs: query point, bottom radius, top radius, height.
    uc_px: f32,
    uc_py: f32,
    uc_r_bottom: f32,
    uc_r_top: f32,
    uc_h: f32,
    pad0: f32,
}

struct Outcome {
    vesica_2d_value: f32,
    vesica_segment_value: f32,
    uneven_capsule_2d_value: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outcome>;

// Scalar 2-vector Euclidean length, matching the reference `length2`.
fn len2(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

// Scalar 3-vector Euclidean length, matching the reference `length`.
fn len3(x: f32, y: f32, z: f32) -> f32 {
    return sqrt(x * x + y * y + z * z);
}

// Reproduces Rust `f32::signum` for every finite non-negative-zero input:
// positive -> +1, negative -> -1. The fixtures keep `offset` strictly
// positive, so this folds to +1 there while staying faithful off the branch.
fn signum_f32(x: f32) -> f32 {
    return select(1.0, -1.0, x < 0.0);
}

// Exact signed distance to a 2D vesica (symmetric lens): fold to the first
// quadrant, then choose the nearest feature — a cusp or one of the two arcs —
// by one linear test.
fn vesica_2d(px_in: f32, py_in: f32, radius: f32, offset: f32) -> f32 {
    let px = abs(px_in);
    let py = abs(py_in);
    let b = sqrt(radius * radius - offset * offset);
    if ((py - b) * offset > px * b) {
        return len2(px, py - b) * signum_f32(offset);
    }
    return len2(px + offset, py) - radius;
}

// Exact signed distance to a 3D vesica swept along the segment a->b: project
// onto the axis, reduce to an axial/radial pair, then measure against the
// shared tip or the generating arc.
fn vesica_segment(
    px: f32, py: f32, pz: f32,
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    width: f32,
) -> f32 {
    let cx = (ax + bx) * 0.5;
    let cy = (ay + by) * 0.5;
    let cz = (az + bz) * 0.5;
    let bax = bx - ax;
    let bay = by - ay;
    let baz = bz - az;
    let l = len3(bax, bay, baz);
    let vx = bax / l;
    let vy = bay / l;
    let vz = baz / l;
    let pcx = px - cx;
    let pcy = py - cy;
    let pcz = pz - cz;
    let y = pcx * vx + pcy * vy + pcz * vz;
    let perpx = pcx - y * vx;
    let perpy = pcy - y * vy;
    let perpz = pcz - y * vz;
    let qx = len3(perpx, perpy, perpz);
    let qy = abs(y);
    let r = 0.5 * l;
    let d = 0.5 * (r * r - width * width) / width;
    var hx: f32;
    var hy: f32;
    var hz: f32;
    if (r * qx < d * (qy - r)) {
        hx = 0.0;
        hy = r;
        hz = 0.0;
    } else {
        hx = -d;
        hy = 0.0;
        hz = d + width;
    }
    return len2(qx - hx, qy - hy) - hz;
}

// Exact signed distance to a 2D uneven capsule: fold x to its magnitude, then a
// single slope test selects the bottom cap, the top cap or the tangent flank.
fn uneven_capsule_2d(px_in: f32, py_in: f32, r_bottom: f32, r_top: f32, h: f32) -> f32 {
    let px = abs(px_in);
    let py = py_in;
    let b = (r_bottom - r_top) / h;
    let a = sqrt(max(1.0 - b * b, 0.0));
    let k = -b * px + a * py;
    if (k < 0.0) {
        return len2(px, py) - r_bottom;
    }
    if (k > a * h) {
        return len2(px, py - h) - r_top;
    }
    return a * px + b * py - r_bottom;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Outcome;
    out.vesica_2d_value = vesica_2d(q.vesica_px, q.vesica_py, q.vesica_radius, q.vesica_offset);
    out.vesica_segment_value = vesica_segment(
        q.seg_px, q.seg_py, q.seg_pz,
        q.seg_ax, q.seg_ay, q.seg_az,
        q.seg_bx, q.seg_by, q.seg_bz,
        q.seg_width,
    );
    out.uneven_capsule_2d_value =
        uneven_capsule_2d(q.uc_px, q.uc_py, q.uc_r_bottom, q.uc_r_top, q.uc_h);
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_VESICA_WGSL`].
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
/// the independent inputs of all three primitives, plus one pad word to an
/// `80`-byte, `16`-byte-multiple stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `vesica_2d` query point `x`.
    vesica_px: f32,
    /// `vesica_2d` query point `y`.
    vesica_py: f32,
    /// `vesica_2d` lens radius.
    vesica_radius: f32,
    /// `vesica_2d` circle-centre half-separation.
    vesica_offset: f32,
    /// `vesica_segment` query point `x`.
    seg_px: f32,
    /// `vesica_segment` query point `y`.
    seg_py: f32,
    /// `vesica_segment` query point `z`.
    seg_pz: f32,
    /// `vesica_segment` endpoint `a` `x`.
    seg_ax: f32,
    /// `vesica_segment` endpoint `a` `y`.
    seg_ay: f32,
    /// `vesica_segment` endpoint `a` `z`.
    seg_az: f32,
    /// `vesica_segment` endpoint `b` `x`.
    seg_bx: f32,
    /// `vesica_segment` endpoint `b` `y`.
    seg_by: f32,
    /// `vesica_segment` endpoint `b` `z`.
    seg_bz: f32,
    /// `vesica_segment` half-thickness.
    seg_width: f32,
    /// `uneven_capsule_2d` query point `x`.
    uc_px: f32,
    /// `uneven_capsule_2d` query point `y`.
    uc_py: f32,
    /// `uneven_capsule_2d` bottom radius.
    uc_r_bottom: f32,
    /// `uneven_capsule_2d` top radius.
    uc_r_top: f32,
    /// `uneven_capsule_2d` height.
    uc_h: f32,
    /// Padding word to the `80`-byte stride.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Outcome`
/// struct: the three signed distances plus one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `vesica_2d` signed distance.
    vesica_2d_value: f32,
    /// `vesica_segment` signed distance.
    vesica_segment_value: f32,
    /// `uneven_capsule_2d` signed distance.
    uneven_capsule_2d_value: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the signed-distance twin: the independent inputs of all three
/// primitives, grouped by primitive and mirroring the `CPU` golden signatures.
///
/// `vesica_*` feed `vesica_2d`, `segment_*` feed `vesica_segment`, and
/// `uneven_*` feed `uneven_capsule_2d`. The host owns the `CSG` composition and
/// enqueues one [`SdfVesicaQuery`] per sample it needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfVesicaQuery {
    /// `vesica_2d` query point.
    pub vesica_point: [f32; 2],
    /// `vesica_2d` lens radius (must exceed `vesica_offset`).
    pub vesica_radius: f32,
    /// `vesica_2d` circle-centre half-separation.
    pub vesica_offset: f32,
    /// `vesica_segment` query point.
    pub segment_point: [f32; 3],
    /// `vesica_segment` endpoint `a`.
    pub segment_a: [f32; 3],
    /// `vesica_segment` endpoint `b`.
    pub segment_b: [f32; 3],
    /// `vesica_segment` half-thickness.
    pub segment_width: f32,
    /// `uneven_capsule_2d` query point.
    pub uneven_point: [f32; 2],
    /// `uneven_capsule_2d` bottom radius.
    pub uneven_r_bottom: f32,
    /// `uneven_capsule_2d` top radius.
    pub uneven_r_top: f32,
    /// `uneven_capsule_2d` height.
    pub uneven_h: f32,
}

impl SdfVesicaQuery {
    /// Builds a query from the independent inputs of all three primitives.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the three golden signatures packed into one query"
    )]
    pub const fn new(
        vesica_point: [f32; 2],
        vesica_radius: f32,
        vesica_offset: f32,
        segment_point: [f32; 3],
        segment_a: [f32; 3],
        segment_b: [f32; 3],
        segment_width: f32,
        uneven_point: [f32; 2],
        uneven_r_bottom: f32,
        uneven_r_top: f32,
        uneven_h: f32,
    ) -> SdfVesicaQuery {
        SdfVesicaQuery {
            vesica_point,
            vesica_radius,
            vesica_offset,
            segment_point,
            segment_a,
            segment_b,
            segment_width,
            uneven_point,
            uneven_r_bottom,
            uneven_r_top,
            uneven_h,
        }
    }
}

/// One resolved query of the signed-distance twin: the three signed distances,
/// negative inside each surface, positive outside, zero on the boundary.
///
/// `vesica_2d_value` is `vesica_2d`, `vesica_segment_value` is
/// `vesica_segment`, and `uneven_capsule_2d_value` is `uneven_capsule_2d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfVesicaResult {
    /// Signed distance to the 2D vesica.
    pub vesica_2d_value: f32,
    /// Signed distance to the arbitrary-segment 3D vesica.
    pub vesica_segment_value: f32,
    /// Signed distance to the 2D uneven capsule.
    pub uneven_capsule_2d_value: f32,
}

/// Encodes one [`SdfVesicaQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfVesicaQuery) -> GpuQuery {
    GpuQuery {
        vesica_px: q.vesica_point[0],
        vesica_py: q.vesica_point[1],
        vesica_radius: q.vesica_radius,
        vesica_offset: q.vesica_offset,
        seg_px: q.segment_point[0],
        seg_py: q.segment_point[1],
        seg_pz: q.segment_point[2],
        seg_ax: q.segment_a[0],
        seg_ay: q.segment_a[1],
        seg_az: q.segment_a[2],
        seg_bx: q.segment_b[0],
        seg_by: q.segment_b[1],
        seg_bz: q.segment_b[2],
        seg_width: q.segment_width,
        uc_px: q.uneven_point[0],
        uc_py: q.uneven_point[1],
        uc_r_bottom: q.uneven_r_bottom,
        uc_r_top: q.uneven_r_top,
        uc_h: q.uneven_h,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfVesicaResult`].
fn decode_result(raw: &GpuResult) -> SdfVesicaResult {
    SdfVesicaResult {
        vesica_2d_value: raw.vesica_2d_value,
        vesica_segment_value: raw.vesica_segment_value,
        uneven_capsule_2d_value: raw.uneven_capsule_2d_value,
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

/// A compiled, reusable signed-distance compute pipeline, twinning the `CPU`
/// golden `vesica_2d`, `vesica_segment` and `uneven_capsule_2d`.
pub struct GpuSdfVesica {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfVesica {
    /// Compiles the signed-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfVesica {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_vesica"),
            source: ShaderSource::Wgsl(SDF_VESICA_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_vesica_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_vesica_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_vesica_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfVesica {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfVesicaResult`] per
    /// input, in order.
    ///
    /// The distances match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[SdfVesicaQuery]) -> Vec<SdfVesicaResult> {
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
            label: Some("prism_volumetric_sdf_vesica_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_vesica_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_vesica_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_vesica_bind_group"),
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
            label: Some("prism_volumetric_sdf_vesica_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_vesica_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_vesica_pass"),
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
