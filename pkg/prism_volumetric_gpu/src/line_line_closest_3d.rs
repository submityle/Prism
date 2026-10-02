//! `wgpu` compute twin of the infinite-line / ray closest-point geometry
//! contract
//! ([`line_line_closest_3d`](prism_render_architecture::particle::line_line_closest_3d)).
//!
//! The `CPU` golden
//! [`line_line_closest_3d`](prism_render_architecture::particle::line_line_closest_3d)
//! owns the closed-form proximity math between two *infinite* 3D lines and the
//! bounded ray-vs-ray variant. For a query `(p1, d1, p2, d2)` it answers two
//! questions: the mutually closest points of the two supporting lines
//! `A(s) = p1 + s · d1` and `B(t) = p2 + t · d2` with unconstrained parameters
//! ([`line_line_closest`](prism_render_architecture::particle::line_line_closest_3d::line_line_closest)),
//! and the closest points of the two rays restricted to the non-negative
//! half-lines `s, t ≥ 0`
//! ([`ray_ray_closest`](prism_render_architecture::particle::line_line_closest_3d::ray_ray_closest)).
//! [`GpuLineLineClosest3d`] is the on-device twin: one thread per query
//! reproduces both answers, so a passing real-device parity test is direct
//! evidence the ported kernel solves the same geometry and classifies the same
//! degenerate cases the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for both the
//! line-line and the ray-ray variant: the parameters `s` and `t`, the closest
//! point on each line/ray (`point_on_a`, `point_on_b`) and the Euclidean
//! `distance` between them. The reference's branches are mirrored exactly: the
//! full-rank line solve, the scale-relative parallel pin (`denom.abs()` at or
//! below `EPS · a · c` fixes `s = 0` and drops a perpendicular onto line `B`),
//! and the zero-length-direction guards. The ray variant mirrors Ericson's
//! clamp-and-reproject recipe branch for branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `sqrt`, `+ - * /` and unsigned bit arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow` or `tan`, and no optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units
//! in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields, tight enough
//! to catch a genuinely wrong port (a dropped branch, a swapped coefficient, a
//! missing clamp) yet loose enough to admit legal fused multiply-add
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`line_line_closest_3d`](prism_render_architecture::particle::line_line_closest_3d);
//! no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::line_line_closest_3d::ClosestLines;
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

/// The portable core-`WGSL` line-line / ray-ray closest-point kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`line_line_closest_3d`](prism_render_architecture::particle::line_line_closest_3d)
/// branch for branch; see the module documentation for the algorithm.
const LINE_LINE_CLOSEST_3D_WGSL: &str = r#"
// Infinite-line / ray closest-point twin: one thread per query reproduces the
// mutually closest points of two supporting lines A(s) = p1 + s*d1 and
// B(t) = p2 + t*d2 (unconstrained s, t) and of the two rays restricted to the
// half-lines s, t >= 0. It mirrors the CPU golden
// `particle::line_line_closest_3d` branch for branch, uses only the portable
// core-WGSL subset (min/max/abs/sqrt and + - * / plus unsigned bit math) and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::line_line_closest_3d; no third-party engine source or derived code.

// Epsilon guarding every divide and magnitude test, matching the reference
// `EPS`, so no exact == / != on an f32 is ever needed. A determinant is treated
// as parallel when its absolute value is at or below EPS * a * c (scale
// relative), and any squared length at or below EPS is a degenerate direction.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Base point of the first line/ray; a pad lane follows.
    p1: vec3<f32>,
    pad0: f32,
    // Direction of the first line/ray (need not be unit length); a pad follows.
    d1: vec3<f32>,
    pad1: f32,
    // Base point of the second line/ray; a pad lane follows.
    p2: vec3<f32>,
    pad2: f32,
    // Direction of the second line/ray; a pad lane follows.
    d2: vec3<f32>,
    pad3: f32,
}

struct Result {
    // line_line_closest s and t, the Euclidean distance, then ray_ray_closest
    // s: four scalars filling one vec4 slot.
    line_s: f32,
    line_t: f32,
    line_distance: f32,
    ray_s: f32,
    // ray_ray_closest t and distance, with two pad lanes.
    ray_t: f32,
    ray_distance: f32,
    pad0: f32,
    pad1: f32,
    // line_line_closest point_on_a; a pad lane follows.
    line_point_on_a: vec3<f32>,
    pad2: f32,
    // line_line_closest point_on_b; a pad lane follows.
    line_point_on_b: vec3<f32>,
    pad3: f32,
    // ray_ray_closest point_on_a; a pad lane follows.
    ray_point_on_a: vec3<f32>,
    pad4: f32,
    // ray_ray_closest point_on_b; a pad lane follows.
    ray_point_on_b: vec3<f32>,
    pad5: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Euclidean length sqrt(v . v), matching the reference `v_length`.
fn v_length(v: vec3<f32>) -> f32 {
    return sqrt(dot(v, v));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let p1 = q.p1;
    let d1 = q.d1;
    let p2 = q.p2;
    let d2 = q.d2;

    let r = p1 - p2;
    let a = dot(d1, d1);
    let b = dot(d1, d2);
    let c = dot(d2, d2);
    let d = dot(d1, r);
    let e = dot(d2, r);
    let denom = a * c - b * b;

    // --- line_line_closest: unconstrained supporting lines ------------------
    var line_s: f32 = 0.0;
    var line_t: f32 = 0.0;
    // Scale-relative parallel test: a*c is the largest magnitude the
    // determinant could reach (b*b <= a*c by Cauchy-Schwarz).
    if (abs(denom) <= EPS * a * c) {
        // Parallel (or a degenerate direction): pin s = 0 and drop a
        // perpendicular onto line B, guarding the divide when B is degenerate.
        line_s = 0.0;
        if (c > EPS) {
            line_t = e / c;
        } else {
            line_t = 0.0;
        }
    } else {
        // Full-rank system: unique mutually-closest pair.
        line_s = (b * e - c * d) / denom;
        line_t = (a * e - b * d) / denom;
    }
    let line_point_on_a = p1 + d1 * line_s;
    let line_point_on_b = p2 + d2 * line_t;
    let line_distance = v_length(line_point_on_a - line_point_on_b);

    // --- ray_ray_closest: half-lines s, t >= 0 (Ericson 5.1.9) --------------
    // Solve for s on ray A, clamping onto s >= 0; fall back to 0 when parallel.
    var ray_s: f32 = 0.0;
    if (abs(denom) > EPS * a * c) {
        ray_s = max((b * e - c * d) / denom, 0.0);
    } else {
        ray_s = 0.0;
    }
    // Re-project onto ray B: t = (b*s + e) / c, guarding a degenerate B.
    var ray_t: f32 = 0.0;
    if (c > EPS) {
        ray_t = (b * ray_s + e) / c;
    } else {
        ray_t = 0.0;
    }
    // If t left the half-line, clamp it and re-project back onto ray A.
    if (ray_t < 0.0) {
        ray_t = 0.0;
        if (a > EPS) {
            ray_s = max((-d) / a, 0.0);
        } else {
            ray_s = 0.0;
        }
    }
    let ray_point_on_a = p1 + d1 * ray_s;
    let ray_point_on_b = p2 + d2 * ray_t;
    let ray_distance = v_length(ray_point_on_a - ray_point_on_b);

    var out: Result;
    out.line_s = line_s;
    out.line_t = line_t;
    out.line_distance = line_distance;
    out.ray_s = ray_s;
    out.ray_t = ray_t;
    out.ray_distance = ray_distance;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.line_point_on_a = line_point_on_a;
    out.pad2 = 0.0;
    out.line_point_on_b = line_point_on_b;
    out.pad3 = 0.0;
    out.ray_point_on_a = ray_point_on_a;
    out.pad4 = 0.0;
    out.ray_point_on_b = ray_point_on_b;
    out.pad5 = 0.0;
    results[idx] = out;
}
"#;

/// One closest-point query: the base point and direction of two lines/rays, the
/// same inputs the reference `line_line_closest` and `ray_ray_closest` consume.
/// The directions need not be unit length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineLineQuery {
    /// Base point `p1` of the first line/ray, `A(s) = p1 + s · d1`.
    pub p1: [f32; 3],
    /// Direction `d1` of the first line/ray.
    pub d1: [f32; 3],
    /// Base point `p2` of the second line/ray, `B(t) = p2 + t · d2`.
    pub p2: [f32; 3],
    /// Direction `d2` of the second line/ray.
    pub d2: [f32; 3],
}

impl LineLineQuery {
    /// Builds a query from two base points and two directions.
    #[must_use]
    pub const fn new(p1: [f32; 3], d1: [f32; 3], p2: [f32; 3], d2: [f32; 3]) -> LineLineQuery {
        LineLineQuery { p1, d1, p2, d2 }
    }
}

/// The resolved answer for one query: the infinite-line closest pair and the
/// ray-restricted closest pair, each mirroring a reference `ClosestLines`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineLineResult {
    /// The closest-point record of the two infinite supporting lines, matching
    /// `line_line_closest`.
    pub line: ClosestLines,
    /// The closest-point record of the two rays `s, t ≥ 0`, matching
    /// `ray_ray_closest`.
    pub ray: ClosestLines,
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding
/// `(p1.xyz, pad)`, `(d1.xyz, pad)`, `(p2.xyz, pad)` and `(d2.xyz, pad)` — `64`
/// bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Base point of the first line/ray.
    p1: [f32; 3],
    /// Padding lane after the first base point.
    pad0: f32,
    /// Direction of the first line/ray.
    d1: [f32; 3],
    /// Padding lane after the first direction.
    pad1: f32,
    /// Base point of the second line/ray.
    p2: [f32; 3],
    /// Padding lane after the second base point.
    pad2: f32,
    /// Direction of the second line/ray.
    d2: [f32; 3],
    /// Padding lane after the second direction.
    pad3: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &LineLineQuery) -> GpuQuery {
        GpuQuery {
            p1: query.p1,
            pad0: 0.0,
            d1: query.d1,
            pad1: 0.0,
            p2: query.p2,
            pad2: 0.0,
            d2: query.d2,
            pad3: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot
/// `(line_s, line_t, line_distance, ray_s)`, a `(ray_t, ray_distance)` slot with
/// two pad lanes, then four `vec4` slots for the line and ray closest points —
/// `96` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Line-line parameter along the first line.
    line_s: f32,
    /// Line-line parameter along the second line.
    line_t: f32,
    /// Line-line closest-point distance.
    line_distance: f32,
    /// Ray-ray parameter along the first ray.
    ray_s: f32,
    /// Ray-ray parameter along the second ray.
    ray_t: f32,
    /// Ray-ray closest-point distance.
    ray_distance: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Line-line closest point on the first line.
    line_point_on_a: [f32; 3],
    /// Padding lane after the line point on `A`.
    pad2: f32,
    /// Line-line closest point on the second line.
    line_point_on_b: [f32; 3],
    /// Padding lane after the line point on `B`.
    pad3: f32,
    /// Ray-ray closest point on the first ray.
    ray_point_on_a: [f32; 3],
    /// Padding lane after the ray point on `A`.
    pad4: f32,
    /// Ray-ray closest point on the second ray.
    ray_point_on_b: [f32; 3],
    /// Padding lane after the ray point on `B`.
    pad5: f32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable line-line / ray-ray closest-point compute pipeline.
pub struct GpuLineLineClosest3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLineLineClosest3d {
    /// Compiles the line-line / ray-ray closest-point kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLineLineClosest3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d"),
            source: ShaderSource::Wgsl(LINE_LINE_CLOSEST_3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLineLineClosest3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`LineLineResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers (`line_line_closest` and
    /// `ray_ray_closest`) to within the tolerance documented on this module. An
    /// empty input returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[LineLineQuery]) -> Vec<LineLineResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_line_line_closest_3d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_line_line_closest_3d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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

/// Decodes one packed [`GpuResult`] into the public [`LineLineResult`].
fn decode_result(raw: &GpuResult) -> LineLineResult {
    LineLineResult {
        line: ClosestLines {
            s: raw.line_s,
            t: raw.line_t,
            point_on_a: raw.line_point_on_a,
            point_on_b: raw.line_point_on_b,
            distance: raw.line_distance,
        },
        ray: ClosestLines {
            s: raw.ray_s,
            t: raw.ray_t,
            point_on_a: raw.ray_point_on_a,
            point_on_b: raw.ray_point_on_b,
            distance: raw.ray_distance,
        },
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
