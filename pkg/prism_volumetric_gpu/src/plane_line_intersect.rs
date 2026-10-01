//! `wgpu` compute twin of the oriented-plane / line-intersection geometry
//! contract
//! ([`plane_line_intersect`](prism_render_architecture::particle::plane_line_intersect)).
//!
//! The `CPU` golden
//! [`plane_line_intersect`](prism_render_architecture::particle::plane_line_intersect)
//! owns the single scalar solve that turns a plane `normal · x = d` and a line
//! `p0 + t * dir` into a concrete crossing point plus its ray parameter `t`. It
//! exposes three entry points re-derived from the same point-normal form: the
//! infinite-line solve
//! [`line_plane_intersect`](prism_render_architecture::particle::plane_line_intersect::line_plane_intersect)
//! (a three-way [`LinePlaneResult`](prism_render_architecture::particle::plane_line_intersect::LinePlaneResult)
//! of `Point` / `Parallel` / `Coincident`), the finite-segment solve
//! [`segment_plane_intersect`](prism_render_architecture::particle::plane_line_intersect::segment_plane_intersect)
//! (a `Point` kept only when `t` lands in `[0, 1]`), and the ray solve
//! [`ray_plane_intersect`](prism_render_architecture::particle::plane_line_intersect::ray_plane_intersect)
//! (a `Point` kept only when `t >= 0`), alongside the
//! [`signed_distance`](prism_render_architecture::particle::plane_line_intersect::signed_distance)
//! helper `normal · point - d`. [`GpuPlaneLineIntersect`] is the on-device twin:
//! one thread per query reproduces all three answers plus the signed distance of
//! `p0`, so a passing real-device parity test is direct evidence the ported
//! kernel solves the same geometry and classifies the same degenerate cases the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced for one shared
//! geometry `(p0, dir, p1, normal, d)`: the infinite-line classification code
//! and (when it is a crossing) its `t` and point, the segment hit flag with its
//! `t` and point, the ray hit flag with its `t` and point, and the signed
//! distance of `p0`. The reference's parallel / coincident split is mirrored by
//! an integer classification code (`0` crossing, `1` parallel, `2` coincident):
//! `|normal · dir| <= EPS` pins the line to parallel or coincident by the
//! tolerant on-plane test on `p0`, exactly as the golden guard does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `dot`,
//! `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no `sqrt` (the
//! solve is a single ratio, never a length) and no optional device feature, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! divide, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! The integer classification code and the hit flags are compared with an exact
//! `==`; they share every branch because the fixtures stay well clear of the
//! `EPS` guard and the `[0, 1]` / `t >= 0` boundaries. The continuous `f32`
//! fields are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on them, tight enough to catch a
//! genuinely wrong port yet loose enough to admit legal fused multiply-add
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`plane_line_intersect`](prism_render_architecture::particle::plane_line_intersect);
//! no third-party engine source or derived code.

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

/// Classification code for an infinite line that crosses the plane at a single
/// point, mirroring `LinePlaneResult::Point`.
pub const CLASS_POINT: u32 = 0;

/// Classification code for an infinite line parallel to the plane and off it,
/// mirroring `LinePlaneResult::Parallel`.
pub const CLASS_PARALLEL: u32 = 1;

/// Classification code for an infinite line lying inside the plane, mirroring
/// `LinePlaneResult::Coincident`.
pub const CLASS_COINCIDENT: u32 = 2;

/// The portable core-`WGSL` plane / line-intersection kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`plane_line_intersect`](prism_render_architecture::particle::plane_line_intersect)
/// branch for branch; see the module documentation for the algorithm.
const PLANE_LINE_INTERSECT_WGSL: &str = r#"
// Oriented-plane / line-intersection twin: one thread per query reproduces the
// infinite-line classification and crossing, the finite-segment hit, the ray
// hit and the signed distance of p0. It mirrors the CPU golden
// `particle::plane_line_intersect` branch for branch, uses only the portable
// core-WGSL subset (abs, dot and + - * /), needs no sqrt (the solve is a single
// ratio) and takes no optional feature, so it runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: twinned from this repository's particle::plane_line_intersect; no
// third-party engine source or derived code.

// Epsilon used to guard the division and to classify the line as parallel /
// on-plane, and to widen the segment [0, 1] and ray t >= 0 acceptance bands,
// without ever writing an exact == / != on an f32. Matches the reference `EPS`.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Line / ray origin; a pad lane follows.
    p0: vec3<f32>,
    pad0: f32,
    // Infinite-line and ray direction; a pad lane follows.
    dir: vec3<f32>,
    pad1: f32,
    // Segment far endpoint (the segment runs p0 -> p1); a pad lane follows.
    p1: vec3<f32>,
    pad2: f32,
    // Plane normal in `normal · x = d`, with the plane constant d in the w lane.
    normal: vec3<f32>,
    d: f32,
}

struct Result {
    // Infinite-line classification code (0 point, 1 parallel, 2 coincident),
    // then the segment and ray hit flags (1 hit, 0 miss) and one pad lane.
    line_code: u32,
    seg_hit: u32,
    ray_hit: u32,
    pad_code: u32,
    // The infinite-line, segment and ray parameters, then the signed distance of
    // p0 to the plane: four scalars filling one vec4 slot.
    line_t: f32,
    seg_t: f32,
    ray_t: f32,
    signed_dist: f32,
    // Infinite-line crossing point; a pad lane follows.
    line_point: vec3<f32>,
    pad3: f32,
    // Segment crossing point; a pad lane follows.
    seg_point: vec3<f32>,
    pad4: f32,
    // Ray crossing point; a pad lane follows.
    ray_point: vec3<f32>,
    pad5: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The outcome of the infinite-line solve: a classification code plus the ray
// parameter t (meaningful only when code == 0, the crossing case).
struct LineSolve {
    code: u32,
    t: f32,
}

// Solve the infinite line `p0 + t * dir` against the plane `normal · x = d`,
// mirroring the reference `line_plane_intersect`. When `|normal · dir| <= EPS`
// the line is parallel: coincident (code 2) if p0 lies on the plane within EPS,
// otherwise parallel (code 1). Otherwise it is a crossing (code 0) at t.
fn solve_line(p0: vec3<f32>, dir: vec3<f32>, normal: vec3<f32>, d: f32) -> LineSolve {
    var out: LineSolve;
    let denom = dot(normal, dir);
    if (abs(denom) <= EPS) {
        let sd = dot(normal, p0) - d;
        if (abs(sd) <= EPS) {
            out.code = 2u;
        } else {
            out.code = 1u;
        }
        out.t = 0.0;
        return out;
    }
    out.code = 0u;
    out.t = (d - dot(normal, p0)) / denom;
    return out;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let p0 = q.p0;
    let dir = q.dir;
    let p1 = q.p1;
    let normal = q.normal;
    let d = q.d;

    let zero = vec3<f32>(0.0, 0.0, 0.0);

    // Infinite line: classify, and recover the crossing point when it crosses.
    let line = solve_line(p0, dir, normal, d);
    var line_t: f32 = 0.0;
    var line_point: vec3<f32> = zero;
    if (line.code == 0u) {
        line_t = line.t;
        line_point = p0 + dir * line.t;
    }

    // Finite segment p0 -> p1: a crossing is kept only when t is in [0, 1]
    // (endpoints included via EPS); a parallel or coincident segment misses.
    let seg_dir = p1 - p0;
    let seg = solve_line(p0, seg_dir, normal, d);
    var seg_hit: u32 = 0u;
    var seg_t: f32 = 0.0;
    var seg_point: vec3<f32> = zero;
    if (seg.code == 0u) {
        if (seg.t >= -EPS && seg.t <= 1.0 + EPS) {
            seg_hit = 1u;
            seg_t = seg.t;
            seg_point = p0 + seg_dir * seg.t;
        }
    }

    // Ray origin + t * dir: a crossing is kept only when t >= 0 (origin included
    // via EPS); a parallel or coincident ray misses.
    var ray_hit: u32 = 0u;
    var ray_t: f32 = 0.0;
    var ray_point: vec3<f32> = zero;
    if (line.code == 0u) {
        if (line.t >= -EPS) {
            ray_hit = 1u;
            ray_t = line.t;
            ray_point = p0 + dir * line.t;
        }
    }

    var out: Result;
    out.line_code = line.code;
    out.seg_hit = seg_hit;
    out.ray_hit = ray_hit;
    out.pad_code = 0u;
    out.line_t = line_t;
    out.seg_t = seg_t;
    out.ray_t = ray_t;
    out.signed_dist = dot(normal, p0) - d;
    out.line_point = line_point;
    out.pad3 = 0.0;
    out.seg_point = seg_point;
    out.pad4 = 0.0;
    out.ray_point = ray_point;
    out.pad5 = 0.0;
    results[idx] = out;
}
"#;

/// One plane / line-intersection query: a shared geometry fed to all three
/// reference solves. The infinite line and ray use `p0` + `t` * `dir`; the
/// segment runs `p0` -> `p1`; the plane is `normal · x = d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneLineQuery {
    /// The line / ray origin and the segment's near endpoint.
    pub p0: [f32; 3],
    /// The infinite-line and ray direction.
    pub dir: [f32; 3],
    /// The segment's far endpoint; the segment direction is `p1 - p0`.
    pub p1: [f32; 3],
    /// The plane normal in `normal · x = d` (need not be unit length).
    pub normal: [f32; 3],
    /// The plane constant `d` in `normal · x = d`.
    pub d: f32,
}

impl PlaneLineQuery {
    /// Builds a query from the shared geometry.
    #[must_use]
    pub const fn new(
        p0: [f32; 3],
        dir: [f32; 3],
        p1: [f32; 3],
        normal: [f32; 3],
        d: f32,
    ) -> PlaneLineQuery {
        PlaneLineQuery {
            p0,
            dir,
            p1,
            normal,
            d,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across its three twinned solves plus the signed distance of `p0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneLineResult {
    /// Infinite-line classification code: [`CLASS_POINT`], [`CLASS_PARALLEL`] or
    /// [`CLASS_COINCIDENT`], matching the `LinePlaneResult` variant.
    pub line_code: u32,
    /// The infinite-line parameter `t`; meaningful only when
    /// `line_code == CLASS_POINT`.
    pub line_t: f32,
    /// The infinite-line crossing point; meaningful only when
    /// `line_code == CLASS_POINT`.
    pub line_point: [f32; 3],
    /// Whether the finite segment `p0` -> `p1` crosses the plane with `t` in
    /// `[0, 1]`, matching `segment_plane_intersect` returning `Some`.
    pub segment_hit: bool,
    /// The segment crossing parameter `t`; meaningful only when `segment_hit`.
    pub segment_t: f32,
    /// The segment crossing point; meaningful only when `segment_hit`.
    pub segment_point: [f32; 3],
    /// Whether the ray crosses the plane with `t >= 0`, matching
    /// `ray_plane_intersect` returning `Some`.
    pub ray_hit: bool,
    /// The ray crossing parameter `t`; meaningful only when `ray_hit`.
    pub ray_t: f32,
    /// The ray crossing point; meaningful only when `ray_hit`.
    pub ray_point: [f32; 3],
    /// The signed distance `normal · p0 - d`, matching `signed_distance`.
    pub signed_distance_p0: f32,
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding
/// `(p0.xyz, pad)`, `(dir.xyz, pad)`, `(p1.xyz, pad)` and `(normal.xyz, d)` —
/// `64` bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Line / ray origin and segment near endpoint.
    p0: [f32; 3],
    /// Padding lane after the origin.
    pad0: f32,
    /// Infinite-line and ray direction.
    dir: [f32; 3],
    /// Padding lane after the direction.
    pad1: f32,
    /// Segment far endpoint.
    p1: [f32; 3],
    /// Padding lane after the far endpoint.
    pad2: f32,
    /// Plane normal.
    normal: [f32; 3],
    /// Plane constant `d`, packed into the normal slot's `w` lane.
    d: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &PlaneLineQuery) -> GpuQuery {
        GpuQuery {
            p0: query.p0,
            pad0: 0.0,
            dir: query.dir,
            pad1: 0.0,
            p1: query.p1,
            pad2: 0.0,
            normal: query.normal,
            d: query.d,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-`u32` slot
/// `(line_code, seg_hit, ray_hit, pad)`, a four-`f32` slot
/// `(line_t, seg_t, ray_t, signed_dist)`, then three `vec4` slots for the
/// infinite-line, segment and ray crossing points — `80` bytes matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Infinite-line classification code.
    line_code: u32,
    /// Segment hit flag.
    seg_hit: u32,
    /// Ray hit flag.
    ray_hit: u32,
    /// Padding word.
    pad_code: u32,
    /// Infinite-line parameter.
    line_t: f32,
    /// Segment parameter.
    seg_t: f32,
    /// Ray parameter.
    ray_t: f32,
    /// Signed distance of `p0` to the plane.
    signed_dist: f32,
    /// Infinite-line crossing point.
    line_point: [f32; 3],
    /// Padding lane after the line point.
    pad3: f32,
    /// Segment crossing point.
    seg_point: [f32; 3],
    /// Padding lane after the segment point.
    pad4: f32,
    /// Ray crossing point.
    ray_point: [f32; 3],
    /// Padding lane after the ray point.
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

/// A compiled, reusable plane / line-intersection compute pipeline.
pub struct GpuPlaneLineIntersect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPlaneLineIntersect {
    /// Compiles the plane / line-intersection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPlaneLineIntersect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_plane_line_intersect"),
            source: ShaderSource::Wgsl(PLANE_LINE_INTERSECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_plane_line_intersect_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_plane_line_intersect_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_plane_line_intersect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPlaneLineIntersect {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`PlaneLineResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers (`line_plane_intersect`,
    /// `segment_plane_intersect`, `ray_plane_intersect` and `signed_distance`):
    /// the classification code and hit flags match exactly, and the continuous
    /// fields match to within the tolerance documented on this module. An empty
    /// input returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[PlaneLineQuery]) -> Vec<PlaneLineResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_plane_line_intersect_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_plane_line_intersect_output"),
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
            label: Some("prism_volumetric_plane_line_intersect_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_plane_line_intersect_bind_group"),
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
            label: Some("prism_volumetric_plane_line_intersect_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_plane_line_intersect_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_plane_line_intersect_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`PlaneLineResult`].
fn decode_result(raw: &GpuResult) -> PlaneLineResult {
    PlaneLineResult {
        line_code: raw.line_code,
        line_t: raw.line_t,
        line_point: raw.line_point,
        segment_hit: raw.seg_hit != 0,
        segment_t: raw.seg_t,
        segment_point: raw.seg_point,
        ray_hit: raw.ray_hit != 0,
        ray_t: raw.ray_t,
        ray_point: raw.ray_point,
        signed_distance_p0: raw.signed_dist,
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
