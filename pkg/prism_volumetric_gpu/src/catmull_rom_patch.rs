//! `wgpu` compute twin of the bicubic Catmull-Rom surface patch golden
//! [`CatmullRomPatch`](prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch).
//!
//! A [`CatmullRomPatch`](prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch)
//! is a `4×4` interpolating control net stored row-major
//! (`control[row * 4 + col]`, `col` along `u`, `row` along `v`). Unlike a
//! Bézier patch, the Catmull-Rom surface passes *through* its inner `2×2`
//! points and uses the outer ring only to shape the boundary tangents. A
//! uniform Catmull-Rom span `[c0, c1, c2, c3]` is exactly a cubic Bézier
//! segment over `[c1, c2]`, so the net is converted to Bézier form by applying
//! the span rule `b0 = c1`, `b1 = c1 + (c2 − c0)/6`, `b2 = c2 − (c3 − c1)/6`,
//! `b3 = c2` along each `u` row and then along each `v` column — a purely
//! linear tensor-product change of basis with no transcendental call. The
//! resulting Bézier net is evaluated with De Casteljau's algorithm (nested
//! `lerp`s) for the surface point `P(u, v)`, its partials `∂P/∂u` and `∂P/∂v`,
//! and the unit normal `∂P/∂u × ∂P/∂v`.
//!
//! [`GpuCatmullRomPatch`] is the on-device twin that runs one thread per query
//! and reproduces every lane; a passing real-device parity test is therefore
//! direct evidence the ported kernel folds the same span-to-Bézier conversion,
//! the same De Casteljau point and tangent evaluation, and the same
//! degenerate-tangent search the reference does, not merely that its shader
//! compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces
//! [`CatmullRomPatch::point`](prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch::point)
//! and
//! [`CatmullRomPatch::normal`](prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch::normal)
//! step for step: the two-pass span-to-Bézier conversion, the bicubic De
//! Casteljau point, the `∂P/∂u` and `∂P/∂v` partials, and the normal search
//! that nudges the sample a few steps toward the patch interior along the
//! shorter axis until a non-degenerate tangent frame is found, falling back to
//! `[0, 0, 1]` when the cross product collapses at every probe. A `degenerate`
//! flag records whether that fallback was taken.
//!
//! # What stays on the host
//!
//! Only the per-query closed form is twinned. The reference tessellation
//! (`tessellate`, the variable-length [`IndexedBilinearPatchMesh`] and its
//! `BVH`) is not a per-lane kernel and stays on the host; the twin evaluates
//! one `(control net, u, v)` sample per thread.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `cross`, `dot`,
//! `clamp`, `sqrt`, `+ - * /` and comparisons — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `round`, `smoothstep`, `cbrt`, no `u64`/`i64`/`u16`/`i16`/`f64`
//! and no optional device feature. It therefore runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of linear interpolations and
//! one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the continuous `point` and
//! `normal` values yet an *exact* match on the discrete `degenerate` flag,
//! which is routed through the same tangent-collapse guard the reference uses
//! so a sample placed on a well-conditioned net folds the identical verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::catmull_rom_patch`；
//! 无第三方引擎源码或衍生代码。

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Number of control points in the `4×4` net.
const NET_SIZE: usize = 16;

/// Discrete flag written by the kernel when the tangent frame collapses at
/// every probe and the normal falls back to `[0, 0, 1]`. A direct `f32`
/// equality is forbidden, so the kernel emits this integer flag rather than a
/// sentinel float.
const DEGENERATE_FLAG: u32 = 1;

/// The portable core-`WGSL` Catmull-Rom patch kernel source. Inlined so the
/// crate ships as a single source file. Mirrors the `CPU` golden
/// [`CatmullRomPatch::point`](prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch::point)
/// and
/// [`CatmullRomPatch::normal`](prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch::normal);
/// see the module documentation for the algorithm.
const CATMULL_ROM_PATCH_WGSL: &str = r#"
// Bicubic Catmull-Rom surface patch twin: one thread per (control net, u, v)
// query converts the 4x4 interpolating net to Bezier form via two linear
// span passes, then evaluates the bicubic De Casteljau point, the two partial
// derivatives and the unit normal. It mirrors the CPU golden
// `ray_scene::catmull_rom_patch::CatmullRomPatch` point/normal step for step,
// uses only the portable core-WGSL subset (cross/dot/clamp/sqrt, + - * / and
// comparisons), and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::catmull_rom_patch；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. The 16 control points are each padded to a vec4 lane so the
// storage array needs no manual vec3 alignment arithmetic; `uv` carries the
// evaluation parameters in its x and y lanes.
struct Query {
    control: array<vec4<f32>, 16>,
    uv: vec4<f32>,
}

// One result: the surface point and unit normal (each in xyz of a vec4) plus a
// degenerate flag and padding words.
struct Result {
    point: vec4<f32>,
    normal: vec4<f32>,
    degenerate: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// Normal search outcome: the unit normal and whether the tangent-frame search
// exhausted every probe and fell back to the up axis.
struct NormalOut {
    n: vec3<f32>,
    deg: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Linear interpolation a + t*(b - a).
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// Cubic Bezier point at t over four control points (three nested lerps).
fn cubic_point(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
    let a = lerp3(p0, p1, t);
    let b = lerp3(p1, p2, t);
    let c = lerp3(p2, p3, t);
    let d = lerp3(a, b, t);
    let e = lerp3(b, c, t);
    return lerp3(d, e, t);
}

// Cubic Bezier derivative at t: 3 times the quadratic De Casteljau over the
// three adjacent control-point differences.
fn cubic_deriv(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
    let d0 = p1 - p0;
    let d1 = p2 - p1;
    let d2 = p3 - p2;
    let a = lerp3(d0, d1, t);
    let b = lerp3(d1, d2, t);
    let q = lerp3(a, b, t);
    return q * 3.0;
}

// Surface point P(u, v): collapse each v-row to a point by a cubic De
// Casteljau in u, then collapse the four results by a cubic De Casteljau in v.
fn surface_point(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_point(bez[0], bez[1], bez[2], bez[3], u);
    let c1 = cubic_point(bez[4], bez[5], bez[6], bez[7], u);
    let c2 = cubic_point(bez[8], bez[9], bez[10], bez[11], u);
    let c3 = cubic_point(bez[12], bez[13], bez[14], bez[15], u);
    return cubic_point(c0, c1, c2, c3, v);
}

// Partial derivative dP/du: each v-row contributes its u-tangent, blended by a
// cubic De Casteljau in v.
fn surface_partial_u(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_deriv(bez[0], bez[1], bez[2], bez[3], u);
    let c1 = cubic_deriv(bez[4], bez[5], bez[6], bez[7], u);
    let c2 = cubic_deriv(bez[8], bez[9], bez[10], bez[11], u);
    let c3 = cubic_deriv(bez[12], bez[13], bez[14], bez[15], u);
    return cubic_point(c0, c1, c2, c3, v);
}

// Partial derivative dP/dv: each v-row is collapsed to a point by a cubic De
// Casteljau in u, then differentiated by a cubic derivative in v.
fn surface_partial_v(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let c0 = cubic_point(bez[0], bez[1], bez[2], bez[3], u);
    let c1 = cubic_point(bez[4], bez[5], bez[6], bez[7], u);
    let c2 = cubic_point(bez[8], bez[9], bez[10], bez[11], u);
    let c3 = cubic_point(bez[12], bez[13], bez[14], bez[15], u);
    return cubic_deriv(c0, c1, c2, c3, v);
}

// Unit surface normal with the reference's degenerate-tangent search: nudge the
// sample a few steps toward the interior until a non-degenerate frame is found,
// else fall back to the up axis and flag the lane degenerate.
fn surface_normal(bez: array<vec3<f32>, 16>, u: f32, v: f32) -> NormalOut {
    var res: NormalOut;
    for (var probe: i32 = 0; probe < 4; probe = probe + 1) {
        let eps = 1.0e-3 * f32(probe);
        let uu = clamp(u + eps, 0.0, 1.0);
        let vv = clamp(v + eps, 0.0, 1.0);
        let du = surface_partial_u(bez, uu, vv);
        let dv = surface_partial_v(bez, uu, vv);
        let nvec = cross(du, dv);
        let len2 = dot(nvec, nvec);
        if (len2 > 0.0) {
            let inv = 1.0 / sqrt(len2);
            res.n = nvec * inv;
            res.deg = 0u;
            return res;
        }
    }
    res.n = vec3<f32>(0.0, 0.0, 1.0);
    res.deg = 1u;
    return res;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];

    // Load the control net as vec3.
    var ctrl: array<vec3<f32>, 16>;
    for (var i: i32 = 0; i < 16; i = i + 1) {
        ctrl[i] = q.control[i].xyz;
    }

    // Catmull-Rom -> Bezier: two linear span passes (rows along u, then columns
    // along v). The span rule is b0 = c1, b1 = c1 + (c2 - c0)/6,
    // b2 = c2 - (c3 - c1)/6, b3 = c2.
    let sixth = 1.0 / 6.0;
    var tmp: array<vec3<f32>, 16>;
    for (var row: i32 = 0; row < 4; row = row + 1) {
        let base = row * 4;
        tmp[base] = ctrl[base + 1];
        tmp[base + 1] = ctrl[base + 1] + (ctrl[base + 2] - ctrl[base]) * sixth;
        tmp[base + 2] = ctrl[base + 2] - (ctrl[base + 3] - ctrl[base + 1]) * sixth;
        tmp[base + 3] = ctrl[base + 2];
    }
    var bez: array<vec3<f32>, 16>;
    for (var col: i32 = 0; col < 4; col = col + 1) {
        let c0 = tmp[col];
        let c1 = tmp[col + 4];
        let c2 = tmp[col + 8];
        let c3 = tmp[col + 12];
        bez[col] = c1;
        bez[col + 4] = c1 + (c2 - c0) * sixth;
        bez[col + 8] = c2 - (c3 - c1) * sixth;
        bez[col + 12] = c2;
    }

    let u = q.uv.x;
    let v = q.uv.y;
    let p = surface_point(bez, u, v);
    let nrm = surface_normal(bez, u, v);

    var r: Result;
    r.point = vec4<f32>(p, 0.0);
    r.normal = vec4<f32>(nrm.n, 0.0);
    r.degenerate = nrm.deg;
    r.pad0 = 0u;
    r.pad1 = 0u;
    r.pad2 = 0u;
    results[idx] = r;
}
"#;

/// One Catmull-Rom patch query: the `4×4` row-major control net and the
/// evaluation parameters `(u, v) ∈ [0, 1]²`.
///
/// Mirrors a single reference
/// [`CatmullRomPatch`](prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch)
/// evaluation. The net is stored row-major (`control[row * 4 + col]`), with
/// `col` advancing along `u` and `row` advancing along `v`. Carrying the net
/// per query lets one dispatch evaluate many distinct patches. Derives only
/// [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatmullRomPatchQuery {
    /// The `4×4` Catmull-Rom control net, row-major (`col` along `u`, `row`
    /// along `v`).
    pub control: [[f32; 3]; NET_SIZE],
    /// The `u` parameter in `[0, 1]`.
    pub u: f32,
    /// The `v` parameter in `[0, 1]`.
    pub v: f32,
}

impl CatmullRomPatchQuery {
    /// Builds one patch query from its control net and parameters.
    #[must_use]
    pub fn new(control: [[f32; 3]; NET_SIZE], u: f32, v: f32) -> CatmullRomPatchQuery {
        CatmullRomPatchQuery { control, u, v }
    }
}

/// The evaluated sample for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `point` is the surface position `P(u, v)` and `normal` is the unit surface
/// normal `∂P/∂u × ∂P/∂v`. `degenerate` is `true` when the reference tangent
/// search exhausted every probe and the normal fell back to `[0, 0, 1]`.
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32`
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatmullRomPatchResult {
    /// The surface position `P(u, v)`.
    pub point: [f32; 3],
    /// The unit surface normal at `(u, v)`.
    pub normal: [f32; 3],
    /// Whether the tangent frame collapsed and the normal fell back to the up
    /// axis.
    pub degenerate: bool,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`CATMULL_ROM_PATCH_WGSL`]: the query count and three pad words
/// — `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `272`-byte `std430` stride matching `Query` in the
/// shader: the 16 control points each padded to a `vec4` lane and the `(u, v)`
/// parameters packed into the `x`/`y` lanes of a `vec4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// The 16 control points in `xyz`; each `w` lane is unused padding.
    control: [[f32; 4]; NET_SIZE],
    /// The `(u, v)` parameters in `x`/`y`; `z`/`w` are unused padding.
    uv: [f32; 4],
}

impl GpuQuery {
    /// Packs a [`CatmullRomPatchQuery`] into the `std430` upload layout.
    fn from_query(query: &CatmullRomPatchQuery) -> GpuQuery {
        let mut control = [[0.0f32; 4]; NET_SIZE];
        for (slot, c) in query.control.iter().enumerate() {
            control[slot] = [c[0], c[1], c[2], 0.0];
        }
        GpuQuery {
            control,
            uv: [query.u, query.v, 0.0, 0.0],
        }
    }
}

/// One result as read back. `48`-byte `std430` stride matching `Result` in the
/// shader: the surface point and unit normal (each in the `xyz` of a `vec4`),
/// the degenerate flag and three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Surface point in `xyz`; the `w` lane is unused padding.
    point: [f32; 4],
    /// Unit normal in `xyz`; the `w` lane is unused padding.
    normal: [f32; 4],
    /// Degenerate flag (`1` = tangent frame collapsed).
    degenerate: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Maps one kernel `Result` lane back to the host [`CatmullRomPatchResult`].
fn decode_result(raw: &GpuResult) -> CatmullRomPatchResult {
    CatmullRomPatchResult {
        point: [raw.point[0], raw.point[1], raw.point[2]],
        normal: [raw.normal[0], raw.normal[1], raw.normal[2]],
        degenerate: raw.degenerate == DEGENERATE_FLAG,
    }
}

/// A compiled, reusable Catmull-Rom patch evaluation pipeline.
pub struct GpuCatmullRomPatch {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCatmullRomPatch {
    /// Compiles the Catmull-Rom patch kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCatmullRomPatch {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch"),
            source: ShaderSource::Wgsl(CATMULL_ROM_PATCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCatmullRomPatch {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one
    /// [`CatmullRomPatchResult`] per query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`CatmullRomPatch::point`](prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch::point)
    /// and
    /// [`CatmullRomPatch::normal`](prism_render_architecture::ray_scene::catmull_rom_patch::CatmullRomPatch::normal)
    /// evaluated on `q`'s net at `(q.u, q.v)`. An empty `queries` slice yields
    /// an empty result — storage buffers cannot be zero-sized, so it is handled
    /// by an early return before any dispatch.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[CatmullRomPatchQuery],
    ) -> Vec<CatmullRomPatchResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_catmull_rom_patch_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_catmull_rom_patch_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
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
