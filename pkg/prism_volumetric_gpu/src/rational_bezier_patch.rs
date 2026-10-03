//! `wgpu` compute twin of the rational bicubic Bézier (single-span `NURBS`)
//! surface patch evaluator from this repository's
//! `prism_render_architecture::ray_scene::rational_bezier_patch`.
//!
//! The `CPU` golden `RationalBezierPatch` owns a `4x4` control net with a
//! positive weight per control point and evaluates the projective surface
//! `P(u, v) = sum Bi(u) Bj(v) wij Pij / sum Bi(u) Bj(v) wij`, a single-span
//! `NURBS` patch (uniform clamped knots). Each control point is lifted to the
//! homogeneous coordinate `[w*x, w*y, w*z, w]` and De Casteljau's algorithm
//! runs there — pure linear interpolation, no transcendental basis — then the
//! result is projected back with one division by the homogeneous weight.
//!
//! # What is twinned
//!
//! [`GpuRationalBezierPatch`] reproduces four public reference methods per
//! query at a parameter `(u, v)`:
//!
//! - `point`: the projected surface position `H.xyz / H.w` (falling back to the
//!   raw numerator when the homogeneous weight collapses).
//! - `partial_u` and `partial_v`: the projected partial derivatives via the
//!   quotient rule `dP/du = (Hu.xyz * H.w - H.xyz * Hu.w) / H.w^2`.
//! - `normal`: the unit cross product of the quotient-rule numerators (whose
//!   shared `1 / H.w^2` factor cancels under normalization), with the
//!   reference's interior-nudging fallback loop.
//!
//! The twin runs one thread per query and reproduces the same answers the
//! reference produces, so a passing real-device parity test is direct evidence
//! the ported kernel evaluates the same rational surface and classifies the
//! same degenerate case, not merely that its shader compiles.
//!
//! # Degeneracy
//!
//! The reference `normal` tries up to four samples, each nudged a few steps
//! toward the patch interior, and returns the first whose numerator cross
//! product is non-degenerate; failing all four it returns the fixed fallback
//! `[0, 0, 1]`. The twin mirrors this loop and reports a `degenerate` flag that
//! is `1` exactly when every step collapsed and the fallback was taken, `0`
//! otherwise. The division guards (`abs(H.w) > 1e-20`, `H.w^2 > 1e-20`,
//! `len > 1e-20`) are reproduced branch for branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt`, `+ - * /` — with no `sin`, `cos`, `tan`, `exp`,
//! `log`, `pow` or optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The only non-rational operation is the normalization
//! `sqrt`, matching the reference's `f32::sqrt` call.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test therefore asserts a tolerance (`abs_diff <= 1e-4 ||
//! rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on the `f32` fields while pinning the
//! `degenerate` flag exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::rational_bezier_patch`；无第三方引擎源码或衍生代码。

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly
/// default shared by every one-thread-per-element kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Number of control points in the `4x4` net; also the weight count.
const CONTROL_COUNT: usize = 16;

/// Inlined `WGSL` compute shader source. Keeping it in the Rust binary avoids
/// shipping a sidecar asset and keeps the twin and its kernel versioned as a
/// single source file. The single entry point `solve` mirrors the `CPU` golden
/// `RationalBezierPatch` methods branch for branch; see the module
/// documentation for the algorithm.
const RATIONAL_BEZIER_PATCH_WGSL: &str = r#"
// Rational bicubic Bézier (single-span NURBS) surface twin: one thread per
// query lifts the 4x4 control net to homogeneous coordinates, runs De
// Casteljau along u then v (pure linear interpolation), projects the point,
// forms the quotient-rule partial derivatives and the unit normal from their
// cross product. It mirrors the CPU golden RationalBezierPatch, uses only the
// portable core-WGSL subset (min/max/clamp/abs/sqrt and + - * /) and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::rational_bezier_patch；无第三方引擎源码或衍生代码。

// Division/normalization guard matching the reference 1e-20 thresholds.
const GUARD: f32 = 1.0e-20;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. The 4x4 control net is packed as 16 vec4 slots carrying
// (point.xyz, weight), followed by the surface parameter (u, v) and padding.
struct Query {
    cp: array<vec4<f32>, 16>,
    u: f32,
    v: f32,
    pad0: f32,
    pad1: f32,
}

// One result: the projected point, the degenerate flag, both partials and the
// oriented unit normal, each vec3 on its own 16-byte std430 slot.
struct Result {
    point: vec3<f32>,
    degenerate: u32,
    partial_u: vec3<f32>,
    pad0: f32,
    partial_v: vec3<f32>,
    pad1: f32,
    normal: vec3<f32>,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Component-wise linear interpolation a + (b - a) * t over a 4-vector.
fn lerp4(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    return a + (b - a) * t;
}

// Cubic De Casteljau point at t over four homogeneous control points.
fn cubic_point(p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, p3: vec4<f32>, t: f32) -> vec4<f32> {
    let a = lerp4(p0, p1, t);
    let b = lerp4(p1, p2, t);
    let c = lerp4(p2, p3, t);
    let d = lerp4(a, b, t);
    let e = lerp4(b, c, t);
    return lerp4(d, e, t);
}

// Cubic De Casteljau derivative at t: 3 * the quadratic De Casteljau over the
// adjacent control-point differences, in homogeneous coordinates.
fn cubic_deriv(p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, p3: vec4<f32>, t: f32) -> vec4<f32> {
    let d0 = p1 - p0;
    let d1 = p2 - p1;
    let d2 = p3 - p2;
    let a = lerp4(d0, d1, t);
    let b = lerp4(d1, d2, t);
    let q = lerp4(a, b, t);
    return q * 3.0;
}

// Homogeneous surface point H(u, v) = [w*x, w*y, w*z, w] from the lifted net.
fn homogeneous_point(h: array<vec4<f32>, 16>, u: f32, v: f32) -> vec4<f32> {
    let c0 = cubic_point(h[0], h[1], h[2], h[3], u);
    let c1 = cubic_point(h[4], h[5], h[6], h[7], u);
    let c2 = cubic_point(h[8], h[9], h[10], h[11], u);
    let c3 = cubic_point(h[12], h[13], h[14], h[15], u);
    return cubic_point(c0, c1, c2, c3, v);
}

// Homogeneous partial dH/du.
fn homogeneous_partial_u(h: array<vec4<f32>, 16>, u: f32, v: f32) -> vec4<f32> {
    let c0 = cubic_deriv(h[0], h[1], h[2], h[3], u);
    let c1 = cubic_deriv(h[4], h[5], h[6], h[7], u);
    let c2 = cubic_deriv(h[8], h[9], h[10], h[11], u);
    let c3 = cubic_deriv(h[12], h[13], h[14], h[15], u);
    return cubic_point(c0, c1, c2, c3, v);
}

// Homogeneous partial dH/dv.
fn homogeneous_partial_v(h: array<vec4<f32>, 16>, u: f32, v: f32) -> vec4<f32> {
    let c0 = cubic_point(h[0], h[1], h[2], h[3], u);
    let c1 = cubic_point(h[4], h[5], h[6], h[7], u);
    let c2 = cubic_point(h[8], h[9], h[10], h[11], u);
    let c3 = cubic_point(h[12], h[13], h[14], h[15], u);
    return cubic_deriv(c0, c1, c2, c3, v);
}

// Quotient-rule numerator of dP/du: Hu.xyz * H.w - H.xyz * Hu.w.
fn partial_u_numerator(h: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let hh = homogeneous_point(h, u, v);
    let hu = homogeneous_partial_u(h, u, v);
    return vec3<f32>(
        hu.x * hh.w - hh.x * hu.w,
        hu.y * hh.w - hh.y * hu.w,
        hu.z * hh.w - hh.z * hu.w,
    );
}

// Quotient-rule numerator of dP/dv: Hv.xyz * H.w - H.xyz * Hv.w.
fn partial_v_numerator(h: array<vec4<f32>, 16>, u: f32, v: f32) -> vec3<f32> {
    let hh = homogeneous_point(h, u, v);
    let hv = homogeneous_partial_v(h, u, v);
    return vec3<f32>(
        hv.x * hh.w - hh.x * hv.w,
        hv.y * hh.w - hh.y * hv.w,
        hv.z * hh.w - hh.z * hv.w,
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Lift each control point to the homogeneous coordinate [w*x, w*y, w*z, w].
    var h: array<vec4<f32>, 16>;
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
        let c = q.cp[k];
        h[k] = vec4<f32>(c.x * c.w, c.y * c.w, c.z * c.w, c.w);
    }

    let u = q.u;
    let v = q.v;

    // Projected surface point: divide the homogeneous point by its weight,
    // falling back to the raw numerator when the weight collapses.
    let hp = homogeneous_point(h, u, v);
    var point = vec3<f32>(hp.x, hp.y, hp.z);
    if (abs(hp.w) > GUARD) {
        let inv_w = 1.0 / hp.w;
        point = vec3<f32>(hp.x * inv_w, hp.y * inv_w, hp.z * inv_w);
    }

    // Projected partials via the quotient rule, divided by H.w^2 when safe.
    let w2 = hp.w * hp.w;
    let nu = partial_u_numerator(h, u, v);
    let nv = partial_v_numerator(h, u, v);
    var partial_u = nu;
    var partial_v = nv;
    if (w2 > GUARD) {
        let inv_w2 = 1.0 / w2;
        partial_u = nu * inv_w2;
        partial_v = nv * inv_w2;
    }

    // Analytic normal: cross of the quotient-rule numerators, retried a few
    // steps toward the interior; falls back to +Z when every step collapses.
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    var degenerate: u32 = 1u;
    for (var step: i32 = 0; step < 4; step = step + 1) {
        let eps = 1.0e-3 * f32(step);
        let uu = clamp(u + eps, 0.0, 1.0);
        let vv = clamp(v + eps, 0.0, 1.0);
        let du = partial_u_numerator(h, uu, vv);
        let dv = partial_v_numerator(h, uu, vv);
        let cx = du.y * dv.z - du.z * dv.y;
        let cy = du.z * dv.x - du.x * dv.z;
        let cz = du.x * dv.y - du.y * dv.x;
        let cross = vec3<f32>(cx, cy, cz);
        let len2 = cross.x * cross.x + cross.y * cross.y + cross.z * cross.z;
        if (len2 > 0.0) {
            let len = sqrt(len2);
            if (len > GUARD) {
                normal = cross * (1.0 / len);
            }
            degenerate = 0u;
            break;
        }
    }

    var out: Result;
    out.point = point;
    out.degenerate = degenerate;
    out.partial_u = partial_u;
    out.pad0 = 0.0;
    out.partial_v = partial_v;
    out.pad1 = 0.0;
    out.normal = normal;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// One rational-Bézier-patch query: the `4x4` control net, its 16 weights and
/// the surface parameter at which to evaluate.
///
/// Mirrors a single reference `RationalBezierPatch` evaluation. The net is
/// row-major (`control[row * 4 + col]`, `col` along `u`, `row` along `v`) and
/// index-aligned with `weights`. Weights are expected strictly positive so the
/// surface stays in the convex hull of the net. Derives only [`PartialEq`] (no
/// [`Eq`] / [`Hash`]) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RationalBezierPatchQuery {
    /// The `4x4` control net, row-major (`col` along `u`, `row` along `v`).
    pub control: [[f32; 3]; CONTROL_COUNT],
    /// Per-control-point weights, row-major and index-aligned with `control`.
    pub weights: [f32; CONTROL_COUNT],
    /// Surface parameter along `u`, expected in `[0, 1]`.
    pub u: f32,
    /// Surface parameter along `v`, expected in `[0, 1]`.
    pub v: f32,
}

impl RationalBezierPatchQuery {
    /// Builds a query over the control net, weights and parameter `(u, v)`.
    #[must_use]
    pub const fn new(
        control: [[f32; 3]; CONTROL_COUNT],
        weights: [f32; CONTROL_COUNT],
        u: f32,
        v: f32,
    ) -> RationalBezierPatchQuery {
        RationalBezierPatchQuery {
            control,
            weights,
            u,
            v,
        }
    }
}

/// The evaluated surface sample for one query, the host-side mirror of the
/// kernel's `Result` lane.
///
/// `point`, `partial_u`, `partial_v` and `normal` are the projected position,
/// both partial derivatives and the oriented unit normal; `degenerate` is `1`
/// exactly when the normal computation exhausted its interior-nudging retries
/// and returned the fixed `[0, 0, 1]` fallback. Derives only [`PartialEq`] (no
/// [`Eq`] / [`Hash`]) because it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RationalBezierPatchResult {
    /// Projected surface position `H.xyz / H.w`.
    pub point: [f32; 3],
    /// Projected partial derivative `dP/du`.
    pub partial_u: [f32; 3],
    /// Projected partial derivative `dP/dv`.
    pub partial_v: [f32; 3],
    /// Oriented unit surface normal.
    pub normal: [f32; 3],
    /// Degenerate flag: `1` when the normal fell back to `[0, 0, 1]`.
    pub degenerate: u32,
}

/// `repr(C)` `std430` layout of one packed query: 16 `vec4` slots each holding
/// `(control.xyz, weight)`, then `(u, v)` with two pad lanes — `272` bytes,
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Control net packed as `(point.xyz, weight)` per slot.
    cp: [[f32; 4]; CONTROL_COUNT],
    /// Surface parameter along `u`.
    u: f32,
    /// Surface parameter along `v`.
    v: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &RationalBezierPatchQuery) -> GpuQuery {
        let mut cp = [[0.0f32; 4]; CONTROL_COUNT];
        for (slot, (p, w)) in cp
            .iter_mut()
            .zip(query.control.iter().zip(query.weights.iter()))
        {
            *slot = [p[0], p[1], p[2], *w];
        }
        GpuQuery {
            cp,
            u: query.u,
            v: query.v,
            pad0: 0.0,
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: the point and the degenerate flag,
/// then both partials and the normal, each `vec3` on its own `16`-byte slot —
/// `64` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Projected surface position.
    point: [f32; 3],
    /// Degenerate flag (`1` = fallback normal taken).
    degenerate: u32,
    /// Projected `dP/du`.
    partial_u: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// Projected `dP/dv`.
    partial_v: [f32; 3],
    /// Padding lane.
    pad1: f32,
    /// Oriented unit normal.
    normal: [f32; 3],
    /// Padding lane.
    pad2: f32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// round the uniform block out to `16` bytes.
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

/// Decodes one packed `GpuResult` into the public [`RationalBezierPatchResult`].
fn decode_result(raw: &GpuResult) -> RationalBezierPatchResult {
    RationalBezierPatchResult {
        point: raw.point,
        partial_u: raw.partial_u,
        partial_v: raw.partial_v,
        normal: raw.normal,
        degenerate: raw.degenerate,
    }
}

/// Builds one storage/uniform buffer bind-group-layout entry.
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

/// On-device twin of the rational bicubic Bézier patch evaluator.
///
/// Owns the compiled [`ComputePipeline`] and its [`BindGroupLayout`]; build it
/// once with [`GpuRationalBezierPatch::new`] and reuse it across
/// [`GpuRationalBezierPatch::evaluate`] calls.
pub struct GpuRationalBezierPatch {
    /// The compiled shader module (retained so the pipeline stays valid).
    #[expect(
        dead_code,
        reason = "retained so the compiled module outlives the pipeline"
    )]
    module: ShaderModule,
    /// The bind group layout shared by every dispatch.
    layout: BindGroupLayout,
    /// The compute pipeline running the `solve` entry point.
    pipeline: ComputePipeline,
}

impl GpuRationalBezierPatch {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRationalBezierPatch {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_rational_bezier_patch_shader"),
            source: ShaderSource::Wgsl(RATIONAL_BEZIER_PATCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_rational_bezier_patch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_rational_bezier_patch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_rational_bezier_patch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRationalBezierPatch {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query on-device and returns one
    /// [`RationalBezierPatchResult`] per input, in order.
    ///
    /// Each result equals the reference `RationalBezierPatch` answer to within
    /// the tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RationalBezierPatchQuery],
    ) -> Vec<RationalBezierPatchResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_rational_bezier_patch_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_rational_bezier_patch_output"),
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
            label: Some("prism_volumetric_rational_bezier_patch_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_rational_bezier_patch_bind_group"),
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
            label: Some("prism_volumetric_rational_bezier_patch_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_rational_bezier_patch_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_rational_bezier_patch_pass"),
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
    }
}
