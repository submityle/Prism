//! `wgpu` compute twin of three 2D analytic signed-distance *gradient*
//! (surface-normal) primitives of the `CPU` golden path
//! ([`box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::box_2d_gradient),
//! [`circle_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::circle_2d_gradient)
//! and [`rounded_box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::rounded_box_2d_gradient)).
//!
//! Implicit 2D modelling needs the *analytic* gradient of a signed-distance
//! field — the exact outward unit normal in closed form — rather than a central
//! difference sampled on a grid. The reference derives three such gradients: the
//! axis-aligned rectangle [`box_2d_gradient`] (axis normal along the straight
//! walls, radial normal in the corner region), the circle
//! [`circle_2d_gradient`] (the outward radial unit vector `point / |point|`
//! independent of radius), and the per-corner [`rounded_box_2d_gradient`] (which
//! reduces to [`box_2d_gradient`] at the quadrant's shrunk half-extent because
//! the constant corner offset drops out of the gradient). [`GpuSdfGradient2d`]
//! is the on-device twin: each thread reads one point plus every shape's
//! parameters and writes all three analytic gradient vectors, reproducing the
//! reference closed forms with only `sqrt`, `abs`, `min`, `max`, `select`,
//! products and quotients — never a finite difference.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfGradient2dQuery`] — a query `point` plus the
//! box half-extent (`box_half_extent`), the rounded-box half-extent
//! (`rounded_box_half_extent`) and its four per-corner radii
//! (`rounded_box_radii`, ordered `[top_right, bottom_right, top_left,
//! bottom_left]`) — and writes one [`SdfGradient2dResult`] holding the three
//! analytic gradient vectors `box_2d_gradient_value`, `circle_2d_gradient_value`
//! and `rounded_box_2d_gradient_value`.
//!
//! The `box_2d_gradient` kernel folds the point into the first quadrant via
//! `abs`, forms the corner offset `q = |point| - half_extent` and its clamped
//! overshoot `m = max(q, 0)`: when the overshoot has positive length the
//! gradient is the normalised overshoot with each axis' original sign restored;
//! otherwise (interior) it is the unit vector along the least-negative axis.
//! The `circle_2d_gradient` kernel returns the normalised position, falling back
//! to the zero vector at the centre where the direction is undefined. The
//! `rounded_box_2d_gradient` kernel selects the active per-corner radius from
//! the point's quadrant and evaluates the `box_2d_gradient` closed form at the
//! shrunk half-extent `half_extent - r`.
//!
//! # What stays on the host
//!
//! The signed-distance fields themselves, the domain and `CSG` operators that
//! compose these atoms, and the ray-marcher that evaluates them all stay on the
//! host; the device sees only the three stateless, fixed-width gradient
//! evaluations, one query at a time, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! The interior branches return exact axis unit vectors (integer components, no
//! `sqrt`), so they are bit-exact; the exterior and circle branches thread
//! through one `sqrt` and a divide, so the `CPU` and `GPU` are not bit-exact and
//! a `GPU` result may land a few units in the last place from the scalar
//! reference. The parity test asserts each gradient component within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, loose enough to admit a legal
//! last-place difference yet tight enough to catch a wrong port. Fixtures and
//! the randomized sweep keep every query a safe margin off each gradient's
//! discontinuity (the coordinate axes where the sign and radius selection
//! switch, the shrunk-box corner/interior diagonal, and the circle centre) via
//! rejection sampling, so neither side ever selects a different branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`, `min`,
//! `max`, `select`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and no
//! `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
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

/// The portable core-`WGSL` 2D gradient kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::box_2d_gradient),
/// [`circle_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::circle_2d_gradient)
/// and [`rounded_box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::rounded_box_2d_gradient)
/// closed forms; see the module documentation for the algorithm.
const SDF_GRADIENT2D_WGSL: &str = r#"
// 2D analytic signed-distance gradient twin: one thread computes one query
// point's box, circle and rounded-box surface normals (unit gradients),
// mirroring the CPU golden
// `ray_scene::sdf_primitives::{box_2d_gradient, circle_2d_gradient, rounded_box_2d_gradient}`
// with only sqrt, abs, min, max, select, products and quotients — never a
// finite difference. The signed-distance fields, the domain/CSG operators and
// the ray-marcher stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point components.
    px: f32,
    py: f32,
    // Box half-extents along x and y.
    box_hx: f32,
    box_hy: f32,
    // Rounded-box half-extents along x and y.
    rb_hx: f32,
    rb_hy: f32,
    // Rounded-box per-corner radii: top-right, bottom-right, top-left,
    // bottom-left.
    rb_r0: f32,
    rb_r1: f32,
    rb_r2: f32,
    rb_r3: f32,
    pad0: f32,
    pad1: f32,
}

struct Gradients {
    // Box gradient (unit outward normal) components.
    box_gx: f32,
    box_gy: f32,
    // Circle gradient components.
    circle_gx: f32,
    circle_gy: f32,
    // Rounded-box gradient components.
    rb_gx: f32,
    rb_gy: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Gradients>;

// Rust `f32::signum` for the non-zero domain: `+1` for non-negative inputs,
// `-1` for negative inputs. Fixtures stay off the exact zero so the two-way
// select never straddles the convention boundary.
fn signum_rs(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Exact 2D box surface normal: fold into the first quadrant, normalise the
// clamped overshoot outside, pick the least-negative axis inside.
fn box_2d_gradient_v(point: vec2<f32>, half_extent: vec2<f32>) -> vec2<f32> {
    let qx = abs(point.x) - half_extent.x;
    let qy = abs(point.y) - half_extent.y;
    let mx = max(qx, 0.0);
    let my = max(qy, 0.0);
    let len = sqrt(mx * mx + my * my);
    if (len > 0.0) {
        return vec2<f32>(signum_rs(point.x) * mx / len, signum_rs(point.y) * my / len);
    }
    if (qx >= qy) {
        return vec2<f32>(signum_rs(point.x), 0.0);
    }
    return vec2<f32>(0.0, signum_rs(point.y));
}

// Exact circle surface normal: the normalised position, zero at the centre.
fn circle_2d_gradient_v(point: vec2<f32>) -> vec2<f32> {
    let l = sqrt(point.x * point.x + point.y * point.y);
    if (l > 0.0) {
        return vec2<f32>(point.x / l, point.y / l);
    }
    return vec2<f32>(0.0, 0.0);
}

// Exact rounded-box surface normal: pick the active per-corner radius from the
// point's quadrant, evaluate the box gradient at the shrunk half-extent.
fn rounded_box_2d_gradient_v(
    point: vec2<f32>,
    half_extent: vec2<f32>,
    r0: f32,
    r1: f32,
    r2: f32,
    r3: f32,
) -> vec2<f32> {
    let rx = select(r2, r0, point.x > 0.0);
    let ry = select(r3, r1, point.x > 0.0);
    let r = select(ry, rx, point.y > 0.0);
    return box_2d_gradient_v(point, vec2<f32>(half_extent.x - r, half_extent.y - r));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let point = vec2<f32>(q.px, q.py);

    let box_g = box_2d_gradient_v(point, vec2<f32>(q.box_hx, q.box_hy));
    let circle_g = circle_2d_gradient_v(point);
    let rb_g = rounded_box_2d_gradient_v(
        point,
        vec2<f32>(q.rb_hx, q.rb_hy),
        q.rb_r0,
        q.rb_r1,
        q.rb_r2,
        q.rb_r3,
    );

    var out: Gradients;
    out.box_gx = box_g.x;
    out.box_gy = box_g.y;
    out.circle_gx = circle_g.x;
    out.circle_gy = circle_g.y;
    out.rb_gx = rb_g.x;
    out.rb_gy = rb_g.y;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching the `WGSL` `Params`.
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
/// the query point plus every shape's parameters, padded to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Box half-extent along `x`.
    box_hx: f32,
    /// Box half-extent along `y`.
    box_hy: f32,
    /// Rounded-box half-extent along `x`.
    rb_hx: f32,
    /// Rounded-box half-extent along `y`.
    rb_hy: f32,
    /// Rounded-box top-right corner radius.
    rb_r0: f32,
    /// Rounded-box bottom-right corner radius.
    rb_r1: f32,
    /// Rounded-box top-left corner radius.
    rb_r2: f32,
    /// Rounded-box bottom-left corner radius.
    rb_r3: f32,
    /// Padding word to a `48`-byte stride.
    pad0: f32,
    /// Padding word to a `48`-byte stride.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Gradients`
/// struct: the three gradient vectors plus two pad words to a `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Box gradient `x`.
    box_gx: f32,
    /// Box gradient `y`.
    box_gy: f32,
    /// Circle gradient `x`.
    circle_gx: f32,
    /// Circle gradient `y`.
    circle_gy: f32,
    /// Rounded-box gradient `x`.
    rb_gx: f32,
    /// Rounded-box gradient `y`.
    rb_gy: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One query for the 2D gradient twin: the query `point` plus the box and
/// rounded-box shape parameters (the circle gradient needs only the point).
///
/// `point` is the evaluation position; `box_half_extent` is the
/// [`box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::box_2d_gradient)
/// rectangle half-extent; `rounded_box_half_extent` and `rounded_box_radii`
/// (ordered `[top_right, bottom_right, top_left, bottom_left]`) are the
/// [`rounded_box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::rounded_box_2d_gradient)
/// half-extent and per-corner radii.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfGradient2dQuery {
    /// Query point `[x, y]`.
    pub point: [f32; 2],
    /// Box rectangle half-extent `[x, y]`.
    pub box_half_extent: [f32; 2],
    /// Rounded-box rectangle half-extent `[x, y]`.
    pub rounded_box_half_extent: [f32; 2],
    /// Rounded-box per-corner radii `[top_right, bottom_right, top_left, bottom_left]`.
    pub rounded_box_radii: [f32; 4],
}

impl SdfGradient2dQuery {
    /// Builds a query from the point and the box and rounded-box parameters.
    #[must_use]
    pub const fn new(
        point: [f32; 2],
        box_half_extent: [f32; 2],
        rounded_box_half_extent: [f32; 2],
        rounded_box_radii: [f32; 4],
    ) -> SdfGradient2dQuery {
        SdfGradient2dQuery {
            point,
            box_half_extent,
            rounded_box_half_extent,
            rounded_box_radii,
        }
    }
}

/// One resolved query of the 2D gradient twin: the box, circle and rounded-box
/// analytic gradient vectors (unit outward normals) at the query point.
///
/// `box_2d_gradient_value` is
/// [`box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::box_2d_gradient);
/// `circle_2d_gradient_value` is
/// [`circle_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::circle_2d_gradient);
/// `rounded_box_2d_gradient_value` is
/// [`rounded_box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::rounded_box_2d_gradient).
/// Each is a unit vector away from the surface, or the zero vector where the
/// gradient direction is undefined (the circle centre).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfGradient2dResult {
    /// Box gradient (unit outward normal) `[x, y]`.
    pub box_2d_gradient_value: [f32; 2],
    /// Circle gradient (unit outward normal) `[x, y]`.
    pub circle_2d_gradient_value: [f32; 2],
    /// Rounded-box gradient (unit outward normal) `[x, y]`.
    pub rounded_box_2d_gradient_value: [f32; 2],
}

/// Encodes one [`SdfGradient2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfGradient2dQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        box_hx: q.box_half_extent[0],
        box_hy: q.box_half_extent[1],
        rb_hx: q.rounded_box_half_extent[0],
        rb_hy: q.rounded_box_half_extent[1],
        rb_r0: q.rounded_box_radii[0],
        rb_r1: q.rounded_box_radii[1],
        rb_r2: q.rounded_box_radii[2],
        rb_r3: q.rounded_box_radii[3],
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfGradient2dResult`].
fn decode_result(raw: &GpuResult) -> SdfGradient2dResult {
    SdfGradient2dResult {
        box_2d_gradient_value: [raw.box_gx, raw.box_gy],
        circle_2d_gradient_value: [raw.circle_gx, raw.circle_gy],
        rounded_box_2d_gradient_value: [raw.rb_gx, raw.rb_gy],
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

/// A compiled, reusable 2D gradient compute pipeline, twinning the `CPU` golden
/// [`box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::box_2d_gradient),
/// [`circle_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::circle_2d_gradient)
/// and [`rounded_box_2d_gradient`](prism_render_architecture::ray_scene::sdf_primitives::rounded_box_2d_gradient).
pub struct GpuSdfGradient2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfGradient2d {
    /// Compiles the 2D gradient kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfGradient2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_gradient2d"),
            source: ShaderSource::Wgsl(SDF_GRADIENT2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_gradient2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_gradient2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_gradient2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfGradient2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfGradient2dResult`]
    /// per input, in order.
    ///
    /// The gradient vectors match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfGradient2dQuery],
    ) -> Vec<SdfGradient2dResult> {
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
            label: Some("prism_volumetric_sdf_gradient2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_gradient2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_gradient2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_gradient2d_bind_group"),
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
            label: Some("prism_volumetric_sdf_gradient2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_gradient2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_gradient2d_pass"),
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
