//! `wgpu` compute twin of the four planar box-family signed-distance
//! primitives of the `CPU` golden path
//! ([`box_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_2d),
//! [`box_frame_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_frame_2d),
//! [`oriented_box_2d`](prism_render_architecture::ray_scene::sdf_primitives::oriented_box_2d)
//! and
//! [`rhombus_2d`](prism_render_architecture::ray_scene::sdf_primitives::rhombus_2d)).
//!
//! Two-dimensional implicit modelling needs *analytic* primitives whose exact
//! signed distance is known in closed form rather than sampled on a grid. The
//! reference derives four such distances in the plane: an axis-aligned
//! rectangle
//! ([`box_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_2d)),
//! its hollow frame
//! ([`box_frame_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_frame_2d),
//! the rectangle field's magnitude thinned by a wall thickness), a rectangle
//! given by its centre-line endpoints and width
//! ([`oriented_box_2d`](prism_render_architecture::ray_scene::sdf_primitives::oriented_box_2d),
//! Inigo-Quilez `sdOrientedBox`), and the rhombus
//! ([`rhombus_2d`](prism_render_architecture::ray_scene::sdf_primitives::rhombus_2d),
//! Inigo-Quilez `sdRhombus`). [`GpuSdfBox2d`] is the on-device twin: each
//! thread reads one point plus all four shapes' parameters and writes all four
//! signed distances, reproducing the reference closed forms with only `sqrt`,
//! `abs`, `min`, `max`, `clamp`, `select`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfBox2dQuery`] — a query `point` plus the
//! axis-aligned rectangle `box_half_extent`, the frame
//! `frame_half_extent`/`frame_thickness`, the oriented-box endpoints
//! `oriented_a`/`oriented_b` and its `oriented_thickness`, and the rhombus
//! `rhombus_half_diag` — and writes one [`SdfBox2dResult`] holding the four
//! signed distances `box_2d_sd`, `box_frame_2d_sd`, `oriented_box_2d_sd` and
//! `rhombus_2d_sd`.
//!
//! The rectangle kernel reduces the point to its per-axis overshoot
//! `q = |point| - half_extent`, forming the exterior overshoot length plus the
//! interior least-negative face distance. The frame kernel takes the magnitude
//! of that rectangle field and subtracts the wall thickness, carving a hollow
//! border. The oriented-box kernel rotates the centred point into the box's
//! local frame — local `x` along the unit endpoint direction `d`, local `y`
//! along its normal — then applies the same rectangle split against the
//! half-length and half-thickness. The rhombus kernel folds to the first
//! quadrant, finds the nearest point on the slanted edge via the `ndot`
//! parameter `h`, and recovers the interior sign from the edge half-plane test.
//!
//! # What stays on the host
//!
//! The planar domain and `CSG` operators that compose these atoms into complex
//! outlines, the extrusion that lifts them to 3D, and the surface-normal
//! estimation all stay on the host; the device sees only the four stateless,
//! fixed-width signed-distance evaluations, one query at a time, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every distance threads through `sqrt`, products and quotients, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (relative error
//! floored at `1e-6`), tight enough to catch a genuinely wrong port yet loose
//! enough to admit a legal last-place difference. The interior/exterior
//! `max(q.x, q.y) < 0` tests and the rhombus edge-sign test are ordered
//! comparisons; fixtures and the randomized sweep stay a safe margin away from
//! each boundary, so the `CPU` and `GPU` always pick the same branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`,
//! `min`, `max`, `clamp`, `select`, `+ - * /` and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry,
//! no `round` and no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
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

/// The inline `WGSL` source of the planar box-family signed-distance twin. The
/// crate ships the kernel as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`box_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_2d),
/// [`box_frame_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_frame_2d),
/// [`oriented_box_2d`](prism_render_architecture::ray_scene::sdf_primitives::oriented_box_2d)
/// and
/// [`rhombus_2d`](prism_render_architecture::ray_scene::sdf_primitives::rhombus_2d)
/// closed forms; see the module documentation for the algorithm.
const SDF_BOX2D_WGSL: &str = r#"
// Planar box-family signed-distance twin: one thread computes one query point's
// axis-aligned rectangle, hollow frame, oriented box and rhombus signed
// distances, mirroring the CPU golden
// `ray_scene::sdf_primitives::{box_2d, box_frame_2d, oriented_box_2d, rhombus_2d}`
// with only sqrt, abs, min, max, clamp, select, products and quotients. The
// planar CSG operators and the extrusion stay on the host.
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
    // Axis-aligned rectangle half extents.
    box_hx: f32,
    box_hy: f32,
    // Hollow frame half extents and wall thickness.
    frame_hx: f32,
    frame_hy: f32,
    frame_t: f32,
    // Oriented box: centre-line endpoints and width (thickness).
    oa_x: f32,
    oa_y: f32,
    ob_x: f32,
    ob_y: f32,
    o_t: f32,
    // Rhombus half diagonals.
    rho_x: f32,
    rho_y: f32,
}

struct Distances {
    // Axis-aligned rectangle signed distance.
    box_2d_sd: f32,
    // Hollow-frame signed distance.
    box_frame_2d_sd: f32,
    // Oriented-box signed distance.
    oriented_box_2d_sd: f32,
    // Rhombus signed distance.
    rhombus_2d_sd: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Euclidean length of a 2-vector; the shared planar reduction.
fn length2(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

// Sign with the scalar reference's convention (+1 at zero), expressed with
// select so no bare float equality appears. Away from the rejection-sampled
// boundaries the argument is never near zero, so this matches the golden.
fn signum(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Axis-aligned rectangle signed distance: exterior overshoot length plus the
// interior least-negative face distance.
fn box_2d(px: f32, py: f32, hx: f32, hy: f32) -> f32 {
    let qx = abs(px) - hx;
    let qy = abs(py) - hy;
    return length2(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Distances;

    // Axis-aligned rectangle.
    out.box_2d_sd = box_2d(q.px, q.py, q.box_hx, q.box_hy);

    // Hollow frame = |rectangle field| - wall thickness.
    out.box_frame_2d_sd = abs(box_2d(q.px, q.py, q.frame_hx, q.frame_hy)) - q.frame_t;

    // Oriented box: rotate the centred point into the box's local frame, then
    // apply the rectangle split against the half-length and half-thickness.
    let abx = q.ob_x - q.oa_x;
    let aby = q.ob_y - q.oa_y;
    let seg_len = length2(abx, aby);
    let dx = abx / seg_len;
    let dy = aby / seg_len;
    let cx = q.px - 0.5 * (q.oa_x + q.ob_x);
    let cy = q.py - 0.5 * (q.oa_y + q.ob_y);
    let o0 = abs(dx * cx + dy * cy) - 0.5 * seg_len;
    let o1 = abs(-dy * cx + dx * cy) - 0.5 * q.o_t;
    out.oriented_box_2d_sd =
        length2(max(o0, 0.0), max(o1, 0.0)) + min(max(o0, o1), 0.0);

    // Rhombus: fold to the first quadrant, nearest point on the slanted edge via
    // the ndot parameter h, interior sign from the edge half-plane test.
    let bx = q.rho_x;
    let by = q.rho_y;
    let rpx = abs(q.px);
    let rpy = abs(q.py);
    let nd = (bx - 2.0 * rpx) * bx - (by - 2.0 * rpy) * by;
    let h = clamp(nd / (bx * bx + by * by), -1.0, 1.0);
    let rqx = rpx - 0.5 * bx * (1.0 - h);
    let rqy = rpy - 0.5 * by * (1.0 + h);
    let rd = length2(rqx, rqy);
    out.rhombus_2d_sd = rd * signum(rpx * by + rpy * bx - bx * by);

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_BOX2D_WGSL`].
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
/// the point components plus all four shapes' parameters. Fourteen `f32` fields
/// pack to a `56`-byte, `4`-byte-aligned stride with no trailing pad.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Rectangle half extent `x`.
    box_hx: f32,
    /// Rectangle half extent `y`.
    box_hy: f32,
    /// Frame half extent `x`.
    frame_hx: f32,
    /// Frame half extent `y`.
    frame_hy: f32,
    /// Frame wall thickness.
    frame_t: f32,
    /// Oriented-box endpoint `a.x`.
    oa_x: f32,
    /// Oriented-box endpoint `a.y`.
    oa_y: f32,
    /// Oriented-box endpoint `b.x`.
    ob_x: f32,
    /// Oriented-box endpoint `b.y`.
    ob_y: f32,
    /// Oriented-box width (thickness).
    o_t: f32,
    /// Rhombus half diagonal `x`.
    rho_x: f32,
    /// Rhombus half diagonal `y`.
    rho_y: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the four signed distances packing to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Rectangle signed distance.
    box_2d_sd: f32,
    /// Hollow-frame signed distance.
    box_frame_2d_sd: f32,
    /// Oriented-box signed distance.
    oriented_box_2d_sd: f32,
    /// Rhombus signed distance.
    rhombus_2d_sd: f32,
}

/// One query for the planar box-family signed-distance twin: the query `point`
/// plus the rectangle, frame, oriented-box and rhombus shape parameters.
///
/// `point` is the evaluation position; `box_half_extent` are the
/// [`box_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_2d)
/// half extents; `frame_half_extent`/`frame_thickness` are the
/// [`box_frame_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_frame_2d)
/// half extents and wall thickness; `oriented_a`/`oriented_b`/
/// `oriented_thickness` are the
/// [`oriented_box_2d`](prism_render_architecture::ray_scene::sdf_primitives::oriented_box_2d)
/// centre-line endpoints and width; `rhombus_half_diag` are the
/// [`rhombus_2d`](prism_render_architecture::ray_scene::sdf_primitives::rhombus_2d)
/// half diagonals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfBox2dQuery {
    /// Query point `[x, y]`.
    pub point: [f32; 2],
    /// Rectangle half extents `[x, y]`.
    pub box_half_extent: [f32; 2],
    /// Frame half extents `[x, y]`.
    pub frame_half_extent: [f32; 2],
    /// Frame wall thickness.
    pub frame_thickness: f32,
    /// Oriented-box centre-line endpoint `a` `[x, y]`.
    pub oriented_a: [f32; 2],
    /// Oriented-box centre-line endpoint `b` `[x, y]`.
    pub oriented_b: [f32; 2],
    /// Oriented-box width (thickness).
    pub oriented_thickness: f32,
    /// Rhombus half diagonals `[x, y]`.
    pub rhombus_half_diag: [f32; 2],
}

impl SdfBox2dQuery {
    /// Builds a query from the point and all four shapes' parameters.
    #[must_use]
    pub const fn new(
        point: [f32; 2],
        box_half_extent: [f32; 2],
        frame_half_extent: [f32; 2],
        frame_thickness: f32,
        oriented_a: [f32; 2],
        oriented_b: [f32; 2],
        oriented_thickness: f32,
        rhombus_half_diag: [f32; 2],
    ) -> SdfBox2dQuery {
        SdfBox2dQuery {
            point,
            box_half_extent,
            frame_half_extent,
            frame_thickness,
            oriented_a,
            oriented_b,
            oriented_thickness,
            rhombus_half_diag,
        }
    }
}

/// One resolved query of the planar box-family signed-distance twin: the
/// rectangle, frame, oriented-box and rhombus signed distances at the point.
///
/// `box_2d_sd` is
/// [`box_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_2d);
/// `box_frame_2d_sd` is
/// [`box_frame_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_frame_2d);
/// `oriented_box_2d_sd` is
/// [`oriented_box_2d`](prism_render_architecture::ray_scene::sdf_primitives::oriented_box_2d);
/// `rhombus_2d_sd` is
/// [`rhombus_2d`](prism_render_architecture::ray_scene::sdf_primitives::rhombus_2d).
/// Each is negative inside the solid, positive outside, zero on the boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfBox2dResult {
    /// Rectangle signed distance.
    pub box_2d_sd: f32,
    /// Hollow-frame signed distance.
    pub box_frame_2d_sd: f32,
    /// Oriented-box signed distance.
    pub oriented_box_2d_sd: f32,
    /// Rhombus signed distance.
    pub rhombus_2d_sd: f32,
}

/// Encodes one [`SdfBox2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfBox2dQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        box_hx: q.box_half_extent[0],
        box_hy: q.box_half_extent[1],
        frame_hx: q.frame_half_extent[0],
        frame_hy: q.frame_half_extent[1],
        frame_t: q.frame_thickness,
        oa_x: q.oriented_a[0],
        oa_y: q.oriented_a[1],
        ob_x: q.oriented_b[0],
        ob_y: q.oriented_b[1],
        o_t: q.oriented_thickness,
        rho_x: q.rhombus_half_diag[0],
        rho_y: q.rhombus_half_diag[1],
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfBox2dResult`].
fn decode_result(raw: &GpuResult) -> SdfBox2dResult {
    SdfBox2dResult {
        box_2d_sd: raw.box_2d_sd,
        box_frame_2d_sd: raw.box_frame_2d_sd,
        oriented_box_2d_sd: raw.oriented_box_2d_sd,
        rhombus_2d_sd: raw.rhombus_2d_sd,
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

/// A compiled, reusable planar box-family signed-distance compute pipeline,
/// twinning the `CPU` golden
/// [`box_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_2d),
/// [`box_frame_2d`](prism_render_architecture::ray_scene::sdf_primitives::box_frame_2d),
/// [`oriented_box_2d`](prism_render_architecture::ray_scene::sdf_primitives::oriented_box_2d)
/// and
/// [`rhombus_2d`](prism_render_architecture::ray_scene::sdf_primitives::rhombus_2d).
pub struct GpuSdfBox2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfBox2d {
    /// Compiles the planar box-family signed-distance compute pipeline on the
    /// given context.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfBox2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_box2d_module"),
            source: ShaderSource::Wgsl(SDF_BOX2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_box2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_box2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_box2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfBox2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfBox2dResult`] per
    /// input, in order.
    ///
    /// The signed distances match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[SdfBox2dQuery]) -> Vec<SdfBox2dResult> {
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
            label: Some("prism_volumetric_sdf_box2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_box2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_box2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_box2d_bind_group"),
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
            label: Some("prism_volumetric_sdf_box2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_box2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_box2d_pass"),
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
