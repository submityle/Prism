//! `wgpu` compute twin of three exact 3D signed-distance primitives from the
//! reference ray-scene `SDF` primitive library
//! (`prism_render_architecture::ray_scene::sdf_primitives`).
//!
//! Three closed-form distance evaluators are twinned, each the exact field of a
//! 3D shape and all transcendental-free (only `abs`, `clamp`, `min`, `max`,
//! products and one `sqrt`):
//!
//! - `box_frame(point, half_extent, thickness)`: the exact field of the hollow
//!   wireframe of an axis-aligned box — the twelve square-section bars of width
//!   `thickness` running along the edges of a box with the given `half_extent`.
//!   The point is folded into the positive octant and offset by the frame
//!   thickness, then the three axis-aligned bar families are measured with the
//!   interior / exterior split and combined with a minimum, so the hollow
//!   interior and the bar solids both carry a true distance.
//! - `vertical_capsule(point, height, radius)`: the exact field of the upright
//!   pill — the segment from the origin to `(0, height, 0)` inflated by
//!   `radius`. Clamping the query height into `[0, height]` snaps it onto the
//!   nearest axis point; the field is the distance to that point minus the
//!   radius.
//! - `segment_3d(point, a, b)`: the exact unsigned distance to the finite 3D
//!   segment `a`-`b` (the zero-radius skeleton of a capsule). The point is
//!   projected onto the segment with the projection parameter clamped to
//!   `[0, 1]`; a degenerate zero-length segment pins the parameter at the start
//!   point.
//!
//! # What is twinned
//!
//! One thread resolves one query. Each [`SdfPrimitives3dQuery`] carries the
//! point and the per-shape parameters; the kernel evaluates all three fields
//! and writes one [`SdfPrimitives3dResult`] holding the three distances. The
//! twin spells out the same closed form with the same ordered clamps, folds and
//! guarded divisions as the reference, so a passing real-device parity test is
//! direct evidence the ported kernel evaluates the same distance the reference
//! does, not merely that the shader compiles.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: each field is a fixed,
//! bounded sequence of clamps, products and one `sqrt` that runs entirely on
//! device. The host only flattens the query batch into a `std430` storage
//! buffer and short-circuits an empty batch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Correctness model
//!
//! Each distance is a *continuous* quantity threaded through `sqrt` and
//! products, so the `CPU` and `GPU` are not bit-exact: a device `sqrt` may land
//! a few units in the last place from the scalar reference. The parity test
//! compares with an absolute-or-relative tolerance (`abs <= 1e-4 || rel <=
//! 1e-3`, relative floor `1e-6`). All three fields are continuous everywhere
//! (there is no discrete classification), so the only genuine degeneracies are
//! the zero-length `segment_3d` (`a == b`, where the projection division is
//! guarded and both sides pin the parameter at the start) and the capsule axis
//! and box-frame bar ties, which affect the gradient but not the value. The
//! fixtures and the randomized sweep nonetheless keep the segment endpoints a
//! clear margin apart so the guarded branch never sits on its threshold.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt`, `+ - * /` and ordered comparisons — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and
//! no `u64` / `u16` / `i64` / `f64`. No optional device feature is required, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic, so the kernel
//! provably terminates.
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

/// Euclidean length of a 3D vector; one `sqrt`, which is a core arithmetic
/// primitive rather than a transcendental.
fn length3(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt()
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::box_frame`, reproduced without importing the
/// golden so the twin stays self-contained.
///
/// Folds the point into the positive octant, offsets it by the frame
/// `thickness`, and measures the three axis-aligned bar families with the
/// interior / exterior split, combining them with a minimum.
#[must_use]
pub fn box_frame_sdf(point: [f32; 3], half_extent: [f32; 3], thickness: f32) -> f32 {
    let p = [
        point[0].abs() - half_extent[0],
        point[1].abs() - half_extent[1],
        point[2].abs() - half_extent[2],
    ];
    let q = [
        (p[0] + thickness).abs() - thickness,
        (p[1] + thickness).abs() - thickness,
        (p[2] + thickness).abs() - thickness,
    ];
    let bar_x =
        length3(p[0].max(0.0), q[1].max(0.0), q[2].max(0.0)) + p[0].max(q[1].max(q[2])).min(0.0);
    let bar_y =
        length3(q[0].max(0.0), p[1].max(0.0), q[2].max(0.0)) + q[0].max(p[1].max(q[2])).min(0.0);
    let bar_z =
        length3(q[0].max(0.0), q[1].max(0.0), p[2].max(0.0)) + q[0].max(q[1].max(p[2])).min(0.0);
    bar_x.min(bar_y).min(bar_z)
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::vertical_capsule`, reproduced without importing
/// the golden.
///
/// Clamps the query height into `[0, height]` to snap onto the nearest axis
/// point, then returns the distance to that point minus the `radius`.
#[must_use]
pub fn vertical_capsule_sdf(point: [f32; 3], height: f32, radius: f32) -> f32 {
    let qy = point[1] - point[1].clamp(0.0, height);
    length3(point[0], qy, point[2]) - radius
}

/// Host-side independent reimplementation of the golden
/// `ray_scene::sdf_primitives::segment_3d`, reproduced without importing the
/// golden.
///
/// Projects the point onto the finite segment `a`-`b` with the projection
/// parameter clamped to `[0, 1]`; a degenerate zero-length segment pins the
/// parameter at the start point and the division is guarded.
#[must_use]
pub fn segment_3d_sdf(point: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let pa = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
    let ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ba_len_sq = ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2];
    let h = if ba_len_sq > f32::MIN_POSITIVE {
        ((pa[0] * ba[0] + pa[1] * ba[1] + pa[2] * ba[2]) / ba_len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    length3(pa[0] - ba[0] * h, pa[1] - ba[1] * h, pa[2] - ba[2] * h)
}

/// The portable core-`WGSL` 3D-primitive `SDF` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the three golden evaluators; see the module documentation.
const SDF_PRIMITIVES_3D_WGSL: &str = r#"
// 3D primitive SDF twin: one thread evaluates the box-frame, vertical-capsule
// and segment fields for one query, mirroring the golden {box_frame,
// vertical_capsule, segment_3d} with only clamps, folds, products and one sqrt.
// Transcendental-free.
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point.
    px: f32,
    py: f32,
    pz: f32,
    // Box-frame half-extent and bar thickness.
    fr_hx: f32,
    fr_hy: f32,
    fr_hz: f32,
    fr_thickness: f32,
    // Vertical-capsule height and radius.
    cap_height: f32,
    cap_radius: f32,
    // Segment endpoints a and b.
    seg_ax: f32,
    seg_ay: f32,
    seg_az: f32,
    seg_bx: f32,
    seg_by: f32,
    seg_bz: f32,
}

struct SdfResult {
    dist_box_frame: f32,
    dist_capsule: f32,
    dist_segment: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<SdfResult>;

// Smallest positive normal f32, mirroring the host guard `f32::MIN_POSITIVE`
// against a degenerate (zero-length) segment projection division.
const SEGMENT_EPS: f32 = 1.17549435e-38;

fn length3(x: f32, y: f32, z: f32) -> f32 {
    return sqrt(x * x + y * y + z * z);
}

fn box_frame_sdf(
    px: f32, py: f32, pz: f32,
    hx: f32, hy: f32, hz: f32,
    thickness: f32,
) -> f32 {
    let p0 = abs(px) - hx;
    let p1 = abs(py) - hy;
    let p2 = abs(pz) - hz;
    let q0 = abs(p0 + thickness) - thickness;
    let q1 = abs(p1 + thickness) - thickness;
    let q2 = abs(p2 + thickness) - thickness;
    let bar_x = length3(max(p0, 0.0), max(q1, 0.0), max(q2, 0.0))
        + min(max(p0, max(q1, q2)), 0.0);
    let bar_y = length3(max(q0, 0.0), max(p1, 0.0), max(q2, 0.0))
        + min(max(q0, max(p1, q2)), 0.0);
    let bar_z = length3(max(q0, 0.0), max(q1, 0.0), max(p2, 0.0))
        + min(max(q0, max(q1, p2)), 0.0);
    return min(min(bar_x, bar_y), bar_z);
}

fn vertical_capsule_sdf(px: f32, py: f32, pz: f32, height: f32, radius: f32) -> f32 {
    let qy = py - clamp(py, 0.0, height);
    return length3(px, qy, pz) - radius;
}

fn segment_sdf(
    px: f32, py: f32, pz: f32,
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
) -> f32 {
    let pax = px - ax;
    let pay = py - ay;
    let paz = pz - az;
    let bax = bx - ax;
    let bay = by - ay;
    let baz = bz - az;
    let ba_len_sq = bax * bax + bay * bay + baz * baz;
    var h = 0.0;
    if (ba_len_sq > SEGMENT_EPS) {
        h = clamp((pax * bax + pay * bay + paz * baz) / ba_len_sq, 0.0, 1.0);
    }
    return length3(pax - bax * h, pay - bay * h, paz - baz * h);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: SdfResult;
    out.dist_box_frame = box_frame_sdf(
        q.px, q.py, q.pz,
        q.fr_hx, q.fr_hy, q.fr_hz,
        q.fr_thickness,
    );
    out.dist_capsule = vertical_capsule_sdf(q.px, q.py, q.pz, q.cap_height, q.cap_radius);
    out.dist_segment = segment_sdf(
        q.px, q.py, q.pz,
        q.seg_ax, q.seg_ay, q.seg_az,
        q.seg_bx, q.seg_by, q.seg_bz,
    );
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`SDF_PRIMITIVES_3D_WGSL`].
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

/// `repr(C)` `std430` layout of one query: the point and the per-shape
/// parameters, matching the `WGSL` `Query` struct (fifteen `f32`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Box-frame half-extent `x`.
    fr_hx: f32,
    /// Box-frame half-extent `y`.
    fr_hy: f32,
    /// Box-frame half-extent `z`.
    fr_hz: f32,
    /// Box-frame bar thickness.
    fr_thickness: f32,
    /// Vertical-capsule height along `+y`.
    cap_height: f32,
    /// Vertical-capsule radius.
    cap_radius: f32,
    /// Segment endpoint `a` `x`.
    seg_ax: f32,
    /// Segment endpoint `a` `y`.
    seg_ay: f32,
    /// Segment endpoint `a` `z`.
    seg_az: f32,
    /// Segment endpoint `b` `x`.
    seg_bx: f32,
    /// Segment endpoint `b` `y`.
    seg_by: f32,
    /// Segment endpoint `b` `z`.
    seg_bz: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `SdfResult`
/// struct: the three distances.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the box frame.
    dist_box_frame: f32,
    /// Signed distance to the vertical capsule.
    dist_capsule: f32,
    /// Unsigned distance to the 3D segment.
    dist_segment: f32,
}

/// One query: the point plus the per-shape parameters for the three fields.
///
/// The `fr_*` quadruple carries the box frame's half-extent and thickness, the
/// `cap_*` pair the vertical capsule, and the `seg_*` sextuple the segment
/// endpoints. The host enqueues one query per evaluation, and an empty batch is
/// short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfPrimitives3dQuery {
    /// Query point `x`.
    pub px: f32,
    /// Query point `y`.
    pub py: f32,
    /// Query point `z`.
    pub pz: f32,
    /// Box-frame half-extent `x`.
    pub fr_hx: f32,
    /// Box-frame half-extent `y`.
    pub fr_hy: f32,
    /// Box-frame half-extent `z`.
    pub fr_hz: f32,
    /// Box-frame bar thickness.
    pub fr_thickness: f32,
    /// Vertical-capsule height along `+y`.
    pub cap_height: f32,
    /// Vertical-capsule radius.
    pub cap_radius: f32,
    /// Segment endpoint `a` `x`.
    pub seg_ax: f32,
    /// Segment endpoint `a` `y`.
    pub seg_ay: f32,
    /// Segment endpoint `a` `z`.
    pub seg_az: f32,
    /// Segment endpoint `b` `x`.
    pub seg_bx: f32,
    /// Segment endpoint `b` `y`.
    pub seg_by: f32,
    /// Segment endpoint `b` `z`.
    pub seg_bz: f32,
}

impl SdfPrimitives3dQuery {
    /// Builds a query from the point and the per-shape parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query packs the point and three shapes' parameters as flat scalars for a std430 slot"
    )]
    pub const fn new(
        px: f32,
        py: f32,
        pz: f32,
        fr_hx: f32,
        fr_hy: f32,
        fr_hz: f32,
        fr_thickness: f32,
        cap_height: f32,
        cap_radius: f32,
        seg_ax: f32,
        seg_ay: f32,
        seg_az: f32,
        seg_bx: f32,
        seg_by: f32,
        seg_bz: f32,
    ) -> SdfPrimitives3dQuery {
        SdfPrimitives3dQuery {
            px,
            py,
            pz,
            fr_hx,
            fr_hy,
            fr_hz,
            fr_thickness,
            cap_height,
            cap_radius,
            seg_ax,
            seg_ay,
            seg_az,
            seg_bx,
            seg_by,
            seg_bz,
        }
    }
}

/// One resolved query: the three distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfPrimitives3dResult {
    /// Signed distance to the box frame.
    pub dist_box_frame: f32,
    /// Signed distance to the vertical capsule.
    pub dist_capsule: f32,
    /// Unsigned distance to the 3D segment.
    pub dist_segment: f32,
}

/// Encodes one [`SdfPrimitives3dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfPrimitives3dQuery) -> GpuQuery {
    GpuQuery {
        px: q.px,
        py: q.py,
        pz: q.pz,
        fr_hx: q.fr_hx,
        fr_hy: q.fr_hy,
        fr_hz: q.fr_hz,
        fr_thickness: q.fr_thickness,
        cap_height: q.cap_height,
        cap_radius: q.cap_radius,
        seg_ax: q.seg_ax,
        seg_ay: q.seg_ay,
        seg_az: q.seg_az,
        seg_bx: q.seg_bx,
        seg_by: q.seg_by,
        seg_bz: q.seg_bz,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfPrimitives3dResult`].
fn decode_result(raw: &GpuResult) -> SdfPrimitives3dResult {
    SdfPrimitives3dResult {
        dist_box_frame: raw.dist_box_frame,
        dist_capsule: raw.dist_capsule,
        dist_segment: raw.dist_segment,
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

/// A compiled, reusable 3D-primitive `SDF` compute pipeline, twinning the
/// golden `ray_scene::sdf_primitives` evaluators `box_frame`,
/// `vertical_capsule` and `segment_3d`.
pub struct GpuSdfPrimitives3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfPrimitives3d {
    /// Compiles the 3D-primitive `SDF` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfPrimitives3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_primitives_3d"),
            source: ShaderSource::Wgsl(SDF_PRIMITIVES_3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_primitives_3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_primitives_3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_primitives_3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfPrimitives3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`SdfPrimitives3dResult`] per input, in order.
    ///
    /// Each distance equals the reference within floating-point tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfPrimitives3dQuery],
    ) -> Vec<SdfPrimitives3dResult> {
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
            label: Some("prism_volumetric_sdf_primitives_3d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_primitives_3d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_primitives_3d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_primitives_3d_bind_group"),
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
            label: Some("prism_volumetric_sdf_primitives_3d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_primitives_3d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_primitives_3d_pass"),
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
