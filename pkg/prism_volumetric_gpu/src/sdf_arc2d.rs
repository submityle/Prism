//! `wgpu` compute twin of four analytic two-dimensional arc/disk
//! signed-distance primitives from the `CPU` golden path
//! (`prism_render_architecture::ray_scene::sdf_primitives`): the Inigo Quilez
//! `pie`, `cut_disk_2d`, `horseshoe_2d` and `tunnel_2d` closed forms.
//!
//! Vector graphics, gauge dials, keyhole masks and procedural profile
//! extrusion evaluate these analytic arc outlines whose exact Euclidean
//! distance is known in closed form rather than sampling a baked field. This
//! module is the on-device twin of four of those atoms, each a true distance
//! built from folds, clamps, branch selects and a `sqrt`:
//!
//! - `pie`: a circular sector (wedge) centred on `+y`, its half-aperture passed
//!   as the baked `(sin, cos)` pair; the field is the arc circle clamped
//!   against the straight edge ray, signed so the wedge interior is negative.
//! - `cut_disk_2d`: a disk of `radius` sliced by the horizontal chord at
//!   `cut_height`; the arc, chord and corner regions each carry their own exact
//!   distance.
//! - `horseshoe_2d`: a `C`-shaped ring arc, its mouth aperture baked as a
//!   `(sin, cos)` pair; the query folds by `|x|`, rotates into the arc frame,
//!   and measures a half-infinite rounded strip.
//! - `tunnel_2d`: a rounded arch (half-stadium) of half-width `half_width` and
//!   `height`, a box half-space combined with the capping semicircle.
//!
//! [`GpuSdfArc2d`] evaluates all four for one query per thread, reproducing the
//! reference closed forms with only `abs`, `min`, `max`, `clamp`, `select`, a
//! sign test and a final `sqrt` — no transcendental — so a passing real-device
//! parity test is direct evidence the ported kernel computes the same distances
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfArc2dQuery`] — the shared query point plus the
//! per-shape parameters (the `pie` and `horseshoe_2d` baked `(sin, cos)`
//! apertures and radii, the `cut_disk_2d` radius and chord height, and the
//! `tunnel_2d` half-width and height) — and writes one [`SdfArc2dResult`]
//! holding the four signed distances. The aperture angles are supplied as
//! `(sin, cos)` inputs, never computed on-device, so the kernel stays free of
//! trigonometry.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these outlines, the `extrude`
//! and `revolution` lift to three dimensions, the angle-to-`(sin, cos)` baking,
//! and any acceleration structure all stay on the host; the device sees only
//! the stateless, fixed-width distance evaluation, one query at a time, so a
//! storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every output threads through products, quotients and a `sqrt`, so the `CPU`
//! and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few units
//! in the last place from the scalar reference. The parity test asserts each
//! distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` with a relative
//! floor of `1e-6`. Where a shape's internal `signum` flips sign the governing
//! region boundary is a locus where the two branches agree, so the field stays
//! continuous; the randomized sweep still rejects samples near each branch,
//! fold or sign boundary to keep the comparison far from any such edge.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `select`, `sqrt`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round` and no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs
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

/// The portable core-`WGSL` kernel for the four two-dimensional arc/disk
/// signed-distance outlines, embedded inline so the twin ships as a single
/// source file. The single entry point `solve` runs one thread per query.
const SDF_ARC2D_WGSL: &str = r#"
// 2D arc/disk signed-distance twin: one thread computes one query's pie,
// cut-disk, horseshoe and tunnel distances, mirroring the CPU golden
// ray_scene::sdf_primitives::{pie, cut_disk_2d, horseshoe_2d, tunnel_2d} with
// only abs, min, max, clamp, select and a final sqrt. The CSG/extrude operators
// and the angle-to-(sin,cos) baking stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::sdf_primitives；无
// 第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Shared query point.
    px: f32,
    py: f32,
    // Pie: baked half-aperture (sin, cos) and sector radius.
    pie_sin: f32,
    pie_cos: f32,
    pie_radius: f32,
    // Cut disk: radius and horizontal chord height.
    cut_radius: f32,
    cut_height: f32,
    // Horseshoe: baked mouth-aperture (sin, cos), ring radius, prong length
    // and ring half-thickness.
    hs_sin: f32,
    hs_cos: f32,
    hs_radius: f32,
    hs_arm: f32,
    hs_thickness: f32,
    // Tunnel: half-width and arch height.
    tunnel_half_width: f32,
    tunnel_height: f32,
    pad0: f32,
    pad1: f32,
}

struct Distances {
    // Signed distance to the circular sector (pie wedge).
    pie: f32,
    // Signed distance to the chord-cut disk.
    cut_disk: f32,
    // Signed distance to the horseshoe ring arc.
    horseshoe: f32,
    // Signed distance to the rounded tunnel arch.
    tunnel: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Euclidean length of a 2-vector.
fn length2(v: vec2<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y);
}

// Squared length of a 2-vector (no sqrt), matching the reference dot2_2.
fn dot2_2(v: vec2<f32>) -> f32 {
    return v.x * v.x + v.y * v.y;
}

// Rust `f32::signum` reimplemented for the branch sign: +1 for a non-negative
// argument, -1 otherwise. Where each shape uses this the governing region
// boundary is a locus where the branches agree, so the choice at an exact zero
// never produces a cliff.
fn sign_pos(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

// Exact signed distance to a circular sector (pie wedge) centred on +y with
// half-aperture (sin, cos).
fn pie(point: vec2<f32>, sc: vec2<f32>, radius: f32) -> f32 {
    let px = abs(point.x);
    let p = vec2<f32>(px, point.y);
    let l = length2(p) - radius;
    let t = clamp(p.x * sc.x + p.y * sc.y, 0.0, radius);
    let m = length2(vec2<f32>(p.x - sc.x * t, p.y - sc.y * t));
    let edge_sign = sign_pos(sc.y * p.x - sc.x * p.y);
    return max(l, m * edge_sign);
}

// Exact signed distance to a disk of `radius` cut by the horizontal chord at
// `cut_height`.
fn cut_disk_2d(point: vec2<f32>, radius: f32, cut_height: f32) -> f32 {
    let r = radius;
    let h = -cut_height;
    let w = sqrt(max(r * r - h * h, 0.0));
    let p = vec2<f32>(abs(point.x), -point.y);
    let s = max((h - r) * p.x * p.x + w * w * (h + r - 2.0 * p.y), h * p.x - w * p.y);
    if (s < 0.0) {
        return length2(p) - r;
    } else if (p.x < w) {
        return h - p.y;
    }
    return length2(vec2<f32>(p.x - w, p.y - h));
}

// Exact signed distance to a horseshoe ring arc with baked mouth aperture
// (sin, cos).
fn horseshoe_2d(point: vec2<f32>, sc: vec2<f32>, radius: f32, arm: f32, thickness: f32) -> f32 {
    let c_cos = sc.y;
    let c_sin = sc.x;
    let px = abs(point.x);
    let l = length2(vec2<f32>(px, point.y));
    let rx = -c_cos * px + c_sin * point.y;
    let ry = c_sin * px + c_cos * point.y;
    var qx: f32;
    if (ry > 0.0 || rx > 0.0) {
        qx = rx;
    } else {
        qx = l * sign_pos(-c_cos);
    }
    var qy: f32;
    if (rx > 0.0) {
        qy = ry;
    } else {
        qy = l;
    }
    let bx = qx - arm;
    let by = abs(qy - radius) - thickness;
    return length2(vec2<f32>(max(bx, 0.0), max(by, 0.0))) + min(max(bx, by), 0.0);
}

// Exact signed distance to a rounded tunnel arch (half-stadium) of half-width
// and height.
fn tunnel_2d(point: vec2<f32>, half_width: f32, height: f32) -> f32 {
    let px = abs(point.x);
    let py = -point.y;
    var qx = px - half_width;
    let qy = py - height;
    let d1 = dot2_2(vec2<f32>(max(qx, 0.0), qy));
    if (py <= 0.0) {
        qx = length2(vec2<f32>(px, py)) - half_width;
    }
    let d2 = dot2_2(vec2<f32>(qx, max(qy, 0.0)));
    let d = sqrt(min(d1, d2));
    return select(d, -d, max(qx, qy) < 0.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let point = vec2<f32>(q.px, q.py);

    var out: Distances;
    out.pie = pie(point, vec2<f32>(q.pie_sin, q.pie_cos), q.pie_radius);
    out.cut_disk = cut_disk_2d(point, q.cut_radius, q.cut_height);
    out.horseshoe = horseshoe_2d(
        point,
        vec2<f32>(q.hs_sin, q.hs_cos),
        q.hs_radius,
        q.hs_arm,
        q.hs_thickness,
    );
    out.tunnel = tunnel_2d(point, q.tunnel_half_width, q.tunnel_height);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_ARC2D_WGSL`].
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
/// the shared query point plus the per-shape parameters flattened to scalar
/// `f32` lanes, with two pad words to a `64`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Pie half-aperture sine.
    pie_sin: f32,
    /// Pie half-aperture cosine.
    pie_cos: f32,
    /// Pie sector radius.
    pie_radius: f32,
    /// Cut-disk radius.
    cut_radius: f32,
    /// Cut-disk chord height.
    cut_height: f32,
    /// Horseshoe mouth-aperture sine.
    hs_sin: f32,
    /// Horseshoe mouth-aperture cosine.
    hs_cos: f32,
    /// Horseshoe ring radius.
    hs_radius: f32,
    /// Horseshoe prong length.
    hs_arm: f32,
    /// Horseshoe ring half-thickness.
    hs_thickness: f32,
    /// Tunnel half-width.
    tunnel_half_width: f32,
    /// Tunnel arch height.
    tunnel_height: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the four signed distances to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed distance to the pie wedge.
    pie: f32,
    /// Signed distance to the chord-cut disk.
    cut_disk: f32,
    /// Signed distance to the horseshoe ring arc.
    horseshoe: f32,
    /// Signed distance to the rounded tunnel arch.
    tunnel: f32,
}

/// One query for the two-dimensional arc/disk distance twin: the shared query
/// point and the per-shape parameters.
///
/// `point` is the position whose distances are sought. `pie_sin`/`pie_cos` are
/// the pie wedge's baked half-aperture `(sin, cos)` and `pie_radius` its radius.
/// `cut_radius` and `cut_height` are the cut disk's radius and chord height
/// (`-cut_radius < cut_height < cut_radius`). `hs_sin`/`hs_cos` are the
/// horseshoe's baked mouth-aperture `(sin, cos)` (keep `hs_cos > 0`),
/// `hs_radius` its ring radius, `hs_arm` its prong length and `hs_thickness`
/// its ring half-thickness. `tunnel_half_width` and `tunnel_height` are the
/// rounded arch's half-width and height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfArc2dQuery {
    /// Shared query point.
    pub point: [f32; 2],
    /// Pie half-aperture sine.
    pub pie_sin: f32,
    /// Pie half-aperture cosine.
    pub pie_cos: f32,
    /// Pie sector radius.
    pub pie_radius: f32,
    /// Cut-disk radius.
    pub cut_radius: f32,
    /// Cut-disk chord height.
    pub cut_height: f32,
    /// Horseshoe mouth-aperture sine.
    pub hs_sin: f32,
    /// Horseshoe mouth-aperture cosine.
    pub hs_cos: f32,
    /// Horseshoe ring radius.
    pub hs_radius: f32,
    /// Horseshoe prong length.
    pub hs_arm: f32,
    /// Horseshoe ring half-thickness.
    pub hs_thickness: f32,
    /// Tunnel half-width.
    pub tunnel_half_width: f32,
    /// Tunnel arch height.
    pub tunnel_height: f32,
}

impl SdfArc2dQuery {
    /// Builds a query from the shared point and the per-shape parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query flattens four shapes' parameters into one record"
    )]
    pub const fn new(
        point: [f32; 2],
        pie_sin: f32,
        pie_cos: f32,
        pie_radius: f32,
        cut_radius: f32,
        cut_height: f32,
        hs_sin: f32,
        hs_cos: f32,
        hs_radius: f32,
        hs_arm: f32,
        hs_thickness: f32,
        tunnel_half_width: f32,
        tunnel_height: f32,
    ) -> SdfArc2dQuery {
        SdfArc2dQuery {
            point,
            pie_sin,
            pie_cos,
            pie_radius,
            cut_radius,
            cut_height,
            hs_sin,
            hs_cos,
            hs_radius,
            hs_arm,
            hs_thickness,
            tunnel_half_width,
            tunnel_height,
        }
    }
}

/// One resolved query of the two-dimensional arc/disk distance twin: the four
/// signed distances.
///
/// `pie` is `prism_render_architecture::ray_scene::sdf_primitives::pie`,
/// `cut_disk` is `cut_disk_2d`, `horseshoe` is `horseshoe_2d` and `tunnel` is
/// `tunnel_2d`; each is negative inside the respective shape and positive
/// outside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfArc2dResult {
    /// Signed distance to the pie wedge.
    pub pie: f32,
    /// Signed distance to the chord-cut disk.
    pub cut_disk: f32,
    /// Signed distance to the horseshoe ring arc.
    pub horseshoe: f32,
    /// Signed distance to the rounded tunnel arch.
    pub tunnel: f32,
}

/// Encodes one [`SdfArc2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfArc2dQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pie_sin: q.pie_sin,
        pie_cos: q.pie_cos,
        pie_radius: q.pie_radius,
        cut_radius: q.cut_radius,
        cut_height: q.cut_height,
        hs_sin: q.hs_sin,
        hs_cos: q.hs_cos,
        hs_radius: q.hs_radius,
        hs_arm: q.hs_arm,
        hs_thickness: q.hs_thickness,
        tunnel_half_width: q.tunnel_half_width,
        tunnel_height: q.tunnel_height,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfArc2dResult`].
fn decode_result(raw: &GpuResult) -> SdfArc2dResult {
    SdfArc2dResult {
        pie: raw.pie,
        cut_disk: raw.cut_disk,
        horseshoe: raw.horseshoe,
        tunnel: raw.tunnel,
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

/// A compiled, reusable two-dimensional arc/disk distance compute pipeline,
/// twinning the `CPU` golden
/// `prism_render_architecture::ray_scene::sdf_primitives` outlines `pie`,
/// `cut_disk_2d`, `horseshoe_2d` and `tunnel_2d`.
pub struct GpuSdfArc2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfArc2d {
    /// Compiles the two-dimensional arc/disk distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfArc2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_arc2d"),
            source: ShaderSource::Wgsl(SDF_ARC2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_arc2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_arc2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_arc2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfArc2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfArc2dResult`] per
    /// input, in order.
    ///
    /// The distances match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[SdfArc2dQuery]) -> Vec<SdfArc2dResult> {
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
            label: Some("prism_volumetric_sdf_arc2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_arc2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_arc2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_arc2d_bind_group"),
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
            label: Some("prism_volumetric_sdf_arc2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_arc2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_arc2d_pass"),
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
