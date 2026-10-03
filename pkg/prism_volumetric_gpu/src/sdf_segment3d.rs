//! `wgpu` compute twin of the four analytic axis/segment signed-distance
//! primitives of the `CPU` golden path `line_sdf`, `infinite_cylinder`,
//! `infinite_cone` and `cylinder_segment` in
//! `prism_render_architecture::ray_scene::sdf_primitives`.
//!
//! Implicit modelling composes complex solids from *analytic* primitives whose
//! exact signed distance is known in closed form rather than sampled on a grid.
//! The reference derives four line/segment solids: an infinite `line_sdf`
//! (unsigned perpendicular distance to a line through the origin), the
//! Inigo-Quilez `infinite_cylinder` (a `y`-parallel axis offset in the `xz`
//! plane), the Inigo-Quilez `infinite_cone` (an unbounded cone whose apex sits
//! at the origin, aperture pre-baked as `[sin, cos]`), and the Inigo-Quilez
//! `cylinder_segment` (a finite capped cylinder between arbitrary endpoints).
//! [`GpuSdfSegment3d`] is the on-device twin: each thread reads one point plus
//! all four shapes' parameters and writes all four signed distances,
//! reproducing the reference closed forms with only `sqrt`, `abs`, `min`,
//! `max`, `clamp`, `select`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfSegment3dQuery`] — a query `point` plus the
//! `line_sdf` `line_direction`, the `infinite_cylinder` (`cyl_axis_xz`,
//! `cyl_radius`), the `infinite_cone` `cone_sin_cos` and the
//! `cylinder_segment` (`seg_a`, `seg_b`, `seg_radius`) parameters — and writes
//! one [`SdfSegment3dResult`] holding the four signed distances `line_sd`,
//! `infinite_cylinder_sd`, `infinite_cone_sd` and `cylinder_segment_sd`.
//!
//! The `line_sdf` kernel subtracts the point's projection onto the direction
//! and returns the length of the perpendicular residual. The
//! `infinite_cylinder` kernel returns the `xz`-plane radial offset minus the
//! radius. The `infinite_cone` kernel reduces the point to the meridian pair
//! `(radial, y)`, projects it onto the flank ray clamped at the apex, takes the
//! residual length and flips its sign on the axis side of the flank. The
//! `cylinder_segment` kernel forms the radial and axial residuals in
//! `baba`-scaled units, negates the nearer squared residual inside and sums the
//! positive squared residuals outside, then rescales by a single final `sqrt`
//! and divide.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms into complex shapes,
//! the ray-marcher that evaluates them along a ray, and the surface-normal
//! estimation all stay on the host; the device sees only the four stateless,
//! fixed-width signed-distance evaluations, one query at a time, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! All four distances thread through `sqrt`, products and quotients, so the
//! `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few
//! units in the last place from the scalar reference. The parity test asserts
//! each distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, tight enough
//! to catch a genuinely wrong port yet loose enough to admit a legal last-place
//! difference. Fixtures stay clear of the `infinite_cone` sign-flip boundary
//! and of the `cylinder_segment` interior/exterior transition, where the `CPU`
//! and `GPU` could pick different sides of a branch.
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

/// The portable core-`WGSL` axis/segment signed-distance kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `line_sdf`, `infinite_cylinder`,
/// `infinite_cone` and `cylinder_segment` closed forms; see the module
/// documentation for the algorithm.
const SDF_SEGMENT3D_WGSL: &str = r#"
// Axis/segment signed-distance twin: one thread computes one query point's
// line, infinite-cylinder, infinite-cone and cylinder-segment distances,
// mirroring the CPU golden `ray_scene::sdf_primitives::{line_sdf,
// infinite_cylinder, infinite_cone, cylinder_segment}` with only sqrt, abs,
// min, max, clamp, select, products and quotients. The domain/CSG operators
// and the ray-marcher stay on the host.
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
    pz: f32,
    // Line direction (need not be normalised; must be non-zero).
    line_dx: f32,
    line_dy: f32,
    line_dz: f32,
    // Infinite cylinder: axis point in the xz plane and radius.
    cyl_axis_x: f32,
    cyl_axis_z: f32,
    cyl_radius: f32,
    // Infinite cone: pre-baked aperture [sin, cos].
    cone_sin: f32,
    cone_cos: f32,
    // Cylinder segment endpoint a.
    seg_ax: f32,
    seg_ay: f32,
    seg_az: f32,
    // Cylinder segment endpoint b.
    seg_bx: f32,
    seg_by: f32,
    seg_bz: f32,
    // Cylinder segment radius.
    seg_radius: f32,
    pad0: f32,
    pad1: f32,
}

struct Distances {
    // Infinite-line unsigned distance.
    line_sd: f32,
    // Infinite-cylinder signed distance.
    infinite_cylinder_sd: f32,
    // Infinite-cone signed distance.
    infinite_cone_sd: f32,
    // Cylinder-segment signed distance.
    cylinder_segment_sd: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Distances;

    // --- Infinite line: length of the perpendicular residual. ---
    let dd = q.line_dx * q.line_dx + q.line_dy * q.line_dy + q.line_dz * q.line_dz;
    let pd = q.px * q.line_dx + q.py * q.line_dy + q.pz * q.line_dz;
    let t = pd / dd;
    let lr_x = q.px - q.line_dx * t;
    let lr_y = q.py - q.line_dy * t;
    let lr_z = q.pz - q.line_dz * t;
    out.line_sd = sqrt(lr_x * lr_x + lr_y * lr_y + lr_z * lr_z);

    // --- Infinite cylinder: xz radial offset minus radius. ---
    let cyl_ox = q.px - q.cyl_axis_x;
    let cyl_oz = q.pz - q.cyl_axis_z;
    out.infinite_cylinder_sd = sqrt(cyl_ox * cyl_ox + cyl_oz * cyl_oz) - q.cyl_radius;

    // --- Infinite cone: meridian flank projection, signed on the axis side. ---
    let cone_qx = sqrt(q.px * q.px + q.pz * q.pz);
    let cone_qy = q.py;
    let cone_proj = max(cone_qx * q.cone_sin + cone_qy * q.cone_cos, 0.0);
    let cone_wx = cone_qx - q.cone_sin * cone_proj;
    let cone_wy = cone_qy - q.cone_cos * cone_proj;
    let cone_d = sqrt(cone_wx * cone_wx + cone_wy * cone_wy);
    let cone_inside = (q.cone_cos * cone_qx - q.cone_sin * cone_qy) < 0.0;
    out.infinite_cone_sd = select(cone_d, -cone_d, cone_inside);

    // --- Cylinder segment: baba-scaled radial/axial residuals. ---
    let ba_x = q.seg_bx - q.seg_ax;
    let ba_y = q.seg_by - q.seg_ay;
    let ba_z = q.seg_bz - q.seg_az;
    let pa_x = q.px - q.seg_ax;
    let pa_y = q.py - q.seg_ay;
    let pa_z = q.pz - q.seg_az;
    let baba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z;
    let paba = pa_x * ba_x + pa_y * ba_y + pa_z * ba_z;
    let perp_x = pa_x * baba - ba_x * paba;
    let perp_y = pa_y * baba - ba_y * paba;
    let perp_z = pa_z * baba - ba_z * paba;
    let perp_len = sqrt(perp_x * perp_x + perp_y * perp_y + perp_z * perp_z);
    let seg_x = perp_len - q.seg_radius * baba;
    let seg_y = abs(paba - baba * 0.5) - baba * 0.5;
    let seg_x2 = seg_x * seg_x;
    let seg_y2 = seg_y * seg_y * baba;
    let interior = max(seg_x, seg_y) < 0.0;
    let seg_outside = select(0.0, seg_x2, seg_x > 0.0) + select(0.0, seg_y2, seg_y > 0.0);
    let seg_d = select(seg_outside, -min(seg_x2, seg_y2), interior);
    out.cylinder_segment_sd = sign(seg_d) * sqrt(abs(seg_d)) / baba;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_SEGMENT3D_WGSL`].
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
/// the point components plus all four shapes' parameters and two pad words to a
/// `80`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Line direction `x`.
    line_dx: f32,
    /// Line direction `y`.
    line_dy: f32,
    /// Line direction `z`.
    line_dz: f32,
    /// Infinite-cylinder axis `x` in the `xz` plane.
    cyl_axis_x: f32,
    /// Infinite-cylinder axis `z` in the `xz` plane.
    cyl_axis_z: f32,
    /// Infinite-cylinder radius.
    cyl_radius: f32,
    /// Infinite-cone aperture sine.
    cone_sin: f32,
    /// Infinite-cone aperture cosine.
    cone_cos: f32,
    /// Cylinder-segment endpoint `a` `x`.
    seg_ax: f32,
    /// Cylinder-segment endpoint `a` `y`.
    seg_ay: f32,
    /// Cylinder-segment endpoint `a` `z`.
    seg_az: f32,
    /// Cylinder-segment endpoint `b` `x`.
    seg_bx: f32,
    /// Cylinder-segment endpoint `b` `y`.
    seg_by: f32,
    /// Cylinder-segment endpoint `b` `z`.
    seg_bz: f32,
    /// Cylinder-segment radius.
    seg_radius: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the four signed distances at a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Infinite-line unsigned distance.
    line_sd: f32,
    /// Infinite-cylinder signed distance.
    infinite_cylinder_sd: f32,
    /// Infinite-cone signed distance.
    infinite_cone_sd: f32,
    /// Cylinder-segment signed distance.
    cylinder_segment_sd: f32,
}

/// One query for the axis/segment signed-distance twin: the query `point` plus
/// the `line_sdf`, `infinite_cylinder`, `infinite_cone` and `cylinder_segment`
/// shape parameters.
///
/// `point` is the evaluation position; `line_direction` is the (non-zero)
/// `line_sdf` direction; `cyl_axis_xz`/`cyl_radius` place the
/// `infinite_cylinder`; `cone_sin_cos` is the `infinite_cone` aperture as
/// `[sin, cos]`; `seg_a`/`seg_b`/`seg_radius` are the `cylinder_segment`
/// endpoints and radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSegment3dQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Infinite-line direction `[x, y, z]` (non-zero; need not be normalised).
    pub line_direction: [f32; 3],
    /// Infinite-cylinder axis point `[x, z]` in the `xz` plane.
    pub cyl_axis_xz: [f32; 2],
    /// Infinite-cylinder radius.
    pub cyl_radius: f32,
    /// Infinite-cone aperture `[sin, cos]`.
    pub cone_sin_cos: [f32; 2],
    /// Cylinder-segment endpoint `a` `[x, y, z]`.
    pub seg_a: [f32; 3],
    /// Cylinder-segment endpoint `b` `[x, y, z]` (distinct from `seg_a`).
    pub seg_b: [f32; 3],
    /// Cylinder-segment radius.
    pub seg_radius: f32,
}

impl SdfSegment3dQuery {
    /// Builds a query from the point and all four shapes' parameters.
    #[must_use]
    pub const fn new(
        point: [f32; 3],
        line_direction: [f32; 3],
        cyl_axis_xz: [f32; 2],
        cyl_radius: f32,
        cone_sin_cos: [f32; 2],
        seg_a: [f32; 3],
        seg_b: [f32; 3],
        seg_radius: f32,
    ) -> SdfSegment3dQuery {
        SdfSegment3dQuery {
            point,
            line_direction,
            cyl_axis_xz,
            cyl_radius,
            cone_sin_cos,
            seg_a,
            seg_b,
            seg_radius,
        }
    }
}

/// One resolved query of the axis/segment signed-distance twin: the `line_sdf`,
/// `infinite_cylinder`, `infinite_cone` and `cylinder_segment` distances at the
/// query point.
///
/// `line_sd` is an unsigned distance (a line has no interior); the other three
/// are negative inside the solid, positive outside, zero on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSegment3dResult {
    /// Infinite-line unsigned distance.
    pub line_sd: f32,
    /// Infinite-cylinder signed distance.
    pub infinite_cylinder_sd: f32,
    /// Infinite-cone signed distance.
    pub infinite_cone_sd: f32,
    /// Cylinder-segment signed distance.
    pub cylinder_segment_sd: f32,
}

/// Encodes one [`SdfSegment3dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfSegment3dQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        line_dx: q.line_direction[0],
        line_dy: q.line_direction[1],
        line_dz: q.line_direction[2],
        cyl_axis_x: q.cyl_axis_xz[0],
        cyl_axis_z: q.cyl_axis_xz[1],
        cyl_radius: q.cyl_radius,
        cone_sin: q.cone_sin_cos[0],
        cone_cos: q.cone_sin_cos[1],
        seg_ax: q.seg_a[0],
        seg_ay: q.seg_a[1],
        seg_az: q.seg_a[2],
        seg_bx: q.seg_b[0],
        seg_by: q.seg_b[1],
        seg_bz: q.seg_b[2],
        seg_radius: q.seg_radius,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfSegment3dResult`].
fn decode_result(raw: &GpuResult) -> SdfSegment3dResult {
    SdfSegment3dResult {
        line_sd: raw.line_sd,
        infinite_cylinder_sd: raw.infinite_cylinder_sd,
        infinite_cone_sd: raw.infinite_cone_sd,
        cylinder_segment_sd: raw.cylinder_segment_sd,
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

/// A compiled, reusable axis/segment signed-distance compute pipeline, twinning
/// the `CPU` golden `line_sdf`, `infinite_cylinder`, `infinite_cone` and
/// `cylinder_segment` of `prism_render_architecture::ray_scene::sdf_primitives`.
pub struct GpuSdfSegment3d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfSegment3d {
    /// Compiles the axis/segment signed-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfSegment3d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_segment3d"),
            source: ShaderSource::Wgsl(SDF_SEGMENT3D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_segment3d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_segment3d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_segment3d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfSegment3d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfSegment3dResult`]
    /// per input, in order.
    ///
    /// The signed distances match the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfSegment3dQuery],
    ) -> Vec<SdfSegment3dResult> {
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
            label: Some("prism_volumetric_sdf_segment3d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_segment3d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_segment3d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_segment3d_bind_group"),
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
            label: Some("prism_volumetric_sdf_segment3d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_segment3d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_segment3d_pass"),
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
