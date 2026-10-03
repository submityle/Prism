//! `wgpu` compute twin of the three planar line/stadium/ring signed-distance
//! primitives of the `CPU` golden path
//! ([`segment_2d`](prism_render_architecture::ray_scene::sdf_primitives::segment_2d),
//! [`capsule_2d`](prism_render_architecture::ray_scene::sdf_primitives::capsule_2d)
//! and
//! [`annulus_2d`](prism_render_architecture::ray_scene::sdf_primitives::annulus_2d)).
//!
//! Two-dimensional implicit modelling needs *analytic* primitives whose exact
//! signed distance is known in closed form rather than sampled on a grid. The
//! reference derives three such distances in the plane: the unsigned distance
//! to a line segment
//! ([`segment_2d`](prism_render_architecture::ray_scene::sdf_primitives::segment_2d)),
//! the stadium that inflates that segment by a radius
//! ([`capsule_2d`](prism_render_architecture::ray_scene::sdf_primitives::capsule_2d)),
//! and the ring (annulus) that thickens a circle into a band
//! ([`annulus_2d`](prism_render_architecture::ray_scene::sdf_primitives::annulus_2d)).
//! [`GpuSdfCapsule2d`] is the on-device twin: each thread reads one point plus
//! all three shapes' parameters and writes all three signed distances,
//! reproducing the reference closed forms with only `sqrt`, `abs`, `min`,
//! `max`, `clamp`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfCapsule2dQuery`] — a query `point` plus the
//! capsule endpoints `capsule_a`/`capsule_b` and `capsule_radius`, the segment
//! endpoints `segment_a`/`segment_b`, and the annulus `annulus_radius`/
//! `annulus_half_width` — and writes one [`SdfCapsule2dResult`] holding the
//! three signed distances `capsule_2d_sd`, `segment_2d_sd` and `annulus_2d_sd`.
//!
//! The segment kernel projects the query onto the line through the endpoints,
//! clamps the projection parameter `h` to `[0, 1]` so the nearest point stays
//! on the finite segment, and returns the length of the residual vector. The
//! capsule kernel is that segment distance thinned by the stadium `radius`,
//! turning the zero level set into two end discs joined by a slab. The annulus
//! kernel takes the point's radius, folds it about the circle of radius
//! `annulus_radius` with an absolute value, and thins the result by the band
//! half-width, yielding the signed distance to a ring.
//!
//! # What stays on the host
//!
//! The planar domain and `CSG` operators that compose these atoms into complex
//! outlines, the extrusion that lifts them to 3D, and the surface-normal
//! estimation all stay on the host; the device sees only the three stateless,
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
//! enough to admit a legal last-place difference. The projection parameter
//! `clamp` to `[0, 1]` and the annulus absolute value are continuous, so a
//! near-crease query only moves the result by a last-place amount; the only
//! genuine hazard is a degenerate segment whose endpoints coincide, dividing by
//! a zero squared length. Fixtures and the randomized sweep keep every segment
//! a safe margin longer than zero, so that division is always well formed and
//! the `CPU` and `GPU` agree.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`,
//! `min`, `max`, `clamp`, `+ - * /` and unsigned index arithmetic — with no
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

/// The inline `WGSL` source of the planar line/stadium/ring signed-distance
/// twin. The crate ships the kernel as a single source file. The single entry
/// point `solve` mirrors the `CPU` golden
/// [`segment_2d`](prism_render_architecture::ray_scene::sdf_primitives::segment_2d),
/// [`capsule_2d`](prism_render_architecture::ray_scene::sdf_primitives::capsule_2d)
/// and
/// [`annulus_2d`](prism_render_architecture::ray_scene::sdf_primitives::annulus_2d)
/// closed forms; see the module documentation for the algorithm.
const SDF_CAPSULE2D_WGSL: &str = r#"
// Planar line/stadium/ring signed-distance twin: one thread computes one query
// point's segment, capsule (stadium) and annulus (ring) signed distances, using
// only sqrt, abs, min, max, clamp, products and quotients. The planar CSG
// operators and the extrusion stay on the host.
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
    // Capsule (stadium) segment endpoints and inflation radius.
    cap_ax: f32,
    cap_ay: f32,
    cap_bx: f32,
    cap_by: f32,
    cap_r: f32,
    // Standalone segment endpoints.
    seg_ax: f32,
    seg_ay: f32,
    seg_bx: f32,
    seg_by: f32,
    // Annulus (ring) circle radius and band half-width.
    ann_r: f32,
    ann_hw: f32,
}

struct Distances {
    // Capsule (stadium) signed distance.
    capsule_2d_sd: f32,
    // Segment unsigned distance.
    segment_2d_sd: f32,
    // Annulus (ring) signed distance.
    annulus_2d_sd: f32,
    // Padding word to a 16-byte stride.
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Euclidean length of a 2-vector; the shared planar reduction.
fn length2(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

// Unsigned distance to the finite segment a->b: project the query onto the
// line, clamp the parameter to [0, 1], then measure the residual. Endpoints
// kept a safe margin apart so the squared-length divisor is never zero.
fn segment_2d(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let pax = px - ax;
    let pay = py - ay;
    let bax = bx - ax;
    let bay = by - ay;
    let h = clamp((pax * bax + pay * bay) / (bax * bax + bay * bay), 0.0, 1.0);
    return length2(pax - bax * h, pay - bay * h);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Distances;

    // Segment: the unsigned distance to the finite line segment.
    out.segment_2d_sd = segment_2d(q.px, q.py, q.seg_ax, q.seg_ay, q.seg_bx, q.seg_by);

    // Capsule (stadium) = segment distance to the capsule endpoints minus the
    // inflation radius.
    out.capsule_2d_sd =
        segment_2d(q.px, q.py, q.cap_ax, q.cap_ay, q.cap_bx, q.cap_by) - q.cap_r;

    // Annulus (ring) = |distance-from-origin - circle radius| - band half-width.
    out.annulus_2d_sd = abs(length2(q.px, q.py) - q.ann_r) - q.ann_hw;

    out.pad0 = 0.0;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_CAPSULE2D_WGSL`].
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
/// the point components plus all three shapes' parameters. Thirteen `f32`
/// fields pack to a `52`-byte, `4`-byte-aligned stride with no trailing pad.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Capsule endpoint `a.x`.
    cap_ax: f32,
    /// Capsule endpoint `a.y`.
    cap_ay: f32,
    /// Capsule endpoint `b.x`.
    cap_bx: f32,
    /// Capsule endpoint `b.y`.
    cap_by: f32,
    /// Capsule inflation radius.
    cap_r: f32,
    /// Segment endpoint `a.x`.
    seg_ax: f32,
    /// Segment endpoint `a.y`.
    seg_ay: f32,
    /// Segment endpoint `b.x`.
    seg_bx: f32,
    /// Segment endpoint `b.y`.
    seg_by: f32,
    /// Annulus circle radius.
    ann_r: f32,
    /// Annulus band half-width.
    ann_hw: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: three signed distances plus one pad word packing to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Capsule (stadium) signed distance.
    capsule_2d_sd: f32,
    /// Segment unsigned distance.
    segment_2d_sd: f32,
    /// Annulus (ring) signed distance.
    annulus_2d_sd: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the planar line/stadium/ring signed-distance twin: the query
/// `point` plus the capsule, segment and annulus shape parameters.
///
/// `point` is the evaluation position; `capsule_a`/`capsule_b`/`capsule_radius`
/// are the
/// [`capsule_2d`](prism_render_architecture::ray_scene::sdf_primitives::capsule_2d)
/// stadium endpoints and inflation radius; `segment_a`/`segment_b` are the
/// [`segment_2d`](prism_render_architecture::ray_scene::sdf_primitives::segment_2d)
/// endpoints; `annulus_radius`/`annulus_half_width` are the
/// [`annulus_2d`](prism_render_architecture::ray_scene::sdf_primitives::annulus_2d)
/// circle radius and band half-width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCapsule2dQuery {
    /// Query point `[x, y]`.
    pub point: [f32; 2],
    /// Capsule endpoint `a` `[x, y]`.
    pub capsule_a: [f32; 2],
    /// Capsule endpoint `b` `[x, y]`.
    pub capsule_b: [f32; 2],
    /// Capsule inflation radius.
    pub capsule_radius: f32,
    /// Segment endpoint `a` `[x, y]`.
    pub segment_a: [f32; 2],
    /// Segment endpoint `b` `[x, y]`.
    pub segment_b: [f32; 2],
    /// Annulus circle radius.
    pub annulus_radius: f32,
    /// Annulus band half-width.
    pub annulus_half_width: f32,
}

impl SdfCapsule2dQuery {
    /// Builds a query from the point and all three shapes' parameters.
    #[must_use]
    pub const fn new(
        point: [f32; 2],
        capsule_a: [f32; 2],
        capsule_b: [f32; 2],
        capsule_radius: f32,
        segment_a: [f32; 2],
        segment_b: [f32; 2],
        annulus_radius: f32,
        annulus_half_width: f32,
    ) -> SdfCapsule2dQuery {
        SdfCapsule2dQuery {
            point,
            capsule_a,
            capsule_b,
            capsule_radius,
            segment_a,
            segment_b,
            annulus_radius,
            annulus_half_width,
        }
    }
}

/// One resolved query of the planar line/stadium/ring signed-distance twin: the
/// capsule, segment and annulus distances at the point.
///
/// `capsule_2d_sd` is
/// [`capsule_2d`](prism_render_architecture::ray_scene::sdf_primitives::capsule_2d);
/// `segment_2d_sd` is
/// [`segment_2d`](prism_render_architecture::ray_scene::sdf_primitives::segment_2d);
/// `annulus_2d_sd` is
/// [`annulus_2d`](prism_render_architecture::ray_scene::sdf_primitives::annulus_2d).
/// The capsule and annulus distances are negative inside the solid, positive
/// outside and zero on the boundary; the segment distance is unsigned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCapsule2dResult {
    /// Capsule (stadium) signed distance.
    pub capsule_2d_sd: f32,
    /// Segment unsigned distance.
    pub segment_2d_sd: f32,
    /// Annulus (ring) signed distance.
    pub annulus_2d_sd: f32,
}

/// Encodes one [`SdfCapsule2dQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfCapsule2dQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        cap_ax: q.capsule_a[0],
        cap_ay: q.capsule_a[1],
        cap_bx: q.capsule_b[0],
        cap_by: q.capsule_b[1],
        cap_r: q.capsule_radius,
        seg_ax: q.segment_a[0],
        seg_ay: q.segment_a[1],
        seg_bx: q.segment_b[0],
        seg_by: q.segment_b[1],
        ann_r: q.annulus_radius,
        ann_hw: q.annulus_half_width,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfCapsule2dResult`].
fn decode_result(raw: &GpuResult) -> SdfCapsule2dResult {
    SdfCapsule2dResult {
        capsule_2d_sd: raw.capsule_2d_sd,
        segment_2d_sd: raw.segment_2d_sd,
        annulus_2d_sd: raw.annulus_2d_sd,
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

/// A compiled, reusable planar line/stadium/ring signed-distance compute
/// pipeline, twinning the `CPU` golden
/// [`segment_2d`](prism_render_architecture::ray_scene::sdf_primitives::segment_2d),
/// [`capsule_2d`](prism_render_architecture::ray_scene::sdf_primitives::capsule_2d)
/// and
/// [`annulus_2d`](prism_render_architecture::ray_scene::sdf_primitives::annulus_2d).
pub struct GpuSdfCapsule2d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfCapsule2d {
    /// Compiles the planar line/stadium/ring signed-distance compute pipeline on
    /// the given context.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfCapsule2d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_capsule2d_module"),
            source: ShaderSource::Wgsl(SDF_CAPSULE2D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_capsule2d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_capsule2d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_capsule2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfCapsule2d {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfCapsule2dResult`]
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
        queries: &[SdfCapsule2dQuery],
    ) -> Vec<SdfCapsule2dResult> {
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
            label: Some("prism_volumetric_sdf_capsule2d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_capsule2d_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_capsule2d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_capsule2d_bind_group"),
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
            label: Some("prism_volumetric_sdf_capsule2d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_capsule2d_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_capsule2d_pass"),
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
