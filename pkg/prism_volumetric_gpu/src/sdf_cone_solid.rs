//! `wgpu` compute twin of three analytic cone/sector signed-distance primitives
//! of the `CPU` golden path
//! ([`cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::cone_sdf),
//! [`round_cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_sdf)
//! and
//! [`solid_angle`](prism_render_architecture::ray_scene::sdf_primitives::solid_angle)).
//!
//! Implicit modelling needs *analytic* primitives whose exact signed distance
//! is known in closed form rather than sampled on a grid. The reference derives
//! three revolved shapes around the `y` axis: a solid right circular
//! [`cone_sdf`] (apex at the origin, opening down `-y`), the tapered-capsule
//! [`round_cone_sdf`] (the convex hull of two spheres), and the "ice-cream"
//! [`solid_angle`] sector (a ball intersected with an infinite cone). Each
//! folds its only trigonometry into caller-supplied constants so the routine
//! stays transcendental-free. [`GpuSdfConeSolid`] is the on-device twin: each
//! thread reads one point plus all three shapes' parameters and writes all
//! three signed distances, reproducing the reference closed forms with only
//! `sqrt`, `abs`, `min`, `max`, `clamp`, products and quotients.
//!
//! # What is twinned
//!
//! Each thread reads one [`SdfConeSolidQuery`] — a query `point` plus the cone
//! (`base_radius`, `height`), round-cone (`r1`, `r2`, `round_height`) and
//! solid-angle (`sin_aperture`, `cos_aperture`, `radius`) parameters — and
//! writes one [`SdfConeSolidResult`] holding the three signed distances. All
//! three reduce the point to meridian coordinates `(radial distance from the y
//! axis, height)`, then evaluate the exact nearest-feature distance: the cone
//! takes the smaller of the lateral-edge and base-cap segment distances with
//! the sign recovered from two half-plane tests; the round cone selects one of
//! three governing regions by the flank-normal projection; and the solid angle
//! combines the bounding-ball distance with the signed flank distance.
//!
//! # What stays on the host
//!
//! The domain and `CSG` operators that compose these atoms into complex shapes,
//! the ray-marcher that evaluates them along a ray, and the surface-normal
//! estimation all stay on the host; the device sees only the three stateless,
//! fixed-width signed-distance evaluations, one query at a time, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! All three distances thread through `sqrt`, products and quotients, so the
//! `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` or divide may land a few
//! units in the last place from the scalar reference. The parity test asserts
//! each distance within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`. The cone's
//! sign and the solid angle's flank sign flip exactly on the surface where the
//! distance magnitude passes through zero, so the discrete flip is absorbed by
//! the absolute bound; fixtures otherwise stay clear of the flank rays where a
//! last-place difference could pick a different half-plane.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `abs`, `min`,
//! `max`, `clamp`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `round` and
//! no `ceil`, and no `f64`/`u64`/`u16`/`i64`/`i16`. The sign of a nonzero value
//! is recovered with an ordered comparison rather than the built-in `sign` so
//! it matches the reference `f32::signum` on the tested domain. It runs
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

/// The portable core-`WGSL` cone/sector signed-distance kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::cone_sdf),
/// [`round_cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_sdf)
/// and
/// [`solid_angle`](prism_render_architecture::ray_scene::sdf_primitives::solid_angle)
/// closed forms; see the module documentation for the algorithm.
const SDF_CONE_SOLID_WGSL: &str = r#"
// Cone/sector signed-distance twin: one thread computes one query point's solid
// cone, round-cone and solid-angle signed distances, mirroring the CPU golden
// `ray_scene::sdf_primitives::{cone_sdf, round_cone_sdf, solid_angle}` with only
// sqrt, abs, min, max, clamp, products and quotients. The domain/CSG operators
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
    // Solid cone: base radius and height.
    base_radius: f32,
    height: f32,
    // Round cone: lower radius, upper radius, axial length.
    r1: f32,
    r2: f32,
    round_height: f32,
    // Solid angle: aperture sine/cosine and ball radius.
    sin_aperture: f32,
    cos_aperture: f32,
    radius: f32,
    pad0: f32,
}

struct Distances {
    // Solid-cone signed distance.
    cone_sd: f32,
    // Round-cone signed distance.
    round_cone_sd: f32,
    // Solid-angle signed distance.
    solid_angle_sd: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Distances>;

// Euclidean length of a 2-vector.
fn length2(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

// Sign of a nonzero value, matching the reference `f32::signum` on the tested
// domain (negatives return -1, everything else returns +1). Fixtures stay away
// from the exact zero where the sign bit would decide.
fn sign_of(x: f32) -> f32 {
    if (x < 0.0) {
        return -1.0;
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Distances;

    // --- Solid cone (apex at origin, opening down -y) ---
    // Apex-to-base-rim edge in the meridian (radial, axial) plane.
    let cq0 = q.base_radius;
    let cq1 = -q.height;
    // Meridian query coordinates.
    let cw0 = length2(q.px, q.pz);
    let cw1 = q.py;
    let cq_dot = cq0 * cq0 + cq1 * cq1;
    let ct = clamp((cw0 * cq0 + cw1 * cq1) / cq_dot, 0.0, 1.0);
    let ca0 = cw0 - cq0 * ct;
    let ca1 = cw1 - cq1 * ct;
    let cu = clamp(cw0 / cq0, 0.0, 1.0);
    let cb0 = cw0 - cq0 * cu;
    let cb1 = cw1 - cq1;
    let ck = sign_of(cq1);
    let cd = min(ca0 * ca0 + ca1 * ca1, cb0 * cb0 + cb1 * cb1);
    let csign = max(ck * (cw0 * cq1 - cw1 * cq0), ck * (cw1 - cq1));
    out.cone_sd = sqrt(max(cd, 0.0)) * sign_of(csign);

    // --- Round cone (tapered capsule along +y) ---
    let rq0 = length2(q.px, q.pz);
    let rq1 = q.py;
    let b = (q.r1 - q.r2) / q.round_height;
    let a = sqrt(max(1.0 - b * b, 0.0));
    let rk = -b * rq0 + a * rq1;
    if (rk < 0.0) {
        out.round_cone_sd = length2(rq0, rq1) - q.r1;
    } else if (rk > a * q.round_height) {
        out.round_cone_sd = length2(rq0, rq1 - q.round_height) - q.r2;
    } else {
        out.round_cone_sd = a * rq0 + b * rq1 - q.r1;
    }

    // --- Solid angle (ball intersected with an infinite cone) ---
    let sq0 = length2(q.px, q.pz);
    let sq1 = q.py;
    let ball = length2(sq0, sq1) - q.radius;
    let proj = clamp(sq0 * q.sin_aperture + sq1 * q.cos_aperture, 0.0, q.radius);
    let flank = length2(sq0 - q.sin_aperture * proj, sq1 - q.cos_aperture * proj);
    var flank_sign: f32;
    if (q.cos_aperture * sq0 - q.sin_aperture * sq1 < 0.0) {
        flank_sign = -1.0;
    } else {
        flank_sign = 1.0;
    }
    out.solid_angle_sd = max(ball, flank * flank_sign);

    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_CONE_SOLID_WGSL`].
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
/// the point components plus all three shapes' parameters and one pad word to a
/// `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point `x`.
    px: f32,
    /// Query point `y`.
    py: f32,
    /// Query point `z`.
    pz: f32,
    /// Solid-cone base radius.
    base_radius: f32,
    /// Solid-cone height.
    height: f32,
    /// Round-cone lower radius.
    r1: f32,
    /// Round-cone upper radius.
    r2: f32,
    /// Round-cone axial length.
    round_height: f32,
    /// Solid-angle aperture sine.
    sin_aperture: f32,
    /// Solid-angle aperture cosine.
    cos_aperture: f32,
    /// Solid-angle ball radius.
    radius: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Distances`
/// struct: the three signed distances plus one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Solid-cone signed distance.
    cone_sd: f32,
    /// Round-cone signed distance.
    round_cone_sd: f32,
    /// Solid-angle signed distance.
    solid_angle_sd: f32,
    /// Padding word.
    pad0: f32,
}

/// One query for the cone/sector signed-distance twin: the query `point` plus
/// the solid-cone, round-cone and solid-angle shape parameters.
///
/// `point` is the evaluation position; `base_radius`/`height` describe the
/// [`cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::cone_sdf);
/// `r1`/`r2`/`round_height` describe the
/// [`round_cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_sdf);
/// `sin_aperture`/`cos_aperture`/`radius` describe the
/// [`solid_angle`](prism_render_architecture::ray_scene::sdf_primitives::solid_angle).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfConeSolidQuery {
    /// Query point `[x, y, z]`.
    pub point: [f32; 3],
    /// Solid-cone base radius.
    pub base_radius: f32,
    /// Solid-cone height.
    pub height: f32,
    /// Round-cone lower radius.
    pub r1: f32,
    /// Round-cone upper radius.
    pub r2: f32,
    /// Round-cone axial length.
    pub round_height: f32,
    /// Solid-angle aperture sine.
    pub sin_aperture: f32,
    /// Solid-angle aperture cosine.
    pub cos_aperture: f32,
    /// Solid-angle ball radius.
    pub radius: f32,
}

impl SdfConeSolidQuery {
    /// Builds a query from the point and all three shapes' parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the twin packs three independent primitives' parameters into one query"
    )]
    pub const fn new(
        point: [f32; 3],
        base_radius: f32,
        height: f32,
        r1: f32,
        r2: f32,
        round_height: f32,
        sin_aperture: f32,
        cos_aperture: f32,
        radius: f32,
    ) -> SdfConeSolidQuery {
        SdfConeSolidQuery {
            point,
            base_radius,
            height,
            r1,
            r2,
            round_height,
            sin_aperture,
            cos_aperture,
            radius,
        }
    }
}

/// One resolved query of the cone/sector signed-distance twin: the solid-cone,
/// round-cone and solid-angle signed distances at the query point.
///
/// `cone_sd` is
/// [`cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::cone_sdf);
/// `round_cone_sd` is
/// [`round_cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_sdf);
/// `solid_angle_sd` is
/// [`solid_angle`](prism_render_architecture::ray_scene::sdf_primitives::solid_angle).
/// All are negative inside the solid, positive outside, zero on the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfConeSolidResult {
    /// Solid-cone signed distance.
    pub cone_sd: f32,
    /// Round-cone signed distance.
    pub round_cone_sd: f32,
    /// Solid-angle signed distance.
    pub solid_angle_sd: f32,
}

/// Encodes one [`SdfConeSolidQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfConeSolidQuery) -> GpuQuery {
    GpuQuery {
        px: q.point[0],
        py: q.point[1],
        pz: q.point[2],
        base_radius: q.base_radius,
        height: q.height,
        r1: q.r1,
        r2: q.r2,
        round_height: q.round_height,
        sin_aperture: q.sin_aperture,
        cos_aperture: q.cos_aperture,
        radius: q.radius,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfConeSolidResult`].
fn decode_result(raw: &GpuResult) -> SdfConeSolidResult {
    SdfConeSolidResult {
        cone_sd: raw.cone_sd,
        round_cone_sd: raw.round_cone_sd,
        solid_angle_sd: raw.solid_angle_sd,
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

/// A compiled, reusable cone/sector signed-distance compute pipeline, twinning
/// the `CPU` golden
/// [`cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::cone_sdf),
/// [`round_cone_sdf`](prism_render_architecture::ray_scene::sdf_primitives::round_cone_sdf)
/// and
/// [`solid_angle`](prism_render_architecture::ray_scene::sdf_primitives::solid_angle).
pub struct GpuSdfConeSolid {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfConeSolid {
    /// Compiles the cone/sector signed-distance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfConeSolid {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_cone_solid"),
            source: ShaderSource::Wgsl(SDF_CONE_SOLID_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_cone_solid_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_cone_solid_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_cone_solid_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfConeSolid {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SdfConeSolidResult`]
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
        queries: &[SdfConeSolidQuery],
    ) -> Vec<SdfConeSolidResult> {
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
            label: Some("prism_volumetric_sdf_cone_solid_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_cone_solid_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_cone_solid_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_cone_solid_bind_group"),
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
            label: Some("prism_volumetric_sdf_cone_solid_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_cone_solid_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_cone_solid_pass"),
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
