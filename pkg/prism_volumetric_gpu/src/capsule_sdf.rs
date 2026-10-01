//! `wgpu` compute twin of the analytic capsule signed-distance-field (`SDF`)
//! golden
//! ([`capsule_sdf`](prism_render_architecture::particle::capsule_sdf),
//! particle design §10, §14).
//!
//! The `CPU` golden
//! [`capsule_sdf`](prism_render_architecture::particle::capsule_sdf) owns the
//! closed-form distance to a capsule: the line segment `a`-`b` inflated by a
//! sweep radius `r`. Its core is the clamped point-to-segment projection
//! ([`sd_segment`](prism_render_architecture::particle::capsule_sdf::sd_segment)):
//! the query point `p` is projected onto the core axis with the parameter
//! `t = clamp(dot(p - a, b - a) / dot(b - a, b - a), 0, 1)`, the closest axis
//! point is `c = a + t * (b - a)`, and the unsigned axis distance is
//! `length(p - c)`. The capsule field
//! ([`sd_capsule`](prism_render_architecture::particle::capsule_sdf::sd_capsule))
//! then subtracts the radius, so it is negative inside the swept volume, zero on
//! the surface and positive outside.
//!
//! [`GpuCapsuleSdf`] is the on-device twin: one thread per `(point, capsule)`
//! query reproduces the same closed form branch for branch, so a passing
//! real-device parity test is direct evidence the ported kernel evaluates the
//! same geometry and classifies the same degenerate case (a collapsed segment
//! `a` ≈ `b`, which falls back to a sphere distance about `a`) the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Both distances the reference exposes are reproduced per query: the unsigned
//! `segment_distance` from `p` to the core axis
//! ([`sd_segment`](prism_render_architecture::particle::capsule_sdf::sd_segment))
//! and the signed `signed_distance` to the capsule surface
//! ([`sd_capsule`](prism_render_architecture::particle::capsule_sdf::sd_capsule)).
//! The reference's single degenerate branch is mirrored: a core-axis squared
//! length at or below the compare epsilon collapses the projection parameter to
//! zero, so the distance is measured from the endpoint `a` (a sphere `SDF`)
//! instead of dividing by zero.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `abs`, `+ - * /`, the `dot` builtin and one `sqrt` for the genuine
//! Euclidean length — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` distances, tight
//! enough to catch a genuinely wrong port (a dropped branch, a swapped
//! coefficient, a wrong clamp) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；
//! standard `iq`-style capsule signed-distance field plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::capsule_sdf::{sd_capsule, sd_segment, Vec3};
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

/// The portable core-`WGSL` capsule-`SDF` kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`capsule_sdf`](prism_render_architecture::particle::capsule_sdf) branch for
/// branch; see the module documentation for the algorithm.
const CAPSULE_SDF_WGSL: &str = r#"
// Capsule-SDF twin: one thread per (point, capsule) query reproduces the
// unsigned point-to-segment distance and the signed capsule distance. It
// mirrors the CPU golden particle::capsule_sdf branch for branch, uses only the
// portable core-WGSL subset (clamp/min/max/abs and + - * / plus the dot builtin
// and one sqrt) and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::capsule_sdf; no
// third-party engine source or derived code.

// Epsilon that guards the one division (the core-axis squared length) so the
// kernel never writes an exact == / != on an f32 and never emits a NaN for a
// collapsed segment. Matches the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Query point projected onto the core axis; the fourth lane carries the
    // sweep radius r.
    point: vec3<f32>,
    r: f32,
    // Core-axis start endpoint a; a pad lane follows.
    a: vec3<f32>,
    pad0: f32,
    // Core-axis end endpoint b; a pad lane follows.
    b: vec3<f32>,
    pad1: f32,
}

struct Result {
    // Unsigned point-to-segment distance, the signed capsule distance and two
    // pad lanes: four scalars filling one vec4 slot.
    segment_distance: f32,
    signed_distance: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unsigned distance from p to the segment a -> b, mirroring the reference
// `sd_segment`. The projection parameter is clamped to [0, 1] so points beyond
// either end measure to the nearer endpoint; a degenerate segment (squared
// length at or below CMP_EPS) collapses the parameter to 0, the endpoint a.
fn segment_distance(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let denom = dot(ba, ba);
    var h: f32 = 0.0;
    if (denom <= CMP_EPS) {
        h = 0.0;
    } else {
        h = clamp(dot(pa, ba) / denom, 0.0, 1.0);
    }
    let d = pa - ba * h;
    return sqrt(dot(d, d));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let seg = segment_distance(q.point, q.a, q.b);

    var out: Result;
    out.segment_distance = seg;
    out.signed_distance = seg - q.r;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One capsule-`SDF` query: the point `p` evaluated against the capsule whose
/// core axis is the segment `a`-`b` inflated by the sweep radius `r` — the same
/// inputs the reference
/// [`sd_capsule`](prism_render_architecture::particle::capsule_sdf::sd_capsule)
/// consumes.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleSdfQuery {
    /// The query point whose distance to the capsule is evaluated.
    pub point: Vec3,
    /// Core-axis start endpoint `a`.
    pub a: Vec3,
    /// Core-axis end endpoint `b`.
    pub b: Vec3,
    /// Sweep radius `r` around the core axis.
    pub r: f32,
}

impl CapsuleSdfQuery {
    /// Builds a query from the evaluation point, the two core-axis endpoints and
    /// the sweep radius.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(point: Vec3, a: Vec3, b: Vec3, r: f32) -> CapsuleSdfQuery {
        CapsuleSdfQuery { point, a, b, r }
    }
}

/// The resolved answer for one query, mirroring both distances the reference
/// exposes.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapsuleSdfResult {
    /// Unsigned distance from the query point to the core axis, matching
    /// [`sd_segment`](prism_render_architecture::particle::capsule_sdf::sd_segment).
    pub segment_distance: f32,
    /// Signed distance to the capsule surface (negative inside, zero on the
    /// shell, positive outside), matching
    /// [`sd_capsule`](prism_render_architecture::particle::capsule_sdf::sd_capsule).
    pub signed_distance: f32,
}

/// Evaluates the `CPU` golden for one query, delegating field for field to the
/// reference
/// [`sd_segment`](prism_render_architecture::particle::capsule_sdf::sd_segment)
/// and
/// [`sd_capsule`](prism_render_architecture::particle::capsule_sdf::sd_capsule)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &CapsuleSdfQuery) -> CapsuleSdfResult {
    CapsuleSdfResult {
        segment_distance: sd_segment(query.point, query.a, query.b),
        signed_distance: sd_capsule(query.point, query.a, query.b, query.r),
    }
}

/// `repr(C)` `std430` layout of one packed query: three `vec4` slots holding
/// `(point.xyz, r)`, `(a.xyz, pad)` and `(b.xyz, pad)` — `48` bytes, each
/// `vec3` on its `16`-byte-aligned slot exactly as the `WGSL` `Query` struct
/// reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Query point.
    point: [f32; 3],
    /// Sweep radius, packed in the fourth lane of the first slot.
    r: f32,
    /// Core-axis start endpoint.
    a: [f32; 3],
    /// Padding lane after the start endpoint.
    pad0: f32,
    /// Core-axis end endpoint.
    b: [f32; 3],
    /// Padding lane after the end endpoint.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &CapsuleSdfQuery) -> GpuQuery {
        GpuQuery {
            point: [query.point.x, query.point.y, query.point.z],
            r: query.r,
            a: [query.a.x, query.a.y, query.a.z],
            pad0: 0.0,
            b: [query.b.x, query.b.y, query.b.z],
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot
/// `(segment_distance, signed_distance, pad, pad)` — `16` bytes matching the
/// `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Unsigned point-to-segment distance.
    segment_distance: f32,
    /// Signed capsule distance.
    signed_distance: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
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

/// A compiled, reusable capsule-`SDF` compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；
/// no third-party engine source or derived code.
pub struct GpuCapsuleSdf {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCapsuleSdf {
    /// Compiles the capsule-`SDF` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapsuleSdf {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_capsule_sdf"),
            source: ShaderSource::Wgsl(CAPSULE_SDF_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_capsule_sdf_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_capsule_sdf_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_capsule_sdf_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCapsuleSdf {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`CapsuleSdfResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference
    /// [`sd_segment`](prism_render_architecture::particle::capsule_sdf::sd_segment)
    /// and
    /// [`sd_capsule`](prism_render_architecture::particle::capsule_sdf::sd_capsule)
    /// answers to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[CapsuleSdfQuery]) -> Vec<CapsuleSdfResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_capsule_sdf_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_capsule_sdf_output"),
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
            label: Some("prism_volumetric_capsule_sdf_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_capsule_sdf_bind_group"),
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
            label: Some("prism_volumetric_capsule_sdf_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_capsule_sdf_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_capsule_sdf_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`CapsuleSdfResult`].
fn decode_result(raw: &GpuResult) -> CapsuleSdfResult {
    CapsuleSdfResult {
        segment_distance: raw.segment_distance,
        signed_distance: raw.signed_distance,
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
