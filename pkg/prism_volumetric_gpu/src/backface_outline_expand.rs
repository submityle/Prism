//! `wgpu` compute twin of the inverted-hull backface outline-expansion contract
//! ([`backface_outline_expand`](prism_render_architecture::particle::backface_outline_expand),
//! particle design §18).
//!
//! The `CPU` golden
//! [`backface_outline_expand`](prism_render_architecture::particle::backface_outline_expand)
//! owns the object-space geometry of the classic *背面外扩描边* (backface
//! expand) stroke: every vertex of an outline shell is pushed outward along its
//! (unit) shading normal by a resolved world distance `d`, and the shell is then
//! drawn front-face-culled so its back faces peek out around the silhouette.
//! [`GpuBackfaceOutlineExpand`] is the on-device twin: one thread per vertex
//! resolves the same distance and reproduces the same displacement, so a passing
//! real-device parity test is direct evidence the ported kernel moves every
//! vertex exactly where the reference does and classifies the same degenerate
//! normals, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each vertex reproduces the full per-vertex contract of the reference:
//!
//! * the resolved outward world distance
//!   [`OutlineExpandParams::expand_distance`](prism_render_architecture::particle::backface_outline_expand::OutlineExpandParams::expand_distance),
//!   which floors the thickness at `0.0`, resolves it per the active
//!   [`ExpandSpace`](prism_render_architecture::particle::backface_outline_expand::ExpandSpace)
//!   (`World` is the identity, `Screen` is the guarded division
//!   [`expand_distance_screen`](prism_render_architecture::particle::backface_outline_expand::expand_distance_screen)
//!   `d = pixel_width * view_depth / focal_scale`), and finally clamps to the
//!   positive `max_distance`;
//! * the displaced position
//!   [`displace_along_normal`](prism_render_architecture::particle::backface_outline_expand::displace_along_normal),
//!   `p + normalize_or_zero(n) * d`;
//! * the fallback displacement
//!   [`displace_with_fallback`](prism_render_architecture::particle::backface_outline_expand::displace_with_fallback),
//!   which substitutes the geometric `fallback` normal when the shading normal
//!   is degenerate (both degenerate leaves the vertex untouched);
//! * the signed
//!   [`outward_offset`](prism_render_architecture::particle::backface_outline_expand::outward_offset),
//!   the projection of the applied displacement onto the unit normal.
//!
//! The reference's two spaces are mirrored branch for branch, and the two
//! degeneracy guards — a `focal_scale` at or below `MIN_FOCAL` collapsing the
//! screen distance to `0.0`, and a zero-length normal normalizing to the zero
//! vector so the vertex is left unmoved rather than emitting a `NaN` — are
//! reproduced tap for tap.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `+ - * /`, unsigned comparison and `sqrt` (used solely by the guarded
//! `normalize_or_zero`) — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan` and
//! no optional device feature. In particular the screen-space distance is a
//! guarded division; the caller owns the projection and supplies `focal_scale`
//! already, so the kernel never evaluates `tan`, exactly as the reference
//! documents. It therefore runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Each vertex is a fixed, non-reorderable sequence of multiplies, adds and
//! divides (plus one `sqrt` inside `normalize_or_zero`), so `CPU` and `GPU`
//! evaluate the same closed form in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The parity
//! test therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`)
//! on the `f32` fields, tight enough to catch a genuinely wrong port (a dropped
//! branch, a swapped coefficient, a wrong clamp) yet loose enough to admit legal
//! fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`backface_outline_expand`](prism_render_architecture::particle::backface_outline_expand);
//! no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::backface_outline_expand::{
    ExpandSpace, OutlineExpandParams,
};
use prism_render_architecture::particle::Vec3;
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

/// `std430` tag written for [`ExpandSpace::World`], matching the reference
/// `SPACE_TAG_WORLD`.
const SPACE_TAG_WORLD: u32 = 0;

/// `std430` tag written for [`ExpandSpace::Screen`], matching the reference
/// `SPACE_TAG_SCREEN`.
const SPACE_TAG_SCREEN: u32 = 1;

/// The portable core-`WGSL` outline-expansion kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `expand` mirrors
/// the `CPU` golden
/// [`backface_outline_expand`](prism_render_architecture::particle::backface_outline_expand)
/// per-vertex contract; see the module documentation for the algorithm.
const BACKFACE_OUTLINE_EXPAND_WGSL: &str = r#"
// Backface outline-expand twin: one thread per vertex reproduces the resolved
// outward distance, the normal displacement, the fallback displacement and the
// signed outward offset. It mirrors the CPU golden
// `particle::backface_outline_expand` branch for branch, uses only the portable
// core-WGSL subset (min/max/abs and + - * / plus unsigned compares) with sqrt
// used solely by the guarded normalize, never evaluates tan (the caller owns
// the projection and supplies focal_scale), and takes no optional feature, so
// it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// particle::backface_outline_expand; no third-party engine source or derived
// code.

// Squared-length threshold below which a vector is treated as the zero vector
// by normalize_or_zero, matching the reference `Vec3::normalize_or_zero`
// (`EPS_LEN_SQ = 1e-12`), without ever writing an exact == / != on an f32.
const EPS_LEN_SQ: f32 = 1.0e-12;

// Minimum focal length (in pixels) guarding the screen-space division, matching
// the reference `MIN_FOCAL`. A focal length at or below this magnitude collapses
// the resolved distance to zero rather than dividing by (near) zero.
const MIN_FOCAL: f32 = 1.0e-6;

// std430 space tags, matching the reference SPACE_TAG_WORLD / SPACE_TAG_SCREEN.
const SPACE_TAG_WORLD: u32 = 0u;

struct Params {
    // Stroke thickness: world units for World space, pixels for Screen space.
    thickness: f32,
    // Vertical focal length in pixels for Screen space (unused for World).
    focal_scale: f32,
    // Hard upper bound on the resolved world distance; non-positive is unbounded.
    max_distance: f32,
    // View-space +z depth at which a Screen-space stroke is resolved.
    view_depth: f32,
    // Number of vertices in the storage arrays.
    count: u32,
    // Active space: SPACE_TAG_WORLD or SPACE_TAG_SCREEN.
    space_tag: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

struct Vertex {
    // Object-space vertex position; a pad lane follows.
    position: vec3<f32>,
    pad0: f32,
    // Object-space shading normal (need not be unit length); a pad lane follows.
    normal: vec3<f32>,
    pad1: f32,
    // Geometric fallback normal used when `normal` is degenerate; a pad follows.
    fallback: vec3<f32>,
    pad2: f32,
}

struct Result {
    // displace_along_normal position; a pad lane follows.
    displaced: vec3<f32>,
    pad0: f32,
    // displace_with_fallback position; a pad lane follows.
    fallback_displaced: vec3<f32>,
    pad1: f32,
    // Resolved outward world distance, then the signed outward offset; two pads.
    distance: f32,
    outward_offset: f32,
    pad2: f32,
    pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unit vector along `v`, or the zero vector when `v` is (numerically) zero, so
// normalization never yields a NaN. Mirrors `Vec3::normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// World-space distance keeping a `pixel_width`-wide stroke constant on screen at
// `view_depth`: d = pixel_width * view_depth / focal_scale. A degenerate
// focal_scale (|focal_scale| <= MIN_FOCAL) yields 0.0. Mirrors
// `expand_distance_screen`; never evaluates tan.
fn expand_distance_screen(pixel_width: f32, view_depth: f32, focal_scale: f32) -> f32 {
    if (abs(focal_scale) <= MIN_FOCAL) {
        return 0.0;
    }
    return pixel_width * view_depth / focal_scale;
}

// Resolves the outward world distance for the active space, mirroring
// `OutlineExpandParams::expand_distance`: floor the thickness at 0, resolve per
// space (World is the identity, Screen is the guarded division), then clamp to a
// positive max_distance.
fn resolve_distance() -> f32 {
    let thickness = max(params.thickness, 0.0);
    var raw: f32;
    if (params.space_tag == SPACE_TAG_WORLD) {
        raw = thickness;
    } else {
        raw = expand_distance_screen(thickness, params.view_depth, params.focal_scale);
    }
    if (params.max_distance > 0.0) {
        return min(raw, params.max_distance);
    }
    return raw;
}

@compute @workgroup_size(64)
fn expand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let v = vertices[idx];
    let dist = resolve_distance();

    // displace_along_normal: p + normalize_or_zero(n) * d.
    let unit = normalize_or_zero(v.normal);
    let displaced = v.position + unit * dist;

    // outward_offset: projection of (displaced - position) onto the unit normal.
    let offset = dot(displaced - v.position, unit);

    // displace_with_fallback: use the shading normal when it survives
    // normalization, otherwise the geometric fallback; both degenerate leaves
    // the vertex untouched.
    var dir = unit;
    if (!(dot(unit, unit) > 0.0)) {
        dir = normalize_or_zero(v.fallback);
    }
    let fallback_displaced = v.position + dir * dist;

    var out: Result;
    out.displaced = displaced;
    out.pad0 = 0.0;
    out.fallback_displaced = fallback_displaced;
    out.pad1 = 0.0;
    out.distance = dist;
    out.outward_offset = offset;
    out.pad2 = 0.0;
    out.pad3 = 0.0;
    results[idx] = out;
}
"#;

/// One vertex of the outline shell: the inputs the reference displacement
/// functions consume.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineExpandVertex {
    /// Object-space vertex position.
    pub position: Vec3,
    /// Object-space shading normal (need not be unit length). [`Vec3::ZERO`]
    /// means "no authored normal" and triggers the fallback branch.
    pub normal: Vec3,
    /// Geometric fallback normal used by `displace_with_fallback` when `normal`
    /// is degenerate.
    pub fallback: Vec3,
}

impl OutlineExpandVertex {
    /// Builds a vertex from a position, shading normal and fallback normal.
    #[must_use]
    pub const fn new(position: Vec3, normal: Vec3, fallback: Vec3) -> OutlineExpandVertex {
        OutlineExpandVertex {
            position,
            normal,
            fallback,
        }
    }
}

/// The resolved outline-expansion answer for one vertex, mirroring every value
/// the reference reports per vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineExpandResult {
    /// The displaced position, matching `displace_along_normal` /
    /// `OutlineExpandParams::displace_vertex`.
    pub displaced: Vec3,
    /// The displaced position using the fallback branch, matching
    /// `displace_with_fallback` at the resolved distance.
    pub fallback_displaced: Vec3,
    /// The resolved outward world distance, matching
    /// `OutlineExpandParams::expand_distance`.
    pub distance: f32,
    /// The signed applied offset, matching `outward_offset`.
    pub outward_offset: f32,
}

/// `repr(C)` `std430` uniform parameters for one dispatch: the thickness,
/// focal length, distance clamp and view depth filling one `vec4`, then the
/// vertex count, the space tag and two pad words filling a second `vec4` — `32`
/// bytes matching the `WGSL` `Params` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Stroke thickness (world units or pixels per the space tag).
    thickness: f32,
    /// Vertical focal length in pixels for the screen-space division.
    focal_scale: f32,
    /// Hard upper bound on the resolved world distance; non-positive is unbounded.
    max_distance: f32,
    /// View-space `+z` depth at which a screen-space stroke is resolved.
    view_depth: f32,
    /// Number of vertices in the storage arrays.
    count: u32,
    /// Active space discriminant (`0` world, `1` screen).
    space_tag: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one packed vertex: three `vec4` slots holding
/// `(position.xyz, pad)`, `(normal.xyz, pad)` and `(fallback.xyz, pad)` — `48`
/// bytes, each `vec3` on its `16`-byte-aligned slot exactly as the `WGSL`
/// `Vertex` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVertex {
    /// Object-space vertex position.
    position: [f32; 3],
    /// Padding lane after the position.
    pad0: f32,
    /// Object-space shading normal.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad1: f32,
    /// Geometric fallback normal.
    fallback: [f32; 3],
    /// Padding lane after the fallback.
    pad2: f32,
}

impl GpuVertex {
    /// Packs one vertex into its `std430` image.
    fn new(vertex: &OutlineExpandVertex) -> GpuVertex {
        GpuVertex {
            position: [vertex.position.x, vertex.position.y, vertex.position.z],
            pad0: 0.0,
            normal: [vertex.normal.x, vertex.normal.y, vertex.normal.z],
            pad1: 0.0,
            fallback: [vertex.fallback.x, vertex.fallback.y, vertex.fallback.z],
            pad2: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a `vec4` slot for the displaced
/// position, a `vec4` slot for the fallback displacement, then the resolved
/// distance, the signed outward offset and two pad lanes filling a third `vec4`
/// — `48` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Displaced position.
    displaced: [f32; 3],
    /// Padding lane after the displaced position.
    pad0: f32,
    /// Fallback-branch displaced position.
    fallback_displaced: [f32; 3],
    /// Padding lane after the fallback displacement.
    pad1: f32,
    /// Resolved outward world distance.
    distance: f32,
    /// Signed applied outward offset.
    outward_offset: f32,
    /// Padding lane.
    pad2: f32,
    /// Padding lane.
    pad3: f32,
}

/// A compiled, reusable backface outline-expansion compute pipeline, twinning
/// the `CPU` golden
/// [`backface_outline_expand`](prism_render_architecture::particle::backface_outline_expand).
pub struct GpuBackfaceOutlineExpand {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBackfaceOutlineExpand {
    /// Compiles the outline-expansion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBackfaceOutlineExpand {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_backface_outline_expand"),
            source: ShaderSource::Wgsl(BACKFACE_OUTLINE_EXPAND_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_backface_outline_expand_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_backface_outline_expand_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_backface_outline_expand_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("expand"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBackfaceOutlineExpand {
            module,
            layout,
            pipeline,
        }
    }

    /// Expands every vertex on-device for the given `params` at a single
    /// `view_depth`, returning one [`OutlineExpandResult`] per input, in order.
    ///
    /// Each result equals the reference answers
    /// (`OutlineExpandParams::expand_distance`, `displace_along_normal`,
    /// `displace_with_fallback` and `outward_offset`) to within the tolerance
    /// documented on this module. An empty input returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        params: &OutlineExpandParams,
        view_depth: f32,
        vertices: &[OutlineExpandVertex],
    ) -> Vec<OutlineExpandResult> {
        if vertices.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = vertices.len();

        let (focal_scale, space_tag) = match params.space {
            ExpandSpace::World => (0.0_f32, SPACE_TAG_WORLD),
            ExpandSpace::Screen { focal_scale } => (focal_scale, SPACE_TAG_SCREEN),
        };
        let gpu_params = GpuParams {
            thickness: params.thickness,
            focal_scale,
            max_distance: params.max_distance,
            view_depth,
            count: count as u32,
            space_tag,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_backface_outline_expand_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });

        let packed: Vec<GpuVertex> = vertices.iter().map(GpuVertex::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_backface_outline_expand_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_backface_outline_expand_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_backface_outline_expand_bind_group"),
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
            label: Some("prism_volumetric_backface_outline_expand_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_backface_outline_expand_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_backface_outline_expand_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per vertex, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`OutlineExpandResult`].
fn decode_result(raw: &GpuResult) -> OutlineExpandResult {
    OutlineExpandResult {
        displaced: Vec3::new(raw.displaced[0], raw.displaced[1], raw.displaced[2]),
        fallback_displaced: Vec3::new(
            raw.fallback_displaced[0],
            raw.fallback_displaced[1],
            raw.fallback_displaced[2],
        ),
        distance: raw.distance,
        outward_offset: raw.outward_offset,
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
