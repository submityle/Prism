//! `wgpu` compute twin of the deterministic decal-projection golden
//! ([`decal`](prism_render_architecture::particle::decal), particle design §15).
//!
//! The `CPU` golden [`decal`](prism_render_architecture::particle::decal) turns
//! an oriented-bounding-box (`OBB`) projector into a surface query: given a
//! world-space receiver point and its surface normal, it maps the point into the
//! projector's normalized local cube `[-1, 1]^3`, clips against that cube, and
//! combines an angle fade with a depth fade into a single opacity. The one-call
//! composition is
//! [`projected_sample`](prism_render_architecture::particle::decal::projected_sample):
//! it computes the local coordinates once
//! ([`world_to_decal_local`](prism_render_architecture::particle::decal::world_to_decal_local)),
//! clips with the unit-interval test
//! ([`within_unit`](prism_render_architecture::particle::decal) feeds
//! [`contains`](prism_render_architecture::particle::decal::contains) and
//! [`decal_uv`](prism_render_architecture::particle::decal::decal_uv)), remaps
//! the local `x`/`y` to a `UV` in `[0, 1]^2`, then multiplies
//! [`angle_fade`](prism_render_architecture::particle::decal::angle_fade) by
//! [`depth_fade`](prism_render_architecture::particle::decal::depth_fade).
//!
//! [`GpuDecal`] is the on-device twin: one thread per receiver point reproduces
//! that composition branch for branch, so a passing real-device parity test is
//! direct evidence the ported kernel maps the same `UV`, folds the same combined
//! fade and classifies the same clip (inside versus clipped) the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries a projector (world `center` plus an orthonormal
//! `right`/`up`/`forward` basis and per-axis `half_extents`), the two fade bands
//! and one receiver `(world, surface_normal)` pair. The kernel reproduces
//! [`projected_sample`](prism_render_architecture::particle::decal::projected_sample)
//! and writes a discrete `hit` flag (the `Option` tag: `1` for `Some`, `0` for
//! the clipped `None`), the sampled `UV` and the combined `fade`. The projector
//! basis is built host-side by the reference
//! [`from_forward_up`](prism_render_architecture::particle::decal::DecalProjector::from_forward_up)
//! so the kernel receives an already-orthonormal, degenerate-safe frame and
//! focuses on the per-point projection math.
//!
//! The reference's degenerate branches are mirrored exactly: a half-extent at or
//! below the compare epsilon yields a local `0.0` on that axis (the guarded
//! divide `safe_ratio`); a back-facing surface (alignment at or below zero)
//! fades the angle term to `0.0`; a collapsed angle band (`cos_full` within the
//! epsilon of `cos_threshold`) becomes a hard step at `cos_full`; and a
//! collapsed depth band (`fade_end` within the epsilon of `fade_start`) becomes
//! a hard step at `fade_end`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `abs`, `+ - * /`, the `dot` builtin and one `sqrt` for the
//! `normalize_or_zero` length — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, `smoothstep` or `round`, and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Every angle enters as a numeric
//! cosine obtained from a `dot`, never as radians, exactly like the reference.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` `UV` and `fade`
//! fields while pinning the discrete `hit` flag with an exact `==`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::decal`；
//! standard deferred-decal `OBB` projection plus `wgpu` compute dispatch; no
//! third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::decal::{
    projected_sample, DecalFadeParams, DecalProjector,
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

/// Discrete `hit` code written by the kernel for a receiver point inside the
/// projection box: matches the host `== 1` decode in [`decode_result`]. The flag
/// carries the reference `Option` tag (`Some` versus the clipped `None`) as a
/// `u32` so parity pins it with an exact integer compare instead of an `f32`
/// equality.
const CODE_HIT: u32 = 1;

/// The portable core-`WGSL` decal-projection kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`projected_sample`](prism_render_architecture::particle::decal::projected_sample)
/// branch for branch; see the module documentation for the algorithm.
const DECAL_WGSL: &str = r#"
// Decal-projection twin: one thread per receiver point reproduces the CPU
// golden particle::decal::projected_sample branch for branch. It maps the world
// point into the projector's normalized local cube, clips against it, remaps the
// local x/y to a UV in [0, 1]^2 and multiplies the angle fade by the depth fade.
// It uses only the portable core-WGSL subset (clamp/min/max/abs and + - * / plus
// the dot builtin and one sqrt for normalize_or_zero) and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::decal; no third-party
// engine source or derived code.

// Guards the per-axis half-extent and fade-band divisions and stands in for the
// reference `EPS`, so the kernel never writes an exact == / != on an f32 and
// never emits a NaN for a zero-thickness projector or a collapsed fade band.
const DECAL_EPS: f32 = 1.0e-6;

// Squared-length floor below which a vector normalizes to zero, matching the
// reference `EPS_LEN_SQ` used by `normalize_or_zero`.
const DECAL_EPS_LEN_SQ: f32 = 1.0e-12;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // World-space projector center; a pad lane follows each vec3 so every field
    // sits on its 16-byte std430 slot.
    center: vec3<f32>,
    pad_c: f32,
    // Unit basis axis mapped to local x (UV u).
    right: vec3<f32>,
    pad_r: f32,
    // Unit basis axis mapped to local y (UV v).
    up: vec3<f32>,
    pad_u: f32,
    // Unit projection direction mapped to local z (depth).
    forward: vec3<f32>,
    pad_f: f32,
    // Positive half-sizes along right, up and forward.
    half_extents: vec3<f32>,
    pad_h: f32,
    // The receiver point queried against the projector.
    world: vec3<f32>,
    pad_w: f32,
    // The receiver surface normal feeding the angle fade.
    surface_normal: vec3<f32>,
    pad_n: f32,
    // Fade bands packed into one vec4:
    // (depth_fade_start, depth_fade_end, cos_threshold, cos_full).
    fade: vec4<f32>,
}

struct Result {
    // Option tag: 1 inside the box (Some), 0 clipped (None).
    hit: u32,
    // Sampled UV in [0, 1]^2 and the combined angle*depth fade; meaningful only
    // when hit == 1, zero on a clipped lane.
    uv_u: f32,
    uv_v: f32,
    fade: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Divides numerator by denominator, falling back to 0 when the denominator is
// within DECAL_EPS of zero, mirroring the reference `safe_ratio`.
fn safe_ratio(numerator: f32, denominator: f32) -> f32 {
    if (abs(denominator) <= DECAL_EPS) {
        return 0.0;
    }
    return numerator / denominator;
}

// True when value lies within the local unit interval [-1, 1], mirroring the
// reference `within_unit`.
fn within_unit(value: f32) -> bool {
    return value >= -1.0 && value <= 1.0;
}

// Unit vector along v, or zero when v is (numerically) the zero vector, matching
// the reference `normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > DECAL_EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// World point into the projector's normalized local coordinates, mirroring the
// reference `world_to_decal_local`.
fn world_to_decal_local(q: Query) -> vec3<f32> {
    let delta = q.world - q.center;
    return vec3<f32>(
        safe_ratio(dot(delta, q.right), q.half_extents.x),
        safe_ratio(dot(delta, q.up), q.half_extents.y),
        safe_ratio(dot(delta, q.forward), q.half_extents.z),
    );
}

// Angle-based edge fade, mirroring the reference `angle_fade`. The alignment is
// -(normal . forward) after normalizing both; a back-facing surface fades to 0,
// a degenerate band collapses to a hard step at cos_full.
fn angle_fade(
    surface_normal: vec3<f32>,
    projector_forward: vec3<f32>,
    cos_threshold: f32,
    cos_full: f32,
) -> f32 {
    let normal = normalize_or_zero(surface_normal);
    let forward = normalize_or_zero(projector_forward);
    let alignment = -dot(normal, forward);
    if (alignment <= 0.0) {
        return 0.0;
    }
    let span = cos_full - cos_threshold;
    if (abs(span) <= DECAL_EPS) {
        if (alignment >= cos_full) {
            return 1.0;
        }
        return 0.0;
    }
    return clamp((alignment - cos_threshold) / span, 0.0, 1.0);
}

// Depth-based edge fade, mirroring the reference `depth_fade`. Full at and
// before fade_start, zero at and after fade_end; a degenerate band collapses to
// a hard step at fade_end.
fn depth_fade(local_z: f32, fade_start: f32, fade_end: f32) -> f32 {
    let span = fade_end - fade_start;
    if (abs(span) <= DECAL_EPS) {
        if (local_z >= fade_end) {
            return 0.0;
        }
        return 1.0;
    }
    return clamp((fade_end - local_z) / span, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let local = world_to_decal_local(q);

    var out: Result;
    if (!within_unit(local.x) || !within_unit(local.y) || !within_unit(local.z)) {
        // Clipped by the projection box: the reference returns None.
        out.hit = 0u;
        out.uv_u = 0.0;
        out.uv_v = 0.0;
        out.fade = 0.0;
        results[idx] = out;
        return;
    }

    let angle = angle_fade(q.surface_normal, q.forward, q.fade.z, q.fade.w);
    let depth = depth_fade(local.z, q.fade.x, q.fade.y);
    out.hit = 1u;
    out.uv_u = (local.x + 1.0) * 0.5;
    out.uv_v = (local.y + 1.0) * 0.5;
    out.fade = angle * depth;
    results[idx] = out;
}
"#;

/// One decal-projection query: a projector, the two fade bands and one receiver
/// `(world, surface_normal)` pair — the same inputs the reference
/// [`projected_sample`](prism_render_architecture::particle::decal::projected_sample)
/// consumes.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::decal`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecalQuery {
    /// The oriented-box projector, built host-side by
    /// [`DecalProjector::from_forward_up`](prism_render_architecture::particle::decal::DecalProjector::from_forward_up).
    pub projector: DecalProjector,
    /// The angle and depth fade bands.
    pub params: DecalFadeParams,
    /// The receiver point queried against the projector.
    pub world: Vec3,
    /// The receiver surface normal feeding the angle fade.
    pub surface_normal: Vec3,
}

impl DecalQuery {
    /// Builds a query from a projector, its fade bands and one receiver point.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::decal`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(
        projector: DecalProjector,
        params: DecalFadeParams,
        world: Vec3,
        surface_normal: Vec3,
    ) -> DecalQuery {
        DecalQuery {
            projector,
            params,
            world,
            surface_normal,
        }
    }
}

/// The resolved projection for one query, mirroring the reference
/// [`Option<DecalSample>`](prism_render_architecture::particle::decal::DecalSample).
///
/// `hit` is the `Option` tag: `true` for an inside point (`Some`), `false` for a
/// clipped point (`None`). `uv` and `fade` are meaningful only when `hit` is
/// `true`; they are zero on a clipped lane.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::decal`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecalProjection {
    /// `true` when the receiver point lies inside the projection box (`Some`),
    /// `false` when the projection clips it (`None`).
    pub hit: bool,
    /// Sampled decal texture coordinate in `[0, 1]^2`; meaningful only when
    /// `hit`.
    pub uv: [f32; 2],
    /// Combined (angle times depth) opacity multiplier in `[0, 1]`; meaningful
    /// only when `hit`.
    pub fade: f32,
}

/// Evaluates the `CPU` golden for one query, mapping the reference
/// [`projected_sample`](prism_render_architecture::particle::decal::projected_sample)
/// `Option` into the flat [`DecalProjection`] the parity test pins against.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::decal`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &DecalQuery) -> DecalProjection {
    match projected_sample(
        &query.projector,
        query.world,
        query.surface_normal,
        query.params,
    ) {
        Some(sample) => DecalProjection {
            hit: true,
            uv: sample.uv,
            fade: sample.fade,
        },
        None => DecalProjection {
            hit: false,
            uv: [0.0, 0.0],
            fade: 0.0,
        },
    }
}

/// `repr(C)` `std430` layout of one packed query: seven `vec4` slots carrying
/// `center`, `right`, `up`, `forward`, `half_extents`, `world` and
/// `surface_normal` (each `vec3` on its `16`-byte-aligned slot, a pad lane
/// filling the fourth) plus one `vec4` holding the two fade bands — `128` bytes
/// matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Projector center.
    center: [f32; 3],
    /// Padding lane after the center.
    pad_c: f32,
    /// Local-`x` basis axis.
    right: [f32; 3],
    /// Padding lane after `right`.
    pad_r: f32,
    /// Local-`y` basis axis.
    up: [f32; 3],
    /// Padding lane after `up`.
    pad_u: f32,
    /// Projection direction (local `z`).
    forward: [f32; 3],
    /// Padding lane after `forward`.
    pad_f: f32,
    /// Per-axis half-extents.
    half_extents: [f32; 3],
    /// Padding lane after the half-extents.
    pad_h: f32,
    /// Receiver point.
    world: [f32; 3],
    /// Padding lane after the receiver point.
    pad_w: f32,
    /// Receiver surface normal.
    surface_normal: [f32; 3],
    /// Padding lane after the normal.
    pad_n: f32,
    /// Fade bands: `(depth_fade_start, depth_fade_end, cos_threshold,
    /// cos_full)`.
    fade: [f32; 4],
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &DecalQuery) -> GpuQuery {
        let p = &query.projector;
        let f = &query.params;
        GpuQuery {
            center: [p.center.x, p.center.y, p.center.z],
            pad_c: 0.0,
            right: [p.right.x, p.right.y, p.right.z],
            pad_r: 0.0,
            up: [p.up.x, p.up.y, p.up.z],
            pad_u: 0.0,
            forward: [p.forward.x, p.forward.y, p.forward.z],
            pad_f: 0.0,
            half_extents: [p.half_extents.x, p.half_extents.y, p.half_extents.z],
            pad_h: 0.0,
            world: [query.world.x, query.world.y, query.world.z],
            pad_w: 0.0,
            surface_normal: [
                query.surface_normal.x,
                query.surface_normal.y,
                query.surface_normal.z,
            ],
            pad_n: 0.0,
            fade: [
                f.depth_fade_start,
                f.depth_fade_end,
                f.cos_threshold,
                f.cos_full,
            ],
        }
    }
}

/// `repr(C)` `std430` layout of one result: a four-scalar slot `(hit, uv_u,
/// uv_v, fade)` — `16` bytes matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `Option` tag: `1` inside the box, `0` clipped.
    hit: u32,
    /// Sampled `UV` `u` component.
    uv_u: f32,
    /// Sampled `UV` `v` component.
    uv_v: f32,
    /// Combined angle times depth fade.
    fade: f32,
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

/// A compiled, reusable decal-projection compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::decal`；
/// no third-party engine source or derived code.
pub struct GpuDecal {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDecal {
    /// Compiles the decal-projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::decal`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDecal {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_decal"),
            source: ShaderSource::Wgsl(DECAL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_decal_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_decal_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_decal_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDecal {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects every receiver point on-device and returns one
    /// [`DecalProjection`] per input, in order.
    ///
    /// Each result equals the reference
    /// [`projected_sample`](prism_render_architecture::particle::decal::projected_sample)
    /// answer: the `hit` tag matches exactly and, when `hit`, the `UV` and
    /// `fade` match to within the tolerance documented on this module. An empty
    /// input returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::decal`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[DecalQuery]) -> Vec<DecalProjection> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_decal_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_decal_output"),
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
            label: Some("prism_volumetric_decal_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_decal_bind_group"),
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
            label: Some("prism_volumetric_decal_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_decal_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_decal_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per receiver point, flattened to a 1-D dispatch.
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

/// Decodes one packed [`GpuResult`] into the public [`DecalProjection`], mapping
/// the `u32` tag back to the `Option` boolean with an exact integer compare.
fn decode_result(raw: &GpuResult) -> DecalProjection {
    DecalProjection {
        hit: raw.hit == CODE_HIT,
        uv: [raw.uv_u, raw.uv_v],
        fade: raw.fade,
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
