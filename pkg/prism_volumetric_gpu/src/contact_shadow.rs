//! `wgpu` compute twin of the screen-space contact-shadow (`SSCS`) short-range
//! ray-march
//! ([`contact_shadow_occlusion`](prism_render_architecture::particle::contact_shadow::contact_shadow_occlusion),
//! design sections 16-21).
//!
//! A translucent sprite floating just above a surface leaves a tell-tale gap
//! unless the few pixels where it nearly touches are darkened. Cascaded and
//! distance-field shadow maps are too coarse for that contact region, so
//! production engines add a cheap screen-space pass: from the shaded point they
//! step a handful of samples along the light direction, read the already-shaded
//! scene depth at each step, and darken the point when the ray dips *behind* a
//! surface that sits between it and the camera. The whole march lives in
//! `view`-space depth (larger means farther from the camera), so this `GPU`
//! kernel reproduces the reference march pixel for pixel.
//!
//! The `CPU` golden
//! [`particle::contact_shadow`](prism_render_architecture::particle::contact_shadow)
//! owns that march; [`GpuContactShadow`] is the on-device twin that runs one
//! thread per shaded pixel and returns the same `0..=1` lighting factor
//! (`1.0` = fully lit / unoccluded, `0.0` = fully shadowed). A passing
//! real-device parity test is therefore direct evidence the ported kernel
//! marches the same positions in the same order the reference does, not merely
//! that its shader compiles.
//!
//! # What is twinned
//!
//! The full reference pipeline is reproduced: (1) the self-contained integer
//! avalanche hash turning a per-pixel seed into a sub-step jitter offset
//! ([`jitter_offset`](prism_render_architecture::particle::contact_shadow::jitter_offset)),
//! (2) the jittered normalized march positions
//! ([`march_fractions`](prism_render_architecture::particle::contact_shadow::march_fractions)),
//! and (3) the per-step acceptance-window test folded into a rational distance
//! falloff plus a `smoothstep` soft edge
//! ([`contact_shadow_occlusion`](prism_render_architecture::particle::contact_shadow::contact_shadow_occlusion)).
//! The reproduced jitter hash is a real feature of the pass, not a general-
//! purpose codec, and it is reproduced bit for bit (integer avalanche plus an
//! exact `2^-32` scale), so the jittered positions match the reference exactly.
//!
//! # Step-for-step parity
//!
//! The kernel mirrors the reference exactly: the same `hash_u32` avalanche, the
//! same `jitter / 2^32` unit scale, the same `(index + jitter) / step_count`
//! fraction, the same `ray_z = start_depth + ray_depth_span * frac`, the same
//! half-open `bias <= diff < bias + thickness` acceptance window, the same
//! first-hit-wins search, the same `dist = frac * max_distance`, the same
//! rational falloff `1 / (1 + falloff * dist * dist)`, the same
//! `1 - smoothstep(0, 1, frac)` soft edge, the same `clamp01(intensity *
//! falloff * edge)` shadow strength and the same `clamp01(1 - shadow)` lighting
//! factor. An empty schedule (`step_count == 0`) or a non-positive `thickness`
//! (empty window) leaves the point fully lit, exactly as the reference does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `+ - * /` and unsigned integer arithmetic / bit shifts —
//! with no `sin`, `cos`, `exp`, `log`, `pow` or optional device feature, so it
//! runs unmodified on Metal, Vulkan and DX12. There is no transcendental call
//! at all (not even `sqrt`): the march is pure comparisons, multiplies and one
//! guarded reciprocal whose denominator `1 + falloff * dist * dist` is at least
//! `1`, so no divide ever hits a vanishing denominator.
//!
//! # Correctness model
//!
//! The jitter hash is exact integer arithmetic plus an exact power-of-two
//! scale, so the per-pixel jitter offset and the jittered fractions are
//! bit-identical on `CPU` and `GPU`. The downstream occlusion folds those
//! fractions through a short closed-form expression; it is not guaranteed
//! bit-exact because a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate (the `start_depth + ray_depth_span * frac` step, the falloff
//! denominator, the `smoothstep` cubic), perturbing the low mantissa bits by a
//! few units in the last place. The parity test asserts an absolute tolerance
//! tight enough to catch a genuinely wrong port (a swapped window bound, a
//! dropped first-hit break, a wrong falloff or soft edge) yet loose enough to
//! admit that legal fused multiply-add contraction, and additionally checks the
//! saturated hard-shadow case bit for bit (where the result collapses to an
//! exact `0.0` or `1.0`) to pin the jitter-gated hit/miss boundary.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard screen-space contact-shadow (`SSCS`) short-range depth
//! ray-march plus `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::contact_shadow::{
    ContactShadowParams, CONTACT_SHADOW_PARAMS_STRIDE,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` contact-shadow kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` march hash-for-hash and
/// step-for-step; see the module documentation for the algorithm.
///
/// The `Params` struct lays the six packed [`ContactShadowParams`] scalars out
/// exactly as
/// [`ContactShadowParams::to_std430`](prism_render_architecture::particle::contact_shadow::ContactShadowParams::to_std430)
/// does (two `vec4` slots, [`CONTACT_SHADOW_PARAMS_STRIDE`] bytes), reusing the
/// first trailing pad word to carry the per-dispatch pixel count.
const CONTACT_SHADOW_WGSL: &str = r#"
// Contact-shadow twin: one thread per shaded pixel marches the jittered,
// view-space depth samples and writes the 0..=1 lighting factor (1.0 = lit,
// 0.0 = shadowed). It mirrors the CPU golden
// `particle::contact_shadow::contact_shadow_occlusion` and reproduces the same
// integer jitter hash bit for bit, uses only the portable core-WGSL subset
// (min/max/clamp and + - * / plus integer bit ops), and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard screen-space contact-shadow (SSCS) depth ray-march; no
// Unreal Engine source or derived code.

struct Params {
    // Number of samples taken along the light ray (a fixed short march).
    step_count: u32,
    // View-space distance the march covers; scales the distance falloff.
    max_distance: f32,
    // Width of the acceptance window above `bias`.
    thickness: f32,
    // Minimum depth gap that counts as an occluder (skips self-contact acne).
    bias: f32,
    // Scales how strongly a contact hit darkens the shaded point.
    intensity: f32,
    // Rational distance-falloff coefficient; larger fades distant contacts.
    falloff: f32,
    // Pixel (shaded-point) count, one thread each. Reuses the first pad word of
    // the shared `std430` params layout.
    pixel_count: u32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
}

// One shaded pixel. 32-byte std430 stride matching the host `GpuQuery`.
struct Query {
    // View-space depth of the shaded point.
    start_depth: f32,
    // Signed view-space depth change accumulated over the full march.
    ray_depth_span: f32,
    // Per-pixel seed for the sub-step jitter hash.
    jitter_seed: u32,
    // Start index of this pixel's scene-depth run in the flat depth buffer.
    depth_offset: u32,
    // Number of scene-depth samples supplied for this pixel.
    depth_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read> scene_depths: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

// 2^32 as an f32: the normalizing span turning a u32 hash word into a unit
// fraction, matching the reference `U32_SPAN`.
const U32_SPAN: f32 = 4294967296.0;

// Integer avalanche hash mixing a u32 seed into a well-distributed u32, matching
// the reference `hash_u32`: xor-shifts and odd-constant wrapping multiplies.
// WGSL u32 multiply wraps on overflow, exactly like the reference `wrapping_mul`.
fn hash_u32(seed: u32) -> u32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// Deterministic sub-step jitter offset in [0, 1) for a per-pixel seed, matching
// the reference `jitter_offset`. The u32-to-f32 cast plus the exact 2^-32 scale
// is bit-reproducible, so the jittered positions match the reference exactly.
fn jitter_offset(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / U32_SPAN;
}

// Clamps a scalar into 0..=1, matching the reference `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Hermite smoothstep over the fixed [0, 1] interval evaluated at `x`, matching
// the reference `smoothstep(0.0, 1.0, x)`. The span is a constant 1.0 (well
// above the reference `MIN_EDGE` guard), so the degenerate-interval branch is
// never taken and the body reduces to the standard cubic.
fn smoothstep01(x: f32) -> f32 {
    let t = clamp01(x);
    return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(64)
fn march_contact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_index = gid.x;
    if (pixel_index >= params.pixel_count) {
        return;
    }
    let query = queries[pixel_index];

    // Acceptance window `(bias .. bias + thickness)`. A non-positive thickness
    // makes `hi <= lo`, so no gap can land inside and the point stays lit.
    let lo = params.bias;
    let hi = params.bias + params.thickness;

    var factor = 1.0;
    // A zero step count yields an empty schedule: the point stays fully lit.
    if (params.step_count != 0u) {
        let jitter = jitter_offset(query.jitter_seed);
        let inv_steps = 1.0 / f32(params.step_count);
        // The reference zips the fixed schedule against the supplied depths, so
        // the effective step count is the shorter of the two.
        var steps = params.step_count;
        if (query.depth_count < steps) {
            steps = query.depth_count;
        }
        for (var k = 0u; k < steps; k = k + 1u) {
            let frac = (f32(k) + jitter) * inv_steps;
            let scene_z = scene_depths[query.depth_offset + k];
            let ray_z = query.start_depth + query.ray_depth_span * frac;
            let diff = ray_z - scene_z;
            // Half-open window `[lo, hi)`, matching `(lo..hi).contains(&diff)`.
            if (diff >= lo && diff < hi) {
                let dist = frac * params.max_distance;
                // Guarded reciprocal: the denominator is at least 1.0.
                let fall = 1.0 / (1.0 + params.falloff * dist * dist);
                let edge = 1.0 - smoothstep01(frac);
                let shadow = clamp01(params.intensity * fall * edge);
                factor = clamp01(1.0 - shadow);
                // First hit wins, matching the reference `find_map`.
                break;
            }
        }
    }
    results[pixel_index] = factor;
}
"#;

/// One shaded pixel's contact-shadow query: the twin of a single
/// [`contact_shadow_occlusion`](prism_render_architecture::particle::contact_shadow::contact_shadow_occlusion)
/// call.
///
/// `scene_depths` are the `view`-space scene depths fetched at each march step,
/// in march order. The reference zips its fixed schedule against this slice, so
/// the effective step count is `min(params.step_count, scene_depths.len())`;
/// a slice shorter than `step_count` simply stops the march early, exactly as
/// the reference does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactShadowQuery<'depths> {
    /// `view`-space depth of the shaded point (larger means farther).
    pub start_depth: f32,
    /// Signed `view`-space depth change accumulated over the full march toward
    /// the light.
    pub ray_depth_span: f32,
    /// Per-pixel seed for the sub-step jitter hash.
    pub jitter_seed: u32,
    /// `view`-space scene depths fetched at each march step, in march order.
    pub scene_depths: &'depths [f32],
}

/// Uniform parameters for one contact-shadow dispatch. `repr(C)` `std430`
/// layout matching `Params` in [`CONTACT_SHADOW_WGSL`]: the six packed
/// [`ContactShadowParams`] scalars in
/// [`ContactShadowParams::to_std430`](prism_render_architecture::particle::contact_shadow::ContactShadowParams::to_std430)
/// order, then the pixel count (reusing the first pad word) and one pad word —
/// [`CONTACT_SHADOW_PARAMS_STRIDE`] (`32`) bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    step_count: u32,
    max_distance: f32,
    thickness: f32,
    bias: f32,
    intensity: f32,
    falloff: f32,
    pixel_count: u32,
    pad0: u32,
}

/// One shaded pixel as uploaded. `32`-byte `std430` stride matching `Query` in
/// the shader: the two depth scalars, the jitter seed, the depth run offset and
/// count, then three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    start_depth: f32,
    ray_depth_span: f32,
    jitter_seed: u32,
    depth_offset: u32,
    depth_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable screen-space contact-shadow pipeline.
pub struct GpuContactShadow {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuContactShadow {
    /// Compiles the contact-shadow march kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuContactShadow {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_contact_shadow"),
            source: ShaderSource::Wgsl(CONTACT_SHADOW_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_contact_shadow_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_contact_shadow_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_contact_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("march_contact"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuContactShadow {
            module,
            layout,
            pipeline,
        }
    }

    /// Marches every pixel in `queries` under the shared `params`, returning one
    /// `0..=1` lighting factor per pixel in input order.
    ///
    /// The returned factor for pixel `q` equals
    /// [`contact_shadow_occlusion`](prism_render_architecture::particle::contact_shadow::contact_shadow_occlusion)`(params, q.start_depth, q.ray_depth_span, q.scene_depths, q.jitter_seed)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        params: &ContactShadowParams,
        queries: &[ContactShadowQuery<'_>],
    ) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        // Flatten the per-pixel depth runs into one contiguous buffer, recording
        // each pixel's run offset and length so the kernel reads the same depths
        // the reference zips against.
        let mut flat_depths: Vec<f32> = Vec::new();
        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| {
                let depth_offset = flat_depths.len() as u32;
                flat_depths.extend_from_slice(q.scene_depths);
                GpuQuery {
                    start_depth: q.start_depth,
                    ray_depth_span: q.ray_depth_span,
                    jitter_seed: q.jitter_seed,
                    depth_offset,
                    depth_count: q.scene_depths.len() as u32,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                }
            })
            .collect();

        // Storage buffers cannot be zero-sized; a batch whose pixels supply no
        // depths at all still needs one element to bind.
        if flat_depths.is_empty() {
            flat_depths.push(0.0);
        }

        let gpu_params = GpuParams {
            step_count: params.step_count,
            max_distance: params.max_distance,
            thickness: params.thickness,
            bias: params.bias,
            intensity: params.intensity,
            falloff: params.falloff,
            pixel_count: queries.len() as u32,
            pad0: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_contact_shadow_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_contact_shadow_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let depths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_contact_shadow_depths"),
            contents: bytemuck::cast_slice(&flat_depths),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_contact_shadow_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_contact_shadow_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_contact_shadow_bind_group"),
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
                    resource: depths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_contact_shadow_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_contact_shadow_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel, in workgroups of 64 (the kernel's size).
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let factors = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(factors.len(), queries.len());
        debug_assert_eq!(CONTACT_SHADOW_PARAMS_STRIDE, size_of::<GpuParams>());
        factors
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
