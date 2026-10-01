//! `wgpu` compute twin of the soft-particle depth-fade contract
//! ([`soft_particle`](prism_render_architecture::particle::soft_particle),
//! design section 14, "soft particles").
//!
//! A translucent sprite rasterized against solid geometry otherwise cuts a hard
//! seam along the intersection line; the soft-particle fade replaces that binary
//! depth-test cut with a smooth ramp driven by the gap between the already
//! rendered scene depth and the particle fragment's own `view`-space depth, and
//! a second ramp that dissolves particles crowding the near plane. The `CPU`
//! golden
//! [`SoftParticleParams::combined_fade`](prism_render_architecture::particle::soft_particle::SoftParticleParams::combined_fade)
//! owns that math; [`GpuSoftParticle`] is the on-device twin that runs one
//! thread per sample and reproduces the same opacity multiplier, so a passing
//! real-device parity test is direct evidence the ported kernel composes the
//! same seam and near fades the reference does, not merely that its shader
//! compiles.
//!
//! # What is twinned
//!
//! Two entry points mirror the golden module function for function:
//!
//! 1. `combined_fade_main` composes the seam fade
//!    [`contact_fade`](prism_render_architecture::particle::soft_particle::contact_fade)`(scene_depth - particle_depth, fade_distance)`
//!    with the near-plane fade
//!    [`NearFade::camera_proximity_fade`](prism_render_architecture::particle::soft_particle::NearFade::camera_proximity_fade)`(view_depth)`
//!    and multiplies them in the reference's order (`seam * near`). The golden
//!    [`combined_fade`](prism_render_architecture::particle::soft_particle::SoftParticleParams::combined_fade)
//!    evaluates the near fade at the particle's own depth; the twin accepts a
//!    separate `view_depth` so the kernel generalizes cleanly, and feeding
//!    `view_depth == particle_depth` reproduces the golden `combined_fade`
//!    exactly (the parity suite pins both relations).
//! 2. `linearize_main` reconstructs linear `view`-space depth from `NDC` depth
//!    via
//!    [`linearize_depth`](prism_render_architecture::particle::soft_particle::linearize_depth)`(ndc_depth, near, far)`.
//!
//! Both the degenerate guards are mirrored bit for bit: a non-positive
//! `fade_distance` disables the seam fade and returns `1`, a near-plane band
//! narrower than [`EPS`](prism_render_architecture::particle::soft_particle::EPS)
//! collapses to a hard step instead of dividing by a vanishing width, and a
//! frustum whose `near` and `far` are within `EPS` returns `near` rather than
//! dividing by a vanishing range.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `clamp`, `abs` and
//! `+ - * /` — with no `exp`, `pow`, `sqrt` or optional device feature, so they
//! run unmodified on `Metal`, `Vulkan` and `DX12`. There is no transcendental
//! call on this path.
//!
//! # Correctness model
//!
//! Each fade is a fixed, non-reorderable sequence of a subtract, one divide and
//! a `clamp` (or a guarded branch), so `CPU` and `GPU` evaluate the same
//! closed-form algebra. They are not bit-exact: a `GPU` may fuse a multiply-add
//! the scalar reference leaves separate, perturbing the low mantissa bits by a
//! few units in the last place. The parity test therefore asserts
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, tight enough to fail a genuinely
//! wrong port (a dropped guard, a swapped seam sign, a missing clamp) yet loose
//! enough to admit a legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard soft-particle depth fade (design section 14) plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::soft_particle::SoftParticleParams;
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

/// The portable core-`WGSL` soft-particle combined-fade kernel, embedded inline.
/// The entry point `combined_fade_main` mirrors the `CPU` golden
/// [`combined_fade`](prism_render_architecture::particle::soft_particle::SoftParticleParams::combined_fade)
/// function for function; see the module documentation for the algorithm.
const COMBINED_FADE_WGSL: &str = r#"
// Soft-particle depth-fade twin: one thread per sample composes the seam fade
// (from the scene-minus-particle depth gap) with the near-plane proximity fade,
// mirroring the CPU golden `particle::soft_particle`. Uses only the portable
// core-WGSL subset (clamp/abs and + - * /) and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12. There is no transcendental call.
//
// Provenance: standard soft-particle depth fade (design section 14); no
// third-party engine source or derived code.

// Degenerate-interval guard matching `EPS` in the CPU `soft_particle` module.
const EPS: f32 = 1.0e-6;

// Combined-fade uniform: the shared soft-particle parameters plus the sample
// count. Matches `FadeParams` in `src/soft_particle.rs`.
struct FadeParams {
    fade_distance: f32,
    near_start: f32,
    near_end: f32,
    count: u32,
}

// One combined-fade sample: the opaque scene depth behind the fragment, the
// particle fragment's own depth, and the view-space depth the near fade reads.
// Matches `GpuSample` in `src/soft_particle.rs`.
struct Sample {
    scene_depth: f32,
    particle_depth: f32,
    view_depth: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: FadeParams;
@group(0) @binding(1) var<storage, read> samples: array<Sample>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// The soft-particle seam fade on the pre-computed gap, mirroring the CPU golden
// `contact_fade`. A non-positive `fade_distance` disables the fade and returns
// 1; otherwise the gap is scaled by the band and clamped into [0, 1].
fn contact_fade(depth_diff: f32, fade_distance: f32) -> f32 {
    if (fade_distance <= 0.0) {
        return 1.0;
    }
    return clamp(depth_diff / fade_distance, 0.0, 1.0);
}

// The near-plane proximity fade, mirroring the CPU golden
// `NearFade::camera_proximity_fade`. A band narrower than EPS collapses to a
// hard step (0 before `near_end`, 1 at or beyond it) rather than dividing by a
// vanishing width.
fn camera_proximity_fade(view_depth: f32, near_start: f32, near_end: f32) -> f32 {
    let width = near_end - near_start;
    if (abs(width) < EPS) {
        if (view_depth < near_end) {
            return 0.0;
        }
        return 1.0;
    }
    return clamp((view_depth - near_start) / width, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn combined_fade_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let s = samples[idx];
    // Reference order: the seam fade, then the near fade, then their product.
    let seam = contact_fade(s.scene_depth - s.particle_depth, params.fade_distance);
    let near = camera_proximity_fade(s.view_depth, params.near_start, params.near_end);
    results[idx] = seam * near;
}
"#;

/// The portable core-`WGSL` depth-linearization kernel, embedded inline. The
/// entry point `linearize_main` mirrors the `CPU` golden
/// [`linearize_depth`](prism_render_architecture::particle::soft_particle::linearize_depth).
const LINEARIZE_WGSL: &str = r#"
// Depth-linearization twin: one thread per NDC depth reconstructs linear
// view-space depth, mirroring the CPU golden `linearize_depth`. Uses only the
// portable core-WGSL subset (abs and + - * /) and takes no optional feature.
//
// Provenance: standard perspective depth linearization (design section 14); no
// third-party engine source or derived code.

// Degenerate-range guard matching `EPS` in the CPU `soft_particle` module.
const EPS: f32 = 1.0e-6;

// Linearize uniform: the frustum near/far planes plus the NDC-depth count.
// Matches `LinearizeParams` in `src/soft_particle.rs`.
struct LinearizeParams {
    near: f32,
    far: f32,
    count: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: LinearizeParams;
@group(0) @binding(1) var<storage, read> ndc_depths: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Reconstructs linear view-space depth from NDC depth, mirroring the CPU golden
// `linearize_depth`. A frustum whose near and far are within EPS returns `near`
// rather than dividing by a vanishing range.
fn linearize_depth(ndc_depth: f32, near: f32, far: f32) -> f32 {
    let range = far - near;
    if (abs(range) < EPS) {
        return near;
    }
    return (near * far) / (far - ndc_depth * range);
}

@compute @workgroup_size(64)
fn linearize_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    results[idx] = linearize_depth(ndc_depths[idx], params.near, params.far);
}
"#;

/// One combined-fade sample: the opaque scene depth behind the fragment, the
/// particle fragment's own `view`-space depth, and the `view`-space depth the
/// near-plane fade reads.
///
/// All three are linear `view`-space depths where larger means farther. Feeding
/// `view_depth == particle_depth` reproduces the golden
/// [`combined_fade`](prism_render_architecture::particle::soft_particle::SoftParticleParams::combined_fade)
/// exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftParticleSample {
    /// Opaque scene depth already in the depth buffer behind this fragment.
    pub scene_depth: f32,
    /// The particle fragment's own `view`-space depth.
    pub particle_depth: f32,
    /// The `view`-space depth the near-plane proximity fade is evaluated at.
    pub view_depth: f32,
}

/// One combined-fade request: the shared [`SoftParticleParams`] and the batch of
/// per-fragment [`SoftParticleSample`]s to evaluate.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftParticleQuery {
    /// The seam-band width and near-plane fade window shared by every sample.
    pub params: SoftParticleParams,
    /// The per-fragment depth samples; one fade value is returned per sample.
    pub samples: Vec<SoftParticleSample>,
}

/// One depth-linearization request: the frustum `near`/`far` planes and the
/// batch of `NDC` depths to reconstruct into `view`-space depth.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearizeQuery {
    /// Frustum near-plane distance (the `view`-space depth at `ndc_depth == 0`).
    pub near: f32,
    /// Frustum far-plane distance (the `view`-space depth at `ndc_depth == 1`).
    pub far: f32,
    /// The `NDC` depths in `[0, 1]`; one linear depth is returned per entry.
    pub ndc_depths: Vec<f32>,
}

/// One combined-fade sample as uploaded. `16`-byte `repr(C)` matching `Sample`
/// in [`COMBINED_FADE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    /// Opaque scene depth behind the fragment.
    scene_depth: f32,
    /// The particle fragment's own `view`-space depth.
    particle_depth: f32,
    /// The `view`-space depth the near fade reads.
    view_depth: f32,
    /// Padding to a `16`-byte, `std430`-friendly stride.
    pad0: f32,
}

/// Combined-fade uniform parameters for one dispatch. `16`-byte `repr(C)`
/// matching `FadeParams` in [`COMBINED_FADE_WGSL`]: the shared soft-particle
/// parameters plus the sample count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FadeParams {
    /// Seam-fade band width.
    fade_distance: f32,
    /// Near-plane fade start (fully faded at or below this `view`-space depth).
    near_start: f32,
    /// Near-plane fade end (fully visible at or beyond this `view`-space depth).
    near_end: f32,
    /// Number of samples in the dispatch.
    count: u32,
}

/// Linearize uniform parameters for one dispatch. `16`-byte `repr(C)` matching
/// `LinearizeParams` in [`LINEARIZE_WGSL`]: the frustum planes plus the
/// `NDC`-depth count and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearizeParams {
    /// Frustum near-plane distance.
    near: f32,
    /// Frustum far-plane distance.
    far: f32,
    /// Number of `NDC` depths in the dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
}

/// A compiled, reusable soft-particle pipeline pair (combined fade and depth
/// linearization).
pub struct GpuSoftParticle {
    #[expect(
        dead_code,
        reason = "kept alive so the combined-fade pipeline it produced stays valid"
    )]
    module_fade: ShaderModule,
    #[expect(
        dead_code,
        reason = "kept alive so the linearize pipeline it produced stays valid"
    )]
    module_linearize: ShaderModule,
    layout: BindGroupLayout,
    pipeline_fade: ComputePipeline,
    pipeline_linearize: ComputePipeline,
}

impl GpuSoftParticle {
    /// Compiles the combined-fade and linearize kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftParticle {
        let device = ctx.device();
        let module_fade = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_particle_combined_fade"),
            source: ShaderSource::Wgsl(COMBINED_FADE_WGSL.into()),
        });
        let module_linearize = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_particle_linearize"),
            source: ShaderSource::Wgsl(LINEARIZE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_particle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_particle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline_fade = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_particle_combined_fade_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_fade,
            entry_point: Some("combined_fade_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_linearize = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_particle_linearize_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_linearize,
            entry_point: Some("linearize_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftParticle {
            module_fade,
            module_linearize,
            layout,
            pipeline_fade,
            pipeline_linearize,
        }
    }

    /// Evaluates the combined soft-particle fade for every sample in
    /// `query.samples`, returning one opacity multiplier per sample in input
    /// order.
    ///
    /// The returned value for sample `s` equals
    /// [`contact_fade`](prism_render_architecture::particle::soft_particle::contact_fade)`(s.scene_depth - s.particle_depth, params.fade_distance)`
    /// times
    /// [`camera_proximity_fade`](prism_render_architecture::particle::soft_particle::NearFade::camera_proximity_fade)`(s.view_depth)`,
    /// to within the tolerance documented on this module; with
    /// `s.view_depth == s.particle_depth` that is exactly the golden
    /// [`combined_fade`](prism_render_architecture::particle::soft_particle::SoftParticleParams::combined_fade).
    /// An empty sample batch yields an empty result with no dispatch issued —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &SoftParticleQuery) -> Vec<f32> {
        if query.samples.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_samples: Vec<GpuSample> = query
            .samples
            .iter()
            .map(|s| GpuSample {
                scene_depth: s.scene_depth,
                particle_depth: s.particle_depth,
                view_depth: s.view_depth,
                pad0: 0.0,
            })
            .collect();
        let gpu_params = FadeParams {
            fade_distance: query.params.fade_distance,
            near_start: query.params.near_start,
            near_end: query.params.near_end,
            count: query.samples.len() as u32,
        };

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_particle_fade_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_particle_samples"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: BufferUsages::STORAGE,
        });

        self.dispatch(
            ctx,
            &self.pipeline_fade,
            &params_buf,
            &input_buf,
            query.samples.len(),
        )
    }

    /// Reconstructs linear `view`-space depth for every `NDC` depth in
    /// `query.ndc_depths`, returning one depth per entry in input order.
    ///
    /// The returned value for entry `d` equals
    /// [`linearize_depth`](prism_render_architecture::particle::soft_particle::linearize_depth)`(d, query.near, query.far)`
    /// to within the tolerance documented on this module. An empty batch yields
    /// an empty result with no dispatch issued.
    #[must_use]
    pub fn eval_linearize(&self, ctx: &GpuContext, query: &LinearizeQuery) -> Vec<f32> {
        if query.ndc_depths.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = LinearizeParams {
            near: query.near,
            far: query.far,
            count: query.ndc_depths.len() as u32,
            pad0: 0,
        };

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_particle_linearize_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_particle_ndc_depths"),
            contents: bytemuck::cast_slice(&query.ndc_depths),
            usage: BufferUsages::STORAGE,
        });

        self.dispatch(
            ctx,
            &self.pipeline_linearize,
            &params_buf,
            &input_buf,
            query.ndc_depths.len(),
        )
    }

    /// Binds the uniform and input buffers, runs `pipeline` with one thread per
    /// element and reads the `count` `f32` results back.
    ///
    /// Shared by [`GpuSoftParticle::eval`] and
    /// [`GpuSoftParticle::eval_linearize`]: both bind a uniform, a read-only
    /// input storage buffer and a writable `f32` output buffer, differing only
    /// in the kernel and the element layout the caller already packed.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        params_buf: &wgpu::Buffer,
        input_buf: &wgpu::Buffer,
        count: usize,
    ) -> Vec<f32> {
        let device = ctx.device();
        let out_bytes = (count as u64) * (size_of::<f32>() as u64);

        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_particle_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_particle_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_particle_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_particle_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_particle_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
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
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(values.len(), count);
        values
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
