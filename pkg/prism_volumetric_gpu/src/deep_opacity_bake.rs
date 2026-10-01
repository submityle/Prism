//! `wgpu` compute twin of the `CPU` golden `deep opacity map` baker
//! ([`DeepOpacityRecorder::bake_depth_profile`](prism_render_architecture::particle::deep_opacity_bake::DeepOpacityRecorder::bake_depth_profile)).
//!
//! A `deep opacity map` records, from the light's viewpoint, a monotonically
//! non-increasing transmittance-versus-depth curve `T(depth)` along a light ray
//! and discretizes it into `N` [`DeepOpacityLayer`](prism_render_architecture::particle::volumetrics::DeepOpacityLayer)
//! samples (design §20, "Deep shadow / deep opacity maps：从光源视角记录沿光线
//! 的透过率函数"). The `CPU` golden
//! [`DeepOpacityRecorder`](prism_render_architecture::particle::deep_opacity_bake::DeepOpacityRecorder)
//! owns that recording math; [`GpuDeepOpacityBake`] is the on-device twin,
//! validated against that reference so a passing real-device parity test is
//! direct evidence the ported kernel bakes the same transmittance curve, not
//! merely that its shader compiles.
//!
//! # Algorithm
//!
//! The twin reproduces the reference exactly. Layer `0` is seeded at the near
//! plane with transmittance `1.0` (the fully-lit near side). Each later layer
//! `i` integrates the interval `[layer_depth(i − 1), layer_depth(i)]`
//! front-to-back in equal sub-steps no longer than `step_size`, accumulating
//! transmittance by the algebraic step-opacity recurrence the ray-march stage
//! already uses: per sub-step of length `sub_length`,
//!
//! - optical thickness `tau = sigma * sub_length` (extinction × step length),
//! - step opacity `alpha = clamp(tau, 0, 1)` (a first-order absorber),
//! - transmittance product `transmittance *= 1 - alpha`.
//!
//! The caller supplies the per-depth extinction samples `sigma` already
//! evaluated at the march sub-step centers that [`march_centers`] reports, so
//! the kernel consumes exactly the same `sigma` values the reference samples
//! and runs the identical recurrence in the identical order.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `ceil` and `+ − × ÷` — with no `exp`, `pow`, `log` or trig and no
//! optional device feature, so it runs unmodified on Metal, Vulkan and DX12.
//! The forbidden `Beer-Lambert` `exp` is replaced by the same algebraic
//! increment product `transmittance *= 1 - alpha` the `CPU` reference uses.
//!
//! # Correctness model
//!
//! The recurrence contains no transcendental call, so `CPU` and `GPU` evaluate
//! the same closed-form algebra in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few `ULP`. The parity test therefore
//! asserts a tolerance (`abs_diff < 1e-6` or `rel_diff < 1e-5`), tight enough
//! to catch a genuinely wrong port yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Lokovic-Veach / Yuksel-Keyser deep-opacity recording
//! plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::deep_opacity_bake::DeepOpacityRecorder;
use prism_render_architecture::particle::volumetrics::DeepOpacityLayer;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` baker kernel, embedded inline so the twin ships as
/// a single source file. Mirrors the `CPU` recurrence exactly; see the module
/// documentation for the algorithm.
const DEEP_OPACITY_BAKE_WGSL: &str = r#"
// Deep-opacity baker twin: one thread per profile walks the light ray
// front-to-back, accumulating transmittance by the algebraic step-opacity
// recurrence `transmittance *= 1 - clamp(sigma * sub_length, 0, 1)` and writing
// `layer_count` `(depth, transmittance)` layers. It mirrors the CPU golden
// `DeepOpacityRecorder::bake_depth_profile`, uses only the portable core-WGSL
// subset (min/max/clamp/ceil and + - * /), and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Lokovic-Veach / Yuksel-Keyser deep-opacity recording; no
// Unreal Engine source or derived code.

struct Params {
    // Number of recorded layers N (at least 2).
    layer_count: u32,
    // Number of profiles baked in this dispatch (one thread each).
    profile_count: u32,
    // Extinction samples per profile, equal to the sum over layers of their
    // sub-step counts (the `march_centers` length).
    samples_per_profile: u32,
    pad0: u32,
    // Depth of layer 0 along the light ray.
    near_plane: f32,
    // Depth of the last layer along the light ray.
    far_plane: f32,
    // Maximum march sub-step length between adjacent layers.
    step_size: f32,
    // Fixed depth span between two adjacent layers (`layer_span`).
    span: f32,
}

// One recorded layer. 16-byte std430 stride: the pair plus two pad words.
struct Layer {
    depth: f32,
    transmittance: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> samples: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<Layer>;

// Clamps `x` into [0, 1], mirroring the reference's first-order absorber.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let profile = gid.x;
    if (profile >= params.profile_count) {
        return;
    }

    let count = params.layer_count;
    let near = params.near_plane;
    let span = params.span;
    let step = params.step_size;

    // Layer 0: the fully-lit near side; nothing has absorbed yet.
    var transmittance = 1.0;
    let base = profile * count;
    var seed: Layer;
    seed.depth = near;
    seed.transmittance = transmittance;
    seed.pad0 = 0.0;
    seed.pad1 = 0.0;
    results[base] = seed;

    // Read extinction samples sequentially in the same march order the host
    // `march_centers` lays them out.
    var cursor = profile * params.samples_per_profile;
    var layer = 1u;
    while (layer < count) {
        let depth_lo = near + f32(layer - 1u) * span;
        let depth_hi = near + f32(layer) * span;
        let interval = depth_hi - depth_lo;
        // At least one sub-step; cap the sub-step length at `step_size`.
        let sub_steps = u32(max(ceil(interval / step), 1.0));
        let sub_length = interval / f32(sub_steps);

        var sub = 0u;
        while (sub < sub_steps) {
            // Negative samples floor to zero: a non-negative absorber.
            let sigma = max(samples[cursor], 0.0);
            let tau = sigma * sub_length;
            let alpha = clamp01(tau);
            transmittance = transmittance * (1.0 - alpha);
            cursor = cursor + 1u;
            sub = sub + 1u;
        }

        var out: Layer;
        out.depth = depth_hi;
        out.transmittance = transmittance;
        out.pad0 = 0.0;
        out.pad1 = 0.0;
        results[base + layer] = out;
        layer = layer + 1u;
    }
}
"#;

/// Uniform parameters for one baker dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`DEEP_OPACITY_BAKE_WGSL`]: three index words plus one pad word,
/// followed by the four `f32` geometry words — `32` bytes total with no
/// interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of recorded layers `N`.
    layer_count: u32,
    /// Number of profiles baked in this dispatch.
    profile_count: u32,
    /// Extinction samples per profile (the `march_centers` length).
    samples_per_profile: u32,
    /// Padding to keep the following `f32` block `16`-byte aligned.
    pad0: u32,
    /// Depth of layer `0`.
    near_plane: f32,
    /// Depth of the last layer.
    far_plane: f32,
    /// Maximum march sub-step length.
    step_size: f32,
    /// Fixed depth span between adjacent layers.
    span: f32,
}

/// One recorded layer as read back. `16`-byte `std430` stride matching `Layer` in
/// the shader: the `(depth, transmittance)` pair plus two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLayer {
    /// Distance along the light ray at which this layer was recorded.
    depth: f32,
    /// Surviving transmittance at `depth`, in `0..=1`.
    transmittance: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// The depth of every march sub-step center a bake samples, in march order.
///
/// Mirrors the private sampling loop of
/// [`DeepOpacityRecorder::bake_depth_profile`](prism_render_architecture::particle::deep_opacity_bake::DeepOpacityRecorder::bake_depth_profile):
/// for each layer `1..layer_count`, the interval
/// `[layer_depth(i − 1), layer_depth(i)]` is split into equal sub-steps no
/// longer than `step_size` and sampled at each sub-step center (midpoint rule).
/// A caller evaluates its extinction function at exactly these depths to build
/// the per-profile sample slice [`GpuDeepOpacityBake::bake`] consumes, so the
/// kernel reads the identical `sigma` values the reference would and the two
/// run the identical recurrence.
///
/// The returned length is the required samples-per-profile count.
#[must_use]
pub fn march_centers(recorder: &DeepOpacityRecorder) -> Vec<f32> {
    let near = recorder.near_plane();
    let span = recorder.layer_span();
    let step = recorder.step_size();
    let mut centers = Vec::new();
    for layer in 1..recorder.layer_count() {
        let depth_lo = near + ((layer - 1) as f32) * span;
        let depth_hi = near + (layer as f32) * span;
        let interval = depth_hi - depth_lo;
        // At least one sub-step; cap the sub-step length at `step_size`. The
        // `as u32` cast saturates for pathological inputs, staying finite.
        let sub_steps = (interval / step).ceil().max(1.0) as u32;
        let sub_length = interval / (sub_steps as f32);
        for sub in 0..sub_steps {
            // Midpoint rule: sample at the center of each sub-step. The literal
            // `0.5` is the half-step offset, not a tunable magic number.
            centers.push(depth_lo + ((sub as f32) + 0.5) * sub_length);
        }
    }
    centers
}

/// A compiled, reusable `deep opacity map` baker pipeline.
pub struct GpuDeepOpacityBake {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDeepOpacityBake {
    /// Compiles the baker kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDeepOpacityBake {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake"),
            source: ShaderSource::Wgsl(DEEP_OPACITY_BAKE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bake"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDeepOpacityBake {
            module,
            layout,
            pipeline,
        }
    }

    /// Bakes a layered transmittance curve for every profile against the shared
    /// `recorder` configuration, returning one `layer_count`-long
    /// [`DeepOpacityLayer`](prism_render_architecture::particle::volumetrics::DeepOpacityLayer)
    /// curve per profile in input order.
    ///
    /// Each entry of `profiles` is the extinction samples `sigma` for one bake,
    /// evaluated at the march sub-step centers [`march_centers`] reports, so
    /// every profile must have exactly `march_centers(recorder).len()` samples.
    /// The returned curve for profile `p` equals
    /// [`DeepOpacityRecorder::bake_depth_profile`](prism_render_architecture::particle::deep_opacity_bake::DeepOpacityRecorder::bake_depth_profile)
    /// evaluated on an extinction function returning `profiles[p]` at those
    /// centers, to within the tolerance documented on this module. An empty
    /// `profiles` slice yields an empty result — storage buffers cannot be
    /// zero-sized, so it is handled by an early return.
    ///
    /// # Panics
    ///
    /// Panics if any profile's sample count differs from
    /// `march_centers(recorder).len()`, which would misalign the kernel's
    /// per-profile sample window.
    #[must_use]
    pub fn bake(
        &self,
        ctx: &GpuContext,
        recorder: &DeepOpacityRecorder,
        profiles: &[Vec<f32>],
    ) -> Vec<Vec<DeepOpacityLayer>> {
        if profiles.is_empty() {
            return Vec::new();
        }

        let centers = march_centers(recorder);
        let samples_per_profile = centers.len();
        let layer_count = recorder.layer_count() as usize;

        let mut flat = Vec::with_capacity(profiles.len() * samples_per_profile);
        for profile in profiles {
            assert_eq!(
                profile.len(),
                samples_per_profile,
                "each profile must carry exactly `march_centers(recorder).len()` extinction samples"
            );
            flat.extend_from_slice(profile);
        }

        let device = ctx.device();

        let gpu_params = Params {
            layer_count: recorder.layer_count(),
            profile_count: profiles.len() as u32,
            samples_per_profile: samples_per_profile as u32,
            pad0: 0,
            near_plane: recorder.near_plane(),
            far_plane: recorder.far_plane(),
            step_size: recorder.step_size(),
            span: recorder.layer_span(),
        };

        let out_len = profiles.len() * layer_count;
        let out_bytes = (out_len as u64) * (size_of::<GpuLayer>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let samples_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake_samples"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: samples_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_deep_opacity_bake_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_deep_opacity_bake_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per profile, in workgroups of 64 (the kernel's size).
            let groups = (profiles.len() as u32).div_ceil(64);
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
        let gpu_layers = bytemuck::cast_slice::<u8, GpuLayer>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_layers.len(), out_len);

        gpu_layers
            .chunks_exact(layer_count)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|layer| DeepOpacityLayer::new(layer.depth, layer.transmittance))
                    .collect()
            })
            .collect()
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
