//! `wgpu` compute twin of the Variance-Shadow-Map (`VSM`) soft-shadow test
//! ([`variance_shadow`](prism_render_architecture::particle::variance_shadow),
//! design §16-§21, "矩法软影").
//!
//! A classic depth shadow map stores one occluder depth per texel and answers
//! the shadow test with a hard `receiver > occluder` compare, so it cannot be
//! pre-filtered the way a color texture can. `VSM` stores the first two
//! statistical moments of the occluder-depth distribution per texel, `E[d]`
//! and `E[d^2]`; those moments are linear and so a blur or mip fetch may
//! average them, and the shadow test becomes a probabilistic *upper bound* on
//! the fraction of occluders in front of the receiver via the one-sided
//! `Chebyshev` inequality.
//!
//! The `CPU` golden
//! [`variance_shadow`](prism_render_architecture::particle::variance_shadow)
//! owns that math; [`GpuVarianceShadow`] is the on-device twin that runs one
//! thread per receiver and returns the same per-receiver visibility the batch
//! form
//! [`resolve_visibility`](prism_render_architecture::particle::variance_shadow::resolve_visibility)
//! produces. For each receiver the kernel recovers the variance
//! `max(m2 - m1^2, 0)` from the pre-filtered moments
//! ([`variance`](prism_render_architecture::particle::variance_shadow::variance)),
//! floors it at `min_variance`, evaluates the single-sided `Chebyshev` upper
//! bound
//! ([`chebyshev_upper_bound`](prism_render_architecture::particle::variance_shadow::chebyshev_upper_bound)),
//! and remaps the result with the light-bleed `linstep`
//! ([`light_bleed_reduction`](prism_render_architecture::particle::variance_shadow::light_bleed_reduction)).
//! A passing real-device parity test is therefore direct evidence the ported
//! kernel reproduces the same visibility the reference does, not merely that
//! its shader compiles.
//!
//! # Scope
//!
//! Only the per-receiver, embarrassingly-parallel *resolve* path is ported: the
//! moments are taken as already-filtered input (the weighted-average
//! [`filter_moments`](prism_render_architecture::particle::variance_shadow::filter_moments)
//! blur that produces them is a neighbourhood reduction, not a per-receiver
//! kernel, so it stays on the host). The result equals
//! [`light_bleed_reduction`](prism_render_architecture::particle::variance_shadow::light_bleed_reduction)`(`[`chebyshev_upper_bound`](prism_render_architecture::particle::variance_shadow::chebyshev_upper_bound)`(...), bleed)`
//! lane for lane.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — multiply, add,
//! subtract, a single reciprocal divide, `min`/`max`/`clamp` and unsigned
//! compares — with no `sin`, `cos`, `exp`, `log`, `pow` or optional device
//! feature, so it runs unmodified on Metal, Vulkan and DX12. There is no
//! transcendental call at all on this path (not even `sqrt`), matching the
//! reference, and the only divide is `variance / denominator`, reached only
//! after the `MIN_DENOM` guard proves the denominator is strictly positive.
//!
//! # Correctness model
//!
//! Each lane is a fixed, non-reorderable sequence of multiplies, adds and one
//! divide, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test asserts a tolerance (`abs_diff <= 1e-5` or
//! `rel_diff <= 1e-5`) tight enough to catch a genuinely wrong port (a dropped
//! variance floor, a missing in-front early-out, a flipped light-bleed span)
//! yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard two-moment Variance Shadow Maps (`Donnelly` and
//! `Lauritzen` 2006) plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::variance_shadow::Moments;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` `VSM` resolve kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`resolve_visibility`](prism_render_architecture::particle::variance_shadow::resolve_visibility)
/// exactly; see the module documentation for the algorithm.
const VARIANCE_SHADOW_WGSL: &str = r#"
// Variance-Shadow-Map resolve twin: one thread per receiver recovers the
// variance from the pre-filtered occluder moments, floors it at `min_variance`,
// evaluates the one-sided Chebyshev visibility upper bound, and remaps the
// result through the light-bleed `linstep`. It mirrors the CPU golden
// `particle::variance_shadow::resolve_visibility`, uses only the portable
// core-WGSL subset (+ - * / min max clamp and unsigned compares, no
// transcendental), and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: standard two-moment Variance Shadow Maps; no Unreal Engine source
// or derived code.

struct Params {
    // Lower bound on the recovered variance; bounds light bleed and keeps the
    // Chebyshev denominator away from zero. Matches the reference `min_variance`.
    min_variance: f32,
    // Light-bleed reduction amount: the `[bleed, 1]` sub-range is stretched onto
    // `[0, 1]`. Matches the reference `bleed`.
    bleed: f32,
    // Number of valid receivers in `receivers`.
    count: u32,
    // Padding to a 16-byte boundary.
    pad0: u32,
}

// One receiver's input. 16-byte std430 stride: the pre-filtered occluder
// moments (m1 = E[d], m2 = E[d^2]), the receiver depth and a pad word, matching
// the host `GpuReceiver`.
struct Receiver {
    m1: f32,
    m2: f32,
    receiver_depth: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> receivers: array<Receiver>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Denominators and interval widths with magnitude below this are treated as
// (near) zero, matching the reference `MIN_DENOM`.
const MIN_DENOM: f32 = 1.0e-6;

// Clamps a scalar into the 0..=1 visibility range, mirroring the reference
// `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// The one-sided Chebyshev upper bound on receiver visibility, mirroring the
// reference `chebyshev_upper_bound`. A receiver at or in front of the mean
// occluder depth is fully visible; otherwise the bound is
// `variance / (variance + (t - m1)^2)` with the variance floored at
// `min_variance`.
fn chebyshev_upper_bound(m1: f32, m2: f32, receiver_depth: f32, min_variance: f32) -> f32 {
    if (receiver_depth <= m1) {
        return 1.0;
    }
    // Reference order: variance clamps the raw moment difference to be
    // non-negative, then the min_variance floor is applied.
    var variance = max(m2 - m1 * m1, 0.0);
    variance = max(variance, min_variance);
    let diff = receiver_depth - m1;
    let denom = variance + diff * diff;
    if (denom < MIN_DENOM) {
        return 0.0;
    }
    return clamp01(variance / denom);
}

// Suppresses VSM light bleed by remapping a visibility bound with a `linstep`:
// the `[amount, 1]` sub-range is stretched onto `[0, 1]`. Mirrors the reference
// `light_bleed_reduction`, including the degenerate hard-step fallback when the
// span collapses.
fn light_bleed_reduction(p: f32, amount: f32) -> f32 {
    let lo = clamp01(amount);
    let span = 1.0 - lo;
    if (span < MIN_DENOM) {
        if (p < lo) {
            return 0.0;
        }
        return 1.0;
    }
    return clamp01((p - lo) / span);
}

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let receiver = receivers[idx];
    let p = chebyshev_upper_bound(
        receiver.m1,
        receiver.m2,
        receiver.receiver_depth,
        params.min_variance,
    );
    results[idx] = light_bleed_reduction(p, params.bleed);
}
"#;

/// One receiver's `VSM` resolve inputs: the pre-filtered occluder moments and
/// the receiver depth to test against them.
///
/// Mirrors the `(moments_map[i], receiver_depths[i])` pair the reference
/// [`resolve_visibility`](prism_render_architecture::particle::variance_shadow::resolve_visibility)
/// consumes. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32`
/// depths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarianceShadowQuery {
    /// The pre-filtered occluder moments `(E[d], E[d^2])` sampled at this
    /// receiver (the blurred/mip-fetched moment pair).
    pub moments: Moments,
    /// The receiver depth tested against the occluder distribution.
    pub receiver_depth: f32,
}

/// Uniform parameters for one `VSM` resolve dispatch. `repr(C)` layout matching
/// `Params` in [`VARIANCE_SHADOW_WGSL`]: the variance floor, the light-bleed
/// amount, the receiver count and one pad word — `16` bytes, each field at the
/// `std140` uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Lower bound on the recovered variance (`min_variance`).
    min_variance: f32,
    /// Light-bleed reduction amount (`bleed`).
    bleed: f32,
    /// Number of receivers in the dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
}

/// One receiver's inputs as uploaded. `16`-byte `std430` stride matching
/// `Receiver` in the shader: the two occluder moments, the receiver depth and a
/// pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuReceiver {
    /// First occluder moment `E[d]`.
    m1: f32,
    /// Second occluder moment `E[d^2]`.
    m2: f32,
    /// Receiver depth tested against the occluder distribution.
    receiver_depth: f32,
    /// Padding word.
    pad: f32,
}

/// A compiled, reusable `VSM` resolve pipeline.
pub struct GpuVarianceShadow {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVarianceShadow {
    /// Compiles the `VSM` resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVarianceShadow {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_variance_shadow"),
            source: ShaderSource::Wgsl(VARIANCE_SHADOW_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_variance_shadow_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_variance_shadow_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_variance_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVarianceShadow {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves the light-bleed-corrected `VSM` visibility for every receiver in
    /// `queries` under the shared `min_variance` floor and `bleed` amount,
    /// returning one visibility in `0..=1` per receiver in input order.
    ///
    /// The returned result for query `q` equals
    /// [`light_bleed_reduction`](prism_render_architecture::particle::variance_shadow::light_bleed_reduction)`(`[`chebyshev_upper_bound`](prism_render_architecture::particle::variance_shadow::chebyshev_upper_bound)`(&q.moments, q.receiver_depth, min_variance), bleed)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        min_variance: f32,
        bleed: f32,
        queries: &[VarianceShadowQuery],
    ) -> Vec<f32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = Params {
            min_variance,
            bleed,
            count: queries.len() as u32,
            pad0: 0,
        };

        let receivers: Vec<GpuReceiver> = queries
            .iter()
            .map(|q| GpuReceiver {
                m1: q.moments.m1,
                m2: q.moments.m2,
                receiver_depth: q.receiver_depth,
                pad: 0.0,
            })
            .collect();

        // One `f32` visibility per receiver.
        let out_bytes = (queries.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_variance_shadow_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let receivers_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_variance_shadow_receivers"),
            contents: bytemuck::cast_slice(&receivers),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_variance_shadow_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_variance_shadow_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_variance_shadow_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: receivers_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_variance_shadow_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_variance_shadow_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per receiver, in workgroups of 64 (the kernel's size).
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
        let gpu_results = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
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
