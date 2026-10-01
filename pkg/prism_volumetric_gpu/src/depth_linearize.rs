//! `wgpu` compute twin of the perspective depth-linearization golden
//! ([`depth_linearize`](prism_render_architecture::particle::depth_linearize),
//! design §16-§21).
//!
//! A perspective projection stores a `1 / z` shaped depth, not a metric
//! view-space distance, so any screen-space particle pass that fades against the
//! scene, computes a circle-of-confusion or applies linear fog must first undo
//! that non-linearity. The `CPU` golden
//! [`depth_linearize`](prism_render_architecture::particle::depth_linearize)
//! owns that arithmetic primitive and its exact inverse;
//! [`GpuDepthLinearize`] is the on-device twin that reproduces the same scalar
//! math so a passing real-device parity test is direct evidence the ported
//! kernel divides the same denominators, takes the same degenerate fallbacks and
//! flips the same reverse-`Z` endpoints the reference does, not merely that its
//! shader compiles.
//!
//! # What is twinned
//!
//! A single [`DepthLinearizeQuery`] drives every portable per-element function
//! at once, and the combined [`DepthLinearizeResult`] carries one output each:
//!
//! * [`linearize_01`](prism_render_architecture::particle::depth_linearize::linearize_01)
//!   — the `[0, 1]` (`D3D` / `wgpu`) depth to view-space distance
//!   `near * far / (far - depth * (far - near))`, with the reverse-`Z` pre-flip.
//! * [`delinearize_01`](prism_render_architecture::particle::depth_linearize::delinearize_01)
//!   — the exact inverse `far * (linear - near) / (linear * range)`, with the
//!   reverse-`Z` post-flip and the non-positive-linear fallback.
//! * [`linear_to_01_normalized`](prism_render_architecture::particle::depth_linearize::linear_to_01_normalized)
//!   — the plain linear remap `(linear - near) / range` clamped to `[0, 1]`.
//! * [`ndc_to_view_z`](prism_render_architecture::particle::depth_linearize::ndc_to_view_z)
//!   — the `OpenGL` `[-1, 1]` `NDC` form
//!   `2 * near * far / (far + near - ndc * range)`.
//! * [`perspective_interpolate`](prism_render_architecture::particle::depth_linearize::perspective_interpolate)
//!   — the `1 / w`-weighted attribute blend with the screen-linear `lerp`
//!   fallback.
//!
//! The batch [`linearize_buffer`](prism_render_architecture::particle::depth_linearize::linearize_buffer)
//! is twinned by a second entry point that takes the shared
//! [`DepthParams`] as a uniform and maps
//! [`linearize_01`](prism_render_architecture::particle::depth_linearize::linearize_01)
//! across a storage buffer of depths, one thread per element.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `clamp`, `abs` and
//! `+ - * /` on scalars plus unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `sqrt`, `tan`, built-in `smoothstep` or optional device
//! feature, so they run unmodified on `Metal`, `Vulkan` and `DX12`. Every
//! routine is plain rational `f32` algebra guarded against division by zero with
//! an epsilon compare, matching the reference exactly.
//!
//! # Correctness model
//!
//! Each output is a fixed, non-reorderable sequence of multiplies, adds and one
//! divide, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a swapped `near` / `far`, a dropped reverse-`Z` flip, a missing
//! degenerate guard) yet loose enough to admit legal fused multiply-add
//! contraction. Fixtures are kept clear of the division-by-zero cracks so a
//! `GPU`'s fused multiply-add cannot flip a degenerate branch.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::depth_linearize`；
//! 无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::depth_linearize::DepthParams;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// `f32` lanes per packed [`DepthLinearizeQuery`] in the flat input buffer:
/// three [`DepthParams`] scalars, five per-function inputs, the three
/// `perspective_interpolate` reciprocal weights, and one pad lane to a clean
/// three-`vec4` grouping.
const QUERY_STRIDE: usize = 12;

/// `f32` lanes per packed [`DepthLinearizeResult`] in the flat output buffer:
/// the five function outputs followed by three pad lanes to a clean two-`vec4`
/// grouping.
const RESULT_STRIDE: usize = 8;

/// The portable core-`WGSL` depth-linearization kernels, embedded inline so the
/// twin ships as a single source file. The `evaluate` entry mirrors the five
/// per-element golden functions and the `linearize_buffer` entry mirrors the
/// batch form; see the module documentation for the algorithm.
const DEPTH_LINEARIZE_WGSL: &str = r#"
// Depth-linearization twin. Two entry points share one bind-group layout
// (a uniform plus a read storage buffer and a read_write storage buffer, both
// flat `array<f32>`). `evaluate` runs one thread per DepthLinearizeQuery and
// reproduces the whole portable depth_linearize surface -- linearize_01,
// delinearize_01, linear_to_01_normalized, ndc_to_view_z and
// perspective_interpolate -- reading 12 lanes per query and writing 8 lanes per
// result. `linearize_buffer` runs one thread per depth sample and maps
// linearize_01 across a storage buffer with the shared DepthParams supplied as
// a uniform. Both use only the portable core-WGSL subset (clamp/abs and
// + - * / plus unsigned index math, no transcendental, no sqrt, no built-in
// smoothstep) and take no optional feature, so they run unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::depth_linearize;
// no third-party engine source or derived code.

struct Params {
    // Shared DepthParams for the `linearize_buffer` entry (unused by `evaluate`,
    // which carries per-query params in the input buffer).
    near: f32,
    far: f32,
    // Reverse-Z flag packed as 1.0 / 0.0, matching DepthParams::to_std430.
    reverse_flag: f32,
    // Number of valid elements (queries for `evaluate`, depths for
    // `linearize_buffer`).
    count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

// Epsilon below which a denominator or the `far - near` range is treated as
// degenerate so the routine falls back instead of dividing by zero, matching
// the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

// Converts a [0, 1] (D3D / wgpu) non-linear depth into a positive view-space
// linear distance, mirroring `linearize_01`. Reverse-Z pre-flips the depth; a
// degenerate range or denominator falls back to a clip-plane distance.
fn linearize_01(near: f32, far: f32, reverse: bool, ndc_depth_01: f32) -> f32 {
    let range = far - near;
    if (abs(range) < CMP_EPS) {
        return near;
    }
    var depth = ndc_depth_01;
    if (reverse) {
        depth = 1.0 - ndc_depth_01;
    }
    let denom = far - depth * range;
    if (abs(denom) < CMP_EPS) {
        return far;
    }
    return near * far / denom;
}

// Inverse of `linearize_01`, mirroring `delinearize_01`. A degenerate range or
// a non-positive linear distance falls back to a clip-plane depth; reverse-Z
// post-flips the result.
fn delinearize_01(near: f32, far: f32, reverse: bool, linear: f32) -> f32 {
    let range = far - near;
    if (abs(range) < CMP_EPS) {
        if (reverse) {
            return 1.0;
        }
        return 0.0;
    }
    if (abs(linear) < CMP_EPS) {
        if (reverse) {
            return 1.0;
        }
        return 0.0;
    }
    let depth = far * (linear - near) / (linear * range);
    if (reverse) {
        return 1.0 - depth;
    }
    return depth;
}

// Plain linear remap of a view-space distance to [0, 1], mirroring
// `linear_to_01_normalized`. A degenerate range falls back to 0.
fn linear_to_01_normalized(near: f32, far: f32, linear: f32) -> f32 {
    let range = far - near;
    if (abs(range) < CMP_EPS) {
        return 0.0;
    }
    return clamp((linear - near) / range, 0.0, 1.0);
}

// Converts an OpenGL [-1, 1] NDC depth into a positive view-space linear
// distance, mirroring `ndc_to_view_z`. Reverse-Z flips the NDC sign; a
// degenerate range or denominator falls back to a clip-plane distance.
fn ndc_to_view_z(near: f32, far: f32, reverse: bool, ndc_z: f32) -> f32 {
    let range = far - near;
    if (abs(range) < CMP_EPS) {
        return near;
    }
    var ndc = ndc_z;
    if (reverse) {
        ndc = -ndc_z;
    }
    let denom = far + near - ndc * range;
    if (abs(denom) < CMP_EPS) {
        return far;
    }
    return 2.0 * near * far / denom;
}

// Perspective-correct attribute interpolation, mirroring
// `perspective_interpolate`: blends `attr / w` and `1 / w` linearly and
// divides. A degenerate weight sum falls back to a screen-linear lerp.
fn perspective_interpolate(a: f32, b: f32, inv_w_a: f32, inv_w_b: f32, t: f32) -> f32 {
    let w0 = inv_w_a * (1.0 - t);
    let w1 = inv_w_b * t;
    let denom = w0 + w1;
    if (abs(denom) < CMP_EPS) {
        return a + (b - a) * t;
    }
    return (a * w0 + b * w1) / denom;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let base = idx * 12u;
    let near = src[base + 0u];
    let far = src[base + 1u];
    let reverse = src[base + 2u] > 0.5;
    let depth_01 = src[base + 3u];
    let linear = src[base + 4u];
    let ndc_z = src[base + 5u];
    let lerp_a = src[base + 6u];
    let lerp_b = src[base + 7u];
    let inv_w_a = src[base + 8u];
    let inv_w_b = src[base + 9u];
    let t = src[base + 10u];
    // src[base + 11u] is the pad lane.

    let obase = idx * 8u;
    dst[obase + 0u] = linearize_01(near, far, reverse, depth_01);
    dst[obase + 1u] = delinearize_01(near, far, reverse, linear);
    dst[obase + 2u] = linear_to_01_normalized(near, far, linear);
    dst[obase + 3u] = ndc_to_view_z(near, far, reverse, ndc_z);
    dst[obase + 4u] = perspective_interpolate(lerp_a, lerp_b, inv_w_a, inv_w_b, t);
    dst[obase + 5u] = 0.0;
    dst[obase + 6u] = 0.0;
    dst[obase + 7u] = 0.0;
}

@compute @workgroup_size(64)
fn linearize_buffer(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let reverse = params.reverse_flag > 0.5;
    dst[idx] = linearize_01(params.near, params.far, reverse, src[idx]);
}
"#;

/// Inputs for one depth-linearization query (design §16-§21). A single query
/// drives every portable per-element function so the combined result can be
/// compared against the whole golden surface at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthLinearizeQuery {
    /// The depth-range parameters shared by every function in this query.
    pub params: DepthParams,
    /// The `[0, 1]` (`D3D` / `wgpu`) depth fed to
    /// [`linearize_01`](prism_render_architecture::particle::depth_linearize::linearize_01).
    pub depth_01: f32,
    /// The view-space linear distance fed to
    /// [`delinearize_01`](prism_render_architecture::particle::depth_linearize::delinearize_01)
    /// and
    /// [`linear_to_01_normalized`](prism_render_architecture::particle::depth_linearize::linear_to_01_normalized).
    pub linear: f32,
    /// The `OpenGL` `[-1, 1]` `NDC` depth fed to
    /// [`ndc_to_view_z`](prism_render_architecture::particle::depth_linearize::ndc_to_view_z).
    pub ndc_z: f32,
    /// The first endpoint attribute for
    /// [`perspective_interpolate`](prism_render_architecture::particle::depth_linearize::perspective_interpolate).
    pub lerp_a: f32,
    /// The second endpoint attribute for
    /// [`perspective_interpolate`](prism_render_architecture::particle::depth_linearize::perspective_interpolate).
    pub lerp_b: f32,
    /// The first endpoint's `1 / w` reciprocal depth weight.
    pub inv_w_a: f32,
    /// The second endpoint's `1 / w` reciprocal depth weight.
    pub inv_w_b: f32,
    /// The screen-linear interpolation parameter in `[0, 1]`.
    pub t: f32,
}

/// The device-computed result for one [`DepthLinearizeQuery`], one scalar per
/// twinned golden function for a direct parity comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthLinearizeResult {
    /// `linearize_01(params, depth_01)`.
    pub linearized: f32,
    /// `delinearize_01(params, linear)`.
    pub delinearized: f32,
    /// `linear_to_01_normalized(params, linear)`.
    pub normalized: f32,
    /// `ndc_to_view_z(params, ndc_z)`.
    pub view_z: f32,
    /// `perspective_interpolate(lerp_a, lerp_b, inv_w_a, inv_w_b, t)`.
    pub perspective: f32,
}

/// `std430` uniform layout matching `Params` in [`DEPTH_LINEARIZE_WGSL`], `16`
/// bytes: the shared [`DepthParams`] scalars plus the element count.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    near: f32,
    far: f32,
    reverse_flag: f32,
    count: u32,
}

impl GpuParams {
    /// Builds the uniform for the `evaluate` entry, whose per-query params live
    /// in the input buffer; only `count` is read, so the range fields are zero.
    fn for_queries(count: u32) -> GpuParams {
        GpuParams {
            near: 0.0,
            far: 0.0,
            reverse_flag: 0.0,
            count,
        }
    }

    /// Builds the uniform for the `linearize_buffer` entry from the shared
    /// [`DepthParams`] and the depth-sample count.
    fn for_buffer(params: &DepthParams, count: u32) -> GpuParams {
        let reverse_flag: f32 = if params.reverse_z { 1.0 } else { 0.0 };
        GpuParams {
            near: params.near,
            far: params.far,
            reverse_flag,
            count,
        }
    }
}

/// A compiled, reusable depth-linearization pipeline pair sharing one bind-group
/// layout.
pub struct GpuDepthLinearize {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_eval: ComputePipeline,
    pipeline_buffer: ComputePipeline,
}

impl GpuDepthLinearize {
    /// Compiles the depth-linearization kernels on `ctx`.
    ///
    /// Both entry points use only the portable core-`WGSL` subset, so no
    /// optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDepthLinearize {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_depth_linearize"),
            source: ShaderSource::Wgsl(DEPTH_LINEARIZE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_depth_linearize_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_depth_linearize_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline_eval = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_depth_linearize_evaluate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_buffer = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_depth_linearize_buffer_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("linearize_buffer"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDepthLinearize {
            module,
            layout,
            pipeline_eval,
            pipeline_buffer,
        }
    }

    /// Runs the per-element depth-linearization evaluation for every query,
    /// returning one [`DepthLinearizeResult`] per query in input order.
    ///
    /// The returned result for query `q` reproduces, field for field, the golden
    /// [`linearize_01`](prism_render_architecture::particle::depth_linearize::linearize_01),
    /// [`delinearize_01`](prism_render_architecture::particle::depth_linearize::delinearize_01),
    /// [`linear_to_01_normalized`](prism_render_architecture::particle::depth_linearize::linear_to_01_normalized),
    /// [`ndc_to_view_z`](prism_render_architecture::particle::depth_linearize::ndc_to_view_z)
    /// and
    /// [`perspective_interpolate`](prism_render_architecture::particle::depth_linearize::perspective_interpolate).
    /// An empty `queries` slice yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return and no dispatch.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[DepthLinearizeQuery],
    ) -> Vec<DepthLinearizeResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let mut input: Vec<f32> = Vec::with_capacity(queries.len() * QUERY_STRIDE);
        for q in queries {
            let reverse_flag: f32 = if q.params.reverse_z { 1.0 } else { 0.0 };
            input.push(q.params.near);
            input.push(q.params.far);
            input.push(reverse_flag);
            input.push(q.depth_01);
            input.push(q.linear);
            input.push(q.ndc_z);
            input.push(q.lerp_a);
            input.push(q.lerp_b);
            input.push(q.inv_w_a);
            input.push(q.inv_w_b);
            input.push(q.t);
            input.push(0.0);
        }
        debug_assert_eq!(input.len(), queries.len() * QUERY_STRIDE);

        let out_len = queries.len() * RESULT_STRIDE;
        let out_bytes = (out_len * size_of::<f32>()) as u64;

        let params = GpuParams::for_queries(queries.len() as u32);
        let flat = self.dispatch(
            ctx,
            device,
            &self.pipeline_eval,
            &params,
            &input,
            out_bytes,
            queries.len() as u32,
        );

        flat.chunks_exact(RESULT_STRIDE)
            .map(|lanes| DepthLinearizeResult {
                linearized: lanes[0],
                delinearized: lanes[1],
                normalized: lanes[2],
                view_z: lanes[3],
                perspective: lanes[4],
            })
            .collect()
    }

    /// Linearizes a whole slice of `[0, 1]` depth-buffer values with the shared
    /// `params` supplied as a uniform, preserving order and length.
    ///
    /// Reproduces the golden
    /// [`linearize_buffer`](prism_render_architecture::particle::depth_linearize::linearize_buffer)
    /// (equivalently
    /// [`linearize_01`](prism_render_architecture::particle::depth_linearize::linearize_01)
    /// mapped over the input). An empty `depths` slice yields an empty result —
    /// storage buffers cannot be zero-sized, so it is handled by an early return
    /// and no dispatch.
    #[must_use]
    pub fn linearize_buffer(
        &self,
        ctx: &GpuContext,
        params: &DepthParams,
        depths: &[f32],
    ) -> Vec<f32> {
        if depths.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let out_bytes = size_of_val(depths) as u64;
        let gpu_params = GpuParams::for_buffer(params, depths.len() as u32);
        self.dispatch(
            ctx,
            device,
            &self.pipeline_buffer,
            &gpu_params,
            depths,
            out_bytes,
            depths.len() as u32,
        )
    }

    /// Uploads `input`, runs `pipeline` over `count` threads, and reads back the
    /// `out_bytes`-sized flat `f32` output buffer. Shared by both entries since
    /// they use the identical bind-group layout.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        device: &wgpu::Device,
        pipeline: &ComputePipeline,
        params: &GpuParams,
        input: &[f32],
        out_bytes: u64,
        count: u32,
    ) -> Vec<f32> {
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_depth_linearize_params"),
            contents: bytemuck::bytes_of(params),
            usage: BufferUsages::UNIFORM,
        });
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_depth_linearize_input"),
            contents: bytemuck::cast_slice(input),
            usage: BufferUsages::STORAGE,
        });
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_depth_linearize_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_depth_linearize_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_depth_linearize_bind_group"),
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

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_depth_linearize_encoder"),
        });
        {
            let groups = count.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_depth_linearize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();
        flat
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
