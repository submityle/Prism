//! `wgpu` compute twin of the multi-stop `RGBA` colour-gradient sampler
//! ([`color_gradient`](prism_render_architecture::particle::color_gradient),
//! particle design §5, §8, §30).
//!
//! The `CPU` golden
//! [`color_gradient`](prism_render_architecture::particle::color_gradient)
//! sweeps a particle's tint along an authored over-life colour ramp: given a
//! sorted set of [`ColorStop`](prism_render_architecture::particle::color_gradient::ColorStop)s
//! (each a `position` plus an [`Rgba`](prism_render_architecture::particle::color_gradient::Rgba)),
//! [`ColorGradient::sample`](prism_render_architecture::particle::color_gradient::ColorGradient::sample)
//! maps a normalized age `t` to a colour by holding the first / last stop
//! outside the domain and linearly blending the two bracketing stops inside it.
//!
//! [`GpuColorGradient`] is the on-device twin: one thread per query reproduces
//! that control flow branch for branch over a bounded, shared stop ring, so a
//! passing real-device parity test is direct evidence the ported kernel selects
//! the same segment and blends the same colour the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! Every query carries one scalar `t` and resolves to one
//! [`ColorGradientSample`] (`r`, `g`, `b`, `a`). The host packs the gradient's
//! sorted stops once into a fixed-capacity [`MAX_STOPS`] block with a live
//! `stop_count`; each thread then mirrors the reference `sample`: a `t` at or
//! below the first stop holds the first colour, a `t` at or above the last stop
//! holds the last colour, and any interior `t` locates its bracketing segment
//! and blends component-wise by the local parameter `(t - lo.position) /
//! (hi.position - lo.position)`.
//!
//! # Fixed capacity
//!
//! The stop ring is a fixed `std430` block of [`MAX_STOPS`] entries in the
//! uniform parameters, with `stop_count` naming how many are live, so the
//! kernel needs no dynamic allocation and keeps the one-thread-per-element
//! dispatch rule. The host rejects any gradient whose stop count exceeds the
//! capacity, and short-circuits an empty gradient (no stops) to the transparent
//! guard without a dispatch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`,
//! `+ - * /`, unsigned index arithmetic and a bounded loop over the live stops —
//! with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no inverse trigonometry,
//! no `sqrt` and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The loop is bounded by [`MAX_STOPS`], so the kernel
//! provably terminates.
//!
//! # Correctness model
//!
//! The endpoint-hold and segment-selection decisions are discrete `f32`
//! magnitude comparisons (`<=`, `>=`, `<`), so for a `t` clear of the stop
//! positions the `CPU` and `GPU` take the same branch and pick the same
//! segment. The blended channels thread through a subtract, a multiply and an
//! add plus one guarded division, so `CPU` and `GPU` are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! channel, tight enough to catch a genuinely wrong port yet loose enough to
//! admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! A zero-width segment (two stops sharing a position, a hard colour step)
//! would divide by zero; the kernel checks the span against [`SEG_EPS`] and
//! collapses the local parameter to `0.0`, holding the lower stop's colour,
//! exactly as the reference does. An empty query batch short-circuits on the
//! host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::color_gradient`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::color_gradient::ColorStop;
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

/// Fixed capacity of the shared stop ring. A gradient may carry up to this many
/// [`ColorStop`](prism_render_architecture::particle::color_gradient::ColorStop)s
/// in the uniform `std430` block; the host rejects any gradient that exceeds
/// this bound. Must match the `STOP_CAP` constant inside [`COLOR_GRADIENT_WGSL`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::color_gradient`；无第三方引擎源码或衍生代码。
pub const MAX_STOPS: usize = 16;

/// The portable core-`WGSL` colour-gradient kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`color_gradient`](prism_render_architecture::particle::color_gradient)
/// branch for branch; see the module documentation for the algorithm.
const COLOR_GRADIENT_WGSL: &str = r#"
// Colour-gradient twin: one thread per query maps a scalar t to an RGBA by
// holding the first / last stop outside the domain and linearly blending the
// two bracketing stops inside it. It mirrors the CPU golden
// particle::color_gradient branch for branch, uses only the portable core-WGSL
// subset (abs/min and + - * / plus unsigned index math and a bounded loop) and
// takes no optional feature, so it runs unmodified on Metal, Vulkan and DX12.
// The loop is bounded by STOP_CAP, so the kernel provably terminates.
//
// Provenance: twinned from this repository's particle::color_gradient; no
// third-party engine source or derived code.

// Minimum segment width below which the span is treated as a hard step, so the
// local parameter collapses to 0 rather than dividing by (near) zero. Matches
// the reference `SEG_EPS`; this is the compare rule used instead of an f32 ==.
const SEG_EPS: f32 = 1.0e-12;

// Fixed stop-ring capacity, matching the host `MAX_STOPS`.
const STOP_CAP: u32 = 16u;

// One authored colour stop: an RGBA `color` and the normalized `position` it
// takes, with three pad lanes keeping the struct 16-byte aligned on device.
struct ColorStop {
    color: vec4<f32>,
    position: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    query_count: u32,
    // Number of live stops in `stops`, at most STOP_CAP.
    stop_count: u32,
    pad0: u32,
    pad1: u32,
    // Fixed-capacity, sorted stop ring; only the first `stop_count` are live.
    stops: array<ColorStop, STOP_CAP>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<vec4<f32>>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let n = params.stop_count;

    // Empty gradient guards to transparent black, matching the reference
    // `Rgba::TRANSPARENT`. The host also short-circuits this case with no
    // dispatch, so this branch is a defensive mirror of the golden guard.
    var rgba: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (n == 0u) {
        results[idx] = rgba;
        return;
    }

    let t = queries[idx];

    // Endpoint holds: a t at or below the first stop, or at or above the last
    // stop, takes that stop's colour unchanged.
    let first = params.stops[0];
    if (t <= first.position) {
        results[idx] = first.color;
        return;
    }
    let last = params.stops[n - 1u];
    if (t >= last.position) {
        results[idx] = last.color;
        return;
    }

    // Count of stops whose position is <= t. Since the ring is sorted this is
    // the reference `partition_point` result; the bracketing segment's lower
    // endpoint sits one back.
    var upper: u32 = 0u;
    for (var k: u32 = 0u; k < n; k = k + 1u) {
        if (params.stops[k].position <= t) {
            upper = upper + 1u;
        }
    }
    // upper >= 1 here, since t > first.position means stop 0 counts.
    let i = min(upper - 1u, n - 2u);
    let s0 = params.stops[i];
    let s1 = params.stops[i + 1u];

    // Local parameter (t - lo.position) / (hi.position - lo.position); a hard
    // step (span within SEG_EPS) collapses it to 0 so the division never
    // propagates NaN, matching the reference.
    let dp = s1.position - s0.position;
    var u: f32 = 0.0;
    if (abs(dp) > SEG_EPS) {
        u = (t - s0.position) / dp;
    }

    // Component-wise lerp a + (b - a) * u, matching the reference `Rgba::lerp`
    // channel order and arithmetic.
    rgba = s0.color + (s1.color - s0.color) * u;
    results[idx] = rgba;
}
"#;

/// `repr(C)` `std430` layout of one packed colour stop, matching the `WGSL`
/// `ColorStop` struct: a `vec4` colour followed by the scalar `position` and
/// three pad lanes keeping the entry `16`-byte aligned (`32` bytes total).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuColorStop {
    /// The stop's `RGBA` colour (`HDR` channels may exceed `1.0`).
    color: [f32; 4],
    /// The stop's normalized position along the gradient's life.
    position: f32,
    /// Padding lane.
    pad0: f32,
    /// Padding lane.
    pad1: f32,
    /// Padding lane.
    pad2: f32,
}

/// Uniform parameters for one dispatch: the query count, the live stop count,
/// two pad words completing the `16`-byte header, and the fixed-capacity sorted
/// stop ring shared by every query, matching the `WGSL` `Params` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the input and output buffers.
    query_count: u32,
    /// Number of live stops in `stops`.
    stop_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Fixed-capacity stop ring; only the first `stop_count` entries are live.
    stops: [GpuColorStop; MAX_STOPS],
}

/// `repr(C)` `std430` layout of one result: a single `RGBA` `vec4`, matching
/// the `WGSL` `results` element.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    /// The sampled `RGBA` colour.
    color: [f32; 4],
}

/// One colour-gradient query: a single normalized age `t`, the input the
/// reference
/// [`ColorGradient::sample`](prism_render_architecture::particle::color_gradient::ColorGradient::sample)
/// consumes.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::color_gradient`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorGradientQuery {
    /// Normalized age at which to sample the gradient.
    pub t: f32,
}

impl ColorGradientQuery {
    /// Builds a query that samples the gradient at normalized age `t`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::color_gradient`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(t: f32) -> ColorGradientQuery {
        ColorGradientQuery { t }
    }
}

/// The resolved colour for one query, mirroring the reference
/// [`Rgba`](prism_render_architecture::particle::color_gradient::Rgba): four
/// linear channels whose values may exceed `1.0` for `HDR`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::color_gradient`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorGradientSample {
    /// Red channel (linear, may exceed `1.0` for `HDR`).
    pub r: f32,
    /// Green channel (linear, may exceed `1.0` for `HDR`).
    pub g: f32,
    /// Blue channel (linear, may exceed `1.0` for `HDR`).
    pub b: f32,
    /// Alpha channel (linear, may exceed `1.0` for `HDR` premultiply).
    pub a: f32,
}

/// Packs one golden [`ColorStop`](prism_render_architecture::particle::color_gradient::ColorStop)
/// into its `std430` [`GpuColorStop`] slot.
fn encode_stop(stop: &ColorStop) -> GpuColorStop {
    GpuColorStop {
        color: [stop.color.r, stop.color.g, stop.color.b, stop.color.a],
        position: stop.position,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuSample`] into the public [`ColorGradientSample`].
fn decode_sample(raw: &GpuSample) -> ColorGradientSample {
    ColorGradientSample {
        r: raw.color[0],
        g: raw.color[1],
        b: raw.color[2],
        a: raw.color[3],
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

/// A compiled, reusable colour-gradient compute pipeline, twinning the `CPU`
/// golden
/// [`color_gradient`](prism_render_architecture::particle::color_gradient).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::color_gradient`；无第三方引擎源码或衍生代码。
pub struct GpuColorGradient {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuColorGradient {
    /// Compiles the colour-gradient kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::color_gradient`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuColorGradient {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_color_gradient"),
            source: ShaderSource::Wgsl(COLOR_GRADIENT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_color_gradient_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_color_gradient_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_color_gradient_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuColorGradient {
            module,
            layout,
            pipeline,
        }
    }

    /// Samples the gradient defined by `stops` at every query in `queries`,
    /// returning one [`ColorGradientSample`] per input, in order.
    ///
    /// `stops` must be sorted by position ascending (the order the reference
    /// [`ColorGradient`](prism_render_architecture::particle::color_gradient::ColorGradient)
    /// stores them after
    /// [`from_stops`](prism_render_architecture::particle::color_gradient::ColorGradient::from_stops)).
    /// Each sample equals the reference
    /// [`ColorGradient::sample`](prism_render_architecture::particle::color_gradient::ColorGradient::sample)
    /// to within the tolerance documented on this module. An empty gradient
    /// (no stops) short-circuits to the transparent guard for every query, and
    /// an empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics if `stops` carries more than [`MAX_STOPS`] entries, since the
    /// fixed uniform block cannot hold them.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::color_gradient`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn sample(
        &self,
        ctx: &GpuContext,
        stops: &[ColorStop],
        queries: &[ColorGradientQuery],
    ) -> Vec<ColorGradientSample> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        assert!(
            stops.len() <= MAX_STOPS,
            "gradient must carry at most MAX_STOPS stops"
        );
        // An empty gradient samples to transparent black everywhere; the
        // reference short-circuits the same way, so no dispatch is issued.
        if stops.is_empty() {
            return vec![
                ColorGradientSample {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                };
                count
            ];
        }
        let device = ctx.device();

        let mut packed_stops = [GpuColorStop {
            color: [0.0; 4],
            position: 0.0,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        }; MAX_STOPS];
        for (slot, stop) in packed_stops.iter_mut().zip(stops.iter()) {
            *slot = encode_stop(stop);
        }
        let params = GpuParams {
            query_count: count as u32,
            stop_count: stops.len() as u32,
            pad0: 0,
            pad1: 0,
            stops: packed_stops,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_color_gradient_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let ts: Vec<f32> = queries.iter().map(|q| q.t).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_color_gradient_queries"),
            contents: bytemuck::cast_slice(&ts),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuSample>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_color_gradient_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_color_gradient_bind_group"),
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
            label: Some("prism_volumetric_color_gradient_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_color_gradient_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_color_gradient_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuSample>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_sample).collect()
    }
}
