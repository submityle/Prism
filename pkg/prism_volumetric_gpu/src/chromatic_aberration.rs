//! `wgpu` compute twin of the screen-space `chromatic` aberration split
//! ([`chromatic_aberration`](prism_render_architecture::particle::chromatic_aberration),
//! particle design §16-§21).
//!
//! The `CPU` golden
//! [`chromatic_aberration`](prism_render_architecture::particle::chromatic_aberration)
//! owns the deterministic per-pixel `RGB` split a post-process compositor layers
//! over lens-like renderers: the scene color is sampled three times per pixel,
//! each `RGB` channel offset along the outward radial direction from an optical
//! center, with the offset growing toward the frame edges where lens dispersion
//! is worst. Its four pieces are the integer power
//! [`pow_u32`](prism_render_architecture::particle::chromatic_aberration), the
//! rational-polynomial
//! [`radial_intensity`](prism_render_architecture::particle::chromatic_aberration::ChromaParams::radial_intensity)
//! shaped by the multiply-only `smoothstep`, the three per-channel
//! [`channel_offsets`](prism_render_architecture::particle::chromatic_aberration::ChromaParams::channel_offsets),
//! and the clamped per-channel
//! [`sample_uvs`](prism_render_architecture::particle::chromatic_aberration::ChromaParams::sample_uvs).
//!
//! [`GpuChromaticAberration`] is the on-device twin: one thread solves one
//! source `UV`, so a passing real-device parity test is direct evidence the
//! ported kernel samples the same three split `UV`s and the same radial
//! intensity the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For a batch of independent source `UV`s sharing one [`ChromaParams`], every
//! per-`UV` answer the reference computes is reproduced: the three per-channel
//! sampled `UV`s `[red, green, blue]` (each clamped into the `[0, 1]` texture
//! domain) and the shared `radial_intensity` at the sampled radius. The private
//! integer power
//! [`pow_u32`](prism_render_architecture::particle::chromatic_aberration) is
//! reproduced as a bounded `WGSL` `for` loop (`exp` accumulating multiplies,
//! never the forbidden `pow`), and the `smoothstep` is unrolled to the same
//! multiply-only `t * t * (3 - 2 t)` Hermite polynomial.
//!
//! # Correctness model
//!
//! Each sample threads through multiplies, adds, one divide, one `sqrt` (the
//! radius) and the `pow_u32` loop, so `CPU` and `GPU` are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous quantity,
//! tight enough to catch a genuinely wrong port (a dropped term, a swapped
//! channel sign, a wrong clamp) yet loose enough to admit legal fused
//! multiply-add contraction. The `pow_u32` loop bound is an exact `u32` the host
//! guarantees stays small (`exp <= 16`), so both devices execute the identical
//! number of iterations.
//!
//! # Degenerate inputs
//!
//! There is no division by a data-dependent denominator: the radial-intensity
//! denominator `1 + r^2` is at least `1`, so no guard epsilon is needed and no
//! `NaN` is produced. Every sampled `UV` is soft-clamped into `[0, 1]`; the
//! clamp is `1`-Lipschitz, so a few units in the last place of slack never flip
//! the result by more than the tolerance. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`,
//! `+ - * /`, one `sqrt` and a bounded `for` loop — with no `sin`, `cos`, `tan`,
//! no inverse trigonometry, no `exp`, `log`, `pow` or `smoothstep` builtin and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The single loop has a host-bounded `u32` trip count, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::chromatic_aberration`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::chromatic_aberration::ChromaParams;
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

/// The portable core-`WGSL` `chromatic`-aberration kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`chromatic_aberration`](prism_render_architecture::particle::chromatic_aberration)
/// branch for branch; see the module documentation for the algorithm.
const CHROMATIC_ABERRATION_WGSL: &str = r#"
// Chromatic-aberration twin: one thread per source UV reproduces the three
// per-channel sampled UVs (each clamped into [0, 1]) and the shared radial
// intensity. It mirrors the CPU golden particle::chromatic_aberration branch for
// branch, uses only the portable core-WGSL subset (clamp/max and + - * / plus
// one sqrt and a bounded for loop) and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12. The pow_u32 loop reproduces the private
// integer power with an accumulating multiply instead of the forbidden pow, and
// the smoothstep is unrolled to the multiply-only Hermite polynomial.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::chromatic_aberration；
// 无第三方引擎源码或衍生代码。

// Fixed exponent of the radial-intensity power term, matching the reference
// `pow_u32(r, 2)`. The host guarantees any such exponent stays small (<= 16) so
// the loop trip count is a tiny exact u32 on both devices.
const CHROMA_POW_EXP: u32 = 2u;

struct Meta {
    // Number of source UVs in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Chroma {
    // The seven ChromaParams scalars packed exactly as the reference `to_std430`
    // lays them out, followed by one padding scalar so the block fills two vec4
    // slots. The center is kept as two scalars (not a vec2) so the WGSL offsets
    // match the std430 byte layout the host writes.
    strength: f32,
    radial_falloff_k: f32,
    r_scale: f32,
    g_scale: f32,
    b_scale: f32,
    center_x: f32,
    center_y: f32,
    pad: f32,
}

struct Query {
    // The source UV sampled by this thread; two pad lanes keep the slot 16-byte
    // aligned.
    uv: vec2<f32>,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // The three per-channel sampled UVs [red, green, blue], the shared radial
    // intensity and one pad lane filling two vec4 slots.
    r_uv: vec2<f32>,
    g_uv: vec2<f32>,
    b_uv: vec2<f32>,
    radial: f32,
    pad: f32,
}

@group(0) @binding(0) var<uniform> meta_info: Meta;
@group(0) @binding(1) var<uniform> chroma: Chroma;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// Clamps a scalar into the closed unit interval [0, 1]; mirrors the reference
// `clamp01`.
fn clamp01(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Clamps a UV component-wise into the [0, 1] texture domain; mirrors the
// reference `clamp_uv`.
fn clamp_uv(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(clamp01(uv.x), clamp01(uv.y));
}

// Integer power base^exp built from an accumulating multiply loop, reproducing
// the private reference `pow_u32` without the forbidden pow. The trip count is a
// host-bounded exact u32, so the kernel provably terminates.
fn pow_u32(base: f32, exp: u32) -> f32 {
    var acc: f32 = 1.0;
    for (var i: u32 = 0u; i < exp; i = i + 1u) {
        acc = acc * base;
    }
    return acc;
}

// The multiply-only smoothstep shaper t^2 (3 - 2 t) after clamping t into
// [0, 1]; mirrors the reference `smoothstep01`.
fn smoothstep01(t: f32) -> f32 {
    let c = clamp01(t);
    return c * c * (3.0 - 2.0 * c);
}

// The rational-polynomial radial intensity r^2 (1 + k r^2) / (1 + r^2) soft-
// clamped into [0, 1]; mirrors the reference `radial_intensity`. A negative
// falloff coefficient is floored to zero and the denominator is at least 1, so
// no guard epsilon is needed.
fn radial_intensity(r: f32) -> f32 {
    let k = max(chroma.radial_falloff_k, 0.0);
    let r2 = pow_u32(r, CHROMA_POW_EXP);
    let num = r2 * (1.0 + k * r2);
    let den = 1.0 + r2;
    return smoothstep01(num / den);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= meta_info.count) {
        return;
    }
    let uv = queries[idx].uv;
    let center = vec2<f32>(chroma.center_x, chroma.center_y);

    // The outward radial vector uv - center and its length, exactly as the
    // reference `channel_offsets` computes them.
    let d = uv - center;
    let r = sqrt(d.x * d.x + d.y * d.y);
    let base = chroma.strength * radial_intensity(r);

    // Red pushes outward (+r_scale), green stays on the baseline (+g_scale,
    // typically 0) and blue pulls inward (-b_scale), matching the reference
    // channel signs.
    let r_gain = base * chroma.r_scale;
    let g_gain = base * chroma.g_scale;
    let b_gain = -base * chroma.b_scale;
    let r_off = vec2<f32>(d.x * r_gain, d.y * r_gain);
    let g_off = vec2<f32>(d.x * g_gain, d.y * g_gain);
    let b_off = vec2<f32>(d.x * b_gain, d.y * b_gain);

    var out: Result;
    out.r_uv = clamp_uv(uv + r_off);
    out.g_uv = clamp_uv(uv + g_off);
    out.b_uv = clamp_uv(uv + b_off);
    out.radial = radial_intensity(r);
    out.pad = 0.0;
    results[idx] = out;
}
"#;

/// One source `UV` for the `chromatic`-aberration twin: the pixel coordinate
/// whose three per-channel split samples are evaluated against the shared
/// [`ChromaParams`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::chromatic_aberration`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChromaticAberrationQuery {
    /// The source `UV` coordinate sampled by this query.
    pub uv: [f32; 2],
}

impl ChromaticAberrationQuery {
    /// Builds a query from the source `UV` coordinate.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::chromatic_aberration`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub const fn new(uv: [f32; 2]) -> ChromaticAberrationQuery {
        ChromaticAberrationQuery { uv }
    }
}

/// The resolved answer for one source `UV`, mirroring the per-channel sampled
/// `UV`s and the shared radial intensity the reference exposes.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::chromatic_aberration`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChromaticAberrationResult {
    /// The three per-channel sampled `UV`s `[red, green, blue]`, each clamped
    /// into the `[0, 1]` texture domain, matching
    /// [`ChromaParams::sample_uvs`](prism_render_architecture::particle::chromatic_aberration::ChromaParams::sample_uvs).
    pub sample_uvs: [[f32; 2]; 3],
    /// The shared radial intensity at the sampled radius, matching
    /// [`ChromaParams::radial_intensity`](prism_render_architecture::particle::chromatic_aberration::ChromaParams::radial_intensity).
    pub radial_intensity: f32,
}

/// Evaluates the `CPU` golden for one source `UV` under `params`, delegating the
/// per-channel samples to the reference
/// [`ChromaParams::sample_uvs`](prism_render_architecture::particle::chromatic_aberration::ChromaParams::sample_uvs)
/// and the radial intensity to
/// [`ChromaParams::radial_intensity`](prism_render_architecture::particle::chromatic_aberration::ChromaParams::radial_intensity)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::chromatic_aberration`；无第三方引擎源码或衍生代码。
#[must_use]
pub fn golden(
    params: &ChromaParams,
    query: &ChromaticAberrationQuery,
) -> ChromaticAberrationResult {
    let uv = query.uv;
    let dx = uv[0] - params.center[0];
    let dy = uv[1] - params.center[1];
    let r = (dx * dx + dy * dy).sqrt();
    ChromaticAberrationResult {
        sample_uvs: params.sample_uvs(uv),
        radial_intensity: params.radial_intensity(r),
    }
}

/// `repr(C)` `std430` layout of the dispatch meta: the source-`UV` count plus
/// three pad words to fill a `16`-byte, `std140`-aligned uniform struct matching
/// `Meta` in [`CHROMATIC_ABERRATION_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMeta {
    /// Number of source `UV`s in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one source `UV`, matching the `WGSL` `Query`
/// struct. Two trailing pad lanes keep the slot `16`-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// The source `UV` coordinate.
    uv: [f32; 2],
    /// Pad lane after `uv`.
    pad0: f32,
    /// Pad lane after `pad0`.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one [`ChromaticAberrationQuery`] into its `std430` image.
    fn new(query: &ChromaticAberrationQuery) -> GpuQuery {
        GpuQuery {
            uv: query.uv,
            pad0: 0.0,
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the three per-channel sampled `UV`s, the shared radial intensity and one pad
/// lane filling two `vec4` slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Red-channel sampled `UV`.
    r_uv: [f32; 2],
    /// Green-channel sampled `UV`.
    g_uv: [f32; 2],
    /// Blue-channel sampled `UV`.
    b_uv: [f32; 2],
    /// Shared radial intensity at the sampled radius.
    radial: f32,
    /// Padding lane.
    pad: f32,
}

/// Decodes one packed [`GpuResult`] into the public [`ChromaticAberrationResult`].
fn decode_result(raw: &GpuResult) -> ChromaticAberrationResult {
    ChromaticAberrationResult {
        sample_uvs: [raw.r_uv, raw.g_uv, raw.b_uv],
        radial_intensity: raw.radial,
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

/// A compiled, reusable `chromatic`-aberration compute pipeline, twinning the
/// `CPU` golden
/// [`chromatic_aberration`](prism_render_architecture::particle::chromatic_aberration).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::chromatic_aberration`；无第三方引擎源码或衍生代码。
pub struct GpuChromaticAberration {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuChromaticAberration {
    /// Compiles the `chromatic`-aberration kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::chromatic_aberration`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuChromaticAberration {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_chromatic_aberration"),
            source: ShaderSource::Wgsl(CHROMATIC_ABERRATION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Uniform),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuChromaticAberration {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every source `UV` in `queries` under the shared `params` and
    /// returns one [`ChromaticAberrationResult`] per input, in order.
    ///
    /// Each result equals the reference
    /// [`ChromaParams::sample_uvs`](prism_render_architecture::particle::chromatic_aberration::ChromaParams::sample_uvs)
    /// and
    /// [`ChromaParams::radial_intensity`](prism_render_architecture::particle::chromatic_aberration::ChromaParams::radial_intensity)
    /// answers to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::chromatic_aberration`；无第三方引擎源码或衍生代码。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        params: &ChromaParams,
        queries: &[ChromaticAberrationQuery],
    ) -> Vec<ChromaticAberrationResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let meta = GpuMeta {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let meta_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_meta"),
            contents: bytemuck::bytes_of(&meta),
            usage: BufferUsages::UNIFORM,
        });

        // The ChromaParams block is packed field for field by the reference
        // `to_std430`, so the device uniform sees exactly the host byte layout.
        let chroma_bytes = params.to_std430();
        let chroma_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_params"),
            contents: &chroma_bytes,
            usage: BufferUsages::UNIFORM,
        });

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: meta_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: chroma_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_chromatic_aberration_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_chromatic_aberration_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per source UV, flattened to a 1-D dispatch.
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
