//! `wgpu` compute twin of the measured-conductor `Fresnel` presets
//! ([`metal`](prism_render_architecture::reference_pt::metal) +
//! [`conductor`](prism_render_architecture::reference_pt::conductor)).
//!
//! A real metal's colour is set by its complex index of refraction `eta + i*k`
//! sampled per red/green/blue channel. Looking those constants up by hand is
//! error-prone, so the reference curates the canonical measured triplets for
//! the six conductors an artist actually reaches for — gold, silver, copper,
//! aluminium, iron and chromium — and feeds them to the exact unpolarized
//! conductor `Fresnel` equations. At normal incidence the result is the metal's
//! base colour; it rises toward one at grazing with the per-channel curvature
//! that gives real metals their angular hue drift.
//!
//! This module is the on-device twin of that lookup-plus-`Fresnel` path.
//! [`GpuMetalFresnel`] evaluates one query per thread: it indexes the baked
//! complex-index table by a metal id and evaluates the exact
//! `fresnel_conductor` closed form, reproducing the reference's arithmetic —
//! only `sqrt`, products, quotients, clamps and guarded selects — so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same reflectance the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each thread reads one [`MetalFresnelQuery`] — a metal id in `0..6` and the
//! cosine of the incidence angle — and writes one [`MetalFresnelResult`]
//! holding the per-channel reflectance and a `valid` flag. The kernel mirrors
//! [`Metal::complex_ior`](prism_render_architecture::reference_pt::metal::Metal::complex_ior)
//! (the baked `eta`/`k` table) composed with
//! [`fresnel_conductor`](prism_render_architecture::reference_pt::conductor::fresnel_conductor),
//! itself three independent evaluations of the private per-channel
//! `fresnel_conductor_channel`.
//!
//! # What stays on the host
//!
//! Nothing of the closed form stays on the host: the whole table lookup and the
//! three-channel `Fresnel` evaluation run on the device. The host only chooses
//! which metal and which angle to query and marshals the batch, so a storage
//! buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Each channel threads through two real square roots and a pair of guarded
//! quotients, so the `CPU` and `GPU` are not bit-exact: a `GPU` `sqrt` or
//! divide may land a few units in the last place from the scalar reference. The
//! parity test asserts each channel within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (relative floored at `1e-6`), tight enough to catch a
//! wrong port yet loose enough to admit a legal last-place difference. The
//! `valid` flag — set when the metal id is in range — is compared exactly.
//!
//! # Degenerate inputs
//!
//! A metal id of `6` or above is out of range: the kernel reports a zero
//! reflectance on every channel and `valid = 0`. Every division inside the
//! per-channel `Fresnel` is guarded by an ordered comparison (`select`), so the
//! grazing limit (where the `s`-polarized denominator would vanish) falls back
//! to a reflectance of one with no `NaN`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `clamp`,
//! `min`, `max`, `select`, `+ - * /` and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round` and no `f32` remainder, and no `f64`/`u64`/`u16`/`i64`/`i16`. It runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::metal` 与 `prism_render_architecture::reference_pt::conductor`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
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

/// Number of curated metal presets; a metal id at or above this is out of range
/// and reports `valid = 0`.
pub const METAL_COUNT: u32 = 6;

/// The portable core-`WGSL` conductor-`Fresnel`-preset kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`Metal::complex_ior`](prism_render_architecture::reference_pt::metal::Metal::complex_ior)
/// table composed with
/// [`fresnel_conductor`](prism_render_architecture::reference_pt::conductor::fresnel_conductor);
/// see the module documentation for the algorithm.
const METAL_FRESNEL_PRESET_WGSL: &str = r#"
// Measured-conductor Fresnel-preset twin: one thread looks up a metal's baked
// complex index of refraction `eta + i*k` by id and evaluates the exact
// per-channel unpolarized conductor Fresnel reflectance, mirroring the CPU
// golden `reference_pt::metal::Metal::complex_ior` composed with
// `reference_pt::conductor::fresnel_conductor`. Only sqrt, products, quotients,
// clamps and guarded selects appear; no transcendental call is needed.
//
// Provenance: 孪生自本仓 prism_render_architecture::reference_pt::metal 与
// prism_render_architecture::reference_pt::conductor；无第三方引擎源码或衍生代码。

// Number of curated metal presets; an id at or above this is out of range.
const METAL_COUNT: u32 = 6u;

// Per-channel real index eta for each preset, indexed by metal id:
// 0 Gold, 1 Silver, 2 Copper, 3 Aluminium, 4 Iron, 5 Chromium.
const PRESET_ETA: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(0.143, 0.375, 1.442),
    vec3<f32>(0.155, 0.116, 0.138),
    vec3<f32>(0.200, 0.924, 1.102),
    vec3<f32>(1.345, 0.965, 0.617),
    vec3<f32>(2.911, 2.950, 2.580),
    vec3<f32>(3.181, 3.079, 2.392),
);

// Per-channel extinction coefficient k for each preset, same id order.
const PRESET_K: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>(3.983, 2.386, 1.603),
    vec3<f32>(4.818, 3.122, 2.146),
    vec3<f32>(3.912, 2.448, 2.137),
    vec3<f32>(7.474, 6.399, 5.303),
    vec3<f32>(3.089, 2.931, 2.767),
    vec3<f32>(3.329, 3.340, 3.148),
);

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Preset id in 0..6 selecting the metal; 6 or above is out of range.
    metal_id: u32,
    // Cosine of the incidence angle, clamped to [0, 1] before use.
    cos_theta: f32,
    pad0: u32,
    pad1: u32,
}

struct Reflect {
    // Per-channel conductor Fresnel reflectance.
    r: f32,
    g: f32,
    b: f32,
    // 1 when metal_id < METAL_COUNT, else 0 (and the channels are zero).
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Reflect>;

// The exact unpolarized conductor Fresnel reflectance for one wavelength
// channel, given the incidence cosine and the channel's complex index eta+i*k.
// Mirrors the reference `fresnel_conductor_channel`: only sqrt, products and
// guarded quotients, so no transcendental call is needed.
fn fresnel_conductor_channel(cos0: f32, eta: f32, k: f32) -> f32 {
    let ci = clamp(cos0, 0.0, 1.0);
    let cos2 = ci * ci;
    let sin2 = 1.0 - cos2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - sin2;
    let a2b2 = sqrt(max(t0 * t0 + 4.0 * eta2 * k2, 0.0));
    let a = sqrt(max(0.5 * (a2b2 + t0), 0.0));
    // s-polarized reflectance (grazing denominator guarded to a mirror).
    let t1 = a2b2 + cos2;
    let t2 = 2.0 * a * ci;
    let denom_s = t1 + t2;
    let r_s = select(1.0, (t1 - t2) / denom_s, denom_s > 0.0);
    // p-polarized reflectance, expressed relative to r_s.
    let t3 = cos2 * a2b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let denom_p = t3 + t4;
    let r_p = select(r_s, r_s * (t3 - t4) / denom_p, denom_p > 0.0);
    return clamp(0.5 * (r_s + r_p), 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Reflect;
    out.r = 0.0;
    out.g = 0.0;
    out.b = 0.0;
    out.valid = 0u;

    if (q.metal_id < METAL_COUNT) {
        out.valid = 1u;
        let eta = PRESET_ETA[q.metal_id];
        let k = PRESET_K[q.metal_id];
        out.r = fresnel_conductor_channel(q.cos_theta, eta.x, k.x);
        out.g = fresnel_conductor_channel(q.cos_theta, eta.y, k.y);
        out.b = fresnel_conductor_channel(q.cos_theta, eta.z, k.z);
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`METAL_FRESNEL_PRESET_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the metal id and the incidence cosine, padded to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Preset id in `0..6`.
    metal_id: u32,
    /// Incidence cosine.
    cos_theta: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Reflect`
/// struct: the three reflectance channels and the `valid` flag, a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Red reflectance.
    r: f32,
    /// Green reflectance.
    g: f32,
    /// Blue reflectance.
    b: f32,
    /// Validity flag (`1` or `0`).
    valid: u32,
}

/// One query for the conductor-`Fresnel`-preset twin: a metal id selecting the
/// curated complex index, and the cosine of the incidence angle.
///
/// `metal_id` selects the preset — `0` gold, `1` silver, `2` copper, `3`
/// aluminium, `4` iron, `5` chromium — and anything at or above
/// [`METAL_COUNT`] is out of range. `cos_theta` is clamped to `[0, 1]` by the
/// kernel. The host enqueues one [`MetalFresnelQuery`] per reflectance it needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetalFresnelQuery {
    /// Preset id in `0..6`.
    pub metal_id: u32,
    /// Cosine of the incidence angle.
    pub cos_theta: f32,
}

impl MetalFresnelQuery {
    /// Builds a query from a metal id and an incidence cosine.
    #[must_use]
    pub const fn new(metal_id: u32, cos_theta: f32) -> MetalFresnelQuery {
        MetalFresnelQuery {
            metal_id,
            cos_theta,
        }
    }
}

/// One resolved query of the conductor-`Fresnel`-preset twin: the per-channel
/// reflectance and the validity flag.
///
/// `reflectance` is
/// [`fresnel_conductor`](prism_render_architecture::reference_pt::conductor::fresnel_conductor)
/// evaluated at the selected metal's complex index; `valid` is `1` when the
/// metal id is in range, else `0` (in which case `reflectance` is zero).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetalFresnelResult {
    /// Per-channel conductor `Fresnel` reflectance `[r, g, b]`.
    pub reflectance: [f32; 3],
    /// `1` when the metal id is in range, else `0`.
    pub valid: u32,
}

/// Encodes one [`MetalFresnelQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MetalFresnelQuery) -> GpuQuery {
    GpuQuery {
        metal_id: q.metal_id,
        cos_theta: q.cos_theta,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MetalFresnelResult`].
fn decode_result(raw: &GpuResult) -> MetalFresnelResult {
    MetalFresnelResult {
        reflectance: [raw.r, raw.g, raw.b],
        valid: raw.valid,
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

/// A compiled, reusable conductor-`Fresnel`-preset compute pipeline, twinning
/// the `CPU` golden
/// [`Metal::complex_ior`](prism_render_architecture::reference_pt::metal::Metal::complex_ior)
/// table composed with
/// [`fresnel_conductor`](prism_render_architecture::reference_pt::conductor::fresnel_conductor).
pub struct GpuMetalFresnel {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMetalFresnel {
    /// Compiles the conductor-`Fresnel`-preset kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMetalFresnel {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_metal_fresnel"),
            source: ShaderSource::Wgsl(METAL_FRESNEL_PRESET_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_metal_fresnel_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_metal_fresnel_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_metal_fresnel_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMetalFresnel {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one
    /// [`MetalFresnelResult`] per input, in order.
    ///
    /// The reflectances match the reference to within the tolerance documented
    /// on this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MetalFresnelQuery],
    ) -> Vec<MetalFresnelResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_metal_fresnel_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_metal_fresnel_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_metal_fresnel_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_metal_fresnel_bind_group"),
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
            label: Some("prism_volumetric_metal_fresnel_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_metal_fresnel_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_metal_fresnel_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
