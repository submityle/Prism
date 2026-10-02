//! `wgpu` compute twin of the `subsurface`-scattering (`SSS`) `wrap` lighting
//! gold standard for `Prism` translucent particles
//! ([`sss_wrap`](prism_render_architecture::particle::sss_wrap), particle design
//! §16-21).
//!
//! The `CPU` golden
//! [`sss_wrap`](prism_render_architecture::particle::sss_wrap) owns a
//! transcendental-free `subsurface` model thin translucent media (back-lit
//! smoke, wax, skin-like sprites, foliage cards) need because light *wraps*
//! around the shaded point and re-emerges on the shadowed side, tinted by the
//! wavelength-dependent scattering distance. It is built from four pieces: the
//! scalar `wrap` diffuse
//! ([`wrap_ndotl`](prism_render_architecture::particle::sss_wrap::wrap_ndotl)),
//! the per-channel `RGB` scatter `染色`
//! ([`scatter_color`](prism_render_architecture::particle::sss_wrap::scatter_color)),
//! the thickness-driven back transmission
//! ([`thickness_transmission`](prism_render_architecture::particle::sss_wrap::thickness_transmission))
//! and the combining
//! [`SssParams::evaluate`](prism_render_architecture::particle::sss_wrap::SssParams::evaluate).
//!
//! [`GpuSssWrap`] is the on-device twin: one thread per query reproduces the
//! same closed form branch for branch, so a passing real-device parity test is
//! direct evidence the ported kernel evaluates the same `subsurface` response
//! the reference does — the same guarded `wrap` denominator, the same per
//! channel scatter broadening and the same integer-power back lobe — not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! The whole [`SssParams::evaluate`](prism_render_architecture::particle::sss_wrap::SssParams::evaluate)
//! sample is reproduced per query: the three-channel wrapped diffuse (the scalar
//! [`wrap_ndotl`](prism_render_architecture::particle::sss_wrap::wrap_ndotl) term
//! broadened per channel by
//! [`scatter_color`](prism_render_architecture::particle::sss_wrap::scatter_color)
//! and lifted by the ambient floor) and the scalar back transmission from
//! [`thickness_transmission`](prism_render_architecture::particle::sss_wrap::thickness_transmission).
//! The private integer power `ipow` (a `for`-loop multiply chain, never a `pow`)
//! is exercised transitively inside the transmission lobe, and the reference's
//! `smoothstep` soft edge is reproduced by its hand-expanded `3 * t^2 - 2 * t^3`
//! polynomial since the view lobe always runs on the fixed `0..=1` edge
//! interval.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `+ - * /`, one `sqrt` for the direction normalize and `bitcast` to
//! unpack the packed parameter words — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry, no `smoothstep` builtin and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The only loop is the bounded `ipow` multiply chain over the integer
//! `transmission_power`, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt` per direction normalize, so `CPU` and `GPU` evaluate the same
//! closed-form algebra in the same associativity. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The parity
//! test therefore asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`)
//! on every continuous `f32` lane, while the integer `transmission_power` lobe
//! count is bit-exact because it is an integer multiply chain.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sss_wrap`；
//! standard `wrap`-diffuse `subsurface` approximation plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::sss_wrap::{SssParams, SssSample};
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

/// The portable core-`WGSL` `subsurface`-`wrap` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`sss_wrap`](prism_render_architecture::particle::sss_wrap) branch for
/// branch; see the module documentation for the algorithm.
const SSS_WRAP_WGSL: &str = r#"
// Subsurface-wrap twin: one thread per query reproduces the three-channel
// wrapped diffuse (wrap_ndotl broadened per channel by scatter_color and lifted
// by the ambient floor) and the scalar thickness-driven back transmission of the
// CPU golden particle::sss_wrap, branch for branch. It uses only the portable
// core-WGSL subset (min/max/clamp and + - * / plus one sqrt per direction
// normalize and bitcast to unpack the packed parameter words), replaces the
// reference pow with a bounded ipow multiply chain and the reference smoothstep
// with its hand-expanded 3*t^2 - 2*t^3 polynomial, and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::sss_wrap; no third-party
// engine source or derived code.

// Generic denominator / soft-edge guard below which a division collapses to a
// hard step, so no divide-by-zero can produce a NaN. Matches the reference
// MIN_EDGE; the compare rule used instead of an f32 == / !=.
const MIN_EDGE: f32 = 1.0e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Surface normal; the fourth lane carries the medium thickness.
    normal: vec3<f32>,
    thickness: f32,
    // Light direction; a pad lane follows.
    light: vec3<f32>,
    pad0: f32,
    // View direction; a pad lane follows.
    view: vec3<f32>,
    pad1: f32,
    // SssParams::to_std430 image: wrap, scatter_r, scatter_g, scatter_b,
    // thickness_scale, transmission_power (raw u32), ambient and one pad word.
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    p6: u32,
    p7: u32,
}

struct Result {
    // Per-channel RGB wrapped diffuse plus scatter and ambient; the fourth lane
    // carries the scalar back transmission.
    diffuse: vec3<f32>,
    transmission: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Dot product of two vec3 values, mirroring the reference dot3 associativity.
fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Unit vector along v, or the zero vector for a (near-)zero input so no NaN can
// leak, mirroring the reference normalize3 (squared length floored at MIN_EDGE).
fn normalize3(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot3(v, v);
    if (len_sq < MIN_EDGE) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv_len = 1.0 / sqrt(len_sq);
    return vec3<f32>(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}

// Raises base to the integer power exp by repeated multiplication, mirroring the
// reference ipow: exp == 0 yields 1.0. Replaces the reference pow.
fn ipow(base: f32, exp: u32) -> f32 {
    var acc: f32 = 1.0;
    for (var i: u32 = 0u; i < exp; i = i + 1u) {
        acc = acc * base;
    }
    return acc;
}

// Hand-expanded smoothstep on the fixed 0..=1 edge interval, mirroring the
// reference smoothstep(0, 1, x) (span == 1, never the degenerate branch): the
// t*t*(3 - 2*t) interpolation of the clamped input. Replaces the smoothstep
// builtin.
fn smoothstep01(x: f32) -> f32 {
    let t = clamp(x, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// wrap-diffuse NdotL, mirroring the reference wrap_ndotl: bends the Lambert
// terminator past zero. The denominator is guarded by MIN_EDGE so a degenerate
// wrap <= -1 cannot divide by zero.
fn wrap_ndotl(n_dot_l: f32, wrap: f32) -> f32 {
    let denom = max(1.0 + wrap, MIN_EDGE);
    return clamp((n_dot_l + wrap) / denom, 0.0, 1.0);
}

// Per-channel RGB scatter dyeing of the wrapped-in shadow light, mirroring the
// reference scatter_color: each channel uses a wider effective wrap
// (wrap + radius) and returns the extra colored light beyond the hard
// terminator, scaled by the channel's scatter radius.
fn scatter_color(n_dot_l: f32, wrap: f32, scatter_rgb: vec3<f32>) -> vec3<f32> {
    let core = clamp(n_dot_l, 0.0, 1.0);
    let radius_r = max(scatter_rgb.x, 0.0);
    let radius_g = max(scatter_rgb.y, 0.0);
    let radius_b = max(scatter_rgb.z, 0.0);
    let wrapped_r = wrap_ndotl(n_dot_l, wrap + radius_r);
    let wrapped_g = wrap_ndotl(n_dot_l, wrap + radius_g);
    let wrapped_b = wrap_ndotl(n_dot_l, wrap + radius_b);
    let extra_r = max(wrapped_r - core, 0.0);
    let extra_g = max(wrapped_g - core, 0.0);
    let extra_b = max(wrapped_b - core, 0.0);
    return vec3<f32>(radius_r * extra_r, radius_g * extra_g, radius_b * extra_b);
}

// Thickness-driven back transmission, mirroring the reference
// thickness_transmission: thin regions glow brightest via the rational
// attenuation 1 / (1 + thickness_scale * thickness), gated by the back-facing
// view lobe shaped by smoothstep and sharpened by the integer transmission_power.
fn thickness_transmission(
    thickness: f32,
    light_dir: vec3<f32>,
    view_dir: vec3<f32>,
    thickness_scale: f32,
    transmission_power: u32,
) -> f32 {
    let l = normalize3(light_dir);
    let v = normalize3(view_dir);
    let back = clamp(-dot3(v, l), 0.0, 1.0);
    let lobe = ipow(smoothstep01(back), transmission_power);
    let t = max(thickness, 0.0);
    let k = max(thickness_scale, 0.0);
    let attenuation = 1.0 / (1.0 + k * t);
    return attenuation * lobe;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Unpack the SssParams::to_std430 image: five f32 lanes and the raw u32
    // transmission_power, mirroring the host packing.
    let wrap = bitcast<f32>(q.p0);
    let scatter_rgb = vec3<f32>(bitcast<f32>(q.p1), bitcast<f32>(q.p2), bitcast<f32>(q.p3));
    let thickness_scale = bitcast<f32>(q.p4);
    let transmission_power = q.p5;
    let ambient = bitcast<f32>(q.p6);

    let n = normalize3(q.normal);
    let l = normalize3(q.light);
    let n_dot_l = dot3(n, l);
    let wrapped = wrap_ndotl(n_dot_l, wrap);
    let scatter = scatter_color(n_dot_l, wrap, scatter_rgb);
    let diffuse = vec3<f32>(
        max(wrapped + scatter.x + ambient, 0.0),
        max(wrapped + scatter.y + ambient, 0.0),
        max(wrapped + scatter.z + ambient, 0.0),
    );
    let transmission =
        thickness_transmission(q.thickness, q.light, q.view, thickness_scale, transmission_power);

    var out: Result;
    out.diffuse = diffuse;
    out.transmission = transmission;
    results[idx] = out;
}
"#;

/// One `subsurface`-`wrap` shading query: the directions `normal`, `light_dir`
/// and `view_dir`, the medium `thickness` and the shared [`SssParams`] tunables
/// — the same inputs the reference
/// [`SssParams::evaluate`](prism_render_architecture::particle::sss_wrap::SssParams::evaluate)
/// consumes.
///
/// The directions are normalized on device exactly as the reference normalizes
/// them, so unit-length inputs are not required.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sss_wrap`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SssWrapQuery {
    /// Surface normal (normalized on device).
    pub normal: [f32; 3],
    /// Light direction (normalized on device).
    pub light_dir: [f32; 3],
    /// View direction (normalized on device).
    pub view_dir: [f32; 3],
    /// Medium thickness feeding the back-transmission attenuation.
    pub thickness: f32,
    /// Shared `subsurface` `wrap` tunables (`wrap`, scatter radii,
    /// `thickness_scale`, `transmission_power`, `ambient`).
    pub params: SssParams,
}

impl SssWrapQuery {
    /// Builds a query from the directions, the medium `thickness` and the shared
    /// [`SssParams`].
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::sss_wrap`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(
        normal: [f32; 3],
        light_dir: [f32; 3],
        view_dir: [f32; 3],
        thickness: f32,
        params: SssParams,
    ) -> SssWrapQuery {
        SssWrapQuery {
            normal,
            light_dir,
            view_dir,
            thickness,
            params,
        }
    }
}

/// One evaluated `subsurface` shading sample, mirroring the reference
/// [`SssSample`](prism_render_architecture::particle::sss_wrap::SssSample).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sss_wrap`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SssWrapSample {
    /// Per-channel `RGB` wrapped diffuse plus scatter `染色` and ambient,
    /// matching
    /// [`SssSample::diffuse`](prism_render_architecture::particle::sss_wrap::SssSample::diffuse).
    pub diffuse: [f32; 3],
    /// Scalar back-transmission glow in `0..=1`, matching
    /// [`SssSample::transmission`](prism_render_architecture::particle::sss_wrap::SssSample::transmission).
    pub transmission: f32,
}

/// Evaluates the `CPU` golden for one query, delegating to the reference
/// [`SssParams::evaluate`](prism_render_architecture::particle::sss_wrap::SssParams::evaluate)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sss_wrap`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &SssWrapQuery) -> SssWrapSample {
    let sample: SssSample = query.params.evaluate(
        query.normal,
        query.light_dir,
        query.view_dir,
        query.thickness,
    );
    SssWrapSample {
        diffuse: sample.diffuse,
        transmission: sample.transmission,
    }
}

/// `repr(C)` `std430` layout of one packed query: three `vec4` slots holding
/// `(normal.xyz, thickness)`, `(light.xyz, pad)` and `(view.xyz, pad)` followed
/// by the eight-word [`SssParams::to_std430`] image — `80` bytes, each `vec3` on
/// its `16`-byte-aligned slot exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Surface normal.
    normal: [f32; 3],
    /// Medium thickness, packed in the fourth lane of the first slot.
    thickness: f32,
    /// Light direction.
    light: [f32; 3],
    /// Padding lane after the light direction.
    pad0: f32,
    /// View direction.
    view: [f32; 3],
    /// Padding lane after the view direction.
    pad1: f32,
    /// The eight-word [`SssParams::to_std430`] image (two `vec4` slots).
    params: [u32; 8],
}

impl GpuQuery {
    /// Packs one query into its `std430` image, reusing the reference
    /// [`SssParams::to_std430`] layout for the parameter words.
    fn new(query: &SssWrapQuery) -> GpuQuery {
        GpuQuery {
            normal: query.normal,
            thickness: query.thickness,
            light: query.light_dir,
            pad0: 0.0,
            view: query.view_dir,
            pad1: 0.0,
            params: query.params.to_std430(),
        }
    }
}

/// `repr(C)` `std430` layout of one result: a single `vec4` slot holding
/// `(diffuse.xyz, transmission)` — `16` bytes matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Per-channel `RGB` wrapped diffuse plus scatter and ambient.
    diffuse: [f32; 3],
    /// Scalar back transmission.
    transmission: f32,
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

/// Decodes one packed [`GpuResult`] into the public [`SssWrapSample`].
fn decode_result(raw: &GpuResult) -> SssWrapSample {
    SssWrapSample {
        diffuse: raw.diffuse,
        transmission: raw.transmission,
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

/// A compiled, reusable `subsurface`-`wrap` compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::sss_wrap`；
/// no third-party engine source or derived code.
pub struct GpuSssWrap {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSssWrap {
    /// Compiles the `subsurface`-`wrap` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::sss_wrap`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSssWrap {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sss_wrap"),
            source: ShaderSource::Wgsl(SSS_WRAP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sss_wrap_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sss_wrap_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sss_wrap_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSssWrap {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`SssWrapSample`] per input,
    /// in order.
    ///
    /// Each result equals the reference
    /// [`SssParams::evaluate`](prism_render_architecture::particle::sss_wrap::SssParams::evaluate)
    /// answer to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::sss_wrap`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[SssWrapQuery]) -> Vec<SssWrapSample> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sss_wrap_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sss_wrap_output"),
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
            label: Some("prism_volumetric_sss_wrap_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sss_wrap_bind_group"),
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
            label: Some("prism_volumetric_sss_wrap_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sss_wrap_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sss_wrap_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
