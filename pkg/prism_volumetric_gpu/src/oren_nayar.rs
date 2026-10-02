//! `wgpu` compute twin of the rough-diffuse `BRDF` gold standard for `Prism`
//! particle shading
//! ([`oren_nayar`](prism_render_architecture::particle::oren_nayar), particle
//! design §17 `PBR` diffuse, §18 stylized base).
//!
//! The `CPU` golden
//! [`oren_nayar`](prism_render_architecture::particle::oren_nayar) owns two
//! transcendental-free rough-diffuse closures that real rough dielectrics (dust,
//! chalk, dry smoke, clay-like debris) need because they retro-reflect toward
//! the silhouette instead of darkening uniformly like Lambert:
//!
//! 1. The trig-free Oren-Nayar form
//!    ([`oren_nayar`](prism_render_architecture::particle::oren_nayar::oren_nayar),
//!    with its coefficient split
//!    [`oren_nayar_coeffs`](prism_render_architecture::particle::oren_nayar::oren_nayar_coeffs)):
//!    with the clamped cosines `NoL`, `NoV` and `LoV = dot(light, view)`, the
//!    `Fujii`/`Gotanda` direction-only rewrite reflects
//!    `albedo / pi * NoL * (A + B * s / t)`, where `s = LoV - NoL * NoV`,
//!    `t = if s > 0 { max(NoL, NoV) } else { 1 }`,
//!    `A = 1 - 0.5 * sigma^2 / (sigma^2 + 0.33)` and
//!    `B = 0.45 * sigma^2 / (sigma^2 + 0.09)`.
//! 2. The Burley / Disney roughness-aware diffuse
//!    ([`burley_diffuse`](prism_render_architecture::particle::oren_nayar::burley_diffuse)):
//!    with the half vector `H = normalize(light + view)`, `LoH = dot(light, H)`
//!    and `FD90 = 0.5 + 2 * roughness * LoH^2`, the reflectance is
//!    `albedo / pi * (1 + (FD90 - 1) * (1 - NoL)^5) * (1 + (FD90 - 1) * (1 - NoV)^5)`.
//!
//! [`GpuOrenNayar`] is the on-device twin: one thread per shading query
//! reproduces both closures branch for branch, so a passing real-device parity
//! test is direct evidence the ported kernel evaluates the same closed form the
//! reference does — the same `sigma = 0` collapse to Lambert, the same
//! back-lit / back-facing gate to zero, and the same `s <= 0` denominator
//! branch — not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Both reflected values the reference exposes are reproduced per query: the
//! Oren-Nayar reflected radiance
//! ([`oren_nayar`](prism_render_architecture::particle::oren_nayar::oren_nayar))
//! and the Burley diffuse reflectance
//! ([`burley_diffuse`](prism_render_architecture::particle::oren_nayar::burley_diffuse)),
//! computed in one dispatch from the same packed directions. The coefficient
//! split
//! ([`oren_nayar_coeffs`](prism_render_architecture::particle::oren_nayar::oren_nayar_coeffs))
//! is exercised transitively inside the Oren-Nayar path. The reference's two
//! discrete branches are mirrored: the back-lit / back-facing gate
//! (`NoL <= 0` or `NoV <= 0`) returns zero, and the Oren-Nayar denominator
//! collapses to one when the azimuthal term `s` is non-positive, with a tiny
//! floor on `t` guarding the `inf * 0` corner.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `+ - * /`, the `dot` builtin and the one `sqrt` the
//! reference itself uses to normalize the half vector — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The fifth
//! power is an inline integer multiply chain, not `pow`. There is no loop: each
//! thread performs a fixed, bounded sequence of arithmetic, so the kernel
//! provably terminates.
//!
//! # Correctness model
//!
//! Each closure is a fixed, non-reorderable sequence of multiplies, adds,
//! divides and (for Burley) one `sqrt`, so `CPU` and `GPU` evaluate the same
//! closed-form algebra in the same associativity. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few units in the last place. The
//! parity test therefore asserts `abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! tight enough to catch a genuinely wrong port (a dropped branch, a swapped
//! coefficient, a wrong clamp) yet loose enough to admit legal fused
//! multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::oren_nayar`；
//! standard trig-free Oren-Nayar (`Fujii`/`Gotanda`) and Burley / Disney
//! diffuse plus `wgpu` compute dispatch; no third-party engine source or
//! derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::oren_nayar::{burley_diffuse, oren_nayar};
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

/// The portable core-`WGSL` rough-diffuse kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`oren_nayar`](prism_render_architecture::particle::oren_nayar) branch for
/// branch; see the module documentation for the algorithm.
const OREN_NAYAR_WGSL: &str = r#"
// Rough-diffuse twin: one thread per shading query reproduces both the trig-free
// Oren-Nayar reflected radiance and the Burley / Disney diffuse reflectance of
// the CPU golden particle::oren_nayar, branch for branch. It uses only the
// portable core-WGSL subset (min/max/clamp/abs and + - * / plus the dot builtin
// and one sqrt for the half-vector normalize) and takes no optional feature, so
// it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::oren_nayar; no
// third-party engine source or derived code.

// Reciprocal of pi, matching the reference `INV_PI` (core::f32::consts::FRAC_1_PI);
// the crate is transcendental-free so pi is spelled as a constant.
const INV_PI: f32 = 0.31830988618379069;
// Classic Oren-Nayar denominator offsets and leading factors, matching the
// reference `A_OFFSET`, `B_OFFSET`, `A_SCALE`, `B_SCALE`.
const A_OFFSET: f32 = 0.33;
const B_OFFSET: f32 = 0.09;
const A_SCALE: f32 = 0.5;
const B_SCALE: f32 = 0.45;
// Minimum Oren-Nayar `s / t` denominator, matching the reference `MIN_T`: guards
// the `inf * 0 = NaN` corner where s > 0 yet max(NoL, NoV) is numerically zero.
const MIN_T: f32 = 1.0e-6;
// Squared-length floor for normalize_or_zero, matching the reference
// `EPS_LEN_SQ`, so the half-vector normalize never yields NaN.
const EPS_LEN_SQ: f32 = 1.0e-12;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Surface normal; the fourth lane carries the Oren-Nayar roughness sigma.
    normal: vec3<f32>,
    sigma: f32,
    // Light direction; the fourth lane carries the Burley perceptual roughness.
    light: vec3<f32>,
    roughness: f32,
    // View direction; a pad lane follows.
    view: vec3<f32>,
    pad0: f32,
    // Linear-RGB albedo; a pad lane follows.
    albedo: vec3<f32>,
    pad1: f32,
}

struct Result {
    // Oren-Nayar reflected radiance and one pad lane.
    oren: vec3<f32>,
    pad0: f32,
    // Burley diffuse reflectance and one pad lane.
    burley: vec3<f32>,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Unit vector along v, or the zero vector when v is numerically zero, mirroring
// the reference `Vec3::normalize_or_zero` (squared length floored at EPS_LEN_SQ).
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Trig-free Oren-Nayar reflected radiance, mirroring the reference `oren_nayar`
// (which folds in `oren_nayar_coeffs`). Back-lit or back-facing geometry returns
// zero; at sigma = 0 it reduces to Lambert albedo / pi * NoL.
fn oren_nayar_eval(
    normal: vec3<f32>,
    light: vec3<f32>,
    view: vec3<f32>,
    albedo: vec3<f32>,
    sigma: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(normal, light), 0.0);
    let n_dot_v = max(dot(normal, view), 0.0);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let sig = max(sigma, 0.0);
    let sigma_sq = sig * sig;
    let a = 1.0 - A_SCALE * sigma_sq / (sigma_sq + A_OFFSET);
    let b = B_SCALE * sigma_sq / (sigma_sq + B_OFFSET);
    let s = dot(light, view) - n_dot_l * n_dot_v;
    var t: f32 = 1.0;
    if (s > 0.0) {
        t = max(max(n_dot_l, n_dot_v), MIN_T);
    }
    let lobe = a + b * s / t;
    let scale = INV_PI * n_dot_l * max(lobe, 0.0);
    return albedo * scale;
}

// Burley / Disney roughness-aware diffuse reflectance, mirroring the reference
// `burley_diffuse`. The fifth power is an inline multiply chain, not pow. Back-lit
// or back-facing geometry returns zero.
fn burley_eval(
    normal: vec3<f32>,
    light: vec3<f32>,
    view: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(normal, light), 0.0);
    let n_dot_v = max(dot(normal, view), 0.0);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let hv = normalize_or_zero(light + view);
    let l_dot_h = max(dot(light, hv), 0.0);
    let r = max(roughness, 0.0);
    let fd90 = 0.5 + 2.0 * r * (l_dot_h * l_dot_h);
    let ml = 1.0 - n_dot_l;
    let mv = 1.0 - n_dot_v;
    let pow_l = ml * ml * ml * ml * ml;
    let pow_v = mv * mv * mv * mv * mv;
    let fl = 1.0 + (fd90 - 1.0) * pow_l;
    let fv = 1.0 + (fd90 - 1.0) * pow_v;
    return albedo * (INV_PI * fl * fv);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.oren = oren_nayar_eval(q.normal, q.light, q.view, q.albedo, q.sigma);
    out.pad0 = 0.0;
    out.burley = burley_eval(q.normal, q.light, q.view, q.albedo, q.roughness);
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One rough-diffuse shading query: the unit directions `normal`, `light` and
/// `view`, the linear-`RGB` `albedo`, the Oren-Nayar slope deviation `sigma`
/// (radians) and the Burley perceptual `roughness` — the same inputs the
/// reference
/// [`oren_nayar`](prism_render_architecture::particle::oren_nayar::oren_nayar)
/// and
/// [`burley_diffuse`](prism_render_architecture::particle::oren_nayar::burley_diffuse)
/// consume.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::oren_nayar`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrenNayarQuery {
    /// Surface normal (unit length).
    pub normal: Vec3,
    /// Light direction (unit length).
    pub light: Vec3,
    /// View direction (unit length).
    pub view: Vec3,
    /// Linear-`RGB` albedo.
    pub albedo: Vec3,
    /// Oren-Nayar slope standard deviation `sigma`, in radians.
    pub sigma: f32,
    /// Burley perceptual roughness in `0..=1`.
    pub roughness: f32,
}

impl OrenNayarQuery {
    /// Builds a query from the directions, albedo, `sigma` and `roughness`.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::oren_nayar`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub const fn new(
        normal: Vec3,
        light: Vec3,
        view: Vec3,
        albedo: Vec3,
        sigma: f32,
        roughness: f32,
    ) -> OrenNayarQuery {
        OrenNayarQuery {
            normal,
            light,
            view,
            albedo,
            sigma,
            roughness,
        }
    }
}

/// The resolved reflectances for one query: both the Oren-Nayar reflected
/// radiance and the Burley diffuse reflectance the reference exposes.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::oren_nayar`；
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrenNayarResult {
    /// Oren-Nayar reflected radiance, matching
    /// [`oren_nayar`](prism_render_architecture::particle::oren_nayar::oren_nayar).
    pub oren: Vec3,
    /// Burley diffuse reflectance, matching
    /// [`burley_diffuse`](prism_render_architecture::particle::oren_nayar::burley_diffuse).
    pub burley: Vec3,
}

/// Evaluates the `CPU` golden for one query, delegating to the reference
/// [`oren_nayar`](prism_render_architecture::particle::oren_nayar::oren_nayar)
/// and
/// [`burley_diffuse`](prism_render_architecture::particle::oren_nayar::burley_diffuse)
/// so the host side and the device twin are checked against the same source of
/// truth.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::oren_nayar`；
/// no third-party engine source or derived code.
#[must_use]
pub fn golden(query: &OrenNayarQuery) -> OrenNayarResult {
    OrenNayarResult {
        oren: oren_nayar(
            query.normal,
            query.light,
            query.view,
            query.albedo,
            query.sigma,
        ),
        burley: burley_diffuse(
            query.normal,
            query.light,
            query.view,
            query.albedo,
            query.roughness,
        ),
    }
}

/// `repr(C)` `std430` layout of one packed query: four `vec4` slots holding
/// `(normal.xyz, sigma)`, `(light.xyz, roughness)`, `(view.xyz, pad)` and
/// `(albedo.xyz, pad)` — `64` bytes, each `vec3` on its `16`-byte-aligned slot
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Surface normal.
    normal: [f32; 3],
    /// Oren-Nayar `sigma`, packed in the fourth lane of the first slot.
    sigma: f32,
    /// Light direction.
    light: [f32; 3],
    /// Burley roughness, packed in the fourth lane of the second slot.
    roughness: f32,
    /// View direction.
    view: [f32; 3],
    /// Padding lane after the view direction.
    pad0: f32,
    /// Linear-`RGB` albedo.
    albedo: [f32; 3],
    /// Padding lane after the albedo.
    pad1: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &OrenNayarQuery) -> GpuQuery {
        GpuQuery {
            normal: [query.normal.x, query.normal.y, query.normal.z],
            sigma: query.sigma,
            light: [query.light.x, query.light.y, query.light.z],
            roughness: query.roughness,
            view: [query.view.x, query.view.y, query.view.z],
            pad0: 0.0,
            albedo: [query.albedo.x, query.albedo.y, query.albedo.z],
            pad1: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: two `vec4` slots holding
/// `(oren.xyz, pad)` and `(burley.xyz, pad)` — `32` bytes matching the `WGSL`
/// `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Oren-Nayar reflected radiance.
    oren: [f32; 3],
    /// Padding lane.
    pad0: f32,
    /// Burley diffuse reflectance.
    burley: [f32; 3],
    /// Padding lane.
    pad1: f32,
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

/// A compiled, reusable rough-diffuse compute pipeline.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::oren_nayar`；
/// no third-party engine source or derived code.
pub struct GpuOrenNayar {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuOrenNayar {
    /// Compiles the rough-diffuse kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::oren_nayar`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuOrenNayar {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_oren_nayar"),
            source: ShaderSource::Wgsl(OREN_NAYAR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_oren_nayar_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_oren_nayar_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_oren_nayar_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuOrenNayar {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`OrenNayarResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference
    /// [`oren_nayar`](prism_render_architecture::particle::oren_nayar::oren_nayar)
    /// and
    /// [`burley_diffuse`](prism_render_architecture::particle::oren_nayar::burley_diffuse)
    /// answers to within the tolerance documented on this module. An empty input
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::oren_nayar`；
    /// no third-party engine source or derived code.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[OrenNayarQuery]) -> Vec<OrenNayarResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_oren_nayar_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_oren_nayar_output"),
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
            label: Some("prism_volumetric_oren_nayar_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_oren_nayar_bind_group"),
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
            label: Some("prism_volumetric_oren_nayar_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_oren_nayar_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_oren_nayar_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`OrenNayarResult`].
fn decode_result(raw: &GpuResult) -> OrenNayarResult {
    OrenNayarResult {
        oren: Vec3::new(raw.oren[0], raw.oren[1], raw.oren[2]),
        burley: Vec3::new(raw.burley[0], raw.burley[1], raw.burley[2]),
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
