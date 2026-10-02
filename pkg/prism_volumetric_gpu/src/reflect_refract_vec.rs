//! `wgpu` compute twin of the specular-reflection / `Snell`-refraction golden
//! ([`reflect_refract_vec`](prism_render_architecture::particle::reflect_refract_vec),
//! particle design §10, §14).
//!
//! The `CPU` golden
//! [`reflect_refract_vec`](prism_render_architecture::particle::reflect_refract_vec)
//! owns the *optical response* half of the particle-vs-surface interaction
//! math. Given an `incident` direction and a surface `normal` it answers, in
//! closed form, which way a ray or velocity goes after the surface: a mirror
//! bounce ([`reflect`](prism_render_architecture::particle::reflect_refract_vec::reflect)),
//! a bent transmission across an index-of-refraction boundary
//! ([`refract`](prism_render_architecture::particle::reflect_refract_vec::refract)),
//! the critical-angle test that decides whether a transmitted ray exists at all
//! ([`is_total_internal_reflection`](prism_render_architecture::particle::reflect_refract_vec::is_total_internal_reflection)),
//! the `Schlick` approximation of the `Fresnel` reflectance
//! ([`fresnel_schlick_reflectance`](prism_render_architecture::particle::reflect_refract_vec::fresnel_schlick_reflectance))
//! and its normal-incidence seed
//! ([`fresnel_schlick_r0`](prism_render_architecture::particle::reflect_refract_vec::fresnel_schlick_r0)),
//! and a kinematic restitution bounce
//! ([`reflect_with_restitution`](prism_render_architecture::particle::reflect_refract_vec::reflect_with_restitution)).
//!
//! [`GpuReflectRefractVec`] is the on-device twin: one thread per query
//! reproduces every one of those answers, so a passing real-device parity test
//! is direct evidence the ported kernel solves the same optics and classifies
//! the same degenerate case the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Every per-query answer the reference computes is reproduced: the mirror
//! `reflect_dir`, the transmitted `refract_dir` paired with a `refract_valid`
//! flag (the `GPU` image of the reference `Option`: a total-internal-reflection
//! geometry sets the flag false and leaves the direction zero), the standalone
//! `is_total_internal_reflection` flag, the `Schlick` `fresnel_reflectance` and
//! its `fresnel_r0` seed, and the `restitution_dir` velocity after a coefficient
//! bounce. The reference's one real branch — the discriminant `k < 0` of
//! [`refract`](prism_render_architecture::particle::reflect_refract_vec::refract),
//! i.e. total internal reflection past the critical angle — is mirrored exactly:
//! `k < 0` yields no transmitted ray (`refract_valid` false, zero direction, and
//! `is_tir` true), while `k >= 0` evaluates the vector-form `Snell` transmission
//! `eta * incident + (eta * cos_i - sqrt(k)) * normal`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, the `dot`
//! builtin, `+ - * /` and one `sqrt` for the `Snell` transmission — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no transcendental and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! `Schlick` fifth power is written as an explicit multiplication chain, exactly
//! as the reference writes it, never a `pow` call.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32` fields, tight enough
//! to catch a genuinely wrong port (a dropped branch, a swapped coefficient, a
//! wrong clamp) yet loose enough to admit legal fused multiply-add contraction,
//! while the `refract_valid` and `is_tir` classification flags are compared
//! bit-exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`reflect_refract_vec`](prism_render_architecture::particle::reflect_refract_vec);
//! no third-party engine source or derived code.

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

/// The portable core-`WGSL` reflect/refract kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`reflect_refract_vec`](prism_render_architecture::particle::reflect_refract_vec)
/// function for function; see the module documentation for the algorithm.
const REFLECT_REFRACT_VEC_WGSL: &str = r#"
// Reflect/refract optical-response twin: one thread per query reproduces the
// mirror reflection, the vector-form Snell refraction with its total-internal-
// reflection discriminant, the standalone TIR flag, the Schlick Fresnel
// reflectance and its normal-incidence seed, and the coefficient-of-restitution
// velocity bounce. It mirrors the CPU golden reflect_refract_vec function for
// function.
//
// Portability: only the core subset (clamp, dot, + - * / and one sqrt) is used;
// no sin/cos/exp/log/pow/tan and no optional device feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's particle::reflect_refract_vec; no
// third-party engine source or derived code.

// Epsilon guarding the length divide inside the restitution normalise, matching
// the reference `v_normalize` guard; never an exact == / != on an f32.
const NORM_EPS: f32 = 1.0e-12;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Incident direction (into the surface); assumed unit for reflect/refract.
    incident: vec3<f32>,
    pad0: f32,
    // Surface normal (out of the surface); assumed unit for reflect/refract and
    // normalised internally for the restitution bounce.
    normal: vec3<f32>,
    pad1: f32,
    // Velocity fed to the coefficient-of-restitution bounce.
    velocity: vec3<f32>,
    pad2: f32,
    // Ratio of indices n1 / n2 for the Snell refraction.
    eta: f32,
    // Coefficient of restitution for the velocity bounce.
    restitution: f32,
    // Cosine of the view angle for the Schlick reflectance.
    cos_theta: f32,
    // Normal-incidence reflectance seed for the Schlick reflectance.
    r0: f32,
    // Incoming index of refraction for the r0 seed.
    n1: f32,
    // Outgoing index of refraction for the r0 seed.
    n2: f32,
    pad3: f32,
    pad4: f32,
}

struct Result {
    // Mirror reflection direction.
    reflect_dir: vec3<f32>,
    pad0: f32,
    // Snell transmitted direction; zero when total internal reflection fires.
    refract_dir: vec3<f32>,
    pad1: f32,
    // Velocity after the coefficient-of-restitution bounce.
    restitution_dir: vec3<f32>,
    pad2: f32,
    // Schlick Fresnel reflectance.
    fresnel_reflectance: f32,
    // Schlick normal-incidence reflectance from the index pair.
    fresnel_r0: f32,
    // 1 when a transmitted ray exists (reference Some), 0 at TIR (reference None).
    refract_valid: u32,
    // 1 when the configuration undergoes total internal reflection.
    is_tir: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Returns the unit vector in the direction of v, mirroring the reference
// `v_normalize`: a length at or below NORM_EPS leaves the input unchanged, so
// this never divides by zero.
fn normalize_guarded(v: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(v, v));
    if (len > NORM_EPS) {
        return v * (1.0 / len);
    }
    return v;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let incident = q.incident;
    let normal = q.normal;

    // reflect: i - 2 (i·n) n.
    let d = dot(incident, normal);
    let reflect_dir = incident - normal * (2.0 * d);

    // refract: vector-form Snell with cos_i = -(i·n) and discriminant
    // k = 1 - eta² (1 - cos_i²). k < 0 is total internal reflection: no
    // transmitted ray, matching the reference None.
    let cos_i = -dot(incident, normal);
    let k = 1.0 - q.eta * q.eta * (1.0 - cos_i * cos_i);
    var refract_dir = vec3<f32>(0.0, 0.0, 0.0);
    var refract_valid: u32 = 0u;
    var is_tir: u32 = 0u;
    if (k < 0.0) {
        is_tir = 1u;
        refract_valid = 0u;
        refract_dir = vec3<f32>(0.0, 0.0, 0.0);
    } else {
        refract_valid = 1u;
        refract_dir = incident * q.eta + normal * (q.eta * cos_i - sqrt(k));
    }

    // Schlick reflectance: r0 + (1 - r0) (1 - cos)^5, cos clamped to [0, 1], the
    // fifth power an explicit multiplication chain.
    let cos_c = clamp(q.cos_theta, 0.0, 1.0);
    let m = 1.0 - cos_c;
    let m5 = m * m * m * m * m;
    let fresnel_reflectance = q.r0 + (1.0 - q.r0) * m5;

    // Schlick r0 from the index pair: ((n1 - n2) / (n1 + n2))².
    let rr = (q.n1 - q.n2) / (q.n1 + q.n2);
    let fresnel_r0 = rr * rr;

    // Restitution bounce: negate and scale the normal component, keep tangential.
    let n = normalize_guarded(normal);
    let vn = dot(q.velocity, n);
    let normal_component = n * vn;
    let tangential = q.velocity - normal_component;
    let rebound = normal_component * (-q.restitution);
    let restitution_dir = tangential + rebound;

    var out: Result;
    out.reflect_dir = reflect_dir;
    out.pad0 = 0.0;
    out.refract_dir = refract_dir;
    out.pad1 = 0.0;
    out.restitution_dir = restitution_dir;
    out.pad2 = 0.0;
    out.fresnel_reflectance = fresnel_reflectance;
    out.fresnel_r0 = fresnel_r0;
    out.refract_valid = refract_valid;
    out.is_tir = is_tir;
    results[idx] = out;
}
"#;

/// One reflect/refract query: the direction inputs plus the scalar parameters
/// the reference's six functions consume.
///
/// `incident` and `normal` are the shared direction inputs; `reflect`,
/// `refract` and `is_total_internal_reflection` assume both are unit length,
/// while `reflect_with_restitution` normalises `normal` internally. The scalar
/// fields feed the remaining functions: `eta` the `Snell` refraction,
/// `restitution` the bounce, `cos_theta` and `r0` the `Schlick` reflectance, and
/// `n1` / `n2` the `fresnel_r0` seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectRefractQuery {
    /// Incident direction, into the surface along the ray of travel.
    pub incident: [f32; 3],
    /// Surface normal, out of the surface toward the incoming ray.
    pub normal: [f32; 3],
    /// Velocity fed to the coefficient-of-restitution bounce.
    pub velocity: [f32; 3],
    /// Ratio of indices `n1 / n2` for the `Snell` refraction.
    pub eta: f32,
    /// Coefficient of restitution for the velocity bounce.
    pub restitution: f32,
    /// Cosine of the view angle for the `Schlick` reflectance.
    pub cos_theta: f32,
    /// Normal-incidence reflectance seed for the `Schlick` reflectance.
    pub r0: f32,
    /// Incoming index of refraction for the `fresnel_r0` seed.
    pub n1: f32,
    /// Outgoing index of refraction for the `fresnel_r0` seed.
    pub n2: f32,
}

impl ReflectRefractQuery {
    /// Builds a query from the direction inputs and scalar parameters.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the six reference functions' combined inputs in one query"
    )]
    pub const fn new(
        incident: [f32; 3],
        normal: [f32; 3],
        velocity: [f32; 3],
        eta: f32,
        restitution: f32,
        cos_theta: f32,
        r0: f32,
        n1: f32,
        n2: f32,
    ) -> ReflectRefractQuery {
        ReflectRefractQuery {
            incident,
            normal,
            velocity,
            eta,
            restitution,
            cos_theta,
            r0,
            n1,
            n2,
        }
    }
}

/// The resolved answer for one query, mirroring every value the reference
/// reports across its six twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectRefractResult {
    /// Mirror reflection direction, matching `reflect`.
    pub reflect_dir: [f32; 3],
    /// Transmitted direction, matching the inner vector of `refract`'s
    /// `Some`; left zero when `refract_valid` is false.
    pub refract_dir: [f32; 3],
    /// Whether a transmitted ray exists, matching `refract` returning `Some`.
    pub refract_valid: bool,
    /// Whether the configuration undergoes total internal reflection, matching
    /// `is_total_internal_reflection`.
    pub is_total_internal_reflection: bool,
    /// `Schlick` `Fresnel` reflectance, matching `fresnel_schlick_reflectance`.
    pub fresnel_reflectance: f32,
    /// `Schlick` normal-incidence reflectance, matching `fresnel_schlick_r0`.
    pub fresnel_r0: f32,
    /// Velocity after the coefficient-of-restitution bounce, matching
    /// `reflect_with_restitution`.
    pub restitution_dir: [f32; 3],
}

/// `repr(C)` `std430` layout of one packed query: three `vec3` slots for
/// `incident`, `normal` and `velocity` (each on its own `16`-byte-aligned slot
/// with a trailing pad lane), then two scalar `vec4` slots holding
/// `(eta, restitution, cos_theta, r0)` and `(n1, n2, pad, pad)` — `80` bytes,
/// exactly as the `WGSL` `Query` struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Incident direction.
    incident: [f32; 3],
    /// Padding lane after the incident direction.
    pad0: f32,
    /// Surface normal.
    normal: [f32; 3],
    /// Padding lane after the normal.
    pad1: f32,
    /// Restitution velocity.
    velocity: [f32; 3],
    /// Padding lane after the velocity.
    pad2: f32,
    /// Refraction index ratio `n1 / n2`.
    eta: f32,
    /// Coefficient of restitution.
    restitution: f32,
    /// View-angle cosine for the `Schlick` reflectance.
    cos_theta: f32,
    /// Normal-incidence reflectance seed.
    r0: f32,
    /// Incoming index of refraction.
    n1: f32,
    /// Outgoing index of refraction.
    n2: f32,
    /// Padding lane.
    pad3: f32,
    /// Padding lane.
    pad4: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &ReflectRefractQuery) -> GpuQuery {
        GpuQuery {
            incident: query.incident,
            pad0: 0.0,
            normal: query.normal,
            pad1: 0.0,
            velocity: query.velocity,
            pad2: 0.0,
            eta: query.eta,
            restitution: query.restitution,
            cos_theta: query.cos_theta,
            r0: query.r0,
            n1: query.n1,
            n2: query.n2,
            pad3: 0.0,
            pad4: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: three `vec3` slots for the mirror
/// reflection, the transmitted direction and the restitution velocity (each
/// with a trailing pad lane), then a four-scalar slot holding
/// `(fresnel_reflectance, fresnel_r0, refract_valid, is_tir)` — `64` bytes
/// matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Mirror reflection direction.
    reflect_dir: [f32; 3],
    /// Padding lane after the reflection.
    pad0: f32,
    /// Transmitted direction (zero at total internal reflection).
    refract_dir: [f32; 3],
    /// Padding lane after the refraction.
    pad1: f32,
    /// Restitution velocity.
    restitution_dir: [f32; 3],
    /// Padding lane after the restitution velocity.
    pad2: f32,
    /// `Schlick` reflectance.
    fresnel_reflectance: f32,
    /// `Schlick` normal-incidence reflectance seed.
    fresnel_r0: f32,
    /// `1` when a transmitted ray exists, `0` at total internal reflection.
    refract_valid: u32,
    /// `1` when the configuration undergoes total internal reflection.
    is_tir: u32,
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

/// A compiled, reusable reflect/refract compute pipeline.
pub struct GpuReflectRefractVec {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuReflectRefractVec {
    /// Compiles the reflect/refract kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuReflectRefractVec {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_reflect_refract_vec"),
            source: ShaderSource::Wgsl(REFLECT_REFRACT_VEC_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_reflect_refract_vec_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_reflect_refract_vec_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_reflect_refract_vec_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuReflectRefractVec {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`ReflectRefractResult`] per
    /// input, in order.
    ///
    /// Each result equals the reference answers (`reflect`, `refract`,
    /// `is_total_internal_reflection`, `fresnel_schlick_reflectance`,
    /// `fresnel_schlick_r0` and `reflect_with_restitution`) to within the
    /// tolerance documented on this module, with the `refract_valid` and
    /// `is_total_internal_reflection` classification flags matching exactly. An
    /// empty input returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[ReflectRefractQuery],
    ) -> Vec<ReflectRefractResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_reflect_refract_vec_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_reflect_refract_vec_output"),
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
            label: Some("prism_volumetric_reflect_refract_vec_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_reflect_refract_vec_bind_group"),
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
            label: Some("prism_volumetric_reflect_refract_vec_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_reflect_refract_vec_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_reflect_refract_vec_pass"),
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

/// Decodes one packed [`GpuResult`] into the public [`ReflectRefractResult`].
fn decode_result(raw: &GpuResult) -> ReflectRefractResult {
    ReflectRefractResult {
        reflect_dir: raw.reflect_dir,
        refract_dir: raw.refract_dir,
        refract_valid: raw.refract_valid != 0,
        is_total_internal_reflection: raw.is_tir != 0,
        fresnel_reflectance: raw.fresnel_reflectance,
        fresnel_r0: raw.fresnel_r0,
        restitution_dir: raw.restitution_dir,
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
