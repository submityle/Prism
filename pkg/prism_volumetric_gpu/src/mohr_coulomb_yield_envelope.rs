//! `wgpu` compute twin of the Mohr–Coulomb shear-failure envelope, from the
//! `CPU` golden `prism_physics_core::collider::mohr_coulomb_yield`'s
//! `MohrCoulombCriterion`.
//!
//! The Mohr–Coulomb criterion is the classical yield surface for cohesive
//! frictional materials (soils, powders, rock). It is defined by an internal
//! friction angle `phi` (radians) and a cohesion intercept `c`. On a plane
//! carrying normal stress `sigma_n` and shear `tau`, failure occurs when the
//! mobilised shear reaches the available strength `tau_f = c + sigma_n *
//! tan(phi)`. For a principal-stress state the Mohr circle has centre `p` and
//! radius `R`, and the yield function is `f = R - (c cos(phi) + p sin(phi))`.
//! This module ports those stateless closed forms onto the device: one thread
//! resolves one query, so a passing real-device parity test is direct evidence
//! the ported kernel computes the same envelope the reference does.
//!
//! # What is twinned
//!
//! Each query carries a friction angle, a cohesion, a `mode` selector and the
//! stress inputs for both modes. The kernel mirrors `MohrCoulombCriterion::new`
//! (criterion validity), `friction_coefficient`, `shear_strength`,
//! `unconfined_compressive_strength`, `cohesion_apex`, and one of
//! `evaluate_principal` / `evaluate_plane` selected by `mode`:
//!
//! * The criterion is valid only when `phi` and `c` are finite,
//!   `0 <= phi < pi/2` and `c >= 0`. An invalid criterion zeros every output.
//! * `friction_coefficient = tan(phi) = sin(phi)/cos(phi)`.
//! * `shear_strength_at_normal = c + sigma_n * tan(phi)` for the query's
//!   `sigma_n`.
//! * `unconfined_compressive_strength = 2 c cos(phi) / (1 - sin(phi))`.
//! * `cohesion_apex = c / tan(phi)`, with a separate `apex_valid` flag that is
//!   `0` for a frictionless material (`phi = 0`, apex unbounded).
//! * `mode = 0` evaluates a principal-stress triple: `sigma_max`/`sigma_min`
//!   drive `center = (sigma_max + sigma_min)/2`, `radius = (sigma_max -
//!   sigma_min)/2`, `strength = c cos(phi) + center sin(phi)` and `yield =
//!   radius - strength`. `mode = 1` evaluates a single plane: `normal_stress =
//!   sigma_n`, `shear_stress = tau`, `strength = c + sigma_n tan(phi)` and
//!   `yield = tau - strength`.
//! * A principal evaluation requires three finite principal stresses; a plane
//!   evaluation requires a finite `sigma_n`, a finite `tau` and `tau >= 0`.
//!   The overall `valid` flag is the criterion validity combined with the
//!   selected mode's state validity.
//!
//! # Correctness model
//!
//! The golden evaluates its trigonometry in `f64` and casts to `f32`; the
//! parity oracle does the same, while the kernel uses the portable `f32`
//! `sin`/`cos`/`tan` builtins, so `CPU` and `GPU` are not necessarily bit-exact.
//! Each continuous output is compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` and `apex_valid` flags are
//! compared exactly. The parity test keeps the friction angle well away from
//! `pi/2` (where `tan` and `1 - sin(phi)` are ill-conditioned) so round-off
//! cannot flip a validity decision.
//!
//! # Degenerate inputs
//!
//! A non-finite or out-of-range friction angle, a negative cohesion, or a
//! non-finite / negative-shear stress state yields `valid = 0` with zeroed
//! outputs. Every divisor (`cos(phi)` in `tan`, `1 - sin(phi)` in the
//! unconfined strength, `tan(phi)` in the apex) is fed through a `select` guard
//! so the un-taken branch never divides by zero. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `sin`, `cos`, `tan`, `+ - * /`, `select` and unsigned index arithmetic —
//! with no `f64`/`u64`/`i64`, no `round`, no `pow` and no `f32` remainder, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with
//! the ordered compare `abs(x) < 3.0e38` (which rejects both infinities and
//! `NaN`) rather than a bare `x == x`; the only equality is the integer `mode`
//! comparison. There is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::mohr_coulomb_yield`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Mohr–Coulomb envelope kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `MohrCoulombCriterion`; see the module
/// documentation for the closed forms.
const MOHR_COULOMB_YIELD_ENVELOPE_WGSL: &str = r#"
// Mohr-Coulomb envelope twin: one thread per query reproduces the criterion
// validity gate, the friction coefficient, the shear strength at a normal
// stress, the unconfined compressive strength, the cohesion apex, and a
// principal- or plane-mode yield evaluation. It uses only the portable
// core-WGSL subset (abs, min, max, sin, cos, tan, + - * /, select plus unsigned
// index math), has no loop and no data-dependent branch (only an early
// out-of-range return), so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN); the only equality is the
// integer mode comparison.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Internal friction angle in radians.
    friction_angle: f32,
    // Cohesion intercept.
    cohesion: f32,
    // Normal stress used by shear_strength_at_normal and the plane mode.
    sigma_n: f32,
    // Shear-stress magnitude used by the plane mode.
    tau: f32,
    // Principal stress components (any order) used by the principal mode.
    sigma0: f32,
    sigma1: f32,
    sigma2: f32,
    // Evaluation selector: 0 = principal, 1 = plane.
    mode: u32,
}

struct Result {
    // tan(phi) when the criterion is valid, else 0.
    friction_coefficient: f32,
    // c + sigma_n * tan(phi) for the query's sigma_n, else 0.
    shear_strength_at_normal: f32,
    // 2 c cos(phi) / (1 - sin(phi)) when valid, else 0.
    unconfined_compressive_strength: f32,
    // c / tan(phi) when apex_valid, else 0.
    cohesion_apex: f32,
    // Representative normal stress of the evaluated state.
    normal_stress: f32,
    // Mobilised shear stress of the evaluated state.
    shear_stress: f32,
    // Available shear strength of the evaluated state.
    shear_strength: f32,
    // Yield function f = shear_stress - shear_strength.
    yield_function: f32,
    // 1 when the cohesion apex is bounded (phi > 0), else 0.
    apex_valid: u32,
    // 1 when the criterion and the selected mode's state are both valid.
    valid: u32,
    // Padding words to a 16-byte-friendly stride.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const HALF_PI: f32 = 1.5707964;

fn is_finite(x: f32) -> bool {
    return abs(x) < FINITE_LIMIT;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let phi = q.friction_angle;
    let coh = q.cohesion;

    // Criterion validity: both inputs finite, 0 <= phi < pi/2 and cohesion
    // non-negative. All ordered compares; NaN fails every one.
    let crit_valid =
        is_finite(phi) && is_finite(coh) && (phi >= 0.0) && (phi < HALF_PI) && (coh >= 0.0);

    // Trigonometry (f32 builtins). cos(phi) > 0 for every valid phi; the guard
    // keeps the un-taken (invalid) branch from dividing by zero.
    let ss = sin(phi);
    let cc = cos(phi);
    let cos_guard = select(1.0, cc, crit_valid);
    let mu = ss / cos_guard;

    let friction_coefficient = select(0.0, mu, crit_valid);
    let shear_at = coh + q.sigma_n * mu;
    let shear_strength_at_normal = select(0.0, shear_at, crit_valid);

    // Unconfined compressive strength: 2 c cos / (1 - sin). For valid phi < pi/2
    // the denominator 1 - sin(phi) is strictly positive.
    let unc_denom = select(1.0, 1.0 - ss, crit_valid);
    let unc = 2.0 * coh * cc / unc_denom;
    let unconfined_compressive_strength = select(0.0, unc, crit_valid);

    // Cohesion apex: bounded only for a frictional material (phi > 0 => mu > 0).
    let apex_ok = crit_valid && (phi > 0.0) && (mu > 0.0);
    let apex_denom = select(1.0, mu, apex_ok);
    let cohesion_apex = select(0.0, coh / apex_denom, apex_ok);
    let apex_valid = select(0u, 1u, apex_ok);

    // Principal-mode state.
    let smax = max(max(q.sigma0, q.sigma1), q.sigma2);
    let smin = min(min(q.sigma0, q.sigma1), q.sigma2);
    let p_center = 0.5 * (smax + smin);
    let p_radius = 0.5 * (smax - smin);
    let p_strength = coh * cc + p_center * ss;
    let p_yield = p_radius - p_strength;
    let principal_valid = is_finite(q.sigma0) && is_finite(q.sigma1) && is_finite(q.sigma2);

    // Plane-mode state.
    let pl_normal = q.sigma_n;
    let pl_shear = q.tau;
    let pl_strength = coh + q.sigma_n * mu;
    let pl_yield = q.tau - pl_strength;
    let plane_valid = is_finite(q.sigma_n) && is_finite(q.tau) && (q.tau >= 0.0);

    let is_principal = (q.mode == 0u);
    let state_valid = select(plane_valid, principal_valid, is_principal);
    let overall = crit_valid && state_valid;

    let normal_stress = select(pl_normal, p_center, is_principal);
    let shear_stress = select(pl_shear, p_radius, is_principal);
    let state_strength = select(pl_strength, p_strength, is_principal);
    let yield_function = select(pl_yield, p_yield, is_principal);

    var out: Result;
    out.friction_coefficient = friction_coefficient;
    out.shear_strength_at_normal = shear_strength_at_normal;
    out.unconfined_compressive_strength = unconfined_compressive_strength;
    out.cohesion_apex = cohesion_apex;
    out.normal_stress = select(0.0, normal_stress, overall);
    out.shear_stress = select(0.0, shear_stress, overall);
    out.shear_strength = select(0.0, state_strength, overall);
    out.yield_function = select(0.0, yield_function, overall);
    out.apex_valid = apex_valid;
    out.valid = select(0u, 1u, overall);
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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
/// seven `f32` stress / envelope words plus the `u32` mode selector — `8` words
/// (`32` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    friction_angle: f32,
    cohesion: f32,
    sigma_n: f32,
    tau: f32,
    sigma0: f32,
    sigma1: f32,
    sigma2: f32,
    mode: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: eight continuous `f32` outputs, two `u32` flags and two padding
/// words — `12` words (`48` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    friction_coefficient: f32,
    shear_strength_at_normal: f32,
    unconfined_compressive_strength: f32,
    cohesion_apex: f32,
    normal_stress: f32,
    shear_stress: f32,
    shear_strength: f32,
    yield_function: f32,
    apex_valid: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
}

/// Which stress state a query evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MohrCoulombMode {
    /// A principal-stress triple drives the Mohr circle.
    Principal,
    /// A single plane with normal stress and shear magnitude.
    Plane,
}

/// One Mohr–Coulomb envelope query: the criterion parameters, a `sigma_n` for
/// the strength readout, the stress inputs for both modes, and a mode selector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MohrCoulombYieldEnvelopeQuery {
    /// Internal friction angle in radians.
    pub friction_angle: f32,
    /// Cohesion intercept of the envelope.
    pub cohesion: f32,
    /// Normal stress used for `shear_strength_at_normal` and the plane mode.
    pub sigma_n: f32,
    /// Shear-stress magnitude used by the plane mode.
    pub tau: f32,
    /// Principal stress components (any order) used by the principal mode.
    pub principal: [f32; 3],
    /// Evaluation selector.
    pub mode: MohrCoulombMode,
}

impl MohrCoulombYieldEnvelopeQuery {
    /// Builds a principal-mode query from the criterion parameters, a `sigma_n`
    /// readout stress, and the principal-stress triple.
    #[must_use]
    pub fn principal(
        friction_angle: f32,
        cohesion: f32,
        sigma_n: f32,
        principal: [f32; 3],
    ) -> MohrCoulombYieldEnvelopeQuery {
        MohrCoulombYieldEnvelopeQuery {
            friction_angle,
            cohesion,
            sigma_n,
            tau: 0.0,
            principal,
            mode: MohrCoulombMode::Principal,
        }
    }

    /// Builds a plane-mode query from the criterion parameters and the plane's
    /// normal stress and shear magnitude.
    #[must_use]
    pub fn plane(
        friction_angle: f32,
        cohesion: f32,
        sigma_n: f32,
        tau: f32,
    ) -> MohrCoulombYieldEnvelopeQuery {
        MohrCoulombYieldEnvelopeQuery {
            friction_angle,
            cohesion,
            sigma_n,
            tau,
            principal: [0.0, 0.0, 0.0],
            mode: MohrCoulombMode::Plane,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `MohrCoulombCriterion` outputs for that stress state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MohrCoulombYieldEnvelopeResult {
    /// `tan(phi)` when the criterion is valid, else `0`.
    pub friction_coefficient: f32,
    /// `c + sigma_n * tan(phi)` for the query's `sigma_n`, else `0`.
    pub shear_strength_at_normal: f32,
    /// `2 c cos(phi) / (1 - sin(phi))` when valid, else `0`.
    pub unconfined_compressive_strength: f32,
    /// `c / tan(phi)` when `apex_valid`, else `0`.
    pub cohesion_apex: f32,
    /// Representative normal stress (plane normal stress, or circle centre).
    pub normal_stress: f32,
    /// Mobilised shear stress (plane shear, or circle radius).
    pub shear_stress: f32,
    /// Available shear strength of the evaluated state.
    pub shear_strength: f32,
    /// Yield function `f = shear_stress - shear_strength`.
    pub yield_function: f32,
    /// `1` when the cohesion apex is bounded (`phi > 0`), else `0`.
    pub apex_valid: u32,
    /// `1` when the criterion and the selected mode's state are both valid.
    pub valid: u32,
}

/// Encodes one [`MohrCoulombYieldEnvelopeQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &MohrCoulombYieldEnvelopeQuery) -> GpuQuery {
    let mode = match q.mode {
        MohrCoulombMode::Principal => 0u32,
        MohrCoulombMode::Plane => 1u32,
    };
    GpuQuery {
        friction_angle: q.friction_angle,
        cohesion: q.cohesion,
        sigma_n: q.sigma_n,
        tau: q.tau,
        sigma0: q.principal[0],
        sigma1: q.principal[1],
        sigma2: q.principal[2],
        mode,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`MohrCoulombYieldEnvelopeResult`].
fn decode_result(raw: &GpuResult) -> MohrCoulombYieldEnvelopeResult {
    MohrCoulombYieldEnvelopeResult {
        friction_coefficient: raw.friction_coefficient,
        shear_strength_at_normal: raw.shear_strength_at_normal,
        unconfined_compressive_strength: raw.unconfined_compressive_strength,
        cohesion_apex: raw.cohesion_apex,
        normal_stress: raw.normal_stress,
        shear_stress: raw.shear_stress,
        shear_strength: raw.shear_strength,
        yield_function: raw.yield_function,
        apex_valid: raw.apex_valid,
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

/// A compiled, reusable Mohr–Coulomb envelope compute pipeline, twinning the
/// `CPU` golden `MohrCoulombCriterion`.
pub struct GpuMohrCoulombYieldEnvelope {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMohrCoulombYieldEnvelope {
    /// Compiles the Mohr–Coulomb envelope kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMohrCoulombYieldEnvelope {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope"),
            source: ShaderSource::Wgsl(MOHR_COULOMB_YIELD_ENVELOPE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMohrCoulombYieldEnvelope {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`MohrCoulombYieldEnvelopeResult`] per input, in order.
    ///
    /// The `valid` and `apex_valid` flags match the reference exactly and the
    /// continuous outputs to the module's tolerance. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MohrCoulombYieldEnvelopeQuery],
    ) -> Vec<MohrCoulombYieldEnvelopeResult> {
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
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope_bind_group"),
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
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mohr_coulomb_yield_envelope_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mohr_coulomb_yield_envelope_pass"),
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
