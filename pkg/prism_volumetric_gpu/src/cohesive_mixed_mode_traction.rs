//! `wgpu` compute twin of the mixed-mode bilinear cohesive-zone traction law,
//! from the `CPU` golden `prism_physics_core::collider::cohesive_zone`'s
//! `cohesive_traction` (together with `CohesiveModel::new` and `damage_at`).
//!
//! A cohesive zone maps a relative displacement jump `delta` across an
//! interface, with unit outward `normal`, to a traction vector `t`, tracking
//! an irreversible history variable `kappa = max lambda` so decohesion never
//! heals. This module ports that single stateless closed form onto the device:
//! one thread resolves one interface evaluation, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same traction,
//! damage and advanced-history flag the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! The model parameters are derived exactly as `CohesiveModel::new` does:
//!
//! * onset separation `d0 = strength / stiffness`,
//! * final separation `df = 2 * fracture_energy / strength`,
//! * the model is valid only when every input is finite, `stiffness`,
//!   `strength`, `fracture_energy` are strictly positive, `shear_weight >= 0`,
//!   and the softening branch exists (`df > d0`).
//!
//! For a valid model each query reproduces `cohesive_traction`:
//!
//! * `n = normal / |normal|`; a near-zero `|normal| <= EPS` yields a zero
//!   traction, `effective_separation = 0`, `damage = damage_at(kappa_in)` and
//!   leaves the history untouched (`advanced = false`, `kappa_out = kappa_in`).
//! * `delta_n = delta . n`, `tangent = delta - delta_n * n`,
//!   `delta_t = |tangent|`, `open_n = max(delta_n, 0)`.
//! * `lambda = sqrt(open_n^2 + beta^2 * delta_t^2)`.
//! * `advanced = lambda > kappa_in && lambda > d0`;
//!   `kappa_out = max(kappa_in, lambda)`.
//! * `damage = damage_at(kappa_out)`: `0` for `k <= d0`, `1` for `k >= df`,
//!   else `clamp(df * (k - d0) / (k * (df - d0)), 0, 1)`; `secant = 1 - damage`.
//! * `normal_traction = secant * K * delta_n` in tension (`delta_n >= 0`),
//!   full penalty `K * delta_n` in compression.
//! * `tangent_traction = secant * K * tangent` when `delta_t > 0`, else zero.
//! * `traction = normal_traction * n + tangent_traction`.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through operators a `GPU` may contract, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact; the
//! continuous scalars and the traction vector are compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`), while the discrete
//! `advanced` and `valid` flags are compared exactly. The damage law has knees
//! at `kappa = d0` and `kappa = df`, and the `advanced` flag toggles at
//! `lambda = kappa_in` and `lambda = d0`; the random fixtures are
//! rejection-sampled clear of all of these so round-off cannot flip a flag.
//!
//! # Degenerate inputs
//!
//! A near-zero normal is handled explicitly (zero traction, history preserved).
//! When the model is not constructible (non-finite inputs, non-positive
//! stiffness/strength/fracture energy, negative shear weight, or `df <= d0`),
//! or any input is non-finite, the kernel emits a zeroed result with
//! `valid = 0` and `kappa_out = kappa_in`, mirrored on the host so the two
//! sides agree deterministically rather than racing propagated `NaN`. Every
//! guard is an ordered compare fed to `select`, never a bare `f32` equality,
//! and every divisor is `select`-guarded so an un-taken branch never divides by
//! zero. An empty query batch short-circuits on the host with no dispatch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `length`,
//! `dot`, `sqrt`, `max`, `clamp`, `+ - * /` and `select` with unsigned index
//! arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no `round`,
//! no `f32` remainder, no `u64`/`i64`/`f64`/`u16`/`i16`, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::cohesive_zone`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` cohesive-traction kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `cohesive_traction`; see the module documentation for the
/// closed form.
const COHESIVE_MIXED_MODE_TRACTION_WGSL: &str = r#"
// Mixed-mode bilinear cohesive-traction twin: one thread per query reproduces
// cohesive_traction, including the CohesiveModel::new derivation and damage_at.
// It uses only the portable core-WGSL subset (abs, length, dot, sqrt, max,
// clamp, + - * /, select plus unsigned index math), has no loop and no
// transcendental, so it provably terminates. Every degeneracy guard is an
// ordered abs < limit compare fed to select (never a bare equality), and every
// divisor is select-guarded so an un-taken branch never divides by zero.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    stiffness: f32,
    strength: f32,
    fracture_energy: f32,
    shear_weight: f32,
    delta: vec3<f32>,
    pad0: f32,
    normal: vec3<f32>,
    pad1: f32,
    kappa_in: f32,
    pad2: f32,
    pad3: f32,
    pad4: f32,
}

struct Result {
    traction: vec3<f32>,
    effective_separation: f32,
    damage: f32,
    normal_traction: f32,
    kappa_out: f32,
    advanced: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// f32::EPSILON, the near-zero-normal threshold used by the golden.
const EPS: f32 = 1.1920929e-7;
// Finite-magnitude limit: abs(x) < FINITE_LIMIT rejects both +-inf and NaN.
const FINITE_LIMIT: f32 = 3.0e38;

fn is_finite(x: f32) -> bool {
    return abs(x) < FINITE_LIMIT;
}

// Secant damage d(kappa) for a monotone history variable kappa >= 0, matching
// the golden damage_at. The middle-branch divisor is select-guarded so the
// un-taken branch never divides by zero.
fn damage_at(kappa: f32, d0: f32, df: f32) -> f32 {
    let below = kappa <= d0;
    let above = kappa >= df;
    let num = df * (kappa - d0);
    let den = kappa * (df - d0);
    let safe_den = select(1.0, den, den > 0.0);
    let mid = clamp(num / safe_den, 0.0, 1.0);
    // Golden order: below -> 0 first, then above -> 1, else mid.
    let v = select(mid, 1.0, above);
    return select(v, 0.0, below);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let stiffness = q.stiffness;
    let strength = q.strength;
    let fracture_energy = q.fracture_energy;
    let beta = q.shear_weight;
    let kappa_in = q.kappa_in;

    // Guard the model-derivation divisors so an invalid model never produces a
    // NaN that leaks past the valid gate.
    let safe_stiffness = select(1.0, stiffness, stiffness > 0.0);
    let safe_strength = select(1.0, strength, strength > 0.0);
    let d0 = safe_strength / safe_stiffness;
    let df = 2.0 * fracture_energy / safe_strength;

    let params_finite = is_finite(stiffness) && is_finite(strength)
        && is_finite(fracture_energy) && is_finite(beta);
    let inputs_finite = is_finite(q.delta.x) && is_finite(q.delta.y) && is_finite(q.delta.z)
        && is_finite(q.normal.x) && is_finite(q.normal.y) && is_finite(q.normal.z)
        && is_finite(kappa_in);
    let positive = stiffness > 0.0 && strength > 0.0 && fracture_energy > 0.0 && beta >= 0.0;
    let softening = df > d0;
    let valid = params_finite && inputs_finite && positive && softening;

    // Degenerate normal: zero traction, history preserved, damage at kappa_in.
    let n_len = length(q.normal);
    let degenerate_normal = n_len <= EPS;
    let safe_n_len = select(n_len, 1.0, degenerate_normal);
    let n = q.normal / safe_n_len;

    let delta_n = dot(q.delta, n);
    let tangent = q.delta - delta_n * n;
    let delta_t = length(tangent);
    let open_n = max(delta_n, 0.0);
    let lambda = sqrt(open_n * open_n + beta * beta * delta_t * delta_t);

    let advanced_bool = (lambda > kappa_in) && (lambda > d0);
    let kappa_grown = max(kappa_in, lambda);
    let damage_grown = damage_at(kappa_grown, d0, df);
    let secant = 1.0 - damage_grown;
    let k = stiffness;

    let nt_tension = secant * k * delta_n;
    let nt_comp = k * delta_n;
    let normal_traction = select(nt_comp, nt_tension, delta_n >= 0.0);
    let tangent_scale = select(0.0, secant * k, delta_t > 0.0);
    let tangent_traction = tangent * tangent_scale;
    let traction = normal_traction * n + tangent_traction;

    // Fold the near-zero-normal degenerate path over the active result.
    let damage_rest = damage_at(kappa_in, d0, df);
    let eff_nz = select(lambda, 0.0, degenerate_normal);
    let dmg_nz = select(damage_grown, damage_rest, degenerate_normal);
    let trac_nz = select(traction, vec3<f32>(0.0, 0.0, 0.0), degenerate_normal);
    let nt_nz = select(normal_traction, 0.0, degenerate_normal);
    let adv_nz = advanced_bool && !degenerate_normal;
    let kappa_nz = select(kappa_grown, kappa_in, degenerate_normal);

    // Gate everything by model validity; an invalid model zeroes the result.
    var out: Result;
    out.traction = select(vec3<f32>(0.0, 0.0, 0.0), trac_nz, valid);
    out.effective_separation = select(0.0, eff_nz, valid);
    out.damage = select(0.0, dmg_nz, valid);
    out.normal_traction = select(0.0, nt_nz, valid);
    out.kappa_out = select(kappa_in, kappa_nz, valid);
    out.advanced = select(0u, 1u, adv_nz && valid);
    out.valid = select(0u, 1u, valid);
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The four scalar model parameters fill the first 16-byte block, then `delta`
/// and `normal` are each a `vec3<f32>` on a 16-byte-aligned slot, and
/// `kappa_in` fills the final block; the whole query is `64` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    stiffness: f32,
    strength: f32,
    fracture_energy: f32,
    shear_weight: f32,
    delta: [f32; 3],
    pad0: f32,
    normal: [f32; 3],
    pad1: f32,
    kappa_in: f32,
    pad2: f32,
    pad3: f32,
    pad4: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the traction vector shares its 16-byte slot with
/// `effective_separation`, followed by the remaining scalars and the discrete
/// flags, padded to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    traction: [f32; 3],
    effective_separation: f32,
    damage: f32,
    normal_traction: f32,
    kappa_out: f32,
    advanced: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One cohesive-traction query: the model parameters, the displacement jump,
/// the interface normal and the incoming irreversible history `kappa_in`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveMixedModeTractionQuery {
    /// Penalty / initial-slope stiffness `K`.
    pub stiffness: f32,
    /// Peak traction (interface strength) `sigma_c`.
    pub strength: f32,
    /// Fracture energy `G_c` (area under the traction-separation curve).
    pub fracture_energy: f32,
    /// Shear mode-mixity weight `beta` applied to the tangential separation.
    pub shear_weight: f32,
    /// Relative displacement jump `delta` across the interface.
    pub delta: [f32; 3],
    /// Interface outward normal (normalised defensively in the kernel).
    pub normal: [f32; 3],
    /// Incoming irreversible history variable `kappa_in = max lambda` so far.
    pub kappa_in: f32,
}

impl CohesiveMixedModeTractionQuery {
    /// Builds a query from the model parameters, the displacement jump, the
    /// interface normal and the incoming history variable.
    #[must_use]
    pub fn new(
        stiffness: f32,
        strength: f32,
        fracture_energy: f32,
        shear_weight: f32,
        delta: [f32; 3],
        normal: [f32; 3],
        kappa_in: f32,
    ) -> CohesiveMixedModeTractionQuery {
        CohesiveMixedModeTractionQuery {
            stiffness,
            strength,
            fracture_energy,
            shear_weight,
            delta,
            normal,
            kappa_in,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `cohesive_traction` output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveMixedModeTractionResult {
    /// Effective (mixed-mode) separation `lambda` of this evaluation.
    pub effective_separation: f32,
    /// Secant decohesion damage `d in [0, 1]` after updating the history.
    pub damage: f32,
    /// Full traction vector `t` developed across the interface.
    pub traction: [f32; 3],
    /// Signed normal component of the traction (`t . n`).
    pub normal_traction: f32,
    /// Whether this evaluation advanced the irreversible history `kappa`.
    pub advanced: bool,
    /// Outgoing history variable `kappa_out = max(kappa_in, lambda)`.
    pub kappa_out: f32,
    /// Whether the model was constructible and all inputs finite; `false`
    /// yields a fully zeroed result with `kappa_out = kappa_in`.
    pub valid: bool,
}

/// Encodes one [`CohesiveMixedModeTractionQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &CohesiveMixedModeTractionQuery) -> GpuQuery {
    GpuQuery {
        stiffness: q.stiffness,
        strength: q.strength,
        fracture_energy: q.fracture_energy,
        shear_weight: q.shear_weight,
        delta: q.delta,
        pad0: 0.0,
        normal: q.normal,
        pad1: 0.0,
        kappa_in: q.kappa_in,
        pad2: 0.0,
        pad3: 0.0,
        pad4: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`CohesiveMixedModeTractionResult`].
fn decode_result(raw: &GpuResult) -> CohesiveMixedModeTractionResult {
    CohesiveMixedModeTractionResult {
        effective_separation: raw.effective_separation,
        damage: raw.damage,
        traction: raw.traction,
        normal_traction: raw.normal_traction,
        advanced: raw.advanced != 0,
        kappa_out: raw.kappa_out,
        valid: raw.valid != 0,
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

/// A compiled, reusable cohesive-traction compute pipeline, twinning the `CPU`
/// golden `cohesive_traction`.
pub struct GpuCohesiveMixedModeTraction {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCohesiveMixedModeTraction {
    /// Compiles the cohesive-traction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCohesiveMixedModeTraction {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cohesive_mixed_mode_traction"),
            source: ShaderSource::Wgsl(COHESIVE_MIXED_MODE_TRACTION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_mixed_mode_traction_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cohesive_mixed_mode_traction_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cohesive_mixed_mode_traction_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCohesiveMixedModeTraction {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`CohesiveMixedModeTractionResult`] per input, in order.
    ///
    /// The outputs match the reference to the module's tolerance. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[CohesiveMixedModeTractionQuery],
    ) -> Vec<CohesiveMixedModeTractionResult> {
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
            label: Some("prism_volumetric_cohesive_mixed_mode_traction_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cohesive_mixed_mode_traction_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cohesive_mixed_mode_traction_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cohesive_mixed_mode_traction_bind_group"),
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
            label: Some("prism_volumetric_cohesive_mixed_mode_traction_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cohesive_mixed_mode_traction_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cohesive_mixed_mode_traction_pass"),
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
