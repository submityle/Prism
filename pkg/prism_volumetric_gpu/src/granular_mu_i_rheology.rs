//! `wgpu` compute twin of the granular `μ(I)` rheology closed form, from the
//! `CPU` golden `prism_physics_core::collider::granular_rheology`'s
//! `GranularRheology`.
//!
//! The `μ(I)` rheology maps a local shear rate and confining pressure to a
//! dimensionless inertial number, an effective friction coefficient, a shear
//! stress and an effective viscosity for a dense granular flow. This module
//! ports the four stateless closed forms `inertial_number`, `friction`,
//! `shear_stress` and `effective_viscosity` onto the device: one thread
//! resolves one query carrying the material parameters directly, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same four quantities the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, for one `(shear_rate, pressure)` pair
//! and the model parameters `mu_static`, `mu_dynamic`, `i_ref`,
//! `grain_diameter` and `grain_density`:
//!
//! * `inertial_number`: `I = γ̇ · d / √(P / ρ_s)`. Invalid (`valid = 0`) when the
//!   shear rate is non-finite or `< 0`, the pressure is non-finite or `<= 0`,
//!   or the denominator `√(P / ρ_s)` is not strictly positive.
//! * `friction`: `μ(I) = μ_s + (μ_2 − μ_s) / (I_0 / I + 1)` for `I > 0`; exactly
//!   `μ_s` at `I <= 0`; `μ_2` for `I = +∞` and `μ_s` for any other non-finite
//!   `I`. The stored friction is `μ` evaluated at the stored inertial number
//!   (which is `0` when the inertial number is invalid, so it degrades to
//!   `μ_s`); friction is always defined and carries no validity flag.
//! * `shear_stress`: `τ = μ(I) · P`, valid exactly when the inertial number is.
//! * `effective_viscosity`: `η = τ / γ̇`, valid when the shear rate is finite and
//!   strictly positive and the shear stress is valid.
//!
//! # Result encoding
//!
//! Each `Option`-returning quantity is encoded as a value plus a `u32` validity
//! flag; `None` becomes value `0` with `valid = 0`. The friction coefficient is
//! always defined and has no flag.
//!
//! # Correctness model
//!
//! The golden is evaluated entirely in `f32`, matching the device, so the only
//! gap is the usual `GPU` operator contraction. Continuous scalars are compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete validity
//! flags are compared exactly. The parity test keeps swept inputs strictly
//! inside the valid region and clear of the `I = 0` knee so round-off cannot
//! flip a discrete decision.
//!
//! # Degenerate inputs
//!
//! A non-finite or negative shear rate, a non-finite or non-positive pressure,
//! or a non-positive effective-time denominator yields an invalid inertial
//! number (and hence invalid shear stress and viscosity). Every divisor is fed
//! through a `select` guard so the un-taken branch never evaluates a division by
//! zero or produces a poisoning `inf`/`NaN`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sqrt`,
//! `+ - * /`, `select` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no `f64`.
//! Finiteness is tested with the ordered compare `abs(x) < 3.0e38` (which
//! rejects both infinities and `NaN`) rather than a bare `x == x`, and validity
//! with ordered `>`/`>=`/`<=`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_rheology`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `μ(I)`-rheology kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `GranularRheology` methods; see the module documentation for the
/// closed forms.
const GRANULAR_MU_I_RHEOLOGY_WGSL: &str = r#"
// Granular mu(I) rheology twin: one thread per query reproduces the inertial
// number, friction coefficient, shear stress and effective viscosity. It uses
// only the portable core-WGSL subset (abs, sqrt, + - * /, select plus unsigned
// index math), has no loop and no data-dependent branch, so it provably
// terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) and validity ordered >/>=/<= compares, all fed to select.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    mu_static: f32,
    mu_dynamic: f32,
    i_ref: f32,
    grain_diameter: f32,
    grain_density: f32,
    shear_rate: f32,
    pressure: f32,
    pad0: f32,
}

struct Result {
    inertial_number: f32,
    inertial_valid: u32,
    friction: f32,
    shear_stress: f32,
    shear_stress_valid: u32,
    effective_viscosity: f32,
    effective_viscosity_valid: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let sr = q.shear_rate;
    let p = q.pressure;
    let gd = q.grain_density;

    // Inertial number validity: shear rate finite and >= 0, pressure finite and
    // > 0, effective-time denominator strictly positive.
    let sr_ok = (abs(sr) < FINITE_LIMIT) && (sr >= 0.0);
    let p_ok = (abs(p) < FINITE_LIMIT) && (p > 0.0);
    let gd_ok = (abs(gd) < FINITE_LIMIT) && (gd > 0.0);

    // Guard the arithmetic so the un-taken branch never divides by zero or
    // produces a poisoning inf/NaN. When valid these equal the raw inputs.
    let safe_p = select(1.0, p, p_ok);
    let safe_gd = select(1.0, gd, gd_ok);
    let micro = safe_p / safe_gd;
    let denom = sqrt(micro);
    let denom_ok = denom > 0.0;
    let inertial_ok = sr_ok && p_ok && gd_ok && denom_ok;

    let safe_denom = select(1.0, denom, inertial_ok);
    let num_sr = select(0.0, sr, inertial_ok);
    let inertial_raw = num_sr * q.grain_diameter / safe_denom;
    let inertial_value = select(0.0, inertial_raw, inertial_ok);

    // Friction mu(I) evaluated on the stored inertial number (0 when invalid,
    // which degrades to mu_static). Ordered compares classify the regimes.
    let big = inertial_value;
    let fin = abs(big) < FINITE_LIMIT;
    let pinf = big > FINITE_LIMIT;
    let pos = big > 0.0;
    let safe_big = select(1.0, big, pos);
    let ratio = q.i_ref / safe_big + 1.0;
    let safe_ratio = select(1.0, ratio, ratio > 0.0);
    let extra = (q.mu_dynamic - q.mu_static) / safe_ratio;
    let continuous = q.mu_static + extra;
    // finite: I > 0 -> continuous, else mu_static.
    let finite_val = select(q.mu_static, continuous, pos);
    // non-finite: +inf -> mu_dynamic, else mu_static.
    let nonfinite_val = select(q.mu_static, q.mu_dynamic, pinf);
    let fric = select(nonfinite_val, finite_val, fin);

    // Shear stress tau = mu(I) * pressure, valid exactly when the inertial
    // number is.
    let safe_shear_p = select(1.0, p, inertial_ok);
    let shear_raw = fric * safe_shear_p;
    let shear_value = select(0.0, shear_raw, inertial_ok);

    // Effective viscosity eta = tau / shear_rate for a finite, strictly
    // positive shear rate and a valid shear stress.
    let sr_pos = (abs(sr) < FINITE_LIMIT) && (sr > 0.0);
    let eff_ok = sr_pos && inertial_ok;
    let safe_sr = select(1.0, sr, eff_ok);
    let eff_raw = shear_value / safe_sr;
    let eff_value = select(0.0, eff_raw, eff_ok);

    var out: Result;
    out.inertial_number = inertial_value;
    out.inertial_valid = select(0u, 1u, inertial_ok);
    out.friction = fric;
    out.shear_stress = shear_value;
    out.shear_stress_valid = select(0u, 1u, inertial_ok);
    out.effective_viscosity = eff_value;
    out.effective_viscosity_valid = select(0u, 1u, eff_ok);
    out.pad0 = 0u;
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
/// the seven material/state scalars padded to `8` `f32` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    mu_static: f32,
    mu_dynamic: f32,
    i_ref: f32,
    grain_diameter: f32,
    grain_density: f32,
    shear_rate: f32,
    pressure: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: four quantities, three validity flags and a pad — `8` words
/// (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    inertial_number: f32,
    inertial_valid: u32,
    friction: f32,
    shear_stress: f32,
    shear_stress_valid: u32,
    effective_viscosity: f32,
    effective_viscosity_valid: u32,
    pad0: u32,
}

/// One `μ(I)`-rheology query: the model parameters plus the local shear rate and
/// confining pressure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularMuIRheologyQuery {
    /// Static (quasi-static) friction coefficient `μ_s`.
    pub mu_static: f32,
    /// Dynamic (high-inertia) friction coefficient `μ_2`.
    pub mu_dynamic: f32,
    /// Reference inertial number `I_0` of the friction law.
    pub i_ref: f32,
    /// Grain diameter `d`.
    pub grain_diameter: f32,
    /// Grain material density `ρ_s`.
    pub grain_density: f32,
    /// Local shear rate `γ̇`.
    pub shear_rate: f32,
    /// Confining pressure `P`.
    pub pressure: f32,
}

impl GranularMuIRheologyQuery {
    /// Builds a query from the model parameters and the shear-rate/pressure
    /// state.
    #[must_use]
    pub fn new(
        mu_static: f32,
        mu_dynamic: f32,
        i_ref: f32,
        grain_diameter: f32,
        grain_density: f32,
        shear_rate: f32,
        pressure: f32,
    ) -> GranularMuIRheologyQuery {
        GranularMuIRheologyQuery {
            mu_static,
            mu_dynamic,
            i_ref,
            grain_diameter,
            grain_density,
            shear_rate,
            pressure,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `GranularRheology` outputs for that state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularMuIRheologyResult {
    /// The inertial number `I` when valid, else `0`.
    pub inertial_number: f32,
    /// `1` when the inertial number is defined, else `0`.
    pub inertial_valid: u32,
    /// The friction coefficient `μ(I)`, always defined.
    pub friction: f32,
    /// The shear stress `τ = μ(I) · P` when valid, else `0`.
    pub shear_stress: f32,
    /// `1` when the shear stress is defined, else `0`.
    pub shear_stress_valid: u32,
    /// The effective viscosity `η = τ / γ̇` when valid, else `0`.
    pub effective_viscosity: f32,
    /// `1` when the effective viscosity is defined, else `0`.
    pub effective_viscosity_valid: u32,
}

/// Encodes one [`GranularMuIRheologyQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GranularMuIRheologyQuery) -> GpuQuery {
    GpuQuery {
        mu_static: q.mu_static,
        mu_dynamic: q.mu_dynamic,
        i_ref: q.i_ref,
        grain_diameter: q.grain_diameter,
        grain_density: q.grain_density,
        shear_rate: q.shear_rate,
        pressure: q.pressure,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`GranularMuIRheologyResult`].
fn decode_result(raw: &GpuResult) -> GranularMuIRheologyResult {
    GranularMuIRheologyResult {
        inertial_number: raw.inertial_number,
        inertial_valid: raw.inertial_valid,
        friction: raw.friction,
        shear_stress: raw.shear_stress,
        shear_stress_valid: raw.shear_stress_valid,
        effective_viscosity: raw.effective_viscosity,
        effective_viscosity_valid: raw.effective_viscosity_valid,
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

/// A compiled, reusable `μ(I)`-rheology compute pipeline, twinning the `CPU`
/// golden `GranularRheology`.
pub struct GpuGranularMuIRheology {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGranularMuIRheology {
    /// Compiles the `μ(I)`-rheology kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGranularMuIRheology {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_granular_mu_i_rheology"),
            source: ShaderSource::Wgsl(GRANULAR_MU_I_RHEOLOGY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_granular_mu_i_rheology_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_granular_mu_i_rheology_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_granular_mu_i_rheology_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGranularMuIRheology {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`GranularMuIRheologyResult`] per input, in order.
    ///
    /// The validity flags match the reference exactly and the continuous
    /// scalars to the module's tolerance. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GranularMuIRheologyQuery],
    ) -> Vec<GranularMuIRheologyResult> {
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
            label: Some("prism_volumetric_granular_mu_i_rheology_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_granular_mu_i_rheology_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_granular_mu_i_rheology_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_granular_mu_i_rheology_bind_group"),
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
            label: Some("prism_volumetric_granular_mu_i_rheology_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_granular_mu_i_rheology_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_granular_mu_i_rheology_pass"),
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
