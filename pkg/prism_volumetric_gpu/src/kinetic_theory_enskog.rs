//! `wgpu` compute twin of the Enskog granular kinetic-theory descriptors, from
//! the `CPU` golden `prism_physics_core::collider::kinetic_theory`'s
//! `GranularKineticState::new` plus its depth-independent getters.
//!
//! Treating an agitated granular assembly as a gas of inelastic hard spheres
//! lets Enskog kinetic theory predict microscopic transport scales from four
//! coarse fields: the number density `n`, the grain diameter `d`, the solid
//! fraction `φ`, and the granular temperature `T` (velocity-variance units).
//! This module ports that single stateless derivation onto the device: one
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same descriptors the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the constitutive correlation:
//!
//! * The Carnahan–Starling pair correlation at contact,
//!   `g0 = (2 − φ) / (2 (1 − φ)³)`. The reference evaluates the cube in `f64`;
//!   this `WGSL` twin evaluates it in `f32` (the device has no `f64`), and the
//!   parity tolerance absorbs the narrowing.
//! * The Enskog mean free path `ℓ = 1 / (√2 · π · n · d² · g0)`.
//! * The per-particle collision frequency `ω = 4 · n · d² · g0 · √(π T)`.
//! * The mean collision time `1 / ω` (invalid when the gas is frozen,
//!   `T = 0 ⇒ ω = 0`), the thermal velocity scale `√max(T, 0)`, and the `RMS`
//!   fluctuation speed `√max(3T, 0)`.
//!
//! The construction is invalid (`valid = 0`, all outputs `0`) for a non-finite
//! input, a non-positive `n` or `d`, `φ` outside `[0, 1)`, a negative `T`, a
//! non-positive or non-finite `g0`, or a non-finite/non-positive mean-free-path
//! denominator.
//!
//! # Correctness model
//!
//! The reference evaluates the pair correlation in `f64` then narrows to `f32`
//! and the rest in `f32`; the kernel evaluates the whole chain in `f32`, so
//! `CPU` and `GPU` need not be bit-exact (a `GPU` may contract a multiply-add,
//! and the `f64`/`f32` cube differs). Each valid scalar is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flags are compared exactly. The parity test keeps random parameters
//! away from the `φ → 1` and `T → 0` knees so the validity decision cannot be
//! flipped by round-off.
//!
//! # Degenerate inputs
//!
//! Any rejected input yields `valid = 0` with all outputs `0`. The two divisors
//! `2 (1 − φ)³` and `√2 π n d² g0`, and the reciprocal `1 / ω`, are guarded with
//! `select` so no unselected branch produces an infinity or `NaN`. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset — `abs`, `max`, `sqrt`,
//! `+ - * /`, `select` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round` and no `f32` remainder, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the
//! ordered compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`)
//! rather than a bare `x == x`, and the range gates use ordered compares; there
//! is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::kinetic_theory`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Enskog kinetic-theory kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden; see the module documentation for the closed form.
const KINETIC_THEORY_ENSKOG_WGSL: &str = r#"
// Enskog kinetic-theory twin: one thread per query evaluates the dense-gas
// granular descriptors. It uses only the portable core-WGSL subset (abs, max,
// sqrt, + - * /, select plus unsigned index math), takes no optional feature
// and has no loop, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) and the range gates are
// ordered compares, all fed to select; every divisor is guarded so no
// unselected branch yields inf/NaN.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Number density n.
    number_density: f32,
    // Grain diameter d.
    grain_diameter: f32,
    // Solid (packing) fraction phi in [0, 1).
    solid_fraction: f32,
    // Granular temperature T >= 0.
    granular_temperature: f32,
}

struct Result {
    // Carnahan-Starling pair correlation g0 when valid, else 0.
    pair_correlation: f32,
    // Enskog mean free path when valid, else 0.
    mean_free_path: f32,
    // Per-particle collision frequency when valid, else 0.
    collision_frequency: f32,
    // Mean collision time 1/omega when valid and omega > 0, else 0.
    mean_collision_time: f32,
    // Thermal velocity scale sqrt(max(T, 0)) when valid, else 0.
    thermal_velocity_scale: f32,
    // RMS fluctuation speed sqrt(max(3T, 0)) when valid, else 0.
    rms_fluctuation_speed: f32,
    // 1 when the state is valid and omega > 0, else 0.
    mean_collision_time_valid: u32,
    // 1 when every construction gate passes, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
const SQRT_2: f32 = 1.4142135623730951;
const PI: f32 = 3.14159265358979;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let n = q.number_density;
    let d = q.grain_diameter;
    let phi = q.solid_fraction;
    let temperature = q.granular_temperature;

    // Input finiteness and range gates via ordered compares.
    let n_ok = (abs(n) < FINITE_LIMIT) && (n > 0.0);
    let d_ok = (abs(d) < FINITE_LIMIT) && (d > 0.0);
    let phi_ok = (abs(phi) < FINITE_LIMIT) && (phi >= 0.0) && (phi < 1.0);
    let temp_ok = (abs(temperature) < FINITE_LIMIT) && (temperature >= 0.0);
    let input_ok = n_ok && d_ok && phi_ok && temp_ok;

    // Carnahan-Starling pair correlation g0 = (2 - phi) / (2 (1 - phi)^3).
    let one_minus = 1.0 - phi;
    let one_minus_cubed = one_minus * one_minus * one_minus;
    let g0_denom = 2.0 * one_minus_cubed;
    let g0_denom_ok = g0_denom > 0.0;
    let g0_denom_safe = select(1.0, g0_denom, g0_denom_ok);
    let g0 = (2.0 - phi) / g0_denom_safe;
    let g0_ok = (abs(g0) < FINITE_LIMIT) && (g0 > 0.0);
    let pair_correlation = g0;

    let d2 = d * d;

    // Mean free path ell = 1 / (sqrt2 pi n d^2 g0).
    let denom = SQRT_2 * PI * n * d2 * pair_correlation;
    let denom_ok = (abs(denom) < FINITE_LIMIT) && (denom > 0.0);
    let denom_safe = select(1.0, denom, denom_ok);
    let mean_free_path = 1.0 / denom_safe;

    // Collision frequency omega = 4 n d^2 g0 sqrt(pi T).
    let thermal = sqrt(max(PI * temperature, 0.0));
    let collision_frequency = 4.0 * n * d2 * pair_correlation * thermal;

    let mfp_ok = abs(mean_free_path) < FINITE_LIMIT;
    let cf_ok = abs(collision_frequency) < FINITE_LIMIT;

    let ok = input_ok && g0_ok && denom_ok && mfp_ok && cf_ok;

    // Mean collision time 1/omega, invalid when the gas is frozen (omega <= 0).
    let cf_pos = collision_frequency > 0.0;
    let cf_safe = select(1.0, collision_frequency, cf_pos);
    let mean_collision_time = 1.0 / cf_safe;
    let mct_ok = ok && cf_pos;

    // Thermal velocity scale sqrt(max(T, 0)) and RMS speed sqrt(max(3T, 0)).
    let thermal_velocity_scale = sqrt(max(temperature, 0.0));
    let rms_fluctuation_speed = sqrt(max(3.0 * temperature, 0.0));

    var out: Result;
    out.pair_correlation = select(0.0, pair_correlation, ok);
    out.mean_free_path = select(0.0, mean_free_path, ok);
    out.collision_frequency = select(0.0, collision_frequency, ok);
    out.mean_collision_time = select(0.0, mean_collision_time, mct_ok);
    out.thermal_velocity_scale = select(0.0, thermal_velocity_scale, ok);
    out.rms_fluctuation_speed = select(0.0, rms_fluctuation_speed, ok);
    out.mean_collision_time_valid = select(0u, 1u, mct_ok);
    out.valid = select(0u, 1u, ok);
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
/// the four coarse fields, naturally `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    number_density: f32,
    grain_diameter: f32,
    solid_fraction: f32,
    granular_temperature: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the six descriptors plus two validity flags, `8` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    pair_correlation: f32,
    mean_free_path: f32,
    collision_frequency: f32,
    mean_collision_time: f32,
    thermal_velocity_scale: f32,
    rms_fluctuation_speed: f32,
    mean_collision_time_valid: u32,
    valid: u32,
}

/// One Enskog kinetic-theory query: the four coarse granular-gas fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KineticTheoryEnskogQuery {
    /// Number density `n`, grains per unit volume (`> 0`).
    pub number_density: f32,
    /// Grain diameter `d` (`> 0`).
    pub grain_diameter: f32,
    /// Solid (packing) fraction `φ ∈ [0, 1)`.
    pub solid_fraction: f32,
    /// Granular temperature `T ≥ 0`, velocity-variance units.
    pub granular_temperature: f32,
}

impl KineticTheoryEnskogQuery {
    /// Builds a query from the four coarse fields.
    #[must_use]
    pub fn new(
        number_density: f32,
        grain_diameter: f32,
        solid_fraction: f32,
        granular_temperature: f32,
    ) -> KineticTheoryEnskogQuery {
        KineticTheoryEnskogQuery {
            number_density,
            grain_diameter,
            solid_fraction,
            granular_temperature,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference Enskog
/// kinetic-theory descriptors for that granular-gas state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KineticTheoryEnskogResult {
    /// Carnahan–Starling pair correlation `g0` when valid, else `0`.
    pub pair_correlation: f32,
    /// Enskog mean free path `ℓ` when valid, else `0`.
    pub mean_free_path: f32,
    /// Per-particle collision frequency `ω` when valid, else `0`.
    pub collision_frequency: f32,
    /// Mean collision time `1/ω` when valid and `ω > 0`, else `0`.
    pub mean_collision_time: f32,
    /// Thermal velocity scale `√max(T, 0)` when valid, else `0`.
    pub thermal_velocity_scale: f32,
    /// `RMS` fluctuation speed `√max(3T, 0)` when valid, else `0`.
    pub rms_fluctuation_speed: f32,
    /// `true` when the state is valid and `ω > 0`, so `mean_collision_time` is
    /// finite.
    pub mean_collision_time_valid: bool,
    /// `true` when every construction gate passes, else `false`.
    pub valid: bool,
}

/// Encodes one [`KineticTheoryEnskogQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &KineticTheoryEnskogQuery) -> GpuQuery {
    GpuQuery {
        number_density: q.number_density,
        grain_diameter: q.grain_diameter,
        solid_fraction: q.solid_fraction,
        granular_temperature: q.granular_temperature,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`KineticTheoryEnskogResult`].
fn decode_result(raw: &GpuResult) -> KineticTheoryEnskogResult {
    KineticTheoryEnskogResult {
        pair_correlation: raw.pair_correlation,
        mean_free_path: raw.mean_free_path,
        collision_frequency: raw.collision_frequency,
        mean_collision_time: raw.mean_collision_time,
        thermal_velocity_scale: raw.thermal_velocity_scale,
        rms_fluctuation_speed: raw.rms_fluctuation_speed,
        mean_collision_time_valid: raw.mean_collision_time_valid != 0,
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

/// A compiled, reusable Enskog kinetic-theory compute pipeline, twinning the
/// `CPU` golden `prism_physics_core::collider::kinetic_theory` descriptors.
pub struct GpuKineticTheoryEnskog {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuKineticTheoryEnskog {
    /// Compiles the Enskog kinetic-theory kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuKineticTheoryEnskog {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_kinetic_theory_enskog"),
            source: ShaderSource::Wgsl(KINETIC_THEORY_ENSKOG_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_kinetic_theory_enskog_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_kinetic_theory_enskog_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_kinetic_theory_enskog_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuKineticTheoryEnskog {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`KineticTheoryEnskogResult`] per input, in order.
    ///
    /// The `valid` flags match the reference exactly and the six descriptors to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[KineticTheoryEnskogQuery],
    ) -> Vec<KineticTheoryEnskogResult> {
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
            label: Some("prism_volumetric_kinetic_theory_enskog_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_kinetic_theory_enskog_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_kinetic_theory_enskog_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_kinetic_theory_enskog_bind_group"),
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
            label: Some("prism_volumetric_kinetic_theory_enskog_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_kinetic_theory_enskog_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_kinetic_theory_enskog_pass"),
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
