//! `wgpu` compute twin of the dense-granular-gas equation of state, from the
//! `CPU` golden
//! `prism_physics_core::collider::granular_eos`'s
//! `GranularEquationOfState`.
//!
//! Kinetic theory of granular flow writes the granular pressure as the sum of a
//! kinetic (streaming) term and a collisional term. For smooth inelastic
//! spheres of restitution `e` at bulk density `ρ`, granular temperature `T`,
//! solid fraction `φ` and contact radial-distribution value `g0`:
//!
//! ```text
//! p = ρ T [ 1 + 2 (1 + e) φ g0 ]
//! ```
//!
//! The bracket is the compressibility factor `Z = 1 + 2 (1 + e) φ g0`; its
//! leading `1` is the ideal-gas (kinetic) term and `2 (1 + e) φ g0` is the
//! collisional enhancement (`Z - 1`). This module ports that stateless
//! constitutive relation onto the device: one thread resolves one query, so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same pressures the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces every accessor of
//! `GranularEquationOfState` for one state:
//!
//! * The constructor gate: `e` must be finite and in `[0, 1]`, else nothing is
//!   defined.
//! * `valid_packing`: `φ` finite in `[0, 1)` and `g0` finite with `g0 >= 1`,
//!   gating the compressibility factor and collisional enhancement.
//! * `kinetic_pressure`: `ρ` finite `> 0` and `T` finite `>= 0`.
//! * `compressibility_factor` `Z = 1 + 2 (1 + e) φ g0`.
//! * `collisional_enhancement` `2 (1 + e) φ g0` (`Z - 1`).
//! * `kinetic_pressure` `p_k = ρ T`.
//! * `collisional_pressure` `p_c = p_k · (Z - 1)`, valid only when both the
//!   packing and the kinetic state are valid.
//! * `pressure` `p = p_k · Z`, valid under the same combined gate.
//!
//! # Correctness model
//!
//! The continuous arithmetic is pure multiply/add, so `CPU` and `GPU` evaluate
//! the same closed form but need not be bit-exact (a `GPU` may contract a
//! multiply-add); each valid scalar is compared with an `abs <= 1e-4 ||
//! rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete `valid` flags are
//! compared exactly; the parity test keeps random inputs well inside the valid
//! region so a validity decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! Any out-of-range or non-finite argument clears the dependent outputs to
//! `0` with their `valid` flag `0`. There are no divisions, so no divisor
//! guards are required. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - *`,
//! `select` and unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so it
//! runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with
//! the ordered compare `abs(x) < 3.0e38` (which rejects both infinities and
//! `NaN`) rather than a bare `x == x`, and ranges with ordered `>=`/`<`/`<=`;
//! there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_eos`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` equation-of-state kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `GranularEquationOfState`; see the module documentation for
/// the closed forms.
const GRANULAR_EQUATION_OF_STATE_WGSL: &str = r#"
// Granular equation-of-state twin: one thread per query reproduces the
// compressibility factor, collisional enhancement, kinetic pressure,
// collisional pressure and total pressure. It uses only the portable core-WGSL
// subset (abs, + - *, select plus unsigned index math), has no loop and no
// data-dependent branch, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) and the ranges ordered
// >=/</<= compares, all fed to select. There are no divisions.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Coefficient of normal restitution e.
    restitution: f32,
    // Solid fraction phi.
    solid_fraction: f32,
    // Contact radial-distribution value g0.
    g0: f32,
    // Bulk density rho.
    bulk_density: f32,
    // Granular temperature T.
    temperature: f32,
    // Padding words to a 32-byte stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Compressibility factor Z and its validity.
    compressibility_factor: f32,
    compressibility_valid: u32,
    // Collisional enhancement Z-1 and its validity.
    collisional_enhancement: f32,
    enhancement_valid: u32,
    // Kinetic pressure rho*T and its validity.
    kinetic_pressure: f32,
    kinetic_valid: u32,
    // Collisional pressure p_k*(Z-1) and its validity.
    collisional_pressure: f32,
    collisional_valid: u32,
    // Total pressure p_k*Z and its validity.
    pressure: f32,
    pressure_valid: u32,
    // Padding words to a 48-byte stride.
    pad0: u32,
    pad1: u32,
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

    let e = q.restitution;
    let phi = q.solid_fraction;
    let g0 = q.g0;
    let rho = q.bulk_density;
    let temp = q.temperature;

    // Constructor gate: restitution finite and in [0, 1].
    let e_ok = (abs(e) < FINITE_LIMIT) && (e >= 0.0) && (e <= 1.0);
    // Packing gate: phi finite in [0, 1), g0 finite with g0 >= 1.
    let phi_ok = (abs(phi) < FINITE_LIMIT) && (phi >= 0.0) && (phi < 1.0);
    let g0_ok = (abs(g0) < FINITE_LIMIT) && (g0 >= 1.0);
    let packing_ok = phi_ok && g0_ok;
    // Kinetic gate: density finite and > 0, temperature finite and >= 0.
    let rho_ok = (abs(rho) < FINITE_LIMIT) && (rho > 0.0);
    let temp_ok = (abs(temp) < FINITE_LIMIT) && (temp >= 0.0);
    let kinetic_ok = rho_ok && temp_ok;

    // Combined validity gates.
    let z_ok = e_ok && packing_ok;
    let kin_ok = e_ok && kinetic_ok;
    let both_ok = e_ok && packing_ok && kinetic_ok;

    // Pure multiply/add closed forms, in golden operator order.
    let enhancement = 2.0 * (1.0 + e) * phi * g0;
    let z = 1.0 + enhancement;
    let p_k = rho * temp;
    let p_c = p_k * enhancement;
    let p = p_k * z;

    var out: Result;
    out.compressibility_factor = select(0.0, z, z_ok);
    out.compressibility_valid = select(0u, 1u, z_ok);
    out.collisional_enhancement = select(0.0, enhancement, z_ok);
    out.enhancement_valid = select(0u, 1u, z_ok);
    out.kinetic_pressure = select(0.0, p_k, kin_ok);
    out.kinetic_valid = select(0u, 1u, kin_ok);
    out.collisional_pressure = select(0.0, p_c, both_ok);
    out.collisional_valid = select(0u, 1u, both_ok);
    out.pressure = select(0.0, p, both_ok);
    out.pressure_valid = select(0u, 1u, both_ok);
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
/// the five state scalars padded to `8` `f32` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    restitution: f32,
    solid_fraction: f32,
    g0: f32,
    bulk_density: f32,
    temperature: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: five quantities, five validity flags and two pads — `12` words
/// (`48` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    compressibility_factor: f32,
    compressibility_valid: u32,
    collisional_enhancement: f32,
    enhancement_valid: u32,
    kinetic_pressure: f32,
    kinetic_valid: u32,
    collisional_pressure: f32,
    collisional_valid: u32,
    pressure: f32,
    pressure_valid: u32,
    pad0: u32,
    pad1: u32,
}

/// One equation-of-state query: the restitution plus the packing and kinetic
/// state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularEquationOfStateQuery {
    /// Coefficient of normal restitution `e`, expected in `[0, 1]`.
    pub restitution: f32,
    /// Solid fraction `φ`, expected in `[0, 1)`.
    pub solid_fraction: f32,
    /// Contact radial-distribution value `g0`, expected `>= 1`.
    pub g0: f32,
    /// Bulk density `ρ`, expected `> 0`.
    pub bulk_density: f32,
    /// Granular temperature `T`, expected `>= 0`.
    pub temperature: f32,
}

impl GranularEquationOfStateQuery {
    /// Builds a query from the restitution and the packing/kinetic state.
    #[must_use]
    pub fn new(
        restitution: f32,
        solid_fraction: f32,
        g0: f32,
        bulk_density: f32,
        temperature: f32,
    ) -> GranularEquationOfStateQuery {
        GranularEquationOfStateQuery {
            restitution,
            solid_fraction,
            g0,
            bulk_density,
            temperature,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `GranularEquationOfState` accessors for that state. Each quantity carries an
/// independent validity flag (`1` defined, `0` degenerate) and is `0` when
/// invalid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularEquationOfStateResult {
    /// Compressibility factor `Z = 1 + 2 (1 + e) φ g0` when valid, else `0`.
    pub compressibility_factor: f32,
    /// `1` when the compressibility factor is defined, else `0`.
    pub compressibility_valid: u32,
    /// Collisional enhancement `2 (1 + e) φ g0` (`Z - 1`) when valid, else `0`.
    pub collisional_enhancement: f32,
    /// `1` when the collisional enhancement is defined, else `0`.
    pub enhancement_valid: u32,
    /// Kinetic pressure `p_k = ρ T` when valid, else `0`.
    pub kinetic_pressure: f32,
    /// `1` when the kinetic pressure is defined, else `0`.
    pub kinetic_valid: u32,
    /// Collisional pressure `p_c = p_k (Z - 1)` when valid, else `0`.
    pub collisional_pressure: f32,
    /// `1` when the collisional pressure is defined, else `0`.
    pub collisional_valid: u32,
    /// Total pressure `p = p_k Z` when valid, else `0`.
    pub pressure: f32,
    /// `1` when the total pressure is defined, else `0`.
    pub pressure_valid: u32,
}

/// Encodes one [`GranularEquationOfStateQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &GranularEquationOfStateQuery) -> GpuQuery {
    GpuQuery {
        restitution: q.restitution,
        solid_fraction: q.solid_fraction,
        g0: q.g0,
        bulk_density: q.bulk_density,
        temperature: q.temperature,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`GranularEquationOfStateResult`].
fn decode_result(raw: &GpuResult) -> GranularEquationOfStateResult {
    GranularEquationOfStateResult {
        compressibility_factor: raw.compressibility_factor,
        compressibility_valid: raw.compressibility_valid,
        collisional_enhancement: raw.collisional_enhancement,
        enhancement_valid: raw.enhancement_valid,
        kinetic_pressure: raw.kinetic_pressure,
        kinetic_valid: raw.kinetic_valid,
        collisional_pressure: raw.collisional_pressure,
        collisional_valid: raw.collisional_valid,
        pressure: raw.pressure,
        pressure_valid: raw.pressure_valid,
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

/// A compiled, reusable equation-of-state compute pipeline, twinning the `CPU`
/// golden `GranularEquationOfState`.
pub struct GpuGranularEquationOfState {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGranularEquationOfState {
    /// Compiles the equation-of-state kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGranularEquationOfState {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_granular_equation_of_state"),
            source: ShaderSource::Wgsl(GRANULAR_EQUATION_OF_STATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_granular_equation_of_state_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_granular_equation_of_state_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_granular_equation_of_state_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGranularEquationOfState {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`GranularEquationOfStateResult`] per input, in order.
    ///
    /// Each `valid` flag matches the reference exactly and each defined scalar
    /// to the module's tolerance. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GranularEquationOfStateQuery],
    ) -> Vec<GranularEquationOfStateResult> {
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
            label: Some("prism_volumetric_granular_equation_of_state_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_granular_equation_of_state_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_granular_equation_of_state_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_granular_equation_of_state_bind_group"),
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
            label: Some("prism_volumetric_granular_equation_of_state_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_granular_equation_of_state_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_granular_equation_of_state_pass"),
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
