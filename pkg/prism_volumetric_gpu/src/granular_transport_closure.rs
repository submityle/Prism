//! `wgpu` compute twin of the dense-granular `KTGF` transport closure, from the
//! `CPU` golden `prism_physics_core::collider::granular_transport`'s
//! `GranularTransport`.
//!
//! Lun–Savage–Jeffrey–Chepurniy kinetic-theory closures give two uncoupled,
//! independently unit-testable closed forms for a dense granular gas:
//!
//! * the **bulk viscosity**
//!   `xi = (4/3) * rho_s * phi^2 * d * g0 * (1 + e) * sqrt(Theta / pi)`, and
//! * the **collisional dissipation rate**
//!   `gamma = (12 * (1 - e^2) / (d * sqrt(pi))) * rho_s * phi^2 * g0 * Theta^(3/2)`.
//!
//! Here `rho_s` is the solid density, `phi` the volume fraction, `d` the grain
//! diameter, `g0` the radial distribution function at contact, `e` the normal
//! restitution and `Theta` the granular temperature. This module ports both
//! stateless closed forms onto the device: one thread resolves one query, so a
//! passing real-device parity test is direct evidence the ported kernel
//! computes the same transport coefficients the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries the fixed restitution `e` and a state
//! `(rho_s, phi, d, g0, Theta)`:
//!
//! * The model is rejected when `e` is non-finite or outside `[0, 1]`.
//! * The state is rejected unless every component is finite and
//!   `rho_s > 0`, `phi in [0, 1)`, `d > 0`, `g0 >= 1`, `Theta >= 0`.
//! * On rejection the result is `valid = 0` with both coefficients `0`.
//! * Otherwise `xi` and `gamma` are evaluated in the golden operator order,
//!   with `Theta^(3/2) = Theta * sqrt(Theta)` (no `pow`). The elastic limit
//!   `e = 1` gives `gamma = 0`; `Theta = 0` gives both `sqrt(Theta) = 0` and so
//!   `gamma = 0` while `xi = 0`.
//!
//! # Correctness model
//!
//! The continuous arithmetic threads through multiplies, adds, a division and a
//! `sqrt`, which a `GPU` may contract, so `CPU` and `GPU` are not necessarily
//! bit-exact; each valid coefficient is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete
//! `valid` flag is compared exactly; the parity sweep keeps the state strictly
//! inside its valid box so the validity decision cannot be flipped by round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite or out-of-range model or state yields `valid = 0` with both
//! coefficients `0`. When the state is valid `d > 0`, so the division is well
//! defined; the kernel still feeds the divisor through a `select` guard so the
//! un-taken (invalid) branch never divides by zero. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sqrt`,
//! `+ - * /`, `select` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no `u64`,
//! `i64`, `f64`, `u16` or `i16`, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. Finiteness is tested with the ordered compare `abs(x) < 3.0e38`
//! (which rejects both infinities and `NaN`) rather than a bare `x == x`, and
//! validity with ordered range compares; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::granular_transport`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` granular-transport kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `GranularTransport`; see the module documentation for the
/// closed forms.
const GRANULAR_TRANSPORT_CLOSURE_WGSL: &str = r#"
// Granular-transport twin: one thread per query reproduces the bulk viscosity
// and collisional dissipation of the Lun et al. KTGF closure. It uses only the
// portable core-WGSL subset (abs, sqrt, + - * /, select plus unsigned index
// math), takes no optional feature, and has no loop and no branch, so it
// provably terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) and the validity box is a set of ordered range compares,
// all fed to select. There is no bare f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Normal restitution e, valid in [0, 1].
    restitution: f32,
    // Solid density rho_s (> 0).
    rho_s: f32,
    // Volume fraction phi ([0, 1)).
    phi: f32,
    // Grain diameter d (> 0).
    d: f32,
    // Radial distribution function at contact g0 (>= 1).
    g0: f32,
    // Granular temperature Theta (>= 0).
    theta: f32,
    // Padding words to a 32-byte std430 stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Bulk viscosity xi when valid, else 0.
    bulk_viscosity: f32,
    // Collisional dissipation gamma when valid, else 0.
    collisional_dissipation: f32,
    // 1 when the model and state are valid, else 0.
    valid: u32,
    // Padding word to a 16-byte std430 stride.
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;
// core::f32::consts::PI rounded to f32.
const PI: f32 = 3.1415927;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let e = q.restitution;
    let rho_s = q.rho_s;
    let phi = q.phi;
    let d = q.d;
    let g0 = q.g0;
    let theta = q.theta;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let finite = (abs(e) < FINITE_LIMIT)
        && (abs(rho_s) < FINITE_LIMIT)
        && (abs(phi) < FINITE_LIMIT)
        && (abs(d) < FINITE_LIMIT)
        && (abs(g0) < FINITE_LIMIT)
        && (abs(theta) < FINITE_LIMIT);

    // Model valid when e in [0, 1]; state valid per the golden valid_state.
    let model_ok = (e >= 0.0) && (e <= 1.0);
    let state_ok = (rho_s > 0.0)
        && (phi >= 0.0)
        && (phi < 1.0)
        && (d > 0.0)
        && (g0 >= 1.0)
        && (theta >= 0.0);
    let ok = finite && model_ok && state_ok;

    // Guard the divisor so the un-taken (invalid) branch never divides by zero;
    // when ok, d is strictly positive.
    let d_denom = select(1.0, d, ok);

    let phi_sq = phi * phi;
    let sqrt_theta = sqrt(theta);

    // Bulk viscosity in the golden operator order:
    // (4/3)*rho_s*phi^2*d*g0*(1+e)*sqrt(Theta/pi).
    let sqrt_theta_over_pi = sqrt(theta / PI);
    let xi = (4.0 / 3.0) * rho_s * phi_sq * d * g0 * (1.0 + e) * sqrt_theta_over_pi;

    // Collisional dissipation in the golden operator order:
    // (12*(1-e^2)/(d*sqrt(pi)))*rho_s*phi^2*g0*Theta^(3/2), Theta^(3/2)=Theta*sqrt(Theta).
    let one_minus_e2 = 1.0 - e * e;
    let sqrt_pi = sqrt(PI);
    let theta_three_halves = theta * sqrt_theta;
    let prefactor = 12.0 * one_minus_e2 / (d_denom * sqrt_pi);
    let gamma = prefactor * rho_s * phi_sq * g0 * theta_three_halves;

    var out: Result;
    out.bulk_viscosity = select(0.0, xi, ok);
    out.collisional_dissipation = select(0.0, gamma, ok);
    out.valid = select(0u, 1u, ok);
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The six inputs are padded to `8` `f32` words (`32` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    restitution: f32,
    rho_s: f32,
    phi: f32,
    d: f32,
    g0: f32,
    theta: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the two coefficients, the validity flag and a padding word — `4`
/// words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    bulk_viscosity: f32,
    collisional_dissipation: f32,
    valid: u32,
    pad0: u32,
}

/// One granular-transport query: the fixed restitution plus the state
/// `(rho_s, phi, d, g0, Theta)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularTransportClosureQuery {
    /// Normal restitution `e`, valid in `[0, 1]`.
    pub restitution: f32,
    /// Solid (grain material) density `rho_s`, valid `> 0`.
    pub rho_s: f32,
    /// Volume fraction `phi`, valid in `[0, 1)`.
    pub phi: f32,
    /// Grain diameter `d`, valid `> 0`.
    pub d: f32,
    /// Radial distribution function at contact `g0`, valid `>= 1`.
    pub g0: f32,
    /// Granular temperature `Theta`, valid `>= 0`.
    pub theta: f32,
}

impl GranularTransportClosureQuery {
    /// Builds a query from the restitution and the five state parameters.
    #[must_use]
    pub fn new(
        restitution: f32,
        rho_s: f32,
        phi: f32,
        d: f32,
        g0: f32,
        theta: f32,
    ) -> GranularTransportClosureQuery {
        GranularTransportClosureQuery {
            restitution,
            rho_s,
            phi,
            d,
            g0,
            theta,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `GranularTransport` output for that model and state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularTransportClosureResult {
    /// The bulk viscosity `xi` when valid, else `0`.
    pub bulk_viscosity: f32,
    /// The collisional dissipation rate `gamma` when valid, else `0`.
    pub collisional_dissipation: f32,
    /// `1` when the model and state are valid, else `0`.
    pub valid: u32,
}

/// Encodes one [`GranularTransportClosureQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &GranularTransportClosureQuery) -> GpuQuery {
    GpuQuery {
        restitution: q.restitution,
        rho_s: q.rho_s,
        phi: q.phi,
        d: q.d,
        g0: q.g0,
        theta: q.theta,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`GranularTransportClosureResult`].
fn decode_result(raw: &GpuResult) -> GranularTransportClosureResult {
    GranularTransportClosureResult {
        bulk_viscosity: raw.bulk_viscosity,
        collisional_dissipation: raw.collisional_dissipation,
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

/// A compiled, reusable granular-transport compute pipeline, twinning the `CPU`
/// golden `GranularTransport`.
pub struct GpuGranularTransportClosure {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGranularTransportClosure {
    /// Compiles the granular-transport kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGranularTransportClosure {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_granular_transport_closure"),
            source: ShaderSource::Wgsl(GRANULAR_TRANSPORT_CLOSURE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_granular_transport_closure_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_granular_transport_closure_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_granular_transport_closure_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGranularTransportClosure {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`GranularTransportClosureResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and each coefficient to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GranularTransportClosureQuery],
    ) -> Vec<GranularTransportClosureResult> {
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
            label: Some("prism_volumetric_granular_transport_closure_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_granular_transport_closure_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_granular_transport_closure_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_granular_transport_closure_bind_group"),
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
            label: Some("prism_volumetric_granular_transport_closure_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_granular_transport_closure_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_granular_transport_closure_pass"),
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
