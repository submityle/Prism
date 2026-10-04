//! `wgpu` compute twin of the Beverloo slot-orifice discharge correlation,
//! from the `CPU` golden `prism_physics_core::collider::hopper_discharge`'s
//! `BeverlooSlot` constructor and flow getters.
//!
//! When a packed hopper is opened, grains drain through a bottom slot at a rate
//! that is almost independent of the fill height and is instead set by the
//! orifice size. The empirical Beverloo correlation captures this: for a long
//! slot of clear width `W` and length `L` discharging grains of diameter `d`,
//! the mass flow rate is
//!
//! ```text
//! Q = C · ρ_b · √g · L · (W − k·d)^{3/2}
//! ```
//!
//! where `ρ_b` is the bulk density, `g` gravity, `C` an empirical discharge
//! coefficient and `k` a shape factor that shrinks the effective aperture
//! because grain centres cannot reach the very edge. When `W ≤ k·d` the arch
//! spans the slot and flow stops — the orifice jams.
//!
//! This module ports the `BeverlooSlot` constructor's validity gate and its
//! aperture/jam/flow getters onto the device: one thread resolves one query, so
//! a passing real-device parity test is direct evidence the ported kernel
//! computes the same quantities the reference does, not merely that the shader
//! compiles.
//!
//! Only `BeverlooSlot` is twinned; the sibling `DischargeCensus` and
//! `mass_flow_between` helpers operate on variable-length position/radius slices
//! with a reduction and are out of scope for this fixed-width, one-query kernel.
//!
//! # What is twinned
//!
//! For each query the kernel evaluates, in the golden operator order:
//!
//! * `effective_aperture = W − k·d` (may be non-positive, signalling a jam).
//! * `jams = effective_aperture ≤ 0`.
//! * `mass_flow_rate = C · ρ_b · √g · L · (aperture · √aperture)`, which is `0`
//!   for a jammed orifice.
//! * `volumetric_flow_rate = mass_flow_rate / ρ_b`.
//!
//! # Validity model
//!
//! The master validity flag mirrors the golden `BeverlooSlot::new` gate and the
//! `mass_flow_rate` argument gate combined: it is set only when the slot inputs
//! are finite with `C > 0`, `k ≥ 0`, `W > 0`, `L > 0` and the flow inputs are
//! finite with `ρ_b > 0`, `g > 0`, `d > 0`. A jammed orifice (`aperture ≤ 0`)
//! is still a valid prediction: the master flag stays set, `jams` is `true` and
//! every flow rate is zero, so `valid` distinguishes a physically jammed slot
//! (`Q = 0`) from parameters the constructor rejects (all-zero outputs with
//! `valid = false`). When the master flag is cleared the reported aperture is
//! zeroed and `jams` is `false`.
//!
//! # Correctness model
//!
//! The golden is pure `f32`; this kernel is pure `f32` too, and the host oracle
//! re-derives the same closed form in `f32` in the same operator order. The
//! `(W − k·d)^{3/2}` term is formed as `aperture · √aperture` to avoid `powf`,
//! exactly as the reference does. Continuous scalars are compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`); the sweep keeps
//! `|aperture|` away from the `aperture = 0` knee so the `^{3/2}` term's
//! relative error cannot cause a false mismatch. The `jams` and `valid` words
//! are compared exactly.
//!
//! # Degenerate inputs
//!
//! A cleared master flag yields all-zero scalars, `jams = false` and
//! `valid = false`. The `√aperture` argument is `select`-guarded to `1` unless
//! the orifice is flowing, the `√g` argument is `select`-guarded likewise, and
//! the `/ ρ_b` divisor is `select`-guarded to `1` unless the master flag is set,
//! so no `inf`/`NaN` survives into a discarded branch. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses the portable core-`WGSL` subset — `abs`, `sqrt`, `select`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round` and no `f32` remainder, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the ordered
//! compare `abs(x) < 3.0e38` (which rejects both infinities and `NaN`) rather
//! than a bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::hopper_discharge`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Beverloo slot-discharge kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `BeverlooSlot` constructor and getters; see the
/// module documentation for the closed form.
const BEVERLOO_SLOT_DISCHARGE_WGSL: &str = r#"
// Beverloo slot-discharge twin: one thread per query evaluates the effective
// aperture W - k*d, the jam flag, the mass flow rate C*rho*sqrt(g)*L*
// (aperture*sqrt(aperture)) and the volumetric flow rate Q/rho, from the slot
// coefficients and the flow parameters. It uses only the portable core-WGSL
// subset (abs, sqrt, select, + - * / plus unsigned index math), has no loop and
// no branch, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) fed to select; no bare
// f32 equality anywhere. Every divisor and sqrt argument is select-guarded so
// no inf/NaN survives into a discarded degenerate branch.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Empirical discharge coefficient C (> 0).
    discharge_coeff: f32,
    // Shape factor k (>= 0) shrinking the effective aperture.
    shape_factor: f32,
    // Clear slot width W (> 0), the flow-limiting dimension.
    width: f32,
    // Slot length L (> 0), entering the flow rate linearly.
    slot_length: f32,
    // Bulk density rho_b (> 0).
    bulk_density: f32,
    // Gravitational acceleration g (> 0).
    gravity: f32,
    // Grain diameter d (> 0).
    grain_diameter: f32,
    // Padding to a 16-byte-friendly stride.
    pad0: f32,
}

struct Result {
    // Effective aperture W - k*d.
    effective_aperture: f32,
    // Mass flow rate Q.
    mass_flow_rate: f32,
    // Volumetric flow rate Q / rho_b.
    volumetric_flow_rate: f32,
    // Jam flag (aperture <= 0 while valid).
    jams: u32,
    // Master validity (constructor and flow-argument gate passed).
    valid: u32,
    // Padding to a 16-byte-friendly stride.
    pad0: u32,
    pad1: u32,
    pad2: u32,
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

    let c = q.discharge_coeff;
    let k = q.shape_factor;
    let w = q.width;
    let slot_length = q.slot_length;
    let rho = q.bulk_density;
    let grav = q.gravity;
    let d = q.grain_diameter;

    // Constructor validity gate (BeverlooSlot::new), in order.
    let finite_slot = (abs(c) < FINITE_LIMIT)
        && (abs(k) < FINITE_LIMIT)
        && (abs(w) < FINITE_LIMIT)
        && (abs(slot_length) < FINITE_LIMIT);
    let slot_ok = finite_slot
        && (c > 0.0)
        && (k >= 0.0)
        && (w > 0.0)
        && (slot_length > 0.0);

    // Flow-argument gate (mass_flow_rate), in order.
    let finite_flow = (abs(rho) < FINITE_LIMIT)
        && (abs(grav) < FINITE_LIMIT)
        && (abs(d) < FINITE_LIMIT);
    let flow_ok = finite_flow && (rho > 0.0) && (grav > 0.0) && (d > 0.0);

    let master_ok = slot_ok && flow_ok;

    let aperture = w - k * d;
    let jam = aperture <= 0.0;
    let flowing = master_ok && (aperture > 0.0);

    // Guard the sqrt argument and gravity so the discarded branch stays finite.
    let ap_guarded = select(1.0, aperture, flowing);
    let aperture_three_halves = aperture * sqrt(ap_guarded);
    let grav_guarded = select(1.0, grav, flowing);
    let q_flow = c * rho * sqrt(grav_guarded) * slot_length * aperture_three_halves;
    let mass_flow = select(0.0, q_flow, flowing);

    // Guard the volumetric divisor before dividing by the bulk density.
    let rho_den = select(1.0, rho, master_ok);
    let vol_flow = select(0.0, mass_flow / rho_den, master_ok);

    let eff_ap_out = select(0.0, aperture, master_ok);

    var out: Result;
    out.effective_aperture = eff_ap_out;
    out.mass_flow_rate = mass_flow;
    out.volumetric_flow_rate = vol_flow;
    out.jams = select(0u, 1u, master_ok && jam);
    out.valid = select(0u, 1u, master_ok);
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the four slot coefficients, the three flow parameters and one padding word —
/// `8` `f32` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    discharge_coeff: f32,
    shape_factor: f32,
    width: f32,
    slot_length: f32,
    bulk_density: f32,
    gravity: f32,
    grain_diameter: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: three `f32` flow scalars, two `u32` flag words and three padding
/// words — `8` words (`32` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    effective_aperture: f32,
    mass_flow_rate: f32,
    volumetric_flow_rate: f32,
    jams: u32,
    valid: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One Beverloo slot-discharge query: the four coefficients that build the slot
/// model plus the three flow parameters at which to evaluate the correlation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeverlooSlotDischargeQuery {
    /// Empirical discharge coefficient `C` (`> 0`).
    pub discharge_coeff: f32,
    /// Shape factor `k` (`≥ 0`).
    pub shape_factor: f32,
    /// Clear slot width `W` (`> 0`).
    pub width: f32,
    /// Slot length `L` (`> 0`).
    pub length: f32,
    /// Bulk density `ρ_b` (`> 0`).
    pub bulk_density: f32,
    /// Gravitational acceleration `g` (`> 0`).
    pub gravity: f32,
    /// Grain diameter `d` (`> 0`).
    pub grain_diameter: f32,
}

impl BeverlooSlotDischargeQuery {
    /// Builds a query from the four slot coefficients and the three flow
    /// parameters.
    #[must_use]
    pub fn new(
        discharge_coeff: f32,
        shape_factor: f32,
        width: f32,
        length: f32,
        bulk_density: f32,
        gravity: f32,
        grain_diameter: f32,
    ) -> BeverlooSlotDischargeQuery {
        BeverlooSlotDischargeQuery {
            discharge_coeff,
            shape_factor,
            width,
            length,
            bulk_density,
            gravity,
            grain_diameter,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference correlation.
///
/// When `valid` is `false` the constructor or flow-argument gate rejected the
/// inputs and every scalar is zero with `jams = false`. When `valid` is `true`
/// the effective aperture and jam flag are meaningful; a jammed orifice reports
/// `jams = true` with every flow rate zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeverlooSlotDischargeResult {
    /// Effective aperture `W − k·d`, zero when `valid` is `false`.
    pub effective_aperture: f32,
    /// Whether the orifice jams (`aperture ≤ 0`), always `false` when `valid`
    /// is `false`.
    pub jams: bool,
    /// Mass flow rate `Q`, zero when jammed or invalid.
    pub mass_flow_rate: f32,
    /// Volumetric flow rate `Q / ρ_b`, zero when jammed or invalid.
    pub volumetric_flow_rate: f32,
    /// Master validity: the constructor and flow-argument gate accepted the
    /// inputs.
    pub valid: bool,
}

/// Encodes one [`BeverlooSlotDischargeQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &BeverlooSlotDischargeQuery) -> GpuQuery {
    GpuQuery {
        discharge_coeff: q.discharge_coeff,
        shape_factor: q.shape_factor,
        width: q.width,
        slot_length: q.length,
        bulk_density: q.bulk_density,
        gravity: q.gravity,
        grain_diameter: q.grain_diameter,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`BeverlooSlotDischargeResult`].
fn decode_result(raw: &GpuResult) -> BeverlooSlotDischargeResult {
    BeverlooSlotDischargeResult {
        effective_aperture: raw.effective_aperture,
        jams: raw.jams != 0,
        mass_flow_rate: raw.mass_flow_rate,
        volumetric_flow_rate: raw.volumetric_flow_rate,
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

/// A compiled, reusable Beverloo slot-discharge compute pipeline, twinning the
/// `CPU` golden `BeverlooSlot` constructor and getters.
pub struct GpuBeverlooSlotDischarge {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBeverlooSlotDischarge {
    /// Compiles the Beverloo slot-discharge kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBeverlooSlotDischarge {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_beverloo_slot_discharge"),
            source: ShaderSource::Wgsl(BEVERLOO_SLOT_DISCHARGE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_beverloo_slot_discharge_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_beverloo_slot_discharge_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_beverloo_slot_discharge_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBeverlooSlotDischarge {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`BeverlooSlotDischargeResult`] per input, in order.
    ///
    /// Each continuous scalar matches the reference to the module's tolerance
    /// and each flag word matches exactly. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[BeverlooSlotDischargeQuery],
    ) -> Vec<BeverlooSlotDischargeResult> {
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
            label: Some("prism_volumetric_beverloo_slot_discharge_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_beverloo_slot_discharge_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_beverloo_slot_discharge_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_beverloo_slot_discharge_bind_group"),
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
            label: Some("prism_volumetric_beverloo_slot_discharge_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_beverloo_slot_discharge_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_beverloo_slot_discharge_pass"),
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
