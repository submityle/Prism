//! `wgpu` compute twin of the bulk-powder flowability indices of the `CPU`
//! golden `prism_physics_core::collider::flowability::PowderFlowability`.
//!
//! Granular flowability is graded from two bulk measurements: the poured
//! (aerated) bulk density `ρ_b` and the tapped density `ρ_t`. The two densities
//! define the Carr compressibility index `C = 100 · (ρ_t − ρ_b)/ρ_t` (percent)
//! and the Hausner ratio `H = ρ_t / ρ_b` (`≥ 1`), from which a qualitative USP
//! flow character is read off the Carr index on a seven-level scale.
//!
//! This module ports that single stateless derivation onto the device: one
//! thread resolves one query, so a passing real-device parity test is direct
//! evidence the ported kernel computes the same indices and classification the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `PowderFlowability::from_densities`
//! followed by `carr_index`, `hausner_ratio` and `flow_character`:
//!
//! * The inputs are valid only when `ρ_b` and `ρ_t` are finite and strictly
//!   positive and `ρ_t ≥ ρ_b` (tapping can only compact the bed), and both
//!   derived indices are finite.
//! * `carr_index = 100 · (ρ_t − ρ_b)/ρ_t`, `hausner_ratio = ρ_t / ρ_b`.
//! * `flow_character` is the USP class as a `u32` `0..=6`: `C ≤ 10` Excellent
//!   `0`, `≤ 15` Good `1`, `≤ 20` Fair `2`, `≤ 25` Passable `3`, `≤ 31` Poor
//!   `4`, `≤ 37` `VeryPoor` `5`, else `ExtremelyPoor` `6`.
//! * An invalid query yields `valid = 0` with both indices `0` and class `0`.
//!
//! # Correctness model
//!
//! The arithmetic threads through a subtraction, two divisions and a multiply,
//! so `CPU` and `GPU` evaluate the same closed form but need not be bit-exact
//! (a `GPU` may contract a multiply-add). Each continuous output is compared
//! with an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The
//! discrete `flow_character` and `valid` flags are compared exactly; the parity
//! test keeps the Carr index clear of the `10 / 15 / 20 / 25 / 31 / 37`
//! classification thresholds so round-off cannot flip the discrete class.
//!
//! # Degenerate inputs
//!
//! A non-finite or non-positive density, or `ρ_t < ρ_b`, yields `valid = 0`
//! with all outputs `0`. The divisors are guarded with a `select` so the
//! invalid branch divides by `1` and no `NaN`/`Inf` leaks into an output. An
//! empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ - * /`,
//! ordered compares and `select` with unsigned index arithmetic — with no
//! `sin`, `cos`, `tan`, `log`, `pow`, `exp`, no `round`, no `f32` remainder and
//! no `sqrt`, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! Finiteness is tested with the ordered compare `abs(x) < 3.0e38` (which
//! rejects both infinities and `NaN`) rather than a bare `x == x`, and the
//! classification is a chain of ordered `<=` compares; there is no `f32`
//! equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::flowability`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` powder flowability kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `PowderFlowability`; see the module documentation for the
/// closed form.
const POWDER_FLOWABILITY_INDEX_WGSL: &str = r#"
// Powder flowability twin: one thread per query reproduces the Carr index,
// Hausner ratio and USP flow character of PowderFlowability. It uses only the
// portable core-WGSL subset (abs, + - * /, ordered compares and select plus
// unsigned index math), has no loop, and classifies with a chain of ordered
// <= compares, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) fed to select; there is
// no f32 equality.

struct Params {
    // Number of queries in the input and output buffers.
    count: u32,
    // Padding to a 16-byte uniform block.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Poured/aerated bulk density rho_b (> 0).
    bulk_density: f32,
    // Tapped density rho_t (>= rho_b).
    tapped_density: f32,
    // Padding to a 16-byte stride.
    pad0: f32,
    pad1: f32,
}

struct Flow {
    // Carr compressibility index 100*(rho_t - rho_b)/rho_t when valid, else 0.
    carr_index: f32,
    // Hausner ratio rho_t/rho_b when valid, else 0.
    hausner_ratio: f32,
    // USP flow character 0..=6 when valid, else 0.
    flow_character: u32,
    // 1 when the inputs and derived indices are valid, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Flow>;

const FINITE_LIMIT: f32 = 3.0e38;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let rb = q.bulk_density;
    let rt = q.tapped_density;

    // Validity gate, all ordered compares (NaN comparisons are false). bulk and
    // tapped must be finite and strictly positive and the tapped density must
    // not fall below the poured density.
    let valid_rb = (abs(rb) < FINITE_LIMIT) && (rb > 0.0);
    let valid_rt = (abs(rt) < FINITE_LIMIT) && (rt > 0.0);
    let order_ok = rt >= rb;
    let base_valid = valid_rb && valid_rt && order_ok;

    // Guard divisors so the invalid branch divides by 1 and never emits NaN/Inf.
    let safe_rt = select(1.0, rt, base_valid);
    let safe_rb = select(1.0, rb, base_valid);
    let carr = 100.0 * (rt - rb) / safe_rt;
    let hausner = rt / safe_rb;

    // Derived indices must also be finite to mirror the golden's final check.
    let outputs_finite = (abs(carr) < FINITE_LIMIT) && (abs(hausner) < FINITE_LIMIT);
    let valid = base_valid && outputs_finite;

    // USP flow character from the Carr index: a chain of ordered <= compares.
    var fc: u32 = 6u;
    if (carr <= 10.0) {
        fc = 0u;
    } else if (carr <= 15.0) {
        fc = 1u;
    } else if (carr <= 20.0) {
        fc = 2u;
    } else if (carr <= 25.0) {
        fc = 3u;
    } else if (carr <= 31.0) {
        fc = 4u;
    } else if (carr <= 37.0) {
        fc = 5u;
    } else {
        fc = 6u;
    }

    var out: Flow;
    out.carr_index = select(0.0, carr, valid);
    out.hausner_ratio = select(0.0, hausner, valid);
    out.flow_character = select(0u, fc, valid);
    out.valid = select(0u, 1u, valid);
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
/// the two bulk densities padded to `4` `f32` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    bulk_density: f32,
    tapped_density: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Flow` struct:
/// the Carr index, Hausner ratio, flow-character class and validity flag — `4`
/// words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    carr_index: f32,
    hausner_ratio: f32,
    flow_character: u32,
    valid: u32,
}

/// One powder flowability query: the poured bulk density and the tapped
/// density.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowderFlowabilityIndexQuery {
    /// Poured/aerated bulk density `ρ_b` (expected `> 0`).
    pub bulk_density: f32,
    /// Tapped density `ρ_t` (expected `≥ ρ_b`).
    pub tapped_density: f32,
}

impl PowderFlowabilityIndexQuery {
    /// Builds a query from the poured bulk density and the tapped density.
    #[must_use]
    pub fn new(bulk_density: f32, tapped_density: f32) -> PowderFlowabilityIndexQuery {
        PowderFlowabilityIndexQuery {
            bulk_density,
            tapped_density,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `PowderFlowability` derivation for that density pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowderFlowabilityIndexResult {
    /// Carr compressibility index `100 · (ρ_t − ρ_b)/ρ_t` when valid, else `0`.
    pub carr_index: f32,
    /// Hausner ratio `ρ_t / ρ_b` when valid, else `0`.
    pub hausner_ratio: f32,
    /// USP flow-character class `0..=6` when valid, else `0`: `0` Excellent,
    /// `1` Good, `2` Fair, `3` Passable, `4` Poor, `5` `VeryPoor`, `6`
    /// `ExtremelyPoor`.
    pub flow_character: u32,
    /// `true` when the inputs and derived indices are valid, else `false`.
    pub valid: bool,
}

/// Encodes one [`PowderFlowabilityIndexQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &PowderFlowabilityIndexQuery) -> GpuQuery {
    GpuQuery {
        bulk_density: q.bulk_density,
        tapped_density: q.tapped_density,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`PowderFlowabilityIndexResult`].
fn decode_result(raw: &GpuResult) -> PowderFlowabilityIndexResult {
    PowderFlowabilityIndexResult {
        carr_index: raw.carr_index,
        hausner_ratio: raw.hausner_ratio,
        flow_character: raw.flow_character,
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

/// A compiled, reusable powder flowability compute pipeline, twinning the `CPU`
/// golden `PowderFlowability`.
pub struct GpuPowderFlowabilityIndex {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPowderFlowabilityIndex {
    /// Compiles the powder flowability kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPowderFlowabilityIndex {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_powder_flowability_index"),
            source: ShaderSource::Wgsl(POWDER_FLOWABILITY_INDEX_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_powder_flowability_index_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_powder_flowability_index_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_powder_flowability_index_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPowderFlowabilityIndex {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`PowderFlowabilityIndexResult`] per input, in order.
    ///
    /// The `flow_character` and `valid` flags match the reference exactly and
    /// each continuous output to the module's tolerance. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[PowderFlowabilityIndexQuery],
    ) -> Vec<PowderFlowabilityIndexResult> {
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
            label: Some("prism_volumetric_powder_flowability_index_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_powder_flowability_index_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_powder_flowability_index_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_powder_flowability_index_bind_group"),
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
            label: Some("prism_volumetric_powder_flowability_index_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_powder_flowability_index_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_powder_flowability_index_pass"),
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
