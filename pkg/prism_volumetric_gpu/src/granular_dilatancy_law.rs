//! `wgpu` compute twin of the granular linear dilatancy law, from the `CPU`
//! golden `prism_physics_core::collider::granular_rheology`'s `DilatancyLaw`.
//!
//! The dilatancy law captures the decrease of a granular packing fraction from
//! its dense static value `phi_max` as the inertial number `I` grows, with
//! slope `a` (the dilatancy coefficient): `phi(I) = clamp(phi_max - a*I, 0,
//! phi_max)`. This module ports that single stateless closed form onto the
//! device: one thread resolves one query, so a passing real-device parity test
//! is direct evidence the ported kernel computes the same packing fraction the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `DilatancyLaw::new` validity plus
//! `DilatancyLaw::volume_fraction` for one `(phi_max, slope, I)` triple:
//!
//! * The model is valid only when `phi_max` is finite and `phi_max in (0, 1]`
//!   and `slope` is finite and `slope >= 0`. An invalid model yields
//!   `valid = 0`, `volume_fraction = 0`.
//! * Otherwise, a non-finite or non-positive inertial number returns the dense
//!   limit `phi_max`; a finite positive inertial number returns
//!   `clamp(phi_max - slope*I, 0, phi_max)`.
//!
//! # Correctness model
//!
//! The continuous arithmetic (one multiply, one subtract and a `clamp`) threads
//! through operators that a `GPU` may contract, so `CPU` and `GPU` are not
//! necessarily bit-exact; the valid `volume_fraction` scalar is compared with
//! an `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete
//! `valid` flag is compared exactly. The parity sweep stays away from the clamp
//! knee `I = phi_max / slope`, where `CPU`/`GPU` round-off could straddle the
//! clamp segment boundary, while still covering the unsaturated, dense-limit and
//! saturated regimes separately.
//!
//! # Degenerate inputs
//!
//! An out-of-range or non-finite model parameter yields `valid = 0` with
//! `volume_fraction = 0`. A non-finite or non-positive inertial number is not
//! an error: it selects the dense limit `phi_max`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `+ - *`, `select` and unsigned index arithmetic — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder and no `sqrt`, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested with
//! the ordered compare `abs(x) < 3.0e38` (which rejects both infinities and
//! `NaN`) rather than a bare `x == x`, and ranges with ordered `>` / `<=` / `>=`
//! compares; there is no `f32` equality anywhere.
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

/// The portable core-`WGSL` dilatancy-law kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `DilatancyLaw::new` plus `DilatancyLaw::volume_fraction`; see
/// the module documentation for the closed form.
const GRANULAR_DILATANCY_LAW_WGSL: &str = r#"
// Dilatancy-law twin: one thread per query reproduces volume_fraction gated by
// the model validity. It uses only the portable core-WGSL subset (abs, clamp,
// + - *, select plus unsigned index math), takes no optional feature, and has
// no loop and no branch, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) and ranges are ordered
// > / <= / >= compares, all fed to select; no bare f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Dense static packing fraction phi_max, required in (0, 1].
    phi_max: f32,
    // Dilatancy slope a, required >= 0.
    slope: f32,
    // Inertial number I; non-finite or <= 0 selects the dense limit.
    inertial_number: f32,
    // Padding word to a 16-byte-friendly stride.
    pad0: f32,
}

struct Result {
    // phi(I) when the model is valid, else 0.
    volume_fraction: f32,
    // 1 when the model parameters are valid, else 0.
    valid: u32,
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
    let phi_max = q.phi_max;
    let slope = q.slope;
    let inertial = q.inertial_number;

    // Model validity mirrors DilatancyLaw::new: phi_max finite and in (0, 1],
    // slope finite and >= 0. Finiteness is an ordered abs < 3.0e38 compare,
    // which is false for both infinities and NaN; no bare f32 equality.
    let phi_finite = abs(phi_max) < FINITE_LIMIT;
    let slope_finite = abs(slope) < FINITE_LIMIT;
    let phi_in_range = (phi_max > 0.0) && (phi_max <= 1.0);
    let slope_ok = slope >= 0.0;
    let ok = phi_finite && slope_finite && phi_in_range && slope_ok;

    // Dense limit when the inertial number is non-finite or non-positive. The
    // non-finite test is the negated ordered finiteness compare, so NaN (whose
    // comparisons are all false) is correctly caught as non-finite.
    let inertial_nonfinite = !(abs(inertial) < FINITE_LIMIT);
    let dense = inertial_nonfinite || (inertial <= 0.0);

    // Linear branch clamped to [0, phi_max]; the dense branch returns phi_max.
    // select picks the dense value whenever the inertial number is degenerate,
    // so the possibly-NaN linear arm is discarded in that case.
    let linear = clamp(phi_max - slope * inertial, 0.0, phi_max);
    let phi = select(linear, phi_max, dense);

    var out: Result;
    out.volume_fraction = select(0.0, phi, ok);
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// Three scalars padded to `4` `f32` words (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    phi_max: f32,
    slope: f32,
    inertial_number: f32,
    pad0: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the packing fraction and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    volume_fraction: f32,
    valid: u32,
}

/// One dilatancy-law query: the model parameters plus the inertial number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularDilatancyLawQuery {
    /// Dense static packing fraction `phi_max`, required in `(0, 1]`.
    pub phi_max: f32,
    /// Dilatancy slope `a`, required `>= 0`.
    pub slope: f32,
    /// Inertial number `I`; non-finite or `<= 0` selects the dense limit.
    pub inertial_number: f32,
}

impl GranularDilatancyLawQuery {
    /// Builds a query from the model parameters and the inertial number.
    #[must_use]
    pub fn new(phi_max: f32, slope: f32, inertial_number: f32) -> GranularDilatancyLawQuery {
        GranularDilatancyLawQuery {
            phi_max,
            slope,
            inertial_number,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `DilatancyLaw` output for that triple.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GranularDilatancyLawResult {
    /// The packing fraction `phi(I)` when the model is valid, else `0`.
    pub volume_fraction: f32,
    /// `1` when the model parameters are valid, else `0`.
    pub valid: u32,
}

/// Encodes one [`GranularDilatancyLawQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &GranularDilatancyLawQuery) -> GpuQuery {
    GpuQuery {
        phi_max: q.phi_max,
        slope: q.slope,
        inertial_number: q.inertial_number,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`GranularDilatancyLawResult`].
fn decode_result(raw: &GpuResult) -> GranularDilatancyLawResult {
    GranularDilatancyLawResult {
        volume_fraction: raw.volume_fraction,
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

/// A compiled, reusable dilatancy-law compute pipeline, twinning the `CPU`
/// golden `DilatancyLaw`.
pub struct GpuGranularDilatancyLaw {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGranularDilatancyLaw {
    /// Compiles the dilatancy-law kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGranularDilatancyLaw {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_granular_dilatancy_law"),
            source: ShaderSource::Wgsl(GRANULAR_DILATANCY_LAW_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_granular_dilatancy_law_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_granular_dilatancy_law_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_granular_dilatancy_law_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGranularDilatancyLaw {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`GranularDilatancyLawResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the
    /// `volume_fraction` scalar to the module's tolerance. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GranularDilatancyLawQuery],
    ) -> Vec<GranularDilatancyLawResult> {
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
            label: Some("prism_volumetric_granular_dilatancy_law_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_granular_dilatancy_law_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_granular_dilatancy_law_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_granular_dilatancy_law_bind_group"),
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
            label: Some("prism_volumetric_granular_dilatancy_law_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_granular_dilatancy_law_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_granular_dilatancy_law_pass"),
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
