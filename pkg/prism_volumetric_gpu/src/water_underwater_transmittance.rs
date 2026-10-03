//! `wgpu` compute twin of the stateless underwater-visibility primitives
//! [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance)
//! and
//! [`is_visible`](prism_render_architecture::water::underwater::is_visible).
//!
//! The underwater frontend attenuates radiance through turbid water with a
//! `Beer-Lambert` law: transmittance is `exp(-extinction * distance)` evaluated
//! by the crate's hand-rolled
//! [`exp_approx`](prism_render_architecture::water::exp_approx) (the workspace
//! determinism policy forbids `f32::exp`), clamped into `0..=1`. A target stays
//! visible while that transmittance holds at or above a transmittance
//! `threshold`. Both quantities are pure clamp, multiply and add arithmetic with
//! no floating-point transcendental, so the port is faithful.
//!
//! [`GpuWaterUnderwaterTransmittance`] is the on-device twin of those two
//! primitives. One thread solves one query, reproducing the reference's
//! floored, clamped `Beer-Lambert` transmittance and its visibility predicate,
//! so a passing real-device parity test is direct evidence the ported kernel
//! computes the same attenuation the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces both reference outputs for each query:
//!
//! * `transmittance`:
//!   [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance)
//!   — the extinction and distance are floored to zero, their product negated,
//!   fed through `exp_approx`, and the result clamped into `0..=1`.
//! * `visible`:
//!   [`is_visible`](prism_render_architecture::water::underwater::is_visible) —
//!   `1` when the transmittance is at or above the `0..=1`-clamped threshold,
//!   else `0`, encoded as a `u32` so the `bool` survives readback.
//!
//! The reference
//! [`exp_approx`](prism_render_architecture::water::exp_approx) is reproduced
//! verbatim in `WGSL`: the limit identity `exp(x) = (1 + x / 4096)^4096`
//! evaluated by twelve squarings — pure add and multiply, with the base floored
//! to zero so an extreme negative argument saturates to `0` instead of going
//! negative, exactly as the reference does.
//!
//! # What stays on the host
//!
//! Nothing in these two primitives is stateful: they are closed-form `f32`
//! maps. The surrounding underwater frontend — the per-channel depth color
//! shift, the phase function, the god-ray accumulation and the frontend
//! selection — is separate host code and is never dispatched by this twin.
//!
//! # Correctness model
//!
//! Both the host and the device evaluate the identical twelve-squaring
//! `exp_approx` recurrence over the identical floored product, so the
//! transmittance agrees to within floating-point tolerance and the parity test
//! asserts it with an absolute-or-relative closeness check. The only discrete
//! decision — the visibility threshold comparison — is kept away from its exact
//! crossing by the fixtures, so a last-place rounding difference cannot flip the
//! `visible` flag; that flag is then asserted with an exact integer `==`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`,
//! multiply, divide, add, ordered comparison and a bounded `for` loop — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, no inverse trigonometry, no `sqrt` and no
//! `u64`. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The loop runs a fixed twelve iterations, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::underwater`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` underwater-transmittance kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` goldens
/// [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance)
/// and [`is_visible`](prism_render_architecture::water::underwater::is_visible);
/// see the module documentation for the algorithm.
const WATER_UNDERWATER_TRANSMITTANCE_WGSL: &str = r#"
// Underwater-visibility twin: one thread resolves one query into a Beer-Lambert
// transmittance (exp_approx of the negated, floored extinction-distance product,
// clamped to 0..=1) and a visibility flag (transmittance at or above the
// clamped threshold), mirroring the CPU goldens
// `water::underwater::{beer_lambert_transmittance, is_visible}` with only clamp,
// max, multiply, divide, add and a bounded squaring loop. It owns no frontend
// selection or shared base services; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::underwater；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Extinction coefficient; floored to zero before use.
    extinction: f32,
    // Path length / distance through the water; floored to zero before use.
    dist: f32,
    // Visibility transmittance threshold; clamped into 0..=1 before compare.
    threshold: f32,
}

struct Result {
    // Clamped Beer-Lambert transmittance in 0..=1.
    transmittance: f32,
    // 1 when the transmittance is at or above the clamped threshold, else 0.
    visible: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Hand-rolled exp via the limit identity exp(x) = (1 + x/4096)^4096 evaluated by
// twelve squarings — pure add/multiply. The base is floored to zero so an
// extreme negative argument saturates to 0 instead of going negative. This is a
// verbatim port of the reference `water::exp_approx`.
fn exp_approx(x: f32) -> f32 {
    var base: f32 = 1.0 + x / 4096.0;
    if (base < 0.0) {
        base = 0.0;
    }
    for (var i: i32 = 0; i < 12; i = i + 1) {
        base = base * base;
    }
    return base;
}

// Beer-Lambert transmittance: floor the inputs, negate their product, exp and
// clamp into 0..=1.
fn beer_lambert(extinction: f32, dist: f32) -> f32 {
    let e = max(extinction, 0.0);
    let d = max(dist, 0.0);
    return clamp(exp_approx(-(e * d)), 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let t = beer_lambert(q.extinction, q.dist);
    let thr = clamp(q.threshold, 0.0, 1.0);

    var out: Result;
    out.transmittance = t;
    // Encode the visibility bool as a u32 so it survives storage readback.
    if (t >= thr) {
        out.visible = 1u;
    } else {
        out.visible = 0u;
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_UNDERWATER_TRANSMITTANCE_WGSL`].
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
/// the extinction, the distance and the visibility threshold.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Extinction coefficient (the golden `extinction`).
    extinction: f32,
    /// Path length through the water (the golden `distance`).
    dist: f32,
    /// Visibility transmittance threshold (the golden `threshold`).
    threshold: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the clamped transmittance and the `u32`-encoded visibility flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Clamped `Beer-Lambert` transmittance in `0..=1`.
    transmittance: f32,
    /// `1` when visible (transmittance at or above the threshold), else `0`.
    visible: u32,
}

/// One underwater-visibility query: the extinction, the path distance and the
/// transmittance threshold, mirroring the arguments the goldens
/// [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance)
/// and [`is_visible`](prism_render_architecture::water::underwater::is_visible)
/// read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterUnderwaterTransmittanceQuery {
    /// Extinction coefficient (the golden `extinction`); floored to zero.
    pub extinction: f32,
    /// Path length through the water (the golden `distance`); floored to zero.
    pub distance: f32,
    /// Visibility transmittance threshold (the golden `threshold`); clamped
    /// into `0..=1`.
    pub threshold: f32,
}

impl WaterUnderwaterTransmittanceQuery {
    /// Builds a query from the extinction, the path distance and the
    /// visibility threshold.
    #[must_use]
    pub const fn new(
        extinction: f32,
        distance: f32,
        threshold: f32,
    ) -> WaterUnderwaterTransmittanceQuery {
        WaterUnderwaterTransmittanceQuery {
            extinction,
            distance,
            threshold,
        }
    }
}

/// One resolved underwater-visibility response, mirroring the pair of golden
/// outputs with the `bool` visibility flag encoded as a `u32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterUnderwaterTransmittanceResult {
    /// Clamped `Beer-Lambert` transmittance (the golden
    /// `beer_lambert_transmittance`).
    pub transmittance: f32,
    /// Visibility flag encoded as a `u32` (`1` for the golden `is_visible`
    /// returning `true`, else `0`).
    pub visible: u32,
}

/// Encodes one [`WaterUnderwaterTransmittanceQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &WaterUnderwaterTransmittanceQuery) -> GpuQuery {
    GpuQuery {
        extinction: q.extinction,
        dist: q.distance,
        threshold: q.threshold,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterUnderwaterTransmittanceResult`].
fn decode_result(raw: &GpuResult) -> WaterUnderwaterTransmittanceResult {
    WaterUnderwaterTransmittanceResult {
        transmittance: raw.transmittance,
        visible: raw.visible,
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

/// A compiled, reusable underwater-transmittance compute pipeline, twinning the
/// stateless `f32` primitives of the `CPU` goldens
/// [`beer_lambert_transmittance`](prism_render_architecture::water::underwater::beer_lambert_transmittance)
/// and [`is_visible`](prism_render_architecture::water::underwater::is_visible).
pub struct GpuWaterUnderwaterTransmittance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterUnderwaterTransmittance {
    /// Compiles the underwater-transmittance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterUnderwaterTransmittance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_underwater_transmittance"),
            source: ShaderSource::Wgsl(WATER_UNDERWATER_TRANSMITTANCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_underwater_transmittance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_underwater_transmittance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_underwater_transmittance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterUnderwaterTransmittance {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`WaterUnderwaterTransmittanceResult`] per input, in order.
    ///
    /// The transmittance equals the reference to within floating-point
    /// tolerance and the visibility flag matches exactly. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterUnderwaterTransmittanceQuery],
    ) -> Vec<WaterUnderwaterTransmittanceResult> {
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
            label: Some("prism_volumetric_water_underwater_transmittance_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_underwater_transmittance_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_underwater_transmittance_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_underwater_transmittance_bind_group"),
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
            label: Some("prism_volumetric_water_underwater_transmittance_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_underwater_transmittance_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_underwater_transmittance_pass"),
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
