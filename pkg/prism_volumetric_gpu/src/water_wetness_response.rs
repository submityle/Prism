//! `wgpu` compute twin of the shoreline wetness response primitives
//! ([`wetness`](prism_render_architecture::water::wetness)).
//!
//! The `CPU` golden module derives wet-surface shading decisions over a
//! caller-owned moisture state. This twin ports its four stateless, per-point
//! numeric kernels so a shading pass can evaluate them on-device:
//!
//! * [`wet_albedo_scale`](prism_render_architecture::water::wetness::wet_albedo_scale)
//!   — the albedo multiplier at a saturation, scaling linearly from `1` (dry)
//!   down to `1 - darkening_strength` (fully wet).
//! * [`capillary_height`](prism_render_architecture::water::wetness::capillary_height)
//!   — the capillary wet-band height a distance above the waterline, rising to
//!   `max_capillary_height` at the line for a soaked surface and fading
//!   linearly to zero at the reach.
//! * [`puddle_depth`](prism_render_architecture::water::wetness::puddle_depth)
//!   — the standing puddle depth after one rain / drain step, clamped
//!   non-negative.
//! * [`is_puddle`](prism_render_architecture::water::wetness::is_puddle)
//!   — whether a depth clears the puddle threshold (any positive depth when the
//!   threshold is non-positive).
//!
//! [`GpuWaterWetnessResponse`] is the on-device twin: one thread evaluates one
//! query, branching on an operation code and reproducing the reference closed
//! form exactly, so a passing real-device parity test is direct evidence the
//! ported kernel computes the same wetness response the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each [`WaterWetnessResponseQuery`] carries the per-point drivers for one of
//! the four operations; the shared [`WaterWetnessResponseParams`] (a flattened
//! mirror of the reference `WetnessParams`) rides in the uniform block. The
//! kernel returns one [`WaterWetnessResponseResult`] — a scalar for the three
//! continuous kernels, a boolean for the puddle predicate.
//!
//! # What stays on the host
//!
//! The time-stepping envelopes
//! ([`absorb`](prism_render_architecture::water::wetness::absorb),
//! [`dry`](prism_render_architecture::water::wetness::dry),
//! [`update_wetness`](prism_render_architecture::water::wetness::update_wetness),
//! [`step_moisture`](prism_render_architecture::water::wetness::step_moisture))
//! are intentionally left on the host: they thread through the shared
//! `exp_approx` exponential envelope, which this fixed-width numeric twin does
//! not port. The empty-batch short-circuit also stays on the host, since a
//! storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The puddle predicate is a magnitude comparison, so its boolean outcome is
//! exact and asserted with `==`. The three continuous kernels thread only
//! through `clamp`, `min`, `max`, a divide and multiply-add, so for fixtures
//! chosen clear of a threshold tie the `CPU` and `GPU` agree to a tight
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`); the small slack admits
//! a legal last-place divide difference while still catching a genuinely wrong
//! port.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `min`,
//! `max`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `sqrt`, no inverse trigonometry, no `round`, and no
//! bare f32 equality. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of arithmetic, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::wetness`；无第三方引擎源码或衍生代码。
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

/// Operation code for `wet_albedo_scale`.
const OP_WET_ALBEDO_SCALE: u32 = 0;
/// Operation code for `capillary_height`.
const OP_CAPILLARY_HEIGHT: u32 = 1;
/// Operation code for `puddle_depth`.
const OP_PUDDLE_DEPTH: u32 = 2;
/// Operation code for `is_puddle`.
const OP_IS_PUDDLE: u32 = 3;

/// The portable core-`WGSL` wetness-response kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` branches
/// on the per-query operation code and mirrors the `CPU` golden
/// [`wetness`](prism_render_architecture::water::wetness) closed forms; see the
/// module documentation for the algorithm.
const WATER_WETNESS_RESPONSE_WGSL: &str = r#"
// wgpu compute twin of prism_render_architecture::water::wetness: the four
// stateless per-point wetness-response kernels (wet_albedo_scale,
// capillary_height, puddle_depth, is_puddle), selected by an operation code,
// with only clamp, min, max and + - * /. The exponential time-stepping
// envelopes (absorb/dry/update_wetness/step_moisture) stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::wetness；无第三方
// 引擎源码或衍生代码。

// Shared non-negative epsilon, matching water::EPS. Degenerate reaches and
// thresholds collapse below this, exactly as the reference guards them.
const EPS: f32 = 1e-6;

// Operation codes, mirroring the Rust-side constants.
const OP_WET_ALBEDO_SCALE: u32 = 0u;
const OP_CAPILLARY_HEIGHT: u32 = 1u;
const OP_PUDDLE_DEPTH: u32 = 2u;
const OP_IS_PUDDLE: u32 = 3u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    // Flattened WetnessParams: capillary reach, soak / dry rates, peak albedo
    // darkening and the puddle threshold. absorb_rate / dry_rate drive only the
    // host-side time-stepping envelopes and are unused here, kept so the
    // uniform layout mirrors the full WetnessParams.
    max_capillary_height: f32,
    absorb_rate: f32,
    dry_rate: f32,
    darkening_strength: f32,
    puddle_threshold: f32,
    pad0: u32,
    pad1: u32,
}

struct Query {
    // Operation selector (see the OP_* constants).
    op: u32,
    // Generic scalar drivers, interpreted per operation:
    //   wet_albedo_scale: a = wetness
    //   capillary_height: a = wetness, b = dist_above_water
    //   puddle_depth:     a = accumulated, b = rain_rate, c = drain_rate, d = dt
    //   is_puddle:        a = depth
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Scalar output for the three continuous kernels (0 for the predicate).
    value: f32,
    // Boolean output for is_puddle, encoded as 1 / 0 (0 for the scalar kernels).
    flag: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var value: f32 = 0.0;
    var flag: u32 = 0u;

    if (q.op == OP_WET_ALBEDO_SCALE) {
        // wet_albedo_scale: scale linearly from 1 (dry) down to
        // 1 - darkening (fully wet), clamped to that band.
        let w = clamp(q.a, 0.0, 1.0);
        let darkening = clamp(params.darkening_strength, 0.0, 1.0);
        value = clamp(1.0 - darkening * w, 1.0 - darkening, 1.0);
    } else if (q.op == OP_CAPILLARY_HEIGHT) {
        // capillary_height: reach * wetness * falloff, where falloff fades
        // linearly from 1 at the waterline to 0 at the reach. A non-positive
        // reach disables the effect.
        let reach = params.max_capillary_height;
        if (reach <= EPS) {
            value = 0.0;
        } else {
            let w = clamp(q.a, 0.0, 1.0);
            let falloff = clamp(1.0 - max(q.b, 0.0) / reach, 0.0, 1.0);
            value = reach * w * falloff;
        }
    } else if (q.op == OP_PUDDLE_DEPTH) {
        // puddle_depth: accumulate rain minus drainage over dt, clamped so a
        // puddle never goes negative. Each driver takes only its non-negative
        // part.
        let base = max(q.a, 0.0);
        value = max(base + (max(q.b, 0.0) - max(q.c, 0.0)) * max(q.d, 0.0), 0.0);
    } else {
        // is_puddle: a positive depth over the threshold is a puddle; a
        // non-positive threshold admits any strictly positive depth.
        let threshold = max(params.puddle_threshold, 0.0);
        if (threshold <= EPS) {
            if (q.a > EPS) {
                flag = 1u;
            }
        } else {
            if (q.a >= threshold) {
                flag = 1u;
            }
        }
    }

    var out: Result;
    out.value = value;
    out.flag = flag;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus the flattened
/// `WetnessParams` fields and two pad words, filling a `16`-byte,
/// `std140`-aligned uniform struct matching `Params` in
/// [`WATER_WETNESS_RESPONSE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Maximum capillary rise `max_capillary_height`.
    max_capillary_height: f32,
    /// Absorption rate `absorb_rate` (host-only envelope driver; carried for
    /// layout parity).
    absorb_rate: f32,
    /// Drying rate `dry_rate` (host-only envelope driver; carried for layout
    /// parity).
    dry_rate: f32,
    /// Peak albedo darkening `darkening_strength`.
    darkening_strength: f32,
    /// Puddle threshold depth `puddle_threshold`.
    puddle_threshold: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one query: an operation code, four generic
/// scalar drivers, and three pad words to a `32`-byte stride, matching the
/// `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation selector (see the `OP_*` constants).
    op: u32,
    /// First scalar driver (operation-specific).
    a: f32,
    /// Second scalar driver (operation-specific).
    b: f32,
    /// Third scalar driver (operation-specific).
    c: f32,
    /// Fourth scalar driver (operation-specific).
    d: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: a scalar value, a boolean flag word, and two pad words to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Scalar output for the continuous kernels.
    value: f32,
    /// Boolean output for the predicate, encoded `1` / `0`.
    flag: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// Shared tuning for the wetness response, a flattened mirror of the reference
/// `WetnessParams`.
///
/// All queries in one [`GpuWaterWetnessResponse::evaluate`] dispatch share this
/// block; it rides in the uniform buffer. `absorb_rate` and `dry_rate` drive
/// only the host-side time-stepping envelopes and are unused by the four
/// twinned kernels, but are carried so the layout mirrors the full
/// `WetnessParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterWetnessResponseParams {
    /// Maximum capillary rise above the waterline `max_capillary_height`; also
    /// the reach over which the wet band fades to nothing. `<= 0` disables
    /// capillary rise.
    pub max_capillary_height: f32,
    /// Absorption rate `absorb_rate` (host-only envelope driver).
    pub absorb_rate: f32,
    /// Drying rate `dry_rate` (host-only envelope driver).
    pub dry_rate: f32,
    /// Peak fractional albedo darkening `darkening_strength` in `0..=1`.
    pub darkening_strength: f32,
    /// Puddle threshold depth `puddle_threshold` in meters.
    pub puddle_threshold: f32,
}

impl WaterWetnessResponseParams {
    /// Builds a params block from the five `WetnessParams` fields.
    #[must_use]
    pub const fn new(
        max_capillary_height: f32,
        absorb_rate: f32,
        dry_rate: f32,
        darkening_strength: f32,
        puddle_threshold: f32,
    ) -> WaterWetnessResponseParams {
        WaterWetnessResponseParams {
            max_capillary_height,
            absorb_rate,
            dry_rate,
            darkening_strength,
            puddle_threshold,
        }
    }
}

/// One wetness-response query: the operation to run plus its per-point drivers.
///
/// Mirrors the arguments the matching `CPU` golden
/// [`wetness`](prism_render_architecture::water::wetness) function reads (the
/// shared `WetnessParams` rides in [`WaterWetnessResponseParams`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WaterWetnessResponseQuery {
    /// Albedo multiplier at `wetness` (twins `wet_albedo_scale`).
    WetAlbedoScale {
        /// Surface saturation in `0..=1`.
        wetness: f32,
    },
    /// Capillary band height at `dist_above_water` for `wetness` (twins
    /// `capillary_height`).
    CapillaryHeight {
        /// Surface saturation in `0..=1`.
        wetness: f32,
        /// Height above the waterline in meters.
        dist_above_water: f32,
    },
    /// Puddle depth after one step (twins `puddle_depth`).
    PuddleDepth {
        /// Prior accumulated depth in meters.
        accumulated: f32,
        /// Rain fill rate.
        rain_rate: f32,
        /// Drainage rate.
        drain_rate: f32,
        /// Step duration in seconds.
        dt: f32,
    },
    /// Whether `depth` clears the puddle threshold (twins `is_puddle`).
    IsPuddle {
        /// Standing depth in meters.
        depth: f32,
    },
}

/// One wetness-response result, mirroring the value the matching `CPU` golden
/// returns.
///
/// The three continuous kernels return a [`WaterWetnessResponseResult::Scalar`];
/// the puddle predicate returns a [`WaterWetnessResponseResult::Puddle`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WaterWetnessResponseResult {
    /// Scalar output of `wet_albedo_scale`, `capillary_height` or
    /// `puddle_depth`.
    Scalar(f32),
    /// Boolean output of `is_puddle`.
    Puddle(bool),
}

/// Encodes one [`WaterWetnessResponseQuery`] into its `std430` [`GpuQuery`]
/// slot, packing the operation code and the per-point drivers.
fn encode_query(q: &WaterWetnessResponseQuery) -> GpuQuery {
    let mut out = GpuQuery {
        op: 0,
        a: 0.0,
        b: 0.0,
        c: 0.0,
        d: 0.0,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    match *q {
        WaterWetnessResponseQuery::WetAlbedoScale { wetness } => {
            out.op = OP_WET_ALBEDO_SCALE;
            out.a = wetness;
        }
        WaterWetnessResponseQuery::CapillaryHeight {
            wetness,
            dist_above_water,
        } => {
            out.op = OP_CAPILLARY_HEIGHT;
            out.a = wetness;
            out.b = dist_above_water;
        }
        WaterWetnessResponseQuery::PuddleDepth {
            accumulated,
            rain_rate,
            drain_rate,
            dt,
        } => {
            out.op = OP_PUDDLE_DEPTH;
            out.a = accumulated;
            out.b = rain_rate;
            out.c = drain_rate;
            out.d = dt;
        }
        WaterWetnessResponseQuery::IsPuddle { depth } => {
            out.op = OP_IS_PUDDLE;
            out.a = depth;
        }
    }
    out
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterWetnessResponseResult`], using the originating query to pick the
/// scalar or boolean shape.
fn decode_result(query: &WaterWetnessResponseQuery, raw: &GpuResult) -> WaterWetnessResponseResult {
    match *query {
        WaterWetnessResponseQuery::IsPuddle { .. } => {
            WaterWetnessResponseResult::Puddle(raw.flag != 0)
        }
        _ => WaterWetnessResponseResult::Scalar(raw.value),
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

/// A compiled, reusable wetness-response compute pipeline, twinning the four
/// stateless per-point kernels of the `CPU` golden
/// [`wetness`](prism_render_architecture::water::wetness).
pub struct GpuWaterWetnessResponse {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterWetnessResponse {
    /// Compiles the wetness-response kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterWetnessResponse {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_wetness_response"),
            source: ShaderSource::Wgsl(WATER_WETNESS_RESPONSE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_wetness_response_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_wetness_response_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_wetness_response_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterWetnessResponse {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` under the shared `params` and returns
    /// one [`WaterWetnessResponseResult`] per input, in order.
    ///
    /// Each result equals the matching `CPU` golden
    /// [`wetness`](prism_render_architecture::water::wetness) outcome: the
    /// puddle predicate matches exactly, while the continuous kernels match
    /// within the tolerance documented on this module. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        params: WaterWetnessResponseParams,
        queries: &[WaterWetnessResponseQuery],
    ) -> Vec<WaterWetnessResponseResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: count as u32,
            max_capillary_height: params.max_capillary_height,
            absorb_rate: params.absorb_rate,
            dry_rate: params.dry_rate,
            darkening_strength: params.darkening_strength,
            puddle_threshold: params.puddle_threshold,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_wetness_response_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_wetness_response_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_wetness_response_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_wetness_response_bind_group"),
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
            label: Some("prism_volumetric_water_wetness_response_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_wetness_response_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_wetness_response_pass"),
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

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, slot)| decode_result(query, slot))
            .collect()
    }
}
