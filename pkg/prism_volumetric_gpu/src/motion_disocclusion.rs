//! `wgpu` compute twin of the per-sample history-rejection heuristic inside the
//! motion disocclusion contract
//! ([`disocclusion`](prism_render_architecture::motion::disocclusion), motion
//! temporal core).
//!
//! The `CPU` golden
//! [`classify`](prism_render_architecture::motion::disocclusion::classify) takes
//! a current and a reprojected-history
//! [`SurfacePoint`](prism_render_architecture::motion::disocclusion::SurfacePoint)
//! plus the tunable
//! [`DisocclusionParams`](prism_render_architecture::motion::disocclusion::DisocclusionParams)
//! and returns a graded confidence in `[0, 1]`, a hard accept / reject verdict,
//! and a bit set of rejection reasons. It combines three independent signals —
//! surface-identity equality, depth continuity
//! ([`depth_consistency`](prism_render_architecture::motion::disocclusion::depth_consistency)),
//! and normal continuity
//! ([`normal_consistency`](prism_render_architecture::motion::disocclusion::normal_consistency)) —
//! by taking the minimum of the three sub-confidences, so a single failing
//! signal collapses the result.
//!
//! [`GpuMotionDisocclusion`] is the on-device twin of that per-sample closed
//! form: one thread classifies one `(current, history, params)` sample,
//! reproducing the reference's exact arithmetic so a passing real-device parity
//! test is direct evidence the ported kernel computes the same confidence,
//! accept flag, and reason bits the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Per sample the kernel reproduces, in order: the surface-identity equality
//! test (and its unconditional zero-confidence rejection when
//! `require_surface_match` is set), the depth-continuity grade
//! `clamp01(1 - relative / tolerance)` with a relative gap floored by a shared
//! epsilon, the normal-continuity grade `clamp01((dot3 - threshold) / (1 -
//! threshold))`, the `min` combination of the three sub-confidences, the
//! `DEPTH_DISCONTINUITY` / `NORMAL_DISCONTINUITY` / `SURFACE_MISMATCH` reason bit
//! raises, and the final `accepted = reasons.is_empty() && confidence >=
//! accept_threshold` decision. The reason bit values mirror the golden
//! `RejectionReasons` constants (`SURFACE_MISMATCH = 1 << 0`,
//! `DEPTH_DISCONTINUITY = 1 << 1`, `NORMAL_DISCONTINUITY = 1 << 2`).
//!
//! # What stays on the host
//!
//! The parameter sanitization in
//! [`DisocclusionParams::new`](prism_render_architecture::motion::disocclusion::DisocclusionParams::new) —
//! the `NaN` resolution, the `EPS` tolerance floor, and the `[-1, 1)` /
//! `[0, 1]` clamps — stays on the host; the device consumes already-sanitized
//! parameters so the kernel needs no `NaN` detection (`WGSL` cannot test
//! `is_nan` without an `f32` equality). The per-verdict motion-flag byte
//! ([`history_flags`](prism_render_architecture::motion::disocclusion::history_flags))
//! and the view-global invalidation override
//! ([`view_forces_rejection`](prism_render_architecture::motion::disocclusion::view_forces_rejection))
//! are host-side policy that frames a batch, not per-sample arithmetic, and are
//! not twinned. The `u64` `surface_id` has no `WGSL` analogue, so it is carried
//! as two `u32` halves and compared half-for-half on the device, reproducing the
//! host's integer `!=`.
//!
//! # Correctness model
//!
//! The accept flag and the reason bit mask are integers / booleans built from
//! integer equality and sign comparisons, so for fixtures chosen clear of a
//! decision boundary (a sub-confidence near zero, a combined confidence near
//! `accept_threshold`) the `CPU` and `GPU` agree exactly and the parity test
//! asserts an exact `==` on each. The continuous `confidence` threads through a
//! subtract and a divide only (no `sqrt`, no transcendental), so `CPU` and `GPU`
//! match to within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`, `abs`,
//! `+ - * /`, bit-or, and integer equality — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no `round`,
//! and no `sqrt`. There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic, so the kernel provably terminates. No optional device
//! feature is required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::motion::disocclusion`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` per-sample disocclusion kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`classify`](prism_render_architecture::motion::disocclusion::classify)
/// closed form; see the module documentation for the algorithm.
const MOTION_DISOCCLUSION_WGSL: &str = r#"
// Per-sample disocclusion twin: one thread classifies one (current, history,
// params) sample into a graded confidence, an accept flag, and a reason bit
// mask, mirroring the CPU golden `motion::disocclusion::classify` closed form
// with only min/max/abs, + - * / and integer equality. Parameter sanitization,
// the motion-flag byte, and the view-global invalidation override stay on the
// host.
//
// Provenance: 孪生自本仓 prism_render_architecture::motion::disocclusion；无第三方
// 引擎源码或衍生代码。

// Shared small epsilon matching `motion::EPS`, used to floor the relative depth
// gap denominator and the depth tolerance so no divide-by-zero occurs.
const EPS: f32 = 1.0e-6;

// Reason bit values mirroring the golden `RejectionReasons` constants.
const SURFACE_MISMATCH: u32 = 1u;      // 1 << 0
const DEPTH_DISCONTINUITY: u32 = 2u;   // 1 << 1
const NORMAL_DISCONTINUITY: u32 = 4u;  // 1 << 2

struct Params {
    // Number of samples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Current surface sample: depth, unit normal, and the two u32 halves of the
    // u64 surface id (lo then hi).
    cur_depth: f32,
    cur_nx: f32,
    cur_ny: f32,
    cur_nz: f32,
    cur_id_lo: u32,
    cur_id_hi: u32,
    // History (reprojected) surface sample, same layout.
    his_depth: f32,
    his_nx: f32,
    his_ny: f32,
    his_nz: f32,
    his_id_lo: u32,
    his_id_hi: u32,
    // Already-sanitized parameters (NaN/clamp handled host-side).
    require_surface_match: u32,
    depth_relative_tolerance: f32,
    normal_cos_threshold: f32,
    accept_threshold: f32,
}

struct Result {
    // Graded validity in [0, 1].
    confidence: f32,
    // 1 when the history should be blended in, 0 otherwise.
    accepted: u32,
    // Rejection reason bit mask (see the reason constants above).
    reasons: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Clamp to [0, 1]. Inputs are finite for sanitized parameters and finite
// fixtures, so the branch ordering matches the golden `clamp01` (which also
// resolves NaN to 0 host-side; NaN never reaches the device here).
fn clamp01(x: f32) -> f32 {
    if (x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
}

// Dot product of two 3-component vectors.
fn dot3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    return ax * bx + ay * by + az * bz;
}

// Depth-continuity confidence: 1 at a perfect match, falling linearly to 0 as
// the relative gap reaches the tolerance, and 0 beyond it.
fn depth_consistency(current_depth: f32, history_depth: f32, tolerance: f32) -> f32 {
    let tol = max(tolerance, EPS);
    let denom = max(max(abs(current_depth), abs(history_depth)), EPS);
    let relative = abs(current_depth - history_depth) / denom;
    return clamp01(1.0 - relative / tol);
}

// Normal-continuity confidence: 1 when the normals align, falling linearly to 0
// as their cosine drops to the threshold, and 0 below it.
fn normal_consistency(
    cnx: f32, cny: f32, cnz: f32,
    hnx: f32, hny: f32, hnz: f32,
    cos_threshold: f32,
) -> f32 {
    let threshold = clamp(cos_threshold, -1.0, 1.0 - EPS);
    let c = dot3(cnx, cny, cnz, hnx, hny, hnz);
    return clamp01((c - threshold) / (1.0 - threshold));
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var reasons: u32 = 0u;
    var confidence: f32 = 1.0;

    // Surface identity: a provable mismatch rejects the history outright.
    let id_differs = (q.cur_id_lo != q.his_id_lo) || (q.cur_id_hi != q.his_id_hi);
    if (q.require_surface_match == 1u && id_differs) {
        reasons = reasons | SURFACE_MISMATCH;
        confidence = 0.0;
    }

    // Depth continuity.
    let depth_conf = depth_consistency(q.cur_depth, q.his_depth, q.depth_relative_tolerance);
    if (depth_conf <= 0.0) {
        reasons = reasons | DEPTH_DISCONTINUITY;
    }
    confidence = min(confidence, depth_conf);

    // Normal continuity.
    let normal_conf = normal_consistency(
        q.cur_nx, q.cur_ny, q.cur_nz,
        q.his_nx, q.his_ny, q.his_nz,
        q.normal_cos_threshold,
    );
    if (normal_conf <= 0.0) {
        reasons = reasons | NORMAL_DISCONTINUITY;
    }
    confidence = min(confidence, normal_conf);

    var accepted: u32 = 0u;
    if (reasons == 0u && confidence >= q.accept_threshold) {
        accepted = 1u;
    }

    var out: Result;
    out.confidence = confidence;
    out.accepted = accepted;
    out.reasons = reasons;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MOTION_DISOCCLUSION_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid samples in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one classification query, matching the `WGSL`
/// `Query` struct: both surface points' depth, normal, and split `surface_id`
/// halves, followed by the sanitized parameters. All members are scalars, so the
/// struct is a dense `64`-byte block with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Current surface depth.
    cur_depth: f32,
    /// Current normal `x`.
    cur_nx: f32,
    /// Current normal `y`.
    cur_ny: f32,
    /// Current normal `z`.
    cur_nz: f32,
    /// Low `32` bits of the current `surface_id`.
    cur_id_lo: u32,
    /// High `32` bits of the current `surface_id`.
    cur_id_hi: u32,
    /// History surface depth.
    his_depth: f32,
    /// History normal `x`.
    his_nx: f32,
    /// History normal `y`.
    his_ny: f32,
    /// History normal `z`.
    his_nz: f32,
    /// Low `32` bits of the history `surface_id`.
    his_id_lo: u32,
    /// High `32` bits of the history `surface_id`.
    his_id_hi: u32,
    /// `1` when mismatched ids reject unconditionally, `0` otherwise.
    require_surface_match: u32,
    /// Sanitized relative depth tolerance.
    depth_relative_tolerance: f32,
    /// Sanitized normal cosine threshold.
    normal_cos_threshold: f32,
    /// Sanitized accept threshold.
    accept_threshold: f32,
}

/// `repr(C)` `std430` layout of one classification result, matching the `WGSL`
/// `Result` struct: the graded confidence, the accept flag, and the reason bit
/// mask, plus one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Graded validity in `[0, 1]`.
    confidence: f32,
    /// `1` when the history should be blended in, `0` otherwise.
    accepted: u32,
    /// Rejection reason bit mask.
    reasons: u32,
    /// Padding word.
    pad0: u32,
}

/// A single surface sample for the disocclusion twin, mirroring the golden
/// [`SurfacePoint`](prism_render_architecture::motion::disocclusion::SurfacePoint).
///
/// `depth` is whatever monotone depth the pipeline stores (only relative gaps
/// are compared), `normal` is expected to be unit length in a shared space, and
/// `surface_id` follows the golden convention where `0` means "no stable id".
/// The `u64` id is carried whole here and split into two `u32` halves only at
/// the device boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionSurfacePoint {
    /// Stored depth (linear or `NDC`); only relative gaps are compared.
    pub depth: f32,
    /// Unit surface normal in a shared space.
    pub normal: [f32; 3],
    /// Stable surface identity; `0` denotes "unknown / none".
    pub surface_id: u64,
}

impl MotionSurfacePoint {
    /// Builds a surface sample.
    #[must_use]
    pub const fn new(depth: f32, normal: [f32; 3], surface_id: u64) -> MotionSurfacePoint {
        MotionSurfacePoint {
            depth,
            normal,
            surface_id,
        }
    }
}

/// Tunable thresholds for the disocclusion twin, mirroring the golden
/// [`DisocclusionParams`](prism_render_architecture::motion::disocclusion::DisocclusionParams).
///
/// These are consumed as-is by the device: the host is expected to build them
/// from the golden
/// [`DisocclusionParams::new`](prism_render_architecture::motion::disocclusion::DisocclusionParams::new)
/// so the `NaN` resolution and the tolerance / threshold clamps have already
/// been applied, since the kernel performs no `NaN` detection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionDisocclusionParams {
    /// Maximum tolerated relative depth difference (sanitized, floored at the
    /// shared epsilon).
    pub depth_relative_tolerance: f32,
    /// Minimum `cos(angle)` between normals (sanitized, clamped to `[-1, 1)`).
    pub normal_cos_threshold: f32,
    /// When set, mismatched surface ids reject the history unconditionally.
    pub require_surface_match: bool,
    /// Combined confidence at or above which the history is accepted (sanitized,
    /// clamped to `[0, 1]`).
    pub accept_threshold: f32,
}

impl MotionDisocclusionParams {
    /// Builds a parameter set. Callers should pass values already sanitized by
    /// the golden
    /// [`DisocclusionParams::new`](prism_render_architecture::motion::disocclusion::DisocclusionParams::new).
    #[must_use]
    pub const fn new(
        depth_relative_tolerance: f32,
        normal_cos_threshold: f32,
        require_surface_match: bool,
        accept_threshold: f32,
    ) -> MotionDisocclusionParams {
        MotionDisocclusionParams {
            depth_relative_tolerance,
            normal_cos_threshold,
            require_surface_match,
            accept_threshold,
        }
    }
}

/// One per-sample classification query: the current and reprojected-history
/// [`MotionSurfacePoint`] plus the [`MotionDisocclusionParams`] to apply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionDisocclusionQuery {
    /// The current-frame surface sample.
    pub current: MotionSurfacePoint,
    /// The reprojected history surface sample.
    pub history: MotionSurfacePoint,
    /// The thresholds to classify against.
    pub params: MotionDisocclusionParams,
}

impl MotionDisocclusionQuery {
    /// Builds a classification query.
    #[must_use]
    pub const fn new(
        current: MotionSurfacePoint,
        history: MotionSurfacePoint,
        params: MotionDisocclusionParams,
    ) -> MotionDisocclusionQuery {
        MotionDisocclusionQuery {
            current,
            history,
            params,
        }
    }
}

/// One resolved disocclusion verdict, mirroring the golden
/// [`DisocclusionVerdict`](prism_render_architecture::motion::disocclusion::DisocclusionVerdict).
///
/// `confidence` is the `min` of the surface, depth, and normal sub-confidences;
/// `accepted` is `true` when no reason bit fired and the confidence reached the
/// accept threshold; `reasons` is the raw bit mask (`1 << 0` surface mismatch,
/// `1 << 1` depth discontinuity, `1 << 2` normal discontinuity).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionDisocclusionResult {
    /// Graded validity in `[0, 1]`.
    pub confidence: f32,
    /// `true` when the reprojected history should be blended in.
    pub accepted: bool,
    /// Rejection reason bit mask.
    pub reasons: u32,
}

/// Encodes one [`MotionDisocclusionQuery`] into its `std430` [`GpuQuery`] slot,
/// splitting each `u64` `surface_id` into its low and high `u32` halves.
fn encode_query(q: &MotionDisocclusionQuery) -> GpuQuery {
    let cur = q.current;
    let his = q.history;
    let p = q.params;
    GpuQuery {
        cur_depth: cur.depth,
        cur_nx: cur.normal[0],
        cur_ny: cur.normal[1],
        cur_nz: cur.normal[2],
        cur_id_lo: (cur.surface_id & 0xffff_ffff) as u32,
        cur_id_hi: (cur.surface_id >> 32) as u32,
        his_depth: his.depth,
        his_nx: his.normal[0],
        his_ny: his.normal[1],
        his_nz: his.normal[2],
        his_id_lo: (his.surface_id & 0xffff_ffff) as u32,
        his_id_hi: (his.surface_id >> 32) as u32,
        require_surface_match: u32::from(p.require_surface_match),
        depth_relative_tolerance: p.depth_relative_tolerance,
        normal_cos_threshold: p.normal_cos_threshold,
        accept_threshold: p.accept_threshold,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MotionDisocclusionResult`],
/// turning the `accepted` word back into a [`bool`].
fn decode_result(raw: &GpuResult) -> MotionDisocclusionResult {
    MotionDisocclusionResult {
        confidence: raw.confidence,
        accepted: raw.accepted != 0,
        reasons: raw.reasons,
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

/// A compiled, reusable per-sample disocclusion compute pipeline, twinning the
/// numeric core of the `CPU` golden
/// [`classify`](prism_render_architecture::motion::disocclusion::classify).
pub struct GpuMotionDisocclusion {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMotionDisocclusion {
    /// Compiles the per-sample disocclusion kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMotionDisocclusion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_motion_disocclusion"),
            source: ShaderSource::Wgsl(MOTION_DISOCCLUSION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_motion_disocclusion_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_motion_disocclusion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_motion_disocclusion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMotionDisocclusion {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies every sample in `queries` and returns one
    /// [`MotionDisocclusionResult`] per input, in order.
    ///
    /// The `accepted` flag and the `reasons` bit mask equal the reference
    /// exactly for samples clear of a decision boundary; the `confidence`
    /// matches to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MotionDisocclusionQuery],
    ) -> Vec<MotionDisocclusionResult> {
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
            label: Some("prism_volumetric_motion_disocclusion_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_disocclusion_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_disocclusion_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_motion_disocclusion_bind_group"),
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
            label: Some("prism_volumetric_motion_disocclusion_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_motion_disocclusion_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_motion_disocclusion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
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
