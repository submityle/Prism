//! `wgpu` compute twin of the per-surface transparency routing decision inside
//! the transparency contract
//! ([`routing`](prism_render_architecture::transparency::routing)).
//!
//! The `CPU` golden
//! [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path)
//! turns a surface's content class plus its blending needs and the backend's
//! capabilities into a concrete
//! [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath),
//! and
//! [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for)
//! turns that path into the three shared-target writes the frame graph wires up
//! (the `TAA` reactive mask, motion vectors, and the ray-tracing scene). Both
//! are pure, closed-form decisions over a surface's discriminant class, a layer
//! count, and three booleans — there is no variable-length aggregate and no
//! floating-point arithmetic — so the whole decision ports to the device as an
//! integer-and-boolean kernel.
//!
//! [`GpuTransparencyPathSelect`] is the on-device twin of that pair: one thread
//! resolves one surface, reproducing the golden `match` to a path discriminant
//! and then reproducing [`outputs_for`] to a packed bit mask. A passing
//! real-device parity test is direct evidence the ported kernel takes the same
//! routing branch and emits the same shared-target writes the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one surface the kernel reproduces:
//! - the path selection `match` on the surface's content class
//!   ([`TransparentKind`](prism_render_architecture::transparency::routing::TransparentKind)),
//!   its layer count, its `order_independent` and `high_fidelity` flags, and the
//!   backend's `moment_oit` capability, yielding a `path_code` equal to the
//!   golden
//!   [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath)
//!   `as u32`;
//! - the [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for)
//!   decision for that path, packed into an `outputs_bits` mask whose bit `0` is
//!   `writes_reactive_mask`, bit `1` is `writes_motion`, and bit `2` is
//!   `contributes_to_ray_scene`.
//!
//! # Discriminant alignment
//!
//! This module does not re-export the golden enums; it defines its own
//! `u32` discriminant constants and keeps them aligned with the golden
//! declaration order so a `path_code` can be compared to the golden
//! `TransparencyPath as u32`. The content-class codes
//! ([`KIND_GENERAL`](TransparencyPathSelectQuery::KIND_GENERAL) and friends)
//! mirror
//! [`TransparentKind`](prism_render_architecture::transparency::routing::TransparentKind)
//! `as u32`, and the path codes
//! ([`PATH_SORTED`](TransparencyPathSelectResult::PATH_SORTED) and friends)
//! mirror
//! [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath)
//! `as u32`.
//!
//! # What stays on the host
//!
//! The draw-bucketing aggregate — the `TransparencyBins` fan-out of a frame's
//! transparent draws into one `Vec` per path and its `bucket`/`total`/`is_empty`
//! queries — is variable-length container work the host owns; the device never
//! sees it. The host enqueues one [`TransparencyPathSelectQuery`] per surface,
//! so a storage buffer is never zero-sized; an empty batch short-circuits on the
//! host.
//!
//! # Correctness model
//!
//! Every quantity this kernel produces — the `path_code` and the `outputs_bits`
//! — is a discriminant or a bit mask built from integer comparisons and boolean
//! logic, with no floating-point arithmetic anywhere. The `CPU` and `GPU`
//! therefore agree bit-for-bit and the parity test asserts an exact `==` on each
//! output; there is no tolerance and no degenerate numeric region.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `u32` comparisons,
//! boolean `&&`/`||`, and bitwise `|` — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry, no `round`, `ceil` or `floor`, and no
//! `64`-bit integers. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of comparisons, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::transparency::routing`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` per-surface transparency routing kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path)
/// and
/// [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for);
/// see the module documentation for the algorithm.
const TRANSPARENCY_PATH_SELECT_WGSL: &str = r#"
// Per-surface transparency routing twin: one thread turns one surface's content
// class, layer count and blending flags plus the backend's moment-OIT
// capability into a path discriminant and the packed shared-target writes that
// path performs, mirroring the CPU golden `transparency::routing` match with
// only u32 comparisons and boolean logic. It owns no draw bucketing; that
// variable-length aggregate stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::transparency::routing；无第三方
// 引擎源码或衍生代码。

// Content-class discriminants, aligned with the golden `TransparentKind as u32`.
const KIND_GENERAL: u32 = 0u;
const KIND_WATER: u32 = 1u;
const KIND_HAIR: u32 = 2u;
const KIND_GLASS: u32 = 3u;
const KIND_VOLUME: u32 = 4u;

// Path discriminants, aligned with the golden `TransparencyPath as u32`.
const PATH_SORTED: u32 = 0u;
const PATH_WEIGHTED_OIT: u32 = 1u;
const PATH_MOMENT_OIT: u32 = 2u;
const PATH_LAYERED_GLASS: u32 = 3u;
const PATH_SINGLE_LAYER_WATER: u32 = 4u;
const PATH_HAIR_VISIBILITY: u32 = 5u;
const PATH_VOLUMETRIC: u32 = 6u;

// Shared-target write bits, packed into `outputs_bits`.
const OUT_REACTIVE_MASK: u32 = 1u;
const OUT_MOTION: u32 = 2u;
const OUT_RAY_SCENE: u32 = 4u;

struct Params {
    // Number of surfaces in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Content-class discriminant (0..=4, matching the golden TransparentKind).
    kind_code: u32,
    // Number of stacked refractive layers (only meaningful for glass).
    layer_count: u32,
    // 1 when the surface fragments overlap unpredictably (order-independent).
    order_independent: u32,
    // 1 when the surface wants moment-based OIT over the weighted approximation.
    high_fidelity: u32,
    // 1 when the backend can run the moment-based OIT resolve.
    moment_oit: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // Resolved path discriminant (matching the golden TransparencyPath as u32).
    path_code: u32,
    // Packed shared-target writes: bit0 reactive, bit1 motion, bit2 ray scene.
    outputs_bits: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Reproduces `select_transparency_path`: water, hair and volumes take their
// dedicated paths; glass sorts when single-layer and uses the layered resolve
// otherwise; a general surface sorts when order-dependent, else takes moment
// OIT when it asks for fidelity and the backend supports it, else weighted OIT.
fn select_path(q: Query) -> u32 {
    if (q.kind_code == KIND_WATER) {
        return PATH_SINGLE_LAYER_WATER;
    }
    if (q.kind_code == KIND_HAIR) {
        return PATH_HAIR_VISIBILITY;
    }
    if (q.kind_code == KIND_VOLUME) {
        return PATH_VOLUMETRIC;
    }
    if (q.kind_code == KIND_GLASS) {
        if (q.layer_count > 1u) {
            return PATH_LAYERED_GLASS;
        }
        return PATH_SORTED;
    }
    // General surface.
    if (q.order_independent != 0u) {
        if (q.high_fidelity != 0u && q.moment_oit != 0u) {
            return PATH_MOMENT_OIT;
        }
        return PATH_WEIGHTED_OIT;
    }
    return PATH_SORTED;
}

// Reproduces `outputs_for`: every path writes the reactive mask; order-dependent
// and refractive paths emit motion vectors; refractive/reflective and marched
// media feed the ray scene.
fn outputs_for(path: u32) -> u32 {
    var bits: u32 = OUT_REACTIVE_MASK;
    if (path == PATH_SORTED || path == PATH_HAIR_VISIBILITY
        || path == PATH_LAYERED_GLASS || path == PATH_SINGLE_LAYER_WATER) {
        bits = bits | OUT_MOTION;
    }
    if (path == PATH_LAYERED_GLASS || path == PATH_SINGLE_LAYER_WATER
        || path == PATH_VOLUMETRIC) {
        bits = bits | OUT_RAY_SCENE;
    }
    return bits;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let path = select_path(q);

    var out: Result;
    out.path_code = path;
    out.outputs_bits = outputs_for(path);
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the surface count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`TRANSPARENCY_PATH_SELECT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid surfaces in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one surface query: a content-class code, a layer
/// count, and three boolean flags encoded as `u32`, plus three pad words to a
/// `32`-byte stride, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Content-class discriminant (`0..=4`).
    kind_code: u32,
    /// Number of stacked refractive layers.
    layer_count: u32,
    /// `1` when the surface is order-independent, `0` otherwise.
    order_independent: u32,
    /// `1` when the surface wants high-fidelity moment `OIT`, `0` otherwise.
    high_fidelity: u32,
    /// `1` when the backend supports moment `OIT`, `0` otherwise.
    moment_oit: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one surface result, matching the `WGSL` `Result`
/// struct: a path discriminant and a packed output mask plus two pad words to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Resolved path discriminant.
    path_code: u32,
    /// Packed shared-target writes.
    outputs_bits: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One per-surface routing query for the transparency path-selection twin: the
/// content-class discriminant, the refractive layer count, and the three
/// booleans that steer selection.
///
/// The host owns the surrounding draw-bucketing aggregate and enqueues one
/// [`TransparencyPathSelectQuery`] per surface, mirroring the inputs of the
/// reference
/// [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path).
/// The `kind_code` uses the `KIND_*` discriminants on this type, which are kept
/// aligned with the golden
/// [`TransparentKind`](prism_render_architecture::transparency::routing::TransparentKind)
/// `as u32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransparencyPathSelectQuery {
    /// Content-class discriminant (`0..=4`); see the `KIND_*` constants.
    pub kind_code: u32,
    /// Number of stacked refractive layers (only meaningful for glass).
    pub layer_count: u32,
    /// `true` when the surface fragments overlap unpredictably.
    pub order_independent: bool,
    /// `true` when the surface wants moment-based `OIT`.
    pub high_fidelity: bool,
    /// `true` when the backend can run the moment-based `OIT` resolve.
    pub moment_oit: bool,
}

impl TransparencyPathSelectQuery {
    /// Content-class code for a plain blended surface, mirroring the golden
    /// `TransparentKind::General as u32`.
    pub const KIND_GENERAL: u32 = 0;
    /// Content-class code for a water body, mirroring the golden
    /// `TransparentKind::Water as u32`.
    pub const KIND_WATER: u32 = 1;
    /// Content-class code for hair or fur, mirroring the golden
    /// `TransparentKind::Hair as u32`.
    pub const KIND_HAIR: u32 = 2;
    /// Content-class code for refractive glass, mirroring the golden
    /// `TransparentKind::Glass as u32`.
    pub const KIND_GLASS: u32 = 3;
    /// Content-class code for participating media, mirroring the golden
    /// `TransparentKind::Volume as u32`.
    pub const KIND_VOLUME: u32 = 4;

    /// Builds a query for a surface of class `kind_code` with the given layer
    /// count and blending flags.
    #[must_use]
    pub const fn new(
        kind_code: u32,
        layer_count: u32,
        order_independent: bool,
        high_fidelity: bool,
        moment_oit: bool,
    ) -> TransparencyPathSelectQuery {
        TransparencyPathSelectQuery {
            kind_code,
            layer_count,
            order_independent,
            high_fidelity,
            moment_oit,
        }
    }
}

/// One resolved surface routing decision, mirroring the reference
/// [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path)
/// and
/// [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for).
///
/// `path_code` equals the golden
/// [`TransparencyPath`](prism_render_architecture::transparency::TransparencyPath)
/// `as u32` (see the `PATH_*` constants), and `outputs_bits` packs the three
/// shared-target writes with bit `0` `writes_reactive_mask`, bit `1`
/// `writes_motion`, and bit `2` `contributes_to_ray_scene` (see the `OUT_*`
/// constants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransparencyPathSelectResult {
    /// Resolved path discriminant, equal to the golden `TransparencyPath as u32`.
    pub path_code: u32,
    /// Packed shared-target writes; see the `OUT_*` bit constants.
    pub outputs_bits: u32,
}

impl TransparencyPathSelectResult {
    /// Path code for back-to-front alpha blending, mirroring the golden
    /// `TransparencyPath::Sorted as u32`.
    pub const PATH_SORTED: u32 = 0;
    /// Path code for weighted-blended `OIT`, mirroring the golden
    /// `TransparencyPath::WeightedOit as u32`.
    pub const PATH_WEIGHTED_OIT: u32 = 1;
    /// Path code for moment-based `OIT`, mirroring the golden
    /// `TransparencyPath::MomentOit as u32`.
    pub const PATH_MOMENT_OIT: u32 = 2;
    /// Path code for the multi-layer glass resolve, mirroring the golden
    /// `TransparencyPath::LayeredGlass as u32`.
    pub const PATH_LAYERED_GLASS: u32 = 3;
    /// Path code for the single-layer water resolve, mirroring the golden
    /// `TransparencyPath::SingleLayerWater as u32`.
    pub const PATH_SINGLE_LAYER_WATER: u32 = 4;
    /// Path code for the hair/fur visibility resolve, mirroring the golden
    /// `TransparencyPath::HairVisibility as u32`.
    pub const PATH_HAIR_VISIBILITY: u32 = 5;
    /// Path code for the participating-media volume march, mirroring the golden
    /// `TransparencyPath::Volumetric as u32`.
    pub const PATH_VOLUMETRIC: u32 = 6;

    /// Output bit for `writes_reactive_mask`.
    pub const OUT_REACTIVE_MASK: u32 = 1;
    /// Output bit for `writes_motion`.
    pub const OUT_MOTION: u32 = 2;
    /// Output bit for `contributes_to_ray_scene`.
    pub const OUT_RAY_SCENE: u32 = 4;

    /// Returns `true` when the resolved path writes the `TAA` reactive mask.
    #[must_use]
    pub const fn writes_reactive_mask(&self) -> bool {
        self.outputs_bits & Self::OUT_REACTIVE_MASK != 0
    }

    /// Returns `true` when the resolved path emits motion vectors.
    #[must_use]
    pub const fn writes_motion(&self) -> bool {
        self.outputs_bits & Self::OUT_MOTION != 0
    }

    /// Returns `true` when the resolved path contributes to the ray scene.
    #[must_use]
    pub const fn contributes_to_ray_scene(&self) -> bool {
        self.outputs_bits & Self::OUT_RAY_SCENE != 0
    }
}

/// Encodes one [`TransparencyPathSelectQuery`] into its `std430` [`GpuQuery`]
/// slot, turning the booleans into `0`/`1` words.
fn encode_query(q: &TransparencyPathSelectQuery) -> GpuQuery {
    GpuQuery {
        kind_code: q.kind_code,
        layer_count: q.layer_count,
        order_independent: u32::from(q.order_independent),
        high_fidelity: u32::from(q.high_fidelity),
        moment_oit: u32::from(q.moment_oit),
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`TransparencyPathSelectResult`].
fn decode_result(raw: &GpuResult) -> TransparencyPathSelectResult {
    TransparencyPathSelectResult {
        path_code: raw.path_code,
        outputs_bits: raw.outputs_bits,
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

/// A compiled, reusable per-surface transparency routing compute pipeline,
/// twinning the `CPU` golden
/// [`select_transparency_path`](prism_render_architecture::transparency::routing::select_transparency_path)
/// and
/// [`outputs_for`](prism_render_architecture::transparency::routing::outputs_for).
pub struct GpuTransparencyPathSelect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTransparencyPathSelect {
    /// Compiles the transparency routing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTransparencyPathSelect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_transparency_path_select"),
            source: ShaderSource::Wgsl(TRANSPARENCY_PATH_SELECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_transparency_path_select_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_transparency_path_select_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_transparency_path_select_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTransparencyPathSelect {
            module,
            layout,
            pipeline,
        }
    }

    /// Routes every surface in `queries` and returns one
    /// [`TransparencyPathSelectResult`] per input, in order.
    ///
    /// The `path_code` and `outputs_bits` equal the reference exactly, since
    /// both are integer discriminants and bit masks with no floating-point
    /// arithmetic. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[TransparencyPathSelectQuery],
    ) -> Vec<TransparencyPathSelectResult> {
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
            label: Some("prism_volumetric_transparency_path_select_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_transparency_path_select_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_transparency_path_select_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_transparency_path_select_bind_group"),
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
            label: Some("prism_volumetric_transparency_path_select_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_transparency_path_select_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_transparency_path_select_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per surface, flattened to a 1-D dispatch.
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
