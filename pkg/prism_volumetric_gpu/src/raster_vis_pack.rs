//! `wgpu` compute twin of the vis-buffer packing primitives inside the meshlet
//! software rasterizer contract
//! ([`software_raster`](prism_render_architecture::virtual_geometry::software_raster)).
//!
//! The `CPU` golden exposes four `const` entry points that together define how a
//! depth and a payload fold into one vis-buffer word and back:
//! [`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth)
//! clamps a reversed-Z depth to `[0, 1]` and takes its raw `IEEE` bit pattern as
//! a compositing key,
//! [`pack_vis`](prism_render_architecture::virtual_geometry::software_raster::pack_vis)
//! packs that key into the high `32` bits of a `u64` and the payload into the
//! low `32`, and
//! [`vis_depth`](prism_render_architecture::virtual_geometry::software_raster::vis_depth)
//! /
//! [`vis_payload`](prism_render_architecture::virtual_geometry::software_raster::vis_payload)
//! extract the two fields back out.
//!
//! [`GpuRasterVisPack`] is the on-device twin of that packing core: one thread
//! turns one `(depth, payload)` pair into the two `32`-bit halves of the packed
//! word — `hi = encode_depth(depth)` and `lo = payload` — so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same depth key and field layout the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For a `(depth, payload)` pair the reference computes
//! `depth_key = encode_depth(depth) = clamp(depth, 0, 1).to_bits()` and
//! `packed = pack_vis(depth_key, payload) = ((depth_key as u64) << 32) | payload`.
//! The twin reproduces the two halves of that word directly:
//! `hi = bitcast<u32>(clamp(depth, 0, 1))` (the depth key, equal to
//! [`vis_depth`](prism_render_architecture::virtual_geometry::software_raster::vis_depth)`(packed)`)
//! and `lo = payload` (equal to
//! [`vis_payload`](prism_render_architecture::virtual_geometry::software_raster::vis_payload)`(packed)`).
//! Because `WGSL` core has no `u64`, the `u64` word is never materialized on the
//! device; it is represented throughout as the `(hi, lo)` pair the host also
//! splits the reference word into.
//!
//! # What stays on the host
//!
//! The reference packs into a genuine `u64` and composites whole vis-buffers of
//! those words; that `u64` storage and the per-pixel maximum composite are
//! variable-length container work owned by the host. The device twin only
//! produces the two field halves of one word per query, which the host reads
//! back and compares against the split of the reference's `u64`.
//!
//! # Correctness model
//!
//! Every output is a discrete `32`-bit value: `hi` is the raw bit pattern of a
//! clamped depth (a `bitcast`, not an arithmetic result) and `lo` is the payload
//! copied verbatim. The clamp is an exact magnitude selection with no rounding,
//! so the `CPU` and `GPU` agree bit-for-bit and the parity test asserts an exact
//! integer `==` on both halves — no floating-point tolerance is involved.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::virtual_geometry::software_raster`；无第三方引擎源码或衍生代码。

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

/// The portable core-`WGSL` vis-buffer packing kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden packing core
/// ([`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth)
/// /
/// [`pack_vis`](prism_render_architecture::virtual_geometry::software_raster::pack_vis));
/// see the module documentation for the algorithm.
const RASTER_VIS_PACK_WGSL: &str = r#"
// Vis-buffer packing twin: one thread turns one (depth, payload) pair into the
// two 32-bit halves of the packed word, mirroring the CPU golden
// `virtual_geometry::software_raster` packing core with only clamp and bitcast.
// WGSL core has no u64, so the packed word is represented as the (hi, lo) pair
// hi = depth_key, lo = payload rather than a synthesized 64-bit value.
//
// Provenance: 孪生自本仓 prism_render_architecture::virtual_geometry::software_raster；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The reversed-Z depth to encode; clamped to [0, 1] before the bitcast.
    depth: f32,
    // The payload copied verbatim into the low half of the word.
    payload: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // High half: depth_key = bitcast<u32>(clamp(depth, 0, 1)).
    hi: u32,
    // Low half: the payload echoed back unchanged.
    lo: u32,
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

    // encode_depth: clamp the reversed-Z depth to [0, 1] (a stray negative or
    // super-unit depth cannot masquerade as nearest) and take the raw IEEE bit
    // pattern as the compositing key.
    let depth_key = bitcast<u32>(clamp(q.depth, 0.0, 1.0));

    var out: Result;
    // The packed u64 is ((depth_key as u64) << 32) | payload; its two 32-bit
    // halves are the depth key (high) and the payload (low).
    out.hi = depth_key;
    out.lo = q.payload;
    out.pad0 = 0u;
    out.pad1 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RASTER_VIS_PACK_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pairs in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one packing query: a depth and a payload plus
/// two pad words to a `16`-byte stride, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// The reversed-Z depth to encode.
    depth: f32,
    /// The payload copied verbatim into the low half.
    payload: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one packing result, matching the `WGSL`
/// `Result` struct: the two `32`-bit word halves plus two pad words to a
/// `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// High half: the depth key `bitcast<u32>(clamp(depth, 0, 1))`.
    hi: u32,
    /// Low half: the payload echoed back unchanged.
    lo: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One packing query for the vis-buffer twin: a reversed-Z `depth` and the
/// `payload` to fold into one vis-buffer word.
///
/// The host owns the surrounding `u64` vis-buffer and its per-pixel maximum
/// composite; it enqueues one [`RasterVisPackQuery`] per `(depth, payload)` pair
/// it needs the device to pack, matching the reference packing core
/// ([`encode_depth`](prism_render_architecture::virtual_geometry::software_raster::encode_depth)
/// /
/// [`pack_vis`](prism_render_architecture::virtual_geometry::software_raster::pack_vis)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RasterVisPackQuery {
    /// The reversed-Z depth to encode; clamped to `[0, 1]` before the bitcast.
    pub depth: f32,
    /// The payload copied verbatim into the low half of the packed word.
    pub payload: u32,
}

impl RasterVisPackQuery {
    /// Builds a query packing `payload` with `depth`.
    #[must_use]
    pub const fn new(depth: f32, payload: u32) -> RasterVisPackQuery {
        RasterVisPackQuery { depth, payload }
    }
}

/// One packed vis-buffer word, split into its two `32`-bit halves, mirroring the
/// reference `u64` word
/// [`pack_vis`](prism_render_architecture::virtual_geometry::software_raster::pack_vis)`(encode_depth(depth), payload)`.
///
/// `hi` is the depth key — the high `32` bits of the word, equal to
/// [`vis_depth`](prism_render_architecture::virtual_geometry::software_raster::vis_depth)
/// of the reference word — and `lo` is the payload, its low `32` bits, equal to
/// [`vis_payload`](prism_render_architecture::virtual_geometry::software_raster::vis_payload).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RasterVisPackResult {
    /// High half: the depth key `encode_depth(depth)`.
    pub hi: u32,
    /// Low half: the payload echoed back unchanged.
    pub lo: u32,
}

/// Encodes one [`RasterVisPackQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &RasterVisPackQuery) -> GpuQuery {
    GpuQuery {
        depth: q.depth,
        payload: q.payload,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`RasterVisPackResult`].
fn decode_result(raw: &GpuResult) -> RasterVisPackResult {
    RasterVisPackResult {
        hi: raw.hi,
        lo: raw.lo,
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

/// A compiled, reusable vis-buffer packing compute pipeline, twinning the
/// packing core of the `CPU` golden
/// [`software_raster`](prism_render_architecture::virtual_geometry::software_raster).
pub struct GpuRasterVisPack {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRasterVisPack {
    /// Compiles the vis-buffer packing kernel on `ctx`.
    ///
    /// The kernel uses only `clamp` and `bitcast`, so it needs no optional
    /// device feature and runs on any core-`WGSL` adapter.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRasterVisPack {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_raster_vis_pack_module"),
            source: ShaderSource::Wgsl(RASTER_VIS_PACK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_raster_vis_pack_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_raster_vis_pack_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_raster_vis_pack_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRasterVisPack {
            module,
            layout,
            pipeline,
        }
    }

    /// Packs every `(depth, payload)` pair in `queries` and returns one
    /// [`RasterVisPackResult`] per input, in order.
    ///
    /// Both halves equal the reference exactly: `hi` is the bitcast of the
    /// clamped depth and `lo` is the payload copied verbatim, so the parity test
    /// asserts an exact integer `==` on each. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RasterVisPackQuery],
    ) -> Vec<RasterVisPackResult> {
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
            label: Some("prism_volumetric_raster_vis_pack_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_raster_vis_pack_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_raster_vis_pack_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_raster_vis_pack_bind_group"),
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
            label: Some("prism_volumetric_raster_vis_pack_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_raster_vis_pack_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_raster_vis_pack_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pair, flattened to a 1-D dispatch.
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
