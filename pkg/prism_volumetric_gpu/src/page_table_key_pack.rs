//! `wgpu` compute twin of the stateless texture-streaming page-key packing pair
//! [`GpuPageTable::compare_words`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::compare_words)
//! and
//! [`GpuPageTable::unpack_key`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::unpack_key).
//!
//! The indirection page table serialises each resident
//! [`TexturePageKey`](prism_render_architecture::texture_streaming::TexturePageKey)
//! into three `u32` *compare words* laid out so an unsigned lexicographic
//! compare of `(w0, w1, w2)` equals the derived [`Ord`] on the key: `w0` is the
//! texture id, `w1` packs `(mip << 24) | (layer << 8)` with the low eight bits
//! reserved, and `w2` packs `(x << 16) | y`. The inverse reconstructs the key by
//! masking and shifting the same words. Both maps are pure integer bit work with
//! no floating-point and no transcendental, so the port is bit-exact.
//!
//! [`GpuPageTableKeyPack`] is the on-device twin of that pack/unpack pair. One
//! thread solves one query, packing the key fields into the three compare words
//! and independently unpacking a second word triple back into key fields, so a
//! passing real-device parity test is direct evidence the ported kernel computes
//! the same words and the same reconstructed key the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces both directions exactly. Packing forms `w0 = texture`,
//! `w1 = (mip << 24) | (layer << 8)` and `w2 = (x << 16) | y` from the key
//! fields. Unpacking forms `texture = w0`, `mip = w1 >> 24`,
//! `layer = (w1 >> 8) & 0xffff`, `x = w2 >> 16` and `y = w2 & 0xffff` from a
//! supplied word triple, matching the reference masks and shifts. The reference
//! stores `mip` as a `u8` and `layer`, `x`, `y` as `u16`; the kernel widens
//! every field to `u32` on the host and reproduces the shifts and masks, which
//! are identical for the `u8` / `u16` ranges the host supplies, so the host
//! narrows the unpacked `u32` fields back to their reference widths without loss.
//!
//! # What stays on the host
//!
//! The surrounding [`GpuPageTable`](prism_render_architecture::texture_streaming::indirection::GpuPageTable)
//! owns a variable-length, key-sorted `u32` word buffer and resolves a key with
//! a `usize`-indexed binary search over that buffer; the variable-length
//! aggregate, the `usize` indexing and the ordered scan are stateful host
//! infrastructure and are never dispatched. Only the two stateless per-key maps —
//! the pack and the unpack — are twinned.
//!
//! # Correctness model
//!
//! Both maps are pure `u32` bit work built from shifts, masks and bitwise or, so
//! `CPU` and `GPU` agree bit-for-bit and the parity test asserts an exact `==`
//! on the three packed words and on every unpacked field.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned shift, mask
//! and bitwise or — with no `sin`, `cos`, `exp`, `log`, `pow`, no inverse
//! trigonometry, no `sqrt` and no `u64`. Every shift amount is a compile-time
//! constant below `32`. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of integer arithmetic, so the kernel
//! provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::texture_streaming::indirection::GpuPageTable`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` page-key pack/unpack kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`compare_words`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::compare_words)
/// and
/// [`unpack_key`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::unpack_key);
/// see the module documentation for the layout.
const PAGE_TABLE_KEY_PACK_WGSL: &str = r#"
// Texture-streaming page-key pack/unpack twin: one thread packs a page key into
// its three compare words and independently unpacks a second word triple back
// into key fields, mirroring the CPU golden `indirection::GpuPageTable`
// compare_words / unpack_key with only unsigned shift, mask and bitwise or. It
// owns no page-table buffer or binary search; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::texture_streaming::indirection
// ::GpuPageTable；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Key fields to pack into compare words (widened to u32 on the host).
    texture: u32,
    mip: u32,
    layer: u32,
    x: u32,
    y: u32,
    // Compare words to unpack back into key fields.
    word0: u32,
    word1: u32,
    word2: u32,
}

struct Result {
    // Compare words packed from the key fields above.
    word0: u32,
    word1: u32,
    word2: u32,
    // Key fields unpacked from the supplied word triple.
    texture: u32,
    mip: u32,
    layer: u32,
    x: u32,
    y: u32,
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

    var out: Result;

    // Pack: w0 = texture, w1 = (mip << 24) | (layer << 8), w2 = (x << 16) | y.
    // The low eight bits of w1 are reserved and left zero, matching the golden.
    out.word0 = q.texture;
    out.word1 = (q.mip << 24u) | (q.layer << 8u);
    out.word2 = (q.x << 16u) | q.y;

    // Unpack: reconstruct the key from the supplied word triple with the same
    // masks and shifts the golden uses; the reserved low eight bits of word1 are
    // ignored.
    out.texture = q.word0;
    out.mip = q.word1 >> 24u;
    out.layer = (q.word1 >> 8u) & 0xffffu;
    out.x = q.word2 >> 16u;
    out.y = q.word2 & 0xffffu;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`PAGE_TABLE_KEY_PACK_WGSL`].
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

/// `repr(C)` `std430` layout of one pack/unpack query, matching the `WGSL`
/// `Query` struct: the five key fields to pack and the three compare words to
/// unpack, a natural `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Texture id field of the key to pack.
    texture: u32,
    /// Mip level field of the key to pack (`0` = finest).
    mip: u32,
    /// Array-layer or cube-face field of the key to pack.
    layer: u32,
    /// Page-grid X field of the key to pack.
    x: u32,
    /// Page-grid Y field of the key to pack.
    y: u32,
    /// First compare word to unpack.
    word0: u32,
    /// Second compare word to unpack.
    word1: u32,
    /// Third compare word to unpack.
    word2: u32,
}

/// `repr(C)` `std430` layout of one pack/unpack result, matching the `WGSL`
/// `Result` struct: the three packed compare words and the five unpacked key
/// fields, a natural `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// First compare word packed from the key.
    word0: u32,
    /// Second compare word packed from the key.
    word1: u32,
    /// Third compare word packed from the key.
    word2: u32,
    /// Texture id unpacked from the supplied word triple.
    texture: u32,
    /// Mip level unpacked from the supplied word triple.
    mip: u32,
    /// Array-layer or cube-face unpacked from the supplied word triple.
    layer: u32,
    /// Page-grid X unpacked from the supplied word triple.
    x: u32,
    /// Page-grid Y unpacked from the supplied word triple.
    y: u32,
}

/// One pack/unpack query: the five key fields to pack into compare words and a
/// three-word triple to unpack back into key fields, mirroring the inputs the
/// reference
/// [`compare_words`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::compare_words)
/// and
/// [`unpack_key`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::unpack_key)
/// read.
///
/// The host widens the reference `u8` / `u16` key fields to `u32` here and owns
/// the surrounding page-table buffer and binary search.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PageTableKeyPackQuery {
    /// Texture id field of the key to pack (the golden `texture`).
    pub texture: u32,
    /// Mip level field of the key to pack (the golden `mip`, `0` = finest).
    pub mip: u32,
    /// Array-layer or cube-face field of the key to pack (the golden `layer`).
    pub layer: u32,
    /// Page-grid X field of the key to pack (the golden `x`).
    pub x: u32,
    /// Page-grid Y field of the key to pack (the golden `y`).
    pub y: u32,
    /// First compare word to unpack (the golden `w0`).
    pub word0: u32,
    /// Second compare word to unpack (the golden `w1`).
    pub word1: u32,
    /// Third compare word to unpack (the golden `w2`).
    pub word2: u32,
}

impl PageTableKeyPackQuery {
    /// Builds a query from the five key fields to pack and the three compare
    /// words to unpack.
    #[must_use]
    pub const fn new(
        texture: u32,
        mip: u32,
        layer: u32,
        x: u32,
        y: u32,
        word0: u32,
        word1: u32,
        word2: u32,
    ) -> PageTableKeyPackQuery {
        PageTableKeyPackQuery {
            texture,
            mip,
            layer,
            x,
            y,
            word0,
            word1,
            word2,
        }
    }
}

/// One resolved pack/unpack query, mirroring the reference maps.
///
/// `word0`, `word1` and `word2` are the compare words packed from the key
/// fields; `texture`, `mip`, `layer`, `x` and `y` are the key fields unpacked
/// from the supplied word triple.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PageTableKeyPackResult {
    /// First compare word packed from the key (the golden `w0`).
    pub word0: u32,
    /// Second compare word packed from the key (the golden `w1`).
    pub word1: u32,
    /// Third compare word packed from the key (the golden `w2`).
    pub word2: u32,
    /// Texture id unpacked from the word triple (the golden `texture`).
    pub texture: u32,
    /// Mip level unpacked from the word triple (the golden `mip`).
    pub mip: u32,
    /// Array-layer or cube-face unpacked from the word triple (the golden
    /// `layer`).
    pub layer: u32,
    /// Page-grid X unpacked from the word triple (the golden `x`).
    pub x: u32,
    /// Page-grid Y unpacked from the word triple (the golden `y`).
    pub y: u32,
}

/// Encodes one [`PageTableKeyPackQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &PageTableKeyPackQuery) -> GpuQuery {
    GpuQuery {
        texture: q.texture,
        mip: q.mip,
        layer: q.layer,
        x: q.x,
        y: q.y,
        word0: q.word0,
        word1: q.word1,
        word2: q.word2,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`PageTableKeyPackResult`].
fn decode_result(raw: &GpuResult) -> PageTableKeyPackResult {
    PageTableKeyPackResult {
        word0: raw.word0,
        word1: raw.word1,
        word2: raw.word2,
        texture: raw.texture,
        mip: raw.mip,
        layer: raw.layer,
        x: raw.x,
        y: raw.y,
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

/// A compiled, reusable page-key pack/unpack compute pipeline, twinning the
/// stateless `u32` maps of the `CPU` golden
/// [`compare_words`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::compare_words)
/// and
/// [`unpack_key`](prism_render_architecture::texture_streaming::indirection::GpuPageTable::unpack_key).
pub struct GpuPageTableKeyPack {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPageTableKeyPack {
    /// Compiles the page-key pack/unpack kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPageTableKeyPack {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_page_table_key_pack"),
            source: ShaderSource::Wgsl(PAGE_TABLE_KEY_PACK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_page_table_key_pack_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_page_table_key_pack_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_page_table_key_pack_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPageTableKeyPack {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`PageTableKeyPackResult`] per input, in order.
    ///
    /// Each packed word and unpacked field equals the reference exactly. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[PageTableKeyPackQuery],
    ) -> Vec<PageTableKeyPackResult> {
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
            label: Some("prism_volumetric_page_table_key_pack_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_page_table_key_pack_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_page_table_key_pack_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_page_table_key_pack_bind_group"),
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
            label: Some("prism_volumetric_page_table_key_pack_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_page_table_key_pack_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_page_table_key_pack_pass"),
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
