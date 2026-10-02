//! `wgpu` compute twin of the mip-safe `atlas` gutter / inner-padding contract
//! ([`atlas_mip_padding`](prism_render_architecture::particle::atlas_mip_padding),
//! particle design §16, §22).
//!
//! The `CPU` golden
//! [`atlas_mip_padding`](prism_render_architecture::particle::atlas_mip_padding)
//! owns the integer rules a baker replays to reserve a gutter of padding
//! `texel`s around every `atlas` block so a `mipmap` chain never averages across
//! a block boundary. The gutter-width, safe-level-count, edge-extension and
//! padded-rectangle rules are all pure `u32`/`i32` bit arithmetic, so they form
//! a bit-for-bit lock-step currency between the reference path and a `GPU`
//! kernel. [`GpuAtlasMipPadding`] is the on-device twin: one thread evaluates
//! one query, routed through a [`GpuAtlasPadOp`] selector so one dispatch can
//! mix every operation in a single batch, so a passing real-device parity test
//! is direct evidence the ported kernel reproduces the exact clamp, mirror,
//! saturation and shift the reference computes, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! The twin mirrors the purely-integer subset line for line:
//! [`required_padding`](prism_render_architecture::particle::atlas_mip_padding::required_padding),
//! [`max_safe_mip_levels`](prism_render_architecture::particle::atlas_mip_padding::max_safe_mip_levels),
//! [`gutter_source_index`](prism_render_architecture::particle::atlas_mip_padding::gutter_source_index)
//! across all three [`PadMode`](prism_render_architecture::particle::atlas_mip_padding::PadMode)
//! border rules, and the three
//! [`Rect`](prism_render_architecture::particle::atlas_mip_padding::Rect)
//! accessors
//! [`Rect::right`](prism_render_architecture::particle::atlas_mip_padding::Rect::right),
//! [`Rect::bottom`](prism_render_architecture::particle::atlas_mip_padding::Rect::bottom)
//! and
//! [`Rect::pad_rect`](prism_render_architecture::particle::atlas_mip_padding::Rect::pad_rect).
//! The border rules are selected by [`GpuPadMode`], the operations by
//! [`GpuAtlasPadOp`].
//!
//! # What is not twinned
//!
//! Four golden items are intentionally excluded because they fall outside the
//! portable pure-integer subset:
//! [`Rect::build_padded_uv`](prism_render_architecture::particle::atlas_mip_padding::Rect::build_padded_uv)
//! returns normalized `f32` `UV` rectangles, which belong to the continuous-`f32`
//! tolerance regime rather than this exact-`==` integer twin;
//! [`Rect::border_copy_map`](prism_render_architecture::particle::atlas_mip_padding::Rect::border_copy_map)
//! returns a variable-length [`alloc::vec::Vec`] whose length depends on the
//! block size, which a fixed one-thread-per-query kernel cannot size ahead of
//! time;
//! [`uv_buffer_bytes`](prism_render_architecture::particle::atlas_mip_padding::uv_buffer_bytes)
//! returns a [`u64`], and `WGSL` has no `64`-bit integer type (only `i32`, `u32`,
//! `f32` and `bool`); and
//! [`gpu_storage_bytes`](prism_render_architecture::particle::atlas_mip_padding::gpu_storage_bytes)
//! is a host-side [`usize`] `std430` layout helper with no on-device meaning.
//!
//! # Correctness model
//!
//! Every twinned operation is exact integer arithmetic — a bounded left shift,
//! a leading-zero count, a signed clamp, a signed mirror fold, and saturating
//! `u32` add/subtract — so the `CPU` and `GPU` agree bit for bit. The parity
//! test therefore asserts a strict `==` on every output word with no tolerance:
//! any mismatch is a genuine port defect. Saturating `u32` add reproduces the
//! reference `saturating_add` with a wrap-detect (`s < a` after a modulo-`2^32`
//! sum) and saturating `u32` subtract with the `a < b` guard, since `WGSL` has
//! no saturating integer helper.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer `+ - *`, the
//! bitwise shift `<<`, `min`/`clamp`, `countLeadingZeros`, the signed remainder
//! `%` and a `switch` on the operation and mode codes — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt` and no
//! `64`-bit integer, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! There is no loop: each thread performs a fixed, bounded sequence of integer
//! operations, so the kernel provably terminates.
//!
//! # Degenerate inputs
//!
//! An empty query batch short-circuits on the host with no dispatch, since a
//! storage buffer cannot be zero-sized. A zero `size` resolves every gutter
//! lookup to the [`u32::MAX`] sentinel, matching the reference. A zero
//! `mip_levels` needs no gutter and returns `0`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_mip_padding`
//! 的纯 `u32`/`i32` 子集；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `atlas` mip-padding kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` golden
/// [`atlas_mip_padding`](prism_render_architecture::particle::atlas_mip_padding)
/// pure-integer subset line for line; see the module documentation for the
/// algorithm.
const ATLAS_MIP_PADDING_WGSL: &str = r#"
// Atlas mip-padding twin: one thread per query dispatches on an operation code
// to one of the six pure-integer operations and writes a vec4<u32> result (the
// scalar operations fill component x, pad_rect fills all four). It mirrors the
// CPU golden particle::atlas_mip_padding line for line, uses only the portable
// core-WGSL subset (integer + - *, the shift <<, min/clamp, countLeadingZeros,
// the signed remainder % and a switch on the op and mode codes), has no 64-bit
// integer and takes no optional feature, so it runs unmodified on Metal, Vulkan
// and DX12. Every operation is exact integer arithmetic, so CPU and GPU agree
// bit for bit. There is no loop, so the kernel provably terminates.
//
// Provenance: twinned from this repository's particle::atlas_mip_padding
// pure-u32/i32 subset; no third-party engine source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 32-byte std430 stride matching the host `GpuQuery`: the four u32
// rect fields, the signed gutter coordinate, the pad width, the border mode
// code and the operation code. Unused fields are ignored per operation.
struct Query {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    coord: i32,
    pad: u32,
    mode: u32,
    op: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<vec4<u32>>;

// u32::MAX, the "no source texel" sentinel returned for out-of-range gutter
// lookups and a zero block size.
const SENTINEL: u32 = 4294967295u;

// Saturating u32 add. WGSL integer `+` wraps modulo 2^32, so the sum is below
// `a` exactly on overflow, reproducing the reference `saturating_add` by
// clamping to u32::MAX there.
fn sat_add_u32(a: u32, b: u32) -> u32 {
    let s = a + b;
    if (s < a) {
        return SENTINEL;
    }
    return s;
}

// Saturating u32 subtract, clamping to zero on underflow to reproduce the
// reference `saturating_sub`.
fn sat_sub_u32(a: u32, b: u32) -> u32 {
    if (a < b) {
        return 0u;
    }
    return a - b;
}

// required_padding: `1 << (mip_levels - 1)` is the coarsest level's
// half-footprint. A zero level count needs no gutter; the shift is clamped to
// 31 so an absurd level count saturates instead of overflowing.
fn required_padding(mip_levels: u32) -> u32 {
    if (mip_levels == 0u) {
        return 0u;
    }
    let shift = min(mip_levels - 1u, 31u);
    return 1u << shift;
}

// max_safe_mip_levels: `floor(log2(min(w, h))) + 1`, computed as
// `32 - countLeadingZeros(m)` for the smaller dimension `m`, matching the
// reference use of leading-zero counting. A zero dimension carries no level.
fn max_safe_mip_levels(w: u32, h: u32) -> u32 {
    let m = min(w, h);
    if (m == 0u) {
        return 0u;
    }
    return 32u - countLeadingZeros(m);
}

// gutter_source_index: maps a block-local coordinate back to a source texel in
// [0, size) under the border mode. Mode 0 clamps to the nearest edge, mode 1
// mirrors across the edges without repeating them, mode 2 is transparent
// (returns the sentinel outside [0, size)). A zero size returns the sentinel.
// The `size > i32::MAX` guard reproduces the reference `try_from` clamp to
// i32::MAX; the mirror period `s * 2` matches the reference within the realistic
// size domain (block sizes stay far below i32::MAX / 2, so no overflow).
fn gutter_source_index(coord: i32, size: u32, mode: u32) -> u32 {
    if (size == 0u) {
        return SENTINEL;
    }
    var s: i32;
    if (size > 2147483647u) {
        s = 2147483647;
    } else {
        s = i32(size);
    }
    switch (mode) {
        case 0u: {
            // ClampEdge: clamp into [0, size - 1]; the result is non-negative.
            return u32(clamp(coord, 0, s - 1));
        }
        case 1u: {
            // Mirror: reflect across the edges without repeating them.
            let period = s * 2;
            var c = coord % period;
            if (c < 0) {
                c = c + period;
            }
            if (c >= s) {
                c = period - 1 - c;
            }
            return u32(c);
        }
        case 2u: {
            // Transparent: in-range coordinates pass through, else sentinel.
            if (coord >= 0 && coord < s) {
                return u32(coord);
            }
            return SENTINEL;
        }
        default: {
            return SENTINEL;
        }
    }
}

// pad_rect: grow a rect outward by `pad` on every side, clamping the origin at
// (0, 0) and folding the clipped left/top padding into the width/height so the
// exclusive right/bottom edges stay at x + w + pad and y + h + pad. All
// arithmetic saturates, matching the reference.
fn pad_rect(x: u32, y: u32, w: u32, h: u32, pad: u32) -> vec4<u32> {
    let nx = sat_sub_u32(x, pad);
    let ny = sat_sub_u32(y, pad);
    let left = x - nx;
    let top = y - ny;
    let nw = sat_add_u32(sat_add_u32(w, left), pad);
    let nh = sat_add_u32(sat_add_u32(h, top), pad);
    return vec4<u32>(nx, ny, nw, nh);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];
    var out = vec4<u32>(0u, 0u, 0u, 0u);
    switch (q.op) {
        case 0u: { out.x = required_padding(q.x); }
        case 1u: { out.x = max_safe_mip_levels(q.w, q.h); }
        case 2u: { out.x = gutter_source_index(q.coord, q.w, q.mode); }
        case 3u: { out.x = sat_add_u32(q.x, q.w); }
        case 4u: { out.x = sat_add_u32(q.y, q.h); }
        case 5u: { out = pad_rect(q.x, q.y, q.w, q.h, q.pad); }
        default: { out = vec4<u32>(0u, 0u, 0u, 0u); }
    }
    results[idx] = out;
}
"#;

/// The border rule the gutter uses outside a block's content region.
///
/// Each variant names one of the golden
/// [`PadMode`](prism_render_architecture::particle::atlas_mip_padding::PadMode)
/// cases; the stable [`GpuPadMode::to_code`] mapping is the mode code the
/// kernel's `switch` dispatches on in
/// [`gutter_source_index`](prism_render_architecture::particle::atlas_mip_padding::gutter_source_index).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_mip_padding`；
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuPadMode {
    /// Repeat the nearest edge `texel` outward (`ClampEdge`).
    ClampEdge,
    /// Reflect the content across each edge without repeating it (`Mirror`).
    Mirror,
    /// Leave the gutter empty; out-of-range coordinates resolve to the
    /// [`u32::MAX`] sentinel (`Transparent`).
    Transparent,
}

impl GpuPadMode {
    /// The stable `u32` code the kernel `switch` dispatches on.
    #[must_use]
    const fn to_code(self) -> u32 {
        match self {
            GpuPadMode::ClampEdge => 0,
            GpuPadMode::Mirror => 1,
            GpuPadMode::Transparent => 2,
        }
    }
}

/// The `atlas` mip-padding operation an individual query selects.
///
/// Each variant names one of the purely-integer golden items mirrored by this
/// twin; the stable [`GpuAtlasPadOp::to_code`] mapping is the operation code the
/// kernel's `switch` dispatches on. The operands each operation reads are:
/// [`GpuAtlasPadOp::RequiredPadding`] reads [`GpuAtlasPadQuery::x`] as the mip
/// level count; [`GpuAtlasPadOp::MaxSafeMipLevels`] reads [`GpuAtlasPadQuery::w`]
/// and [`GpuAtlasPadQuery::h`]; [`GpuAtlasPadOp::GutterSourceIndex`] reads
/// [`GpuAtlasPadQuery::coord`], [`GpuAtlasPadQuery::w`] as the size and
/// [`GpuAtlasPadQuery::mode`]; [`GpuAtlasPadOp::RectRight`] reads
/// [`GpuAtlasPadQuery::x`] and [`GpuAtlasPadQuery::w`];
/// [`GpuAtlasPadOp::RectBottom`] reads [`GpuAtlasPadQuery::y`] and
/// [`GpuAtlasPadQuery::h`]; and [`GpuAtlasPadOp::PadRect`] reads all four rect
/// fields plus [`GpuAtlasPadQuery::pad`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_mip_padding`；
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuAtlasPadOp {
    /// Gutter width for a `mipmap` chain (`required_padding`).
    RequiredPadding,
    /// Largest safe `mipmap` level count for a block (`max_safe_mip_levels`).
    MaxSafeMipLevels,
    /// Map a padded coordinate back to a source `texel` (`gutter_source_index`).
    GutterSourceIndex,
    /// Exclusive right edge of a rect (`Rect::right`).
    RectRight,
    /// Exclusive bottom edge of a rect (`Rect::bottom`).
    RectBottom,
    /// Rect grown outward by a gutter (`Rect::pad_rect`); fills all four result
    /// components with `x`, `y`, `w`, `h`.
    PadRect,
}

impl GpuAtlasPadOp {
    /// The stable `u32` code the kernel `switch` dispatches on.
    #[must_use]
    const fn to_code(self) -> u32 {
        match self {
            GpuAtlasPadOp::RequiredPadding => 0,
            GpuAtlasPadOp::MaxSafeMipLevels => 1,
            GpuAtlasPadOp::GutterSourceIndex => 2,
            GpuAtlasPadOp::RectRight => 3,
            GpuAtlasPadOp::RectBottom => 4,
            GpuAtlasPadOp::PadRect => 5,
        }
    }
}

/// One `atlas` mip-padding query: the operation, its mode and raw operands.
///
/// The rect fields `x`, `y`, `w`, `h` are `texel` coordinates/sizes; `coord` is
/// the signed block-local gutter coordinate; `pad` is the gutter width for
/// [`GpuAtlasPadOp::PadRect`]; and `mode` selects the border rule for
/// [`GpuAtlasPadOp::GutterSourceIndex`]. Fields unused by the selected operation
/// are ignored by the kernel. For [`GpuAtlasPadOp::GutterSourceIndex`] the block
/// size is read from `w`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_mip_padding`；
/// 无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuAtlasPadQuery {
    /// Rect left edge, the mip level count, or the `right` left operand.
    pub x: u32,
    /// Rect top edge, or the `bottom` top operand.
    pub y: u32,
    /// Rect width, the block size for the gutter lookup, or a dimension.
    pub w: u32,
    /// Rect height, or a dimension.
    pub h: u32,
    /// Signed block-local gutter coordinate (read by `GutterSourceIndex`).
    pub coord: i32,
    /// Gutter width in `texel`s (read by `PadRect`).
    pub pad: u32,
    /// Border rule for the gutter lookup (read by `GutterSourceIndex`).
    pub mode: GpuPadMode,
    /// Which golden operation to evaluate.
    pub op: GpuAtlasPadOp,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`ATLAS_MIP_PADDING_WGSL`]: the query count and three pad words —
/// `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `32`-byte `std430` stride matching `Query` in the
/// shader: four `u32` rect fields, the `i32` gutter coordinate, the `u32` pad
/// width, the `u32` border mode code and the `u32` operation code.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Rect left edge or mip level count.
    x: u32,
    /// Rect top edge.
    y: u32,
    /// Rect width or block size.
    w: u32,
    /// Rect height.
    h: u32,
    /// Signed gutter coordinate.
    coord: i32,
    /// Gutter width.
    pad: u32,
    /// Border mode code from [`GpuPadMode::to_code`].
    mode: u32,
    /// Operation code from [`GpuAtlasPadOp::to_code`].
    op: u32,
}

impl GpuQuery {
    /// Packs a [`GpuAtlasPadQuery`] into the `std430` upload layout.
    fn from_query(query: &GpuAtlasPadQuery) -> GpuQuery {
        GpuQuery {
            x: query.x,
            y: query.y,
            w: query.w,
            h: query.h,
            coord: query.coord,
            pad: query.pad,
            mode: query.mode.to_code(),
            op: query.op.to_code(),
        }
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

/// A compiled, reusable `atlas` mip-padding operation pipeline.
pub struct GpuAtlasMipPadding {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAtlasMipPadding {
    /// Compiles the `atlas` mip-padding kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAtlasMipPadding {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding"),
            source: ShaderSource::Wgsl(ATLAS_MIP_PADDING_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAtlasMipPadding {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning the resulting
    /// `[u32; 4]` for each query in input order.
    ///
    /// Scalar operations return their value in component `0` (the remaining
    /// components are `0`); [`GpuAtlasPadOp::PadRect`] fills all four components
    /// with the padded rect's `x`, `y`, `w`, `h`. Each returned word equals the
    /// corresponding golden function evaluated on the query operands bit for
    /// bit. An empty `queries` slice yields an empty vector — storage buffers
    /// cannot be zero-sized, so it is handled by an early return before any
    /// dispatch.
    #[must_use]
    pub fn run(&self, ctx: &GpuContext, queries: &[GpuAtlasPadQuery]) -> Vec<[u32; 4]> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<[u32; 4]>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_atlas_mip_padding_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_atlas_mip_padding_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, [u32; 4]>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw
    }
}
