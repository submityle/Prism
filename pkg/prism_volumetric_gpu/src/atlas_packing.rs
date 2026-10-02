//! `wgpu` compute twin of the deterministic, closed-form core of the
//! sprite-atlas shelf rectangle packer
//! ([`atlas_packing`](prism_render_architecture::particle::atlas_packing),
//! particle design §16, §22).
//!
//! The `CPU` golden
//! [`atlas_packing`](prism_render_architecture::particle::atlas_packing) owns a
//! stateful bake pass: a stable sort by decreasing height, a variable-length
//! shelf-packing loop that grows a `Vec` of placements, and `u64` byte/area
//! accounting that must not overflow. None of that fits a
//! one-thread-one-element kernel. What *does* port cleanly is the per-rectangle
//! numeric spine the loop and the renderer evaluate point-wise: the saturating
//! right/bottom edge
//! ([`PackedRect::right`](prism_render_architecture::particle::atlas_packing::PackedRect::right),
//! [`PackedRect::bottom`](prism_render_architecture::particle::atlas_packing::PackedRect::bottom)),
//! the normalized `UV` rectangle
//! ([`PackedRect::uv_rect`](prism_render_architecture::particle::atlas_packing::PackedRect::uv_rect)),
//! the pairwise half-open overlap test
//! ([`rects_overlap`](prism_render_architecture::particle::atlas_packing::rects_overlap)),
//! the single-rectangle coverage fraction
//! ([`occupancy`](prism_render_architecture::particle::atlas_packing::occupancy)),
//! the fits predicate inlined in the packer loop, and the one-step shelf
//! placement transition that loop performs per rectangle.
//!
//! [`GpuAtlasPacking`] is the on-device twin: one thread solves one
//! [`AtlasPackingQuery`] and writes one [`AtlasPackingResult`], so a passing
//! real-device parity test is direct evidence the ported kernel folds the same
//! edges, verdicts, fractions and placements the reference does, not merely that
//! the shader compiles.
//!
//! # What is twinned
//!
//! Each query selects one routine by a `u32` tag and the kernel reproduces it
//! branch for branch: the `x + width` saturating edge, the `[0, 1]`-normalized
//! `UV` rectangle (all-zero for a zero-dimension `atlas`), the half-open
//! `a.x < b.right && ...` overlap verdict, the clamped single-rectangle coverage
//! fraction, the "rectangle plus padding fits the remaining shelf" predicate,
//! and the shelf transition that wraps to a new row, reports whether the
//! rectangle fits and advances the cursor.
//!
//! # What is left on the host
//!
//! The variable-length and `u64`-accounting machinery is deliberately *not*
//! twinned here, since a single thread cannot own unbounded, growing state or a
//! 64-bit accumulator:
//! [`ShelfPacker::pack`](prism_render_architecture::particle::atlas_packing::ShelfPacker::pack)
//! sorts and drives the growing placement `Vec`; the full
//! [`occupancy`](prism_render_architecture::particle::atlas_packing::occupancy)
//! accumulates `used` area across a variable-length slice in `u64`;
//! [`RectSize::area`](prism_render_architecture::particle::atlas_packing::RectSize::area)
//! and
//! [`uv_buffer_bytes`](prism_render_architecture::particle::atlas_packing::uv_buffer_bytes)
//! are `u64` byte/area computations that saturate against overflow. The host
//! runs those (keeping its `u64` guards) and feeds the kernel the fixed-length,
//! stateless inputs (one edge, one rectangle, a cursor-and-rectangle step) each
//! twinned routine consumes.
//!
//! # No transcendental math
//!
//! Every routine is unsigned integer arithmetic or a `u32`-to-`f32` widen plus a
//! divide and a clamp. The kernel uses no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no inverse trigonometry, no `smoothstep` and no `round`, and it uses
//! only `i32`/`u32`/`f32` (no `u64`): the `u64` area/byte accounting stays on the
//! host, and the twin reproduces the single-rectangle coverage fraction in `f32`
//! (safe because its fixtures keep every operand well below the `2^24` exact
//! range).
//!
//! # Correctness model
//!
//! The dispatch tag is an integer classification, so the kernel runs exactly the
//! branch the host requested. The integer and boolean routines (the saturating
//! edges, the overlap and fits verdicts and the shelf placement) are bit-exact
//! and are compared with `==`. The continuous entries (the `UV` rectangle and
//! the coverage fraction) thread through a `u32`-to-`f32` widen and a divide, so
//! `CPU` and `GPU` are not guaranteed bit-exact: a `GPU` may round a widen or a
//! divide a hair differently. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every
//! continuous quantity.
//!
//! # Degenerate inputs
//!
//! The reference returns an all-zero `UV` rectangle and a zero coverage fraction
//! for a zero-dimension `atlas`; the twin reproduces both guards exactly. The
//! fixtures otherwise keep the `atlas` dimensions positive and every edge well
//! below `u32::MAX` (one saturating fixture aside, compared with `==`), so the
//! widening divisions are exact and no accumulation can overflow.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_packing`；无第三方引擎源码或衍生代码。
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
use prism_render_architecture::particle::atlas_packing::{occupancy, rects_overlap, PackedRect};

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` atlas-packing kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`atlas_packing`](prism_render_architecture::particle::atlas_packing) numeric
/// core branch for branch; see the module documentation for the algorithm.
const ATLAS_PACKING_WGSL: &str = r#"
// Atlas-packing twin: one thread per query runs the routine its `tag` selects,
// reproducing the CPU golden `particle::atlas_packing` numeric core branch for
// branch. It uses only the portable core-WGSL subset (min/max/clamp/floor, the
// u32 arithmetic and comparisons and + - * / plus a widening divide), needs no
// transcendental call, no u64 and no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. The kernel has no loop, so it provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::atlas_packing；无第三方引擎源码
// 或衍生代码。

// Routine tags; the host casts its query discriminant straight to these codes.
const TAG_BOTTOM_EDGE: u32 = 0u;
const TAG_FITS: u32 = 1u;
const TAG_OCCUPANCY: u32 = 2u;
const TAG_OVERLAP: u32 = 3u;
const TAG_RIGHT_EDGE: u32 = 4u;
const TAG_SHELF_PLACEMENT: u32 = 5u;
const TAG_UV_RECT: u32 = 6u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Routine selector matching the TAG_* codes.
    tag: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // Primary unsigned lane. Its meaning is per-routine:
    //   bottom/right edge: (origin, extent, _, _)
    //   fits:              (cursor_x, cursor_y, padded_w, padded_h)
    //   occupancy:         (width, height, atlas_w, atlas_h)
    //   overlap:           rect a as (x, y, width, height)
    //   shelf placement:   (cursor_x, cursor_y, shelf_height, padded_w)
    //   uv rect:           (x, y, width, height)
    ua: vec4<u32>,
    // Secondary unsigned lane.
    //   fits / uv rect:    (atlas_w, atlas_h, _, _)
    //   overlap:           rect b as (x, y, width, height)
    //   shelf placement:   (padded_h, atlas_w, atlas_h, _)
    ub: vec4<u32>,
}

struct Result {
    // Continuous lane: uv rect (u0,v0,u1,v1) or coverage fraction in x.
    scalar: vec4<f32>,
    // Word lane: edge / 0-or-1 verdict in x; shelf placement fits,x,y in x,y,z.
    word: vec4<u32>,
    // Second word lane: shelf placement new cursor_x, cursor_y, shelf_height.
    word2: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Saturating u32 add: WGSL u32 addition wraps by definition, so detect the wrap
// (the sum dropped below an operand) and clamp to u32::MAX, mirroring the host
// `saturating_add`.
fn sat_add(a: u32, b: u32) -> u32 {
    let s = a + b;
    return select(s, 0xFFFFFFFFu, s < a);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    out.scalar = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.word = vec4<u32>(0u, 0u, 0u, 0u);
    out.word2 = vec4<u32>(0u, 0u, 0u, 0u);

    if (q.tag == TAG_BOTTOM_EDGE || q.tag == TAG_RIGHT_EDGE) {
        // Both are the same saturating `origin + extent` edge.
        out.word.x = sat_add(q.ua.x, q.ua.y);
    } else if (q.tag == TAG_FITS) {
        // Fits when neither the padded right nor the padded bottom overflows the
        // atlas; the golden packer fails on the strict `>` of either axis.
        let over_x = sat_add(q.ua.x, q.ua.z) > q.ub.x;
        let over_y = sat_add(q.ua.y, q.ua.w) > q.ub.y;
        let fits = !(over_x || over_y);
        out.word.x = select(0u, 1u, fits);
    } else if (q.tag == TAG_OCCUPANCY) {
        // Single-rectangle coverage fraction, clamped to [0, 1]; a zero-area
        // atlas yields zero, matching the golden `total == 0` guard.
        if (q.ua.z == 0u || q.ua.w == 0u) {
            out.scalar.x = 0.0;
        } else {
            let used = f32(q.ua.x) * f32(q.ua.y);
            let total = f32(q.ua.z) * f32(q.ua.w);
            out.scalar.x = clamp(used / total, 0.0, 1.0);
        }
    } else if (q.tag == TAG_OVERLAP) {
        // Half-open overlap: rectangles touching edge-to-edge do not overlap.
        let a_right = sat_add(q.ua.x, q.ua.z);
        let a_bottom = sat_add(q.ua.y, q.ua.w);
        let b_right = sat_add(q.ub.x, q.ub.z);
        let b_bottom = sat_add(q.ub.y, q.ub.w);
        let hit = (q.ua.x < b_right) && (q.ub.x < a_right)
            && (q.ua.y < b_bottom) && (q.ub.y < a_bottom);
        out.word.x = select(0u, 1u, hit);
    } else if (q.tag == TAG_SHELF_PLACEMENT) {
        // One step of the golden packer loop: wrap to a fresh shelf when the
        // current row cannot hold the padded rectangle, test the fit, then
        // advance the cursor and grow the shelf height.
        let cursor_x = q.ua.x;
        let cursor_y = q.ua.y;
        let shelf_height = q.ua.z;
        let padded_w = q.ua.w;
        let padded_h = q.ub.x;
        let atlas_w = q.ub.y;
        let atlas_h = q.ub.z;

        var cx = cursor_x;
        var cy = cursor_y;
        var sh = shelf_height;
        if (sat_add(cursor_x, padded_w) > atlas_w) {
            cy = sat_add(cursor_y, shelf_height);
            cx = 0u;
            sh = 0u;
        }
        let over_x = sat_add(cx, padded_w) > atlas_w;
        let over_y = sat_add(cy, padded_h) > atlas_h;
        let fits = !(over_x || over_y);
        out.word.x = select(0u, 1u, fits);
        out.word.y = cx;
        out.word.z = cy;
        out.word2.x = sat_add(cx, padded_w);
        out.word2.y = cy;
        out.word2.z = max(sh, padded_h);
    } else {
        // TAG_UV_RECT: normalized UV rectangle, all-zero for a zero-dimension
        // atlas so a degenerate atlas never divides by zero.
        if (q.ub.x == 0u || q.ub.y == 0u) {
            out.scalar = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        } else {
            let aw = f32(q.ub.x);
            let ah = f32(q.ub.y);
            let right = sat_add(q.ua.x, q.ua.z);
            let bottom = sat_add(q.ua.y, q.ua.w);
            out.scalar.x = f32(q.ua.x) / aw;
            out.scalar.y = f32(q.ua.y) / ah;
            out.scalar.z = f32(right) / aw;
            out.scalar.w = f32(bottom) / ah;
        }
    }

    results[idx] = out;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the batch.
    count: u32,
    /// Padding to a `16`-byte boundary.
    pad0: u32,
    /// Padding to a `16`-byte boundary.
    pad1: u32,
    /// Padding to a `16`-byte boundary.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Routine tag selecting the branch to run.
    tag: u32,
    /// Padding to a `16`-byte boundary.
    pad0: u32,
    /// Padding to a `16`-byte boundary.
    pad1: u32,
    /// Padding to a `16`-byte boundary.
    pad2: u32,
    /// Primary unsigned lane (see the `WGSL` `Query` doc for the per-routine
    /// meaning).
    ua: [u32; 4],
    /// Secondary unsigned lane.
    ub: [u32; 4],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Continuous lane: `UV` rectangle or coverage fraction in `x`.
    scalar: [f32; 4],
    /// Word lane: edge or `0`/`1` verdict in `x`; placement `fits`, `x`, `y` in
    /// `x`, `y`, `z`.
    word: [u32; 4],
    /// Second word lane: placement new `cursor_x`, `cursor_y`, `shelf_height`.
    word2: [u32; 4],
}

/// One query for the atlas-packing twin: a tagged union selecting which golden
/// routine to run with its typed inputs.
///
/// Each variant twins exactly one deterministic, closed-form routine of the
/// golden [`atlas_packing`](prism_render_architecture::particle::atlas_packing)
/// contract; the stateful packer, the `u64` accounting and the variable-length
/// aggregation stay on the host (see the module docs).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_packing`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AtlasPackingQuery {
    /// The saturating bottom edge `y + height`, twinning
    /// [`PackedRect::bottom`](prism_render_architecture::particle::atlas_packing::PackedRect::bottom).
    BottomEdge {
        /// Top edge in texels.
        y: u32,
        /// Height in texels.
        height: u32,
    },
    /// The packer's inlined fits predicate: whether a padded rectangle placed at
    /// the cursor stays within the `atlas` on both axes.
    Fits {
        /// Current shelf cursor `x` in texels.
        cursor_x: u32,
        /// Current shelf cursor `y` in texels.
        cursor_y: u32,
        /// Rectangle width already grown by the gutter padding.
        padded_width: u32,
        /// Rectangle height already grown by the gutter padding.
        padded_height: u32,
        /// Clamped `atlas` width in texels.
        atlas_width: u32,
        /// Clamped `atlas` height in texels.
        atlas_height: u32,
    },
    /// The single-rectangle coverage fraction, twinning the per-rectangle term
    /// of
    /// [`occupancy`](prism_render_architecture::particle::atlas_packing::occupancy)
    /// (the host keeps the `u64` accumulation across the full slice).
    Occupancy {
        /// Rectangle width in texels.
        width: u32,
        /// Rectangle height in texels.
        height: u32,
        /// `atlas` width in texels.
        atlas_width: u32,
        /// `atlas` height in texels.
        atlas_height: u32,
    },
    /// The pairwise half-open overlap test, twinning
    /// [`rects_overlap`](prism_render_architecture::particle::atlas_packing::rects_overlap).
    Overlap {
        /// First rectangle as `(x, y, width, height)` in texels.
        a: [u32; 4],
        /// Second rectangle as `(x, y, width, height)` in texels.
        b: [u32; 4],
    },
    /// The saturating right edge `x + width`, twinning
    /// [`PackedRect::right`](prism_render_architecture::particle::atlas_packing::PackedRect::right).
    RightEdge {
        /// Left edge in texels.
        x: u32,
        /// Width in texels.
        width: u32,
    },
    /// One step of the shelf-placement transition the golden packer loop runs
    /// per rectangle: wrap to a new row if needed, test the fit, and report the
    /// placement and the advanced cursor.
    ShelfPlacement {
        /// Current shelf cursor `x` in texels.
        cursor_x: u32,
        /// Current shelf cursor `y` in texels.
        cursor_y: u32,
        /// Current shelf height in texels.
        shelf_height: u32,
        /// Rectangle width already grown by the gutter padding.
        padded_width: u32,
        /// Rectangle height already grown by the gutter padding.
        padded_height: u32,
        /// Clamped `atlas` width in texels.
        atlas_width: u32,
        /// Clamped `atlas` height in texels.
        atlas_height: u32,
    },
    /// The normalized `UV` rectangle `[u0, v0, u1, v1]`, twinning
    /// [`PackedRect::uv_rect`](prism_render_architecture::particle::atlas_packing::PackedRect::uv_rect).
    UvRect {
        /// Left edge in texels.
        x: u32,
        /// Top edge in texels.
        y: u32,
        /// Width in texels.
        width: u32,
        /// Height in texels.
        height: u32,
        /// `atlas` width in texels.
        atlas_width: u32,
        /// `atlas` height in texels.
        atlas_height: u32,
    },
}

impl AtlasPackingQuery {
    /// Returns the `WGSL` routine tag for this query.
    fn tag(&self) -> u32 {
        match self {
            AtlasPackingQuery::BottomEdge { .. } => 0,
            AtlasPackingQuery::Fits { .. } => 1,
            AtlasPackingQuery::Occupancy { .. } => 2,
            AtlasPackingQuery::Overlap { .. } => 3,
            AtlasPackingQuery::RightEdge { .. } => 4,
            AtlasPackingQuery::ShelfPlacement { .. } => 5,
            AtlasPackingQuery::UvRect { .. } => 6,
        }
    }
}

/// One resolved answer for a single query: a tagged union whose variant matches
/// the routine the corresponding [`AtlasPackingQuery`] selected.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_packing`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AtlasPackingResult {
    /// A saturating texel edge (`right` or `bottom`).
    Edge(u32),
    /// A boolean verdict (`fits` or `overlap`).
    Flag(bool),
    /// A resolved shelf-placement step.
    Placement {
        /// Whether the rectangle fit the (possibly wrapped) shelf.
        fits: bool,
        /// Placement left edge in texels.
        x: u32,
        /// Placement top edge in texels.
        y: u32,
        /// Shelf cursor `x` after advancing past this rectangle.
        cursor_x: u32,
        /// Shelf cursor `y` after this rectangle.
        cursor_y: u32,
        /// Shelf height after this rectangle.
        shelf_height: u32,
    },
    /// A continuous scalar (the coverage fraction).
    Scalar(f32),
    /// A normalized `UV` rectangle `[u0, v0, u1, v1]`.
    Uv([f32; 4]),
}

/// The `CPU` golden verdict for one query, dispatching to the reference entry
/// points (or faithfully re-deriving the inline closed forms) so callers (and
/// the parity test) can pin the twin lane for lane.
///
/// The public golden
/// [`PackedRect::right`](prism_render_architecture::particle::atlas_packing::PackedRect::right),
/// [`PackedRect::bottom`](prism_render_architecture::particle::atlas_packing::PackedRect::bottom),
/// [`PackedRect::uv_rect`](prism_render_architecture::particle::atlas_packing::PackedRect::uv_rect),
/// [`rects_overlap`](prism_render_architecture::particle::atlas_packing::rects_overlap)
/// and
/// [`occupancy`](prism_render_architecture::particle::atlas_packing::occupancy)
/// (over a single-rectangle slice) are called directly; the packer's inline
/// fits predicate and its per-rectangle shelf transition are re-derived from the
/// loop body of
/// [`ShelfPacker::pack`](prism_render_architecture::particle::atlas_packing::ShelfPacker::pack),
/// since they are not exported as standalone functions.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_packing`；无第三方引擎源码或衍生代码。
#[must_use]
pub fn cpu_reference(query: &AtlasPackingQuery) -> AtlasPackingResult {
    match query {
        AtlasPackingQuery::BottomEdge { y, height } => {
            let rect = PackedRect {
                id: 0,
                x: 0,
                y: *y,
                width: 0,
                height: *height,
            };
            AtlasPackingResult::Edge(rect.bottom())
        }
        AtlasPackingQuery::Fits {
            cursor_x,
            cursor_y,
            padded_width,
            padded_height,
            atlas_width,
            atlas_height,
        } => {
            let over_x = cursor_x.saturating_add(*padded_width) > *atlas_width;
            let over_y = cursor_y.saturating_add(*padded_height) > *atlas_height;
            AtlasPackingResult::Flag(!(over_x || over_y))
        }
        AtlasPackingQuery::Occupancy {
            width,
            height,
            atlas_width,
            atlas_height,
        } => {
            let rect = PackedRect {
                id: 0,
                x: 0,
                y: 0,
                width: *width,
                height: *height,
            };
            AtlasPackingResult::Scalar(occupancy(&[rect], *atlas_width, *atlas_height))
        }
        AtlasPackingQuery::Overlap { a, b } => {
            let ra = PackedRect {
                id: 0,
                x: a[0],
                y: a[1],
                width: a[2],
                height: a[3],
            };
            let rb = PackedRect {
                id: 1,
                x: b[0],
                y: b[1],
                width: b[2],
                height: b[3],
            };
            AtlasPackingResult::Flag(rects_overlap(&ra, &rb))
        }
        AtlasPackingQuery::RightEdge { x, width } => {
            let rect = PackedRect {
                id: 0,
                x: *x,
                y: 0,
                width: *width,
                height: 0,
            };
            AtlasPackingResult::Edge(rect.right())
        }
        AtlasPackingQuery::ShelfPlacement {
            cursor_x,
            cursor_y,
            shelf_height,
            padded_width,
            padded_height,
            atlas_width,
            atlas_height,
        } => {
            let mut cx = *cursor_x;
            let mut cy = *cursor_y;
            let mut sh = *shelf_height;
            if cx.saturating_add(*padded_width) > *atlas_width {
                cy = cy.saturating_add(sh);
                cx = 0;
                sh = 0;
            }
            let over_x = cx.saturating_add(*padded_width) > *atlas_width;
            let over_y = cy.saturating_add(*padded_height) > *atlas_height;
            let fits = !(over_x || over_y);
            AtlasPackingResult::Placement {
                fits,
                x: cx,
                y: cy,
                cursor_x: cx.saturating_add(*padded_width),
                cursor_y: cy,
                shelf_height: sh.max(*padded_height),
            }
        }
        AtlasPackingQuery::UvRect {
            x,
            y,
            width,
            height,
            atlas_width,
            atlas_height,
        } => {
            let rect = PackedRect {
                id: 0,
                x: *x,
                y: *y,
                width: *width,
                height: *height,
            };
            AtlasPackingResult::Uv(rect.uv_rect(*atlas_width, *atlas_height))
        }
    }
}

/// Encodes one [`AtlasPackingQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &AtlasPackingQuery) -> GpuQuery {
    let mut g = GpuQuery::zeroed();
    g.tag = q.tag();
    match q {
        AtlasPackingQuery::BottomEdge { y, height } => {
            g.ua = [*y, *height, 0, 0];
        }
        AtlasPackingQuery::RightEdge { x, width } => {
            g.ua = [*x, *width, 0, 0];
        }
        AtlasPackingQuery::Fits {
            cursor_x,
            cursor_y,
            padded_width,
            padded_height,
            atlas_width,
            atlas_height,
        } => {
            g.ua = [*cursor_x, *cursor_y, *padded_width, *padded_height];
            g.ub = [*atlas_width, *atlas_height, 0, 0];
        }
        AtlasPackingQuery::Occupancy {
            width,
            height,
            atlas_width,
            atlas_height,
        } => {
            g.ua = [*width, *height, *atlas_width, *atlas_height];
        }
        AtlasPackingQuery::Overlap { a, b } => {
            g.ua = *a;
            g.ub = *b;
        }
        AtlasPackingQuery::ShelfPlacement {
            cursor_x,
            cursor_y,
            shelf_height,
            padded_width,
            padded_height,
            atlas_width,
            atlas_height,
        } => {
            g.ua = [*cursor_x, *cursor_y, *shelf_height, *padded_width];
            g.ub = [*padded_height, *atlas_width, *atlas_height, 0];
        }
        AtlasPackingQuery::UvRect {
            x,
            y,
            width,
            height,
            atlas_width,
            atlas_height,
        } => {
            g.ua = [*x, *y, *width, *height];
            g.ub = [*atlas_width, *atlas_height, 0, 0];
        }
    }
    g
}

/// Decodes one packed [`GpuResult`] into the public [`AtlasPackingResult`],
/// selecting the variant from the query's routine.
fn decode_result(q: &AtlasPackingQuery, raw: &GpuResult) -> AtlasPackingResult {
    match q {
        AtlasPackingQuery::BottomEdge { .. } | AtlasPackingQuery::RightEdge { .. } => {
            AtlasPackingResult::Edge(raw.word[0])
        }
        AtlasPackingQuery::Fits { .. } | AtlasPackingQuery::Overlap { .. } => {
            AtlasPackingResult::Flag(raw.word[0] != 0)
        }
        AtlasPackingQuery::Occupancy { .. } => AtlasPackingResult::Scalar(raw.scalar[0]),
        AtlasPackingQuery::ShelfPlacement { .. } => AtlasPackingResult::Placement {
            fits: raw.word[0] != 0,
            x: raw.word[1],
            y: raw.word[2],
            cursor_x: raw.word2[0],
            cursor_y: raw.word2[1],
            shelf_height: raw.word2[2],
        },
        AtlasPackingQuery::UvRect { .. } => AtlasPackingResult::Uv(raw.scalar),
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

/// A compiled, reusable atlas-packing compute pipeline, twinning the `CPU`
/// golden [`atlas_packing`](prism_render_architecture::particle::atlas_packing)
/// numeric core.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::atlas_packing`；无第三方引擎源码或衍生代码。
pub struct GpuAtlasPacking {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuAtlasPacking {
    /// Compiles the atlas-packing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuAtlasPacking {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_atlas_packing"),
            source: ShaderSource::Wgsl(ATLAS_PACKING_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_atlas_packing_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_atlas_packing_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_atlas_packing_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuAtlasPacking {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`AtlasPackingResult`]
    /// per input, in order.
    ///
    /// The result variant matches the routine each query selected, matching the
    /// reference to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[AtlasPackingQuery],
    ) -> Vec<AtlasPackingResult> {
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
            label: Some("prism_volumetric_atlas_packing_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_atlas_packing_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_atlas_packing_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_atlas_packing_bind_group"),
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
            label: Some("prism_volumetric_atlas_packing_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_atlas_packing_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_atlas_packing_pass"),
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
            .map(|(q, r)| decode_result(q, r))
            .collect()
    }
}
