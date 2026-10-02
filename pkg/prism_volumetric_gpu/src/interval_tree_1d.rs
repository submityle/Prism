//! `wgpu` compute twin of the augmented one-dimensional (`1D`) interval tree
//! ([`interval_tree_1d`](prism_render_architecture::particle::interval_tree_1d)).
//!
//! A particle engine keeps a *collection* of `1D` intervals — lifetime windows,
//! emitter bursts, one axis of many spans — and repeatedly asks spatial queries
//! against the whole set: how many stored intervals are stabbed by a frame
//! coordinate `p`, how many overlap a scheduled window `[lo, hi]`. The `CPU`
//! golden [`IntervalTree`](prism_render_architecture::particle::interval_tree_1d::IntervalTree)
//! answers each query in `O(log n + k)` by walking a balanced, `max`-endpoint
//! augmented binary-search tree (`BST`); [`GpuIntervalTree1d`] is the on-device
//! twin that answers a whole *batch* of those queries at once, one thread per
//! query.
//!
//! # What is twinned (counts only)
//!
//! The golden exposes both index-reporting queries
//! ([`query_point`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::query_point),
//! [`query_overlap`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::query_overlap))
//! and a stabbing count
//! ([`count_point`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::count_point)).
//! Only the *counts* are ported to the device:
//!
//! - [`GpuIntervalTree1d::count_point`] reproduces the golden `count_point`
//!   exactly: the number of stored intervals `[lo, hi]` with `lo <= p <= hi`.
//! - [`GpuIntervalTree1d::count_overlap`] reproduces the golden
//!   `query_overlap(q).len()` exactly: the number of stored intervals that share
//!   at least one point with the closed query interval `[qlo, qhi]`.
//!
//! The index-set variants `query_point` / `query_overlap` return a `Vec<usize>`
//! whose length is dynamic and whose element order is unspecified; collecting
//! those variable-length, order-dependent index lists on the device would need a
//! second compaction pass, so they stay host-side on the golden reference. This
//! twin answers only the two scalar counts, which are exact integers and compare
//! with `==`.
//!
//! # `i64` → `i32` subset
//!
//! The golden coordinates are `i64`; `WGSL` has no `i64` (only `i32`/`u32`/`f32`
//! /`bool`), so this twin covers the `i32` subset. Endpoints and query
//! coordinates must satisfy `|lo|, |hi| <= `[`COORD_LIMIT`] (one million), which
//! keeps every `max_hi` comparison and every stab/overlap test inside `i32`
//! range with no overflow. The stored set is capped at [`MAX_INTERVALS`] and a
//! batch at [`MAX_RESULTS`] queries. Within those bounds the integer arithmetic
//! is identical to the golden `i64` path, so the counts match bit for bit.
//!
//! # Build on the host, walk on the device
//!
//! The balanced, `max_hi`-augmented tree is built once on the host with the same
//! median-split-over-low-endpoint layout the golden uses, then flattened to a
//! node pool (`lo`, `hi`, `max_hi`, `left`, `right`, all `i32`, with `-1` as the
//! `NIL` child sentinel) and uploaded. Each `GPU` thread walks that pool with an
//! explicit stack instead of recursion — `WGSL` forbids recursive functions —
//! applying the same `max_hi` subtree pruning the golden recursion does, and
//! writes one `u32` count. Because counting is order-independent, the iterative
//! walk visits exactly the same node set the golden traversal does and arrives
//! at the same total.
//!
//! # Degenerate inputs
//!
//! An empty query batch short-circuits on the host with no dispatch (a storage
//! buffer cannot be zero-sized). An empty interval set is uploaded as a single
//! placeholder node with the tree root pinned to `-1`, so every thread's walk
//! starts at `NIL` and returns `0`, matching the golden empty tree.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — signed/unsigned
//! comparison, `+ - /`, index addressing and a fixed-size stack array — with no
//! transcendental call, no intrinsic, no optional device feature and no `i64`,
//! so they run unmodified on `Metal`, `Vulkan` and `DX12`. The walk is bounded
//! by the fixed stack capacity and the strictly shrinking stack, so it provably
//! terminates with no runaway loop.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::interval_tree_1d`；无第三方引擎源码或衍生代码。
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
/// that divides evenly on every target backend.
const WORKGROUP_SIZE: u32 = 64;

/// Inclusive bound on the magnitude of every endpoint and query coordinate.
///
/// Endpoints and query coordinates must satisfy `|lo|, |hi| <= COORD_LIMIT`.
/// One million leaves every `max_hi` comparison and every stab/overlap test far
/// inside `i32` range, so the twin never overflows where the golden `i64` path
/// cannot.
pub const COORD_LIMIT: i32 = 1_000_000;

/// Upper bound on the number of stored intervals in one tree.
pub const MAX_INTERVALS: usize = 4096;

/// Upper bound on the number of queries in one batch.
pub const MAX_RESULTS: usize = 4096;

/// Host handle meaning "no child" / "empty subtree" in the flattened node pool.
///
/// Node links are `i32` indices into the uploaded node array; this `-1` marks
/// the absence of a child, mirroring the golden `usize::MAX` `NIL` sentinel.
const NIL: i32 = -1;

/// A `wgpu` compute twin of the augmented `1D` interval tree, twinning the `CPU`
/// golden [`IntervalTree`](prism_render_architecture::particle::interval_tree_1d::IntervalTree)
/// stabbing and overlap *counts*.
///
/// Build it once with [`GpuIntervalTree1d::new`]; then call
/// [`GpuIntervalTree1d::count_point`] or [`GpuIntervalTree1d::count_overlap`]
/// with a slice of intervals and a batch of queries. The tree is rebuilt and
/// uploaded per call, so the same instance serves any interval set.
pub struct GpuIntervalTree1d {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_count_point: ComputePipeline,
    pipeline_count_overlap: ComputePipeline,
}

/// One batched query against the stored interval set.
///
/// Carries a closed coordinate pair `[lo, hi]` in canonical `lo <= hi` form.
/// [`GpuIntervalTree1d::count_overlap`] treats it as the overlap interval
/// `[lo, hi]`; [`GpuIntervalTree1d::count_point`] treats it as the stabbing
/// point `lo` and ignores `hi`. Use [`IntervalTree1dQuery::point`] for a stab
/// and [`IntervalTree1dQuery::interval`] for an overlap window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntervalTree1dQuery {
    /// Lower (inclusive) query coordinate, or the stab point for `count_point`.
    pub lo: i32,
    /// Upper (inclusive) query coordinate.
    pub hi: i32,
}

impl IntervalTree1dQuery {
    /// Builds a point (stabbing) query at coordinate `p`.
    ///
    /// The result has `lo == hi == p`, so it is also a degenerate overlap
    /// window `[p, p]`.
    #[must_use]
    pub fn point(p: i32) -> IntervalTree1dQuery {
        IntervalTree1dQuery { lo: p, hi: p }
    }

    /// Builds a closed overlap query `[lo, hi]`, swapping reversed endpoints.
    ///
    /// The result is always in canonical `lo <= hi` form, matching the golden
    /// [`Interval::new`](prism_render_architecture::particle::interval_tree_1d::Interval::new).
    #[must_use]
    pub fn interval(lo: i32, hi: i32) -> IntervalTree1dQuery {
        if lo > hi {
            IntervalTree1dQuery { lo: hi, hi: lo }
        } else {
            IntervalTree1dQuery { lo, hi }
        }
    }
}

/// One batched query answer: the number of stored intervals matched.
///
/// For [`GpuIntervalTree1d::count_point`] this equals the golden
/// [`count_point`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::count_point);
/// for [`GpuIntervalTree1d::count_overlap`] it equals the golden
/// [`query_overlap`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::query_overlap)
/// result length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntervalTree1dResult {
    /// Number of stored intervals matched by this query.
    pub count: u32,
}

/// Twin compute shader: one thread per query over an uploaded augmented tree.
///
/// Two entry points mirror the two golden count primitives; both share one
/// bind-group layout and the same iterative traversal helper.
const INTERVAL_TREE_1D_WGSL: &str = r#"
// Interval-tree 1D twin: one thread per query. Two entry points mirror the CPU
// golden `particle::interval_tree_1d` counts: `count_point` reproduces
// `count_point` (stabbing count) and `count_overlap` reproduces
// `query_overlap(..).len()` (interval-overlap count). The balanced, max_hi
// augmented tree is built on the host and uploaded as a flat node pool
// (lo, hi, max_hi, left, right), all i32, with -1 as the NIL child. Each thread
// walks it with an explicit stack (no recursion) and writes one u32 count. Pure
// i32 compares and index arithmetic: no transcendental, no intrinsic, no i64,
// portable on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::interval_tree_1d；无第三方
// 引擎源码或衍生代码。

const STACK_CAP: u32 = 64u;

struct Params {
    // Number of valid nodes in the pool (0 means an empty tree with a single
    // placeholder node that is never read because `root` is -1).
    node_count: u32,
    // Number of valid queries; threads past this short-circuit.
    query_count: u32,
    // Handle of the root node, or -1 for an empty tree.
    root: i32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
}

struct Node {
    lo: i32,
    hi: i32,
    max_hi: i32,
    left: i32,
    right: i32,
}

struct Query {
    lo: i32,
    hi: i32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

// Iterative stab/overlap count over the augmented interval tree, mirroring the
// golden `collect_overlap` / `count_stab` recursion with an explicit stack.
// Counts every stored interval [lo, hi] that overlaps the closed query
// [qlo, qhi]; a point stab is the degenerate qlo == qhi case. The max_hi test
// prunes a whole subtree when every interval in it ends before qlo, and the
// right child is pushed only when the node's lo does not already exceed qhi,
// exactly as the golden traversal decides.
fn count_overlap_range(qlo: i32, qhi: i32) -> u32 {
    if (params.root < 0) {
        return 0u;
    }
    var stack: array<i32, STACK_CAP>;
    var sp = 0u;
    stack[sp] = params.root;
    sp = sp + 1u;
    var count = 0u;
    while (sp > 0u) {
        sp = sp - 1u;
        let handle = stack[sp];
        if (handle < 0) {
            continue;
        }
        let n = nodes[u32(handle)];
        if (n.max_hi < qlo) {
            continue;
        }
        if (n.lo <= qhi && n.hi >= qlo) {
            count = count + 1u;
        }
        if (n.left >= 0 && sp < STACK_CAP) {
            stack[sp] = n.left;
            sp = sp + 1u;
        }
        if (n.lo <= qhi && n.right >= 0 && sp < STACK_CAP) {
            stack[sp] = n.right;
            sp = sp + 1u;
        }
    }
    return count;
}

@compute @workgroup_size(64)
fn count_point(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let p = queries[idx].lo;
    dst[idx] = count_overlap_range(p, p);
}

@compute @workgroup_size(64)
fn count_overlap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    let q = queries[idx];
    dst[idx] = count_overlap_range(q.lo, q.hi);
}
"#;

/// Uniform parameters for one dispatch: node count, query count and the root
/// handle, padded to a `16`-byte, `std140`-aligned struct matching `Params` in
/// [`INTERVAL_TREE_1D_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid nodes in the pool.
    node_count: u32,
    /// Number of valid queries in the input and output buffers.
    query_count: u32,
    /// Handle of the root node, or `-1` for an empty tree.
    root: i32,
    /// Padding word.
    pad0: u32,
}

/// One flattened tree node, matching the `WGSL` `Node` struct field for field.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNode {
    /// Low endpoint of this node's interval (the `BST` key).
    lo: i32,
    /// High endpoint of this node's interval.
    hi: i32,
    /// Maximum `hi` over the entire subtree rooted at this node.
    max_hi: i32,
    /// Left child handle, or [`NIL`].
    left: i32,
    /// Right child handle, or [`NIL`].
    right: i32,
}

/// One query packed for the `WGSL` `Query` struct: a closed `i32` pair.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Lower (inclusive) query coordinate.
    lo: i32,
    /// Upper (inclusive) query coordinate.
    hi: i32,
}

impl GpuIntervalTree1d {
    /// Compiles the two interval-tree count kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuIntervalTree1d {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_interval_tree_1d"),
            source: ShaderSource::Wgsl(INTERVAL_TREE_1D_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_interval_tree_1d_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_interval_tree_1d_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline_count_point = make(
            "count_point",
            "prism_volumetric_interval_tree_1d_count_point_pipeline",
        );
        let pipeline_count_overlap = make(
            "count_overlap",
            "prism_volumetric_interval_tree_1d_count_overlap_pipeline",
        );
        GpuIntervalTree1d {
            module,
            layout,
            pipeline_count_point,
            pipeline_count_overlap,
        }
    }

    /// Returns, for each query, the number of stored intervals that contain the
    /// stab point `query.lo`, mirroring the golden
    /// [`count_point`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::count_point).
    ///
    /// Each query is treated as the point `lo` (its `hi` is ignored); build them
    /// with [`IntervalTree1dQuery::point`]. Intervals are closed `[lo, hi]`
    /// pairs and are canonicalized to `lo <= hi` on the host. Returns one result
    /// per query, in order. An empty query batch returns an empty vector with no
    /// dispatch issued; an empty interval set yields all-zero counts.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the interval set exceeds [`MAX_INTERVALS`],
    /// the batch exceeds [`MAX_RESULTS`], or any coordinate exceeds
    /// [`COORD_LIMIT`] in magnitude.
    #[must_use]
    pub fn count_point(
        &self,
        ctx: &GpuContext,
        intervals: &[(i32, i32)],
        queries: &[IntervalTree1dQuery],
    ) -> Vec<IntervalTree1dResult> {
        self.dispatch(ctx, &self.pipeline_count_point, intervals, queries)
    }

    /// Returns, for each query, the number of stored intervals that overlap the
    /// closed query interval `[lo, hi]`, mirroring the golden
    /// [`query_overlap`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::query_overlap)
    /// result length.
    ///
    /// Two closed intervals overlap when they share at least one point,
    /// including a single touching endpoint. Build queries with
    /// [`IntervalTree1dQuery::interval`]. Intervals are closed `[lo, hi]` pairs
    /// and are canonicalized to `lo <= hi` on the host. Returns one result per
    /// query, in order. An empty query batch returns an empty vector with no
    /// dispatch issued; an empty interval set yields all-zero counts.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when the interval set exceeds [`MAX_INTERVALS`],
    /// the batch exceeds [`MAX_RESULTS`], or any coordinate exceeds
    /// [`COORD_LIMIT`] in magnitude.
    #[must_use]
    pub fn count_overlap(
        &self,
        ctx: &GpuContext,
        intervals: &[(i32, i32)],
        queries: &[IntervalTree1dQuery],
    ) -> Vec<IntervalTree1dResult> {
        self.dispatch(ctx, &self.pipeline_count_overlap, intervals, queries)
    }

    /// Builds the augmented tree on the host, uploads it with the query batch,
    /// runs `pipeline` one thread per query and reads the `u32` counts back.
    ///
    /// Empty query batches short-circuit without a dispatch because a storage
    /// buffer cannot be zero-sized; an empty interval set is uploaded as a single
    /// placeholder node with the root pinned to [`NIL`] so every walk returns
    /// `0`.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        intervals: &[(i32, i32)],
        queries: &[IntervalTree1dQuery],
    ) -> Vec<IntervalTree1dResult> {
        debug_assert!(
            intervals.len() <= MAX_INTERVALS,
            "interval count exceeds MAX_INTERVALS"
        );
        debug_assert!(
            queries.len() <= MAX_RESULTS,
            "query count exceeds MAX_RESULTS"
        );
        debug_assert!(
            intervals
                .iter()
                .all(|&(a, b)| a.abs() <= COORD_LIMIT && b.abs() <= COORD_LIMIT),
            "interval coordinate exceeds COORD_LIMIT"
        );
        debug_assert!(
            queries
                .iter()
                .all(|q| q.lo.abs() <= COORD_LIMIT && q.hi.abs() <= COORD_LIMIT),
            "query coordinate exceeds COORD_LIMIT"
        );
        let query_count = queries.len();
        if query_count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let (nodes, root) = build_nodes(intervals);
        let params = GpuParams {
            node_count: nodes.len() as u32,
            query_count: query_count as u32,
            root,
            pad0: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_interval_tree_1d_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        // A storage buffer cannot be zero-sized; for an empty tree upload one
        // placeholder node the kernel never reads because `root` is -1.
        let placeholder = [GpuNode {
            lo: 0,
            hi: 0,
            max_hi: 0,
            left: NIL,
            right: NIL,
        }];
        let node_contents: &[GpuNode] = if nodes.is_empty() {
            &placeholder
        } else {
            &nodes
        };
        let nodes_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_interval_tree_1d_nodes"),
            contents: bytemuck::cast_slice(node_contents),
            usage: BufferUsages::STORAGE,
        });
        let packed: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery { lo: q.lo, hi: q.hi })
            .collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_interval_tree_1d_queries"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (query_count * size_of::<u32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_interval_tree_1d_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_interval_tree_1d_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: nodes_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_interval_tree_1d_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_interval_tree_1d_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_interval_tree_1d_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
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
        let counts = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();
        counts
            .into_iter()
            .map(|count| IntervalTree1dResult { count })
            .collect()
    }
}

/// Builds the flattened, `max_hi`-augmented node pool and returns it with the
/// root handle (or [`NIL`] for an empty set).
///
/// The intervals are canonicalized to `lo <= hi`, sorted by low endpoint (ties
/// broken by high endpoint then original index for a deterministic layout) and
/// assembled by recursive median split, so the height is `O(log n)` — the exact
/// layout the golden
/// [`build`](prism_render_architecture::particle::interval_tree_1d::IntervalTree::build)
/// produces.
fn build_nodes(intervals: &[(i32, i32)]) -> (Vec<GpuNode>, i32) {
    let canon: Vec<(i32, i32)> = intervals
        .iter()
        .map(|&(a, b)| if a > b { (b, a) } else { (a, b) })
        .collect();
    let mut order: Vec<usize> = (0..canon.len()).collect();
    order.sort_by(|&a, &b| {
        canon[a]
            .0
            .cmp(&canon[b].0)
            .then(canon[a].1.cmp(&canon[b].1))
            .then(a.cmp(&b))
    });
    let mut nodes: Vec<GpuNode> = Vec::with_capacity(canon.len());
    let root = build_subtree(&canon, &order, 0, order.len(), &mut nodes);
    (nodes, root)
}

/// Recursively builds a balanced subtree over `order[start..end]`, filling in
/// each node's `max_hi` augmentation on the way back up and returning the
/// subtree root handle (or [`NIL`] when the range is empty).
fn build_subtree(
    canon: &[(i32, i32)],
    order: &[usize],
    start: usize,
    end: usize,
    nodes: &mut Vec<GpuNode>,
) -> i32 {
    if start >= end {
        return NIL;
    }
    let mid = start + (end - start) / 2;
    let (lo, hi) = canon[order[mid]];
    let pos = nodes.len() as i32;
    nodes.push(GpuNode {
        lo,
        hi,
        max_hi: hi,
        left: NIL,
        right: NIL,
    });
    let left = build_subtree(canon, order, start, mid, nodes);
    let right = build_subtree(canon, order, mid + 1, end, nodes);
    let mut m = hi;
    if left != NIL {
        m = m.max(nodes[left as usize].max_hi);
    }
    if right != NIL {
        m = m.max(nodes[right as usize].max_hi);
    }
    nodes[pos as usize].left = left;
    nodes[pos as usize].right = right;
    nodes[pos as usize].max_hi = m;
    pos
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
