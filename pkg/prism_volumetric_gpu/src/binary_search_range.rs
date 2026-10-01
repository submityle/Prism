//! `wgpu` compute twin of the half-open `u32` binary-search range queries
//! ([`binary_search_range`](prism_render_architecture::particle::binary_search_range)).
//!
//! A `GPU`-driven `VFX` pipeline repeatedly consumes a sequence that an earlier
//! stage already produced in non-decreasing order — quantized view-depth keys,
//! sort-key buckets, spatial-hash cell ranges — and then asks positional
//! questions about it: "where would this key insert", "how many particles share
//! this exact key", "is this key present at all". The `CPU` golden
//! [`binary_search_range`](prism_render_architecture::particle::binary_search_range)
//! answers those questions serially in `O(log n)` comparisons; this module is
//! the on-device twin that answers a whole batch of them at once, one thread per
//! query.
//!
//! [`GpuBinarySearchRange`] holds one sorted `u32` array in a storage buffer and
//! a batch of query targets in another, and dispatches a one-dimensional grid of
//! threads where thread `i` binary-searches the shared array for query `i` and
//! writes a single `u32` insertion index. Two kernels mirror the two reference
//! primitives tap for tap:
//!
//! - [`GpuBinarySearchRange::lower_bound`] runs the golden
//!   [`lower_bound`](prism_render_architecture::particle::binary_search_range::lower_bound):
//!   it returns the index of the first element `>= target`, the left-most
//!   insertion point.
//! - [`GpuBinarySearchRange::upper_bound`] runs the golden
//!   [`upper_bound`](prism_render_architecture::particle::binary_search_range::upper_bound):
//!   it returns the index of the first element `> target`, the right-most
//!   insertion point.
//!
//! The two remaining reference primitives are pure index algebra over those two
//! results and are derived on the host without a third kernel:
//! [`GpuBinarySearchRange::equal_range`] pairs the two bounds into the half-open
//! interval `[lo, hi)` (empty exactly when `lo == hi`), mirroring the golden
//! [`equal_range`](prism_render_architecture::particle::binary_search_range::equal_range),
//! and [`GpuBinarySearchRange::contains`] reports `lo < hi`, mirroring the golden
//! [`contains`](prism_render_architecture::particle::binary_search_range::contains).
//!
//! # Correctness model
//!
//! Every search is pure integer logic: `u32` `<`/`<=` comparisons on the keys
//! and `u32` bookkeeping on the indices, with no element moved, no key divided,
//! and nothing cast to floating point. Each kernel reproduces the overflow-safe
//! half-open loop of the reference exactly — `lo = 0`, `hi = len`,
//! `while lo < hi { mid = lo + (hi - lo) / 2; ... }` — so the midpoint can never
//! overflow and the returned index is always a valid insertion point in
//! `0..=len`. Because there is no rounding anywhere, `CPU` and `GPU` compute
//! identical indices and the parity test asserts an exact `==` on every output.
//!
//! # Degenerate inputs
//!
//! The empty array and the all-equal array are valid inputs, handled without
//! special-casing in the kernel. An empty query batch short-circuits on the host
//! with no dispatch (a storage buffer cannot be zero-sized). An empty sorted
//! array is uploaded as a one-element placeholder with `data_len` pinned to `0`,
//! so the search loop never executes and every query returns `0`, matching the
//! reference. The caller's precondition — the array is sorted in non-decreasing
//! order — is assumed exactly as in the golden standard; when it does not hold
//! the results are unspecified but still memory safe (every index stays in
//! bounds).
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — unsigned comparison,
//! `+ - /`, and index addressing — with no transcendental call, no intrinsic, no
//! optional device feature and no `u64`. The search loop is bounded by its
//! strictly shrinking `[lo, hi)` interval (each iteration either raises `lo` or
//! lowers `hi`), so it provably terminates in `O(log n)` steps with no risk of a
//! runaway loop. The kernels therefore run unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::binary_search_range`；无第三方引擎源码或衍生代码。
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

/// The half-open `u32` binary-search kernels, mirroring the `CPU` golden
/// [`binary_search_range`](prism_render_architecture::particle::binary_search_range)
/// loop for loop. One source file hosts two entry points — `lower_bound` and
/// `upper_bound` — sharing one bind-group layout.
const BINARY_SEARCH_RANGE_WGSL: &str = r#"
// Binary-search range twin: one thread per query. Two entry points mirror the
// CPU golden `particle::binary_search_range` u32 primitives `lower_bound` and
// `upper_bound`. `data` is the shared sorted array; `queries` holds one target
// per thread; `dst` receives one insertion index per query. Pure u32 compares
// and index arithmetic: no transcendental, no intrinsic, no u64, portable on
// Metal, Vulkan and DX12. The search loop is bounded by its strictly shrinking
// [lo, hi) interval, so it terminates in O(log n) steps with no runaway loop.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::binary_search_range；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of valid elements in the shared sorted array; the search upper
    // bound. Zero means an empty array (the `data` buffer still holds one
    // placeholder element that is never read).
    data_len: u32,
    // Number of valid queries; threads past this short-circuit.
    query_count: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read> queries: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

// First index with `data[i] >= target`, or `data_len` when every element is
// `< target`. Overflow-safe midpoint `lo + (hi - lo) / 2`, matching the golden
// `lower_bound` exactly.
fn lower_bound_one(needle: u32) -> u32 {
    var lo = 0u;
    var hi = params.data_len;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        if (data[mid] < needle) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// First index with `data[i] > target`, or `data_len` when every element is
// `<= target`. Matches the golden `upper_bound` exactly; the only difference
// from `lower_bound_one` is the `<=` versus `<` comparison.
fn upper_bound_one(needle: u32) -> u32 {
    var lo = 0u;
    var hi = params.data_len;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        if (data[mid] <= needle) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(64)
fn lower_bound(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    dst[idx] = lower_bound_one(queries[idx]);
}

@compute @workgroup_size(64)
fn upper_bound(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }
    dst[idx] = upper_bound_one(queries[idx]);
}
"#;

/// Uniform parameters for one dispatch: the array length and query count plus
/// padding to a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BINARY_SEARCH_RANGE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid elements in the shared sorted array.
    data_len: u32,
    /// Number of valid queries in the input and output buffers.
    query_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// A compiled, reusable pair of half-open `u32` binary-search kernels
/// (`lower_bound` and `upper_bound`), twinning the `CPU` golden
/// [`binary_search_range`](prism_render_architecture::particle::binary_search_range).
pub struct GpuBinarySearchRange {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_lower_bound: ComputePipeline,
    pipeline_upper_bound: ComputePipeline,
}

impl GpuBinarySearchRange {
    /// Compiles the two binary-search kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBinarySearchRange {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_binary_search_range"),
            source: ShaderSource::Wgsl(BINARY_SEARCH_RANGE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_binary_search_range_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_binary_search_range_pipeline_layout"),
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
        let pipeline_lower_bound = make(
            "lower_bound",
            "prism_volumetric_binary_search_range_lower_bound_pipeline",
        );
        let pipeline_upper_bound = make(
            "upper_bound",
            "prism_volumetric_binary_search_range_upper_bound_pipeline",
        );
        GpuBinarySearchRange {
            module,
            layout,
            pipeline_lower_bound,
            pipeline_upper_bound,
        }
    }

    /// Returns, for each query, the index of the first element in the sorted
    /// `data` array that is `>= target`, mirroring the golden
    /// [`lower_bound`](prism_render_architecture::particle::binary_search_range::lower_bound).
    ///
    /// Returns one index per query, in order. An empty query batch returns an
    /// empty vector with no dispatch issued (a storage buffer cannot be
    /// zero-sized). An empty `data` array yields all-zero indices. Assumes
    /// `data` is sorted in non-decreasing order; the result is unspecified (but
    /// in-bounds) otherwise.
    #[must_use]
    pub fn lower_bound(&self, ctx: &GpuContext, data: &[u32], queries: &[u32]) -> Vec<u32> {
        self.dispatch(ctx, &self.pipeline_lower_bound, data, queries)
    }

    /// Returns, for each query, the index of the first element in the sorted
    /// `data` array that is `> target`, mirroring the golden
    /// [`upper_bound`](prism_render_architecture::particle::binary_search_range::upper_bound).
    ///
    /// Returns one index per query, in order. An empty query batch returns an
    /// empty vector with no dispatch issued. An empty `data` array yields
    /// all-zero indices. Assumes `data` is sorted in non-decreasing order; the
    /// result is unspecified (but in-bounds) otherwise.
    #[must_use]
    pub fn upper_bound(&self, ctx: &GpuContext, data: &[u32], queries: &[u32]) -> Vec<u32> {
        self.dispatch(ctx, &self.pipeline_upper_bound, data, queries)
    }

    /// Returns, for each query, the half-open interval `[lo, hi)` of elements
    /// equal to the target as the pair `(lower_bound, upper_bound)`, mirroring
    /// the golden
    /// [`equal_range`](prism_render_architecture::particle::binary_search_range::equal_range).
    ///
    /// The interval is empty exactly when `lo == hi`. This runs both kernels and
    /// pairs their outputs on the host; no third kernel is needed. Returns one
    /// pair per query, in order, and an empty vector for an empty query batch.
    #[must_use]
    pub fn equal_range(&self, ctx: &GpuContext, data: &[u32], queries: &[u32]) -> Vec<(u32, u32)> {
        let lo = self.lower_bound(ctx, data, queries);
        let hi = self.upper_bound(ctx, data, queries);
        lo.into_iter().zip(hi).collect()
    }

    /// Returns, for each query, whether the target occurs anywhere in the sorted
    /// `data` array, mirroring the golden
    /// [`contains`](prism_render_architecture::particle::binary_search_range::contains).
    ///
    /// A target is present exactly when its half-open equal range is non-empty,
    /// i.e. `lower_bound < upper_bound`, so this derives the answer from the two
    /// bound kernels on the host. Returns one flag per query, in order, and an
    /// empty vector for an empty query batch.
    #[must_use]
    pub fn contains(&self, ctx: &GpuContext, data: &[u32], queries: &[u32]) -> Vec<bool> {
        let lo = self.lower_bound(ctx, data, queries);
        let hi = self.upper_bound(ctx, data, queries);
        lo.into_iter().zip(hi).map(|(l, h)| l < h).collect()
    }

    /// Issues one `1-D` dispatch of `pipeline`, searching the shared sorted
    /// `data` array for every entry of `queries` and reading the `u32` indices
    /// back. Empty query batches short-circuit without a dispatch because a
    /// storage buffer cannot be zero-sized; an empty `data` array is uploaded as
    /// a single placeholder element with `data_len` pinned to `0` so the search
    /// loop never reads it.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        data: &[u32],
        queries: &[u32],
    ) -> Vec<u32> {
        let query_count = queries.len();
        if query_count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            data_len: data.len() as u32,
            query_count: query_count as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_binary_search_range_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        // A storage buffer cannot be zero-sized; for an empty array upload one
        // placeholder word that the kernel never reads because `data_len` is 0.
        let placeholder = [0u32];
        let data_contents: &[u32] = if data.is_empty() { &placeholder } else { data };
        let data_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_binary_search_range_data"),
            contents: bytemuck::cast_slice(data_contents),
            usage: BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_binary_search_range_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = size_of_val(queries) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_binary_search_range_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_binary_search_range_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: data_buf.as_entire_binding(),
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
            label: Some("prism_volumetric_binary_search_range_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_binary_search_range_encoder"),
        });
        {
            let groups = (query_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_binary_search_range_pass"),
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
        let result = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();
        result
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
