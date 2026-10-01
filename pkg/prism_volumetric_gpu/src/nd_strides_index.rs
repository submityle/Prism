//! `wgpu` compute twin of the generic `N-D` strides-index integer contract
//! ([`nd_strides_index`](prism_render_architecture::particle::nd_strides_index),
//! design §27 buffer addressing).
//!
//! A dense multi-dimensional array of a given `shape` is stored in one flat
//! buffer. The *stride* of a dimension is how many flat elements you skip to
//! advance that coordinate by one; the flat offset of a coordinate tuple is the
//! dot product of the coordinates with the strides, and the inverse recovers
//! the coordinates by repeated integer division and remainder — no floating
//! point anywhere. For `row-major` (`C` order) the last dimension is contiguous
//! (stride `1`); for `col-major` (`Fortran` order) the first dimension is
//! contiguous. For `shape [2, 3, 4]` the `row-major` strides are `[12, 4, 1]`
//! and the `col-major` strides are `[1, 2, 6]`.
//!
//! The `CPU` golden
//! [`nd_strides_index`](prism_render_architecture::particle::nd_strides_index)
//! owns that algebra over variable-length `&[usize]` slices. [`GpuNdStridesIndex`]
//! is the on-device twin that runs **one thread per query** over a fixed
//! [`MAX_RANK`] array layout and reproduces the same `u32`-domain arithmetic
//! element for element, so a passing real-device parity test is direct evidence
//! the ported kernels compute the identical offsets, not merely that the
//! shaders compile.
//!
//! # What is twinned
//!
//! Six per-query kernels mirror the six golden functions:
//!
//! * [`GpuNdStridesIndex::row_major_strides`] and
//!   [`GpuNdStridesIndex::col_major_strides`] mirror
//!   [`row_major_strides`](prism_render_architecture::particle::nd_strides_index::row_major_strides)
//!   and
//!   [`col_major_strides`](prism_render_architecture::particle::nd_strides_index::col_major_strides).
//! * [`GpuNdStridesIndex::linear_from_coords`] mirrors
//!   [`linear_from_coords`](prism_render_architecture::particle::nd_strides_index::linear_from_coords),
//!   the coordinate-stride dot product bounded by the shorter length.
//! * [`GpuNdStridesIndex::coords_from_linear`] mirrors
//!   [`coords_from_linear`](prism_render_architecture::particle::nd_strides_index::coords_from_linear),
//!   the division-remainder inverse for either major order.
//! * [`GpuNdStridesIndex::total_elements`] mirrors
//!   [`total_elements`](prism_render_architecture::particle::nd_strides_index::total_elements),
//!   the product of the extents (`1` for the empty scalar shape).
//! * [`GpuNdStridesIndex::coords_in_bounds`] mirrors
//!   [`coords_in_bounds`](prism_render_architecture::particle::nd_strides_index::coords_in_bounds),
//!   the rank-match plus per-axis range check.
//!
//! # Rank and overflow
//!
//! The golden uses variable-length `usize` slices; the twin uses a fixed
//! [`MAX_RANK`] of `8` and carries the live rank as a `u32`, so entries past the
//! rank are ignored. All device arithmetic is `u32`. The golden runs on a
//! `64`-bit host, but the twinned problem sizes stay far below `2^32`, so the
//! `usize`-to-`u32` conversion is value-preserving for every twinned case; the
//! host builders clamp each slice length to [`MAX_RANK`] rather than overflow a
//! fixed array. The golden's own overflow note (large products exceeding the
//! element-count bound) is a host-side caller contract and is not exercised by
//! the twin.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — the arithmetic
//! `+ - * /`, the remainder `%`, `min`, and unsigned comparisons — with no
//! transcendental call, no `smoothstep`, no optional device feature and no
//! `u64`. Every loop runs under the constant [`MAX_RANK`] bound so it unrolls
//! statically on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Every transform is pure unsigned integer arithmetic with no rounding
//! anywhere, so `CPU` and `GPU` compute identical values. The parity test
//! therefore asserts an exact `==` on every stride, coordinate, offset, element
//! count and bounds flag, with no tolerance: any mismatch is a genuine port bug.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::nd_strides_index`；无第三方引擎源码或衍生代码。
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
/// shared by the other twins in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// Largest rank the kernels accept. Every `shape`, `coords` and `strides` tuple
/// is carried in a fixed `u32` array of this length, and every device loop
/// iterates at most this many times so it unrolls statically. `8` comfortably
/// covers the volume, grid and tensor shapes this twin addresses.
pub const MAX_RANK: usize = 8;

/// The six per-query `N-D` strides kernels, mirroring the `CPU` golden
/// [`nd_strides_index`](prism_render_architecture::particle::nd_strides_index)
/// step for step. One source file hosts all entry points sharing a single
/// bind-group layout (uniform params, read-only query array, read-write result
/// array).
const ND_STRIDES_INDEX_WGSL: &str = r#"
// N-D strides-index twin: one thread per query. Every entry point reads one
// NdQuery and writes one NdResult. All arithmetic is u32; the fixed MAX_RANK
// array layout stands in for the golden's variable-length usize slices.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::nd_strides_index；
// 无第三方引擎源码或衍生代码。

const MAX_RANK: u32 = 8u;

struct Params {
    // Number of valid queries; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct NdQuery {
    // Extent of each dimension (first `rank` entries meaningful).
    shape: array<u32, 8>,
    // Coordinate tuple (first `coords_rank` entries meaningful).
    coords: array<u32, 8>,
    // Precomputed strides for the linear_from_coords dot product.
    strides: array<u32, 8>,
    // Length of `shape`/`strides`.
    rank: u32,
    // Length of `coords` (may differ from `rank` for the bounds check).
    coords_rank: u32,
    // Flat offset for the inverse mapping.
    linear: u32,
    // Non-zero selects row-major (C) order; zero selects col-major (Fortran).
    row_major: u32,
}

struct NdResult {
    // Strides output (row_major_strides / col_major_strides).
    strides: array<u32, 8>,
    // Coordinate output (coords_from_linear).
    coords: array<u32, 8>,
    // Flat offset output (linear_from_coords).
    linear: u32,
    // Element-count output (total_elements).
    total: u32,
    // Bounds flag output (coords_in_bounds): 1 valid, 0 invalid.
    in_bounds: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<NdQuery>;
@group(0) @binding(2) var<storage, read_write> results: array<NdResult>;

// Strides for `shape` in either major order, selected by `row_major`. Mirrors
// `row_major_strides` / `col_major_strides`: the contiguous dimension gets
// stride 1 and each neighbor is the running product of the already-placed
// extents. The empty shape yields all-zero strides (none are meaningful).
fn compute_strides(q: NdQuery) -> array<u32, 8> {
    let n = min(q.rank, MAX_RANK);
    var out = array<u32, 8>();
    if (n == 0u) {
        return out;
    }
    if (q.row_major != 0u) {
        out[n - 1u] = 1u;
        var i = n - 1u;
        while (i > 0u) {
            i = i - 1u;
            out[i] = out[i + 1u] * q.shape[i + 1u];
        }
    } else {
        out[0] = 1u;
        var i = 1u;
        while (i < n) {
            out[i] = out[i - 1u] * q.shape[i - 1u];
            i = i + 1u;
        }
    }
    return out;
}

@compute @workgroup_size(64)
fn strides(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    results[idx].strides = compute_strides(queries[idx]);
}

// Flat offset = dot product of coords with strides, bounded by the shorter of
// the two lengths. Mirrors `linear_from_coords`, whose `zip` stops at the
// shorter slice. Empty inputs yield 0.
@compute @workgroup_size(64)
fn linear_from_coords(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let n = min(min(q.coords_rank, q.rank), MAX_RANK);
    var sum = 0u;
    for (var i = 0u; i < MAX_RANK; i = i + 1u) {
        if (i >= n) {
            break;
        }
        sum = sum + q.coords[i] * q.strides[i];
    }
    results[idx].linear = sum;
}

// Inverse mapping: rebuild coords from a flat offset by repeated remainder and
// division, walking the dimensions fastest-varying first. Mirrors
// `coords_from_linear`: row-major walks the last dimension first, col-major the
// first dimension first. Every extent must be non-zero (caller contract).
@compute @workgroup_size(64)
fn coords_from_linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let n = min(q.rank, MAX_RANK);
    var coords = array<u32, 8>();
    var rem = q.linear;
    if (q.row_major != 0u) {
        var i = n;
        while (i > 0u) {
            i = i - 1u;
            coords[i] = rem % q.shape[i];
            rem = rem / q.shape[i];
        }
    } else {
        var i = 0u;
        while (i < n) {
            coords[i] = rem % q.shape[i];
            rem = rem / q.shape[i];
            i = i + 1u;
        }
    }
    results[idx].coords = coords;
}

// Product of the extents. Mirrors `total_elements`: the empty shape yields 1
// (the scalar), and any zero extent yields 0.
@compute @workgroup_size(64)
fn total_elements(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let n = min(q.rank, MAX_RANK);
    var prod = 1u;
    for (var i = 0u; i < MAX_RANK; i = i + 1u) {
        if (i >= n) {
            break;
        }
        prod = prod * q.shape[i];
    }
    results[idx].total = prod;
}

// Rank-match plus per-axis range check. Mirrors `coords_in_bounds`: the ranks
// must be equal and every coordinate strictly below its extent. The empty
// coords/empty shape pair is the valid scalar address.
@compute @workgroup_size(64)
fn coords_in_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var ok = 1u;
    if (q.coords_rank != q.rank) {
        ok = 0u;
    } else {
        let n = min(q.rank, MAX_RANK);
        for (var i = 0u; i < MAX_RANK; i = i + 1u) {
            if (i >= n) {
                break;
            }
            if (q.coords[i] >= q.shape[i]) {
                ok = 0u;
            }
        }
    }
    results[idx].in_bounds = ok;
}
"#;

/// Uniform parameters for one dispatch: the query `count` plus three pad words
/// so the struct is a `16`-byte, `16`-byte-aligned uniform. Layout matches
/// `Params` in [`ND_STRIDES_INDEX_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One device query. Fixed `MAX_RANK` arrays stand in for the golden's
/// variable-length slices; layout matches `NdQuery` in [`ND_STRIDES_INDEX_WGSL`]
/// (all `u32`, so the natural `repr(C)` layout and the `WGSL` storage layout
/// agree).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    shape: [u32; MAX_RANK],
    coords: [u32; MAX_RANK],
    strides: [u32; MAX_RANK],
    rank: u32,
    coords_rank: u32,
    linear: u32,
    row_major: u32,
}

/// One device result. Layout matches `NdResult` in [`ND_STRIDES_INDEX_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    strides: [u32; MAX_RANK],
    coords: [u32; MAX_RANK],
    linear: u32,
    total: u32,
    in_bounds: u32,
    pad: u32,
}

/// A host-built query describing one `N-D` addressing problem.
///
/// The fields are fixed `MAX_RANK` arrays with explicit live ranks, standing in
/// for the golden's variable-length `&[usize]` slices. Build one with the
/// constructor matching the function you want to twin
/// ([`NdStridesQuery::from_shape`], [`NdStridesQuery::from_coords_strides`],
/// [`NdStridesQuery::from_linear_shape`], [`NdStridesQuery::from_coords_shape`]);
/// each fills only the fields its kernel reads and leaves the rest zero.
#[derive(Clone, Copy, Debug)]
pub struct NdStridesQuery {
    /// Extent of each dimension; only the first [`NdStridesQuery::rank`] entries
    /// are meaningful.
    pub shape: [u32; MAX_RANK],
    /// Coordinate tuple; only the first [`NdStridesQuery::coords_rank`] entries
    /// are meaningful.
    pub coords: [u32; MAX_RANK],
    /// Precomputed strides for [`GpuNdStridesIndex::linear_from_coords`]; only
    /// the first [`NdStridesQuery::rank`] entries are meaningful.
    pub strides: [u32; MAX_RANK],
    /// Length of `shape` and `strides`.
    pub rank: u32,
    /// Length of `coords` (may differ from `rank` for the bounds check).
    pub coords_rank: u32,
    /// Flat offset for [`GpuNdStridesIndex::coords_from_linear`].
    pub linear: u32,
    /// `true` selects `row-major` (`C`) order; `false` selects `col-major`
    /// (`Fortran`) order for [`GpuNdStridesIndex::coords_from_linear`].
    pub row_major: bool,
}

impl NdStridesQuery {
    /// A fully zeroed query: empty rank, all arrays zero, `row-major`.
    #[must_use]
    pub fn zeroed() -> NdStridesQuery {
        NdStridesQuery {
            shape: [0; MAX_RANK],
            coords: [0; MAX_RANK],
            strides: [0; MAX_RANK],
            rank: 0,
            coords_rank: 0,
            linear: 0,
            row_major: true,
        }
    }

    /// Builds a query carrying only `shape`, for the strides and
    /// [`GpuNdStridesIndex::total_elements`] kernels.
    ///
    /// # Panics
    ///
    /// Panics if `shape` is longer than [`MAX_RANK`].
    #[must_use]
    pub fn from_shape(shape: &[u32]) -> NdStridesQuery {
        assert!(shape.len() <= MAX_RANK, "shape rank exceeds MAX_RANK");
        let mut q = NdStridesQuery::zeroed();
        q.shape[..shape.len()].copy_from_slice(shape);
        q.rank = shape.len() as u32;
        q.coords_rank = q.rank;
        q
    }

    /// Builds a query carrying `coords` and `strides`, for
    /// [`GpuNdStridesIndex::linear_from_coords`]. The kernel bounds the dot
    /// product by the shorter of the two lengths, exactly as the golden's `zip`.
    ///
    /// # Panics
    ///
    /// Panics if `coords` or `strides` is longer than [`MAX_RANK`].
    #[must_use]
    pub fn from_coords_strides(coords: &[u32], strides: &[u32]) -> NdStridesQuery {
        assert!(coords.len() <= MAX_RANK, "coords rank exceeds MAX_RANK");
        assert!(strides.len() <= MAX_RANK, "strides rank exceeds MAX_RANK");
        let mut q = NdStridesQuery::zeroed();
        q.coords[..coords.len()].copy_from_slice(coords);
        q.strides[..strides.len()].copy_from_slice(strides);
        q.rank = strides.len() as u32;
        q.coords_rank = coords.len() as u32;
        q
    }

    /// Builds a query carrying `linear`, `shape` and the major order, for
    /// [`GpuNdStridesIndex::coords_from_linear`].
    ///
    /// # Panics
    ///
    /// Panics if `shape` is longer than [`MAX_RANK`].
    #[must_use]
    pub fn from_linear_shape(linear: u32, shape: &[u32], row_major: bool) -> NdStridesQuery {
        let mut q = NdStridesQuery::from_shape(shape);
        q.linear = linear;
        q.row_major = row_major;
        q
    }

    /// Builds a query carrying `coords` and `shape`, for
    /// [`GpuNdStridesIndex::coords_in_bounds`]. The two lengths are kept
    /// independent so a rank mismatch reports out of bounds, as the golden does.
    ///
    /// # Panics
    ///
    /// Panics if `coords` or `shape` is longer than [`MAX_RANK`].
    #[must_use]
    pub fn from_coords_shape(coords: &[u32], shape: &[u32]) -> NdStridesQuery {
        assert!(coords.len() <= MAX_RANK, "coords rank exceeds MAX_RANK");
        let mut q = NdStridesQuery::from_shape(shape);
        q.coords[..coords.len()].copy_from_slice(coords);
        q.coords_rank = coords.len() as u32;
        q
    }

    /// Lowers this host query to its device form, forcing `row_major` to
    /// `force_row_major` when `Some` (used by the two strides entry points whose
    /// method name is authoritative) or honoring the stored flag when `None`.
    fn to_gpu(self, force_row_major: Option<bool>) -> GpuQuery {
        let row_major = force_row_major.unwrap_or(self.row_major);
        GpuQuery {
            shape: self.shape,
            coords: self.coords,
            strides: self.strides,
            rank: self.rank,
            coords_rank: self.coords_rank,
            linear: self.linear,
            row_major: u32::from(row_major),
        }
    }
}

/// A compiled, reusable set of the six `N-D` strides-index kernels sharing one
/// bind-group layout.
pub struct GpuNdStridesIndex {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    strides_pipeline: ComputePipeline,
    linear_pipeline: ComputePipeline,
    coords_pipeline: ComputePipeline,
    total_pipeline: ComputePipeline,
    bounds_pipeline: ComputePipeline,
}

impl GpuNdStridesIndex {
    /// Compiles the six `N-D` strides-index kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuNdStridesIndex {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_nd_strides_index"),
            source: ShaderSource::Wgsl(ND_STRIDES_INDEX_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_nd_strides_index_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_nd_strides_index_pipeline_layout"),
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
        let strides_pipeline = make("strides", "prism_volumetric_nd_strides_index_strides");
        let linear_pipeline = make(
            "linear_from_coords",
            "prism_volumetric_nd_strides_index_linear",
        );
        let coords_pipeline = make(
            "coords_from_linear",
            "prism_volumetric_nd_strides_index_coords",
        );
        let total_pipeline = make("total_elements", "prism_volumetric_nd_strides_index_total");
        let bounds_pipeline = make(
            "coords_in_bounds",
            "prism_volumetric_nd_strides_index_bounds",
        );
        GpuNdStridesIndex {
            module,
            layout,
            strides_pipeline,
            linear_pipeline,
            coords_pipeline,
            total_pipeline,
            bounds_pipeline,
        }
    }

    /// Returns the `row-major` (`C` order) strides for each query's `shape`,
    /// mirroring
    /// [`row_major_strides`](prism_render_architecture::particle::nd_strides_index::row_major_strides).
    ///
    /// Each output vector is truncated to that query's `rank`, matching the
    /// golden's return length. An empty batch returns an empty vector with no
    /// dispatch issued (a storage buffer cannot be zero-sized).
    #[must_use]
    pub fn row_major_strides(&self, ctx: &GpuContext, queries: &[NdStridesQuery]) -> Vec<Vec<u32>> {
        let results = self.dispatch(ctx, &self.strides_pipeline, queries, Some(true));
        strides_outputs(&results, queries)
    }

    /// Returns the `col-major` (`Fortran` order) strides for each query's
    /// `shape`, mirroring
    /// [`col_major_strides`](prism_render_architecture::particle::nd_strides_index::col_major_strides).
    ///
    /// Each output vector is truncated to that query's `rank`. An empty batch
    /// returns an empty vector with no dispatch issued.
    #[must_use]
    pub fn col_major_strides(&self, ctx: &GpuContext, queries: &[NdStridesQuery]) -> Vec<Vec<u32>> {
        let results = self.dispatch(ctx, &self.strides_pipeline, queries, Some(false));
        strides_outputs(&results, queries)
    }

    /// Returns the flat offset `dot(coords, strides)` for each query, bounded by
    /// the shorter of the two lengths, mirroring
    /// [`linear_from_coords`](prism_render_architecture::particle::nd_strides_index::linear_from_coords).
    ///
    /// An empty batch returns an empty vector with no dispatch issued.
    #[must_use]
    pub fn linear_from_coords(&self, ctx: &GpuContext, queries: &[NdStridesQuery]) -> Vec<u32> {
        let results = self.dispatch(ctx, &self.linear_pipeline, queries, None);
        results.iter().map(|r| r.linear).collect()
    }

    /// Recovers the coordinate tuple for each query's flat `linear` offset in
    /// its `shape` and major order, mirroring
    /// [`coords_from_linear`](prism_render_architecture::particle::nd_strides_index::coords_from_linear).
    ///
    /// Each output vector is truncated to that query's `rank`. Every extent must
    /// be non-zero (caller contract). An empty batch returns an empty vector
    /// with no dispatch issued.
    #[must_use]
    pub fn coords_from_linear(
        &self,
        ctx: &GpuContext,
        queries: &[NdStridesQuery],
    ) -> Vec<Vec<u32>> {
        let results = self.dispatch(ctx, &self.coords_pipeline, queries, None);
        results
            .iter()
            .zip(queries)
            .map(|(r, q)| r.coords[..rank_usize(q)].to_vec())
            .collect()
    }

    /// Returns the element count (product of the extents) for each query's
    /// `shape`, mirroring
    /// [`total_elements`](prism_render_architecture::particle::nd_strides_index::total_elements).
    ///
    /// The empty shape yields `1`; any zero extent yields `0`. An empty batch
    /// returns an empty vector with no dispatch issued.
    #[must_use]
    pub fn total_elements(&self, ctx: &GpuContext, queries: &[NdStridesQuery]) -> Vec<u32> {
        let results = self.dispatch(ctx, &self.total_pipeline, queries, None);
        results.iter().map(|r| r.total).collect()
    }

    /// Returns whether each query's `coords` names a valid cell of its `shape`,
    /// mirroring
    /// [`coords_in_bounds`](prism_render_architecture::particle::nd_strides_index::coords_in_bounds).
    ///
    /// The ranks must match and every coordinate must be strictly below its
    /// extent. An empty batch returns an empty vector with no dispatch issued.
    #[must_use]
    pub fn coords_in_bounds(&self, ctx: &GpuContext, queries: &[NdStridesQuery]) -> Vec<bool> {
        let results = self.dispatch(ctx, &self.bounds_pipeline, queries, None);
        results.iter().map(|r| r.in_bounds != 0).collect()
    }

    /// Runs one entry point over the whole batch and reads the result array
    /// back. Returns an empty vector without issuing a dispatch when the batch
    /// is empty, since a storage buffer cannot be zero-sized.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        queries: &[NdStridesQuery],
        force_row_major: Option<bool>,
    ) -> Vec<GpuResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> =
            queries.iter().map(|q| q.to_gpu(force_row_major)).collect();
        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_nd_strides_index_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let query_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_nd_strides_index_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_nd_strides_index_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_nd_strides_index_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: query_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_nd_strides_index_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_nd_strides_index_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_nd_strides_index_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage_buf, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage_buf.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        read_mapped(&stage_buf)
    }
}

/// Truncates each query's device strides output to its live `rank`, matching the
/// golden's variable-length return.
fn strides_outputs(results: &[GpuResult], queries: &[NdStridesQuery]) -> Vec<Vec<u32>> {
    results
        .iter()
        .zip(queries)
        .map(|(r, q)| r.strides[..rank_usize(q)].to_vec())
        .collect()
}

/// The query's live `rank` as a `usize`, clamped to [`MAX_RANK`] for slicing.
fn rank_usize(q: &NdStridesQuery) -> usize {
    (q.rank as usize).min(MAX_RANK)
}

/// Reads a mapped staging buffer back into a [`GpuResult`] vector and unmaps it.
fn read_mapped(stage: &wgpu::Buffer) -> Vec<GpuResult> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let result = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
    drop(view);
    stage.unmap();
    result
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
