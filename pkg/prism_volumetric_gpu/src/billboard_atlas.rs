//! `wgpu` compute twin of the multi-view `impostor` `atlas` view-selection and
//! cell `UV` location contract
//! ([`billboard_atlas`](prism_render_architecture::particle::billboard_atlas),
//! particle design §16, §22).
//!
//! The `CPU` golden
//! [`billboard_atlas`](prism_render_architecture::particle::billboard_atlas)
//! maps a view direction onto a regular `N`x`N` `octahedral` grid of baked
//! sprites and locates each cell inside the `atlas`. [`GpuBillboardAtlas`] is the
//! on-device twin of the module's deterministic, per-query functions: the
//! `octahedral` direction encode and decode
//! ([`oct_encode`](prism_render_architecture::particle::billboard_atlas::oct_encode)
//! and
//! [`oct_decode`](prism_render_architecture::particle::billboard_atlas::oct_decode)),
//! the single-cell view selection
//! ([`ImpostorGrid::view_to_cell`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::view_to_cell)),
//! the integer grid arithmetic
//! ([`ImpostorGrid::cell_coord`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::cell_coord),
//! [`ImpostorGrid::cell_index`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::cell_index)
//! and
//! [`ImpostorGrid::cell_count`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::cell_count)),
//! and the normalized cell `UV` rectangle
//! ([`ImpostorGrid::cell_uv_rect`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::cell_uv_rect)).
//! One thread solves one query, so a passing real-device parity test is direct
//! evidence the ported kernel selects the same cell and reports the same `UV`
//! rectangle the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Every deterministic per-query answer the reference computes is reproduced for
//! a batch of independent queries: the `octahedral` encode of the query
//! direction, the decode of that encoding back to a unit direction, the single
//! nearest view cell for the direction (its flattened index, column and row),
//! the `(col, row)` decomposition of a supplied flattened cell index, the
//! flattened index of a supplied `(col, row)` pair, the total cell count, and
//! the normalized `UV` rectangle of a supplied cell. The variable-length nearest
//! `k` blend
//! ([`ImpostorGrid::nearest_views`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::nearest_views))
//! is deliberately **not** twinned: it carries a data-dependent insertion sort
//! over a candidate set, which is outside this one-thread-per-query contract.
//!
//! # Correctness model
//!
//! The cell indices, columns, rows and counts are discrete classifications built
//! from integer arithmetic and `floor`-style `f32`-to-`u32` truncation, so for
//! directions clear of the `octahedral` seam and the cell quantization
//! boundaries the `CPU` and `GPU` agree exactly and the parity test asserts an
//! exact `==`. The `octahedral` encode and decode and the `UV` rectangle thread
//! through multiplies, adds, one guarded reciprocal and one `sqrt`, so `CPU` and
//! `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every continuous quantity.
//!
//! # Degenerate inputs
//!
//! A `grid_dim` of zero is clamped up to one cell, mirroring the reference
//! [`ImpostorGrid::new`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::new),
//! so cell arithmetic never divides by zero. A zero-length direction encodes to
//! the origin and decodes to `+Z`, matching the reference fallback, so no result
//! is ever `NaN`. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `sqrt`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of arithmetic, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::billboard_atlas`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `billboard`-`atlas` kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`billboard_atlas`](prism_render_architecture::particle::billboard_atlas)
/// branch for branch; see the module documentation for the algorithm.
const BILLBOARD_ATLAS_WGSL: &str = r#"
// Billboard-atlas twin: one thread per query reproduces the octahedral encode
// and decode, the single-cell view selection, the integer grid arithmetic
// (cell_coord / cell_index / cell_count) and the normalized cell UV rectangle.
// It mirrors the CPU golden `particle::billboard_atlas` branch for branch, uses
// only the portable core-WGSL subset (abs/min/max/sqrt and + - * / plus
// unsigned index math), needs no transcendental call and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12. There is no loop, so
// the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::billboard_atlas；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // View direction to encode and to select a cell for.
    dir: vec3<f32>,
    // Cells per grid axis; clamped up to one, matching the reference `new`.
    grid_dim: u32,
    // Flattened cell index for the cell_coord and cell_uv_rect probes.
    cell: u32,
    // Column for the cell_index probe.
    col: u32,
    // Row for the cell_index probe.
    row: u32,
    // Atlas texel dimensions; carried for layout parity, unused by the geometry.
    atlas_width: u32,
    atlas_height: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Result {
    // oct_decode of the encoded direction, with a pad lane.
    dec_oct: vec3<f32>,
    pad0: f32,
    // cell_uv_rect [u0, v0, u1, v1] of the probe cell.
    uv_rect: vec4<f32>,
    // oct_encode of the query direction.
    enc_oct: vec2<f32>,
    // view_to_cell result: flattened index, column, row.
    view_index: u32,
    view_col: u32,
    view_row: u32,
    // cell_coord of the probe cell: column, row.
    coord_col: u32,
    coord_row: u32,
    // cell_index of the (col, row) probe.
    index_from_colrow: u32,
    // cell_count = grid_dim * grid_dim.
    cell_total: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Returns 1.0 for a non-negative input and -1.0 otherwise; mirrors the
// reference `sign_not_zero` (zero is treated as positive).
fn sign_not_zero(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0;
    }
    return -1.0;
}

// Normalizes a 3-vector, returning +Z for a zero-length input so the result is
// always a valid unit direction; mirrors the reference `normalize3`.
fn normalize3(v: vec3<f32>) -> vec3<f32> {
    let len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (len_sq > 0.0) {
        let inv = 1.0 / sqrt(len_sq);
        return vec3<f32>(v.x * inv, v.y * inv, v.z * inv);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// Encodes a direction onto the full-sphere octahedral square; mirrors the
// reference `oct_encode`.
fn oct_encode(dir: vec3<f32>) -> vec2<f32> {
    let l1 = abs(dir.x) + abs(dir.y) + abs(dir.z);
    if (l1 <= 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let inv = 1.0 / l1;
    var px = dir.x * inv;
    var py = dir.y * inv;
    let pz = dir.z * inv;
    if (pz < 0.0) {
        let folded_x = (1.0 - abs(py)) * sign_not_zero(px);
        let folded_y = (1.0 - abs(px)) * sign_not_zero(py);
        px = folded_x;
        py = folded_y;
    }
    return vec2<f32>(px, py);
}

// Decodes an octahedral square coordinate back to a unit direction; mirrors the
// reference `oct_decode`.
fn oct_decode(oct: vec2<f32>) -> vec3<f32> {
    var x = oct.x;
    var y = oct.y;
    let z = 1.0 - abs(x) - abs(y);
    let t = max(-z, 0.0);
    if (x >= 0.0) {
        x = x - t;
    } else {
        x = x + t;
    }
    if (y >= 0.0) {
        y = y - t;
    } else {
        y = y + t;
    }
    return normalize3(vec3<f32>(x, y, z));
}

// Clamps a floating grid coordinate to a valid integer cell axis in
// [0, dim - 1]; mirrors the reference `clamp_index`. The u32() conversion
// truncates toward zero, which equals floor for the non-negative value.
fn clamp_index(value: f32, dim: u32) -> u32 {
    let clamped = max(value, 0.0);
    let idx = u32(clamped);
    return min(idx, dim - 1u);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // The reference `new` clamps grid_dim up to one so the grid always has at
    // least a single cell and cell arithmetic never divides by zero.
    let dim = max(q.grid_dim, 1u);
    let g = f32(dim);

    // oct_encode of the query direction, then oct_decode of that encoding.
    let enc = oct_encode(q.dir);
    let dec = oct_decode(enc);

    // view_to_cell: normalize, encode, map the [-1, 1] square to [0, 1], scale
    // to the grid and clamp to a valid cell axis.
    let ndir = normalize3(q.dir);
    let voct = oct_encode(ndir);
    let s = voct.x * 0.5 + 0.5;
    let t = voct.y * 0.5 + 0.5;
    let vcol = clamp_index(s * g, dim);
    let vrow = clamp_index(t * g, dim);
    let vindex = vrow * dim + vcol;

    // cell_count: grid_dim squared (fixtures keep grid_dim small so the product
    // never overflows u32, matching the reference saturating multiply).
    let total = dim * dim;

    // cell_coord of the probe cell: clamp into range then split by the axis.
    let clamped_cell = min(q.cell, total - 1u);
    let ccol = clamped_cell % dim;
    let crow = clamped_cell / dim;

    // cell_index of the (col, row) probe: clamp each axis then flatten.
    let max_axis = dim - 1u;
    let ic = min(q.col, max_axis);
    let ir = min(q.row, max_axis);
    let cindex = ir * dim + ic;

    // cell_uv_rect of the probe cell: pure grid arithmetic from its (col, row).
    let invg = 1.0 / g;
    let u0 = f32(ccol) * invg;
    let v0 = f32(crow) * invg;
    let uv = vec4<f32>(u0, v0, u0 + invg, v0 + invg);

    var out: Result;
    out.dec_oct = dec;
    out.pad0 = 0.0;
    out.uv_rect = uv;
    out.enc_oct = enc;
    out.view_index = vindex;
    out.view_col = vcol;
    out.view_row = vrow;
    out.coord_col = ccol;
    out.coord_row = crow;
    out.index_from_colrow = cindex;
    out.cell_total = total;
    out.pad1 = 0u;
    out.pad2 = 0u;
    out.pad3 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`BILLBOARD_ATLAS_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The `vec3` direction carries a trailing word (`grid_dim`) so the following
/// scalars stay `16`-byte aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// View direction to encode and select a cell for.
    dir: [f32; 3],
    /// Cells per grid axis.
    grid_dim: u32,
    /// Flattened cell index probe for `cell_coord` and `cell_uv_rect`.
    cell: u32,
    /// Column probe for `cell_index`.
    col: u32,
    /// Row probe for `cell_index`.
    row: u32,
    /// `atlas` width in texels (layout parity only).
    atlas_width: u32,
    /// `atlas` height in texels (layout parity only).
    atlas_height: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// `oct_decode` of the encoded direction.
    oct_decode: [f32; 3],
    /// Padding lane after `oct_decode`.
    pad0: f32,
    /// `cell_uv_rect` of the probe cell.
    uv_rect: [f32; 4],
    /// `oct_encode` of the query direction.
    oct_encode: [f32; 2],
    /// `view_to_cell` flattened index.
    view_index: u32,
    /// `view_to_cell` column.
    view_col: u32,
    /// `view_to_cell` row.
    view_row: u32,
    /// `cell_coord` column of the probe cell.
    coord_col: u32,
    /// `cell_coord` row of the probe cell.
    coord_row: u32,
    /// `cell_index` of the `(col, row)` probe.
    index_from_colrow: u32,
    /// `cell_count` of the grid.
    cell_total: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// Padding word.
    pad3: u32,
}

/// One query for the `billboard`-`atlas` twin: a view direction, the grid
/// dimensions, and the integer probes for the grid-arithmetic functions.
///
/// The encode, decode, view selection, grid arithmetic and `UV` location are
/// independent, so a single query exercises every twinned function at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBillboardAtlasQuery {
    /// View direction fed to `oct_encode` and `view_to_cell`.
    pub dir: [f32; 3],
    /// Cells per grid axis; a value of zero is clamped up to one on device.
    pub grid_dim: u32,
    /// Flattened cell index probe for `cell_coord` and `cell_uv_rect`.
    pub cell: u32,
    /// Column probe for `cell_index`.
    pub col: u32,
    /// Row probe for `cell_index`.
    pub row: u32,
    /// `atlas` width in texels, carried for parity with the reference block.
    pub atlas_width: u32,
    /// `atlas` height in texels, carried for parity with the reference block.
    pub atlas_height: u32,
}

/// One resolved answer for a single query, mirroring every deterministic value
/// the reference reports across its twinned functions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBillboardAtlasResult {
    /// `octahedral` encoding of the query direction, matching
    /// [`oct_encode`](prism_render_architecture::particle::billboard_atlas::oct_encode).
    pub oct_encode: [f32; 2],
    /// `octahedral` decoding of `oct_encode`, matching
    /// [`oct_decode`](prism_render_architecture::particle::billboard_atlas::oct_decode).
    pub oct_decode: [f32; 3],
    /// Flattened cell index of the nearest view, matching the `index` of
    /// [`ImpostorGrid::view_to_cell`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::view_to_cell).
    pub view_cell_index: u32,
    /// Column of the nearest view, matching the `col` of
    /// [`ImpostorGrid::view_to_cell`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::view_to_cell).
    pub view_cell_col: u32,
    /// Row of the nearest view, matching the `row` of
    /// [`ImpostorGrid::view_to_cell`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::view_to_cell).
    pub view_cell_row: u32,
    /// Column of the probe cell, matching the first element of
    /// [`ImpostorGrid::cell_coord`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::cell_coord).
    pub cell_col: u32,
    /// Row of the probe cell, matching the second element of
    /// [`ImpostorGrid::cell_coord`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::cell_coord).
    pub cell_row: u32,
    /// Flattened index of the `(col, row)` probe, matching
    /// [`ImpostorGrid::cell_index`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::cell_index).
    pub cell_index: u32,
    /// Total number of cells, matching
    /// [`ImpostorGrid::cell_count`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::cell_count).
    pub cell_count: u32,
    /// Normalized `UV` rectangle `[u0, v0, u1, v1]` of the probe cell, matching
    /// [`ImpostorGrid::cell_uv_rect`](prism_render_architecture::particle::billboard_atlas::ImpostorGrid::cell_uv_rect).
    pub cell_uv_rect: [f32; 4],
}

/// Encodes one [`GpuBillboardAtlasQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &GpuBillboardAtlasQuery) -> GpuQuery {
    GpuQuery {
        dir: q.dir,
        grid_dim: q.grid_dim,
        cell: q.cell,
        col: q.col,
        row: q.row,
        atlas_width: q.atlas_width,
        atlas_height: q.atlas_height,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`GpuBillboardAtlasResult`].
fn decode_result(raw: &GpuResult) -> GpuBillboardAtlasResult {
    GpuBillboardAtlasResult {
        oct_encode: raw.oct_encode,
        oct_decode: raw.oct_decode,
        view_cell_index: raw.view_index,
        view_cell_col: raw.view_col,
        view_cell_row: raw.view_row,
        cell_col: raw.coord_col,
        cell_row: raw.coord_row,
        cell_index: raw.index_from_colrow,
        cell_count: raw.cell_total,
        cell_uv_rect: raw.uv_rect,
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

/// A compiled, reusable `billboard`-`atlas` compute pipeline, twinning the `CPU`
/// golden
/// [`billboard_atlas`](prism_render_architecture::particle::billboard_atlas).
pub struct GpuBillboardAtlas {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBillboardAtlas {
    /// Compiles the `billboard`-`atlas` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBillboardAtlas {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_billboard_atlas"),
            source: ShaderSource::Wgsl(BILLBOARD_ATLAS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_billboard_atlas_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_billboard_atlas_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_billboard_atlas_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBillboardAtlas {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`GpuBillboardAtlasResult`] per input, in order.
    ///
    /// The cell indices, columns, rows and counts equal the reference exactly
    /// for directions clear of the `octahedral` seam and the cell quantization
    /// boundaries; the `octahedral` encode and decode and the `UV` rectangle
    /// match to within the tolerance documented on this module. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuBillboardAtlasQuery],
    ) -> Vec<GpuBillboardAtlasResult> {
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
            label: Some("prism_volumetric_billboard_atlas_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_billboard_atlas_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_billboard_atlas_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_billboard_atlas_bind_group"),
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
            label: Some("prism_volumetric_billboard_atlas_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_billboard_atlas_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_billboard_atlas_pass"),
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
