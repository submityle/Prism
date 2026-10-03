//! `wgpu` compute twin of the per-cell-triangle ray intersection on an implicit
//! displacement heightfield
//! (`prism_render_architecture::ray_scene::heightfield`).
//!
//! A [`Heightfield`] stores a `width * height` grid of height samples plus a
//! planar domain (`origin` and the `extent` of the grid in the `X`/`Z` plane)
//! and *implicitly* triangulates it: grid vertex `(ix, iz)` maps to world
//! position `origin + (ix/(width-1) * extent.x, heights[iz*width+ix],
//! iz/(height-1) * extent.y)`, and each grid cell is split into two triangles.
//! The golden `Heightfield::intersect_cell_triangle` runs a Möller–Trumbore
//! test against one named `(cell, triangle)` pair.
//!
//! [`GpuHeightfieldCell`] is the on-device twin that runs one thread per
//! `(cell, triangle, ray)` query against a shared height grid, reproducing the
//! golden closed form lane for lane: the implicit triangle corners, the
//! Möller–Trumbore barycentric solve, the ray-interval clamp, the barycentric
//! reconstruction of the hit `position` and domain `uv`, and the geometric
//! normal oriented against the ray.
//!
//! # What is twinned
//!
//! Only the single `(cell, triangle)`-vs-ray closed form is twinned — the
//! whole-field `BVH` build and its stack traversal stay on the host, which also
//! decides which cells/triangles to enqueue. The kernel decodes a query's
//! `cell` into grid coordinates (`ix = cell % (width-1)`, `iz = cell /
//! (width-1)`), reads the three corner heights, forms the two edge vectors, and
//! runs Möller–Trumbore with the same `EPS` determinant guard, the same `u`,
//! `v`, `u + v` and interval rejections, and the same barycentric blend of the
//! corner positions and domain `UV`s. The geometric normal `e1 x e2` is flipped
//! to face the incident ray and normalized with the golden near-zero fallback.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `dot`, `cross`, a single `sqrt`, integer `/` and `%`, and `+ - * /` — with
//! no `sin`, `cos`, `exp`, `log`, `pow` or optional device feature, so it runs
//! unmodified on Metal, Vulkan and DX12. A direct `f32` equality is forbidden,
//! so the determinant test compares `abs(det)` against [`EPS`] rather than
//! exact zero, and the near-zero normal fallback compares a squared length
//! against a floor.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of the same products,
//! quotients and one `sqrt` the scalar reference evaluates, in the same
//! associativity. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the continuous `t`, `u`, `v`,
//! `position`, `normal` and `uv` yet an *exact* match on the discrete `hit`,
//! `front_face`, `cell` and `triangle` fields, with fixtures and the randomized
//! sweep kept clear of the branch-switch loci so both sides fold the identical
//! verdict.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::heightfield`；无第三方引擎源码或衍生代码。

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

/// The portable core-`WGSL` ray/heightfield-cell kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` golden
/// `Heightfield::intersect_cell_triangle` step for step; see the module
/// documentation for the algorithm.
const HEIGHTFIELD_CELL_WGSL: &str = r#"
// Ray vs heightfield-cell twin: one thread per (cell, triangle, ray) query
// decodes the implicit triangle corners from a shared height grid, runs a
// Moller-Trumbore barycentric solve, and reconstructs the hit position, domain
// UV and oriented unit normal. It mirrors the CPU golden
// `ray_scene::heightfield::Heightfield::intersect_cell_triangle` guard for
// guard, uses only the portable core-WGSL subset (abs/min/max/dot/cross, one
// sqrt, integer / and %, and + - * /), and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::heightfield;
// no third-party engine source or derived code.

struct Params {
    // Grid columns (vertices along X); always at least two.
    width: u32,
    // Grid rows (vertices along Z); always at least two.
    height: u32,
    // Number of valid queries.
    count: u32,
    // Padding to a 16-byte boundary before the vec4 members.
    pad0: u32,
    // World-space corner the grid's (0, 0) vertex maps to; w is pad.
    origin: vec4<f32>,
    // Grid extent along X and Z in world units; zw are pad.
    extent: vec4<f32>,
}

// One query. 48-byte std430 stride matching the host `GpuQuery`: the ray origin
// (with t_min in w), the ray direction (with t_max in w), the flattened cell
// index and the triangle selector, padded to a 16-byte multiple.
struct Query {
    origin: vec4<f32>,
    dir: vec4<f32>,
    cell: u32,
    tri: u32,
    pad0: u32,
    pad1: u32,
}

// One result. 80-byte std430 stride matching the host `GpuResult`.
struct Result {
    position: vec4<f32>,
    normal: vec4<f32>,
    t: f32,
    u: f32,
    v: f32,
    hit: u32,
    uv0: f32,
    uv1: f32,
    front_face: u32,
    cell: u32,
    triangle: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> heights: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

// Determinant magnitude floor classifying the ray as parallel to the triangle.
// The reference uses `det.abs() < 1e-8`; the twin compares the same magnitude
// against this shared floor.
const EPS: f32 = 1.0e-8;

// Squared-length floor for the near-zero normal fallback, matching the golden
// `normalize_or` guard `len_sq < 1e-24`.
const LEN_SQ_FLOOR: f32 = 1.0e-24;

// Domain UV of grid vertex (cx, cz) in [0, 1]^2.
fn grid_param(cx: u32, cz: u32) -> vec2<f32> {
    let fx = f32(cx) / f32(params.width - 1u);
    let fz = f32(cz) / f32(params.height - 1u);
    return vec2<f32>(fx, fz);
}

// World-space position of grid vertex (cx, cz).
fn vertex_pos(cx: u32, cz: u32) -> vec3<f32> {
    let p = grid_param(cx, cz);
    let h = heights[cz * params.width + cx];
    return vec3<f32>(
        params.origin.x + p.x * params.extent.x,
        params.origin.y + h,
        params.origin.z + p.y * params.extent.y,
    );
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }

    let q = queries[idx];

    var res: Result;
    res.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    res.normal = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    res.t = 0.0;
    res.u = 0.0;
    res.v = 0.0;
    res.hit = 0u;
    res.uv0 = 0.0;
    res.uv1 = 0.0;
    res.front_face = 0u;
    res.cell = 0u;
    res.triangle = 0u;
    res.pad0 = 0u;
    res.pad1 = 0u;
    res.pad2 = 0u;

    // Decode the flattened cell into grid coordinates (cells_x = width - 1).
    let cells_x = params.width - 1u;
    let ix = q.cell % cells_x;
    let iz = q.cell / cells_x;

    // Corner grid coordinates by triangle selector.
    let i0x = ix;
    let i0z = iz;
    var i1x = ix + 1u;
    var i1z = iz;
    var i2x = ix + 1u;
    var i2z = iz + 1u;
    if (q.tri != 0u) {
        i1x = ix + 1u;
        i1z = iz + 1u;
        i2x = ix;
        i2z = iz + 1u;
    }

    let p0 = vertex_pos(i0x, i0z);
    let p1 = vertex_pos(i1x, i1z);
    let p2 = vertex_pos(i2x, i2z);
    let uv0 = grid_param(i0x, i0z);
    let uv1 = grid_param(i1x, i1z);
    let uv2 = grid_param(i2x, i2z);

    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let dir = q.dir.xyz;
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);
    if (abs(det) < EPS) {
        results[idx] = res;
        return;
    }
    let inv_det = 1.0 / det;
    let tvec = q.origin.xyz - p0;
    let u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) {
        results[idx] = res;
        return;
    }
    let qvec = cross(tvec, e1);
    let v = dot(dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        results[idx] = res;
        return;
    }
    let t = dot(e2, qvec) * inv_det;
    let t_lo = q.origin.w;
    let t_hi = q.dir.w;
    if (t < t_lo || t > t_hi) {
        results[idx] = res;
        return;
    }

    let w0 = 1.0 - u - v;
    let position = p0 * w0 + p1 * u + p2 * v;

    let ng_raw = cross(e1, e2);
    let facing = dot(dir, ng_raw) < 0.0;
    var geo = ng_raw;
    if (!facing) {
        geo = -ng_raw;
    }
    var normal = geo;
    let len_sq = dot(geo, geo);
    if (len_sq >= LEN_SQ_FLOOR) {
        normal = geo * (1.0 / sqrt(len_sq));
    }

    let uv = uv0 * w0 + uv1 * u + uv2 * v;

    res.position = vec4<f32>(position, 0.0);
    res.normal = vec4<f32>(normal, 0.0);
    res.t = t;
    res.u = u;
    res.v = v;
    res.hit = 1u;
    res.uv0 = uv.x;
    res.uv1 = uv.y;
    res.front_face = select(0u, 1u, facing);
    res.cell = q.cell;
    res.triangle = q.tri;
    results[idx] = res;
}
"#;

/// One ray vs heightfield-cell-triangle query: the named cell/triangle and the
/// ray. The height grid itself is shared across the batch and passed separately
/// to [`GpuHeightfieldCell::evaluate`].
///
/// Mirrors a single reference `Heightfield::intersect_cell_triangle` call. The
/// `t_min`/`t_max` interval is applied verbatim (no clamp), matching the golden
/// method which reads `ray.t_min()`/`ray.t_max()` directly. Derives only
/// [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightfieldCellQuery {
    /// Flattened cell index (`iz * (width - 1) + ix`) to test.
    pub cell: u32,
    /// Which of the cell's two triangles (`0` or `1`).
    pub triangle: u32,
    /// Ray origin.
    pub origin: [f32; 3],
    /// Ray direction (never assumed unit; `t` is in `direction` lengths).
    pub direction: [f32; 3],
    /// Lower bound of the ray interval.
    pub t_min: f32,
    /// Upper bound of the ray interval.
    pub t_max: f32,
}

/// The resolved verdict for one query, the host-side mirror of the kernel's
/// `Result` lane.
///
/// `hit` is `1` when the ray struck the cell triangle inside its interval and
/// `0` otherwise. `t`, `u`, `v`, `position`, `normal`, `uv`, `front_face`,
/// `cell` and `triangle` are meaningful only when `hit` is `1` (they are left
/// zeroed on a miss). Derives only [`PartialEq`] (no `Eq`/`Hash`) because it
/// holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightfieldCellResult {
    /// Whether the ray crossed the triangle inside its interval (`1` = hit).
    pub hit: u32,
    /// Ray parameter at the intersection; meaningful only when `hit`.
    pub t: f32,
    /// Barycentric `u` within the struck triangle; meaningful only when `hit`.
    pub u: f32,
    /// Barycentric `v` within the struck triangle; meaningful only when `hit`.
    pub v: f32,
    /// World-space hit position; meaningful only when `hit`.
    pub position: [f32; 3],
    /// Unit geometric normal oriented against the ray; meaningful only when `hit`.
    pub normal: [f32; 3],
    /// Interpolated domain `UV` in `[0, 1]^2`; meaningful only when `hit`.
    pub uv: [f32; 2],
    /// Whether the ray struck the front (counter-clockwise) face (`1` = front).
    pub front_face: u32,
    /// Flattened cell index that was struck; echoes the query on a hit.
    pub cell: u32,
    /// Which triangle was struck (`0` or `1`); echoes the query on a hit.
    pub triangle: u32,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`HEIGHTFIELD_CELL_WGSL`]: the grid dimensions and query count,
/// then the domain `origin` and `extent` as `vec4` lanes — `48` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Grid columns.
    width: u32,
    /// Grid rows.
    height: u32,
    /// Number of valid queries.
    count: u32,
    /// Padding word before the `vec4` members.
    pad0: u32,
    /// Domain origin in `xyz`; the `w` lane is unused padding.
    origin: [f32; 4],
    /// Grid extent in `xy`; the `zw` lanes are unused padding.
    extent: [f32; 4],
}

/// One query as uploaded. `48`-byte `std430` stride matching `Query` in the
/// shader: the ray origin (with `t_min` in `w`), the ray direction (with
/// `t_max` in `w`), the cell index, the triangle selector and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Ray origin in `xyz`; `w` carries `t_min`.
    origin: [f32; 4],
    /// Ray direction in `xyz`; `w` carries `t_max`.
    dir: [f32; 4],
    /// Flattened cell index.
    cell: u32,
    /// Triangle selector (`0` or `1`).
    tri: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One result as read back. `80`-byte `std430` stride matching `Result` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Hit position in `xyz`; the `w` lane is unused padding.
    position: [f32; 4],
    /// Oriented unit normal in `xyz`; the `w` lane is unused padding.
    normal: [f32; 4],
    /// Ray parameter at the intersection.
    t: f32,
    /// Barycentric `u`.
    u: f32,
    /// Barycentric `v`.
    v: f32,
    /// Hit flag (`1` = hit).
    hit: u32,
    /// Domain `U` coordinate.
    uv0: f32,
    /// Domain `V` coordinate.
    uv1: f32,
    /// Front-face flag (`1` = front).
    front_face: u32,
    /// Flattened cell index that was struck.
    cell: u32,
    /// Triangle selector that was struck.
    triangle: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Packs a [`HeightfieldCellQuery`] into the `std430` upload layout.
fn encode_query(q: &HeightfieldCellQuery) -> GpuQuery {
    GpuQuery {
        origin: [q.origin[0], q.origin[1], q.origin[2], q.t_min],
        dir: [q.direction[0], q.direction[1], q.direction[2], q.t_max],
        cell: q.cell,
        tri: q.triangle,
        pad0: 0,
        pad1: 0,
    }
}

/// Maps one kernel `Result` lane back to the host [`HeightfieldCellResult`].
fn decode_result(raw: &GpuResult) -> HeightfieldCellResult {
    HeightfieldCellResult {
        hit: raw.hit,
        t: raw.t,
        u: raw.u,
        v: raw.v,
        position: [raw.position[0], raw.position[1], raw.position[2]],
        normal: [raw.normal[0], raw.normal[1], raw.normal[2]],
        uv: [raw.uv0, raw.uv1],
        front_face: raw.front_face,
        cell: raw.cell,
        triangle: raw.triangle,
    }
}

/// A compiled, reusable ray/heightfield-cell intersection pipeline.
pub struct GpuHeightfieldCell {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHeightfieldCell {
    /// Compiles the ray/heightfield-cell kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHeightfieldCell {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_heightfield_cell"),
            source: ShaderSource::Wgsl(HEIGHTFIELD_CELL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_heightfield_cell_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_heightfield_cell_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_heightfield_cell_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHeightfieldCell {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` against the shared height grid,
    /// returning one [`HeightfieldCellResult`] per query in input order.
    ///
    /// `width`/`height` are the grid vertex counts (each at least two),
    /// `origin`/`extent` the planar domain, and `heights` the row-major
    /// `width * height` samples (`heights[iz * width + ix]`). The returned
    /// result for query `q` mirrors `Heightfield::intersect_cell_triangle`
    /// evaluated on `q`'s cell/triangle and ray. An empty `queries` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return before any dispatch.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        width: u32,
        height: u32,
        origin: [f32; 3],
        extent: [f32; 2],
        heights: &[f32],
        queries: &[HeightfieldCellQuery],
    ) -> Vec<HeightfieldCellResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            width,
            height,
            count: count as u32,
            pad0: 0,
            origin: [origin[0], origin[1], origin[2], 0.0],
            extent: [extent[0], extent[1], 0.0, 0.0],
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_heightfield_cell_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let heights_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_heightfield_cell_heights"),
            contents: bytemuck::cast_slice(heights),
            usage: BufferUsages::STORAGE,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_heightfield_cell_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_heightfield_cell_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_heightfield_cell_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: heights_buf.as_entire_binding(),
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
            label: Some("prism_volumetric_heightfield_cell_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_heightfield_cell_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_heightfield_cell_pass"),
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
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
