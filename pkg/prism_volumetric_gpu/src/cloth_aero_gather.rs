//! `wgpu` compute twin of the race-free, per-vertex cloth aerodynamic gather
//! [`accumulate_aero_gather`](prism_render_architecture::cloth::aero_gather::accumulate_aero_gather).
//!
//! Wind acts on a cloth mesh *per triangle*: each face decomposes the relative
//! wind into a drag component along its normal and a lift component in its
//! plane, scales the sum by the triangle area (and, in the quadratic model, by
//! the dynamic pressure), and spreads a third of that force onto each of its
//! three vertices. A `GPU` pass that launched one thread per triangle would
//! have three threads racing to add into one vertex, so the production path
//! inverts the loop: one thread per **vertex** *gathers* the forces of the
//! triangles incident to it, reading a single frozen velocity snapshot so every
//! face sees the same input state (an explicit / `Jacobi` update). Every
//! velocity is then written exactly once with no atomics.
//!
//! [`GpuClothAeroGather`] is the on-device twin of that `Jacobi` gather. One
//! thread owns one vertex of one query, walks the vertex's ascending `CSR`
//! incidence, reuses the shared per-face force
//! [`triangle_aero_force`](prism_render_architecture::cloth::wind::triangle_wind_force)
//! and the integer-hash `turbulence_offset`, and writes the updated velocity,
//! so a passing real-device parity run is direct evidence the ported kernel
//! computes the same velocity field the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For each query and each free vertex `v` the kernel reproduces
//! `velocity[v] += (sum over incident faces of face_force * (1/3)) * inverse_mass * dt`,
//! where each `face_force` is evaluated against the frozen snapshot velocities:
//!
//! * `triangle_aero_force`: `cross = (p1 - p0) x (p2 - p0)`, `cls = |cross|^2`;
//!   a degenerate `cls <= 1e-12` contributes `0`; otherwise `area = 0.5 *
//!   sqrt(cls)`, `normal = cross / sqrt(cls)`, `rel = wind - (v0 + v1 + v2) /
//!   3`, `dir = normal * dot(rel, normal) * drag + (rel - that) * lift`, and
//!   `pressure = area` (linear) or `area * 0.5 * air_density * sqrt(|rel|^2)`
//!   when `air_density > 0`.
//! * `turbulence_offset`: a zero or negative `turbulence` yields `0`; otherwise
//!   the three vertex indices are mixed with the classic spatial-hash primes
//!   and run through an integer avalanche `hash_to_unit` per axis, scaled by
//!   `turbulence` — pure `u32` wrapping arithmetic, bit-identical on every
//!   platform.
//! * Pinned vertices (`inverse_mass <= 0`), vertices past the active count, and
//!   degenerate dispatches (non-positive or non-finite `dt`, empty particle or
//!   triangle set) pass the input velocity through unchanged, matching every
//!   reference no-op.
//!
//! # What stays on the host
//!
//! The variable-length `CSR` adjacency is pre-resolved on the host into a
//! fixed-length `offsets`/`entries` pair and uploaded per query, and the wind
//! and aero coefficients are sanitized host-side (non-finite to `0`,
//! coefficients `max(0)`, turbulence clamped to `0..=1`) exactly as
//! `prism_physics_core`'s aero sanitizers do, so the device needs no `NaN`
//! detection. The topology build, the frame state machine, and the per-mesh
//! allocation stay on the host; this twin models only the stateless per-vertex
//! gather, zero-padded to the fixed caps.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `dot`,
//! `cross`, `min`, `+ - * /`, and `u32` wrapping bit math for the hash — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, no inverse trigonometry, no `u64`.
//! No optional device feature is required, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::aero_gather`；无第三方引擎源码或衍生代码。
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

/// Fixed per-query vertex cap. The host zero-pads each query's columns to this
/// size and dispatches `MAX_VERTS` threads per query; the kernel exits threads
/// past the active vertex count.
const MAX_VERTS: usize = 64;

/// Fixed per-query triangle cap.
const MAX_TRIS: usize = 96;

/// Fixed per-query incidence cap, `3 * MAX_TRIS`: every in-range triangle
/// contributes exactly three `(vertex, triangle)` incidences to the `CSR`
/// entries.
const MAX_INCIDENT: usize = 3 * MAX_TRIS;

/// Flat length of the padded position and velocity columns, `3 * MAX_VERTS`.
const POS_LEN: usize = 3 * MAX_VERTS;

/// Length of the padded `CSR` offsets column, `MAX_VERTS + 1`.
const OFFSETS_LEN: usize = MAX_VERTS + 1;

/// Flat length of the padded triangle column, `3 * MAX_TRIS`.
const TRI_LEN: usize = 3 * MAX_TRIS;

/// The portable core-`WGSL` per-vertex aero-gather kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`accumulate_aero_gather`](prism_render_architecture::cloth::aero_gather::accumulate_aero_gather);
/// see the module documentation for the algorithm.
const CLOTH_AERO_GATHER_WGSL: &str = r#"
// Per-vertex Jacobi aero gather: one thread owns one vertex of one query. It
// walks the vertex's ascending CSR incidence, sums a third of each incident
// triangle's aero force (evaluated against the frozen input/snapshot
// velocities), and writes velocity += accum * inverse_mass * dt. Pinned
// vertices, vertices past the active count, and degenerate dispatches pass the
// input velocity through unchanged. Only sqrt/dot/cross/min, + - * / on f32,
// and u32 wrapping bit math (for the turbulence hash) are used — no
// transcendental, no u64.
//
// Provenance: 孪生自本仓 `prism_render_architecture::cloth::aero_gather`；
// 无第三方引擎源码或衍生代码。

// Per-query vertex cap; one thread per vertex, threads past count*VERTS exit.
const VERTS: u32 = 64u;
// Squared edge-cross length below which a triangle is treated as degenerate.
const EPS_LEN_SQ: f32 = 1e-12;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Flat per-vertex positions, (x,y,z) interleaved, zero-padded to 3*VERTS.
    positions: array<f32, 192>,
    // Flat per-vertex snapshot velocities, same layout as positions.
    velocities: array<f32, 192>,
    // Per-vertex inverse mass; <= 0 pins the vertex.
    inverse_masses: array<f32, 64>,
    // Flat per-triangle vertex indices, (i0,i1,i2) interleaved, 3*MAX_TRIS.
    triangles: array<u32, 288>,
    // CSR start offsets into entries, length VERTS+1, non-decreasing.
    offsets: array<u32, 65>,
    // Flattened ascending triangle indices per vertex, length 3*MAX_TRIS.
    entries: array<u32, 288>,
    // Active particle count (host guarantees columns are index-aligned).
    count: u32,
    // Adjacency vertex count; the gather spans min(adj_vertex_count, count).
    adj_vertex_count: u32,
    // Active triangle count; entries referencing >= this are skipped.
    triangle_count: u32,
    // 1 when the dispatch is a live gather, 0 for a degenerate no-op.
    live_flag: u32,
    // Sanitized ambient wind velocity.
    wind_x: f32,
    wind_y: f32,
    wind_z: f32,
    // Sanitized turbulence strength in 0..=1.
    turbulence: f32,
    // Sanitized non-negative drag coefficient.
    drag: f32,
    // Sanitized non-negative lift coefficient.
    lift: f32,
    // Sanitized non-negative fluid density (> 0 selects the quadratic model).
    air_density: f32,
    // Integration step; only read when active == 1.
    dt: f32,
}

struct Result {
    // Updated per-vertex velocities, flat (x,y,z), zero-padded to 3*VERTS.
    velocities: array<f32, 192>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

fn load_pos(qi: u32, i: u32) -> vec3<f32> {
    let base = i * 3u;
    return vec3<f32>(
        queries[qi].positions[base],
        queries[qi].positions[base + 1u],
        queries[qi].positions[base + 2u],
    );
}

fn load_vel(qi: u32, i: u32) -> vec3<f32> {
    let base = i * 3u;
    return vec3<f32>(
        queries[qi].velocities[base],
        queries[qi].velocities[base + 1u],
        queries[qi].velocities[base + 2u],
    );
}

// Integer avalanche (xorshift-multiply) mapping a seed to a value in [-1, 1].
fn hash_to_unit(seed: u32) -> f32 {
    var h = seed * 0x9E3779B1u;
    h = h ^ (h >> 15u);
    h = h * 0x85EBCA77u;
    h = h ^ (h >> 13u);
    h = h * 0xC2B2AE3Du;
    h = h ^ (h >> 16u);
    let unit = f32(h >> 8u) * (1.0 / 16777216.0);
    return unit * 2.0 - 1.0;
}

// Deterministic per-triangle turbulence jitter from the three vertex indices.
fn turbulence_offset(i0: u32, i1: u32, i2: u32, turb: f32) -> vec3<f32> {
    if (turb <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let base = (i0 * 73856093u) ^ (i1 * 19349663u) ^ (i2 * 83492791u);
    let jitter = vec3<f32>(
        hash_to_unit(base ^ 0x00A55A00u),
        hash_to_unit(base ^ 0x5A0000A5u),
        hash_to_unit(base ^ 0x00FF00FFu),
    );
    return jitter * turb;
}

// Per-triangle aerodynamic force from the relative wind (sanitized aero coeffs).
fn triangle_aero_force(
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    wind: vec3<f32>,
    drag: f32,
    lift: f32,
    air_density: f32,
) -> vec3<f32> {
    let crs = cross(p1 - p0, p2 - p0);
    let cls = dot(crs, crs);
    if (cls <= EPS_LEN_SQ) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let root = sqrt(cls);
    let area = 0.5 * root;
    let normal = crs * (1.0 / root);
    let fv = (v0 + v1 + v2) * (1.0 / 3.0);
    let rel = wind - fv;
    let nc = normal * dot(rel, normal);
    let tc = rel - nc;
    let dir = nc * drag + tc * lift;
    var pressure = area;
    if (air_density > 0.0) {
        pressure = area * (0.5 * air_density * sqrt(dot(rel, rel)));
    }
    return dir * pressure;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let total = params.count * VERTS;
    if (tid >= total) {
        return;
    }
    let qi = tid / VERTS;
    let v = tid % VERTS;
    let out_base = v * 3u;

    // Jacobi: the result defaults to the frozen input velocity. Pinned
    // vertices, vertices past the active span, and no-op dispatches keep it.
    var vel = load_vel(qi, v);

    let count = queries[qi].count;
    let vertices = min(queries[qi].adj_vertex_count, count);
    let tri_count = queries[qi].triangle_count;
    let inv_mass = queries[qi].inverse_masses[v];

    if (queries[qi].live_flag != 0u && v < vertices && inv_mass > 0.0) {
        let wind = vec3<f32>(
            queries[qi].wind_x,
            queries[qi].wind_y,
            queries[qi].wind_z,
        );
        let turb = queries[qi].turbulence;
        let drag = queries[qi].drag;
        let lift = queries[qi].lift;
        let air = queries[qi].air_density;

        var accum = vec3<f32>(0.0, 0.0, 0.0);
        let start = queries[qi].offsets[v];
        let end = queries[qi].offsets[v + 1u];
        for (var e = start; e < end; e = e + 1u) {
            let t = queries[qi].entries[e];
            if (t >= tri_count) {
                continue;
            }
            let ti = t * 3u;
            let i0 = queries[qi].triangles[ti];
            let i1 = queries[qi].triangles[ti + 1u];
            let i2 = queries[qi].triangles[ti + 2u];
            if (i0 >= count || i1 >= count || i2 >= count) {
                continue;
            }
            let wind_vec = wind + turbulence_offset(i0, i1, i2, turb);
            let force = triangle_aero_force(
                load_pos(qi, i0),
                load_pos(qi, i1),
                load_pos(qi, i2),
                load_vel(qi, i0),
                load_vel(qi, i1),
                load_vel(qi, i2),
                wind_vec,
                drag,
                lift,
                air,
            );
            accum = accum + force * (1.0 / 3.0);
        }
        vel = vel + accum * (inv_mass * queries[qi].dt);
    }

    results[qi].velocities[out_base] = vel.x;
    results[qi].velocities[out_base + 1u] = vel.y;
    results[qi].velocities[out_base + 2u] = vel.z;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`CLOTH_AERO_GATHER_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the padded particle columns, the pre-resolved `CSR` adjacency, the active
/// counts, the degenerate-dispatch flag, and the sanitized wind and aero
/// coefficients.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Flat per-vertex positions, `(x, y, z)` interleaved.
    positions: [f32; POS_LEN],
    /// Flat per-vertex snapshot velocities, `(x, y, z)` interleaved.
    velocities: [f32; POS_LEN],
    /// Per-vertex inverse mass (`0` pins the vertex).
    inverse_masses: [f32; MAX_VERTS],
    /// Flat per-triangle vertex indices, `(i0, i1, i2)` interleaved.
    triangles: [u32; TRI_LEN],
    /// `CSR` start offsets into `entries`, length `MAX_VERTS + 1`.
    offsets: [u32; OFFSETS_LEN],
    /// Flattened ascending triangle indices per vertex.
    entries: [u32; MAX_INCIDENT],
    /// Active particle count.
    count: u32,
    /// Adjacency vertex count.
    adj_vertex_count: u32,
    /// Active triangle count.
    triangle_count: u32,
    /// `1` for a live gather, `0` for a degenerate no-op.
    active: u32,
    /// Sanitized ambient wind `x`.
    wind_x: f32,
    /// Sanitized ambient wind `y`.
    wind_y: f32,
    /// Sanitized ambient wind `z`.
    wind_z: f32,
    /// Sanitized turbulence strength in `0..=1`.
    turbulence: f32,
    /// Sanitized non-negative drag coefficient.
    drag: f32,
    /// Sanitized non-negative lift coefficient.
    lift: f32,
    /// Sanitized non-negative fluid density.
    air_density: f32,
    /// Integration step (`dt`).
    dt: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the padded updated velocity column.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Flat per-vertex updated velocities, `(x, y, z)` interleaved.
    velocities: [f32; POS_LEN],
}

/// Replaces a non-finite scalar with `0`, mirroring the physics-core
/// `sanitize_finite`.
fn sanitize_finite(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Clamps a scalar to `0..=1`, mapping any non-finite input to `0`, mirroring
/// the physics-core `sanitize_unit`.
fn sanitize_unit(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Clamps a scalar non-negative, mapping any non-finite input to `0`, mirroring
/// the physics-core `sanitize_non_negative`.
fn sanitize_non_negative(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// One aero-gather query: the padded particle columns, the pre-resolved `CSR`
/// adjacency, and the raw wind and aero inputs, mirroring the arguments the
/// golden
/// [`accumulate_aero_gather`](prism_render_architecture::cloth::aero_gather::accumulate_aero_gather)
/// consumes. The constructor packs and host-sanitizes the inputs into the
/// device layout.
#[derive(Clone, Copy)]
pub struct ClothAeroGatherQuery {
    /// Packed, sanitized device payload.
    inner: GpuQuery,
}

impl ClothAeroGatherQuery {
    /// Builds a query from the particle columns, the pre-resolved `CSR`
    /// adjacency, and the raw wind and aero inputs.
    ///
    /// `positions`, `velocities` and `inverse_masses` are index-aligned
    /// particle columns; `triangles` are the mesh faces; `offsets` and
    /// `entries` are the flattened `CSR` adjacency (as returned by the golden
    /// `VertexTriangleAdjacency`), with `adj_vertex_count` the adjacency's
    /// vertex count. The wind and aero coefficients are sanitized here (
    /// non-finite to `0`, coefficients `max(0)`, turbulence clamped to
    /// `0..=1`) exactly as `prism_physics_core` does internally, so the device
    /// path needs no `NaN` detection. The `active` flag reproduces the golden
    /// degenerate guard: a non-positive or non-finite `dt`, an empty particle
    /// set, or an empty triangle set is a no-op.
    ///
    /// # Panics
    ///
    /// Panics when any column exceeds the fixed caps (`MAX_VERTS` vertices,
    /// `MAX_TRIS` triangles, `MAX_INCIDENT` incidences, `MAX_VERTS + 1`
    /// offsets).
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a gather query is intrinsically the particle columns, the CSR \
                  adjacency, and the wind/aero inputs; grouping them into \
                  structs here would only shadow the golden arguments"
    )]
    pub fn new(
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        inverse_masses: &[f32],
        triangles: &[[u32; 3]],
        offsets: &[u32],
        entries: &[u32],
        adj_vertex_count: u32,
        wind_velocity: [f32; 3],
        turbulence: f32,
        drag: f32,
        lift: f32,
        air_density: f32,
        dt: f32,
    ) -> ClothAeroGatherQuery {
        let count = positions.len();
        assert!(count <= MAX_VERTS, "vertex count exceeds MAX_VERTS");
        assert!(
            velocities.len() == count && inverse_masses.len() == count,
            "particle columns must be index-aligned"
        );
        assert!(
            triangles.len() <= MAX_TRIS,
            "triangle count exceeds MAX_TRIS"
        );
        assert!(offsets.len() <= OFFSETS_LEN, "offsets exceed MAX_VERTS + 1");
        assert!(entries.len() <= MAX_INCIDENT, "entries exceed MAX_INCIDENT");

        let mut inner = GpuQuery::zeroed();
        for (vertex, pos) in positions.iter().enumerate() {
            let base = vertex * 3;
            inner.positions[base] = pos[0];
            inner.positions[base + 1] = pos[1];
            inner.positions[base + 2] = pos[2];
        }
        for (vertex, vel) in velocities.iter().enumerate() {
            let base = vertex * 3;
            inner.velocities[base] = vel[0];
            inner.velocities[base + 1] = vel[1];
            inner.velocities[base + 2] = vel[2];
        }
        for (slot, mass) in inner.inverse_masses.iter_mut().zip(inverse_masses.iter()) {
            *slot = *mass;
        }
        for (tri, face) in triangles.iter().enumerate() {
            let base = tri * 3;
            inner.triangles[base] = face[0];
            inner.triangles[base + 1] = face[1];
            inner.triangles[base + 2] = face[2];
        }
        for (slot, value) in inner.offsets.iter_mut().zip(offsets.iter()) {
            *slot = *value;
        }
        for (slot, value) in inner.entries.iter_mut().zip(entries.iter()) {
            *slot = *value;
        }

        let triangle_count = triangles.len() as u32;
        let active = u32::from(dt.is_finite() && dt > 0.0 && count > 0 && triangle_count > 0);

        inner.count = count as u32;
        inner.adj_vertex_count = adj_vertex_count;
        inner.triangle_count = triangle_count;
        inner.active = active;
        inner.wind_x = sanitize_finite(wind_velocity[0]);
        inner.wind_y = sanitize_finite(wind_velocity[1]);
        inner.wind_z = sanitize_finite(wind_velocity[2]);
        inner.turbulence = sanitize_unit(turbulence);
        inner.drag = sanitize_non_negative(drag);
        inner.lift = sanitize_non_negative(lift);
        inner.air_density = sanitize_non_negative(air_density);
        inner.dt = dt;

        ClothAeroGatherQuery { inner }
    }

    /// The active particle count this query packs.
    #[must_use]
    pub fn count(&self) -> usize {
        self.inner.count as usize
    }
}

/// One resolved aero-gather response: the updated per-vertex velocities,
/// mirroring the velocities the golden
/// [`accumulate_aero_gather`](prism_render_architecture::cloth::aero_gather::accumulate_aero_gather)
/// writes back. Padding past the active vertex count stays at the input value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothAeroGatherResult {
    /// Updated per-vertex velocities, one `[x, y, z]` per vertex slot.
    pub velocities: [[f32; 3]; MAX_VERTS],
}

impl ClothAeroGatherResult {
    /// The updated velocity of vertex `index`.
    #[must_use]
    pub fn velocity(&self, index: usize) -> [f32; 3] {
        self.velocities[index]
    }
}

/// Encodes one [`ClothAeroGatherQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothAeroGatherQuery) -> GpuQuery {
    q.inner
}

/// Decodes one packed [`GpuResult`] into the public [`ClothAeroGatherResult`].
fn decode_result(raw: &GpuResult) -> ClothAeroGatherResult {
    let mut velocities = [[0.0f32; 3]; MAX_VERTS];
    for (vertex, out) in velocities.iter_mut().enumerate() {
        let base = vertex * 3;
        *out = [
            raw.velocities[base],
            raw.velocities[base + 1],
            raw.velocities[base + 2],
        ];
    }
    ClothAeroGatherResult { velocities }
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

/// A compiled, reusable per-vertex aero-gather compute pipeline, twinning the
/// stateless `Jacobi` gather of the `CPU` golden
/// [`accumulate_aero_gather`](prism_render_architecture::cloth::aero_gather::accumulate_aero_gather).
pub struct GpuClothAeroGather {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothAeroGather {
    /// Compiles the per-vertex aero-gather kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothAeroGather {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_aero_gather"),
            source: ShaderSource::Wgsl(CLOTH_AERO_GATHER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_aero_gather_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_aero_gather_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_aero_gather_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothAeroGather {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves the updated velocity field of every query in `queries` and
    /// returns one [`ClothAeroGatherResult`] per input, in order.
    ///
    /// The velocity field matches the reference to within floating-point
    /// tolerance. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothAeroGatherQuery],
    ) -> Vec<ClothAeroGatherResult> {
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
            label: Some("prism_volumetric_cloth_aero_gather_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_aero_gather_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_aero_gather_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_aero_gather_bind_group"),
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
            label: Some("prism_volumetric_cloth_aero_gather_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_aero_gather_encoder"),
        });
        {
            // One thread per vertex of each query, flattened to a 1-D dispatch
            // of count * MAX_VERTS threads.
            let total = (count as u32) * (MAX_VERTS as u32);
            let groups = total.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_aero_gather_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
