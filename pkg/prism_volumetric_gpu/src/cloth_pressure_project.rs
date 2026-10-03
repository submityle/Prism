//! `wgpu` compute twin of the stateless pressure (closed-mesh volume) `XPBD`
//! projection inside the cloth pressure contract
//! ([`project_pressure`](prism_render_architecture::cloth::pressure::project_pressure)).
//!
//! The `CPU` golden
//! [`project_pressure`](prism_render_architecture::cloth::pressure::project_pressure)
//! sanitizes its [`PressureParams`](prism_render_architecture::cloth::pressure::PressureParams)
//! and performs one compliant `XPBD` step over a closed, outward-wound triangle
//! mesh with the Lagrange multiplier `lambda` fixed at zero: it accumulates the
//! signed enclosed volume `V = (1/6) Σ p0 · (p1 × p2)` and the per-vertex volume
//! gradient `∇ᵢ`, forms the volume error `C = V - overpressure · rest_volume`,
//! and distributes a mass-weighted correction `Δxᵢ = inv_massᵢ · Δλ · ∇ᵢ` with
//! `Δλ = -C / (Σ inv_massᵢ |∇ᵢ|² + compliance / dt²)` to the free particles.
//!
//! [`GpuClothPressureProject`] is the on-device twin of that closed-form step.
//! One thread solves one whole mesh: it replays the reference's triangle-order
//! volume and gradient accumulation, the parameter sanitize, the
//! `dt <= 0` / empty-mesh / near-zero-denominator short-circuits, the
//! out-of-range triangle skip, and the per-vertex position update, so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same correction the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one mesh the twin reproduces every particle's updated position. The host
//! pre-resolves the variable-length `ClothParticle` slice into fixed-length
//! `positions` and `inverse_masses` arrays and the triangle list into a
//! fixed-length index array plus a `triangle_count`, matching the reference's
//! structure-of-arrays conversion and its `positions.get` out-of-range skip. An
//! in-range pinned particle (`inverse_mass <= 0`) still contributes to the
//! volume and gradient yet never moves, exactly as the reference does.
//!
//! # What stays on the host
//!
//! The variable-length particle and triangle slices, the
//! [`ClothParticle`](prism_render_architecture::cloth::ClothParticle)
//! structure-of-arrays conversion, the accumulation of the running Lagrange
//! multiplier across multiple substeps, and any multi-substep Gauss-Seidel
//! sweep remain host work; the device sees only one mesh's fixed-length arrays
//! per query and runs a single projection with `lambda = 0`, matching the
//! render-layer `project_pressure` convention of one compliant pressure
//! projection per call.
//!
//! # Correctness model
//!
//! The updated positions thread through only `+ - * /`, `cross` and `dot` with
//! no `sqrt` and no transcendental, so the `CPU` and `GPU` agree to within the
//! documented tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`). The
//! degenerate branches — `dt <= 0`, an empty mesh (no particles or no
//! triangles), and a near-zero denominator (`denom < 1e-12`, a collapsed or
//! all-pinned rigid mesh) — are reproduced exactly so a fixture that lands in
//! any of them leaves the positions unchanged just as the reference does.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `cross`,
//! `dot`, `+ - * /`, bounded loops over the mesh and unsigned index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round`, no `sqrt`. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::pressure`；无第三方引擎源码或衍生代码。
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
/// used across this crate's one-thread-per-element kernels; here one element is
/// one whole mesh projection.
const WORKGROUP_SIZE: u32 = 64;

/// Maximum vertices a single mesh query carries. The unit-cube fixture uses `8`;
/// the generous cap leaves headroom for perturbed-cube sweeps while keeping the
/// fixed-length device arrays small.
const MAX_PARTICLES: usize = 16;

/// Maximum triangles a single mesh query carries. The unit cube uses `12`.
const MAX_TRIANGLES: usize = 24;

/// Flattened vertex-position array length (`MAX_PARTICLES * 3`).
const POS_LEN: usize = MAX_PARTICLES * 3;

/// Flattened triangle-index array length (`MAX_TRIANGLES * 3`).
const TRI_LEN: usize = MAX_TRIANGLES * 3;

/// The portable core-`WGSL` pressure projection kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`project_pressure`](prism_render_architecture::cloth::pressure::project_pressure)
/// closed-form step; see the module documentation for the algorithm.
const CLOTH_PRESSURE_PROJECT_WGSL: &str = r#"
// Pressure (closed-mesh volume) XPBD projection twin: one thread projects one
// whole mesh, mirroring the CPU golden `cloth::pressure::project_pressure`
// closed form (with lambda = 0) using only max, cross, dot and + - * /. It owns
// no variable-length particle/triangle slice and no multi-substep lambda
// accumulation; those stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::cloth::pressure；无第三方
// 引擎源码或衍生代码。

const INV_SIX: f32 = 1.0 / 6.0;
// f32::MIN_POSITIVE; the sanitize lower-bounds overpressure by this value.
const MIN_POSITIVE: f32 = 1.17549435e-38;
// EPS_LEN_SQ: a denominator at or below this is a degenerate / collapsed mesh.
const EPS_LEN_SQ: f32 = 0.000000000001;

struct Params {
    // Number of mesh queries in the storage arrays; threads past this return.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Flattened vertex positions [x0, y0, z0, x1, ...], MAX_PARTICLES * 3.
    positions: array<f32, 48>,
    // Per-vertex inverse mass; a non-positive value pins that vertex.
    inv_mass: array<f32, 16>,
    // Flattened triangle vertex indices, MAX_TRIANGLES * 3.
    tris: array<u32, 72>,
    // Number of valid vertices (<= MAX_PARTICLES).
    particle_count: u32,
    // Number of valid triangles (<= MAX_TRIANGLES).
    triangle_count: u32,
    // Raw rest volume (sanitized on device: NaN -> 0, else magnitude).
    rest_volume: f32,
    // Raw overpressure (sanitized: NaN -> 1, else max with MIN_POSITIVE).
    overpressure: f32,
    // Raw compliance (clamped non-negative on device, mirroring value()).
    compliance: f32,
    // Substep time; a non-positive dt is a no-op.
    dt: f32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // Flattened updated vertex positions, MAX_PARTICLES * 3.
    out_pos: array<f32, 48>,
    // Vertex count echoed back for host-side slicing.
    particle_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// NaN detection without an f32 `==`/`!=`: every ordered comparison with NaN is
// false, so a value is NaN exactly when it is neither >= 0 nor <= 0.
fn is_nan(v: f32) -> bool {
    return !(v >= 0.0 || v <= 0.0);
}

// Mirrors PressureParams::sanitized() for the rest volume: NaN -> 0, else |v|.
fn sanitize_rest(v: f32) -> f32 {
    if (is_nan(v)) {
        return 0.0;
    }
    return abs(v);
}

// Mirrors PressureParams::sanitized() for the overpressure: NaN -> 1, else the
// value clamped strictly positive.
fn sanitize_over(v: f32) -> f32 {
    if (is_nan(v)) {
        return 1.0;
    }
    return max(v, MIN_POSITIVE);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    var q = queries[idx];

    // Default: positions unchanged. Overwritten by the solved update below when
    // no short-circuit fires, matching the reference early returns.
    var out: Result;
    out.out_pos = q.positions;
    out.particle_count = q.particle_count;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    let pcount = q.particle_count;
    let tcount = q.triangle_count;
    let dt = q.dt;

    // No-op when dt <= 0, there are no particles, or there are no triangles.
    if (dt <= 0.0 || pcount == 0u || tcount == 0u) {
        results[idx] = out;
        return;
    }

    let rest = sanitize_rest(q.rest_volume);
    let over = sanitize_over(q.overpressure);
    let tgt = over * rest;
    let compliance = max(q.compliance, 0.0);

    // Per-vertex volume gradient, flattened; zero-initialized explicitly.
    var grad: array<f32, 48>;
    for (var i = 0u; i < 48u; i = i + 1u) {
        grad[i] = 0.0;
    }

    // Pass 1: accumulate signed volume and per-vertex gradient in triangle
    // order. Out-of-range triangles are skipped; pinned vertices still
    // contribute, exactly as the reference does.
    var volume = 0.0;
    for (var t = 0u; t < tcount; t = t + 1u) {
        let base = t * 3u;
        let i0 = q.tris[base];
        let i1 = q.tris[base + 1u];
        let i2 = q.tris[base + 2u];
        if (i0 >= pcount || i1 >= pcount || i2 >= pcount) {
            continue;
        }
        let a0 = i0 * 3u;
        let a1 = i1 * 3u;
        let a2 = i2 * 3u;
        let p0 = vec3<f32>(q.positions[a0], q.positions[a0 + 1u], q.positions[a0 + 2u]);
        let p1 = vec3<f32>(q.positions[a1], q.positions[a1 + 1u], q.positions[a1 + 2u]);
        let p2 = vec3<f32>(q.positions[a2], q.positions[a2 + 1u], q.positions[a2 + 2u]);
        let c12 = cross(p1, p2);
        let c20 = cross(p2, p0);
        let c01 = cross(p0, p1);
        volume = volume + dot(p0, c12);
        grad[a0] = grad[a0] + c12.x * INV_SIX;
        grad[a0 + 1u] = grad[a0 + 1u] + c12.y * INV_SIX;
        grad[a0 + 2u] = grad[a0 + 2u] + c12.z * INV_SIX;
        grad[a1] = grad[a1] + c20.x * INV_SIX;
        grad[a1 + 1u] = grad[a1 + 1u] + c20.y * INV_SIX;
        grad[a1 + 2u] = grad[a1 + 2u] + c20.z * INV_SIX;
        grad[a2] = grad[a2] + c01.x * INV_SIX;
        grad[a2 + 1u] = grad[a2 + 1u] + c01.y * INV_SIX;
        grad[a2 + 2u] = grad[a2 + 2u] + c01.z * INV_SIX;
    }
    volume = volume * INV_SIX;

    let c = volume - tgt;

    // Denominator Σ inv_massᵢ · |gradᵢ|² over free vertices, in vertex order.
    var denom = 0.0;
    for (var i = 0u; i < pcount; i = i + 1u) {
        let w = q.inv_mass[i];
        if (w <= 0.0) {
            continue;
        }
        let b = i * 3u;
        let gx = grad[b];
        let gy = grad[b + 1u];
        let gz = grad[b + 2u];
        denom = denom + w * (gx * gx + gy * gy + gz * gz);
    }
    let alpha_tilde = compliance / (dt * dt);
    denom = denom + alpha_tilde;

    // Near-zero denominator: a no-op so the update never divides by zero.
    if (denom < EPS_LEN_SQ) {
        results[idx] = out;
        return;
    }

    // lambda starts at zero, so Δλ = (-c - alpha_tilde * 0) / denom = -c / denom.
    let d_lambda = -c / denom;
    for (var i = 0u; i < pcount; i = i + 1u) {
        let w = q.inv_mass[i];
        if (w <= 0.0) {
            continue;
        }
        let b = i * 3u;
        let coeff = w * d_lambda;
        out.out_pos[b] = q.positions[b] + grad[b] * coeff;
        out.out_pos[b + 1u] = q.positions[b + 1u] + grad[b + 1u] * coeff;
        out.out_pos[b + 2u] = q.positions[b + 2u] + grad[b + 2u] * coeff;
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the mesh-query count plus three pad
/// words to fill a `16`-byte uniform struct matching `Params` in
/// [`CLOTH_PRESSURE_PROJECT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid mesh queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one mesh query, matching the `WGSL` `Query`
/// struct: the flattened vertex positions, per-vertex inverse masses, flattened
/// triangle indices, the vertex and triangle counts, the raw pressure
/// parameters and `dt`, plus two pad words to a `16`-byte-multiple stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Flattened vertex positions `[x0, y0, z0, x1, ...]`.
    positions: [f32; POS_LEN],
    /// Per-vertex inverse mass; non-positive pins the vertex.
    inv_mass: [f32; MAX_PARTICLES],
    /// Flattened triangle vertex indices.
    tris: [u32; TRI_LEN],
    /// Number of valid vertices.
    particle_count: u32,
    /// Number of valid triangles.
    triangle_count: u32,
    /// Raw rest volume (sanitized on device).
    rest_volume: f32,
    /// Raw overpressure (sanitized on device).
    overpressure: f32,
    /// Raw compliance (clamped non-negative on device).
    compliance: f32,
    /// Substep time.
    dt: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one mesh result, matching the `WGSL` `Result`
/// struct: the flattened updated vertex positions, the echoed vertex count, and
/// three pad words to a `16`-byte-multiple stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Flattened updated vertex positions.
    out_pos: [f32; POS_LEN],
    /// Vertex count echoed back for host-side slicing.
    particle_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One pressure projection query: a closed, outward-wound triangle mesh plus its
/// pressure parameters and substep.
///
/// The host owns the variable-length particle and triangle slices and the
/// multi-substep Lagrange accumulation; it pre-resolves one mesh into
/// `positions` (one `[x, y, z]` per vertex), `inverse_masses` (a non-positive
/// value pins that vertex), and `triangles` (outward-wound vertex indices).
/// `rest_volume`, `overpressure` and `compliance` are the raw
/// [`PressureParams`](prism_render_architecture::cloth::pressure::PressureParams)
/// fields; the device sanitizes them exactly like the reference `sanitized`.
/// At most [`MAX_PARTICLES`](self) vertices and [`MAX_TRIANGLES`](self)
/// triangles are honored; any excess is ignored.
#[derive(Clone, Debug, PartialEq)]
pub struct ClothPressureProjectQuery {
    /// One world-space position `[x, y, z]` per vertex.
    pub positions: Vec<[f32; 3]>,
    /// One inverse mass per vertex; a non-positive value pins the vertex.
    pub inverse_masses: Vec<f32>,
    /// Outward-wound triangle vertex indices into `positions`.
    pub triangles: Vec<[u32; 3]>,
    /// Raw rest volume; sanitized to its magnitude (`NaN` becomes `0`).
    pub rest_volume: f32,
    /// Raw overpressure; sanitized strictly positive (`NaN` becomes `1`).
    pub overpressure: f32,
    /// Raw compliance; clamped non-negative to mirror the reference.
    pub compliance: f32,
    /// Substep time `dt`; a non-positive value makes the projection a no-op.
    pub dt: f32,
}

/// One resolved pressure projection, mirroring the reference
/// [`project_pressure`](prism_render_architecture::cloth::pressure::project_pressure)
/// outcome: the updated world-space position of every vertex.
///
/// `positions` holds the solved vertex positions in input order; out-of-range,
/// pinned, or short-circuited vertices are returned unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct ClothPressureProjectResult {
    /// The updated world-space position `[x, y, z]` of every vertex.
    pub positions: Vec<[f32; 3]>,
}

/// Encodes one [`ClothPressureProjectQuery`] into its `std430` [`GpuQuery`]
/// slot, flattening and zero-padding the vertex and triangle arrays to the
/// fixed device capacities.
fn encode_query(q: &ClothPressureProjectQuery) -> GpuQuery {
    let mut positions = [0.0f32; POS_LEN];
    let pcount = q.positions.len().min(MAX_PARTICLES);
    for (slot, p) in q.positions.iter().take(MAX_PARTICLES).enumerate() {
        let base = slot * 3;
        positions[base] = p[0];
        positions[base + 1] = p[1];
        positions[base + 2] = p[2];
    }
    let mut inv_mass = [0.0f32; MAX_PARTICLES];
    for (slot, m) in q.inverse_masses.iter().take(MAX_PARTICLES).enumerate() {
        inv_mass[slot] = *m;
    }
    let mut tris = [0u32; TRI_LEN];
    let tcount = q.triangles.len().min(MAX_TRIANGLES);
    for (slot, tri) in q.triangles.iter().take(MAX_TRIANGLES).enumerate() {
        let base = slot * 3;
        tris[base] = tri[0];
        tris[base + 1] = tri[1];
        tris[base + 2] = tri[2];
    }
    GpuQuery {
        positions,
        inv_mass,
        tris,
        particle_count: pcount as u32,
        triangle_count: tcount as u32,
        rest_volume: q.rest_volume,
        overpressure: q.overpressure,
        compliance: q.compliance,
        dt: q.dt,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`ClothPressureProjectResult`], unflattening the first `particle_count`
/// vertex positions.
fn decode_result(raw: &GpuResult) -> ClothPressureProjectResult {
    let count = (raw.particle_count as usize).min(MAX_PARTICLES);
    let mut positions = Vec::with_capacity(count);
    for slot in 0..count {
        let base = slot * 3;
        positions.push([
            raw.out_pos[base],
            raw.out_pos[base + 1],
            raw.out_pos[base + 2],
        ]);
    }
    ClothPressureProjectResult { positions }
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

/// A compiled, reusable pressure projection compute pipeline, twinning the `CPU`
/// golden
/// [`project_pressure`](prism_render_architecture::cloth::pressure::project_pressure).
pub struct GpuClothPressureProject {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothPressureProject {
    /// Compiles the pressure projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothPressureProject {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_pressure_project"),
            source: ShaderSource::Wgsl(CLOTH_PRESSURE_PROJECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_pressure_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_pressure_project_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_pressure_project_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothPressureProject {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects every mesh in `queries` and returns one
    /// [`ClothPressureProjectResult`] per input, in order.
    ///
    /// The updated positions equal the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothPressureProjectQuery],
    ) -> Vec<ClothPressureProjectResult> {
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
            label: Some("prism_volumetric_cloth_pressure_project_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_pressure_project_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_pressure_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_pressure_project_bind_group"),
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
            label: Some("prism_volumetric_cloth_pressure_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_pressure_project_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_pressure_project_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per mesh, flattened to a 1-D dispatch.
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
