//! `wgpu` compute twin of the artist-painted backstop projection from the
//! `CPU` golden `prism_render_architecture::cloth::painted::apply_painted_backstop`.
//!
//! A painted garment anchors each vertex to a skinned reference pose and sinks a
//! cushion sphere one radius *behind* that anchor along its outward normal, so
//! the cloth cannot pass through the character body. A vertex caught inside the
//! cushion is pushed radially back onto its surface; everything else is left
//! untouched. This module ports that single stateless per-vertex projection onto
//! the device: one thread resolves one vertex, so a passing real-device parity
//! test is direct evidence the ported kernel takes the same pinned / disabled /
//! degenerate-normal / outside-the-sphere branch the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `apply_painted_backstop` composed with
//! the `PaintedConstraint::clamped` backstop clamp and `Vec3::normalize_or_zero`:
//! the authored `backstop` is sanitized (`NaN -> 0`, otherwise `max(.., 0)`), a
//! non-positive result or a (near-)zero anchor normal disables the pass, the
//! cushion centre is placed at `anchor - normalize(normal) * backstop`, and a
//! vertex whose squared distance to that centre is below `radius^2` is projected
//! radially onto the sphere (or, when it sits exactly at the centre, pushed out
//! along the normal). There is no loop: each thread performs a fixed, bounded
//! sequence of arithmetic plus one `sqrt`, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! Every quantity threads through multiplies, adds, a guarded division and
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous position component. The discrete `valid` flag is compared
//! exactly.
//!
//! # Degenerate inputs
//!
//! A pinned vertex, a sanitized `backstop <= 0`, an anchor normal whose squared
//! length is at or below `EPS_LEN_SQ`, and a vertex at or outside the cushion
//! (`dist_sq >= radius^2`) are all no-ops: the twin reports `valid = 0` and
//! echoes the input position. A vertex exactly at the cushion centre
//! (`dist_sq <= EPS_LEN_SQ`) takes the normal-direction fallback. Fixtures and
//! the sweep keep configurations away from the `dist_sq == radius^2` surface and
//! the `nlen_sq == EPS_LEN_SQ` knee so a last-bit difference cannot flip a
//! branch. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `select`,
//! `sqrt`, `dot`, `+ - * /` and unsigned index arithmetic — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, `pow`, no `round`, no `copysign`, no `f32`
//! remainder and no optional device feature, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The only `f32` equality is the `NaN` self-test
//! `!(x == x)`; every other comparison is ordered.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::painted::apply_painted_backstop`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` painted-backstop kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `apply_painted_backstop` branch for branch; see the module
/// documentation for the algorithm.
const CLOTH_PAINTED_BACKSTOP_WGSL: &str = r#"
// Painted-backstop twin: one thread per vertex pushes a vertex out of the
// cushion sphere centred one radius behind its skinned anchor along the anchor
// normal, exactly as apply_painted_backstop (with PaintedConstraint::clamped and
// Vec3::normalize_or_zero) does. It mirrors the CPU golden branch for branch,
// uses only the portable core-WGSL subset (max/select/sqrt/dot and + - * / plus
// unsigned index math), takes no optional feature, and has no loop, so the
// kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::cloth::painted::apply_painted_backstop；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Simulated vertex position, world space.
    px: f32,
    py: f32,
    pz: f32,
    // Non-zero when the vertex is pinned (the pass is skipped).
    pinned: u32,
    // Skinned anchor position, world space.
    ax: f32,
    ay: f32,
    az: f32,
    // Authored backstop radius, pre-clamp (may be negative or NaN).
    backstop: f32,
    // Outward anchor normal, world space (may be zero to disable).
    nx: f32,
    ny: f32,
    nz: f32,
    // Padding word so the struct stride stays a clean 48 bytes.
    pad: f32,
}

struct Result {
    // Resolved position: the projection when valid, else the input echoed back.
    rx: f32,
    ry: f32,
    rz: f32,
    // 1 when the vertex was projected onto the cushion, 0 for a no-op branch.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Squared-length floor matching the reference cloth EPS_LEN_SQ, used both to
// reject a degenerate anchor normal and to detect a vertex at the cushion
// centre.
const EPS_LEN_SQ: f32 = 1e-12;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let particle = vec3<f32>(q.px, q.py, q.pz);
    let anchor = vec3<f32>(q.ax, q.ay, q.az);
    let anchor_normal = vec3<f32>(q.nx, q.ny, q.nz);

    // Default: a no-op that echoes the input position with valid = 0.
    var out: Result;
    out.rx = particle.x;
    out.ry = particle.y;
    out.rz = particle.z;
    out.valid = 0u;

    // PaintedConstraint::clamped().backstop: NaN collapses to 0, otherwise the
    // authored value is floored at 0. The NaN self-test !(x == x) is the only
    // f32 equality in the kernel.
    let bs_nan = !(q.backstop == q.backstop);
    let bs = select(max(q.backstop, 0.0), 0.0, bs_nan);

    if (q.pinned != 0u) {
        results[idx] = out;
        return;
    }
    if (bs <= 0.0) {
        results[idx] = out;
        return;
    }

    // Vec3::normalize_or_zero guard: a (near-)zero normal disables the pass.
    let nlen_sq = dot(anchor_normal, anchor_normal);
    if (nlen_sq <= EPS_LEN_SQ) {
        results[idx] = out;
        return;
    }
    let inv = 1.0 / sqrt(nlen_sq);
    let normal = anchor_normal * inv;

    let center = anchor - normal * bs;
    let rel = particle - center;
    let dist_sq = dot(rel, rel);
    let radius = bs;
    if (dist_sq >= radius * radius) {
        // On or outside the cushion surface: leave the vertex untouched.
        results[idx] = out;
        return;
    }

    if (dist_sq > EPS_LEN_SQ) {
        let dist = sqrt(dist_sq);
        let proj = center + rel * (radius / dist);
        out.rx = proj.x;
        out.ry = proj.y;
        out.rz = proj.z;
    } else {
        // Degenerate: the vertex sits at the cushion centre. Push it out along
        // the anchor normal to the nearest surface point (the anchor).
        let proj = center + normal * radius;
        out.rx = proj.x;
        out.ry = proj.y;
        out.rz = proj.z;
    }
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
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
/// Every `vec3` input is flattened to scalar lanes so the layout never trips a
/// `16`-byte vector-alignment rule; the kernel rebuilds each `vec3<f32>`. The
/// trailing `pad` keeps the stride a clean `48` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    px: f32,
    py: f32,
    pz: f32,
    pinned: u32,
    ax: f32,
    ay: f32,
    az: f32,
    backstop: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    pad: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the resolved position and the validity flag, `16` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    rx: f32,
    ry: f32,
    rz: f32,
    valid: u32,
}

/// One painted-backstop query: a simulated vertex, its pinned flag, the skinned
/// anchor pose, the authored (pre-clamp) backstop radius and the outward anchor
/// normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothPaintedBackstopQuery {
    /// Simulated vertex position, world space.
    pub particle_pos: [f32; 3],
    /// Non-zero when the vertex is pinned, which disables the pass.
    pub pinned: u32,
    /// Skinned anchor position, world space.
    pub anchor_pos: [f32; 3],
    /// Authored backstop radius before `clamped` sanitizes it (`NaN -> 0`,
    /// otherwise `max(.., 0)`).
    pub backstop: f32,
    /// Outward anchor normal, world space; a (near-)zero normal disables the
    /// pass.
    pub anchor_normal: [f32; 3],
}

impl ClothPaintedBackstopQuery {
    /// Builds a query from the vertex, its pinned flag, the anchor pose, the
    /// authored backstop and the anchor normal.
    #[must_use]
    pub fn new(
        particle_pos: [f32; 3],
        pinned: u32,
        anchor_pos: [f32; 3],
        backstop: f32,
        anchor_normal: [f32; 3],
    ) -> ClothPaintedBackstopQuery {
        ClothPaintedBackstopQuery {
            particle_pos,
            pinned,
            anchor_pos,
            backstop,
            anchor_normal,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `apply_painted_backstop` output for that vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothPaintedBackstopResult {
    /// The resolved position: the radial projection when the vertex was inside
    /// the cushion, otherwise the input position echoed unchanged.
    pub position: [f32; 3],
    /// `1` when the vertex was projected onto the cushion surface, `0` for any
    /// no-op branch (pinned, disabled, degenerate normal, or already outside).
    pub valid: u32,
}

/// Encodes one [`ClothPaintedBackstopQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothPaintedBackstopQuery) -> GpuQuery {
    GpuQuery {
        px: q.particle_pos[0],
        py: q.particle_pos[1],
        pz: q.particle_pos[2],
        pinned: q.pinned,
        ax: q.anchor_pos[0],
        ay: q.anchor_pos[1],
        az: q.anchor_pos[2],
        backstop: q.backstop,
        nx: q.anchor_normal[0],
        ny: q.anchor_normal[1],
        nz: q.anchor_normal[2],
        pad: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ClothPaintedBackstopResult`].
fn decode_result(raw: &GpuResult) -> ClothPaintedBackstopResult {
    ClothPaintedBackstopResult {
        position: [raw.rx, raw.ry, raw.rz],
        valid: raw.valid,
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

/// A compiled, reusable painted-backstop compute pipeline, twinning the `CPU`
/// golden `apply_painted_backstop`.
pub struct GpuClothPaintedBackstop {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothPaintedBackstop {
    /// Compiles the painted-backstop kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothPaintedBackstop {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_painted_backstop"),
            source: ShaderSource::Wgsl(CLOTH_PAINTED_BACKSTOP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_painted_backstop_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_painted_backstop_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_painted_backstop_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothPaintedBackstop {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ClothPaintedBackstopResult`] per input, in order.
    ///
    /// Each continuous position component matches the reference to within the
    /// tolerance documented on this module; the `valid` flag matches exactly. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothPaintedBackstopQuery],
    ) -> Vec<ClothPaintedBackstopResult> {
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
            label: Some("prism_volumetric_cloth_painted_backstop_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_painted_backstop_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_painted_backstop_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_painted_backstop_bind_group"),
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
            label: Some("prism_volumetric_cloth_painted_backstop_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_painted_backstop_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_painted_backstop_pass"),
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
