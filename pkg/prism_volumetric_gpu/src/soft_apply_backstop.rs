//! `wgpu` compute twin of the backstop clamp from the `CPU` golden
//! `prism_physics_core::soft::collision::body::apply_backstop`.
//!
//! A backstop keeps a soft-body particle from sinking too far behind a skinned
//! surface: given an anchor `origin`, an outward plane `normal` (not required to
//! be unit length) and a `distance`, the signed distance of the particle along
//! the unit normal, `s = n.dot(pos - origin)`, is clamped so it never drops
//! below `-distance`. When it does, the particle is pushed forward along the
//! unit normal onto the limiting plane; otherwise it holds. A (near) zero normal
//! has no defined plane, so the particle is returned untouched rather than
//! producing a `NaN`. This module models one particle as one query — the
//! position `pos`, the plane `origin` and `normal`, and the `distance` — and
//! ports that clamp onto the device: one thread resolves one particle, so a
//! passing real-device parity test is direct evidence the kernel takes the same
//! push/hold branch and lands on the same limiting plane the reference does.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the body of `apply_backstop`:
//! `len_sq = dot(normal, normal)`; if `len_sq <= 1e-12` the normal is inert and
//! the position is echoed; otherwise `n = normal / sqrt(len_sq)`,
//! `s = dot(n, pos - origin)`, `min_s = -distance`, and when `s < min_s` the
//! point moves to `pos + n * (min_s - s)`, else it holds. The slice iteration
//! and inverse-mass gating of the array pass are not twinned here; this kernel
//! is the pure per-particle clamp.
//!
//! # Correctness model
//!
//! The output is a position (`new_px`, `new_py`, `new_pz`), compared with an
//! absolute-or-relative tolerance because it is a continuous `f32` clamp, plus a
//! discrete `valid` word compared exactly. `valid` is `1` only when the particle
//! was actually pushed (a live normal and `s < min_s`); an inert normal or a
//! particle already on the front side is a legal echo and reports `valid = 0`.
//!
//! The degenerate guard and the push predicate are built from ordered compares
//! (`len_sq <= 1e-12` and `s < min_s`) and the result is chosen with `select`,
//! so there is no bare `f32` equality anywhere in the kernel. The divisor is
//! guarded with `select(1.0, len_sq, ok)` so the not-taken arm never forms an
//! `inf`/`NaN` even on a `Metal` fast-math driver that folds `x == x` to `true`.
//!
//! # Degenerate inputs
//!
//! A (near) zero normal (`len_sq <= 1e-12`) echoes the position with
//! `valid = 0`. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, `sqrt`, and `+ - *` on `f32`/`vec3<f32>` — with no `sin`, `cos`,
//! `tan`, `exp`, `log`, `pow`, no `round`, no `f32` remainder, no bare `f32`
//! equality and no `u64`/`i64`/`f64`, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. All vectors are flattened to scalar `f32` lanes in the storage
//! buffers, so no `vec3` alignment rule can perturb the `std430` stride.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` backstop-clamp kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// body of the `CPU` golden `apply_backstop`; see the module documentation for
/// the algorithm.
const SOFT_APPLY_BACKSTOP_WGSL: &str = r#"
// Backstop clamp twin: one thread per query reproduces the body of
// apply_backstop. The particle is pushed forward onto the limiting plane only
// when the plane normal is live (len_sq > 1e-12) and the signed distance along
// the unit normal has dropped below -distance; otherwise the position is echoed.
// It uses only the portable core-WGSL subset (ordered compares, select, sqrt,
// + - * on f32/vec3), takes no optional feature and has no loop.

struct Params {
    // Number of valid queries in this dispatch.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle position.
    px: f32,
    py: f32,
    pz: f32,
    // Backstop anchor point.
    ox: f32,
    oy: f32,
    oz: f32,
    // Outward plane normal (need not be unit length).
    nx: f32,
    ny: f32,
    nz: f32,
    // How far behind origin (along -normal) the particle may travel.
    distance: f32,
}

struct Result {
    // Clamped particle position.
    new_px: f32,
    new_py: f32,
    new_pz: f32,
    // 1 when the particle was pushed, 0 when the position was echoed.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let pos = vec3<f32>(q.px, q.py, q.pz);
    let origin = vec3<f32>(q.ox, q.oy, q.oz);
    let normal = vec3<f32>(q.nx, q.ny, q.nz);

    let len_sq = dot(normal, normal);
    // Ordered guard; no bare x == x so a fast-math driver cannot fold it.
    let ok = len_sq > 1e-12;
    // Guarded divisor: the not-taken arm divides by 1.0, never by ~0, so no
    // inf/NaN leaks into the select.
    let safe_len_sq = select(1.0, len_sq, ok);
    let n = normal * (1.0 / sqrt(safe_len_sq));

    let s = dot(n, pos - origin);
    let min_s = -q.distance;
    let pushed = s < min_s;

    let pushed_pos = pos + n * (min_s - s);
    let moved = ok && pushed;
    // select(false_value, true_value, condition).
    let new_pos = select(pos, pushed_pos, moved);

    var res: Result;
    res.new_px = new_pos.x;
    res.new_py = new_pos.y;
    res.new_pz = new_pos.z;
    res.valid = select(0u, 1u, moved);
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SOFT_APPLY_BACKSTOP_WGSL`].
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
/// All ten lanes are scalar `f32`, so the layout is a flat `40`-byte stride with
/// alignment `4` and no internal padding, and a batch of two or more packs
/// contiguously with no `vec3` alignment rule to trip.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    px: f32,
    py: f32,
    pz: f32,
    ox: f32,
    oy: f32,
    oz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    distance: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Three position words plus one `valid` word give a flat `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    new_px: f32,
    new_py: f32,
    new_pz: f32,
    valid: u32,
}

/// One query for the backstop-clamp twin: a particle position
/// (`px`, `py`, `pz`), the backstop anchor (`ox`, `oy`, `oz`), the outward plane
/// `normal` (`nx`, `ny`, `nz`, need not be unit length) and the `distance` the
/// particle may travel behind the anchor.
///
/// The whole twinned per-particle clamp is driven by this one tuple, so a single
/// query exercises the inert-normal guard, the already-ahead echo and the
/// push-back onto the limiting plane at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftApplyBackstopQuery {
    /// Particle position, x component.
    pub px: f32,
    /// Particle position, y component.
    pub py: f32,
    /// Particle position, z component.
    pub pz: f32,
    /// Backstop anchor point, x component.
    pub ox: f32,
    /// Backstop anchor point, y component.
    pub oy: f32,
    /// Backstop anchor point, z component.
    pub oz: f32,
    /// Outward plane normal, x component (need not be unit length).
    pub nx: f32,
    /// Outward plane normal, y component (need not be unit length).
    pub ny: f32,
    /// Outward plane normal, z component (need not be unit length).
    pub nz: f32,
    /// How far behind the anchor (along `-normal`) the particle may travel.
    pub distance: f32,
}

impl SoftApplyBackstopQuery {
    /// Builds a query from the particle position, backstop anchor, plane normal
    /// and travel distance, in field order.
    #[must_use]
    pub fn new(
        px: f32,
        py: f32,
        pz: f32,
        ox: f32,
        oy: f32,
        oz: f32,
        nx: f32,
        ny: f32,
        nz: f32,
        distance: f32,
    ) -> SoftApplyBackstopQuery {
        SoftApplyBackstopQuery {
            px,
            py,
            pz,
            ox,
            oy,
            oz,
            nx,
            ny,
            nz,
            distance,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `apply_backstop` per-particle clamp.
///
/// (`new_px`, `new_py`, `new_pz`) is the clamped position. `valid` is `1` only
/// when the particle was pushed forward onto the limiting plane; an inert normal
/// or a particle already on the front side echoes the position with `valid = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftApplyBackstopResult {
    /// Clamped particle position, x component.
    pub new_px: f32,
    /// Clamped particle position, y component.
    pub new_py: f32,
    /// Clamped particle position, z component.
    pub new_pz: f32,
    /// `1` when the particle was pushed, `0` when the position was echoed.
    pub valid: u32,
}

/// Encodes one [`SoftApplyBackstopQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SoftApplyBackstopQuery) -> GpuQuery {
    GpuQuery {
        px: q.px,
        py: q.py,
        pz: q.pz,
        ox: q.ox,
        oy: q.oy,
        oz: q.oz,
        nx: q.nx,
        ny: q.ny,
        nz: q.nz,
        distance: q.distance,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SoftApplyBackstopResult`].
fn decode_result(raw: &GpuResult) -> SoftApplyBackstopResult {
    SoftApplyBackstopResult {
        new_px: raw.new_px,
        new_py: raw.new_py,
        new_pz: raw.new_pz,
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

/// A compiled, reusable backstop-clamp compute pipeline, twinning the `CPU`
/// golden `apply_backstop`.
pub struct GpuSoftApplyBackstop {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftApplyBackstop {
    /// Compiles the backstop-clamp kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftApplyBackstop {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_apply_backstop"),
            source: ShaderSource::Wgsl(SOFT_APPLY_BACKSTOP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_apply_backstop_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_apply_backstop_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_apply_backstop_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftApplyBackstop {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftApplyBackstopResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftApplyBackstopQuery],
    ) -> Vec<SoftApplyBackstopResult> {
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
            label: Some("prism_volumetric_soft_apply_backstop_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_apply_backstop_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_apply_backstop_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_apply_backstop_bind_group"),
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
            label: Some("prism_volumetric_soft_apply_backstop_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_apply_backstop_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_apply_backstop_pass"),
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
