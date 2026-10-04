//! `wgpu` compute twin of the shape-goal pull-back from the `CPU` golden
//! `prism_physics_core::soft::constraint::linear_stiffness::project_pull_to_target`.
//!
//! `TressFX`/groom-style strand solvers keep hair close to an authored shape by
//! pulling each free particle a fraction of the way toward its geometric target
//! every step. The reference sweeps a slice of particles: when the whole-sweep
//! `stiffness` is positive, each particle whose inverse mass is strictly
//! positive is nudged `stiffness` of the way to its target, so a `stiffness` of
//! `1` snaps it exactly onto the target; pinned particles (inverse mass `0`) and
//! a non-positive `stiffness` leave the position unchanged. This module models
//! one particle as one query — the particle position `p`, its inverse mass `w`,
//! its `target`, and the shared `stiffness` — and ports the per-particle blend
//! onto the device: one thread resolves one particle, so a passing real-device
//! parity test is direct evidence the kernel takes the same move/no-op branch
//! and computes the same geometric mix the reference does.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the per-particle body of
//! `project_pull_to_target`: `moved = (w > 0) && (stiffness > 0)`, and
//! `new_p = moved ? p + (target - p) * stiffness : p`. The slice iteration, the
//! short-`targets` skip and the sequential sweep bookkeeping are not twinned
//! here; this kernel is the pure per-particle geometric pull.
//!
//! # Correctness model
//!
//! The output is a position (`new_px`, `new_py`, `new_pz`), compared with an
//! absolute-or-relative tolerance because it is a continuous `f32` blend, plus a
//! discrete `valid` word compared exactly. The golden has no division and no
//! rejected input: a pinned particle or a non-positive `stiffness` is a *legal
//! no-op*, not a degenerate case, so `valid` is always `1`; the field exists
//! only to match the shared three-piece shape across this crate's twins.
//!
//! The move predicate is built from ordered compares (`w > 0.0` and
//! `stiffness > 0.0`) and the result is chosen with `select`, so there is no
//! bare `f32` equality anywhere in the kernel.
//!
//! # Degenerate inputs
//!
//! None: every query produces a defined position. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, and `+ - *` on `f32`/`vec3<f32>` — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no `f32` remainder, no bare `f32` equality
//! and no `u64`/`i64`/`f64`, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. All vectors are flattened to scalar `f32` lanes in the storage
//! buffers, so no `vec3` alignment rule can perturb the `std430` stride.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::linear_stiffness`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` shape-goal pull-back kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the per-particle body of the `CPU` golden `project_pull_to_target`; see the
/// module documentation for the algorithm.
const SOFT_PULL_TO_TARGET_WGSL: &str = r#"
// Shape-goal pull-back twin: one thread per query reproduces the per-particle
// body of project_pull_to_target. A particle moves only when it is free
// (inverse mass w > 0) and the stiffness fraction is positive; a moving
// particle is blended stiffness of the way toward its target, so stiffness = 1
// snaps it onto the target and stiffness = 0.5 moves it halfway. Pinned
// particles (w <= 0) and a non-positive stiffness hold position. It uses only
// the portable core-WGSL subset (ordered compares, select, + - * on f32/vec3),
// takes no optional feature and has no loop.

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
    // Inverse mass: strictly positive means free to move, otherwise pinned.
    w: f32,
    // Geometric shape target.
    tx: f32,
    ty: f32,
    tz: f32,
    // Fraction of the way to pull toward the target this step.
    stiffness: f32,
}

struct Result {
    // Pulled-back particle position.
    new_px: f32,
    new_py: f32,
    new_pz: f32,
    // Always 1: the golden has no rejected input.
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

    let p = vec3<f32>(q.px, q.py, q.pz);
    let tgt = vec3<f32>(q.tx, q.ty, q.tz);

    // A particle moves only when it is free and the stiffness is positive;
    // both are ordered compares, so there is no bare f32 equality.
    let moved = (q.w > 0.0) && (q.stiffness > 0.0);
    // select(false_value, true_value, condition).
    let pulled = select(p, p + (tgt - p) * q.stiffness, moved);

    var res: Result;
    res.new_px = pulled.x;
    res.new_py = pulled.y;
    res.new_pz = pulled.z;
    res.valid = 1u;
    results[idx] = res;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SOFT_PULL_TO_TARGET_WGSL`].
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
/// All eight lanes are scalar `f32`, so the layout is a flat `32`-byte stride
/// (already a `16`-byte multiple) with no vector alignment rule to trip, and a
/// batch of two or more packs contiguously.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    px: f32,
    py: f32,
    pz: f32,
    w: f32,
    tx: f32,
    ty: f32,
    tz: f32,
    stiffness: f32,
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

/// One query for the shape-goal pull-back twin: a particle position
/// (`px`, `py`, `pz`), its inverse mass `w`, its geometric `target`
/// (`tx`, `ty`, `tz`), and the shared `stiffness` fraction.
///
/// The whole twinned per-particle blend is driven by this one tuple, so a
/// single query exercises the free/pinned predicate, the non-positive-stiffness
/// no-op and the geometric mix at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftPullToTargetQuery {
    /// Particle position, x component.
    pub px: f32,
    /// Particle position, y component.
    pub py: f32,
    /// Particle position, z component.
    pub pz: f32,
    /// Inverse mass: strictly positive means free to move, otherwise pinned.
    pub w: f32,
    /// Geometric shape target, x component.
    pub tx: f32,
    /// Geometric shape target, y component.
    pub ty: f32,
    /// Geometric shape target, z component.
    pub tz: f32,
    /// Fraction of the way to pull toward the target this step.
    pub stiffness: f32,
}

impl SoftPullToTargetQuery {
    /// Builds a query from the particle position, inverse mass, target and
    /// stiffness, in field order.
    #[must_use]
    pub fn new(
        px: f32,
        py: f32,
        pz: f32,
        w: f32,
        tx: f32,
        ty: f32,
        tz: f32,
        stiffness: f32,
    ) -> SoftPullToTargetQuery {
        SoftPullToTargetQuery {
            px,
            py,
            pz,
            w,
            tx,
            ty,
            tz,
            stiffness,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `project_pull_to_target` per-particle update.
///
/// (`new_px`, `new_py`, `new_pz`) is the pulled-back position. `valid` is always
/// `1`: the golden has no rejected input, since a pinned particle or a
/// non-positive stiffness is a legal no-op rather than a degenerate case.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftPullToTargetResult {
    /// Pulled-back particle position, x component.
    pub new_px: f32,
    /// Pulled-back particle position, y component.
    pub new_py: f32,
    /// Pulled-back particle position, z component.
    pub new_pz: f32,
    /// Always `1`: the golden produces a defined position for every input.
    pub valid: u32,
}

/// Encodes one [`SoftPullToTargetQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SoftPullToTargetQuery) -> GpuQuery {
    GpuQuery {
        px: q.px,
        py: q.py,
        pz: q.pz,
        w: q.w,
        tx: q.tx,
        ty: q.ty,
        tz: q.tz,
        stiffness: q.stiffness,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SoftPullToTargetResult`].
fn decode_result(raw: &GpuResult) -> SoftPullToTargetResult {
    SoftPullToTargetResult {
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

/// A compiled, reusable shape-goal pull-back compute pipeline, twinning the
/// `CPU` golden `project_pull_to_target`.
pub struct GpuSoftPullToTarget {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftPullToTarget {
    /// Compiles the shape-goal pull-back kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftPullToTarget {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_pull_to_target"),
            source: ShaderSource::Wgsl(SOFT_PULL_TO_TARGET_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_pull_to_target_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_pull_to_target_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_pull_to_target_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftPullToTarget {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SoftPullToTargetResult`]
    /// per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftPullToTargetQuery],
    ) -> Vec<SoftPullToTargetResult> {
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
            label: Some("prism_volumetric_soft_pull_to_target_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_pull_to_target_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_pull_to_target_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_pull_to_target_bind_group"),
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
            label: Some("prism_volumetric_soft_pull_to_target_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_pull_to_target_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_pull_to_target_pass"),
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
