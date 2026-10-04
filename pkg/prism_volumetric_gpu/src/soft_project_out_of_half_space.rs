//! `wgpu` compute twin of the half-space projection kernel from the `CPU`
//! golden
//! `prism_physics_core::soft::collision::body::project_out_of_half_space`.
//!
//! A half-space constraint keeps a particle on the feasible side of the plane
//! `normal . x == offset`. When the particle sits on the infeasible side
//! (`normal . pos < offset`) it is pushed along `normal` onto the plane; the
//! push length is scaled by `1 / |normal|^2` so the plane geometry is honoured
//! for any normal scale, not just a unit normal. One thread projects one
//! independent query and writes the resolved position plus a validity flag.
//!
//! [`GpuSoftProjectOutOfHalfSpace`] is the on-device twin; a passing real-device
//! parity test is direct evidence the ported kernel reproduces the same
//! squared-length guard, the same signed distance, the same feasible-side
//! early-out and the same `t = -signed / len_sq` projection the reference
//! computes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate: the
//! squared normal length `len_sq = normal . normal`; the degenerate zero-normal
//! rejection (`len_sq <= EPS_LEN_SQ`, echo `pos`, `valid = 0`); the signed plane
//! distance `signed = normal . pos - offset`; the feasible-side early-out
//! (`signed >= 0`, echo `pos`, `valid = 0`); the push parameter
//! `t = -signed / len_sq`; and the projected position `pos + normal * t`
//! (`valid = 1`). There is no loop: each thread performs a fixed, bounded
//! sequence of multiplies, adds, divides and selects, so the kernel provably
//! terminates.
//!
//! # Correctness model
//!
//! The projected position threads through a dot product, a divide and a
//! multiply-add, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous channel. The discrete `valid` flag is compared exactly: it
//! is `1` only when the particle is projected onto the plane (a non-degenerate
//! normal and an infeasible-side start) and `0` for either echo path, in which
//! case the position passes through unchanged.
//!
//! # Degenerate inputs
//!
//! A (near) zero normal has no defined plane, so when
//! `len_sq <= EPS_LEN_SQ = 1e-12` the point is echoed (`valid = 0`) rather than
//! dividing by zero. A particle already on the feasible side (`signed >= 0`,
//! including exactly on the plane) is left untouched (`valid = 0`). The divisor
//! `len_sq` is guarded with a unit fallback so the unselected arm cannot raise
//! an infinity before the valid gate drops it. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `dot`,
//! `select`, `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, no `round`,
//! no float `%`, and no `u64` / `i64` / `f64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The degeneracy tests use ordered comparisons feeding
//! `select`, which are robust under `Metal`'s fast-math (an `x == x` test would
//! be folded to `true`).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body`；无第三方
//! 引擎源码或衍生代码。
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

/// The portable core-`WGSL` half-space projection kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `project_out_of_half_space`; see the module docs for the
/// closed form it reproduces.
const SOFT_PROJECT_OUT_OF_HALF_SPACE_WGSL: &str = r#"
// Half-space projection twin: one thread projects one independent query, pushing
// a particle on the infeasible side of the plane `normal . x == offset` along
// `normal` onto the plane and writing the resolved position plus a validity
// flag. It mirrors the CPU golden exactly and uses only the portable core
// subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    // Padding words so the uniform struct fills 16 bytes.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Particle position.
    posx: f32,
    posy: f32,
    posz: f32,
    // Plane normal (need not be unit length).
    nx: f32,
    ny: f32,
    nz: f32,
    // Plane offset: the plane is `normal . x == offset`.
    offset: f32,
}

struct Soln {
    // Resolved particle position.
    new_posx: f32,
    new_posy: f32,
    new_posz: f32,
    // 1 when the particle was projected onto the plane, 0 when echoed.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Soln>;

// Squared length below which a normal has no defined plane, matching the golden
// crate::soft::collision::EPS_LEN_SQ.
const EPS_LEN_SQ: f32 = 1e-12;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let pos = vec3<f32>(q.posx, q.posy, q.posz);
    let normal = vec3<f32>(q.nx, q.ny, q.nz);

    let len_sq = dot(normal, normal);
    // Zero normal has no defined plane: echo the position.
    let normal_ok = len_sq > EPS_LEN_SQ;

    let signed = dot(normal, pos) - q.offset;
    // Already on the feasible side (including on the plane): echo.
    let infeasible = signed < 0.0;

    // Guard the divisor so the unselected arm cannot raise an infinity before
    // the valid gate drops it.
    let safe_len_sq = select(1.0, len_sq, normal_ok);
    let t_push = -signed / safe_len_sq;
    let projected = pos + normal * t_push;

    let accepted = normal_ok && infeasible;

    // Echo the position unchanged on either rejection.
    let out_pos = select(pos, projected, accepted);

    var out: Soln;
    out.new_posx = out_pos.x;
    out.new_posy = out_pos.y;
    out.new_posz = out_pos.z;
    out.valid = select(0u, 1u, accepted);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in the kernel.
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
/// Seven `f32` give a fixed `28`-byte stride with no trailing pad, since the
/// struct alignment is `4` and `28` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    posx: f32,
    posy: f32,
    posz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    offset: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Soln` struct.
/// Three `f32` plus one `u32` give a fixed `16`-byte stride with no pad, since
/// the struct alignment is `4` and `16` is already a multiple of it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    new_posx: f32,
    new_posy: f32,
    new_posz: f32,
    valid: u32,
}

/// One query for the half-space projection twin: the particle position, the
/// plane normal (any scale) and the plane offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftProjectOutOfHalfSpaceQuery {
    /// `x` of the particle position.
    pub posx: f32,
    /// `y` of the particle position.
    pub posy: f32,
    /// `z` of the particle position.
    pub posz: f32,
    /// `x` of the plane normal (need not be unit length).
    pub nx: f32,
    /// `y` of the plane normal (need not be unit length).
    pub ny: f32,
    /// `z` of the plane normal (need not be unit length).
    pub nz: f32,
    /// Plane offset: the plane is `normal . x == offset`.
    pub offset: f32,
}

impl SoftProjectOutOfHalfSpaceQuery {
    /// Builds a query from the particle position, the plane normal (any scale)
    /// and the plane offset.
    ///
    /// The position and normal are grouped into fixed-length arrays so the
    /// constructor stays within a small, readable argument count.
    #[must_use]
    pub fn new(pos: [f32; 3], normal: [f32; 3], offset: f32) -> SoftProjectOutOfHalfSpaceQuery {
        SoftProjectOutOfHalfSpaceQuery {
            posx: pos[0],
            posy: pos[1],
            posz: pos[2],
            nx: normal[0],
            ny: normal[1],
            nz: normal[2],
            offset,
        }
    }
}

/// One resolved answer for a single query: the resolved particle position and
/// the validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftProjectOutOfHalfSpaceResult {
    /// Resolved `x` of the particle position.
    pub new_posx: f32,
    /// Resolved `y` of the particle position.
    pub new_posy: f32,
    /// Resolved `z` of the particle position.
    pub new_posz: f32,
    /// `1` when the particle was projected onto the plane, `0` when echoed.
    pub valid: u32,
}

/// Encodes one [`SoftProjectOutOfHalfSpaceQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &SoftProjectOutOfHalfSpaceQuery) -> GpuQuery {
    GpuQuery {
        posx: q.posx,
        posy: q.posy,
        posz: q.posz,
        nx: q.nx,
        ny: q.ny,
        nz: q.nz,
        offset: q.offset,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`SoftProjectOutOfHalfSpaceResult`].
fn decode_result(raw: &GpuResult) -> SoftProjectOutOfHalfSpaceResult {
    SoftProjectOutOfHalfSpaceResult {
        new_posx: raw.new_posx,
        new_posy: raw.new_posy,
        new_posz: raw.new_posz,
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

/// A compiled, reusable half-space projection compute pipeline, twinning the
/// `CPU` golden `project_out_of_half_space`.
pub struct GpuSoftProjectOutOfHalfSpace {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSoftProjectOutOfHalfSpace {
    /// Compiles the half-space projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSoftProjectOutOfHalfSpace {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_half_space"),
            source: ShaderSource::Wgsl(SOFT_PROJECT_OUT_OF_HALF_SPACE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_half_space_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_half_space_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_half_space_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSoftProjectOutOfHalfSpace {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SoftProjectOutOfHalfSpaceResult`] per input, in order.
    ///
    /// The continuous channels match the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SoftProjectOutOfHalfSpaceQuery],
    ) -> Vec<SoftProjectOutOfHalfSpaceResult> {
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
            label: Some("prism_volumetric_soft_project_out_of_half_space_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_half_space_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_half_space_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_half_space_bind_group"),
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
            label: Some("prism_volumetric_soft_project_out_of_half_space_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_soft_project_out_of_half_space_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_soft_project_out_of_half_space_pass"),
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
