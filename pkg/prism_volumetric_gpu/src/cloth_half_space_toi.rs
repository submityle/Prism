//! `wgpu` compute twin of the cloth continuous-collision half-space
//! time-of-impact query from the `CPU` golden
//! `prism_physics_core::soft::collision::ccd::half_space_toi` (the render-layer
//! `prism_render_architecture::cloth::ccd` is a thin façade over this physics
//! closed form).
//!
//! A cloth particle sweeping from `prev` to `curr` across one substep may pass
//! through a one-sided plane (a floor, a wall, a collision proxy face). The
//! reference finds the earliest fraction `t` in `0..=1` at which the point
//! reaches the plane `normal · x = offset`, so a position-based solver can
//! clamp the move to that contact. The signed distance
//! `s(t) = normal · p(t) - offset` is linear in `t`; the crossing is where `s`
//! reaches zero while decreasing. A point that already starts behind the plane
//! reports `t = 0`; a (near) zero normal has no defined plane and misses; a
//! segment parallel to or receding from the plane misses; a crossing past the
//! end of the segment (`t > 1`) misses.
//!
//! This module ports that stateless, closed-form query onto the device: one
//! thread resolves one swept segment, emitting whether it hits and, if so, the
//! clamped time of impact. [`GpuClothHalfSpaceToi`] is the on-device twin: a
//! passing real-device parity test is direct evidence the kernel takes the same
//! ordered branches and the same linear-crossing arithmetic the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! * `half_space_toi(prev, curr, normal, offset)` — the whole closed form:
//!   the zero-normal guard, the `s0 <= 0` already-behind short-circuit returning
//!   `t = 0`, the `ds >= -EPS_COEF` parallel/receding guard, the crossing
//!   `t = -s0 / ds`, and the `t <= 1` acceptance clamped by `t.max(0)`.
//!
//! The capsule and cylinder sweeps in the same reference file are deliberately
//! *not* twinned here; they are separate queries.
//!
//! # Result encoding
//!
//! The reference returns `Option<f32>`. The twin flattens that into `hit`
//! (`1` for `Some`, `0` for `None`) and `t` (the clamped time of impact, `0`
//! when there is no hit). `hit = 1` with `t = 0` is a point that starts on or
//! behind the plane. `valid` is always `1`; it exists so the layout matches the
//! crate's other classifier twins and leaves room for future rejection paths.
//!
//! # Correctness model
//!
//! `hit` and `valid` are discrete and are compared exactly. The time of impact
//! `t` is the only continuous channel; it is compared with an absolute-or-
//! relative tolerance because the division `-s0 / ds` is a single floating-point
//! operation whose last bit can differ between the host `f32` and the device.
//! Fixtures and the random sweep keep samples clear of the `s0 = 0`, the
//! `ds = -EPS_COEF` and the `t = 1` knee points so the discrete `hit` channel
//! never flips on a rounding tie.
//!
//! # Degenerate inputs
//!
//! A normal whose squared length is at or below `EPS_LEN_SQ` (`1e-12`) has no
//! defined plane and misses. A segment whose along-normal delta `ds` is at or
//! above `-EPS_COEF` (`-1e-12`) is parallel to or receding from the plane and
//! misses. The division guards against a zero divisor by never dividing on a
//! branch that is not selected: the shader clamps the divisor away from zero on
//! the inactive arm with `min(ds, -EPS_COEF)` so no `NaN` or infinity can
//! propagate out of a `select`. An empty query batch short-circuits on the host
//! with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — ordered compares,
//! `select`, `min`, `max` and scalar arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `round`, no float modulo, no bare float equality and
//! no optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::ccd`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` cloth half-space time-of-impact kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden `half_space_toi` branch for branch; see the
/// module documentation for the algorithm.
const CLOTH_HALF_SPACE_TOI_WGSL: &str = r#"
// Cloth half-space TOI twin: one thread per swept segment reproduces the
// earliest crossing of the plane normal . x = offset by a point moving from
// prev to curr, mirroring ccd::half_space_toi.
// Provenance: 孪生自本仓 prism_physics_core::soft::collision::ccd；无第三方引擎源码或衍生代码。

// Squared-length floor below which the normal has no defined plane.
const EPS_LEN_SQ: f32 = 1e-12;
// Along-normal delta floor below which the segment is parallel/receding.
const EPS_COEF: f32 = 1e-12;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    prev_x: f32,
    prev_y: f32,
    prev_z: f32,
    curr_x: f32,
    curr_y: f32,
    curr_z: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    offset: f32,
    pad0: f32,
    pad1: f32,
}

struct Result {
    // 1 when the swept segment reaches the plane within 0..=1, else 0.
    hit: u32,
    // Clamped time of impact; 0 when there is no hit.
    toi: f32,
    // Always 1; present for layout parity with the crate's other twins.
    valid: u32,
    pad0: u32,
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

    let prev = vec3<f32>(q.prev_x, q.prev_y, q.prev_z);
    let curr = vec3<f32>(q.curr_x, q.curr_y, q.curr_z);
    let normal = vec3<f32>(q.normal_x, q.normal_y, q.normal_z);

    var out: Result;
    out.hit = 0u;
    out.toi = 0.0;
    out.valid = 1u;
    out.pad0 = 0u;

    let len_sq = dot(normal, normal);
    let has_plane = len_sq > EPS_LEN_SQ;

    let s0 = dot(normal, prev) - q.offset;
    let already_behind = s0 <= 0.0;

    let ds = dot(normal, curr - prev);
    let approaching = ds < -EPS_COEF;

    // Guard the divisor on the inactive arm so no NaN/Inf leaks through select.
    let safe_ds = min(ds, -EPS_COEF);
    let t_raw = -s0 / safe_ds;
    let t_clamped = max(t_raw, 0.0);
    let within_segment = t_raw <= 1.0;

    // Crossing branch: approaching the plane and the crossing lands in 0..=1.
    let crossing_hit = approaching && within_segment;
    // Combine: a point already behind hits at t = 0; otherwise it must cross.
    let hit_when_planed = already_behind || crossing_hit;
    let toi_when_planed = select(t_clamped, 0.0, already_behind);

    out.hit = select(0u, select(0u, 1u, hit_when_planed), has_plane);
    out.toi = select(0.0, select(0.0, toi_when_planed, hit_when_planed), has_plane);

    results[idx] = out;
}
"#;

/// `repr(C)` `std430` layout of the dispatch parameters.
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
/// Ten payload `f32` lanes plus two padding `f32` lanes keep the stride a flat
/// `48` bytes, a multiple of `16` with no vector-alignment surprises.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    prev_x: f32,
    prev_y: f32,
    prev_z: f32,
    curr_x: f32,
    curr_y: f32,
    curr_z: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    offset: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Three payload words plus one padding word keep the stride a flat
/// `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    hit: u32,
    toi: f32,
    valid: u32,
    pad0: u32,
}

/// One cloth half-space time-of-impact query: the swept endpoints `prev` and
/// `curr`, the plane normal, and the plane offset along that normal. The vector
/// fields are flattened to scalars so the `std430` stride stays an unambiguous
/// flat layout with no `vec3` alignment padding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothHalfSpaceToiQuery {
    /// `x` of the segment start position.
    pub prev_x: f32,
    /// `y` of the segment start position.
    pub prev_y: f32,
    /// `z` of the segment start position.
    pub prev_z: f32,
    /// `x` of the segment end position.
    pub curr_x: f32,
    /// `y` of the segment end position.
    pub curr_y: f32,
    /// `z` of the segment end position.
    pub curr_z: f32,
    /// `x` of the plane normal.
    pub normal_x: f32,
    /// `y` of the plane normal.
    pub normal_y: f32,
    /// `z` of the plane normal.
    pub normal_z: f32,
    /// Plane offset: the plane is `normal · x = offset`.
    pub offset: f32,
}

impl ClothHalfSpaceToiQuery {
    /// Builds a half-space TOI query from its swept endpoints, plane normal and
    /// offset.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query mirrors the golden prev, curr, normal and offset inputs verbatim as flat scalars"
    )]
    pub fn new(
        prev_x: f32,
        prev_y: f32,
        prev_z: f32,
        curr_x: f32,
        curr_y: f32,
        curr_z: f32,
        normal_x: f32,
        normal_y: f32,
        normal_z: f32,
        offset: f32,
    ) -> ClothHalfSpaceToiQuery {
        ClothHalfSpaceToiQuery {
            prev_x,
            prev_y,
            prev_z,
            curr_x,
            curr_y,
            curr_z,
            normal_x,
            normal_y,
            normal_z,
            offset,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `half_space_toi` `Option<f32>` flattened into a hit flag and a time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothHalfSpaceToiResult {
    /// `1` when the swept segment reaches the plane within `0..=1`, else `0`.
    pub hit: u32,
    /// Clamped time of impact in `0..=1`; `0` when there is no hit.
    pub t: f32,
    /// Always `1`; present for layout parity with the crate's other twins.
    pub valid: u32,
}

/// Encodes one [`ClothHalfSpaceToiQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothHalfSpaceToiQuery) -> GpuQuery {
    GpuQuery {
        prev_x: q.prev_x,
        prev_y: q.prev_y,
        prev_z: q.prev_z,
        curr_x: q.curr_x,
        curr_y: q.curr_y,
        curr_z: q.curr_z,
        normal_x: q.normal_x,
        normal_y: q.normal_y,
        normal_z: q.normal_z,
        offset: q.offset,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ClothHalfSpaceToiResult`].
fn decode_result(raw: &GpuResult) -> ClothHalfSpaceToiResult {
    ClothHalfSpaceToiResult {
        hit: raw.hit,
        t: raw.toi,
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

/// A compiled, reusable cloth half-space TOI compute pipeline, twinning the
/// `CPU` golden `half_space_toi`.
pub struct GpuClothHalfSpaceToi {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothHalfSpaceToi {
    /// Compiles the cloth half-space TOI kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothHalfSpaceToi {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_half_space_toi"),
            source: ShaderSource::Wgsl(CLOTH_HALF_SPACE_TOI_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_half_space_toi_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_half_space_toi_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_half_space_toi_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothHalfSpaceToi {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ClothHalfSpaceToiResult`] per input, in order.
    ///
    /// An empty `queries` batch returns an empty vector with no dispatch
    /// issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothHalfSpaceToiQuery],
    ) -> Vec<ClothHalfSpaceToiResult> {
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
            label: Some("prism_volumetric_cloth_half_space_toi_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_half_space_toi_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_half_space_toi_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_half_space_toi_bind_group"),
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
            label: Some("prism_volumetric_cloth_half_space_toi_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_half_space_toi_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_half_space_toi_pass"),
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
