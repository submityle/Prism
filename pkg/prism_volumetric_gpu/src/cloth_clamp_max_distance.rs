//! `wgpu` compute twin of the painted max-distance clamp from the `CPU` golden
//! `prism_render_architecture::cloth::painted::clamp_max_distance` together with
//! `prism_render_architecture::cloth::asset::PaintedConstraint::clamped`.
//!
//! A painted garment caps how far each simulated vertex may drift from its
//! skinned anchor. Per particle the golden first sanitises the painted
//! `max_distance` (a `NaN` collapses to `0`, a negative value is lifted to `0`,
//! and the `+inf` "no cap" default is preserved), then applies a post-solve
//! positional clamp: a pinned vertex is left untouched, an uncapped (`+inf`)
//! vertex is left untouched, a vertex already inside its max-distance sphere is
//! left untouched, and an over-limit vertex is projected radially back onto the
//! sphere. When the drift is shorter than the length epsilon the vertex is
//! welded straight to the anchor instead of normalising a (near-)zero vector.
//! Velocity is not part of this pass. One thread solves one query.
//!
//! [`GpuClothClampMaxDistance`] is the on-device twin; a passing real-device
//! parity test is direct evidence the ported kernel reproduces the same
//! sanitisation, the same skip predicates and the same radial projection the
//! reference computes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate: the
//! `clamped` sanitiser `max_distance' = if max_distance.is_nan() { 0 } else
//! { max_distance.max(0) }`; the pinned skip; the non-finite (`+inf`) skip; the
//! inside-sphere skip (`dist_sq <= max_distance'^2`); the radial projection
//! `anchor + drift * (max_distance' / sqrt(dist_sq))` when the drift exceeds the
//! length epsilon; and the weld to `anchor` when `dist_sq <= EPS_LEN_SQ`. There
//! is no loop: each thread performs a fixed, bounded sequence of multiplies,
//! adds, a single `sqrt`, clamps and selects, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! The output position threads through subtracts, a dot product, a `sqrt` and a
//! divide, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous position channel. The discrete `valid` flag is compared
//! exactly; it is always `1`, since the clamp accepts every input and never
//! faults.
//!
//! # Degenerate inputs
//!
//! A `NaN` `max_distance` collapses to `0` (a weld-to-anchor pin); a `+inf`
//! `max_distance` leaves the vertex free; a negative `max_distance` is lifted to
//! `0`. A drift at or below the length epsilon `EPS_LEN_SQ = 1e-12` welds the
//! vertex to the anchor rather than normalising a vanishing vector; the kernel
//! guards the division denominator with `max(dist_sq, EPS_LEN_SQ)` so the
//! unselected projection arm can never raise an infinity that pollutes the
//! chosen result. An empty query batch short-circuits on the host with no
//! dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `dot`, `sqrt`,
//! `clamp`, `select`, `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`, no
//! `round`, no float `%`, and no `u64` / `i64` / `f64`, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. The non-finite and `NaN` tests avoid bare float
//! equality by using ordered comparisons against the largest finite `f32`
//! magnitude and the self-inequality `x == x`, both of which behave correctly
//! for `NaN` and the infinities.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::painted` 与
//! `prism_render_architecture::cloth::asset`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` painted max-distance clamp kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `clamp_max_distance` composed with
/// `PaintedConstraint::clamped`; see the module documentation.
const CLOTH_CLAMP_MAX_DISTANCE_WGSL: &str = r#"
// Painted max-distance clamp twin: one thread per query sanitises the painted
// max_distance, applies the pinned / uncapped / inside-sphere skips, and either
// projects an over-limit vertex radially onto its max-distance sphere or welds
// it to the anchor. It mirrors the CPU golden exactly and uses only the
// portable core subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    // Padding words so the uniform struct fills 16 bytes.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Post-solve particle position before the clamp.
    particle_x: f32,
    particle_y: f32,
    particle_z: f32,
    // Skinned anchor the drift is measured from.
    anchor_x: f32,
    anchor_y: f32,
    anchor_z: f32,
    // Raw painted max_distance; sanitised by the clamped() rule before use.
    max_distance: f32,
    // Non-zero when the vertex is pinned and must not move.
    pinned: u32,
}

struct Result {
    // Clamped particle position after the pass.
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    // Always 1: the clamp accepts every input, so no query faults.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Squared length below which a drift is treated as zero, matching the golden
// cloth EPS_LEN_SQ. Projecting such a drift would normalise a vanishing vector,
// so the vertex is welded straight to the anchor instead.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Largest finite f32 magnitude. A non-negative value is finite exactly when it
// is at most this bound: an infinity fails the ordered compare, reproducing
// is_finite without any bare float equality.
const FINITE_LIMIT: f32 = 3.40282347e38;

// The clamped() sanitiser for a painted max_distance: a NaN collapses to 0, a
// negative value is lifted to 0, and a finite non-negative value (including the
// +inf "no cap" default) is otherwise preserved. The NaN test uses the IEEE
// self-inequality rather than a bare equality against a literal.
fn clamped_max_distance(md: f32) -> f32 {
    let is_nan = !(md == md);
    return select(max(md, 0.0), 0.0, is_nan);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let particle = vec3<f32>(q.particle_x, q.particle_y, q.particle_z);
    let anchor = vec3<f32>(q.anchor_x, q.anchor_y, q.anchor_z);

    let md = clamped_max_distance(q.max_distance);
    // md is non-negative here, so a finite md satisfies md <= FINITE_LIMIT;
    // the +inf "no cap" default fails this and is treated as uncapped.
    let md_finite = md <= FINITE_LIMIT;
    let pinned = q.pinned != 0u;

    let drift = particle - anchor;
    let dist_sq = dot(drift, drift);
    let max_sq = md * md;

    // Guard the projection denominator so the unselected arm never raises an
    // infinity: a drift at or below EPS_LEN_SQ is welded to the anchor instead.
    let safe_dist_sq = max(dist_sq, EPS_LEN_SQ);
    let dist = sqrt(safe_dist_sq);
    let scale = md / dist;
    let projected = anchor + drift * scale;

    // Over-limit vertices project onto the sphere unless the drift is vanishing,
    // in which case they weld to the anchor.
    let over_limit_pos = select(anchor, projected, dist_sq > EPS_LEN_SQ);
    // Inside the sphere the vertex is left untouched.
    let clamped_pos = select(over_limit_pos, particle, dist_sq <= max_sq);
    // A pinned or uncapped vertex is skipped entirely.
    let skip = pinned || !md_finite;
    let final_pos = select(clamped_pos, particle, skip);

    var out: Result;
    out.pos_x = final_pos.x;
    out.pos_y = final_pos.y;
    out.pos_z = final_pos.z;
    out.valid = 1u;
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
/// Seven `f32` plus one `u32` give a fixed `32`-byte stride with no trailing
/// pad, since the eight `4`-byte scalars already fill a `16`-byte multiple.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    particle_x: f32,
    particle_y: f32,
    particle_z: f32,
    anchor_x: f32,
    anchor_y: f32,
    anchor_z: f32,
    max_distance: f32,
    pinned: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. Three `f32` plus one `u32` give a fixed `16`-byte stride with no pad.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    valid: u32,
}

/// One query for the painted max-distance clamp twin: the post-solve particle
/// position, its skinned anchor, the raw painted `max_distance`, and whether the
/// vertex is pinned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothClampMaxDistanceQuery {
    /// Post-solve particle position before the clamp.
    pub particle: [f32; 3],
    /// Skinned anchor the drift is measured from.
    pub anchor: [f32; 3],
    /// Raw painted `max_distance`; sanitised by the `clamped` rule before use.
    pub max_distance: f32,
    /// Non-zero when the vertex is pinned and must not move.
    pub pinned: u32,
}

impl ClothClampMaxDistanceQuery {
    /// Builds a query from the particle position, anchor, raw `max_distance` and
    /// pinned flag.
    #[must_use]
    pub fn new(
        particle: [f32; 3],
        anchor: [f32; 3],
        max_distance: f32,
        pinned: u32,
    ) -> ClothClampMaxDistanceQuery {
        ClothClampMaxDistanceQuery {
            particle,
            anchor,
            max_distance,
            pinned,
        }
    }
}

/// One resolved answer for a single query: the clamped particle position and the
/// validity flag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothClampMaxDistanceResult {
    /// Clamped particle position after the pass.
    pub position: [f32; 3],
    /// Always `1`: the clamp accepts every input, so no query faults.
    pub valid: u32,
}

/// Encodes one [`ClothClampMaxDistanceQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ClothClampMaxDistanceQuery) -> GpuQuery {
    GpuQuery {
        particle_x: q.particle[0],
        particle_y: q.particle[1],
        particle_z: q.particle[2],
        anchor_x: q.anchor[0],
        anchor_y: q.anchor[1],
        anchor_z: q.anchor[2],
        max_distance: q.max_distance,
        pinned: q.pinned,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ClothClampMaxDistanceResult`].
fn decode_result(raw: &GpuResult) -> ClothClampMaxDistanceResult {
    ClothClampMaxDistanceResult {
        position: [raw.pos_x, raw.pos_y, raw.pos_z],
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

/// A compiled, reusable painted max-distance clamp compute pipeline, twinning
/// the `CPU` golden `clamp_max_distance` composed with
/// `PaintedConstraint::clamped`.
pub struct GpuClothClampMaxDistance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuClothClampMaxDistance {
    /// Compiles the painted max-distance clamp kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothClampMaxDistance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_cloth_clamp_max_distance"),
            source: ShaderSource::Wgsl(CLOTH_CLAMP_MAX_DISTANCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_cloth_clamp_max_distance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_cloth_clamp_max_distance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_cloth_clamp_max_distance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothClampMaxDistance {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ClothClampMaxDistanceResult`] per input, in order.
    ///
    /// Each continuous position channel matches the reference to within the
    /// tolerance documented on this module; the `valid` flag matches exactly. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ClothClampMaxDistanceQuery],
    ) -> Vec<ClothClampMaxDistanceResult> {
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
            label: Some("prism_volumetric_cloth_clamp_max_distance_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_cloth_clamp_max_distance_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_cloth_clamp_max_distance_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_cloth_clamp_max_distance_bind_group"),
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
            label: Some("prism_volumetric_cloth_clamp_max_distance_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_cloth_clamp_max_distance_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_cloth_clamp_max_distance_pass"),
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
