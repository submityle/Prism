//! `wgpu` compute twin of the closed-form curvature-mask core
//! `curvature_masks_from_principals` from the `CPU` golden
//! `prism_render_architecture::ray_scene::mesh_sdf_curvature_masks`.
//!
//! `AAA` material pipelines weather a surface from its curvature: dirt and
//! ambient occlusion gather in concave crevices (a *cavity* mask) while paint
//! and gilding rub off the convex ridges that catch contact (an *edge-wear*
//! mask). The golden `curvature_masks_from_principals` turns the two principal
//! curvatures (convex positive) into those two normalized masks plus a combined
//! signed-curvature channel, each a smoothstep ramp between a clean-surface
//! threshold and a saturation curvature.
//!
//! [`GpuSdfCurvatureMasks`] is the on-device twin: one thread solves one query,
//! so a passing real-device parity test is direct evidence the ported kernel
//! computes the same masks and takes the same degenerate-bound branch the
//! reference does, not merely that the shader compiles. The field-level
//! sampling entry point `sdf_curvature_masks` is deliberately *not* twinned
//! here; its curvature sampling is covered by the `mesh_sdf_curvature` twin.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces the three continuous channels the
//! reference returns from a pair of principal curvatures and the tuning
//! parameters: the cavity mask, the edge-wear mask, and the signed-curvature
//! channel. There is no loop and no field sampling: each thread performs a
//! fixed, bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Correctness model
//!
//! Every channel threads through multiplies, adds, one guarded division and a
//! cubic polynomial, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate. The parity test asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on
//! every continuous quantity, tight enough to catch a genuinely wrong port yet
//! loose enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! When the ramp bounds collapse or invert (`hi <= lo`, from a threshold at or
//! above one or a non-positive saturation) the ramp becomes a hard step just
//! above `hi` instead of dividing by a zero-width band, matching the reference.
//! The host and kernel share the ordered compares `hi <= lo` and `value > hi`
//! so they take the same branch; the signed-curvature divide guards its
//! denominator with `max(saturation, f32::MIN_POSITIVE)`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `clamp`,
//! `select`, `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no
//! `round`, no `sqrt` and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::mesh_sdf_curvature_masks`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` curvature-mask kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `curvature_masks_from_principals` branch for branch; see the
/// module documentation for the algorithm.
const MESH_SDF_CURVATURE_MASKS_WGSL: &str = r#"
// Curvature-mask twin: one thread per query reproduces the cavity, edge-wear
// and signed-curvature channels that `curvature_masks_from_principals` derives
// from two principal curvatures and the ramp tuning. It mirrors the CPU golden
// branch for branch, uses only the portable core-WGSL subset (max/clamp/select
// and + - * / plus unsigned index math), needs no sqrt and no transcendental
// call and takes no optional feature, so it runs unmodified on Metal, Vulkan
// and DX12. There is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::ray_scene::mesh_sdf_curvature_masks；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Largest (most convex) principal curvature, convex positive.
    principal_max: f32,
    // Smallest (most concave) principal curvature.
    principal_min: f32,
    // Curvature magnitude mapping to a fully saturated mask value of 1.
    saturation_curvature: f32,
    // Fraction of the saturation, below which the mask reads zero.
    threshold: f32,
}

struct Result {
    // Cavity (crevice) weight in [0, 1].
    cavity: f32,
    // Edge-wear (ridge/corner) weight in [0, 1].
    edge_wear: f32,
    // Combined signed curvature in [-1, 1].
    signed_curvature: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Smoothstep ramp of `value` from `lo` (maps to 0) to `hi` (maps to 1), clamped
// outside [lo, hi]. Degenerate or inverted bounds (hi <= lo) collapse to a hard
// step just above `hi`, so a zero-width band never divides by zero. Mirrors the
// reference `smooth_ramp`, with the cubic expanded as t * t * (3 - 2 * t).
fn smooth_ramp(value: f32, lo: f32, hi: f32) -> f32 {
    if (hi <= lo) {
        return select(0.0, 1.0, value > hi);
    }
    let t = clamp((value - lo) / (hi - lo), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let saturation = q.saturation_curvature;
    // Clean-surface threshold scaled into curvature units, and the saturation
    // the ramp reaches full strength at.
    let lo = clamp(q.threshold, 0.0, 1.0) * saturation;
    let hi = saturation;

    // Edge wear ramps on the largest convex principal curvature; cavity ramps
    // on the magnitude of the most concave one.
    let convex = max(q.principal_max, 0.0);
    let concave = max(-q.principal_min, 0.0);

    let edge_wear = smooth_ramp(convex, lo, hi);
    let cavity = smooth_ramp(concave, lo, hi);

    // Signed-curvature channel is the mean curvature normalized by the
    // saturation, guarded against a non-positive saturation and clamped to the
    // reported [-1, 1] range. f32::MIN_POSITIVE is bit pattern 0x00800000.
    let min_positive = bitcast<f32>(8388608u);
    let mean = 0.5 * (q.principal_max + q.principal_min);
    let signed_curvature = clamp(mean / max(saturation, min_positive), -1.0, 1.0);

    var out: Result;
    out.cavity = cavity;
    out.edge_wear = edge_wear;
    out.signed_curvature = signed_curvature;
    out.pad0 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MESH_SDF_CURVATURE_MASKS_WGSL`].
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
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Largest (most convex) principal curvature.
    principal_max: f32,
    /// Smallest (most concave) principal curvature.
    principal_min: f32,
    /// Curvature magnitude mapping to a fully saturated mask value.
    saturation_curvature: f32,
    /// Fraction of the saturation below which the mask reads zero.
    threshold: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Cavity (crevice) weight.
    cavity: f32,
    /// Edge-wear (ridge/corner) weight.
    edge_wear: f32,
    /// Combined signed curvature.
    signed_curvature: f32,
    /// Padding lane.
    pad0: f32,
}

/// One query for the curvature-mask twin: the two principal curvatures (convex
/// positive) and the ramp tuning the reference `CurvatureMaskParams` carries.
///
/// The three output channels are all derived from this one tuple, so a single
/// query exercises the whole twinned core at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCurvatureMasksQuery {
    /// Largest (most convex) principal curvature, convex positive.
    pub principal_max: f32,
    /// Smallest (most concave) principal curvature.
    pub principal_min: f32,
    /// Principal-curvature magnitude mapping to a fully saturated mask value of
    /// `1.0`; the reference `saturation_curvature`.
    pub saturation_curvature: f32,
    /// Fraction of `saturation_curvature`, in `[0, 1)`, below which the mask
    /// reads zero; the reference `threshold`.
    pub threshold: f32,
}

impl SdfCurvatureMasksQuery {
    /// Builds a query from the two principal curvatures and the ramp tuning.
    #[must_use]
    pub fn new(
        principal_max: f32,
        principal_min: f32,
        saturation_curvature: f32,
        threshold: f32,
    ) -> SdfCurvatureMasksQuery {
        SdfCurvatureMasksQuery {
            principal_max,
            principal_min,
            saturation_curvature,
            threshold,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `CurvatureMasks` channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCurvatureMasksResult {
    /// Cavity weight in `[0, 1]`: `1.0` deep in a sharp concave crevice.
    pub cavity: f32,
    /// Edge-wear weight in `[0, 1]`: `1.0` on a sharp convex ridge or corner.
    pub edge_wear: f32,
    /// Combined signed curvature in `[-1, 1]`: positive convex, negative
    /// concave.
    pub signed_curvature: f32,
}

/// Encodes one [`SdfCurvatureMasksQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfCurvatureMasksQuery) -> GpuQuery {
    GpuQuery {
        principal_max: q.principal_max,
        principal_min: q.principal_min,
        saturation_curvature: q.saturation_curvature,
        threshold: q.threshold,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfCurvatureMasksResult`].
fn decode_result(raw: &GpuResult) -> SdfCurvatureMasksResult {
    SdfCurvatureMasksResult {
        cavity: raw.cavity,
        edge_wear: raw.edge_wear,
        signed_curvature: raw.signed_curvature,
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

/// A compiled, reusable curvature-mask compute pipeline, twinning the `CPU`
/// golden `curvature_masks_from_principals`.
pub struct GpuSdfCurvatureMasks {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfCurvatureMasks {
    /// Compiles the curvature-mask kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfCurvatureMasks {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_masks"),
            source: ShaderSource::Wgsl(MESH_SDF_CURVATURE_MASKS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_masks_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_masks_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_masks_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfCurvatureMasks {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`SdfCurvatureMasksResult`] per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfCurvatureMasksQuery],
    ) -> Vec<SdfCurvatureMasksResult> {
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
            label: Some("prism_volumetric_mesh_sdf_curvature_masks_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_masks_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_masks_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_masks_bind_group"),
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
            label: Some("prism_volumetric_mesh_sdf_curvature_masks_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mesh_sdf_curvature_masks_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mesh_sdf_curvature_masks_pass"),
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
