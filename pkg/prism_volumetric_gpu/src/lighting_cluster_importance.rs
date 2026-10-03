//! `wgpu` compute twin of the per-pair cluster/light importance weight inside
//! the stochastic many-light sampler
//! ([`stochastic`](prism_render_architecture::lighting::stochastic),
//! `MegaLights`-style tile selection).
//!
//! The `CPU` golden
//! [`cluster_light_importance`](prism_render_architecture::lighting::stochastic::cluster_light_importance)
//! scores how strongly one light should illuminate one cluster. It takes the
//! cluster's axis-aligned bounds and the light's bounding sphere plus scalar
//! power, and returns a conservative inverse-square falloff weight: `0` when the
//! light's influence sphere does not reach the cluster box, else
//! `power / max(d², ε)` with `d` the nearest distance from the box to the light
//! center. That weight drives importance sampling, so only its ratios between
//! lights matter, but reproducing it exactly on the device lets a tile selector
//! build identical per-light probabilities `CPU`-side and `GPU`-side.
//!
//! [`GpuLightClusterImportance`] is the on-device twin of that per-pair closed
//! form: one thread scores one `(cluster, light)` pair, reproducing the
//! reference's exact arithmetic so a passing real-device parity test is direct
//! evidence the ported kernel computes the same importance the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Per pair the kernel reproduces, in order: the per-axis clamped squared
//! distance from the light center to the cluster box
//! (`distance_sq_point_aabb`), the overlap test `d² > radius²` that zeroes a
//! light out of range, the `power <= 0` guard that clamps a non-positive power
//! to zero, and the final `power / max(d², MIN_DISTANCE_SQ)` falloff. The
//! squared-distance floor `MIN_DISTANCE_SQ = 1.0e-4` mirrors the golden constant
//! so a light centered inside a cluster earns a large but finite weight instead
//! of dividing by zero.
//!
//! # What stays on the host
//!
//! Only the single scalar closed form is twinned. The surrounding tile selector
//! ([`select_tile_lights`](prism_render_architecture::lighting::stochastic::select_tile_lights)) —
//! its variable-length overlap gather, prefix-sum `CDF`, integer-hash draws with
//! replacement, and deduplicating weight accumulation — stays on the host, as do
//! the clustered-culling assignment and the `MegaLights` budget policy that
//! frame a batch. The host builds each `(cluster, light)` pair and consumes the
//! returned importances; the device performs no gather and no reduction.
//!
//! # Correctness model
//!
//! The overlap classification (`d² > radius²`) is an ordered comparison, so for
//! fixtures chosen clear of the tangency boundary the `CPU` and `GPU` land on
//! the same side and agree on whether the importance is exactly zero. The
//! continuous importance threads through subtracts, multiplies, and a single
//! divide only (no `sqrt`, no transcendental), so where it is non-zero the
//! `CPU` and `GPU` match to within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `+ - * /`, and ordered comparisons — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no `round`, and no
//! `sqrt`. There is no loop over a variable count: each thread performs a fixed,
//! bounded sequence of arithmetic, so the kernel provably terminates. No
//! optional device feature is required, so it runs unmodified across backends.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::stochastic`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` per-pair cluster/light importance kernel, embedded
/// inline so the twin ships as a single source file. The single entry point
/// `solve` mirrors the `CPU` golden
/// [`cluster_light_importance`](prism_render_architecture::lighting::stochastic::cluster_light_importance)
/// closed form; see the module documentation for the algorithm.
const LIGHTING_CLUSTER_IMPORTANCE_WGSL: &str = r#"
// Per-pair cluster/light importance twin: one thread scores one (cluster, light)
// pair into a scalar inverse-square falloff weight, mirroring the CPU golden
// `lighting::stochastic::cluster_light_importance` closed form with only
// min/max, + - * / and ordered comparisons. The variable-length tile selector,
// its CDF draws, and clustered culling stay on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::lighting::stochastic；无第三方
// 引擎源码或衍生代码。

// Floor on squared distance matching the golden `MIN_DISTANCE_SQ`, so a light
// centered inside a cluster earns a large but finite weight, never a divide by
// zero.
const MIN_DISTANCE_SQ: f32 = 1.0e-4;

struct Params {
    // Number of pairs in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Cluster axis-aligned minimum corner.
    cluster_min_x: f32,
    cluster_min_y: f32,
    cluster_min_z: f32,
    // Cluster axis-aligned maximum corner.
    cluster_max_x: f32,
    cluster_max_y: f32,
    cluster_max_z: f32,
    // Light bounding-sphere center.
    light_center_x: f32,
    light_center_y: f32,
    light_center_z: f32,
    // Light influence radius.
    light_radius: f32,
    // Scalar light power driving importance.
    light_power: f32,
    // Padding to a 16-byte stride multiple.
    pad0: f32,
}

struct Result {
    // Scalar importance weight (>= 0).
    importance: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Squared distance from a point to an axis-aligned box, per-axis clamped: an
// axis contributes nothing while the point is inside its span, else the squared
// overshoot past the near face. Ordered comparisons only (no f32 equality).
fn distance_sq_point_aabb(
    px: f32, py: f32, pz: f32,
    minx: f32, miny: f32, minz: f32,
    maxx: f32, maxy: f32, maxz: f32,
) -> f32 {
    var total: f32 = 0.0;

    if (px < minx) {
        let d = minx - px;
        total = total + d * d;
    } else if (px > maxx) {
        let d = px - maxx;
        total = total + d * d;
    }

    if (py < miny) {
        let d = miny - py;
        total = total + d * d;
    } else if (py > maxy) {
        let d = py - maxy;
        total = total + d * d;
    }

    if (pz < minz) {
        let d = minz - pz;
        total = total + d * d;
    } else if (pz > maxz) {
        let d = pz - maxz;
        total = total + d * d;
    }

    return total;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let d2 = distance_sq_point_aabb(
        q.light_center_x, q.light_center_y, q.light_center_z,
        q.cluster_min_x, q.cluster_min_y, q.cluster_min_z,
        q.cluster_max_x, q.cluster_max_y, q.cluster_max_z,
    );
    let r2 = q.light_radius * q.light_radius;

    var importance: f32 = 0.0;
    if (d2 > r2) {
        // Light sphere does not reach the cluster: no contribution.
        importance = 0.0;
    } else {
        var power: f32 = 0.0;
        if (q.light_power > 0.0) {
            power = q.light_power;
        }
        let denom = max(d2, MIN_DISTANCE_SQ);
        importance = power / denom;
    }

    var out: Result;
    out.importance = importance;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    out.pad2 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the pair count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`LIGHTING_CLUSTER_IMPORTANCE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid pairs in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one importance query, matching the `WGSL`
/// `Query` struct: the cluster box corners, the light center, radius, and power,
/// plus one pad word. All members are scalars, so the struct is a dense
/// `48`-byte block with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Cluster minimum corner `x`.
    cluster_min_x: f32,
    /// Cluster minimum corner `y`.
    cluster_min_y: f32,
    /// Cluster minimum corner `z`.
    cluster_min_z: f32,
    /// Cluster maximum corner `x`.
    cluster_max_x: f32,
    /// Cluster maximum corner `y`.
    cluster_max_y: f32,
    /// Cluster maximum corner `z`.
    cluster_max_z: f32,
    /// Light center `x`.
    light_center_x: f32,
    /// Light center `y`.
    light_center_y: f32,
    /// Light center `z`.
    light_center_z: f32,
    /// Light influence radius.
    light_radius: f32,
    /// Scalar light power.
    light_power: f32,
    /// Padding word to a `16`-byte stride multiple.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one importance result, matching the `WGSL`
/// `Result` struct: the scalar importance plus three pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Scalar importance weight (`>= 0`).
    importance: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// One `(cluster, light)` importance query, mirroring the inputs of the golden
/// [`cluster_light_importance`](prism_render_architecture::lighting::stochastic::cluster_light_importance).
///
/// `cluster_min` / `cluster_max` are the cluster's axis-aligned bounds in the
/// same space as the light; `light_center` and `light_radius` are the light's
/// bounding sphere; `light_power` is the scalar power driving importance (a
/// non-positive power is treated as zero, matching the golden guard).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightClusterImportanceQuery {
    /// Cluster axis-aligned minimum corner.
    pub cluster_min: [f32; 3],
    /// Cluster axis-aligned maximum corner.
    pub cluster_max: [f32; 3],
    /// Light bounding-sphere center.
    pub light_center: [f32; 3],
    /// Light influence radius.
    pub light_radius: f32,
    /// Scalar light power driving importance.
    pub light_power: f32,
}

impl LightClusterImportanceQuery {
    /// Builds an importance query.
    #[must_use]
    pub const fn new(
        cluster_min: [f32; 3],
        cluster_max: [f32; 3],
        light_center: [f32; 3],
        light_radius: f32,
        light_power: f32,
    ) -> LightClusterImportanceQuery {
        LightClusterImportanceQuery {
            cluster_min,
            cluster_max,
            light_center,
            light_radius,
            light_power,
        }
    }
}

/// One resolved importance weight, the scalar result of the golden
/// [`cluster_light_importance`](prism_render_architecture::lighting::stochastic::cluster_light_importance).
///
/// `importance` is `0` when the light sphere does not reach the cluster or the
/// light power is non-positive, else the inverse-square falloff weight
/// `power / max(d², MIN_DISTANCE_SQ)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightClusterImportanceResult {
    /// Scalar importance weight (`>= 0`).
    pub importance: f32,
}

/// Encodes one [`LightClusterImportanceQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &LightClusterImportanceQuery) -> GpuQuery {
    GpuQuery {
        cluster_min_x: q.cluster_min[0],
        cluster_min_y: q.cluster_min[1],
        cluster_min_z: q.cluster_min[2],
        cluster_max_x: q.cluster_max[0],
        cluster_max_y: q.cluster_max[1],
        cluster_max_z: q.cluster_max[2],
        light_center_x: q.light_center[0],
        light_center_y: q.light_center[1],
        light_center_z: q.light_center[2],
        light_radius: q.light_radius,
        light_power: q.light_power,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`LightClusterImportanceResult`].
fn decode_result(raw: &GpuResult) -> LightClusterImportanceResult {
    LightClusterImportanceResult {
        importance: raw.importance,
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

/// A compiled, reusable per-pair cluster/light importance compute pipeline,
/// twinning the numeric core of the `CPU` golden
/// [`cluster_light_importance`](prism_render_architecture::lighting::stochastic::cluster_light_importance).
pub struct GpuLightClusterImportance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLightClusterImportance {
    /// Compiles the per-pair importance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLightClusterImportance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_lighting_cluster_importance"),
            source: ShaderSource::Wgsl(LIGHTING_CLUSTER_IMPORTANCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_lighting_cluster_importance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_lighting_cluster_importance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_lighting_cluster_importance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLightClusterImportance {
            module,
            layout,
            pipeline,
        }
    }

    /// Scores every pair in `queries` and returns one
    /// [`LightClusterImportanceResult`] per input, in order.
    ///
    /// The zero / non-zero classification equals the reference exactly for pairs
    /// clear of the tangency boundary; the non-zero importance matches to within
    /// the tolerance documented on this module. An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[LightClusterImportanceQuery],
    ) -> Vec<LightClusterImportanceResult> {
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
            label: Some("prism_volumetric_lighting_cluster_importance_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_lighting_cluster_importance_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_lighting_cluster_importance_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_lighting_cluster_importance_bind_group"),
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
            label: Some("prism_volumetric_lighting_cluster_importance_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_lighting_cluster_importance_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_lighting_cluster_importance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pair, flattened to a 1-D dispatch.
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
