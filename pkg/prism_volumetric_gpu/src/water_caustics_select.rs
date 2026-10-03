//! `wgpu` compute twin of the water caustics route selector
//! (`crate`-external golden `water::caustics::select_caustics`).
//!
//! A water surface renderer picks one of three caustics techniques per receiver
//! depending on how far the receiver is from the camera and a quality bias:
//! photon-mapped caustics for near receivers, ray-traced caustics for the
//! middle band, and a cheap Jacobian projection for far receivers. The golden
//! `select_caustics` returns a `CausticsMethod` whose `cost_rank` is the
//! discrete method index this twin reproduces: `JacobianProjection` is `0`,
//! `RayTraced` is `1`, and `PhotonMapped` is `2`.
//!
//! # What is twinned
//!
//! One thread resolves one query. For a receiver at `camera_distance` with
//! `quality_bias`, the band limits are expanded by `1 + 0.5 * clamp(bias, 0, 1)`
//! (up to 50% at full bias); the camera distance and both thresholds are first
//! clamped non-negative. The receiver falls into the photon band when
//! `distance <= photon_max`, otherwise the ray band when
//! `distance <= max(ray_max, photon_max)`, otherwise the Jacobian band. The twin
//! spells out that comparison ladder with the same ordered compares as the
//! reference, so a passing real-device parity test is direct evidence the ported
//! kernel routes identically, not merely that the shader compiles.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: the whole selector is a
//! fixed, bounded sequence of arithmetic and comparisons that runs on device.
//! The host only flattens the query batch into a `std430` storage buffer and
//! short-circuits an empty batch (a storage buffer cannot be zero-sized).
//!
//! # Correctness model
//!
//! The method index is a *discrete classification*, so the `CPU` and `GPU`
//! agree exactly and the parity test asserts an exact integer `==` (tolerance
//! `0`). Fixtures that sit a receiver exactly on a band edge are included
//! deliberately: because the reference uses `<=` and the twin uses the same
//! ordered compare on identically rounded operands, the boundary resolves the
//! same way on both sides.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `+ - *` and ordered comparisons — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt`, no `round` and no
//! `u64`/`u16`/`i64`/`f64`. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of arithmetic, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture` 的 `water::caustics::select_caustics`；无第三方引擎源码或衍生代码。
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

/// Method index for the far-receiver Jacobian-projection route, matching the
/// golden `CausticsMethod::JacobianProjection` `cost_rank`.
const METHOD_JACOBIAN: u32 = 0;
/// Method index for the mid-band ray-traced route, matching the golden
/// `CausticsMethod::RayTraced` `cost_rank`.
const METHOD_RAY_TRACED: u32 = 1;
/// Method index for the near-receiver photon-mapped route, matching the golden
/// `CausticsMethod::PhotonMapped` `cost_rank`.
const METHOD_PHOTON_MAPPED: u32 = 2;

/// Host-side independent reimplementation of the golden
/// `water::caustics::select_caustics` reduced to its discrete method index.
///
/// This mirrors the reference semantics exactly without importing the golden,
/// so the twin stays self-contained and free of any cross-crate private
/// dependency: the camera distance and both thresholds are clamped
/// non-negative, the bands are expanded by `1 + 0.5 * clamp(bias, 0, 1)`, and
/// the receiver is routed by the `<=` ladder into
/// [`METHOD_PHOTON_MAPPED`], [`METHOD_RAY_TRACED`], or [`METHOD_JACOBIAN`].
#[must_use]
pub fn select_caustics_index(
    camera_distance: f32,
    quality_bias: f32,
    photon_max_distance: f32,
    ray_max_distance: f32,
) -> u32 {
    let distance = camera_distance.max(0.0);
    let bias = quality_bias.clamp(0.0, 1.0);
    let expand = 1.0 + 0.5 * bias;
    let photon_max = photon_max_distance.max(0.0) * expand;
    let ray_max = ray_max_distance.max(0.0) * expand;
    if distance <= photon_max {
        METHOD_PHOTON_MAPPED
    } else if distance <= ray_max.max(photon_max) {
        METHOD_RAY_TRACED
    } else {
        METHOD_JACOBIAN
    }
}

/// The portable core-`WGSL` caustics-selector kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the golden `water::caustics::select_caustics` reduced to its method index;
/// see the module documentation for the algorithm.
const WATER_CAUSTICS_SELECT_WGSL: &str = r#"
// Water caustics route selector twin: one thread routes one receiver into the
// photon-mapped (2), ray-traced (1) or Jacobian-projection (0) band, mirroring
// the CPU golden `water::caustics::select_caustics` with only min/max/clamp,
// + - * and ordered comparisons.
//
// Provenance: 孪生自本仓 prism_render_architecture 的 water::caustics::select_caustics；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Receiver distance from the camera.
    camera_distance: f32,
    // Quality bias in 0..=1 expanding the expensive bands.
    quality_bias: f32,
    // At or below this (expanded) distance photon mapping is used.
    photon_max_distance: f32,
    // At or below this (expanded) distance ray tracing is used.
    ray_max_distance: f32,
}

struct CausticResult {
    // Resolved method index: 0 Jacobian, 1 ray-traced, 2 photon-mapped.
    method_index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<CausticResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Clamp distance and thresholds non-negative, then expand the bands by
    // 1 + 0.5 * clamp(bias, 0, 1), matching the reference.
    let dist = max(q.camera_distance, 0.0);
    let bias = clamp(q.quality_bias, 0.0, 1.0);
    let expand = 1.0 + 0.5 * bias;
    let photon_lim = max(q.photon_max_distance, 0.0) * expand;
    let ray_lim = max(q.ray_max_distance, 0.0) * expand;

    // The `<=` ladder: photon band first, then the ray band bounded by the
    // larger of the two expanded limits, else the Jacobian fallback.
    var method_idx: u32 = 0u;
    if (dist <= photon_lim) {
        method_idx = 2u;
    } else if (dist <= max(ray_lim, photon_lim)) {
        method_idx = 1u;
    } else {
        method_idx = 0u;
    }

    var out: CausticResult;
    out.method_index = method_idx;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`WATER_CAUSTICS_SELECT_WGSL`].
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

/// `repr(C)` `std430` layout of one selector query: the receiver distance, the
/// quality bias, and the two band thresholds, matching the `WGSL` `Query`
/// struct's `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Receiver distance from the camera.
    camera_distance: f32,
    /// Quality bias in `0..=1`.
    quality_bias: f32,
    /// Photon-band threshold distance (pre-expansion).
    photon_max_distance: f32,
    /// Ray-band threshold distance (pre-expansion).
    ray_max_distance: f32,
}

/// `repr(C)` `std430` layout of one selector result, matching the `WGSL`
/// `CausticResult` struct: the method index plus three pad words to a `16`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Resolved caustics method index.
    method_index: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One caustics-route query: a receiver `camera_distance`, a `quality_bias`, and
/// the two band thresholds `photon_max_distance` and `ray_max_distance`.
///
/// The fields mirror the golden `water::caustics::select_caustics` arguments
/// (the thresholds are the two `CausticsThresholds` fields); the host enqueues
/// one query per receiver, and an empty batch is short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterCausticsSelectQuery {
    /// Receiver distance from the camera.
    pub camera_distance: f32,
    /// Quality bias in `0..=1` expanding the expensive bands.
    pub quality_bias: f32,
    /// Photon-band threshold distance (pre-expansion).
    pub photon_max_distance: f32,
    /// Ray-band threshold distance (pre-expansion).
    pub ray_max_distance: f32,
}

impl WaterCausticsSelectQuery {
    /// Builds a query from a receiver distance, quality bias, and the two band
    /// thresholds.
    #[must_use]
    pub const fn new(
        camera_distance: f32,
        quality_bias: f32,
        photon_max_distance: f32,
        ray_max_distance: f32,
    ) -> WaterCausticsSelectQuery {
        WaterCausticsSelectQuery {
            camera_distance,
            quality_bias,
            photon_max_distance,
            ray_max_distance,
        }
    }
}

/// One resolved caustics route: the discrete `method_index` (`0` Jacobian, `1`
/// ray-traced, `2` photon-mapped), equal to the golden `CausticsMethod`
/// `cost_rank`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaterCausticsSelectResult {
    /// Resolved caustics method index.
    pub method_index: u32,
}

/// Encodes one [`WaterCausticsSelectQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &WaterCausticsSelectQuery) -> GpuQuery {
    GpuQuery {
        camera_distance: q.camera_distance,
        quality_bias: q.quality_bias,
        photon_max_distance: q.photon_max_distance,
        ray_max_distance: q.ray_max_distance,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterCausticsSelectResult`].
fn decode_result(raw: &GpuResult) -> WaterCausticsSelectResult {
    WaterCausticsSelectResult {
        method_index: raw.method_index,
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

/// A compiled, reusable caustics-selector compute pipeline, twinning the golden
/// `water::caustics::select_caustics` reduced to its discrete method index.
pub struct GpuWaterCausticsSelect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterCausticsSelect {
    /// Compiles the caustics-selector kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterCausticsSelect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_caustics_select"),
            source: ShaderSource::Wgsl(WATER_CAUSTICS_SELECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_caustics_select_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_caustics_select_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_caustics_select_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterCausticsSelect {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`WaterCausticsSelectResult`] per input, in order.
    ///
    /// The method index equals the reference exactly (a discrete
    /// classification). An empty `queries` batch returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterCausticsSelectQuery],
    ) -> Vec<WaterCausticsSelectResult> {
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
            label: Some("prism_volumetric_water_caustics_select_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_caustics_select_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_caustics_select_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_caustics_select_bind_group"),
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
            label: Some("prism_volumetric_water_caustics_select_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_caustics_select_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_caustics_select_pass"),
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
