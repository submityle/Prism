//! `wgpu` compute twin of the spectral-cascade distance weighting inside the
//! ocean level-of-detail contract
//! ([`ocean_lod`](prism_render_architecture::water::ocean_lod)).
//!
//! The `CPU` golden
//! [`ocean_lod`](prism_render_architecture::water::ocean_lod) owns the
//! deterministic ocean-surface LOD classification: `clipmap` ring selection,
//! geomorph blending, and the spectral cascade fade weights. This twin
//! reproduces the cascade fade on device: for a camera distance it fills the
//! per-cascade displacement weight each spectral band contributes, mirroring
//! [`cascade_weights_into`](prism_render_architecture::water::ocean_lod::cascade_weights_into)
//! and its per-cascade kernel
//! [`cascade_distance_weight`](prism_render_architecture::water::ocean_lod::cascade_distance_weight).
//! One thread resolves one body's full cascade weight vector.
//!
//! The `clipmap` ring selection, geomorph weighting and the variable-length
//! patch binning of
//! [`ocean_lod`](prism_render_architecture::water::ocean_lod) are intentionally
//! out of scope here; this module twins only the cascade fade closed form.
//!
//! # What is twinned
//!
//! For one body at `distance` and an
//! [`OceanCascadeConfig`](prism_render_architecture::water::ocean_lod::OceanCascadeConfig)
//! the kernel reproduces, writing one weight per cascade:
//! - `fade_begin(cascade) = fade_start + cascade * reach_per_cascade`.
//! - `cascade_distance_weight(cascade, distance, cfg)`: with
//!   `begin = fade_begin(cascade)`, returns `1.0` when `distance <= begin`;
//!   `0.0` when `fade_range <= EPS` (`EPS = 1e-6`); otherwise
//!   `clamp(1.0 - (distance - begin) / fade_range, 0, 1)`.
//! - `cascade_weights_into(distance, cfg, out)`: writes
//!   `count = min(cfg.cascade_count, out.len())` weights in cascade order and
//!   returns `count`. The buffer length `out.len()` is twinned by the
//!   [`WaterOceanCascadeQuery::out_len`] field (both capped to
//!   [`MAX_CASCADES`]); slots past `count` are left zero.
//!
//! # What stays on the host
//!
//! The `clipmap` ring walk, the geomorph band weighting, and the
//! variable-length patch-binning plan of
//! [`ocean_lod`](prism_render_architecture::water::ocean_lod) stay host-side;
//! they are container work with no fixed-width device analogue. The host also
//! owns the empty-batch short-circuit (a storage buffer cannot be zero-sized)
//! and the packing of the public [`WaterOceanCascadeQuery`] into its `std430`
//! slot.
//!
//! # Correctness model
//!
//! Each cascade weight threads through a subtract, a divide and a `clamp`, so
//! the `CPU` and `GPU` are not bit-exact across the divide; each weight is
//! asserted within a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`),
//! tight enough to catch a wrong port yet loose enough to admit a legal
//! last-place difference. The returned `count` is pure integer arithmetic
//! (`min` of two counts) and is asserted exactly. Fixtures keep `fade_range`
//! well clear of the `EPS` degenerate threshold and the distance clear of each
//! cascade's `fade_begin`, so the two agree on every branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `clamp`,
//! `+ - * /`, a bounded loop over at most [`MAX_CASCADES`] cascades, and
//! unsigned index arithmetic — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `tan`, `sqrt`, no inverse trigonometry, no `round` or `ceil`, and no
//! `u64`/`u16`/`i64`/`f64`. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::ocean_lod`；无第三方引擎源码或衍生代码。
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

/// Maximum number of spectral cascades the device result holds. The reference
/// [`OceanCascadeConfig`](prism_render_architecture::water::ocean_lod::OceanCascadeConfig)
/// has no hard cap, so the twin fixes a fixed-width ceiling of `8`; both
/// `cascade_count` and the twinned buffer length `out_len` are clamped to it.
pub const MAX_CASCADES: usize = 8;

/// The portable core-`WGSL` cascade-weight kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`cascade_weights_into`](prism_render_architecture::water::ocean_lod::cascade_weights_into);
/// see the module documentation for the algorithm.
const WATER_OCEAN_CASCADE_WGSL: &str = r#"
// Ocean spectral-cascade fade twin: one thread fills one body's per-cascade
// displacement weights, mirroring the CPU golden `water::ocean_lod` closed form
// with only min/clamp and + - * / over a bounded cascade loop. It owns no ring
// selection and no variable-length binning.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::ocean_lod；无第三方引擎
// 源码或衍生代码。

const EPS: f32 = 1.0e-6;
const MAX_CASCADES: u32 = 8u;

struct Params {
    // Number of bodies in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    distance: f32,
    fade_start: f32,
    fade_range: f32,
    reach_per_cascade: f32,
    cascade_count: u32,
    out_len: u32,
}

struct Result {
    weights: array<f32, 8>,
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Distance weight for a single cascade: full weight up to `fade_begin`, then a
// linear ramp down to zero across `fade_range`, zero for a degenerate range.
fn cascade_weight(cascade: u32, q: Query) -> f32 {
    let begin = q.fade_start + f32(cascade) * q.reach_per_cascade;
    if (q.distance <= begin) {
        return 1.0;
    }
    if (q.fade_range <= EPS) {
        return 0.0;
    }
    let t = (q.distance - begin) / q.fade_range;
    return clamp(1.0 - t, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: Result;
    // Zero every fixed-width slot first; slots past `count` stay zero.
    for (var i: u32 = 0u; i < MAX_CASCADES; i = i + 1u) {
        out.weights[i] = 0.0;
    }

    // count = min(cascade_count, out_len), both already capped to MAX_CASCADES.
    var count = min(q.cascade_count, q.out_len);
    count = min(count, MAX_CASCADES);
    for (var c: u32 = 0u; c < count; c = c + 1u) {
        out.weights[c] = cascade_weight(c, q);
    }

    out.count = count;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the body count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`WATER_OCEAN_CASCADE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid bodies in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one cascade query: four scalars and two counts
/// to a `24`-byte stride, matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Camera distance in meters.
    distance: f32,
    /// Distance at which cascade `0` begins to fade.
    fade_start: f32,
    /// Width of the linear fade from full weight to zero.
    fade_range: f32,
    /// Extra reach granted to each coarser cascade before it fades.
    reach_per_cascade: f32,
    /// Number of spectral cascades.
    cascade_count: u32,
    /// Twinned destination buffer length, capped to `MAX_CASCADES`.
    out_len: u32,
}

/// `repr(C)` `std430` layout of one cascade result, matching the `WGSL` `Result`
/// struct: eight weights, a count, and three pad words to a `48`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Per-cascade displacement weights; slots past `count` are zero.
    weights: [f32; MAX_CASCADES],
    /// Number of weights written: `min(cascade_count, out_len)`.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One cascade query: a camera distance plus the fields of the reference
/// [`OceanCascadeConfig`](prism_render_architecture::water::ocean_lod::OceanCascadeConfig),
/// flattened, plus the twinned destination buffer length `out_len`.
///
/// `distance`, `fade_start`, `fade_range`, `reach_per_cascade` and
/// `cascade_count` feed
/// [`cascade_weights_into`](prism_render_architecture::water::ocean_lod::cascade_weights_into);
/// `out_len` twins the length of the caller's output slice (capped to
/// [`MAX_CASCADES`]), so the returned `count` is `min(cascade_count, out_len)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOceanCascadeQuery {
    /// Camera distance in meters.
    pub distance: f32,
    /// Distance at which the finest cascade (index `0`) begins to fade.
    pub fade_start: f32,
    /// Width of the linear fade from full weight to zero, in meters.
    pub fade_range: f32,
    /// Extra reach granted to each coarser cascade before it fades.
    pub reach_per_cascade: f32,
    /// Number of spectral cascades stacked into the displacement.
    pub cascade_count: u32,
    /// Length of the twinned destination buffer, clamped to [`MAX_CASCADES`].
    pub out_len: u32,
}

/// One resolved cascade result: the per-cascade displacement weights the
/// reference
/// [`cascade_weights_into`](prism_render_architecture::water::ocean_lod::cascade_weights_into)
/// writes, plus the count it returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterOceanCascadeResult {
    /// Per-cascade displacement weights in `0..=1`; slots past `count` are zero.
    pub weights: [f32; MAX_CASCADES],
    /// Number of valid weights: `min(cascade_count, out_len)`.
    pub count: u32,
}

/// Encodes one [`WaterOceanCascadeQuery`] into its `std430` [`GpuQuery`] slot,
/// clamping `out_len` and `cascade_count` to [`MAX_CASCADES`] so the device
/// loop stays inside the fixed-width result.
fn encode_query(q: &WaterOceanCascadeQuery) -> GpuQuery {
    let cap = MAX_CASCADES as u32;
    GpuQuery {
        distance: q.distance,
        fade_start: q.fade_start,
        fade_range: q.fade_range,
        reach_per_cascade: q.reach_per_cascade,
        cascade_count: q.cascade_count.min(cap),
        out_len: q.out_len.min(cap),
    }
}

/// Decodes one packed [`GpuResult`] into the public [`WaterOceanCascadeResult`].
fn decode_result(raw: &GpuResult) -> WaterOceanCascadeResult {
    WaterOceanCascadeResult {
        weights: raw.weights,
        count: raw.count,
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

/// A compiled, reusable cascade-weight compute pipeline, twinning the spectral
/// cascade fade of the `CPU` golden
/// [`ocean_lod`](prism_render_architecture::water::ocean_lod) module.
pub struct GpuWaterOceanCascade {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterOceanCascade {
    /// Compiles the cascade-weight kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterOceanCascade {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_ocean_cascade"),
            source: ShaderSource::Wgsl(WATER_OCEAN_CASCADE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_ocean_cascade_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_ocean_cascade_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_ocean_cascade_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterOceanCascade {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every body in `queries` and returns one
    /// [`WaterOceanCascadeResult`] per input, in order.
    ///
    /// Each continuous weight matches the reference within the tolerance
    /// documented on this module and the `count` matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterOceanCascadeQuery],
    ) -> Vec<WaterOceanCascadeResult> {
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
            label: Some("prism_volumetric_water_ocean_cascade_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_ocean_cascade_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_ocean_cascade_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_ocean_cascade_bind_group"),
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
            label: Some("prism_volumetric_water_ocean_cascade_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_ocean_cascade_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_ocean_cascade_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per body, flattened to a 1-D dispatch.
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
