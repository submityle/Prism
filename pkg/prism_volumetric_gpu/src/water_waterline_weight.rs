//! `wgpu` compute twin of the waterline mask and above/below-water transition
//! ([`waterline`](prism_render_architecture::water::waterline)).
//!
//! The waterline is where the animated water surface crosses solid geometry: a
//! shoreline, a pier piling, a swimmer's torso. Shading needs three things
//! there — a boolean above/below test to route underwater shading, a soft
//! transition weight so the surface intersection does not alias into a hard
//! line, and a shallow-water shoreline band that seeds wet sand and shore foam.
//! The `CPU` golden derives all three from a sample's world height, the local
//! water-surface height, and the total water depth with nothing but signed
//! comparisons, a guarded divide, and clamping, so it ports to the device
//! directly.
//!
//! [`GpuWaterWaterlineWeight`] is the on-device twin: one thread resolves one
//! sample, reproducing
//! [`submersion_depth`](prism_render_architecture::water::waterline::submersion_depth),
//! [`is_underwater`](prism_render_architecture::water::waterline::is_underwater),
//! [`waterline_weight`](prism_render_architecture::water::waterline::waterline_weight)
//! and
//! [`shoreline_band`](prism_render_architecture::water::waterline::shoreline_band).
//! A passing real-device parity test is direct evidence the ported kernel
//! reproduces the sign test, the soft ramp, and the shoreline falloff the
//! reference performs, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For one query holding a sample height `sample_y`, the local water-surface
//! height `water_surface_y`, and the total `water_depth`, together with the
//! dispatch-wide tuning `transition_half_width` and `shoreline_depth`, the
//! kernel reproduces:
//! - the signed submersion depth `water_surface_y - sample_y`;
//! - the inclusive underwater flag `submersion_depth >= 0`;
//! - the soft waterline weight, a linear ramp across `2 *
//!   transition_half_width` centred on the surface and clamped to `0..=1`, which
//!   collapses to a hard step when the band is degenerate; and
//! - the shoreline-band weight, zero for dry or deep samples and rising toward
//!   `1` as the water depth drops below `shoreline_depth`.
//!
//! The degenerate-band test is `transition_half_width <= EPS`, matching the
//! golden guard with the shared `water` module constant `EPS` (`1e-6`), and the
//! shoreline reach guard is `shoreline_depth > EPS`.
//!
//! # What stays on the host
//!
//! The tuning [`WaterlineParams`](prism_render_architecture::water::waterline::WaterlineParams)
//! is dispatch-wide, so its two fields ride in the uniform block rather than in
//! each query; the host passes them to [`GpuWaterWaterlineWeight::evaluate`]. An
//! empty batch short-circuits on the host, since a storage buffer cannot be
//! zero-sized.
//!
//! # Correctness model
//!
//! The kernel performs only signed comparisons, a guarded divide by a non-zero
//! band width or reach, and clamps, so a correct port reproduces the reference
//! to within floating-point rounding. The parity test asserts the shared
//! continuous tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every
//! continuous output and an exact match on the underwater flag. Fixtures keep
//! the submersion depth away from zero so the flag is unambiguous.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — a signed comparison,
//! `clamp`, `max`, `+`, `-`, `*`, `/` — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry, no `sqrt`, and no `64`-bit integers or
//! floats. No optional device feature is required, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::waterline`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` waterline kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` mirrors the `CPU`
/// golden `submersion_depth`, `is_underwater`, `waterline_weight` and
/// `shoreline_band`; see the module documentation for the algorithm.
const WATER_WATERLINE_WEIGHT_WGSL: &str = r#"
// Waterline twin: one thread resolves one sample's signed submersion depth, its
// inclusive underwater flag, the soft waterline ramp weight, and the shoreline
// band weight, mirroring the CPU golden
// `water::waterline::{submersion_depth, is_underwater, waterline_weight,
// shoreline_band}` with only a signed comparison, a guarded divide, max and
// clamp.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::waterline；无第三方引擎源码
// 或衍生代码。

// Shared `water` module epsilon guarding the degenerate band and reach, copied
// from the golden `water::EPS`.
const EPS: f32 = 1e-6;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    // Half-width of the soft transition straddling the surface, in meters.
    transition_half_width: f32,
    // Total water depth below which a submerged sample joins the shoreline band.
    shoreline_depth: f32,
    pad0: u32,
}

struct Query {
    // Sample world height, in meters.
    sample_y: f32,
    // Local water-surface height, in meters.
    water_surface_y: f32,
    // Total water depth beneath the sample, in meters.
    water_depth: f32,
    pad0: u32,
}

struct Result {
    // Signed submersion depth: water_surface_y - sample_y.
    submersion_depth: f32,
    // Inclusive underwater flag (1 = at or below the surface, else 0).
    is_underwater: u32,
    // Soft waterline weight in [0, 1].
    waterline_weight: f32,
    // Shoreline-band weight in [0, 1].
    shoreline_band: f32,
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

    // Signed submersion depth and the inclusive underwater flag.
    let depth = q.water_surface_y - q.sample_y;
    var underwater: u32 = 0u;
    if (depth >= 0.0) {
        underwater = 1u;
    }

    // Soft waterline ramp: a degenerate band collapses to a hard step.
    let half = params.transition_half_width;
    var weight: f32 = 0.0;
    if (half <= EPS) {
        if (depth >= 0.0) {
            weight = 1.0;
        } else {
            weight = 0.0;
        }
    } else {
        weight = clamp((depth + half) / (2.0 * half), 0.0, 1.0);
    }

    // Shoreline band: dry or deep samples read 0; it rises as depth shrinks.
    var band: f32 = 0.0;
    if (underwater == 1u && params.shoreline_depth > EPS) {
        let clamped_depth = max(q.water_depth, 0.0);
        band = clamp(1.0 - clamped_depth / params.shoreline_depth, 0.0, 1.0);
    }

    var out: Result;
    out.submersion_depth = depth;
    out.is_underwater = underwater;
    out.waterline_weight = weight;
    out.shoreline_band = band;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus the two
/// dispatch-wide tuning fields and one pad word, filling a `16`-byte,
/// `std140`-aligned uniform struct matching `Params` in
/// [`WATER_WATERLINE_WEIGHT_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Half-width of the soft transition straddling the surface, in meters.
    transition_half_width: f32,
    /// Total water depth below which a submerged sample joins the shoreline
    /// band, in meters.
    shoreline_depth: f32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one query: the sample height, the water-surface
/// height and the total water depth plus one pad word, a `16`-byte stride
/// matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Sample world height, in meters.
    sample_y: f32,
    /// Local water-surface height, in meters.
    water_surface_y: f32,
    /// Total water depth beneath the sample, in meters.
    water_depth: f32,
    /// Padding word.
    pad0: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result` struct:
/// the signed submersion depth, the underwater flag, the soft waterline weight
/// and the shoreline-band weight, a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Signed submersion depth: `water_surface_y - sample_y`.
    submersion_depth: f32,
    /// Inclusive underwater flag (`1` at or below the surface, else `0`).
    is_underwater: u32,
    /// Soft waterline weight in `0..=1`.
    waterline_weight: f32,
    /// Shoreline-band weight in `0..=1`.
    shoreline_band: f32,
}

/// One waterline query for the twin: a sample's height, the local water-surface
/// height, and the total water depth beneath it.
///
/// The dispatch-wide tuning (`transition_half_width`, `shoreline_depth`) is
/// supplied once to [`GpuWaterWaterlineWeight::evaluate`], mirroring the
/// reference
/// [`WaterlineParams`](prism_render_architecture::water::waterline::WaterlineParams).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterWaterlineWeightQuery {
    /// Sample world height, in meters.
    pub sample_y: f32,
    /// Local water-surface height, in meters.
    pub water_surface_y: f32,
    /// Total water depth beneath the sample, in meters.
    pub water_depth: f32,
}

impl WaterWaterlineWeightQuery {
    /// Builds a query from a sample height, the water-surface height, and the
    /// total water depth, all in meters.
    #[must_use]
    pub const fn new(
        sample_y: f32,
        water_surface_y: f32,
        water_depth: f32,
    ) -> WaterWaterlineWeightQuery {
        WaterWaterlineWeightQuery {
            sample_y,
            water_surface_y,
            water_depth,
        }
    }
}

/// One resolved waterline sample, mirroring the reference
/// [`submersion_depth`](prism_render_architecture::water::waterline::submersion_depth),
/// [`is_underwater`](prism_render_architecture::water::waterline::is_underwater),
/// [`waterline_weight`](prism_render_architecture::water::waterline::waterline_weight)
/// and
/// [`shoreline_band`](prism_render_architecture::water::waterline::shoreline_band).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterWaterlineWeightResult {
    /// Signed submersion depth: `water_surface_y - sample_y`.
    pub submersion_depth: f32,
    /// Inclusive underwater flag: `true` at or below the water surface.
    pub is_underwater: bool,
    /// Soft waterline weight in `0..=1`: `0` fully in air, `1` fully submerged.
    pub waterline_weight: f32,
    /// Shoreline-band weight in `0..=1` for a submerged, shallow sample.
    pub shoreline_band: f32,
}

/// Encodes one [`WaterWaterlineWeightQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &WaterWaterlineWeightQuery) -> GpuQuery {
    GpuQuery {
        sample_y: q.sample_y,
        water_surface_y: q.water_surface_y,
        water_depth: q.water_depth,
        pad0: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterWaterlineWeightResult`], mapping the underwater flag word to a
/// [`bool`].
fn decode_result(raw: &GpuResult) -> WaterWaterlineWeightResult {
    WaterWaterlineWeightResult {
        submersion_depth: raw.submersion_depth,
        is_underwater: raw.is_underwater != 0,
        waterline_weight: raw.waterline_weight,
        shoreline_band: raw.shoreline_band,
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

/// A compiled, reusable waterline compute pipeline, twinning the `CPU` golden
/// waterline mask from
/// [`waterline`](prism_render_architecture::water::waterline).
pub struct GpuWaterWaterlineWeight {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterWaterlineWeight {
    /// Compiles the waterline kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterWaterlineWeight {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_waterline_weight"),
            source: ShaderSource::Wgsl(WATER_WATERLINE_WEIGHT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_waterline_weight_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_waterline_weight_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_waterline_weight_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterWaterlineWeight {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` under the dispatch-wide tuning
    /// (`transition_half_width`, `shoreline_depth`) and returns one
    /// [`WaterWaterlineWeightResult`] per input, in order.
    ///
    /// Each output matches the reference: the signed submersion depth, the
    /// inclusive underwater flag, the soft waterline ramp weight, and the
    /// shoreline-band weight. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        transition_half_width: f32,
        shoreline_depth: f32,
        queries: &[WaterWaterlineWeightQuery],
    ) -> Vec<WaterWaterlineWeightResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            transition_half_width,
            shoreline_depth,
            pad0: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_waterline_weight_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_waterline_weight_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_waterline_weight_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_waterline_weight_bind_group"),
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
            label: Some("prism_volumetric_water_waterline_weight_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_waterline_weight_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_waterline_weight_pass"),
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
