//! `wgpu` compute twin of the wet-hair optical/physical response from the `CPU`
//! golden `prism_render_architecture::hair::wetness`.
//!
//! When hair takes on water the change is driven by a single scalar saturation
//! `w` in `[0, 1]` (`0` fully dry, `1` fully saturated). The golden module maps
//! that scalar to a set of deterministic parameter *modifiers* and offers two
//! consumers that this twin ports: the pigment absorption multiplier applied
//! per `RGB` channel (`apply_absorption`) and the additive, unit-clamped
//! roughness offset applied to a base roughness (`apply_roughness`). Both are a
//! straight linear interpolation between the dry endpoint (`w = 0`) and the wet
//! endpoint (`w = 1`), so there is no transcendental math and the map is exactly
//! golden-comparable. One thread solves one query.
//!
//! [`GpuHairWetnessResponse`] is the on-device twin; a passing real-device
//! parity test is direct evidence the ported kernel reproduces the same
//! saturation sanitisation, the same linearly interpolated wet modifiers, the
//! same per-channel absorption scale and the same clamped roughness offset the
//! reference computes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces, independently of the golden crate:
//! `sanitize_wetness(wetness)` (non-finite or out-of-range inputs collapse to
//! the valid `[0, 1]` range); the `sigma_a` multiplier
//! `lerp(1.0, SIGMA_A_MUL_WET, w)` and the roughness offset
//! `lerp(0.0, ROUGHNESS_DELTA_WET, w)`; the per-channel absorption scale
//! `base * sigma_a_mul`; and the unit-clamped roughness
//! `clamp(base + roughness_delta, 0, 1)`. There is no loop: each thread performs
//! a fixed, bounded sequence of multiplies, adds, clamps and selects, so the
//! kernel provably terminates.
//!
//! # Correctness model
//!
//! Every continuous quantity threads through multiplies and adds, so `CPU` and
//! `GPU` are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate. The parity test asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on every continuous channel. The
//! discrete `valid` flag is compared exactly; it is always `1`, since the
//! sanitiser accepts every input and the map never faults.
//!
//! # Degenerate inputs
//!
//! A `NaN`, `+inf`, `-inf`, negative, or greater-than-one `wetness` is sanitised
//! to `[0, 1]` (non-finite collapses to `0.0`, fully dry), so no query is
//! rejected and `valid` is always `1`. An empty query batch short-circuits on
//! the host with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `select`,
//! `+ - *` — with no `sin`, `cos`, `exp`, `log`, `pow`, no `round`, no float
//! `%`, and no `u64` / `i64` / `f64`, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. The non-finite test avoids bare float equality by using two
//! ordered comparisons against the largest finite `f32` magnitude, which both
//! fail for `NaN` and for either infinity.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::wetness`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` wet-hair response kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `sanitize_wetness`, `wet_hair_response`, `apply_absorption`
/// and `apply_roughness`; see the module documentation.
const HAIR_WETNESS_RESPONSE_WGSL: &str = r#"
// Wet-hair response twin: one thread per query sanitises the wetness scalar to
// [0, 1], linearly interpolates the sigma_a multiplier and roughness offset,
// then applies the per-channel absorption scale and the unit-clamped roughness.
// It mirrors the CPU golden exactly and uses only the portable core subset.

struct Params {
    // Number of valid queries in the input and output buffers.
    count: u32,
    // Padding words so the uniform struct fills 16 bytes.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Raw saturation scalar; sanitised to [0, 1] before use.
    wetness: f32,
    // Base roughness the wet offset is added to.
    base_roughness: f32,
    // Base RGB pigment absorption, scaled per channel by sigma_a_mul.
    base_abs_r: f32,
    base_abs_g: f32,
    base_abs_b: f32,
    // Padding words to a 32-byte stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Result {
    // Wet-deepened absorption: base_abs * sigma_a_mul, per channel.
    out_abs_r: f32,
    out_abs_g: f32,
    out_abs_b: f32,
    // Unit-clamped wet roughness: clamp(base_roughness + roughness_delta, 0, 1).
    out_roughness: f32,
    // Sanitised saturation actually used, in [0, 1].
    sanitized_wetness: f32,
    // Always 1: the sanitiser accepts every input, so no query faults.
    valid: u32,
    // Padding words to a 32-byte stride.
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// sigma_a multiplier at full saturation; dry endpoint is 1.0.
const SIGMA_A_MUL_WET: f32 = 1.5;
// Additive roughness offset at full saturation; dry endpoint is 0.0.
const ROUGHNESS_DELTA_WET: f32 = -0.35;
// Largest finite f32 magnitude; the nearest finite value to the IEEE maximum.
// A value is finite exactly when it lies within [-LIMIT, LIMIT]: NaN fails both
// ordered compares (every NaN comparison is false) and either infinity fails
// one, so this reproduces is_finite without any bare float equality.
const WETNESS_FINITE_LIMIT: f32 = 3.40282347e38;

// Linear interpolation between the dry endpoint a and the wet endpoint b by w.
fn lerp_f32(a: f32, b: f32, w: f32) -> f32 {
    return a + (b - a) * w;
}

// Clamp an arbitrary wetness to [0, 1]; non-finite inputs collapse to 0.0.
fn sanitize_wetness(w: f32) -> f32 {
    let finite = (w <= WETNESS_FINITE_LIMIT) && (w >= -WETNESS_FINITE_LIMIT);
    return select(0.0, clamp(w, 0.0, 1.0), finite);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let w = sanitize_wetness(q.wetness);
    let sigma_a_mul = lerp_f32(1.0, SIGMA_A_MUL_WET, w);
    let roughness_delta = lerp_f32(0.0, ROUGHNESS_DELTA_WET, w);
    var out: Result;
    out.out_abs_r = q.base_abs_r * sigma_a_mul;
    out.out_abs_g = q.base_abs_g * sigma_a_mul;
    out.out_abs_b = q.base_abs_b * sigma_a_mul;
    out.out_roughness = clamp(q.base_roughness + roughness_delta, 0.0, 1.0);
    out.sanitized_wetness = w;
    out.valid = 1u;
    out.pad0 = 0.0;
    out.pad1 = 0.0;
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
/// All fields are scalar `f32`, so the struct aligns to `4` bytes with a fixed
/// `32`-byte stride; the three trailing pads keep that stride explicit.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    wetness: f32,
    base_roughness: f32,
    base_abs_r: f32,
    base_abs_g: f32,
    base_abs_b: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct, with two trailing pads for a fixed `32`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    out_abs_r: f32,
    out_abs_g: f32,
    out_abs_b: f32,
    out_roughness: f32,
    sanitized_wetness: f32,
    valid: u32,
    pad0: f32,
    pad1: f32,
}

/// One query for the wet-hair response twin: the raw saturation scalar plus the
/// base `RGB` pigment absorption and the base roughness the modifiers act on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairWetnessResponseQuery {
    /// Raw water-saturation scalar; sanitised to `[0, 1]` before use.
    pub wetness: f32,
    /// Base `RGB` pigment absorption `sigma_a`, scaled per channel.
    pub base_absorption: [f32; 3],
    /// Base roughness the wet offset is added to.
    pub base_roughness: f32,
}

impl HairWetnessResponseQuery {
    /// Builds a query from the saturation scalar, base absorption and base
    /// roughness.
    #[must_use]
    pub fn new(
        wetness: f32,
        base_absorption: [f32; 3],
        base_roughness: f32,
    ) -> HairWetnessResponseQuery {
        HairWetnessResponseQuery {
            wetness,
            base_absorption,
            base_roughness,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `apply_absorption` and `apply_roughness` over the sanitised saturation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairWetnessResponseResult {
    /// Wet-deepened `RGB` absorption: `base_absorption * sigma_a_mul`.
    pub out_absorption: [f32; 3],
    /// Unit-clamped wet roughness: `clamp(base_roughness + roughness_delta, 0, 1)`.
    pub out_roughness: f32,
    /// Sanitised saturation actually used, in `[0, 1]`.
    pub sanitized_wetness: f32,
    /// Always `1`: the sanitiser accepts every input, so no query faults.
    pub valid: u32,
}

/// Encodes one [`HairWetnessResponseQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &HairWetnessResponseQuery) -> GpuQuery {
    GpuQuery {
        wetness: q.wetness,
        base_roughness: q.base_roughness,
        base_abs_r: q.base_absorption[0],
        base_abs_g: q.base_absorption[1],
        base_abs_b: q.base_absorption[2],
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`HairWetnessResponseResult`].
fn decode_result(raw: &GpuResult) -> HairWetnessResponseResult {
    HairWetnessResponseResult {
        out_absorption: [raw.out_abs_r, raw.out_abs_g, raw.out_abs_b],
        out_roughness: raw.out_roughness,
        sanitized_wetness: raw.sanitized_wetness,
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

/// A compiled, reusable wet-hair response compute pipeline, twinning the `CPU`
/// golden `sanitize_wetness`, `wet_hair_response`, `apply_absorption` and
/// `apply_roughness`.
pub struct GpuHairWetnessResponse {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairWetnessResponse {
    /// Compiles the wet-hair response kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairWetnessResponse {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_wetness_response"),
            source: ShaderSource::Wgsl(HAIR_WETNESS_RESPONSE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_wetness_response_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_wetness_response_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_wetness_response_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairWetnessResponse {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`HairWetnessResponseResult`] per input, in order.
    ///
    /// Each continuous channel matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairWetnessResponseQuery],
    ) -> Vec<HairWetnessResponseResult> {
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
            label: Some("prism_volumetric_hair_wetness_response_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_wetness_response_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_wetness_response_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_wetness_response_bind_group"),
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
            label: Some("prism_volumetric_hair_wetness_response_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_wetness_response_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_wetness_response_pass"),
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
