//! `wgpu` compute twin of the breaking-wave classifier numeric core inside the
//! water breaking/spray contract
//! ([`breaking`](prism_render_architecture::water::breaking)).
//!
//! The `CPU` golden
//! [`breaking`](prism_render_architecture::water::breaking) turns three
//! per-sample surface metrics — the displacement-gradient `steepness`, the
//! choppy-displacement `jacobian` (a fold when at or below zero), and the crest
//! `curvature` — into a normalized breaking intensity, a qualitative
//! [`BreakingClass`](prism_render_architecture::water::breaking::BreakingClass),
//! and a foam-source strength. Those three evaluations are pure, deterministic
//! threshold arithmetic with no float equality: a pair of saturating ramps, an
//! average, a handful of magnitude comparisons, and one guarded multiply.
//!
//! [`GpuWaterBreakingClassify`] is the on-device twin of that numeric core. One
//! thread classifies one surface sample, reproducing the reference's exact
//! closed form, so a passing real-device parity test is direct evidence the
//! ported kernel takes the same discrete class branch and emits the same
//! intensity and foam strength the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For one sample the kernel reproduces, in order:
//!
//! 1. [`breaking_intensity`](prism_render_architecture::water::breaking::breaking_intensity):
//!    the average of three saturating ramps — a steepness ramp rising from its
//!    threshold `edge` to `2 * edge`, a fold ramp rising as the Jacobian drops
//!    below its threshold toward (and past) zero, and a curvature ramp — each
//!    clamped to `0..=1`, then the mean clamped to `0..=1`;
//! 2. [`classify_breaking`](prism_render_architecture::water::breaking::classify_breaking):
//!    a folded Jacobian (`jacobian <= jacobian_fold_threshold`) or an intensity
//!    at or above `breaking_intensity` is `Breaking`; otherwise a steepness past
//!    `steepness_threshold` is `Cresting`; everything else is `Calm`;
//! 3. [`foam_source_strength`](prism_render_architecture::water::breaking::foam_source_strength):
//!    zero unless the sample is `Breaking`, otherwise `intensity * max(max_rate, 0)`.
//!
//! The emitted result carries the continuous `intensity` and `foam` plus the
//! discrete `class` code (`0` `Calm`, `1` `Cresting`, `2` `Breaking`).
//!
//! # What stays on the host
//!
//! The crest-spray burst planner
//! [`plan_spray`](prism_render_architecture::water::breaking::plan_spray) — a
//! `Vec3` tangent/normal blend, a `normalize_or_zero`, and a round-to-`u32`
//! particle count — is not twinned here; it is a separate vector-valued plan the
//! host owns. The host enqueues one sample per thread; an empty batch
//! short-circuits with no dispatch, since a storage buffer cannot be zero-sized.
//!
//! # Correctness model
//!
//! The `class` code is a discrete decision and is asserted exactly (`==`). The
//! continuous `intensity` and `foam` thread through subtracts, divides, `clamp`
//! and one multiply, so the two engines agree to within a legal last-place
//! difference; the parity test asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on both. The only class comparison whose two sides are
//! *computed* (rather than copied inputs) is `intensity >= breaking_intensity`;
//! fixtures keep that margin well clear of zero so `CPU` and `GPU` cannot
//! straddle it.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`,
//! `+ - * /` and unsigned index arithmetic — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no `smoothstep`, no `round`
//! and no `sqrt`. No optional device feature is required, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a
//! fixed, bounded sequence of branches and arithmetic, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::breaking`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` breaking-wave classifier kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`breaking`](prism_render_architecture::water::breaking) numeric core for one
/// surface sample; see the module documentation for the rules.
const WATER_BREAKING_CLASSIFY_WGSL: &str = r#"
// Breaking-wave classifier twin: one thread turns one sample's steepness,
// Jacobian and curvature (plus its criteria thresholds and foam max_rate) into
// a normalized intensity, a discrete class code and a foam-source strength,
// mirroring the CPU golden `water::breaking` closed forms rule-for-rule with
// only clamp, max and + - * /. It owns no spray-burst plan; that vector-valued
// planner stays host-side.
//
// Provenance: 孪生自本仓 prism_render_architecture::water::breaking；无第三方
// 引擎源码或衍生代码。

// Shared displacement/length guard; mirrors the golden water module `EPS`.
const EPS: f32 = 1.0e-6;

struct Params {
    // Number of samples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Displacement-gradient steepness (slope magnitude).
    steepness: f32,
    // Choppy-displacement Jacobian (a fold when at or below zero).
    jacobian: f32,
    // Local crest curvature magnitude.
    curvature: f32,
    // Steepness at which cresting begins; contribution saturates at twice it.
    steepness_threshold: f32,
    // Jacobian at or below which a fold is counted.
    jacobian_fold_threshold: f32,
    // Curvature at which the crest is sharp enough to spume.
    curvature_threshold: f32,
    // Intensity at or above which the sample is classified as fully breaking.
    breaking_intensity: f32,
    // Foam-field max emission rate (per second); the breaking foam strength
    // scales with it. Clamped to non-negative in the kernel.
    max_rate: f32,
}

struct Result {
    // Normalized breaking intensity in [0, 1].
    intensity: f32,
    // Foam-source strength: 0 unless Breaking, else intensity * max(max_rate, 0).
    foam: f32,
    // Discrete class code: 0 Calm, 1 Cresting, 2 Breaking.
    class_code: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Saturating ramp: 0 at or below `edge`, 1 at or above 2 * edge. Guards the
// normalizer with EPS exactly as the golden `ramp_above` does.
fn ramp_above(value: f32, edge: f32) -> f32 {
    var scale: f32 = EPS;
    if (edge > EPS) {
        scale = edge;
    }
    return clamp((value - edge) / scale, 0.0, 1.0);
}

// Saturating ramp for a metric that grows as it drops below `edge`: 0 at or
// above `edge`, rising to 1 as the value falls to zero and clamped at 1 for a
// completed fold. Mirrors the golden `ramp_below`.
fn ramp_below(value: f32, edge: f32) -> f32 {
    var scale: f32 = EPS;
    if (edge > EPS) {
        scale = edge;
    }
    return clamp((edge - value) / scale, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let steep = ramp_above(q.steepness, q.steepness_threshold);
    let fold = ramp_below(q.jacobian, q.jacobian_fold_threshold);
    let curve = ramp_above(q.curvature, q.curvature_threshold);
    let intensity = clamp((steep + fold + curve) / 3.0, 0.0, 1.0);

    // Classify: a folded Jacobian or an intensity at/above the breaking
    // threshold is Breaking (2); otherwise any steepness past its threshold is
    // Cresting (1); everything else is Calm (0).
    let folded = q.jacobian <= q.jacobian_fold_threshold;
    var breaking_class: u32 = 0u;
    if (folded || intensity >= q.breaking_intensity) {
        breaking_class = 2u;
    } else if (q.steepness > q.steepness_threshold) {
        breaking_class = 1u;
    }

    // Foam source: 0 unless Breaking, else intensity * max(max_rate, 0).
    var foam: f32 = 0.0;
    if (breaking_class == 2u) {
        foam = intensity * max(q.max_rate, 0.0);
    }

    var out: Result;
    out.intensity = intensity;
    out.foam = foam;
    out.class_code = breaking_class;
    out.pad0 = 0u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the sample count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`WATER_BREAKING_CLASSIFY_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid samples in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one sample query, matching the `WGSL` `Query`
/// struct: the three surface metrics, the four criteria thresholds, and the
/// foam `max_rate`, flattened to a `32`-byte stride of eight `f32` words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Displacement-gradient steepness.
    steepness: f32,
    /// Choppy-displacement Jacobian.
    jacobian: f32,
    /// Local crest curvature magnitude.
    curvature: f32,
    /// Steepness threshold where cresting begins.
    steepness_threshold: f32,
    /// Jacobian fold threshold.
    jacobian_fold_threshold: f32,
    /// Curvature threshold for spume.
    curvature_threshold: f32,
    /// Intensity threshold for a fully breaking sample.
    breaking_intensity: f32,
    /// Foam max emission rate.
    max_rate: f32,
}

/// `repr(C)` `std430` layout of one sample result, matching the `WGSL` `Result`
/// struct: the continuous `intensity` and `foam`, the discrete `class_code`,
/// and one pad word to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Normalized breaking intensity in `[0, 1]`.
    intensity: f32,
    /// Foam-source strength.
    foam: f32,
    /// Discrete class code: `0` `Calm`, `1` `Cresting`, `2` `Breaking`.
    class_code: u32,
    /// Padding word.
    pad0: u32,
}

/// One per-sample query for the breaking-wave classifier twin: the three
/// surface metrics plus the four criteria thresholds and the foam `max_rate`,
/// flattened from the golden
/// [`BreakingSample`](prism_render_architecture::water::breaking::BreakingSample)
/// and
/// [`BreakingCriteria`](prism_render_architecture::water::breaking::BreakingCriteria).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterBreakingClassifyQuery {
    /// Displacement-gradient steepness (the golden `BreakingSample` `steepness`).
    pub steepness: f32,
    /// Choppy-displacement Jacobian (the golden `BreakingSample` `jacobian`).
    pub jacobian: f32,
    /// Local crest curvature (the golden `BreakingSample` `curvature`).
    pub curvature: f32,
    /// Steepness threshold (the golden `BreakingCriteria` `steepness_threshold`).
    pub steepness_threshold: f32,
    /// Jacobian fold threshold (the golden `BreakingCriteria`
    /// `jacobian_fold_threshold`).
    pub jacobian_fold_threshold: f32,
    /// Curvature threshold (the golden `BreakingCriteria` `curvature_threshold`).
    pub curvature_threshold: f32,
    /// Breaking intensity threshold (the golden `BreakingCriteria`
    /// `breaking_intensity`).
    pub breaking_intensity: f32,
    /// Foam max emission rate (the golden `foam_source_strength` `max_rate`).
    pub max_rate: f32,
}

impl WaterBreakingClassifyQuery {
    /// Builds a query from the three metrics, the four thresholds and the foam
    /// `max_rate`.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "the query flattens the golden sample, criteria and foam rate into one std430 record"
    )]
    pub const fn new(
        steepness: f32,
        jacobian: f32,
        curvature: f32,
        steepness_threshold: f32,
        jacobian_fold_threshold: f32,
        curvature_threshold: f32,
        breaking_intensity: f32,
        max_rate: f32,
    ) -> WaterBreakingClassifyQuery {
        WaterBreakingClassifyQuery {
            steepness,
            jacobian,
            curvature,
            steepness_threshold,
            jacobian_fold_threshold,
            curvature_threshold,
            breaking_intensity,
            max_rate,
        }
    }
}

/// The discrete breaking state the twin reports, mirroring the golden
/// [`BreakingClass`](prism_render_architecture::water::breaking::BreakingClass).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaterBreakingClass {
    /// Below all thresholds: a smooth, unbroken surface (golden `Calm`).
    Calm,
    /// Steep but not yet folded: a sharpening crest (golden `Cresting`).
    Cresting,
    /// Folded / high-intensity: an actively breaking crest (golden `Breaking`).
    Breaking,
}

impl WaterBreakingClass {
    /// Maps the kernel's `u32` class code back to the enum. Any out-of-range
    /// code is treated as `Calm`, though the kernel only emits `0`, `1` or `2`.
    #[must_use]
    fn from_code(code: u32) -> WaterBreakingClass {
        match code {
            1 => WaterBreakingClass::Cresting,
            2 => WaterBreakingClass::Breaking,
            _ => WaterBreakingClass::Calm,
        }
    }
}

/// One resolved sample of the breaking-wave classifier twin, mirroring the
/// golden
/// [`breaking`](prism_render_architecture::water::breaking) numeric core.
///
/// `intensity` is the normalized breaking intensity, `class` is the discrete
/// [`WaterBreakingClass`], and `foam` is the foam-source strength.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterBreakingClassifyResult {
    /// Normalized breaking intensity in `[0, 1]`.
    pub intensity: f32,
    /// Foam-source strength (`0` unless `Breaking`).
    pub foam: f32,
    /// Discrete breaking class.
    pub class: WaterBreakingClass,
}

/// Encodes one [`WaterBreakingClassifyQuery`] into its `std430` [`GpuQuery`]
/// slot. The layouts are field-for-field identical, so this is a direct copy.
fn encode_query(q: &WaterBreakingClassifyQuery) -> GpuQuery {
    GpuQuery {
        steepness: q.steepness,
        jacobian: q.jacobian,
        curvature: q.curvature,
        steepness_threshold: q.steepness_threshold,
        jacobian_fold_threshold: q.jacobian_fold_threshold,
        curvature_threshold: q.curvature_threshold,
        breaking_intensity: q.breaking_intensity,
        max_rate: q.max_rate,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`WaterBreakingClassifyResult`], mapping the `class_code` `u32` back to the
/// [`WaterBreakingClass`] enum.
fn decode_result(raw: &GpuResult) -> WaterBreakingClassifyResult {
    WaterBreakingClassifyResult {
        intensity: raw.intensity,
        foam: raw.foam,
        class: WaterBreakingClass::from_code(raw.class_code),
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

/// A compiled, reusable breaking-wave classifier compute pipeline, twinning the
/// numeric core of the `CPU` golden
/// [`breaking`](prism_render_architecture::water::breaking).
pub struct GpuWaterBreakingClassify {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterBreakingClassify {
    /// Compiles the breaking-wave classifier kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterBreakingClassify {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_breaking_classify"),
            source: ShaderSource::Wgsl(WATER_BREAKING_CLASSIFY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_breaking_classify_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_breaking_classify_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_breaking_classify_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterBreakingClassify {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies every sample in `queries` and returns one
    /// [`WaterBreakingClassifyResult`] per input, in order.
    ///
    /// Every output matches the reference exactly on the discrete `class` and
    /// within the tolerance documented on this module for the continuous
    /// `intensity` and `foam`. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterBreakingClassifyQuery],
    ) -> Vec<WaterBreakingClassifyResult> {
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
            label: Some("prism_volumetric_water_breaking_classify_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_breaking_classify_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_breaking_classify_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_breaking_classify_bind_group"),
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
            label: Some("prism_volumetric_water_breaking_classify_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_breaking_classify_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_breaking_classify_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
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
