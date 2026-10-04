//! `wgpu` compute twin of the hair near-/far-field scatter LOD kernel,
//! mirroring this repository's `prism_render_architecture::hair::scatter_lod`
//! module.
//!
//! A production hair renderer evaluates its scattering lobes under two regimes
//! depending on how large a fibre projects on screen (the `d'Eon` 2011
//! energy-conserving near-/far-field split): per-fibre near field when the
//! projected width is at or above `near_px`, an analytic widened far field at
//! or below `far_px`, and a continuous blend between the two so the LOD ladder
//! crosses the boundary without a pop. This twin reproduces, for one query per
//! thread, the three closed forms of that path that take scalars in and
//! scalars out:
//!
//! - `scatter_blend`: the continuous far-field blend factor in `[0, 1]`
//!   (`0` = pure near-field, `1` = pure far-field), a monotone non-increasing
//!   ramp across the transition band with a hard step for a degenerate band.
//! - `scatter_regime`: the discrete regime (`Near = 0`, `Far = 1`,
//!   `Blended = 2`) classified from the blend by the `EPS` / `1 - EPS` knees.
//! - `far_field_roughness_gain`: the lobe-widening gain `1 + clamp01(blend) *
//!   sanitize_nonneg(max_gain)`, always `>= 1`.
//!
//! The array-batched lobe-weight / pdf / stratified-sampling helpers of the
//! golden module are intentionally not twinned; they use dynamic vectors and a
//! random stream, whereas this kernel is a stateless scalar map, one thread per
//! query.
//!
//! # Finite sanitisation
//!
//! The golden path routes every input through `sanitize_nonneg` (non-finite or
//! negative becomes `0`) or `clamp01` (non-finite becomes `0`, else clamp to
//! `[0, 1]`), and treats a non-finite width as the finest footprint (blend
//! `1`). `WGSL` has no `isFinite`, so the kernel implements it as `(x == x) &&
//! (abs(x) < 3.4e38)`: the `x == x` self-compare is false only for `NaN` and is
//! the one deliberate bare-equality idiom used here, documented in the shader;
//! the magnitude test rejects the infinities. The host oracle uses the
//! equivalent `f32::is_finite`, so a `NaN` or infinite input sanitises to the
//! same deterministic value on both sides and parity holds without a bare `f32`
//! ordering equality.
//!
//! # Precision model
//!
//! The golden path evaluates in `f32`; this twin and its host oracle both
//! evaluate the same closed form in `f32`, each query a fixed, non-reorderable
//! sequence of absolute values, ordered compares, one divide and one
//! multiply-add. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! units in the last place. The parity test therefore asserts `abs_diff <=
//! 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on the continuous `blend` and
//! `roughness_gain` and an exact `==` on the discrete `regime` and `valid`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `clamp`,
//! `max`, `select` and `+ - * /` — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, `round` or optional device feature, and no `u64`, `i64` or `f64`, so
//! it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::scatter_lod`；无第三方引擎源码或衍生代码。

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly
/// default shared by every one-thread-per-element kernel in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// The reference epsilon bracketing the blend knees and the degenerate-band
/// fall-through, matching the golden module's `EPS`.
pub const EPS: f32 = 1.0e-6;

/// Inlined `WGSL` compute shader source. Keeping it in the Rust binary avoids
/// shipping a sidecar asset and keeps the twin and its kernel versioned as a
/// single source file. The single entry point `scatter_lod` mirrors the
/// threshold sanitisation, the continuous blend ramp, the discrete regime
/// classification and the roughness gain of the `CPU` golden module; see the
/// module documentation for the algorithm.
const HAIR_SCATTER_LOD_WGSL: &str = r#"
// Hair scatter LOD twin: one thread per query evaluates the sanitized
// thresholds, the continuous far-field blend, the discrete scatter regime and
// the far-field roughness gain. It uses only the portable core-WGSL subset
// (abs/clamp/max/select and + - * /) with no u64/i64/f64 and no transcendental,
// so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::hair::scatter_lod；无第三方引擎源码或衍生代码。

const EPS: f32 = 1e-6;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the projected fibre width, the near/far pixel thresholds and the
// maximum roughness gain. Four f32 lanes pack to exactly 16 bytes.
struct Query {
    fiber_width_px: f32,
    near_px: f32,
    far_px: f32,
    max_gain: f32,
}

// One result: the continuous blend, the continuous roughness gain, the discrete
// regime code (0 Near, 1 Far, 2 Blended) and the always-1 valid flag.
struct Res {
    blend: f32,
    roughness_gain: f32,
    regime: u32,
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

// x.is_finite(): a NaN fails the self-compare x == x (the one deliberate bare
// equality idiom here) and the infinities fail the magnitude test, so only a
// finite value passes. Mirrors the golden f32::is_finite without an ordering
// equality.
fn is_finite(x: f32) -> bool {
    return (x == x) && (abs(x) < 3.4e38);
}

// Non-finite or negative -> 0, else x. Golden sanitize_nonneg.
fn sanitize_nonneg(x: f32) -> f32 {
    return select(0.0, max(x, 0.0), is_finite(x));
}

// Non-finite -> 0, else clamp to [0, 1]. Golden clamp01.
fn clamp01(x: f32) -> f32 {
    return select(0.0, clamp(x, 0.0, 1.0), is_finite(x));
}

// Continuous far-field blend for a projected fibre width. Thresholds are
// sanitized to near >= far >= 0 first; the width is sanitized to a finite,
// non-negative value (non-finite -> 0 -> blend 1).
fn scatter_blend(fiber_width_px: f32, near_in: f32, far_in: f32) -> f32 {
    let far = sanitize_nonneg(far_in);
    let near = max(sanitize_nonneg(near_in), far);
    let w = sanitize_nonneg(fiber_width_px);
    if (w >= near) {
        return 0.0;
    }
    if (w <= far) {
        return 1.0;
    }
    let span = near - far;
    if (span <= EPS) {
        return 1.0;
    }
    return clamp((near - w) / span, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn scatter_lod(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let blend = scatter_blend(q.fiber_width_px, q.near_px, q.far_px);

    var regime: u32 = 2u;
    if (blend <= EPS) {
        regime = 0u;
    } else if (blend >= 1.0 - EPS) {
        regime = 1u;
    }

    let gain = 1.0 + clamp01(blend) * sanitize_nonneg(q.max_gain);

    var out: Res;
    out.blend = blend;
    out.roughness_gain = gain;
    out.regime = regime;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// One hair scatter-LOD query: the projected fibre width to classify, the near
/// and far pixel thresholds that bracket the transition and the maximum
/// far-field roughness gain.
///
/// None of the fields need be finite or positive; the kernel sanitizes them
/// before use (non-finite or negative widths/thresholds collapse to `0`, and
/// the thresholds are reordered to `near_px >= far_px >= 0`). Derives only
/// [`PartialEq`] (no [`Eq`] / [`Hash`]) because it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairScatterLodQuery {
    /// Projected fibre width in pixels; non-finite / negative is sanitized to
    /// `0` (the finest, most far-field footprint).
    pub fiber_width_px: f32,
    /// At or above this width the fibre is pure near-field.
    pub near_px: f32,
    /// At or below this width the fibre is pure far-field.
    pub far_px: f32,
    /// Maximum far-field roughness gain; negative / non-finite is treated as
    /// `0` (no widening).
    pub max_gain: f32,
}

impl HairScatterLodQuery {
    /// Builds a query from a width, the near/far thresholds and a maximum gain.
    #[must_use]
    pub const fn new(
        fiber_width_px: f32,
        near_px: f32,
        far_px: f32,
        max_gain: f32,
    ) -> HairScatterLodQuery {
        HairScatterLodQuery {
            fiber_width_px,
            near_px,
            far_px,
            max_gain,
        }
    }
}

/// The hair scatter-LOD outputs for one query, the host-side mirror of the
/// kernel's `Res` lane.
///
/// `blend` is the continuous far-field factor in `[0, 1]`, `roughness_gain` the
/// lobe-widening gain (`>= 1`), `regime` the discrete code (`0` Near, `1` Far,
/// `2` Blended) and `valid` the always-`1` flag. Derives only [`PartialEq`]
/// (no [`Eq`] / [`Hash`]) because it holds `f32` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairScatterLodResult {
    /// Continuous far-field blend in `[0, 1]`.
    pub blend: f32,
    /// Discrete scatter regime: `0` Near, `1` Far, `2` Blended.
    pub regime: u32,
    /// Far-field roughness gain, always `>= 1`.
    pub roughness_gain: f32,
    /// Always `1`: inputs sanitize deterministically, so there is no rejection.
    pub valid: u32,
}

/// `repr(C)` `std430` layout of one packed query: the width, the near and far
/// thresholds and the maximum gain — `16` bytes, exactly as the `WGSL` `Query`
/// struct reads it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Projected fibre width.
    fiber_width_px: f32,
    /// Near threshold.
    near_px: f32,
    /// Far threshold.
    far_px: f32,
    /// Maximum roughness gain.
    max_gain: f32,
}

impl GpuQuery {
    /// Packs one query into its `std430` image.
    fn new(query: &HairScatterLodQuery) -> GpuQuery {
        GpuQuery {
            fiber_width_px: query.fiber_width_px,
            near_px: query.near_px,
            far_px: query.far_px,
            max_gain: query.max_gain,
        }
    }
}

/// `repr(C)` `std430` layout of one result: the blend, the roughness gain, the
/// regime code and the `valid` flag in the same order as the `WGSL` `Res`
/// struct — `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Continuous blend.
    blend: f32,
    /// Continuous roughness gain.
    roughness_gain: f32,
    /// Discrete regime code.
    regime: u32,
    /// Always-`1` valid flag.
    valid: u32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// round the uniform block out to `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Decodes one packed `GpuResult` into the public [`HairScatterLodResult`].
fn decode_result(raw: &GpuResult) -> HairScatterLodResult {
    HairScatterLodResult {
        blend: raw.blend,
        regime: raw.regime,
        roughness_gain: raw.roughness_gain,
        valid: raw.valid,
    }
}

/// Builds one storage/uniform buffer bind-group-layout entry.
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

/// On-device twin of the hair scatter-LOD kernel.
///
/// Owns the compiled [`ComputePipeline`] and its [`BindGroupLayout`]; build it
/// once with [`GpuHairScatterLod::new`] and reuse it across
/// [`GpuHairScatterLod::evaluate`] calls.
pub struct GpuHairScatterLod {
    /// The compiled shader module (retained so the pipeline stays valid).
    #[expect(
        dead_code,
        reason = "retained so the compiled module outlives the pipeline"
    )]
    module: ShaderModule,
    /// The bind group layout shared by every dispatch.
    layout: BindGroupLayout,
    /// The compute pipeline running the `scatter_lod` entry point.
    pipeline: ComputePipeline,
}

impl GpuHairScatterLod {
    /// Compiles the kernel and builds the reusable pipeline on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairScatterLod {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_shader"),
            source: ShaderSource::Wgsl(HAIR_SCATTER_LOD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("scatter_lod"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairScatterLod {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every hair scatter-LOD query on-device and returns one
    /// [`HairScatterLodResult`] per input, in order.
    ///
    /// Each result equals the reference closed form to within the tolerance
    /// documented on this module. An empty input returns an empty vector with
    /// no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[HairScatterLodQuery],
    ) -> Vec<HairScatterLodResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_hair_scatter_lod_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_hair_scatter_lod_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
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
        debug_assert_eq!(raw.len(), count);

        raw.iter().map(decode_result).collect()
    }
}
