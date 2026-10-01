//! `wgpu` compute twin of the flipbook sub-frame blending math
//! ([`flipbook_blend`](prism_render_architecture::particle::flipbook_blend),
//! design section 15).
//!
//! A smooth flipbook turns a stepped sprite-sheet animation into a
//! motion-blurred one by resolving a continuous frame *phase* into the pair of
//! adjacent cells to sample and the fractional weight between them. The `CPU`
//! golden
//! [`flipbook_blend`](prism_render_architecture::particle::flipbook_blend)
//! owns that math — [`blend_for_phase`](prism_render_architecture::particle::flipbook_blend::blend_for_phase),
//! [`blend_for_age`](prism_render_architecture::particle::flipbook_blend::blend_for_age),
//! [`blend_for_life`](prism_render_architecture::particle::flipbook_blend::blend_for_life),
//! [`AtlasLayout::uv_rect`](prism_render_architecture::particle::flipbook_blend::AtlasLayout::uv_rect)
//! and [`sample_rects`](prism_render_architecture::particle::flipbook_blend::sample_rects).
//! [`GpuFlipbookBlend`] is the on-device twin that runs one thread per sample
//! and reproduces the same `(frame_a, frame_b, blend)` triple and the same two
//! `UV` rectangles, so a passing real-device parity test is direct evidence the
//! ported kernel floors the same phase, wraps the same frame indices and lays
//! out the same atlas cells the reference does, not merely that its shader
//! compiles.
//!
//! # What is twinned
//!
//! One kernel covers the whole stack. Each sample carries a *kind* discriminant
//! selecting which reference entry point it mirrors: a raw frame `phase`
//! ([`blend_for_phase`](prism_render_architecture::particle::flipbook_blend::blend_for_phase)),
//! an `age_seconds` scaled by the batch `fps`
//! ([`blend_for_age`](prism_render_architecture::particle::flipbook_blend::blend_for_age)),
//! or a normalized `life` fraction scaled by the frame count
//! ([`blend_for_life`](prism_render_architecture::particle::flipbook_blend::blend_for_life)).
//! The kernel reduces each to a phase, resolves the two-frame blend, then maps
//! both frame indices through the `columns x rows`
//! [`AtlasLayout`](prism_render_architecture::particle::flipbook_blend::AtlasLayout)
//! exactly as
//! [`sample_rects`](prism_render_architecture::particle::flipbook_blend::sample_rects)
//! does, emitting the blend plus both `UV` rectangles per sample.
//!
//! # Correctness model
//!
//! The frame indices are integer arithmetic — a `floor`, a `u32` cast, a
//! modulo or a clamp — so they reproduce *bit for bit* and the parity test
//! asserts them with exact `==`. The blend weight and the `UV` rectangle
//! coordinates are a `floor`-based fract and a few multiplies and divides by the
//! integer grid dimensions; `CPU` and `GPU` evaluate the same closed form, but a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few `ULP`. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! fields, tight enough to catch a genuinely wrong port (a swapped wrap branch,
//! a transposed `col`/`row`, a dropped `+ 1` frame) yet loose enough to admit
//! legal fused multiply-add contraction.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `+ - * /`, modulo and unsigned integer index arithmetic —
//! with no `sin`, `cos`, `exp`, `log`, `pow` or optional device feature, so the
//! frame wrap is reproduced with integer modulo rather than any transcendental,
//! and the twin runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Prism's own flipbook sub-frame blend design (section 15) plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::flipbook_blend::{AtlasLayout, FlipbookBlend, WrapMode};
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
/// default used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` flipbook blend kernel, embedded inline so the twin
/// ships as a single source file. The entry point `flipbook_main` mirrors the
/// `CPU` golden `particle::flipbook_blend` sample for sample; see the module
/// documentation for the algorithm.
const FLIPBOOK_BLEND_WGSL: &str = r#"
// Flipbook sub-frame blend twin: one thread per sample resolves a continuous
// frame phase into a two-frame (frame_a, frame_b, blend) triple and the two
// atlas UV rectangles. It mirrors the CPU golden `particle::flipbook_blend`,
// uses only the portable core-WGSL subset (min/max/clamp/floor and + - * /
// plus unsigned modulo/index math), takes no optional feature, and reproduces
// the frame wrap with integer modulo rather than any transcendental, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: Prism flipbook sub-frame blend design (section 15); no
// third-party engine source or derived code.

struct Params {
    // Number of samples in the batch (one thread each).
    count: u32,
    // Total animation frames; 0 means a single static cell.
    frames: u32,
    // Atlas columns (0 treated as 1 to avoid divide-by-zero).
    columns: u32,
    // Atlas rows (0 treated as 1 to avoid divide-by-zero).
    rows: u32,
    // Frames per second, shared by every `age` sample in the batch.
    fps: f32,
    // Wrap mode: 0 = Clamp (hold last frame), 1 = Loop (modulo wrap).
    wrap: u32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

struct Sample {
    // The raw input: a phase, an age in seconds, or a life fraction.
    value: f32,
    // Kind discriminant: 0 = phase, 1 = age, 2 = life.
    kind: u32,
    pad0: u32,
    pad1: u32,
}

struct Result {
    // The floor frame the phase currently sits on.
    frame_a: u32,
    // The next frame the phase advances toward.
    frame_b: u32,
    // Fractional weight in 0.0..=1.0 from frame_a toward frame_b.
    blend: f32,
    pad0: u32,
    // [u_min, v_min, u_max, v_max] of frame_a.
    rect_a: vec4<f32>,
    // [u_min, v_min, u_max, v_max] of frame_b.
    rect_b: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> samples: array<Sample>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Kind discriminants, mirroring the host enum ordinals.
const KIND_PHASE: u32 = 0u;
const KIND_AGE: u32 = 1u;
const KIND_LIFE: u32 = 2u;

// Wrap-mode discriminants.
const WRAP_CLAMP: u32 = 0u;

// Saturating `+ 1`, mirroring the reference `u32::saturating_add(1)` so a phase
// floored to u32::MAX does not wrap back to zero.
fn sat_add_one(x: u32) -> u32 {
    if (x == 0xffffffffu) {
        return x;
    }
    return x + 1u;
}

// Maps a flat frame index onto its [u_min, v_min, u_max, v_max] UV rectangle,
// mirroring `AtlasLayout::uv_rect`: col = frame % columns, row = frame /
// columns, each cell spanning 1/columns in U and 1/rows in V, with V running
// top-to-bottom. A 0 dimension is treated as 1.
fn uv_rect(frame_index: u32) -> vec4<f32> {
    let columns = max(params.columns, 1u);
    let rows = max(params.rows, 1u);
    let col = frame_index % columns;
    let row = frame_index / columns;
    let inv_columns = 1.0 / f32(columns);
    let inv_rows = 1.0 / f32(rows);
    let u_min = f32(col) * inv_columns;
    let v_min = f32(row) * inv_rows;
    return vec4<f32>(u_min, v_min, u_min + inv_columns, v_min + inv_rows);
}

@compute @workgroup_size(64)
fn flipbook_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let s = samples[idx];

    // Reduce each sample kind to a frame phase, mirroring the reference entry
    // points: `blend_for_age` scales a clamped age by a clamped fps, and
    // `blend_for_life` scales a clamped life fraction by the frame count.
    var phase: f32 = s.value;
    if (s.kind == KIND_AGE) {
        phase = max(s.value, 0.0) * max(params.fps, 0.0);
    } else if (s.kind == KIND_LIFE) {
        phase = clamp(s.value, 0.0, 1.0) * f32(params.frames);
    }

    var frame_a: u32 = 0u;
    var frame_b: u32 = 0u;
    var blend: f32 = 0.0;

    // `frames == 0` is a single static cell: both frames 0, blend 0.
    if (params.frames != 0u) {
        let clamped = max(phase, 0.0);
        let base = floor(clamped);
        // `clamped >= 0` and `base <= clamped`, so the fract lands in 0.0..1.0.
        blend = clamp(clamped - base, 0.0, 1.0);
        // `base >= 0`, so the cast floors toward zero safely.
        let raw = u32(base);
        let last = params.frames - 1u;
        let next = sat_add_one(raw);
        if (params.wrap == WRAP_CLAMP) {
            frame_a = min(raw, last);
            frame_b = min(next, last);
        } else {
            frame_a = raw % params.frames;
            frame_b = next % params.frames;
        }
    }

    results[idx].frame_a = frame_a;
    results[idx].frame_b = frame_b;
    results[idx].blend = blend;
    results[idx].pad0 = 0u;
    results[idx].rect_a = uv_rect(frame_a);
    results[idx].rect_b = uv_rect(frame_b);
}
"#;

/// The per-sample input to a flipbook blend batch, selecting which reference
/// entry point the sample mirrors.
///
/// The frame count, `fps`, wrap mode and atlas layout are shared by the whole
/// [`FlipbookQuery`], so a sample varies only in its raw value and kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlipbookSample {
    /// A raw floating frame counter, mirroring
    /// [`blend_for_phase`](prism_render_architecture::particle::flipbook_blend::blend_for_phase).
    Phase(f32),
    /// A particle `age_seconds`, scaled by the batch `fps`, mirroring
    /// [`blend_for_age`](prism_render_architecture::particle::flipbook_blend::blend_for_age).
    Age(f32),
    /// A normalized `life` fraction in `0.0..=1.0`, scaled by the frame count,
    /// mirroring
    /// [`blend_for_life`](prism_render_architecture::particle::flipbook_blend::blend_for_life).
    Life(f32),
}

/// A resolved two-frame blend plus the two atlas `UV` rectangles a shader needs
/// for a two-sample `lerp`, the per-sample output of [`GpuFlipbookBlend::eval`].
///
/// Mirrors the tuple
/// [`sample_rects`](prism_render_architecture::particle::flipbook_blend::sample_rects)
/// returns: the `(frame_a, frame_b, blend)` triple carried by `blend`, plus
/// `rect_a` and `rect_b` as `[u_min, v_min, u_max, v_max]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipbookResult {
    /// The resolved two-frame blend (`frame_a`, `frame_b`, `blend`).
    pub blend: FlipbookBlend,
    /// The `[u_min, v_min, u_max, v_max]` `UV` rectangle of `frame_a`.
    pub rect_a: [f32; 4],
    /// The `[u_min, v_min, u_max, v_max]` `UV` rectangle of `frame_b`.
    pub rect_b: [f32; 4],
}

/// One flipbook blend batch: the shared atlas and animation configuration plus
/// the per-sample inputs.
///
/// Every sample shares the same `layout`, `frames`, `fps` and `wrap`, matching
/// the reference's per-call parameters; only the [`FlipbookSample`] value and
/// kind vary across the batch.
#[derive(Clone, Debug, PartialEq)]
pub struct FlipbookQuery {
    /// The `columns x rows` atlas layout used to map frame indices to `UV`
    /// rectangles.
    pub layout: AtlasLayout,
    /// Total animation frames; `0` collapses to a single static cell.
    pub frames: u32,
    /// Frames per second, applied to every [`FlipbookSample::Age`] sample.
    pub fps: f32,
    /// How the flipbook behaves once the phase runs past the last frame.
    pub wrap: WrapMode,
    /// The per-sample inputs, evaluated one GPU thread each.
    pub samples: Vec<FlipbookSample>,
}

/// One sample as uploaded. `16`-byte `repr(C)` matching `Sample` in
/// [`FLIPBOOK_BLEND_WGSL`]: the raw value, the kind discriminant and two pad
/// words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    /// The raw input value (phase, age or life).
    value: f32,
    /// Kind discriminant: `0` phase, `1` age, `2` life.
    kind: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One result as read back. `48`-byte `repr(C)` matching `Result` in
/// [`FLIPBOOK_BLEND_WGSL`]: the two frame indices, the blend weight, a pad word
/// and the two `UV` rectangles.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The floor frame index.
    frame_a: u32,
    /// The next frame index.
    frame_b: u32,
    /// The fractional blend weight.
    blend: f32,
    /// Padding word aligning the following `vec4` to `16` bytes.
    pad0: u32,
    /// The `[u_min, v_min, u_max, v_max]` rectangle of `frame_a`.
    rect_a: [f32; 4],
    /// The `[u_min, v_min, u_max, v_max]` rectangle of `frame_b`.
    rect_b: [f32; 4],
}

/// Uniform parameters for one dispatch. `32`-byte `repr(C)` matching `Params`
/// in [`FLIPBOOK_BLEND_WGSL`]: the sample count, the frame count, the atlas
/// dimensions, the `fps`, the wrap ordinal and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of samples in the batch.
    count: u32,
    /// Total animation frames.
    frames: u32,
    /// Atlas columns.
    columns: u32,
    /// Atlas rows.
    rows: u32,
    /// Frames per second shared across the batch.
    fps: f32,
    /// Wrap-mode ordinal: `0` Clamp, `1` Loop.
    wrap: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// Maps a [`WrapMode`] to the shader wrap ordinal (`0` Clamp, `1` Loop).
fn wrap_ordinal(wrap: WrapMode) -> u32 {
    match wrap {
        WrapMode::Clamp => 0,
        WrapMode::Loop => 1,
    }
}

/// Packs one [`FlipbookSample`] into its uploaded value/kind pair.
fn pack_sample(sample: FlipbookSample) -> GpuSample {
    let (value, kind) = match sample {
        FlipbookSample::Phase(v) => (v, 0u32),
        FlipbookSample::Age(v) => (v, 1u32),
        FlipbookSample::Life(v) => (v, 2u32),
    };
    GpuSample {
        value,
        kind,
        pad0: 0,
        pad1: 0,
    }
}

/// A compiled, reusable flipbook blend pipeline.
pub struct GpuFlipbookBlend {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFlipbookBlend {
    /// Compiles the flipbook blend kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFlipbookBlend {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_flipbook_blend"),
            source: ShaderSource::Wgsl(FLIPBOOK_BLEND_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_flipbook_blend_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_flipbook_blend_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_flipbook_blend_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("flipbook_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFlipbookBlend {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every sample in `query`, returning one [`FlipbookResult`] per
    /// sample in input order.
    ///
    /// For a sample `s` the result's `blend` equals the reference
    /// `blend_for_phase`/`blend_for_age`/`blend_for_life` for that kind, and
    /// `rect_a`/`rect_b` equal
    /// [`sample_rects`](prism_render_architecture::particle::flipbook_blend::sample_rects)`(&query.layout, &blend)`.
    /// An empty `query.samples` yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return with no dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &FlipbookQuery) -> Vec<FlipbookResult> {
        if query.samples.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_samples: Vec<GpuSample> = query.samples.iter().map(|&s| pack_sample(s)).collect();

        let gpu_params = GpuParams {
            count: query.samples.len() as u32,
            frames: query.frames,
            columns: query.layout.columns,
            rows: query.layout.rows,
            fps: query.fps,
            wrap: wrap_ordinal(query.wrap),
            pad0: 0,
            pad1: 0,
        };

        let out_bytes = (query.samples.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_flipbook_blend_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let samples_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_flipbook_blend_samples"),
            contents: bytemuck::cast_slice(&gpu_samples),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_flipbook_blend_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_flipbook_blend_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_flipbook_blend_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: samples_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_flipbook_blend_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_flipbook_blend_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
            let groups = (query.samples.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), query.samples.len());

        raw.into_iter()
            .map(|r| FlipbookResult {
                blend: FlipbookBlend {
                    frame_a: r.frame_a,
                    frame_b: r.frame_b,
                    blend: r.blend,
                },
                rect_a: r.rect_a,
                rect_b: r.rect_b,
            })
            .collect()
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
