//! `wgpu` compute twin of the screen-space ambient-occlusion depth fold
//! ([`ao_from_samples`](prism_render_architecture::particle::ao_sample::ao_from_samples)).
//!
//! The `CPU` golden
//! [`ao_from_samples`](prism_render_architecture::particle::ao_sample::ao_from_samples)
//! folds a set of scene depths fetched around a shaded point into a single
//! ambient-occlusion term in `0..=1`: a sample occludes when it is nearer the
//! camera than the shaded point (`center_depth - d > 0`), weighted by a
//! `smoothstep(0, 1, range / delta)` soft range check so occluders far beyond
//! `range` fade out rather than cut off hard. The mean occlusion is inverted to
//! the convention **`1.0` = fully lit / unoccluded**, **`0.0` = fully
//! occluded**; an empty sample set is unoccluded (`1.0`).
//!
//! [`GpuParticleAoSamples`] is the on-device twin of exactly that fold. One
//! thread solves one whole query: it walks that query's slice of the shared
//! flattened depth buffer **in input order**, so the floating-point summation
//! matches the reference's left-to-right accumulation with no reorderable
//! reduction, and reproduces the `delta > 0` occlusion gate, the inlined
//! `smoothstep` weight, the mean, and the final `clamp01(1 - mean)` inversion.
//! A passing real-device parity test is therefore direct evidence the ported
//! kernel computes the same occlusion the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For one query the twin reproduces the single scalar occlusion term over a
//! variable-length depth set. The host flattens the batch's variable-length
//! `depths` slices into one shared storage buffer and hands each query an
//! `offset` and `count` into it, mirroring the reference's slice iteration. A
//! query with no depths returns `1.0` exactly as the reference's empty-slice
//! short-circuit does.
//!
//! # What stays on the host
//!
//! The variable-length depth gather itself — which pixels are sampled, the
//! kernel offsets, and the scene-depth fetches that fill each `depths` slice —
//! stays host (or upstream-pass) work; the device sees only the resolved depth
//! values, the shaded point's `center_depth`, and the `range`.
//!
//! # Correctness model
//!
//! The fold threads through only `+ - * /`, `clamp`, and comparisons with no
//! `sqrt` and no transcendental, and the per-query accumulation is serial, so
//! the `CPU` and `GPU` evaluate the same expression in the same order. They are
//! not bit-exact — a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate — so results are compared within `abs_diff <= 1e-5` or
//! `rel_diff <= 1e-4`.
//!
//! Since `smoothstep`'s interval is the fixed `edge0 = 0`, `edge1 = 1`, its span
//! is the constant `1.0`, always above the reference's degenerate-interval
//! guard, so the weight is simply `t * t * (3 - 2 * t)` with `t = clamp(range /
//! delta, 0, 1)` — the reference's exact closed form on this interval.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, the four
//! arithmetic operators, bounded loops and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no
//! `round`, no `sqrt`, and no optional device feature, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ao_sample`；无第三方引擎源码或衍生代码。
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
/// used across this crate's one-thread-per-element kernels; here one element is
/// one whole ambient-occlusion depth fold.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` ambient-occlusion depth-fold kernel, embedded inline
/// so the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden
/// [`ao_from_samples`](prism_render_architecture::particle::ao_sample::ao_from_samples);
/// see the module documentation for the algorithm.
const PARTICLE_AO_SAMPLES_WGSL: &str = r#"
// Ambient-occlusion depth-fold twin: one thread folds one query's slice of the
// shared flattened depth buffer in input order, mirroring the CPU golden
// `particle::ao_sample::ao_from_samples` using only clamp and + - * /. It owns
// no variable-length depth gather; that stays on the host.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::ao_sample；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this return.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Start of this query's depth run in the shared flattened depth buffer.
    offset: u32,
    // Number of depth samples for this query (0 -> unoccluded 1.0).
    sample_count: u32,
    // Depth of the shaded point; larger means farther from the camera.
    center_depth: f32,
    // Soft range check width fed to the smoothstep weight.
    range: f32,
}

struct AoResult {
    // Ambient-occlusion term in 0..=1 (1 = fully lit, 0 = fully occluded).
    ao: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read> depths: array<f32>;
@group(0) @binding(3) var<storage, read_write> results: array<AoResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var out: AoResult;
    out.pad0 = 0u;
    out.pad1 = 0u;
    out.pad2 = 0u;

    // Empty sample set is unoccluded, matching the reference's empty-slice
    // short-circuit.
    if (q.sample_count == 0u) {
        out.ao = 1.0;
        results[idx] = out;
        return;
    }

    // Serial, input-order accumulation so the floating-point summation matches
    // the reference's left-to-right fold with no reorderable reduction.
    var occlusion = 0.0;
    for (var i = 0u; i < q.sample_count; i = i + 1u) {
        let d = depths[q.offset + i];
        let delta = q.center_depth - d;
        if (delta > 0.0) {
            // smoothstep(0, 1, range / delta): span is the constant 1.0, always
            // above the reference's degenerate-interval guard, so the weight is
            // just t*t*(3 - 2t) with t = clamp(range / delta, 0, 1).
            let t = clamp(q.range / delta, 0.0, 1.0);
            occlusion = occlusion + t * t * (3.0 - 2.0 * t);
        }
    }

    let mean = occlusion / f32(q.sample_count);
    out.ao = clamp(1.0 - mean, 0.0, 1.0);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`PARTICLE_AO_SAMPLES_WGSL`].
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the depth-run offset and count into the shared flattened depth buffer, the
/// shaded point's `center_depth`, and the soft-range width. The four scalars
/// already fill a `16`-byte stride, so no padding is needed.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Start of this query's depth run in the shared flattened depth buffer.
    offset: u32,
    /// Number of depth samples for this query.
    sample_count: u32,
    /// Depth of the shaded point.
    center_depth: f32,
    /// Soft range check width.
    range: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `AoResult`
/// struct: the ambient-occlusion term plus three pad words to a
/// `16`-byte-multiple stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Ambient-occlusion term in `0..=1`.
    ao: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One ambient-occlusion depth-fold query: the scene depths fetched around a
/// shaded point plus that point's depth and the soft-range width.
///
/// The host owns the variable-length depth gather; it hands over the resolved
/// `depths` (one scene depth per kernel offset, larger meaning farther from the
/// camera), the shaded point's `center_depth`, and the `range`. An empty
/// `depths` set resolves to the unoccluded `1.0`, mirroring the reference.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleAoSamplesQuery {
    /// Scene depths fetched around the shaded point.
    pub depths: Vec<f32>,
    /// Depth of the shaded point; larger means farther from the camera.
    pub center_depth: f32,
    /// Soft range check width fed to the `smoothstep` weight.
    pub range: f32,
}

/// One resolved ambient-occlusion term, mirroring the reference
/// [`ao_from_samples`](prism_render_architecture::particle::ao_sample::ao_from_samples)
/// outcome.
///
/// `ao` follows the convention `1.0` = fully lit / unoccluded, `0.0` = fully
/// occluded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleAoSamplesResult {
    /// Ambient-occlusion term in `0..=1`.
    pub ao: f32,
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

/// A compiled, reusable ambient-occlusion depth-fold compute pipeline, twinning
/// the `CPU` golden
/// [`ao_from_samples`](prism_render_architecture::particle::ao_sample::ao_from_samples).
pub struct GpuParticleAoSamples {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuParticleAoSamples {
    /// Compiles the ambient-occlusion depth-fold kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuParticleAoSamples {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_particle_ao_samples"),
            source: ShaderSource::Wgsl(PARTICLE_AO_SAMPLES_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuParticleAoSamples {
            module,
            layout,
            pipeline,
        }
    }

    /// Folds every query in `queries` and returns one
    /// [`ParticleAoSamplesResult`] per input, in order.
    ///
    /// The ambient-occlusion term equals the reference to within the tolerance
    /// documented on this module. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ParticleAoSamplesQuery],
    ) -> Vec<ParticleAoSamplesResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        // Flatten every query's variable-length depths into one shared buffer,
        // recording each query's offset and count.
        let mut flat_depths: Vec<f32> = Vec::new();
        let encoded: Vec<GpuQuery> = queries
            .iter()
            .map(|q| {
                let offset = flat_depths.len() as u32;
                flat_depths.extend_from_slice(&q.depths);
                GpuQuery {
                    offset,
                    sample_count: q.depths.len() as u32,
                    center_depth: q.center_depth,
                    range: q.range,
                }
            })
            .collect();
        // A storage buffer cannot be zero-sized; a single dummy depth is never
        // read because every empty query has `sample_count == 0`.
        if flat_depths.is_empty() {
            flat_depths.push(0.0);
        }

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });
        let depths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_depths"),
            contents: bytemuck::cast_slice(&flat_depths),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_bind_group"),
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
                    resource: depths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_particle_ao_samples_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_particle_ao_samples_pass"),
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

        raw.iter()
            .map(|r| ParticleAoSamplesResult { ao: r.ao })
            .collect()
    }
}
