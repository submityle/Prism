//! `wgpu` compute twin of the fractal cloud noise
//! ([`fbm`](prism_render_architecture::volumetric::noise::fbm) /
//! [`worley_fbm`](prism_render_architecture::volumetric::noise::worley_fbm)).
//!
//! The base cloud shape stacks several octaves of coherent noise with geometric
//! frequency growth (lacunarity) and amplitude decay (gain), amplitude-
//! normalised into `[0, 1]`. The [`crate::perlin_worley`] twin already exercises
//! these accumulators, but only at the fixed default octave counts baked into
//! that entry point. [`GpuNoiseFbm`] exposes the full
//! `(point, seed, octaves)` surface of the `CPU` goldens
//! [`fbm`](prism_render_architecture::volumetric::noise::fbm) and
//! [`worley_fbm`](prism_render_architecture::volumetric::noise::worley_fbm), so
//! a passing real-device parity test is direct evidence the ported kernel
//! reproduces the same fractal values at any octave count — including the
//! `octaves == 0` early-out that yields `0` rather than dividing by zero.
//!
//! # What the kernel evaluates
//!
//! For each [`NoiseFbmQuery`] the kernel returns both fractal fields at the
//! same point/seed/octave count:
//!
//! * `fbm` — `octaves` of Perlin noise, amplitude-normalised to `[0, 1]`;
//! * `worley_fbm` — `octaves` of billowy inverted-Worley noise in `[0, 1]`.
//!
//! Every gradient and feature point is selected by the same deterministic
//! integer hash (FNV-1a byte mixing plus an xorshift-multiply avalanche) as the
//! reference, and `WGSL` unsigned integers wrap on overflow exactly like Rust's
//! `wrapping_*`, so the lattice work is bit-identical. Only the float fBm
//! accumulation and the Worley `sqrt` can differ, and only by legal
//! multiply-add contraction.
//!
//! # Portability
//!
//! The kernel uses only `floor`, `sqrt`, `min`, `clamp` and multiply/add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The integer lattice work is bit-exact; the float accumulation is not, since
//! a `GPU` may fuse a multiply-add the scalar `CPU` reference leaves separate.
//! The parity test therefore asserts a tolerance (`abs_diff < 1e-5`) rather than
//! exact equality — tight enough that a wrong octave loop, a swapped field, or a
//! missing normalisation fails, loose enough that legal fma contraction passes.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Nubis-style Perlin / inverted-Worley fractal cloud
//! noise plus `wgpu` compute dispatch; no Unreal Engine source or derived code.

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

/// One fractal-noise query: the sample point, seed and octave count.
///
/// The fields are exactly the arguments of the `CPU` goldens
/// [`fbm`](prism_render_architecture::volumetric::noise::fbm) and
/// [`worley_fbm`](prism_render_architecture::volumetric::noise::worley_fbm):
///
/// * `x` / `y` / `z` — sample point coordinates;
/// * `seed` — deterministic noise seed;
/// * `octaves` — number of fractal octaves (`0` yields `0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseFbmQuery {
    /// Sample `x` coordinate.
    pub x: f32,
    /// Sample `y` coordinate.
    pub y: f32,
    /// Sample `z` coordinate.
    pub z: f32,
    /// Deterministic noise seed.
    pub seed: u32,
    /// Number of fractal octaves.
    pub octaves: u32,
}

/// Both fractal fields evaluated at one query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseFbmResult {
    /// `fbm(point, seed, octaves)`.
    pub fbm: f32,
    /// `worley_fbm(point, seed, octaves)`.
    pub worley_fbm: f32,
}

/// One query as uploaded. `24`-byte `repr(C)` matching `Query` in
/// `shaders/noise_fbm.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    x: f32,
    y: f32,
    z: f32,
    seed: u32,
    octaves: u32,
    pad0: u32,
}

/// One result as read back. `8`-byte `repr(C)` matching `Result` in
/// `shaders/noise_fbm.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    fbm: f32,
    worley_fbm: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/noise_fbm.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable fractal-noise pipeline.
pub struct GpuNoiseFbm {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuNoiseFbm {
    /// Compiles the fractal-noise kernel and builds the compute pipeline on
    /// `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuNoiseFbm {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_noise_fbm_module"),
            source: ShaderSource::Wgsl(include_str!("../shaders/noise_fbm.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_noise_fbm_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_noise_fbm_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_noise_fbm_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("noise_fbm_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuNoiseFbm {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates `fbm` and `worley_fbm` for every query in `queries`, returning
    /// one [`NoiseFbmResult`] per query in input order.
    ///
    /// The returned fields for query `q` equal
    /// [`fbm`](prism_render_architecture::volumetric::noise::fbm)`((q.x, q.y, q.z), q.seed, q.octaves)`
    /// and
    /// [`worley_fbm`](prism_render_architecture::volumetric::noise::worley_fbm)`((q.x, q.y, q.z), q.seed, q.octaves)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[NoiseFbmQuery]) -> Vec<NoiseFbmResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                x: q.x,
                y: q.y,
                z: q.z,
                seed: q.seed,
                octaves: q.octaves,
                pad0: 0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_noise_fbm_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_noise_fbm_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_noise_fbm_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_noise_fbm_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_noise_fbm_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_noise_fbm_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_noise_fbm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
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
        debug_assert_eq!(raw.len(), queries.len());
        raw.into_iter()
            .map(|r| NoiseFbmResult {
                fbm: r.fbm,
                worley_fbm: r.worley_fbm,
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
