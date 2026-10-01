//! `wgpu` compute twin of the animated film-grain post-process
//! ([`FilmGrainParams::apply`](prism_render_architecture::particle::film_grain::FilmGrainParams::apply),
//! design §16, §21).
//!
//! Physical film records light on silver-halide crystals of finite size, so a
//! developed frame carries a stochastic, per-frame speckle whose visibility is
//! luminance dependent: shadows and midtones show the grain plainly while
//! highlights bleach it out. The `CPU` golden
//! [`film_grain`](prism_render_architecture::particle::film_grain) owns that
//! math; [`GpuFilmGrain`] is the on-device twin that runs one thread per pixel
//! and reproduces the identical composited `RGB` the batch form produces, so a
//! passing real-device parity test is direct evidence the ported kernel hashes
//! the same cells, applies the same luminance response and composites the same
//! blend the reference does — not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The whole four-stage pipeline is reproduced:
//!
//! * the `xor-shift` + odd-multiply avalanche
//!   [`hash_u32`](prism_render_architecture::particle::film_grain::hash_u32) and
//!   its unit-float wrapper
//!   [`grain_hash01`](prism_render_architecture::particle::film_grain::grain_hash01),
//! * the smoothstep-bilinear sized
//!   [`value_noise01`](prism_render_architecture::particle::film_grain::value_noise01)
//!   and its zero-centered
//!   [`signed_grain`](prism_render_architecture::particle::film_grain::signed_grain),
//! * the `Rec. 709`
//!   [`luma709`](prism_render_architecture::particle::film_grain::luma709) and
//!   the rational-polynomial
//!   [`FilmGrainParams::response`](prism_render_architecture::particle::film_grain::FilmGrainParams::response),
//! * both [`GrainBlend`](prism_render_architecture::particle::film_grain::GrainBlend)
//!   composites (additive and polynomial `Pegtop` soft light).
//!
//! [`GpuFilmGrain::apply`] evaluates the full composite per pixel;
//! [`GpuFilmGrain::hash`] exposes the raw avalanche hash so the parity test can
//! assert bit-exact integer agreement independently of the floating pipeline.
//!
//! # Correctness model
//!
//! `WGSL` `u32` arithmetic is defined to wrap on overflow, exactly matching the
//! `CPU` golden's `wrapping_mul` / `wrapping_add`, so the integer hash
//! reproduces the bit pattern exactly — [`GpuFilmGrain::hash`] is compared bit
//! for bit. The floating noise, response and blend evaluate the same closed
//! form in the same order; they are not bit-exact because a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the composited color, tight
//! enough to catch a genuinely wrong port (a swapped tap, a wrong constant, a
//! missing clamp) yet loose enough to admit legal fused multiply-add
//! contraction.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `+ - * /`, bitwise `^ & | << >>` and `f32` conversion — with no
//! transcendental call and no optional device feature, so they run unmodified
//! on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: hand-rolled integer-hash value-noise film grain plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::film_grain::FilmGrainParams;
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
/// the sibling twins use.
const WORKGROUP_SIZE: u32 = 64;

/// One input pixel for [`GpuFilmGrain::apply`]: the linear `RGB` color and the
/// integer pixel coordinate the grain field is sampled at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilmGrainPixel {
    /// The linear `RGB` color to composite grain onto.
    pub color: [f32; 3],
    /// The pixel `x` coordinate.
    pub x: u32,
    /// The pixel `y` coordinate.
    pub y: u32,
}

impl FilmGrainPixel {
    /// Builds a pixel from its color and coordinate.
    #[must_use]
    pub fn new(color: [f32; 3], x: u32, y: u32) -> FilmGrainPixel {
        FilmGrainPixel { color, x, y }
    }
}

/// One film-grain dispatch: the shared parameter block, the frame index and the
/// pixels to composite.
#[derive(Clone, Debug, PartialEq)]
pub struct FilmGrainQuery {
    /// The grain parameters (intensity, cell size, response tunables, blend).
    pub params: FilmGrainParams,
    /// The frame index; distinct frames yield a distinct grain field.
    pub frame: u32,
    /// The pixels to composite grain onto, returned in input order.
    pub pixels: Vec<FilmGrainPixel>,
}

/// Uniform parameters for one grain dispatch. `32`-byte `repr(C)` matching
/// `Params` in `shaders/film_grain.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrainParamsGpu {
    intensity: f32,
    shadow_boost: f32,
    highlight_rolloff: f32,
    cell_size: u32,
    blend_code: u32,
    frame: u32,
    count: u32,
    pad: u32,
}

/// One input pixel as uploaded. `32`-byte `repr(C)` matching `Pixel` in
/// `shaders/film_grain.wesl`: the coordinate plus two pad words, then the color
/// in the first three lanes of a `vec4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPixel {
    x: u32,
    y: u32,
    pad0: u32,
    pad1: u32,
    color: [f32; 4],
}

/// Uniform parameters for one raw-hash dispatch. `16`-byte `repr(C)` matching
/// `Params` in `shaders/film_grain_hash.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HashParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable film-grain pipeline pair: the per-pixel composite and
/// the raw avalanche hash.
pub struct GpuFilmGrain {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module_apply: ShaderModule,
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module_hash: ShaderModule,
    layout: BindGroupLayout,
    pipeline_apply: ComputePipeline,
    pipeline_hash: ComputePipeline,
}

impl GpuFilmGrain {
    /// Compiles the film-grain composite and raw-hash kernels on `ctx`.
    ///
    /// Both kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required. The two pipelines share one bind-group layout
    /// (uniform, read-only storage, read-write storage) because their binding
    /// shapes are identical.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFilmGrain {
        let device = ctx.device();
        let module_apply = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_film_grain_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/film_grain.wesl").into()),
        });
        let module_hash = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_film_grain_hash_shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/film_grain_hash.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_film_grain_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_film_grain_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline_apply = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_film_grain_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_apply,
            entry_point: Some("film_grain_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_hash = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_film_grain_hash_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module_hash,
            entry_point: Some("film_grain_hash_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFilmGrain {
            module_apply,
            module_hash,
            layout,
            pipeline_apply,
            pipeline_hash,
        }
    }

    /// Applies film grain to every pixel in `query`, returning the composited
    /// linear `RGB` in input order.
    ///
    /// The returned color for pixel `p` equals
    /// [`FilmGrainParams::apply`](prism_render_architecture::particle::film_grain::FilmGrainParams::apply)`(p.color, p.x, p.y, query.frame)`
    /// within the documented tolerance. An empty pixel slice yields an empty
    /// result — storage buffers cannot be zero-sized, so it is handled by an
    /// early return.
    #[must_use]
    pub fn apply(&self, ctx: &GpuContext, query: &FilmGrainQuery) -> Vec<[f32; 3]> {
        if query.pixels.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_pixels: Vec<GpuPixel> = query
            .pixels
            .iter()
            .map(|p| GpuPixel {
                x: p.x,
                y: p.y,
                pad0: 0,
                pad1: 0,
                color: [p.color[0], p.color[1], p.color[2], 0.0],
            })
            .collect();

        let gpu_params = GrainParamsGpu {
            intensity: query.params.intensity,
            shadow_boost: query.params.shadow_boost,
            highlight_rolloff: query.params.highlight_rolloff,
            cell_size: query.params.cell_size,
            blend_code: query.params.blend.code(),
            frame: query.frame,
            count: query.pixels.len() as u32,
            pad: 0,
        };

        let out_bytes = (query.pixels.len() as u64) * (size_of::<[f32; 4]>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_film_grain_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let pixels_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_film_grain_pixels"),
            contents: bytemuck::cast_slice(&gpu_pixels),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_film_grain_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_film_grain_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_film_grain_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: pixels_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_film_grain_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_film_grain_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_apply);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel, flattened to a 1-D dispatch.
            let groups = (query.pixels.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let flat = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(flat.len(), query.pixels.len());
        flat.into_iter().map(|c| [c[0], c[1], c[2]]).collect()
    }

    /// Evaluates the raw avalanche hash
    /// [`hash_u32`](prism_render_architecture::particle::film_grain::hash_u32)
    /// for every seed, returning one hash per seed in input order.
    ///
    /// `WGSL` `u32` arithmetic wraps on overflow, so each returned hash equals
    /// the `CPU` golden bit-exactly. An empty slice yields an empty result.
    #[must_use]
    pub fn hash(&self, ctx: &GpuContext, seeds: &[u32]) -> Vec<u32> {
        if seeds.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = HashParams {
            count: seeds.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (seeds.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_film_grain_hash_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let seeds_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_film_grain_hash_seeds"),
            contents: bytemuck::cast_slice(seeds),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_film_grain_hash_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_film_grain_hash_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_film_grain_hash_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: seeds_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_film_grain_hash_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_film_grain_hash_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_hash);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (seeds.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let raw = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), seeds.len());
        raw
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
