//! `wgpu` compute twin of the aerial-perspective blend
//! ([`blend_with_atmosphere`](prism_render_architecture::volumetric::atmosphere::blend_with_atmosphere)).
//!
//! The atmosphere hookup (design section 8) fades a cloud toward the airlight it
//! sits in front of and the airlight the view path accumulates. `cloud_color`
//! is the cloud's out-scattered radiance, `cloud_transmittance` (`0..=1`) is how
//! much distant sky shows *through* the cloud, `inscatter` is the sampled
//! airlight, and `weight` (`0..=1`) is the distance fade. The result is two
//! nested convex combinations:
//!
//! ```text
//! composited = lerp(cloud_color, inscatter, saturate(cloud_transmittance))
//! result     = lerp(composited,  inscatter, saturate(weight))
//! ```
//!
//! Because both steps are convex combinations, every output channel stays
//! within the interval spanned by `cloud_color` and `inscatter`, so the blend is
//! energy-conserving and never over-exposes beyond its brightest source. The
//! `CPU` golden
//! [`blend_with_atmosphere`](prism_render_architecture::volumetric::atmosphere::blend_with_atmosphere)
//! owns that math; [`GpuBlendWithAtmosphere`] is the on-device twin that runs
//! one thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The `lerp` is expanded to the *same* closed form the CPU `math` module uses
//! and `saturate` clamps both weights, so `CPU` and `GPU` evaluate identical
//! algebra (the only slack is a legal multiply-add contraction of a few `ULP`).
//! The parity test asserts a tight tolerance (`abs_diff < 1e-6` or
//! `rel_diff < 1e-5`). The scenes also assert every output channel stays within
//! the closed interval spanned by `cloud_color` and `inscatter` (the
//! energy-conservation bound), so a degenerate kernel could not pass.
//!
//! # Layout
//!
//! Each query packs the cloud colour with its transmittance in one `vec4` and
//! the in-scatter with the fade weight in a second, and results are `vec4`
//! (`xyz = rgb`, `w` unused), so every buffer stays `16`-byte aligned.
//!
//! # Portability
//!
//! The kernel is `clamp`/`saturate` and multiply/add in the portable core-`WGSL`
//! subset — no `exp`, `pow` or optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard aerial-perspective composite plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

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

/// One aerial-perspective blend query: a cloud colour, its transmittance, the
/// sampled in-scatter, and the distance-fade weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendQuery {
    /// The cloud's out-scattered radiance (linear `RGB`).
    pub cloud_color: [f32; 3],
    /// How much distant sky shows through the cloud, in `0..=1` (saturated).
    pub cloud_transmittance: f32,
    /// The airlight sampled from the shared atmosphere service (linear `RGB`).
    pub inscatter: [f32; 3],
    /// The distance fade, in `0..=1` (saturated).
    pub weight: f32,
}

/// One blend result: the composited linear `RGB` radiance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendedColor {
    /// Linear red channel.
    pub r: f32,
    /// Linear green channel.
    pub g: f32,
    /// Linear blue channel.
    pub b: f32,
}

/// One query as uploaded: cloud colour packed with transmittance, in-scatter
/// packed with weight. `32`-byte `repr(C)` matching `Query` in
/// `shaders/blend_with_atmosphere.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    cloud: [f32; 4],
    air: [f32; 4],
}

/// One result as read back. `16`-byte stride matching `results` in the shader
/// (the `RGB` triple plus one pad word).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuColor {
    r: f32,
    g: f32,
    b: f32,
    pad: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/blend_with_atmosphere.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable aerial-perspective blend pipeline.
pub struct GpuBlendWithAtmosphere {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBlendWithAtmosphere {
    /// Compiles the aerial-perspective blend kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBlendWithAtmosphere {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_blend_with_atmosphere"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/blend_with_atmosphere.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_blend_with_atmosphere_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_blend_with_atmosphere_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_blend_with_atmosphere_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("blend_with_atmosphere_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBlendWithAtmosphere {
            module,
            layout,
            pipeline,
        }
    }

    /// Composites every query in `queries`, returning one [`BlendedColor`] per
    /// query in input order.
    ///
    /// The returned colour for query `q` equals
    /// [`blend_with_atmosphere`](prism_render_architecture::volumetric::atmosphere::blend_with_atmosphere)`(...)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[BlendQuery]) -> Vec<BlendedColor> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                cloud: [
                    q.cloud_color[0],
                    q.cloud_color[1],
                    q.cloud_color[2],
                    q.cloud_transmittance,
                ],
                air: [q.inscatter[0], q.inscatter[1], q.inscatter[2], q.weight],
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuColor>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_blend_with_atmosphere_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_blend_with_atmosphere_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_blend_with_atmosphere_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_blend_with_atmosphere_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_blend_with_atmosphere_bind_group"),
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
            label: Some("prism_volumetric_blend_with_atmosphere_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_blend_with_atmosphere_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuColor>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());
        gpu_results
            .into_iter()
            .map(|c| BlendedColor {
                r: c.r,
                g: c.g,
                b: c.b,
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
