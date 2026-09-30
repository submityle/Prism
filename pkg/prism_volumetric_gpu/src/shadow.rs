//! `wgpu` compute twin of the cloud self-shadow Beer-Lambert accumulation
//! ([`accumulate_shadow`](prism_render_architecture::volumetric::shadow::accumulate_shadow)).
//!
//! Cloud self-shadowing (design section 12) marches each light ray through the
//! density field, summing `max(density, 0) * step` into an optical depth, then
//! maps it through Beer-Lambert extinction saturated into `[0, 1]`. The `CPU`
//! golden
//! [`accumulate_shadow`](prism_render_architecture::volumetric::shadow::accumulate_shadow)
//! owns that math; [`GpuShadow`] is the on-device twin that runs one thread per
//! shadow ray and reproduces the same surviving transmittance.
//!
//! # Correctness model
//!
//! The extinction is evaluated with the *same* hand-rolled `exp_approx` the
//! reference uses — base-two range reduction with a fractional seven-term
//! polynomial times an integer power assembled from the `f32` exponent field —
//! not the device-native `exp`. Mirroring the polynomial keeps the twin
//! bit-close to the reference, so the parity test asserts a tight tolerance
//! (`abs_diff < 1e-6` or `rel_diff < 1e-5`) and proves the whole extinction
//! algorithm ported, rather than admitting an unbounded native-`exp` deviation.
//! The only slack is a legal multiply-add contraction of a few `ULP` in the
//! optical-depth sum and the polynomial. The parity scenes additionally assert
//! the documented monotonicity (denser or longer rays never raise the result)
//! and the `[0, 1]` range, so a degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is `floor`, `bitcast`, integer ops and multiply/add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Beer-Lambert volumetric shadowing plus `wgpu` compute
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

/// One shadow ray: the density samples marched along the light direction and
/// the (uniform) march step between them.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowRay {
    /// Density samples along the light ray, in march order.
    pub densities: Vec<f32>,
    /// Uniform step length between consecutive samples.
    pub step: f32,
}

/// One shadow ray as uploaded. `16`-byte `repr(C)` matching `Ray` in
/// `shaders/shadow.wesl`: where the ray's density run starts, how many samples
/// it spans and its march step.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRay {
    offset: u32,
    sample_count: u32,
    step: f32,
    pad: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/shadow.wesl`: the ray count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    ray_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable cloud self-shadow pipeline.
pub struct GpuShadow {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuShadow {
    /// Compiles the cloud self-shadow kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuShadow {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_shadow"),
            source: ShaderSource::Wgsl(include_str!("../shaders/shadow.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_shadow_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_shadow_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_shadow_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("accumulate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuShadow {
            module,
            layout,
            pipeline,
        }
    }

    /// Accumulates optical depth along each ray and returns its surviving
    /// Beer-Lambert transmittance, one value per ray in input order.
    ///
    /// The returned value for ray `r` equals
    /// [`accumulate_shadow`](prism_render_architecture::volumetric::shadow::accumulate_shadow)`(&r.densities, r.step)`
    /// to within the tolerance documented on this module. An empty `rays` slice
    /// yields an empty result — storage buffers cannot be zero-sized, so it is
    /// handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, rays: &[ShadowRay]) -> Vec<f32> {
        if rays.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        // Flatten every ray's samples into one contiguous run and record each
        // ray's offset/length so a single thread walks its own segment.
        let mut densities: Vec<f32> = Vec::new();
        let mut gpu_rays: Vec<GpuRay> = Vec::with_capacity(rays.len());
        for r in rays {
            let offset = densities.len() as u32;
            densities.extend_from_slice(&r.densities);
            gpu_rays.push(GpuRay {
                offset,
                sample_count: r.densities.len() as u32,
                step: r.step,
                pad: 0,
            });
        }
        // Storage buffers cannot be zero-sized; when no ray carries any sample
        // the run is empty. A single unread padding element keeps the binding
        // valid without changing any result (every `sample_count` is zero).
        if densities.is_empty() {
            densities.push(0.0);
        }

        let gpu_params = Params {
            ray_count: rays.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (rays.len() as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_shadow_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let rays_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_shadow_rays"),
            contents: bytemuck::cast_slice(&gpu_rays),
            usage: BufferUsages::STORAGE,
        });
        let densities_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_shadow_densities"),
            contents: bytemuck::cast_slice(&densities),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_shadow_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_shadow_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_shadow_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: rays_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: densities_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_shadow_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_shadow_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (rays.len() as u32).div_ceil(64);
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
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(values.len(), rays.len());
        values
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
