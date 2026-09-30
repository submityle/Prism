//! `wgpu` compute twin of the sunset in-scatter tint
//! ([`sunset_inscatter_tint`](prism_render_architecture::volumetric::atmosphere::sunset_inscatter_tint)).
//!
//! The atmosphere hookup (design section 8) warms the sampled airlight as the
//! sun sinks toward the horizon. The reddening amount
//! [`sunset_reddening`](prism_render_architecture::volumetric::spectral::sunset_reddening)
//! drives a linear interpolation from the neutral tint `(1, 1, 1)` (sun high,
//! no twilight) toward a warm twilight tint: a fixed short-to-long twilight
//! spectral distribution is normalised to a partition of unity, collapsed to
//! linear `RGB` via the same `spectral_to_rgb` math, then rescaled so its
//! brightest channel is exactly one.
//!
//! Because every channel of the warm tint is in `0..=1` and the neutral tint is
//! one, the interpolated tint is in `0..=1` on every channel, so multiplying the
//! sampled airlight by it can only *attenuate* (never amplify) a channel and the
//! downstream composite stays energy-conserving. As the sun drops the blue and
//! green channels attenuate faster than red, warming the airlight. The `CPU`
//! golden
//! [`sunset_inscatter_tint`](prism_render_architecture::volumetric::atmosphere::sunset_inscatter_tint)
//! owns that math; [`GpuSunsetInscatterTint`] is the on-device twin that runs
//! one thread per query and reproduces the same value.
//!
//! # Correctness model
//!
//! The reddening `smoothstep` and per-band Gaussian response use the *same*
//! closed forms and the *same* hand-rolled `exp_approx` the reference uses — not
//! the device-native `exp` — and the `lerp` is expanded to the *same* closed
//! form. Mirroring those keeps the twin bit-close to the reference, so the
//! parity test asserts a tight tolerance (`abs_diff < 1e-6` or
//! `rel_diff < 1e-5`). The scenes also assert every channel stays in `0..=1`,
//! the tint is neutral one above the reddening cutoff, and it warms
//! monotonically (red never falls below green never falls below blue as the sun
//! drops), so a degenerate kernel could not pass.
//!
//! # Layout
//!
//! Results are `vec4` (`xyz = rgb` tint, `w` unused) so the storage buffer stays
//! `16`-byte aligned.
//!
//! # Portability
//!
//! The kernel is `floor`, `bitcast`, integer ops, `clamp`/`saturate` and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow` or optional
//! device feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard horizon-reddening falloff plus CIE-flavoured
//! spectral-to-RGB collapse plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

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

/// One sunset in-scatter tint query: the sun's angular altitude in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SunsetInscatterTintQuery {
    /// Sun's angular altitude in radians; zero at the horizon, positive above.
    pub sun_altitude: f32,
}

/// One in-scatter tint result: the linear `RGB` multiplier applied to sampled
/// airlight. Every channel is in `0..=1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InscatterTint {
    /// Linear red multiplier in `0..=1`.
    pub r: f32,
    /// Linear green multiplier in `0..=1`.
    pub g: f32,
    /// Linear blue multiplier in `0..=1`.
    pub b: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/sunset_inscatter_tint.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    sun_altitude: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// One result as read back. `16`-byte stride matching `results` in the shader
/// (the `RGB` tint plus one pad word).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTint {
    r: f32,
    g: f32,
    b: f32,
    pad: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/sunset_inscatter_tint.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable sunset in-scatter tint pipeline.
pub struct GpuSunsetInscatterTint {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSunsetInscatterTint {
    /// Compiles the sunset in-scatter tint kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSunsetInscatterTint {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sunset_inscatter_tint"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/sunset_inscatter_tint.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sunset_inscatter_tint_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sunset_inscatter_tint_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sunset_inscatter_tint_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sunset_inscatter_tint_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSunsetInscatterTint {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the in-scatter tint for every query in `queries`, returning one
    /// [`InscatterTint`] per query in input order.
    ///
    /// The returned triple for query `q` equals
    /// [`sunset_inscatter_tint`](prism_render_architecture::volumetric::atmosphere::sunset_inscatter_tint)`(q.sun_altitude)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[SunsetInscatterTintQuery],
    ) -> Vec<InscatterTint> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                sun_altitude: q.sun_altitude,
                pad0: 0.0,
                pad1: 0.0,
                pad2: 0.0,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuTint>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sunset_inscatter_tint_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sunset_inscatter_tint_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sunset_inscatter_tint_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sunset_inscatter_tint_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sunset_inscatter_tint_bind_group"),
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
            label: Some("prism_volumetric_sunset_inscatter_tint_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sunset_inscatter_tint_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuTint>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());
        gpu_results
            .into_iter()
            .map(|t| InscatterTint {
                r: t.r,
                g: t.g,
                b: t.b,
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
