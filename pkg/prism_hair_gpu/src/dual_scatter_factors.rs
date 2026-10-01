//! `wgpu` compute twin of Prism's `Zinke` dual-scattering forward/back factor
//! estimator
//! ([`dual_scatter_factors`](prism_render_architecture::hair::dual_scatter_sh::dual_scatter_factors)).
//!
//! A `Zinke` 2008 dual-scattering groom folds its global multiple-forward and
//! local back scattering into two averaged attenuation factors: `a_f`
//! (`forward`), the mean transmittance over the forward (`+z`) hemisphere that
//! drives the global multiplier `a_f^n`, and `a_b` (`backward`), the mean over
//! the backward (`-z`) hemisphere that drives the local back-scatter term. Both
//! land in `[0, 1]`.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairDualScatterFactors::factors`] takes a batch of
//! [`TransmittanceSample`](prism_render_architecture::hair::dual_scatter_sh::TransmittanceSample)s
//! and returns the
//! [`ScatterFactors`](prism_render_architecture::hair::dual_scatter_sh::ScatterFactors)
//! the golden produces — a hemisphere-split averaging reduction over the whole
//! batch, genuinely distinct from the `SH` projection twin
//! ([`GpuHairProjectSh`](crate::dual_scatter_project::GpuHairProjectSh)) and the
//! integer-power twin
//! ([`GpuHairForwardScatterPower`](crate::forward_scatter_power::GpuHairForwardScatterPower)).
//!
//! # Two threads, one hemisphere each
//!
//! The dispatch is a single workgroup of two live invocations
//! (`@compute @workgroup_size(64)`, `global_invocation_id.x >= 2` early-returns):
//! lane `0` accumulates the `+z` hemisphere into `forward`, lane `1` the `-z`
//! hemisphere into `backward`. Each lane walks every sample in authored order,
//! normalises the direction (`normalize_or_zero`), sanitises and clamps the
//! value to `[0, 1]`, sums its hemisphere and averages (an empty hemisphere
//! yields `0`). Because each factor is summed over its hemisphere in the exact
//! order the golden uses, the only `CPU`/`GPU` divergence is the final divide.
//!
//! # Portability
//!
//! The kernel uses only multiply/add, one `sqrt` (vector normalisation) and a
//! divide, with no `exp`, `pow`, `sin` or optional device feature, so the twin
//! runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each factor is a sequential hemisphere sum followed by one divide and a
//! clamp, so `CPU` and `GPU` agree to within the documented fma tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than bit-for-bit. The
//! sanitiser mirrors the golden exactly (`is_finite` test rejecting `NaN`/±inf,
//! the degenerate-length guard collapsing a direction to zero, the `[0, 1]`
//! clamp on both the per-sample value and the final average), so non-finite
//! sample values or directions produce the same bounded, finite result in
//! `[0, 1]`.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `Zinke` 2008 dual-scattering averaged factors plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::dual_scatter_sh::{
    dual_scatter_factors, ScatterFactors, TransmittanceSample,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of scalar factors the kernel writes back (`forward`, `backward`).
const FACTOR_COUNT: usize = 2;

/// Uniform parameters for one factor dispatch. Layout matches `Params` in
/// `shaders/dual_scatter_factors.wesl`: the sample count in a single `16`-byte
/// uniform slot (one `u32` plus padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    sample_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable dual-scattering factor pipeline.
pub struct GpuHairDualScatterFactors {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairDualScatterFactors {
    /// Compiles the dual-scattering factor kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (multiply/add plus
    /// one `sqrt` and a divide), so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairDualScatterFactors {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_dual_scatter_factors"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dual_scatter_factors.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_dual_scatter_factors_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_dual_scatter_factors_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_dual_scatter_factors_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairDualScatterFactors {
            module,
            layout,
            pipeline,
        }
    }

    /// Estimates the `Zinke` forward/back dual-scattering factors from a batch of
    /// directional transmittance `samples`, returning the same
    /// [`ScatterFactors`] as the `CPU` golden
    /// [`dual_scatter_factors`](prism_render_architecture::hair::dual_scatter_sh::dual_scatter_factors)
    /// to within the module's documented fma tolerance.
    ///
    /// An empty batch returns the all-zero
    /// [`ScatterFactors::default`](prism_render_architecture::hair::dual_scatter_sh::ScatterFactors)
    /// without a dispatch — storage buffers cannot be zero-sized and the golden
    /// yields the same zero factors for empty input.
    #[must_use]
    pub fn factors(&self, ctx: &GpuContext, samples: &[TransmittanceSample]) -> ScatterFactors {
        let sample_count = samples.len();
        if sample_count == 0 {
            return ScatterFactors::default();
        }

        let device = ctx.device();

        let uniforms = Params {
            sample_count: sample_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Samples flattened to four floats each (dir.x, dir.y, dir.z, value) to
        // dodge the std430 vec3 stride.
        let mut sample_values: Vec<f32> = Vec::with_capacity(sample_count * 4);
        for sample in samples {
            sample_values.push(sample.dir.x);
            sample_values.push(sample.dir.y);
            sample_values.push(sample.dir.z);
            sample_values.push(sample.value);
        }

        // Output is the two scalar factors.
        let out_bytes = (FACTOR_COUNT as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_dual_scatter_factors_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let samples_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_dual_scatter_factors_samples"),
            contents: bytemuck::cast_slice(&sample_values),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_dual_scatter_factors_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_dual_scatter_factors_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_dual_scatter_factors_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_dual_scatter_factors_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_dual_scatter_factors_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // The two factor lanes fit in a single 64-wide workgroup.
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let values = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        ScatterFactors {
            forward: values[0],
            backward: values[1],
        }
    }
}

/// The `CPU` golden dual-scattering factor estimator, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_dual_scatter_factors(samples: &[TransmittanceSample]) -> ScatterFactors {
    dual_scatter_factors(samples)
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
