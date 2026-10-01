//! `wgpu` compute twin of Prism's second-order spherical-harmonic transmittance
//! projection
//! ([`project_sh`](prism_render_architecture::hair::dual_scatter_sh::project_sh),
//! which composes
//! [`sh_basis`](prism_render_architecture::hair::dual_scatter_sh::sh_basis) with
//! a Monte-Carlo coefficient accumulation).
//!
//! A `Zinke` dual-scattering groom caches its low-frequency directional
//! transmittance in nine real second-order `SH` coefficients. [`project_sh`] is
//! the *fit* side of that cache: given a batch of directional transmittance
//! samples it estimates `c_lm = ∫ T(ω) Y_lm(ω) dω` by the Monte-Carlo sum
//! `Σ T·Y_lm·w` with the uniform solid-angle weight `w = 4π / N`. It is the
//! inverse of [`eval_sh`](prism_render_architecture::hair::dual_scatter_sh::eval_sh):
//! `eval_sh` reconstructs a transmittance from the coefficients, this projection
//! fits the coefficients from the samples (different input, different output,
//! different operator), so the two twins are genuinely distinct.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairProjectSh::project`] takes a batch of
//! [`TransmittanceSample`](prism_render_architecture::hair::dual_scatter_sh::TransmittanceSample)s
//! and returns the nine band-major
//! [`ShCoeffs`](prism_render_architecture::hair::dual_scatter_sh::ShCoeffs) the
//! golden produces — a reduction over the whole batch rather than an
//! element-wise map.
//!
//! # One thread per coefficient, samples walked in order
//!
//! The dispatch is a single workgroup of nine live invocations
//! (`@compute @workgroup_size(64)`, `global_invocation_id.x >= 9` early-returns):
//! thread `k` walks every sample in authored order and accumulates
//! `c[k] += (sanitize(value)·weight)·basis_k(normalize_or_zero(dir))`. Because
//! each coefficient is summed over the samples in the exact order the golden
//! uses, the per-coefficient accumulation order matches bit-for-bit; the only
//! `CPU`/`GPU` divergence is fma fusion of the per-sample multiply-add.
//!
//! # The weight lives on the host
//!
//! The solid-angle weight `w = 4π / N` is computed on the host as an `f32` and
//! uploaded in the uniform, so the shader never spells out a `π` literal whose
//! bit pattern could drift from
//! [`core::f32::consts::PI`]. Values are sanitised before the weight multiply
//! (matching the golden's `sanitize(value) * weight`) and each coefficient is
//! sanitised once more after the accumulation.
//!
//! # Portability
//!
//! The basis uses only multiply/add plus one `sqrt` (vector normalisation), with
//! no `exp`, `pow`, `sin` or optional device feature, so the twin runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each coefficient is a long multiply-add chain a `GPU` may fuse, so `CPU` and
//! `GPU` agree to within the documented fma tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) rather than bit-for-bit. The sanitiser mirrors the golden
//! exactly (`is_finite` test rejecting `NaN`/±inf, the degenerate-length guard
//! collapsing a direction to zero, the final per-coefficient `sanitize`), so
//! non-finite sample values or directions produce the same bounded, finite
//! result.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard orthonormal real spherical-harmonic projection plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use core::f32::consts::PI;
use prism_render_architecture::hair::dual_scatter_sh::{
    project_sh, ShCoeffs, TransmittanceSample, SH_COEFFS,
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

/// Uniform parameters for one projection dispatch. Layout matches `Params` in
/// `shaders/dual_scatter_project.wesl`: the sample count and the host-computed
/// solid-angle weight in a single `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    sample_count: u32,
    weight: f32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable per-coefficient `SH` projection pipeline.
pub struct GpuHairProjectSh {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairProjectSh {
    /// Compiles the per-coefficient `SH` projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (multiply/add plus
    /// one `sqrt`), so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairProjectSh {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_project_sh"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dual_scatter_project.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_project_sh_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_project_sh_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_project_sh_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairProjectSh {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects a batch of directional transmittance `samples` onto the nine
    /// band-major second-order real `SH` coefficients, returning the same
    /// [`ShCoeffs`] as the `CPU` golden
    /// [`project_sh`](prism_render_architecture::hair::dual_scatter_sh::project_sh)
    /// to within the module's documented fma tolerance.
    ///
    /// An empty batch returns the all-zero
    /// [`ShCoeffs::default`](prism_render_architecture::hair::dual_scatter_sh::ShCoeffs)
    /// without a dispatch — storage buffers cannot be zero-sized and the golden
    /// short-circuits empty input the same way.
    #[must_use]
    pub fn project(&self, ctx: &GpuContext, samples: &[TransmittanceSample]) -> ShCoeffs {
        let sample_count = samples.len();
        if sample_count == 0 {
            return ShCoeffs::default();
        }

        let device = ctx.device();

        // The uniform solid-angle weight is computed on the host as an f32 so
        // the shader never spells out a pi literal (bit-matching the golden's
        // (4.0 * PI) / N).
        let weight = (4.0 * PI) / sample_count as f32;
        let uniforms = Params {
            sample_count: sample_count as u32,
            weight,
            pad0: 0,
            pad1: 0,
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

        // Output is the nine band-major coefficients.
        let out_bytes = (SH_COEFFS as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_project_sh_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let samples_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_project_sh_samples"),
            contents: bytemuck::cast_slice(&sample_values),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_project_sh_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_project_sh_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_project_sh_bind_group"),
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
            label: Some("prism_hair_project_sh_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_project_sh_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Nine coefficients fit in a single 64-wide workgroup.
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

        let mut c = [0.0_f32; SH_COEFFS];
        c.copy_from_slice(&values[..SH_COEFFS]);
        ShCoeffs { c }
    }
}

/// The `CPU` golden `SH` projection, re-exported so the parity test can assert
/// the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_project_sh(samples: &[TransmittanceSample]) -> ShCoeffs {
    project_sh(samples)
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
