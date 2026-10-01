//! `wgpu` compute twin of Prism's per-wavelength spectral melanin-absorption
//! sampler
//! ([`spectrum_sample_map`](prism_render_architecture::hair::spectral_absorption::spectrum_sample_map)).
//!
//! The sibling [`melanin`](crate::melanin) twin folds the two pigment
//! concentrations into three RGB absorption coefficients — enough for an RGB
//! pipeline. A spectral renderer instead needs the full `lambda -> sigma_a`
//! curve so that dispersion, fluorescent dye and narrow-band studio lights read
//! correctly, and so pigmented hair avoids the characteristic RGB "hue-shift
//! under saturated light" error. This twin samples that curve: for a single
//! fibre's two non-negative pigment concentrations it maps a whole batch of
//! wavelengths to their `sigma_a`, each wavelength independently, by linearly
//! interpolating the two uniformly sampled visible-band spectra (380-730nm) and
//! folding `sigma_a = eu * eu_sample + pheo * pheo_sample`.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairSpectrumSample::eval`] takes one `(eumelanin, pheomelanin)` pair and
//! a slice of wavelengths, returning one `sigma_a` per wavelength in input
//! order — the array-in/array-out form used to resample a whole spectral
//! tabulation for one fibre. The wavelength index is the invocation id
//! (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the wavelength count early-return.
//!
//! # Spectra live on the host, not in the shader
//!
//! The two canonical 15-point per-unit pigment spectra
//! ([`EUMELANIN_SPECTRUM`](prism_render_architecture::hair::spectral_absorption::EUMELANIN_SPECTRUM),
//! [`PHEOMELANIN_SPECTRUM`](prism_render_architecture::hair::spectral_absorption::PHEOMELANIN_SPECTRUM))
//! are read from the architecture crate and uploaded in one storage buffer
//! (`[0..15]` eumelanin, `[15..30]` pheomelanin), so the shader never duplicates
//! the magic table constants and cannot drift from the golden. Only the
//! structural band layout (380nm .. 730nm, 15 samples) lives in the shader,
//! exactly like the fixed 64-wide dispatch shape.
//!
//! # Distinct from the RGB `melanin` twin
//!
//! The `melanin` twin outputs a three-channel RGB `sigma_a` from fixed-primary
//! constants passed as a uniform; this twin outputs a per-wavelength scalar
//! `sigma_a` by `floor`-indexed linear interpolation into a 15-point `LUT`
//! sampled across the full visible band. RGB-primary fold versus
//! spectral-`LUT` interpolation make the two genuinely distinct ports.
//!
//! # Portability
//!
//! The kernel uses only `floor`, `clamp`, dynamic array indexing and
//! multiply/add in the portable core-`WGSL` subset — no `exp`, `pow`, `sin` or
//! optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! The interpolation is a two-term multiply-add a `GPU` may fuse, so `CPU` and
//! `GPU` agree to within the documented fma tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) rather than bit-for-bit. The concentration guard mirrors
//! the golden's `clamp_concentration` exactly (`v == v && v > 0.0 &&
//! v <= MAX_FINITE_F32` rejects `NaN`, non-positive and `+inf`) and the
//! wavelength guard mirrors `sanitized_wavelength` (non-finite -> short
//! endpoint, otherwise clamped into the sampled band), so negative, non-finite
//! and out-of-range inputs collapse to the same values the golden emits and
//! every `sigma_a` stays finite and non-negative.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `Chiang` 2016 / `d'Eon` 2011 pigment model plus `Wilkie` 2014
//! spectral sampling and `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::spectral_absorption::{
    spectrum_sample_map, EUMELANIN_SPECTRUM, PHEOMELANIN_SPECTRUM, SPECTRUM_SAMPLES,
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

/// Uniform parameters for one spectral-sample dispatch. Layout matches `Params`
/// in `shaders/spectrum_sample_map.wesl`: the fibre's two pigment concentrations
/// and the wavelength count, in one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    eumelanin: f32,
    pheomelanin: f32,
    wavelength_count: u32,
    pad0: u32,
}

/// A compiled, reusable per-wavelength spectral-absorption pipeline.
pub struct GpuHairSpectrumSample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairSpectrumSample {
    /// Compiles the per-wavelength spectral-absorption kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairSpectrumSample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_spectrum_sample"),
            source: ShaderSource::Wgsl(include_str!("../shaders/spectrum_sample_map.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_spectrum_sample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_spectrum_sample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_spectrum_sample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairSpectrumSample {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps each queried wavelength to its spectral absorption coefficient
    /// `sigma_a` for the given pigment concentrations, returning one scalar per
    /// wavelength in input order.
    ///
    /// Element `i` equals the `CPU` golden
    /// [`spectrum_sample_map`](prism_render_architecture::hair::spectral_absorption::spectrum_sample_map)
    /// of `wavelengths[i]` to within the module's documented fma tolerance, with
    /// negative / non-finite concentrations collapsing to `0` and out-of-range /
    /// non-finite wavelengths clamping to the sampled endpoints. An empty batch
    /// yields an empty vector without a dispatch — storage buffers cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        eumelanin: f32,
        pheomelanin: f32,
        wavelengths: &[f32],
    ) -> Vec<f32> {
        let wavelength_count = wavelengths.len();
        if wavelength_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // The two per-pigment spectra, concatenated so the shader indexes
        // eumelanin at [0..15] and pheomelanin at [15..30]; read straight from
        // the architecture crate so the magic table constants never drift.
        let mut spectra: Vec<f32> = Vec::with_capacity(2 * SPECTRUM_SAMPLES);
        spectra.extend_from_slice(&EUMELANIN_SPECTRUM);
        spectra.extend_from_slice(&PHEOMELANIN_SPECTRUM);

        // The shader sanitizes concentrations and wavelengths itself
        // (bit-faithfully to the golden), so upload the raw authored values.
        let uniforms = Params {
            eumelanin,
            pheomelanin,
            wavelength_count: wavelength_count as u32,
            pad0: 0,
        };

        let out_bytes = (wavelength_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_spectrum_sample_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let spectra_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_spectrum_sample_spectra"),
            contents: bytemuck::cast_slice(&spectra),
            usage: BufferUsages::STORAGE,
        });
        let wavelengths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_spectrum_sample_wavelengths"),
            contents: bytemuck::cast_slice(wavelengths),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_spectrum_sample_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_spectrum_sample_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_spectrum_sample_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: spectra_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wavelengths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_spectrum_sample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_spectrum_sample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (wavelength_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden spectral-sample map, re-exported so the parity test can
/// assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_spectrum_sample_map(
    eumelanin: f32,
    pheomelanin: f32,
    wavelengths: &[f32],
) -> Vec<f32> {
    spectrum_sample_map(eumelanin, pheomelanin, wavelengths)
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
