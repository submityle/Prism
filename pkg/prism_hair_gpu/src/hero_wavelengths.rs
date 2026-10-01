//! `wgpu` compute twin of Prism's `Hero-wavelength` stratified spectral sampler
//! ([`hero_wavelengths`](prism_render_architecture::hair::spectral_absorption::hero_wavelengths)
//! plus
//! [`hero_sigma_a`](prism_render_architecture::hair::spectral_absorption::hero_sigma_a)).
//!
//! A spectral hair `BSDF` following `Wilkie` 2014 ("Hero Wavelength Spectral
//! Sampling") carries four stratified wavelengths per stochastic sample: a
//! primary (hero) wavelength plus three equally spaced cyclic companions through
//! the visible band. Sampling four correlated wavelengths at once decorrelates
//! chromatic noise and renders dispersion faithfully without one pass per colour
//! channel. This twin builds those four wavelengths from a stratified unit
//! coordinate and a per-sample rotation, then folds each with the fibre's two
//! pigment concentrations into `sigma_a` using the same 15-point visible-band
//! `LUT` interpolation (380-730nm) as the sibling [`spectrum_sample_map`]
//! twin.
//!
//! [`spectrum_sample_map`]: crate::spectrum_sample_map
//!
//! # What the kernel evaluates
//!
//! [`GpuHairHeroWavelengths::eval`] takes a batch of
//! `(u, rotate, eumelanin, pheomelanin)` samples and returns one
//! [`HeroSample`] per input in order — the four wavelengths and their four
//! `sigma_a`. The sample index is the invocation id (`@compute
//! @workgroup_size(64)`, one-dimensional dispatch over `global_invocation_id.x`);
//! invocations past the sample count early-return.
//!
//! # Spectra live on the host, not in the shader
//!
//! The two canonical 15-point per-unit pigment spectra
//! ([`EUMELANIN_SPECTRUM`](prism_render_architecture::hair::spectral_absorption::EUMELANIN_SPECTRUM),
//! [`PHEOMELANIN_SPECTRUM`](prism_render_architecture::hair::spectral_absorption::PHEOMELANIN_SPECTRUM))
//! are read from the architecture crate and uploaded in one storage buffer
//! (`[0..15]` eumelanin, `[15..30]` pheomelanin), so the shader never duplicates
//! the magic table constants and cannot drift from the golden. Only the
//! structural band layout (380nm .. 730nm, 15 samples, quarter-band stride)
//! lives in the shader, exactly like the fixed 64-wide dispatch shape.
//!
//! # Distinct from the `spectrum_sample_map` twin
//!
//! The `spectrum_sample_map` twin maps a caller-supplied list of arbitrary
//! wavelengths to `sigma_a` for one fibre. This twin *derives* its four
//! wavelengths from a stratified `(u, rotate)` coordinate by cyclic
//! quarter-band rotation (the `Wilkie` 2014 hero layout) before the same `LUT`
//! fold — so the input contract (stratified sample vs. explicit wavelengths) and
//! the wavelength-generation step are genuinely different ports.
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
//! Each `sigma_a` is a two-term interpolation multiply-add a `GPU` may fuse, so
//! `CPU` and `GPU` agree to within the documented fma tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than bit-for-bit; the
//! wavelengths themselves are a wrap plus a single multiply-add and compare the
//! same way. The wrap, concentration and wavelength guards mirror the golden's
//! `wrap_unit`, `clamp_concentration` and `sanitized_wavelength` exactly, so
//! non-finite / out-of-range stratified coordinates fold into `[0, 1)`,
//! negative / non-finite concentrations collapse to `0`, and every emitted
//! wavelength stays in `[380, 730)`nm with a finite, non-negative `sigma_a`.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `Chiang` 2016 / `d'Eon` 2011 pigment model plus `Wilkie` 2014
//! hero-wavelength spectral sampling and `wgpu` compute dispatch; no Unreal
//! Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::spectral_absorption::{
    hero_sigma_a, hero_wavelengths, EUMELANIN_SPECTRUM, PHEOMELANIN_SPECTRUM, SPECTRUM_SAMPLES,
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

/// One evaluated `Hero-wavelength` sample: the four stratified wavelengths (in
/// nanometres, hero at index `0`) and their four spectral absorption
/// coefficients `sigma_a`, matching the golden
/// [`HeroWavelengths`](prism_render_architecture::hair::spectral_absorption::HeroWavelengths)
/// / [`hero_sigma_a`](prism_render_architecture::hair::spectral_absorption::hero_sigma_a).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeroSample {
    /// The four sampled wavelengths in nanometres, each within `[380, 730)`.
    pub lambdas: [f32; 4],
    /// The spectral absorption coefficient `sigma_a` at each wavelength.
    pub sigma_a: [f32; 4],
}

/// Uniform parameters for one hero-wavelength dispatch. Layout matches `Params`
/// in `shaders/hero_wavelengths.wesl`: the sample count in one `16`-byte uniform
/// slot (the three pads keep the uniform `16`-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    sample_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable `Hero-wavelength` spectral-sampling pipeline.
pub struct GpuHairHeroWavelengths {
    #[expect(
        dead_code,
        reason = "retained so the compiled module outlives the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairHeroWavelengths {
    /// Compiles the `Hero-wavelength` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairHeroWavelengths {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_hero_wavelengths"),
            source: ShaderSource::Wgsl(include_str!("../shaders/hero_wavelengths.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_hero_wavelengths_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_hero_wavelengths_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_hero_wavelengths_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairHeroWavelengths {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates one [`HeroSample`] per `(u, rotate, eumelanin, pheomelanin)`
    /// input, in order.
    ///
    /// Sample `i` equals the `CPU` golden pairing of
    /// [`hero_wavelengths`](prism_render_architecture::hair::spectral_absorption::hero_wavelengths)
    /// with
    /// [`hero_sigma_a`](prism_render_architecture::hair::spectral_absorption::hero_sigma_a)
    /// to within the module's documented fma tolerance, with non-finite /
    /// out-of-range coordinates folding into `[0, 1)`, negative / non-finite
    /// concentrations collapsing to `0`, and every wavelength in `[380, 730)`nm.
    /// An empty batch yields an empty vector without a dispatch — storage
    /// buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, samples: &[(f32, f32, f32, f32)]) -> Vec<HeroSample> {
        let sample_count = samples.len();
        if sample_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // The two per-pigment spectra, concatenated so the shader indexes
        // eumelanin at [0..15] and pheomelanin at [15..30]; read straight from
        // the architecture crate so the magic table constants never drift.
        let mut spectra: Vec<f32> = Vec::with_capacity(2 * SPECTRUM_SAMPLES);
        spectra.extend_from_slice(&EUMELANIN_SPECTRUM);
        spectra.extend_from_slice(&PHEOMELANIN_SPECTRUM);

        // Four f32 per sample: u, rotate, eumelanin, pheomelanin. The shader
        // sanitizes coordinates and concentrations itself (bit-faithfully to the
        // golden), so upload the raw authored values.
        let mut inputs: Vec<f32> = Vec::with_capacity(sample_count * 4);
        for &(u, rotate, eumelanin, pheomelanin) in samples {
            inputs.push(u);
            inputs.push(rotate);
            inputs.push(eumelanin);
            inputs.push(pheomelanin);
        }

        let uniforms = Params {
            sample_count: sample_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Eight f32 per sample: four wavelengths then four sigma_a.
        let out_bytes = (sample_count as u64) * 8 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_hero_wavelengths_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let spectra_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_hero_wavelengths_spectra"),
            contents: bytemuck::cast_slice(&spectra),
            usage: BufferUsages::STORAGE,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_hero_wavelengths_inputs"),
            contents: bytemuck::cast_slice(&inputs),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_hero_wavelengths_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_hero_wavelengths_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_hero_wavelengths_bind_group"),
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
                    resource: inputs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_hero_wavelengths_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_hero_wavelengths_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (sample_count as u32).div_ceil(64);
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        let mut out = Vec::with_capacity(sample_count);
        for chunk in flat.chunks_exact(8) {
            out.push(HeroSample {
                lambdas: [chunk[0], chunk[1], chunk[2], chunk[3]],
                sigma_a: [chunk[4], chunk[5], chunk[6], chunk[7]],
            });
        }
        out
    }
}

/// The `CPU` golden hero-wavelength sample, re-exported so the parity test can
/// assert the device twin against the identical reference it mirrors: the four
/// wavelengths from
/// [`hero_wavelengths`](prism_render_architecture::hair::spectral_absorption::hero_wavelengths)
/// paired with their
/// [`hero_sigma_a`](prism_render_architecture::hair::spectral_absorption::hero_sigma_a).
#[must_use]
pub fn reference_hero_sample(u: f32, rotate: f32, eumelanin: f32, pheomelanin: f32) -> HeroSample {
    let hero = hero_wavelengths(u, rotate);
    let sigma = hero_sigma_a(eumelanin, pheomelanin, hero);
    HeroSample {
        lambdas: hero.lambdas,
        sigma_a: sigma,
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
