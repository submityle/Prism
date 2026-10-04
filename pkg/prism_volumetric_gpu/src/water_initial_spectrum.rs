//! `wgpu` compute twin of the dependency-free `CPU` golden ocean
//! initial-spectrum build
//! ([`build_initial_spectrum`](prism_render_architecture::water::initial_spectrum::build_initial_spectrum),
//! in [`initial_spectrum`](prism_render_architecture::water::initial_spectrum)).
//!
//! The [`water_surface_synth`](crate::water_surface_synth) twin already runs the
//! `advance -> ifft2 -> surface` tail of the spectral-ocean chain on device
//! from a host-built `h0(k)` / `h0(-k)` field. This twin builds that time-zero
//! field itself, so the whole `initial_spectrum -> advance -> ifft2 -> surface`
//! path is GPU self-consistent and real-device verifiable end to end — the
//! `Tessendorf` construction shipping oceans (`WaveWorks`, `Crest`, UE5 Water)
//! run once per sea state.
//!
//! # What is twinned
//!
//! For an `n x n` row-major grid the kernel reproduces the golden conventions
//! exactly: for each cell it draws a deterministic complex Gaussian
//! (`splitmix32` hash of the cell index and seed, then the Irwin-Hall
//! twelve-uniform central-limit draw), scales it by `sqrt(E(k) / 2)` for the
//! `+k` amplitude, and scales the *mirror* cell's own Gaussian by
//! `sqrt(E(-k) / 2)` for the `-k` amplitude — the Hermitian pairing
//! [`advance_amplitude`](prism_render_architecture::water::spectrum::advance_amplitude)
//! relies on. `E(k)` is the `Phillips`, `JONSWAP`, or Pierson-Moskowitz energy
//! density, evaluated with the same hand-rolled
//! [`exp_approx`](prism_render_architecture::water::exp_approx) the `CPU` golden
//! uses.
//!
//! Only the full-band field (`band == None`, i.e.
//! [`build_initial_spectrum`](prism_render_architecture::water::initial_spectrum::build_initial_spectrum))
//! is twinned; the cascade band split stays on the `CPU` for a later kernel.
//! Unlike the butterfly twins, the resolution is *not* restricted to a power of
//! two — the spectrum build is defined for any positive `N`.
//!
//! # Correctness model
//!
//! The build is embarrassingly parallel: one invocation owns one cell, with no
//! barrier and no workgroup memory. The mirror cell's Gaussian is recomputed
//! independently per thread (the `CPU` golden pre-draws every cell's Gaussian
//! only as an optimisation; `cell_gaussian` is a pure function of the cell
//! index and seed, so the `GPU` reproduces it without any shared storage).
//! `WGSL` `u32` arithmetic wraps on overflow, matching the golden's `wrapping_*`
//! hash ops exactly, so the integer Gaussian field is bit-identical; the only
//! residual is the last-place slack of the `exp_approx` polynomial under a `GPU`
//! fused multiply-add. The parity test asserts each amplitude component within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel is integer hash plus multiply/add and the one permitted float
//! intrinsic `sqrt` in the portable core-`WGSL` subset — no `sin`, `cos`,
//! `exp`, `pow`, no optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::initial_spectrum`；无第三方引擎源码或衍生代码。
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

/// Workgroup width of the spectrum build; must match `@workgroup_size(256)` in
/// `shaders/water_initial_spectrum.wesl`.
const WORKGROUP_SIZE: u32 = 256;

/// Which statistical spectrum shapes the initial energy distribution. Mirrors
/// [`SpectrumKind`](prism_render_architecture::water::spectrum::SpectrumKind);
/// the integer tags match the `KIND_*` discriminants in the shader.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WaterSpectrumKind {
    /// `Phillips` spectrum: the classic wind-driven ocean spectrum.
    Phillips,
    /// `JONSWAP` spectrum: a fetch-limited peak-enhanced growing wind sea.
    Jonswap,
    /// Pierson-Moskowitz spectrum: a fully developed sea in equilibrium.
    PiersonMoskowitz,
}

impl WaterSpectrumKind {
    /// The shader discriminant for this kind.
    #[must_use]
    fn tag(self) -> u32 {
        match self {
            WaterSpectrumKind::Phillips => 0,
            WaterSpectrumKind::Jonswap => 1,
            WaterSpectrumKind::PiersonMoskowitz => 2,
        }
    }
}

/// One complex spectral amplitude of the built field (`h0(k)` / `h0(-k)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSpectrumComplex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl WaterSpectrumComplex {
    /// Builds a complex amplitude from its parts.
    #[must_use]
    pub const fn new(re: f32, im: f32) -> WaterSpectrumComplex {
        WaterSpectrumComplex { re, im }
    }
}

/// One built ocean initial-spectrum field, row-major over the `resolution x
/// resolution` grid (row `i` walks `+z`/`kz`, column `j` walks `+x`/`kx`).
///
/// Mirrors the `CPU`
/// [`OceanSpectrumField`](prism_render_architecture::water::initial_spectrum::OceanSpectrumField):
/// `h0` and `h0_neg` are the time-zero amplitudes at `+k` and `-k`, the same
/// length. A degenerate request (zero resolution or non-positive patch) yields
/// `resolution == 0` and empty buffers.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterInitialSpectrumField {
    /// Grid resolution `N` (the field holds `N * N` complex amplitudes).
    pub resolution: u32,
    /// Initial amplitudes at `+k`, row-major.
    pub h0: Vec<WaterSpectrumComplex>,
    /// Initial amplitudes at `-k` (the mirror-cell draw), row-major.
    pub h0_neg: Vec<WaterSpectrumComplex>,
}

/// One complex sample as laid out in the storage buffers. `8`-byte `repr(C)`
/// matching `Complex` in `shaders/water_initial_spectrum.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuComplex {
    re: f32,
    im: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_initial_spectrum.wesl` (`48` bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    seed: u32,
    kind: u32,
    directional_exponent: u32,
    wind_x: f32,
    wind_y: f32,
    amplitude: f32,
    peak_enhancement: f32,
    min_wavelength: f32,
    patch_size: f32,
    gravity: f32,
    pad: f32,
}

/// A compiled, reusable ocean initial-spectrum pipeline.
pub struct GpuWaterInitialSpectrum {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterInitialSpectrum {
    /// Compiles the initial-spectrum kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterInitialSpectrum {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_initial_spectrum"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/water_initial_spectrum.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_initial_spectrum_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_initial_spectrum_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_initial_spectrum_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("spectrum_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterInitialSpectrum {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds the `n x n` initial spectral field for one wind-driven sea state,
    /// matching the `CPU` golden
    /// [`build_initial_spectrum`](prism_render_architecture::water::initial_spectrum::build_initial_spectrum)
    /// within the tolerance documented on this module.
    ///
    /// `wind` is the downwind vector `(x, y)`; its magnitude is the wind speed
    /// `V` (m/s). `amplitude`, `peak_enhancement`, `min_wavelength`, and
    /// `directional_exponent` shape the chosen `kind`; `seed` selects the
    /// deterministic Gaussian draw. A non-positive `patch_size` or a zero
    /// `n` yields an empty field (`resolution == 0`, no dispatch issued, since a
    /// storage buffer cannot be zero-sized). `n` may be any positive value — the
    /// spectrum build does not require a power of two.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "The sea-state parameters mirror the CPU golden's SpectrumParams + build_initial_spectrum signature one-to-one; bundling them into a host struct would diverge from the twinned API."
    )]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        n: u32,
        patch_size: f32,
        kind: WaterSpectrumKind,
        wind: (f32, f32),
        amplitude: f32,
        peak_enhancement: f32,
        min_wavelength: f32,
        directional_exponent: u32,
        seed: u32,
    ) -> WaterInitialSpectrumField {
        if n == 0 || patch_size <= 0.0 {
            return WaterInitialSpectrumField {
                resolution: 0,
                h0: Vec::new(),
                h0_neg: Vec::new(),
            };
        }
        let cells = (n as usize) * (n as usize);
        let device = ctx.device();

        let params = Params {
            n,
            seed,
            kind: kind.tag(),
            directional_exponent,
            wind_x: wind.0,
            wind_y: wind.1,
            amplitude,
            peak_enhancement,
            min_wavelength,
            patch_size,
            // Match the golden spectra's gravitational constant exactly.
            gravity: prism_render_architecture::water::GRAVITY,
            pad: 0.0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_initial_spectrum_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let out_bytes = (cells * size_of::<GpuComplex>()) as u64;
        let make_out = |label: &str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: out_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let h0_buf = make_out("prism_volumetric_water_initial_spectrum_h0");
        let h0_neg_buf = make_out("prism_volumetric_water_initial_spectrum_h0_neg");

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_initial_spectrum_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: h0_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: h0_neg_buf.as_entire_binding(),
                },
            ],
        });

        let make_stage = |label: &str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: out_bytes,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let h0_stage = make_stage("prism_volumetric_water_initial_spectrum_h0_stage");
        let h0_neg_stage = make_stage("prism_volumetric_water_initial_spectrum_h0_neg_stage");

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_initial_spectrum_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_initial_spectrum_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cell; cover `n*n` cells, rounding the last
            // workgroup up (the shader guards `idx >= n*n`).
            let groups = (cells as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&h0_buf, 0, &h0_stage, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&h0_neg_buf, 0, &h0_neg_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        h0_stage.slice(..).map_async(MapMode::Read, |_| {});
        h0_neg_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let h0 = read_complex(&h0_stage);
        let h0_neg = read_complex(&h0_neg_stage);

        WaterInitialSpectrumField {
            resolution: n,
            h0,
            h0_neg,
        }
    }
}

/// Reads back a mapped complex staging buffer into owned amplitudes, unmapping
/// it.
fn read_complex(stage: &wgpu::Buffer) -> Vec<WaterSpectrumComplex> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, GpuComplex>(&view)
        .iter()
        .map(|c| WaterSpectrumComplex::new(c.re, c.im))
        .collect();
    drop(view);
    stage.unmap();
    out
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
