//! `wgpu` compute twin of the dependency-free `CPU` golden ocean-surface
//! synthesis
//! ([`synthesize_surface`](prism_render_architecture::water::synthesis::synthesize_surface),
//! in [`synthesis`](prism_render_architecture::water::synthesis)).
//!
//! The earlier [`water_fft`](crate::water_fft) twin pinned the 1D butterfly in
//! isolation; this twin closes the loop by running the whole spectral-ocean
//! chain on device: evolve the time-zero complex amplitudes `h0(k)` / `h0(-k)`
//! to `h(k, t)`, inverse-transform that frequency grid into a spatial height
//! field, and add the horizontal `Tessendorf` choppiness displacement — the
//! exact sequence shipping oceans (`WaveWorks`, `Crest`, UE5 Water) run every
//! frame to turn a spectrum into a surface an author can see. Validating the
//! `initial_spectrum -> advance -> ifft2 -> surface` path as one unit (not just
//! per operator) is what proves the real-device spectral ocean actually
//! produces the golden surface.
//!
//! # What is twinned
//!
//! For a batch of `count` patches, each an `n x n` row-major grid laid out back
//! to back, the kernel reproduces the golden conventions exactly:
//! - advance: `h(k, t) = h0(k) e^{+i w t} + conj(h0(-k)) e^{-i w t}`, with the
//!   deep-water dispersion `w = sqrt(g*k)` (`g` = [`water::GRAVITY`]) and the
//!   `DC` guard `k <= 0 -> w = 0`.
//! - height: `real( ifft2( h(k, t) ) )` times the `(-1)^(row+col)` recentring
//!   sign that stands in for the centred spectrum's `fftshift`.
//! - choppiness: `real( ifft2( -i (k/|k|) h(k, t) ) )` per horizontal axis,
//!   with the same recentring sign and the `K_EPS` `DC` guard; the `choppiness`
//!   scale blends it in (`0` = pure height, `1` = full `Tessendorf`).
//!
//! Each workgroup synthesises one patch cooperatively in workgroup memory
//! (`n <= MAX_N = 32`, so `n*n <= 1024` threads), running the separable inverse
//! `FFT` — a raw butterfly over every row, a raw butterfly over every column,
//! then a single `1/(n*n)` scale — bit for bit against the `CPU`
//! [`fft::ifft2`](prism_render_architecture::water::fft::ifft2).
//!
//! The dispersion uses the one permitted float intrinsic `sqrt`; every phasor
//! goes through the hand-rolled
//! [`sin_approx`](prism_render_architecture::water::sin_approx) /
//! [`cos_approx`](prism_render_architecture::water::cos_approx) mirrored in the
//! shader, since the determinism policy forbids the hardware transcendentals.
//!
//! # Correctness model
//!
//! Height and both choppiness fields each thread through `2*log2(n)` butterfly
//! stages of range-reduced Taylor `sin`/`cos` and complex multiply/add; the
//! advance adds two more phasors. The `CPU` and `GPU` share the same polynomial
//! and stage structure, so that approximation is common-mode and cancels; the
//! residual is only the last-place slack of a `GPU` fused multiply-add the
//! scalar reference leaves separate, accumulated over the stages. The parity
//! test asserts each component within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`.
//!
//! # Portability
//!
//! The kernel is `floor`/`select`/multiply/add plus `workgroupBarrier` in the
//! portable core-`WGSL` subset — no `sin`, `cos`, `exp`, `pow`, no optional
//! device feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::synthesis`；无第三方引擎源码或衍生代码。
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

/// Largest patch resolution the single-workgroup scratch supports. `n <= 32`
/// keeps the thread count `n*n <= 1024` (within Metal's and Vulkan's workgroup
/// limits) and the four scratch planes at 16 KiB. Matches `MAX_N` and the
/// `@workgroup_size(1024)` in `shaders/water_surface_synth.wesl`.
pub const MAX_N: u32 = 32;

/// One complex spectral amplitude of the input field (`h0(k)` / `h0(-k)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSurfaceComplex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl WaterSurfaceComplex {
    /// Builds a complex amplitude from its parts.
    #[must_use]
    pub const fn new(re: f32, im: f32) -> WaterSurfaceComplex {
        WaterSurfaceComplex { re, im }
    }
}

/// One synthesised ocean-surface patch, row-major over the `resolution x
/// resolution` grid (row `i` walks `+z`, column `j` walks `+x`).
///
/// Mirrors the `CPU`
/// [`OceanSurface`](prism_render_architecture::water::synthesis::OceanSurface):
/// `height` is the vertical displacement; `displacement_x` / `displacement_z`
/// are the horizontal `Tessendorf` choppiness offsets (zero when synthesised
/// with `choppiness == 0`).
#[derive(Clone, Debug, PartialEq)]
pub struct WaterSurface {
    /// Grid resolution `N` (the surface holds `N * N` samples per patch).
    pub resolution: u32,
    /// Vertical displacement per texel, row-major.
    pub height: Vec<f32>,
    /// Horizontal `+x` displacement per texel (choppiness), row-major.
    pub displacement_x: Vec<f32>,
    /// Horizontal `+z` displacement per texel (choppiness), row-major.
    pub displacement_z: Vec<f32>,
}

/// One complex sample as laid out in the storage buffers. `8`-byte `repr(C)`
/// matching `Complex` in `shaders/water_surface_synth.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuComplex {
    re: f32,
    im: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_surface_synth.wesl` (`32` bytes = two `vec4`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    bits: u32,
    count: u32,
    pad: u32,
    patch_size: f32,
    time: f32,
    choppiness: f32,
    gravity: f32,
}

/// A compiled, reusable ocean-surface-synthesis pipeline.
pub struct GpuWaterSurfaceSynth {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterSurfaceSynth {
    /// Compiles the surface-synthesis kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterSurfaceSynth {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_surface_synth"),
            source: ShaderSource::Wgsl(include_str!("../shaders/water_surface_synth.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_surface_synth_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_surface_synth_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_surface_synth_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("synth_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterSurfaceSynth {
            module,
            layout,
            pipeline,
        }
    }

    /// Synthesises `count = h0.len() / (n*n)` ocean-surface patches, each from an
    /// `n x n` row-major centred spectrum laid out back to back, returning the
    /// three spatial fields concatenated in the same patch layout.
    ///
    /// `h0` / `h0_neg` are the centred time-zero amplitudes
    /// [`OceanSpectrumField::h0`](prism_render_architecture::water::initial_spectrum::OceanSpectrumField)
    /// (same length). Each output component matches the `CPU` golden
    /// [`synthesize_surface`](prism_render_architecture::water::synthesis::synthesize_surface)
    /// within the tolerance documented on this module.
    ///
    /// An empty input returns an empty surface with no dispatch issued (a
    /// storage buffer cannot be zero-sized). `n` must be a power of two in
    /// `1..=MAX_N`, `h0` and `h0_neg` must share a length that is a whole number
    /// of `n*n` patches.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        h0: &[WaterSurfaceComplex],
        h0_neg: &[WaterSurfaceComplex],
        n: u32,
        patch_size: f32,
        time: f32,
        choppiness: f32,
    ) -> WaterSurface {
        assert_eq!(
            h0.len(),
            h0_neg.len(),
            "h0 and h0_neg must carry the same number of amplitudes"
        );
        if h0.is_empty() {
            return WaterSurface {
                resolution: n,
                height: Vec::new(),
                displacement_x: Vec::new(),
                displacement_z: Vec::new(),
            };
        }
        assert!(n >= 1, "resolution n must be at least 1");
        assert!(n <= MAX_N, "resolution n must not exceed MAX_N");
        assert!(is_power_of_two(n), "resolution n must be a power of two");
        let total = (n * n) as usize;
        assert!(
            h0.len().is_multiple_of(total),
            "input length must be a whole number of n*n patches"
        );
        let count = h0.len() / total;
        let device = ctx.device();

        let params = Params {
            n,
            bits: n.trailing_zeros(),
            count: count as u32,
            pad: 0,
            patch_size,
            time,
            choppiness,
            // Match the golden dispersion's gravitational constant exactly.
            gravity: prism_render_architecture::water::GRAVITY,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_surface_synth_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let h0_enc: Vec<GpuComplex> = h0
            .iter()
            .map(|c| GpuComplex { re: c.re, im: c.im })
            .collect();
        let h0_neg_enc: Vec<GpuComplex> = h0_neg
            .iter()
            .map(|c| GpuComplex { re: c.re, im: c.im })
            .collect();
        let h0_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_surface_synth_h0"),
            contents: bytemuck::cast_slice(&h0_enc),
            usage: BufferUsages::STORAGE,
        });
        let h0_neg_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_surface_synth_h0_neg"),
            contents: bytemuck::cast_slice(&h0_neg_enc),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (h0.len() * size_of::<f32>()) as u64;
        let make_out = |label: &str| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: out_bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let height_buf = make_out("prism_volumetric_water_surface_synth_height");
        let disp_x_buf = make_out("prism_volumetric_water_surface_synth_disp_x");
        let disp_z_buf = make_out("prism_volumetric_water_surface_synth_disp_z");

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_surface_synth_bind_group"),
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
                BindGroupEntry {
                    binding: 3,
                    resource: height_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: disp_x_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: disp_z_buf.as_entire_binding(),
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
        let height_stage = make_stage("prism_volumetric_water_surface_synth_height_stage");
        let disp_x_stage = make_stage("prism_volumetric_water_surface_synth_disp_x_stage");
        let disp_z_stage = make_stage("prism_volumetric_water_surface_synth_disp_z_stage");

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_surface_synth_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_surface_synth_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per `n*n` patch; the workgroup cooperates on its own
            // synthesis in workgroup-shared scratch.
            pass.dispatch_workgroups(count as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&height_buf, 0, &height_stage, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&disp_x_buf, 0, &disp_x_stage, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&disp_z_buf, 0, &disp_z_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        height_stage.slice(..).map_async(MapMode::Read, |_| {});
        disp_x_stage.slice(..).map_async(MapMode::Read, |_| {});
        disp_z_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let height = read_f32(&height_stage);
        let displacement_x = read_f32(&disp_x_stage);
        let displacement_z = read_f32(&disp_z_stage);

        WaterSurface {
            resolution: n,
            height,
            displacement_x,
            displacement_z,
        }
    }
}

/// Reads back a mapped `f32` staging buffer into an owned vector, unmapping it.
fn read_f32(stage: &wgpu::Buffer) -> Vec<f32> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();
    out
}

/// Returns `true` when `n` is a positive power of two, mirroring
/// [`fft::is_power_of_two`](prism_render_architecture::water::fft::is_power_of_two).
#[must_use]
fn is_power_of_two(n: u32) -> bool {
    n != 0 && (n & (n - 1)) == 0
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
