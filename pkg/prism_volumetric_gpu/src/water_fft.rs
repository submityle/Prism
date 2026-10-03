//! `wgpu` compute twin of the dependency-free `CPU` golden radix-2
//! `Cooley-Tukey` butterfly `FFT`
//! ([`fft`](prism_render_architecture::water::fft::fft) /
//! [`ifft`](prism_render_architecture::water::fft::ifft), in
//! [`fft`](prism_render_architecture::water::fft)).
//!
//! The animated spectral ocean is the inverse `FFT` of a time-evolving Fourier
//! field; the butterfly transform is the `O(N log N)` kernel that makes that
//! synthesis affordable (`WaveWorks`, `Crest` and UE5's Water all ship a
//! separable `Cooley-Tukey` pass). This twin runs that transform on device: one
//! workgroup transforms one length-`n` array cooperatively in workgroup memory,
//! bit-reversing the input, running `log2(n)` butterfly stages separated by a
//! `workgroupBarrier`, and — on the inverse path — scaling by `1/n`.
//!
//! # What is twinned
//!
//! For a batch of `count` arrays, each of length `n`, laid out back to back, the
//! kernel reproduces the golden conventions exactly:
//! - `fft`: `X[k] = sum_n x[n] e^{-i 2*PI k n / N}` (no scale).
//! - `ifft`: `x[n] = (1/N) sum_k X[k] e^{+i 2*PI k n / N}`.
//! - Non-power-of-two (and `n <= 1`) lengths are returned verbatim — a
//!   deterministic no-op matching the golden's "skip, do not crash" contract,
//!   with the inverse scale left off exactly as `ifft` leaves it off there.
//!
//! The twiddle `sin`/`cos` use the hand-rolled
//! [`sin_approx`](prism_render_architecture::water::sin_approx) /
//! [`cos_approx`](prism_render_architecture::water::cos_approx) mirrored in the
//! shader, since the determinism policy forbids the hardware transcendentals.
//!
//! # Correctness model
//!
//! Each output threads through `log2(n)` butterfly stages of range-reduced
//! Taylor `sin`/`cos` and complex multiply/add. The `CPU` and `GPU` share the
//! same polynomial and stage structure, so that approximation is common-mode
//! and cancels; the residual is only the last-place slack of a `GPU` fused
//! multiply-add the scalar reference leaves separate, accumulated over the
//! stages. The parity test asserts each component within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` — tight enough to fail a wrong port (a dropped
//! bit-reversal, a flipped twiddle sign, a missing inverse scale) yet loose
//! enough to admit the accumulated last-place slack.
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
//! Provenance: 孪生自本仓 `prism_render_architecture::water::fft`；无第三方引擎源码或衍生代码。
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

/// Largest transform length the workgroup-shared scratch supports. Matches
/// `MAX_N` and the `@workgroup_size(256)` in `shaders/water_fft.wesl`.
pub const MAX_N: u32 = 256;

/// One complex sample, used for both the input and the transformed output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFftComplex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl WaterFftComplex {
    /// Builds a complex sample from its parts.
    #[must_use]
    pub const fn new(re: f32, im: f32) -> WaterFftComplex {
        WaterFftComplex { re, im }
    }
}

/// One complex sample as laid out in the storage buffers. `8`-byte `repr(C)`
/// matching `Complex` in `shaders/water_fft.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuComplex {
    re: f32,
    im: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_fft.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    inverse: u32,
    count: u32,
    bits: u32,
}

/// A compiled, reusable radix-2 butterfly `FFT` pipeline.
pub struct GpuWaterFft {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterFft {
    /// Compiles the butterfly `FFT` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterFft {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_fft"),
            source: ShaderSource::Wgsl(include_str!("../shaders/water_fft.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_fft_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_fft_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_fft_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("fft_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterFft {
            module,
            layout,
            pipeline,
        }
    }

    /// Transforms a batch of `count = input.len() / n` length-`n` arrays laid out
    /// back to back, returning one `WaterFftComplex` per input in the same
    /// layout.
    ///
    /// `inverse` selects the inverse transform (`+` twiddle sign and the `1/n`
    /// scale). Each component matches the `CPU` golden
    /// [`fft`](prism_render_architecture::water::fft::fft) /
    /// [`ifft`](prism_render_architecture::water::fft::ifft) within the tolerance
    /// documented on this module. Non-power-of-two (and `n <= 1`) lengths are
    /// returned verbatim.
    ///
    /// An empty `input` returns an empty vector with no dispatch issued (a
    /// storage buffer cannot be zero-sized). `n` must be in `1..=MAX_N` and must
    /// divide `input.len()` evenly.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        input: &[WaterFftComplex],
        n: u32,
        inverse: bool,
    ) -> Vec<WaterFftComplex> {
        if input.is_empty() {
            return Vec::new();
        }
        assert!(n >= 1, "transform length n must be at least 1");
        assert!(n <= MAX_N, "transform length n must not exceed MAX_N");
        let n_usize = n as usize;
        assert!(
            input.len().is_multiple_of(n_usize),
            "input length must be a whole number of length-n arrays"
        );
        let count = input.len() / n_usize;
        let device = ctx.device();

        let params = Params {
            n,
            inverse: u32::from(inverse),
            count: count as u32,
            // Trailing-zero count equals log2(n) for a power of two; the shader
            // ignores it on the non-power-of-two passthrough path.
            bits: if is_power_of_two(n) {
                n.trailing_zeros()
            } else {
                0
            },
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_fft_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuComplex> = input
            .iter()
            .map(|c| GpuComplex { re: c.re, im: c.im })
            .collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_fft_input"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (input.len() * size_of::<GpuComplex>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_fft_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_fft_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_fft_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_fft_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_fft_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per length-`n` array; the workgroup cooperates on
            // its own transform in workgroup-shared scratch.
            pass.dispatch_workgroups(count as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuComplex>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter()
            .map(|c| WaterFftComplex { re: c.re, im: c.im })
            .collect()
    }
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
