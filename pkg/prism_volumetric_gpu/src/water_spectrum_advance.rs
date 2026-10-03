//! `wgpu` compute twin of the Tessendorf ocean spectral amplitude advance
//! ([`dispersion`](prism_render_architecture::water::spectrum::dispersion) +
//! [`advance_amplitude`](prism_render_architecture::water::spectrum::advance_amplitude),
//! in [`spectrum`](prism_render_architecture::water::spectrum)).
//!
//! The animated ocean surface is the inverse `FFT` of a time-evolving Fourier
//! field. The evolution of one amplitude is the heart of that frame: for a wave
//! vector of magnitude `k` the deep-water dispersion relation gives the angular
//! frequency `omega = sqrt(g*k)`, and the complex amplitude at time `t` is the
//! Hermitian-paired phasor advance
//! `h(k, t) = h0(+k) e^{+i omega t} + conj(h0(-k)) e^{-i omega t}`, the pairing
//! that keeps the reconstructed height field real-valued. This twin reproduces
//! that advance on device: one thread evolves one amplitude.
//!
//! The spectral `evolve` `WESL` kernel
//! ([`water::gpu::spectrum_evolve_kernel`](prism_render_architecture::water::gpu))
//! packs eight such advanced fields into four complex buffers for a separable
//! `FFT`; this module isolates the single-amplitude advance so a wide sweep of
//! wave numbers and times exercises the dispersion guard, the phasor algebra
//! and the Hermitian conjugate pairing in one place, pinned directly against
//! the public `CPU` golden.
//!
//! # What is twinned
//!
//! For one query `(h0, h0_neg, k, t)` the kernel reproduces, writing one
//! complex result:
//! - `omega = dispersion(k)`: `sqrt(g*k)` with `g = 9.81`, and `0` for
//!   `k <= 0`.
//! - `theta = omega * t`.
//! - `forward = h0 * (cos(theta) + i sin(theta))` (`Complex::mul_phasor`).
//! - `backward = conj(h0_neg) * (cos(-theta) + i sin(-theta))`.
//! - `result = forward + backward`.
//!
//! The `sin`/`cos` use the hand-rolled
//! [`sin_approx`](prism_render_architecture::water::sin_approx) /
//! [`cos_approx`](prism_render_architecture::water::cos_approx) mirrored in the
//! shader, since the determinism policy forbids the hardware transcendentals.
//!
//! # Correctness model
//!
//! The phasor threads through a range-reduced Taylor polynomial, two complex
//! products and a sum, and `omega` through a `sqrt`, so the `CPU` and `GPU` are
//! not bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few `ULP`. Each component is
//! asserted within a tolerance (`abs_diff <= 1e-5` or `rel_diff <= 1e-4`),
//! tight enough to catch a genuinely wrong port — a dropped conjugate, a
//! flipped phasor sign, a missing dispersion guard — yet loose enough to admit
//! legal last-place slack.
//!
//! # Portability
//!
//! The kernel is `floor`/`select`/`sqrt` plus multiply/add in the portable
//! core-`WGSL` subset — no `sin`, `cos`, `exp`, `pow`, no optional device
//! feature — so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::water::spectrum`；无第三方引擎源码或衍生代码。
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// One amplitude-advance query: the `+k` and `-k` initial complex amplitudes,
/// the wave-number magnitude, and the time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSpectrumAdvanceQuery {
    /// Real part of the `+k` amplitude `h0`.
    pub h0_re: f32,
    /// Imaginary part of the `+k` amplitude `h0`.
    pub h0_im: f32,
    /// Real part of the `-k` amplitude `h0_neg`.
    pub h0_neg_re: f32,
    /// Imaginary part of the `-k` amplitude `h0_neg`.
    pub h0_neg_im: f32,
    /// Wave-number magnitude `k` (rad/m).
    pub k: f32,
    /// Time `t` (seconds).
    pub t: f32,
}

/// The advanced complex amplitude for one query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSpectrumAdvanceResult {
    /// Real part of the advanced amplitude.
    pub re: f32,
    /// Imaginary part of the advanced amplitude.
    pub im: f32,
}

/// One query as uploaded. `32`-byte `repr(C)` matching `Query` in
/// `shaders/water_spectrum_advance.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    h0_re: f32,
    h0_im: f32,
    h0_neg_re: f32,
    h0_neg_im: f32,
    k: f32,
    t: f32,
    pad0: f32,
    pad1: f32,
}

/// One result as read back. `8`-byte `repr(C)` matching `Res` in
/// `shaders/water_spectrum_advance.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    re: f32,
    im: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/water_spectrum_advance.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable spectral amplitude-advance pipeline.
pub struct GpuWaterSpectrumAdvance {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuWaterSpectrumAdvance {
    /// Compiles the amplitude-advance kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuWaterSpectrumAdvance {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/water_spectrum_advance.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("advance_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuWaterSpectrumAdvance {
            module,
            layout,
            pipeline,
        }
    }

    /// Advances every query in `queries` and returns one
    /// [`WaterSpectrumAdvanceResult`] per input, in order.
    ///
    /// Each component matches the `CPU` golden
    /// [`advance_amplitude`](prism_render_architecture::water::spectrum::advance_amplitude)
    /// (composed with
    /// [`dispersion`](prism_render_architecture::water::spectrum::dispersion))
    /// within the tolerance documented on this module. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[WaterSpectrumAdvanceQuery],
    ) -> Vec<WaterSpectrumAdvanceResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                h0_re: q.h0_re,
                h0_im: q.h0_im,
                h0_neg_re: q.h0_neg_re,
                h0_neg_im: q.h0_neg_im,
                k: q.k,
                t: q.t,
                pad0: 0.0,
                pad1: 0.0,
            })
            .collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_water_spectrum_advance_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_water_spectrum_advance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per amplitude, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter()
            .map(|r| WaterSpectrumAdvanceResult { re: r.re, im: r.im })
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
