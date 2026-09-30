//! `wgpu` compute twin of the spectral-to-RGB collapse
//! ([`spectral_to_rgb`](prism_render_architecture::volumetric::spectral::spectral_to_rgb)).
//!
//! The spectral night-sky model (design section 8b) represents a spectral power
//! distribution as a partition-of-unity set of wavelength-bucket weights
//! ([`SpectralBands`](prism_render_architecture::volumetric::spectral::SpectralBands)).
//! [`spectral_to_rgb`](prism_render_architecture::volumetric::spectral::spectral_to_rgb)
//! assigns each bucket a representative wavelength at its centre within the
//! visible span, converts it to a non-negative, `L1`-normalised CIE-flavoured
//! `RGB` response, and accumulates it weighted by the bucket weight:
//!
//! ```text
//! frac_i = (i + 0.5) / n
//! nm_i   = lerp(VIS_MIN_NM, VIS_MAX_NM, frac_i)
//! rgb    = sum_i weights[i] * cie_rgb_response(nm_i)
//! ```
//!
//! Because each per-band response sums to one across its channels, the total of
//! the returned `RGB` channels equals the sum of the band weights (one for a
//! normalised, non-empty band set), so the mapping conserves weight and never
//! produces a negative channel. An empty band set maps to the zero vector. The
//! `CPU` golden
//! [`spectral_to_rgb`](prism_render_architecture::volumetric::spectral::spectral_to_rgb)
//! owns that math; [`GpuSpectralToRgb`] is the on-device twin that runs one
//! thread per distribution and reproduces the same value.
//!
//! # Correctness model
//!
//! The per-band Gaussian response uses the *same* hand-rolled `exp_approx` the
//! reference uses — base-two range reduction with a fractional seven-term
//! polynomial times an integer power assembled from the `f32` exponent field —
//! not the device-native `exp`, and the `lerp` is expanded to the *same* closed
//! form the `CPU` `math` module uses. Mirroring those keeps the twin bit-close
//! to the reference, so the parity test asserts a tight tolerance
//! (`abs_diff < 1e-6` or `rel_diff < 1e-5`). The scenes also assert every
//! channel is non-negative and that the channels sum to the band-weight total
//! (one for a normalised set, zero for an empty one), so a degenerate kernel
//! could not pass.
//!
//! # Layout
//!
//! Each query indexes a shared, flattened weights buffer via an `offset`/`len`
//! pair, so distributions of any bucket count coexist in one dispatch. Results
//! are `vec4` (`xyz = rgb`, `w` unused) so the storage buffer stays `16`-byte
//! aligned.
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
//! Provenance: standard CIE-flavoured spectral-to-RGB collapse plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::spectral::SpectralBands;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One spectral-to-RGB result: the linear `RGB` triple the distribution maps to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectralRgb {
    /// Linear red channel, non-negative.
    pub r: f32,
    /// Linear green channel, non-negative.
    pub g: f32,
    /// Linear blue channel, non-negative.
    pub b: f32,
}

/// One query as uploaded: an `offset`/`len` window into the flattened weights
/// buffer. `16`-byte `repr(C)` matching `Query` in
/// `shaders/spectral_to_rgb.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    offset: u32,
    len: u32,
    pad0: u32,
    pad1: u32,
}

/// One result as read back. `16`-byte stride matching `results` in the shader
/// (the `RGB` triple plus one pad word).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuRgb {
    r: f32,
    g: f32,
    b: f32,
    pad: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/spectral_to_rgb.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable spectral-to-RGB pipeline.
pub struct GpuSpectralToRgb {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSpectralToRgb {
    /// Compiles the spectral-to-RGB shader on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSpectralToRgb {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb"),
            source: ShaderSource::Wgsl(include_str!("../shaders/spectral_to_rgb.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("spectral_to_rgb_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSpectralToRgb {
            module,
            layout,
            pipeline,
        }
    }

    /// Collapses every band distribution in `bands` to linear `RGB`, returning
    /// one [`SpectralRgb`] per input in order.
    ///
    /// The returned triple for distribution `d` equals
    /// [`spectral_to_rgb`](prism_render_architecture::volumetric::spectral::spectral_to_rgb)`(&d)`
    /// to within the tolerance documented on this module. An empty `bands`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, bands: &[SpectralBands]) -> Vec<SpectralRgb> {
        if bands.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        // Flatten each distribution's weights into one shared buffer, recording
        // the offset/len window each query reads.
        let mut flat_weights: Vec<f32> = Vec::new();
        let mut gpu_queries: Vec<GpuQuery> = Vec::with_capacity(bands.len());
        for band in bands {
            let offset = flat_weights.len() as u32;
            let w = band.weights();
            flat_weights.extend_from_slice(w);
            gpu_queries.push(GpuQuery {
                offset,
                len: w.len() as u32,
                pad0: 0,
                pad1: 0,
            });
        }
        // Storage buffers cannot be zero-sized; when every distribution is
        // empty the flattened buffer is empty, so pad it with one unread word.
        if flat_weights.is_empty() {
            flat_weights.push(0.0);
        }

        let gpu_params = Params {
            count: bands.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (bands.len() as u64) * (size_of::<GpuRgb>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let weights_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_weights"),
            contents: bytemuck::cast_slice(&flat_weights),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_bind_group"),
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
                    resource: weights_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_spectral_to_rgb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_spectral_to_rgb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (bands.len() as u32).div_ceil(64);
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuRgb>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), bands.len());
        gpu_results
            .into_iter()
            .map(|c| SpectralRgb {
                r: c.r,
                g: c.g,
                b: c.b,
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
