//! `wgpu` compute twin of Prism's hair BSDF lobe-importance pdf
//! ([`lobe_pdf`](prism_render_architecture::hair::scatter_lod::lobe_pdf)
//! composed with
//! [`lobe_weights`](prism_render_architecture::hair::scatter_lod::lobe_weights)).
//!
//! A strand's reflectance splits into three Marschner lobes — `R` (surface
//! reflection), `TT` (single transmission) and `TRT` (transmit-reflect-transmit)
//! — and a path tracer importance-samples them by energy. From one fibre's
//! optical inputs (a fresnel reflectance proxy and an absorption-path proxy) the
//! groom pipeline derives the un-normalised per-lobe energies
//!
//! * `R   = F`
//! * `TT  = (1-F)^2 * T`
//! * `TRT = (1-F)^2 * F * T^2`
//!
//! where `T = 1 / (1 + absorption)` is a monotone *rational* stand-in for
//! Beer-Lambert transmittance (the real `exp(-absorption)` lives only in the
//! shading closure), then normalises them into a sampling pdf that is
//! non-negative and sums to one, with a uniform `1/3` fallback for a dead
//! all-zero stack. Every term is a product / rational / normalise of
//! non-negative factors, so the whole map is golden-comparable, array in, array
//! out, panic-free, with no transcendental.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairScatterLod::eval`] takes a batch of per-fibre
//! [`LobeOpticalInput`](prism_render_architecture::hair::scatter_lod::LobeOpticalInput)s
//! and returns, for each one, the three pdf probabilities packed as three
//! consecutive `f32`s in lobe order (`R`, `TT`, `TRT`), preserving input order.
//! The element index is the invocation id (`@compute @workgroup_size(64)`,
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! element count early-return.
//!
//! # No host magic constants
//!
//! Unlike the pigment twins there are no material constants to upload: the pdf
//! is a pure structural fold, so the only uniform is the element count. The
//! `EPS` dead-stack threshold and the three-lobe count are mirrored verbatim
//! from the golden source as structural numerical guards, and both the fresnel
//! and absorption inputs are sanitized in-shader bit-faithfully to the golden's
//! `clamp01` / `sanitize_nonneg`, so the host uploads the raw authored values
//! unchanged.
//!
//! # Portability
//!
//! The kernel uses only multiply/add, `clamp`, `max` and one reciprocal — no
//! `exp`, `pow`, `sin` or optional device feature — so the twin runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Each probability is a chain of products and a normalising reciprocal a `GPU`
//! may fuse, so `CPU` and `GPU` agree to within the documented fma tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than bit-for-bit. The
//! fresnel/absorption guards mirror the golden's `clamp01` / `sanitize_nonneg`
//! exactly (the finite test `x == x` rejects `NaN`, the finite-magnitude bound
//! rejects `+/-inf`), so negative, out-of-range and non-finite inputs produce
//! the same bounded pdf, the result is always non-negative and sums to one, and
//! a dead all-zero stack falls back to the same uniform `1/3` per lobe.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Prism's own deterministic Marschner lobe-importance fold plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::scatter_lod::{
    lobe_pdf, lobe_weights, LobeOpticalInput, LobePdf, LOBE_COUNT,
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

/// Number of `f32` probabilities emitted per fibre, matching the three
/// Marschner lobes `R`/`TT`/`TRT` ([`LOBE_COUNT`]).
pub const PDF_PER_ELEMENT: usize = LOBE_COUNT;

/// Uniform parameters for one scatter-lod dispatch. Layout matches `Params` in
/// `shaders/scatter_lod.wesl`: just the element count padded out to one
/// `16`-byte uniform slot (the pdf fold needs no material constants).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-fibre lobe-pdf pipeline.
pub struct GpuHairScatterLod {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairScatterLod {
    /// Compiles the per-fibre lobe-pdf kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairScatterLod {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_scatter_lod"),
            source: ShaderSource::Wgsl(include_str!("../shaders/scatter_lod.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_scatter_lod_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_scatter_lod_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_scatter_lod_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairScatterLod {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps each fibre's optical input to its three-lobe sampling pdf, returning
    /// them flattened three `f32`s per element in input order (lobe order `R`,
    /// `TT`, `TRT`).
    ///
    /// The three values for element `i` equal the fields of the `CPU` golden
    /// [`lobe_pdf`](prism_render_architecture::hair::scatter_lod::lobe_pdf) of
    /// `lobe_weights(inputs[i])` to within the module's documented fma
    /// tolerance, with negative, out-of-range and non-finite inputs collapsing
    /// to the same bounded pdf and a dead stack to the uniform `1/3` fallback.
    /// An empty batch yields an empty vector without a dispatch — storage
    /// buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, inputs: &[LobeOpticalInput]) -> Vec<f32> {
        let element_count = inputs.len();
        if element_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            element_count: element_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Pack each input as (fresnel, absorption); the shader sanitizes both
        // itself, so upload the raw authored values unchanged.
        let packed: Vec<[f32; 2]> = inputs.iter().map(|i| [i.fresnel, i.absorption]).collect();

        // Output is three f32s (one LobePdf worth) per element.
        let out_len = element_count * PDF_PER_ELEMENT;
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_scatter_lod_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_scatter_lod_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_scatter_lod_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_scatter_lod_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_scatter_lod_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: inputs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_scatter_lod_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_scatter_lod_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (element_count as u32).div_ceil(64);
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

/// The `CPU` golden sampling pdf for one fibre's optical input, re-exported so
/// the parity test can assert the device twin against the identical reference it
/// mirrors.
#[must_use]
pub fn reference_pdf(input: LobeOpticalInput) -> LobePdf {
    lobe_pdf(lobe_weights(input))
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
