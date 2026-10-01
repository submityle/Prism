//! `wgpu` compute twin of Prism's Marschner lobe importance sampling
//! ([`sample_lobe`](prism_render_architecture::hair::scatter_lod::sample_lobe)).
//!
//! A strand's reflectance splits into three Marschner lobes — `R`, `TT`, `TRT`
//! (see [`crate::scatter_lod`]). A path tracer importance-samples one lobe per
//! bounce from a canonical uniform `u`: it walks the per-lobe pdf as a cdf
//! (`R -> TT -> TRT`), picks the lobe whose slice contains `u`, and remaps `u`
//! back into `[0, 1)` inside that slice so the caller can reuse it as a fresh
//! stratified sample for the lobe's own azimuthal/longitudinal term. This twin
//! evaluates that choice batch-wide on the device, one thread per `(pdf, u)`
//! tuple — the array-in/array-out form the sampler consumes per path.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairSampleLobe::eval`] takes a batch of `(LobePdf, u)` tuples and
//! returns, per tuple, three consecutive `f32`s in input order: the chosen lobe
//! index (`0 = R`, `1 = TT`, `2 = TRT`, stored as an exactly representable
//! `f32`), the remapped sample in `[0, 1)`, and the chosen lobe's probability.
//! The tuple index is the invocation id (`@compute @workgroup_size(64)`, a
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! element count early-return.
//!
//! # Correctness model
//!
//! Every input is sanitised in-shader bit-faithfully to the golden: `u` is
//! clamped to `[0, 1]` (`clamp01`, a non-finite `u` collapsing to `0`) and each
//! pdf component is clamped to a finite, non-negative value (`sanitize_nonneg`).
//! The cdf is walked `R -> TT -> TRT` and the final `TRT` lobe absorbs the upper
//! end (including `u == 1`), so a valid lobe is always returned even for a
//! degenerate all-zero pdf. The chosen lobe index and its probability are exact
//! (a select and a pass of a sanitised component), so only the
//! `(u - lo) / p` remap divides; the twin matches the integer lobe index exactly
//! and the two floats within the fma tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`).
//!
//! # Portability
//!
//! The kernel uses only compares, `clamp`, `max`, add and one divide in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature — so
//! the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `Marschner` 2003 three-lobe hair BSDF importance sampling plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::scatter_lod::{sample_lobe, LobePdf, LobeSample};

use crate::context::GpuContext;

/// Number of `f32`s each tuple produces: lobe index, remapped `u`, probability.
pub const SAMPLE_PER_ELEMENT: usize = 3;

/// Number of `f32`s each input tuple occupies: `pdf.r`, `pdf.tt`, `pdf.trt`, `u`.
const INPUT_PER_ELEMENT: usize = 4;

/// Uniform parameters for one lobe-sampling dispatch. Layout matches `Params`
/// in `shaders/sample_lobe.wesl`: the element count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    elem_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-tuple lobe-importance-sampling pipeline.
pub struct GpuHairSampleLobe {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairSampleLobe {
    /// Compiles the per-tuple lobe-sampling kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairSampleLobe {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_sample_lobe"),
            source: ShaderSource::Wgsl(include_str!("../shaders/sample_lobe.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_sample_lobe_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_sample_lobe_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_sample_lobe_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairSampleLobe {
            module,
            layout,
            pipeline,
        }
    }

    /// Importance-samples one lobe per `(pdf, u)` tuple, returning one
    /// [`LobeSample`] per input in order.
    ///
    /// The sample for tuple `i` matches the `CPU` golden
    /// [`sample_lobe`](prism_render_architecture::hair::scatter_lod::sample_lobe):
    /// the lobe index is exact, and `remapped_u`/`pdf` match within the fma
    /// tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`), with `u` clamped to
    /// `[0, 1]` and each pdf component sanitised to the same finite, non-negative
    /// value the reference uses. An empty batch yields an empty vector without a
    /// dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, tuples: &[(LobePdf, f32)]) -> Vec<LobeSample> {
        let elem_count = tuples.len();
        if elem_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            elem_count: elem_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Flatten each tuple to 4 f32: pdf.r, pdf.tt, pdf.trt, u. The raw
        // authored values are uploaded unchanged; the shader sanitises them.
        let mut flat: Vec<f32> = Vec::with_capacity(elem_count * INPUT_PER_ELEMENT);
        for &(pdf, u) in tuples {
            flat.push(pdf.r);
            flat.push(pdf.tt);
            flat.push(pdf.trt);
            flat.push(u);
        }

        let out_len = elem_count * SAMPLE_PER_ELEMENT;
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_sample_lobe_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_sample_lobe_inputs"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_sample_lobe_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_sample_lobe_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_sample_lobe_bind_group"),
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
            label: Some("prism_hair_sample_lobe_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_sample_lobe_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (elem_count as u32).div_ceil(64);
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
        let raw = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        let mut out = Vec::with_capacity(elem_count);
        for chunk in raw.chunks_exact(SAMPLE_PER_ELEMENT) {
            out.push(LobeSample {
                lobe: chunk[0] as usize,
                remapped_u: chunk[1],
                pdf: chunk[2],
            });
        }
        out
    }
}

/// The `CPU` golden lobe sample for one `(pdf, u)` tuple, re-exported so the
/// parity test can assert the device twin against the identical reference it
/// mirrors.
#[must_use]
pub fn reference_sample_lobe(pdf: LobePdf, u: f32) -> LobeSample {
    sample_lobe(pdf, u)
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
