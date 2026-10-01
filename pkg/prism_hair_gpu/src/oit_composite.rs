//! `wgpu` compute twin of Prism's `OIT` `k-layer` composite goldens
//! ([`composite_transmittance`](prism_render_architecture::hair::oit_frontend::composite_transmittance)
//! and
//! [`composite_coverage`](prism_render_architecture::hair::oit_frontend::composite_coverage)).
//!
//! Hair silhouettes are built from thousands of sub-pixel strands whose
//! semi-transparent edges overlap in depth, so a `per-pixel` linked list
//! (`PPLL`) captures every fragment covering a pixel and the resolve keeps the
//! nearest handful as a `front-to-back` stack plus one flattened tail opacity.
//! Resolving the visible occlusion of that stack is a `multi-layer` `alpha`
//! blend (`MLAB`): the transmittance is `product(1 - alpha_i)` over the resolved
//! layers times the tail's own `(1 - tail_alpha)`, and the coverage is its
//! complement. AMD `TressFX` and UE5 Groom both ship a variant of this
//! `k-layer` resolve.
//!
//! This twin evaluates that composite batch-wide on the device, one thread per
//! pixel — the array-in/array-out form the `OIT` resolve pass consumes.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairOitComposite::eval`] takes a batch of resolved per-pixel stacks
//! (each a [`LayeredFragments`](prism_render_architecture::hair::oit_frontend::LayeredFragments)
//! as produced by
//! [`sort_and_clip`](prism_render_architecture::hair::oit_frontend::sort_and_clip))
//! and returns one `(transmittance, coverage)` pair per pixel in input order.
//! The pixel index is the invocation id (`@compute @workgroup_size(64)`, a
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! pixel count early-return.
//!
//! # Correctness model
//!
//! The host flattens each pixel's resolved stack into a fixed `8`-slot alpha row
//! (matching the golden's `MAX_OIT_LAYERS`), padding the unused tail slots with
//! `0.0`. Because `(1 - 0) == 1` is the exact multiplicative identity in
//! `IEEE-754`, multiplying through all `8` slots in layer order and then the
//! tail reproduces the golden's shorter product bit-for-bit — the padding never
//! perturbs the result. The `alpha` and tail values are already sanitised to
//! `0..=1` by `sort_and_clip`, so the kernel does no clamping of its own beyond
//! the final `[0, 1]` saturation the goldens apply.
//!
//! The composite is a plain product chain (subtracts, multiplies and a clamp),
//! so the device result is effectively bit-identical to the scalar reference;
//! the twin is nonetheless matched against a tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) for robustness.
//!
//! # Portability
//!
//! The kernel uses only subtracts, multiplies and `clamp` in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so the twin
//! runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `d'Eon`/`TressFX` `k-layer` `MLAB` `alpha` compositing plus
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

use prism_render_architecture::hair::oit_frontend::{
    composite_coverage, composite_transmittance, LayeredFragments,
};

use crate::context::GpuContext;

/// Fixed resolved-layer slot count; matches the golden's `MAX_OIT_LAYERS`.
const MAX_OIT_LAYERS: usize = 8;

/// Uniform parameters for one composite dispatch. Layout matches `Params` in
/// `shaders/oit_composite.wesl`: the pixel count padded to one `16`-byte uniform
/// slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    pixel_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable `per-pixel` `OIT` `k-layer` composite pipeline.
pub struct GpuHairOitComposite {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairOitComposite {
    /// Compiles the `per-pixel` composite kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairOitComposite {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_oit_composite"),
            source: ShaderSource::Wgsl(include_str!("../shaders/oit_composite.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_oit_composite_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_oit_composite_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_oit_composite_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairOitComposite {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the `k-layer` composite for a batch of resolved pixel stacks,
    /// returning one `(transmittance, coverage)` pair per pixel in input order.
    ///
    /// Each pixel's resolved stack is flattened on the host into the fixed
    /// `8`-slot alpha row the kernel reads, padding unused slots with `0.0`
    /// (the exact multiplicative identity) so the device product matches the
    /// golden's shorter product. An empty batch yields an empty vector without a
    /// dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, pixels: &[LayeredFragments]) -> Vec<(f32, f32)> {
        let pixel_count = pixels.len();
        if pixel_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // Flatten each pixel's resolved stack into a fixed 8-slot alpha row plus
        // its tail. Unused layer slots pad with 0.0 so (1 - 0) = 1 is the exact
        // identity in the kernel's product, matching the golden's shorter loop.
        let mut alphas = Vec::with_capacity(pixel_count * MAX_OIT_LAYERS);
        let mut tails = Vec::with_capacity(pixel_count);
        for pixel in pixels {
            for slot in 0..MAX_OIT_LAYERS {
                let alpha = pixel
                    .layers
                    .get(slot)
                    .map_or(0.0, |fragment| fragment.alpha);
                alphas.push(alpha);
            }
            tails.push(pixel.tail_alpha);
        }

        let uniforms = Params {
            pixel_count: pixel_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_len = pixel_count * 2;
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_oit_composite_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let alphas_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_oit_composite_alphas"),
            contents: bytemuck::cast_slice(&alphas),
            usage: BufferUsages::STORAGE,
        });
        let tails_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_oit_composite_tails"),
            contents: bytemuck::cast_slice(&tails),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_oit_composite_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_oit_composite_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_oit_composite_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: alphas_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: tails_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_oit_composite_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_oit_composite_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (pixel_count as u32).div_ceil(64);
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

        flat.chunks_exact(2)
            .map(|pair| (pair[0], pair[1]))
            .collect()
    }
}

/// The `CPU` golden composite for one resolved pixel stack, re-exported so the
/// parity test can assert the device twin against the identical references it
/// mirrors: `(composite_transmittance, composite_coverage)`.
#[must_use]
pub fn reference_composite(layered: &LayeredFragments) -> (f32, f32) {
    (
        composite_transmittance(layered),
        composite_coverage(layered),
    )
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
