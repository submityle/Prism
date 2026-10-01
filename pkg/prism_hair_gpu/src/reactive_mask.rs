//! `wgpu` compute twin of Prism's fibre-aware temporal reactive-mask kernel
//! ([`reactivity_map`](prism_render_architecture::hair::reactive_mask::reactivity_map),
//! i.e.
//! [`pixel_reactivity`](prism_render_architecture::hair::reactive_mask::pixel_reactivity)
//! mapped over a span).
//!
//! Thin hair fibres are the worst case for temporal anti-aliasing (`TAA`) and
//! temporal upscalers (`TSR`): a strand covers only a fraction of a pixel, so
//! its sub-pixel coverage flickers between frames and the history buffer smears
//! those flickers into ghosts and trails. The real-time fix, popularised by
//! `UE5`'s reactive mask and visible in motion-heavy titles such as
//! `Alan Wake 2`, is to emit a per-pixel `reactivity` in `[0, max_reactivity]`:
//! `0` trusts the temporal history fully, `1` leans on the current frame and
//! rejects stale history. Hair pushes `reactivity` up where ghosting is most
//! likely — low sub-pixel `coverage`, high screen-space `velocity`, and large
//! `depth_delta` (edges and disocclusions).
//!
//! # What the kernel evaluates
//!
//! [`GpuHairReactiveMask::eval`] takes a batch of per-pixel
//! [`PixelReactiveInput`](prism_render_architecture::hair::reactive_mask::PixelReactiveInput)s
//! and returns, for each one, a single `reactivity` `f32` in input order. The
//! value is a weighted sum of three monotone ramps — `1 - coverage`, the
//! velocity saturation ramp `v / (v + k)` and the depth saturation ramp
//! `d / (d + k)` — clamped into `[0, max_reactivity]`. The element index is the
//! invocation id (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the element count early-return.
//!
//! # Host uploads raw policy
//!
//! The per-term weights and the `max_reactivity` clamp are uploaded raw as a
//! uniform; the shader sanitizes them in-place bit-faithfully to the golden's
//! [`ReactiveParams::sanitized`](prism_render_architecture::hair::reactive_mask::ReactiveParams::sanitized)
//! (an idempotent fold), so the host never pre-folds the policy. The two
//! saturation constants are mirrored verbatim from the golden as structural
//! numbers.
//!
//! # Portability
//!
//! The kernel uses only multiply/add, subtraction, one reciprocal, `abs` and
//! `clamp` — no `exp`, `pow`, `sin` or optional device feature — so the twin
//! runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The weighted sum is a chain of products and adds a `GPU` may fuse, so `CPU`
//! and `GPU` agree to within the documented fma tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`) rather than bit-for-bit. The weight/coverage/magnitude and
//! `max_reactivity` guards mirror the golden exactly (the finite test `x == x`
//! rejects `NaN`, the finite-magnitude bound rejects `+/-inf`), so negative,
//! out-of-range and non-finite inputs produce the same bounded result, which is
//! always finite and lands in `[0, max_reactivity]`.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Prism's own deterministic temporal reactive-mask fold plus
//! `wgpu` compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::reactive_mask::{
    pixel_reactivity, PixelReactiveInput, ReactiveParams,
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

/// Uniform parameters for one reactive-mask dispatch. Layout matches `Params`
/// in `shaders/reactive_mask.wesl`: the four raw policy scalars (sanitized in
/// shader) followed by the element count, padded out to a `32`-byte block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    coverage_weight: f32,
    velocity_weight: f32,
    depth_weight: f32,
    max_reactivity: f32,
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-pixel reactive-mask pipeline.
pub struct GpuHairReactiveMask {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairReactiveMask {
    /// Compiles the per-pixel reactive-mask kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairReactiveMask {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_reactive_mask"),
            source: ShaderSource::Wgsl(include_str!("../shaders/reactive_mask.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_reactive_mask_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_reactive_mask_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_reactive_mask_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairReactiveMask {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps each pixel's input to its `reactivity`, returning one `f32` per
    /// element in input order.
    ///
    /// The value for element `i` equals the `CPU` golden
    /// [`pixel_reactivity`](prism_render_architecture::hair::reactive_mask::pixel_reactivity)
    /// of `inputs[i]` to within the module's documented fma tolerance, with
    /// negative, out-of-range and non-finite weights / inputs collapsing to the
    /// same bounded result in `[0, max_reactivity]`. An empty batch yields an
    /// empty vector without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        params: ReactiveParams,
        inputs: &[PixelReactiveInput],
    ) -> Vec<f32> {
        let element_count = inputs.len();
        if element_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // Upload the raw authored policy; the shader sanitizes it itself,
        // mirroring the golden's idempotent `sanitized()`.
        let uniforms = Params {
            coverage_weight: params.coverage_weight,
            velocity_weight: params.velocity_weight,
            depth_weight: params.depth_weight,
            max_reactivity: params.max_reactivity,
            element_count: element_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Pack each input as (coverage, screen_velocity, depth_delta, pad); the
        // shader sanitizes every field, so upload the raw authored values.
        let packed: Vec<[f32; 4]> = inputs
            .iter()
            .map(|i| [i.coverage, i.screen_velocity, i.depth_delta, 0.0])
            .collect();

        // One reactivity f32 per element.
        let out_len = element_count;
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_reactive_mask_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_reactive_mask_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_reactive_mask_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_reactive_mask_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_reactive_mask_bind_group"),
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
            label: Some("prism_hair_reactive_mask_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_reactive_mask_pass"),
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

/// The `CPU` golden `reactivity` for one pixel's input under `params`,
/// re-exported so the parity test can assert the device twin against the
/// identical reference it mirrors.
#[must_use]
pub fn reference_reactivity(params: ReactiveParams, input: PixelReactiveInput) -> f32 {
    pixel_reactivity(params, input)
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
