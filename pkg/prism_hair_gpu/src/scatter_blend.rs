//! `wgpu` compute twin of Prism's near-/far-field scatter-blend ramp
//! ([`scatter_blend`](prism_render_architecture::hair::scatter_lod::scatter_blend)).
//!
//! A production hair renderer splits the three `Marschner`/`Chiang` lobes into a
//! near-field regime (close-up, per-fibre) and an analytic far-field regime
//! (distant, sub-pixel) following the `d'Eon` 2011 energy-conserving split. The
//! blend factor is the continuous `[0, 1]` crossfade between the two across a
//! screen-space transition band, which is what lets the §4 LOD ladder cross the
//! near->far boundary without a pop. This twin evaluates that ramp batch-wide on
//! the device, one thread per projected fibre width — the array-in/array-out
//! form the LOD/raster path consumes per fibre or per cluster.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairScatterBlend::eval`] takes a batch of projected fibre widths (in
//! pixels) and a single [`ScatterLodThresholds`] pair, and returns one blend
//! factor per width in input order: `0` at/above the near threshold (pure
//! near-field), `1` at/below the far threshold (pure far-field), linearly ramped
//! between. The width index is the invocation id (`@compute
//! @workgroup_size(64)`, a one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the width count early-return.
//!
//! # Correctness model
//!
//! The thresholds are a per-dispatch constant, so the host sanitises them once
//! with the golden's [`ScatterLodThresholds::sanitized`] (`near_px >= far_px >=
//! 0`, both finite) and uploads the finished bounds as uniforms; the per-element
//! work is sanitising the width (a non-finite width collapses to `0`, the finest
//! footprint, matching the golden) and evaluating the ramp. The hard `0.0`/`1.0`
//! saturation returns are exact; only the in-band `(near - w) / span` ramp
//! divides, which the `GPU` may round a few `ULP` differently from the scalar
//! reference, so the twin is matched against a tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`), not bit-for-bit.
//!
//! # Portability
//!
//! The kernel uses only compares, `max`, a divide and `clamp` in the portable
//! core-`WGSL` subset — no `exp`, `pow` or optional device feature — so the twin
//! runs unmodified on Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: `d'Eon` 2011 near-/far-field hair scattering split plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

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

use prism_render_architecture::hair::scatter_lod::{scatter_blend, ScatterLodThresholds};

use crate::context::GpuContext;

/// Uniform parameters for one scatter-blend dispatch. Layout matches `Params` in
/// `shaders/scatter_blend.wesl`: the width count plus the host-sanitised
/// near/far thresholds, padded to one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width_count: u32,
    near_px: f32,
    far_px: f32,
    pad0: u32,
}

/// A compiled, reusable per-width near-/far-field scatter-blend pipeline.
pub struct GpuHairScatterBlend {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairScatterBlend {
    /// Compiles the per-width scatter-blend kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairScatterBlend {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_scatter_blend"),
            source: ShaderSource::Wgsl(include_str!("../shaders/scatter_blend.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_scatter_blend_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_scatter_blend_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_scatter_blend_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairScatterBlend {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps each projected fibre width to its near->far blend factor, returning
    /// one `f32` in `[0, 1]` per input in order.
    ///
    /// The blend for width `i` matches the `CPU` golden
    /// [`scatter_blend`](prism_render_architecture::hair::scatter_lod::scatter_blend)
    /// of `widths[i]` under the same (raw) thresholds within the fma tolerance
    /// (`abs_diff < 1e-4` or `rel_diff < 1e-3`), with non-finite widths sanitised
    /// to the same `0` the reference uses. The thresholds are sanitised once on
    /// the host exactly as the golden does internally. An empty batch yields an
    /// empty vector without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        widths: &[f32],
        thresholds: ScatterLodThresholds,
    ) -> Vec<f32> {
        let width_count = widths.len();
        if width_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        // Sanitise the thresholds once, exactly as the golden does internally, so
        // the uniform the shader reads matches the reference's working bounds.
        let t = thresholds.sanitized();
        let uniforms = Params {
            width_count: width_count as u32,
            near_px: t.near_px,
            far_px: t.far_px,
            pad0: 0,
        };

        let out_bytes = (width_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_scatter_blend_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let widths_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_scatter_blend_widths"),
            contents: bytemuck::cast_slice(widths),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_scatter_blend_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_scatter_blend_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_scatter_blend_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: widths_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_scatter_blend_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_scatter_blend_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (width_count as u32).div_ceil(64);
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

/// The `CPU` golden scatter blend for one width, re-exported so the parity test
/// can assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_scatter_blend(fiber_width_px: f32, thresholds: ScatterLodThresholds) -> f32 {
    scatter_blend(fiber_width_px, thresholds)
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
