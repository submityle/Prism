//! `wgpu` compute twin of Prism's far-field roughness gain
//! ([`far_field_roughness_gain`](prism_render_architecture::hair::scatter_lod::far_field_roughness_gain)).
//!
//! As a fibre shrinks on screen the near-/far-field blend ramps toward the
//! analytic far-field aggregate (see [`crate::scatter_blend`]). The far-field
//! regime widens the `Marschner`/`Chiang` lobes by `1 + blend * max_gain`, which
//! integrates many sub-pixel fibres per sample and removes the shimmer/aliasing a
//! sharp lobe would produce. This twin evaluates that gain batch-wide on the
//! device, one thread per `(blend, max_gain)` pair — the array-in/array-out form
//! the far-field shading path consumes per fibre or per cluster.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairFarFieldGain::eval`] takes a batch of `(blend, max_gain)` pairs and
//! returns one roughness gain (always `>= 1`) per pair in input order. The pair
//! index is the invocation id (`@compute @workgroup_size(64)`, a one-dimensional
//! dispatch over `global_invocation_id.x`); invocations past the pair count
//! early-return.
//!
//! # Correctness model
//!
//! Both inputs are sanitised in-shader bit-faithfully to the golden: the blend
//! is clamped to `[0, 1]` (`clamp01`, a non-finite blend collapsing to `0`) and
//! a negative or non-finite `max_gain` collapses to `0` (`sanitize_nonneg`), so a
//! stray `NaN` can never poison the gain. The `1 + blend * max_gain` is a single
//! multiply-add the `GPU` may fuse where the scalar reference leaves it separate,
//! so the twin is matched against a tolerance (`abs_diff < 1e-4` or
//! `rel_diff < 1e-3`), not bit-for-bit.
//!
//! # Portability
//!
//! The kernel uses only compares, `clamp`, `max` and a multiply-add in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature — so
//! the twin runs unmodified on Metal, Vulkan and DX12.
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

use prism_render_architecture::hair::scatter_lod::far_field_roughness_gain;

use crate::context::GpuContext;

/// Uniform parameters for one roughness-gain dispatch. Layout matches `Params`
/// in `shaders/far_field_gain.wesl`: the pair count padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    pair_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable per-pair far-field roughness-gain pipeline.
pub struct GpuHairFarFieldGain {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairFarFieldGain {
    /// Compiles the per-pair roughness-gain kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairFarFieldGain {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_far_field_gain"),
            source: ShaderSource::Wgsl(include_str!("../shaders/far_field_gain.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_far_field_gain_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_far_field_gain_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_far_field_gain_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairFarFieldGain {
            module,
            layout,
            pipeline,
        }
    }

    /// Maps each `(blend, max_gain)` pair to its far-field roughness gain,
    /// returning one `f32` (always `>= 1`) per input in order.
    ///
    /// The gain for pair `i` matches the `CPU` golden
    /// [`far_field_roughness_gain`](prism_render_architecture::hair::scatter_lod::far_field_roughness_gain)
    /// within the fma tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`), with the
    /// blend clamped to `[0, 1]` and negative/non-finite `max_gain` sanitised to
    /// the same `0` the reference uses. An empty batch yields an empty vector
    /// without a dispatch — storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, pairs: &[(f32, f32)]) -> Vec<f32> {
        let pair_count = pairs.len();
        if pair_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            pair_count: pair_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Flatten each pair to 2 f32: blend, max_gain. The raw authored values
        // are uploaded unchanged; the shader sanitises them bit-faithfully.
        let mut flat: Vec<f32> = Vec::with_capacity(pair_count * 2);
        for &(blend, max_gain) in pairs {
            flat.push(blend);
            flat.push(max_gain);
        }

        let out_bytes = (pair_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_far_field_gain_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_far_field_gain_inputs"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_far_field_gain_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_far_field_gain_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_far_field_gain_bind_group"),
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
            label: Some("prism_hair_far_field_gain_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_far_field_gain_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (pair_count as u32).div_ceil(64);
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

/// The `CPU` golden far-field roughness gain for one pair, re-exported so the
/// parity test can assert the device twin against the identical reference it
/// mirrors.
#[must_use]
pub fn reference_far_field_gain(blend: f32, max_gain: f32) -> f32 {
    far_field_roughness_gain(blend, max_gain)
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
