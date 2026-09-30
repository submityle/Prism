//! `wgpu` compute twin of Prism's continuous-LOD screen-door dither decision
//! ([`strand_survives_dither`](prism_render_architecture::hair::transition::strand_survives_dither)).
//!
//! When a groom cross-fades between two LOD tiers, popping is hidden by
//! dissolving the finer representation strand-by-strand instead of switching all
//! strands at once. Each strand hashes to a stable value in `[0, 1)` and keeps
//! drawing as the finer tier while that hash is at least the cross-fade `blend`,
//! so the kept fraction is `(1 - blend)`: at `blend == 0` every strand survives,
//! and at `blend == 1` none do (the hash is always `< 1`). This is a pure
//! per-strand decision with no cross-strand dependency, so it maps to one `GPU`
//! thread per strand. The `CPU` golden is
//! [`strand_survives_dither`](prism_render_architecture::hair::transition::strand_survives_dither);
//! this crate is the on-device twin that runs the identical decision so a passing
//! real-device parity test is direct evidence the ported kernel dissolves the
//! same strands as the reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairTransition::eval`] takes a per-groom `seed`, the cross-fade `blend`
//! and a strand count, and returns one `bool` per strand: `true` when the strand
//! keeps drawing as the finer tier at that blend, `false` when it has dissolved.
//! The strand index is simply the invocation id, so no per-strand input buffer
//! is needed.
//!
//! # Portability
//!
//! The kernel uses only integer arithmetic and one `u32`-to-`f32` conversion in
//! the portable core-`WGSL` subset — no `exp`, `pow`, `sin` or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The decision reproduces the golden's hand-written `splitmix64` integer hash
//! bit-for-bit (a `vec2<u32>` (lo, hi) emulation stands in for the 64-bit
//! integer baseline `WGSL` lacks) and compares the resulting float against
//! `blend`. There is no transcendental and no fused multiply-add anywhere on the
//! path, so the `CPU` and `GPU` produce the identical survive flag for every
//! strand. The parity test therefore asserts exact equality rather than a
//! tolerance.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard stable-hash screen-door / stochastic-LOD dither plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

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

/// Uniform parameters for one dither dispatch. Layout matches `Params` in
/// `shaders/transition.wesl`: the per-groom seed, the strand count bounding the
/// dispatch and the cross-fade blend, padded to one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    seed: u32,
    strand_count: u32,
    blend: f32,
    pad: u32,
}

/// A compiled, reusable per-strand dither pipeline.
pub struct GpuHairTransition {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairTransition {
    /// Compiles the per-strand dither kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairTransition {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_transition"),
            source: ShaderSource::Wgsl(include_str!("../shaders/transition.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_transition_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_transition_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_transition_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairTransition {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the screen-door dither survive flag for every strand at the
    /// given cross-fade `blend`, returning one `bool` per strand.
    ///
    /// `seed` scopes the per-groom hash pattern and `strand_count` is the number
    /// of strands to decide (their indices are `0..strand_count`). The flag for
    /// strand `i` equals the `CPU` golden
    /// [`strand_survives_dither`](prism_render_architecture::hair::transition::strand_survives_dither)
    /// with the same `seed`, `i` and `blend` exactly — the path is
    /// transcendental-free and fma-free, so the twin is bit-identical, not merely
    /// close. A zero strand count yields an empty vector without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, seed: u32, blend: f32, strand_count: usize) -> Vec<bool> {
        if strand_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            seed,
            strand_count: strand_count as u32,
            blend,
            pad: 0,
        };

        let out_bytes = (strand_count as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_transition_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_transition_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_transition_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_transition_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_transition_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_transition_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (strand_count as u32).div_ceil(64);
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
        let flags = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        flags.into_iter().map(|f| f != 0).collect()
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
