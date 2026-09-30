//! `wgpu` compute twin of the virtual-geometry 64-bit vis-buffer word codec
//! ([`pack_vis`](prism_render_architecture::virtual_geometry::pack_vis),
//! [`vis_depth`](prism_render_architecture::virtual_geometry::vis_depth),
//! [`vis_payload`](prism_render_architecture::virtual_geometry::vis_payload)).
//!
//! The software rasterizer composites a 64-bit visibility word through a
//! 64-bit `atomicMax`: the reversed-Z depth key in the high `32` bits (so the
//! nearest surface wins the max) and the payload in the low `32` bits (carried
//! along for free). The CPU golden owns that bit layout; [`GpuVisWordCodec`]
//! is the on-device twin that runs one thread per pair, packs it, then unpacks
//! both fields back out, so the 64-bit word ABI a `GPU`-driven rasterizer must
//! emit is validated against the reference in isolation.
//!
//! This is the 64-bit sibling of the portable 32-bit
//! [`GpuVisPayloadCodec`](crate::GpuVisPayloadCodec). It closes a real
//! validation gap: the payload-raster twin checks the *atomic composite* of
//! that word, but only within a one-`ULP` depth tolerance, and never runs the
//! pack/unpack on device in isolation. This kernel runs the pure 64-bit
//! shift-and-mask codec per thread and is asserted bit-exact against the
//! reference, so the packed word and both unpacked fields are pinned down on
//! their own.
//!
//! # Portability
//!
//! The kernel uses the `SHADER_INT64` `u64` type (no atomics), so - like the
//! payload raster - [`new`](GpuVisWordCodec::new) returns [`None`] unless
//! [`GpuContext::supports_u64_atomics`] is `true`, and callers skip gracefully
//! where 64-bit integers are unavailable.
//!
//! # Correctness model
//!
//! 64-bit shift, OR and AND are defined identically on every backend that
//! supports them, so the packed word and both unpacked fields are bit-exact
//! against the reference and are asserted index-for-index with no tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Prism's own vis-buffer word bit layout plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

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

/// One `(depth_key, payload)` pair to encode into a 64-bit vis-buffer word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisWordInput {
    /// Reversed-Z depth key, stored in the word's high `32` bits.
    pub depth_key: u32,
    /// Payload, stored in the word's low `32` bits.
    pub payload: u32,
}

/// One codec result: the packed 64-bit word plus both fields unpacked back
/// out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisWordCodec {
    /// The packed 64-bit visibility word.
    pub packed: u64,
    /// `vis_depth(packed)` - the word's high `32` bits.
    pub depth: u32,
    /// `vis_payload(packed)` - the word's low `32` bits.
    pub payload: u32,
}

/// Uniform parameters for one codec dispatch. Layout matches `Params` in
/// `shaders/vis_word_codec.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One input upload. `8`-byte stride matching `PackInput` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPackInput {
    depth_key: u32,
    payload: u32,
}

/// One output word. `16`-byte stride matching `CodecOutput` in the shader: a
/// `u64` (`8`-byte aligned) plus two `u32` fields, `8 + 4 + 4 = 16`, no pad.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCodecOutput {
    packed: u64,
    depth: u32,
    payload: u32,
}

/// A compiled, reusable 64-bit vis-buffer-word-codec pipeline.
pub struct GpuVisWordCodec {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVisWordCodec {
    /// Compiles the 64-bit word-codec kernel on `ctx`.
    ///
    /// Returns [`None`] when `ctx` was not created with the 64-bit integer
    /// features (see [`GpuContext::supports_u64_atomics`]); the `u64` kernel
    /// would fail to compile on such a device, so callers skip instead.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Option<GpuVisWordCodec> {
        if !ctx.supports_u64_atomics() {
            return None;
        }
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_vis_word_codec"),
            source: ShaderSource::Wgsl(include_str!("../shaders/vis_word_codec.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_vis_word_codec_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_vis_word_codec_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_vis_word_codec_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("codec"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Some(GpuVisWordCodec {
            module,
            layout,
            pipeline,
        })
    }

    /// Encodes each pair in `inputs`, returning one [`VisWordCodec`] per input
    /// in order: the packed 64-bit word plus both fields unpacked back out.
    ///
    /// Each returned value equals
    /// [`pack_vis`](prism_render_architecture::virtual_geometry::pack_vis),
    /// [`vis_depth`](prism_render_architecture::virtual_geometry::vis_depth)
    /// and
    /// [`vis_payload`](prism_render_architecture::virtual_geometry::vis_payload)
    /// evaluated on the same pair. An empty `inputs` slice yields an empty
    /// result - storage buffers cannot be zero-sized, so it is handled by an
    /// early return.
    #[must_use]
    pub fn encode(&self, ctx: &GpuContext, inputs: &[VisWordInput]) -> Vec<VisWordCodec> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: u32::try_from(inputs.len())
                .expect("input count must fit in u32 for the GPU dispatch"),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let gpu_inputs: Vec<GpuPackInput> = inputs
            .iter()
            .map(|p| GpuPackInput {
                depth_key: p.depth_key,
                payload: p.payload,
            })
            .collect();

        let out_bytes = (inputs.len() as u64) * (size_of::<GpuCodecOutput>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_word_codec_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_word_codec_inputs"),
            contents: bytemuck::cast_slice(&gpu_inputs),
            usage: BufferUsages::STORAGE,
        });
        let outputs_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_word_codec_outputs"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let outputs_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_word_codec_outputs_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_vis_word_codec_bind_group"),
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
                    resource: outputs_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_vis_word_codec_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_vis_word_codec_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&outputs_buf, 0, &outputs_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        outputs_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = outputs_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_outputs = bytemuck::cast_slice::<u8, GpuCodecOutput>(&view).to_vec();
        drop(view);
        outputs_stage.unmap();
        debug_assert_eq!(gpu_outputs.len(), inputs.len());
        gpu_outputs
            .into_iter()
            .map(|o| VisWordCodec {
                packed: o.packed,
                depth: o.depth,
                payload: o.payload,
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
