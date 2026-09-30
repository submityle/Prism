//! `wgpu` compute twin of the virtual-geometry vis-buffer payload codec
//! ([`pack_cluster_triangle`](prism_render_architecture::virtual_geometry::pack_cluster_triangle),
//! [`cluster_of`](prism_render_architecture::virtual_geometry::cluster_of),
//! [`triangle_of`](prism_render_architecture::virtual_geometry::triangle_of)).
//!
//! The software rasterizer records, beside each depth key, a 32-bit payload
//! word that packs the cluster id in the high bits and the triangle index
//! within the cluster in the low
//! [`CLUSTER_TRIANGLE_BITS`](prism_render_architecture::virtual_geometry::CLUSTER_TRIANGLE_BITS)
//! bits. The CPU golden owns that bit layout; [`GpuVisPayloadCodec`] is the
//! on-device twin that runs one thread per pair, packs it, then unpacks both
//! fields back out, so the visibility-payload ABI a `GPU`-driven rasterizer
//! must emit is validated against the reference in isolation rather than only
//! as a sub-step buried inside the full raster kernel.
//!
//! # Portability
//!
//! The kernel composites only the 32-bit payload word through integer shifts
//! and masks, in the portable core-`WGSL` subset, so it needs no optional
//! device feature and runs unmodified on Metal, Vulkan and DX12. The 64-bit
//! `(depth << 32) | payload` visibility key is out of scope here - it needs
//! `SHADER_INT64` and is covered by the payload-raster twin.
//!
//! # Correctness model
//!
//! Integer shift, OR and AND are defined identically on every backend, so the
//! packed word and both unpacked fields are bit-exact against the reference and
//! are asserted index-for-index with no tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: Prism's own vis-buffer payload bit layout plus `wgpu` compute
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

/// One `(cluster_id, triangle_id)` pair to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadInput {
    /// Cluster id, stored in the payload's high bits.
    pub cluster_id: u32,
    /// Triangle index within the cluster, masked into the payload's low
    /// [`CLUSTER_TRIANGLE_BITS`](prism_render_architecture::virtual_geometry::CLUSTER_TRIANGLE_BITS)
    /// bits.
    pub triangle_id: u32,
}

/// One codec result: the packed payload word plus both fields unpacked back
/// out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadCodec {
    /// The packed 32-bit visibility payload.
    pub packed: u32,
    /// `cluster_of(packed)` - the payload's high bits.
    pub cluster: u32,
    /// `triangle_of(packed)` - the payload's low bits.
    pub triangle: u32,
}

/// Uniform parameters for one codec dispatch. Layout matches `Params` in
/// `shaders/vis_payload_codec.wesl`.
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
    cluster_id: u32,
    triangle_id: u32,
}

/// One output word. `16`-byte stride matching `CodecOutput` in the shader
/// (three `u32` fields plus one pad word).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCodecOutput {
    packed: u32,
    cluster: u32,
    triangle: u32,
    pad0: u32,
}

/// A compiled, reusable vis-buffer-payload-codec pipeline.
pub struct GpuVisPayloadCodec {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVisPayloadCodec {
    /// Compiles the payload-codec kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so it needs no
    /// optional device feature.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_vis_payload_codec"),
            source: ShaderSource::Wgsl(include_str!("../shaders/vis_payload_codec.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_vis_payload_codec_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_vis_payload_codec_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_vis_payload_codec_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("codec"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVisPayloadCodec {
            module,
            layout,
            pipeline,
        }
    }

    /// Encodes each pair in `inputs`, returning one [`PayloadCodec`] per input
    /// in order: the packed payload plus both fields unpacked back out.
    ///
    /// Each returned value equals
    /// [`pack_cluster_triangle`](prism_render_architecture::virtual_geometry::pack_cluster_triangle),
    /// [`cluster_of`](prism_render_architecture::virtual_geometry::cluster_of)
    /// and
    /// [`triangle_of`](prism_render_architecture::virtual_geometry::triangle_of)
    /// evaluated on the same pair. An empty `inputs` slice yields an empty
    /// result - storage buffers cannot be zero-sized, so it is handled by an
    /// early return.
    #[must_use]
    pub fn encode(&self, ctx: &GpuContext, inputs: &[PayloadInput]) -> Vec<PayloadCodec> {
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
                cluster_id: p.cluster_id,
                triangle_id: p.triangle_id,
            })
            .collect();

        let out_bytes = (inputs.len() as u64) * (size_of::<GpuCodecOutput>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_payload_codec_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_vis_payload_codec_inputs"),
            contents: bytemuck::cast_slice(&gpu_inputs),
            usage: BufferUsages::STORAGE,
        });
        let outputs_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_payload_codec_outputs"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let outputs_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_vis_payload_codec_outputs_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_vis_payload_codec_bind_group"),
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
            label: Some("prism_vis_payload_codec_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_vis_payload_codec_pass"),
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
            .map(|o| PayloadCodec {
                packed: o.packed,
                cluster: o.cluster,
                triangle: o.triangle,
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
