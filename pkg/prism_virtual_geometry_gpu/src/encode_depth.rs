//! `wgpu` compute twin of the virtual-geometry vis-buffer depth-key encode
//! ([`encode_depth`](prism_render_architecture::virtual_geometry::encode_depth)).
//!
//! The vis-buffer resolves the nearest surface with an `atomicMax` over a u32
//! compositing key derived from reversed-Z NDC depth. The CPU golden
//! [`encode_depth`](prism_render_architecture::virtual_geometry::encode_depth)
//! owns that contract - `bitcast<u32>(clamp(depth, 0, 1))` - and
//! [`GpuEncodeDepth`] is the on-device twin that runs one thread per depth and
//! returns the same key. The vis-buffer raster twins already encode depth
//! inline for the depths their triangles happen to produce; isolating the
//! encode here diffs the clamp + bitcast contract to the bit across the full
//! finite input domain the rasterizer can feed.
//!
//! # Portability
//!
//! The kernel is one `clamp` plus one `bitcast` in the portable core-`WGSL`
//! subset, so it needs no optional device feature and runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Correctness model
//!
//! The encode is bit-exact. `clamp` is comparison + select (no arithmetic, no
//! rounding) and `bitcast` copies the bit pattern, so a finite non-negative
//! input returns unchanged, a strictly negative input clamps to `+0.0`, and a
//! value above `1.0` (including `+inf`) clamps to `1.0` - identical to the
//! sequential `CPU` golden, asserted with zero tolerance.
//!
//! `NaN`, `-0.0` and subnormals are deliberately outside the parity domain:
//! `WGSL` `min`/`max` `NaN` handling and the `max(-0.0, +0.0)` result sign are
//! backend-indeterminate, and GPUs may flush subnormals to zero, whereas the
//! rasterizer only ever feeds finite, normal, non-negative-zero depths, so
//! honesty requires the twin claim bit-exactness only on that domain.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard reversed-Z depth-key encode for `atomicMax` vis-buffer
//! compositing plus `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

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

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/encode_depth.wesl`: the depth count then three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable depth-key-encode pipeline.
pub struct GpuEncodeDepth {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuEncodeDepth {
    /// Compiles the depth-key-encode kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so it needs no
    /// optional device feature.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_encode_depth"),
            source: ShaderSource::Wgsl(include_str!("../shaders/encode_depth.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_encode_depth_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: true },
                ),
                buffer_entry(
                    2,
                    BufferBindingType::Storage { read_only: false },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_encode_depth_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_encode_depth_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("encode"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEncodeDepth {
            module,
            layout,
            pipeline,
        }
    }

    /// Encodes each reversed-Z depth into its `atomicMax` compositing key
    /// on-device, returning one u32 per depth in input order.
    ///
    /// Each returned key equals
    /// [`encode_depth`](prism_render_architecture::virtual_geometry::encode_depth)`(depth)`
    /// for the input depth. An empty `depths` slice yields an empty result -
    /// storage buffers cannot be zero-sized, so it is handled by an early
    /// return.
    #[must_use]
    pub fn encode(&self, ctx: &GpuContext, depths: &[f32]) -> Vec<u32> {
        if depths.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: u32::try_from(depths.len())
                .expect("depth count must fit in u32 for the GPU dispatch"),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (depths.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_encode_depth_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_encode_depth_inputs"),
            contents: bytemuck::cast_slice(depths),
            usage: BufferUsages::STORAGE,
        });
        let keys_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_encode_depth_keys"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let keys_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_encode_depth_keys_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_encode_depth_bind_group"),
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
                    resource: keys_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_encode_depth_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_encode_depth_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&keys_buf, 0, &keys_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        keys_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = keys_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_keys = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        keys_stage.unmap();
        debug_assert_eq!(gpu_keys.len(), depths.len());
        gpu_keys
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
