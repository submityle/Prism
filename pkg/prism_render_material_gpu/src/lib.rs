//! **GPU hardware-decode oracle for block-compressed texture codecs.**
//!
//! The pure-CPU block decoders in
//! [`prism_render_material::texture_codec`] turn BC/ETC2 blocks into texels
//! with hand-written integer math. Some modes (BC7 single-subset 4/5/6, BC6H
//! mode 11) are already implemented; the partitioned/delta modes are not,
//! precisely because hand-transcribing their bit layout from the spec without a
//! ground truth risks a silent wrong decode. This crate removes that risk: it
//! hands a raw compressed block to the GPU's *native* texture-decompression
//! unit via a WGSL `textureLoad` and reads the decoded texels back, giving a
//! hardware ground truth to validate every CPU decoder against.
//!
//! The flow is: upload the 8/16-byte block into a 4x4 texture of the matching
//! compressed [`wgpu::TextureFormat`], dispatch a 4x4 compute grid that
//! `textureLoad`s each texel into a storage buffer, and read it back. Unorm
//! formats come back as `[[u8; 4]; 16]` (row-major, texel `t = y*4 + x`); the
//! BC6H HDR formats come back as `[[f32; 3]; 16]`.
//!
//! On a host without a usable adapter (or one lacking the needed
//! `TEXTURE_COMPRESSION_*` feature) [`BlockOracle::try_new`] returns `None` so
//! callers skip gracefully, exactly as the other `prism_*_gpu` twins do.
//!
//! This crate contains no Unreal Engine source or derived code: it only drives
//! the platform's own standardized hardware decoder.

use futures_lite::future::block_on;
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor, InstanceFlags, MapMode,
    Origin3d, PipelineLayoutDescriptor, PollType, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureViewDescriptor, TextureViewDimension,
};

const SHADER: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>, 16>;

@compute @workgroup_size(4, 4, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < 4u && gid.y < 4u) {
        let texel = textureLoad(src, vec2<i32>(i32(gid.x), i32(gid.y)), 0);
        dst[gid.y * 4u + gid.x] = texel;
    }
}
"#;

/// A reusable GPU block-decode oracle bound to one device/queue/pipeline.
pub struct BlockOracle {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// The texture-compression feature families the adapter actually exposes.
    features: Features,
}

impl BlockOracle {
    /// Create the oracle, or return `None` when no adapter with block-decode
    /// support is reachable (so tests skip rather than fail).
    #[must_use]
    pub fn try_new() -> Option<BlockOracle> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
            flags: InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
            backend_options: BackendOptions::default(),
        });
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;

        // Request whichever compression families this adapter supports; without
        // at least BC there is nothing to validate here.
        let avail = adapter.features();
        let wanted = Features::TEXTURE_COMPRESSION_BC
            | Features::TEXTURE_COMPRESSION_ETC2
            | Features::TEXTURE_COMPRESSION_ASTC
            | Features::TEXTURE_COMPRESSION_ASTC_HDR;
        let features = avail & wanted;
        if !features.contains(Features::TEXTURE_COMPRESSION_BC) {
            return None;
        }

        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
            required_features: features,
            ..Default::default()
        }))
        .ok()?;

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_block_oracle"),
            source: ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_block_oracle_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_block_oracle_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_block_oracle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Some(BlockOracle {
            device,
            queue,
            pipeline,
            layout,
            features,
        })
    }

    /// The texture-compression feature families this oracle can decode.
    #[must_use]
    pub fn features(&self) -> Features {
        self.features
    }

    /// Decode one compressed `block` of the given `format` through the GPU and
    /// return the raw `vec4<f32>` texels (row-major, texel `t = y*4 + x`).
    ///
    /// `format` must be a 4x4 block-compressed format and `block` its single
    /// block (8 bytes for BC1/BC4, 16 bytes for the rest).
    #[must_use]
    pub fn decode_raw(&self, format: TextureFormat, block: &[u8]) -> [[f32; 4]; 16] {
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("prism_block_oracle_tex"),
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            block,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(block.len() as u32),
                rows_per_image: Some(1),
            },
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&TextureViewDescriptor::default());

        let out = self.device.create_buffer(&BufferDescriptor {
            label: Some("prism_block_oracle_out"),
            size: 256,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("prism_block_oracle_staging"),
            size: 256,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_block_oracle_bg"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, 256);
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        let _ = self.device.poll(PollType::wait_indefinitely());
        let data = slice
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");

        let mut texels = [[0.0f32; 4]; 16];
        for (t, texel) in texels.iter_mut().enumerate() {
            for (c, chan) in texel.iter_mut().enumerate() {
                let o = (t * 4 + c) * 4;
                *chan = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
            }
        }
        drop(data);
        staging.unmap();
        texels
    }

    /// Decode an LDR unorm block and quantise back to 8-bit `RGBA` texels,
    /// matching the `[[u8; 4]; 16]` convention of the CPU decoders.
    #[must_use]
    pub fn decode_unorm8(&self, format: TextureFormat, block: &[u8]) -> [[u8; 4]; 16] {
        let raw = self.decode_raw(format, block);
        let mut out = [[0u8; 4]; 16];
        for (t, texel) in out.iter_mut().enumerate() {
            for c in 0..4 {
                let v = (raw[t][c].clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                texel[c] = v.min(255) as u8;
            }
        }
        out
    }

    /// Decode an HDR BC6H block to `[[f32; 3]; 16]`, matching the CPU decoder.
    #[must_use]
    pub fn decode_rgb_f32(&self, format: TextureFormat, block: &[u8]) -> [[f32; 3]; 16] {
        let raw = self.decode_raw(format, block);
        let mut out = [[0.0f32; 3]; 16];
        for (t, texel) in out.iter_mut().enumerate() {
            texel.copy_from_slice(&raw[t][0..3]);
        }
        out
    }
}
