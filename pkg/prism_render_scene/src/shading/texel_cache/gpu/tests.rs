//! Real-device parity for the texel-cache GPU page-write ABI.
//!
//! This is intentionally a narrow GPU integration test. It proves that the
//! shipping page-write kernel can consume job/page-table storage buffers and
//! write a storage atlas without cross-page overlap, without out-of-range
//! invocation writes, and only when generation/epoch metadata agrees. The
//! atlas payload is an ABI/debug identity payload; this is not a material
//! shading result and does not claim complete P3-a shade-cache integration.

use bevy_platform::future::block_on;
use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    DeviceDescriptor, Extent3d, Instance, InstanceDescriptor, InstanceFlags, MapMode,
    PipelineLayoutDescriptor, PollType, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StorageTextureAccess, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureFormat, TextureUsages,
    TextureViewDescriptor, TextureViewDimension,
};

const ATLAS_WIDTH: u32 = 12;
const ATLAS_HEIGHT: u32 = 4;
const PAGE_WIDTH: u32 = 4;
const PAGE_HEIGHT: u32 = 4;
const ROW_PITCH: u32 = 256;
const SENTINEL: u32 = 0xdead_beef;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Job {
    width: u32,
    height: u32,
    atlas_x: u32,
    atlas_y: u32,
    page_index: u32,
    generation: u32,
    epoch: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PageEntry {
    atlas_x: u32,
    atlas_y: u32,
    generation: u32,
    epoch: u32,
}

fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = match block_on(instance.request_adapter(&RequestAdapterOptions::default())) {
        Ok(adapter) => adapter,
        Err(error) => {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU set: adapter request failed: {error}"
            );
            return None;
        }
    };
    match block_on(adapter.request_device(&DeviceDescriptor::default())) {
        Ok(device_queue) => Some(device_queue),
        Err(error) => {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU set: device request failed: {error}"
            );
            None
        }
    }
}

fn readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u32> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(PollType::wait_indefinitely())
        .expect("GPU readback poll should complete");
    receiver
        .recv()
        .expect("GPU readback callback should run")
        .expect("GPU readback mapping should succeed");
    let mapped = slice
        .get_mapped_range()
        .expect("mapped readback range should be available");
    let words = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    words
}

fn atlas_word(words: &[u32], x: u32, y: u32, channel: usize) -> u32 {
    let row_words = (ROW_PITCH / 4) as usize;
    words[y as usize * row_words + x as usize * 4 + channel]
}

#[test]
fn gpu_page_write_preserves_atlas_boundaries_and_page_table_proof() {
    let Some((device, queue)) = try_device() else {
        eprintln!(
            "gpu_page_write_preserves_atlas_boundaries_and_page_table_proof: no usable wgpu adapter, skipping"
        );
        return;
    };

    let jobs = [
        // The first page deliberately has width 3: the fourth x invocation is
        // dispatched but must not write the atlas sentinel.
        Job {
            width: 3,
            height: PAGE_HEIGHT,
            atlas_x: 0,
            atlas_y: 0,
            page_index: 0,
            generation: 7,
            epoch: 11,
            padding: 0,
        },
        Job {
            width: PAGE_WIDTH,
            height: PAGE_HEIGHT,
            atlas_x: 4,
            atlas_y: 0,
            page_index: 1,
            generation: 13,
            epoch: 17,
            padding: 0,
        },
        // This page is spatially valid but intentionally stale. The shader
        // must leave its entire region untouched when the proof disagrees.
        Job {
            width: PAGE_WIDTH,
            height: PAGE_HEIGHT,
            atlas_x: 8,
            atlas_y: 0,
            page_index: 2,
            generation: 19,
            epoch: 23,
            padding: 0,
        },
        // This page has matching metadata but extends past the right edge of
        // the atlas. Only the in-bounds portion may be written.
        Job {
            width: PAGE_WIDTH,
            height: PAGE_HEIGHT,
            atlas_x: 10,
            atlas_y: 0,
            page_index: 3,
            generation: 29,
            epoch: 31,
            padding: 0,
        },
    ];
    let page_table = [
        PageEntry {
            atlas_x: 0,
            atlas_y: 0,
            generation: 7,
            epoch: 11,
        },
        PageEntry {
            atlas_x: 4,
            atlas_y: 0,
            generation: 13,
            epoch: 17,
        },
        PageEntry {
            atlas_x: 8,
            atlas_y: 0,
            generation: 18,
            epoch: 23,
        },
        PageEntry {
            atlas_x: 10,
            atlas_y: 0,
            generation: 29,
            epoch: 31,
        },
    ];

    let jobs_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("texel-cache GPU jobs"),
        contents: bytemuck::cast_slice(&jobs),
        usage: BufferUsages::STORAGE,
    });
    let page_table_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("texel-cache GPU page table"),
        contents: bytemuck::cast_slice(&page_table),
        usage: BufferUsages::STORAGE,
    });
    let atlas = device.create_texture(&TextureDescriptor {
        label: Some("texel-cache GPU storage atlas"),
        size: Extent3d {
            width: ATLAS_WIDTH,
            height: ATLAS_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba32Uint,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let atlas_view = atlas.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2),
        ..Default::default()
    });

    let sentinel_row = vec![SENTINEL; (ROW_PITCH / 4) as usize];
    let sentinel_data = sentinel_row.repeat(ATLAS_HEIGHT as usize);
    let readback_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("texel-cache atlas readback"),
        size: u64::from(ROW_PITCH * ATLAS_HEIGHT),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("texel-cache GPU page-write shader"),
        source: ShaderSource::Wgsl(include_str!("atlas.wgsl").into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("texel-cache GPU page-write bindings"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Uint,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("texel-cache GPU page-write pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("texel-cache GPU page-write pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("shade_pages"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("texel-cache GPU page-write bind group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&atlas_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: jobs_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: page_table_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("texel-cache GPU parity encoder"),
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &atlas,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(&sentinel_data),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(ROW_PITCH),
            rows_per_image: Some(ATLAS_HEIGHT),
        },
        Extent3d {
            width: ATLAS_WIDTH,
            height: ATLAS_HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("texel-cache GPU page-write dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(jobs.len() as u32, 1, 1);
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &atlas,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW_PITCH),
                rows_per_image: Some(ATLAS_HEIGHT),
            },
        },
        Extent3d {
            width: ATLAS_WIDTH,
            height: ATLAS_HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let words = readback(&device, &readback_buffer);
    for y in 0..ATLAS_HEIGHT {
        for x in 0..ATLAS_WIDTH {
            let expected = if x < 3 {
                [0, x, 7, 11]
            } else if (4..8).contains(&x) {
                [1, x - 4, 13, 17]
            } else if x < 10 {
                [SENTINEL; 4]
            } else if x < ATLAS_WIDTH {
                [3, x - 10, 29, 31]
            } else {
                unreachable!()
            };
            for (channel, &value) in expected.iter().enumerate() {
                assert_eq!(
                    atlas_word(&words, x, y, channel),
                    value,
                    "atlas mismatch at ({x}, {y}), channel {channel}"
                );
            }
        }
    }
}
