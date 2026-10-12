//! Real-device parity for the material-VT page-table word ABI.
//!
//! The test uploads the exact words emitted by
//! [`MaterialVirtualTexturePageTableSnapshot`] to a real storage buffer and
//! performs the shader-side binary search. It deliberately tests two hits and
//! a miss so a table that is merely well-formed on the CPU cannot pass with a
//! wrong packing or ordering contract.

use bevy_platform::future::block_on;
use bytemuck::{Pod, Zeroable};
use core::mem::size_of;
use prism_render_architecture::texture_streaming::TexturePageKey;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor, Instance,
    InstanceDescriptor, InstanceFlags, MapMode, PipelineLayoutDescriptor, PollType,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use super::page_table::{
    MaterialVirtualTexturePageRecord, MaterialVirtualTexturePageTableSnapshot, MATERIAL_VT_MISS,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Query {
    texture: u32,
    mip_layer: u32,
    page_xy: u32,
    padding: u32,
}

const SHADER: &str = r#"
struct Query {
    texture: u32,
    mip_layer: u32,
    page_xy: u32,
    padding: u32,
};

@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn compare(e0: u32, e1: u32, e2: u32, t0: u32, t1: u32, t2: u32) -> i32 {
    if e0 < t0 { return -1; }
    if e0 > t0 { return 1; }
    if e1 < t1 { return -1; }
    if e1 > t1 { return 1; }
    if e2 < t2 { return -1; }
    if e2 > t2 { return 1; }
    return 0;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&queries) { return; }
    let query = queries[id.x];
    let entry_count = arrayLength(&table) / 4u;
    var lo = 0u;
    var hi = entry_count;
    var result = 0xffffffffu;
    loop {
        if lo >= hi { break; }
        let mid = lo + (hi - lo) / 2u;
        let base = mid * 4u;
        let order = compare(
            table[base], table[base + 1u], table[base + 2u],
            query.texture, query.mip_layer, query.page_xy,
        );
        if order < 0 {
            lo = mid + 1u;
        } else if order > 0 {
            hi = mid;
        } else {
            result = table[base + 3u];
            break;
        }
    }
    output[id.x] = result;
}
"#;

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
        .expect("GPU page-table readback poll should complete");
    receiver
        .recv()
        .expect("GPU page-table readback callback should run")
        .expect("GPU page-table readback mapping should succeed");
    let mapped = slice
        .get_mapped_range()
        .expect("GPU page-table readback mapping should be available");
    let result = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    result
}

fn key(texture: u32, x: u16) -> TexturePageKey {
    TexturePageKey {
        texture,
        mip: 2,
        layer: 3,
        x,
        y: 5,
    }
}

#[test]
fn gpu_material_vt_page_table_words_match_shader_binary_search() {
    let Some((device, queue)) = try_device() else {
        eprintln!(
            "gpu_material_vt_page_table_words_match_shader_binary_search: no usable wgpu adapter, skipping"
        );
        return;
    };

    let snapshot = MaterialVirtualTexturePageTableSnapshot::try_from_records(
        4,
        9,
        [
            MaterialVirtualTexturePageRecord {
                key: key(7, 0),
                slot: 13,
                generation: 9,
                epoch: 4,
                ready: true,
            },
            MaterialVirtualTexturePageRecord {
                key: key(7, 1),
                slot: 29,
                generation: 9,
                epoch: 4,
                ready: true,
            },
            MaterialVirtualTexturePageRecord {
                key: key(8, 0),
                slot: 41,
                generation: 9,
                epoch: 4,
                ready: true,
            },
        ],
    )
    .expect("valid material-VT snapshot");

    let queries = [
        Query {
            texture: 7,
            mip_layer: (2 << 24) | (3 << 8),
            page_xy: (1 << 16) | 5,
            padding: 0,
        },
        Query {
            texture: 8,
            mip_layer: (2 << 24) | (3 << 8),
            page_xy: 5,
            padding: 0,
        },
        Query {
            texture: 7,
            mip_layer: (2 << 24) | (3 << 8),
            page_xy: (2 << 16) | 5,
            padding: 0,
        },
    ];

    let table_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("material-VT page-table words"),
        contents: bytemuck::cast_slice(snapshot.gpu_words()),
        usage: BufferUsages::STORAGE,
    });
    let queries_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("material-VT page-table queries"),
        contents: bytemuck::cast_slice(&queries),
        usage: BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("material-VT page-table output"),
        size: (queries.len() * size_of::<u32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("material-VT page-table readback"),
        size: (queries.len() * size_of::<u32>()) as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("material-VT page-table binary search"),
        source: ShaderSource::Wgsl(SHADER.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("material-VT page-table test layout"),
        entries: &[
            storage_entry(0, false),
            storage_entry(1, false),
            storage_entry(2, true),
        ],
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("material-VT page-table binary search pipeline"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("material-VT page-table test pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        })),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("material-VT page-table test bind group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: table_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: queries_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("material-VT page-table test encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("material-VT page-table binary search pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(queries.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &readback_buffer,
        0,
        (queries.len() * size_of::<u32>()) as u64,
    );
    queue.submit(Some(encoder.finish()));

    assert_eq!(
        readback(&device, &readback_buffer),
        [29, 41, MATERIAL_VT_MISS]
    );
}

fn storage_entry(binding: u32, writable: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage {
                read_only: !writable,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
