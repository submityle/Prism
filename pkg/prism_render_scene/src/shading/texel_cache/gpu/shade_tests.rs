//! Real-device parity for the texture-space irradiance writer.
//!
//! The test exercises the complete ABI path: canonical shading vertices,
//! [`super::shade::ShadeJob`], the indexed shade page table, and an
//! `rgba32float` storage atlas.  The sentinel checks make stale proofs,
//! triangle-exterior texels, and atlas clipping observable rather than merely
//! validating that a dispatch completed.

use bevy_platform::future::block_on;
use bytemuck::cast_slice;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    DeviceDescriptor, Extent3d, Instance, InstanceDescriptor, InstanceFlags, MapMode,
    PipelineLayoutDescriptor, PollType, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StorageTextureAccess, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

use crate::{
    RenderShadingGeometry, RenderShadingGeometryHeader, RenderShadingPrimitive, RenderShadingVertex,
};

use super::page_table::GpuShadePageEntry;
use super::shade::ShadeJob;
use crate::shading::texel_cache::build_face_chart;

const ATLAS_WIDTH: u32 = 8;
const ATLAS_HEIGHT: u32 = 4;
const ROW_PITCH: u32 = 256;
const SENTINEL: [f32; 4] = [-11.0, -13.0, -17.0, -19.0];

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

fn readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
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
    let values = cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}

fn atlas_pixel(values: &[f32], x: u32, y: u32) -> [f32; 4] {
    let row_values = (ROW_PITCH / 4) as usize;
    let offset = y as usize * row_values + x as usize * 4;
    values[offset..offset + 4]
        .try_into()
        .expect("rgba32float pixel must have four channels")
}

fn geometry() -> RenderShadingGeometry {
    let vertex = |uv: [f32; 2]| RenderShadingVertex {
        position: [uv[0], uv[1], 0.0],
        _position_padding: 0.0,
        normal: [0.0, 0.0, 1.0],
        _normal_padding: 0.0,
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv,
        flags: 0,
        _padding: 0,
    };
    RenderShadingGeometry {
        vertices: vec![vertex([0.0, 0.0]), vertex([1.0, 0.0]), vertex([0.0, 1.0])],
        primitives: vec![RenderShadingPrimitive {
            indices: [0, 1, 2],
            flags: 0,
        }],
        flags: 0,
    }
}

#[test]
fn gpu_shade_writes_irradiance_only_for_valid_texels_and_proofs() {
    let Some((device, queue)) = try_device() else {
        eprintln!(
            "gpu_shade_writes_irradiance_only_for_valid_texels_and_proofs: no usable wgpu adapter, skipping"
        );
        return;
    };

    let geometry = geometry();
    let chart = build_face_chart(&geometry, 0, 4.0, 4).expect("unit triangle chart");
    // The first three rows deliberately form the same UV triangle with
    // back-facing normals. If the shader treats local indices as global, the
    // valid page receives no irradiance; only adding the production header
    // offset reaches the intended packed rows.
    let mut decoy_rows = geometry.vertices.clone();
    for vertex in &mut decoy_rows {
        vertex.normal = [0.0, 0.0, -1.0];
    }
    let packed_vertices = decoy_rows
        .into_iter()
        .chain(geometry.vertices.iter().copied())
        .collect::<Vec<_>>();
    let packed_header = RenderShadingGeometryHeader {
        vertex_offset: 3,
        vertex_count: geometry.vertices.len() as u32,
        ..Default::default()
    };
    let valid = ShadeJob::from_face_with_header(
        &geometry,
        packed_header,
        chart,
        [0, 0],
        0,
        7,
        11,
        [0.0, 0.0, 1.0],
        [4.0, 2.0, 1.0],
    )
    .expect("valid shade job");
    let stale = ShadeJob::from_face_with_header(
        &geometry,
        packed_header,
        chart,
        [4, 0],
        1,
        13,
        17,
        [0.0, 0.0, 1.0],
        [4.0, 2.0, 1.0],
    )
    .expect("stale-proof shade job");
    let clipped = ShadeJob::from_face_with_header(
        &geometry,
        packed_header,
        chart,
        [6, 0],
        2,
        19,
        23,
        [0.0, 0.0, 1.0],
        [4.0, 2.0, 1.0],
    )
    .expect("clipped shade job");
    let jobs = [valid, stale, clipped];
    let page_table = [
        GpuShadePageEntry {
            atlas_x: 0,
            atlas_y: 0,
            generation: 7,
            epoch: 11,
        },
        // The job expects generation 13, so every texel in this page must
        // remain the sentinel.
        GpuShadePageEntry {
            atlas_x: 4,
            atlas_y: 0,
            generation: 12,
            epoch: 17,
        },
        GpuShadePageEntry {
            atlas_x: 6,
            atlas_y: 0,
            generation: 19,
            epoch: 23,
        },
    ];

    let jobs_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("shade-cache irradiance jobs"),
        contents: cast_slice(&jobs),
        usage: BufferUsages::STORAGE,
    });
    let page_table_words: Vec<u32> = page_table.iter().flat_map(|entry| entry.words()).collect();
    let page_table_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("shade-cache indexed page table"),
        contents: cast_slice(&page_table_words),
        usage: BufferUsages::STORAGE,
    });
    let vertices_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("shade-cache packed global shading vertices"),
        contents: cast_slice(&packed_vertices),
        usage: BufferUsages::STORAGE,
    });
    let atlas = device.create_texture(&TextureDescriptor {
        label: Some("shade-cache irradiance atlas"),
        size: Extent3d {
            width: ATLAS_WIDTH,
            height: ATLAS_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let atlas_view = atlas.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2),
        ..Default::default()
    });
    let sentinel_values = SENTINEL.repeat((ROW_PITCH / 16) as usize * ATLAS_HEIGHT as usize);
    let readback_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("shade-cache irradiance readback"),
        size: u64::from(ROW_PITCH * ATLAS_HEIGHT),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("shade-cache irradiance shader"),
        source: ShaderSource::Wgsl(include_str!("shade.wgsl").into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("shade-cache irradiance bindings"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
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
            BindGroupLayoutEntry {
                binding: 3,
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
        label: Some("shade-cache irradiance pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("shade-cache irradiance pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("shade_pages"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("shade-cache irradiance bind group"),
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
            BindGroupEntry {
                binding: 3,
                resource: vertices_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("shade-cache irradiance parity encoder"),
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &atlas,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        cast_slice(&sentinel_values),
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
            label: Some("shade-cache irradiance dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // x/y tile the chart in 8x8 workgroups; z selects one job.
        pass.dispatch_workgroups(1, 1, jobs.len() as u32);
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

    let values = readback(&device, &readback_buffer);
    for y in 0..ATLAS_HEIGHT {
        for x in 0..ATLAS_WIDTH {
            let expected = if x < 4 && x + y < 4 {
                [4.0, 2.0, 1.0, 1.0]
            } else if x >= 6 && x < 8 && (x - 6) + y < 4 {
                [4.0, 2.0, 1.0, 1.0]
            } else {
                SENTINEL
            };
            assert_eq!(
                atlas_pixel(&values, x, y),
                expected,
                "irradiance atlas mismatch at ({x}, {y})"
            );
        }
    }
}
