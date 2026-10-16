//! Physical-GPU regressions for the RHI's unsigned four-channel ID format.
//! Set PRISM_REQUIRE_GPU=1 to fail instead of skipping without hardware.
//! These tests do not establish Scene production integration.

use prism_render_driver as rhi;
use prism_render_driver_wgpu::{
    require_hardware_gpu, HardwareDeviceError, WgpuDevice, WgpuInstance, WgpuQueue,
};
use rhi::{RenderDevice as _, RenderQueue as _};

fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
    match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => {
            eprintln!(
                "Rgba32Uint REAL HARDWARE GPU: {:?}",
                pair.0.wgpu_adapter().get_info()
            );
            Some(pair)
        }
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no adapter"
            );
            eprintln!("SKIPPED Rgba32Uint GPU test: no adapter");
            None
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires physical hardware: {evidence}"
            );
            eprintln!("SKIPPED Rgba32Uint physical-GPU evidence: {evidence}");
            None
        }
        Err(HardwareDeviceError::Device(error)) => {
            panic!("physical adapter failed to create a device: {error}");
        }
    }
}

fn readback_buffer(device: &WgpuDevice, size: u64) -> rhi::BufferId {
    device.create_buffer(&rhi::BufferDescriptor::new(
        size,
        rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
    ))
}

fn texture_copy(
    texture: rhi::TextureId,
    buffer: rhi::BufferId,
    size: rhi::Extent3d,
    row_stride: u32,
) -> rhi::CopyCommand {
    rhi::CopyCommand::CopyTextureToBuffer {
        src: texture,
        src_mip_level: 0,
        src_origin: [0; 3],
        dst: buffer,
        dst_offset: 0,
        bytes_per_row: Some(row_stride),
        rows_per_image: Some(size.height),
        size,
    }
}

fn assert_packed_rows(actual: &[u8], expected: &[u8], size: rhi::Extent3d, row_stride: u32) {
    let row_bytes = size.width * rhi::TextureFormat::Rgba32Uint.bytes_per_texel();
    let rows = size.height * size.depth_or_array_layers;
    assert_eq!(actual.len(), (rows * row_stride) as usize);
    assert_eq!(expected.len(), (rows * row_bytes) as usize);
    for row in 0..rows {
        let actual_start = (row * row_stride) as usize;
        let expected_start = (row * row_bytes) as usize;
        assert_eq!(
            &actual[actual_start..actual_start + row_bytes as usize],
            &expected[expected_start..expected_start + row_bytes as usize],
            "all four u32 channels must survive exactly in row {row}"
        );
    }
}

#[test]
fn rgba32uint_upload_preserves_packed_rows_layers_and_all_u32_bits() {
    let Some((device, queue)) = device_or_skip() else {
        return;
    };
    let size = rhi::Extent3d {
        width: 3,
        height: 2,
        depth_or_array_layers: 2,
    };
    let format = rhi::TextureFormat::Rgba32Uint;
    let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        size,
        format,
        rhi::TextureUsages::COPY_DST | rhi::TextureUsages::COPY_SRC,
    ));
    assert_eq!(
        device.wgpu_texture(texture).format(),
        wgpu::TextureFormat::Rgba32Uint
    );
    let view = device.create_texture_view(
        texture,
        &rhi::TextureViewDescriptor {
            format: Some(format),
            dimension: rhi::TextureViewDimension::D2Array,
            ..Default::default()
        },
    );
    let packed: Vec<u8> = (0..12u32)
        .flat_map(|pixel| {
            [
                0x0100_0001 + pixel,
                0x8000_0000 + pixel,
                u32::MAX - pixel,
                pixel,
            ]
        })
        .flat_map(u32::to_le_bytes)
        .collect();
    let row_bytes = size.width * format.bytes_per_texel();
    assert_eq!(row_bytes, 48);
    assert_eq!(packed.len(), 192);
    queue.write_texture(
        rhi::TextureWrite {
            texture,
            mip_level: 0,
            origin: [0; 3],
            size,
        },
        &packed,
        rhi::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(row_bytes),
            rows_per_image: Some(size.height),
        },
    );

    let row_stride = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer_size = u64::from(row_stride * size.height * size.depth_or_array_layers);
    assert_eq!(buffer_size, 1024);
    let buffer = readback_buffer(&device, buffer_size);
    let mut encoder = rhi::CommandEncoder::labeled("rgba32uint-layer-copy");
    encoder.push_transfer_pass(None, vec![texture_copy(texture, buffer, size, row_stride)]);
    // Use the same native encode path that a host-owned submission consumes.
    queue.wgpu_queue().submit([queue.encode(&encoder.finish())]);
    let actual = queue.read_buffer(buffer, buffer_size);
    assert_packed_rows(&actual, &packed, size, row_stride);

    device.destroy_buffer(buffer);
    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}

const INTEGER_ATTACHMENTS: &str = r#"
@vertex fn vertex_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[index], 0.0, 1.0);
}

struct IntegerAttachments {
    @location(0) ids: vec4<u32>,
    @location(1) metadata: vec4<u32>,
};

@fragment fn fragment_main(@builtin(position) position: vec4<f32>) -> IntegerAttachments {
    let x = u32(position.x);
    let y = u32(position.y);
    var output: IntegerAttachments;
    output.ids = vec4<u32>(0x01000001u + x, 0x80000000u + y, 0xffffffffu, 0u);
    output.metadata = vec4<u32>(0xdeadbeefu, 0x7fffffffu, 42u, x + 3u * y);
    return output;
}
"#;

#[test]
fn rgba32uint_two_raster_attachments_preserve_unsigned_shader_outputs() {
    let Some((device, queue)) = device_or_skip() else {
        return;
    };
    let size = rhi::Extent3d::new_2d(3, 2);
    let format = rhi::TextureFormat::Rgba32Uint;
    let textures = [(); 2].map(|()| {
        device.create_texture(&rhi::TextureDescriptor::new_2d(
            size,
            format,
            rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
        ))
    });
    for texture in textures {
        assert_eq!(
            device.wgpu_texture(texture).format(),
            wgpu::TextureFormat::Rgba32Uint
        );
    }
    let views = textures.map(|texture| {
        device.create_texture_view(
            texture,
            &rhi::TextureViewDescriptor {
                format: Some(format),
                ..Default::default()
            },
        )
    });
    let module =
        device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(INTEGER_ATTACHMENTS));
    let pipeline = device.create_render_pipeline(&rhi::RenderPipelineDescriptor {
        label: Some("rgba32uint-two-integer-targets".into()),
        layout: None,
        vertex: rhi::VertexState {
            module,
            entry_point: "vertex_main".into(),
            buffers: Vec::new(),
        },
        primitive: rhi::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: rhi::MultisampleState::default(),
        fragment: Some(rhi::FragmentState {
            module,
            entry_point: "fragment_main".into(),
            targets: vec![Some(rhi::ColorTargetState::opaque(format)); 2],
        }),
    });
    let row_stride = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer_size = u64::from(row_stride * size.height);
    let buffers = [(); 2].map(|()| readback_buffer(&device, buffer_size));
    let mut encoder = rhi::CommandEncoder::labeled("rgba32uint-integer-raster");
    encoder.push_render_pass(
        rhi::RenderPassDescriptor {
            label: None,
            color_attachments: views
                .map(|view| {
                    Some(rhi::ColorAttachment {
                        view,
                        resolve_target: None,
                        load: rhi::LoadOp::Clear(rhi::Color::BLACK),
                        store: rhi::StoreOp::Store,
                    })
                })
                .to_vec(),
            depth_stencil_attachment: None,
        },
        vec![
            rhi::RenderCommand::SetPipeline(pipeline),
            rhi::RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ],
    );
    encoder.push_transfer_pass(
        None,
        textures
            .into_iter()
            .zip(buffers)
            .map(|(texture, buffer)| texture_copy(texture, buffer, size, row_stride))
            .collect(),
    );
    queue.wgpu_queue().submit([queue.encode(&encoder.finish())]);
    let mut expected_ids = Vec::new();
    let mut expected_metadata = Vec::new();
    for y in 0..size.height {
        for x in 0..size.width {
            for value in [0x0100_0001 + x, 0x8000_0000 + y, u32::MAX, 0] {
                expected_ids.extend_from_slice(&value.to_le_bytes());
            }
            for value in [0xdead_beef, 0x7fff_ffff, 42, x + 3 * y] {
                expected_metadata.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    for (buffer, expected) in buffers.into_iter().zip([expected_ids, expected_metadata]) {
        assert_packed_rows(
            &queue.read_buffer(buffer, buffer_size),
            &expected,
            size,
            row_stride,
        );
        device.destroy_buffer(buffer);
    }
    device.destroy_render_pipeline(pipeline);
    device.destroy_shader_module(module);
    for view in views {
        device.destroy_texture_view(view);
    }
    for texture in textures {
        device.destroy_texture(texture);
    }
}
