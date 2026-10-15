//! Real-device verification for the camera view ABI and RHI upload path.

use prism_camera::{GpuViewRecord, VIEW_SCHEMA_VERSION};
use prism_camera_gpu::{RhiViewBuffer, ViewBufferRing, WGSL_VIEW_RECORD_CONSUMER};
use prism_render_driver_wgpu::WgpuInstance;
use std::sync::mpsc;

fn gpu_or_skip() -> Option<(
    prism_render_driver_wgpu::WgpuDevice,
    prism_render_driver_wgpu::WgpuQueue,
)> {
    match WgpuInstance::new().request_default_device() {
        Ok(pair) => Some(pair),
        Err(error) => {
            if std::env::var_os("PRISM_REQUIRE_GPU").is_some() {
                panic!("required camera GPU is unavailable: {error}");
            }
            eprintln!("skipping camera GPU ABI test: {error}");
            None
        }
    }
}

#[test]
fn view_upload_reaches_wgsl_consumer_with_declared_fields() {
    let Some((device, queue)) = gpu_or_skip() else {
        return;
    };

    let view_buffer = RhiViewBuffer::new(&device, 2, 2).expect("camera view buffer");
    let mut ring = ViewBufferRing::new(2, 2).expect("camera ring");
    let slot = ring.begin_frame(0).expect("frame slot");
    let record = GpuViewRecord {
        clip_from_world: [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        clip_from_world_unjittered: [
            [101.0, 102.0, 103.0, 104.0],
            [105.0, 106.0, 107.0, 108.0],
            [109.0, 110.0, 111.0, 112.0],
            [113.0, 114.0, 115.0, 116.0],
        ],
        prev_clip_from_world: [
            [201.0, 202.0, 203.0, 204.0],
            [205.0, 206.0, 207.0, 208.0],
            [209.0, 210.0, 211.0, 212.0],
            [213.0, 214.0, 215.0, 216.0],
        ],
        view_from_world: [
            [301.0, 302.0, 303.0, 304.0],
            [305.0, 306.0, 307.0, 308.0],
            [309.0, 310.0, 311.0, 312.0],
            [313.0, 314.0, 315.0, 316.0],
        ],
        world_from_view: [
            [401.0, 402.0, 403.0, 404.0],
            [405.0, 406.0, 407.0, 408.0],
            [409.0, 410.0, 411.0, 412.0],
            [413.0, 414.0, 415.0, 416.0],
        ],
        world_origin_hi: [501.0, 502.0, 503.0, 504.0],
        world_origin_lo: [601.0, 602.0, 603.0, 604.0],
        exposure: 3.5,
        schema_version: VIEW_SCHEMA_VERSION,
        flags: 0xA5,
        view_index: 1,
        jitter: [701.0, 702.0],
        viewport: [801.0, 802.0, 803.0, 804.0],
        frustum: [
            [901.0, 902.0, 903.0, 904.0],
            [905.0, 906.0, 907.0, 908.0],
            [909.0, 910.0, 911.0, 912.0],
            [913.0, 914.0, 915.0, 916.0],
            [917.0, 918.0, 919.0, 920.0],
            [921.0, 922.0, 923.0, 924.0],
        ],
    };
    let plan = ring.plan_uploads(slot, &[record]).expect("upload plan");
    let mut upload = view_buffer.upload_backend(&queue);
    ring.submit(&plan, &mut upload).expect("RHI upload");

    let output = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera ABI output"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera ABI readback"),
        size: 256,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader = device
        .wgpu_device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("camera ABI consumer"),
            source: wgpu::ShaderSource::Wgsl(WGSL_VIEW_RECORD_CONSUMER.into()),
        });
    let layout = device
        .wgpu_device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera ABI layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let bind_group = device
        .wgpu_device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera ABI bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: device
                        .wgpu_buffer(view_buffer.buffer_id())
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
    let pipeline_layout =
        device
            .wgpu_device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("camera ABI pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
    let pipeline = device
        .wgpu_device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("camera ABI pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let mut encoder =
        device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("camera ABI encoder"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("camera ABI pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 256);
    queue.wgpu_queue().submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    assert!(device.poll_wait());
    receiver
        .recv()
        .expect("camera ABI map callback")
        .expect("camera ABI map");
    let data = slice.get_mapped_range().expect("camera ABI mapped range");
    let values: [u32; 64] = core::array::from_fn(|index| {
        let offset = index * 4;
        u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
    });
    let expected: [u32; 64] = [
        1.0f32.to_bits(),
        2.0f32.to_bits(),
        3.0f32.to_bits(),
        4.0f32.to_bits(),
        105.0f32.to_bits(),
        106.0f32.to_bits(),
        107.0f32.to_bits(),
        108.0f32.to_bits(),
        209.0f32.to_bits(),
        210.0f32.to_bits(),
        211.0f32.to_bits(),
        212.0f32.to_bits(),
        313.0f32.to_bits(),
        314.0f32.to_bits(),
        315.0f32.to_bits(),
        316.0f32.to_bits(),
        401.0f32.to_bits(),
        402.0f32.to_bits(),
        403.0f32.to_bits(),
        404.0f32.to_bits(),
        501.0f32.to_bits(),
        502.0f32.to_bits(),
        503.0f32.to_bits(),
        504.0f32.to_bits(),
        601.0f32.to_bits(),
        602.0f32.to_bits(),
        603.0f32.to_bits(),
        604.0f32.to_bits(),
        3.5f32.to_bits(),
        VIEW_SCHEMA_VERSION,
        0xA5,
        1,
        701.0f32.to_bits(),
        702.0f32.to_bits(),
        801.0f32.to_bits(),
        802.0f32.to_bits(),
        803.0f32.to_bits(),
        804.0f32.to_bits(),
        901.0f32.to_bits(),
        902.0f32.to_bits(),
        903.0f32.to_bits(),
        904.0f32.to_bits(),
        905.0f32.to_bits(),
        906.0f32.to_bits(),
        907.0f32.to_bits(),
        908.0f32.to_bits(),
        909.0f32.to_bits(),
        910.0f32.to_bits(),
        911.0f32.to_bits(),
        912.0f32.to_bits(),
        913.0f32.to_bits(),
        914.0f32.to_bits(),
        915.0f32.to_bits(),
        916.0f32.to_bits(),
        917.0f32.to_bits(),
        918.0f32.to_bits(),
        919.0f32.to_bits(),
        920.0f32.to_bits(),
        921.0f32.to_bits(),
        922.0f32.to_bits(),
        923.0f32.to_bits(),
        924.0f32.to_bits(),
        0,
        0,
    ];
    assert_eq!(values, expected);
    drop(data);
    readback.unmap();
    view_buffer.destroy(&device);
}
