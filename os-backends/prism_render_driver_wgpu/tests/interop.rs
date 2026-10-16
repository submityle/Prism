//! Native host-device adoption and record-only handoff on physical hardware.
//!
//! No native resource import is claimed here. Device equality checks are
//! deliberately confined to devices requested from one wgpu Instance: wgpu
//! 30's native IDs alone are not identities across independent Instances.
//! Set PRISM_REQUIRE_GPU=1 so missing/non-hardware adapters fail, not skip.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use prism_render_driver as rhi;
use prism_render_driver_wgpu::{
    require_hardware_gpu, AdapterEvidence, DeviceLostReason, WgpuAdapter, WgpuDevice, WgpuInstance,
    WgpuQueue,
};
use rhi::{Adapter as _, RenderDevice as _};

/// The native host requests the device; adoption must not request another one.
struct Host {
    instance: WgpuInstance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Host {
    fn adopt(&self) -> (WgpuDevice, WgpuQueue) {
        WgpuDevice::adopt(
            self.adapter.clone(),
            self.device.clone(),
            self.queue.clone(),
        )
    }
}

fn host_or_skip() -> Option<Host> {
    let instance = WgpuInstance::new();
    let Some(adapter) = instance.request_adapter(wgpu::PowerPreference::HighPerformance, false)
    else {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1 but no adapter"
        );
        eprintln!("SKIPPED native interop GPU test: no adapter");
        return None;
    };
    let description = WgpuAdapter::new(adapter.clone());
    let evidence = AdapterEvidence::from_info(description.info());
    if !evidence.is_hardware() {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1 requires physical hardware: {evidence}"
        );
        eprintln!("SKIPPED native interop physical-GPU evidence: {evidence}");
        return None;
    }
    eprintln!("native interop REAL HARDWARE GPU: {evidence}");
    // Keep the host device's enabled features below the adapter's available
    // feature set. Adoption must report this device, not recreate a wider one.
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("native-interop-host"),
            required_features: wgpu::Features::empty(),
            ..Default::default()
        }))
        .expect("physical adapter must create the host device");
    Some(Host {
        instance,
        adapter,
        device,
        queue,
    })
}

/// Read through host-owned staging: mapping is not part of the RHI buffer
/// contract. Before the recorded work is submitted this observes zeroes, so
/// an encode implementation that secretly submits cannot pass the assertion.
fn read_mapped(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<u8> {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("host-owned-interop-staging"),
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
    queue.submit([encoder.finish()]);
    let result = Arc::new(Mutex::new(None));
    let callback_result = Arc::clone(&result);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |status| {
            *callback_result.lock().expect("map result lock") = Some(status);
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("host poll must complete native mapping");
    assert_eq!(result.lock().expect("map result lock").take(), Some(Ok(())));
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .expect("successful native map must expose bytes");
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}

fn readback_buffer(device: &WgpuDevice, size: u64) -> rhi::BufferId {
    device.create_buffer(&rhi::BufferDescriptor::new(
        size,
        rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
    ))
}

#[test]
fn adoption_retains_host_handles_and_enabled_capabilities() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let (device, queue) = host.adopt();
    assert_eq!(device.wgpu_device(), &host.device);
    assert_eq!(device.wgpu_adapter(), &host.adapter);
    assert_eq!(queue.wgpu_queue(), &host.queue);
    assert_eq!(device.capabilities().features, rhi::Features::INDIRECT_DRAW);
    assert_eq!(
        device.capabilities().limits.max_texture_dimension_2d,
        host.device.limits().max_texture_dimension_2d
    );
    assert!(device.take_device_lost().is_none());

    // A distinct logical device on the same Instance/physical GPU must not
    // pass device equality. This is not a claim about cross-Instance IDs.
    let other_adapter = host
        .instance
        .request_adapter(wgpu::PowerPreference::HighPerformance, false)
        .expect("the same Instance must still provide an adapter");
    let (other_device, other_queue) = futures_lite::future::block_on(
        other_adapter.request_device(&wgpu::DeviceDescriptor::default()),
    )
    .expect("second host logical device");
    assert_ne!(device.wgpu_device(), &other_device);
    assert_ne!(queue.wgpu_queue(), &other_queue);
}

#[test]
fn repeated_adoption_keeps_rhi_resource_tables_separate() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let (first, first_queue) = host.adopt();
    let (second, second_queue) = host.adopt();
    let descriptor = rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_DST);
    let first_id = first.create_buffer(&descriptor);
    let second_id = second.create_buffer(&descriptor);
    assert_eq!(first_id, second_id, "each fresh table starts independently");
    assert_ne!(
        first.wgpu_buffer(first_id),
        second.wgpu_buffer(second_id),
        "equal table-local IDs must still resolve to distinct native objects"
    );
    first.destroy_buffer(first_id);
    drop((first_queue, first));
    assert_eq!(second.wgpu_buffer(second_id).size(), 4);
    second.destroy_buffer(second_id);
    drop((second_queue, second));
}

fn install_host_loss_callback(host: &Host) -> Arc<Mutex<Option<wgpu::DeviceLostReason>>> {
    let observed = Arc::new(Mutex::new(None));
    let callback_observed = Arc::clone(&observed);
    host.device
        .set_device_lost_callback(move |reason, _message| {
            *callback_observed.lock().expect("host loss callback lock") = Some(reason);
        });
    observed
}

fn assert_host_loss_callback(host: &Host, observed: &Mutex<Option<wgpu::DeviceLostReason>>) {
    host.device.destroy();
    for _ in 0..200 {
        let _ = host.device.poll(wgpu::PollType::wait_indefinitely());
        if observed.lock().expect("host loss lock").is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        *observed.lock().expect("host loss lock"),
        Some(wgpu::DeviceLostReason::Destroyed),
        "adoption must leave the host's actual device-lost callback installed"
    );
}

#[test]
fn adoption_does_not_replace_host_device_lost_callback() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let observed = install_host_loss_callback(&host);
    let (device, _queue) = host.adopt();
    assert_host_loss_callback(&host, &observed);
    assert!(
        device.take_device_lost().is_none(),
        "the adopted bridge must not take over the host's loss notification"
    );
}

#[test]
fn dropping_adopted_pair_preserves_host_device_and_callback() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let observed = install_host_loss_callback(&host);
    let (device, queue) = host.adopt();
    let buffer = device.create_buffer(&rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_DST));
    device.destroy_buffer(buffer);
    drop((queue, device));
    assert!(observed.lock().expect("host loss lock").is_none());
    // Actual creation and polling, not merely comparing a retained handle.
    let _host_buffer = host.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("host-after-bridge-drop"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    host.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("dropping the bridge must not destroy the host device");
    assert_host_loss_callback(&host, &observed);
}

#[test]
fn owned_bringup_still_reports_prism_device_loss() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let adapter = host
        .instance
        .request_adapter(wgpu::PowerPreference::HighPerformance, false)
        .expect("fresh adapter for owned bringup");
    let (device, _queue) = host
        .instance
        .create_device(&adapter)
        .expect("owned bringup must still succeed");
    device.wgpu_device().destroy();
    let mut observed = None;
    for _ in 0..200 {
        let _ = device.poll_wait();
        observed = device.take_device_lost();
        if observed.is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(observed, Some(DeviceLostReason::Destroyed));
}

#[test]
fn native_encode_transfer_waits_for_host_submission() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let (device, queue) = host.adopt();
    let source = device.create_buffer(&rhi::BufferDescriptor {
        label: Some("host-handoff-source".into()),
        size: 4,
        usage: rhi::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    let native_source = device.wgpu_buffer(source);
    let payload = [0x12, 0x34, 0x56, 0x78];
    native_source
        .slice(..)
        .get_mapped_range_mut()
        .expect("source is mapped at creation")
        .copy_from_slice(&payload);
    native_source.unmap();
    let destination = readback_buffer(&device, 4);
    let native_destination = device.wgpu_buffer(destination);
    let commands = rhi::CommandBuffer {
        label: Some("native-handoff-transfer".into()),
        passes: vec![rhi::Pass::Transfer {
            label: None,
            commands: vec![rhi::CopyCommand::CopyBufferToBuffer {
                src: source,
                src_offset: 0,
                dst: destination,
                dst_offset: 0,
                size: 4,
            }],
        }],
    };
    let native_commands = queue.encode(&commands);
    assert_eq!(
        read_mapped(&host.device, &host.queue, &native_destination),
        [0; 4]
    );
    let index = host.queue.submit([native_commands]);
    host.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: None,
        })
        .expect("host-owned transfer submission must complete");
    assert_eq!(
        read_mapped(&host.device, &host.queue, &native_destination),
        payload
    );
    device.destroy_buffer(destination);
    device.destroy_buffer(source);
}

#[test]
fn native_encode_rejects_another_logical_devices_queue() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let (device, queue) = host.adopt();
    // Keep both logical devices in the same Global. Cross-Instance native
    // IDs can collide and cannot be used for ownership validation.
    let other_adapter = host
        .instance
        .request_adapter(wgpu::PowerPreference::HighPerformance, false)
        .expect("second adapter in the same Instance");
    let (other_device, other_queue) = futures_lite::future::block_on(
        other_adapter.request_device(&wgpu::DeviceDescriptor::default()),
    )
    .expect("second logical device for actual submission mismatch");
    let source = device.create_buffer(&rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_SRC));
    let destination = readback_buffer(&device, 4);
    let commands = rhi::CommandBuffer {
        label: Some("native-handoff-wrong-device".into()),
        passes: vec![rhi::Pass::Transfer {
            label: None,
            commands: vec![rhi::CopyCommand::CopyBufferToBuffer {
                src: source,
                src_offset: 0,
                dst: destination,
                dst_offset: 0,
                size: 4,
            }],
        }],
    };
    let native_commands = queue.encode(&commands);
    let scope = other_device.push_error_scope(wgpu::ErrorFilter::Validation);
    other_queue.submit([native_commands]);
    let error = futures_lite::future::block_on(scope.pop())
        .expect("wgpu must reject a real command buffer from another logical device");
    assert!(matches!(&error, wgpu::Error::Validation { .. }));
    assert!(
        error.to_string().contains("doesn't match"),
        "expected the native owner-device mismatch, not another validation failure: {error}"
    );
    device.destroy_buffer(destination);
    device.destroy_buffer(source);
}

#[test]
fn native_encode_compute_and_transfer_are_submitted_by_host() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let (device, queue) = host.adopt();
    let storage = device.create_buffer(&rhi::BufferDescriptor::new(
        4,
        rhi::BufferUsages::STORAGE | rhi::BufferUsages::COPY_SRC,
    ));
    let destination = readback_buffer(&device, 4);
    let native_destination = device.wgpu_buffer(destination);
    let module = device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(
        "@group(0) @binding(0) var<storage, read_write> value: u32;\n\
         @compute @workgroup_size(1) fn main() { value = 42u; }",
    ));
    let layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
        label: None,
        entries: vec![rhi::BindGroupLayoutEntry {
            binding: 0,
            visibility: rhi::ShaderStages::COMPUTE,
            ty: rhi::BindingType::Buffer {
                ty: rhi::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });
    let pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
        label: None,
        layout: Some(pipeline_layout),
        module,
        entry_point: "main".into(),
    });
    let group = device.create_bind_group(&rhi::BindGroupDescriptor {
        label: None,
        layout: Some(layout),
        entries: vec![rhi::BindGroupEntry {
            binding: 0,
            resource: rhi::BindingResource::Buffer {
                buffer: storage,
                offset: 0,
                size: None,
            },
        }],
    });
    let commands = rhi::CommandBuffer {
        label: Some("native-handoff-compute".into()),
        passes: vec![
            rhi::Pass::Compute {
                label: None,
                commands: vec![
                    rhi::ComputeCommand::SetPipeline(pipeline),
                    rhi::ComputeCommand::SetBindGroup {
                        index: 0,
                        bind_group: group,
                        dynamic_offsets: Vec::new(),
                    },
                    rhi::ComputeCommand::Dispatch { x: 1, y: 1, z: 1 },
                ],
            },
            rhi::Pass::Transfer {
                label: None,
                commands: vec![rhi::CopyCommand::CopyBufferToBuffer {
                    src: storage,
                    src_offset: 0,
                    dst: destination,
                    dst_offset: 0,
                    size: 4,
                }],
            },
        ],
    };
    let native_commands = queue.encode(&commands);
    assert_eq!(
        read_mapped(&host.device, &host.queue, &native_destination),
        [0; 4]
    );
    host.queue.submit([native_commands]);
    assert_eq!(
        read_mapped(&host.device, &host.queue, &native_destination),
        42_u32.to_le_bytes()
    );
    device.destroy_bind_group(group);
    device.destroy_compute_pipeline(pipeline);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_bind_group_layout(layout);
    device.destroy_shader_module(module);
    device.destroy_buffer(destination);
    device.destroy_buffer(storage);
}

#[test]
fn native_encode_render_attachment_and_copy_are_submitted_by_host() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let (device, queue) = host.adopt();
    let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(1, 1),
        rhi::TextureFormat::Rgba8Unorm,
        rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
    ));
    let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());
    let destination = readback_buffer(&device, 256);
    let native_destination = device.wgpu_buffer(destination);
    let commands = rhi::CommandBuffer {
        label: Some("native-handoff-render".into()),
        passes: vec![
            rhi::Pass::Render {
                descriptor: rhi::RenderPassDescriptor {
                    label: None,
                    color_attachments: vec![Some(rhi::ColorAttachment {
                        view,
                        resolve_target: None,
                        load: rhi::LoadOp::Clear(rhi::Color::new(1.0, 0.0, 0.0, 1.0)),
                        store: rhi::StoreOp::Store,
                    })],
                    depth_stencil_attachment: None,
                },
                commands: Vec::new(),
            },
            rhi::Pass::Transfer {
                label: None,
                commands: vec![rhi::CopyCommand::CopyTextureToBuffer {
                    src: texture,
                    src_mip_level: 0,
                    src_origin: [0; 3],
                    dst: destination,
                    dst_offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                    size: rhi::Extent3d::new_2d(1, 1),
                }],
            },
        ],
    };
    let native_commands = queue.encode(&commands);
    assert_eq!(
        read_mapped(&host.device, &host.queue, &native_destination),
        vec![0; 256]
    );
    host.queue.submit([native_commands]);
    let bytes = read_mapped(&host.device, &host.queue, &native_destination);
    assert_eq!(&bytes[..4], &[255, 0, 0, 255]);
    device.destroy_buffer(destination);
    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}
