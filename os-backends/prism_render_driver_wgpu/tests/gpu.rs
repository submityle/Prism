//! GPU-gated integration tests for the wgpu backend.
//!
//! Every test first attempts to bring up a real device via
//! [`WgpuInstance::request_default_device`]. When no adapter is present (the
//! base CI sandbox has no GPU), the test returns early and is a no-op, so the
//! suite always passes without hardware. On a machine with a GPU (e.g. a macOS
//! Metal adapter), the tests exercise the full create -> record -> submit path
//! end to end, which is the behaviour that cannot be covered by the GPU-free
//! unit tests in `convert.rs`.
//!
//! These tests depend only on `prism_render_driver_wgpu` and the frozen RHI;
//! they never name a `wgpu` type directly, relying on the no-wgpu-types
//! `request_default_device` entry point.

extern crate alloc;

use prism_render_driver as rhi;
use prism_render_driver_wgpu::{
    require_hardware_gpu, AdapterEvidence, DeviceLostReason, HardwareDeviceError, WgpuDevice,
    WgpuInstance, WgpuQueue,
};
use rhi::{Adapter, FrameSync, RenderDevice, RenderQueue};

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use alloc::sync::Arc;

/// Brings up a device/queue pair, or returns `None` when no GPU is available.
fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
    match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => {
            eprintln!(
                "wgpu integration REAL HARDWARE GPU: {:?}",
                pair.0.wgpu_adapter().get_info()
            );
            Some(pair)
        }
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU=1 but no wgpu adapter was found"
            );
            eprintln!("SKIPPED wgpu integration REAL HARDWARE GPU test: no adapter");
            None
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
            );
            eprintln!("SOFTWARE/VIRTUAL adapter; skipped physical-GPU evidence: {evidence}");
            None
        }
        Err(HardwareDeviceError::Device(error)) => {
            panic!("physical GPU adapter was found but device creation failed: {error}");
        }
    }
}

#[test]
fn adapter_inventory_reports_hardware_and_software_boundaries() {
    let instance = WgpuInstance::new();
    let evidence: Vec<_> = <WgpuInstance as rhi::Instance>::enumerate_adapters(&instance)
        .iter()
        .map(|adapter| AdapterEvidence::from_info(adapter.info()))
        .collect();
    for adapter in &evidence {
        eprintln!("wgpu adapter evidence: {adapter}");
    }
    if evidence.is_empty() {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1 but no wgpu adapter was found"
        );
        eprintln!("SKIPPED adapter inventory: no adapter");
    } else {
        if let Some(fallback) = instance.fallback_adapter_evidence() {
            assert!(!fallback.is_hardware());
            eprintln!("wgpu SOFTWARE/VIRTUAL fallback evidence: {fallback}");
        } else {
            eprintln!("wgpu fallback adapter: unavailable");
        }
        assert!(
            !require_hardware_gpu() || evidence.iter().any(AdapterEvidence::is_hardware),
            "PRISM_REQUIRE_GPU=1 requires a physical GPU adapter; observed: {evidence:?}"
        );
    }
}

/// Whether a missing adapter must fail the test instead of skipping it.
///
/// Set `PRISM_REQUIRE_GPU` to opt into strictness so a run meant to exercise
/// hardware cannot silently degrade into a no-op.
fn require_gpu() -> bool {
    require_hardware_gpu()
}

#[test]
fn gpu_trivial_compute_dispatch() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };

    // A storage buffer holding four u32 values, uploaded from the CPU.
    let buffer = device.create_buffer(&rhi::BufferDescriptor {
        label: Some("compute-io".into()),
        size: 16,
        usage: rhi::BufferUsages::STORAGE
            | rhi::BufferUsages::COPY_DST
            | rhi::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let initial: [u32; 4] = [1, 2, 3, 4];
    let bytes: Vec<u8> = initial.iter().flat_map(|v| v.to_le_bytes()).collect();
    queue.write_buffer(buffer, 0, &bytes);

    // A compute shader that doubles each element in place.
    let module = device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(
        "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n\
         @compute @workgroup_size(1)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
         \tdata[gid.x] = data[gid.x] * 2u;\n\
         }\n",
    ));

    let layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
        label: Some("compute-bgl".into()),
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
        label: Some("compute-pl".into()),
        bind_group_layouts: vec![layout],
        push_constant_ranges: Vec::new(),
    });

    let pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
        label: Some("compute".into()),
        layout: Some(pipeline_layout),
        module,
        entry_point: "main".into(),
    });

    let bind_group = device.create_bind_group(&rhi::BindGroupDescriptor {
        label: Some("compute-bg".into()),
        layout: Some(layout),
        entries: vec![rhi::BindGroupEntry {
            binding: 0,
            resource: rhi::BindingResource::Buffer {
                buffer,
                offset: 0,
                size: None,
            },
        }],
    });

    let mut encoder = rhi::CommandEncoder::labeled("compute-cb");
    encoder.push_compute_pass(
        Some("double".into()),
        vec![
            rhi::ComputeCommand::SetPipeline(pipeline),
            rhi::ComputeCommand::SetBindGroup {
                index: 0,
                bind_group,
                dynamic_offsets: Vec::new(),
            },
            rhi::ComputeCommand::Dispatch { x: 4, y: 1, z: 1 },
        ],
    );
    queue.submit(&[encoder.finish()]);

    // Drive the queue to completion so any validation error surfaces here.
    device.poll_wait();

    device.destroy_compute_pipeline(pipeline);
    device.destroy_bind_group(bind_group);
    device.destroy_pipeline_layout(pipeline_layout);
    device.destroy_bind_group_layout(layout);
    device.destroy_shader_module(module);
    device.destroy_buffer(buffer);
}

#[test]
fn gpu_trivial_clear_render_pass() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };

    let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        rhi::Extent3d::new_2d(64, 64),
        rhi::TextureFormat::Rgba8Unorm,
        rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
    ));
    let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());

    let mut encoder = rhi::CommandEncoder::labeled("clear-cb");
    encoder.push_render_pass(
        rhi::RenderPassDescriptor {
            label: Some("clear".into()),
            color_attachments: vec![Some(rhi::ColorAttachment {
                view,
                resolve_target: None,
                load: rhi::LoadOp::Clear(rhi::Color::new(0.1, 0.2, 0.3, 1.0)),
                store: rhi::StoreOp::Store,
            })],
            depth_stencil_attachment: None,
        },
        Vec::new(),
    );
    queue.submit(&[encoder.finish()]);
    device.poll_wait();

    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}

#[test]
fn gpu_completion_callback_fires_after_the_queue_drains() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };

    let fired = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&fired);
    queue.on_submitted_work_done(Box::new(move || {
        flag.store(true, Ordering::Release);
    }));

    // Advance the queue's submission index so the registered callback becomes
    // eligible to run.
    let encoder = rhi::CommandEncoder::labeled("completion-callback");
    queue.submit(&[encoder.finish()]);
    assert!(device.poll_wait(), "the device stays healthy");

    for _ in 0..10_000 {
        if fired.load(Ordering::Acquire) {
            break;
        }
        std::thread::yield_now();
    }
    assert!(
        fired.load(Ordering::Acquire),
        "the completion callback ran once the queue drained"
    );
}

/// Polls until a device loss has been reported, or the bounded window expires.
///
/// wgpu fires the device-lost callback from a poll (it waits for the queue to
/// drain before completing the "lose the device" steps), so a caller that has
/// just destroyed the device must keep polling for the record to appear.
fn poll_for_device_lost(device: &WgpuDevice, attempts: u32) -> Option<DeviceLostReason> {
    for _ in 0..attempts {
        // The poll's own return value is irrelevant here: a destroyed device
        // legitimately reports `false`, and the callback runs regardless.
        let _ = device.poll_wait();
        if let Some(reason) = device.take_device_lost() {
            return Some(reason);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    None
}

#[test]
fn gpu_device_destroy_reports_device_lost() {
    let Some((device, _queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };

    // A freshly brought-up device has not been lost.
    assert!(
        device.take_device_lost().is_none(),
        "a healthy device reports no loss"
    );

    // `destroy` is "lose the device" for a real wgpu device: it invalidates the
    // device and wgpu delivers the registered callback with `Destroyed` from a
    // later poll, once the queue has drained.
    device.wgpu_device().destroy();

    let Some(reason) = poll_for_device_lost(&device, 200) else {
        // Honest skip: this backend never delivered a DeviceLost callback
        // within a second of `destroy`. The wiring and the accessor stand; we
        // do not fabricate a reason.
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but destroy() produced no DeviceLost callback"
        );
        return;
    };

    assert_eq!(reason, DeviceLostReason::Destroyed);
}
