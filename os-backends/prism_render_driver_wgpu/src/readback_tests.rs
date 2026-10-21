use super::*;
use crate::{HardwareDeviceError, WgpuDevice, WgpuInstance};
use rhi::{BufferReadback, BufferReadbackError as Error, RenderDevice, RenderQueue};
use std::time::{Duration, Instant};

fn range(buffer: rhi::BufferId, offset: u64, size: u64) -> rhi::BufferReadbackRange {
    rhi::BufferReadbackRange {
        buffer,
        offset,
        size,
    }
}

#[test]
fn async_readback_layout_packs_exact_ranges_and_accepts_empty_batch() {
    let id = rhi::BufferId::from_parts(0, 0);
    let packed = layout(&[range(id, 4, 4), range(id, 16, 12)], 16, 64).unwrap();
    assert_eq!(packed.bytes, 16);
    assert_eq!(packed.ranges, vec![0..4, 4..16]);
    let empty = layout(&[], 0, 0).unwrap();
    assert_eq!(empty.bytes, 0);
    assert!(empty.ranges.is_empty());
}

#[test]
fn async_readback_layout_rejects_empty_unaligned_and_overflowing_ranges() {
    let id = rhi::BufferId::from_parts(0, 0);
    for invalid in [
        range(id, 0, 0),
        range(id, 1, 4),
        range(id, 0, 6),
        range(id, u64::MAX - 3, 4),
        range(id, 0, 68),
    ] {
        assert!(matches!(
            layout(&[range(id, 0, 4), invalid], u64::MAX, 64),
            Err(Error::InvalidRange { index: 1 })
        ));
    }
    // Each range fits u64; their packed sum must not wrap before budgeting.
    assert!(matches!(
        layout(
            &[range(id, 0, u64::MAX - 3), range(id, 0, 4)],
            u64::MAX,
            u64::MAX
        ),
        Err(Error::InvalidRange { index: 1 })
    ));
}

#[test]
fn async_readback_layout_checks_budget_and_packed_device_address_limits() {
    let id = rhi::BufferId::from_parts(0, 0);
    let ranges = [range(id, 0, 8), range(id, 0, 8)];
    assert!(matches!(
        layout(&ranges, 16, 16),
        Ok(Layout { bytes: 16, .. })
    ));
    assert!(matches!(
        layout(&ranges, 15, 16),
        Err(Error::BudgetExceeded {
            requested: 16,
            limit: 15
        })
    ));
    assert!(matches!(
        layout(&ranges, 16, 12),
        Err(Error::ResourceUnavailable)
    ));
    let too_large = (isize::MAX as u64 / 4 + 1) * 4;
    assert!(matches!(
        layout(&[range(id, 0, too_large)], u64::MAX, u64::MAX),
        Err(Error::ResourceUnavailable)
    ));
}

#[test]
fn async_readback_callback_status_distinguishes_pending_ready_and_failed() {
    assert_eq!(mapping_ready(PENDING), Ok(false));
    assert_eq!(mapping_ready(READY), Ok(true));
    assert_eq!(mapping_ready(FAILED), Err(Error::MapFailed));
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WgpuBufferReadback>();
}

fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
    match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => {
            eprintln!(
                "wgpu REAL HARDWARE GPU: {:?}",
                pair.0.wgpu_adapter().get_info()
            );
            Some(pair)
        }
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !crate::require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1: no readback GPU adapter"
            );
            eprintln!("SKIPPED async readback physical-GPU test: no adapter");
            None
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !crate::require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical readback GPU; observed {evidence}"
            );
            eprintln!("SKIPPED async readback physical-GPU test: {evidence}");
            None
        }
        Err(HardwareDeviceError::Device(error)) => {
            panic!("physical readback GPU device creation failed: {error}");
        }
    }
}

fn complete(ticket: &mut dyn BufferReadback) -> Vec<Vec<u8>> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match ticket.try_complete().expect("readback must not fail") {
            Some(bytes) => return bytes,
            None => {
                assert!(
                    Instant::now() < deadline,
                    "nonblocking readback did not complete"
                );
                // Poll only; even the acceptance helper must never use Wait.
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

#[test]
fn physical_gpu_async_readback_captures_k_before_k_plus_one_and_retains_sources() {
    let Some((device, queue)) = device_or_skip() else {
        return;
    };
    let descriptor = rhi::BufferDescriptor::new(
        32,
        rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
    );
    let first = device.create_buffer(&descriptor);
    let second = device.create_buffer(&descriptor);
    let k_first: Vec<u8> = (0..32).collect();
    let k_second: Vec<u8> = (64..96).collect();
    queue.write_buffer(first, 0, &k_first);
    queue.write_buffer(second, 0, &k_second);
    // The second packed range starts at byte 4, not MAP_ALIGNMENT (8).
    // Mapping once at offset zero must still return each exact slice.
    let ranges = [
        range(first, 4, 4),
        range(second, 8, 12),
        range(first, 16, 8),
    ];
    let mut snapshot = queue.request_buffer_readback(&ranges, 24).unwrap();

    // No try_complete/poll intervenes between snapshot submission and K+1.
    queue.write_buffer(first, 0, &[0xaa; 32]);
    queue.write_buffer(second, 0, &[0xbb; 32]);
    let mut next = queue.request_buffer_readback(&ranges, 24).unwrap();

    // Both tickets must own native buffers independently of recycled IDs.
    device.destroy_buffer(first);
    device.destroy_buffer(second);
    let replacement = device.create_buffer(&descriptor);
    assert!(replacement.index() == first.index() || replacement.index() == second.index());
    assert_ne!(replacement, first);
    assert_ne!(replacement, second);
    queue.write_buffer(replacement, 0, &[0xcc; 32]);
    queue.submit(&[]);

    assert_eq!(
        complete(snapshot.as_mut()),
        vec![
            k_first[4..8].to_vec(),
            k_second[8..20].to_vec(),
            k_first[16..24].to_vec(),
        ]
    );
    assert_eq!(snapshot.try_complete(), Err(Error::Consumed));
    assert_eq!(
        complete(next.as_mut()),
        vec![vec![0xaa; 4], vec![0xbb; 12], vec![0xaa; 8]]
    );
    assert_eq!(next.try_complete(), Err(Error::Consumed));

    // Cancelling delivery drops only handles; later work on the device survives.
    drop(
        queue
            .request_buffer_readback(&[range(replacement, 0, 4)], 4)
            .unwrap(),
    );
    let mut after_cancel = queue
        .request_buffer_readback(&[range(replacement, 0, 4)], 4)
        .unwrap();
    assert_eq!(complete(after_cancel.as_mut()), vec![vec![0xcc; 4]]);
    device.destroy_buffer(replacement);
}

#[test]
fn physical_gpu_async_readback_rejects_invalid_batches_without_panicking() {
    let Some((device, queue)) = device_or_skip() else {
        return;
    };
    let source = device.create_buffer(&rhi::BufferDescriptor::new(
        16,
        rhi::BufferUsages::COPY_SRC | rhi::BufferUsages::COPY_DST,
    ));
    let no_copy_src =
        device.create_buffer(&rhi::BufferDescriptor::new(16, rhi::BufferUsages::COPY_DST));
    let admitted = range(source, 0, 4);
    for bad in [
        range(source, 0, 0),
        range(source, 2, 4),
        range(source, 0, 6),
        range(source, 16, 4),
    ] {
        assert!(matches!(
            queue.request_buffer_readback(&[admitted, bad], 32),
            Err(Error::InvalidRange { index: 1 })
        ));
    }
    assert!(matches!(
        queue.request_buffer_readback(&[range(source, 0, 16)], 15),
        Err(Error::BudgetExceeded {
            requested: 16,
            limit: 15
        })
    ));
    assert!(matches!(
        queue.request_buffer_readback(&[admitted, range(no_copy_src, 0, 4)], 8),
        Err(Error::ResourceUnavailable)
    ));
    let native_limit = device.wgpu_device().limits().max_buffer_size;
    assert!(matches!(
        queue.request_buffer_readback(&[range(source, 0, native_limit + 4)], u64::MAX),
        Err(Error::InvalidRange { index: 0 })
    ));

    device.destroy_buffer(no_copy_src);
    let reused = device.create_buffer(&rhi::BufferDescriptor::new(16, rhi::BufferUsages::COPY_SRC));
    assert_eq!(reused.index(), no_copy_src.index());
    for missing in [no_copy_src, rhi::BufferId::from_parts(u32::MAX, u32::MAX)] {
        assert!(matches!(
            queue.request_buffer_readback(&[admitted, range(missing, 0, 4)], 8),
            Err(Error::ResourceUnavailable)
        ));
    }
    let mut empty = queue.request_buffer_readback(&[], 0).unwrap();
    assert_eq!(empty.try_complete(), Ok(Some(Vec::new())));
    assert_eq!(empty.try_complete(), Err(Error::Consumed));
    // Failed admission must not poison subsequent valid queue snapshots.
    queue.write_buffer(source, 0, &[1, 2, 3, 4]);
    let mut valid = queue.request_buffer_readback(&[admitted], 4).unwrap();
    assert_eq!(complete(valid.as_mut()), vec![vec![1, 2, 3, 4]]);
    device.destroy_buffer(source);
    device.destroy_buffer(reused);
}

#[test]
fn physical_gpu_async_readback_reports_map_failure_after_cancelled_mapping() {
    let Some((device, queue)) = device_or_skip() else {
        return;
    };
    let source = device.create_buffer(&rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_SRC));
    let mut ticket = WgpuBufferReadback::begin(&queue, &[range(source, 0, 4)], 4).unwrap();
    // Explicit healthy cancellation is test-only fault injection. If mapping
    // already completed, revoke that mapping too; neither path may return bytes.
    ticket.staging.as_ref().unwrap().unmap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match ticket.try_complete() {
            Err(Error::MapFailed) => break,
            Ok(None) => {
                assert!(
                    Instant::now() < deadline,
                    "cancelled mapping callback did not finish"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            result => panic!("cancelled map must not deliver a snapshot: {result:?}"),
        }
    }
    assert_eq!(ticket.try_complete(), Err(Error::MapFailed));
    drop(ticket);
    device.destroy_buffer(source);
}

#[test]
fn physical_gpu_async_readback_device_loss_stays_latched_after_reason_is_drained() {
    let Some((device, queue)) = device_or_skip() else {
        return;
    };
    let source = device.create_buffer(&rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_SRC));
    let mut ticket = queue
        .request_buffer_readback(&[range(source, 0, 4)], 4)
        .unwrap();
    device.wgpu_device().destroy();
    let deadline = Instant::now() + Duration::from_secs(10);
    let observed = loop {
        // Only the harness drives the initial loss notification. Once it has
        // fired, ticket completion/drop and admission must not poll the device.
        let _ = device.wgpu_device().poll(wgpu::PollType::Poll);
        if let Some(reason) = device.take_device_lost() {
            break reason;
        }
        assert!(
            Instant::now() < deadline,
            "device loss callback did not arrive"
        );
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(observed, rhi::DeviceLostReason::Destroyed);
    assert!(device.take_device_lost().is_none());
    assert!(queue.inner.is_lost());
    assert_eq!(ticket.try_complete(), Err(Error::DeviceLost));
    assert_eq!(ticket.try_complete(), Err(Error::DeviceLost));
    assert!(matches!(
        queue.request_buffer_readback(&[range(source, 0, 4)], 4),
        Err(Error::DeviceLost)
    ));
    assert!(matches!(
        queue.request_buffer_readback(&[], 0),
        Err(Error::DeviceLost)
    ));
    drop(ticket);
}

#[test]
fn physical_gpu_async_readback_adopt_forwards_host_loss_without_replacing_callback() {
    let Some((original, original_queue)) = device_or_skip() else {
        return;
    };
    let native_device = original.wgpu_device().clone();
    let native_queue = original_queue.wgpu_queue().clone();
    let adapter = original.wgpu_adapter().clone();
    type Notify = Box<dyn Fn(rhi::DeviceLostReason) + Send + Sync>;
    let forward = Arc::new(std::sync::Mutex::new(None::<Notify>));
    let host_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let callback_forward = Arc::clone(&forward);
    let callback_calls = Arc::clone(&host_calls);
    // Install the host callback BEFORE adoption: adoption must preserve it.
    native_device.set_device_lost_callback(move |reason, _| {
        callback_calls.fetch_add(1, Ordering::AcqRel);
        callback_forward
            .lock()
            .unwrap()
            .as_ref()
            .expect("host must install the forwarding function before teardown")(
            crate::loss::map_device_lost(reason),
        );
    });
    let (device, queue) = WgpuDevice::adopt(adapter, native_device.clone(), native_queue.clone());
    let notify = device.device_loss_notifier();
    *forward.lock().unwrap() = Some(Box::new(notify.clone()));
    drop(original_queue);
    drop(original);

    let native_source = native_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("adopt-readback-host-source"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let source = device.register_buffer(&native_source);
    native_queue.write_buffer(&native_source, 0, &[9, 8, 7, 6]);
    let mut completed = queue
        .request_buffer_readback(&[range(source, 0, 4)], 4)
        .unwrap();
    assert_eq!(complete(completed.as_mut()), vec![vec![9, 8, 7, 6]]);
    let mut pending = queue
        .request_buffer_readback(&[range(source, 0, 4)], 4)
        .unwrap();
    assert!(device.take_device_lost().is_none());
    assert!(!queue.inner.is_lost());

    // Test-only external destruction, with no concurrent readback calls.
    // Only the host/harness polls until the preserved callback forwards loss.
    native_device.destroy();
    let deadline = Instant::now() + Duration::from_secs(10);
    while host_calls.load(Ordering::Acquire) == 0 || !queue.inner.is_lost() {
        let _ = native_device.poll(wgpu::PollType::Poll);
        assert!(
            Instant::now() < deadline,
            "adopt host callback did not forward loss"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(host_calls.load(Ordering::Acquire), 1);
    assert_eq!(
        device.take_device_lost(),
        Some(rhi::DeviceLostReason::Destroyed)
    );
    assert!(device.take_device_lost().is_none());
    // No native poll or explicit unmap/destroy in these ticket operations.
    assert_eq!(pending.try_complete(), Err(Error::DeviceLost));
    assert_eq!(pending.try_complete(), Err(Error::DeviceLost));
    assert_eq!(completed.try_complete(), Err(Error::Consumed));
    assert!(matches!(
        queue.request_buffer_readback(&[range(source, 0, 4)], 4),
        Err(Error::DeviceLost)
    ));
    assert!(matches!(
        queue.request_buffer_readback(&[], 0),
        Err(Error::DeviceLost)
    ));
    notify(rhi::DeviceLostReason::Unknown);
    assert!(
        device.take_device_lost().is_none(),
        "duplicate must not republish"
    );
    drop(pending);
    drop(completed);
    drop(queue);
    drop(device);
    // The returned function has no borrow of, or native handle to, the bridge.
    notify(rhi::DeviceLostReason::Destroyed);
}

#[test]
fn physical_gpu_async_readback_owned_host_notification_gates_before_external_destroy() {
    let Some((device, queue)) = device_or_skip() else {
        return;
    };
    let source = device.create_buffer(&rhi::BufferDescriptor::new(4, rhi::BufferUsages::COPY_SRC));
    let mut pending = queue
        .request_buffer_readback(&[range(source, 0, 4)], 4)
        .unwrap();
    let notify = device.device_loss_notifier();
    let duplicate = notify.clone();
    // Cooperative teardown can close admission before native callback delivery.
    notify(rhi::DeviceLostReason::Destroyed);
    assert_eq!(
        device.take_device_lost(),
        Some(rhi::DeviceLostReason::Destroyed)
    );
    assert_eq!(pending.try_complete(), Err(Error::DeviceLost));
    assert!(matches!(
        queue.request_buffer_readback(&[range(source, 0, 4)], 4),
        Err(Error::DeviceLost)
    ));
    device.wgpu_device().destroy();
    // Neither completion nor drop may explicitly poll/unmap this lost backend.
    duplicate(rhi::DeviceLostReason::Unknown);
    assert!(device.take_device_lost().is_none());
    assert_eq!(pending.try_complete(), Err(Error::DeviceLost));
    drop(pending);
    drop(queue);
    drop(device);
    notify(rhi::DeviceLostReason::Destroyed);
}
