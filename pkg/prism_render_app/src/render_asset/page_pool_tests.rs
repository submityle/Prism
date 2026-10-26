//! Contract tests for the fixed-page GPU pool.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::AtomicBool;

use prism_asset::{AssetSources, MemSource, PageBudget, PageKey, PageResidency, PageSize};
use prism_render_architecture::gpu_scene::GpuCompletionValue;
use prism_render_driver::{
    BufferUsages, CommandEncoder, DeviceDescriptor, FrameSync, Instance, RenderQueue,
    RequestAdapterOptions,
};
use prism_render_driver_null::{NullDevice, NullInstance, NullQueue};

use super::{
    FixedPagePool, FixedPagePoolDescriptor, FixedPagePoolEvent, FixedPageResidencyBridge,
    FixedPageSlotKey, FixedPageUpload, GpuPrepareBudget, PageAllocation, PagePoolError,
    PrepareContext,
};

fn null_device() -> (NullDevice, NullQueue) {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("the null backend always has an adapter");
    instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null backend always creates a device")
}

fn null_pool(device: &NullDevice, queue: &NullQueue) -> FixedPagePool {
    let mut budget = GpuPrepareBudget::new(64);
    let cancellation = AtomicBool::new(false);
    let mut context = PrepareContext::new(device, queue, &mut budget, &cancellation);
    let pool = FixedPagePool::new(
        &mut context,
        &FixedPagePoolDescriptor::new(16, 4, BufferUsages::COPY_DST | BufferUsages::COPY_SRC),
    )
    .expect("the null device accepts the fixed-page buffer");
    context.commit();
    pool
}

#[test]
fn host_metadata_admission_precedes_gpu_allocation() {
    let (device, queue) = null_device();
    let mut descriptor = FixedPagePoolDescriptor::new(4, u32::MAX, BufferUsages::COPY_DST);
    let required = descriptor.required_host_bytes().unwrap();
    let mut budget = GpuPrepareBudget::new(u64::MAX);
    let cancellation = AtomicBool::new(false);
    {
        let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
        let error = FixedPagePool::new(&mut context, &descriptor).err().unwrap();
        assert_eq!(
            error,
            PagePoolError::HostBudgetExceeded {
                required,
                limit: descriptor.max_host_bytes,
            }
        );
        assert_eq!(device.live_resource_count(), 0);
        // Even a failed prepare that the caller commits has no GPU side effects.
        context.commit();
    }
    descriptor.page_count = 4;
    descriptor.max_host_bytes = descriptor.required_host_bytes().unwrap() - 1;
    {
        let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
        assert!(matches!(
            FixedPagePool::new(&mut context, &descriptor),
            Err(PagePoolError::HostBudgetExceeded { .. })
        ));
        assert_eq!(device.live_resource_count(), 0);
    }
    descriptor.max_host_bytes += 1;
    let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
    let pool = FixedPagePool::new(&mut context, &descriptor).unwrap();
    context.commit();
    assert_eq!(pool.free_page_count(), 4);
    pool.destroy(&device);
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn mixed_completion_batches_preserve_order_and_quarantine_unfinished_pages() {
    let (device, queue) = null_device();
    let mut pool = null_pool(&device, &queue);
    let mut allocations = Vec::new();
    for revision in 0..4 {
        let mut upload = pool
            .begin_upload(FixedPageUpload::new(revision, &[0; 4]))
            .unwrap();
        let allocation = upload.allocation();
        upload.upload(&queue).unwrap();
        upload.commit().unwrap();
        allocations.push(allocation);
    }
    for (allocation, completion) in allocations.iter().zip([7, 3, 7, 3]) {
        pool.begin_release(*allocation, 10)
            .unwrap()
            .commit_deferred(GpuCompletionValue(completion))
            .unwrap();
    }
    assert_eq!(pool.pending_reclaim_page_count(), 4);
    let first = pool.reclaim_completed(GpuCompletionValue(3)).unwrap();
    assert_eq!(
        first
            .iter()
            .map(|event| event.allocation)
            .collect::<Vec<_>>(),
        vec![allocations[1], allocations[3]]
    );
    assert_eq!(pool.free_page_count(), 2);
    assert_eq!(pool.pending_reclaim_page_count(), 2);
    assert!(pool
        .reclaim_completed(GpuCompletionValue(6))
        .unwrap()
        .is_empty());
    let second = pool.reclaim_completed(GpuCompletionValue(7)).unwrap();
    assert_eq!(
        second
            .iter()
            .map(|event| event.allocation)
            .collect::<Vec<_>>(),
        vec![allocations[0], allocations[2]]
    );
    assert_eq!(pool.free_page_count(), 4);
    assert_eq!(pool.pending_reclaim_page_count(), 0);
    pool.destroy(&device);
}

/// A forwarding queue makes the RHI call observable without replacing the
/// real null device or changing the pool's upload path.
#[derive(Default)]
struct RecordingQueue {
    writes: std::sync::Mutex<Vec<(prism_render_driver::BufferId, u64, Vec<u8>)>>,
}

impl RenderQueue for RecordingQueue {
    fn write_buffer(&self, buffer: prism_render_driver::BufferId, offset: u64, data: &[u8]) {
        self.writes
            .lock()
            .expect("recording queue lock")
            .push((buffer, offset, data.to_vec()));
    }

    fn write_texture(
        &self,
        _destination: prism_render_driver::TextureWrite,
        _data: &[u8],
        _layout: prism_render_driver::ImageDataLayout,
    ) {
    }

    fn submit(&self, _command_buffers: &[prism_render_driver::CommandBuffer]) {}
}

#[test]
fn null_rhi_allocation_upload_replace_and_release_are_transactional() {
    let (device, null_queue) = null_device();
    let mut pool = null_pool(&device, &null_queue);
    let recording_queue = RecordingQueue::default();

    let first = {
        let mut transaction = pool
            .begin_upload(FixedPageUpload::new(7, &[1, 2, 3, 4, 5, 6, 7, 8]))
            .expect("one page fits");
        let allocation = transaction.allocation();
        transaction
            .upload(&recording_queue)
            .expect("the RHI upload is accepted");
        assert_eq!(
            transaction.commit(),
            Ok(FixedPagePoolEvent::Allocated {
                allocation,
                revision: 7,
            })
        );
        allocation
    };

    let writes = recording_queue.writes.lock().expect("recording queue lock");
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0].0, pool.buffer());
    assert_eq!(writes[0].1, 0);
    assert_eq!(writes[0].2, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    drop(writes);
    assert_eq!(pool.resident_page_count(), 1);

    // Dropping an unfinished replacement preserves the old resident range
    // and returns only the new reservation.
    {
        let mut replacement = pool
            .begin_replace(first, FixedPageUpload::new(8, &[9, 10, 11, 12]))
            .expect("a second page is available for staging");
        replacement
            .upload(&recording_queue)
            .expect("the replacement upload is accepted");
    }
    assert_eq!(pool.resident_page_count(), 1);
    assert_eq!(pool.free_page_count(), 3);

    let second = {
        let mut replacement = pool
            .begin_replace(
                first,
                FixedPageUpload::new(
                    9,
                    &[
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                        32,
                    ],
                ),
            )
            .expect("two staged pages fit");
        let allocation = replacement.allocation();
        replacement
            .upload(&recording_queue)
            .expect("the replacement upload is accepted");
        assert_eq!(
            replacement.commit(),
            Ok(FixedPagePoolEvent::Replaced {
                previous: first,
                allocation,
                revision: 9,
            })
        );
        allocation
    };
    assert_eq!(pool.resident_page_count(), 2);
    assert_eq!(pool.free_page_count(), 2);

    let release = pool
        .begin_release(second, 10)
        .expect("the replacement is resident");
    assert_eq!(
        release.commit(),
        Ok(FixedPagePoolEvent::Released {
            allocation: second,
            revision: 10,
        })
    );
    assert_eq!(pool.resident_page_count(), 0);
    assert_eq!(pool.free_page_count(), 4);

    pool.destroy(&device);
    assert!(pool.is_destroyed());
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn deferred_replacement_reclaims_only_after_completion_proof() {
    let (device, queue) = null_device();
    let mut pool = null_pool(&device, &queue);
    let recording_queue = RecordingQueue::default();

    let first = {
        let mut transaction = pool
            .begin_upload(FixedPageUpload::new(1, &[0; 16]))
            .expect("the first page fits");
        let allocation = transaction.allocation();
        transaction
            .upload(&recording_queue)
            .expect("upload succeeds");
        transaction.commit().expect("first allocation publishes");
        allocation
    };

    let deferred = {
        let mut transaction = pool
            .begin_replace(first, FixedPageUpload::new(2, &[1; 16]))
            .expect("replacement staging fits");
        let allocation = transaction.allocation();
        transaction
            .upload(&recording_queue)
            .expect("upload succeeds");
        let deferred = transaction
            .commit_deferred(GpuCompletionValue(7))
            .expect("deferred replacement publishes");
        assert_eq!(deferred.previous, first);
        assert_eq!(deferred.allocation, allocation);
        assert_eq!(deferred.revision, 2);
        assert_eq!(deferred.reclaim_after, GpuCompletionValue(7));
        deferred
    };

    assert_eq!(pool.resident_page_count(), 1);
    assert_eq!(pool.pending_reclaim_page_count(), 1);
    assert_eq!(pool.free_page_count(), 2);
    assert!(pool
        .reclaim_completed(GpuCompletionValue(6))
        .expect("an incomplete proof is valid")
        .is_empty());
    assert_eq!(pool.pending_reclaim_page_count(), 1);
    assert_eq!(pool.free_page_count(), 2);

    pool.begin_release(deferred.allocation, 3)
        .expect("the replacement is resident")
        .commit()
        .expect("the replacement release commits");
    assert_eq!(pool.resident_page_count(), 0);
    assert_eq!(pool.free_page_count(), 3);

    let error = match pool.begin_upload(FixedPageUpload::new(4, &[0; 64])) {
        Ok(_) => panic!("the retiring page must not be reused early"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        PagePoolError::OutOfPages {
            requested: 4,
            largest_contiguous: 3,
        }
    );

    let reclaimed = pool
        .reclaim_completed(GpuCompletionValue(7))
        .expect("the completion proof permits reclamation");
    assert_eq!(reclaimed.len(), 1);
    assert_eq!(reclaimed[0].allocation, first);
    assert_eq!(reclaimed[0].reclaim_after, GpuCompletionValue(7));
    assert_eq!(reclaimed[0].completed, GpuCompletionValue(7));
    assert_eq!(pool.pending_reclaim_page_count(), 0);
    assert_eq!(pool.free_page_count(), 4);

    let mut reuse = pool
        .begin_upload(FixedPageUpload::new(5, &[0; 64]))
        .expect("the reclaimed pages are reusable");
    reuse.upload(&recording_queue).expect("upload succeeds");
    reuse.commit().expect("reused allocation publishes");

    assert!(pool
        .reclaim_completed(GpuCompletionValue(7))
        .expect("reclaim is idempotent after the transition")
        .is_empty());
    pool.destroy(&device);
}

#[test]
fn deferred_reclaims_are_independent_and_follow_completion_values() {
    let (device, queue) = null_device();
    let mut pool = null_pool(&device, &queue);
    let recording_queue = RecordingQueue::default();

    let upload = |pool: &mut FixedPagePool, revision: u64, byte: u8| {
        let bytes = [byte; 16];
        let mut transaction = pool
            .begin_upload(FixedPageUpload::new(revision, &bytes))
            .expect("page fits");
        let allocation = transaction.allocation();
        transaction
            .upload(&recording_queue)
            .expect("upload succeeds");
        transaction.commit().expect("upload publishes");
        allocation
    };
    let first = upload(&mut pool, 1, 1);
    let second = upload(&mut pool, 2, 2);

    let replace = |pool: &mut FixedPagePool,
                   previous: PageAllocation,
                   revision: u64,
                   byte: u8,
                   reclaim_after: u64| {
        let bytes = [byte; 16];
        let mut transaction = pool
            .begin_replace(previous, FixedPageUpload::new(revision, &bytes))
            .expect("replacement fits");
        transaction
            .upload(&recording_queue)
            .expect("upload succeeds");
        transaction
            .commit_deferred(GpuCompletionValue(reclaim_after))
            .expect("deferred replacement publishes")
    };
    let first_replacement = replace(&mut pool, first, 3, 3, 3);
    let second_replacement = replace(&mut pool, second, 4, 4, 7);
    assert_eq!(pool.pending_reclaim_page_count(), 2);
    assert_eq!(pool.free_page_count(), 0);

    let first_reclaimed = pool
        .reclaim_completed(GpuCompletionValue(4))
        .expect("first completion permits one reclaim");
    assert_eq!(first_reclaimed.len(), 1);
    assert_eq!(first_reclaimed[0].allocation, first_replacement.previous);
    assert_eq!(pool.pending_reclaim_page_count(), 1);
    assert_eq!(pool.free_page_count(), 1);

    assert!(pool
        .reclaim_completed(GpuCompletionValue(6))
        .expect("second completion is not ready")
        .is_empty());
    let second_reclaimed = pool
        .reclaim_completed(GpuCompletionValue(7))
        .expect("second completion permits the final reclaim");
    assert_eq!(second_reclaimed.len(), 1);
    assert_eq!(second_reclaimed[0].allocation, second_replacement.previous);
    assert_eq!(pool.pending_reclaim_page_count(), 0);
    assert_eq!(pool.free_page_count(), 2);
    pool.destroy(&device);
}

#[test]
fn null_rhi_prepare_context_rollback_releases_pool_buffer() {
    let (device, queue) = null_device();
    let cancellation = AtomicBool::new(false);
    let mut budget = GpuPrepareBudget::new(32);
    {
        let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
        let error = match FixedPagePool::new(
            &mut context,
            &FixedPagePoolDescriptor::new(16, 4, BufferUsages::COPY_DST | BufferUsages::COPY_SRC),
        ) {
            Ok(_) => panic!("the contextual budget is smaller than the backing buffer"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            PagePoolError::Context(super::PrepareContextError::BudgetExceeded {
                requested: 64,
                remaining: 32,
            })
        );
    }
    assert_eq!(budget.used(), 0);
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn resident_cpu_page_uploads_through_the_render_bridge() {
    let (device, queue) = null_device();
    let mut source = MemSource::new();
    source.insert("payload.bin", (0u8..16).collect::<Vec<u8>>());
    let mut sources = AssetSources::new();
    sources.mount("memory", Arc::new(source), 0);

    let mut residency = PageResidency::new(
        PageSize::new(16).expect("power-of-two page"),
        PageBudget::new(16, 1).expect("one page budget"),
    )
    .expect("aligned page budget");
    let page = residency
        .request(
            PageKey::from_sources(&sources, "memory://payload.bin", 0),
            prism_tasks::Priority::Normal,
            0,
        )
        .expect("page request");
    residency.load(&sources, page).expect("range read");

    let cancellation = AtomicBool::new(false);
    let mut budget = GpuPrepareBudget::new(64);
    let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
    let mut bridge = FixedPageResidencyBridge::new(
        &mut context,
        &FixedPagePoolDescriptor::new(16, 2, BufferUsages::COPY_DST),
    )
    .expect("GPU pool");
    context.commit();

    let recording_queue = RecordingQueue::default();
    let event = bridge
        .upload_page_deferred(&residency, page, 3, GpuCompletionValue(3), &recording_queue)
        .expect("resident page upload");
    assert!(matches!(
        event,
        FixedPagePoolEvent::Allocated { revision: 3, .. }
    ));
    assert_eq!(recording_queue.writes.lock().unwrap().len(), 1);
    let first_allocation = bridge.allocation(page).expect("published GPU allocation");

    let replaced = bridge
        .upload_page_deferred(&residency, page, 4, GpuCompletionValue(4), &recording_queue)
        .expect("resident page replacement");
    let second_allocation = bridge.allocation(page).expect("replacement allocation");
    assert_ne!(first_allocation, second_allocation);
    assert!(matches!(
        replaced,
        FixedPagePoolEvent::Replaced { revision: 4, .. }
    ));
    assert_eq!(recording_queue.writes.lock().unwrap().len(), 2);

    let released = bridge.release_page(page, 4).expect("GPU page release");
    assert!(matches!(
        released,
        FixedPagePoolEvent::Released { revision: 4, .. }
    ));
    bridge.destroy(&device);
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn render_bridge_deferred_replacement_requires_completion_before_reuse() {
    let (device, queue) = null_device();
    let mut source = MemSource::new();
    source.insert("payload.bin", (0u8..16).collect::<Vec<u8>>());
    let mut sources = AssetSources::new();
    sources.mount("memory", Arc::new(source), 0);

    let mut residency = PageResidency::new(
        PageSize::new(16).expect("power-of-two page"),
        PageBudget::new(16, 1).expect("one page budget"),
    )
    .expect("aligned page budget");
    let page = residency
        .request(
            PageKey::from_sources(&sources, "memory://payload.bin", 0),
            prism_tasks::Priority::Normal,
            0,
        )
        .expect("page request");
    residency.load(&sources, page).expect("range read");

    let cancellation = AtomicBool::new(false);
    let mut budget = GpuPrepareBudget::new(64);
    let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
    let mut bridge = FixedPageResidencyBridge::new(
        &mut context,
        &FixedPagePoolDescriptor::new(16, 2, BufferUsages::COPY_DST),
    )
    .expect("GPU pool");
    context.commit();
    let recording_queue = RecordingQueue::default();

    let first = bridge
        .upload_page_deferred(&residency, page, 1, GpuCompletionValue(1), &recording_queue)
        .expect("initial upload");
    let first = match first {
        FixedPagePoolEvent::Allocated { allocation, .. } => allocation,
        other => panic!("unexpected initial event: {other:?}"),
    };
    let replacement = bridge
        .upload_page_deferred(&residency, page, 2, GpuCompletionValue(9), &recording_queue)
        .expect("deferred replacement");
    assert!(matches!(
        replacement,
        FixedPagePoolEvent::Replaced { previous, .. } if previous == first
    ));
    assert_eq!(bridge.pool().pending_reclaim_page_count(), 1);
    assert!(bridge
        .reclaim_completed(GpuCompletionValue(8))
        .expect("incomplete completion is accepted")
        .is_empty());
    let reclaimed = bridge
        .reclaim_completed(GpuCompletionValue(9))
        .expect("completion permits reclaim");
    assert_eq!(reclaimed.len(), 1);
    assert_eq!(reclaimed[0].allocation, first);
    bridge.destroy(&device);
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn stable_slot_replacement_retires_old_revision_and_ignores_stale_release() {
    let (device, queue) = null_device();
    let mut residency = PageResidency::new(
        PageSize::new(16).expect("power-of-two page"),
        PageBudget::new(32, 2).expect("two page budget"),
    )
    .expect("aligned page budget");
    let old_page = residency
        .request(
            PageKey::new("memory://payload.bin", 0, 1),
            prism_tasks::Priority::Normal,
            0,
        )
        .expect("old revision request");
    let new_page = residency
        .request(
            PageKey::new("memory://payload.bin", 0, 2),
            prism_tasks::Priority::Normal,
            0,
        )
        .expect("new revision request");

    let cancellation = AtomicBool::new(false);
    let mut budget = GpuPrepareBudget::new(64);
    let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
    let mut bridge = FixedPageResidencyBridge::new(
        &mut context,
        &FixedPagePoolDescriptor::new(16, 2, BufferUsages::COPY_DST),
    )
    .expect("GPU pool");
    context.commit();
    let recording_queue = RecordingQueue::default();
    let slot = FixedPageSlotKey {
        path: "memory://payload.bin".into(),
        page_index: 0,
    };

    bridge
        .upload_page_bytes_for_slot(
            &slot,
            old_page,
            1,
            &[0; 16],
            GpuCompletionValue(7),
            &recording_queue,
        )
        .expect("old revision upload");
    bridge
        .upload_page_bytes_for_slot(
            &slot,
            new_page,
            2,
            &[1; 16],
            GpuCompletionValue(7),
            &recording_queue,
        )
        .expect("new revision replaces the stable slot");
    assert_eq!(
        bridge.upload_page_bytes_for_slot(
            &slot,
            old_page,
            1,
            &[9; 16],
            GpuCompletionValue(8),
            &recording_queue,
        ),
        Err(super::PageResidencyBridgeError::StaleRevision {
            current: 2,
            incoming: 1,
        })
    );
    assert_eq!(bridge.pool().resident_page_count(), 1);
    assert_eq!(bridge.pool().pending_reclaim_page_count(), 1);

    assert_eq!(
        bridge.release_page_deferred(old_page, 1, GpuCompletionValue(8)),
        Err(super::PageResidencyBridgeError::UnknownPage(old_page))
    );
    assert!(bridge
        .reclaim_completed(GpuCompletionValue(6))
        .expect("old revision is still in flight")
        .is_empty());
    assert_eq!(
        bridge
            .reclaim_completed(GpuCompletionValue(7))
            .expect("replacement completion permits reclaim")
            .len(),
        1
    );

    bridge
        .release_page_deferred(new_page, 2, GpuCompletionValue(9))
        .expect("current revision release");
    assert_eq!(bridge.pool().pending_reclaim_page_count(), 1);
    bridge.destroy(&device);
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn null_rhi_rejects_stale_and_non_contiguous_allocations() {
    let (device, queue) = null_device();
    let mut pool = null_pool(&device, &queue);
    let recording_queue = RecordingQueue::default();

    let mut a = pool
        .begin_upload(FixedPageUpload::new(1, &[0; 16]))
        .expect("first page");
    a.upload(&recording_queue).expect("upload a");
    let a = match a.commit().expect("commit a") {
        FixedPagePoolEvent::Allocated { allocation, .. } => allocation,
        other => panic!("unexpected event: {other:?}"),
    };
    let mut b = pool
        .begin_upload(FixedPageUpload::new(2, &[0; 32]))
        .expect("two pages");
    b.upload(&recording_queue).expect("upload b");
    let b = b.commit().expect("commit b");
    let b = match b {
        FixedPagePoolEvent::Allocated { allocation, .. } => allocation,
        other => panic!("unexpected event: {other:?}"),
    };

    // Release the first allocation, leaving two one-page holes around the
    // resident two-page allocation. A two-page upload cannot fit even though
    // two pages are free in total.
    let release = pool.begin_release(a, 3).expect("a is resident");
    release.commit().expect("release a");
    let error = match pool.begin_upload(FixedPageUpload::new(4, &[0; 32])) {
        Ok(_) => panic!("two contiguous pages are not available"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        PagePoolError::OutOfPages {
            requested: 2,
            largest_contiguous: 1,
        }
    );
    assert!(pool.begin_release(a, 5).is_err(), "a is stale");

    let release = pool.begin_release(b, 6).expect("b is resident");
    release.commit().expect("release b");
    pool.destroy(&device);
}

/// A GPU-gated test. When an adapter exists, `WgpuQueue::read_buffer` verifies
/// the actual bytes in the backing buffer after page uploads and replacement.
/// When no adapter exists, the test is an explicit, opt-in-aware skip rather
/// than a claim that the null backend proved hardware visibility.
#[test]
fn gpu_fixed_page_pool_upload_and_replace_are_readable() {
    use prism_render_driver_wgpu::{require_hardware_gpu, WgpuInstance};

    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(prism_render_driver_wgpu::HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no wgpu adapter was found"
            );
            eprintln!("SKIPPED fixed-page REAL HARDWARE GPU test: no adapter");
            return;
        }
        Err(prism_render_driver_wgpu::HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
            );
            eprintln!("SOFTWARE/VIRTUAL adapter; skipped physical-GPU evidence: {evidence}");
            return;
        }
        Err(prism_render_driver_wgpu::HardwareDeviceError::Device(error)) => {
            panic!("physical GPU adapter was found but device creation failed: {error}");
        }
    };
    eprintln!(
        "fixed-page REAL HARDWARE GPU: {:?}",
        device.wgpu_adapter().get_info()
    );

    let mut budget = GpuPrepareBudget::new(64);
    let cancellation = AtomicBool::new(false);
    let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
    let mut pool = FixedPagePool::new(
        &mut context,
        &FixedPagePoolDescriptor::new(16, 4, BufferUsages::COPY_DST | BufferUsages::COPY_SRC),
    )
    .expect("the real device accepts the fixed-page buffer");
    context.commit();

    let first = {
        let mut transaction = pool
            .begin_upload(FixedPageUpload::new(11, &[1, 2, 3, 4, 5, 6, 7, 8]))
            .expect("first upload fits");
        let allocation = transaction.allocation();
        transaction.upload(&queue).expect("queue write is valid");
        transaction.commit().expect("publish first upload");
        allocation
    };
    assert_eq!(
        queue.read_buffer(pool.buffer(), 8),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );

    let second = {
        let mut transaction = pool
            .begin_replace(
                first,
                FixedPageUpload::new(12, &[9, 10, 11, 12, 13, 14, 15, 16]),
            )
            .expect("replacement staging fits");
        let allocation = transaction.allocation();
        transaction.upload(&queue).expect("queue write is valid");
        transaction.commit().expect("publish replacement");
        allocation
    };
    let bytes = queue.read_buffer(pool.buffer(), 32);
    assert_eq!(&bytes[0..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
    let offset = second.offset_bytes(pool.page_size()) as usize;
    assert_eq!(&bytes[offset..offset + 8], &[9, 10, 11, 12, 13, 14, 15, 16]);

    let release = pool
        .begin_release(second, 13)
        .expect("replacement is resident");
    release.commit().expect("release replacement");
    pool.destroy(&device);
}

/// The deferred page state is retired by the same real queue completion token
/// that ordered the upload. A bookkeeping value alone must not make the old
/// range reusable before the native submission has completed.
#[test]
fn gpu_queue_completion_proves_deferred_page_reclaim() {
    use prism_render_driver_wgpu::{require_hardware_gpu, WgpuInstance};

    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(prism_render_driver_wgpu::HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no wgpu adapter was found"
            );
            eprintln!("SKIPPED page reclaim REAL HARDWARE GPU test: no adapter");
            return;
        }
        Err(prism_render_driver_wgpu::HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
            );
            eprintln!("SOFTWARE/VIRTUAL adapter; skipped physical-GPU evidence: {evidence}");
            return;
        }
        Err(prism_render_driver_wgpu::HardwareDeviceError::Device(error)) => {
            panic!("physical GPU adapter was found but device creation failed: {error}");
        }
    };

    let mut budget = GpuPrepareBudget::new(64);
    let cancellation = AtomicBool::new(false);
    let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);
    let mut pool = FixedPagePool::new(
        &mut context,
        &FixedPagePoolDescriptor::new(16, 2, BufferUsages::COPY_DST | BufferUsages::COPY_SRC),
    )
    .expect("the real device accepts the fixed-page buffer");
    context.commit();

    let first = {
        let mut upload = pool
            .begin_upload(FixedPageUpload::new(1, &[1; 16]))
            .expect("the first page fits");
        let allocation = upload.allocation();
        upload.upload(&queue).expect("first upload is accepted");
        upload.commit().expect("first upload publishes");
        allocation
    };
    let replacement = {
        let mut upload = pool
            .begin_replace(first, FixedPageUpload::new(2, &[2; 16]))
            .expect("the second page stages the replacement");
        let allocation = upload.allocation();
        upload
            .upload(&queue)
            .expect("replacement upload is accepted");
        upload
            .commit_deferred(GpuCompletionValue(1))
            .expect("replacement publishes with deferred reclaim");
        allocation
    };
    assert_eq!(pool.pending_reclaim_page_count(), 1);
    assert!(
        pool.begin_upload(FixedPageUpload::new(3, &[3; 16]))
            .is_err(),
        "the retired page is not reusable before completion proof"
    );
    assert!(pool
        .reclaim_completed(GpuCompletionValue(0))
        .expect("incomplete proof is valid")
        .is_empty());

    let token = queue.submit_tracked(&[CommandEncoder::new().finish()]);
    assert!(
        queue.wait_for_submission(&token),
        "native queue completion is confirmed"
    );
    assert_eq!(
        pool.reclaim_completed(GpuCompletionValue(1))
            .expect("completion proof reclaims the retired page")
            .len(),
        1
    );
    let reused = pool
        .begin_upload(FixedPageUpload::new(3, &[3; 16]))
        .expect("the page becomes reusable only after completion");
    assert_eq!(reused.allocation().first_page(), first.first_page());
    assert_ne!(reused.allocation().token(), replacement.token());
    drop(reused);
    pool.destroy(&device);
}

#[test]
fn descriptor_and_upload_validation_are_explicit() {
    let (device, queue) = null_device();
    let mut budget = GpuPrepareBudget::unlimited();
    let cancellation = AtomicBool::new(false);
    let mut context = PrepareContext::new(&device, &queue, &mut budget, &cancellation);

    let error = match FixedPagePool::new(
        &mut context,
        &FixedPagePoolDescriptor::new(10, 1, BufferUsages::COPY_DST),
    ) {
        Ok(_) => panic!("a non-aligned page size must be rejected"),
        Err(error) => error,
    };
    assert_eq!(error, PagePoolError::PageSizeNotAligned { page_size: 10 });

    let error = match FixedPagePool::new(
        &mut context,
        &FixedPagePoolDescriptor::new(16, 1, BufferUsages::COPY_SRC),
    ) {
        Ok(_) => panic!("a pool without COPY_DST must be rejected"),
        Err(error) => error,
    };
    assert_eq!(error, PagePoolError::MissingCopyDestination);

    let mut pool = FixedPagePool::new(
        &mut context,
        &FixedPagePoolDescriptor::new(16, 1, BufferUsages::COPY_DST),
    )
    .expect("the descriptor is valid");
    let error = match pool.begin_upload(FixedPageUpload::new(1, &[1, 2, 3])) {
        Ok(_) => panic!("an unaligned upload must be rejected"),
        Err(error) => error,
    };
    assert_eq!(error, PagePoolError::UploadNotAligned { byte_len: 3 });
}
