//! Contract tests for the fixed-page GPU pool.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::AtomicBool;

use prism_asset::{AssetSources, MemSource, PageBudget, PageKey, PageResidency, PageSize};
use prism_render_driver::{
    BufferUsages, DeviceDescriptor, Instance, RenderQueue, RequestAdapterOptions,
};
use prism_render_driver_null::{NullDevice, NullInstance, NullQueue};

use super::{
    FixedPagePool, FixedPagePoolDescriptor, FixedPagePoolEvent, FixedPageResidencyBridge,
    FixedPageUpload, GpuPrepareBudget, PagePoolError, PrepareContext,
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
        .upload_page(&residency, page, 3, &recording_queue)
        .expect("resident page upload");
    assert!(matches!(
        event,
        FixedPagePoolEvent::Allocated { revision: 3, .. }
    ));
    assert_eq!(recording_queue.writes.lock().unwrap().len(), 1);
    let first_allocation = bridge.allocation(page).expect("published GPU allocation");

    let replaced = bridge
        .upload_page(&residency, page, 4, &recording_queue)
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
    use prism_render_driver_wgpu::WgpuInstance;

    let Some((device, queue)) = WgpuInstance::new().request_default_device().ok() else {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };

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
