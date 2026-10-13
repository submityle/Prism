//! Behavioural tests for the render-asset table and its sync.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::{AtomicUsize, Ordering};

use prism_asset::{
    Asset, AssetId, AssetIndex, AssetLoader, AssetServer, Assets, LoadContext, LoadError, MemSource,
};
use prism_ecs::world::World;
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, DeviceDescriptor, Instance, RenderDevice,
    RenderQueue, RequestAdapterOptions,
};
use prism_render_driver_null::{NullDevice, NullInstance};

use super::{
    extract_assets, prepare_extracted_assets, prepare_extracted_assets_with_context,
    AssetSyncReport, ExtractedAssets, GpuPrepareBudget, PrepareAssetError, PrepareContextError,
    RenderAsset, RenderAssets,
};

/// A source asset: raw bytes with no GPU meaning of its own.
#[derive(Clone, Debug, PartialEq)]
struct Blob(Vec<u8>);

impl Asset for Blob {
    const TYPE_NAME: &'static str = "prism_render_app::Blob";
}

struct BlobLoader;

impl AssetLoader for BlobLoader {
    type Asset = Blob;

    fn extensions(&self) -> &'static [&'static str] {
        &["blob"]
    }

    fn load(&self, bytes: &[u8], _ctx: &mut LoadContext) -> Result<Blob, LoadError> {
        Ok(Blob(bytes.to_vec()))
    }
}

/// The GPU form: one real RHI buffer, so the table's lifecycle is observable
/// through the device (create on prepare, destroy on unload).
struct GpuBlob {
    buffer: BufferId,
    bytes: u64,
    /// Whether this prepare reused the previous representation's buffer.
    reused: bool,
}

/// A source whose dependency becomes ready between prepare passes without a
/// source change or a new asset event.
#[derive(Clone)]
struct RetryOnceBlob {
    attempts: Arc<AtomicUsize>,
    bytes: usize,
}

impl Asset for RetryOnceBlob {
    const TYPE_NAME: &'static str = "prism_render_app::RetryOnceBlob";
}

struct RetryOnceGpuBlob {
    buffer: BufferId,
    bytes: u64,
}

impl RenderAsset for RetryOnceGpuBlob {
    type SourceAsset = RetryOnceBlob;

    fn prepare(
        source: &RetryOnceBlob,
        id: AssetId<RetryOnceBlob>,
        device: &dyn RenderDevice,
        _queue: &dyn RenderQueue,
        _previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<RetryOnceBlob>> {
        if source.attempts.fetch_add(1, Ordering::SeqCst) == 0 {
            return Err(PrepareAssetError::RetryNextUpdate(id));
        }
        let bytes = source.bytes as u64;
        Ok(Self {
            buffer: device.create_buffer(&BufferDescriptor {
                label: Some("retry-once-gpu-blob".into()),
                size: bytes,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            bytes,
        })
    }

    fn unload(prepared: &Self, device: &dyn RenderDevice) {
        device.destroy_buffer(prepared.buffer);
    }
}

struct TransactionalBlob {
    buffer: BufferId,
    bytes: u64,
}

impl RenderAsset for TransactionalBlob {
    type SourceAsset = Blob;

    fn prepare(
        _source: &Blob,
        _id: AssetId<Blob>,
        _device: &dyn RenderDevice,
        _queue: &dyn RenderQueue,
        _previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Blob>> {
        unreachable!("the contextual path is the contract under test")
    }

    fn prepare_with_context(
        source: &Blob,
        id: AssetId<Blob>,
        context: &mut super::super::render_asset::PrepareContext<'_>,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Blob>> {
        let buffer = match context.create_buffer(&BufferDescriptor {
            label: Some("transactional-blob".into()),
            size: source.0.len() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }) {
            Ok(buffer) => buffer,
            Err(PrepareContextError::BudgetExceeded { .. } | PrepareContextError::Cancelled) => {
                return Err(PrepareAssetError::RetryNextUpdate(id));
            }
        };
        if source.0.len() == 2 {
            return Err(PrepareAssetError::RetryNextUpdate(id));
        }
        if previous.is_some() {
            context.replace_previous();
        }
        Ok(Self {
            buffer,
            bytes: source.0.len() as u64,
        })
    }

    fn unload(prepared: &Self, device: &dyn RenderDevice) {
        device.destroy_buffer(prepared.buffer);
    }
}

impl RenderAsset for GpuBlob {
    type SourceAsset = Blob;

    fn prepare(
        source: &Blob,
        id: AssetId<Blob>,
        device: &dyn RenderDevice,
        _queue: &dyn RenderQueue,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Blob>> {
        let bytes = source.0.len() as u64;
        // An empty blob cannot be prepared yet: the test's "not ready" case.
        if bytes == 0 {
            return Err(PrepareAssetError::RetryNextUpdate(id));
        }
        let reused = previous.is_some_and(|previous| previous.bytes >= bytes);
        if let Some(previous) = previous {
            if reused {
                return Ok(Self {
                    buffer: previous.buffer,
                    bytes: previous.bytes,
                    reused: true,
                });
            }
            // It has to grow: release the old buffer rather than leak it.
            device.destroy_buffer(previous.buffer);
        }
        Ok(Self {
            buffer: device.create_buffer(&BufferDescriptor {
                label: Some("gpu-blob".into()),
                size: bytes,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bytes,
            reused: false,
        })
    }

    fn unload(prepared: &Self, device: &dyn RenderDevice) {
        device.destroy_buffer(prepared.buffer);
    }
}

fn null_device() -> (NullDevice, impl RenderQueue) {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up")
}

/// One extract + prepare pass over `main`, the way the frame runs them.
fn run_frame<A: RenderAsset>(
    main: &mut World,
    render: &mut World,
    device: &dyn RenderDevice,
    queue: &dyn RenderQueue,
) -> AssetSyncReport {
    extract_assets::<A>(main, render);
    let mut extracted = render
        .remove_resource::<ExtractedAssets<A>>()
        .unwrap_or_default();
    let mut table = render
        .remove_resource::<RenderAssets<A>>()
        .unwrap_or_default();
    let report = prepare_extracted_assets::<A>(&mut extracted, &mut table, device, queue);
    render.insert_resource(extracted);
    render.insert_resource(table);
    report
}

#[test]
fn preparing_an_asset_creates_its_gpu_representation() {
    let (device, queue) = null_device();
    let mut main = World::new();
    let mut render = World::new();
    let mut sources = Assets::new();
    let handle = sources.insert(Blob(alloc::vec![1, 2, 3]));
    let id = handle.id();
    main.insert_resource(sources);

    let report = run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);

    assert_eq!(report.prepared, 1, "the added asset is prepared");
    assert_eq!(report.released, 0);
    let table = render.resource::<RenderAssets<GpuBlob>>();
    let prepared = table.get(id).expect("the representation is in the table");
    assert_eq!(prepared.bytes, 3);
    assert!(!prepared.reused, "the first prepare allocates");
    assert_eq!(
        device.live_resource_count(),
        1,
        "exactly one GPU buffer was created"
    );
}

#[test]
fn extract_and_prepare_uses_asset_server_identity_and_source_snapshot() {
    let (device, queue) = null_device();
    let server = AssetServer::new();
    server.register_loader(BlobLoader);
    let mut source = MemSource::new();
    source.insert("value.blob", vec![9, 8, 7]);
    server.mount("mem", Arc::new(source), 0);
    let handle = server.load::<Blob>("value.blob");
    let id = handle.id();

    let mut main = World::new();
    let mut render = World::new();
    main.insert_resource(server);

    let report = run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);
    assert_eq!(report.prepared, 1);
    let prepared = render
        .resource::<RenderAssets<GpuBlob>>()
        .get(id)
        .expect("server asset was prepared");
    assert_eq!(prepared.bytes, 3);

    drop(handle);
    main.resource::<AssetServer>().collect_releases();
    let report = run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);
    assert_eq!(report.released, 1);
    assert!(!render.resource::<RenderAssets<GpuBlob>>().contains(id));
}

#[test]
fn re_preparing_reuses_the_buffer_it_already_has() {
    let (device, queue) = null_device();
    let mut main = World::new();
    let mut render = World::new();
    let mut sources = Assets::new();
    let handle = sources.insert(Blob(alloc::vec![1, 2, 3, 4]));
    let id = handle.id();
    main.insert_resource(sources);
    run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);
    let first = render
        .resource::<RenderAssets<GpuBlob>>()
        .get(id)
        .expect("prepared")
        .buffer;

    // Modify the source: a payload that still fits the existing buffer.
    main.resource_mut::<Assets<Blob>>()
        .get_mut(id)
        .expect("the asset is live")
        .0 = alloc::vec![9, 9];
    let report = run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);

    assert_eq!(report.prepared, 1, "the modification re-prepares");
    let table = render.resource::<RenderAssets<GpuBlob>>();
    let prepared = table.get(id).expect("still prepared");
    assert!(prepared.reused, "the previous buffer was reused");
    assert_eq!(prepared.buffer, first, "the buffer id is unchanged");
    assert_eq!(
        device.live_resource_count(),
        1,
        "re-preparing allocated nothing"
    );
}

#[test]
fn a_removed_source_releases_its_gpu_representation() {
    let (device, queue) = null_device();
    let mut main = World::new();
    let mut render = World::new();
    let mut sources = Assets::new();
    let handle = sources.insert(Blob(alloc::vec![1, 2, 3]));
    let id = handle.id();
    main.insert_resource(sources);
    run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);
    assert_eq!(device.live_resource_count(), 1);

    drop(handle);
    main.resource_mut::<Assets<Blob>>().remove(id);
    let report = run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);

    assert_eq!(
        report.released, 1,
        "the removal releases the representation"
    );
    let table = render.resource::<RenderAssets<GpuBlob>>();
    assert!(table.get(id).is_none(), "the table no longer holds it");
    assert_eq!(
        device.live_resource_count(),
        0,
        "the GPU buffer was destroyed, not leaked"
    );
}

#[test]
fn a_deferred_prepare_keeps_the_previous_representation() {
    let (device, queue) = null_device();
    let mut main = World::new();
    let mut render = World::new();
    let mut sources = Assets::new();
    let handle = sources.insert(Blob(alloc::vec![1, 2, 3]));
    let id = handle.id();
    main.insert_resource(sources);
    run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);
    let prepared = render
        .resource::<RenderAssets<GpuBlob>>()
        .get(id)
        .expect("prepared")
        .buffer;

    // The source is now un-prepared-able (a dependency went away, say).
    main.resource_mut::<Assets<Blob>>()
        .get_mut(id)
        .expect("the asset is live")
        .0
        .clear();
    let report = run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);

    assert_eq!(report.retried, 1, "the prepare asked to retry");
    assert_eq!(report.prepared, 0);
    let table = render.resource::<RenderAssets<GpuBlob>>();
    assert_eq!(
        table.get(id).expect("still prepared").buffer,
        prepared,
        "a retry never publishes a half-prepared asset"
    );

    // Once the source is usable again, the next pass publishes the new form.
    main.resource_mut::<Assets<Blob>>()
        .get_mut(id)
        .expect("the asset is live")
        .0 = alloc::vec![7; 8];
    let report = run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);
    assert_eq!(report.prepared, 1, "the retry succeeded on the next pass");
    let table = render.resource::<RenderAssets<GpuBlob>>();
    assert_eq!(table.get(id).expect("prepared").bytes, 8);
}

#[test]
fn a_new_asset_whose_prepare_defers_is_not_in_the_table() {
    let (device, queue) = null_device();
    let mut main = World::new();
    let mut render = World::new();
    let mut sources = Assets::new();
    let id = sources.insert(Blob(Vec::new())).id();
    main.insert_resource(sources);

    let report = run_frame::<GpuBlob>(&mut main, &mut render, &device, &queue);

    assert_eq!(report.retried, 1);
    let table = render.resource::<RenderAssets<GpuBlob>>();
    assert!(
        !table.contains(id),
        "a deferred asset must not appear as prepared"
    );
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn a_retry_is_prepared_again_without_a_source_change() {
    let (device, queue) = null_device();
    let mut main = World::new();
    let mut render = World::new();
    let attempts = Arc::new(AtomicUsize::new(0));
    let mut sources = Assets::new();
    let handle = sources.insert(RetryOnceBlob {
        attempts: attempts.clone(),
        bytes: 6,
    });
    let id = handle.id();
    main.insert_resource(sources);

    let first = run_frame::<RetryOnceGpuBlob>(&mut main, &mut render, &device, &queue);
    assert_eq!(first.retried, 1);
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    assert!(!render
        .resource::<RenderAssets<RetryOnceGpuBlob>>()
        .contains(id));
    assert_eq!(
        render
            .resource::<ExtractedAssets<RetryOnceGpuBlob>>()
            .extracted()
            .len(),
        1,
        "the retry remains pending"
    );

    // No main-world mutation and therefore no new AssetEvent: the pending
    // copied source itself drives the second attempt.
    let second = run_frame::<RetryOnceGpuBlob>(&mut main, &mut render, &device, &queue);
    assert_eq!(second.prepared, 1);
    assert_eq!(second.retried, 0);
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        render
            .resource::<RenderAssets<RetryOnceGpuBlob>>()
            .get(id)
            .expect("the retry was prepared")
            .bytes,
        6
    );
    assert!(render
        .resource::<ExtractedAssets<RetryOnceGpuBlob>>()
        .is_empty());
}

#[test]
fn removal_clears_a_pending_retry_without_requeueing_it() {
    let (device, queue) = null_device();
    let mut main = World::new();
    let mut render = World::new();
    let attempts = Arc::new(AtomicUsize::new(0));
    let mut sources = Assets::new();
    let handle = sources.insert(RetryOnceBlob {
        attempts: attempts.clone(),
        bytes: 6,
    });
    let id = handle.id();
    main.insert_resource(sources);

    let first = run_frame::<RetryOnceGpuBlob>(&mut main, &mut render, &device, &queue);
    assert_eq!(first.retried, 1);
    drop(handle);
    main.resource_mut::<Assets<RetryOnceBlob>>().remove(id);

    let second = run_frame::<RetryOnceGpuBlob>(&mut main, &mut render, &device, &queue);
    assert_eq!(second.prepared, 0);
    assert_eq!(second.retried, 0, "the removed retry was discarded");
    assert_eq!(
        attempts.load(Ordering::SeqCst),
        1,
        "removed assets are not retried"
    );
    assert!(render
        .resource::<RenderAssets<RetryOnceGpuBlob>>()
        .get(id)
        .is_none());
    assert!(render
        .resource::<ExtractedAssets<RetryOnceGpuBlob>>()
        .is_empty());
}

#[test]
fn duplicate_extractions_use_the_latest_copy_and_one_retry() {
    let (device, queue) = null_device();
    let id = AssetId::<Blob>::new(AssetIndex::from_parts(0, 1));
    let mut extracted = ExtractedAssets::<GpuBlob>::new();
    let mut prepared = RenderAssets::<GpuBlob>::new();

    extracted.push_extracted(id, Blob(alloc::vec![1, 2, 3]));
    extracted.push_extracted(id, Blob(Vec::new()));
    let report =
        prepare_extracted_assets::<GpuBlob>(&mut extracted, &mut prepared, &device, &queue);

    assert_eq!(report.prepared, 0, "the latest copy is deferred");
    assert_eq!(
        report.retried, 1,
        "duplicate copies collapse to one attempt"
    );
    assert_eq!(extracted.extracted().len(), 1);
    assert_eq!(extracted.extracted()[0].1, Blob(Vec::new()));
    assert!(!prepared.contains(id));
}

#[test]
fn contextual_budget_and_cancel_roll_back_and_retry_without_publishing() {
    let (device, queue) = null_device();
    let id = AssetId::<Blob>::new(AssetIndex::from_parts(0, 1));
    let mut extracted = ExtractedAssets::<TransactionalBlob>::new();
    let mut prepared = RenderAssets::<TransactionalBlob>::new();
    extracted.push_extracted(id, Blob(vec![1, 2, 3, 4]));
    let mut budget = GpuPrepareBudget::new(3);
    let cancellation = AtomicBool::new(false);
    let report = prepare_extracted_assets_with_context(
        &mut extracted,
        &mut prepared,
        &device,
        &queue,
        &mut budget,
        &cancellation,
    );
    assert_eq!(report.retried, 1);
    assert_eq!(
        budget.used(),
        0,
        "failed allocation reservation is refunded"
    );
    assert_eq!(
        device.live_resource_count(),
        0,
        "budget failure created no buffer"
    );
    assert!(!prepared.contains(id));

    budget = GpuPrepareBudget::new(8);
    cancellation.store(true, Ordering::Release);
    let report = prepare_extracted_assets_with_context(
        &mut extracted,
        &mut prepared,
        &device,
        &queue,
        &mut budget,
        &cancellation,
    );
    assert_eq!(report.retried, 1);
    assert_eq!(budget.used(), 0);
    assert_eq!(device.live_resource_count(), 0);

    cancellation.store(false, Ordering::Release);
    let report = prepare_extracted_assets_with_context(
        &mut extracted,
        &mut prepared,
        &device,
        &queue,
        &mut budget,
        &cancellation,
    );
    assert_eq!(report.prepared, 1);
    assert_eq!(device.live_resource_count(), 1);
}

#[test]
fn contextual_replacement_unloads_old_representation_once() {
    let (device, queue) = null_device();
    let id = AssetId::<Blob>::new(AssetIndex::from_parts(0, 1));
    let mut extracted = ExtractedAssets::<TransactionalBlob>::new();
    let mut prepared = RenderAssets::<TransactionalBlob>::new();
    let cancellation = AtomicBool::new(false);
    let mut budget = GpuPrepareBudget::new(32);
    extracted.push_extracted(id, Blob(vec![1]));
    let first = prepare_extracted_assets_with_context(
        &mut extracted,
        &mut prepared,
        &device,
        &queue,
        &mut budget,
        &cancellation,
    );
    assert_eq!(first.prepared, 1);
    assert_eq!(device.live_resource_count(), 1);
    extracted.push_extracted(id, Blob(vec![1, 2, 3]));
    let second = prepare_extracted_assets_with_context(
        &mut extracted,
        &mut prepared,
        &device,
        &queue,
        &mut budget,
        &cancellation,
    );
    assert_eq!(second.prepared, 1);
    assert_eq!(
        device.live_resource_count(),
        1,
        "replacement unloads the old buffer"
    );
    assert_eq!(prepared.get(id).expect("replacement").bytes, 3);
}

/// The extract half: the render world receives an owned copy of what changed,
/// and ids (not copies) of what went away. The simulation keeps its asset.
#[test]
fn extract_copies_changes_and_leaves_the_simulation_asset_alone() {
    let mut main = World::new();
    let mut render = World::new();
    let mut sources = Assets::new();
    let handle = sources.insert(Blob(alloc::vec![4, 5, 6]));
    let id = handle.id();
    main.insert_resource(sources);

    extract_assets::<GpuBlob>(&mut main, &mut render);

    let frame = render.resource::<ExtractedAssets<GpuBlob>>();
    assert_eq!(frame.extracted().len(), 1, "the added asset crossed");
    assert_eq!(frame.extracted()[0].0, id);
    assert_eq!(frame.extracted()[0].1, Blob(alloc::vec![4, 5, 6]));
    assert!(frame.removed().is_empty());
    assert_eq!(frame.len(), 1);
    assert!(
        main.resource::<Assets<Blob>>().contains(id),
        "extraction is a copy: the simulation still owns its asset"
    );
    // The prepare stage consumes the frame each time; simulate that here so the
    // next extract is compared on its own contents.
    let _ = render.resource_mut::<ExtractedAssets<GpuBlob>>().take();

    // A modification crosses as a copy too; a removal crosses as an id.
    main.resource_mut::<Assets<Blob>>()
        .get_mut(id)
        .expect("live")
        .0 = alloc::vec![7];
    extract_assets::<GpuBlob>(&mut main, &mut render);
    let frame = render.resource::<ExtractedAssets<GpuBlob>>();
    assert_eq!(frame.extracted().len(), 1);
    assert_eq!(frame.extracted()[0].1, Blob(alloc::vec![7]));
    let _ = render.resource_mut::<ExtractedAssets<GpuBlob>>().take();

    main.resource_mut::<Assets<Blob>>().remove(id);
    extract_assets::<GpuBlob>(&mut main, &mut render);
    let frame = render.resource::<ExtractedAssets<GpuBlob>>();
    assert!(frame.extracted().is_empty());
    assert_eq!(frame.removed(), &[id]);
}

/// Assets carried by a frame whose prepare never ran are kept, not dropped: the
/// simulation's event queue was already drained, so losing them here would lose
/// the change for good.
#[test]
fn unconsumed_extractions_are_kept_for_the_next_prepare() {
    let mut main = World::new();
    let mut render = World::new();
    let mut sources = Assets::new();
    sources.insert(Blob(alloc::vec![1]));
    main.insert_resource(sources);

    extract_assets::<GpuBlob>(&mut main, &mut render);
    // No prepare stage ran: a second extract must not wipe the first.
    extract_assets::<GpuBlob>(&mut main, &mut render);

    let frame = render.resource::<ExtractedAssets<GpuBlob>>();
    assert_eq!(
        frame.extracted().len(),
        1,
        "the unconsumed extraction is still pending"
    );
}

/// A frame with no asset change carries nothing, and an extract with no arena at
/// all is a no-op rather than a panic (a renderer may be added before the
/// simulation ever inserts the family).
#[test]
fn extract_without_changes_or_without_an_arena_is_a_no_op() {
    let mut main = World::new();
    let mut render = World::new();

    extract_assets::<GpuBlob>(&mut main, &mut render);
    assert!(
        render.get_resource::<ExtractedAssets<GpuBlob>>().is_none(),
        "nothing to extract means nothing to allocate"
    );

    main.insert_resource(Assets::<Blob>::new());
    extract_assets::<GpuBlob>(&mut main, &mut render);
    assert!(render.get_resource::<ExtractedAssets<GpuBlob>>().is_none());
}

/// A registered family is a working pipeline, not just a table: the frame's
/// extract copies the changed asset across, the prepare stage runs it, and a
/// removed asset releases its GPU form — all driven by the app's own update.
#[test]
fn a_registered_family_prepares_and_releases_across_frames() {
    use crate::render_asset::{register_render_assets, AssetSyncStats};

    let mut app = prism_app::App::new();
    app.add_plugins(crate::PrismRenderPlugin::new(NullInstance::new()));
    register_render_assets::<NullInstance, GpuBlob>(&mut app);

    let mut sources = Assets::new();
    let handle = sources.insert(Blob(alloc::vec![1, 2, 3]));
    let id = handle.id();
    app.world_mut().insert_resource(sources);

    app.update();

    let render = app
        .get_sub_app(crate::RenderApp)
        .expect("the render sub-app");
    assert!(
        render
            .world
            .resource::<RenderAssets<GpuBlob>>()
            .contains(id),
        "the frame prepared the added asset"
    );
    assert_eq!(
        render
            .world
            .resource::<AssetSyncStats<GpuBlob>>()
            .last
            .prepared,
        1
    );

    // The source goes away: the next frame releases the GPU form.
    drop(handle);
    app.world_mut().resource_mut::<Assets<Blob>>().remove(id);
    app.update();

    let render = app
        .get_sub_app(crate::RenderApp)
        .expect("the render sub-app");
    assert!(
        !render
            .world
            .resource::<RenderAssets<GpuBlob>>()
            .contains(id),
        "the released asset is out of the table"
    );
    assert_eq!(
        render
            .world
            .resource::<AssetSyncStats<GpuBlob>>()
            .last
            .released,
        1
    );
}

#[test]
fn the_table_exposes_what_it_holds() {
    let (device, _queue) = null_device();
    let mut table = RenderAssets::<GpuBlob>::new();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);

    let first = GpuBlob {
        buffer: device.create_buffer(&BufferDescriptor::new(4, BufferUsages::STORAGE)),
        bytes: 4,
        reused: false,
    };
    let second = GpuBlob {
        buffer: device.create_buffer(&BufferDescriptor::new(8, BufferUsages::STORAGE)),
        bytes: 8,
        reused: false,
    };
    let id_a = AssetId::<Blob>::new(AssetIndex::from_parts(0, 1));
    let id_b = AssetId::<Blob>::new(AssetIndex::from_parts(1, 1));
    assert!(table.insert(id_a, first).is_none(), "a fresh insert");
    assert!(table.insert(id_b, second).is_none());
    assert_eq!(table.len(), 2);
    assert!(table.contains(id_a) && table.contains(id_b));

    table.get_mut(id_a).expect("live").bytes = 16;
    assert_eq!(table.get(id_a).expect("live").bytes, 16);
    let seen: Vec<AssetId<Blob>> = table.iter().map(|(id, _)| id).collect();
    assert_eq!(seen, alloc::vec![id_a, id_b], "iteration is in id order");
    assert_eq!(table.iter_mut().count(), 2);

    let taken = table.remove(id_b).expect("the entry is removable");
    assert_eq!(taken.bytes, 8, "the removed entry is handed back");
    assert_eq!(table.len(), 1);
}
