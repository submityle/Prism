//! Behavioural tests for the render-asset table and its sync.

use alloc::vec::Vec;

use prism_asset::{Asset, AssetEvent, AssetId, Assets};
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, DeviceDescriptor, Instance, RenderDevice,
    RequestAdapterOptions,
};
use prism_render_driver_null::{NullDevice, NullInstance};

use super::{PrepareAssetError, RenderAsset, RenderAssets, sync_assets};
use super::prepare::drain_and_sync_assets;

/// A source asset: raw bytes with no GPU meaning of its own.
#[derive(Clone, Debug, PartialEq)]
struct Blob(Vec<u8>);

impl Asset for Blob {
    const TYPE_NAME: &'static str = "prism_render_app::Blob";
}

/// The GPU form: one real RHI buffer, so the table's lifecycle is observable
/// through the device (create on prepare, destroy on unload).
struct GpuBlob {
    buffer: BufferId,
    bytes: u64,
    /// Whether this prepare reused the previous representation's buffer.
    reused: bool,
}

impl RenderAsset for GpuBlob {
    type SourceAsset = Blob;

    fn prepare(
        source: &Blob,
        id: AssetId<Blob>,
        device: &dyn RenderDevice,
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

fn null_device() -> NullDevice {
    let options = RequestAdapterOptions::default();
    let adapter = NullInstance::new()
        .request_adapter(&options)
        .expect("the null backend always has an adapter");
    let (device, _queue) = NullInstance::new()
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null device always comes up");
    device
}

#[test]
fn preparing_an_asset_creates_its_gpu_representation() {
    let device = null_device();
    let mut sources = Assets::new();
    let mut table = RenderAssets::<GpuBlob>::new();
    let handle = sources.insert(Blob(alloc::vec![1, 2, 3]));
    let id = handle.id();

    let report = drain_and_sync_assets(&mut sources, &mut table, &device);

    assert_eq!(report.prepared, 1, "the added asset is prepared");
    assert_eq!(report.released, 0);
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
fn re_preparing_reuses_the_buffer_it_already_has() {
    let device = null_device();
    let mut sources = Assets::new();
    let mut table = RenderAssets::<GpuBlob>::new();
    let handle = sources.insert(Blob(alloc::vec![1, 2, 3, 4]));
    let id = handle.id();
    drain_and_sync_assets(&mut sources, &mut table, &device);
    let first = table.get(id).expect("prepared").buffer;

    // Modify the source: a payload that still fits the existing buffer.
    sources.get_mut(id).expect("the asset is live").0 = alloc::vec![9, 9];
    let report = drain_and_sync_assets(&mut sources, &mut table, &device);

    assert_eq!(report.prepared, 1, "the modification re-prepares");
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
    let device = null_device();
    let mut sources = Assets::new();
    let mut table = RenderAssets::<GpuBlob>::new();
    let handle = sources.insert(Blob(alloc::vec![1, 2, 3]));
    let id = handle.id();
    drain_and_sync_assets(&mut sources, &mut table, &device);
    assert_eq!(device.live_resource_count(), 1);

    drop(handle);
    sources.remove(id);
    let report = drain_and_sync_assets(&mut sources, &mut table, &device);

    assert_eq!(report.released, 1, "the removal releases the representation");
    assert!(table.get(id).is_none(), "the table no longer holds it");
    assert_eq!(
        device.live_resource_count(),
        0,
        "the GPU buffer was destroyed, not leaked"
    );
}

#[test]
fn a_deferred_prepare_keeps_the_previous_representation() {
    let device = null_device();
    let mut sources = Assets::new();
    let mut table = RenderAssets::<GpuBlob>::new();
    let handle = sources.insert(Blob(alloc::vec![1, 2, 3]));
    let id = handle.id();
    drain_and_sync_assets(&mut sources, &mut table, &device);
    let prepared = table.get(id).expect("prepared").buffer;

    // The source is now un-prepared-able (a dependency went away, say).
    sources.get_mut(id).expect("the asset is live").0.clear();
    let report = drain_and_sync_assets(&mut sources, &mut table, &device);

    assert_eq!(report.retried, 1, "the prepare asked to retry");
    assert_eq!(report.prepared, 0);
    assert_eq!(
        table.get(id).expect("still prepared").buffer,
        prepared,
        "a retry never publishes a half-prepared asset"
    );

    // Once the source is usable again, the next pass publishes the new form.
    sources.get_mut(id).expect("the asset is live").0 = alloc::vec![7; 8];
    let report = drain_and_sync_assets(&mut sources, &mut table, &device);
    assert_eq!(report.prepared, 1, "the retry succeeded on the next pass");
    assert_eq!(table.get(id).expect("prepared").bytes, 8);
}

#[test]
fn a_new_asset_whose_prepare_defers_is_not_in_the_table() {
    let device = null_device();
    let mut sources = Assets::new();
    let mut table = RenderAssets::<GpuBlob>::new();
    let id = sources.insert(Blob(Vec::new())).id();

    let report = drain_and_sync_assets(&mut sources, &mut table, &device);

    assert_eq!(report.retried, 1);
    assert!(
        !table.contains(id),
        "a deferred asset must not appear as prepared"
    );
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn events_that_do_not_change_the_gpu_form_are_ignored() {
    let device = null_device();
    let mut sources = Assets::new();
    let mut table = RenderAssets::<GpuBlob>::new();
    let id = sources.insert(Blob(alloc::vec![1])).id();
    drain_and_sync_assets(&mut sources, &mut table, &device);

    let report = sync_assets(
        &sources,
        &mut table,
        &device,
        [
            AssetEvent::Unused { id },
            AssetEvent::LoadedWithDependencies { id },
        ],
    );

    assert_eq!(report, super::AssetSyncReport::default());
    assert!(table.contains(id), "the prepared form is untouched");
    assert_eq!(device.live_resource_count(), 1);
}

#[test]
fn the_table_exposes_what_it_holds() {
    let device = null_device();
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
    let id_a = AssetId::<Blob>::new(prism_asset::AssetIndex::from_parts(0, 1));
    let id_b = AssetId::<Blob>::new(prism_asset::AssetIndex::from_parts(1, 1));
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
