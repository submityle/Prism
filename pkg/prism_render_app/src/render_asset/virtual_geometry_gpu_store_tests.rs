//! Null-RHI regression tests for CLST GPU publication storage.

use alloc::{sync::Arc, vec, vec::Vec};
use core::any::Any;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use prism_render_architecture::{
    gpu_scene::{SceneBounds, SceneMaterialHandle},
    paging::RequestBatch,
    virtual_geometry::{
        cluster_payload::encode_cluster_payload, ClusterNode, GeometryPageKey,
        VirtualGeometryAsset, VirtualGeometryAssetHeader, VirtualGeometryClusterRecord,
        VirtualGeometryCompression, VirtualGeometryMaterialBinding, VirtualGeometryPage,
        VirtualGeometryPageRecord, VIRTUAL_GEOMETRY_ASSET_SCHEMA,
    },
};
use prism_render_driver::{
    BufferId, CommandBuffer, DeviceDescriptor, FrameSync, ImageDataLayout, Instance, RenderQueue,
    TextureWrite,
};
use prism_render_driver_null::{NullDevice, NullInstance, NullQueue};

use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Write {
    buffer: BufferId,
    offset: u64,
    data: Vec<u8>,
}

#[derive(Default)]
struct RecordingNullQueue {
    writes: Mutex<Vec<Write>>,
    idle_confirmed: AtomicBool,
}

impl RecordingNullQueue {
    fn take_writes(&self) -> Vec<Write> {
        core::mem::take(&mut *self.writes.lock().expect("write log lock"))
    }
}

impl RenderQueue for RecordingNullQueue {
    fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) {
        self.writes.lock().expect("write log lock").push(Write {
            buffer,
            offset,
            data: data.to_vec(),
        });
        NullQueue::new().write_buffer(buffer, offset, data);
    }

    fn write_texture(&self, destination: TextureWrite, data: &[u8], layout: ImageDataLayout) {
        NullQueue::new().write_texture(destination, data, layout);
    }

    fn submit(&self, command_buffers: &[CommandBuffer]) {
        NullQueue::new().submit(command_buffers);
    }
}

impl FrameSync for RecordingNullQueue {
    fn submit_tracked(&self, command_buffers: &[CommandBuffer]) -> Box<dyn Any + Send + Sync> {
        NullQueue::new().submit_tracked(command_buffers)
    }

    fn wait_for(&self, token: &(dyn Any + Send + Sync)) -> bool {
        NullQueue::new().wait_for(token)
    }

    fn wait_idle(&self) -> bool {
        self.idle_confirmed.load(Ordering::Acquire) && NullQueue::new().wait_idle()
    }

    fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
        NullQueue::new().on_submitted_work_done(callback);
    }
}

fn device() -> NullDevice {
    let instance = NullInstance::new();
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("Null RHI exposes its test adapter");
    instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("Null RHI device")
        .0
}

fn bounds() -> SceneBounds {
    SceneBounds {
        center: [0.0; 3],
        radius: 1.0,
        half_extents: [1.0; 3],
        _padding: 0.0,
    }
}

fn asset() -> Arc<VirtualGeometryAsset> {
    let bounds = bounds();
    let payload = encode_cluster_payload(
        bounds,
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        &[0, 1, 2],
    )
    .expect("valid CLST page payload");
    let payload_size = payload.len() as u32;
    let first = GeometryPageKey::new(42, 0);
    let second = GeometryPageKey::new(42, 1);
    let asset = VirtualGeometryAsset {
        header: VirtualGeometryAssetHeader {
            asset_id: 42,
            schema_version: VIRTUAL_GEOMETRY_ASSET_SCHEMA,
            page_size: 1024,
            build_settings_hash: [1; 32],
            content_hash: [2; 32],
            bounds,
            material_slot_count: 1,
            fallback_mesh_hash: [3; 32],
        },
        nodes: vec![ClusterNode::leaf(bounds, 0.0, first)],
        roots: vec![0],
        pages: vec![
            VirtualGeometryPage {
                record: VirtualGeometryPageRecord {
                    key: first,
                    compression: VirtualGeometryCompression::None,
                    uncompressed_size: payload_size,
                    priority: 1,
                    cluster_start: 0,
                    cluster_count: 1,
                },
                payload: payload.clone(),
            },
            VirtualGeometryPage {
                record: VirtualGeometryPageRecord {
                    key: second,
                    compression: VirtualGeometryCompression::None,
                    uncompressed_size: payload_size,
                    priority: 1,
                    cluster_start: 1,
                    cluster_count: 0,
                },
                payload,
            },
        ],
        clusters: vec![VirtualGeometryClusterRecord {
            node_index: 0,
            page: first,
            material_slot: 0,
            vertex_count: 3,
            triangle_count: 1,
            payload_offset: 0,
            payload_size,
            bounds,
            normal_cone: [0.0, 0.0, 1.0, 1.0],
        }],
    };
    asset.validate().expect("valid two-page fixture");
    Arc::new(asset)
}

fn register(
    store: &mut VirtualGeometryGpuPublicationStore<NullInstance>,
    queue: &dyn RenderQueue,
) -> VirtualGeometryGpuBuffers {
    store
        .register(
            asset(),
            [5; 32],
            7,
            1,
            1,
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: SceneMaterialHandle::new(2, 3),
            }],
            queue,
        )
        .expect("register CLST asset")
}

#[test]
fn null_rhi_registration_zeroes_physical_storage_before_tables() {
    let device = Arc::new(device());
    let queue = RecordingNullQueue::default();
    let mut store = VirtualGeometryGpuPublicationStore::<NullInstance> {
        device: Arc::clone(&device),
        assets: BTreeMap::new(),
    };
    let buffers = register(&mut store, &queue);
    let writes = queue.take_writes();

    assert_eq!(writes.len(), 3);
    assert_eq!(writes[0].buffer, buffers.page_storage);
    assert_eq!(writes[0].offset, 0);
    assert_eq!(writes[0].data.len(), 1024);
    assert!(writes[0].data.iter().all(|byte| *byte == 0));
    assert_eq!(writes[1].buffer, buffers.page_table);
    assert_eq!(
        writes[1].data.len(),
        2 * core::mem::size_of::<prism_render_architecture::virtual_geometry::VirtualGeometryGpuPage>(
        )
    );
    assert_eq!(writes[2].buffer, buffers.cluster_table);
    assert_eq!(device.live_resource_count(), 3);
}

#[test]
fn reconcile_orders_storage_before_table_clears_evictions_and_reuses_slots() {
    let device = Arc::new(device());
    let queue = RecordingNullQueue::default();
    let mut store = VirtualGeometryGpuPublicationStore::<NullInstance> {
        device: Arc::clone(&device),
        assets: BTreeMap::new(),
    };
    let buffers = register(&mut store, &queue);
    queue.take_writes();

    let first = GeometryPageKey::new(42, 0);
    let second = GeometryPageKey::new(42, 1);
    let mut requests = RequestBatch::new();
    requests.record(first, 1.0);
    let admitted = store
        .reconcile(42, &requests, 1, &queue)
        .expect("first page admission");
    assert_eq!(admitted.streamed_in[0].physical_page, 0);
    let first_writes = queue.take_writes();
    assert_eq!(first_writes.len(), 2);
    assert_eq!(first_writes[0].buffer, buffers.page_storage);
    assert_eq!(first_writes[0].offset, 0);
    assert_eq!(first_writes[0].data.len(), 1024);
    assert_eq!(first_writes[1].buffer, buffers.page_table);

    let mut requests = RequestBatch::new();
    requests.record(second, 2.0);
    let replaced = store
        .reconcile(42, &requests, 2, &queue)
        .expect("second page replaces first");
    assert_eq!(replaced.evicted[0].physical_page, 0);
    assert_eq!(replaced.streamed_in[0].physical_page, 0);
    assert!(!store.publication(42).expect("publication").pages()[0].is_resident());
    assert!(store.publication(42).expect("publication").pages()[1].is_resident());

    let writes = queue.take_writes();
    assert_eq!(writes.len(), 4);
    assert_eq!(writes[0].buffer, buffers.page_storage);
    assert_eq!(writes[0].offset, 0);
    assert!(writes[0].data.iter().all(|byte| *byte == 0));
    assert_eq!(writes[1].buffer, buffers.page_storage);
    assert_eq!(writes[1].offset, 0);
    assert_eq!(writes[2].buffer, buffers.page_table);
    assert_eq!(writes[3].buffer, buffers.page_table);
}

#[test]
fn failed_registration_and_reconcile_do_not_change_owned_state() {
    let device = Arc::new(device());
    let queue = RecordingNullQueue::default();
    let mut store = VirtualGeometryGpuPublicationStore::<NullInstance> {
        device: Arc::clone(&device),
        assets: BTreeMap::new(),
    };
    let result = store.register(asset(), [5; 32], 0, 1, 1, &[], &queue);
    assert!(matches!(
        result,
        Err(VirtualGeometryGpuStoreError::Publication(_))
    ));
    assert_eq!(device.live_resource_count(), 0);
    assert!(queue.take_writes().is_empty());

    let buffers = register(&mut store, &queue);
    queue.take_writes();
    let before = store.publication(42).expect("publication").clone();
    assert_eq!(
        store.reconcile(999, &RequestBatch::new(), 1, &queue),
        Err(VirtualGeometryGpuStoreError::UnknownAsset(999))
    );
    assert_eq!(store.publication(42), Some(&before));
    assert!(queue.take_writes().is_empty());
    assert_eq!(store.buffers(42), Some(buffers));
}

#[test]
fn store_drop_releases_all_null_rhi_buffers() {
    let device = Arc::new(device());
    let queue = RecordingNullQueue::default();
    let mut store = VirtualGeometryGpuPublicationStore::<NullInstance> {
        device: Arc::clone(&device),
        assets: BTreeMap::new(),
    };
    register(&mut store, &queue);
    assert_eq!(device.live_resource_count(), 3);
    drop(store);
    assert_eq!(device.live_resource_count(), 0);
}

#[test]
fn retire_requires_confirmed_completion_and_is_idempotent() {
    let device = Arc::new(device());
    let queue = RecordingNullQueue::default();
    let mut store = VirtualGeometryGpuPublicationStore::<NullInstance> {
        device: Arc::clone(&device),
        assets: BTreeMap::new(),
    };
    let buffers = register(&mut store, &queue);

    assert_eq!(
        store.retire(&queue),
        Err(VirtualGeometryGpuStoreError::QueueCompletionUnconfirmed)
    );
    assert_eq!(device.live_resource_count(), 3);
    assert_eq!(store.buffers(42), Some(buffers));

    queue.idle_confirmed.store(true, Ordering::Release);
    store.retire(&queue).expect("confirmed retirement");
    assert_eq!(device.live_resource_count(), 0);
    assert_eq!(store.buffers(42), None);
    store.retire(&queue).expect("empty-store retirement");
}
