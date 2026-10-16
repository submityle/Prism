//! Deterministic discovery-complexity and payload regressions, without a GPU.
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use prism_app::App;
use prism_ecs::archetype::{ArchetypeId, COLUMN_CHANGE_LOG_CAPACITY};
use prism_ecs::component::{shared_box_of, Component, ComponentId, Components, StorageType};
use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_render_app::{
    EcsColumnDiscoveryStats, EcsColumnUploadError, EcsColumnUploadStats, EcsResidentColumns,
    GpuColumnAppExt, GpuColumnSyncReport, PrismRenderPlugin, RenderApp, RenderAppConfig,
};
use prism_render_driver::{
    BufferId, CommandBuffer, DeviceDescriptor, Extent3d, ImageDataLayout, Instance, RenderQueue,
    RequestAdapterOptions, TextureWrite,
};
use prism_render_driver_null::{NullDevice, NullInstance};

// 32 rows per chunk alone, 16 alongside Extra. Vary every lane to detect bad
// offsets, stale bytes, and incorrect column addressing (not just copy counts).
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Row([u32; 128]);
impl Component for Row {}

fn row(seed: u32) -> Row {
    Row(core::array::from_fn(|lane| {
        seed.wrapping_mul(1009) ^ lane as u32
    }))
}

#[derive(Clone, Copy)]
struct Extra([u32; 128]);
impl Component for Extra {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Marker;
impl Component for Marker {}

// Arbitrarily many archetypes without arbitrarily many Rust component types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BatchKey(u32);
impl Component for BatchKey {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut Components, id: ComponentId) {
        components.set_shared_box(id, shared_box_of::<Self>());
    }
}

#[derive(Default)]
struct RecordingQueue {
    buffers: Mutex<HashMap<BufferId, Vec<u8>>>,
    writes: Mutex<Vec<(BufferId, u64, usize)>>,
}

impl RenderQueue for RecordingQueue {
    fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) {
        assert_eq!(offset % 4, 0);
        assert_eq!(data.len() % 4, 0);
        let mut buffers = self.buffers.lock().unwrap();
        let bytes = buffers.entry(buffer).or_default();
        let start = offset as usize;
        bytes.resize(bytes.len().max(start + data.len()), 0);
        bytes[start..start + data.len()].copy_from_slice(data);
        self.writes
            .lock()
            .unwrap()
            .push((buffer, offset, data.len()));
    }

    fn write_texture(&self, _: TextureWrite, _: &[u8], _: ImageDataLayout) {
        panic!("column synchronization must not upload textures");
    }

    fn submit(&self, _: &[CommandBuffer]) {
        panic!("submission belongs to the driver, not column synchronization");
    }
}

fn null() -> (Arc<NullDevice>, RecordingQueue) {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .unwrap();
    let (device, _) = instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .unwrap();
    (Arc::new(device), RecordingQueue::default())
}

fn arch(world: &World, entity: Entity) -> ArchetypeId {
    world.entities().location(entity).unwrap().archetype_id
}

// Order: journal events, discovery table visits, chunk candidates, typed slices,
// explicit full-registry baselines. Payload addressing is counted separately.
fn work<T: Component + Pod>(cache: &EcsResidentColumns<T>) -> (usize, usize, usize, usize, usize) {
    let d: EcsColumnDiscoveryStats = cache.discovery_stats();
    (
        d.events_read,
        d.archetypes_visited,
        d.chunks_visited,
        d.payload_chunks_read,
        d.full_rebuilds,
    )
}

fn clean<T: Component + Pod>(
    cache: &mut EcsResidentColumns<T>,
    queue: &RecordingQueue,
    world: &World,
) {
    let writes = queue.writes.lock().unwrap().len();
    assert_eq!(
        cache.sync(queue, world).unwrap(),
        EcsColumnUploadStats::default()
    );
    assert_eq!(work(cache), (0, 0, 0, 0, 0));
    assert_eq!(queue.writes.lock().unwrap().len(), writes);
}

fn verify(cache: &EcsResidentColumns<Row>, queue: &RecordingQueue, world: &World, id: ArchetypeId) {
    let view = cache.column(id).unwrap();
    let table = world.archetypes().get(id).unwrap().table();
    assert_eq!(view.rows, table.len());
    let expected: Vec<u8> = table
        .entities()
        .iter()
        .flat_map(|&entity| {
            bytemuck::bytes_of(world.get::<Row>(entity).unwrap())
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(view.live_bytes() as usize, expected.len());
    if !expected.is_empty() {
        let buffers = queue.buffers.lock().unwrap();
        assert_eq!(
            &buffers[&view.buffer_id().unwrap()][..expected.len()],
            expected
        );
    }
}

#[test]
fn discovery_work_scales_with_delta_not_archetypes_chunks_or_typed_slices() {
    // Increase the archetype registry and per-table physical chunk count
    // independently; large case has 32,768 rows and 2,048 live typed slices.
    for (archetypes, chunks_per_table) in [(1, 1), (16, 8), (128, 16)] {
        let (device, queue) = null();
        let mut world = World::new();
        let mut last = None;
        for key in 0..archetypes {
            for index in 0..chunks_per_table * 16 {
                last = Some(world.spawn((row(index), Extra([0; 128]), BatchKey(key))));
            }
        }
        let entity = last.unwrap();
        let id = arch(&world, entity);
        let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
        let baseline = cache.sync(&queue, &world).unwrap();
        assert_eq!(
            baseline.upload_calls,
            (archetypes * chunks_per_table) as usize
        );
        assert_eq!(work(&cache).3, baseline.upload_calls);
        assert_eq!(work(&cache).4, 1);
        for _ in 0..8 {
            world.increment_change_tick();
            clean(&mut cache, &queue, &world);
        }

        *world.get_mut::<Row>(entity).unwrap() = row(999_999);
        assert_eq!(
            cache.sync(&queue, &world).unwrap(),
            EcsColumnUploadStats {
                allocations: 0,
                upload_calls: 1,
                uploaded_bytes: 16 * 512,
            }
        );
        assert_eq!(work(&cache), (1, 2, 1, 1, 0));
        verify(&cache, &queue, &world, id);
        let writes = queue.writes.lock().unwrap();
        assert_eq!(
            writes.last().unwrap().1,
            u64::from(chunks_per_table - 1) * 16 * 512
        );
        drop(writes);

        // Repeated same-tick writes are journaled but payload chunks deduplicate.
        for seed in 0..16 {
            *world.get_mut::<Row>(entity).unwrap() = row(seed);
        }
        assert_eq!(cache.sync(&queue, &world).unwrap().upload_calls, 1);
        assert_eq!(work(&cache), (16, 2, 1, 1, 0));
        verify(&cache, &queue, &world, id);

        // Unrelated writes must not inspect even the table containing T.
        world.get_mut::<Extra>(entity).unwrap().0[0] = 1;
        assert_eq!(
            cache.sync(&queue, &world).unwrap(),
            EcsColumnUploadStats::default()
        );
        assert_eq!(work(&cache), (1, 0, 0, 0, 0));
        clean(&mut cache, &queue, &world);
        eprintln!("scale: {archetypes} archetypes x {chunks_per_table} chunks; clean=(0,0,0,0,0), single=(1,2,1,1,0)");
    }
}

#[test]
fn prewarm_reservations_are_not_live_gpu_capacity_or_baseline_candidates() {
    let (device, queue) = null();
    let mut world = World::new();
    let id = world.prewarm::<Row>(32 * 1024);
    let table = world.archetypes().get(id).unwrap().table();
    assert_eq!(
        table.chunk_count(),
        0,
        "logical windows count only live rows"
    );
    assert_eq!(
        table.iter_chunks().count(),
        1024,
        "physical reservation is real"
    );
    let mut cache = EcsResidentColumns::<Row>::new(device.clone(), &mut world).unwrap();
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
    assert_eq!((work(&cache).2, work(&cache).3, work(&cache).4), (0, 0, 1));
    assert_eq!(device.live_resource_count(), 0);
    assert_eq!(cache.column(id).unwrap().capacity_bytes(), 0);
    clean(&mut cache, &queue, &world);

    let entity = world.spawn(row(7));
    assert_eq!(arch(&world, entity), id);
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 1,
            upload_calls: 1,
            uploaded_bytes: 512,
        }
    );
    assert_eq!(cache.column(id).unwrap().capacity_bytes(), 32 * 512);
    assert_eq!((work(&cache).2, work(&cache).3), (1, 1));
    verify(&cache, &queue, &world, id);

    // Extending an existing reservation is not a live membership mutation.
    world.prewarm::<Row>(32 * 2048);
    clean(&mut cache, &queue, &world);
    cache.reset();
    cache.sync(&queue, &world).unwrap();
    assert_eq!((work(&cache).2, work(&cache).3, work(&cache).4), (1, 1, 1));
    assert_eq!(cache.column(id).unwrap().capacity_bytes(), 32 * 512);
    verify(&cache, &queue, &world, id);
    clean(&mut cache, &queue, &world);
}

#[test]
fn independent_cursors_observe_same_tick_backdated_and_failed_world_syncs() {
    let (device, queue) = null();
    let mut world = World::new();
    let entities: Vec<_> = (0..65).map(|i| world.spawn(row(i))).collect();
    let id = arch(&world, entities[0]);
    let historical = world.change_tick();
    let mut fast = EcsResidentColumns::<Row>::new(device.clone(), &mut world).unwrap();
    let mut slow = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    fast.sync(&queue, &world).unwrap();
    slow.sync(&queue, &world).unwrap();
    let other = World::new();
    for (index, seed) in [(40, 100), (40, 200), (64, 300)] {
        *world.get_mut::<Row>(entities[index]).unwrap() = row(seed);
        assert_eq!(
            fast.sync(&queue, &other),
            Err(EcsColumnUploadError::WrongWorld)
        );
        assert_eq!(fast.sync(&queue, &world).unwrap().upload_calls, 1);
        assert_eq!(work(&fast), (1, 2, 1, 1, 0));
        verify(&fast, &queue, &world, id);
        clean(&mut fast, &queue, &world);
    }
    assert_eq!(world.change_tick(), historical);
    assert_eq!(slow.sync(&queue, &world).unwrap().upload_calls, 2);
    assert_eq!(work(&slow), (3, 2, 2, 2, 0));
    verify(&slow, &queue, &world, id);

    for _ in 0..2 {
        world.increment_change_tick();
        let mut value = world.get_mut::<Row>(entities[40]).unwrap();
        *value.bypass_change_detection() = row(123_456);
        value.set_last_changed(historical);
        drop(value);
        for cache in [&mut fast, &mut slow] {
            assert_eq!(cache.sync(&queue, &world).unwrap().upload_calls, 1);
            assert_eq!(work(cache), (1, 2, 1, 1, 0));
            verify(cache, &queue, &world, id);
            clean(cache, &queue, &world);
        }
    }
}

#[test]
fn retained_log_boundary_is_incremental_and_overflow_rebuilds_once() {
    let (device, queue) = null();
    let mut world = World::new();
    let entities: Vec<_> = (0..65).map(|i| world.spawn(row(i))).collect();
    let id = arch(&world, entities[0]);
    let mut fast = EcsResidentColumns::<Row>::new(device.clone(), &mut world).unwrap();
    let mut lagging = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    fast.sync(&queue, &world).unwrap();
    lagging.sync(&queue, &world).unwrap();
    for seed in 0..COLUMN_CHANGE_LOG_CAPACITY {
        *world.get_mut::<Row>(entities[64]).unwrap() = row(seed as u32);
    }
    assert_eq!(fast.sync(&queue, &world).unwrap().upload_calls, 1);
    assert_eq!(work(&fast), (COLUMN_CHANGE_LOG_CAPACITY, 2, 1, 1, 0));
    *world.get_mut::<Row>(entities[64]).unwrap() = row(888_888);
    assert_eq!(
        lagging.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 3,
            uploaded_bytes: 65 * 512,
        }
    );
    assert_eq!(
        (
            work(&lagging).0,
            work(&lagging).2,
            work(&lagging).3,
            work(&lagging).4
        ),
        (0, 3, 3, 1)
    );
    verify(&lagging, &queue, &world, id);
    clean(&mut lagging, &queue, &world);
    assert_eq!(fast.sync(&queue, &world).unwrap().upload_calls, 1);
    assert_eq!(work(&fast), (1, 2, 1, 1, 0));
    verify(&fast, &queue, &world, id);
    clean(&mut fast, &queue, &world);
}

#[test]
fn swap_remove_migration_and_empty_tables_update_only_membership_candidates() {
    let (device, queue) = null();
    let mut world = World::new();
    let entities: Vec<_> = (0..65).map(|i| world.spawn(row(i))).collect();
    let source = arch(&world, entities[0]);
    let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    let original = cache.column(source).unwrap();
    assert!(world.despawn(entities[0])); // swaps the tail into row 0 and frees chunk 2
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 32 * 512,
        }
    );
    assert_eq!(
        (
            work(&cache).1,
            work(&cache).2,
            work(&cache).3,
            work(&cache).4
        ),
        (2, 2, 1, 0)
    );
    verify(&cache, &queue, &world, source);

    assert!(world.insert(entities[1], Extra([1; 128])));
    let destination = arch(&world, entities[1]);
    cache.sync(&queue, &world).unwrap();
    assert_eq!(
        (
            work(&cache).1,
            work(&cache).2,
            work(&cache).3,
            work(&cache).4
        ),
        (4, 3, 3, 0)
    );
    verify(&cache, &queue, &world, source);
    verify(&cache, &queue, &world, destination);
    assert!(world.remove::<Extra>(entities[1]));
    cache.sync(&queue, &world).unwrap();
    verify(&cache, &queue, &world, source);
    verify(&cache, &queue, &world, destination);
    assert_eq!(
        cache.column(source).unwrap().buffer_id(),
        original.buffer_id()
    );
    assert_eq!(
        cache.column(source).unwrap().capacity_bytes(),
        original.capacity_bytes()
    );

    for &entity in &entities[1..] {
        assert!(world.despawn(entity));
    }
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
    assert_eq!(
        (
            work(&cache).1,
            work(&cache).2,
            work(&cache).3,
            work(&cache).4
        ),
        (2, 2, 0, 0)
    );
    assert_eq!(cache.column(source).unwrap().rows, 0);
    assert_eq!(
        cache.column(source).unwrap().buffer_id(),
        original.buffer_id()
    );
    clean(&mut cache, &queue, &world);
}

#[test]
fn recreated_block_with_equal_entities_rows_tick_and_epoch_is_not_clean() {
    let (device, queue) = null();
    let mut world = World::new();
    let entities: Vec<_> = (0..32).map(|i| world.spawn(row(i))).collect();
    let id = arch(&world, entities[0]);
    let stamp = |world: &World| {
        let chunk = world
            .archetypes()
            .get(id)
            .unwrap()
            .table()
            .chunk_info(0)
            .unwrap();
        (chunk.incarnation, chunk.meta.structural_epoch())
    };
    let before = stamp(&world);
    let tick = world.change_tick();
    let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    let retained = cache.column(id).unwrap();
    for &entity in entities.iter().rev() {
        assert!(world.insert(entity, Extra([1; 128])));
    }
    *world.get_mut::<Row>(entities[17]).unwrap() = row(777_777);
    for &entity in &entities {
        assert!(world.remove::<Extra>(entity));
    }
    assert_eq!(
        world.archetypes().get(id).unwrap().table().entities(),
        entities
    );
    assert_eq!(world.change_tick(), tick);
    assert_ne!(stamp(&world).0, before.0);
    assert_eq!(stamp(&world).1, before.1);
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 32 * 512,
        }
    );
    // The newly seen intermediate archetype is empty at extraction, so its
    // baseline has no live windows; only the recreated source block is fetched.
    assert_eq!((work(&cache).2, work(&cache).3, work(&cache).4), (1, 1, 0));
    assert_eq!(cache.column(id).unwrap().buffer_id(), retained.buffer_id());
    verify(&cache, &queue, &world, id);
    clean(&mut cache, &queue, &world);
}

#[test]
fn growth_initializes_all_live_chunks_and_tail_recreation_preserves_clean_front() {
    let (device, queue) = null();
    let mut world = World::new();
    let mut entities: Vec<_> = (0..32).map(|i| world.spawn(row(i))).collect();
    let id = arch(&world, entities[0]);
    let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    let old = cache.column(id).unwrap();
    let old_bytes = queue.buffers.lock().unwrap()[&old.buffer_id().unwrap()].clone();
    entities.extend((32..64).map(|i| world.spawn(row(i))));
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 1,
            upload_calls: 2,
            uploaded_bytes: 64 * 512,
        }
    );
    assert_eq!((work(&cache).2, work(&cache).3, work(&cache).4), (2, 2, 0));
    let view = cache.column(id).unwrap();
    assert!(view.generation() > old.generation());
    assert_eq!(
        queue.buffers.lock().unwrap()[&old.buffer_id().unwrap()],
        old_bytes
    );
    verify(&cache, &queue, &world, id);
    let sentinel = row(333_333);
    queue.write_buffer(
        view.buffer_id().unwrap(),
        5 * 512,
        bytemuck::bytes_of(&sentinel),
    );
    let front = world
        .archetypes()
        .get(id)
        .unwrap()
        .table()
        .chunk_info(0)
        .unwrap()
        .incarnation;
    let tail = world
        .archetypes()
        .get(id)
        .unwrap()
        .table()
        .chunk_info(1)
        .unwrap()
        .incarnation;
    for &entity in entities[32..].iter().rev() {
        assert!(world.despawn(entity));
    }
    for i in 32..64 {
        world.spawn(row(1000 + i));
    }
    assert_eq!(
        world
            .archetypes()
            .get(id)
            .unwrap()
            .table()
            .chunk_info(0)
            .unwrap()
            .incarnation,
        front
    );
    assert_ne!(
        world
            .archetypes()
            .get(id)
            .unwrap()
            .table()
            .chunk_info(1)
            .unwrap()
            .incarnation,
        tail
    );
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 32 * 512,
        }
    );
    assert_eq!(
        (
            work(&cache).1,
            work(&cache).2,
            work(&cache).3,
            work(&cache).4
        ),
        (2, 1, 1, 0)
    );
    let buffers = queue.buffers.lock().unwrap();
    let bytes = &buffers[&view.buffer_id().unwrap()];
    assert_eq!(&bytes[5 * 512..6 * 512], bytemuck::bytes_of(&sentinel));
    assert_eq!(
        &bytes[32 * 512..64 * 512],
        bytemuck::cast_slice(world.column_chunk::<Row>(id, 1).unwrap())
    );
    drop(buffers);
    clean(&mut cache, &queue, &world);
}

#[test]
fn restore_discards_abandoned_archetypes_and_resumes_incremental_discovery() {
    let (device, queue) = null();
    let mut world = World::new();
    world.register_snapshot_component::<Row>();
    let entities: Vec<_> = (0..35).map(|i| world.spawn(row(i))).collect();
    let snapshot = world.snapshot();
    let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    *world.get_mut::<Row>(entities[10]).unwrap() = row(12345);
    let abandoned = world.spawn((row(99), Extra([0; 128])));
    let abandoned_id = arch(&world, abandoned);
    cache.sync(&queue, &world).unwrap();
    let before = cache.column(arch(&world, entities[0])).unwrap();
    let before_bytes = queue.buffers.lock().unwrap()[&before.buffer_id().unwrap()].clone();
    world.restore(&snapshot);
    let restored_id = arch(&world, entities[0]);
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 1,
            upload_calls: 2,
            uploaded_bytes: 35 * 512,
        }
    );
    assert_eq!((work(&cache).2, work(&cache).3, work(&cache).4), (2, 2, 1));
    assert!(cache.column(abandoned_id).is_none());
    assert!(cache.column(restored_id).unwrap().generation() > before.generation());
    assert_eq!(
        queue.buffers.lock().unwrap()[&before.buffer_id().unwrap()],
        before_bytes
    );
    verify(&cache, &queue, &world, restored_id);
    clean(&mut cache, &queue, &world);
    *world.get_mut::<Row>(entities[34]).unwrap() = row(54321);
    assert_eq!(cache.sync(&queue, &world).unwrap().uploaded_bytes, 3 * 512);
    assert_eq!(work(&cache), (1, 2, 1, 1, 0));
    verify(&cache, &queue, &world, restored_id);
    clean(&mut cache, &queue, &world);
}

#[test]
fn zero_sized_columns_track_extent_without_chunks_payload_or_allocations() {
    let (device, queue) = null();
    let mut world = World::new();
    let entity = world.spawn(Marker);
    let id = arch(&world, entity);
    let mut cache = EcsResidentColumns::<Marker>::new(device.clone(), &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    assert_eq!(cache.column(id).unwrap().rows, 1);
    assert!(world.despawn(entity));
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
    assert_eq!(
        (
            work(&cache).1,
            work(&cache).2,
            work(&cache).3,
            work(&cache).4
        ),
        (2, 0, 0, 0)
    );
    assert_eq!(cache.column(id).unwrap().rows, 0);
    assert_eq!(device.live_resource_count(), 0);
    clean(&mut cache, &queue, &world);
}

#[test]
fn automatic_extract_uses_the_same_zero_scan_and_sparse_discovery_path() {
    let mut app = App::new();
    app.register_gpu_column::<Row>().unwrap();
    app.add_plugins(PrismRenderPlugin::with_config(
        NullInstance::new(),
        RenderAppConfig {
            extent: Extent3d::new_2d(16, 16),
            material_capacity: 2,
            material_parameter_words: 32,
            material_texture_rows: 2,
            ..Default::default()
        },
    ));
    let mut last = None;
    for key in 0..32 {
        for index in 0..128 {
            last = Some(
                app.world_mut()
                    .spawn((row(index), Extra([0; 128]), BatchKey(key))),
            );
        }
    }
    fn render(app: &App) -> &World {
        &app.get_sub_app(RenderApp).unwrap().world
    }
    app.update();
    assert_eq!(
        render(&app)
            .resource::<GpuColumnSyncReport<Row>>()
            .result
            .unwrap()
            .upload_calls,
        256
    );
    for _ in 0..4 {
        app.update();
        assert_eq!(
            render(&app)
                .resource::<GpuColumnSyncReport<Row>>()
                .result
                .unwrap(),
            EcsColumnUploadStats::default()
        );
        assert_eq!(
            work(render(&app).resource::<EcsResidentColumns<Row>>()),
            (0, 0, 0, 0, 0)
        );
    }
    *app.world_mut().get_mut::<Row>(last.unwrap()).unwrap() = row(123_456);
    app.update();
    assert_eq!(
        render(&app)
            .resource::<GpuColumnSyncReport<Row>>()
            .result
            .unwrap()
            .upload_calls,
        1
    );
    assert_eq!(
        work(render(&app).resource::<EcsResidentColumns<Row>>()),
        (1, 2, 1, 1, 0)
    );
}

mod capacity_failure {
    use super::*;
    use prism_render_driver as rhi;
    use rhi::RenderDevice;

    struct LimitedDevice {
        inner: Arc<NullDevice>,
        caps: rhi::DeviceCapabilities,
    }

    // Only the advertised buffer limit differs from the normal null device.
    macro_rules! create {
        ($($method:ident($descriptor:ty) -> $id:ty;)*) => {$(
            fn $method(&self, descriptor: &$descriptor) -> $id {
                self.inner.$method(descriptor)
            }
        )*};
    }

    macro_rules! destroy {
        ($($method:ident($id:ty);)*) => {$(
            fn $method(&self, id: $id) { self.inner.$method(id); }
        )*};
    }

    impl RenderDevice for LimitedDevice {
        fn capabilities(&self) -> &rhi::DeviceCapabilities {
            &self.caps
        }

        fn create_buffer(&self, descriptor: &rhi::BufferDescriptor) -> BufferId {
            assert!(descriptor.size <= self.caps.limits.max_storage_buffer_binding_size);
            self.inner.create_buffer(descriptor)
        }

        fn create_texture_view(
            &self,
            texture: rhi::TextureId,
            descriptor: &rhi::TextureViewDescriptor,
        ) -> rhi::TextureViewId {
            self.inner.create_texture_view(texture, descriptor)
        }

        create! {
            create_texture(rhi::TextureDescriptor) -> rhi::TextureId;
            create_sampler(rhi::SamplerDescriptor) -> rhi::SamplerId;
            create_shader_module(rhi::ShaderModuleDescriptor) -> rhi::ShaderModuleId;
            create_bind_group_layout(rhi::BindGroupLayoutDescriptor) -> rhi::BindGroupLayoutId;
            create_bind_group(rhi::BindGroupDescriptor) -> rhi::BindGroupId;
            create_pipeline_layout(rhi::PipelineLayoutDescriptor) -> rhi::PipelineLayoutId;
            create_render_pipeline(rhi::RenderPipelineDescriptor) -> rhi::RenderPipelineId;
            create_compute_pipeline(rhi::ComputePipelineDescriptor) -> rhi::ComputePipelineId;
        }

        destroy! {
            destroy_buffer(BufferId);
            destroy_texture(rhi::TextureId);
            destroy_texture_view(rhi::TextureViewId);
            destroy_sampler(rhi::SamplerId);
            destroy_shader_module(rhi::ShaderModuleId);
            destroy_bind_group_layout(rhi::BindGroupLayoutId);
            destroy_bind_group(rhi::BindGroupId);
            destroy_pipeline_layout(rhi::PipelineLayoutId);
            destroy_render_pipeline(rhi::RenderPipelineId);
            destroy_compute_pipeline(rhi::ComputePipelineId);
        }
    }

    #[test]
    fn preflight_failure_does_not_write_mutate_or_acknowledge_any_affected_table() {
        let (inner, queue) = null();
        let mut caps = *inner.capabilities();
        caps.limits.max_storage_buffer_binding_size = 32 * 512;
        let device = Arc::new(LimitedDevice {
            inner: inner.clone(),
            caps,
        });
        let mut world = World::new();
        // The valid dirty table sorts BEFORE the overflowing table. A per-table
        // validation/upload loop would write a partial frame before failing.
        let small = world.spawn((row(0), Extra([0; 128])));
        let large = world.spawn(row(1));
        for seed in 2..33 {
            world.spawn(row(seed));
        }
        let small_id = arch(&world, small);
        let large_id = arch(&world, large);
        assert!(small_id < large_id);
        let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
        cache.sync(&queue, &world).unwrap();
        let previous_work = work(&cache);
        let previous = cache.column(large_id).unwrap();
        let writes = queue.writes.lock().unwrap().len();
        let buffers = queue.buffers.lock().unwrap().clone();
        *world.get_mut::<Row>(small).unwrap() = row(987_654);
        let overflow = world.spawn(row(100));
        for _ in 0..2 {
            assert_eq!(
                cache.sync(&queue, &world),
                Err(EcsColumnUploadError::CapacityExceeded)
            );
            assert_eq!(queue.writes.lock().unwrap().len(), writes);
            assert_eq!(*queue.buffers.lock().unwrap(), buffers);
            assert_eq!(inner.live_resource_count(), 2);
            assert_eq!(cache.column(large_id).unwrap().rows, 32);
            assert_eq!(
                cache.column(large_id).unwrap().generation(),
                previous.generation()
            );
            assert_eq!(work(&cache), previous_work);
        }
        assert!(world.despawn(overflow));
        assert_eq!(
            cache.sync(&queue, &world).unwrap(),
            EcsColumnUploadStats {
                allocations: 0,
                upload_calls: 1,
                uploaded_bytes: 512,
            }
        );
        verify(&cache, &queue, &world, small_id);
        verify(&cache, &queue, &world, large_id);
        clean(&mut cache, &queue, &world);
    }
}
