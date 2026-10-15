//! Public resident-column API, structural invalidation, and GPU readback coverage.
//! PRISM_REQUIRE_GPU=1 makes missing hardware a failure instead of a skip.
#![forbid(unsafe_code)]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use prism_ecs::archetype::ArchetypeId;
use prism_ecs::component::{Component, StorageType};
use prism_ecs::entity::Entity;
use prism_ecs::world::World;
use prism_render_app::{EcsColumnUploadError, EcsColumnUploadStats, EcsResidentColumns};
use prism_render_driver::{
    BufferDescriptor, BufferUsages, CommandEncoder, CopyCommand, DeviceDescriptor, Instance,
    RenderDevice, RenderQueue, RequestAdapterOptions,
};
use prism_render_driver_null::{NullDevice, NullInstance};
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};

// A physical chunk holds 32 of these, but only 16 alongside Extra. Every lane
// varies so stride, row order, chunk gaps, and tail corruption are observable.
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

#[repr(transparent)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Odd([u8; 3]);
impl Component for Odd {}

#[repr(transparent)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Sparse(u32);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

// WGSL equivalent: struct Compact { a: u32, b: u32, c: u32 }. Its array
// stride is 12, unlike vec3<u32>'s 16, so no implicit padding is permitted.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Compact([u32; 3]);
impl Component for Compact {}

fn gpu() -> Option<(Arc<WgpuDevice>, WgpuQueue)> {
    match WgpuInstance::new().request_default_device() {
        Ok((device, queue)) => {
            eprintln!(
                "resident-column GPU: {:?}",
                device.wgpu_adapter().get_info()
            );
            Some((Arc::new(device), queue))
        }
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "required GPU: {error}"
            );
            eprintln!("SKIPPED resident-column GPU test: {error}");
            None
        }
    }
}

fn null() -> (Arc<NullDevice>, impl RenderQueue) {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .unwrap();
    let (device, queue) = instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .unwrap();
    (Arc::new(device), queue)
}

fn arch(world: &World, entity: Entity) -> ArchetypeId {
    world.entities().location(entity).unwrap().archetype_id
}

fn chunk_stamp(world: &World, archetype: ArchetypeId, index: usize) -> (u64, u64) {
    let chunk = world
        .archetypes()
        .get(archetype)
        .unwrap()
        .table()
        .iter_chunks()
        .nth(index)
        .unwrap();
    (chunk.incarnation, chunk.meta.structural_epoch())
}

// Recreate a physical block while preserving every entity identity, row order,
// change tick, and final structural epoch. Identity scans cannot detect this.
fn recreate_chunk_with_same_rows(world: &mut World, entities: &[Entity]) {
    let archetype = arch(world, entities[0]);
    let before = chunk_stamp(world, archetype, 0);
    let tick = world.change_tick();
    for &entity in entities.iter().rev() {
        assert!(world.insert(entity, Extra([1; 128])));
    }
    assert!(world
        .archetypes()
        .get(archetype)
        .unwrap()
        .table()
        .is_empty());
    *world.get_mut::<Row>(entities[17]).unwrap() = row(999_999);
    for &entity in entities {
        assert!(world.remove::<Extra>(entity));
    }
    let table = world.archetypes().get(archetype).unwrap().table();
    assert_eq!(table.entities(), entities);
    assert_eq!(world.change_tick(), tick);
    let after = chunk_stamp(world, archetype, 0);
    assert_ne!(
        after.0, before.0,
        "reallocated blocks need fresh incarnations"
    );
    assert_eq!(after.1, before.1, "exercise equal structural epochs");
    let mut spans = Vec::new();
    table.column_upload_spans(
        world.components().id_of::<Row>().unwrap(),
        tick,
        tick,
        &mut spans,
    );
    assert!(
        spans.is_empty(),
        "exercise structural rather than tick detection"
    );
}

fn expected(world: &World, archetype: ArchetypeId) -> Vec<u8> {
    world
        .archetypes()
        .get(archetype)
        .unwrap()
        .table()
        .entities()
        .iter()
        .flat_map(|&entity| {
            bytemuck::bytes_of(world.get::<Row>(entity).unwrap())
                .iter()
                .copied()
        })
        .collect()
}

fn roundtrip(
    cache: &EcsResidentColumns<Row>,
    queue: &WgpuQueue,
    world: &World,
    archetype: ArchetypeId,
) {
    let view = cache.column(archetype).unwrap();
    assert_eq!(view.stride, 512);
    assert_eq!(view.rows, world.archetypes().get(archetype).unwrap().len());
    assert_eq!(
        queue.read_buffer(view.buffer_id().unwrap(), view.live_bytes()),
        expected(world, archetype)
    );
}

#[test]
fn real_gpu_dirty_only_upload_preserves_clean_gpu_bytes() {
    let Some((device, queue)) = gpu() else { return };
    let mut world = World::new();
    let entities: Vec<_> = (0..99).map(|i| world.spawn(row(i))).collect();
    let archetype = arch(&world, entities[0]);
    let mut cache = EcsResidentColumns::<Row>::new(device.clone(), &mut world).unwrap();
    let initial = cache.sync(&queue, &world).unwrap();
    assert_eq!(
        initial,
        EcsColumnUploadStats {
            allocations: 1,
            upload_calls: 4,
            uploaded_bytes: 99 * 512
        }
    );
    roundtrip(&cache, &queue, &world, archetype);
    let original = cache.column(archetype).unwrap();

    // Poison an unchanged GPU-only row. A hidden whole-column upload would
    // erase this sentinel and fail the subsequent real-device readback.
    let sentinel = row(0xdead_beef);
    queue.write_buffer(
        original.buffer_id().unwrap(),
        5 * 512,
        bytemuck::bytes_of(&sentinel),
    );
    world.increment_change_tick();
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
    world.increment_change_tick();
    *world.get_mut::<Row>(entities[40]).unwrap() = row(444);
    *world.get_mut::<Row>(entities[98]).unwrap() = row(888);
    let dirty = cache.sync(&queue, &world).unwrap();
    assert_eq!(
        dirty,
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 2,
            uploaded_bytes: 35 * 512
        }
    );
    let current = cache.column(archetype).unwrap();
    assert_eq!(current.buffer_id(), original.buffer_id());
    assert_eq!(current.generation(), original.generation());
    let mut bytes = expected(&world, archetype);
    bytes[5 * 512..6 * 512].copy_from_slice(bytemuck::bytes_of(&sentinel));
    assert_eq!(
        queue.read_buffer(current.buffer_id().unwrap(), current.live_bytes()),
        bytes
    );
    assert!(device.poll_wait());
}

#[test]
fn real_gpu_growth_shrink_swap_remove_empty_and_retained_views() {
    let Some((device, queue)) = gpu() else { return };
    let mut world = World::new();
    let mut entities: Vec<_> = (0..33).map(|i| world.spawn(row(i))).collect();
    let archetype = arch(&world, entities[0]);
    let mut cache = EcsResidentColumns::<Row>::new(device.clone(), &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    let old = cache.column(archetype).unwrap();
    assert_eq!(old.capacity_bytes(), 64 * 512);

    world.increment_change_tick();
    entities.extend((33..64).map(|i| world.spawn(row(i))));
    let within = cache.sync(&queue, &world).unwrap();
    assert_eq!(
        within,
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 32 * 512
        }
    );
    roundtrip(&cache, &queue, &world, archetype);
    let retained_bytes = queue.read_buffer(old.buffer_id().unwrap(), old.live_bytes());

    world.increment_change_tick();
    entities.push(world.spawn(row(64)));
    let grown = cache.sync(&queue, &world).unwrap();
    assert_eq!(
        grown,
        EcsColumnUploadStats {
            allocations: 1,
            upload_calls: 3,
            uploaded_bytes: 65 * 512
        }
    );
    let current = cache.column(archetype).unwrap();
    assert_ne!(current.buffer_id(), old.buffer_id());
    assert!(current.generation() > old.generation());
    assert_eq!(current.capacity_bytes(), 128 * 512);
    roundtrip(&cache, &queue, &world, archetype);
    assert_eq!(
        queue.read_buffer(old.buffer_id().unwrap(), old.live_bytes()),
        retained_bytes
    );

    // Removal moves the old tail into row 2 without changing its value tick.
    world.increment_change_tick();
    assert!(world.despawn(entities.remove(2)));
    let shrunk = cache.sync(&queue, &world).unwrap();
    assert_eq!(
        shrunk,
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 32 * 512
        }
    );
    roundtrip(&cache, &queue, &world, archetype);
    assert_eq!(
        cache.column(archetype).unwrap().buffer_id(),
        current.buffer_id()
    );

    for entity in entities {
        assert!(world.despawn(entity));
    }
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
    let empty = cache.column(archetype).unwrap();
    assert_eq!(empty.rows, 0);
    assert_eq!(empty.live_bytes(), 0);
    assert_eq!(empty.capacity_bytes(), current.capacity_bytes());
    world.spawn(row(900));
    let refill = cache.sync(&queue, &world).unwrap();
    assert_eq!(
        refill,
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 512
        }
    );
    roundtrip(&cache, &queue, &world, archetype);
    let last = cache.column(archetype).unwrap();
    cache.reset();
    assert!(cache.column(archetype).is_none());
    cache.sync(&queue, &world).unwrap();
    assert!(cache.column(archetype).unwrap().generation() > last.generation());
    drop(cache);
    assert_eq!(
        queue.read_buffer(last.buffer_id().unwrap(), 512),
        expected(&world, archetype)
    );
    assert!(device.poll_wait());
}

#[test]
fn real_gpu_archetype_migration_preserves_ticks_but_updates_both_buffers() {
    let Some((device, queue)) = gpu() else { return };
    let mut world = World::new();
    let entities: Vec<_> = (0..64).map(|i| world.spawn(row(i))).collect();
    let tagged = world.spawn((row(1000), Extra([0; 128])));
    assert_eq!(world.get::<Extra>(tagged).unwrap().0[0], 0);
    let source = arch(&world, entities[0]);
    let destination = arch(&world, tagged);
    assert_eq!(
        world
            .archetypes()
            .get(source)
            .unwrap()
            .table()
            .rows_per_chunk(),
        32
    );
    assert_eq!(
        world
            .archetypes()
            .get(destination)
            .unwrap()
            .table()
            .rows_per_chunk(),
        16
    );
    let mut cache = EcsResidentColumns::<Row>::new(device.clone(), &mut world).unwrap();
    assert_eq!(cache.sync(&queue, &world).unwrap().allocations, 2);
    let before_source = cache.column(source).unwrap();
    let before_destination = cache.column(destination).unwrap();
    let last = world.change_tick();
    world.increment_change_tick();
    assert!(world.insert(entities[0], Extra([1; 128])));
    // The CPU dirty-span API intentionally tracks value changes, not row moves.
    let id = world.components().id_of::<Row>().unwrap();
    let mut spans = Vec::new();
    for archetype in [source, destination] {
        world
            .archetypes()
            .get(archetype)
            .unwrap()
            .table()
            .column_upload_spans(id, last, world.change_tick(), &mut spans);
    }
    assert!(
        spans.is_empty(),
        "row movement must exercise structural tracking"
    );
    let moved = cache.sync(&queue, &world).unwrap();
    assert_eq!(
        moved,
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 3,
            uploaded_bytes: (63 + 2) * 512
        }
    );
    roundtrip(&cache, &queue, &world, source);
    roundtrip(&cache, &queue, &world, destination);
    assert_eq!(
        cache.column(source).unwrap().buffer_id(),
        before_source.buffer_id()
    );
    assert_eq!(
        cache.column(destination).unwrap().buffer_id(),
        before_destination.buffer_id()
    );
    assert!(world.remove::<Extra>(entities[0]));
    cache.sync(&queue, &world).unwrap();
    roundtrip(&cache, &queue, &world, source);
    roundtrip(&cache, &queue, &world, destination);
}

#[test]
fn real_gpu_recreated_chunk_with_equal_epoch_and_length_uploads_new_entities() {
    let Some((device, queue)) = gpu() else { return };
    let mut world = World::new();
    let entities: Vec<_> = (0..32).map(|i| world.spawn(row(i))).collect();
    let archetype = arch(&world, entities[0]);
    let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    let incarnation = chunk_stamp(&world, archetype, 0).0;
    let epoch = world
        .archetypes()
        .get(archetype)
        .unwrap()
        .table()
        .chunk_meta(0)
        .structural_epoch();
    // No intervening sync and no tick advancement. Recreated ChunkMeta starts
    // again at zero; its final epoch equals the old block's epoch.
    for entity in entities {
        assert!(world.despawn(entity));
    }
    for i in 0..32 {
        world.spawn(row(100 + i));
    }
    assert_eq!(
        world
            .archetypes()
            .get(archetype)
            .unwrap()
            .table()
            .chunk_meta(0)
            .structural_epoch(),
        epoch
    );
    assert_ne!(chunk_stamp(&world, archetype, 0).0, incarnation);
    let update = cache.sync(&queue, &world).unwrap();
    assert_eq!(
        update,
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 32 * 512
        }
    );
    roundtrip(&cache, &queue, &world, archetype);
}

#[test]
fn recreated_chunk_with_same_entities_rows_tick_and_epoch_is_not_clean() {
    let (device, queue) = null();
    let mut world = World::new();
    let entities: Vec<_> = (0..32).map(|i| world.spawn(row(i))).collect();
    let archetype = arch(&world, entities[0]);
    let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    let before = cache.column(archetype).unwrap();
    recreate_chunk_with_same_rows(&mut world, &entities);
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 32 * 512,
        }
    );
    let after = cache.column(archetype).unwrap();
    assert_eq!(after.buffer_id(), before.buffer_id());
    assert_eq!(after.generation(), before.generation());
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
}

#[test]
fn real_gpu_recreated_chunk_with_same_entities_and_epoch_refreshes_values() {
    let Some((device, queue)) = gpu() else { return };
    let mut world = World::new();
    let entities: Vec<_> = (0..32).map(|i| world.spawn(row(i))).collect();
    let archetype = arch(&world, entities[0]);
    let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    recreate_chunk_with_same_rows(&mut world, &entities);
    let update = cache.sync(&queue, &world).unwrap();
    assert_eq!(update.allocations, 0);
    assert_eq!(update.upload_calls, 1);
    assert_eq!(update.uploaded_bytes, 32 * 512);
    roundtrip(&cache, &queue, &world, archetype);
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
}

#[test]
fn real_gpu_recreated_tail_chunk_preserves_clean_front_chunk() {
    let Some((device, queue)) = gpu() else { return };
    let mut world = World::new();
    let entities: Vec<_> = (0..64).map(|i| world.spawn(row(i))).collect();
    let archetype = arch(&world, entities[0]);
    let mut cache = EcsResidentColumns::<Row>::new(device, &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    let front = chunk_stamp(&world, archetype, 0);
    let tail = chunk_stamp(&world, archetype, 1);
    let before = cache.column(archetype).unwrap();
    let sentinel = row(777_777);
    queue.write_buffer(
        before.buffer_id().unwrap(),
        5 * 512,
        bytemuck::bytes_of(&sentinel),
    );
    for &entity in entities[32..].iter().rev() {
        assert!(world.despawn(entity));
    }
    for i in 0..32 {
        world.spawn(row(100 + i));
    }
    assert_eq!(chunk_stamp(&world, archetype, 0), front);
    let new_tail = chunk_stamp(&world, archetype, 1);
    assert_ne!(new_tail.0, tail.0);
    assert_eq!(new_tail.1, tail.1);
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 32 * 512,
        }
    );
    let after = cache.column(archetype).unwrap();
    assert_eq!(after.buffer_id(), before.buffer_id());
    let mut bytes = expected(&world, archetype);
    bytes[5 * 512..6 * 512].copy_from_slice(bytemuck::bytes_of(&sentinel));
    assert_eq!(
        queue.read_buffer(after.buffer_id().unwrap(), after.live_bytes()),
        bytes
    );
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
}

#[test]
fn real_gpu_snapshot_reset_and_replacement_device_roundtrip() {
    let Some((device, queue)) = gpu() else { return };
    let mut world = World::new();
    world.register_snapshot_component::<Row>();
    let entities: Vec<_> = (0..35).map(|i| world.spawn(row(i))).collect();
    let snapshot = world.snapshot();
    let archetype = arch(&world, entities[0]);
    let mut cache = EcsResidentColumns::<Row>::new(device.clone(), &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    world.increment_change_tick();
    *world.get_mut::<Row>(entities[10]).unwrap() = row(12345);
    cache.sync(&queue, &world).unwrap();
    let before = cache.column(archetype).unwrap();
    let previous_bytes = queue.read_buffer(before.buffer_id().unwrap(), before.live_bytes());
    world.restore(&snapshot);
    // Restore changes the world's archetype generation. Even reused ids/ticks
    // must automatically invalidate allocation and structural history.
    let restored_archetype = arch(&world, entities[0]);
    let restored = cache.sync(&queue, &world).unwrap();
    assert_eq!(restored.uploaded_bytes, 35 * 512);
    assert_eq!(restored.allocations, 1);
    assert!(cache.column(restored_archetype).unwrap().generation() > before.generation());
    roundtrip(&cache, &queue, &world, restored_archetype);
    assert_eq!(
        queue.read_buffer(before.buffer_id().unwrap(), before.live_bytes()),
        previous_bytes
    );

    let Some((replacement_device, replacement_queue)) = gpu() else {
        return;
    };
    cache.reset_device(replacement_device.clone());
    assert_eq!(
        cache.sync(&replacement_queue, &world).unwrap().allocations,
        1
    );
    roundtrip(&cache, &replacement_queue, &world, restored_archetype);
    // The old view still resolves through the original device after replacement.
    assert_eq!(
        queue.read_buffer(before.buffer_id().unwrap(), before.live_bytes()),
        previous_bytes
    );
    assert!(device.poll_wait());
    assert!(replacement_device.poll_wait());
}

#[test]
fn real_gpu_compact_layout_new_archetype_and_partial_tail() {
    let Some((device, queue)) = gpu() else { return };
    let mut world = World::new();
    // Construct and sync the query before the matching archetype exists.
    let mut cache = EcsResidentColumns::<Compact>::new(device, &mut world).unwrap();
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
    let entities: Vec<_> = (0..100)
        .map(|i| world.spawn((Compact([i, i ^ 0xaabbccdd, !i]), Extra([i; 128]))))
        .collect();
    let archetype = arch(&world, entities[0]);
    let rows_per_chunk = world
        .archetypes()
        .get(archetype)
        .unwrap()
        .table()
        .rows_per_chunk();
    assert_eq!(rows_per_chunk, 31);
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 1,
            upload_calls: 4,
            uploaded_bytes: 1200,
        }
    );
    let view = cache.column(archetype).unwrap();
    assert_eq!(view.stride, 12);
    let initial: Vec<u8> = entities
        .iter()
        .flat_map(|&entity| {
            bytemuck::bytes_of(world.get::<Compact>(entity).unwrap())
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(
        queue.read_buffer(view.buffer_id().unwrap(), view.live_bytes()),
        initial
    );

    world.increment_change_tick();
    *world.get_mut::<Compact>(entities[99]).unwrap() = Compact([7, 9, 11]);
    assert_eq!(
        cache.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats {
            allocations: 0,
            upload_calls: 1,
            uploaded_bytes: 7 * 12,
        }
    );
    let mut updated = initial;
    updated[99 * 12..].copy_from_slice(bytemuck::bytes_of(&Compact([7, 9, 11])));
    assert_eq!(
        queue.read_buffer(view.buffer_id().unwrap(), view.live_bytes()),
        updated
    );
}

#[test]
fn real_gpu_recorded_consumer_survives_growth_reset_and_cache_drop() {
    let Some((device, queue)) = gpu() else { return };
    let mut world = World::new();
    let entities: Vec<_> = (0..33).map(|i| world.spawn(row(i))).collect();
    let archetype = arch(&world, entities[0]);
    let original_bytes = expected(&world, archetype);
    let mut cache = EcsResidentColumns::<Row>::new(device.clone(), &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    let retained = cache.column(archetype).unwrap();
    let destination = device.create_buffer(&BufferDescriptor::new(
        retained.live_bytes(),
        BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    ));
    let mut commands = CommandEncoder::labeled("retained ECS allocation consumer");
    commands.push_transfer_pass(
        None,
        vec![CopyCommand::CopyBufferToBuffer {
            src: retained.buffer_id().unwrap(),
            src_offset: 0,
            dst: destination,
            dst_offset: 0,
            size: retained.live_bytes(),
        }],
    );
    // No submission/readback has flushed the initial queue writes yet.
    world.increment_change_tick();
    for i in 33..65 {
        world.spawn(row(i));
    }
    assert_eq!(cache.sync(&queue, &world).unwrap().allocations, 1);
    cache.reset();
    drop(cache);
    queue.submit(&[commands.finish()]);
    assert_eq!(
        queue.read_buffer(destination, retained.live_bytes()),
        original_bytes
    );
    drop(retained);
    device.destroy_buffer(destination);
    assert!(device.poll_wait());
}

#[test]
fn validation_zero_sized_rows_world_identity_and_resource_release() {
    let (device, queue) = null();
    let mut world = World::new();
    assert!(matches!(
        EcsResidentColumns::<Odd>::new(device.clone(), &mut world),
        Err(EcsColumnUploadError::UnalignedStride { stride: 3 })
    ));
    assert!(matches!(
        EcsResidentColumns::<Sparse>::new(device.clone(), &mut world),
        Err(EcsColumnUploadError::NotTableStorage)
    ));
    let marker_entity = world.spawn(Marker);
    let mut markers = EcsResidentColumns::<Marker>::new(device.clone(), &mut world).unwrap();
    assert_eq!(
        markers.sync(&queue, &world).unwrap(),
        EcsColumnUploadStats::default()
    );
    let marker = markers.column(arch(&world, marker_entity)).unwrap();
    assert_eq!((marker.rows, marker.stride, marker.live_bytes()), (1, 0, 0));
    assert_eq!(marker.buffer_id(), None);
    assert_eq!(device.live_resource_count(), 0);

    let entity = world.spawn(row(7));
    let archetype = arch(&world, entity);
    let mut cache = EcsResidentColumns::<Row>::new(device.clone(), &mut world).unwrap();
    cache.sync(&queue, &world).unwrap();
    assert_eq!(device.live_resource_count(), 1);
    let retained = cache.column(archetype).unwrap();
    let other_world = World::new();
    assert_eq!(
        cache.sync(&queue, &other_world),
        Err(EcsColumnUploadError::WrongWorld)
    );
    cache.reset();
    assert_eq!(
        device.live_resource_count(),
        1,
        "view owns allocation after reset"
    );
    cache.sync(&queue, &world).unwrap();
    assert_eq!(device.live_resource_count(), 2);
    drop(cache);
    assert_eq!(device.live_resource_count(), 1);
    drop(retained);
    assert_eq!(
        device.live_resource_count(),
        0,
        "last owner releases buffer"
    );
}

// Exercise the production extract/recovery scheduling with an observable,
// loss-injecting backend. Actual byte-transfer/readback tests above use wgpu;
// this backend makes uploads during a lost-device frame a deterministic failure.
mod automatic_recovery {
    use super::*;
    use prism_app::App;
    use prism_render_app::{
        GpuColumnAppExt, GpuColumnSyncReport, PrismRenderPlugin, RenderApp, RenderAppConfig,
        RenderRuntimeResource,
    };
    use prism_render_driver as rhi;
    use prism_render_driver_null::{NullAdapter, NullQueue, NullSurface};
    use rhi::FrameSync;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    };

    #[derive(Default)]
    struct Probe {
        loss: Mutex<Option<rhi::DeviceLostReason>>,
        columns: Mutex<Vec<rhi::BufferId>>,
        writes: Mutex<Vec<(rhi::BufferId, u64, Vec<u8>)>>,
    }

    #[derive(Default)]
    struct Controls {
        fail_device: AtomicBool,
        devices: Mutex<Vec<Arc<Probe>>>,
    }

    impl Controls {
        fn current(&self) -> Arc<Probe> {
            self.devices.lock().unwrap().last().unwrap().clone()
        }

        fn lose(&self) {
            *self.current().loss.lock().unwrap() = Some(rhi::DeviceLostReason::Destroyed);
        }
    }

    struct Device {
        inner: NullDevice,
        probe: Arc<Probe>,
    }

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

    impl RenderDevice for Device {
        fn capabilities(&self) -> &rhi::DeviceCapabilities {
            self.inner.capabilities()
        }

        fn create_buffer(&self, descriptor: &rhi::BufferDescriptor) -> rhi::BufferId {
            let id = self.inner.create_buffer(descriptor);
            if descriptor
                .label
                .as_deref()
                .is_some_and(|s| s.starts_with("ecs-column-"))
            {
                self.probe.columns.lock().unwrap().push(id);
            }
            id
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

        fn create_texture_view(
            &self,
            texture: rhi::TextureId,
            descriptor: &rhi::TextureViewDescriptor,
        ) -> rhi::TextureViewId {
            self.inner.create_texture_view(texture, descriptor)
        }

        fn take_device_lost(&self) -> Option<rhi::DeviceLostReason> {
            self.probe.loss.lock().unwrap().take()
        }

        destroy! {
            destroy_buffer(rhi::BufferId);
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

    struct Queue(Arc<Probe>);

    impl RenderQueue for Queue {
        fn write_buffer(&self, buffer: rhi::BufferId, offset: u64, data: &[u8]) {
            if self.0.columns.lock().unwrap().contains(&buffer) {
                self.0
                    .writes
                    .lock()
                    .unwrap()
                    .push((buffer, offset, data.to_vec()));
            }
            NullQueue.write_buffer(buffer, offset, data);
        }

        fn write_texture(
            &self,
            destination: rhi::TextureWrite,
            data: &[u8],
            layout: rhi::ImageDataLayout,
        ) {
            NullQueue.write_texture(destination, data, layout);
        }

        fn submit(&self, commands: &[rhi::CommandBuffer]) {
            NullQueue.submit(commands);
        }
    }

    impl FrameSync for Queue {
        fn submit_tracked(&self, commands: &[rhi::CommandBuffer]) -> rhi::CompletionToken {
            NullQueue.submit_tracked(commands)
        }

        fn wait_for(&self, token: &(dyn core::any::Any + Send + Sync)) -> bool {
            NullQueue.wait_for(token)
        }

        fn wait_idle(&self) -> bool {
            NullQueue.wait_idle()
        }

        fn on_submitted_work_done(&self, callback: Box<dyn FnOnce() + Send + 'static>) {
            NullQueue.on_submitted_work_done(callback);
        }
    }

    struct Backend(Arc<Controls>);

    impl Instance for Backend {
        type Adapter = NullAdapter;
        type Device = Device;
        type Queue = Queue;
        type Surface = NullSurface;

        fn enumerate_adapters(&self) -> Vec<NullAdapter> {
            NullInstance::new().enumerate_adapters()
        }

        fn request_adapter(&self, options: &rhi::RequestAdapterOptions) -> Option<NullAdapter> {
            NullInstance::new().request_adapter(options)
        }

        fn request_adapter_for_surface(
            &self,
            options: &rhi::RequestAdapterOptions,
            surface: &NullSurface,
        ) -> Option<NullAdapter> {
            NullInstance::new().request_adapter_for_surface(options, surface)
        }

        fn create_device(
            &self,
            adapter: &NullAdapter,
            descriptor: &rhi::DeviceDescriptor,
        ) -> Result<(Device, Queue), rhi::DeviceError> {
            if self.0.fail_device.load(Ordering::SeqCst) {
                return Err(rhi::DeviceError::Request(
                    "injected column recovery failure".into(),
                ));
            }
            let (inner, _) = NullInstance::new().create_device(adapter, descriptor)?;
            let probe = Arc::new(Probe::default());
            self.0.devices.lock().unwrap().push(probe.clone());
            Ok((
                Device {
                    inner,
                    probe: probe.clone(),
                },
                Queue(probe),
            ))
        }

        fn create_surface<W>(
            &self,
            _window: Arc<W>,
            _device: &Device,
            _queue: &Queue,
        ) -> Result<NullSurface, rhi::SurfaceError> {
            Err(rhi::SurfaceError::Unsupported)
        }
    }

    fn render(app: &App) -> &World {
        &app.get_sub_app(RenderApp).unwrap().world
    }

    fn report<T: Component + Pod>(app: &App) -> Result<EcsColumnUploadStats, EcsColumnUploadError> {
        render(app).resource::<GpuColumnSyncReport<T>>().result
    }

    fn generation(app: &App) -> u64 {
        render(app)
            .resource::<RenderRuntimeResource<Backend>>()
            .runtime()
            .device_generation()
    }

    #[test]
    fn automatic_upload_stops_on_loss_retries_and_fully_initializes_replacement() {
        let controls = Arc::new(Controls::default());
        let mut app = App::new();
        app.register_gpu_column::<Row>().unwrap();
        app.add_plugins(PrismRenderPlugin::with_config(
            Backend(controls.clone()),
            RenderAppConfig {
                extent: rhi::Extent3d::new_2d(16, 16),
                material_capacity: 2,
                material_parameter_words: 32,
                material_texture_rows: 2,
                ..Default::default()
            },
        ));
        // Re-declaration after installing the plugin must remain idempotent.
        app.register_gpu_column::<Row>().unwrap();
        let entities: Vec<_> = (0..35).map(|i| app.world_mut().spawn(row(i))).collect();
        let archetype = arch(app.world(), entities[0]);
        app.update();
        assert_eq!(report::<Row>(&app).unwrap().uploaded_bytes, 35 * 512);
        let old = controls.current();
        assert_eq!(old.writes.lock().unwrap().len(), 2);
        let retained = render(&app)
            .resource::<EcsResidentColumns<Row>>()
            .column(archetype)
            .unwrap();
        app.update();
        assert_eq!(report::<Row>(&app), Ok(EcsColumnUploadStats::default()));
        assert_eq!(old.writes.lock().unwrap().len(), 2);

        app.world_mut().increment_change_tick();
        *app.world_mut().get_mut::<Row>(entities[31]).unwrap() = row(909);
        controls.lose();
        controls.fail_device.store(true, Ordering::SeqCst);
        app.update();
        assert_eq!(report::<Row>(&app), Err(EcsColumnUploadError::DeviceLost));
        assert!(render(&app)
            .resource::<EcsResidentColumns<Row>>()
            .column(archetype)
            .is_none());
        assert!(
            retained.buffer_id().is_some(),
            "prior consumers retain their old allocation"
        );
        assert_eq!(
            render(&app)
                .resource::<RenderRuntimeResource<Backend>>()
                .runtime()
                .device_lost(),
            Some(rhi::DeviceLostReason::Destroyed)
        );

        // A type first declared during loss must not allocate on the dead device.
        app.register_gpu_column::<Compact>().unwrap();
        app.world_mut().spawn(Compact([7, 8, 9]));
        for _ in 0..2 {
            app.update();
            assert_eq!(report::<Row>(&app), Err(EcsColumnUploadError::DeviceLost));
            assert_eq!(
                report::<Compact>(&app),
                Err(EcsColumnUploadError::DeviceLost)
            );
            assert!(render(&app)
                .get_resource::<EcsResidentColumns<Compact>>()
                .is_none());
            assert_eq!(generation(&app), 0);
            assert_eq!(old.writes.lock().unwrap().len(), 2);
            assert_eq!(old.columns.lock().unwrap().len(), 1);
        }

        controls.fail_device.store(false, Ordering::SeqCst);
        app.update(); // Extract remains gated; the render driver replaces the device.
        assert_eq!(generation(&app), 1);
        assert_eq!(report::<Row>(&app), Err(EcsColumnUploadError::DeviceLost));
        let replacement = controls.current();
        assert!(!Arc::ptr_eq(&old, &replacement));
        assert!(replacement.writes.lock().unwrap().is_empty());
        app.update(); // First extraction on the healthy replacement fully initializes.
        assert_eq!(
            report::<Row>(&app),
            Ok(EcsColumnUploadStats {
                allocations: 1,
                upload_calls: 2,
                uploaded_bytes: 35 * 512,
            })
        );
        assert_eq!(report::<Compact>(&app).unwrap().uploaded_bytes, 12);
        let buffer = render(&app)
            .resource::<EcsResidentColumns<Row>>()
            .column(archetype)
            .unwrap()
            .buffer_id()
            .unwrap();
        let writes = replacement.writes.lock().unwrap();
        let expected = expected(app.world(), archetype);
        let mut copied = 0;
        for (_, offset, bytes) in writes.iter().filter(|(id, _, _)| *id == buffer) {
            assert_eq!(*offset as usize, copied);
            assert_eq!(bytes, &expected[copied..copied + bytes.len()]);
            copied += bytes.len();
        }
        assert_eq!(copied, expected.len());
        let write_count = writes.len();
        drop(writes);
        app.update();
        assert_eq!(report::<Row>(&app), Ok(EcsColumnUploadStats::default()));
        assert_eq!(report::<Compact>(&app), Ok(EcsColumnUploadStats::default()));
        assert_eq!(replacement.writes.lock().unwrap().len(), write_count);
        assert_eq!(old.writes.lock().unwrap().len(), 2);

        // Loss on an otherwise clean frame also removes all bindable cache rows.
        controls.lose();
        app.update();
        assert_eq!(report::<Row>(&app), Err(EcsColumnUploadError::DeviceLost));
        assert!(render(&app)
            .resource::<EcsResidentColumns<Row>>()
            .column(archetype)
            .is_none());
        assert_eq!(replacement.writes.lock().unwrap().len(), write_count);
        app.update();
        app.update();
        assert_eq!(generation(&app), 2);
        assert_eq!(report::<Row>(&app).unwrap().uploaded_bytes, 35 * 512);
        assert_eq!(report::<Compact>(&app).unwrap().uploaded_bytes, 12);
    }
}
