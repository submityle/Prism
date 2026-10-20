use super::*;
use std::collections::{BTreeSet, HashSet};
use std::error::Error;

fn sector(id: i64) -> SectorId {
    SectorId::new(2, id, -id, id * 3)
}

fn proxy(id: u32) -> ProxyInstance {
    ProxyInstance::new(id, 7, [id as f32, -2.0, 3.0], 0.5, 1.0, 100.0)
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    // Arc<()> value equality does not distinguish instances; compare addresses.
    #[cfg(feature = "gpu")]
    identity: *const (),
    slots: Vec<Slot>,
    dense: Vec<ProxyInstance>,
    dense_slots: Vec<u32>,
    free: Vec<u32>,
    dirty: Vec<bool>,
    sectors: BTreeMap<SectorId, Vec<ProxyHandle>>,
    sector_capacity: usize,
    pending: bool,
    allocations: [usize; 5],
}

fn allocations(registry: &ProxyRegistry) -> [usize; 5] {
    [
        registry.slots.capacity(),
        registry.dense.capacity(),
        registry.dense_slots.capacity(),
        registry.free.capacity(),
        registry.dirty.capacity(),
    ]
}

fn snapshot(registry: &ProxyRegistry) -> Snapshot {
    Snapshot {
        #[cfg(feature = "gpu")]
        identity: Arc::as_ptr(registry.identity()),
        slots: registry.slots.clone(),
        dense: registry.dense.clone(),
        dense_slots: registry.dense_slots.clone(),
        free: registry.free.clone(),
        dirty: registry.dirty.clone(),
        sectors: registry.sectors.clone(),
        sector_capacity: registry.sector_capacity(),
        pending: registry.pending_uploads,
        allocations: allocations(registry),
    }
}

fn ranges(registry: &ProxyRegistry) -> Vec<Range<usize>> {
    registry.dirty_ranges().collect()
}

fn assert_invariants(registry: &ProxyRegistry) {
    let capacity = registry.capacity();
    assert_eq!(registry.sector_count(), registry.sectors.len());
    assert!(registry.sector_count() <= registry.sector_capacity());
    assert_eq!(registry.dirty.len(), capacity);
    assert_eq!(registry.dense_slots.len(), registry.len());
    assert!(registry.len() <= capacity);
    assert_eq!(registry.is_empty(), registry.proxies().is_empty());
    assert!(registry.dirty[registry.len()..].iter().all(|dirty| !dirty));
    assert!(ranges(registry).is_empty() || registry.has_pending_uploads());
    let free: BTreeSet<_> = registry.free.iter().copied().collect();
    assert_eq!(free.len(), registry.free.len());
    let mut owned = BTreeSet::new();
    for handles in registry.sectors.values() {
        let mut ids = BTreeSet::new();
        for &handle in handles {
            assert!(owned.insert(handle));
            assert!(ids.insert(registry.get(handle).unwrap().proxy_id));
        }
    }
    assert_eq!(owned.len(), registry.len());
    for (index, &slot_id) in registry.dense_slots.iter().enumerate() {
        let handle = registry.handle_at_dense(index).unwrap();
        assert_eq!(handle.slot(), slot_id);
        assert_eq!(
            handle.generation(),
            registry.slots[slot_id as usize].generation
        );
        assert_eq!(registry.dense_slot(handle), Some(index));
        assert_eq!(registry.get(handle), Some(&registry.proxies()[index]));
        assert!(owned.contains(&handle));
    }
    assert_eq!(registry.handle_at_dense(registry.len()), None);
    assert_eq!(registry.handle_at_dense(usize::MAX), None);
    let mut retired = 0;
    for (index, slot) in registry.slots.iter().enumerate() {
        if let Some(dense) = slot.dense {
            assert!(!free.contains(&(index as u32)));
            assert_eq!(registry.dense_slots[dense], index as u32);
        } else if slot.generation == u64::MAX {
            retired += 1;
            assert!(!free.contains(&(index as u32)));
        } else {
            assert!(free.contains(&(index as u32)));
        }
    }
    assert_eq!(registry.len() + free.len() + retired, capacity);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_identity_is_stable_across_moves_mutations_and_acknowledgment() {
    let registry = ProxyRegistry::with_sector_capacity(3, 4).unwrap();
    let identity = Arc::clone(registry.identity());
    let mut moved = Box::new(registry);
    assert!(Arc::ptr_eq(&identity, moved.identity()));
    moved.insert_sector(sector(0), &[]).unwrap();
    let first = moved.insert_sector(sector(1), &[proxy(1)]).unwrap()[0];
    let last = moved.insert_sector(sector(2), &[proxy(2)]).unwrap()[0];
    moved.acknowledge_upload();
    let mut changed = proxy(2);
    changed.position[0] = 9.0;
    moved.update(last, changed).unwrap();
    moved.remove_sector(sector(1));
    assert_eq!(moved.dense_slot(last), Some(0));
    let before = snapshot(&moved);
    assert!(moved.update(first, proxy(1)).is_err());
    assert!(moved.insert_sector(sector(0), &[]).is_err());
    assert_eq!(snapshot(&moved), before);
    moved.mark_all_dirty();
    moved.acknowledge_upload();
    moved.remove_sector(sector(0));
    moved.remove_sector(sector(2));
    let registry = *moved;
    assert!(Arc::ptr_eq(&identity, registry.identity()));
    assert_eq!(Arc::strong_count(&identity), 2);
    assert_invariants(&registry);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_identity_is_distinct_for_each_constructor_even_at_zero_capacity() {
    for capacity in [0, 2] {
        let default = ProxyRegistry::new(capacity).unwrap();
        let explicit = ProxyRegistry::with_sector_capacity(capacity, 0).unwrap();
        let other_default = ProxyRegistry::new(capacity).unwrap();
        let other_explicit = ProxyRegistry::with_sector_capacity(capacity, 0).unwrap();
        let registries = [default, explicit, other_default, other_explicit];
        for (index, registry) in registries.iter().enumerate() {
            for other in &registries[index + 1..] {
                assert!(!Arc::ptr_eq(registry.identity(), other.identity()));
            }
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_owner_retains_identity_across_replacement_preventing_allocation_aba() {
    let mut registry = ProxyRegistry::new(1).unwrap();
    let stale = registry.insert_sector(sector(1), &[proxy(1)]).unwrap()[0];
    registry.acknowledge_upload();
    let owner_identity = Arc::clone(registry.identity());
    for capacity in [0, 1, 3].into_iter().cycle().take(128) {
        let old = std::mem::replace(&mut registry, ProxyRegistry::new(capacity).unwrap());
        drop(old);
        assert_eq!(Arc::strong_count(&owner_identity), 1);
        assert!(!Arc::ptr_eq(&owner_identity, registry.identity()));
        assert!(!registry.has_pending_uploads());
        if capacity != 0 {
            let current = registry.insert_sector(sector(1), &[proxy(9)]).unwrap()[0];
            // Handles deliberately do not encode registry ownership; the GPU
            // owner's identity guard must reject replacement independently.
            assert_eq!(current, stale);
            assert_eq!(registry.get(stale), Some(&proxy(9)));
        }
    }
}

#[test]
fn handle_traits_and_getters_preserve_identity() {
    fn traits<T: Copy + Eq + Ord + std::hash::Hash>() {}
    traits::<ProxyHandle>();
    let mut registry = ProxyRegistry::new(2).unwrap();
    let handles = registry
        .insert_sector(sector(1), &[proxy(5), proxy(6)])
        .unwrap();
    assert_eq!(handles[0].slot(), 0);
    assert_eq!(handles[0].generation(), 0);
    assert!(handles[0] < handles[1]);
    assert_eq!(handles.iter().copied().collect::<HashSet<_>>().len(), 2);
}

#[test]
fn compaction_keeps_handles_and_sector_insertion_order_stable() {
    let mut registry = ProxyRegistry::new(8).unwrap();
    let a = registry
        .insert_sector(sector(1), &[proxy(10), proxy(11)])
        .unwrap();
    let b = registry
        .insert_sector(sector(2), &[proxy(20), proxy(21), proxy(22)])
        .unwrap();
    let c = registry
        .insert_sector(sector(3), &[proxy(30), proxy(31)])
        .unwrap();
    registry.acknowledge_upload();
    assert!(registry.remove_sector(sector(1)));
    assert_eq!(registry.sector_handles(sector(2)), Some(b.as_slice()));
    assert_eq!(registry.sector_handles(sector(3)), Some(c.as_slice()));
    assert_eq!(
        registry
            .proxies()
            .iter()
            .map(|p| p.proxy_id)
            .collect::<Vec<_>>(),
        [31, 30, 20, 21, 22]
    );
    for (&handle, id) in b.iter().zip([20, 21, 22]).chain(c.iter().zip([30, 31])) {
        assert_eq!(registry.get(handle).unwrap().proxy_id, id);
    }
    assert_eq!(registry.dense_slot(c[1]), Some(0));
    assert_eq!(registry.dense_slot(c[0]), Some(1));
    assert!(a.iter().all(|&handle| registry.get(handle).is_none()));
    assert_eq!(ranges(&registry), vec![0..2]);
    assert_invariants(&registry);
}

#[test]
fn removing_a_sector_handles_its_own_members_moving_from_the_tail() {
    let mut registry = ProxyRegistry::new(4).unwrap();
    let survivor = registry.insert_sector(sector(1), &[proxy(1)]).unwrap()[0];
    let removed = registry
        .insert_sector(sector(2), &[proxy(2), proxy(3), proxy(4)])
        .unwrap();
    registry.acknowledge_upload();
    assert!(registry.remove_sector(sector(2)));
    assert_eq!(registry.get(survivor), Some(&proxy(1)));
    assert!(removed
        .into_iter()
        .all(|handle| registry.get(handle).is_none()));
    assert!(ranges(&registry).is_empty());
    assert!(registry.has_pending_uploads());
    assert_invariants(&registry);
}

#[test]
fn equal_proxy_ids_in_different_sectors_update_independently() {
    let mut registry = ProxyRegistry::new(2).unwrap();
    let a = registry.insert_sector(sector(1), &[proxy(8)]).unwrap()[0];
    let b = registry.insert_sector(sector(2), &[proxy(8)]).unwrap()[0];
    let mut changed = proxy(8);
    changed.cluster_id = 99;
    changed.position = [10.0, 20.0, 30.0];
    changed.radius = 4.0;
    changed.geometric_error = 0.0;
    changed.max_distance = 0.0;
    changed.flags = u32::MAX;
    registry.update(b, changed).unwrap();
    assert_eq!(registry.get(a), Some(&proxy(8)));
    assert_eq!(registry.get(b), Some(&changed));
    registry.remove_sector(sector(1));
    assert_eq!(registry.get(b), Some(&changed));
    assert_invariants(&registry);
}

#[test]
fn reused_slots_never_resolve_stale_handles() {
    let mut registry = ProxyRegistry::new(1).unwrap();
    let mut stale = Vec::new();
    for generation in 0..128 {
        let handle = registry.insert_sector(sector(1), &[proxy(1)]).unwrap()[0];
        assert_eq!(handle.slot(), 0);
        assert_eq!(handle.generation(), generation);
        for &old in &stale {
            assert_eq!(registry.get(old), None);
            assert_eq!(registry.dense_slot(old), None);
            let before = snapshot(&registry);
            assert!(matches!(
                registry.update(old, proxy(1)),
                Err(ProxyRegistryError::InvalidHandle { .. })
            ));
            assert_eq!(snapshot(&registry), before);
        }
        registry.remove_sector(sector(1));
        stale.push(handle);
        assert_invariants(&registry);
    }
}

fn set_generation(
    registry: &mut ProxyRegistry,
    sector_id: SectorId,
    generation: u64,
) -> ProxyHandle {
    let handle = &mut registry.sectors.get_mut(&sector_id).unwrap()[0];
    registry.slots[handle.slot as usize].generation = generation;
    handle.generation = generation;
    *handle
}

#[test]
fn reaching_generation_max_permanently_retires_the_slot() {
    let mut registry = ProxyRegistry::new(2).unwrap();
    let survivor = registry.insert_sector(sector(9), &[proxy(9)]).unwrap()[0];
    registry.insert_sector(sector(1), &[proxy(1)]).unwrap();
    let old = set_generation(&mut registry, sector(1), u64::MAX - 2);
    registry.remove_sector(sector(1));
    let last = registry.insert_sector(sector(1), &[proxy(1)]).unwrap()[0];
    assert_eq!(last.slot(), old.slot());
    assert_eq!(last.generation(), u64::MAX - 1);
    assert_eq!(registry.get(old), None);
    registry.remove_sector(sector(1));
    assert_eq!(registry.slots[last.slot() as usize].generation, u64::MAX);
    assert_eq!(registry.get(last), None);
    assert_eq!(registry.capacity(), 2);
    assert_eq!(registry.len(), 1);
    let before = snapshot(&registry);
    assert!(matches!(
        registry.insert_sector(sector(2), &[proxy(2)]),
        Err(ProxyRegistryError::CapacityExceeded { available: 0, .. })
    ));
    assert_eq!(snapshot(&registry), before);
    assert_eq!(registry.get(survivor), Some(&proxy(9)));
    assert_invariants(&registry);
}

#[test]
fn removing_a_privately_constructed_max_generation_does_not_wrap() {
    let mut registry = ProxyRegistry::new(1).unwrap();
    registry.insert_sector(sector(1), &[proxy(1)]).unwrap();
    let handle = set_generation(&mut registry, sector(1), u64::MAX);
    assert!(registry.remove_sector(sector(1)));
    assert_eq!(registry.slots[0].generation, u64::MAX);
    assert!(registry.free.is_empty());
    assert_eq!(registry.get(handle), None);
    assert_invariants(&registry);
}

#[test]
fn zero_capacity_registers_empty_sectors_and_never_sets_pending() {
    let mut registry = ProxyRegistry::new(0).unwrap();
    assert_eq!(allocations(&registry), [0; 5]);
    assert_eq!(registry.capacity(), 0);
    assert_eq!(registry.sector_capacity(), 1);
    assert_eq!(registry.sector_count(), 0);
    assert!(registry.is_empty());
    let before = snapshot(&registry);
    assert!(matches!(
        registry.insert_sector(sector(1), &[proxy(1)]),
        Err(ProxyRegistryError::CapacityExceeded { available: 0, .. })
    ));
    assert_eq!(snapshot(&registry), before);
    assert_eq!(registry.insert_sector(sector(1), &[]).unwrap(), []);
    assert_eq!(registry.sector_count(), 1);
    assert_eq!(registry.sector_handles(sector(1)), Some([].as_slice()));
    assert!(!registry.has_pending_uploads());
    let before = snapshot(&registry);
    assert!(matches!(
        registry.insert_sector(sector(1), &[]),
        Err(ProxyRegistryError::SectorAlreadyExists { .. })
    ));
    assert_eq!(snapshot(&registry), before);
    assert!(matches!(
        registry.insert_sector(sector(2), &[proxy(1)]),
        Err(ProxyRegistryError::SectorCapacityExceeded { capacity: 1 })
    ));
    assert_eq!(snapshot(&registry), before);
    registry.mark_all_dirty();
    assert!(!registry.has_pending_uploads());
    assert!(registry.remove_sector(sector(1)));
    assert!(!registry.remove_sector(sector(1)));
    assert!(!registry.has_pending_uploads());
    assert!(ranges(&registry).is_empty());
    assert_invariants(&registry);
}

#[test]
fn empty_sector_bookkeeping_is_explicitly_independent_of_proxy_capacity() {
    let mut registry = ProxyRegistry::with_sector_capacity(0, 128).unwrap();
    assert_eq!(registry.sector_capacity(), 128);
    for id in 0..128 {
        assert!(registry.insert_sector(sector(id), &[]).unwrap().is_empty());
    }
    assert_eq!(registry.sector_count(), 128);
    assert_eq!(registry.capacity(), 0);
    assert!(registry.is_empty());
    assert!(!registry.has_pending_uploads());
    for id in 0..128 {
        assert!(registry.remove_sector(sector(id)));
    }
    assert!(registry.sectors.is_empty());
    assert!(!registry.has_pending_uploads());
    assert_invariants(&registry);
}

#[test]
fn default_sector_budget_is_proxy_capacity_with_a_one_sector_minimum() {
    for proxy_capacity in [0, 1, 2, 8] {
        let registry = ProxyRegistry::new(proxy_capacity).unwrap();
        assert_eq!(registry.capacity(), proxy_capacity);
        assert_eq!(registry.sector_capacity(), proxy_capacity.max(1));
        assert_eq!(registry.sector_count(), 0);
        assert_invariants(&registry);
    }
}

#[test]
fn explicit_zero_sector_budget_rejects_all_inputs_atomically() {
    for proxy_capacity in [0, 4] {
        let mut registry = ProxyRegistry::with_sector_capacity(proxy_capacity, 0).unwrap();
        assert_eq!(registry.sector_capacity(), 0);
        let mut bad = proxy(1);
        bad.radius = f32::NAN;
        for instances in [vec![], vec![proxy(1)], vec![bad], vec![proxy(1), proxy(1)]] {
            let before = snapshot(&registry);
            assert!(matches!(
                registry.insert_sector(sector(1), &instances),
                Err(ProxyRegistryError::SectorCapacityExceeded { capacity: 0 })
            ));
            assert_eq!(snapshot(&registry), before);
        }
        assert_eq!(registry.sector_count(), 0);
        assert!(!registry.remove_sector(sector(1)));
        assert!(!registry.has_pending_uploads());
        assert_invariants(&registry);
    }
}

#[test]
fn sector_budget_can_exhaust_while_proxy_slots_remain_available() {
    let mut registry = ProxyRegistry::with_sector_capacity(8, 1).unwrap();
    registry
        .insert_sector(sector(1), &[proxy(1), proxy(2)])
        .unwrap();
    registry.acknowledge_upload();
    assert_eq!(registry.len(), 2);
    assert_eq!(registry.free.len(), 6);
    assert_eq!(registry.sector_count(), 1);
    assert_eq!(registry.sector_capacity(), 1);
    let before = snapshot(&registry);
    assert!(matches!(
        registry.insert_sector(sector(2), &[proxy(3)]),
        Err(ProxyRegistryError::SectorCapacityExceeded { capacity: 1 })
    ));
    assert_eq!(snapshot(&registry), before);
    assert_invariants(&registry);
}

#[test]
fn empty_sector_flood_is_bounded_in_default_and_explicit_configurations() {
    for mut registry in [
        ProxyRegistry::new(0).unwrap(),
        ProxyRegistry::with_sector_capacity(0, 0).unwrap(),
        ProxyRegistry::with_sector_capacity(0, 3).unwrap(),
        ProxyRegistry::with_sector_capacity(5, 2).unwrap(),
    ] {
        let capacity = registry.sector_capacity();
        let initial_allocations = allocations(&registry);
        for id in 0..2_048 {
            if registry.sector_count() < capacity {
                assert!(registry.insert_sector(sector(id), &[]).unwrap().is_empty());
            } else {
                let before = snapshot(&registry);
                assert!(matches!(
                    registry.insert_sector(sector(id), &[]),
                    Err(ProxyRegistryError::SectorCapacityExceeded { capacity: actual }) if actual == capacity
                ));
                assert_eq!(snapshot(&registry), before);
            }
            assert!(registry.sector_count() <= capacity);
            assert!(registry.is_empty());
            assert!(!registry.has_pending_uploads());
            assert_eq!(allocations(&registry), initial_allocations);
        }
        assert_eq!(registry.sector_count(), capacity);
        assert_invariants(&registry);
    }
}

#[test]
fn duplicate_then_sector_budget_then_proxy_validation_has_atomic_precedence() {
    let mut registry = ProxyRegistry::with_sector_capacity(2, 1).unwrap();
    registry.insert_sector(sector(1), &[proxy(1)]).unwrap();
    let mut bad = proxy(1);
    bad.position[0] = f32::INFINITY;
    let inputs = [
        vec![],
        vec![bad],
        vec![proxy(1), proxy(1)],
        vec![proxy(1); 3],
    ];
    for pending in [true, false] {
        if !pending {
            registry.acknowledge_upload();
        }
        let before = snapshot(&registry);
        for instances in &inputs {
            assert!(matches!(
                registry.insert_sector(sector(1), instances),
                Err(ProxyRegistryError::SectorAlreadyExists { .. })
            ));
            assert_eq!(snapshot(&registry), before);
            assert!(matches!(
                registry.insert_sector(sector(2), instances),
                Err(ProxyRegistryError::SectorCapacityExceeded { capacity: 1 })
            ));
            assert_eq!(snapshot(&registry), before);
        }
    }
    registry.remove_sector(sector(1));
    let before = snapshot(&registry);
    assert!(matches!(
        registry.insert_sector(sector(2), &[bad]),
        Err(ProxyRegistryError::InvalidProxy { .. })
    ));
    assert_eq!(snapshot(&registry), before);
    assert!(matches!(
        registry.insert_sector(sector(2), &[proxy(1), proxy(1)]),
        Err(ProxyRegistryError::DuplicateProxyId { .. })
    ));
    assert_eq!(snapshot(&registry), before);
    assert!(matches!(
        registry.insert_sector(sector(2), &[proxy(1); 3]),
        Err(ProxyRegistryError::CapacityExceeded { .. })
    ));
    assert_eq!(snapshot(&registry), before);
    registry.insert_sector(sector(2), &[proxy(1)]).unwrap();
    assert_eq!(registry.sector_count(), 1);
    assert_invariants(&registry);
}

#[test]
fn removing_empty_and_nonempty_sectors_restores_exactly_one_budget_unit() {
    let mut registry = ProxyRegistry::with_sector_capacity(3, 2).unwrap();
    registry.insert_sector(sector(1), &[]).unwrap();
    let old = registry.insert_sector(sector(2), &[proxy(1)]).unwrap()[0];
    registry.acknowledge_upload();
    assert_eq!(registry.sector_count(), 2);
    let before = snapshot(&registry);
    assert!(!registry.remove_sector(sector(99)));
    assert_eq!(snapshot(&registry), before);
    assert!(registry.remove_sector(sector(1)));
    assert_eq!(registry.sector_count(), 1);
    assert!(!registry.has_pending_uploads());
    registry.insert_sector(sector(3), &[]).unwrap();
    assert_eq!(registry.sector_count(), 2);
    assert!(registry.remove_sector(sector(2)));
    assert_eq!(registry.sector_count(), 1);
    assert!(registry.has_pending_uploads());
    assert_eq!(registry.get(old), None);
    registry
        .insert_sector(sector(4), &[proxy(1), proxy(2), proxy(3)])
        .unwrap();
    assert_eq!(registry.sector_count(), 2);
    assert_eq!(registry.sector_capacity(), 2);
    assert_eq!(registry.len(), 3);
    assert_invariants(&registry);
}

#[test]
fn full_capacity_rejects_insert_atomically_but_accepts_empty_sectors() {
    let mut registry = ProxyRegistry::new(2).unwrap();
    registry
        .insert_sector(sector(1), &[proxy(1), proxy(2)])
        .unwrap();
    registry.acknowledge_upload();
    let before = snapshot(&registry);
    assert!(matches!(
        registry.insert_sector(sector(2), &[proxy(3)]),
        Err(ProxyRegistryError::CapacityExceeded {
            requested: 1,
            available: 0
        })
    ));
    assert_eq!(snapshot(&registry), before);
    registry.insert_sector(sector(2), &[]).unwrap();
    assert!(!registry.has_pending_uploads());
    assert!(registry.remove_sector(sector(2)));
    assert!(!registry.has_pending_uploads());
    registry.remove_sector(sector(1));
    assert_eq!(
        registry
            .insert_sector(sector(1), &[proxy(3), proxy(4)])
            .unwrap()
            .len(),
        2
    );
    assert_invariants(&registry);
}

#[test]
fn capacity_above_u32_max_is_rejected_without_allocation() {
    if let Some(capacity) = (u32::MAX as usize).checked_add(1) {
        assert!(
            matches!(ProxyRegistry::new(capacity), Err(ProxyRegistryError::CapacityTooLarge { capacity: actual }) if actual == capacity)
        );
    }
}

#[test]
fn failed_reservation_exposes_an_error_source() {
    let error = reserved::<u64>(usize::MAX).unwrap_err();
    assert!(matches!(error, ProxyRegistryError::AllocationFailed(_)));
    assert!(error.source().is_some());
    assert!(!error.to_string().is_empty());
}

#[test]
fn duplicate_sector_or_nonadjacent_proxy_ids_leave_every_field_unchanged() {
    let mut registry = ProxyRegistry::new(8).unwrap();
    registry
        .insert_sector(sector(1), &[proxy(1), proxy(2)])
        .unwrap();
    registry.acknowledge_upload();
    for pending in [false, true] {
        if pending {
            registry.mark_all_dirty();
        }
        let before = snapshot(&registry);
        assert!(matches!(
            registry.insert_sector(sector(1), &[proxy(3)]),
            Err(ProxyRegistryError::SectorAlreadyExists { .. })
        ));
        assert_eq!(snapshot(&registry), before);
        assert!(matches!(
            registry.insert_sector(sector(2), &[proxy(9), proxy(3), proxy(9)]),
            Err(ProxyRegistryError::DuplicateProxyId { proxy_id: 9 })
        ));
        assert_eq!(snapshot(&registry), before);
        assert_eq!(registry.sector_handles(sector(2)), None);
    }
}

#[test]
fn dropping_prepared_sector_update_preserves_every_registry_field() {
    let mut registry = ProxyRegistry::with_sector_capacity(3, 2).unwrap();
    registry
        .insert_sector(sector(1), &[proxy(1), proxy(2)])
        .unwrap();
    registry.acknowledge_upload();
    let before = snapshot(&registry);

    drop(
        registry
            .prepare_sector_update(sector(1), vec![proxy(7), proxy(8), proxy(9)])
            .unwrap(),
    );

    assert_eq!(snapshot(&registry), before);
    assert_invariants(&registry);
}

#[test]
fn prepared_replacement_reuses_full_sector_capacity_and_invalidates_old_handles() {
    let mut registry = ProxyRegistry::with_sector_capacity(3, 2).unwrap();
    let old = registry
        .insert_sector(sector(1), &[proxy(1), proxy(2)])
        .unwrap();
    let survivor = registry.insert_sector(sector(2), &[proxy(3)]).unwrap()[0];
    registry.acknowledge_upload();

    registry
        .prepare_sector_update(sector(1), vec![proxy(7), proxy(8)])
        .unwrap()
        .commit();

    assert!(old.into_iter().all(|handle| registry.get(handle).is_none()));
    assert_eq!(registry.get(survivor), Some(&proxy(3)));
    let replacement = registry.sector_handles(sector(1)).unwrap();
    assert_eq!(replacement.len(), 2);
    assert_eq!(registry.get(replacement[0]), Some(&proxy(7)));
    assert_eq!(registry.get(replacement[1]), Some(&proxy(8)));
    assert_eq!(registry.len(), 3);
    assert_eq!(registry.sector_count(), 2);
    assert!(registry.has_pending_uploads());
    assert_invariants(&registry);
}

#[test]
fn replacement_rejects_slots_that_must_retire_without_mutation() {
    let mut registry = ProxyRegistry::new(1).unwrap();
    registry.insert_sector(sector(1), &[proxy(1)]).unwrap();
    set_generation(&mut registry, sector(1), u64::MAX - 1);
    registry.acknowledge_upload();
    let before = snapshot(&registry);

    assert!(matches!(
        registry.prepare_sector_update(sector(1), vec![proxy(2)]),
        Err(ProxyRegistryError::CapacityExceeded {
            requested: 1,
            available: 0
        })
    ));

    assert_eq!(snapshot(&registry), before);
    assert_invariants(&registry);
}

#[test]
fn prepared_new_sector_applies_normal_sector_and_proxy_budgets() {
    let mut registry = ProxyRegistry::with_sector_capacity(2, 1).unwrap();
    registry
        .prepare_sector_update(sector(1), vec![proxy(1), proxy(2)])
        .unwrap()
        .commit();
    assert_eq!(registry.len(), 2);
    assert_eq!(registry.sector_count(), 1);
    let before = snapshot(&registry);
    assert!(matches!(
        registry.prepare_sector_update(sector(2), Vec::new()),
        Err(ProxyRegistryError::SectorCapacityExceeded { capacity: 1 })
    ));
    assert_eq!(snapshot(&registry), before);
    assert_invariants(&registry);
}

fn invalid_proxies() -> Vec<ProxyInstance> {
    let mut invalid = Vec::new();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for axis in 0..3 {
            let mut p = proxy(1);
            p.position[axis] = value;
            invalid.push(p);
        }
        let mut radius = proxy(1);
        radius.radius = value;
        invalid.push(radius);
        let mut error = proxy(1);
        error.geometric_error = value;
        invalid.push(error);
        let mut distance = proxy(1);
        distance.max_distance = value;
        invalid.push(distance);
    }
    let mut radius = proxy(1);
    radius.radius = -1.0;
    invalid.push(radius);
    let mut error = proxy(1);
    error.geometric_error = -1.0;
    invalid.push(error);
    let mut distance = proxy(1);
    distance.max_distance = -1.0;
    invalid.push(distance);
    invalid
}

#[test]
fn every_invalid_numeric_field_rejects_insert_and_update_atomically() {
    let mut registry = ProxyRegistry::new(4).unwrap();
    let handle = registry.insert_sector(sector(1), &[proxy(1)]).unwrap()[0];
    for pending in [false, true] {
        registry.acknowledge_upload();
        if pending {
            registry.mark_all_dirty();
        }
        for bad in invalid_proxies() {
            let before = snapshot(&registry);
            assert!(matches!(
                registry.insert_sector(sector(2), &[proxy(2), bad]),
                Err(ProxyRegistryError::InvalidProxy { proxy_id: 1 })
            ));
            assert_eq!(snapshot(&registry), before);
            assert!(matches!(
                registry.update(handle, bad),
                Err(ProxyRegistryError::InvalidProxy { proxy_id: 1 })
            ));
            assert_eq!(snapshot(&registry), before);
        }
    }
}

#[test]
fn identity_changes_and_invalid_handles_are_atomic() {
    let mut registry = ProxyRegistry::new(2).unwrap();
    let handle = registry.insert_sector(sector(1), &[proxy(1)]).unwrap()[0];
    registry.acknowledge_upload();
    let before = snapshot(&registry);
    assert!(matches!(
        registry.update(handle, proxy(2)),
        Err(ProxyRegistryError::ProxyIdChanged {
            expected: 1,
            actual: 2
        })
    ));
    assert_eq!(snapshot(&registry), before);
    for invalid in [
        ProxyHandle {
            slot: u32::MAX,
            generation: 0,
        },
        ProxyHandle {
            slot: 1,
            generation: 0,
        },
        ProxyHandle {
            slot: handle.slot(),
            generation: 1,
        },
    ] {
        assert_eq!(registry.get(invalid), None);
        assert_eq!(registry.dense_slot(invalid), None);
        assert!(matches!(
            registry.update(invalid, proxy(1)),
            Err(ProxyRegistryError::InvalidHandle { .. })
        ));
        assert_eq!(snapshot(&registry), before);
    }
    assert!(!registry.remove_sector(sector(99)));
    assert_eq!(snapshot(&registry), before);
}

#[test]
fn dirty_ranges_coalesce_without_acknowledging_and_ignore_inactive_rows() {
    let mut registry = ProxyRegistry::new(12).unwrap();
    let proxies: Vec<_> = (0..8).map(proxy).collect();
    let handles = registry.insert_sector(sector(1), &proxies).unwrap();
    assert_eq!(ranges(&registry), vec![0..8]);
    registry.acknowledge_upload();
    assert!(!registry.has_pending_uploads());
    for index in [0, 1, 3, 6, 7] {
        registry.update(handles[index], proxies[index]).unwrap();
    }
    assert_eq!(ranges(&registry), vec![0..2, 3..4, 6..8]);
    assert_eq!(ranges(&registry), vec![0..2, 3..4, 6..8]);
    assert!(registry.has_pending_uploads());
    registry.update(handles[2], proxies[2]).unwrap();
    assert_eq!(ranges(&registry), vec![0..4, 6..8]);
    registry.mark_all_dirty();
    assert_eq!(ranges(&registry), vec![0..8]);
    registry.acknowledge_upload();
    assert!(ranges(&registry).is_empty());
    assert!(!registry.has_pending_uploads());
    assert_invariants(&registry);
}

#[test]
fn tail_only_deletion_requires_count_upload_even_with_no_dirty_ranges() {
    let mut registry = ProxyRegistry::new(3).unwrap();
    registry.insert_sector(sector(1), &[proxy(1)]).unwrap();
    registry.insert_sector(sector(2), &[proxy(2)]).unwrap();
    registry.acknowledge_upload();
    registry.remove_sector(sector(2));
    assert_eq!(registry.len(), 1);
    assert!(ranges(&registry).is_empty());
    assert!(registry.has_pending_uploads());
    registry.acknowledge_upload();
    assert!(!registry.has_pending_uploads());
    registry.remove_sector(sector(1));
    assert!(registry.is_empty());
    assert!(ranges(&registry).is_empty());
    assert!(registry.has_pending_uploads());
    registry.mark_all_dirty();
    assert!(registry.has_pending_uploads());
    registry.acknowledge_upload();
    assert!(!registry.has_pending_uploads());
    assert_invariants(&registry);
}

#[test]
fn deleting_dirty_tail_clears_it_and_reinsert_marks_only_new_prefix() {
    let mut registry = ProxyRegistry::new(8).unwrap();
    registry
        .insert_sector(sector(1), &[proxy(1), proxy(2)])
        .unwrap();
    registry
        .insert_sector(sector(2), &[proxy(3), proxy(4)])
        .unwrap();
    assert_eq!(ranges(&registry), vec![0..4]);
    registry.remove_sector(sector(2));
    assert_eq!(ranges(&registry), vec![0..2]);
    assert_invariants(&registry);
    registry.remove_sector(sector(1));
    assert!(registry.dirty.iter().all(|dirty| !dirty));
    assert!(ranges(&registry).is_empty());
    registry.acknowledge_upload();
    registry.insert_sector(sector(3), &[proxy(3)]).unwrap();
    assert_eq!(ranges(&registry), vec![0..1]);
    assert_invariants(&registry);
}

#[test]
fn core_vector_allocations_never_grow_under_repeated_full_reuse() {
    let mut registry = ProxyRegistry::new(32).unwrap();
    let initial = allocations(&registry);
    assert!(initial.into_iter().all(|capacity| capacity >= 32));
    let proxies: Vec<_> = (0..32).map(proxy).collect();
    for _ in 0..256 {
        registry.insert_sector(sector(1), &proxies).unwrap();
        registry.acknowledge_upload();
        registry.remove_sector(sector(1));
        assert_eq!(allocations(&registry), initial);
        assert_invariants(&registry);
    }
}

#[test]
fn validation_errors_have_display_and_implement_error() {
    let handle = ProxyHandle {
        slot: 7,
        generation: 3,
    };
    let errors = [
        ProxyRegistryError::CapacityTooLarge {
            capacity: usize::MAX,
        },
        ProxyRegistryError::CapacityExceeded {
            requested: 3,
            available: 2,
        },
        ProxyRegistryError::SectorAlreadyExists { sector: sector(1) },
        ProxyRegistryError::SectorCapacityExceeded { capacity: 2 },
        ProxyRegistryError::DuplicateProxyId { proxy_id: 1 },
        ProxyRegistryError::InvalidProxy { proxy_id: 1 },
        ProxyRegistryError::InvalidHandle { handle },
        ProxyRegistryError::ProxyIdChanged {
            expected: 1,
            actual: 2,
        },
    ];
    for error in errors {
        let error: &dyn Error = &error;
        assert!(!error.to_string().is_empty());
        assert!(error.source().is_none());
    }
}

#[derive(Default)]
struct Model {
    sectors: BTreeMap<SectorId, Vec<(ProxyHandle, ProxyInstance)>>,
    dense: Vec<ProxyHandle>,
    dirty: Vec<bool>,
    pending: bool,
    stale: Vec<ProxyHandle>,
}

impl Model {
    fn check(&self, registry: &ProxyRegistry) {
        assert_invariants(registry);
        assert_eq!(self.dense.len(), registry.len());
        assert_eq!(self.pending, registry.has_pending_uploads());
        let expected_dirty: Vec<_> = registry.dirty_ranges().flatten().collect();
        let model_dirty: Vec<_> = self
            .dirty
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| d.then_some(i))
            .collect();
        assert_eq!(expected_dirty, model_dirty);
        assert_eq!(self.sectors.len(), registry.sector_count());
        for (&sector, members) in &self.sectors {
            let expected: Vec<_> = members.iter().map(|&(handle, _)| handle).collect();
            assert_eq!(registry.sector_handles(sector), Some(expected.as_slice()));
            for &(handle, instance) in members {
                assert_eq!(registry.get(handle), Some(&instance));
            }
        }
        for (index, &handle) in self.dense.iter().enumerate() {
            assert_eq!(registry.handle_at_dense(index), Some(handle));
            assert_eq!(registry.dense_slot(handle), Some(index));
        }
        for &handle in &self.stale {
            assert_eq!(registry.get(handle), None);
            assert_eq!(registry.dense_slot(handle), None);
        }
    }
}

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn randomized_sequences_match_an_independent_sector_and_dense_model() {
    for (capacity, sector_capacity, seed) in [
        (0, 0, 1),
        (0, 1, 3),
        (1, 8, 9),
        (7, 1, 0x1234),
        (7, 20, 0x1235),
        (32, 0, 0x1234_5678),
        (32, 3, 0x1234_5679),
    ] {
        let mut rng = seed;
        let mut registry = ProxyRegistry::with_sector_capacity(capacity, sector_capacity).unwrap();
        let initial = allocations(&registry);
        let mut model = Model::default();
        let mut operations = [0; 10];
        for step in 0..6_000 {
            let which = random(&mut rng) as usize % operations.len();
            operations[which] += 1;
            let sector_id = sector((random(&mut rng) % 12) as i64);
            match which {
                0..=2 => {
                    let count = (random(&mut rng) % 5) as usize;
                    let instances: Vec<_> = (0..count).map(|id| proxy(id as u32)).collect();
                    let valid = !model.sectors.contains_key(&sector_id)
                        && model.sectors.len() < sector_capacity
                        && count <= capacity - model.dense.len();
                    let before = snapshot(&registry);
                    match registry.insert_sector(sector_id, &instances) {
                        Ok(handles) => {
                            assert!(valid, "seed {seed}, step {step}");
                            model.pending |= count != 0;
                            model.dense.extend_from_slice(&handles);
                            model.dirty.extend(std::iter::repeat_n(true, count));
                            model
                                .sectors
                                .insert(sector_id, handles.into_iter().zip(instances).collect());
                        }
                        Err(_) => {
                            assert!(!valid, "seed {seed}, step {step}");
                            assert_eq!(snapshot(&registry), before);
                        }
                    }
                }
                3 => {
                    let members = model.sectors.remove(&sector_id);
                    assert_eq!(registry.remove_sector(sector_id), members.is_some());
                    if let Some(members) = members {
                        model.pending |= !members.is_empty();
                        for (handle, _) in members {
                            let index = model.dense.iter().position(|&h| h == handle).unwrap();
                            model.dense.swap_remove(index);
                            model.dirty.swap_remove(index);
                            if index < model.dirty.len() {
                                model.dirty[index] = true;
                            }
                            model.stale.push(handle);
                        }
                    }
                }
                4 if !model.dense.is_empty() => {
                    let index = random(&mut rng) as usize % model.dense.len();
                    let handle = model.dense[index];
                    let instance = model
                        .sectors
                        .values_mut()
                        .flatten()
                        .find(|(h, _)| *h == handle)
                        .unwrap();
                    instance.1.position[0] = step as f32;
                    instance.1.flags = step;
                    registry.update(handle, instance.1).unwrap();
                    model.dirty[index] = true;
                    model.pending = true;
                }
                5 => {
                    registry.acknowledge_upload();
                    model.dirty.fill(false);
                    model.pending = false;
                }
                6 => {
                    registry.mark_all_dirty();
                    model.dirty.fill(true);
                    model.pending |= !model.dense.is_empty();
                }
                7 => {
                    let before = snapshot(&registry);
                    assert!(registry
                        .insert_sector(sector_id, &[proxy(1), proxy(1)])
                        .is_err());
                    assert_eq!(snapshot(&registry), before);
                }
                8 => {
                    let before = snapshot(&registry);
                    let mut bad = proxy(0);
                    bad.position[2] = f32::NAN;
                    assert!(registry.insert_sector(sector_id, &[bad]).is_err());
                    assert_eq!(snapshot(&registry), before);
                    if let Some(&handle) = model.dense.first() {
                        bad.proxy_id = registry.get(handle).unwrap().proxy_id;
                        assert!(registry.update(handle, bad).is_err());
                        assert_eq!(snapshot(&registry), before);
                    }
                }
                9 if !model.stale.is_empty() => {
                    let handle = model.stale[random(&mut rng) as usize % model.stale.len()];
                    let before = snapshot(&registry);
                    assert!(matches!(
                        registry.update(handle, proxy(0)),
                        Err(ProxyRegistryError::InvalidHandle { .. })
                    ));
                    assert_eq!(snapshot(&registry), before);
                }
                _ => {}
            }
            assert_eq!(allocations(&registry), initial);
            model.check(&registry);
        }
        assert!(operations.into_iter().all(|count| count > 0));
        for sector_id in model.sectors.keys().copied().collect::<Vec<_>>() {
            registry.remove_sector(sector_id);
        }
        assert!(registry.is_empty());
        assert!(registry.dirty.iter().all(|dirty| !dirty));
        assert!(ranges(&registry).is_empty());
        registry.acknowledge_upload();
        assert!(!registry.has_pending_uploads());
        assert_invariants(&registry);
    }
}
