#[cfg(test)]
mod deferred_payload_tests {
    use crate::{
        Asset, AssetError, AssetEvent, AssetStores, AssetTypeId, Assets, ErasedAssetStore,
        ErrorRegistry, LoadState, StoreError, UntypedAssetId,
    };
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::any::Any;
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct Value {
        serial: usize,
        drops: Arc<AtomicUsize>,
    }

    impl Value {
        fn new(serial: usize, drops: &Arc<AtomicUsize>) -> Self {
            Self {
                serial,
                drops: Arc::clone(drops),
            }
        }
    }

    impl Asset for Value {
        const TYPE_NAME: &'static str = "stores_deferred_tests::Value";
    }

    impl Drop for Value {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct Other(Value);

    impl Asset for Other {
        const TYPE_NAME: &'static str = "stores_deferred_tests::Other";
    }

    #[test]
    fn fulfilling_pending_slot_adds_value_without_retiring_it() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut stores = AssetStores::new();
        let handle = stores.reserve::<Value>();
        let id = handle.id();
        let typed = id.typed::<Value>().unwrap();
        let mut retired = Vec::new();

        assert_eq!(stores.load_state(id), LoadState::Loading);
        assert_eq!(
            stores.fulfill_erased_deferred(id, Box::new(Value::new(1, &drops)), &mut retired),
            Ok(true)
        );
        assert!(retired.is_empty());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(stores.get(typed).unwrap().serial, 1);
        assert_eq!(stores.load_state(id), LoadState::Loaded);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Added { id: typed }]
        );

        drop(stores);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn replacing_ready_payloads_appends_old_values_until_explicit_drain() {
        let first_drops = Arc::new(AtomicUsize::new(0));
        let second_drops = Arc::new(AtomicUsize::new(0));
        let current_drops = Arc::new(AtomicUsize::new(0));
        let mut stores = AssetStores::new();
        let handle = stores.insert(Value::new(1, &first_drops));
        let id = handle.id();
        let typed = id.typed::<Value>().unwrap();
        stores.arena_mut::<Value>().unwrap().drain_events();
        let mut retired = Vec::new();

        assert_eq!(
            stores.fulfill_erased_deferred(
                id,
                Box::new(Value::new(2, &second_drops)),
                &mut retired,
            ),
            Ok(true)
        );
        assert_eq!(first_drops.load(Ordering::SeqCst), 0);
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].downcast_ref::<Value>().unwrap().serial, 1);
        assert_eq!(stores.get(typed).unwrap().serial, 2);

        assert_eq!(
            stores.fulfill_erased_deferred(
                id,
                Box::new(Value::new(3, &current_drops)),
                &mut retired,
            ),
            Ok(true)
        );
        assert_eq!(retired.len(), 2);
        assert_eq!(retired[0].downcast_ref::<Value>().unwrap().serial, 1);
        assert_eq!(retired[1].downcast_ref::<Value>().unwrap().serial, 2);
        assert_eq!(first_drops.load(Ordering::SeqCst), 0);
        assert_eq!(second_drops.load(Ordering::SeqCst), 0);
        assert_eq!(current_drops.load(Ordering::SeqCst), 0);
        assert_eq!(stores.get(typed).unwrap().serial, 3);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [
                AssetEvent::Modified { id: typed },
                AssetEvent::Modified { id: typed },
            ]
        );
        assert_eq!(first_drops.load(Ordering::SeqCst), 0);
        assert_eq!(second_drops.load(Ordering::SeqCst), 0);

        drop(retired.drain(..));
        assert!(retired.is_empty());
        assert_eq!(first_drops.load(Ordering::SeqCst), 1);
        assert_eq!(second_drops.load(Ordering::SeqCst), 1);
        assert_eq!(current_drops.load(Ordering::SeqCst), 0);
        drop(stores);
        assert_eq!(current_drops.load(Ordering::SeqCst), 1);
        assert_eq!(first_drops.load(Ordering::SeqCst), 1);
        assert_eq!(second_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failing_ready_payload_retires_it_once_and_allows_later_fulfillment() {
        let old_drops = Arc::new(AtomicUsize::new(0));
        let new_drops = Arc::new(AtomicUsize::new(0));
        let mut errors = ErrorRegistry::new();
        let error = errors.record(AssetError::new("test.asset", "failed"));
        let mut stores = AssetStores::new();
        let handle = stores.insert(Value::new(1, &old_drops));
        let id = handle.id();
        let typed = id.typed::<Value>().unwrap();
        stores.arena_mut::<Value>().unwrap().drain_events();
        let mut retired = Vec::new();

        assert!(stores.fail_erased_deferred(id, error, &mut retired));
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].downcast_ref::<Value>().unwrap().serial, 1);
        assert_eq!(old_drops.load(Ordering::SeqCst), 0);
        assert!(stores.contains(id));
        assert!(stores.get(typed).is_none());
        assert_eq!(stores.load_state(id), LoadState::Failed(error));
        assert!(stores.fail_erased_deferred(id, error, &mut retired));
        assert_eq!(retired.len(), 1);

        assert_eq!(
            stores.fulfill_erased_deferred(id, Box::new(Value::new(2, &new_drops)), &mut retired),
            Ok(true)
        );
        assert_eq!(retired.len(), 1);
        assert_eq!(stores.get(typed).unwrap().serial, 2);
        assert_eq!(stores.load_state(id), LoadState::Loaded);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [
                AssetEvent::Failed { id: typed, error },
                AssetEvent::Failed { id: typed, error },
                AssetEvent::Added { id: typed },
            ]
        );
        assert_eq!(old_drops.load(Ordering::SeqCst), 0);
        assert_eq!(new_drops.load(Ordering::SeqCst), 0);

        drop(retired.drain(..));
        assert_eq!(old_drops.load(Ordering::SeqCst), 1);
        assert_eq!(new_drops.load(Ordering::SeqCst), 0);
        drop(stores);
        assert_eq!(old_drops.load(Ordering::SeqCst), 1);
        assert_eq!(new_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failing_pending_slot_has_no_payload_to_retire() {
        let mut errors = ErrorRegistry::new();
        let error = errors.record(AssetError::new("test.asset", "failed"));
        let mut stores = AssetStores::new();
        let handle = stores.reserve::<Value>();
        let id = handle.id();
        let typed = id.typed::<Value>().unwrap();
        let mut retired = Vec::new();

        assert!(stores.fail_erased_deferred(id, error, &mut retired));
        assert!(retired.is_empty());
        assert!(stores.contains(id));
        assert!(stores.get(typed).is_none());
        assert_eq!(stores.load_state(id), LoadState::Failed(error));
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Failed { id: typed, error }]
        );
    }

    #[test]
    fn unknown_arena_rejects_input_into_queue_without_registering_a_store() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut source = Assets::<Value>::new();
        let handle = source.reserve();
        let id = handle.id().untyped();
        let candidate: Box<dyn Any + Send> = Box::new(Value::new(1, &drops));
        let mut stores = AssetStores::new();
        let mut retired = Vec::new();
        let mut errors = ErrorRegistry::new();
        let error = errors.record(AssetError::new("test.asset", "failed"));

        assert_eq!(
            stores.validate_fulfill_erased(id, candidate.as_ref()),
            Err(StoreError::UnknownType(AssetTypeId::of::<Value>()))
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(stores.is_empty());
        assert_eq!(
            stores.fulfill_erased_deferred(id, candidate, &mut retired),
            Err(StoreError::UnknownType(AssetTypeId::of::<Value>()))
        );
        assert!(!stores.fail_erased_deferred(id, error, &mut retired));
        assert!(stores.is_empty());
        assert_eq!(stores.total_assets(), 0);
        assert_eq!(retired.len(), 1);
        let rejected = retired[0].downcast_ref::<Value>().unwrap();
        assert_eq!(rejected.serial, 1);
        assert!(Arc::ptr_eq(&rejected.drops, &drops));
        assert_eq!(source.load_state(handle.id()), LoadState::Loading);
        assert!(source.drain_events().is_empty());
        drop(stores);
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        drop(retired.drain(..));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(retired);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn erased_arena_rejects_wrong_typed_id_without_touching_matching_slot_index() {
        let live_drops = Arc::new(AtomicUsize::new(0));
        let rejected_drops = Arc::new(AtomicUsize::new(0));
        let mut arena = Assets::<Value>::new();
        let handle = arena.insert(Value::new(1, &live_drops));
        let typed = handle.id();
        // Use the live slot's exact index/generation with another public type tag.
        let wrong_id = UntypedAssetId::new(typed.index(), AssetTypeId::of::<Other>());
        let candidate: Box<dyn Any + Send> = Box::new(Value::new(2, &rejected_drops));
        let mut retired = Vec::new();
        let mut errors = ErrorRegistry::new();
        let error = errors.record(AssetError::new("test.asset", "failed"));
        let mismatch = StoreError::TypeMismatch {
            expected: AssetTypeId::of::<Value>(),
        };

        {
            let erased: &mut dyn ErasedAssetStore = &mut arena;
            assert_eq!(
                erased.validate_fulfill_erased(wrong_id, candidate.as_ref()),
                Err(mismatch)
            );
            assert_eq!(
                erased.fulfill_erased_deferred(wrong_id, candidate, &mut retired),
                Err(mismatch)
            );
            assert!(!erased.fail_erased_deferred(wrong_id, error, &mut retired));
        }
        assert_eq!(arena.get(typed).unwrap().serial, 1);
        assert_eq!(arena.load_state(typed), LoadState::Loaded);
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.drain_events(), [AssetEvent::Added { id: typed }]);
        assert_eq!(retired.len(), 1);
        let rejected = retired[0].downcast_ref::<Value>().unwrap();
        assert_eq!(rejected.serial, 2);
        assert!(Arc::ptr_eq(&rejected.drops, &rejected_drops));
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        assert_eq!(rejected_drops.load(Ordering::SeqCst), 0);

        drop(retired.drain(..));
        assert_eq!(rejected_drops.load(Ordering::SeqCst), 1);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        drop(arena);
        assert_eq!(live_drops.load(Ordering::SeqCst), 1);
        assert_eq!(rejected_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn boxed_value_type_mismatch_preserves_input_and_current_payload() {
        let live_drops = Arc::new(AtomicUsize::new(0));
        let rejected_drops = Arc::new(AtomicUsize::new(0));
        let mut stores = AssetStores::new();
        let handle = stores.insert(Value::new(1, &live_drops));
        let id = handle.id();
        let typed = id.typed::<Value>().unwrap();
        let candidate: Box<dyn Any + Send> = Box::new(Other(Value::new(2, &rejected_drops)));
        let mut retired = Vec::new();
        let mismatch = StoreError::TypeMismatch {
            expected: AssetTypeId::of::<Value>(),
        };

        assert_eq!(
            stores.validate_fulfill_erased(id, candidate.as_ref()),
            Err(mismatch)
        );
        assert_eq!(
            stores.fulfill_erased_deferred(id, candidate, &mut retired),
            Err(mismatch)
        );
        assert_eq!(retired.len(), 1);
        let rejected = &retired[0].downcast_ref::<Other>().unwrap().0;
        assert_eq!(rejected.serial, 2);
        assert!(Arc::ptr_eq(&rejected.drops, &rejected_drops));
        assert_eq!(stores.get(typed).unwrap().serial, 1);
        assert_eq!(stores.load_state(id), LoadState::Loaded);
        assert_eq!(stores.total_assets(), 1);
        assert!(!stores.contains_type_of::<Other>());
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Added { id: typed }]
        );
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        assert_eq!(rejected_drops.load(Ordering::SeqCst), 0);

        drop(retired.drain(..));
        assert_eq!(rejected_drops.load(Ordering::SeqCst), 1);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        drop(stores);
        assert_eq!(live_drops.load(Ordering::SeqCst), 1);
        assert_eq!(rejected_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stale_id_rejects_inputs_before_and_after_slot_reuse_until_explicit_drain() {
        let vacant_drops = Arc::new(AtomicUsize::new(0));
        let reused_drops = Arc::new(AtomicUsize::new(0));
        let live_drops = Arc::new(AtomicUsize::new(0));
        let mut stores = AssetStores::new();
        let stale_handle = stores.reserve::<Value>();
        let stale = stale_handle.id();
        let stale_typed = stale.typed::<Value>().unwrap();
        assert!(stores
            .arena_mut::<Value>()
            .unwrap()
            .remove(stale_typed)
            .is_none());
        let mut errors = ErrorRegistry::new();
        let error = errors.record(AssetError::new("test.asset", "failed"));
        let mut retired = Vec::new();
        let candidate: Box<dyn Any + Send> = Box::new(Value::new(1, &vacant_drops));

        assert_eq!(
            stores.validate_fulfill_erased(stale, candidate.as_ref()),
            Ok(false)
        );
        assert_eq!(
            stores.fulfill_erased_deferred(stale, candidate, &mut retired),
            Ok(false)
        );
        assert!(!stores.fail_erased_deferred(stale, error, &mut retired));
        assert_eq!(stores.total_assets(), 0);
        assert_eq!(stores.load_state(stale), LoadState::NotLoaded);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Removed { id: stale_typed }]
        );

        let live_handle = stores.insert(Value::new(3, &live_drops));
        let live = live_handle.id();
        let live_typed = live.typed::<Value>().unwrap();
        assert_eq!(stale.index().index(), live.index().index());
        assert_ne!(stale.index().generation(), live.index().generation());
        let candidate: Box<dyn Any + Send> = Box::new(Value::new(2, &reused_drops));
        assert_eq!(
            stores.validate_fulfill_erased(stale, candidate.as_ref()),
            Ok(false)
        );
        assert_eq!(
            stores.fulfill_erased_deferred(stale, candidate, &mut retired),
            Ok(false)
        );
        assert!(!stores.fail_erased_deferred(stale, error, &mut retired));
        assert!(!stores.contains(stale));
        assert_eq!(stores.load_state(stale), LoadState::NotLoaded);
        assert_eq!(stores.get(live_typed).unwrap().serial, 3);
        assert_eq!(stores.load_state(live), LoadState::Loaded);
        assert_eq!(stores.total_assets(), 1);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Added { id: live_typed }]
        );
        assert_eq!(retired.len(), 2);
        let first = retired[0].downcast_ref::<Value>().unwrap();
        let second = retired[1].downcast_ref::<Value>().unwrap();
        assert_eq!(first.serial, 1);
        assert!(Arc::ptr_eq(&first.drops, &vacant_drops));
        assert_eq!(second.serial, 2);
        assert!(Arc::ptr_eq(&second.drops, &reused_drops));
        assert_eq!(vacant_drops.load(Ordering::SeqCst), 0);
        assert_eq!(reused_drops.load(Ordering::SeqCst), 0);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);

        drop(retired.drain(..));
        assert_eq!(vacant_drops.load(Ordering::SeqCst), 1);
        assert_eq!(reused_drops.load(Ordering::SeqCst), 1);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        drop(stores);
        assert_eq!(live_drops.load(Ordering::SeqCst), 1);
        assert_eq!(vacant_drops.load(Ordering::SeqCst), 1);
        assert_eq!(reused_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn preflight_preserves_pending_ready_failed_payloads_and_queued_events() {
        let live_drops = Arc::new(AtomicUsize::new(0));
        let candidate_drops = Arc::new(AtomicUsize::new(0));
        let candidate: Box<dyn Any + Send> = Box::new(Value::new(2, &candidate_drops));
        let mut stores = AssetStores::new();
        let handle = stores.reserve::<Value>();
        let id = handle.id();
        let typed = id.typed::<Value>().unwrap();
        let mut retired = Vec::new();
        let mut errors = ErrorRegistry::new();
        let error = errors.record(AssetError::new("test.asset", "failed"));

        for _ in 0..2 {
            assert_eq!(
                stores.validate_fulfill_erased(id, candidate.as_ref()),
                Ok(true)
            );
        }
        assert_eq!(stores.load_state(id), LoadState::Loading);
        assert!(stores.get(typed).is_none());
        assert_eq!(stores.arena::<Value>().unwrap().pending_event_count(), 0);
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 0);

        assert_eq!(
            stores.fulfill_erased_deferred(id, Box::new(Value::new(1, &live_drops)), &mut retired),
            Ok(true)
        );
        let current = stores.get(typed).unwrap() as *const Value;
        for _ in 0..2 {
            assert_eq!(
                stores.validate_fulfill_erased(id, candidate.as_ref()),
                Ok(true)
            );
        }
        assert_eq!(stores.get(typed).unwrap() as *const Value, current);
        assert_eq!(stores.get(typed).unwrap().serial, 1);
        assert_eq!(stores.load_state(id), LoadState::Loaded);
        assert_eq!(stores.arena::<Value>().unwrap().pending_event_count(), 1);
        assert!(retired.is_empty());
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 0);

        assert!(stores.fail_erased_deferred(id, error, &mut retired));
        for _ in 0..2 {
            assert_eq!(
                stores.validate_fulfill_erased(id, candidate.as_ref()),
                Ok(true)
            );
        }
        assert!(stores.get(typed).is_none());
        assert!(stores.contains(id));
        assert_eq!(stores.load_state(id), LoadState::Failed(error));
        assert_eq!(stores.arena::<Value>().unwrap().pending_event_count(), 2);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [
                AssetEvent::Added { id: typed },
                AssetEvent::Failed { id: typed, error },
            ]
        );
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].downcast_ref::<Value>().unwrap().serial, 1);
        assert_eq!(candidate.downcast_ref::<Value>().unwrap().serial, 2);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 0);

        drop(stores);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        drop(retired.drain(..));
        assert_eq!(live_drops.load(Ordering::SeqCst), 1);
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 0);
        drop(candidate);
        assert_eq!(candidate_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn erased_collection_invalidates_generation_before_queued_payload_is_dropped() {
        for advancing in [true, false] {
            let old_drops = Arc::new(AtomicUsize::new(0));
            let new_drops = Arc::new(AtomicUsize::new(0));
            let mut arena = Assets::<Value>::new();
            let handle = arena.insert(Value::new(1, &old_drops));
            let old = handle.id();
            arena.drain_events();
            drop(handle);
            let mut retired = Vec::new();

            let reclaimed = {
                let erased: &mut dyn ErasedAssetStore = &mut arena;
                if advancing {
                    erased.collect_releases_deferred(&mut retired)
                } else {
                    erased.collect_releases_without_advancing_deferred(&mut retired)
                }
            };
            assert_eq!(reclaimed, 1);
            assert_eq!(arena.frame(), if advancing { 1 } else { 0 });
            assert!(arena.is_empty());
            assert!(!arena.contains(old));
            assert!(arena.get(old).is_none());
            assert_eq!(arena.load_state(old), LoadState::NotLoaded);
            assert_eq!(
                arena.drain_events(),
                [
                    AssetEvent::Unused { id: old },
                    AssetEvent::Removed { id: old }
                ]
            );
            assert_eq!(retired.len(), 1);
            let queued = retired[0].downcast_ref::<Value>().unwrap();
            assert_eq!(queued.serial, 1);
            assert!(Arc::ptr_eq(&queued.drops, &old_drops));
            assert_eq!(old_drops.load(Ordering::SeqCst), 0);

            let replacement = arena.insert(Value::new(2, &new_drops));
            let new = replacement.id();
            assert_eq!(old.index().index(), new.index().index());
            assert_ne!(old.index().generation(), new.index().generation());
            let reclaimed = {
                let erased: &mut dyn ErasedAssetStore = &mut arena;
                assert_eq!(
                    erased.validate_fulfill_erased(old.untyped(), retired[0].as_ref()),
                    Ok(false)
                );
                assert!(!erased.contains_erased(old.untyped()));
                if advancing {
                    erased.collect_releases_deferred(&mut retired)
                } else {
                    erased.collect_releases_without_advancing_deferred(&mut retired)
                }
            };
            assert_eq!(reclaimed, 0);
            assert_eq!(retired.len(), 1);
            assert_eq!(arena.get(new).unwrap().serial, 2);
            assert_eq!(arena.drain_events(), [AssetEvent::Added { id: new }]);
            assert_eq!(old_drops.load(Ordering::SeqCst), 0);
            assert_eq!(new_drops.load(Ordering::SeqCst), 0);

            drop(arena);
            assert_eq!(old_drops.load(Ordering::SeqCst), 0);
            assert_eq!(new_drops.load(Ordering::SeqCst), 1);
            drop(retired);
            assert_eq!(old_drops.load(Ordering::SeqCst), 1);
            assert_eq!(new_drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn registry_collection_appends_payloads_from_all_arenas_and_preserves_owned_slots() {
        for advancing in [true, false] {
            let queued_drops = Arc::new(AtomicUsize::new(0));
            let value_drops = Arc::new(AtomicUsize::new(0));
            let other_drops = Arc::new(AtomicUsize::new(0));
            let owned_drops = Arc::new(AtomicUsize::new(0));
            let replacement_drops = Arc::new(AtomicUsize::new(0));
            let mut stores = AssetStores::new();
            let value_handle = stores.insert(Value::new(1, &value_drops));
            let other_handle = stores.insert(Other(Value::new(2, &other_drops)));
            let owned_handle = stores.insert(Value::new(3, &owned_drops));
            let value_id = value_handle.id();
            let value_typed = value_id.typed::<Value>().unwrap();
            let other_id = other_handle.id();
            let other_typed = other_id.typed::<Other>().unwrap();
            let owned_typed = owned_handle.id().typed::<Value>().unwrap();
            stores.arena_mut::<Value>().unwrap().drain_events();
            stores.arena_mut::<Other>().unwrap().drain_events();
            drop(value_handle);
            drop(other_handle);
            let mut retired: Vec<Box<dyn Any + Send>> = Vec::new();
            retired.push(Box::new(Value::new(0, &queued_drops)));

            let reclaimed = if advancing {
                stores.collect_releases_deferred(&mut retired)
            } else {
                stores.collect_releases_without_advancing_deferred(&mut retired)
            };
            assert_eq!(reclaimed, 2);
            assert_eq!(stores.total_assets(), 1);
            assert!(!stores.contains(value_id));
            assert!(!stores.contains(other_id));
            assert_eq!(stores.load_state(value_id), LoadState::NotLoaded);
            assert_eq!(stores.load_state(other_id), LoadState::NotLoaded);
            assert_eq!(stores.get(owned_typed).unwrap().serial, 3);
            assert_eq!(stores.load_state(owned_handle.id()), LoadState::Loaded);
            assert_eq!(
                stores.arena::<Value>().unwrap().frame(),
                if advancing { 1 } else { 0 }
            );
            assert_eq!(
                stores.arena::<Other>().unwrap().frame(),
                if advancing { 1 } else { 0 }
            );
            assert_eq!(
                stores.arena_mut::<Value>().unwrap().drain_events(),
                [
                    AssetEvent::Unused { id: value_typed },
                    AssetEvent::Removed { id: value_typed },
                ]
            );
            assert_eq!(
                stores.arena_mut::<Other>().unwrap().drain_events(),
                [
                    AssetEvent::Unused { id: other_typed },
                    AssetEvent::Removed { id: other_typed },
                ]
            );
            assert_eq!(retired.len(), 3);
            assert_eq!(retired[0].downcast_ref::<Value>().unwrap().serial, 0);
            // Arena traversal order is not part of the public contract.
            let queued_value = retired
                .iter()
                .filter_map(|value| value.downcast_ref::<Value>())
                .find(|value| value.serial == 1)
                .unwrap();
            assert!(Arc::ptr_eq(&queued_value.drops, &value_drops));
            let queued_other = retired
                .iter()
                .find_map(|value| value.downcast_ref::<Other>())
                .unwrap();
            assert_eq!(queued_other.0.serial, 2);
            assert!(Arc::ptr_eq(&queued_other.0.drops, &other_drops));

            let replacement = stores.insert(Value::new(4, &replacement_drops));
            let replacement_id = replacement.id();
            assert_eq!(value_id.index().index(), replacement_id.index().index());
            assert_ne!(
                value_id.index().generation(),
                replacement_id.index().generation()
            );
            assert_eq!(
                stores.validate_fulfill_erased(value_id, queued_value),
                Ok(false)
            );
            assert!(!stores.contains(value_id));
            let reclaimed = if advancing {
                stores.collect_releases_deferred(&mut retired)
            } else {
                stores.collect_releases_without_advancing_deferred(&mut retired)
            };
            assert_eq!(reclaimed, 0);
            assert_eq!(retired.len(), 3);
            assert_eq!(stores.total_assets(), 2);
            assert_eq!(
                stores.arena_mut::<Value>().unwrap().drain_events(),
                [AssetEvent::Added {
                    id: replacement_id.typed::<Value>().unwrap()
                }]
            );
            assert!(stores
                .arena_mut::<Other>()
                .unwrap()
                .drain_events()
                .is_empty());
            assert_eq!(queued_drops.load(Ordering::SeqCst), 0);
            assert_eq!(value_drops.load(Ordering::SeqCst), 0);
            assert_eq!(other_drops.load(Ordering::SeqCst), 0);
            assert_eq!(owned_drops.load(Ordering::SeqCst), 0);
            assert_eq!(replacement_drops.load(Ordering::SeqCst), 0);

            drop(stores);
            assert_eq!(queued_drops.load(Ordering::SeqCst), 0);
            assert_eq!(value_drops.load(Ordering::SeqCst), 0);
            assert_eq!(other_drops.load(Ordering::SeqCst), 0);
            assert_eq!(owned_drops.load(Ordering::SeqCst), 1);
            assert_eq!(replacement_drops.load(Ordering::SeqCst), 1);
            drop(retired);
            assert_eq!(queued_drops.load(Ordering::SeqCst), 1);
            assert_eq!(value_drops.load(Ordering::SeqCst), 1);
            assert_eq!(other_drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn same_frame_deferred_collection_observes_abandonment_without_advancing_grace() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut stores = AssetStores::new();
        let handle = stores.insert(Value::new(1, &drops));
        let id = handle.id();
        let typed = id.typed::<Value>().unwrap();
        stores.arena_mut::<Value>().unwrap().set_grace_frames(2);
        stores.arena_mut::<Value>().unwrap().drain_events();
        drop(handle);
        let mut retired = Vec::new();

        assert_eq!(
            stores.collect_releases_without_advancing_deferred(&mut retired),
            0
        );
        assert_eq!(stores.arena::<Value>().unwrap().frame(), 0);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Unused { id: typed }]
        );
        for _ in 0..3 {
            assert_eq!(
                stores.collect_releases_without_advancing_deferred(&mut retired),
                0
            );
            assert_eq!(stores.arena::<Value>().unwrap().frame(), 0);
            assert!(stores.contains(id));
            assert_eq!(stores.get(typed).unwrap().serial, 1);
            assert!(stores
                .arena_mut::<Value>()
                .unwrap()
                .drain_events()
                .is_empty());
            assert!(retired.is_empty());
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        }

        assert_eq!(stores.collect_releases_deferred(&mut retired), 0);
        assert_eq!(stores.arena::<Value>().unwrap().frame(), 1);
        for _ in 0..3 {
            assert_eq!(
                stores.collect_releases_without_advancing_deferred(&mut retired),
                0
            );
            assert_eq!(stores.arena::<Value>().unwrap().frame(), 1);
            assert!(stores.contains(id));
            assert!(stores
                .arena_mut::<Value>()
                .unwrap()
                .drain_events()
                .is_empty());
            assert!(retired.is_empty());
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        }

        assert_eq!(stores.collect_releases_deferred(&mut retired), 1);
        assert_eq!(stores.arena::<Value>().unwrap().frame(), 2);
        assert!(!stores.contains(id));
        assert_eq!(stores.load_state(id), LoadState::NotLoaded);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Removed { id: typed }]
        );
        assert_eq!(retired.len(), 1);
        assert_eq!(retired[0].downcast_ref::<Value>().unwrap().serial, 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(
            stores.collect_releases_without_advancing_deferred(&mut retired),
            0
        );
        assert_eq!(stores.arena::<Value>().unwrap().frame(), 2);
        assert!(stores
            .arena_mut::<Value>()
            .unwrap()
            .drain_events()
            .is_empty());
        assert_eq!(retired.len(), 1);
        drop(stores);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(retired);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn same_frame_deferred_collection_does_not_age_keepalive_reservations() {
        use crate::{RetainBudget, RetentionPolicy};

        let drops = Arc::new(AtomicUsize::new(0));
        let mut stores = AssetStores::new();
        stores.register::<Value>();
        let arena = stores.arena_mut::<Value>().unwrap();
        arena
            .set_retain_budget(RetainBudget {
                max_bytes: 8,
                max_entries: 1,
            })
            .unwrap();
        let handle = arena.insert(Value::new(1, &drops));
        let typed = handle.id();
        let id = typed.untyped();
        arena
            .retain(&handle, 8, RetentionPolicy::KeepAlive { frames: 1 })
            .unwrap();
        let weak = handle.downgrade();
        arena.drain_events();
        drop(handle);
        let mut retired = Vec::new();

        assert_eq!(stores.collect_releases_deferred(&mut retired), 0);
        assert_eq!(stores.arena::<Value>().unwrap().frame(), 1);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Unused { id: typed }]
        );
        for _ in 0..3 {
            assert_eq!(
                stores.collect_releases_without_advancing_deferred(&mut retired),
                0
            );
            let arena = stores.arena::<Value>().unwrap();
            assert_eq!(arena.frame(), 1);
            assert_eq!(
                arena.retain_cache().get(typed).unwrap().idle_frames,
                Some(0)
            );
            assert_eq!(arena.pending_event_count(), 0);
            assert!(stores.contains(id));
            assert!(weak.upgrade().is_some());
            assert!(retired.is_empty());
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        }

        assert_eq!(stores.collect_releases_deferred(&mut retired), 1);
        assert_eq!(stores.arena::<Value>().unwrap().frame(), 2);
        assert!(stores
            .arena::<Value>()
            .unwrap()
            .retain_cache()
            .get(typed)
            .is_none());
        assert!(!stores.contains(id));
        assert!(weak.upgrade().is_none());
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Removed { id: typed }]
        );
        assert_eq!(retired.len(), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(stores);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(retired);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn deferred_collection_keeps_pending_slots_and_counts_failed_slots_without_payloads() {
        for advancing in [true, false] {
            let drops = Arc::new(AtomicUsize::new(0));
            let mut stores = AssetStores::new();
            let pending = stores.reserve::<Value>();
            let failed = stores.reserve::<Value>();
            let pending_id = pending.id();
            let pending_typed = pending_id.typed::<Value>().unwrap();
            let failed_id = failed.id();
            let failed_typed = failed_id.typed::<Value>().unwrap();
            let mut errors = ErrorRegistry::new();
            let error = errors.record(AssetError::new("test.asset", "failed"));
            assert!(stores.fail_erased(failed_id, error));
            stores.arena_mut::<Value>().unwrap().drain_events();
            drop(pending);
            drop(failed);
            let mut retired = Vec::new();

            let reclaimed = if advancing {
                stores.collect_releases_deferred(&mut retired)
            } else {
                stores.collect_releases_without_advancing_deferred(&mut retired)
            };
            assert_eq!(reclaimed, 1);
            assert!(retired.is_empty());
            assert_eq!(stores.total_assets(), 1);
            assert!(stores.contains(pending_id));
            assert_eq!(stores.load_state(pending_id), LoadState::Loading);
            assert!(!stores.contains(failed_id));
            assert_eq!(stores.load_state(failed_id), LoadState::NotLoaded);
            assert_eq!(
                stores.arena_mut::<Value>().unwrap().drain_events(),
                [
                    AssetEvent::Unused { id: pending_typed },
                    AssetEvent::Unused { id: failed_typed },
                    AssetEvent::Removed { id: failed_typed },
                ]
            );
            let frame = stores.arena::<Value>().unwrap().frame();
            assert_eq!(
                stores.collect_releases_without_advancing_deferred(&mut retired),
                0
            );
            assert_eq!(stores.arena::<Value>().unwrap().frame(), frame);
            assert!(stores
                .arena_mut::<Value>()
                .unwrap()
                .drain_events()
                .is_empty());

            assert_eq!(
                stores.fulfill_erased_deferred(
                    pending_id,
                    Box::new(Value::new(1, &drops)),
                    &mut retired,
                ),
                Ok(true)
            );
            assert!(retired.is_empty());
            assert_eq!(
                stores.collect_releases_without_advancing_deferred(&mut retired),
                1
            );
            assert_eq!(stores.arena::<Value>().unwrap().frame(), frame);
            assert_eq!(stores.total_assets(), 0);
            assert!(!stores.contains(pending_id));
            assert_eq!(stores.load_state(pending_id), LoadState::NotLoaded);
            assert_eq!(
                stores.arena_mut::<Value>().unwrap().drain_events(),
                [
                    AssetEvent::Added { id: pending_typed },
                    AssetEvent::Removed { id: pending_typed },
                ]
            );
            assert_eq!(retired.len(), 1);
            assert_eq!(retired[0].downcast_ref::<Value>().unwrap().serial, 1);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(stores);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(retired);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn erased_eager_removal_invalidates_generation_and_defers_drop_without_advancing() {
        let old_drops = Arc::new(AtomicUsize::new(0));
        let replacement_drops = Arc::new(AtomicUsize::new(0));
        let mut arena = Assets::<Value>::with_grace_frames(10);
        let handle = arena.insert(Value::new(1, &old_drops));
        let old = handle.id();
        arena.drain_events();
        drop(handle);
        let mut retired = Vec::new();

        let reclaimed = {
            let erased: &mut dyn ErasedAssetStore = &mut arena;
            erased.remove_unused_deferred(&mut retired)
        };
        assert_eq!(reclaimed, 1);
        assert_eq!(arena.frame(), 0);
        assert!(arena.is_empty());
        assert!(!arena.contains(old));
        assert!(arena.get(old).is_none());
        assert_eq!(arena.load_state(old), LoadState::NotLoaded);
        assert_eq!(arena.drain_events(), [AssetEvent::Removed { id: old }]);
        assert_eq!(retired.len(), 1);
        let queued = retired[0].downcast_ref::<Value>().unwrap();
        assert_eq!(queued.serial, 1);
        assert!(Arc::ptr_eq(&queued.drops, &old_drops));
        assert_eq!(old_drops.load(Ordering::SeqCst), 0);

        let replacement = arena.insert(Value::new(2, &replacement_drops));
        let new = replacement.id();
        assert_eq!(old.index().index(), new.index().index());
        assert_ne!(old.index().generation(), new.index().generation());
        {
            let erased: &mut dyn ErasedAssetStore = &mut arena;
            assert_eq!(
                erased.validate_fulfill_erased(old.untyped(), queued),
                Ok(false)
            );
            assert!(!erased.contains_erased(old.untyped()));
            assert_eq!(erased.remove_unused_deferred(&mut retired), 0);
        }
        assert_eq!(arena.frame(), 0);
        assert_eq!(retired.len(), 1);
        assert_eq!(arena.get(new).unwrap().serial, 2);
        assert_eq!(arena.drain_events(), [AssetEvent::Added { id: new }]);
        assert_eq!(old_drops.load(Ordering::SeqCst), 0);
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 0);

        drop(arena);
        assert_eq!(old_drops.load(Ordering::SeqCst), 0);
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 1);
        drop(retired);
        assert_eq!(old_drops.load(Ordering::SeqCst), 1);
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn registry_eager_removal_counts_all_abandoned_slots_but_only_queues_ready_payloads() {
        let queued_drops = Arc::new(AtomicUsize::new(0));
        let value_drops = Arc::new(AtomicUsize::new(0));
        let other_drops = Arc::new(AtomicUsize::new(0));
        let owned_drops = Arc::new(AtomicUsize::new(0));
        let mut stores = AssetStores::new();
        let ready = stores.insert(Value::new(1, &value_drops));
        let pending = stores.reserve::<Value>();
        let failed = stores.reserve::<Value>();
        let other = stores.insert(Other(Value::new(2, &other_drops)));
        let owned = stores.insert(Value::new(3, &owned_drops));
        let ready_id = ready.id();
        let pending_id = pending.id();
        let failed_id = failed.id();
        let other_id = other.id();
        let owned_typed = owned.id().typed::<Value>().unwrap();
        let mut errors = ErrorRegistry::new();
        let error = errors.record(AssetError::new("test.asset", "failed"));
        assert!(stores.fail_erased(failed_id, error));
        stores.arena_mut::<Value>().unwrap().set_grace_frames(10);
        stores.arena_mut::<Other>().unwrap().set_grace_frames(10);
        stores.arena_mut::<Value>().unwrap().drain_events();
        stores.arena_mut::<Other>().unwrap().drain_events();
        drop(ready);
        drop(pending);
        drop(failed);
        drop(other);
        let mut retired: Vec<Box<dyn Any + Send>> = Vec::new();
        retired.push(Box::new(Value::new(0, &queued_drops)));

        assert_eq!(stores.remove_unused_deferred(&mut retired), 4);
        assert_eq!(stores.total_assets(), 1);
        for id in [ready_id, pending_id, failed_id, other_id] {
            assert!(!stores.contains(id));
            assert_eq!(stores.load_state(id), LoadState::NotLoaded);
        }
        assert_eq!(stores.get(owned_typed).unwrap().serial, 3);
        assert_eq!(stores.load_state(owned.id()), LoadState::Loaded);
        assert_eq!(stores.arena::<Value>().unwrap().frame(), 0);
        assert_eq!(stores.arena::<Other>().unwrap().frame(), 0);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [
                AssetEvent::Removed {
                    id: ready_id.typed::<Value>().unwrap()
                },
                AssetEvent::Removed {
                    id: pending_id.typed::<Value>().unwrap()
                },
                AssetEvent::Removed {
                    id: failed_id.typed::<Value>().unwrap()
                },
            ]
        );
        assert_eq!(
            stores.arena_mut::<Other>().unwrap().drain_events(),
            [AssetEvent::Removed {
                id: other_id.typed::<Other>().unwrap()
            }]
        );
        assert_eq!(retired.len(), 3);
        assert_eq!(retired[0].downcast_ref::<Value>().unwrap().serial, 0);
        // Cross-arena queue ordering is deliberately not asserted.
        let queued_value = retired
            .iter()
            .filter_map(|value| value.downcast_ref::<Value>())
            .find(|value| value.serial == 1)
            .unwrap();
        assert!(Arc::ptr_eq(&queued_value.drops, &value_drops));
        let queued_other = retired
            .iter()
            .find_map(|value| value.downcast_ref::<Other>())
            .unwrap();
        assert_eq!(queued_other.0.serial, 2);
        assert!(Arc::ptr_eq(&queued_other.0.drops, &other_drops));

        assert_eq!(stores.remove_unused_deferred(&mut retired), 0);
        assert_eq!(retired.len(), 3);
        assert_eq!(stores.arena::<Value>().unwrap().frame(), 0);
        assert_eq!(stores.arena::<Other>().unwrap().frame(), 0);
        assert!(stores
            .arena_mut::<Value>()
            .unwrap()
            .drain_events()
            .is_empty());
        assert!(stores
            .arena_mut::<Other>()
            .unwrap()
            .drain_events()
            .is_empty());
        assert_eq!(queued_drops.load(Ordering::SeqCst), 0);
        assert_eq!(value_drops.load(Ordering::SeqCst), 0);
        assert_eq!(other_drops.load(Ordering::SeqCst), 0);
        assert_eq!(owned_drops.load(Ordering::SeqCst), 0);

        drop(stores);
        assert_eq!(queued_drops.load(Ordering::SeqCst), 0);
        assert_eq!(value_drops.load(Ordering::SeqCst), 0);
        assert_eq!(other_drops.load(Ordering::SeqCst), 0);
        assert_eq!(owned_drops.load(Ordering::SeqCst), 1);
        drop(retired);
        assert_eq!(queued_drops.load(Ordering::SeqCst), 1);
        assert_eq!(value_drops.load(Ordering::SeqCst), 1);
        assert_eq!(other_drops.load(Ordering::SeqCst), 1);
        assert_eq!(owned_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn ordinary_eager_removal_still_drops_payloads_immediately_and_ignores_grace() {
        let arena_drops = Arc::new(AtomicUsize::new(0));
        let mut arena = Assets::<Value>::with_grace_frames(10);
        let handle = arena.insert(Value::new(1, &arena_drops));
        let typed = handle.id();
        arena.drain_events();
        drop(handle);
        let reclaimed = {
            let erased: &mut dyn ErasedAssetStore = &mut arena;
            erased.remove_unused()
        };
        assert_eq!(reclaimed, 1);
        assert_eq!(arena_drops.load(Ordering::SeqCst), 1);
        assert_eq!(arena.frame(), 0);
        assert!(arena.is_empty());
        assert_eq!(arena.drain_events(), [AssetEvent::Removed { id: typed }]);

        let store_drops = Arc::new(AtomicUsize::new(0));
        let mut stores = AssetStores::new();
        let handle = stores.insert(Value::new(2, &store_drops));
        let id = handle.id();
        stores.arena_mut::<Value>().unwrap().set_grace_frames(10);
        stores.arena_mut::<Value>().unwrap().drain_events();
        drop(handle);
        assert_eq!(stores.remove_unused(), 1);
        assert_eq!(store_drops.load(Ordering::SeqCst), 1);
        assert_eq!(stores.total_assets(), 0);
        assert!(!stores.contains(id));
        assert_eq!(stores.arena::<Value>().unwrap().frame(), 0);
        assert_eq!(
            stores.arena_mut::<Value>().unwrap().drain_events(),
            [AssetEvent::Removed {
                id: id.typed::<Value>().unwrap()
            }]
        );
        drop(stores);
        drop(arena);
        assert_eq!(arena_drops.load(Ordering::SeqCst), 1);
        assert_eq!(store_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn ordinary_collection_still_drops_payloads_immediately() {
        for advancing in [true, false] {
            let arena_drops = Arc::new(AtomicUsize::new(0));
            let mut arena = Assets::<Value>::new();
            let handle = arena.insert(Value::new(1, &arena_drops));
            let typed = handle.id();
            arena.drain_events();
            drop(handle);
            let reclaimed = {
                let erased: &mut dyn ErasedAssetStore = &mut arena;
                if advancing {
                    erased.collect_releases()
                } else {
                    erased.collect_releases_without_advancing()
                }
            };
            assert_eq!(reclaimed, 1);
            assert_eq!(arena_drops.load(Ordering::SeqCst), 1);
            assert!(arena.is_empty());
            assert_eq!(arena.frame(), if advancing { 1 } else { 0 });
            assert_eq!(
                arena.drain_events(),
                [
                    AssetEvent::Unused { id: typed },
                    AssetEvent::Removed { id: typed }
                ]
            );

            let store_drops = Arc::new(AtomicUsize::new(0));
            let mut stores = AssetStores::new();
            let handle = stores.insert(Value::new(2, &store_drops));
            let id = handle.id();
            let typed = id.typed::<Value>().unwrap();
            stores.arena_mut::<Value>().unwrap().drain_events();
            drop(handle);
            let reclaimed = if advancing {
                stores.collect_releases()
            } else {
                stores.collect_releases_without_advancing()
            };
            assert_eq!(reclaimed, 1);
            assert_eq!(store_drops.load(Ordering::SeqCst), 1);
            assert_eq!(stores.total_assets(), 0);
            assert!(!stores.contains(id));
            assert_eq!(
                stores.arena_mut::<Value>().unwrap().drain_events(),
                [
                    AssetEvent::Unused { id: typed },
                    AssetEvent::Removed { id: typed }
                ]
            );
            drop(stores);
            drop(arena);
            assert_eq!(arena_drops.load(Ordering::SeqCst), 1);
            assert_eq!(store_drops.load(Ordering::SeqCst), 1);
        }
    }
}
