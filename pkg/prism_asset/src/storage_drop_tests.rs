#[cfg(all(test, panic = "unwind"))]
mod typed_collection_drop_tests {
    extern crate std;

    use crate::{Asset, AssetEvent, Assets, LoadState, RetainBudget, RetentionPolicy};
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    struct Probe {
        serial: usize,
        drops: Arc<AtomicUsize>,
        panic_on_drop: Arc<AtomicBool>,
    }

    impl Probe {
        fn new(serial: usize, drops: &Arc<AtomicUsize>) -> Self {
            Self {
                serial,
                drops: Arc::clone(drops),
                panic_on_drop: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    impl Asset for Probe {
        const TYPE_NAME: &'static str = "storage_drop_tests::Probe";
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            if self.panic_on_drop.swap(false, Ordering::SeqCst) {
                panic!("intentional typed asset destructor panic");
            }
        }
    }

    #[test]
    fn remove_unused_commits_slot_and_events_before_drop_panic_and_can_resume() {
        let panic_drops = Arc::new(AtomicUsize::new(0));
        let sibling_drops = Arc::new(AtomicUsize::new(0));
        let replacement_drops = Arc::new(AtomicUsize::new(0));
        let mut assets = Assets::<Probe>::with_grace_frames(10);
        let value = Probe::new(1, &panic_drops);
        let panic_on_drop = Arc::clone(&value.panic_on_drop);
        let panicking = assets.insert(value);
        let sibling = assets.insert(Probe::new(2, &sibling_drops));
        let old = panicking.id();
        let sibling_id = sibling.id();
        assets.drain_events();
        drop(panicking);
        drop(sibling);

        let result = catch_unwind(AssertUnwindSafe(|| {
            panic_on_drop.store(true, Ordering::SeqCst);
            assets.remove_unused()
        }));
        assert!(result.is_err());
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
        assert_eq!(sibling_drops.load(Ordering::SeqCst), 0);
        assert_eq!(assets.frame(), 0);
        assert_eq!(assets.len(), 1);
        assert!(!assets.contains(old));
        assert!(assets.get(old).is_none());
        assert_eq!(assets.load_state(old), LoadState::NotLoaded);
        assert_eq!(assets.get(sibling_id).unwrap().serial, 2);
        assert_eq!(assets.drain_events(), [AssetEvent::Removed { id: old }]);

        // A free slot and its next generation must already be usable at unwind.
        let replacement = assets.insert(Probe::new(3, &replacement_drops));
        let new = replacement.id();
        assert_eq!(new.index().index(), old.index().index());
        assert_eq!(new.index().generation(), old.index().generation() + 1);
        assert!(!assets.contains(old));
        assert_eq!(assets.get(new).unwrap().serial, 3);
        assert_eq!(assets.drain_events(), [AssetEvent::Added { id: new }]);

        // No handle is dropped between the interrupted sweep and this retry.
        assert_eq!(assets.remove_unused(), 1);
        assert_eq!(sibling_drops.load(Ordering::SeqCst), 1);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
        assert_eq!(assets.frame(), 0);
        assert_eq!(assets.len(), 1);
        assert!(!assets.contains(sibling_id));
        assert_eq!(
            assets.drain_events(),
            [AssetEvent::Removed { id: sibling_id }]
        );
        assert_eq!(assets.remove_unused(), 0);
        assert!(assets.drain_events().is_empty());

        drop(replacement);
        assert_eq!(assets.remove_unused(), 1);
        assert_eq!(assets.drain_events(), [AssetEvent::Removed { id: new }]);
        assert!(assets.is_empty());
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 1);
        drop(assets);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
        assert_eq!(sibling_drops.load(Ordering::SeqCst), 1);
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 1);
    }

    // Select only public typed entry points; no retire callback is invoked here.
    fn collect(assets: &mut Assets<Probe>, retained: bool) -> usize {
        if retained {
            assets.collect_retained().reclaimed
        } else {
            assets.collect_releases()
        }
    }

    fn assert_collection_resumes_after_drop_panic(retained: bool) {
        let panic_drops = Arc::new(AtomicUsize::new(0));
        let sibling_drops = Arc::new(AtomicUsize::new(0));
        let replacement_drops = Arc::new(AtomicUsize::new(0));
        let mut assets = Assets::<Probe>::new();
        let value = Probe::new(1, &panic_drops);
        let panic_on_drop = Arc::clone(&value.panic_on_drop);
        let panicking = assets.insert(value);
        let sibling = assets.insert(Probe::new(2, &sibling_drops));
        let old = panicking.id();
        let sibling_id = sibling.id();
        assets.drain_events();
        drop(panicking);
        drop(sibling);

        let result = catch_unwind(AssertUnwindSafe(|| {
            panic_on_drop.store(true, Ordering::SeqCst);
            collect(&mut assets, retained)
        }));
        assert!(result.is_err());
        assert_eq!(assets.frame(), 1);
        assert_eq!(assets.len(), 1);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
        assert_eq!(sibling_drops.load(Ordering::SeqCst), 0);
        assert!(!assets.contains(old));
        assert!(assets.get(old).is_none());
        assert_eq!(assets.load_state(old), LoadState::NotLoaded);
        assert_eq!(assets.get(sibling_id).unwrap().serial, 2);
        assert_eq!(
            assets.drain_events(),
            [
                AssetEvent::Unused { id: old },
                AssetEvent::Removed { id: old }
            ]
        );

        let replacement = assets.insert(Probe::new(3, &replacement_drops));
        let new = replacement.id();
        assert_eq!(new.index().index(), old.index().index());
        assert_eq!(new.index().generation(), old.index().generation() + 1);
        assert!(!assets.contains(old));
        assert_eq!(assets.drain_events(), [AssetEvent::Added { id: new }]);

        // Retrying must revisit the unprocessed sibling even without a new drop
        // signal; keeping the replacement owned must not suppress that scan.
        assert_eq!(collect(&mut assets, retained), 1);
        assert_eq!(assets.frame(), 2);
        assert_eq!(assets.len(), 1);
        assert!(!assets.contains(sibling_id));
        assert_eq!(assets.load_state(sibling_id), LoadState::NotLoaded);
        assert_eq!(assets.get(new).unwrap().serial, 3);
        assert_eq!(
            assets.drain_events(),
            [
                AssetEvent::Unused { id: sibling_id },
                AssetEvent::Removed { id: sibling_id },
            ]
        );
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
        assert_eq!(sibling_drops.load(Ordering::SeqCst), 1);
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 0);
        assert_eq!(collect(&mut assets, retained), 0);
        assert_eq!(assets.frame(), 3);
        assert!(assets.drain_events().is_empty());

        drop(replacement);
        assert_eq!(collect(&mut assets, retained), 1);
        assert_eq!(
            assets.drain_events(),
            [
                AssetEvent::Unused { id: new },
                AssetEvent::Removed { id: new }
            ]
        );
        assert!(assets.is_empty());
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 1);
        drop(assets);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
        assert_eq!(sibling_drops.load(Ordering::SeqCst), 1);
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn collect_releases_commits_slot_and_events_before_drop_panic_and_can_resume() {
        assert_collection_resumes_after_drop_panic(false);
    }

    #[test]
    fn collect_retained_commits_slot_and_events_before_drop_panic_and_can_resume() {
        assert_collection_resumes_after_drop_panic(true);
    }

    fn assert_mid_grace_slot_survives_another_payload_drop_panic(retained: bool) {
        let grace_drops = Arc::new(AtomicUsize::new(0));
        let panic_drops = Arc::new(AtomicUsize::new(0));
        let replacement_drops = Arc::new(AtomicUsize::new(0));
        let mut assets = Assets::<Probe>::with_grace_frames(2);
        assets
            .set_retain_budget(RetainBudget {
                max_bytes: 8,
                max_entries: 1,
            })
            .unwrap();
        // The younger slot precedes the panicking slot in the arena scan, so
        // its grace bookkeeping is observed before the destructor unwinds.
        let young = assets.insert(Probe::new(1, &grace_drops));
        let young_id = young.id();
        let value = Probe::new(2, &panic_drops);
        let panic_on_drop = Arc::clone(&value.panic_on_drop);
        let panicking = assets.insert(value);
        let old = panicking.id();
        assets
            .retain(&panicking, 8, RetentionPolicy::KeepAlive { frames: 2 })
            .unwrap();
        assets.drain_events();
        drop(panicking);

        // Grace starts while the cache still holds the older value. Nothing is
        // yet abandoned in the slot scan, so it has no pending grace work.
        assert_eq!(collect(&mut assets, retained), 0);
        assert_eq!(assets.frame(), 1);
        assert_eq!(assets.drain_events(), [AssetEvent::Unused { id: old }]);
        assert_eq!(collect(&mut assets, retained), 0);
        assert_eq!(assets.frame(), 2);
        assert!(assets.drain_events().is_empty());
        assert_eq!(assets.retain_cache().get(old).unwrap().idle_frames, Some(1));
        drop(young);

        // At frame 3 the old reservation expires and is already grace-mature;
        // the younger slot has just begun its own, independent two-frame grace.
        let result = catch_unwind(AssertUnwindSafe(|| {
            panic_on_drop.store(true, Ordering::SeqCst);
            collect(&mut assets, retained)
        }));
        assert!(result.is_err());
        assert_eq!(assets.frame(), 3);
        assert_eq!(assets.len(), 1);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
        assert_eq!(grace_drops.load(Ordering::SeqCst), 0);
        assert!(!assets.contains(old));
        assert_eq!(assets.load_state(old), LoadState::NotLoaded);
        assert!(assets.retain_cache().get(old).is_none());
        assert_eq!(assets.retain_cache().usage().entries, 0);
        assert_eq!(assets.retain_cache().usage().bytes, 0);
        assert_eq!(assets.get(young_id).unwrap().serial, 1);
        assert_eq!(
            assets.drain_events(),
            [
                AssetEvent::Unused { id: young_id },
                AssetEvent::Removed { id: old }
            ]
        );

        let replacement = assets.insert(Probe::new(3, &replacement_drops));
        let new = replacement.id();
        assert_eq!(new.index().index(), old.index().index());
        assert_eq!(new.index().generation(), old.index().generation() + 1);
        assert!(!assets.contains(old));
        assert_eq!(assets.drain_events(), [AssetEvent::Added { id: new }]);

        // No new handle release may be needed either to retain the remaining
        // grace frame or to reclaim the slot at its original maturity point.
        assert_eq!(collect(&mut assets, retained), 0);
        assert_eq!(assets.frame(), 4);
        assert!(assets.contains(young_id));
        assert_eq!(assets.load_state(young_id), LoadState::Loaded);
        assert_eq!(grace_drops.load(Ordering::SeqCst), 0);
        assert!(assets.drain_events().is_empty());
        assert_eq!(collect(&mut assets, retained), 1);
        assert_eq!(assets.frame(), 5);
        assert!(!assets.contains(young_id));
        assert_eq!(assets.load_state(young_id), LoadState::NotLoaded);
        assert_eq!(
            assets.drain_events(),
            [AssetEvent::Removed { id: young_id }]
        );
        assert_eq!(grace_drops.load(Ordering::SeqCst), 1);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
        assert_eq!(assets.len(), 1);
        assert_eq!(assets.get(new).unwrap().serial, 3);
        assert_eq!(collect(&mut assets, retained), 0);
        assert!(assets.drain_events().is_empty());

        // Reusing this second free slot verifies that the resumed scan also
        // completes generation and free-list bookkeeping exactly once.
        let recycled = assets.reserve();
        assert_eq!(recycled.id().index().index(), young_id.index().index());
        assert_eq!(
            recycled.id().index().generation(),
            young_id.index().generation() + 1
        );
        assert!(!assets.contains(young_id));
        assert_eq!(assets.get(new).unwrap().serial, 3);
        assert!(assets.drain_events().is_empty());
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 0);
        drop(assets);
        assert_eq!(grace_drops.load(Ordering::SeqCst), 1);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
        assert_eq!(replacement_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn collect_releases_resumes_mid_grace_slot_after_expired_keepalive_drop_panics() {
        assert_mid_grace_slot_survives_another_payload_drop_panic(false);
    }

    #[test]
    fn collect_retained_resumes_mid_grace_slot_after_expired_keepalive_drop_panics() {
        assert_mid_grace_slot_survives_another_payload_drop_panic(true);
    }
}
