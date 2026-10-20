//! Storage lifecycle, grace, generational retirement and reservation invalidation.

use super::Assets;
use crate::asset::Asset;
use crate::error::{AssetError, ErrorRegistry};
use crate::retain_cache::{RetainBudget, RetainError, RetainRemovalReason, RetentionPolicy};

#[derive(Debug, PartialEq, Eq)]
struct Sample(u32);
impl Asset for Sample {
    const TYPE_NAME: &'static str = "test::storage_retention::Sample";
}

fn arena() -> Assets<Sample> {
    let mut assets = Assets::new();
    assets
        .set_retain_budget(RetainBudget {
            max_bytes: 64,
            max_entries: 8,
        })
        .unwrap();
    assets
}

/// A two-frame window expires exactly two reclaim points after first observed idle.
#[test]
fn keepalive_boundary_and_unused_event_are_exact() {
    let mut assets = arena();
    let handle = assets.insert(Sample(1));
    assets
        .retain(&handle, 4, RetentionPolicy::KeepAlive { frames: 2 })
        .unwrap();
    let id = handle.id();
    let weak = handle.downgrade();
    assets.drain_events();
    drop(handle);
    assert_eq!(assets.collect_releases(), 0);
    let events = assets.drain_events();
    assert_eq!(events.len(), 1);
    assert!(events[0].is_unused());
    assert!(weak.upgrade().is_some());
    assert_eq!(assets.collect_releases(), 0);
    assert!(assets.drain_events().is_empty());
    let result = assets.collect_retained();
    assert_eq!(result.reclaimed, 1);
    assert_eq!(result.removals.len(), 1);
    assert_eq!(result.removals[0].reason, RetainRemovalReason::Expired);
    assert!(weak.upgrade().is_none());
    assert!(!assets.contains(id));
    let events = assets.drain_events();
    assert_eq!(events.len(), 1);
    assert!(events[0].is_removed());
}

/// Zero frames releases immediately; the original reclamation API must advance policies too.
#[test]
fn zero_frame_keepalive_and_immediate_policy() {
    let mut assets = arena();
    let handle = assets.insert(Sample(1));
    assets
        .retain(&handle, 4, RetentionPolicy::KeepAlive { frames: 0 })
        .unwrap();
    drop(handle);
    assert_eq!(assets.collect_releases(), 1);
    let handle = assets.insert(Sample(2));
    assets.pin(&handle, 4).unwrap();
    assets
        .retain(&handle, 0, RetentionPolicy::Immediate)
        .unwrap();
    assert_eq!(assets.retain_cache().usage().entries, 0);
    drop(handle);
    assert_eq!(assets.collect_releases(), 1);
}

/// Reacquisition restarts KeepAlive and emits a new Unused only on a later idle episode.
#[test]
fn external_ownership_restarts_keepalive() {
    let mut assets = arena();
    let handle = assets.insert(Sample(1));
    assets
        .retain(&handle, 4, RetentionPolicy::KeepAlive { frames: 1 })
        .unwrap();
    let id = handle.id();
    drop(handle);
    assets.collect_releases();
    let reacquired = assets.acquire_retained(id).unwrap();
    for _ in 0..3 {
        assert_eq!(assets.collect_releases(), 0);
    }
    assets.drain_events();
    drop(reacquired);
    assert_eq!(assets.collect_releases(), 0);
    assert_eq!(
        assets
            .drain_events()
            .iter()
            .filter(|e| e.is_unused())
            .count(),
        1
    );
    assert_eq!(assets.collect_releases(), 1);
}

/// Weak upgrades are external ownership too, even without an explicit cache accessor.
#[test]
fn weak_reacquisition_resets_observed_idle_episode() {
    let mut assets = arena();
    let handle = assets.insert(Sample(1));
    assets
        .retain(&handle, 4, RetentionPolicy::KeepAlive { frames: 1 })
        .unwrap();
    let weak = handle.downgrade();
    drop(handle);
    assets.collect_releases();
    let revived = weak.upgrade().unwrap();
    assert_eq!(assets.collect_releases(), 0);
    assets.drain_events();
    drop(revived);
    assert_eq!(assets.collect_releases(), 0);
    assert_eq!(
        assets
            .drain_events()
            .iter()
            .filter(|e| e.is_unused())
            .count(),
        1
    );
    assert_eq!(assets.collect_releases(), 1);
}

/// Cache expiry does not bypass the arena's downstream grace requirement.
#[test]
fn keepalive_and_grace_share_first_idle_observation() {
    let mut assets = arena();
    assets.set_grace_frames(3);
    let handle = assets.insert(Sample(1));
    assets
        .retain(&handle, 4, RetentionPolicy::KeepAlive { frames: 1 })
        .unwrap();
    drop(handle);
    assert_eq!(assets.collect_releases(), 0);
    let expired = assets.collect_retained();
    assert_eq!(expired.removals.len(), 1);
    assert_eq!(expired.reclaimed, 0);
    assert_eq!(assets.collect_releases(), 0);
    assert_eq!(assets.collect_releases(), 1);
    let events = assets.drain_events();
    assert_eq!(events.iter().filter(|e| e.is_unused()).count(), 1);
    assert_eq!(events.iter().filter(|e| e.is_removed()).count(), 1);
}

/// Explicit arena removal clears even a pin and a stale handle cannot retain the reused slot.
#[test]
fn stale_handle_cannot_retain_or_unpin_new_generation() {
    let mut assets = arena();
    let old = assets.insert(Sample(1));
    assets.pin(&old, 4).unwrap();
    assert_eq!(assets.remove(old.id()), Some(Sample(1)));
    assert_eq!(assets.retain_cache().usage().bytes, 0);
    let new = assets.insert(Sample(2));
    assets.pin(&new, 4).unwrap();
    assert_eq!(old.id().index().index(), new.id().index().index());
    assert_eq!(assets.pin(&old, 1), Err(RetainError::InvalidHandle));
    assert_eq!(assets.unpin(&old), Err(RetainError::InvalidHandle));
    assert_eq!(assets.touch_retained(&old), Err(RetainError::InvalidHandle));
    assert!(assets.acquire_retained(old.id()).is_none());
    assert_eq!(assets.retain_cache().usage().bytes, 4);
}

/// Equal typed ids from independent arenas must not pass an allocation identity check.
#[test]
fn foreign_arena_handle_is_rejected_despite_equal_id() {
    let mut first = arena();
    let mut second = arena();
    let a = first.insert(Sample(1));
    let b = second.insert(Sample(2));
    assert_eq!(a.id(), b.id());
    assert_ne!(a.handle_id(), b.handle_id());
    first.pin(&a, 4).unwrap();
    assert!(!first.owns_handle(&b));
    assert_eq!(first.pin(&b, 4), Err(RetainError::InvalidHandle));
    assert_eq!(first.unpin(&b), Err(RetainError::InvalidHandle));
    assert_eq!(first.retain_cache().usage().bytes, 4);
}

/// Unaccounted mutable access, replacement and failure all revoke the old reservation.
#[test]
fn payload_changes_revoke_stale_byte_accounting() {
    let mut assets = arena();
    let handle = assets.insert(Sample(1));
    assets.pin(&handle, 4).unwrap();
    assets.get_mut(handle.id()).unwrap().0 = 2;
    assert_eq!(assets.retain_cache().usage().bytes, 0);
    assets.pin(&handle, 8).unwrap();
    assets.fulfill(handle.id(), Sample(3));
    assert_eq!(assets.retain_cache().usage().bytes, 0);
    assets.pin(&handle, 8).unwrap();
    let error = ErrorRegistry::new().record(AssetError::new("sample", "failed"));
    assets.fail(handle.id(), error);
    assert_eq!(assets.retain_cache().usage().bytes, 0);
    assert_eq!(assets.pin(&handle, 0), Err(RetainError::NotReady));
    let pending = assets.reserve();
    assert_eq!(assets.pin(&pending, 0), Err(RetainError::NotReady));
}

/// Pinned values survive idle collection and eager unused sweeps without hidden overcommit.
#[test]
fn pinned_survives_until_explicit_budget_checked_release() {
    let mut assets = arena();
    let handle = assets.insert(Sample(1));
    let id = handle.id();
    assets.pin(&handle, 4).unwrap();
    drop(handle);
    for _ in 0..4 {
        assert_eq!(assets.collect_releases(), 0);
    }
    assert_eq!(assets.remove_unused(), 0);
    let handle = assets.acquire_retained(id).unwrap();
    assets.unpin(&handle).unwrap();
    drop(handle);
    assert_eq!(assets.collect_releases(), 1);
}

/// A wrapping global frame cannot prematurely expire a long keepalive counter.
#[test]
fn keepalive_age_does_not_depend_on_wrapping_global_frame() {
    let mut assets = arena();
    assets.frame = u64::MAX;
    let handle = assets.insert(Sample(1));
    assets
        .retain(&handle, 4, RetentionPolicy::KeepAlive { frames: u32::MAX })
        .unwrap();
    let id = handle.id();
    drop(handle);
    for _ in 0..3 {
        assert_eq!(assets.collect_releases(), 0);
    }
    assert_eq!(assets.retain_cache().get(id).unwrap().idle_frames, Some(2));
}

/// Exhausted generations retire permanently instead of aliasing an ancient generation zero id.
#[test]
fn exhausted_generation_is_never_recycled() {
    let mut assets = arena();
    let old = assets.insert(Sample(1));
    let position = old.id().index().index();
    assets.remove(old.id());
    assets.slots[position as usize].generation = u32::MAX;
    let last = assets.insert(Sample(2));
    assert_eq!(last.id().index().generation(), u32::MAX);
    assets.pin(&last, 4).unwrap();
    assets.remove(last.id());
    assert_eq!(assets.retain_cache().usage().entries, 0);
    let fresh = assets.insert(Sample(3));
    assert_ne!(fresh.id().index().index(), position);
    assert!(assets.get(old.id()).is_none());
    assert!(assets.get(last.id()).is_none());
}

/// Collector tombstones cannot be bypassed through a cache reacquisition path.
#[cfg(feature = "std")]
#[test]
fn retired_handle_cannot_be_reacquired_or_readmitted() {
    let mut assets = arena();
    let handle = assets.insert(Sample(1));
    let id = handle.id();
    assets.pin(&handle, 4).unwrap();
    let inner = assets.slots[id.index().index() as usize]
        .entry
        .as_ref()
        .unwrap()
        .handle
        .upgrade()
        .unwrap();
    inner.set_retired_for_collection(true);
    assert_eq!(assets.pin(&handle, 4), Err(RetainError::InvalidHandle));
    assert!(assets.acquire_retained(id).is_none());
    drop((inner, handle));
    let result = assets.collect_retained();
    assert_eq!(result.removals[0].reason, RetainRemovalReason::Retired);
    assert_eq!(result.reclaimed, 1);
}
