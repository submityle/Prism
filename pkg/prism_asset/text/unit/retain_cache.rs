//! Admission, deterministic eviction and strict byte/entry reservation boundaries.

use crate::asset::Asset;
use crate::retain_cache::{RetainBudget, RetainError, RetainRemovalReason, RetentionPolicy};
use crate::storage::Assets;
use alloc::vec::Vec;

struct Sample;
impl Asset for Sample {
    const TYPE_NAME: &'static str = "test::retain_cache::Sample";
}

fn arena(bytes: u64, entries: usize) -> Assets<Sample> {
    let mut assets = Assets::new();
    assets
        .set_retain_budget(RetainBudget {
            max_bytes: bytes,
            max_entries: entries,
        })
        .unwrap();
    assets
}

/// Explicit access order, not arena index or unordered iteration, chooses LRU victims.
#[test]
fn lru_order_and_diagnostics_are_deterministic() {
    let mut assets = arena(12, 3);
    let first = assets.insert(Sample);
    let second = assets.insert(Sample);
    let third = assets.insert(Sample);
    for handle in [&first, &second, &third] {
        assets.retain(handle, 4, RetentionPolicy::Lru).unwrap();
    }
    assets.touch_retained(&first).unwrap();
    let first_id = first.id();
    let second_id = second.id();
    let second_allocation = second.handle_id();
    let third_id = third.id();
    drop((first, second, third));
    let incoming = assets.insert(Sample);
    let removals = assets.retain(&incoming, 8, RetentionPolicy::Lru).unwrap();
    assert_eq!(removals.len(), 2);
    assert_eq!(removals[0].id, second_id.untyped());
    assert_eq!(removals[0].handle_id, second_allocation);
    assert_eq!(removals[1].id, third_id.untyped());
    assert!(removals
        .iter()
        .all(|r| r.reason == RetainRemovalReason::Budget && r.bytes == 4));
    assert!(assets.retain_cache().get(first_id).is_some());
    assert_eq!(assets.retain_cache().usage().bytes, 12);
    assert_eq!(assets.collect_releases(), 2);
}

/// Pinning uses the same budget; even potential LRU victims survive a rejected transaction.
#[test]
fn impossible_admission_does_not_partially_evict() {
    let mut assets = arena(8, 3);
    let idle = assets.insert(Sample);
    let pinned = assets.insert(Sample);
    assets.retain(&idle, 4, RetentionPolicy::Lru).unwrap();
    assets.pin(&pinned, 4).unwrap();
    let idle_id = idle.id();
    drop(idle);
    let candidate = assets.insert(Sample);
    let before: Vec<_> = assets.retain_cache().iter().collect();
    let error = assets.pin(&candidate, 5).unwrap_err();
    assert!(matches!(
        error,
        RetainError::BudgetExceeded {
            required_bytes: 9,
            ..
        }
    ));
    assert_eq!(assets.retain_cache().iter().collect::<Vec<_>>(), before);
    assert!(assets.acquire_retained(idle_id).is_some());
    assert_eq!(assets.retain_cache().usage().bytes, 8);
}

/// A smaller budget commits deterministic evictions only if its pinned floor fits.
#[test]
fn budget_shrink_is_atomic_and_pinned_is_not_exempt() {
    let mut assets = arena(12, 3);
    let pinned = assets.insert(Sample);
    let idle = assets.insert(Sample);
    assets.pin(&pinned, 4).unwrap();
    assets.retain(&idle, 8, RetentionPolicy::Lru).unwrap();
    drop(idle);
    let original = assets.retain_cache().budget();
    assert!(assets
        .set_retain_budget(RetainBudget {
            max_bytes: 3,
            max_entries: 3
        })
        .is_err());
    assert_eq!(assets.retain_cache().budget(), original);
    assert_eq!(assets.retain_cache().usage().bytes, 12);
    let removals = assets
        .set_retain_budget(RetainBudget {
            max_bytes: 4,
            max_entries: 1,
        })
        .unwrap();
    assert_eq!(removals.len(), 1);
    assert_eq!(assets.retain_cache().usage().bytes, 4);
    assert_eq!(
        assets.retain_cache().get(pinned.id()).unwrap().policy,
        RetentionPolicy::Pinned
    );
}

/// Zero-byte entries cannot bypass metadata bounds; pinned zero-byte entries still block admission.
#[test]
fn entry_count_limits_zero_byte_assets() {
    let mut assets = arena(0, 1);
    let first = assets.insert(Sample);
    let second = assets.insert(Sample);
    assets.pin(&first, 0).unwrap();
    assert!(matches!(
        assets.pin(&second, 0),
        Err(RetainError::BudgetExceeded {
            required_entries: 2,
            ..
        })
    ));
    assets.unpin(&first).unwrap();
    assets.retain(&first, 0, RetentionPolicy::Lru).unwrap();
    drop(first);
    assert_eq!(assets.pin(&second, 0).unwrap().len(), 1);
    assert_eq!(assets.retain_cache().usage().entries, 1);
}

/// External ownership and fixed KeepAlive windows are not eligible LRU victims.
#[test]
fn active_lru_and_keepalive_reservations_are_protected() {
    let mut assets = arena(8, 2);
    let active = assets.insert(Sample);
    let keep = assets.insert(Sample);
    assets.retain(&active, 4, RetentionPolicy::Lru).unwrap();
    assets
        .retain(&keep, 4, RetentionPolicy::KeepAlive { frames: 3 })
        .unwrap();
    drop(keep);
    let new = assets.insert(Sample);
    assert!(assets.retain(&new, 1, RetentionPolicy::Lru).is_err());
    drop(active);
    assert_eq!(
        assets.retain(&new, 1, RetentionPolicy::Lru).unwrap().len(),
        1
    );
    assert_eq!(assets.retain_cache().usage().bytes, 5);
}

/// Policy updates and repeated pin calls replace a reservation, never double-charge it.
#[test]
fn policy_replacement_and_unpin_are_exact() {
    let mut assets = arena(8, 1);
    let handle = assets.insert(Sample);
    assets.pin(&handle, 4).unwrap();
    assets.pin(&handle, 8).unwrap();
    assert_eq!(assets.retain_cache().usage().bytes, 8);
    assert!(assets.retain(&handle, 9, RetentionPolicy::Lru).is_err());
    assert_eq!(
        assets.retain_cache().get(handle.id()).unwrap().policy,
        RetentionPolicy::Pinned
    );
    assets.retain(&handle, 3, RetentionPolicy::Lru).unwrap();
    assert_eq!(assets.retain_cache().usage().bytes, 3);
    let removed = assets.unpin(&handle).unwrap().unwrap();
    assert_eq!(removed.reason, RetainRemovalReason::Explicit);
    assert_eq!(removed.bytes, 3);
    assert!(assets.unpin(&handle).unwrap().is_none());
    assert!(assets.contains(handle.id()));
}

/// Wide planning arithmetic rejects an overflowing sum without mutating existing usage.
#[test]
fn retention_byte_sum_cannot_overflow() {
    let mut assets = arena(u64::MAX, 2);
    let first = assets.insert(Sample);
    let next = assets.insert(Sample);
    assets.pin(&first, u64::MAX).unwrap();
    let error = assets.pin(&next, 1).unwrap_err();
    match error {
        RetainError::BudgetExceeded { required_bytes, .. } => {
            assert_eq!(required_bytes, u128::from(u64::MAX) + 1)
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(assets.retain_cache().usage().bytes, u64::MAX);
}

/// Reacquiring a retained handle updates order and does not mint a new allocation.
#[test]
fn reacquire_preserves_allocation_and_updates_lru() {
    let mut assets = arena(8, 2);
    let first = assets.insert(Sample);
    let second = assets.insert(Sample);
    assets.retain(&first, 4, RetentionPolicy::Lru).unwrap();
    assets.retain(&second, 4, RetentionPolicy::Lru).unwrap();
    let first_id = first.id();
    let allocation = first.handle_id();
    let weak = first.downgrade();
    let second_id = second.id();
    drop((first, second));
    let reacquired = assets.acquire_retained(first_id).unwrap();
    assert_eq!(reacquired.handle_id(), allocation);
    assert_eq!(weak.upgrade().unwrap().handle_id(), allocation);
    drop(reacquired);
    let next = assets.insert(Sample);
    let removals = assets.pin(&next, 4).unwrap();
    assert_eq!(removals[0].id, second_id.untyped());
}
