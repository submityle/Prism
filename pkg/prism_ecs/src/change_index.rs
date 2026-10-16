//! Exact entity change index used by change observers and reactions.
//!
//! The normal `Changed<T>` query remains the general-purpose query filter. This
//! index is the push-side fast path for registered change observers: mutable
//! borrows and structural writes append the owning entity to the component's
//! tick batch, so observers consume only batches created since their cursor.

use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::hint::spin_loop;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::change::Tick;
use crate::component::ComponentId;
use crate::entity::Entity;

struct SpinLock<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

// SAFETY: all access to `value` is serialized by `locked`; sharing the lock
// across threads therefore requires only that the protected value is `Send`.
unsafe impl<T: Send> Sync for SpinLock<T> {}
// SAFETY: moving the lock transfers exclusive ownership of its `UnsafeCell<T>`;
// `T: Send` is sufficient for that ownership transfer.
unsafe impl<T: Send> Send for SpinLock<T> {}

impl<T> SpinLock<T> {
    fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    fn lock(&self) -> Guard<'_, T> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        Guard { lock: self }
    }
}

struct Guard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> core::ops::Deref for Guard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: the guard owns the lock for the lifetime of this borrow.
        unsafe { &*self.lock.value.get() }
    }
}

impl<T> core::ops::DerefMut for Guard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: the guard owns the lock exclusively.
        unsafe { &mut *self.lock.value.get() }
    }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

struct ChangeBatch {
    tick: Tick,
    entities: Vec<Entity>,
}

struct ComponentLog {
    active: usize,
    batches: Vec<ChangeBatch>,
}

struct TrackerState {
    epoch: u64,
    revision: u64,
    logs: Vec<Option<ComponentLog>>,
}

/// Cursor into one component's exact change log.
#[derive(Clone, Copy, Default)]
pub(crate) struct ChangeCursor {
    pub(crate) epoch: u64,
    pub(crate) next_batch: usize,
}

/// Collection diagnostics kept deliberately small so tests and future tracing
/// can verify that delivery is proportional to indexed changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ChangeIndexStats {
    pub(crate) batches: usize,
    pub(crate) events: usize,
}

/// Per-world exact change index.
pub(crate) struct ChangeTracker {
    state: SpinLock<TrackerState>,
}

/// Stable tracker identity carried by a query fetch. The entity is supplied by
/// the row walker, so one immutable value can be copied by serial and parallel
/// query paths without sharing row-specific state.
#[derive(Clone, Copy)]
pub struct ChangeRecorder {
    tracker: *const ChangeTracker,
    component: ComponentId,
}

/// Complete write identity for one component/entity pair.
#[derive(Clone, Copy)]
pub(crate) struct ChangeRecord {
    tracker: *const ChangeTracker,
    component: ComponentId,
    entity: Entity,
}

impl ChangeRecorder {
    #[inline]
    pub(crate) fn new(tracker: *const ChangeTracker, component: ComponentId) -> Self {
        Self { tracker, component }
    }

    #[inline]
    pub(crate) fn for_entity(self, entity: Entity) -> ChangeRecord {
        ChangeRecord {
            tracker: self.tracker,
            component: self.component,
            entity,
        }
    }
}

impl ChangeRecord {
    #[inline]
    pub(crate) fn record(self, tick: Tick) {
        // SAFETY: the pointer is derived from the Arc-owned tracker held by the
        // World/SparseSets for the complete query borrow. ChangeTracker's lock
        // makes concurrent row writers safe.
        unsafe { (&*self.tracker).record(self.component, self.entity, tick) };
    }
}

impl Default for ChangeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ChangeTracker {
    pub(crate) fn new() -> Self {
        Self {
            state: SpinLock::new(TrackerState {
                epoch: 0,
                revision: 0,
                logs: Vec::new(),
            }),
        }
    }

    #[inline]
    pub(crate) fn recorder(&self, component: ComponentId) -> ChangeRecorder {
        ChangeRecorder::new(self as *const Self, component)
    }

    /// O(1) snapshot of all indexed writes and resets, independent of ticks and
    /// per-component batch boundaries. A saturated counter disables clean-frame
    /// skipping permanently rather than aliasing an earlier revision.
    pub(crate) fn revision(&self) -> Option<u64> {
        let revision = self.state.lock().revision;
        (revision != u64::MAX).then_some(revision)
    }

    fn ensure_log(state: &mut TrackerState, component: ComponentId) -> &mut ComponentLog {
        let index = component.index() as usize;
        if state.logs.len() <= index {
            state.logs.resize_with(index + 1, || None);
        }
        state.logs[index].get_or_insert_with(|| ComponentLog {
            active: 0,
            batches: Vec::new(),
        })
    }

    pub(crate) fn activate(&self, component: ComponentId) -> ChangeCursor {
        let mut state = self.state.lock();
        let epoch = state.epoch;
        let log = Self::ensure_log(&mut state, component);
        log.active += 1;
        ChangeCursor {
            epoch,
            next_batch: log.batches.len(),
        }
    }

    pub(crate) fn deactivate(&self, component: ComponentId) {
        let mut state = self.state.lock();
        let index = component.index() as usize;
        let Some(Some(log)) = state.logs.get_mut(index) else {
            return;
        };
        log.active = log.active.saturating_sub(1);
        if log.active == 0 {
            log.batches.clear();
        }
    }

    #[inline]
    pub(crate) fn record(&self, component: ComponentId, entity: Entity, tick: Tick) {
        let mut state = self.state.lock();
        if Self::ensure_log(&mut state, component).active == 0 {
            return;
        }
        // Count every append, including another write in the same tick batch.
        state.revision = state.revision.saturating_add(1);
        let log = state.logs[component.index() as usize]
            .as_mut()
            .expect("ensured component log");
        if let Some(batch) = log.batches.last_mut()
            && batch.tick == tick
        {
            batch.entities.push(entity);
            return;
        }
        log.batches.push(ChangeBatch {
            tick,
            entities: alloc::vec![entity],
        });
    }

    pub(crate) fn collect(
        &self,
        component: ComponentId,
        cursor: &mut ChangeCursor,
        last_run: Tick,
        this_run: Tick,
        out: &mut Vec<Entity>,
    ) -> ChangeIndexStats {
        let state = self.state.lock();
        let Some(Some(log)) = state.logs.get(component.index() as usize) else {
            return ChangeIndexStats::default();
        };
        if cursor.epoch != state.epoch {
            cursor.epoch = state.epoch;
            cursor.next_batch = 0;
        }
        let start = cursor.next_batch.min(log.batches.len());
        let mut stats = ChangeIndexStats::default();
        for batch in &log.batches[start..] {
            cursor.next_batch += 1;
            if batch.tick.is_newer_than(last_run, this_run) {
                stats.batches += 1;
                stats.events += batch.entities.len();
                out.extend_from_slice(&batch.entities);
            }
        }
        stats
    }

    /// Drop all old log storage after a snapshot restore while preserving the
    /// registered observer set. Existing cursors detect the epoch transition.
    pub(crate) fn reset(&self) -> ChangeCursor {
        let mut state = self.state.lock();
        state.epoch = state.epoch.wrapping_add(1);
        state.revision = state.revision.saturating_add(1);
        for log in state.logs.iter_mut().flatten() {
            log.batches.clear();
        }
        ChangeCursor {
            epoch: state.epoch,
            next_batch: 0,
        }
    }
}

#[cfg(test)]
mod revision_tests {
    use super::*;

    #[test]
    fn saturated_revision_never_aliases_a_clean_baseline() {
        let tracker = ChangeTracker::new();
        tracker.state.lock().revision = u64::MAX - 1;
        assert_eq!(tracker.revision(), Some(u64::MAX - 1));
        tracker.reset();
        assert_eq!(tracker.revision(), None);
        tracker.reset();
        assert_eq!(tracker.revision(), None);
    }
}
