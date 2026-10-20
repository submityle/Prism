//! Exact entity change index used by change observers and reactions.
//!
//! The normal `Changed<T>` query remains the general-purpose query filter. This
//! index is the push-side fast path for registered change observers: mutable
//! borrows and structural writes append the owning entity to the component's
//! tick batch, so observers consume only batches created since their cursor.

use alloc::collections::VecDeque;
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
    batches: VecDeque<ChangeBatch>,
    base_batch: usize,
    event_count: usize,
    loss_revision: u64,
    discarding_tick: Option<Tick>,
}

// Change records are an acceleration index, not the authoritative component
// state. If a consumer falls behind, it can rebuild its candidate set from
// current component ticks. Keep the acceleration structure bounded and make
// that loss explicit to every cursor.
pub(crate) const MAX_RETAINED_BATCHES: usize = 4096;
pub(crate) const MAX_RETAINED_EVENTS: usize = 65_536;

struct TrackerState {
    epoch: u64,
    revision: u64,
    logs: Vec<Option<ComponentLog>>,
    /// Components with at least one indexed write since the last observer
    /// candidate snapshot. The parallel `dirty_flags` table makes repeated
    /// writes O(1) without growing this list.
    dirty_components: Vec<ComponentId>,
    dirty_flags: Vec<bool>,
}

/// Cursor into one component's exact change log.
#[derive(Clone, Copy, Default)]
pub(crate) struct ChangeCursor {
    pub(crate) epoch: u64,
    pub(crate) next_batch: usize,
    loss_revision: u64,
}

/// Collection diagnostics kept deliberately small so tests and future tracing
/// can verify that delivery is proportional to indexed changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ChangeIndexStats {
    pub(crate) batches: usize,
    pub(crate) events: usize,
    pub(crate) gap: bool,
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
                dirty_components: Vec::new(),
                dirty_flags: Vec::new(),
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

    /// Atomically take the component-level write summary used to select
    /// change-observer candidates. Writes made after this call (notably from
    /// callbacks) start the next summary and therefore remain visible to the
    /// next flush.
    pub(crate) fn take_dirty_components(&self) -> Vec<ComponentId> {
        let mut state = self.state.lock();
        let dirty = core::mem::take(&mut state.dirty_components);
        for component in &dirty {
            state.dirty_flags[component.index() as usize] = false;
        }
        dirty
    }

    fn ensure_log(state: &mut TrackerState, component: ComponentId) -> &mut ComponentLog {
        let index = component.index() as usize;
        if state.logs.len() <= index {
            state.logs.resize_with(index + 1, || None);
        }
        state.logs[index].get_or_insert_with(|| ComponentLog {
            active: 0,
            batches: VecDeque::new(),
            base_batch: 0,
            event_count: 0,
            loss_revision: 0,
            discarding_tick: None,
        })
    }

    pub(crate) fn activate(&self, component: ComponentId) -> ChangeCursor {
        let mut state = self.state.lock();
        let epoch = state.epoch;
        let log = Self::ensure_log(&mut state, component);
        log.active += 1;
        ChangeCursor {
            epoch,
            next_batch: log.base_batch + log.batches.len(),
            loss_revision: log.loss_revision,
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
            log.base_batch = log.base_batch.saturating_add(log.batches.len());
            log.batches.clear();
            log.event_count = 0;
            log.discarding_tick = None;
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
        let component_index = component.index() as usize;
        if state.dirty_flags.len() <= component_index {
            state.dirty_flags.resize(component_index + 1, false);
        }
        if !state.dirty_flags[component_index] {
            state.dirty_flags[component_index] = true;
            state.dirty_components.push(component);
        }
        let log = state.logs[component_index]
            .as_mut()
            .expect("ensured component log");
        if log.discarding_tick == Some(tick) {
            // A cursor may have activated after the batch first overflowed.
            // Publish every later dropped write so that such a cursor still
            // receives a gap and can rebuild from authoritative state.
            log.loss_revision = log.loss_revision.saturating_add(1);
            return;
        }
        if log.discarding_tick.is_some() {
            log.discarding_tick = None;
        }
        if let Some(batch) = log.batches.back_mut()
            && batch.tick == tick
        {
            if log.event_count == MAX_RETAINED_EVENTS {
                log.base_batch = log.base_batch.saturating_add(log.batches.len());
                log.batches.clear();
                log.event_count = 0;
                log.loss_revision = log.loss_revision.saturating_add(1);
                log.discarding_tick = Some(tick);
                return;
            }
            batch.entities.push(entity);
            log.event_count += 1;
            return;
        }
        log.batches.push_back(ChangeBatch {
            tick,
            entities: alloc::vec![entity],
        });
        log.event_count += 1;
        while log.batches.len() > MAX_RETAINED_BATCHES || log.event_count > MAX_RETAINED_EVENTS {
            let expired = log.batches.pop_front().expect("over-limit log is nonempty");
            log.event_count -= expired.entities.len();
            log.base_batch = log.base_batch.saturating_add(1);
            log.loss_revision = log.loss_revision.saturating_add(1);
        }
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
        let mut gap = false;
        if cursor.epoch != state.epoch {
            cursor.epoch = state.epoch;
            cursor.next_batch = 0;
            cursor.loss_revision = log.loss_revision;
            gap = true;
        }
        if cursor.loss_revision != log.loss_revision {
            gap = true;
        }
        if cursor.next_batch < log.base_batch {
            gap = true;
            cursor.next_batch = log.base_batch;
        }
        let start = cursor
            .next_batch
            .saturating_sub(log.base_batch)
            .min(log.batches.len());
        let mut stats = ChangeIndexStats {
            gap,
            ..ChangeIndexStats::default()
        };
        for batch in log.batches.iter().skip(start) {
            cursor.next_batch += 1;
            if batch.tick.is_newer_than(last_run, this_run) {
                stats.batches += 1;
                stats.events += batch.entities.len();
                out.extend_from_slice(&batch.entities);
            }
        }
        cursor.loss_revision = log.loss_revision;
        stats
    }

    /// Drop all old log storage after a snapshot restore while preserving the
    /// registered observer set. Existing cursors detect the epoch transition.
    pub(crate) fn reset(&self) -> ChangeCursor {
        let mut state = self.state.lock();
        state.epoch = state.epoch.wrapping_add(1);
        state.revision = state.revision.saturating_add(1);
        for log in state.logs.iter_mut().flatten() {
            log.base_batch = log.base_batch.saturating_add(log.batches.len());
            log.batches.clear();
            log.event_count = 0;
            log.discarding_tick = None;
            log.loss_revision = log.loss_revision.saturating_add(1);
        }
        state.dirty_components.clear();
        state.dirty_flags.fill(false);
        ChangeCursor {
            epoch: state.epoch,
            next_batch: 0,
            loss_revision: state
                .logs
                .iter()
                .flatten()
                .map(|log| log.loss_revision)
                .max()
                .unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod revision_tests {
    use super::*;

    fn entity(index: u32) -> Entity {
        Entity::from_bits((1_u64 << 32) | u64::from(index)).unwrap()
    }

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

    #[test]
    fn collect_delivers_indexed_batches_across_tick_wrap() {
        let tracker = ChangeTracker::new();
        let component = ComponentId::new(0);
        let mut cursor = tracker.activate(component);
        let batches = [
            (Tick::new(u32::MAX - 1), alloc::vec![entity(1)]),
            (Tick::MAX, alloc::vec![entity(2)]),
            (Tick::ZERO, alloc::vec![entity(3), entity(4)]),
            (Tick::new(1), alloc::vec![entity(5)]),
        ];
        let mut last_run = Tick::new(u32::MAX - 2);

        for (tick, expected) in batches {
            for &changed_entity in &expected {
                tracker.record(component, changed_entity, tick);
            }

            let mut collected = Vec::new();
            let stats = tracker.collect(component, &mut cursor, last_run, tick, &mut collected);

            assert_eq!(collected, expected, "incorrect indexed batch at {tick:?}");
            assert_eq!(stats.batches, 1);
            assert_eq!(stats.events, expected.len());
            last_run = tick;
        }
    }

    #[test]
    fn lagging_cursor_reports_gap_and_retained_history_is_bounded() {
        let tracker = ChangeTracker::new();
        let component = ComponentId::new(0);
        let mut cursor = tracker.activate(component);
        for tick in 1..=(MAX_RETAINED_BATCHES as u32 + 32) {
            tracker.record(component, entity(tick), Tick::new(tick));
        }

        let log = tracker.state.lock();
        let component_log = log.logs[component.index() as usize]
            .as_ref()
            .expect("active log exists");
        assert_eq!(component_log.batches.len(), MAX_RETAINED_BATCHES);
        assert!(component_log.event_count <= MAX_RETAINED_EVENTS);
        drop(log);

        let mut out = Vec::new();
        let stats = tracker.collect(
            component,
            &mut cursor,
            Tick::ZERO,
            Tick::new(MAX_RETAINED_BATCHES as u32 + 32),
            &mut out,
        );
        assert!(
            stats.gap,
            "evicted records must be visible to lagging consumers"
        );
        assert_eq!(out.len(), MAX_RETAINED_BATCHES);
    }

    #[test]
    fn oversized_single_tick_batch_is_capped_until_next_tick() {
        let tracker = ChangeTracker::new();
        let component = ComponentId::new(0);
        let mut cursor = tracker.activate(component);
        let oversized_tick = Tick::new(1);
        for index in 0..=(MAX_RETAINED_EVENTS as u32) {
            tracker.record(component, entity(index), oversized_tick);
        }

        let mut out = Vec::new();
        let stats = tracker.collect(component, &mut cursor, Tick::ZERO, oversized_tick, &mut out);
        assert!(stats.gap);
        assert!(out.is_empty());
        {
            let state = tracker.state.lock();
            let log = state.logs[component.index() as usize].as_ref().unwrap();
            assert!(log.batches.is_empty());
            assert_eq!(log.event_count, 0);
        }

        tracker.record(component, entity(1), Tick::new(2));
        let stats = tracker.collect(
            component,
            &mut cursor,
            oversized_tick,
            Tick::new(2),
            &mut out,
        );
        assert!(!stats.gap);
        assert_eq!(out, [entity(1)]);
    }

    #[test]
    fn new_cursor_is_not_blind_to_dropped_writes_in_overflow_tick() {
        let tracker = ChangeTracker::new();
        let component = ComponentId::new(0);
        let mut old_cursor = tracker.activate(component);
        let tick = Tick::new(1);
        for index in 0..=MAX_RETAINED_EVENTS as u32 {
            tracker.record(component, entity(index), tick);
        }
        let mut new_cursor = tracker.activate(component);
        tracker.record(component, entity(90_000), tick);

        let mut out = Vec::new();
        let stats = tracker.collect(component, &mut new_cursor, Tick::ZERO, tick, &mut out);
        assert!(stats.gap);
        assert!(out.is_empty());
        let stats = tracker.collect(component, &mut old_cursor, Tick::ZERO, tick, &mut out);
        assert!(stats.gap);
    }
}
