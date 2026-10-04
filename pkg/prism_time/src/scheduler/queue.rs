//! [`Scheduler`]: the deterministic min-heap timer queue.

use super::handle::{Fired, TimerHandle, TimerKind};
use crate::Duration;
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::cmp::Reverse;

/// Nanoseconds in one second.
const NANOS_PER_SEC: u128 = 1_000_000_000;

/// Convert a `u128` nanosecond count into a [`Duration`], saturating rather
/// than overflowing the seconds field.
#[inline]
fn duration_from_nanos_u128(nanos: u128) -> Duration {
    let secs = (nanos / NANOS_PER_SEC).min(u64::MAX as u128) as u64;
    let sub = (nanos % NANOS_PER_SEC) as u32;
    Duration::new(secs, sub)
}

/// A backing record for one scheduled timer. Slots are reused via a free list;
/// `generation` distinguishes a reused slot from a retired handle.
#[derive(Clone, Debug)]
struct Slot<T> {
    /// Reuse generation; bumped when the slot is retired (cancel / one-shot
    /// fire) so stale handles and queue entries no longer match.
    generation: u32,
    /// Supersede epoch; bumped on [`Scheduler::reschedule`] so queue entries
    /// enqueued before the reschedule are ignored, while the handle (keyed on
    /// `generation`) stays valid.
    epoch: u32,
    /// Whether this slot currently holds a live timer.
    active: bool,
    /// One-shot or periodic.
    kind: TimerKind,
    /// Period in nanoseconds for a [`TimerKind::Repeat`] timer (`0` otherwise).
    period: u128,
    /// Next scheduled fire time, in elapsed nanoseconds.
    due: u128,
    /// Caller payload, cloned into each [`Fired`] event.
    payload: T,
}

/// An entry in the min-heap: the key is `(due, seq)` so earlier fire times, and
/// then earlier schedule order, come out first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    /// Scheduled fire time, in elapsed nanoseconds.
    due: u128,
    /// Monotonic insertion sequence; the deterministic tie-breaker for equal
    /// `due` values.
    seq: u64,
    /// Backing slot index.
    index: u32,
    /// Slot `generation` captured at enqueue time.
    generation: u32,
    /// Slot `epoch` captured at enqueue time.
    epoch: u32,
}

impl Ord for Entry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.due
            .cmp(&other.due)
            .then(self.seq.cmp(&other.seq))
            .then(self.index.cmp(&other.index))
    }
}

impl PartialOrd for Entry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A deterministic in-game timer scheduler.
///
/// Drive it with [`advance`](Self::advance), feeding the same per-frame delta
/// the clocks use. Due timers are returned as an ordered [`Fired`] stream. See
/// the [module docs](super) for the determinism guarantees.
#[derive(Clone, Debug)]
pub struct Scheduler<T> {
    /// Elapsed scheduler time in nanoseconds (monotonic, exact integer).
    now: u128,
    /// Monotonic insertion counter for deterministic tie-breaking.
    seq: u64,
    /// Slot table (indexed by [`TimerHandle::index`]).
    slots: Vec<Slot<T>>,
    /// Free slot indices available for reuse.
    free: Vec<u32>,
    /// The fire-time min-heap (wrapped in [`Reverse`] over a max-heap).
    heap: BinaryHeap<Reverse<Entry>>,
    /// Number of live (active) timers.
    active_count: usize,
    /// Per-advance cap on fired events (keeps a tiny-period timer under a huge
    /// delta from spinning unbounded within one call; the remainder drains on
    /// the next advance).
    max_fires_per_advance: u32,
}

impl<T> Scheduler<T> {
    /// Default per-advance fire cap (see [`max_fires_per_advance`](Self::max_fires_per_advance)).
    pub const DEFAULT_MAX_FIRES_PER_ADVANCE: u32 = 4096;

    /// An empty scheduler at elapsed time zero.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            now: 0,
            seq: 0,
            slots: Vec::new(),
            free: Vec::new(),
            heap: BinaryHeap::new(),
            active_count: 0,
            max_fires_per_advance: Self::DEFAULT_MAX_FIRES_PER_ADVANCE,
        }
    }

    /// Builder: set the per-advance fire cap. A value of `0` disables firing on
    /// [`advance`](Self::advance) (time still advances; due timers wait).
    #[inline]
    #[must_use]
    pub fn with_max_fires_per_advance(mut self, cap: u32) -> Self {
        self.max_fires_per_advance = cap;
        self
    }

    /// The per-advance fire cap.
    #[inline]
    #[must_use]
    pub fn max_fires_per_advance(&self) -> u32 {
        self.max_fires_per_advance
    }

    /// Set the per-advance fire cap.
    #[inline]
    pub fn set_max_fires_per_advance(&mut self, cap: u32) {
        self.max_fires_per_advance = cap;
    }

    /// Elapsed scheduler time as a [`Duration`].
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        duration_from_nanos_u128(self.now)
    }

    /// Elapsed scheduler time in exact nanoseconds.
    #[inline]
    #[must_use]
    pub fn elapsed_nanos(&self) -> u128 {
        self.now
    }

    /// Number of live timers (scheduled and not yet retired).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.active_count
    }

    /// Whether there are no live timers.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    /// Whether at least one live timer is due at the current elapsed time
    /// (i.e. an [`advance`](Self::advance) or [`drain_due`](Self::drain_due)
    /// would fire something). Skips lazily-invalidated queue entries.
    #[must_use]
    pub fn has_due(&self) -> bool {
        for Reverse(entry) in &self.heap {
            if entry.due > self.now {
                // Not guaranteed to be the minimum (iteration order is heap
                // order, not sorted), so keep scanning rather than returning.
                continue;
            }
            if self.entry_is_live(entry) {
                return true;
            }
        }
        false
    }

    /// Whether a handle still refers to a live timer.
    #[inline]
    #[must_use]
    pub fn is_active(&self, handle: TimerHandle) -> bool {
        self.slots
            .get(handle.index as usize)
            .is_some_and(|s| s.active && s.generation == handle.generation)
    }

    /// Schedule `payload` to fire once, `delay` after the current elapsed time.
    /// A zero `delay` fires at the next drain.
    #[inline]
    pub fn schedule_after(&mut self, delay: Duration, payload: T) -> TimerHandle {
        let due = self.now.saturating_add(delay.as_nanos());
        self.insert(TimerKind::Once, 0, due, payload)
    }

    /// Schedule `payload` to fire once when elapsed time reaches `when`. If
    /// `when` is already in the past it fires at the next drain.
    #[inline]
    pub fn schedule_at(&mut self, when: Duration, payload: T) -> TimerHandle {
        self.insert(TimerKind::Once, 0, when.as_nanos(), payload)
    }

    /// Schedule `payload` to fire every `period`, the first fire landing one
    /// `period` from now.
    ///
    /// # Panics
    /// Panics if `period` is zero (a zero period would never advance past
    /// `now` and so could never make forward progress).
    #[inline]
    pub fn schedule_every(&mut self, period: Duration, payload: T) -> TimerHandle {
        self.schedule_every_from(period, period, payload)
    }

    /// Schedule a periodic `payload`: the first fire lands `first` from now,
    /// then every `period` thereafter.
    ///
    /// # Panics
    /// Panics if `period` is zero.
    pub fn schedule_every_from(
        &mut self,
        first: Duration,
        period: Duration,
        payload: T,
    ) -> TimerHandle {
        let period_nanos = period.as_nanos();
        assert!(period_nanos != 0, "scheduler period must be non-zero");
        let due = self.now.saturating_add(first.as_nanos());
        self.insert(TimerKind::Repeat, period_nanos, due, payload)
    }

    /// Cancel a timer. Returns `true` if the handle referred to a live timer
    /// (which is now retired), `false` if it was already dead.
    pub fn cancel(&mut self, handle: TimerHandle) -> bool {
        if let Some(slot) = self.slots.get_mut(handle.index as usize)
            && slot.active
            && slot.generation == handle.generation
        {
            slot.active = false;
            slot.generation = slot.generation.wrapping_add(1);
            self.free.push(handle.index);
            self.active_count -= 1;
            return true;
        }
        false
    }

    /// Reschedule a live timer to fire `new_delay` from the current elapsed
    /// time. For a periodic timer this also shifts its phase (subsequent fires
    /// are `+period` from the new fire time). The handle stays valid. Returns
    /// `true` on success, `false` if the handle was dead.
    pub fn reschedule(&mut self, handle: TimerHandle, new_delay: Duration) -> bool {
        let due = self.now.saturating_add(new_delay.as_nanos());
        let seq = self.next_seq();
        if let Some(slot) = self.slots.get_mut(handle.index as usize)
            && slot.active
            && slot.generation == handle.generation
        {
            slot.epoch = slot.epoch.wrapping_add(1);
            slot.due = due;
            let entry = Entry {
                due,
                seq,
                index: handle.index,
                generation: slot.generation,
                epoch: slot.epoch,
            };
            self.heap.push(Reverse(entry));
            return true;
        }
        false
    }

    /// Advance elapsed time by `delta` and fire every timer that becomes due,
    /// appending the ordered [`Fired`] events to `out`. Returns the number
    /// fired.
    pub fn advance(&mut self, delta: Duration, out: &mut Vec<Fired<T>>) -> usize
    where
        T: Clone,
    {
        self.now = self.now.saturating_add(delta.as_nanos());
        self.drain(out)
    }

    /// Convenience wrapper around [`advance`](Self::advance) that allocates and
    /// returns the fired events.
    #[inline]
    pub fn advance_collect(&mut self, delta: Duration) -> Vec<Fired<T>>
    where
        T: Clone,
    {
        let mut out = Vec::new();
        self.advance(delta, &mut out);
        out
    }

    /// Fire every timer already due at the current elapsed time without
    /// advancing. Use after an [`advance`](Self::advance) hit the per-advance
    /// fire cap to continue draining the backlog.
    #[inline]
    pub fn drain_due(&mut self, out: &mut Vec<Fired<T>>) -> usize
    where
        T: Clone,
    {
        self.drain(out)
    }

    /// Remove every timer, keeping the elapsed time and the fire cap.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.heap.clear();
        self.active_count = 0;
    }

    // --- internals ---------------------------------------------------------

    /// Allocate (or reuse) a slot and enqueue its first fire.
    fn insert(&mut self, kind: TimerKind, period: u128, due: u128, payload: T) -> TimerHandle {
        let seq = self.next_seq();
        let (index, generation, epoch) = if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.generation = slot.generation.wrapping_add(1);
            slot.epoch = 0;
            slot.active = true;
            slot.kind = kind;
            slot.period = period;
            slot.due = due;
            slot.payload = payload;
            (index, slot.generation, slot.epoch)
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                epoch: 0,
                active: true,
                kind,
                period,
                due,
                payload,
            });
            (index, 0, 0)
        };
        self.active_count += 1;
        self.heap.push(Reverse(Entry {
            due,
            seq,
            index,
            generation,
            epoch,
        }));
        TimerHandle { index, generation }
    }

    /// Next deterministic insertion sequence number.
    #[inline]
    fn next_seq(&mut self) -> u64 {
        let s = self.seq;
        self.seq = self.seq.wrapping_add(1);
        s
    }

    /// Whether a heap entry still matches its slot (not cancelled, not
    /// superseded by a reschedule).
    #[inline]
    fn entry_is_live(&self, entry: &Entry) -> bool {
        self.slots
            .get(entry.index as usize)
            .is_some_and(|s| s.active && s.generation == entry.generation && s.epoch == entry.epoch)
    }

    /// Pop and fire due timers (bounded by the per-advance cap).
    fn drain(&mut self, out: &mut Vec<Fired<T>>) -> usize
    where
        T: Clone,
    {
        let mut fired: u32 = 0;
        while fired < self.max_fires_per_advance {
            let Some(&Reverse(top)) = self.heap.peek() else {
                break;
            };
            if top.due > self.now {
                break;
            }
            self.heap.pop();
            if !self.entry_is_live(&top) {
                // Stale entry (cancelled or superseded); drop it lazily.
                continue;
            }

            let (payload, kind, period) = {
                let slot = &self.slots[top.index as usize];
                (slot.payload.clone(), slot.kind, slot.period)
            };
            out.push(Fired {
                handle: TimerHandle {
                    index: top.index,
                    generation: top.generation,
                },
                payload,
                at: duration_from_nanos_u128(top.due),
                kind,
            });
            fired += 1;

            match kind {
                TimerKind::Once => {
                    let slot = &mut self.slots[top.index as usize];
                    slot.active = false;
                    slot.generation = slot.generation.wrapping_add(1);
                    self.free.push(top.index);
                    self.active_count -= 1;
                }
                TimerKind::Repeat => {
                    let next = top.due.saturating_add(period);
                    let seq = self.next_seq();
                    let slot = &mut self.slots[top.index as usize];
                    slot.due = next;
                    let entry = Entry {
                        due: next,
                        seq,
                        index: top.index,
                        generation: slot.generation,
                        epoch: slot.epoch,
                    };
                    self.heap.push(Reverse(entry));
                }
            }
        }
        fired as usize
    }
}

impl<T> Default for Scheduler<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
