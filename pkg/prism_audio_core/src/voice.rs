//! Fixed-capacity voice pool with priority stealing and virtual-voice behavior.
//!
//! A *voice* is one sounding instance of a source (a single `Event` trigger can
//! spawn several). Rendering an unbounded number of them would blow the CPU
//! budget, so — like Wwise's Virtual Voices / Playback Limit and Godot's
//! `AudioStreamPolyphonic` — Resonance manages voices through a **pre-allocated
//! pool** that never grows on the audio thread. This module is the pure,
//! device-agnostic bookkeeping layer: it decides *which* voices are audible,
//! which are demoted to inaudible "virtual" state, which are stolen, and where
//! each playhead sits. It renders no audio itself; the render graph
//! ([`crate::graph`]) drives the nodes behind the voices the pool keeps
//! physical.
//!
//! # Model
//!
//! Every slot is [`Free`](VoiceState::Free), [`Physical`](VoiceState::Physical)
//! (audible, rendered), or [`Virtual`](VoiceState::Virtual) (culled but still
//! tracked so it can be revived). Two independent limits govern the pool:
//!
//! - **Capacity** — the total number of slots (physical + virtual). When it is
//!   exhausted, [`allocate`](VoicePool::allocate) either steals the globally
//!   weakest voice (if the newcomer is more important) or refuses.
//! - **Physical budget** — the maximum number of *audible* voices, tracking the
//!   CPU budget of the quality governor. Exceeding it demotes the weakest
//!   physical voices to virtual according to their [`VirtualBehavior`].
//!
//! Per-group **playback limits** cap concurrent instances of one logical source
//! (e.g. at most eight simultaneous footsteps), enforced with a configurable
//! [`LimitPolicy`]. A single logical source can therefore reuse several pooled
//! voices for overlapping one-shots without hand-managing instances.
//!
//! Effective **importance** (an already-combined estimate of explicit priority
//! × loudness × distance × masking) is the single scalar the pool ranks voices
//! by, so higher layers stay free to compute it however they like.
//!
//! # Real-time contract
//!
//! The pool reserves all storage in [`VoicePool::new`] (and when group limits
//! are configured off-thread). [`allocate`](VoicePool::allocate),
//! [`release`](VoicePool::release), [`advance`](VoicePool::advance),
//! [`revoice`](VoicePool::revoice), and every query are allocation-free,
//! lock-free, and panic-free, so they are safe to call from the audio callback.
//!
//! # Example
//!
//! ```
//! use prism_audio_core::voice::{LimitPolicy, VirtualBehavior, VoiceGroup, VoicePool, VoiceRequest};
//!
//! // 4 slots total, at most 2 audible at once.
//! let mut pool = VoicePool::new(4, 2);
//! let footsteps = VoiceGroup(1);
//! pool.set_group_limit(footsteps, 3, LimitPolicy::ReplaceOldest);
//!
//! let a = pool.allocate(VoiceRequest::new(footsteps, 0.9)).unwrap();
//! let b = pool.allocate(VoiceRequest::new(footsteps, 0.5)).unwrap();
//! // Third footstep exceeds the physical budget: the weakest one goes virtual.
//! let _c = pool.allocate(VoiceRequest {
//!     behavior: VirtualBehavior::ContinueVirtual,
//!     ..VoiceRequest::new(footsteps, 0.8)
//! });
//! assert_eq!(pool.physical_count(), 2);
//! assert!(pool.is_virtual(b)); // 0.5 was the weakest and lost its physical slot
//! assert!(pool.is_physical(a));
//! ```

use alloc::vec::Vec;

use crate::math::Sample;

/// Effective, already-combined voice importance (priority × loudness × distance
/// × masking). Higher means "keep me audible"; the pool never interprets the
/// individual factors, only compares the final scalar.
pub type Importance = Sample;

/// What happens to a voice when it is culled from the audible set, and how it
/// resumes if revived. Mirrors Wwise's Virtual Voice Behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum VirtualBehavior {
    /// Stay virtual and keep advancing the playhead silently, so a revived
    /// voice resumes exactly in sync. Correct for loops and music beds.
    ContinueVirtual,
    /// Stop and free the slot immediately on cull; never becomes virtual.
    Kill,
    /// Stay virtual with the playhead frozen; a revived voice restarts from the
    /// beginning. Correct for short, retriggerable one-shots.
    RestartFromBeginning,
    /// Stay virtual with the playhead frozen but count elapsed wall-clock time;
    /// a revived voice jumps forward to where it *would* be, then keeps playing.
    PlayFromElapsedTime,
}

impl VirtualBehavior {
    /// Whether a culled voice keeps occupying a slot (`true`) or frees it
    /// (`false`, i.e. [`Kill`](VirtualBehavior::Kill)).
    #[must_use]
    #[inline]
    pub const fn keeps_slot(self) -> bool {
        !matches!(self, VirtualBehavior::Kill)
    }
}

/// Lifecycle state of a pool slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum VoiceState {
    /// Unused slot, available for [`VoicePool::allocate`].
    Free,
    /// Audible voice that the render graph should process this block.
    Physical,
    /// Culled voice retained for possible revival; not rendered.
    Virtual,
}

/// Policy applied when a per-group [playback limit](VoicePool::set_group_limit)
/// is already saturated and another instance is requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum LimitPolicy {
    /// Refuse the new request, leaving existing instances untouched.
    RejectNewest,
    /// Free the oldest instance in the group to make room for the newcomer.
    ReplaceOldest,
    /// Free the least-important instance, but only if the newcomer is strictly
    /// more important; otherwise refuse.
    ReplaceWeakest,
}

/// Opaque identifier for a logical source / instance category, used to enforce
/// per-group [playback limits](VoicePool::set_group_limit). Group `0` is the
/// conventional "ungrouped" bucket and is never limited unless configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceGroup(
    /// Caller-defined group id; the pool treats it as an opaque key.
    pub u32,
);

/// Hysteresis band the [`VoicePool::revoice`] pass uses to decide when a voice
/// crosses between physical and virtual, avoiding chatter at the boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VirtualizationThresholds {
    /// Importance a *virtual* voice must exceed to be promoted back to physical.
    pub enter: Importance,
    /// Importance a *physical* voice must fall below to be demoted to virtual.
    /// Keep `leave < enter` for a stable hysteresis gap.
    pub leave: Importance,
}

impl VirtualizationThresholds {
    /// Builds a threshold band, clamping so `leave <= enter` always holds.
    #[must_use]
    #[inline]
    pub fn new(enter: Importance, leave: Importance) -> Self {
        let enter = sanitize(enter);
        let leave = sanitize(leave).min(enter);
        Self { enter, leave }
    }
}

/// A request to spawn a voice.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceRequest {
    /// Group used for per-source playback limiting.
    pub group: VoiceGroup,
    /// Explicit caller priority, retained for telemetry and tie-breaking.
    pub priority: u8,
    /// Effective importance used for stealing and virtualization ranking.
    pub importance: Importance,
    /// Behavior applied if the voice is later culled.
    pub behavior: VirtualBehavior,
}

impl VoiceRequest {
    /// Convenience constructor: ungrouped-safe group, default priority, and the
    /// [`ContinueVirtual`](VirtualBehavior::ContinueVirtual) behavior.
    #[must_use]
    #[inline]
    pub fn new(group: VoiceGroup, importance: Importance) -> Self {
        Self {
            group,
            priority: 0,
            importance,
            behavior: VirtualBehavior::ContinueVirtual,
        }
    }
}

/// A stable, generation-checked reference to a pooled voice. Reusing a slot
/// bumps its generation, so a handle to a released voice safely resolves to
/// `None` instead of aliasing the new occupant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceHandle {
    /// Slot index within the pool.
    index: u32,
    /// Generation the slot had when this handle was minted.
    generation: u32,
}

impl VoiceHandle {
    /// The slot index this handle points at (for telemetry / debugging).
    #[must_use]
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The generation stamp captured when the handle was minted.
    #[must_use]
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// Read-only snapshot of a live voice, returned by [`VoicePool::get`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiceInfo {
    /// Group the voice belongs to.
    pub group: VoiceGroup,
    /// Explicit caller priority.
    pub priority: u8,
    /// Current effective importance.
    pub importance: Importance,
    /// Culling behavior.
    pub behavior: VirtualBehavior,
    /// Lifecycle state (never [`Free`](VoiceState::Free) for a live snapshot).
    pub state: VoiceState,
    /// Frames the audible playhead has advanced.
    pub elapsed: u64,
}

/// Internal per-slot storage. Kept private so invariants live entirely inside
/// [`VoicePool`].
#[derive(Debug, Clone, Copy)]
struct Slot {
    /// Current lifecycle state.
    state: VoiceState,
    /// Owning group.
    group: VoiceGroup,
    /// Explicit caller priority.
    priority: u8,
    /// Effective importance (always finite; sanitized on write).
    importance: Importance,
    /// Culling behavior.
    behavior: VirtualBehavior,
    /// Audible playhead position in frames.
    elapsed: u64,
    /// Wall-clock frames accrued while virtual, for
    /// [`PlayFromElapsedTime`](VirtualBehavior::PlayFromElapsedTime).
    virtual_frames: u64,
    /// Monotonic allocation sequence for age ordering.
    seq: u64,
    /// Generation stamp; bumped every time the slot is freed.
    generation: u32,
}

impl Slot {
    /// A pristine free slot.
    const fn vacant() -> Self {
        Self {
            state: VoiceState::Free,
            group: VoiceGroup(0),
            priority: 0,
            importance: 0.0,
            behavior: VirtualBehavior::Kill,
            elapsed: 0,
            virtual_frames: 0,
            seq: 0,
            generation: 0,
        }
    }

    /// Whether this slot currently holds a live (physical or virtual) voice.
    #[inline]
    const fn is_live(&self) -> bool {
        !matches!(self.state, VoiceState::Free)
    }
}

/// A configured per-group playback limit.
#[derive(Debug, Clone, Copy)]
struct GroupLimit {
    /// Group the limit applies to.
    group: VoiceGroup,
    /// Maximum concurrent live instances in the group.
    max: usize,
    /// Policy applied when the limit is already reached.
    policy: LimitPolicy,
}

/// Fixed-capacity voice pool. See the [module docs](self) for the model and the
/// real-time contract.
#[derive(Debug, Clone)]
pub struct VoicePool {
    /// Backing slot storage; length equals the fixed capacity.
    slots: Vec<Slot>,
    /// Configured per-group playback limits (small; scanned linearly).
    limits: Vec<GroupLimit>,
    /// Maximum number of simultaneously audible (physical) voices.
    max_physical: usize,
    /// Monotonic allocation counter for age ordering.
    next_seq: u64,
}

impl VoicePool {
    /// Creates a pool with `capacity` total slots and a physical (audible)
    /// budget of `max_physical`, clamped to `capacity`. All storage is
    /// allocated here so the audio thread never does.
    #[must_use]
    pub fn new(capacity: usize, max_physical: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(Slot::vacant());
        }
        Self {
            slots,
            limits: Vec::new(),
            max_physical: max_physical.min(capacity),
            next_seq: 0,
        }
    }

    /// Total number of slots (physical + virtual + free).
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Current physical (audible) budget.
    #[must_use]
    #[inline]
    pub fn max_physical(&self) -> usize {
        self.max_physical
    }

    /// Adjusts the physical budget (clamped to capacity) and immediately
    /// re-enforces it, demoting the weakest physical voices if the budget
    /// shrank. Intended for the quality governor's CPU-budget feedback loop.
    pub fn set_max_physical(&mut self, max_physical: usize) {
        self.max_physical = max_physical.min(self.slots.len());
        self.enforce_physical_limit();
    }

    /// Configures (or replaces) the playback limit for `group`. Configuration
    /// may allocate and is meant to run off the audio thread.
    pub fn set_group_limit(&mut self, group: VoiceGroup, max: usize, policy: LimitPolicy) {
        if let Some(existing) = self.limits.iter_mut().find(|l| l.group == group) {
            existing.max = max;
            existing.policy = policy;
        } else {
            self.limits.push(GroupLimit { group, max, policy });
        }
    }

    /// Removes any playback limit configured for `group`.
    pub fn clear_group_limit(&mut self, group: VoiceGroup) {
        self.limits.retain(|l| l.group != group);
    }

    /// Attempts to spawn a voice, returning its handle.
    ///
    /// Returns `None` when the request is refused: the group's playback limit
    /// rejected it, or the pool is full and the newcomer is not more important
    /// than the weakest resident. On success the voice starts
    /// [`Physical`](VoiceState::Physical); if that exceeds the physical budget
    /// the weakest physical voice (possibly the newcomer) is demoted per its
    /// behavior, so a returned handle may already be virtual.
    pub fn allocate(&mut self, request: VoiceRequest) -> Option<VoiceHandle> {
        let importance = sanitize(request.importance);

        // 1. Per-group playback limit.
        if let Some(limit) = self.group_limit(request.group)
            && self.group_count(request.group) >= limit.max
        {
            match limit.policy {
                LimitPolicy::RejectNewest => return None,
                LimitPolicy::ReplaceOldest => {
                    let victim = self.oldest_in_group(request.group)?;
                    self.free_slot(victim);
                }
                LimitPolicy::ReplaceWeakest => {
                    let victim = self.weakest_in_group(request.group)?;
                    if self.slots[victim].importance >= importance {
                        return None;
                    }
                    self.free_slot(victim);
                }
            }
        }

        // 2. Find a free slot, or steal the globally weakest voice.
        let index = match self.free_index() {
            Some(i) => i,
            None => {
                let victim = self.weakest_index()?;
                if self.slots[victim].importance >= importance {
                    return None; // newcomer is the weakest: refuse.
                }
                self.free_slot(victim);
                victim
            }
        };

        // 3. Occupy the slot as a physical voice.
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let generation = {
            let slot = &mut self.slots[index];
            slot.state = VoiceState::Physical;
            slot.group = request.group;
            slot.priority = request.priority;
            slot.importance = importance;
            slot.behavior = request.behavior;
            slot.elapsed = 0;
            slot.virtual_frames = 0;
            slot.seq = seq;
            slot.generation
        };

        // 4. Respect the physical budget (may virtualize the newcomer).
        self.enforce_physical_limit();

        Some(VoiceHandle {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "index is a slot position < capacity, which fits in u32 in practice"
            )]
            index: index as u32,
            generation,
        })
    }

    /// Releases a voice, freeing its slot. Returns `true` if the handle was
    /// live, `false` if it was stale or already free.
    pub fn release(&mut self, handle: VoiceHandle) -> bool {
        match self.resolve(handle) {
            Some(index) => {
                self.free_slot(index);
                true
            }
            None => false,
        }
    }

    /// Updates a live voice's effective importance. Returns `false` for a stale
    /// handle. Does not itself re-rank; call [`revoice`](Self::revoice) or rely
    /// on the next [`allocate`](Self::allocate) to act on the change.
    pub fn set_importance(&mut self, handle: VoiceHandle, importance: Importance) -> bool {
        match self.resolve(handle) {
            Some(index) => {
                self.slots[index].importance = sanitize(importance);
                true
            }
            None => false,
        }
    }

    /// Advances every live voice's playhead by `frames`.
    ///
    /// Physical voices and [`ContinueVirtual`](VirtualBehavior::ContinueVirtual)
    /// virtual voices advance their audible playhead; a
    /// [`PlayFromElapsedTime`](VirtualBehavior::PlayFromElapsedTime) virtual
    /// voice instead accrues wall-clock frames to fast-forward on revival; a
    /// [`RestartFromBeginning`](VirtualBehavior::RestartFromBeginning) virtual
    /// voice stays frozen. Saturating arithmetic keeps it panic-free.
    pub fn advance(&mut self, frames: u64) {
        for slot in &mut self.slots {
            match slot.state {
                VoiceState::Physical => {
                    slot.elapsed = slot.elapsed.saturating_add(frames);
                }
                VoiceState::Virtual => match slot.behavior {
                    VirtualBehavior::ContinueVirtual => {
                        slot.elapsed = slot.elapsed.saturating_add(frames);
                    }
                    VirtualBehavior::PlayFromElapsedTime => {
                        slot.virtual_frames = slot.virtual_frames.saturating_add(frames);
                    }
                    VirtualBehavior::RestartFromBeginning | VirtualBehavior::Kill => {}
                },
                VoiceState::Free => {}
            }
        }
    }

    /// Applies the hysteresis virtualization pass: demotes physical voices whose
    /// importance fell below `thresholds.leave`, then promotes the most
    /// important virtual voices that exceed `thresholds.enter` while physical
    /// headroom remains. Promotion honors each voice's behavior when resuming
    /// the playhead.
    pub fn revoice(&mut self, thresholds: VirtualizationThresholds) {
        // Demote faded-out physical voices.
        for i in 0..self.slots.len() {
            if matches!(self.slots[i].state, VoiceState::Physical)
                && self.slots[i].importance < thresholds.leave
            {
                self.cull(i);
            }
        }

        // Promote the strongest eligible virtual voices while there is room.
        loop {
            if self.physical_count() >= self.max_physical {
                break;
            }
            let Some(best) = self.best_virtual_above(thresholds.enter) else {
                break;
            };
            self.promote(best);
        }
    }

    /// Forces a live voice into the virtual state per its behavior (or frees it
    /// if its behavior is [`Kill`](VirtualBehavior::Kill)). Returns `false` for
    /// a stale handle or one that is not currently physical.
    pub fn virtualize(&mut self, handle: VoiceHandle) -> bool {
        match self.resolve(handle) {
            Some(index) if matches!(self.slots[index].state, VoiceState::Physical) => {
                self.cull(index);
                true
            }
            _ => false,
        }
    }

    /// Promotes a virtual voice back to physical, resuming its playhead per its
    /// behavior. Returns `false` for a stale handle or one that is not virtual.
    /// Does not check the physical budget; the caller (or a following
    /// [`revoice`](Self::revoice)) is responsible for staying within it.
    pub fn revive(&mut self, handle: VoiceHandle) -> bool {
        match self.resolve(handle) {
            Some(index) if matches!(self.slots[index].state, VoiceState::Virtual) => {
                self.promote(index);
                true
            }
            _ => false,
        }
    }

    /// Returns a read-only snapshot of a live voice, or `None` if stale/free.
    #[must_use]
    pub fn get(&self, handle: VoiceHandle) -> Option<VoiceInfo> {
        let index = self.resolve(handle)?;
        let slot = &self.slots[index];
        Some(VoiceInfo {
            group: slot.group,
            priority: slot.priority,
            importance: slot.importance,
            behavior: slot.behavior,
            state: slot.state,
            elapsed: slot.elapsed,
        })
    }

    /// Whether `handle` currently resolves to a physical (audible) voice.
    #[must_use]
    #[inline]
    pub fn is_physical(&self, handle: VoiceHandle) -> bool {
        matches!(
            self.resolve(handle).map(|i| self.slots[i].state),
            Some(VoiceState::Physical)
        )
    }

    /// Whether `handle` currently resolves to a virtual voice.
    #[must_use]
    #[inline]
    pub fn is_virtual(&self, handle: VoiceHandle) -> bool {
        matches!(
            self.resolve(handle).map(|i| self.slots[i].state),
            Some(VoiceState::Virtual)
        )
    }

    /// Number of physical (audible) voices.
    #[must_use]
    pub fn physical_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s.state, VoiceState::Physical))
            .count()
    }

    /// Number of virtual voices.
    #[must_use]
    pub fn virtual_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s.state, VoiceState::Virtual))
            .count()
    }

    /// Number of live (physical + virtual) voices.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_live()).count()
    }

    /// Number of free slots available for [`allocate`](Self::allocate).
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s.state, VoiceState::Free))
            .count()
    }

    /// Number of live voices belonging to `group`.
    #[must_use]
    pub fn group_count(&self, group: VoiceGroup) -> usize {
        self.slots
            .iter()
            .filter(|s| s.is_live() && s.group == group)
            .count()
    }

    /// Iterates handles of every physical voice, e.g. to drive rendering. The
    /// iterator borrows the pool and allocates nothing.
    pub fn physical_handles(&self) -> impl Iterator<Item = VoiceHandle> + '_ {
        self.handles_in_state(VoiceState::Physical)
    }

    /// Iterates handles of every virtual voice.
    pub fn virtual_handles(&self) -> impl Iterator<Item = VoiceHandle> + '_ {
        self.handles_in_state(VoiceState::Virtual)
    }

    /// Frees every slot and resets the allocation sequence.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            if slot.is_live() {
                slot.state = VoiceState::Free;
                slot.generation = slot.generation.wrapping_add(1);
            }
        }
        self.next_seq = 0;
    }

    // --- internal helpers -------------------------------------------------

    /// Shared body for the physical/virtual handle iterators.
    fn handles_in_state(&self, state: VoiceState) -> impl Iterator<Item = VoiceHandle> + '_ {
        self.slots.iter().enumerate().filter_map(move |(i, s)| {
            if s.state == state {
                Some(VoiceHandle {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "i is a slot position < capacity, which fits in u32 in practice"
                    )]
                    index: i as u32,
                    generation: s.generation,
                })
            } else {
                None
            }
        })
    }

    /// Resolves a handle to a live slot index, honoring the generation stamp.
    fn resolve(&self, handle: VoiceHandle) -> Option<usize> {
        let index = handle.index as usize;
        let slot = self.slots.get(index)?;
        if slot.is_live() && slot.generation == handle.generation {
            Some(index)
        } else {
            None
        }
    }

    /// Frees slot `index`, bumping its generation to invalidate old handles.
    fn free_slot(&mut self, index: usize) {
        let slot = &mut self.slots[index];
        slot.state = VoiceState::Free;
        slot.generation = slot.generation.wrapping_add(1);
    }

    /// Culls physical slot `index` into virtual state, or frees it when its
    /// behavior is [`Kill`](VirtualBehavior::Kill).
    fn cull(&mut self, index: usize) {
        if self.slots[index].behavior.keeps_slot() {
            self.slots[index].state = VoiceState::Virtual;
        } else {
            self.free_slot(index);
        }
    }

    /// Promotes virtual slot `index` to physical, resuming the playhead per its
    /// behavior.
    fn promote(&mut self, index: usize) {
        let slot = &mut self.slots[index];
        match slot.behavior {
            VirtualBehavior::RestartFromBeginning => {
                slot.elapsed = 0;
                slot.virtual_frames = 0;
            }
            VirtualBehavior::PlayFromElapsedTime => {
                slot.elapsed = slot.elapsed.saturating_add(slot.virtual_frames);
                slot.virtual_frames = 0;
            }
            VirtualBehavior::ContinueVirtual | VirtualBehavior::Kill => {}
        }
        slot.state = VoiceState::Physical;
    }

    /// Demotes weakest physical voices until the physical budget is met.
    fn enforce_physical_limit(&mut self) {
        while self.physical_count() > self.max_physical {
            let Some(victim) = self.weakest_physical_index() else {
                break;
            };
            self.cull(victim);
        }
    }

    /// Looks up the configured limit for `group`, if any.
    fn group_limit(&self, group: VoiceGroup) -> Option<GroupLimit> {
        self.limits.iter().copied().find(|l| l.group == group)
    }

    /// First free slot index, if any.
    fn free_index(&self) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| matches!(s.state, VoiceState::Free))
    }

    /// Index of the globally least-important live voice.
    fn weakest_index(&self) -> Option<usize> {
        self.min_by_importance(Slot::is_live)
    }

    /// Index of the least-important physical voice.
    fn weakest_physical_index(&self) -> Option<usize> {
        self.min_by_importance(|s| matches!(s.state, VoiceState::Physical))
    }

    /// Index of the least-important live voice in `group`.
    fn weakest_in_group(&self, group: VoiceGroup) -> Option<usize> {
        self.min_by_importance(|s| s.is_live() && s.group == group)
    }

    /// Index of the oldest (lowest sequence) live voice in `group`.
    fn oldest_in_group(&self, group: VoiceGroup) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.is_live() && slot.group == group {
                match best {
                    Some(b) if self.slots[b].seq <= slot.seq => {}
                    _ => best = Some(i),
                }
            }
        }
        best
    }

    /// Index of the most-important virtual voice above `threshold`, if any.
    fn best_virtual_above(&self, threshold: Importance) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            if matches!(slot.state, VoiceState::Virtual) && slot.importance > threshold {
                match best {
                    Some(b) if self.slots[b].importance >= slot.importance => {}
                    _ => best = Some(i),
                }
            }
        }
        best
    }

    /// Index of the minimum-importance slot matching `pred`, breaking ties by
    /// lower sequence (older) so eviction is deterministic.
    fn min_by_importance(&self, pred: impl Fn(&Slot) -> bool) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            if !pred(slot) {
                continue;
            }
            match best {
                Some(b) => {
                    let cur = &self.slots[b];
                    if slot.importance < cur.importance
                        || (slot.importance == cur.importance && slot.seq < cur.seq)
                    {
                        best = Some(i);
                    }
                }
                None => best = Some(i),
            }
        }
        best
    }
}

/// Replaces non-finite importance values with zero so ordering never sees NaN.
#[inline]
fn sanitize(v: Importance) -> Importance {
    if v.is_finite() { v } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: VoiceGroup = VoiceGroup(1);

    #[test]
    fn allocate_until_full_then_reject_weaker() {
        let mut pool = VoicePool::new(2, 2);
        let a = pool.allocate(VoiceRequest::new(G, 0.5)).unwrap();
        let b = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        assert_eq!(pool.active_count(), 2);
        // Weaker newcomer than the weakest resident (0.5) is refused.
        assert!(pool.allocate(VoiceRequest::new(G, 0.4)).is_none());
        assert!(pool.is_physical(a));
        assert!(pool.is_physical(b));
    }

    #[test]
    fn steal_weakest_when_full() {
        let mut pool = VoicePool::new(2, 2);
        let weak = pool.allocate(VoiceRequest::new(G, 0.2)).unwrap();
        let _strong = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        // Stronger newcomer steals the 0.2 voice's slot.
        let newcomer = pool.allocate(VoiceRequest::new(G, 0.5)).unwrap();
        assert!(pool.get(weak).is_none(), "stolen handle must go stale");
        assert!(pool.is_physical(newcomer));
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn physical_budget_virtualizes_weakest() {
        let mut pool = VoicePool::new(4, 2);
        let a = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        let b = pool.allocate(VoiceRequest::new(G, 0.3)).unwrap();
        let c = pool.allocate(VoiceRequest::new(G, 0.7)).unwrap();
        assert_eq!(pool.physical_count(), 2);
        assert_eq!(pool.virtual_count(), 1);
        assert!(pool.is_virtual(b), "0.3 is weakest and should be virtual");
        assert!(pool.is_physical(a));
        assert!(pool.is_physical(c));
    }

    #[test]
    fn kill_behavior_frees_slot_on_cull() {
        let mut pool = VoicePool::new(4, 1);
        let killed = pool
            .allocate(VoiceRequest {
                behavior: VirtualBehavior::Kill,
                ..VoiceRequest::new(G, 0.2)
            })
            .unwrap();
        let _strong = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        // The Kill voice was over budget and should be gone (not virtual).
        assert!(pool.get(killed).is_none());
        assert_eq!(pool.virtual_count(), 0);
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn continue_virtual_keeps_advancing_playhead() {
        let mut pool = VoicePool::new(4, 1);
        let looped = pool.allocate(VoiceRequest::new(G, 0.2)).unwrap(); // ContinueVirtual
        let _strong = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        assert!(pool.is_virtual(looped));
        pool.advance(1000);
        assert_eq!(pool.get(looped).unwrap().elapsed, 1000);
    }

    #[test]
    fn play_from_elapsed_fast_forwards_on_revive() {
        let mut pool = VoicePool::new(4, 1);
        let v = pool
            .allocate(VoiceRequest {
                behavior: VirtualBehavior::PlayFromElapsedTime,
                ..VoiceRequest::new(G, 0.2)
            })
            .unwrap();
        let strong = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        assert!(pool.is_virtual(v));
        pool.advance(500); // accrued while virtual, playhead frozen
        assert_eq!(pool.get(v).unwrap().elapsed, 0);
        pool.release(strong);
        assert!(pool.revive(v));
        assert_eq!(pool.get(v).unwrap().elapsed, 500);
    }

    #[test]
    fn restart_from_beginning_resets_on_revive() {
        let mut pool = VoicePool::new(4, 2);
        let v = pool
            .allocate(VoiceRequest {
                behavior: VirtualBehavior::RestartFromBeginning,
                ..VoiceRequest::new(G, 0.8)
            })
            .unwrap();
        pool.advance(300);
        assert_eq!(pool.get(v).unwrap().elapsed, 300);
        assert!(pool.virtualize(v));
        pool.advance(1000); // frozen while virtual
        assert!(pool.revive(v));
        assert_eq!(pool.get(v).unwrap().elapsed, 0);
    }

    #[test]
    fn group_limit_reject_newest() {
        let mut pool = VoicePool::new(8, 8);
        pool.set_group_limit(G, 2, LimitPolicy::RejectNewest);
        assert!(pool.allocate(VoiceRequest::new(G, 0.5)).is_some());
        assert!(pool.allocate(VoiceRequest::new(G, 0.5)).is_some());
        assert!(pool.allocate(VoiceRequest::new(G, 0.9)).is_none());
        assert_eq!(pool.group_count(G), 2);
    }

    #[test]
    fn group_limit_replace_oldest() {
        let mut pool = VoicePool::new(8, 8);
        pool.set_group_limit(G, 2, LimitPolicy::ReplaceOldest);
        let first = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        let _second = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        let _third = pool.allocate(VoiceRequest::new(G, 0.1)).unwrap();
        // Oldest (first) is replaced even though it was the most important.
        assert!(pool.get(first).is_none());
        assert_eq!(pool.group_count(G), 2);
    }

    #[test]
    fn group_limit_replace_weakest_respects_importance() {
        let mut pool = VoicePool::new(8, 8);
        pool.set_group_limit(G, 2, LimitPolicy::ReplaceWeakest);
        let weak = pool.allocate(VoiceRequest::new(G, 0.2)).unwrap();
        let _strong = pool.allocate(VoiceRequest::new(G, 0.8)).unwrap();
        // A louder newcomer replaces the weakest.
        assert!(pool.allocate(VoiceRequest::new(G, 0.5)).is_some());
        assert!(pool.get(weak).is_none());
        // A quieter newcomer is refused.
        assert!(pool.allocate(VoiceRequest::new(G, 0.1)).is_none());
    }

    #[test]
    fn revoice_hysteresis_demotes_and_promotes() {
        let mut pool = VoicePool::new(4, 1);
        let a = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        let b = pool.allocate(VoiceRequest::new(G, 0.4)).unwrap();
        assert!(pool.is_physical(a) && pool.is_virtual(b));
        // a fades below `leave`, b rises above `enter`.
        pool.set_importance(a, 0.1);
        pool.set_importance(b, 0.8);
        pool.revoice(VirtualizationThresholds::new(0.6, 0.3));
        assert!(pool.is_virtual(a), "faded voice should be demoted");
        assert!(pool.is_physical(b), "risen voice should be promoted");
    }

    #[test]
    fn stale_handle_is_rejected() {
        let mut pool = VoicePool::new(1, 1);
        let a = pool.allocate(VoiceRequest::new(G, 0.5)).unwrap();
        assert!(pool.release(a));
        assert!(!pool.release(a), "double release is a no-op");
        assert!(pool.get(a).is_none());
        // Slot reuse yields a fresh generation, so the old handle stays stale.
        let b = pool.allocate(VoiceRequest::new(G, 0.5)).unwrap();
        assert_ne!(a.generation(), b.generation());
        assert!(pool.get(a).is_none());
        assert!(pool.get(b).is_some());
    }

    #[test]
    fn non_finite_importance_is_sanitized() {
        let mut pool = VoicePool::new(2, 2);
        let nan = pool.allocate(VoiceRequest::new(G, Sample::NAN)).unwrap();
        assert_eq!(pool.get(nan).unwrap().importance, 0.0);
        // A normal voice outranks the sanitized (0.0) one when stealing.
        let _strong = pool.allocate(VoiceRequest::new(G, 0.5)).unwrap();
        let stealer = pool.allocate(VoiceRequest::new(G, 0.1)).unwrap();
        assert!(pool.get(nan).is_none(), "0.0 voice should be stolen first");
        assert!(pool.is_physical(stealer));
    }

    #[test]
    fn set_max_physical_reenforces_budget() {
        let mut pool = VoicePool::new(4, 4);
        let a = pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        let b = pool.allocate(VoiceRequest::new(G, 0.5)).unwrap();
        let c = pool.allocate(VoiceRequest::new(G, 0.2)).unwrap();
        assert_eq!(pool.physical_count(), 3);
        pool.set_max_physical(1);
        assert_eq!(pool.physical_count(), 1);
        assert!(pool.is_physical(a));
        assert!(pool.is_virtual(b));
        assert!(pool.is_virtual(c));
    }

    #[test]
    fn physical_handles_iterate_only_audible() {
        let mut pool = VoicePool::new(4, 2);
        pool.allocate(VoiceRequest::new(G, 0.9)).unwrap();
        pool.allocate(VoiceRequest::new(G, 0.8)).unwrap();
        pool.allocate(VoiceRequest::new(G, 0.1)).unwrap();
        let physical: Vec<_> = pool.physical_handles().collect();
        assert_eq!(physical.len(), 2);
        assert!(physical.iter().all(|&h| pool.is_physical(h)));
        assert_eq!(pool.virtual_handles().count(), 1);
    }
}
