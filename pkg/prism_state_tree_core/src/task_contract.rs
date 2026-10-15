//! Public, host-neutral contracts for StateTree tasks and task hand-off.
//!
//! The contracts in this module contain only stable scalar identifiers and
//! `alloc` collections. They do not depend on ECS entities, clocks, gameplay
//! types, renderer state, or a particular scheduler. A host may therefore
//! compile them into a task plan, use them in a replay record, or translate
//! them into a backend-specific command without changing their semantics.

extern crate alloc;

use alloc::{collections::VecDeque, vec::Vec};

/// The result reported by a task lifecycle callback.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Outcome {
    /// The task completed its work successfully.
    Success,
    /// The task completed its work unsuccessfully.
    Failure,
    /// The task remains active and may be evaluated again according to its
    /// [`Cadence`].
    Running,
}

impl Outcome {
    /// Returns `true` when the task has reached a terminal result.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }

    /// Returns `true` only for [`Outcome::Success`].
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns `true` only for [`Outcome::Failure`].
    #[must_use]
    pub const fn is_failure(self) -> bool {
        matches!(self, Self::Failure)
    }

    /// Returns `true` only for [`Outcome::Running`].
    #[must_use]
    pub const fn is_running(self) -> bool {
        matches!(self, Self::Running)
    }
}

/// The fixed-step trigger policy used to schedule a task evaluation.
///
/// `EveryNFrames(n)` is evaluated at fixed steps `0`, `n`, `2n`, and so on.
/// A value of zero is representable because this enum is a public data
/// contract; [`Cadence::validate`] rejects it and [`Cadence::is_due`] treats
/// it as disabled rather than allowing a divide-by-zero path.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Cadence {
    /// The task is not ticked by cadence scheduling.
    Never,
    /// The task is eligible on every fixed step.
    EveryStep,
    /// The task is eligible every `n` fixed steps, including step zero.
    EveryNFrames(u32),
    /// The task is eligible when the declared dependency changes.
    OnChange(u32),
    /// The task is eligible when the supplied event kind is observed.
    OnEvent(u32),
}

impl Cadence {
    /// Alias for [`Cadence::EveryStep`] using the terminology common in host
    /// frame schedulers.
    #[allow(non_upper_case_globals)]
    pub const EveryFrame: Self = Self::EveryStep;

    /// Creates an every-`n` cadence and rejects a zero interval.
    pub const fn every_n_frames(frames: u32) -> Result<Self, CadenceError> {
        if frames == 0 {
            Err(CadenceError::ZeroInterval)
        } else {
            Ok(Self::EveryNFrames(frames))
        }
    }

    /// Validates the cadence declaration.
    pub const fn validate(self) -> Result<(), CadenceError> {
        match self {
            Self::EveryNFrames(0) => Err(CadenceError::ZeroInterval),
            _ => Ok(()),
        }
    }

    /// Returns whether this cadence is eligible at one fixed-step sample.
    ///
    /// `changed_dependency` should contain the dependency key that changed at
    /// this sample, if any. `event_kind` should contain the event kind
    /// delivered at this sample, if any. The method is pure and does not
    /// mutate a scheduler cursor, so repeated calls with equal inputs always
    /// return equal results.
    #[must_use]
    pub const fn is_due(
        self,
        fixed_step: u64,
        changed_dependency: Option<u32>,
        event_kind: Option<u32>,
    ) -> bool {
        match self {
            Self::Never => false,
            Self::EveryStep => true,
            Self::EveryNFrames(frames) => frames != 0 && fixed_step.is_multiple_of(frames as u64),
            Self::OnChange(dependency) => match changed_dependency {
                Some(value) => value == dependency,
                None => false,
            },
            Self::OnEvent(event) => match event_kind {
                Some(value) => value == event,
                None => false,
            },
        }
    }
}

/// An invalid [`Cadence`] declaration.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CadenceError {
    /// An every-`n` cadence was given an interval of zero.
    ZeroInterval,
}

/// A deterministic read/write declaration for host resources.
///
/// Resource keys are opaque values owned by the host. Read keys are
/// read-only accesses. A write key is treated as both a read and a write for
/// conflict detection; the canonical representation keeps it only in the
/// write collection. Both collections are kept sorted and unique, making
/// equality, hashing by a caller, and pairwise conflict checks independent of
/// declaration order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AccessSet {
    reads: Vec<u32>,
    writes: Vec<u32>,
}

impl AccessSet {
    /// Creates an empty access declaration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Creates a canonical access declaration from read-only and write keys.
    ///
    /// Duplicate keys are collapsed. If a key appears in both inputs, the
    /// write declaration takes precedence.
    #[must_use]
    pub fn from_parts(reads: &[u32], writes: &[u32]) -> Self {
        let mut access = Self {
            reads: reads.to_vec(),
            writes: writes.to_vec(),
        };
        access.reads.sort_unstable();
        access.reads.dedup();
        access.writes.sort_unstable();
        access.writes.dedup();
        access
            .reads
            .retain(|key| access.writes.binary_search(key).is_err());
        access
    }

    /// Returns a declaration containing one read-only key.
    #[must_use]
    pub fn with_read(key: u32) -> Self {
        let mut access = Self::new();
        access.add_read(key);
        access
    }

    /// Returns a declaration containing one write key.
    #[must_use]
    pub fn with_write(key: u32) -> Self {
        let mut access = Self::new();
        access.add_write(key);
        access
    }

    /// Adds a read-only key, returning `true` when the declaration changed.
    ///
    /// A key already declared as a write is already covered and therefore does
    /// not create a second entry.
    pub fn add_read(&mut self, key: u32) -> bool {
        if self.writes.binary_search(&key).is_ok() {
            return false;
        }
        insert_sorted_unique(&mut self.reads, key)
    }

    /// Adds a write key, returning `true` when the declaration changed.
    ///
    /// A prior read-only declaration of the same key is replaced by the
    /// stronger write declaration.
    pub fn add_write(&mut self, key: u32) -> bool {
        let removed_read = remove_sorted(&mut self.reads, key);
        let inserted_write = insert_sorted_unique(&mut self.writes, key);
        removed_read || inserted_write
    }

    /// Returns the sorted read-only keys.
    #[must_use]
    pub fn reads(&self) -> &[u32] {
        &self.reads
    }

    /// Returns the sorted write keys.
    #[must_use]
    pub fn writes(&self) -> &[u32] {
        &self.writes
    }

    /// Returns whether this declaration has no resources.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty()
    }

    /// Returns the total number of canonical resource keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.reads.len() + self.writes.len()
    }

    /// Returns whether a key is declared as read-only.
    #[must_use]
    pub fn contains_read(&self, key: u32) -> bool {
        self.reads.binary_search(&key).is_ok()
    }

    /// Returns whether a key is declared as written.
    #[must_use]
    pub fn contains_write(&self, key: u32) -> bool {
        self.writes.binary_search(&key).is_ok()
    }

    /// Returns whether two declarations cannot execute concurrently.
    #[must_use]
    pub fn conflicts_with(&self, other: &Self) -> bool {
        intersects(&self.writes, &other.reads)
            || intersects(&self.writes, &other.writes)
            || intersects(&self.reads, &other.writes)
    }

    /// Returns whether two declarations can execute concurrently.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        !self.conflicts_with(other)
    }

    /// Adds every access from another declaration in canonical order.
    pub fn extend(&mut self, other: &Self) {
        for &key in &other.reads {
            self.add_read(key);
        }
        for &key in &other.writes {
            self.add_write(key);
        }
    }
}

impl Default for AccessSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Replay and prediction classification for a task.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ReplaySafety {
    /// The task is pure and has no external side effect.
    Pure,
    /// The task is replayable when its effects use the host's deterministic
    /// deferred command path.
    ReplaySafe,
    /// The task depends on authority-owned or otherwise transient results and
    /// is not eligible for a prediction replay path.
    AuthorityOnly,
}

impl ReplaySafety {
    /// Returns the less permissive classification of two task requirements.
    #[must_use]
    pub const fn combine(self, other: Self) -> Self {
        if self as u8 >= other as u8 {
            self
        } else {
            other
        }
    }

    /// Returns whether this classification can participate in replay.
    #[must_use]
    pub const fn is_replayable(self) -> bool {
        !matches!(self, Self::AuthorityOnly)
    }

    /// Returns whether this classification can participate in client
    /// prediction according to the contract alone.
    #[must_use]
    pub const fn allows_prediction(self) -> bool {
        self.is_replayable()
    }
}

/// A structured reason supplied to a task abort callback.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AbortReason {
    /// A selected state transition replaced the task's owning state.
    Transition,
    /// A higher-priority request superseded the task.
    HigherPriority,
    /// A parent state exited and took the task with it.
    ParentExit,
    /// The host explicitly cancelled the task.
    Cancelled,
    /// The task's plan revision changed and its instance data was not kept.
    PlanRevisionChanged,
    /// The host is shutting down the task scheduler.
    Shutdown,
    /// A host-defined reason identified by a stable numeric code.
    External(u32),
}

/// A totally ordered interruption level.
///
/// Larger levels represent stronger interruption authority. The common names
/// are associated constants so a host can use `InterruptTier::Critical` while
/// still creating additional levels with [`InterruptTier::custom`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct InterruptTier(u8);

impl InterruptTier {
    /// No interruption authority.
    pub const NONE: Self = Self(0);
    /// Low interruption authority.
    pub const LOW: Self = Self(1);
    /// Ordinary interruption authority.
    pub const NORMAL: Self = Self(2);
    /// High interruption authority.
    pub const HIGH: Self = Self(3);
    /// Critical interruption authority.
    pub const CRITICAL: Self = Self(4);

    /// Camel-case alias for [`InterruptTier::NONE`].
    #[allow(non_upper_case_globals)]
    pub const None: Self = Self::NONE;
    /// Camel-case alias for [`InterruptTier::LOW`].
    #[allow(non_upper_case_globals)]
    pub const Low: Self = Self::LOW;
    /// Camel-case alias for [`InterruptTier::NORMAL`].
    #[allow(non_upper_case_globals)]
    pub const Normal: Self = Self::NORMAL;
    /// Camel-case alias for [`InterruptTier::HIGH`].
    #[allow(non_upper_case_globals)]
    pub const High: Self = Self::HIGH;
    /// Camel-case alias for [`InterruptTier::CRITICAL`].
    #[allow(non_upper_case_globals)]
    pub const Critical: Self = Self::CRITICAL;

    /// Creates a host-defined interruption level.
    #[must_use]
    pub const fn custom(level: u8) -> Self {
        Self(level)
    }

    /// Returns the numeric ordering level.
    #[must_use]
    pub const fn level(self) -> u8 {
        self.0
    }

    /// Returns whether this level can supersede the supplied active level.
    ///
    /// Equal levels do not supersede one another; same-tier arbitration stays
    /// with the scheduler's ordinary priority and declaration ordering.
    #[must_use]
    pub const fn can_interrupt(self, active: Self) -> bool {
        self.0 > active.0
    }
}

/// Compact set of interruption levels used by a [`CancelWindow`].
///
/// Levels `0..=127` are representable. The fixed bitset keeps window checks
/// allocation-free and constant-time while leaving the level ordering open to
/// host-defined values within that range.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InterruptTierSet(u128);

impl InterruptTierSet {
    /// Creates an empty tier set.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Creates a set containing every representable tier.
    #[must_use]
    pub const fn all() -> Self {
        Self(u128::MAX)
    }

    /// Creates a set containing one tier, or returns an error for a level
    /// outside the compact bitset.
    pub const fn only(tier: InterruptTier) -> Result<Self, InterruptTierError> {
        match tier.0 {
            0..=127 => Ok(Self(1u128 << tier.0)),
            level => Err(InterruptTierError::Unrepresentable(level)),
        }
    }

    /// Creates a set from its raw bit representation.
    #[must_use]
    pub const fn from_bits(bits: u128) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    #[must_use]
    pub const fn bits(self) -> u128 {
        self.0
    }

    /// Returns whether the set contains a tier.
    #[must_use]
    pub const fn contains(self, tier: InterruptTier) -> bool {
        tier.0 < 128 && (self.0 & (1u128 << tier.0)) != 0
    }

    /// Adds a tier and returns an error when its level cannot be represented.
    pub fn insert(&mut self, tier: InterruptTier) -> Result<(), InterruptTierError> {
        if tier.0 >= 128 {
            return Err(InterruptTierError::Unrepresentable(tier.0));
        }
        self.0 |= 1u128 << tier.0;
        Ok(())
    }

    /// Removes a tier from the set.
    pub const fn remove(&mut self, tier: InterruptTier) {
        if tier.0 < 128 {
            self.0 &= !(1u128 << tier.0);
        }
    }
}

impl Default for InterruptTierSet {
    fn default() -> Self {
        Self::empty()
    }
}

/// Failure while constructing an [`InterruptTierSet`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InterruptTierError {
    /// The level is outside the compact 128-level representation.
    Unrepresentable(u8),
}

/// Whether an endpoint of a fixed-step [`CancelWindow`] is included.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WindowBound {
    /// The endpoint step belongs to the window.
    Inclusive,
    /// The endpoint step does not belong to the window.
    Exclusive,
}

/// A fixed-step cancellation interval with explicit endpoint semantics.
///
/// The interval is evaluated using integer fixed steps, never wall-clock
/// time. Its opening and closing endpoints can independently be inclusive or
/// exclusive, and the allowed interruption tiers are checked in the same
/// operation. This makes step `opens_at` and step `closes_at` directly
/// testable without floating-point rounding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CancelWindow {
    opens_at: u64,
    closes_at: u64,
    opening: WindowBound,
    closing: WindowBound,
    allowed_tiers: InterruptTierSet,
}

impl CancelWindow {
    /// Creates an inclusive window that accepts every representable tier.
    pub const fn new(opens_at: u64, closes_at: u64) -> Result<Self, CancelWindowError> {
        Self::with_bounds(
            opens_at,
            closes_at,
            WindowBound::Inclusive,
            WindowBound::Inclusive,
            InterruptTierSet::all(),
        )
    }

    /// Creates a window with explicit endpoint and tier semantics.
    pub const fn with_bounds(
        opens_at: u64,
        closes_at: u64,
        opening: WindowBound,
        closing: WindowBound,
        allowed_tiers: InterruptTierSet,
    ) -> Result<Self, CancelWindowError> {
        if opens_at > closes_at {
            return Err(CancelWindowError::Reversed {
                opens_at,
                closes_at,
            });
        }
        Ok(Self {
            opens_at,
            closes_at,
            opening,
            closing,
            allowed_tiers,
        })
    }

    /// Returns the opening fixed step.
    #[must_use]
    pub const fn opens_at(self) -> u64 {
        self.opens_at
    }

    /// Returns the closing fixed step.
    #[must_use]
    pub const fn closes_at(self) -> u64 {
        self.closes_at
    }

    /// Returns the opening endpoint rule.
    #[must_use]
    pub const fn opening(self) -> WindowBound {
        self.opening
    }

    /// Returns the closing endpoint rule.
    #[must_use]
    pub const fn closing(self) -> WindowBound {
        self.closing
    }

    /// Returns the interruption tiers accepted by the window.
    #[must_use]
    pub const fn allowed_tiers(self) -> InterruptTierSet {
        self.allowed_tiers
    }

    /// Returns whether the interval contains a fixed step, ignoring tier.
    #[must_use]
    pub const fn contains_step(self, step: u64) -> bool {
        let opens = match self.opening {
            WindowBound::Inclusive => step >= self.opens_at,
            WindowBound::Exclusive => step > self.opens_at,
        };
        let closes = match self.closing {
            WindowBound::Inclusive => step <= self.closes_at,
            WindowBound::Exclusive => step < self.closes_at,
        };
        opens && closes
    }

    /// Returns whether a tier may cancel at a fixed step.
    #[must_use]
    pub const fn contains(self, step: u64, tier: InterruptTier) -> bool {
        self.contains_step(step) && self.allowed_tiers.contains(tier)
    }

    /// Returns whether the interval has no step satisfying both endpoints.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        let first_step = match self.opening {
            WindowBound::Inclusive => self.opens_at,
            WindowBound::Exclusive => {
                if self.opens_at == u64::MAX {
                    return true;
                }
                self.opens_at + 1
            }
        };
        let last_step = match self.closing {
            WindowBound::Inclusive => self.closes_at,
            WindowBound::Exclusive => {
                if self.closes_at == 0 {
                    return true;
                }
                self.closes_at - 1
            }
        };
        first_step > last_step
    }
}

/// Failure while constructing a [`CancelWindow`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CancelWindowError {
    /// The opening step is later than the closing step.
    Reversed {
        /// Opening fixed step supplied to the constructor.
        opens_at: u64,
        /// Closing fixed step supplied to the constructor.
        closes_at: u64,
    },
}

/// A task request sent to the host-owned StateTree arbitration phase.
///
/// Numeric identifiers are deliberately opaque to the core. Their mapping to
/// states, subtree assets, payload storage, or host-defined commands belongs
/// to the caller. Each variant is fixed-size and therefore safe to retain in
/// a preallocated [`RequestQueue`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RequestKind {
    /// Ask the host to select a transition target identified by `target`.
    Transition(u32),
    /// Ask the host to push a subtree and pass an opaque payload handle.
    PushSubtree {
        /// Stable subtree or asset identity.
        subtree: u32,
        /// Host-owned payload identity.
        payload: u64,
    },
    /// Report successful completion of the task-owned segment.
    Complete,
    /// Report unsuccessful completion of the task-owned segment.
    Fail,
    /// Ask the host to cancel the task with a structured reason.
    Cancel {
        /// Reason presented to the task abort path.
        reason: AbortReason,
    },
    /// Carry a host-defined request code and opaque payload.
    Custom {
        /// Stable host-defined request code.
        kind: u32,
        /// Host-owned request payload.
        payload: u64,
    },
}

impl RequestKind {
    /// Returns whether this request reports a terminal task result.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Complete | Self::Fail | Self::Cancel { .. })
    }
}

/// Failure returned when inserting a request into a [`RequestQueue`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RequestQueueError {
    /// The queue was created with zero capacity.
    ZeroCapacity,
    /// The queue already contains its configured number of requests.
    Full,
}

/// A FIFO request queue with a fixed capacity and no growth during enqueue.
///
/// The constructor reserves the selected capacity up front. `push` checks the
/// logical capacity before touching the underlying deque, so a successful
/// enqueue does not allocate after construction. Requests retain insertion
/// order and are never implicitly coalesced or reordered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestQueue {
    capacity: usize,
    requests: VecDeque<RequestKind>,
}

impl RequestQueue {
    /// Creates an empty queue with the selected maximum number of requests.
    ///
    /// A zero capacity is representable so configuration can be propagated
    /// without a panic; every insertion into it returns
    /// [`RequestQueueError::ZeroCapacity`].
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            requests: VecDeque::with_capacity(capacity),
        }
    }

    /// Returns the configured maximum number of requests.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of retained requests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Returns whether no requests are retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Returns whether the queue has reached its configured capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.capacity == 0 || self.requests.len() == self.capacity
    }

    /// Appends one request without allocating or reordering existing entries.
    pub fn push(&mut self, request: RequestKind) -> Result<(), RequestQueueError> {
        if self.capacity == 0 {
            return Err(RequestQueueError::ZeroCapacity);
        }
        if self.requests.len() == self.capacity {
            return Err(RequestQueueError::Full);
        }
        self.requests.push_back(request);
        Ok(())
    }

    /// Alias for [`RequestQueue::push`] that reads naturally at a task call
    /// site.
    pub fn enqueue(&mut self, request: RequestKind) -> Result<(), RequestQueueError> {
        self.push(request)
    }

    /// Returns the oldest request without removing it.
    #[must_use]
    pub fn front(&self) -> Option<&RequestKind> {
        self.requests.front()
    }

    /// Removes and returns the oldest request.
    pub fn pop(&mut self) -> Option<RequestKind> {
        self.requests.pop_front()
    }

    /// Returns requests in FIFO order.
    pub fn iter(&self) -> impl Iterator<Item = &RequestKind> {
        self.requests.iter()
    }

    /// Removes every retained request without changing the configured
    /// capacity.
    pub fn clear(&mut self) {
        self.requests.clear();
    }
}

fn insert_sorted_unique(values: &mut Vec<u32>, value: u32) -> bool {
    match values.binary_search(&value) {
        Ok(_) => false,
        Err(index) => {
            values.insert(index, value);
            true
        }
    }
}

fn remove_sorted(values: &mut Vec<u32>, value: u32) -> bool {
    let Ok(index) = values.binary_search(&value) else {
        return false;
    };
    values.remove(index);
    true
}

fn intersects(left: &[u32], right: &[u32]) -> bool {
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            core::cmp::Ordering::Less => left_index += 1,
            core::cmp::Ordering::Greater => right_index += 1,
            core::cmp::Ordering::Equal => return true,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_helpers_are_exhaustive() {
        assert!(Outcome::Success.is_terminal());
        assert!(Outcome::Failure.is_terminal());
        assert!(!Outcome::Running.is_terminal());
        assert!(Outcome::Success.is_success());
        assert!(Outcome::Failure.is_failure());
        assert!(Outcome::Running.is_running());
    }

    #[test]
    fn cadence_is_fixed_step_and_deterministic() {
        assert_eq!(Cadence::EveryFrame, Cadence::EveryStep);
        assert_eq!(Cadence::every_n_frames(0), Err(CadenceError::ZeroInterval));
        let cadence = Cadence::every_n_frames(3).unwrap();
        assert!(!cadence.is_due(1, None, None));
        assert!(cadence.is_due(0, None, None));
        assert!(cadence.is_due(3, None, None));
        assert!(!cadence.is_due(4, None, None));
        assert_eq!(
            cadence.is_due(12, None, None),
            cadence.is_due(12, None, None)
        );
        assert!(Cadence::OnChange(7).is_due(99, Some(7), None));
        assert!(!Cadence::OnChange(7).is_due(99, Some(8), None));
        assert!(Cadence::OnEvent(5).is_due(99, None, Some(5)));
        assert!(!Cadence::Never.is_due(0, Some(7), Some(5)));
    }

    #[test]
    fn access_sets_are_canonical_and_conflicts_are_symmetric() {
        let first = AccessSet::from_parts(&[9, 2, 2], &[7, 4, 7, 2]);
        let second = AccessSet::from_parts(&[2, 9], &[4, 7]);
        assert_eq!(first.reads(), &[9]);
        assert_eq!(first.writes(), &[2, 4, 7]);
        assert_eq!(first, AccessSet::from_parts(&[9], &[7, 2, 4]));
        assert!(first.conflicts_with(&second));
        assert_eq!(first.conflicts_with(&second), second.conflicts_with(&first));

        let reads_a = AccessSet::with_read(11);
        let reads_b = AccessSet::with_read(11);
        assert!(reads_a.is_compatible_with(&reads_b));

        let mut merged = AccessSet::with_read(3);
        merged.extend(&AccessSet::with_write(1));
        merged.extend(&AccessSet::with_write(3));
        assert_eq!(merged.reads(), &[]);
        assert_eq!(merged.writes(), &[1, 3]);
    }

    #[test]
    fn replay_safety_combines_to_the_strongest_requirement() {
        assert_eq!(
            ReplaySafety::Pure.combine(ReplaySafety::ReplaySafe),
            ReplaySafety::ReplaySafe
        );
        assert_eq!(
            ReplaySafety::ReplaySafe.combine(ReplaySafety::AuthorityOnly),
            ReplaySafety::AuthorityOnly
        );
        assert!(ReplaySafety::Pure.allows_prediction());
        assert!(!ReplaySafety::AuthorityOnly.is_replayable());
    }

    #[test]
    fn interruption_levels_are_ordered_and_extendable() {
        assert!(InterruptTier::Critical.can_interrupt(InterruptTier::Normal));
        assert!(!InterruptTier::Normal.can_interrupt(InterruptTier::Normal));
        let custom = InterruptTier::custom(17);
        assert_eq!(custom.level(), 17);
        assert!(custom.can_interrupt(InterruptTier::Critical));

        let only = InterruptTierSet::only(InterruptTier::Critical).unwrap();
        assert!(only.contains(InterruptTier::Critical));
        assert!(!only.contains(InterruptTier::Normal));
        assert_eq!(
            InterruptTierSet::only(InterruptTier::custom(255)),
            Err(InterruptTierError::Unrepresentable(255))
        );
    }

    #[test]
    fn cancel_window_checks_all_open_closed_endpoints() {
        let inclusive = CancelWindow::new(4, 6).unwrap();
        assert!(!inclusive.contains_step(3));
        assert!(inclusive.contains_step(4));
        assert!(inclusive.contains_step(6));
        assert!(!inclusive.contains_step(7));

        let left_open = CancelWindow::with_bounds(
            4,
            6,
            WindowBound::Exclusive,
            WindowBound::Inclusive,
            InterruptTierSet::all(),
        )
        .unwrap();
        assert!(!left_open.contains_step(4));
        assert!(left_open.contains_step(5));
        assert!(left_open.contains_step(6));

        let right_open = CancelWindow::with_bounds(
            4,
            6,
            WindowBound::Inclusive,
            WindowBound::Exclusive,
            InterruptTierSet::all(),
        )
        .unwrap();
        assert!(right_open.contains_step(4));
        assert!(right_open.contains_step(5));
        assert!(!right_open.contains_step(6));

        let both_open = CancelWindow::with_bounds(
            4,
            6,
            WindowBound::Exclusive,
            WindowBound::Exclusive,
            InterruptTierSet::only(InterruptTier::Critical).unwrap(),
        )
        .unwrap();
        assert!(!both_open.contains(4, InterruptTier::Critical));
        assert!(both_open.contains(5, InterruptTier::Critical));
        assert!(!both_open.contains(5, InterruptTier::Normal));
        assert!(!both_open.contains(6, InterruptTier::Critical));
        assert_eq!(
            CancelWindow::new(7, 6),
            Err(CancelWindowError::Reversed {
                opens_at: 7,
                closes_at: 6
            })
        );

        let point = CancelWindow::new(8, 8).unwrap();
        assert!(point.contains_step(8));
        assert!(!point.is_empty());
        let empty = CancelWindow::with_bounds(
            8,
            8,
            WindowBound::Exclusive,
            WindowBound::Inclusive,
            InterruptTierSet::all(),
        )
        .unwrap();
        assert!(empty.is_empty());
        assert!(!empty.contains_step(8));

        let adjacent = CancelWindow::with_bounds(
            4,
            5,
            WindowBound::Exclusive,
            WindowBound::Exclusive,
            InterruptTierSet::all(),
        )
        .unwrap();
        assert!(adjacent.is_empty());
    }

    #[test]
    fn request_queue_is_bounded_fifo_and_deterministic() {
        let mut zero = RequestQueue::new(0);
        assert!(zero.is_empty());
        assert!(zero.is_full());
        assert_eq!(
            zero.push(RequestKind::Complete),
            Err(RequestQueueError::ZeroCapacity)
        );

        let mut queue = RequestQueue::new(3);
        let requests = [
            RequestKind::Transition(4),
            RequestKind::Complete,
            RequestKind::Custom {
                kind: 9,
                payload: 11,
            },
        ];
        for request in requests {
            queue.push(request).unwrap();
        }
        assert!(queue.is_full());
        assert_eq!(queue.front(), Some(&RequestKind::Transition(4)));
        assert_eq!(queue.push(RequestKind::Fail), Err(RequestQueueError::Full));
        assert_eq!(queue.pop(), Some(RequestKind::Transition(4)));
        assert_eq!(queue.pop(), Some(RequestKind::Complete));
        assert_eq!(
            queue.pop(),
            Some(RequestKind::Custom {
                kind: 9,
                payload: 11
            })
        );
        assert_eq!(queue.pop(), None);
        assert!(RequestKind::Cancel {
            reason: AbortReason::Cancelled
        }
        .is_terminal());
        assert!(!RequestKind::Transition(4).is_terminal());
    }
}
