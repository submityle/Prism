//! Change-tick and dirty tracking for local transforms.
//!
//! `Local` is authoritative and [`crate::GlobalTransform`] is only a cache, so
//! a propagation pass must know *which* locals changed since it last ran. This
//! module provides that bookkeeping without committing to a traversal strategy:
//!
//! - M1 performs a full pass every frame, but still calls [`ChangeTicks::mark`]
//!   on every edit and [`ChangeTicks::end_pass`] after each pass so the dirty
//!   state is exercised and correct.
//! - M2 can read [`ChangeTicks::is_changed`] per node and, together with
//!   [`crate::hierarchy::Hierarchy::children`], collect the dirty subtrees and
//!   skip clean ones — no API change required.
//!
//! The model is a single monotonically increasing [`Tick`] counter. Each node
//! records the tick at which its local last changed; a node is "changed since
//! the last pass" when that recorded tick is greater than the tick captured by
//! the previous [`ChangeTicks::end_pass`].

use alloc::vec::Vec;

use crate::hierarchy::NodeId;

/// A monotonically increasing logical timestamp.
///
/// Ticks are compared by value; a larger tick happened later. The counter is
/// 64-bit, so for any realistic session it never wraps.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Tick(u64);

impl Tick {
    /// The earliest tick, ordered before every change.
    pub const ZERO: Tick = Tick(0);

    /// The raw counter value.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Per-node change timestamps plus the pass bookkeeping that turns them into a
/// boolean "dirty since last pass" signal.
///
/// A [`ChangeTicks`] has one slot per node and must be kept the same length as
/// its [`crate::hierarchy::Hierarchy`]; [`ChangeTicks::push`] adds a slot for a
/// newly spawned node.
#[derive(Clone, Debug)]
pub struct ChangeTicks {
    changed: Vec<Tick>,
    current: Tick,
    last_pass: Tick,
}

impl Default for ChangeTicks {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ChangeTicks {
    /// Create an empty tracker. The current tick starts at `1` and the last
    /// pass at [`Tick::ZERO`], so any node added before the first pass is
    /// reported as changed (its world transform has never been computed).
    #[inline]
    pub const fn new() -> Self {
        Self { changed: Vec::new(), current: Tick(1), last_pass: Tick::ZERO }
    }

    /// Number of tracked nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.changed.len()
    }

    /// Whether no nodes are tracked.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty()
    }

    /// The current tick that edits are stamped with.
    #[inline]
    pub fn current_tick(&self) -> Tick {
        self.current
    }

    /// The tick captured by the most recent [`ChangeTicks::end_pass`].
    #[inline]
    pub fn last_pass(&self) -> Tick {
        self.last_pass
    }

    /// Add a slot for a newly spawned node, marked changed at the current tick
    /// so the next pass computes its world transform.
    #[inline]
    pub fn push(&mut self) {
        self.changed.push(self.current);
    }

    /// Stamp `node` as changed at the current tick.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn mark(&mut self, node: NodeId) {
        self.changed[node.index()] = self.current;
    }

    /// The tick at which `node`'s local last changed.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn changed_tick(&self, node: NodeId) -> Tick {
        self.changed[node.index()]
    }

    /// Whether `node`'s local changed since the last completed pass.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn is_changed(&self, node: NodeId) -> bool {
        self.changed[node.index()] > self.last_pass
    }

    /// Record that a propagation pass just finished: everything stamped up to
    /// now becomes "clean", and the current tick advances so subsequent edits
    /// are stamped strictly after this pass and are therefore detected.
    #[inline]
    pub fn end_pass(&mut self) {
        self.last_pass = self.current;
        self.current = Tick(self.current.0 + 1);
    }
}
