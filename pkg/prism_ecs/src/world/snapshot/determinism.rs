//! Dual-run determinism auditing (design §14 "逐帧状态哈希去同步" / §24.4
//! "审计双跑定位首个发散 tick").
//!
//! Rollback / lockstep networking (Quantum / GGPO / Overwatch form) rests on a
//! single promise: **same input → bit-equivalent output**. When two peers (or
//! two local replays) drift apart the symptom is a mismatched per-frame
//! [`state_hash`](super::WorldSnapshot::state_hash), but the raw hash only says
//! *that* they diverged, never *when* or *where*. Shipping a deterministic
//! simulation requires turning that single mismatched bit into an actionable
//! coordinate, and that debugging capability is exactly what the `determinism`
//! feature tier adds on top of the always-on hashing primitive.
//!
//! This module provides the two localization stages a determinism engineer
//! walks in order:
//!
//! 1. **Which tick?** [`FrameHashLog`] records a run's per-tick state hashes;
//!    [`FrameHashLog::first_divergence`] compares two logs and returns the
//!    first [`TickDivergence`] — the earliest frame at which the runs stop
//!    agreeing (differing hash, drifting tick cadence, or one run ending
//!    early). Comparing *hashes* keeps a desync audit cheap enough to run every
//!    frame of a soak test without storing full snapshots.
//! 2. **Which component on which entity?** Once the first bad tick is known,
//!    capture a [`WorldSnapshot`](super::WorldSnapshot) of each run at that tick
//!    and call [`locate_divergence`] (or
//!    [`WorldSnapshot::locate_divergence`](super::WorldSnapshot::locate_divergence)).
//!    It walks the two snapshots in the *same* deterministic fold order
//!    [`state_hash`](super::WorldSnapshot::state_hash) uses — tick cursors,
//!    allocator liveness, entity list ascending by [`Entity::to_bits`], then
//!    columns ascending by [`ComponentId`] and holders by row — and returns the
//!    first [`SnapshotDivergence`]: the precise structural or value coordinate
//!    where the two frames stop being byte-equal.
//!
//! [`locate_divergence`] is the localizing counterpart of
//! [`WorldSnapshot::structurally_eq`](super::WorldSnapshot::structurally_eq):
//! `structurally_eq` answers the yes/no question, this answers the *where*
//! question, over the identical comparison set (tick cursors, allocator state,
//! entity list, every column's holders/ticks/value bytes, and opt-in
//! resources). The two therefore agree exactly: `structurally_eq` returns
//! `true` iff `locate_divergence` returns [`None`].
//!
//! # Honesty boundary
//! This is a debugging / auditing tier, not the determinism guarantee itself —
//! that guarantee is produced by stable ordering (§8.4), stable allocation
//! (§14), and the deterministic fold in [`hash`](super::hash). The auditor
//! only *observes* divergence that already happened; it cannot prevent it.
//! Value-level localization is subject to the same honesty boundary as the
//! hash: it compares the *bytes* every captured column carries, so it detects
//! value drift for any snapshotted component regardless of hash registration
//! (unlike the hash, which only folds values for hash-registered components).

use alloc::vec::Vec;

use crate::component::ComponentId;
use crate::entity::Entity;

use super::WorldSnapshot;

/// One recorded frame of a run: the simulation tick and the world
/// [`state_hash`](WorldSnapshot::state_hash) taken at that tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHash {
    /// The simulation tick (frame index) this hash was captured at.
    pub tick: u64,
    /// The deterministic world state hash at [`tick`](Self::tick).
    pub hash: u64,
}

/// An ordered recording of a run's per-tick [`state_hash`](WorldSnapshot::state_hash)
/// values, the cheap first stage of a dual-run desync audit (design §24.4).
///
/// A log is append-only in simulation order: feed one frame hash per tick with
/// [`record`](Self::record) (or [`record_snapshot`](Self::record_snapshot)),
/// then compare two runs' logs with [`first_divergence`](Self::first_divergence)
/// to find the earliest frame they disagree on. Storing a `u64` per frame keeps
/// a full-match audit affordable over long soak runs where keeping every
/// [`WorldSnapshot`] would not be.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameHashLog {
    frames: Vec<FrameHash>,
}

impl FrameHashLog {
    /// Create an empty log.
    #[inline]
    pub const fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Create an empty log pre-sized for `capacity` frames.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frames: Vec::with_capacity(capacity),
        }
    }

    /// Append one frame: the world `hash` observed at simulation `tick`.
    #[inline]
    pub fn record(&mut self, tick: u64, hash: u64) {
        self.frames.push(FrameHash { tick, hash });
    }

    /// Append one frame by folding `snapshot`'s deterministic
    /// [`state_hash`](WorldSnapshot::state_hash) at simulation `tick`.
    #[inline]
    pub fn record_snapshot(&mut self, tick: u64, snapshot: &WorldSnapshot) {
        self.record(tick, snapshot.state_hash());
    }

    /// The recorded frames in simulation order.
    #[inline]
    pub fn frames(&self) -> &[FrameHash] {
        &self.frames
    }

    /// The number of recorded frames.
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no frames have been recorded.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Compare this log (`left`) against `other` (`right`) frame-by-frame and
    /// return the first point they diverge, or [`None`] if every shared frame
    /// agrees and both runs are the same length.
    ///
    /// Comparison is positional (frame index), mirroring two lockstep peers
    /// that step the same tick cadence: at each index a differing *tick*
    /// ([`TickDivergence::Tick`]) is reported before a differing *hash*
    /// ([`TickDivergence::Hash`]), since a cadence drift makes the hash
    /// comparison meaningless. If one run recorded more frames than the other,
    /// the first unmatched index is reported as [`TickDivergence::Length`].
    pub fn first_divergence(&self, other: &FrameHashLog) -> Option<TickDivergence> {
        let shared = self.frames.len().min(other.frames.len());
        for index in 0..shared {
            let left = self.frames[index];
            let right = other.frames[index];
            if left.tick != right.tick {
                return Some(TickDivergence::Tick {
                    index,
                    left: left.tick,
                    right: right.tick,
                });
            }
            if left.hash != right.hash {
                return Some(TickDivergence::Hash {
                    tick: left.tick,
                    left: left.hash,
                    right: right.hash,
                });
            }
        }
        if self.frames.len() != other.frames.len() {
            return Some(TickDivergence::Length {
                index: shared,
                left_len: self.frames.len(),
                right_len: other.frames.len(),
            });
        }
        None
    }
}

/// The first frame at which two [`FrameHashLog`]s stop agreeing, produced by
/// [`FrameHashLog::first_divergence`]. `left` refers to the receiver log,
/// `right` to its argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TickDivergence {
    /// Both runs reported the same tick at this index but different state
    /// hashes — the canonical desync. Capture a snapshot of each run at `tick`
    /// and feed them to [`locate_divergence`] to find the responsible cell.
    Hash {
        /// The agreed simulation tick whose state hashes differ.
        tick: u64,
        /// The left run's state hash at `tick`.
        left: u64,
        /// The right run's state hash at `tick`.
        right: u64,
    },
    /// At frame index `index` the two runs recorded *different* ticks — the
    /// simulation cadence itself drifted (e.g. one run stepped an extra fixed
    /// update), so no meaningful per-tick hash comparison is possible past here.
    Tick {
        /// The frame index at which the recorded ticks differ.
        index: usize,
        /// The left run's tick at `index`.
        left: u64,
        /// The right run's tick at `index`.
        right: u64,
    },
    /// Every shared frame agreed but the runs have different lengths: one ended
    /// at frame `index` while the other continued.
    Length {
        /// The first frame index present in only one run.
        index: usize,
        /// The left run's total recorded frame count.
        left_len: usize,
        /// The right run's total recorded frame count.
        right_len: usize,
    },
}

/// The first coordinate at which two [`WorldSnapshot`]s stop being byte-equal,
/// produced by [`locate_divergence`]. Variants are ordered to match the
/// deterministic fold order of [`state_hash`](WorldSnapshot::state_hash): tick
/// cursors, then allocator liveness, then the entity list, then columns
/// (component set, holders, change ticks, value bytes), then opt-in resources.
/// `left` refers to the first snapshot argument, `right` to the second.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotDivergence {
    /// The two snapshots were captured at different world change ticks.
    ChangeTick {
        /// The left snapshot's change tick.
        left: u64,
        /// The right snapshot's change tick.
        right: u64,
    },
    /// The two snapshots carried different one-shot-read baseline ticks.
    BaselineTick {
        /// The left snapshot's baseline (last-change) tick.
        left: u64,
        /// The right snapshot's baseline (last-change) tick.
        right: u64,
    },
    /// The allocators represent a different number of live entities (including
    /// component-less live entities absent from the captured entity list).
    LiveEntityCount {
        /// The left snapshot's live-entity count.
        left: u32,
        /// The right snapshot's live-entity count.
        right: u32,
    },
    /// The captured entity lists first differ at position `index`: either the
    /// entities at that slot differ, or one list ended there (`None`).
    EntityList {
        /// The entity-list position at which the two lists first differ.
        index: usize,
        /// The left snapshot's entity at `index`, or `None` if its list ended.
        left: Option<Entity>,
        /// The right snapshot's entity at `index`, or `None` if its list ended.
        right: Option<Entity>,
    },
    /// The allocator free lists / generations differ even though live counts and
    /// the captured entity list matched (a deeper allocator-state desync that
    /// would surface as mismatched entity reuse on a later frame).
    AllocatorState,
    /// The column sets first differ at column position `index` (columns are
    /// ordered ascending by [`ComponentId`]): either the component ids differ,
    /// or one snapshot has no column at that position (`None`).
    ColumnSet {
        /// The column position at which the two column lists first differ.
        index: usize,
        /// The left snapshot's component id at `index`, or `None` if it ran out.
        left: Option<ComponentId>,
        /// The right snapshot's component id at `index`, or `None` if it ran out.
        right: Option<ComponentId>,
    },
    /// Within a shared `component` column, the holder sets first differ at
    /// holder position `index`: the owning entities differ, or one column has
    /// fewer holders (`None`).
    Holder {
        /// The component whose column's holder set diverges.
        component: ComponentId,
        /// The in-column holder position at which they first differ.
        index: usize,
        /// The left column's owning entity at `index`, or `None` if it ran out.
        left: Option<Entity>,
        /// The right column's owning entity at `index`, or `None` if it ran out.
        right: Option<Entity>,
    },
    /// A component's recorded change ticks differ for `entity` — same value may
    /// be present but change detection would behave differently on replay.
    ChangeTickCell {
        /// The component whose change ticks differ for `entity`.
        component: ComponentId,
        /// The entity whose cell's change ticks differ.
        entity: Entity,
        /// The left cell's `(added, changed)` ticks.
        left: (u64, u64),
        /// The right cell's `(added, changed)` ticks.
        right: (u64, u64),
    },
    /// A component value's raw bytes differ for `entity` — the classic value
    /// desync (two peers computed a different result for the same component).
    Value {
        /// The component whose stored bytes differ for `entity`.
        component: ComponentId,
        /// The entity whose component value diverged.
        entity: Entity,
    },
    /// The opt-in resource sets first differ at resource position `index`.
    ResourceSet {
        /// The resource position at which the two resource lists first differ.
        index: usize,
    },
    /// A captured resource's identity, type, or value hash differs at position
    /// `index`.
    Resource {
        /// The resource position whose captured value diverges.
        index: usize,
    },
}

/// Locate the first point at which snapshots `left` and `right` stop being
/// byte-equal, in the deterministic fold order of
/// [`state_hash`](WorldSnapshot::state_hash), or [`None`] if they are
/// structurally identical (design §24.4).
///
/// This is the localizing form of
/// [`WorldSnapshot::structurally_eq`](WorldSnapshot::structurally_eq): it
/// compares the identical set of state (tick cursors, allocator liveness,
/// entity list, every column's holders / change ticks / value bytes, and opt-in
/// resources) but returns *where* the first difference is instead of a bool.
/// Consequently `locate_divergence(a, b).is_none() == a.structurally_eq(b)`.
pub fn locate_divergence(
    left: &WorldSnapshot,
    right: &WorldSnapshot,
) -> Option<SnapshotDivergence> {
    // 1. Tick cursors (folded first by `state_hash`).
    if left.change_tick != right.change_tick {
        return Some(SnapshotDivergence::ChangeTick {
            left: left.change_tick.get() as u64,
            right: right.change_tick.get() as u64,
        });
    }
    if left.last_change_tick != right.last_change_tick {
        return Some(SnapshotDivergence::BaselineTick {
            left: left.last_change_tick.get() as u64,
            right: right.last_change_tick.get() as u64,
        });
    }

    // 2. Allocator liveness.
    if left.entities_state.live_len() != right.entities_state.live_len() {
        return Some(SnapshotDivergence::LiveEntityCount {
            left: left.entities_state.live_len(),
            right: right.entities_state.live_len(),
        });
    }

    // 3. Captured entity list, element by element.
    let shared_entities = left.entities.len().min(right.entities.len());
    for index in 0..shared_entities {
        if left.entities[index] != right.entities[index] {
            return Some(SnapshotDivergence::EntityList {
                index,
                left: Some(left.entities[index]),
                right: Some(right.entities[index]),
            });
        }
    }
    if left.entities.len() != right.entities.len() {
        return Some(SnapshotDivergence::EntityList {
            index: shared_entities,
            left: left.entities.get(shared_entities).copied(),
            right: right.entities.get(shared_entities).copied(),
        });
    }

    // Deeper allocator desync (free list / generations) that the live count +
    // entity list did not reveal but `structurally_eq` would still reject.
    if left.entities_state != right.entities_state {
        return Some(SnapshotDivergence::AllocatorState);
    }

    // 4. Columns, ascending by `ComponentId`.
    let shared_columns = left.columns.len().min(right.columns.len());
    for index in 0..shared_columns {
        let lc = &left.columns[index];
        let rc = &right.columns[index];
        if lc.component != rc.component {
            return Some(SnapshotDivergence::ColumnSet {
                index,
                left: Some(lc.component),
                right: Some(rc.component),
            });
        }
        if let Some(div) = locate_column_divergence(left, right, lc, rc) {
            return Some(div);
        }
    }
    if left.columns.len() != right.columns.len() {
        return Some(SnapshotDivergence::ColumnSet {
            index: shared_columns,
            left: left.columns.get(shared_columns).map(|c| c.component),
            right: right.columns.get(shared_columns).map(|c| c.component),
        });
    }

    // 5. Opt-in resources, ascending by id (capture order).
    let shared_resources = left.resources.len().min(right.resources.len());
    for index in 0..shared_resources {
        let lr = &left.resources[index];
        let rr = &right.resources[index];
        if lr.id() != rr.id() || lr.type_id() != rr.type_id() {
            return Some(SnapshotDivergence::ResourceSet { index });
        }
        if lr.value_hash() != rr.value_hash() {
            return Some(SnapshotDivergence::Resource { index });
        }
    }
    if left.resources.len() != right.resources.len() {
        return Some(SnapshotDivergence::ResourceSet {
            index: shared_resources,
        });
    }

    None
}

/// Compare the holder sets, change ticks, and value bytes of two columns that
/// already share a [`ComponentId`], returning the first divergent cell. The
/// holder entities are read from each snapshot's own entity list (columns index
/// into it by row), matching the fold in
/// [`state_hash`](WorldSnapshot::state_hash).
fn locate_column_divergence(
    left: &WorldSnapshot,
    right: &WorldSnapshot,
    lc: &super::column::SnapshotColumn,
    rc: &super::column::SnapshotColumn,
) -> Option<SnapshotDivergence> {
    let component = lc.component;
    let shared = lc.len().min(rc.len());
    for i in 0..shared {
        let lrow = lc.rows[i] as usize;
        let rrow = rc.rows[i] as usize;
        let lentity = left.entities[lrow];
        let rentity = right.entities[rrow];
        if lentity != rentity {
            return Some(SnapshotDivergence::Holder {
                component,
                index: i,
                left: Some(lentity),
                right: Some(rentity),
            });
        }
        if lc.added[i] != rc.added[i] || lc.changed[i] != rc.changed[i] {
            return Some(SnapshotDivergence::ChangeTickCell {
                component,
                entity: lentity,
                left: (lc.added[i].get() as u64, lc.changed[i].get() as u64),
                right: (rc.added[i].get() as u64, rc.changed[i].get() as u64),
            });
        }
        // SAFETY: `i < shared <= lc.len()` and `i < shared <= rc.len()`.
        let (lb, rb) = unsafe { (lc.value_bytes(i), rc.value_bytes(i)) };
        if lb != rb {
            return Some(SnapshotDivergence::Value {
                component,
                entity: lentity,
            });
        }
    }
    if lc.len() != rc.len() {
        return Some(SnapshotDivergence::Holder {
            component,
            index: shared,
            left: lc
                .rows
                .get(shared)
                .map(|&r| left.entities[r as usize]),
            right: rc
                .rows
                .get(shared)
                .map(|&r| right.entities[r as usize]),
        });
    }
    None
}

impl WorldSnapshot {
    /// Locate the first point at which `self` and `other` stop being byte-equal
    /// (design §24.4), or [`None`] if they are structurally identical. The
    /// ergonomic method form of [`locate_divergence`]; see it for the exact
    /// comparison set and ordering.
    #[inline]
    pub fn locate_divergence(&self, other: &WorldSnapshot) -> Option<SnapshotDivergence> {
        locate_divergence(self, other)
    }
}

#[cfg(test)]
mod tests;
