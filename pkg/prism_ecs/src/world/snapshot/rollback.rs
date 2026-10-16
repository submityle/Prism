//! Bounded rollback history: one full anchor followed by adjacent-frame deltas.
//!
//! Frames are ordered by numeric key, not arrival order. Out-of-order insertion
//! preserves other recorded states; at capacity the smallest key is discarded
//! (including an incoming key older than the window). Replacing a key changes
//! only that frame, not the independently recorded future.
//!
//! Reads reconstruct owned snapshots instead of caching a full copy per frame.
//! Thus `get` and `latest` now return owned snapshots: callers of the borrowed
//! API must pass `&snapshot` to `World::restore`. `oldest` still borrows the
//! anchor. Use `latest_frame` to inspect the newest key without reconstruction.
//!
//! Component payload storage is one anchor plus each delta's fresh values.
//! [`capture`](SnapshotRing::capture) reuses versioned immutable component
//! chunks. Such buffers are also shared during diff/reconstruction, without
//! cloning or comparing their values. Clone-only components remain conservative.
//! Metadata still visits cells, resources/relations remain full copies, and
//! reads still walk the delta prefix; this is not constant-time history access.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;

use super::{SnapshotDelta, WorldSnapshot};
use crate::world::World;

enum StoredSnapshot {
    Anchor(WorldSnapshot),
    Delta(SnapshotDelta),
}

struct Frame {
    number: u64,
    snapshot: StoredSnapshot,
    resource_values: usize,
    relation_edges: usize,
}

/// Bounded, numerically ordered snapshot history (design §14).
///
/// Exactly the first retained frame is a full anchor; other frames are deltas
/// against their immediate predecessors. Eviction promotes the second frame
/// to an anchor, preserving its capture lineage so later deltas remain valid.
/// Equal hashes never establish value equality or delta compatibility.
pub struct SnapshotRing {
    capacity: usize,
    frames: VecDeque<Frame>,
    revision: u64,
    identity: Arc<()>,
}

/// Ownership of the entries displaced by one strictly increasing append.
/// Consumed by [`SnapshotRing::undo_append`] before any other ring mutation.
pub struct SnapshotAppendUndo {
    frame: u64,
    revision: u64,
    evicted: Option<Frame>,
    promoted: Option<StoredSnapshot>,
    identity: Arc<()>,
}

/// Displaced snapshot storage awaiting destruction after a history mutation.
///
/// Drop this only after publishing associated metadata (for example a history
/// cursor). User destructors can panic; the ring mutation is already committed.
#[must_use = "publish associated metadata before dropping displaced snapshots"]
pub struct SnapshotRetired {
    _snapshots: [Option<StoredSnapshot>; 2],
    _frames: [Option<Frame>; 2],
}

impl SnapshotRing {
    /// Retain at most `capacity` distinct frame keys.
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SnapshotRing capacity must be non-zero");
        Self {
            capacity,
            frames: VecDeque::with_capacity(capacity),
            revision: 0,
            identity: Arc::new(()),
        }
    }

    /// The maximum number of retained frames.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of retained frames.
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the history is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Incrementally capture the live world and record a frame. Reusable
    /// components share clean chunk buffers with the latest recorded capture;
    /// diff, eviction and history reads retain those immutable buffers too.
    /// This closes a world change-tick boundary on successful capture, as
    /// documented by [`World::snapshot_incremental`](crate::world::World::snapshot_incremental).
    pub fn capture(&mut self, frame: u64, world: &mut World) {
        self.try_capture(frame, world)
            .unwrap_or_else(|missing| panic!("history capture missing clone glue for {missing:?}"));
    }

    /// Fallible [`capture`](Self::capture). Missing clone glue does not alter
    /// either the ring or the world change tick. The tick boundary is closed
    /// only after history insertion succeeds, including any required rebasing.
    pub fn try_capture(
        &mut self,
        frame: u64,
        world: &mut World,
    ) -> Result<(), Vec<crate::component::ComponentId>> {
        let previous = self.latest().map(|(_, snapshot)| snapshot);
        let snapshot = super::capture::capture_incremental(world, previous.as_ref())?;
        let retired = self.push_deferred(frame, snapshot);
        world.increment_change_tick();
        drop(retired);
        Ok(())
    }

    /// Insert or replace exactly `frame`, retaining the greatest `capacity` keys.
    ///
    /// Insertion/replacement reconnects adjacent deltas without changing any
    /// other frame's ticks, values, resources or relations. A full ring ignores
    /// a new key smaller than its oldest key. To abandon a future branch, call
    /// `truncate_after` explicitly.
    ///
    /// Value comparisons/clones, including eviction rebase, finish before the
    /// history is changed. If one unwinds, keys and recorded states remain
    /// unchanged. This assumes Clone/PartialEq do not mutate shared captured
    /// values; destructor panics and allocator aborts are not rollbackable.
    pub fn push(&mut self, frame: u64, snapshot: WorldSnapshot) {
        drop(self.push_deferred(frame, snapshot));
    }

    /// Like [`push`](Self::push), but defer destruction of displaced values
    /// until the returned storage is dropped. Preparation panics leave the
    /// ring unchanged; destruction panics happen after the mutation commits.
    pub fn push_deferred(&mut self, frame: u64, snapshot: WorldSnapshot) -> SnapshotRetired {
        if self.latest_frame().is_none_or(|latest| frame > latest) {
            let undo = self.append_reversible(frame, snapshot);
            return SnapshotRetired {
                _snapshots: [undo.promoted, None],
                _frames: [undo.evicted, None],
            };
        }
        let index = self
            .frames
            .iter()
            .position(|entry| entry.number >= frame)
            .unwrap_or(self.frames.len());
        let replaces = self.frames.get(index).is_some_and(|f| f.number == frame);
        let evicts = !replaces && self.frames.len() == self.capacity;
        if evicts && index == 0 {
            return SnapshotRetired {
                _snapshots: [Some(StoredSnapshot::Anchor(snapshot)), None],
                _frames: [None, None],
            };
        }

        let successor_index = index + usize::from(replaces);
        let successor = if successor_index < self.frames.len() {
            let old = self.reconstruct_index(successor_index);
            Some(snapshot.diff(&old))
        } else {
            None
        };
        // An insertion directly after the old anchor promotes the input itself.
        // Otherwise materialize the old second frame before removing its base.
        let promoted = if evicts && index > 1 {
            Some(self.reconstruct_index(1))
        } else {
            None
        };
        let resource_values = snapshot.resource_count();
        let relation_edges = snapshot.relation_count();
        let incoming = if index == 0 || (evicts && index == 1) {
            StoredSnapshot::Anchor(snapshot)
        } else {
            let predecessor = self.reconstruct_index(index - 1);
            let delta = predecessor.diff(&snapshot);
            // Do not leave the consumed input's destructor until after commit.
            drop(snapshot);
            StoredSnapshot::Delta(delta)
        };
        let incoming = Frame {
            number: frame,
            snapshot: incoming,
            resource_values,
            relation_edges,
        };

        // Reserve before commit. Keep displaced values alive until all edges
        // are connected, so even their Drop observes a valid history.
        if !replaces {
            self.frames.reserve(1);
        }
        let old_successor = successor.map(|delta| {
            core::mem::replace(
                &mut self.frames[successor_index].snapshot,
                StoredSnapshot::Delta(delta),
            )
        });
        let old_frame = if replaces {
            Some(core::mem::replace(&mut self.frames[index], incoming))
        } else {
            self.frames.insert(index, incoming);
            None
        };
        let evicted = if evicts {
            self.frames.pop_front()
        } else {
            None
        };
        let old_promoted = promoted.map(|anchor| {
            core::mem::replace(&mut self.frames[0].snapshot, StoredSnapshot::Anchor(anchor))
        });
        self.advance_revision();
        SnapshotRetired {
            _snapshots: [old_successor, old_promoted],
            _frames: [old_frame, evicted],
        }
    }

    /// Append a frame newer than every retained key, retaining displaced
    /// storage without cloning it so a failed downstream send can be undone.
    ///
    /// # Panics
    /// Panics if `frame` is not strictly newer than the last retained key.
    /// Preparation (including consuming the input snapshot) precedes commit;
    /// dropping the returned token commits the append, not an automatic undo.
    pub fn append_reversible(&mut self, frame: u64, snapshot: WorldSnapshot) -> SnapshotAppendUndo {
        assert!(self.latest_frame().is_none_or(|latest| frame > latest));
        let evicts = self.frames.len() == self.capacity;
        let promoted = if evicts && self.frames.len() > 1 {
            Some(self.reconstruct_index(1))
        } else {
            None
        };
        let resource_values = snapshot.resource_count();
        let relation_edges = snapshot.relation_count();
        let stored = if self.frames.is_empty() || (evicts && self.capacity == 1) {
            StoredSnapshot::Anchor(snapshot)
        } else {
            let predecessor = self.reconstruct_index(self.frames.len() - 1);
            let delta = predecessor.diff(&snapshot);
            drop(snapshot);
            StoredSnapshot::Delta(delta)
        };
        self.frames.reserve(1);
        self.frames.push_back(Frame {
            number: frame,
            snapshot: stored,
            resource_values,
            relation_edges,
        });
        let evicted = if evicts {
            self.frames.pop_front()
        } else {
            None
        };
        let promoted = promoted.map(|anchor| {
            core::mem::replace(&mut self.frames[0].snapshot, StoredSnapshot::Anchor(anchor))
        });
        self.advance_revision();
        SnapshotAppendUndo {
            frame,
            revision: self.revision,
            evicted,
            promoted,
            identity: self.identity.clone(),
        }
    }

    /// Undo precisely the immediately preceding append, including eviction and
    /// delta-anchor promotion. A stale token is refused without changing keys.
    pub fn undo_append(&mut self, undo: SnapshotAppendUndo) -> bool {
        self.undo_append_deferred(undo).is_some()
    }

    /// Undo an append, deferring destruction until associated metadata can be
    /// published. Returns `None` for a foreign or stale token without changing
    /// the ring. Rejected tokens are consumed and their destructors may panic.
    pub fn undo_append_deferred(&mut self, undo: SnapshotAppendUndo) -> Option<SnapshotRetired> {
        if !Arc::ptr_eq(&self.identity, &undo.identity)
            || self.revision != undo.revision
            || self.latest_frame() != Some(undo.frame)
        {
            return None;
        }
        let removed = self.frames.pop_back();
        let promoted = undo
            .promoted
            .map(|old| core::mem::replace(&mut self.frames[0].snapshot, old));
        if let Some(evicted) = undo.evicted {
            self.frames.push_front(evicted);
        }
        self.advance_revision();
        Some(SnapshotRetired {
            _snapshots: [promoted, None],
            _frames: [removed, None],
        })
    }

    fn advance_revision(&mut self) {
        // Outstanding tokens keep the old allocation alive: wrapping a counter
        // must not let an old token authenticate a different mutation (ABA).
        if self.revision == u64::MAX {
            self.identity = Arc::new(());
        }
        self.revision = self.revision.wrapping_add(1);
    }

    /// Reconstruct an owned snapshot for `frame`, without caching it.
    ///
    /// Returns `None` for an unrecorded/evicted key. Clone failures propagate
    /// without changing history. Cost is proportional to cells materialized
    /// along the prefix, not just this frame's changed cells.
    /// Migrate `world.restore(ring.get(f).unwrap())` to binding the result and
    /// passing `&snapshot`. The result outlives mutations of the ring.
    pub fn get(&self, frame: u64) -> Option<WorldSnapshot> {
        self.frames
            .iter()
            .position(|entry| entry.number == frame)
            .map(|index| self.reconstruct_index(index))
    }

    /// Whether `frame` is retained, without reconstructing values.
    #[inline]
    pub fn contains(&self, frame: u64) -> bool {
        self.frames.iter().any(|entry| entry.number == frame)
    }

    /// The greatest retained frame number, without reconstruction.
    pub fn latest_frame(&self) -> Option<u64> {
        self.frames.back().map(|entry| entry.number)
    }

    /// The greatest retained frame and its reconstructed owned snapshot.
    ///
    /// Use `latest_frame` when only the frame number is needed.
    pub fn latest(&self) -> Option<(u64, WorldSnapshot)> {
        let frame = self.latest_frame()?;
        Some((frame, self.reconstruct_index(self.frames.len() - 1)))
    }

    /// Borrow the smallest retained frame and its full anchor.
    pub fn oldest(&self) -> Option<(u64, &WorldSnapshot)> {
        self.frames.front().map(|entry| match &entry.snapshot {
            StoredSnapshot::Anchor(anchor) => (entry.number, anchor),
            StoredSnapshot::Delta(_) => unreachable!("first retained frame is always an anchor"),
        })
    }

    /// Retained frame keys in ascending numeric order, regardless of arrival.
    pub fn frames(&self) -> impl Iterator<Item = u64> + '_ {
        self.frames.iter().map(|entry| entry.number)
    }

    /// Drop all keys greater than `frame`, even if the cutoff was not recorded.
    ///
    /// Prefix deltas remain valid. A cutoff older than the anchor clears history.
    /// The entire tail is detached before any user destructor is called.
    pub fn truncate_after(&mut self, frame: u64) {
        let keep = self.frames.partition_point(|entry| entry.number <= frame);
        let retired = self.frames.split_off(keep);
        self.advance_revision();
        drop(retired);
    }

    /// Drop all history, installing the empty state before values are dropped.
    pub fn clear(&mut self) {
        self.advance_revision();
        let old = core::mem::take(&mut self.frames);
        drop(old);
    }

    /// Number of full snapshots retained internally (zero or one).
    pub fn full_snapshot_count(&self) -> usize {
        usize::from(!self.is_empty())
    }

    /// Number of retained delta frames, excluding the anchor.
    pub fn delta_count(&self) -> usize {
        self.len().saturating_sub(1)
    }

    /// Logical retained cells: anchor cells plus fresh cells in every delta.
    ///
    /// Immutable chunks may share allocations and retain additional buffer
    /// slots, so this is NOT a physical value/byte/allocation count. Excludes per-cell selectors, entity
    /// and allocator metadata, ticks, resources, relations, allocation capacity
    /// and user-owned heap payload sizes. Reconstruction scratch and returned
    /// snapshots are not retained internally and are also excluded.
    pub fn stored_component_cell_count(&self) -> usize {
        self.frames
            .iter()
            .map(|entry| match &entry.snapshot {
                StoredSnapshot::Anchor(anchor) => anchor.columns.iter().map(|c| c.len()).sum(),
                StoredSnapshot::Delta(delta) => delta.changed_cell_count(),
            })
            .sum()
    }

    /// Per-cell base references in retained deltas, not owned component values.
    pub fn reused_component_cell_count(&self) -> usize {
        self.frames
            .iter()
            .map(|entry| match &entry.snapshot {
                StoredSnapshot::Anchor(_) => 0,
                StoredSnapshot::Delta(delta) => delta.reused_cell_count(),
            })
            .sum()
    }

    /// Captured resource values across the anchor and deltas.
    ///
    /// Resources are copied in full per frame. Excludes their heap payload sizes;
    /// this is not a total-memory measurement.
    pub fn stored_resource_count(&self) -> usize {
        self.frames.iter().map(|entry| entry.resource_values).sum()
    }

    /// Logical relation edges across frames, not bytes or index entries.
    ///
    /// Relations are copied in full per frame, with both ordered indexes and
    /// canonical comparison data; each logical edge occupies multiple entries.
    pub fn stored_relation_count(&self) -> usize {
        self.frames.iter().map(|entry| entry.relation_edges).sum()
    }

    fn reconstruct_index(&self, index: usize) -> WorldSnapshot {
        let (_, anchor) = self.oldest().expect("reconstructing a retained frame");
        if index == 0 {
            return clone_anchor(anchor);
        }
        let mut snapshot = None;
        for entry in self.frames.iter().take(index + 1).skip(1) {
            let StoredSnapshot::Delta(delta) = &entry.snapshot else {
                unreachable!("only the first retained frame is an anchor");
            };
            snapshot = Some(delta.apply(snapshot.as_ref().unwrap_or(anchor)));
        }
        snapshot.expect("a non-anchor frame has a delta")
    }
}

/// Clone the anchor without invoking equality or changing capture lineage.
fn clone_anchor(anchor: &WorldSnapshot) -> WorldSnapshot {
    let mut columns = Vec::with_capacity(anchor.columns.len());
    for source in &anchor.columns {
        let mut column = source.empty_like();
        for (slot, &row) in source.rows.iter().enumerate() {
            // SAFETY: slot is a live source cell; empty_like preserves every
            // layout/type/glue contract. push_cloned_from also copies its ticks.
            unsafe { column.push_cloned_from(source, slot, row) };
        }
        columns.push(column);
    }
    WorldSnapshot {
        identity: anchor.identity.clone(),
        world_id: anchor.world_id,
        schema_generation: anchor.schema_generation,
        change_tick: anchor.change_tick,
        last_change_tick: anchor.last_change_tick,
        entities_state: anchor.entities_state.clone(),
        entities: anchor.entities.clone(),
        columns,
        resources: super::resource::clone_resources(&anchor.resources),
        relations: anchor.relations.clone(),
    }
}

#[cfg(test)]
mod reversible_tests {
    use super::*;

    #[test]
    fn revision_wrap_changes_identity_and_refuses_an_aba_token() {
        let world = World::new();
        let mut ring = SnapshotRing::new(4);
        let stale = ring.append_reversible(1, world.snapshot());
        ring.revision = u64::MAX;
        ring.push(2, world.snapshot());
        ring.truncate_after(1);
        assert_eq!(ring.revision, stale.revision);
        assert_eq!(ring.latest_frame(), Some(stale.frame));
        assert!(!ring.undo_append(stale));
        assert_eq!(ring.frames().collect::<Vec<_>>(), [1]);
    }

    #[test]
    fn numeric_frame_keys_do_not_wrap_and_clear_invalidates_tokens() {
        let world = World::new();
        for capacity in [1, 4] {
            let mut ring = SnapshotRing::new(capacity);
            ring.push(u64::MAX - 1, world.snapshot());
            let undo = ring.append_reversible(u64::MAX, world.snapshot());
            assert!(ring.undo_append(undo));
            assert_eq!(ring.latest_frame(), Some(u64::MAX - 1));
            let stale = ring.append_reversible(u64::MAX, world.snapshot());
            ring.clear();
            ring.push(u64::MAX, world.snapshot());
            assert!(!ring.undo_append(stale));
            assert_eq!(ring.latest_frame(), Some(u64::MAX));
        }
    }
}
