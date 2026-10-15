//! World-backed checkpoints, optionally paired with the existing UI [`Frame`].
//!
//! Seeking restores entities, components, ticks and opt-in snapshot resources.
//! It does not replay hooks, observers, UI backend operations or external I/O.
//! Register clone glue for all resident components before recording. Histories
//! are bound to one World; runtime component schema changes invalidate older
//! checkpoints. Capacity and branch behavior match the UI-only timeline.
//!
//! Checkpoints share one anchor + delta [`SnapshotRing`], not one full World
//! snapshot per frame. [`EcsFrame::snapshot`] reconstructs an owned value; code
//! that previously borrowed its result must now bind it and pass `&snapshot`
//! to `World::restore`. No reconstructed snapshots are cached by the history.
//! SnapshotRing currently has a fixed, nonzero capacity, so an unbounded UI
//! history grows geometrically by reconstructing a new ring when needed.
//! This growth and ordinary reads can be expensive: deltas retain structural
//! metadata and full resources/relations, and reconstruction clones the prefix.

use alloc::{string::String, sync::Arc, vec::Vec};
use core::fmt;
use std::sync::RwLock;

use prism_ecs::{
    component::ComponentId,
    world::{
        snapshot::{SnapshotRing, WorldSnapshot},
        World, WorldId,
    },
};

use crate::Frame;

/// A recoverable failure that leaves history and the navigation cursor intact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EcsHistoryError {
    /// The supplied world is not the one this history was created against.
    WrongWorld,
    /// These resident component types need snapshot clone registration.
    MissingCloneGlue(Vec<ComponentId>),
    /// The requested index is outside the retained history.
    MissingFrame(usize),
    /// A runtime component schema replacement invalidated this checkpoint.
    SchemaChanged,
}

impl fmt::Display for EcsHistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongWorld => f.write_str("ECS history belongs to another World"),
            Self::MissingCloneGlue(ids) => write!(f, "missing snapshot clone glue: {ids:?}"),
            Self::MissingFrame(index) => write!(f, "ECS checkpoint {index} is not retained"),
            Self::SchemaChanged => {
                f.write_str("ECS checkpoint predates a component schema replacement")
            }
        }
    }
}

impl std::error::Error for EcsHistoryError {}

/// One World checkpoint and its UI frame, backed by the timeline's delta ring.
pub struct EcsFrame {
    label: String,
    key: u64,
    schema_generation: u64,
    snapshots: Arc<RwLock<SnapshotRing>>,
    ui: Option<Frame>,
}

impl EcsFrame {
    /// User-facing checkpoint label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Reconstruct the actual ECS state restored by seek, undo and redo.
    ///
    /// Unlike the original borrowed API, this returns an owned snapshot, which
    /// remains valid after navigation, eviction, branching or clearing history.
    /// Bind the result and use `world.restore(&snapshot)` when restoring it.
    /// This clones the delta prefix's values and does not cache the result.
    pub fn snapshot(&self) -> WorldSnapshot {
        self.snapshots
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(self.key)
            .expect("every retained UI frame has a retained World checkpoint")
    }

    /// The associated UI frame, if the caller supplied one.
    pub fn ui(&self) -> Option<&Frame> {
        self.ui.as_ref()
    }
}

/// Logical retained ECS payload counts, not allocated bytes or capture cost.
///
/// UI frames, entity/column metadata, delta selectors, allocation capacity and
/// user-owned heap payloads are excluded. Resources and relations are still
/// stored in full for every frame by SnapshotDelta. Immutable component chunks
/// can share allocations or retain additional buffer slots, so these counts
/// must not be interpreted as physical value or allocation counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EcsHistoryStorage {
    /// Full anchors retained (zero for empty history, otherwise one).
    pub full_snapshots: usize,
    /// Adjacent-frame deltas retained, excluding the anchor.
    pub deltas: usize,
    /// Logical component cells in the anchor and fresh cells in each delta.
    pub component_cells: usize,
    /// Component selectors reusing predecessor values, not owned values.
    pub reused_component_cells: usize,
    /// Full captured resource values across all retained frames.
    pub resources: usize,
    /// Logical relationship edges across all retained frames.
    pub relation_edges: usize,
}

/// A bounded, branching history that restores the live World on navigation.
///
/// Indices address the current retained slice, as in [`crate::Timeline`];
/// eviction shifts them. A capacity of zero means unbounded retention.
pub struct EcsTimeline {
    world: WorldId,
    frames: Vec<EcsFrame>,
    snapshots: Arc<RwLock<SnapshotRing>>,
    cursor: Option<usize>,
    capacity: usize,
}

impl EcsTimeline {
    /// Bind an empty history to `world`.
    pub fn new(world: &World, capacity: usize) -> Self {
        Self {
            world: world.id(),
            frames: Vec::new(),
            snapshots: Arc::new(RwLock::new(SnapshotRing::new(capacity.max(1)))),
            cursor: None,
            capacity,
        }
    }

    /// Validate identity without capturing, restoring or mutating anything.
    pub fn validate_world(&self, world: &World) -> Result<(), EcsHistoryError> {
        if world.id() == self.world {
            Ok(())
        } else {
            Err(EcsHistoryError::WrongWorld)
        }
    }

    /// Record a World checkpoint without an associated UI frame.
    pub fn record(
        &mut self,
        world: &World,
        label: impl Into<String>,
    ) -> Result<usize, EcsHistoryError> {
        self.record_with_ui(world, label, None)
    }

    /// Record actual World state together with a UI inspection frame.
    ///
    /// Capture completes before discarding the redo branch or evicting a frame;
    /// a missing clone registration cannot destroy the existing history.
    pub fn record_with_ui(
        &mut self,
        world: &World,
        label: impl Into<String>,
        ui: Option<Frame>,
    ) -> Result<usize, EcsHistoryError> {
        self.validate_world(world)?;
        let label = label.into();
        let snapshot = world
            .try_snapshot()
            .map_err(EcsHistoryError::MissingCloneGlue)?;
        let schema_generation = snapshot.schema_generation();
        let next = self.cursor.map_or(0, |index| index + 1);
        let key = self.frames.get(next).map_or_else(
            || {
                self.current()
                    .map_or(Some(0), |frame| frame.key.checked_add(1))
            },
            |frame| Some(frame.key),
        );
        // Reserve metadata before committing the ring. At a branch, replace
        // the first redo key before truncation: SnapshotRing prepares all
        // fallible value clones/comparisons before committing that replacement.
        // A capture or delta failure therefore leaves the redo branch intact.
        self.frames.reserve(1);
        // SnapshotRing commits only after fallible preparation. Recovering a
        // poisoned lock therefore permits another recording after a user's
        // Clone/PartialEq panic, without losing the old navigation branch.
        let mut snapshots = self
            .snapshots
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let grows = self.capacity == 0 && next == snapshots.capacity();
        let renumbers = key.is_none();
        let key = if grows || renumbers {
            let capacity = if grows {
                snapshots
                    .capacity()
                    .checked_mul(2)
                    .expect("history capacity overflow")
            } else {
                snapshots.capacity()
            };
            let mut replacement = SnapshotRing::new(capacity);
            for (index, frame) in self.frames.iter().take(next).enumerate() {
                let key = if renumbers { index as u64 } else { frame.key };
                replacement.push(key, snapshots.get(frame.key).expect("retained checkpoint"));
            }
            let key = key.unwrap_or(next as u64);
            replacement.push(key, snapshot);
            *snapshots = replacement;
            if renumbers {
                for (index, frame) in self.frames.iter_mut().take(next).enumerate() {
                    frame.key = index as u64;
                }
            }
            key
        } else {
            let key = key.expect("key overflow is handled by renumbering");
            snapshots.push(key, snapshot);
            snapshots.truncate_after(key);
            key
        };
        drop(snapshots);
        self.frames.truncate(next);
        self.frames.push(EcsFrame {
            label,
            key,
            schema_generation,
            snapshots: Arc::clone(&self.snapshots),
            ui,
        });
        self.cursor = Some(self.frames.len() - 1);
        self.enforce_capacity();
        Ok(self.cursor.expect("record inserted a frame"))
    }

    /// Restore a retained checkpoint and move the cursor only after restoration.
    pub fn seek(&mut self, world: &mut World, index: usize) -> Result<&EcsFrame, EcsHistoryError> {
        self.validate_world(world)?;
        let frame = self
            .frames
            .get(index)
            .ok_or(EcsHistoryError::MissingFrame(index))?;
        if frame.schema_generation != world.components().schema_generation() {
            return Err(EcsHistoryError::SchemaChanged);
        }
        let snapshot = frame.snapshot();
        world.restore(&snapshot);
        self.cursor = Some(index);
        Ok(&self.frames[index])
    }

    /// Restore the preceding checkpoint, or return `None` at the oldest entry.
    pub fn undo(&mut self, world: &mut World) -> Result<Option<&EcsFrame>, EcsHistoryError> {
        self.validate_world(world)?;
        match self.cursor.and_then(|index| index.checked_sub(1)) {
            Some(index) => self.seek(world, index).map(Some),
            None => Ok(None),
        }
    }

    /// Restore the next checkpoint, or return `None` at the newest entry.
    pub fn redo(&mut self, world: &mut World) -> Result<Option<&EcsFrame>, EcsHistoryError> {
        self.validate_world(world)?;
        match self
            .cursor
            .map(|index| index + 1)
            .filter(|&index| index < self.frames.len())
        {
            Some(index) => self.seek(world, index).map(Some),
            None => Ok(None),
        }
    }

    /// The retained checkpoints in recording order.
    pub fn frames(&self) -> &[EcsFrame] {
        &self.frames
    }
    /// The current retained index, or `None` before the first recording.
    pub fn cursor(&self) -> Option<usize> {
        self.cursor
    }
    /// The current checkpoint.
    pub fn current(&self) -> Option<&EcsFrame> {
        self.cursor.and_then(|index| self.frames.get(index))
    }
    /// Number of retained checkpoints.
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    /// Whether no checkpoints are retained.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
    /// Whether an earlier checkpoint is available.
    pub fn can_undo(&self) -> bool {
        self.cursor.is_some_and(|index| index > 0)
    }
    /// Whether a later checkpoint is available.
    pub fn can_redo(&self) -> bool {
        self.cursor
            .is_some_and(|index| index + 1 < self.frames.len())
    }
    /// Retention bound, or `None` for an unbounded history.
    pub fn capacity(&self) -> Option<usize> {
        (self.capacity != 0).then_some(self.capacity)
    }

    /// Inspect retained ECS storage without reconstructing any checkpoints.
    pub fn storage(&self) -> EcsHistoryStorage {
        let snapshots = self
            .snapshots
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        EcsHistoryStorage {
            full_snapshots: snapshots.full_snapshot_count(),
            deltas: snapshots.delta_count(),
            component_cells: snapshots.stored_component_cell_count(),
            reused_component_cells: snapshots.reused_component_cell_count(),
            resources: snapshots.stored_resource_count(),
            relation_edges: snapshots.stored_relation_count(),
        }
    }

    /// Discard checkpoints while preserving identity and retention settings.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.snapshots
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        self.cursor = None;
    }

    fn enforce_capacity(&mut self) {
        if self.capacity != 0 && self.frames.len() > self.capacity {
            let excess = self.frames.len() - self.capacity;
            self.frames.drain(..excess);
            self.cursor = self.cursor.map(|index| index.saturating_sub(excess));
        }
    }
}
