//! Bounded runtime scheduling for a toroidal SDF clipmap.
//!
//! The residency crate owns integer window geometry. This module adds the
//! runtime transaction boundary: center rolls are prepared against a revision,
//! dirty bricks are kept only for the current window, and a prepared roll is
//! committed without exposing a partially updated state.

extern crate alloc;

use alloc::{sync::Arc, vec::Vec};
use core::{fmt, mem::size_of};

use prism_world_residency::{
    BrickCoord, SdfClipmapConfig, SdfClipmapError, SdfClipmapUpdate, SdfToroidalClipmap,
};

/// Runtime bounds for one toroidal SDF clipmap scheduler.
///
/// `clipmap.max_update_bytes` bounds the residency planner's center-roll
/// payload. The two dirty limits independently bound the runtime's explicit
/// dirty-brick list, including the automatically dirty newly-entered bricks.
/// Mark requests have separate scratch limits, charged before deduplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfClipmapConfig {
    /// Fixed geometry and logical center-roll budget.
    pub clipmap: SdfClipmapConfig,
    /// Maximum number of dirty coordinates retained for the current window.
    pub max_dirty_bricks: usize,
    /// Maximum logical bytes charged to the dirty coordinate list.
    pub max_dirty_bytes: usize,
    /// Maximum number of input coordinates in one mark request, duplicates included.
    pub max_mark_request_bricks: usize,
    /// Maximum logical coordinate bytes copied for one mark request.
    pub max_mark_request_bytes: usize,
}

impl WorldSdfClipmapConfig {
    /// Creates a checked runtime clipmap configuration.
    pub fn new(
        clipmap: SdfClipmapConfig,
        max_dirty_bricks: usize,
        max_dirty_bytes: usize,
    ) -> Result<Self, WorldSdfClipmapError> {
        SdfToroidalClipmap::new(clipmap).map_err(WorldSdfClipmapError::Clipmap)?;
        if max_dirty_bytes == 0 {
            return Err(WorldSdfClipmapError::InvalidDirtyBudget);
        }
        let config = Self {
            clipmap,
            max_dirty_bricks,
            max_dirty_bytes,
            max_mark_request_bricks: max_dirty_bricks,
            max_mark_request_bytes: max_dirty_bytes,
        };
        config.validate_dirty_count(0)?;
        Ok(config)
    }

    /// Overrides mark-request scratch limits independently of the retained list.
    ///
    /// By default, [`Self::new`] uses the dirty limits for both. Request limits
    /// apply to raw input, including duplicates; clear requests need no copy of
    /// the input and are not subject to these limits.
    pub fn with_mark_request_budget(
        mut self,
        max_bricks: usize,
        max_bytes: usize,
    ) -> Result<Self, WorldSdfClipmapError> {
        if max_bytes == 0 {
            return Err(WorldSdfClipmapError::InvalidMarkRequestBudget);
        }
        self.max_mark_request_bricks = max_bricks;
        self.max_mark_request_bytes = max_bytes;
        Ok(self)
    }

    /// Returns the logical byte charge for one dirty brick coordinate.
    pub const fn dirty_coord_bytes() -> usize {
        size_of::<BrickCoord>()
    }

    fn validate_dirty_count(&self, count: usize) -> Result<(), WorldSdfClipmapError> {
        if count > self.max_dirty_bricks {
            return Err(WorldSdfClipmapError::DirtyCapacityExceeded {
                required: count,
                capacity: self.max_dirty_bricks,
            });
        }
        let required = count
            .checked_mul(Self::dirty_coord_bytes())
            .ok_or(WorldSdfClipmapError::ArithmeticOverflow)?;
        if required > self.max_dirty_bytes {
            return Err(WorldSdfClipmapError::DirtyBudgetExceeded {
                required,
                budget: self.max_dirty_bytes,
            });
        }
        Ok(())
    }

    fn validate_mark_request(&self, count: usize) -> Result<(), WorldSdfClipmapError> {
        if count > self.max_mark_request_bricks {
            return Err(WorldSdfClipmapError::MarkRequestCapacityExceeded {
                required: count,
                capacity: self.max_mark_request_bricks,
            });
        }
        let required = count
            .checked_mul(Self::dirty_coord_bytes())
            .ok_or(WorldSdfClipmapError::ArithmeticOverflow)?;
        if required > self.max_mark_request_bytes {
            return Err(WorldSdfClipmapError::MarkRequestBudgetExceeded {
                required,
                budget: self.max_mark_request_bytes,
            });
        }
        Ok(())
    }
}

/// Errors returned by the runtime SDF clipmap scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldSdfClipmapError {
    /// The residency clipmap rejected geometry, coordinates, or its update
    /// budget.
    Clipmap(SdfClipmapError),
    /// A checked count or byte computation overflowed.
    ArithmeticOverflow,
    /// A fallible host allocation could not be satisfied.
    AllocationFailed,
    /// The dirty list would exceed its item capacity.
    DirtyCapacityExceeded {
        /// Number of dirty coordinates required.
        required: usize,
        /// Configured dirty-coordinate capacity.
        capacity: usize,
    },
    /// The dirty list would exceed its logical byte budget.
    DirtyBudgetExceeded {
        /// Logical bytes required by the dirty list.
        required: usize,
        /// Configured logical byte budget.
        budget: usize,
    },
    /// A dirty coordinate is not in the currently committed window.
    DirtyOutsideWindow {
        /// Coordinate rejected by the current-window fence.
        coord: BrickCoord,
    },
    /// The dirty byte budget must be non-zero.
    InvalidDirtyBudget,
    /// The mark-request scratch byte budget must be non-zero.
    InvalidMarkRequestBudget,
    /// Raw mark input, including duplicates, exceeds its item limit.
    MarkRequestCapacityExceeded {
        /// Number of input coordinates.
        required: usize,
        /// Configured input-coordinate capacity.
        capacity: usize,
    },
    /// Raw mark input, including duplicates, exceeds its scratch byte limit.
    MarkRequestBudgetExceeded {
        /// Logical coordinate bytes required by the input copy.
        required: usize,
        /// Configured scratch byte budget.
        budget: usize,
    },
    /// A prepared operation was made against an older scheduler revision.
    RevisionConflict {
        /// Revision captured by the operation.
        expected: u64,
        /// Scheduler revision at the time of the attempted operation.
        actual: u64,
    },
    /// The scheduler revision cannot be incremented safely.
    RevisionOverflow,
    /// A plan belongs to another scheduler, even if its revision matches.
    ForeignPlan,
}

impl fmt::Display for WorldSdfClipmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clipmap(error) => write!(f, "world SDF clipmap error: {error}"),
            Self::ArithmeticOverflow => f.write_str("world SDF clipmap arithmetic overflow"),
            Self::AllocationFailed => f.write_str("world SDF clipmap allocation failed"),
            Self::DirtyCapacityExceeded { required, capacity } => write!(
                f,
                "world SDF clipmap needs {required} dirty bricks, capacity {capacity}"
            ),
            Self::DirtyBudgetExceeded { required, budget } => write!(
                f,
                "world SDF clipmap dirty list needs {required} bytes, budget {budget}"
            ),
            Self::DirtyOutsideWindow { coord } => {
                write!(f, "dirty brick {coord:?} is outside the current SDF window")
            }
            Self::InvalidDirtyBudget => f.write_str("world SDF clipmap dirty budget is zero"),
            Self::InvalidMarkRequestBudget => {
                f.write_str("world SDF clipmap mark-request budget is zero")
            }
            Self::MarkRequestCapacityExceeded { required, capacity } => write!(
                f,
                "world SDF clipmap mark request needs {required} coordinates, capacity {capacity}"
            ),
            Self::MarkRequestBudgetExceeded { required, budget } => write!(
                f,
                "world SDF clipmap mark request needs {required} scratch bytes, budget {budget}"
            ),
            Self::RevisionConflict { expected, actual } => write!(
                f,
                "world SDF clipmap revision conflict: expected {expected}, actual {actual}"
            ),
            Self::RevisionOverflow => f.write_str("world SDF clipmap revision overflow"),
            Self::ForeignPlan => f.write_str("world SDF clipmap plan belongs to another scheduler"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WorldSdfClipmapError {}

/// A center-roll plan prepared against one scheduler revision.
///
/// The plan owns the prospective toroidal state and dirty list. Preparing it
/// does not mutate the scheduler; [`WorldSdfClipmapScheduler::commit`] applies
/// it only if its revision fence still matches.
#[derive(Debug)]
pub struct WorldSdfClipmapPlan {
    owner: Arc<()>,
    base_revision: u64,
    target_revision: u64,
    center: BrickCoord,
    clipmap: SdfToroidalClipmap,
    update: SdfClipmapUpdate,
    dirty: Vec<BrickCoord>,
}

impl WorldSdfClipmapPlan {
    /// Returns the scheduler revision captured when this plan was prepared.
    pub const fn base_revision(&self) -> u64 {
        self.base_revision
    }

    /// Returns the revision that will be visible after this plan is committed.
    pub const fn target_revision(&self) -> u64 {
        self.target_revision
    }

    /// Returns the proposed base-level center.
    pub const fn center(&self) -> BrickCoord {
        self.center
    }

    /// Returns the entered and exited coordinates for every mip.
    pub const fn update(&self) -> &SdfClipmapUpdate {
        &self.update
    }

    /// Returns coordinates entering the proposed window, grouped by mip.
    pub fn entered(&self) -> &[Vec<BrickCoord>] {
        self.update.layers()
    }

    /// Returns coordinates exiting the proposed window, grouped by mip.
    pub fn exited(&self) -> &[Vec<BrickCoord>] {
        self.update.exited_layers()
    }

    /// Returns the dirty coordinates that will remain after commit.
    pub fn dirty_bricks(&self) -> &[BrickCoord] {
        &self.dirty
    }

    /// Returns whether this plan changes the center or emits any delta.
    pub fn is_noop(&self) -> bool {
        self.target_revision == self.base_revision && self.update.is_empty()
    }
}

/// The result of a successfully committed center roll.
#[derive(Debug)]
pub struct WorldSdfClipmapCommit {
    revision: u64,
    update: SdfClipmapUpdate,
}

impl WorldSdfClipmapCommit {
    /// Returns the scheduler revision after the commit.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the committed entered/exited delta.
    pub const fn update(&self) -> &SdfClipmapUpdate {
        &self.update
    }

    /// Returns coordinates entering the committed window, grouped by mip.
    pub fn entered(&self) -> &[Vec<BrickCoord>] {
        self.update.layers()
    }

    /// Returns coordinates exiting the committed window, grouped by mip.
    pub fn exited(&self) -> &[Vec<BrickCoord>] {
        self.update.exited_layers()
    }

    /// Consumes the commit result and returns its residency delta.
    pub fn into_update(self) -> SdfClipmapUpdate {
        self.update
    }
}

/// Bounded runtime scheduler for a toroidal SDF clipmap window.
///
/// Center rolls are two-phase: [`Self::plan_center`] prepares a complete
/// next state and [`Self::commit`] publishes it. Newly entered bricks are
/// marked dirty because no valid samples have been acknowledged for them.
/// Dirty coordinates that leave the window are dropped automatically.
#[derive(Debug)]
pub struct WorldSdfClipmapScheduler {
    owner: Arc<()>,
    config: WorldSdfClipmapConfig,
    clipmap: SdfToroidalClipmap,
    revision: u64,
    dirty: Vec<BrickCoord>,
}

impl WorldSdfClipmapScheduler {
    /// Creates an uninitialized bounded scheduler.
    pub fn new(config: WorldSdfClipmapConfig) -> Result<Self, WorldSdfClipmapError> {
        if config.max_dirty_bytes == 0 {
            return Err(WorldSdfClipmapError::InvalidDirtyBudget);
        }
        if config.max_mark_request_bytes == 0 {
            return Err(WorldSdfClipmapError::InvalidMarkRequestBudget);
        }
        let clipmap =
            SdfToroidalClipmap::new(config.clipmap).map_err(WorldSdfClipmapError::Clipmap)?;
        config.validate_dirty_count(0)?;
        Ok(Self {
            owner: Arc::new(()),
            config,
            clipmap,
            revision: 0,
            dirty: Vec::new(),
        })
    }

    /// Creates a scheduler and commits its first center roll.
    pub fn with_center(
        config: WorldSdfClipmapConfig,
        center: BrickCoord,
    ) -> Result<(Self, WorldSdfClipmapCommit), WorldSdfClipmapError> {
        let mut scheduler = Self::new(config)?;
        let plan = scheduler.plan_center(center)?;
        let commit = scheduler.commit(plan)?;
        Ok((scheduler, commit))
    }

    /// Returns the checked runtime configuration.
    pub const fn config(&self) -> WorldSdfClipmapConfig {
        self.config
    }

    /// Returns the current revision used by center and dirty operations.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the committed base-level center, if initialized.
    pub const fn center(&self) -> Option<BrickCoord> {
        self.clipmap.center()
    }

    /// Returns the current dirty coordinates in deterministic order.
    pub fn dirty_bricks(&self) -> &[BrickCoord] {
        &self.dirty
    }

    /// Returns whether a coordinate is dirty in the current window.
    pub fn is_dirty(&self, coord: BrickCoord) -> bool {
        self.dirty.binary_search(&coord).is_ok()
    }

    /// Returns the stable toroidal slot for a coordinate in the current window.
    pub fn slot_index(&self, coord: BrickCoord) -> Option<usize> {
        self.clipmap.slot_index(coord)
    }

    /// Prepares a center roll against the scheduler's current revision.
    pub fn plan_center(
        &self,
        center: BrickCoord,
    ) -> Result<WorldSdfClipmapPlan, WorldSdfClipmapError> {
        let target_revision = if self.center() == Some(center) {
            self.revision
        } else {
            self.revision
                .checked_add(1)
                .ok_or(WorldSdfClipmapError::RevisionOverflow)?
        };

        let mut clipmap = self
            .clipmap
            .try_clone()
            .map_err(WorldSdfClipmapError::Clipmap)?;
        let update = clipmap
            .update_center(center)
            .map_err(WorldSdfClipmapError::Clipmap)?;

        let entered_count = update.layers().iter().try_fold(0_usize, |total, layer| {
            total
                .checked_add(layer.len())
                .ok_or(WorldSdfClipmapError::ArithmeticOverflow)
        })?;
        let retained_count = self
            .dirty
            .iter()
            .filter(|&&coord| clipmap.slot_index(coord).is_some())
            .count();
        let upper_bound = retained_count
            .checked_add(entered_count)
            .ok_or(WorldSdfClipmapError::ArithmeticOverflow)?;
        self.config.validate_dirty_count(upper_bound)?;

        let mut dirty = Vec::new();
        dirty
            .try_reserve_exact(upper_bound)
            .map_err(|_| WorldSdfClipmapError::AllocationFailed)?;
        dirty.extend(
            self.dirty
                .iter()
                .copied()
                .filter(|&coord| clipmap.slot_index(coord).is_some()),
        );
        dirty.extend(update.layers().iter().flatten().copied());
        dirty.sort_unstable();
        dirty.dedup();
        self.config.validate_dirty_count(dirty.len())?;

        Ok(WorldSdfClipmapPlan {
            owner: Arc::clone(&self.owner),
            base_revision: self.revision,
            target_revision,
            center,
            clipmap,
            update,
            dirty,
        })
    }

    /// Prepares a center roll only when `expected_revision` is still current.
    pub fn plan_center_at(
        &self,
        expected_revision: u64,
        center: BrickCoord,
    ) -> Result<WorldSdfClipmapPlan, WorldSdfClipmapError> {
        self.check_revision(expected_revision)?;
        self.plan_center(center)
    }

    /// Commits a prepared center roll if its revision fence still matches.
    pub fn commit(
        &mut self,
        plan: WorldSdfClipmapPlan,
    ) -> Result<WorldSdfClipmapCommit, WorldSdfClipmapError> {
        if !Arc::ptr_eq(&self.owner, &plan.owner) {
            return Err(WorldSdfClipmapError::ForeignPlan);
        }
        self.check_revision(plan.base_revision)?;
        self.clipmap = plan.clipmap;
        self.dirty = plan.dirty;
        self.revision = plan.target_revision;
        Ok(WorldSdfClipmapCommit {
            revision: self.revision,
            update: plan.update,
        })
    }

    /// Plans and commits a center roll in one call.
    pub fn update_center(
        &mut self,
        center: BrickCoord,
    ) -> Result<WorldSdfClipmapCommit, WorldSdfClipmapError> {
        let plan = self.plan_center(center)?;
        self.commit(plan)
    }

    /// Marks current-window coordinates dirty and returns the number newly added.
    ///
    /// Every valid nonempty request advances the revision, including a request
    /// consisting entirely of already-dirty coordinates. A new edit must fence
    /// out results computed before it, even when list membership is unchanged.
    pub fn mark_dirty(&mut self, coords: &[BrickCoord]) -> Result<usize, WorldSdfClipmapError> {
        self.mark_dirty_at(self.revision, coords)
    }

    /// Marks current-window coordinates dirty behind a revision fence.
    ///
    /// Raw input is subject to the configured mark-request scratch limits before
    /// deduplication. Empty input is a no-op. Errors preserve all scheduler state.
    pub fn mark_dirty_at(
        &mut self,
        expected_revision: u64,
        coords: &[BrickCoord],
    ) -> Result<usize, WorldSdfClipmapError> {
        self.check_revision(expected_revision)?;
        self.config.validate_mark_request(coords.len())?;
        self.validate_coords(coords)?;
        if coords.is_empty() {
            return Ok(0);
        }
        let mut requested = Vec::new();
        requested
            .try_reserve_exact(coords.len())
            .map_err(|_| WorldSdfClipmapError::AllocationFailed)?;
        requested.extend_from_slice(coords);
        requested.sort_unstable();
        requested.dedup();
        let added = requested
            .iter()
            .filter(|coord| self.dirty.binary_search(coord).is_err())
            .count();
        let required = self
            .dirty
            .len()
            .checked_add(added)
            .ok_or(WorldSdfClipmapError::ArithmeticOverflow)?;
        self.config.validate_dirty_count(required)?;
        if added == 0 {
            self.bump_revision()?;
            return Ok(0);
        }

        let mut next = Vec::new();
        next.try_reserve_exact(required)
            .map_err(|_| WorldSdfClipmapError::AllocationFailed)?;
        let mut existing = self.dirty.iter().copied().peekable();
        for coord in requested {
            while existing.peek().is_some_and(|&old| old < coord) {
                // The peek above guarantees that this consumes an element.
                next.extend(existing.next());
            }
            if existing.peek() == Some(&coord) {
                existing.next();
            }
            next.push(coord);
        }
        next.extend(existing);
        self.bump_revision()?;
        self.dirty = next;
        Ok(added)
    }

    /// Clears current-window dirty coordinates and returns the number removed.
    pub fn clear_dirty(&mut self, coords: &[BrickCoord]) -> Result<usize, WorldSdfClipmapError> {
        self.clear_dirty_at(self.revision, coords)
    }

    /// Clears current-window dirty coordinates behind a revision fence.
    pub fn clear_dirty_at(
        &mut self,
        expected_revision: u64,
        coords: &[BrickCoord],
    ) -> Result<usize, WorldSdfClipmapError> {
        self.check_revision(expected_revision)?;
        self.validate_coords(coords)?;
        if coords.is_empty() || self.dirty.is_empty() {
            return Ok(0);
        }
        // Scratch depends only on the bounded dirty set, never the request
        // length. Repeated coordinates therefore cannot amplify allocation.
        let mut removals = Vec::new();
        removals
            .try_reserve_exact(self.dirty.len())
            .map_err(|_| WorldSdfClipmapError::AllocationFailed)?;
        removals.resize(self.dirty.len(), 0_u8);
        let mut removed = 0;
        for coord in coords {
            if let Ok(index) = self.dirty.binary_search(coord)
                && removals[index] == 0
            {
                removals[index] = 1;
                removed += 1;
            }
        }
        if removed == 0 {
            return Ok(0);
        }
        self.bump_revision()?;
        let mut index = 0;
        self.dirty.retain(|_| {
            let keep = removals[index] == 0;
            index += 1;
            keep
        });
        Ok(removed)
    }

    /// Removes and returns all dirty coordinates from the current window.
    ///
    /// This transfers work ownership; it is not an acknowledgement queue. The
    /// caller must re-mark still-resident coordinates if processing fails. Use
    /// `dirty_bricks` plus `clear_dirty_at` after successful work to leave work
    /// pending until an acknowledgement passes the revision fence.
    pub fn take_dirty(&mut self) -> Result<Vec<BrickCoord>, WorldSdfClipmapError> {
        if self.dirty.is_empty() {
            return Ok(Vec::new());
        }
        self.bump_revision()?;
        Ok(core::mem::take(&mut self.dirty))
    }

    fn validate_coords(&self, coords: &[BrickCoord]) -> Result<(), WorldSdfClipmapError> {
        for &coord in coords {
            if self.clipmap.slot_index(coord).is_none() {
                return Err(WorldSdfClipmapError::DirtyOutsideWindow { coord });
            }
        }
        Ok(())
    }

    fn check_revision(&self, expected: u64) -> Result<(), WorldSdfClipmapError> {
        if expected == self.revision {
            Ok(())
        } else {
            Err(WorldSdfClipmapError::RevisionConflict {
                expected,
                actual: self.revision,
            })
        }
    }

    fn bump_revision(&mut self) -> Result<(), WorldSdfClipmapError> {
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or(WorldSdfClipmapError::RevisionOverflow)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(max_dirty_bricks: usize, max_dirty_bytes: usize) -> WorldSdfClipmapConfig {
        let clipmap = SdfClipmapConfig::new(2, 1, 16 * 1024).unwrap();
        WorldSdfClipmapConfig::new(clipmap, max_dirty_bricks, max_dirty_bytes).unwrap()
    }

    fn center(x: i32) -> BrickCoord {
        BrickCoord::new(x, 0, 0, 0)
    }

    #[test]
    fn center_roll_is_transactional_and_marks_entered_bricks_dirty() {
        let mut scheduler = WorldSdfClipmapScheduler::new(config(
            32,
            32 * WorldSdfClipmapConfig::dirty_coord_bytes(),
        ))
        .unwrap();

        let plan = scheduler.plan_center(center(0)).unwrap();
        assert_eq!(scheduler.center(), None);
        assert_eq!(scheduler.revision(), 0);
        assert_eq!(plan.entered().iter().map(Vec::len).sum::<usize>(), 18);
        assert!(plan.exited().iter().all(Vec::is_empty));
        assert_eq!(plan.dirty_bricks().len(), 18);

        let commit = scheduler.commit(plan).unwrap();
        assert_eq!(commit.revision(), 1);
        assert_eq!(scheduler.center(), Some(center(0)));
        assert_eq!(scheduler.dirty_bricks().len(), 18);
    }

    #[test]
    fn roll_drops_exited_dirty_bricks_and_retains_current_dirty_bricks() {
        let mut scheduler = WorldSdfClipmapScheduler::with_center(
            config(32, 32 * WorldSdfClipmapConfig::dirty_coord_bytes()),
            center(0),
        )
        .unwrap()
        .0;
        let initial_dirty: Vec<_> = scheduler.dirty_bricks().to_vec();
        scheduler.clear_dirty(&initial_dirty).unwrap();
        let retained = BrickCoord::new(0, 0, 0, 0);
        scheduler.mark_dirty(&[retained]).unwrap();

        let plan = scheduler.plan_center(center(1)).unwrap();
        assert!(plan
            .exited()
            .iter()
            .flatten()
            .any(|&coord| { coord == BrickCoord::new(-1, 0, -1, 0) }));
        assert!(plan.dirty_bricks().contains(&retained));
        assert!(!plan.dirty_bricks().contains(&BrickCoord::new(-1, 0, -1, 0)));
        assert!(plan
            .entered()
            .iter()
            .flatten()
            .all(|coord| plan.dirty_bricks().contains(coord)));

        scheduler.commit(plan).unwrap();
        assert!(!scheduler.is_dirty(BrickCoord::new(-1, 0, -1, 0)));
        assert!(scheduler.is_dirty(retained));
    }

    #[test]
    fn stale_center_plan_is_rejected_by_revision_fence() {
        let mut scheduler = WorldSdfClipmapScheduler::new(config(
            32,
            32 * WorldSdfClipmapConfig::dirty_coord_bytes(),
        ))
        .unwrap();
        let stale = scheduler.plan_center(center(0)).unwrap();
        scheduler.update_center(center(1)).unwrap();
        let error = scheduler.commit(stale).unwrap_err();
        assert_eq!(
            error,
            WorldSdfClipmapError::RevisionConflict {
                expected: 0,
                actual: 1,
            }
        );
        assert_eq!(scheduler.center(), Some(center(1)));
    }

    #[test]
    fn dirty_capacity_failure_leaves_scheduler_unchanged() {
        let mut scheduler =
            WorldSdfClipmapScheduler::new(config(0, WorldSdfClipmapConfig::dirty_coord_bytes()))
                .unwrap();
        let error = scheduler.update_center(center(0)).unwrap_err();
        assert!(matches!(
            error,
            WorldSdfClipmapError::DirtyCapacityExceeded {
                required: 18,
                capacity: 0
            }
        ));
        assert_eq!(scheduler.center(), None);
        assert_eq!(scheduler.revision(), 0);
        assert!(scheduler.dirty_bricks().is_empty());
    }

    #[test]
    fn explicit_dirty_operations_are_fenced_and_idempotent() {
        let mut scheduler = WorldSdfClipmapScheduler::with_center(
            config(32, 32 * WorldSdfClipmapConfig::dirty_coord_bytes()),
            center(0),
        )
        .unwrap()
        .0;
        let all: Vec<_> = scheduler.dirty_bricks().to_vec();
        scheduler
            .clear_dirty_at(scheduler.revision(), &all)
            .unwrap();
        let revision = scheduler.revision();
        let coord = BrickCoord::new(0, 0, 0, 0);
        assert_eq!(
            scheduler.mark_dirty_at(revision, &[coord, coord]).unwrap(),
            1
        );
        let current = scheduler.revision();
        assert_eq!(scheduler.mark_dirty_at(current, &[coord]).unwrap(), 0);
        assert!(matches!(
            scheduler.clear_dirty_at(revision, &[coord]),
            Err(WorldSdfClipmapError::RevisionConflict { .. })
        ));
    }

    #[test]
    fn invalid_dirty_budget_is_rejected_before_scheduler_creation() {
        let clipmap = SdfClipmapConfig::new(1, 0, 1024).unwrap();
        assert_eq!(
            WorldSdfClipmapConfig::new(clipmap, 1, 0).unwrap_err(),
            WorldSdfClipmapError::InvalidDirtyBudget
        );
        assert_eq!(
            WorldSdfClipmapScheduler::new(WorldSdfClipmapConfig {
                clipmap,
                max_dirty_bricks: 1,
                max_dirty_bytes: 0,
                max_mark_request_bricks: 1,
                max_mark_request_bytes: 16,
            })
            .unwrap_err(),
            WorldSdfClipmapError::InvalidDirtyBudget
        );
    }

    #[test]
    fn matching_revision_does_not_admit_another_schedulers_plan() {
        let config = config(32, 32 * WorldSdfClipmapConfig::dirty_coord_bytes());
        let source = WorldSdfClipmapScheduler::new(config).unwrap();
        let mut other = WorldSdfClipmapScheduler::new(config).unwrap();
        let plan = source.plan_center(center(1)).unwrap();
        assert!(matches!(
            other.commit(plan),
            Err(WorldSdfClipmapError::ForeignPlan)
        ));
        assert_eq!(other.center(), None);
        assert_eq!(other.revision(), 0);
        assert!(other.dirty_bricks().is_empty());

        // A move preserves identity; a newly allocated scheduler does not.
        let plan = source.plan_center(center(0)).unwrap();
        let mut moved = source;
        moved.commit(plan).unwrap();
        let plan = moved.plan_center(center(1)).unwrap();
        drop(moved);
        assert!(matches!(
            other.commit(plan),
            Err(WorldSdfClipmapError::ForeignPlan)
        ));
    }

    #[test]
    fn repeated_dirty_edit_fences_stale_acknowledgements_and_plans() {
        let mut scheduler = WorldSdfClipmapScheduler::with_center(
            config(32, 32 * WorldSdfClipmapConfig::dirty_coord_bytes()),
            center(0),
        )
        .unwrap()
        .0;
        let coord = center(0);
        let revision = scheduler.revision();
        let plan = scheduler.plan_center(center(1)).unwrap();
        assert_eq!(
            scheduler.mark_dirty_at(revision, &[coord, coord]).unwrap(),
            0
        );
        assert_eq!(scheduler.revision(), revision + 1);
        assert!(matches!(
            scheduler.clear_dirty_at(revision, &[coord]),
            Err(WorldSdfClipmapError::RevisionConflict { .. })
        ));
        assert!(matches!(
            scheduler.commit(plan),
            Err(WorldSdfClipmapError::RevisionConflict { .. })
        ));
        assert!(scheduler.is_dirty(coord));
        assert_eq!(scheduler.clear_dirty(&[coord, coord]).unwrap(), 1);
        let revision = scheduler.revision();
        assert_eq!(scheduler.mark_dirty(&[]).unwrap(), 0);
        assert_eq!(scheduler.clear_dirty(&[coord]).unwrap(), 0);
        assert_eq!(scheduler.revision(), revision);
    }

    #[test]
    fn clear_duplicates_use_dirty_bounded_scratch_and_validate_all_input() {
        let clipmap = SdfClipmapConfig::new(1, 0, 1024).unwrap();
        let config =
            WorldSdfClipmapConfig::new(clipmap, 1, WorldSdfClipmapConfig::dirty_coord_bytes())
                .unwrap();
        let mut scheduler = WorldSdfClipmapScheduler::with_center(config, center(0))
            .unwrap()
            .0;
        let mut request = alloc::vec![center(0); 100_000];
        request.push(center(1));
        let revision = scheduler.revision();
        assert_eq!(
            scheduler.clear_dirty(&request),
            Err(WorldSdfClipmapError::DirtyOutsideWindow { coord: center(1) })
        );
        assert_eq!(scheduler.revision(), revision);
        assert!(scheduler.is_dirty(center(0)));
        request.pop();
        assert_eq!(scheduler.clear_dirty(&request).unwrap(), 1);
        assert!(scheduler.dirty_bricks().is_empty());
    }

    #[test]
    fn bulk_mark_merges_reverse_order_and_duplicates_with_retained_entries() {
        let clipmap = SdfClipmapConfig::new(1, 64, 2 * 1024 * 1024).unwrap();
        let count = 129 * 129;
        let config = WorldSdfClipmapConfig::new(
            clipmap,
            count,
            count * WorldSdfClipmapConfig::dirty_coord_bytes(),
        )
        .unwrap()
        .with_mark_request_budget(
            count * 2,
            count * 2 * WorldSdfClipmapConfig::dirty_coord_bytes(),
        )
        .unwrap();
        let mut scheduler = WorldSdfClipmapScheduler::with_center(config, center(0))
            .unwrap()
            .0;
        let all = scheduler.take_dirty().unwrap();
        assert_eq!(all.len(), count);
        let retained: Vec<_> = all.iter().step_by(3).copied().collect();
        scheduler.mark_dirty(&retained).unwrap();
        let mut request: Vec<_> = all.iter().rev().copied().collect();
        request.extend_from_slice(&all);
        assert_eq!(
            scheduler.mark_dirty(&request).unwrap(),
            count - retained.len()
        );
        assert_eq!(scheduler.dirty_bricks(), all);
    }

    #[test]
    fn mark_request_limits_precede_allocation_and_do_not_change_state() {
        let clipmap = SdfClipmapConfig::new(1, 0, 1024).unwrap();
        let bytes = WorldSdfClipmapConfig::dirty_coord_bytes();
        let base = WorldSdfClipmapConfig::new(clipmap, 1, bytes).unwrap();
        let mut scheduler = WorldSdfClipmapScheduler::with_center(base, center(0))
            .unwrap()
            .0;
        let revision = scheduler.revision();
        assert_eq!(
            scheduler.mark_dirty(&[center(0); 2]),
            Err(WorldSdfClipmapError::MarkRequestCapacityExceeded {
                required: 2,
                capacity: 1
            })
        );
        assert_eq!(scheduler.revision(), revision);
        assert!(scheduler.is_dirty(center(0)));

        let config = base.with_mark_request_budget(2, bytes).unwrap();
        let mut scheduler = WorldSdfClipmapScheduler::with_center(config, center(0))
            .unwrap()
            .0;
        let revision = scheduler.revision();
        assert_eq!(
            scheduler.mark_dirty(&[center(0); 2]),
            Err(WorldSdfClipmapError::MarkRequestBudgetExceeded {
                required: 2 * bytes,
                budget: bytes
            })
        );
        assert_eq!(scheduler.revision(), revision);
        let config = base.with_mark_request_budget(2, 2 * bytes).unwrap();
        let mut scheduler = WorldSdfClipmapScheduler::with_center(config, center(0))
            .unwrap()
            .0;
        assert_eq!(scheduler.mark_dirty(&[center(0); 2]).unwrap(), 0);
        assert_eq!(
            base.with_mark_request_budget(1, 0),
            Err(WorldSdfClipmapError::InvalidMarkRequestBudget)
        );
        assert_eq!(
            WorldSdfClipmapScheduler::new(WorldSdfClipmapConfig {
                max_mark_request_bytes: 0,
                ..base
            })
            .unwrap_err(),
            WorldSdfClipmapError::InvalidMarkRequestBudget
        );
    }

    #[test]
    fn dirty_byte_failure_and_invalid_mark_preserve_current_state() {
        let bytes = WorldSdfClipmapConfig::dirty_coord_bytes();
        let mut scheduler = WorldSdfClipmapScheduler::new(config(32, bytes)).unwrap();
        assert!(matches!(
            scheduler.update_center(center(0)),
            Err(WorldSdfClipmapError::DirtyBudgetExceeded { .. })
        ));
        assert_eq!(scheduler.revision(), 0);
        assert_eq!(scheduler.center(), None);

        let mut scheduler =
            WorldSdfClipmapScheduler::with_center(config(32, 32 * bytes), center(0))
                .unwrap()
                .0;
        let all = scheduler.take_dirty().unwrap();
        let revision = scheduler.revision();
        assert!(matches!(
            scheduler.mark_dirty(&[all[0], center(100)]),
            Err(WorldSdfClipmapError::DirtyOutsideWindow { .. })
        ));
        assert_eq!(scheduler.revision(), revision);
        assert!(scheduler.dirty_bricks().is_empty());
        scheduler.mark_dirty(&all).unwrap();
        assert_eq!(scheduler.dirty_bricks(), all);
    }

    #[test]
    fn revision_overflow_does_not_publish_new_dirty_coordinates() {
        let mut scheduler = WorldSdfClipmapScheduler::with_center(
            config(32, 32 * WorldSdfClipmapConfig::dirty_coord_bytes()),
            center(0),
        )
        .unwrap()
        .0;
        scheduler.take_dirty().unwrap();
        scheduler.revision = u64::MAX;
        assert_eq!(
            scheduler.mark_dirty(&[center(0)]),
            Err(WorldSdfClipmapError::RevisionOverflow)
        );
        assert!(scheduler.dirty_bricks().is_empty());
        assert_eq!(scheduler.center(), Some(center(0)));
        assert_eq!(scheduler.revision(), u64::MAX);
    }

    #[test]
    fn revision_overflow_keeps_dirty_membership_and_center_unchanged() {
        let mut scheduler = WorldSdfClipmapScheduler::with_center(
            config(32, 32 * WorldSdfClipmapConfig::dirty_coord_bytes()),
            center(0),
        )
        .unwrap()
        .0;
        scheduler.revision = u64::MAX;
        let dirty = scheduler.dirty_bricks().to_vec();
        assert_eq!(
            scheduler.mark_dirty(&[center(0)]),
            Err(WorldSdfClipmapError::RevisionOverflow)
        );
        assert_eq!(
            scheduler.clear_dirty(&[center(0)]),
            Err(WorldSdfClipmapError::RevisionOverflow)
        );
        assert_eq!(
            scheduler.take_dirty(),
            Err(WorldSdfClipmapError::RevisionOverflow)
        );
        assert!(matches!(
            scheduler.plan_center(center(1)),
            Err(WorldSdfClipmapError::RevisionOverflow)
        ));
        assert_eq!(scheduler.dirty_bricks(), dirty);
        assert_eq!(scheduler.center(), Some(center(0)));
        let plan = scheduler.plan_center(center(0)).unwrap();
        assert!(plan.is_noop());
        scheduler.commit(plan).unwrap();
        assert_eq!(scheduler.revision(), u64::MAX);
    }
}
