//! Bounded undo/redo history for deterministic dock actions.

use crate::{reduce, Action, DockLayout, LayoutError};
use alloc::vec::Vec;

/// Failure returned by [`DockHistory`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HistoryError {
    /// The action was rejected by the layout reducer.
    Layout(LayoutError),
    /// No older state is available.
    NothingToUndo,
    /// No newer state is available.
    NothingToRedo,
    /// A history with zero capacity cannot retain a state transition.
    ZeroCapacity,
}

/// Bounded, snapshot-based undo/redo for dock layouts.
///
/// Snapshots are owned and validated by the reducer. A new action clears the
/// redo branch; undo and redo never call factories, touch windows, or perform
/// IO, so the history is safe to use from an editor command queue.
#[derive(Clone, Debug)]
pub struct DockHistory {
    capacity: usize,
    undo: Vec<DockLayout>,
    redo: Vec<DockLayout>,
}

impl DockHistory {
    /// Creates a history that retains at most `capacity` previous states.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// Returns the configured snapshot capacity.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns whether an undo operation is available.
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// Returns whether a redo operation is available.
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Number of retained undo snapshots.
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Number of retained redo snapshots.
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Applies `action`, recording the previous state on the undo stack.
    pub fn execute(
        &mut self,
        current: &DockLayout,
        action: Action,
    ) -> Result<DockLayout, HistoryError> {
        if self.capacity == 0 {
            return Err(HistoryError::ZeroCapacity);
        }
        let next = reduce(current, action).map_err(HistoryError::Layout)?;
        if next == *current {
            return Ok(next);
        }
        Self::push_bounded(self.capacity, &mut self.undo, current.clone());
        self.redo.clear();
        Ok(next)
    }

    /// Reverts to the most recent previous state.
    pub fn undo(&mut self, current: &DockLayout) -> Result<DockLayout, HistoryError> {
        let previous = self.undo.pop().ok_or(HistoryError::NothingToUndo)?;
        Self::push_bounded(self.capacity, &mut self.redo, current.clone());
        Ok(previous)
    }

    /// Reapplies the most recently undone state.
    pub fn redo(&mut self, current: &DockLayout) -> Result<DockLayout, HistoryError> {
        let next = self.redo.pop().ok_or(HistoryError::NothingToRedo)?;
        Self::push_bounded(self.capacity, &mut self.undo, current.clone());
        Ok(next)
    }

    /// Clears both branches while retaining the configured capacity.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    fn push_bounded(capacity: usize, stack: &mut Vec<DockLayout>, value: DockLayout) {
        stack.push(value);
        if stack.len() > capacity {
            stack.remove(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DockNode, Rect};

    fn initial() -> DockLayout {
        DockLayout {
            root: DockNode::panel("one"),
            ..DockLayout::default()
        }
    }

    #[test]
    fn execute_undo_redo_is_transactional() {
        let mut history = DockHistory::new(8);
        let first = initial();
        let second = history
            .execute(
                &first,
                Action::Float {
                    panel: "one".into(),
                    rect: Rect {
                        x: 1,
                        y: 2,
                        width: 320,
                        height: 200,
                    },
                },
            )
            .unwrap();
        assert!(history.can_undo());
        let restored = history.undo(&second).unwrap();
        assert_eq!(restored, first);
        let redone = history.redo(&restored).unwrap();
        assert_eq!(redone, second);
    }

    #[test]
    fn new_branch_clears_redo_and_capacity_is_bounded() {
        let mut history = DockHistory::new(1);
        let first = initial();
        let second = history
            .execute(
                &first,
                Action::Float {
                    panel: "one".into(),
                    rect: Rect {
                        x: 0,
                        y: 0,
                        width: 10,
                        height: 10,
                    },
                },
            )
            .unwrap();
        let third = history.undo(&second).unwrap();
        assert_eq!(third, first);
        let _ = history
            .execute(
                &third,
                Action::Float {
                    panel: "one".into(),
                    rect: Rect {
                        x: 4,
                        y: 5,
                        width: 10,
                        height: 10,
                    },
                },
            )
            .unwrap();
        assert!(!history.can_redo());
        assert_eq!(history.undo_len(), 1);
    }
}
