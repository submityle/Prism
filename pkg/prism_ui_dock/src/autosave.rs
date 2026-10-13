//! Host-driven workspace autosave scheduling.
//!
//! This module deliberately does not spawn a thread or perform work from a
//! reducer callback. The host supplies a monotonic [`Instant`] and chooses a
//! worker/executor boundary for [`WorkspaceAutosave::poll`] or `flush`.

use std::time::{Duration, Instant};

use crate::{WorkspaceFileStore, WorkspaceManager, WorkspaceStoreError};

/// Result of polling a host-driven autosave controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AutosavePoll {
    /// No unsaved workspace mutation is pending.
    Clean,
    /// A mutation is pending but its debounce deadline has not arrived.
    Waiting {
        /// Monotonic instant at which the next save becomes eligible.
        deadline: Instant,
    },
    /// A pending snapshot was durably committed.
    Saved,
}

/// Debounced workspace snapshot scheduler backed by [`WorkspaceFileStore`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceAutosave {
    store: WorkspaceFileStore,
    throttle: Duration,
    deadline: Option<Instant>,
}

impl WorkspaceAutosave {
    /// Creates a controller with a debounce duration.
    #[must_use]
    pub fn new(store: WorkspaceFileStore, throttle: Duration) -> Self {
        Self {
            store,
            throttle,
            deadline: None,
        }
    }

    /// Returns the configured debounce duration.
    #[must_use]
    pub const fn throttle(&self) -> Duration {
        self.throttle
    }

    /// Returns the next save deadline, if the workspace is dirty.
    #[must_use]
    pub const fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Returns whether a save is pending.
    #[must_use]
    pub const fn is_dirty(&self) -> bool {
        self.deadline.is_some()
    }

    /// Marks the workspace dirty and resets the debounce deadline.
    pub fn mark_dirty(&mut self, now: Instant) {
        self.deadline = Some(now + self.throttle);
    }

    /// Polls the debounce deadline and saves once the host-provided instant is
    /// due. A failed save leaves the controller dirty for retry.
    pub fn poll(
        &mut self,
        now: Instant,
        manager: &WorkspaceManager,
    ) -> Result<AutosavePoll, WorkspaceStoreError> {
        let Some(deadline) = self.deadline else {
            return Ok(AutosavePoll::Clean);
        };
        if now < deadline {
            return Ok(AutosavePoll::Waiting { deadline });
        }
        self.store.save(manager)?;
        self.deadline = None;
        Ok(AutosavePoll::Saved)
    }

    /// Saves immediately when dirty, bypassing the debounce deadline.
    ///
    /// A failed write keeps the dirty marker so a host can retry or surface the
    /// error without losing the pending snapshot contract.
    pub fn flush(
        &mut self,
        manager: &WorkspaceManager,
    ) -> Result<AutosavePoll, WorkspaceStoreError> {
        if self.deadline.is_none() {
            return Ok(AutosavePoll::Clean);
        }
        self.store.save(manager)?;
        self.deadline = None;
        Ok(AutosavePoll::Saved)
    }

    /// Returns the underlying crash-safe file store.
    #[must_use]
    pub const fn store(&self) -> &WorkspaceFileStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DockLayout, WorkspaceLoadSource};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture() -> (WorkspaceAutosave, WorkspaceManager, PathBuf) {
        let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!("prism-dock-autosave-{id}"));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("workspace.prism");
        let store = WorkspaceFileStore::new(&path);
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("coding").unwrap();
        manager
            .add_perspective("default", &DockLayout::default())
            .unwrap();
        (
            WorkspaceAutosave::new(store, Duration::from_millis(500)),
            manager,
            directory,
        )
    }

    #[test]
    fn debounce_waits_then_commits_and_clears_dirty_state() {
        let (mut autosave, manager, directory) = fixture();
        let start = Instant::now();
        autosave.mark_dirty(start);
        assert!(matches!(
            autosave.poll(start + Duration::from_millis(499), &manager),
            Ok(AutosavePoll::Waiting { .. })
        ));
        assert!(autosave.is_dirty());
        assert!(matches!(
            autosave.poll(start + Duration::from_millis(500), &manager),
            Ok(AutosavePoll::Saved)
        ));
        assert!(!autosave.is_dirty());
        let loaded = autosave
            .store()
            .load(Default::default(), WorkspaceManager::new());
        assert_eq!(loaded.source, WorkspaceLoadSource::Primary);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn flush_bypasses_deadline_and_clean_flush_is_a_noop() {
        let (mut autosave, manager, directory) = fixture();
        assert!(matches!(autosave.flush(&manager), Ok(AutosavePoll::Clean)));
        autosave.mark_dirty(Instant::now());
        assert!(matches!(autosave.flush(&manager), Ok(AutosavePoll::Saved)));
        assert!(!autosave.is_dirty());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn failed_save_keeps_dirty_marker_for_retry() {
        let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!("prism-dock-save-error-{id}"));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("workspace.prism");
        fs::create_dir_all(&path).unwrap();
        let mut autosave = WorkspaceAutosave::new(WorkspaceFileStore::new(path), Duration::ZERO);
        let manager = WorkspaceManager::new();
        autosave.mark_dirty(Instant::now());
        assert!(autosave.poll(Instant::now(), &manager).is_err());
        assert!(autosave.is_dirty());
        fs::remove_dir_all(directory).unwrap();
    }
}
