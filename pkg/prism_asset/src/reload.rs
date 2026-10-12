//! Epoch-based reload result arbitration.
//!
//! This small component is usable from the `no_std + alloc` core. A reload
//! captures a [`ReloadCommitGuard`] when it starts. Only the guard for the
//! latest epoch may run its commit closure; the closure is serialized with
//! starting newer epochs so a newer reload cannot begin between validation
//! and applying the result.

use alloc::sync::Arc;
use core::hint::spin_loop;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

struct EpochState {
    epoch: AtomicU64,
    gate: AtomicBool,
}

/// Shared monotonically advancing reload epoch.
///
/// Clone this value into load tasks that may complete independently. Starting
/// a newer reload supersedes every guard created for an earlier epoch.
#[derive(Clone)]
pub struct ReloadEpoch {
    state: Arc<EpochState>,
}

/// The decision made when a reload result tries to commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReloadCommit<T> {
    /// The guard was current and its commit closure was applied.
    Committed(T),
    /// A newer reload started first, so the closure was not run.
    Superseded,
}

/// A capability to commit a result only if its captured epoch is still latest.
#[derive(Clone)]
pub struct ReloadCommitGuard {
    state: Arc<EpochState>,
    /// Cloned guards represent the same reload attempt and must not be able
    /// to apply its staged result more than once.
    committed: Arc<AtomicBool>,
    epoch: u64,
    valid: bool,
}

impl ReloadEpoch {
    /// Creates an epoch source with no reloads started yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(EpochState {
                epoch: AtomicU64::new(0),
                gate: AtomicBool::new(false),
            }),
        }
    }

    /// Starts a reload and returns its commit capability.
    #[must_use]
    pub fn begin(&self) -> ReloadCommitGuard {
        self.try_begin().unwrap_or_else(|| ReloadCommitGuard {
            state: Arc::clone(&self.state),
            committed: Arc::new(AtomicBool::new(false)),
            epoch: 0,
            valid: false,
        })
    }

    /// Starts a reload, returning `None` after the finite epoch space is
    /// exhausted. Callers that can surface a reload error should prefer this
    /// method; [`ReloadEpoch::begin`] remains a compatibility convenience and
    /// returns a guard that can only be superseded in that case.
    #[must_use]
    pub fn try_begin(&self) -> Option<ReloadCommitGuard> {
        let _gate = self.lock();
        let current = self.state.epoch.load(Ordering::Relaxed);
        let epoch = current.checked_add(1)?;
        self.state.epoch.store(epoch, Ordering::Release);
        Some(ReloadCommitGuard {
            state: Arc::clone(&self.state),
            committed: Arc::new(AtomicBool::new(false)),
            epoch,
            valid: true,
        })
    }

    /// Returns the latest started epoch.
    #[must_use]
    pub fn current(&self) -> u64 {
        self.state.epoch.load(Ordering::Acquire)
    }

    fn lock(&self) -> EpochLock<'_> {
        while self
            .state
            .gate
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        EpochLock { state: &self.state }
    }
}

impl Default for ReloadEpoch {
    fn default() -> Self {
        Self::new()
    }
}

impl ReloadCommitGuard {
    /// The epoch captured when this reload began.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Applies `commit` only if this guard still represents the latest epoch
    /// and no clone of this guard has already committed.
    ///
    /// The closure runs while epoch advancement is excluded. Keep it short and
    /// do not call [`ReloadEpoch::begin`] on this same epoch from the closure.
    pub fn commit<T>(&self, commit: impl FnOnce() -> T) -> ReloadCommit<T> {
        let _gate = EpochLock::acquire(&self.state);
        if !self.valid || self.state.epoch.load(Ordering::Relaxed) != self.epoch {
            return ReloadCommit::Superseded;
        }
        if self.committed.swap(true, Ordering::Relaxed) {
            return ReloadCommit::Superseded;
        }
        ReloadCommit::Committed(commit())
    }
}

struct EpochLock<'a> {
    state: &'a EpochState,
}

impl<'a> EpochLock<'a> {
    fn acquire(state: &'a EpochState) -> Self {
        while state
            .gate
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            spin_loop();
        }
        Self { state }
    }
}

impl Drop for EpochLock<'_> {
    fn drop(&mut self) {
        self.state.gate.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::{ReloadCommit, ReloadEpoch};
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn older_epoch_is_superseded_without_running_commit() {
        let epochs = ReloadEpoch::new();
        let old = epochs.begin();
        let latest = epochs.begin();
        let effects = AtomicUsize::new(0);

        assert_eq!(
            old.commit(|| effects.fetch_add(1, Ordering::Relaxed)),
            ReloadCommit::Superseded
        );
        assert_eq!(
            latest.commit(|| effects.fetch_add(1, Ordering::Relaxed)),
            ReloadCommit::Committed(0)
        );
        assert_eq!(effects.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn cloned_epoch_sources_share_superseding_state() {
        let epochs = ReloadEpoch::new();
        let other = epochs.clone();
        let guard = epochs.begin();
        let next = other.begin();

        assert!(next.epoch() > guard.epoch());
        assert_eq!(guard.commit(|| ()), ReloadCommit::Superseded);
        assert_eq!(epochs.current(), next.epoch());
    }

    #[test]
    fn cloned_commit_guard_can_apply_staged_result_only_once() {
        let epochs = ReloadEpoch::new();
        let guard = epochs.begin();
        let duplicate = guard.clone();
        let effects = AtomicUsize::new(0);

        assert_eq!(
            guard.commit(|| effects.fetch_add(1, Ordering::Relaxed)),
            ReloadCommit::Committed(0)
        );
        assert_eq!(
            duplicate.commit(|| effects.fetch_add(1, Ordering::Relaxed)),
            ReloadCommit::Superseded
        );
        assert_eq!(effects.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn exhausted_epoch_space_cannot_wrap_into_a_committable_result() {
        let epochs = ReloadEpoch::new();
        epochs.state.epoch.store(u64::MAX, Ordering::Relaxed);
        assert!(epochs.try_begin().is_none());
        assert_eq!(epochs.begin().commit(|| 1), ReloadCommit::Superseded);
        assert_eq!(epochs.current(), u64::MAX);
    }
}
