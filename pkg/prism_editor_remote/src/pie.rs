//! Play-In-Editor isolation primitives.
//!
//! A [`PieSession`] owns two independent snapshots: the immutable edit-state
//! baseline and a runtime working copy. Runtime mutations are applied only to
//! the working copy and are retained as an ordered log. Stopping a session
//! never mutates the edit baseline; a commit produces an explicit change set
//! which the editor may apply after its own revision check.

use crate::model::{apply_changes, Change, ChangeSet, SyncError, TargetId, WorldSnapshot};

/// Stable identifier for one PIE session.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct PieSessionId(u64);

impl PieSessionId {
    /// Creates an ID from its stable numeric representation.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the numeric representation.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Lifecycle state of a PIE session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieState {
    /// Runtime mutations are accepted.
    Running,
    /// Runtime is stopped and cannot receive more mutations.
    Stopped,
}

/// Deterministic policy selected when stopping a PIE session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieStopPolicy {
    /// Discard runtime mutations and leave the editor baseline unchanged.
    Rollback,
    /// Return runtime mutations as an explicit commit set; applying it to the
    /// editor remains a separate, revision-checked operation.
    Commit,
}

/// One ordered runtime mutation batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PieRuntimeRecord {
    /// Monotonic sequence number within the session.
    pub sequence: u64,
    /// Changes applied to the runtime working copy.
    pub changes: ChangeSet,
}

/// Result of stopping a PIE session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PieStopResult {
    /// Session that was stopped.
    pub session: PieSessionId,
    /// Selected stop policy.
    pub policy: PieStopPolicy,
    /// Edit baseline captured at start and never mutated by the session.
    pub edit_snapshot: WorldSnapshot,
    /// Final runtime-only snapshot.
    pub runtime_snapshot: WorldSnapshot,
    /// Ordered runtime records retained for diagnostics/replay.
    pub runtime_log: Vec<PieRuntimeRecord>,
    /// Explicit editor commit set. Present only for `Commit`.
    pub commit: Option<ChangeSet>,
}

impl PieStopResult {
    /// Applies an explicit commit to an editor snapshot after checking target
    /// and base revision. Rollback results cannot mutate an editor snapshot.
    pub fn apply_commit(&self, editor: &mut WorldSnapshot) -> Result<(), SyncError> {
        let commit = self
            .commit
            .as_ref()
            .ok_or_else(|| SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRequest,
                detail: String::from("the PIE stop result does not contain a commit"),
            })?;
        if editor.target != commit_target(self.edit_snapshot.target) {
            return Err(SyncError::TargetMismatch {
                expected: editor.target,
                actual: self.edit_snapshot.target,
            });
        }
        if editor.revision != commit.base_revision {
            return Err(SyncError::RevisionConflict {
                target: editor.target,
                expected: editor.revision,
                actual: commit.base_revision,
            });
        }
        let mut candidate = editor.clone();
        apply_changes(&mut candidate, &commit.changes)?;
        candidate.revision = commit.target_revision;
        *editor = candidate;
        Ok(())
    }
}

/// A PIE runtime isolated from the editor's live snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PieSession {
    session: PieSessionId,
    state: PieState,
    edit_snapshot: WorldSnapshot,
    runtime_snapshot: WorldSnapshot,
    runtime_log: Vec<PieRuntimeRecord>,
}

impl PieSession {
    /// Starts a session by cloning the edit snapshot into a private runtime
    /// working copy.
    #[must_use]
    pub fn start(session: PieSessionId, edit_snapshot: WorldSnapshot) -> Self {
        Self {
            session,
            state: PieState::Running,
            runtime_snapshot: edit_snapshot.clone(),
            edit_snapshot,
            runtime_log: Vec::new(),
        }
    }

    /// Returns the session ID.
    #[must_use]
    pub const fn session(&self) -> PieSessionId {
        self.session
    }

    /// Returns the lifecycle state.
    #[must_use]
    pub const fn state(&self) -> PieState {
        self.state
    }

    /// Returns the untouched edit baseline.
    #[must_use]
    pub const fn edit_snapshot(&self) -> &WorldSnapshot {
        &self.edit_snapshot
    }

    /// Returns the isolated runtime working copy.
    #[must_use]
    pub const fn runtime_snapshot(&self) -> &WorldSnapshot {
        &self.runtime_snapshot
    }

    /// Returns the ordered runtime mutation log.
    #[must_use]
    pub fn runtime_log(&self) -> &[PieRuntimeRecord] {
        &self.runtime_log
    }

    /// Applies a non-empty ordered mutation batch only to the runtime copy.
    pub fn apply_runtime_changes(&mut self, changes: Vec<Change>) -> Result<ChangeSet, SyncError> {
        if self.state != PieState::Running {
            return Err(SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRequest,
                detail: String::from("PIE session is already stopped"),
            });
        }
        if changes.is_empty() {
            return Err(SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRequest,
                detail: String::from("runtime change batches must not be empty"),
            });
        }
        let base_revision = self.runtime_snapshot.revision;
        let target_revision = base_revision.next();
        if target_revision == base_revision {
            return Err(SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRevisionRange,
                detail: String::from("runtime revision cannot advance past u64::MAX"),
            });
        }
        let mut candidate = self.runtime_snapshot.clone();
        apply_changes(&mut candidate, &changes)?;
        candidate.revision = target_revision;
        let change_set = ChangeSet::new(base_revision, target_revision, changes)?;
        let sequence = u64::try_from(self.runtime_log.len())
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRevisionRange,
                detail: String::from("PIE runtime log sequence exhausted"),
            })?;
        self.runtime_snapshot = candidate;
        self.runtime_log.push(PieRuntimeRecord {
            sequence,
            changes: change_set.clone(),
        });
        Ok(change_set)
    }

    /// Stops once using an explicit rollback or commit policy. The returned
    /// commit is data only; the edit snapshot inside this session is never
    /// modified.
    pub fn stop(&mut self, policy: PieStopPolicy) -> Result<PieStopResult, SyncError> {
        if self.state != PieState::Running {
            return Err(SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRequest,
                detail: String::from("PIE session is already stopped"),
            });
        }
        self.state = PieState::Stopped;
        let commit = match policy {
            PieStopPolicy::Rollback => None,
            PieStopPolicy::Commit => Some(self.commit_set()?),
        };
        Ok(PieStopResult {
            session: self.session,
            policy,
            edit_snapshot: self.edit_snapshot.clone(),
            runtime_snapshot: self.runtime_snapshot.clone(),
            runtime_log: self.runtime_log.clone(),
            commit,
        })
    }

    fn commit_set(&self) -> Result<ChangeSet, SyncError> {
        let mut changes = Vec::new();
        for record in &self.runtime_log {
            changes.extend(record.changes.changes.iter().cloned());
        }
        if changes.is_empty() {
            Ok(ChangeSet::empty(self.edit_snapshot.revision))
        } else {
            ChangeSet::new(
                self.edit_snapshot.revision,
                self.runtime_snapshot.revision,
                changes,
            )
        }
    }
}

fn commit_target(target: TargetId) -> TargetId {
    target
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComponentId, EntityId, EntityRecord, Revision, Value};

    fn snapshot() -> WorldSnapshot {
        let target = TargetId::from_raw(9);
        let mut snapshot = WorldSnapshot::empty(target);
        snapshot.revision = Revision::from_raw(4);
        snapshot
            .entities
            .insert(EntityId::from_raw(1), EntityRecord::new());
        snapshot
    }

    fn set_value(value: i64) -> Change {
        Change::SetComponent {
            entity: EntityId::from_raw(1),
            component: ComponentId::from_static("Value"),
            value: Value::Signed(value),
        }
    }

    #[test]
    fn runtime_changes_do_not_mutate_edit_snapshot() {
        let mut session = PieSession::start(PieSessionId::from_raw(1), snapshot());
        session
            .apply_runtime_changes(vec![set_value(7)])
            .expect("change");
        assert_eq!(session.edit_snapshot().revision, Revision::from_raw(4));
        assert_eq!(session.runtime_snapshot().revision, Revision::from_raw(5));
        assert!(!session.edit_snapshot().entities[&EntityId::from_raw(1)]
            .components
            .contains_key(&ComponentId::from_static("Value")));
    }

    #[test]
    fn runtime_log_is_ordered_and_replayable() {
        let mut session = PieSession::start(PieSessionId::from_raw(1), snapshot());
        session
            .apply_runtime_changes(vec![set_value(1)])
            .expect("first");
        session
            .apply_runtime_changes(vec![set_value(2)])
            .expect("second");
        assert_eq!(session.runtime_log()[0].sequence, 1);
        assert_eq!(session.runtime_log()[1].sequence, 2);
        assert_eq!(
            session.runtime_snapshot().entities[&EntityId::from_raw(1)].components
                [&ComponentId::from_static("Value")],
            Value::Signed(2)
        );
    }

    #[test]
    fn rollback_discards_commit_without_touching_baseline() {
        let mut session = PieSession::start(PieSessionId::from_raw(1), snapshot());
        session
            .apply_runtime_changes(vec![set_value(7)])
            .expect("change");
        let result = session.stop(PieStopPolicy::Rollback).expect("stop");
        assert_eq!(result.commit, None);
        assert_eq!(result.edit_snapshot.revision, Revision::from_raw(4));
        assert_eq!(result.runtime_snapshot.revision, Revision::from_raw(5));
    }

    #[test]
    fn commit_returns_explicit_set_without_implicit_edit_mutation() {
        let edit = snapshot();
        let mut session = PieSession::start(PieSessionId::from_raw(1), edit.clone());
        session
            .apply_runtime_changes(vec![set_value(7)])
            .expect("change");
        let result = session.stop(PieStopPolicy::Commit).expect("stop");
        assert_eq!(result.edit_snapshot, edit);
        assert_eq!(
            result.commit.as_ref().expect("commit").base_revision,
            Revision::from_raw(4)
        );
        assert_eq!(
            result.commit.as_ref().expect("commit").target_revision,
            Revision::from_raw(5)
        );
    }

    #[test]
    fn explicit_commit_updates_editor_only_when_applied() {
        let mut session = PieSession::start(PieSessionId::from_raw(1), snapshot());
        session
            .apply_runtime_changes(vec![set_value(7)])
            .expect("change");
        let result = session.stop(PieStopPolicy::Commit).expect("stop");
        let mut editor = snapshot();
        result.apply_commit(&mut editor).expect("apply");
        assert_eq!(editor.revision, Revision::from_raw(5));
        assert_eq!(
            editor.entities[&EntityId::from_raw(1)].components[&ComponentId::from_static("Value")],
            Value::Signed(7)
        );
    }

    #[test]
    fn commit_rejects_editor_revision_conflict() {
        let mut session = PieSession::start(PieSessionId::from_raw(1), snapshot());
        session
            .apply_runtime_changes(vec![set_value(7)])
            .expect("change");
        let result = session.stop(PieStopPolicy::Commit).expect("stop");
        let mut editor = snapshot();
        editor.revision = Revision::from_raw(8);
        assert!(matches!(
            result.apply_commit(&mut editor),
            Err(SyncError::RevisionConflict { .. })
        ));
    }

    #[test]
    fn stopped_session_rejects_new_runtime_changes_and_second_stop() {
        let mut session = PieSession::start(PieSessionId::from_raw(1), snapshot());
        session.stop(PieStopPolicy::Rollback).expect("stop");
        assert!(session.apply_runtime_changes(vec![set_value(1)]).is_err());
        assert!(session.stop(PieStopPolicy::Rollback).is_err());
    }

    #[test]
    fn invalid_runtime_batch_is_atomic() {
        let mut session = PieSession::start(PieSessionId::from_raw(1), snapshot());
        let error = session
            .apply_runtime_changes(vec![Change::DeleteEntity {
                entity: EntityId::from_raw(99),
            }])
            .expect_err("missing entity");
        assert!(matches!(error, SyncError::MissingEntity { .. }));
        assert_eq!(session.runtime_snapshot(), session.edit_snapshot());
        assert!(session.runtime_log().is_empty());
    }
}
