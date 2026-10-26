//! Durable live-upgrade coordinator: shadow, deterministic compare, cutover and rollback.

use crate::{
    DurableMutation, DurableRecord, DurableStore, ExecutionEvent, ExecutionReplay, RecordKey,
    ReplayDivergence, RevisionGuard, VsError, VsResult,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpgradeStatus {
    Shadowing,
    Diverged,
    Draining,
    Promoted,
    RolledBack,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpgradeRecord {
    pub run_id: String,
    pub from_revision: String,
    pub to_revision: String,
    pub status: UpgradeStatus,
    pub expected: Vec<ExecutionEvent>,
    pub shadow: Vec<ExecutionEvent>,
    pub divergence: Option<ReplayDivergence>,
}

/// Host-provided runner for the candidate module. Implementations can invoke
/// a `WorkflowVm`, a remote worker, or a host-owned WASM sandbox.
pub trait ShadowExecutor: Send + Sync {
    fn run(&self, revision: &str, input: &[u8]) -> VsResult<Vec<ExecutionEvent>>;
}

pub struct UpgradeCoordinator {
    store: Arc<dyn DurableStore>,
}

impl UpgradeCoordinator {
    pub fn new(store: Arc<dyn DurableStore>) -> Self {
        Self { store }
    }
    fn key(run_id: &str) -> VsResult<RecordKey> {
        RecordKey::new("upgrades", run_id)
    }
    pub fn begin(
        &self,
        run_id: &str,
        from_revision: &str,
        to_revision: &str,
        expected: Vec<ExecutionEvent>,
    ) -> VsResult<DurableRecord> {
        if run_id.is_empty() || from_revision.is_empty() || to_revision.is_empty() {
            return Err(VsError::Validate(
                "upgrade identities must not be empty".into(),
            ));
        }
        ExecutionReplay::validate(&expected, run_id)?;
        let payload = UpgradeRecord {
            run_id: run_id.into(),
            from_revision: from_revision.into(),
            to_revision: to_revision.into(),
            status: UpgradeStatus::Shadowing,
            expected,
            shadow: Vec::new(),
            divergence: None,
        };
        self.store.commit(
            &Self::key(run_id)?,
            RevisionGuard::Create,
            DurableMutation::default().with_payload(&payload)?,
        )
    }
    pub fn load(&self, run_id: &str) -> VsResult<Option<(DurableRecord, UpgradeRecord)>> {
        let key = Self::key(run_id)?;
        let Some(record) = self.store.load(&key)? else {
            return Ok(None);
        };
        Ok(Some((record.clone(), record.decode()?)))
    }
    pub fn record_shadow(
        &self,
        run_id: &str,
        expected_revision: u64,
        events: Vec<ExecutionEvent>,
    ) -> VsResult<DurableRecord> {
        let key = Self::key(run_id)?;
        let Some(current) = self.store.load(&key)? else {
            return Err(VsError::Runtime("unknown upgrade".into()));
        };
        let mut payload: UpgradeRecord = current.decode()?;
        if !matches!(
            payload.status,
            UpgradeStatus::Shadowing | UpgradeStatus::Diverged
        ) {
            return Err(VsError::Runtime("upgrade is not shadowing".into()));
        }
        payload.shadow.extend(events);
        ExecutionReplay::validate(&payload.shadow, &payload.run_id)?;
        payload.divergence = ExecutionReplay::compare(&payload.expected, &payload.shadow)?;
        payload.status = if payload.divergence.is_some() {
            UpgradeStatus::Diverged
        } else {
            UpgradeStatus::Shadowing
        };
        self.store.commit(
            &key,
            RevisionGuard::Match(expected_revision),
            DurableMutation::default().with_payload(&payload)?,
        )
    }

    /// Execute the candidate revision and atomically persist its comparison.
    pub fn run_shadow<E: ShadowExecutor>(
        &self,
        run_id: &str,
        expected_revision: u64,
        input: &[u8],
        executor: &E,
    ) -> VsResult<DurableRecord> {
        let (_, upgrade) = self
            .load(run_id)?
            .ok_or_else(|| VsError::Runtime("unknown upgrade".into()))?;
        let events = executor.run(&upgrade.to_revision, input)?;
        self.record_shadow(run_id, expected_revision, events)
    }
    pub fn promote(&self, run_id: &str, expected_revision: u64) -> VsResult<DurableRecord> {
        self.transition(
            run_id,
            expected_revision,
            UpgradeStatus::Promoted,
            &[UpgradeStatus::Shadowing],
        )
    }
    pub fn drain(&self, run_id: &str, expected_revision: u64) -> VsResult<DurableRecord> {
        self.transition(
            run_id,
            expected_revision,
            UpgradeStatus::Draining,
            &[UpgradeStatus::Shadowing],
        )
    }
    pub fn rollback(&self, run_id: &str, expected_revision: u64) -> VsResult<DurableRecord> {
        self.transition(
            run_id,
            expected_revision,
            UpgradeStatus::RolledBack,
            &[
                UpgradeStatus::Shadowing,
                UpgradeStatus::Diverged,
                UpgradeStatus::Draining,
            ],
        )
    }
    fn transition(
        &self,
        run_id: &str,
        expected_revision: u64,
        status: UpgradeStatus,
        allowed: &[UpgradeStatus],
    ) -> VsResult<DurableRecord> {
        let key = Self::key(run_id)?;
        let Some(current) = self.store.load(&key)? else {
            return Err(VsError::Runtime("unknown upgrade".into()));
        };
        let mut payload: UpgradeRecord = current.decode()?;
        if !allowed.contains(&payload.status) {
            return Err(VsError::Runtime(format!(
                "invalid upgrade transition from {:?}",
                payload.status
            )));
        }
        payload.status = status;
        self.store.commit(
            &key,
            RevisionGuard::Match(expected_revision),
            DurableMutation::default().with_payload(&payload)?,
        )
    }
}
