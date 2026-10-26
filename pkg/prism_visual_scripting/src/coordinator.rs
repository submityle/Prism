//! Durable run coordination and checkpoint revision control.

use crate::{
    DurableMutation, DurableRecord, DurableStore, RecordKey, RevisionGuard, VsError, VsResult,
};
use crate::{MigrationRegistry, Value};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VersionedCheckpoint {
    pub schema_version: u32,
    pub module_revision: String,
    pub migration_digest: [u8; 32],
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Created,
    Running,
    Suspended,
    Completed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    pub execution_id: String,
    pub status: RunStatus,
    pub checkpoint: Vec<u8>,
}

pub struct RunCoordinator {
    store: Arc<dyn DurableStore>,
}

impl RunCoordinator {
    pub fn new(store: Arc<dyn DurableStore>) -> Self {
        Self { store }
    }

    pub fn key(run_id: &str) -> VsResult<RecordKey> {
        RecordKey::new("runs", run_id)
    }

    pub fn start(
        &self,
        run_id: &str,
        execution_id: &str,
        checkpoint: Vec<u8>,
    ) -> VsResult<DurableRecord> {
        if execution_id.is_empty() {
            return Err(VsError::Validate(
                "execution identity must not be empty".into(),
            ));
        }
        let record = RunRecord {
            execution_id: execution_id.to_string(),
            status: RunStatus::Created,
            checkpoint,
        };
        self.store.commit(
            &Self::key(run_id)?,
            RevisionGuard::Create,
            DurableMutation::default().with_payload(&record)?,
        )
    }

    pub fn load(&self, run_id: &str) -> VsResult<Option<(DurableRecord, RunRecord)>> {
        let key = Self::key(run_id)?;
        let Some(record) = self.store.load(&key)? else {
            return Ok(None);
        };
        let payload = record.decode()?;
        Ok(Some((record, payload)))
    }

    pub fn checkpoint(
        &self,
        run_id: &str,
        expected_revision: u64,
        execution_id: &str,
        status: RunStatus,
        checkpoint: Vec<u8>,
    ) -> VsResult<DurableRecord> {
        let key = Self::key(run_id)?;
        let Some(current) = self.store.load(&key)? else {
            return Err(VsError::Runtime(format!("unknown run {run_id}")));
        };
        let current_payload: RunRecord = current.decode()?;
        if current_payload.execution_id != execution_id {
            return Err(VsError::Runtime("run execution identity mismatch".into()));
        }
        let payload = RunRecord {
            execution_id: execution_id.to_string(),
            status,
            checkpoint,
        };
        self.store.commit(
            &key,
            RevisionGuard::Match(expected_revision),
            DurableMutation::default().with_payload(&payload)?,
        )
    }

    /// Atomically migrate a JSON/Value checkpoint before committing the new
    /// schema and module revision. The CAS guard makes concurrent resume safe.
    #[allow(clippy::too_many_arguments)]
    pub fn checkpoint_migrated(
        &self,
        run_id: &str,
        expected_revision: u64,
        execution_id: &str,
        status: RunStatus,
        target_schema: u32,
        module_revision: &str,
        migrations: &MigrationRegistry,
    ) -> VsResult<DurableRecord> {
        let key = Self::key(run_id)?;
        let Some(current) = self.store.load(&key)? else {
            return Err(VsError::Runtime(format!("unknown run {run_id}")));
        };
        let current_payload: RunRecord = current.decode()?;
        if current_payload.execution_id != execution_id {
            return Err(VsError::Runtime("run execution identity mismatch".into()));
        }
        let checkpoint: VersionedCheckpoint =
            serde_json::from_slice(&current_payload.checkpoint)
                .map_err(|e| VsError::Runtime(format!("decode versioned checkpoint: {e}")))?;
        let migrated =
            migrations.migrate(checkpoint.payload, checkpoint.schema_version, target_schema)?;
        let envelope = VersionedCheckpoint {
            schema_version: target_schema,
            module_revision: module_revision.into(),
            migration_digest: migrations.chain_digest(checkpoint.schema_version, target_schema)?,
            payload: migrated,
        };
        let bytes = serde_json::to_vec(&envelope)
            .map_err(|e| VsError::Runtime(format!("encode versioned checkpoint: {e}")))?;
        self.checkpoint(run_id, expected_revision, execution_id, status, bytes)
    }
}
