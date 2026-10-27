//! Durable Saga region with reverse-order compensation.
//!
//! A Saga records only committed side effects. When the forward path fails, compensation intents
//! are exposed one at a time in reverse commit order. Each intent becomes a normal
//! [`EffectRequest`](crate::EffectRequest), so existing idempotency, sandbox, permission, and audit
//! infrastructure remains the only side-effect boundary.

use crate::{EffectExecutor, EffectRequest, EffectSpec, Value, VsError, VsResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SagaStatus {
    Active,
    Completed,
    Compensating,
    Compensated,
    CompensationFailed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompensationIntent {
    pub spec: EffectSpec,
    pub inputs: Vec<Value>,
    #[serde(default)]
    pub config: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SagaStep {
    pub step_id: String,
    pub sequence: u64,
    pub forward_effect_id: String,
    pub compensation: Option<CompensationIntent>,
    pub compensated: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PendingCompensation {
    pub step_id: String,
    pub effect_id: String,
    pub request: EffectRequest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SagaState {
    pub id: String,
    pub revision: u64,
    pub status: SagaStatus,
    pub steps: Vec<SagaStep>,
    pub failure: Option<String>,
    pub compensation_failure: Option<String>,
    next_sequence: u64,
    committed_step_ids: BTreeSet<String>,
    pending: Option<PendingCompensation>,
}

impl SagaState {
    pub fn new(id: impl Into<String>) -> VsResult<Self> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(VsError::Runtime("saga id must not be empty".into()));
        }
        Ok(Self {
            id,
            revision: 0,
            status: SagaStatus::Active,
            steps: Vec::new(),
            failure: None,
            compensation_failure: None,
            next_sequence: 1,
            committed_step_ids: BTreeSet::new(),
            pending: None,
        })
    }

    /// Records a forward effect only after it has committed successfully.
    /// Identical `step_id` delivery is idempotent and returns `false`.
    pub fn commit_step(
        &mut self,
        step_id: impl Into<String>,
        forward_effect_id: impl Into<String>,
        compensation: Option<CompensationIntent>,
    ) -> VsResult<bool> {
        self.require_status(SagaStatus::Active)?;
        let step_id = step_id.into();
        let forward_effect_id = forward_effect_id.into();
        if step_id.is_empty() || forward_effect_id.is_empty() {
            return Err(VsError::Runtime(
                "saga step and forward effect identities must not be empty".into(),
            ));
        }
        if self.committed_step_ids.contains(&step_id) {
            let Some(existing) = self.steps.iter().find(|step| step.step_id == step_id) else {
                return Err(VsError::Runtime(
                    "saga committed step index is inconsistent".into(),
                ));
            };
            if existing.forward_effect_id != forward_effect_id
                || existing.compensation != compensation
            {
                return Err(VsError::Runtime(format!(
                    "saga step {step_id} was replayed with different content"
                )));
            }
            return Ok(false);
        }
        if let Some(intent) = &compensation {
            intent.spec.validate()?;
        }
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| VsError::Runtime("saga step sequence exhausted".into()))?;
        self.committed_step_ids.insert(step_id.clone());
        self.steps.push(SagaStep {
            step_id,
            sequence,
            forward_effect_id,
            compensation,
            compensated: false,
        });
        self.bump_revision()?;
        Ok(true)
    }

    pub fn complete(&mut self) -> VsResult<()> {
        self.require_status(SagaStatus::Active)?;
        self.status = SagaStatus::Completed;
        self.bump_revision()
    }

    pub fn begin_compensation(&mut self, failure: impl Into<String>) -> VsResult<()> {
        self.require_status(SagaStatus::Active)?;
        let failure = failure.into();
        if failure.is_empty() {
            return Err(VsError::Runtime("saga failure must not be empty".into()));
        }
        self.failure = Some(failure);
        self.status = SagaStatus::Compensating;
        self.bump_revision()?;
        if !self.has_uncompensated_steps() {
            self.status = SagaStatus::Compensated;
            self.bump_revision()?;
        }
        Ok(())
    }

    /// Returns the same request until it is acknowledged, which makes crash recovery deterministic.
    pub fn next_compensation(&mut self) -> VsResult<Option<PendingCompensation>> {
        self.require_status(SagaStatus::Compensating)?;
        if let Some(pending) = &self.pending {
            return Ok(Some(pending.clone()));
        }
        let Some(step) = self
            .steps
            .iter()
            .rev()
            .find(|step| !step.compensated && step.compensation.is_some())
        else {
            self.status = SagaStatus::Compensated;
            self.bump_revision()?;
            return Ok(None);
        };
        let Some(intent) = step.compensation.as_ref() else {
            return Err(VsError::Runtime(
                "saga compensation state is inconsistent".into(),
            ));
        };
        let effect_id = compensation_effect_id(&self.id, step);
        let request = EffectRequest {
            id: effect_id.clone(),
            execution_id: format!("saga:{}", self.id),
            task_id: 0,
            frame_depth: 0,
            sequence: step.sequence,
            node_id: 0,
            spec: intent.spec.clone(),
            inputs: intent.inputs.clone(),
            config: intent.config.clone(),
            attempt: 1,
        };
        let pending = PendingCompensation {
            step_id: step.step_id.clone(),
            effect_id,
            request,
        };
        self.pending = Some(pending.clone());
        self.bump_revision()?;
        Ok(Some(pending))
    }

    pub fn acknowledge_compensation(&mut self, effect_id: &str) -> VsResult<()> {
        self.require_status(SagaStatus::Compensating)?;
        let pending = self
            .pending
            .take()
            .ok_or_else(|| VsError::Runtime("saga has no pending compensation".into()))?;
        if pending.effect_id != effect_id {
            self.pending = Some(pending);
            return Err(VsError::Runtime(
                "saga compensation effect identity mismatch".into(),
            ));
        }
        let step = self
            .steps
            .iter_mut()
            .find(|step| step.step_id == pending.step_id)
            .ok_or_else(|| VsError::Runtime("saga pending step is missing".into()))?;
        step.compensated = true;
        self.bump_revision()?;
        if !self.has_uncompensated_steps() {
            self.status = SagaStatus::Compensated;
            self.bump_revision()?;
        }
        Ok(())
    }

    pub fn fail_compensation(&mut self, effect_id: &str, error: impl Into<String>) -> VsResult<()> {
        self.require_status(SagaStatus::Compensating)?;
        let pending = self
            .pending
            .as_ref()
            .ok_or_else(|| VsError::Runtime("saga has no pending compensation".into()))?;
        if pending.effect_id != effect_id {
            return Err(VsError::Runtime(
                "saga compensation effect identity mismatch".into(),
            ));
        }
        self.compensation_failure = Some(error.into());
        self.status = SagaStatus::CompensationFailed;
        self.bump_revision()
    }

    fn has_uncompensated_steps(&self) -> bool {
        self.steps
            .iter()
            .any(|step| !step.compensated && step.compensation.is_some())
    }

    fn require_status(&self, expected: SagaStatus) -> VsResult<()> {
        if self.status != expected {
            return Err(VsError::Runtime(format!(
                "saga {} is {:?}, expected {:?}",
                self.id, self.status, expected
            )));
        }
        Ok(())
    }

    fn bump_revision(&mut self) -> VsResult<()> {
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| VsError::Runtime("saga revision exhausted".into()))?;
        Ok(())
    }
}

fn compensation_effect_id(saga_id: &str, step: &SagaStep) -> String {
    let mut digest = Sha256::new();
    digest.update(saga_id.as_bytes());
    digest.update([0]);
    digest.update(step.step_id.as_bytes());
    digest.update([0]);
    digest.update(step.forward_effect_id.as_bytes());
    digest.update(step.sequence.to_le_bytes());
    let suffix = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("compensation:{suffix}")
}

pub struct SagaOrchestrator<'a> {
    effects: &'a dyn EffectExecutor,
}

impl<'a> SagaOrchestrator<'a> {
    pub fn new(effects: &'a dyn EffectExecutor) -> Self {
        Self { effects }
    }

    /// Drives compensation until the Saga reaches a terminal state. State is
    /// advanced after every effect so callers can checkpoint between polls.
    pub async fn compensate(&self, saga: &mut SagaState) -> VsResult<()> {
        loop {
            let Some(pending) = saga.next_compensation()? else {
                return Ok(());
            };
            match self.effects.execute(pending.request.clone()).await {
                Ok(_) => {
                    saga.acknowledge_compensation(&pending.effect_id)?;
                    if saga.status == SagaStatus::Compensated {
                        return Ok(());
                    }
                }
                Err(error) => {
                    saga.fail_compensation(&pending.effect_id, error.message.clone())?;
                    return Err(VsError::Effect(error.message));
                }
            }
        }
    }
}
