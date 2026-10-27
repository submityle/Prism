//! Structured execution event stream used for audit, replay tooling, and durable projections.

use crate::{EffectError, EffectSpec, Value, VsResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::sync::Mutex;

pub const MAX_EXECUTION_ID_BYTES: usize = 128;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionEventKind {
    NodeStarted {
        def_id: String,
    },
    NodeCompleted {
        def_id: String,
    },
    EffectRequested {
        id: String,
        spec: EffectSpec,
        attempt: u16,
    },
    EffectCompleted {
        id: String,
        outputs: Vec<Value>,
        #[serde(default)]
        chunks: Vec<Value>,
    },
    EffectFailed {
        id: String,
        error: EffectError,
        attempt: u16,
    },
    Suspended {
        suspension_id: String,
    },
    Resumed {
        suspension_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionEvent {
    #[serde(default)]
    pub sequence: u64,
    pub execution_id: String,
    pub task_id: u64,
    pub frame_depth: usize,
    pub node_id: u32,
    pub timestamp_ms: u64,
    pub event: ExecutionEventKind,
}

pub trait ExecutionEventSink: Send + Sync {
    fn append(&self, event: ExecutionEvent) -> VsResult<()>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayReport {
    pub execution_id: String,
    pub event_count: usize,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayDivergence {
    pub sequence: u64,
    pub expected_digest: [u8; 32],
    pub actual_digest: [u8; 32],
}

pub struct ExecutionReplay;

impl ExecutionReplay {
    pub fn validate(events: &[ExecutionEvent], execution_id: &str) -> VsResult<ReplayReport> {
        if execution_id.is_empty() || execution_id.len() > MAX_EXECUTION_ID_BYTES {
            return Err(crate::VsError::Validate(
                "replay execution identity must be non-empty and at most 128 bytes".into(),
            ));
        }
        for (index, event) in events.iter().enumerate() {
            let expected = next_execution_sequence(index)?;
            if event.sequence != expected {
                return Err(crate::VsError::Runtime(format!(
                    "replay event sequence mismatch: expected {expected}, got {}",
                    event.sequence
                )));
            }
            if event.execution_id != execution_id {
                return Err(crate::VsError::Runtime(format!(
                    "replay execution identity mismatch at sequence {}",
                    event.sequence
                )));
            }
        }
        Ok(ReplayReport {
            execution_id: execution_id.to_string(),
            event_count: events.len(),
            digest: Self::digest(events)?,
        })
    }

    pub fn compare(
        expected: &[ExecutionEvent],
        actual: &[ExecutionEvent],
    ) -> VsResult<Option<ReplayDivergence>> {
        Self::validate_pair(expected)?;
        Self::validate_pair(actual)?;
        if let (Some(expected_first), Some(actual_first)) = (expected.first(), actual.first()) {
            if expected_first.execution_id.is_empty()
                || expected_first.execution_id.len() > MAX_EXECUTION_ID_BYTES
                || actual_first.execution_id.is_empty()
                || actual_first.execution_id.len() > MAX_EXECUTION_ID_BYTES
            {
                return Err(crate::VsError::Validate(
                    "replay execution identity must not be empty".into(),
                ));
            }
            if expected_first.execution_id != actual_first.execution_id
                || expected
                    .iter()
                    .any(|event| event.execution_id != expected_first.execution_id)
                || actual
                    .iter()
                    .any(|event| event.execution_id != actual_first.execution_id)
            {
                return Err(crate::VsError::Runtime(
                    "replay execution identity mismatch".into(),
                ));
            }
        } else if expected.len() != actual.len() {
            return Err(crate::VsError::Runtime(
                "replay execution identity mismatch".into(),
            ));
        }
        for (left, right) in expected.iter().zip(actual) {
            let left_digest = event_digest(left)?;
            let right_digest = event_digest(right)?;
            if left_digest != right_digest {
                return Ok(Some(ReplayDivergence {
                    sequence: left.sequence,
                    expected_digest: left_digest,
                    actual_digest: right_digest,
                }));
            }
        }
        if expected.len() != actual.len() {
            let sequence = expected
                .get(actual.len())
                .or_else(|| actual.get(expected.len()))
                .map_or(1, |event| event.sequence);
            let expected_digest = expected
                .get(actual.len())
                .map(event_digest)
                .transpose()?
                .unwrap_or([0; 32]);
            let actual_digest = actual
                .get(expected.len())
                .map(event_digest)
                .transpose()?
                .unwrap_or([0; 32]);
            return Ok(Some(ReplayDivergence {
                sequence,
                expected_digest,
                actual_digest,
            }));
        }
        Ok(None)
    }

    pub fn digest(events: &[ExecutionEvent]) -> VsResult<[u8; 32]> {
        let bytes = serde_json::to_vec(events)
            .map_err(|error| crate::VsError::Runtime(format!("encode replay events: {error}")))?;
        Ok(Sha256::digest(bytes).into())
    }

    fn validate_pair(events: &[ExecutionEvent]) -> VsResult<()> {
        for (index, event) in events.iter().enumerate() {
            if event.sequence != next_execution_sequence(index)? {
                return Err(crate::VsError::Runtime(
                    "replay event sequence is not contiguous".into(),
                ));
            }
        }
        Ok(())
    }
}

fn event_digest(event: &ExecutionEvent) -> VsResult<[u8; 32]> {
    let bytes = serde_json::to_vec(event)
        .map_err(|error| crate::VsError::Runtime(format!("encode replay event: {error}")))?;
    Ok(Sha256::digest(bytes).into())
}

pub fn next_execution_sequence(event_count: usize) -> VsResult<u64> {
    u64::try_from(event_count)
        .ok()
        .and_then(|sequence| sequence.checked_add(1))
        .ok_or_else(|| crate::VsError::Runtime("execution event sequence exhausted".into()))
}

#[derive(Default)]
pub struct InMemoryEventLog {
    events: Mutex<Vec<ExecutionEvent>>,
}

impl InMemoryEventLog {
    pub fn events(&self) -> Vec<ExecutionEvent> {
        match self.events.lock() {
            Ok(events) => events.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    pub fn events_since(&self, sequence: u64) -> Vec<ExecutionEvent> {
        self.events()
            .into_iter()
            .filter(|event| event.sequence > sequence)
            .collect()
    }
}

impl ExecutionEventSink for InMemoryEventLog {
    fn append(&self, mut event: ExecutionEvent) -> VsResult<()> {
        let mut events = self.events.lock().expect("event log poisoned");
        event.sequence = next_execution_sequence(events.len())?;
        events.push(event);
        Ok(())
    }
}
