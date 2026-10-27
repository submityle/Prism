//! Typed durable wait conditions.

use crate::{ExecNext, SuspensionResolution, Value, VsError, VsResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AwaitCondition {
    #[default]
    External,
    Timer {
        deadline_ms: u64,
    },
    Signal {
        topic: String,
        correlation: String,
    },
    Approval {
        subject: String,
    },
    ChildRun {
        run_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AwaitRequest {
    pub id: String,
    pub condition: AwaitCondition,
    pub payload: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AwaitReady {
    pub request: AwaitRequest,
    pub resolution: SuspensionResolution,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AwaitBrokerState {
    pub pending: BTreeMap<String, AwaitRequest>,
    pub ready: BTreeMap<String, AwaitReady>,
}

/// Deterministic broker for typed waits. The broker only matches conditions; the
/// VM remains responsible for validating output arity and control routing.
#[derive(Default)]
pub struct InMemoryAwaitBroker {
    state: Mutex<AwaitBrokerState>,
}

impl InMemoryAwaitBroker {
    pub fn snapshot(&self) -> VsResult<AwaitBrokerState> {
        self.state
            .lock()
            .map(|state| state.clone())
            .map_err(|_| VsError::Runtime("await broker poisoned".into()))
    }

    pub fn restore(&self, snapshot: AwaitBrokerState) -> VsResult<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| VsError::Runtime("await broker poisoned".into()))?;
        for (id, request) in &snapshot.pending {
            validate_request(id, request)?;
        }
        for (id, ready) in &snapshot.ready {
            validate_request(id, &ready.request)?;
        }
        if snapshot
            .pending
            .keys()
            .any(|id| snapshot.ready.contains_key(id))
        {
            return Err(VsError::Validate(
                "await identity appears in both pending and ready".into(),
            ));
        }
        *state = snapshot;
        Ok(())
    }

    pub fn register(&self, request: AwaitRequest) -> VsResult<bool> {
        validate_request(&request.id, &request)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| VsError::Runtime("await broker poisoned".into()))?;
        if let Some(existing) = state.pending.get(&request.id) {
            if existing != &request {
                return Err(VsError::Runtime(format!(
                    "await identity {} was reused with different content",
                    request.id
                )));
            }
            return Ok(false);
        }
        if let Some(existing) = state.ready.get(&request.id) {
            if existing.request != request {
                return Err(VsError::Runtime(format!(
                    "await identity {} was reused with different content",
                    request.id
                )));
            }
            return Ok(false);
        }
        state.pending.insert(request.id.clone(), request);
        Ok(true)
    }

    pub fn cancel(&self, id: &str) -> VsResult<bool> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| VsError::Runtime("await broker poisoned".into()))?;
        Ok(state.pending.remove(id).is_some() || state.ready.remove(id).is_some())
    }

    pub fn resolve(&self, id: &str, resolution: SuspensionResolution) -> VsResult<bool> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| VsError::Runtime("await broker poisoned".into()))?;
        let Some(request) = state.pending.remove(id) else {
            if state.ready.contains_key(id) {
                return Ok(false);
            }
            return Err(VsError::Runtime(format!("unknown await {id}")));
        };
        state.ready.insert(
            id.to_string(),
            AwaitReady {
                request,
                resolution,
            },
        );
        Ok(true)
    }

    pub fn publish_signal(
        &self,
        topic: &str,
        correlation: &str,
        payload: Value,
    ) -> VsResult<usize> {
        self.resolve_matching(
            |request| {
                matches!(
                    &request.condition,
                    AwaitCondition::Signal {
                        topic: expected_topic,
                        correlation: expected_correlation,
                    } if expected_topic == topic && expected_correlation == correlation
                )
            },
            payload,
        )
    }

    pub fn approve(&self, subject: &str, payload: Value) -> VsResult<usize> {
        self.resolve_matching(
            |request| matches!(&request.condition, AwaitCondition::Approval { subject: expected } if expected == subject),
            payload,
        )
    }

    pub fn complete_child(&self, run_id: &str, payload: Value) -> VsResult<usize> {
        self.resolve_matching(
            |request| matches!(&request.condition, AwaitCondition::ChildRun { run_id: expected } if expected == run_id),
            payload,
        )
    }

    pub fn tick(&self, now_ms: u64) -> VsResult<usize> {
        self.resolve_matching(
            |request| matches!(&request.condition, AwaitCondition::Timer { deadline_ms } if *deadline_ms <= now_ms),
            Value::Unit,
        )
    }

    pub fn drain_ready(&self) -> VsResult<Vec<AwaitReady>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| VsError::Runtime("await broker poisoned".into()))?;
        Ok(std::mem::take(&mut state.ready).into_values().collect())
    }

    fn resolve_matching(
        &self,
        predicate: impl Fn(&AwaitRequest) -> bool,
        payload: Value,
    ) -> VsResult<usize> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| VsError::Runtime("await broker poisoned".into()))?;
        let ids = state
            .pending
            .iter()
            .filter(|(_, request)| predicate(request))
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in &ids {
            let request = state
                .pending
                .remove(id)
                .ok_or_else(|| VsError::Runtime("await broker lost pending request".into()))?;
            state.ready.insert(
                id.clone(),
                AwaitReady {
                    request,
                    resolution: SuspensionResolution {
                        outputs: vec![payload.clone()],
                        next: ExecNext::End,
                    },
                },
            );
        }
        Ok(ids.len())
    }
}

fn validate_request(id: &str, request: &AwaitRequest) -> VsResult<()> {
    if id.is_empty() || request.id != id {
        return Err(VsError::Validate(
            "await identity must not be empty or inconsistent".into(),
        ));
    }
    match &request.condition {
        AwaitCondition::Signal { topic, correlation }
            if topic.is_empty() || correlation.is_empty() =>
        {
            Err(VsError::Validate(
                "signal topic and correlation must not be empty".into(),
            ))
        }
        AwaitCondition::Approval { subject } if subject.is_empty() => Err(VsError::Validate(
            "approval subject must not be empty".into(),
        )),
        AwaitCondition::ChildRun { run_id } if run_id.is_empty() => {
            Err(VsError::Validate("child run id must not be empty".into()))
        }
        _ => Ok(()),
    }
}
