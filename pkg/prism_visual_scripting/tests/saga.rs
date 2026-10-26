use async_trait::async_trait;
use std::collections::BTreeMap;
use prism_visual_scripting::prelude::*;

fn spec(operation: &str) -> EffectSpec {
    EffectSpec {
        capability: "warehouse".into(),
        operation: operation.into(),
        class: EffectClass::ExternalWrite,
        idempotency: IdempotencyPolicy::Required,
        retry: RetryPolicy::never(),
        timeout_ms: 1000,
        required_permissions: vec!["warehouse.write".into()],
    }
}

fn compensation(operation: &str, input: i64) -> CompensationIntent {
    CompensationIntent {
        spec: spec(operation),
        inputs: vec![Value::I64(input)],
        config: BTreeMap::new(),
    }
}

#[test]
fn saga_compensates_committed_steps_in_reverse_order() {
    let mut saga = SagaState::new("order-7").unwrap();
    assert!(
        saga.commit_step(
            "reserve",
            "effect-reserve",
            Some(compensation("release", 7)),
        )
        .unwrap()
    );
    assert!(
        saga.commit_step("charge", "effect-charge", Some(compensation("refund", 7)),)
            .unwrap()
    );
    saga.begin_compensation("shipping failed").unwrap();
    let first = saga.next_compensation().unwrap().unwrap();
    assert_eq!(first.step_id, "charge");
    assert_eq!(first.request.spec.operation, "refund");
    assert_eq!(first.request.inputs, vec![Value::I64(7)]);
    let replay = saga.next_compensation().unwrap().unwrap();
    assert_eq!(replay, first);
    saga.acknowledge_compensation(&first.effect_id).unwrap();
    let second = saga.next_compensation().unwrap().unwrap();
    assert_eq!(second.step_id, "reserve");
    saga.acknowledge_compensation(&second.effect_id).unwrap();
    assert_eq!(saga.status, SagaStatus::Compensated);
}

#[test]
fn saga_checkpoint_preserves_pending_compensation_identity() {
    let mut saga = SagaState::new("order-8").unwrap();
    saga.commit_step(
        "reserve",
        "effect-reserve",
        Some(compensation("release", 8)),
    )
    .unwrap();
    saga.begin_compensation("payment rejected").unwrap();
    let pending = saga.next_compensation().unwrap().unwrap();
    let encoded = serde_json::to_vec(&saga).unwrap();
    let mut restored: SagaState = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored.next_compensation().unwrap().unwrap(), pending);
    restored
        .acknowledge_compensation(&pending.effect_id)
        .unwrap();
    assert_eq!(restored.status, SagaStatus::Compensated);
}

#[test]
fn saga_rejects_step_replay_with_changed_compensation() {
    let mut saga = SagaState::new("order-9").unwrap();
    saga.commit_step(
        "reserve",
        "effect-reserve",
        Some(compensation("release", 9)),
    )
    .unwrap();
    let error = saga
        .commit_step(
            "reserve",
            "effect-reserve",
            Some(compensation("release-different", 9)),
        )
        .unwrap_err();
    assert!(error.to_string().contains("different content"));
}

#[test]
fn saga_isolates_compensation_failure_and_rejects_further_mutation() {
    let mut saga = SagaState::new("order-10").unwrap();
    saga.commit_step("charge", "effect-charge", Some(compensation("refund", 10)))
        .unwrap();
    saga.begin_compensation("downstream failed").unwrap();
    let pending = saga.next_compensation().unwrap().unwrap();
    saga.fail_compensation(&pending.effect_id, "refund provider unavailable")
        .unwrap();
    assert_eq!(saga.status, SagaStatus::CompensationFailed);
    assert!(saga.next_compensation().is_err());
}

struct SuccessfulCompensations;

#[async_trait]
impl EffectExecutor for SuccessfulCompensations {
    async fn execute(&self, request: EffectRequest) -> Result<EffectResponse, EffectError> {
        Ok(EffectResponse {
            outputs: vec![Value::String(request.spec.operation.into())],
            next: ExecNext::End,
            chunks: vec![],
        })
    }
}

#[tokio::test]
async fn saga_orchestrator_drives_all_compensations() {
    let mut saga = SagaState::new("order-runtime").unwrap();
    saga.commit_step("reserve", "forward-1", Some(compensation("release", 1)))
        .unwrap();
    saga.commit_step("charge", "forward-2", Some(compensation("refund", 1)))
        .unwrap();
    saga.begin_compensation("delivery failed").unwrap();
    SagaOrchestrator::new(&SuccessfulCompensations)
        .compensate(&mut saga)
        .await
        .unwrap();
    assert_eq!(saga.status, SagaStatus::Compensated);
    assert!(saga.steps.iter().all(|step| step.compensated));
}
