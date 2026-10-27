use async_trait::async_trait;
use prism_visual_scripting::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn node(id: NodeId, def: &str) -> Node {
    Node {
        id,
        def: NodeDefRef {
            id: def.into(),
            version: None,
        },
        title: None,
        config: HashMap::new(),
    }
}

struct MockEffects {
    calls: Mutex<u16>,
    fail_until: u16,
}

struct SlowEffects;

#[async_trait]
impl EffectExecutor for SlowEffects {
    async fn execute(&self, _request: EffectRequest) -> Result<EffectResponse, EffectError> {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Ok(EffectResponse {
            outputs: vec![Value::I64(1)],
            next: ExecNext::End,
            chunks: vec![],
        })
    }
}

#[async_trait]
impl EffectExecutor for MockEffects {
    async fn execute(&self, request: EffectRequest) -> Result<EffectResponse, EffectError> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        if *calls <= self.fail_until {
            return Err(EffectError::retryable("temporary upstream failure"));
        }
        assert!(!request.execution_id.is_empty());
        Ok(EffectResponse {
            outputs: vec![Value::I64(42)],
            next: ExecNext::End,
            chunks: vec![],
        })
    }
}

fn effect_def() -> NodeDefinition {
    NodeDefinition {
        id: "test.effect".into(),
        purity: Purity::Impure,
        inputs: vec![],
        outputs: vec![
            Pin {
                name: "then".into(),
                dir: PinDir::Out,
                ty: ValueType::Exec,
                default: None,
            },
            Pin {
                name: "result".into(),
                dir: PinDir::Out,
                ty: ValueType::I64,
                default: None,
            },
        ],
        runner: RunnerKind::effect(EffectSpec {
            capability: "test.upstream".into(),
            operation: "read".into(),
            class: EffectClass::ExternalWrite,
            idempotency: IdempotencyPolicy::Required,
            retry: RetryPolicy {
                max_attempts: 3,
                initial_backoff_ms: 0,
                max_backoff_ms: 0,
            },
            timeout_ms: 1000,
            required_permissions: vec![],
        }),
    }
}

fn timeout_effect_def() -> NodeDefinition {
    let mut def = effect_def();
    if let RunnerKind::Effect(spec) = &mut def.runner {
        spec.timeout_ms = 1;
        spec.retry.max_attempts = 1;
    }
    def
}

#[tokio::test]
async fn effect_runs_with_retry_and_audits_identity() {
    let mut library = NodeLibrary::new();
    library.register(effect_def());
    let graph = Graph {
        name: "effects".into(),
        nodes: vec![node(1, "test.effect")],
        edges: vec![],
        variables: vec![],
        entries: vec![1],
        controls: Default::default(),
    };
    let vm = Vm::new(compile(&graph, &library).unwrap());
    let effects = MockEffects {
        calls: Mutex::new(0),
        fail_until: 1,
    };
    let events = InMemoryEventLog::default();
    let env = ExecutionEnv::new()
        .with_execution_id("run-1")
        .with_effect_executor(&effects)
        .with_event_sink(&events);
    vm.run_entry_async_with_env(1, &mut InMemoryHost::default(), &env)
        .await
        .unwrap();
    assert_eq!(*effects.calls.lock().unwrap(), 2);
    let events = events.events();
    assert_eq!(events.len(), 4);
    assert_eq!(
        events
            .iter()
            .map(|event| event.sequence)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    let ids: Vec<String> = events
        .iter()
        .filter_map(|event| match &event.event {
            ExecutionEventKind::EffectRequested { id, .. }
            | ExecutionEventKind::EffectFailed { id, .. }
            | ExecutionEventKind::EffectCompleted { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    assert!(ids.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn virtual_clock_rejects_checked_overflow_without_wrapping() {
    let clock = VirtualClock::new(u64::MAX - 1);
    let error = clock.try_advance(2).unwrap_err();
    assert!(error.to_string().contains("virtual clock exhausted"));
    assert_eq!(clock.now_millis(), u64::MAX - 1);
}

#[tokio::test]
async fn virtual_clock_advances_retry_without_real_sleep() {
    let mut library = NodeLibrary::new();
    let mut def = effect_def();
    if let RunnerKind::Effect(spec) = &mut def.runner {
        spec.retry.initial_backoff_ms = 50;
        spec.retry.max_backoff_ms = 500;
    }
    library.register(def);
    let graph = Graph {
        name: "effects".into(),
        nodes: vec![node(1, "test.effect")],
        edges: vec![],
        variables: vec![],
        entries: vec![1],
        controls: Default::default(),
    };
    let vm = Vm::new(compile(&graph, &library).unwrap());
    let effects = MockEffects {
        calls: Mutex::new(0),
        fail_until: 1,
    };
    let clock = VirtualClock::new(100);
    let env = ExecutionEnv::new()
        .with_execution_id("run-clock")
        .with_effect_executor(&effects)
        .with_clock(&clock);
    vm.run_entry_async_with_env(1, &mut InMemoryHost::default(), &env)
        .await
        .unwrap();
    assert_eq!(clock.now_millis(), 150);
}

#[tokio::test]
async fn effect_requires_stable_execution_identity() {
    let mut library = NodeLibrary::new();
    library.register(effect_def());
    let graph = Graph {
        name: "effects".into(),
        nodes: vec![node(1, "test.effect")],
        edges: vec![],
        variables: vec![],
        entries: vec![1],
        controls: Default::default(),
    };
    let vm = Vm::new(compile(&graph, &library).unwrap());
    let effects = MockEffects {
        calls: Mutex::new(0),
        fail_until: 0,
    };
    let error = vm
        .run_entry_async_with_env(
            1,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new().with_effect_executor(&effects),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, VsError::MissingExecutionId));
}

#[tokio::test]
async fn memoizing_executor_rejects_identity_reuse_with_changed_payload() {
    let inner = MockEffects {
        calls: Mutex::new(0),
        fail_until: 0,
    };
    let executor = MemoizingEffectExecutor::new(inner);
    let spec = effect_def();
    let request = |input| EffectRequest {
        id: "effect:fixed".into(),
        execution_id: "run".into(),
        task_id: 1,
        frame_depth: 0,
        sequence: 0,
        node_id: 9,
        spec: match &spec.runner {
            RunnerKind::Effect(spec) => spec.clone(),
            _ => unreachable!(),
        },
        inputs: vec![Value::I64(input)],
        config: Default::default(),
        attempt: 1,
    };
    executor.execute(request(1)).await.unwrap();
    let error = executor.execute(request(2)).await.unwrap_err();
    assert!(!error.retryable);
    assert!(error.message.contains("different payload"));
}

struct CountingEffects {
    calls: Arc<AtomicUsize>,
}

struct ChunkEffects;

#[async_trait]
impl EffectExecutor for ChunkEffects {
    async fn execute(&self, _request: EffectRequest) -> Result<EffectResponse, EffectError> {
        Ok(EffectResponse {
            outputs: vec![Value::I64(12)],
            next: ExecNext::End,
            chunks: vec![Value::String("delta".into())],
        })
    }
}

struct SecretTaint;

impl TaintProvider for SecretTaint {
    fn taint(&self, _inputs: &[Value]) -> TaintSet {
        let mut labels = TaintSet::default();
        labels.insert("secret").unwrap();
        labels
    }
}

#[async_trait]
impl EffectExecutor for CountingEffects {
    async fn execute(&self, _request: EffectRequest) -> Result<EffectResponse, EffectError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(EffectResponse {
            outputs: vec![Value::I64(99)],
            next: ExecNext::End,
            chunks: vec![],
        })
    }
}

#[tokio::test]
async fn durable_executor_replays_committed_response_after_restart() {
    let calls = Arc::new(AtomicUsize::new(0));
    let store: Arc<dyn DurableStore> = Arc::new(InMemoryDurableStore::default());
    let spec = effect_def();
    let request = EffectRequest {
        id: "effect:durable".into(),
        execution_id: "run".into(),
        task_id: 1,
        frame_depth: 0,
        sequence: 0,
        node_id: 9,
        spec: match &spec.runner {
            RunnerKind::Effect(spec) => spec.clone(),
            _ => unreachable!(),
        },
        inputs: vec![Value::I64(1)],
        config: Default::default(),
        attempt: 1,
    };
    let first = DurableEffectExecutor::new(
        CountingEffects {
            calls: calls.clone(),
        },
        store.clone(),
    );
    assert_eq!(
        first.execute(request.clone()).await.unwrap().outputs,
        vec![Value::I64(99)]
    );
    drop(first);

    let restarted = DurableEffectExecutor::new(
        CountingEffects {
            calls: calls.clone(),
        },
        store,
    );
    assert_eq!(
        restarted.execute(request).await.unwrap().outputs,
        vec![Value::I64(99)]
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn effect_chunks_are_forwarded_and_taint_policy_is_enforced() {
    let mut library = NodeLibrary::new();
    library.register(effect_def());
    let graph = Graph {
        name: "effects".into(),
        nodes: vec![node(1, "test.effect")],
        edges: vec![],
        variables: vec![],
        entries: vec![1],
        controls: Default::default(),
    };
    let vm = Vm::new(compile(&graph, &library).unwrap());
    let (tx, mut rx) = tokio::sync::mpsc::channel(2);
    let effects = ChunkEffects;
    let env = ExecutionEnv::new()
        .with_execution_id("run-chunks")
        .with_effect_executor(&effects)
        .with_chunk_tx(&tx);
    vm.run_entry_async_with_env(1, &mut InMemoryHost::default(), &env)
        .await
        .unwrap();
    assert!(matches!(
        rx.recv().await,
        Some(Chunk::Delta(Value::String(value))) if value.as_ref() == "delta"
    ));

    let mut policy = TaintPolicy::default();
    policy.deny("secret", EffectClass::ExternalWrite).unwrap();
    let error = vm
        .run_entry_async_with_env(
            1,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new()
                .with_execution_id("run-tainted")
                .with_effect_executor(&effects)
                .with_policy(&policy)
                .with_taints(&SecretTaint),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("cannot flow"));
}

#[tokio::test]
async fn effect_timeout_is_enforced_by_the_vm() {
    let mut library = NodeLibrary::new();
    library.register(timeout_effect_def());
    let graph = Graph {
        name: "effects".into(),
        nodes: vec![node(1, "test.effect")],
        edges: vec![],
        variables: vec![],
        entries: vec![1],
        controls: Default::default(),
    };
    let vm = Vm::new(compile(&graph, &library).unwrap());
    let error = vm
        .run_entry_async_with_env(
            1,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new()
                .with_execution_id("run-timeout")
                .with_effect_executor(&SlowEffects),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, VsError::Effect(message) if message.contains("timeout")));
}
