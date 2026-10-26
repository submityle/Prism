use std::sync::Arc;
use prism_visual_scripting::prelude::*;

#[test]
fn static_report_surfaces_recursive_fork_and_unbounded_cost() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(NodeDefinition {
        id: "write".into(),
        purity: Purity::Impure,
        inputs: vec![],
        outputs: vec![],
        runner: RunnerKind::effect(EffectSpec {
            capability: "db".into(),
            operation: "put".into(),
            class: EffectClass::ExternalWrite,
            idempotency: IdempotencyPolicy::Required,
            retry: RetryPolicy {
                max_attempts: 3,
                initial_backoff_ms: 1,
                max_backoff_ms: 2,
            },
            timeout_ms: 10,
            required_permissions: vec![],
        }),
    });
    let graph = Graph {
        name: "main".into(),
        nodes: vec![Node {
            id: 1,
            def: NodeDefRef {
                id: "write".into(),
                version: None,
            },
            title: None,
            config: Default::default(),
        }],
        controls: [(
            1,
            ControlNode::Fork {
                branches: vec![GraphCall {
                    graph: "main".into(),
                    entry: 1,
                }],
                policy: JoinPolicy::All,
                max_concurrency: 1,
            },
        )]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let report = analyze_graph(&graph, &lib).unwrap();
    let codes: Vec<_> = report.findings.iter().map(|f| f.code.as_str()).collect();
    assert!(codes.contains(&"RECURSIVE_FORK"));
    assert!(codes.contains(&"MISSING_COMPENSATION"));
}

#[test]
fn upgrade_shadow_compare_and_rollback_are_durable() {
    let store: Arc<dyn DurableStore> = Arc::new(InMemoryDurableStore::default());
    let coordinator = UpgradeCoordinator::new(store);
    let event = ExecutionEvent {
        sequence: 1,
        execution_id: "run".into(),
        task_id: 1,
        frame_depth: 0,
        node_id: 1,
        timestamp_ms: 1,
        event: ExecutionEventKind::NodeStarted { def_id: "a".into() },
    };
    let first = coordinator
        .begin("run", "old", "new", vec![event.clone()])
        .unwrap();
    let second = coordinator
        .record_shadow("run", first.revision, vec![event])
        .unwrap();
    assert_eq!(
        coordinator.load("run").unwrap().unwrap().1.status,
        UpgradeStatus::Shadowing
    );
    let rolled = coordinator.rollback("run", second.revision).unwrap();
    assert_eq!(rolled.revision, 3);
    assert_eq!(
        coordinator.load("run").unwrap().unwrap().1.status,
        UpgradeStatus::RolledBack
    );
}

struct Candidate(Vec<ExecutionEvent>);
impl ShadowExecutor for Candidate {
    fn run(&self, revision: &str, _input: &[u8]) -> VsResult<Vec<ExecutionEvent>> {
        assert_eq!(revision, "new");
        Ok(self.0.clone())
    }
}

#[test]
fn upgrade_can_execute_candidate_shadow() {
    let store: Arc<dyn DurableStore> = Arc::new(InMemoryDurableStore::default());
    let coordinator = UpgradeCoordinator::new(store);
    let event = ExecutionEvent {
        sequence: 1,
        execution_id: "run".into(),
        task_id: 1,
        frame_depth: 0,
        node_id: 1,
        timestamp_ms: 1,
        event: ExecutionEventKind::NodeStarted { def_id: "a".into() },
    };
    let first = coordinator
        .begin("run", "old", "new", vec![event.clone()])
        .unwrap();
    let second = coordinator
        .run_shadow("run", first.revision, b"input", &Candidate(vec![event]))
        .unwrap();
    assert_eq!(second.revision, 2);
}
