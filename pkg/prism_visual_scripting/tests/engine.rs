use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use prism_visual_scripting::prelude::*;

fn pin(name: &str, dir: PinDir, ty: ValueType) -> Pin {
    Pin {
        name: name.into(),
        dir,
        ty,
        default: None,
    }
}

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

fn edge(from_node: NodeId, from_pin: u16, to_node: NodeId, to_pin: u16) -> Edge {
    Edge {
        from: Endpoint {
            node: from_node,
            pin: from_pin,
        },
        to: Endpoint {
            node: to_node,
            pin: to_pin,
        },
    }
}

fn suspend_graph() -> Graph {
    Graph {
        name: "main".into(),
        nodes: vec![
            node(1, "test.entry"),
            node(2, "test.suspend"),
            node(3, "test.return"),
        ],
        edges: vec![edge(1, 0, 2, 0), edge(2, 0, 3, 0), edge(2, 1, 3, 1)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([(1, ControlNode::FrameEntry), (3, ControlNode::Return)]),
    }
}

fn engine_with_suspend_node() -> VisualScriptEngine {
    let mut library = NodeLibrary::new();
    library.register(NodeDefinition {
        id: "test.entry".into(),
        purity: Purity::Impure,
        inputs: vec![],
        outputs: vec![pin("then", PinDir::Out, ValueType::Exec)],
        runner: RunnerKind::sync(FnRunner(|_, _, _| unreachable!())),
    });
    library.register(NodeDefinition {
        id: "test.suspend".into(),
        purity: Purity::Impure,
        inputs: vec![pin("exec_in", PinDir::In, ValueType::Exec)],
        outputs: vec![
            pin("then", PinDir::Out, ValueType::Exec),
            pin("value", PinDir::Out, ValueType::I64),
        ],
        runner: RunnerKind::sync(FnRunner(|_, _, _| {
            Ok(ExecNext::Suspend(Suspension::approval(
                "approval-1",
                "deploy",
                Value::Unit,
            )))
        })),
    });
    library.register(NodeDefinition {
        id: "test.return".into(),
        purity: Purity::Impure,
        inputs: vec![
            pin("exec_in", PinDir::In, ValueType::Exec),
            pin("value", PinDir::In, ValueType::I64),
        ],
        outputs: vec![],
        runner: RunnerKind::sync(FnRunner(|_, _, _| unreachable!())),
    });
    VisualScriptEngine::with_node_library(library)
}

#[tokio::test]
async fn facade_compiles_runs_and_keeps_event_streams_isolated() {
    let engine = VisualScriptEngine::new();
    let graph = Graph {
        name: "simple".into(),
        nodes: vec![node(1, "event.begin"), node(2, "debug.print")],
        edges: vec![edge(1, 0, 2, 0)],
        entries: vec![1],
        ..Default::default()
    };

    assert!(!engine.analyze(&graph).unwrap().has_errors());
    let status = engine
        .run_graph(
            "exec-a",
            &graph,
            1,
            &mut InMemoryHost::default(),
            &CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(status, VmRunStatus::Completed);
    assert!(engine.events("missing").is_empty());
    engine.validate_replay("exec-a").unwrap();
}

#[tokio::test]
async fn durable_workflow_suspends_and_resumes_through_the_facade() {
    let engine = engine_with_suspend_node();
    engine
        .register_workflow(
            "deploy",
            &BTreeMap::from([("main".into(), suspend_graph())]),
        )
        .unwrap();
    let mut host = InMemoryHost::default();
    let first = engine
        .start_workflow(
            WorkflowStart {
                run_id: "run-1".into(),
                execution_id: "exec-1".into(),
                workflow_id: "deploy".into(),
                graph: "main".into(),
                entry_node: 1,
                args: vec![],
            },
            &mut host,
            &CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
    assert!(matches!(first.status, WorkflowRunStatus::Suspended(_)));
    assert_eq!(first.revision, 3);
    assert_eq!(engine.await_broker().snapshot().unwrap().pending.len(), 1);

    let store = engine.durable_store();
    let mut restarted = engine_with_suspend_node();
    restarted.set_durable_store(store);
    restarted
        .register_workflow(
            "deploy",
            &BTreeMap::from([("main".into(), suspend_graph())]),
        )
        .unwrap();
    assert_eq!(restarted.restore_run_awaits("run-1").unwrap(), 1);

    let resumed = restarted
        .resume_workflow(
            "run-1",
            "approval-1",
            SuspensionResolution {
                outputs: vec![Value::I64(7)],
                next: ExecNext::Pin("then".into()),
            },
            &mut host,
            &CancellationToken::new(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        resumed.status,
        WorkflowRunStatus::Completed(WorkflowCompletion {
            values: vec![Value::I64(7)],
            transition: None,
        })
    );
    assert_eq!(resumed.revision, 5);
    assert_eq!(
        restarted.load_run("run-1").unwrap().unwrap().1.status,
        RunStatus::Completed
    );
}

#[tokio::test]
async fn cancelled_resume_stays_recoverable_without_another_resolution() {
    let engine = engine_with_suspend_node();
    engine
        .register_workflow(
            "deploy",
            &BTreeMap::from([("main".into(), suspend_graph())]),
        )
        .unwrap();
    let mut host = InMemoryHost::default();
    engine
        .start_workflow(
            WorkflowStart {
                run_id: "run-cancel".into(),
                execution_id: "exec-cancel".into(),
                workflow_id: "deploy".into(),
                graph: "main".into(),
                entry_node: 1,
                args: vec![],
            },
            &mut host,
            &CancellationToken::new(),
            None,
        )
        .await
        .unwrap();

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let checkpoint = engine
        .resume_workflow(
            "run-cancel",
            "approval-1",
            SuspensionResolution {
                outputs: vec![Value::I64(9)],
                next: ExecNext::Pin("then".into()),
            },
            &mut host,
            &cancelled,
            None,
        )
        .await
        .unwrap();
    assert!(matches!(checkpoint.status, WorkflowRunStatus::Cancelled(_)));
    assert_eq!(
        engine.load_run("run-cancel").unwrap().unwrap().1.status,
        RunStatus::Running
    );

    let recovered = engine
        .recover_workflow("run-cancel", &mut host, &CancellationToken::new(), None)
        .await
        .unwrap();
    assert_eq!(
        recovered.status,
        WorkflowRunStatus::Completed(WorkflowCompletion {
            values: vec![Value::I64(9)],
            transition: None,
        })
    );
}

#[test]
fn facade_exposes_shared_runtime_components() {
    let mut engine = VisualScriptEngine::new();
    engine.set_durable_store(Arc::new(InMemoryDurableStore::default()));
    assert_eq!(
        engine.admit("run", "tenant", 1, None).unwrap(),
        AdmitResult::Queued
    );
    assert_eq!(engine.poll_scheduled(0).unwrap().len(), 1);
    engine.complete_scheduled("run").unwrap();
    assert!(engine.scheduler_snapshot().unwrap().active.is_empty());
}
