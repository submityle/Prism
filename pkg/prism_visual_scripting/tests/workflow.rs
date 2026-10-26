use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use prism_visual_scripting::prelude::*;
use prism_visual_scripting::{GraphCall, PinIndex, WorkflowProgram, compile};

fn pin(name: &str, dir: PinDir, ty: ValueType, default: Option<Value>) -> Pin {
    Pin {
        name: name.into(),
        dir,
        ty,
        default,
    }
}

fn entry_def(value_type: Option<ValueType>) -> NodeDefinition {
    let mut outputs = vec![pin("then", PinDir::Out, ValueType::Exec, None)];
    if let Some(value_type) = value_type {
        outputs.push(pin("value", PinDir::Out, value_type, None));
    }
    NodeDefinition {
        id: "test.entry".into(),
        purity: Purity::Impure,
        inputs: vec![],
        outputs,
        runner: RunnerKind::sync(FnRunner(|_, _, _| unreachable!())),
    }
}

fn named_return_def(id: &str, value_type: Option<ValueType>) -> NodeDefinition {
    let mut inputs = vec![pin("exec_in", PinDir::In, ValueType::Exec, None)];
    if let Some(value_type) = value_type {
        inputs.push(pin("value", PinDir::In, value_type, None));
    }
    NodeDefinition {
        id: id.into(),
        purity: Purity::Impure,
        inputs,
        outputs: vec![],
        runner: RunnerKind::sync(FnRunner(|_, _, _| unreachable!())),
    }
}

fn return_def(value_type: Option<ValueType>) -> NodeDefinition {
    named_return_def("test.return", value_type)
}

fn call_def(input_type: ValueType, output_type: ValueType) -> NodeDefinition {
    NodeDefinition {
        id: "test.call".into(),
        purity: Purity::Impure,
        inputs: vec![
            pin("exec_in", PinDir::In, ValueType::Exec, None),
            pin("value", PinDir::In, input_type, Some(Value::I64(7))),
        ],
        outputs: vec![
            pin("then", PinDir::Out, ValueType::Exec, None),
            pin("error", PinDir::Out, ValueType::Exec, None),
            pin("value", PinDir::Out, output_type, None),
        ],
        runner: RunnerKind::sync(FnRunner(|_, _, _| unreachable!())),
    }
}

fn fork_def(output_type: ValueType) -> NodeDefinition {
    let mut definition = call_def(ValueType::I64, output_type);
    definition.id = "test.fork".into();
    definition
}

fn transition_def() -> NodeDefinition {
    NodeDefinition {
        id: "test.transition".into(),
        purity: Purity::Impure,
        inputs: vec![pin("exec_in", PinDir::In, ValueType::Exec, None)],
        outputs: vec![],
        runner: RunnerKind::sync(FnRunner(|_, _, _| unreachable!())),
    }
}

fn cancel_def() -> NodeDefinition {
    NodeDefinition {
        id: "test.cancel".into(),
        purity: Purity::Impure,
        inputs: vec![pin("exec_in", PinDir::In, ValueType::Exec, None)],
        outputs: vec![pin("then", PinDir::Out, ValueType::Exec, None)],
        runner: RunnerKind::sync(FnRunner(|_, _, cx| {
            cx.cancel.cancel();
            Ok(ExecNext::Pin("then".into()))
        })),
    }
}

fn suspend_def() -> NodeDefinition {
    NodeDefinition {
        id: "test.suspend".into(),
        purity: Purity::Impure,
        inputs: vec![pin("exec_in", PinDir::In, ValueType::Exec, None)],
        outputs: vec![
            pin("then", PinDir::Out, ValueType::Exec, None),
            pin("value", PinDir::Out, ValueType::I64, None),
        ],
        runner: RunnerKind::sync(FnRunner(|_, _, _| {
            Ok(ExecNext::Suspend(Suspension::new(
                "decision-1",
                Value::String(Arc::from("choose a number")),
            )))
        })),
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

fn edge(from_node: NodeId, from_pin: PinIndex, to_node: NodeId, to_pin: PinIndex) -> Edge {
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

fn identity_graph() -> Graph {
    Graph {
        name: "identity".into(),
        nodes: vec![node(1, "test.entry"), node(2, "test.identity_return")],
        edges: vec![edge(1, 0, 2, 0), edge(1, 1, 2, 1)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([(1, ControlNode::FrameEntry), (2, ControlNode::Return)]),
    }
}

fn module_with_main(main: Graph, lib: &NodeLibrary) -> WorkflowProgram {
    let mut program = WorkflowProgram::new();
    program.insert("main", compile(&main, lib).unwrap());
    program.insert("identity", compile(&identity_graph(), lib).unwrap());
    program
}

#[tokio::test]
async fn function_call_uses_a_durable_call_frame() {
    let mut lib = NodeLibrary::new();
    lib.register(entry_def(Some(ValueType::I64)));
    lib.register(return_def(Some(ValueType::I64)));
    lib.register(named_return_def(
        "test.identity_return",
        Some(ValueType::I64),
    ));
    lib.register(call_def(ValueType::I64, ValueType::I64));

    let main = Graph {
        name: "main".into(),
        nodes: vec![
            node(1, "test.entry"),
            node(2, "test.call"),
            node(3, "test.return"),
        ],
        edges: vec![edge(1, 0, 2, 0), edge(2, 0, 3, 0), edge(2, 2, 3, 1)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([
            (1, ControlNode::FrameEntry),
            (
                2,
                ControlNode::Call {
                    target: GraphCall {
                        graph: "identity".into(),
                        entry: 1,
                    },
                },
            ),
            (3, ControlNode::Return),
        ]),
    };
    let vm = WorkflowVm::new(module_with_main(main, &lib));
    let env = ExecutionEnv::new();
    let result = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![Value::I64(0)],
            None,
            &mut InMemoryHost::default(),
            &env,
        )
        .await
        .unwrap();
    assert_eq!(
        result,
        WorkflowRunStatus::Completed(WorkflowCompletion {
            values: vec![Value::I64(7)],
            transition: None,
        })
    );
}

#[tokio::test]
async fn invalid_function_return_routes_to_call_error_pin() {
    let mut lib = NodeLibrary::new();
    lib.register(entry_def(Some(ValueType::I64)));
    lib.register(return_def(Some(ValueType::I64)));
    lib.register(named_return_def("test.empty_return", None));
    lib.register(call_def(ValueType::I64, ValueType::I64));
    let child = Graph {
        name: "child".into(),
        nodes: vec![node(1, "test.entry"), node(2, "test.empty_return")],
        edges: vec![edge(1, 0, 2, 0)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([(1, ControlNode::FrameEntry), (2, ControlNode::Return)]),
    };
    let main = Graph {
        name: "main".into(),
        nodes: vec![
            node(1, "test.entry"),
            node(2, "test.call"),
            node(3, "test.return"),
        ],
        edges: vec![edge(1, 0, 2, 0), edge(2, 1, 3, 0)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([
            (1, ControlNode::FrameEntry),
            (
                2,
                ControlNode::Call {
                    target: GraphCall {
                        graph: "child".into(),
                        entry: 1,
                    },
                },
            ),
            (3, ControlNode::Return),
        ]),
    };
    let mut program = WorkflowProgram::new();
    program.insert("main", compile(&main, &lib).unwrap());
    program.insert("child", compile(&child, &lib).unwrap());
    let result = WorkflowVm::new(program)
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![Value::I64(1)],
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        result,
        WorkflowRunStatus::Completed(WorkflowCompletion {
            values: vec![Value::I64(0)],
            transition: None,
        })
    );
}

#[tokio::test]
async fn parallel_all_joins_results_in_branch_order() {
    let mut lib = NodeLibrary::new();
    lib.register(entry_def(Some(ValueType::I64)));
    lib.register(return_def(Some(ValueType::List(Box::new(ValueType::I64)))));
    lib.register(named_return_def(
        "test.identity_return",
        Some(ValueType::I64),
    ));
    lib.register(fork_def(ValueType::List(Box::new(ValueType::I64))));
    let main = Graph {
        name: "main".into(),
        nodes: vec![
            node(1, "test.entry"),
            node(2, "test.fork"),
            node(3, "test.return"),
        ],
        edges: vec![edge(1, 0, 2, 0), edge(2, 0, 3, 0), edge(2, 2, 3, 1)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([
            (1, ControlNode::FrameEntry),
            (
                2,
                ControlNode::Fork {
                    branches: vec![
                        GraphCall {
                            graph: "identity".into(),
                            entry: 1,
                        },
                        GraphCall {
                            graph: "identity".into(),
                            entry: 1,
                        },
                    ],
                    policy: JoinPolicy::All,
                    max_concurrency: 1,
                },
            ),
            (3, ControlNode::Return),
        ]),
    };
    let vm = WorkflowVm::new(module_with_main(main, &lib));
    let result = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![Value::I64(0)],
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        result,
        WorkflowRunStatus::Completed(WorkflowCompletion {
            values: vec![Value::List(Arc::new(vec![Value::I64(7), Value::I64(7)]))],
            transition: None,
        })
    );
}

#[tokio::test]
async fn race_resumes_with_the_first_deterministic_result() {
    let mut lib = NodeLibrary::new();
    lib.register(entry_def(Some(ValueType::I64)));
    lib.register(return_def(Some(ValueType::I64)));
    lib.register(named_return_def(
        "test.identity_return",
        Some(ValueType::I64),
    ));
    lib.register(fork_def(ValueType::I64));
    let main = Graph {
        name: "main".into(),
        nodes: vec![
            node(1, "test.entry"),
            node(2, "test.fork"),
            node(3, "test.return"),
        ],
        edges: vec![edge(1, 0, 2, 0), edge(2, 0, 3, 0), edge(2, 2, 3, 1)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([
            (1, ControlNode::FrameEntry),
            (
                2,
                ControlNode::Fork {
                    branches: vec![
                        GraphCall {
                            graph: "identity".into(),
                            entry: 1,
                        },
                        GraphCall {
                            graph: "identity".into(),
                            entry: 1,
                        },
                    ],
                    policy: JoinPolicy::Race,
                    max_concurrency: 2,
                },
            ),
            (3, ControlNode::Return),
        ]),
    };
    let result = WorkflowVm::new(module_with_main(main, &lib))
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![Value::I64(0)],
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        result,
        WorkflowRunStatus::Completed(WorkflowCompletion {
            values: vec![Value::I64(7)],
            transition: None,
        })
    );
}

#[tokio::test]
async fn scheduler_checkpoint_restores_all_tasks_and_joins() {
    let mut lib = NodeLibrary::new();
    lib.register(entry_def(Some(ValueType::I64)));
    lib.register(return_def(Some(ValueType::List(Box::new(ValueType::I64)))));
    lib.register(named_return_def(
        "test.identity_return",
        Some(ValueType::I64),
    ));
    lib.register(fork_def(ValueType::List(Box::new(ValueType::I64))));
    let main = Graph {
        name: "main".into(),
        nodes: vec![
            node(1, "test.entry"),
            node(2, "test.fork"),
            node(3, "test.return"),
        ],
        edges: vec![edge(1, 0, 2, 0), edge(2, 0, 3, 0), edge(2, 2, 3, 1)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([
            (1, ControlNode::FrameEntry),
            (
                2,
                ControlNode::Fork {
                    branches: vec![
                        GraphCall {
                            graph: "identity".into(),
                            entry: 1,
                        },
                        GraphCall {
                            graph: "identity".into(),
                            entry: 1,
                        },
                    ],
                    policy: JoinPolicy::All,
                    max_concurrency: 2,
                },
            ),
            (3, ControlNode::Return),
        ]),
    };
    let vm = WorkflowVm::new(module_with_main(main, &lib)).with_step_budget(3);
    let first = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![Value::I64(0)],
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new(),
        )
        .await
        .unwrap_err();
    assert!(first.to_string().contains("step budget"));

    let vm = WorkflowVm::new(module_with_main(identity_graph(), &lib));
    let cancel = CancellationToken::new();
    cancel.cancel();
    let suspended = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![Value::I64(9)],
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new().with_cancel(&cancel),
        )
        .await
        .unwrap();
    let WorkflowRunStatus::Cancelled(state) = suspended else {
        panic!("cancelled workflow must suspend");
    };
    let restored = serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
    let completed = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![],
            Some(restored),
            &mut InMemoryHost::default(),
            &ExecutionEnv::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        completed,
        WorkflowRunStatus::Completed(WorkflowCompletion {
            values: vec![Value::I64(9)],
            transition: None,
        })
    );
}

#[tokio::test]
async fn transition_is_a_runtime_completion_signal() {
    let mut lib = NodeLibrary::new();
    lib.register(entry_def(Some(ValueType::I64)));
    lib.register(transition_def());
    let graph = Graph {
        name: "machine".into(),
        nodes: vec![node(1, "test.entry"), node(2, "test.transition")],
        edges: vec![edge(1, 0, 2, 0)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([
            (1, ControlNode::FrameEntry),
            (
                2,
                ControlNode::Transition {
                    state: "ready".into(),
                },
            ),
        ]),
    };
    let mut program = WorkflowProgram::new();
    program.insert("machine", compile(&graph, &lib).unwrap());
    let result = WorkflowVm::new(program)
        .run_entry_resumable_async_with_env(
            "machine",
            1,
            vec![Value::I64(0)],
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        result,
        WorkflowRunStatus::Completed(WorkflowCompletion {
            values: vec![],
            transition: Some("ready".into()),
        })
    );
}

#[tokio::test]
async fn cancellation_checkpoint_contains_an_active_join() {
    let mut lib = NodeLibrary::new();
    lib.register(entry_def(Some(ValueType::I64)));
    lib.register(return_def(Some(ValueType::List(Box::new(ValueType::I64)))));
    lib.register(named_return_def(
        "test.identity_return",
        Some(ValueType::I64),
    ));
    lib.register(cancel_def());
    lib.register(fork_def(ValueType::List(Box::new(ValueType::I64))));
    let slow = Graph {
        name: "slow".into(),
        nodes: vec![
            node(1, "test.entry"),
            node(2, "test.cancel"),
            node(3, "test.identity_return"),
        ],
        edges: vec![edge(1, 0, 2, 0), edge(1, 1, 3, 1), edge(2, 0, 3, 0)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([(1, ControlNode::FrameEntry), (3, ControlNode::Return)]),
    };
    let main = Graph {
        name: "main".into(),
        nodes: vec![
            node(1, "test.entry"),
            node(2, "test.fork"),
            node(3, "test.return"),
        ],
        edges: vec![edge(1, 0, 2, 0), edge(2, 0, 3, 0), edge(2, 2, 3, 1)],
        variables: vec![],
        entries: vec![1],
        controls: BTreeMap::from([
            (1, ControlNode::FrameEntry),
            (
                2,
                ControlNode::Fork {
                    branches: vec![
                        GraphCall {
                            graph: "slow".into(),
                            entry: 1,
                        },
                        GraphCall {
                            graph: "slow".into(),
                            entry: 1,
                        },
                    ],
                    policy: JoinPolicy::All,
                    max_concurrency: 2,
                },
            ),
            (3, ControlNode::Return),
        ]),
    };
    let mut program = WorkflowProgram::new();
    program.insert("main", compile(&main, &lib).unwrap());
    program.insert("slow", compile(&slow, &lib).unwrap());
    let vm = WorkflowVm::new(program);
    let first_cancel = CancellationToken::new();
    let first = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![Value::I64(0)],
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new().with_cancel(&first_cancel),
        )
        .await
        .unwrap();
    let WorkflowRunStatus::Cancelled(state) = first else {
        panic!("cancel node should suspend with a live join");
    };
    let encoded = serde_json::to_vec(&state).unwrap();
    assert!(String::from_utf8_lossy(&encoded).contains("joins"));
    let second_cancel = CancellationToken::new();
    let second = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![],
            Some(serde_json::from_slice(&encoded).unwrap()),
            &mut InMemoryHost::default(),
            &ExecutionEnv::new().with_cancel(&second_cancel),
        )
        .await
        .unwrap();
    let WorkflowRunStatus::Cancelled(state) = second else {
        panic!("second branch should also reach its cancellation boundary");
    };
    let completed = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![],
            Some(state),
            &mut InMemoryHost::default(),
            &ExecutionEnv::new(),
        )
        .await
        .unwrap();
    assert!(matches!(completed, WorkflowRunStatus::Completed(_)));
}

#[tokio::test]
async fn workflow_suspension_round_trips_and_resumes_the_waiting_task() {
    let mut lib = NodeLibrary::new();
    lib.register(entry_def(None));
    lib.register(return_def(Some(ValueType::I64)));
    lib.register(named_return_def(
        "test.identity_return",
        Some(ValueType::I64),
    ));
    lib.register(suspend_def());
    let graph = Graph {
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
    };
    let mut program = WorkflowProgram::new();
    program.insert("main", compile(&graph, &lib).unwrap());
    let vm = WorkflowVm::new(program);
    let result = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![],
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new(),
        )
        .await
        .unwrap();
    let WorkflowRunStatus::Suspended(state) = result else {
        panic!("workflow node must suspend its task");
    };
    assert_eq!(state.suspensions()[0].id, "decision-1");

    let encoded = serde_json::to_vec(&state).unwrap();
    let mut restored: WorkflowVmState = serde_json::from_slice(&encoded).unwrap();
    restored
        .resume_suspension(
            "decision-1",
            SuspensionResolution {
                outputs: vec![Value::I64(42)],
                next: ExecNext::Pin("then".into()),
            },
        )
        .unwrap();
    let completed = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![],
            Some(restored),
            &mut InMemoryHost::default(),
            &ExecutionEnv::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        completed,
        WorkflowRunStatus::Completed(WorkflowCompletion {
            values: vec![Value::I64(42)],
            transition: None,
        })
    );
}

#[tokio::test]
async fn resumed_workflow_rejects_a_different_execution_identity() {
    let mut lib = NodeLibrary::new();
    lib.register(entry_def(None));
    lib.register(return_def(Some(ValueType::I64)));
    lib.register(suspend_def());
    let graph = Graph {
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
    };
    let mut program = WorkflowProgram::new();
    program.insert("main", compile(&graph, &lib).unwrap());
    let vm = WorkflowVm::new(program);
    let state = match vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![],
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new().with_execution_id("run-owner"),
        )
        .await
        .unwrap()
    {
        WorkflowRunStatus::Suspended(state) => state,
        _ => panic!("expected suspension checkpoint"),
    };
    let error = vm
        .run_entry_resumable_async_with_env(
            "main",
            1,
            vec![],
            Some(state),
            &mut InMemoryHost::default(),
            &ExecutionEnv::new().with_execution_id("run-attacker"),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("identity mismatch"));
}
