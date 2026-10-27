//! 端到端集成测试。

use prism_visual_scripting::prelude::*;
use prism_visual_scripting::{compile_to_plan, instantiate, PinIndex};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

fn make_node(id: NodeId, def: &str) -> Node {
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
fn ep(node: NodeId, pin: PinIndex) -> Endpoint {
    Endpoint { node, pin }
}
fn edge(from: Endpoint, to: Endpoint) -> Edge {
    Edge { from, to }
}

struct ContextProbe;

impl NodeRunner for ContextProbe {
    fn invoke(
        &self,
        _inputs: &[Value],
        _outputs: &mut [Value],
        ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext> {
        ctx.check_permission("graph.run", "demo")?;
        ctx.consume_budget("nodes", 1.0)?;
        ctx.record_trace(
            "probe.invoke",
            &[("node", Value::String(Arc::from("probe")))],
        );
        Ok(ExecNext::End)
    }
}

struct ConfigProbe {
    seen: Arc<Mutex<Option<Value>>>,
}

impl NodeRunner for ConfigProbe {
    fn invoke(
        &self,
        _inputs: &[Value],
        _outputs: &mut [Value],
        ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext> {
        *self.seen.lock().unwrap() = ctx.config.get("label").cloned();
        Ok(ExecNext::End)
    }
}

struct ErrorProbe;

impl NodeRunner for ErrorProbe {
    fn invoke(
        &self,
        _inputs: &[Value],
        _outputs: &mut [Value],
        _ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext> {
        Ok(ExecNext::Error("expected failure".into()))
    }
}

struct CancelProbe;

impl NodeRunner for CancelProbe {
    fn invoke(
        &self,
        _inputs: &[Value],
        _outputs: &mut [Value],
        ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext> {
        ctx.cancel.cancel();
        Ok(ExecNext::Pin("then".into()))
    }
}

struct SuspendProbe;

impl NodeRunner for SuspendProbe {
    fn invoke(
        &self,
        _inputs: &[Value],
        _outputs: &mut [Value],
        _ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext> {
        Ok(ExecNext::Suspend(Suspension::new(
            "approval-1",
            Value::String(Arc::from("approve deployment")),
        )))
    }
}

#[derive(Default)]
struct ProbeEnv {
    permissions: Mutex<Vec<(String, String)>>,
    budget: Mutex<Vec<(String, f64)>>,
    traces: Mutex<Vec<String>>,
    phases: Mutex<Vec<NodePhase>>,
}

impl PermissionGate for ProbeEnv {
    fn check_permission(&self, action: &str, scope: &str) -> VsResult<()> {
        self.permissions
            .lock()
            .unwrap()
            .push((action.to_string(), scope.to_string()));
        Ok(())
    }
}

impl BudgetMeter for ProbeEnv {
    fn consume_budget(&self, dimension: &str, amount: f64) -> VsResult<()> {
        self.budget
            .lock()
            .unwrap()
            .push((dimension.to_string(), amount));
        Ok(())
    }
}

impl TraceSink for ProbeEnv {
    fn record_trace(&self, event: &str, _attrs: &[(&str, Value)]) {
        self.traces.lock().unwrap().push(event.to_string());
    }
}

impl NodeMiddleware for ProbeEnv {
    fn on_node(&self, info: NodeCallInfo<'_>, _ctx: &mut InvokeCtx<'_>) -> VsResult<()> {
        if info.def_id == "test.context_probe" {
            self.phases.lock().unwrap().push(info.phase);
        }
        Ok(())
    }
}

fn context_probe_def() -> NodeDefinition {
    NodeDefinition {
        id: "test.context_probe".into(),
        purity: Purity::Impure,
        inputs: vec![Pin {
            name: "exec_in".into(),
            dir: PinDir::In,
            ty: ValueType::Exec,
            default: None,
        }],
        outputs: vec![],
        runner: RunnerKind::sync(ContextProbe),
    }
}

fn config_probe_def(seen: Arc<Mutex<Option<Value>>>) -> NodeDefinition {
    NodeDefinition {
        id: "test.config_probe".into(),
        purity: Purity::Impure,
        inputs: vec![Pin {
            name: "exec_in".into(),
            dir: PinDir::In,
            ty: ValueType::Exec,
            default: None,
        }],
        outputs: vec![],
        runner: RunnerKind::sync(ConfigProbe { seen }),
    }
}

fn error_probe_def() -> NodeDefinition {
    NodeDefinition {
        id: "test.error_probe".into(),
        purity: Purity::Impure,
        inputs: vec![Pin {
            name: "exec_in".into(),
            dir: PinDir::In,
            ty: ValueType::Exec,
            default: None,
        }],
        outputs: vec![
            Pin {
                name: "then".into(),
                dir: PinDir::Out,
                ty: ValueType::Exec,
                default: None,
            },
            Pin {
                name: "error".into(),
                dir: PinDir::Out,
                ty: ValueType::Exec,
                default: None,
            },
        ],
        runner: RunnerKind::sync(ErrorProbe),
    }
}

fn cancel_probe_def() -> NodeDefinition {
    NodeDefinition {
        id: "test.cancel_probe".into(),
        purity: Purity::Impure,
        inputs: vec![Pin {
            name: "exec_in".into(),
            dir: PinDir::In,
            ty: ValueType::Exec,
            default: None,
        }],
        outputs: vec![Pin {
            name: "then".into(),
            dir: PinDir::Out,
            ty: ValueType::Exec,
            default: None,
        }],
        runner: RunnerKind::sync(CancelProbe),
    }
}

fn suspend_probe_def() -> NodeDefinition {
    NodeDefinition {
        id: "test.suspend_probe".into(),
        purity: Purity::Impure,
        inputs: vec![Pin {
            name: "exec_in".into(),
            dir: PinDir::In,
            ty: ValueType::Exec,
            default: None,
        }],
        outputs: vec![
            Pin {
                name: "then".into(),
                dir: PinDir::Out,
                ty: ValueType::Exec,
                default: None,
            },
            Pin {
                name: "value".into(),
                dir: PinDir::Out,
                ty: ValueType::String,
                default: None,
            },
        ],
        runner: RunnerKind::sync(SuspendProbe),
    }
}

#[test]
fn library_registers_builtins() {
    let lib = NodeLibrary::with_builtins();
    assert!(lib.get("math.add").is_some());
    assert!(lib.get("flow.branch").is_some());
    assert!(lib.get("flow.sequence").is_some());
    assert!(lib.get("event.begin").is_some());
}

#[test]
fn end_to_end_branch_default_false() {
    // begin --> branch(cond=false default) --false--> print
    //                                     --true---> print2
    let lib = NodeLibrary::with_builtins();
    let g = Graph {
        name: "demo".into(),
        nodes: vec![
            make_node(1, "event.begin"),
            make_node(2, "flow.branch"),
            make_node(3, "debug.print"),
            make_node(4, "debug.print"),
        ],
        edges: vec![
            edge(ep(1, 0), ep(2, 0)), // begin.then -> branch.exec_in
            edge(ep(2, 0), ep(3, 0)), // branch.true  -> print3.exec_in
            edge(ep(2, 1), ep(4, 0)), // branch.false -> print4.exec_in
        ],
        entries: vec![1],
        ..Default::default()
    };

    let program = compile(&g, &lib).expect("compile");
    assert!(program.entries.contains_key(&1));

    let mut host = InMemoryHost::default();
    Vm::new(program).run_all(&mut host).expect("run");
    // cond 默认 false → 走 false 分支 → print4 触发一次
    assert_eq!(host.log_buffer.len(), 1);
}

#[test]
fn execution_env_injects_capabilities_and_middleware() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(context_probe_def());

    let g = Graph {
        nodes: vec![
            make_node(1, "event.begin"),
            make_node(2, "test.context_probe"),
        ],
        edges: vec![edge(ep(1, 0), ep(2, 0))],
        entries: vec![1],
        ..Default::default()
    };

    let vm = Vm::new(compile(&g, &lib).unwrap());
    let mut host = InMemoryHost::default();
    let probe = ProbeEnv::default();
    let env = ExecutionEnv::new()
        .with_permissions(&probe)
        .with_budget(&probe)
        .with_trace(&probe)
        .with_middleware(&probe);

    vm.run_all_with_env(&mut host, &env).unwrap();

    assert_eq!(
        probe.permissions.lock().unwrap().as_slice(),
        &[("graph.run".into(), "demo".into())]
    );
    assert_eq!(
        probe.budget.lock().unwrap().as_slice(),
        &[("nodes".into(), 1.0)]
    );
    assert_eq!(
        probe.traces.lock().unwrap().as_slice(),
        &["probe.invoke".to_string()]
    );
    assert_eq!(
        probe.phases.lock().unwrap().as_slice(),
        &[NodePhase::Before, NodePhase::After]
    );
}

#[test]
fn impure_node_persists_state_across_runs() {
    let lib = NodeLibrary::with_builtins();
    let g = Graph {
        nodes: vec![make_node(1, "event.begin"), make_node(2, "var.inc_counter")],
        edges: vec![edge(ep(1, 0), ep(2, 0))],
        entries: vec![1],
        ..Default::default()
    };

    let vm = Vm::new(compile(&g, &lib).unwrap());
    let mut host = InMemoryHost::default();
    for _ in 0..3 {
        vm.run_all(&mut host).unwrap();
    }
    let counter = host.vars.get("counter").and_then(|v| v.as_f64()).unwrap();
    assert_eq!(counter, 3.0);
}

#[test]
fn type_mismatch_is_caught() {
    let lib = NodeLibrary::with_builtins();
    let g = Graph {
        nodes: vec![make_node(1, "event.begin"), make_node(2, "cmp.greater")],
        edges: vec![edge(ep(1, 0), ep(2, 0))], // exec -> data
        ..Default::default()
    };
    let err = match compile(&g, &lib) {
        Ok(_) => panic!("expected error"),
        Err(error) => error,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("exec/data") || msg.contains("type"),
        "got {msg}"
    );
}

#[test]
fn cycle_in_pure_subgraph_is_rejected() {
    let lib = NodeLibrary::with_builtins();
    let g = Graph {
        nodes: vec![make_node(1, "math.add")],
        edges: vec![edge(ep(1, 0), ep(1, 0))], // 自环
        ..Default::default()
    };
    let err = match compile(&g, &lib) {
        Ok(_) => panic!("expected error"),
        Err(error) => error,
    };
    let msg = format!("{err}");
    assert!(msg.contains("cycle") || msg.contains("Cycle"), "got {msg}");
}

#[test]
fn node_instance_config_reaches_runner() {
    let seen = Arc::new(Mutex::new(None));
    let mut lib = NodeLibrary::with_builtins();
    lib.register(config_probe_def(seen.clone()));
    let mut node = make_node(2, "test.config_probe");
    node.config
        .insert("label".into(), Value::String(Arc::from("configured")));
    let g = Graph {
        nodes: vec![make_node(1, "event.begin"), node],
        edges: vec![edge(ep(1, 0), ep(2, 0))],
        entries: vec![1],
        ..Default::default()
    };

    let vm = Vm::new(compile(&g, &lib).unwrap());
    vm.run_all(&mut InMemoryHost::default()).unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        Some(Value::String(Arc::from("configured")))
    );
}

#[test]
fn structured_error_uses_error_exec_pin() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(error_probe_def());
    let g = Graph {
        nodes: vec![
            make_node(1, "event.begin"),
            make_node(2, "test.error_probe"),
            make_node(3, "debug.print"),
        ],
        edges: vec![edge(ep(1, 0), ep(2, 0)), edge(ep(2, 1), ep(3, 0))],
        entries: vec![1],
        ..Default::default()
    };

    let vm = Vm::new(compile(&g, &lib).unwrap());
    let mut host = InMemoryHost::default();
    vm.run_all(&mut host).unwrap();
    assert_eq!(host.log_buffer.len(), 1);
}

#[test]
fn cancellation_returns_serializable_state_that_resumes_at_next_block() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(cancel_probe_def());
    let g = Graph {
        nodes: vec![
            make_node(1, "event.begin"),
            make_node(2, "test.cancel_probe"),
            make_node(3, "debug.print"),
        ],
        edges: vec![edge(ep(1, 0), ep(2, 0)), edge(ep(2, 0), ep(3, 0))],
        entries: vec![1],
        ..Default::default()
    };

    let vm = Vm::new(compile(&g, &lib).unwrap());
    let cancel = CancellationToken::new();
    let env = ExecutionEnv::new().with_cancel(&cancel);
    let state = match vm
        .run_entry_resumable_with_env(1, None, &mut InMemoryHost::default(), &env)
        .unwrap()
    {
        VmRunStatus::Cancelled(state) => state,
        VmRunStatus::Suspended(_) => panic!("cancel probe must not create a node suspension"),
        VmRunStatus::Completed => panic!("cancel probe must suspend"),
    };
    let encoded = serde_json::to_vec(&state).unwrap();
    let restored: VmState = serde_json::from_slice(&encoded).unwrap();
    let resume_cancel = CancellationToken::new();
    let resume_env = ExecutionEnv::new().with_cancel(&resume_cancel);
    let mut host = InMemoryHost::default();
    assert_eq!(
        vm.run_entry_resumable_with_env(1, Some(restored), &mut host, &resume_env)
            .unwrap(),
        VmRunStatus::Completed
    );
    assert_eq!(host.log_buffer.len(), 1);
}

#[test]
fn resumed_vm_rejects_a_different_execution_identity() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(cancel_probe_def());
    let g = Graph {
        nodes: vec![
            make_node(1, "event.begin"),
            make_node(2, "test.cancel_probe"),
            make_node(3, "debug.print"),
        ],
        edges: vec![edge(ep(1, 0), ep(2, 0)), edge(ep(2, 0), ep(3, 0))],
        entries: vec![1],
        ..Default::default()
    };
    let vm = Vm::new(compile(&g, &lib).unwrap());
    let cancel = CancellationToken::new();
    let state = match vm
        .run_entry_resumable_with_env(
            1,
            None,
            &mut InMemoryHost::default(),
            &ExecutionEnv::new()
                .with_execution_id("run-owner")
                .with_cancel(&cancel),
        )
        .unwrap()
    {
        VmRunStatus::Cancelled(state) => state,
        status => panic!("expected cancellation checkpoint, got {status:?}"),
    };
    let error = vm
        .run_entry_resumable_with_env(
            1,
            Some(state),
            &mut InMemoryHost::default(),
            &ExecutionEnv::new().with_execution_id("run-attacker"),
        )
        .unwrap_err();
    assert!(error.to_string().contains("execution identity mismatch"));
}

#[test]
fn node_suspension_serializes_and_injects_outputs_before_resuming() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(suspend_probe_def());
    let graph = Graph {
        nodes: vec![
            make_node(1, "event.begin"),
            make_node(2, "test.suspend_probe"),
            make_node(3, "debug.print"),
        ],
        edges: vec![
            edge(ep(1, 0), ep(2, 0)),
            edge(ep(2, 0), ep(3, 0)),
            edge(ep(2, 1), ep(3, 1)),
        ],
        entries: vec![1],
        ..Default::default()
    };
    let vm = Vm::new(compile(&graph, &lib).unwrap());

    let result = vm
        .run_entry_resumable_with_env(1, None, &mut InMemoryHost::default(), &ExecutionEnv::new())
        .unwrap();
    let VmRunStatus::Suspended(state) = result else {
        panic!("node must suspend the VM");
    };
    assert_eq!(state.suspension().unwrap().id, "approval-1");

    let encoded = serde_json::to_vec(&state).unwrap();
    let mut restored: VmState = serde_json::from_slice(&encoded).unwrap();
    restored
        .resume_suspension(
            "approval-1",
            SuspensionResolution {
                outputs: vec![Value::String(Arc::from("approved"))],
                next: ExecNext::Pin("then".into()),
            },
        )
        .unwrap();
    let mut host = InMemoryHost::default();
    assert_eq!(
        vm.run_entry_resumable_with_env(1, Some(restored), &mut host, &ExecutionEnv::new(),)
            .unwrap(),
        VmRunStatus::Completed
    );
    assert_eq!(host.log_buffer[0].1, "approved");
}

#[test]
fn non_resumable_api_does_not_report_node_suspension_as_cancellation() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(suspend_probe_def());
    let graph = Graph {
        nodes: vec![
            make_node(1, "event.begin"),
            make_node(2, "test.suspend_probe"),
        ],
        edges: vec![edge(ep(1, 0), ep(2, 0))],
        entries: vec![1],
        ..Default::default()
    };
    let error = Vm::new(compile(&graph, &lib).unwrap())
        .run_all(&mut InMemoryHost::default())
        .unwrap_err();
    assert!(matches!(error, VsError::Suspended(id) if id == "approval-1"));
}

#[test]
fn execution_plan_rejects_a_different_registry_snapshot() {
    let base = NodeLibrary::with_builtins();
    let graph = Graph {
        nodes: vec![make_node(1, "event.begin")],
        entries: vec![1],
        ..Default::default()
    };
    let plan = compile_to_plan(&graph, &base).unwrap();

    let mut changed = NodeLibrary::with_builtins();
    changed.register(context_probe_def());
    let error = match instantiate(&plan, &changed) {
        Ok(_) => panic!("changed registry must be rejected"),
        Err(error) => error,
    };
    assert!(format!("{error}").contains("registry identity"));
}

#[test]
fn compiler_rejects_duplicate_node_ids() {
    let lib = NodeLibrary::with_builtins();
    let graph = Graph {
        nodes: vec![make_node(1, "event.begin"), make_node(1, "event.begin")],
        entries: vec![1],
        ..Default::default()
    };
    let error = match compile(&graph, &lib) {
        Ok(_) => panic!("duplicate node ids must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, VsError::Validate(message) if message.contains("duplicate node id")));
}

#[test]
fn compiler_rejects_unknown_graph_entry() {
    let lib = NodeLibrary::with_builtins();
    let graph = Graph {
        nodes: vec![make_node(1, "event.begin")],
        entries: vec![99],
        ..Default::default()
    };
    let error = match compile(&graph, &lib) {
        Ok(_) => panic!("unknown graph entry must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, VsError::UnknownNode(99)));
}

#[test]
fn compiler_rejects_pure_graph_entry() {
    let lib = NodeLibrary::with_builtins();
    let graph = Graph {
        nodes: vec![make_node(1, "math.add")],
        entries: vec![1],
        ..Default::default()
    };
    let error = match compile(&graph, &lib) {
        Ok(_) => panic!("pure graph entry must be rejected"),
        Err(error) => error,
    };
    assert!(
        matches!(error, VsError::Validate(message) if message.contains("must reference an impure node"))
    );
}

#[test]
fn instantiate_rejects_out_of_bounds_ir_slots() {
    let lib = NodeLibrary::with_builtins();
    let graph = Graph {
        nodes: vec![make_node(1, "event.begin")],
        entries: vec![1],
        ..Default::default()
    };
    let mut plan = compile_to_plan(&graph, &lib).expect("plan");
    plan.blocks[0].instrs.push(Instr::Move { dst: 1, src: 0 });
    let error = match instantiate(&plan, &lib) {
        Ok(_) => panic!("invalid IR slot must be rejected"),
        Err(error) => error,
    };
    assert!(
        matches!(error, VsError::Validate(message) if message.contains("slot range out of bounds"))
    );
}
