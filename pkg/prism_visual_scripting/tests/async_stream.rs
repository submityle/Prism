//! 异步 / 流式 / 取消 集成测试。

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use prism_visual_scripting::PinIndex;
use prism_visual_scripting::prelude::*;

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

/// 异步节点：把固定 chunks 推到 chunk_tx，然后 sleep 让出。
struct StreamerNode {
    deltas: Vec<String>,
}

#[async_trait]
impl AsyncNodeRunner for StreamerNode {
    async fn invoke(
        &self,
        _inputs: &[Value],
        _outputs: &mut [Value],
        ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext> {
        for s in &self.deltas {
            send_chunk(ctx, Chunk::Delta(Value::String(Arc::from(s.as_str())))).await;
            tokio::task::yield_now().await;
        }
        Ok(ExecNext::Pin("then".into()))
    }
}

/// 异步阻塞节点：等到 cancel 触发或永远等下去。
struct ForeverNode;

#[async_trait]
impl AsyncNodeRunner for ForeverNode {
    async fn invoke(
        &self,
        _inputs: &[Value],
        _outputs: &mut [Value],
        ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext> {
        ctx.cancel.cancelled().await;
        Err(VsError::Cancelled)
    }
}

fn streamer_def(deltas: Vec<&str>) -> NodeDefinition {
    use prism_visual_scripting::Pin;
    use prism_visual_scripting::PinDir;
    NodeDefinition {
        id: "test.streamer".into(),
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
        runner: RunnerKind::r#async(StreamerNode {
            deltas: deltas.into_iter().map(String::from).collect(),
        }),
    }
}

fn forever_def() -> NodeDefinition {
    use prism_visual_scripting::Pin;
    use prism_visual_scripting::PinDir;
    NodeDefinition {
        id: "test.forever".into(),
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
        runner: RunnerKind::r#async(ForeverNode),
    }
}

#[tokio::test]
async fn async_runner_in_sync_vm_errors() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(streamer_def(vec!["a"]));

    let g = Graph {
        nodes: vec![make_node(1, "event.begin"), make_node(2, "test.streamer")],
        edges: vec![edge(ep(1, 0), ep(2, 0))],
        entries: vec![1],
        ..Default::default()
    };

    let program = compile(&g, &lib).expect("compile");
    let vm = Vm::new(program);
    let mut host = InMemoryHost::default();
    let err = vm.run_all(&mut host).expect_err("must error");
    assert!(matches!(err, VsError::AsyncRunnerInSyncVm), "got {err:?}");
}

#[tokio::test]
async fn async_vm_streams_chunks_from_node() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(streamer_def(vec!["hello", "world"]));

    let g = Graph {
        nodes: vec![make_node(1, "event.begin"), make_node(2, "test.streamer")],
        edges: vec![edge(ep(1, 0), ep(2, 0))],
        entries: vec![1],
        ..Default::default()
    };

    let program = compile(&g, &lib).expect("compile");
    let vm = Vm::new(program);
    let host = Arc::new(Mutex::new(InMemoryHost::default()));
    let cancel = CancellationToken::new();
    let (tx, mut rx) = mpsc::channel::<Chunk>(8);

    let host_clone = host.clone();
    let cancel_clone = cancel.clone();
    let task = tokio::spawn(async move {
        let mut h = host_clone.lock().await;
        vm.run_all_async(&mut *h, &cancel_clone, Some(&tx)).await
    });

    let mut deltas: Vec<String> = Vec::new();
    while let Some(c) = rx.recv().await {
        if let Chunk::Delta(Value::String(s)) = c {
            deltas.push(s.to_string());
        }
    }
    task.await.unwrap().expect("run ok");
    assert_eq!(deltas, vec!["hello".to_string(), "world".to_string()]);
}

#[tokio::test]
async fn async_vm_respects_cancel() {
    let mut lib = NodeLibrary::with_builtins();
    lib.register(forever_def());

    let g = Graph {
        nodes: vec![make_node(1, "event.begin"), make_node(2, "test.forever")],
        edges: vec![edge(ep(1, 0), ep(2, 0))],
        entries: vec![1],
        ..Default::default()
    };

    let program = compile(&g, &lib).expect("compile");
    let vm = Arc::new(Vm::new(program));
    let host = Arc::new(Mutex::new(InMemoryHost::default()));
    let cancel = CancellationToken::new();

    let vm_clone = vm.clone();
    let host_clone = host.clone();
    let cancel_clone = cancel.clone();
    let task = tokio::spawn(async move {
        let mut h = host_clone.lock().await;
        vm_clone.run_all_async(&mut *h, &cancel_clone, None).await
    });

    // 让节点先进入 await 状态，再触发取消。
    tokio::task::yield_now().await;
    cancel.cancel();

    let res = task.await.unwrap();
    assert!(matches!(res, Err(VsError::Cancelled)), "got {res:?}");
}

#[tokio::test]
async fn async_vm_runs_sync_runners_too() {
    // 复用内置全部 sync 节点，验证 async VM 能跑同步图。
    let lib = NodeLibrary::with_builtins();
    let g = Graph {
        nodes: vec![make_node(1, "event.begin"), make_node(2, "var.inc_counter")],
        edges: vec![edge(ep(1, 0), ep(2, 0))],
        entries: vec![1],
        ..Default::default()
    };

    let vm = Vm::new(compile(&g, &lib).unwrap());
    let mut host = InMemoryHost::default();
    let cancel = CancellationToken::new();
    vm.run_all_async(&mut host, &cancel, None).await.unwrap();
    let counter = host.vars.get("counter").and_then(|v| v.as_f64()).unwrap();
    assert_eq!(counter, 1.0);
}
