//! Criterion benchmarks for the Prism visual scripting compiler and slot VM.
//!
//! The benchmark keeps the graph deterministic and measures two production paths:
//! lowering an authored graph into a slot program and executing a compiled
//! control-flow chain without rebuilding the graph on the hot path.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_visual_scripting::{
    compile, Edge, Endpoint, FairScheduler, Graph, InMemoryHost, Node, NodeDefRef, NodeLibrary,
    SchedulerLimits, Vm,
};
use std::collections::HashMap;
use std::hint::black_box;

fn chain_graph(node_count: usize) -> Graph {
    assert!(node_count >= 2);
    let mut nodes = Vec::with_capacity(node_count);
    nodes.push(Node {
        id: 0,
        def: NodeDefRef {
            id: "event.begin".into(),
            version: None,
        },
        title: None,
        config: HashMap::new(),
    });
    for id in 1..node_count as u32 {
        nodes.push(Node {
            id,
            def: NodeDefRef {
                id: "flow.sequence".into(),
                version: None,
            },
            title: None,
            config: HashMap::new(),
        });
    }
    let edges = (0..(node_count - 1) as u32)
        .map(|id| Edge {
            from: Endpoint { node: id, pin: 0 },
            to: Endpoint {
                node: id + 1,
                pin: 0,
            },
        })
        .collect();
    Graph {
        name: format!("sequence-{node_count}"),
        nodes,
        edges,
        entries: vec![0],
        ..Graph::default()
    }
}

fn bench_compile(c: &mut Criterion) {
    let library = NodeLibrary::with_builtins();
    let mut group = c.benchmark_group("prism_visual_scripting/compile");
    for &count in &[8usize, 64, 256] {
        let graph = chain_graph(count);
        group.bench_with_input(BenchmarkId::from_parameter(count), &graph, |b, graph| {
            b.iter(|| {
                black_box(compile(black_box(graph), &library).expect("benchmark graph compiles"))
            });
        });
    }
    group.finish();
}

fn bench_execute(c: &mut Criterion) {
    let library = NodeLibrary::with_builtins();
    let mut group = c.benchmark_group("prism_visual_scripting/execute");
    for &count in &[8usize, 64, 256] {
        let program = compile(&chain_graph(count), &library).expect("benchmark graph compiles");
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &program,
            |b, program| {
                b.iter(|| {
                    let mut host = InMemoryHost::default();
                    black_box(
                        Vm::new(program.clone())
                            .run_all(&mut host)
                            .expect("benchmark graph executes"),
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_scheduler_admission(c: &mut Criterion) {
    let mut group = c.benchmark_group("prism_visual_scripting/scheduler_admit");
    for &count in &[64usize, 1_024, 4_096] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &count| {
            b.iter_batched(
                || {
                    let mut scheduler = FairScheduler::new(SchedulerLimits {
                        max_active: count,
                        max_active_per_tenant: count,
                        max_queued: count + 1,
                    })
                    .expect("benchmark scheduler limits are valid");
                    for index in 0..count {
                        scheduler
                            .admit(format!("run-{index}"), "bench", 0, None)
                            .expect("benchmark admission succeeds");
                    }
                    scheduler
                },
                |mut scheduler| {
                    black_box(
                        scheduler
                            .admit("run-0", "bench", 0, None)
                            .expect("duplicate lookup"),
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compile,
    bench_execute,
    bench_scheduler_admission
);
criterion_main!(benches);
