use super::*;

fn resource(name: &'static str, lifetime: ResourceLifetime, size: u64) -> ResourceDescriptor {
    ResourceDescriptor {
        name: name.into(),
        lifetime,
        size,
        alignment: 256,
    }
}

#[test]
fn compiles_hazards_versions_queues_and_transient_layout() {
    let mut graph = GpuFrameGraphBuilder::default();
    let scene = graph.add_resource(resource("scene", ResourceLifetime::Persistent, 4096));
    let visible = graph.add_resource(resource("visible", ResourceLifetime::Transient, 1024));
    let cull = graph.add_pass(PassDescriptor {
        name: "cull".into(),
        queue: QueueClass::Compute,
        depends_on: vec![],
        accesses: vec![
            ResourceAccess {
                resource: scene,
                kind: AccessKind::StorageRead,
            },
            ResourceAccess {
                resource: visible,
                kind: AccessKind::StorageWrite,
            },
        ],
    });
    let draw = graph.add_pass(PassDescriptor {
        name: "draw".into(),
        queue: QueueClass::Graphics,
        depends_on: vec![],
        accesses: vec![ResourceAccess {
            resource: visible,
            kind: AccessKind::IndirectRead,
        }],
    });
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.execution_order, vec![cull, draw]);
    assert_eq!(compiled.resource_versions.len(), 1);
    assert_eq!(compiled.barriers.len(), 1);
    assert!(compiled.barriers[0].queue_transfer);
    assert_eq!(compiled.queue_batches.len(), 2);
    assert_eq!(compiled.queue_batches[1].waits_for, vec![0]);
    assert_eq!(compiled.transient_offsets[visible.0 as usize], Some(0));
    assert_eq!(compiled.transient_bytes, 1024);
}

#[test]
fn rejects_invalid_and_cyclic_graphs() {
    let mut graph = GpuFrameGraphBuilder::default();
    graph.add_resource(resource("x", ResourceLifetime::Imported, 0));
    graph.add_pass(PassDescriptor {
        name: "bad".into(),
        queue: QueueClass::Graphics,
        accesses: vec![],
        depends_on: vec![PassId(0)],
    });
    assert!(matches!(
        graph.compile(),
        Err(CompileError::InvalidDependency { .. })
    ));

    graph.clear();
    graph.add_resource(resource("x", ResourceLifetime::Imported, 0));
    graph.add_pass(PassDescriptor {
        name: "a".into(),
        queue: QueueClass::Graphics,
        accesses: vec![],
        depends_on: vec![PassId(1)],
    });
    graph.add_pass(PassDescriptor {
        name: "b".into(),
        queue: QueueClass::Compute,
        accesses: vec![],
        depends_on: vec![PassId(0)],
    });
    assert_eq!(graph.compile().unwrap_err(), CompileError::CyclicDependency);
}

#[test]
fn write_waits_for_all_prior_readers_and_aliases_disjoint_transients() {
    let mut graph = GpuFrameGraphBuilder::default();
    let shared = graph.add_resource(resource("shared", ResourceLifetime::Persistent, 256));
    let first = graph.add_resource(resource("first", ResourceLifetime::Transient, 2048));
    let second = graph.add_resource(resource("second", ResourceLifetime::Transient, 1024));
    let read_a = graph.add_pass(PassDescriptor {
        name: "read a".into(),
        queue: QueueClass::Compute,
        depends_on: vec![],
        accesses: vec![
            ResourceAccess {
                resource: shared,
                kind: AccessKind::StorageRead,
            },
            ResourceAccess {
                resource: first,
                kind: AccessKind::StorageWrite,
            },
        ],
    });
    let read_b = graph.add_pass(PassDescriptor {
        name: "read b".into(),
        queue: QueueClass::Graphics,
        depends_on: vec![],
        accesses: vec![ResourceAccess {
            resource: shared,
            kind: AccessKind::SampledRead,
        }],
    });
    let write = graph.add_pass(PassDescriptor {
        name: "write".into(),
        queue: QueueClass::Transfer,
        depends_on: vec![],
        accesses: vec![ResourceAccess {
            resource: shared,
            kind: AccessKind::TransferWrite,
        }],
    });
    graph.add_pass(PassDescriptor {
        name: "second transient".into(),
        queue: QueueClass::Transfer,
        depends_on: vec![write],
        accesses: vec![ResourceAccess {
            resource: second,
            kind: AccessKind::TransferWrite,
        }],
    });
    let compiled = graph.compile().unwrap();
    assert!(
        compiled
            .execution_order
            .iter()
            .position(|id| *id == write)
            .unwrap()
            > compiled
                .execution_order
                .iter()
                .position(|id| *id == read_a)
                .unwrap()
    );
    assert!(
        compiled
            .execution_order
            .iter()
            .position(|id| *id == write)
            .unwrap()
            > compiled
                .execution_order
                .iter()
                .position(|id| *id == read_b)
                .unwrap()
    );
    assert!(compiled
        .barriers
        .iter()
        .any(|barrier| barrier.source == read_a && barrier.destination == write));
    assert!(compiled
        .barriers
        .iter()
        .any(|barrier| barrier.source == read_b && barrier.destination == write));
    assert_eq!(
        compiled.transient_offsets[first.0 as usize],
        compiled.transient_offsets[second.0 as usize]
    );
    assert_eq!(compiled.transient_bytes, 2048);
}
