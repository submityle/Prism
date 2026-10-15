use super::*;

fn resource(name: &'static str, lifetime: ResourceLifetime, size: u64) -> ResourceDescriptor {
    ResourceDescriptor {
        name: name.into(),
        lifetime,
        size,
        alignment: 256,
    }
}

fn aligned_resource(
    name: &'static str,
    lifetime: ResourceLifetime,
    size: u64,
    alignment: u64,
) -> ResourceDescriptor {
    ResourceDescriptor {
        name: name.into(),
        lifetime,
        size,
        alignment,
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
    assert_eq!(compiled.order, vec![cull.0 as usize, draw.0 as usize]);
    assert_eq!(compiled.logical_barriers.len(), 1);
    assert_eq!(compiled.barrier_count(), 2);
    assert!(compiled.logical_barriers[0].queue_transfer);
    assert_eq!(compiled.queue_batches.len(), 2);
    assert_eq!(compiled.queue_batches[1].waits_for, vec![0]);
    assert_eq!(
        compiled.alias.buffer_slots[visible.0 as usize]
            .unwrap()
            .offset,
        0
    );
    assert_eq!(compiled.alias.heap_size, 1024);
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
    match graph.compile().unwrap_err() {
        CompileError::Cycle {
            unscheduled,
            blocked,
            cycle,
            names,
        } => {
            assert_eq!(unscheduled, 2);
            assert_eq!(blocked, vec![0, 1]);
            assert_eq!(cycle.first(), cycle.last());
            assert_eq!(cycle.len(), 3);
            assert!(names.iter().any(|name| name == "a"));
            assert!(names.iter().any(|name| name == "b"));
        }
        error => panic!("expected cycle diagnostics, got {error:?}"),
    }
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
            .order
            .iter()
            .position(|id| *id == write.0 as usize)
            .unwrap()
            > compiled
                .order
                .iter()
                .position(|id| *id == read_a.0 as usize)
                .unwrap()
    );
    assert!(
        compiled
            .order
            .iter()
            .position(|id| *id == write.0 as usize)
            .unwrap()
            > compiled
                .order
                .iter()
                .position(|id| *id == read_b.0 as usize)
                .unwrap()
    );
    assert!(compiled.logical_barriers.iter().any(|barrier| {
        barrier.source == read_a.0 as usize && barrier.destination == write.0 as usize
    }));
    assert!(compiled.logical_barriers.iter().any(|barrier| {
        barrier.source == read_b.0 as usize && barrier.destination == write.0 as usize
    }));
    assert_eq!(
        compiled.alias.buffer_slots[0].map(|slot| slot.offset),
        compiled.alias.buffer_slots[1].map(|slot| slot.offset)
    );
    assert_eq!(compiled.alias.heap_size, 2048);
}

#[test]
fn transient_plan_aliases_disjoint_transients_into_one_group() {
    // Two same-size transients written by two passes running back to back.
    // Their lifetimes are disjoint, so first-fit reuses one heap block: same
    // offset, one alias group, and savings equal to one resource's footprint.
    let mut graph = GpuFrameGraphBuilder::default();
    let a = graph.add_resource(resource("a", ResourceLifetime::Transient, 1024));
    let b = graph.add_resource(resource("b", ResourceLifetime::Transient, 1024));
    let write_a = graph.add_pass(PassDescriptor {
        name: "write a".into(),
        queue: QueueClass::Compute,
        depends_on: vec![],
        accesses: vec![ResourceAccess {
            resource: a,
            kind: AccessKind::StorageWrite,
        }],
    });
    graph.add_pass(PassDescriptor {
        name: "write b".into(),
        queue: QueueClass::Compute,
        depends_on: vec![write_a],
        accesses: vec![ResourceAccess {
            resource: b,
            kind: AccessKind::StorageWrite,
        }],
    });
    let compiled = graph.compile().unwrap();
    let slot_a = compiled.alias.buffer_slots[a.0 as usize].unwrap();
    let slot_b = compiled.alias.buffer_slots[b.0 as usize].unwrap();
    assert_eq!(slot_a.offset, slot_b.offset);
    assert_eq!(compiled.alias.heap_size, 1024);
}

#[test]
fn transient_plan_keeps_overlapping_transients_separate() {
    // Both transients are live across the same pass, so their lifetimes overlap
    // and cannot alias: distinct offsets, distinct singleton groups, no savings.
    let mut graph = GpuFrameGraphBuilder::default();
    let a = graph.add_resource(resource("a", ResourceLifetime::Transient, 1024));
    let b = graph.add_resource(resource("b", ResourceLifetime::Transient, 1024));
    graph.add_pass(PassDescriptor {
        name: "write both".into(),
        queue: QueueClass::Compute,
        depends_on: vec![],
        accesses: vec![
            ResourceAccess {
                resource: a,
                kind: AccessKind::StorageWrite,
            },
            ResourceAccess {
                resource: b,
                kind: AccessKind::StorageWrite,
            },
        ],
    });
    let compiled = graph.compile().unwrap();
    let slot_a = compiled.alias.buffer_slots[a.0 as usize].unwrap();
    let slot_b = compiled.alias.buffer_slots[b.0 as usize].unwrap();
    assert_ne!(slot_a.offset, slot_b.offset);
    assert_eq!(compiled.alias.heap_size, 2048);
}

#[test]
fn transient_plan_respects_alignment_when_growing_heap() {
    // Two overlapping transients with the unified compiler's 256-byte heap
    // alignment. The first sits at 0 with an aligned 512-byte slot; the second
    // starts at the next aligned slot rather than packing at byte 300.
    let mut graph = GpuFrameGraphBuilder::default();
    let a = graph.add_resource(aligned_resource("a", ResourceLifetime::Transient, 300, 512));
    let b = graph.add_resource(aligned_resource("b", ResourceLifetime::Transient, 300, 512));
    graph.add_pass(PassDescriptor {
        name: "write both".into(),
        queue: QueueClass::Compute,
        depends_on: vec![],
        accesses: vec![
            ResourceAccess {
                resource: a,
                kind: AccessKind::StorageWrite,
            },
            ResourceAccess {
                resource: b,
                kind: AccessKind::StorageWrite,
            },
        ],
    });
    let compiled = graph.compile().unwrap();
    let slot_a = compiled.alias.buffer_slots[a.0 as usize].unwrap();
    let slot_b = compiled.alias.buffer_slots[b.0 as usize].unwrap();
    assert_eq!(slot_a.offset, 0);
    assert_eq!(slot_b.offset, 512);
    assert_eq!(slot_b.offset % 256, 0);
    assert_eq!(compiled.alias.heap_size, 1024);
}

#[test]
fn transient_plan_excludes_persistent_and_imported_resources() {
    // Only the transient resource earns a region; persistent and imported ones
    // stay out of the heap plan entirely.
    let mut graph = GpuFrameGraphBuilder::default();
    let persistent = graph.add_resource(resource("persistent", ResourceLifetime::Persistent, 4096));
    let imported = graph.add_resource(resource("imported", ResourceLifetime::Imported, 8192));
    let transient = graph.add_resource(resource("transient", ResourceLifetime::Transient, 1024));
    graph.add_pass(PassDescriptor {
        name: "touch all".into(),
        queue: QueueClass::Compute,
        depends_on: vec![],
        accesses: vec![
            ResourceAccess {
                resource: persistent,
                kind: AccessKind::StorageRead,
            },
            ResourceAccess {
                resource: imported,
                kind: AccessKind::StorageRead,
            },
            ResourceAccess {
                resource: transient,
                kind: AccessKind::StorageWrite,
            },
        ],
    });
    let compiled = graph.compile().unwrap();
    assert!(compiled.alias.buffer_slots[persistent.0 as usize].is_none());
    assert!(compiled.alias.buffer_slots[imported.0 as usize].is_none());
    assert!(compiled.alias.buffer_slots[transient.0 as usize].is_some());
    assert_eq!(compiled.alias.heap_size, 1024);
}

#[test]
fn transient_plan_alias_safety_holds_across_a_mixed_graph() {
    // A larger graph mixing reuse and overlap; the alias-safety invariant must
    // hold for every group the planner produced.
    let mut graph = GpuFrameGraphBuilder::default();
    let a = graph.add_resource(resource("a", ResourceLifetime::Transient, 1024));
    let b = graph.add_resource(resource("b", ResourceLifetime::Transient, 1024));
    let c = graph.add_resource(resource("c", ResourceLifetime::Transient, 1024));
    let write_ab = graph.add_pass(PassDescriptor {
        name: "write a+b".into(),
        queue: QueueClass::Compute,
        depends_on: vec![],
        accesses: vec![
            ResourceAccess {
                resource: a,
                kind: AccessKind::StorageWrite,
            },
            ResourceAccess {
                resource: b,
                kind: AccessKind::StorageWrite,
            },
        ],
    });
    // c is written only after a and b are done, so it may reuse a block.
    graph.add_pass(PassDescriptor {
        name: "write c".into(),
        queue: QueueClass::Compute,
        depends_on: vec![write_ab],
        accesses: vec![ResourceAccess {
            resource: c,
            kind: AccessKind::StorageWrite,
        }],
    });
    let compiled = graph.compile().unwrap();
    let slot_a = compiled.alias.buffer_slots[a.0 as usize].unwrap();
    let slot_b = compiled.alias.buffer_slots[b.0 as usize].unwrap();
    let slot_c = compiled.alias.buffer_slots[c.0 as usize].unwrap();
    assert_ne!(slot_a.offset, slot_b.offset);
    assert!(slot_c.offset == slot_a.offset || slot_c.offset == slot_b.offset);
    // c reuses one of the two earlier blocks, so the heap stays at two slots.
    assert_eq!(compiled.alias.heap_size, 2048);
}
