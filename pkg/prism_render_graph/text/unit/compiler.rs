//! Shared-kernel and frontend admission regressions.
use crate::capacity::GraphLimits;
use crate::compile::kernel::{self, Access, AllocationRequest};
use crate::frame_graph::{
    AccessInput, AccessKind, GpuFrameGraphBuilder, PassDescriptor, PassId, QueueBatch, QueueClass,
    ResourceAccess, ResourceDescriptor, ResourceId, ResourceLifetime,
};
use crate::{
    Blackboard, BufferDesc, BufferHandle, BufferUse, CompileError, CompiledAccess, LogicalBarrier,
    LogicalResource, PassBuilder, PassFlags, PassKind, RenderGraph, ResourceIndex,
};
use alloc::collections::BTreeSet;
use prism_render_driver::{BufferUsages, Extent3d};

fn resource(size: u64, alignment: u64) -> ResourceDescriptor {
    ResourceDescriptor {
        name: "resource".into(),
        lifetime: ResourceLifetime::Transient,
        size,
        alignment,
    }
}

fn pass(
    name: &'static str,
    queue: QueueClass,
    accesses: Vec<ResourceAccess>,
    depends_on: Vec<PassId>,
) -> PassDescriptor {
    PassDescriptor {
        name: name.into(),
        queue,
        accesses,
        depends_on,
    }
}

fn access(resource: ResourceId, kind: AccessKind) -> ResourceAccess {
    ResourceAccess { resource, kind }
}

/// Exercise natural handle bounds without allocating enormous resource tables.
#[test]
fn limits_reject_before_id_narrowing_or_setup() {
    assert!(GraphLimits::check_count("passes", u32::MAX as usize, usize::MAX).is_ok());
    if usize::BITS > 32 {
        assert!(matches!(
            GraphLimits::check_count("passes", u32::MAX as usize + 1, usize::MAX),
            Err(CompileError::Capacity { .. })
        ));
    }
    assert!(GraphLimits::next_id("resources", usize::MAX, usize::MAX).is_err());
    assert_eq!(
        GraphLimits::default().next_version(u32::MAX),
        Err(CompileError::VersionOverflow {
            version: u32::MAX,
            limit: u32::MAX
        })
    );
    let limits = GraphLimits {
        resources: 1,
        passes: 0,
        ..GraphLimits::default()
    };
    let mut graph = RenderGraph::with_limits(Extent3d::new_2d(1, 1), limits);
    let called = core::cell::Cell::new(false);
    assert!(matches!(
        graph.try_add_compute_pass("rejected", PassFlags::default(), |_| {
            called.set(true);
            |_: &mut crate::ExecuteContext<'_>| {}
        }),
        Err(CompileError::Capacity { kind: "passes", .. })
    ));
    assert!(!called.get());
    assert_eq!(graph.pass_count(), 0);
    let mut declaration = GpuFrameGraphBuilder::with_limits(limits);
    declaration.try_add_resource(resource(256, 256)).unwrap();
    assert!(declaration.try_add_resource(resource(256, 256)).is_err());
    assert_eq!(declaration.resources().len(), 1);
}

/// Failed admissions must not consume an access slot, version or usage bit.
#[test]
fn checked_access_rejection_is_atomic() {
    let mut textures = Vec::new();
    let mut buffers = Vec::new();
    let mut blackboard = Blackboard::new();
    let limits = GraphLimits {
        resources: 1,
        accesses_per_pass: 3,
        versions: 1,
        ..GraphLimits::default()
    };
    let mut builder = PassBuilder::new(
        &mut textures,
        &mut buffers,
        &mut blackboard,
        PassKind::Compute,
        limits,
    );
    let zero = builder
        .try_create_buffer("buffer", BufferDesc::new(256, BufferUsages::NONE))
        .unwrap();
    assert!(builder
        .try_create_persistent_buffer("overflow", BufferDesc::new(256, BufferUsages::NONE))
        .is_err());
    assert!(matches!(
        builder.try_use_buffer(
            BufferHandle::new(ResourceIndex(1), 0),
            BufferUse::StorageRead
        ),
        Err(CompileError::InvalidHandle { .. })
    ));
    assert!(matches!(
        builder.try_use_buffer(
            BufferHandle::new(zero.resource(), 2),
            BufferUse::StorageRead
        ),
        Err(CompileError::InvalidHandle { .. })
    ));
    let one = builder
        .try_use_buffer(zero, BufferUse::StorageWrite)
        .unwrap();
    assert_eq!(one.version(), 1);
    assert!(matches!(
        builder.try_use_buffer(zero, BufferUse::CopyDst),
        Err(CompileError::VersionOverflow { .. })
    ));
    builder
        .try_use_buffer(zero, BufferUse::StorageRead)
        .unwrap(); // stale reads remain legal
    builder.try_use_buffer(one, BufferUse::StorageRead).unwrap();
    assert!(matches!(
        builder.try_use_buffer(one, BufferUse::Uniform),
        Err(CompileError::Capacity {
            kind: "accesses",
            ..
        })
    ));
    let (_, accesses) = builder.finish();
    assert_eq!(accesses.len(), 3);
    assert_eq!(buffers.len(), 1);
    assert_eq!(buffers[0].version, 1);
    assert_eq!(buffers[0].inferred_usage, BufferUsages::STORAGE);
}

/// Malformed declarations are rejected before shared-kernel indexed lookups.
#[test]
fn declaration_validates_resources_dependencies_and_versions() {
    let mut graph = GpuFrameGraphBuilder::default();
    let r = graph.add_resource(resource(256, 256));
    graph.add_pass(pass(
        "invalid",
        QueueClass::Graphics,
        vec![ResourceAccess {
            resource: ResourceId(1),
            kind: AccessKind::StorageRead,
        }],
        vec![],
    ));
    assert!(matches!(
        graph.compile(),
        Err(CompileError::InvalidResource { .. })
    ));
    graph.clear();
    graph.add_resource(resource(256, 256));
    graph.add_pass(pass("self", QueueClass::Graphics, vec![], vec![PassId(0)]));
    assert!(matches!(
        graph.compile(),
        Err(CompileError::InvalidDependency { .. })
    ));
    let mut graph = GpuFrameGraphBuilder::with_limits(GraphLimits {
        versions: 0,
        ..GraphLimits::default()
    });
    graph.add_resource(resource(256, 256));
    graph.add_pass(pass(
        "write",
        QueueClass::Graphics,
        vec![ResourceAccess {
            resource: r,
            kind: AccessKind::StorageWrite,
        }],
        vec![],
    ));
    assert!(matches!(
        graph.compile(),
        Err(CompileError::VersionOverflow { .. })
    ));
}

/// Explicit dependencies, not table layout, choose implicit SSA versions.
#[test]
fn declaration_versions_follow_dependencies_before_deriving_hazards() {
    let mut graph = GpuFrameGraphBuilder::default();
    let r = graph.add_resource(resource(256, 256));
    for (name, kind, dependency) in [
        ("consume", AccessKind::StorageRead, Some(PassId(1))),
        ("overwrite", AccessKind::StorageWrite, Some(PassId(2))),
        ("produce", AccessKind::StorageWrite, None),
    ] {
        graph.add_pass(pass(
            name,
            QueueClass::Compute,
            vec![ResourceAccess { resource: r, kind }],
            dependency.into_iter().collect(),
        ));
    }
    let compiled = graph.compile().unwrap();
    assert_eq!(compiled.order, [2, 1, 0]);
    assert_eq!(compiled.declared_pass_count, 3);
    assert_eq!(compiled.alive, [true; 3]);
    assert_eq!(compiled.position_by_pass, [Some(2), Some(1), Some(0)]);
    assert_eq!(
        compiled
            .logical_barriers
            .iter()
            .map(|barrier| (barrier.source, barrier.destination))
            .collect::<Vec<_>>(),
        [(1, 0), (2, 1)]
    );
    assert!(compiled
        .logical_barriers
        .iter()
        .any(|b| b.source == 2 && b.destination == 1));
    assert!(compiled
        .logical_barriers
        .iter()
        .any(|b| b.source == 1 && b.destination == 0));
    assert_eq!(compiled.alias.heap_size, 256);
}

/// A dependency can deliberately read the imported version before a write;
/// unconstrained declarations still retain their original ordering semantics.
#[test]
fn declaration_forward_war_and_unconstrained_ties_are_stable() {
    for forward in [false, true] {
        let mut graph = GpuFrameGraphBuilder::default();
        let r = graph.add_resource(ResourceDescriptor {
            lifetime: ResourceLifetime::Imported,
            ..resource(256, 256)
        });
        graph.add_pass(pass(
            "write",
            QueueClass::Compute,
            vec![ResourceAccess {
                resource: r,
                kind: AccessKind::StorageWrite,
            }],
            if forward { vec![PassId(1)] } else { vec![] },
        ));
        graph.add_pass(pass(
            "read",
            QueueClass::Graphics,
            vec![ResourceAccess {
                resource: r,
                kind: AccessKind::StorageRead,
            }],
            vec![],
        ));
        let compiled = graph.compile().unwrap();
        let expected = if forward { [1, 0] } else { [0, 1] };
        assert_eq!(compiled.order, expected);
        assert_eq!(compiled.logical_barriers.len(), 1);
        let barrier = compiled.logical_barriers[0];
        assert_eq!(
            (barrier.source, barrier.destination, barrier.queue_transfer,),
            (expected[0], expected[1], true)
        );
        assert!(barrier.queue_transfer);
        assert_eq!(compiled.queue_batches[1].waits_for, [0]);
        assert_eq!(compiled.alias.heap_size, 0);
    }
}

/// Regression: lowering visits producer (1) before consumer (0), but those
/// temporary setup indices must never become the public declaration PassIds.
#[test]
fn declaration_indices_reverse_control_dependency() {
    let mut graph = GpuFrameGraphBuilder::default();
    let consumer = graph.add_pass(pass(
        "consumer",
        QueueClass::Transfer,
        vec![],
        vec![PassId(1)],
    ));
    let producer = graph.add_pass(pass("producer", QueueClass::Compute, vec![], vec![]));
    let plan = graph.compile().unwrap();

    // Before the fix these were [0, 1] and [Some(0), Some(1)], naming
    // the consumer as the producer even though no resource hazard exists.
    assert_eq!(plan.order, [producer.0 as usize, consumer.0 as usize]);
    assert_eq!(plan.declared_pass_count, 2);
    assert_eq!(plan.alive, [true, true]);
    assert_eq!(plan.position_by_pass, [Some(1), Some(0)]);
    assert_eq!(plan.position(consumer.0 as usize), Some(1));
    assert_eq!(plan.position(producer.0 as usize), Some(0));
    assert!(plan.is_alive(consumer.0 as usize));
    assert!(!plan.is_alive(2));
    assert_eq!(plan.position(2), None);
    assert!(plan.logical_barriers.is_empty());
    assert_eq!(plan.attachments, [None, None]);
    assert_eq!(plan.barriers.len(), 2);
    assert!(plan.barriers.iter().all(Vec::is_empty));
    assert_eq!(
        plan.queue_batches,
        [
            QueueBatch {
                queue: QueueClass::Compute,
                pass_range: 0..1,
                waits_for: vec![],
            },
            QueueBatch {
                queue: QueueClass::Transfer,
                pass_range: 1..2,
                waits_for: vec![0],
            },
        ]
    );
    assert_eq!(graph.compile().unwrap(), plan);
    assert_eq!(graph.passes()[consumer.0 as usize].name, "consumer");
    assert_eq!(graph.passes()[consumer.0 as usize].depends_on, [producer]);
}

/// Multiple forward edges exercise both directions of the temporary mapping;
/// independent ready passes still use the original declaration ID tie-break.
#[test]
fn declaration_indices_control_fanin_and_ready_ties() {
    let mut graph = GpuFrameGraphBuilder::default();
    for descriptor in [
        pass(
            "sink",
            QueueClass::Graphics,
            vec![],
            vec![PassId(1), PassId(2)],
        ),
        pass("branch", QueueClass::Transfer, vec![], vec![PassId(3)]),
        pass("independent", QueueClass::Graphics, vec![], vec![]),
        pass("source", QueueClass::Compute, vec![], vec![]),
    ] {
        graph.add_pass(descriptor);
    }
    let plan = graph.compile().unwrap();
    assert_eq!(plan.order, [2, 3, 1, 0]);
    assert_eq!(plan.position_by_pass, [Some(3), Some(2), Some(0), Some(1)]);
    assert_eq!(plan.alive, [true; 4]);
    assert!(plan.logical_barriers.is_empty());
    for (position, &index) in plan.order.iter().enumerate() {
        assert_eq!(plan.position(index), Some(position));
        for &dependency in &graph.passes()[index].depends_on {
            assert!(plan.position(dependency.0 as usize).unwrap() < position);
        }
    }
    assert_eq!(
        plan.queue_batches,
        [
            QueueBatch {
                queue: QueueClass::Graphics,
                pass_range: 0..1,
                waits_for: vec![],
            },
            QueueBatch {
                queue: QueueClass::Compute,
                pass_range: 1..2,
                waits_for: vec![],
            },
            QueueBatch {
                queue: QueueClass::Transfer,
                pass_range: 2..3,
                waits_for: vec![1],
            },
            QueueBatch {
                queue: QueueClass::Graphics,
                pass_range: 3..4,
                waits_for: vec![0, 2],
            },
        ]
    );
}

/// Resource IDs, normalized access ordinals and scheduled-position metadata
/// are separate namespaces from original PassIds and must not be remapped.
#[test]
fn declaration_indices_mixed_resources_preserve_hazards_and_schedule_metadata() {
    let mut graph = GpuFrameGraphBuilder::default();
    let buffer = graph.add_resource(resource(256, 256));
    let texture = graph.add_resource(resource(4, 256));
    graph.add_resource(resource(64, 256)); // Unused buffer, not a pass slot.
    for descriptor in [
        pass(
            "consumer",
            QueueClass::Graphics,
            vec![
                ResourceAccess {
                    resource: buffer,
                    kind: AccessKind::StorageRead,
                },
                ResourceAccess {
                    resource: texture,
                    kind: AccessKind::SampledRead,
                },
            ],
            vec![PassId(2)],
        ),
        pass(
            "overwrite",
            QueueClass::Compute,
            vec![
                ResourceAccess {
                    resource: buffer,
                    kind: AccessKind::StorageWrite,
                },
                ResourceAccess {
                    resource: texture,
                    kind: AccessKind::StorageWrite,
                },
            ],
            vec![PassId(0)],
        ),
        pass(
            "producer",
            QueueClass::Graphics,
            vec![
                ResourceAccess {
                    resource: buffer,
                    kind: AccessKind::StorageWrite,
                },
                ResourceAccess {
                    resource: texture,
                    kind: AccessKind::ColorAttachment,
                },
            ],
            vec![],
        ),
    ] {
        graph.add_pass(descriptor);
    }
    let plan = graph.compile().unwrap();
    assert_eq!(plan.order, [2, 0, 1]);
    assert_eq!(plan.position_by_pass, [Some(1), Some(2), Some(0)]);
    assert_eq!(plan.alive, [true; 3]);
    assert_eq!(plan.logical_barriers.len(), 6);
    for (source, destination, queue_transfer) in [(2, 0, false), (2, 1, true), (0, 1, true)] {
        for (resource, ordinal) in [
            (LogicalResource::Texture(ResourceIndex(0)), 0),
            (LogicalResource::Buffer(ResourceIndex(0)), 1),
        ] {
            assert!(plan.logical_barriers.contains(&LogicalBarrier {
                resource,
                source,
                destination,
                before: ordinal,
                after: ordinal,
                queue_transfer,
            }));
        }
    }
    assert_eq!(plan.used_textures, [true]);
    assert_eq!(plan.used_buffers, [true, false]);
    assert_eq!(plan.alias.texture_slots.len(), 1);
    assert!(plan.alias.texture_slots[0].is_some());
    assert_eq!(plan.alias.buffer_slots.len(), 2);
    assert!(plan.alias.buffer_slots[0].is_some());
    assert_eq!(plan.alias.buffer_slots[1], None);
    let producer_attachments = plan.attachments[plan.position(2).unwrap()]
        .as_ref()
        .unwrap();
    assert_eq!(producer_attachments.colors.len(), 1);
    assert_eq!(producer_attachments.colors[0].resource, ResourceIndex(0));
    assert_eq!(producer_attachments.colors[0].output_version, 1);
    assert_eq!(
        producer_attachments.colors[0].store,
        prism_render_driver::StoreOp::Store
    );
    assert_eq!(plan.attachments[plan.position(0).unwrap()], None);
    assert_eq!(plan.attachments[plan.position(1).unwrap()], None);

    // An identity-ordered declaration must produce identical positional and
    // resource-indexed metadata, without post-processing any plan fields.
    let mut forward = GpuFrameGraphBuilder::default();
    for descriptor in graph.resources() {
        forward.add_resource(descriptor.clone());
    }
    for (index, depends_on) in [(2, vec![]), (0, vec![PassId(0)]), (1, vec![PassId(1)])] {
        forward.add_pass(PassDescriptor {
            depends_on,
            ..graph.passes()[index].clone()
        });
    }
    let identity = forward.compile().unwrap();
    assert_eq!(identity.order, [0, 1, 2]);
    assert_eq!(identity.position_by_pass, [Some(0), Some(1), Some(2)]);
    assert_eq!(plan.barriers, identity.barriers);
    assert_eq!(plan.final_barriers, identity.final_barriers);
    assert_eq!(plan.attachments, identity.attachments);
    assert_eq!(plan.queue_batches, identity.queue_batches);
    assert_eq!(plan.alias, identity.alias);
    assert_eq!(plan.used_textures, identity.used_textures);
    assert_eq!(plan.used_buffers, identity.used_buffers);
}

/// The preliminary solve reports original IDs even when the first blocked
/// declaration is downstream of the cycle, not a cycle member itself.
#[test]
fn declaration_indices_cycle_diagnostics_name_original_passes() {
    let mut graph = GpuFrameGraphBuilder::default();
    for descriptor in [
        pass("downstream", QueueClass::Transfer, vec![], vec![PassId(2)]),
        pass("independent", QueueClass::Graphics, vec![], vec![]),
        pass("cycle-tail", QueueClass::Compute, vec![], vec![PassId(3)]),
        pass("cycle-head", QueueClass::Graphics, vec![], vec![PassId(2)]),
    ] {
        graph.add_pass(descriptor);
    }
    let CompileError::Cycle {
        unscheduled,
        blocked,
        cycle,
        names,
    } = graph.compile().unwrap_err()
    else {
        panic!("expected declaration dependency cycle");
    };
    assert_eq!(unscheduled, 3);
    assert_eq!(blocked, [0, 2, 3]);
    assert_eq!(cycle.len(), 3);
    assert_eq!(cycle.first(), cycle.last());
    assert!(!cycle.contains(&0));
    assert!(!cycle.contains(&1));
    assert_eq!(cycle.len(), names.len());
    for (&index, name) in cycle.iter().zip(&names) {
        assert_eq!(graph.passes()[index].name.as_ref(), name);
    }
    for edge in cycle.windows(2) {
        assert!(graph.passes()[edge[1]]
            .depends_on
            .contains(&PassId(edge[0] as u32)));
    }
}

/// Admission errors must name the original declaring pass, even when earlier
/// declarations have forward dependencies and would lower in a different order.
#[test]
fn declaration_indices_invalid_declarations_keep_original_ids() {
    for dependency in [PassId(1), PassId(9)] {
        let mut graph = GpuFrameGraphBuilder::default();
        graph.add_pass(pass(
            "consumer",
            QueueClass::Graphics,
            vec![],
            vec![PassId(2)],
        ));
        graph.add_pass(pass(
            "invalid",
            QueueClass::Compute,
            vec![],
            vec![dependency],
        ));
        graph.add_pass(pass("producer", QueueClass::Transfer, vec![], vec![]));
        assert_eq!(
            graph.compile(),
            Err(CompileError::InvalidDependency {
                pass: PassId(1),
                dependency
            })
        );
    }
    let mut graph = GpuFrameGraphBuilder::default();
    graph.add_pass(pass(
        "consumer",
        QueueClass::Graphics,
        vec![],
        vec![PassId(2)],
    ));
    graph.add_pass(pass(
        "invalid-resource",
        QueueClass::Compute,
        vec![ResourceAccess {
            resource: ResourceId(9),
            kind: AccessKind::StorageRead,
        }],
        vec![],
    ));
    graph.add_pass(pass("producer", QueueClass::Transfer, vec![], vec![]));
    assert_eq!(
        graph.compile(),
        Err(CompileError::InvalidResource {
            pass: PassId(1),
            resource: ResourceId(9)
        })
    );
}

#[test]
fn declaration_indices_empty_graph_is_unchanged() {
    let plan = GpuFrameGraphBuilder::default().compile().unwrap();
    assert_eq!(plan, crate::ExecutionPlan::default());
}

/// Control admission reuses the compiler's ordering and diagnostics without
/// lowering resources or deriving a second SSA solution.
#[test]
fn declaration_dependency_order_matches_empty_access_compilation() {
    for dependencies in [
        vec![],
        vec![vec![]],
        vec![vec![PassId(1)], vec![PassId(2)], vec![]],
        vec![vec![PassId(2), PassId(3)], vec![], vec![PassId(3)], vec![]],
        vec![vec![PassId(1)], vec![PassId(0)], vec![PassId(1)]],
        vec![vec![PassId(0)]],
        vec![vec![PassId(9)]],
    ] {
        let mut graph = GpuFrameGraphBuilder::default();
        for (index, dependencies) in dependencies.into_iter().enumerate() {
            let mut descriptor = pass("control", QueueClass::Compute, vec![], dependencies);
            descriptor.name = alloc::format!("control-{index}").into();
            graph.add_pass(descriptor);
        }
        assert_eq!(
            graph.dependency_order(),
            graph.compile().map(|plan| plan
                .order
                .into_iter()
                .map(|id| PassId(id as u32))
                .collect::<Vec<_>>())
        );
    }
    let mut graph = GpuFrameGraphBuilder::default();
    graph.add_pass(pass(
        "not-lowered",
        QueueClass::Compute,
        vec![access(ResourceId(99), AccessKind::StorageWrite)],
        vec![],
    ));
    assert_eq!(graph.dependency_order().unwrap(), [PassId(0)]);
    assert!(matches!(
        graph.compile(),
        Err(CompileError::InvalidResource { .. })
    ));
}

/// Source columns are declaration ordinals, even when texture normalization
/// reverses their order. References also admit forward-declared producers.
#[test]
fn declaration_sources_match_numeric_authority_metadata_with_reverse_ids() {
    let make = |numeric: bool| {
        let mut graph = GpuFrameGraphBuilder::default();
        let buffer = graph.add_resource(resource(256, 256));
        let texture = graph.add_resource(resource(4, 256));
        for descriptor in [
            pass(
                "reader",
                QueueClass::Graphics,
                vec![
                    access(buffer, AccessKind::StorageRead),
                    access(texture, AccessKind::SampledRead),
                ],
                if numeric {
                    vec![PassId(1), PassId(3)]
                } else {
                    vec![]
                },
            ),
            pass(
                "second-buffer-write",
                QueueClass::Compute,
                vec![access(buffer, AccessKind::StorageWrite)],
                if numeric { vec![PassId(3)] } else { vec![] },
            ),
            pass(
                "stale-overwrite",
                QueueClass::Graphics,
                vec![
                    access(buffer, AccessKind::StorageWrite),
                    access(texture, AccessKind::ColorAttachment),
                ],
                vec![PassId(3)],
            ),
            pass(
                "producer",
                QueueClass::Graphics,
                vec![
                    access(buffer, AccessKind::StorageWrite),
                    access(texture, AccessKind::ColorAttachment),
                ],
                vec![],
            ),
        ] {
            graph.add_pass(descriptor);
        }
        graph
    };
    let source = |pass, access| AccessInput::ProducedBy {
        pass: PassId(pass),
        access,
    };
    let graph = make(false);
    let inputs = [
        vec![source(1, 0), source(3, 1)],
        vec![source(3, 0)],
        vec![AccessInput::Version(0); 2],
        vec![AccessInput::Version(0); 2],
    ];
    let plan = graph.compile_with_sources(&inputs).unwrap();
    assert_eq!(plan.order, [3, 1, 0, 2]);
    assert_eq!(plan.position_by_pass, [Some(2), Some(1), Some(3), Some(0)]);
    assert_eq!(
        plan.accesses[0][0].resource,
        LogicalResource::Texture(ResourceIndex(0))
    );
    assert_eq!(
        plan.accesses[0][1].resource,
        LogicalResource::Buffer(ResourceIndex(0))
    );
    assert_eq!(
        plan.accesses[0][0].input_version,
        plan.accesses[3][0].output_version.unwrap()
    );
    assert_eq!(
        plan.accesses[0][1].input_version,
        plan.accesses[1][0].output_version.unwrap()
    );
    assert_eq!(plan.accesses[2][1].input_version, 0);
    assert_eq!(plan.accesses[2][1].output_version, Some(3));
    let numeric: Vec<Vec<u32>> = plan
        .accesses
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let columns = if index == 1 { vec![0] } else { vec![1, 0] };
            columns
                .into_iter()
                .map(|column| row[column].input_version)
                .collect()
        })
        .collect();
    assert_eq!(make(true).compile_with_inputs(&numeric).unwrap(), plan);
    assert_eq!(graph.compile_with_sources(&inputs).unwrap(), plan);
}

/// A producer may have several writes to one resource; the source identifies
/// the exact accessor result rather than the producer's final resource handle.
#[test]
fn declaration_sources_use_exact_write_access_and_keep_older_inputs() {
    let mut graph = GpuFrameGraphBuilder::default();
    let r = graph.add_resource(resource(256, 256));
    graph.add_pass(pass(
        "latest-reader",
        QueueClass::Compute,
        vec![access(r, AccessKind::StorageRead)],
        vec![],
    ));
    graph.add_pass(pass(
        "producer",
        QueueClass::Compute,
        vec![
            access(r, AccessKind::StorageWrite),
            access(r, AccessKind::StorageRead),
            access(r, AccessKind::StorageWrite),
        ],
        vec![],
    ));
    let plan = graph
        .compile_with_sources(&[
            vec![AccessInput::ProducedBy {
                pass: PassId(1),
                access: 2,
            }],
            vec![AccessInput::Version(0); 3],
        ])
        .unwrap();
    assert_eq!(plan.order, [1, 0]);
    assert_eq!(
        plan.accesses[0][0].input_version,
        plan.accesses[1][2].output_version.unwrap()
    );
    assert_eq!(plan.accesses[1][1].input_version, 0);
    assert_eq!(plan.accesses[1][0].output_version, Some(1));
    assert_eq!(plan.accesses[1][2].output_version, Some(2));
}

#[test]
fn declaration_sources_reject_invalid_producer_identities() {
    let mut graph = GpuFrameGraphBuilder::default();
    let r = graph.add_resource(resource(256, 256));
    let other = graph.add_resource(resource(256, 256));
    graph.add_pass(pass(
        "reader",
        QueueClass::Compute,
        vec![access(r, AccessKind::StorageRead)],
        vec![],
    ));
    graph.add_pass(pass(
        "source",
        QueueClass::Compute,
        vec![
            access(r, AccessKind::StorageRead),
            access(other, AccessKind::StorageWrite),
            access(r, AccessKind::StorageWrite),
        ],
        vec![],
    ));
    for (producer, producer_access, reason) in [
        (PassId(9), 0, "producer access does not exist"),
        (PassId(1), usize::MAX, "producer access does not exist"),
        (PassId(0), 0, "producer must be a different pass"),
        (PassId(1), 0, "producer access is not a write"),
        (PassId(1), 1, "producer uses a different resource"),
    ] {
        assert_eq!(
            graph.compile_with_sources(&[
                vec![AccessInput::ProducedBy {
                    pass: producer,
                    access: producer_access
                }],
                vec![AccessInput::Version(0); 3],
            ]),
            Err(CompileError::InvalidInputSource {
                pass: PassId(0),
                access: 0,
                producer,
                producer_access,
                reason,
            })
        );
    }
}

#[test]
fn declaration_sources_validate_shape_and_numeric_future_inputs() {
    let mut graph = GpuFrameGraphBuilder::default();
    let r = graph.add_resource(resource(256, 256));
    graph.add_pass(pass(
        "write",
        QueueClass::Compute,
        vec![access(r, AccessKind::StorageWrite)],
        vec![],
    ));
    assert_eq!(
        graph.compile_with_sources(&[]),
        Err(CompileError::InvalidInputShape {
            pass: None,
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        graph.compile_with_sources(&[vec![]]),
        Err(CompileError::InvalidInputShape {
            pass: Some(PassId(0)),
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        graph.compile_with_sources(&[vec![AccessInput::Version(1)]]),
        graph.compile_with_inputs(&[vec![1]])
    );
    assert!(matches!(
        graph.compile_with_sources(&[vec![AccessInput::Version(1)]]),
        Err(CompileError::InvalidHandle { version: 1, .. })
    ));
    assert_eq!(
        GpuFrameGraphBuilder::default()
            .compile_with_sources(&[])
            .unwrap(),
        crate::ExecutionPlan::default()
    );
}

#[test]
fn declaration_sources_and_control_edges_share_original_cycle_diagnostics() {
    let mut graph = GpuFrameGraphBuilder::default();
    let r = graph.add_resource(resource(256, 256));
    graph.add_pass(pass(
        "control-consumer",
        QueueClass::Compute,
        vec![access(r, AccessKind::StorageWrite)],
        vec![PassId(1)],
    ));
    graph.add_pass(pass(
        "source-consumer",
        QueueClass::Compute,
        vec![access(r, AccessKind::StorageWrite)],
        vec![],
    ));
    let Err(CompileError::Cycle {
        blocked,
        cycle,
        names,
        ..
    }) = graph.compile_with_sources(&[
        vec![AccessInput::Version(0)],
        vec![AccessInput::ProducedBy {
            pass: PassId(0),
            access: 0,
        }],
    ])
    else {
        panic!("expected a mixed control/source cycle")
    };
    assert_eq!(blocked, [0, 1]);
    assert_eq!(cycle.first(), cycle.last());
    assert_eq!(
        names,
        cycle
            .iter()
            .map(|&id| graph.passes()[id].name.clone().into_owned())
            .collect::<Vec<_>>()
    );
}

/// The input matrix chooses retained versions; physical writes remain globally
/// monotonic even when their input is stale. Scheduling must not rewrite SSA.
#[test]
fn declaration_inputs_retained_read_and_stale_write() {
    for overwrite_input in [0, 1] {
        let mut graph = GpuFrameGraphBuilder::default();
        let r = graph.add_resource(resource(256, 256));
        for (name, queue, kind) in [
            ("producer", QueueClass::Compute, AccessKind::StorageWrite),
            ("overwrite", QueueClass::Compute, AccessKind::StorageWrite),
            ("old-reader", QueueClass::Transfer, AccessKind::TransferRead),
            (
                "latest-reader",
                QueueClass::Graphics,
                AccessKind::StorageRead,
            ),
        ] {
            graph.add_pass(pass(name, queue, vec![access(r, kind)], vec![]));
        }
        let implicit = graph.compile().unwrap();
        assert_eq!(implicit.order, [0, 1, 2, 3]);
        let inputs = [vec![0], vec![overwrite_input], vec![1], vec![2]];
        let plan = graph.compile_with_inputs(&inputs).unwrap();
        assert_eq!(plan.order, [0, 2, 1, 3]);
        assert_eq!(plan.position_by_pass, [Some(0), Some(2), Some(1), Some(3)]);
        assert_eq!(plan.alive, [true; 4]);
        for (index, (input, output)) in [
            (0, Some(1)),
            (overwrite_input, Some(2)),
            (1, None),
            (2, None),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                plan.accesses[index],
                [CompiledAccess {
                    resource: LogicalResource::Buffer(ResourceIndex(0)),
                    input_version: input,
                    output_version: output,
                }]
            );
        }
        assert_eq!(
            plan.logical_barriers
                .iter()
                .map(|edge| (edge.source, edge.destination))
                .collect::<Vec<_>>(),
            [(0, 1), (0, 2), (1, 3), (2, 1)]
        );
        assert_eq!(plan.queue_batches[2].waits_for, [0, 1]);
        assert_eq!(graph.compile_with_inputs(&inputs).unwrap(), plan);
        let producer = |pass| AccessInput::ProducedBy {
            pass: PassId(pass),
            access: 0,
        };
        assert_eq!(
            graph
                .compile_with_sources(&[
                    vec![AccessInput::Version(0)],
                    vec![if overwrite_input == 0 {
                        AccessInput::Version(0)
                    } else {
                        producer(0)
                    }],
                    vec![producer(0)],
                    vec![producer(1)],
                ])
                .unwrap(),
            plan
        );
        assert_eq!(graph.compile().unwrap(), implicit);
    }
}

/// Both input rows and output rows retain original PassIds. Only output
/// columns normalize texture/buffer namespaces, never the caller's matrix.
#[test]
fn declaration_inputs_mixed_resources_and_reverse_ids() {
    let mut graph = GpuFrameGraphBuilder::default();
    let buffer = graph.add_resource(resource(256, 256));
    let texture = graph.add_resource(resource(4, 256));
    for descriptor in [
        pass(
            "reader",
            QueueClass::Graphics,
            vec![
                access(buffer, AccessKind::StorageRead),
                access(texture, AccessKind::SampledRead),
            ],
            vec![PassId(1)],
        ),
        pass(
            "second-buffer-write",
            QueueClass::Compute,
            vec![access(buffer, AccessKind::StorageWrite)],
            vec![PassId(3)],
        ),
        pass(
            "stale-overwrite",
            QueueClass::Graphics,
            vec![
                access(buffer, AccessKind::StorageWrite),
                access(texture, AccessKind::ColorAttachment),
            ],
            vec![PassId(3)],
        ),
        pass(
            "producer",
            QueueClass::Graphics,
            vec![
                access(buffer, AccessKind::StorageWrite),
                access(texture, AccessKind::ColorAttachment),
            ],
            vec![],
        ),
    ] {
        graph.add_pass(descriptor);
    }
    let plan = graph
        .compile_with_inputs(&[vec![2, 1], vec![1], vec![0, 0], vec![0, 0]])
        .unwrap();
    assert_eq!(plan.order, [3, 1, 0, 2]);
    assert_eq!(plan.position_by_pass, [Some(2), Some(1), Some(3), Some(0)]);
    let t = LogicalResource::Texture(ResourceIndex(0));
    let b = LogicalResource::Buffer(ResourceIndex(0));
    let compiled = |resource, input_version, output_version| CompiledAccess {
        resource,
        input_version,
        output_version,
    };
    assert_eq!(
        plan.accesses,
        [
            vec![compiled(t, 1, None), compiled(b, 2, None)],
            vec![compiled(b, 1, Some(2))],
            vec![compiled(t, 0, Some(2)), compiled(b, 0, Some(3))],
            vec![compiled(t, 0, Some(1)), compiled(b, 0, Some(1))],
        ]
    );
    for edge in &plan.logical_barriers {
        assert_eq!(
            plan.accesses[edge.source][edge.before].resource,
            edge.resource
        );
        assert_eq!(
            plan.accesses[edge.destination][edge.after].resource,
            edge.resource
        );
        assert!(plan.position(edge.source).unwrap() < plan.position(edge.destination).unwrap());
    }
    for (id, version) in [(3, 1), (2, 2)] {
        let attachment = &plan.attachments[plan.position(id).unwrap()]
            .as_ref()
            .unwrap()
            .colors[0];
        assert_eq!(attachment.output_version, version);
        assert_eq!(attachment.resource, ResourceIndex(0));
        assert!(matches!(
            attachment.load,
            prism_render_driver::LoadOp::Clear(_)
        ));
    }
    // Swapping declaration columns is not silently interpreted as normalized
    // texture-first inputs: the texture has no version 2 at this access.
    assert_eq!(
        graph.compile_with_inputs(&[vec![1, 2], vec![1], vec![0, 0], vec![0, 0]]),
        Err(CompileError::InvalidHandle {
            kind: "texture",
            index: 0,
            version: 2
        })
    );
}

#[test]
fn declaration_inputs_reject_shape_mismatches_with_original_pass_ids() {
    let mut graph = GpuFrameGraphBuilder::default();
    let r = graph.add_resource(resource(256, 256));
    graph.add_pass(pass("empty", QueueClass::Graphics, vec![], vec![PassId(1)]));
    graph.add_pass(pass(
        "reader",
        QueueClass::Compute,
        vec![access(r, AccessKind::StorageRead)],
        vec![],
    ));
    for rows in [0, 1, 3] {
        assert_eq!(
            graph.compile_with_inputs(&vec![vec![]; rows]),
            Err(CompileError::InvalidInputShape {
                pass: None,
                expected: 2,
                actual: rows
            })
        );
    }
    for (inputs, index, expected, actual) in [
        (vec![vec![0], vec![0]], 0, 0, 1),
        (vec![vec![], vec![]], 1, 1, 0),
        (vec![vec![], vec![0, 0]], 1, 1, 2),
    ] {
        assert_eq!(
            graph.compile_with_inputs(&inputs),
            Err(CompileError::InvalidInputShape {
                pass: Some(PassId(index)),
                expected,
                actual
            })
        );
    }
    let plan = graph.compile_with_inputs(&[vec![], vec![0]]).unwrap();
    assert_eq!(plan.order, [1, 0]);
    assert!(plan.accesses[0].is_empty());
    assert_eq!(plan.accesses[1][0].input_version, 0);
    assert_eq!(
        GpuFrameGraphBuilder::default()
            .compile_with_inputs(&[])
            .unwrap(),
        crate::ExecutionPlan::default()
    );
}

/// A later producer cannot retroactively validate a future input, for reads
/// or writes, in either resource namespace.
#[test]
fn declaration_inputs_reject_not_yet_existing_versions() {
    for (kind, namespace) in [
        (AccessKind::StorageRead, "buffer"),
        (AccessKind::SampledRead, "texture"),
    ] {
        for first_kind in [kind, AccessKind::StorageWrite] {
            let mut graph = GpuFrameGraphBuilder::default();
            let r = graph.add_resource(resource(256, 256));
            graph.add_pass(pass(
                "future",
                QueueClass::Compute,
                vec![access(r, first_kind)],
                vec![],
            ));
            graph.add_pass(pass(
                "later-producer",
                QueueClass::Graphics,
                vec![access(r, AccessKind::StorageWrite), access(r, kind)],
                vec![],
            ));
            assert_eq!(
                graph.compile_with_inputs(&[vec![1], vec![0, 1]]),
                Err(CompileError::InvalidHandle {
                    kind: namespace,
                    index: 0,
                    version: 1
                })
            );
        }
    }
}

/// Within a pass, declared access order admits versions without creating
/// self-hazards. Old reads do not roll back subsequent write allocation.
#[test]
fn declaration_inputs_same_pass_versions_follow_access_order() {
    for read_kind in [AccessKind::StorageRead, AccessKind::SampledRead] {
        let mut graph = GpuFrameGraphBuilder::default();
        let r = graph.add_resource(resource(256, 256));
        graph.add_pass(pass(
            "intra-pass",
            QueueClass::Graphics,
            vec![
                access(r, AccessKind::StorageWrite),
                access(r, read_kind),
                access(r, AccessKind::StorageWrite),
                access(r, read_kind),
            ],
            vec![],
        ));
        let plan = graph.compile_with_inputs(&[vec![0, 0, 0, 2]]).unwrap();
        assert_eq!(plan.order, [0]);
        assert!(plan.logical_barriers.is_empty());
        assert_eq!(
            plan.accesses[0]
                .iter()
                .map(|a| (a.input_version, a.output_version))
                .collect::<Vec<_>>(),
            [(0, Some(1)), (0, None), (0, Some(2)), (2, None)]
        );
        let namespace = if read_kind == AccessKind::SampledRead {
            "texture"
        } else {
            "buffer"
        };
        assert_eq!(
            graph.compile_with_inputs(&[vec![0, 2, 0, 2]]),
            Err(CompileError::InvalidHandle {
                kind: namespace,
                index: 0,
                version: 2
            })
        );
    }
}

/// The checked authority allocator reports capacity in both input modes;
/// stale inputs must neither reset version allocation nor trigger a panic.
#[test]
fn declaration_inputs_version_capacity_is_fallible() {
    for write_kind in [AccessKind::StorageWrite, AccessKind::ColorAttachment] {
        for limit in [0, 1] {
            let mut graph = GpuFrameGraphBuilder::with_limits(GraphLimits {
                versions: limit,
                ..Default::default()
            });
            let r = graph.add_resource(resource(256, 256));
            for _ in 0..2 {
                graph.add_pass(pass(
                    "write",
                    QueueClass::Graphics,
                    vec![access(r, write_kind)],
                    vec![],
                ));
            }
            let expected = Err(CompileError::VersionOverflow {
                version: limit,
                limit,
            });
            assert_eq!(graph.compile_with_inputs(&[vec![0], vec![0]]), expected);
            assert_eq!(
                graph.compile_with_sources(&[
                    vec![AccessInput::Version(0)],
                    vec![AccessInput::ProducedBy {
                        pass: PassId(0),
                        access: 0
                    }],
                ]),
                expected
            );
            assert_eq!(graph.compile(), expected);
        }
    }
}

/// Lowering succeeds, but a retained read of A conflicts with observing B's
/// later writer. The final solver must report original IDs and real names.
#[test]
fn declaration_inputs_cycle_diagnostics_keep_original_ids() {
    let mut graph = GpuFrameGraphBuilder::default();
    let a = graph.add_resource(resource(256, 256));
    let b = graph.add_resource(resource(256, 256));
    for descriptor in [
        pass(
            "producer",
            QueueClass::Compute,
            vec![access(a, AccessKind::StorageWrite)],
            vec![],
        ),
        pass(
            "overwrite",
            QueueClass::Compute,
            vec![
                access(a, AccessKind::StorageWrite),
                access(b, AccessKind::StorageWrite),
            ],
            vec![],
        ),
        pass(
            "old-reader",
            QueueClass::Compute,
            vec![
                access(a, AccessKind::StorageRead),
                access(b, AccessKind::StorageRead),
            ],
            vec![],
        ),
        pass(
            "downstream",
            QueueClass::Compute,
            vec![access(a, AccessKind::StorageRead)],
            vec![],
        ),
    ] {
        graph.add_pass(descriptor);
    }
    let numeric_error = graph
        .compile_with_inputs(&[vec![0], vec![1, 0], vec![1, 1], vec![2]])
        .unwrap_err();
    let producer = |pass, access| AccessInput::ProducedBy {
        pass: PassId(pass),
        access,
    };
    assert_eq!(
        graph
            .compile_with_sources(&[
                vec![AccessInput::Version(0)],
                vec![producer(0, 0), AccessInput::Version(0)],
                vec![producer(0, 0), producer(1, 1)],
                vec![producer(1, 0)],
            ])
            .unwrap_err(),
        numeric_error
    );
    let CompileError::Cycle {
        blocked,
        cycle,
        names,
        ..
    } = numeric_error
    else {
        panic!("expected a retained-version hazard cycle");
    };
    assert_eq!(blocked, [1, 2, 3]);
    assert_eq!(cycle.len(), 3);
    assert_eq!(cycle.first(), cycle.last());
    assert!(!cycle.contains(&3));
    assert!(cycle.contains(&1) && cycle.contains(&2));
    for (&index, name) in cycle.iter().zip(&names) {
        assert_eq!(graph.passes()[index].name.as_ref(), name);
    }
}

#[test]
fn executable_control_dependencies_preserve_edges_and_batch_waits() {
    let mut graph = RenderGraph::new(Extent3d::new_2d(1, 1));
    let producer = graph
        .try_add_pass_with_dependencies(
            "producer",
            PassKind::Compute,
            PassFlags::NEVER_CULL | PassFlags::ASYNC_COMPUTE,
            vec![],
            |_| |_: &mut crate::ExecuteContext<'_>| {},
        )
        .unwrap();
    let consumer = graph
        .try_add_pass_with_dependencies(
            "consumer",
            PassKind::Transfer,
            PassFlags::NEVER_CULL,
            vec![producer],
            |_| |_: &mut crate::ExecuteContext<'_>| {},
        )
        .unwrap();
    assert_eq!((producer, consumer), (0, 1));
    let plan = graph.compile().unwrap();
    assert_eq!(plan.order, [0, 1]);
    assert_eq!(plan.alive, [true, true]);
    assert_eq!(plan.accesses, [vec![], vec![]]);
    assert!(plan.logical_barriers.is_empty());
    assert_eq!(plan.queue_batches[1].waits_for, [0]);
}

#[test]
fn executable_control_dependencies_reject_before_setup() {
    let mut graph = RenderGraph::with_limits(
        Extent3d::new_2d(1, 1),
        GraphLimits {
            passes: 2,
            ..Default::default()
        },
    );
    graph
        .try_add_pass_with_dependencies(
            "root",
            PassKind::Compute,
            PassFlags::NEVER_CULL,
            vec![],
            |_| |_: &mut crate::ExecuteContext<'_>| {},
        )
        .unwrap();
    let called = core::cell::Cell::new(false);
    for dependency in [1, 2, u32::MAX as usize] {
        assert_eq!(
            graph.try_add_pass_with_dependencies(
                "invalid",
                PassKind::Compute,
                PassFlags::NEVER_CULL,
                vec![dependency],
                |_| {
                    called.set(true);
                    |_: &mut crate::ExecuteContext<'_>| {}
                },
            ),
            Err(CompileError::InvalidDependency {
                pass: PassId(1),
                dependency: PassId(dependency as u32)
            })
        );
        assert!(!called.get());
        assert_eq!(graph.pass_count(), 1);
    }
    assert_eq!(
        graph.try_add_pass_with_dependencies(
            "too-many-dependencies",
            PassKind::Compute,
            PassFlags::NEVER_CULL,
            vec![0, 0, 0],
            |_| {
                called.set(true);
                |_: &mut crate::ExecuteContext<'_>| {}
            },
        ),
        Err(CompileError::Capacity {
            kind: "dependencies",
            count: 3,
            limit: 2
        })
    );
    assert!(!called.get());
    assert_eq!(graph.pass_count(), 1);
}

/// Compiled access rows describe admission, even when culling removes work.
#[test]
fn compiled_access_metadata_keeps_culled_rows() {
    let mut graph = RenderGraph::new(Extent3d::new_2d(1, 1));
    for (name, flags) in [
        ("dead", PassFlags::default()),
        ("live", PassFlags::SIDE_EFFECT),
    ] {
        graph.add_compute_pass(name, flags, |b| {
            let zero = b.create_buffer(name, BufferDesc::new(256, BufferUsages::NONE));
            b.write_storage_buffer(zero);
            |_: &mut crate::ExecuteContext<'_>| {}
        });
    }
    let plan = graph.compile().unwrap();
    assert_eq!(plan.order, [1]);
    assert_eq!(plan.alive, [false, true]);
    assert_eq!(plan.position_by_pass, [None, Some(0)]);
    assert_eq!(plan.accesses.len(), 2);
    for (index, row) in plan.accesses.iter().enumerate() {
        assert_eq!(
            row,
            &[CompiledAccess {
                resource: LogicalResource::Buffer(ResourceIndex(index as u32)),
                input_version: 0,
                output_version: Some(1)
            }]
        );
    }
}

/// Real frontends with matching footprints must project the same kernel result.
#[test]
fn real_frontends_project_identical_kernel_products() {
    let mut executable = RenderGraph::new(Extent3d::new_2d(1, 1));
    let mut handle = None;
    executable.add_compute_pass(
        "write",
        PassFlags::SIDE_EFFECT | PassFlags::ASYNC_COMPUTE,
        |b| {
            let zero = b.create_buffer("resource", BufferDesc::new(256, BufferUsages::NONE));
            handle = Some(b.write_storage_buffer(zero));
            |_: &mut crate::ExecuteContext<'_>| {}
        },
    );
    executable.add_transfer_pass("read", PassFlags::SIDE_EFFECT, |b| {
        b.copy_src_buffer(handle.unwrap());
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    let exec = executable.compile().unwrap();
    let mut declaration = GpuFrameGraphBuilder::default();
    let r = declaration.add_resource(resource(256, 256));
    declaration.add_pass(pass(
        "write",
        QueueClass::Compute,
        vec![ResourceAccess {
            resource: r,
            kind: AccessKind::StorageWrite,
        }],
        vec![],
    ));
    declaration.add_pass(pass(
        "read",
        QueueClass::Transfer,
        vec![ResourceAccess {
            resource: r,
            kind: AccessKind::TransferRead,
        }],
        vec![],
    ));
    let decl = declaration.compile().unwrap();
    assert_eq!(exec.order, decl.order.iter().copied().collect::<Vec<_>>());
    assert_eq!(exec.accesses, decl.accesses);
    assert_eq!(exec.queue_batches, decl.queue_batches);
    assert_eq!(exec.alias.heap_size, decl.alias.heap_size);
    assert_eq!(
        exec.alias.buffer_slots[0].unwrap().offset,
        decl.alias.buffer_slots[0].unwrap().offset
    );
    assert_eq!(exec.logical_barriers.len(), decl.logical_barriers.len());
    let edge = exec.logical_barriers[0];
    let declaration_edge = decl.logical_barriers[0];
    assert_eq!(
        (edge.source, edge.destination, edge.queue_transfer),
        (
            declaration_edge.source,
            declaration_edge.destination,
            declaration_edge.queue_transfer
        )
    );
    assert_eq!(decl.alias.buffer_slots.len(), 1);
}

/// A reader declared late still observes its SSA version before replacement.
#[test]
fn late_old_version_reader_precedes_overwrite() {
    let mut graph = RenderGraph::new(Extent3d::new_2d(1, 1));
    let mut handle = None;
    graph.add_compute_pass("producer", PassFlags::SIDE_EFFECT, |b| {
        let zero = b.create_buffer("resource", BufferDesc::new(256, BufferUsages::NONE));
        handle = Some(b.write_storage_buffer(zero));
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    graph.add_compute_pass("overwrite", PassFlags::SIDE_EFFECT, |b| {
        b.write_storage_buffer(handle.unwrap());
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    graph.add_transfer_pass("old-reader", PassFlags::SIDE_EFFECT, |b| {
        b.copy_src_buffer(handle.unwrap());
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    let plan = graph.compile().unwrap();
    assert_eq!(plan.order, vec![0, 2, 1]);
    assert!(plan
        .logical_barriers
        .iter()
        .any(|h| h.source == 2 && h.destination == 1));
    assert_eq!(
        plan.queue_batches
            .iter()
            .map(|b| b.pass_range.clone())
            .collect::<Vec<_>>(),
        vec![0..1, 1..2, 2..3]
    );
    assert_eq!(plan.queue_batches[2].waits_for, vec![0, 1]);
}

/// Stale write inputs still serialize the physically replaced version and readers.
#[test]
fn stale_write_preserves_physical_waw_and_war_dependencies() {
    let mut graph = RenderGraph::new(Extent3d::new_2d(1, 1));
    let mut zero = None;
    let mut one = None;
    graph.add_compute_pass("first-write", PassFlags::SIDE_EFFECT, |b| {
        let handle = b.create_buffer("resource", BufferDesc::new(256, BufferUsages::NONE));
        zero = Some(handle);
        one = Some(b.write_storage_buffer(handle));
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    graph.add_compute_pass("stale-write", PassFlags::SIDE_EFFECT, |b| {
        let two = b.write_storage_buffer(zero.unwrap());
        assert_eq!(two.version(), 2);
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    graph.add_transfer_pass("old-reader", PassFlags::SIDE_EFFECT, |b| {
        b.copy_src_buffer(one.unwrap());
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    let plan = graph.compile().unwrap();
    assert_eq!(plan.order, vec![0, 2, 1]);
    for source in [0, 2] {
        assert!(plan
            .logical_barriers
            .iter()
            .any(|edge| { edge.source == source && edge.destination == 1 }));
    }
}

/// Real versioned accesses can cycle; diagnostics identify only the closed path.
#[test]
fn executable_cycle_reports_pass_names_and_downstream_blockage() {
    let mut graph = RenderGraph::new(Extent3d::new_2d(1, 1));
    let mut one = None;
    let mut two = None;
    let mut other = None;
    graph.add_compute_pass("producer", PassFlags::SIDE_EFFECT, |b| {
        let zero = b.create_buffer("a", BufferDesc::new(256, BufferUsages::NONE));
        one = Some(b.write_storage_buffer(zero));
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    graph.add_compute_pass("overwrite", PassFlags::SIDE_EFFECT, |b| {
        two = Some(b.write_storage_buffer(one.unwrap()));
        let zero = b.create_buffer("b", BufferDesc::new(256, BufferUsages::NONE));
        other = Some(b.write_storage_buffer(zero));
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    graph.add_compute_pass("old-reader", PassFlags::SIDE_EFFECT, |b| {
        b.read_storage_buffer(one.unwrap());
        b.read_storage_buffer(other.unwrap());
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    graph.add_compute_pass("downstream", PassFlags::SIDE_EFFECT, |b| {
        b.read_storage_buffer(two.unwrap());
        |_: &mut crate::ExecuteContext<'_>| {}
    });
    let CompileError::Cycle {
        blocked,
        cycle,
        names,
        ..
    } = graph.compile().unwrap_err()
    else {
        panic!("expected a versioned dependency cycle");
    };
    assert_eq!(blocked, vec![1, 2, 3]);
    assert_eq!(cycle.first(), cycle.last());
    assert_eq!(cycle.len(), 3);
    assert!(!cycle.contains(&3));
    assert!(names.iter().any(|name| name == "overwrite"));
    assert!(names.iter().any(|name| name == "old-reader"));
    assert!(!names.iter().any(|name| name == "downstream"));
}

/// The closed cycle follows real edges and does not mislabel blocked consumers.
#[test]
fn cycle_reports_actual_closed_path_not_downstream_victims() {
    let explicit = vec![vec![1], vec![0], vec![1]];
    let error = kernel::solve(
        &vec![vec![]; 3],
        &[true; 3],
        &explicit,
        &["a".into(), "b".into(), "downstream".into()],
    )
    .err()
    .unwrap();
    let CompileError::Cycle {
        unscheduled,
        blocked,
        cycle,
        names,
    } = error
    else {
        panic!("expected cycle")
    };
    assert_eq!(unscheduled, 3);
    assert_eq!(blocked, vec![0, 1, 2]);
    assert_eq!(cycle.first(), cycle.last());
    assert!(!cycle.contains(&2));
    assert_eq!(names.len(), cycle.len());
    for edge in cycle.windows(2) {
        assert!(explicit[edge[1]].contains(&edge[0]));
    }
}

/// Intra-pass accesses cannot self-cycle; independent consumers remain stable.
#[test]
fn repeated_same_pass_access_has_no_self_hazard_and_fanout_is_stable() {
    let accesses = vec![
        vec![
            Access {
                resource: 0,
                input: 0,
                output: Some(1),
            },
            Access {
                resource: 0,
                input: 1,
                output: None,
            },
        ],
        vec![Access {
            resource: 0,
            input: 1,
            output: None,
        }],
        vec![Access {
            resource: 0,
            input: 1,
            output: None,
        }],
    ];
    let solved = kernel::solve(
        &accesses,
        &[true; 3],
        &vec![vec![]; 3],
        &["producer".into(), "a".into(), "b".into()],
    )
    .unwrap();
    assert_eq!(solved.order, vec![0, 1, 2]);
    assert_eq!(solved.dependencies[0], BTreeSet::new());
    assert_eq!(solved.hazards.len(), 2);
    assert_eq!(
        kernel::lifetimes(&accesses, &solved.order, 1),
        vec![Some((0, 2))]
    );
}

/// Alignment holes count toward heap extent; overflowing metadata is fallible.
#[test]
fn allocator_uses_address_extent_arbitrary_alignment_and_checked_arithmetic() {
    let request = |size, alignment, first, last| {
        Some(AllocationRequest {
            size,
            alignment,
            first,
            last,
        })
    };
    let (slots, heap) =
        kernel::allocate(&[request(300, 512, 0, 1), request(300, 512, 0, 1), None]).unwrap();
    assert_eq!(heap, 812);
    assert_eq!(slots[1].unwrap().offset, 512);
    assert!(slots[2].is_none());
    let (slots, heap) = kernel::allocate(&[
        request(5, 3, 0, 0),
        request(4, 3, 1, 1),
        request(5, 3, 1, 1),
    ])
    .unwrap();
    assert_eq!(slots[1].unwrap().offset, 0);
    assert_eq!(slots[2].unwrap().offset, 6);
    assert_eq!(heap, 11);
    for slot in slots.into_iter().flatten() {
        assert!(slot.offset + slot.size <= heap);
    }
    assert_eq!(
        kernel::align_up(u64::MAX, 2),
        Err(CompileError::MemoryOverflow)
    );
    assert!(matches!(
        kernel::allocate(&[request(u64::MAX, 1, 0, 1), request(1, 1, 0, 1)]),
        Err(CompileError::MemoryOverflow)
    ));
}
