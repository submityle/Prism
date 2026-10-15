//! Shared-kernel and frontend admission regressions.
use crate::capacity::GraphLimits;
use crate::compile::kernel::{self, Access, AllocationRequest};
use crate::frame_graph::{
    AccessKind, GpuFrameGraphBuilder, PassDescriptor, PassId, QueueClass, ResourceAccess,
    ResourceDescriptor, ResourceId, ResourceLifetime,
};
use crate::{
    Blackboard, BufferDesc, BufferHandle, BufferUse, CompileError, PassBuilder, PassFlags,
    PassKind, RenderGraph, ResourceIndex,
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
    assert_eq!(compiled.order, [0, 1, 2]);
    assert_eq!(compiled.declared_pass_count, 3);
    assert_eq!(
        compiled
            .logical_barriers
            .iter()
            .map(|barrier| (barrier.source, barrier.destination))
            .collect::<Vec<_>>(),
        [(0, 1), (1, 2)]
    );
    assert!(compiled
        .logical_barriers
        .iter()
        .any(|b| b.source == 0 && b.destination == 1));
    assert!(compiled
        .logical_barriers
        .iter()
        .any(|b| b.source == 1 && b.destination == 2));
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
        assert_eq!(compiled.order, [0, 1]);
        assert_eq!(compiled.logical_barriers.len(), 1);
        let barrier = compiled.logical_barriers[0];
        assert_eq!(
            (
                barrier.source,
                barrier.destination,
                barrier.queue_transfer,
            ),
            (0, 1, true)
        );
        assert!(barrier.queue_transfer);
        assert_eq!(compiled.queue_batches[1].waits_for, [0]);
        assert_eq!(compiled.alias.heap_size, 0);
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
    assert_eq!(
        exec.order,
        decl.order
            .iter()
            .copied()
            .collect::<Vec<_>>()
    );
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
