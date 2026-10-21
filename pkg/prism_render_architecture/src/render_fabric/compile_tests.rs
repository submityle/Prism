use super::*;
use crate::{frame_graph::*, gpu_scene::SceneHandle, motion::MotionSource};
use prism_render_graph::{ExecutionPlan, LogicalResource};

const ROOT: FeatureId = FeatureId(700);
const A: ResourceKey = ResourceKey::of(ROOT, 0);
const B: ResourceKey = ResourceKey::of(ROOT, 1);
const UNUSED: ResourceKey = ResourceKey::of(ROOT, 2);

fn table<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}
fn usage(key: ResourceKey, access: AccessKind) -> ResourceUse {
    ResourceUse {
        input: ResourceInput::Automatic,
        key,
        access,
        scope: ResourceScope::Whole,
        frame_lag: None,
    }
}
fn pass(key: u32, queue: QueueClass, accesses: Vec<ResourceUse>) -> PassTemplate {
    PassTemplate {
        depends_on: &[],
        id: PassKey(key),
        queue,
        accesses: table(accesses),
    }
}
fn variant(id: u32, rank: u16, passes: Vec<PassTemplate>) -> FeatureVariant {
    FeatureVariant {
        id: VariantId(id),
        rank,
        min_quality: RenderQualityTier::Low,
        required_caps: CapabilitySet::EMPTY,
        requires_services_extra: ServiceSet::EMPTY,
        passes: table(passes),
        shader_permutation: None,
    }
}
fn catalog(variants: Vec<FeatureVariant>, keys: Vec<ResourceKey>) -> FeatureCatalog {
    FeatureCatalog::new(
        table(vec![FeatureDefinition::subsystem(
            ROOT,
            RenderDomain::Surface,
            ServiceSet::EMPTY,
            table(variants),
        )]),
        table(keys),
    )
    .unwrap()
}
fn select(catalog: &FeatureCatalog) -> FeatureSelection {
    let roots = FrozenRoots::from_build_profile(catalog, &[ROOT], &[]).unwrap();
    FeatureSelection::resolve(
        catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .unwrap()
}
fn buffer(key: ResourceKey, lifetime: ResourceLifetime) -> ResourceTemplate {
    ResourceTemplate {
        key,
        lifetime,
        alignment: 16,
        layout: ResourceLayout::Buffer {
            bytes: 64,
            usage: BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST,
        },
    }
}
fn binding(template: ResourceTemplate, frame: u64, id: u64) -> FrameResourceBinding {
    FrameResourceBinding {
        key: template.key,
        frame,
        version: 5,
        physical: PhysicalResource::Buffer {
            id: BufferId::from_parts(u32::try_from(id).unwrap(), 1),
            offset: 0,
        },
        layout: template.layout,
        alignment: template.alignment,
    }
}
fn fixture() -> FeatureCatalog {
    // Deliberately reverse producer order; resource edges, not this order,
    // establish execution and SSA input versions.
    catalog(
        vec![variant(
            700,
            0,
            vec![
                pass(
                    3,
                    QueueClass::Graphics,
                    vec![usage(B, AccessKind::IndirectRead)],
                ),
                pass(
                    2,
                    QueueClass::Compute,
                    vec![
                        usage(A, AccessKind::StorageRead),
                        usage(B, AccessKind::StorageWrite),
                    ],
                ),
                pass(
                    1,
                    QueueClass::Transfer,
                    vec![usage(A, AccessKind::TransferWrite)],
                ),
            ],
        )],
        vec![A, B],
    )
}
fn compile_fixture(limits: GraphLimits) -> Result<CompiledFeatureGraph, FabricCompileError> {
    let catalog = fixture();
    ResourceTemplates::new(
        &catalog,
        &[
            buffer(A, ResourceLifetime::Transient),
            buffer(B, ResourceLifetime::Transient),
        ],
    )?
    .compile(&select(&catalog), 10, &[], &[], limits)
}

#[test]
fn active_compile_uses_frozen_selection_and_explicit_completed_dependencies() {
    let producer = pass(
        1,
        QueueClass::Compute,
        vec![usage(B, AccessKind::StorageWrite)],
    );
    let mut consumer = pass(
        2,
        QueueClass::Graphics,
        vec![usage(A, AccessKind::StorageRead)],
    );
    consumer.depends_on = table(vec![PassKey(1)]);
    let catalog = catalog(vec![variant(700, 0, vec![producer, consumer])], vec![A, B]);
    let selection = select(&catalog);
    let template = buffer(A, ResourceLifetime::Persistent);
    let imported = binding(template, 10, 55);
    let resources = ResourceTemplates::new(
        &catalog,
        &[template, buffer(B, ResourceLifetime::Transient)],
    )
    .unwrap();

    assert_eq!(
        resources
            .compile_active(
                &selection,
                10,
                &[imported],
                &[],
                &[PassKey(2)],
                &[],
                GraphLimits::default(),
            )
            .unwrap_err(),
        FabricCompileError::MissingActiveDependency {
            pass: PassKey(2),
            dependency: PassKey(1),
        }
    );

    let compiled = resources
        .compile_active(
            &selection,
            10,
            &[imported],
            &[],
            &[PassKey(2)],
            &[PassKey(1)],
            GraphLimits::default(),
        )
        .unwrap();
    assert_eq!(compiled.passes().len(), 1);
    assert_eq!(compiled.passes()[0].key(), PassKey(2));
    assert_eq!(compiled.resources().len(), 1);
    assert_eq!(compiled.resources()[0].external, Some(imported));

    assert_eq!(
        resources
            .compile_active(
                &selection,
                10,
                &[imported],
                &[],
                &[PassKey(3)],
                &[],
                GraphLimits::default(),
            )
            .unwrap_err(),
        FabricCompileError::ForeignActivePass(PassKey(3))
    );
    assert_eq!(
        resources
            .compile_active(
                &selection,
                10,
                &[imported],
                &[],
                &[PassKey(2), PassKey(2)],
                &[PassKey(1)],
                GraphLimits::default(),
            )
            .unwrap_err(),
        FabricCompileError::DuplicateActivePass(PassKey(2))
    );
    assert_eq!(
        resources
            .compile_active(
                &selection,
                10,
                &[imported],
                &[],
                &[PassKey(2)],
                &[PassKey(1), PassKey(1)],
                GraphLimits::default(),
            )
            .unwrap_err(),
        FabricCompileError::DuplicateFulfilledDependency(PassKey(1))
    );
    assert_eq!(
        resources
            .compile_active(
                &selection,
                10,
                &[imported],
                &[],
                &[PassKey(2)],
                &[PassKey(3)],
                GraphLimits::default(),
            )
            .unwrap_err(),
        FabricCompileError::ForeignFulfilledDependency(PassKey(3))
    );
    assert_eq!(
        resources
            .compile_active(
                &selection,
                10,
                &[imported],
                &[],
                &[PassKey(2)],
                &[PassKey(2)],
                GraphLimits::default(),
            )
            .unwrap_err(),
        FabricCompileError::ForeignFulfilledDependency(PassKey(2))
    );
}

#[test]
fn derived_order_and_every_barrier_match_authoritative_resource_graph() {
    let compiled = compile_fixture(GraphLimits::default()).unwrap();
    let mut baseline = GpuFrameGraphBuilder::default();
    let a = baseline
        .try_add_resource(ResourceDescriptor {
            name: "a".into(),
            lifetime: ResourceLifetime::Transient,
            size: 64,
            alignment: 16,
        })
        .unwrap();
    let b = baseline
        .try_add_resource(ResourceDescriptor {
            name: "b".into(),
            lifetime: ResourceLifetime::Transient,
            size: 64,
            alignment: 16,
        })
        .unwrap();
    for (name, queue, accesses) in [
        (
            "read-b",
            QueueClass::Graphics,
            vec![ResourceAccess {
                resource: b,
                kind: AccessKind::IndirectRead,
            }],
        ),
        (
            "read-a/write-b",
            QueueClass::Compute,
            vec![
                ResourceAccess {
                    resource: a,
                    kind: AccessKind::StorageRead,
                },
                ResourceAccess {
                    resource: b,
                    kind: AccessKind::StorageWrite,
                },
            ],
        ),
        (
            "write-a",
            QueueClass::Transfer,
            vec![ResourceAccess {
                resource: a,
                kind: AccessKind::TransferWrite,
            }],
        ),
    ]
    .into_iter()
    .rev()
    {
        baseline
            .try_add_pass(PassDescriptor {
                name: name.into(),
                queue,
                accesses,
                depends_on: vec![],
            })
            .unwrap();
    }
    let expected = baseline.compile().unwrap();
    assert_eq!(
        compiled
            .passes()
            .iter()
            .map(PassSpec::key)
            .collect::<Vec<_>>(),
        vec![PassKey(1), PassKey(2), PassKey(3)]
    );
    assert_eq!(compiled.plan().order, expected.order);
    assert_eq!(
        compiled.barriers(),
        vec![
            Barrier {
                resource: a,
                source: PassId(0),
                destination: PassId(1),
                before: AccessKind::TransferWrite,
                after: AccessKind::StorageRead,
                queue_transfer: true
            },
            Barrier {
                resource: b,
                source: PassId(1),
                destination: PassId(2),
                before: AccessKind::StorageWrite,
                after: AccessKind::IndirectRead,
                queue_transfer: true
            },
        ]
    );
    assert_eq!(compiled.plan().queue_batches, expected.queue_batches);
    assert_eq!(
        compiled.plan().alias.buffer_slots,
        expected.alias.buffer_slots
    );
    assert_eq!(compiled.plan().alias.heap_size, expected.alias.heap_size);
    assert_eq!(compiled.resource_id(A, 10), Some(a));
    assert_eq!(compiled.resource_id(A, 9), None);
    assert_eq!(
        compiled.versions()[0],
        FrameResourceVersion {
            key: A,
            frame: 10,
            identity: None,
            resource: a,
            writer: PassKey(1),
            ssa_version: 1,
            content_version: 1
        }
    );
}

#[test]
fn checked_capacities_reach_authority_without_panics() {
    for (limits, expected) in [
        (
            GraphLimits {
                resources: 1,
                ..GraphLimits::default()
            },
            CompileError::Capacity {
                kind: "resources",
                count: 2,
                limit: 1,
            },
        ),
        (
            GraphLimits {
                passes: 2,
                ..GraphLimits::default()
            },
            CompileError::Capacity {
                kind: "passes",
                count: 3,
                limit: 2,
            },
        ),
        (
            GraphLimits {
                accesses_per_pass: 1,
                ..GraphLimits::default()
            },
            CompileError::Capacity {
                kind: "accesses",
                count: 2,
                limit: 1,
            },
        ),
        (
            GraphLimits {
                versions: 0,
                ..GraphLimits::default()
            },
            CompileError::VersionOverflow {
                version: 0,
                limit: 0,
            },
        ),
    ] {
        assert_eq!(
            compile_fixture(limits).unwrap_err(),
            FabricCompileError::Graph(expected)
        );
    }
}

fn history_fixture() -> FeatureCatalog {
    let mut lagged = usage(B, AccessKind::StorageRead);
    lagged.frame_lag = FrameLag::new(1);
    catalog(
        vec![variant(
            710,
            0,
            vec![
                pass(
                    1,
                    QueueClass::Compute,
                    vec![lagged, usage(A, AccessKind::StorageWrite)],
                ),
                pass(
                    2,
                    QueueClass::Compute,
                    vec![
                        usage(A, AccessKind::StorageRead),
                        usage(B, AccessKind::StorageWrite),
                    ],
                ),
            ],
        )],
        vec![A, B],
    )
}

#[test]
fn spray_foam_history_is_a_real_imported_frame_binding_not_a_same_frame_edge() {
    let catalog = history_fixture();
    let a = buffer(A, ResourceLifetime::Transient);
    let b = buffer(B, ResourceLifetime::Persistent);
    let resources = ResourceTemplates::new(&catalog, &[a, b]).unwrap();
    let compiled = resources
        .compile(
            &select(&catalog),
            10,
            &[binding(b, 9, 9), binding(b, 10, 10)],
            &[],
            GraphLimits::default(),
        )
        .unwrap();
    let history = compiled.resource_id(B, 9).unwrap();
    let current = compiled.resource_id(B, 10).unwrap();
    assert_ne!(history, current);
    assert_eq!(
        compiled.resources()[history.0 as usize].descriptor.lifetime,
        ResourceLifetime::Imported
    );
    assert_eq!(
        compiled.resources()[history.0 as usize]
            .external
            .unwrap()
            .version,
        5
    );
    assert_eq!(
        compiled.passes()[0].descriptor().accesses[0].resource,
        history
    );
    assert!(compiled.plan().logical_barriers.iter().all(|barrier| {
        !matches!(
            barrier.resource,
            LogicalResource::Buffer(index)
                if index.get() == history.0 || index.get() == current.0
        )
    }));
    assert_eq!(
        compiled
            .versions()
            .iter()
            .find(|version| version.key == B)
            .unwrap()
            .content_version,
        6
    );
    assert!(compiled.plan().alias.buffer_slots[history.0 as usize].is_none());
    assert!(compiled.plan().alias.buffer_slots[current.0 as usize].is_none());
}

#[test]
fn history_missing_underflow_and_overlapping_storage_fail() {
    let catalog = history_fixture();
    let b = buffer(B, ResourceLifetime::Persistent);
    let resources =
        ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient), b]).unwrap();
    let selection = select(&catalog);
    assert_eq!(
        resources
            .compile(
                &selection,
                10,
                &[binding(b, 10, 10)],
                &[],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::MissingBinding {
            resource: B,
            frame: 9
        }
    );
    assert_eq!(
        resources
            .compile(&selection, 0, &[], &[], GraphLimits::default())
            .unwrap_err(),
        FabricCompileError::HistoryBeforeFrameZero {
            resource: B,
            lag: 1
        }
    );
    assert!(matches!(
        resources.compile(
            &selection,
            10,
            &[binding(b, 9, 1), binding(b, 10, 1)],
            &[],
            GraphLimits::default()
        ),
        Err(FabricCompileError::OverlappingBindings {
            first_frame: 9,
            second_frame: 10,
            ..
        })
    ));
    let mut wrong = binding(b, 9, 9);
    wrong.layout = ResourceLayout::Buffer {
        bytes: 32,
        usage: BufferUsages::STORAGE,
    };
    assert_eq!(
        resources
            .compile(
                &selection,
                10,
                &[wrong, binding(b, 10, 10)],
                &[],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::InvalidBinding {
            resource: B,
            frame: 9
        }
    );
}

#[test]
fn external_bindings_must_match_the_current_graph_footprint() {
    let catalog = history_fixture();
    let b = buffer(B, ResourceLifetime::Persistent);
    let resources =
        ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient), b]).unwrap();

    assert_eq!(
        resources
            .compile(
                &select(&catalog),
                10,
                &[binding(b, 8, 8), binding(b, 9, 9), binding(b, 10, 10)],
                &[],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::UnexpectedBinding {
            resource: B,
            frame: 8,
        }
    );

    let unused = buffer(ResourceKey::engine(77), ResourceLifetime::Persistent);
    assert_eq!(
        resources
            .compile(
                &select(&catalog),
                10,
                &[
                    binding(unused, 10, 77),
                    binding(b, 9, 9),
                    binding(b, 10, 10)
                ],
                &[],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::UnexpectedBinding {
            resource: ResourceKey::engine(77),
            frame: 10,
        }
    );

    // Transients are graph-owned even when their semantic key is referenced.
    // Silently discarding this binding would allocate different storage from
    // the one the caller supplied and bypass its layout/version validation.
    let transient = buffer(A, ResourceLifetime::Transient);
    assert_eq!(
        resources
            .compile(
                &select(&catalog),
                10,
                &[
                    binding(transient, 10, 77),
                    binding(b, 9, 9),
                    binding(b, 10, 10)
                ],
                &[],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::UnexpectedBinding {
            resource: A,
            frame: 10,
        }
    );
}

#[test]
fn history_requires_persistent_metadata_in_every_variant() {
    let catalog = history_fixture();
    assert!(matches!(
        ResourceTemplates::new(
            &catalog,
            &[
                buffer(A, ResourceLifetime::Transient),
                buffer(B, ResourceLifetime::Transient)
            ]
        ),
        Err(FabricCompileError::HistoryNotPersistent(B))
    ));
}

#[test]
fn current_frame_cycle_is_reported_by_authority() {
    let catalog = catalog(
        vec![variant(
            720,
            0,
            vec![
                pass(
                    1,
                    QueueClass::Compute,
                    vec![
                        usage(B, AccessKind::StorageRead),
                        usage(A, AccessKind::StorageWrite),
                    ],
                ),
                pass(
                    2,
                    QueueClass::Compute,
                    vec![
                        usage(A, AccessKind::StorageRead),
                        usage(B, AccessKind::StorageWrite),
                    ],
                ),
            ],
        )],
        vec![A, B],
    );
    let resources = ResourceTemplates::new(
        &catalog,
        &[
            buffer(A, ResourceLifetime::Transient),
            buffer(B, ResourceLifetime::Transient),
        ],
    )
    .unwrap();
    assert!(matches!(
        resources.compile(&select(&catalog), 1, &[], &[], GraphLimits::default()),
        Err(FabricCompileError::Graph(CompileError::Cycle { .. }))
    ));
}

#[test]
fn read_requires_producer_or_external_initial_storage() {
    let catalog = catalog(
        vec![variant(
            730,
            0,
            vec![pass(
                1,
                QueueClass::Compute,
                vec![usage(A, AccessKind::StorageRead)],
            )],
        )],
        vec![A],
    );
    let resources =
        ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient)]).unwrap();
    assert_eq!(
        resources
            .compile(&select(&catalog), 1, &[], &[], GraphLimits::default())
            .unwrap_err(),
        FabricCompileError::MissingProducer(A)
    );
    let template = buffer(A, ResourceLifetime::Imported);
    let resources = ResourceTemplates::new(&catalog, &[template]).unwrap();
    let compiled = resources
        .compile(
            &select(&catalog),
            1,
            &[binding(template, 1, 1)],
            &[],
            GraphLimits::default(),
        )
        .unwrap();
    assert!(compiled.versions().is_empty());
    assert_eq!(compiled.plan().barrier_count(), 0);
}

#[test]
fn absolute_persistent_versions_are_checked_for_capacity_and_overflow() {
    let catalog = catalog(
        vec![variant(
            740,
            0,
            vec![pass(
                1,
                QueueClass::Compute,
                vec![usage(A, AccessKind::StorageWrite)],
            )],
        )],
        vec![A],
    );
    let template = buffer(A, ResourceLifetime::Persistent);
    let resources = ResourceTemplates::new(&catalog, &[template]).unwrap();
    for (base, limit) in [(5, 5), (u32::MAX, u32::MAX)] {
        let mut external = binding(template, 10, 1);
        external.version = base;
        assert_eq!(
            resources
                .compile(
                    &select(&catalog),
                    10,
                    &[external],
                    &[],
                    GraphLimits {
                        versions: limit,
                        ..GraphLimits::default()
                    }
                )
                .unwrap_err(),
            FabricCompileError::Graph(CompileError::VersionOverflow {
                version: base,
                limit
            })
        );
    }
}

#[test]
fn all_variant_layouts_are_validated_even_when_not_selected() {
    let catalog = catalog(
        vec![
            variant(
                750,
                0,
                vec![pass(
                    1,
                    QueueClass::Compute,
                    vec![usage(A, AccessKind::StorageWrite)],
                )],
            ),
            variant(
                751,
                1,
                vec![pass(
                    2,
                    QueueClass::Graphics,
                    vec![usage(A, AccessKind::SampledRead)],
                )],
            ),
        ],
        vec![A],
    );
    assert!(matches!(
        ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient)]),
        Err(FabricCompileError::InvalidAccess {
            pass: PassKey(2),
            access: AccessKind::SampledRead,
            ..
        })
    ));
}

#[test]
fn template_table_rejects_missing_extra_duplicate_and_invalid_footprints() {
    let catalog = fixture();
    let a = buffer(A, ResourceLifetime::Transient);
    let b = buffer(B, ResourceLifetime::Transient);
    assert!(matches!(
        ResourceTemplates::new(&catalog, &[a]),
        Err(FabricCompileError::MissingTemplate(B))
    ));
    assert!(matches!(
        ResourceTemplates::new(&catalog, &[a, a, b]),
        Err(FabricCompileError::DuplicateTemplate(A))
    ));
    assert!(matches!(
        ResourceTemplates::new(
            &catalog,
            &[a, b, buffer(UNUSED, ResourceLifetime::Transient)]
        ),
        Err(FabricCompileError::UndeclaredTemplate(UNUSED))
    ));
    for alignment in [0, 3] {
        assert!(matches!(
            ResourceTemplates::new(&catalog, &[ResourceTemplate { alignment, ..a }, b]),
            Err(FabricCompileError::InvalidLayout(A))
        ));
    }
    assert!(matches!(
        ResourceTemplates::new(
            &catalog,
            &[
                ResourceTemplate {
                    layout: ResourceLayout::Buffer {
                        bytes: 0,
                        usage: BufferUsages::STORAGE
                    },
                    ..a
                },
                b
            ]
        ),
        Err(FabricCompileError::InvalidLayout(A))
    ));
}

#[test]
fn texture_mips_dimension_formats_and_access_usage_have_real_layout_checks() {
    let texture = ResourceLayout::Texture {
        extent: [8, 4, 2],
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        mip_levels: 3,
        samples: 1,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::RENDER_ATTACHMENT,
    };
    assert_eq!(texture.size(), Some((8 * 4 + 4 * 2 + 2) * 2 * 8));
    assert!(texture.supports(AccessKind::StorageWrite));
    assert!(texture.supports(AccessKind::SampledRead));
    assert!(texture.supports(AccessKind::ColorAttachment));
    assert!(!texture.supports(AccessKind::IndirectRead));
    assert!(!texture.supports(AccessKind::TransferWrite));
    let depth = ResourceLayout::Texture {
        extent: [8, 4, 1],
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::STORAGE_BINDING,
    };
    assert!(depth.supports(AccessKind::DepthAttachment));
    assert!(!depth.supports(AccessKind::ColorAttachment));
    assert!(!depth.supports(AccessKind::StorageWrite));
    let impossible = ResourceLayout::Texture {
        extent: [u32::MAX; 3],
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba32Float,
        mip_levels: 1,
        samples: 16,
        usage: TextureUsages::TEXTURE_BINDING,
    };
    assert!(impossible.size().is_none());
}

fn item(payload: DomainPayload) -> FrameRenderItem {
    item_with_object(
        RenderObjectId::Surface(SceneHandle {
            index: 1,
            generation: 1,
        }),
        payload,
    )
}
fn item_with_object(object_id: RenderObjectId, payload: DomainPayload) -> FrameRenderItem {
    FrameRenderItem {
        domain: object_id.domain(),
        object_id,
        history_key: 19,
        motion_source: MotionSource::Rigid,
        payload,
    }
}
fn payload(template: ResourceTemplate) -> DomainPayload {
    DomainPayload::Resource(ResourcePayload {
        resource: template.key,
        lifetime: template.lifetime,
        size: template.layout.size().unwrap(),
        alignment: template.alignment,
        layout: template.layout,
        role: PayloadRole(2),
    })
}

#[test]
fn payload_is_checked_against_actual_selected_accesses_type_and_layout() {
    let catalog = fixture();
    let a = buffer(A, ResourceLifetime::Transient);
    let b = buffer(B, ResourceLifetime::Transient);
    let resources = ResourceTemplates::new(&catalog, &[a, b]).unwrap();
    let selection = select(&catalog);
    let good = item(payload(a));
    resources
        .compile(&selection, 10, &[], &[good], GraphLimits::default())
        .unwrap();
    let mut bad_domain = good;
    bad_domain.domain = RenderDomain::Volume;
    assert_eq!(
        resources
            .compile(&selection, 10, &[], &[bad_domain], GraphLimits::default())
            .unwrap_err(),
        FabricCompileError::DomainIdentityMismatch {
            object: good.object_id,
            declared: RenderDomain::Volume,
        }
    );
    let mut bad_payload_domain = good;
    bad_payload_domain.object_id = RenderObjectId::Volume(DomainObjectHandle::new(1, 1));
    bad_payload_domain.domain = RenderDomain::Volume;
    assert_eq!(
        resources
            .compile(
                &selection,
                10,
                &[],
                &[bad_payload_domain],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::PayloadDomainMismatch(A)
    );
    let DomainPayload::Resource(mut wrong) = good.payload else {
        unreachable!()
    };
    wrong.size += 1;
    assert_eq!(
        resources
            .compile(
                &selection,
                10,
                &[],
                &[item(DomainPayload::Resource(wrong))],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::PayloadLayoutMismatch(A)
    );
    assert_eq!(
        resources
            .compile(
                &selection,
                10,
                &[],
                &[item(payload(buffer(UNUSED, ResourceLifetime::Transient)))],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::PayloadNotReferenced(UNUSED)
    );
    fn assert_copy<T: Copy>() {}
    assert_copy::<FrameRenderItem>();
    assert_copy::<PayloadValue>();
    resources
        .compile(
            &selection,
            10,
            &[],
            &[item(DomainPayload::Value(ValuePayload {
                role: PayloadRole(4),
                value: PayloadValue::Bounds {
                    min: [0.0; 3],
                    max: [1.0; 3],
                },
            }))],
            GraphLimits::default(),
        )
        .unwrap();
}

#[test]
fn payload_current_access_is_not_shadowed_by_an_earlier_history_access() {
    let catalog = history_fixture();
    let b = buffer(B, ResourceLifetime::Persistent);
    let resources =
        ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient), b]).unwrap();
    let compiled = resources
        .compile(
            &select(&catalog),
            10,
            &[binding(b, 9, 9), binding(b, 10, 10)],
            &[item(payload(b))],
            GraphLimits::default(),
        )
        .unwrap();
    assert_eq!(compiled.items(), &[item(payload(b))]);
}

#[test]
fn payload_roles_are_unique_per_object_domain_and_history_stream() {
    let catalog = fixture();
    let a = buffer(A, ResourceLifetime::Transient);
    let resources =
        ResourceTemplates::new(&catalog, &[a, buffer(B, ResourceLifetime::Transient)]).unwrap();
    let selection = select(&catalog);
    let mut first = item(payload(a));
    // History keys are application-defined; zero is not a reserved sentinel.
    first.history_key = 0;
    assert_eq!(
        resources
            .compile(&selection, 10, &[], &[first, first], GraphLimits::default())
            .unwrap_err(),
        FabricCompileError::DuplicatePayloadRole {
            object: first.object_id,
            domain: first.domain,
            history_key: 0,
            role: first.payload_role(),
        }
    );
    let mut second = first;
    second.object_id = RenderObjectId::Surface(SceneHandle {
        generation: 2,
        ..SceneHandle::new(1, 1)
    });
    let mut third = first;
    third.history_key = 1;
    resources
        .compile(
            &selection,
            10,
            &[],
            &[first, second, third],
            GraphLimits::default(),
        )
        .unwrap();
}

#[test]
fn non_surface_generational_identity_distinguishes_reused_slots() {
    let catalog = fixture();
    let resources = ResourceTemplates::new(
        &catalog,
        &[
            buffer(A, ResourceLifetime::Transient),
            buffer(B, ResourceLifetime::Transient),
        ],
    )
    .unwrap();
    let first = item_with_object(
        RenderObjectId::Particle(DomainObjectHandle::new(7, 1)),
        DomainPayload::Value(ValuePayload {
            value: PayloadValue::Scalar(1.0),
            role: PayloadRole(8),
        }),
    );
    let second = item_with_object(
        RenderObjectId::Particle(DomainObjectHandle::new(7, 2)),
        DomainPayload::Value(ValuePayload {
            value: PayloadValue::Scalar(2.0),
            role: PayloadRole(8),
        }),
    );
    resources
        .compile(
            &select(&catalog),
            10,
            &[],
            &[first, second],
            GraphLimits::default(),
        )
        .unwrap();
}

#[test]
fn execution_uses_bound_passes_and_exact_barriers_and_stops_on_sink_errors() {
    #[derive(Default)]
    struct Sink {
        events: Vec<u32>,
        barriers: Vec<Barrier>,
        fail: bool,
    }
    impl FeatureGraphSink for Sink {
        type Error = &'static str;
        fn begin(
            &mut self,
            resources: &[BoundResource],
            items: &[FrameRenderItem],
            plan: &ExecutionPlan,
        ) -> Result<(), Self::Error> {
            assert_eq!(resources.len(), 2);
            assert!(items.is_empty());
            assert_eq!(plan.declared_pass_count, 3);
            self.events.push(0);
            Ok(())
        }
        fn barrier(&mut self, barrier: &Barrier) -> Result<(), Self::Error> {
            self.barriers.push(*barrier);
            self.events.push(100 + barrier.destination.0);
            Ok(())
        }
        fn pass(&mut self, pass: &PassSpec) -> Result<(), Self::Error> {
            assert_eq!(pass.variant(), VariantId(700));
            assert_eq!(pass.feature(), ROOT);
            assert_eq!(pass.template().queue, pass.descriptor().queue);
            self.events.push(pass.key().0);
            if self.fail {
                Err("recording failed")
            } else {
                Ok(())
            }
        }
        fn finish(&mut self, _: &ExecutionPlan) -> Result<(), Self::Error> {
            self.events.push(999);
            Ok(())
        }
    }
    let graph = compile_fixture(GraphLimits::default()).unwrap();
    let expected = graph.barriers().to_vec();
    let mut sink = Sink::default();
    graph.execute(&mut sink).unwrap();
    assert_eq!(sink.barriers, expected);
    assert_eq!(sink.events, vec![0, 1, 101, 2, 102, 3, 999]);
    let mut sink = Sink {
        fail: true,
        ..Sink::default()
    };
    assert_eq!(
        compile_fixture(GraphLimits::default())
            .unwrap()
            .execute(&mut sink),
        Err("recording failed")
    );
    assert_eq!(sink.events, vec![0, 1]);
}

#[test]
fn compiling_frames_does_not_mutate_frozen_selection_or_roots() {
    let catalog = fixture();
    let selection = select(&catalog);
    let before = selection.clone();
    let resources = ResourceTemplates::new(
        &catalog,
        &[
            buffer(A, ResourceLifetime::Transient),
            buffer(B, ResourceLifetime::Transient),
        ],
    )
    .unwrap();
    for frame in [0, 10, u64::MAX] {
        resources
            .compile(&selection, frame, &[], &[], GraphLimits::default())
            .unwrap();
    }
    assert_eq!(selection, before);
}

#[test]
fn bound_accesses_project_authority_outputs_and_history_content_versions() {
    let graph = compile_fixture(GraphLimits::default()).unwrap();
    for (pass, expected) in graph.passes().iter().zip([
        vec![(A, PassKey(1))],
        vec![(A, PassKey(1)), (B, PassKey(2))],
        vec![(B, PassKey(2))],
    ]) {
        assert_eq!(pass.accesses().len(), pass.descriptor().accesses.len());
        for (access, (key, producer)) in pass.accesses().iter().zip(expected) {
            assert_eq!(access.declaration.key, key);
            assert_eq!(access.resource, graph.resource_id(key, 10).unwrap());
            assert_eq!(access.frame, 10);
            assert_eq!(access.ssa_version, 1);
            assert_eq!(access.content_version, 1);
            assert_eq!(access.producer, Some(producer));
        }
    }
    let catalog = history_fixture();
    let b = buffer(B, ResourceLifetime::Persistent);
    let graph = ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient), b])
        .unwrap()
        .compile(
            &select(&catalog),
            10,
            &[binding(b, 9, 9), binding(b, 10, 10)],
            &[],
            GraphLimits::default(),
        )
        .unwrap();
    let history = graph.passes()[0].accesses()[0];
    assert_eq!(history.frame, 9);
    assert_eq!(history.ssa_version, 0);
    assert_eq!(history.content_version, 5);
    assert_eq!(history.producer, None);
    let current = graph.passes()[1].accesses()[1];
    assert_eq!(current.frame, 10);
    assert_eq!(current.ssa_version, 1);
    assert_eq!(current.content_version, 6);
    assert_eq!(current.producer, Some(PassKey(2)));
}

#[test]
fn multiple_history_lags_and_read_only_external_versions_are_bound_and_checked() {
    let mut lag1 = usage(A, AccessKind::StorageRead);
    lag1.frame_lag = FrameLag::new(1);
    let mut lag2 = lag1;
    lag2.frame_lag = FrameLag::new(2);
    let catalog = catalog(
        vec![variant(
            760,
            0,
            vec![pass(
                1,
                QueueClass::Compute,
                vec![lag2, usage(A, AccessKind::StorageRead), lag1],
            )],
        )],
        vec![A],
    );
    let a = buffer(A, ResourceLifetime::Persistent);
    let resources = ResourceTemplates::new(&catalog, &[a]).unwrap();
    let mut current = binding(a, 10, 10);
    current.version = 7;
    let mut old = binding(a, 8, 8);
    old.version = 3;
    let bindings = [current, binding(a, 9, 9), old];
    let graph = resources
        .compile(
            &select(&catalog),
            10,
            &bindings,
            &[],
            GraphLimits::default(),
        )
        .unwrap();
    assert_eq!(graph.resources().len(), 3);
    assert_eq!(
        graph.passes()[0]
            .accesses()
            .iter()
            .map(|use_| (
                use_.frame,
                use_.content_version,
                use_.ssa_version,
                use_.producer
            ))
            .collect::<Vec<_>>(),
        vec![(8, 3, 0, None), (10, 7, 0, None), (9, 5, 0, None)]
    );
    assert!(graph.versions().is_empty());
    assert_eq!(graph.plan().barrier_count(), 0);
    assert_eq!(
        resources
            .compile(
                &select(&catalog),
                10,
                &bindings,
                &[],
                GraphLimits {
                    versions: 6,
                    ..GraphLimits::default()
                }
            )
            .unwrap_err(),
        FabricCompileError::Graph(CompileError::VersionOverflow {
            version: 7,
            limit: 6
        })
    );
}

#[test]
fn all_variant_access_views_preserve_pass_identity_and_do_not_select() {
    let catalog = catalog(
        vec![
            variant(
                770,
                0,
                vec![pass(
                    1,
                    QueueClass::Compute,
                    vec![usage(A, AccessKind::StorageWrite)],
                )],
            ),
            variant(
                771,
                1,
                vec![pass(
                    2,
                    QueueClass::Transfer,
                    vec![
                        usage(B, AccessKind::TransferRead),
                        usage(A, AccessKind::TransferWrite),
                    ],
                )],
            ),
        ],
        vec![A, B],
    );
    let definition = catalog.feature(ROOT).unwrap();
    assert_eq!(definition.accesses().count(), 3);
    assert_eq!(
        definition.reads().collect::<Vec<_>>(),
        vec![VariantResourceUse {
            feature: ROOT,
            variant: VariantId(771),
            pass: PassKey(2),
            usage: usage(B, AccessKind::TransferRead),
        }]
    );
    assert_eq!(
        definition
            .writes()
            .map(|use_| (use_.variant, use_.pass))
            .collect::<Vec<_>>(),
        vec![(VariantId(770), PassKey(1)), (VariantId(771), PassKey(2))]
    );
    assert_eq!(
        definition.variants[1].passes[0]
            .reads()
            .copied()
            .collect::<Vec<_>>(),
        vec![usage(B, AccessKind::TransferRead)]
    );
    assert_eq!(
        definition.variants[1].passes[0]
            .writes()
            .copied()
            .collect::<Vec<_>>(),
        vec![usage(A, AccessKind::TransferWrite)]
    );
    assert_eq!(select(&catalog).selected()[0].variant_id(), VariantId(770));
}

#[test]
fn selected_cross_feature_writers_and_same_pass_feedback_are_rejected() {
    const OTHER: FeatureId = FeatureId(800);
    let catalog = FeatureCatalog::new(
        table(vec![
            FeatureDefinition::subsystem(
                ROOT,
                RenderDomain::Surface,
                ServiceSet::EMPTY,
                table(vec![variant(
                    780,
                    0,
                    vec![pass(
                        1,
                        QueueClass::Compute,
                        vec![usage(A, AccessKind::StorageWrite)],
                    )],
                )]),
            ),
            FeatureDefinition::subsystem(
                OTHER,
                RenderDomain::Particle,
                ServiceSet::EMPTY,
                table(vec![variant(
                    781,
                    0,
                    vec![pass(
                        2,
                        QueueClass::Compute,
                        vec![usage(A, AccessKind::StorageWrite)],
                    )],
                )]),
            ),
        ]),
        table(vec![A]),
    )
    .unwrap();
    let roots = FrozenRoots::from_build_profile(&catalog, &[ROOT, OTHER], &[]).unwrap();
    let selection = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .unwrap();
    assert_eq!(
        ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient)])
            .unwrap()
            .compile(&selection, 10, &[], &[], GraphLimits::default())
            .unwrap_err(),
        FabricCompileError::MultipleWriters(A)
    );
    let feedback = self::catalog(
        vec![variant(
            782,
            0,
            vec![pass(
                3,
                QueueClass::Compute,
                vec![
                    usage(A, AccessKind::StorageRead),
                    usage(A, AccessKind::StorageWrite),
                ],
            )],
        )],
        vec![A],
    );
    assert_eq!(
        ResourceTemplates::new(&feedback, &[buffer(A, ResourceLifetime::Transient)])
            .unwrap()
            .compile(&select(&feedback), 10, &[], &[], GraphLimits::default())
            .unwrap_err(),
        FabricCompileError::ReadWriteSamePass {
            pass: PassKey(3),
            resource: A
        }
    );
}

#[test]
fn external_storage_requires_exact_type_alignment_range_and_unique_address() {
    let catalog = catalog(
        vec![variant(
            790,
            0,
            vec![pass(
                1,
                QueueClass::Compute,
                vec![
                    usage(A, AccessKind::StorageRead),
                    usage(B, AccessKind::StorageRead),
                ],
            )],
        )],
        vec![A, B],
    );
    let a = buffer(A, ResourceLifetime::Imported);
    let b = buffer(B, ResourceLifetime::Imported);
    let resources = ResourceTemplates::new(&catalog, &[a, b]).unwrap();
    let first = binding(a, 10, 1);
    let second = binding(b, 10, 2);
    let unknown = FrameResourceBinding {
        key: ResourceKey::engine(77),
        ..first
    };
    assert_eq!(
        resources
            .compile(
                &select(&catalog),
                10,
                &[unknown],
                &[],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::UnexpectedBinding {
            resource: ResourceKey::engine(77),
            frame: 10
        }
    );
    for bad in [
        FrameResourceBinding {
            physical: PhysicalResource::Texture(TextureId::from_parts(1, 1)),
            ..first
        },
        FrameResourceBinding {
            alignment: 32,
            ..first
        },
        FrameResourceBinding {
            physical: PhysicalResource::Buffer {
                id: BufferId::from_parts(1, 1),
                offset: 1,
            },
            ..first
        },
        FrameResourceBinding {
            physical: PhysicalResource::Buffer {
                id: BufferId::from_parts(1, 1),
                offset: u64::MAX - 15,
            },
            ..first
        },
    ] {
        assert_eq!(
            resources
                .compile(
                    &select(&catalog),
                    10,
                    &[bad, second],
                    &[],
                    GraphLimits::default()
                )
                .unwrap_err(),
            FabricCompileError::InvalidBinding {
                resource: A,
                frame: 10
            }
        );
    }
    assert_eq!(
        resources
            .compile(
                &select(&catalog),
                10,
                &[first, first, second],
                &[],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::DuplicateBinding {
            resource: A,
            frame: 10
        }
    );
    let adjacent = FrameResourceBinding {
        physical: PhysicalResource::Buffer {
            id: BufferId::from_parts(1, 1),
            offset: 64,
        },
        ..second
    };
    resources
        .compile(
            &select(&catalog),
            10,
            &[adjacent, first],
            &[],
            GraphLimits::default(),
        )
        .unwrap();
    let overlapping = FrameResourceBinding {
        physical: PhysicalResource::Buffer {
            id: BufferId::from_parts(1, 1),
            offset: 48,
        },
        ..second
    };
    assert!(matches!(
        resources.compile(
            &select(&catalog),
            10,
            &[first, overlapping],
            &[],
            GraphLimits::default()
        ),
        Err(FabricCompileError::OverlappingBindings {
            first: A,
            second: B,
            ..
        })
    ));
}

#[test]
fn resource_footprint_sum_overflow_is_an_authority_error() {
    let catalog = fixture();
    let enormous = ResourceTemplate {
        layout: ResourceLayout::Buffer {
            bytes: u64::MAX,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        },
        alignment: 1,
        ..buffer(A, ResourceLifetime::Transient)
    };
    assert_eq!(
        ResourceTemplates::new(
            &catalog,
            &[enormous, buffer(B, ResourceLifetime::Transient)]
        )
        .unwrap()
        .compile(&select(&catalog), 10, &[], &[], GraphLimits::default())
        .unwrap_err(),
        FabricCompileError::Graph(CompileError::MemoryOverflow)
    );
}

#[test]
fn shared_variants_do_not_allow_foreign_feature_contracts_or_candidate_sets() {
    const SERVICE: ServiceId = ServiceId(720);
    const PROVIDER: FeatureId = FeatureId(721);
    let variants = table(vec![variant(720, 0, vec![]), variant(721, 1, vec![])]);
    let root = FeatureDefinition::subsystem(
        ROOT,
        RenderDomain::Surface,
        ServiceSet::EMPTY,
        &variants[1..],
    );
    let source = FeatureCatalog::new(table(vec![root]), &[]).unwrap();
    let selection = select(&source);
    let provider = FeatureDefinition::service(
        PROVIDER,
        table(vec![ServiceOutput {
            service: SERVICE,
            resources: &[],
        }]),
        ServiceSet::EMPTY,
        table(vec![variant(722, 0, vec![])]),
    );
    for changed in [
        FeatureDefinition {
            requires_services: ServiceSet::new(&[SERVICE]),
            ..root
        },
        FeatureDefinition {
            domain: Some(RenderDomain::Volume),
            ..root
        },
        FeatureDefinition::response(ROOT, ServiceSet::EMPTY, root.variants),
        FeatureDefinition { variants, ..root },
    ] {
        let target = FeatureCatalog::new(table(vec![changed, provider]), &[]).unwrap();
        assert_eq!(
            ResourceTemplates::new(&target, &[])
                .unwrap()
                .compile(&selection, 10, &[], &[], GraphLimits::default())
                .unwrap_err(),
            FabricCompileError::ForeignSelection(ROOT)
        );
    }
    // Sharing an unchanged definition across catalogs remains supported.
    let equivalent = FeatureCatalog::new(table(vec![provider, root]), &[]).unwrap();
    ResourceTemplates::new(&equivalent, &[])
        .unwrap()
        .compile(&selection, 10, &[], &[], GraphLimits::default())
        .unwrap();
}

#[test]
fn shared_provider_variants_cannot_reinterpret_the_published_service_interface() {
    const SERVICE: ServiceId = ServiceId(730);
    const PROVIDER: FeatureId = FeatureId(731);
    let provider = FeatureDefinition::service(
        PROVIDER,
        table(vec![ServiceOutput {
            service: SERVICE,
            resources: &[A],
        }]),
        ServiceSet::EMPTY,
        table(vec![variant(
            730,
            0,
            vec![pass(
                730,
                QueueClass::Compute,
                vec![
                    usage(A, AccessKind::StorageWrite),
                    usage(B, AccessKind::StorageWrite),
                ],
            )],
        )]),
    );
    let root = FeatureDefinition::subsystem(
        ROOT,
        RenderDomain::Surface,
        ServiceSet::new(&[SERVICE]),
        table(vec![variant(731, 0, vec![])]),
    );
    let source = FeatureCatalog::new(table(vec![root, provider]), &[A, B]).unwrap();
    let selection = select(&source);
    assert_eq!(selection.service_resources(SERVICE), Some(&[A][..]));
    let changed = FeatureDefinition {
        kind: FeatureKind::Service {
            provides: table(vec![ServiceOutput {
                service: SERVICE,
                resources: &[B],
            }]),
        },
        ..provider
    };
    let target = FeatureCatalog::new(table(vec![root, changed]), &[A, B]).unwrap();
    assert_eq!(
        ResourceTemplates::new(
            &target,
            &[
                buffer(A, ResourceLifetime::Transient),
                buffer(B, ResourceLifetime::Transient)
            ]
        )
        .unwrap()
        .compile(&selection, 10, &[], &[], GraphLimits::default())
        .unwrap_err(),
        FabricCompileError::ForeignSelection(PROVIDER)
    );
}

#[test]
fn foreign_selection_is_rejected_and_template_binding_order_does_not_change_plan() {
    let catalog = fixture();
    let other = fixture();
    let a = buffer(A, ResourceLifetime::Transient);
    let b = buffer(B, ResourceLifetime::Transient);
    let resources = ResourceTemplates::new(&catalog, &[a, b]).unwrap();
    assert_eq!(
        resources
            .compile(&select(&other), 10, &[], &[], GraphLimits::default())
            .unwrap_err(),
        FabricCompileError::ForeignSelection(ROOT)
    );
    let expected = resources
        .compile(&select(&catalog), 10, &[], &[], GraphLimits::default())
        .unwrap();
    let reordered = ResourceTemplates::new(&catalog, &[b, a])
        .unwrap()
        .compile(&select(&catalog), 10, &[], &[], GraphLimits::default())
        .unwrap();
    assert_eq!(expected.plan().order, reordered.plan().order);
    assert_eq!(expected.barriers(), reordered.barriers());
    assert_eq!(expected.versions(), reordered.versions());
    assert_eq!(
        expected
            .passes()
            .iter()
            .map(PassSpec::accesses)
            .collect::<Vec<_>>(),
        reordered
            .passes()
            .iter()
            .map(PassSpec::accesses)
            .collect::<Vec<_>>()
    );
}

#[test]
fn inactive_variant_and_history_only_resources_cannot_masquerade_as_current_payload() {
    let catalog = catalog(
        vec![
            variant(
                810,
                0,
                vec![pass(
                    1,
                    QueueClass::Compute,
                    vec![usage(A, AccessKind::StorageWrite)],
                )],
            ),
            variant(
                811,
                1,
                vec![pass(
                    2,
                    QueueClass::Compute,
                    vec![usage(B, AccessKind::StorageWrite)],
                )],
            ),
        ],
        vec![A, B],
    );
    let a = buffer(A, ResourceLifetime::Transient);
    let b = buffer(B, ResourceLifetime::Transient);
    assert_eq!(
        ResourceTemplates::new(&catalog, &[a, b])
            .unwrap()
            .compile(
                &select(&catalog),
                10,
                &[],
                &[item(payload(b))],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::PayloadNotReferenced(B)
    );
    let mut lag = usage(A, AccessKind::StorageRead);
    lag.frame_lag = FrameLag::new(1);
    let history = self::catalog(
        vec![variant(
            812,
            0,
            vec![pass(3, QueueClass::Compute, vec![lag])],
        )],
        vec![A],
    );
    let a = buffer(A, ResourceLifetime::Persistent);
    assert_eq!(
        ResourceTemplates::new(&history, &[a])
            .unwrap()
            .compile(
                &select(&history),
                10,
                &[binding(a, 9, 1)],
                &[item(payload(a))],
                GraphLimits::default()
            )
            .unwrap_err(),
        FabricCompileError::PayloadNotReferenced(A)
    );
}

#[test]
fn validated_frame_items_reach_the_execution_sink_without_a_second_extraction() {
    struct Sink {
        items: Vec<FrameRenderItem>,
    }
    impl FeatureGraphSink for Sink {
        type Error = ();
        fn begin(
            &mut self,
            _: &[BoundResource],
            items: &[FrameRenderItem],
            _: &ExecutionPlan,
        ) -> Result<(), ()> {
            self.items.extend_from_slice(items);
            Ok(())
        }
        fn barrier(&mut self, _: &Barrier) -> Result<(), ()> {
            Ok(())
        }
        fn pass(&mut self, _: &PassSpec) -> Result<(), ()> {
            Ok(())
        }
        fn finish(&mut self, _: &ExecutionPlan) -> Result<(), ()> {
            Ok(())
        }
    }
    let catalog = fixture();
    let a = buffer(A, ResourceLifetime::Transient);
    let items = [
        item(payload(a)),
        item(DomainPayload::Value(ValuePayload {
            role: PayloadRole(3),
            value: PayloadValue::Scalar(42.0),
        })),
    ];
    let compiled = ResourceTemplates::new(&catalog, &[a, buffer(B, ResourceLifetime::Transient)])
        .unwrap()
        .compile(&select(&catalog), 10, &[], &items, GraphLimits::default())
        .unwrap();
    assert_eq!(compiled.items(), &items);
    let mut sink = Sink { items: Vec::new() };
    compiled.execute(&mut sink).unwrap();
    assert_eq!(sink.items, items);
}

#[test]
fn explicit_writer_chain_keeps_initial_and_intermediate_read_provenance() {
    let access = |kind, input| ResourceUse {
        input,
        ..usage(A, kind)
    };
    let catalog = catalog(
        vec![variant(
            790,
            0,
            vec![
                pass(
                    40,
                    QueueClass::Compute,
                    vec![access(
                        AccessKind::StorageRead,
                        ResourceInput::ProducedBy(PassKey(30)),
                    )],
                ),
                pass(
                    30,
                    QueueClass::Compute,
                    vec![access(
                        AccessKind::StorageWrite,
                        ResourceInput::ProducedBy(PassKey(10)),
                    )],
                ),
                pass(
                    20,
                    QueueClass::Compute,
                    vec![access(
                        AccessKind::StorageRead,
                        ResourceInput::ProducedBy(PassKey(10)),
                    )],
                ),
                pass(
                    10,
                    QueueClass::Compute,
                    vec![access(AccessKind::StorageWrite, ResourceInput::Initial)],
                ),
                pass(
                    5,
                    QueueClass::Compute,
                    vec![access(AccessKind::StorageRead, ResourceInput::Initial)],
                ),
            ],
        )],
        vec![A],
    );
    let a = buffer(A, ResourceLifetime::Imported);
    let graph = ResourceTemplates::new(&catalog, &[a])
        .unwrap()
        .compile(
            &select(&catalog),
            10,
            &[binding(a, 10, 1)],
            &[],
            GraphLimits::default(),
        )
        .unwrap();
    let keys: Vec<_> = graph
        .plan()
        .order
        .iter()
        .map(|&id| graph.passes()[id].key())
        .collect();
    assert_eq!(
        keys,
        vec![
            PassKey(5),
            PassKey(10),
            PassKey(20),
            PassKey(30),
            PassKey(40)
        ]
    );
    for (key, input, output, input_producer) in [
        (5, 0, 0, None),
        (10, 0, 1, None),
        (20, 1, 1, Some(PassKey(10))),
        (30, 1, 2, Some(PassKey(10))),
        (40, 2, 2, Some(PassKey(30))),
    ] {
        let pass = graph
            .passes()
            .iter()
            .find(|pass| pass.key() == PassKey(key))
            .unwrap();
        let bound = pass.accesses()[0];
        assert_eq!(bound.input_ssa_version, input);
        assert_eq!(bound.ssa_version, output);
        assert_eq!(bound.input_content_version, 5 + input);
        assert_eq!(bound.content_version, 5 + output);
        assert_eq!(bound.input_producer, input_producer);
    }
    assert_eq!(graph.versions().len(), 2);
    assert_eq!(graph.versions()[1].writer, PassKey(30));
}

#[test]
fn symbolic_mixed_writer_chain_preserves_admission_ids_and_numeric_plan() {
    let use_ = |key, access, input| ResourceUse {
        input,
        ..usage(key, access)
    };
    let writes = |input| {
        vec![
            use_(A, AccessKind::StorageWrite, input),
            use_(B, AccessKind::StorageWrite, input),
        ]
    };
    let reads = |producer| {
        vec![
            use_(
                A,
                AccessKind::StorageRead,
                ResourceInput::ProducedBy(PassKey(producer)),
            ),
            use_(
                B,
                AccessKind::SampledRead,
                ResourceInput::ProducedBy(PassKey(producer)),
            ),
        ]
    };
    let mut control = pass(50, QueueClass::Transfer, vec![]);
    control.depends_on = table(vec![PassKey(40), PassKey(20)]);
    let catalog = catalog(
        vec![variant(
            796,
            0,
            vec![
                control,
                pass(40, QueueClass::Compute, reads(30)),
                pass(
                    30,
                    QueueClass::Compute,
                    writes(ResourceInput::ProducedBy(PassKey(10))),
                ),
                pass(20, QueueClass::Compute, reads(10)),
                pass(10, QueueClass::Compute, writes(ResourceInput::Initial)),
            ],
        )],
        vec![A, B],
    );
    let texture = ResourceTemplate {
        key: B,
        lifetime: ResourceLifetime::Transient,
        alignment: 256,
        layout: ResourceLayout::Texture {
            extent: [4, 4, 1],
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        },
    };
    let resources =
        ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient), texture])
            .unwrap();
    let graph = resources
        .compile(&select(&catalog), 10, &[], &[], GraphLimits::default())
        .unwrap();
    assert_eq!(
        graph.passes().iter().map(PassSpec::key).collect::<Vec<_>>(),
        [
            PassKey(10),
            PassKey(30),
            PassKey(40),
            PassKey(20),
            PassKey(50)
        ]
    );
    assert_eq!(
        graph
            .plan()
            .order
            .iter()
            .map(|&id| graph.passes()[id].key())
            .collect::<Vec<_>>(),
        [
            PassKey(10),
            PassKey(20),
            PassKey(30),
            PassKey(40),
            PassKey(50)
        ]
    );
    let mut numeric = GpuFrameGraphBuilder::default();
    for bound in graph.resources() {
        assert_eq!(numeric.add_resource(bound.descriptor.clone()), bound.id);
    }
    let mut inputs = Vec::new();
    for pass in graph.passes() {
        assert_eq!(numeric.add_pass(pass.descriptor().clone()), pass.id());
        inputs.push(
            pass.accesses()
                .iter()
                .map(|bound| bound.input_ssa_version)
                .collect(),
        );
        for dependency in &pass.descriptor().depends_on {
            assert!(dependency.0 < pass.id().0);
        }
        if pass.key() == PassKey(50) {
            assert!(pass.accesses().is_empty());
            continue;
        }
        // Declaration columns are buffer then texture; authority columns are
        // texture then buffer. Binding must preserve both resource identities.
        assert_eq!(pass.accesses()[0].declaration.key, A);
        assert_eq!(pass.accesses()[1].declaration.key, B);
        for (ordinal, compiled) in [(0, 1), (1, 0)] {
            let bound = pass.accesses()[ordinal];
            let authority = graph.plan().accesses[pass.id().0 as usize][compiled];
            assert_eq!(bound.input_ssa_version, authority.input_version);
            assert_eq!(
                bound.ssa_version,
                authority.output_version.unwrap_or(authority.input_version)
            );
            assert_eq!(bound.input_content_version, bound.input_ssa_version);
            assert_eq!(bound.content_version, bound.ssa_version);
            let expected = match pass.key() {
                PassKey(10) => (0, 1, None),
                PassKey(30) => (1, 2, Some(PassKey(10))),
                PassKey(20) => (1, 1, Some(PassKey(10))),
                PassKey(40) => (2, 2, Some(PassKey(30))),
                _ => unreachable!(),
            };
            assert_eq!(
                (
                    bound.input_ssa_version,
                    bound.ssa_version,
                    bound.input_producer
                ),
                expected
            );
        }
    }
    assert_eq!(&numeric.compile_with_inputs(&inputs).unwrap(), graph.plan());
    assert_eq!(graph.versions().len(), 4);
    assert!(matches!(
        resources.compile(
            &select(&catalog),
            10,
            &[],
            &[],
            GraphLimits {
                versions: 1,
                ..GraphLimits::default()
            }
        ),
        Err(FabricCompileError::Graph(CompileError::VersionOverflow {
            version: 1,
            limit: 1
        }))
    ));
}

#[test]
fn missing_wrong_and_unselected_producers_fail_before_recording() {
    for (producer, write_key) in [(PassKey(99), B), (PassKey(2), B)] {
        let mut read = usage(A, AccessKind::StorageRead);
        read.input = ResourceInput::ProducedBy(producer);
        let catalog = catalog(
            vec![variant(
                791,
                0,
                vec![
                    pass(1, QueueClass::Compute, vec![read]),
                    pass(
                        2,
                        QueueClass::Compute,
                        vec![usage(write_key, AccessKind::StorageWrite)],
                    ),
                ],
            )],
            vec![A, B],
        );
        let result = ResourceTemplates::new(
            &catalog,
            &[
                buffer(A, ResourceLifetime::Imported),
                buffer(B, ResourceLifetime::Transient),
            ],
        )
        .unwrap()
        .compile(&select(&catalog), 10, &[], &[], GraphLimits::default());
        assert_eq!(
            result.err(),
            Some(FabricCompileError::InvalidProducer {
                pass: PassKey(1),
                resource: A,
                producer,
            })
        );
    }
}

#[test]
fn transient_initial_reads_and_missing_control_prerequisites_are_rejected() {
    let mut read = usage(A, AccessKind::StorageRead);
    read.input = ResourceInput::Initial;
    let catalog = catalog(
        vec![variant(
            792,
            0,
            vec![
                pass(1, QueueClass::Compute, vec![read]),
                pass(
                    2,
                    QueueClass::Compute,
                    vec![usage(A, AccessKind::StorageWrite)],
                ),
            ],
        )],
        vec![A],
    );
    assert_eq!(
        ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient)])
            .unwrap()
            .compile(&select(&catalog), 0, &[], &[], GraphLimits::default())
            .err(),
        Some(FabricCompileError::MissingProducer(A))
    );
    let mut dependent = pass(3, QueueClass::Compute, vec![]);
    dependent.depends_on = table(vec![PassKey(99)]);
    let catalog = self::catalog(vec![variant(793, 0, vec![dependent])], vec![]);
    assert_eq!(
        ResourceTemplates::new(&catalog, &[])
            .unwrap()
            .compile(&select(&catalog), 0, &[], &[], GraphLimits::default())
            .err(),
        Some(FabricCompileError::MissingSelectedPass(PassKey(99)))
    );
}

#[test]
fn explicit_resource_and_control_cycles_report_authoritative_diagnostics() {
    let write = |producer| ResourceUse {
        input: ResourceInput::ProducedBy(PassKey(producer)),
        ..usage(A, AccessKind::StorageWrite)
    };
    let catalog = catalog(
        vec![variant(
            794,
            0,
            vec![
                pass(1, QueueClass::Compute, vec![write(2)]),
                pass(2, QueueClass::Compute, vec![write(1)]),
            ],
        )],
        vec![A],
    );
    let result = ResourceTemplates::new(&catalog, &[buffer(A, ResourceLifetime::Transient)])
        .unwrap()
        .compile(&select(&catalog), 0, &[], &[], GraphLimits::default());
    let Some(FabricCompileError::Graph(CompileError::Cycle { cycle, names, .. })) = result.err()
    else {
        panic!("resource cycle must be rejected by the authoritative scheduler");
    };
    assert_eq!(cycle.first(), cycle.last());
    assert!(names.iter().all(|name| name.starts_with("feature-pass:")));
    let mut first = pass(1, QueueClass::Compute, vec![]);
    first.depends_on = table(vec![PassKey(2)]);
    let mut second = pass(2, QueueClass::Compute, vec![]);
    second.depends_on = table(vec![PassKey(1)]);
    let catalog = self::catalog(vec![variant(795, 0, vec![first, second])], vec![]);
    assert!(matches!(
        ResourceTemplates::new(&catalog, &[]).unwrap().compile(
            &select(&catalog),
            0,
            &[],
            &[],
            GraphLimits::default()
        ),
        Err(FabricCompileError::Graph(CompileError::Cycle { .. }))
    ));
}

/// CPU declaration -> the production runtime -> NullDevice command recording.
/// No alternate executor and no GPU/adapter availability shortcut is involved.
mod object_instances {
    use super::*;
    use prism_render_driver::{
        BufferDescriptor, BufferState, DeviceDescriptor, Extent3d, Instance, RenderDevice,
        TextureDescriptor, TextureState, TextureViewDescriptor, TextureViewDimension,
        TextureViewId,
    };
    use prism_render_graph::{ImportedBuffer, ImportedTexture};
    use std::sync::{Arc, Mutex};

    const PROVIDER: FeatureId = FeatureId(701);
    const SERVICE: ServiceId = ServiceId(701);
    const SHARED: ResourceKey = ResourceKey::of(PROVIDER, 0);
    const ROLE: PayloadRole = PayloadRole(1);
    const OTHER_ROLE: PayloadRole = PayloadRole(2);
    const FRAME: u64 = 10;
    const PASSES: [PassKey; 5] = [PassKey(1), PassKey(2), PassKey(3), PassKey(4), PassKey(10)];

    fn identities() -> Vec<ObjectResourceIdentity> {
        let first = RenderObjectId::Hair(DomainObjectHandle::new(4, 1));
        let recycled = RenderObjectId::Hair(DomainObjectHandle::new(4, 2));
        [(first, 7), (recycled, 7), (first, 8)]
            .into_iter()
            .flat_map(|(object, history)| {
                [ROLE, OTHER_ROLE].map(|role| ObjectResourceIdentity::new(object, history, role))
            })
            .collect()
    }

    struct Fixture {
        catalog: FeatureCatalog,
        templates: Vec<ResourceTemplate>,
        scopes: Vec<ObjectResourceAccess>,
        items: Vec<FrameRenderItem>,
        bindings: Vec<ObjectResourceBinding>,
    }

    impl Fixture {
        fn new(physical: BufferId) -> Self {
            let produced = |writer, kind| ResourceUse {
                input: ResourceInput::ProducedBy(PassKey(writer)),
                ..usage(A, kind)
            };
            let initial = ResourceUse {
                input: ResourceInput::Initial,
                ..usage(A, AccessKind::StorageWrite)
            };
            // Roles distinguish instances, not a waiver of the catalog's
            // key-level writer contract. The second role is imported/read-only;
            // the first exercises initial and intermediate SSA writes.
            let imported_role = ResourceUse {
                input: ResourceInput::Initial,
                ..usage(A, AccessKind::StorageRead)
            };
            let history = ResourceUse {
                input: ResourceInput::Initial,
                frame_lag: Some(FrameLag::new(1).unwrap()),
                ..usage(A, AccessKind::StorageRead)
            };
            let catalog = FeatureCatalog::new(
                table(vec![
                    FeatureDefinition::subsystem(
                        ROOT,
                        RenderDomain::Hair,
                        ServiceSet::new(table(vec![SERVICE])),
                        table(vec![variant(
                            700,
                            0,
                            vec![
                                pass(
                                    2,
                                    QueueClass::Compute,
                                    vec![produced(1, AccessKind::StorageWrite)],
                                ),
                                pass(
                                    3,
                                    QueueClass::Compute,
                                    vec![
                                        produced(1, AccessKind::StorageRead),
                                        history,
                                        usage(SHARED, AccessKind::StorageRead),
                                        imported_role,
                                        history,
                                    ],
                                ),
                                pass(4, QueueClass::Compute, vec![imported_role]),
                                pass(1, QueueClass::Compute, vec![initial]),
                            ],
                        )]),
                    ),
                    FeatureDefinition::service(
                        PROVIDER,
                        table(vec![ServiceOutput {
                            service: SERVICE,
                            resources: table(vec![SHARED]),
                        }]),
                        ServiceSet::EMPTY,
                        table(vec![variant(
                            701,
                            0,
                            vec![pass(
                                10,
                                QueueClass::Compute,
                                vec![usage(SHARED, AccessKind::StorageWrite)],
                            )],
                        )]),
                    ),
                ]),
                table(vec![A, SHARED]),
            )
            .unwrap();
            let scopes = [
                (1, 0, ROLE),
                (2, 0, ROLE),
                (3, 0, ROLE),
                (3, 1, ROLE),
                (3, 3, OTHER_ROLE),
                (3, 4, OTHER_ROLE),
                (4, 0, OTHER_ROLE),
            ]
            .into_iter()
            .map(|(pass, ordinal, role)| ObjectResourceAccess {
                pass: PassKey(pass),
                ordinal,
                role,
                layout: ObjectLayoutPolicy::Sized { max_bytes: 256 },
            })
            .collect();
            let template = buffer(A, ResourceLifetime::Persistent);
            let mut items = Vec::new();
            let mut bindings = Vec::new();
            for (index, identity) in identities().into_iter().enumerate() {
                let current = ResourceLayout::Buffer {
                    bytes: 64 + index as u64 * 16,
                    usage: match template.layout {
                        ResourceLayout::Buffer { usage, .. } => usage,
                        _ => unreachable!(),
                    },
                };
                items.push(FrameRenderItem {
                    object_id: identity.object(),
                    domain: identity.domain(),
                    history_key: identity.history_key(),
                    motion_source: MotionSource::Camera,
                    payload: DomainPayload::Resource(ResourcePayload {
                        resource: A,
                        lifetime: template.lifetime,
                        size: current.size().unwrap(),
                        alignment: template.alignment,
                        layout: current,
                        role: identity.role(),
                    }),
                });
                for (frame, bytes, offset, version) in [
                    (
                        FRAME,
                        64 + index as u64 * 16,
                        index as u64 * 512,
                        10 + index as u32,
                    ),
                    (
                        FRAME - 1,
                        32 + index as u64 * 16,
                        index as u64 * 512 + 256,
                        20 + index as u32,
                    ),
                ] {
                    bindings.push(ObjectResourceBinding {
                        identity,
                        binding: FrameResourceBinding {
                            key: A,
                            frame,
                            version,
                            alignment: template.alignment,
                            physical: PhysicalResource::Buffer {
                                id: physical,
                                offset,
                            },
                            layout: ResourceLayout::Buffer {
                                bytes,
                                usage: match current {
                                    ResourceLayout::Buffer { usage, .. } => usage,
                                    _ => unreachable!(),
                                },
                            },
                        },
                    });
                }
            }
            Self {
                catalog,
                templates: vec![template, buffer(SHARED, ResourceLifetime::Transient)],
                scopes,
                items,
                bindings,
            }
        }

        fn resources(&self) -> ResourceTemplates<'_> {
            ResourceTemplates::new_with_object_accesses(
                &self.catalog,
                &self.templates,
                &self.scopes,
            )
            .unwrap()
        }

        fn with_role_writers(mut self, read: AccessKind, layout: ObjectLayoutPolicy) -> Self {
            (self.catalog, self.scopes) = Self::role_writers(read, layout);
            self
        }

        fn role_writers(
            read: AccessKind,
            layout: ObjectLayoutPolicy,
        ) -> (FeatureCatalog, Vec<ObjectResourceAccess>) {
            let initial = ResourceUse {
                input: ResourceInput::Initial,
                ..usage(A, AccessKind::StorageWrite)
            };
            let produced = |writer, access| ResourceUse {
                input: ResourceInput::ProducedBy(PassKey(writer)),
                ..usage(A, access)
            };
            let history = ResourceUse {
                input: ResourceInput::Initial,
                frame_lag: Some(FrameLag::new(1).unwrap()),
                ..usage(A, read)
            };
            let scopes: Vec<_> = [
                (1, 0, ROLE),
                (1, 1, OTHER_ROLE),
                (2, 0, ROLE),
                (4, 0, OTHER_ROLE),
                (3, 0, ROLE),
                (3, 1, OTHER_ROLE),
                (3, 2, ROLE),
                (3, 3, OTHER_ROLE),
            ]
            .into_iter()
            .map(|(pass, ordinal, role)| ObjectResourceAccess {
                pass: PassKey(pass),
                ordinal,
                role,
                layout,
            })
            .collect();
            let definitions = table(vec![
                FeatureDefinition::subsystem(
                    ROOT,
                    RenderDomain::Hair,
                    ServiceSet::new(table(vec![SERVICE])),
                    table(vec![variant(
                        700,
                        0,
                        vec![
                            pass(
                                2,
                                QueueClass::Compute,
                                vec![produced(1, AccessKind::StorageWrite)],
                            ),
                            pass(
                                3,
                                QueueClass::Compute,
                                vec![
                                    produced(2, read),
                                    produced(4, read),
                                    history,
                                    history,
                                    usage(SHARED, AccessKind::StorageRead),
                                ],
                            ),
                            pass(
                                4,
                                QueueClass::Compute,
                                vec![produced(1, AccessKind::StorageWrite)],
                            ),
                            pass(1, QueueClass::Compute, vec![initial, initial]),
                        ],
                    )]),
                ),
                FeatureDefinition::service(
                    PROVIDER,
                    table(vec![ServiceOutput {
                        service: SERVICE,
                        resources: table(vec![SHARED]),
                    }]),
                    ServiceSet::EMPTY,
                    table(vec![variant(
                        701,
                        0,
                        vec![pass(
                            10,
                            QueueClass::Compute,
                            vec![usage(SHARED, AccessKind::StorageWrite)],
                        )],
                    )]),
                ),
            ]);
            let catalog = FeatureCatalog::new_with_object_accesses(
                definitions,
                table(vec![A, SHARED]),
                &scopes,
            )
            .unwrap();
            (catalog, scopes)
        }
        fn compile(&self) -> Result<CompiledFeatureGraph, FabricCompileError> {
            self.resources().compile_with_instances(
                &select(&self.catalog),
                FRAME,
                &[],
                &self.bindings,
                &self.items,
                GraphLimits::default(),
            )
        }
        fn imports(&self, graph: &CompiledFeatureGraph) -> Vec<ObjectRuntimeImport> {
            self.bindings
                .iter()
                .map(|object| {
                    let PhysicalResource::Buffer { id, .. } = object.binding.physical else {
                        unreachable!()
                    };
                    ObjectRuntimeImport {
                        address: object.address(),
                        binding: object.binding,
                        imported: RuntimeImport::Buffer {
                            resource: graph.resource_id_at(object.address()).unwrap(),
                            imported: ImportedBuffer {
                                buffer: id,
                                entry_state: BufferState::initial(),
                            },
                        },
                    }
                })
                .collect()
        }
    }

    fn fixture() -> Fixture {
        Fixture::new(BufferId::from_parts(70, 1))
    }

    fn recorders() -> Vec<RuntimePassBinding> {
        PASSES
            .into_iter()
            .map(|key| {
                RuntimePassBinding::new(key, |_| {
                    panic!("prepare must never execute a domain-batch recorder")
                })
            })
            .collect()
    }

    fn prepare_error(fixture: &Fixture, imports: &[ObjectRuntimeImport]) -> FabricRuntimeError {
        fixture
            .compile()
            .unwrap()
            .prepare_runtime_with_object_imports(Extent3d::new_2d(1, 1), &[], imports, recorders())
            .err()
            .expect("invalid imports must fail before recording")
    }

    #[test]
    fn explicit_role_writers_compile_and_record_independent_exact_ssa_chains() {
        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance.enumerate_adapters().pop().unwrap();
        let device = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap()
            .0;
        let backing = device.create_buffer(&BufferDescriptor::new(
            3072,
            BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST,
        ));
        let fixture = Fixture::new(backing).with_role_writers(
            AccessKind::StorageRead,
            ObjectLayoutPolicy::Sized { max_bytes: 256 },
        );
        let graph = fixture.compile().unwrap();
        assert_eq!(graph.resources().len(), 13);
        assert_eq!(graph.passes().len(), PASSES.len());
        assert_eq!(select(&fixture.catalog).services().count(), 1);
        let reader = graph
            .passes()
            .iter()
            .find(|p| p.key() == PassKey(3))
            .unwrap();
        for (index, identity) in identities().into_iter().enumerate() {
            let ordinal = usize::from(identity.role() == OTHER_ROLE);
            let writer = if ordinal == 0 { PassKey(2) } else { PassKey(4) };
            let current = reader.access(Some(identity), ordinal).unwrap();
            assert_eq!(current.input_ssa_version, 2);
            assert_eq!(current.input_producer, Some(writer));
            assert_eq!(current.input_content_version, 12 + index as u32);
            assert_eq!(
                current.address(),
                ResourceAddress::object(A, FRAME, identity)
            );
            let history = reader.access(Some(identity), ordinal + 2).unwrap();
            assert_eq!(
                history.address(),
                ResourceAddress::object(A, FRAME - 1, identity)
            );
            assert_eq!(history.ssa_version, 0);
            assert_eq!(history.producer, None);
            assert_eq!(history.content_version, 20 + index as u32);
            let mut versions: Vec<_> = graph
                .versions()
                .iter()
                .filter(|v| v.identity == Some(identity))
                .collect();
            versions.sort_by_key(|version| version.ssa_version);
            assert_eq!(versions.len(), 2);
            assert_eq!(
                (versions[0].writer, versions[0].ssa_version),
                (PassKey(1), 1)
            );
            assert_eq!((versions[1].writer, versions[1].ssa_version), (writer, 2));
        }
        let calls = Arc::new(Mutex::new(Vec::new()));
        let callbacks = PASSES
            .into_iter()
            .map(|key| {
                let calls = Arc::clone(&calls);
                let bindings = fixture.bindings.clone();
                RuntimePassBinding::new(key, move |ctx| {
                    assert_eq!(ctx.items().count(), 6);
                    if key == PassKey(10) {
                        assert_eq!(ctx.accesses().len(), 1);
                        assert!(ctx.buffer(0).is_some());
                    } else {
                        for object in &bindings {
                            let role = usize::from(object.identity.role() == OTHER_ROLE);
                            let current = object.binding.frame == FRAME;
                            let ordinal = match (key.0, current, role) {
                                (1, true, _) => Some(role),
                                (2, true, 0) | (4, true, 1) => Some(0),
                                (3, true, _) => Some(role),
                                (3, false, _) => Some(role + 2),
                                _ => None,
                            };
                            if let Some(ordinal) = ordinal {
                                let PhysicalResource::Buffer { id, offset } =
                                    object.binding.physical
                                else {
                                    unreachable!()
                                };
                                assert_eq!(
                                    ctx.object_buffer(object.identity, ordinal),
                                    Some((id, offset, object.binding.layout.size().unwrap()))
                                );
                                assert_eq!(
                                    ctx.object_access(object.identity, ordinal)
                                        .unwrap()
                                        .bound
                                        .address(),
                                    object.address()
                                );
                                assert!(ctx.shared_access(ordinal).is_none());
                            }
                        }
                        if key == PassKey(3) {
                            assert_eq!(ctx.accesses().len(), 13);
                            assert!(ctx.buffer(4).is_some());
                        }
                    }
                    calls.lock().unwrap().push(key);
                })
            })
            .collect();
        let runtime = graph
            .clone()
            .prepare_runtime_with_object_imports(
                Extent3d::new_2d(1, 1),
                &[],
                &fixture.imports(&graph),
                callbacks,
            )
            .unwrap();
        assert!(calls.lock().unwrap().is_empty());
        let recorded = runtime.record(&device);
        let mut observed = calls.lock().unwrap().clone();
        observed.sort();
        assert_eq!(observed, PASSES);
        assert_eq!(device.live_resource_count(), 2); // backing plus one shared service
        recorded.release(&device);
        assert_eq!(device.live_resource_count(), 1);
        device.destroy_buffer(backing);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn explicit_role_catalog_rejects_same_role_writers_and_shared_reinterpretation() {
        let fixture = fixture().with_role_writers(
            AccessKind::StorageRead,
            ObjectLayoutPolicy::Sized { max_bytes: 256 },
        );
        let definitions = table(fixture.catalog.features().copied().collect());
        let keys = table(vec![A, SHARED]);
        assert!(matches!(
            FeatureCatalog::new(definitions, keys),
            Err(CatalogError::AmbiguousWriter { resource: A, .. })
        ));
        let mut scopes = fixture.scopes.clone();
        for scope in &mut scopes {
            scope.role = ROLE;
        }
        assert!(matches!(
            FeatureCatalog::new_with_object_accesses(definitions, keys, &scopes),
            Err(CatalogError::AmbiguousWriter { resource: A, .. })
        ));
        // Same-role sibling initial writes remain ambiguous even in distinct passes.
        let independent = table(vec![FeatureDefinition::subsystem(
            ROOT,
            RenderDomain::Hair,
            ServiceSet::EMPTY,
            table(vec![variant(
                700,
                0,
                vec![1, 2]
                    .into_iter()
                    .map(|id| {
                        pass(
                            id,
                            QueueClass::Compute,
                            vec![ResourceUse {
                                input: ResourceInput::Initial,
                                ..usage(A, AccessKind::StorageWrite)
                            }],
                        )
                    })
                    .collect(),
            )]),
        )]);
        let scopes: Vec<_> = [1, 2]
            .into_iter()
            .map(|id| ObjectResourceAccess {
                pass: PassKey(id),
                ordinal: 0,
                role: ROLE,
                layout: ObjectLayoutPolicy::Exact,
            })
            .collect();
        assert!(matches!(
            FeatureCatalog::new_with_object_accesses(independent, table(vec![A]), &scopes),
            Err(CatalogError::AmbiguousWriter { resource: A, .. })
        ));
        let mut scopes = fixture.scopes.clone();
        scopes.pop();
        assert_eq!(
            FeatureCatalog::new_with_object_accesses(definitions, keys, &scopes).err(),
            Some(CatalogError::InconsistentObjectResource(A))
        );
        scopes = fixture.scopes.clone();
        scopes.push(ObjectResourceAccess {
            pass: PassKey(10),
            ordinal: 0,
            ..scopes[0]
        });
        assert_eq!(
            FeatureCatalog::new_with_object_accesses(definitions, keys, &scopes).err(),
            Some(CatalogError::InvalidObjectAccess {
                pass: PassKey(10),
                ordinal: 0
            })
        );
        for invalid in [
            ObjectResourceAccess {
                pass: PassKey(99),
                ..fixture.scopes[0]
            },
            ObjectResourceAccess {
                ordinal: 99,
                ..fixture.scopes[0]
            },
            ObjectResourceAccess {
                layout: ObjectLayoutPolicy::Sized { max_bytes: 0 },
                ..fixture.scopes[0]
            },
        ] {
            scopes = fixture.scopes.clone();
            scopes[0] = invalid;
            assert_eq!(
                FeatureCatalog::new_with_object_accesses(definitions, keys, &scopes).err(),
                Some(CatalogError::InvalidObjectAccess {
                    pass: invalid.pass,
                    ordinal: invalid.ordinal
                })
            );
        }
        scopes = fixture.scopes.clone();
        scopes.push(scopes[0]);
        assert_eq!(
            FeatureCatalog::new_with_object_accesses(definitions, keys, &scopes).err(),
            Some(CatalogError::InvalidObjectAccess {
                pass: PassKey(1),
                ordinal: 0
            })
        );
        scopes = fixture.scopes.clone();
        scopes[1].layout = ObjectLayoutPolicy::Exact;
        assert_eq!(
            FeatureCatalog::new_with_object_accesses(definitions, keys, &scopes).err(),
            Some(CatalogError::InconsistentObjectResource(A))
        );
    }

    #[test]
    fn role_writer_proof_cannot_be_erased_relabelled_or_relaxed_at_lowering() {
        let fixture = fixture().with_role_writers(
            AccessKind::StorageRead,
            ObjectLayoutPolicy::Sized { max_bytes: 256 },
        );
        let error = |scopes: &[ObjectResourceAccess]| {
            ResourceTemplates::new_with_object_accesses(
                &fixture.catalog,
                &fixture.templates,
                scopes,
            )
            .err()
        };
        assert_eq!(
            error(&[]),
            Some(FabricCompileError::InvalidObjectAccess {
                pass: PassKey(1),
                ordinal: 0,
            })
        );
        for scopes in [
            fixture.scopes[1..].to_vec(),
            {
                let mut s = fixture.scopes.clone();
                s[0].role = OTHER_ROLE;
                s
            },
            {
                let mut s = fixture.scopes.clone();
                s[0].layout = ObjectLayoutPolicy::Exact;
                s
            },
            {
                let mut s = fixture.scopes.clone();
                s[0].layout = ObjectLayoutPolicy::Sized { max_bytes: 512 };
                s
            },
            {
                let mut s = fixture.scopes.clone();
                s.push(s[0]);
                s
            },
        ] {
            assert_eq!(
                error(&scopes),
                Some(FabricCompileError::InvalidObjectAccess {
                    pass: PassKey(1),
                    ordinal: 0,
                })
            );
        }
        let mut reversed = fixture.scopes.clone();
        reversed.reverse();
        assert!(ResourceTemplates::new_with_object_accesses(
            &fixture.catalog,
            &fixture.templates,
            &reversed,
        )
        .is_ok());
        // A producer of the right key but wrong role cannot authorize a read.
        let mut definitions: Vec<_> = fixture.catalog.features().copied().collect();
        let root = definitions.iter_mut().find(|f| f.id == ROOT).unwrap();
        let mut passes = root.variants[0].passes.to_vec();
        let reader = passes.iter_mut().find(|p| p.id == PassKey(3)).unwrap();
        let mut accesses = reader.accesses.to_vec();
        accesses[1].input = ResourceInput::ProducedBy(PassKey(2));
        reader.accesses = table(accesses);
        root.variants = table(vec![variant(700, 0, passes)]);
        let catalog = FeatureCatalog::new_with_object_accesses(
            table(definitions),
            table(vec![A, SHARED]),
            &fixture.scopes,
        )
        .unwrap();
        let resources = ResourceTemplates::new_with_object_accesses(
            &catalog,
            &fixture.templates,
            &fixture.scopes,
        )
        .unwrap();
        assert_eq!(
            resources
                .compile_with_instances(
                    &select(&catalog),
                    FRAME,
                    &[],
                    &fixture.bindings,
                    &fixture.items,
                    GraphLimits::default(),
                )
                .err(),
            Some(FabricCompileError::InvalidProducer {
                pass: PassKey(3),
                resource: A,
                producer: PassKey(2),
            })
        );
        let mut overlapping = fixture.bindings.clone();
        overlapping[2].binding.physical = overlapping[0].binding.physical;
        assert!(matches!(
            fixture.resources().compile_with_instances(
                &select(&fixture.catalog),
                FRAME,
                &[],
                &overlapping,
                &fixture.items,
                GraphLimits::default(),
            ),
            Err(FabricCompileError::OverlappingBindings { .. })
        ));
    }

    fn texture_fixture(device: &impl RenderDevice) -> (Fixture, Vec<(TextureId, TextureViewId)>) {
        let (catalog, scopes) = Fixture::role_writers(
            AccessKind::SampledRead,
            ObjectLayoutPolicy::Sized { max_bytes: 8192 },
        );
        let usage = TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
        let template = ResourceTemplate {
            key: A,
            lifetime: ResourceLifetime::Persistent,
            alignment: 16,
            layout: ResourceLayout::Texture {
                extent: [16, 16, 1],
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 1,
                samples: 1,
                usage,
            },
        };
        let mut fixture = Fixture {
            catalog,
            scopes,
            templates: vec![template, buffer(SHARED, ResourceLifetime::Transient)],
            items: Vec::new(),
            bindings: Vec::new(),
        };
        let mut textures = Vec::new();
        for (index, identity) in identities().into_iter().enumerate() {
            for history in [false, true] {
                let width = (if history { 4 } else { 8 }) << (index % 2);
                let height = if history { 4 } else { 8 };
                let layers = 1 + index as u32 % 2;
                let mips = 1 + (index as u32 + u32::from(history)) % 3;
                let layout = ResourceLayout::Texture {
                    extent: [width, height, layers],
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    mip_levels: mips,
                    samples: 1,
                    usage,
                };
                let mut descriptor = TextureDescriptor::new_2d(
                    Extent3d {
                        width,
                        height,
                        depth_or_array_layers: layers,
                    },
                    TextureFormat::Rgba8Unorm,
                    usage,
                );
                descriptor.mip_level_count = mips;
                let texture = device.create_texture(&descriptor);
                let view = device.create_texture_view(
                    texture,
                    &TextureViewDescriptor {
                        dimension: if layers == 1 {
                            TextureViewDimension::D2
                        } else {
                            TextureViewDimension::D2Array
                        },
                        mip_level_count: Some(mips),
                        array_layer_count: Some(layers),
                        ..Default::default()
                    },
                );
                textures.push((texture, view));
                fixture.bindings.push(ObjectResourceBinding {
                    identity,
                    binding: FrameResourceBinding {
                        key: A,
                        frame: FRAME - u64::from(history),
                        version: (if history { 20 } else { 10 }) + index as u32,
                        physical: PhysicalResource::Texture(texture),
                        layout,
                        alignment: 16,
                    },
                });
                if !history {
                    fixture.items.push(FrameRenderItem {
                        object_id: identity.object(),
                        domain: identity.domain(),
                        history_key: identity.history_key(),
                        motion_source: MotionSource::Camera,
                        payload: DomainPayload::Resource(ResourcePayload {
                            resource: A,
                            lifetime: template.lifetime,
                            size: layout.size().unwrap(),
                            alignment: 16,
                            layout,
                            role: identity.role(),
                        }),
                    });
                }
            }
        }
        (fixture, textures)
    }

    fn texture_imports(
        fixture: &Fixture,
        graph: &CompiledFeatureGraph,
        textures: &[(TextureId, TextureViewId)],
    ) -> Vec<ObjectRuntimeImport> {
        fixture
            .bindings
            .iter()
            .zip(textures)
            .map(|(object, &(texture, view))| {
                let ResourceLayout::Texture {
                    extent, mip_levels, ..
                } = object.binding.layout
                else {
                    unreachable!()
                };
                assert_eq!(object.binding.physical, PhysicalResource::Texture(texture));
                ObjectRuntimeImport {
                    address: object.address(),
                    binding: object.binding,
                    imported: RuntimeImport::Texture {
                        resource: graph.resource_id_at(object.address()).unwrap(),
                        imported: ImportedTexture {
                            texture,
                            default_view: view,
                            entry_state: TextureState::undefined(),
                            exit_state: TextureState::undefined(),
                            mip_levels,
                            array_layers: extent[2],
                        },
                    },
                }
            })
            .collect()
    }

    #[test]
    fn object_texture_cpu_runtime_preserves_role_generation_history_extent_mips_and_views() {
        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance.enumerate_adapters().pop().unwrap();
        let device = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap()
            .0;
        let (fixture, textures) = texture_fixture(&device);
        let graph = fixture.compile().unwrap();
        let imports = texture_imports(&fixture, &graph, &textures);
        assert_eq!(graph.resources().len(), 13);
        assert_eq!(graph.passes().len(), 5);
        assert_eq!(
            graph.resources().iter().filter(|r| r.key == SHARED).count(),
            1
        );
        let mut unique = std::collections::BTreeSet::new();
        for object in &fixture.bindings {
            let id = graph.resource_id_at(object.address()).unwrap();
            assert!(unique.insert(id.0));
            let bound = &graph.resources()[id.0 as usize];
            assert_eq!(bound.address(), object.address());
            assert_eq!(bound.template.layout, object.binding.layout);
            assert_eq!(bound.external, Some(object.binding));
        }
        for pair in fixture.bindings.chunks_exact(2) {
            assert_eq!(pair[0].identity, pair[1].identity);
            assert_ne!(pair[0].binding.layout, pair[1].binding.layout);
            assert_ne!(pair[0].binding.physical, pair[1].binding.physical);
        }
        let calls = Arc::new(Mutex::new(Vec::new()));
        let callbacks = PASSES
            .into_iter()
            .map(|key| {
                let imports = imports.clone();
                let scopes = fixture.scopes.clone();
                let calls = Arc::clone(&calls);
                RuntimePassBinding::new(key, move |ctx| {
                    assert_eq!(ctx.items().count(), 6);
                    for scope in scopes.iter().filter(|scope| scope.pass == key) {
                        let history = key == PassKey(3) && scope.ordinal >= 2;
                        for object in imports.iter().filter(|object| {
                            object.address.frame == FRAME - u64::from(history)
                                && object.address.identity.unwrap().role() == scope.role
                        }) {
                            let identity = object.address.identity.unwrap();
                            let RuntimeImport::Texture { imported, .. } = object.imported else {
                                unreachable!()
                            };
                            assert_eq!(
                                ctx.object_texture(identity, scope.ordinal),
                                Some(imported.texture)
                            );
                            assert_eq!(
                                ctx.object_texture_view(identity, scope.ordinal),
                                Some(imported.default_view)
                            );
                            assert!(ctx.object_buffer(identity, scope.ordinal).is_none());
                            assert!(ctx.texture(scope.ordinal).is_none());
                            let bound = ctx.object_access(identity, scope.ordinal).unwrap().bound;
                            assert_eq!(bound.address(), object.address);
                            if history {
                                assert_eq!(bound.input_ssa_version, 0);
                                assert_eq!(bound.input_producer, None);
                                assert_eq!(bound.input_content_version, object.binding.version);
                            } else if key == PassKey(3) {
                                assert_eq!(bound.input_ssa_version, 2);
                                assert_eq!(bound.input_content_version, object.binding.version + 2);
                                assert_eq!(
                                    bound.input_producer,
                                    Some(if scope.role == ROLE {
                                        PassKey(2)
                                    } else {
                                        PassKey(4)
                                    })
                                );
                            }
                            let other_role = ObjectResourceIdentity::new(
                                identity.object(),
                                identity.history_key(),
                                if scope.role == ROLE { OTHER_ROLE } else { ROLE },
                            );
                            assert!(ctx.object_texture(other_role, scope.ordinal).is_none());
                        }
                    }
                    if key == PassKey(3) {
                        assert!(ctx.buffer(4).is_some());
                    }
                    if key == PassKey(10) {
                        assert!(ctx.buffer(0).is_some());
                    }
                    calls.lock().unwrap().push(key);
                })
            })
            .collect();
        let external_count = textures.len() * 2;
        assert_eq!(device.live_resource_count(), external_count);
        let runtime = graph
            .prepare_runtime_with_object_imports(Extent3d::new_2d(32, 32), &[], &imports, callbacks)
            .unwrap();
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(device.live_resource_count(), external_count);
        let recorded = runtime.record(&device);
        let mut observed = calls.lock().unwrap().clone();
        observed.sort();
        assert_eq!(observed, PASSES);
        assert_eq!(device.live_resource_count(), external_count + 1);
        recorded.release(&device);
        assert_eq!(device.live_resource_count(), external_count);
        for (texture, view) in textures {
            device.destroy_texture_view(view);
            device.destroy_texture(texture);
        }
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn object_texture_imports_reject_missing_duplicate_wrong_identity_extent_mips_and_layers() {
        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance.enumerate_adapters().pop().unwrap();
        let device = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap()
            .0;
        let (fixture, textures) = texture_fixture(&device);
        let graph = fixture.compile().unwrap();
        let imports = texture_imports(&fixture, &graph, &textures);
        for index in [0, 1] {
            // independently authenticate current and history
            let original = imports[index];
            let mut changed = imports.clone();
            changed.remove(index);
            assert_eq!(
                prepare_error(&fixture, &changed),
                FabricRuntimeError::MissingImport(original.imported.resource())
            );
            changed = imports.clone();
            changed.push(original);
            assert_eq!(
                prepare_error(&fixture, &changed),
                FabricRuntimeError::DuplicateImport(original.imported.resource())
            );
            let ResourceLayout::Texture {
                extent,
                dimension,
                format,
                mip_levels,
                samples,
                usage,
            } = original.binding.layout
            else {
                unreachable!()
            };
            for layout in [
                ResourceLayout::Texture {
                    extent: [extent[0] * 2, extent[1], extent[2]],
                    dimension,
                    format,
                    mip_levels,
                    samples,
                    usage,
                },
                ResourceLayout::Texture {
                    extent,
                    dimension,
                    format,
                    mip_levels: mip_levels + 1,
                    samples,
                    usage,
                },
            ] {
                changed = imports.clone();
                changed[index].binding.layout = layout;
                assert_eq!(
                    prepare_error(&fixture, &changed),
                    FabricRuntimeError::InvalidObjectImport(original.address)
                );
            }
            let RuntimeImport::Texture { resource, imported } = original.imported else {
                unreachable!()
            };
            for imported in [
                ImportedTexture {
                    mip_levels: imported.mip_levels + 1,
                    ..imported
                },
                ImportedTexture {
                    array_layers: imported.array_layers + 1,
                    ..imported
                },
                ImportedTexture {
                    texture: textures[(index + 2) % textures.len()].0,
                    ..imported
                },
            ] {
                changed = imports.clone();
                changed[index].imported = RuntimeImport::Texture { resource, imported };
                assert_eq!(
                    prepare_error(&fixture, &changed),
                    FabricRuntimeError::InvalidImport(resource)
                );
            }
            let identity = original.address.identity.unwrap();
            for address in [
                ResourceAddress::shared(A, original.address.frame),
                ResourceAddress::object(A, FRAME + 1, identity),
                ResourceAddress::object(
                    A,
                    original.address.frame,
                    ObjectResourceIdentity::new(
                        identity.object(),
                        identity.history_key(),
                        OTHER_ROLE,
                    ),
                ),
                ResourceAddress::object(
                    A,
                    original.address.frame,
                    ObjectResourceIdentity::new(identity.object(), 99, identity.role()),
                ),
                ResourceAddress::object(
                    A,
                    original.address.frame,
                    ObjectResourceIdentity::new(
                        RenderObjectId::Hair(DomainObjectHandle::new(4, 99)),
                        identity.history_key(),
                        identity.role(),
                    ),
                ),
                ResourceAddress::object(
                    A,
                    original.address.frame,
                    ObjectResourceIdentity::new(
                        RenderObjectId::Particle(DomainObjectHandle::new(4, 1)),
                        identity.history_key(),
                        identity.role(),
                    ),
                ),
            ] {
                changed = imports.clone();
                changed[index].address = address;
                assert_eq!(
                    prepare_error(&fixture, &changed),
                    FabricRuntimeError::InvalidObjectImport(address)
                );
            }
        }
        for (texture, view) in textures {
            device.destroy_texture_view(view);
            device.destroy_texture(texture);
        }
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn object_texture_sized_layout_keeps_kind_format_usage_dimension_samples_and_alias_strict() {
        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance.enumerate_adapters().pop().unwrap();
        let device = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap()
            .0;
        let (fixture, textures) = texture_fixture(&device);
        for index in [0, 1] {
            let ResourceLayout::Texture {
                extent,
                dimension,
                format,
                mip_levels,
                samples,
                usage,
            } = fixture.bindings[index].binding.layout
            else {
                unreachable!()
            };
            for layout in [
                ResourceLayout::Texture {
                    extent,
                    dimension,
                    format: TextureFormat::Rgba16Float,
                    mip_levels,
                    samples,
                    usage,
                },
                ResourceLayout::Texture {
                    extent,
                    dimension: TextureDimension::D3,
                    format,
                    mip_levels,
                    samples,
                    usage,
                },
                ResourceLayout::Texture {
                    extent,
                    dimension,
                    format,
                    mip_levels,
                    samples,
                    usage: TextureUsages::TEXTURE_BINDING,
                },
                ResourceLayout::Texture {
                    extent,
                    dimension,
                    format,
                    mip_levels: 1,
                    samples: 4,
                    usage,
                },
                ResourceLayout::Texture {
                    extent,
                    dimension,
                    format,
                    mip_levels: 32,
                    samples,
                    usage,
                },
                ResourceLayout::Texture {
                    extent: [0, extent[1], extent[2]],
                    dimension,
                    format,
                    mip_levels,
                    samples,
                    usage,
                },
                ResourceLayout::Texture {
                    extent: [1024, 1024, 1],
                    dimension,
                    format,
                    mip_levels,
                    samples,
                    usage,
                },
                ResourceLayout::Buffer {
                    bytes: 64,
                    usage: BufferUsages::STORAGE,
                },
            ] {
                let mut objects = fixture.bindings.clone();
                objects[index].binding.layout = layout;
                assert_eq!(
                    fixture
                        .resources()
                        .compile_with_instances(
                            &select(&fixture.catalog),
                            FRAME,
                            &[],
                            &objects,
                            &fixture.items,
                            GraphLimits::default(),
                        )
                        .err(),
                    Some(FabricCompileError::InvalidObjectBinding(
                        objects[index].address()
                    ))
                );
            }
        }
        let mut objects = fixture.bindings.clone();
        objects[2].binding.physical = objects[0].binding.physical;
        assert!(matches!(
            fixture.resources().compile_with_instances(
                &select(&fixture.catalog),
                FRAME,
                &[],
                &objects,
                &fixture.items,
                GraphLimits::default(),
            ),
            Err(FabricCompileError::OverlappingBindings { .. })
        ));
        for (texture, view) in textures {
            device.destroy_texture_view(view);
            device.destroy_texture(texture);
        }
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn contains_item_indexes_identity_but_compares_entire_resource_payload_and_motion() {
        let fixture = fixture();
        let graph = fixture.compile().unwrap();
        for item in &fixture.items {
            assert!(graph.contains_item(item));
        }
        let original = fixture.items[0];
        let DomainPayload::Resource(payload) = original.payload else {
            unreachable!()
        };
        for changed in [
            ResourcePayload {
                resource: SHARED,
                ..payload
            },
            ResourcePayload {
                lifetime: ResourceLifetime::Imported,
                ..payload
            },
            ResourcePayload {
                size: payload.size + 1,
                ..payload
            },
            ResourcePayload {
                alignment: payload.alignment * 2,
                ..payload
            },
            ResourcePayload {
                layout: ResourceLayout::Buffer {
                    bytes: 64,
                    usage: BufferUsages::UNIFORM,
                },
                ..payload
            },
            ResourcePayload {
                role: PayloadRole(99),
                ..payload
            },
        ] {
            assert!(!graph.contains_item(&FrameRenderItem {
                payload: DomainPayload::Resource(changed),
                ..original
            }));
        }
        for changed in [
            FrameRenderItem {
                domain: RenderDomain::Particle,
                ..original
            },
            FrameRenderItem {
                history_key: 99,
                ..original
            },
            FrameRenderItem {
                object_id: RenderObjectId::Hair(DomainObjectHandle::new(4, 99)),
                ..original
            },
            FrameRenderItem {
                motion_source: MotionSource::Rigid,
                ..original
            },
            FrameRenderItem {
                payload: DomainPayload::Value(ValuePayload {
                    value: PayloadValue::Scalar(64.0),
                    role: ROLE,
                }),
                ..original
            },
        ] {
            assert!(!graph.contains_item(&changed));
        }
        let clone = graph.clone();
        let mut reversed = fixture;
        reversed.items.reverse();
        let reordered = reversed.compile().unwrap();
        assert_eq!(reordered.items(), reversed.items);
        for item in &reversed.items {
            assert!(clone.contains_item(item));
            assert!(reordered.contains_item(item));
        }
    }

    #[test]
    fn contains_item_value_floats_match_slice_equality_without_ordering_payloads() {
        let mut fixture = fixture();
        let base = fixture.items[0];
        for (role, value) in [
            PayloadValue::Scalar(0.0),
            PayloadValue::Vector([1.0, 2.0, 3.0, 4.0]),
            PayloadValue::Bounds {
                min: [-1.0; 3],
                max: [1.0; 3],
            },
            PayloadValue::Scalar(f32::NAN),
        ]
        .into_iter()
        .enumerate()
        {
            fixture.items.push(FrameRenderItem {
                payload: DomainPayload::Value(ValuePayload {
                    value,
                    role: PayloadRole(90 + role as u16),
                }),
                ..base
            });
        }
        fixture.items.push(FrameRenderItem {
            object_id: RenderObjectId::Particle(DomainObjectHandle::new(4, 1)),
            domain: RenderDomain::Particle,
            payload: DomainPayload::Value(ValuePayload {
                value: PayloadValue::Scalar(1.0),
                role: ROLE,
            }),
            ..base
        });
        let graph = fixture.compile().unwrap();
        for item in &fixture.items {
            assert_eq!(graph.contains_item(item), graph.items().contains(item));
        }
        let zero = fixture.items[6];
        let DomainPayload::Value(value) = zero.payload else {
            unreachable!()
        };
        for scalar in [-0.0, 1.0, f32::NAN] {
            let item = FrameRenderItem {
                payload: DomainPayload::Value(ValuePayload {
                    value: PayloadValue::Scalar(scalar),
                    ..value
                }),
                ..zero
            };
            assert_eq!(graph.contains_item(&item), graph.items().contains(&item));
        }
        assert!(!graph.contains_item(&FrameRenderItem {
            motion_source: MotionSource::Rigid,
            ..zero
        }));
        let mut empty = fixture;
        empty.items.clear();
        empty.bindings.clear();
        assert!(!empty.compile().unwrap().contains_item(&base));
    }

    #[test]
    fn overlap_index_preserves_first_prior_binding_and_validation_precedence() {
        let mut fixture = fixture();
        let physical = |offset| PhysicalResource::Buffer {
            id: BufferId::from_parts(70, 1),
            offset,
        };
        // History is admitted first even though its offset is higher than the
        // current binding. A broad collision must still diagnose history first.
        fixture.bindings[1].binding.physical = physical(128);
        fixture.bindings[2].binding.physical = physical(16);
        let ResourceLayout::Buffer { usage, .. } = fixture.bindings[2].binding.layout else {
            unreachable!()
        };
        fixture.bindings[2].binding.layout = ResourceLayout::Buffer { bytes: 144, usage };
        let DomainPayload::Resource(mut payload) = fixture.items[1].payload else {
            unreachable!()
        };
        payload.size = 144;
        payload.layout = fixture.bindings[2].binding.layout;
        fixture.items[1].payload = DomainPayload::Resource(payload);
        assert_eq!(
            fixture.compile().err(),
            Some(FabricCompileError::OverlappingBindings {
                first: A,
                first_frame: FRAME - 1,
                second: A,
                second_frame: FRAME,
            })
        );
        let address = fixture.bindings[2].address();
        fixture.bindings[2].binding.alignment = 32;
        assert_eq!(
            fixture.compile().err(),
            Some(FabricCompileError::InvalidObjectBinding(address))
        );
        fixture.bindings[2].binding.alignment = 16;
        fixture.bindings[2].binding.version = u32::MAX;
        let limits = GraphLimits {
            versions: 100,
            ..GraphLimits::default()
        };
        assert_eq!(
            fixture
                .resources()
                .compile_with_instances(
                    &select(&fixture.catalog),
                    FRAME,
                    &[],
                    &fixture.bindings,
                    &fixture.items,
                    limits,
                )
                .err(),
            Some(FabricCompileError::Graph(CompileError::VersionOverflow {
                version: u32::MAX,
                limit: 100,
            }))
        );
    }

    #[test]
    fn many_same_key_instances_keep_deterministic_resources_and_indexed_membership() {
        let mut fixture = fixture();
        let source_items = fixture.items.clone();
        let source_bindings = fixture.bindings.clone();
        fixture.items.clear();
        fixture.bindings.clear();
        // Exercise a large identity table without wall-clock assertions. Two
        // roles and two temporal frames per object, still only five batch passes.
        for object in 0..256 {
            let object_id = RenderObjectId::Hair(DomainObjectHandle::new(object, 3));
            for role in 0..2 {
                fixture.items.push(FrameRenderItem {
                    object_id,
                    ..source_items[role]
                });
                for history in 0..2 {
                    let mut binding = source_bindings[role * 2 + history];
                    binding.identity =
                        ObjectResourceIdentity::new(object_id, 7, binding.identity.role());
                    binding.binding.physical = PhysicalResource::Buffer {
                        id: BufferId::from_parts(70, 1),
                        offset: u64::from(object) * 1024 + role as u64 * 256 + history as u64 * 128,
                    };
                    fixture.bindings.push(binding);
                }
            }
        }
        let graph = fixture.compile().unwrap();
        assert_eq!(graph.passes().len(), PASSES.len());
        assert_eq!(graph.resources().len(), 1025);
        assert_eq!(graph.items().len(), 512);
        let prepared = graph
            .clone()
            .prepare_runtime_with_object_imports(
                Extent3d::new_2d(1, 1),
                &[],
                &fixture.imports(&graph),
                recorders(),
            )
            .unwrap();
        assert_eq!(prepared.declaration().passes().len(), PASSES.len());
        for item in &fixture.items {
            assert!(graph.contains_item(item));
        }
        fixture.items.reverse();
        fixture.bindings.reverse();
        let reversed = fixture.compile().unwrap();
        assert_eq!(graph.versions(), reversed.versions());
        assert_eq!(graph.plan().order, reversed.plan().order);
        for resource in graph.resources() {
            assert_eq!(
                reversed.resource_id_at(resource.address()),
                Some(resource.id)
            );
        }
        for item in &fixture.items {
            assert!(reversed.contains_item(item));
        }
    }

    #[test]
    fn same_key_generation_history_and_role_keep_distinct_ssa_and_shared_service() {
        let fixture = fixture();
        let graph = fixture.compile().unwrap();
        assert_eq!(graph.passes().len(), PASSES.len());
        assert_eq!(graph.resources().len(), 13); // six identities, two frames, one service
        assert_eq!(select(&fixture.catalog).services().count(), 1);
        assert_eq!(
            graph.resources().iter().filter(|r| r.key == SHARED).count(),
            1
        );
        assert_eq!(
            graph
                .passes()
                .iter()
                .filter(|p| p.feature() == PROVIDER)
                .count(),
            1
        );
        assert_eq!(graph.resource_id(A, FRAME), None); // no first-object fallback
        assert!(graph.resource_id(SHARED, FRAME).is_some());
        let mut ids = std::collections::BTreeSet::new();
        let reader = graph
            .passes()
            .iter()
            .find(|p| p.key() == PassKey(3))
            .unwrap();
        assert_eq!(reader.accesses().len(), 13);
        assert!(reader.access(None, 0).is_none());
        assert_eq!(reader.access(None, 2).unwrap().declaration.key, SHARED);
        for (index, identity) in identities().into_iter().enumerate() {
            let (ordinal, writer) = if identity.role() == ROLE {
                (0, Some(PassKey(1)))
            } else {
                (3, None)
            };
            let current = reader.access(Some(identity), ordinal).unwrap();
            let history = reader.access(Some(identity), ordinal + 1).unwrap();
            assert_eq!(current.identity, Some(identity));
            assert_eq!(current.template_ordinal, ordinal);
            assert_eq!(
                current.address(),
                ResourceAddress::object(A, FRAME, identity)
            );
            assert_eq!(
                history.address(),
                ResourceAddress::object(A, FRAME - 1, identity)
            );
            let writes = u32::from(identity.role() == ROLE);
            assert_eq!(current.input_ssa_version, writes);
            assert_eq!(current.input_producer, writer);
            assert_eq!(current.input_content_version, 10 + writes + index as u32);
            assert_eq!(history.ssa_version, 0);
            assert_eq!(history.producer, None);
            assert_eq!(history.content_version, 20 + index as u32);
            for bound in [current, history] {
                assert_eq!(graph.resource_id_at(bound.address()), Some(bound.resource));
                assert_eq!(
                    graph.object_resource_id(A, bound.frame, identity),
                    Some(bound.resource)
                );
                assert!(ids.insert(bound.resource.0));
            }
            let versions: Vec<_> = graph
                .versions()
                .iter()
                .filter(|v| v.identity == Some(identity))
                .collect();
            assert_eq!(versions.len(), if identity.role() == ROLE { 2 } else { 0 });
            assert!(versions.iter().all(|v| v.address() == current.address()));
            if identity.role() == ROLE {
                let update = graph
                    .passes()
                    .iter()
                    .find(|p| p.key() == PassKey(2))
                    .unwrap()
                    .access(Some(identity), 0)
                    .unwrap();
                assert_eq!(update.input_producer, Some(PassKey(1)));
                assert_eq!(update.input_ssa_version, 1);
                assert_eq!(update.ssa_version, 2);
                assert_eq!(update.content_version, 12 + index as u32);
            }
            assert!(reader
                .access(Some(identity), if ordinal == 0 { 3 } else { 0 })
                .is_none());
        }
        assert_eq!(ids.len(), 12);
        let position = |key| {
            graph.plan().position_by_pass
                [graph.passes().iter().position(|p| p.key() == key).unwrap()]
        };
        assert!(position(PassKey(1)) < position(PassKey(3)));
        assert!(position(PassKey(3)) < position(PassKey(2))); // old-version WAR, not final-write rebinding
        let mut reversed = fixture;
        reversed.items.reverse();
        reversed.bindings.reverse();
        let other = reversed.compile().unwrap();
        assert_eq!(graph.versions(), other.versions());
        assert_eq!(graph.plan().order, other.plan().order);
        for resource in graph.resources() {
            assert_eq!(other.resource_id_at(resource.address()), Some(resource.id));
        }
    }

    #[test]
    fn cpu_runtime_records_one_batch_per_pass_with_exact_object_ranges_and_ordinals() {
        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance.enumerate_adapters().pop().unwrap();
        let device = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap()
            .0;
        let backing = device.create_buffer(&BufferDescriptor::new(
            3072,
            BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST,
        ));
        let fixture = Fixture::new(backing);
        let graph = fixture.compile().unwrap();
        let imports = fixture.imports(&graph);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let bindings = fixture.bindings.clone();
        let callbacks = PASSES
            .into_iter()
            .map(|key| {
                let calls = Arc::clone(&calls);
                let bindings = bindings.clone();
                RuntimePassBinding::new(key, move |ctx| {
                    assert_eq!(ctx.pass().key(), key);
                    assert_eq!(ctx.items().count(), 6);
                    if key == PassKey(10) {
                        assert_eq!(ctx.accesses().len(), 1);
                        assert!(ctx.buffer(0).is_some());
                        assert!(ctx.object_buffer(identities()[0], 0).is_none());
                    } else {
                        assert!(ctx.buffer(0).is_none());
                        let ordinal = |identity: ObjectResourceIdentity| match key.0 {
                            1 | 2 if identity.role() == ROLE => Some(0),
                            4 if identity.role() == OTHER_ROLE => Some(0),
                            3 if identity.role() == ROLE => Some(0),
                            3 => Some(3),
                            _ => None,
                        };
                        for object in &bindings {
                            let current = object.binding.frame == FRAME;
                            let index = ordinal(object.identity).and_then(|ordinal| {
                                if current {
                                    Some(ordinal)
                                } else if key == PassKey(3) {
                                    Some(ordinal + 1)
                                } else {
                                    None
                                }
                            });
                            if let Some(index) = index {
                                let PhysicalResource::Buffer { id, offset } =
                                    object.binding.physical
                                else {
                                    unreachable!()
                                };
                                assert_eq!(
                                    ctx.object_buffer(object.identity, index),
                                    Some((id, offset, object.binding.layout.size().unwrap()))
                                );
                                assert_eq!(
                                    ctx.object_access(object.identity, index)
                                        .unwrap()
                                        .bound
                                        .address(),
                                    object.address()
                                );
                                assert!(ctx.object_texture(object.identity, index).is_none());
                                assert!(ctx.object_texture_view(object.identity, index).is_none());
                            }
                        }
                        if key == PassKey(3) {
                            assert_eq!(ctx.accesses().len(), 13);
                            assert!(ctx.buffer(2).is_some()); // original ordinal, not expanded index
                            assert!(ctx.object_buffer(identities()[0], 2).is_none());
                        }
                        let wrong_generation = ObjectResourceIdentity::new(
                            RenderObjectId::Hair(DomainObjectHandle::new(4, 3)),
                            7,
                            ROLE,
                        );
                        assert!(ctx.object_buffer(wrong_generation, 0).is_none());
                    }
                    calls.lock().unwrap().push(key);
                })
            })
            .collect();
        let runtime = graph
            .prepare_runtime_with_object_imports(Extent3d::new_2d(1, 1), &[], &imports, callbacks)
            .unwrap();
        assert!(calls.lock().unwrap().is_empty());
        assert_eq!(device.live_resource_count(), 1); // prepare allocates nothing
        let recorded = runtime.record(&device);
        let mut observed = calls.lock().unwrap().clone();
        observed.sort();
        assert_eq!(observed, PASSES);
        assert_eq!(device.live_resource_count(), 2); // only the shared service is allocated
        recorded.release(&device);
        assert_eq!(device.live_resource_count(), 1); // imported backing remains externally owned
        device.destroy_buffer(backing);
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn frozen_device_active_instanced_entrypoint_preserves_selection_and_coverage() {
        let Fixture {
            catalog,
            templates,
            scopes,
            items,
            bindings,
        } = fixture();
        let roots = FrozenRoots::from_build_profile(&catalog, &[ROOT], &[]).unwrap();
        let device = DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::Low,
        )
        .unwrap();
        let selected = device.selection().clone();
        let graph = device
            .compile_active_instanced(
                &templates,
                &scopes,
                FRAME,
                &[],
                &bindings,
                &items,
                &PASSES,
                &[],
                GraphLimits::default(),
            )
            .unwrap();
        assert_eq!(graph.resources().len(), 13);
        assert_eq!(graph.passes().len(), 5);
        assert_eq!(device.selection(), &selected);
        assert_eq!(
            device
                .compile_active_instanced(
                    &templates,
                    &scopes,
                    FRAME,
                    &[],
                    &bindings,
                    &items,
                    &[PassKey(999)],
                    &[],
                    GraphLimits::default()
                )
                .err(),
            Some(FabricCompileError::ForeignActivePass(PassKey(999)))
        );
        assert_eq!(
            device
                .compile_active_instanced(
                    &templates,
                    &scopes,
                    FRAME,
                    &[],
                    &bindings,
                    &items,
                    &[PassKey(1), PassKey(1)],
                    &[],
                    GraphLimits::default()
                )
                .err(),
            Some(FabricCompileError::DuplicateActivePass(PassKey(1)))
        );
    }

    #[test]
    fn compile_rejects_missing_duplicate_and_wrong_identity_bindings_without_shared_fallback() {
        let fixture = fixture();
        let resources = fixture.resources();
        let selection = select(&fixture.catalog);
        let compile = |shared: &[FrameResourceBinding], objects: &[ObjectResourceBinding]| {
            resources
                .compile_with_instances(
                    &selection,
                    FRAME,
                    shared,
                    objects,
                    &fixture.items,
                    GraphLimits::default(),
                )
                .err()
        };
        let mut objects = fixture.bindings.clone();
        let removed = objects.remove(0);
        assert_eq!(
            compile(&[], &objects),
            Some(FabricCompileError::MissingObjectBinding(removed.address()))
        );
        objects = fixture.bindings.clone();
        objects.push(objects[0]);
        assert_eq!(
            compile(&[], &objects),
            Some(FabricCompileError::DuplicateObjectBinding(
                objects[0].address()
            ))
        );
        let wrong = [
            ObjectResourceIdentity::new(
                RenderObjectId::Hair(DomainObjectHandle::new(4, 99)),
                7,
                ROLE,
            ),
            ObjectResourceIdentity::new(
                RenderObjectId::Particle(DomainObjectHandle::new(4, 1)),
                7,
                ROLE,
            ),
            ObjectResourceIdentity::new(identities()[0].object(), 99, ROLE),
            ObjectResourceIdentity::new(identities()[0].object(), 7, PayloadRole(99)),
        ];
        for identity in wrong {
            objects = fixture.bindings.clone();
            objects[0].identity = identity;
            assert_eq!(
                compile(&[], &objects),
                Some(FabricCompileError::UnexpectedObjectBinding(
                    objects[0].address()
                ))
            );
        }
        objects = fixture.bindings.clone();
        objects[0].binding.frame = FRAME + 1;
        assert_eq!(
            compile(&[], &objects),
            Some(FabricCompileError::UnexpectedObjectBinding(
                objects[0].address()
            ))
        );
        objects[0] = fixture.bindings[0];
        objects[0].binding.key = SHARED;
        assert_eq!(
            compile(&[], &objects),
            Some(FabricCompileError::UnexpectedObjectBinding(
                objects[0].address()
            ))
        );
        assert_eq!(
            compile(&[removed.binding], &fixture.bindings),
            Some(FabricCompileError::UnexpectedBinding {
                resource: A,
                frame: FRAME
            })
        );
        let mut objects = fixture.bindings.clone();
        objects.swap(0, 2); // permutation is harmless; identity is not vector position
        assert!(compile(&[], &objects).is_none());
        objects[0].binding.physical = objects[1].binding.physical;
        assert!(matches!(
            compile(&[], &objects),
            Some(FabricCompileError::OverlappingBindings { .. })
        ));
    }

    #[test]
    fn object_payload_coverage_layouts_and_aliases_remain_strict() {
        let fixture = fixture();
        let resources = fixture.resources();
        let selection = select(&fixture.catalog);
        let compile = |objects: &[ObjectResourceBinding], items: &[FrameRenderItem]| {
            resources
                .compile_with_instances(
                    &selection,
                    FRAME,
                    &[],
                    objects,
                    items,
                    GraphLimits::default(),
                )
                .err()
        };
        let mut items = fixture.items.clone();
        items[0].payload = DomainPayload::Value(ValuePayload {
            value: PayloadValue::Scalar(0.0),
            role: ROLE,
        });
        assert_eq!(
            compile(&fixture.bindings, &items),
            Some(FabricCompileError::MissingObjectPayload {
                resource: A,
                identity: identities()[0]
            })
        );
        items = fixture.items.clone();
        items.push(items[0]);
        assert!(matches!(
            compile(&fixture.bindings, &items),
            Some(FabricCompileError::DuplicatePayloadRole { .. })
        ));
        items = fixture.items.clone();
        items[0].domain = RenderDomain::Particle;
        assert!(matches!(
            compile(&fixture.bindings, &items),
            Some(FabricCompileError::DomainIdentityMismatch { .. })
        ));
        items = fixture.items.clone();
        let DomainPayload::Resource(mut payload) = items[0].payload else {
            unreachable!()
        };
        payload.role = PayloadRole(99);
        items[0].payload = DomainPayload::Resource(payload);
        assert_eq!(
            compile(&fixture.bindings, &items),
            Some(FabricCompileError::PayloadInstanceMismatch(A))
        );
        let DomainPayload::Resource(original) = fixture.items[0].payload else {
            unreachable!()
        };
        for changed in [
            ResourcePayload {
                size: original.size + 1,
                ..original
            },
            ResourcePayload {
                alignment: 32,
                ..original
            },
            ResourcePayload {
                lifetime: ResourceLifetime::Imported,
                ..original
            },
            ResourcePayload {
                layout: ResourceLayout::Buffer {
                    bytes: 512,
                    usage: BufferUsages::STORAGE,
                },
                size: 512,
                ..original
            },
            ResourcePayload {
                layout: ResourceLayout::Buffer {
                    bytes: 0,
                    usage: BufferUsages::STORAGE,
                },
                size: 0,
                ..original
            },
            ResourcePayload {
                layout: ResourceLayout::Buffer {
                    bytes: 64,
                    usage: BufferUsages::UNIFORM,
                },
                ..original
            },
        ] {
            items = fixture.items.clone();
            items[0].payload = DomainPayload::Resource(changed);
            assert_eq!(
                compile(&fixture.bindings, &items),
                Some(FabricCompileError::PayloadLayoutMismatch(A))
            );
        }
        for index in [0, 1] {
            // current and history layouts validate independently
            let mut objects = fixture.bindings.clone();
            objects[index].binding.layout = ResourceLayout::Buffer {
                bytes: 512,
                usage: BufferUsages::STORAGE,
            };
            assert_eq!(
                compile(&objects, &fixture.items),
                Some(FabricCompileError::InvalidObjectBinding(
                    objects[index].address()
                ))
            );
            objects = fixture.bindings.clone();
            objects[index].binding.alignment = 32;
            assert_eq!(
                compile(&objects, &fixture.items),
                Some(FabricCompileError::InvalidObjectBinding(
                    objects[index].address()
                ))
            );
            for offset in [1, u64::MAX - 15] {
                objects = fixture.bindings.clone();
                objects[index].binding.physical = PhysicalResource::Buffer {
                    id: BufferId::from_parts(70, 1),
                    offset,
                };
                assert_eq!(
                    compile(&objects, &fixture.items),
                    Some(FabricCompileError::InvalidObjectBinding(
                        objects[index].address()
                    ))
                );
            }
        }
        let mut objects = fixture.bindings.clone();
        objects[2].binding.physical = objects[0].binding.physical;
        assert!(matches!(
            compile(&objects, &fixture.items),
            Some(FabricCompileError::OverlappingBindings { .. })
        ));
    }

    #[test]
    fn scope_declarations_cannot_mix_shared_objects_or_instantiate_service_outputs() {
        let fixture = fixture();
        let error = |scopes: &[ObjectResourceAccess]| {
            ResourceTemplates::new_with_object_accesses(
                &fixture.catalog,
                &fixture.templates,
                scopes,
            )
            .err()
            .unwrap()
        };
        let mut scopes = fixture.scopes.clone();
        scopes.pop();
        assert_eq!(
            error(&scopes),
            FabricCompileError::InconsistentObjectResource(A)
        );
        scopes = fixture.scopes.clone();
        scopes.push(scopes[0]);
        assert_eq!(
            error(&scopes),
            FabricCompileError::InvalidObjectAccess {
                pass: PassKey(1),
                ordinal: 0
            }
        );
        for (pass, ordinal) in [(3, 2), (10, 0), (1, 99), (999, 0)] {
            scopes = fixture.scopes.clone();
            scopes.push(ObjectResourceAccess {
                pass: PassKey(pass),
                ordinal,
                ..scopes[0]
            });
            assert_eq!(
                error(&scopes),
                FabricCompileError::InvalidObjectAccess {
                    pass: PassKey(pass),
                    ordinal
                }
            );
        }
        scopes = fixture.scopes.clone();
        scopes[0].layout = ObjectLayoutPolicy::Exact;
        assert_eq!(
            error(&scopes),
            FabricCompileError::InconsistentObjectResource(A)
        );
        scopes = fixture.scopes.clone();
        scopes[0].layout = ObjectLayoutPolicy::Sized { max_bytes: 32 };
        assert_eq!(
            error(&scopes),
            FabricCompileError::InvalidObjectAccess {
                pass: PassKey(1),
                ordinal: 0
            }
        );
    }

    #[test]
    fn expanded_limits_apply_before_graph_lowering() {
        let fixture = fixture();
        for limits in [
            GraphLimits {
                resources: 12,
                ..GraphLimits::default()
            },
            GraphLimits {
                accesses_per_pass: 12,
                ..GraphLimits::default()
            },
        ] {
            assert!(matches!(
                fixture.resources().compile_with_instances(
                    &select(&fixture.catalog),
                    FRAME,
                    &[],
                    &fixture.bindings,
                    &fixture.items,
                    limits
                ),
                Err(FabricCompileError::Graph(_))
            ));
        }
    }

    #[test]
    fn runtime_rejects_missing_duplicate_wrong_address_range_version_and_physical_imports() {
        let fixture = fixture();
        let graph = fixture.compile().unwrap();
        let imports = fixture.imports(&graph);
        let mut changed = imports.clone();
        let removed = changed.remove(0);
        assert_eq!(
            prepare_error(&fixture, &changed),
            FabricRuntimeError::MissingImport(removed.imported.resource())
        );
        changed = imports.clone();
        changed.push(changed[0]);
        assert_eq!(
            prepare_error(&fixture, &changed),
            FabricRuntimeError::DuplicateImport(changed[0].imported.resource())
        );
        for address in [
            ResourceAddress::shared(A, FRAME),
            ResourceAddress::object(A, FRAME - 1, identities()[0]),
            ResourceAddress::object(A, FRAME, identities()[1]),
            ResourceAddress::object(
                A,
                FRAME,
                ObjectResourceIdentity::new(identities()[0].object(), 99, ROLE),
            ),
            ResourceAddress::object(
                A,
                FRAME,
                ObjectResourceIdentity::new(
                    RenderObjectId::Hair(DomainObjectHandle::new(4, 99)),
                    7,
                    ROLE,
                ),
            ),
        ] {
            changed = imports.clone();
            changed[0].address = address;
            assert_eq!(
                prepare_error(&fixture, &changed),
                FabricRuntimeError::InvalidObjectImport(address)
            );
        }
        changed = imports.clone();
        changed[0].imported = imports[2].imported; // same physical backing, different semantic ID
        assert_eq!(
            prepare_error(&fixture, &changed),
            FabricRuntimeError::InvalidObjectImport(changed[0].address)
        );
        for binding in [
            FrameResourceBinding {
                version: imports[0].binding.version + 1,
                ..imports[0].binding
            },
            FrameResourceBinding {
                physical: imports[2].binding.physical,
                ..imports[0].binding
            },
            FrameResourceBinding {
                layout: imports[1].binding.layout,
                ..imports[0].binding
            },
            FrameResourceBinding {
                frame: FRAME - 1,
                ..imports[0].binding
            },
            FrameResourceBinding {
                alignment: 32,
                ..imports[0].binding
            },
        ] {
            changed = imports.clone();
            changed[0].binding = binding;
            assert_eq!(
                prepare_error(&fixture, &changed),
                FabricRuntimeError::InvalidObjectImport(changed[0].address)
            );
        }
        changed = imports.clone();
        changed[0].imported = RuntimeImport::Buffer {
            resource: imports[0].imported.resource(),
            imported: ImportedBuffer {
                buffer: BufferId::from_parts(71, 1),
                entry_state: BufferState::initial(),
            },
        };
        assert_eq!(
            prepare_error(&fixture, &changed),
            FabricRuntimeError::InvalidImport(imports[0].imported.resource())
        );
        assert_eq!(
            graph
                .prepare_runtime_with_object_imports(
                    Extent3d::new_2d(1, 1),
                    &[imports[0].imported],
                    &imports[1..],
                    recorders()
                )
                .err(),
            Some(FabricRuntimeError::UnexpectedImport(
                imports[0].imported.resource()
            ))
        );
    }

    #[test]
    fn empty_domain_keeps_singleton_services_and_pass_recorders() {
        let mut fixture = fixture();
        fixture.items.clear();
        fixture.bindings.clear();
        let graph = fixture.compile().unwrap();
        assert_eq!(graph.passes().len(), 5);
        assert_eq!(graph.resources().len(), 1);
        assert!(graph.resources().iter().all(|r| r.identity.is_none()));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let callbacks = PASSES
            .into_iter()
            .map(|key| {
                let calls = Arc::clone(&calls);
                RuntimePassBinding::new(key, move |ctx| {
                    assert_eq!(ctx.items().count(), 0);
                    assert!(ctx
                        .accesses()
                        .iter()
                        .all(|access| access.bound.identity.is_none()));
                    calls.lock().unwrap().push(key);
                })
            })
            .collect();
        let instance = prism_render_driver_null::NullInstance::new();
        let adapter = instance.enumerate_adapters().pop().unwrap();
        let device = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .unwrap()
            .0;
        let runtime = graph
            .prepare_runtime_with_object_imports(Extent3d::new_2d(1, 1), &[], &[], callbacks)
            .unwrap();
        let recorded = runtime.record(&device);
        assert_eq!(calls.lock().unwrap().len(), 5);
        recorded.release(&device);
        assert_eq!(device.live_resource_count(), 0);
    }
}
