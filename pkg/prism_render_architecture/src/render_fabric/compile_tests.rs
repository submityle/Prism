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
    FrameRenderItem {
        scene_handle: SceneHandle {
            index: 1,
            generation: 1,
        },
        domain: RenderDomain::Surface,
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
            scene: first.scene_handle,
            domain: first.domain,
            history_key: 0,
            role: first.payload_role(),
        }
    );
    let mut second = first;
    second.scene_handle.generation += 1;
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
