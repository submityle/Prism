use super::{runtime::verify_plan, *};
use crate::frame_graph::{AccessKind, QueueClass, ResourceId, ResourceLifetime};
use prism_render_driver::{
    Barrier, BufferId, BufferState, BufferUsages, Extent3d, SubresourceRange, TextureState,
    TextureViewId,
};
use prism_render_graph::{
    BufferDesc, ExecutionPlan, ImportedBuffer, ImportedTexture, LogicalResource, PassFlags,
    RenderGraph, SizeClass, TextureDesc,
};

const FEATURE: FeatureId = FeatureId(900);
const RESOURCE: ResourceKey = ResourceKey::of(FEATURE, 0);

fn table<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}

fn fixture() -> (FeatureCatalog, ResourceTemplate) {
    let pass = PassTemplate {
        depends_on: &[],
        id: PassKey(1),
        accesses: table(vec![ResourceUse {
            input: ResourceInput::Automatic,
            key: RESOURCE,
            access: AccessKind::StorageWrite,
            scope: ResourceScope::Whole,
            frame_lag: None,
        }]),
        queue: QueueClass::Compute,
    };
    let variant = FeatureVariant {
        id: VariantId(1),
        rank: 0,
        min_quality: RenderQualityTier::Low,
        required_caps: CapabilitySet::EMPTY,
        requires_services_extra: ServiceSet::EMPTY,
        passes: table(vec![pass]),
        shader_permutation: None,
    };
    let catalog = FeatureCatalog::new(
        table(vec![FeatureDefinition::subsystem(
            FEATURE,
            RenderDomain::Surface,
            ServiceSet::EMPTY,
            table(vec![variant]),
        )]),
        table(vec![RESOURCE]),
    )
    .unwrap();
    let resource = ResourceTemplate {
        key: RESOURCE,
        lifetime: ResourceLifetime::Transient,
        layout: ResourceLayout::Buffer {
            bytes: 64,
            usage: BufferUsages::STORAGE,
        },
        alignment: 16,
    };
    (catalog, resource)
}

fn compiled() -> CompiledFeatureGraph {
    let (catalog, resource) = fixture();
    let roots = FrozenRoots::from_build_profile(&catalog, &[FEATURE], &[]).unwrap();
    let selection = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .unwrap();
    ResourceTemplates::new(&catalog, &[resource])
        .unwrap()
        .compile(&selection, 1, &[], &[], GraphLimits::default())
        .unwrap()
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

fn compile_passes(
    passes: Vec<PassTemplate>,
    resources: &[ResourceTemplate],
    external: &[FrameResourceBinding],
) -> CompiledFeatureGraph {
    let catalog = FeatureCatalog::new(
        table(vec![FeatureDefinition::subsystem(
            FEATURE,
            RenderDomain::Surface,
            ServiceSet::EMPTY,
            table(vec![FeatureVariant {
                id: VariantId(1),
                rank: 0,
                min_quality: RenderQualityTier::Low,
                required_caps: CapabilitySet::EMPTY,
                requires_services_extra: ServiceSet::EMPTY,
                passes: table(passes),
                shader_permutation: None,
            }]),
        )]),
        table(resources.iter().map(|resource| resource.key).collect()),
    )
    .unwrap();
    let roots = FrozenRoots::from_build_profile(&catalog, &[FEATURE], &[]).unwrap();
    let selection = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .unwrap();
    ResourceTemplates::new(&catalog, resources)
        .unwrap()
        .compile(&selection, 1, external, &[], GraphLimits::default())
        .unwrap()
}

fn recorders(keys: &[u32]) -> Vec<RuntimePassBinding> {
    keys.iter()
        .map(|&key| {
            RuntimePassBinding::new(PassKey(key), |_| {
                panic!("prepare_runtime must not execute recorders")
            })
        })
        .collect()
}

fn prepare(
    declaration: CompiledFeatureGraph,
    imports: &[RuntimeImport],
    keys: &[u32],
) -> Result<RuntimeFeatureGraph, FabricRuntimeError> {
    declaration.prepare_runtime(Extent3d::new_2d(1, 1), imports, recorders(keys))
}

fn buffer(key: ResourceKey, lifetime: ResourceLifetime) -> ResourceTemplate {
    ResourceTemplate {
        key,
        lifetime,
        layout: ResourceLayout::Buffer {
            bytes: 64,
            usage: BufferUsages::STORAGE,
        },
        alignment: 16,
    }
}

fn texture(key: ResourceKey, lifetime: ResourceLifetime) -> ResourceTemplate {
    ResourceTemplate {
        key,
        lifetime,
        layout: ResourceLayout::Texture {
            extent: [8, 8, 2],
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 2,
            samples: 1,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        },
        alignment: 16,
    }
}

fn binding(template: ResourceTemplate, physical: PhysicalResource) -> FrameResourceBinding {
    FrameResourceBinding {
        key: template.key,
        frame: 1,
        version: 5,
        physical,
        layout: template.layout,
        alignment: template.alignment,
    }
}

fn imported_buffer(id: BufferId) -> ImportedBuffer {
    ImportedBuffer {
        buffer: id,
        entry_state: BufferState::initial(),
    }
}

fn imported_texture(id: TextureId) -> ImportedTexture {
    ImportedTexture {
        texture: id,
        default_view: TextureViewId::from_parts(2, 1),
        entry_state: TextureState::undefined(),
        exit_state: TextureState::present(),
        mip_levels: 2,
        array_layers: 2,
    }
}

fn compiled_import(template: ResourceTemplate, physical: PhysicalResource) -> CompiledFeatureGraph {
    compile_passes(
        vec![pass(
            1,
            QueueClass::Compute,
            vec![usage(template.key, AccessKind::StorageRead)],
        )],
        &[template],
        &[binding(template, physical)],
    )
}

fn duplicate_access_plan(overwrite: bool) -> (
    CompiledFeatureGraph,
    ExecutionPlan,
    [Option<RuntimeResourceHandle>; 2],
) {
    let image = ResourceKey::of(FEATURE, 1);
    let buffer_template = buffer(RESOURCE, ResourceLifetime::Imported);
    let image_template = texture(image, ResourceLifetime::Imported);
    let buffer_id = BufferId::from_parts(1, 1);
    let image_id = TextureId::from_parts(1, 1);
    // A catalog variant has one writer per resource. Imported contents let us
    // exercise repeated WAR sources before that writer and repeated RAW
    // destinations after it without declaring an ambiguous second writer.
    let repeated_reads = |input| {
        vec![
            usage(image, AccessKind::StorageRead),
            usage(RESOURCE, AccessKind::StorageRead),
            usage(image, AccessKind::StorageRead),
            usage(RESOURCE, AccessKind::StorageRead),
        ]
        .into_iter().map(|mut usage| { usage.input = input; usage }).collect()
    };
    let declaration = compile_passes(
        vec![
            pass(1, QueueClass::Compute, repeated_reads(ResourceInput::Initial)),
            pass(
                2,
                QueueClass::Compute,
                vec![
                    usage(RESOURCE, AccessKind::StorageWrite),
                    usage(image, AccessKind::StorageWrite),
                ],
            ),
            pass(3, QueueClass::Compute, repeated_reads(ResourceInput::ProducedBy(PassKey(2)))),
        ],
        &[buffer_template, image_template],
        &[
            binding(
                buffer_template,
                PhysicalResource::Buffer {
                    id: buffer_id,
                    offset: 0,
                },
            ),
            binding(image_template, PhysicalResource::Texture(image_id)),
        ],
    );
    // Use the real compiler for a mixed-namespace WAR/RAW graph. Both
    // resource indices are zero, but the semantic resources are distinct.
    // The overwrite mode deliberately substitutes a WAW-producing final pass
    // in the executable graph, not in the valid catalog declaration.
    let mut graph = RenderGraph::new(Extent3d::new_2d(1, 1));
    let flags = PassFlags::NEVER_CULL | PassFlags::ASYNC_COMPUTE;
    let mut buffer = graph
        .try_import_buffer(
            "buffer",
            BufferDesc::new(64, BufferUsages::STORAGE),
            imported_buffer(buffer_id),
        )
        .unwrap();
    let mut image = graph
        .try_import_texture(
            "image",
            TextureDesc {
                size: SizeClass::Absolute {
                    width: 8,
                    height: 8,
                    depth_or_array_layers: 2,
                },
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                mip_level_count: 2,
                sample_count: 1,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            },
            imported_texture(image_id),
        )
        .unwrap();
    graph.add_compute_pass("read twice before write", flags, |builder| {
        for _ in 0..2 {
            builder.read_storage_texture(image);
            builder.read_storage_buffer(buffer);
        }
        |_| panic!("plan validation must not execute recorders")
    });
    graph.add_compute_pass("write", flags, |builder| {
        buffer = builder.write_storage_buffer(buffer);
        image = builder.write_storage_texture(image);
        |_| panic!("plan validation must not execute recorders")
    });
    graph.add_compute_pass("read twice after write", flags, |builder| {
        if overwrite {
            buffer = builder.write_storage_buffer(buffer);
            image = builder.write_storage_texture(image);
        } else {
            for _ in 0..2 {
                builder.read_storage_texture(image);
                builder.read_storage_buffer(buffer);
            }
        }
        |_| panic!("plan validation must not execute recorders")
    });
    let plan = graph.compile().unwrap();
    let handles = [
        Some(RuntimeResourceHandle::Buffer {
            handle: buffer,
            offset: 0,
            bytes: 64,
        }),
        Some(RuntimeResourceHandle::Texture(image)),
    ];
    (declaration, plan, handles)
}

#[test]
fn runtime_binding_api_is_constructible() {
    let binding = RuntimePassBinding::new(PassKey(1), |_ctx| {});
    drop(binding);
}

#[test]
fn runtime_requires_exact_recorder_coverage() {
    let result = compiled().prepare_runtime(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        &[],
        Vec::new(),
    );
    assert!(matches!(
        result,
        Err(FabricRuntimeError::MissingRecorder(PassKey(1)))
    ));
}

#[test]
fn runtime_lowers_transient_accesses_through_authority() {
    let runtime = compiled()
        .prepare_runtime(
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            &[],
            recorders(&[1]),
        )
        .unwrap();
    assert_eq!(runtime.plan().pass_count(), 1);
    let pass = &runtime.declaration().passes()[0];
    assert_eq!(pass.key(), PassKey(1));
    assert_eq!(pass.domain(), Some(RenderDomain::Surface));
    assert_eq!(pass.accesses().len(), 1);
    assert_eq!(pass.accesses()[0].ssa_version, 1);
    assert_eq!(runtime.plan().used_buffers, vec![true]);
    assert!(runtime.plan().alias.buffer_slots[0].is_some());
}

#[test]
fn runtime_rejects_unexpected_import_without_mutating_graph() {
    let compiled = compiled();
    let import = RuntimeImport::Buffer {
        resource: ResourceId(99),
        imported: ImportedBuffer {
            buffer: BufferId::from_parts(1, 0),
            entry_state: BufferState::initial(),
        },
    };
    let result = compiled.prepare_runtime(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        &[import],
        vec![RuntimePassBinding::new(PassKey(1), |_| {})],
    );
    assert!(matches!(
        result,
        Err(FabricRuntimeError::UnexpectedImport(ResourceId(99)))
    ));
}

#[test]
fn runtime_reports_recorder_errors_in_input_order_before_missing() {
    for (keys, expected) in [
        (
            vec![99, 1, 1],
            FabricRuntimeError::UnexpectedRecorder(PassKey(99)),
        ),
        (
            vec![1, 1, 99],
            FabricRuntimeError::DuplicateRecorder(PassKey(1)),
        ),
        (
            vec![99],
            FabricRuntimeError::UnexpectedRecorder(PassKey(99)),
        ),
    ] {
        assert_eq!(prepare(compiled(), &[], &keys).err(), Some(expected));
    }
}

#[test]
fn runtime_reports_first_missing_recorder_in_pass_order_not_key_order() {
    for (keys, missing) in [(vec![], 30), (vec![30], 10), (vec![10, 30], 20)] {
        let declaration = compile_passes(
            [30, 10, 20]
                .into_iter()
                .map(|key| pass(key, QueueClass::Compute, vec![]))
                .collect(),
            &[],
            &[],
        );
        assert_eq!(
            prepare(declaration, &[], &keys).err(),
            Some(FabricRuntimeError::MissingRecorder(PassKey(missing)))
        );
    }
}

#[test]
fn runtime_recorder_permutations_preserve_plan_and_empty_domain_passes() {
    // Sparse, non-monotonic keys exercise indexed coverage without treating
    // PassKey as a dense vector index or using recorder order as pass order.
    let keys: Vec<_> = (0..128)
        .map(|index| 1000 + ((index * 73) % 128) * 17)
        .collect();
    let mut expected = None;
    for supplied in [keys.clone(), keys.iter().rev().copied().collect()] {
        let declaration = compile_passes(
            keys.iter()
                .map(|&key| pass(key, QueueClass::Compute, vec![]))
                .collect(),
            &[],
            &[],
        );
        let runtime = prepare(declaration, &[], &supplied).unwrap();
        assert_eq!(runtime.plan().pass_count(), keys.len());
        assert_eq!(runtime.plan().order, (0..keys.len()).collect::<Vec<_>>());
        assert_eq!(
            runtime
                .declaration()
                .passes()
                .iter()
                .map(PassSpec::key)
                .collect::<Vec<_>>(),
            keys.iter().copied().map(PassKey).collect::<Vec<_>>()
        );
        if let Some(expected) = &expected {
            assert_eq!(runtime.plan(), expected);
        }
        expected = Some(runtime.plan().clone());
    }
}

#[test]
fn runtime_accepts_empty_graph_without_recorders_or_resources() {
    let runtime = prepare(compile_passes(vec![], &[], &[]), &[], &[]).unwrap();
    assert_eq!(runtime.plan().declared_pass_count, 0);
    assert!(runtime.plan().order.is_empty());
    assert!(runtime.plan().logical_barriers.is_empty());
}

#[test]
fn runtime_projects_storage_texture_hazards_and_mixed_access_ordinals() {
    let image = ResourceKey::of(FEATURE, 1);
    // No sampling or attachments: declaration lowering treats the image as a
    // buffer, whereas executable lowering must use its actual texture layout.
    let declaration = compile_passes(
        vec![
            pass(
                1,
                QueueClass::Compute,
                vec![
                    usage(RESOURCE, AccessKind::StorageWrite),
                    usage(image, AccessKind::StorageWrite),
                ],
            ),
            pass(
                2,
                QueueClass::Graphics,
                vec![
                    usage(image, AccessKind::StorageRead),
                    usage(RESOURCE, AccessKind::StorageRead),
                    usage(image, AccessKind::StorageRead),
                ],
            ),
        ],
        &[
            buffer(RESOURCE, ResourceLifetime::Transient),
            texture(image, ResourceLifetime::Transient),
        ],
        &[],
    );
    let runtime = prepare(declaration, &[], &[2, 1]).unwrap();
    let actual = &runtime.plan().logical_barriers;
    assert_ne!(actual, &runtime.declaration().plan().logical_barriers);
    assert_eq!(actual.len(), 3);
    assert_eq!(runtime.declaration().barriers().len(), 3);
    assert!(actual
        .iter()
        .all(|hazard| hazard.source == 0 && hazard.destination == 1 && hazard.queue_transfer));
    let texture_hazards: Vec<_> = actual
        .iter()
        .filter(|hazard| matches!(hazard.resource, LogicalResource::Texture(_)))
        .collect();
    assert_eq!(texture_hazards.len(), 2);
    assert_eq!(
        (texture_hazards[0].before, texture_hazards[0].after),
        (0, 0)
    );
    assert_eq!(
        (texture_hazards[1].before, texture_hazards[1].after),
        (0, 1)
    );
    let buffer_hazard = actual
        .iter()
        .find(|hazard| matches!(hazard.resource, LogicalResource::Buffer(_)))
        .unwrap();
    assert_eq!((buffer_hazard.before, buffer_hazard.after), (1, 2));
}

#[test]
fn runtime_projects_transfer_only_texture_hazards() {
    let mut template = texture(RESOURCE, ResourceLifetime::Transient);
    let ResourceLayout::Texture {
        usage: ref mut texture_usage,
        ..
    } = template.layout
    else {
        unreachable!()
    };
    *texture_usage = TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
    let declaration = compile_passes(
        vec![
            pass(
                1,
                QueueClass::Transfer,
                vec![usage(RESOURCE, AccessKind::TransferWrite)],
            ),
            pass(
                2,
                QueueClass::Transfer,
                vec![usage(RESOURCE, AccessKind::TransferRead)],
            ),
        ],
        &[template],
        &[],
    );
    let runtime = prepare(declaration, &[], &[2, 1]).unwrap();
    let actual = &runtime.plan().logical_barriers;
    let expected = &runtime.declaration().plan().logical_barriers;
    assert_eq!(actual.len(), 1);
    assert_eq!(expected.len(), 1);
    assert!(matches!(actual[0].resource, LogicalResource::Texture(_)));
    assert!(matches!(expected[0].resource, LogicalResource::Buffer(_)));
    assert_eq!(
        (
            actual[0].source,
            actual[0].before,
            actual[0].destination,
            actual[0].after
        ),
        (0, 0, 1, 0)
    );
    assert!(!actual[0].queue_transfer);
}

#[test]
fn runtime_plan_projection_rejects_changed_topology_and_hazards() {
    let second_key = ResourceKey::of(FEATURE, 1);
    let declaration = compile_passes(
        vec![
            pass(
                1,
                QueueClass::Compute,
                vec![
                    usage(RESOURCE, AccessKind::StorageWrite),
                    usage(second_key, AccessKind::StorageWrite),
                ],
            ),
            pass(
                2,
                QueueClass::Compute,
                vec![
                    usage(RESOURCE, AccessKind::StorageRead),
                    usage(second_key, AccessKind::StorageRead),
                ],
            ),
        ],
        &[
            buffer(RESOURCE, ResourceLifetime::Transient),
            buffer(second_key, ResourceLifetime::Transient),
        ],
        &[],
    );
    // Obtain both the plan and its namespace handles from a real graph, then
    // change one field at a time; no fabricated graph/compiler implementation.
    let mut graph = RenderGraph::new(Extent3d::new_2d(1, 1));
    let flags = PassFlags::NEVER_CULL | PassFlags::ASYNC_COMPUTE;
    let mut written = Vec::new();
    graph.add_compute_pass("write", flags, |builder| {
        for name in ["first", "second"] {
            let handle = builder.create_buffer(name, BufferDesc::new(64, BufferUsages::STORAGE));
            written.push(builder.write_storage_buffer(handle));
        }
        |_| panic!("plan validation must not execute recorders")
    });
    graph.add_compute_pass("read", flags, |builder| {
        for &handle in &written {
            builder.read_storage_buffer(handle);
        }
        |_| panic!("plan validation must not execute recorders")
    });
    let plan = graph.compile().unwrap();
    let handles: Vec<_> = written
        .into_iter()
        .map(|handle| {
            Some(RuntimeResourceHandle::Buffer {
                handle,
                offset: 0,
                bytes: 64,
            })
        })
        .collect();
    assert_eq!(verify_plan(&declaration, &plan, &handles), Ok(()));
    assert_eq!(plan.logical_barriers.len(), 2);
    for change in 0..13 {
        let mut changed = plan.clone();
        match change {
            0 => changed.declared_pass_count += 1,
            1 => changed.order.swap(0, 1),
            2 => changed.alive[0] = false,
            3 => changed.position_by_pass.swap(0, 1),
            4 => changed.queue_batches.clear(),
            5 => {
                changed.logical_barriers.pop();
            }
            6 => changed.logical_barriers.push(changed.logical_barriers[0]),
            7 => changed.logical_barriers[0].source = 1,
            8 => changed.logical_barriers[0].destination = 0,
            9 => changed.logical_barriers[0].before = 1,
            10 => changed.logical_barriers[0].after = usize::MAX,
            11 => changed.logical_barriers[0].resource = changed.logical_barriers[1].resource,
            12 => changed.logical_barriers[0].queue_transfer = true,
            _ => unreachable!(),
        }
        assert_eq!(
            verify_plan(&declaration, &changed, &handles),
            Err(FabricRuntimeError::PlanMismatch),
            "changed field {change}"
        );
    }
}

#[test]
fn runtime_plan_projection_preserves_duplicate_access_provenance_and_multiplicity() {
    let (declaration, plan, handles) = duplicate_access_plan(false);
    assert_eq!(verify_plan(&declaration, &plan, &handles), Ok(()));
    assert_eq!(plan.logical_barriers.len(), 8); // Four WAR, four RAW.
    let mut reordered = plan.clone();
    reordered.logical_barriers.reverse();
    assert_eq!(verify_plan(&declaration, &reordered, &handles), Ok(()));

    for (source, destination) in [(0, 1), (1, 2)] {
        for is_texture in [false, true] {
            let indices: Vec<_> = plan
                .logical_barriers
                .iter()
                .enumerate()
                .filter(|(_, barrier)| {
                    barrier.source == source
                        && barrier.destination == destination
                        && matches!(barrier.resource, LogicalResource::Texture(_)) == is_texture
                })
                .map(|(index, _)| index)
                .collect();
            assert_eq!(indices.len(), 2);
            let first = plan.logical_barriers[indices[0]];
            let second = plan.logical_barriers[indices[1]];
            if source == 0 {
                assert_ne!(first.before, second.before);
                assert_eq!(first.after, second.after);
            } else {
                assert_eq!(first.before, second.before);
                assert_ne!(first.after, second.after);
            }
            // Keep the resource, pass pair, queue flag and total hazard count
            // unchanged; only collapse one repeated access onto the other.
            for (replace, keep) in [(indices[0], indices[1]), (indices[1], indices[0])] {
                let mut changed = plan.clone();
                changed.logical_barriers[replace] = changed.logical_barriers[keep];
                assert_eq!(changed.logical_barriers.len(), plan.logical_barriers.len());
                assert_eq!(
                    verify_plan(&declaration, &changed, &handles),
                    Err(FabricRuntimeError::PlanMismatch),
                    "collapsed access: {source}->{destination}, texture={is_texture}, {replace}"
                );
            }
        }
    }
}

#[test]
fn runtime_plan_projection_rejects_cross_namespace_and_endpoint_identity_changes() {
    let (declaration, plan, handles) = duplicate_access_plan(false);
    assert_eq!(verify_plan(&declaration, &plan, &handles), Ok(()));
    for is_texture in [false, true] {
        let index = plan
            .logical_barriers
            .iter()
            .position(|barrier| {
                barrier.source == 0
                    && barrier.destination == 1
                    && matches!(barrier.resource, LogicalResource::Texture(_)) == is_texture
            })
            .unwrap();
        let opposite = match plan.logical_barriers[index].resource {
            LogicalResource::Texture(index) => {
                assert_eq!(index.get(), 0);
                LogicalResource::Buffer(index)
            }
            LogicalResource::Buffer(index) => {
                assert_eq!(index.get(), 0);
                LogicalResource::Texture(index)
            }
        };
        for change in 0..7 {
            let mut changed = plan.clone();
            let barrier = &mut changed.logical_barriers[index];
            match change {
                // The same numeric index in another namespace is not the same resource.
                0 => barrier.resource = opposite,
                // Destination and barrier still agree; source must agree too.
                1 => barrier.before = if is_texture { 2 } else { 0 },
                // Source and destination disagree even when the barrier's
                // resource is changed to match the new destination.
                2 => {
                    barrier.after = if is_texture { 1 } else { 0 };
                    barrier.resource = opposite;
                }
                3 => barrier.source = usize::MAX,
                4 => barrier.destination = usize::MAX,
                5 => barrier.before = usize::MAX,
                6 => barrier.after = usize::MAX,
                _ => unreachable!(),
            }
            assert_eq!(
                verify_plan(&declaration, &changed, &handles),
                Err(FabricRuntimeError::PlanMismatch),
                "changed identity field {change}, texture={is_texture}"
            );
        }
    }
}

#[test]
fn runtime_plan_projection_rejects_undeclared_overwrite_hazards() {
    let (declaration, plan, handles) = duplicate_access_plan(true);
    // This is a real compiled graph with the same passes, liveness and order,
    // but its final pass overwrites instead of performing the declared reads.
    // In particular, its two WAW hazards must not pass provenance projection.
    assert_eq!(
        plan.declared_pass_count,
        declaration.plan().declared_pass_count
    );
    assert_eq!(plan.order, declaration.plan().order);
    assert_eq!(plan.alive, declaration.plan().alive);
    assert_eq!(plan.position_by_pass, declaration.plan().position_by_pass);
    assert_eq!(plan.queue_batches, declaration.plan().queue_batches);
    let overwrites: Vec<_> = plan
        .logical_barriers
        .iter()
        .filter(|barrier| barrier.source == 1 && barrier.destination == 2)
        .collect();
    assert_eq!(overwrites.len(), 2);
    assert!(overwrites
        .iter()
        .any(|barrier| matches!(barrier.resource, LogicalResource::Buffer(_))));
    assert!(overwrites
        .iter()
        .any(|barrier| matches!(barrier.resource, LogicalResource::Texture(_))));
    assert_eq!(
        verify_plan(&declaration, &plan, &handles),
        Err(FabricRuntimeError::PlanMismatch)
    );
}

#[test]
fn runtime_projects_import_first_resource_indices_for_buffers_and_textures() {
    let external_key = ResourceKey::of(FEATURE, 1);
    for is_texture in [false, true] {
        let transient = if is_texture {
            texture(RESOURCE, ResourceLifetime::Transient)
        } else {
            buffer(RESOURCE, ResourceLifetime::Transient)
        };
        let external = ResourceTemplate {
            key: external_key,
            lifetime: ResourceLifetime::Imported,
            ..transient
        };
        let (physical, imported) = if is_texture {
            let id = TextureId::from_parts(5, 1);
            (
                PhysicalResource::Texture(id),
                RuntimeImport::Texture {
                    resource: ResourceId(1),
                    imported: imported_texture(id),
                },
            )
        } else {
            let id = BufferId::from_parts(5, 1);
            (
                PhysicalResource::Buffer { id, offset: 32 },
                RuntimeImport::Buffer {
                    resource: ResourceId(1),
                    imported: imported_buffer(id),
                },
            )
        };
        let read = if is_texture {
            AccessKind::SampledRead
        } else {
            AccessKind::StorageRead
        };
        let declaration = compile_passes(
            vec![
                pass(
                    1,
                    QueueClass::Compute,
                    vec![usage(RESOURCE, AccessKind::StorageWrite)],
                ),
                pass(
                    2,
                    QueueClass::Graphics,
                    vec![usage(RESOURCE, read), usage(external_key, read)],
                ),
            ],
            &[transient, external],
            &[binding(external, physical)],
        );
        assert_eq!(declaration.resource_id(RESOURCE, 1), Some(ResourceId(0)));
        assert_eq!(
            declaration.resource_id(external_key, 1),
            Some(ResourceId(1))
        );
        let runtime = prepare(declaration, &[imported], &[2, 1]).unwrap();
        assert_eq!(runtime.plan().logical_barriers.len(), 1);
        let resource_index = |resource| match resource {
            LogicalResource::Buffer(index) | LogicalResource::Texture(index) => index.get(),
        };
        assert_eq!(
            resource_index(runtime.declaration().plan().logical_barriers[0].resource),
            0
        );
        assert_eq!(
            resource_index(runtime.plan().logical_barriers[0].resource),
            1
        );
        assert_eq!(
            runtime.declaration().passes()[1].accesses()[1].content_version,
            5
        );
    }
}

#[test]
fn runtime_requires_exact_external_import_coverage() {
    let template = buffer(RESOURCE, ResourceLifetime::Imported);
    let id = BufferId::from_parts(1, 1);
    let physical = PhysicalResource::Buffer { id, offset: 16 };
    let import = RuntimeImport::Buffer {
        resource: ResourceId(0),
        imported: imported_buffer(id),
    };
    assert_eq!(
        prepare(compiled_import(template, physical), &[], &[1]).err(),
        Some(FabricRuntimeError::MissingImport(ResourceId(0)))
    );
    assert_eq!(
        prepare(compiled_import(template, physical), &[import, import], &[1]).err(),
        Some(FabricRuntimeError::DuplicateImport(ResourceId(0)))
    );
    assert_eq!(
        prepare(compiled(), &[import], &[1]).err(),
        Some(FabricRuntimeError::UnexpectedImport(ResourceId(0)))
    );
}

#[test]
fn runtime_rejects_import_kind_and_physical_generation_mismatches() {
    let template = buffer(RESOURCE, ResourceLifetime::Imported);
    let id = BufferId::from_parts(1, 1);
    let physical = PhysicalResource::Buffer { id, offset: 16 };
    for import in [
        RuntimeImport::Buffer {
            resource: ResourceId(0),
            imported: imported_buffer(BufferId::from_parts(1, 2)),
        },
        RuntimeImport::Texture {
            resource: ResourceId(0),
            imported: imported_texture(TextureId::from_parts(1, 1)),
        },
    ] {
        assert_eq!(
            prepare(compiled_import(template, physical), &[import], &[1]).err(),
            Some(FabricRuntimeError::InvalidImport(ResourceId(0)))
        );
    }
}

#[test]
fn runtime_validates_imported_texture_identity_mips_and_layers() {
    let template = texture(RESOURCE, ResourceLifetime::Imported);
    let id = TextureId::from_parts(1, 1);
    let physical = PhysicalResource::Texture(id);
    let imported = imported_texture(id);
    for invalid in [
        ImportedTexture {
            texture: TextureId::from_parts(1, 2),
            ..imported
        },
        ImportedTexture {
            mip_levels: 1,
            ..imported
        },
        ImportedTexture {
            array_layers: 1,
            ..imported
        },
    ] {
        let import = RuntimeImport::Texture {
            resource: ResourceId(0),
            imported: invalid,
        };
        assert_eq!(
            prepare(compiled_import(template, physical), &[import], &[1]).err(),
            Some(FabricRuntimeError::InvalidImport(ResourceId(0)))
        );
    }
    let import = RuntimeImport::Texture {
        resource: ResourceId(0),
        imported,
    };
    let runtime = prepare(compiled_import(template, physical), &[import], &[1]).unwrap();
    assert_eq!(runtime.plan().used_textures, vec![true]);
    assert_eq!(runtime.plan().final_barriers.len(), 1);
    let Barrier::Texture(exit) = runtime.plan().final_barriers[0] else {
        panic!("imported texture requires a texture exit barrier")
    };
    assert_eq!(exit.texture.get(), 0);
    assert_eq!(exit.after, imported.exit_state);
    assert_eq!(
        exit.range,
        SubresourceRange {
            base_mip_level: 0,
            mip_level_count: imported.mip_levels,
            base_array_layer: 0,
            array_layer_count: imported.array_layers,
        }
    );
    assert!(runtime.plan().alias.texture_slots[0].is_none());
}

#[test]
fn runtime_imported_volume_uses_one_array_layer_not_depth_slices() {
    let mut template = texture(RESOURCE, ResourceLifetime::Persistent);
    let ResourceLayout::Texture {
        ref mut extent,
        ref mut dimension,
        ..
    } = template.layout
    else {
        unreachable!()
    };
    *extent = [8, 8, 4];
    *dimension = TextureDimension::D3;
    let id = TextureId::from_parts(1, 1);
    let physical = PhysicalResource::Texture(id);
    for (layers, expected_error) in [(4, true), (1, false)] {
        let import = RuntimeImport::Texture {
            resource: ResourceId(0),
            imported: ImportedTexture {
                array_layers: layers,
                ..imported_texture(id)
            },
        };
        let result = prepare(compiled_import(template, physical), &[import], &[1]);
        if expected_error {
            assert_eq!(
                result.err(),
                Some(FabricRuntimeError::InvalidImport(ResourceId(0)))
            );
        } else {
            let runtime = result.unwrap();
            assert_eq!(runtime.plan().used_textures, vec![true]);
            assert!(runtime.plan().alias.texture_slots[0].is_none());
        }
    }
}

#[test]
fn runtime_retains_initial_and_intermediate_handles_across_multiple_writes() {
    let access = |kind, input| ResourceUse { input, ..usage(RESOURCE, kind) };
    let template = buffer(RESOURCE, ResourceLifetime::Imported);
    let id = BufferId::from_parts(7, 1);
    let declaration = compile_passes(vec![
        pass(40, QueueClass::Compute, vec![access(AccessKind::StorageRead, ResourceInput::ProducedBy(PassKey(30)))]),
        pass(30, QueueClass::Compute, vec![access(AccessKind::StorageWrite, ResourceInput::ProducedBy(PassKey(10)))]),
        pass(20, QueueClass::Compute, vec![access(AccessKind::StorageRead, ResourceInput::ProducedBy(PassKey(10)))]),
        pass(10, QueueClass::Compute, vec![access(AccessKind::StorageWrite, ResourceInput::Initial)]),
        pass(5, QueueClass::Compute, vec![access(AccessKind::StorageRead, ResourceInput::Initial)]),
    ], &[template], &[binding(template, PhysicalResource::Buffer { id, offset: 0 })]);
    let runtime = prepare(declaration, &[RuntimeImport::Buffer {
        resource: ResourceId(0), imported: imported_buffer(id),
    }], &[40, 30, 20, 10, 5]).unwrap();
    let keys: Vec<_> = runtime.plan().order.iter()
        .map(|&index| runtime.declaration().passes()[index].key()).collect();
    assert_eq!(keys, vec![PassKey(5), PassKey(10), PassKey(20), PassKey(30), PassKey(40)]);
    assert_eq!(runtime.declaration().versions().len(), 2);
}

#[test]
fn runtime_preserves_control_only_dependencies_without_synthetic_resource_edges() {
    let mut last = pass(3, QueueClass::Compute, vec![]);
    last.depends_on = table(vec![PassKey(2)]);
    let mut middle = pass(2, QueueClass::Compute, vec![]);
    middle.depends_on = table(vec![PassKey(1)]);
    let declaration = compile_passes(vec![last, middle, pass(1, QueueClass::Compute, vec![])], &[], &[]);
    assert_eq!(declaration.passes()[1].descriptor().depends_on, vec![crate::frame_graph::PassId(0)]);
    assert_eq!(declaration.passes()[2].descriptor().depends_on, vec![crate::frame_graph::PassId(1)]);
    let runtime = prepare(declaration, &[], &[3, 2, 1]).unwrap();
    assert!(runtime.plan().logical_barriers.is_empty());
    assert_eq!(runtime.plan().order, vec![0, 1, 2]);
}
