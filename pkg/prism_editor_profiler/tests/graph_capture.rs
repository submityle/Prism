//! Real architecture compilation/lowering and bounded capture regression tests.
#![cfg(feature = "render-graph")]

use prism_editor_profiler::{
    graph_capture::{architecture::*, *},
    ProfilerError,
};
use prism_render_architecture::{
    frame_graph::{AccessKind, QueueClass, ResourceLifetime},
    memory::{FrameBudgetLedger, FrameWorkBudget},
    render_fabric::*,
};
use prism_render_graph::capacity::GraphLimits;

const ROOT: FeatureId = FeatureId(700);
const BUFFER: ResourceKey = ResourceKey::of(ROOT, 0);
const TEXTURE: ResourceKey = ResourceKey::of(ROOT, 1);
fn table<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}
fn access(key: ResourceKey, access: AccessKind) -> ResourceUse {
    ResourceUse {
        key,
        access,
        scope: ResourceScope::Whole,
        frame_lag: None,
        input: ResourceInput::Automatic,
    }
}
// These are actual catalog declarations compiled and lowered through production
// RenderGraph::prepare. They are not a mock graph, nor an installed app hook.
fn graph() -> CompiledFeatureGraph {
    let passes = table(vec![
        PassTemplate {
            id: PassKey(10),
            queue: QueueClass::Transfer,
            depends_on: &[],
            accesses: table(vec![access(BUFFER, AccessKind::TransferWrite)]),
        },
        PassTemplate {
            id: PassKey(20),
            queue: QueueClass::Graphics,
            depends_on: &[PassKey(10)],
            accesses: table(vec![
                access(BUFFER, AccessKind::StorageRead),
                access(TEXTURE, AccessKind::ColorAttachment),
            ]),
        },
        PassTemplate {
            id: PassKey(30),
            queue: QueueClass::Compute,
            depends_on: &[],
            accesses: table(vec![
                access(BUFFER, AccessKind::StorageRead),
                access(TEXTURE, AccessKind::SampledRead),
            ]),
        },
        // Explicit-only edge to a pass with no accesses must survive capture.
        PassTemplate {
            id: PassKey(40),
            queue: QueueClass::Compute,
            depends_on: &[PassKey(30)],
            accesses: &[],
        },
    ]);
    let variants = table(vec![FeatureVariant {
        id: VariantId(700),
        rank: 0,
        min_quality: RenderQualityTier::Low,
        required_caps: CapabilitySet::EMPTY,
        requires_services_extra: ServiceSet::EMPTY,
        passes,
        shader_permutation: None,
    }]);
    let catalog = FeatureCatalog::new(
        table(vec![FeatureDefinition::subsystem(
            ROOT,
            RenderDomain::Surface,
            ServiceSet::EMPTY,
            variants,
        )]),
        table(vec![BUFFER, TEXTURE]),
    )
    .unwrap();
    let roots = FrozenRoots::from_build_profile(&catalog, &[ROOT], &[]).unwrap();
    let selection = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .unwrap();
    ResourceTemplates::new(
        &catalog,
        &[
            ResourceTemplate {
                key: BUFFER,
                lifetime: ResourceLifetime::Transient,
                alignment: 16,
                layout: ResourceLayout::Buffer {
                    bytes: 64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                },
            },
            ResourceTemplate {
                key: TEXTURE,
                lifetime: ResourceLifetime::Transient,
                alignment: 256,
                layout: ResourceLayout::Texture {
                    extent: [16, 8, 1],
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                },
            },
        ],
    )
    .unwrap()
    .compile(&selection, 42, &[], &[], GraphLimits::default())
    .unwrap()
}
fn capture() -> GraphCaptureReport {
    capture_feature_graph(
        &graph(),
        GraphCaptureInputs::default(),
        GraphCaptureLimits::default(),
    )
    .unwrap()
}

#[test]
fn imported_first_backend_indices_are_mapped_back_for_buffers_and_textures() {
    use prism_render_architecture::frame_graph::ResourceId;
    use prism_render_driver::{BufferState, TextureState, TextureViewId};
    use prism_render_graph::{ImportedBuffer, ImportedTexture, LogicalResource};
    for texture in [false, true] {
        let layout = if texture {
            ResourceLayout::Texture {
                extent: [8, 8, 1],
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            }
        } else {
            ResourceLayout::Buffer {
                bytes: 64,
                usage: BufferUsages::STORAGE,
            }
        };
        let read = if texture {
            AccessKind::SampledRead
        } else {
            AccessKind::StorageRead
        };
        let passes = table(vec![
            PassTemplate {
                id: PassKey(1),
                queue: QueueClass::Compute,
                depends_on: &[],
                accesses: table(vec![access(BUFFER, AccessKind::StorageWrite)]),
            },
            PassTemplate {
                id: PassKey(2),
                queue: QueueClass::Graphics,
                depends_on: &[],
                accesses: table(vec![access(BUFFER, read), access(TEXTURE, read)]),
            },
        ]);
        let variants = table(vec![FeatureVariant {
            id: VariantId(1),
            rank: 0,
            min_quality: RenderQualityTier::Low,
            required_caps: CapabilitySet::EMPTY,
            requires_services_extra: ServiceSet::EMPTY,
            passes,
            shader_permutation: None,
        }]);
        let catalog = FeatureCatalog::new(
            table(vec![FeatureDefinition::subsystem(
                ROOT,
                RenderDomain::Surface,
                ServiceSet::EMPTY,
                variants,
            )]),
            table(vec![BUFFER, TEXTURE]),
        )
        .unwrap();
        let roots = FrozenRoots::from_build_profile(&catalog, &[ROOT], &[]).unwrap();
        let selection = FeatureSelection::resolve(
            &catalog,
            &roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::Low,
        )
        .unwrap();
        let templates = [
            ResourceTemplate {
                key: BUFFER,
                lifetime: ResourceLifetime::Transient,
                alignment: 16,
                layout,
            },
            ResourceTemplate {
                key: TEXTURE,
                lifetime: ResourceLifetime::Imported,
                alignment: 16,
                layout,
            },
        ];
        // Opaque driver IDs establish exact import coverage; this test prepares,
        // and never realizes or submits these resources to a GPU.
        let (physical, import) = if texture {
            let id = TextureId::from_parts(5, 1);
            (
                PhysicalResource::Texture(id),
                RuntimeImport::Texture {
                    resource: ResourceId(1),
                    imported: ImportedTexture {
                        texture: id,
                        default_view: TextureViewId::from_parts(6, 1),
                        entry_state: TextureState::initial(),
                        exit_state: TextureState::initial(),
                        mip_levels: 1,
                        array_layers: 1,
                    },
                },
            )
        } else {
            let id = BufferId::from_parts(5, 1);
            (
                PhysicalResource::Buffer { id, offset: 32 },
                RuntimeImport::Buffer {
                    resource: ResourceId(1),
                    imported: ImportedBuffer {
                        buffer: id,
                        entry_state: BufferState::initial(),
                    },
                },
            )
        };
        let compiled = ResourceTemplates::new(&catalog, &templates)
            .unwrap()
            .compile(
                &selection,
                42,
                &[FrameResourceBinding {
                    key: TEXTURE,
                    frame: 42,
                    version: 5,
                    physical,
                    layout,
                    alignment: 16,
                }],
                &[],
                GraphLimits::default(),
            )
            .unwrap();
        let declared = capture_feature_graph(
            &compiled,
            GraphCaptureInputs::default(),
            GraphCaptureLimits::default(),
        )
        .unwrap();
        let local = |resource| match resource {
            LogicalResource::Buffer(i) | LogicalResource::Texture(i) => i.get(),
        };
        assert_eq!(local(compiled.plan().accesses[0][0].resource), 0);
        let recorders = compiled
            .passes()
            .iter()
            .map(|p| RuntimePassBinding::new(p.key(), |_| {}))
            .collect();
        let runtime = compiled
            .prepare_runtime(
                prism_render_driver::Extent3d::new_2d(8, 8),
                &[import],
                recorders,
            )
            .unwrap();
        assert_eq!(local(runtime.plan().accesses[0][0].resource), 1);
        let report = capture_runtime_graph(
            &runtime,
            GraphCaptureInputs::default(),
            GraphCaptureLimits::default(),
        )
        .unwrap();
        assert_eq!(report.passes(), declared.passes());
        assert_eq!(report.dependencies(), declared.dependencies());
        assert_eq!(report.resources(), declared.resources());
        assert_eq!(report.resources()[1].lifetime, 0);
    }
}

#[test]
fn real_prepared_backend_graph_preserves_passes_resources_ssa_and_hazards() {
    let compiled = graph();
    let declared = capture_feature_graph(
        &compiled,
        GraphCaptureInputs::default(),
        GraphCaptureLimits::default(),
    )
    .unwrap();
    let recorders = compiled
        .passes()
        .iter()
        .map(|pass| RuntimePassBinding::new(pass.key(), |_| {}))
        .collect();
    // Exact production lowering and backend compiler, no fake timestamp or driver.
    let runtime = compiled
        .prepare_runtime(prism_render_driver::Extent3d::new_2d(16, 8), &[], recorders)
        .unwrap();
    let report = capture_runtime_graph(
        &runtime,
        GraphCaptureInputs::default(),
        GraphCaptureLimits::default(),
    )
    .unwrap();
    assert_eq!(report.source(), GraphCaptureSource::PreparedRuntime);
    assert_eq!(report.frame(), 42);
    assert_eq!(report.passes().len(), 4);
    assert_eq!(report.resources().len(), 2);
    assert_eq!(report.order(), &[0, 1, 2, 3]);
    assert_eq!(report.passes(), declared.passes());
    assert_eq!(report.dependencies(), declared.dependencies());
    assert_eq!(report.logical_bytes().unwrap(), 64 + 16 * 8 * 4);
    assert_eq!(report.passes()[1].accesses[0].input_version, 1);
    assert_eq!(report.passes()[1].accesses[1].output_version, Some(1));
    assert!(report
        .dependencies()
        .iter()
        .any(|e| e.before == 2 && e.after == 3 && e.kind == CapturedDependencyKind::Explicit));
    assert!(report.dependencies().iter().any(|e| e.before == 1
        && e.after == 2
        && matches!(
            e.kind,
            CapturedDependencyKind::Hazard {
                resource: 1,
                before_access: 1,
                after_access: 1,
                queue_transfer: true
            }
        )));
    assert!(report.dependencies().iter().any(|e| e.before == 0
        && e.after == 1
        && matches!(
            e.kind,
            CapturedDependencyKind::Hazard {
                resource: 0,
                before_access: 0,
                after_access: 0,
                queue_transfer: true
            }
        )));
    assert!(report
        .critical_path(GraphCaptureLimits::default())
        .unwrap()
        .is_none());
    let wire = report.to_bytes(GraphCaptureLimits::default()).unwrap();
    assert_eq!(
        GraphCaptureReport::from_bytes(&wire, GraphCaptureLimits::default()).unwrap(),
        report
    );
}

#[test]
fn real_cpu_measurements_feed_existing_critical_path_and_budget_ledger() {
    let graph = graph();
    let timings: Vec<_> = (0..4)
        .map(|pass| {
            let start = std::time::Instant::now();
            for i in 0..100 {
                std::hint::black_box(i);
            }
            GraphPassTiming {
                pass,
                duration_ns: u64::try_from(start.elapsed().as_nanos()).unwrap(),
            }
        })
        .collect();
    let mut ledger = FrameBudgetLedger::new(FrameWorkBudget {
        upload_bytes: 4,
        ..FrameWorkBudget::default()
    });
    ledger.record_upload(5);
    let report = capture_feature_graph(
        &graph,
        GraphCaptureInputs {
            timing_source: Some(GraphTimingSource::Cpu),
            timings: &timings,
            work_budget: Some(&ledger),
            alias_heap_budget: Some(0),
        },
        GraphCaptureLimits::default(),
    )
    .unwrap();
    let critical = report
        .critical_path(GraphCaptureLimits::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        critical.total_duration_ns(),
        timings.iter().map(|t| t.duration_ns).sum()
    );
    assert!(report
        .budgets()
        .iter()
        .find(|b| b.category == "work.upload_bytes")
        .unwrap()
        .exceeded());
    assert!(report
        .budgets()
        .iter()
        .find(|b| b.category == "graph.alias_heap_bytes")
        .unwrap()
        .exceeded());
    assert_eq!(
        report
            .replay(GraphCaptureLimits::default())
            .unwrap()
            .nodes()
            .count(),
        4
    );
}

#[test]
fn missing_duplicate_unknown_and_overflowing_timings_are_honest() {
    let graph = graph();
    let l = GraphCaptureLimits::default();
    let one = [GraphPassTiming {
        pass: 0,
        duration_ns: 1,
    }];
    let input = GraphCaptureInputs {
        timings: &one,
        timing_source: Some(GraphTimingSource::Gpu),
        ..Default::default()
    };
    // Only verifies provenance handling; these synthetic weights are not GPU evidence.
    assert!(capture_feature_graph(&graph, input, l)
        .unwrap()
        .critical_path(l)
        .unwrap()
        .is_none());
    assert!(capture_feature_graph(
        &graph,
        GraphCaptureInputs {
            timing_source: None,
            ..input
        },
        l
    )
    .is_err());
    let duplicate = [one[0], one[0]];
    assert!(capture_feature_graph(
        &graph,
        GraphCaptureInputs {
            timings: &duplicate,
            ..input
        },
        l
    )
    .is_err());
    let unknown = [GraphPassTiming {
        pass: 4,
        duration_ns: 1,
    }];
    assert!(capture_feature_graph(
        &graph,
        GraphCaptureInputs {
            timings: &unknown,
            ..input
        },
        l
    )
    .is_err());
    let overflow: Vec<_> = (0..4)
        .map(|pass| GraphPassTiming {
            pass,
            duration_ns: u64::MAX,
        })
        .collect();
    assert_eq!(
        capture_feature_graph(
            &graph,
            GraphCaptureInputs {
                timings: &overflow,
                ..input
            },
            l
        ),
        Err(ProfilerError::ArithmeticOverflow)
    );
}

#[test]
fn every_capture_bound_rejects_before_returning_partial_report() {
    let graph = graph();
    let default = GraphCaptureLimits::default();
    for limits in [
        GraphCaptureLimits {
            passes: 3,
            ..default
        },
        GraphCaptureLimits {
            resources: 1,
            ..default
        },
        GraphCaptureLimits {
            accesses: 1,
            ..default
        },
        GraphCaptureLimits {
            dependencies: 0,
            ..default
        },
        GraphCaptureLimits {
            text_bytes: 1,
            ..default
        },
        GraphCaptureLimits {
            wire_bytes: 1,
            ..default
        },
    ] {
        assert!(capture_feature_graph(&graph, GraphCaptureInputs::default(), limits).is_err());
    }
    assert!(capture_feature_graph(
        &graph,
        GraphCaptureInputs {
            alias_heap_budget: Some(10),
            ..Default::default()
        },
        GraphCaptureLimits {
            budgets: 0,
            ..default
        }
    )
    .is_err());
}

#[test]
fn deterministic_wire_rejects_truncation_trailing_data_unknown_version_and_oversized_lengths() {
    let l = GraphCaptureLimits::default();
    let report = capture();
    let bytes = report.to_bytes(l).unwrap();
    assert_eq!(bytes, capture().to_bytes(l).unwrap());
    assert_eq!(
        GraphCaptureReport::from_bytes(&bytes, l)
            .unwrap()
            .to_bytes(l)
            .unwrap(),
        bytes
    );
    for end in 0..bytes.len() {
        assert!(
            GraphCaptureReport::from_bytes(&bytes[..end], l).is_err(),
            "offset {end}"
        );
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(GraphCaptureReport::from_bytes(&bad, l).is_err());
    let mut bad = bytes.clone();
    bad[4] = 2;
    assert!(GraphCaptureReport::from_bytes(&bad, l).is_err());
    let mut bad = bytes.clone();
    bad[40..48].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(GraphCaptureReport::from_bytes(&bad, l).is_err());
    assert!(GraphCaptureReport::from_bytes(
        &bytes,
        GraphCaptureLimits {
            wire_bytes: bytes.len() - 1,
            ..l
        }
    )
    .is_err());
    assert!(
        GraphCaptureReport::from_bytes(&bytes, GraphCaptureLimits { resources: 1, ..l }).is_err()
    );
    assert!(report
        .to_bytes(GraphCaptureLimits {
            wire_bytes: bytes.len() - 1,
            ..l
        })
        .is_err());
    assert!(report
        .to_bytes(GraphCaptureLimits {
            wire_bytes: bytes.len(),
            ..l
        })
        .is_ok());
}
