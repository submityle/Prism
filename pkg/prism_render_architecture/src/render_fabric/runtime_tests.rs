use super::*;
use crate::frame_graph::{AccessKind, QueueClass, ResourceId, ResourceLifetime};
use prism_render_driver::{BufferId, BufferUsages, Extent3d};

const FEATURE: FeatureId = FeatureId(900);
const RESOURCE: ResourceKey = ResourceKey::of(FEATURE, 0);

fn table<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}

fn fixture() -> (FeatureCatalog, ResourceTemplate) {
    let pass = PassTemplate {
        id: PassKey(1),
        accesses: table(vec![ResourceUse {
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
            vec![RuntimePassBinding::new(PassKey(1), |context| {
                assert_eq!(context.pass().key(), PassKey(1));
                assert_eq!(context.pass().domain(), Some(RenderDomain::Surface));
                assert_eq!(context.accesses().len(), 1);
                assert!(context.buffer(0).is_some());
                assert_eq!(context.items().count(), 0);
            })],
        )
        .unwrap();
    assert_eq!(runtime.plan().pass_count(), 1);
    assert_eq!(runtime.declaration().passes()[0].key(), PassKey(1));
}

#[test]
fn runtime_rejects_unexpected_import_without_mutating_graph() {
    let compiled = compiled();
    let import = RuntimeImport::Buffer {
        resource: ResourceId(99),
        imported: prism_render_graph::ImportedBuffer {
            buffer: BufferId::from_parts(1, 0),
            entry_state: prism_render_driver::BufferState::initial(),
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
