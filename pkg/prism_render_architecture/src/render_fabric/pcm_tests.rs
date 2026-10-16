use super::*;
use crate::backend::capability::Capability as DeviceCapability;
use prism_material_pcm::{
    Capability as PcmCapability, CapabilitySet as PcmCapabilities, ExtensionManifest,
    ExtensionRegistry, Kind as PcmKind, Namespace, Stability,
};
use prism_render_graph::ExecutionPlan;

const PROVIDER: FeatureId = FeatureId(901);
const RESPONSE: FeatureId = FeatureId(902);
const SERVICE: ServiceId = ServiceId(901);
const INPUT: ResourceKey = ResourceKey::of(PROVIDER, 0);
const OUTPUT: ResourceKey = ResourceKey::of(RESPONSE, 0);
const PROVIDER_UID: &str = "org.prism.fixture.provider";
const RESPONSE_UID: &str = "org.prism.fixture.response";
const KEYS: &[ResourceKey] = &[INPUT, OUTPUT];
const BINDINGS: &[PcmFeatureBinding] = &[
    PcmFeatureBinding {
        uid: PROVIDER_UID,
        feature: PROVIDER,
        kind: PcmKind::Provider,
        version: 1,
    },
    PcmFeatureBinding {
        uid: RESPONSE_UID,
        feature: RESPONSE,
        kind: PcmKind::ResponseMode,
        version: 1,
    },
];

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
fn variant(
    id: u32,
    rank: u16,
    caps: CapabilitySet,
    key: u32,
    accesses: Vec<ResourceUse>,
) -> FeatureVariant {
    FeatureVariant {
        id: VariantId(id),
        rank,
        required_caps: caps,
        min_quality: RenderQualityTier::Low,
        requires_services_extra: ServiceSet::EMPTY,
        shader_permutation: None,
        passes: table(vec![PassTemplate {
            depends_on: &[],
            id: PassKey(key),
            queue: QueueClass::Compute,
            accesses: table(accesses),
        }]),
    }
}
fn definitions(caps: CapabilitySet) -> &'static [FeatureDefinition] {
    table(vec![
        FeatureDefinition::service(
            PROVIDER,
            &[ServiceOutput {
                service: SERVICE,
                resources: &[INPUT],
            }],
            ServiceSet::EMPTY,
            table(vec![
                variant(
                    911,
                    0,
                    caps,
                    901,
                    vec![usage(INPUT, AccessKind::StorageWrite)],
                ),
                variant(
                    912,
                    1,
                    caps,
                    903,
                    vec![usage(INPUT, AccessKind::TransferWrite)],
                ),
            ]),
        ),
        FeatureDefinition::response(
            RESPONSE,
            ServiceSet::new(&[SERVICE]),
            table(vec![variant(
                913,
                0,
                CapabilitySet::EMPTY,
                902,
                vec![
                    usage(INPUT, AccessKind::StorageRead),
                    usage(OUTPUT, AccessKind::StorageWrite),
                ],
            )]),
        ),
    ])
}
fn registry(caps: PcmCapabilities) -> ExtensionRegistry {
    let mut result = ExtensionRegistry::new();
    result
        .register(
            ExtensionManifest::new_identity_only(
                PROVIDER_UID,
                Namespace::Studio,
                PcmKind::Provider,
            )
            .with_requires_caps(caps),
        )
        .unwrap();
    result
        .register_new(RESPONSE_UID, Namespace::Studio, PcmKind::ResponseMode)
        .unwrap();
    result
}
fn resolve(
    bridge: &PcmFeatureCatalog,
    caps: CapabilitySet,
) -> Result<FeatureSelection, SelectionError> {
    let roots = FrozenRoots::from_build_profile(bridge.catalog(), &[], &[RESPONSE]).unwrap();
    FeatureSelection::resolve(bridge.catalog(), &roots, caps, RenderQualityTier::Low)
}

#[test]
fn explicit_capability_conversion_does_not_reinterpret_pcm_bits_or_erase_requirements() {
    assert_eq!(
        pcm_device_requirements(PcmCapabilities::EMPTY),
        Ok(CapabilitySet::EMPTY)
    );
    let bindless =
        pcm_device_requirements(PcmCapabilities::of(PcmCapability::BindlessTextures)).unwrap();
    assert_eq!(
        bindless,
        CapabilitySet::from_slice(&[DeviceCapability::BindlessDescriptors])
    );
    assert_ne!(bindless.bits(), PcmCapability::BindlessTextures.bit());
    assert_eq!(
        pcm_device_requirements(PcmCapabilities::of(PcmCapability::ComputeScatter)),
        Ok(CapabilitySet::WEBGPU_BASELINE)
    );
    let both = PcmCapabilities::of(PcmCapability::BindlessTextures)
        .union(PcmCapabilities::of(PcmCapability::ComputeScatter));
    assert_eq!(
        pcm_device_requirements(both),
        Ok(bindless.union(CapabilitySet::WEBGPU_BASELINE))
    );
    for capability in [
        PcmCapability::VirtualTexture,
        PcmCapability::ObjectSpaceCache,
        PcmCapability::StorageF16,
        PcmCapability::HeavySampling,
    ] {
        assert_eq!(
            pcm_device_requirements(PcmCapabilities::of(capability)),
            Err(PcmCapabilityError::Unrepresentable(capability))
        );
    }
    assert_eq!(
        pcm_device_requirements(PcmCapabilities::from_bits(1 << 31)),
        Err(PcmCapabilityError::UnknownBits(1 << 31))
    );
}

#[test]
fn bridge_preserves_authority_and_extracts_reads_writes_from_every_authored_variant() {
    let registry = registry(PcmCapabilities::EMPTY);
    let before = registry.iter().copied().collect::<Vec<_>>();
    let bridge =
        PcmFeatureCatalog::new(&registry, BINDINGS, definitions(CapabilitySet::EMPTY), KEYS)
            .unwrap();
    assert_eq!(registry.iter().copied().collect::<Vec<_>>(), before);
    assert_eq!(bridge.manifest(PROVIDER_UID).unwrap().registry_slot(), None);
    assert_eq!(
        bridge.manifest(RESPONSE_UID).unwrap().registry_slot(),
        registry.slot_of_uid(RESPONSE_UID)
    );
    let provider = bridge.feature(PROVIDER_UID).unwrap();
    assert_eq!(provider.id, PROVIDER);
    assert!(provider.reads().next().is_none());
    assert_eq!(
        provider
            .writes()
            .map(|use_| (use_.variant, use_.pass, use_.usage.key, use_.usage.access))
            .collect::<Vec<_>>(),
        vec![
            (
                VariantId(911),
                PassKey(901),
                INPUT,
                AccessKind::StorageWrite
            ),
            (
                VariantId(912),
                PassKey(903),
                INPUT,
                AccessKind::TransferWrite
            ),
        ]
    );
    assert_eq!(
        bridge
            .feature(RESPONSE_UID)
            .unwrap()
            .reads()
            .next()
            .unwrap()
            .usage
            .key,
        INPUT
    );
    assert_eq!(
        bridge
            .feature(RESPONSE_UID)
            .unwrap()
            .writes()
            .next()
            .unwrap()
            .usage
            .key,
        OUTPUT
    );
    assert_eq!(
        resolve(&bridge, CapabilitySet::WEBGPU_BASELINE)
            .unwrap()
            .selected()
            .iter()
            .map(|selected| selected.variant_id())
            .collect::<Vec<_>>(),
        vec![VariantId(911), VariantId(913)]
    );
    assert_eq!(
        bridge.bindings().collect::<Vec<_>>(),
        vec![(PROVIDER_UID, PROVIDER), (RESPONSE_UID, RESPONSE)]
    );
    assert!(bridge.feature("not.registered").is_none());
}

#[test]
fn pcm_requirements_are_checked_on_inactive_variants_and_enforced_by_the_existing_resolver() {
    let registry = registry(PcmCapabilities::of(PcmCapability::BindlessTextures));
    let required = CapabilitySet::from_slice(&[DeviceCapability::BindlessDescriptors]);
    let mut specs = definitions(required).to_vec();
    let mut variants = specs[0].variants.to_vec();
    variants[1].required_caps = CapabilitySet::EMPTY;
    specs[0].variants = table(variants);
    assert_eq!(
        PcmFeatureCatalog::new(&registry, BINDINGS, table(specs), KEYS).unwrap_err(),
        PcmBridgeError::MissingVariantRequirements {
            uid: PROVIDER_UID,
            feature: PROVIDER,
            variant: VariantId(912),
            missing: required,
        }
    );
    let bridge = PcmFeatureCatalog::new(&registry, BINDINGS, definitions(required), KEYS).unwrap();
    match resolve(&bridge, CapabilitySet::WEBGPU_BASELINE).unwrap_err() {
        SelectionError::NoAdmissibleVariant { feature, rejected } => {
            assert_eq!(feature, PROVIDER);
            assert_eq!(rejected.len(), 2);
            assert!(rejected
                .iter()
                .all(|candidate| candidate.missing_caps == required));
        }
        error => panic!("unexpected error: {error:?}"),
    }
    resolve(&bridge, CapabilitySet::WEBGPU_BASELINE.union(required)).unwrap();
    let unsupported = self::registry(PcmCapabilities::of(PcmCapability::StorageF16));
    assert_eq!(
        PcmFeatureCatalog::new(&unsupported, BINDINGS, definitions(required), KEYS).unwrap_err(),
        PcmBridgeError::Capabilities {
            uid: PROVIDER_UID,
            error: PcmCapabilityError::Unrepresentable(PcmCapability::StorageF16),
        }
    );
}

#[test]
fn promotion_registration_order_and_header_slots_cannot_change_semantic_feature_identity() {
    let definitions = definitions(CapabilitySet::EMPTY);
    let mut first = registry(PcmCapabilities::EMPTY);
    let original = PcmFeatureCatalog::new(&first, BINDINGS, definitions, KEYS).unwrap();
    let before = resolve(&original, CapabilitySet::WEBGPU_BASELINE).unwrap();
    let old_display = original.manifest(PROVIDER_UID).unwrap().id();
    first
        .promote(
            PROVIDER_UID,
            Namespace::Ext,
            "cross_vendor",
            Stability::Candidate,
        )
        .unwrap();
    first
        .promote(
            PROVIDER_UID,
            Namespace::Khr,
            "specification",
            Stability::Stable,
        )
        .unwrap();
    let promoted = PcmFeatureCatalog::new(&first, BINDINGS, definitions, KEYS).unwrap();
    assert_ne!(old_display, promoted.manifest(PROVIDER_UID).unwrap().id());
    assert_eq!(
        resolve(&promoted, CapabilitySet::WEBGPU_BASELINE).unwrap(),
        before
    );
    assert_eq!(
        original.manifest(PROVIDER_UID).unwrap().namespace(),
        Namespace::Studio
    );
    assert_eq!(
        promoted.manifest(PROVIDER_UID).unwrap().namespace(),
        Namespace::Khr
    );
    let mut second = ExtensionRegistry::new();
    second
        .register_new(
            "org.prism.fixture.unmapped",
            Namespace::Studio,
            PcmKind::Channel,
        )
        .unwrap();
    second
        .register_new(RESPONSE_UID, Namespace::Studio, PcmKind::ResponseMode)
        .unwrap();
    second
        .register_identity_only(PROVIDER_UID, Namespace::Studio, PcmKind::Provider)
        .unwrap();
    assert_ne!(
        second.slot_of_uid(RESPONSE_UID),
        first.slot_of_uid(RESPONSE_UID)
    );
    let reordered = PcmFeatureCatalog::new(
        &second,
        table(BINDINGS.iter().copied().rev().collect()),
        definitions,
        KEYS,
    )
    .unwrap();
    assert_eq!(
        resolve(&reordered, CapabilitySet::WEBGPU_BASELINE).unwrap(),
        before
    );
    assert_eq!(
        reordered.bindings().collect::<Vec<_>>(),
        original.bindings().collect::<Vec<_>>()
    );
    assert!(reordered.feature("org.prism.fixture.unmapped").is_none());
}

#[test]
fn bridge_rejects_duplicate_unknown_kind_and_version_identity_relations() {
    let registry = registry(PcmCapabilities::EMPTY);
    let defs = definitions(CapabilitySet::EMPTY);
    assert_eq!(
        PcmFeatureCatalog::new(&registry, table(vec![BINDINGS[0], BINDINGS[0]]), defs, KEYS)
            .unwrap_err(),
        PcmBridgeError::DuplicateUid(PROVIDER_UID)
    );
    let alternate = PcmFeatureBinding {
        feature: PROVIDER,
        ..BINDINGS[1]
    };
    assert_eq!(
        PcmFeatureCatalog::new(&registry, table(vec![BINDINGS[0], alternate]), defs, KEYS)
            .unwrap_err(),
        PcmBridgeError::DuplicateFeature(PROVIDER)
    );
    for (binding, expected) in [
        (
            PcmFeatureBinding {
                uid: "org.prism.fixture.unknown",
                ..BINDINGS[0]
            },
            PcmBridgeError::UnknownUid("org.prism.fixture.unknown"),
        ),
        (
            PcmFeatureBinding {
                feature: FeatureId(999),
                ..BINDINGS[0]
            },
            PcmBridgeError::UnknownFeature(FeatureId(999)),
        ),
        (
            PcmFeatureBinding {
                kind: PcmKind::Composite,
                ..BINDINGS[0]
            },
            PcmBridgeError::KindMismatch {
                uid: PROVIDER_UID,
                expected: PcmKind::Composite,
                actual: PcmKind::Provider,
            },
        ),
        (
            PcmFeatureBinding {
                version: 2,
                ..BINDINGS[0]
            },
            PcmBridgeError::VersionMismatch {
                uid: PROVIDER_UID,
                expected: 2,
                actual: 1,
            },
        ),
    ] {
        assert_eq!(
            PcmFeatureCatalog::new(&registry, table(vec![binding]), defs, KEYS).unwrap_err(),
            expected
        );
    }
    let wrong_feature_kind = PcmFeatureBinding {
        feature: RESPONSE,
        ..BINDINGS[0]
    };
    assert_eq!(
        PcmFeatureCatalog::new(&registry, table(vec![wrong_feature_kind]), defs, KEYS).unwrap_err(),
        PcmBridgeError::FeatureKindMismatch {
            uid: PROVIDER_UID,
            feature: RESPONSE
        }
    );
}

#[test]
fn only_graph_extension_kinds_bridge_and_material_lobes_remain_pcm_owned() {
    let mut registry = registry(PcmCapabilities::EMPTY);
    registry
        .register_identity_only(
            "org.prism.fixture.composite",
            Namespace::Ext,
            PcmKind::Composite,
        )
        .unwrap();
    let bridge = PcmFeatureCatalog::new(
        &registry,
        &[PcmFeatureBinding {
            uid: "org.prism.fixture.composite",
            feature: RESPONSE,
            kind: PcmKind::Composite,
            version: 1,
        }],
        definitions(CapabilitySet::EMPTY),
        KEYS,
    )
    .unwrap();
    assert_eq!(
        bridge.feature("org.prism.fixture.composite").unwrap().kind,
        FeatureKind::Response
    );
    registry
        .register_new("org.prism.fixture.lobe", Namespace::Ext, PcmKind::Lobe)
        .unwrap();
    assert_eq!(
        PcmFeatureCatalog::new(
            &registry,
            &[PcmFeatureBinding {
                uid: "org.prism.fixture.lobe",
                feature: RESPONSE,
                kind: PcmKind::Lobe,
                version: 1
            }],
            definitions(CapabilitySet::EMPTY),
            KEYS
        )
        .unwrap_err(),
        PcmBridgeError::NotGraphExtension {
            uid: "org.prism.fixture.lobe",
            kind: PcmKind::Lobe
        }
    );
}

#[test]
fn manifest_sidecar_catalog_selection_pass_spec_execution_is_one_chain() {
    use crate::frame_graph::{Barrier, ResourceLifetime};
    struct Sink {
        passes: Vec<(FeatureId, VariantId, PassKey)>,
        barriers: Vec<Barrier>,
    }
    impl FeatureGraphSink for Sink {
        type Error = ();
        fn begin(
            &mut self,
            resources: &[BoundResource],
            _: &[FrameRenderItem],
            plan: &ExecutionPlan,
        ) -> Result<(), ()> {
            assert_eq!(resources.len(), 2);
            assert_eq!(plan.declared_pass_count, 2);
            Ok(())
        }
        fn barrier(&mut self, barrier: &Barrier) -> Result<(), ()> {
            self.barriers.push(*barrier);
            Ok(())
        }
        fn pass(&mut self, pass: &PassSpec) -> Result<(), ()> {
            self.passes
                .push((pass.feature(), pass.variant(), pass.key()));
            assert!(pass.accesses().iter().all(|access| access.ssa_version == 1));
            Ok(())
        }
        fn finish(&mut self, _: &ExecutionPlan) -> Result<(), ()> {
            Ok(())
        }
    }
    let bridge = PcmFeatureCatalog::new(
        &registry(PcmCapabilities::EMPTY),
        BINDINGS,
        definitions(CapabilitySet::EMPTY),
        KEYS,
    )
    .unwrap();
    let selection = resolve(&bridge, CapabilitySet::WEBGPU_BASELINE).unwrap();
    let templates = KEYS
        .iter()
        .map(|&key| ResourceTemplate {
            key,
            lifetime: ResourceLifetime::Transient,
            alignment: 16,
            layout: ResourceLayout::Buffer {
                bytes: 64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            },
        })
        .collect::<Vec<_>>();
    let graph = ResourceTemplates::new(bridge.catalog(), &templates)
        .unwrap()
        .compile(&selection, 10, &[], &[], GraphLimits::default())
        .unwrap();
    let expected_barriers = graph.barriers().to_vec();
    assert_eq!(expected_barriers.len(), 1);
    let mut sink = Sink {
        passes: Vec::new(),
        barriers: Vec::new(),
    };
    graph.execute(&mut sink).unwrap();
    assert_eq!(
        sink.passes,
        vec![
            (PROVIDER, VariantId(911), PassKey(901)),
            (RESPONSE, VariantId(913), PassKey(902))
        ]
    );
    assert_eq!(sink.barriers, expected_barriers);
}
