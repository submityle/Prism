use super::*;
use crate::backend::capability::Capability;

const A: FeatureId = FeatureId(100);
const B: FeatureId = FeatureId(200);
const ROOT: FeatureId = FeatureId(1);
const RESPONSE: FeatureId = FeatureId(2);
const SA: ServiceId = ServiceId(10);
const SB: ServiceId = ServiceId(20);
const KA: ResourceKey = ResourceKey::of(A, 0);
const KB: ResourceKey = ResourceKey::of(B, 0);

fn table<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}
fn services(values: &[ServiceId]) -> ServiceSet {
    ServiceSet::new(table(values.to_vec()))
}

#[test]
fn service_set_equality_ignores_declaration_order() {
    assert_eq!(services(&[SA, SB]), services(&[SB, SA]));
    assert_ne!(services(&[SA, SB]), services(&[SA, SA]));
    assert_ne!(services(&[SA, SA]), services(&[SA, SB]));
}
fn usage(key: ResourceKey, access: AccessKind) -> ResourceUse {
    ResourceUse {
        key,
        access,
        scope: ResourceScope::Whole,
        frame_lag: None,
    }
}
fn pass(id: u32, accesses: Vec<ResourceUse>) -> PassTemplate {
    PassTemplate {
        id: PassKey(id),
        accesses: table(accesses),
        queue: QueueClass::Compute,
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
fn provider(
    id: FeatureId,
    service: ServiceId,
    resource: ResourceKey,
    requires: &[ServiceId],
) -> FeatureDefinition {
    FeatureDefinition::service(
        id,
        table(vec![ServiceOutput {
            service,
            resources: table(vec![resource]),
        }]),
        services(requires),
        table(vec![variant(
            id.0,
            0,
            vec![pass(id.0, vec![usage(resource, AccessKind::StorageWrite)])],
        )]),
    )
}
fn root(requires: &[ServiceId]) -> FeatureDefinition {
    FeatureDefinition::subsystem(
        ROOT,
        RenderDomain::Surface,
        services(requires),
        table(vec![variant(1, 0, vec![pass(1, vec![])])]),
    )
}
fn response(requires: &[ServiceId]) -> FeatureDefinition {
    FeatureDefinition::response(
        RESPONSE,
        services(requires),
        table(vec![variant(2, 0, vec![])]),
    )
}
fn catalog(
    features: Vec<FeatureDefinition>,
    keys: Vec<ResourceKey>,
) -> Result<FeatureCatalog, CatalogError> {
    FeatureCatalog::new(table(features), table(keys))
}
fn roots(catalog: &FeatureCatalog) -> FrozenRoots {
    FrozenRoots::from_build_profile(catalog, &[ROOT], &[]).unwrap()
}
fn resolve(catalog: &FeatureCatalog) -> FeatureSelection {
    FeatureSelection::resolve(
        catalog,
        &roots(catalog),
        CapabilitySet::EMPTY,
        RenderQualityTier::Low,
    )
    .unwrap()
}
fn error(features: Vec<FeatureDefinition>, keys: Vec<ResourceKey>) -> CatalogError {
    catalog(features, keys).unwrap_err()
}
fn replace_variants(
    mut feature: FeatureDefinition,
    variants: Vec<FeatureVariant>,
) -> FeatureDefinition {
    feature.variants = table(variants);
    feature
}

#[test]
fn feature_kind_matches_domain_presence() {
    for kind in [
        FeatureKind::Subsystem,
        FeatureKind::Response,
        FeatureKind::Service { provides: &[] },
    ] {
        for domain in [None, Some(RenderDomain::Surface)] {
            let definition = FeatureDefinition {
                kind,
                domain,
                ..root(&[])
            };
            if matches!(kind, FeatureKind::Subsystem) == domain.is_some() {
                assert!(catalog(vec![definition], vec![]).is_ok());
            } else {
                assert_eq!(
                    error(vec![definition], vec![]),
                    CatalogError::KindDomainMismatch(ROOT)
                );
            }
        }
    }
}

#[test]
fn constructors_and_explicit_keys_are_const() {
    const SUB: FeatureDefinition =
        FeatureDefinition::subsystem(ROOT, RenderDomain::Surface, ServiceSet::EMPTY, &[]);
    const SVC: FeatureDefinition = FeatureDefinition::service(A, &[], ServiceSet::EMPTY, &[]);
    const RESP: FeatureDefinition = FeatureDefinition::response(RESPONSE, ServiceSet::EMPTY, &[]);
    assert_eq!(SUB.domain, Some(RenderDomain::Surface));
    assert_eq!(SVC.domain, None);
    assert_eq!(RESP.domain, None);
    assert_eq!(KA.owner(), ResourceOwner::Feature(A));
    assert_eq!(KA.slot(), 0);
    assert_ne!(KA, KB);
    assert_ne!(KA, ResourceKey::engine(0));
}

#[test]
fn duplicate_feature_and_variant_ids_are_errors() {
    assert_eq!(
        error(vec![root(&[]), root(&[])], vec![]),
        CatalogError::DuplicateFeature(ROOT)
    );
    let mut other = response(&[]);
    other.variants = root(&[]).variants;
    assert_eq!(
        error(vec![root(&[]), other], vec![]),
        CatalogError::DuplicateVariant(VariantId(1))
    );
}

#[test]
fn duplicate_rank_and_empty_variants_are_errors() {
    let definition = replace_variants(
        root(&[]),
        vec![variant(1, 0, vec![]), variant(2, 0, vec![])],
    );
    assert_eq!(
        error(vec![definition], vec![]),
        CatalogError::DuplicateRank {
            feature: ROOT,
            rank: 0
        }
    );
    assert_eq!(
        error(vec![replace_variants(root(&[]), vec![])], vec![]),
        CatalogError::NoVariants(ROOT)
    );
}

#[test]
fn resource_keys_are_namespaced_and_unique() {
    assert!(catalog(
        vec![root(&[])],
        vec![ResourceKey::of(ROOT, 0), ResourceKey::engine(0)]
    )
    .is_ok());
    assert_eq!(
        error(vec![root(&[])], vec![KA]),
        CatalogError::UnknownResourceOwner(KA)
    );
    assert_eq!(
        error(vec![root(&[])], vec![ResourceKey::engine(0); 2]),
        CatalogError::DuplicateResource(ResourceKey::engine(0))
    );
}

#[test]
fn undeclared_pass_and_output_resource_references_fail() {
    let definition = replace_variants(
        root(&[]),
        vec![variant(
            1,
            0,
            vec![pass(1, vec![usage(KA, AccessKind::SampledRead)])],
        )],
    );
    assert_eq!(
        error(vec![definition], vec![]),
        CatalogError::UndeclaredResource(KA)
    );
    assert_eq!(
        error(vec![provider(A, SA, KA, &[])], vec![]),
        CatalogError::UndeclaredResource(KA)
    );
}

#[test]
fn every_service_id_has_exactly_one_provider_feature() {
    assert_eq!(
        error(
            vec![provider(A, SA, KA, &[]), provider(B, SA, KB, &[])],
            vec![KA, KB]
        ),
        CatalogError::DuplicateProvider {
            service: SA,
            first: A,
            second: B
        }
    );
    let valid = catalog(vec![provider(A, SA, KA, &[])], vec![KA]).unwrap();
    assert_eq!(valid.provider(SA), Some(A));
    let service_provider = valid.service_provider(SA).unwrap();
    assert_eq!(service_provider.service(), SA);
    assert_eq!(service_provider.feature(), A);
    assert_eq!(service_provider.output().resources, &[KA]);
    assert_eq!(valid.service_output(SA), Some(service_provider.output()));
    assert_eq!(valid.provider(SB), None);
}

#[test]
fn duplicate_service_in_same_provider_is_error() {
    let mut feature = provider(A, SA, KA, &[]);
    feature.kind = FeatureKind::Service {
        provides: table(vec![feature.kind.provides()[0]; 2]),
    };
    assert_eq!(
        error(vec![feature], vec![KA]),
        CatalogError::DuplicateProvider {
            service: SA,
            first: A,
            second: A
        }
    );
}

#[test]
fn missing_hard_and_soft_providers_are_static_errors() {
    assert_eq!(
        error(vec![root(&[SA])], vec![]),
        CatalogError::MissingProvider {
            feature: ROOT,
            requirement: RequirementKind::Hard,
            service: SA
        }
    );
    let mut v = variant(1, 0, vec![]);
    v.requires_services_extra = services(&[SA]);
    assert_eq!(
        error(vec![replace_variants(root(&[]), vec![v])], vec![]),
        CatalogError::MissingProvider {
            feature: ROOT,
            requirement: RequirementKind::Extra(VariantId(1)),
            service: SA
        }
    );
}

#[test]
fn duplicate_requirements_are_errors_not_registration_order() {
    let svc = provider(A, SA, KA, &[]);
    assert_eq!(
        error(vec![root(&[SA, SA]), svc], vec![KA]),
        CatalogError::DuplicateRequirement {
            feature: ROOT,
            requirement: RequirementKind::Hard,
            service: SA
        }
    );
    let mut v = variant(1, 0, vec![]);
    v.requires_services_extra = services(&[SA, SA]);
    assert_eq!(
        error(vec![replace_variants(root(&[]), vec![v]), svc], vec![KA]),
        CatalogError::DuplicateRequirement {
            feature: ROOT,
            requirement: RequirementKind::Extra(VariantId(1)),
            service: SA
        }
    );
}

#[test]
fn feature_dependency_kinds_are_stratified() {
    // All three legal source kinds can require the sole Service provider.
    let valid = catalog(
        vec![
            provider(A, SA, KA, &[]),
            provider(B, SB, KB, &[SA]),
            root(&[SB]),
            response(&[SA]),
        ],
        vec![KA, KB],
    )
    .unwrap();
    let frozen = FrozenRoots::from_build_profile(&valid, &[ROOT], &[RESPONSE]).unwrap();
    let selection = FeatureSelection::resolve(
        &valid,
        &frozen,
        CapabilitySet::EMPTY,
        RenderQualityTier::Low,
    )
    .unwrap();
    assert_eq!(
        selection
            .selected()
            .iter()
            .map(SelectedFeature::feature_id)
            .collect::<Vec<_>>(),
        vec![A, B, ROOT, RESPONSE]
    );
    // Non-service targets cannot publish SA at all; every illegal matrix cell
    // fails the same real provider lookup, not a separate boolean gate.
    for source_kind in [
        FeatureKind::Subsystem,
        FeatureKind::Response,
        FeatureKind::Service { provides: &[] },
    ] {
        for target_kind in [FeatureKind::Subsystem, FeatureKind::Response] {
            let source = FeatureDefinition {
                kind: source_kind,
                domain: matches!(source_kind, FeatureKind::Subsystem)
                    .then_some(RenderDomain::Surface),
                ..root(&[SA])
            };
            let target = FeatureDefinition {
                id: A,
                kind: target_kind,
                domain: matches!(target_kind, FeatureKind::Subsystem)
                    .then_some(RenderDomain::Volume),
                ..response(&[])
            };
            assert_eq!(
                error(vec![source, target], vec![]),
                CatalogError::MissingProvider {
                    feature: ROOT,
                    requirement: RequirementKind::Hard,
                    service: SA
                }
            );
        }
    }
}

#[test]
fn hard_self_and_mutual_cycles_use_authoritative_graph_rejection() {
    assert_eq!(
        error(vec![provider(A, SA, KA, &[SA])], vec![KA]),
        CatalogError::CyclicHardDependencies
    );
    assert_eq!(
        error(
            vec![provider(A, SA, KA, &[SB]), provider(B, SB, KB, &[SA])],
            vec![KA, KB]
        ),
        CatalogError::CyclicHardDependencies
    );
}

#[test]
fn hard_cycle_outside_active_roots_is_still_static_error() {
    assert_eq!(
        error(
            vec![
                root(&[]),
                provider(A, SA, KA, &[SB]),
                provider(B, SB, KB, &[SA])
            ],
            vec![KA, KB]
        ),
        CatalogError::CyclicHardDependencies
    );
}

#[test]
fn every_service_output_resource_is_written_by_every_provider_variant() {
    let good = variant(
        100,
        0,
        vec![pass(100, vec![usage(KA, AccessKind::StorageWrite)])],
    );
    for bad in [
        variant(101, 1, vec![]),
        variant(
            101,
            1,
            vec![pass(101, vec![usage(KA, AccessKind::SampledRead)])],
        ),
    ] {
        let definition = replace_variants(provider(A, SA, KA, &[]), vec![good, bad]);
        assert_eq!(
            error(vec![definition], vec![KA]),
            CatalogError::MissingOutputWrite {
                feature: A,
                variant: VariantId(101),
                service: SA,
                resource: KA
            }
        );
    }
}

#[test]
fn service_variants_publish_the_same_resource_interface() {
    let definition = replace_variants(
        provider(A, SA, KA, &[]),
        vec![
            variant(
                100,
                0,
                vec![pass(100, vec![usage(KA, AccessKind::StorageWrite)])],
            ),
            variant(
                101,
                1,
                vec![pass(101, vec![usage(KA, AccessKind::TransferWrite)])],
            ),
        ],
    );
    let valid = catalog(vec![root(&[SA]), definition], vec![KA]).unwrap();
    let selection = resolve(&valid);
    assert_eq!(selection.service_resources(SA), Some([KA].as_slice()));
    assert_eq!(selection.service_provider(SA), valid.service_provider(SA));
    assert_eq!(selection.service_output(SA), valid.service_output(SA));
    assert_eq!(
        selection.services().collect::<Vec<_>>(),
        vec![valid.service_provider(SA).unwrap()]
    );
    let mut broken = definition;
    broken.variants = table(vec![
        definition.variants[0],
        variant(
            101,
            1,
            vec![pass(101, vec![usage(KB, AccessKind::StorageWrite)])],
        ),
    ]);
    assert_eq!(
        error(vec![broken, provider(B, SB, KB, &[])], vec![KA, KB]),
        CatalogError::MissingOutputWrite {
            feature: A,
            variant: VariantId(101),
            service: SA,
            resource: KA
        }
    );
}

#[test]
fn all_write_access_kinds_satisfy_output_contract() {
    for access in [
        AccessKind::StorageWrite,
        AccessKind::ColorAttachment,
        AccessKind::DepthAttachment,
        AccessKind::TransferWrite,
    ] {
        let mut output_pass = pass(100, vec![usage(KA, access)]);
        output_pass.queue = match access {
            AccessKind::ColorAttachment | AccessKind::DepthAttachment => QueueClass::Graphics,
            AccessKind::StorageWrite | AccessKind::TransferWrite => QueueClass::Compute,
            _ => unreachable!(),
        };
        let feature = replace_variants(
            provider(A, SA, KA, &[]),
            vec![variant(100, 0, vec![output_pass])],
        );
        assert!(catalog(vec![feature], vec![KA]).is_ok());
    }
}

#[test]
fn queue_access_combinations_are_rejected_before_runtime_lowering() {
    let key = ResourceKey::of(ROOT, 0);
    let mut transfer_storage = pass(100, vec![usage(key, AccessKind::StorageWrite)]);
    transfer_storage.queue = QueueClass::Transfer;
    assert_eq!(
        error(
            vec![replace_variants(
                root(&[]),
                vec![variant(100, 0, vec![transfer_storage])]
            )],
            vec![key]
        ),
        CatalogError::UnsupportedQueueAccess {
            pass: PassKey(100),
            queue: QueueClass::Transfer,
            access: AccessKind::StorageWrite,
        }
    );

    let mut compute_attachment = pass(101, vec![usage(key, AccessKind::ColorAttachment)]);
    compute_attachment.queue = QueueClass::Compute;
    assert_eq!(
        error(
            vec![replace_variants(
                root(&[]),
                vec![variant(101, 0, vec![compute_attachment])]
            )],
            vec![key]
        ),
        CatalogError::UnsupportedQueueAccess {
            pass: PassKey(101),
            queue: QueueClass::Compute,
            access: AccessKind::ColorAttachment,
        }
    );
}

#[test]
fn ambiguous_writes_and_duplicate_passes_are_rejected() {
    let writer = pass(100, vec![usage(KA, AccessKind::StorageWrite)]);
    let feature = replace_variants(
        provider(A, SA, KA, &[]),
        vec![variant(
            100,
            0,
            vec![writer, pass(101, writer.accesses.to_vec())],
        )],
    );
    assert_eq!(
        error(vec![feature], vec![KA]),
        CatalogError::AmbiguousWriter {
            feature: A,
            variant: VariantId(100),
            resource: KA
        }
    );
    let feature = replace_variants(
        provider(A, SA, KA, &[]),
        vec![variant(100, 0, vec![writer, writer])],
    );
    assert_eq!(
        error(vec![feature], vec![KA]),
        CatalogError::DuplicatePass {
            variant: VariantId(100),
            pass: PassKey(100)
        }
    );
}

#[test]
fn pass_keys_allow_identical_alternatives_but_not_conflicts() {
    let shared = pass(100, vec![usage(KA, AccessKind::StorageWrite)]);
    let feature = replace_variants(
        provider(A, SA, KA, &[]),
        vec![variant(100, 0, vec![shared]), variant(101, 1, vec![shared])],
    );
    assert!(catalog(vec![feature], vec![KA]).is_ok());
    let mut conflict = shared;
    conflict.queue = QueueClass::Transfer;
    let feature = replace_variants(
        feature,
        vec![
            variant(100, 0, vec![shared]),
            variant(101, 1, vec![conflict]),
        ],
    );
    assert_eq!(
        error(vec![feature], vec![KA]),
        CatalogError::ConflictingPassKey(PassKey(100))
    );
    let cross_owner = replace_variants(root(&[]), vec![variant(1, 0, vec![shared])]);
    assert_eq!(
        error(vec![provider(A, SA, KA, &[]), cross_owner], vec![KA]),
        CatalogError::ConflictingPassKey(PassKey(100))
    );
}

#[test]
fn output_resource_duplicates_are_rejected() {
    let mut feature = provider(A, SA, KA, &[]);
    feature.kind = FeatureKind::Service {
        provides: table(vec![ServiceOutput {
            service: SA,
            resources: table(vec![KA, KA]),
        }]),
    };
    assert_eq!(
        error(vec![feature], vec![KA]),
        CatalogError::DuplicateOutputResource {
            feature: A,
            service: SA,
            resource: KA
        }
    );
}

#[test]
fn frame_lag_is_positive_read_only_declaration() {
    assert_eq!(FrameLag::new(0), None);
    assert_eq!(FrameLag::new(1).unwrap().frames(), 1);
    let mut read = usage(ResourceKey::engine(0), AccessKind::SampledRead);
    read.frame_lag = FrameLag::new(1);
    assert!(catalog(
        vec![replace_variants(
            root(&[]),
            vec![variant(1, 0, vec![pass(1, vec![read])])]
        )],
        vec![read.key]
    )
    .is_ok());
    let write = ResourceUse {
        access: AccessKind::StorageWrite,
        ..read
    };
    assert_eq!(
        error(
            vec![replace_variants(
                root(&[]),
                vec![variant(1, 0, vec![pass(1, vec![write])])]
            )],
            vec![write.key]
        ),
        CatalogError::HistoryWrite {
            variant: VariantId(1),
            pass: PassKey(1),
            resource: write.key
        }
    );
}

#[test]
fn shader_binding_reuses_public_permutation_space() {
    use crate::shader_package::{PermutationKey, PermutationSpace};
    let space = Box::leak(Box::new(PermutationSpace::new([2, 3])));
    for (coords, valid) in [(vec![1, 2], true), (vec![2, 0], false), (vec![1], false)] {
        let key = Box::leak(Box::new(PermutationKey::new(coords)));
        let mut implementation = variant(1, 0, vec![]);
        implementation.shader_permutation = Some(ShaderPermutationBinding { space, key });
        let result = catalog(
            vec![replace_variants(root(&[]), vec![implementation])],
            vec![],
        );
        if valid {
            assert!(result.is_ok());
        } else {
            assert_eq!(
                result.unwrap_err(),
                CatalogError::InvalidShaderPermutation(VariantId(1))
            );
        }
    }
}

#[test]
fn hard_transitive_closure_and_selected_providers() {
    let valid = catalog(
        vec![
            root(&[SB]),
            provider(B, SB, KB, &[SA]),
            provider(A, SA, KA, &[]),
        ],
        vec![KB, KA],
    )
    .unwrap();
    let result = resolve(&valid);
    assert_eq!(result.closure(), &[ROOT, A, B]);
    assert_eq!(
        result
            .selected()
            .iter()
            .map(SelectedFeature::feature_id)
            .collect::<Vec<_>>(),
        vec![A, B, ROOT]
    );
    assert_eq!(result.provider(SA), Some(A));
    assert_eq!(result.provider(SB), Some(B));
}

#[test]
fn active_roots_are_static_and_frozen() {
    let mut unused = provider(B, SB, KB, &[]);
    let mut unusable = unused.variants[0];
    unusable.required_caps = CapabilitySet::from_slice(&[Capability::RayQuery]);
    unused.variants = table(vec![unusable]);
    let valid = catalog(
        vec![root(&[SA]), provider(A, SA, KA, &[]), unused, response(&[])],
        vec![KA, KB],
    )
    .unwrap();
    let result = resolve(&valid);
    assert_eq!(result.closure(), &[ROOT, A]);
    assert!(result.feature(B).is_none());
    assert_eq!(result.roots().require_root(ROOT), Ok(()));
    assert_eq!(
        result.roots().require_root(RESPONSE),
        Err(RootError::OutsideFrozenRoots(RESPONSE))
    );
    assert_eq!(
        result.roots().require_root(A),
        Err(RootError::OutsideFrozenRoots(A))
    );
    assert_eq!(result.roots().ids(), &[ROOT]);
}

#[test]
fn frozen_roots_validate_static_source_kinds_and_unknowns() {
    let valid = catalog(
        vec![root(&[]), response(&[]), provider(A, SA, KA, &[])],
        vec![KA],
    )
    .unwrap();
    assert_eq!(
        FrozenRoots::from_build_profile(&valid, &[A], &[]).unwrap_err(),
        RootError::ExpectedSubsystem(A)
    );
    assert_eq!(
        FrozenRoots::from_build_profile(&valid, &[], &[ROOT]).unwrap_err(),
        RootError::ExpectedResponse(ROOT)
    );
    assert_eq!(
        FrozenRoots::from_build_profile(&valid, &[FeatureId(999)], &[]).unwrap_err(),
        RootError::UnknownFeature(FeatureId(999))
    );
    let roots =
        FrozenRoots::from_build_profile(&valid, &[ROOT, ROOT], &[RESPONSE, RESPONSE]).unwrap();
    assert_eq!(roots.ids(), &[ROOT, RESPONSE]);
    let other = catalog(vec![response(&[])], vec![]).unwrap();
    assert_eq!(
        FeatureSelection::resolve(&other, &roots, CapabilitySet::EMPTY, RenderQualityTier::Low)
            .unwrap_err(),
        SelectionError::Roots(RootError::UnknownFeature(ROOT))
    );
}

#[test]
fn empty_profile_has_empty_selection_even_if_features_need_caps() {
    let valid = catalog(vec![root(&[]), provider(A, SA, KA, &[])], vec![KA]).unwrap();
    let roots = FrozenRoots::from_build_profile(&valid, &[], &[]).unwrap();
    let result =
        FeatureSelection::resolve(&valid, &roots, CapabilitySet::EMPTY, RenderQualityTier::Low)
            .unwrap();
    assert!(result.closure().is_empty());
    assert!(result.selected().is_empty());
    assert_eq!(result.provider(SA), None);
}

#[test]
fn variant_extra_requirements_do_not_grow_the_closure() {
    let baseline = variant(
        101,
        1,
        vec![pass(101, vec![usage(KA, AccessKind::TransferWrite)])],
    );
    let mut preferred = variant(
        100,
        0,
        vec![pass(100, vec![usage(KA, AccessKind::StorageWrite)])],
    );
    let mut results = Vec::new();
    for extra in [ServiceSet::EMPTY, services(&[SB])] {
        preferred.requires_services_extra = extra;
        let feature = replace_variants(provider(A, SA, KA, &[]), vec![preferred, baseline]);
        let valid = catalog(
            vec![root(&[SA]), feature, provider(B, SB, KB, &[])],
            vec![KA, KB],
        )
        .unwrap();
        results.push(resolve(&valid));
    }
    assert_eq!(results[0].closure(), results[1].closure());
    assert_eq!(results[0].closure(), &[ROOT, A]);
    assert_eq!(results[0].feature(A).unwrap().variant_id(), VariantId(100));
    assert_eq!(results[1].feature(A).unwrap().variant_id(), VariantId(101));
    assert_eq!(results[1].provider(SB), None);
}

#[test]
fn soft_provider_in_closure_but_not_yet_selected_does_not_reorder() {
    let mut preferred = variant(
        100,
        0,
        vec![pass(100, vec![usage(KA, AccessKind::StorageWrite)])],
    );
    preferred.requires_services_extra = services(&[SB]);
    let feature = replace_variants(
        provider(A, SA, KA, &[]),
        vec![preferred, variant(101, 1, preferred.passes.to_vec())],
    );
    let valid = catalog(
        vec![root(&[SA, SB]), feature, provider(B, SB, KB, &[])],
        vec![KA, KB],
    )
    .unwrap();
    let result = resolve(&valid);
    assert_eq!(result.feature(A).unwrap().variant_id(), VariantId(101));
    assert_eq!(result.provider(SB), Some(B));
    assert_eq!(result.selected()[0].feature_id(), A);
}

#[test]
fn soft_provider_already_selected_admits_candidate() {
    let mut preferred = variant(
        200,
        0,
        vec![pass(200, vec![usage(KB, AccessKind::StorageWrite)])],
    );
    preferred.requires_services_extra = services(&[SA]);
    let feature = replace_variants(
        provider(B, SB, KB, &[]),
        vec![preferred, variant(201, 1, preferred.passes.to_vec())],
    );
    let valid = catalog(
        vec![root(&[SA, SB]), provider(A, SA, KA, &[]), feature],
        vec![KA, KB],
    )
    .unwrap();
    assert_eq!(
        resolve(&valid).feature(B).unwrap().variant_id(),
        VariantId(200)
    );
}

#[test]
fn soft_cycles_are_admission_only_not_hard_edges() {
    let mut a = provider(A, SA, KA, &[]);
    let mut b = provider(B, SB, KB, &[]);
    let mut va = a.variants[0];
    va.requires_services_extra = services(&[SB]);
    let mut vb = b.variants[0];
    vb.requires_services_extra = services(&[SA]);
    a.variants = table(vec![va, variant(101, 1, va.passes.to_vec())]);
    b.variants = table(vec![vb, variant(201, 1, vb.passes.to_vec())]);
    let valid = catalog(vec![root(&[SA, SB]), a, b], vec![KA, KB]).unwrap();
    let result = resolve(&valid);
    assert_eq!(result.feature(A).unwrap().variant_id(), VariantId(101));
    assert_eq!(result.feature(B).unwrap().variant_id(), VariantId(200));
}

#[test]
fn capability_quality_matrix_selects_minimum_admissible_rank() {
    let mut fast = variant(
        100,
        0,
        vec![pass(100, vec![usage(KA, AccessKind::StorageWrite)])],
    );
    fast.required_caps = CapabilitySet::from_slice(&[Capability::RayQuery]);
    fast.min_quality = RenderQualityTier::High;
    let mut mid = variant(
        101,
        1,
        vec![pass(101, vec![usage(KA, AccessKind::TransferWrite)])],
    );
    mid.required_caps = CapabilitySet::from_slice(&[Capability::Compute]);
    mid.min_quality = RenderQualityTier::Medium;
    let mut low = variant(
        102,
        2,
        vec![pass(102, vec![usage(KA, AccessKind::ColorAttachment)])],
    );
    low.passes = table(vec![PassTemplate {
        queue: QueueClass::Graphics,
        ..low.passes[0]
    }]);
    let valid = catalog(
        vec![
            root(&[SA]),
            replace_variants(provider(A, SA, KA, &[]), vec![low, fast, mid]),
        ],
        vec![KA],
    )
    .unwrap();
    for caps in [
        CapabilitySet::EMPTY,
        CapabilitySet::WEBGPU_BASELINE,
        CapabilitySet::from_slice(&[Capability::Compute, Capability::RayQuery]),
    ] {
        for quality in [
            RenderQualityTier::Low,
            RenderQualityTier::Medium,
            RenderQualityTier::High,
            RenderQualityTier::Ultra,
        ] {
            let result = FeatureSelection::resolve(&valid, &roots(&valid), caps, quality).unwrap();
            let expected = if caps.contains_all(fast.required_caps) && quality >= fast.min_quality {
                fast.id
            } else if caps.contains_all(mid.required_caps) && quality >= mid.min_quality {
                mid.id
            } else {
                low.id
            };
            assert_eq!(result.feature(A).unwrap().variant_id(), expected);
            assert_eq!(result.device_caps(), caps);
            assert_eq!(result.quality(), quality);
            assert_eq!(
                result
                    .passes()
                    .filter(|(id, _)| *id == A)
                    .map(|(_, pass)| pass.id)
                    .collect::<Vec<_>>(),
                vec![PassKey(expected.0)]
            );
        }
    }
}

#[test]
fn no_candidate_reports_caps_quality_and_unselected_services() {
    let mut implementation = variant(1, 0, vec![]);
    implementation.required_caps = CapabilitySet::from_slice(&[Capability::RayQuery]);
    implementation.min_quality = RenderQualityTier::Ultra;
    implementation.requires_services_extra = services(&[SA]);
    let valid = catalog(
        vec![
            replace_variants(root(&[]), vec![implementation]),
            provider(A, SA, KA, &[]),
        ],
        vec![KA],
    )
    .unwrap();
    assert_eq!(
        FeatureSelection::resolve(
            &valid,
            &roots(&valid),
            CapabilitySet::EMPTY,
            RenderQualityTier::Low
        )
        .unwrap_err(),
        SelectionError::NoAdmissibleVariant {
            feature: ROOT,
            rejected: vec![CandidateRejection {
                variant: VariantId(1),
                missing_caps: implementation.required_caps,
                required_quality: Some(RenderQualityTier::Ultra),
                unavailable_services: vec![SA]
            }],
        }
    );
}

#[test]
fn variant_selection_is_deterministic_and_registration_order_independent() {
    let mut a = provider(A, SA, KA, &[]);
    let va = a.variants[0];
    a.variants = table(vec![
        variant(
            101,
            1,
            vec![pass(101, vec![usage(KA, AccessKind::TransferWrite)])],
        ),
        va,
    ]);
    let features = vec![
        root(&[SA, SB]),
        response(&[SB]),
        a,
        provider(B, SB, KB, &[SA]),
    ];
    let original = catalog(features.clone(), vec![KA, KB]).unwrap();
    let frozen = FrozenRoots::from_build_profile(&original, &[ROOT], &[RESPONSE]).unwrap();
    let expected = FeatureSelection::resolve(
        &original,
        &frozen,
        CapabilitySet::EMPTY,
        RenderQualityTier::Low,
    )
    .unwrap();
    // Exhaust all 4! registration orders, also reverse variant declarations.
    for first in 0..4 {
        for second in 0..4 {
            for third in 0..4 {
                for fourth in 0..4 {
                    let indices = [first, second, third, fourth];
                    if indices
                        .iter()
                        .copied()
                        .collect::<alloc::collections::BTreeSet<_>>()
                        .len()
                        != 4
                    {
                        continue;
                    }
                    let mut shuffled: Vec<_> =
                        indices.iter().map(|&index| features[index]).collect();
                    for feature in &mut shuffled {
                        let mut variants = feature.variants.to_vec();
                        variants.reverse();
                        feature.variants = table(variants);
                    }
                    let reordered = catalog(shuffled, vec![KB, KA]).unwrap();
                    let result = FeatureSelection::resolve(
                        &reordered,
                        &frozen,
                        CapabilitySet::EMPTY,
                        RenderQualityTier::Low,
                    )
                    .unwrap();
                    assert_eq!(result, expected);
                    for selected in result.selected() {
                        let feature = reordered.feature(selected.feature_id()).unwrap();
                        for &required in feature
                            .requires_services
                            .as_slice()
                            .iter()
                            .chain(selected.variant().requires_services_extra.as_slice())
                        {
                            assert_eq!(result.provider(required), reordered.provider(required));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn inserting_unused_declarations_preserves_existing_explicit_ids() {
    let valid = catalog(vec![root(&[SA]), provider(A, SA, KA, &[])], vec![KA]).unwrap();
    let larger = catalog(
        vec![
            response(&[]),
            root(&[SA]),
            provider(B, SB, KB, &[]),
            provider(A, SA, KA, &[]),
        ],
        vec![KB, KA, ResourceKey::engine(77)],
    )
    .unwrap();
    assert_eq!(resolve(&valid), resolve(&larger));
    assert_eq!(
        larger.resources().collect::<Vec<_>>(),
        vec![ResourceKey::engine(77), KA, KB]
    );
}

#[test]
fn service_set_superset_is_computed_from_members() {
    assert!(services(&[SA, SB]).is_superset(services(&[SA])));
    assert!(!services(&[SA]).is_superset(services(&[SB])));
}
