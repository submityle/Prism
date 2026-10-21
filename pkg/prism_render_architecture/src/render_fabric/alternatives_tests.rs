use super::*;
use crate::{backend::capability::CapabilitySet, frame_graph::ResourceLifetime};

const FEATURE: FeatureId = FeatureId(900);
const RESOURCE: ResourceKey = ResourceKey::of(FEATURE, 0);
const FIRST: PassKey = PassKey(901);
const SECOND: PassKey = PassKey(902);
const THIRD: PassKey = PassKey(903);
const NO_ALTERNATIVE_GROUPS: &[&[PassKey]] = &[];
const FIRST_SECOND_GROUP: &[PassKey] = &[FIRST, SECOND];
const FIRST_SECOND_GROUPS: &[&[PassKey]] = &[FIRST_SECOND_GROUP];

fn table<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}

fn pass(id: PassKey, input: ResourceInput) -> PassTemplate {
    PassTemplate {
        id,
        queue: QueueClass::Compute,
        depends_on: &[],
        accesses: table(vec![ResourceUse {
            key: RESOURCE,
            input,
            access: AccessKind::StorageWrite,
            scope: ResourceScope::Whole,
            frame_lag: None,
        }]),
    }
}

fn variant(id: u32, passes: Vec<PassTemplate>) -> FeatureVariant {
    FeatureVariant {
        id: VariantId(id),
        rank: (id - 900) as u16,
        min_quality: RenderQualityTier::Low,
        required_caps: CapabilitySet::EMPTY,
        requires_services_extra: ServiceSet::EMPTY,
        passes: table(passes),
        shader_permutation: None,
    }
}

fn definition(groups: &'static [&'static [PassKey]]) -> FeatureDefinition {
    FeatureDefinition::subsystem(
        FEATURE,
        RenderDomain::Surface,
        ServiceSet::EMPTY,
        table(vec![variant(
            900,
            vec![
                pass(FIRST, ResourceInput::Initial),
                pass(SECOND, ResourceInput::Initial),
            ],
        )]),
    )
    .with_pass_alternatives(groups)
}

fn catalog(definition: FeatureDefinition) -> Result<FeatureCatalog, CatalogError> {
    FeatureCatalog::new(table(vec![definition]), &[RESOURCE])
}

#[test]
fn alternative_group_requires_two_distinct_disjoint_members() {
    for group in [&[][..], &[FIRST][..]] {
        assert_eq!(
            catalog(definition(table(vec![group]))).unwrap_err(),
            CatalogError::InvalidPassAlternativeGroup {
                feature: FEATURE,
                group: 0
            }
        );
    }
    for groups in [
        table(vec![&[FIRST, FIRST][..]]),
        table(vec![&[FIRST, SECOND][..], &[SECOND, THIRD][..]]),
    ] {
        assert!(matches!(
            catalog(definition(groups)),
            Err(CatalogError::DuplicateAlternativePass {
                feature: FEATURE,
                ..
            })
        ));
    }
}

#[test]
fn alternative_group_rejects_unknown_and_partial_variant_members() {
    assert_eq!(
        catalog(definition(&[&[FIRST, THIRD]])).unwrap_err(),
        CatalogError::UnknownAlternativePass {
            feature: FEATURE,
            pass: THIRD
        }
    );
    let mut feature = definition(&[&[FIRST, SECOND]]);
    feature.variants = table(vec![
        feature.variants[0],
        variant(901, vec![pass(FIRST, ResourceInput::Initial)]),
    ]);
    assert_eq!(
        catalog(feature).unwrap_err(),
        CatalogError::IncompleteAlternativeGroup {
            feature: FEATURE,
            variant: VariantId(901),
            missing: SECOND
        }
    );
}

#[test]
fn alternative_group_allows_variants_without_any_group_member() {
    let mut feature = definition(&[&[FIRST, SECOND]]);
    feature.variants = table(vec![feature.variants[0], variant(901, vec![])]);
    assert!(catalog(feature).is_ok());
}

#[test]
fn alternative_writers_still_require_explicit_resource_inputs() {
    let mut feature = definition(&[&[FIRST, SECOND]]);
    feature.variants = table(vec![variant(
        900,
        vec![
            pass(FIRST, ResourceInput::Automatic),
            pass(SECOND, ResourceInput::Initial),
        ],
    )]);
    assert_eq!(
        catalog(feature).unwrap_err(),
        CatalogError::AmbiguousWriter {
            feature: FEATURE,
            variant: VariantId(900),
            resource: RESOURCE,
        }
    );
}

#[test]
fn alternative_group_does_not_hide_unrelated_writer_forks() {
    for input in [
        ResourceInput::Initial,
        ResourceInput::ProducedBy(PassKey(904)),
    ] {
        let mut feature = definition(&[&[FIRST, SECOND]]);
        feature.variants = table(vec![variant(
            900,
            vec![pass(FIRST, input), pass(SECOND, input), pass(THIRD, input)],
        )]);
        assert_eq!(
            catalog(feature).unwrap_err(),
            CatalogError::AmbiguousWriter {
                feature: FEATURE,
                variant: VariantId(900),
                resource: RESOURCE,
            }
        );
    }
}

#[test]
fn alternative_group_does_not_hide_duplicate_writes_within_one_pass() {
    let mut first = pass(FIRST, ResourceInput::Initial);
    first.accesses = table(vec![first.accesses[0], first.accesses[0]]);
    let mut feature = definition(&[&[FIRST, SECOND]]);
    feature.variants = table(vec![variant(
        900,
        vec![first, pass(SECOND, ResourceInput::Initial)],
    )]);
    assert_eq!(
        catalog(feature).unwrap_err(),
        CatalogError::AmbiguousWriter {
            feature: FEATURE,
            variant: VariantId(900),
            resource: RESOURCE,
        }
    );
}

#[test]
fn alternative_paths_compile_individually_without_resolving_selection_again() {
    let catalog = catalog(definition(&[&[FIRST, SECOND]])).unwrap();
    let roots = FrozenRoots::from_build_profile(&catalog, &[FEATURE], &[]).unwrap();
    let selection = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::EMPTY,
        RenderQualityTier::Low,
    )
    .unwrap();
    let frozen = selection.clone();
    let templates = ResourceTemplates::new(
        &catalog,
        &[ResourceTemplate {
            key: RESOURCE,
            lifetime: ResourceLifetime::Transient,
            layout: ResourceLayout::Buffer {
                bytes: 64,
                usage: BufferUsages::STORAGE,
            },
            alignment: 16,
        }],
    )
    .unwrap();
    for active in [&[][..], &[FIRST][..], &[SECOND][..]] {
        let graph = templates
            .compile_active(
                &selection,
                10,
                &[],
                &[],
                active,
                &[],
                GraphLimits::default(),
            )
            .unwrap();
        assert_eq!(graph.passes().len(), active.len());
        if let Some(&key) = active.first() {
            assert_eq!(graph.passes()[0].key(), key);
        }
        assert_eq!(selection, frozen);
    }
    let expected = FabricCompileError::ConflictingAlternativePasses {
        feature: FEATURE,
        first: FIRST,
        second: SECOND,
    };
    assert_eq!(
        templates
            .compile(&selection, 10, &[], &[], GraphLimits::default())
            .unwrap_err(),
        expected
    );
    for (active, fulfilled) in [
        (&[FIRST, SECOND][..], &[][..]),
        (&[FIRST][..], &[SECOND][..]),
        (&[][..], &[FIRST, SECOND][..]),
    ] {
        assert_eq!(
            templates
                .compile_active(
                    &selection,
                    10,
                    &[],
                    &[],
                    active,
                    fulfilled,
                    GraphLimits::default(),
                )
                .unwrap_err(),
            expected
        );
    }
}

#[test]
fn alternative_paths_conflict_even_without_shared_resources() {
    let mut feature = definition(&[&[FIRST, SECOND]]);
    let mut passes = feature.variants[0].passes.to_vec();
    for pass in &mut passes {
        pass.accesses = &[];
    }
    feature.variants = table(vec![variant(900, passes)]);
    let catalog = catalog(feature).unwrap();
    let roots = FrozenRoots::from_build_profile(&catalog, &[FEATURE], &[]).unwrap();
    let selection = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::EMPTY,
        RenderQualityTier::Low,
    )
    .unwrap();
    let templates = ResourceTemplates::new(
        &catalog,
        &[ResourceTemplate {
            key: RESOURCE,
            lifetime: ResourceLifetime::Transient,
            layout: ResourceLayout::Buffer {
                bytes: 64,
                usage: BufferUsages::STORAGE,
            },
            alignment: 16,
        }],
    )
    .unwrap();
    assert_eq!(
        templates
            .compile_active(
                &selection,
                10,
                &[],
                &[],
                &[FIRST, SECOND],
                &[],
                GraphLimits::default(),
            )
            .unwrap_err(),
        FabricCompileError::ConflictingAlternativePasses {
            feature: FEATURE,
            first: FIRST,
            second: SECOND,
        }
    );
}

#[test]
fn alternative_declaration_changes_invalidate_a_frozen_selection() {
    // Both catalogs share the exact same variant pointer. Changing only the
    // execution exclusivity contract must still reject the old provenance.
    let mut feature = definition(&[]);
    let passes = feature.variants[0]
        .passes
        .iter()
        .map(|pass| PassTemplate {
            accesses: &[],
            ..*pass
        })
        .collect();
    feature.variants = table(vec![variant(900, passes)]);
    for (original_groups, replacement_groups) in [
        (NO_ALTERNATIVE_GROUPS, FIRST_SECOND_GROUPS),
        (FIRST_SECOND_GROUPS, NO_ALTERNATIVE_GROUPS),
    ] {
        feature.pass_alternatives = original_groups;
        let original = catalog(feature).unwrap();
        let roots = FrozenRoots::from_build_profile(&original, &[FEATURE], &[]).unwrap();
        let selection = FeatureSelection::resolve(
            &original,
            &roots,
            CapabilitySet::EMPTY,
            RenderQualityTier::Low,
        )
        .unwrap();
        feature.pass_alternatives = replacement_groups;
        let replacement = catalog(feature).unwrap();
        let templates = ResourceTemplates::new(
            &replacement,
            &[ResourceTemplate {
                key: RESOURCE,
                lifetime: ResourceLifetime::Transient,
                layout: ResourceLayout::Buffer {
                    bytes: 64,
                    usage: BufferUsages::STORAGE,
                },
                alignment: 16,
            }],
        )
        .unwrap();
        assert_eq!(
            templates
                .compile_active(
                    &selection,
                    10,
                    &[],
                    &[],
                    &[FIRST],
                    &[],
                    GraphLimits::default()
                )
                .unwrap_err(),
            FabricCompileError::ForeignSelection(FEATURE)
        );
    }
}
