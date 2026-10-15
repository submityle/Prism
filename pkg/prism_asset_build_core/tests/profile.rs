#[path = "../src/profile.rs"]
mod profile;

use profile::*;
use std::collections::BTreeSet;

fn schema() -> ProfileSchema {
    ProfileSchema::new(
        "test.profile",
        1,
        [
            (
                "quality",
                ProfileField::new(
                    ProfileFieldType::Unsigned { min: 0, max: 10 },
                    ProfileMergePolicy::ScalarOverride,
                )
                .required(true),
            ),
            (
                "compression",
                ProfileField::new(
                    ProfileFieldType::String {
                        min_bytes: 1,
                        max_bytes: 8,
                        allowed_values: Some(BTreeSet::from(["none".into(), "zstd".into()])),
                    },
                    ProfileMergePolicy::ScalarOverride,
                ),
            ),
            (
                "features",
                ProfileField::new(
                    ProfileFieldType::StringSet {
                        min_items: 0,
                        max_items: 4,
                        allowed_values: Some(BTreeSet::from([
                            "bc7".into(),
                            "astc".into(),
                            "meshlet".into(),
                        ])),
                    },
                    ProfileMergePolicy::SetUnion,
                ),
            ),
            (
                "replace_features",
                ProfileField::new(
                    ProfileFieldType::StringSet {
                        min_items: 0,
                        max_items: 4,
                        allowed_values: None,
                    },
                    ProfileMergePolicy::SetReplace,
                ),
            ),
            (
                "supports_gpu",
                ProfileField::new(ProfileFieldType::Bool, ProfileMergePolicy::ScalarOverride),
            ),
            (
                "memory_mb",
                ProfileField::new(
                    ProfileFieldType::Unsigned {
                        min: 256,
                        max: 65_536,
                    },
                    ProfileMergePolicy::ScalarOverride,
                ),
            ),
        ],
    )
    .unwrap()
}

fn layer(
    layer: ProfileLayer,
    values: impl IntoIterator<Item = (&'static str, ProfileValue)>,
) -> ProfileLayerValues {
    let mut result = ProfileLayerValues::new(layer);
    for (name, value) in values {
        result.set(name, value).unwrap();
    }
    result
}

fn baseline() -> ProfileLayerValues {
    layer(
        ProfileLayer::EngineBaseline,
        [
            ("quality", ProfileValue::Unsigned(3)),
            ("compression", ProfileValue::String("none".into())),
            ("features", ProfileValue::string_set(["meshlet"])),
            ("replace_features", ProfileValue::string_set(["base"])),
            ("supports_gpu", ProfileValue::Bool(false)),
            ("memory_mb", ProfileValue::Unsigned(1024)),
        ],
    )
}

#[test]
fn seven_layer_precedence_is_fixed_and_input_order_independent() {
    let schema = schema();
    let quality = layer(
        ProfileLayer::BuildConfiguration,
        [("quality", ProfileValue::Unsigned(7))],
    );
    let capabilities = layer(
        ProfileLayer::PlatformCapabilities,
        [
            ("supports_gpu", ProfileValue::Bool(true)),
            ("features", ProfileValue::string_set(["bc7"])),
        ],
    );
    let group = layer(
        ProfileLayer::AssetGroupOverride,
        [("quality", ProfileValue::Unsigned(5))],
    );
    let asset = layer(
        ProfileLayer::AssetOverride,
        [("quality", ProfileValue::Unsigned(9))],
    );

    let forward = schema
        .resolve(&[
            baseline(),
            capabilities.clone(),
            quality.clone(),
            group.clone(),
            asset.clone(),
        ])
        .unwrap();
    let reverse = schema
        .resolve(&[asset, group, quality, capabilities, baseline()])
        .unwrap();
    assert_eq!(forward, reverse);
    assert_eq!(
        forward.get("quality").unwrap(),
        Some(&ProfileValue::Unsigned(9))
    );
    assert_eq!(
        forward.get("supports_gpu").unwrap(),
        Some(&ProfileValue::Bool(true))
    );
    assert_eq!(
        forward.get("features").unwrap(),
        Some(&ProfileValue::string_set(["bc7", "meshlet"]))
    );
    assert_eq!(
        forward
            .explain("quality")
            .unwrap()
            .iter()
            .map(|c| c.layer)
            .collect::<Vec<_>>(),
        vec![
            ProfileLayer::EngineBaseline,
            ProfileLayer::BuildConfiguration,
            ProfileLayer::AssetGroupOverride,
            ProfileLayer::AssetOverride
        ]
    );
}

#[test]
fn set_union_and_replace_are_schema_declared() {
    let schema = schema();
    let union = schema
        .resolve(&[
            baseline(),
            layer(
                ProfileLayer::ProjectDefaults,
                [("features", ProfileValue::string_set(["bc7"]))],
            ),
            layer(
                ProfileLayer::AssetOverride,
                [
                    ("features", ProfileValue::string_set(["astc"])),
                    ("replace_features", ProfileValue::string_set(["asset"])),
                ],
            ),
            layer(
                ProfileLayer::AssetGroupOverride,
                [("replace_features", ProfileValue::string_set(["group"]))],
            ),
        ])
        .unwrap();
    assert_eq!(
        union.get("features").unwrap(),
        Some(&ProfileValue::string_set(["astc", "bc7", "meshlet"]))
    );
    assert_eq!(
        union.get("replace_features").unwrap(),
        Some(&ProfileValue::string_set(["asset"]))
    );
}

#[test]
fn duplicate_layers_unknown_fields_types_ranges_and_illegal_overrides_fail() {
    let schema = schema();
    assert!(matches!(
        schema.resolve(&[baseline(), baseline()]),
        Err(ProfileError::DuplicateLayer(ProfileLayer::EngineBaseline))
    ));
    let unknown = layer(
        ProfileLayer::AssetOverride,
        [("unknown", ProfileValue::Bool(true))],
    );
    assert!(matches!(
        schema.resolve(&[baseline(), unknown]),
        Err(ProfileError::UnknownField(_))
    ));
    let wrong_type = layer(
        ProfileLayer::AssetOverride,
        [("quality", ProfileValue::Integer(4))],
    );
    assert!(
        matches!(schema.resolve(&[baseline(), wrong_type]), Err(ProfileError::TypeMismatch { field, .. }) if field == "quality")
    );
    let out_of_range = layer(
        ProfileLayer::AssetOverride,
        [("memory_mb", ProfileValue::Unsigned(128))],
    );
    assert!(
        matches!(schema.resolve(&[baseline(), out_of_range]), Err(ProfileError::ConstraintViolation { field, .. }) if field == "memory_mb")
    );

    let immutable = ProfileSchema::new(
        "immutable",
        1,
        [(
            "toolchain",
            ProfileField::new(
                ProfileFieldType::String {
                    min_bytes: 1,
                    max_bytes: 20,
                    allowed_values: None,
                },
                ProfileMergePolicy::ScalarOverride,
            )
            .override_policy(ProfileOverridePolicy::ForbidChange),
        )],
    )
    .unwrap();
    let first = layer(
        ProfileLayer::EngineBaseline,
        [("toolchain", ProfileValue::String("v1".into()))],
    );
    let changed = layer(
        ProfileLayer::ProjectDefaults,
        [("toolchain", ProfileValue::String("v2".into()))],
    );
    assert!(
        matches!(immutable.resolve(&[first, changed]), Err(ProfileError::IllegalOverride { field, .. }) if field == "toolchain")
    );
}

#[test]
fn canonical_hashes_and_projection_hashes_are_order_independent() {
    let schema = schema();
    let a = schema.resolve(&[baseline()]).unwrap();
    let b = schema
        .resolve(&[layer(
            ProfileLayer::EngineBaseline,
            [
                ("memory_mb", ProfileValue::Unsigned(1024)),
                ("supports_gpu", ProfileValue::Bool(false)),
                ("replace_features", ProfileValue::string_set(["base"])),
                ("features", ProfileValue::string_set(["meshlet"])),
                ("compression", ProfileValue::String("none".into())),
                ("quality", ProfileValue::Unsigned(3)),
            ],
        )])
        .unwrap();
    assert_eq!(a.hash(), b.hash());
    assert_eq!(
        a.project(["quality", "compression"]).unwrap().hash(),
        b.project(["compression", "quality", "quality"])
            .unwrap()
            .hash()
    );
    assert_ne!(
        a.field_hash("quality").unwrap(),
        a.field_hash("memory_mb").unwrap()
    );
}

#[test]
fn unrelated_field_does_not_invalidate_declared_projection() {
    let schema = schema();
    let before = schema.resolve(&[baseline()]).unwrap();
    let after = schema
        .resolve(&[
            baseline(),
            layer(
                ProfileLayer::AssetOverride,
                [("memory_mb", ProfileValue::Unsigned(2048))],
            ),
        ])
        .unwrap();
    let quality_before = before.project(["quality"]).unwrap();
    let quality_after = after.project(["quality"]).unwrap();
    assert_eq!(quality_before.hash(), quality_after.hash());
    assert!(!quality_before
        .differences(&quality_after)
        .invalidates_cache());
    assert!(before.differences(&after).invalidates_cache());
    assert_eq!(
        before
            .differences(&after)
            .fields
            .iter()
            .map(|field| field.field.as_str())
            .collect::<Vec<_>>(),
        vec!["memory_mb"]
    );
}

#[test]
fn reader_projection_captures_actual_dependencies_and_missing_reads_are_explicit() {
    let profile = schema().resolve(&[baseline()]).unwrap();
    let mut reader = profile.reader();
    assert_eq!(
        reader.get("quality").unwrap(),
        Some(&ProfileValue::Unsigned(3))
    );
    let projection = reader.finish();
    assert_eq!(projection.fields().collect::<Vec<_>>(), vec!["quality"]);
    assert!(
        matches!(projection.get("memory_mb"), Err(ProfileError::UnknownField(field)) if field == "memory_mb")
    );
}

#[test]
fn capability_selection_is_explicit_deterministic_and_explainable() {
    let profile = schema()
        .resolve(&[
            baseline(),
            layer(
                ProfileLayer::PlatformCapabilities,
                [
                    ("supports_gpu", ProfileValue::Bool(true)),
                    ("features", ProfileValue::string_set(["bc7"])),
                    ("memory_mb", ProfileValue::Unsigned(4096)),
                ],
            ),
        ])
        .unwrap();
    let candidates = [
        CapabilityCandidate {
            id: "cpu-fallback".into(),
            priority: 1,
            requirements: vec![],
        },
        CapabilityCandidate {
            id: "gpu-bc7".into(),
            priority: 5,
            requirements: vec![
                CapabilityRequirement::Enabled("supports_gpu".into()),
                CapabilityRequirement::SetContains {
                    field: "features".into(),
                    member: "bc7".into(),
                },
            ],
        },
        CapabilityCandidate {
            id: "gpu-heavy".into(),
            priority: 6,
            requirements: vec![CapabilityRequirement::MinimumUnsigned {
                field: "memory_mb".into(),
                minimum: 8192,
            }],
        },
    ];
    let selection = profile.select_capability(&candidates).unwrap();
    assert_eq!(selection.selected.as_deref(), Some("gpu-bc7"));
    assert_eq!(
        selection.dependency_projection.fields().collect::<Vec<_>>(),
        vec!["features", "memory_mb", "supports_gpu"]
    );
    assert!(selection
        .evaluations
        .iter()
        .any(|candidate| candidate.id == "gpu-heavy" && !candidate.is_supported()));
    let reversed = profile
        .select_capability(&[
            candidates[2].clone(),
            candidates[0].clone(),
            candidates[1].clone(),
        ])
        .unwrap();
    assert_eq!(selection.hash(), reversed.hash());
}

#[test]
fn schema_and_candidate_errors_are_explicit() {
    assert!(matches!(
        ProfileSchema::new("bad name", 1, Vec::<(String, ProfileField)>::new()),
        Err(ProfileError::InvalidName(_))
    ));
    assert!(matches!(
        ProfileSchema::new(
            "bad",
            1,
            [(
                "x",
                ProfileField::new(ProfileFieldType::Bool, ProfileMergePolicy::SetUnion)
            )]
        ),
        Err(ProfileError::InvalidSchema(_))
    ));
    let profile = schema().resolve(&[baseline()]).unwrap();
    let duplicate = [
        CapabilityCandidate {
            id: "same".into(),
            priority: 1,
            requirements: vec![],
        },
        CapabilityCandidate {
            id: "same".into(),
            priority: 0,
            requirements: vec![],
        },
    ];
    assert!(
        matches!(profile.select_capability(&duplicate), Err(ProfileError::DuplicateCandidate(id)) if id == "same")
    );
}
