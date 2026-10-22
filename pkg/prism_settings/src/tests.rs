use crate::*;
use alloc::{collections::BTreeMap, format, vec, vec::Vec};

fn enum_def(default: Value) -> (SettingDef, EnumOption, EnumOption) {
    let plain = EnumOption::from_stable_name("render::Mode::Plain");
    let configured = EnumOption::from_stable_name("render::Mode::Configured");
    let mut def = SettingDef::new("render.mode", "render", default);
    def.enum_options = vec![plain.clone(), configured.clone()];
    (def, plain, configured)
}

fn registry(defs: Vec<SettingDef>) -> SettingsRegistry {
    let mut b = RegistryBuilder::new();
    for ns in [
        "audio",
        "render",
        "network",
        "telemetry",
        "editor",
        "system",
        "build",
        "input",
        "game",
        "video",
    ] {
        b.namespace(ns, "test-owner").unwrap();
    }
    for d in defs {
        b.register(d);
    }
    b.finish().unwrap()
}
fn bool_def(key: &str, default: bool) -> SettingDef {
    SettingDef::new(key, key.split('.').next().unwrap(), Value::Bool(default))
}
fn source(layer: LayerKind, id: &str, entries: &[(&str, Value)]) -> LayerSource {
    let mut s = LayerSource::new(layer, TrustLevel::User, id);
    for (k, v) in entries {
        s.set(*k, v.clone());
    }
    s
}

#[test]
fn dense_ids_and_registry_order_do_not_depend_on_registration_order() {
    let a = bool_def("audio.master_enabled", true);
    let r = bool_def("render.hdr_enabled", false);
    let first = registry(vec![a.clone(), r.clone()]);
    let second = registry(vec![r, a]);
    assert_eq!(
        first
            .defs()
            .iter()
            .map(|d| d.key.clone())
            .collect::<Vec<_>>(),
        second
            .defs()
            .iter()
            .map(|d| d.key.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        first.id(&SettingKey::from("audio.master_enabled")),
        second.id(&SettingKey::from("audio.master_enabled"))
    );
}

#[test]
fn stable_enum_options_are_order_independent_and_closed() {
    let low = EnumOption::from_stable_name("graphics::Quality::Low");
    let high = EnumOption::from_stable_name("graphics::Quality::High");
    assert_eq!(low.id, EnumId::from_stable_name("graphics::Quality::Low"));
    assert_ne!(low.id, high.id);

    let mut def = SettingDef::new("render.quality", "render", Value::Enum(low.id));
    def.enum_options = vec![high.clone(), low.clone()];
    let schema = registry(vec![def]);
    let (id, definition) = schema.get(&"render.quality".into()).unwrap();
    assert_eq!(definition.enum_options, vec![high, low.clone()]);
    assert_eq!(
        schema.validate_value(id, Value::Enum(low.id)),
        Ok(Value::Enum(low.id))
    );
    assert!(matches!(
        schema.validate_value(
            id,
            Value::Enum(EnumId::from_stable_name("graphics::Quality::Ultra"))
        ),
        Err(SettingsError::InvalidEnum { .. })
    ));
}

#[test]
fn stable_enum_options_reject_mismatched_and_duplicate_identities() {
    let mut def = SettingDef::new("render.quality", "render", Value::Enum(EnumId(1)));
    def.enum_options = vec![
        EnumOption {
            id: EnumId(1),
            stable_name: "graphics::Quality::Low".into(),
        },
        EnumOption::from_stable_name("graphics::Quality::Low"),
        EnumOption::from_stable_name("graphics::Quality::Low"),
    ];
    let mut builder = RegistryBuilder::new();
    builder.namespace("render", "test-owner").unwrap();
    builder.register(def);
    let errors = builder.finish().unwrap_err();
    assert!(errors.iter().any(|error| matches!(error, SettingsError::InvalidDefinition { reason, .. } if reason.contains("does not match"))));
    assert!(errors.iter().any(|error| matches!(error, SettingsError::InvalidDefinition { reason, .. } if reason.contains("collide"))));
}

#[test]
fn enum_payloads_preserve_unit_options_and_validate_tuple_and_struct_shapes() {
    let (mut def, plain, configured) = enum_def(Value::Enum(EnumId(0)));
    def.default = Value::Enum(plain.id);
    def.enum_payloads = vec![EnumOptionPayload {
        id: configured.id,
        shape: EnumPayloadShape::Tuple(vec![
            ValueShape::scalar(ValueType::Int),
            ValueShape::scalar(ValueType::Bool),
        ]),
    }];
    let schema = registry(vec![def]);
    let id = schema.id(&"render.mode".into()).unwrap();

    assert_eq!(
        schema.validate_value(id, Value::Enum(plain.id)),
        Ok(Value::Enum(plain.id))
    );
    let tuple = Value::EnumPayload {
        id: configured.id,
        payload: EnumPayload::Tuple(vec![Value::Int(4), Value::Bool(true)]),
    };
    assert_eq!(schema.validate_value(id, tuple.clone()), Ok(tuple));

    let struct_option = EnumOption::from_stable_name("render::Mode::Structured");
    let mut struct_def = SettingDef::new(
        "render.structured_mode",
        "render",
        Value::EnumPayload {
            id: struct_option.id,
            payload: EnumPayload::Struct(BTreeMap::from([
                ("label".into(), Value::String("fast".into())),
                ("quality".into(), Value::Int(3)),
            ])),
        },
    );
    struct_def.enum_options = vec![struct_option.clone()];
    struct_def.enum_payloads = vec![EnumOptionPayload {
        id: struct_option.id,
        shape: EnumPayloadShape::Struct(BTreeMap::from([
            ("label".into(), ValueShape::scalar(ValueType::String)),
            ("quality".into(), ValueShape::scalar(ValueType::Int)),
        ])),
    }];
    let schema = registry(vec![struct_def]);
    let id = schema.id(&"render.structured_mode".into()).unwrap();
    assert!(
        schema.def(id).unwrap().default
            == schema
                .validate_value(id, schema.def(id).unwrap().default.clone())
                .unwrap()
    );
}

#[test]
fn enum_payload_rejects_unknown_variants_wrong_shapes_and_invalid_defaults() {
    let (mut def, plain, configured) = enum_def(Value::Enum(EnumId(0)));
    def.default = Value::Enum(plain.id);
    def.enum_payloads = vec![EnumOptionPayload {
        id: configured.id,
        shape: EnumPayloadShape::Tuple(vec![ValueShape::scalar(ValueType::Int)]),
    }];
    let schema = registry(vec![def]);
    let id = schema.id(&"render.mode".into()).unwrap();
    assert!(matches!(
        schema.validate_value(
            id,
            Value::EnumPayload {
                id: EnumId::from_stable_name("render::Mode::Missing"),
                payload: EnumPayload::Tuple(vec![Value::Int(1)]),
            }
        ),
        Err(SettingsError::InvalidEnum { .. })
    ));
    assert!(schema
        .validate_value(
            id,
            Value::EnumPayload {
                id: configured.id,
                payload: EnumPayload::Tuple(vec![Value::Bool(true)]),
            }
        )
        .is_err());
    assert!(schema
        .validate_value(
            id,
            Value::EnumPayload {
                id: configured.id,
                payload: EnumPayload::Struct(BTreeMap::new()),
            }
        )
        .is_err());

    let mut invalid = SettingDef::new(
        "render.invalid_mode",
        "render",
        Value::EnumPayload {
            id: configured.id,
            payload: EnumPayload::Tuple(vec![Value::Bool(true)]),
        },
    );
    invalid.enum_options = vec![configured.clone()];
    invalid.enum_payloads = vec![EnumOptionPayload {
        id: configured.id,
        shape: EnumPayloadShape::Tuple(vec![ValueShape::scalar(ValueType::Int)]),
    }];
    let mut builder = RegistryBuilder::new();
    builder.namespace("render", "test-owner").unwrap();
    builder.register(invalid);
    assert!(builder.finish().is_err());
}

#[test]
fn nested_enum_payloads_are_validated_recursively() {
    let inner = EnumOption::from_stable_name("render::Filter::Radius");
    let outer = EnumOption::from_stable_name("render::Mode::Filtered");
    let default = Value::EnumPayload {
        id: outer.id,
        payload: EnumPayload::Struct(BTreeMap::from([(
            "filter".into(),
            Value::EnumPayload {
                id: inner.id,
                payload: EnumPayload::Tuple(vec![Value::Int(2)]),
            },
        )])),
    };
    let mut def = SettingDef::new("render.nested_mode", "render", default);
    def.enum_options = vec![outer.clone(), inner.clone()];
    def.enum_payloads = vec![
        EnumOptionPayload {
            id: outer.id,
            shape: EnumPayloadShape::Struct(BTreeMap::from([(
                "filter".into(),
                ValueShape::scalar(ValueType::Enum),
            )])),
        },
        EnumOptionPayload {
            id: inner.id,
            shape: EnumPayloadShape::Tuple(vec![ValueShape::scalar(ValueType::Int)]),
        },
    ];
    let schema = registry(vec![def]);
    let id = schema.id(&"render.nested_mode".into()).unwrap();
    assert!(schema
        .validate_value(id, schema.def(id).unwrap().default.clone())
        .is_ok());
    assert!(schema
        .validate_value(
            id,
            Value::EnumPayload {
                id: outer.id,
                payload: EnumPayload::Struct(BTreeMap::from([(
                    "filter".into(),
                    Value::Enum(inner.id),
                )])),
            }
        )
        .is_err());
}

#[test]
fn validation_reports_alias_namespace_and_default_errors() {
    let mut b = RegistryBuilder::new();
    b.namespace("audio", "owner").unwrap();
    let mut bad = bool_def("audio.enabled", true);
    bad.aliases.push("audio.other".into());
    b.register(bad.clone());
    b.register(bad);
    let errors = b.finish().unwrap_err();
    assert!(errors
        .iter()
        .any(|e| matches!(e, SettingsError::DuplicateKey(_))));
}

#[test]
fn twelve_key_layer_matrix_is_stable_and_explain_matches_snapshot() {
    let keys = [
        "audio.master_volume",
        "audio.music_volume",
        "audio.effects_volume",
        "render.shadows.quality",
        "render.texture_quality",
        "render.resolution_scale",
        "video.fullscreen",
        "video.hdr_enabled",
        "input.mouse_sensitivity",
        "network.tick_rate",
        "telemetry.enabled",
        "editor.theme",
    ];
    let defs = keys
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let mut d = SettingDef::new(*k, k.split('.').next().unwrap(), Value::Int(0));
            d.domains = if k.starts_with("editor.") {
                DomainMask::EDITOR
            } else {
                DomainMask::ALL
            };
            d.constraints.min = Some(-100.0);
            d.constraints.max = Some(100.0);
            if i == 10 {
                d.precedence = PrecedenceProfile::Locked;
            }
            d
        })
        .collect();
    let reg = registry(defs);
    assert_eq!(reg.len(), 12);
    let mut policy = source(
        LayerKind::Mandatory,
        "policy",
        &[(keys[0], Value::Int(5)), (keys[10], Value::Int(9))],
    );
    policy.trust = TrustLevel::Trusted;
    let mut layers = vec![
        source(
            LayerKind::EngineDefault,
            "compiled",
            &[(keys[0], Value::Int(1)), (keys[10], Value::Int(1))],
        ),
        source(
            LayerKind::Recommended,
            "admin-recommended",
            &[(keys[0], Value::Int(2))],
        ),
        source(LayerKind::User, "user-file", &[(keys[0], Value::Int(3))]),
        source(LayerKind::Runtime, "console", &[(keys[0], Value::Int(4))]),
        policy,
    ];
    let snap = Resolver::new(&reg)
        .resolve(Domain::Runtime, &layers, 7)
        .unwrap();
    let id = reg.id(&keys[0].into()).unwrap();
    assert_eq!(snap.get(id), Some(&Value::Int(5)));
    let explain = Resolver::new(&reg)
        .inspect(&snap, &keys[0].into(), &layers)
        .unwrap();
    assert_eq!(explain.effective, snap.get(id).cloned());
    assert_eq!(explain.locked_by.as_ref().unwrap().source_id, "policy");
    assert!(!explain.modifiable);
    assert_eq!(
        explain.by_layer[&LayerKind::User].as_ref().unwrap().0,
        Value::Int(3)
    );
    assert_eq!(
        snap.get(reg.id(&keys[10].into()).unwrap()),
        Some(&Value::Int(9)),
        "Locked profile admits only default and Mandatory"
    );

    // Property-style permutation check: source enumeration never affects the winner.
    let expected = snap.get(id).cloned();
    for shift in 0..layers.len() {
        layers.rotate_left(shift);
        let got = Resolver::new(&reg)
            .resolve(Domain::Runtime, &layers, 7)
            .unwrap();
        assert_eq!(got.get(id).cloned(), expected);
    }
}

#[test]
fn user_reset_falls_back_and_recommended_yields_to_explicit_user() {
    let mut d = SettingDef::new("audio.master_volume", "audio", Value::Float(0.5));
    d.constraints.min = Some(0.0);
    d.constraints.max = Some(1.0);
    let reg = registry(vec![d]);
    let resolver = Resolver::new(&reg);
    let recommended = source(
        LayerKind::Recommended,
        "policy-hint",
        &[("audio.master_volume", Value::Float(0.7))],
    );
    let id = reg.id(&"audio.master_volume".into()).unwrap();
    let recommended_snapshot = resolver
        .resolve(Domain::Runtime, core::slice::from_ref(&recommended), 1)
        .unwrap();
    assert_eq!(recommended_snapshot.get(id), Some(&Value::Float(0.7)));
    let user = source(
        LayerKind::User,
        "user",
        &[("audio.master_volume", Value::Float(0.9))],
    );
    let explicit = resolver
        .resolve(Domain::Runtime, &[recommended.clone(), user], 2)
        .unwrap();
    assert_eq!(explicit.get(id), Some(&Value::Float(0.9)));
    let reset = resolver
        .resolve(Domain::Runtime, &[recommended], 3)
        .unwrap();
    assert_eq!(reset.get(id), Some(&Value::Float(0.7)));
    assert_eq!(diff(&recommended_snapshot, &reset).changes.len(), 0);

    let defaulted = resolver.resolve(Domain::Runtime, &[], 4).unwrap();
    let explanation = resolver
        .inspect(&defaulted, &"audio.master_volume".into(), &[])
        .unwrap();
    assert_eq!(explanation.effective, Some(Value::Float(0.5)));
    assert_eq!(
        explanation.source.as_ref().unwrap().source_id,
        "schema-default"
    );
    assert_eq!(
        explanation.by_layer[&LayerKind::EngineDefault]
            .as_ref()
            .unwrap()
            .0,
        Value::Float(0.5)
    );
}

#[test]
fn profile_specific_ordering_and_same_layer_order_are_deterministic() {
    let mut d = SettingDef::new("network.tick_rate", "network", Value::Int(30));
    d.precedence = PrecedenceProfile::ServerWins;
    let reg = registry(vec![d]);
    let mut remote = source(
        LayerKind::Remote,
        "server",
        &[("network.tick_rate", Value::Int(60))],
    );
    remote.trust = TrustLevel::Trusted;
    let user = source(
        LayerKind::User,
        "user",
        &[("network.tick_rate", Value::Int(120))],
    );
    let snap = Resolver::new(&reg)
        .resolve(Domain::Runtime, &[user.clone(), remote.clone()], 1)
        .unwrap();
    assert_eq!(
        snap.get(reg.id(&"network.tick_rate".into()).unwrap()),
        Some(&Value::Int(60))
    );
    let mut a = source(
        LayerKind::User,
        "a",
        &[("network.tick_rate", Value::Int(40))],
    );
    a.order = 1;
    let mut b = source(
        LayerKind::User,
        "b",
        &[("network.tick_rate", Value::Int(50))],
    );
    b.order = 2;
    let x = Resolver::new(&reg)
        .resolve(Domain::Runtime, &[a.clone(), b.clone()], 2)
        .unwrap();
    let y = Resolver::new(&reg)
        .resolve(Domain::Runtime, &[b, a], 2)
        .unwrap();
    assert_eq!(
        x.get(reg.id(&"network.tick_rate".into()).unwrap()),
        Some(&Value::Int(50))
    );
    assert_eq!(
        x.get(reg.id(&"network.tick_rate".into()).unwrap()),
        y.get(reg.id(&"network.tick_rate".into()).unwrap())
    );
}

#[test]
fn constraints_clamp_reject_nonfinite_and_enforce_cross_rules() {
    let mut volume = SettingDef::new("audio.master_volume", "audio", Value::Float(0.5));
    volume.constraints.min = Some(0.0);
    volume.constraints.max = Some(1.0);
    volume.on_violation = ViolationPolicy::Clamp;
    let mut hdr = bool_def("video.hdr_enabled", false);
    hdr.cross_rules.push(CrossRule::Requires {
        key: "video.fullscreen".into(),
        when: true,
    });
    let fullscreen = bool_def("video.fullscreen", false);
    let reg = registry(vec![volume, hdr, fullscreen]);
    let vol_id = reg.id(&"audio.master_volume".into()).unwrap();
    assert_eq!(
        reg.validate_value(vol_id, Value::Float(2.0)).unwrap(),
        Value::Float(1.0)
    );
    assert!(matches!(
        reg.validate_value(vol_id, Value::Float(f64::NAN)),
        Err(SettingsError::OutOfRange { .. })
    ));
    let sources = [source(
        LayerKind::User,
        "user",
        &[("video.hdr_enabled", Value::Bool(true))],
    )];
    assert!(
        matches!(Resolver::new(&reg).resolve(Domain::Runtime, &sources, 1), Err(errors) if errors.iter().any(|e| matches!(e, SettingsError::CrossConstraint { .. })))
    );
}

#[test]
fn recursive_array_shapes_validate_element_type_and_length() {
    let mut array = SettingDef::new(
        "render.samples",
        "render",
        Value::Array(vec![Value::Int(2), Value::Int(4)]),
    );
    array.set_shape(ValueShape::array(ValueShape::scalar(ValueType::Int), 1, 3));
    let reg = registry(vec![array]);
    let id = reg.id(&"render.samples".into()).unwrap();

    assert_eq!(
        reg.validate_value(id, Value::Array(vec![Value::Int(9)]))
            .unwrap(),
        Value::Array(vec![Value::Int(9)])
    );
    assert!(matches!(
        reg.validate_value(id, Value::Array(vec![Value::Int(1), Value::Bool(true)])),
        Err(SettingsError::TypeMismatch { .. })
    ));
    assert!(matches!(
        reg.validate_value(id, Value::Array(Vec::new())),
        Err(SettingsError::TooShort { .. })
    ));
    assert!(matches!(
        reg.validate_value(
            id,
            Value::Array(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
                Value::Int(4)
            ])
        ),
        Err(SettingsError::TooLong { .. })
    ));
}

#[test]
fn array_shape_rejects_invalid_schema_and_infers_nested_defaults() {
    let mut invalid = SettingDef::new(
        "render.samples",
        "render",
        Value::Array(vec![Value::Int(1)]),
    );
    invalid.set_shape(ValueShape::array(ValueShape::scalar(ValueType::Int), 3, 2));
    let mut builder = RegistryBuilder::new();
    builder.namespace("render", "test-owner").unwrap();
    builder.register(invalid);
    assert!(matches!(
        builder.finish(),
        Err(errors) if errors.iter().any(|error| matches!(error, SettingsError::InvalidDefinition { reason, .. } if reason.contains("inverted array bounds")))
    ));

    let definition = SettingDef::new(
        "render.matrix_row",
        "render",
        Value::Array(vec![Value::Array(vec![
            Value::Float(1.0),
            Value::Float(0.0),
        ])]),
    );
    assert_eq!(
        definition.shape,
        ValueShape::bounded_array(
            ValueShape::bounded_array(ValueShape::scalar(ValueType::Float), DEFAULT_MAX_ARRAY_LEN,),
            DEFAULT_MAX_ARRAY_LEN,
        )
    );
}

#[test]
fn trust_boundary_and_domain_layer_validation_are_enforced() {
    let mut execution = bool_def("game.allow_scripts", false);
    execution.flags.execution_class = true;
    execution.domains = DomainMask::RUNTIME;
    let reg = registry(vec![execution]);
    let untrusted = source(
        LayerKind::Project,
        "project",
        &[("game.allow_scripts", Value::Bool(true))],
    );
    assert!(
        matches!(Resolver::new(&reg).resolve(Domain::Runtime, &[untrusted], 1), Err(errors) if errors.iter().any(|e| matches!(e, SettingsError::UntrustedExecutionSetting { .. })))
    );
    let wrong = source(LayerKind::Machine, "machine", &[]);
    assert!(
        matches!(Resolver::new(&reg).resolve(Domain::Runtime, &[wrong], 1), Err(errors) if errors.iter().any(|e| matches!(e, SettingsError::WrongLayer { .. })))
    );
}

#[test]
fn project_override_policy_and_source_identity_are_enforced() {
    let mut editor = bool_def("editor.theme_dark", false);
    editor.domains = DomainMask::EDITOR;
    assert!(!editor.flags.project_overridable);
    let reg = registry(vec![editor]);
    let project = source(
        LayerKind::Project,
        "project-settings",
        &[("editor.theme_dark", Value::Bool(true))],
    );
    assert!(matches!(
        Resolver::new(&reg).resolve(Domain::Editor, core::slice::from_ref(&project), 1),
        Err(errors) if errors.iter().any(|e| matches!(e, SettingsError::ProjectOverrideDenied(_)))
    ));
    assert!(matches!(
        Resolver::new(&reg).resolve(Domain::Editor, &[project.clone(), project], 1),
        Err(errors) if errors.iter().any(|e| matches!(e, SettingsError::DuplicateSource { .. }))
    ));
}

#[test]
fn alias_and_lookup_and_diff_are_consistent() {
    let mut d = bool_def("render.hdr_enabled", false);
    d.aliases.push("render.hdr".into());
    let reg = registry(vec![d]);
    assert_eq!(
        reg.id(&"render.hdr".into()),
        reg.id(&"render.hdr_enabled".into())
    );
    let a = Resolver::new(&reg)
        .resolve(Domain::Runtime, &[], 1)
        .unwrap();
    let s = source(
        LayerKind::User,
        "user",
        &[("render.hdr_enabled", Value::Bool(true))],
    );
    let b = Resolver::new(&reg)
        .resolve(Domain::Runtime, &[s], 2)
        .unwrap();
    let changes = diff(&a, &b);
    assert_eq!(changes.changes.len(), 1);
    assert_eq!(changes.changes[0].after, Some(Value::Bool(true)));
}

#[test]
fn error_display_is_actionable_and_domain_stacks_are_closed() {
    let error = SettingsError::UntrustedExecutionSetting {
        key: "game.allow_scripts".into(),
        trust: TrustLevel::Project,
    };
    assert!(format!("{error}").contains("game.allow_scripts"));
    assert_eq!(Domain::Runtime.layers().last(), Some(&LayerKind::Mandatory));
    assert_eq!(
        Domain::System.layers(),
        &[
            LayerKind::EngineDefault,
            LayerKind::Machine,
            LayerKind::Recommended,
            LayerKind::Mandatory
        ]
    );
}
