//! Stable-name enum migrations.

use prism_settings::{EnumId, EnumPayload, EnumTypeId, Value};
use std::collections::BTreeMap;

/// Explicit remapping of persisted enum variant names across schema versions.
///
/// Multiple old names may target the same new name, which represents a merge.
/// Names not listed here retain their identity; the destination registry then
/// decides whether they remain valid. This makes removals fail closed unless
/// the migration explicitly maps them to a surviving option.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnumRemap {
    options: BTreeMap<EnumId, EnumId>,
    typed_options: BTreeMap<(EnumTypeId, EnumId), (EnumTypeId, EnumId)>,
}

impl EnumRemap {
    /// Creates an empty remap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maps one stable old variant name to a stable destination variant name.
    pub fn map(
        &mut self,
        old_stable_name: impl AsRef<str>,
        new_stable_name: impl AsRef<str>,
    ) -> &mut Self {
        let old = old_stable_name.as_ref();
        let new = new_stable_name.as_ref();
        self.options
            .insert(EnumId::from_stable_name(old), EnumId::from_stable_name(new));
        if let (Some(old_type), Some(new_type)) = (enum_type_name(old), enum_type_name(new)) {
            self.typed_options.insert(
                (
                    EnumTypeId::from_stable_name(old_type),
                    EnumId::from_stable_name(old),
                ),
                (
                    EnumTypeId::from_stable_name(new_type),
                    EnumId::from_stable_name(new),
                ),
            );
        }
        self
    }

    /// Returns the remapped enum identity, preserving values not listed.
    pub fn apply(&self, value: EnumId) -> EnumId {
        self.options.get(&value).copied().unwrap_or(value)
    }

    /// Returns the remapped type and variant identity for a typed enum.
    pub fn apply_typed(&self, enum_type: EnumTypeId, value: EnumId) -> (EnumTypeId, EnumId) {
        self.typed_options
            .get(&(enum_type, value))
            .copied()
            .unwrap_or((enum_type, value))
    }

    /// Returns the number of explicitly mapped old identities.
    pub fn len(&self) -> usize {
        self.options.len()
    }

    /// Returns whether the remap contains no explicit mappings.
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }

    /// Builds a generic migration conversion for one settings key.
    pub fn operation(
        &self,
        key: impl Into<prism_settings::SettingKey>,
    ) -> crate::MigrationOperation {
        let remap = self.clone();
        crate::MigrationOperation::convert(key, move |value| {
            if !contains_enum(&value) {
                return Err(format!("expected enum setting tree, got {value:?}"));
            }
            remap_value(value, &remap)
        })
    }
}

fn contains_enum(value: &Value) -> bool {
    match value {
        Value::Enum(_)
        | Value::EnumPayload { .. }
        | Value::EnumTyped { .. }
        | Value::EnumPayloadTyped { .. } => true,
        Value::Array(values) => values.iter().any(contains_enum),
        Value::Table(values) => values.values().any(contains_enum),
        _ => false,
    }
}

fn remap_value(value: Value, remap: &EnumRemap) -> Result<Value, String> {
    Ok(match value {
        Value::Enum(id) => Value::Enum(remap.apply(id)),
        Value::EnumPayload { id, payload } => Value::EnumPayload {
            id: remap.apply(id),
            payload: match payload {
                EnumPayload::Tuple(values) => EnumPayload::Tuple(
                    values
                        .into_iter()
                        .map(|value| remap_value(value, remap))
                        .collect::<Result<_, _>>()?,
                ),
                EnumPayload::Struct(values) => EnumPayload::Struct(
                    values
                        .into_iter()
                        .map(|(key, value)| remap_value(value, remap).map(|value| (key, value)))
                        .collect::<Result<_, _>>()?,
                ),
            },
        },
        Value::EnumTyped { enum_type, id } => {
            let (enum_type, id) = remap.apply_typed(enum_type, id);
            Value::EnumTyped { enum_type, id }
        }
        Value::EnumPayloadTyped {
            enum_type,
            id,
            payload,
        } => {
            let (enum_type, id) = remap.apply_typed(enum_type, id);
            Value::EnumPayloadTyped {
                enum_type,
                id,
                payload: match payload {
                    EnumPayload::Tuple(values) => EnumPayload::Tuple(
                        values
                            .into_iter()
                            .map(|value| remap_value(value, remap))
                            .collect::<Result<_, _>>()?,
                    ),
                    EnumPayload::Struct(values) => EnumPayload::Struct(
                        values
                            .into_iter()
                            .map(|(key, value)| remap_value(value, remap).map(|value| (key, value)))
                            .collect::<Result<_, _>>()?,
                    ),
                },
            }
        }
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| remap_value(value, remap))
                .collect::<Result<_, _>>()?,
        ),
        Value::Table(values) => Value::Table(
            values
                .into_iter()
                .map(|(key, value)| remap_value(value, remap).map(|value| (key, value)))
                .collect::<Result<_, _>>()?,
        ),
        // Scalar payload leaves are valid (for example tuple payloads such as
        // `Radius(u32)`). They carry no enum identity and remain unchanged.
        other => other,
    })
}

fn enum_type_name(stable_variant_name: &str) -> Option<&str> {
    stable_variant_name
        .rsplit_once("::")
        .map(|(type_name, _)| type_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MigrationChain, SettingsDocument};
    use prism_settings::{EnumOption, RegistryBuilder, SettingDef, SettingKey};

    #[test]
    fn remaps_and_merges_stable_names_through_migration_and_watcher_codec() {
        let mut remap = EnumRemap::new();
        remap
            .map("quality::Epic", "quality::Ultra")
            .map("quality::Cinematic", "quality::Ultra");
        assert_eq!(remap.len(), 2);

        let ultra = EnumOption::from_stable_name("quality::Ultra");
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        let mut definition = SettingDef::new("render.quality", "render", Value::Enum(ultra.id));
        definition.enum_options = vec![ultra.clone()];
        builder.register(definition);
        let registry = builder.finish().unwrap();

        let mut chain = MigrationChain::new(2);
        chain
            .add_step(1, [remap.operation("render.quality")])
            .unwrap();
        for old_name in ["quality::Epic", "quality::Cinematic"] {
            let mut document = SettingsDocument::new(1);
            let mut tagged = toml::Table::new();
            tagged.insert(
                "__prism_value".to_owned(),
                toml::Value::String("enum".to_owned()),
            );
            tagged.insert("value".to_owned(), toml::Value::String(old_name.to_owned()));
            document.preserve_unknown("render.quality", toml::Value::Table(tagged));
            chain.migrate(&mut document).unwrap();
            document.classify_with_registry(&registry).unwrap();
            assert_eq!(
                document
                    .candidates()
                    .get(&SettingKey::from("render.quality")),
                Some(&Value::Enum(ultra.id))
            );
            document.bind_registry(&registry);
            assert!(document
                .encode()
                .unwrap()
                .contains("value = \"quality::Ultra\""));
        }
    }

    #[test]
    fn empty_remap_preserves_ids_and_non_enum_conversion_fails_closed() {
        let remap = EnumRemap::new();
        let existing = EnumId::from_stable_name("quality::High");
        assert_eq!(remap.apply(existing), existing);

        let mut document = SettingsDocument::new(1);
        document.insert("render.quality", Value::Int(1));
        let mut chain = MigrationChain::new(2);
        chain
            .add_step(1, [remap.operation("render.quality")])
            .unwrap();
        assert!(chain.migrate(&mut document).is_err());
        assert_eq!(document.schema_version(), 1);
        assert_eq!(
            document
                .candidates()
                .get(&SettingKey::from("render.quality")),
            Some(&Value::Int(1))
        );
    }

    #[test]
    fn typed_remap_keeps_enum_types_separate_and_can_move_a_type() {
        let mut remap = EnumRemap::new();
        remap.map("settings::First::Value", "settings::Second::Value");
        let first = EnumTypeId::from_stable_name("settings::First");
        let second = EnumTypeId::from_stable_name("settings::Second");
        let value = EnumId::from_stable_name("settings::First::Value");
        let other = EnumId::from_stable_name("settings::Other::Value");
        assert_eq!(
            remap.apply_typed(first, value),
            (second, EnumId::from_stable_name("settings::Second::Value"))
        );
        assert_eq!(remap.apply_typed(first, other), (first, other));
        let remapped = remap_value(
            Value::EnumPayloadTyped {
                enum_type: first,
                id: value,
                payload: EnumPayload::Tuple(vec![Value::EnumTyped {
                    enum_type: first,
                    id: value,
                }]),
            },
            &remap,
        )
        .unwrap();
        assert!(matches!(
            remapped,
            Value::EnumPayloadTyped { enum_type, id, payload: EnumPayload::Tuple(values) }
                if enum_type == second
                    && id == EnumId::from_stable_name("settings::Second::Value")
                    && matches!(values.as_slice(), [Value::EnumTyped { enum_type, id }] if *enum_type == second && *id == EnumId::from_stable_name("settings::Second::Value"))
        ));
    }

    #[test]
    fn remap_recurses_through_payload_arrays_tables_and_set_storage() {
        use prism_settings::EnumPayload;
        let old = EnumId::from_stable_name("render::Mode::Old");
        let new = EnumId::from_stable_name("render::Mode::New");
        let nested_old = EnumId::from_stable_name("render::Filter::Old");
        let nested_new = EnumId::from_stable_name("render::Filter::New");
        let mut remap = EnumRemap::new();
        remap
            .map("render::Mode::Old", "render::Mode::New")
            .map("render::Filter::Old", "render::Filter::New");
        let input = Value::Array(vec![Value::Table(BTreeMap::from([(
            "items".to_owned(),
            Value::Array(vec![Value::EnumPayload {
                id: old,
                payload: EnumPayload::Struct(BTreeMap::from([(
                    "nested".to_owned(),
                    Value::EnumPayload {
                        id: nested_old,
                        payload: EnumPayload::Tuple(vec![Value::Enum(old)]),
                    },
                )])),
            }]),
        )]))]);
        let output = remap_value(input, &remap).unwrap();
        let Value::Array(items) = output else {
            panic!("array expected")
        };
        let Value::Table(fields) = &items[0] else {
            panic!("table expected")
        };
        let Value::Array(items) = &fields["items"] else {
            panic!("array expected")
        };
        let Value::EnumPayload {
            id,
            payload: EnumPayload::Struct(fields),
        } = &items[0]
        else {
            panic!("payload expected")
        };
        assert_eq!(*id, new);
        let Value::EnumPayload {
            id,
            payload: EnumPayload::Tuple(values),
        } = &fields["nested"]
        else {
            panic!("nested payload expected")
        };
        assert_eq!(*id, nested_new);
        assert_eq!(values, &[Value::Enum(new)]);
    }

    #[test]
    fn migration_decodes_and_remaps_enum_payload_from_preserved_toml() {
        use prism_settings::{
            EnumOptionPayload, EnumPayload, EnumPayloadShape, ValueShape, ValueType,
        };
        let old = EnumOption::from_stable_name("render::Mode::Old");
        let new = EnumOption::from_stable_name("render::Mode::New");
        let inner_old = EnumOption::from_stable_name("render::Filter::Old");
        let inner_new = EnumOption::from_stable_name("render::Filter::New");
        let item_old = EnumOption::from_stable_name("render::Preset::Old");
        let item_new = EnumOption::from_stable_name("render::Preset::New");
        let mut remap = EnumRemap::new();
        remap
            .map(old.stable_name.as_str(), new.stable_name.as_str())
            .map(
                inner_old.stable_name.as_str(),
                inner_new.stable_name.as_str(),
            )
            .map(item_old.stable_name.as_str(), item_new.stable_name.as_str());

        let parsed: toml::Value = toml::from_str(
            "value = { __prism_value = \"enum_payload\", variant = \"render::Mode::Old\", payload = { nested = { __prism_value = \"enum\", value = \"render::Filter::Old\" }, items = [{ __prism_value = \"enum\", value = \"render::Preset::Old\" }] } }",
        ).unwrap();
        let raw = parsed.get("value").unwrap().clone();
        let mut document = SettingsDocument::new(1);
        document.preserve_unknown("render.mode", raw);
        let mut chain = MigrationChain::new(2);
        chain.add_step(1, [remap.operation("render.mode")]).unwrap();
        assert_eq!(chain.migrate(&mut document).unwrap().converted, 1);

        let expected = Value::EnumPayload {
            id: new.id,
            payload: EnumPayload::Struct(BTreeMap::from([
                ("nested".to_owned(), Value::Enum(inner_new.id)),
                (
                    "items".to_owned(),
                    Value::Array(vec![Value::Enum(item_new.id)]),
                ),
            ])),
        };
        assert_eq!(
            document.candidates().get(&SettingKey::from("render.mode")),
            Some(&expected)
        );

        let mut definition = SettingDef::new("render.mode", "render", expected.clone());
        definition.enum_options = vec![new, inner_new, item_new];
        definition.enum_payloads = vec![EnumOptionPayload {
            id: EnumId::from_stable_name("render::Mode::New"),
            shape: EnumPayloadShape::Struct(BTreeMap::from([
                ("nested".to_owned(), ValueShape::scalar(ValueType::Enum)),
                (
                    "items".to_owned(),
                    ValueShape::bounded_array(ValueShape::scalar(ValueType::Enum), 8),
                ),
            ])),
        }];
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(definition);
        let registry = builder.finish().unwrap();
        document.bind_registry(&registry);
        assert!(document.encode().unwrap().contains("render::Mode::New"));
    }
}
