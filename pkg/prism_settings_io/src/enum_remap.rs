//! Stable-name enum migrations.

use prism_settings::{EnumId, Value};
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
        self.options.insert(
            EnumId::from_stable_name(old_stable_name.as_ref()),
            EnumId::from_stable_name(new_stable_name.as_ref()),
        );
        self
    }

    /// Returns the remapped enum identity, preserving values not listed.
    pub fn apply(&self, value: EnumId) -> EnumId {
        self.options.get(&value).copied().unwrap_or(value)
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
        crate::MigrationOperation::convert(key, move |value| match value {
            Value::Enum(id) => Ok(Value::Enum(remap.apply(id))),
            other => Err(format!("expected enum setting, got {other:?}")),
        })
    }
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
}
