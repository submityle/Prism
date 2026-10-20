//! ECS integration for the canonical [`prism_settings`] resolver.
//!
//! The schema, precedence policy, and value validation live in
//! [`prism_settings`]. This module only owns the application-facing mutable
//! source set and the last resolved snapshot; it does not define a second
//! settings value model.

use alloc::{borrow::ToOwned, collections::BTreeMap, string::String, vec::Vec};

use prism_ecs::resource::Resource;
use prism_settings::{
    Domain, LayerKind, LayerSource, Resolver, SettingKey, SettingsError, SettingsRegistry,
    Snapshot, TrustLevel, Value,
};

use crate::app::App;

/// The application-owned settings state for one domain.
///
/// Sources are keyed by `(layer, source_id)` so replacement and removal are
/// deterministic. [`resolve`](Self::resolve) freezes them into an immutable
/// snapshot, which is the only value view intended for frame systems.
pub struct SettingsRuntime {
    registry: SettingsRegistry,
    domain: Domain,
    generation: u64,
    sources: BTreeMap<(LayerKind, String), LayerSource>,
    snapshot: Snapshot,
}

impl Resource for SettingsRuntime {}

impl SettingsRuntime {
    /// Creates an application settings resource with schema defaults resolved.
    pub fn new(registry: SettingsRegistry, domain: Domain) -> Result<Self, Vec<SettingsError>> {
        let snapshot = Resolver::new(&registry).resolve(domain, &[], 0)?;
        Ok(Self {
            registry,
            domain,
            generation: 0,
            sources: BTreeMap::new(),
            snapshot,
        })
    }

    /// Returns the frozen schema used by this resource.
    #[must_use]
    pub fn registry(&self) -> &SettingsRegistry {
        &self.registry
    }

    /// Returns the resolved domain.
    #[must_use]
    pub const fn domain(&self) -> Domain {
        self.domain
    }

    /// Returns the last successfully resolved snapshot.
    #[must_use]
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Adds or replaces one source candidate.
    pub fn set(
        &mut self,
        layer: LayerKind,
        trust: TrustLevel,
        source_id: impl Into<String>,
        key: impl Into<SettingKey>,
        value: Value,
    ) {
        let source_id = source_id.into();
        let source = self
            .sources
            .entry((layer, source_id.clone()))
            .or_insert_with(|| LayerSource::new(layer, trust, source_id));
        source.trust = trust;
        source.set(key, value);
    }

    /// Removes one candidate; empty sources are removed as well.
    pub fn clear(&mut self, layer: LayerKind, source_id: &str, key: &SettingKey) -> bool {
        let identity = (layer, source_id.to_owned());
        let Some(source) = self.sources.get_mut(&identity) else {
            return false;
        };
        let removed = source.values.remove(key).is_some();
        if source.values.is_empty() {
            self.sources.remove(&identity);
        }
        removed
    }

    /// Resolves all sources and atomically publishes the new snapshot.
    pub fn resolve(&mut self) -> Result<&Snapshot, Vec<SettingsError>> {
        let sources = self.sources.values().cloned().collect::<Vec<_>>();
        let generation = self.generation.saturating_add(1);
        let snapshot = Resolver::new(&self.registry).resolve(self.domain, &sources, generation)?;
        self.generation = generation;
        self.snapshot = snapshot;
        Ok(&self.snapshot)
    }

    /// Returns the source records in deterministic order for persistence or inspection.
    #[must_use]
    pub fn sources(&self) -> impl Iterator<Item = &LayerSource> {
        self.sources.values()
    }
}

impl App {
    /// Installs the canonical [`SettingsRuntime`] for one domain.
    pub fn install_settings_runtime(
        &mut self,
        registry: SettingsRegistry,
        domain: Domain,
    ) -> Result<&mut Self, Vec<SettingsError>> {
        let runtime = SettingsRuntime::new(registry, domain)?;
        self.insert_resource(runtime);
        Ok(self)
    }

    /// Borrows the canonical settings runtime, if installed.
    #[must_use]
    pub fn settings_runtime(&self) -> Option<&SettingsRuntime> {
        self.world().get_resource::<SettingsRuntime>()
    }

    /// Mutably borrows the canonical settings runtime, if installed.
    #[must_use]
    pub fn settings_runtime_mut(&mut self) -> Option<&mut SettingsRuntime> {
        self.world_mut().get_resource_mut::<SettingsRuntime>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_settings::{RegistryBuilder, SettingDef};

    fn registry() -> SettingsRegistry {
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(SettingDef::new("render.scale", "render", Value::Float(1.0)));
        builder.finish().unwrap()
    }

    #[test]
    fn app_runtime_resolves_new_core_values_and_preserves_defaults() {
        let mut app = App::new();
        app.install_settings_runtime(registry(), Domain::Runtime)
            .unwrap();
        let key = SettingKey::from("render.scale");
        let runtime = app.settings_runtime_mut().unwrap();
        runtime.set(
            LayerKind::User,
            TrustLevel::User,
            "user-file",
            key.clone(),
            Value::Float(0.75),
        );
        runtime.resolve().unwrap();
        let id = runtime.registry().id(&key).unwrap();
        assert_eq!(runtime.snapshot().get(id), Some(&Value::Float(0.75)));
    }

    #[test]
    fn clear_removes_only_the_selected_source_candidate() {
        let mut runtime = SettingsRuntime::new(registry(), Domain::Runtime).unwrap();
        let key = SettingKey::from("render.scale");
        runtime.set(
            LayerKind::User,
            TrustLevel::User,
            "user-file",
            key.clone(),
            Value::Float(0.8),
        );
        runtime.set(
            LayerKind::Runtime,
            TrustLevel::User,
            "console",
            key.clone(),
            Value::Float(0.6),
        );
        runtime.resolve().unwrap();
        assert!(runtime.clear(LayerKind::Runtime, "console", &key));
        runtime.resolve().unwrap();
        let id = runtime.registry().id(&key).unwrap();
        assert_eq!(runtime.snapshot().get(id), Some(&Value::Float(0.8)));
    }
}
