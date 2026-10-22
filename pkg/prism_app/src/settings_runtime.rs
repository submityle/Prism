//! ECS integration for the canonical [`prism_settings`] resolver.
//!
//! The schema, precedence policy, and value validation live in
//! [`prism_settings`]. This module only owns the application-facing mutable
//! source set and the last resolved snapshot; it does not define a second
//! settings value model.

use alloc::{borrow::ToOwned, collections::BTreeMap, string::String, vec::Vec};

use prism_ecs::resource::Resource;
use prism_settings::{
    Domain, LayerKind, LayerSource, Resolver, SettingId, SettingKey, SettingsError,
    SettingsRegistry, Snapshot, TrustLevel, Value,
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

/// Complete rollback image for a settings runtime.
///
/// The source table and generation reconstruct the resolved snapshot during
/// restore; restoring only a snapshot would leave later resolutions based on
/// a different source state.
#[derive(Clone, Debug, PartialEq)]
pub struct SettingsRuntimeCheckpoint {
    generation: u64,
    sources: BTreeMap<(LayerKind, String), LayerSource>,
}

/// Settings source replacement prepared against a specific runtime generation.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSettingsUpdate {
    base_generation: u64,
    source: LayerSource,
    snapshot: Snapshot,
}

/// Failure to prepare or commit one source replacement.
#[derive(Clone, Debug, PartialEq)]
pub enum SettingsRuntimeError {
    /// Candidate resolution rejected a value or source policy.
    Resolve(Vec<SettingsError>),
    /// The runtime changed after the update was prepared.
    Stale {
        /// Runtime generation against which the candidate was prepared.
        expected: u64,
        /// Current runtime generation at commit time.
        actual: u64,
    },
    /// The generation counter cannot advance without wrapping.
    GenerationExhausted,
    /// The prepared snapshot no longer matches the staged sources.
    CandidateMismatch,
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

    /// Returns the generation of the currently published snapshot.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Captures sources and snapshot as one rollback image.
    #[must_use]
    pub fn checkpoint(&self) -> SettingsRuntimeCheckpoint {
        SettingsRuntimeCheckpoint {
            generation: self.generation,
            sources: self.sources.clone(),
        }
    }

    /// Restores a complete checkpoint, validating it against this runtime's
    /// registry and domain before publishing any part of it.
    pub fn restore_checkpoint(
        &mut self,
        checkpoint: SettingsRuntimeCheckpoint,
    ) -> Result<(), Vec<SettingsError>> {
        let values = checkpoint.sources.values().cloned().collect::<Vec<_>>();
        let restored =
            Resolver::new(&self.registry).resolve(self.domain, &values, checkpoint.generation)?;
        self.sources = checkpoint.sources;
        self.generation = checkpoint.generation;
        self.snapshot = restored;
        Ok(())
    }

    /// Resolves the currently installed authoritative sources at a durable
    /// generation without publishing a partial result.
    pub fn rebuild_snapshot_at(&self, generation: u64) -> Result<Snapshot, Vec<SettingsError>> {
        let sources = self.sources.values().cloned().collect::<Vec<_>>();
        Resolver::new(&self.registry).resolve(self.domain, &sources, generation)
    }

    /// Publishes a snapshot rebuilt from this runtime's current sources after
    /// verifying its domain, generation, values, and provenance.
    pub fn publish_rebuilt_snapshot(
        &mut self,
        snapshot: Snapshot,
    ) -> Result<(), SettingsRuntimeError> {
        if snapshot.domain() != self.domain {
            return Err(SettingsRuntimeError::CandidateMismatch);
        }
        let rebuilt = self
            .rebuild_snapshot_at(snapshot.generation())
            .map_err(SettingsRuntimeError::Resolve)?;
        if !same_effective_snapshot(&rebuilt, &snapshot) {
            return Err(SettingsRuntimeError::CandidateMismatch);
        }
        self.generation = snapshot.generation();
        self.snapshot = rebuilt;
        Ok(())
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

    /// Replaces one complete source and publishes its resolved snapshot.
    ///
    /// Replacement is staged against a cloned source map. If resolution
    /// rejects the candidate, both the previous source map and snapshot stay
    /// untouched. This is the operation used by file-backed candidates: a
    /// removed key must disappear instead of surviving as stale state.
    pub fn replace_source(&mut self, source: LayerSource) -> Result<&Snapshot, Vec<SettingsError>> {
        let identity = (source.layer, source.source_id.clone());
        let mut sources = self.sources.clone();
        sources.insert(identity, source);
        self.resolve_staged(sources)
    }

    /// Resolves a full source replacement without mutating the runtime.
    pub fn prepare_source_update(
        &self,
        source: LayerSource,
    ) -> Result<PreparedSettingsUpdate, SettingsRuntimeError> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(SettingsRuntimeError::GenerationExhausted)?;
        let identity = (source.layer, source.source_id.clone());
        let prepared_source = source.clone();
        let mut sources = self.sources.clone();
        sources.insert(identity, source);
        let values = sources.values().cloned().collect::<Vec<_>>();
        let resolved = Resolver::new(&self.registry)
            .resolve(self.domain, &values, generation)
            .map_err(SettingsRuntimeError::Resolve)?;
        Ok(PreparedSettingsUpdate {
            base_generation: self.generation,
            source: prepared_source,
            snapshot: resolved,
        })
    }

    /// Prepares an I/O candidate source after confirming its resolved values
    /// and provenance still match the candidate produced by that pipeline.
    pub fn prepare_candidate_source(
        &self,
        source: LayerSource,
        candidate: &Snapshot,
    ) -> Result<PreparedSettingsUpdate, SettingsRuntimeError> {
        if candidate.domain() != self.domain {
            return Err(SettingsRuntimeError::CandidateMismatch);
        }
        let prepared = self.prepare_source_update(source)?;
        if !same_effective_snapshot(&prepared.snapshot, candidate) {
            return Err(SettingsRuntimeError::CandidateMismatch);
        }
        Ok(prepared)
    }

    /// Commits a prepared update if its base generation is still current.
    ///
    /// Returns the exact pre-commit source/generation checkpoint for rollback.
    /// The candidate is resolved again before publication, so the committed
    /// values and provenance always correspond to the source table installed.
    pub fn commit_prepared_source(
        &mut self,
        prepared: PreparedSettingsUpdate,
    ) -> Result<SettingsRuntimeCheckpoint, SettingsRuntimeError> {
        if prepared.base_generation != self.generation {
            return Err(SettingsRuntimeError::Stale {
                expected: prepared.base_generation,
                actual: self.generation,
            });
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(SettingsRuntimeError::GenerationExhausted)?;
        let identity = (prepared.source.layer, prepared.source.source_id.clone());
        let mut sources = self.sources.clone();
        sources.insert(identity, prepared.source);
        let values = sources.values().cloned().collect::<Vec<_>>();
        let resolved = Resolver::new(&self.registry)
            .resolve(self.domain, &values, generation)
            .map_err(SettingsRuntimeError::Resolve)?;
        if !same_effective_snapshot(&resolved, &prepared.snapshot) {
            return Err(SettingsRuntimeError::CandidateMismatch);
        }
        let checkpoint = self.checkpoint();
        self.sources = sources;
        self.generation = generation;
        self.snapshot = resolved;
        Ok(checkpoint)
    }

    /// Removes one complete source and publishes its resolved snapshot.
    ///
    /// Removing a missing source is still a successful no-op resolve, which
    /// keeps callers from needing a separate existence race check.
    pub fn remove_source(
        &mut self,
        layer: LayerKind,
        source_id: &str,
    ) -> Result<&Snapshot, Vec<SettingsError>> {
        let mut sources = self.sources.clone();
        sources.remove(&(layer, source_id.to_owned()));
        self.resolve_staged(sources)
    }

    /// Returns one complete source candidate by identity.
    #[must_use]
    pub fn source(&self, layer: LayerKind, source_id: &str) -> Option<&LayerSource> {
        self.sources.get(&(layer, source_id.to_owned()))
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
        self.resolve_staged(
            sources
                .into_iter()
                .map(|source| ((source.layer, source.source_id.clone()), source))
                .collect(),
        )
    }

    fn resolve_staged(
        &mut self,
        sources: BTreeMap<(LayerKind, String), LayerSource>,
    ) -> Result<&Snapshot, Vec<SettingsError>> {
        let source_values = sources.values().cloned().collect::<Vec<_>>();
        let generation = self.generation.saturating_add(1);
        let snapshot =
            Resolver::new(&self.registry).resolve(self.domain, &source_values, generation)?;
        self.sources = sources;
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

fn same_effective_snapshot(left: &Snapshot, right: &Snapshot) -> bool {
    left.domain() == right.domain()
        && left.len() == right.len()
        && (0..left.len()).all(|index| {
            let id = SettingId(index as u32);
            left.get(id) == right.get(id) && left.source(id) == right.source(id)
        })
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

    #[test]
    fn replacing_a_source_removes_stale_keys_and_is_atomic_on_rejection() {
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.namespace("audio", "test").unwrap();
        let mut render = SettingDef::new("render.scale", "render", Value::Float(1.0));
        render.constraints.min = Some(0.25);
        render.constraints.max = Some(2.0);
        builder.register(render);
        builder.register(SettingDef::new("audio.volume", "audio", Value::Int(50)));
        let registry = builder.finish().unwrap();
        let mut runtime = SettingsRuntime::new(registry, Domain::Runtime).unwrap();

        let mut source = LayerSource::new(LayerKind::User, TrustLevel::User, "user-file");
        source.set("render.scale", Value::Float(0.75));
        source.set("audio.volume", Value::Int(80));
        runtime.replace_source(source).unwrap();
        assert!(runtime
            .source(LayerKind::User, "user-file")
            .is_some_and(|source| {
                source
                    .values
                    .contains_key(&SettingKey::from("audio.volume"))
            }));

        let mut replacement = LayerSource::new(LayerKind::User, TrustLevel::User, "user-file");
        replacement.set("render.scale", Value::Float(1.25));
        runtime.replace_source(replacement).unwrap();
        assert!(runtime
            .source(LayerKind::User, "user-file")
            .is_some_and(|source| {
                !source
                    .values
                    .contains_key(&SettingKey::from("audio.volume"))
            }));
        let audio_id = runtime
            .registry()
            .id(&SettingKey::from("audio.volume"))
            .unwrap();
        assert_eq!(runtime.snapshot().get(audio_id), Some(&Value::Int(50)));

        let before = runtime.snapshot().clone();
        let mut invalid = LayerSource::new(LayerKind::User, TrustLevel::User, "user-file");
        invalid.set("render.scale", Value::Float(9.0));
        assert!(runtime.replace_source(invalid).is_err());
        assert_eq!(runtime.snapshot(), &before);
        assert_eq!(
            runtime.source(LayerKind::User, "user-file").unwrap().values
                [&SettingKey::from("render.scale")],
            Value::Float(1.25)
        );
    }

    #[test]
    fn checkpoint_restores_source_table_and_snapshot_as_one_state() {
        let mut runtime = SettingsRuntime::new(registry(), Domain::Runtime).unwrap();
        let key = SettingKey::from("render.scale");
        runtime.set(
            LayerKind::User,
            TrustLevel::User,
            "user-file",
            key.clone(),
            Value::Float(0.75),
        );
        runtime.resolve().unwrap();
        let checkpoint = runtime.checkpoint();
        let checkpoint_generation = runtime.generation();

        runtime.remove_source(LayerKind::User, "user-file").unwrap();
        assert_eq!(
            runtime.snapshot().get(runtime.registry().id(&key).unwrap()),
            Some(&Value::Float(1.0))
        );
        runtime.restore_checkpoint(checkpoint).unwrap();

        assert_eq!(runtime.generation(), checkpoint_generation);
        assert_eq!(runtime.snapshot().generation(), checkpoint_generation);
        assert_eq!(
            runtime.snapshot().get(runtime.registry().id(&key).unwrap()),
            Some(&Value::Float(0.75))
        );
        assert!(runtime.source(LayerKind::User, "user-file").is_some());
    }

    #[test]
    fn candidate_install_rejects_stale_values_and_keeps_runtime_unchanged() {
        let mut runtime = SettingsRuntime::new(registry(), Domain::Runtime).unwrap();
        let key = SettingKey::from("render.scale");
        let source = LayerSource::new(LayerKind::User, TrustLevel::User, "user-file");
        let prepared = runtime.prepare_source_update(source).unwrap();

        runtime.set(
            LayerKind::Runtime,
            TrustLevel::User,
            "console",
            key.clone(),
            Value::Float(0.5),
        );
        runtime.resolve().unwrap();
        let before = runtime.checkpoint();
        assert!(matches!(
            runtime.commit_prepared_source(prepared),
            Err(SettingsRuntimeError::Stale { .. })
        ));
        assert_eq!(runtime.checkpoint(), before);
    }

    #[test]
    fn prepared_source_commit_returns_a_rollback_checkpoint() {
        let mut runtime = SettingsRuntime::new(registry(), Domain::Runtime).unwrap();
        let key = SettingKey::from("render.scale");
        let mut source = LayerSource::new(LayerKind::User, TrustLevel::User, "user-file");
        source.set(key.clone(), Value::Float(0.75));
        let prepared = runtime.prepare_source_update(source).unwrap();
        let checkpoint = runtime.commit_prepared_source(prepared).unwrap();
        assert_eq!(
            runtime.snapshot().get(runtime.registry().id(&key).unwrap()),
            Some(&Value::Float(0.75))
        );
        runtime.restore_checkpoint(checkpoint).unwrap();
        assert_eq!(
            runtime.snapshot().get(runtime.registry().id(&key).unwrap()),
            Some(&Value::Float(1.0))
        );
        assert!(runtime.source(LayerKind::User, "user-file").is_none());
    }
}
