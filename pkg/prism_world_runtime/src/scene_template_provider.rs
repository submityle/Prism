//! Bounded loading of published scene templates through the runtime asset
//! manifest and mounted container overlay.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use prism_asset::{AssetEvent, AssetId, Assets, StableGuid};
use prism_asset_container::{ManifestReadError, ManifestReader, OverlaySet, RuntimeManifest};
use prism_scene::{
    decode_template, SceneTemplate, SchemaMigrationError, SchemaMigrationRegistry,
    TemplateCodecError, TemplateDecodeLimits, SCENE_TEMPLATE_KIND, SCENE_TEMPLATE_SCHEMA,
};

/// One scene-template identity invalidated by an asset event batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SceneTemplateInvalidation {
    /// The directly changed or removed template.
    pub changed: StableGuid,
    /// A cached transitive parent template whose dependency closure is stale.
    /// Direct changes have dependent equal to changed.
    pub dependent: StableGuid,
    /// True when the dependent template remains available and can be reloaded.
    pub reloadable: bool,
}

/// Result of committing one template asset batch and admitting resident reloads.
/// The invalidation closure is retained so other live owners, such as nested
/// scene trees, can consume the exact same dependency update.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SceneTemplateApplyOutcome {
    /// Deterministic transitive invalidation closure produced by the cache.
    pub invalidations: Vec<SceneTemplateInvalidation>,
    /// Resident sectors admitted for budgeted reload, in stable order.
    pub reload_sectors: Vec<crate::SectorId>,
}

/// Stable key for a live nested tree owned by [`NestedSceneTreeRegistry`].
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NestedSceneTreeId(u64);

impl NestedSceneTreeId {
    /// Returns the stable numeric value of this registration.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A registered-tree reload failed; ownership remains in the registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NestedSceneTreeRegistryError {
    /// Cause returned by nested-tree candidate staging.
    pub cause: prism_scene::NestedInstantiateError,
}

/// Per-world ownership registry for live nested scene trees. Register every
/// tree root generation with its owning application; committed template
/// invalidations then update matching registrations as one staged batch.
#[derive(Default)]
pub struct NestedSceneTreeRegistry {
    trees: BTreeMap<NestedSceneTreeId, prism_scene::SceneTreeSpawn>,
    next_id: u64,
    pending_invalidations: BTreeSet<StableGuid>,
}

impl NestedSceneTreeRegistry {
    /// Creates an empty tree registry. IDs are stable for each registration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            trees: BTreeMap::new(),
            next_id: 0,
            pending_invalidations: BTreeSet::new(),
        }
    }

    /// Registers ownership of a newly spawned live tree.
    pub fn register(
        &mut self,
        tree: prism_scene::SceneTreeSpawn,
    ) -> Result<NestedSceneTreeId, prism_scene::SceneTreeSpawn> {
        let Some(next) = self.next_id.checked_add(1) else {
            return Err(tree);
        };
        let id = NestedSceneTreeId(self.next_id);
        self.next_id = next;
        let previous = self.trees.insert(id, tree);
        debug_assert!(previous.is_none());
        Ok(id)
    }

    /// Returns the number of registered live trees.
    #[must_use]
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    /// Returns whether the registry owns no live trees.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// Borrows a registered tree by stable ID.
    #[must_use]
    pub fn get(&self, id: NestedSceneTreeId) -> Option<&prism_scene::SceneTreeSpawn> {
        self.trees.get(&id)
    }

    /// Removes ownership without despawning the tree.
    pub fn unregister(&mut self, id: NestedSceneTreeId) -> Option<prism_scene::SceneTreeSpawn> {
        self.trees.remove(&id)
    }

    /// Despawns a registered tree and removes its ownership record.
    pub fn despawn(
        &mut self,
        world: &mut prism_ecs::world::World,
        id: NestedSceneTreeId,
    ) -> Option<usize> {
        self.unregister(id).map(|tree| tree.despawn_tree(world))
    }

    /// Applies a committed template invalidation closure to all owned trees.
    /// Failed batches restore each tree under its original ID and preserve the
    /// complete invalidation set for [`Self::retry_pending`].
    pub fn apply<P: prism_scene::NestedSceneProvider>(
        &mut self,
        world: &mut prism_ecs::world::World,
        outcome: &SceneTemplateApplyOutcome,
        provider: &P,
        limits: prism_scene::NestedSpawnLimits,
    ) -> Result<(), NestedSceneTreeRegistryError> {
        self.pending_invalidations.extend(
            outcome
                .invalidations
                .iter()
                .filter(|event| event.reloadable)
                .map(|event| event.dependent),
        );
        self.reload_pending(world, provider, limits)
    }

    /// Retries the previously failed invalidation closure without requiring a
    /// duplicate asset event batch. Successful publication clears the queue.
    pub fn retry_pending<P: prism_scene::NestedSceneProvider>(
        &mut self,
        world: &mut prism_ecs::world::World,
        provider: &P,
        limits: prism_scene::NestedSpawnLimits,
    ) -> Result<(), NestedSceneTreeRegistryError> {
        self.reload_pending(world, provider, limits)
    }

    fn reload_pending<P: prism_scene::NestedSceneProvider>(
        &mut self,
        world: &mut prism_ecs::world::World,
        provider: &P,
        limits: prism_scene::NestedSpawnLimits,
    ) -> Result<(), NestedSceneTreeRegistryError> {
        if self.pending_invalidations.is_empty() || self.trees.is_empty() {
            return Ok(());
        }
        let ids: Vec<_> = self.trees.keys().copied().collect();
        let trees: Vec<_> = ids
            .iter()
            .map(|id| {
                self.trees
                    .remove(id)
                    .expect("tree key collected from registry")
            })
            .collect();
        let invalidated: Vec<_> = self.pending_invalidations.iter().copied().collect();
        match prism_scene::reload_invalidated_nested_trees(
            world,
            trees,
            &invalidated,
            provider,
            limits,
        ) {
            Ok(replacements) => {
                for (id, tree) in ids.into_iter().zip(replacements) {
                    let previous = self.trees.insert(id, tree);
                    debug_assert!(previous.is_none());
                }
                self.pending_invalidations.clear();
                Ok(())
            }
            Err(error) => {
                let prism_scene::NestedReloadBatchError { cause, trees } = error;
                for (id, tree) in ids.into_iter().zip(trees) {
                    let previous = self.trees.insert(id, tree);
                    debug_assert!(previous.is_none());
                }
                Err(NestedSceneTreeRegistryError { cause })
            }
        }
    }
}

/// An asset event batch violated atomic event/snapshot pairing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneTemplateBatchError {
    /// An added or modified asset has no snapshot in the batch.
    MissingSnapshot(AssetId<SceneTemplate>),
    /// The same generational asset id appears more than once in snapshots.
    DuplicateSnapshot(AssetId<SceneTemplate>),
    /// An added or modified slot contains an invalid scene-template snapshot.
    InvalidSnapshot(AssetId<SceneTemplate>),
}

/// Shared runtime scene-template cache with deterministic reverse invalidation.
#[derive(Clone, Default)]
pub struct SceneTemplateCache {
    templates: BTreeMap<StableGuid, Arc<SceneTemplate>>,
    dependencies: BTreeMap<StableGuid, BTreeSet<StableGuid>>,
    dependents: BTreeMap<StableGuid, BTreeSet<StableGuid>>,
    asset_guids: BTreeMap<AssetId<SceneTemplate>, StableGuid>,
    guid_assets: BTreeMap<StableGuid, BTreeSet<AssetId<SceneTemplate>>>,
    retired_assets: BTreeSet<AssetId<SceneTemplate>>,
}

impl SceneTemplateCache {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a cached template by stable identity.
    #[must_use]
    pub fn get(&self, guid: StableGuid) -> Option<&Arc<SceneTemplate>> {
        self.templates.get(&guid)
    }

    /// Returns the number of currently cached templates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether this cache has no loaded templates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Applies one atomic scene-asset event/snapshot batch.
    /// Duplicate events collapse to one update, and generational asset ids
    /// prevent stale events from mutating a reused slot's current template.
    pub fn apply_asset_batch(
        &mut self,
        events: &[AssetEvent<SceneTemplate>],
        snapshots: &[(AssetId<SceneTemplate>, SceneTemplate)],
    ) -> Result<Vec<SceneTemplateInvalidation>, SceneTemplateBatchError> {
        let mut snapshot_by_id = BTreeMap::new();
        for (id, value) in snapshots {
            if snapshot_by_id.insert(*id, value).is_some() {
                return Err(SceneTemplateBatchError::DuplicateSnapshot(*id));
            }
        }
        let mut additions = BTreeSet::new();
        let mut updates = BTreeSet::new();
        let mut removals = BTreeSet::new();
        for event in events {
            match event {
                AssetEvent::Added { id } => {
                    additions.insert(*id);
                    updates.insert(*id);
                    removals.remove(id);
                    self.retired_assets.remove(id);
                }
                AssetEvent::Modified { id } => {
                    updates.insert(*id);
                    removals.remove(id);
                }
                AssetEvent::Removed { id } => {
                    updates.remove(id);
                    removals.insert(*id);
                    self.retired_assets.insert(*id);
                }
                // Failed replacement loads preserve the cache's last-known-good
                // value. Only a committed Removed event releases its identity.
                AssetEvent::Failed { id, .. } => {
                    additions.remove(id);
                    updates.remove(id);
                    removals.remove(id);
                }
                AssetEvent::Unused { .. } | AssetEvent::LoadedWithDependencies { .. } => {}
            }
        }

        for id in &updates {
            if !snapshot_by_id.contains_key(id) {
                return Err(SceneTemplateBatchError::MissingSnapshot(*id));
            }
            let template = snapshot_by_id[id];
            if template.guid == StableGuid::NIL || template.validate().is_err() {
                return Err(SceneTemplateBatchError::InvalidSnapshot(*id));
            }
        }

        let mut changed = BTreeMap::new();
        for id in removals {
            if let Some(guid) = self.remove_asset(id) {
                changed.insert(guid, false);
            }
        }
        for id in updates {
            // Modified is meaningful only for a slot this cache already
            // tracks. This prevents an old queued event from resurrecting a
            // template after explicit cache removal or after slot reuse.
            if !additions.contains(&id)
                && !self.asset_guids.contains_key(&id)
                && self.retired_assets.contains(&id)
            {
                continue;
            }
            let template = snapshot_by_id[&id];
            if let Some(old_guid) = self.asset_guids.get(&id).copied() {
                if old_guid != template.guid {
                    if self.remove_asset(id).is_some() {
                        changed.insert(old_guid, false);
                    }
                }
            }
            self.asset_guids.insert(id, template.guid);
            self.guid_assets
                .entry(template.guid)
                .or_default()
                .insert(id);
            self.insert_template(Arc::new((*template).clone()));
            changed.insert(template.guid, true);
        }
        Ok(self.propagate_invalidations(changed))
    }

    /// Atomically drains and applies the owning scene asset arena's events.
    pub fn apply_assets(
        &mut self,
        assets: &mut Assets<SceneTemplate>,
    ) -> Result<Vec<SceneTemplateInvalidation>, SceneTemplateBatchError> {
        let (events, snapshots) = assets.drain_events_with_snapshots();
        self.apply_asset_batch(&events, &snapshots)
    }

    /// Publishes a decoded template that did not originate from Assets.
    pub fn insert(
        &mut self,
        template: SceneTemplate,
    ) -> Result<Vec<SceneTemplateInvalidation>, SceneTemplateLoadError> {
        template
            .validate()
            .map_err(|_| SceneTemplateLoadError::InvalidTemplate)?;
        let guid = template.guid;
        self.insert_template(Arc::new(template));
        Ok(self.propagate_invalidations([(guid, true)].into_iter().collect()))
    }

    /// Removes a template and invalidates every cached transitive parent.
    pub fn remove(&mut self, guid: StableGuid) -> Vec<SceneTemplateInvalidation> {
        if let Some(ids) = self.guid_assets.remove(&guid) {
            for id in ids {
                self.asset_guids.remove(&id);
                self.retired_assets.insert(id);
            }
        }
        self.remove_guid(guid);
        self.propagate_invalidations([(guid, false)].into_iter().collect())
    }

    fn insert_template(&mut self, template: Arc<SceneTemplate>) {
        let guid = template.guid;
        self.remove_edges(guid);
        let deps: BTreeSet<_> = template.deps.iter().copied().collect();
        for dependency in &deps {
            self.dependents.entry(*dependency).or_default().insert(guid);
        }
        self.dependencies.insert(guid, deps);
        self.templates.insert(guid, template);
    }

    fn remove_asset(&mut self, id: AssetId<SceneTemplate>) -> Option<StableGuid> {
        let guid = self.asset_guids.remove(&id)?;
        if let Some(owners) = self.guid_assets.get_mut(&guid) {
            owners.remove(&id);
            if owners.is_empty() {
                self.guid_assets.remove(&guid);
                self.remove_guid(guid);
                return Some(guid);
            }
        }
        None
    }

    fn remove_guid(&mut self, guid: StableGuid) {
        self.templates.remove(&guid);
        self.remove_edges(guid);
    }

    fn remove_edges(&mut self, guid: StableGuid) {
        if let Some(deps) = self.dependencies.remove(&guid) {
            for dependency in deps {
                if let Some(parents) = self.dependents.get_mut(&dependency) {
                    parents.remove(&guid);
                    if parents.is_empty() {
                        self.dependents.remove(&dependency);
                    }
                }
            }
        }
    }

    fn propagate_invalidations(
        &self,
        changed: BTreeMap<StableGuid, bool>,
    ) -> Vec<SceneTemplateInvalidation> {
        let mut output = Vec::new();
        for (&guid, &reloadable) in &changed {
            output.push(SceneTemplateInvalidation {
                changed: guid,
                dependent: guid,
                reloadable,
            });
            let mut stack = vec![guid];
            let mut seen = BTreeSet::from([guid]);
            while let Some(current) = stack.pop() {
                if let Some(parents) = self.dependents.get(&current) {
                    for &parent in parents {
                        if seen.insert(parent) {
                            stack.push(parent);
                            output.push(SceneTemplateInvalidation {
                                changed: guid,
                                dependent: parent,
                                reloadable: self.templates.contains_key(&parent),
                            });
                        }
                    }
                }
            }
        }
        output.sort_by_key(|entry| (entry.changed, entry.dependent));
        output
    }
}

/// Errors returned while resolving a published scene-template route.
#[derive(Debug)]
pub enum SceneTemplateLoadError {
    /// The route does not exist or cannot be bound to the mounted overlay.
    Manifest(ManifestReadError),
    /// The route schema does not match the supported PRST schema.
    Schema {
        /// Required PRST schema.
        expected: u32,
        /// Schema found on the route.
        actual: u32,
    },
    /// A known source schema could not be migrated to the current runtime schema.
    Migration(SchemaMigrationError),
    /// The route schema did not match the payload's declared source schema.
    PayloadSchemaMismatch {
        /// Content schema recorded by the runtime route.
        route: u32,
        /// Content schema declared in the PRST payload.
        payload: u32,
    },
    /// The route kind is not the compiled scene-template artifact kind.
    Kind {
        /// Required scene-template artifact kind.
        expected: u128,
        /// Artifact kind found on the route.
        actual: u128,
    },
    /// The container payload exceeds the caller's bound.
    PayloadTooLarge {
        /// Actual uncompressed payload size.
        actual: u64,
        /// Caller-provided size bound.
        limit: u64,
    },
    /// The bounded payload could not be represented by this platform.
    PayloadLengthOverflow(u64),
    /// The route payload is not a valid bounded PRST template.
    Template(TemplateCodecError),
    /// The decoded template identity differs from the manifest route.
    GuidMismatch {
        /// GUID used to resolve the manifest route.
        route: StableGuid,
        /// GUID decoded from the PRST payload.
        template: StableGuid,
    },
    /// The route dependency closure differs from the dependencies embedded in
    /// the validated scene template.
    DependencyMismatch,
    /// A directly published runtime template failed structural validation.
    InvalidTemplate,
}

impl fmt::Display for SceneTemplateLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(f, "scene template manifest lookup failed: {error}"),
            Self::Schema { expected, actual } => write!(
                f,
                "scene template route schema {actual} is unsupported; expected {expected}"
            ),
            Self::Migration(error) => {
                write!(f, "scene template schema migration failed: {error:?}")
            }
            Self::PayloadSchemaMismatch { route, payload } => write!(
                f,
                "scene template route schema {route} disagrees with payload schema {payload}"
            ),
            Self::Kind { expected, actual } => write!(
                f,
                "scene template route kind {actual:#034x} is unsupported; expected {expected:#034x}"
            ),
            Self::PayloadTooLarge { actual, limit } => write!(
                f,
                "scene template payload is {actual} bytes, exceeding the {limit}-byte limit"
            ),
            Self::PayloadLengthOverflow(length) => write!(
                f,
                "scene template payload length {length} cannot be represented on this platform"
            ),
            Self::Template(error) => write!(f, "invalid scene template payload: {error:?}"),
            Self::GuidMismatch { route, template } => write!(
                f,
                "scene template GUID {template:?} does not match route GUID {route:?}"
            ),
            Self::DependencyMismatch => {
                f.write_str("scene template dependencies disagree with runtime manifest routes")
            }
            Self::InvalidTemplate => f.write_str("scene template failed runtime validation"),
        }
    }
}

impl std::error::Error for SceneTemplateLoadError {}

/// Loads one published PRST template using a validated runtime manifest.
///
/// The route schema and payload size are checked before allocating the payload
/// buffer. After bounded decoding, the scene GUID and canonical dependency
/// list are cross-checked against the runtime route metadata. Container mmap
/// can avoid copying the container backing, but this API currently decodes
/// PRST into owned scene columns; it does not claim zero-copy scene loading.
pub fn load_scene_template(
    manifest: &RuntimeManifest,
    overlays: &OverlaySet,
    guid: StableGuid,
    max_payload_bytes: u64,
    decode_limits: TemplateDecodeLimits,
) -> Result<SceneTemplate, SceneTemplateLoadError> {
    load_scene_template_with_migrations(
        manifest,
        overlays,
        guid,
        max_payload_bytes,
        decode_limits,
        &SchemaMigrationRegistry::default(),
    )
}

/// Loads a scene template with application-provided semantic migrations.
///
/// The default [`load_scene_template`] includes only the built-in v1-to-v2
/// identity migration. Applications that change component semantics should
/// pass a registry containing their typed conversions here.
pub fn load_scene_template_with_migrations(
    manifest: &RuntimeManifest,
    overlays: &OverlaySet,
    guid: StableGuid,
    max_payload_bytes: u64,
    decode_limits: TemplateDecodeLimits,
    migrations: &SchemaMigrationRegistry,
) -> Result<SceneTemplate, SceneTemplateLoadError> {
    let reader = manifest
        .bind(overlays)
        .map_err(SceneTemplateLoadError::Manifest)?;
    decode_routed_template(
        &reader,
        manifest,
        guid,
        max_payload_bytes,
        decode_limits,
        migrations,
    )
}

fn decode_routed_template(
    reader: &ManifestReader<'_, '_>,
    manifest: &RuntimeManifest,
    guid: StableGuid,
    max_payload_bytes: u64,
    decode_limits: TemplateDecodeLimits,
    migrations: &SchemaMigrationRegistry,
) -> Result<SceneTemplate, SceneTemplateLoadError> {
    let metadata = manifest
        .metadata(guid)
        .ok_or_else(|| SceneTemplateLoadError::Manifest(ManifestReadError::MissingRoute(guid)))?;
    if metadata.kind != SCENE_TEMPLATE_KIND {
        return Err(SceneTemplateLoadError::Kind {
            expected: SCENE_TEMPLATE_KIND,
            actual: metadata.kind,
        });
    }
    if metadata.schema == 0 || metadata.schema > SCENE_TEMPLATE_SCHEMA {
        return Err(SceneTemplateLoadError::Schema {
            expected: SCENE_TEMPLATE_SCHEMA,
            actual: metadata.schema,
        });
    }
    let entry = reader
        .entry_info(guid)
        .map_err(SceneTemplateLoadError::Manifest)?;
    if entry.raw_len > max_payload_bytes {
        return Err(SceneTemplateLoadError::PayloadTooLarge {
            actual: entry.raw_len,
            limit: max_payload_bytes,
        });
    }
    let payload_len = usize::try_from(entry.raw_len)
        .map_err(|_| SceneTemplateLoadError::PayloadLengthOverflow(entry.raw_len))?;
    let payload = reader
        .read_all(guid)
        .map_err(SceneTemplateLoadError::Manifest)?;
    if payload.len() != payload_len {
        return Err(SceneTemplateLoadError::Template(
            TemplateCodecError::Truncated,
        ));
    }
    let template =
        decode_template(&payload, decode_limits).map_err(SceneTemplateLoadError::Template)?;
    if template.schema != metadata.schema {
        return Err(SceneTemplateLoadError::PayloadSchemaMismatch {
            route: metadata.schema,
            payload: template.schema,
        });
    }
    if template.guid != guid {
        return Err(SceneTemplateLoadError::GuidMismatch {
            route: guid,
            template: template.guid,
        });
    }

    let mut routed_dependencies = metadata.hard_dependencies.clone();
    routed_dependencies.extend_from_slice(&metadata.optional_dependencies);
    routed_dependencies.sort_unstable();
    routed_dependencies.dedup();
    if template.deps.as_ref() != routed_dependencies.as_slice() {
        return Err(SceneTemplateLoadError::DependencyMismatch);
    }
    migrations
        .migrate(template, SCENE_TEMPLATE_SCHEMA)
        .map_err(SceneTemplateLoadError::Migration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use prism_asset::{AssetDependency, AssetTypeId, Assets};
    use prism_asset_container::{ContainerReader, ContainerWriter, ManifestMetadata, OverlaySet};
    use prism_ecs::component::Component;
    use prism_ecs::world::World;
    use prism_scene::{
        encode_template, ArchetypeGroup, InstantiationPlan, ResidencyPlan, SceneTemplate,
    };
    use prism_scene::{
        spawn_nested_typed, LocalEntityId, NestedProviderError, NestedSceneProvider,
        NestedSpawnLimits, SceneInstance, SceneSpawn, SceneTreeSpawn, TypedSpawnPlan,
    };

    #[derive(Debug)]
    struct RuntimeMarker;
    impl Component for RuntimeMarker {}

    struct NestedProvider {
        templates: BTreeMap<StableGuid, SceneTemplate>,
    }

    impl NestedSceneProvider for NestedProvider {
        fn template(&self, guid: StableGuid) -> Option<&SceneTemplate> {
            self.templates.get(&guid)
        }

        fn typed_plan(&self, _guid: StableGuid) -> Result<TypedSpawnPlan, NestedProviderError> {
            Ok(TypedSpawnPlan::new().with_bundle(LocalEntityId::new(0), RuntimeMarker))
        }

        fn replay_instance_overrides(
            &self,
            _world: &mut World,
            _template: StableGuid,
            _previous: &SceneInstance,
            _replacement: &SceneSpawn,
        ) -> Result<(), NestedProviderError> {
            Ok(())
        }
    }

    fn live_tree(guid: StableGuid) -> (World, SceneTreeSpawn, NestedProvider) {
        let mut template = template(Vec::new());
        template.guid = guid;
        let provider = NestedProvider {
            templates: BTreeMap::from([(guid, template.clone())]),
        };
        let mut world = World::new();
        let tree = spawn_nested_typed(
            &mut world,
            &template,
            TypedSpawnPlan::new().with_bundle(LocalEntityId::new(0), RuntimeMarker),
            &provider,
            NestedSpawnLimits::default(),
        )
        .expect("fixture tree spawns");
        (world, tree, provider)
    }

    fn fixture(
        template: &SceneTemplate,
        payload_override: Option<Vec<u8>>,
        schema: u32,
        hard_dependencies: Vec<StableGuid>,
    ) -> (RuntimeManifest, OverlaySet, StableGuid, Vec<u8>) {
        let payload = payload_override.unwrap_or_else(|| encode_template(template).unwrap());
        let guid = template.guid;
        let mut writer = ContainerWriter::new();
        writer.add(guid, payload.clone()).unwrap();
        let bytes = writer.finish().unwrap();
        let reader = ContainerReader::new(bytes).unwrap();
        let mut overlays = OverlaySet::new();
        overlays.mount("scene", 10, reader);
        let mut manifest = overlays.runtime_manifest().unwrap();
        manifest
            .set_metadata(
                guid,
                ManifestMetadata {
                    kind: SCENE_TEMPLATE_KIND,
                    hard_dependencies,
                    schema,
                    ..ManifestMetadata::default()
                },
            )
            .unwrap();
        (manifest, overlays, guid, payload)
    }

    fn template(deps: Vec<StableGuid>) -> SceneTemplate {
        SceneTemplate {
            guid: StableGuid::from_u128(0x5151),
            schema: SCENE_TEMPLATE_SCHEMA,
            plan: InstantiationPlan {
                entity_count: 1,
                groups: vec![ArchetypeGroup::new(
                    prism_scene::ArchetypeSignature::default(),
                    vec![prism_scene::LocalEntityId::new(0)].into_boxed_slice(),
                    Vec::new().into_boxed_slice(),
                )]
                .into_boxed_slice(),
                hierarchy: Vec::new().into_boxed_slice(),
                entity_valued_fields: Vec::new().into_boxed_slice(),
                reserved_local_base: prism_scene::LocalEntityId::RESERVED_BASE,
            },
            names: Default::default(),
            nested: Vec::new().into_boxed_slice(),
            deps: deps.into_boxed_slice(),
            tiers: ResidencyPlan::new(),
        }
    }

    #[test]
    fn scene_template_asset_reports_soft_guid_dependencies() {
        let dependency = StableGuid::from_u128(0x7171);
        let template = template(vec![dependency]);
        let mut references = Vec::new();
        prism_asset::Asset::visit_references(&template, &mut |reference| {
            references.push(reference)
        });
        assert_eq!(
            references,
            vec![AssetDependency::Soft {
                guid: dependency,
                type_id: AssetTypeId::of::<SceneTemplate>(),
            }]
        );
    }

    #[test]
    fn registered_nested_tree_retries_failed_invalidation_without_losing_ownership() {
        let guid = StableGuid::from_u128(0x8282);
        let (mut world, tree, provider) = live_tree(guid);
        let previous_root = tree.root.root.expect("fixture root");
        let mut registry = NestedSceneTreeRegistry::new();
        let id = registry.register(tree).expect("registration id available");
        let outcome = SceneTemplateApplyOutcome {
            invalidations: vec![SceneTemplateInvalidation {
                changed: guid,
                dependent: guid,
                reloadable: true,
            }],
            reload_sectors: Vec::new(),
        };
        let unavailable = NestedProvider {
            templates: BTreeMap::new(),
        };

        assert_eq!(
            registry.apply(
                &mut world,
                &outcome,
                &unavailable,
                NestedSpawnLimits::default(),
            ),
            Err(NestedSceneTreeRegistryError {
                cause: prism_scene::NestedInstantiateError::MissingTemplate(guid),
            })
        );
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get(id).unwrap().root.root, Some(previous_root));
        assert!(world.contains(previous_root));

        registry
            .retry_pending(&mut world, &provider, NestedSpawnLimits::default())
            .expect("pending invalidation retries");
        let current_root = registry.get(id).unwrap().root.root.unwrap();
        assert_ne!(current_root, previous_root);
        assert!(!world.contains(previous_root));
        assert!(world.contains(current_root));
        assert_eq!(registry.despawn(&mut world, id), Some(1));
        assert!(registry.is_empty());
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn asset_event_batch_updates_cache_and_invalidates_transitive_parents() {
        let leaf_guid = StableGuid::from_u128(0x101);
        let middle_guid = StableGuid::from_u128(0x202);
        let mut cache = SceneTemplateCache::new();
        cache.insert(template(vec![middle_guid])).unwrap();
        let root_guid = template(vec![middle_guid]).guid;
        cache
            .insert(SceneTemplate {
                guid: middle_guid,
                deps: vec![leaf_guid].into_boxed_slice(),
                ..template(Vec::new())
            })
            .unwrap();
        cache
            .insert(SceneTemplate {
                guid: leaf_guid,
                ..template(Vec::new())
            })
            .unwrap();

        let mut assets = Assets::new();
        let handle = assets.insert(SceneTemplate {
            guid: leaf_guid,
            ..template(Vec::new())
        });
        let added = cache.apply_assets(&mut assets).unwrap();
        assert!(added
            .iter()
            .any(|event| event.changed == leaf_guid && event.dependent == root_guid));
        assert!(cache.get(leaf_guid).is_some());

        assets.get_mut(handle.id()).unwrap().schema += 1;
        let modified = cache.apply_assets(&mut assets).unwrap();
        assert_eq!(modified.len(), 3);
        assert!(modified
            .iter()
            .any(|event| event.changed == leaf_guid && event.dependent == middle_guid));
        assert!(modified
            .iter()
            .any(|event| event.changed == leaf_guid && event.dependent == root_guid));

        assert!(assets.remove(handle.id()).is_some());
        let removed = cache.apply_assets(&mut assets).unwrap();
        assert_eq!(cache.get(leaf_guid), None);
        assert!(removed
            .iter()
            .any(|event| event.changed == leaf_guid && !event.reloadable));
        assert!(cache.get(middle_guid).is_some());
    }

    #[test]
    fn failed_asset_replacement_keeps_last_known_good_and_duplicate_guid_slots() {
        let guid = StableGuid::from_u128(0x4141);
        let template = SceneTemplate {
            guid,
            ..template(Vec::new())
        };
        let mut cache = SceneTemplateCache::new();
        let mut assets = Assets::new();
        let first = assets.insert(template.clone());
        let second = assets.insert(template.clone());
        cache.apply_assets(&mut assets).unwrap();
        assert!(cache.get(guid).is_some());

        let error = prism_asset::ErrorRegistry::new()
            .record(prism_asset::AssetError::new("scene.prst", "reload failed"));
        assert!(assets.fail(first.id(), error));
        let invalidations = cache.apply_assets(&mut assets).unwrap();
        assert!(invalidations.is_empty());
        assert!(cache.get(guid).is_some());

        assert!(assets.remove(first.id()).is_none());
        let invalidations = cache.apply_assets(&mut assets).unwrap();
        assert!(invalidations.is_empty());
        assert!(cache.get(guid).is_some());
        assert!(assets.contains(second.id()));
    }

    #[test]
    fn explicit_remove_clears_slot_routes_and_missing_modified_snapshot_evicts_stale_data() {
        let guid = StableGuid::from_u128(0x4242);
        let mut cache = SceneTemplateCache::new();
        let mut assets = Assets::new();
        let handle = assets.insert(SceneTemplate {
            guid,
            ..template(Vec::new())
        });
        cache.apply_assets(&mut assets).unwrap();
        assert!(cache.get(guid).is_some());

        let invalidation = cache.remove(guid);
        assert!(invalidation
            .iter()
            .any(|entry| entry.changed == guid && !entry.reloadable));
        let event = AssetEvent::Modified { id: handle.id() };
        let resurrected = cache
            .apply_asset_batch(
                &[event],
                &[(
                    handle.id(),
                    SceneTemplate {
                        guid,
                        ..template(Vec::new())
                    },
                )],
            )
            .unwrap();
        assert!(resurrected.is_empty());
        assert!(cache.get(guid).is_none());

        // A fresh Added event is an explicit re-publication.
        cache
            .apply_asset_batch(
                &[AssetEvent::Added { id: handle.id() }],
                &[(
                    handle.id(),
                    SceneTemplate {
                        guid,
                        ..template(Vec::new())
                    },
                )],
            )
            .unwrap();
        assert!(cache.get(guid).is_some());
        assert!(matches!(
            cache.apply_asset_batch(&[AssetEvent::Modified { id: handle.id() }], &[]),
            Err(SceneTemplateBatchError::MissingSnapshot(_))
        ));
        assert!(cache.get(guid).is_some());
    }

    #[test]
    fn scene_template_cache_keeps_last_known_good_when_failure_follows_modified_event() {
        let guid = StableGuid::from_u128(0x4243);
        let mut assets = Assets::new();
        let handle = assets.insert(SceneTemplate {
            guid,
            ..template(Vec::new())
        });
        let mut cache = SceneTemplateCache::new();
        cache.apply_assets(&mut assets).unwrap();

        assets.get_mut(handle.id()).unwrap().schema += 1;
        let error = prism_asset::ErrorRegistry::new().record(prism_asset::AssetError::new(
            "scene.prst",
            "replacement failed",
        ));
        assert!(assets.fail(handle.id(), error));
        assert!(cache.apply_assets(&mut assets).unwrap().is_empty());
        assert_eq!(cache.get(guid).unwrap().schema, SCENE_TEMPLATE_SCHEMA);
    }

    #[test]
    fn batch_rejects_missing_or_duplicate_event_snapshots_without_mutation() {
        let guid = StableGuid::from_u128(0x4343);
        let template = SceneTemplate {
            guid,
            ..template(Vec::new())
        };
        let mut cache = SceneTemplateCache::new();
        let mut assets = Assets::new();
        let handle = assets.insert(template.clone());
        let events = assets.drain_events();
        assert!(matches!(
            cache.apply_asset_batch(&events, &[]),
            Err(SceneTemplateBatchError::MissingSnapshot(_))
        ));
        assert!(cache.is_empty());

        let snapshot = (handle.id(), template);
        assert!(matches!(
            cache.apply_asset_batch(&events, &[snapshot.clone(), snapshot]),
            Err(SceneTemplateBatchError::DuplicateSnapshot(_))
        ));
        assert!(cache.is_empty());
    }

    #[test]
    fn batch_rejects_invalid_snapshot_before_committing_other_slots() {
        let valid_guid = StableGuid::from_u128(0x4344);
        let invalid_guid = StableGuid::from_u128(0x4345);
        let mut cache = SceneTemplateCache::new();
        let mut assets = Assets::new();
        let valid = assets.insert(SceneTemplate {
            guid: valid_guid,
            ..template(Vec::new())
        });
        let invalid = assets.insert(SceneTemplate {
            guid: invalid_guid,
            ..template(Vec::new())
        });
        let events = assets.drain_events();
        let mut invalid_snapshot = SceneTemplate {
            guid: invalid_guid,
            ..template(Vec::new())
        };
        invalid_snapshot.schema = 0;
        let snapshots = [
            (
                valid.id(),
                SceneTemplate {
                    guid: valid_guid,
                    ..template(Vec::new())
                },
            ),
            (invalid.id(), invalid_snapshot),
        ];

        assert!(matches!(
            cache.apply_asset_batch(&events, &snapshots),
            Err(SceneTemplateBatchError::InvalidSnapshot(id)) if id == invalid.id()
        ));
        assert!(cache.is_empty(), "valid peer must not partially publish");
    }

    #[test]
    fn loads_valid_template_and_checks_route_contract() {
        let dependency = StableGuid::from_u128(0x7171);
        let template = template(vec![dependency]);
        let (manifest, overlays, guid, _) =
            fixture(&template, None, SCENE_TEMPLATE_SCHEMA, vec![dependency]);
        let loaded = load_scene_template(
            &manifest,
            &overlays,
            guid,
            1024 * 1024,
            TemplateDecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(loaded, template);
    }

    #[test]
    fn rejects_schema_oversize_and_dependency_mismatch_before_acceptance() {
        let template = template(Vec::new());
        let (mut wrong_kind, overlays, guid, _) =
            fixture(&template, None, SCENE_TEMPLATE_SCHEMA, Vec::new());
        let mut metadata = wrong_kind.metadata(guid).unwrap().clone();
        metadata.kind ^= 1;
        wrong_kind.set_metadata(guid, metadata).unwrap();
        assert!(matches!(
            load_scene_template(
                &wrong_kind,
                &overlays,
                guid,
                u64::MAX,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::Kind { .. })
        ));

        let (manifest, overlays, guid, payload) =
            fixture(&template, None, SCENE_TEMPLATE_SCHEMA + 1, Vec::new());
        assert!(matches!(
            load_scene_template(
                &manifest,
                &overlays,
                guid,
                u64::MAX,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::Schema { .. })
        ));

        let (manifest, overlays, guid, _) =
            fixture(&template, None, SCENE_TEMPLATE_SCHEMA, Vec::new());
        assert!(matches!(
            load_scene_template(
                &manifest,
                &overlays,
                guid,
                payload.len().saturating_sub(1) as u64,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::PayloadTooLarge { .. })
        ));

        let (manifest, overlays, guid, _) = fixture(
            &template,
            None,
            SCENE_TEMPLATE_SCHEMA,
            vec![StableGuid::from_u128(0x8181)],
        );
        assert!(matches!(
            load_scene_template(
                &manifest,
                &overlays,
                guid,
                u64::MAX,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::DependencyMismatch)
        ));
    }

    #[test]
    fn rejects_corrupt_prst_and_guid_mismatch() {
        let template = template(Vec::new());
        let (manifest, overlays, guid, _) = fixture(
            &template,
            Some(b"NOPE".to_vec()),
            SCENE_TEMPLATE_SCHEMA,
            Vec::new(),
        );
        assert!(matches!(
            load_scene_template(
                &manifest,
                &overlays,
                guid,
                u64::MAX,
                TemplateDecodeLimits::default()
            ),
            Err(SceneTemplateLoadError::Template(_))
        ));

        let other = SceneTemplate {
            guid: StableGuid::from_u128(0x9999),
            ..template
        };
        let (_, _, _, payload) = fixture(&other, None, SCENE_TEMPLATE_SCHEMA, Vec::new());
        let route_guid = StableGuid::from_u128(0x5151);
        let mut writer = ContainerWriter::new();
        writer.add(route_guid, payload).unwrap();
        let reader = ContainerReader::new(writer.finish().unwrap()).unwrap();
        let mut mismatched_overlays = OverlaySet::new();
        mismatched_overlays.mount("scene", 10, reader);
        let mut mismatched = mismatched_overlays.runtime_manifest().unwrap();
        mismatched
            .set_metadata(
                route_guid,
                ManifestMetadata {
                    kind: SCENE_TEMPLATE_KIND,
                    schema: SCENE_TEMPLATE_SCHEMA,
                    ..ManifestMetadata::default()
                },
            )
            .unwrap();
        let reader = mismatched.bind(&mismatched_overlays).unwrap();
        assert!(matches!(
            decode_routed_template(
                &reader,
                &mismatched,
                route_guid,
                u64::MAX,
                TemplateDecodeLimits::default(),
                &SchemaMigrationRegistry::default()
            ),
            Err(SceneTemplateLoadError::GuidMismatch { .. })
        ));
    }

    #[test]
    fn accepts_application_registered_content_migration() {
        fn migrate_v1_to_v2(
            mut template: SceneTemplate,
        ) -> Result<SceneTemplate, SchemaMigrationError> {
            template.schema = 2;
            Ok(template)
        }

        let mut old = template(Vec::new());
        old.schema = 1;
        let (manifest, overlays, guid, _) = fixture(&old, None, 1, Vec::new());
        let mut migrations = SchemaMigrationRegistry::new();
        migrations
            .register(prism_scene::SchemaMigration {
                from: 1,
                to: 2,
                apply: migrate_v1_to_v2,
            })
            .unwrap();
        let loaded = load_scene_template_with_migrations(
            &manifest,
            &overlays,
            guid,
            u64::MAX,
            TemplateDecodeLimits::default(),
            &migrations,
        )
        .unwrap();
        assert_eq!(loaded.schema, SCENE_TEMPLATE_SCHEMA);
    }
}
