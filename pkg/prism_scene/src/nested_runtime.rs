//! Bounded recursive runtime instantiation for typed nested scenes.
//!
//! This module deliberately accepts typed constructors rather than decoding
//! opaque plan columns. It provides bounded nested-tree ownership, replacement,
//! and rollback; callers that start from baked opaque columns must first use the
//! scene materializer registry to produce typed constructors.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use crate::instance::SceneInstance;
use crate::instantiate::{
    despawn_instance, spawn_typed_plan, InstantiateError, SceneAttachError, SceneSpawn,
    TypedSpawnPlan,
};
use crate::template::SceneTemplate;
use prism_asset::StableGuid;
use prism_ecs::hierarchy::ChildOf;
use prism_ecs::world::World;

/// Supplies validated descriptors and typed constructors for nested templates.
pub trait NestedSceneProvider {
    /// Resolves a child descriptor by stable identity.
    fn template(&self, guid: StableGuid) -> Option<&SceneTemplate>;
    /// Creates a fresh typed constructor set for one child instance.
    fn typed_plan(&self, guid: StableGuid) -> Result<TypedSpawnPlan, NestedProviderError>;

    /// Replays provider-owned component overrides from a previous live node
    /// onto a staged replacement node. This callback is invoked only when the
    /// old and new nodes have the same template identity. It must not mutate
    /// the previous instance; on error the replacement tree is discarded.
    fn replay_instance_overrides(
        &self,
        world: &mut World,
        template: StableGuid,
        previous: &SceneInstance,
        replacement: &SceneSpawn,
    ) -> Result<(), NestedProviderError>;
}

/// Provider failure while preparing a child instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NestedProviderError {
    /// The provider could not prepare a constructor set.
    Unavailable,
    /// The provider rejected the requested constructor set with an opaque code.
    Invalid(u32),
}

/// Resource bounds for recursive nested spawning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NestedSpawnLimits {
    /// Maximum template edges below the root. A root with children has depth 1.
    pub max_depth: usize,
    /// Maximum number of scene instances, including the root.
    pub max_instances: usize,
}

impl Default for NestedSpawnLimits {
    fn default() -> Self {
        Self {
            max_depth: 64,
            max_instances: 4096,
        }
    }
}

/// A live scene tree with explicit ownership of every nested instance.
#[derive(Debug)]
pub struct SceneTreeSpawn {
    /// The root instance and its local remap.
    pub root: SceneSpawn,
    /// Child trees in template slot order.
    pub children: Vec<SceneTreeChild>,
}

/// Failed atomic tree replacement. The previous live tree remains owned and
/// is returned so the caller can retry or explicitly tear it down.
#[derive(Debug)]
pub struct NestedReplaceError {
    /// Reason the candidate generation was rejected.
    pub cause: NestedInstantiateError,
    /// Still-live prior generation.
    pub previous: Box<SceneTreeSpawn>,
}

/// A dependent-tree reload batch failed before its publication point. Every
/// previous tree is returned live and in caller order.
#[derive(Debug)]
pub struct NestedReloadBatchError {
    /// Reason the candidate generation was rejected.
    pub cause: NestedInstantiateError,
    /// All original live trees, still authoritative.
    pub trees: Vec<SceneTreeSpawn>,
}

/// One nested child and the parent local slot it occupies.
#[derive(Debug)]
pub struct SceneTreeChild {
    /// Parent local entity used as the attachment point.
    pub slot_root: crate::identity::LocalEntityId,
    /// Child template identity.
    pub template: StableGuid,
    /// Recursively owned child tree.
    pub tree: SceneTreeSpawn,
}

impl SceneTreeSpawn {
    /// Detaches and despawns the complete tree, children before parents.
    pub fn despawn_tree(self, world: &mut World) -> usize {
        let mut removed = 0;
        for child in self.children.into_iter().rev() {
            removed += child.tree.despawn_tree(world);
        }
        removed + despawn_instance(world, &self.root)
    }
}

/// Failure while building a nested scene tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NestedInstantiateError {
    /// A descriptor failed its semantic validation.
    InvalidTemplate(crate::template::TemplateError),
    /// The prior live root does not belong to the template being replaced.
    TemplateIdentityMismatch {
        /// Template recorded by the existing live tree.
        previous: Option<StableGuid>,
        /// Template selected for the replacement.
        replacement: StableGuid,
    },
    /// The provider has no descriptor for a slot.
    MissingTemplate(StableGuid),
    /// The provider could not create a typed constructor set.
    Provider(NestedProviderError),
    /// A template identity was encountered twice on the active recursion path.
    Cycle(StableGuid),
    /// The configured recursion bound was exceeded.
    DepthLimit {
        /// Template that could not be entered.
        guid: StableGuid,
        /// Requested depth, measured in child edges from the root.
        depth: usize,
    },
    /// The configured instance bound was exceeded.
    InstanceLimit,
    /// The typed plan failed to instantiate.
    Spawn(InstantiateError),
    /// The root could not receive its scene anchor.
    Attach(SceneAttachError),
    /// A child root or parent slot was not materialized.
    MissingAttachmentEntity,
    /// The ECS rejected the child relation.
    RelationRejected,
}

/// Spawns a validated root and all typed nested children transactionally.
pub fn spawn_nested_typed<P: NestedSceneProvider>(
    world: &mut World,
    template: &SceneTemplate,
    root_plan: TypedSpawnPlan,
    provider: &P,
    limits: NestedSpawnLimits,
) -> Result<SceneTreeSpawn, NestedInstantiateError> {
    template
        .validate()
        .map_err(NestedInstantiateError::InvalidTemplate)?;
    if limits.max_instances == 0 {
        return Err(NestedInstantiateError::InstanceLimit);
    }
    let mut active = Vec::new();
    active.push(template.guid);
    let mut count = 0;
    spawn_tree(
        world,
        template,
        root_plan,
        provider,
        limits,
        0,
        &mut active,
        &mut count,
        None,
    )
}

/// Rebuilds a live nested tree beside the old tree, then swaps ownership only
/// after every replacement node and typed override replay succeeds. The world
/// is exclusively borrowed throughout, so no observer can resolve a partial
/// generation through this API. On failure, the old tree remains live and the
/// completed staged replacement is destroyed.
pub fn replace_nested_typed<P: NestedSceneProvider>(
    world: &mut World,
    previous: SceneTreeSpawn,
    template: &SceneTemplate,
    root_plan: TypedSpawnPlan,
    provider: &P,
    limits: NestedSpawnLimits,
) -> Result<SceneTreeSpawn, NestedReplaceError> {
    let replacement =
        stage_nested_replacement(world, &previous, template, root_plan, provider, limits)?;
    previous.despawn_tree(world);
    Ok(replacement)
}

/// Reloads every live tree whose root template is in the invalidated GUID set.
/// All replacements are staged before any old tree is retired. `max_instances`
/// bounds the total candidate instances across the batch. A failure destroys
/// completed candidates and returns every old tree to the caller. The
/// invalidation set should contain the cache's complete transitive `dependent`
/// closure so a changed nested template reloads its live roots.
pub fn reload_invalidated_nested_trees<P: NestedSceneProvider>(
    world: &mut World,
    trees: Vec<SceneTreeSpawn>,
    invalidated: &[StableGuid],
    provider: &P,
    limits: NestedSpawnLimits,
) -> Result<Vec<SceneTreeSpawn>, NestedReloadBatchError> {
    if invalidated.is_empty() {
        return Ok(trees);
    }
    let invalidated: BTreeSet<_> = invalidated.iter().copied().collect();
    let mut previous = trees.into_iter();
    let mut staged: Vec<(SceneTreeSpawn, Option<SceneTreeSpawn>)> = Vec::new();
    let mut staged_instances = 0usize;
    while let Some(tree) = previous.next() {
        let root_guid = tree
            .root
            .root
            .and_then(|root| world.get::<SceneInstance>(root))
            .map(|instance| instance.template);
        let Some(guid) = root_guid.filter(|guid| invalidated.contains(guid)) else {
            staged.push((tree, None));
            continue;
        };
        let Some(template) = provider.template(guid) else {
            for (_, candidate) in staged.iter_mut().rev() {
                if let Some(candidate) = candidate.take() {
                    candidate.despawn_tree(world);
                }
            }
            let mut old = staged.into_iter().map(|(old, _)| old).collect::<Vec<_>>();
            old.push(tree);
            old.extend(previous);
            return Err(NestedReloadBatchError {
                cause: NestedInstantiateError::MissingTemplate(guid),
                trees: old,
            });
        };
        let plan = match provider.typed_plan(guid) {
            Ok(plan) => plan,
            Err(cause) => {
                for (_, candidate) in staged.iter_mut().rev() {
                    if let Some(candidate) = candidate.take() {
                        candidate.despawn_tree(world);
                    }
                }
                let mut old = staged.into_iter().map(|(old, _)| old).collect::<Vec<_>>();
                old.push(tree);
                old.extend(previous);
                return Err(NestedReloadBatchError {
                    cause: NestedInstantiateError::Provider(cause),
                    trees: old,
                });
            }
        };
        let remaining = limits.max_instances.saturating_sub(staged_instances);
        let tree_limits = NestedSpawnLimits {
            max_depth: limits.max_depth,
            max_instances: remaining,
        };
        match stage_nested_replacement(world, &tree, template, plan, provider, tree_limits) {
            Ok(candidate) => {
                staged_instances = staged_instances.saturating_add(tree_instance_count(&candidate));
                staged.push((tree, Some(candidate)));
            }
            Err(error) => {
                for (_, candidate) in staged.iter_mut().rev() {
                    if let Some(candidate) = candidate.take() {
                        candidate.despawn_tree(world);
                    }
                }
                let mut old = staged.into_iter().map(|(old, _)| old).collect::<Vec<_>>();
                old.push(*error.previous);
                old.extend(previous);
                return Err(NestedReloadBatchError {
                    cause: error.cause,
                    trees: old,
                });
            }
        }
    }

    let mut replacements = Vec::with_capacity(staged.len());
    for (old, candidate) in staged {
        if let Some(candidate) = candidate {
            old.despawn_tree(world);
            replacements.push(candidate);
        } else {
            replacements.push(old);
        }
    }
    Ok(replacements)
}

fn tree_instance_count(tree: &SceneTreeSpawn) -> usize {
    1usize.saturating_add(
        tree.children
            .iter()
            .map(|child| tree_instance_count(&child.tree))
            .fold(0usize, usize::saturating_add),
    )
}

fn stage_nested_replacement<P: NestedSceneProvider>(
    world: &mut World,
    previous: &SceneTreeSpawn,
    template: &SceneTemplate,
    root_plan: TypedSpawnPlan,
    provider: &P,
    limits: NestedSpawnLimits,
) -> Result<SceneTreeSpawn, NestedReplaceError> {
    let old_guid = previous
        .root
        .root
        .and_then(|root| world.get::<SceneInstance>(root))
        .map(|instance| instance.template);
    if old_guid != Some(template.guid) {
        return Err(NestedReplaceError {
            cause: NestedInstantiateError::TemplateIdentityMismatch {
                previous: old_guid,
                replacement: template.guid,
            },
            previous: Box::new(clone_tree_handle(previous)),
        });
    }
    if let Err(error) = template.validate() {
        return Err(NestedReplaceError {
            cause: NestedInstantiateError::InvalidTemplate(error),
            previous: Box::new(clone_tree_handle(previous)),
        });
    }
    if limits.max_instances == 0 {
        return Err(NestedReplaceError {
            cause: NestedInstantiateError::InstanceLimit,
            previous: Box::new(clone_tree_handle(previous)),
        });
    }
    let mut active = vec![template.guid];
    let mut count = 0;
    let replacement = match spawn_tree(
        world,
        template,
        root_plan,
        provider,
        limits,
        0,
        &mut active,
        &mut count,
        Some(previous),
    ) {
        Ok(replacement) => replacement,
        Err(cause) => {
            return Err(NestedReplaceError {
                cause,
                previous: Box::new(clone_tree_handle(previous)),
            })
        }
    };
    Ok(replacement)
}

fn clone_tree_handle(tree: &SceneTreeSpawn) -> SceneTreeSpawn {
    SceneTreeSpawn {
        root: tree.root.clone(),
        children: tree
            .children
            .iter()
            .map(|child| SceneTreeChild {
                slot_root: child.slot_root,
                template: child.template,
                tree: clone_tree_handle(&child.tree),
            })
            .collect(),
    }
}

fn spawn_tree<P: NestedSceneProvider>(
    world: &mut World,
    template: &SceneTemplate,
    typed: TypedSpawnPlan,
    provider: &P,
    limits: NestedSpawnLimits,
    depth: usize,
    active: &mut Vec<StableGuid>,
    count: &mut usize,
    previous: Option<&SceneTreeSpawn>,
) -> Result<SceneTreeSpawn, NestedInstantiateError> {
    if *count >= limits.max_instances {
        return Err(NestedInstantiateError::InstanceLimit);
    }
    let spawn =
        spawn_typed_plan(world, &template.plan, typed).map_err(NestedInstantiateError::Spawn)?;
    *count += 1;
    if let Err(error) = spawn.attach_instance(
        world,
        template.guid,
        crate::residency::ResidencyTier::Interactive,
    ) {
        despawn_instance(world, &spawn);
        return Err(NestedInstantiateError::Attach(error));
    }
    if let Some(previous) = previous.filter(|old| old.root.root.is_some())
        && let Some(old_root) = previous.root.root
        && let Some(old_instance) = world.get::<SceneInstance>(old_root).cloned()
        && old_instance.template == template.guid
    {
        if let Err(error) =
            provider.replay_instance_overrides(world, template.guid, &old_instance, &spawn)
        {
            despawn_instance(world, &spawn);
            return Err(NestedInstantiateError::Provider(error));
        }
        if let Some(mut new_instance) = world.get_mut::<SceneInstance>(spawn.root.unwrap()) {
            new_instance.overrides = old_instance.overrides.clone();
            new_instance.persistent = old_instance.persistent;
            new_instance.tier = old_instance.tier;
        } else {
            despawn_instance(world, &spawn);
            return Err(NestedInstantiateError::MissingAttachmentEntity);
        }
    }
    let mut children: Vec<SceneTreeChild> = Vec::new();
    let previous_children: BTreeMap<_, _> = previous
        .into_iter()
        .flat_map(|tree| tree.children.iter())
        .map(|child| ((child.slot_root, child.template), &child.tree))
        .collect();
    for slot in template.nested.iter().copied() {
        if depth >= limits.max_depth {
            for child in children.into_iter().rev() {
                child.tree.despawn_tree(world);
            }
            despawn_instance(world, &spawn);
            return Err(NestedInstantiateError::DepthLimit {
                guid: slot.template,
                depth: depth + 1,
            });
        }
        if active.contains(&slot.template) {
            for child in children.into_iter().rev() {
                child.tree.despawn_tree(world);
            }
            despawn_instance(world, &spawn);
            return Err(NestedInstantiateError::Cycle(slot.template));
        }
        let Some(child_template) = provider.template(slot.template) else {
            for child in children.into_iter().rev() {
                child.tree.despawn_tree(world);
            }
            despawn_instance(world, &spawn);
            return Err(NestedInstantiateError::MissingTemplate(slot.template));
        };
        if let Err(error) = child_template.validate() {
            for child in children.drain(..).rev() {
                child.tree.despawn_tree(world);
            }
            despawn_instance(world, &spawn);
            return Err(NestedInstantiateError::InvalidTemplate(error));
        }
        let typed_child = match provider.typed_plan(slot.template) {
            Ok(value) => value,
            Err(error) => {
                for child in children.drain(..).rev() {
                    child.tree.despawn_tree(world);
                }
                despawn_instance(world, &spawn);
                return Err(NestedInstantiateError::Provider(error));
            }
        };
        active.push(slot.template);
        let previous_child = previous_children.get(&(slot.root, slot.template)).copied();
        let child = spawn_tree(
            world,
            child_template,
            typed_child,
            provider,
            limits,
            depth + 1,
            active,
            count,
            previous_child,
        );
        active.pop();
        let child = match child {
            Ok(value) => value,
            Err(error) => {
                for prior in children.into_iter().rev() {
                    prior.tree.despawn_tree(world);
                }
                despawn_instance(world, &spawn);
                return Err(error);
            }
        };
        let Some(parent) = spawn.resolve(slot.root) else {
            child.despawn_tree(world);
            for prior in children.into_iter().rev() {
                prior.tree.despawn_tree(world);
            }
            despawn_instance(world, &spawn);
            return Err(NestedInstantiateError::MissingAttachmentEntity);
        };
        let Some(child_root) = child.root.root else {
            child.despawn_tree(world);
            for prior in children.into_iter().rev() {
                prior.tree.despawn_tree(world);
            }
            despawn_instance(world, &spawn);
            return Err(NestedInstantiateError::MissingAttachmentEntity);
        };
        world.register_child_of();
        if crate::instantiate::would_create_child_cycle(world, child_root, parent) {
            child.despawn_tree(world);
            for prior in children.into_iter().rev() {
                prior.tree.despawn_tree(world);
            }
            despawn_instance(world, &spawn);
            return Err(NestedInstantiateError::RelationRejected);
        }
        world.add_relation::<ChildOf>(child_root, parent);
        children.push(SceneTreeChild {
            slot_root: slot.root,
            template: slot.template,
            tree: child,
        });
    }
    Ok(SceneTreeSpawn {
        root: spawn,
        children,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{LocalEntityId, PersistentId};
    use crate::overrides::{ArchetypeSignature, ComponentTypeId, OverrideValue, PropertyOverride};
    use crate::path::FieldPath;
    use crate::plan::{ArchetypeGroup, InstantiationPlan};
    use crate::residency::ResidencyTier;
    use crate::template::NestedSlot;
    use alloc::boxed::Box;
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use prism_ecs::component::Component;

    #[derive(Debug, PartialEq)]
    struct Marker(u32);
    impl Component for Marker {}

    fn guid(value: u128) -> StableGuid {
        StableGuid::from_u128(value)
    }

    fn plan(count: u32, hierarchy: &[(u32, u32)]) -> InstantiationPlan {
        InstantiationPlan {
            entity_count: count,
            groups: if count == 0 {
                Box::new([])
            } else {
                vec![ArchetypeGroup::new(
                    ArchetypeSignature::default(),
                    (0..count)
                        .map(LocalEntityId::new)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                    Box::new([]),
                )]
                .into_boxed_slice()
            },
            entity_valued_fields: Box::new([]),
            hierarchy: hierarchy
                .iter()
                .map(|&(child, parent)| (LocalEntityId::new(child), LocalEntityId::new(parent)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        }
    }

    fn descriptor(id: u128, children: &[(u32, u128)]) -> SceneTemplate {
        let count = children
            .iter()
            .map(|(slot, _)| slot + 1)
            .max()
            .unwrap_or(1)
            .max(2);
        let hierarchy = children
            .iter()
            .map(|(slot, _)| (*slot, 0))
            .chain(
                (1..count)
                    .filter(|local| !children.iter().any(|(slot, _)| slot == local))
                    .map(|local| (local, 0)),
            )
            .collect::<Vec<_>>();
        let mut template = SceneTemplate::new(guid(id), plan(count, &hierarchy));
        template.nested = children
            .iter()
            .map(|(slot, child)| NestedSlot::new(LocalEntityId::new(*slot), guid(*child)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut deps = children
            .iter()
            .map(|(_, child)| guid(*child))
            .collect::<Vec<_>>();
        deps.sort_unstable();
        deps.dedup();
        template.deps = deps.into_boxed_slice();
        template.validate().unwrap();
        template
    }

    #[derive(Default)]
    struct Provider {
        templates: BTreeMap<StableGuid, SceneTemplate>,
        fail_replay_for: Option<StableGuid>,
    }

    impl NestedSceneProvider for Provider {
        fn template(&self, guid: StableGuid) -> Option<&SceneTemplate> {
            self.templates.get(&guid)
        }
        fn typed_plan(&self, guid: StableGuid) -> Result<TypedSpawnPlan, NestedProviderError> {
            let mut plan = TypedSpawnPlan::new().with_bundle(LocalEntityId::new(0), Marker(1));
            if self
                .templates
                .get(&guid)
                .is_some_and(|template| template.entity_count() > 1)
            {
                plan = plan.with_bundle(LocalEntityId::new(1), Marker(1));
            }
            Ok(plan)
        }
        fn replay_instance_overrides(
            &self,
            world: &mut World,
            template: StableGuid,
            _previous: &SceneInstance,
            replacement: &SceneSpawn,
        ) -> Result<(), NestedProviderError> {
            if self.fail_replay_for == Some(template) {
                return Err(NestedProviderError::Unavailable);
            }
            let root = replacement.root.ok_or(NestedProviderError::Unavailable)?;
            let mut marker = world
                .get_mut::<Marker>(root)
                .ok_or(NestedProviderError::Unavailable)?;
            marker.0 = 2;
            Ok(())
        }
    }

    fn provider() -> Provider {
        Provider {
            templates: BTreeMap::from([
                (guid(1), descriptor(1, &[(1, 2)])),
                (guid(2), descriptor(2, &[])),
                (guid(3), descriptor(3, &[])),
            ]),
            ..Provider::default()
        }
    }

    fn typed_root_plan() -> TypedSpawnPlan {
        TypedSpawnPlan::new()
            .with_bundle(LocalEntityId::new(0), Marker(1))
            .with_bundle(LocalEntityId::new(1), Marker(1))
    }

    #[test]
    fn replacement_stages_complete_tree_preserves_metadata_and_retires_old_tree() {
        let provider = provider();
        let template = provider.template(guid(1)).unwrap();
        let mut world = World::new();
        let previous = spawn_nested_typed(
            &mut world,
            template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let old_root = previous.root.root.unwrap();
        let old_child = previous.children[0].tree.root.root.unwrap();
        {
            let mut instance = world.get_mut::<SceneInstance>(old_root).unwrap();
            instance.persistent = Some(PersistentId::from_u128(0xabc));
            instance.tier = ResidencyTier::Proxy;
            instance.overrides_mut().set_property(PropertyOverride::new(
                LocalEntityId::new(0),
                ComponentTypeId::from_raw(0xdef),
                FieldPath::parse("value").unwrap(),
                OverrideValue::I64(7),
            ));
        }
        let replacement = replace_nested_typed(
            &mut world,
            previous,
            template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let new_root = replacement.root.root.unwrap();
        let new_child = replacement.children[0].tree.root.root.unwrap();
        assert_ne!(old_root, new_root);
        assert_ne!(old_child, new_child);
        assert!(!world.contains(old_root));
        assert!(!world.contains(old_child));
        assert_eq!(world.get::<Marker>(new_root).unwrap().0, 2);
        assert_eq!(world.get::<Marker>(new_child).unwrap().0, 2);
        let instance = world.get::<SceneInstance>(new_root).unwrap();
        assert_eq!(instance.persistent, Some(PersistentId::from_u128(0xabc)));
        assert_eq!(instance.tier, ResidencyTier::Proxy);
        assert_eq!(instance.overrides().unwrap().properties().len(), 1);
        assert_eq!(world.entity_count(), 4);
    }

    #[test]
    fn replacement_replay_failure_keeps_old_tree_and_returns_ownership() {
        let provider = provider();
        let template = provider.template(guid(1)).unwrap().clone();
        let mut world = World::new();
        let previous = spawn_nested_typed(
            &mut world,
            &template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let old_root = previous.root.root.unwrap();
        let old_child = previous.children[0].tree.root.root.unwrap();
        let failing = Provider {
            templates: provider.templates.clone(),
            fail_replay_for: Some(guid(2)),
        };
        let error = replace_nested_typed(
            &mut world,
            previous,
            &template,
            typed_root_plan(),
            &failing,
            NestedSpawnLimits::default(),
        )
        .unwrap_err();
        assert_eq!(
            error.cause,
            NestedInstantiateError::Provider(NestedProviderError::Unavailable)
        );
        assert!(world.contains(old_root));
        assert!(world.contains(old_child));
        assert_eq!(world.entity_count(), 4);
        let retry = replace_nested_typed(
            &mut world,
            *error.previous,
            &template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        assert_eq!(world.entity_count(), 4);
        assert!(world.contains(retry.root.root.unwrap()));
    }

    #[test]
    fn invalidation_batch_reloads_matching_roots_and_leaves_unrelated_roots_untouched() {
        let provider = provider();
        let root_template = provider.template(guid(1)).unwrap();
        let other_template = provider.template(guid(3)).unwrap();
        let mut world = World::new();
        let root = spawn_nested_typed(
            &mut world,
            root_template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let other = spawn_nested_typed(
            &mut world,
            other_template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let old_root = root.root.root.unwrap();
        let old_other = other.root.root.unwrap();

        let replacements = reload_invalidated_nested_trees(
            &mut world,
            vec![root, other],
            &[guid(1)],
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();

        assert_eq!(replacements.len(), 2);
        assert_ne!(replacements[0].root.root.unwrap(), old_root);
        assert_eq!(replacements[1].root.root.unwrap(), old_other);
        assert!(!world.contains(old_root));
        assert!(world.contains(old_other));
        assert_eq!(world.entity_count(), 6);
    }

    #[test]
    fn empty_invalidation_preserves_tree_ownership_and_world() {
        let provider = provider();
        let root_template = provider.template(guid(1)).unwrap();
        let mut world = World::new();
        let tree = spawn_nested_typed(
            &mut world,
            root_template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let root = tree.root.root.unwrap();

        let trees = reload_invalidated_nested_trees(
            &mut world,
            vec![tree],
            &[],
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();

        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].root.root, Some(root));
        assert!(world.contains(root));
        assert_eq!(world.entity_count(), 4);
    }

    #[test]
    fn invalidation_batch_failure_returns_all_old_trees_and_discards_candidates() {
        let provider = provider();
        let root_template = provider.template(guid(1)).unwrap().clone();
        let other_template = provider.template(guid(3)).unwrap().clone();
        let mut world = World::new();
        let root = spawn_nested_typed(
            &mut world,
            &root_template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let other = spawn_nested_typed(
            &mut world,
            &other_template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let old_roots = [root.root.root.unwrap(), other.root.root.unwrap()];
        let failing = Provider {
            templates: provider.templates.clone(),
            fail_replay_for: Some(guid(2)),
        };

        let error = reload_invalidated_nested_trees(
            &mut world,
            vec![root, other],
            &[guid(1), guid(3)],
            &failing,
            NestedSpawnLimits::default(),
        )
        .unwrap_err();

        assert_eq!(error.trees.len(), 2);
        assert_eq!(
            [
                error.trees[0].root.root.unwrap(),
                error.trees[1].root.root.unwrap(),
            ],
            old_roots
        );
        assert!(old_roots.iter().all(|root| world.contains(*root)));
        assert_eq!(world.entity_count(), 6);
    }

    #[test]
    fn invalidation_batch_enforces_one_instance_budget_across_all_candidates() {
        let provider = provider();
        let root_template = provider.template(guid(1)).unwrap().clone();
        let other_template = provider.template(guid(3)).unwrap().clone();
        let mut world = World::new();
        let root = spawn_nested_typed(
            &mut world,
            &root_template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let other = spawn_nested_typed(
            &mut world,
            &other_template,
            typed_root_plan(),
            &provider,
            NestedSpawnLimits::default(),
        )
        .unwrap();
        let old_roots = [root.root.root.unwrap(), other.root.root.unwrap()];

        let error = reload_invalidated_nested_trees(
            &mut world,
            vec![root, other],
            &[guid(1), guid(3)],
            &provider,
            NestedSpawnLimits {
                max_depth: 64,
                max_instances: 2,
            },
        )
        .unwrap_err();

        assert_eq!(error.cause, NestedInstantiateError::InstanceLimit);
        assert_eq!(error.trees.len(), 2);
        assert_eq!(
            [
                error.trees[0].root.root.unwrap(),
                error.trees[1].root.root.unwrap(),
            ],
            old_roots
        );
        assert!(old_roots.iter().all(|root| world.contains(*root)));
        assert_eq!(world.entity_count(), 6);
    }
}
