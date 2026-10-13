//! Bounded recursive runtime instantiation for typed nested scenes.
//!
//! This module deliberately accepts typed constructors rather than decoding opaque
//! plan columns. It provides ownership and rollback for a nested scene tree while
//! leaving bake/materializer integration to a later pipeline.

use alloc::vec::Vec;

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
    let result = spawn_tree(
        world,
        template,
        root_plan,
        provider,
        limits,
        0,
        &mut active,
        &mut count,
    );
    result
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
    let mut children: Vec<SceneTreeChild> = Vec::new();
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
        let child_template = match provider.template(slot.template) {
            Some(value) => value,
            None => {
                for child in children.into_iter().rev() {
                    child.tree.despawn_tree(world);
                }
                despawn_instance(world, &spawn);
                return Err(NestedInstantiateError::MissingTemplate(slot.template));
            }
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
        let child = spawn_tree(
            world,
            child_template,
            typed_child,
            provider,
            limits,
            depth + 1,
            active,
            count,
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
