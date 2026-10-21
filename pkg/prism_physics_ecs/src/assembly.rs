//! ECS lifecycle and generational remapping for core-owned rigid assemblies.

use std::collections::{BTreeMap, BTreeSet};

use prism_ecs::prelude::*;
use prism_physics_core::world::AssemblyWorldError;
use prism_physics_core::BodyHandle;

use crate::{PhysicsBindings, PhysicsWorldResource, RigidBody};

/// Entity-side reference to a body owned by [`PhysicsWorldResource`].
///
/// Bodies described by [`RigidBody`] remain owned by the normal rigid-body
/// synchronizer and are deliberately not eligible for assembly operations.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssemblyBody {
    /// Current generational core handle. During a merge, every member points
    /// to the shared proxy; after split each member receives its own handle.
    pub handle: BodyHandle,
}

/// An explicit request to merge entity-bound core bodies or split a merged
/// proxy. Requests are processed in insertion order at the fixed Sync phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssemblyRequestResult {
    /// Merge request in caller-provided entity order; processing canonicalizes it.
    Merge(Vec<Entity>),
    /// Split the assembly containing this entity.
    Split(Entity),
}

/// Queue of assembly lifecycle requests.
#[derive(Resource, Default)]
pub struct AssemblyRequestQueue {
    requests: Vec<AssemblyRequestResult>,
}

impl AssemblyRequestQueue {
    /// Queue a merge. At processing time all entities must be distinct,
    /// carry [`AssemblyBody`], and not be owned by the ordinary `RigidBody`
    /// synchronizer.
    pub fn merge(&mut self, entities: impl IntoIterator<Item = Entity>) {
        self.requests
            .push(AssemblyRequestResult::Merge(entities.into_iter().collect()));
    }

    /// Queue a split for the assembly currently containing `entity`.
    pub fn split(&mut self, entity: Entity) {
        self.requests.push(AssemblyRequestResult::Split(entity));
    }

    /// Number of requests awaiting the next fixed Sync phase.
    #[must_use]
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Whether the queue has no pending requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

/// Assembly lifecycle failures reported during the latest fixed Sync phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssemblyIssue {
    /// Merge request has fewer than one entity, duplicate entities, or stale/missing bindings.
    InvalidMembers,
    /// A member is owned by the standard ECS `RigidBody` synchronizer.
    RigidBodyOwned,
    /// A body is already part of another ECS-managed assembly.
    AlreadyManaged,
    /// Core rejected the merge (for example, the set is not a connected dynamic component).
    MergeRejected(AssemblyWorldError),
    /// Entity is not currently mapped to a managed merged proxy.
    NotMerged,
    /// Core rejected a requested split.
    SplitRejected(AssemblyWorldError),
    /// A member lost its entity binding and the core rejected automatic release of the proxy.
    MemberLostSplitRejected(AssemblyWorldError),
}

/// One assembly diagnostic, optionally associated with an entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssemblyDiagnostic {
    /// Entity causing the failed request, when identifiable.
    pub entity: Option<Entity>,
    /// Failure classification.
    pub issue: AssemblyIssue,
}

/// Latest fixed Sync assembly diagnostics. The vector is replaced each tick.
#[derive(Resource, Default)]
pub struct AssemblyDiagnostics {
    /// Failures from the latest fixed Sync phase.
    pub issues: Vec<AssemblyDiagnostic>,
}

#[derive(Clone, Debug)]
struct ManagedAssembly {
    proxy: BodyHandle,
    members: Vec<(Entity, BodyHandle)>,
    canonical_entity: Entity,
}

#[derive(Resource, Default)]
struct ManagedAssemblies(BTreeMap<(u32, u32), ManagedAssembly>);

fn body_key(body: BodyHandle) -> (u32, u32) {
    (body.index(), body.generation())
}

fn issue(diagnostics: &mut AssemblyDiagnostics, entity: Option<Entity>, issue: AssemblyIssue) {
    diagnostics
        .issues
        .push(AssemblyDiagnostic { entity, issue });
}

/// Fixed Sync system, ordered after ordinary rigid-body reconciliation.
pub(crate) fn process(world: &mut World) {
    world.init_resource::<AssemblyRequestQueue>();
    world.init_resource::<AssemblyDiagnostics>();
    world.init_resource::<ManagedAssemblies>();
    let requests = std::mem::take(&mut world.resource_mut::<AssemblyRequestQueue>().requests);
    let mut physics = world.remove_resource::<PhysicsWorldResource>().unwrap();
    let mut bindings = world.remove_resource::<PhysicsBindings>().unwrap();
    let mut assemblies = world.remove_resource::<ManagedAssemblies>().unwrap();
    let mut diagnostics = world.remove_resource::<AssemblyDiagnostics>().unwrap();
    diagnostics.issues.clear();

    // A lost member releases the whole proxy back to its original core bodies.
    // This keeps a despawn/removal from stranding every surviving member on a
    // proxy whose ECS identity no longer exists.
    let lost: Vec<_> = assemblies
        .0
        .values()
        .filter(|assembly| {
            assembly.members.iter().any(|(entity, _)| {
                world
                    .get::<AssemblyBody>(*entity)
                    .is_none_or(|binding| binding.handle != assembly.proxy)
            })
        })
        .map(|assembly| body_key(assembly.proxy))
        .collect();
    for key in lost {
        if let Some(assembly) = assemblies.0.get(&key).cloned() {
            if let Err(error) = split_managed(
                world,
                &mut physics,
                &mut bindings,
                &mut assemblies,
                assembly.proxy,
            ) {
                issue(
                    &mut diagnostics,
                    None,
                    AssemblyIssue::MemberLostSplitRejected(error),
                );
                forget_managed(&mut bindings, &mut assemblies, &assembly);
            }
        }
    }

    for request in requests {
        match request {
            AssemblyRequestResult::Merge(mut entities) => {
                entities.sort();
                if entities.is_empty() || entities.windows(2).any(|pair| pair[0] == pair[1]) {
                    issue(&mut diagnostics, None, AssemblyIssue::InvalidMembers);
                    continue;
                }
                let requested_count = entities.len();
                let mut members = Vec::with_capacity(entities.len());
                let mut invalid = false;
                for entity in entities {
                    if world.get::<RigidBody>(entity).is_some()
                        || bindings.bodies.contains_key(&entity)
                    {
                        issue(
                            &mut diagnostics,
                            Some(entity),
                            AssemblyIssue::RigidBodyOwned,
                        );
                        invalid = true;
                        continue;
                    }
                    let Some(member) = world.get::<AssemblyBody>(entity).copied() else {
                        issue(
                            &mut diagnostics,
                            Some(entity),
                            AssemblyIssue::InvalidMembers,
                        );
                        invalid = true;
                        continue;
                    };
                    if assemblies.0.contains_key(&body_key(member.handle))
                        || assemblies
                            .0
                            .values()
                            .any(|assembly| assembly.members.iter().any(|(e, _)| *e == entity))
                    {
                        issue(
                            &mut diagnostics,
                            Some(entity),
                            AssemblyIssue::AlreadyManaged,
                        );
                        invalid = true;
                        continue;
                    }
                    members.push((entity, member.handle));
                }
                if invalid || members.len() != requested_count {
                    continue;
                }
                members
                    .sort_by_key(|(entity, handle)| (handle.index(), handle.generation(), *entity));
                if members.windows(2).any(|pair| pair[0].1 == pair[1].1) {
                    issue(&mut diagnostics, None, AssemblyIssue::InvalidMembers);
                    continue;
                }
                let handles: Vec<_> = members.iter().map(|(_, handle)| *handle).collect();
                match physics.merge_assembly(&handles) {
                    Ok(proxy) => {
                        let canonical_entity = members
                            .iter()
                            .map(|(entity, _)| *entity)
                            .min()
                            .expect("validated nonempty assembly");
                        for (entity, _) in &members {
                            if let Some(mut member) = world.get_mut::<AssemblyBody>(*entity) {
                                member.handle = proxy;
                            }
                            bindings.assembly_bodies.insert(*entity, proxy);
                        }
                        bindings
                            .assembly_entities
                            .insert(body_key(proxy), canonical_entity);
                        assemblies.0.insert(
                            body_key(proxy),
                            ManagedAssembly {
                                proxy,
                                members,
                                canonical_entity,
                            },
                        );
                    }
                    Err(error) => {
                        issue(&mut diagnostics, None, AssemblyIssue::MergeRejected(error))
                    }
                }
            }
            AssemblyRequestResult::Split(entity) => {
                let Some(proxy) = bindings.assembly_bodies.get(&entity).copied() else {
                    issue(&mut diagnostics, Some(entity), AssemblyIssue::NotMerged);
                    continue;
                };
                if let Err(error) =
                    split_managed(world, &mut physics, &mut bindings, &mut assemblies, proxy)
                {
                    issue(
                        &mut diagnostics,
                        Some(entity),
                        AssemblyIssue::SplitRejected(error),
                    );
                }
            }
        }
    }

    sync_unmerged_aliases(
        world,
        &physics,
        &mut bindings,
        &assemblies,
        &mut diagnostics,
    );
    world.insert_resource(physics);
    world.insert_resource(bindings);
    world.insert_resource(assemblies);
    world.insert_resource(diagnostics);
}

fn sync_unmerged_aliases(
    world: &mut World,
    physics: &PhysicsWorldResource,
    bindings: &mut PhysicsBindings,
    assemblies: &ManagedAssemblies,
    diagnostics: &mut AssemblyDiagnostics,
) {
    let query = world.query::<(Entity, &AssemblyBody)>();
    let mut authored: Vec<_> = query
        .iter(world)
        .map(|(entity, body)| (entity, body.handle))
        .collect();
    authored.sort_by_key(|(entity, _)| *entity);
    let merged_entities: BTreeSet<_> = assemblies
        .0
        .values()
        .flat_map(|assembly| assembly.members.iter().map(|(entity, _)| *entity))
        .collect();
    let proxy_keys: BTreeSet<_> = assemblies.0.keys().copied().collect();
    bindings
        .assembly_bodies
        .retain(|entity, _| merged_entities.contains(entity));
    bindings
        .assembly_entities
        .retain(|key, _| proxy_keys.contains(key));

    let mut seen_handles = BTreeMap::new();
    for (entity, handle) in authored {
        if merged_entities.contains(&entity) {
            continue;
        }
        if world.get::<RigidBody>(entity).is_some() || bindings.bodies.contains_key(&entity) {
            issue(diagnostics, Some(entity), AssemblyIssue::RigidBodyOwned);
            continue;
        }
        if !physics.bodies.contains(handle) {
            issue(diagnostics, Some(entity), AssemblyIssue::InvalidMembers);
            continue;
        }
        if seen_handles.insert(body_key(handle), entity).is_some()
            || bindings
                .entities
                .get(&body_key(handle))
                .is_some_and(|mapped| *mapped != entity)
        {
            issue(diagnostics, Some(entity), AssemblyIssue::AlreadyManaged);
            continue;
        }
        bindings.assembly_bodies.insert(entity, handle);
        bindings.assembly_entities.insert(body_key(handle), entity);
    }
}

fn split_managed(
    world: &mut World,
    physics: &mut PhysicsWorldResource,
    bindings: &mut PhysicsBindings,
    assemblies: &mut ManagedAssemblies,
    proxy: BodyHandle,
) -> Result<(), AssemblyWorldError> {
    let Some(assembly) = assemblies.0.get(&body_key(proxy)).cloned() else {
        return Err(AssemblyWorldError::NotMerged);
    };
    let handles = physics.split_assembly(proxy)?;
    forget_managed(bindings, assemblies, &assembly);
    for ((entity, original), handle) in assembly.members.iter().zip(handles) {
        if world.get::<AssemblyBody>(*entity).is_some() {
            if let Some(mut member) = world.get_mut::<AssemblyBody>(*entity) {
                member.handle = handle;
            }
            bindings.assembly_bodies.insert(*entity, handle);
            bindings.assembly_entities.insert(body_key(handle), *entity);
        } else {
            // Entity is gone or explicitly detached; the core body remains
            // world-owned and intentionally has no ECS alias.
            let _ = original;
        }
    }
    Ok(())
}

fn forget_managed(
    bindings: &mut PhysicsBindings,
    assemblies: &mut ManagedAssemblies,
    assembly: &ManagedAssembly,
) {
    assemblies.0.remove(&body_key(assembly.proxy));
    bindings.assembly_entities.remove(&body_key(assembly.proxy));
    for (entity, _) in &assembly.members {
        bindings.assembly_bodies.remove(entity);
    }
    let _ = assembly.canonical_entity;
}
