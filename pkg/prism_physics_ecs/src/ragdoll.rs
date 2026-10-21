//! Native ECS authoring and lifecycle bridge for XPBD ragdolls.
//!
//! A ragdoll component references existing rigid-body entities by stable part
//! id. The ordinary physics synchronizer owns those bodies; this module only
//! reconciles generational handles and installs the definition's constraints
//! into the fixed-step solver.

use std::collections::{BTreeMap, BTreeSet};

use prism_ecs::prelude::*;
use prism_physics_core::joint::ragdoll::{
    RagdollBinding, RagdollBindingError, RagdollDefinition, RagdollPartId,
};
use prism_physics_core::{BodyHandle, XpbdSolver};

use crate::{PhysicsBindings, PhysicsSolver, PhysicsWorldResource, RigidBody};

/// Authoring component that attaches a core ragdoll definition to existing ECS
/// rigid bodies. Part entities are paired with stable IDs from `definition`;
/// their `RigidBody` and `Transform` components remain authoritative for body
/// state and pose.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct EcsRagdoll {
    /// Solver-neutral links and part topology.
    pub definition: RagdollDefinition,
    /// Stable ragdoll part id to entity mapping.
    pub parts: Vec<(RagdollPartId, Entity)>,
}

impl EcsRagdoll {
    /// Creates deterministic authoring data by sorting the supplied mapping by
    /// stable part id. Invalid or incomplete mappings are diagnosed at the
    /// next fixed Sync phase.
    #[must_use]
    pub fn new(
        definition: RagdollDefinition,
        parts: impl IntoIterator<Item = (RagdollPartId, Entity)>,
    ) -> Self {
        let mut parts: Vec<_> = parts.into_iter().collect();
        parts.sort_by_key(|(id, entity)| (*id, *entity));
        Self { definition, parts }
    }
}

/// Reconciliation failure for one authored ragdoll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcsRagdollIssue {
    /// Mapping contains duplicate IDs/entities or does not match the definition.
    InvalidPartMapping,
    /// A mapped entity lacks a synchronized `RigidBody` or a live core body.
    MissingBody,
    /// A previously bound generational handle was invalidated before refresh.
    StaleBody,
    /// Core rejected the definition-to-handle mapping.
    BindingRejected(RagdollBindingError),
}

/// Diagnostic for one ragdoll owner during the latest fixed Sync phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EcsRagdollDiagnostic {
    /// Entity carrying [`EcsRagdoll`].
    pub owner: Entity,
    /// Reconciliation failure.
    pub issue: EcsRagdollIssue,
}

/// Latest fixed Sync ragdoll diagnostics. Replaced every tick; no history is
/// retained.
#[derive(Resource, Default)]
pub struct EcsRagdollDiagnostics {
    /// Failures observed during the most recent reconciliation.
    pub issues: Vec<EcsRagdollDiagnostic>,
}

#[derive(Clone, Debug)]
struct Entry {
    authored: EcsRagdoll,
    binding: RagdollBinding,
}

/// Current ECS ragdoll to generational body bindings.
#[derive(Resource, Default)]
pub struct EcsRagdollBindings {
    entries: BTreeMap<Entity, Entry>,
}

impl EcsRagdollBindings {
    /// Returns the core binding for an active ECS ragdoll owner.
    #[must_use]
    pub fn ragdoll(&self, owner: Entity) -> Option<&RagdollBinding> {
        self.entries.get(&owner).map(|entry| &entry.binding)
    }

    /// Resolves one authored part to its current generational physics handle.
    #[must_use]
    pub fn body(&self, owner: Entity, part: RagdollPartId) -> Option<BodyHandle> {
        self.ragdoll(owner)?.body(part)
    }

    /// Number of currently active ragdoll bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no authored ragdolls are currently active.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn diagnostic(diagnostics: &mut EcsRagdollDiagnostics, owner: Entity, issue: EcsRagdollIssue) {
    diagnostics
        .issues
        .push(EcsRagdollDiagnostic { owner, issue });
}

fn canonical_mapping(mut authored: EcsRagdoll) -> Option<EcsRagdoll> {
    if authored.parts.len() != authored.definition.parts.len() {
        return None;
    }
    let mut entities = BTreeSet::new();
    authored.parts.sort_by_key(|(id, _)| *id);
    if authored.parts.windows(2).any(|pair| pair[0].0 == pair[1].0)
        || authored
            .parts
            .iter()
            .any(|(_, entity)| !entities.insert(*entity))
        || !authored.parts.iter().map(|(id, _)| *id).eq(authored
            .definition
            .parts
            .iter()
            .map(|part| part.id))
    {
        return None;
    }
    Some(authored)
}

/// Reconcile authoring after ordinary rigid-body/assembly Sync and immediately
/// before the XPBD fixed-step system.
pub(crate) fn reconcile(world: &mut World) {
    world.init_resource::<EcsRagdollBindings>();
    world.init_resource::<EcsRagdollDiagnostics>();

    let query = world.query::<(Entity, &EcsRagdoll)>();
    let mut authored: Vec<_> = query
        .iter(world)
        .map(|(owner, component)| (owner, component.clone()))
        .collect();
    authored.sort_by_key(|(owner, _)| *owner);

    let physics = world.remove_resource::<PhysicsWorldResource>().unwrap();
    let body_bindings = world.remove_resource::<PhysicsBindings>().unwrap();
    let mut ragdolls = world.remove_resource::<EcsRagdollBindings>().unwrap();
    let mut diagnostics = world.remove_resource::<EcsRagdollDiagnostics>().unwrap();
    let mut live_owners = BTreeSet::new();
    diagnostics.issues.clear();

    for (owner, component) in authored {
        live_owners.insert(owner);
        let Some(component) = canonical_mapping(component) else {
            ragdolls.entries.remove(&owner);
            diagnostic(&mut diagnostics, owner, EcsRagdollIssue::InvalidPartMapping);
            continue;
        };

        let mut handles = Vec::with_capacity(component.parts.len());
        let mut mapping_issue = None;
        for (_, entity) in &component.parts {
            if world.get::<RigidBody>(*entity).is_none() {
                mapping_issue = Some(EcsRagdollIssue::MissingBody);
                break;
            }
            let Some(handle) = body_bindings.body(*entity) else {
                mapping_issue = Some(EcsRagdollIssue::MissingBody);
                break;
            };
            if !physics.bodies.contains(handle) {
                mapping_issue = Some(EcsRagdollIssue::StaleBody);
                break;
            }
            handles.push(handle);
        }
        if let Some(issue) = mapping_issue {
            ragdolls.entries.remove(&owner);
            diagnostic(&mut diagnostics, owner, issue);
            continue;
        }

        let unchanged = ragdolls.entries.get(&owner).is_some_and(|entry| {
            entry.authored == component
                && component
                    .parts
                    .iter()
                    .zip(&handles)
                    .all(|((id, _), handle)| entry.binding.body(*id) == Some(*handle))
        });
        if unchanged {
            continue;
        }

        match RagdollBinding::try_new(component.definition.clone(), handles) {
            Ok(binding) => {
                ragdolls.entries.insert(
                    owner,
                    Entry {
                        authored: component,
                        binding,
                    },
                );
            }
            Err(error) => {
                ragdolls.entries.remove(&owner);
                diagnostic(
                    &mut diagnostics,
                    owner,
                    EcsRagdollIssue::BindingRejected(error),
                );
            }
        }
    }

    ragdolls
        .entries
        .retain(|owner, _| live_owners.contains(owner));

    let constraints = ragdolls
        .entries
        .values()
        .flat_map(|entry| entry.binding.constraints().iter().copied())
        .collect();
    let mut solver = world.remove_resource::<PhysicsSolver>().unwrap();
    let config = solver.0.config;
    solver.0 = XpbdSolver::with_config(config).with_ragdolls(constraints);

    world.insert_resource(physics);
    world.insert_resource(body_bindings);
    world.insert_resource(ragdolls);
    world.insert_resource(diagnostics);
    world.insert_resource(solver);
}
