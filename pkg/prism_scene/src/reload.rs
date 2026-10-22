//! Deterministic scene-template reload planning.
//!
//! [`SceneReloadPlan`] is intentionally a data plan, not a claim that this
//! crate can mutate a live [`World`].  A world/residency layer can consume the
//! plan to update ECS entities, while save/editor code can inspect diagnostics
//! before applying it.  Persistent ids are tried first; the captured local id
//! is the explicit compatibility fallback.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use prism_asset::StableGuid;

use crate::identity::LocalEntityId;
use crate::overrides::{ComponentTypeId, OverrideError, OverrideSet, StructuralOp};
use crate::persistence::{
    PersistedPropertyOverride, PersistenceError, PersistentEntityMap, SceneInstancePersistence,
};
use crate::template::{SceneTemplate, TemplateError};

/// A deterministic old-to-new template diff at entity granularity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TemplateDelta {
    /// Base entities present only in the new template.
    pub added_entities: Vec<LocalEntityId>,
    /// Base entities absent from the new template.
    pub removed_entities: Vec<LocalEntityId>,
    /// Entities whose archetype or column payload changed.
    pub changed_entities: Vec<LocalEntityId>,
    /// Entities whose parent relationship changed.
    pub hierarchy_changes: Vec<HierarchyChange>,
}

/// One changed parent relationship in a template delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HierarchyChange {
    /// The affected entity.
    pub local: LocalEntityId,
    /// Its old parent, if any.
    pub old_parent: Option<LocalEntityId>,
    /// Its new parent, if any.
    pub new_parent: Option<LocalEntityId>,
}

/// Why an override could not be carried into the new template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReloadOrphanReason {
    /// The target local id is not present after the reload.
    TargetMissing,
    /// The property component is no longer present on the target.
    ComponentMissing,
    /// An added entity's parent is no longer present.
    ParentMissing,
}

/// A reload diagnostic.  Orphans are retained as records and never silently
/// discarded; invalid structural operations are also reported before a
/// best-effort plan is returned.
#[derive(Clone, Debug, PartialEq)]
pub enum ReloadDiagnostic {
    /// A persisted property could not be resolved or is no longer applicable.
    OrphanedProperty {
        /// The original persisted record.
        property: PersistedPropertyOverride,
        /// Why it was not restored.
        reason: ReloadOrphanReason,
    },
    /// A persisted structural operation could not be carried forward.
    OrphanedStructural {
        /// The original operation.
        operation: StructuralOp,
        /// Why it was not restored.
        reason: ReloadOrphanReason,
    },
    /// A structural operation was malformed for the new plan or conflicted
    /// with another retained operation.
    InvalidStructural {
        /// The rejected operation.
        operation: StructuralOp,
        /// The validator's exact reason.
        error: OverrideError,
    },
    /// The restored set failed the final plan validation after all
    /// best-effort filtering. This is retained as a diagnostic rather than
    /// exposing an invalid replay set.
    InvalidRestoredOverrides {
        /// The validator's exact reason.
        error: OverrideError,
    },
}

/// The error cases that prevent construction of a reload plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReloadError {
    /// The old template was not a valid source descriptor.
    InvalidOldTemplate(TemplateError),
    /// The new template was not a valid target descriptor.
    InvalidNewTemplate(TemplateError),
    /// The snapshot came from a different template than the supplied old one.
    TemplateIdentityMismatch {
        /// The expected old template id.
        expected: StableGuid,
        /// The id recorded by the snapshot.
        found: StableGuid,
    },
    /// A caller supplied an invalid identity map.
    InvalidIdentityMap(PersistenceError),
}

/// A pure-data plan for synchronizing an instance from one template to another.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneReloadPlan {
    /// The source template id.
    pub old_template: StableGuid,
    /// The target template id.
    pub new_template: StableGuid,
    /// Entity and hierarchy changes consumers may apply incrementally.
    pub delta: TemplateDelta,
    /// The valid, locally addressed overrides to replay on the new template.
    pub overrides: OverrideSet,
    /// All non-fatal orphan and structural diagnostics.
    pub diagnostics: Vec<ReloadDiagnostic>,
}

impl SceneReloadPlan {
    /// Builds a deterministic reload plan from a persisted instance snapshot.
    pub fn build(
        old: &SceneTemplate,
        new: &SceneTemplate,
        snapshot: &SceneInstancePersistence,
        identities: &PersistentEntityMap,
    ) -> Result<Self, ReloadError> {
        old.validate().map_err(ReloadError::InvalidOldTemplate)?;
        new.validate().map_err(ReloadError::InvalidNewTemplate)?;
        if snapshot.template != old.guid {
            return Err(ReloadError::TemplateIdentityMismatch {
                expected: old.guid,
                found: snapshot.template,
            });
        }

        let delta = diff_templates(old, new);
        let mut diagnostics = Vec::new();
        let mut structural = OverrideSet::new();

        // Validate one operation at a time.  Skipping only the failing op
        // keeps later independent operations usable and makes the result
        // deterministic without pretending to apply anything to a World.
        for operation in &snapshot.structural {
            let mut candidate = structural.clone();
            candidate.push_structural(operation.clone());
            match candidate.validate_for_plan(&new.plan) {
                Ok(()) => structural = candidate,
                Err(error) => {
                    if let Some(reason) =
                        structural_orphan_reason(new, &structural, operation, error)
                    {
                        diagnostics.push(ReloadDiagnostic::OrphanedStructural {
                            operation: operation.clone(),
                            reason,
                        });
                    } else {
                        diagnostics.push(ReloadDiagnostic::InvalidStructural {
                            operation: operation.clone(),
                            error,
                        });
                    }
                }
            }
        }

        let component_state = component_state(new, &structural);
        let mut restored = structural;
        for property in &snapshot.properties {
            let target = identities
                .local_for_persistent(property.persistent.unwrap_or(crate::PersistentId::NIL))
                .filter(|local| template_entity_is_base_or_added(*local, new, &restored))
                .or_else(|| {
                    if template_entity_is_base_or_added(property.local, new, &restored) {
                        Some(property.local)
                    } else {
                        None
                    }
                });
            let Some(target) = target else {
                diagnostics.push(ReloadDiagnostic::OrphanedProperty {
                    property: property.clone(),
                    reason: ReloadOrphanReason::TargetMissing,
                });
                continue;
            };
            if !component_state
                .get(&target)
                .is_some_and(|components| components.contains(&property.component))
            {
                diagnostics.push(ReloadDiagnostic::OrphanedProperty {
                    property: property.clone(),
                    reason: ReloadOrphanReason::ComponentMissing,
                });
                continue;
            }
            restored.set_property(property.with_local(target));
        }

        // This final check covers property targets introduced by a malformed
        // custom identity map and keeps the returned plan valid for consumers.
        if let Err(error) = restored.validate_for_plan(&new.plan) {
            return Ok(Self {
                old_template: old.guid,
                new_template: new.guid,
                delta,
                overrides: OverrideSet::new(),
                diagnostics: {
                    diagnostics.push(ReloadDiagnostic::InvalidRestoredOverrides { error });
                    diagnostics
                },
            });
        }

        Ok(Self {
            old_template: old.guid,
            new_template: new.guid,
            delta,
            overrides: restored,
            diagnostics,
        })
    }
}

/// Free-function spelling for callers that prefer a planner API.
pub fn plan_reload(
    old: &SceneTemplate,
    new: &SceneTemplate,
    snapshot: &SceneInstancePersistence,
    identities: &PersistentEntityMap,
) -> Result<SceneReloadPlan, ReloadError> {
    SceneReloadPlan::build(old, new, snapshot, identities)
}

fn diff_templates(old: &SceneTemplate, new: &SceneTemplate) -> TemplateDelta {
    let old_locals = plan_locals(old);
    let new_locals = plan_locals(new);
    let added_entities = new_locals.difference(&old_locals).copied().collect();
    let removed_entities = old_locals.difference(&new_locals).copied().collect();

    let old_groups = local_groups(old);
    let new_groups = local_groups(new);
    let mut compared_pairs = BTreeMap::new();
    let mut changed = Vec::new();
    for &local in old_locals.intersection(&new_locals) {
        let old_group = old_groups[&local];
        let new_group = new_groups[&local];
        let pair = (old_group, new_group);
        let same_payload = *compared_pairs.entry(pair).or_insert_with(|| {
            let old_group = &old.plan.groups[old_group];
            let new_group = &new.plan.groups[new_group];
            old_group == new_group
        });
        if !same_payload {
            changed.push(local);
        }
    }

    let old_parents = parents(old);
    let new_parents = parents(new);
    let mut hierarchy_changes = Vec::new();
    for local in old_locals.union(&new_locals) {
        let old_parent = old_parents.get(local).copied();
        let new_parent = new_parents.get(local).copied();
        if old_parent != new_parent {
            hierarchy_changes.push(HierarchyChange {
                local: *local,
                old_parent,
                new_parent,
            });
        }
    }
    TemplateDelta {
        added_entities,
        removed_entities,
        changed_entities: changed,
        hierarchy_changes,
    }
}

fn plan_locals(template: &SceneTemplate) -> BTreeSet<LocalEntityId> {
    (0..template.entity_count())
        .map(LocalEntityId::new)
        .collect()
}

fn local_groups(template: &SceneTemplate) -> BTreeMap<LocalEntityId, usize> {
    template
        .plan
        .groups
        .iter()
        .enumerate()
        .flat_map(|(group, descriptor)| {
            descriptor
                .locals
                .iter()
                .copied()
                .map(move |local| (local, group))
        })
        .collect()
}

fn parents(template: &SceneTemplate) -> BTreeMap<LocalEntityId, LocalEntityId> {
    template.plan.hierarchy.iter().copied().collect()
}

fn template_entity_is_base_or_added(
    local: LocalEntityId,
    template: &SceneTemplate,
    overrides: &OverrideSet,
) -> bool {
    (!local.is_reserved() && local.index() < template.entity_count())
        || overrides.structural().iter().any(|operation| {
            matches!(operation, StructuralOp::AddEntity { local: added, .. } if *added == local)
        })
}

fn component_state(
    template: &SceneTemplate,
    overrides: &OverrideSet,
) -> BTreeMap<LocalEntityId, BTreeSet<ComponentTypeId>> {
    let mut state: BTreeMap<LocalEntityId, BTreeSet<ComponentTypeId>> = BTreeMap::new();
    for group in &template.plan.groups {
        for &local in &group.locals {
            state.insert(
                local,
                group.signature.components().iter().copied().collect(),
            );
        }
    }
    for operation in overrides.structural() {
        match operation {
            StructuralOp::AddEntity {
                local, archetype, ..
            } => {
                state.insert(*local, archetype.components().iter().copied().collect());
            }
            StructuralOp::RemoveEntity { local } => {
                state.remove(local);
            }
            StructuralOp::AddComponent {
                target, component, ..
            } => {
                if let Some(components) = state.get_mut(target) {
                    components.insert(*component);
                }
            }
            StructuralOp::RemoveComponent { target, component } => {
                if let Some(components) = state.get_mut(target) {
                    components.remove(component);
                }
            }
        }
    }
    state
}

fn structural_orphan_reason(
    new: &SceneTemplate,
    retained: &OverrideSet,
    operation: &StructuralOp,
    error: OverrideError,
) -> Option<ReloadOrphanReason> {
    match error {
        OverrideError::RemoveEntityMissing { .. }
        | OverrideError::ComponentTargetMissing { .. } => Some(ReloadOrphanReason::TargetMissing),
        OverrideError::AddedEntityParentMissing { parent, .. }
            if !template_entity_is_base_or_added(parent, new, retained) =>
        {
            Some(ReloadOrphanReason::ParentMissing)
        }
        OverrideError::DuplicateRemoveComponent { target, component }
            if matches!(operation, StructuralOp::RemoveComponent { .. })
                && !new.plan.groups.iter().any(|group| {
                    group.locals.contains(&target) && group.signature.contains(component)
                }) =>
        {
            Some(ReloadOrphanReason::ComponentMissing)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{plan_reload, ReloadDiagnostic, ReloadOrphanReason};
    use crate::identity::LocalEntityId;
    use crate::overrides::{
        ArchetypeSignature, ComponentTypeId, OverrideSet, OverrideValue, PropertyOverride,
    };
    use crate::path::FieldPath;
    use crate::persistence::{PersistentEntityMap, SceneInstancePersistence};
    use crate::plan::{ArchetypeGroup, InstantiationPlan};
    use crate::template::SceneTemplate;
    use alloc::boxed::Box;
    use alloc::vec;
    use prism_asset::StableGuid;

    fn template(guid: u128, count: u32) -> SceneTemplate {
        SceneTemplate::new(
            StableGuid::from_u128(guid),
            InstantiationPlan {
                entity_count: count,
                groups: Box::new([ArchetypeGroup::new(
                    ArchetypeSignature::new(vec![ComponentTypeId::from_raw(1)]),
                    (0..count).map(LocalEntityId::new).collect(),
                    Box::new([crate::Column::new(
                        ComponentTypeId::from_raw(1),
                        vec![1; count as usize],
                    )]),
                )]),
                entity_valued_fields: Box::new([]),
                hierarchy: (1..count)
                    .map(|local| (LocalEntityId::new(local), LocalEntityId::new(0)))
                    .collect(),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            },
        )
    }

    #[test]
    fn reload_prefers_persistent_id_and_reports_removed_property() {
        let old = template(1, 2);
        let new = template(2, 1);
        let old_map = PersistentEntityMap::for_template(&old);
        let mut source = OverrideSet::new();
        source.set_property(PropertyOverride::new(
            LocalEntityId::new(1),
            ComponentTypeId::from_raw(1),
            FieldPath::parse("value").unwrap(),
            OverrideValue::U64(4),
        ));
        let snapshot = SceneInstancePersistence::capture(&old, None, &source, &old_map).snapshot;
        let new_map = PersistentEntityMap::default();
        let plan = plan_reload(&old, &new, &snapshot, &new_map).unwrap();
        assert!(plan.overrides.is_empty());
        assert!(plan.diagnostics.iter().any(|diagnostic| matches!(
            diagnostic,
            ReloadDiagnostic::OrphanedProperty {
                reason: ReloadOrphanReason::TargetMissing,
                ..
            }
        )));
        assert_eq!(plan.delta.removed_entities, vec![LocalEntityId::new(1)]);
    }

    #[test]
    fn reload_preserves_property_when_persistent_id_moves_local() {
        let old = template(1, 1);
        let new = template(2, 2);
        let old_map = PersistentEntityMap::for_template(&old);
        let persistent = old_map.persistent_for_local(LocalEntityId::new(0)).unwrap();
        let mut source = OverrideSet::new();
        source.set_property(PropertyOverride::new(
            LocalEntityId::new(0),
            ComponentTypeId::from_raw(1),
            FieldPath::parse("value").unwrap(),
            OverrideValue::U64(7),
        ));
        let snapshot = SceneInstancePersistence::capture(&old, None, &source, &old_map).snapshot;
        let new_map = PersistentEntityMap::new(vec![crate::EntityBinding::new(
            LocalEntityId::new(1),
            persistent,
        )])
        .unwrap();
        let plan = plan_reload(&old, &new, &snapshot, &new_map).unwrap();
        assert_eq!(plan.overrides.properties()[0].target, LocalEntityId::new(1));
        assert!(plan.diagnostics.is_empty());
    }

    #[test]
    fn diff_marks_entities_in_changed_groups() {
        let old = template(3, 4);
        let mut new = template(4, 4);
        new.plan.groups[0].columns[0] =
            crate::Column::new(ComponentTypeId::from_raw(1), vec![9; 4]);

        let plan = plan_reload(
            &old,
            &new,
            &SceneInstancePersistence {
                template: old.guid,
                instance: None,
                properties: Vec::new(),
                structural: Vec::new(),
            },
            &PersistentEntityMap::for_template(&old),
        )
        .unwrap();
        assert_eq!(
            plan.delta.changed_entities,
            (0..4).map(LocalEntityId::new).collect::<Vec<_>>()
        );
    }

    #[test]
    fn diff_detects_signature_changes_even_when_local_ids_are_stable() {
        let old = template(5, 2);
        let mut new = template(6, 2);
        new.plan.groups[0].signature = ArchetypeSignature::new(vec![ComponentTypeId::from_raw(2)]);
        new.plan.groups[0].columns[0] =
            crate::Column::new(ComponentTypeId::from_raw(2), vec![1; 2]);

        let plan = plan_reload(
            &old,
            &new,
            &SceneInstancePersistence {
                template: old.guid,
                instance: None,
                properties: Vec::new(),
                structural: Vec::new(),
            },
            &PersistentEntityMap::for_template(&old),
        )
        .unwrap();
        assert_eq!(
            plan.delta.changed_entities,
            vec![LocalEntityId::new(0), LocalEntityId::new(1)]
        );
    }
}
