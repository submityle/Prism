//! The override system (design §7) — the "crown" of the scene model.
//!
//! An instance may differ from its template along two axes, both stored
//! sparsely so cost is proportional to the *difference*, not the template size:
//!
//! - **Property overrides** ([`PropertyOverride`]) walk a
//!   [`prism_reflect`](https://docs.rs)-style field path on one component of one
//!   entity: the instance changes `transform.translation.x`, and a later
//!   base-template update to every other field still flows through.
//! - **Structural overrides** ([`StructuralOp`]) add or remove whole entities or
//!   components.
//!
//! Property overrides are keyed by `(target, component, path)` and deduplicated
//! last-write-wins ([`OverrideSet::set_property`]). A structural `RemoveEntity`
//! makes any property override that still targets that entity an *orphan*;
//! [`OverrideSet::drop_orphans`] evicts them and reports the evicted keys
//! (design §7.3), which is what keeps a nested-variant replay from resurrecting
//! a deleted entity.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

use crate::identity::LocalEntityId;
use crate::path::FieldPath;
use crate::plan::{InstantiationPlan, PlanError};

/// FNV-1a 64-bit constants.
const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hashes bytes with FNV-1a 64. Integer-only and deterministic.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV64_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    hash
}

/// A stable identifier for a component type (design §7.1).
///
/// Derived from the type's `type_name`, which is stable within a build and — for
/// the concrete types a bake fixes — across builds of the same source. Complex
/// per-project identity tables can be layered on top via [`from_raw`].
///
/// [`from_raw`]: ComponentTypeId::from_raw
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComponentTypeId(u64);

impl ComponentTypeId {
    /// Derives the id for component type `T`.
    #[must_use]
    pub fn of<T: 'static>() -> Self {
        Self(fnv1a_64(core::any::type_name::<T>().as_bytes()))
    }

    /// Wraps an explicit raw id (e.g. one assigned by a bake manifest).
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw id.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// The serialized value of a property override (design §7.1: "覆盖值(序列化态)").
///
/// Scalars cover the common authoring cases; an aggregate or nested value is
/// carried as its serialized [`Bytes`](Self::Bytes), which is exactly the form
/// the reflect/asset pipeline persists.
#[derive(Clone, Debug, PartialEq)]
pub enum OverrideValue {
    /// A boolean field.
    Bool(bool),
    /// A signed integer field.
    I64(i64),
    /// An unsigned integer field.
    U64(u64),
    /// A 32-bit float field.
    F32(f32),
    /// A 64-bit float field.
    F64(f64),
    /// A string field.
    Str(String),
    /// A serialized aggregate or otherwise opaque value.
    Bytes(Vec<u8>),
}

/// A property override on one component of one entity (design §7.1(a)).
#[derive(Clone, Debug, PartialEq)]
pub struct PropertyOverride {
    /// The entity (template-local) the override targets.
    pub target: LocalEntityId,
    /// The component whose field is overridden.
    pub component: ComponentTypeId,
    /// The reflect-style field path within the component.
    pub path: FieldPath,
    /// The serialized override value.
    pub value: OverrideValue,
}

impl PropertyOverride {
    /// Builds a property override.
    #[must_use]
    pub fn new(
        target: LocalEntityId,
        component: ComponentTypeId,
        path: FieldPath,
        value: OverrideValue,
    ) -> Self {
        Self {
            target,
            component,
            path,
            value,
        }
    }

    /// The unique key `(target, component, path)` this override is stored under.
    #[must_use]
    pub fn key(&self) -> (LocalEntityId, ComponentTypeId, &FieldPath) {
        (self.target, self.component, &self.path)
    }
}

/// A sorted, deduplicated set of component type ids.
///
/// Used by [`StructuralOp::AddEntity`] to describe the archetype a structurally
/// added entity is created in (design §7.1(b)).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArchetypeSignature {
    components: Vec<ComponentTypeId>,
}

impl ArchetypeSignature {
    /// Builds a signature, sorting and deduplicating the component list so two
    /// orderings of the same components compare equal.
    #[must_use]
    pub fn new(mut components: Vec<ComponentTypeId>) -> Self {
        components.sort_unstable();
        components.dedup();
        Self { components }
    }

    /// The sorted, deduplicated components.
    #[must_use]
    pub fn components(&self) -> &[ComponentTypeId] {
        &self.components
    }

    /// Number of distinct components.
    #[must_use]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Whether the signature has no components.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Whether `component` is part of the signature.
    #[must_use]
    pub fn contains(&self, component: ComponentTypeId) -> bool {
        self.components.binary_search(&component).is_ok()
    }
}

/// A structural override: add or remove an entity, or add or remove a component
/// (design §7.1(b)).
///
/// `AddEntity` allocates a [`LocalEntityId`] in the reserved segment (design
/// §6/§7.1), so it can never collide with base numbering and, importantly,
/// survives a base re-bake that reorders base entities.
#[derive(Clone, Debug, PartialEq)]
pub enum StructuralOp {
    /// Adds an entity under `parent` with the given archetype and serialized row.
    AddEntity {
        /// The reserved-segment id of the new entity.
        local: LocalEntityId,
        /// The (template-local) parent it is attached under.
        parent: LocalEntityId,
        /// The archetype the entity is created in.
        archetype: ArchetypeSignature,
        /// The serialized component row.
        data: Vec<u8>,
    },
    /// Removes a template entity from this instance.
    RemoveEntity {
        /// The template-local id to remove.
        local: LocalEntityId,
    },
    /// Adds a component to an existing entity.
    AddComponent {
        /// The target entity.
        target: LocalEntityId,
        /// The component to add.
        component: ComponentTypeId,
        /// The serialized component data.
        data: Vec<u8>,
    },
    /// Removes a component from an existing entity.
    RemoveComponent {
        /// The target entity.
        target: LocalEntityId,
        /// The component to remove.
        component: ComponentTypeId,
    },
}

/// A sparse set of property and structural overrides for one instance
/// (design §7.2).
///
/// The dynamic runtime set is deliberately small — the design pre-composes the
/// bulk of authoring-time overrides into the bake (§7.5) and only replays the
/// overrides that changed at runtime — so storage is a compact, insertion-ordered
/// `Vec`, which keeps replay deterministic and avoids a hash map's non-portable
/// ordering.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OverrideSet {
    properties: Vec<PropertyOverride>,
    structural: Vec<StructuralOp>,
}

impl OverrideSet {
    /// An empty override set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the set holds no overrides at all (the zero-cost instance case).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty() && self.structural.is_empty()
    }

    /// Number of property overrides (structural ops are counted separately).
    #[must_use]
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }

    /// Number of structural ops.
    #[must_use]
    pub fn structural_count(&self) -> usize {
        self.structural.len()
    }

    /// Validates entity references and structural-operation ordering against a
    /// base template containing `entity_count` entities (`0..entity_count`).
    ///
    /// Property targets may refer to a base entity or any entity declared by
    /// an [`StructuralOp::AddEntity`] in this set. Structural operations are
    /// checked in insertion order: added entities must use reserved ids in the
    /// same dense order consumed by [`crate::RemapTable`], their parents must
    /// already exist, removals must name a live entity exactly once and must
    /// not leave a live added child behind, and component operations must
    /// target a live entity without repeating the same add/remove pair.
    ///
    /// This is intentionally a structural validator: `OverrideSet` does not
    /// carry the base template's component inventory, so it cannot establish
    /// whether a removed component exists in the base archetype. It also does
    /// not interpret property paths or validate value/component types.
    pub fn validate_for_template(&self, entity_count: u32) -> Result<(), OverrideError> {
        self.validate_for_template_with_hierarchy(entity_count, &[])
    }

    /// Validates an override set with the base template's hierarchy.
    ///
    /// [`validate_for_template`](Self::validate_for_template) retains its
    /// compact count-only API for callers that have not loaded the hierarchy.
    /// Runtime/template loaders should use [`validate_for_plan`](Self::validate_for_plan)
    /// so component operations can also be checked against the base archetypes.
    pub fn validate_for_template_with_hierarchy(
        &self,
        entity_count: u32,
        hierarchy: &[(LocalEntityId, LocalEntityId)],
    ) -> Result<(), OverrideError> {
        self.validate_impl(entity_count, hierarchy, None)
    }

    /// Validates against the authoritative plan, including each base entity's
    /// component signature and each structurally-added entity's declared
    /// archetype. Component operations must perform real state transitions:
    /// adding an existing component or removing an absent component is rejected.
    pub fn validate_for_plan(&self, plan: &InstantiationPlan) -> Result<(), OverrideError> {
        plan.validate()
            .map_err(|error| OverrideError::InvalidBasePlan { error })?;
        self.validate_impl(plan.entity_count, &plan.hierarchy, Some(plan))
    }

    fn validate_impl(
        &self,
        entity_count: u32,
        hierarchy: &[(LocalEntityId, LocalEntityId)],
        plan: Option<&InstantiationPlan>,
    ) -> Result<(), OverrideError> {
        let mut declared_added = BTreeSet::new();
        for op in &self.structural {
            if let StructuralOp::AddEntity { local, .. } = op {
                declared_added.insert(*local);
            }
        }

        for property in &self.properties {
            if !is_base_entity(property.target, entity_count)
                && !declared_added.contains(&property.target)
            {
                return Err(OverrideError::PropertyTargetMissing {
                    target: property.target,
                });
            }
        }

        let mut added = BTreeSet::new();
        let mut removed = BTreeSet::new();
        let mut added_components = BTreeSet::new();
        let mut removed_components = BTreeSet::new();
        let mut component_state = plan.map(|base| {
            let mut state = BTreeMap::new();
            for group in &base.groups {
                for &local in &group.locals {
                    state.insert(
                        local,
                        group
                            .signature
                            .components()
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>(),
                    );
                }
            }
            state
        });
        let mut parents = BTreeMap::new();

        for &(child, parent) in hierarchy {
            if child == LocalEntityId::new(0)
                || !is_base_entity(child, entity_count)
                || !is_base_entity(parent, entity_count)
                || parents.insert(child, parent).is_some()
            {
                return Err(OverrideError::BaseHierarchyInvalid { child, parent });
            }
        }

        for op in &self.structural {
            match op {
                StructuralOp::AddEntity {
                    local,
                    parent,
                    archetype,
                    ..
                } => {
                    if !local.is_reserved() {
                        return Err(OverrideError::AddedEntityIdNotReserved { local: *local });
                    }
                    // The id itself is representable, but there must also be a
                    // representable successor if this reserved id is consumed.
                    if local.index().checked_add(1).is_none() {
                        return Err(OverrideError::ReservedIdOverflow { local: *local });
                    }
                    if added.contains(local) {
                        return Err(OverrideError::DuplicateAddedEntity { local: *local });
                    }
                    let expected_offset = added.len() as u32;
                    let expected = LocalEntityId::RESERVED_BASE
                        .checked_add(expected_offset)
                        .ok_or(OverrideError::ReservedIdOverflow { local: *local })?;
                    if local.index() != expected {
                        return Err(OverrideError::ReservedIdNotDense {
                            expected: LocalEntityId(expected),
                            found: *local,
                        });
                    }
                    if local == parent {
                        return Err(OverrideError::ParentIsSelf { local: *local });
                    }
                    if !entity_is_live(*parent, entity_count, &added, &removed) {
                        return Err(OverrideError::AddedEntityParentMissing {
                            local: *local,
                            parent: *parent,
                        });
                    }
                    added.insert(*local);
                    parents.insert(*local, *parent);
                    if let Some(state) = &mut component_state {
                        state.insert(*local, archetype.components().iter().copied().collect());
                    }
                }
                StructuralOp::RemoveEntity { local } => {
                    if removed.contains(local) {
                        return Err(OverrideError::DuplicateRemoveEntity { local: *local });
                    }
                    if !entity_is_live(*local, entity_count, &added, &removed) {
                        return Err(OverrideError::RemoveEntityMissing { local: *local });
                    }
                    if let Some((&child, _)) = parents.iter().find(|(child, parent)| {
                        **parent == *local
                            && entity_is_live(**child, entity_count, &added, &removed)
                    }) {
                        return Err(OverrideError::RemoveEntityHasLiveChild {
                            parent: *local,
                            child,
                        });
                    }
                    removed.insert(*local);
                    if let Some(state) = &mut component_state {
                        state.remove(local);
                    }
                }
                StructuralOp::AddComponent {
                    target, component, ..
                } => {
                    if !entity_is_live(*target, entity_count, &added, &removed) {
                        return Err(OverrideError::ComponentTargetMissing { target: *target });
                    }
                    if let Some(state) = &mut component_state {
                        let components = state
                            .get_mut(target)
                            .expect("live entity has component state");
                        if !components.insert(*component) {
                            return Err(OverrideError::DuplicateAddComponent {
                                target: *target,
                                component: *component,
                            });
                        }
                    } else if !added_components.insert((*target, *component)) {
                        return Err(OverrideError::DuplicateAddComponent {
                            target: *target,
                            component: *component,
                        });
                    }
                }
                StructuralOp::RemoveComponent { target, component } => {
                    if !entity_is_live(*target, entity_count, &added, &removed) {
                        return Err(OverrideError::ComponentTargetMissing { target: *target });
                    }
                    if let Some(state) = &mut component_state {
                        let components = state
                            .get_mut(target)
                            .expect("live entity has component state");
                        if !components.remove(component) {
                            return Err(OverrideError::DuplicateRemoveComponent {
                                target: *target,
                                component: *component,
                            });
                        }
                    } else if !removed_components.insert((*target, *component)) {
                        return Err(OverrideError::DuplicateRemoveComponent {
                            target: *target,
                            component: *component,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// The property overrides, in insertion order.
    #[must_use]
    pub fn properties(&self) -> &[PropertyOverride] {
        &self.properties
    }

    /// The structural ops, in application order.
    #[must_use]
    pub fn structural(&self) -> &[StructuralOp] {
        &self.structural
    }

    /// Looks up the property override for `(target, component, path)`.
    #[must_use]
    pub fn property(
        &self,
        target: LocalEntityId,
        component: ComponentTypeId,
        path: &FieldPath,
    ) -> Option<&PropertyOverride> {
        self.properties
            .iter()
            .find(|ov| ov.target == target && ov.component == component && &ov.path == path)
    }

    /// Inserts or replaces the property override for its `(target, component,
    /// path)` key, returning the previous value if one existed (last-write-wins).
    pub fn set_property(&mut self, ov: PropertyOverride) -> Option<PropertyOverride> {
        if let Some(existing) = self
            .properties
            .iter_mut()
            .find(|e| e.target == ov.target && e.component == ov.component && e.path == ov.path)
        {
            return Some(core::mem::replace(existing, ov));
        }
        self.properties.push(ov);
        None
    }

    /// Removes and returns the property override for `(target, component, path)`.
    pub fn remove_property(
        &mut self,
        target: LocalEntityId,
        component: ComponentTypeId,
        path: &FieldPath,
    ) -> Option<PropertyOverride> {
        let index = self
            .properties
            .iter()
            .position(|ov| ov.target == target && ov.component == component && &ov.path == path)?;
        Some(self.properties.remove(index))
    }

    /// Appends a structural op.
    pub fn push_structural(&mut self, op: StructuralOp) {
        self.structural.push(op);
    }

    /// The set of template-local entity ids removed by a `RemoveEntity` op.
    #[must_use]
    pub fn removed_entities(&self) -> Vec<LocalEntityId> {
        self.structural
            .iter()
            .filter_map(|op| match op {
                StructuralOp::RemoveEntity { local } => Some(*local),
                _ => None,
            })
            .collect()
    }

    /// Evicts property overrides whose target entity or component was removed
    /// by the structural ops in this set, returning the evicted `(target,
    /// component, path)` keys (design §7.3 `OrphanOverride`). An
    /// `AddComponent` later in the operation sequence makes a prior
    /// `RemoveComponent` live again.
    pub fn drop_orphans(&mut self) -> Vec<(LocalEntityId, ComponentTypeId, FieldPath)> {
        let removed = self.removed_entities();
        let mut removed_components = BTreeSet::new();
        for op in &self.structural {
            match op {
                StructuralOp::AddComponent {
                    target, component, ..
                } => {
                    removed_components.remove(&(*target, *component));
                }
                StructuralOp::RemoveComponent { target, component } => {
                    removed_components.insert((*target, *component));
                }
                StructuralOp::AddEntity { .. } | StructuralOp::RemoveEntity { .. } => {}
            }
        }
        if removed.is_empty() && removed_components.is_empty() {
            return Vec::new();
        }
        let mut orphans = Vec::new();
        let mut kept = Vec::with_capacity(self.properties.len());
        for ov in self.properties.drain(..) {
            if removed.contains(&ov.target)
                || removed_components.contains(&(ov.target, ov.component))
            {
                orphans.push((ov.target, ov.component, ov.path));
            } else {
                kept.push(ov);
            }
        }
        self.properties = kept;
        orphans
    }
}

/// A structural defect in an [`OverrideSet`] relative to a base template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverrideError {
    /// The authoritative base plan is structurally invalid.
    InvalidBasePlan {
        /// The plan validation defect.
        error: PlanError,
    },
    /// A property override targets neither a base entity nor a declared addition.
    PropertyTargetMissing {
        /// The invalid property target.
        target: LocalEntityId,
    },
    /// An added entity used an id from the base segment.
    AddedEntityIdNotReserved {
        /// The non-reserved id used by the add operation.
        local: LocalEntityId,
    },
    /// An added entity did not use the next dense reserved-segment id.
    ReservedIdNotDense {
        /// The next id required by the remap table.
        expected: LocalEntityId,
        /// The sparse id supplied by the override.
        found: LocalEntityId,
    },
    /// An added entity reused an id already declared by an earlier addition.
    DuplicateAddedEntity {
        /// The id declared more than once.
        local: LocalEntityId,
    },
    /// An added entity's parent did not exist when the operation was applied.
    AddedEntityParentMissing {
        /// The entity being added.
        local: LocalEntityId,
        /// The parent not yet available at this operation.
        parent: LocalEntityId,
    },
    /// An added entity was attached to itself.
    ParentIsSelf {
        /// The entity that names itself as parent.
        local: LocalEntityId,
    },
    /// The supplied base hierarchy contains an invalid or duplicate edge.
    BaseHierarchyInvalid {
        /// The child local id.
        child: LocalEntityId,
        /// The parent local id.
        parent: LocalEntityId,
    },
    /// Removing an entity would leave a live child without a parent.
    RemoveEntityHasLiveChild {
        /// The entity being removed.
        parent: LocalEntityId,
        /// The child that would become dangling.
        child: LocalEntityId,
    },
    /// A reserved local id has no representable successor (u32 overflow).
    ReservedIdOverflow {
        /// The reserved id whose successor cannot be represented.
        local: LocalEntityId,
    },
    /// A remove operation targets an entity that is not currently alive.
    RemoveEntityMissing {
        /// The entity requested for removal.
        local: LocalEntityId,
    },
    /// An entity is removed more than once.
    DuplicateRemoveEntity {
        /// The entity removed more than once.
        local: LocalEntityId,
    },
    /// A component operation targets an entity that is not currently alive.
    ComponentTargetMissing {
        /// The entity targeted by the component operation.
        target: LocalEntityId,
    },
    /// The same component is added to the same entity more than once.
    DuplicateAddComponent {
        /// The entity receiving the duplicate add.
        target: LocalEntityId,
        /// The component repeated on that entity.
        component: ComponentTypeId,
    },
    /// The same component is removed from the same entity more than once.
    DuplicateRemoveComponent {
        /// The entity receiving the duplicate remove.
        target: LocalEntityId,
        /// The component repeated on that entity.
        component: ComponentTypeId,
    },
}

fn is_base_entity(local: LocalEntityId, entity_count: u32) -> bool {
    !local.is_reserved() && local.index() < entity_count
}

fn entity_is_live(
    local: LocalEntityId,
    entity_count: u32,
    added: &BTreeSet<LocalEntityId>,
    removed: &BTreeSet<LocalEntityId>,
) -> bool {
    (is_base_entity(local, entity_count) || added.contains(&local)) && !removed.contains(&local)
}

#[cfg(test)]
mod tests {
    use super::{
        ArchetypeSignature, ComponentTypeId, OverrideSet, OverrideValue, PropertyOverride,
        StructuralOp,
    };
    use crate::identity::LocalEntityId;
    use crate::path::FieldPath;
    use alloc::vec;

    struct Transform;
    struct Health;

    fn ov(target: u32, component: ComponentTypeId, path: &str, value: f32) -> PropertyOverride {
        PropertyOverride::new(
            LocalEntityId::new(target),
            component,
            FieldPath::parse(path).unwrap(),
            OverrideValue::F32(value),
        )
    }

    #[test]
    fn component_type_id_is_stable_and_distinct() {
        assert_eq!(
            ComponentTypeId::of::<Transform>(),
            ComponentTypeId::of::<Transform>()
        );
        assert_ne!(
            ComponentTypeId::of::<Transform>(),
            ComponentTypeId::of::<Health>()
        );
        assert_eq!(ComponentTypeId::from_raw(7).raw(), 7);
    }

    #[test]
    fn archetype_signature_sorts_and_dedups() {
        let a = ComponentTypeId::of::<Transform>();
        let b = ComponentTypeId::of::<Health>();
        let s = ArchetypeSignature::new(vec![b, a, b]);
        assert_eq!(s.len(), 2);
        assert!(s.contains(a) && s.contains(b));
        assert_eq!(s, ArchetypeSignature::new(vec![a, b]));
        assert!(ArchetypeSignature::default().is_empty());
    }

    #[test]
    fn property_override_dedups_last_write_wins() {
        let t = ComponentTypeId::of::<Transform>();
        let mut set = OverrideSet::new();
        assert!(set.is_empty());

        assert_eq!(set.set_property(ov(1, t, "translation.x", 1.0)), None);
        let previous = set
            .set_property(ov(1, t, "translation.x", 2.0))
            .expect("second write replaces the first");
        assert_eq!(previous.value, OverrideValue::F32(1.0));
        assert_eq!(set.property_count(), 1, "the key is unique");

        // A different path is a distinct key.
        set.set_property(ov(1, t, "translation.y", 3.0));
        assert_eq!(set.property_count(), 2);

        let path = FieldPath::parse("translation.x").unwrap();
        assert_eq!(
            set.property(LocalEntityId::new(1), t, &path).unwrap().value,
            OverrideValue::F32(2.0)
        );
    }

    #[test]
    fn remove_property_matches_key() {
        let t = ComponentTypeId::of::<Transform>();
        let mut set = OverrideSet::new();
        set.set_property(ov(1, t, "translation.x", 1.0));
        set.set_property(ov(1, t, "translation.y", 2.0));

        let path = FieldPath::parse("translation.x").unwrap();
        let removed = set.remove_property(LocalEntityId::new(1), t, &path);
        assert!(removed.is_some());
        assert_eq!(set.property_count(), 1);
        assert_eq!(set.remove_property(LocalEntityId::new(1), t, &path), None);
    }

    #[test]
    fn drop_orphans_evicts_overrides_for_removed_entities() {
        let t = ComponentTypeId::of::<Transform>();
        let mut set = OverrideSet::new();
        set.set_property(ov(1, t, "translation.x", 1.0));
        set.set_property(ov(2, t, "translation.x", 2.0));
        set.push_structural(StructuralOp::RemoveEntity {
            local: LocalEntityId::new(1),
        });

        assert_eq!(set.removed_entities(), vec![LocalEntityId::new(1)]);
        let orphans = set.drop_orphans();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].0, LocalEntityId::new(1));
        assert_eq!(set.property_count(), 1, "the surviving target is kept");
        assert!(set.property_count() == 1);
        // The kept override is the one for target 2.
        assert_eq!(set.properties()[0].target, LocalEntityId::new(2));
    }

    #[test]
    fn drop_orphans_is_a_noop_without_removals() {
        let t = ComponentTypeId::of::<Transform>();
        let mut set = OverrideSet::new();
        set.set_property(ov(1, t, "translation.x", 1.0));
        assert!(set.drop_orphans().is_empty());
        assert_eq!(set.property_count(), 1);
    }

    #[test]
    fn drop_orphans_evicts_properties_for_removed_components() {
        let component = ComponentTypeId::from_raw(9);
        let mut set = OverrideSet::new();
        set.set_property(ov(0, component, "value", 1.0));
        set.push_structural(StructuralOp::RemoveComponent {
            target: LocalEntityId::new(0),
            component,
        });
        let orphans = set.drop_orphans();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].1, component);
        assert!(set.properties().is_empty());
    }

    #[test]
    fn structural_ops_are_ordered_and_non_empty() {
        let h = ComponentTypeId::of::<Health>();
        let mut set = OverrideSet::new();
        set.push_structural(StructuralOp::RemoveComponent {
            target: LocalEntityId::new(0),
            component: h,
        });
        set.push_structural(StructuralOp::RemoveEntity {
            local: LocalEntityId::new(0),
        });
        assert_eq!(set.structural_count(), 2);
        assert!(!set.is_empty());
    }

    fn add_entity(set: &mut OverrideSet, local: u32, parent: u32) {
        set.push_structural(StructuralOp::AddEntity {
            local: LocalEntityId::new(local),
            parent: LocalEntityId::new(parent),
            archetype: ArchetypeSignature::default(),
            data: vec![],
        });
    }

    #[test]
    fn validate_accepts_base_and_declared_added_property_targets() {
        let component = ComponentTypeId::from_raw(1);
        let mut set = OverrideSet::new();
        add_entity(&mut set, LocalEntityId::RESERVED_BASE, 0);
        set.set_property(ov(0, component, "x", 1.0));
        set.set_property(ov(LocalEntityId::RESERVED_BASE, component, "x", 2.0));
        assert_eq!(set.validate_for_template(1), Ok(()));
    }

    #[test]
    fn validate_rejects_property_target_outside_base_and_additions() {
        let mut set = OverrideSet::new();
        set.set_property(ov(2, ComponentTypeId::from_raw(1), "x", 1.0));
        assert_eq!(
            set.validate_for_template(2),
            Err(super::OverrideError::PropertyTargetMissing {
                target: LocalEntityId::new(2)
            })
        );
    }

    #[test]
    fn validate_rejects_non_reserved_added_id() {
        let mut set = OverrideSet::new();
        add_entity(&mut set, 1, 0);
        assert_eq!(
            set.validate_for_template(2),
            Err(super::OverrideError::AddedEntityIdNotReserved {
                local: LocalEntityId::new(1)
            })
        );
    }

    #[test]
    fn validate_rejects_duplicate_added_id() {
        let mut set = OverrideSet::new();
        add_entity(&mut set, LocalEntityId::RESERVED_BASE, 0);
        add_entity(&mut set, LocalEntityId::RESERVED_BASE, 0);
        assert_eq!(
            set.validate_for_template(1),
            Err(super::OverrideError::DuplicateAddedEntity {
                local: LocalEntityId::new(LocalEntityId::RESERVED_BASE)
            })
        );
    }

    #[test]
    fn validate_requires_add_parent_to_exist_before_add() {
        let mut set = OverrideSet::new();
        add_entity(
            &mut set,
            LocalEntityId::RESERVED_BASE,
            LocalEntityId::RESERVED_BASE + 1,
        );
        assert_eq!(
            set.validate_for_template(1),
            Err(super::OverrideError::AddedEntityParentMissing {
                local: LocalEntityId::new(LocalEntityId::RESERVED_BASE),
                parent: LocalEntityId::new(LocalEntityId::RESERVED_BASE + 1)
            })
        );
    }

    #[test]
    fn validate_rejects_self_parent() {
        let mut set = OverrideSet::new();
        add_entity(
            &mut set,
            LocalEntityId::RESERVED_BASE,
            LocalEntityId::RESERVED_BASE,
        );
        assert_eq!(
            set.validate_for_template(1),
            Err(super::OverrideError::ParentIsSelf {
                local: LocalEntityId::new(LocalEntityId::RESERVED_BASE)
            })
        );
    }

    #[test]
    fn validate_rejects_reserved_id_successor_overflow() {
        let mut set = OverrideSet::new();
        add_entity(&mut set, u32::MAX, 0);
        assert_eq!(
            set.validate_for_template(1),
            Err(super::OverrideError::ReservedIdOverflow {
                local: LocalEntityId::new(u32::MAX)
            })
        );
    }

    #[test]
    fn validate_rejects_sparse_reserved_id() {
        let mut set = OverrideSet::new();
        add_entity(&mut set, LocalEntityId::RESERVED_BASE + 1, 0);
        assert_eq!(
            set.validate_for_template(1),
            Err(super::OverrideError::ReservedIdNotDense {
                expected: LocalEntityId::reserved(0),
                found: LocalEntityId::reserved(1),
            })
        );
    }

    #[test]
    fn hierarchy_aware_validation_rejects_dangling_child() {
        let mut set = OverrideSet::new();
        set.push_structural(StructuralOp::RemoveEntity {
            local: LocalEntityId::new(0),
        });
        assert_eq!(
            set.validate_for_template_with_hierarchy(
                2,
                &[(LocalEntityId::new(1), LocalEntityId::new(0))],
            ),
            Err(super::OverrideError::RemoveEntityHasLiveChild {
                parent: LocalEntityId::new(0),
                child: LocalEntityId::new(1),
            })
        );
    }

    #[test]
    fn validate_rejects_missing_or_repeated_entity_removal() {
        let mut missing = OverrideSet::new();
        missing.push_structural(StructuralOp::RemoveEntity {
            local: LocalEntityId::new(4),
        });
        assert_eq!(
            missing.validate_for_template(1),
            Err(super::OverrideError::RemoveEntityMissing {
                local: LocalEntityId::new(4)
            })
        );

        let mut repeated = OverrideSet::new();
        for _ in 0..2 {
            repeated.push_structural(StructuralOp::RemoveEntity {
                local: LocalEntityId::new(0),
            });
        }
        assert_eq!(
            repeated.validate_for_template(1),
            Err(super::OverrideError::DuplicateRemoveEntity {
                local: LocalEntityId::new(0)
            })
        );
    }

    #[test]
    fn validate_component_ops_require_live_target() {
        let mut set = OverrideSet::new();
        set.push_structural(StructuralOp::AddComponent {
            target: LocalEntityId::new(2),
            component: ComponentTypeId::from_raw(1),
            data: vec![],
        });
        assert_eq!(
            set.validate_for_template(2),
            Err(super::OverrideError::ComponentTargetMissing {
                target: LocalEntityId::new(2)
            })
        );
    }

    #[test]
    fn plan_validation_rejects_add_of_existing_component_and_remove_of_absent() {
        use crate::plan::{ArchetypeGroup, InstantiationPlan};
        let component = ComponentTypeId::from_raw(7);
        let plan = InstantiationPlan {
            entity_count: 1,
            groups: vec![ArchetypeGroup::new(
                ArchetypeSignature::new(vec![component]),
                vec![LocalEntityId::new(0)].into_boxed_slice(),
                vec![crate::plan::Column::new(component, vec![0u8])].into_boxed_slice(),
            )]
            .into_boxed_slice(),
            entity_valued_fields: vec![].into_boxed_slice(),
            hierarchy: vec![].into_boxed_slice(),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        };
        let mut add = OverrideSet::new();
        add.push_structural(StructuralOp::AddComponent {
            target: LocalEntityId::new(0),
            component,
            data: vec![],
        });
        assert!(matches!(
            add.validate_for_plan(&plan),
            Err(super::OverrideError::DuplicateAddComponent { .. })
        ));

        let mut remove = OverrideSet::new();
        remove.push_structural(StructuralOp::RemoveComponent {
            target: LocalEntityId::new(0),
            component: ComponentTypeId::from_raw(99),
        });
        assert!(matches!(
            remove.validate_for_plan(&plan),
            Err(super::OverrideError::DuplicateRemoveComponent { .. })
        ));
    }

    #[test]
    fn validate_rejects_repeated_add_component() {
        let mut set = OverrideSet::new();
        for _ in 0..2 {
            set.push_structural(StructuralOp::AddComponent {
                target: LocalEntityId::new(0),
                component: ComponentTypeId::from_raw(7),
                data: vec![],
            });
        }
        assert_eq!(
            set.validate_for_template(1),
            Err(super::OverrideError::DuplicateAddComponent {
                target: LocalEntityId::new(0),
                component: ComponentTypeId::from_raw(7)
            })
        );
    }

    #[test]
    fn validate_rejects_repeated_remove_component() {
        let mut set = OverrideSet::new();
        for _ in 0..2 {
            set.push_structural(StructuralOp::RemoveComponent {
                target: LocalEntityId::new(0),
                component: ComponentTypeId::from_raw(7),
            });
        }
        assert_eq!(
            set.validate_for_template(1),
            Err(super::OverrideError::DuplicateRemoveComponent {
                target: LocalEntityId::new(0),
                component: ComponentTypeId::from_raw(7)
            })
        );
    }

    #[test]
    fn validate_rejects_component_op_after_entity_removal() {
        let mut set = OverrideSet::new();
        set.push_structural(StructuralOp::RemoveEntity {
            local: LocalEntityId::new(0),
        });
        set.push_structural(StructuralOp::RemoveComponent {
            target: LocalEntityId::new(0),
            component: ComponentTypeId::from_raw(7),
        });
        assert_eq!(
            set.validate_for_template(1),
            Err(super::OverrideError::ComponentTargetMissing {
                target: LocalEntityId::new(0)
            })
        );
    }
}
