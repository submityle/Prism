//! The immutable, shareable [`SceneTemplate`] descriptor (design §5.1, §9).
//!
//! A template is the runtime-resident form of a scene asset: it is created by
//! the bake pipeline, loaded through `prism_asset`, and shared by any number of
//! instances. It is deliberately ECS-free — it holds only identity, the
//! flattened [`InstantiationPlan`], the local name table, the nested-instance
//! slots and the dependency closure, all keyed by [`StableGuid`] (and, in the
//! plan, by [`LocalEntityId`]) so it carries no live handles.

use alloc::boxed::Box;

use prism_asset::StableGuid;

use crate::identity::{LocalEntityId, NameTable};
use crate::overrides::{OverrideError, OverrideSet};
use crate::plan::{InstantiationPlan, PlanError};
use crate::residency::{ResidencyPlan, ResidencyTier};

/// A nested instance slot inside a template (design §9.1).
///
/// A template may contain other templates as nested instances; each slot names
/// the local entity the nested instance is rooted at and the sub-template's
/// asset identity. A *level* is simply a top-level template with loose entities
/// plus such nested slots — level and prefab are one mechanism, not two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NestedSlot {
    /// The local entity the nested instance is rooted at.
    pub root: LocalEntityId,
    /// The nested sub-template's asset identity.
    pub template: StableGuid,
}

/// Structural failure in a runtime-resident scene template descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemplateError {
    /// The template or a dependency used the reserved nil GUID.
    NilGuid,
    /// The flattened instantiation plan is invalid.
    InvalidPlan(PlanError),
    /// A name was empty or targeted a non-base local entity.
    InvalidNameTarget(LocalEntityId),
    /// A residency annotation targeted a non-base local entity.
    InvalidTierTarget(LocalEntityId),
    /// A nested slot used a nil child template GUID.
    NilNestedTemplate,
    /// A nested slot pointed directly back to this template.
    SelfNestedTemplate,
    /// A nested slot root was not a base entity in this template.
    InvalidNestedRoot(LocalEntityId),
    /// Two nested slots claimed the same root entity.
    DuplicateNestedRoot(LocalEntityId),
    /// A nested template was omitted from the dependency closure.
    MissingNestedDependency(StableGuid),
    /// A dependency entry used the reserved nil GUID.
    NilDependency,
    /// The dependency closure contained this template's own GUID.
    SelfDependency,
    /// The dependency closure contained the same GUID twice.
    DuplicateDependency(StableGuid),
}

impl NestedSlot {
    /// Builds a nested slot.
    #[must_use]
    pub const fn new(root: LocalEntityId, template: StableGuid) -> Self {
        Self { root, template }
    }
}

/// A runtime-resident, immutable scene template (design §5.1).
///
/// The full bake-time `InstantiationPlan` (design §8) is produced and consumed
/// by the instantiation pipeline; this descriptor carries the identity and the
/// structural facts the runtime model needs to validate an instance, resolve
/// names, walk nested slots and reason about the dependency closure.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneTemplate {
    /// The template's stable asset identity (design §6 `StableGuid`).
    pub guid: StableGuid,
    /// The flattened instantiation plan (design §5.1, §8.1).
    pub plan: InstantiationPlan,
    /// Template-local `name -> LocalEntityId` table.
    pub names: NameTable,
    /// Nested instance slots, in author order.
    pub nested: Box<[NestedSlot]>,
    /// Per-entity residency annotations (design §5.1 `ResidencyPlan`, §11.1).
    pub tiers: ResidencyPlan,
    /// The dependency closure (nested sub-templates and referenced assets), by
    /// identity.
    pub deps: Box<[StableGuid]>,
}

impl SceneTemplate {
    /// Builds a bare template around a validated instantiation plan.
    #[must_use]
    pub fn new(guid: StableGuid, plan: InstantiationPlan) -> Self {
        Self {
            guid,
            plan,
            names: NameTable::new(),
            nested: Box::new([]),
            tiers: ResidencyPlan::new(),
            deps: Box::new([]),
        }
    }

    /// Builds and validates a runtime template in one fallible operation.
    pub fn try_new(guid: StableGuid, plan: InstantiationPlan) -> Result<Self, TemplateError> {
        let template = Self::new(guid, plan);
        template.validate()?;
        Ok(template)
    }

    /// Builds an entity-less template (a valid empty plan).
    #[must_use]
    pub fn empty(guid: StableGuid) -> Self {
        Self::new(
            guid,
            InstantiationPlan {
                entity_count: 0,
                groups: Box::new([]),
                entity_valued_fields: Box::new([]),
                hierarchy: Box::new([]),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            },
        )
    }

    /// Builds and validates an empty runtime template.
    pub fn try_empty(guid: StableGuid) -> Result<Self, TemplateError> {
        Self::try_new(
            guid,
            InstantiationPlan {
                entity_count: 0,
                groups: Box::new([]),
                entity_valued_fields: Box::new([]),
                hierarchy: Box::new([]),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            },
        )
    }

    /// Annotates one local entity's residency tier (builder style).
    #[must_use]
    pub fn with_tier(mut self, local: LocalEntityId, tier: ResidencyTier) -> Self {
        self.tiers.set(local, tier);
        self
    }

    /// Whether `local` is a base-segment entity of this template.
    #[must_use]
    pub fn contains_local(&self, local: LocalEntityId) -> bool {
        !local.is_reserved() && local.index() < self.plan.entity_count
    }

    /// Number of base-segment entities the template instantiates.
    #[must_use]
    pub fn entity_count(&self) -> u32 {
        self.plan.entity_count
    }

    /// Resolves a local name through the template's [`NameTable`].
    #[must_use]
    pub fn local_of(&self, name: &str) -> Option<LocalEntityId> {
        self.names.get(name)
    }

    /// The nested slot rooted at `root`, if any.
    #[must_use]
    pub fn nested_at(&self, root: LocalEntityId) -> Option<&NestedSlot> {
        self.nested.iter().find(|slot| slot.root == root)
    }

    /// Number of nested instance slots.
    #[must_use]
    pub fn nested_count(&self) -> usize {
        self.nested.len()
    }

    /// Number of dependencies in the closure.
    #[must_use]
    pub fn dependency_count(&self) -> usize {
        self.deps.len()
    }

    /// Validates all runtime descriptor invariants before asset publication or
    /// instantiation.
    pub fn validate(&self) -> Result<(), TemplateError> {
        if self.guid.is_nil() {
            return Err(TemplateError::NilGuid);
        }
        self.plan.validate().map_err(TemplateError::InvalidPlan)?;
        for (name, local) in self.names.iter() {
            if name.is_empty() || !self.contains_local(local) {
                return Err(TemplateError::InvalidNameTarget(local));
            }
        }
        for (local, _) in self.tiers.iter() {
            if !self.contains_local(local) {
                return Err(TemplateError::InvalidTierTarget(local));
            }
        }
        let mut nested_roots = alloc::collections::BTreeSet::new();
        for slot in self.nested.iter() {
            if slot.template.is_nil() {
                return Err(TemplateError::NilNestedTemplate);
            }
            if slot.template == self.guid {
                return Err(TemplateError::SelfNestedTemplate);
            }
            if !self.contains_local(slot.root) {
                return Err(TemplateError::InvalidNestedRoot(slot.root));
            }
            if !nested_roots.insert(slot.root) {
                return Err(TemplateError::DuplicateNestedRoot(slot.root));
            }
        }
        let mut deps = alloc::collections::BTreeSet::new();
        for dependency in self.deps.iter().copied() {
            if dependency.is_nil() {
                return Err(TemplateError::NilDependency);
            }
            if dependency == self.guid {
                return Err(TemplateError::SelfDependency);
            }
            if !deps.insert(dependency) {
                return Err(TemplateError::DuplicateDependency(dependency));
            }
        }
        for slot in self.nested.iter() {
            if !deps.contains(&slot.template) {
                return Err(TemplateError::MissingNestedDependency(slot.template));
            }
        }
        Ok(())
    }

    /// Validates an instance/variant override against this template's complete
    /// base hierarchy.
    ///
    /// The count-only validator is useful for lightweight callers, but it
    /// cannot know whether removing a base parent would strand a child. This
    /// template-owned entry point supplies the authoritative hierarchy from
    /// the validated instantiation plan.
    pub fn validate_overrides(&self, overrides: &OverrideSet) -> Result<(), OverrideError> {
        overrides.validate_for_template_with_hierarchy(self.plan.entity_count, &self.plan.hierarchy)
    }
}

#[cfg(test)]
mod tests {
    use super::{NestedSlot, SceneTemplate, TemplateError};
    use crate::identity::LocalEntityId;
    use crate::overrides::ArchetypeSignature;
    use crate::plan::{ArchetypeGroup, InstantiationPlan};
    use crate::residency::ResidencyTier;
    use alloc::boxed::Box;
    use prism_asset::StableGuid;

    fn guid(seed: u128) -> StableGuid {
        StableGuid::from_u128(seed)
    }

    /// A valid plan with `n` entities in one empty-signature group.
    fn plan_of(n: u32) -> InstantiationPlan {
        let locals: Box<[LocalEntityId]> = (0..n).map(LocalEntityId::new).collect();
        InstantiationPlan {
            entity_count: n,
            groups: Box::new([ArchetypeGroup::new(
                ArchetypeSignature::default(),
                locals,
                Box::new([]),
            )]),
            entity_valued_fields: Box::new([]),
            hierarchy: (1..n)
                .map(|local| (LocalEntityId::new(local), LocalEntityId::new(0)))
                .collect(),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        }
    }

    #[test]
    fn contains_local_bounds_base_segment_only() {
        let template = SceneTemplate::new(guid(1), plan_of(3));
        assert!(template.contains_local(LocalEntityId::new(0)));
        assert!(template.contains_local(LocalEntityId::new(2)));
        assert!(!template.contains_local(LocalEntityId::new(3)));
        // Reserved ids are never part of the base set.
        assert!(!template.contains_local(LocalEntityId::reserved(0)));
        assert_eq!(template.entity_count(), 3);
    }

    #[test]
    fn names_and_nested_slots_resolve() {
        let mut template = SceneTemplate::new(guid(1), plan_of(4));
        template.names.insert("root", LocalEntityId::new(0));
        template.names.insert("wheel", LocalEntityId::new(3));
        template.nested = Box::new([NestedSlot::new(LocalEntityId::new(1), guid(2))]);
        template.deps = Box::new([guid(2), guid(3)]);

        assert_eq!(template.local_of("wheel"), Some(LocalEntityId::new(3)));
        assert_eq!(template.local_of("missing"), None);
        assert_eq!(
            template.nested_at(LocalEntityId::new(1)),
            Some(&NestedSlot::new(LocalEntityId::new(1), guid(2)))
        );
        assert_eq!(template.nested_at(LocalEntityId::new(2)), None);
        assert_eq!(template.nested_count(), 1);
        assert_eq!(template.dependency_count(), 2);
    }

    #[test]
    fn new_template_starts_empty() {
        let template = SceneTemplate::empty(guid(7));
        assert_eq!(template.nested_count(), 0);
        assert_eq!(template.dependency_count(), 0);
        assert!(template.names.is_empty());
        assert!(template.tiers.is_empty());
        assert_eq!(template.entity_count(), 0);
        assert!(template.plan.validate().is_ok());
    }

    #[test]
    fn validates_overrides_against_the_base_hierarchy() {
        let template = SceneTemplate::new(guid(8), plan_of(2));
        let mut overrides = crate::overrides::OverrideSet::new();
        overrides.push_structural(crate::overrides::StructuralOp::RemoveEntity {
            local: LocalEntityId::new(0),
        });
        assert_eq!(
            template.validate_overrides(&overrides),
            Err(crate::overrides::OverrideError::RemoveEntityHasLiveChild {
                parent: LocalEntityId::new(0),
                child: LocalEntityId::new(1),
            })
        );
    }

    #[test]
    fn validates_descriptor_identity_and_nested_closure() {
        let template_guid = guid(1);
        let mut template = SceneTemplate::new(template_guid, plan_of(2));
        template.names.insert("root", LocalEntityId::new(0));
        template = template.with_tier(LocalEntityId::new(1), ResidencyTier::Proxy);
        template.nested = Box::new([NestedSlot::new(LocalEntityId::new(1), guid(2))]);
        template.deps = Box::new([guid(2)]);
        assert!(template.validate().is_ok());

        template.names.insert("dangling", LocalEntityId::new(2));
        assert_eq!(
            template.validate(),
            Err(TemplateError::InvalidNameTarget(LocalEntityId::new(2)))
        );
    }

    #[test]
    fn rejects_duplicate_nested_roots_and_dependencies() {
        let template_guid = guid(1);
        let mut template = SceneTemplate::new(template_guid, plan_of(2));
        template.nested = Box::new([
            NestedSlot::new(LocalEntityId::new(0), guid(2)),
            NestedSlot::new(LocalEntityId::new(0), guid(3)),
        ]);
        assert_eq!(
            template.validate(),
            Err(TemplateError::DuplicateNestedRoot(LocalEntityId::new(0)))
        );

        template.nested = Box::new([]);
        template.deps = Box::new([guid(2), guid(2)]);
        assert_eq!(
            template.validate(),
            Err(TemplateError::DuplicateDependency(guid(2)))
        );

        template.deps = Box::new([]);
        template.nested = Box::new([NestedSlot::new(LocalEntityId::new(0), guid(2))]);
        assert_eq!(
            template.validate(),
            Err(TemplateError::MissingNestedDependency(guid(2)))
        );

        template.deps = Box::new([guid(1)]);
        template.nested = Box::new([]);
        assert_eq!(template.validate(), Err(TemplateError::SelfDependency));

        template.nested = Box::new([NestedSlot::new(LocalEntityId::new(0), guid(1))]);
        assert_eq!(template.validate(), Err(TemplateError::SelfNestedTemplate));
    }

    #[test]
    fn fallible_constructors_validate_before_returning() {
        assert_eq!(
            SceneTemplate::try_empty(StableGuid::NIL),
            Err(TemplateError::NilGuid)
        );
        assert!(SceneTemplate::try_empty(guid(9)).is_ok());
    }

    #[test]
    fn carries_residency_tiers() {
        let template = SceneTemplate::new(guid(1), plan_of(3))
            .with_tier(LocalEntityId::new(2), ResidencyTier::Proxy);
        assert_eq!(
            template.tiers.tier_of(LocalEntityId::new(2)),
            ResidencyTier::Proxy
        );
        assert_eq!(
            template.tiers.tier_of(LocalEntityId::new(0)),
            ResidencyTier::Interactive
        );
    }
}
