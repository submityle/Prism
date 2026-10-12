//! The live instance anchor component (design §5.2, §11).
//!
//! Instantiating a template produces an entity subtree whose **root** carries a
//! [`SceneInstance`]. The component is the one place a live instance records its
//! origin: the template identity, the [`RemapTable`] that keeps the template's
//! internal references pointing inside this instance, the optional override set
//! (COW — `None` costs nothing), the optional cross-session [`PersistentId`], and
//! the current [`ResidencyTier`].

use alloc::boxed::Box;

use prism_asset::StableGuid;
use prism_ecs::component::Component;
use prism_ecs::entity::Entity;

use crate::identity::{LocalEntityId, PersistentId};
use crate::overrides::OverrideSet;
use crate::remap::RemapTable;

/// A scene content residency tier (design §11.1).
///
/// The same scene may mix tiers. A **live** [`SceneInstance`] is
/// [`Interactive`](Self::Interactive) or [`Streamed`](Self::Streamed);
/// [`Proxy`](Self::Proxy) describes content that is *not* instantiated in the
/// ECS (the flat `SoA` proxy stream), named here so residency plans and the
/// materialize/dematerialize transitions share one vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResidencyTier {
    /// Tier A — fully instantiated as live entities (interactive objects).
    Interactive,
    /// Tier B — a sub-scene block instantiated only while in range.
    Streamed,
    /// Tier C — flat proxy stream contributing GPU-resident instances; not ECS.
    Proxy,
}

/// Root anchor of a live scene instance (design §5.2).
#[derive(Clone, Debug, PartialEq)]
pub struct SceneInstance {
    /// The source template's identity (design §5.2 soft handle, by GUID).
    pub template: StableGuid,
    /// `LocalEntityId -> Entity` map for *this* instance (design §6.1).
    pub remap: RemapTable,
    /// This instance's override set; `None` until a field is first overridden.
    pub overrides: Option<Box<OverrideSet>>,
    /// Cross-session / network locator (design §6.2, §15).
    pub persistent: Option<PersistentId>,
    /// The instance's current residency tier.
    pub tier: ResidencyTier,
}

impl SceneInstance {
    /// Builds an instance anchor from a template identity, its remap table and a
    /// residency tier.
    #[must_use]
    pub fn new(template: StableGuid, remap: RemapTable, tier: ResidencyTier) -> Self {
        Self {
            template,
            remap,
            overrides: None,
            persistent: None,
            tier,
        }
    }

    /// Attaches a cross-session [`PersistentId`].
    #[must_use]
    pub fn with_persistent(mut self, persistent: PersistentId) -> Self {
        self.persistent = Some(persistent);
        self
    }

    /// Resolves a template-local entity id to this instance's live entity.
    #[must_use]
    pub fn resolve(&self, local: LocalEntityId) -> Option<Entity> {
        self.remap.get(local)
    }

    /// The override set, if the instance has been customized.
    #[must_use]
    pub fn overrides(&self) -> Option<&OverrideSet> {
        self.overrides.as_deref()
    }

    /// Whether the instance carries any override.
    #[must_use]
    pub fn has_overrides(&self) -> bool {
        self.overrides.as_deref().is_some_and(|set| !set.is_empty())
    }

    /// The override set, allocating an empty one on first use (COW).
    pub fn overrides_mut(&mut self) -> &mut OverrideSet {
        self.overrides.get_or_insert_with(Box::default)
    }
}

impl Component for SceneInstance {}

#[cfg(test)]
mod tests {
    use super::{ResidencyTier, SceneInstance};
    use crate::identity::{LocalEntityId, PersistentId};
    use crate::overrides::{ComponentTypeId, OverrideValue, PropertyOverride};
    use crate::path::FieldPath;
    use crate::remap::RemapTable;
    use prism_asset::StableGuid;
    use prism_ecs::entity::Entity;

    fn entity(index: u32, generation: u32) -> Entity {
        Entity::from_bits((u64::from(generation) << 32) | u64::from(index)).unwrap()
    }

    fn instance() -> SceneInstance {
        let mut remap = RemapTable::new();
        remap.push_base(entity(0, 1));
        remap.push_base(entity(1, 1));
        SceneInstance::new(
            StableGuid::from_u128(0x1234),
            remap,
            ResidencyTier::Interactive,
        )
    }

    #[test]
    fn resolves_locals_through_the_remap() {
        let inst = instance();
        assert_eq!(inst.resolve(LocalEntityId::new(1)), Some(entity(1, 1)));
        assert_eq!(inst.resolve(LocalEntityId::new(9)), None);
    }

    #[test]
    fn overrides_are_lazily_allocated() {
        let mut inst = instance();
        assert!(!inst.has_overrides());
        assert!(inst.overrides().is_none());

        inst.overrides_mut().set_property(PropertyOverride::new(
            LocalEntityId::new(0),
            ComponentTypeId::from_raw(1),
            FieldPath::parse("v").unwrap(),
            OverrideValue::I64(5),
        ));
        assert!(inst.has_overrides());
        assert_eq!(inst.overrides().unwrap().property_count(), 1);
    }

    #[test]
    fn carries_persistent_id() {
        let inst = instance().with_persistent(PersistentId::from_u128(0xabc));
        assert_eq!(inst.persistent, Some(PersistentId::from_u128(0xabc)));
        assert_eq!(inst.tier, ResidencyTier::Interactive);
    }
}
