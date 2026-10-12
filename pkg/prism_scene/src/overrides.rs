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

use alloc::string::String;
use alloc::vec::Vec;

use crate::identity::LocalEntityId;
use crate::path::FieldPath;

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

    /// Evicts property overrides whose target was removed by a `RemoveEntity`
    /// structural op in this set, returning the evicted `(target, component,
    /// path)` keys (design §7.3 `OrphanOverride`).
    pub fn drop_orphans(&mut self) -> Vec<(LocalEntityId, ComponentTypeId, FieldPath)> {
        let removed = self.removed_entities();
        if removed.is_empty() {
            return Vec::new();
        }
        let mut orphans = Vec::new();
        let mut kept = Vec::with_capacity(self.properties.len());
        for ov in self.properties.drain(..) {
            if removed.contains(&ov.target) {
                orphans.push((ov.target, ov.component, ov.path));
            } else {
                kept.push(ov);
            }
        }
        self.properties = kept;
        orphans
    }
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
}
