//! Pure-data persistence for scene-instance overrides.
//!
//! This module deliberately does not perform file I/O and does not touch an
//! ECS [`World`].  It turns the runtime-local [`OverrideSet`] into an owned,
//! deterministic snapshot that a world/save system can encode, then lets the
//! reload planner resolve that snapshot against a newer template.

use alloc::vec::Vec;

use prism_asset::StableGuid;

use crate::identity::{LocalEntityId, PersistentId};
use crate::instance::SceneInstance;
use crate::overrides::{ComponentTypeId, OverrideSet, PropertyOverride, StructuralOp};
use crate::path::FieldPath;
use crate::template::SceneTemplate;

/// The stable identity of one template entity for persistence and reload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntityBinding {
    /// The local id used by the template plan.
    pub local: LocalEntityId,
    /// The cross-session id used by a save or reload operation.
    pub persistent: PersistentId,
}

impl EntityBinding {
    /// Creates a local-to-persistent binding.
    #[must_use]
    pub const fn new(local: LocalEntityId, persistent: PersistentId) -> Self {
        Self { local, persistent }
    }
}

/// A deterministic bidirectional index for template entity identities.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PersistentEntityMap {
    /// Local-id order supports deterministic serialization and local lookup.
    by_local: Vec<EntityBinding>,
    /// Persistent-id order makes reverse reload lookup logarithmic as well.
    by_persistent: Vec<EntityBinding>,
}

impl PersistentEntityMap {
    /// Builds an identity map, rejecting duplicate local or persistent keys.
    pub fn new(mut bindings: Vec<EntityBinding>) -> Result<Self, PersistenceError> {
        for binding in &bindings {
            if binding.persistent.is_nil() {
                return Err(PersistenceError::NilPersistentId {
                    local: binding.local,
                });
            }
        }
        bindings.sort_unstable_by_key(|binding| binding.local);
        for pair in bindings.windows(2) {
            if pair[0].local == pair[1].local {
                return Err(PersistenceError::DuplicateLocalId {
                    local: pair[0].local,
                });
            }
        }
        let mut by_persistent = bindings.clone();
        by_persistent.sort_unstable_by_key(|binding| binding.persistent);
        for pair in by_persistent.windows(2) {
            if pair[0].persistent == pair[1].persistent {
                return Err(PersistenceError::DuplicatePersistentId {
                    persistent: pair[0].persistent,
                });
            }
        }
        Ok(Self {
            by_local: bindings,
            by_persistent,
        })
    }

    /// Builds the conventional root-template identity map.
    ///
    /// Nested templates or authored stable keys should provide an explicit
    /// map with [`Self::new`] so their GUID chain is represented exactly.
    #[must_use]
    pub fn for_template(template: &SceneTemplate) -> Self {
        let bindings: Vec<EntityBinding> = (0..template.entity_count())
            .map(|index| {
                let local = LocalEntityId::new(index);
                EntityBinding::new(local, PersistentId::derive(&[template.guid], &[local]))
            })
            .collect();
        let mut by_persistent = bindings.clone();
        by_persistent.sort_unstable_by_key(|binding| binding.persistent);
        Self {
            by_local: bindings,
            by_persistent,
        }
    }

    /// Returns the binding for a local id.
    #[must_use]
    pub fn persistent_for_local(&self, local: LocalEntityId) -> Option<PersistentId> {
        self.by_local
            .binary_search_by_key(&local, |binding| binding.local)
            .ok()
            .map(|index| self.by_local[index].persistent)
    }

    /// Returns the local id for a persistent id.
    #[must_use]
    pub fn local_for_persistent(&self, persistent: PersistentId) -> Option<LocalEntityId> {
        self.by_persistent
            .binary_search_by_key(&persistent, |binding| binding.persistent)
            .ok()
            .map(|index| self.by_persistent[index].local)
    }

    /// Returns all bindings in local-id order.
    #[must_use]
    pub fn bindings(&self) -> &[EntityBinding] {
        &self.by_local
    }
}

/// A property override in persistence form.
#[derive(Clone, Debug, PartialEq)]
pub struct PersistedPropertyOverride {
    /// The local id at the time the snapshot was captured.
    pub local: LocalEntityId,
    /// The stable id, when the source identity map contained this entity.
    pub persistent: Option<PersistentId>,
    /// The overridden component.
    pub component: ComponentTypeId,
    /// The overridden field path.
    pub path: FieldPath,
    /// The serialized value.
    pub value: crate::OverrideValue,
}

impl PersistedPropertyOverride {
    /// Converts this record back to a runtime-local property override.
    #[must_use]
    pub fn with_local(&self, local: LocalEntityId) -> PropertyOverride {
        PropertyOverride::new(local, self.component, self.path.clone(), self.value.clone())
    }
}

/// An owned, ECS-free scene-instance persistence record.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneInstancePersistence {
    /// The template from which the snapshot was captured.
    pub template: StableGuid,
    /// Optional stable id of the instance root.
    pub instance: Option<PersistentId>,
    /// Sparse property overrides, in their deterministic runtime order.
    pub properties: Vec<PersistedPropertyOverride>,
    /// Structural operations, in replay order.
    pub structural: Vec<StructuralOp>,
}

impl SceneInstancePersistence {
    /// Captures a runtime override set without performing serialization.
    pub fn capture(
        template: &SceneTemplate,
        instance: Option<PersistentId>,
        overrides: &OverrideSet,
        identities: &PersistentEntityMap,
    ) -> PersistenceCapture {
        let mut diagnostics = Vec::new();
        let mut properties = Vec::with_capacity(overrides.property_count());
        for property in overrides.properties() {
            let persistent = identities.persistent_for_local(property.target);
            if persistent.is_none() && !property.target.is_reserved() {
                diagnostics.push(PersistenceDiagnostic::MissingEntityBinding {
                    local: property.target,
                    component: property.component,
                    path: property.path.clone(),
                });
            }
            properties.push(PersistedPropertyOverride {
                local: property.target,
                persistent,
                component: property.component,
                path: property.path.clone(),
                value: property.value.clone(),
            });
        }
        PersistenceCapture {
            snapshot: Self {
                template: template.guid,
                instance,
                properties,
                structural: overrides.structural().to_vec(),
            },
            diagnostics,
        }
    }

    /// Captures the override set carried by a [`SceneInstance`].
    pub fn capture_instance(
        template: &SceneTemplate,
        instance: &SceneInstance,
        identities: &PersistentEntityMap,
    ) -> PersistenceCapture {
        let empty = OverrideSet::new();
        Self::capture(
            template,
            instance.persistent,
            instance.overrides().unwrap_or(&empty),
            identities,
        )
    }

    /// Whether this snapshot contains no property or structural changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty() && self.structural.is_empty()
    }
}

/// A capture result keeps diagnostics alongside the snapshot so a missing
/// identity cannot silently turn into data loss.
#[derive(Clone, Debug, PartialEq)]
pub struct PersistenceCapture {
    /// The owned snapshot suitable for an external encoder.
    pub snapshot: SceneInstancePersistence,
    /// Non-fatal capture diagnostics.
    pub diagnostics: Vec<PersistenceDiagnostic>,
}

/// A capture-time persistence diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub enum PersistenceDiagnostic {
    /// A base-segment property had no persistent identity; its local id is
    /// retained so reload can still use the documented local-id fallback.
    MissingEntityBinding {
        /// The unbound entity.
        local: LocalEntityId,
        /// The affected component.
        component: ComponentTypeId,
        /// The affected path.
        path: FieldPath,
    },
}

/// Errors constructing a persistent identity map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistenceError {
    /// A binding used the unset persistent id.
    NilPersistentId {
        /// The local id attached to the invalid binding.
        local: LocalEntityId,
    },
    /// Two bindings used the same local id.
    DuplicateLocalId {
        /// The duplicated local id.
        local: LocalEntityId,
    },
    /// Two bindings used the same persistent id.
    DuplicatePersistentId {
        /// The duplicated persistent id.
        persistent: PersistentId,
    },
}

/// Backwards-friendly name for callers that think in terms of a sparse set.
pub type PersistentOverrideSet = SceneInstancePersistence;

#[cfg(test)]
mod tests {
    use super::{EntityBinding, PersistentEntityMap, SceneInstancePersistence};
    use crate::identity::{LocalEntityId, PersistentId};
    use crate::overrides::{ComponentTypeId, OverrideSet, OverrideValue, PropertyOverride};
    use crate::path::FieldPath;
    use crate::plan::InstantiationPlan;
    use crate::template::SceneTemplate;
    use alloc::boxed::Box;
    use prism_asset::StableGuid;

    fn template() -> SceneTemplate {
        SceneTemplate::new(
            StableGuid::from_u128(7),
            InstantiationPlan {
                entity_count: 1,
                groups: Box::new([crate::ArchetypeGroup::new(
                    crate::ArchetypeSignature::default(),
                    Box::new([LocalEntityId::new(0)]),
                    Box::new([]),
                )]),
                entity_valued_fields: Box::new([]),
                hierarchy: Box::new([]),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            },
        )
    }

    #[test]
    fn capture_preserves_local_and_persistent_keys() {
        let template = template();
        let identities = PersistentEntityMap::for_template(&template);
        let mut overrides = OverrideSet::new();
        overrides.set_property(PropertyOverride::new(
            LocalEntityId::new(0),
            ComponentTypeId::from_raw(3),
            FieldPath::parse("value").unwrap(),
            OverrideValue::U64(9),
        ));

        let captured = SceneInstancePersistence::capture(
            &template,
            Some(PersistentId::from_u128(99)),
            &overrides,
            &identities,
        );
        assert!(captured.diagnostics.is_empty());
        assert_eq!(captured.snapshot.properties[0].local, LocalEntityId::new(0));
        assert_eq!(
            captured.snapshot.properties[0].persistent,
            identities.persistent_for_local(LocalEntityId::new(0))
        );
    }

    #[test]
    fn explicit_identity_map_supports_local_reassignment() {
        let map = PersistentEntityMap::new(vec![EntityBinding::new(
            LocalEntityId::new(4),
            PersistentId::from_u128(8),
        )])
        .unwrap();
        assert_eq!(
            map.local_for_persistent(PersistentId::from_u128(8)),
            Some(LocalEntityId::new(4))
        );
        assert_eq!(
            map.persistent_for_local(LocalEntityId::new(4)),
            Some(PersistentId::from_u128(8))
        );
        assert_eq!(
            map.bindings(),
            &[EntityBinding::new(
                LocalEntityId::new(4),
                PersistentId::from_u128(8)
            )]
        );
    }

    #[test]
    fn reverse_identity_index_is_sorted_and_bijective() {
        let map = PersistentEntityMap::new(vec![
            EntityBinding::new(LocalEntityId::new(17), PersistentId::from_u128(2)),
            EntityBinding::new(LocalEntityId::new(3), PersistentId::from_u128(91)),
            EntityBinding::new(LocalEntityId::new(9), PersistentId::from_u128(7)),
        ])
        .unwrap();
        assert_eq!(
            map.bindings()
                .iter()
                .map(|entry| entry.local)
                .collect::<Vec<_>>(),
            vec![
                LocalEntityId::new(3),
                LocalEntityId::new(9),
                LocalEntityId::new(17)
            ]
        );
        for binding in map.bindings() {
            assert_eq!(
                map.persistent_for_local(binding.local),
                Some(binding.persistent)
            );
            assert_eq!(
                map.local_for_persistent(binding.persistent),
                Some(binding.local)
            );
        }
    }
}
