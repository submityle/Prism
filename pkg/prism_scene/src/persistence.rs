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
use crate::overrides::{
    ComponentTypeId, OverrideError, OverrideSet, PropertyOverride, StructuralOp,
};
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

    /// Rebuilds the sparse override set for the current template. Persisted
    /// property targets prefer stable entity identity and therefore survive a
    /// base-local reordering. Local-id-only properties and structural targets
    /// are accepted only when the template identity is unchanged. Structural
    /// payloads remain opaque and are copied without schema interpretation.
    pub fn restore_overrides(
        &self,
        current_template: &SceneTemplate,
        identities: &PersistentEntityMap,
    ) -> Result<OverrideSet, PersistenceRestoreError> {
        let same_template = self.template == current_template.guid;
        let mut overrides = OverrideSet::new();

        for property in &self.properties {
            let local = if let Some(persistent) = property.persistent {
                identities
                    .local_for_persistent(persistent)
                    .ok_or(PersistenceRestoreError::MissingPersistentEntity { persistent })?
            } else if same_template {
                property.local
            } else {
                return Err(PersistenceRestoreError::LocalIdentityTemplateMismatch {
                    snapshot: self.template,
                    current: current_template.guid,
                    local: property.local,
                });
            };
            if overrides.set_property(property.with_local(local)).is_some() {
                return Err(PersistenceRestoreError::DuplicateProperty {
                    target: local,
                    component: property.component,
                    path: property.path.clone(),
                });
            }
        }

        if !same_template && !self.structural.is_empty() {
            return Err(PersistenceRestoreError::StructuralTemplateMismatch {
                snapshot: self.template,
                current: current_template.guid,
            });
        }
        for operation in &self.structural {
            overrides.push_structural(operation.clone());
        }
        overrides
            .validate_for_plan(&current_template.plan)
            .map_err(PersistenceRestoreError::InvalidOverrides)?;
        Ok(overrides)
    }

    /// Whether this snapshot contains no property or structural changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty() && self.structural.is_empty()
    }
}

/// A persisted scene snapshot cannot be safely applied to the requested
/// template without additional stable structural identities or migration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistenceRestoreError {
    /// A stable entity ID in the snapshot is absent from the current template.
    MissingPersistentEntity {
        /// The unresolved stable identity.
        persistent: PersistentId,
    },
    /// A local-only property cannot be mapped across template identities.
    LocalIdentityTemplateMismatch {
        /// Template identity stored with the snapshot.
        snapshot: StableGuid,
        /// Template identity requested for restoration.
        current: StableGuid,
        /// Local entity identity that has no persistent counterpart.
        local: LocalEntityId,
    },
    /// Structural operations contain template-local targets but no stable IDs.
    StructuralTemplateMismatch {
        /// Template identity stored with the snapshot.
        snapshot: StableGuid,
        /// Template identity requested for restoration.
        current: StableGuid,
    },
    /// Two persisted properties resolve to the same runtime property key.
    DuplicateProperty {
        /// Resolved runtime-local entity identity.
        target: LocalEntityId,
        /// Property component identity.
        component: ComponentTypeId,
        /// Property field path.
        path: FieldPath,
    },
    /// The restored delta is structurally invalid for the current template.
    InvalidOverrides(OverrideError),
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
    use super::{
        EntityBinding, PersistedPropertyOverride, PersistenceRestoreError, PersistentEntityMap,
        SceneInstancePersistence,
    };
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

    #[test]
    fn restore_rebinds_properties_by_persistent_identity_after_local_reorder() {
        let source_guid = StableGuid::from_u128(7);
        let current_guid = StableGuid::from_u128(8);
        let persistent = PersistentId::from_u128(77);
        let snapshot = SceneInstancePersistence {
            template: source_guid,
            instance: Some(PersistentId::from_u128(99)),
            properties: vec![PersistedPropertyOverride {
                local: LocalEntityId::new(1),
                persistent: Some(persistent),
                component: ComponentTypeId::from_raw(3),
                path: FieldPath::parse("value").unwrap(),
                value: OverrideValue::U64(42),
            }],
            structural: Vec::new(),
        };
        let identities =
            PersistentEntityMap::new(vec![EntityBinding::new(LocalEntityId::new(9), persistent)])
                .unwrap();

        let current_template = SceneTemplate::new(
            current_guid,
            InstantiationPlan {
                entity_count: 10,
                groups: Box::new([crate::ArchetypeGroup::new(
                    crate::ArchetypeSignature::default(),
                    (0..10)
                        .map(LocalEntityId::new)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                    Box::new([]),
                )]),
                entity_valued_fields: Box::new([]),
                hierarchy: (1..10)
                    .map(|local| (LocalEntityId::new(local), LocalEntityId::new(local - 1)))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            },
        );
        let restored = snapshot
            .restore_overrides(&current_template, &identities)
            .unwrap();
        assert_eq!(restored.property_count(), 1);
        assert_eq!(restored.properties()[0].target, LocalEntityId::new(9));
        assert_eq!(restored.properties()[0].value, OverrideValue::U64(42));
    }

    #[test]
    fn restore_fails_closed_for_missing_identity_or_unmigrated_structure() {
        let source_guid = StableGuid::from_u128(7);
        let current_guid = StableGuid::from_u128(8);
        let snapshot = SceneInstancePersistence {
            template: source_guid,
            instance: None,
            properties: vec![PersistedPropertyOverride {
                local: LocalEntityId::new(0),
                persistent: Some(PersistentId::from_u128(77)),
                component: ComponentTypeId::from_raw(3),
                path: FieldPath::parse("value").unwrap(),
                value: OverrideValue::U64(42),
            }],
            structural: Vec::new(),
        };
        assert_eq!(
            snapshot.restore_overrides(
                &template_with_guid(current_guid),
                &PersistentEntityMap::default()
            ),
            Err(PersistenceRestoreError::MissingPersistentEntity {
                persistent: PersistentId::from_u128(77)
            })
        );

        let structural = SceneInstancePersistence {
            template: source_guid,
            instance: None,
            properties: Vec::new(),
            structural: vec![crate::StructuralOp::RemoveEntity {
                local: LocalEntityId::new(2),
            }],
        };
        assert_eq!(
            structural.restore_overrides(
                &template_with_guid(current_guid),
                &PersistentEntityMap::default()
            ),
            Err(PersistenceRestoreError::StructuralTemplateMismatch {
                snapshot: source_guid,
                current: current_guid
            })
        );
    }

    #[test]
    fn restore_keeps_structural_replay_order_for_same_template() {
        let template = StableGuid::from_u128(7);
        let snapshot = SceneInstancePersistence {
            template,
            instance: None,
            properties: Vec::new(),
            structural: vec![crate::StructuralOp::AddEntity {
                local: LocalEntityId::reserved(0),
                parent: LocalEntityId::new(0),
                archetype: crate::ArchetypeSignature::default(),
                data: Vec::new(),
            }],
        };
        let restored = snapshot
            .restore_overrides(
                &template_with_guid(template),
                &PersistentEntityMap::default(),
            )
            .unwrap();
        assert_eq!(restored.structural(), snapshot.structural);
    }

    fn template_with_guid(guid: StableGuid) -> SceneTemplate {
        SceneTemplate::new(
            guid,
            InstantiationPlan {
                entity_count: 3,
                groups: Box::new([crate::ArchetypeGroup::new(
                    crate::ArchetypeSignature::default(),
                    Box::new([
                        LocalEntityId::new(0),
                        LocalEntityId::new(1),
                        LocalEntityId::new(2),
                    ]),
                    Box::new([]),
                )]),
                entity_valued_fields: Box::new([]),
                hierarchy: vec![
                    (LocalEntityId::new(1), LocalEntityId::new(0)),
                    (LocalEntityId::new(2), LocalEntityId::new(1)),
                ]
                .into_boxed_slice(),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            },
        )
    }
}
