//! Deterministic remapping of template-local entity identities.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::{
    LocalEntityId, NameTable, OverrideSet, ResidencyPlan, SceneInstancePersistence, SceneTemplate,
    StructuralOp, TemplateError,
};

/// Failure while validating or applying a local-entity identity migration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalEntityMigrationError {
    /// The source map omitted a base identity.
    MissingSource(LocalEntityId),
    /// A source identity was declared more than once.
    DuplicateSource(LocalEntityId),
    /// A destination is reserved or outside the dense base range.
    TargetOutOfRange {
        /// Source identity being remapped.
        source: LocalEntityId,
        /// Destination identity requested by the mapping.
        target: LocalEntityId,
    },
    /// Two source identities map to the same destination.
    DuplicateTarget(LocalEntityId),
    /// The template root identity must remain local id zero.
    RootIdentityChanged {
        /// Requested destination identity.
        target: LocalEntityId,
    },
    /// A template metadata reference was not in the complete map.
    UnknownReference(LocalEntityId),
    /// An entity-valued field slot is invalid for its column payload.
    InvalidEntityField(crate::FieldSlot),
    /// Multiple field slots address the same entity row and byte offset.
    DuplicateEntityField(crate::FieldSlot),
    /// A column contains an entity reference absent from the complete map.
    UnknownEntityReference {
        /// Field slot whose row contained the invalid reference.
        field: crate::FieldSlot,
        /// Referenced identity absent from the migration map.
        target: LocalEntityId,
    },
    /// Source and migration entity counts differ.
    EntityCountMismatch {
        /// Entity count in the source template.
        source: u32,
        /// Entity count covered by the mapping.
        mapping: u32,
    },
    /// The migrated template violates a template invariant.
    InvalidTemplate(TemplateError),
}

/// A complete bijection over the base entity range of one scene template.
///
/// Construction is O(E log E); each reference rewrite is O(log E). Reserved
/// structural-override identities are stable and are not included in the map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalEntityMigration {
    entity_count: u32,
    by_source: BTreeMap<LocalEntityId, LocalEntityId>,
}

impl LocalEntityMigration {
    /// Validates a complete bijection over base identities 0 through E-1.
    pub fn new(
        entity_count: u32,
        mappings: impl IntoIterator<Item = (LocalEntityId, LocalEntityId)>,
    ) -> Result<Self, LocalEntityMigrationError> {
        let mut by_source = BTreeMap::new();
        let mut targets = BTreeSet::new();
        for (source, target) in mappings {
            if source.is_reserved() || source.index() >= entity_count {
                return Err(LocalEntityMigrationError::MissingSource(source));
            }
            if target.is_reserved() || target.index() >= entity_count {
                return Err(LocalEntityMigrationError::TargetOutOfRange { source, target });
            }
            if by_source.insert(source, target).is_some() {
                return Err(LocalEntityMigrationError::DuplicateSource(source));
            }
            if !targets.insert(target) {
                return Err(LocalEntityMigrationError::DuplicateTarget(target));
            }
            if (source == LocalEntityId::new(0)) != (target == LocalEntityId::new(0)) {
                return Err(LocalEntityMigrationError::RootIdentityChanged { target });
            }
        }
        for index in 0..entity_count {
            let source = LocalEntityId::new(index);
            if !by_source.contains_key(&source) {
                return Err(LocalEntityMigrationError::MissingSource(source));
            }
        }
        Ok(Self {
            entity_count,
            by_source,
        })
    }

    /// Builds an identity mapping for an unchanged local numbering.
    #[must_use]
    pub fn identity(entity_count: u32) -> Self {
        Self {
            entity_count,
            by_source: (0..entity_count)
                .map(|i| {
                    let local = LocalEntityId::new(i);
                    (local, local)
                })
                .collect(),
        }
    }

    /// Maps one base identity; reserved structural identities pass through.
    #[must_use]
    pub fn map(&self, source: LocalEntityId) -> Option<LocalEntityId> {
        if source.is_reserved() {
            Some(source)
        } else {
            self.by_source.get(&source).copied()
        }
    }

    /// Rewrites all LocalEntityId-bearing template metadata and validates it.
    ///
    /// O(E log E + M log E), where M is the number of names, tiers and nested
    /// roots. Component payload references require registered field offsets and
    /// are deliberately handled by the schema owner's typed migration callback.
    pub fn migrate_template(
        &self,
        source: &SceneTemplate,
    ) -> Result<SceneTemplate, LocalEntityMigrationError> {
        if source.entity_count() != self.entity_count {
            return Err(LocalEntityMigrationError::EntityCountMismatch {
                source: source.entity_count(),
                mapping: self.entity_count,
            });
        }
        let mut migrated = source.clone();
        migrated.plan.groups = source
            .plan
            .groups
            .iter()
            .enumerate()
            .map(|(group_index, group)| {
                let locals = group
                    .locals
                    .iter()
                    .map(|local| {
                        self.map(*local)
                            .ok_or(LocalEntityMigrationError::UnknownReference(*local))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let row_by_local: BTreeMap<_, _> = group
                    .locals
                    .iter()
                    .enumerate()
                    .map(|(row, local)| (*local, row))
                    .collect();
                let mut columns = group.columns.to_vec();
                for (column_index, column) in columns.iter_mut().enumerate() {
                    let fields: Vec<_> = source
                        .plan
                        .entity_valued_fields
                        .iter()
                        .filter(|field| {
                            field.group as usize == group_index
                                && field.column as usize == column_index
                        })
                        .collect();
                    if fields.is_empty() {
                        continue;
                    }
                    let rows = group.locals.len();
                    if rows == 0 || column.blob.len() % rows != 0 {
                        return Err(LocalEntityMigrationError::InvalidEntityField(*fields[0]));
                    }
                    let row_bytes = column.blob.len() / rows;
                    let mut bytes = column.blob.to_vec();
                    let mut occupied_slots = BTreeSet::new();
                    for field in fields {
                        let field_offset = field.byte_offset as usize;
                        let Some(field_end) = field_offset.checked_add(4) else {
                            return Err(LocalEntityMigrationError::InvalidEntityField(*field));
                        };
                        if field_end > row_bytes {
                            return Err(LocalEntityMigrationError::InvalidEntityField(*field));
                        }
                        let row = row_by_local
                            .get(&field.local)
                            .copied()
                            .ok_or(LocalEntityMigrationError::InvalidEntityField(*field))?;
                        if !occupied_slots.insert((row, field_offset)) {
                            return Err(LocalEntityMigrationError::DuplicateEntityField(*field));
                        }
                        let offset = row * row_bytes + field_offset;
                        let raw = u32::from_le_bytes(
                            bytes[offset..offset + 4]
                                .try_into()
                                .expect("four-byte field range was validated"),
                        );
                        let old_target = LocalEntityId::new(raw);
                        let new_target = self.map(old_target).ok_or(
                            LocalEntityMigrationError::UnknownEntityReference {
                                field: *field,
                                target: old_target,
                            },
                        )?;
                        bytes[offset..offset + 4]
                            .copy_from_slice(&new_target.index().to_le_bytes());
                    }
                    column.blob = alloc::sync::Arc::from(bytes);
                }
                Ok(crate::ArchetypeGroup::new(
                    group.signature.clone(),
                    locals.into_boxed_slice(),
                    columns.into_boxed_slice(),
                ))
            })
            .collect::<Result<Vec<_>, LocalEntityMigrationError>>()?
            .into_boxed_slice();
        migrated.plan.entity_valued_fields = source
            .plan
            .entity_valued_fields
            .iter()
            .map(|field| {
                Ok(crate::FieldSlot {
                    local: self
                        .map(field.local)
                        .ok_or(LocalEntityMigrationError::UnknownReference(field.local))?,
                    ..*field
                })
            })
            .collect::<Result<Vec<_>, LocalEntityMigrationError>>()?
            .into_boxed_slice();
        migrated.plan.hierarchy = source
            .plan
            .hierarchy
            .iter()
            .map(|&(child, parent)| {
                Ok((
                    self.map(child)
                        .ok_or(LocalEntityMigrationError::UnknownReference(child))?,
                    self.map(parent)
                        .ok_or(LocalEntityMigrationError::UnknownReference(parent))?,
                ))
            })
            .collect::<Result<Vec<_>, LocalEntityMigrationError>>()?
            .into_boxed_slice();
        let mut names = NameTable::new();
        for (name, local) in source.names.iter() {
            names.insert(
                name,
                self.map(local)
                    .ok_or(LocalEntityMigrationError::UnknownReference(local))?,
            );
        }
        migrated.names = names;
        let mut tiers = ResidencyPlan::new();
        for (local, tier) in source.tiers.iter() {
            tiers.set(
                self.map(local)
                    .ok_or(LocalEntityMigrationError::UnknownReference(local))?,
                tier,
            );
        }
        migrated.tiers = tiers;
        for slot in &mut migrated.nested {
            slot.root = self
                .map(slot.root)
                .ok_or(LocalEntityMigrationError::UnknownReference(slot.root))?;
        }
        migrated
            .validate()
            .map_err(LocalEntityMigrationError::InvalidTemplate)?;
        Ok(migrated)
    }

    /// Rewrites persisted local references while keeping stable IDs and values.
    pub fn migrate_persistence(
        &self,
        source: &SceneInstancePersistence,
    ) -> Result<SceneInstancePersistence, LocalEntityMigrationError> {
        let mut migrated = source.clone();
        for property in &mut migrated.properties {
            property.local = self
                .map(property.local)
                .ok_or(LocalEntityMigrationError::UnknownReference(property.local))?;
        }
        for operation in &mut migrated.structural {
            map_structural(operation, self)?;
        }
        Ok(migrated)
    }

    /// Rewrites persisted local, component and field identities explicitly.
    pub fn migrate_persistence_with_fields(
        &self,
        source: &SceneInstancePersistence,
        fields: &FieldIdentityMigration,
    ) -> Result<SceneInstancePersistence, FieldIdentityMigrationError> {
        let mut migrated = source.clone();
        for property in &mut migrated.properties {
            property.local = self
                .map(property.local)
                .ok_or(FieldIdentityMigrationError::UnknownEntity(property.local))?;
            let (component, path) =
                fields
                    .map(property.component, &property.path)
                    .ok_or_else(|| FieldIdentityMigrationError::MissingField {
                        component: property.component,
                        path: property.path.clone(),
                    })?;
            property.component = *component;
            property.path = path.clone();
        }
        for operation in &mut migrated.structural {
            map_structural(operation, self).map_err(|error| match error {
                LocalEntityMigrationError::UnknownReference(local) => {
                    FieldIdentityMigrationError::UnknownEntity(local)
                }
                _ => unreachable!("structural mapping only returns unknown-entity errors"),
            })?;
        }
        Ok(migrated)
    }

    /// Rewrites sparse runtime overrides without changing payload or order.
    pub fn migrate_overrides(
        &self,
        source: &OverrideSet,
    ) -> Result<OverrideSet, LocalEntityMigrationError> {
        let mut migrated = OverrideSet::new();
        for property in source.properties() {
            let target = self
                .map(property.target)
                .ok_or(LocalEntityMigrationError::UnknownReference(property.target))?;
            migrated.set_property(crate::PropertyOverride::new(
                target,
                property.component,
                property.path.clone(),
                property.value.clone(),
            ));
        }
        for operation in source.structural() {
            let mut operation = operation.clone();
            map_structural(&mut operation, self)?;
            migrated.push_structural(operation);
        }
        Ok(migrated)
    }

    /// Rewrites both local entity targets and schema-owned field identities.
    pub fn migrate_overrides_with_fields(
        &self,
        source: &OverrideSet,
        fields: &FieldIdentityMigration,
    ) -> Result<OverrideSet, FieldIdentityMigrationError> {
        let mut migrated = OverrideSet::new();
        for property in source.properties() {
            migrated.set_property(self.migrate_property(property, fields)?);
        }
        for operation in source.structural() {
            let mut operation = operation.clone();
            map_structural(&mut operation, self).map_err(|error| match error {
                LocalEntityMigrationError::UnknownReference(local) => {
                    FieldIdentityMigrationError::UnknownEntity(local)
                }
                _ => unreachable!("structural mapping only returns unknown-entity errors"),
            })?;
            migrated.push_structural(operation);
        }
        Ok(migrated)
    }

    /// Rewrites component identities and field paths in one property override.
    ///
    /// The schema owner supplies an explicit component and path mapping.
    /// Missing entries fail closed; values remain opaque and unchanged.
    pub fn migrate_property(
        &self,
        source: &crate::PropertyOverride,
        fields: &FieldIdentityMigration,
    ) -> Result<crate::PropertyOverride, FieldIdentityMigrationError> {
        let target = self
            .map(source.target)
            .ok_or(FieldIdentityMigrationError::UnknownEntity(source.target))?;
        let (component, path) = fields.map(source.component, &source.path).ok_or(
            FieldIdentityMigrationError::MissingField {
                component: source.component,
                path: source.path.clone(),
            },
        )?;
        Ok(crate::PropertyOverride::new(
            target,
            *component,
            path.clone(),
            source.value.clone(),
        ))
    }
}

/// Failure while constructing or applying a component-field identity mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldIdentityMigrationError {
    /// A source component/path pair was registered more than once.
    DuplicateSource {
        /// Original component identity.
        component: crate::ComponentTypeId,
        /// Original field path.
        path: crate::FieldPath,
    },
    /// Two source fields map to the same destination component/path.
    DuplicateTarget {
        /// Destination component identity.
        component: crate::ComponentTypeId,
        /// Destination field path.
        path: crate::FieldPath,
    },
    /// No field mapping exists for the requested property.
    MissingField {
        /// Component identity in the source schema.
        component: crate::ComponentTypeId,
        /// Field path in the source schema.
        path: crate::FieldPath,
    },
    /// The property refers to an entity absent from the identity mapping.
    UnknownEntity(LocalEntityId),
}

/// An explicit, deterministic mapping of component-field identities.
///
/// Every supplied source pair and destination pair must be unique. This avoids
/// silently merging two independently authored fields after schema migration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldIdentityMigration {
    by_source: BTreeMap<
        (crate::ComponentTypeId, crate::FieldPath),
        (crate::ComponentTypeId, crate::FieldPath),
    >,
    targets: BTreeSet<(crate::ComponentTypeId, crate::FieldPath)>,
}

impl FieldIdentityMigration {
    /// Builds a unique source-to-destination field mapping.
    pub fn new(
        mappings: impl IntoIterator<
            Item = (
                (crate::ComponentTypeId, crate::FieldPath),
                (crate::ComponentTypeId, crate::FieldPath),
            ),
        >,
    ) -> Result<Self, FieldIdentityMigrationError> {
        let mut migration = Self::default();
        for ((source_component, source_path), (target_component, target_path)) in mappings {
            let source = (source_component, source_path);
            let target = (target_component, target_path);
            if migration.by_source.contains_key(&source) {
                return Err(FieldIdentityMigrationError::DuplicateSource {
                    component: source.0,
                    path: source.1,
                });
            }
            if !migration.targets.insert(target.clone()) {
                return Err(FieldIdentityMigrationError::DuplicateTarget {
                    component: target.0,
                    path: target.1,
                });
            }
            migration.by_source.insert(source, target);
        }
        Ok(migration)
    }

    /// Returns the mapped component/path pair, if explicitly registered.
    #[must_use]
    pub fn map(
        &self,
        component: crate::ComponentTypeId,
        path: &crate::FieldPath,
    ) -> Option<&(crate::ComponentTypeId, crate::FieldPath)> {
        self.by_source.get(&(component, path.clone()))
    }
}

fn map_structural(
    operation: &mut StructuralOp,
    migration: &LocalEntityMigration,
) -> Result<(), LocalEntityMigrationError> {
    let map = |local: &mut LocalEntityId| {
        *local = migration
            .map(*local)
            .ok_or(LocalEntityMigrationError::UnknownReference(*local))?;
        Ok::<(), LocalEntityMigrationError>(())
    };
    match operation {
        StructuralOp::AddEntity { local, parent, .. } => {
            map(local)?;
            map(parent)?;
        }
        StructuralOp::RemoveEntity { local } => map(local)?,
        StructuralOp::AddComponent { target, .. }
        | StructuralOp::RemoveComponent { target, .. } => map(target)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArchetypeGroup, ArchetypeSignature, Column, ComponentTypeId, FieldPath, FieldSlot,
        InstantiationPlan, OverrideValue, PropertyOverride,
    };
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec;
    use prism_asset::StableGuid;

    fn template() -> SceneTemplate {
        let component = ComponentTypeId::from_raw(7);
        let mut blob = Vec::new();
        for target in [1u32, 2, 0] {
            blob.extend_from_slice(&target.to_le_bytes());
            blob.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        }
        let mut template = SceneTemplate::new(
            StableGuid::from_u128(9),
            InstantiationPlan {
                entity_count: 3,
                groups: Box::new([ArchetypeGroup::new(
                    ArchetypeSignature::new(vec![component]),
                    Box::new([
                        LocalEntityId::new(0),
                        LocalEntityId::new(1),
                        LocalEntityId::new(2),
                    ]),
                    Box::new([Column::new(component, Arc::<[u8]>::from(blob))]),
                )]),
                entity_valued_fields: Box::new([FieldSlot {
                    local: LocalEntityId::new(0),
                    group: 0,
                    column: 0,
                    byte_offset: 0,
                }]),
                hierarchy: Box::new([
                    (LocalEntityId::new(1), LocalEntityId::new(0)),
                    (LocalEntityId::new(2), LocalEntityId::new(0)),
                ]),
                reserved_local_base: LocalEntityId::RESERVED_BASE,
            },
        );
        template.names.insert("root", LocalEntityId::new(0));
        template.names.insert("child", LocalEntityId::new(1));
        template.with_tier(LocalEntityId::new(2), crate::ResidencyTier::Proxy)
    }

    #[test]
    fn rejects_incomplete_and_non_bijective_mapping_before_template_work() {
        assert_eq!(
            LocalEntityMigration::new(2, [(LocalEntityId::new(0), LocalEntityId::new(0))]),
            Err(LocalEntityMigrationError::MissingSource(
                LocalEntityId::new(1)
            ))
        );
        assert_eq!(
            LocalEntityMigration::new(
                2,
                [
                    (LocalEntityId::new(0), LocalEntityId::new(0)),
                    (LocalEntityId::new(1), LocalEntityId::new(0)),
                ]
            ),
            Err(LocalEntityMigrationError::DuplicateTarget(
                LocalEntityId::new(0)
            ))
        );
        assert!(matches!(
            LocalEntityMigration::new(
                2,
                [
                    (LocalEntityId::new(0), LocalEntityId::new(1)),
                    (LocalEntityId::new(1), LocalEntityId::new(0)),
                ]
            ),
            Err(LocalEntityMigrationError::RootIdentityChanged { .. })
        ));
    }

    #[test]
    fn migrates_all_template_references_and_registered_entity_field_rows() {
        let migration = LocalEntityMigration::new(
            3,
            [
                (LocalEntityId::new(0), LocalEntityId::new(0)),
                (LocalEntityId::new(1), LocalEntityId::new(2)),
                (LocalEntityId::new(2), LocalEntityId::new(1)),
            ],
        )
        .unwrap();
        let old = template();
        let migrated = migration.migrate_template(&old).unwrap();
        assert_eq!(migrated.names.get("root"), Some(LocalEntityId::new(0)));
        assert_eq!(migrated.names.get("child"), Some(LocalEntityId::new(2)));
        assert_eq!(
            migrated.tiers.tier_of(LocalEntityId::new(1)),
            crate::ResidencyTier::Proxy
        );
        assert_eq!(
            migrated.tiers.tier_of(LocalEntityId::new(2)),
            crate::ResidencyTier::Interactive
        );
        assert_eq!(
            migrated.plan.hierarchy.as_ref(),
            &[
                (LocalEntityId::new(2), LocalEntityId::new(0)),
                (LocalEntityId::new(1), LocalEntityId::new(0))
            ]
        );
        assert_eq!(
            migrated.plan.entity_valued_fields[0].local,
            LocalEntityId::new(0)
        );
        let bytes = &migrated.plan.groups[0].columns[0].blob;
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 0);
        assert_eq!(&bytes[4..8], &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn migrates_persistence_and_structural_ids_but_preserves_reserved_ids_and_values() {
        let migration = LocalEntityMigration::new(
            3,
            [
                (LocalEntityId::new(0), LocalEntityId::new(0)),
                (LocalEntityId::new(1), LocalEntityId::new(2)),
                (LocalEntityId::new(2), LocalEntityId::new(1)),
            ],
        )
        .unwrap();
        let old = template();
        let mut overrides = OverrideSet::new();
        overrides.set_property(PropertyOverride::new(
            LocalEntityId::new(0),
            ComponentTypeId::from_raw(7),
            FieldPath::parse("x").unwrap(),
            OverrideValue::U64(42),
        ));
        overrides.push_structural(StructuralOp::AddEntity {
            local: LocalEntityId::reserved(0),
            parent: LocalEntityId::new(1),
            archetype: ArchetypeSignature::default(),
            data: vec![1, 2],
        });
        overrides.push_structural(StructuralOp::RemoveEntity {
            local: LocalEntityId::reserved(0),
        });
        let migrated = migration.migrate_overrides(&overrides).unwrap();
        assert_eq!(migrated.properties()[0].target, LocalEntityId::new(0));
        assert_eq!(migrated.properties()[0].value, OverrideValue::U64(42));
        assert!(
            matches!(migrated.structural()[0], StructuralOp::AddEntity { local, parent, .. }
            if local == LocalEntityId::reserved(0) && parent == LocalEntityId::new(2))
        );

        let map = crate::PersistentEntityMap::for_template(&old);
        let snapshot = SceneInstancePersistence::capture(&old, None, &overrides, &map).snapshot;
        let migrated_snapshot = migration.migrate_persistence(&snapshot).unwrap();
        assert_eq!(migrated_snapshot.properties[0].local, LocalEntityId::new(0));
        assert_eq!(
            migrated_snapshot.properties[0].persistent,
            snapshot.properties[0].persistent
        );
    }

    #[test]
    fn migrates_component_and_field_identity_without_changing_values() {
        let entities = LocalEntityMigration::identity(1);
        let old_component = ComponentTypeId::from_raw(7);
        let new_component = ComponentTypeId::from_raw(8);
        let old_path = FieldPath::parse("translation.x").unwrap();
        let new_path = FieldPath::parse("position.horizontal").unwrap();
        let fields = FieldIdentityMigration::new([(
            (old_component, old_path.clone()),
            (new_component, new_path.clone()),
        )])
        .unwrap();
        let source = PropertyOverride::new(
            LocalEntityId::new(0),
            old_component,
            old_path,
            OverrideValue::F32(3.5),
        );
        let migrated = entities.migrate_property(&source, &fields).unwrap();
        assert_eq!(migrated.component, new_component);
        assert_eq!(migrated.path, new_path);
        assert_eq!(migrated.value, OverrideValue::F32(3.5));
        assert!(matches!(
            entities.migrate_property(&source, &FieldIdentityMigration::default()),
            Err(FieldIdentityMigrationError::MissingField { .. })
        ));
        assert!(matches!(
            FieldIdentityMigration::new([
                (
                    (old_component, FieldPath::parse("a").unwrap()),
                    (new_component, FieldPath::parse("x").unwrap())
                ),
                (
                    (old_component, FieldPath::parse("b").unwrap()),
                    (new_component, FieldPath::parse("x").unwrap())
                ),
            ]),
            Err(FieldIdentityMigrationError::DuplicateTarget { .. })
        ));

        let mut overrides = OverrideSet::new();
        overrides.set_property(source.clone());
        let snapshot = SceneInstancePersistence {
            template: StableGuid::from_u128(9),
            instance: None,
            properties: vec![crate::PersistedPropertyOverride {
                local: LocalEntityId::new(0),
                persistent: None,
                component: old_component,
                path: FieldPath::parse("translation.x").unwrap(),
                value: OverrideValue::F32(3.5),
            }],
            structural: Vec::new(),
        };
        let migrated_snapshot = entities
            .migrate_persistence_with_fields(&snapshot, &fields)
            .unwrap();
        assert_eq!(migrated_snapshot.properties[0].component, new_component);
        assert_eq!(migrated_snapshot.properties[0].path, new_path);
        assert_eq!(
            migrated_snapshot.properties[0].value,
            OverrideValue::F32(3.5)
        );
        assert_eq!(
            entities
                .migrate_overrides_with_fields(&overrides, &fields)
                .unwrap()
                .properties()[0]
                .path,
            FieldPath::parse("position.horizontal").unwrap()
        );
    }
}
