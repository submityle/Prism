//! The flattened instantiation plan (design §8.1) — the bake-to-runtime
//! contract that turns "spawn entities one at a time and insert components" into
//! "reserve once, copy whole archetype columns, remap entity-valued slots".
//!
//! A plan is produced by the bake (`prism_scene_bake`) and consumed by the
//! runtime instantiation path (design §8.2). This module owns the plan's shape
//! and its structural validation, which is what guards the hot path: an invalid
//! plan is rejected up front rather than corrupting a world mid-instantiation.
//!
//! Column payloads are [`Arc`]-shared and immutable, so the same template
//! instantiated many times shares one backing buffer per column (COW, design
//! §8.3) — only the instance's [`RemapTable`](crate::remap::RemapTable) and any
//! overridden column are owned per instance.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::identity::LocalEntityId;
use crate::overrides::{ArchetypeSignature, ComponentTypeId};

/// One component column of an archetype group: a component type id and its
/// alignment-stable, immutable `SoA` blob (design §8.1).
///
/// The blob is [`Arc`]-shared so N instances of the same template share one
/// buffer (design §8.3 COW); a column that must be remapped or overridden is
/// copied on write by the instance, never mutated through this handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Column {
    /// Which component this column stores.
    pub component: ComponentTypeId,
    /// The column's rows, laid out consecutively (`SoA`), one row per entity in
    /// the owning group.
    pub blob: Arc<[u8]>,
}

impl Column {
    /// Builds a column from a component id and an owned or shared byte blob.
    #[must_use]
    pub fn new(component: ComponentTypeId, blob: impl Into<Arc<[u8]>>) -> Self {
        Self {
            component,
            blob: blob.into(),
        }
    }

    /// Byte length of the whole column.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.blob.len()
    }

    /// The number of distinct instances sharing this column buffer.
    #[must_use]
    pub fn shared_count(&self) -> usize {
        Arc::strong_count(&self.blob)
    }
}

/// A run of entities that share one archetype, with their component columns
/// stored consecutively (design §8.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchetypeGroup {
    /// The component set shared by every entity in the group.
    pub signature: ArchetypeSignature,
    /// The group's entities, in column order.
    pub locals: Box<[LocalEntityId]>,
    /// One column per component in `signature`.
    pub columns: Box<[Column]>,
}

impl ArchetypeGroup {
    /// Builds a group.
    #[must_use]
    pub fn new(
        signature: ArchetypeSignature,
        locals: Box<[LocalEntityId]>,
        columns: Box<[Column]>,
    ) -> Self {
        Self {
            signature,
            locals,
            columns,
        }
    }

    /// Number of entities in the group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.locals.len()
    }

    /// Whether the group is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.locals.is_empty()
    }
}

/// The position of one entity-valued field that must be remapped at
/// instantiation (design §8.1 `FieldSlot`).
///
/// Such a field stores a template-local [`LocalEntityId`] at a known byte
/// offset; the runtime patched it to the instance's live entity via the
/// [`RemapTable`](crate::remap::RemapTable) — offset-driven, reflection-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldSlot {
    /// The entity that owns the field.
    pub local: LocalEntityId,
    /// Index into [`InstantiationPlan::groups`].
    pub group: u32,
    /// Index into the owning group's `columns`.
    pub column: u32,
    /// Byte offset of the field within one row of the column.
    pub byte_offset: u32,
}

/// A flattened, validated instantiation plan (design §8.1).
///
/// All entity-valued references are expressed as [`LocalEntityId`]s; the runtime
/// patch step is driven purely by [`entity_valued_fields`](Self::entity_valued_fields)
/// offsets, and the hierarchy is a flat `(child, parent)` list — no reflection
/// runs on the hot path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstantiationPlan {
    /// Total number of base-segment entities the plan instantiates.
    pub entity_count: u32,
    /// Entities grouped by archetype, each with columnar payloads.
    pub groups: Box<[ArchetypeGroup]>,
    /// Entity-valued field positions to remap (design §8.2 step 4).
    pub entity_valued_fields: Box<[FieldSlot]>,
    /// Flat `(child, parent)` hierarchy to re-link (design §8.2 step 5).
    pub hierarchy: Box<[(LocalEntityId, LocalEntityId)]>,
    /// First `LocalEntityId` a structural override may allocate; equals
    /// [`LocalEntityId::RESERVED_BASE`] (design §6/§7.1).
    pub reserved_local_base: u32,
}

/// A structural defect that makes an [`InstantiationPlan`] unusable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// A group's local id is in the reserved segment or `>= entity_count`.
    LocalOutOfRange {
        /// The offending id.
        local: LocalEntityId,
    },
    /// The same local id appears in more than one group.
    DuplicateLocal {
        /// The duplicated id.
        local: LocalEntityId,
    },
    /// The number of locals across all groups does not equal `entity_count`.
    LocalCountMismatch {
        /// The plan's declared `entity_count`.
        expected: u32,
        /// The number of locals actually listed.
        found: u32,
    },
    /// A column's component type is not part of its group's signature.
    ColumnNotInSignature {
        /// The group index.
        group: u32,
        /// The offending component.
        component: ComponentTypeId,
    },
    /// A field slot references a group or column that does not exist.
    FieldSlotOutOfRange {
        /// The offending slot.
        slot: FieldSlot,
    },
    /// A hierarchy edge names a local id that is not part of the plan.
    HierarchyUnknownLocal {
        /// The unknown id.
        local: LocalEntityId,
    },
    /// `reserved_local_base` is not [`LocalEntityId::RESERVED_BASE`].
    ReservedBaseMismatch {
        /// The value found in the plan.
        found: u32,
    },
    /// A non-empty plan does not contain local zero, its root.
    MissingRoot,
    /// Local zero is declared as a child of another entity.
    RootHasParent,
    /// A hierarchy child is declared more than once.
    DuplicateHierarchyChild {
        /// The duplicated child local id.
        child: LocalEntityId,
    },
    /// A base entity is not connected to local zero.
    DisconnectedLocal {
        /// The disconnected local id.
        local: LocalEntityId,
    },
    /// The base hierarchy contains a cycle.
    HierarchyCycle {
        /// A local id participating in the cycle.
        local: LocalEntityId,
    },
    /// Validation could not reserve the scratch memory required for the
    /// declared entity count.
    AllocationFailed,
}

impl InstantiationPlan {
    /// Validates every structural invariant the runtime relies on, returning the
    /// first defect found.
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.reserved_local_base != LocalEntityId::RESERVED_BASE {
            return Err(PlanError::ReservedBaseMismatch {
                found: self.reserved_local_base,
            });
        }

        let entity_count = self.entity_count as usize;
        let mut local_groups = try_filled(entity_count, None)?;
        let mut local_count = 0usize;
        for (group_index, group) in self.groups.iter().enumerate() {
            for &local in group.locals.iter() {
                if local.is_reserved() || local.index() >= self.entity_count {
                    return Err(PlanError::LocalOutOfRange { local });
                }
                let owner = &mut local_groups[local.index() as usize];
                if owner.replace(group_index as u32).is_some() {
                    return Err(PlanError::DuplicateLocal { local });
                }
                local_count += 1;
            }
            for column in group.columns.iter() {
                if !group.signature.contains(column.component) {
                    return Err(PlanError::ColumnNotInSignature {
                        group: group_index as u32,
                        component: column.component,
                    });
                }
            }
        }

        if local_count as u32 != self.entity_count {
            return Err(PlanError::LocalCountMismatch {
                expected: self.entity_count,
                found: local_count as u32,
            });
        }

        if self.entity_count != 0 && local_groups[0].is_none() {
            return Err(PlanError::MissingRoot);
        }

        for &slot in self.entity_valued_fields.iter() {
            let Some(group) = self.groups.get(slot.group as usize) else {
                return Err(PlanError::FieldSlotOutOfRange { slot });
            };
            if slot.column as usize >= group.columns.len()
                || local_groups
                    .get(slot.local.index() as usize)
                    .copied()
                    .flatten()
                    != Some(slot.group)
            {
                return Err(PlanError::FieldSlotOutOfRange { slot });
            }
        }

        let mut parents: Vec<Option<LocalEntityId>> = try_filled(entity_count, None)?;
        for &(child, parent) in self.hierarchy.iter() {
            if child.is_reserved()
                || local_groups
                    .get(child.index() as usize)
                    .copied()
                    .flatten()
                    .is_none()
            {
                return Err(PlanError::HierarchyUnknownLocal { local: child });
            }
            if parent.is_reserved()
                || local_groups
                    .get(parent.index() as usize)
                    .copied()
                    .flatten()
                    .is_none()
            {
                return Err(PlanError::HierarchyUnknownLocal { local: parent });
            }
            if child == LocalEntityId::new(0) {
                return Err(PlanError::RootHasParent);
            }
            if parents[child.index() as usize].replace(parent).is_some() {
                return Err(PlanError::DuplicateHierarchyChild { child });
            }
        }

        if self.entity_count != 0 {
            let mut colors = try_filled(entity_count, 0u8)?;
            colors[0] = 2;
            for index in 1..self.entity_count as usize {
                if colors[index] == 2 {
                    continue;
                }
                let start = LocalEntityId::new(index as u32);
                let mut path = Vec::new();
                let mut cursor = start;
                loop {
                    let cursor_index = cursor.index() as usize;
                    match colors[cursor_index] {
                        2 => break,
                        1 => return Err(PlanError::HierarchyCycle { local: cursor }),
                        _ => {}
                    }
                    colors[cursor_index] = 1;
                    path.push(cursor);
                    let Some(parent) = parents[cursor_index] else {
                        return Err(PlanError::DisconnectedLocal { local: start });
                    };
                    cursor = parent;
                }
                for local in path {
                    colors[local.index() as usize] = 2;
                }
            }
        }

        Ok(())
    }

    /// The number of archetype groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// The group index owning `local`, if any.
    #[must_use]
    pub fn group_of(&self, local: LocalEntityId) -> Option<u32> {
        self.groups
            .iter()
            .position(|group| group.locals.contains(&local))
            .map(|index| index as u32)
    }
}

fn try_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, PlanError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| PlanError::AllocationFailed)?;
    values.resize(len, value);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::{ArchetypeGroup, Column, FieldSlot, InstantiationPlan, PlanError};
    use crate::identity::LocalEntityId;
    use crate::overrides::{ArchetypeSignature, ComponentTypeId};
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec;

    fn comp(raw: u64) -> ComponentTypeId {
        ComponentTypeId::from_raw(raw)
    }

    fn group(signature: &[u64], locals: &[u32], columns: &[u64]) -> ArchetypeGroup {
        ArchetypeGroup::new(
            ArchetypeSignature::new(signature.iter().map(|&c| comp(c)).collect()),
            locals.iter().map(|&l| LocalEntityId::new(l)).collect(),
            columns
                .iter()
                .map(|&c| Column::new(comp(c), vec![0u8; 4].into_boxed_slice()))
                .collect(),
        )
    }

    fn base_plan() -> InstantiationPlan {
        InstantiationPlan {
            entity_count: 3,
            groups: Box::new([group(&[1, 2], &[0, 1], &[1, 2]), group(&[3], &[2], &[3])]),
            entity_valued_fields: Box::new([FieldSlot {
                local: LocalEntityId::new(0),
                group: 0,
                column: 1,
                byte_offset: 0,
            }]),
            hierarchy: Box::new([
                (LocalEntityId::new(1), LocalEntityId::new(0)),
                (LocalEntityId::new(2), LocalEntityId::new(0)),
            ]),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        }
    }

    #[test]
    fn valid_plan_passes() {
        let plan = base_plan();
        assert_eq!(plan.validate(), Ok(()));
        assert_eq!(plan.group_count(), 2);
        assert_eq!(plan.group_of(LocalEntityId::new(2)), Some(1));
        assert_eq!(plan.group_of(LocalEntityId::new(9)), None);
    }

    #[test]
    fn column_shares_its_blob() {
        let shared: Arc<[u8]> = vec![1u8, 2, 3, 4].into();
        let a = Column::new(comp(1), shared.clone());
        let b = Column::new(comp(1), shared.clone());
        assert_eq!(a.blob, b.blob);
        // `shared` + a + b hold the same allocation.
        assert_eq!(a.shared_count(), 3);
        assert_eq!(a.byte_len(), 4);
    }

    #[test]
    fn rejects_local_out_of_range() {
        let mut plan = base_plan();
        plan.groups[1].locals = Box::new([LocalEntityId::new(7)]);
        assert_eq!(
            plan.validate(),
            Err(PlanError::LocalOutOfRange {
                local: LocalEntityId::new(7)
            })
        );
    }

    #[test]
    fn rejects_duplicate_local() {
        let mut plan = base_plan();
        plan.groups[1].locals = Box::new([LocalEntityId::new(0)]);
        assert_eq!(
            plan.validate(),
            Err(PlanError::DuplicateLocal {
                local: LocalEntityId::new(0)
            })
        );
    }

    #[test]
    fn rejects_local_count_mismatch() {
        let mut plan = base_plan();
        plan.entity_count = 5;
        assert_eq!(
            plan.validate(),
            Err(PlanError::LocalCountMismatch {
                expected: 5,
                found: 3
            })
        );
    }

    #[test]
    fn rejects_column_outside_signature() {
        let mut plan = base_plan();
        plan.groups[0].columns = Box::new([Column::new(comp(99), vec![0u8; 4].into_boxed_slice())]);
        assert_eq!(
            plan.validate(),
            Err(PlanError::ColumnNotInSignature {
                group: 0,
                component: comp(99)
            })
        );
    }

    #[test]
    fn rejects_bad_field_slot() {
        let mut plan = base_plan();
        plan.entity_valued_fields = Box::new([FieldSlot {
            local: LocalEntityId::new(0),
            group: 5,
            column: 0,
            byte_offset: 0,
        }]);
        assert!(matches!(
            plan.validate(),
            Err(PlanError::FieldSlotOutOfRange { .. })
        ));
    }

    #[test]
    fn rejects_unknown_hierarchy_local() {
        let mut plan = base_plan();
        plan.hierarchy = Box::new([(LocalEntityId::new(1), LocalEntityId::new(8))]);
        assert_eq!(
            plan.validate(),
            Err(PlanError::HierarchyUnknownLocal {
                local: LocalEntityId::new(8)
            })
        );
    }

    #[test]
    fn rejects_reserved_base_mismatch() {
        let mut plan = base_plan();
        plan.reserved_local_base = 0;
        assert_eq!(
            plan.validate(),
            Err(PlanError::ReservedBaseMismatch { found: 0 })
        );
    }

    #[test]
    fn requires_one_rooted_hierarchy() {
        let mut plan = base_plan();
        plan.hierarchy = Box::new([(LocalEntityId::new(2), LocalEntityId::new(1))]);
        assert_eq!(
            plan.validate(),
            Err(PlanError::DisconnectedLocal {
                local: LocalEntityId::new(1)
            })
        );

        let mut plan = base_plan();
        plan.hierarchy = Box::new([
            (LocalEntityId::new(1), LocalEntityId::new(0)),
            (LocalEntityId::new(2), LocalEntityId::new(0)),
        ]);
        assert_eq!(plan.validate(), Ok(()));
    }

    #[test]
    fn rejects_root_parent_duplicate_parent_and_cycle() {
        let mut plan = base_plan();
        plan.hierarchy = Box::new([
            (LocalEntityId::new(0), LocalEntityId::new(1)),
            (LocalEntityId::new(1), LocalEntityId::new(0)),
        ]);
        assert_eq!(plan.validate(), Err(PlanError::RootHasParent));

        let mut plan = base_plan();
        plan.hierarchy = Box::new([
            (LocalEntityId::new(1), LocalEntityId::new(0)),
            (LocalEntityId::new(1), LocalEntityId::new(2)),
            (LocalEntityId::new(2), LocalEntityId::new(0)),
        ]);
        assert_eq!(
            plan.validate(),
            Err(PlanError::DuplicateHierarchyChild {
                child: LocalEntityId::new(1)
            })
        );

        let mut plan = base_plan();
        plan.hierarchy = Box::new([
            (LocalEntityId::new(1), LocalEntityId::new(2)),
            (LocalEntityId::new(2), LocalEntityId::new(1)),
        ]);
        assert_eq!(
            plan.validate(),
            Err(PlanError::HierarchyCycle {
                local: LocalEntityId::new(1)
            })
        );
    }
}
