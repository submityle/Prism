//! Deterministic scene-plan cost accounting.
//!
//! The cost model is intentionally a count model rather than a platform
//! benchmark. It lets loaders and streaming systems reject work before a frame
//! budget is exceeded, while real benchmark results remain backend-specific.

use crate::plan::InstantiationPlan;

/// Work and memory counts implied by one instantiation plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScenePlanCost {
    /// Number of ECS entities reserved by the base plan.
    pub entities: usize,
    /// Number of component columns copied or decoded.
    pub columns: usize,
    /// Total serialized column payload bytes.
    pub payload_bytes: usize,
    /// Number of hierarchy edges to install.
    pub hierarchy_edges: usize,
    /// Number of entity-valued fields requiring remap.
    pub remap_slots: usize,
}

impl ScenePlanCost {
    /// Returns a conservative scalar work estimate for budget sorting.
    #[must_use]
    pub fn work_units(self) -> usize {
        self.entities
            .saturating_add(self.columns)
            .saturating_add(self.hierarchy_edges)
            .saturating_add(self.remap_slots)
    }
}

impl InstantiationPlan {
    /// Computes deterministic counts without allocating or decoding payloads.
    #[must_use]
    pub fn cost(&self) -> ScenePlanCost {
        ScenePlanCost {
            entities: self.entity_count as usize,
            columns: self.groups.iter().map(|group| group.columns.len()).sum(),
            payload_bytes: self
                .groups
                .iter()
                .flat_map(|group| group.columns.iter())
                .map(|column| column.blob.len())
                .sum(),
            hierarchy_edges: self.hierarchy.len(),
            remap_slots: self.entity_valued_fields.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::LocalEntityId;
    use crate::overrides::ArchetypeSignature;
    use crate::plan::{ArchetypeGroup, Column};
    use alloc::boxed::Box;
    use alloc::vec;

    #[test]
    fn counts_plan_work_without_decoding_columns() {
        let plan = InstantiationPlan {
            entity_count: 2,
            groups: Box::new([ArchetypeGroup::new(
                ArchetypeSignature::default(),
                Box::new([LocalEntityId::new(0), LocalEntityId::new(1)]),
                Box::new([]),
            )]),
            entity_valued_fields: Box::new([]),
            hierarchy: Box::new([(LocalEntityId::new(1), LocalEntityId::new(0))]),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        };
        let cost = plan.cost();
        assert_eq!(cost.entities, 2);
        assert_eq!(cost.hierarchy_edges, 1);
        assert_eq!(cost.work_units(), 3);
    }

    #[test]
    fn payload_bytes_are_summed_across_columns() {
        let component = crate::ComponentTypeId::from_raw(1);
        let plan = InstantiationPlan {
            entity_count: 1,
            groups: Box::new([ArchetypeGroup::new(
                ArchetypeSignature::new(vec![component]),
                Box::new([LocalEntityId::new(0)]),
                Box::new([Column::new(component, vec![1, 2, 3])]),
            )]),
            entity_valued_fields: Box::new([]),
            hierarchy: Box::new([]),
            reserved_local_base: LocalEntityId::RESERVED_BASE,
        };
        assert_eq!(plan.cost().payload_bytes, 3);
    }
}
