//! Node-graph to `IR` compilation contract (design §6).
//!
//! An authored particle graph compiles to an intermediate representation that
//! drives the rest of the subsystem: attribute demand analysis (which feeds the
//! `SoA` allocation in [`super::attributes`]), stage ordering, ping-pong
//! decisions, and barrier minimization. This `CPU`-verifiable layer models the
//! analysis passes; the `WESL` codegen is pending the `GPU` backend.

use alloc::vec::Vec;

use super::attributes::{AttributeAccess, AttributeSemantic, AttributeUsage};

/// The aggregated attribute demand a compiled graph places on the pool.
///
/// The compiler unions every stage's per-attribute access so the layout planner
/// allocates exactly the buffers that are touched (design §5.1, §6).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AttributeDemand {
    usages: Vec<AttributeUsage>,
}

impl AttributeDemand {
    /// An empty demand.
    #[must_use]
    pub fn new() -> Self {
        Self { usages: Vec::new() }
    }

    /// Records that a stage touches `semantic` (with format from `usage`),
    /// unioning the access with any previously recorded access.
    pub fn touch(&mut self, usage: AttributeUsage) {
        if let Some(existing) = self
            .usages
            .iter_mut()
            .find(|u| u.semantic == usage.semantic)
        {
            existing.access = existing.access.union(usage.access);
        } else {
            self.usages.push(usage);
        }
    }

    /// The merged usages, in first-touch order.
    #[must_use]
    pub fn usages(&self) -> &[AttributeUsage] {
        &self.usages
    }

    /// Whether a semantic is demanded with a non-empty access.
    #[must_use]
    pub fn demands(&self, semantic: AttributeSemantic) -> bool {
        self.usages
            .iter()
            .any(|u| u.semantic == semantic && !u.access.is_empty())
    }
}

/// Convenience: build a read-write usage with the semantic's default format.
#[must_use]
pub fn read_write(semantic: AttributeSemantic) -> AttributeUsage {
    let format = semantic
        .default_format()
        .unwrap_or(super::attributes::AttributeFormat::F32);
    AttributeUsage::new(
        semantic,
        format,
        AttributeAccess::READ.union(AttributeAccess::WRITE),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_unions_access_for_repeated_semantics() {
        let mut demand = AttributeDemand::new();
        demand.touch(AttributeUsage::new(
            AttributeSemantic::Position,
            super::super::attributes::AttributeFormat::Vec3,
            AttributeAccess::READ,
        ));
        demand.touch(AttributeUsage::new(
            AttributeSemantic::Position,
            super::super::attributes::AttributeFormat::Vec3,
            AttributeAccess::WRITE,
        ));
        assert_eq!(demand.usages().len(), 1);
        let u = demand.usages()[0];
        assert!(u.access.contains(AttributeAccess::READ));
        assert!(u.access.contains(AttributeAccess::WRITE));
    }

    #[test]
    fn demands_reports_touched_semantics() {
        let mut demand = AttributeDemand::new();
        demand.touch(read_write(AttributeSemantic::Velocity));
        assert!(demand.demands(AttributeSemantic::Velocity));
        assert!(!demand.demands(AttributeSemantic::Color));
    }
}
