//! Deterministic single-plan cache for interactive dock rendering.

use crate::{
    encode_layout, solve_with, DockLayout, LayoutMetrics, RenderPlan, RenderPlanError, Viewport,
};
use alloc::vec::Vec;

/// Cache key for a solved plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderPlanKey {
    /// Content fingerprint of the validated layout.
    pub layout_fingerprint: u64,
    /// Logical viewport.
    pub viewport: Viewport,
    /// Geometry policy.
    pub metrics: LayoutMetrics,
}

/// A bounded cache for the most recently used plan.
///
/// Dock interaction is usually locality-heavy: pointer motion changes the
/// preview while the layout and metrics remain stable. Keeping one validated
/// plan avoids repeated full solves without introducing stale mutable state.
#[derive(Clone, Debug, Default)]
pub struct RenderPlanCache {
    key: Option<RenderPlanKey>,
    plan: Option<RenderPlan>,
    hits: u64,
    misses: u64,
}

impl RenderPlanCache {
    /// Creates an empty cache.
    pub const fn new() -> Self {
        Self {
            key: None,
            plan: None,
            hits: 0,
            misses: 0,
        }
    }

    /// Returns the number of cache hits.
    pub const fn hits(&self) -> u64 {
        self.hits
    }

    /// Returns the number of cache misses.
    pub const fn misses(&self) -> u64 {
        self.misses
    }

    /// Returns the cached plan when the exact key matches.
    pub fn get_or_solve(
        &mut self,
        layout: &DockLayout,
        viewport: Viewport,
        metrics: LayoutMetrics,
    ) -> Result<&RenderPlan, RenderPlanError> {
        let key = RenderPlanKey {
            layout_fingerprint: fingerprint(layout).map_err(RenderPlanError::InvalidLayout)?,
            viewport,
            metrics,
        };
        if self.key == Some(key) {
            self.hits = self.hits.saturating_add(1);
            return Ok(self.plan.as_ref().expect("cache key and plan are paired"));
        }
        self.misses = self.misses.saturating_add(1);
        let plan = solve_with(layout, viewport, metrics)?;
        self.key = Some(key);
        self.plan = Some(plan);
        Ok(self.plan.as_ref().expect("plan was inserted"))
    }

    /// Invalidates the retained plan and counters.
    pub fn clear(&mut self) {
        self.key = None;
        self.plan = None;
        self.hits = 0;
        self.misses = 0;
    }
}

fn fingerprint(layout: &DockLayout) -> Result<u64, crate::LayoutError> {
    let bytes = encode_layout(layout).map_err(|error| match error {
        crate::PersistError::InvalidLayout(layout) => layout,
        _ => crate::LayoutError {
            issue: crate::ValidationIssue::ResourceLimit("layout encoding"),
            path: Vec::new(),
        },
    })?;
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DockNode, Viewport};

    #[test]
    fn exact_inputs_hit_and_any_input_change_misses() {
        let layout = DockLayout {
            root: DockNode::panel("panel"),
            ..DockLayout::default()
        };
        let mut cache = RenderPlanCache::new();
        cache
            .get_or_solve(&layout, Viewport::new(640, 480), LayoutMetrics::default())
            .unwrap();
        cache
            .get_or_solve(&layout, Viewport::new(640, 480), LayoutMetrics::default())
            .unwrap();
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
        cache
            .get_or_solve(&layout, Viewport::new(800, 480), LayoutMetrics::default())
            .unwrap();
        assert_eq!(cache.misses(), 2);
    }
}
