//! Per-view streaming and `LOD` importance, and its aggregation.
//!
//! Every view contributes a [`ViewImportance`]: three non-negative weights that
//! say how aggressively the shared subsystems should serve it. `streaming`
//! drives virtual-resource residency priority, `geometry_lod` biases geometry
//! detail selection, and `shadow_lod` biases shadow-map detail. A frame usually
//! renders several views at once (the main camera, a handful of shadow views, a
//! reflection probe), so the shared budget must be driven by a *combination* of
//! their importances — either the component-wise maximum (serve the most
//! demanding view) or a weighted accumulation (share the budget by view kind).
//!
//! All values are sanitized to be finite and non-negative, so a stray `NaN` or
//! negative weight from a caller can never poison the shared budget. Comparisons
//! use `f32::max`, never transcendental functions, and clamping always tests for
//! `NaN` explicitly before calling `clamp`.

use super::{ViewImportance, ViewKind};

impl ViewImportance {
    /// Largest weight any component may take after sanitization.
    pub const MAX_COMPONENT: f32 = 1.0e6;

    /// All-zero importance: the neutral element for [`ViewImportance::combine_max`].
    pub const ZERO: Self = Self {
        streaming: 0.0,
        geometry_lod: 0.0,
        shadow_lod: 0.0,
    };

    /// Clamps one component to the finite, non-negative `[0, MAX_COMPONENT]`
    /// range, mapping `NaN` to `0.0`.
    #[must_use]
    fn sanitize(value: f32) -> f32 {
        if value.is_nan() {
            0.0
        } else {
            value.clamp(0.0, Self::MAX_COMPONENT)
        }
    }

    /// Builds a sanitized importance from its three components.
    #[must_use]
    pub fn new(streaming: f32, geometry_lod: f32, shadow_lod: f32) -> Self {
        Self {
            streaming: Self::sanitize(streaming),
            geometry_lod: Self::sanitize(geometry_lod),
            shadow_lod: Self::sanitize(shadow_lod),
        }
    }

    /// Builds an importance with the same sanitized weight on every component.
    #[must_use]
    pub fn uniform(weight: f32) -> Self {
        let weight = Self::sanitize(weight);
        Self {
            streaming: weight,
            geometry_lod: weight,
            shadow_lod: weight,
        }
    }

    /// Component-wise maximum of two importances.
    ///
    /// This is how the shared budget is driven when it should satisfy the most
    /// demanding view: each subsystem sees the peak demand across all views.
    #[must_use]
    pub fn combine_max(self, other: Self) -> Self {
        Self {
            streaming: self.streaming.max(other.streaming),
            geometry_lod: self.geometry_lod.max(other.geometry_lod),
            shadow_lod: self.shadow_lod.max(other.shadow_lod),
        }
    }

    /// Scales every component by a sanitized non-negative `weight`.
    ///
    /// The result is re-sanitized, so scaling can never push a component past
    /// [`ViewImportance::MAX_COMPONENT`] or introduce a non-finite value.
    #[must_use]
    pub fn scaled(self, weight: f32) -> Self {
        let weight = Self::sanitize(weight);
        Self::new(
            self.streaming * weight,
            self.geometry_lod * weight,
            self.shadow_lod * weight,
        )
    }

    /// Component-wise sum of two importances, re-sanitized.
    ///
    /// Combined with [`ViewImportance::scaled`], this accumulates a weighted
    /// budget contribution across views.
    #[must_use]
    pub fn combine_add(self, other: Self) -> Self {
        Self::new(
            self.streaming + other.streaming,
            self.geometry_lod + other.geometry_lod,
            self.shadow_lod + other.shadow_lod,
        )
    }

    /// The largest of the three components.
    ///
    /// A single scalar useful for coarse budget decisions, such as ranking views
    /// or picking an overall streaming urgency.
    #[must_use]
    pub fn dominant(self) -> f32 {
        self.streaming.max(self.geometry_lod).max(self.shadow_lod)
    }

    /// Whether every component is finite (always true after sanitization).
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.streaming.is_finite() && self.geometry_lod.is_finite() && self.shadow_lod.is_finite()
    }
}

impl Default for ViewImportance {
    fn default() -> Self {
        Self::ZERO
    }
}

impl ViewKind {
    /// Default per-view importance for this kind of view.
    ///
    /// Primary views (main camera, stereo eyes, offline tiles) demand full
    /// detail; auxiliary views (shadows, reflections, captures) demand less, and
    /// shadow views in particular weight `shadow_lod` above their streaming and
    /// geometry needs.
    #[must_use]
    pub fn default_importance(self) -> ViewImportance {
        let (streaming, geometry_lod, shadow_lod) = match self {
            Self::Main | Self::StereoEye | Self::OfflineTile => (1.0, 1.0, 1.0),
            Self::Editor => (0.75, 0.75, 0.5),
            Self::Reflection => (0.6, 0.5, 0.4),
            Self::Shadow => (0.5, 0.5, 1.0),
            Self::SceneCapture => (0.4, 0.4, 0.3),
        };
        ViewImportance::new(streaming, geometry_lod, shadow_lod)
    }

    /// Relative weight this kind of view carries in a weighted budget split.
    ///
    /// Used by [`crate::view_family::registry::ViewRegistry::weighted_importance`]
    /// to bias the shared budget toward primary views.
    #[must_use]
    pub fn budget_weight(self) -> f32 {
        match self {
            Self::Main | Self::StereoEye => 1.0,
            Self::OfflineTile => 0.8,
            Self::Editor => 0.75,
            Self::Reflection | Self::Shadow => 0.5,
            Self::SceneCapture => 0.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sanitizes_nan_and_negatives() {
        let imp = ViewImportance::new(f32::NAN, -3.0, 2.0);
        assert_eq!(imp, ViewImportance::new(0.0, 0.0, 2.0));
        assert!(imp.is_finite());
    }

    #[test]
    fn new_clamps_above_max_component() {
        let imp = ViewImportance::new(f32::INFINITY, ViewImportance::MAX_COMPONENT * 4.0, 1.0);
        assert_eq!(imp.streaming, ViewImportance::MAX_COMPONENT);
        assert_eq!(imp.geometry_lod, ViewImportance::MAX_COMPONENT);
    }

    #[test]
    fn combine_max_takes_component_peaks() {
        let a = ViewImportance::new(0.2, 0.9, 0.1);
        let b = ViewImportance::new(0.8, 0.3, 0.5);
        assert_eq!(a.combine_max(b), ViewImportance::new(0.8, 0.9, 0.5));
    }

    #[test]
    fn scaled_and_add_accumulate_weighted_budget() {
        let a = ViewImportance::uniform(1.0).scaled(0.5);
        let b = ViewImportance::uniform(1.0).scaled(0.25);
        assert_eq!(a.combine_add(b), ViewImportance::uniform(0.75));
    }

    #[test]
    fn dominant_returns_largest_component() {
        let imp = ViewImportance::new(0.2, 0.7, 0.5);
        assert_eq!(imp.dominant(), 0.7);
    }

    #[test]
    fn shadow_view_weights_shadow_lod_highest() {
        let imp = ViewKind::Shadow.default_importance();
        assert_eq!(imp.shadow_lod, 1.0);
        assert!(imp.shadow_lod > imp.streaming);
    }

    #[test]
    fn primary_views_demand_full_detail() {
        for kind in [ViewKind::Main, ViewKind::StereoEye, ViewKind::OfflineTile] {
            assert_eq!(kind.default_importance(), ViewImportance::uniform(1.0));
        }
    }

    #[test]
    fn scaled_stays_within_max_component() {
        let imp = ViewImportance::uniform(ViewImportance::MAX_COMPONENT).scaled(10.0);
        assert_eq!(imp.streaming, ViewImportance::MAX_COMPONENT);
    }
}
