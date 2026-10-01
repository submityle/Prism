//! Responsive breakpoint resolution.
//!
//! Where the `prism_ui_style` cascade merges a whole class list (base, then
//! breakpoints, then interaction states) into a token-resolved
//! [`ComputedStyle`](prism_ui_style::ComputedStyle), the [`MediaResolver`] here
//! answers a narrower question: *for this one class, at this viewport width,
//! which property values are currently live?*
//!
//! Resolution is mobile-first, exactly matching Tailwind semantics: the base
//! map always applies, and every breakpoint whose `min-width` threshold the
//! viewport meets is layered on top in ascending order, so a wider viewport
//! inherits the overrides of all narrower ones. Token references are left
//! intact — this layer decides *which* value is active, not what it resolves
//! to.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};

use prism_ui_style::{
    Breakpoint, Class, InteractionState, MatchContext, PropMap, StyleProp, StyleValue,
};

use crate::scoped_sheet::ScopedSheet;

/// Resolves breakpoint (and optionally interaction-state) overrides for a
/// single viewport context.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediaResolver {
    ctx: MatchContext,
}

impl MediaResolver {
    /// Creates a resolver for a bare viewport width (no active states).
    #[must_use]
    pub fn new(viewport_width: f32) -> Self {
        Self {
            ctx: MatchContext::new(viewport_width),
        }
    }

    /// Creates a resolver from a full [`MatchContext`], carrying any active
    /// interaction states along with the viewport width.
    #[must_use]
    pub fn from_context(ctx: MatchContext) -> Self {
        Self { ctx }
    }

    /// Returns the viewport width this resolver was built with.
    #[must_use]
    pub fn viewport_width(&self) -> f32 {
        self.ctx.viewport_width
    }

    /// Returns the largest breakpoint active for this resolver's width.
    #[must_use]
    pub fn active_breakpoint(&self) -> Breakpoint {
        self.ctx.active_breakpoint()
    }

    /// Returns `true` if `breakpoint` is active for this resolver's width.
    #[must_use]
    pub fn is_active(&self, breakpoint: Breakpoint) -> bool {
        breakpoint.matches(self.ctx.viewport_width)
    }

    /// Computes the active property map for `class` at this resolver's width,
    /// applying base and matching breakpoint overrides (but not states).
    #[must_use]
    pub fn active_props(&self, class: &Class) -> PropMap {
        self.layer_breakpoints(class, self.ctx.viewport_width)
    }

    /// Computes the active property map for `class` at an explicit width,
    /// ignoring the resolver's own stored width.
    #[must_use]
    pub fn active_props_at(&self, class: &Class, viewport_width: f32) -> PropMap {
        self.layer_breakpoints(class, viewport_width)
    }

    /// Computes the active property map for `class` including both breakpoint
    /// overrides *and* the resolver context's active interaction states.
    ///
    /// Layering follows the cascade's order: base, then matching breakpoints in
    /// ascending `min-width` order, then matching states in priority order.
    #[must_use]
    pub fn active_props_with_states(&self, class: &Class) -> PropMap {
        let mut merged = self.layer_breakpoints(class, self.ctx.viewport_width);
        for state in InteractionState::ALL {
            if !self.ctx.states.contains(state) {
                continue;
            }
            if let Some(overrides) = class.states.get(&state) {
                merge(&mut merged, overrides);
            }
        }
        merged
    }

    /// Returns the single value `prop` would take on `class` at this width, if
    /// any breakpoint or the base map sets it.
    #[must_use]
    pub fn active_value(&self, class: &Class, prop: StyleProp) -> Option<StyleValue> {
        self.active_props(class).get(&prop).cloned()
    }

    /// Resolves every class in a [`ScopedSheet`] at this resolver's context,
    /// keyed by scoped class name.
    #[must_use]
    pub fn resolve_sheet(&self, scoped: &ScopedSheet) -> ResolvedSheet {
        let mut out: BTreeMap<String, PropMap> = BTreeMap::new();
        for scoped_name in scoped.scoped_names() {
            if let Some(class) = scoped.sheet().get(scoped_name) {
                out.insert(
                    scoped_name.to_string(),
                    self.active_props_with_states(class),
                );
            }
        }
        ResolvedSheet { classes: out }
    }

    /// Resolves an explicit list of classes at this resolver's context, keyed
    /// by each class's own name.
    #[must_use]
    pub fn resolve_classes<'a, I>(&self, classes: I) -> ResolvedSheet
    where
        I: IntoIterator<Item = &'a Class>,
    {
        let mut out: BTreeMap<String, PropMap> = BTreeMap::new();
        for class in classes {
            out.insert(class.name.clone(), self.active_props_with_states(class));
        }
        ResolvedSheet { classes: out }
    }

    /// Shared base + breakpoint layering used by the public entry points.
    ///
    /// Mirrors the cascade's breakpoint layer: start from the class's base
    /// map, then fold in every breakpoint whose threshold the width meets, in
    /// ascending `min-width` order. [`Breakpoint::Base`] has a `min-width` of
    /// `0.0`, so it always matches and is applied first.
    fn layer_breakpoints(&self, class: &Class, viewport_width: f32) -> PropMap {
        let mut merged: PropMap = class.base.clone();
        for breakpoint in Breakpoint::ALL {
            if !breakpoint.matches(viewport_width) {
                continue;
            }
            if let Some(overrides) = class.breakpoints.get(&breakpoint) {
                merge(&mut merged, overrides);
            }
        }
        merged
    }
}

/// Merges `source` into `target` with last-writer-wins semantics.
fn merge(target: &mut PropMap, source: &PropMap) {
    for (prop, value) in source {
        target.insert(*prop, value.clone());
    }
}

/// The resolved active-property maps for a set of classes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolvedSheet {
    classes: BTreeMap<String, PropMap>,
}

impl ResolvedSheet {
    /// Returns the active property map for `name`, if the class was resolved.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PropMap> {
        self.classes.get(name)
    }

    /// Returns the number of resolved classes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// Returns `true` if no classes were resolved.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// Iterates over `(class name, active props)` in sorted name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &PropMap)> {
        self.classes
            .iter()
            .map(|(name, props)| (name.as_str(), props))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ui_style::StyleSheet;

    use crate::scope::Scope;

    fn sample_class() -> Class {
        Class::new("box")
            .with(StyleProp::FontSize, StyleValue::px(12.0))
            .with(StyleProp::Opacity, StyleValue::number(1.0))
            .with_breakpoint(Breakpoint::Sm, StyleProp::FontSize, StyleValue::px(14.0))
            .with_breakpoint(Breakpoint::Md, StyleProp::FontSize, StyleValue::px(16.0))
            .with_breakpoint(Breakpoint::Lg, StyleProp::FontSize, StyleValue::px(20.0))
    }

    #[test]
    fn base_only_below_first_breakpoint() {
        let class = sample_class();
        let props = MediaResolver::new(500.0).active_props(&class);
        assert_eq!(props.get(&StyleProp::FontSize), Some(&StyleValue::px(12.0)));
        assert_eq!(
            props.get(&StyleProp::Opacity),
            Some(&StyleValue::number(1.0))
        );
    }

    #[test]
    fn breakpoint_boundaries_are_inclusive() {
        let class = sample_class();
        // 639 is below sm's 640 threshold.
        let below = MediaResolver::new(639.0).active_props(&class);
        assert_eq!(below.get(&StyleProp::FontSize), Some(&StyleValue::px(12.0)));
        // Exactly 640 activates sm.
        let at = MediaResolver::new(640.0).active_props(&class);
        assert_eq!(at.get(&StyleProp::FontSize), Some(&StyleValue::px(14.0)));
    }

    #[test]
    fn overrides_cascade_mobile_first() {
        let class = sample_class();
        // At 800px, sm + md match; md (ascending order) wins over sm.
        let props = MediaResolver::new(800.0).active_props(&class);
        assert_eq!(props.get(&StyleProp::FontSize), Some(&StyleValue::px(16.0)));
        // Non-overridden base props survive.
        assert_eq!(
            props.get(&StyleProp::Opacity),
            Some(&StyleValue::number(1.0))
        );
    }

    #[test]
    fn largest_matching_breakpoint_wins() {
        let class = sample_class();
        let props = MediaResolver::new(1300.0).active_props(&class);
        // Lg is the widest override present; xl has none, so lg stays live.
        assert_eq!(props.get(&StyleProp::FontSize), Some(&StyleValue::px(20.0)));
    }

    #[test]
    fn active_breakpoint_matches_width() {
        assert_eq!(
            MediaResolver::new(0.0).active_breakpoint(),
            Breakpoint::Base
        );
        assert_eq!(
            MediaResolver::new(640.0).active_breakpoint(),
            Breakpoint::Sm
        );
        assert_eq!(
            MediaResolver::new(767.0).active_breakpoint(),
            Breakpoint::Sm
        );
        assert_eq!(
            MediaResolver::new(768.0).active_breakpoint(),
            Breakpoint::Md
        );
        assert_eq!(
            MediaResolver::new(1024.0).active_breakpoint(),
            Breakpoint::Lg
        );
        assert_eq!(
            MediaResolver::new(1280.0).active_breakpoint(),
            Breakpoint::Xl
        );
    }

    #[test]
    fn active_value_reads_single_prop() {
        let class = sample_class();
        let resolver = MediaResolver::new(768.0);
        assert_eq!(
            resolver.active_value(&class, StyleProp::FontSize),
            Some(StyleValue::px(16.0))
        );
        assert_eq!(resolver.active_value(&class, StyleProp::Width), None);
    }

    #[test]
    fn active_props_at_ignores_stored_width() {
        let class = sample_class();
        let resolver = MediaResolver::new(100.0);
        let props = resolver.active_props_at(&class, 1024.0);
        assert_eq!(props.get(&StyleProp::FontSize), Some(&StyleValue::px(20.0)));
    }

    #[test]
    fn states_layer_over_breakpoints() {
        let class = sample_class().with_state(
            InteractionState::Hover,
            StyleProp::FontSize,
            StyleValue::px(99.0),
        );
        let ctx = MatchContext::new(1024.0).with_state(InteractionState::Hover);
        let resolver = MediaResolver::from_context(ctx);
        // Hover override beats the lg breakpoint override.
        let props = resolver.active_props_with_states(&class);
        assert_eq!(props.get(&StyleProp::FontSize), Some(&StyleValue::px(99.0)));
        // Without states, the breakpoint value stands.
        assert_eq!(
            resolver.active_props(&class).get(&StyleProp::FontSize),
            Some(&StyleValue::px(20.0))
        );
    }

    #[test]
    fn resolve_sheet_covers_every_scoped_class() {
        let sheet = StyleSheet::new().with_class(sample_class());
        let scoped = Scope::from_name("W").with_local("box").scope(&sheet);
        let resolved = MediaResolver::new(768.0).resolve_sheet(&scoped);
        assert_eq!(resolved.len(), 1);
        let scoped_name = scoped.scoped_name("box").unwrap();
        let props = resolved.get(scoped_name).unwrap();
        assert_eq!(props.get(&StyleProp::FontSize), Some(&StyleValue::px(16.0)));
    }

    #[test]
    fn resolve_classes_keys_by_class_name() {
        let class = sample_class();
        let resolved = MediaResolver::new(640.0).resolve_classes([&class]);
        assert!(!resolved.is_empty());
        let props = resolved.get("box").unwrap();
        assert_eq!(props.get(&StyleProp::FontSize), Some(&StyleValue::px(14.0)));
        assert_eq!(resolved.iter().count(), 1);
    }

    #[test]
    fn base_breakpoint_override_applies_first() {
        // An explicit `Base` breakpoint override should sit above base props
        // but below wider matching breakpoints.
        let class = Class::new("b")
            .with(StyleProp::Width, StyleValue::px(1.0))
            .with_breakpoint(Breakpoint::Base, StyleProp::Width, StyleValue::px(2.0))
            .with_breakpoint(Breakpoint::Md, StyleProp::Width, StyleValue::px(3.0));
        let narrow = MediaResolver::new(0.0).active_props(&class);
        assert_eq!(narrow.get(&StyleProp::Width), Some(&StyleValue::px(2.0)));
        let wide = MediaResolver::new(800.0).active_props(&class);
        assert_eq!(wide.get(&StyleProp::Width), Some(&StyleValue::px(3.0)));
    }
}
