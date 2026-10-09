//! Classes and style sheets.
//!
//! A [`Class`] is a named bag of properties with optional per-interaction-state
//! and per-breakpoint overrides. A [`StyleSheet`] collects many classes by
//! name so the cascade can look them up when resolving an element's applied
//! class list.

use alloc::collections::BTreeMap;
use alloc::string::String;

use crate::selector::{Breakpoint, InteractionState};
use crate::value::{Keyword, StyleProp, StyleValue};

/// A map of properties to values.
pub type PropMap = BTreeMap<StyleProp, StyleValue>;

/// A named collection of style properties with optional variant overrides.
///
/// The `base` map always applies. The `states` and `breakpoints` maps hold
/// overrides that only apply when the matching interaction state or breakpoint
/// is active in the [`crate::selector::MatchContext`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Class {
    /// The class name.
    pub name: String,
    /// Properties that always apply.
    pub base: PropMap,
    /// Per-interaction-state override maps.
    pub states: BTreeMap<InteractionState, PropMap>,
    /// Per-breakpoint override maps.
    pub breakpoints: BTreeMap<Breakpoint, PropMap>,
}

impl Class {
    /// Creates a new, empty class with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base: PropMap::new(),
            states: BTreeMap::new(),
            breakpoints: BTreeMap::new(),
        }
    }

    /// Sets a base property in place.
    pub fn set(&mut self, prop: StyleProp, value: StyleValue) {
        self.base.insert(prop, value);
    }

    /// Sets a base property, returning the class for chaining.
    #[must_use]
    pub fn with(mut self, prop: StyleProp, value: StyleValue) -> Self {
        self.set(prop, value);
        self
    }

    /// Sets a property that only applies in the given interaction state.
    pub fn set_state(&mut self, state: InteractionState, prop: StyleProp, value: StyleValue) {
        self.states.entry(state).or_default().insert(prop, value);
    }

    /// Builder form of [`Class::set_state`].
    #[must_use]
    pub fn with_state(
        mut self,
        state: InteractionState,
        prop: StyleProp,
        value: StyleValue,
    ) -> Self {
        self.set_state(state, prop, value);
        self
    }

    /// Sets a property that only applies at the given breakpoint.
    pub fn set_breakpoint(&mut self, breakpoint: Breakpoint, prop: StyleProp, value: StyleValue) {
        self.breakpoints
            .entry(breakpoint)
            .or_default()
            .insert(prop, value);
    }

    /// Builder form of [`Class::set_breakpoint`].
    #[must_use]
    pub fn with_breakpoint(
        mut self,
        breakpoint: Breakpoint,
        prop: StyleProp,
        value: StyleValue,
    ) -> Self {
        self.set_breakpoint(breakpoint, prop, value);
        self
    }

    /// Sets left and right padding to the same value (the `padding-x`
    /// shorthand).
    pub fn set_padding_x(&mut self, value: StyleValue) {
        self.base.insert(StyleProp::PaddingLeft, value.clone());
        self.base.insert(StyleProp::PaddingRight, value);
    }

    /// Builder form of [`Class::set_padding_x`].
    #[must_use]
    pub fn with_padding_x(mut self, value: StyleValue) -> Self {
        self.set_padding_x(value);
        self
    }

    /// Sets top and bottom padding to the same value (the `padding-y`
    /// shorthand).
    pub fn set_padding_y(&mut self, value: StyleValue) {
        self.base.insert(StyleProp::PaddingTop, value.clone());
        self.base.insert(StyleProp::PaddingBottom, value);
    }

    /// Builder form of [`Class::set_padding_y`].
    #[must_use]
    pub fn with_padding_y(mut self, value: StyleValue) -> Self {
        self.set_padding_y(value);
        self
    }

    /// Sets left and right margin to the same value (the `margin-x`
    /// shorthand).
    pub fn set_margin_x(&mut self, value: StyleValue) {
        self.base.insert(StyleProp::MarginLeft, value.clone());
        self.base.insert(StyleProp::MarginRight, value);
    }

    /// Builder form of [`Class::set_margin_x`].
    #[must_use]
    pub fn with_margin_x(mut self, value: StyleValue) -> Self {
        self.set_margin_x(value);
        self
    }

    /// Sets top and bottom margin to the same value (the `margin-y`
    /// shorthand).
    pub fn set_margin_y(&mut self, value: StyleValue) {
        self.base.insert(StyleProp::MarginTop, value.clone());
        self.base.insert(StyleProp::MarginBottom, value);
    }

    /// Builder form of [`Class::set_margin_y`].
    #[must_use]
    pub fn with_margin_y(mut self, value: StyleValue) -> Self {
        self.set_margin_y(value);
        self
    }

    /// Sets a drop shadow from its longhands: offset, blur and color (spread
    /// stays zero). The `color` is any [`StyleValue`], so a design token such
    /// as `StyleValue::token("glass.shadow")` works and is resolved by the
    /// cascade. For a non-zero spread or an inner shadow set
    /// [`StyleProp::ShadowSpread`] / [`StyleProp::ShadowInset`] directly, or use
    /// [`Class::set_inset_shadow`].
    pub fn set_shadow(&mut self, offset_x: f32, offset_y: f32, blur: f32, color: StyleValue) {
        self.base
            .insert(StyleProp::ShadowOffsetX, StyleValue::px(offset_x));
        self.base
            .insert(StyleProp::ShadowOffsetY, StyleValue::px(offset_y));
        self.base.insert(StyleProp::ShadowBlur, StyleValue::px(blur));
        self.base.insert(StyleProp::ShadowColor, color);
    }

    /// Builder form of [`Class::set_shadow`].
    #[must_use]
    pub fn with_shadow(
        mut self,
        offset_x: f32,
        offset_y: f32,
        blur: f32,
        color: StyleValue,
    ) -> Self {
        self.set_shadow(offset_x, offset_y, blur, color);
        self
    }

    /// Sets an inset (inner) shadow from its longhands. Identical to
    /// [`Class::set_shadow`] but flagged inset.
    pub fn set_inset_shadow(
        &mut self,
        offset_x: f32,
        offset_y: f32,
        blur: f32,
        color: StyleValue,
    ) {
        self.set_shadow(offset_x, offset_y, blur, color);
        self.base
            .insert(StyleProp::ShadowInset, StyleValue::keyword(Keyword::Inset));
    }

    /// Builder form of [`Class::set_inset_shadow`].
    #[must_use]
    pub fn with_inset_shadow(
        mut self,
        offset_x: f32,
        offset_y: f32,
        blur: f32,
        color: StyleValue,
    ) -> Self {
        self.set_inset_shadow(offset_x, offset_y, blur, color);
        self
    }

    /// Turns the box into an approximate frosted-glass surface: a translucent
    /// `tint` with a backdrop-`blur` hint and an optional rim `highlight`. The
    /// tint/highlight accept any [`StyleValue`], so theme tokens such as
    /// `StyleValue::token("glass.tint")` resolve through the cascade. The blur
    /// is a device hint; the reference backend renders the translucent
    /// approximation instead of a true (expensive) backdrop blur.
    pub fn set_glass(&mut self, blur: f32, tint: StyleValue, highlight: Option<StyleValue>) {
        self.base.insert(StyleProp::GlassBlur, StyleValue::px(blur));
        self.base.insert(StyleProp::GlassTint, tint);
        if let Some(highlight) = highlight {
            self.base.insert(StyleProp::GlassHighlight, highlight);
        }
    }

    /// Builder form of [`Class::set_glass`].
    #[must_use]
    pub fn with_glass(
        mut self,
        blur: f32,
        tint: StyleValue,
        highlight: Option<StyleValue>,
    ) -> Self {
        self.set_glass(blur, tint, highlight);
        self
    }
}

/// A collection of [`Class`]es keyed by name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StyleSheet {
    classes: BTreeMap<String, Class>,
}

impl StyleSheet {
    /// Creates an empty style sheet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            classes: BTreeMap::new(),
        }
    }

    /// Inserts or replaces a class (keyed by its own name).
    pub fn insert(&mut self, class: Class) {
        self.classes.insert(class.name.clone(), class);
    }

    /// Inserts a class, returning the sheet for chaining.
    #[must_use]
    pub fn with_class(mut self, class: Class) -> Self {
        self.insert(class);
        self
    }

    /// Looks up a class by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Class> {
        self.classes.get(name)
    }

    /// Returns the number of classes in the sheet.
    #[must_use]
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// Returns `true` if the sheet contains no classes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// Iterates over `(name, class)` pairs in sorted name order.
    ///
    /// Useful for tooling that needs to walk every registered rule, such as
    /// exporting the sheet to another format (CSS, a style gallery, docs).
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Class)> {
        self.classes.iter()
    }
}

#[cfg(test)]
mod glass_shadow_tests {
    use super::*;
    use crate::value::Color;

    #[test]
    fn with_shadow_expands_to_longhands() {
        let class = Class::new("card").with_shadow(
            0.0,
            8.0,
            24.0,
            StyleValue::Color(Color::rgba8(0, 0, 0, 38)),
        );
        assert_eq!(
            class.base.get(&StyleProp::ShadowOffsetY),
            Some(&StyleValue::px(8.0))
        );
        assert_eq!(
            class.base.get(&StyleProp::ShadowBlur),
            Some(&StyleValue::px(24.0))
        );
        assert_eq!(
            class.base.get(&StyleProp::ShadowColor),
            Some(&StyleValue::Color(Color::rgba8(0, 0, 0, 38)))
        );
        assert!(!class.base.contains_key(&StyleProp::ShadowInset));
    }

    #[test]
    fn with_inset_shadow_sets_inset_keyword() {
        let class = Class::new("well").with_inset_shadow(
            0.0,
            2.0,
            4.0,
            StyleValue::Color(Color::rgba8(0, 0, 0, 60)),
        );
        assert_eq!(
            class.base.get(&StyleProp::ShadowInset),
            Some(&StyleValue::keyword(Keyword::Inset))
        );
    }

    #[test]
    fn with_glass_expands_and_accepts_tokens() {
        let class = Class::new("panel").with_glass(
            20.0,
            StyleValue::token("glass.tint"),
            Some(StyleValue::token("glass.highlight")),
        );
        assert_eq!(
            class.base.get(&StyleProp::GlassBlur),
            Some(&StyleValue::px(20.0))
        );
        assert_eq!(
            class.base.get(&StyleProp::GlassTint),
            Some(&StyleValue::token("glass.tint"))
        );
        assert_eq!(
            class.base.get(&StyleProp::GlassHighlight),
            Some(&StyleValue::token("glass.highlight"))
        );
    }

    #[test]
    fn with_glass_highlight_is_optional() {
        let class = Class::new("panel").with_glass(0.0, StyleValue::token("glass.tint"), None);
        assert!(!class.base.contains_key(&StyleProp::GlassHighlight));
    }
}
