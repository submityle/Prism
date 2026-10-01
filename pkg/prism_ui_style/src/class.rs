//! Classes and style sheets.
//!
//! A [`Class`] is a named bag of properties with optional per-interaction-state
//! and per-breakpoint overrides. A [`StyleSheet`] collects many classes by
//! name so the cascade can look them up when resolving an element's applied
//! class list.

use alloc::collections::BTreeMap;
use alloc::string::String;

use crate::selector::{Breakpoint, InteractionState};
use crate::value::{StyleProp, StyleValue};

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
}
