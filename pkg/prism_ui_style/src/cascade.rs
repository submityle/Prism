//! The cascade resolver.
//!
//! Given an ordered list of class names, a [`MatchContext`], a [`StyleSheet`]
//! and a [`TokenStore`], the cascade produces a [`ComputedStyle`]: a flat map
//! of properties to fully-resolved values.
//!
//! # Layering
//!
//! Values are applied in three deterministic layers, each overriding the last:
//!
//! 1. **Base** — every class's base map, in class-list order.
//! 2. **Breakpoints** — matching breakpoints in ascending min-width order, and
//!    within each breakpoint, in class-list order.
//! 3. **States** — matching interaction states in
//!    [`InteractionState`](crate::selector::InteractionState) priority order,
//!    and within each state, in class-list order.
//!
//! Within every layer, later writes win (last-writer-wins). Unknown class names
//! are skipped. All token references are resolved before the result is
//! returned, so a [`ComputedStyle`] never contains a [`StyleValue::TokenRef`].

use alloc::collections::BTreeMap;

use crate::class::{PropMap, StyleSheet};
use crate::error::StyleError;
use crate::selector::{Breakpoint, InteractionState, MatchContext};
use crate::token::TokenStore;
use crate::value::{StyleProp, StyleValue};

/// A fully-resolved set of style properties.
///
/// Every value is a literal; token references have already been resolved.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ComputedStyle {
    props: PropMap,
}

impl ComputedStyle {
    /// Creates an empty computed style.
    #[must_use]
    pub fn new() -> Self {
        Self {
            props: PropMap::new(),
        }
    }

    /// Returns the resolved value for `prop`, if present.
    #[must_use]
    pub fn get(&self, prop: StyleProp) -> Option<&StyleValue> {
        self.props.get(&prop)
    }

    /// Returns `true` if `prop` has a value.
    #[must_use]
    pub fn contains(&self, prop: StyleProp) -> bool {
        self.props.contains_key(&prop)
    }

    /// Iterates over the resolved properties in deterministic key order.
    pub fn iter(&self) -> impl Iterator<Item = (&StyleProp, &StyleValue)> {
        self.props.iter()
    }

    /// Returns the number of resolved properties.
    #[must_use]
    pub fn len(&self) -> usize {
        self.props.len()
    }

    /// Returns `true` if there are no resolved properties.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }
}

/// A cascade resolver bound to a style sheet and token store.
#[derive(Clone, Copy, Debug)]
pub struct Cascade<'a> {
    /// The style sheet classes are looked up in.
    pub sheet: &'a StyleSheet,
    /// The token store used to resolve token references.
    pub tokens: &'a TokenStore,
}

impl<'a> Cascade<'a> {
    /// Creates a cascade over the given style sheet and token store.
    #[must_use]
    pub fn new(sheet: &'a StyleSheet, tokens: &'a TokenStore) -> Self {
        Self { sheet, tokens }
    }

    /// Resolves `class_names` against `ctx` into a [`ComputedStyle`].
    ///
    /// See the [module documentation](crate::cascade) for the layering rules.
    ///
    /// # Errors
    ///
    /// Returns a [`StyleError`] if any applied value references a token that is
    /// unknown or forms a reference cycle.
    pub fn resolve(
        &self,
        class_names: &[&str],
        ctx: &MatchContext,
    ) -> Result<ComputedStyle, StyleError> {
        resolve(self.sheet, self.tokens, class_names, ctx)
    }
}

/// Resolves an ordered class list into a [`ComputedStyle`].
///
/// This is the free-function form of [`Cascade::resolve`].
///
/// # Errors
///
/// Returns a [`StyleError`] if any applied value references a token that is
/// unknown or forms a reference cycle.
pub fn resolve(
    sheet: &StyleSheet,
    tokens: &TokenStore,
    class_names: &[&str],
    ctx: &MatchContext,
) -> Result<ComputedStyle, StyleError> {
    let mut merged: PropMap = BTreeMap::new();

    // Layer 1: base properties, in class-list order.
    for name in class_names {
        if let Some(class) = sheet.get(name) {
            apply(&mut merged, &class.base);
        }
    }

    // Layer 2: matching breakpoints, ascending min-width, then class-list
    // order. `Base` has a `min_width` of `0.0`, so it always matches and is
    // applied first.
    for breakpoint in Breakpoint::ALL {
        if !breakpoint.matches(ctx.viewport_width) {
            continue;
        }
        for name in class_names {
            if let Some(class) = sheet.get(name)
                && let Some(overrides) = class.breakpoints.get(&breakpoint)
            {
                apply(&mut merged, overrides);
            }
        }
    }

    // Layer 3: matching interaction states, in priority order.
    for state in InteractionState::ALL {
        if !ctx.states.contains(state) {
            continue;
        }
        for name in class_names {
            if let Some(class) = sheet.get(name)
                && let Some(overrides) = class.states.get(&state)
            {
                apply(&mut merged, overrides);
            }
        }
    }

    // Resolve every token reference into a literal value.
    let mut computed = ComputedStyle::new();
    for (prop, value) in &merged {
        computed.props.insert(*prop, tokens.resolve_value(value)?);
    }
    Ok(computed)
}

fn apply(target: &mut PropMap, source: &PropMap) {
    for (prop, value) in source {
        target.insert(*prop, value.clone());
    }
}
