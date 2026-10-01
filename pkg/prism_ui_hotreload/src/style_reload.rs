//! Hot-swapping stylesheets by diffing resolved class styles.
//!
//! Because a [`StyleSheet`] does not expose its full set of class names, a
//! style diff works over a caller-supplied list of class names. For each name
//! it resolves the old and new [`ComputedStyle`] through the cascade and
//! classifies the result as unchanged, added, removed, or changed. A changed
//! class carries the property-level before and after values so a backend can
//! apply exactly the deltas.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use prism_ui_style::{
    resolve, ComputedStyle, MatchContext, StyleError, StyleProp, StyleSheet, StyleValue, TokenStore,
};

/// A single property value belonging to a class, used in before/after lists.
#[derive(Clone, Debug, PartialEq)]
pub struct PropValue {
    /// The property.
    pub prop: StyleProp,
    /// Its resolved value.
    pub value: StyleValue,
}

/// How a single class name changed between the old and new stylesheet.
#[derive(Clone, Debug, PartialEq)]
pub enum ClassChange {
    /// The class resolves to the same properties in both sheets.
    Unchanged(String),
    /// The class resolves to properties in the new sheet but none in the old.
    Added(String),
    /// The class resolves to properties in the old sheet but none in the new.
    Removed(String),
    /// The class resolves to differing properties.
    Changed {
        /// The class name.
        name: String,
        /// Property values present in the old sheet that differ from the new.
        before: Vec<PropValue>,
        /// Property values present in the new sheet that differ from the old.
        after: Vec<PropValue>,
    },
}

impl ClassChange {
    /// Returns the class name this change refers to.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            ClassChange::Unchanged(name)
            | ClassChange::Added(name)
            | ClassChange::Removed(name)
            | ClassChange::Changed { name, .. } => name,
        }
    }

    /// Returns `true` when this change is not [`ClassChange::Unchanged`].
    #[must_use]
    pub fn is_change(&self) -> bool {
        !matches!(self, ClassChange::Unchanged(_))
    }
}

/// The result of diffing resolved styles for a list of class names.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleDiff {
    /// One entry per input class name, in input order.
    changes: Vec<ClassChange>,
}

impl StyleDiff {
    /// Returns every per-class result, in the order the names were given.
    #[must_use]
    pub fn all(&self) -> &[ClassChange] {
        &self.changes
    }

    /// Returns only the classes that actually changed.
    #[must_use]
    pub fn changed(&self) -> Vec<&ClassChange> {
        self.changes.iter().filter(|c| c.is_change()).collect()
    }

    /// Returns `true` when no class changed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changes.iter().all(|c| !c.is_change())
    }

    /// Renders a stable, human-readable report of the changed classes.
    #[must_use]
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("style diff: {} changed\n", self.changed().len()));
        for change in &self.changes {
            match change {
                ClassChange::Unchanged(_) => {}
                ClassChange::Added(name) => out.push_str(&format!("  + .{name}\n")),
                ClassChange::Removed(name) => out.push_str(&format!("  - .{name}\n")),
                ClassChange::Changed {
                    name,
                    before,
                    after,
                } => {
                    out.push_str(&format!("  ~ .{name}\n"));
                    for item in before {
                        out.push_str(&format!("      - {:?} = {:?}\n", item.prop, item.value));
                    }
                    for item in after {
                        out.push_str(&format!("      + {:?} = {:?}\n", item.prop, item.value));
                    }
                }
            }
        }
        out
    }
}

/// Collects the property values of `style` whose value differs from `other`.
fn differing(style: &ComputedStyle, other: &ComputedStyle) -> Vec<PropValue> {
    let mut out = Vec::new();
    for (prop, value) in style.iter() {
        if other.get(*prop) != Some(value) {
            out.push(PropValue {
                prop: *prop,
                value: value.clone(),
            });
        }
    }
    out
}

/// Classifies one class name given its old and new resolved styles.
fn classify(name: &str, old: &ComputedStyle, new: &ComputedStyle) -> ClassChange {
    match (old.is_empty(), new.is_empty()) {
        (true, true) => ClassChange::Unchanged(name.to_string()),
        (true, false) => ClassChange::Added(name.to_string()),
        (false, true) => ClassChange::Removed(name.to_string()),
        (false, false) => {
            if old == new {
                ClassChange::Unchanged(name.to_string())
            } else {
                ClassChange::Changed {
                    name: name.to_string(),
                    before: differing(old, new),
                    after: differing(new, old),
                }
            }
        }
    }
}

/// Diffs the resolved styles of each name in `names` between two stylesheets.
///
/// Each name is resolved against `old` and `new` using the shared `tokens` and
/// match `ctx`, then classified. Resolution against each sheet passes the
/// single class name, so the diff reflects that class in isolation.
///
/// # Errors
///
/// Returns the first [`StyleError`] produced while resolving any class against
/// either stylesheet (for example an unknown token reference).
pub fn diff_classes(
    old: &StyleSheet,
    new: &StyleSheet,
    tokens: &TokenStore,
    names: &[&str],
    ctx: &MatchContext,
) -> Result<StyleDiff, StyleError> {
    let mut changes = Vec::with_capacity(names.len());
    for name in names {
        let single = [*name];
        let old_style = resolve(old, tokens, &single, ctx)?;
        let new_style = resolve(new, tokens, &single, ctx)?;
        changes.push(classify(name, &old_style, &new_style));
    }
    Ok(StyleDiff { changes })
}

#[cfg(test)]
mod tests {
    use super::{diff_classes, ClassChange};
    use prism_ui_style::{Class, MatchContext, StyleProp, StyleSheet, StyleValue, TokenStore};

    fn ctx() -> MatchContext {
        MatchContext::new(1024.0)
    }

    fn sheet_with(name: &str, prop: StyleProp, value: StyleValue) -> StyleSheet {
        StyleSheet::new().with_class(Class::new(name).with(prop, value))
    }

    #[test]
    fn unchanged_class_reports_no_change() {
        let old = sheet_with("btn", StyleProp::Width, StyleValue::px(10.0));
        let new = sheet_with("btn", StyleProp::Width, StyleValue::px(10.0));
        let diff = diff_classes(&old, &new, &TokenStore::new(), &["btn"], &ctx()).unwrap();
        assert!(diff.is_empty());
        assert!(matches!(diff.all()[0], ClassChange::Unchanged(_)));
    }

    #[test]
    fn added_class_detected() {
        let old = StyleSheet::new();
        let new = sheet_with("btn", StyleProp::Width, StyleValue::px(10.0));
        let diff = diff_classes(&old, &new, &TokenStore::new(), &["btn"], &ctx()).unwrap();
        assert!(matches!(diff.all()[0], ClassChange::Added(_)));
        assert_eq!(diff.changed().len(), 1);
    }

    #[test]
    fn removed_class_detected() {
        let old = sheet_with("btn", StyleProp::Width, StyleValue::px(10.0));
        let new = StyleSheet::new();
        let diff = diff_classes(&old, &new, &TokenStore::new(), &["btn"], &ctx()).unwrap();
        assert!(matches!(diff.all()[0], ClassChange::Removed(_)));
    }

    #[test]
    fn changed_class_lists_before_and_after() {
        let old = sheet_with("btn", StyleProp::Width, StyleValue::px(10.0));
        let new = sheet_with("btn", StyleProp::Width, StyleValue::px(20.0));
        let diff = diff_classes(&old, &new, &TokenStore::new(), &["btn"], &ctx()).unwrap();
        let change = &diff.all()[0];
        assert!(matches!(change, ClassChange::Changed { .. }));
        if let ClassChange::Changed {
            name,
            before,
            after,
        } = change
        {
            assert_eq!(name, "btn");
            assert_eq!(before.len(), 1);
            assert_eq!(after.len(), 1);
            assert_eq!(before[0].value, StyleValue::px(10.0));
            assert_eq!(after[0].value, StyleValue::px(20.0));
        }
    }

    #[test]
    fn changed_class_tracks_added_property() {
        let old = sheet_with("card", StyleProp::Width, StyleValue::px(10.0));
        let new = StyleSheet::new().with_class(
            Class::new("card")
                .with(StyleProp::Width, StyleValue::px(10.0))
                .with(StyleProp::Height, StyleValue::px(5.0)),
        );
        let diff = diff_classes(&old, &new, &TokenStore::new(), &["card"], &ctx()).unwrap();
        let change = &diff.all()[0];
        assert!(matches!(change, ClassChange::Changed { .. }));
        if let ClassChange::Changed { before, after, .. } = change {
            // Width is unchanged, so only Height differs.
            assert!(before.is_empty());
            assert_eq!(after.len(), 1);
            assert_eq!(after[0].prop, StyleProp::Height);
        }
    }

    #[test]
    fn report_and_name_accessor() {
        let old = sheet_with("a", StyleProp::Width, StyleValue::px(1.0));
        let new = sheet_with("a", StyleProp::Width, StyleValue::px(2.0));
        let diff = diff_classes(&old, &new, &TokenStore::new(), &["a", "missing"], &ctx()).unwrap();
        assert_eq!(diff.all().len(), 2);
        assert_eq!(diff.all()[0].name(), "a");
        assert!(diff.report().contains("1 changed"));
    }
}
