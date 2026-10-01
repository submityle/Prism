//! Controls and the mutable argument set that drive a story's rendering.
//!
//! A [`ControlValue`] is a single, typed, user-tweakable parameter — the
//! workbench analogue of a Storybook "control". An [`ArgSet`] is an ordered,
//! type-keyed bag of those values that a story reads while it renders; changing
//! a value in the set and re-rendering is how a story's different states are
//! explored.
//!
//! [`ArgSet::set`] is deliberately *type-preserving*: it updates an existing
//! entry only when the replacement has the same [`ControlKind`], and for a
//! [`ControlValue::Select`] it additionally requires the chosen option to be
//! one of that control's declared options. This mirrors how a real control
//! panel constrains edits so a story never observes an impossible argument.

#![forbid(unsafe_code)]

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use alloc::collections::BTreeMap;

/// The discriminant of a [`ControlValue`], used for type checking without
/// inspecting the payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ControlKind {
    /// A boolean toggle control.
    Bool,
    /// A numeric control carrying an `f64`.
    Number,
    /// A free-form text control.
    Text,
    /// A single-choice control constrained to a fixed option list.
    Select,
}

impl ControlKind {
    /// A short, stable, human-readable label for the kind.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            ControlKind::Bool => "Bool",
            ControlKind::Number => "Number",
            ControlKind::Text => "Text",
            ControlKind::Select => "Select",
        }
    }
}

impl fmt::Display for ControlKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// A single typed control value backing one story argument.
#[derive(Clone, Debug, PartialEq)]
pub enum ControlValue {
    /// A boolean toggle.
    Bool(bool),
    /// A numeric value.
    Number(f64),
    /// A free-form text value.
    Text(String),
    /// A single selected option drawn from a fixed set of candidates.
    Select {
        /// The currently selected option. Always one of `options`.
        selected: String,
        /// The ordered list of candidate options the user may pick from.
        options: Vec<String>,
    },
}

impl ControlValue {
    /// Builds a validated [`ControlValue::Select`].
    ///
    /// # Errors
    ///
    /// Returns [`ControlError::UnknownOption`] when `selected` is not present in
    /// `options`.
    pub fn select<S, I, O>(selected: S, options: I) -> Result<Self, ControlError>
    where
        S: Into<String>,
        I: IntoIterator<Item = O>,
        O: Into<String>,
    {
        let selected = selected.into();
        let options: Vec<String> = options.into_iter().map(Into::into).collect();
        if options.contains(&selected) {
            Ok(ControlValue::Select { selected, options })
        } else {
            Err(ControlError::UnknownOption {
                option: selected,
                options,
            })
        }
    }

    /// The [`ControlKind`] of this value.
    #[must_use]
    pub fn kind(&self) -> ControlKind {
        match self {
            ControlValue::Bool(_) => ControlKind::Bool,
            ControlValue::Number(_) => ControlKind::Number,
            ControlValue::Text(_) => ControlKind::Text,
            ControlValue::Select { .. } => ControlKind::Select,
        }
    }

    /// Reads the boolean payload, or `None` when this is not a [`ControlValue::Bool`].
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ControlValue::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Reads the numeric payload, or `None` when this is not a [`ControlValue::Number`].
    #[must_use]
    pub fn as_number(&self) -> Option<f64> {
        match self {
            ControlValue::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// Reads the text payload, or `None` when this is not a [`ControlValue::Text`].
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ControlValue::Text(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Reads the selected option, or `None` when this is not a [`ControlValue::Select`].
    #[must_use]
    pub fn as_selected(&self) -> Option<&str> {
        match self {
            ControlValue::Select { selected, .. } => Some(selected.as_str()),
            _ => None,
        }
    }

    /// Borrows the candidate options, or `None` when this is not a [`ControlValue::Select`].
    #[must_use]
    pub fn options(&self) -> Option<&[String]> {
        match self {
            ControlValue::Select { options, .. } => Some(options.as_slice()),
            _ => None,
        }
    }
}

/// An error raised while validating a control edit on an [`ArgSet`].
#[derive(Clone, Debug, PartialEq)]
pub enum ControlError {
    /// No control exists under the given key.
    Missing {
        /// The key that was looked up.
        key: String,
    },
    /// The replacement value's kind differed from the existing control's kind.
    TypeMismatch {
        /// The key being updated.
        key: String,
        /// The kind the existing control holds.
        expected: ControlKind,
        /// The kind of the rejected replacement value.
        found: ControlKind,
    },
    /// A [`ControlValue::Select`] was given an option outside its candidate set.
    UnknownOption {
        /// The rejected option.
        option: String,
        /// The candidate options the control allows.
        options: Vec<String>,
    },
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControlError::Missing { key } => {
                write!(f, "no control registered under key `{key}`")
            }
            ControlError::TypeMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "control `{key}` is a {expected} control but was given a {found} value",
            ),
            ControlError::UnknownOption { option, options } => {
                write!(f, "option `{option}` is not one of [")?;
                for (index, candidate) in options.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "`{candidate}`")?;
                }
                f.write_str("]")
            }
        }
    }
}

impl core::error::Error for ControlError {}

/// An ordered, type-keyed bag of [`ControlValue`]s that a story reads to render.
///
/// Keys are stored in a [`BTreeMap`], so iteration order is stable and sorted —
/// which keeps rendered snapshots deterministic. Use the builder-style
/// [`ArgSet::with`] to seed defaults, [`ArgSet::set`] for type-checked updates,
/// and the typed `*_arg` accessors to read values back.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArgSet {
    values: BTreeMap<String, ControlValue>,
}

impl ArgSet {
    /// Creates an empty argument set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    /// Inserts a default control, returning the updated set (builder style).
    ///
    /// Any existing entry under `key` is replaced unconditionally; this is the
    /// seeding path and performs no type checking. Use [`ArgSet::set`] for
    /// type-preserving edits.
    #[must_use]
    pub fn with(mut self, key: impl Into<String>, value: ControlValue) -> Self {
        self.values.insert(key.into(), value);
        self
    }

    /// Inserts or replaces a control unconditionally, returning the previous value.
    pub fn insert(&mut self, key: impl Into<String>, value: ControlValue) -> Option<ControlValue> {
        self.values.insert(key.into(), value)
    }

    /// Updates an existing control in a type-preserving way.
    ///
    /// # Errors
    ///
    /// * [`ControlError::Missing`] when no control exists under `key`.
    /// * [`ControlError::TypeMismatch`] when `value` has a different
    ///   [`ControlKind`] than the stored control.
    /// * [`ControlError::UnknownOption`] when updating a [`ControlValue::Select`]
    ///   with a `selected` option outside the stored candidate set.
    pub fn set(&mut self, key: impl Into<String>, value: ControlValue) -> Result<(), ControlError> {
        let key = key.into();
        let existing = self
            .values
            .get(&key)
            .ok_or_else(|| ControlError::Missing { key: key.clone() })?;

        let expected = existing.kind();
        let found = value.kind();
        if expected != found {
            return Err(ControlError::TypeMismatch {
                key,
                expected,
                found,
            });
        }

        if let (
            ControlValue::Select { options, .. },
            ControlValue::Select {
                selected: new_selected,
                ..
            },
        ) = (existing, &value)
            && !options.contains(new_selected)
        {
            return Err(ControlError::UnknownOption {
                option: new_selected.clone(),
                options: options.clone(),
            });
        }

        self.values.insert(key, value);
        Ok(())
    }

    /// Convenience for setting a [`ControlValue::Bool`] via [`ArgSet::set`].
    ///
    /// # Errors
    ///
    /// Propagates the errors documented on [`ArgSet::set`].
    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) -> Result<(), ControlError> {
        self.set(key, ControlValue::Bool(value))
    }

    /// Convenience for setting a [`ControlValue::Number`] via [`ArgSet::set`].
    ///
    /// # Errors
    ///
    /// Propagates the errors documented on [`ArgSet::set`].
    pub fn set_number(&mut self, key: impl Into<String>, value: f64) -> Result<(), ControlError> {
        self.set(key, ControlValue::Number(value))
    }

    /// Convenience for setting a [`ControlValue::Text`] via [`ArgSet::set`].
    ///
    /// # Errors
    ///
    /// Propagates the errors documented on [`ArgSet::set`].
    pub fn set_text(
        &mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), ControlError> {
        self.set(key, ControlValue::Text(value.into()))
    }

    /// Selects a new option on an existing [`ControlValue::Select`] control.
    ///
    /// The control's candidate set is preserved; only the selection changes.
    ///
    /// # Errors
    ///
    /// * [`ControlError::Missing`] when no control exists under `key`.
    /// * [`ControlError::TypeMismatch`] when the stored control is not a
    ///   [`ControlValue::Select`].
    /// * [`ControlError::UnknownOption`] when `option` is outside the candidate set.
    pub fn select(
        &mut self,
        key: impl Into<String>,
        option: impl Into<String>,
    ) -> Result<(), ControlError> {
        let key = key.into();
        let existing = self
            .values
            .get(&key)
            .ok_or_else(|| ControlError::Missing { key: key.clone() })?;

        let options = match existing {
            ControlValue::Select { options, .. } => options.clone(),
            other => {
                return Err(ControlError::TypeMismatch {
                    key,
                    expected: ControlKind::Select,
                    found: other.kind(),
                });
            }
        };

        let option = option.into();
        if !options.contains(&option) {
            return Err(ControlError::UnknownOption { option, options });
        }

        self.values.insert(
            key,
            ControlValue::Select {
                selected: option,
                options,
            },
        );
        Ok(())
    }

    /// Borrows the control stored under `key`, if any.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&ControlValue> {
        self.values.get(key)
    }

    /// Reads a boolean argument, or `None` when absent or of another kind.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.values.get(key).and_then(ControlValue::as_bool)
    }

    /// Reads a numeric argument, or `None` when absent or of another kind.
    #[must_use]
    pub fn get_number(&self, key: &str) -> Option<f64> {
        self.values.get(key).and_then(ControlValue::as_number)
    }

    /// Reads a text argument, or `None` when absent or of another kind.
    #[must_use]
    pub fn get_text(&self, key: &str) -> Option<&str> {
        self.values.get(key).and_then(ControlValue::as_text)
    }

    /// Reads the selected option of a select argument, or `None` otherwise.
    #[must_use]
    pub fn get_selected(&self, key: &str) -> Option<&str> {
        self.values.get(key).and_then(ControlValue::as_selected)
    }

    /// Returns `true` when a control exists under `key`.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// The number of controls in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when the set holds no controls.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterates the control keys in sorted order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.values.keys().map(String::as_str)
    }

    /// Iterates the `(key, value)` pairs in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ControlValue)> {
        self.values.iter().map(|(key, value)| (key.as_str(), value))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn select_constructor_validates_membership() {
        let control = ControlValue::select("md", ["sm", "md", "lg"]).expect("md is valid");
        assert_eq!(control.as_selected(), Some("md"));
        assert_eq!(control.options().map(<[String]>::len), Some(3));

        let err = ControlValue::select("xl", ["sm", "md", "lg"]).unwrap_err();
        assert!(matches!(err, ControlError::UnknownOption { .. }));
    }

    #[test]
    fn kind_and_typed_accessors() {
        assert_eq!(ControlValue::Bool(true).kind(), ControlKind::Bool);
        assert_eq!(ControlValue::Number(1.5).as_number(), Some(1.5));
        assert_eq!(ControlValue::Text("hi".into()).as_text(), Some("hi"));
        assert_eq!(ControlValue::Bool(true).as_number(), None);
    }

    #[test]
    fn set_preserves_type() {
        let mut args = ArgSet::new().with("flag", ControlValue::Bool(false));
        args.set("flag", ControlValue::Bool(true))
            .expect("same kind");
        assert_eq!(args.get_bool("flag"), Some(true));
    }

    #[test]
    fn set_rejects_type_mismatch() {
        let mut args = ArgSet::new().with("flag", ControlValue::Bool(false));
        let err = args
            .set("flag", ControlValue::Number(1.0))
            .expect_err("kinds differ");
        assert_eq!(
            err,
            ControlError::TypeMismatch {
                key: "flag".into(),
                expected: ControlKind::Bool,
                found: ControlKind::Number,
            }
        );
        // The original value is untouched after a rejected edit.
        assert_eq!(args.get_bool("flag"), Some(false));
    }

    #[test]
    fn set_rejects_missing_key() {
        let mut args = ArgSet::new();
        let err = args
            .set("nope", ControlValue::Bool(true))
            .expect_err("absent key");
        assert_eq!(err, ControlError::Missing { key: "nope".into() });
    }

    #[test]
    fn select_edit_constrained_to_options() {
        let mut args = ArgSet::new().with(
            "size",
            ControlValue::select("sm", ["sm", "lg"]).expect("valid"),
        );

        args.select("size", "lg").expect("lg is an option");
        assert_eq!(args.get_selected("size"), Some("lg"));

        let err = args.select("size", "xl").expect_err("xl not an option");
        assert!(matches!(err, ControlError::UnknownOption { .. }));
        // Rejected select leaves the previous selection intact.
        assert_eq!(args.get_selected("size"), Some("lg"));
    }

    #[test]
    fn iteration_is_sorted_and_stable() {
        let args = ArgSet::new()
            .with("zed", ControlValue::Bool(true))
            .with("alpha", ControlValue::Number(1.0));
        let keys: Vec<&str> = args.keys().collect();
        assert_eq!(keys, ["alpha", "zed"]);
    }

    #[test]
    fn error_display_is_readable() {
        let err = ControlError::UnknownOption {
            option: "xl".into(),
            options: alloc::vec!["sm".into(), "lg".into()],
        };
        let shown = alloc::format!("{err}");
        assert_eq!(shown, "option `xl` is not one of [`sm`, `lg`]");
    }
}
