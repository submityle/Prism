//! Field identity and the per-field reactive state the form owns.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;
use core::fmt;

use prism_ui_reactive::Signal;

use crate::validator::BoxedValidator;

/// An opaque, cheaply clonable key identifying a field within a form.
///
/// Internally a reference-counted string, so cloning is a pointer bump and the
/// key is usable as an ordered map key.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FieldId(Rc<str>);

impl FieldId {
    /// Create a field key from any string-like value.
    pub fn new(key: impl AsRef<str>) -> Self {
        Self(Rc::from(key.as_ref()))
    }

    /// Borrow the key as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The anonymous placeholder key used by validators before the owning form
    /// assigns the real one.
    pub(crate) fn anonymous() -> Self {
        Self(Rc::from(""))
    }
}

impl From<&str> for FieldId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for FieldId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&String> for FieldId {
    fn from(value: &String) -> Self {
        Self::new(value)
    }
}

impl fmt::Debug for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FieldId({:?})", &self.0)
    }
}

impl fmt::Display for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Parse the trimmed contents of a field as a signed 64-bit integer.
///
/// Surrounding whitespace is ignored. This is the integer accessor used by both
/// the `int_range` validator and the form's typed read helper, so parsing stays
/// consistent across the crate.
pub fn parse_i64(text: &str) -> Result<i64, core::num::ParseIntError> {
    text.trim().parse::<i64>()
}

/// The reactive state the form keeps for one registered field.
pub(crate) struct Field {
    /// The field's text, the single source of truth for its value.
    pub(crate) text: Signal<String>,
    /// Whether the field has been focused/blurred by the user.
    pub(crate) touched: Cell<bool>,
    /// Whether the field's value has been changed through the form.
    pub(crate) dirty: Cell<bool>,
    /// The validators applied, in declaration order.
    pub(crate) validators: Vec<BoxedValidator>,
}
