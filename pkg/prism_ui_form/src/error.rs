//! The error type produced by a failed validation rule.

use alloc::string::String;

use crate::field::FieldId;

/// A single validation failure.
///
/// Each error names the field that failed and carries a human-readable
/// message. Built-in validators construct errors without a field (an anonymous
/// placeholder); the owning form rewrites the field to the real key when it runs
/// the rule, so callers always observe a correctly tagged error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationError {
    /// The field this error applies to.
    pub field: FieldId,
    /// A human-readable description of what went wrong.
    pub message: String,
}

impl ValidationError {
    /// Construct an error for `field` with the given `message`.
    pub fn new(field: FieldId, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }

    /// Construct an error that is not yet associated with a field.
    ///
    /// The owning form fills in the field key before the error is surfaced to a
    /// caller.
    pub(crate) fn message_only(message: impl Into<String>) -> Self {
        Self {
            field: FieldId::anonymous(),
            message: message.into(),
        }
    }
}
