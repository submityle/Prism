//! Reflection-driven editor state for Prism.
//!
//! This crate is the adapter between the type-agnostic editor kernel and the
//! reflection kernel. It owns canonical field paths, path-bearing inspector
//! projections, metadata validation, and reversible field edit commands.
//! It contains no UI, runtime, rendering, or engine-specific dependencies.

#![forbid(unsafe_code)]

mod edit;
mod inspector;
mod path;
mod validation;

pub use edit::{EditorError, FieldEditCommand, ReflectEditor, ReflectEditorContext};
pub use inspector::{project_inspector, InspectorProjection, InspectorProjectionNode};
pub use path::{FieldPath, FieldPathError};
pub use validation::{validate, ValidationIssue, ValidationIssueKind, ValidationReport};
