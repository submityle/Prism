//! Attribute/field metadata attachable to a reflected type (design §11).
//!
//! Reflection alone describes a type's *shape*; editors, validators, and
//! schema tooling also need its *intent*: numeric ranges, documentation,
//! categories, read-only/hidden hints, defaults, and arbitrary custom
//! key/value attributes. [`TypeMetadata`] bundles that per-type, with a
//! [`FieldMetadata`] entry per field, and implements
//! [`TypeData`](crate::TypeData) so it can be attached to a
//! [`TypeRegistration`](crate::TypeRegistration) and queried at runtime through
//! the existing [`TypeRegistry`](crate::TypeRegistry):
//!
//! ```
//! use prism_reflect::prelude::*;
//! use prism_reflect::schema::{AttributeValue, FieldMetadata, TypeMetadata};
//!
//! #[derive(Reflect, Default)]
//! struct Light {
//!     intensity: f32,
//!     id: u64,
//! }
//!
//! let mut registry = TypeRegistry::new();
//! registry.register::<Light>();
//! registry.register_type_data::<Light, _>(
//!     TypeMetadata::new()
//!         .with_docs("A point light.")
//!         .with_field(
//!             FieldMetadata::new("intensity")
//!                 .with_docs("Luminous intensity.")
//!                 .with_category("Appearance")
//!                 .with_range(0.0, 100_000.0),
//!         )
//!         .with_field(FieldMetadata::new("id").readonly(true)),
//! );
//!
//! let meta = registry
//!     .get_with_name(core::any::type_name::<Light>())
//!     .and_then(|r| r.data::<TypeMetadata>())
//!     .unwrap();
//! assert_eq!(meta.field("intensity").unwrap().range(), Some((0.0, 100_000.0)));
//! assert!(meta.field("id").unwrap().is_readonly());
//! let _ = AttributeValue::Bool(true);
//! ```

use crate::type_data::TypeData;
use core::any::Any;
use std::boxed::Box;
use std::string::String;
use std::vec::Vec;

/// A single typed metadata attribute value.
///
/// Used for per-field defaults and arbitrary custom key/value attributes that
/// do not map onto one of the structured [`FieldMetadata`] slots.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AttributeValue {
    /// A boolean attribute.
    Bool(bool),
    /// A signed integer attribute.
    Int(i64),
    /// A floating-point attribute.
    Float(f64),
    /// A text attribute.
    Text(String),
}

/// Metadata describing a single field of a reflected type.
///
/// Build one with [`FieldMetadata::new`] and the chained `with_*`/flag setters,
/// then attach it to a [`TypeMetadata`] via [`TypeMetadata::with_field`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FieldMetadata {
    name: String,
    docs: Option<String>,
    category: Option<String>,
    readonly: bool,
    hidden: bool,
    required: bool,
    range: Option<(f64, f64)>,
    default: Option<AttributeValue>,
    custom: Vec<(String, AttributeValue)>,
}

impl FieldMetadata {
    /// Begin describing the field named `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Attach a documentation string.
    #[must_use]
    pub fn with_docs(mut self, docs: impl Into<String>) -> Self {
        self.docs = Some(docs.into());
        self
    }

    /// Attach an inspector category/grouping label.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Mark (or clear) the field as read-only.
    #[must_use]
    pub fn readonly(mut self, readonly: bool) -> Self {
        self.readonly = readonly;
        self
    }

    /// Mark (or clear) the field as hidden from inspectors.
    #[must_use]
    pub fn hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }

    /// Mark (or clear) the field as required by schema validation.
    #[must_use]
    pub fn required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// Attach an inclusive numeric range constraint (`min..=max`).
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.range = Some((min, max));
        self
    }

    /// Attach a default-value attribute.
    #[must_use]
    pub fn with_default(mut self, default: AttributeValue) -> Self {
        self.default = Some(default);
        self
    }

    /// Attach (or replace) an arbitrary custom key/value attribute.
    #[must_use]
    pub fn with_custom(mut self, key: impl Into<String>, value: AttributeValue) -> Self {
        let key = key.into();
        if let Some(slot) = self.custom.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.custom.push((key, value));
        }
        self
    }

    /// The field name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The documentation string, if any.
    #[must_use]
    pub fn docs(&self) -> Option<&str> {
        self.docs.as_deref()
    }

    /// The inspector category, if any.
    #[must_use]
    pub fn category(&self) -> Option<&str> {
        self.category.as_deref()
    }

    /// Whether the field is read-only.
    #[must_use]
    pub fn is_readonly(&self) -> bool {
        self.readonly
    }

    /// Whether the field is hidden.
    #[must_use]
    pub fn is_hidden(&self) -> bool {
        self.hidden
    }

    /// Whether the field is required by schema validation.
    #[must_use]
    pub fn is_required(&self) -> bool {
        self.required
    }

    /// The inclusive numeric range `(min, max)`, if constrained.
    #[must_use]
    pub fn range(&self) -> Option<(f64, f64)> {
        self.range
    }

    /// The default-value attribute, if any.
    #[must_use]
    pub fn default_value(&self) -> Option<&AttributeValue> {
        self.default.as_ref()
    }

    /// Look up a custom attribute by key.
    #[must_use]
    pub fn custom(&self, key: &str) -> Option<&AttributeValue> {
        self.custom.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

/// Metadata describing a reflected type and its fields.
///
/// Attach it to a registered type with
/// [`TypeRegistry::register_type_data`](crate::TypeRegistry::register_type_data)
/// and query it back through
/// [`TypeRegistration::data`](crate::TypeRegistration::data).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TypeMetadata {
    docs: Option<String>,
    fields: Vec<FieldMetadata>,
    custom: Vec<(String, AttributeValue)>,
}

impl TypeMetadata {
    /// Begin an empty type-metadata record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a type-level documentation string.
    #[must_use]
    pub fn with_docs(mut self, docs: impl Into<String>) -> Self {
        self.docs = Some(docs.into());
        self
    }

    /// Add (or replace, by name) a field metadata entry.
    #[must_use]
    pub fn with_field(mut self, field: FieldMetadata) -> Self {
        if let Some(slot) = self.fields.iter_mut().find(|f| f.name == field.name) {
            *slot = field;
        } else {
            self.fields.push(field);
        }
        self
    }

    /// Attach (or replace) an arbitrary type-level custom attribute.
    #[must_use]
    pub fn with_custom(mut self, key: impl Into<String>, value: AttributeValue) -> Self {
        let key = key.into();
        if let Some(slot) = self.custom.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.custom.push((key, value));
        }
        self
    }

    /// The type-level documentation string, if any.
    #[must_use]
    pub fn docs(&self) -> Option<&str> {
        self.docs.as_deref()
    }

    /// All field metadata entries, in insertion order.
    #[must_use]
    pub fn fields(&self) -> &[FieldMetadata] {
        &self.fields
    }

    /// Look up a field's metadata by name.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&FieldMetadata> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Look up a type-level custom attribute by key.
    #[must_use]
    pub fn custom(&self, key: &str) -> Option<&AttributeValue> {
        self.custom.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

impl TypeData for TypeMetadata {
    fn clone_type_data(&self) -> Box<dyn TypeData> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
