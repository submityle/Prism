//! Static type descriptors cached per type via `OnceLock`.
//!
//! `TypeInfo` is the compile-time-generated, runtime-zero-construction
//! description of a reflected type. In M0 it covers the `Struct`,
//! `TupleStruct`, and `Value` kinds; the remaining kinds (enum/list/array/
//! map/set/opaque) land in M1 per the design roadmap (§22).

use std::vec::Vec;

/// The reflected shape of a type.
///
/// Each variant carries a descriptor whose field metadata is fixed at
/// `#[derive(Reflect)]` time and cached behind a `OnceLock`, so repeated
/// access returns the same `&'static` with no construction cost.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TypeInfo {
    /// A struct with named fields.
    Struct(StructInfo),
    /// A tuple struct with positional (unnamed) fields.
    TupleStruct(TupleStructInfo),
    /// A leaf value (numeric/bool/string/char/...).
    Value(ValueInfo),
}

impl TypeInfo {
    /// The fully-qualified type name (`core::any::type_name`).
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        match self {
            TypeInfo::Struct(info) => info.type_name(),
            TypeInfo::TupleStruct(info) => info.type_name(),
            TypeInfo::Value(info) => info.type_name(),
        }
    }
}

/// Descriptor for a named-field struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructInfo {
    type_name: &'static str,
    fields: Vec<NamedField>,
}

impl StructInfo {
    /// Build a struct descriptor from its type name and ordered fields.
    #[must_use]
    pub fn new(type_name: &'static str, fields: Vec<NamedField>) -> Self {
        Self { type_name, fields }
    }

    /// The fully-qualified type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The ordered field descriptors (order is a serialization invariant).
    #[must_use]
    pub fn fields(&self) -> &[NamedField] {
        &self.fields
    }

    /// Look up a field descriptor by name.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&NamedField> {
        self.fields.iter().find(|f| f.name() == name)
    }

    /// Number of fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

/// A single named field within a [`StructInfo`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedField {
    name: &'static str,
    type_name: &'static str,
}

impl NamedField {
    /// Build a named-field descriptor.
    #[must_use]
    pub fn new(name: &'static str, type_name: &'static str) -> Self {
        Self { name, type_name }
    }

    /// The field name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The field's fully-qualified type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }
}

/// Descriptor for a tuple struct (positional fields).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TupleStructInfo {
    type_name: &'static str,
    fields: Vec<UnnamedField>,
}

impl TupleStructInfo {
    /// Build a tuple-struct descriptor from its type name and ordered fields.
    #[must_use]
    pub fn new(type_name: &'static str, fields: Vec<UnnamedField>) -> Self {
        Self { type_name, fields }
    }

    /// The fully-qualified type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The ordered field descriptors.
    #[must_use]
    pub fn fields(&self) -> &[UnnamedField] {
        &self.fields
    }

    /// Look up a field descriptor by position.
    #[must_use]
    pub fn field_at(&self, index: usize) -> Option<&UnnamedField> {
        self.fields.get(index)
    }

    /// Number of fields.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

/// A single positional field within a [`TupleStructInfo`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnnamedField {
    index: usize,
    type_name: &'static str,
}

impl UnnamedField {
    /// Build an unnamed-field descriptor.
    #[must_use]
    pub fn new(index: usize, type_name: &'static str) -> Self {
        Self { index, type_name }
    }

    /// The field's positional index.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// The field's fully-qualified type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }
}

/// Descriptor for a leaf value type (numeric/bool/string/char/...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueInfo {
    type_name: &'static str,
}

impl ValueInfo {
    /// Build a value descriptor.
    #[must_use]
    pub fn new(type_name: &'static str) -> Self {
        Self { type_name }
    }

    /// The fully-qualified type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }
}
