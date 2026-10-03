//! Static type descriptors cached per type via `OnceLock`.
//!
//! `TypeInfo` is the compile-time-generated, runtime-zero-construction
//! description of a reflected type. It covers the `Struct`, `TupleStruct`, and
//! `Value` leaf kinds plus the M1 container kinds `Enum`, `List`, `Array`,
//! `Map`, and `Set` (design roadmap §22).

use std::vec::Vec;

/// The reflected shape of a type.
///
/// Each variant carries a descriptor whose field metadata is fixed at
/// `#[derive(Reflect)]` time (or in the hand-written leaf/container impls) and
/// cached behind a `OnceLock`, so repeated access returns the same `&'static`
/// with no construction cost.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TypeInfo {
    /// A struct with named fields.
    Struct(StructInfo),
    /// A tuple struct with positional (unnamed) fields.
    TupleStruct(TupleStructInfo),
    /// An enum with unit/tuple/struct variants.
    Enum(EnumInfo),
    /// A growable, homogeneous list (`Vec<T>`).
    List(ListInfo),
    /// A fixed-length array (`[T; N]`).
    Array(ArrayInfo),
    /// A key/value map (`HashMap`/`BTreeMap`).
    Map(MapInfo),
    /// A unique-value set (`HashSet`/`BTreeSet`).
    Set(SetInfo),
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
            TypeInfo::Enum(info) => info.type_name(),
            TypeInfo::List(info) => info.type_name(),
            TypeInfo::Array(info) => info.type_name(),
            TypeInfo::Map(info) => info.type_name(),
            TypeInfo::Set(info) => info.type_name(),
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

/// A single positional field within a [`TupleStructInfo`] or tuple variant.
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

/// Descriptor for an enum type and its ordered variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumInfo {
    type_name: &'static str,
    variants: Vec<VariantInfo>,
}

impl EnumInfo {
    /// Build an enum descriptor from its type name and ordered variants.
    #[must_use]
    pub fn new(type_name: &'static str, variants: Vec<VariantInfo>) -> Self {
        Self { type_name, variants }
    }

    /// The fully-qualified type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The ordered variant descriptors.
    #[must_use]
    pub fn variants(&self) -> &[VariantInfo] {
        &self.variants
    }

    /// Look up a variant descriptor by name.
    #[must_use]
    pub fn variant(&self, name: &str) -> Option<&VariantInfo> {
        self.variants.iter().find(|v| v.name() == name)
    }

    /// Look up a variant descriptor by declaration index.
    #[must_use]
    pub fn variant_at(&self, index: usize) -> Option<&VariantInfo> {
        self.variants.get(index)
    }

    /// Number of variants.
    #[must_use]
    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }
}

/// Descriptor for a single enum variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantInfo {
    name: &'static str,
    index: usize,
    kind: VariantKind,
}

impl VariantInfo {
    /// Build a variant descriptor from its name, declaration index, and shape.
    #[must_use]
    pub fn new(name: &'static str, index: usize, kind: VariantKind) -> Self {
        Self { name, index, kind }
    }

    /// The variant name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The variant's declaration index.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// The variant's shape (unit/tuple/struct) and field metadata.
    #[must_use]
    pub fn kind(&self) -> &VariantKind {
        &self.kind
    }

    /// Number of fields carried by the variant.
    #[must_use]
    pub fn field_count(&self) -> usize {
        match &self.kind {
            VariantKind::Unit => 0,
            VariantKind::Tuple(fields) => fields.len(),
            VariantKind::Struct(fields) => fields.len(),
        }
    }
}

/// The shape of an enum variant together with its field metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum VariantKind {
    /// A unit variant with no fields (e.g. `None`).
    Unit,
    /// A tuple variant with positional fields (e.g. `Some(T)`).
    Tuple(Vec<UnnamedField>),
    /// A struct variant with named fields.
    Struct(Vec<NamedField>),
}

/// Descriptor for a growable, homogeneous list (`Vec<T>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListInfo {
    type_name: &'static str,
    item_type_name: &'static str,
}

impl ListInfo {
    /// Build a list descriptor from the list type name and item type name.
    #[must_use]
    pub fn new(type_name: &'static str, item_type_name: &'static str) -> Self {
        Self { type_name, item_type_name }
    }

    /// The fully-qualified list type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The fully-qualified element type name.
    #[must_use]
    pub fn item_type_name(&self) -> &'static str {
        self.item_type_name
    }
}

/// Descriptor for a fixed-length array (`[T; N]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayInfo {
    type_name: &'static str,
    item_type_name: &'static str,
    capacity: usize,
}

impl ArrayInfo {
    /// Build an array descriptor from the array/element type names and length.
    #[must_use]
    pub fn new(type_name: &'static str, item_type_name: &'static str, capacity: usize) -> Self {
        Self { type_name, item_type_name, capacity }
    }

    /// The fully-qualified array type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The fully-qualified element type name.
    #[must_use]
    pub fn item_type_name(&self) -> &'static str {
        self.item_type_name
    }

    /// The compile-time array length `N`.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Descriptor for a key/value map (`HashMap`/`BTreeMap`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapInfo {
    type_name: &'static str,
    key_type_name: &'static str,
    value_type_name: &'static str,
}

impl MapInfo {
    /// Build a map descriptor from the map, key, and value type names.
    #[must_use]
    pub fn new(
        type_name: &'static str,
        key_type_name: &'static str,
        value_type_name: &'static str,
    ) -> Self {
        Self { type_name, key_type_name, value_type_name }
    }

    /// The fully-qualified map type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The fully-qualified key type name.
    #[must_use]
    pub fn key_type_name(&self) -> &'static str {
        self.key_type_name
    }

    /// The fully-qualified value type name.
    #[must_use]
    pub fn value_type_name(&self) -> &'static str {
        self.value_type_name
    }
}

/// Descriptor for a unique-value set (`HashSet`/`BTreeSet`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetInfo {
    type_name: &'static str,
    value_type_name: &'static str,
}

impl SetInfo {
    /// Build a set descriptor from the set and value type names.
    #[must_use]
    pub fn new(type_name: &'static str, value_type_name: &'static str) -> Self {
        Self { type_name, value_type_name }
    }

    /// The fully-qualified set type name.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The fully-qualified value type name.
    #[must_use]
    pub fn value_type_name(&self) -> &'static str {
        self.value_type_name
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
