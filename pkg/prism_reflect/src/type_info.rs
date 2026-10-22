//! Static type descriptors cached per type via `OnceLock`.
//!
//! `TypeInfo` is the compile-time-generated, runtime-zero-construction
//! description of a reflected type. It covers the `Struct`, `TupleStruct`, and
//! `Value` leaf kinds plus the M1 container kinds `Enum`, `List`, `Array`,
//! `Map`, and `Set` (design roadmap §22).

use alloc::vec::Vec;

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
#[derive(Debug, Clone)]
pub struct NamedField {
    name: &'static str,
    type_name: &'static str,
    type_info: Option<fn() -> &'static TypeInfo>,
}

impl PartialEq for NamedField {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.type_name == other.type_name
    }
}

impl Eq for NamedField {}

impl NamedField {
    /// Build a named-field descriptor.
    #[must_use]
    pub fn new(name: &'static str, type_name: &'static str) -> Self {
        Self {
            name,
            type_name,
            type_info: None,
        }
    }

    /// Build a named-field descriptor with a lazy accessor to its static type.
    /// The function is stored but never called while the containing descriptor
    /// is initialized, so recursive types cannot re-enter their `OnceLock`.
    #[must_use]
    pub fn with_type_info(
        name: &'static str,
        type_name: &'static str,
        type_info: fn() -> &'static TypeInfo,
    ) -> Self {
        Self {
            name,
            type_name,
            type_info: Some(type_info),
        }
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

    /// Return the static field type descriptor when one was supplied.
    #[must_use]
    pub fn type_info(&self) -> Option<&'static TypeInfo> {
        self.type_info.map(|get| get())
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
#[derive(Debug, Clone)]
pub struct UnnamedField {
    index: usize,
    type_name: &'static str,
    type_info: Option<fn() -> &'static TypeInfo>,
}

impl PartialEq for UnnamedField {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.type_name == other.type_name
    }
}

impl Eq for UnnamedField {}

impl UnnamedField {
    /// Build an unnamed-field descriptor.
    #[must_use]
    pub fn new(index: usize, type_name: &'static str) -> Self {
        Self {
            index,
            type_name,
            type_info: None,
        }
    }

    /// Build a positional-field descriptor with a lazy static type accessor.
    /// The accessor is not invoked during the containing `OnceLock` init.
    #[must_use]
    pub fn with_type_info(
        index: usize,
        type_name: &'static str,
        type_info: fn() -> &'static TypeInfo,
    ) -> Self {
        Self {
            index,
            type_name,
            type_info: Some(type_info),
        }
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

    /// Return the static field type descriptor when one was supplied.
    #[must_use]
    pub fn type_info(&self) -> Option<&'static TypeInfo> {
        self.type_info.map(|get| get())
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
        Self {
            type_name,
            variants,
        }
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
#[derive(Debug, Clone)]
pub struct ListInfo {
    type_name: &'static str,
    item_type_name: &'static str,
    item_type_info: Option<fn() -> &'static TypeInfo>,
}

impl PartialEq for ListInfo {
    fn eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name && self.item_type_name == other.item_type_name
    }
}
impl Eq for ListInfo {}

impl ListInfo {
    /// Build a list descriptor from the list type name and item type name.
    #[must_use]
    pub fn new(type_name: &'static str, item_type_name: &'static str) -> Self {
        Self {
            type_name,
            item_type_name,
            item_type_info: None,
        }
    }

    /// Build a list descriptor with a lazy item type accessor.
    #[must_use]
    pub fn with_item_type_info(
        type_name: &'static str,
        item_type_name: &'static str,
        item_type_info: fn() -> &'static TypeInfo,
    ) -> Self {
        Self { type_name, item_type_name, item_type_info: Some(item_type_info) }
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

    /// Return the static item type descriptor when available.
    #[must_use]
    pub fn item_type_info(&self) -> Option<&'static TypeInfo> {
        self.item_type_info.map(|get| get())
    }
}

/// Descriptor for a fixed-length array (`[T; N]`).
#[derive(Debug, Clone)]
pub struct ArrayInfo {
    type_name: &'static str,
    item_type_name: &'static str,
    capacity: usize,
    item_type_info: Option<fn() -> &'static TypeInfo>,
}

impl PartialEq for ArrayInfo {
    fn eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name
            && self.item_type_name == other.item_type_name
            && self.capacity == other.capacity
    }
}
impl Eq for ArrayInfo {}

impl ArrayInfo {
    /// Build an array descriptor from the array/element type names and length.
    #[must_use]
    pub fn new(type_name: &'static str, item_type_name: &'static str, capacity: usize) -> Self {
        Self {
            type_name,
            item_type_name,
            capacity,
            item_type_info: None,
        }
    }

    /// Build an array descriptor with a lazy item type accessor.
    #[must_use]
    pub fn with_item_type_info(
        type_name: &'static str,
        item_type_name: &'static str,
        capacity: usize,
        item_type_info: fn() -> &'static TypeInfo,
    ) -> Self {
        Self { type_name, item_type_name, capacity, item_type_info: Some(item_type_info) }
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

    /// Return the static item type descriptor when available.
    #[must_use]
    pub fn item_type_info(&self) -> Option<&'static TypeInfo> {
        self.item_type_info.map(|get| get())
    }
}

/// Descriptor for a key/value map (`HashMap`/`BTreeMap`).
#[derive(Debug, Clone)]
pub struct MapInfo {
    type_name: &'static str,
    key_type_name: &'static str,
    value_type_name: &'static str,
    key_type_info: Option<fn() -> &'static TypeInfo>,
    value_type_info: Option<fn() -> &'static TypeInfo>,
}

impl PartialEq for MapInfo {
    fn eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name
            && self.key_type_name == other.key_type_name
            && self.value_type_name == other.value_type_name
    }
}
impl Eq for MapInfo {}

impl MapInfo {
    /// Build a map descriptor from the map, key, and value type names.
    #[must_use]
    pub fn new(
        type_name: &'static str,
        key_type_name: &'static str,
        value_type_name: &'static str,
    ) -> Self {
        Self {
            type_name,
            key_type_name,
            value_type_name,
            key_type_info: None,
            value_type_info: None,
        }
    }

    /// Build a map descriptor with lazy key/value type accessors.
    #[must_use]
    pub fn with_type_info(
        type_name: &'static str,
        key_type_name: &'static str,
        value_type_name: &'static str,
        key_type_info: fn() -> &'static TypeInfo,
        value_type_info: fn() -> &'static TypeInfo,
    ) -> Self {
        Self { type_name, key_type_name, value_type_name, key_type_info: Some(key_type_info), value_type_info: Some(value_type_info) }
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

    /// Return the static key type descriptor when available.
    #[must_use]
    pub fn key_type_info(&self) -> Option<&'static TypeInfo> {
        self.key_type_info.map(|get| get())
    }

    /// Return the static value type descriptor when available.
    #[must_use]
    pub fn value_type_info(&self) -> Option<&'static TypeInfo> {
        self.value_type_info.map(|get| get())
    }
}

/// Descriptor for a unique-value set (`HashSet`/`BTreeSet`).
#[derive(Debug, Clone)]
pub struct SetInfo {
    type_name: &'static str,
    value_type_name: &'static str,
    value_type_info: Option<fn() -> &'static TypeInfo>,
}

impl PartialEq for SetInfo {
    fn eq(&self, other: &Self) -> bool {
        self.type_name == other.type_name && self.value_type_name == other.value_type_name
    }
}
impl Eq for SetInfo {}

impl SetInfo {
    /// Build a set descriptor from the set and value type names.
    #[must_use]
    pub fn new(type_name: &'static str, value_type_name: &'static str) -> Self {
        Self { type_name, value_type_name, value_type_info: None }
    }

    /// Build a set descriptor with a lazy value type accessor.
    #[must_use]
    pub fn with_value_type_info(
        type_name: &'static str,
        value_type_name: &'static str,
        value_type_info: fn() -> &'static TypeInfo,
    ) -> Self {
        Self { type_name, value_type_name, value_type_info: Some(value_type_info) }
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

    /// Return the static value type descriptor when available.
    #[must_use]
    pub fn value_type_info(&self) -> Option<&'static TypeInfo> {
        self.value_type_info.map(|get| get())
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
