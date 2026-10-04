//! Editable-property enumeration for scripts and editors (design §24.6).
//!
//! Where [`PropertyBridge`](super::PropertyBridge) is the *verb* side of the
//! bridge (get/set a property by path), this module is the *noun* side: given a
//! live `&dyn Reflect` root it lists the top-level editable properties as
//! [`PropertyDescriptor`]s, each carrying the display metadata an editor needs
//! to bind a widget (kind, read-only flag, numeric range, docs, category,
//! default) without any hand-written per-type panel.
//!
//! The descriptor list is deliberately shallow: it enumerates the root struct's
//! own fields. A full recursively nested tree is
//! [`inspect`](crate::integration::inspect)'s job; a descriptor's
//! [`path`](PropertyDescriptor::path) is exactly the string
//! [`PropertyBridge`](super::PropertyBridge) accepts, so an editor pairs the two
//! directly.

use alloc::borrow::ToOwned;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::schema::{AttributeValue, FieldMetadata};
use crate::{Reflect, ReflectRef, TypeMetadata, TypeRegistry};

/// The structural kind of an editable property, mirroring [`ReflectRef`] with a
/// finer split of scalar leaves so an editor can pick an appropriate widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PropertyKind {
    /// A boolean leaf (checkbox).
    Bool,
    /// A signed or unsigned integer leaf (integer spinner).
    Integer,
    /// A floating-point leaf (float spinner/slider).
    Float,
    /// A string leaf (text field).
    Text,
    /// A named-field struct (nested panel).
    Struct,
    /// A tuple struct with positional fields.
    TupleStruct,
    /// An enum, editable by variant.
    Enum,
    /// A growable list.
    List,
    /// A fixed-length array.
    Array,
    /// A key/value map.
    Map,
    /// A unique-value set.
    Set,
    /// A leaf value that is not one of the recognised scalar kinds.
    Opaque,
}

impl PropertyKind {
    /// Whether this kind is one of the four scalar leaves exchangeable as an
    /// [`AttributeValue`] through
    /// [`PropertyBridge::read_scalar`](super::PropertyBridge::read_scalar).
    #[must_use]
    pub fn is_scalar(self) -> bool {
        matches!(
            self,
            PropertyKind::Bool | PropertyKind::Integer | PropertyKind::Float | PropertyKind::Text
        )
    }
}

/// A single editable property of a reflected value (design §24.6).
///
/// An editor renders one widget per descriptor and reads/writes it through a
/// [`PropertyBridge`](super::PropertyBridge) using [`path`](Self::path); a
/// scripting runtime enumerates the same list to discover a type's surface.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyDescriptor {
    /// The property's field name relative to the root.
    pub name: String,
    /// The access path accepted by [`PropertyBridge`](super::PropertyBridge).
    ///
    /// For a top-level field this equals [`name`](Self::name); it is a distinct
    /// field so nested descriptors (should the surface grow) stay addressable.
    pub path: String,
    /// The represented field type name.
    pub type_name: String,
    /// The structural/scalar kind of the property.
    pub kind: PropertyKind,
    /// Whether the field is read-only (editor shows it disabled).
    pub readonly: bool,
    /// Inclusive numeric range `(min, max)` for a slider/clamp, if any.
    pub range: Option<(f64, f64)>,
    /// Documentation/tooltip text, if any.
    pub docs: Option<String>,
    /// Grouping category label, if any.
    pub category: Option<String>,
    /// The field's default attribute value, if the metadata declared one.
    pub default: Option<AttributeValue>,
}

/// Enumerate the top-level editable properties of `root`, resolving per-field
/// metadata through `registry` (design §24.6).
///
/// Only a named-field struct exposes properties; any other kind (tuple struct,
/// enum, collection, or leaf value) yields an empty list, since it has no named
/// top-level fields to bind. Fields flagged
/// [`hidden`](FieldMetadata::is_hidden) in the owning type's [`TypeMetadata`]
/// are omitted so an editor never shows an engine-private field.
#[must_use]
pub fn properties(root: &dyn Reflect, registry: &TypeRegistry) -> Vec<PropertyDescriptor> {
    let ReflectRef::Struct(structure) = root.reflect_ref() else {
        return Vec::new();
    };
    let meta = type_metadata(root, registry);
    let mut out = Vec::new();
    for i in 0..structure.field_count() {
        let Some(name) = structure.name_at(i) else {
            continue;
        };
        let Some(field) = structure.field_at(i) else {
            continue;
        };
        let field_meta = meta.and_then(|m| m.field(name));
        if field_meta.is_some_and(FieldMetadata::is_hidden) {
            continue;
        }
        out.push(PropertyDescriptor {
            name: name.to_string(),
            path: name.to_string(),
            type_name: field.type_name().to_string(),
            kind: kind_of(field),
            readonly: field_meta.is_some_and(FieldMetadata::is_readonly),
            range: field_meta.and_then(FieldMetadata::range),
            docs: field_meta.and_then(|m| m.docs().map(ToOwned::to_owned)),
            category: field_meta.and_then(|m| m.category().map(ToOwned::to_owned)),
            default: field_meta.and_then(|m| m.default_value().cloned()),
        });
    }
    out
}

/// Resolve a value's registered [`TypeMetadata`], if any.
fn type_metadata<'r>(value: &dyn Reflect, registry: &'r TypeRegistry) -> Option<&'r TypeMetadata> {
    registry
        .get_with_name(value.type_name())
        .and_then(|reg| reg.data::<TypeMetadata>())
}

/// Classify a reflected value into a [`PropertyKind`], splitting leaf values
/// into their scalar sub-kinds by concrete type.
fn kind_of(value: &dyn Reflect) -> PropertyKind {
    match value.reflect_ref() {
        ReflectRef::Struct(_) => PropertyKind::Struct,
        ReflectRef::TupleStruct(_) => PropertyKind::TupleStruct,
        ReflectRef::Enum(_) => PropertyKind::Enum,
        ReflectRef::List(_) => PropertyKind::List,
        ReflectRef::Array(_) => PropertyKind::Array,
        ReflectRef::Map(_) => PropertyKind::Map,
        ReflectRef::Set(_) => PropertyKind::Set,
        ReflectRef::Value(_) => scalar_kind(value),
    }
}

/// Narrow a leaf value to its scalar [`PropertyKind`] by concrete type.
fn scalar_kind(value: &dyn Reflect) -> PropertyKind {
    let any = value.as_any();
    if any.is::<bool>() {
        return PropertyKind::Bool;
    }
    if any.is::<String>() || any.is::<&'static str>() {
        return PropertyKind::Text;
    }
    if any.is::<f32>() || any.is::<f64>() {
        return PropertyKind::Float;
    }
    macro_rules! is_int {
        ($($ty:ty),* $(,)?) => {
            $(if any.is::<$ty>() { return PropertyKind::Integer; })*
        };
    }
    is_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
    PropertyKind::Opaque
}
