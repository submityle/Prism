//! Editor inspector model: turn any `&dyn Reflect` into a serializable tree of
//! fields that an editor can render as a property panel, pulling display hints
//! (docs, category, numeric range, read-only/hidden flags) from the registered
//! [`TypeMetadata`](crate::TypeMetadata) when present.
//!
//! This is the reflection half of design §24.6's "脚本与编辑器属性桥": it does
//! not draw any user interface (UI); it produces a backend-agnostic
//! [`InspectorNode`] tree so a downstream editor crate can bind widgets without
//! hand-writing per-type panels.

use alloc::borrow::ToOwned;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::schema::FieldMetadata;
use crate::{Reflect, ReflectRef, TypeMetadata, TypeRegistry};

/// The structural category of an [`InspectorNode`], mirroring [`ReflectRef`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectorKind {
    /// A named-field struct.
    Struct,
    /// A tuple struct with positional fields.
    TupleStruct,
    /// An enum, labelled with its active variant.
    Enum,
    /// A growable list.
    List,
    /// A fixed-length array.
    Array,
    /// A key/value map.
    Map,
    /// A unique-value set.
    Set,
    /// A leaf value with no reflectable children.
    Value,
}

/// Inspector display hints resolved from a field's [`FieldMetadata`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InspectorHints {
    /// Documentation/tooltip text.
    pub docs: Option<String>,
    /// Grouping category label.
    pub category: Option<String>,
    /// Whether the field should be shown but not edited.
    pub readonly: bool,
    /// Inclusive numeric range `(min, max)` for a slider/clamp.
    pub range: Option<(f64, f64)>,
}

impl InspectorHints {
    fn from_metadata(meta: &FieldMetadata) -> Self {
        Self {
            docs: meta.docs().map(ToOwned::to_owned),
            category: meta.category().map(ToOwned::to_owned),
            readonly: meta.is_readonly(),
            range: meta.range(),
        }
    }
}

/// One node in an inspector tree.
///
/// `label` names the node relative to its parent (a field name, a tuple index,
/// a list index, or a map-key rendering). `type_name` is the represented type.
/// Leaf [`InspectorKind::Value`] nodes carry a `value` string rendering;
/// container nodes carry `children`.
#[derive(Debug, Clone, PartialEq)]
pub struct InspectorNode {
    /// The node label relative to its parent.
    pub label: String,
    /// The represented type name.
    pub type_name: String,
    /// The structural kind of the node.
    pub kind: InspectorKind,
    /// For an enum node, the active variant name.
    pub variant: Option<String>,
    /// For a leaf value, a human-readable rendering of the value.
    pub value: Option<String>,
    /// Display hints resolved from field metadata.
    pub hints: InspectorHints,
    /// Child nodes for container kinds.
    pub children: Vec<InspectorNode>,
}

/// Build an inspector tree for `root`, resolving field hints through
/// `registry`. Fields flagged [`hidden`](FieldMetadata::is_hidden) in the
/// owning type's [`TypeMetadata`] are omitted from the tree.
#[must_use]
pub fn inspect(root: &dyn Reflect, registry: &TypeRegistry) -> InspectorNode {
    build(root, "root".to_string(), InspectorHints::default(), registry)
}

fn type_metadata<'r>(value: &dyn Reflect, registry: &'r TypeRegistry) -> Option<&'r TypeMetadata> {
    registry
        .get_with_name(value.type_name())
        .and_then(|reg| reg.data::<TypeMetadata>())
}

fn build(
    value: &dyn Reflect,
    label: String,
    hints: InspectorHints,
    registry: &TypeRegistry,
) -> InspectorNode {
    let type_name = value.type_name().to_string();
    match value.reflect_ref() {
        ReflectRef::Struct(s) => {
            let meta = type_metadata(value, registry);
            let mut children = Vec::new();
            for i in 0..s.field_count() {
                let Some(name) = s.name_at(i) else { continue };
                let Some(field) = s.field_at(i) else { continue };
                let field_meta = meta.and_then(|m| m.field(name));
                if field_meta.is_some_and(FieldMetadata::is_hidden) {
                    continue;
                }
                let field_hints = field_meta.map(InspectorHints::from_metadata).unwrap_or_default();
                children.push(build(field, name.to_string(), field_hints, registry));
            }
            InspectorNode {
                label,
                type_name,
                kind: InspectorKind::Struct,
                variant: None,
                value: None,
                hints,
                children,
            }
        }
        ReflectRef::TupleStruct(ts) => {
            let mut children = Vec::new();
            for i in 0..ts.field_count() {
                if let Some(field) = ts.field(i) {
                    children.push(build(field, i.to_string(), InspectorHints::default(), registry));
                }
            }
            InspectorNode {
                label,
                type_name,
                kind: InspectorKind::TupleStruct,
                variant: None,
                value: None,
                hints,
                children,
            }
        }
        ReflectRef::Enum(e) => {
            let mut children = Vec::new();
            for i in 0..e.field_count() {
                if let Some(field) = e.field_at(i) {
                    children.push(build(field, i.to_string(), InspectorHints::default(), registry));
                }
            }
            InspectorNode {
                label,
                type_name,
                kind: InspectorKind::Enum,
                variant: Some(e.variant_name().to_string()),
                value: None,
                hints,
                children,
            }
        }
        ReflectRef::List(list) => {
            let children = list
                .iter_reflect()
                .enumerate()
                .map(|(i, item)| build(item, i.to_string(), InspectorHints::default(), registry))
                .collect();
            InspectorNode {
                label,
                type_name,
                kind: InspectorKind::List,
                variant: None,
                value: None,
                hints,
                children,
            }
        }
        ReflectRef::Array(array) => {
            let children = array
                .iter_reflect()
                .enumerate()
                .map(|(i, item)| build(item, i.to_string(), InspectorHints::default(), registry))
                .collect();
            InspectorNode {
                label,
                type_name,
                kind: InspectorKind::Array,
                variant: None,
                value: None,
                hints,
                children,
            }
        }
        ReflectRef::Map(map) => {
            let children = map
                .iter_reflect()
                .map(|(k, v)| build(v, leaf_render(k), InspectorHints::default(), registry))
                .collect();
            InspectorNode {
                label,
                type_name,
                kind: InspectorKind::Map,
                variant: None,
                value: None,
                hints,
                children,
            }
        }
        ReflectRef::Set(set) => {
            let children = set
                .iter_reflect()
                .map(|item| {
                    let rendered = leaf_render(item);
                    build(item, rendered, InspectorHints::default(), registry)
                })
                .collect();
            InspectorNode {
                label,
                type_name,
                kind: InspectorKind::Set,
                variant: None,
                value: None,
                hints,
                children,
            }
        }
        ReflectRef::Value(v) => InspectorNode {
            label,
            type_name,
            kind: InspectorKind::Value,
            variant: None,
            value: Some(leaf_render(v)),
            hints,
            children: Vec::new(),
        },
    }
}

/// Render a leaf value to a short display string via its reflected debug form.
fn leaf_render(value: &dyn Reflect) -> String {
    format!("{:?}", DebugLeaf(value))
}

struct DebugLeaf<'a>(&'a dyn Reflect);

impl core::fmt::Debug for DebugLeaf<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // A reflected leaf is almost always a std scalar/string whose `Any`
        // downcast gives the cleanest rendering; fall back to the type name.
        let any = self.0.as_any();
        macro_rules! try_render {
            ($($ty:ty),* $(,)?) => {$(
                if let Some(v) = any.downcast_ref::<$ty>() {
                    return write!(f, "{v:?}");
                }
            )*};
        }
        try_render!(bool, char, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64, String);
        if let Some(v) = any.downcast_ref::<&'static str>() {
            return write!(f, "{v:?}");
        }
        write!(f, "<{}>", self.0.type_name())
    }
}
