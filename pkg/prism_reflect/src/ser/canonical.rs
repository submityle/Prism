//! Bounded canonical writing of the existing PRB1 binary format.
//!
//! Map keys and set elements are ordered by their canonical node bytes (without
//! stream headers). Other children retain schema order. No reflected values
//! are cloned, and lists/arrays use ordinary nodes rather than POD blobs.

use super::binary::{BinaryEncoder, MAGIC, VERSION};
use super::encode::Encoder;
use super::primitive::{leaf_primitive, Primitive};
use super::{SerializeError, StableTypeId};
use crate::{
    DynamicEnum, DynamicStruct, DynamicTupleStruct, Reflect, ReflectRef, TypeInfo, TypeRegistry,
    VariantKind, VariantType,
};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Resource limits for canonical serialization, including sorting buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalLimits {
    /// Maximum final stream size, including the 13-byte PRB1 header.
    pub max_bytes: usize,
    /// Maximum node depth, with the root at depth 1 (including leaf nodes).
    pub max_depth: usize,
    /// Maximum fields/elements/entries in any one composite node.
    pub max_collection_len: usize,
    /// Maximum total nodes; each map key and value counts separately.
    pub max_nodes: usize,
}

impl Default for CanonicalLimits {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024,
            max_depth: 64,
            max_collection_len: 65_536,
            max_nodes: 65_536,
        }
    }
}

/// A value cannot be represented canonically within the requested limits.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CanonicalError {
    /// A leaf is outside the PRB1 built-in primitive set.
    Serialize(SerializeError),
    /// The next write would exceed the total byte budget.
    BytesLimitExceeded {
        /// Configured maximum stream size.
        limit: usize,
    },
    /// The next node would exceed the nesting budget.
    DepthLimitExceeded {
        /// Configured maximum node depth.
        limit: usize,
    },
    /// A composite exceeds the per-collection budget.
    CollectionTooLarge {
        /// Configured maximum collection length.
        limit: usize,
        /// Reflected field/element/entry count.
        declared: usize,
    },
    /// A node or its immediate children cannot fit the remaining node budget.
    NodeLimitExceeded {
        /// Configured maximum total node count.
        limit: usize,
    },
    /// NaN and infinities have no canonical representation.
    NonFiniteFloat {
        /// The offending floating-point type.
        type_name: &'static str,
    },
    /// Two map keys produced identical canonical node bytes.
    DuplicateMapKey,
    /// Two set elements produced identical canonical node bytes.
    DuplicateSetElement,
    /// Reflected fields, variant metadata, or iterator length are inconsistent.
    InvalidShape {
        /// The offending reflected type.
        type_name: &'static str,
    },
    /// A declared nonprimitive child schema is absent from the registry.
    UnregisteredType {
        /// The declared target type requiring registration.
        type_name: &'static str,
    },
    /// A leaf has a different concrete primitive type than the target schema.
    PrimitiveMismatch {
        /// Required primitive type.
        expected: &'static str,
        /// Reflected type of the supplied value.
        actual: &'static str,
    },
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serialize(error) => error.fmt(f),
            Self::BytesLimitExceeded { limit } => {
                write!(f, "canonical binary exceeds byte limit {limit}")
            }
            Self::DepthLimitExceeded { limit } => {
                write!(f, "canonical binary exceeds depth limit {limit}")
            }
            Self::CollectionTooLarge { limit, declared } => {
                write!(
                    f,
                    "canonical collection length {declared} exceeds limit {limit}"
                )
            }
            Self::NodeLimitExceeded { limit } => {
                write!(f, "canonical binary exceeds node limit {limit}")
            }
            Self::NonFiniteFloat { type_name } => {
                write!(f, "cannot canonically serialize non-finite `{type_name}`")
            }
            Self::DuplicateMapKey => write!(f, "duplicate canonical map key"),
            Self::DuplicateSetElement => write!(f, "duplicate canonical set element"),
            Self::InvalidShape { type_name } => {
                write!(f, "inconsistent reflected shape for `{type_name}`")
            }
            Self::UnregisteredType { type_name } => {
                write!(f, "unregistered canonical schema `{type_name}`")
            }
            Self::PrimitiveMismatch { expected, actual } => {
                write!(f, "canonical schema requires `{expected}`, got `{actual}`")
            }
        }
    }
}

impl core::error::Error for CanonicalError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Serialize(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SerializeError> for CanonicalError {
    fn from(error: SerializeError) -> Self {
        Self::Serialize(error)
    }
}

/// Write a bounded canonical PRB1 stream for a reflected value.
///
/// Typed structs, tuple structs, and enum payloads follow their reflection
/// schema. Dynamic structs/tuples/enums, which have no populated static
/// schema, retain their reflected field order. Map keys and set elements sort
/// lexicographically by canonical node bytes, rejecting collisions. Floating
/// negative zero becomes positive zero; non-finite floats are rejected.
///
/// # Errors
/// Returns [`CanonicalError`] for resource limits, unsupported leaves,
/// non-finite floats, duplicate canonical keys/elements, or inconsistent
/// reflection implementations. Limits are checked before writing bytes or
/// processing children. Sorting buffers share the stream's byte/node budget.
pub fn to_canonical_binary(
    value: &dyn Reflect,
    limits: CanonicalLimits,
) -> Result<Vec<u8>, CanonicalError> {
    write_binary(value, value.type_name(), None, limits)
}

/// Write a migrated reflected tree using an explicit target schema.
///
/// The header uses `info.type_name()`. Runtime type names, including those of
/// nested dynamic values, need not match the target. Named fields are read by
/// target name in target order; enum variants are selected by target name and
/// written with the target index. Every node must match the declared kind and
/// shape, and primitive leaves must have the exact concrete primitive type.
///
/// Declared nonprimitive children are resolved through `registry`, including
/// the element schemas of empty collections. Primitive schemas need no
/// registration. This shares the ordinary writer's limits and sorting rules.
///
/// # Errors
/// Returns [`CanonicalError`] for invalid target shapes or primitive types,
/// unregistered child schemas, or any ordinary canonical serialization error.
pub fn to_canonical_binary_with_schema(
    value: &dyn Reflect,
    info: &'static TypeInfo,
    registry: &TypeRegistry,
    limits: CanonicalLimits,
) -> Result<Vec<u8>, CanonicalError> {
    write_binary(
        value,
        info.type_name(),
        Some(Schema::new(info, registry)),
        limits,
    )
}

#[derive(Clone, Copy)]
enum Declared {
    Primitive(Primitive),
    Info(&'static TypeInfo),
}

#[derive(Clone, Copy)]
struct Schema<'a> {
    declared: Declared,
    registry: &'a TypeRegistry,
}

impl<'a> Schema<'a> {
    fn new(info: &'static TypeInfo, registry: &'a TypeRegistry) -> Self {
        let declared = match info {
            TypeInfo::Value(_) => leaf_primitive(info.type_name())
                .map(Declared::Primitive)
                .unwrap_or(Declared::Info(info)),
            _ => Declared::Info(info),
        };
        Self { declared, registry }
    }

    fn child(self, type_name: &'static str) -> Result<Self, CanonicalError> {
        if let Some(primitive) = leaf_primitive(type_name) {
            return Ok(Self {
                declared: Declared::Primitive(primitive),
                ..self
            });
        }
        let info = self
            .registry
            .get_with_name(type_name)
            .ok_or(CanonicalError::UnregisteredType { type_name })?
            .type_info();
        Ok(Self::new(info, self.registry))
    }
}

fn child_schema<'a>(
    schema: Option<Schema<'a>>,
    type_name: &'static str,
) -> Result<Option<Schema<'a>>, CanonicalError> {
    schema.map(|schema| schema.child(type_name)).transpose()
}

fn write_binary(
    value: &dyn Reflect,
    root_name: &'static str,
    schema: Option<Schema<'_>>,
    limits: CanonicalLimits,
) -> Result<Vec<u8>, CanonicalError> {
    let mut budget = Budget {
        limits,
        bytes: 0,
        nodes: 0,
    };
    let mut encoder = BinaryEncoder::new();
    budget.charge_bytes(MAGIC.len() + 1 + 8)?;
    encoder.out.extend_from_slice(&MAGIC);
    encoder.out.push(VERSION);
    let root_id = StableTypeId::of_path(root_name).value();
    encoder.out.extend_from_slice(&root_id.to_le_bytes());
    budget.node(value, &mut encoder, 1, schema)?;
    Ok(encoder.out)
}

struct Budget {
    limits: CanonicalLimits,
    // Includes bytes already written into any detached sorting buffer. Moving
    // those bytes to their parent must not charge them a second time.
    bytes: usize,
    nodes: usize,
}

impl Budget {
    fn charge_bytes(&mut self, count: usize) -> Result<(), CanonicalError> {
        if count > self.limits.max_bytes - self.bytes {
            return Err(CanonicalError::BytesLimitExceeded {
                limit: self.limits.max_bytes,
            });
        }
        self.bytes += count;
        Ok(())
    }

    fn check_depth(&self, depth: usize) -> Result<(), CanonicalError> {
        if depth > self.limits.max_depth {
            return Err(CanonicalError::DepthLimitExceeded {
                limit: self.limits.max_depth,
            });
        }
        Ok(())
    }

    // Reject impossible immediate children before obtaining/iterating any of
    // them or allocating sorting storage. A map entry needs at least 2 nodes.
    fn children(&self, len: usize, per_item: usize, depth: usize) -> Result<(), CanonicalError> {
        if len > self.limits.max_collection_len {
            return Err(CanonicalError::CollectionTooLarge {
                limit: self.limits.max_collection_len,
                declared: len,
            });
        }
        if len > 0 {
            if depth >= self.limits.max_depth {
                return Err(CanonicalError::DepthLimitExceeded {
                    limit: self.limits.max_depth,
                });
            }
            if len > (self.limits.max_nodes - self.nodes) / per_item {
                return Err(CanonicalError::NodeLimitExceeded {
                    limit: self.limits.max_nodes,
                });
            }
        }
        Ok(())
    }

    fn node(
        &mut self,
        value: &dyn Reflect,
        encoder: &mut BinaryEncoder,
        depth: usize,
        schema: Option<Schema<'_>>,
    ) -> Result<(), CanonicalError> {
        self.check_depth(depth)?;
        if self.nodes == self.limits.max_nodes {
            return Err(CanonicalError::NodeLimitExceeded {
                limit: self.limits.max_nodes,
            });
        }
        self.nodes += 1;
        let target_info = match schema.map(|schema| schema.declared) {
            Some(Declared::Info(info)) => Some(info),
            _ => None,
        };
        let invalid = || CanonicalError::InvalidShape {
            type_name: target_info.map_or(value.type_name(), TypeInfo::type_name),
        };
        let reflected = value.reflect_ref();
        if let Some(Declared::Primitive(expected)) = schema.map(|schema| schema.declared) {
            let ReflectRef::Value(leaf) = reflected else {
                return Err(invalid());
            };
            return self.leaf(leaf, encoder, Some(expected));
        }
        if let Some(info) = target_info {
            if !matches!(
                (info, &reflected),
                (TypeInfo::Struct(_), ReflectRef::Struct(_))
                    | (TypeInfo::TupleStruct(_), ReflectRef::TupleStruct(_))
                    | (TypeInfo::Enum(_), ReflectRef::Enum(_))
                    | (TypeInfo::List(_), ReflectRef::List(_))
                    | (TypeInfo::Array(_), ReflectRef::Array(_))
                    | (TypeInfo::Map(_), ReflectRef::Map(_))
                    | (TypeInfo::Set(_), ReflectRef::Set(_))
                    | (TypeInfo::Value(_), ReflectRef::Value(_))
            ) {
                return Err(invalid());
            }
            if let TypeInfo::Value(_) = info {
                return Err(SerializeError::UnsupportedLeaf {
                    type_name: info.type_name(),
                }
                .into());
            }
        }
        match reflected {
            ReflectRef::Struct(s) => {
                let count = s.field_count();
                self.children(count, 1, depth)?;
                self.charge_bytes(1 + varint_len(count))?;
                encoder.begin_struct(count);
                if schema.is_none() && value.as_any().is::<DynamicStruct>() {
                    for index in 0..count {
                        self.node(
                            s.field_at(index).ok_or_else(invalid)?,
                            encoder,
                            depth + 1,
                            None,
                        )?;
                    }
                } else {
                    let TypeInfo::Struct(info) = target_info.unwrap_or_else(|| value.type_info())
                    else {
                        return Err(invalid());
                    };
                    if info.field_count() != count {
                        return Err(invalid());
                    }
                    for field in info.fields() {
                        self.node(
                            s.field(field.name()).ok_or_else(invalid)?,
                            encoder,
                            depth + 1,
                            child_schema(schema, field.type_name())?,
                        )?;
                    }
                }
            }
            ReflectRef::TupleStruct(tuple) => {
                let count = tuple.field_count();
                self.children(count, 1, depth)?;
                self.charge_bytes(1 + varint_len(count))?;
                encoder.begin_tuple_struct(count);
                if schema.is_none() && value.as_any().is::<DynamicTupleStruct>() {
                    for index in 0..count {
                        self.node(
                            tuple.field(index).ok_or_else(invalid)?,
                            encoder,
                            depth + 1,
                            None,
                        )?;
                    }
                } else {
                    let TypeInfo::TupleStruct(info) =
                        target_info.unwrap_or_else(|| value.type_info())
                    else {
                        return Err(invalid());
                    };
                    if info.field_count() != count {
                        return Err(invalid());
                    }
                    for field in info.fields() {
                        self.node(
                            tuple.field(field.index()).ok_or_else(invalid)?,
                            encoder,
                            depth + 1,
                            child_schema(schema, field.type_name())?,
                        )?;
                    }
                }
            }
            ReflectRef::Enum(e) => {
                let count = e.field_count();
                self.children(count, 1, depth)?;
                let variant_type = e.variant_type();
                let variant = if schema.is_none() && value.as_any().is::<DynamicEnum>() {
                    None
                } else {
                    let TypeInfo::Enum(info) = target_info.unwrap_or_else(|| value.type_info())
                    else {
                        return Err(invalid());
                    };
                    let variant = if schema.is_some() {
                        info.variant(e.variant_name())
                    } else {
                        info.variant_at(e.variant_index())
                    }
                    .ok_or_else(invalid)?;
                    if variant.name() != e.variant_name()
                        || (schema.is_none() && variant.index() != e.variant_index())
                        || variant.field_count() != count
                        || !matches!(
                            (variant.kind(), variant_type),
                            (VariantKind::Unit, VariantType::Unit)
                                | (VariantKind::Tuple(_), VariantType::Tuple)
                                | (VariantKind::Struct(_), VariantType::Struct)
                        )
                    {
                        return Err(invalid());
                    }
                    Some(variant)
                };
                let index = variant.map_or(e.variant_index(), |variant| variant.index());
                self.charge_bytes(1 + varint_len(index) + varint_len(count))?;
                encoder.begin_enum(index, e.variant_name(), variant_type, count);
                if let Some(variant) = variant {
                    match variant.kind() {
                        VariantKind::Unit => {}
                        VariantKind::Tuple(fields) => {
                            for field in fields {
                                self.node(
                                    e.field_at(field.index()).ok_or_else(invalid)?,
                                    encoder,
                                    depth + 1,
                                    child_schema(schema, field.type_name())?,
                                )?;
                            }
                        }
                        VariantKind::Struct(fields) => {
                            for field in fields {
                                self.node(
                                    e.field(field.name()).ok_or_else(invalid)?,
                                    encoder,
                                    depth + 1,
                                    child_schema(schema, field.type_name())?,
                                )?;
                            }
                        }
                    }
                } else {
                    for index in 0..count {
                        self.node(
                            e.field_at(index).ok_or_else(invalid)?,
                            encoder,
                            depth + 1,
                            None,
                        )?;
                    }
                }
            }
            ReflectRef::List(list) => {
                let len = list.len();
                self.children(len, 1, depth)?;
                let item_schema = match target_info {
                    Some(TypeInfo::List(info)) => child_schema(schema, info.item_type_name())?,
                    _ => None,
                };
                self.charge_bytes(1 + varint_len(len))?;
                encoder.begin_list(len);
                for index in 0..len {
                    self.node(
                        list.get(index).ok_or_else(invalid)?,
                        encoder,
                        depth + 1,
                        item_schema,
                    )?;
                }
            }
            ReflectRef::Array(array) => {
                let len = array.len();
                self.children(len, 1, depth)?;
                let item_schema = match target_info {
                    Some(TypeInfo::Array(info)) => {
                        if info.capacity() != len {
                            return Err(invalid());
                        }
                        child_schema(schema, info.item_type_name())?
                    }
                    _ => None,
                };
                self.charge_bytes(1 + varint_len(len))?;
                encoder.begin_array(len);
                for index in 0..len {
                    self.node(
                        array.get(index).ok_or_else(invalid)?,
                        encoder,
                        depth + 1,
                        item_schema,
                    )?;
                }
            }
            ReflectRef::Map(map) => {
                let len = map.len();
                self.children(len, 2, depth)?;
                let (key_schema, value_schema) = match target_info {
                    Some(TypeInfo::Map(info)) => (
                        child_schema(schema, info.key_type_name())?,
                        child_schema(schema, info.value_type_name())?,
                    ),
                    _ => (None, None),
                };
                self.charge_bytes(1 + varint_len(len))?;
                encoder.begin_map(len);
                let mut entries = Vec::new();
                for (key, mapped) in map.iter_reflect() {
                    if entries.len() == len {
                        return Err(invalid());
                    }
                    let key_bytes = self.node_bytes(key, depth + 1, key_schema)?;
                    entries.push((key_bytes, mapped));
                }
                if entries.len() != len {
                    return Err(invalid());
                }
                entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                    return Err(CanonicalError::DuplicateMapKey);
                }
                for (mut key_bytes, mapped) in entries {
                    // Already charged while encoding into the sorting buffer.
                    encoder.out.append(&mut key_bytes);
                    self.node(mapped, encoder, depth + 1, value_schema)?;
                }
            }
            ReflectRef::Set(set) => {
                let len = set.len();
                self.children(len, 1, depth)?;
                let item_schema = match target_info {
                    Some(TypeInfo::Set(info)) => child_schema(schema, info.value_type_name())?,
                    _ => None,
                };
                self.charge_bytes(1 + varint_len(len))?;
                encoder.begin_set(len);
                let mut elements = Vec::new();
                for element in set.iter_reflect() {
                    if elements.len() == len {
                        return Err(invalid());
                    }
                    elements.push(self.node_bytes(element, depth + 1, item_schema)?);
                }
                if elements.len() != len {
                    return Err(invalid());
                }
                elements.sort_unstable();
                if elements.windows(2).any(|pair| pair[0] == pair[1]) {
                    return Err(CanonicalError::DuplicateSetElement);
                }
                for mut bytes in elements {
                    encoder.out.append(&mut bytes);
                }
            }
            ReflectRef::Value(leaf) => self.leaf(leaf, encoder, None)?,
        }
        Ok(())
    }

    fn node_bytes(
        &mut self,
        value: &dyn Reflect,
        depth: usize,
        schema: Option<Schema<'_>>,
    ) -> Result<Vec<u8>, CanonicalError> {
        let mut encoder = BinaryEncoder::new();
        self.node(value, &mut encoder, depth, schema)?;
        Ok(encoder.out)
    }

    fn leaf(
        &mut self,
        leaf: &dyn Reflect,
        encoder: &mut BinaryEncoder,
        expected: Option<Primitive>,
    ) -> Result<(), CanonicalError> {
        let any = leaf.as_any();
        if let Some(expected) = expected {
            let matches = match expected {
                Primitive::Bool => any.is::<bool>(),
                Primitive::Char => any.is::<char>(),
                Primitive::I8 => any.is::<i8>(),
                Primitive::I16 => any.is::<i16>(),
                Primitive::I32 => any.is::<i32>(),
                Primitive::I64 => any.is::<i64>(),
                Primitive::I128 => any.is::<i128>(),
                Primitive::Isize => any.is::<isize>(),
                Primitive::U8 => any.is::<u8>(),
                Primitive::U16 => any.is::<u16>(),
                Primitive::U32 => any.is::<u32>(),
                Primitive::U64 => any.is::<u64>(),
                Primitive::U128 => any.is::<u128>(),
                Primitive::Usize => any.is::<usize>(),
                Primitive::F32 => any.is::<f32>(),
                Primitive::F64 => any.is::<f64>(),
                Primitive::String => any.is::<String>(),
            };
            if !matches {
                return Err(CanonicalError::PrimitiveMismatch {
                    expected: expected.type_name(),
                    actual: leaf.type_name(),
                });
            }
        }
        macro_rules! fixed {
            ($($ty:ty => $method:ident, $width:expr);* $(;)?) => {
                $(if let Some(value) = any.downcast_ref::<$ty>() {
                    self.charge_bytes(2 + $width)?;
                    encoder.$method(*value);
                    return Ok(());
                })*
            };
        }
        fixed! {
            bool => encode_bool, 1;
            char => encode_char, 4;
            i8 => encode_i8, 1;
            i16 => encode_i16, 2;
            i32 => encode_i32, 4;
            i64 => encode_i64, 8;
            i128 => encode_i128, 16;
            isize => encode_isize, 8;
            u8 => encode_u8, 1;
            u16 => encode_u16, 2;
            u32 => encode_u32, 4;
            u64 => encode_u64, 8;
            u128 => encode_u128, 16;
            usize => encode_usize, 8;
        }
        if let Some(value) = any.downcast_ref::<f32>() {
            if !value.is_finite() {
                return Err(CanonicalError::NonFiniteFloat {
                    type_name: leaf.type_name(),
                });
            }
            self.charge_bytes(2 + 4)?;
            encoder.encode_f32(if *value == 0.0 { 0.0 } else { *value });
        } else if let Some(value) = any.downcast_ref::<f64>() {
            if !value.is_finite() {
                return Err(CanonicalError::NonFiniteFloat {
                    type_name: leaf.type_name(),
                });
            }
            self.charge_bytes(2 + 8)?;
            encoder.encode_f64(if *value == 0.0 { 0.0 } else { *value });
        } else if let Some(value) = any.downcast_ref::<String>() {
            self.charge_bytes(2 + varint_len(value.len()))?;
            self.charge_bytes(value.len())?;
            encoder.encode_str(value);
        } else {
            return Err(SerializeError::UnsupportedLeaf {
                type_name: leaf.type_name(),
            }
            .into());
        }
        Ok(())
    }
}

// Exact width of the unsigned LEB128 emitted by BinaryEncoder.
fn varint_len(value: usize) -> usize {
    let mut value = value as u64;
    let mut len = 1;
    while value >= 0x80 {
        value >>= 7;
        len += 1;
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ser::binary::{from_binary, to_binary};
    use crate::ser::primitive::node_tag;
    use crate::{
        DynamicArray, DynamicList, DynamicMap, DynamicSet, DynamicVariant, FromReflect, List,
        ReflectMut, Typed,
    };
    use alloc::boxed::Box;
    use alloc::vec;
    use core::any::Any;
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

    fn canonical(value: &dyn Reflect) -> Vec<u8> {
        to_canonical_binary(value, CanonicalLimits::default()).unwrap()
    }

    #[derive(crate::Reflect, Debug, PartialEq)]
    struct Fields {
        zebra: u32,
        alpha: String,
    }

    #[derive(crate::Reflect, Debug, PartialEq)]
    struct Pair(i8, u128);

    #[derive(crate::Reflect, Debug, PartialEq)]
    enum Choice {
        Unit,
        Tuple(i8, u128),
        Struct { zebra: u32, alpha: String },
    }

    #[derive(crate::Reflect, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct RenamedFields {
        first: u32,
        second: u32,
    }

    #[derive(crate::Reflect, Debug, PartialEq)]
    enum MigratedChoice {
        Unit,
        Tuple(RenamedFields),
        Struct {
            right: RenamedFields,
            left: RenamedFields,
        },
    }

    #[derive(crate::Reflect, Debug, PartialEq)]
    struct MigratedRoot {
        records: Vec<RenamedFields>,
        lookup: BTreeMap<String, RenamedFields>,
        choice: MigratedChoice,
    }

    fn migrated_fields(first: u32, second: u32) -> DynamicStruct {
        let mut value = DynamicStruct::new();
        value.set_represented_type_name("old::Fields");
        value.insert("second", second);
        value.insert("old_first", first);
        let renamed = value.remove("old_first").unwrap();
        value.insert_boxed("first", renamed);
        value
    }

    fn assert_target<T>(value: &dyn Reflect, target: &T, registry: &TypeRegistry)
    where
        T: Reflect + Typed + FromReflect + PartialEq + fmt::Debug,
    {
        let info = <T as Typed>::type_info();
        let bytes =
            to_canonical_binary_with_schema(value, info, registry, CanonicalLimits::default())
                .unwrap();
        assert_eq!(bytes, canonical(target));
        let decoded = from_binary(&bytes, registry, info).unwrap();
        assert_eq!(T::from_reflect(&*decoded).unwrap(), *target);
    }

    #[test]
    fn explicit_schema_reorders_renamed_nested_structs_in_list_map_and_enum() {
        let mut registry = TypeRegistry::new();
        registry.register::<RenamedFields>();
        registry.register::<Vec<RenamedFields>>();
        registry.register::<BTreeMap<String, RenamedFields>>();
        registry.register::<MigratedChoice>();
        for variant in 0..3 {
            let mut records = DynamicList::new();
            records.push_value(migrated_fields(11, 22));
            let mut lookup = DynamicMap::new();
            lookup.insert_value(String::from("key"), migrated_fields(33, 44));
            let (payload, target_choice) = match variant {
                0 => (DynamicVariant::Unit, MigratedChoice::Unit),
                1 => (
                    DynamicVariant::Tuple(vec![Box::new(migrated_fields(55, 66))]),
                    MigratedChoice::Tuple(RenamedFields {
                        first: 55,
                        second: 66,
                    }),
                ),
                _ => (
                    DynamicVariant::Struct(vec![
                        ("left", Box::new(migrated_fields(77, 88))),
                        ("right", Box::new(migrated_fields(55, 66))),
                    ]),
                    MigratedChoice::Struct {
                        right: RenamedFields {
                            first: 55,
                            second: 66,
                        },
                        left: RenamedFields {
                            first: 77,
                            second: 88,
                        },
                    },
                ),
            };
            let name = ["Unit", "Tuple", "Struct"][variant];
            // Deliberately stale variant index and old type names at all levels.
            let mut choice = DynamicEnum::new(99, name, payload);
            choice.set_represented_type_name("old::Choice");
            let mut root = DynamicStruct::new();
            root.set_represented_type_name("old::Root");
            root.insert("choice", choice);
            root.insert("lookup", lookup);
            root.insert("records", records);
            let target = MigratedRoot {
                records: vec![RenamedFields {
                    first: 11,
                    second: 22,
                }],
                lookup: BTreeMap::from([(
                    String::from("key"),
                    RenamedFields {
                        first: 33,
                        second: 44,
                    },
                )]),
                choice: target_choice,
            };
            // SchemaView also ensures the writer never clones the source tree
            // or consults its own (intentionally unrelated) static schema.
            let view = SchemaView {
                storage: Box::new(root),
                schema: <Fields as Typed>::type_info(),
            };
            assert_target(&view, &target, &registry);
        }
    }

    #[test]
    fn explicit_schema_propagates_to_array_tuple_map_keys_and_set_elements() {
        let mut registry = TypeRegistry::new();
        registry.register::<RenamedFields>();
        let mut array = DynamicArray::new();
        array.push_value(migrated_fields(1, 2));
        assert_target(
            &array,
            &[RenamedFields {
                first: 1,
                second: 2,
            }],
            &registry,
        );

        #[derive(crate::Reflect, Debug, PartialEq)]
        struct Wrapped(RenamedFields, u8);
        let mut tuple = DynamicTupleStruct::new();
        tuple.insert(migrated_fields(1, 2));
        tuple.insert(9_u8);
        assert_target(
            &tuple,
            &Wrapped(
                RenamedFields {
                    first: 1,
                    second: 2,
                },
                9,
            ),
            &registry,
        );

        let mut map = DynamicMap::new();
        map.insert_value(migrated_fields(1, 2), 7_u8);
        map.insert_value(migrated_fields(256, 3), 8_u8);
        assert_target(
            &map,
            &BTreeMap::from([
                (
                    RenamedFields {
                        first: 1,
                        second: 2,
                    },
                    7_u8,
                ),
                (
                    RenamedFields {
                        first: 256,
                        second: 3,
                    },
                    8_u8,
                ),
            ]),
            &registry,
        );
        let mut set = DynamicSet::new();
        set.push_value(migrated_fields(1, 2));
        set.push_value(migrated_fields(256, 3));
        assert_target(
            &set,
            &BTreeSet::from([
                RenamedFields {
                    first: 1,
                    second: 2,
                },
                RenamedFields {
                    first: 256,
                    second: 3,
                },
            ]),
            &registry,
        );
    }

    #[test]
    fn explicit_schema_rejects_missing_extra_replaced_and_wrong_kind_fields() {
        let registry = TypeRegistry::new();
        let info = <RenamedFields as Typed>::type_info();
        let mut missing = DynamicStruct::new();
        missing.insert("first", 1_u32);
        let mut extra = migrated_fields(1, 2);
        extra.insert("extra", 3_u32);
        let mut replaced = DynamicStruct::new();
        replaced.insert("first", 1_u32);
        replaced.insert("old_second", 2_u32);
        for value in [
            &missing as &dyn Reflect,
            &extra,
            &replaced,
            &DynamicList::new(),
            &0_u32,
        ] {
            assert!(matches!(
                to_canonical_binary_with_schema(value, info, &registry, CanonicalLimits::default()),
                Err(CanonicalError::InvalidShape { .. })
            ));
        }
        let mut wrong_leaf = migrated_fields(1, 2);
        wrong_leaf.insert("second", 2_i32);
        assert!(matches!(
            to_canonical_binary_with_schema(
                &wrong_leaf,
                info,
                &registry,
                CanonicalLimits::default()
            ),
            Err(CanonicalError::PrimitiveMismatch {
                expected: "u32",
                actual: "i32"
            })
        ));
        let mut wrong_kind = migrated_fields(1, 2);
        wrong_kind.insert("second", DynamicStruct::new());
        assert!(matches!(
            to_canonical_binary_with_schema(
                &wrong_kind,
                info,
                &registry,
                CanonicalLimits::default()
            ),
            Err(CanonicalError::InvalidShape { .. })
        ));
        assert!(matches!(
            to_canonical_binary_with_schema(
                &vec![1_i32],
                <Vec<u32> as Typed>::type_info(),
                &registry,
                CanonicalLimits::default(),
            ),
            Err(CanonicalError::PrimitiveMismatch { .. })
        ));
        let mut wrong_map = DynamicMap::new();
        wrong_map.insert_value(1_u8, 2_u8);
        assert!(matches!(
            to_canonical_binary_with_schema(
                &wrong_map,
                <BTreeMap<u16, u8> as Typed>::type_info(),
                &registry,
                CanonicalLimits::default(),
            ),
            Err(CanonicalError::PrimitiveMismatch { .. })
        ));
    }

    #[test]
    fn explicit_schema_checks_tuple_array_and_enum_shapes() {
        let registry = TypeRegistry::new();
        let limits = CanonicalLimits::default();
        let mut tuple = DynamicTupleStruct::new();
        tuple.insert(-3_i8);
        assert!(matches!(
            to_canonical_binary_with_schema(
                &tuple,
                <Pair as Typed>::type_info(),
                &registry,
                limits
            ),
            Err(CanonicalError::InvalidShape { .. })
        ));
        tuple.insert(1_u128);
        assert_target(&tuple, &Pair(-3, 1), &registry);
        tuple.insert(0_u8);
        assert!(matches!(
            to_canonical_binary_with_schema(
                &tuple,
                <Pair as Typed>::type_info(),
                &registry,
                limits
            ),
            Err(CanonicalError::InvalidShape { .. })
        ));
        for len in [0, 2] {
            let mut array = DynamicArray::new();
            for _ in 0..len {
                array.push_value(1_u8);
            }
            assert!(matches!(
                to_canonical_binary_with_schema(
                    &array,
                    <[u8; 1] as Typed>::type_info(),
                    &registry,
                    limits
                ),
                Err(CanonicalError::InvalidShape { .. })
            ));
        }
        let invalid_enums = [
            DynamicEnum::new(0, "Unknown", DynamicVariant::Unit),
            DynamicEnum::new(0, "Tuple", DynamicVariant::Unit),
            DynamicEnum::new(1, "Tuple", DynamicVariant::Tuple(vec![Box::new(-3_i8)])),
            DynamicEnum::new(
                2,
                "Struct",
                DynamicVariant::Struct(vec![
                    ("zebra", Box::new(1_u32)),
                    ("old_alpha", Box::new(String::new())),
                ]),
            ),
        ];
        for value in invalid_enums {
            assert!(matches!(
                to_canonical_binary_with_schema(
                    &value,
                    <Choice as Typed>::type_info(),
                    &registry,
                    limits
                ),
                Err(CanonicalError::InvalidShape { .. })
            ));
        }
    }

    #[test]
    fn explicit_schema_reports_unregistered_declared_children_even_in_empty_collections() {
        let registry = TypeRegistry::new();
        let limits = CanonicalLimits::default();
        let target_name = <RenamedFields as Typed>::type_info().type_name();
        let mut list = DynamicList::new();
        for populated in [false, true] {
            if populated {
                list.push_value(migrated_fields(1, 2));
            }
            assert_eq!(
                to_canonical_binary_with_schema(
                    &list,
                    <Vec<RenamedFields> as Typed>::type_info(),
                    &registry,
                    limits
                ),
                Err(CanonicalError::UnregisteredType {
                    type_name: target_name
                })
            );
        }
        let mut root = DynamicStruct::new();
        root.insert("records", DynamicList::new());
        root.insert("lookup", DynamicMap::new());
        root.insert("choice", DynamicEnum::new(0, "Unit", DynamicVariant::Unit));
        assert_eq!(
            to_canonical_binary_with_schema(
                &root,
                <MigratedRoot as Typed>::type_info(),
                &registry,
                limits
            ),
            Err(CanonicalError::UnregisteredType {
                type_name: <Vec<RenamedFields> as Typed>::type_info().type_name()
            })
        );
        assert_target(&vec![1_u32, 2], &vec![1_u32, 2], &registry);
    }

    #[test]
    fn explicit_schema_rejects_collisions_after_schema_ordering_before_map_values() {
        let mut registry = TypeRegistry::new();
        registry.register::<RenamedFields>();
        let mut direct = DynamicStruct::new();
        direct.insert("first", 1_u32);
        direct.insert("second", 2_u32);
        let mut map = DynamicMap::new();
        map.insert_value(migrated_fields(1, 2), f64::NAN);
        map.insert_value(direct, f64::INFINITY);
        assert!(matches!(
            to_canonical_binary_with_schema(
                &map,
                <BTreeMap<RenamedFields, f64> as Typed>::type_info(),
                &registry,
                CanonicalLimits::default()
            ),
            Err(CanonicalError::DuplicateMapKey)
        ));
        let mut set = DynamicSet::new();
        set.push_value(migrated_fields(1, 2));
        set.push_value(migrated_fields(1, 2));
        assert!(matches!(
            to_canonical_binary_with_schema(
                &set,
                <BTreeSet<RenamedFields> as Typed>::type_info(),
                &registry,
                CanonicalLimits::default()
            ),
            Err(CanonicalError::DuplicateSetElement)
        ));
    }

    #[test]
    fn explicit_schema_preserves_global_sorting_budgets_and_early_rejection() {
        let mut registry = TypeRegistry::new();
        registry.register::<RenamedFields>();
        let mut map = DynamicMap::new();
        map.insert_value(migrated_fields(1, 2), 7_u8);
        let info = <BTreeMap<RenamedFields, u8> as Typed>::type_info();
        let bytes =
            to_canonical_binary_with_schema(&map, info, &registry, CanonicalLimits::default())
                .unwrap();
        let exact = CanonicalLimits {
            max_bytes: bytes.len(),
            max_nodes: 5,
            max_depth: 3,
            max_collection_len: 2,
        };
        assert_eq!(
            to_canonical_binary_with_schema(&map, info, &registry, exact).unwrap(),
            bytes
        );
        for limits in [
            CanonicalLimits {
                max_bytes: exact.max_bytes - 1,
                ..exact
            },
            CanonicalLimits {
                max_nodes: 4,
                ..exact
            },
            CanonicalLimits {
                max_depth: 2,
                ..exact
            },
            CanonicalLimits {
                max_collection_len: 1,
                ..exact
            },
        ] {
            assert!(to_canonical_binary_with_schema(&map, info, &registry, limits).is_err());
        }
        assert!(matches!(
            to_canonical_binary_with_schema(
                &UntouchedList(2),
                <Vec<u8> as Typed>::type_info(),
                &registry,
                CanonicalLimits {
                    max_nodes: 2,
                    ..CanonicalLimits::default()
                }
            ),
            Err(CanonicalError::NodeLimitExceeded { .. })
        ));
    }

    // Exposes deliberately reordered dynamic storage through a typed schema.
    // A writer must consult that schema rather than field_at ordering.
    struct SchemaView {
        storage: Box<dyn Reflect>,
        schema: &'static TypeInfo,
    }

    impl Reflect for SchemaView {
        fn type_name(&self) -> &'static str {
            self.schema.type_name()
        }
        fn type_info(&self) -> &'static TypeInfo {
            self.schema
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn into_any(self: Box<Self>) -> Box<dyn Any> {
            self
        }
        fn as_reflect(&self) -> &dyn Reflect {
            self
        }
        fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
            self
        }
        fn reflect_ref(&self) -> ReflectRef<'_> {
            self.storage.reflect_ref()
        }
        fn reflect_mut(&mut self) -> ReflectMut<'_> {
            self.storage.reflect_mut()
        }
        fn reflect_clone(&self) -> Box<dyn Reflect> {
            panic!("canonical serialization must not clone reflected values");
        }
    }

    #[test]
    fn default_limits() {
        assert_eq!(
            CanonicalLimits::default(),
            CanonicalLimits {
                max_bytes: 1_048_576,
                max_depth: 64,
                max_collection_len: 65_536,
                max_nodes: 65_536,
            }
        );
    }

    #[test]
    fn every_primitive_matches_prb1_and_checks_exact_byte_budget() {
        let values: Vec<Box<dyn Reflect>> = vec![
            Box::new(true),
            Box::new('界'),
            Box::new(i8::MIN),
            Box::new(i16::MIN),
            Box::new(i32::MIN),
            Box::new(i64::MIN),
            Box::new(i128::MIN),
            Box::new(isize::MIN),
            Box::new(u8::MAX),
            Box::new(u16::MAX),
            Box::new(u32::MAX),
            Box::new(u64::MAX),
            Box::new(u128::MAX),
            Box::new(usize::MAX),
            Box::new(1.25_f32),
            Box::new(-2.5_f64),
            Box::new(String::new()),
            Box::new(String::from("界").repeat(128)),
        ];
        for value in values {
            let expected = to_binary(&*value).unwrap();
            let mut limits = CanonicalLimits {
                max_bytes: expected.len(),
                max_depth: 1,
                max_collection_len: 0,
                max_nodes: 1,
            };
            assert_eq!(to_canonical_binary(&*value, limits).unwrap(), expected);
            assert_eq!(
                to_canonical_binary_with_schema(
                    &*value,
                    value.type_info(),
                    &TypeRegistry::new(),
                    limits,
                )
                .unwrap(),
                expected
            );
            let wrong_schema = if value.as_any().is::<u8>() {
                <u16 as Typed>::type_info()
            } else {
                <u8 as Typed>::type_info()
            };
            assert!(matches!(
                to_canonical_binary_with_schema(
                    &*value,
                    wrong_schema,
                    &TypeRegistry::new(),
                    limits,
                ),
                Err(CanonicalError::PrimitiveMismatch { .. })
            ));
            limits.max_bytes -= 1;
            assert!(matches!(
                to_canonical_binary(&*value, limits),
                Err(CanonicalError::BytesLimitExceeded { .. })
            ));
        }
    }

    #[test]
    fn typed_fields_follow_schema_and_round_trip_without_cloning() {
        let expected = Fields {
            zebra: 7,
            alpha: String::from("a"),
        };
        let mut storage = DynamicStruct::new();
        storage.insert("alpha", String::from("a"));
        storage.insert("zebra", 7_u32);
        let view = SchemaView {
            storage: Box::new(storage),
            schema: <Fields as Typed>::type_info(),
        };
        let bytes = canonical(&view);
        assert_eq!(bytes, to_binary(&expected).unwrap());
        let decoded = from_binary(&bytes, &TypeRegistry::new(), view.schema).unwrap();
        assert_eq!(Fields::from_reflect(&*decoded).unwrap(), expected);
        let pair = Pair(-3, u128::MAX);
        assert_eq!(canonical(&pair), to_binary(&pair).unwrap());
        let decoded = from_binary(
            &canonical(&pair),
            &TypeRegistry::new(),
            <Pair as Typed>::type_info(),
        )
        .unwrap();
        assert_eq!(Pair::from_reflect(&*decoded).unwrap(), pair);
    }

    #[test]
    fn every_enum_shape_matches_prb1_including_schema_field_order() {
        let choices = [
            Choice::Unit,
            Choice::Tuple(-3, u128::MAX),
            Choice::Struct {
                zebra: 7,
                alpha: String::from("a"),
            },
        ];
        for choice in choices {
            let bytes = canonical(&choice);
            assert_eq!(bytes, to_binary(&choice).unwrap());
            let decoded =
                from_binary(&bytes, &TypeRegistry::new(), <Choice as Typed>::type_info()).unwrap();
            assert_eq!(Choice::from_reflect(&*decoded).unwrap(), choice);
        }
        let storage = DynamicEnum::new(
            2,
            "Struct",
            DynamicVariant::Struct(vec![
                ("alpha", Box::new(String::from("a"))),
                ("zebra", Box::new(7_u32)),
            ]),
        );
        let view = SchemaView {
            storage: Box::new(storage),
            schema: <Choice as Typed>::type_info(),
        };
        assert_eq!(
            canonical(&view),
            canonical(&Choice::Struct {
                zebra: 7,
                alpha: String::from("a"),
            })
        );
        for value in [None, Some(4_u8)] {
            assert_eq!(canonical(&value), to_binary(&value).unwrap());
        }
    }

    #[test]
    fn numeric_lists_and_arrays_decode_as_ordinary_nodes() {
        let list = vec![1_i32, -2, 3];
        let bytes = canonical(&list);
        assert_eq!(bytes[13], node_tag::LIST);
        let decoded = from_binary(
            &bytes,
            &TypeRegistry::new(),
            <Vec<i32> as Typed>::type_info(),
        )
        .unwrap();
        assert_eq!(Vec::<i32>::from_reflect(&*decoded).unwrap(), list);
        let array = [0.0_f32, -0.0, 3.5];
        let bytes = canonical(&array);
        assert_eq!(bytes[13], node_tag::ARRAY);
        let decoded = from_binary(
            &bytes,
            &TypeRegistry::new(),
            <[f32; 3] as Typed>::type_info(),
        )
        .unwrap();
        let result = <[f32; 3]>::from_reflect(&*decoded).unwrap();
        assert_eq!(result, array);
        assert_eq!(result[1].to_bits(), 0);
    }

    #[test]
    fn map_and_set_sort_by_node_bytes_instead_of_natural_order() {
        // u16 LE node bytes sort 256 before 1.
        let map = BTreeMap::from([(1_u16, 9_u8), (256, 8)]);
        let mut ordered = DynamicMap::new();
        ordered.set_represented_type_name(map.type_name());
        ordered.insert_value(256_u16, 8_u8);
        ordered.insert_value(1_u16, 9_u8);
        let bytes = canonical(&map);
        assert_eq!(bytes, to_binary(&ordered).unwrap());
        let decoded = from_binary(
            &bytes,
            &TypeRegistry::new(),
            <BTreeMap<u16, u8> as Typed>::type_info(),
        )
        .unwrap();
        assert_eq!(BTreeMap::<u16, u8>::from_reflect(&*decoded).unwrap(), map);
        let set = BTreeSet::from([1_u16, 256]);
        let mut ordered = DynamicSet::new();
        ordered.set_represented_type_name(set.type_name());
        ordered.push_value(256_u16);
        ordered.push_value(1_u16);
        let bytes = canonical(&set);
        assert_eq!(bytes, to_binary(&ordered).unwrap());
        let decoded = from_binary(
            &bytes,
            &TypeRegistry::new(),
            <BTreeSet<u16> as Typed>::type_info(),
        )
        .unwrap();
        assert_eq!(BTreeSet::<u16>::from_reflect(&*decoded).unwrap(), set);
    }

    #[test]
    fn nested_unordered_collections_ignore_insertion_order() {
        fn value(reverse: bool) -> HashMap<String, HashSet<u16>> {
            let mut entries = vec![
                (String::from("z"), HashSet::from([1, 256])),
                (String::from("aa"), HashSet::from([7, 5])),
            ];
            if reverse {
                entries.reverse();
            }
            entries.into_iter().collect()
        }
        let expected = value(false);
        let bytes = canonical(&expected);
        assert_eq!(bytes, canonical(&value(true)));
        let mut registry = TypeRegistry::new();
        registry.register::<HashSet<u16>>();
        let decoded = from_binary(
            &bytes,
            &registry,
            <HashMap<String, HashSet<u16>> as Typed>::type_info(),
        )
        .unwrap();
        assert_eq!(
            HashMap::<String, HashSet<u16>>::from_reflect(&*decoded).unwrap(),
            expected
        );
    }

    #[test]
    fn floats_normalize_zero_and_reject_non_finite_values_at_any_depth() {
        assert_eq!(canonical(&-0.0_f32), canonical(&0.0_f32));
        assert_eq!(canonical(&-0.0_f64), canonical(&0.0_f64));
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                to_canonical_binary(&value, CanonicalLimits::default()),
                Err(CanonicalError::NonFiniteFloat { .. })
            ));
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                to_canonical_binary(&[value], CanonicalLimits::default()),
                Err(CanonicalError::NonFiniteFloat { .. })
            ));
        }
        let mut map = DynamicMap::new();
        map.insert_value(0_u8, f64::NAN);
        assert!(matches!(
            to_canonical_binary(&map, CanonicalLimits::default()),
            Err(CanonicalError::NonFiniteFloat { .. })
        ));
        let mut set = DynamicSet::new();
        set.push_value(f32::INFINITY);
        assert!(matches!(
            to_canonical_binary(&set, CanonicalLimits::default()),
            Err(CanonicalError::NonFiniteFloat { .. })
        ));
    }

    #[test]
    fn canonical_collisions_are_rejected_before_map_values_are_encoded() {
        // DynamicMap/Set do not deduplicate composite values. These keys and
        // elements differ only in zero's sign, which canonicalization removes.
        let mut map = DynamicMap::new();
        map.insert_value(vec![-0.0_f32], f64::NAN);
        map.insert_value(vec![0.0_f32], f64::INFINITY);
        assert!(matches!(
            to_canonical_binary(&map, CanonicalLimits::default()),
            Err(CanonicalError::DuplicateMapKey)
        ));
        let mut set = DynamicSet::new();
        assert!(set.push_value(vec![-0.0_f64]));
        assert!(set.push_value(vec![0.0_f64]));
        assert!(matches!(
            to_canonical_binary(&set, CanonicalLimits::default()),
            Err(CanonicalError::DuplicateSetElement)
        ));
    }

    #[test]
    fn sorting_buffers_share_exact_global_byte_and_node_limits() {
        let mut inner = DynamicMap::new();
        inner.insert_value(2_u8, 8_u8);
        inner.insert_value(1_u8, 9_u8);
        let mut key = DynamicList::new();
        key.push_value(inner);
        let mut set = DynamicSet::new();
        set.push_value(3_u8);
        set.push_value(4_u8);
        let mut outer = DynamicMap::new();
        outer.insert_value(key, set);
        // outer + key list + inner map + 4 leaves + set + 2 leaves = 10 nodes.
        let expected = canonical(&outer);
        let mut limits = CanonicalLimits {
            max_bytes: expected.len(),
            max_nodes: 10,
            max_depth: 4,
            max_collection_len: 2,
        };
        assert_eq!(to_canonical_binary(&outer, limits).unwrap(), expected);
        limits.max_bytes -= 1;
        assert!(matches!(
            to_canonical_binary(&outer, limits),
            Err(CanonicalError::BytesLimitExceeded { .. })
        ));
        limits.max_bytes += 1;
        limits.max_nodes -= 1;
        assert!(matches!(
            to_canonical_binary(&outer, limits),
            Err(CanonicalError::NodeLimitExceeded { .. })
        ));
    }

    // Any attempt to inspect a child panics. Length/depth/minimum-node checks
    // must reject this collection before calling get or allocating by len.
    struct UntouchedList(usize);

    impl Reflect for UntouchedList {
        fn type_name(&self) -> &'static str {
            core::any::type_name::<Self>()
        }
        fn type_info(&self) -> &'static TypeInfo {
            <Vec<u8> as Typed>::type_info()
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn into_any(self: Box<Self>) -> Box<dyn Any> {
            self
        }
        fn as_reflect(&self) -> &dyn Reflect {
            self
        }
        fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
            self
        }
        fn reflect_ref(&self) -> ReflectRef<'_> {
            ReflectRef::List(self)
        }
        fn reflect_mut(&mut self) -> ReflectMut<'_> {
            ReflectMut::List(self)
        }
        fn reflect_clone(&self) -> Box<dyn Reflect> {
            panic!("must not clone")
        }
    }

    impl List for UntouchedList {
        fn len(&self) -> usize {
            self.0
        }
        fn get(&self, _: usize) -> Option<&dyn Reflect> {
            panic!("must reject before get")
        }
        fn get_mut(&mut self, _: usize) -> Option<&mut dyn Reflect> {
            panic!("must not mutate")
        }
        fn push(&mut self, value: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
            Err(value)
        }
        fn as_list(&self) -> &dyn List {
            self
        }
    }

    #[test]
    fn impossible_collections_are_rejected_before_child_access() {
        let limits = CanonicalLimits::default();
        assert!(matches!(
            to_canonical_binary(&UntouchedList(usize::MAX), limits),
            Err(CanonicalError::CollectionTooLarge { .. })
        ));
        assert!(matches!(
            to_canonical_binary(
                &UntouchedList(2),
                CanonicalLimits {
                    max_depth: 1,
                    ..limits
                }
            ),
            Err(CanonicalError::DepthLimitExceeded { .. })
        ));
        assert!(matches!(
            to_canonical_binary(
                &UntouchedList(2),
                CanonicalLimits {
                    max_nodes: 2,
                    ..limits
                }
            ),
            Err(CanonicalError::NodeLimitExceeded { .. })
        ));
        assert!(matches!(
            to_canonical_binary(
                &UntouchedList(usize::MAX),
                CanonicalLimits {
                    max_nodes: usize::MAX,
                    max_collection_len: usize::MAX,
                    ..limits
                }
            ),
            Err(CanonicalError::NodeLimitExceeded { .. })
        ));
        assert!(matches!(
            to_canonical_binary(
                &UntouchedList(1),
                CanonicalLimits {
                    max_bytes: 14,
                    ..limits
                }
            ),
            Err(CanonicalError::BytesLimitExceeded { .. })
        ));
    }

    #[test]
    fn zero_limits_and_exact_depth_node_collection_boundaries() {
        let limits = CanonicalLimits::default();
        assert!(matches!(
            to_canonical_binary(
                &true,
                CanonicalLimits {
                    max_bytes: 0,
                    ..limits
                }
            ),
            Err(CanonicalError::BytesLimitExceeded { .. })
        ));
        assert!(matches!(
            to_canonical_binary(
                &true,
                CanonicalLimits {
                    max_depth: 0,
                    ..limits
                }
            ),
            Err(CanonicalError::DepthLimitExceeded { .. })
        ));
        assert!(matches!(
            to_canonical_binary(
                &true,
                CanonicalLimits {
                    max_nodes: 0,
                    ..limits
                }
            ),
            Err(CanonicalError::NodeLimitExceeded { .. })
        ));
        assert!(to_canonical_binary(
            &Vec::<u8>::new(),
            CanonicalLimits {
                max_depth: 1,
                max_nodes: 1,
                max_collection_len: 0,
                ..limits
            }
        )
        .is_ok());
        let value = vec![vec![1_u8]];
        let exact = CanonicalLimits {
            max_depth: 3,
            max_nodes: 3,
            max_collection_len: 1,
            ..limits
        };
        assert!(to_canonical_binary(&value, exact).is_ok());
        assert!(matches!(
            to_canonical_binary(
                &value,
                CanonicalLimits {
                    max_depth: 2,
                    ..exact
                }
            ),
            Err(CanonicalError::DepthLimitExceeded { .. })
        ));
        assert!(matches!(
            to_canonical_binary(
                &value,
                CanonicalLimits {
                    max_nodes: 2,
                    ..exact
                }
            ),
            Err(CanonicalError::NodeLimitExceeded { .. })
        ));
        assert!(matches!(
            to_canonical_binary(
                &value,
                CanonicalLimits {
                    max_collection_len: 0,
                    ..exact
                }
            ),
            Err(CanonicalError::CollectionTooLarge { .. })
        ));
    }

    #[test]
    fn unsupported_leaves_and_inconsistent_schemas_return_errors() {
        let error = to_canonical_binary(&(), CanonicalLimits::default()).unwrap_err();
        assert!(matches!(
            error,
            CanonicalError::Serialize(SerializeError::UnsupportedLeaf { .. })
        ));
        assert!(core::error::Error::source(&error).is_some());
        assert!(!error.to_string().is_empty());
        let bad = SchemaView {
            storage: Box::new(DynamicStruct::new()),
            schema: <Fields as Typed>::type_info(),
        };
        assert!(matches!(
            to_canonical_binary(&bad, CanonicalLimits::default()),
            Err(CanonicalError::InvalidShape { .. })
        ));
    }
}
