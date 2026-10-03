//! The compact, self-describing binary serialization format.
//!
//! A stream is a fixed header followed by one recursively-encoded root node:
//!
//! ```text
//! header := MAGIC(4) VERSION(1) ROOT_STABLE_ID(u64 LE)
//! node   := NODE_TAG(1) body
//! ```
//!
//! Each composite node writes its element count as an unsigned LEB128 varint,
//! then its children; a leaf (`Value`) node writes a primitive tag byte and the
//! value's little-endian bytes. Field names, variant names, and child element
//! types are **not** stored — reconstruction is guided entirely by the target
//! [`TypeInfo`] resolved through the [`TypeRegistry`], while the stored node and
//! primitive tags let the reader detect corruption or a schema mismatch. The
//! header's [`StableTypeId`](crate::StableTypeId) pins the root type across
//! builds (design §22).

use crate::kinds::VariantType;
use crate::reflect::Reflect;
use crate::ser::de::{Schema, resolve, root_schema};
use crate::ser::encode::{Encoder, serialize_value};
use crate::ser::error::{DeserializeError, SerializeError};
use crate::ser::primitive::{ByteReader, Primitive, leaf_primitive, node_tag, write_varint};
use crate::ser::stable_id::StableTypeId;
use crate::type_info::TypeInfo;
use crate::{
    DynamicArray, DynamicEnum, DynamicList, DynamicMap, DynamicSet, DynamicStruct,
    DynamicTupleStruct, DynamicVariant, TypeRegistry,
};
use std::boxed::Box;
use std::string::String;
use std::vec::Vec;

/// The 4-byte magic that opens every binary stream (`PRB1` = Prism Reflect
/// Binary v1).
pub const MAGIC: [u8; 4] = *b"PRB1";

/// The binary format version written into (and required by) the header.
pub const VERSION: u8 = 1;

/// An [`Encoder`] that appends the binary wire format to an owned buffer.
struct BinaryEncoder {
    out: Vec<u8>,
}

impl BinaryEncoder {
    fn new() -> Self {
        Self { out: Vec::new() }
    }

    /// Write a leaf node: the `Value` node tag, the primitive tag, then bytes.
    fn leaf(&mut self, primitive: Primitive, bytes: &[u8]) {
        self.out.push(node_tag::VALUE);
        self.out.push(primitive.tag());
        self.out.extend_from_slice(bytes);
    }

    /// Write a composite node tag followed by its varint element count.
    fn open(&mut self, tag: u8, count: u64) {
        self.out.push(tag);
        write_varint(&mut self.out, count);
    }
}

impl Encoder for BinaryEncoder {
    fn encode_bool(&mut self, value: bool) {
        self.leaf(Primitive::Bool, &[u8::from(value)]);
    }
    fn encode_char(&mut self, value: char) {
        self.leaf(Primitive::Char, &u32::from(value).to_le_bytes());
    }
    fn encode_i8(&mut self, value: i8) {
        self.leaf(Primitive::I8, &value.to_le_bytes());
    }
    fn encode_i16(&mut self, value: i16) {
        self.leaf(Primitive::I16, &value.to_le_bytes());
    }
    fn encode_i32(&mut self, value: i32) {
        self.leaf(Primitive::I32, &value.to_le_bytes());
    }
    fn encode_i64(&mut self, value: i64) {
        self.leaf(Primitive::I64, &value.to_le_bytes());
    }
    fn encode_i128(&mut self, value: i128) {
        self.leaf(Primitive::I128, &value.to_le_bytes());
    }
    fn encode_isize(&mut self, value: isize) {
        self.leaf(Primitive::Isize, &(value as i64).to_le_bytes());
    }
    fn encode_u8(&mut self, value: u8) {
        self.leaf(Primitive::U8, &value.to_le_bytes());
    }
    fn encode_u16(&mut self, value: u16) {
        self.leaf(Primitive::U16, &value.to_le_bytes());
    }
    fn encode_u32(&mut self, value: u32) {
        self.leaf(Primitive::U32, &value.to_le_bytes());
    }
    fn encode_u64(&mut self, value: u64) {
        self.leaf(Primitive::U64, &value.to_le_bytes());
    }
    fn encode_u128(&mut self, value: u128) {
        self.leaf(Primitive::U128, &value.to_le_bytes());
    }
    fn encode_usize(&mut self, value: usize) {
        self.leaf(Primitive::Usize, &(value as u64).to_le_bytes());
    }
    fn encode_f32(&mut self, value: f32) {
        self.leaf(Primitive::F32, &value.to_bits().to_le_bytes());
    }
    fn encode_f64(&mut self, value: f64) {
        self.leaf(Primitive::F64, &value.to_bits().to_le_bytes());
    }
    fn encode_str(&mut self, value: &str) {
        self.out.push(node_tag::VALUE);
        self.out.push(Primitive::String.tag());
        write_varint(&mut self.out, value.len() as u64);
        self.out.extend_from_slice(value.as_bytes());
    }

    fn begin_struct(&mut self, count: usize) {
        self.open(node_tag::STRUCT, count as u64);
    }
    fn before_struct_field(&mut self, _name: &str, _index: usize) {}
    fn end_struct(&mut self) {}

    fn begin_tuple_struct(&mut self, count: usize) {
        self.open(node_tag::TUPLE_STRUCT, count as u64);
    }
    fn before_tuple_struct_field(&mut self, _index: usize) {}
    fn end_tuple_struct(&mut self) {}

    fn begin_enum(
        &mut self,
        variant_index: usize,
        _variant_name: &str,
        _variant_type: VariantType,
        count: usize,
    ) {
        self.out.push(node_tag::ENUM);
        write_varint(&mut self.out, variant_index as u64);
        write_varint(&mut self.out, count as u64);
    }
    fn before_enum_tuple_field(&mut self, _index: usize) {}
    fn before_enum_struct_field(&mut self, _name: &str, _index: usize) {}
    fn end_enum(&mut self) {}

    fn begin_list(&mut self, len: usize) {
        self.open(node_tag::LIST, len as u64);
    }
    fn before_list_element(&mut self, _index: usize) {}
    fn end_list(&mut self) {}

    fn begin_array(&mut self, len: usize) {
        self.open(node_tag::ARRAY, len as u64);
    }
    fn before_array_element(&mut self, _index: usize) {}
    fn end_array(&mut self) {}

    fn begin_set(&mut self, len: usize) {
        self.open(node_tag::SET, len as u64);
    }
    fn before_set_element(&mut self, _index: usize) {}
    fn end_set(&mut self) {}

    fn begin_map(&mut self, len: usize) {
        self.open(node_tag::MAP, len as u64);
    }
    fn before_map_key(&mut self, _index: usize) {}
    fn before_map_value(&mut self, _index: usize) {}
    fn end_map(&mut self) {}
}

/// Serialize any reflected value to the compact binary format.
///
/// The output is `MAGIC + VERSION + root StableTypeId + body`, where the body
/// is the recursively-encoded value.
///
/// # Errors
/// Returns [`SerializeError::UnsupportedLeaf`] when the value (or a nested
/// element) is a `Value`-kind type outside the built-in leaf set.
pub fn to_binary(value: &dyn Reflect) -> Result<Vec<u8>, SerializeError> {
    let mut encoder = BinaryEncoder::new();
    encoder.out.extend_from_slice(&MAGIC);
    encoder.out.push(VERSION);
    let root_id = StableTypeId::of_path(value.type_name()).value();
    encoder.out.extend_from_slice(&root_id.to_le_bytes());
    serialize_value(value, &mut encoder)?;
    Ok(encoder.out)
}

/// Deserialize a reflected value from the binary format against a target type.
///
/// The target [`TypeInfo`] (plus the `registry` for nested types) guides
/// reconstruction into a [`Dynamic*`](crate::dynamic) tree whose represented
/// type names are stamped from the schema, so the result round-trips back to
/// the original concrete value via [`FromReflect`](crate::FromReflect).
///
/// # Errors
/// Returns a [`DeserializeError`] for a bad header, a stable-id mismatch with
/// `target`, a corrupt tag, an unregistered nested type, a leaf-type mismatch,
/// or trailing bytes.
pub fn from_binary(
    bytes: &[u8],
    registry: &TypeRegistry,
    target: &TypeInfo,
) -> Result<Box<dyn Reflect>, DeserializeError> {
    let mut reader = ByteReader::new(bytes);
    let magic = reader.read_array::<4>()?;
    if magic != MAGIC {
        return Err(DeserializeError::BadMagic);
    }
    let version = reader.read_u8()?;
    if version != VERSION {
        return Err(DeserializeError::UnsupportedVersion(version));
    }
    let found_id = u64::from_le_bytes(reader.read_array::<8>()?);
    let expected_id = StableTypeId::of_path(target.type_name()).value();
    if found_id != expected_id {
        return Err(DeserializeError::StableIdMismatch {
            expected: expected_id,
            found: found_id,
        });
    }

    let schema = root_schema(target);
    let value = read_node(&mut reader, registry, &schema)?;

    if reader.is_empty() {
        Ok(value)
    } else {
        Err(DeserializeError::TrailingData)
    }
}

/// Read one node from `reader` under the guidance of `schema`.
fn read_node(
    reader: &mut ByteReader<'_>,
    registry: &TypeRegistry,
    schema: &Schema<'_>,
) -> Result<Box<dyn Reflect>, DeserializeError> {
    let tag = reader.read_u8()?;
    match schema {
        Schema::Primitive(primitive) => {
            expect_tag(tag, node_tag::VALUE, "Value")?;
            read_primitive(reader, *primitive)
        }
        Schema::Info(info) => read_info_node(reader, registry, tag, info),
    }
}

/// Read a composite (or registered-leaf) node whose shape is `info`.
fn read_info_node(
    reader: &mut ByteReader<'_>,
    registry: &TypeRegistry,
    tag: u8,
    info: &TypeInfo,
) -> Result<Box<dyn Reflect>, DeserializeError> {
    match info {
        TypeInfo::Struct(struct_info) => {
            expect_tag(tag, node_tag::STRUCT, "Struct")?;
            let count = reader.read_len()?;
            if count != struct_info.field_count() {
                return Err(DeserializeError::FieldCountMismatch);
            }
            let mut dynamic = DynamicStruct::new();
            dynamic.set_represented_type_name(struct_info.type_name());
            for field in struct_info.fields() {
                let child_schema = resolve(registry, field.type_name())?;
                let child = read_node(reader, registry, &child_schema)?;
                dynamic.insert_boxed(field.name(), child);
            }
            Ok(Box::new(dynamic))
        }
        TypeInfo::TupleStruct(tuple_info) => {
            expect_tag(tag, node_tag::TUPLE_STRUCT, "TupleStruct")?;
            let count = reader.read_len()?;
            if count != tuple_info.field_count() {
                return Err(DeserializeError::FieldCountMismatch);
            }
            let mut dynamic = DynamicTupleStruct::new();
            dynamic.set_represented_type_name(tuple_info.type_name());
            for field in tuple_info.fields() {
                let child_schema = resolve(registry, field.type_name())?;
                let child = read_node(reader, registry, &child_schema)?;
                dynamic.insert_boxed(child);
            }
            Ok(Box::new(dynamic))
        }
        TypeInfo::Enum(enum_info) => {
            expect_tag(tag, node_tag::ENUM, "Enum")?;
            let variant_index = reader.read_len()?;
            let count = reader.read_len()?;
            let variant = enum_info
                .variant_at(variant_index)
                .ok_or(DeserializeError::UnknownVariant)?;
            use crate::type_info::VariantKind;
            let dynamic_variant = match variant.kind() {
                VariantKind::Unit => {
                    if count != 0 {
                        return Err(DeserializeError::FieldCountMismatch);
                    }
                    DynamicVariant::Unit
                }
                VariantKind::Tuple(fields) => {
                    if count != fields.len() {
                        return Err(DeserializeError::FieldCountMismatch);
                    }
                    let mut values = Vec::with_capacity(fields.len());
                    for field in fields {
                        let child_schema = resolve(registry, field.type_name())?;
                        values.push(read_node(reader, registry, &child_schema)?);
                    }
                    DynamicVariant::Tuple(values)
                }
                VariantKind::Struct(fields) => {
                    if count != fields.len() {
                        return Err(DeserializeError::FieldCountMismatch);
                    }
                    let mut values = Vec::with_capacity(fields.len());
                    for field in fields {
                        let child_schema = resolve(registry, field.type_name())?;
                        let child = read_node(reader, registry, &child_schema)?;
                        values.push((field.name(), child));
                    }
                    DynamicVariant::Struct(values)
                }
            };
            let mut dynamic =
                DynamicEnum::new(variant_index, variant.name(), dynamic_variant);
            dynamic.set_represented_type_name(enum_info.type_name());
            Ok(Box::new(dynamic))
        }
        TypeInfo::List(list_info) => {
            expect_tag(tag, node_tag::LIST, "List")?;
            let len = reader.read_len()?;
            let item_schema = resolve(registry, list_info.item_type_name())?;
            let mut dynamic = DynamicList::new();
            dynamic.set_represented_type_name(list_info.type_name());
            for _ in 0..len {
                let child = read_node(reader, registry, &item_schema)?;
                dynamic.push_boxed(child);
            }
            Ok(Box::new(dynamic))
        }
        TypeInfo::Array(array_info) => {
            expect_tag(tag, node_tag::ARRAY, "Array")?;
            let len = reader.read_len()?;
            let item_schema = resolve(registry, array_info.item_type_name())?;
            let mut dynamic = DynamicArray::new();
            dynamic.set_represented_type_name(array_info.type_name());
            for _ in 0..len {
                let child = read_node(reader, registry, &item_schema)?;
                dynamic.push_boxed(child);
            }
            Ok(Box::new(dynamic))
        }
        TypeInfo::Map(map_info) => {
            expect_tag(tag, node_tag::MAP, "Map")?;
            let len = reader.read_len()?;
            let key_schema = resolve(registry, map_info.key_type_name())?;
            let value_schema = resolve(registry, map_info.value_type_name())?;
            let mut dynamic = DynamicMap::new();
            dynamic.set_represented_type_name(map_info.type_name());
            for _ in 0..len {
                let key = read_node(reader, registry, &key_schema)?;
                let mapped = read_node(reader, registry, &value_schema)?;
                dynamic.insert_boxed(key, mapped);
            }
            Ok(Box::new(dynamic))
        }
        TypeInfo::Set(set_info) => {
            expect_tag(tag, node_tag::SET, "Set")?;
            let len = reader.read_len()?;
            let value_schema = resolve(registry, set_info.value_type_name())?;
            let mut dynamic = DynamicSet::new();
            dynamic.set_represented_type_name(set_info.type_name());
            for _ in 0..len {
                let child = read_node(reader, registry, &value_schema)?;
                dynamic.push_boxed(child);
            }
            Ok(Box::new(dynamic))
        }
        TypeInfo::Value(value_info) => {
            expect_tag(tag, node_tag::VALUE, "Value")?;
            match leaf_primitive(value_info.type_name()) {
                Some(primitive) => read_primitive(reader, primitive),
                None => Err(DeserializeError::LeafTypeMismatch {
                    expected: value_info.type_name(),
                }),
            }
        }
    }
}

/// Read a leaf primitive: a validating tag byte then its little-endian bytes.
fn read_primitive(
    reader: &mut ByteReader<'_>,
    primitive: Primitive,
) -> Result<Box<dyn Reflect>, DeserializeError> {
    let tag = reader.read_u8()?;
    let found = Primitive::from_tag(tag)?;
    if found != primitive {
        return Err(DeserializeError::LeafTypeMismatch {
            expected: primitive.type_name(),
        });
    }
    Ok(match primitive {
        Primitive::Bool => Box::new(reader.read_u8()? != 0),
        Primitive::Char => {
            let code = u32::from_le_bytes(reader.read_array::<4>()?);
            Box::new(char::from_u32(code).ok_or(DeserializeError::InvalidChar(code))?)
        }
        Primitive::I8 => Box::new(i8::from_le_bytes(reader.read_array::<1>()?)),
        Primitive::I16 => Box::new(i16::from_le_bytes(reader.read_array::<2>()?)),
        Primitive::I32 => Box::new(i32::from_le_bytes(reader.read_array::<4>()?)),
        Primitive::I64 => Box::new(i64::from_le_bytes(reader.read_array::<8>()?)),
        Primitive::I128 => Box::new(i128::from_le_bytes(reader.read_array::<16>()?)),
        Primitive::Isize => {
            let value = i64::from_le_bytes(reader.read_array::<8>()?);
            Box::new(
                isize::try_from(value).map_err(|_| DeserializeError::TrailingData)?,
            )
        }
        Primitive::U8 => Box::new(reader.read_u8()?),
        Primitive::U16 => Box::new(u16::from_le_bytes(reader.read_array::<2>()?)),
        Primitive::U32 => Box::new(u32::from_le_bytes(reader.read_array::<4>()?)),
        Primitive::U64 => Box::new(u64::from_le_bytes(reader.read_array::<8>()?)),
        Primitive::U128 => Box::new(u128::from_le_bytes(reader.read_array::<16>()?)),
        Primitive::Usize => {
            let value = u64::from_le_bytes(reader.read_array::<8>()?);
            Box::new(
                usize::try_from(value).map_err(|_| DeserializeError::TrailingData)?,
            )
        }
        Primitive::F32 => {
            Box::new(f32::from_bits(u32::from_le_bytes(reader.read_array::<4>()?)))
        }
        Primitive::F64 => {
            Box::new(f64::from_bits(u64::from_le_bytes(reader.read_array::<8>()?)))
        }
        Primitive::String => {
            let len = reader.read_len()?;
            let bytes = reader.read_bytes(len)?;
            let text = String::from_utf8(bytes.to_vec())
                .map_err(|_| DeserializeError::InvalidUtf8)?;
            Box::new(text)
        }
    })
}

/// Validate a node tag against the kind the schema expects at this position.
fn expect_tag(found: u8, expected: u8, expected_kind: &'static str) -> Result<(), DeserializeError> {
    if found == expected {
        return Ok(());
    }
    let found_kind = node_kind_name(found)?;
    Err(DeserializeError::KindMismatch {
        expected: expected_kind,
        found: found_kind,
    })
}

/// Human-readable name for a node tag (also rejects out-of-range tags).
fn node_kind_name(tag: u8) -> Result<&'static str, DeserializeError> {
    Ok(match tag {
        node_tag::STRUCT => "Struct",
        node_tag::TUPLE_STRUCT => "TupleStruct",
        node_tag::ENUM => "Enum",
        node_tag::LIST => "List",
        node_tag::ARRAY => "Array",
        node_tag::MAP => "Map",
        node_tag::SET => "Set",
        node_tag::VALUE => "Value",
        other => return Err(DeserializeError::UnknownNodeTag(other)),
    })
}
