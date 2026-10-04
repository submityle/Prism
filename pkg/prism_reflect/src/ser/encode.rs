//! The format-agnostic [`Encoder`] trait and the reflection traversal driver.
//!
//! [`serialize_value`] walks any `&dyn Reflect` through its
//! [`ReflectRef`](crate::ReflectRef) view and issues structural callbacks to an
//! [`Encoder`]; the binary and RON back-ends implement [`Encoder`] to turn
//! those callbacks into bytes or text. Leaf values are funnelled through
//! [`encode_leaf`], which downcasts through the built-in primitive set so every
//! format shares one definition of "what is a leaf".

use crate::kinds::VariantType;
use crate::reflect::Reflect;
use crate::ser::error::SerializeError;
use crate::ser::pod;
use crate::ser::primitive::Primitive;
use crate::type_info::{TypeInfo, VariantKind};
use crate::{DynamicEnum, DynamicVariant, ReflectRef};
use alloc::string::String;
use alloc::vec::Vec;

/// A format back-end that turns structural traversal callbacks into output.
///
/// The driver ([`serialize_value`]) guarantees a well-nested call sequence:
/// every `begin_*` is matched by its `end_*`, each child is preceded by the
/// matching `before_*` boundary callback, and leaf emitters are only called for
/// `Value`-kind nodes. Implementors keep whatever state (byte buffer, text
/// buffer with a delimiter stack) they need.
pub trait Encoder {
    /// Emit a `bool` leaf.
    fn encode_bool(&mut self, value: bool);
    /// Emit a `char` leaf.
    fn encode_char(&mut self, value: char);
    /// Emit an `i8` leaf.
    fn encode_i8(&mut self, value: i8);
    /// Emit an `i16` leaf.
    fn encode_i16(&mut self, value: i16);
    /// Emit an `i32` leaf.
    fn encode_i32(&mut self, value: i32);
    /// Emit an `i64` leaf.
    fn encode_i64(&mut self, value: i64);
    /// Emit an `i128` leaf.
    fn encode_i128(&mut self, value: i128);
    /// Emit an `isize` leaf.
    fn encode_isize(&mut self, value: isize);
    /// Emit a `u8` leaf.
    fn encode_u8(&mut self, value: u8);
    /// Emit a `u16` leaf.
    fn encode_u16(&mut self, value: u16);
    /// Emit a `u32` leaf.
    fn encode_u32(&mut self, value: u32);
    /// Emit a `u64` leaf.
    fn encode_u64(&mut self, value: u64);
    /// Emit a `u128` leaf.
    fn encode_u128(&mut self, value: u128);
    /// Emit a `usize` leaf.
    fn encode_usize(&mut self, value: usize);
    /// Emit an `f32` leaf.
    fn encode_f32(&mut self, value: f32);
    /// Emit an `f64` leaf.
    fn encode_f64(&mut self, value: f64);
    /// Emit a string leaf.
    fn encode_str(&mut self, value: &str);

    /// Whether this back-end wants homogeneous fixed-width numeric sequences
    /// delivered as a single bulk blob via [`encode_pod_blob`](Self::encode_pod_blob)
    /// instead of per-element callbacks (design §24.3).
    ///
    /// Defaults to `false`: text and other back-ends keep the per-element path.
    /// A back-end that returns `true` **must** override
    /// [`encode_pod_blob`](Self::encode_pod_blob); the driver only calls it when
    /// this returns `true`.
    fn wants_pod_blobs(&self) -> bool {
        false
    }

    /// Emit a homogeneous fixed-width numeric sequence as one bulk blob.
    ///
    /// `raw_le` holds `count` little-endian elements of `primitive`
    /// back-to-back. Called only when [`wants_pod_blobs`](Self::wants_pod_blobs)
    /// returns `true`; the default panics to flag a back-end that opted in
    /// without providing an implementation.
    fn encode_pod_blob(&mut self, primitive: Primitive, count: usize, raw_le: &[u8]) {
        let _ = (primitive, count, raw_le);
        unreachable!("encode_pod_blob called without opting in via wants_pod_blobs");
    }

    /// Begin a named-field struct with `count` fields.
    fn begin_struct(&mut self, count: usize);
    /// Boundary before the struct field named `name` at position `index`.
    fn before_struct_field(&mut self, name: &str, index: usize);
    /// End the current struct.
    fn end_struct(&mut self);

    /// Begin a tuple struct with `count` fields.
    fn begin_tuple_struct(&mut self, count: usize);
    /// Boundary before the tuple-struct field at `index`.
    fn before_tuple_struct_field(&mut self, index: usize);
    /// End the current tuple struct.
    fn end_tuple_struct(&mut self);

    /// Begin an enum value with the active variant's identity and field count.
    fn begin_enum(
        &mut self,
        variant_index: usize,
        variant_name: &str,
        variant_type: VariantType,
        count: usize,
    );
    /// Boundary before a tuple-variant field at `index`.
    fn before_enum_tuple_field(&mut self, index: usize);
    /// Boundary before a struct-variant field named `name` at `index`.
    fn before_enum_struct_field(&mut self, name: &str, index: usize);
    /// End the current enum value.
    fn end_enum(&mut self);

    /// Begin a list of `len` elements.
    fn begin_list(&mut self, len: usize);
    /// Boundary before the list element at `index`.
    fn before_list_element(&mut self, index: usize);
    /// End the current list.
    fn end_list(&mut self);

    /// Begin an array of `len` elements.
    fn begin_array(&mut self, len: usize);
    /// Boundary before the array element at `index`.
    fn before_array_element(&mut self, index: usize);
    /// End the current array.
    fn end_array(&mut self);

    /// Begin a set of `len` elements.
    fn begin_set(&mut self, len: usize);
    /// Boundary before the set element at `index`.
    fn before_set_element(&mut self, index: usize);
    /// End the current set.
    fn end_set(&mut self);

    /// Begin a map of `len` entries.
    fn begin_map(&mut self, len: usize);
    /// Boundary before the key of entry `index`.
    fn before_map_key(&mut self, index: usize);
    /// Boundary before the value of entry `index`.
    fn before_map_value(&mut self, index: usize);
    /// End the current map.
    fn end_map(&mut self);
}

/// Walk `value` and drive `encoder` with the matching structural callbacks.
///
/// # Errors
/// Returns [`SerializeError::UnsupportedLeaf`] when a `Value`-kind node holds a
/// concrete type outside the built-in leaf set.
pub fn serialize_value(
    value: &dyn Reflect,
    encoder: &mut dyn Encoder,
) -> Result<(), SerializeError> {
    match value.reflect_ref() {
        ReflectRef::Struct(s) => {
            let count = s.field_count();
            encoder.begin_struct(count);
            for index in 0..count {
                let name = s.name_at(index).unwrap_or("");
                encoder.before_struct_field(name, index);
                let field = s
                    .field_at(index)
                    .expect("field_at(index) within field_count must exist");
                serialize_value(field, encoder)?;
            }
            encoder.end_struct();
        }
        ReflectRef::TupleStruct(ts) => {
            let count = ts.field_count();
            encoder.begin_tuple_struct(count);
            for index in 0..count {
                encoder.before_tuple_struct_field(index);
                let field = ts
                    .field(index)
                    .expect("field(index) within field_count must exist");
                serialize_value(field, encoder)?;
            }
            encoder.end_tuple_struct();
        }
        ReflectRef::Enum(e) => {
            let count = e.field_count();
            let variant_type = e.variant_type();
            encoder.begin_enum(e.variant_index(), e.variant_name(), variant_type, count);
            match variant_type {
                VariantType::Unit => {}
                VariantType::Tuple => {
                    for index in 0..count {
                        encoder.before_enum_tuple_field(index);
                        let field = e
                            .field_at(index)
                            .expect("tuple-variant field within count must exist");
                        serialize_value(field, encoder)?;
                    }
                }
                VariantType::Struct => {
                    let names = enum_struct_field_names(value, e.variant_name()).ok_or(
                        SerializeError::UnsupportedLeaf {
                            type_name: value.type_name(),
                        },
                    )?;
                    for index in 0..count {
                        let name = names.get(index).copied().unwrap_or("");
                        encoder.before_enum_struct_field(name, index);
                        let field = e
                            .field_at(index)
                            .expect("struct-variant field within count must exist");
                        serialize_value(field, encoder)?;
                    }
                }
            }
            encoder.end_enum();
        }
        ReflectRef::List(list) => {
            let len = list.len();
            let blob = if encoder.wants_pod_blobs() {
                pod::collect(list.iter_reflect(), len)
            } else {
                None
            };
            if let Some((primitive, raw)) = blob {
                encoder.encode_pod_blob(primitive, len, &raw);
            } else {
                encoder.begin_list(len);
                for index in 0..len {
                    encoder.before_list_element(index);
                    let element = list.get(index).expect("list element within len must exist");
                    serialize_value(element, encoder)?;
                }
                encoder.end_list();
            }
        }
        ReflectRef::Array(array) => {
            let len = array.len();
            let blob = if encoder.wants_pod_blobs() {
                pod::collect(array.iter_reflect(), len)
            } else {
                None
            };
            if let Some((primitive, raw)) = blob {
                encoder.encode_pod_blob(primitive, len, &raw);
            } else {
                encoder.begin_array(len);
                for index in 0..len {
                    encoder.before_array_element(index);
                    let element = array
                        .get(index)
                        .expect("array element within len must exist");
                    serialize_value(element, encoder)?;
                }
                encoder.end_array();
            }
        }
        ReflectRef::Set(set) => {
            let len = set.len();
            encoder.begin_set(len);
            for (index, element) in set.iter_reflect().enumerate() {
                encoder.before_set_element(index);
                serialize_value(element, encoder)?;
            }
            encoder.end_set();
        }
        ReflectRef::Map(map) => {
            let len = map.len();
            encoder.begin_map(len);
            for (index, (key, mapped)) in map.iter_reflect().enumerate() {
                encoder.before_map_key(index);
                serialize_value(key, encoder)?;
                encoder.before_map_value(index);
                serialize_value(mapped, encoder)?;
            }
            encoder.end_map();
        }
        ReflectRef::Value(leaf) => encode_leaf(leaf, encoder)?,
    }
    Ok(())
}

/// Emit a single leaf value by downcasting through the built-in primitive set.
///
/// # Errors
/// Returns [`SerializeError::UnsupportedLeaf`] when `leaf` is not one of the
/// supported primitive types.
pub fn encode_leaf(leaf: &dyn Reflect, encoder: &mut dyn Encoder) -> Result<(), SerializeError> {
    let any = leaf.as_any();
    if let Some(v) = any.downcast_ref::<bool>() {
        encoder.encode_bool(*v);
    } else if let Some(v) = any.downcast_ref::<char>() {
        encoder.encode_char(*v);
    } else if let Some(v) = any.downcast_ref::<i8>() {
        encoder.encode_i8(*v);
    } else if let Some(v) = any.downcast_ref::<i16>() {
        encoder.encode_i16(*v);
    } else if let Some(v) = any.downcast_ref::<i32>() {
        encoder.encode_i32(*v);
    } else if let Some(v) = any.downcast_ref::<i64>() {
        encoder.encode_i64(*v);
    } else if let Some(v) = any.downcast_ref::<i128>() {
        encoder.encode_i128(*v);
    } else if let Some(v) = any.downcast_ref::<isize>() {
        encoder.encode_isize(*v);
    } else if let Some(v) = any.downcast_ref::<u8>() {
        encoder.encode_u8(*v);
    } else if let Some(v) = any.downcast_ref::<u16>() {
        encoder.encode_u16(*v);
    } else if let Some(v) = any.downcast_ref::<u32>() {
        encoder.encode_u32(*v);
    } else if let Some(v) = any.downcast_ref::<u64>() {
        encoder.encode_u64(*v);
    } else if let Some(v) = any.downcast_ref::<u128>() {
        encoder.encode_u128(*v);
    } else if let Some(v) = any.downcast_ref::<usize>() {
        encoder.encode_usize(*v);
    } else if let Some(v) = any.downcast_ref::<f32>() {
        encoder.encode_f32(*v);
    } else if let Some(v) = any.downcast_ref::<f64>() {
        encoder.encode_f64(*v);
    } else if let Some(v) = any.downcast_ref::<String>() {
        encoder.encode_str(v);
    } else {
        return Err(SerializeError::UnsupportedLeaf {
            type_name: leaf.type_name(),
        });
    }
    Ok(())
}

/// Resolve the ordered field names of the active struct variant.
///
/// Concrete enums carry the names in their static [`TypeInfo`]; a
/// [`DynamicEnum`] has an empty `TypeInfo`, so its payload is inspected
/// directly. Returns `None` only when neither source can supply names, which
/// the driver reports as an unsupported value.
fn enum_struct_field_names(value: &dyn Reflect, variant_name: &str) -> Option<Vec<&'static str>> {
    if let TypeInfo::Enum(info) = value.type_info()
        && let Some(variant) = info.variant(variant_name)
        && let VariantKind::Struct(fields) = variant.kind()
    {
        return Some(
            fields
                .iter()
                .map(crate::type_info::NamedField::name)
                .collect(),
        );
    }
    if let Some(dynamic) = value.as_any().downcast_ref::<DynamicEnum>()
        && let DynamicVariant::Struct(fields) = dynamic.variant()
    {
        return Some(fields.iter().map(|(name, _)| *name).collect());
    }
    None
}
