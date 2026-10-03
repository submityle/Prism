//! Recursive state transfer between reflected values (`apply`/patch).
//!
//! [`apply_impl`] is the engine behind [`Reflect::apply`](crate::Reflect::apply):
//! it copies the state of a *source* value into a *target* value of a matching
//! kind, recursing field-by-field (structs by field name, tuple structs and
//! enums by field index, lists/arrays by index, maps by key, sets by value).
//! Leaf (`Value`-kind) types short-circuit this machinery by overriding
//! [`Reflect::apply`](crate::Reflect::apply) with a direct clone-assign.
//!
//! A [`DynamicEnum`](crate::DynamicEnum) target may *switch variants* when the
//! source selects a different one; concrete enums require the active variant to
//! match and only patch its fields.

use crate::dynamic::{DynamicEnum, DynamicVariant};
use crate::kinds::{Array, Enum, List, Map, Set, VariantType};
use crate::reflect::{Reflect, Struct, TupleStruct};
use crate::type_info::{TypeInfo, VariantKind};
use crate::{ReflectMut, ReflectRef};
use core::fmt;
use std::vec::Vec;

/// An error produced while [`apply`](crate::Reflect::apply)-ing one reflected
/// value onto another.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ApplyError {
    /// The source and target reflect different kinds (e.g. a list onto a map).
    KindMismatch {
        /// The source value's type name.
        source: &'static str,
        /// The target value's type name.
        target: &'static str,
    },
    /// The source and target are leaf values of incompatible concrete types.
    TypeMismatch {
        /// The source value's type name.
        source: &'static str,
        /// The target value's type name.
        target: &'static str,
    },
    /// A concrete enum target cannot change to the source's active variant.
    VariantMismatch {
        /// The source variant name.
        source: &'static str,
        /// The target variant name.
        target: &'static str,
    },
    /// A list/array element from the source is not accepted by the target.
    IncompatibleElement {
        /// The target container's type name.
        target: &'static str,
    },
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApplyError::KindMismatch { source, target } => {
                write!(f, "cannot apply `{source}` onto `{target}`: kind mismatch")
            }
            ApplyError::TypeMismatch { source, target } => {
                write!(f, "cannot apply `{source}` onto `{target}`: type mismatch")
            }
            ApplyError::VariantMismatch { source, target } => write!(
                f,
                "cannot apply variant `{source}` onto concrete variant `{target}`"
            ),
            ApplyError::IncompatibleElement { target } => {
                write!(f, "incompatible element pushed into `{target}`")
            }
        }
    }
}

impl std::error::Error for ApplyError {}

/// Copy the state of `source` into `target`, dispatching on `target`'s kind.
///
/// This is the default behaviour of [`Reflect::apply`](crate::Reflect::apply);
/// leaf value types override `apply` directly and never reach here.
///
/// # Errors
/// Returns an [`ApplyError`] when the two values' kinds disagree, a leaf type
/// mismatch occurs, a concrete enum variant cannot be switched, or a container
/// rejects an element.
pub(crate) fn apply_impl(target: &mut dyn Reflect, source: &dyn Reflect) -> Result<(), ApplyError> {
    match target.reflect_mut() {
        ReflectMut::Struct(dst) => apply_struct(dst, source),
        ReflectMut::TupleStruct(dst) => apply_tuple_struct(dst, source),
        ReflectMut::Enum(dst) => apply_enum(dst, source),
        ReflectMut::List(dst) => apply_list(dst, source),
        ReflectMut::Array(dst) => apply_array(dst, source),
        ReflectMut::Map(dst) => apply_map(dst, source),
        ReflectMut::Set(dst) => apply_set(dst, source),
        ReflectMut::Value(dst) => Err(ApplyError::TypeMismatch {
            source: source.type_name(),
            target: dst.type_name(),
        }),
    }
}

/// Patch a struct target field-by-field, matching source fields by name.
fn apply_struct(dst: &mut dyn Struct, source: &dyn Reflect) -> Result<(), ApplyError> {
    let target_name = dst.type_name();
    let ReflectRef::Struct(src) = source.reflect_ref() else {
        return Err(ApplyError::KindMismatch {
            source: source.type_name(),
            target: target_name,
        });
    };
    for index in 0..src.field_count() {
        let Some(name) = src.name_at(index) else {
            continue;
        };
        let Some(src_field) = src.field_at(index) else {
            continue;
        };
        if let Some(dst_field) = dst.field_mut(name) {
            dst_field.apply(src_field)?;
        }
    }
    Ok(())
}

/// Patch a tuple struct target field-by-field, matching source fields by index.
fn apply_tuple_struct(dst: &mut dyn TupleStruct, source: &dyn Reflect) -> Result<(), ApplyError> {
    let target_name = dst.type_name();
    let ReflectRef::TupleStruct(src) = source.reflect_ref() else {
        return Err(ApplyError::KindMismatch {
            source: source.type_name(),
            target: target_name,
        });
    };
    for index in 0..src.field_count() {
        let Some(src_field) = src.field(index) else {
            continue;
        };
        if let Some(dst_field) = dst.field_mut(index) {
            dst_field.apply(src_field)?;
        }
    }
    Ok(())
}

/// Build the field names of the source enum's active struct variant.
fn struct_variant_field_names(src: &dyn Enum) -> Vec<&'static str> {
    if let TypeInfo::Enum(info) = src.type_info() {
        if let Some(variant) = info.variant_at(src.variant_index()) {
            if let VariantKind::Struct(fields) = variant.kind() {
                return fields.iter().map(crate::NamedField::name).collect();
            }
        }
    }
    Vec::new()
}

/// Reconstruct a [`DynamicVariant`] mirroring the source enum's active variant.
fn build_dynamic_variant(src: &dyn Enum) -> DynamicVariant {
    if let Some(dynamic) = src.as_any().downcast_ref::<DynamicEnum>() {
        return dynamic.clone_variant();
    }
    match src.variant_type() {
        VariantType::Unit => DynamicVariant::Unit,
        VariantType::Tuple => {
            let mut fields = Vec::with_capacity(src.field_count());
            for index in 0..src.field_count() {
                if let Some(field) = src.field_at(index) {
                    fields.push(field.reflect_clone());
                }
            }
            DynamicVariant::Tuple(fields)
        }
        VariantType::Struct => {
            let names = struct_variant_field_names(src);
            let mut fields = Vec::with_capacity(names.len());
            for (index, name) in names.iter().enumerate() {
                if let Some(field) = src.field_at(index) {
                    fields.push((*name, field.reflect_clone()));
                }
            }
            DynamicVariant::Struct(fields)
        }
    }
}

/// Patch an enum target: switch variants for a [`DynamicEnum`], else patch the
/// matching concrete variant's fields.
fn apply_enum(dst: &mut dyn Enum, source: &dyn Reflect) -> Result<(), ApplyError> {
    let target_name = dst.type_name();
    let ReflectRef::Enum(src) = source.reflect_ref() else {
        return Err(ApplyError::KindMismatch {
            source: source.type_name(),
            target: target_name,
        });
    };

    if dst.as_any().is::<DynamicEnum>() {
        let variant = build_dynamic_variant(src);
        let variant_index = src.variant_index();
        let variant_name = src.variant_name();
        let dynamic = dst
            .as_any_mut()
            .downcast_mut::<DynamicEnum>()
            .expect("checked that the target is a DynamicEnum immediately above");
        dynamic.set_variant(variant_index, variant_name, variant);
        return Ok(());
    }

    if dst.variant_name() != src.variant_name() {
        return Err(ApplyError::VariantMismatch {
            source: src.variant_name(),
            target: dst.variant_name(),
        });
    }
    for index in 0..src.field_count() {
        if let Some(src_field) = src.field_at(index) {
            if let Some(dst_field) = dst.field_at_mut(index) {
                dst_field.apply(src_field)?;
            }
        }
    }
    Ok(())
}

/// Patch a list target: recurse into shared indices, then grow with clones.
fn apply_list(dst: &mut dyn List, source: &dyn Reflect) -> Result<(), ApplyError> {
    let target_name = dst.type_name();
    let ReflectRef::List(src) = source.reflect_ref() else {
        return Err(ApplyError::KindMismatch {
            source: source.type_name(),
            target: target_name,
        });
    };
    let src_len = src.len();
    let shared = src_len.min(dst.len());
    for index in 0..shared {
        if let Some(src_el) = src.get(index) {
            if let Some(dst_el) = dst.get_mut(index) {
                dst_el.apply(src_el)?;
            }
        }
    }
    for index in dst.len()..src_len {
        if let Some(src_el) = src.get(index) {
            let cloned = src_el.reflect_clone();
            dst.push(cloned)
                .map_err(|_| ApplyError::IncompatibleElement {
                    target: target_name,
                })?;
        }
    }
    Ok(())
}

/// Patch an array target: recurse into indices shared by both lengths.
fn apply_array(dst: &mut dyn Array, source: &dyn Reflect) -> Result<(), ApplyError> {
    let target_name = dst.type_name();
    let ReflectRef::Array(src) = source.reflect_ref() else {
        return Err(ApplyError::KindMismatch {
            source: source.type_name(),
            target: target_name,
        });
    };
    let shared = src.len().min(dst.len());
    for index in 0..shared {
        if let Some(src_el) = src.get(index) {
            if let Some(dst_el) = dst.get_mut(index) {
                dst_el.apply(src_el)?;
            }
        }
    }
    Ok(())
}

/// Patch a map target: recurse into existing keys, insert clones for new ones.
fn apply_map(dst: &mut dyn Map, source: &dyn Reflect) -> Result<(), ApplyError> {
    let target_name = dst.type_name();
    let ReflectRef::Map(src) = source.reflect_ref() else {
        return Err(ApplyError::KindMismatch {
            source: source.type_name(),
            target: target_name,
        });
    };
    for (key, value) in src.iter_reflect() {
        if dst.get(key).is_some() {
            if let Some(dst_value) = dst.get_mut(key) {
                dst_value.apply(value)?;
            }
        } else {
            let _ = dst.insert(key.reflect_clone(), value.reflect_clone());
        }
    }
    Ok(())
}

/// Patch a set target: insert clones of any source values not already present.
fn apply_set(dst: &mut dyn Set, source: &dyn Reflect) -> Result<(), ApplyError> {
    let target_name = dst.type_name();
    let ReflectRef::Set(src) = source.reflect_ref() else {
        return Err(ApplyError::KindMismatch {
            source: source.type_name(),
            target: target_name,
        });
    };
    for value in src.iter_reflect() {
        if !dst.contains(value) {
            let _ = dst.insert(value.reflect_clone());
        }
    }
    Ok(())
}
