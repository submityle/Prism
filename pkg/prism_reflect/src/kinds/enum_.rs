//! The `Enum` kind: Rust enums with unit, tuple, and struct variants.

use crate::reflect::Reflect;
use crate::type_info::{EnumInfo, TypeInfo, UnnamedField, VariantInfo, VariantKind};
use crate::{ReflectMut, ReflectRef, Typed};
use std::boxed::Box;
use std::vec;

/// The structural shape of the currently-active enum variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VariantType {
    /// A unit variant with no fields.
    Unit,
    /// A tuple variant with positional fields.
    Tuple,
    /// A struct variant with named fields.
    Struct,
}

/// Reflected access to an enum value and its active variant.
pub trait Enum: Reflect {
    /// The name of the active variant.
    fn variant_name(&self) -> &'static str;
    /// The declaration index of the active variant.
    fn variant_index(&self) -> usize;
    /// The structural shape of the active variant.
    fn variant_type(&self) -> VariantType;
    /// Borrow a named field of the active (struct) variant.
    fn field(&self, name: &str) -> Option<&dyn Reflect>;
    /// Mutably borrow a named field of the active (struct) variant.
    fn field_mut(&mut self, name: &str) -> Option<&mut dyn Reflect>;
    /// Borrow a positional field of the active (tuple/struct) variant.
    fn field_at(&self, index: usize) -> Option<&dyn Reflect>;
    /// Mutably borrow a positional field of the active (tuple/struct) variant.
    fn field_at_mut(&mut self, index: usize) -> Option<&mut dyn Reflect>;
    /// Number of fields carried by the active variant.
    fn field_count(&self) -> usize;
}

impl<T: Reflect + Typed> Enum for Option<T> {
    fn variant_name(&self) -> &'static str {
        match self {
            Some(_) => "Some",
            None => "None",
        }
    }

    fn variant_index(&self) -> usize {
        match self {
            None => 0,
            Some(_) => 1,
        }
    }

    fn variant_type(&self) -> VariantType {
        match self {
            None => VariantType::Unit,
            Some(_) => VariantType::Tuple,
        }
    }

    fn field(&self, _name: &str) -> Option<&dyn Reflect> {
        None
    }

    fn field_mut(&mut self, _name: &str) -> Option<&mut dyn Reflect> {
        None
    }

    fn field_at(&self, index: usize) -> Option<&dyn Reflect> {
        match (self, index) {
            (Some(value), 0) => Some(value as &dyn Reflect),
            _ => None,
        }
    }

    fn field_at_mut(&mut self, index: usize) -> Option<&mut dyn Reflect> {
        match (self, index) {
            (Some(value), 0) => Some(value as &mut dyn Reflect),
            _ => None,
        }
    }

    fn field_count(&self) -> usize {
        match self {
            Some(_) => 1,
            None => 0,
        }
    }
}

impl<T: Reflect + Typed> Reflect for Option<T> {
    fn type_name(&self) -> &'static str {
        ::core::any::type_name::<Self>()
    }
    fn type_info(&self) -> &'static TypeInfo {
        <Self as Typed>::type_info()
    }
    fn as_any(&self) -> &dyn ::core::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn ::core::any::Any> {
        self
    }
    fn as_reflect(&self) -> &dyn Reflect {
        self
    }
    fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
        self
    }
    fn reflect_ref(&self) -> ReflectRef<'_> {
        ReflectRef::Enum(self)
    }
    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Enum(self)
    }
}

impl<T: Reflect + Typed> Typed for Option<T> {
    fn type_info() -> &'static TypeInfo {
        crate::cache::intern::<Self, _>(|| {
            TypeInfo::Enum(EnumInfo::new(
                ::core::any::type_name::<Self>(),
                vec![
                    VariantInfo::new("None", 0, VariantKind::Unit),
                    VariantInfo::new(
                        "Some",
                        1,
                        VariantKind::Tuple(vec![UnnamedField::new(
                            0,
                            ::core::any::type_name::<T>(),
                        )]),
                    ),
                ],
            ))
        })
    }
}

impl<T: Reflect + Typed, E: Reflect + Typed> Enum for Result<T, E> {
    fn variant_name(&self) -> &'static str {
        match self {
            Ok(_) => "Ok",
            Err(_) => "Err",
        }
    }

    fn variant_index(&self) -> usize {
        match self {
            Ok(_) => 0,
            Err(_) => 1,
        }
    }

    fn variant_type(&self) -> VariantType {
        VariantType::Tuple
    }

    fn field(&self, _name: &str) -> Option<&dyn Reflect> {
        None
    }

    fn field_mut(&mut self, _name: &str) -> Option<&mut dyn Reflect> {
        None
    }

    fn field_at(&self, index: usize) -> Option<&dyn Reflect> {
        match (self, index) {
            (Ok(value), 0) => Some(value as &dyn Reflect),
            (Err(value), 0) => Some(value as &dyn Reflect),
            _ => None,
        }
    }

    fn field_at_mut(&mut self, index: usize) -> Option<&mut dyn Reflect> {
        match (self, index) {
            (Ok(value), 0) => Some(value as &mut dyn Reflect),
            (Err(value), 0) => Some(value as &mut dyn Reflect),
            _ => None,
        }
    }

    fn field_count(&self) -> usize {
        1
    }
}

impl<T: Reflect + Typed, E: Reflect + Typed> Reflect for Result<T, E> {
    fn type_name(&self) -> &'static str {
        ::core::any::type_name::<Self>()
    }
    fn type_info(&self) -> &'static TypeInfo {
        <Self as Typed>::type_info()
    }
    fn as_any(&self) -> &dyn ::core::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any {
        self
    }
    fn into_any(self: Box<Self>) -> Box<dyn ::core::any::Any> {
        self
    }
    fn as_reflect(&self) -> &dyn Reflect {
        self
    }
    fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
        self
    }
    fn reflect_ref(&self) -> ReflectRef<'_> {
        ReflectRef::Enum(self)
    }
    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Enum(self)
    }
}

impl<T: Reflect + Typed, E: Reflect + Typed> Typed for Result<T, E> {
    fn type_info() -> &'static TypeInfo {
        crate::cache::intern::<Self, _>(|| {
            TypeInfo::Enum(EnumInfo::new(
                ::core::any::type_name::<Self>(),
                vec![
                    VariantInfo::new(
                        "Ok",
                        0,
                        VariantKind::Tuple(vec![UnnamedField::new(
                            0,
                            ::core::any::type_name::<T>(),
                        )]),
                    ),
                    VariantInfo::new(
                        "Err",
                        1,
                        VariantKind::Tuple(vec![UnnamedField::new(
                            0,
                            ::core::any::type_name::<E>(),
                        )]),
                    ),
                ],
            ))
        })
    }
}

impl<T: Reflect + Typed> crate::registry::GetTypeRegistration for Option<T> {
    fn get_type_registration() -> crate::registry::TypeRegistration {
        crate::registry::TypeRegistration::of::<Self>()
    }
}

impl<T: Reflect + Typed, E: Reflect + Typed> crate::registry::GetTypeRegistration
    for Result<T, E>
{
    fn get_type_registration() -> crate::registry::TypeRegistration {
        crate::registry::TypeRegistration::of::<Self>()
    }
}
