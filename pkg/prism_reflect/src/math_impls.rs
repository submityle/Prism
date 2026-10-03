//! `Reflect` implementations for `prism_math` value types.
//!
//! Gated behind the `math` feature. Each type is reflected as a `Struct` kind
//! so editors/serializers can traverse its named components, and ships a
//! [`GetTypeRegistration`](crate::registry::GetTypeRegistration) impl for the
//! [`TypeRegistry`](crate::registry::TypeRegistry).

use crate::reflect::{Reflect, Struct, Typed};
use crate::type_info::{NamedField, StructInfo, TypeInfo};
use crate::{ReflectMut, ReflectRef};
use prism_math::{Mat4, Quat, Vec2, Vec3, Vec3A, Vec4};
use std::boxed::Box;
use std::sync::OnceLock;
use std::vec;

macro_rules! impl_reflect_math_struct {
    ($ty:ty { $($field:ident : $fty:ty),+ $(,)? }) => {
        impl Struct for $ty {
            fn field(&self, name: &str) -> Option<&dyn Reflect> {
                match name {
                    $( stringify!($field) => Some(&self.$field as &dyn Reflect), )+
                    _ => None,
                }
            }
            fn field_mut(&mut self, name: &str) -> Option<&mut dyn Reflect> {
                match name {
                    $( stringify!($field) => Some(&mut self.$field as &mut dyn Reflect), )+
                    _ => None,
                }
            }
            fn field_at(&self, index: usize) -> Option<&dyn Reflect> {
                [$( &self.$field as &dyn Reflect ),+].into_iter().nth(index)
            }
            fn name_at(&self, index: usize) -> Option<&'static str> {
                [$( stringify!($field) ),+].get(index).copied()
            }
            fn field_count(&self) -> usize {
                [$( stringify!($field) ),+].len()
            }
        }

        impl Reflect for $ty {
            fn type_name(&self) -> &'static str {
                ::core::any::type_name::<$ty>()
            }
            fn type_info(&self) -> &'static TypeInfo {
                <$ty as Typed>::type_info()
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
                ReflectRef::Struct(self)
            }
            fn reflect_mut(&mut self) -> ReflectMut<'_> {
                ReflectMut::Struct(self)
            }
            fn reflect_clone(&self) -> Box<dyn Reflect> {
                Box::new(::core::clone::Clone::clone(self))
            }
        }

        impl crate::FromReflect for $ty {
            fn from_reflect(reflect: &dyn Reflect) -> Option<Self> {
                reflect.as_any().downcast_ref::<$ty>().cloned()
            }
        }

        impl Typed for $ty {
            fn type_info() -> &'static TypeInfo {
                static CELL: OnceLock<TypeInfo> = OnceLock::new();
                CELL.get_or_init(|| {
                    TypeInfo::Struct(StructInfo::new(
                        ::core::any::type_name::<$ty>(),
                        vec![
                            $( NamedField::new(
                                stringify!($field),
                                ::core::any::type_name::<$fty>(),
                            ), )+
                        ],
                    ))
                })
            }
        }

        impl crate::registry::GetTypeRegistration for $ty {
            fn get_type_registration() -> crate::registry::TypeRegistration {
                crate::registry::TypeRegistration::of::<$ty>()
            }
        }
    };
}

impl_reflect_math_struct!(Vec2 { x: f32, y: f32 });
impl_reflect_math_struct!(Vec3 { x: f32, y: f32, z: f32 });
impl_reflect_math_struct!(Vec3A { x: f32, y: f32, z: f32 });
impl_reflect_math_struct!(Vec4 { x: f32, y: f32, z: f32, w: f32 });
impl_reflect_math_struct!(Quat { x: f32, y: f32, z: f32, w: f32 });
impl_reflect_math_struct!(Mat4 {
    x_axis: Vec4,
    y_axis: Vec4,
    z_axis: Vec4,
    w_axis: Vec4,
});
