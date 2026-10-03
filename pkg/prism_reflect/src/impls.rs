//! `Reflect` + `Typed` + `GetTypeRegistration` implementations for leaf
//! (`Value`-kind) std types.

use crate::reflect::{Reflect, Typed};
use crate::registry::GetTypeRegistration;

/// Implement `Reflect` + `Typed` + `GetTypeRegistration` (as a `Value` leaf)
/// for each given type.
#[macro_export]
macro_rules! impl_reflect_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl $crate::Reflect for $ty {
                fn type_name(&self) -> &'static str { ::core::any::type_name::<$ty>() }
                fn type_info(&self) -> &'static $crate::TypeInfo {
                    <$ty as $crate::Typed>::type_info()
                }
                fn as_any(&self) -> &dyn ::core::any::Any { self }
                fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any { self }
                fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn ::core::any::Any> { self }
                fn as_reflect(&self) -> &dyn $crate::Reflect { self }
                fn as_reflect_mut(&mut self) -> &mut dyn $crate::Reflect { self }
                fn reflect_ref(&self) -> $crate::ReflectRef<'_> {
                    $crate::ReflectRef::Value(self)
                }
                fn reflect_mut(&mut self) -> $crate::ReflectMut<'_> {
                    $crate::ReflectMut::Value(self)
                }
                fn reflect_clone(&self) -> ::std::boxed::Box<dyn $crate::Reflect> {
                    ::std::boxed::Box::new(::core::clone::Clone::clone(self))
                }
                fn apply(
                    &mut self,
                    source: &dyn $crate::Reflect,
                ) -> ::core::result::Result<(), $crate::ApplyError> {
                    match $crate::Reflect::as_any(source).downcast_ref::<$ty>() {
                        ::core::option::Option::Some(value) => {
                            *self = ::core::clone::Clone::clone(value);
                            ::core::result::Result::Ok(())
                        }
                        ::core::option::Option::None => {
                            ::core::result::Result::Err($crate::ApplyError::TypeMismatch {
                                source: $crate::Reflect::type_name(source),
                                target: ::core::any::type_name::<$ty>(),
                            })
                        }
                    }
                }
            }

            impl $crate::FromReflect for $ty {
                fn from_reflect(
                    reflect: &dyn $crate::Reflect,
                ) -> ::core::option::Option<Self> {
                    $crate::Reflect::as_any(reflect)
                        .downcast_ref::<$ty>()
                        .cloned()
                }
            }

            impl $crate::Typed for $ty {
                fn type_info() -> &'static $crate::TypeInfo {
                    static CELL: ::std::sync::OnceLock<$crate::TypeInfo> =
                        ::std::sync::OnceLock::new();
                    CELL.get_or_init(|| {
                        $crate::TypeInfo::Value($crate::ValueInfo::new(
                            ::core::any::type_name::<$ty>(),
                        ))
                    })
                }
            }

            impl $crate::registry::GetTypeRegistration for $ty {
                fn get_type_registration() -> $crate::registry::TypeRegistration {
                    $crate::registry::TypeRegistration::of::<$ty>()
                }
            }
        )*
    };
}

impl_reflect_value!(bool, char);
impl_reflect_value!(i8, i16, i32, i64, i128, isize);
impl_reflect_value!(u8, u16, u32, u64, u128, usize);
impl_reflect_value!(f32, f64);
impl_reflect_value!(String);

// Compile-time check that the leaf impls satisfy the core traits.
const _: fn() = || {
    fn _assert<T: Reflect + Typed + GetTypeRegistration + crate::FromReflect>() {}
    _assert::<i32>();
    _assert::<bool>();
    _assert::<String>();
};
