//! Reflected trait objects (design §13, §22 — milestone **M5**).
//!
//! `#[reflect_trait]` in Bevy is an attribute macro; here the same capability
//! is offered as the declarative [`reflect_trait!`] macro. It generates a
//! [`TypeData`](crate::TypeData) payload (conventionally `ReflectMyTrait`) that,
//! once attached to a registered type, recovers a `&dyn MyTrait` from a
//! `&dyn Reflect`. That enables **data-driven polymorphic dispatch**: given a
//! reflected value and the registry, call a trait method without knowing the
//! concrete type (e.g. an editor drawing a gizmo for every `Drawable`).
//!
//! ```
//! use prism_reflect::{Reflect, TypeRegistry, reflect_trait};
//!
//! trait Describe: Reflect {
//!     fn describe(&self) -> String;
//! }
//!
//! reflect_trait!(
//!     /// Reflected accessor for [`Describe`].
//!     pub ReflectDescribe for Describe
//! );
//!
//! #[derive(Reflect)]
//! struct Widget {
//!     label: String,
//! }
//!
//! impl Describe for Widget {
//!     fn describe(&self) -> String {
//!         format!("widget: {}", self.label)
//!     }
//! }
//!
//! let mut registry = TypeRegistry::new();
//! registry.register::<Widget>();
//! registry.register_type_data::<Widget, ReflectDescribe>(ReflectDescribe::from_type::<Widget>());
//!
//! let value = Widget { label: "ok".into() };
//! let registration = registry.get(value.as_any().type_id()).expect("registered");
//! let reflect_describe = registration.data::<ReflectDescribe>().expect("type data");
//! let described = reflect_describe.get(&value).expect("downcast").describe();
//! assert_eq!(described, "widget: ok");
//! ```

/// Generate a [`TypeData`](crate::TypeData) accessor for a trait so a
/// `&dyn Trait` can be recovered from a `&dyn Reflect`.
///
/// The invocation `reflect_trait!($(#[meta])* $vis $Data for $Trait)` defines a
/// `Copy` struct `$Data` holding type-erased accessors. Build one per concrete
/// type with `$Data::from_type::<T>()` (where `T: $Trait + Reflect`), attach it
/// to the registry with
/// [`register_type_data`](crate::TypeRegistry::register_type_data), then call
/// [`get`]/[`get_mut`] to obtain the trait object.
///
/// The generated code contains no `unsafe`: recovery goes through
/// [`Any`](core::any::Any) downcasting.
///
/// [`get`]: #method.get
/// [`get_mut`]: #method.get_mut
#[macro_export]
macro_rules! reflect_trait {
    ($(#[$meta:meta])* $vis:vis $data:ident for $trait_name:path) => {
        $(#[$meta])*
        #[derive(::core::clone::Clone, ::core::marker::Copy)]
        $vis struct $data {
            get_fn: for<'r> fn(
                &'r dyn $crate::Reflect,
            ) -> ::core::option::Option<&'r (dyn $trait_name + 'static)>,
            get_mut_fn: for<'r> fn(
                &'r mut dyn $crate::Reflect,
            ) -> ::core::option::Option<&'r mut (dyn $trait_name + 'static)>,
        }

        impl $data {
            /// Build the accessor for the concrete type `T`.
            #[must_use]
            pub fn from_type<T: $trait_name + $crate::Reflect>() -> Self {
                Self {
                    get_fn: |reflect| {
                        ::core::option::Option::map(
                            $crate::Reflect::as_any(reflect).downcast_ref::<T>(),
                            |value| value as &(dyn $trait_name + 'static),
                        )
                    },
                    get_mut_fn: |reflect| {
                        ::core::option::Option::map(
                            $crate::Reflect::as_any_mut(reflect).downcast_mut::<T>(),
                            |value| value as &mut (dyn $trait_name + 'static),
                        )
                    },
                }
            }

            /// Recover a shared `&dyn` trait reference from a reflected value,
            /// or `None` when the value is not of the type this accessor was
            /// built for.
            #[must_use]
            pub fn get<'r>(
                &self,
                reflect: &'r dyn $crate::Reflect,
            ) -> ::core::option::Option<&'r (dyn $trait_name + 'static)> {
                (self.get_fn)(reflect)
            }

            /// Recover a mutable `&mut dyn` trait reference from a reflected
            /// value, or `None` on a type mismatch.
            #[must_use]
            pub fn get_mut<'r>(
                &self,
                reflect: &'r mut dyn $crate::Reflect,
            ) -> ::core::option::Option<&'r mut (dyn $trait_name + 'static)> {
                (self.get_mut_fn)(reflect)
            }
        }

        impl $crate::TypeData for $data {
            fn clone_type_data(&self) -> $crate::__macro_exports::Box<dyn $crate::TypeData> {
                $crate::__macro_exports::Box::new(*self)
            }

            fn as_any(&self) -> &dyn ::core::any::Any {
                self
            }
        }
    };
}
