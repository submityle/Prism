//! `prism_reflect` — Prism's reflection kernel (the "type-truth layer").
//!
//! Reflection bridges Rust's static type world to the dynamic data-driven
//! world (editor, scripting, serialization, networking). A type opts in once
//! with `#[derive(Reflect)]` and becomes uniformly traversable, down-castable,
//! and (in later milestones) serializable and script-callable.
//!
//! # Scope through M2 (this crate)
//! - [`Reflect`] trait with [`ReflectRef`]/[`ReflectMut`] down-casting views and
//!   owned `Box<dyn Any>` recovery via [`Reflect::into_any`].
//! - Static, `OnceLock`-cached [`TypeInfo`] with every kind: `Struct`,
//!   `TupleStruct`, `Enum`, `List`, `Array`, `Map`, `Set`, and `Value`.
//! - Field/variant/element traversal through the [`Struct`], [`TupleStruct`],
//!   [`Enum`], [`List`], [`Array`], [`Map`], and [`Set`] subtraits.
//! - [`TypeRegistry`] + [`TypeData`] with [`GetTypeRegistration`] and the
//!   [`ReflectDefault`] default-constructor payload.
//! - `#[derive(Reflect)]` for named-field structs, tuple structs, and enums.
//! - [`impl_reflect_value!`] leaf impls for std scalar/string types, plus
//!   `Vec`/array/map/set/`Option`/`Result` and (behind the `math` feature)
//!   `prism_math` value types.
//! - **M2 dynamic layer:** runtime-constructed [`DynamicStruct`],
//!   [`DynamicTupleStruct`], [`DynamicEnum`], [`DynamicList`], [`DynamicArray`],
//!   and [`DynamicMap`] values; [`FromReflect`] to rebuild a concrete `T` from
//!   any `&dyn Reflect`; [`Reflect::apply`]/[`Reflect::reflect_clone`] recursive
//!   state transfer; and [`ParsedPath`] access-path navigation via
//!   [`reflect_path`]/[`reflect_path_mut`]. `#[derive(Reflect)]` generates
//!   `FromReflect` and `reflect_clone` alongside the kind traits.
//!
//! Later milestones (design §22): reflection-driven serialization +
//! `StableTypeId` (M3), attribute metadata + schema migration (M4), function
//! reflection + `reflect_trait` (M5), and ECS/scene/editor/script/network
//! integration (M6).
//!
//! This crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.

#![forbid(unsafe_code)]

// Allows the derive macro's `::prism_reflect::` paths to resolve inside this
// crate's own tests and doctests.
extern crate self as prism_reflect;

mod apply;
mod cache;
mod dynamic;
mod from_reflect;
mod impls;
mod kinds;
#[cfg(feature = "math")]
mod math_impls;
mod path;
mod reflect;
mod registry;
mod type_data;
mod type_info;

pub use prism_reflect_macros::Reflect;

pub use kinds::{Array, ArrayIter, Enum, List, ListIter, Map, MapIter, Set, SetIter, VariantType};
pub use apply::ApplyError;
pub use dynamic::{
    DynamicArray, DynamicEnum, DynamicList, DynamicMap, DynamicStruct, DynamicTupleStruct,
    DynamicVariant,
};
pub use from_reflect::FromReflect;
pub use path::{Access, ParsePathError, ParsedPath, reflect_path, reflect_path_mut};
pub use reflect::{Reflect, ReflectMut, ReflectRef, Struct, TupleStruct, Typed};
pub use registry::{GetTypeRegistration, TypeRegistration, TypeRegistry};
pub use type_data::{ReflectDefault, TypeData};
pub use type_info::{
    ArrayInfo, EnumInfo, ListInfo, MapInfo, NamedField, SetInfo, StructInfo, TupleStructInfo,
    TypeInfo, UnnamedField, ValueInfo, VariantInfo, VariantKind,
};

/// Convenient re-exports for downstream crates.
pub mod prelude {
    pub use crate::{
        Access, Array, ArrayInfo, ApplyError, DynamicArray, DynamicEnum, DynamicList, DynamicMap,
        DynamicStruct, DynamicTupleStruct, DynamicVariant, Enum, EnumInfo, FromReflect,
        GetTypeRegistration, List, ListInfo, Map, MapInfo, NamedField, ParsePathError, ParsedPath,
        Reflect, ReflectDefault, ReflectMut, ReflectRef, Set, SetInfo, Struct, StructInfo,
        TupleStruct, TupleStructInfo, TypeData, TypeInfo, TypeRegistration, TypeRegistry, Typed,
        UnnamedField, ValueInfo, VariantInfo, VariantKind, VariantType, reflect_path,
        reflect_path_mut,
    };
}

#[cfg(test)]
mod tests;
