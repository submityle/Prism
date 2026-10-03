//! `prism_reflect` — Prism's reflection kernel (the "type-truth layer").
//!
//! Reflection bridges Rust's static type world to the dynamic data-driven
//! world (editor, scripting, serialization, networking). A type opts in once
//! with `#[derive(Reflect)]` and becomes uniformly traversable, down-castable,
//! and (in later milestones) serializable and script-callable.
//!
//! # M0 scope (this crate)
//! - [`Reflect`] trait with [`ReflectRef`]/[`ReflectMut`] down-casting views.
//! - Static, `OnceLock`-cached [`TypeInfo`] with the `Struct`, `TupleStruct`,
//!   and `Value` kinds.
//! - [`Struct`]/[`TupleStruct`] field traversal (by name and index).
//! - `#[derive(Reflect)]` for named-field and tuple structs.
//! - [`impl_reflect_value!`] leaf impls for std scalar/string types.
//!
//! Later milestones (design §22): enum/list/array/map/set kinds + `TypeRegistry`
//! (M1), `Dynamic*`/`FromReflect`/`apply`/path access (M2), reflection-driven
//! serialization + `StableTypeId` (M3), attribute metadata + schema migration
//! (M4), function reflection + `reflect_trait` (M5), and ECS/scene/editor/script/
//! network integration (M6).
//!
//! This crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.

#![forbid(unsafe_code)]

// Allows the derive macro's `::prism_reflect::` paths to resolve inside this
// crate's own tests and doctests.
extern crate self as prism_reflect;

mod impls;
mod reflect;
mod type_info;

pub use prism_reflect_macros::Reflect;

pub use reflect::{Reflect, ReflectMut, ReflectRef, Struct, TupleStruct, Typed};
pub use type_info::{
    NamedField, StructInfo, TupleStructInfo, TypeInfo, UnnamedField, ValueInfo,
};

/// Convenient re-exports for downstream crates.
pub mod prelude {
    pub use crate::{
        NamedField, Reflect, ReflectMut, ReflectRef, Struct, StructInfo, TupleStruct,
        TupleStructInfo, TypeInfo, Typed, UnnamedField, ValueInfo,
    };
}

#[cfg(test)]
mod tests;
