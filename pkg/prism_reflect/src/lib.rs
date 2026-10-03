//! `prism_reflect` — Prism's reflection kernel (the "type-truth layer").
//!
//! Reflection bridges Rust's static type world to the dynamic data-driven
//! world (editor, scripting, serialization, networking). A type opts in once
//! with `#[derive(Reflect)]` and becomes uniformly traversable, down-castable,
//! and (in later milestones) serializable and script-callable.
//!
//! # Scope through M3 (this crate)
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
//! - **M3 serialization:** reflection-driven [`to_binary`]/[`from_binary`]
//!   (a compact self-describing binary format) and [`to_ron`]/[`from_ron`]
//!   (self-contained RON text), plus [`StableTypeId`], a deterministic
//!   cross-build type id derived from the type path.
//!
//! - **M4 metadata + schema:** attribute/field [`metadata`](schema::metadata)
//!   ([`TypeMetadata`](schema::TypeMetadata)) queryable through the registry,
//!   schema [`version`](schema::version)ing ([`SchemaRegistry`](schema::SchemaRegistry)),
//!   composable [`migration`](schema::migration) chains
//!   ([`Migration`](schema::Migration)), [`validate`](schema::validate)ion, and
//!   versioned serialization
//!   ([`to_versioned_binary`](schema::to_versioned_binary)/
//!   [`from_versioned_binary`](schema::from_versioned_binary)) layered on the
//!   M3 serializer so an old payload migrates step-by-step before deserialization.
//!
//! - **M5 function reflection + trait objects:** register free functions and
//!   methods into a [`FunctionRegistry`] and [`call`](FunctionRegistry::call)
//!   them **by name** with a type-erased [`ArgList`], returning a reflected
//!   result; strict arity/argument-type validation surfaces a typed
//!   [`FunctionError`] instead of undefined behaviour (design §23's
//!   "函数反射安全"). The [`reflect_trait!`] macro generates a
//!   [`TypeData`] accessor that recovers a `&dyn Trait` from a `&dyn Reflect`
//!   for data-driven dispatch. [`StructTypeBuilder`]/[`EnumTypeBuilder`]
//!   construct and register [`TypeInfo`] for types defined at runtime, and
//!   [`diff`]/[`merge`] compute and apply a minimal [`Patch`] between two
//!   reflected values (design §24.4).
//!
//! - **M6 integration:** the [`integration`] layer turns the reflection core
//!   into the shared contract behind entity-component-system (ECS) snapshots,
//!   scenes, editors, scripting, and networking:
//!   [`DynamicScene`](integration::DynamicScene) for ordered, type-tagged
//!   component snapshots with binary and framed-text serialization;
//!   [`inspect`](integration::inspect) for a backend-agnostic editor property
//!   tree with metadata-driven hints; [`replicated_diff`](integration::replicated_diff)
//!   for field-level network deltas honouring `replicate`/`no_replicate`
//!   intent; and [`ScriptBridge`](integration::ScriptBridge) for safe path
//!   get/set plus call-by-name. The `compat-bevy` feature adds a
//!   [`bevy_reflect`-compatible prelude](compat_bevy) (no `bevy_*` dependency).
//!
//! This crate contains no Unreal Engine source or derived code and depends on
//! no `bevy_*` crate.

#![forbid(unsafe_code)]

// Allows the derive macro's `::prism_reflect::` paths to resolve inside this
// crate's own tests and doctests.
extern crate self as prism_reflect;

extern crate alloc;

mod apply;
mod cache;
mod diff;
mod dynamic;
mod from_reflect;
mod func;
mod impls;
mod kinds;
#[cfg(feature = "math")]
mod math_impls;
mod path;
mod reflect;
mod reflect_trait;
mod registry;
mod runtime_type;
pub mod schema;
#[cfg(feature = "compat-bevy")]
pub mod compat_bevy;
pub mod integration;
mod ser;
mod type_data;
mod type_info;

pub use prism_reflect_macros::Reflect;

#[doc(hidden)]
pub mod __macro_exports {
    //! Internal re-exports used by `reflect_trait!`; not a stable public API.
    pub use alloc::boxed::Box;
}

pub use kinds::{Array, ArrayIter, Enum, List, ListIter, Map, MapIter, Set, SetIter, VariantType};
pub use apply::ApplyError;
pub use dynamic::{
    DynamicArray, DynamicEnum, DynamicList, DynamicMap, DynamicSet, DynamicStruct,
    DynamicTupleStruct, DynamicVariant,
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
pub use ser::{
    DeserializeError, SerializeError, StableTypeId, from_binary, from_ron, to_binary, to_ron,
};
pub use diff::{DiffError, Patch, diff, merge};
pub use func::{
    ArgList, DynamicFunction, FunctionError, FunctionInfo, FunctionRegistry, IntoFunction,
};
pub use runtime_type::{EnumTypeBuilder, StructTypeBuilder};
// `reflect_trait!` is `#[macro_export]`, so it is already available at the
// crate root; nothing to re-export here.
pub use schema::{
    AttributeValue, FieldMetadata, MigrateError, Migration, SchemaRegistry, SchemaVersion,
    TypeMetadata, TypeSchema, ValidationError,
};
pub use integration::{
    DynamicScene, InspectorHints, InspectorKind, InspectorNode, ReplicationError, ReplicationPlan,
    ReplicationPolicy, SceneEntry, SceneError, ScriptBridge, ScriptError, apply_replicated, inspect,
    replicated_diff,
};

/// Convenient re-exports for downstream crates.
pub mod prelude {
    pub use crate::{
        Access, ApplyError, ArgList, Array, ArrayInfo, DeserializeError, DiffError, DynamicArray,
        DynamicEnum, DynamicFunction, DynamicList, DynamicMap, DynamicSet, DynamicStruct,
        DynamicTupleStruct, DynamicVariant, Enum, EnumInfo, EnumTypeBuilder, FromReflect,
        FunctionError, FunctionInfo, FunctionRegistry, GetTypeRegistration, IntoFunction, List,
        ListInfo, Map, MapInfo, NamedField, ParsePathError, ParsedPath, Patch, Reflect,
        ReflectDefault, ReflectMut, ReflectRef, SerializeError, Set, SetInfo, StableTypeId, Struct,
        StructInfo, StructTypeBuilder, TupleStruct, TupleStructInfo, TypeData, TypeInfo,
        TypeRegistration, TypeRegistry, Typed, UnnamedField, ValueInfo, VariantInfo, VariantKind,
        VariantType, diff, from_binary, from_ron, merge, reflect_path, reflect_path_mut, to_binary,
        to_ron,
    };
    pub use crate::reflect_trait;
    pub use crate::integration::{
        DynamicScene, InspectorNode, ReplicationPlan, ReplicationPolicy, ScriptBridge,
        apply_replicated, inspect, replicated_diff,
    };
    pub use crate::schema::{
        AttributeValue, FieldMetadata, MigrateError, Migration, SchemaRegistry, SchemaVersion,
        TypeMetadata, TypeSchema, ValidationError, from_versioned_binary, from_versioned_ron,
        to_versioned_binary, to_versioned_ron, validate, validate_version,
    };
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_m6;
