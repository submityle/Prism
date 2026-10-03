//! `bevy_reflect`-compatible aliases and prelude.
//!
//! Enabled by the `compat-bevy` feature, this module re-exports Prism's
//! reflection types under the names a `bevy_reflect` user expects, easing
//! migration without pulling in any `bevy_*` crate (design §22
//! "`bevy_reflect` 兼容 prelude"). The underlying types are Prism's own; only the spelling is
//! made familiar.

pub use crate::{
    Array, DynamicEnum, DynamicList, DynamicMap, DynamicStruct, DynamicTupleStruct, Enum,
    FromReflect, GetTypeRegistration, List, Map, Reflect, ReflectMut, ReflectRef, Set, Struct,
    TupleStruct, TypeInfo, TypeRegistration, TypeRegistry, Typed,
};

/// The `TypeRegistry` under its `bevy_reflect` arc-wrapper-free spelling.
pub type AppTypeRegistry = TypeRegistry;

/// Reflection-driven partial reflect view, spelled as in `bevy_reflect`.
pub use crate::ReflectRef as PartialReflectRef;

/// A `bevy_reflect`-style prelude.
pub mod prelude {
    pub use super::{
        AppTypeRegistry, Array, DynamicEnum, DynamicList, DynamicMap, DynamicStruct,
        DynamicTupleStruct, Enum, FromReflect, GetTypeRegistration, List, Map, Reflect, ReflectMut,
        ReflectRef, Set, Struct, TupleStruct, TypeInfo, TypeRegistration, TypeRegistry, Typed,
    };
}
