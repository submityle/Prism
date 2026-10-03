//! Runtime-constructed reflected values (`Dynamic*`).
//!
//! The `Dynamic*` types let code build and mutate reflected data without a
//! concrete Rust type behind it. Each one implements the matching kind trait
//! ([`Struct`](crate::Struct), [`List`](crate::List), ...) plus
//! [`Reflect`](crate::Reflect), holding its fields/elements as boxed
//! `dyn Reflect` values. They are the construction half of M2's dynamic story:
//! a `Dynamic*` can be [`apply`](crate::Reflect::apply)-ed onto a concrete
//! value or turned back into one with [`FromReflect`](crate::FromReflect).
//!
//! Because a dynamic value has no single static Rust type, it reports an
//! optional *represented* type name (the concrete type it stands in for, when
//! known) and a placeholder [`TypeInfo`](crate::TypeInfo) whose kind matches
//! the value but whose field metadata is empty. Use the represented type name
//! for identity and the kind traits for traversal.

mod arrays;
mod enums;
mod equality;
mod lists;
mod maps;
mod structs;
mod tuple_structs;

pub use arrays::DynamicArray;
pub use enums::{DynamicEnum, DynamicVariant};
pub use lists::DynamicList;
pub use maps::DynamicMap;
pub use structs::DynamicStruct;
pub use tuple_structs::DynamicTupleStruct;

pub(crate) use equality::reflect_values_equal;
