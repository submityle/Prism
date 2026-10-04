//! Reflection-driven serialization (design §10, §22 — milestone M3).
//!
//! This module turns any `&dyn Reflect` into bytes or text and back without a
//! per-type `serde` impl: a single traversal driver walks the value's
//! [`ReflectRef`](crate::ReflectRef) view and feeds structural callbacks to a
//! format back-end, while deserialization is guided by the target
//! [`TypeInfo`](crate::TypeInfo) resolved through a
//! [`TypeRegistry`](crate::TypeRegistry) and rebuilds a
//! [`Dynamic*`](crate::dynamic) tree that round-trips to the original concrete
//! value via [`FromReflect`](crate::FromReflect).
//!
//! Two back-ends ship here, sharing the traversal driver and schema resolver:
//! - [`to_binary`]/[`from_binary`] — a compact, self-describing binary format
//!   whose header pins the root type with a [`StableTypeId`].
//! - [`to_ron`]/[`from_ron`] — a self-contained, anonymous RON text format.
//!
//! [`StableTypeId`] derives a deterministic, cross-build type id from the type
//! *path* (not [`core::any::TypeId`], which is not stable across compilations),
//! so streams stay identifiable across runs, platforms, and builds.
//!
//! # Supported subset
//! Every reflected kind is covered — struct, tuple struct, enum (unit, tuple,
//! and struct variants), list, array, map, set, and the built-in leaf
//! `Value`s. An opaque `Value`-kind type outside the built-in leaf set is the
//! only serialization failure ([`SerializeError::UnsupportedLeaf`]). Nested
//! composite types must be registered in the [`TypeRegistry`](crate::TypeRegistry)
//! before deserialization so their shape can be resolved.

mod binary;
mod de;
mod encode;
mod error;
mod limits;
mod pod;
mod primitive;
mod ron;
mod stable_id;

pub use binary::{from_binary, from_binary_with_limits, to_binary};
pub use limits::DeserializeLimits;
pub use error::{DeserializeError, SerializeError};
pub use ron::{from_ron, to_ron};
pub use stable_id::StableTypeId;
