//! Script and editor property bridge (design §24.6) — ✅ delivered.
//!
//! Reflection is the single façade through which a scripting runtime or an
//! editor inspector touches engine state without any hand-written per-type
//! glue. This module turns the reflection core
//! ([`Reflect`](crate::Reflect)/[`ParsedPath`](crate::ParsedPath) navigation),
//! the registry's [`TypeMetadata`](crate::TypeMetadata), and the §24.7 clamp
//! into one safe property surface:
//!
//! 1. [`PropertyBridge`] — a type-safe get/set layer over a live
//!    `&dyn Reflect` root. It reads and writes properties *by path*
//!    (`"transform.translation[0]"`), downcasts a property to a concrete type
//!    ([`PropertyBridge::get_as`]), and exchanges leaf values as a dynamic
//!    [`AttributeValue`](crate::schema::AttributeValue) scalar for runtimes
//!    that do not know Rust types at compile time
//!    ([`PropertyBridge::read_scalar`]/[`PropertyBridge::set_scalar`]). Writes
//!    honour the field's metadata: a read-only field is rejected, and a
//!    range-constrained field is clamped into its bounds after assignment,
//!    reusing [`schema::clamp`](crate::schema::clamp).
//! 2. [`PropertyDescriptor`] / [`properties`] — enumerate a struct's top-level
//!    editable properties with their display metadata (kind, read-only flag,
//!    numeric range, docs, category, default), so an editor can bind widgets to
//!    a property panel without a hand-written per-type UI. Hidden fields are
//!    omitted.
//!
//! This is the data half of the bridge; the deeper, recursively nested
//! inspector tree lives in [`inspect`](crate::integration::inspect), and
//! call-by-name dispatch lives in [`ScriptBridge`](crate::integration::ScriptBridge).
//! The property bridge stays backend-agnostic: it draws no user interface and
//! embeds no scripting runtime.
//!
//! # Round-trip parity
//! For any editable leaf property, reading its scalar and writing the same
//! scalar back is a no-op, and reading after a `set_scalar` returns the value
//! just written (clamped into range when the field is range-constrained). This
//! get/set round-trip identity is oracle-checked in `tests_property_bridge`.

mod bridge;
mod descriptor;

pub use bridge::{PropertyBridge, PropertyError};
pub use descriptor::{properties, PropertyDescriptor, PropertyKind};
