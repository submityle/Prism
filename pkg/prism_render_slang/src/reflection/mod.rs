//! Reflection: turning Slang reflection JSON into a stable ABI model.
//!
//! [`model`] defines the backend-neutral layout types; [`parse`] builds them
//! from `slangc -reflection-json` output. The resulting [`AbiModel`] is the
//! single source of truth that [`crate::codegen`] turns into `#[repr(C)]`
//! Rust bindings.

pub mod model;
pub mod parse;

pub use model::{AbiModel, Field, FieldType, Scalar, StructLayout};
pub use parse::{parse_reflection, parse_value};
