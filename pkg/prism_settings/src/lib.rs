//! Deterministic settings schema and resolution core.
//!
//! The crate is self-contained and supports `no_std + alloc`. It deliberately
//! contains no file format, application facade, UI, or operating-system code.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

mod model;
mod registry;
mod resolver;

pub use model::*;
pub use registry::*;
pub use resolver::*;

#[cfg(test)]
mod tests;
