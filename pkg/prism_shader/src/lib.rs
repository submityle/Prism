//! `prism_shader` — the backend-agnostic shader composition kernel.
//!
//! This crate sits one level above [`prism_render_driver`]'s raw
//! `ShaderSource` (`WGSL` / `WESL` / `SPIR-V`): it turns *authored* modules plus
//! a *permutation* of shader-defs into one final, deterministic source string
//! ready to hand to the driver. It borrows the proven shape of mature shader
//! preprocessor layers (a `naga_oil`-style module + preprocessor + composer)
//! without copying any of their source.
//!
//! The M0 surface is four small, pure, deterministic pieces:
//!
//! - [`def`]: typed shader-defs ([`ShaderDefValue`], [`ShaderDefs`]) in canonical
//!   name-sorted order.
//! - [`permutation`]: an order-independent, versioned [`PermutationId`] suitable
//!   as a pipeline-state-object cache key.
//! - [`expr`] + [`preprocess`]: a total `C`-style preprocessor
//!   (`#ifdef` / `#ifndef` / `#if` / `#elif` / `#else` / `#endif`) with a full
//!   recursive-descent integer expression evaluator over the defs.
//! - [`module`] + [`compose`]: import resolution (`#import` / `#include`) with
//!   deterministic topological ordering, dedup, and cycle detection, exposed via
//!   [`ShaderComposer`].
//!
//! Everything is `no_std + alloc` and free of `unsafe`; `std` only gates
//! host-side tooling that may be added later.
//!
//! [`prism_render_driver`]: https://docs.rs/prism_render_driver
//! [`ShaderDefValue`]: crate::def::ShaderDefValue
//! [`ShaderDefs`]: crate::def::ShaderDefs
//! [`PermutationId`]: crate::permutation::PermutationId
//! [`ShaderComposer`]: crate::compose::ShaderComposer

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

pub mod compose;
pub mod def;
pub mod expr;
pub mod module;
pub mod permutation;
pub mod preprocess;

pub use compose::{ComposeError, ShaderComposer};
pub use def::{ShaderDefValue, ShaderDefs};
pub use expr::{ExprError, evaluate};
pub use module::ShaderModule;
pub use permutation::PermutationId;
pub use preprocess::{PreprocessError, preprocess};

/// The crate's most common exports, for a single glob import.
pub mod prelude {
    pub use crate::compose::{ComposeError, ShaderComposer};
    pub use crate::def::{ShaderDefValue, ShaderDefs};
    pub use crate::expr::{ExprError, evaluate};
    pub use crate::module::ShaderModule;
    pub use crate::permutation::PermutationId;
    pub use crate::preprocess::{PreprocessError, preprocess};
}

#[cfg(test)]
mod tests;
