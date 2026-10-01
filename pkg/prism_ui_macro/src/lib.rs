//! Procedural macros for Prism's **Loom** UI runtime.
//!
//! This crate implements the [`loom!`] declarative DSL, a small, readable
//! notation for describing a [`prism_ui::Element`] tree. The macro parses its
//! input with [`syn`] into a typed AST (see the `ast` module) and lowers that
//! AST into a chain of builder calls on `::prism_ui::Element` (see the `lower`
//! module).
//!
//! The emitted code uses fully-qualified paths such as `::prism_ui::Element`
//! and `::prism_ui::style::StyleValue`, so the macro works regardless of what
//! the caller has imported.
//!
//! [`prism_ui::Element`]: https://docs.rs/prism_ui
//!
//! # Example
//!
//! ```ignore
//! use prism_ui_macro::loom;
//!
//! let view = loom! {
//!     box {
//!         class: "card", "elevated";
//!         key: 42;
//!         style: {
//!             width: px(300.0);
//!             background_color: token("color.bg");
//!             flex_direction: column;
//!             opacity: 0.5;
//!         };
//!         text("Hello");
//!         box { class: "row"; }
//!     }
//! };
//! ```

#![forbid(unsafe_code)]

mod ast;
mod lower;

use proc_macro::TokenStream;
use syn::parse_macro_input;

use crate::ast::LoomInput;

/// Builds a [`prism_ui::Element`] tree from the Loom DSL.
///
/// See the [crate-level documentation](crate) for the full grammar. In short,
/// the macro accepts a single root node (`box`, `text(..)` or `custom(..)`)
/// whose brace block may contain attributes (`class`, `key`, `style`) and
/// nested child nodes, plus a `for_each(..)` splice for dynamic child lists.
///
/// Each node lowers to `::prism_ui::Element::box_()` / `::text(..)` /
/// `::custom(..)` followed by chained `.class(..)`, `.key_int(..)` /
/// `.key_str(..)`, `.style(..)`, `.child(..)` and `.children(..)` calls.
///
/// [`prism_ui::Element`]: https://docs.rs/prism_ui
#[proc_macro]
pub fn loom(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as LoomInput);
    lower::lower_node(&parsed.node).into()
}
