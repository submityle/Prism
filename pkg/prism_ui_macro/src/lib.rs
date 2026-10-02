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
//! # Macros
//!
//! * [`loom!`] — build a [`prism_ui::Element`] tree.
//! * [`loom_program!`] — expand to a stable, textual description of a tree's
//!   lowering program plus its static/dynamic classification. This backs the
//!   dual-mode (§9.1) snapshot tooling and the static-hoisting (§9.2)
//!   diagnostics; see the `dualmode` and `hoist` modules.
//! * [`bind!`] — register a `prism_ui_ecs` field binding from a one-line
//!   `signal <-> $entity.Component.field` declaration (§9.9); see the `bind`
//!   module.
//!
//! # Internal module map
//!
//! * `ast` — the parsed DSL.
//! * `lower` — the single shared AST lowerer used by both compilation modes.
//! * `dualmode` — the mode-agnostic semantic program and reproducible
//!   attribute ordering (§9.1).
//! * `hoist` — static-subtree classification for template hoisting (§9.2).
//! * `bind` — `$` auto field-binding codegen (§9.9).
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
mod bind;
mod dualmode;
mod hoist;
mod lower;

use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

use crate::ast::LoomInput;
use crate::bind::BindInput;
use crate::dualmode::Mode;
use crate::hoist::StaticClass;

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
/// # Reactive-read sigil `$`
///
/// A `text(..)` or `custom(..)` content expression may be prefixed with `$` to
/// mark a **reactive read**: `text($label)` lowers to `text(label.get())`,
/// performing a tracked [`Signal`](prism_ui::reactive::Signal) read. When the
/// `loom!` tree is built inside a reactive view (an effect driven by
/// [`ReactiveView`](prism_ui::ReactiveView)), that read records a dependency so
/// the view re-runs — and the subtree reconciles — whenever the signal changes.
/// Without `$` the expression is spliced verbatim. `$` is pure sugar for an
/// explicit `.get()`; it adds no hidden state.
///
/// [`prism_ui::Element`]: https://docs.rs/prism_ui
#[proc_macro]
pub fn loom(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as LoomInput);
    lower::lower_node(&parsed.node, "").into()
}

/// Expands to a `&'static str` describing a Loom tree's lowering program.
///
/// The first line is `static` or `dynamic` — the hoisting classification from
/// [`hoist::classify_subtree`](crate) (§9.2) — followed by the stable semantic
/// program signature from the `dualmode` module (§9.1). The program is computed
/// in **both** [`Mode::Interpret`] and [`Mode::Freeze`] and the macro emits a
/// compile error if they disagree, enforcing the "interpret result == freeze
/// result" invariant at the use site; in practice they always agree because
/// both share the lowerer.
///
/// This is build/snapshot tooling: it lets reproducible-build and
/// static-hoisting tests assert on a tree's canonical shape without inspecting
/// generated tokens.
///
/// The grammar is identical to [`loom!`].
#[proc_macro]
pub fn loom_program(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as LoomInput);
    let node = &parsed.node;

    let interpret = dualmode::lower_in_mode(node, "", Mode::Interpret);
    let freeze = dualmode::lower_in_mode(node, "", Mode::Freeze);
    if interpret != freeze {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "internal error: interpret and freeze lowering programs diverged",
        )
        .to_compile_error()
        .into();
    }

    let tag = match hoist::classify_subtree(node) {
        StaticClass::Static => "static",
        StaticClass::Dynamic => "dynamic",
    };
    let signature = dualmode::program_signature(&freeze);
    let text = format!("{tag}\n{signature}");
    quote! { #text }.into()
}

/// Registers a `prism_ui_ecs` field binding from a one-line declaration (§9.9).
///
/// # Syntax
///
/// ```ignore
/// bind!(BRIDGE, SIGNAL <-  $ENTITY.COMPONENT.FIELD : TYPE); // one-way read
/// bind!(BRIDGE, SIGNAL <-> $ENTITY.COMPONENT.FIELD : TYPE); // two-way
/// ```
///
/// * `BRIDGE` is an expression evaluating to a `prism_ui_ecs::EcsBridge`.
/// * `SIGNAL` is the reactive signal expression (parsed verbatim up to the
///   arrow, so keep it free of top-level `<`; bind complex signals to a local
///   first).
/// * `$ENTITY.COMPONENT.FIELD` names the entity binding, the component type and
///   the projected field (a named field or a tuple index such as `0`).
/// * `TYPE` is the projected field type.
///
/// # Expansion
///
/// The one-way form expands to `BRIDGE.bind::<COMPONENT, TYPE>(ENTITY, SIGNAL,
/// reader)` and the two-way form to `BRIDGE.bind_two_way::<..>(ENTITY, SIGNAL,
/// reader, writer)`, where `reader` clones the field (read path reuses `Ref`
/// tick change detection in `FieldBinding::pull`) and `writer` assigns it (write
/// path uses `Mut` plus the equality guard in `FieldBinding::push`). Frame
/// ordering — pull before push — comes from `prism_ui_ecs::add_loom_sync_systems`,
/// which chains `LoomSyncSet::Pull` before `LoomSyncSet::Push`.
///
/// An illegal field path is a **compile-time** error whose span points at the
/// offending token; the macro never panics at run time.
#[proc_macro]
pub fn bind(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as BindInput);
    bind::expand(parsed).into()
}
