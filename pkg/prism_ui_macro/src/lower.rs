//! Lowering from the parsed [`crate::ast`] to builder-call token streams.
//!
//! Each [`Node`] becomes a `::prism_ui::Element` constructor followed by a
//! chain of builder methods. All emitted paths are fully qualified so the
//! generated code does not depend on the caller's imports.
//!
//! This module is the **single shared lowerer** referenced by the dual-mode
//! design (§9.1): both the development-time interpreter and the release-time
//! freezer feed through [`lower_node`], so "interpret result == freeze result"
//! holds by construction. Attribute emission follows the canonical order
//! computed by [`crate::dualmode`], which makes the frozen token output
//! reproducible regardless of the order attributes were written in source.
//!
//! # Stable node ids
//!
//! Every statically-known node is tagged with a compile-time
//! `::prism_ui::StableId` describing its position in the macro call tree. The
//! root path is `""`, a node's first child is `"0"`, its grandchild `"0/1"`,
//! and so on. These ids are deterministic (the same source lowers to the same
//! ids every time) and are used by hot reload to align nodes across edits even
//! when sibling insertion or deletion shifts positions.
//!
//! Dynamic `for_each(..)` splices are *not* assigned a stable id: their length
//! is unknown at compile time, so their children align at run time by their
//! explicit reconciliation [`Key`](prism_ui::Key) instead of by a static path.

use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::ast::{Attr, CallKind, Child, ContentExpr, Node, NodeKind, StyleVal};
use crate::dualmode;

/// Lowers a [`Node`] into the builder-call chain that constructs it.
///
/// `path` is the node's `/`-separated position path within the macro call tree
/// (the root is `""`). The node is tagged with a `::prism_ui::StableId` built
/// from this path, and each static child is lowered with the path extended by
/// its sibling index.
pub(crate) fn lower_node(node: &Node, path: &str) -> TokenStream {
    let mut expr = match &node.kind {
        NodeKind::Box => quote! { ::prism_ui::Element::box_() },
        NodeKind::Text(content) => {
            let value = lower_content(content);
            quote! { ::prism_ui::Element::text(#value) }
        }
        NodeKind::Custom(name) => {
            let value = lower_content(name);
            quote! { ::prism_ui::Element::custom(#value) }
        }
    };

    for attr in dualmode::canonical_attr_order(&node.attrs) {
        expr = lower_attr(expr, attr);
    }

    expr = quote! { #expr.with_stable_id(::prism_ui::StableId::new(#path)) };

    for (index, child) in node.children.iter().enumerate() {
        expr = match child {
            Child::Node(child_node) => {
                let child_path = child_path(path, index);
                let child_expr = lower_node(child_node, &child_path);
                quote! { #expr.child(#child_expr) }
            }
            Child::ForEach(iter) => quote! { #expr.children(#iter) },
        };
    }

    expr
}

/// Extends `parent` with `index` to form a child's position path.
///
/// The root path is the empty string, so its first child is `"0"` rather than
/// `"/0"`; deeper children are `"0/1"`, `"0/1/2"` and so on.
fn child_path(parent: &str, index: usize) -> String {
    if parent.is_empty() {
        index.to_string()
    } else {
        format!("{parent}/{index}")
    }
}

/// Lowers a `text(..)` / `custom(..)` [`ContentExpr`] to its value token stream.
///
/// A plain expression is spliced verbatim; a `$`-prefixed one becomes
/// `(expr).get()`, performing a tracked [`Signal`](prism_ui::reactive::Signal)
/// read so the enclosing reactive view re-runs when the signal changes.
fn lower_content(content: &ContentExpr) -> TokenStream {
    let expr = &content.expr;
    if content.reactive {
        quote! { (#expr).get() }
    } else {
        quote! { #expr }
    }
}

/// Appends the builder calls for a single attribute to `expr`.
fn lower_attr(expr: TokenStream, attr: &Attr) -> TokenStream {
    match attr {
        Attr::Class(names) => {
            let mut acc = expr;
            for name in names {
                acc = quote! { #acc.class(#name) };
            }
            acc
        }
        Attr::KeyInt(int) => quote! { #expr.key_int(#int) },
        Attr::KeyStr(string) => quote! { #expr.key_str(#string) },
        Attr::Style(entries) => {
            let mut acc = expr;
            for entry in entries {
                let prop = snake_to_pascal(&entry.prop);
                let value = lower_style_val(&entry.value);
                acc = quote! {
                    #acc.style(::prism_ui::style::StyleProp::#prop, #value)
                };
            }
            acc
        }
    }
}

/// Lowers a single style value to a `::prism_ui::style::StyleValue` expression.
fn lower_style_val(value: &StyleVal) -> TokenStream {
    match value {
        StyleVal::Number(lit) => {
            quote! { ::prism_ui::style::StyleValue::Number((#lit) as f32) }
        }
        StyleVal::Keyword(ident) => {
            let keyword = snake_to_pascal(ident);
            quote! {
                ::prism_ui::style::StyleValue::Keyword(::prism_ui::style::Keyword::#keyword)
            }
        }
        StyleVal::Call { kind, args, .. } => match kind {
            CallKind::Px => quote! { ::prism_ui::style::StyleValue::px(#args) },
            CallKind::Token => quote! { ::prism_ui::style::StyleValue::token(#args) },
            CallKind::Rgba8 => quote! { ::prism_ui::style::StyleValue::rgba8(#args) },
        },
    }
}

/// Converts a `snake_case` identifier to a `PascalCase` identifier, preserving
/// the source span so diagnostics point at the original token.
fn snake_to_pascal(ident: &Ident) -> Ident {
    let input = ident.to_string();
    let mut pascal = String::with_capacity(input.len());
    for segment in input.split('_') {
        let mut chars = segment.chars();
        if let Some(first) = chars.next() {
            pascal.extend(first.to_uppercase());
            pascal.push_str(chars.as_str());
        }
    }
    Ident::new(&pascal, ident.span())
}
