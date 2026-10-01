//! Lowering from the parsed [`crate::ast`] to builder-call token streams.
//!
//! Each [`Node`] becomes a `::prism_ui::Element` constructor followed by a
//! chain of builder methods. All emitted paths are fully qualified so the
//! generated code does not depend on the caller's imports.

use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::ast::{Attr, Child, Node, NodeKind, StyleVal};

/// Lowers a [`Node`] into the builder-call chain that constructs it.
pub(crate) fn lower_node(node: &Node) -> TokenStream {
    let mut expr = match &node.kind {
        NodeKind::Box => quote! { ::prism_ui::Element::box_() },
        NodeKind::Text(content) => quote! { ::prism_ui::Element::text(#content) },
        NodeKind::Custom(name) => quote! { ::prism_ui::Element::custom(#name) },
    };

    for attr in &node.attrs {
        expr = lower_attr(expr, attr);
    }

    for child in &node.children {
        expr = match child {
            Child::Node(child_node) => {
                let child_expr = lower_node(child_node);
                quote! { #expr.child(#child_expr) }
            }
            Child::ForEach(iter) => quote! { #expr.children(#iter) },
        };
    }

    expr
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
        StyleVal::Call { func, args } => match func.to_string().as_str() {
            "px" => quote! { ::prism_ui::style::StyleValue::px(#args) },
            "token" => quote! { ::prism_ui::style::StyleValue::token(#args) },
            "rgba8" => quote! { ::prism_ui::style::StyleValue::rgba8(#args) },
            // The parser only ever constructs the three cases above.
            other => unreachable!("unexpected style constructor `{other}`"),
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
