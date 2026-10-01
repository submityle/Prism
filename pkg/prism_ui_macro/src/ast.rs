//! Parsed representation of the `loom!` DSL.
//!
//! The types here mirror the DSL grammar one-to-one and implement
//! [`syn::parse::Parse`], so parsing is a direct recursive descent over the
//! macro input. Lowering of this AST to builder calls lives in
//! [`crate::lower`].

use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{braced, parenthesized, token, Expr, Ident, Lit, LitInt, LitStr, Token};

/// Top-level macro input: a single root [`Node`] plus an optional trailing
/// semicolon.
pub(crate) struct LoomInput {
    /// The root node of the tree.
    pub(crate) node: Node,
}

impl Parse for LoomInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let node = input.parse()?;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
        }
        if !input.is_empty() {
            return Err(input.error("unexpected trailing tokens after `loom!` root node"));
        }
        Ok(Self { node })
    }
}

/// A single UI node: a `box`, a `text(..)` run or a `custom(..)` element.
pub(crate) struct Node {
    /// Which kind of element this node builds.
    pub(crate) kind: NodeKind,
    /// Attributes declared in the node's block, in source order.
    pub(crate) attrs: Vec<Attr>,
    /// Child statements declared in the node's block, in source order.
    pub(crate) children: Vec<Child>,
}

/// The element constructor a [`Node`] maps to.
pub(crate) enum NodeKind {
    /// `box` -> `Element::box_()`.
    Box,
    /// `text(EXPR)` -> `Element::text(EXPR)`.
    Text(Expr),
    /// `custom(EXPR)` -> `Element::custom(EXPR)`.
    Custom(Expr),
}

/// A child statement inside a node block.
pub(crate) enum Child {
    /// A nested node, lowered via `.child(..)`.
    Node(Node),
    /// A `for_each(EXPR)` splice, lowered via `.children(EXPR)`.
    ForEach(Expr),
}

/// An attribute line inside a node block.
pub(crate) enum Attr {
    /// `class: a, b, c;` -> one `.class(..)` per name.
    Class(Vec<Expr>),
    /// `key: 42;` -> `.key_int(42)`.
    KeyInt(LitInt),
    /// `key: "id";` -> `.key_str("id")`.
    KeyStr(LitStr),
    /// `style: { .. };` -> one `.style(..)` per entry.
    Style(Vec<StyleEntry>),
}

/// A single `prop: value` pair inside a `style { .. }` block.
pub(crate) struct StyleEntry {
    /// The `snake_case` property identifier (e.g. `background_color`).
    pub(crate) prop: Ident,
    /// The value assigned to the property.
    pub(crate) value: StyleVal,
}

/// A style value as written in the DSL.
pub(crate) enum StyleVal {
    /// A bare numeric literal -> `StyleValue::Number(x as f32)`.
    Number(Lit),
    /// A bare keyword identifier -> `StyleValue::Keyword(Keyword::X)`.
    Keyword(Ident),
    /// A constructor call: `px(..)`, `token(..)` or `rgba8(..)`.
    Call {
        /// The constructor name (one of `px`, `token`, `rgba8`).
        func: Ident,
        /// The call arguments, spliced verbatim.
        args: Punctuated<Expr, Token![,]>,
    },
}

impl Parse for Node {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = Ident::parse_any(input)?;
        let kind = match name.to_string().as_str() {
            "box" => NodeKind::Box,
            "text" => {
                let content;
                parenthesized!(content in input);
                NodeKind::Text(content.parse()?)
            }
            "custom" => {
                let content;
                parenthesized!(content in input);
                NodeKind::Custom(content.parse()?)
            }
            _ => {
                return Err(syn::Error::new(
                    name.span(),
                    "expected a node: `box`, `text(..)` or `custom(..)`",
                ));
            }
        };

        let mut attrs = Vec::new();
        let mut children = Vec::new();
        if input.peek(token::Brace) {
            let content;
            braced!(content in input);
            parse_block(&content, &mut attrs, &mut children)?;
        }

        Ok(Self {
            kind,
            attrs,
            children,
        })
    }
}

/// Parses the body of a node block into attributes and child statements.
fn parse_block(
    input: ParseStream,
    attrs: &mut Vec<Attr>,
    children: &mut Vec<Child>,
) -> syn::Result<()> {
    while !input.is_empty() {
        if input.peek(Ident) && input.peek2(Token![:]) {
            attrs.push(input.parse()?);
        } else {
            children.push(parse_child(input)?);
        }
    }
    Ok(())
}

/// Parses a single child statement: either `for_each(..)` or a nested node.
fn parse_child(input: ParseStream) -> syn::Result<Child> {
    let fork = input.fork();
    let name = Ident::parse_any(&fork)?;
    let child = if name == "for_each" {
        Ident::parse_any(input)?;
        let content;
        parenthesized!(content in input);
        Child::ForEach(content.parse()?)
    } else {
        Child::Node(input.parse()?)
    };
    if input.peek(Token![;]) {
        input.parse::<Token![;]>()?;
    }
    Ok(child)
}

impl Parse for Attr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let attr = match name.to_string().as_str() {
            "class" => {
                let names = Punctuated::<Expr, Token![,]>::parse_separated_nonempty(input)?;
                Attr::Class(names.into_iter().collect())
            }
            "key" => {
                let lit: Lit = input.parse()?;
                match lit {
                    Lit::Int(int) => Attr::KeyInt(int),
                    Lit::Str(string) => Attr::KeyStr(string),
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "`key` must be an integer or string literal",
                        ));
                    }
                }
            }
            "style" => {
                let content;
                braced!(content in input);
                Attr::Style(parse_style_entries(&content)?)
            }
            _ => {
                return Err(syn::Error::new(
                    name.span(),
                    "unknown attribute: expected `class`, `key` or `style`",
                ));
            }
        };
        input.parse::<Token![;]>()?;
        Ok(attr)
    }
}

/// Parses the semicolon-separated entries of a `style { .. }` block.
fn parse_style_entries(input: ParseStream) -> syn::Result<Vec<StyleEntry>> {
    let mut entries = Vec::new();
    loop {
        if input.is_empty() {
            break;
        }
        entries.push(input.parse()?);
        if input.is_empty() {
            break;
        }
        input.parse::<Token![;]>()?;
    }
    Ok(entries)
}

impl Parse for StyleEntry {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let prop = Ident::parse_any(input)?;
        input.parse::<Token![:]>()?;
        let value = input.parse()?;
        Ok(Self { prop, value })
    }
}

impl Parse for StyleVal {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Lit) {
            let lit: Lit = input.parse()?;
            return match lit {
                Lit::Int(_) | Lit::Float(_) => Ok(StyleVal::Number(lit)),
                other => Err(syn::Error::new_spanned(
                    other,
                    "style value literal must be a number; use `token(\"..\")` for strings",
                )),
            };
        }

        let func = Ident::parse_any(input)?;
        if input.peek(token::Paren) {
            let content;
            parenthesized!(content in input);
            let args = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?;
            match func.to_string().as_str() {
                "px" | "token" | "rgba8" => Ok(StyleVal::Call { func, args }),
                _ => Err(syn::Error::new(
                    func.span(),
                    "unknown style constructor: expected `px`, `token` or `rgba8`",
                )),
            }
        } else {
            Ok(StyleVal::Keyword(func))
        }
    }
}
