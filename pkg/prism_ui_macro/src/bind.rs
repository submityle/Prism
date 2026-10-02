//! `$` auto field-binding sugar (§9.9).
//!
//! The [`bind!`](crate::bind) macro turns a one-line declaration into the code
//! that registers a `prism_ui_ecs::FieldBinding` on an
//! [`EcsBridge`](prism_ui_ecs::EcsBridge), tying one field of an ECS component
//! to a reactive signal:
//!
//! ```ignore
//! // one-way: component field -> signal
//! bind!(bridge, label <- $entity.Name.0 : String);
//! // two-way: component field <-> signal
//! bind!(bridge, health <-> $entity.Health.current : u32);
//! ```
//!
//! The read path reuses the component's change tick (`Ref`-style change
//! detection inside [`FieldBinding::pull`]); the write path uses a `Mut` plus
//! an equality guard inside [`FieldBinding::push`]. The frame ordering
//! "pull before push" is provided by the already-shipped schedule helper
//! `prism_ui_ecs::add_loom_sync_systems`, which configures
//! `(LoomSyncSet::Pull, LoomSyncSet::Push).chain()`; `bind!` only registers the
//! binding into the bridge those systems drive.
//!
//! Field-path parsing is a pure function ([`FieldPath::parse`] via
//! [`syn::parse::Parse`]), so it is unit-testable in isolation, and an illegal
//! path produces a **compile-time** error whose span points at the offending
//! token — never a runtime panic.
//!
//! [`FieldBinding::pull`]: prism_ui_ecs::FieldBinding::pull
//! [`FieldBinding::push`]: prism_ui_ecs::FieldBinding::push

use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, Member, Token, Type};

/// A parsed `$entity.Component.field` binding path.
///
/// `entity` is the entity-holding binding, `component` is the ECS component
/// type, and `field` is the projected field (a named field or a tuple index
/// such as `0`).
pub(crate) struct FieldPath {
    /// The entity value (the local binding after the `$` sigil).
    pub(crate) entity: Ident,
    /// The component type whose field is projected.
    pub(crate) component: Ident,
    /// The projected field (named or tuple index).
    pub(crate) field: Member,
}

impl Parse for FieldPath {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        eat_dollar(input)?;
        let entity: Ident = input
            .parse()
            .map_err(|_| syn::Error::new(input.span(), "expected an entity binding after `$`"))?;
        input.parse::<Token![.]>().map_err(|_| {
            syn::Error::new(
                entity.span(),
                "field path must be `$entity.Component.field`: missing `.Component`",
            )
        })?;
        let component: Ident = input.parse().map_err(|_| {
            syn::Error::new(input.span(), "expected a component type after `$entity.`")
        })?;
        input.parse::<Token![.]>().map_err(|_| {
            syn::Error::new(
                component.span(),
                "field path must be `$entity.Component.field`: missing `.field`",
            )
        })?;
        let field: Member = input.parse().map_err(|_| {
            syn::Error::new(
                input.span(),
                "expected a field name or tuple index after the component",
            )
        })?;
        Ok(Self {
            entity,
            component,
            field,
        })
    }
}

impl FieldPath {
    /// The reader closure `|c: &Component| c.field.clone()`.
    fn reader(&self) -> TokenStream {
        let component = &self.component;
        let field = &self.field;
        quote! { |__loom_c: &#component| ::core::clone::Clone::clone(&__loom_c.#field) }
    }

    /// The writer closure `|c: &mut Component, v: &T| { c.field = v.clone(); }`.
    fn writer(&self, ty: &Type) -> TokenStream {
        let component = &self.component;
        let field = &self.field;
        quote! {
            |__loom_c: &mut #component, __loom_v: &#ty| {
                __loom_c.#field = ::core::clone::Clone::clone(__loom_v);
            }
        }
    }
}

/// Consumes a leading `$` sigil or reports a compile-time error.
fn eat_dollar(input: ParseStream) -> syn::Result<()> {
    input.step(|cursor| match cursor.punct() {
        Some((punct, rest)) if punct.as_char() == '$' => Ok(((), rest)),
        _ => Err(cursor.error("field binding path must start with the `$` sigil")),
    })
}

/// The binding direction selected by the arrow operator.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// `<-` : one-way, component field -> signal.
    OneWay,
    /// `<->` : two-way, component field <-> signal.
    TwoWay,
}

/// Parsed `bind!(..)` invocation.
pub(crate) struct BindInput {
    /// The `EcsBridge` expression the binding is registered on.
    bridge: Expr,
    /// The signal expression (collected verbatim up to the arrow).
    signal: TokenStream,
    /// The binding direction.
    direction: Direction,
    /// The `$entity.Component.field` path.
    path: FieldPath,
    /// The projected field type.
    ty: Type,
}

impl Parse for BindInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let bridge: Expr = input.parse()?;
        input.parse::<Token![,]>()?;

        // Collect the signal expression verbatim up to the arrow. We cannot use
        // `Expr` here because its parser would treat the leading `<` of the
        // arrow as a comparison operator. Stopping at the first top-level `<`
        // keeps common signal targets (`sig`, `state.value`, `sigs[0]`)
        // working.
        let mut signal = TokenStream::new();
        if input.peek(Token![<]) {
            return Err(input.error("expected a signal expression before the binding arrow"));
        }
        while !input.peek(Token![<]) {
            if input.is_empty() {
                return Err(input
                    .error("expected a binding arrow `<-` or `<->` after the signal expression"));
            }
            let tt: proc_macro2::TokenTree = input.parse()?;
            tt.to_tokens(&mut signal);
        }

        let direction = parse_direction(input)?;
        let path: FieldPath = input.parse()?;
        input.parse::<Token![:]>().map_err(|_| {
            syn::Error::new(
                input.span(),
                "a binding needs a field type: `.. : Type` (e.g. `: u32`)",
            )
        })?;
        let ty: Type = input.parse()?;

        if !input.is_empty() {
            return Err(input.error("unexpected trailing tokens after `bind!` arguments"));
        }

        Ok(Self {
            bridge,
            signal,
            direction,
            path,
            ty,
        })
    }
}

/// Parses the binding arrow, returning the selected [`Direction`].
///
/// After consuming the leading `<`, the remaining `->` (two-way) or `-`
/// (one-way) disambiguates the two forms.
fn parse_direction(input: ParseStream) -> syn::Result<Direction> {
    input.parse::<Token![<]>()?;
    if input.peek(Token![->]) {
        input.parse::<Token![->]>()?;
        Ok(Direction::TwoWay)
    } else if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        Ok(Direction::OneWay)
    } else {
        Err(input.error("expected a binding arrow `<-` or `<->`"))
    }
}

/// Expands a parsed [`BindInput`] into the bridge-registration call.
pub(crate) fn expand(input: BindInput) -> TokenStream {
    let BindInput {
        bridge,
        signal,
        direction,
        path,
        ty,
    } = input;
    let entity = &path.entity;
    let component = &path.component;
    let reader = path.reader();

    match direction {
        Direction::OneWay => quote! {
            (#bridge).bind::<#component, #ty>(#entity, #signal, #reader)
        },
        Direction::TwoWay => {
            let writer = path.writer(&ty);
            quote! {
                (#bridge).bind_two_way::<#component, #ty>(
                    #entity,
                    #signal,
                    #reader,
                    #writer,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_str;

    #[test]
    fn parses_named_field_path() {
        let path: FieldPath = parse_str("$entity.Health.current").expect("valid path");
        assert_eq!(path.entity.to_string(), "entity");
        assert_eq!(path.component.to_string(), "Health");
        assert!(matches!(path.field, Member::Named(ref ident) if ident == "current"));
    }

    #[test]
    fn parses_tuple_index_field_path() {
        let path: FieldPath = parse_str("$e.Name.0").expect("valid path");
        assert_eq!(path.component.to_string(), "Name");
        assert!(matches!(path.field, Member::Unnamed(ref index) if index.index == 0));
    }

    #[test]
    fn missing_dollar_is_an_error() {
        assert!(parse_str::<FieldPath>("entity.Health.current").is_err());
    }

    #[test]
    fn missing_field_is_an_error() {
        assert!(parse_str::<FieldPath>("$entity.Health").is_err());
    }

    #[test]
    fn missing_component_is_an_error() {
        assert!(parse_str::<FieldPath>("$entity").is_err());
    }

    #[test]
    fn one_way_expansion_calls_bind() {
        let input: BindInput =
            parse_str("bridge, label <- $entity.Name.0 : String").expect("valid bind");
        let code = expand(input).to_string();
        assert!(code.contains("bind :: < Name , String >"), "got: {code}");
        assert!(!code.contains("bind_two_way"), "got: {code}");
    }

    #[test]
    fn two_way_expansion_calls_bind_two_way() {
        let input: BindInput =
            parse_str("bridge, hp <-> $entity.Health.current : u32").expect("valid bind");
        let code = expand(input).to_string();
        assert!(
            code.contains("bind_two_way :: < Health , u32 >"),
            "got: {code}"
        );
    }

    #[test]
    fn dotted_signal_expression_is_preserved() {
        let input: BindInput =
            parse_str("ctx.bridge, state.value <-> $entity.Health.current : u32")
                .expect("valid bind");
        assert_eq!(input.signal.to_string(), "state . value");
        assert_eq!(input.bridge.to_token_stream().to_string(), "ctx . bridge");
    }

    #[test]
    fn missing_type_is_an_error() {
        assert!(parse_str::<BindInput>("bridge, label <- $entity.Name.0").is_err());
    }

    #[test]
    fn missing_arrow_is_an_error() {
        assert!(parse_str::<BindInput>("bridge, label $entity.Name.0 : String").is_err());
    }

    #[test]
    fn empty_signal_is_an_error() {
        assert!(parse_str::<BindInput>("bridge, <- $entity.Name.0 : String").is_err());
    }

    #[test]
    fn reader_clones_the_field() {
        let path: FieldPath = parse_str("$e.Health.current").expect("valid path");
        let reader = path.reader().to_string();
        assert!(reader.contains("__loom_c . current"), "got: {reader}");
    }

    #[test]
    fn writer_assigns_the_field() {
        let path: FieldPath = parse_str("$e.Health.current").expect("valid path");
        let ty: Type = parse_str("u32").expect("valid type");
        let writer = path.writer(&ty).to_string();
        assert!(writer.contains("__loom_c . current ="), "got: {writer}");
    }
}
