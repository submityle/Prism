//! Expansion logic for `#[derive(Relation)]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, LitBool, LitStr};

/// A `prism_ecs::relation::CleanupPolicy` variant selected by name in a
/// `#[relation(on_delete = "...")]` / `#[relation(on_delete_target = "...")]`
/// attribute.
#[derive(Clone, Copy)]
enum PolicyToken {
    /// Mirrors `CleanupPolicy::Remove` (the default on both deletion slots).
    Remove,
    /// Mirrors `CleanupPolicy::Delete`.
    Delete,
    /// Mirrors `CleanupPolicy::Panic`.
    Panic,
}

impl PolicyToken {
    /// The matching `CleanupPolicy` variant as an absolute path token.
    fn path(self) -> TokenStream {
        match self {
            PolicyToken::Remove => quote!(prism_ecs::relation::CleanupPolicy::Remove),
            PolicyToken::Delete => quote!(prism_ecs::relation::CleanupPolicy::Delete),
            PolicyToken::Panic => quote!(prism_ecs::relation::CleanupPolicy::Panic),
        }
    }

    /// Map a string literal (`"Remove"`/`"Delete"`/`"Panic"`, with lowercase
    /// short forms accepted) onto a variant, erroring on anything else with a
    /// span pointing at the offending literal.
    fn parse(lit: &LitStr) -> syn::Result<Self> {
        match lit.value().as_str() {
            "Remove" | "remove" => Ok(PolicyToken::Remove),
            "Delete" | "delete" => Ok(PolicyToken::Delete),
            "Panic" | "panic" => Ok(PolicyToken::Panic),
            other => Err(syn::Error::new_spanned(
                lit,
                format!(
                    "unknown cleanup policy `{other}`; expected \"Remove\", \"Delete\", or \"Panic\""
                ),
            )),
        }
    }
}

/// The parsed `#[relation(...)]` configuration, mirroring the public fields of
/// `prism_ecs::relation::RelationKind`.
struct RelationConfig {
    fragmenting: bool,
    transitive: bool,
    exclusive: bool,
    on_delete: PolicyToken,
    on_delete_target: PolicyToken,
}

impl Default for RelationConfig {
    fn default() -> Self {
        // Matches `RelationKind::default()`: non-fragmenting, non-transitive,
        // non-exclusive, and `CleanupPolicy::Remove` on both deletion slots.
        Self {
            fragmenting: false,
            transitive: false,
            exclusive: false,
            on_delete: PolicyToken::Remove,
            on_delete_target: PolicyToken::Remove,
        }
    }
}

/// Parse a boolean flag that is either bare (`exclusive`, meaning `true`) or
/// given an explicit value (`exclusive = false`).
fn parse_flag(meta: &syn::meta::ParseNestedMeta) -> syn::Result<bool> {
    if meta.input.peek(syn::Token![=]) {
        let lit: LitBool = meta.value()?.parse()?;
        Ok(lit.value())
    } else {
        Ok(true)
    }
}

/// Collect every `#[relation(...)]` attribute on `input` into a single
/// [`RelationConfig`]. Later attributes override earlier ones for the same key.
fn parse_relation_attrs(input: &DeriveInput) -> syn::Result<RelationConfig> {
    let mut config = RelationConfig::default();
    for attr in &input.attrs {
        if !attr.path().is_ident("relation") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            let ident = meta
                .path
                .get_ident()
                .ok_or_else(|| meta.error("expected an identifier in `#[relation(...)]`"))?;
            match ident.to_string().as_str() {
                "fragmenting" => config.fragmenting = parse_flag(&meta)?,
                "transitive" => config.transitive = parse_flag(&meta)?,
                "exclusive" => config.exclusive = parse_flag(&meta)?,
                "on_delete" => {
                    let lit: LitStr = meta.value()?.parse()?;
                    config.on_delete = PolicyToken::parse(&lit)?;
                }
                "on_delete_target" => {
                    let lit: LitStr = meta.value()?.parse()?;
                    config.on_delete_target = PolicyToken::parse(&lit)?;
                }
                other => {
                    return Err(meta.error(format!(
                        "unknown `relation` attribute key `{other}`; expected `fragmenting`, \
                         `transitive`, `exclusive`, `on_delete`, or `on_delete_target`"
                    )));
                }
            }
            Ok(())
        })?;
    }
    Ok(config)
}

/// Expand a `#[derive(Relation)]` input into an
/// `impl prism_ecs::relation::Relation` block whose associated `const KIND`
/// is the compile-time [`RelationKind`](prism_ecs::relation::RelationKind)
/// literal described by the `#[relation(...)]` attributes.
///
/// Because `Relation: Component`, the type must also derive (or hand-write)
/// [`Component`](macro@crate::Component); the generated impl does not restate
/// that bound. Generics and `where`-clauses are threaded through verbatim via
/// [`syn::Generics::split_for_impl`]. Unions are rejected; structs and enums
/// are accepted (relation kinds are typically zero-sized markers).
pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    if let syn::Data::Union(data) = &input.data {
        return Err(syn::Error::new_spanned(
            data.union_token,
            "`#[derive(Relation)]` cannot be applied to a `union`; a relation kind must be a \
             `struct` or `enum`",
        ));
    }

    let config = parse_relation_attrs(input)?;

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let RelationConfig {
        fragmenting,
        transitive,
        exclusive,
        on_delete,
        on_delete_target,
    } = config;
    let on_delete = on_delete.path();
    let on_delete_target = on_delete_target.path();

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics prism_ecs::relation::Relation for #name #ty_generics #where_clause {
            const KIND: prism_ecs::relation::RelationKind = prism_ecs::relation::RelationKind {
                fragmenting: #fragmenting,
                transitive: #transitive,
                exclusive: #exclusive,
                on_delete: #on_delete,
                on_delete_target: #on_delete_target,
            };
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn expand_str(input: DeriveInput) -> String {
        expand(&input).unwrap().to_string()
    }

    #[test]
    fn default_kind_when_no_attribute() {
        let di: DeriveInput = parse_quote! { struct ChildOf; };
        let out = expand_str(di);
        assert!(out.contains("impl prism_ecs :: relation :: Relation for ChildOf"));
        assert!(out.contains("const KIND"));
        // All flags default to `false`.
        assert!(out.contains("fragmenting : false"));
        assert!(out.contains("transitive : false"));
        assert!(out.contains("exclusive : false"));
        // Both deletion slots default to `Remove`.
        assert_eq!(out.matches("CleanupPolicy :: Remove").count(), 2);
    }

    #[test]
    fn bare_flags_mean_true() {
        let di: DeriveInput = parse_quote! {
            #[relation(fragmenting, transitive, exclusive)]
            struct ChildOf;
        };
        let out = expand_str(di);
        assert!(out.contains("fragmenting : true"));
        assert!(out.contains("transitive : true"));
        assert!(out.contains("exclusive : true"));
    }

    #[test]
    fn explicit_bool_values_are_honoured() {
        let di: DeriveInput = parse_quote! {
            #[relation(exclusive = true, fragmenting = false)]
            struct ChildOf;
        };
        let out = expand_str(di);
        assert!(out.contains("exclusive : true"));
        assert!(out.contains("fragmenting : false"));
    }

    #[test]
    fn cleanup_policies_map_to_variants() {
        let di: DeriveInput = parse_quote! {
            #[relation(on_delete = "Panic", on_delete_target = "Delete")]
            struct ChildOf;
        };
        let out = expand_str(di);
        assert!(out.contains("on_delete : prism_ecs :: relation :: CleanupPolicy :: Panic"));
        assert!(
            out.contains("on_delete_target : prism_ecs :: relation :: CleanupPolicy :: Delete")
        );
    }

    #[test]
    fn lowercase_policy_forms_are_accepted() {
        let di: DeriveInput = parse_quote! {
            #[relation(on_delete_target = "delete")]
            struct ChildOf;
        };
        let out = expand_str(di);
        assert!(
            out.contains("on_delete_target : prism_ecs :: relation :: CleanupPolicy :: Delete")
        );
    }

    #[test]
    fn unknown_policy_string_is_an_error() {
        let di: DeriveInput = parse_quote! {
            #[relation(on_delete = "Nope")]
            struct ChildOf;
        };
        assert!(expand(&di).is_err());
    }

    #[test]
    fn unknown_key_is_an_error() {
        let di: DeriveInput = parse_quote! {
            #[relation(bogus)]
            struct ChildOf;
        };
        assert!(expand(&di).is_err());
    }

    #[test]
    fn union_is_rejected() {
        let di: DeriveInput = parse_quote! {
            union U { a: u32, b: f32 }
        };
        assert!(expand(&di).is_err());
    }

    #[test]
    fn enum_is_accepted() {
        let di: DeriveInput = parse_quote! {
            enum Owns { Weak, Strong }
        };
        let out = expand_str(di);
        assert!(out.contains("impl prism_ecs :: relation :: Relation for Owns"));
    }

    #[test]
    fn generics_and_where_are_threaded() {
        let di: DeriveInput = parse_quote! {
            #[relation(exclusive)]
            struct Owns<T> where T: Send + Sync + 'static { _marker: core::marker::PhantomData<T> }
        };
        let out = expand_str(di);
        assert!(out.contains("impl < T > prism_ecs :: relation :: Relation for Owns < T >"));
        assert!(out.contains("where T : Send + Sync + 'static"));
    }
}
