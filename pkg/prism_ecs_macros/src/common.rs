//! Shared helpers used by both derive expansions: attribute parsing and
//! `where`-clause synthesis.

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{Attribute, Generics, LitStr, Type};

/// The storage strategy requested via `#[component(storage = "...")]`.
///
/// Mirrors `prism_ecs::component::StorageType`. We keep a tiny local mirror
/// rather than depending on `prism_ecs` (which would be a cyclic dependency);
/// the variant name is re-emitted verbatim behind the absolute path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Storage {
    /// Columnar table storage (`StorageType::Table`).
    Table,
    /// Sparse-set storage (`StorageType::SparseSet`).
    SparseSet,
}

impl Storage {
    /// The matching `StorageType` variant as a path token
    /// (`prism_ecs::component::StorageType::<Variant>`).
    pub fn variant_path(self) -> TokenStream {
        match self {
            Storage::Table => quote!(prism_ecs::component::StorageType::Table),
            Storage::SparseSet => quote!(prism_ecs::component::StorageType::SparseSet),
        }
    }
}

/// Parse the optional `#[component(storage = "Table" | "SparseSet")]`
/// attribute.
///
/// Returns:
/// - `Ok(None)` when no `#[component(...)]` attribute is present (or none set
///   `storage`),
/// - `Ok(Some(storage))` for a recognised value,
/// - `Err(..)` for an unknown key or an unrecognised storage string, with a
///   span pointing at the offending tokens.
pub fn parse_storage_attr(attrs: &[Attribute]) -> syn::Result<Option<Storage>> {
    let mut found: Option<Storage> = None;
    for attr in attrs {
        if !attr.path().is_ident("component") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("storage") {
                let lit: LitStr = meta.value()?.parse()?;
                let storage = match lit.value().as_str() {
                    "Table" => Storage::Table,
                    "SparseSet" => Storage::SparseSet,
                    other => {
                        return Err(meta.error(format!(
                            "unknown storage type `{other}`; expected \"Table\" or \"SparseSet\""
                        )));
                    }
                };
                found = Some(storage);
                Ok(())
            } else {
                Err(meta.error("unknown `component` attribute key; expected `storage`"))
            }
        })?;
    }
    Ok(found)
}

/// Build a `where`-clause token stream that combines the type's existing
/// predicates with one `ty: prism_ecs::bundle::Bundle` bound per *distinct*
/// field type.
///
/// Field types are de-duplicated by their token representation so a struct
/// with several fields of the same type emits the bound only once. Returns an
/// empty stream when there is nothing to constrain.
pub fn bundle_where_clause(generics: &Generics, field_types: &[&Type]) -> TokenStream {
    let mut predicates: Vec<TokenStream> = Vec::new();

    if let Some(existing) = generics.where_clause.as_ref() {
        for pred in &existing.predicates {
            predicates.push(quote!(#pred));
        }
    }

    let mut seen: Vec<String> = Vec::new();
    for ty in field_types {
        let key = ty.to_token_stream().to_string();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        predicates.push(quote!(#ty: prism_ecs::bundle::Bundle));
    }

    if predicates.is_empty() {
        TokenStream::new()
    } else {
        quote!(where #(#predicates),*)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::{DeriveInput, parse_quote};

    fn attrs_of(input: DeriveInput) -> Vec<Attribute> {
        input.attrs
    }

    #[test]
    fn storage_absent_is_none() {
        let di: DeriveInput = parse_quote! { struct Foo; };
        assert_eq!(parse_storage_attr(&attrs_of(di)).unwrap(), None);
    }

    #[test]
    fn storage_table_and_sparse_set() {
        let di: DeriveInput = parse_quote! {
            #[component(storage = "Table")]
            struct Foo;
        };
        assert_eq!(parse_storage_attr(&attrs_of(di)).unwrap(), Some(Storage::Table));

        let di: DeriveInput = parse_quote! {
            #[component(storage = "SparseSet")]
            struct Foo;
        };
        assert_eq!(
            parse_storage_attr(&attrs_of(di)).unwrap(),
            Some(Storage::SparseSet)
        );
    }

    #[test]
    fn storage_unknown_value_errors() {
        let di: DeriveInput = parse_quote! {
            #[component(storage = "Nope")]
            struct Foo;
        };
        assert!(parse_storage_attr(&attrs_of(di)).is_err());
    }

    #[test]
    fn storage_unknown_key_errors() {
        let di: DeriveInput = parse_quote! {
            #[component(bogus = "x")]
            struct Foo;
        };
        assert!(parse_storage_attr(&attrs_of(di)).is_err());
    }

    #[test]
    fn where_clause_dedups_and_is_empty_when_nothing() {
        let generics: Generics = parse_quote! {};
        assert!(bundle_where_clause(&generics, &[]).is_empty());

        let a: Type = parse_quote!(A);
        let b: Type = parse_quote!(B);
        let out = bundle_where_clause(&generics, &[&a, &b, &a]).to_string();
        // Bound emitted once per distinct type.
        assert_eq!(out.matches("Bundle").count(), 2);
    }
}
