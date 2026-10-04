//! Shared helpers used by both derive expansions: attribute parsing and
//! `where`-clause synthesis.

use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
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
    /// Unity-style shared storage (`StorageType::Shared`): the value is
    /// de-duplicated and used as a per-archetype batch key (design §6). The
    /// derive additionally overrides `install_storage_glue` to register the
    /// value-boxing glue, and the component must therefore be `Eq + Hash`.
    Shared,
}

impl Storage {
    /// The matching `StorageType` variant as a path token
    /// (`prism_ecs::component::StorageType::<Variant>`).
    pub fn variant_path(self) -> TokenStream {
        match self {
            Storage::Table => quote!(prism_ecs::component::StorageType::Table),
            Storage::SparseSet => quote!(prism_ecs::component::StorageType::SparseSet),
            Storage::Shared => quote!(prism_ecs::component::StorageType::Shared),
        }
    }

    /// Whether this storage strategy needs `install_storage_glue` to be
    /// overridden by the derive. Only [`Storage::Shared`] does: it must register
    /// its [`shared_box_of`](prism_ecs::component::shared_box_of) value-boxing
    /// glue so the structural code can intern the component's values.
    pub fn needs_install_glue(self) -> bool {
        matches!(self, Storage::Shared)
    }
}

/// Parse the optional `#[component(storage = "Table" | "SparseSet" | "Shared")]`
/// attribute (the lowercase short forms `"table"`, `"sparse"`, `"shared"`
/// used by design §6 are also accepted).
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
                    "Table" | "table" => Storage::Table,
                    "SparseSet" | "sparse" | "sparse_set" => Storage::SparseSet,
                    "Shared" | "shared" => Storage::Shared,
                    other => {
                        return Err(meta.error(format!(
                            "unknown storage type `{other}`; expected \"Table\", \"SparseSet\" (or \"sparse\"), or \"Shared\" (or \"shared\")"
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
    use syn::{parse_quote, DeriveInput};

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
        assert_eq!(
            parse_storage_attr(&attrs_of(di)).unwrap(),
            Some(Storage::Table)
        );

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
    fn storage_shared_and_lowercase_forms() {
        for src in ["Shared", "shared"] {
            let di: DeriveInput = syn::parse_str::<DeriveInput>(&format!(
                "#[component(storage = \"{src}\")] struct Foo;"
            ))
            .unwrap();
            assert_eq!(
                parse_storage_attr(&attrs_of(di)).unwrap(),
                Some(Storage::Shared),
                "storage = {src}"
            );
        }
        // Lowercase short forms from design §6 are accepted too.
        let di: DeriveInput = parse_quote! {
            #[component(storage = "sparse")]
            struct Foo;
        };
        assert_eq!(
            parse_storage_attr(&attrs_of(di)).unwrap(),
            Some(Storage::SparseSet)
        );
        let di: DeriveInput = parse_quote! {
            #[component(storage = "table")]
            struct Foo;
        };
        assert_eq!(
            parse_storage_attr(&attrs_of(di)).unwrap(),
            Some(Storage::Table)
        );
    }

    #[test]
    fn shared_needs_install_glue_others_do_not() {
        assert!(Storage::Shared.needs_install_glue());
        assert!(!Storage::Table.needs_install_glue());
        assert!(!Storage::SparseSet.needs_install_glue());
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
