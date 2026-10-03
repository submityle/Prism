//! Expansion logic for `#[derive(Component)]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::DeriveInput;

use crate::common;

/// Expand a `#[derive(Component)]` input into an
/// `impl prism_ecs::component::Component` block.
///
/// The impl body is empty unless a `#[component(storage = "...")]` attribute
/// requested an explicit storage strategy, in which case a matching
/// `const STORAGE` is emitted. The type's generics and `where`-clause are
/// preserved verbatim via [`syn::Generics::split_for_impl`].
pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let storage = common::parse_storage_attr(&input.attrs)?;

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Only emit `const STORAGE` when explicitly requested; otherwise the
    // trait's default (`StorageType::Table`) applies.
    let body = match storage {
        Some(storage) => {
            let variant = storage.variant_path();
            quote! {
                const STORAGE: prism_ecs::component::StorageType = #variant;
            }
        }
        None => TokenStream::new(),
    };

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics prism_ecs::component::Component for #name #ty_generics #where_clause {
            #body
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
    fn plain_struct_has_empty_body() {
        let di: DeriveInput = parse_quote! { struct Foo { x: u32 } };
        let out = expand_str(di);
        assert!(out.contains("impl prism_ecs :: component :: Component for Foo"));
        assert!(!out.contains("STORAGE"));
    }

    #[test]
    fn storage_sparse_set_emits_const() {
        let di: DeriveInput = parse_quote! {
            #[component(storage = "SparseSet")]
            struct Foo;
        };
        let out = expand_str(di);
        assert!(out.contains("const STORAGE"));
        assert!(out.contains("StorageType :: SparseSet"));
    }

    #[test]
    fn storage_table_emits_const() {
        let di: DeriveInput = parse_quote! {
            #[component(storage = "Table")]
            struct Foo;
        };
        let out = expand_str(di);
        assert!(out.contains("StorageType :: Table"));
    }

    #[test]
    fn generics_and_where_are_threaded() {
        let di: DeriveInput = parse_quote! {
            struct Wrapper<T> where T: Send + Sync + 'static { inner: T }
        };
        let out = expand_str(di);
        assert!(out.contains("impl < T > prism_ecs :: component :: Component for Wrapper < T >"));
        assert!(out.contains("where T : Send + Sync + 'static"));
    }

    #[test]
    fn invalid_storage_is_compile_error() {
        let di: DeriveInput = parse_quote! {
            #[component(storage = "Bogus")]
            struct Foo;
        };
        assert!(expand(&di).is_err());
    }
}
