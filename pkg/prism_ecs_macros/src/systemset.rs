//! Expansion logic for `#[derive(SystemSet)]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields};

/// Expand a `#[derive(SystemSet)]` input into an
/// `impl prism_ecs::schedule::SystemSet` block.
///
/// The generated `set_id` returns a stable
/// [`SystemSetId`](prism_ecs::schedule::SystemSetId):
///
/// - **struct** (unit, tuple, or named) — a single label, so `set_id` is
///   `SystemSetId::of::<Self>()` (discriminant `0`). Field *values* are ignored:
///   a struct label denotes one set regardless of its contents.
/// - **fieldless enum** — one distinct set per variant, so `set_id` matches on
///   `self` and returns `SystemSetId::with::<Self>(i)` for the `i`-th variant.
///
/// A `union`, or an enum with any data-carrying variant, is a compile error:
/// a system-set label must have a finite, value-independent identity.
pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let body = match &input.data {
        Data::Struct(_) => quote! {
            prism_ecs::schedule::SystemSetId::of::<Self>()
        },
        Data::Enum(data) => {
            let mut arms: Vec<TokenStream> = Vec::with_capacity(data.variants.len());
            for (index, variant) in data.variants.iter().enumerate() {
                if !matches!(variant.fields, Fields::Unit) {
                    return Err(syn::Error::new_spanned(
                        &variant.fields,
                        "`#[derive(SystemSet)]` supports only fieldless enum variants; \
                         a data-carrying variant has no stable value-independent identity",
                    ));
                }
                let ident = &variant.ident;
                let discriminant = index as u64;
                arms.push(quote! {
                    Self::#ident => prism_ecs::schedule::SystemSetId::with::<Self>(#discriminant),
                });
            }
            quote! {
                match self {
                    #(#arms)*
                }
            }
        }
        Data::Union(data) => {
            return Err(syn::Error::new_spanned(
                data.union_token,
                "`#[derive(SystemSet)]` cannot be derived for a union",
            ));
        }
    };

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics prism_ecs::schedule::SystemSet for #name #ty_generics #where_clause {
            fn set_id(&self) -> prism_ecs::schedule::SystemSetId {
                #body
            }
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
    fn unit_struct_uses_of() {
        let di: DeriveInput = parse_quote! { struct Physics; };
        let out = expand_str(di);
        assert!(out.contains("impl prism_ecs :: schedule :: SystemSet for Physics"));
        assert!(out.contains("SystemSetId :: of :: < Self > ()"));
    }

    #[test]
    fn fieldless_enum_matches_variants() {
        let di: DeriveInput = parse_quote! {
            enum Sync { Pull, Push }
        };
        let out = expand_str(di);
        assert!(out.contains("match self"));
        assert!(out.contains("Self :: Pull => prism_ecs :: schedule :: SystemSetId :: with :: < Self > (0u64)"));
        assert!(out.contains("Self :: Push => prism_ecs :: schedule :: SystemSetId :: with :: < Self > (1u64)"));
    }

    #[test]
    fn data_enum_variant_is_error() {
        let di: DeriveInput = parse_quote! {
            enum Bad { A(u32), B }
        };
        assert!(expand(&di).is_err());
    }

    #[test]
    fn union_is_error() {
        let di: DeriveInput = parse_quote! {
            union U { a: u32, b: f32 }
        };
        assert!(expand(&di).is_err());
    }

    #[test]
    fn generics_are_threaded() {
        let di: DeriveInput = parse_quote! {
            struct Wrapper<T> where T: Send + Sync + 'static { inner: core::marker::PhantomData<T> }
        };
        let out = expand_str(di);
        assert!(out.contains("impl < T > prism_ecs :: schedule :: SystemSet for Wrapper < T >"));
        assert!(out.contains("where T : Send + Sync + 'static"));
    }
}
