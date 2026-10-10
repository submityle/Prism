//! Expansion logic for `#[derive(Event)]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::DeriveInput;

/// Expand a `#[derive(Event)]` input into an
/// `impl prism_ecs::event::Event` block.
///
/// [`Event`](prism_ecs::event::Event) is a bare marker trait
/// (`Send + Sync + 'static`); the double-buffered [`Events<E>`] machinery
/// (design §16.7) only requires the type to be thread-shareable and owned, so
/// the generated `impl` has an empty body. The type's generics and
/// `where`-clause are preserved verbatim via
/// [`syn::Generics::split_for_impl`].
///
/// [`Events<E>`]: prism_ecs::event::Events
pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics prism_ecs::event::Event for #name #ty_generics #where_clause {}
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
    fn plain_struct_has_empty_marker_impl() {
        let di: DeriveInput = parse_quote! { struct Collision { a: u32, b: u32 } };
        let out = expand_str(di);
        assert!(out.contains("impl prism_ecs :: event :: Event for Collision"));
        // Marker trait: no associated items in the body.
        assert!(!out.contains("fn "));
        assert!(!out.contains("const "));
    }

    #[test]
    fn unit_and_tuple_structs_are_supported() {
        let unit: DeriveInput = parse_quote! { struct AppExit; };
        assert!(expand_str(unit).contains("impl prism_ecs :: event :: Event for AppExit"));

        let tuple: DeriveInput = parse_quote! { struct Damage(f32); };
        assert!(expand_str(tuple).contains("impl prism_ecs :: event :: Event for Damage"));
    }

    #[test]
    fn generics_and_where_are_threaded() {
        let di: DeriveInput = parse_quote! {
            struct Payload<T> where T: Send + Sync + 'static { value: T }
        };
        let out = expand_str(di);
        assert!(out.contains("impl < T > prism_ecs :: event :: Event for Payload < T >"));
        assert!(out.contains("where T : Send + Sync + 'static"));
    }

    #[test]
    fn data_carrying_enum_is_supported() {
        let di: DeriveInput = parse_quote! { enum Input { Key(u32), Click { x: f32, y: f32 } } };
        let out = expand_str(di);
        assert!(out.contains("impl prism_ecs :: event :: Event for Input"));
    }
}
