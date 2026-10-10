//! Expansion logic for `#[derive(Resource)]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::DeriveInput;

/// Expand a `#[derive(Resource)]` input into an
/// `impl prism_ecs::resource::Resource` block.
///
/// [`Resource`](prism_ecs::resource::Resource) is a bare marker trait
/// (`Send + Sync + 'static`), so the generated `impl` has an empty body. The
/// type's generics and `where`-clause are preserved verbatim via
/// [`syn::Generics::split_for_impl`], so a generic singleton such as
/// `Store<T>` can derive the trait once and apply to every valid `T`.
///
/// Unlike `#[derive(Component)]`, there is no storage attribute: a resource is
/// stored once per world, not per entity, so there is no per-type storage
/// strategy to select.
pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics prism_ecs::resource::Resource for #name #ty_generics #where_clause {}
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
        let di: DeriveInput = parse_quote! { struct Clock { tick: u64 } };
        let out = expand_str(di);
        assert!(out.contains("impl prism_ecs :: resource :: Resource for Clock"));
        // Marker trait: no associated items in the body.
        assert!(!out.contains("fn "));
        assert!(!out.contains("const "));
    }

    #[test]
    fn unit_and_tuple_structs_are_supported() {
        let unit: DeriveInput = parse_quote! { struct Paused; };
        assert!(expand_str(unit).contains("impl prism_ecs :: resource :: Resource for Paused"));

        let tuple: DeriveInput = parse_quote! { struct FrameCount(u64); };
        assert!(
            expand_str(tuple).contains("impl prism_ecs :: resource :: Resource for FrameCount")
        );
    }

    #[test]
    fn generics_and_where_are_threaded() {
        let di: DeriveInput = parse_quote! {
            struct Store<T> where T: Send + Sync + 'static { inner: T }
        };
        let out = expand_str(di);
        assert!(out.contains("impl < T > prism_ecs :: resource :: Resource for Store < T >"));
        assert!(out.contains("where T : Send + Sync + 'static"));
    }

    #[test]
    fn fieldless_enum_is_supported() {
        let di: DeriveInput = parse_quote! { enum Mode { Edit, Play } };
        let out = expand_str(di);
        assert!(out.contains("impl prism_ecs :: resource :: Resource for Mode"));
    }
}
