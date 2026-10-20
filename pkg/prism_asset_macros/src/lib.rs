//! Procedural derives for asset identity and dependency disclosure.
//!
//! Use the re-exports in `prism_asset::prelude`. Expansion uses only `core`
//! and the runtime's public contracts, so consumers remain `no_std + alloc`.
//! This compiler-host crate intentionally uses `std` to resolve renamed Cargo
//! dependencies; none of that code is linked into the asset runtime.

#![deny(missing_docs)]

mod expand;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

fn derive(input: DeriveInput, asset: bool) -> syn::Result<proc_macro2::TokenStream> {
    let path = expand::crate_override(&input)?.map_or_else(
        || match proc_macro_crate::crate_name("prism_asset") {
            Ok(proc_macro_crate::FoundCrate::Itself) => Ok(quote!(::prism_asset)),
            Ok(proc_macro_crate::FoundCrate::Name(name)) => {
                let ident = syn::parse_str::<syn::Ident>(&format!("r#{name}"))?;
                Ok(quote!(::#ident))
            }
            Err(error) => Err(syn::Error::new_spanned(
                &input.ident,
                format!(
                    "cannot find prism_asset: {error}; use #[asset(crate = \"path\")] for a facade"
                ),
            )),
        },
        |path| Ok(quote!(#path)),
    )?;
    expand::expand(&input, &path, asset)
}

/// Derives `prism_asset::Asset` and `VisitAssetDependencies`.
///
/// Structs and enums are supported. `Handle`, `SoftHandle`, and
/// `UntypedHandle` fields are discovered through `Option`, `Vec`, `Box`,
/// `Arc`, arrays, references, and tuples. Other fields are not dependencies.
/// `#[dependency]` opts an alias/custom nested field into trait-based traversal;
/// `#[asset(skip)]` explicitly excludes a field. Weak handles are not promoted.
///
/// `#[asset(type_name = "my::stable::Name")]` overrides the default
/// `concat!(module_path!(), "::", stringify!(Type))` persisted identity.
/// `#[asset(crate = "path::to::facade")]` overrides Cargo path discovery.
///
/// Asset type/lifetime/const parameters are rejected: one constant name cannot
/// distinguish arbitrary monomorphizations. Use a concrete asset wrapper or a
/// handwritten `Asset` implementation with an explicit per-type identity.
/// Generic dependency records can derive `VisitAssetDependencies` instead.
///
/// ```compile_fail
/// #[derive(prism_asset_macros::Asset)]
/// struct AmbiguousIdentity<T>(T);
/// ```
///
/// ```compile_fail
/// #[derive(prism_asset_macros::Asset)]
/// struct Borrowed<'a>(&'a str);
/// ```
///
/// ```compile_fail
/// #[derive(prism_asset_macros::Asset)]
/// union CannotSafelyVisit { number: u64, other: f64 }
/// ```
///
/// ```compile_fail
/// #[derive(prism_asset_macros::Asset)]
/// struct AmbiguousArray<const N: usize>([u8; N]);
/// ```
///
/// ```compile_fail
/// use prism_asset_runtime::{Asset, Handle};
/// #[derive(Asset)]
/// struct Image;
/// #[derive(Asset)]
/// struct Unsupported { images: core::result::Result<Handle<Image>, ()> }
/// ```
///
/// ```compile_fail
/// use prism_asset_runtime::{Asset, Handle};
/// #[derive(Asset)]
/// struct Image;
/// #[derive(Asset)]
/// struct Conflicting { #[dependency] #[asset(skip)] image: Handle<Image> }
/// ```
#[proc_macro_derive(Asset, attributes(asset, dependency))]
pub fn derive_asset(input: TokenStream) -> TokenStream {
    derive(parse_macro_input!(input as DeriveInput), true)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derives dependency traversal for a nested record without making it an asset.
///
/// Supports the same field shapes and attributes as `Asset`, preserves generics
/// and where clauses, and adds bounds only to visited leaf types. Annotate a
/// field containing such a record with `#[dependency]`. Do not derive this
/// alongside `Asset`, which already generates the implementation.
#[proc_macro_derive(VisitAssetDependencies, attributes(asset, dependency))]
pub fn derive_visit_asset_dependencies(input: TokenStream) -> TokenStream {
    derive(parse_macro_input!(input as DeriveInput), false)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
