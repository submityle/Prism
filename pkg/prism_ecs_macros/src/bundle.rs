//! Expansion logic for `#[derive(Bundle)]`.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Ident, Type};

use crate::common;

/// The field layout a bundle struct destructures into: the ordered field
/// types, the bindings used in the `get_components` destructure pattern, and
/// the pattern itself.
struct FieldPlan {
    /// Field types in declaration order (parallel to `bindings`).
    types: Vec<Type>,
    /// Binding identifiers in declaration order (parallel to `types`).
    bindings: Vec<Ident>,
    /// The `let Self { .. } = self;` / `let Self(..) = self;` destructure, or
    /// empty for a unit struct.
    pattern: TokenStream,
}

/// Expand a `#[derive(Bundle)]` input into an
/// `unsafe impl prism_ecs::bundle::Bundle` block.
///
/// Rejects enums and unions with a `compile_error!`-carrying [`syn::Error`].
pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let data = match &input.data {
        Data::Struct(data) => data,
        Data::Enum(data) => {
            return Err(syn::Error::new(
                data.enum_token.span,
                "`Bundle` cannot be derived for enums; a bundle must be a struct whose fields are themselves bundles",
            ));
        }
        Data::Union(data) => {
            return Err(syn::Error::new(
                data.union_token.span,
                "`Bundle` cannot be derived for unions; a bundle must be a struct whose fields are themselves bundles",
            ));
        }
    };

    let plan = plan_fields(&data.fields);

    let name = &input.ident;
    let (impl_generics, ty_generics, _) = input.generics.split_for_impl();
    let field_type_refs: Vec<&Type> = plan.types.iter().collect();
    let where_clause = common::bundle_where_clause(&input.generics, &field_type_refs);

    let types = &plan.types;
    let bindings = &plan.bindings;
    let pattern = &plan.pattern;

    // Unused parameters when the struct has no fields: name them with a leading
    // underscore to stay warning/clippy clean.
    let has_fields = !types.is_empty();
    let components_param = if has_fields {
        quote!(components)
    } else {
        quote!(_components)
    };
    let out_param = if has_fields {
        quote!(out)
    } else {
        quote!(_out)
    };
    let func_param = if has_fields {
        quote!(func)
    } else {
        quote!(_func)
    };

    // `get_components` body: destructure, then forward each field once. The
    // `unsafe` block is only emitted when there is at least one forwarding call
    // so an empty `unsafe {}` (which clippy flags as `unused_unsafe`) is never
    // produced.
    let get_components_body = if has_fields {
        quote! {
            #pattern
            // SAFETY: forwarded — each field is itself a `Bundle` and is moved
            // out of `self` exactly once by the destructure above, so its
            // `get_components` yields its values once and never double-drops.
            unsafe {
                #(
                    <#types as prism_ecs::bundle::Bundle>::get_components(#bindings, func);
                )*
            }
        }
    } else {
        TokenStream::new()
    };

    Ok(quote! {
        // SAFETY: `component_ids` and `get_components` iterate the same fields
        // in the same declaration order, and each field type upholds the
        // `Bundle` one-id/one-value contract, so the concatenation does too.
        #[automatically_derived]
        unsafe impl #impl_generics prism_ecs::bundle::Bundle for #name #ty_generics #where_clause {
            fn component_ids(
                #components_param: &mut prism_ecs::component::Components,
                #out_param: &mut Vec<prism_ecs::component::ComponentId>,
            ) {
                #(
                    <#types as prism_ecs::bundle::Bundle>::component_ids(components, out);
                )*
            }

            unsafe fn get_components(self, #func_param: &mut dyn FnMut(*mut u8)) {
                #get_components_body
            }
        }
    })
}

/// Compute the [`FieldPlan`] for a struct's fields, handling named, tuple, and
/// unit shapes.
fn plan_fields(fields: &Fields) -> FieldPlan {
    match fields {
        Fields::Named(named) => {
            let mut types = Vec::with_capacity(named.named.len());
            let mut bindings = Vec::with_capacity(named.named.len());
            for field in &named.named {
                types.push(field.ty.clone());
                // Named fields are destructured by their own identifier.
                bindings.push(field.ident.clone().expect("named field has an ident"));
            }
            let pattern = quote! { let Self { #(#bindings),* } = self; };
            FieldPlan {
                types,
                bindings,
                pattern,
            }
        }
        Fields::Unnamed(unnamed) => {
            let mut types = Vec::with_capacity(unnamed.unnamed.len());
            let mut bindings = Vec::with_capacity(unnamed.unnamed.len());
            for (i, field) in unnamed.unnamed.iter().enumerate() {
                types.push(field.ty.clone());
                bindings.push(format_ident!("__prism_bundle_field_{}", i));
            }
            let pattern = quote! { let Self( #(#bindings),* ) = self; };
            FieldPlan {
                types,
                bindings,
                pattern,
            }
        }
        Fields::Unit => FieldPlan {
            types: Vec::new(),
            bindings: Vec::new(),
            pattern: TokenStream::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn expand_str(input: DeriveInput) -> String {
        expand(&input).unwrap().to_string()
    }

    #[test]
    fn named_struct_forwards_in_order() {
        let di: DeriveInput = parse_quote! {
            struct Physics { pos: Position, vel: Velocity }
        };
        let out = expand_str(di);
        assert!(out.contains("unsafe impl prism_ecs :: bundle :: Bundle for Physics"));
        // component_ids forwards both field types in declaration order.
        let pos = out.find("Position").unwrap();
        let vel = out.find("Velocity").unwrap();
        assert!(pos < vel, "declaration order must be preserved");
        // Field bounds appear on the impl.
        assert!(out.contains("Position : prism_ecs :: bundle :: Bundle"));
        assert!(out.contains("Velocity : prism_ecs :: bundle :: Bundle"));
        // Destructures by field name.
        assert!(out.contains("let Self { pos , vel } = self"));
    }

    #[test]
    fn tuple_struct_uses_positional_bindings() {
        let di: DeriveInput = parse_quote! { struct Physics(Position, Velocity); };
        let out = expand_str(di);
        assert!(out.contains("let Self ("));
        assert!(out.contains("__prism_bundle_field_0"));
        assert!(out.contains("__prism_bundle_field_1"));
    }

    #[test]
    fn unit_struct_has_empty_bodies_and_underscored_params() {
        let di: DeriveInput = parse_quote! { struct Empty; };
        let out = expand_str(di);
        assert!(out.contains("_components"));
        assert!(out.contains("_out"));
        assert!(out.contains("_func"));
        // No destructure and no unsafe block for a field-less bundle.
        assert!(!out.contains("let Self"));
        assert!(!out.contains("unsafe {"));
    }

    #[test]
    fn generic_struct_threads_generics_and_bounds() {
        let di: DeriveInput = parse_quote! {
            struct Pair<A, B> { a: A, b: B }
        };
        let out = expand_str(di);
        assert!(
            out.contains("unsafe impl < A , B > prism_ecs :: bundle :: Bundle for Pair < A , B >")
        );
        assert!(out.contains("A : prism_ecs :: bundle :: Bundle"));
        assert!(out.contains("B : prism_ecs :: bundle :: Bundle"));
    }

    #[test]
    fn duplicate_field_types_bound_once() {
        let di: DeriveInput = parse_quote! { struct Two { a: Tracked, b: Tracked } };
        let out = expand_str(di);
        assert_eq!(
            out.matches("Tracked : prism_ecs :: bundle :: Bundle")
                .count(),
            1
        );
    }

    #[test]
    fn enum_is_rejected() {
        let di: DeriveInput = parse_quote! { enum E { A, B } };
        assert!(expand(&di).is_err());
    }

    #[test]
    fn union_is_rejected() {
        let di: DeriveInput = parse_quote! { union U { a: u32, b: f32 } };
        assert!(expand(&di).is_err());
    }
}
