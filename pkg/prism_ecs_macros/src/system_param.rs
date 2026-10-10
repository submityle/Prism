//! Expansion logic for `#[derive(SystemParam)]`.
//!
//! The derive turns a `struct` whose fields are themselves [`SystemParam`]s
//! into a single composite `SystemParam`, so a system can take one named
//! bundle of parameters instead of a long positional tuple:
//!
//! ```ignore
//! #[derive(SystemParam)]
//! struct PhysicsCtx<'w, 's> {
//!     clock: Res<'w, Clock>,
//!     score: ResMut<'w, Score>,
//!     scratch: Local<'s, Vec<Entity>>,
//!     commands: Commands<'w, 's>,
//! }
//!
//! fn step(ctx: PhysicsCtx) { /* ctx.clock, ctx.score, … */ }
//! ```
//!
//! # How the lifetimes map
//!
//! The generated impl generalises the manual tuple impl in
//! `prism_ecs::system::param`. Each field type `Fi` is written with the
//! struct's own `'w`/`'s`, and the GAT machinery does the lifetime folding for
//! free:
//!
//! * `<Fi as SystemParam>::State` is always `Send + Sync + 'static`, so the
//!   `State` tuple is `'static` even though the field types mention the impl
//!   lifetimes `'w`/`'s`.
//! * `type Item<'wp, 's p>` is the struct re-parameterised so that `'w ↦ 'wp`
//!   and `'s ↦ 'sp` (type/const generics pass through unchanged); each field
//!   is rebuilt with `<Fi as SystemParam>::get_param`, which re-stamps the
//!   borrow with the method lifetimes.
//!
//! # Lifetime contract
//!
//! A derived `SystemParam` struct may declare at most the two lifetimes `'w`
//! (world borrow) and `'s` (per-system state borrow), in that order, named
//! exactly `w` and `s`. Any other lifetime name, a reversed order, or more than
//! these two is a compile error with a pointed message. Type and const generics
//! are unrestricted and are threaded through verbatim.
//!
//! Enums and unions are rejected: a composite param must be a `struct` whose
//! fields are each a `SystemParam`.

use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{Data, DeriveInput, Fields, GenericParam, Ident, Type};

/// Expand a `#[derive(SystemParam)]` input into an
/// `unsafe impl prism_ecs::system::SystemParam` block.
pub fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let data = match &input.data {
        Data::Struct(data) => data,
        Data::Enum(data) => {
            return Err(syn::Error::new(
                data.enum_token.span,
                "`SystemParam` cannot be derived for enums; a system param must be a struct \
                 whose fields are themselves system params",
            ));
        }
        Data::Union(data) => {
            return Err(syn::Error::new(
                data.union_token.span,
                "`SystemParam` cannot be derived for unions; a system param must be a struct \
                 whose fields are themselves system params",
            ));
        }
    };

    validate_lifetimes(input)?;

    let name = &input.ident;
    let (impl_generics, ty_generics, _) = input.generics.split_for_impl();

    // The `Item<'wp, 'sp>` associated type: the struct re-parameterised with the
    // method lifetimes substituted for the struct's own `'w`/`'s`. Built by
    // hand because `ty_generics` would reuse the struct's `'w`/`'s`, which are
    // the impl generics, not the GAT's `'wp`/`'sp`.
    let item_ty = build_item_type(input, name);

    // Ordered field types plus the constructor expression for `get_param`.
    let (field_types, construct) = plan_fields(name, &data.fields);
    let has_fields = !field_types.is_empty();

    // State-tuple destructure bindings, one per field, in declaration order.
    let bindings: Vec<Ident> = (0..field_types.len())
        .map(|i| format_ident!("__prism_sysparam_state_{i}"))
        .collect();

    let where_tokens = system_param_where_clause(input, &field_types);

    // Method bodies. When the struct has no fields the `State` tuple is `()`,
    // so there is nothing to destructure or forward; the top-level
    // `#[allow(unused_variables)]` keeps the untouched `state`/`world` params
    // warning-clean.
    let init_body = quote! {
        ( #( <#field_types as prism_ecs::system::SystemParam>::init_state(world), )* )
    };

    let update_access_body = if has_fields {
        quote! {
            let ( #( #bindings, )* ) = state;
            #( <#field_types as prism_ecs::system::SystemParam>::update_access(#bindings, access); )*
        }
    } else {
        TokenStream::new()
    };

    let get_param_body = if has_fields {
        quote! {
            let ( #( #bindings, )* ) = state;
            #construct
        }
    } else {
        construct
    };

    let apply_body = if has_fields {
        quote! {
            let ( #( #bindings, )* ) = state;
            #( <#field_types as prism_ecs::system::SystemParam>::apply(#bindings, world); )*
        }
    } else {
        TokenStream::new()
    };

    Ok(quote! {
        // SAFETY: each field is itself a `SystemParam` that declares its own
        // access in `update_access` and builds its own disjoint item in
        // `get_param`; borrowing disjoint fields of the `State` tuple hands each
        // field a non-aliasing `&mut` to its own state. Any genuine overlap
        // between two fields is rejected by the per-system conflict analysis
        // (the panicking access adders), exactly as for the tuple impl this
        // generalises.
        #[automatically_derived]
        #[allow(unused_variables)]
        unsafe impl #impl_generics prism_ecs::system::SystemParam for #name #ty_generics #where_tokens {
            type State = ( #( <#field_types as prism_ecs::system::SystemParam>::State, )* );
            type Item<'wp, 'sp> = #item_ty;

            #[inline]
            fn init_state(world: &mut prism_ecs::world::World) -> Self::State {
                #init_body
            }

            #[inline]
            fn update_access(state: &Self::State, access: &mut prism_ecs::query::Access) {
                #update_access_body
            }

            #[inline]
            #[allow(clippy::undocumented_unsafe_blocks)]
            unsafe fn get_param<'wp, 'sp>(
                state: &'sp mut Self::State,
                world: prism_ecs::system::UnsafeWorldCell<'wp>,
            ) -> Self::Item<'wp, 'sp> {
                #get_param_body
            }

            #[inline]
            fn apply(state: &mut Self::State, world: &mut prism_ecs::world::World) {
                #apply_body
            }
        }
    })
}

/// Validate that the struct's lifetimes are a prefix of `['w, 's]`, named
/// exactly `w`/`s`, in that order. Type and const generics are unrestricted.
fn validate_lifetimes(input: &DeriveInput) -> syn::Result<()> {
    let mut seen_w = false;
    let mut seen_s = false;
    for param in &input.generics.params {
        if let GenericParam::Lifetime(def) = param {
            let name = def.lifetime.ident.to_string();
            match name.as_str() {
                "w" => {
                    if seen_s {
                        return Err(syn::Error::new_spanned(
                            &def.lifetime,
                            "`SystemParam` derive requires lifetimes in the order `<'w, 's>`; \
                             `'w` must come before `'s`",
                        ));
                    }
                    seen_w = true;
                }
                "s" => {
                    seen_s = true;
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        &def.lifetime,
                        format!(
                            "`SystemParam` derive only allows the lifetimes `'w` (world) and \
                             `'s` (system state); found `'{other}`"
                        ),
                    ));
                }
            }
        }
    }
    let _ = seen_w;
    Ok(())
}

/// Build the `Item<'wp, 'sp>` type: `#name` re-parameterised with `'w ↦ 'wp`,
/// `'s ↦ 'sp`, and all type/const generics passed through unchanged. Returns a
/// bare `#name` when the struct has no generics.
fn build_item_type(input: &DeriveInput, name: &Ident) -> TokenStream {
    let mut args: Vec<TokenStream> = Vec::new();
    for param in &input.generics.params {
        match param {
            GenericParam::Lifetime(def) => {
                let mapped = if def.lifetime.ident == "w" {
                    quote!('wp)
                } else {
                    quote!('sp)
                };
                args.push(mapped);
            }
            GenericParam::Type(ty) => {
                let ident = &ty.ident;
                args.push(quote!(#ident));
            }
            GenericParam::Const(c) => {
                let ident = &c.ident;
                args.push(quote!(#ident));
            }
        }
    }
    if args.is_empty() {
        quote!(#name)
    } else {
        quote!(#name < #( #args ),* >)
    }
}

/// Compute the ordered field types and the `get_param` constructor expression
/// for the three struct shapes (named / tuple / unit). The constructor rebuilds
/// each field from its destructured state binding via the field type's own
/// `SystemParam::get_param`.
fn plan_fields(name: &Ident, fields: &Fields) -> (Vec<Type>, TokenStream) {
    match fields {
        Fields::Named(named) => {
            let mut types = Vec::with_capacity(named.named.len());
            let mut inits = Vec::with_capacity(named.named.len());
            for (i, field) in named.named.iter().enumerate() {
                let ty = field.ty.clone();
                let ident = field.ident.clone().expect("named field has an ident");
                let binding = format_ident!("__prism_sysparam_state_{i}");
                inits.push(quote! {
                    #ident: unsafe {
                        <#ty as prism_ecs::system::SystemParam>::get_param(#binding, world)
                    }
                });
                types.push(ty);
            }
            (types, quote! { #name { #( #inits ),* } })
        }
        Fields::Unnamed(unnamed) => {
            let mut types = Vec::with_capacity(unnamed.unnamed.len());
            let mut inits = Vec::with_capacity(unnamed.unnamed.len());
            for (i, field) in unnamed.unnamed.iter().enumerate() {
                let ty = field.ty.clone();
                let binding = format_ident!("__prism_sysparam_state_{i}");
                inits.push(quote! {
                    unsafe {
                        <#ty as prism_ecs::system::SystemParam>::get_param(#binding, world)
                    }
                });
                types.push(ty);
            }
            (types, quote! { #name ( #( #inits ),* ) })
        }
        Fields::Unit => (Vec::new(), quote! { #name }),
    }
}

/// Build the impl's `where`-clause: the struct's existing predicates plus one
/// `Fi: prism_ecs::system::SystemParam` bound per *distinct* field type that
/// mentions one of the struct's generic type/const parameters.
///
/// Concrete field types (only lifetimes + concrete type names, e.g.
/// `Res<'w, Clock>` or `Commands<'w, 's>`) deliberately get **no** explicit
/// bound: the compiler proves `Fi: SystemParam` straight from the field type's
/// own impl, and — crucially on edition-2024 toolchains (see rust-lang/rust
/// issue #152409) — an explicit where-bound on a concrete type *shadows* that
/// impl's associated-type definition, which stops `<Fi as SystemParam>::Item`
/// from normalising to the field's concrete `Item`. Emitting the bound only
/// for generic-dependent fields keeps `get_param`'s reconstruction well-typed
/// while still deferring the requirement for genuinely generic fields.
fn system_param_where_clause(input: &DeriveInput, field_types: &[Type]) -> TokenStream {
    let mut predicates: Vec<TokenStream> = Vec::new();

    if let Some(existing) = input.generics.where_clause.as_ref() {
        for pred in &existing.predicates {
            predicates.push(quote!(#pred));
        }
    }

    // Idents of the struct's generic type/const params. A field type that names
    // one of these cannot be proven `SystemParam` from a concrete impl, so it
    // keeps an explicit (non-shadowing) deferred bound.
    let generic_idents: Vec<String> = input
        .generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(ty) => Some(ty.ident.to_string()),
            GenericParam::Const(c) => Some(c.ident.to_string()),
            GenericParam::Lifetime(_) => None,
        })
        .collect();

    let mut seen: Vec<String> = Vec::new();
    for ty in field_types {
        if !type_mentions_any_ident(ty, &generic_idents) {
            continue;
        }
        let key = ty.to_token_stream().to_string();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        predicates.push(quote!(#ty: prism_ecs::system::SystemParam));
    }

    if predicates.is_empty() {
        TokenStream::new()
    } else {
        quote!(where #( #predicates ),*)
    }
}

/// Return `true` if the type's token stream references any of the given
/// identifiers (the struct's generic type/const parameters), walking into
/// nested groups such as `Local<'s, T>` or `Query<'w, 's, &'static T>`.
fn type_mentions_any_ident(ty: &Type, idents: &[String]) -> bool {
    fn walk(stream: TokenStream, idents: &[String]) -> bool {
        stream.into_iter().any(|tree| match tree {
            proc_macro2::TokenTree::Ident(ident) => {
                let name = ident.to_string();
                idents.contains(&name)
            }
            proc_macro2::TokenTree::Group(group) => walk(group.stream(), idents),
            _ => false,
        })
    }
    if idents.is_empty() {
        return false;
    }
    walk(ty.to_token_stream(), idents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn expand_str(input: DeriveInput) -> String {
        expand(&input).unwrap().to_string()
    }

    #[test]
    fn named_struct_builds_composite_impl() {
        let di: DeriveInput = parse_quote! {
            struct PhysicsCtx<'w, 's> {
                clock: Res<'w, Clock>,
                score: ResMut<'w, Score>,
                scratch: Local<'s, u32>,
                commands: Commands<'w, 's>,
            }
        };
        let out = expand_str(di);
        // Composite unsafe impl with both lifetimes threaded through.
        assert!(out.contains(
            "unsafe impl < 'w , 's > prism_ecs :: system :: SystemParam for PhysicsCtx < 'w , 's >"
        ));
        // State is the tuple of each field's State.
        assert!(out.contains("type State = ("));
        assert!(out.contains("as prism_ecs :: system :: SystemParam > :: State"));
        // Item re-stamps the struct with the GAT lifetimes `'wp`/`'sp`.
        assert!(out.contains("type Item < 'wp , 'sp > = PhysicsCtx < 'wp , 'sp >"));
        // get_param reconstructs each named field through its own get_param.
        assert!(out.contains("clock : unsafe"));
        assert!(out.contains("commands : unsafe"));
        // Concrete field types get *no* explicit `SystemParam` where-bound: the
        // compiler proves them from each field's own impl, and emitting the
        // bound would shadow that impl's `Item` definition (see
        // `system_param_where_clause`). An all-concrete struct therefore has no
        // generated `where` clause at all.
        assert!(!out.contains("Res < 'w , Clock > : prism_ecs :: system :: SystemParam"));
        assert!(!out.contains("Commands < 'w , 's > : prism_ecs :: system :: SystemParam"));
        assert!(!out.contains(" where "));
    }

    #[test]
    fn tuple_struct_uses_positional_constructor() {
        let di: DeriveInput = parse_quote! {
            struct Pair<'w>(Res<'w, A>, ResMut<'w, B>);
        };
        let out = expand_str(di);
        assert!(out.contains("for Pair < 'w >"));
        // Positional reconstruction `Pair( unsafe { .. }, unsafe { .. } )`.
        assert!(out.contains("Pair ("));
        assert!(out.contains("__prism_sysparam_state_0"));
        assert!(out.contains("__prism_sysparam_state_1"));
        // Only `'w` is mapped (no `'s`): Item is `Pair<'wp>`.
        assert!(out.contains("type Item < 'wp , 'sp > = Pair < 'wp >"));
    }

    #[test]
    fn unit_struct_has_empty_state_and_bodies() {
        let di: DeriveInput = parse_quote! { struct Noop; };
        let out = expand_str(di);
        assert!(out.contains("for Noop"));
        // Empty state tuple, Item is the bare name, constructor is just `Noop`.
        assert!(out.contains("type State = ()"));
        assert!(out.contains("type Item < 'wp , 'sp > = Noop"));
        // No destructure / no field forwarding.
        assert!(!out.contains("__prism_sysparam_state_0"));
    }

    #[test]
    fn type_and_const_generics_thread_through() {
        let di: DeriveInput = parse_quote! {
            struct Buffered<'s, T: Send + Sync + 'static, const N: usize> {
                scratch: Local<'s, T>,
            }
        };
        let out = expand_str(di);
        // Type and const generics are passed through unchanged into Item.
        assert!(out.contains("type Item < 'wp , 'sp > = Buffered < 'sp , T , N >"));
        // A field that mentions the generic `T` keeps a deferred bound, since
        // it cannot be proven `SystemParam` from a concrete impl.
        assert!(out.contains("Local < 's , T > : prism_ecs :: system :: SystemParam"));
    }

    #[test]
    fn enum_is_rejected() {
        let di: DeriveInput = parse_quote! { enum Bad { A, B } };
        let err = expand(&di).unwrap_err().to_string();
        assert!(err.contains("cannot be derived for enums"));
    }

    #[test]
    fn union_is_rejected() {
        let di: DeriveInput = parse_quote! { union Bad { a: u32, b: f32 } };
        let err = expand(&di).unwrap_err().to_string();
        assert!(err.contains("cannot be derived for unions"));
    }

    #[test]
    fn disallowed_lifetime_name_is_rejected() {
        let di: DeriveInput = parse_quote! {
            struct Bad<'a> { r: Res<'a, A> }
        };
        let err = expand(&di).unwrap_err().to_string();
        assert!(err.contains("only allows the lifetimes"));
    }

    #[test]
    fn reversed_lifetime_order_is_rejected() {
        let di: DeriveInput = parse_quote! {
            struct Bad<'s, 'w> { r: Res<'w, A>, l: Local<'s, u32> }
        };
        let err = expand(&di).unwrap_err().to_string();
        assert!(err.contains("must come before"));
    }
}
