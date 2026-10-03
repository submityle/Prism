//! Derive macro for `prism_reflect`.
//!
//! `#[derive(Reflect)]` generates `Reflect` + the matching kind subtrait
//! (`Struct` | `TupleStruct` | `Enum`) + `Typed` + `GetTypeRegistration`
//! implementations with a cached `TypeInfo` for:
//!
//! - named-field structs (`Struct` kind),
//! - tuple structs (`TupleStruct` kind), and
//! - enums with unit, tuple, and struct variants (`Enum` kind).
//!
//! All generated code refers to the kernel through the absolute
//! `::prism_reflect::` path, so it works both inside the kernel crate (which
//! aliases `extern crate self as prism_reflect;`) and in downstream crates.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Index, parse_macro_input};

/// Derive `Reflect` (and its companion traits) for a struct, tuple struct, or
/// enum.
#[proc_macro_derive(Reflect)]
pub fn derive_reflect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = input.ident.clone();

    let expanded = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(_) => derive_named_struct(&ident, &data.fields),
            Fields::Unnamed(_) => derive_tuple_struct(&ident, &data.fields),
            Fields::Unit => {
                return syn::Error::new_spanned(
                    &input.ident,
                    "#[derive(Reflect)] does not support unit structs",
                )
                .to_compile_error()
                .into();
            }
        },
        Data::Enum(data) => derive_enum(&ident, data),
        Data::Union(_) => {
            return syn::Error::new_spanned(
                &input.ident,
                "#[derive(Reflect)] does not support unions",
            )
            .to_compile_error()
            .into();
        }
    };

    expanded.into()
}

/// Emit the `GetTypeRegistration` impl shared by every derived kind.
fn get_type_registration(ident: &syn::Ident) -> TokenStream2 {
    quote! {
        impl ::prism_reflect::GetTypeRegistration for #ident {
            fn get_type_registration() -> ::prism_reflect::TypeRegistration {
                ::prism_reflect::TypeRegistration::of::<#ident>()
            }
        }
    }
}

/// Generate impls for a named-field struct.
fn derive_named_struct(ident: &syn::Ident, fields: &Fields) -> TokenStream2 {
    let Fields::Named(named) = fields else {
        unreachable!("derive_named_struct called with non-named fields");
    };
    let names: Vec<_> = named
        .named
        .iter()
        .map(|f| f.ident.clone().expect("named field has an identifier"))
        .collect();
    let name_strs: Vec<String> = names.iter().map(ToString::to_string).collect();
    let types: Vec<_> = named.named.iter().map(|f| f.ty.clone()).collect();
    let count = names.len();
    let indices: Vec<usize> = (0..count).collect();
    let registration = get_type_registration(ident);

    quote! {
        impl ::prism_reflect::Reflect for #ident {
            fn type_name(&self) -> &'static str { ::core::any::type_name::<#ident>() }
            fn type_info(&self) -> &'static ::prism_reflect::TypeInfo {
                <#ident as ::prism_reflect::Typed>::type_info()
            }
            fn as_any(&self) -> &dyn ::core::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any { self }
            fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn ::core::any::Any> { self }
            fn as_reflect(&self) -> &dyn ::prism_reflect::Reflect { self }
            fn as_reflect_mut(&mut self) -> &mut dyn ::prism_reflect::Reflect { self }
            fn reflect_ref(&self) -> ::prism_reflect::ReflectRef<'_> {
                ::prism_reflect::ReflectRef::Struct(self)
            }
            fn reflect_mut(&mut self) -> ::prism_reflect::ReflectMut<'_> {
                ::prism_reflect::ReflectMut::Struct(self)
            }
        }

        impl ::prism_reflect::Struct for #ident {
            fn field(&self, name: &str) -> ::core::option::Option<&dyn ::prism_reflect::Reflect> {
                match name {
                    #( #name_strs => ::core::option::Option::Some(&self.#names as &dyn ::prism_reflect::Reflect), )*
                    _ => ::core::option::Option::None,
                }
            }
            fn field_mut(&mut self, name: &str) -> ::core::option::Option<&mut dyn ::prism_reflect::Reflect> {
                match name {
                    #( #name_strs => ::core::option::Option::Some(&mut self.#names as &mut dyn ::prism_reflect::Reflect), )*
                    _ => ::core::option::Option::None,
                }
            }
            fn field_at(&self, index: usize) -> ::core::option::Option<&dyn ::prism_reflect::Reflect> {
                match index {
                    #( #indices => ::core::option::Option::Some(&self.#names as &dyn ::prism_reflect::Reflect), )*
                    _ => ::core::option::Option::None,
                }
            }
            fn name_at(&self, index: usize) -> ::core::option::Option<&'static str> {
                match index {
                    #( #indices => ::core::option::Option::Some(#name_strs), )*
                    _ => ::core::option::Option::None,
                }
            }
            fn field_count(&self) -> usize { #count }
        }

        impl ::prism_reflect::Typed for #ident {
            fn type_info() -> &'static ::prism_reflect::TypeInfo {
                static CELL: ::std::sync::OnceLock<::prism_reflect::TypeInfo> =
                    ::std::sync::OnceLock::new();
                CELL.get_or_init(|| {
                    ::prism_reflect::TypeInfo::Struct(::prism_reflect::StructInfo::new(
                        ::core::any::type_name::<#ident>(),
                        ::std::vec![
                            #( ::prism_reflect::NamedField::new(
                                #name_strs,
                                ::core::any::type_name::<#types>(),
                            ), )*
                        ],
                    ))
                })
            }
        }

        #registration
    }
}

/// Generate impls for a tuple struct.
fn derive_tuple_struct(ident: &syn::Ident, fields: &Fields) -> TokenStream2 {
    let Fields::Unnamed(unnamed) = fields else {
        unreachable!("derive_tuple_struct called with non-unnamed fields");
    };
    let count = unnamed.unnamed.len();
    let types: Vec<_> = unnamed.unnamed.iter().map(|f| f.ty.clone()).collect();
    let indices: Vec<usize> = (0..count).collect();
    let tuple_indices: Vec<Index> = (0..count).map(Index::from).collect();
    let registration = get_type_registration(ident);

    quote! {
        impl ::prism_reflect::Reflect for #ident {
            fn type_name(&self) -> &'static str { ::core::any::type_name::<#ident>() }
            fn type_info(&self) -> &'static ::prism_reflect::TypeInfo {
                <#ident as ::prism_reflect::Typed>::type_info()
            }
            fn as_any(&self) -> &dyn ::core::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any { self }
            fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn ::core::any::Any> { self }
            fn as_reflect(&self) -> &dyn ::prism_reflect::Reflect { self }
            fn as_reflect_mut(&mut self) -> &mut dyn ::prism_reflect::Reflect { self }
            fn reflect_ref(&self) -> ::prism_reflect::ReflectRef<'_> {
                ::prism_reflect::ReflectRef::TupleStruct(self)
            }
            fn reflect_mut(&mut self) -> ::prism_reflect::ReflectMut<'_> {
                ::prism_reflect::ReflectMut::TupleStruct(self)
            }
        }

        impl ::prism_reflect::TupleStruct for #ident {
            fn field(&self, index: usize) -> ::core::option::Option<&dyn ::prism_reflect::Reflect> {
                match index {
                    #( #indices => ::core::option::Option::Some(&self.#tuple_indices as &dyn ::prism_reflect::Reflect), )*
                    _ => ::core::option::Option::None,
                }
            }
            fn field_mut(&mut self, index: usize) -> ::core::option::Option<&mut dyn ::prism_reflect::Reflect> {
                match index {
                    #( #indices => ::core::option::Option::Some(&mut self.#tuple_indices as &mut dyn ::prism_reflect::Reflect), )*
                    _ => ::core::option::Option::None,
                }
            }
            fn field_count(&self) -> usize { #count }
        }

        impl ::prism_reflect::Typed for #ident {
            fn type_info() -> &'static ::prism_reflect::TypeInfo {
                static CELL: ::std::sync::OnceLock<::prism_reflect::TypeInfo> =
                    ::std::sync::OnceLock::new();
                CELL.get_or_init(|| {
                    ::prism_reflect::TypeInfo::TupleStruct(::prism_reflect::TupleStructInfo::new(
                        ::core::any::type_name::<#ident>(),
                        ::std::vec![
                            #( ::prism_reflect::UnnamedField::new(
                                #indices,
                                ::core::any::type_name::<#types>(),
                            ), )*
                        ],
                    ))
                })
            }
        }

        #registration
    }
}

/// Generate impls for an enum (unit/tuple/struct variants).
fn derive_enum(ident: &syn::Ident, data: &syn::DataEnum) -> TokenStream2 {
    // Per-method match arms, assembled variant by variant.
    let mut name_arms = Vec::new();
    let mut index_arms = Vec::new();
    let mut type_arms = Vec::new();
    let mut field_arms = Vec::new();
    let mut field_mut_arms = Vec::new();
    let mut field_at_arms = Vec::new();
    let mut field_at_mut_arms = Vec::new();
    let mut count_arms = Vec::new();
    let mut variant_infos = Vec::new();

    for (vindex, variant) in data.variants.iter().enumerate() {
        let vident = &variant.ident;
        let vname = vident.to_string();

        name_arms.push(quote! { Self::#vident { .. } => #vname, });
        index_arms.push(quote! { Self::#vident { .. } => #vindex, });

        match &variant.fields {
            Fields::Unit => {
                type_arms.push(quote! { Self::#vident => ::prism_reflect::VariantType::Unit, });
                field_arms.push(quote! { Self::#vident => ::core::option::Option::None, });
                field_mut_arms.push(quote! { Self::#vident => ::core::option::Option::None, });
                field_at_arms.push(quote! { Self::#vident => ::core::option::Option::None, });
                field_at_mut_arms.push(quote! { Self::#vident => ::core::option::Option::None, });
                count_arms.push(quote! { Self::#vident => 0usize, });
                variant_infos.push(quote! {
                    ::prism_reflect::VariantInfo::new(
                        #vname,
                        #vindex,
                        ::prism_reflect::VariantKind::Unit,
                    )
                });
            }
            Fields::Unnamed(unnamed) => {
                let fcount = unnamed.unnamed.len();
                let binds: Vec<_> = (0..fcount).map(|i| format_ident!("__f{}", i)).collect();
                let idxs: Vec<usize> = (0..fcount).collect();
                let types: Vec<_> = unnamed.unnamed.iter().map(|f| f.ty.clone()).collect();

                type_arms.push(quote! { Self::#vident(..) => ::prism_reflect::VariantType::Tuple, });
                field_arms.push(quote! { Self::#vident(..) => ::core::option::Option::None, });
                field_mut_arms.push(quote! { Self::#vident(..) => ::core::option::Option::None, });
                field_at_arms.push(quote! {
                    Self::#vident( #( #binds ),* ) => match index {
                        #( #idxs => ::core::option::Option::Some(#binds as &dyn ::prism_reflect::Reflect), )*
                        _ => ::core::option::Option::None,
                    },
                });
                field_at_mut_arms.push(quote! {
                    Self::#vident( #( #binds ),* ) => match index {
                        #( #idxs => ::core::option::Option::Some(#binds as &mut dyn ::prism_reflect::Reflect), )*
                        _ => ::core::option::Option::None,
                    },
                });
                count_arms.push(quote! { Self::#vident(..) => #fcount, });
                variant_infos.push(quote! {
                    ::prism_reflect::VariantInfo::new(
                        #vname,
                        #vindex,
                        ::prism_reflect::VariantKind::Tuple(::std::vec![
                            #( ::prism_reflect::UnnamedField::new(
                                #idxs,
                                ::core::any::type_name::<#types>(),
                            ), )*
                        ]),
                    )
                });
            }
            Fields::Named(named) => {
                let fidents: Vec<_> = named
                    .named
                    .iter()
                    .map(|f| f.ident.clone().expect("named field has an identifier"))
                    .collect();
                let fnames: Vec<String> = fidents.iter().map(ToString::to_string).collect();
                let fidxs: Vec<usize> = (0..fidents.len()).collect();
                let fcount = fidents.len();
                let types: Vec<_> = named.named.iter().map(|f| f.ty.clone()).collect();

                type_arms.push(quote! { Self::#vident { .. } => ::prism_reflect::VariantType::Struct, });
                field_arms.push(quote! {
                    Self::#vident { #( #fidents ),* } => match name {
                        #( #fnames => ::core::option::Option::Some(#fidents as &dyn ::prism_reflect::Reflect), )*
                        _ => ::core::option::Option::None,
                    },
                });
                field_mut_arms.push(quote! {
                    Self::#vident { #( #fidents ),* } => match name {
                        #( #fnames => ::core::option::Option::Some(#fidents as &mut dyn ::prism_reflect::Reflect), )*
                        _ => ::core::option::Option::None,
                    },
                });
                field_at_arms.push(quote! {
                    Self::#vident { #( #fidents ),* } => match index {
                        #( #fidxs => ::core::option::Option::Some(#fidents as &dyn ::prism_reflect::Reflect), )*
                        _ => ::core::option::Option::None,
                    },
                });
                field_at_mut_arms.push(quote! {
                    Self::#vident { #( #fidents ),* } => match index {
                        #( #fidxs => ::core::option::Option::Some(#fidents as &mut dyn ::prism_reflect::Reflect), )*
                        _ => ::core::option::Option::None,
                    },
                });
                count_arms.push(quote! { Self::#vident { .. } => #fcount, });
                variant_infos.push(quote! {
                    ::prism_reflect::VariantInfo::new(
                        #vname,
                        #vindex,
                        ::prism_reflect::VariantKind::Struct(::std::vec![
                            #( ::prism_reflect::NamedField::new(
                                #fnames,
                                ::core::any::type_name::<#types>(),
                            ), )*
                        ]),
                    )
                });
            }
        }
    }

    let registration = get_type_registration(ident);

    quote! {
        impl ::prism_reflect::Reflect for #ident {
            fn type_name(&self) -> &'static str { ::core::any::type_name::<#ident>() }
            fn type_info(&self) -> &'static ::prism_reflect::TypeInfo {
                <#ident as ::prism_reflect::Typed>::type_info()
            }
            fn as_any(&self) -> &dyn ::core::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any { self }
            fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn ::core::any::Any> { self }
            fn as_reflect(&self) -> &dyn ::prism_reflect::Reflect { self }
            fn as_reflect_mut(&mut self) -> &mut dyn ::prism_reflect::Reflect { self }
            fn reflect_ref(&self) -> ::prism_reflect::ReflectRef<'_> {
                ::prism_reflect::ReflectRef::Enum(self)
            }
            fn reflect_mut(&mut self) -> ::prism_reflect::ReflectMut<'_> {
                ::prism_reflect::ReflectMut::Enum(self)
            }
        }

        impl ::prism_reflect::Enum for #ident {
            fn variant_name(&self) -> &'static str {
                match self { #( #name_arms )* }
            }
            fn variant_index(&self) -> usize {
                match self { #( #index_arms )* }
            }
            fn variant_type(&self) -> ::prism_reflect::VariantType {
                match self { #( #type_arms )* }
            }
            fn field(&self, name: &str) -> ::core::option::Option<&dyn ::prism_reflect::Reflect> {
                match self { #( #field_arms )* }
            }
            fn field_mut(&mut self, name: &str) -> ::core::option::Option<&mut dyn ::prism_reflect::Reflect> {
                match self { #( #field_mut_arms )* }
            }
            fn field_at(&self, index: usize) -> ::core::option::Option<&dyn ::prism_reflect::Reflect> {
                match self { #( #field_at_arms )* }
            }
            fn field_at_mut(&mut self, index: usize) -> ::core::option::Option<&mut dyn ::prism_reflect::Reflect> {
                match self { #( #field_at_mut_arms )* }
            }
            fn field_count(&self) -> usize {
                match self { #( #count_arms )* }
            }
        }

        impl ::prism_reflect::Typed for #ident {
            fn type_info() -> &'static ::prism_reflect::TypeInfo {
                static CELL: ::std::sync::OnceLock<::prism_reflect::TypeInfo> =
                    ::std::sync::OnceLock::new();
                CELL.get_or_init(|| {
                    ::prism_reflect::TypeInfo::Enum(::prism_reflect::EnumInfo::new(
                        ::core::any::type_name::<#ident>(),
                        ::std::vec![ #( #variant_infos, )* ],
                    ))
                })
            }
        }

        #registration
    }
}
