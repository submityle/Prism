//! Derive macro for `prism_reflect`.
//!
//! `#[derive(Reflect)]` generates `Reflect` + (`Struct` | `TupleStruct`) +
//! `Typed` implementations with a cached `TypeInfo` for named-field structs and
//! tuple structs. Enums and generics are handled in a later milestone.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Index, parse_macro_input};

/// Derive `Reflect` for a struct or tuple struct.
#[proc_macro_derive(Reflect)]
pub fn derive_reflect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = input.ident.clone();

    let data = match &input.data {
        Data::Struct(data) => data,
        _ => {
            return syn::Error::new_spanned(
                &input.ident,
                "#[derive(Reflect)] (M0) supports only structs and tuple structs",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = match &data.fields {
        Fields::Named(named) => {
            let names: Vec<_> = named
                .named
                .iter()
                .map(|f| f.ident.clone().unwrap())
                .collect();
            let name_strs: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            let types: Vec<_> = named.named.iter().map(|f| f.ty.clone()).collect();
            let count = names.len();
            let indices: Vec<usize> = (0..count).collect();

            quote! {
                impl ::prism_reflect::Reflect for #ident {
                    fn type_name(&self) -> &'static str { ::core::any::type_name::<#ident>() }
                    fn type_info(&self) -> &'static ::prism_reflect::TypeInfo {
                        <#ident as ::prism_reflect::Typed>::type_info()
                    }
                    fn as_any(&self) -> &dyn ::core::any::Any { self }
                    fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any { self }
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
            }
        }
        Fields::Unnamed(unnamed) => {
            let count = unnamed.unnamed.len();
            let types: Vec<_> = unnamed.unnamed.iter().map(|f| f.ty.clone()).collect();
            let indices: Vec<usize> = (0..count).collect();
            let tuple_indices: Vec<Index> = (0..count).map(Index::from).collect();

            quote! {
                impl ::prism_reflect::Reflect for #ident {
                    fn type_name(&self) -> &'static str { ::core::any::type_name::<#ident>() }
                    fn type_info(&self) -> &'static ::prism_reflect::TypeInfo {
                        <#ident as ::prism_reflect::Typed>::type_info()
                    }
                    fn as_any(&self) -> &dyn ::core::any::Any { self }
                    fn as_any_mut(&mut self) -> &mut dyn ::core::any::Any { self }
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
            }
        }
        Fields::Unit => {
            return syn::Error::new_spanned(
                &input.ident,
                "#[derive(Reflect)] (M0) does not support unit structs",
            )
            .to_compile_error()
            .into();
        }
    };

    expanded.into()
}
