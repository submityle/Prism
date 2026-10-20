//! Validates derive inputs and emits ordered, borrow-only dependency walks.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Field, Fields, GenericArgument, LitStr, Member, Path,
    PathArguments, Type,
};

#[derive(Default)]
struct Settings {
    type_name: Option<LitStr>,
    crate_path: Option<Path>,
}

fn settings(input: &DeriveInput) -> syn::Result<Settings> {
    let mut settings = Settings::default();
    for attribute in &input.attrs {
        if attribute.path().is_ident("dependency") {
            return Err(syn::Error::new_spanned(
                attribute,
                "#[dependency] belongs on a field",
            ));
        }
        if !attribute.path().is_ident("asset") {
            continue;
        }
        require_nonempty(attribute)?;
        attribute.parse_nested_meta(|meta| {
            if meta.path.is_ident("type_name") {
                if settings.type_name.is_some() {
                    return Err(meta.error("duplicate asset type_name"));
                }
                let name: LitStr = meta.value()?.parse()?;
                if name.value().trim().is_empty() {
                    return Err(syn::Error::new_spanned(
                        name,
                        "asset type_name must not be empty",
                    ));
                }
                settings.type_name = Some(name);
                Ok(())
            } else if meta.path.is_ident("crate") {
                if settings.crate_path.is_some() {
                    return Err(meta.error("duplicate asset crate path"));
                }
                let path: LitStr = meta.value()?.parse()?;
                settings.crate_path = Some(path.parse()?);
                Ok(())
            } else {
                Err(meta.error(
                    "expected asset(type_name = \"...\") or asset(crate = \"...\") on a type",
                ))
            }
        })?;
    }
    Ok(settings)
}

pub(crate) fn crate_override(input: &DeriveInput) -> syn::Result<Option<Path>> {
    Ok(settings(input)?.crate_path)
}

fn require_nonempty(attribute: &Attribute) -> syn::Result<()> {
    let list = attribute.meta.require_list()?;
    if list.tokens.is_empty() {
        return Err(syn::Error::new_spanned(
            attribute,
            "asset attribute must not be empty",
        ));
    }
    Ok(())
}

enum Plan {
    Leaf(Type),
    Optional(Box<Plan>),
    Sequence(Box<Plan>),
    Deref(Box<Plan>),
    Tuple(Vec<(usize, Plan)>),
}

impl Plan {
    fn emit(&self, value: TokenStream, path: &TokenStream, serial: &mut usize) -> TokenStream {
        let binding = format_ident!("__prism_asset_element_{}", *serial);
        *serial += 1;
        match self {
            Self::Leaf(_) => quote! {
                #path::VisitAssetDependencies::visit_asset_dependencies(#value, __prism_asset_visit);
            },
            Self::Optional(inner) => {
                let body = inner.emit(quote!(#binding), path, serial);
                quote! {
                    if let ::core::option::Option::Some(#binding) = #value { #body }
                }
            }
            Self::Sequence(inner) => {
                let body = inner.emit(quote!(#binding), path, serial);
                quote! { for #binding in #value { #body } }
            }
            Self::Deref(inner) => inner.emit(quote!(&**(#value)), path, serial),
            Self::Tuple(elements) => {
                let bodies = elements.iter().map(|(index, inner)| {
                    let index = syn::Index::from(*index);
                    inner.emit(quote!(&(#value).#index), path, serial)
                });
                quote!(#(#bodies)*)
            }
        }
    }

    fn bounds(&self, types: &mut Vec<Type>) {
        match self {
            Self::Leaf(ty) => types.push(ty.clone()),
            Self::Optional(inner) | Self::Sequence(inner) | Self::Deref(inner) => {
                inner.bounds(types);
            }
            Self::Tuple(elements) => elements.iter().for_each(|(_, plan)| plan.bounds(types)),
        }
    }
}

fn contains_handle(ty: &Type) -> bool {
    match ty {
        Type::Path(path) => path.qself.as_ref().is_some_and(|qself| contains_handle(&qself.ty)) || path.path.segments.iter().any(|segment| {
            matches!(segment.ident.to_string().as_str(), "Handle" | "SoftHandle" | "UntypedHandle")
                || match &segment.arguments {
                    PathArguments::AngleBracketed(arguments) => arguments.args.iter().any(|argument| {
                        matches!(argument, GenericArgument::Type(inner) if contains_handle(inner))
                    }),
                    _ => false,
                }
        }),
        Type::Array(array) => contains_handle(&array.elem),
        Type::Slice(slice) => contains_handle(&slice.elem),
        Type::Reference(reference) => contains_handle(&reference.elem),
        Type::Ptr(pointer) => contains_handle(&pointer.elem),
        Type::BareFn(function) => function.inputs.iter().any(|argument| contains_handle(&argument.ty))
            || matches!(&function.output, syn::ReturnType::Type(_, ty) if contains_handle(ty)),
        Type::Tuple(tuple) => tuple.elems.iter().any(contains_handle),
        Type::Paren(paren) => contains_handle(&paren.elem),
        Type::Group(group) => contains_handle(&group.elem),
        _ => false,
    }
}

fn automatic_plan(ty: &Type) -> syn::Result<Option<Plan>> {
    let plan = match ty {
        Type::Path(path) => {
            let Some(segment) = path.path.segments.last() else {
                return Ok(None);
            };
            match segment.ident.to_string().as_str() {
                "Handle" | "SoftHandle" | "UntypedHandle" => Some(Plan::Leaf(ty.clone())),
                "Option" | "Vec" | "Box" | "Arc" if path.qself.is_none() => {
                    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
                        return Ok(None);
                    };
                    let mut types = arguments.args.iter().filter_map(|argument| match argument {
                        GenericArgument::Type(inner) => Some(inner),
                        _ => None,
                    });
                    let Some(inner) = types.next() else {
                        return Ok(None);
                    };
                    // Allocator/custom arities need explicit trait-based traversal;
                    // never guess which of multiple type arguments owns dependencies.
                    if types.next().is_some() {
                        return unsupported(ty);
                    }
                    automatic_plan(inner)?.map(|inner| match segment.ident.to_string().as_str() {
                        "Option" => Plan::Optional(Box::new(inner)),
                        "Vec" => Plan::Sequence(Box::new(inner)),
                        _ => Plan::Deref(Box::new(inner)),
                    })
                }
                "WeakHandle" | "UntypedWeakHandle" | "PhantomData" => None,
                _ => return unsupported(ty),
            }
        }
        Type::Array(array) => {
            automatic_plan(&array.elem)?.map(|plan| Plan::Sequence(Box::new(plan)))
        }
        Type::Slice(slice) => {
            automatic_plan(&slice.elem)?.map(|plan| Plan::Sequence(Box::new(plan)))
        }
        Type::Reference(reference) => {
            automatic_plan(&reference.elem)?.map(|plan| Plan::Deref(Box::new(plan)))
        }
        Type::Tuple(tuple) => {
            let mut elements = Vec::new();
            for (index, ty) in tuple.elems.iter().enumerate() {
                if let Some(plan) = automatic_plan(ty)? {
                    elements.push((index, plan));
                }
            }
            (!elements.is_empty()).then_some(Plan::Tuple(elements))
        }
        Type::Paren(paren) => return automatic_plan(&paren.elem),
        Type::Group(group) => return automatic_plan(&group.elem),
        _ => return unsupported(ty),
    };
    Ok(plan)
}

fn unsupported(ty: &Type) -> syn::Result<Option<Plan>> {
    if contains_handle(ty) {
        Err(syn::Error::new_spanned(ty, "unsupported dependency container; implement VisitAssetDependencies and mark #[dependency], or explicitly #[asset(skip)]"))
    } else {
        Ok(None)
    }
}

fn field_plan(field: &Field) -> syn::Result<Option<Plan>> {
    let mut dependency = false;
    let mut skip = false;
    for attribute in &field.attrs {
        if attribute.path().is_ident("dependency") {
            if dependency || !matches!(attribute.meta, syn::Meta::Path(_)) {
                return Err(syn::Error::new_spanned(
                    attribute,
                    "expected one bare #[dependency] attribute",
                ));
            }
            dependency = true;
        } else if attribute.path().is_ident("asset") {
            require_nonempty(attribute)?;
            attribute.parse_nested_meta(|meta| {
                if !meta.path.is_ident("skip") || skip {
                    return Err(meta.error("expected one #[asset(skip)] attribute on a field"));
                }
                skip = true;
                Ok(())
            })?;
        }
    }
    if dependency && skip {
        return Err(syn::Error::new_spanned(
            field,
            "#[dependency] conflicts with #[asset(skip)]",
        ));
    }
    if skip {
        Ok(None)
    } else if dependency {
        Ok(Some(Plan::Leaf(field.ty.clone())))
    } else {
        automatic_plan(&field.ty)
    }
}

fn validate_variant(attributes: &[Attribute]) -> syn::Result<()> {
    for attribute in attributes {
        if attribute.path().is_ident("asset") || attribute.path().is_ident("dependency") {
            return Err(syn::Error::new_spanned(
                attribute,
                "asset traversal attributes belong on the type or its fields, not a variant",
            ));
        }
    }
    Ok(())
}

fn field_bodies(
    fields: &Fields,
    path: &TokenStream,
    enum_fields: bool,
    bounds: &mut Vec<Type>,
    serial: &mut usize,
) -> syn::Result<(Vec<TokenStream>, Vec<TokenStream>)> {
    let mut patterns = Vec::new();
    let mut bodies = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        let member = field
            .ident
            .clone()
            .map_or_else(|| Member::from(index), Member::Named);
        if let Some(plan) = field_plan(field)? {
            plan.bounds(bounds);
            let value = if enum_fields {
                let binding = format_ident!("__prism_asset_field_{index}");
                patterns.push(quote!(#member: #binding));
                quote!(#binding)
            } else {
                quote!(&self.#member)
            };
            bodies.push(plan.emit(value, path, serial));
        }
    }
    Ok((patterns, bodies))
}

pub(crate) fn expand(
    input: &DeriveInput,
    path: &TokenStream,
    asset: bool,
) -> syn::Result<TokenStream> {
    let settings = settings(input)?;
    if asset && !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(&input.generics, "Asset derive requires a concrete type without type, const, or lifetime parameters; use a concrete wrapper with a distinct stable identity (generic records can derive VisitAssetDependencies)"));
    }
    if !asset && settings.type_name.is_some() {
        return Err(syn::Error::new_spanned(
            input,
            "type_name is only valid for Asset derive",
        ));
    }
    let mut bounds = Vec::new();
    let mut serial = 0;
    let body = match &input.data {
        Data::Struct(data) => {
            let (_, bodies) = field_bodies(&data.fields, path, false, &mut bounds, &mut serial)?;
            quote!(#(#bodies)*)
        }
        Data::Enum(data) => {
            let mut arms = Vec::new();
            for variant in &data.variants {
                validate_variant(&variant.attrs)?;
                let name = &variant.ident;
                let (patterns, bodies) = field_bodies(&variant.fields, path, true, &mut bounds, &mut serial)?;
                arms.push(quote!(Self::#name { #(#patterns,)* .. } => { #(#bodies)* }));
            }
            if arms.is_empty() {
                quote!(match *self {})
            } else {
                quote!(match self { #(#arms,)* })
            }
        }
        Data::Union(data) => return Err(syn::Error::new_spanned(data.union_token, "asset dependency derives do not support unions: active fields cannot be visited safely")),
    };
    let name = &input.ident;
    let mut generics = input.generics.clone();
    // Recursive concrete records must not acquire a `Self: Visitor` bound;
    // Rust checks their leaf calls against the implementations being emitted.
    if !generics.params.is_empty() {
        for ty in bounds {
            generics
                .make_where_clause()
                .predicates
                .push(syn::parse_quote!(#ty: #path::VisitAssetDependencies));
        }
    }
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    let visitor = quote! {
        impl #impl_generics #path::VisitAssetDependencies for #name #type_generics #where_clause {
            fn visit_asset_dependencies(&self, __prism_asset_visit: &mut dyn ::core::ops::FnMut(#path::AssetDependency)) {
                #body
            }
        }
    };
    if !asset {
        return Ok(visitor);
    }
    let type_name = settings.type_name.map_or_else(
        || {
            quote!(::core::concat!(
                ::core::module_path!(),
                "::",
                ::core::stringify!(#name)
            ))
        },
        |name| quote!(#name),
    );
    Ok(quote! {
        #visitor
        impl #path::Asset for #name {
            const TYPE_NAME: &'static str = #type_name;
            fn asset_type() -> #path::AssetTypeId {
                #path::AssetTypeId::of::<Self>()
            }
            fn visit_dependencies(&self, __prism_asset_visit: &mut dyn ::core::ops::FnMut(#path::UntypedAssetId)) {
                #path::VisitAssetDependencies::visit_asset_dependencies(self, &mut |dependency| {
                    if let #path::AssetDependency::Strong(id) = dependency { __prism_asset_visit(id); }
                });
            }
            fn visit_references(&self, __prism_asset_visit: &mut dyn ::core::ops::FnMut(#path::AssetDependency)) {
                #path::VisitAssetDependencies::visit_asset_dependencies(self, __prism_asset_visit);
            }
        }
    })
}
