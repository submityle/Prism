//! Parsing of the `#[reflect(...)]` helper attribute and `#[doc]` comments into
//! runtime [`TypeMetadata`] builder tokens (design §24.7).
//!
//! `#[derive(Reflect)]` declares `attributes(reflect)`, so a reflected type and
//! its named fields may carry `#[reflect(...)]` hints plus ordinary `///` doc
//! comments. This module reads those at expansion time and emits a
//! `::prism_reflect::schema::TypeMetadata::new()....` construction expression
//! that `GetTypeRegistration` inserts, so the metadata is attached to the type's
//! [`TypeRegistration`](prism_reflect::TypeRegistration) automatically on
//! `register::<T>()` — no manual `register_type_data` call required.
//!
//! Supported field-level `#[reflect(...)]` keys:
//!
//! - `rename = "name"` — editor/serde display name (stored as custom `"rename"`).
//! - `docs = "..."` / `tooltip = "..."` — documentation text (accumulated).
//! - `category = "Group"` — inspector grouping label.
//! - `range(min..=max)` / `clamp(min..=max)` — inclusive numeric range used by
//!   [`schema::validate`](prism_reflect::schema::validate) and
//!   [`schema::clamp`](prism_reflect::schema::clamp).
//! - `on_violation = "clamp"` or `on_violation = "reject"` — settings range
//!   behavior consumed by Prism's reflected settings registry.
//! - `scope`, `override_scope`, `precedence`, and comma-separated `domains` —
//!   canonical settings source and visibility policies.
//! - `project_overridable`, `bootstrap`, and `execution_class` — settings
//!   resolver flags (boolean, bare or `= bool`).
//! - `max_string_bytes` and `max_array_len` — canonical value size limits.
//! - `aliases = "old.key,older.key"` — deprecated persisted keys that resolve
//!   to the canonical setting key.
//! - `default = <lit>` — default-value attribute.
//! - `readonly` / `hidden` / `required` / `setting` — boolean flags (bare or
//!   `= bool`). `setting` explicitly opts a field into settings-schema
//!   derivation; reflection alone does not make a field a setting.
//! - `skip` — stored as custom `"skip"` (serde/inspector skip hint).
//!
//! Supported type-level `#[reflect(...)]` keys: `docs = "..."`, plus any other
//! `key = <lit>` recorded as a type-level custom attribute. Type-level and
//! field-level `///` doc comments are folded into the respective `docs` slots.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DeriveInput, Expr, ExprLit, ExprUnary, Fields, Lit, UnOp};

/// A literal attribute value destined for `schema::AttributeValue`.
enum AttrLit {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Text(String),
}

impl AttrLit {
    /// Emit the `schema::AttributeValue` construction expression.
    fn tokens(&self) -> TokenStream2 {
        match self {
            Self::Bool(b) => quote! { ::prism_reflect::schema::AttributeValue::Bool(#b) },
            Self::Int(i) => quote! { ::prism_reflect::schema::AttributeValue::Int(#i) },
            Self::UInt(i) => quote! { ::prism_reflect::schema::AttributeValue::UInt(#i) },
            Self::Float(f) => quote! { ::prism_reflect::schema::AttributeValue::Float(#f) },
            Self::Text(s) => quote! {
                ::prism_reflect::schema::AttributeValue::Text(::std::string::String::from(#s))
            },
        }
    }
}

/// Accumulated field-level metadata parsed from attributes.
#[derive(Default)]
struct FieldMeta {
    rename: Option<String>,
    docs: Vec<String>,
    category: Option<String>,
    readonly: bool,
    hidden: bool,
    setting: bool,
    required: bool,
    skip: bool,
    range: Option<(f64, f64)>,
    default: Option<AttrLit>,
    custom: Vec<(String, AttrLit)>,
}

impl FieldMeta {
    /// Whether any metadata was recorded for this field.
    fn is_empty(&self) -> bool {
        self.rename.is_none()
            && self.docs.is_empty()
            && self.category.is_none()
            && !self.readonly
            && !self.hidden
            && !self.setting
            && !self.required
            && !self.skip
            && self.range.is_none()
            && self.default.is_none()
            && self.custom.is_empty()
    }

    /// Emit the `schema::FieldMetadata::new(name)....` builder, if non-empty.
    fn tokens(&self, field_name: &str) -> Option<TokenStream2> {
        if self.is_empty() {
            return None;
        }
        let mut ts = quote! { ::prism_reflect::schema::FieldMetadata::new(#field_name) };
        if !self.docs.is_empty() {
            let joined = self.docs.join("\n");
            ts = quote! { #ts.with_docs(#joined) };
        }
        if let Some(category) = &self.category {
            ts = quote! { #ts.with_category(#category) };
        }
        if self.readonly {
            ts = quote! { #ts.readonly(true) };
        }
        if self.hidden {
            ts = quote! { #ts.hidden(true) };
        }
        if self.setting {
            ts = quote! { #ts.setting(true) };
        }
        if self.required {
            ts = quote! { #ts.required(true) };
        }
        if let Some((min, max)) = self.range {
            ts = quote! { #ts.with_range(#min, #max) };
        }
        if let Some(default) = &self.default {
            let value = default.tokens();
            ts = quote! { #ts.with_default(#value) };
        }
        if let Some(rename) = &self.rename {
            ts = quote! {
                #ts.with_custom(
                    "rename",
                    ::prism_reflect::schema::AttributeValue::Text(
                        ::std::string::String::from(#rename),
                    ),
                )
            };
        }
        if self.skip {
            ts = quote! {
                #ts.with_custom("skip", ::prism_reflect::schema::AttributeValue::Bool(true))
            };
        }
        for (key, value) in &self.custom {
            let value = value.tokens();
            ts = quote! { #ts.with_custom(#key, #value) };
        }
        Some(ts)
    }
}

/// Accumulated type-level metadata parsed from attributes.
#[derive(Default)]
struct TypeMeta {
    docs: Vec<String>,
    custom: Vec<(String, AttrLit)>,
}

impl TypeMeta {
    fn is_empty(&self) -> bool {
        self.docs.is_empty() && self.custom.is_empty()
    }
}

/// Build the `schema::TypeMetadata` construction expression for `input`, or
/// `None` when the type and its fields carry no metadata-bearing attributes.
///
/// Named-field structs contribute per-field metadata; tuple structs, unit
/// structs, and enums contribute type-level docs/custom only (they have no
/// named fields for the schema metadata model to key on).
pub fn type_metadata_tokens(input: &DeriveInput) -> syn::Result<Option<TokenStream2>> {
    let mut type_meta = TypeMeta::default();
    collect_doc(&input.attrs, &mut type_meta.docs);
    for attr in &input.attrs {
        if attr.path().is_ident("reflect") {
            parse_type_reflect(attr, &mut type_meta)?;
        }
    }

    let mut field_entries: Vec<TokenStream2> = Vec::new();
    if let Data::Struct(data) = &input.data
        && let Fields::Named(named) = &data.fields
    {
        for field in &named.named {
            let field_name = field
                .ident
                .as_ref()
                .expect("named field has an identifier")
                .to_string();
            let mut field_meta = FieldMeta::default();
            collect_doc(&field.attrs, &mut field_meta.docs);
            for attr in &field.attrs {
                if attr.path().is_ident("reflect") {
                    parse_field_reflect(attr, &mut field_meta)?;
                }
            }
            if let Some(tokens) = field_meta.tokens(&field_name) {
                field_entries.push(tokens);
            }
        }
    }

    if type_meta.is_empty() && field_entries.is_empty() {
        return Ok(None);
    }

    let mut ts = quote! { ::prism_reflect::schema::TypeMetadata::new() };
    if !type_meta.docs.is_empty() {
        let joined = type_meta.docs.join("\n");
        ts = quote! { #ts.with_docs(#joined) };
    }
    for (key, value) in &type_meta.custom {
        let value = value.tokens();
        ts = quote! { #ts.with_custom(#key, #value) };
    }
    for entry in field_entries {
        ts = quote! { #ts.with_field(#entry) };
    }
    Ok(Some(ts))
}

/// Fold `#[doc = "..."]` comments into `out` (trimmed, one entry per line).
fn collect_doc(attrs: &[syn::Attribute], out: &mut Vec<String>) {
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(nv) = &attr.meta
            && let Expr::Lit(ExprLit {
                lit: Lit::Str(text),
                ..
            }) = &nv.value
        {
            out.push(text.value().trim().to_string());
        }
    }
}

/// Parse a type-level `#[reflect(...)]` attribute into `tm`.
fn parse_type_reflect(attr: &syn::Attribute, tm: &mut TypeMeta) -> syn::Result<()> {
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("docs") {
            let text: syn::LitStr = meta.value()?.parse()?;
            tm.docs.push(text.value());
            return Ok(());
        }
        let key = meta
            .path
            .get_ident()
            .ok_or_else(|| meta.error("expected an identifier in #[reflect(...)]"))?
            .to_string();
        let lit: Lit = meta.value()?.parse()?;
        tm.custom.push((key, lit_to_attr(&lit)?));
        Ok(())
    })
}

/// Parse a field-level `#[reflect(...)]` attribute into `fm`.
fn parse_field_reflect(attr: &syn::Attribute, fm: &mut FieldMeta) -> syn::Result<()> {
    attr.parse_nested_meta(|meta| {
        let path = &meta.path;
        if path.is_ident("rename") {
            let text: syn::LitStr = meta.value()?.parse()?;
            fm.rename = Some(text.value());
        } else if path.is_ident("docs") || path.is_ident("tooltip") {
            let text: syn::LitStr = meta.value()?.parse()?;
            fm.docs.push(text.value());
        } else if path.is_ident("category") {
            let text: syn::LitStr = meta.value()?.parse()?;
            fm.category = Some(text.value());
        } else if path.is_ident("default") {
            let expr: Expr = meta.value()?.parse()?;
            fm.default = Some(expr_to_attr(&expr)?);
        } else if path.is_ident("skip") {
            fm.skip = parse_flag(&meta)?;
        } else if path.is_ident("readonly") {
            fm.readonly = parse_flag(&meta)?;
        } else if path.is_ident("hidden") {
            fm.hidden = parse_flag(&meta)?;
        } else if path.is_ident("setting") {
            fm.setting = parse_flag(&meta)?;
        } else if path.is_ident("required") {
            fm.required = parse_flag(&meta)?;
        } else if path.is_ident("on_violation") {
            let text: syn::LitStr = meta.value()?.parse()?;
            match text.value().as_str() {
                "clamp" | "reject" => {}
                _ => return Err(meta.error("on_violation must be \"clamp\" or \"reject\"")),
            }
            fm.custom
                .push(("on_violation".to_owned(), AttrLit::Text(text.value())));
        } else if path.is_ident("scope")
            || path.is_ident("override_scope")
            || path.is_ident("precedence")
            || path.is_ident("domains")
            || path.is_ident("aliases")
        {
            let text: syn::LitStr = meta.value()?.parse()?;
            fm.custom.push((
                path.get_ident().unwrap().to_string(),
                AttrLit::Text(text.value()),
            ));
        } else if path.is_ident("project_overridable")
            || path.is_ident("bootstrap")
            || path.is_ident("execution_class")
        {
            let value = parse_flag(&meta)?;
            fm.custom
                .push((path.get_ident().unwrap().to_string(), AttrLit::Bool(value)));
        } else if path.is_ident("max_string_bytes") || path.is_ident("max_array_len") {
            let value: syn::LitInt = meta.value()?.parse()?;
            let value = value.base10_parse::<u64>()?;
            fm.custom
                .push((path.get_ident().unwrap().to_string(), AttrLit::UInt(value)));
        } else if path.is_ident("clamp") || path.is_ident("range") {
            let content;
            syn::parenthesized!(content in meta.input);
            let range: syn::ExprRange = content.parse()?;
            fm.range = Some(range_to_bounds(&range)?);
        } else {
            return Err(meta.error("unknown #[reflect(...)] field attribute"));
        }
        Ok(())
    })
}

/// Parse a boolean flag that is either bare (`flag`) or `flag = <bool>`.
fn parse_flag(meta: &syn::meta::ParseNestedMeta) -> syn::Result<bool> {
    if meta.input.peek(syn::Token![=]) {
        let value: syn::LitBool = meta.value()?.parse()?;
        Ok(value.value)
    } else {
        Ok(true)
    }
}

/// Convert an attribute literal into an [`AttrLit`].
fn lit_to_attr(lit: &Lit) -> syn::Result<AttrLit> {
    match lit {
        Lit::Bool(b) => Ok(AttrLit::Bool(b.value)),
        Lit::Int(i) => match i.base10_parse::<i64>() {
            Ok(value) => Ok(AttrLit::Int(value)),
            Err(_) => Ok(AttrLit::UInt(i.base10_parse()?)),
        },
        Lit::Float(f) => {
            let value = f.base10_parse::<f64>()?;
            if !value.is_finite() {
                return Err(syn::Error::new_spanned(f, "default must be finite"));
            }
            Ok(AttrLit::Float(value))
        }
        Lit::Str(s) => Ok(AttrLit::Text(s.value())),
        other => Err(syn::Error::new_spanned(
            other,
            "unsupported literal in #[reflect(...)]: expected bool, integer, float, or string",
        )),
    }
}

/// Convert a default-value expression, accepting unary-negative numeric
/// literals in addition to the ordinary literal forms.
fn expr_to_attr(expr: &Expr) -> syn::Result<AttrLit> {
    match expr {
        Expr::Lit(ExprLit { lit, .. }) => lit_to_attr(lit),
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => {
            match expr_to_attr(expr)? {
                AttrLit::Int(value) => value.checked_neg().map(AttrLit::Int).ok_or_else(|| {
                    syn::Error::new_spanned(expr, "negative integer default is outside i64")
                }),
                AttrLit::UInt(value) => i64::try_from(-i128::from(value))
                    .map(AttrLit::Int)
                    .map_err(|_| {
                        syn::Error::new_spanned(expr, "negative integer default is outside i64")
                    }),
                AttrLit::Float(value) => Ok(AttrLit::Float(-value)),
                _ => Err(syn::Error::new_spanned(
                    expr,
                    "unary minus requires a numeric default literal",
                )),
            }
        }
        Expr::Group(group) => expr_to_attr(&group.expr),
        Expr::Paren(paren) => expr_to_attr(&paren.expr),
        other => Err(syn::Error::new_spanned(
            other,
            "unsupported expression in #[reflect(default = ...)]: expected a literal",
        )),
    }
}

/// Extract the inclusive `(min, max)` bounds from a `min..=max` range literal.
fn range_to_bounds(range: &syn::ExprRange) -> syn::Result<(f64, f64)> {
    let start = range
        .start
        .as_deref()
        .ok_or_else(|| syn::Error::new_spanned(range, "range needs a lower bound"))?;
    let end = range
        .end
        .as_deref()
        .ok_or_else(|| syn::Error::new_spanned(range, "range needs an upper bound"))?;
    Ok((expr_to_f64(start)?, expr_to_f64(end)?))
}

/// Evaluate a numeric-literal expression (allowing a unary minus) to `f64`.
fn expr_to_f64(expr: &Expr) -> syn::Result<f64> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Float(f), ..
        }) => f.base10_parse::<f64>(),
        Expr::Lit(ExprLit {
            lit: Lit::Int(i), ..
        }) => i.base10_parse::<f64>(),
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => Ok(-expr_to_f64(expr)?),
        Expr::Group(group) => expr_to_f64(&group.expr),
        Expr::Paren(paren) => expr_to_f64(&paren.expr),
        other => Err(syn::Error::new_spanned(
            other,
            "expected a numeric literal in range bound",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{expr_to_attr, AttrLit};

    #[test]
    fn numeric_default_limits_preserve_the_metadata_kind() {
        let signed = syn::parse_str("-9223372036854775808").unwrap();
        assert!(matches!(
            expr_to_attr(&signed).unwrap(),
            AttrLit::Int(i64::MIN)
        ));

        let unsigned = syn::parse_str("18446744073709551615").unwrap();
        assert!(matches!(
            expr_to_attr(&unsigned).unwrap(),
            AttrLit::UInt(u64::MAX)
        ));
    }

    #[test]
    fn invalid_defaults_return_errors_instead_of_panicking() {
        for source in [
            "-9223372036854775809",
            "18446744073709551616",
            "1e999",
            "-true",
            "some_function()",
        ] {
            let expr = syn::parse_str(source).unwrap();
            assert!(
                expr_to_attr(&expr).is_err(),
                "accepted invalid default: {source}"
            );
        }
    }
}
