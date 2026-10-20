//! Expansion validation: rejected input must fail before emitting a partial impl.

#[path = "../../src/expand.rs"]
mod expand;

use quote::quote;
use syn::{parse_quote, DeriveInput};

/// Check diagnostic text without depending on compiler-version-specific spans.
fn rejected(input: DeriveInput, asset: bool, diagnostic: &str) {
    let error = expand::expand(&input, &quote!(::runtime), asset).unwrap_err();
    assert!(error.to_string().contains(diagnostic), "{error}");
}

/// Persisted identity must not collapse distinct monomorphizations or lifetimes.
#[test]
fn generic_assets_have_an_explicit_diagnostic() {
    for input in [
        parse_quote!(
            struct Generic<T>(T);
        ),
        parse_quote!(
            struct Borrowed<'a>(&'a str);
        ),
        parse_quote!(
            struct Fixed<const N: usize>([u8; N]);
        ),
    ] {
        rejected(input, true, "requires a concrete type");
    }
}

/// An active union field cannot be determined safely by generated traversal.
#[test]
fn unions_are_rejected_by_both_derives() {
    for asset in [false, true] {
        rejected(
            parse_quote!(union Data { a: u64, b: f64 }),
            asset,
            "do not support unions",
        );
    }
}

/// Misspellings and conflicts must not silently suppress a dependency edge.
#[test]
fn invalid_attributes_are_rejected() {
    for (input, message) in [
        (
            parse_quote!(
                #[asset()]
                struct A;
            ),
            "must not be empty",
        ),
        (
            parse_quote!(
                #[asset(type_name = " ")]
                struct A;
            ),
            "must not be empty",
        ),
        (
            parse_quote!(
                #[asset(type_name = "a", type_name = "b")]
                struct A;
            ),
            "duplicate asset type_name",
        ),
        (
            parse_quote!(
                #[asset(crate = "runtime", crate = "other")]
                struct A;
            ),
            "duplicate asset crate",
        ),
        (
            parse_quote!(
                #[asset(unknown)]
                struct A;
            ),
            "expected asset(type_name",
        ),
        (
            parse_quote!(
                #[dependency]
                struct A;
            ),
            "belongs on a field",
        ),
        (
            parse_quote!(
                struct A {
                    #[asset(skip)]
                    #[dependency]
                    value: Handle<Image>,
                }
            ),
            "conflicts",
        ),
        (
            parse_quote!(
                struct A {
                    #[asset(skip, skip)]
                    value: Handle<Image>,
                }
            ),
            "expected one",
        ),
        (
            parse_quote!(
                struct A {
                    #[dependency()]
                    value: Handle<Image>,
                }
            ),
            "bare #[dependency]",
        ),
        (
            parse_quote!(
                struct A {
                    #[dependency]
                    #[dependency]
                    value: Handle<Image>,
                }
            ),
            "bare #[dependency]",
        ),
        (
            parse_quote!(
                struct A {
                    #[asset(omit)]
                    value: Handle<Image>,
                }
            ),
            "expected one",
        ),
        (
            parse_quote!(
                enum A {
                    #[asset(skip)]
                    Ignored,
                }
            ),
            "not a variant",
        ),
    ] {
        rejected(input, true, message);
    }
    rejected(
        parse_quote!(
            #[asset(type_name = "a")]
            struct A;
        ),
        false,
        "only valid for Asset",
    );
    let invalid: DeriveInput = parse_quote!(
        #[asset(crate = "not a path")]
        struct A;
    );
    assert!(expand::crate_override(&invalid).is_err());
}

/// Unknown containers visibly containing handles require a deliberate decision.
#[test]
fn unknown_containers_do_not_drop_visible_handles() {
    for input in [
        parse_quote!(
            struct A(HashMap<String, Handle<Image>>);
        ),
        parse_quote!(
            struct A(Option<Vec<Result<SoftHandle<Image>, Error>>>);
        ),
        parse_quote!(
            struct A(Vec<Handle<Image>, Allocator>);
        ),
        parse_quote!(
            struct A(*const Handle<Image>);
        ),
        parse_quote!(
            struct A(fn() -> Handle<Image>);
        ),
    ] {
        rejected(input, true, "unsupported dependency container");
    }
}

/// All supported data forms must produce syntactically complete impls.
#[test]
fn supported_expansions_parse_as_rust_items() {
    for input in [
        parse_quote!(
            struct Empty;
        ),
        parse_quote!(
            enum Never {}
        ),
        parse_quote!(
            struct Tuple(Handle<Image>, u8, Option<Vec<SoftHandle<Image>>>);
        ),
        parse_quote!(
            enum Choice {
                Unit,
                Tuple(Handle<Image>, u8),
                Named { value: Option<SoftHandle<Image>> },
            }
        ),
        parse_quote!(
            struct Nested {
                a: Vec<Option<Vec<Handle<Image>>>>,
                b: (u32, [SoftHandle<Image>; 2]),
                c: Box<Arc<Handle<Image>>>,
            }
        ),
        parse_quote!(
            struct Recursive {
                #[dependency]
                next: Option<Box<Recursive>>,
            }
        ),
        parse_quote!(
            struct Deliberate {
                #[asset(skip)]
                ignored: HashMap<u32, Handle<Image>>,
                #[dependency]
                alias: TextureAlias,
            }
        ),
    ] {
        let tokens = expand::expand(&input, &quote!(::runtime), true).unwrap();
        syn::parse2::<syn::File>(tokens).unwrap();
    }
}

/// Generic helpers retain their caller's bounds and add only visited-type bounds.
#[test]
fn helper_generics_and_crate_override_are_preserved() {
    let input = parse_quote!(
        #[asset(crate = "facade::assets")]
        struct Group<'a, T, const N: usize>
        where
            T: Clone,
        {
            #[dependency]
            values: &'a [T; N],
            ignored: core::marker::PhantomData<T>,
        }
    );
    let path = expand::crate_override(&input).unwrap().unwrap();
    assert_eq!(quote!(#path).to_string(), "facade :: assets");
    let output = expand::expand(&input, &quote!(facade::assets), false).unwrap();
    let generated = output.to_string();
    assert!(generated.contains("T : Clone"));
    assert!(generated.contains("VisitAssetDependencies"));
    assert!(!generated.contains("std ::"));
    syn::parse2::<syn::File>(output).unwrap();
}
