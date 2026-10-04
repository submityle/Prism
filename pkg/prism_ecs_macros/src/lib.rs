//! Derive macros for the [`prism_ecs`] kernel.
//!
//! This crate is the companion proc-macro crate referenced by `prism_ecs`'s
//! [`Component`] and [`Bundle`] traits. It provides two `#[derive(...)]` macros
//! that generate the trait `impl`s so that plain data types and groups of
//! bundles can be used directly with the ECS without hand-writing boilerplate.
//!
//! | Macro                  | Generated impl                              |
//! |------------------------|---------------------------------------------|
//! | [`Component`]          | `impl prism_ecs::component::Component`       |
//! | [`Bundle`]             | `unsafe impl prism_ecs::bundle::Bundle`      |
//!
//! All generated code refers to the target traits and types through the
//! **absolute path** `prism_ecs::...`, so the derives can be invoked from any
//! crate that has `prism_ecs` in its dependency graph without extra imports.
//!
//! # Provenance
//!
//! Engine-agnostic. Contains no Unreal Engine source or derived code and
//! depends on no `bevy_*` crate; the expansions are produced from standard,
//! publicly documented `syn`/`quote` proc-macro techniques.
//!
//! [`prism_ecs`]: https://docs.rs/prism_ecs
//! [`Component`]: macro@Component
//! [`Bundle`]: macro@Bundle

mod bundle;
mod common;
mod component;
mod systemset;

use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput};

/// Derive [`prism_ecs::component::Component`] for a `Send + Sync + 'static`
/// data type.
///
/// `Component` is a marker trait, so the generated `impl` is empty by default
/// and the trait's default associated const
/// (`STORAGE == StorageType::Table`) applies.
///
/// # Storage attribute
///
/// An optional `#[component(storage = "...")]` attribute selects the storage
/// strategy by emitting an explicit `const STORAGE` in the `impl`:
///
/// - *(absent)* — leave `STORAGE` unset; the trait default
///   [`StorageType::Table`] applies.
/// - `#[component(storage = "Table")]` — emit
///   `const STORAGE: StorageType = StorageType::Table;`.
/// - `#[component(storage = "SparseSet")]` — emit
///   `const STORAGE: StorageType = StorageType::SparseSet;`.
///
/// Any other value is a compile error.
///
/// Generics and `where`-clauses on the type are preserved.
///
/// # Examples
///
/// ```ignore
/// use prism_ecs_macros::Component;
///
/// #[derive(Component)]
/// struct Position { x: f32, y: f32 }
///
/// #[derive(Component)]
/// #[component(storage = "SparseSet")]
/// struct Selected;
///
/// #[derive(Component)]
/// struct Wrapper<T: Send + Sync + 'static>(T);
/// ```
///
/// [`StorageType::Table`]: prism_ecs::component::StorageType::Table
/// [`StorageType::SparseSet`]: prism_ecs::component::StorageType::SparseSet
#[proc_macro_derive(Component, attributes(component))]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    component::expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derive [`prism_ecs::bundle::Bundle`] for a struct whose every field type is
/// itself a `Bundle`.
///
/// Because every [`Component`](macro@Component) is a one-element bundle and
/// tuples of bundles are bundles, a struct of components (or of nested
/// bundles) is a natural composite bundle. The generated `unsafe impl`:
///
/// - [`component_ids`] calls `<FieldTy as Bundle>::component_ids` for every
///   field **in declaration order**.
/// - [`get_components`] destructures `self` and forwards each field to
///   `<FieldTy as Bundle>::get_components` in the **same order**, moving every
///   field out exactly once (never dropping it twice).
///
/// Named structs, tuple structs, and unit structs are all supported. Each
/// distinct field type is added as a `FieldTy: prism_ecs::bundle::Bundle`
/// bound on the generated `impl`, and the type's own generics/`where`-clause
/// are threaded through.
///
/// Deriving `Bundle` for an `enum` or `union` is a compile error: a bundle is
/// a fixed, ordered group of component values, which an enum's "one variant at
/// a time" shape cannot express.
///
/// # Examples
///
/// ```ignore
/// use prism_ecs_macros::{Bundle, Component};
///
/// #[derive(Component)] struct Position(f32, f32);
/// #[derive(Component)] struct Velocity(f32, f32);
///
/// #[derive(Bundle)]
/// struct Physics { pos: Position, vel: Velocity }
///
/// #[derive(Bundle)]
/// struct PhysicsTuple(Position, Velocity);
/// ```
///
/// [`component_ids`]: prism_ecs::bundle::Bundle::component_ids
/// [`get_components`]: prism_ecs::bundle::Bundle::get_components
#[proc_macro_derive(Bundle)]
pub fn derive_bundle(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    bundle::expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derive [`prism_ecs::schedule::SystemSet`] for a label type used to group and
/// order systems (design §8.2).
///
/// A `SystemSet` is a hashable, `'static` label; the generated impl reports a
/// stable [`SystemSetId`](prism_ecs::schedule::SystemSetId):
///
/// - A **struct** (unit, tuple, or named) is a single label: `set_id` returns
///   `SystemSetId::of::<Self>()`. The struct's field *values* do not affect its
///   identity.
/// - A **fieldless enum** yields one distinct set per variant: `set_id` matches
///   on `self` and returns `SystemSetId::with::<Self>(i)` for the `i`-th
///   variant, so `MySet::A` and `MySet::B` are independent ordering anchors.
///
/// Deriving for a `union`, or for an enum with any data-carrying variant, is a
/// compile error: a set label must have a finite, value-independent identity.
/// You will almost always pair this with
/// `#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]`.
///
/// # Examples
///
/// ```ignore
/// use prism_ecs_macros::SystemSet;
///
/// #[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// struct Physics;
///
/// #[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// enum SyncSet { Pull, Push }
/// ```
#[proc_macro_derive(SystemSet)]
pub fn derive_system_set(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    systemset::expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
