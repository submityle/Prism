//! `prism_ui_component` — a composable component model for Loom.
//!
//! This crate layers a small, familiar component abstraction on top of the
//! data-only [`Element`](prism_ui::Element) trees produced by `prism_ui`. It
//! answers three recurring composition questions:
//!
//! * **How is a reusable unit of UI defined?** A [`Component`] turns a typed
//!   props value into an [`Element`](prism_ui::Element). Implement it on a
//!   struct, or wrap a closure in a [`FnComponent`]; either way
//!   [`mount_component`] renders it.
//! * **How do deeply nested components share services?** A [`ContextMap`]
//!   provides type-keyed dependency injection with lexical, override-local
//!   [`child`](ContextMap::child) scopes.
//! * **How does a parent hand structured children to a child?** [`Slots`]
//!   models named child lists plus a default slot, and [`SlottedComponent`]
//!   drops a chosen slot into a container element.
//!
//! [`ComponentCtx`] bundles a [`ContextMap`] and optional [`Slots`] into the
//! ambient environment a component renders against.
//!
//! # Example
//!
//! ```
//! use prism_ui::{Element, ElementKind};
//! use prism_ui_component::{ComponentCtx, ContextMap, render_with_context};
//!
//! // A service made available to the whole subtree.
//! struct Theme {
//!     accent: &'static str,
//! }
//!
//! let mut context = ContextMap::new();
//! context.provide(Theme { accent: "violet" });
//!
//! let ctx = ComponentCtx::new(&context);
//! let view = render_with_context(&ctx, |ctx| {
//!     let theme = ctx.inject::<Theme>().expect("theme provided");
//!     Element::box_().child(Element::text(theme.accent))
//! });
//!
//! assert_eq!(view.kind(), &ElementKind::Box);
//! assert_eq!(view.child_elements()[0].text_content(), Some("violet"));
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod component;
pub mod context;
pub mod lifecycle;
pub mod props;
pub mod slots;

pub use component::{render_with_context, ComponentCtx};
pub use context::ContextMap;
pub use lifecycle::{mount, LifecycleScope, Mounted};
pub use props::{mount_component, Component, FnComponent, Props};
pub use slots::{Slots, SlottedComponent};
