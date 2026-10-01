#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

//! Loom scoped styles and responsive `@media` resolution.
//!
//! This crate adds two component-oriented capabilities on top of
//! [`prism_ui_style`], the design-token/cascade layer of Prism's Loom UI:
//!
//! * **Component-scoped styles** ([`scope`]). A [`ScopeId`] gives each component
//!   a stable, collision-resistant identity (like Vue's `data-v-xxxxxxxx`
//!   attribute or a CSS-Modules hash). A [`Scope`] registers the local class
//!   names a component owns and then rewrites *only those* names — both in the
//!   component's [`StyleSheet`](prism_ui_style::StyleSheet) (producing a
//!   [`ScopedSheet`]) and across a [`prism_ui::Element`] subtree — so two
//!   components can reuse the class name `title` without clashing. Global and
//!   unknown classes are passed through untouched.
//!
//! * **Responsive breakpoint resolution** ([`media`]). A [`MediaResolver`]
//!   collapses a class's base and per-breakpoint overrides into the exact set
//!   of properties that are live at a given viewport width, using Tailwind's
//!   mobile-first cascade (a wider viewport inherits every narrower
//!   breakpoint's overrides).
//!
//! # Example
//!
//! ```
//! use prism_ui::Element;
//! use prism_ui_scoped::{MediaResolver, Scope};
//! use prism_ui_style::{Breakpoint, Class, StyleProp, StyleSheet, StyleValue};
//!
//! // A component owns a `title` class: 14px by default, 24px from `md` up.
//! let sheet = StyleSheet::new().with_class(
//!     Class::new("title")
//!         .with(StyleProp::FontSize, StyleValue::px(14.0))
//!         .with_breakpoint(Breakpoint::Md, StyleProp::FontSize, StyleValue::px(24.0)),
//! );
//!
//! // Scope the sheet so `title` becomes unique to this component.
//! let scope = Scope::from_name("Hero").with_local("title");
//! let scoped = scope.scope(&sheet);
//! let scoped_name = scoped.scoped_name("title").unwrap().to_string();
//! assert_ne!(scoped_name, "title");
//!
//! // The same rewrite applies to the view tree, so references stay consistent.
//! let view = Element::box_().child(Element::text("Hi").class("title"));
//! let scoped_view = scope.apply(&view);
//! assert_eq!(scoped_view.child_elements()[0].class_names(), &[scoped_name.clone()]);
//!
//! // Below `md` the base size is live; at/above `md` the override wins.
//! let class = scoped.sheet().get(&scoped_name).unwrap();
//! let narrow = MediaResolver::new(640.0).active_props(class);
//! let wide = MediaResolver::new(768.0).active_props(class);
//! assert_eq!(narrow.get(&StyleProp::FontSize), Some(&StyleValue::px(14.0)));
//! assert_eq!(wide.get(&StyleProp::FontSize), Some(&StyleValue::px(24.0)));
//! ```

extern crate alloc;

pub mod media;
pub mod scope;
pub mod scoped_sheet;

pub use media::{MediaResolver, ResolvedSheet};
pub use scope::{Scope, ScopeAllocator, ScopeId};
pub use scoped_sheet::ScopedSheet;
