//! `prism_ui_a11y` — an accessibility layer for Loom ([`prism_ui`]) views.
//!
//! This crate turns raw [`Element`](prism_ui::Element) trees into a structured,
//! assistive-technology-friendly model: semantic [`Role`]s, accessible
//! [`Label`]s (including `labelled-by` references), dynamic [`AriaState`], a
//! computed sequential [`FocusOrder`], role-aware [`KeyboardNav`], buffered
//! [`LiveRegion`] announcements, and deterministic
//! [`screen_reader_text`] composition.
//!
//! The design mirrors the web platform's accessibility model closely enough to
//! be familiar, while staying `no_std`-friendly and allocation-aware.
//!
//! # Workflow
//!
//! 1. Build an [`A11yTree`] of [`A11yNode`]s (one per element [`Key`](prism_ui::Key)).
//! 2. Compute a [`FocusOrder`] to drive Tab navigation.
//! 3. Feed [`KeyInput`] through [`KeyboardNav`] to get a [`NavAction`].
//! 4. Speak nodes with [`screen_reader_text`] and surface updates through a
//!    [`LiveRegion`].
//!
//! # Example
//!
//! ```
//! use prism_ui::Key;
//! use prism_ui_a11y::{
//!     A11yNode, A11yTree, AriaState, FocusOrder, KeyInput, KeyboardNav, Label,
//!     NavAction, Role, screen_reader_text,
//! };
//!
//! // A tiny form: a checkbox labelled by a separate text node, then a button.
//! let mut tree = A11yTree::new();
//! tree.insert(
//!     A11yNode::builder(Key::Str("accept".into()), Role::Checkbox)
//!         .label(Label::labelled_by(Key::Str("accept-label".into())))
//!         .state(AriaState::new().checked(false).required(true))
//!         .build(),
//! );
//! tree.insert(
//!     A11yNode::builder(Key::Str("accept-label".into()), Role::Presentation)
//!         .label(Label::text("Accept terms"))
//!         .build(),
//! );
//! tree.insert(
//!     A11yNode::builder(Key::Str("submit".into()), Role::Button)
//!         .label(Label::text("Submit"))
//!         .build(),
//! );
//!
//! // Screen-reader output resolves the label and spells out state.
//! assert_eq!(
//!     screen_reader_text(&tree, &Key::Str("accept".into())),
//!     "checkbox, Accept terms, not checked, required",
//! );
//!
//! // Focus order lists both interactive controls (the label is presentational).
//! let focus = FocusOrder::compute(&tree);
//! assert_eq!(
//!     focus.keys(),
//!     [Key::Str("accept".into()), Key::Str("submit".into())],
//! );
//!
//! // Tab from the checkbox moves to the next stop; Space toggles it.
//! let nav = KeyboardNav::new();
//! assert_eq!(
//!     nav.resolve(&tree, &Key::Str("accept".into()), KeyInput::Space),
//!     NavAction::Toggle,
//! );
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod derive;
pub mod focus;
pub mod keyboard;
pub mod label;
pub mod live;
pub mod node;
pub mod role;
pub mod sr;
pub mod state;
pub mod tree;

pub use derive::{accessible_text, element_key, label_from_element};
pub use focus::FocusOrder;
pub use keyboard::{Arrow, FocusMove, KeyInput, KeyboardNav, NavAction};
pub use label::Label;
pub use live::{LiveRegion, Politeness};
pub use node::{A11yNode, A11yNodeBuilder};
pub use role::Role;
pub use sr::{describe_node, screen_reader_text};
pub use state::AriaState;
pub use tree::A11yTree;
