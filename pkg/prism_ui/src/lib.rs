//! `prism_ui` — **Loom**, Prism's declarative UI/scene notation.
//!
//! Loom is a retained-mode view system in the spirit of `SolidJS`, `SwiftUI`
//! and Flutter, redesigned for a game engine's performance envelope. It is
//! engine-agnostic: the runtime computes *what changed* and emits a minimal
//! stream of ops to a pluggable [`Backend`], so the same view code can target a
//! headless test harness or a GPU renderer.
//!
//! # Architecture
//!
//! Loom is layered; each layer is an independent crate and can be used alone:
//!
//! * [`reactive`] — a glitch-free signal/memo/effect graph.
//! * [`tree`] — a generational arena, retained tree and keyed reconciler.
//! * [`style`] — design tokens, classes, selectors and the cascade.
//! * [`layout`] — a pure-Rust flexbox solver.
//! * [`anim`] — easings, springs, timelines and transitions.
//! * [`loom!`] — a declarative macro DSL that lowers to [`Element`] builders.
//!
//! This umbrella crate wires them together behind one [`Ui`] runtime driven by
//! cheap, data-only [`Element`] trees.
//!
//! # The core contract
//!
//! **Cost is proportional to change, not scene size.** Building an `Element`
//! tree every frame is cheap; the runtime reconciles it against retained state
//! and only tells the backend about real deltas.
//!
//! # Example
//!
//! ```
//! use prism_ui::{Element, RecordingBackend, Ui};
//! use prism_ui::layout::{AvailableSpace, Size};
//!
//! // Build a tiny view: a column with one text child.
//! let view = Element::box_().child(Element::text("hello"));
//!
//! let mut ui = Ui::new(RecordingBackend::new());
//! ui.mount(&view);
//! ui.compute_layout(Size::new(
//!     AvailableSpace::Definite(800.0),
//!     AvailableSpace::Definite(600.0),
//! ));
//!
//! // Two nodes were materialised (the box and the text).
//! assert_eq!(ui.node_count(), 2);
//!
//! // Re-rendering an identical tree produces no new creates.
//! let before = ui.backend().len();
//! ui.update(&view);
//! assert_eq!(ui.backend().len(), before);
//! ```
//!
//! # The `loom!` macro
//!
//! The same tree can be written with the [`loom!`] DSL, which lowers to the
//! very same [`Element`] builder calls at compile time:
//!
//! ```
//! use prism_ui::{loom, ElementKind};
//!
//! let view = loom! {
//!     box {
//!         class: "card";
//!         text("hello");
//!     }
//! };
//!
//! assert_eq!(view.kind(), &ElementKind::Box);
//! assert_eq!(view.class_names(), &["card".to_string()]);
//! assert_eq!(view.child_elements()[0].text_content(), Some("hello"));
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod backend;
pub mod element;
pub mod paint;
pub mod reactive_view;
pub mod style_map;
pub mod ui;

pub use backend::{Backend, BackendId, BackendOp, RecordingBackend};
pub use element::{Element, ElementKind, Key};
pub use paint::PaintStyle;
pub use reactive_view::ReactiveView;
pub use style_map::build_styles;
pub use ui::Ui;

/// Easings, springs, timelines and transitions.
pub use prism_ui_anim as anim;
/// The pure-Rust flexbox solver.
pub use prism_ui_layout as layout;
/// The reactive signal/memo/effect graph.
pub use prism_ui_reactive as reactive;
/// Design tokens, classes, selectors and the cascade.
pub use prism_ui_style as style;
/// The retained tree and keyed reconciler.
pub use prism_ui_tree as tree;

/// The `loom!` declarative DSL macro (re-exported for a single import).
pub use prism_ui_macro::loom;
