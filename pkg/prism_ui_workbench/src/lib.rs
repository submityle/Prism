//! `prism_ui_workbench` — a Storybook-style component workbench for Loom.
//!
//! This crate is the workbench layer of Prism's Loom declarative UI system. It
//! does for Loom components what **Storybook**, `SwiftUI`'s `#Preview`, or Ladle
//! do for their ecosystems: it lets you register, organise, and render a
//! component's distinct states — *stories* — in an isolated environment, for
//! preview and testing.
//!
//! # The pieces
//!
//! * [`ControlValue`] / [`ArgSet`] ([`controls`]) — typed, tweakable parameters
//!   ("controls") a story reads while it renders. Editing a value and
//!   re-rendering is how states are explored.
//! * [`Story`] / [`StoryBuilder`] / [`StoryContext`] ([`story`]) — a named use
//!   case holding default args and a render function over the ambient context.
//! * [`Workbench`] ([`registry`]) — a hierarchical, deterministically ordered
//!   registry of stories grouped by a slash-delimited path such as
//!   `"Forms/Button"`.
//! * [`render_story`] / [`RenderResult`] ([`harness`]) — an isolated render
//!   harness: each story renders inside its own reactive runtime, so stories
//!   never interfere, and the result carries a deterministic tree rendering for
//!   snapshots.
//!
//! # Example
//!
//! Register a story, flip one of its controls, and observe the rendered tree
//! change — all in isolation:
//!
//! ```
//! use prism_ui::{Element, ElementKind};
//! use prism_ui_workbench::{render_story, ControlValue, Workbench};
//!
//! // A button story whose fill is driven by a boolean control.
//! let story = prism_ui_workbench::Story::builder("Primary")
//!     .arg("filled", ControlValue::Bool(true))
//!     .arg("label", ControlValue::Text("Click".into()))
//!     .build(|ctx| {
//!         let class = if ctx.bool_arg("filled").unwrap_or(false) {
//!             "btn-filled"
//!         } else {
//!             "btn-outline"
//!         };
//!         let label = ctx.text_arg("label").unwrap_or("Button");
//!         Element::box_().class(class).child(Element::text(label))
//!     });
//!
//! let mut workbench = Workbench::new();
//! workbench.add("Forms/Button", story);
//!
//! let story = workbench
//!     .get("Forms/Button", "Primary")
//!     .expect("story is registered");
//!
//! // Render with the story's default args.
//! let filled = render_story(story, story.default_args());
//! assert_eq!(filled.element().kind(), &ElementKind::Box);
//! assert_eq!(filled.element().class_names(), &["btn-filled".to_string()]);
//! assert_eq!(filled.tree(), "Box\n  Text \"Click\"\n");
//!
//! // Flip the `filled` control and re-render: the output changes.
//! let mut args = story.default_args().clone();
//! args.set_bool("filled", false).expect("same control kind");
//! let outline = render_story(story, &args);
//! assert_eq!(outline.element().class_names(), &["btn-outline".to_string()]);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod controls;
pub mod harness;
pub mod registry;
pub mod story;

pub use controls::{ArgSet, ControlError, ControlKind, ControlValue};
pub use harness::{render_story, render_story_default, render_story_with_context, RenderResult};
pub use registry::Workbench;
pub use story::{Story, StoryBuilder, StoryContext};
