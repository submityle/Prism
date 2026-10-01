//! `prism_ui_hotreload` — state-preserving hot-reload core for Loom.
//!
//! When a `.loom` view or `.loom.style` sheet changes on disk, the UI should
//! update in place without throwing away transient per-node state (scroll
//! offsets, text-field contents, animation progress, and so on). This crate
//! implements the engine-agnostic core of that behaviour:
//!
//! * [`identity`] assigns every node a stable [`identity::NodePath`] so a node
//!   in the new tree can be matched to its counterpart in the old tree.
//! * [`plan`] diffs the old and new trees into a [`ReloadPlan`] that classifies
//!   each node as preserved, added, removed, or recreated.
//! * [`statestore`] stores per-node state keyed by path and applies a plan,
//!   dropping only the state that can no longer be valid.
//! * [`style_reload`] diffs the resolved styles of a list of class names so a
//!   backend can hot-swap exactly the properties that changed.
//! * [`reload`] ties the tree and its state together in a [`HotReloader`].
//!
//! Everything is deterministic and uses only integer and ordering operations,
//! so reloads are fully reproducible.
//!
//! # Example
//!
//! ```
//! use prism_ui::{Element, Key};
//! use prism_ui_hotreload::{identity::paths_of, HotReloader};
//!
//! // A keyed list: an item to keep, one whose kind will change, one to drop.
//! let old = Element::box_()
//!     .child(Element::text("keep").key_str("a"))
//!     .child(Element::box_().key_str("b"))
//!     .child(Element::text("gone").key_str("c"));
//!
//! let mut reloader: HotReloader<&str> = HotReloader::new(old.clone());
//!
//! // Seed per-node state for the "a" and "b" items. Collect first so the
//! // borrow of `current()` ends before the mutable `state_mut()` call.
//! let seeds: Vec<(_, &str)> = paths_of(reloader.current())
//!     .into_iter()
//!     .filter_map(|(path, element)| match element.explicit_key() {
//!         Some(Key::Str(k)) if k == "a" => Some((path, "a-state")),
//!         Some(Key::Str(k)) if k == "b" => Some((path, "b-state")),
//!         _ => None,
//!     })
//!     .collect();
//! for (path, state) in seeds {
//!     reloader.state_mut().insert(path, state);
//! }
//!
//! // New source: "a" preserved, "b" changes kind (box -> text) so it is
//! // recreated, "c" is removed, and "d" is added.
//! let new = Element::box_()
//!     .child(Element::text("keep").key_str("a"))
//!     .child(Element::text("changed").key_str("b"))
//!     .child(Element::text("new").key_str("d"));
//!
//! let report = reloader.reload(new);
//!
//! // Root plus the "a" text are preserved; "d" is added; "b" state is dropped.
//! assert_eq!(report.preserved, 2);
//! assert_eq!(report.added, 1);
//! assert_eq!(report.dropped, 1);
//!
//! // The preserved item keeps its state; the recreated item lost its state.
//! let a_path = paths_of(reloader.current())
//!     .into_iter()
//!     .find(|(_, e)| e.explicit_key() == Some(&Key::Str("a".into())))
//!     .map(|(p, _)| p)
//!     .unwrap();
//! assert_eq!(reloader.state().get(&a_path), Some(&"a-state"));
//! assert_eq!(reloader.state().len(), 1);
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod identity;
pub mod plan;
pub mod reload;
pub mod statestore;
pub mod style_reload;

pub use identity::{paths_of, NodeIdent, NodePath};
pub use plan::{plan, NodeChange, PlanCounts, Recreated, ReloadPlan};
pub use reload::HotReloader;
pub use statestore::{ReloadReport, StateStore};
pub use style_reload::{diff_classes, ClassChange, PropValue, StyleDiff};
