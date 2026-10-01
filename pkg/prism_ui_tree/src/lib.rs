//! `prism_ui_tree` — the retained node tree and keyed reconciler for Prism's
//! Loom UI/scene notation.
//!
//! This crate owns **identity and structure**: a generational [`Arena`] of
//! [`Node`]s linked into a tree, plus a keyed reconciler that reuses existing
//! nodes across re-renders instead of rebuilding subtrees. Reusing nodes is
//! what lets the higher Loom layers patch component *fields* in place rather
//! than inserting/removing components and churning ECS archetypes.
//!
//! The reconciler ([`diff_keyed`]) minimises moves via a longest-increasing-
//! subsequence pass, the same technique used by `SolidJS` and Vue 3, so the work
//! done is proportional to how much the list actually changed.
//!
//! It builds on [`prism_ui_reactive`] (re-exported as [`reactive`]) for the
//! fine-grained signals that drive those updates.
//!
//! ```
//! use prism_ui_tree::Tree;
//!
//! // A tree keyed by u32 carrying string payloads.
//! let mut tree: Tree<u32, &'static str> = Tree::new();
//! let root = tree.create(None, "root");
//!
//! // First render: keys [1, 2, 3].
//! tree.reconcile_children(root, &[1, 2, 3], |_k| "item");
//! assert_eq!(tree.children(root).len(), 3);
//!
//! // Re-render with a reordered + extended list; nodes 1..=3 are reused.
//! let diff = tree.reconcile_children(root, &[3, 1, 2, 4], |_k| "item");
//! assert_eq!(diff.create_count(), 1); // only key 4 is new
//! assert_eq!(tree.children(root).len(), 4);
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod arena;
pub mod node;
pub mod reconcile;
pub mod tree;

pub use arena::{Arena, NodeId};
pub use node::Node;
pub use reconcile::{diff_keyed, Diff, DiffOp};
pub use tree::Tree;

/// Re-export of the reactive core this crate is built on.
pub use prism_ui_reactive as reactive;
