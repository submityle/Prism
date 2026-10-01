//! `prism_ui_reactive` — the fine-grained reactive core of Prism's Loom
//! UI/scene notation.
//!
//! This crate provides the three reactive primitives that every higher Loom
//! layer builds on:
//!
//! * [`Signal`] — a writable source value.
//! * [`Memo`] — a cached value derived from other reactive values.
//! * [`Effect`] — a side effect that re-runs when its dependencies change.
//!
//! All three are created from a [`Runtime`], a cheap clonable handle to a shared
//! dependency graph. Dependencies are tracked **automatically**: whatever a memo
//! or effect reads while it runs becomes a dependency, and the set is refreshed
//! on every run so conditional dependencies work correctly.
//!
//! # Design
//!
//! Updates are **glitch-free** and **minimal**: a write pushes staleness down
//! the graph, reads pull values back up, each node recomputes at most once per
//! update, and a node only disturbs its own observers when its value actually
//! changes. See [`runtime`] for the full algorithm. This mirrors the fine-grained
//! model popularised by SolidJS/Leptos rather than a coarse virtual-DOM diff,
//! which is what lets Loom patch individual component fields instead of
//! rebuilding subtrees.
//!
//! The runtime is single-threaded and `no_std`-compatible (requires `alloc`).
//!
//! # Example
//!
//! ```
//! use prism_ui_reactive::Runtime;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! let rt = Runtime::new();
//! let count = rt.signal(0i32);
//! let doubled = rt.memo({
//!     let count = count.clone();
//!     move || count.get() * 2
//! });
//!
//! let log = Rc::new(RefCell::new(Vec::new()));
//! let _effect = rt.effect({
//!     let doubled = doubled.clone();
//!     let log = log.clone();
//!     move || log.borrow_mut().push(doubled.get())
//! });
//!
//! assert_eq!(*log.borrow(), vec![0]);
//! count.set(5);
//! assert_eq!(doubled.get(), 10);
//! assert_eq!(*log.borrow(), vec![0, 10]);
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod effect;
pub mod introspect;
mod memo;
mod node;
pub mod runtime;
mod signal;

pub use effect::Effect;
pub use introspect::{GraphSnapshot, NodeInfo, NodeKindInfo};
pub use memo::Memo;
pub use node::NodeId;
pub use runtime::Runtime;
pub use signal::Signal;
