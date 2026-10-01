//! `prism_ui_store` — Loom's predictable, structured global-state layer.
//!
//! This crate is Loom's answer to Redux/Zustand/Pinia: a thin, predictable
//! state container built directly on the fine-grained reactivity of
//! [`prism_ui_reactive`]. A [`Store`] owns a single value `S` backed by a
//! reactive [`Signal`](prism_ui_reactive::Signal), so every read performed
//! inside a memo or effect is tracked automatically and every commit notifies
//! exactly the observers that are affected.
//!
//! # Pieces
//!
//! * [`Store`] — the reactive state container. Reads are tracked, writes run
//!   middleware and notify observers.
//! * [`Store::select`] — derive a fine-grained [`Memo`](prism_ui_reactive::Memo)
//!   over a slice of the state. The memo recomputes whenever the state changes
//!   but only disturbs its own downstream observers when the *selected* slice
//!   actually changes.
//! * [`Middleware`] — a hook invoked around every committing write, plus the
//!   ready-made [`LoggingMiddleware`] that records `(prev, next)` snapshots.
//!
//! # Commit policy
//!
//! [`Store::update`] and [`Store::set`] always commit: they run middleware and
//! notify observers unconditionally, because without a `PartialEq` bound the
//! store cannot tell whether the new value differs from the old one. When `S`
//! is [`PartialEq`], [`Store::set_if_changed`] gives you no-op suppression: an
//! equal value is dropped before any middleware runs and no notification is
//! propagated. This split keeps the general case predictable while still
//! offering cheap change-gating when the type supports it.
//!
//! # Example
//!
//! ```
//! use prism_ui_reactive::Runtime;
//! use prism_ui_store::Store;
//!
//! let rt = Runtime::new();
//! let store = Store::new(&rt, 0i32);
//! assert_eq!(store.get(), 0);
//!
//! let doubled = store.select(|&n| n * 2);
//! store.update(|n| *n += 5);
//! assert_eq!(store.get(), 5);
//! assert_eq!(doubled.get(), 10);
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod middleware;
mod selector;
mod store;

pub use middleware::{CommitLog, LoggingMiddleware, Middleware};
pub use store::Store;
