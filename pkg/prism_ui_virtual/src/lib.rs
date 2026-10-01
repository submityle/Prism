//! `prism_ui_virtual` — list virtualization for Prism's Loom UI notation.
//!
//! Long lists are expensive to materialize in full: a hundred-thousand-row log
//! view only ever shows a few dozen rows at a time, yet a naive tree would
//! build an element for every row every frame. This crate windows that work.
//! Given list metrics and a scroll [`Viewport`], it computes the slice of items
//! that intersect the visible window (plus a configurable overscan margin) and
//! builds a container with just that slice, bracketed by two spacer boxes that
//! reserve the exact off-screen extent. The scrollbar therefore behaves as if
//! the whole list were present while only the visible items are built.
//!
//! # Layers
//!
//! * [`Viewport`] — the scroll offset, visible length and overscan.
//! * [`FixedList`] — constant-time metrics for uniform item sizes.
//! * [`VariableList`] — prefix-sum metrics for per-item sizes, with a binary
//!   search from pixel offset to item index.
//! * [`RecyclePool`] — deterministic reuse of a bounded set of render slots as
//!   items scroll in and out of view.
//! * [`virtualize_fixed`] / [`virtualize_variable`] — assemble the windowed
//!   [`Element`](prism_ui::Element) tree.
//!
//! # Example
//!
//! ```
//! use prism_ui::Element;
//! use prism_ui_virtual::{virtualize_fixed, FixedList, Viewport};
//!
//! // 1000 rows, each 20px tall, no gap.
//! let list = FixedList::new(1000, 20.0, 0.0);
//! // Scrolled to 100px with a 80px window: rows 5..=9 are visible.
//! let viewport = Viewport::new(100.0, 80.0);
//!
//! let (range, view) = virtualize_fixed(&list, &viewport, |i| {
//!     Element::text(format!("row {i}"))
//! });
//!
//! assert_eq!(range, 5..10);
//! // Container children: leading spacer + 5 rows + trailing spacer.
//! assert_eq!(view.child_elements().len(), 7);
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod fixed;
mod pool;
mod variable;
mod viewport;
mod virtualize;

pub use fixed::FixedList;
pub use pool::{RecyclePool, SlotId};
pub use variable::VariableList;
pub use viewport::Viewport;
pub use virtualize::{
    virtualize_fixed, virtualize_variable, CONTAINER_CLASS, ITEM_CLASS, SPACER_CLASS,
};
