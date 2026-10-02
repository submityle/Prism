//! `prism_ui_layout` — a pure-Rust flexbox layout solver for Prism's Loom
//! UI/scene notation.
//!
//! The solver is engine-agnostic and deterministic: identical inputs always
//! produce identical box geometry. It is `no_std`-capable (requiring only
//! `alloc`), depends on no external crates, and implements the CSS Flexible
//! Box algorithm from scratch rather than wrapping an existing engine.
//!
//! # Overview
//!
//! Build a [`LayoutTree`], attach styled boxes with [`LayoutTree::new_leaf`],
//! [`LayoutTree::new_leaf_with_measure`], and [`LayoutTree::new_node`], then
//! call [`LayoutTree::compute_layout`]. Read results back with
//! [`LayoutTree::layout`].
//!
//! ```
//! use prism_ui_layout::{
//!     AvailableSpace, Dimension, Display, LayoutStyle, LayoutTree, Size,
//! };
//!
//! let mut tree = LayoutTree::new();
//! let child = tree.new_leaf(LayoutStyle {
//!     size: Size::new(Dimension::Points(50.0), Dimension::Points(50.0)),
//!     ..LayoutStyle::default()
//! });
//! let root = tree.new_node(
//!     LayoutStyle { display: Display::Flex, ..LayoutStyle::default() },
//!     &[child],
//! );
//! tree.compute_layout(
//!     root,
//!     Size::new(AvailableSpace::Definite(200.0), AvailableSpace::Definite(100.0)),
//! );
//! assert_eq!(tree.layout(child).size, Size::new(50.0, 50.0));
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod absolute;
pub mod dirty;
mod flex;
pub mod geometry;
pub mod grid;
pub mod incremental;
pub mod measure;
pub mod modifier;
pub mod protocol;
pub mod result;
pub mod stack;
pub mod style;
pub mod tree;
pub mod wrap;

pub use absolute::AbsoluteProtocol;
pub use dirty::{is_relayout_boundary, DirtyFlags};
pub use geometry::{AvailableSpace, Dimension, Edges, Point, Rect, Size};
pub use grid::GridProtocol;
pub use measure::Measure;
pub use modifier::{Modifier, ModifierChain, Resolved};
pub use protocol::{AlignX, AlignY, Alignment, Constraints, FlexProtocol, LayoutProtocol};
pub use result::Layout;
pub use stack::StackProtocol;
pub use style::{
    AlignContent, AlignItems, Display, FlexDirection, FlexWrap, JustifyContent, LayoutStyle,
    Position,
};
pub use tree::{LayoutTree, NodeId};
pub use wrap::WrapProtocol;
