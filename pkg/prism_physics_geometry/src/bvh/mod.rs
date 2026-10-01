//! Dynamic bounding-volume hierarchy broad-phase.
//!
//! This module provides [`DynamicBvh`], a dynamic axis-aligned bounding box
//! tree, split across an internal node pool (`node`), the tree maintenance
//! core (`tree`), read-only overlap/ray queries (`query`), and nearest-point
//! queries (`nearest`), and all-pairs overlap (`pairs`).

mod nearest;
mod node;
mod pairs;
mod query;
mod tree;

pub use tree::DynamicBvh;
