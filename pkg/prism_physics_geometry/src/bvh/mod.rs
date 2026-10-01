//! Dynamic bounding-volume hierarchy broad-phase.
//!
//! This module provides [`DynamicBvh`], a dynamic axis-aligned bounding box
//! tree, split across an internal node pool (`node`), the tree maintenance
//! core (`tree`), read-only overlap/ray queries (`query`), and nearest-point
//! queries (`nearest`), all-pairs overlap (`pairs`), and frustum culling
//! (`frustum`).

mod frustum;
mod nearest;
mod node;
mod pairs;
mod query;
mod tree;

pub use tree::DynamicBvh;
