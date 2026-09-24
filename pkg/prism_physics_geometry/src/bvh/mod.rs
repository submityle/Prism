//! Dynamic bounding-volume hierarchy broad-phase.
//!
//! This module provides [`DynamicBvh`], a dynamic axis-aligned bounding box
//! tree, split across an internal node pool (`node`), the tree maintenance
//! core (`tree`), and read-only queries (`query`).

mod node;
mod query;
mod tree;

pub use tree::DynamicBvh;
