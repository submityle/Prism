//! Many-light importance sampling: hierarchical light BVH + ReGIR grid.
//!
//! # Conventions
//! * `no_std`: containers via `alloc`; math via `bevy_math`; transcendentals
//!   via `bevy_math::ops` (never `f32::exp`).
//! * All items are deterministic pure functions with `#[cfg(test)]` coverage;
//!   outputs are the numerical reference for the WESL/GPU twins.
//!
//! * [`light_tree`] — a binary light BVH with Conty-Estevez cluster bounds
//!   (AABB + emission cone + power) and a stochastic top-down walk that returns
//!   a leaf emitter and the exact selection `pdf`.
//! * [`regir`] — a uniform world-space light grid whose per-cell reservoirs are
//!   filled by streaming RIS (reusing the screen-probe reservoir) for
//!   constant-cost many-light lookups.

pub mod light_tree;
pub mod regir;

pub use light_tree::{LightBounds, LightCone, LightSample, LightTree, LightTreeNode, importance};
pub use regir::{GridConfig, GridLight, GridQuery, cell_target, fill_cell_reservoir, query};
