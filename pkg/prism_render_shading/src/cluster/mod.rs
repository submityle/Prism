//! Backend-neutral clustered forward (Forward+) light-culling core.
//!
//! This module owns the CPU golden reference that partitions a camera frustum
//! into froxels and buckets punctual lights into them, so the shading resolve
//! pass only iterates the handful of lights that can actually reach a pixel
//! instead of the whole scene light list.  The pieces are split by concern to
//! keep each a small, independently testable numerical twin of the future GPU
//! culling shader:
//!
//! * [`grid`] - the froxel [`ClusterGrid`]: screen tiling and exponential depth
//!   slicing, plus the `view_z <-> slice` mapping.
//! * [`bounds`] - per-froxel view-space [`ClusterAabb`] reconstruction via the
//!   inverse projection ([`ClusterBoundsBuilder`]).
//! * [`assign`] - the sphere / spot-cone light-to-cluster assignment producing
//!   the GPU-shaped [`ClusterLightAssignment`] offset/count table.
//!
//! The GPU compute twin (froxel bounds + light culling dispatch) and the
//! resolve-side wiring that consumes the index table are built in the render
//! crate and require on-device validation for numerical parity.

pub mod assign;
pub mod bounds;
pub mod grid;

pub use assign::{assign_lights_to_clusters, ClusterAssignmentConfig, ClusterLightAssignment};
pub use bounds::{ClusterAabb, ClusterBoundsBuilder};
pub use grid::ClusterGrid;
