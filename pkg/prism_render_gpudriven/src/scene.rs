//! The deterministic test scene the GPU test and the CPU reference share.
//!
//! The lattice is deliberately well separated from the frustum planes: every
//! instance sits at least one box half-extent clear of the boundary, so the
//! keep/reject decision is unambiguous and the GPU and CPU agree without any
//! floating-point tie-breaking.

use alloc::vec::Vec;

use crate::frustum::{Aabb, Frustum};

/// The default lattice resolution: `10 x 10 x 10 = 1000` instances.
pub const DEFAULT_GRID: u32 = 10;
/// The default distance between neighbouring lattice instances.
pub const DEFAULT_SPACING: f32 = 0.24;
/// The default instance half-extent (every box is a cube).
pub const DEFAULT_HALF_EXTENT: f32 = 0.05;

/// The default view volume: the axis-aligned box `[-0.5, 0.5]^3`.
///
/// With the default lattice this keeps exactly `4^3 = 64` of the 1000
/// instances — enough survivors that a GPU/CPU disagreement is obvious, and
/// enough rejected instances that a cull that does nothing fails the test.
#[must_use]
pub fn view_box_frustum() -> Frustum {
    Frustum::from_box([-0.5; 3], [0.5; 3])
}

/// A `grid^3` cubic lattice of identical AABBs, laid out row-major so instance
/// `i` decomposes as `x = i % grid`, `y = (i / grid) % grid`,
/// `z = i / (grid * grid)` and therefore has a stable, reproducible id.
///
/// The lattice is centred on the origin and spaced `spacing` apart. Ids stay
/// below `grid^3`, which the draw stage relies on to map each survivor to a
/// distinct screen cell.
#[must_use]
pub fn lattice_instances(grid: u32, spacing: f32, half_extent: f32) -> Vec<Aabb> {
    assert!(grid > 0, "the lattice needs at least one instance per axis");
    let center_offset = (grid as f32 - 1.0) * 0.5;
    let half = [half_extent; 3];
    let mut instances = Vec::with_capacity((grid * grid * grid) as usize);
    for z in 0..grid {
        for y in 0..grid {
            for x in 0..grid {
                let center = [
                    (x as f32 - center_offset) * spacing,
                    (y as f32 - center_offset) * spacing,
                    (z as f32 - center_offset) * spacing,
                ];
                instances.push(Aabb::from_center(center, half));
            }
        }
    }
    instances
}
