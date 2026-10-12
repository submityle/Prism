//! The deterministic test scene the GPU test and the CPU reference share.
//!
//! The lattice is deliberately well separated from the frustum planes: every
//! instance sits at least one box half-extent clear of the boundary, so the
//! keep/reject decision is unambiguous and the GPU and CPU agree without any
//! floating-point tie-breaking.

use alloc::vec::Vec;

use crate::frustum::{Aabb, Frustum};
use crate::instances::{id_color, Instance};

/// The default lattice resolution: `10 x 10 x 10 = 1000` instances.
pub const DEFAULT_GRID: u32 = 10;
/// The default distance between neighbouring lattice instances.
pub const DEFAULT_SPACING: f32 = 0.24;
/// The default instance half-extent (every box is a cube).
pub const DEFAULT_HALF_EXTENT: f32 = 0.05;

/// The lattice layer the plane-view frustums keep: the middle of the grid, so
/// the layers on either side are one full spacing away and fall cleanly outside
/// the slab.
pub const VIEW_LAYER: u32 = DEFAULT_GRID / 2;

/// The default view volume: the axis-aligned box `[-0.5, 0.5]^3`.
///
/// With the default lattice this keeps exactly `4^3 = 64` of the 1000
/// instances — enough survivors that a GPU/CPU disagreement is obvious, and
/// enough rejected instances that a cull that does nothing fails the test.
#[must_use]
pub fn view_box_frustum() -> Frustum {
    Frustum::from_box([-0.5; 3], [0.5; 3])
}

/// The half-width of the XY window that contains every lattice column of an
/// unmodified [`lattice_instances`] batch.
///
/// The outermost column centre sits at `(grid - 1) / 2 * spacing`; adding one
/// half-extent reaches its outer face.
#[must_use]
pub fn lattice_half_width() -> f32 {
    (DEFAULT_GRID as f32 - 1.0) * 0.5 * DEFAULT_SPACING + DEFAULT_HALF_EXTENT
}

/// The Z centre of lattice layer `layer`.
#[must_use]
pub fn layer_center(layer: u32) -> f32 {
    (layer as f32 - (DEFAULT_GRID as f32 - 1.0) * 0.5) * DEFAULT_SPACING
}

/// A thin slab that keeps exactly the lattice sites in layer `layer` and rejects
/// every site in the layers above and below.
///
/// The slab is centred on the layer and one full `spacing` tall, so the
/// neighbouring layer's nearest face is half a spacing clear of the slab — a
/// margin no floating-point tie can straddle. `xy_half` is the slab's XY
/// half-width, which selects how many columns survive.
#[must_use]
pub fn layer_slab(layer: u32, xy_half: f32) -> Frustum {
    let z = layer_center(layer);
    let half_gap = DEFAULT_SPACING * 0.5;
    Frustum::from_box(
        [-xy_half, -xy_half, z - half_gap],
        [xy_half, xy_half, z + half_gap],
    )
}

/// The plane-view frustum that keeps the whole `grid x grid` layer.
///
/// Because every survivor shares one Z coordinate, the orthographic view along
/// `-Z` maps each survivor to its own screen cell: the projection is injective,
/// so the CPU reference can name the exact pixel each survivor paints.
#[must_use]
pub fn plane_view_frustum() -> Frustum {
    layer_slab(VIEW_LAYER, lattice_half_width())
}

/// The plane-view frustum that keeps only the central `2 x 2` columns.
///
/// The other columns of the same layer are still candidates, so they prove that
/// a culled instance leaves the background untouched at the pixel it would have
/// painted.
#[must_use]
pub fn plane_view_center_frustum() -> Frustum {
    layer_slab(VIEW_LAYER, DEFAULT_SPACING)
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

/// The first `count` instances of the smallest cubic lattice that holds them.
///
/// The benchmark feeds this to both submission paths so the only difference
/// between them is how the frame is submitted, not what it contains.
///
/// # Panics
///
/// Panics when `count` is zero.
#[must_use]
pub fn candidate_lattice(count: u32) -> Vec<Aabb> {
    assert!(count > 0, "a candidate lattice needs at least one instance");
    let grid = (count as f64).cbrt().ceil() as u32;
    let mut instances = lattice_instances(grid, DEFAULT_SPACING, DEFAULT_HALF_EXTENT);
    instances.truncate(count as usize);
    instances
}

/// The draw-side records for a batch of instance AABBs, each colored by its id.
///
/// The color is derived from the instance's index, which is the id the cull pass
/// appends and the draw reads back, so a rendered pixel identifies its instance
/// exactly.
#[must_use]
pub fn instance_records(aabbs: &[Aabb]) -> Vec<Instance> {
    aabbs
        .iter()
        .enumerate()
        .map(|(index, aabb)| Instance::new(aabb.center(), id_color(index as u32)))
        .collect()
}
