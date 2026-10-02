//! Unit tests for the physical page atlas tiling math: `atlas_tile_origin` and
//! `physical_pages_per_edge`.
//!
//! These pin the CPU-side twin of the tile addressing
//! `shaders/vsm_sample.wesl` performs, so a change to either side that breaks
//! the packing is caught before it reaches the GPU.

use bevy_math::UVec2;

use super::resources::{atlas_tile_origin, physical_pages_per_edge};

#[test]
fn physical_pages_per_edge_is_ceil_sqrt_min_one() {
    // Zero pages must still yield a 1x1 grid so the atlas is never zero-sized.
    assert_eq!(physical_pages_per_edge(0), 1);
    assert_eq!(physical_pages_per_edge(1), 1);
    // Exact squares stay exact.
    assert_eq!(physical_pages_per_edge(4), 2);
    assert_eq!(physical_pages_per_edge(64), 8);
    assert_eq!(physical_pages_per_edge(4096), 64);
    // Non-squares round the edge up so every page has a tile.
    assert_eq!(physical_pages_per_edge(5), 3);
    assert_eq!(physical_pages_per_edge(2), 2);
    assert_eq!(physical_pages_per_edge(65), 9);
    assert_eq!(physical_pages_per_edge(4097), 65);
}

#[test]
fn atlas_tile_origin_maps_page_zero_to_the_corner() {
    // Page 0 is always the top-left tile regardless of grid / page size.
    assert_eq!(atlas_tile_origin(0, 8, 128), UVec2::ZERO);
    assert_eq!(atlas_tile_origin(0, 1, 128), UVec2::ZERO);
}

#[test]
fn atlas_tile_origin_walks_across_a_row() {
    // Within the first row the y origin stays 0 and x advances by page_size.
    let page_size = 128;
    let edge = 8;
    assert_eq!(atlas_tile_origin(1, edge, page_size), UVec2::new(128, 0));
    assert_eq!(atlas_tile_origin(3, edge, page_size), UVec2::new(384, 0));
    // The last tile of the first row sits at (edge-1) * page_size in x.
    assert_eq!(
        atlas_tile_origin(edge - 1, edge, page_size),
        UVec2::new((edge - 1) * page_size, 0),
    );
}

#[test]
fn atlas_tile_origin_wraps_to_the_next_row() {
    // The tile just past the first row wraps to x = 0, y = page_size.
    let page_size = 128;
    let edge = 8;
    assert_eq!(atlas_tile_origin(edge, edge, page_size), UVec2::new(0, 128));
    // An interior tile on the second row: page 10 in an 8-wide grid is cell
    // (2, 1) -> (2 * 128, 1 * 128).
    assert_eq!(atlas_tile_origin(10, edge, page_size), UVec2::new(256, 128));
    // The last tile of a full 8x8 grid is cell (7, 7).
    assert_eq!(
        atlas_tile_origin(63, edge, page_size),
        UVec2::new(7 * page_size, 7 * page_size),
    );
}

#[test]
fn atlas_tile_origin_treats_a_zero_edge_as_one() {
    // A degenerate zero edge is clamped to a single column so the math never
    // divides / mods by zero; every page then stacks down column 0.
    let page_size = 64;
    assert_eq!(atlas_tile_origin(0, 0, page_size), UVec2::ZERO);
    assert_eq!(
        atlas_tile_origin(3, 0, page_size),
        UVec2::new(0, 3 * page_size)
    );
}
