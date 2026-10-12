//! The procedural dissolve mask and its coverage, shared by the shading resolve
//! (which lights the cut line) and the raster blend/alpha-test passes (which
//! consume the cut as coverage). §17.1.
//!
//! Kept in its own module, with no dependency beyond the lobe-sampling hash, so
//! the forward raster pipelines can import the coverage without dragging the
//! whole BSDF module graph in with it.
//!
//! Mirrors `prism_render_shading::dynamic::{dissolve_mask, dissolve_coverage}`:
//! the mask is a hash of the *quantised* surface cell, so the cut is stable
//! under sub-cell motion (no shimmer) and needs no authored mask map or extra
//! binding — and both lines derive it identically.

import prism_render_scene::shaders::lobe_sampling::{
    ls_hash_combine,
    ls_hash_u32,
    ls_uniform_from_hash,
};

/// Mask cells per unit of the surface cell space. Fine enough to read as a
/// ragged cut, coarse enough that a pixel does not straddle a cell on its own.
const DISSOLVE_MASK_CELLS: f32 = 96.0;
/// Width of the cut line in mask units: the band over which coverage falls from
/// `1` to `0` (and over which the shading resolve lights the rim).
const DISSOLVE_EDGE_WIDTH: f32 = 0.12;

/// Quantise one cell coordinate into a mask lattice index (saturating cast).
fn dyn_quantize_cell(coord: f32) -> u32 {
    let scaled = coord * DISSOLVE_MASK_CELLS;
    let floored = clamp(floor(scaled), -2147483600.0, 2147483600.0);
    return bitcast<u32>(i32(floored));
}

/// The procedural dissolve mask for one surface cell, in `[0, 1)`.
fn dyn_dissolve_mask(cell: vec2<f32>) -> f32 {
    return ls_uniform_from_hash(ls_hash_combine(
        ls_hash_u32(dyn_quantize_cell(cell.x)),
        dyn_quantize_cell(cell.y),
    ));
}

/// Share of the pixel that survives the cut, in `[0, 1]`.
///
/// Multiply this into the material's alpha where alpha is consumed (alpha test /
/// blend), and use the same value's complement to light the rim in the shading
/// resolve, so the visible edge and the coverage edge cannot disagree.
fn dyn_dissolve_coverage(dissolve: f32, cell: vec2<f32>) -> f32 {
    let progress = clamp(dissolve, 0.0, 1.0);
    if (progress <= 0.0) {
        // Inert: exact identity, and no mask hash is even evaluated.
        return 1.0;
    }
    return clamp((dyn_dissolve_mask(cell) - progress) / DISSOLVE_EDGE_WIDTH, 0.0, 1.0);
}
