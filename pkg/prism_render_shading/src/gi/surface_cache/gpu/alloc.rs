//! `CPU` mirror of the surfel-allocation (atlas-addressing) producer kernel.
//!
//! [`allocate_slot`] is the scalar twin of `surfel_alloc_main` in
//! `shaders/surfel_alloc.wesl`: given the atlas dimensions and one
//! [`GpuSurfelAllocRequest`] it resolves the request's global atlas texel and
//! tile exactly as the kernel does, returning a [`GpuSurfelAllocSlot`]. It is a
//! *faithful op-for-op transcription* of the shader rather than a call into the
//! [`SurfelAtlas`](crate::gi::surface_cache::atlas::SurfelAtlas) golden, so the
//! parity test it anchors is an independent cross-check: it fails if either the
//! shader or the golden drifts.
//!
//! # Bit-exact parity
//!
//! The addressing is integer-exact apart from the octahedral `dir_to_oct`
//! fold, which is a closed-form sequence of `abs` / add / reciprocal / multiply
//! / `floor` in the portable scalar subset — no transcendental, no reorderable
//! reduction, no fused multiply-add in the reference path — so the kernel, this
//! mirror, and the [`atlas`](crate::gi::surface_cache::atlas) golden evaluate
//! the identical closed form in the identical order and agree bit-for-bit on
//! every lane. The reciprocal matches the golden's `l1.recip()`
//! (`f32::recip` is specified as `1.0 / self`), and `clamp_index`'s
//! `!(value > 0.0)` guard collapses both non-positive and `NaN` inputs to zero,
//! matching the golden `clamp_index` on the clamped `[0, res]` domain.
//!
//! Provenance: standard octahedral surfel-atlas addressing; no Unreal Engine
//! source or derived code.

use crate::gi::surface_cache::atlas::SurfelAtlas;
use crate::gi::surface_cache::gpu::abi::{GpuSurfelAllocRequest, GpuSurfelAllocSlot};

/// Smallest positive normal `f32`; guards the octahedral `L1` reciprocal
/// against a zero-length direction, matching `OCT_MIN_POSITIVE` in the shader
/// and `f32::MIN_POSITIVE` in the golden `dir_to_oct`.
const OCT_MIN_POSITIVE: f32 = f32::MIN_POSITIVE;

/// Branchless sign mapping zero to `+1` (GLSL `signNotZero`); keeps the
/// octahedral fold continuous across the seam. Mirrors `sign_not_zero` in the
/// shader and the golden.
#[inline]
fn sign_not_zero(value: f32) -> f32 {
    if value >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Encode a direction into octahedral `UV` in `[0, 1]^2`, mirroring the shader
/// `dir_to_oct` op-for-op (and therefore the golden
/// [`octahedral::dir_to_oct`](crate::gi::world_space::octahedral::dir_to_oct)).
#[inline]
fn dir_to_oct(dir_x: f32, dir_y: f32, dir_z: f32) -> (f32, f32) {
    let l1 = dir_x.abs() + dir_y.abs() + dir_z.abs();
    if l1 <= OCT_MIN_POSITIVE {
        return (0.5, 0.5);
    }
    let inv = l1.recip();
    let mut px = dir_x * inv;
    let mut py = dir_y * inv;
    if dir_z < 0.0 {
        let fx = (1.0 - py.abs()) * sign_not_zero(px);
        let fy = (1.0 - px.abs()) * sign_not_zero(py);
        px = fx;
        py = fy;
    }
    (px * 0.5 + 0.5, py * 0.5 + 0.5)
}

/// Clamp a floored texel index into `[0, max_idx]`, mirroring the golden
/// `SurfelAtlas::clamp_index` byte-for-byte: non-finite or non-positive
/// inputs collapse to zero. On the clamped `[0, res]` domain the inputs are
/// always finite, so this agrees with the shader's `!(value > 0.0)` guard.
#[inline]
fn clamp_index(value: f32, max_idx: u32) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    (value as u32).min(max_idx)
}

/// Nearest tile-local texel for an octahedral `UV`, mirroring the shader
/// `uv_to_local` and the golden `SurfelAtlas::uv_to_local`.
#[inline]
fn uv_to_local(uv_x: f32, uv_y: f32, res: u32) -> (u32, u32) {
    let res_f = res as f32;
    let fx = (uv_x.clamp(0.0, 1.0) * res_f).floor();
    let fy = (uv_y.clamp(0.0, 1.0) * res_f).floor();
    let max_idx = res - 1;
    (clamp_index(fx, max_idx), clamp_index(fy, max_idx))
}

/// Resolve one allocation request against the atlas, the scalar twin of
/// `surfel_alloc_main`.
///
/// Returns an all-zero [`GpuSurfelAllocSlot::invalid`] when `surfel_id` is at
/// or beyond the atlas capacity; otherwise the slot carries the global atlas
/// texel, the surfel's tile coordinate, and the valid flag.
#[must_use]
pub fn allocate_slot(atlas: &SurfelAtlas, request: &GpuSurfelAllocRequest) -> GpuSurfelAllocSlot {
    let capacity = atlas.tiles_per_row * atlas.tile_rows;
    if request.surfel_id >= capacity {
        return GpuSurfelAllocSlot::invalid();
    }

    let col = request.surfel_id % atlas.tiles_per_row;
    let row = request.surfel_id / atlas.tiles_per_row;
    let origin_x = col * atlas.tile_resolution;
    let origin_y = row * atlas.tile_resolution;

    let (uv_x, uv_y) = dir_to_oct(request.dir_x, request.dir_y, request.dir_z);
    let (local_x, local_y) = uv_to_local(uv_x, uv_y, atlas.tile_resolution);

    GpuSurfelAllocSlot {
        texel_x: origin_x + local_x,
        texel_y: origin_y + local_y,
        tile_col: col,
        tile_row: row,
        flags: crate::gi::surface_cache::gpu::abi::SURFEL_ALLOC_FLAG_VALID,
    }
}
