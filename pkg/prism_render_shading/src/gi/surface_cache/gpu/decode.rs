//! `CPU` mirror of the surfel-atlas decode (gather) producer kernel.
//!
//! [`decode_slot`] is the scalar twin of `surfel_decode_main` in
//! `shaders/surfel_decode.wesl`: given the atlas dimensions and one
//! [`GpuSurfelDecodeRequest`] it resolves the surfel's tile, rejects texels
//! outside it, and octahedrally decodes the tile-local texel centre back to a
//! unit direction — exactly as the kernel does, returning a
//! [`GpuSurfelDecodeResult`]. It is a *faithful op-for-op transcription* of the
//! shader rather than a call into the
//! [`SurfelAtlas`](crate::gi::surface_cache::atlas::SurfelAtlas) golden, so the
//! parity test it anchors is an independent cross-check: it fails if either the
//! shader or the golden drifts.
//!
//! # Bit-exact parity
//!
//! The addressing is integer-exact, and the octahedral decode is the same
//! closed form (`abs` / add / multiply / compare / branchless sign) the golden
//! `oct_to_dir` evaluates in the identical order, finishing with the identical
//! [`Vec3::normalize`] call — so this mirror and the
//! [`atlas`](crate::gi::surface_cache::atlas) golden agree bit-for-bit on every
//! lane. (On device the final `normalize` is the one true-machine op whose last
//! bit may differ, which is acceptable under the no-`GPU` paradigm.)
//!
//! Provenance: standard octahedral surfel-atlas addressing; no Unreal Engine
//! source or derived code.

use bevy_math::Vec3;

use crate::gi::surface_cache::atlas::SurfelAtlas;
use crate::gi::surface_cache::gpu::abi::{
    GpuSurfelDecodeRequest, GpuSurfelDecodeResult, SURFEL_DECODE_FLAG_VALID,
};

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

/// Decode an octahedral `UV` in `[0, 1]^2` to a unit direction, mirroring the
/// shader `oct_to_dir` op-for-op (and therefore the golden
/// [`octahedral::oct_to_dir`](crate::gi::world_space::octahedral::oct_to_dir)).
#[inline]
fn oct_to_dir(uv_x: f32, uv_y: f32) -> Vec3 {
    let ex = uv_x * 2.0 - 1.0;
    let ey = uv_y * 2.0 - 1.0;
    let z = 1.0 - ex.abs() - ey.abs();
    let mut x = ex;
    let mut y = ey;
    if z < 0.0 {
        x = (1.0 - ey.abs()) * sign_not_zero(ex);
        y = (1.0 - ex.abs()) * sign_not_zero(ey);
    }
    Vec3::new(x, y, z).normalize()
}

/// Unit direction stored at a tile-local texel centre, mirroring the shader
/// `local_texel_to_dir` and the golden `SurfelAtlas::local_texel_to_dir`.
#[inline]
fn local_texel_to_dir(lx: u32, ly: u32, resolution: u32) -> Vec3 {
    let res = resolution as f32;
    let max_idx = resolution - 1;
    let cx = (lx.min(max_idx) as f32 + 0.5) / res;
    let cy = (ly.min(max_idx) as f32 + 0.5) / res;
    oct_to_dir(cx, cy)
}

/// Resolve one decode request against the atlas, the scalar twin of
/// `surfel_decode_main`.
///
/// Returns an all-zero [`GpuSurfelDecodeResult::invalid`] when `surfel_id` is at
/// or beyond the atlas capacity or when the texel falls outside the surfel's
/// own tile; otherwise the result carries the decoded unit direction and the
/// valid flag.
#[must_use]
pub fn decode_slot(atlas: &SurfelAtlas, request: &GpuSurfelDecodeRequest) -> GpuSurfelDecodeResult {
    let capacity = atlas.tiles_per_row * atlas.tile_rows;
    if request.surfel_id >= capacity {
        return GpuSurfelDecodeResult::invalid();
    }

    let res = atlas.tile_resolution;
    let col = request.surfel_id % atlas.tiles_per_row;
    let row = request.surfel_id / atlas.tiles_per_row;
    let origin_x = col * res;
    let origin_y = row * res;

    if request.texel_x < origin_x || request.texel_y < origin_y {
        return GpuSurfelDecodeResult::invalid();
    }
    let lx = request.texel_x - origin_x;
    let ly = request.texel_y - origin_y;
    if lx >= res || ly >= res {
        return GpuSurfelDecodeResult::invalid();
    }

    let dir = local_texel_to_dir(lx, ly, res);
    GpuSurfelDecodeResult {
        dir_x: dir.x,
        dir_y: dir.y,
        dir_z: dir.z,
        flags: SURFEL_DECODE_FLAG_VALID,
    }
}
