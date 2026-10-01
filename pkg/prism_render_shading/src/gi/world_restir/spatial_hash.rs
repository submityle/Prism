//! SHARC-style spatial hashing for world-space ReSTIR — CPU golden.
//!
//! World-space ReSTIR stores reservoirs in a sparse, view-independent grid so
//! that global-illumination samples persist across frames and can be reused by
//! any shading point that lands in the same neighbourhood.  Following NVIDIA's
//! *Spatially Hashed Radiance Cache* (SHARC), a cell is identified not by a
//! dense voxel index but by a hash of the quantised world **position**, a
//! distance-scaled **grid level**, and a quantised surface **normal**: binning
//! the normal keeps front- and back-faces (and differently oriented surfaces
//! that happen to share a voxel) in separate reservoirs, which stops radiance
//! from the wrong hemisphere leaking in.
//!
//! The grid level makes the effective cell size grow with camera distance so
//! each cell covers a roughly constant screen footprint: nearby surfaces get
//! fine cells (sharp indirect detail) while distant surfaces get coarse cells
//! (cheaper, noise-free reuse).  A full key is reduced to two numbers that the
//! storage layer consumes directly:
//!
//! * [`hash_key`] / [`bucket_index`] — the slot a key maps to in a fixed-size
//!   open-addressed table.
//! * [`checksum`] — an independent hash stored alongside the slot so a hash
//!   **collision** (two distinct keys landing on the same bucket) is detected
//!   and the colliding sample is rejected rather than silently merged.
//!
//! # Conventions
//! * Positions are right-handed world-space `(x, y, z)`; cells are
//!   `floor(position / cell_size + jitter)` per axis, matching the floor-toward
//!   `-inf` quantisation used by [`crate::gi::world_space::radiance_cache`].
//! * `jitter` is a caller-supplied offset in `[0, 1)^3` *cell units* (the
//!   randomness is injected by the caller, never generated here) that
//!   decorrelates the grid phase to hide the regular cell boundaries; a zero
//!   jitter reproduces the plain axis-aligned grid.
//! * The grid level is `floor(log2(distance * level_scale / base_cell_size))`
//!   clamped to `[0, MAX_LEVEL]`; `level_scale <= 0` disables level scaling and
//!   yields a uniform grid at `base_cell_size`.
//! * Normals are quantised through the sibling octahedral map
//!   ([`crate::gi::world_space::octahedral::dir_to_oct`]) into a
//!   `resolution x resolution` grid of bins, so the normal binning round-trips
//!   to the same bin the GPU twin addresses.
//! * Transcendental math goes through [`bevy_math::ops`]; every helper is a
//!   deterministic pure function (no RNG, no I/O, no GPU, no `unsafe`) and is
//!   defended against zero / negative / non-finite inputs so it can never emit
//!   a `NaN` or panic.

use bevy_math::{ops, IVec3, Vec3};

use crate::gi::world_space::octahedral::dir_to_oct;

/// Upper bound on the grid level so `2^level` stays finite and the biased cell
/// coordinates remain well inside the packed key range.
pub const MAX_LEVEL: i32 = 24;

/// Number of bits each signed cell axis is packed into (biased to unsigned).
const COORD_BITS: u32 = 21;
/// Mask selecting the low [`COORD_BITS`] of a packed axis.
const COORD_MASK: u64 = (1 << COORD_BITS) - 1;
/// Bias added to a signed axis so `+/- 2^(COORD_BITS-1)` maps into the mask.
const COORD_BIAS: i64 = 1 << (COORD_BITS - 1);

/// 64-bit mixing seed for the bucket hash (`hash_key`).
const SEED_KEY: u64 = 0x9e37_79b9_7f4a_7c15;
/// 64-bit mixing seed for the collision checksum (distinct from `SEED_KEY`).
const SEED_SUM: u64 = 0xc2b2_ae3d_27d4_eb4f;

/// Parameters controlling how world geometry is quantised into hash cells.
///
/// All fields are `f32`/`u32` to mirror the GPU uniform twin.  A reasonable
/// default is [`HashGridParams::DEFAULT`]: a 1-unit base cell, 8x8 normal bins,
/// unit level scaling, and no jitter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HashGridParams {
    /// Edge length of a level-0 cell in world units (clamped to be positive).
    pub base_cell_size: f32,
    /// Side count of the `resolution x resolution` normal-bin grid (clamped to
    /// at least `1`).
    pub normal_resolution: u32,
    /// Distance-to-cell-size scale; `<= 0` disables level scaling.
    pub level_scale: f32,
    /// Grid-phase jitter in `[0, 1)^3` cell units, supplied by the caller.
    pub jitter: Vec3,
}

impl Default for HashGridParams {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl HashGridParams {
    /// A sensible default: 1-unit base cell, 8x8 normal bins, unit level scale,
    /// zero jitter.
    pub const DEFAULT: Self = Self {
        base_cell_size: 1.0,
        normal_resolution: 8,
        level_scale: 1.0,
        jitter: Vec3::ZERO,
    };
}

/// A fully-resolved hash-grid cell identity.
///
/// Two shading points produce the *same* key — and therefore share a reservoir
/// — exactly when they fall in the same jittered cell at the same grid level
/// with the same quantised normal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashGridKey {
    /// Integer cell coordinate `floor(position / cell_size + jitter)`.
    pub cell_coord: IVec3,
    /// Distance-derived grid level (`cell_size = base * 2^level`).
    pub level: i32,
    /// Octahedral normal bin in `[0, resolution^2)`.
    pub normal_bin: u32,
}

/// Clamps a cell size to a strictly positive value so reciprocals never blow up.
#[inline]
fn safe_cell_size(cell_size: f32) -> f32 {
    if cell_size.is_finite() && cell_size > f32::MIN_POSITIVE {
        cell_size
    } else {
        f32::MIN_POSITIVE
    }
}

/// Normalises `n`, falling back to `+z` for a degenerate (zero / non-finite)
/// input so the octahedral binning never sees a bad direction.
#[inline]
fn normalize_or_z(n: Vec3) -> Vec3 {
    let len_sq = n.length_squared();
    if len_sq.is_finite() && len_sq > f32::MIN_POSITIVE {
        n * len_sq.sqrt().recip()
    } else {
        Vec3::Z
    }
}

/// Returns the edge length of a cell at `level`: `base_cell_size * 2^level`.
///
/// `base_cell_size` is clamped positive and `level` is clamped to
/// `[0, MAX_LEVEL]`, so the result is always finite and strictly positive.
#[inline]
pub fn cell_size_at_level(base_cell_size: f32, level: i32) -> f32 {
    let base = safe_cell_size(base_cell_size);
    let level = level.clamp(0, MAX_LEVEL) as f32;
    base * ops::exp2(level)
}

/// Chooses the grid level for a world position given the camera position.
///
/// The level is `floor(log2(distance * level_scale / base_cell_size))` clamped
/// to `[0, MAX_LEVEL]`, so the effective cell size grows with camera distance
/// and each cell covers a roughly constant screen footprint.  Returns `0` for a
/// non-positive `level_scale`, for a distance within one base cell, or for any
/// non-finite input, giving a uniform fine grid as the safe fallback.
#[inline]
pub fn grid_level(position: Vec3, camera_position: Vec3, params: &HashGridParams) -> i32 {
    let base = safe_cell_size(params.base_cell_size);
    if !(params.level_scale > 0.0) || !params.level_scale.is_finite() {
        return 0;
    }
    let dist = (position - camera_position).length();
    if !dist.is_finite() || dist <= base {
        return 0;
    }
    // ratio >= 1 so log2 >= 0; floor gives the coarsest level whose cell size
    // does not exceed the (scaled) viewing distance.
    let ratio = (dist * params.level_scale / base).max(1.0);
    let level = ops::log2(ratio).floor();
    if level.is_finite() {
        (level as i32).clamp(0, MAX_LEVEL)
    } else {
        0
    }
}

/// Quantises a world position into its `(cell_coord, level)` pair.
///
/// The level is derived from the camera distance via [`grid_level`]; the
/// position is then floored into cell units at that level with the configured
/// jitter applied.  A non-finite position collapses to the origin cell so the
/// result is always a valid, finite coordinate.
#[inline]
pub fn quantize_position(
    position: Vec3,
    camera_position: Vec3,
    params: &HashGridParams,
) -> (IVec3, i32) {
    let level = grid_level(position, camera_position, params);
    if !position.is_finite() {
        return (IVec3::ZERO, level);
    }
    let size = cell_size_at_level(params.base_cell_size, level);
    let inv = size.recip();
    let j = params.jitter;
    let cell = IVec3::new(
        (position.x * inv + j.x).floor() as i32,
        (position.y * inv + j.y).floor() as i32,
        (position.z * inv + j.z).floor() as i32,
    );
    (cell, level)
}

/// Quantises a surface normal into an octahedral bin in `[0, resolution^2)`.
///
/// The normal is projected through [`dir_to_oct`] to a `[0, 1]^2` UV and binned
/// on a `resolution x resolution` lattice.  `resolution` is clamped to at least
/// `1`; a degenerate normal falls back to the `+z` direction's bin.
#[inline]
pub fn quantize_normal(normal: Vec3, resolution: u32) -> u32 {
    let res = resolution.max(1);
    let uv = dir_to_oct(normalize_or_z(normal));
    let res_f = res as f32;
    let bx = (uv.x * res_f).floor().clamp(0.0, res_f - 1.0) as u32;
    let by = (uv.y * res_f).floor().clamp(0.0, res_f - 1.0) as u32;
    by * res + bx
}

/// Assembles the full [`HashGridKey`] for a shading point.
///
/// Combines [`quantize_position`] and [`quantize_normal`] under one set of
/// [`HashGridParams`].
#[inline]
pub fn compute_key(
    position: Vec3,
    normal: Vec3,
    camera_position: Vec3,
    params: &HashGridParams,
) -> HashGridKey {
    let (cell_coord, level) = quantize_position(position, camera_position, params);
    let normal_bin = quantize_normal(normal, params.normal_resolution);
    HashGridKey {
        cell_coord,
        level,
        normal_bin,
    }
}

/// Packs a key's fields into a single 64-bit integer prior to hashing.
///
/// Each signed axis is biased into [`COORD_BITS`] bits (collision-free within
/// `+/- 2^(COORD_BITS-1)` of the origin); the level and normal bin are folded
/// in with distinct odd multipliers so they perturb independent bit regions.
#[inline]
fn pack(key: &HashGridKey) -> u64 {
    let encode = |v: i32| -> u64 { (((v as i64) + COORD_BIAS) as u64) & COORD_MASK };
    let coord = encode(key.cell_coord.x)
        | (encode(key.cell_coord.y) << COORD_BITS)
        | (encode(key.cell_coord.z) << (2 * COORD_BITS));
    let level = ((key.level as i64) as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let normal = (key.normal_bin as u64).wrapping_mul(0xff51_afd7_ed55_8ccd);
    coord ^ level ^ normal
}

/// MurmurHash3 64-bit finalizer (`fmix64`): avalanches every input bit.
#[inline]
fn fmix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

/// Full 64-bit hash of a key, used to index the reservoir table.
#[inline]
pub fn hash_key(key: &HashGridKey) -> u64 {
    fmix64(pack(key) ^ SEED_KEY)
}

/// Independent 32-bit collision checksum stored beside a slot.
///
/// Derived from a *different* seed than [`hash_key`] and folded down from the
/// avalanched high bits, so two keys sharing a bucket almost always differ in
/// their checksum and the collision is caught by the storage layer.
#[inline]
pub fn checksum(key: &HashGridKey) -> u32 {
    let h = fmix64(pack(key) ^ SEED_SUM);
    // Combine both halves so no single input region is ignored.
    ((h >> 32) as u32) ^ (h as u32)
}

/// Maps a key onto a bucket in a table of `capacity` slots (`capacity >= 1`).
#[inline]
pub fn bucket_index(key: &HashGridKey, capacity: u32) -> u32 {
    let cap = capacity.max(1) as u64;
    (hash_key(key) % cap) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> HashGridParams {
        HashGridParams::DEFAULT
    }

    #[test]
    fn cell_size_doubles_per_level_and_guards_degenerate_base() {
        assert!((cell_size_at_level(1.0, 0) - 1.0).abs() < 1e-6);
        assert!((cell_size_at_level(1.0, 1) - 2.0).abs() < 1e-6);
        assert!((cell_size_at_level(2.0, 3) - 16.0).abs() < 1e-5);
        // Degenerate base is clamped positive; result stays finite.
        let s = cell_size_at_level(0.0, 5);
        assert!(s.is_finite() && s > 0.0);
        let s = cell_size_at_level(-3.0, 2);
        assert!(s.is_finite() && s > 0.0);
    }

    #[test]
    fn grid_level_grows_with_distance_and_is_monotone() {
        let p = params();
        let cam = Vec3::ZERO;
        let near = grid_level(Vec3::new(0.5, 0.0, 0.0), cam, &p);
        let mid = grid_level(Vec3::new(4.0, 0.0, 0.0), cam, &p);
        let far = grid_level(Vec3::new(64.0, 0.0, 0.0), cam, &p);
        assert_eq!(near, 0);
        assert!(mid >= near && far >= mid, "near={near} mid={mid} far={far}");
        assert!(far > near);
        // log2(64) == 6 with unit scale / base.
        assert_eq!(far, 6);
    }

    #[test]
    fn grid_level_disabled_and_degenerate_inputs_are_zero() {
        let mut p = params();
        p.level_scale = 0.0;
        assert_eq!(grid_level(Vec3::splat(1000.0), Vec3::ZERO, &p), 0);
        p.level_scale = f32::NAN;
        assert_eq!(grid_level(Vec3::splat(1000.0), Vec3::ZERO, &p), 0);
        let p = params();
        assert_eq!(grid_level(Vec3::splat(f32::NAN), Vec3::ZERO, &p), 0);
    }

    #[test]
    fn quantize_position_floors_toward_negative_infinity() {
        let p = params();
        let cam = Vec3::ZERO;
        // Within one base cell -> level 0, unit cells.
        let (c, lvl) = quantize_position(Vec3::new(0.9, 0.1, 0.0), cam, &p);
        assert_eq!(lvl, 0);
        assert_eq!(c, IVec3::new(0, 0, 0));
        let (c, _) = quantize_position(Vec3::new(-0.1, 0.0, 0.0), cam, &p);
        assert_eq!(c.x, -1);
    }

    #[test]
    fn quantize_position_nonfinite_is_origin() {
        let p = params();
        let (c, _) = quantize_position(Vec3::new(f32::INFINITY, 0.0, 0.0), Vec3::ZERO, &p);
        assert_eq!(c, IVec3::ZERO);
    }

    #[test]
    fn jitter_shifts_cell_phase() {
        let mut p = params();
        p.level_scale = 0.0; // keep a uniform grid so only jitter changes the cell.
        let pos = Vec3::new(0.4, 0.4, 0.4);
        let (c0, _) = quantize_position(pos, Vec3::ZERO, &p);
        p.jitter = Vec3::splat(0.9);
        let (c1, _) = quantize_position(pos, Vec3::ZERO, &p);
        // 0.4 + 0.9 = 1.3 -> floor 1, so the jitter bumps every axis up by one.
        assert_eq!(c0, IVec3::ZERO);
        assert_eq!(c1, IVec3::splat(1));
    }

    #[test]
    fn normal_bins_are_in_range_and_separate_hemispheres() {
        let res = 8u32;
        for n in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(1.0, 1.0, 1.0),
        ] {
            let b = quantize_normal(n, res);
            assert!(b < res * res, "bin {b} out of range");
        }
        // Opposite normals must not share a bin (that is the whole point).
        assert_ne!(quantize_normal(Vec3::Z, res), quantize_normal(Vec3::NEG_Z, res));
        // Degenerate normal falls back to +z's bin.
        assert_eq!(quantize_normal(Vec3::ZERO, res), quantize_normal(Vec3::Z, res));
    }

    #[test]
    fn normal_resolution_is_clamped() {
        // resolution 0 behaves as 1: a single bin.
        assert_eq!(quantize_normal(Vec3::X, 0), 0);
    }

    #[test]
    fn hash_and_checksum_are_deterministic() {
        let k = compute_key(Vec3::new(1.0, 2.0, 3.0), Vec3::Y, Vec3::ZERO, &params());
        assert_eq!(hash_key(&k), hash_key(&k));
        assert_eq!(checksum(&k), checksum(&k));
        assert_eq!(bucket_index(&k, 1024), bucket_index(&k, 1024));
    }

    #[test]
    fn distinct_keys_mostly_differ_and_bucket_is_in_range() {
        use alloc::collections::BTreeSet;
        let mut p = params();
        p.level_scale = 0.0; // uniform grid so every integer cell is distinct.
        let cam = Vec3::ZERO;
        let mut keys = BTreeSet::new();
        let mut hashes = BTreeSet::new();
        let mut collisions = 0usize;
        let mut total = 0usize;
        for x in -4..4 {
            for y in -4..4 {
                for z in -4..4 {
                    let pos = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    let k = compute_key(pos, Vec3::Z, cam, &p);
                    keys.insert((k.cell_coord.x, k.cell_coord.y, k.cell_coord.z, k.level, k.normal_bin));
                    let h = hash_key(&k);
                    if !hashes.insert(h) {
                        collisions += 1;
                    }
                    total += 1;
                    let b = bucket_index(&k, 4096);
                    assert!(b < 4096);
                }
            }
        }
        // All cells are geometrically distinct.
        assert_eq!(keys.len(), total);
        // The 64-bit hash must be collision-free over this small block.
        assert_eq!(collisions, 0, "unexpected hash collisions: {collisions}");
    }

    #[test]
    fn hash_key_depends_on_level_and_normal() {
        let base = HashGridKey {
            cell_coord: IVec3::new(3, -2, 5),
            level: 1,
            normal_bin: 7,
        };
        let diff_level = HashGridKey { level: 2, ..base };
        let diff_normal = HashGridKey { normal_bin: 8, ..base };
        assert_ne!(hash_key(&base), hash_key(&diff_level));
        assert_ne!(hash_key(&base), hash_key(&diff_normal));
    }
}
