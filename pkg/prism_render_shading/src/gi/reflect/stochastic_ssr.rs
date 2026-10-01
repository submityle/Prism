//! Stochastic Hi-Z screen-space reflection — CPU golden.
//!
//! Real-time screen-space reflection (SSR) marches a reflected ray through the
//! depth buffer looking for the first surface it crosses.  A naive per-texel
//! linear march is prohibitively expensive for grazing rays, so production
//! tracers accelerate it with a **Hi-Z (hierarchical-Z) min pyramid**: a mip
//! chain whose coarse levels store the *minimum* depth (closest surface) of the
//! texels they cover.  A ray that stays in front of the minimum depth of a
//! coarse cell is guaranteed to miss every surface inside that cell, so the
//! march can skip the whole cell in one step and only descend to finer levels
//! near a potential hit.  Stochastic SSR additionally jitters the reflected
//! direction with a GGX half-vector sample so that rough reflections converge
//! under temporal accumulation instead of showing a single mirror streak.
//!
//! This module is the backend-neutral reference for that tracer:
//!
//! * [`HiZPyramid`] owns a min/max mip chain, built either from a base-level
//!   depth image via [`HiZPyramid::from_base`] or from caller-supplied per-mip
//!   slices via [`HiZPyramid::from_mip_slices`], and exposes [`HiZPyramid::hiz_fetch`].
//! * [`hiz_trace`] performs the cell-crossing hierarchical march and reports the
//!   hit UV, hit depth, validity, and the number of iterations it consumed.
//! * [`linear_trace`] is the honest single-resolution baseline used to assert
//!   that the hierarchical march never costs more iterations.
//! * [`ggx_importance_sample`] jitters the reflected direction with a GGX NDF
//!   half-vector sample and returns the solid-angle pdf.
//!
//! # Conventions
//! * `no_std`: containers via `alloc`; math via `bevy_math`; transcendentals via
//!   [`bevy_math::ops`] (never `f32::exp`).  `sqrt` uses the inherent method.
//! * Depth is a scalar in `[0, 1]` with **larger meaning farther** from the
//!   camera (a standard forward depth buffer).  A reflected ray that travels
//!   into the scene therefore has an increasing depth component.
//! * UV lives in `[0, 1]^2`; `(0, 0)` is the first texel's corner.  Integer cell
//!   coordinates are `floor(uv * mip_size)` clamped to the mip's bounds.
//! * The Hi-Z pyramid stores the per-cell *minimum* depth (the closest surface)
//!   when reduced with [`HiZReduce::Min`]; this is the reduction [`hiz_trace`]
//!   is designed for.  [`HiZReduce::Max`] is provided for farthest-surface
//!   queries and symmetry.
//! * Every function is deterministic: no RNG, no I/O, no GPU, no `unsafe`, no
//!   global state.  Randomness enters only through explicit `u ∈ [0, 1)` inputs.
//!   Degenerate inputs are clamped so results are always finite (never `NaN`).

use alloc::vec::Vec;
use bevy_math::{ops, UVec2, Vec2, Vec3};

use crate::gi::sample::world_from_local;

/// Smallest roughness treated as a real GGX lobe; anything at or below
/// [`MIRROR_ROUGHNESS`] collapses to a perfect mirror.
pub const MIRROR_ROUGHNESS: f32 = 1.0e-4;
/// Lower clamp applied to a genuine (non-mirror) roughness so the GGX
/// distribution never degenerates numerically.
pub const MIN_ROUGHNESS: f32 = 1.0e-3;
/// A tiny epsilon used to nudge the ray just across a cell boundary so the
/// march cannot stall on the edge it just reached.
const CELL_EPSILON: f32 = 1.0e-6;
/// Smallest magnitude the screen-space depth slope may have before the march
/// treats the ray as degenerate (parallel to the depth planes).
const MIN_DEPTH_SLOPE: f32 = 1.0e-6;

/// Reduction operator used when building coarse Hi-Z levels from finer ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HiZReduce {
    /// Keep the minimum (closest) depth of the covered texels — the reduction
    /// [`hiz_trace`] assumes.
    Min,
    /// Keep the maximum (farthest) depth of the covered texels.
    Max,
}

impl HiZReduce {
    /// Combines two depths under this reduction.
    #[inline]
    fn fold(self, a: f32, b: f32) -> f32 {
        match self {
            HiZReduce::Min => a.min(b),
            HiZReduce::Max => a.max(b),
        }
    }

    /// The identity element for an empty fold (so a 1-wide odd row is safe).
    #[inline]
    fn identity(self) -> f32 {
        match self {
            HiZReduce::Min => f32::INFINITY,
            HiZReduce::Max => f32::NEG_INFINITY,
        }
    }
}

/// A hierarchical-Z depth pyramid: a mip chain of per-cell reduced depths.
///
/// Level 0 is the full-resolution depth image; each coarser level halves the
/// resolution (rounding up) and stores the [`HiZReduce`]-folded depth of the
/// up-to-`2x2` block it covers.  All levels are packed into a single flat
/// buffer; [`HiZPyramid::hiz_fetch`] addresses them through per-level offsets.
#[derive(Clone, Debug)]
pub struct HiZPyramid {
    sizes: Vec<UVec2>,
    offsets: Vec<usize>,
    data: Vec<f32>,
}

impl HiZPyramid {
    /// Number of mip levels in a full chain for `base`, down to the `1x1` top.
    ///
    /// Returns at least `1` (level 0 always exists), even for a degenerate zero
    /// or one-texel base.
    #[inline]
    pub fn full_level_count(base: UVec2) -> u32 {
        let max_dim = base.x.max(base.y).max(1);
        // floor(log2(max_dim)) + 1
        32 - (max_dim.leading_zeros())
    }

    /// Computes the dimensions of `level` for a `base`-sized pyramid.
    ///
    /// Each axis is halved per level, rounding up, and clamped to at least `1`.
    #[inline]
    pub fn level_size(base: UVec2, level: u32) -> UVec2 {
        // `>>` floors; add `2^level - 1` first so coverage rounds up and never
        // loses the last odd texel.  Each axis is clamped to at least `1`.
        let span = 1u32 << level;
        UVec2::new(
            ((base.x + span - 1) >> level).max(1),
            ((base.y + span - 1) >> level).max(1),
        )
    }

    /// Builds a pyramid by reducing a base-level depth image.
    ///
    /// * `base` — level-0 dimensions `(width, height)`.
    /// * `depth` — row-major level-0 depths, length `base.x * base.y`.  A short
    ///   slice is padded with the reduction identity so the call never panics.
    /// * `reduce` — [`HiZReduce::Min`] for a closest-surface pyramid (the one
    ///   [`hiz_trace`] expects) or [`HiZReduce::Max`].
    /// * `levels` — number of levels to build, clamped to `[1, full]`.
    ///
    /// Non-finite base depths are replaced by the reduction identity so coarse
    /// levels stay finite.
    pub fn from_base(base: UVec2, depth: &[f32], reduce: HiZReduce, levels: u32) -> Self {
        let full = Self::full_level_count(base);
        let levels = levels.clamp(1, full);

        let mut sizes = Vec::with_capacity(levels as usize);
        let mut offsets = Vec::with_capacity(levels as usize);
        for l in 0..levels {
            sizes.push(Self::level_size(base, l));
        }
        let total: usize = sizes.iter().map(|s| (s.x as usize) * (s.y as usize)).sum();
        let mut data = Vec::with_capacity(total);

        // Level 0: copy / sanitise the base depths.
        let base_len = (base.x as usize) * (base.y as usize);
        offsets.push(0);
        for i in 0..base_len {
            let d = depth.get(i).copied().unwrap_or(reduce.identity());
            data.push(if d.is_finite() { d } else { reduce.identity() });
        }

        // Coarser levels: reduce the previous level.
        for l in 1..levels {
            offsets.push(data.len());
            let size = sizes[l as usize];
            let prev = sizes[(l - 1) as usize];
            let prev_off = offsets[(l - 1) as usize];
            for y in 0..size.y {
                for x in 0..size.x {
                    let mut acc = reduce.identity();
                    for dy in 0..2u32 {
                        for dx in 0..2u32 {
                            let px = (x * 2 + dx).min(prev.x - 1);
                            let py = (y * 2 + dy).min(prev.y - 1);
                            let idx = prev_off + (py as usize) * (prev.x as usize) + px as usize;
                            acc = reduce.fold(acc, data[idx]);
                        }
                    }
                    data.push(acc);
                }
            }
        }

        Self {
            sizes,
            offsets,
            data,
        }
    }

    /// Builds a pyramid directly from caller-supplied per-mip depth slices.
    ///
    /// `mips[0]` is level 0 (size `base`) and each subsequent slice is the next
    /// coarser level, whose size is derived from `base` via [`HiZPyramid::level_size`].
    /// Missing or short slices are padded with `+∞`; non-finite entries are
    /// replaced with `+∞` so a min-pyramid stays usable.  `base` with a zero
    /// axis is promoted to `1`.  At least one level is always produced.
    pub fn from_mip_slices(base: UVec2, mips: &[&[f32]]) -> Self {
        let base = UVec2::new(base.x.max(1), base.y.max(1));
        let full = Self::full_level_count(base);
        let levels = (mips.len() as u32).clamp(1, full);

        let mut sizes = Vec::with_capacity(levels as usize);
        let mut offsets = Vec::with_capacity(levels as usize);
        let mut data = Vec::new();
        for l in 0..levels {
            let size = Self::level_size(base, l);
            offsets.push(data.len());
            sizes.push(size);
            let count = (size.x as usize) * (size.y as usize);
            let slice = mips.get(l as usize).copied().unwrap_or(&[]);
            for i in 0..count {
                let d = slice.get(i).copied().unwrap_or(f32::INFINITY);
                data.push(if d.is_finite() { d } else { f32::INFINITY });
            }
        }

        Self {
            sizes,
            offsets,
            data,
        }
    }

    /// Number of mip levels stored.
    #[inline]
    pub fn levels(&self) -> u32 {
        self.sizes.len() as u32
    }

    /// Dimensions of `level`, clamped to the top level if out of range.
    #[inline]
    pub fn mip_size(&self, level: u32) -> UVec2 {
        let l = (level as usize).min(self.sizes.len() - 1);
        self.sizes[l]
    }

    /// Fetches the reduced depth at `cell` on `level`.
    ///
    /// The level and cell coordinates are clamped to valid bounds (clamp-to-edge),
    /// so any input returns a stored, finite depth without panicking.
    #[inline]
    pub fn hiz_fetch(&self, level: u32, cell: UVec2) -> f32 {
        let l = (level as usize).min(self.sizes.len() - 1);
        let size = self.sizes[l];
        let x = cell.x.min(size.x - 1);
        let y = cell.y.min(size.y - 1);
        let idx = self.offsets[l] + (y as usize) * (size.x as usize) + x as usize;
        self.data[idx]
    }

    /// Floating-point `(width, height)` of `level`, for cell arithmetic.
    #[inline]
    fn mip_size_f(&self, level: u32) -> Vec2 {
        let s = self.mip_size(level);
        Vec2::new(s.x as f32, s.y as f32)
    }
}

/// Outcome of a screen-space march.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HiZHit {
    /// Whether the march found a surface crossing.
    pub hit: bool,
    /// UV of the hit (or of the last marched position on a miss), clamped to
    /// `[0, 1]^2`.
    pub uv: Vec2,
    /// Depth at [`HiZHit::uv`].
    pub depth: f32,
    /// Number of march iterations consumed (the acceleration metric).
    pub iterations: u32,
}

/// Branchless sign that maps zero to `+1` (keeps a stationary axis moving
/// "forward" so boundary math stays well defined).
#[inline]
fn sign_pos(v: f32) -> f32 {
    if v >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Integer cell coordinate of a UV at a given float mip size.
#[inline]
fn cell_of(uv: Vec2, size: Vec2) -> Vec2 {
    Vec2::new(ops::floor(uv.x * size.x), ops::floor(uv.y * size.y))
}

/// Reciprocal that returns a large finite-signed value for a near-zero
/// denominator so downstream `min` selects the other, meaningful axis.
#[inline]
fn safe_delta(numer: f32, denom: f32) -> f32 {
    if denom.abs() > MIN_DEPTH_SLOPE {
        numer / denom
    } else {
        // Never crosses this axis: push its crossing-distance to +∞.
        f32::INFINITY
    }
}

/// Advances the ray (origin `o`, depth-parametrised direction `d` with
/// `d.z == 1`) to the first boundary of `cell` in the travel direction.
#[inline]
fn intersect_cell_boundary(
    o: Vec3,
    d: Vec3,
    cell: Vec2,
    cell_count: Vec2,
    cross_step: Vec2,
    cross_offset: Vec2,
) -> Vec3 {
    // For a positive step the far edge is `cell + 1`; for a negative step it is
    // the current `cell` index.  `max(cross_step, 0)` yields exactly that.
    let edge_index = cell + cross_step.max(Vec2::ZERO);
    let boundary = edge_index / cell_count + cross_offset;
    let dx = safe_delta(boundary.x - o.x, d.x);
    let dy = safe_delta(boundary.y - o.y, d.y);
    let t = dx.min(dy);
    o + d * t
}

/// Reconstructs the ray position at depth `z` (uses `d.z == 1`).
#[inline]
fn intersect_depth_plane(o: Vec3, d: Vec3, z: f32) -> Vec3 {
    o + d * (z - o.z)
}

/// Hierarchical (Hi-Z) screen-space march for the first surface crossing.
///
/// Marches the ray `origin_uv + s * dir_screen.xy` with depth
/// `origin_depth + s * dir_screen.z` through a [`HiZReduce::Min`] pyramid using
/// the classic cell-crossing descend/ascend scheme (Uludağ, *Hi-Z Screen-Space
/// Cone-Traced Reflections*, GPU Pro 5):
///
/// * At the current level the ray is advanced to the stored minimum-depth plane
///   of its cell.  If that intersection stays inside the cell, a surface may be
///   there, so the march descends to a finer level.
/// * Otherwise the ray crosses the cell boundary first (empty cell), so it is
///   advanced to the boundary and the march ascends to a coarser level, skipping
///   the empty space in one step.
///
/// A hit is reported when the march descends below level 0.  `dir_screen.z` must
/// have magnitude above [`MIN_DEPTH_SLOPE`]; a ray parallel to the depth planes
/// is reported as a deterministic miss with zero iterations.
///
/// The result is always finite and the UV is clamped to `[0, 1]^2`.
pub fn hiz_trace(
    pyramid: &HiZPyramid,
    origin_uv: Vec2,
    origin_depth: f32,
    dir_screen: Vec3,
    max_iterations: u32,
) -> HiZHit {
    let o = Vec3::new(origin_uv.x, origin_uv.y, origin_depth);
    let dz = dir_screen.z;
    if dz.abs() < MIN_DEPTH_SLOPE || !dir_screen.is_finite() || !o.is_finite() {
        return HiZHit {
            hit: false,
            uv: origin_uv.clamp(Vec2::ZERO, Vec2::ONE),
            depth: origin_depth,
            iterations: 0,
        };
    }

    // Parametrise by depth: `d.z == 1`, so `intersect_depth_plane` is exact.
    let d = dir_screen / dz;
    let cross_step = Vec2::new(sign_pos(dir_screen.x), sign_pos(dir_screen.y));
    let cross_offset = cross_step * CELL_EPSILON;

    let top = (pyramid.levels() - 1) as i32;
    let stop: i32 = 0;
    let mut level: i32 = 0;
    let mut iterations: u32 = 0;

    // Nudge off the origin so we start inside the first cell's far side.
    let start_size = pyramid.mip_size_f(0);
    let start_cell = cell_of(Vec2::new(o.x, o.y), start_size);
    let mut ray = intersect_cell_boundary(o, d, start_cell, start_size, cross_step, cross_offset);

    while level >= stop && iterations < max_iterations {
        let lvl = level as u32;
        let size = pyramid.mip_size_f(lvl);
        let old_cell = cell_of(Vec2::new(ray.x, ray.y), size);
        let fetch_cell = UVec2::new(
            old_cell.x.max(0.0) as u32,
            old_cell.y.max(0.0) as u32,
        );
        let min_z = pyramid.hiz_fetch(lvl, fetch_cell);

        // Candidate: slide to this cell's minimum-depth plane.
        let plane = intersect_depth_plane(o, d, min_z);
        // Is that plane ahead of the current ray along the travel direction?
        let plane_ahead = (plane.z - ray.z) * dz >= 0.0;

        let mut next = ray;
        let mut ascend = false;
        if plane_ahead {
            next = plane;
            let new_cell = cell_of(Vec2::new(next.x, next.y), size);
            if new_cell.x != old_cell.x || new_cell.y != old_cell.y {
                // The plane lies beyond this cell: cell is empty, skip it.
                ascend = true;
            }
        } else {
            // Surface is behind us within this cell: treat as empty, skip.
            ascend = true;
        }

        if ascend {
            next = intersect_cell_boundary(o, d, old_cell, size, cross_step, cross_offset);
            level = (level + 2).min(top);
        }

        ray = next;
        level -= 1;
        iterations += 1;
    }

    let hit = level < stop;
    let uv = Vec2::new(ray.x, ray.y).clamp(Vec2::ZERO, Vec2::ONE);
    HiZHit {
        hit,
        uv,
        depth: ray.z,
        iterations,
    }
}

/// Single-resolution baseline march, one level-0 texel per step.
///
/// Steps the ray along `dir_screen` in increments that advance roughly one
/// level-0 texel along the dominant screen axis, fetching the level-0 depth at
/// each sample.  A hit is reported the first time the ray's depth reaches or
/// passes the sampled surface depth while moving into the scene.  This is the
/// honest cost baseline that [`hiz_trace`] must beat (or match) in iterations.
pub fn linear_trace(
    pyramid: &HiZPyramid,
    origin_uv: Vec2,
    origin_depth: f32,
    dir_screen: Vec3,
    max_iterations: u32,
) -> HiZHit {
    let o = Vec3::new(origin_uv.x, origin_uv.y, origin_depth);
    if !dir_screen.is_finite() || !o.is_finite() {
        return HiZHit {
            hit: false,
            uv: origin_uv.clamp(Vec2::ZERO, Vec2::ONE),
            depth: origin_depth,
            iterations: 0,
        };
    }

    let base = pyramid.mip_size_f(0);
    // Pick a parameter step so the dominant axis advances one texel per step.
    let texel = Vec2::new(1.0 / base.x, 1.0 / base.y);
    let step_x = safe_delta(texel.x, dir_screen.x.abs());
    let step_y = safe_delta(texel.y, dir_screen.y.abs());
    let step = step_x.min(step_y);
    let step = if step.is_finite() && step > 0.0 {
        step
    } else {
        return HiZHit {
            hit: false,
            uv: origin_uv.clamp(Vec2::ZERO, Vec2::ONE),
            depth: origin_depth,
            iterations: 0,
        };
    };

    let dz = dir_screen.z;
    let mut iterations: u32 = 0;
    let mut s = step;
    while iterations < max_iterations {
        iterations += 1;
        let p = o + dir_screen * s;
        if p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 {
            return HiZHit {
                hit: false,
                uv: Vec2::new(p.x, p.y).clamp(Vec2::ZERO, Vec2::ONE),
                depth: p.z,
                iterations,
            };
        }
        let size = pyramid.mip_size(0);
        let cell = UVec2::new(
            (ops::floor(p.x * base.x) as u32).min(size.x - 1),
            (ops::floor(p.y * base.y) as u32).min(size.y - 1),
        );
        let surface = pyramid.hiz_fetch(0, cell);
        // Hit when the ray has reached/passed the surface while moving forward.
        if (p.z - surface) * dz >= 0.0 && (surface - origin_depth) * dz >= 0.0 {
            return HiZHit {
                hit: true,
                uv: Vec2::new(p.x, p.y).clamp(Vec2::ZERO, Vec2::ONE),
                depth: p.z,
                iterations,
            };
        }
        s += step;
    }

    HiZHit {
        hit: false,
        uv: origin_uv.clamp(Vec2::ZERO, Vec2::ONE),
        depth: origin_depth,
        iterations,
    }
}

/// A GGX-sampled reflected direction with its importance-sampling pdf.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GgxSample {
    /// Unit reflected direction in world space.
    pub direction: Vec3,
    /// Solid-angle pdf of [`GgxSample::direction`] under GGX importance sampling.
    /// A perfect mirror (roughness `≤` [`MIRROR_ROUGHNESS`]) reports `1.0`.
    pub pdf: f32,
}

/// Normalises `v`, falling back to `fallback` for a degenerate (zero) input.
#[inline]
fn normalize_or(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE && len_sq.is_finite() {
        v * len_sq.sqrt().recip()
    } else {
        fallback
    }
}

/// Mirror reflection of an incident direction about a unit normal:
/// `incident - 2 (n·incident) n`.
#[inline]
pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - 2.0 * normal.dot(incident) * normal
}

/// GGX half-vector importance sampling of a reflected direction.
///
/// * `incident` — unit direction of travel of the view ray *into* the surface
///   (camera → surface).
/// * `normal` — unit surface normal facing the camera.
/// * `roughness` — perceptual roughness; clamped to `[`[`MIN_ROUGHNESS`]`, 1]`
///   for a real lobe, or treated as a perfect mirror at or below
///   [`MIRROR_ROUGHNESS`].
/// * `u` — a stratified sample in `[0, 1)^2` (the explicit randomness).
///
/// Returns the reflected unit direction and its solid-angle pdf
/// `D(h) · (n·h) / (4 · |v·h|)`.  For a mirror the reflected direction is the
/// exact `reflect(incident, normal)` with pdf `1`.  All outputs are finite and
/// clamped; denominators are guarded so the pdf never diverges to `NaN`.
pub fn ggx_importance_sample(incident: Vec3, normal: Vec3, roughness: f32, u: Vec2) -> GgxSample {
    let n = normalize_or(normal, Vec3::Z);
    let v = normalize_or(incident, -n);

    if roughness <= MIRROR_ROUGHNESS {
        let dir = normalize_or(reflect(v, n), n);
        return GgxSample {
            direction: dir,
            pdf: 1.0,
        };
    }

    let r = roughness.clamp(MIN_ROUGHNESS, 1.0);
    let alpha = r * r;
    let a2 = alpha * alpha;

    // Clamp the sample off the open ends so the trig below never divides by 0.
    let u1 = u.x.clamp(0.0, 1.0 - 1.0e-6);
    let u2 = u.y.clamp(0.0, 1.0);

    // GGX NDF inversion: cos(theta_h) of the sampled half-vector.
    let cos_t = ((1.0 - u1) / (1.0 + (a2 - 1.0) * u1)).max(0.0).sqrt();
    let cos_t = cos_t.clamp(0.0, 1.0);
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let phi = core::f32::consts::TAU * u2;

    let h_local = [sin_t * ops::cos(phi), sin_t * ops::sin(phi), cos_t];
    let h = normalize_or(Vec3::from(world_from_local(h_local, [n.x, n.y, n.z])), n);

    let dir = normalize_or(reflect(v, h), reflect(v, n));

    let n_dot_h = n.dot(h).max(0.0);
    let denom = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    let d_ndf = a2 / (core::f32::consts::PI * (denom * denom).max(f32::MIN_POSITIVE));
    let pdf_h = d_ndf * n_dot_h;
    let v_dot_h = v.dot(h).abs().max(1.0e-6);
    let pdf = (pdf_h / (4.0 * v_dot_h)).max(0.0);

    GgxSample {
        direction: dir,
        pdf: if pdf.is_finite() { pdf } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a flat-wall depth image: every texel at `wall_depth`.
    fn flat_wall(base: UVec2, wall_depth: f32) -> Vec<f32> {
        alloc::vec![wall_depth; (base.x * base.y) as usize]
    }

    #[test]
    fn level_sizes_round_up_and_bottom_at_one() {
        let base = UVec2::new(64, 64);
        assert_eq!(HiZPyramid::full_level_count(base), 7);
        assert_eq!(HiZPyramid::level_size(base, 0), UVec2::new(64, 64));
        assert_eq!(HiZPyramid::level_size(base, 1), UVec2::new(32, 32));
        assert_eq!(HiZPyramid::level_size(base, 6), UVec2::new(1, 1));
        // Odd/non-square base rounds up.
        let odd = UVec2::new(5, 3);
        assert_eq!(HiZPyramid::level_size(odd, 1), UVec2::new(3, 2));
    }

    #[test]
    fn from_base_min_reduces_closest_depth() {
        let base = UVec2::new(2, 2);
        // Depths: closest is 0.2.
        let depth = alloc::vec![0.9, 0.5, 0.2, 0.7];
        let pyr = HiZPyramid::from_base(base, &depth, HiZReduce::Min, 2);
        assert_eq!(pyr.levels(), 2);
        assert_eq!(pyr.mip_size(1), UVec2::new(1, 1));
        assert!((pyr.hiz_fetch(1, UVec2::ZERO) - 0.2).abs() < 1e-7);
        // Max pyramid keeps the farthest.
        let pyr_max = HiZPyramid::from_base(base, &depth, HiZReduce::Max, 2);
        assert!((pyr_max.hiz_fetch(1, UVec2::ZERO) - 0.9).abs() < 1e-7);
    }

    #[test]
    fn from_mip_slices_matches_fetch() {
        let base = UVec2::new(2, 2);
        let l0 = [0.1f32, 0.2, 0.3, 0.4];
        let l1 = [0.1f32];
        let mips: [&[f32]; 2] = [&l0, &l1];
        let pyr = HiZPyramid::from_mip_slices(base, &mips);
        assert_eq!(pyr.levels(), 2);
        assert!((pyr.hiz_fetch(0, UVec2::new(1, 1)) - 0.4).abs() < 1e-7);
        assert!((pyr.hiz_fetch(1, UVec2::ZERO) - 0.1).abs() < 1e-7);
    }

    #[test]
    fn fetch_clamps_out_of_range() {
        let base = UVec2::new(4, 4);
        let depth = flat_wall(base, 0.5);
        let pyr = HiZPyramid::from_base(base, &depth, HiZReduce::Min, 3);
        // Over-large cell and level both clamp to a valid, finite value.
        let v = pyr.hiz_fetch(99, UVec2::new(999, 999));
        assert!(v.is_finite());
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn flat_wall_specular_hit_reaches_expected_uv() {
        let base = UVec2::new(64, 64);
        let depth = flat_wall(base, 0.6);
        let pyr = HiZPyramid::from_base(base, &depth, HiZReduce::Min, 7);

        let origin = Vec2::new(0.2, 0.5);
        let origin_depth = 0.1;
        // Reaches z = 0.6 at s = 1.0 → uv.x = 0.2 + 0.4 = 0.6.
        let dir = Vec3::new(0.4, 0.0, 0.5);
        let hit = hiz_trace(&pyr, origin, origin_depth, dir, 128);
        assert!(hit.hit, "expected a hit on the flat wall");
        assert!(
            (hit.uv.x - 0.6).abs() < 0.03,
            "hit uv.x {} not near 0.6",
            hit.uv.x
        );
        assert!((hit.uv.y - 0.5).abs() < 0.03, "hit uv.y {}", hit.uv.y);
        assert!((hit.depth - 0.6).abs() < 0.05, "hit depth {}", hit.depth);
    }

    #[test]
    fn hiz_never_costs_more_iterations_than_linear() {
        // Far background with a wall only on the right; lots of empty space to
        // skip, which is where the hierarchy wins.
        let base = UVec2::new(64, 64);
        let mut depth = flat_wall(base, 1.0);
        for y in 0..base.y {
            for x in 0..base.x {
                if x as f32 / base.x as f32 > 0.8 {
                    depth[(y * base.x + x) as usize] = 0.5;
                }
            }
        }
        let pyr = HiZPyramid::from_base(base, &depth, HiZReduce::Min, 7);

        let origin = Vec2::new(0.05, 0.5);
        let origin_depth = 0.1;
        let dir = Vec3::new(0.9, 0.0, 0.45);
        let hiz = hiz_trace(&pyr, origin, origin_depth, dir, 256);
        let lin = linear_trace(&pyr, origin, origin_depth, dir, 256);
        assert!(hiz.hit, "hi-z should find the wall");
        assert!(lin.hit, "linear should find the wall");
        assert!(
            hiz.iterations <= lin.iterations,
            "hi-z {} iters should be <= linear {} iters",
            hiz.iterations,
            lin.iterations
        );
    }

    #[test]
    fn degenerate_depth_slope_is_a_safe_miss() {
        let base = UVec2::new(16, 16);
        let depth = flat_wall(base, 0.5);
        let pyr = HiZPyramid::from_base(base, &depth, HiZReduce::Min, 5);
        let hit = hiz_trace(&pyr, Vec2::new(0.5, 0.5), 0.3, Vec3::new(1.0, 0.0, 0.0), 64);
        assert!(!hit.hit);
        assert_eq!(hit.iterations, 0);
        assert!(hit.uv.is_finite() && hit.depth.is_finite());
    }

    #[test]
    fn hiz_trace_is_deterministic() {
        let base = UVec2::new(32, 32);
        let depth = flat_wall(base, 0.7);
        let pyr = HiZPyramid::from_base(base, &depth, HiZReduce::Min, 6);
        let a = hiz_trace(&pyr, Vec2::new(0.1, 0.4), 0.1, Vec3::new(0.6, 0.1, 0.5), 128);
        let b = hiz_trace(&pyr, Vec2::new(0.1, 0.4), 0.1, Vec3::new(0.6, 0.1, 0.5), 128);
        assert_eq!(a, b);
    }

    #[test]
    fn mirror_roughness_degenerates_to_perfect_reflection() {
        let incident = Vec3::new(0.3, -1.0, 0.2).normalize();
        let normal = Vec3::Y;
        let s = ggx_importance_sample(incident, normal, 0.0, Vec2::new(0.37, 0.81));
        let expected = reflect(incident, normal);
        assert!(
            (s.direction - expected).length() < 1e-5,
            "mirror dir {:?} vs {:?}",
            s.direction,
            expected
        );
        assert!((s.pdf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ggx_sample_is_unit_and_finite_pdf() {
        let incident = Vec3::new(0.1, -1.0, 0.3).normalize();
        let normal = Vec3::Y;
        for i in 0..8u32 {
            for j in 0..8u32 {
                let u = Vec2::new((i as f32 + 0.5) / 8.0, (j as f32 + 0.5) / 8.0);
                let s = ggx_importance_sample(incident, normal, 0.4, u);
                assert!(
                    (s.direction.length() - 1.0).abs() < 1e-4,
                    "non-unit dir {:?}",
                    s.direction
                );
                assert!(s.pdf.is_finite() && s.pdf >= 0.0, "pdf {}", s.pdf);
            }
        }
    }

    #[test]
    fn ggx_at_normal_incidence_centre_sample_is_mirror() {
        // u.x = 0 forces cos(theta_h) = 1 → half-vector == normal → mirror.
        let incident = Vec3::new(0.0, -1.0, 0.0);
        let normal = Vec3::Y;
        let s = ggx_importance_sample(incident, normal, 0.3, Vec2::new(0.0, 0.5));
        let expected = reflect(incident, normal);
        assert!(
            (s.direction - expected).length() < 1e-4,
            "dir {:?} vs {:?}",
            s.direction,
            expected
        );
    }

    #[test]
    fn ggx_is_deterministic() {
        let incident = Vec3::new(0.2, -1.0, 0.1).normalize();
        let a = ggx_importance_sample(incident, Vec3::Y, 0.5, Vec2::new(0.25, 0.75));
        let b = ggx_importance_sample(incident, Vec3::Y, 0.5, Vec2::new(0.25, 0.75));
        assert_eq!(a, b);
    }
}
