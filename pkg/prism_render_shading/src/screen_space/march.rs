//! Hierarchical (Hi-Z) screen-space ray march.
//!
//! The tracer walks the reflection ray built by [`super::ray`] across a
//! min-reduction depth pyramid (the same reverse-Z "nearest depth" HZB the
//! visibility pass builds).  At each step it compares the interpolated ray
//! depth against the *nearest* scene depth stored for the current mip cell:
//!
//! * If the whole segment is nearer to the camera than everything in the cell,
//!   no surface can be there, so the ray skips the entire cell and climbs to a
//!   coarser mip — this is what makes the march sub-linear in screen distance.
//! * Otherwise the ray may cross a surface in the cell, so it descends to a
//!   finer mip; at the most detailed mip a straddle (or a thin-surface hit
//!   within the thickness tolerance) is reported as an intersection.
//!
//! Reverse-Z convention: device depth `1` is the near plane, `0` the far
//! plane, so a *larger* depth is *nearer* and the pyramid stores the per-cell
//! maximum.  Everything is expressed against a [`DepthPyramid`] so the CPU
//! golden and the WESL twin (`ssr.wesl`) trace identical geometry.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::{IVec2, UVec2, Vec2};

/// A reverse-Z "nearest depth" pyramid: each mip cell holds the maximum device
/// depth (nearest surface) of the finer texels it covers.
///
/// This is the CPU reference for the GPU HZB the tracer samples; tests build
/// one with [`DepthPyramid::from_nearest_reduction`] and the shader twin reads
/// the equivalent texture mips with `textureLoad`.
#[derive(Clone, Debug)]
pub struct DepthPyramid {
    mips: Vec<Vec<f32>>,
    sizes: Vec<UVec2>,
}

impl DepthPyramid {
    /// Builds the pyramid from a mip-0 device-depth grid in row-major order.
    ///
    /// Each coarser level is a 2x2 max-reduction of the level below (odd
    /// dimensions clamp the trailing column/row), so a coarse cell reports the
    /// nearest surface anywhere beneath it.  `size` is the mip-0 texel extent.
    pub fn from_nearest_reduction(mip0: &[f32], size: UVec2) -> Self {
        assert_eq!(
            mip0.len(),
            (size.x * size.y) as usize,
            "mip0 length must equal width * height",
        );
        let mut mips = vec![mip0.to_vec()];
        let mut sizes = vec![size];
        let mut current = size;
        while current.x > 1 || current.y > 1 {
            let next = UVec2::new(current.x.max(1).div_ceil(2), current.y.max(1).div_ceil(2));
            let prev = mips.last().expect("previous mip");
            let prev_size = *sizes.last().expect("previous size");
            let mut level = vec![0.0f32; (next.x * next.y) as usize];
            for y in 0..next.y {
                for x in 0..next.x {
                    let mut nearest = f32::NEG_INFINITY;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sx = (x * 2 + dx).min(prev_size.x - 1);
                            let sy = (y * 2 + dy).min(prev_size.y - 1);
                            let d = prev[(sy * prev_size.x + sx) as usize];
                            nearest = nearest.max(d);
                        }
                    }
                    level[(y * next.x + x) as usize] = nearest;
                }
            }
            mips.push(level);
            sizes.push(next);
            current = next;
        }
        Self { mips, sizes }
    }

    /// Number of mip levels, including mip 0.
    pub fn mip_count(&self) -> u32 {
        self.mips.len() as u32
    }

    /// Texel extent of `mip`.
    pub fn size(&self, mip: u32) -> UVec2 {
        self.sizes[(mip as usize).min(self.sizes.len() - 1)]
    }

    /// Nearest device depth (reverse-Z maximum) stored at `cell` on `mip`.
    ///
    /// Out-of-range mips clamp to the coarsest level and out-of-range cells
    /// clamp to the edge, matching a clamped `textureLoad`.
    pub fn nearest_depth(&self, mip: u32, cell: IVec2) -> f32 {
        let mip = (mip as usize).min(self.mips.len() - 1);
        let size = self.sizes[mip];
        let x = cell.x.clamp(0, size.x as i32 - 1) as u32;
        let y = cell.y.clamp(0, size.y as i32 - 1) as u32;
        self.mips[mip][(y * size.x + x) as usize]
    }
}

/// Tunables for the hierarchical march.
#[derive(Clone, Copy, Debug)]
pub struct SsrMarchConfig {
    /// Hard cap on trace iterations (guards against pathological rays).
    pub max_iterations: u32,
    /// Finest mip the trace refines down to (usually `0`).
    pub most_detailed_mip: u32,
    /// Device-depth tolerance for accepting a hit on a surface the ray passes
    /// just behind (thin-object handling).  Larger values accept reflections
    /// off thin geometry at the cost of over-occlusion.
    pub thickness: f32,
}

impl Default for SsrMarchConfig {
    fn default() -> Self {
        Self {
            max_iterations: 128,
            most_detailed_mip: 0,
            thickness: 0.02,
        }
    }
}

/// Result of a hierarchical march.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsrMarchResult {
    /// Whether an intersection was found.
    pub hit: bool,
    /// UV of the intersection (or the last sampled point on a miss).
    pub uv: Vec2,
    /// Device depth at the intersection.
    pub depth: f32,
    /// Normalized ray parameter `[0, 1]` at the intersection.
    pub travel: f32,
    /// Iterations consumed (useful for cost visualisation and tests).
    pub iterations: u32,
}

/// Linear interpolation of a scalar; UV and device depth are both affine in the
/// ray parameter because projection maps 3D lines to straight NDC lines.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smallest ray parameter `> t` (by at least `EPS`) at which the ray crosses
/// into the next integer texel cell at `size`, or `1.0` if it reaches the end
/// first.
fn next_cell_boundary(start: Vec2, delta: Vec2, size: UVec2, t: f32) -> f32 {
    const EPS: f32 = 1.0e-6;
    let sx = size.x.max(1) as f32;
    let sy = size.y.max(1) as f32;
    let mut best = 1.0f32;

    // Axis X.
    let dx = delta.x * sx;
    if dx.abs() > EPS {
        let pos = (start.x + delta.x * t) * sx;
        let boundary = if dx > 0.0 {
            libm_floorf(pos) + 1.0
        } else {
            libm_ceilf(pos) - 1.0
        };
        // Solve (start.x + t'*delta.x) * sx == boundary.
        let t_axis = (boundary / sx - start.x) / delta.x;
        if t_axis > t + EPS && t_axis < best {
            best = t_axis;
        }
    }
    // Axis Y.
    let dy = delta.y * sy;
    if dy.abs() > EPS {
        let pos = (start.y + delta.y * t) * sy;
        let boundary = if dy > 0.0 {
            libm_floorf(pos) + 1.0
        } else {
            libm_ceilf(pos) - 1.0
        };
        let t_axis = (boundary / sy - start.y) / delta.y;
        if t_axis > t + EPS && t_axis < best {
            best = t_axis;
        }
    }
    best.min(1.0)
}

fn libm_floorf(x: f32) -> f32 {
    bevy_math::ops::floor(x)
}

fn libm_ceilf(x: f32) -> f32 {
    -bevy_math::ops::floor(-x)
}

/// Marches the screen-space `start -> end` ray across the depth `pyramid`,
/// returning the first intersection (or the terminating miss).
///
/// `start_uv`/`end_uv` and `start_depth`/`end_depth` are the projected ray
/// endpoints; UV and depth interpolate linearly in the returned `travel`.
pub fn march_hierarchical(
    pyramid: &DepthPyramid,
    start_uv: Vec2,
    start_depth: f32,
    end_uv: Vec2,
    end_depth: f32,
    config: SsrMarchConfig,
) -> SsrMarchResult {
    let delta = end_uv - start_uv;
    let base = pyramid.mip_count().saturating_sub(1);
    let most_detailed = config.most_detailed_mip.min(base);

    let mut level = most_detailed;
    let mut t = 0.0f32;
    let mut iterations = 0u32;

    let depth_at = |t: f32| lerp(start_depth, end_depth, t);
    let uv_at = |t: f32| start_uv + delta * t;

    while iterations < config.max_iterations && t < 1.0 {
        iterations += 1;
        let uv = uv_at(t);
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            // Ran off the framebuffer: nothing to reflect.
            return SsrMarchResult {
                hit: false,
                uv,
                depth: depth_at(t),
                travel: t,
                iterations,
            };
        }

        let size = pyramid.size(level);
        let cell = IVec2::new(
            (uv.x * size.x as f32) as i32,
            (uv.y * size.y as f32) as i32,
        );
        let t_next = next_cell_boundary(start_uv, delta, size, t);

        let depth_near = depth_at(t); // largest ray depth (nearest) over segment
        let depth_far = depth_at(t_next); // smallest ray depth (farthest)
        let cell_nearest = pyramid.nearest_depth(level, cell); // nearest surface

        // Reverse-Z: the ray is entirely in front of the cell's nearest
        // surface when even its farthest (smallest) depth exceeds it.
        if depth_far > cell_nearest {
            // Skip the whole cell and climb to a coarser mip.
            t = t_next;
            if level < base {
                level += 1;
            }
            continue;
        }

        if level > most_detailed {
            // The cell may hold a surface; refine without advancing.
            level -= 1;
            continue;
        }

        // Most detailed mip: resolve the crossing against the texel's surface.
        if depth_near >= cell_nearest {
            // Segment straddles the surface: solve the exact crossing.
            let span = depth_near - depth_far;
            let t_hit = if span > 1.0e-8 {
                t + (depth_near - cell_nearest) / span * (t_next - t)
            } else {
                t
            };
            return SsrMarchResult {
                hit: true,
                uv: uv_at(t_hit),
                depth: cell_nearest,
                travel: t_hit,
                iterations,
            };
        }

        // Segment is entirely behind the surface; accept only thin overhangs
        // within the thickness tolerance, otherwise keep marching past it.
        if cell_nearest - depth_near <= config.thickness {
            return SsrMarchResult {
                hit: true,
                uv,
                depth: cell_nearest,
                travel: t,
                iterations,
            };
        }
        t = t_next;
    }

    SsrMarchResult {
        hit: false,
        uv: uv_at(t.min(1.0)),
        depth: depth_at(t.min(1.0)),
        travel: t.min(1.0),
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_wall(size: UVec2, depth: f32) -> DepthPyramid {
        let mip0 = vec![depth; (size.x * size.y) as usize];
        DepthPyramid::from_nearest_reduction(&mip0, size)
    }

    #[test]
    fn pyramid_reduction_keeps_nearest_surface() {
        // 2x2 grid; the coarse mip must report the nearest (max) depth.
        let mip0 = vec![0.1, 0.9, 0.4, 0.2];
        let pyramid = DepthPyramid::from_nearest_reduction(&mip0, UVec2::new(2, 2));
        assert_eq!(pyramid.mip_count(), 2);
        assert!((pyramid.nearest_depth(1, IVec2::ZERO) - 0.9).abs() < 1.0e-6);
    }

    #[test]
    fn ray_in_front_of_everything_misses() {
        // A wall at depth 0.2 (far); the ray stays nearer (0.8..0.7) the whole
        // way, so it should run off the end without a hit.
        let pyramid = flat_wall(UVec2::new(16, 16), 0.2);
        let result = march_hierarchical(
            &pyramid,
            Vec2::new(0.1, 0.5),
            0.8,
            Vec2::new(0.9, 0.5),
            0.7,
            SsrMarchConfig::default(),
        );
        assert!(!result.hit);
    }

    #[test]
    fn ray_crossing_a_wall_hits_it() {
        // Wall at depth 0.5; ray descends from 0.8 (near) to 0.2 (far), so it
        // must cross the wall partway across the screen.
        let pyramid = flat_wall(UVec2::new(64, 64), 0.5);
        let result = march_hierarchical(
            &pyramid,
            Vec2::new(0.1, 0.5),
            0.8,
            Vec2::new(0.9, 0.5),
            0.2,
            SsrMarchConfig::default(),
        );
        assert!(result.hit, "expected an intersection with the wall");
        assert!((result.depth - 0.5).abs() < 1.0e-3);
        // The crossing is where ray depth == 0.5: halfway along 0.8->0.2.
        assert!((result.travel - 0.5).abs() < 0.05);
    }

    #[test]
    fn hierarchical_skip_is_cheaper_than_the_iteration_cap() {
        // A long empty run should terminate via mip skipping well under the
        // per-texel iteration count of a naive linear march.
        let pyramid = flat_wall(UVec2::new(512, 512), 0.05);
        let result = march_hierarchical(
            &pyramid,
            Vec2::new(0.02, 0.5),
            0.9,
            Vec2::new(0.98, 0.5),
            0.8,
            SsrMarchConfig::default(),
        );
        assert!(!result.hit);
        // 512 texels wide, but coarse mips let us bail in far fewer steps.
        assert!(result.iterations < 64, "iterations = {}", result.iterations);
    }

    #[test]
    fn foreground_block_reflects_before_the_background() {
        // Left half is a near block (depth 0.7), right half is far (0.1). A ray
        // descending from 0.75 should hit the near block early on the left.
        let size = UVec2::new(64, 1);
        let mut mip0 = vec![0.1f32; 64];
        for pixel in mip0.iter_mut().take(20) {
            *pixel = 0.7;
        }
        let pyramid = DepthPyramid::from_nearest_reduction(&mip0, size);
        let result = march_hierarchical(
            &pyramid,
            Vec2::new(0.05, 0.5),
            0.75,
            Vec2::new(0.95, 0.5),
            0.05,
            SsrMarchConfig::default(),
        );
        assert!(result.hit);
        // Hit should land within the near block on the left third.
        assert!(result.uv.x < 0.4, "hit uv.x = {}", result.uv.x);
    }
}
