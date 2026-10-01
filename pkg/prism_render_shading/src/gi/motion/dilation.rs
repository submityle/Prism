//! Velocity dilation (closest-depth / max-magnitude) — CPU golden.
//!
//! Raw per-pixel motion vectors are noisy at silhouettes: a background pixel
//! adjacent to a fast foreground object still carries the slow background
//! velocity, so TAA reprojects it to the *wrong* history and smears a ghost
//! along the edge.  **Dilation** fixes this by replacing each pixel's velocity
//! with a representative velocity gathered from its `3x3` neighbourhood.
//!
//! Two classic selection rules are provided:
//!
//! * **Closest-depth** — pick the velocity of the *nearest* neighbour (the one
//!   most likely to be the moving foreground), the standard choice for TAA
//!   history reprojection.  Depth ordering is explicit via [`DepthOrdering`] so
//!   both standard and reverse-Z buffers are supported.
//! * **Max-magnitude** — pick the neighbour with the largest velocity, used when
//!   the goal is to bound the motion (e.g. feeding motion-blur tile reduction).
//!
//! # Conventions
//! * A `3x3` neighbourhood is gathered with **clamp-to-edge** addressing, so
//!   border pixels simply repeat the frame edge rather than reading garbage.
//! * The neighbourhood is laid out row-major with the **centre at index 4**;
//!   ties (equal depth / equal magnitude) are resolved in favour of the centre,
//!   so a flat neighbourhood is a no-op.
//! * Depth semantics are caller-declared through [`DepthOrdering`]; non-finite
//!   depths are treated as *infinitely far* so they never win the "nearest"
//!   test.  Non-finite velocities are sanitized to zero.
//! * Every function is deterministic and never emits a `NaN`.  Grid helpers
//!   allocate their output `Vec` but perform no RNG/IO/GPU/unsafe work; a
//!   degenerate (zero) extent is floored to a single pixel.

use alloc::vec::Vec;
use bevy_math::Vec2;

/// Smallest grid extent (per axis) the dilation helpers will address.
const MIN_EXTENT: usize = 1;

/// Index of the centre tap within a row-major `3x3` neighbourhood.
const CENTER_INDEX: usize = 4;

/// Replaces any non-finite component of a velocity with `0.0`.
#[inline]
fn sanitize_velocity(v: Vec2) -> Vec2 {
    Vec2::new(
        if v.x.is_finite() { v.x } else { 0.0 },
        if v.y.is_finite() { v.y } else { 0.0 },
    )
}

/// Which depth value counts as "nearer" to the camera.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthOrdering {
    /// Smaller depth is nearer — standard `[0, 1]` or positive linear eye depth.
    SmallerIsNearer,
    /// Larger depth is nearer — reverse-Z depth buffers.
    LargerIsNearer,
}

impl DepthOrdering {
    /// Returns `true` when `candidate` is strictly nearer than `incumbent`
    /// under this ordering.
    ///
    /// Non-finite depths are mapped to the "farthest" sentinel for the active
    /// ordering, so a `NaN`/`inf` depth can never be judged nearer than a finite
    /// one and never wins a tie.
    #[inline]
    fn is_nearer(self, candidate: f32, incumbent: f32) -> bool {
        match self {
            DepthOrdering::SmallerIsNearer => {
                let c = if candidate.is_finite() { candidate } else { f32::INFINITY };
                let i = if incumbent.is_finite() { incumbent } else { f32::INFINITY };
                c < i
            }
            DepthOrdering::LargerIsNearer => {
                let c = if candidate.is_finite() { candidate } else { f32::NEG_INFINITY };
                let i = if incumbent.is_finite() { incumbent } else { f32::NEG_INFINITY };
                c > i
            }
        }
    }
}

/// A velocity paired with the depth of the surface it belongs to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VelocityDepth {
    /// Screen-space velocity (UV units), sanitized on construction via the
    /// selection helpers.
    pub velocity: Vec2,
    /// Scene depth; interpretation is governed by [`DepthOrdering`].
    pub depth: f32,
}

impl VelocityDepth {
    /// Creates a sample, sanitizing the velocity (non-finite -> zero).
    #[inline]
    pub fn new(velocity: Vec2, depth: f32) -> Self {
        Self {
            velocity: sanitize_velocity(velocity),
            depth,
        }
    }
}

/// Selects the velocity+depth of the **nearest** tap in a `3x3` neighbourhood.
///
/// Seeded with the centre tap so ties resolve to the centre; each other tap
/// replaces the incumbent only when strictly nearer under `ordering`.  The
/// returned velocity is always finite.
#[inline]
pub fn closest_depth_velocity(samples: &[VelocityDepth; 9], ordering: DepthOrdering) -> VelocityDepth {
    let mut best = VelocityDepth {
        velocity: sanitize_velocity(samples[CENTER_INDEX].velocity),
        depth: samples[CENTER_INDEX].depth,
    };
    for (i, s) in samples.iter().enumerate() {
        if i == CENTER_INDEX {
            continue;
        }
        if ordering.is_nearer(s.depth, best.depth) {
            best = VelocityDepth {
                velocity: sanitize_velocity(s.velocity),
                depth: s.depth,
            };
        }
    }
    best
}

/// Selects the **largest-magnitude** velocity in a `3x3` neighbourhood.
///
/// Seeded with the centre tap so ties resolve to the centre; comparison uses
/// squared length (no `sqrt`, exact ordering).  The result is always finite.
#[inline]
pub fn max_magnitude_velocity(samples: &[Vec2; 9]) -> Vec2 {
    let mut best = sanitize_velocity(samples[CENTER_INDEX]);
    let mut best_len2 = best.length_squared();
    for (i, &v) in samples.iter().enumerate() {
        if i == CENTER_INDEX {
            continue;
        }
        let v = sanitize_velocity(v);
        let len2 = v.length_squared();
        if len2 > best_len2 {
            best = v;
            best_len2 = len2;
        }
    }
    best
}

/// Floors a requested extent to at least one texel.
#[inline]
fn clamp_extent(n: usize) -> usize {
    n.max(MIN_EXTENT)
}

/// Clamps an integer coordinate into `[0, extent - 1]` (clamp-to-edge).
#[inline]
fn clamp_coord(c: i32, extent: usize) -> usize {
    if c < 0 {
        0
    } else {
        let c = c as usize;
        if c >= extent { extent - 1 } else { c }
    }
}

/// Gathers a clamp-to-edge `3x3` neighbourhood around `(x, y)` row-major.
///
/// The nine taps are returned in reading order (top-left first, centre at index
/// 4, bottom-right last); out-of-range coordinates are clamped to the edge.
#[inline]
fn gather_3x3<T, F>(x: usize, y: usize, width: usize, height: usize, read: &F) -> [T; 9]
where
    T: Copy,
    F: Fn(usize, usize) -> T,
{
    let tap = |dx: i32, dy: i32| {
        let sx = clamp_coord(x as i32 + dx, width);
        let sy = clamp_coord(y as i32 + dy, height);
        read(sx, sy)
    };
    [
        tap(-1, -1),
        tap(0, -1),
        tap(1, -1),
        tap(-1, 0),
        tap(0, 0),
        tap(1, 0),
        tap(-1, 1),
        tap(0, 1),
        tap(1, 1),
    ]
}

/// Dilates a velocity field by **closest depth** over every pixel.
///
/// `read` returns the [`VelocityDepth`] at an in-bounds pixel; it is invoked
/// with clamp-to-edge coordinates only.  The output is row-major, `width *
/// height`, with each entry the nearest-tap velocity for that pixel.  Extents
/// are floored to one texel.
pub fn dilate_closest_depth<F>(
    width: usize,
    height: usize,
    ordering: DepthOrdering,
    read: F,
) -> Vec<Vec2>
where
    F: Fn(usize, usize) -> VelocityDepth,
{
    let w = clamp_extent(width);
    let h = clamp_extent(height);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let taps = gather_3x3(x, y, w, h, &read);
            out.push(closest_depth_velocity(&taps, ordering).velocity);
        }
    }
    out
}

/// Dilates a velocity field by **max magnitude** over every pixel.
///
/// Mirrors [`dilate_closest_depth`] but selects the largest-magnitude velocity
/// in each `3x3` neighbourhood; useful as a pre-pass for motion-blur tiling.
pub fn dilate_max_magnitude<F>(width: usize, height: usize, read: F) -> Vec<Vec2>
where
    F: Fn(usize, usize) -> Vec2,
{
    let w = clamp_extent(width);
    let h = clamp_extent(height);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let taps = gather_3x3(x, y, w, h, &read);
            out.push(max_magnitude_velocity(&taps));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vd(vx: f32, vy: f32, depth: f32) -> VelocityDepth {
        VelocityDepth::new(Vec2::new(vx, vy), depth)
    }

    #[test]
    fn closest_depth_picks_nearer_smaller() {
        let mut s = [vd(0.0, 0.0, 10.0); 9];
        // One neighbour is much nearer and carries a distinct velocity.
        s[0] = vd(0.3, -0.1, 1.0);
        let out = closest_depth_velocity(&s, DepthOrdering::SmallerIsNearer);
        assert!((out.velocity - Vec2::new(0.3, -0.1)).length() < 1e-7);
        assert_eq!(out.depth, 1.0);
    }

    #[test]
    fn closest_depth_reverse_z_picks_larger() {
        let mut s = [vd(0.0, 0.0, 0.1); 9];
        s[8] = vd(-0.2, 0.4, 0.9);
        let out = closest_depth_velocity(&s, DepthOrdering::LargerIsNearer);
        assert!((out.velocity - Vec2::new(-0.2, 0.4)).length() < 1e-7);
    }

    #[test]
    fn closest_depth_tie_keeps_center() {
        // All depths equal: centre velocity must survive.
        let mut s = [vd(1.0, 0.0, 5.0); 9];
        s[CENTER_INDEX] = vd(0.0, 7.0, 5.0);
        let out = closest_depth_velocity(&s, DepthOrdering::SmallerIsNearer);
        assert!((out.velocity - Vec2::new(0.0, 7.0)).length() < 1e-7);
    }

    #[test]
    fn non_finite_depth_never_wins() {
        let mut s = [vd(0.0, 0.0, 10.0); 9];
        s[CENTER_INDEX] = vd(0.5, 0.5, 10.0);
        s[2] = vd(9.0, 9.0, f32::NAN);
        let out = closest_depth_velocity(&s, DepthOrdering::SmallerIsNearer);
        // NaN depth is treated as farthest; centre (finite, equal to others) wins.
        assert!((out.velocity - Vec2::new(0.5, 0.5)).length() < 1e-7);
    }

    #[test]
    fn max_magnitude_picks_largest() {
        let mut s = [Vec2::ZERO; 9];
        s[CENTER_INDEX] = Vec2::new(0.1, 0.0);
        s[6] = Vec2::new(0.0, 0.9);
        let out = max_magnitude_velocity(&s);
        assert!((out - Vec2::new(0.0, 0.9)).length() < 1e-7);
    }

    #[test]
    fn max_magnitude_tie_keeps_center() {
        let mut s = [Vec2::new(0.5, 0.0); 9];
        s[CENTER_INDEX] = Vec2::new(0.0, 0.5); // same magnitude as the rest
        let out = max_magnitude_velocity(&s);
        assert!((out - Vec2::new(0.0, 0.5)).length() < 1e-7);
    }

    #[test]
    fn max_magnitude_sanitizes_nan_velocity() {
        let mut s = [Vec2::ZERO; 9];
        s[1] = Vec2::new(f32::NAN, 2.0);
        s[CENTER_INDEX] = Vec2::new(0.1, 0.0);
        let out = max_magnitude_velocity(&s);
        assert!(out.is_finite());
        // The NaN tap collapses to (0,2) which beats the 0.1 centre.
        assert!((out - Vec2::new(0.0, 2.0)).length() < 1e-7);
    }

    #[test]
    fn dilate_grid_edge_clamp_and_shape() {
        // 3x3 grid; put a very near, fast pixel at the corner (0,0).
        let fast = Vec2::new(0.4, 0.0);
        let read = |x: usize, y: usize| {
            if x == 0 && y == 0 {
                VelocityDepth::new(fast, 0.0)
            } else {
                VelocityDepth::new(Vec2::ZERO, 100.0)
            }
        };
        let out = dilate_closest_depth(3, 3, DepthOrdering::SmallerIsNearer, read);
        assert_eq!(out.len(), 9);
        // Pixel (1,1) sees (0,0) in its neighbourhood -> inherits the fast vel.
        assert!((out[1 * 3 + 1] - fast).length() < 1e-7);
        // Pixel (2,2) does not touch (0,0) -> stays zero.
        assert!(out[2 * 3 + 2].length() < 1e-7);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn degenerate_extent_is_single_pixel() {
        let out = dilate_max_magnitude(0, 0, |_x, _y| Vec2::new(0.2, 0.0));
        assert_eq!(out.len(), 1);
        assert!((out[0] - Vec2::new(0.2, 0.0)).length() < 1e-7);
    }

    #[test]
    fn clamp_coord_edges() {
        assert_eq!(clamp_coord(-5, 4), 0);
        assert_eq!(clamp_coord(10, 4), 3);
        assert_eq!(clamp_coord(2, 4), 2);
    }
}
