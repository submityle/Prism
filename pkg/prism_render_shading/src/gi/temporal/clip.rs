//! Neighborhood color clipping (YCoCg AABB / variance) — CPU golden.
//!
//! The core of ghosting suppression in a temporal resolve is **neighborhood
//! clamping**: the reprojected history colour is constrained to the convex
//! bounds of the current frame's local neighborhood before it is blended in.
//! If the surface changed, history now lies far outside that neighborhood and
//! gets pulled back onto its surface, erasing the smear.
//!
//! Two refinements, both standard in AAA TAA, live here:
//!
//! * **YCoCg working space** — clamping in a luma/chroma basis
//!   ([`rgb_to_ycocg`] / [`ycocg_to_rgb`]) decorrelates intensity from colour,
//!   so a bright/dark change clamps on luma without introducing hue shifts.
//! * **Clip, don't clamp** — [`clip_to_aabb`] intersects the *ray* from the
//!   neighborhood mean toward history with the box surface, instead of a
//!   per-channel `clamp` that collapses the colour onto a box corner (which
//!   itself causes ghosting). The variance box ([`VarianceAabb`], mean ± γσ)
//!   gives a tighter, noise-aware bound than a raw min/max AABB.
//!
//! A [`luma_blend_weight`] helper provides the Karis-style luminance feedback
//! factor used to down-weight bright, flickery history.
//!
//! # Conventions
//! * The YCoCg transform is the lossless integer-friendly lift
//!   `Y = (R + 2G + B)/4`, `Co = (R - B)/2`, `Cg = (2G - R - B)/4`; its inverse
//!   is exact in real arithmetic (round-trip tested).
//! * `clip_to_aabb` returns history unchanged when it already lies inside the
//!   box, and otherwise the point where the mean→history segment pierces the
//!   box, so the result is always within the box (no overshoot, no corner
//!   collapse).
//! * All helpers are deterministic, allocation-free (no RNG/IO/GPU/unsafe),
//!   defend against zero-extent boxes and non-finite inputs, and never emit NaN.

use bevy_math::Vec3;

/// Rec.709 luma primaries, shared with the `denoise` sibling modules.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

/// Replaces a non-finite scalar with `fallback`.
#[inline]
fn finite_or(x: f32, fallback: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        fallback
    }
}

/// Replaces any non-finite component of a vector with `0.0`.
#[inline]
fn sanitize(v: Vec3) -> Vec3 {
    Vec3::new(finite_or(v.x, 0.0), finite_or(v.y, 0.0), finite_or(v.z, 0.0))
}

/// Converts a linear RGB colour to the YCoCg lift basis.
///
/// `x = Y` (luma), `y = Co` (orange–blue chroma), `z = Cg` (green–magenta
/// chroma).  The transform is exactly invertible by [`ycocg_to_rgb`].
#[inline]
pub fn rgb_to_ycocg(rgb: Vec3) -> Vec3 {
    let c = sanitize(rgb);
    let y = 0.25 * c.x + 0.5 * c.y + 0.25 * c.z;
    let co = 0.5 * c.x - 0.5 * c.z;
    let cg = -0.25 * c.x + 0.5 * c.y - 0.25 * c.z;
    Vec3::new(y, co, cg)
}

/// Inverse of [`rgb_to_ycocg`]: converts YCoCg back to linear RGB.
#[inline]
pub fn ycocg_to_rgb(ycocg: Vec3) -> Vec3 {
    let c = sanitize(ycocg);
    let tmp = c.x - c.z; // Y - Cg = (R + B) / 2
    let r = tmp + c.y;
    let g = c.x + c.z;
    let b = tmp - c.y;
    Vec3::new(r, g, b)
}

/// Rec.709 luminance of a *linear RGB* colour.
#[inline]
pub fn luminance(rgb: Vec3) -> f32 {
    let c = sanitize(rgb);
    LUMA_R * c.x + LUMA_G * c.y + LUMA_B * c.z
}

/// An axis-aligned bounding box in colour space (any basis), carrying its
/// center and half-extent so [`clip_to_aabb`] can work directly from them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorAabb {
    /// Per-channel minimum corner.
    pub min: Vec3,
    /// Per-channel maximum corner.
    pub max: Vec3,
}

impl ColorAabb {
    /// Builds a box from explicit corners, swapping any inverted axis so
    /// `min <= max` holds componentwise.
    #[inline]
    pub fn from_corners(a: Vec3, b: Vec3) -> Self {
        let a = sanitize(a);
        let b = sanitize(b);
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// Box center `(min + max) / 2`.
    #[inline]
    pub fn center(&self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    /// Non-negative half-extent `(max - min) / 2`.
    #[inline]
    pub fn half_extent(&self) -> Vec3 {
        (0.5 * (self.max - self.min)).max(Vec3::ZERO)
    }

    /// Returns `true` when `p` lies inside the box (inclusive) within a small
    /// epsilon on each axis.
    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        let p = sanitize(p);
        let eps = 1.0e-6;
        p.x >= self.min.x - eps
            && p.x <= self.max.x + eps
            && p.y >= self.min.y - eps
            && p.y <= self.max.y + eps
            && p.z >= self.min.z - eps
            && p.z <= self.max.z + eps
    }
}

/// Online accumulator for a neighborhood's first and second colour moments plus
/// its min/max corner, so both the raw AABB and the variance box can be built
/// from a single pass over the neighbors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeighborhoodStats {
    count: f32,
    m1: Vec3,
    m2: Vec3,
    lo: Vec3,
    hi: Vec3,
}

impl Default for NeighborhoodStats {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl NeighborhoodStats {
    /// An empty accumulator.
    pub const EMPTY: Self = Self {
        count: 0.0,
        m1: Vec3::ZERO,
        m2: Vec3::ZERO,
        lo: Vec3::splat(f32::INFINITY),
        hi: Vec3::splat(f32::NEG_INFINITY),
    };

    /// Folds one neighbor colour (already in the working basis) into the stats.
    #[inline]
    pub fn push(&mut self, c: Vec3) {
        let c = sanitize(c);
        self.count += 1.0;
        self.m1 += c;
        self.m2 += c * c;
        self.lo = self.lo.min(c);
        self.hi = self.hi.max(c);
    }

    /// Per-channel mean of the folded colours (zero when empty).
    #[inline]
    pub fn mean(&self) -> Vec3 {
        if self.count > 0.0 {
            self.m1 / self.count
        } else {
            Vec3::ZERO
        }
    }

    /// Per-channel population standard deviation (non-negative, zero when empty).
    #[inline]
    pub fn std_dev(&self) -> Vec3 {
        if self.count > 0.0 {
            let mean = self.mean();
            let var = (self.m2 / self.count - mean * mean).max(Vec3::ZERO);
            Vec3::new(var.x.sqrt(), var.y.sqrt(), var.z.sqrt())
        } else {
            Vec3::ZERO
        }
    }

    /// The raw min/max AABB over the folded colours.  Degenerates to a zero-size
    /// box at the origin when empty.
    #[inline]
    pub fn minmax_aabb(&self) -> ColorAabb {
        if self.count > 0.0 {
            ColorAabb {
                min: self.lo,
                max: self.hi,
            }
        } else {
            ColorAabb {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
            }
        }
    }

    /// The variance box `mean ± gamma * sigma`, the noise-aware clamp bound.
    ///
    /// `gamma` controls tightness (typical TAA values `~1.0`); it is clamped
    /// non-negative.  The returned box is intersected with the raw min/max AABB
    /// so it can never be *looser* than the true neighborhood extent.
    #[inline]
    pub fn variance_aabb(&self, gamma: f32) -> ColorAabb {
        let gamma = finite_or(gamma, 1.0).max(0.0);
        let mean = self.mean();
        let sigma = self.std_dev();
        let vmin = mean - gamma * sigma;
        let vmax = mean + gamma * sigma;
        let raw = self.minmax_aabb();
        let lo = vmin.max(raw.min);
        let hi = vmax.min(raw.max);
        // Floating-point intersection can invert a near-zero-extent box; keep
        // `min <= max` componentwise so downstream clamping stays well-defined.
        ColorAabb {
            min: lo.min(hi),
            max: lo.max(hi),
        }
    }
}

/// Clips `history` to `aabb` along the segment from `toward` (typically the
/// neighborhood mean) to `history`.
///
/// If `history` is already inside the box it is returned unchanged.  Otherwise
/// the segment `toward → history` is scaled down to the first box face it
/// crosses, so the result lies *on* the box surface in the direction of the
/// original history colour — preserving hue far better than a per-channel
/// clamp, which would snap to a corner.  A near-zero box extent collapses the
/// result to the box center.
#[inline]
pub fn clip_to_aabb(aabb: ColorAabb, toward: Vec3, history: Vec3) -> Vec3 {
    let center = aabb.center();
    let extent = aabb.half_extent();
    let history = sanitize(history);
    let toward = sanitize(toward);

    // Direction from the anchor point toward history.
    let dir = history - toward;

    // Parametric distance (0..1 along `toward -> history`) at which each axis
    // hits the box face measured from `center`. We solve, per axis:
    //   |toward.k + t*dir.k - center.k| = extent.k
    // and take the smallest positive t that is <= 1; t >= 1 means history is
    // inside along that axis.
    let mut t_min = 1.0_f32;
    let mut any_outside = false;

    for k in 0..3 {
        let d = dir[k];
        let e = extent[k];
        let from_center = toward[k] - center[k];

        if e <= 1.0e-8 {
            // Degenerate axis: only the single plane value `center[k]` is
            // admissible, so history is outside unless it already sits on it.
            if (from_center + d).abs() > e + 1.0e-6 {
                any_outside = true;
                if d.abs() > 1.0e-8 {
                    let t = (center[k] - toward[k]) / d;
                    if (0.0..=1.0).contains(&t) {
                        t_min = t_min.min(t);
                    }
                }
            }
            continue;
        }

        // Current (history) offset from center along this axis.
        let hist_off = from_center + d;
        if hist_off.abs() > e + 1.0e-6 {
            any_outside = true;
            if d.abs() > 1.0e-8 {
                // Target face: +extent if moving positive, -extent if negative.
                let face = if d > 0.0 { e } else { -e };
                let t = (face - from_center) / d;
                if (0.0..=1.0).contains(&t) {
                    t_min = t_min.min(t);
                }
            }
        }
    }

    if !any_outside {
        return history;
    }
    let t = t_min.clamp(0.0, 1.0);
    let clipped = toward + dir * t;
    // Numerical safety: hard-clamp onto the box in case of residual overshoot.
    // Order the bounds defensively so a float-inverted box cannot panic.
    let lo = aabb.min.min(aabb.max);
    let hi = aabb.min.max(aabb.max);
    Vec3::new(
        clipped.x.clamp(lo.x, hi.x),
        clipped.y.clamp(lo.y, hi.y),
        clipped.z.clamp(lo.z, hi.z),
    )
}

/// Karis-style luminance feedback weight for blending current vs. history.
///
/// Each colour is weighted by `1 / (1 + luma)` so bright, flickery samples
/// contribute less; the returned value is the normalized weight assigned to the
/// **current** sample in a two-sample blend (history gets `1 - w`).  Negative
/// luminances are clamped to zero so the weights stay in `(0, 1)`.
#[inline]
pub fn luma_blend_weight(current_rgb: Vec3, history_rgb: Vec3) -> f32 {
    let wc = 1.0 / (1.0 + luminance(current_rgb).max(0.0));
    let wh = 1.0 / (1.0 + luminance(history_rgb).max(0.0));
    let denom = wc + wh;
    if denom > 1.0e-8 {
        (wc / denom).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Convenience: clip `history_rgb` against the variance box of a neighborhood,
/// doing the YCoCg round-trip internally.
///
/// `stats` must already hold neighbor colours in **YCoCg** space (push
/// `rgb_to_ycocg(neighbor)`); `history_rgb` is given in linear RGB and the
/// clipped result is returned in linear RGB.
#[inline]
pub fn clip_history_ycocg(stats: &NeighborhoodStats, history_rgb: Vec3, gamma: f32) -> Vec3 {
    let aabb = stats.variance_aabb(gamma);
    let hist_y = rgb_to_ycocg(history_rgb);
    let clipped = clip_to_aabb(aabb, stats.mean(), hist_y);
    ycocg_to_rgb(clipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    fn approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1.0e-4
    }

    #[test]
    fn ycocg_roundtrip() {
        let colors = [
            Vec3::new(0.2, 0.5, 0.9),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.33, 0.33, 0.33),
            Vec3::new(2.5, 0.1, 4.0),
        ];
        for c in colors {
            let rt = ycocg_to_rgb(rgb_to_ycocg(c));
            assert!(approx(rt, c), "roundtrip failed for {c:?} -> {rt:?}");
        }
    }

    #[test]
    fn neutral_grey_has_zero_chroma() {
        let y = rgb_to_ycocg(Vec3::splat(0.7));
        assert!((y.y).abs() < EPS);
        assert!((y.z).abs() < EPS);
        assert!((y.x - 0.7).abs() < EPS);
    }

    #[test]
    fn stats_mean_and_std() {
        let mut s = NeighborhoodStats::default();
        s.push(Vec3::splat(1.0));
        s.push(Vec3::splat(3.0));
        assert!(approx(s.mean(), Vec3::splat(2.0)));
        assert!(approx(s.std_dev(), Vec3::splat(1.0)));
    }

    #[test]
    fn point_inside_box_unchanged() {
        let aabb = ColorAabb::from_corners(Vec3::splat(-1.0), Vec3::splat(1.0));
        let p = Vec3::new(0.2, -0.3, 0.5);
        let clipped = clip_to_aabb(aabb, Vec3::ZERO, p);
        assert!(approx(clipped, p));
    }

    #[test]
    fn point_outside_pulled_to_surface() {
        let aabb = ColorAabb::from_corners(Vec3::splat(-1.0), Vec3::splat(1.0));
        // History far along +x; clipping from center must land on the +x face.
        let p = Vec3::new(5.0, 0.0, 0.0);
        let clipped = clip_to_aabb(aabb, Vec3::ZERO, p);
        assert!(aabb.contains(clipped));
        assert!((clipped.x - 1.0).abs() < 1.0e-4, "x={}", clipped.x);
        // Direction preserved: y and z stay near zero.
        assert!(clipped.y.abs() < 1.0e-4 && clipped.z.abs() < 1.0e-4);
    }

    #[test]
    fn clip_preserves_direction_diagonal() {
        let aabb = ColorAabb::from_corners(Vec3::splat(-1.0), Vec3::splat(1.0));
        let p = Vec3::new(4.0, 2.0, 0.0);
        let clipped = clip_to_aabb(aabb, Vec3::ZERO, p);
        assert!(aabb.contains(clipped));
        // The clip point should be colinear with the center->history ray.
        // p is (4,2,0); hitting x=1 face at t=0.25 gives (1,0.5,0).
        assert!(approx(clipped, Vec3::new(1.0, 0.5, 0.0)));
    }

    #[test]
    fn variance_box_tighter_than_minmax() {
        let mut s = NeighborhoodStats::default();
        // One outlier should widen min/max more than mean±sigma.
        for v in [1.0, 1.0, 1.0, 1.0, 10.0] {
            s.push(Vec3::splat(v));
        }
        let raw = s.minmax_aabb();
        let var = s.variance_aabb(1.0);
        assert!(var.max.x <= raw.max.x + EPS);
        assert!(var.min.x >= raw.min.x - EPS);
    }

    #[test]
    fn variance_box_within_minmax_always() {
        let mut s = NeighborhoodStats::default();
        for v in [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6] {
            s.push(rgb_to_ycocg(Vec3::new(v, 1.0 - v, 0.5)));
        }
        let raw = s.minmax_aabb();
        let var = s.variance_aabb(1.0);
        // componentwise containment
        assert!(var.min.cmpge(raw.min).all());
        assert!(var.max.cmple(raw.max).all());
    }

    #[test]
    fn luma_weight_favours_darker() {
        // Bright history should get less weight -> current weight > 0.5.
        let w = luma_blend_weight(Vec3::splat(0.1), Vec3::splat(10.0));
        assert!(w > 0.5);
        assert!((0.0..=1.0).contains(&w));
    }

    #[test]
    fn clip_history_ycocg_pulls_ghost() {
        let mut s = NeighborhoodStats::default();
        // Neighborhood is dark grey.
        for _ in 0..9 {
            s.push(rgb_to_ycocg(Vec3::splat(0.1)));
        }
        // History is a bright white ghost.
        let ghost = Vec3::splat(1.0);
        let clipped = clip_history_ycocg(&s, ghost, 1.0);
        // Clipped luma must be far below the ghost's.
        assert!(luminance(clipped) < luminance(ghost));
    }

    #[test]
    fn degenerate_box_collapses_to_center() {
        let aabb = ColorAabb::from_corners(Vec3::splat(0.5), Vec3::splat(0.5));
        let clipped = clip_to_aabb(aabb, Vec3::splat(0.5), Vec3::new(5.0, -2.0, 3.0));
        assert!(approx(clipped, Vec3::splat(0.5)));
    }

    #[test]
    fn non_finite_safe() {
        let y = rgb_to_ycocg(Vec3::new(f32::NAN, 1.0, f32::INFINITY));
        assert!(y.x.is_finite() && y.y.is_finite() && y.z.is_finite());
    }

    #[test]
    fn deterministic() {
        let aabb = ColorAabb::from_corners(Vec3::splat(-1.0), Vec3::splat(1.0));
        let a = clip_to_aabb(aabb, Vec3::ZERO, Vec3::new(3.0, 1.0, -2.0));
        let b = clip_to_aabb(aabb, Vec3::ZERO, Vec3::new(3.0, 1.0, -2.0));
        assert_eq!(a, b);
    }
}
