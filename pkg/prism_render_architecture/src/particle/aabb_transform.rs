//! Affine `AABB` transform and set-algebra contract for the particle bounds
//! pipeline (design §12, §13).
//!
//! [`super::bounds`] already owns the parallel `AABB` *reduction* (folding a
//! live-particle pool down to one box) plus velocity expansion. This module is
//! orthogonal and self-contained: it owns the *matrix transform* of a box under
//! a column-major affine `mat4` and the fuller set algebra
//! (`intersection` / `contains` / `overlaps` / `volume`) that frustum culling,
//! spatial queries, and `BVH` refit consume — none of which the reduction file
//! provides. To stay a single self-contained contract it defines its own
//! [`Aabb`] rather than depending on the reduction module's box.
//!
//! The transform uses Arvo's method: a box centered at `c` with half-extent `h`
//! maps under an affine matrix `M` to a new box centered at `M * c` whose
//! half-extent along axis `i` is `sum_j |M_ij| * h_j`. For a centered box this
//! is the exact tight `AABB`, so [`Aabb::transform_by_mat4`] agrees with the
//! brute-force 8-corner reference [`Aabb::transform_corners`] up to rounding.
//!
//! Everything is pure `min` / `max` / multiply-add and `f32::abs`: no
//! transcendental function is ever called, so the `CPU` result is deterministic
//! and a future `GPU` kernel can match it. `GPU` packing follows the shared
//! `std430` `vec4` alignment from [`super::gpu_layout`]: a box is two `vec4`
//! slots (`min.xyz` + pad, `max.xyz` + pad).

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Absolute tolerance for `f32` equality comparisons in this module.
///
/// Production code never uses floating-point `==` / `!=`; only the tests below
/// compare `(a - b).abs() < CMP_EPS`, so the constant is test-only.
#[cfg(test)]
const CMP_EPS: f32 = 1e-5;

/// Total `std430` byte size of one transformed box: two `vec4` slots holding
/// `min.xyz` + pad and `max.xyz` + pad.
pub const AABB_TRANSFORM_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// A self-contained axis-aligned bounding box (`AABB`) in some affine space.
///
/// Holds `f32` fields, so it derives [`PartialEq`] but not [`Eq`]. An *empty*
/// box seeds `min` with `f32::MAX` and `max` with `f32::MIN`, so the first
/// [`Aabb::expand_point`] on any axis always wins and the empty box is the
/// identity element of [`Aabb::union`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Per-axis minimum corner.
    pub min: [f32; 3],
    /// Per-axis maximum corner.
    pub max: [f32; 3],
}

impl Aabb {
    /// Builds a box directly from its `min` and `max` corners.
    #[must_use]
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Returns the empty box: `min` at `+INF` semantics (`f32::MAX`) and `max`
    /// at `-INF` semantics (`f32::MIN`), the identity for [`Aabb::union`].
    #[must_use]
    pub fn empty() -> Self {
        Self {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        }
    }

    /// Builds a box from its `center` and full `extent` (edge lengths). The
    /// half-extent is `extent * 0.5`; a negative extent yields an empty box.
    #[must_use]
    pub fn from_center_extent(center: [f32; 3], extent: [f32; 3]) -> Self {
        let hx = extent[0] * 0.5;
        let hy = extent[1] * 0.5;
        let hz = extent[2] * 0.5;
        Self {
            min: [center[0] - hx, center[1] - hy, center[2] - hz],
            max: [center[0] + hx, center[1] + hy, center[2] + hz],
        }
    }

    /// Returns `true` when the box holds no points, i.e. any axis has
    /// `min > max` (never uses `==`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0] || self.min[1] > self.max[1] || self.min[2] > self.max[2]
    }

    /// Returns the box center (midpoint of `min` and `max`).
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Returns the per-axis half-extent (half the edge lengths).
    #[must_use]
    pub fn half_extent(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }

    /// Returns the per-axis full extent (edge lengths, `max - min`).
    #[must_use]
    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Grows the box in place so it contains the point `p`.
    pub fn expand_point(&mut self, p: [f32; 3]) {
        self.min[0] = self.min[0].min(p[0]);
        self.min[1] = self.min[1].min(p[1]);
        self.min[2] = self.min[2].min(p[2]);
        self.max[0] = self.max[0].max(p[0]);
        self.max[1] = self.max[1].max(p[1]);
        self.max[2] = self.max[2].max(p[2]);
    }

    /// Returns the smallest box containing both `self` and `other`.
    #[must_use]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Returns the overlap box of `self` and `other`, or `None` when they are
    /// disjoint on any axis. A face- or edge-touch yields a valid degenerate
    /// (zero-thickness) box rather than `None`.
    #[must_use]
    pub fn intersection(&self, other: &Aabb) -> Option<Aabb> {
        let lo = [
            self.min[0].max(other.min[0]),
            self.min[1].max(other.min[1]),
            self.min[2].max(other.min[2]),
        ];
        let hi = [
            self.max[0].min(other.max[0]),
            self.max[1].min(other.max[1]),
            self.max[2].min(other.max[2]),
        ];
        if lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2] {
            None
        } else {
            Some(Aabb { min: lo, max: hi })
        }
    }

    /// Returns `true` when the point `p` lies inside the closed box (boundary
    /// included). Uses [`RangeInclusive::contains`] so no bare `>=` / `<=` chain
    /// is written by hand.
    #[must_use]
    pub fn contains_point(&self, p: [f32; 3]) -> bool {
        (self.min[0]..=self.max[0]).contains(&p[0])
            && (self.min[1]..=self.max[1]).contains(&p[1])
            && (self.min[2]..=self.max[2]).contains(&p[2])
    }

    /// Returns `true` when `other` is fully contained in `self` (boundary
    /// touching counts as contained).
    #[must_use]
    pub fn contains_aabb(&self, other: &Aabb) -> bool {
        self.min[0] <= other.min[0]
            && self.min[1] <= other.min[1]
            && self.min[2] <= other.min[2]
            && other.max[0] <= self.max[0]
            && other.max[1] <= self.max[1]
            && other.max[2] <= self.max[2]
    }

    /// Returns `true` when `self` and `other` overlap (share any volume, or
    /// touch on a face/edge/corner).
    #[must_use]
    pub fn overlaps(&self, other: &Aabb) -> bool {
        self.min[0] <= other.max[0]
            && other.min[0] <= self.max[0]
            && self.min[1] <= other.max[1]
            && other.min[1] <= self.max[1]
            && self.min[2] <= other.max[2]
            && other.min[2] <= self.max[2]
    }

    /// Returns the total surface area `2 * (dx*dy + dy*dz + dz*dx)`, using only
    /// multiply-add (no transcendental function). Handy as a `SAH` cost term.
    #[must_use]
    pub fn surface_area(&self) -> f32 {
        let d = self.extent();
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }

    /// Returns the enclosed volume `dx * dy * dz`.
    #[must_use]
    pub fn volume(&self) -> f32 {
        let d = self.extent();
        d[0] * d[1] * d[2]
    }

    /// Returns a copy grown outward by `m` on every axis (a shrink for a
    /// negative `m`).
    #[must_use]
    pub fn expand_margin(&self, m: f32) -> Aabb {
        Aabb {
            min: [self.min[0] - m, self.min[1] - m, self.min[2] - m],
            max: [self.max[0] + m, self.max[1] + m, self.max[2] + m],
        }
    }

    /// Builds the tight box over a point cloud. An empty slice yields the empty
    /// box (identity), never a `NaN`.
    #[must_use]
    pub fn from_points(points: &[[f32; 3]]) -> Aabb {
        let mut b = Aabb::empty();
        for &p in points {
            b.expand_point(p);
        }
        b
    }

    /// Folds a slice of boxes into their union. An empty slice yields the empty
    /// box (identity of [`Aabb::union`]).
    #[must_use]
    pub fn merge_slice(boxes: &[Aabb]) -> Aabb {
        let mut b = Aabb::empty();
        for item in boxes {
            b = b.union(item);
        }
        b
    }

    /// Transforms the box by a column-major affine `mat4` using Arvo's method.
    ///
    /// `m` is column-major: `m[c]` is column `c`, so matrix element
    /// `(row i, col j)` is `m[j][i]` and the translation is the fourth column
    /// `m[3][0..3]`. The new center is `M * center`; the new half-extent along
    /// axis `i` is `sum_j |m[j][i]| * half[j]`, which is the exact tight box for
    /// a centered input. An empty box maps to the empty box.
    #[must_use]
    pub fn transform_by_mat4(&self, m: &[[f32; 4]; 4]) -> Aabb {
        if self.is_empty() {
            return Aabb::empty();
        }
        let c = self.center();
        let h = self.half_extent();
        let (cx, hx) = affine_row(m, 0, c, h);
        let (cy, hy) = affine_row(m, 1, c, h);
        let (cz, hz) = affine_row(m, 2, c, h);
        Aabb {
            min: [cx - hx, cy - hy, cz - hz],
            max: [cx + hx, cy + hy, cz + hz],
        }
    }

    /// Transforms all eight corners by the affine `mat4` and returns their tight
    /// box. This is the brute-force reference [`Aabb::transform_by_mat4`] must
    /// agree with; it is exact for any affine map. An empty box maps to empty.
    #[must_use]
    pub fn transform_corners(&self, m: &[[f32; 4]; 4]) -> Aabb {
        if self.is_empty() {
            return Aabb::empty();
        }
        let lo = self.min;
        let hi = self.max;
        let mut out = Aabb::empty();
        for corner in 0u32..8 {
            let p = [
                if corner & 0b001 != 0 { hi[0] } else { lo[0] },
                if corner & 0b010 != 0 { hi[1] } else { lo[1] },
                if corner & 0b100 != 0 { hi[2] } else { lo[2] },
            ];
            out.expand_point(transform_point(m, p));
        }
        out
    }

    /// Packs the box into its `std430` byte record: `min.xyz` in the first
    /// `vec4` slot (with a zero pad lane) and `max.xyz` in the second.
    #[must_use]
    pub fn to_std430(&self) -> [u8; AABB_TRANSFORM_STD430_SIZE] {
        let mut bytes = [0u8; AABB_TRANSFORM_STD430_SIZE];
        write_vec3(&mut bytes[0..VEC4_STRIDE], self.min);
        write_vec3(&mut bytes[VEC4_STRIDE..2 * VEC4_STRIDE], self.max);
        bytes
    }
}

/// Total `std430` byte size of a storage buffer holding `count` transformed
/// boxes, clamped up to a single element (a `WebGPU` binding may not be empty).
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(AABB_TRANSFORM_STD430_SIZE, count)
}

/// Computes the transformed center/half-extent contribution for output axis
/// `i` under column-major affine `m`: `(m[3][i] + sum_j m[j][i]*c[j],
/// sum_j |m[j][i]|*h[j])`.
fn affine_row(m: &[[f32; 4]; 4], i: usize, c: [f32; 3], h: [f32; 3]) -> (f32, f32) {
    let mut ci = m[3][i];
    let mut hi = 0.0;
    for (j, (&cj, &hj)) in c.iter().zip(h.iter()).enumerate() {
        let e = m[j][i];
        ci += e * cj;
        hi += e.abs() * hj;
    }
    (ci, hi)
}

/// Applies the full column-major affine `m` to a point `p` (rotation/scale plus
/// the fourth-column translation).
fn transform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let mut out = [m[3][0], m[3][1], m[3][2]];
    for (j, &pj) in p.iter().enumerate() {
        out[0] += m[j][0] * pj;
        out[1] += m[j][1] * pj;
        out[2] += m[j][2] * pj;
    }
    out
}

/// Writes a `vec3` as three little-endian `f32` lanes into the front of a
/// `vec4`-sized slot, leaving the trailing pad lane untouched (zero).
fn write_vec3(slot: &mut [u8], v: [f32; 3]) {
    for (chunk, &component) in slot.chunks_mut(4).zip(v.iter()) {
        chunk.copy_from_slice(&component.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    fn approx_box(a: &Aabb, b: &Aabb) -> bool {
        approx3(a.min, b.min) && approx3(a.max, b.max)
    }

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    fn f32_from_le(bytes: &[u8]) -> f32 {
        let mut b = [0u8; 4];
        b.copy_from_slice(bytes);
        f32::from_le_bytes(b)
    }

    #[test]
    fn new_stores_corners() {
        let b = Aabb::new([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
        assert!(approx3(b.min, [-1.0, -2.0, -3.0]));
        assert!(approx3(b.max, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn empty_box_is_empty() {
        let b = Aabb::empty();
        assert!(b.is_empty());
    }

    #[test]
    fn from_center_extent_roundtrips_center_and_extent() {
        let b = Aabb::from_center_extent([2.0, -1.0, 4.0], [4.0, 2.0, 6.0]);
        assert!(approx3(b.center(), [2.0, -1.0, 4.0]));
        assert!(approx3(b.extent(), [4.0, 2.0, 6.0]));
        assert!(approx3(b.half_extent(), [2.0, 1.0, 3.0]));
    }

    #[test]
    fn is_empty_true_when_min_exceeds_max() {
        let b = Aabb::new([1.0, 0.0, 0.0], [-1.0, 1.0, 1.0]);
        assert!(b.is_empty());
        let ok = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(!ok.is_empty());
    }

    #[test]
    fn center_is_midpoint() {
        let b = Aabb::new([0.0, 0.0, 0.0], [2.0, 4.0, 8.0]);
        assert!(approx3(b.center(), [1.0, 2.0, 4.0]));
    }

    #[test]
    fn extent_and_half_extent_agree() {
        let b = Aabb::new([-3.0, -3.0, -3.0], [3.0, 3.0, 3.0]);
        assert!(approx3(b.extent(), [6.0, 6.0, 6.0]));
        assert!(approx3(b.half_extent(), [3.0, 3.0, 3.0]));
    }

    #[test]
    fn expand_point_grows_box() {
        let mut b = Aabb::empty();
        b.expand_point([1.0, 2.0, 3.0]);
        b.expand_point([-1.0, 5.0, 0.0]);
        assert!(approx3(b.min, [-1.0, 2.0, 0.0]));
        assert!(approx3(b.max, [1.0, 5.0, 3.0]));
    }

    #[test]
    fn union_contains_both() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([2.0, -1.0, 0.5], [3.0, 0.0, 4.0]);
        let u = a.union(&b);
        assert!(approx3(u.min, [0.0, -1.0, 0.0]));
        assert!(approx3(u.max, [3.0, 1.0, 4.0]));
        assert!(u.contains_aabb(&a));
        assert!(u.contains_aabb(&b));
    }

    #[test]
    fn union_with_empty_is_identity() {
        let a = Aabb::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        let u = Aabb::empty().union(&a);
        assert!(approx_box(&u, &a));
    }

    #[test]
    fn intersection_of_overlapping_boxes() {
        let a = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = Aabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        let x = a.intersection(&b).expect("boxes overlap");
        assert!(approx3(x.min, [1.0, 1.0, 1.0]));
        assert!(approx3(x.max, [2.0, 2.0, 2.0]));
    }

    #[test]
    fn intersection_of_disjoint_boxes_is_none() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn intersection_touching_is_degenerate_some() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
        let x = a.intersection(&b).expect("boxes touch on a face");
        assert!(approx(x.extent()[0], 0.0));
        assert!(approx(x.min[0], 1.0));
        assert!(approx(x.max[0], 1.0));
    }

    #[test]
    fn contains_point_inside_and_on_boundary() {
        let b = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert!(b.contains_point([1.0, 1.0, 1.0]));
        assert!(b.contains_point([0.0, 2.0, 1.0]));
    }

    #[test]
    fn contains_point_outside() {
        let b = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert!(!b.contains_point([3.0, 1.0, 1.0]));
        assert!(!b.contains_point([1.0, -0.5, 1.0]));
    }

    #[test]
    fn contains_aabb_true_and_false() {
        let outer = Aabb::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let inner = Aabb::new([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
        let poking = Aabb::new([1.0, 1.0, 1.0], [11.0, 2.0, 2.0]);
        assert!(outer.contains_aabb(&inner));
        assert!(!outer.contains_aabb(&poking));
    }

    #[test]
    fn overlaps_true_and_false() {
        let a = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = Aabb::new([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
        let c = Aabb::new([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn surface_area_of_unit_cube() {
        let b = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(approx(b.surface_area(), 6.0));
    }

    #[test]
    fn volume_of_box() {
        let b = Aabb::new([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        assert!(approx(b.volume(), 24.0));
    }

    #[test]
    fn expand_margin_grows_and_shrinks() {
        let b = Aabb::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let grown = b.expand_margin(1.0);
        assert!(approx3(grown.min, [-1.0, -1.0, -1.0]));
        assert!(approx3(grown.max, [3.0, 3.0, 3.0]));
        let shrunk = b.expand_margin(-0.5);
        assert!(approx3(shrunk.min, [0.5, 0.5, 0.5]));
        assert!(approx3(shrunk.max, [1.5, 1.5, 1.5]));
    }

    #[test]
    fn from_points_bounds_cloud() {
        let pts = [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0], [0.0, 0.0, 0.0]];
        let b = Aabb::from_points(&pts);
        assert!(approx3(b.min, [-4.0, 0.0, -6.0]));
        assert!(approx3(b.max, [1.0, 5.0, 3.0]));
    }

    #[test]
    fn from_points_empty_slice_is_empty_box() {
        let b = Aabb::from_points(&[]);
        assert!(b.is_empty());
    }

    #[test]
    fn merge_slice_unions_all() {
        let boxes = [
            Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Aabb::new([-2.0, 3.0, 0.0], [0.0, 4.0, 2.0]),
        ];
        let m = Aabb::merge_slice(&boxes);
        assert!(approx3(m.min, [-2.0, 0.0, 0.0]));
        assert!(approx3(m.max, [1.0, 4.0, 2.0]));
    }

    #[test]
    fn transform_identity_is_noop() {
        let b = Aabb::new([-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]);
        assert!(approx_box(&b.transform_by_mat4(&IDENTITY), &b));
        assert!(approx_box(&b.transform_corners(&IDENTITY), &b));
    }

    #[test]
    fn transform_translation_shifts_box() {
        let b = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut m = IDENTITY;
        m[3] = [10.0, -5.0, 2.0, 1.0];
        let t = b.transform_by_mat4(&m);
        assert!(approx3(t.min, [10.0, -5.0, 2.0]));
        assert!(approx3(t.max, [11.0, -4.0, 3.0]));
    }

    #[test]
    fn transform_scale_grows_half_extent() {
        let b = Aabb::from_center_extent([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let mut m = IDENTITY;
        m[0][0] = 3.0;
        m[1][1] = 4.0;
        m[2][2] = 0.5;
        let t = b.transform_by_mat4(&m);
        assert!(approx3(t.half_extent(), [3.0, 4.0, 0.5]));
        assert!(approx3(t.center(), [0.0, 0.0, 0.0]));
    }

    #[test]
    fn transform_rotation_90_about_z_swaps_axes() {
        // Column-major 90-degree rotation about z: exact integer entries.
        let m = [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let b = Aabb::new([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
        let t = b.transform_by_mat4(&m);
        assert!(approx3(t.min, [-2.0, -1.0, -3.0]));
        assert!(approx3(t.max, [2.0, 1.0, 3.0]));
    }

    #[test]
    fn transform_by_mat4_matches_corners_for_rotation() {
        // 3-4-5 in-plane rotation about z (orthonormal, no transcendentals),
        // combined with a scale and translation.
        let (cos, sin) = (0.6f32, 0.8f32);
        let m = [
            [cos, sin, 0.0, 0.0],
            [-sin, cos, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [7.0, -3.0, 1.0, 1.0],
        ];
        let b = Aabb::new([-1.0, -2.0, -0.5], [3.0, 1.0, 2.5]);
        let arvo = b.transform_by_mat4(&m);
        let brute = b.transform_corners(&m);
        assert!(approx_box(&arvo, &brute));
    }

    #[test]
    fn transform_center_equals_matrix_times_center() {
        let (cos, sin) = (0.6f32, 0.8f32);
        let m = [
            [cos, sin, 0.0, 0.0],
            [-sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [5.0, 6.0, 7.0, 1.0],
        ];
        let b = Aabb::new([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]);
        let center = b.center();
        let expected = [
            m[3][0] + m[0][0] * center[0] + m[1][0] * center[1] + m[2][0] * center[2],
            m[3][1] + m[0][1] * center[0] + m[1][1] * center[1] + m[2][1] * center[2],
            m[3][2] + m[0][2] * center[0] + m[1][2] * center[1] + m[2][2] * center[2],
        ];
        let t = b.transform_by_mat4(&m);
        assert!(approx3(t.center(), expected));
    }

    #[test]
    fn transform_empty_stays_empty() {
        let e = Aabb::empty();
        assert!(e.transform_by_mat4(&IDENTITY).is_empty());
        assert!(e.transform_corners(&IDENTITY).is_empty());
    }

    #[test]
    fn to_std430_layout_and_roundtrip() {
        let b = Aabb::new([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        let bytes = b.to_std430();
        assert_eq!(bytes.len(), AABB_TRANSFORM_STD430_SIZE);
        assert_eq!(AABB_TRANSFORM_STD430_SIZE, 2 * VEC4_STRIDE);
        assert!(approx(f32_from_le(&bytes[0..4]), 1.0));
        assert!(approx(f32_from_le(&bytes[4..8]), 2.0));
        assert!(approx(f32_from_le(&bytes[8..12]), 3.0));
        assert!(approx(f32_from_le(&bytes[16..20]), 4.0));
        assert!(approx(f32_from_le(&bytes[20..24]), 5.0));
        assert!(approx(f32_from_le(&bytes[24..28]), 6.0));
        // Pad lanes stay zero.
        assert_eq!(&bytes[12..16], &[0u8; 4]);
        assert_eq!(&bytes[28..32], &[0u8; 4]);
    }

    #[test]
    fn std430_size_is_two_vec4_slots() {
        let bytes = Aabb::empty().to_std430();
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(gpu_storage_bytes(0), AABB_TRANSFORM_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(1), 32);
        assert_eq!(gpu_storage_bytes(4), 128);
    }
}
