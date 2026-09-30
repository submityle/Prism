//! Half-space (plane) clipping of segments and convex polygons for the particle
//! culling and slicing passes (design §12, §13).
//!
//! Frustum culling, near/far slicing of ribbon and trail geometry, and volume
//! clipping all reduce to the same primitive: cut a segment or a convex polygon
//! against an oriented plane and keep the part on the *inside* half-space. This
//! module owns the `CPU`-verifiable reference for that operation so a future
//! `GPU` kernel can reproduce it bit for bit. The polygon path is the classic
//! `Sutherland-Hodgman` convex-polygon clip: it walks the polygon edges once,
//! emitting inside vertices and boundary intersection points in order, which
//! turns a convex input into a convex output against a single plane. Clipping a
//! polygon against a whole frustum is just applying the single-plane clip once
//! per plane ([`clip_polygon_planes`]).
//!
//! # Conventions
//! A [`Plane`] stores the coefficients of `n·p + d = 0`. The *inside* half-space
//! is where `n·p + d >= 0`, so a plane whose normal points into the frustum
//! keeps the visible side. [`Plane::signed_distance`] returns `n·p + d`
//! directly; boundary intersections use the parameter `t = da / (da - db)` and a
//! linear interpolation of the two endpoints, so the math stays affine.
//!
//! # Determinism
//! The only non-`+ - * /` primitive used is `f32::sqrt` (inside
//! [`Plane::normalized`]); there are no transcendental functions and no platform
//! math, so the reference is deterministic. `f32` magnitudes are never compared
//! with `==`/`!=`; comparisons go through [`CMP_EPS`] or ordinary `<`/`>=`
//! ordering against zero.
//!
//! # Layout
//! [`to_std430`] packs a plane into one `vec4` slot (`normal.xyz`, then `d`),
//! matching the shared [`crate::particle::gpu_layout`] stride so the plane list
//! a `WebGPU` culling kernel binds has a stable `ABI`.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Epsilon for tolerant `f32` comparisons; direct `==`/`!=` is forbidden.
///
/// Doubles as the squared-length floor below which [`Plane::normalized`] treats
/// a normal as degenerate and returns the plane unchanged instead of dividing.
pub const CMP_EPS: f32 = 1e-6;

/// Byte size of one packed [`Plane`] in a `std430` storage buffer: the three
/// normal components plus `d` fill exactly one `vec4` slot
/// ([`crate::particle::gpu_layout::VEC4_STRIDE`]).
pub const PLANE_CLIP_STD430_SIZE: usize = VEC4_STRIDE;

/// An oriented plane written as the coefficients of `n·p + d = 0`.
///
/// The half-space `n·p + d >= 0` is the *inside* the clippers keep; flip the
/// normal (and the sign of `d`) to keep the opposite side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// The plane normal `n`. Need not be unit length; [`Plane::normalized`]
    /// rescales it (and `d`) when a true signed distance is required.
    pub normal: [f32; 3],
    /// The plane constant `d` in `n·p + d = 0`.
    pub d: f32,
}

impl Plane {
    /// Builds a plane from raw `n·p + d = 0` coefficients without normalizing.
    #[must_use]
    pub const fn new(normal: [f32; 3], d: f32) -> Self {
        Self { normal, d }
    }

    /// Evaluates `n·p + d`, the signed distance scaled by `|n|`.
    ///
    /// The sign tells which half-space `p` lies in (positive is inside); the
    /// magnitude is a true Euclidean distance only when the normal is unit
    /// length (see [`Plane::normalized`]).
    #[must_use]
    pub fn signed_distance(&self, p: [f32; 3]) -> f32 {
        self.normal[0] * p[0] + self.normal[1] * p[1] + self.normal[2] * p[2] + self.d
    }

    /// Returns an equivalent plane with a unit-length normal.
    ///
    /// Dividing both `n` and `d` by `|n|` leaves the zero set (the plane itself)
    /// unchanged while making [`Plane::signed_distance`] a true Euclidean
    /// distance. A degenerate near-zero normal (squared length below
    /// [`CMP_EPS`]) is returned unchanged rather than producing a `NaN`. This is
    /// the only place `f32::sqrt` is used.
    #[must_use]
    pub fn normalized(&self) -> Plane {
        let len_sq = self.normal[0] * self.normal[0]
            + self.normal[1] * self.normal[1]
            + self.normal[2] * self.normal[2];
        if len_sq < CMP_EPS {
            return *self;
        }
        let inv_len = 1.0 / len_sq.sqrt();
        Plane {
            normal: [
                self.normal[0] * inv_len,
                self.normal[1] * inv_len,
                self.normal[2] * inv_len,
            ],
            d: self.d * inv_len,
        }
    }
}

/// Which half-space a point falls in relative to a [`Plane`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Side {
    /// `n·p + d > eps`: strictly inside the kept half-space.
    Inside,
    /// `n·p + d < -eps`: strictly outside the kept half-space.
    Outside,
    /// Within `eps` of the plane: treated as lying on the boundary.
    On,
}

/// Classifies a point against a plane with a symmetric `eps` boundary band.
#[must_use]
pub fn classify(plane: &Plane, p: [f32; 3], eps: f32) -> Side {
    let dist = plane.signed_distance(p);
    if dist > eps {
        Side::Inside
    } else if dist < -eps {
        Side::Outside
    } else {
        Side::On
    }
}

/// Linearly interpolates two points: `a + (b - a) * t`.
#[must_use]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Clips a segment against a plane, returning the part inside the half-space.
///
/// Returns `None` when both endpoints are strictly outside. When the segment
/// straddles the plane the outside endpoint is replaced by the boundary
/// intersection `a + (b - a) * t` with `t = da / (da - db)`, so the returned
/// pair always lies in the inside half-space with `a`/`b` order preserved.
#[must_use]
pub fn clip_segment(plane: &Plane, a: [f32; 3], b: [f32; 3]) -> Option<([f32; 3], [f32; 3])> {
    let da = plane.signed_distance(a);
    let db = plane.signed_distance(b);
    let a_inside = da >= 0.0;
    let b_inside = db >= 0.0;
    match (a_inside, b_inside) {
        (true, true) => Some((a, b)),
        (false, false) => None,
        (true, false) => {
            let t = da / (da - db);
            Some((a, lerp3(a, b, t)))
        }
        (false, true) => {
            let t = da / (da - db);
            Some((lerp3(a, b, t), b))
        }
    }
}

/// Returns the crossing parameter `t` where segment `a → b` meets the plane.
///
/// `t = da / (da - db)` is the fraction along the segment at which
/// `n·p + d = 0`. Returns `None` when the segment is parallel to the plane (the
/// denominator is within [`CMP_EPS`] of zero) or when the crossing lies outside
/// the segment (`t` not in `0..=1`).
#[must_use]
pub fn intersect_param(plane: &Plane, a: [f32; 3], b: [f32; 3]) -> Option<f32> {
    let da = plane.signed_distance(a);
    let db = plane.signed_distance(b);
    let denom = da - db;
    if denom.abs() < CMP_EPS {
        return None;
    }
    let t = da / denom;
    if (0.0..=1.0).contains(&t) {
        Some(t)
    } else {
        None
    }
}

/// Clips a convex polygon against a single plane (`Sutherland-Hodgman`).
///
/// Walks the polygon edges once. For each edge it emits the current vertex when
/// it is inside, then emits the boundary intersection whenever the edge crosses
/// the plane, producing the vertices of the clipped convex polygon in order.
/// A fewer-than-three-vertex input, or a result degenerating below three
/// vertices (the polygon lies wholly outside, or clips to a sliver), yields an
/// empty [`Vec`].
#[must_use]
pub fn clip_polygon(plane: &Plane, poly: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    let mut out: Vec<[f32; 3]> = Vec::new();
    for (i, &cur) in poly.iter().enumerate() {
        let nxt = poly[(i + 1) % n];
        let dc = plane.signed_distance(cur);
        let dn = plane.signed_distance(nxt);
        let cur_inside = dc >= 0.0;
        let nxt_inside = dn >= 0.0;
        if cur_inside {
            out.push(cur);
        }
        if cur_inside != nxt_inside {
            let t = dc / (dc - dn);
            out.push(lerp3(cur, nxt, t));
        }
    }
    if out.len() < 3 {
        Vec::new()
    } else {
        out
    }
}

/// Clips a convex polygon against every plane in turn (a frustum clip).
///
/// Applies [`clip_polygon`] once per plane, feeding each result into the next.
/// Because each single-plane clip preserves convexity, the fold yields the
/// intersection of the polygon with all the half-spaces. Clipping stops early
/// and returns an empty [`Vec`] as soon as the polygon is fully removed.
#[must_use]
pub fn clip_polygon_planes(planes: &[Plane], poly: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut current = poly.to_vec();
    for plane in planes {
        if current.len() < 3 {
            return Vec::new();
        }
        current = clip_polygon(plane, &current);
    }
    current
}

/// Packs a plane into its `std430` block as little-endian
/// `normal.x, normal.y, normal.z, d`, spanning one `vec4` slot
/// ([`PLANE_CLIP_STD430_SIZE`] bytes).
#[must_use]
pub fn to_std430(plane: &Plane) -> [u8; PLANE_CLIP_STD430_SIZE] {
    let mut bytes = [0u8; PLANE_CLIP_STD430_SIZE];
    bytes[0..4].copy_from_slice(&plane.normal[0].to_le_bytes());
    bytes[4..8].copy_from_slice(&plane.normal[1].to_le_bytes());
    bytes[8..12].copy_from_slice(&plane.normal[2].to_le_bytes());
    bytes[12..16].copy_from_slice(&plane.d.to_le_bytes());
    bytes
}

/// Total `std430` byte size of a storage buffer holding `count` packed planes,
/// clamped up to a single element per the shared [`storage_bytes`] rule.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(PLANE_CLIP_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the `f32` comparisons used by the tests; direct
    /// `==` on floating point is intentionally avoided.
    const TEST_EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn approx_pt(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    /// Reads a little-endian `f32` back out of packed `std430` bytes.
    fn read_le_f32(bytes: &[u8], offset: usize) -> f32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[offset..offset + 4]);
        f32::from_le_bytes(buf)
    }

    fn unit_x_plane_at(x: f32) -> Plane {
        // Inside where x <= boundary: normal points toward -x.
        Plane::new([-1.0, 0.0, 0.0], x)
    }

    #[test]
    fn new_stores_raw_coefficients() {
        let p = Plane::new([1.0, 2.0, 3.0], -4.0);
        assert!(approx_pt(p.normal, [1.0, 2.0, 3.0]));
        assert!(approx(p.d, -4.0));
    }

    #[test]
    fn signed_distance_is_dot_plus_d() {
        // Plane x = 2 with unit normal toward +x.
        let p = Plane::new([1.0, 0.0, 0.0], -2.0);
        assert!(approx(p.signed_distance([5.0, 0.0, 0.0]), 3.0));
        assert!(approx(p.signed_distance([2.0, 9.0, 9.0]), 0.0));
        assert!(approx(p.signed_distance([0.0, 0.0, 0.0]), -2.0));
    }

    #[test]
    fn normalized_makes_unit_normal_and_scales_d() {
        let p = Plane::new([0.0, 3.0, 0.0], 6.0).normalized();
        assert!(approx_pt(p.normal, [0.0, 1.0, 0.0]));
        assert!(approx(p.d, 2.0));
    }

    #[test]
    fn normalized_preserves_the_zero_set() {
        let raw = Plane::new([2.0, 0.0, 0.0], -4.0);
        let norm = raw.normalized();
        // Both must vanish at x = 2 and agree in sign elsewhere.
        assert!(approx(norm.signed_distance([2.0, 1.0, 1.0]), 0.0));
        assert!(norm.signed_distance([5.0, 0.0, 0.0]) > 0.0);
    }

    #[test]
    fn normalized_degenerate_normal_returned_unchanged() {
        let raw = Plane::new([0.0, 0.0, 0.0], 7.0);
        let norm = raw.normalized();
        assert!(approx_pt(norm.normal, [0.0, 0.0, 0.0]));
        assert!(approx(norm.d, 7.0));
    }

    #[test]
    fn classify_splits_inside_outside_on() {
        let p = Plane::new([1.0, 0.0, 0.0], 0.0);
        assert_eq!(classify(&p, [1.0, 0.0, 0.0], 1e-6), Side::Inside);
        assert_eq!(classify(&p, [-1.0, 0.0, 0.0], 1e-6), Side::Outside);
        assert_eq!(classify(&p, [0.0, 5.0, 5.0], 1e-6), Side::On);
    }

    #[test]
    fn classify_eps_band_captures_near_boundary() {
        let p = Plane::new([1.0, 0.0, 0.0], 0.0);
        // Just inside geometrically, but within the eps band -> On.
        assert_eq!(classify(&p, [1e-4, 0.0, 0.0], 1e-3), Side::On);
        // Outside the band -> Inside.
        assert_eq!(classify(&p, [1.0, 0.0, 0.0], 1e-3), Side::Inside);
    }

    #[test]
    fn clip_segment_fully_inside_is_unchanged() {
        let p = unit_x_plane_at(1.0); // inside where x <= 1
        let clipped = clip_segment(&p, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]);
        let (a, b) = clipped.expect("segment is inside");
        assert!(approx_pt(a, [0.0, 0.0, 0.0]));
        assert!(approx_pt(b, [0.5, 0.0, 0.0]));
    }

    #[test]
    fn clip_segment_fully_outside_is_none() {
        let p = unit_x_plane_at(1.0); // inside where x <= 1
        let clipped = clip_segment(&p, [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]);
        assert!(clipped.is_none());
    }

    #[test]
    fn clip_segment_a_inside_replaces_b_with_crossing() {
        let p = unit_x_plane_at(1.0); // inside where x <= 1
        let (a, b) =
            clip_segment(&p, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]).expect("segment crosses the plane");
        assert!(approx_pt(a, [0.0, 0.0, 0.0]));
        assert!(approx_pt(b, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn clip_segment_b_inside_replaces_a_with_crossing() {
        let p = unit_x_plane_at(1.0); // inside where x <= 1
        let (a, b) =
            clip_segment(&p, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]).expect("segment crosses the plane");
        assert!(approx_pt(a, [1.0, 0.0, 0.0]));
        assert!(approx_pt(b, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn intersect_param_finds_midpoint() {
        let p = unit_x_plane_at(1.0);
        let t = intersect_param(&p, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]).expect("crossing exists");
        assert!(approx(t, 0.5));
    }

    #[test]
    fn intersect_param_parallel_is_none() {
        // Segment lies in the plane y = 0; both endpoints have equal distance.
        let p = Plane::new([0.0, 1.0, 0.0], 0.0);
        assert!(intersect_param(&p, [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn intersect_param_out_of_range_is_none() {
        let p = unit_x_plane_at(5.0); // crossing at x = 5, outside 0..=2
        assert!(intersect_param(&p, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]).is_none());
    }

    fn unit_triangle() -> [[f32; 3]; 3] {
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    }

    #[test]
    fn clip_polygon_fully_inside_is_unchanged() {
        let p = unit_x_plane_at(10.0); // inside where x <= 10: triangle fits
        let tri = unit_triangle();
        let out = clip_polygon(&p, &tri);
        assert_eq!(out.len(), 3);
        assert!(approx_pt(out[0], tri[0]));
        assert!(approx_pt(out[1], tri[1]));
        assert!(approx_pt(out[2], tri[2]));
    }

    #[test]
    fn clip_polygon_fully_outside_is_empty() {
        // Inside where x <= -5: the unit triangle is entirely outside.
        let p = unit_x_plane_at(-5.0);
        let out = clip_polygon(&p, &unit_triangle());
        assert!(out.is_empty());
    }

    #[test]
    fn clip_polygon_cuts_triangle_into_quad() {
        // Inside where x <= 0.5 slices the triangle across two edges.
        let p = unit_x_plane_at(0.5);
        let out = clip_polygon(&p, &unit_triangle());
        assert_eq!(out.len(), 4);
        assert!(approx_pt(out[0], [0.0, 0.0, 0.0]));
        assert!(approx_pt(out[1], [0.5, 0.0, 0.0]));
        assert!(approx_pt(out[2], [0.5, 0.5, 0.0]));
        assert!(approx_pt(out[3], [0.0, 1.0, 0.0]));
    }

    #[test]
    fn clip_polygon_half_cut_square_stays_quad() {
        // Square in the XY plane, inside where x <= 0.5.
        let square = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let out = clip_polygon(&unit_x_plane_at(0.5), &square);
        assert_eq!(out.len(), 4);
        // Every surviving vertex must satisfy x <= 0.5 + eps.
        assert!(out.iter().all(|v| v[0] <= 0.5 + TEST_EPS));
    }

    #[test]
    fn clip_polygon_degenerate_input_is_empty() {
        let two: [[f32; 3]; 2] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(clip_polygon(&unit_x_plane_at(0.5), &two).is_empty());
        let empty: [[f32; 3]; 0] = [];
        assert!(clip_polygon(&unit_x_plane_at(0.5), &empty).is_empty());
    }

    #[test]
    fn clip_polygon_planes_intersects_two_half_spaces() {
        // Keep x <= 0.75 and y <= 0.75; the triangle survives as a pentagon.
        let planes = [
            Plane::new([-1.0, 0.0, 0.0], 0.75),
            Plane::new([0.0, -1.0, 0.0], 0.75),
        ];
        let out = clip_polygon_planes(&planes, &unit_triangle());
        assert!(out.len() >= 3);
        assert!(out
            .iter()
            .all(|v| v[0] <= 0.75 + TEST_EPS && v[1] <= 0.75 + TEST_EPS));
    }

    #[test]
    fn clip_polygon_planes_all_inside_is_unchanged() {
        let planes = [
            Plane::new([-1.0, 0.0, 0.0], 10.0),
            Plane::new([0.0, -1.0, 0.0], 10.0),
        ];
        let out = clip_polygon_planes(&planes, &unit_triangle());
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn clip_polygon_planes_empty_when_removed() {
        // First plane keeps x <= -5: nothing survives, so the fold bails out.
        let planes = [unit_x_plane_at(-5.0), Plane::new([0.0, -1.0, 0.0], 10.0)];
        assert!(clip_polygon_planes(&planes, &unit_triangle()).is_empty());
    }

    #[test]
    fn clip_polygon_planes_no_planes_returns_copy() {
        let planes: [Plane; 0] = [];
        let out = clip_polygon_planes(&planes, &unit_triangle());
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn std430_layout_encodes_normal_then_d() {
        let plane = Plane::new([1.0, -2.0, 0.5], 3.25);
        let bytes = to_std430(&plane);
        assert_eq!(bytes.len(), PLANE_CLIP_STD430_SIZE);
        assert!(approx(read_le_f32(&bytes, 0), 1.0));
        assert!(approx(read_le_f32(&bytes, 4), -2.0));
        assert!(approx(read_le_f32(&bytes, 8), 0.5));
        assert!(approx(read_le_f32(&bytes, 12), 3.25));
    }

    #[test]
    fn std430_size_matches_vec4_stride() {
        assert_eq!(PLANE_CLIP_STD430_SIZE, VEC4_STRIDE);
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(gpu_storage_bytes(0), PLANE_CLIP_STD430_SIZE);
        assert_eq!(gpu_storage_bytes(6), PLANE_CLIP_STD430_SIZE * 6);
    }
}
