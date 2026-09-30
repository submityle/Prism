//! Polygonal aperture-blade shape for the particle `bokeh` / lens-occlusion
//! pass (design §16, §21).
//!
//! A physical camera stops light down with a diaphragm of `N` straight blades;
//! the resulting aperture is a regular (optionally rounded) convex polygon, and
//! every out-of-focus highlight a `GPU` gather pass paints is a scaled copy of
//! that polygon — the *bokeh* shape. Production stacks (Unreal's diaphragm
//! `DoF`, `Frostbite`'s gather `DoF`, Unity `HDRP`'s physical camera) drive
//! their `bokeh` kernel from exactly this polygon. This module owns the
//! `CPU`-verifiable *geometry* of that aperture: the signed-distance field
//! (`SDF`) of the blade polygon, the inside/outside test, and the soft edge
//! weight a gather kernel multiplies each tap by.
//!
//! # Boundary with [`super::depth_of_field`]
//!
//! [`super::depth_of_field`] owns the circle-of-confusion (`CoC`) diameter, the
//! gather radius, and the per-tap energy falloff — *how large and how bright*
//! each `bokeh` disc is. This module is orthogonal: it owns *what shape* the
//! disc has. It computes no `CoC`, reads no lens parameters, and reuses none of
//! that module's types; a caller scales an [`Aperture`] `SDF` by the `CoC`
//! radius the `DoF` pass returns.
//!
//! # Trig-free contract
//!
//! Regular-polygon blade normals depend on `cos`/`sin`, which this crate never
//! calls. Instead the caller supplies the unit outward normal of each blade
//! edge (already rotated by the desired aperture spin), and every method here
//! is pure multiply-add, `max`, and a single `f32::sqrt` for the round limit.
//! The soft edge uses a `smoothstep` cubic (`t * t * (3 - 2 * t)`); no
//! transcendental function and no `f32::round`/`f32::ceil` is ever called, so
//! results are bit-reproducible against a future `GPU` kernel.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Magnitudes below this collapse a division or interpolation span to a hard
/// step, and serve as the surface tolerance for [`Aperture::contains`], so no
/// divide-by-(near)-zero can produce a `NaN`.
const MIN_EDGE: f32 = 1e-6;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided in the contract code.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// Byte size of the `std430` packing of an [`Aperture`]'s scalar header.
///
/// The blade count (`u32`), the `apothem`, and the `roundness` fill the first
/// three lanes of a single `vec4` slot (16 bytes) with a zeroed padding tail;
/// the variable-length edge-normal array is bound as a separate storage buffer
/// by the pass, not packed here.
pub const APERTURE_STD430_SIZE: usize = VEC4_STRIDE;

/// Hermite `smoothstep` from `edge0` to `edge1` evaluated at `x`.
///
/// Returns `0.0` at or below `edge0`, `1.0` at or above `edge1`, and the
/// `t * t * (3 - 2 * t)` interpolation in between. A degenerate (near-zero
/// width) interval collapses to a hard step at `edge1` rather than dividing by
/// zero.
#[must_use]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span < MIN_EDGE {
        return if x < edge1 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Euclidean length of a 2D vector via a single `f32::sqrt`.
#[must_use]
fn length2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// A polygonal camera-aperture shape: the convex intersection of `blade`
/// half-planes, optionally rounded toward a circle.
///
/// `edge_normals` holds the unit outward normal of each blade edge (the caller
/// supplies these already rotated by the aperture spin, since the crate never
/// calls trigonometry). `apothem` is the inradius — the perpendicular distance
/// from the centre to each edge. `roundness` in `[0, 1]` blends the polygon
/// `SDF` (`0`) toward the circular `SDF` (`1`).
#[derive(Clone, Debug, PartialEq)]
pub struct Aperture {
    /// Unit outward normal of each blade edge, pre-rotated by the caller.
    pub edge_normals: Vec<[f32; 2]>,
    /// Inradius: perpendicular centre-to-edge distance, clamped non-negative.
    pub apothem: f32,
    /// Polygon-to-circle blend in `[0, 1]`: `0` = polygon, `1` = circle.
    pub roundness: f32,
}

impl Aperture {
    /// Builds an aperture from pre-computed unit edge normals.
    ///
    /// `apothem` is clamped to be non-negative and `roundness` to `[0, 1]`; the
    /// `edge_normals` are stored verbatim (use [`Aperture::from_blade_dirs`] to
    /// normalize raw directions).
    #[must_use]
    pub fn new(edge_normals: Vec<[f32; 2]>, apothem: f32, roundness: f32) -> Self {
        Self {
            edge_normals,
            apothem: apothem.max(0.0),
            roundness: roundness.clamp(0.0, 1.0),
        }
    }

    /// Builds an aperture by normalizing raw blade directions into unit edge
    /// normals.
    ///
    /// Each direction is divided by its length; degenerate (near-zero-length)
    /// directions are skipped so no `NaN` normal enters the shape. `apothem`
    /// and `roundness` are clamped as in [`Aperture::new`].
    #[must_use]
    pub fn from_blade_dirs(dirs: &[[f32; 2]], apothem: f32, roundness: f32) -> Self {
        let mut edge_normals = Vec::with_capacity(dirs.len());
        for d in dirs {
            let len = length2(*d);
            if len < MIN_EDGE {
                continue;
            }
            edge_normals.push([d[0] / len, d[1] / len]);
        }
        Self::new(edge_normals, apothem, roundness)
    }

    /// Number of blade edges (half-planes) defining the polygon.
    #[must_use]
    pub fn blade_count(&self) -> usize {
        self.edge_normals.len()
    }

    /// Signed distance to the convex blade polygon: the intersection of the
    /// blade half-planes.
    ///
    /// For each edge the supporting-plane distance is `dot(p, n_k) - apothem`;
    /// the polygon `SDF` is their maximum (negative inside, zero on an edge,
    /// positive outside). With no edges the polygon is undefined and this falls
    /// back to the circular `SDF` so downstream tests stay well defined.
    #[must_use]
    pub fn polygon_sdf(&self, p: [f32; 2]) -> f32 {
        if self.edge_normals.is_empty() {
            return self.circle_sdf(p);
        }
        let max_dist = self.edge_normals.iter().fold(f32::NEG_INFINITY, |m, n| {
            let d = p[0] * n[0] + p[1] * n[1];
            if d > m {
                d
            } else {
                m
            }
        });
        max_dist - self.apothem
    }

    /// Signed distance to the inscribed circle of radius `apothem`:
    /// `|p| - apothem`.
    #[must_use]
    pub fn circle_sdf(&self, p: [f32; 2]) -> f32 {
        length2(p) - self.apothem
    }

    /// Signed distance to the rounded aperture: a linear blend of the polygon
    /// `SDF` and the circular `SDF` by `roundness` (`0` = polygon, `1` =
    /// circle).
    #[must_use]
    pub fn sdf(&self, p: [f32; 2]) -> f32 {
        let poly = self.polygon_sdf(p);
        let circle = self.circle_sdf(p);
        poly + (circle - poly) * self.roundness
    }

    /// Whether `p` lies inside or on the aperture boundary (`sdf <= 0` within a
    /// small surface tolerance).
    #[must_use]
    pub fn contains(&self, p: [f32; 2]) -> bool {
        self.sdf(p) <= MIN_EDGE
    }

    /// Soft `bokeh` mask weight in `[0, 1]` for `p`, fading across a band of
    /// half-width `softness` centred on the aperture edge.
    ///
    /// Deep inside the aperture the weight is `1`, on the edge it is `0.5`, and
    /// beyond the outer band it is `0`; the transition is a `smoothstep` cubic.
    /// A `softness` of `0` yields a hard `1`/`0` mask at the boundary. The
    /// weight is monotonically non-increasing as `p` moves outward.
    #[must_use]
    pub fn edge_weight(&self, p: [f32; 2], softness: f32) -> f32 {
        let s = softness.max(0.0);
        1.0 - smoothstep(-s, s, self.sdf(p))
    }

    /// Evaluates [`Aperture::edge_weight`] at every point, returning one weight
    /// per input in order. An empty input yields an empty output.
    #[must_use]
    pub fn sample_weights(&self, points: &[[f32; 2]], softness: f32) -> Vec<f32> {
        points
            .iter()
            .map(|p| self.edge_weight(*p, softness))
            .collect()
    }

    /// Packs the scalar header into its `std430` byte layout.
    ///
    /// The blade count (`u32`), `apothem`, and `roundness` fill the first three
    /// lanes of one `vec4` slot with a zeroed padding tail; a blade count that
    /// exceeds `u32` saturates to `u32::MAX`.
    #[must_use]
    pub fn to_std430(&self) -> [u8; APERTURE_STD430_SIZE] {
        let count = u32::try_from(self.edge_normals.len()).unwrap_or(u32::MAX);
        let mut bytes = [0u8; APERTURE_STD430_SIZE];
        bytes[0..4].copy_from_slice(&count.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.apothem.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.roundness.to_le_bytes());
        bytes
    }

    /// Total `std430` byte size of a storage buffer holding `count` packed
    /// aperture headers, clamped up to a single element so an empty buffer
    /// still yields a valid non-zero `WebGPU` binding.
    #[must_use]
    pub fn gpu_storage_bytes(count: usize) -> usize {
        storage_bytes(APERTURE_STD430_SIZE, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit outward normals of an axis-aligned square aperture (four blades).
    fn square_normals() -> Vec<[f32; 2]> {
        alloc::vec![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    }

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    #[test]
    fn square_center_is_inside() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(ap.contains([0.0, 0.0]));
        assert!(ap.sdf([0.0, 0.0]) < 0.0);
    }

    #[test]
    fn square_far_point_is_outside() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(!ap.contains([2.0, 0.0]));
        assert!(ap.sdf([2.0, 0.0]) > 0.0);
    }

    #[test]
    fn sdf_is_zero_on_the_apothem_edge() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(approx_eq(ap.polygon_sdf([1.0, 0.0]), 0.0));
        assert!(approx_eq(ap.polygon_sdf([0.0, 1.0]), 0.0));
    }

    #[test]
    fn polygon_sdf_is_max_of_half_planes() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        // Corner (1,1): both x and y half-planes read distance 1, so sdf = 0.
        assert!(approx_eq(ap.polygon_sdf([1.0, 1.0]), 0.0));
        // Outside corner (2,2): max half-plane distance 2, sdf = 1.
        assert!(approx_eq(ap.polygon_sdf([2.0, 2.0]), 1.0));
    }

    #[test]
    fn circle_sdf_matches_radial_distance() {
        let ap = Aperture::new(square_normals(), 5.0, 0.0);
        // 3-4-5 triangle: |(3,4)| = 5, on the circle.
        assert!(approx_eq(ap.circle_sdf([3.0, 4.0]), 0.0));
        assert!(approx_eq(ap.circle_sdf([0.0, 0.0]), -5.0));
    }

    #[test]
    fn roundness_one_equals_circle_sdf() {
        let ap = Aperture::new(square_normals(), 1.0, 1.0);
        for p in [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0], [2.0, 0.0]] {
            assert!(approx_eq(ap.sdf(p), ap.circle_sdf(p)));
        }
    }

    #[test]
    fn roundness_zero_equals_polygon_sdf() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        for p in [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0], [2.0, 0.0]] {
            assert!(approx_eq(ap.sdf(p), ap.polygon_sdf(p)));
        }
    }

    #[test]
    fn roundness_half_interpolates_between_the_two() {
        let ap = Aperture::new(square_normals(), 1.0, 0.5);
        let p = [1.5, 0.0];
        let mid = 0.5 * (ap.polygon_sdf(p) + ap.circle_sdf(p));
        assert!(approx_eq(ap.sdf(p), mid));
    }

    #[test]
    fn from_blade_dirs_normalizes_directions() {
        let ap = Aperture::from_blade_dirs(&[[3.0, 4.0], [-10.0, 0.0]], 1.0, 0.0);
        assert_eq!(ap.blade_count(), 2);
        assert!(approx_eq(length2(ap.edge_normals[0]), 1.0));
        assert!(approx_eq(length2(ap.edge_normals[1]), 1.0));
        assert!(approx_eq(ap.edge_normals[0][0], 0.6));
        assert!(approx_eq(ap.edge_normals[0][1], 0.8));
    }

    #[test]
    fn from_blade_dirs_skips_degenerate_directions() {
        let ap = Aperture::from_blade_dirs(&[[0.0, 0.0], [1.0, 0.0]], 1.0, 0.0);
        assert_eq!(ap.blade_count(), 1);
        assert!(approx_eq(ap.edge_normals[0][0], 1.0));
    }

    #[test]
    fn contains_is_true_inside_and_false_outside() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(ap.contains([0.9, 0.0]));
        assert!(ap.contains([-0.5, 0.5]));
        assert!(!ap.contains([1.1, 0.0]));
        assert!(!ap.contains([1.0, 1.0001]));
    }

    #[test]
    fn contains_on_edge_within_tolerance() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(ap.contains([1.0, 0.0]));
    }

    #[test]
    fn edge_weight_is_one_deep_inside() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(approx_eq(ap.edge_weight([0.0, 0.0], 0.1), 1.0));
    }

    #[test]
    fn edge_weight_is_zero_well_outside() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(approx_eq(ap.edge_weight([3.0, 0.0], 0.1), 0.0));
    }

    #[test]
    fn edge_weight_is_half_on_the_edge() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(approx_eq(ap.edge_weight([1.0, 0.0], 0.25), 0.5));
    }

    #[test]
    fn edge_weight_is_monotone_outward() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        let mut prev = 2.0;
        let mut x = 0.0;
        while x <= 2.0 {
            let w = ap.edge_weight([x, 0.0], 0.3);
            assert!(w <= prev + CMP_EPS);
            assert!(w >= 0.0);
            assert!(w <= 1.0);
            prev = w;
            x += 0.1;
        }
    }

    #[test]
    fn edge_weight_zero_softness_is_a_hard_mask() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(approx_eq(ap.edge_weight([0.5, 0.0], 0.0), 1.0));
        assert!(approx_eq(ap.edge_weight([1.5, 0.0], 0.0), 0.0));
    }

    #[test]
    fn rotation_preserves_the_shape() {
        // A valid rotation from the 3-4-5 unit vector (c^2 + s^2 = 1).
        let (c, s) = (0.6f32, 0.8f32);
        let rot = |v: [f32; 2]| [c * v[0] - s * v[1], s * v[0] + c * v[1]];
        let base = Aperture::new(square_normals(), 1.0, 0.0);
        let rotated_normals: Vec<[f32; 2]> = base.edge_normals.iter().map(|n| rot(*n)).collect();
        let rotated = Aperture::new(rotated_normals, 1.0, 0.0);
        for p in [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0], [2.0, 0.0], [-0.7, 0.2]] {
            // A rotation preserves dot products, so the SDF of the rotated
            // point under the rotated aperture matches the original SDF.
            assert!(approx_eq(rotated.sdf(rot(p)), base.sdf(p)));
        }
    }

    #[test]
    fn empty_normals_fall_back_to_circle() {
        let ap = Aperture::new(Vec::new(), 1.0, 0.0);
        assert_eq!(ap.blade_count(), 0);
        assert!(approx_eq(ap.polygon_sdf([2.0, 0.0]), 1.0));
        assert!(approx_eq(ap.sdf([2.0, 0.0]), ap.circle_sdf([2.0, 0.0])));
    }

    #[test]
    fn constructor_clamps_apothem_and_roundness() {
        let ap = Aperture::new(square_normals(), -3.0, 5.0);
        assert!(approx_eq(ap.apothem, 0.0));
        assert!(approx_eq(ap.roundness, 1.0));
        let ap2 = Aperture::new(square_normals(), 2.0, -1.0);
        assert!(approx_eq(ap2.roundness, 0.0));
    }

    #[test]
    fn sample_weights_maps_every_point_in_range() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        let pts = [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]];
        let w = ap.sample_weights(&pts, 0.25);
        assert_eq!(w.len(), pts.len());
        for value in &w {
            assert!(*value >= 0.0);
            assert!(*value <= 1.0);
        }
        assert!(approx_eq(w[0], 1.0));
        assert!(approx_eq(w[1], 0.5));
        assert!(approx_eq(w[2], 0.0));
    }

    #[test]
    fn sample_weights_empty_input_is_empty() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert!(ap.sample_weights(&[], 0.1).is_empty());
    }

    #[test]
    fn std430_size_is_one_vec4() {
        assert_eq!(APERTURE_STD430_SIZE, 16);
        assert_eq!(APERTURE_STD430_SIZE % VEC4_STRIDE, 0);
    }

    #[test]
    fn std430_roundtrip_preserves_fields() {
        let ap = Aperture::new(square_normals(), 1.5, 0.25);
        let bytes = ap.to_std430();
        assert_eq!(bytes.len(), APERTURE_STD430_SIZE);
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            4
        );
        assert_eq!(
            f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]).to_bits(),
            1.5f32.to_bits()
        );
        assert_eq!(
            f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]).to_bits(),
            0.25f32.to_bits()
        );
        // Padding tail is zeroed.
        assert_eq!(bytes[12], 0);
        assert_eq!(bytes[13], 0);
        assert_eq!(bytes[14], 0);
        assert_eq!(bytes[15], 0);
    }

    #[test]
    fn gpu_storage_bytes_scales_and_reserves_one() {
        assert_eq!(Aperture::gpu_storage_bytes(0), APERTURE_STD430_SIZE);
        assert_eq!(Aperture::gpu_storage_bytes(1), APERTURE_STD430_SIZE);
        assert_eq!(Aperture::gpu_storage_bytes(8), APERTURE_STD430_SIZE * 8);
    }

    #[test]
    fn blade_count_reports_normal_count() {
        let ap = Aperture::new(square_normals(), 1.0, 0.0);
        assert_eq!(ap.blade_count(), 4);
    }
}
