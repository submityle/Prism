//! `Brown-Conrady` radial + tangential lens-distortion model and its inverse
//! (design §16-§21).
//!
//! Real camera lenses do not image a straight world line to a straight image
//! line: a wide-angle lens bows lines outward (*barrel* distortion) and a
//! telephoto pinches them inward (*pincushion* distortion), while a decentred
//! or tilted element adds an asymmetric *tangential* smear. The photogrammetry
//! standard for describing both is the `Brown-Conrady` polynomial: a radial
//! series in even powers of the image radius plus a two-term tangential
//! correction. This module is the deterministic `CPU` reference for that model
//! and its inverse, so a future `GPU` warp kernel can be checked bit for bit
//! against it.
//!
//! # Distinction from the sibling lens modules
//!
//! This module is deliberately *not* a colour split and *not* a glare sprite,
//! and it shares **no** types with them:
//!
//! * [`super::chromatic_aberration`] models per-channel *radial colour split*
//!   (wavelength-dependent magnification); it moves the red/green/blue records
//!   apart but does not bend geometry as a whole.
//! * [`super::lens_flare`] models the *ghost/halo sprite chain* an optical
//!   train throws when a bright source is in frame; it is an additive overlay,
//!   not a coordinate warp.
//!
//! `lens_distortion` owns the pure *geometric* map: it takes a normalized
//! image-plane coordinate and returns where the lens actually lands it (or,
//! inversely, recovers the ideal coordinate from an observed one). It owns its
//! own coefficient block, [`DistortionCoeffs`], and never borrows a colour or
//! sprite type from its neighbours.
//!
//! # Forward and inverse maps
//!
//! The forward map [`distort`] is a direct polynomial evaluation. The inverse
//! map [`undistort`] has no closed form, so it runs a fixed
//! [`UNDISTORT_ITERATIONS`]-step fixed-point iteration (the standard
//! `OpenCV`-style refinement) using only multiply/add/divide. Every routine
//! avoids transcendental calls and integer-power intrinsics: powers of the
//! radius are formed by repeated multiplication (`r2 = x*x + y*y`,
//! `r4 = r2*r2`, `r6 = r4*r2`), keeping the reference reproducible against the
//! determinism contract of [`super::simulation`].

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Number of fixed-point refinement steps [`undistort`] runs.
///
/// Ten iterations converge the `Brown-Conrady` inverse to well below single
/// precision for the moderate coefficients a physical lens exhibits, while
/// keeping the cost a small constant so a `GPU` port can unroll it.
pub const UNDISTORT_ITERATIONS: usize = 10;

/// `std430` byte size of a serialized [`DistortionCoeffs`] block.
///
/// The five scalar coefficients occupy the first five `f32` slots; the block is
/// padded up to two `vec4` slots (`2 * VEC4_STRIDE`) so it honours the 16-byte
/// `std430` base alignment a `GPU` uniform/storage binding expects.
pub const LENS_DISTORTION_STD430_SIZE: usize = 2 * VEC4_STRIDE;

/// `Brown-Conrady` distortion coefficients for one lens.
///
/// The `k*` terms are the *radial* series (even powers of the image radius) and
/// the `p*` terms are the *tangential* (decentring) correction. All are defined
/// in the normalized image plane, so they are resolution independent. A lens
/// with every coefficient zero is a perfect pinhole; see [`DistortionCoeffs::none`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistortionCoeffs {
    /// First radial coefficient (scales `r^2`); positive is pincushion,
    /// negative is barrel.
    pub k1: f32,
    /// Second radial coefficient (scales `r^4`); shapes the mid-field falloff.
    pub k2: f32,
    /// Third radial coefficient (scales `r^6`); shapes the extreme edge.
    pub k3: f32,
    /// First tangential coefficient of the decentring correction.
    pub p1: f32,
    /// Second tangential coefficient of the decentring correction.
    pub p2: f32,
}

impl DistortionCoeffs {
    /// Builds a coefficient block from explicit radial (`k1`, `k2`, `k3`) and
    /// tangential (`p1`, `p2`) terms.
    #[must_use]
    pub const fn new(k1: f32, k2: f32, k3: f32, p1: f32, p2: f32) -> Self {
        Self { k1, k2, k3, p1, p2 }
    }

    /// The identity lens: every coefficient zero, so [`distort`] and
    /// [`undistort`] are both the identity map.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

impl Default for DistortionCoeffs {
    /// Defaults to the identity lens, [`DistortionCoeffs::none`].
    fn default() -> Self {
        Self::none()
    }
}

/// Evaluates the radial magnification factor `1 + k1*r^2 + k2*r^4 + k3*r^6`.
///
/// The even powers are formed by repeated multiplication of the squared radius
/// `r2`, never by an integer-power intrinsic, so the result is deterministic.
#[must_use]
fn radial_factor(coeffs: &DistortionCoeffs, r2: f32) -> f32 {
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    1.0 + coeffs.k1 * r2 + coeffs.k2 * r4 + coeffs.k3 * r6
}

/// Evaluates the `Brown-Conrady` tangential offset `[dx, dy]` at coordinate
/// `(x, y)` whose squared radius is `r2`.
///
/// This is the decentring term added on top of the radial scaling in
/// [`distort`]; it vanishes when both tangential coefficients are zero.
#[must_use]
fn tangential_offset(coeffs: &DistortionCoeffs, x: f32, y: f32, r2: f32) -> [f32; 2] {
    let dx = 2.0 * coeffs.p1 * x * y + coeffs.p2 * (r2 + 2.0 * x * x);
    let dy = coeffs.p1 * (r2 + 2.0 * y * y) + 2.0 * coeffs.p2 * x * y;
    [dx, dy]
}

/// Applies the forward `Brown-Conrady` map to a normalized `undistorted`
/// coordinate, returning where the lens lands it.
///
/// The map scales the coordinate by the radial factor and adds the tangential
/// offset. The optical centre `(0, 0)` is always a fixed point.
#[must_use]
pub fn distort(coeffs: &DistortionCoeffs, undistorted: [f32; 2]) -> [f32; 2] {
    let [x, y] = undistorted;
    let r2 = x * x + y * y;
    let radial = radial_factor(coeffs, r2);
    let [dx, dy] = tangential_offset(coeffs, x, y, r2);
    [x * radial + dx, y * radial + dy]
}

/// Inverts the forward map: recovers the ideal coordinate that [`distort`]
/// would have carried to `distorted`.
///
/// The `Brown-Conrady` inverse has no closed form, so this runs a fixed
/// [`UNDISTORT_ITERATIONS`]-step fixed-point refinement seeded at the observed
/// point, using only multiply/add/divide. The radial factor stays close to one
/// for physical coefficients, so the division never approaches zero.
#[must_use]
pub fn undistort(coeffs: &DistortionCoeffs, distorted: [f32; 2]) -> [f32; 2] {
    let [xd, yd] = distorted;
    let mut x = xd;
    let mut y = yd;
    for _ in 0..UNDISTORT_ITERATIONS {
        let r2 = x * x + y * y;
        let radial = radial_factor(coeffs, r2);
        let [dx, dy] = tangential_offset(coeffs, x, y, r2);
        x = (xd - dx) / radial;
        y = (yd - dy) / radial;
    }
    [x, y]
}

/// Applies the purely radial part of the map to a scalar radius `r`.
///
/// This is the geometry along a ray through the optical centre, ignoring the
/// tangential term; it returns `r * (1 + k1*r^2 + k2*r^4 + k3*r^6)`.
#[must_use]
pub fn distort_radius(coeffs: &DistortionCoeffs, r: f32) -> f32 {
    let r2 = r * r;
    r * radial_factor(coeffs, r2)
}

/// Applies [`distort`] to every point in `points`, preserving order and length.
///
/// This is the batch entry point a `GPU` warp pass mirrors; the returned vector
/// has exactly one distorted coordinate per input coordinate.
#[must_use]
pub fn distort_grid(coeffs: &DistortionCoeffs, points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    points.iter().map(|&p| distort(coeffs, p)).collect()
}

/// Serializes a coefficient block to its `std430` byte image (little-endian).
///
/// The five `f32` coefficients occupy the first five scalar slots in
/// declaration order (`k1`, `k2`, `k3`, `p1`, `p2`); the remainder of the two
/// `vec4` slots is zero padding.
#[must_use]
pub fn to_std430(coeffs: &DistortionCoeffs) -> [u8; LENS_DISTORTION_STD430_SIZE] {
    let mut bytes = [0u8; LENS_DISTORTION_STD430_SIZE];
    bytes[0..4].copy_from_slice(&coeffs.k1.to_le_bytes());
    bytes[4..8].copy_from_slice(&coeffs.k2.to_le_bytes());
    bytes[8..12].copy_from_slice(&coeffs.k3.to_le_bytes());
    bytes[12..16].copy_from_slice(&coeffs.p1.to_le_bytes());
    bytes[16..20].copy_from_slice(&coeffs.p2.to_le_bytes());
    bytes
}

/// Total `std430` byte size of a storage buffer holding `count` packed
/// coefficient blocks, clamped up to a single element per the shared
/// [`storage_bytes`] rule so a `WebGPU` binding is never zero-sized.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(LENS_DISTORTION_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for forward/inverse round-trip comparisons.
    const ROUNDTRIP_EPS: f32 = 1e-4;

    /// Near-exact tolerance used where the arithmetic is expected to be exact.
    const EXACT_EPS: f32 = 1e-7;

    /// Asserts two coordinates agree componentwise within `eps`.
    fn assert_close(a: [f32; 2], b: [f32; 2], eps: f32) {
        assert!((a[0] - b[0]).abs() < eps, "x: {} vs {}", a[0], b[0]);
        assert!((a[1] - b[1]).abs() < eps, "y: {} vs {}", a[1], b[1]);
    }

    /// A representative barrel lens (negative leading radial term).
    fn barrel() -> DistortionCoeffs {
        DistortionCoeffs::new(-0.25, 0.06, -0.01, 0.0, 0.0)
    }

    /// A representative pincushion lens (positive leading radial term).
    fn pincushion() -> DistortionCoeffs {
        DistortionCoeffs::new(0.2, 0.03, 0.0, 0.0, 0.0)
    }

    /// A lens with a decentring (tangential) component.
    fn decentred() -> DistortionCoeffs {
        DistortionCoeffs::new(-0.15, 0.02, 0.0, 0.01, -0.008)
    }

    #[test]
    fn none_is_all_zero() {
        let c = DistortionCoeffs::none();
        assert!(c.k1.abs() < EXACT_EPS);
        assert!(c.k2.abs() < EXACT_EPS);
        assert!(c.k3.abs() < EXACT_EPS);
        assert!(c.p1.abs() < EXACT_EPS);
        assert!(c.p2.abs() < EXACT_EPS);
    }

    #[test]
    fn default_equals_none() {
        assert_eq!(DistortionCoeffs::default(), DistortionCoeffs::none());
    }

    #[test]
    fn new_stores_fields() {
        let c = DistortionCoeffs::new(1.0, 2.0, 3.0, 4.0, 5.0);
        assert!((c.k1 - 1.0).abs() < EXACT_EPS);
        assert!((c.k2 - 2.0).abs() < EXACT_EPS);
        assert!((c.k3 - 3.0).abs() < EXACT_EPS);
        assert!((c.p1 - 4.0).abs() < EXACT_EPS);
        assert!((c.p2 - 5.0).abs() < EXACT_EPS);
    }

    #[test]
    fn identity_distort_is_noop() {
        let c = DistortionCoeffs::none();
        for &p in &[[0.0, 0.0], [0.3, -0.4], [-0.7, 0.1], [0.5, 0.5]] {
            assert_close(distort(&c, p), p, EXACT_EPS);
        }
    }

    #[test]
    fn identity_undistort_is_noop() {
        let c = DistortionCoeffs::none();
        for &p in &[[0.0, 0.0], [0.3, -0.4], [-0.7, 0.1]] {
            assert_close(undistort(&c, p), p, EXACT_EPS);
        }
    }

    #[test]
    fn center_fixed_under_distort() {
        for c in &[barrel(), pincushion(), decentred()] {
            assert_close(distort(c, [0.0, 0.0]), [0.0, 0.0], EXACT_EPS);
        }
    }

    #[test]
    fn center_fixed_under_undistort() {
        for c in &[barrel(), pincushion(), decentred()] {
            assert_close(undistort(c, [0.0, 0.0]), [0.0, 0.0], EXACT_EPS);
        }
    }

    #[test]
    fn roundtrip_barrel() {
        let c = barrel();
        for &p in &[[0.1, 0.0], [0.3, -0.4], [-0.5, 0.2], [0.6, 0.6]] {
            let there = distort(&c, p);
            let back = undistort(&c, there);
            assert_close(back, p, ROUNDTRIP_EPS);
        }
    }

    #[test]
    fn roundtrip_pincushion() {
        let c = pincushion();
        for &p in &[[0.1, 0.0], [0.3, -0.4], [-0.5, 0.2]] {
            let there = distort(&c, p);
            let back = undistort(&c, there);
            assert_close(back, p, ROUNDTRIP_EPS);
        }
    }

    #[test]
    fn roundtrip_decentred() {
        let c = decentred();
        for &p in &[[0.1, 0.05], [0.3, -0.4], [-0.45, 0.2]] {
            let there = distort(&c, p);
            let back = undistort(&c, there);
            assert_close(back, p, ROUNDTRIP_EPS);
        }
    }

    #[test]
    fn inverse_roundtrip_distort_of_undistort() {
        let c = barrel();
        for &p in &[[0.2, 0.1], [-0.35, 0.25]] {
            let ideal = undistort(&c, p);
            let observed = distort(&c, ideal);
            assert_close(observed, p, ROUNDTRIP_EPS);
        }
    }

    #[test]
    fn pincushion_pushes_outward() {
        let c = pincushion();
        for &r in &[0.2_f32, 0.5, 0.8] {
            assert!(distort_radius(&c, r) > r);
        }
    }

    #[test]
    fn barrel_pulls_inward() {
        let c = barrel();
        for &r in &[0.2_f32, 0.5, 0.8] {
            assert!(distort_radius(&c, r) < r);
        }
    }

    #[test]
    fn distort_radius_zero_is_zero() {
        assert!(distort_radius(&barrel(), 0.0).abs() < EXACT_EPS);
        assert!(distort_radius(&pincushion(), 0.0).abs() < EXACT_EPS);
    }

    #[test]
    fn distort_radius_identity() {
        let c = DistortionCoeffs::none();
        for &r in &[0.0_f32, 0.25, 0.5, 1.0] {
            assert!((distort_radius(&c, r) - r).abs() < EXACT_EPS);
        }
    }

    #[test]
    fn pincushion_radius_monotonic() {
        let c = pincushion();
        let mut prev = distort_radius(&c, 0.0);
        for &r in &[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] {
            let cur = distort_radius(&c, r);
            assert!(cur > prev, "not increasing at r={r}");
            prev = cur;
        }
    }

    #[test]
    fn radial_matches_axis_distort() {
        // Along the +x axis with no tangential term, distort's x-component is
        // exactly the scalar radial map.
        let c = pincushion();
        for &r in &[0.2_f32, 0.5, 0.75] {
            let scalar = distort_radius(&c, r);
            let full = distort(&c, [r, 0.0]);
            assert!((full[0] - scalar).abs() < EXACT_EPS);
            assert!(full[1].abs() < EXACT_EPS);
        }
    }

    #[test]
    fn tangential_zero_when_p_zero() {
        // A purely radial lens leaves a point on the diagonal on its diagonal.
        let c = pincushion();
        let p = [0.4, 0.4];
        let d = distort(&c, p);
        assert!((d[0] - d[1]).abs() < EXACT_EPS);
    }

    #[test]
    fn tangential_breaks_symmetry() {
        // With a tangential term the diagonal symmetry is broken.
        let c = decentred();
        let p = [0.4, 0.4];
        let d = distort(&c, p);
        assert!((d[0] - d[1]).abs() > EXACT_EPS);
    }

    #[test]
    fn higher_order_terms_matter() {
        let base = DistortionCoeffs::new(0.2, 0.0, 0.0, 0.0, 0.0);
        let with_k2 = DistortionCoeffs::new(0.2, 0.3, 0.0, 0.0, 0.0);
        let r = 0.7;
        assert!((distort_radius(&base, r) - distort_radius(&with_k2, r)).abs() > 1e-3);
    }

    #[test]
    fn grid_preserves_length() {
        let c = barrel();
        let pts = [[0.1, 0.2], [0.3, -0.1], [-0.4, 0.35]];
        assert_eq!(distort_grid(&c, &pts).len(), pts.len());
    }

    #[test]
    fn grid_empty_is_empty() {
        let c = barrel();
        let pts: [[f32; 2]; 0] = [];
        assert!(distort_grid(&c, &pts).is_empty());
    }

    #[test]
    fn grid_matches_scalar_distort() {
        let c = decentred();
        let pts = [[0.1, 0.2], [0.3, -0.1], [-0.4, 0.35], [0.0, 0.0]];
        let grid = distort_grid(&c, &pts);
        for (g, &p) in grid.iter().zip(pts.iter()) {
            assert_close(*g, distort(&c, p), EXACT_EPS);
        }
    }

    #[test]
    fn std430_size_is_thirty_two() {
        let bytes = to_std430(&barrel());
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn std430_layout_is_declaration_order() {
        let c = DistortionCoeffs::new(0.1, 0.2, 0.3, 0.4, 0.5);
        let bytes = to_std430(&c);
        assert_eq!(&bytes[0..4], &c.k1.to_le_bytes());
        assert_eq!(&bytes[4..8], &c.k2.to_le_bytes());
        assert_eq!(&bytes[8..12], &c.k3.to_le_bytes());
        assert_eq!(&bytes[12..16], &c.p1.to_le_bytes());
        assert_eq!(&bytes[16..20], &c.p2.to_le_bytes());
    }

    #[test]
    fn std430_tail_is_zero_padded() {
        let bytes = to_std430(&DistortionCoeffs::new(1.0, 2.0, 3.0, 4.0, 5.0));
        assert!(bytes[20..].iter().all(|&b| b == 0));
    }

    #[test]
    fn gpu_storage_bytes_scales_and_clamps() {
        assert_eq!(gpu_storage_bytes(0), 32);
        assert_eq!(gpu_storage_bytes(1), 32);
        assert_eq!(gpu_storage_bytes(8), 8 * 32);
    }

    #[test]
    fn std430_size_matches_two_vec4() {
        assert_eq!(gpu_storage_bytes(1), 2 * VEC4_STRIDE);
    }
}
