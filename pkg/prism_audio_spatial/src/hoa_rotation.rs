//! Higher-order Ambisonics (HOA) soundfield rotation via real spherical-
//! harmonic rotation matrices.
//!
//! Rotating a scene-based Ambisonic bus lets a decoder keep a source fixed in
//! the world while the listener turns their head (or a whole recorded field is
//! reoriented), *without* re-encoding every source from scratch. This module
//! rotates a buffer of `SN3D`/`ACN` real spherical-harmonic coefficients (the
//! same convention produced by [`crate::hoa`]) in place, up to
//! [`crate::hoa::MAX_HOA_ORDER`] (third order, 16 channels).
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. The real
//! spherical-harmonic rotation is built with the **Ivanic-Ruedenberg**
//! recurrence (J. Ivanic and K. Ruedenberg, *J. Phys. Chem.* 1996, with the
//! 1998 sign/normalisation erratum), a publicly documented method for
//! constructing degree-`l` real SH rotation matrices from the degree-1 block
//! and the degree-`(l - 1)` block via the standard `U`/`V`/`W` coefficients.
//! Everything here is implemented from that public literature.
//!
//! # Method
//!
//! Real spherical harmonics of a fixed degree `l` transform among themselves
//! under a 3D rotation, so a rotation acts block-diagonally on an `ACN` buffer:
//! degree `l` occupies the contiguous channels
//! [`acn_index`]`(l, -l) ..= acn_index(l, l)` and is mixed by a
//! `(2l + 1) x (2l + 1)` matrix `R^l`. The zeroth-order (`W`) block is the
//! `1 x 1` identity (an omni component is rotation invariant).
//!
//! The degree-1 block `R^1` is derived directly from the rotation's `Mat3`.
//! Because the three first-order `SN3D`/`ACN` harmonics are exactly the
//! (sign-flipped) listener-local Cartesian axes used by [`crate::hoa`]
//! (`ACN 1,2,3 = left, up, front = -x, +y, -z`), the first-order coefficient
//! vector is `c = A d` with `A = diag(-1, +1, -1)`. Rotating the *field* sends
//! a source direction `d` to `Q d` (with `Q` the rotation matrix), so the
//! coefficients transform by `R^1 = A Q A^-1 = A Q A` (since `A^2 = I`), i.e.
//! `R^1[i][j] = sign[i] * sign[j] * Q[i][j]` with `sign = (-1, +1, -1)`. Higher
//! blocks `R^2`, `R^3` are then generated from `R^1` and the previous block by
//! the Ivanic-Ruedenberg recurrence.
//!
//! # Coordinate / axis convention
//!
//! Identical to [`crate::hoa`]: directions are listener-local and match Bevy
//! (`-Z` forward, `+X` right, `+Y` up); the acoustic axes are
//! `front = -z`, `left = -x`, `up = +y`. The golden equivalence this module
//! guarantees is "rotate the coefficients" == "rotate the direction, then
//! encode": `rotate_hoa(encode_hoa(d), q) == encode_hoa(q * d)`.
//!
//! # Real-time contract
//!
//! [`HoaRotationMatrix::apply`] and [`rotate_hoa`]'s multiply step are
//! **allocation free, lock free, and panic free**: the rotation matrix lives in
//! a fixed-size stack array and the transform is a plain block matrix-vector
//! product. Building the matrix ([`HoaRotationMatrix::from_quat`]) is a
//! control-rate operation (once per rotation change) that likewise never
//! allocates, locks, or panics; a degenerate (zero / non-finite) quaternion
//! falls back to the identity rotation.
//!
//! # Determinism
//!
//! All transcendental / length math routes through [`bevy_math::ops`]
//! (libm-backed) rather than `f32` intrinsics, so rotation is bit-reproducible
//! across targets and can be golden-compared sample-for-sample. This is
//! enforced by the workspace lints.

use bevy_math::{Mat3, Quat, Vec3, ops};

use prism_audio_core::math::Sample;

use crate::hoa::{MAX_HOA_CHANNELS, MAX_HOA_ORDER, acn_index, hoa_channel_count};

/// The largest per-degree block dimension, `2 * MAX_HOA_ORDER + 1` (third
/// order gives a `7 x 7` block).
const MAX_BLOCK_DIM: usize = 2 * MAX_HOA_ORDER + 1;

/// The three signs that map the first-order `SN3D`/`ACN` harmonics
/// (`left, up, front`) to the Cartesian axes (`-x, +y, -z`); see the module
/// docs. Indexed by the local block index `m + 1` for `m` in `-1 ..= 1`.
const FIRST_ORDER_SIGNS: [Sample; 3] = [-1.0, 1.0, -1.0];

/// Returns component `i` (`0 = x`, `1 = y`, `2 = z`) of a [`Vec3`].
#[inline]
fn component(v: Vec3, i: usize) -> Sample {
    match i {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Reads `R^1[a][b]` for signed orders `a, b` in `-1 ..= 1`, returning `0` for
/// any out-of-range index so the recurrence stays panic free.
#[inline]
fn r1_get(r1: &[[Sample; 3]; 3], a: isize, b: isize) -> Sample {
    let ai = match a {
        -1 => 0usize,
        0 => 1,
        1 => 2,
        _ => return 0.0,
    };
    let bi = match b {
        -1 => 0usize,
        0 => 1,
        1 => 2,
        _ => return 0.0,
    };
    r1[ai][bi]
}

/// Reads the previous-degree block value `R^(l-1)[a][b]` where the block is
/// stored with offset `bound = l - 1` (so index `a + bound`), returning `0` for
/// any index outside `-bound ..= bound`. This keeps the recurrence both
/// panic free and correct: out-of-block terms are exactly the ones the
/// `u`/`v`/`w` coefficients multiply by zero.
#[inline]
fn block_get(block: &[[Sample; MAX_BLOCK_DIM]; MAX_BLOCK_DIM], bound: isize, a: isize, b: isize) -> Sample {
    if a < -bound || a > bound || b < -bound || b > bound {
        return 0.0;
    }
    let ai = usize::try_from(a + bound).unwrap_or(0);
    let bi = usize::try_from(b + bound).unwrap_or(0);
    block[ai][bi]
}

/// The Ivanic-Ruedenberg `P` helper: `P^l_{i,a,b}` built from the first-order
/// block and the previous-degree block.
#[inline]
fn p_term(
    i: isize,
    a: isize,
    b: isize,
    l: isize,
    r1: &[[Sample; 3]; 3],
    prev: &[[Sample; MAX_BLOCK_DIM]; MAX_BLOCK_DIM],
) -> Sample {
    let lprev = l - 1;
    if b == l {
        r1_get(r1, i, 1) * block_get(prev, lprev, a, lprev)
            - r1_get(r1, i, -1) * block_get(prev, lprev, a, -lprev)
    } else if b == -l {
        r1_get(r1, i, 1) * block_get(prev, lprev, a, -lprev)
            + r1_get(r1, i, -1) * block_get(prev, lprev, a, lprev)
    } else {
        r1_get(r1, i, 0) * block_get(prev, lprev, a, b)
    }
}

/// The Ivanic-Ruedenberg `U` term, `U^l_{m,n} = P^l_{0,m,n}`.
#[inline]
fn u_term(
    m: isize,
    n: isize,
    l: isize,
    r1: &[[Sample; 3]; 3],
    prev: &[[Sample; MAX_BLOCK_DIM]; MAX_BLOCK_DIM],
) -> Sample {
    p_term(0, m, n, l, r1, prev)
}

/// The Ivanic-Ruedenberg `V` term.
#[inline]
fn v_term(
    m: isize,
    n: isize,
    l: isize,
    r1: &[[Sample; 3]; 3],
    prev: &[[Sample; MAX_BLOCK_DIM]; MAX_BLOCK_DIM],
) -> Sample {
    if m == 0 {
        p_term(1, 1, n, l, r1, prev) + p_term(-1, -1, n, l, r1, prev)
    } else if m > 0 {
        let d: Sample = if m == 1 { 1.0 } else { 0.0 };
        p_term(1, m - 1, n, l, r1, prev) * ops::sqrt(1.0 + d)
            - p_term(-1, -m + 1, n, l, r1, prev) * (1.0 - d)
    } else {
        let d: Sample = if m == -1 { 1.0 } else { 0.0 };
        p_term(1, m + 1, n, l, r1, prev) * (1.0 - d)
            + p_term(-1, -m - 1, n, l, r1, prev) * ops::sqrt(1.0 + d)
    }
}

/// The Ivanic-Ruedenberg `W` term (only evaluated when its coefficient `w` is
/// non-zero, i.e. for `m != 0`).
#[inline]
fn w_term(
    m: isize,
    n: isize,
    l: isize,
    r1: &[[Sample; 3]; 3],
    prev: &[[Sample; MAX_BLOCK_DIM]; MAX_BLOCK_DIM],
) -> Sample {
    if m > 0 {
        p_term(1, m + 1, n, l, r1, prev) + p_term(-1, -m - 1, n, l, r1, prev)
    } else if m < 0 {
        p_term(1, m - 1, n, l, r1, prev) - p_term(-1, -m + 1, n, l, r1, prev)
    } else {
        0.0
    }
}

/// Builds the first-order rotation block `R^1` from a rotation matrix `Q`.
///
/// `R^1[i][j] = sign[i] * sign[j] * Q[i][j]` with `sign = (-1, +1, -1)`, mapping
/// the `SN3D`/`ACN` first-order harmonics (`left, up, front`) through the
/// listener-local Cartesian axes (`-x, +y, -z`). Local block index `i`
/// corresponds to order `m = i - 1`.
fn first_order_block(q: Mat3) -> [[Sample; 3]; 3] {
    let cols = [q.x_axis, q.y_axis, q.z_axis];
    let mut r1 = [[0.0 as Sample; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            // Q[i][j] is component i of column j (glam is column-major).
            let q_ij = component(cols[j], i);
            r1[i][j] = FIRST_ORDER_SIGNS[i] * FIRST_ORDER_SIGNS[j] * q_ij;
        }
    }
    r1
}

/// A precomputed real spherical-harmonic rotation matrix for an `SN3D`/`ACN`
/// Ambisonic buffer, laid out block-diagonally by degree up to
/// [`MAX_HOA_ORDER`].
///
/// Build it once per rotation change with [`HoaRotationMatrix::from_quat`]
/// (control rate), then rotate any number of coefficient frames in place with
/// [`HoaRotationMatrix::apply`] (real-time, allocation/lock/panic free).
///
/// ```
/// use bevy_math::{Quat, Vec3};
/// use core::f32::consts::FRAC_PI_2;
/// use prism_audio_spatial::hoa::{encode_hoa, MAX_HOA_CHANNELS};
/// use prism_audio_spatial::hoa_rotation::HoaRotationMatrix;
///
/// // Encode a source straight ahead (Bevy -Z) at third order.
/// let mut coeffs = [0.0f32; MAX_HOA_CHANNELS];
/// let n = encode_hoa(Vec3::new(0.0, 0.0, -1.0), 3, &mut coeffs);
///
/// // Rotating the field by +90 deg about +Y should equal encoding the source
/// // at the rotated direction.
/// let q = Quat::from_rotation_y(FRAC_PI_2);
/// let rot = HoaRotationMatrix::from_quat(3, q);
/// rot.apply(&mut coeffs[..n]);
///
/// let mut expected = [0.0f32; MAX_HOA_CHANNELS];
/// encode_hoa(q * Vec3::new(0.0, 0.0, -1.0), 3, &mut expected);
/// for (a, b) in coeffs.iter().zip(expected.iter()) {
///     assert!((a - b).abs() < 1.0e-4);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct HoaRotationMatrix {
    matrix: [[Sample; MAX_HOA_CHANNELS]; MAX_HOA_CHANNELS],
    channels: usize,
    order: usize,
}

impl HoaRotationMatrix {
    /// Builds the rotation matrix for the given `order` (clamped to
    /// [`MAX_HOA_ORDER`]) from `rotation`.
    ///
    /// **Control rate / non-real-time** but still allocation, lock, and panic
    /// free. A non-finite or (near) zero-length quaternion is treated as the
    /// identity rotation.
    #[must_use]
    #[expect(
        clippy::needless_range_loop,
        reason = "the Ivanic-Ruedenberg recurrence and block placement walk paired (row, col) SH orders with signed offset arithmetic; explicit indices are clearer and match the neighbouring hoa module"
    )]
    pub fn from_quat(order: usize, rotation: Quat) -> Self {
        let order = order.min(MAX_HOA_ORDER);
        let channels = hoa_channel_count(order);

        let mut matrix = [[0.0 as Sample; MAX_HOA_CHANNELS]; MAX_HOA_CHANNELS];
        // Zeroth order: the omni W component is rotation invariant.
        matrix[0][0] = 1.0;
        if order == 0 {
            return Self { matrix, channels, order };
        }

        // Sanitise the quaternion off the hot path: guard against non-finite or
        // zero-length input (which normalise would turn into NaN).
        let q = if rotation.is_finite() && rotation.length_squared() > 1.0e-12 {
            rotation.normalize()
        } else {
            Quat::IDENTITY
        };
        let mat = Mat3::from_quat(q);
        let r1 = first_order_block(mat);

        // Place R^1 into the big matrix at the first-order ACN offset and seed
        // the previous-block store.
        let mut prev = [[0.0 as Sample; MAX_BLOCK_DIM]; MAX_BLOCK_DIM];
        for i in 0..3 {
            for j in 0..3 {
                prev[i][j] = r1[i][j];
                matrix[acn_index(1, i as isize - 1)][acn_index(1, j as isize - 1)] = r1[i][j];
            }
        }

        // Build higher-degree blocks by the Ivanic-Ruedenberg recurrence.
        for l in 2..=order {
            let li = l as isize;
            let mut cur = [[0.0 as Sample; MAX_BLOCK_DIM]; MAX_BLOCK_DIM];
            let dim = 2 * l + 1;
            for mi in 0..dim {
                let m = mi as isize - li;
                for ni in 0..dim {
                    let n = ni as isize - li;

                    let lf = l as Sample;
                    let mf = m as Sample;
                    let nf = n as Sample;
                    let am = if m < 0 { -m } else { m };
                    let amf = am as Sample;
                    let dd: Sample = if m == 0 { 1.0 } else { 0.0 };

                    let denom = if n == li || n == -li {
                        2.0 * lf * (2.0 * lf - 1.0)
                    } else {
                        (lf + nf) * (lf - nf)
                    };

                    let u = ops::sqrt((lf * lf - mf * mf) / denom);
                    let v = 0.5
                        * ops::sqrt((1.0 + dd) * (lf + amf - 1.0) * (lf + amf) / denom)
                        * (1.0 - 2.0 * dd);
                    let w = -0.5 * ops::sqrt((lf - amf - 1.0) * (lf - amf) / denom) * (1.0 - dd);

                    let mut value = 0.0 as Sample;
                    if u != 0.0 {
                        value += u * u_term(m, n, li, &r1, &prev);
                    }
                    if v != 0.0 {
                        value += v * v_term(m, n, li, &r1, &prev);
                    }
                    if w != 0.0 {
                        value += w * w_term(m, n, li, &r1, &prev);
                    }
                    cur[mi][ni] = value;
                    matrix[acn_index(l, m)][acn_index(l, n)] = value;
                }
            }
            prev = cur;
        }

        Self { matrix, channels, order }
    }

    /// The Ambisonic order this matrix rotates.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// The number of `ACN` channels this matrix spans, `(order + 1)^2`.
    #[inline]
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Rotates `coeffs` in place (block matrix-vector product).
    ///
    /// Only the leading `min(channels, coeffs.len())` channels are touched, so
    /// a shorter buffer never panics; trailing channels are left unchanged.
    /// **Real-time**: allocation, lock, and panic free.
    #[expect(
        clippy::needless_range_loop,
        reason = "a dense matrix-vector product over paired (row, col) channel indices reads more clearly with explicit indices than zipped iterators"
    )]
    pub fn apply(&self, coeffs: &mut [Sample]) {
        let n = self.channels.min(coeffs.len());
        let mut out = [0.0 as Sample; MAX_HOA_CHANNELS];
        for i in 0..n {
            let mut acc = 0.0 as Sample;
            for j in 0..n {
                acc += self.matrix[i][j] * coeffs[j];
            }
            out[i] = acc;
        }
        coeffs[..n].copy_from_slice(&out[..n]);
    }
}

/// Rotates an `SN3D`/`ACN` Ambisonic coefficient buffer in place by `rotation`.
///
/// `order` is clamped to [`MAX_HOA_ORDER`]; at most `(order + 1)^2` leading
/// channels are transformed, so a shorter `coeffs` slice is safe. This is a
/// convenience wrapper that builds a [`HoaRotationMatrix`] on the stack and
/// applies it once; for repeated rotation by the same quaternion across many
/// frames, cache a [`HoaRotationMatrix`] and call
/// [`HoaRotationMatrix::apply`] instead.
///
/// Allocation and lock free; panic free for any input.
#[inline]
pub fn rotate_hoa(coeffs: &mut [Sample], order: usize, rotation: Quat) {
    let matrix = HoaRotationMatrix::from_quat(order, rotation);
    matrix.apply(coeffs);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hoa::encode_hoa;
    use bevy_math::Quat;
    use core::f32::consts::{FRAC_PI_2, PI};

    const EPS: Sample = 1.0e-4;

    fn approx(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= EPS
    }

    fn assert_rotation_matches_encode(dir: Vec3, q: Quat, order: usize) {
        let mut rotated = [0.0 as Sample; MAX_HOA_CHANNELS];
        let n = encode_hoa(dir, order, &mut rotated);
        rotate_hoa(&mut rotated[..n], order, q);

        let mut expected = [0.0 as Sample; MAX_HOA_CHANNELS];
        let m = encode_hoa(q * dir, order, &mut expected);
        assert_eq!(n, m);
        for k in 0..n {
            assert!(
                approx(rotated[k], expected[k]),
                "order {order} channel {k}: {} vs {}",
                rotated[k],
                expected[k]
            );
        }
    }

    #[test]
    fn identity_quaternion_leaves_coeffs_unchanged() {
        let dir = Vec3::new(0.3, -0.6, 0.74).normalize();
        for order in 0..=MAX_HOA_ORDER {
            let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
            let n = encode_hoa(dir, order, &mut coeffs);
            let before = coeffs;
            rotate_hoa(&mut coeffs[..n], order, Quat::IDENTITY);
            for k in 0..n {
                assert!(approx(coeffs[k], before[k]), "channel {k} moved");
            }
        }
    }

    #[test]
    fn rotate_field_equals_rotate_direction_about_y() {
        let dirs = [
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-0.3, 0.5, -0.8).normalize(),
        ];
        let angles = [FRAC_PI_2, PI, -FRAC_PI_2, 0.37];
        for &dir in &dirs {
            for &a in &angles {
                let q = Quat::from_rotation_y(a);
                for order in 1..=MAX_HOA_ORDER {
                    assert_rotation_matches_encode(dir, q, order);
                }
            }
        }
    }

    #[test]
    fn rotate_field_equals_rotate_direction_about_x_and_z() {
        let dir = Vec3::new(0.2, -0.5, -0.84).normalize();
        for &a in &[FRAC_PI_2, PI, 0.9] {
            for q in [Quat::from_rotation_x(a), Quat::from_rotation_z(a)] {
                for order in 1..=MAX_HOA_ORDER {
                    assert_rotation_matches_encode(dir, q, order);
                }
            }
        }
    }

    #[test]
    fn composite_rotation_matches_encode() {
        let q = Quat::from_rotation_y(0.6) * Quat::from_rotation_x(-0.4) * Quat::from_rotation_z(1.1);
        let dir = Vec3::new(0.5, 0.5, -0.5).normalize();
        for order in 1..=MAX_HOA_ORDER {
            assert_rotation_matches_encode(dir, q, order);
        }
    }

    #[test]
    fn zeroth_order_w_is_invariant() {
        let q = Quat::from_rotation_y(1.234) * Quat::from_rotation_x(0.5);
        let mut coeffs = [2.5 as Sample; MAX_HOA_CHANNELS];
        rotate_hoa(&mut coeffs[..1], 0, q);
        assert!(approx(coeffs[0], 2.5), "W changed: {}", coeffs[0]);
    }

    #[test]
    fn per_degree_energy_is_conserved() {
        let dir = Vec3::new(-0.4, 0.7, -0.59).normalize();
        let q = Quat::from_rotation_y(0.8) * Quat::from_rotation_z(-0.3);
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        let n = encode_hoa(dir, MAX_HOA_ORDER, &mut coeffs);
        let before = coeffs;
        rotate_hoa(&mut coeffs[..n], MAX_HOA_ORDER, q);

        // A rotation is orthonormal within each degree block: the per-degree
        // sum of squares is preserved.
        for l in 0..=MAX_HOA_ORDER {
            let lo = l * l;
            let hi = (l + 1) * (l + 1);
            let mut e0 = 0.0 as Sample;
            let mut e1 = 0.0 as Sample;
            for k in lo..hi {
                e0 += before[k] * before[k];
                e1 += coeffs[k] * coeffs[k];
            }
            assert!((e0 - e1).abs() <= 1.0e-3, "degree {l}: {e0} vs {e1}");
        }
    }

    #[test]
    fn order_is_clamped_to_max() {
        let m = HoaRotationMatrix::from_quat(99, Quat::from_rotation_y(0.5));
        assert_eq!(m.order(), MAX_HOA_ORDER);
        assert_eq!(m.channels(), MAX_HOA_CHANNELS);
    }

    #[test]
    fn short_buffer_does_not_panic() {
        let q = Quat::from_rotation_y(FRAC_PI_2);
        // Only two channels available for a nominal third-order rotation.
        let mut coeffs = [1.0 as Sample; 2];
        rotate_hoa(&mut coeffs, 3, q);
        // W (index 0) is invariant; index 1 is inside the first-order block.
        assert!(coeffs[0].is_finite());
        assert!(coeffs[1].is_finite());
    }

    #[test]
    fn degenerate_quaternion_falls_back_to_identity() {
        let dir = Vec3::new(0.1, -0.2, -0.97).normalize();
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        let n = encode_hoa(dir, MAX_HOA_ORDER, &mut coeffs);
        let before = coeffs;
        // A zero quaternion is non-normalisable and must not produce NaNs.
        rotate_hoa(&mut coeffs[..n], MAX_HOA_ORDER, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0));
        for k in 0..n {
            assert!(coeffs[k].is_finite());
            assert!(approx(coeffs[k], before[k]), "channel {k} moved on identity fallback");
        }
    }

    #[test]
    fn matrix_can_be_reused_across_frames() {
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let q = Quat::from_rotation_y(FRAC_PI_2);
        let rot = HoaRotationMatrix::from_quat(2, q);

        let mut expected = [0.0 as Sample; MAX_HOA_CHANNELS];
        let n = encode_hoa(q * dir, 2, &mut expected);

        for _ in 0..3 {
            let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
            encode_hoa(dir, 2, &mut coeffs);
            rot.apply(&mut coeffs[..n]);
            for k in 0..n {
                assert!(approx(coeffs[k], expected[k]), "channel {k}");
            }
        }
    }
}
