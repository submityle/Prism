//! `CIE` 1976 `L*u*v*` (`CIELUV`) contract for the particle color pipeline: the
//! perceptually-uniform space whose *chromaticity* leg (`u'`, `v'`) is an affine
//! projection of `CIE` `XYZ`, which is why additive-light tooling (bloom tints,
//! emitter glow mixing) prefers it over `CIELAB` for saturation and simple
//! same-lightness blends.
//!
//! Where [`crate::particle::cielab`] owns the `CIELAB` cube-root opponent space
//! and its `ΔE76` / `ΔE94` metrics, and [`crate::particle::cie_xyz`] owns the
//! `sRGB` primaries and `Bradford` adaptation, this module is deliberately
//! narrow and *self-contained*: it converts `CIE` `XYZ` (already adapted to the
//! `D65` reference white by the caller) to and from `CIELUV`, exposes the `u'v'`
//! chromaticity helper, and computes the chroma `C*uv`, the saturation `s_uv`,
//! and the `CIELUV` color difference `ΔE*uv`. It defines its *own* [`Xyz`],
//! [`Luv`], [`UvPrime`], and [`cbrt_newton`], sharing no types with its `CIELAB`
//! or `Oklab` siblings.
//!
//! The lightness transfer is the standard `CIE` piecewise curve
//! `L* = 116 f(Y/Yn) − 16`, with `f` taking the cube root above the knot
//! `δ³ = (6/29)³` and a linear toe `t / (3δ²) + 4/29` at or below it, so the two
//! legs meet continuously at `f(δ³) = δ`. The chromaticity leg is the exact
//! affine map `u* = 13 L* (u' − u'n)`, `v* = 13 L* (v' − v'n)`, which the inverse
//! undoes precisely.
//!
//! Determinism rules (design §29) hold: the arithmetic is pure `+ − * /` plus
//! `f32::sqrt`, `f32::abs`, `f32::min`, `f32::max`, and `f32::clamp`. There are
//! no transcendental calls (`sin` / `cos` / `exp` / `ln` / `pow` / `cbrt`); the
//! cube root is a self-contained Newton iteration in [`cbrt_newton`], every
//! threshold is a named constant, and `f32` equality is only ever tested through
//! the [`CMP_EPS`] guard, never with `==` / `!=`. The `std430` packing mirrors an
//! aligned `vec4` slot so the future `GPU` color kernel binds against a stable
//! `ABI` and reproduces this `CPU` reference bit for bit.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Byte size of the `std430` packing of a [`Luv`] triple: three `f32` scalars
/// promoted to one aligned `vec4` slot (16 bytes), leaving a one-scalar padding
/// tail so the block honors the `std430` 16-byte base alignment.
pub const CIELUV_STD430_SIZE: usize = VEC4_STRIDE;

/// Absolute tolerance for the guarded `f32` comparisons; direct `==` / `!=` on
/// floating point is intentionally avoided in both production and test code.
pub const CMP_EPS: f32 = 1.0e-6;

/// The shared `CIE` knot `δ = 6/29`; the lightness curve switches from its
/// linear toe to the cube-root leg here. Written as a division of named integers
/// so it is never mistaken for a bare approximation of a mathematical constant.
const DELTA: f32 = 6.0 / 29.0;

/// The knot in domain space, `δ³`; the forward nonlinearity uses the cube root
/// above this input and the linear toe at or below it.
const DELTA_CUBED: f32 = DELTA * DELTA * DELTA;

/// The linear-toe slope denominator `3δ²`. The forward toe divides by it and the
/// inverse toe multiplies by it, keeping the two legs exact inverses.
const THREE_DELTA_SQ: f32 = 3.0 * DELTA * DELTA;

/// The linear-toe offset `4/29`, the value both legs take at `t = 0`.
const TOE_OFFSET: f32 = 4.0 / 29.0;

/// Number of Newton refinement steps [`cbrt_newton`] runs after range reduction;
/// the reduced input lies in `[1, 8)` where the quadratically convergent
/// iteration reaches `f32` precision well within this budget.
const CBRT_ITERATIONS: usize = 16;

/// Upper bound of the [`cbrt_newton`] range-reduction window: inputs above it are
/// divided by eight (halving the cube root) until they fall inside `[1, 8)`.
const CBRT_WINDOW_HI: f32 = 8.0;

/// Lower bound of the [`cbrt_newton`] range-reduction window: inputs below it are
/// multiplied by eight (doubling the cube root) until they reach `[1, 8)`.
const CBRT_WINDOW_LO: f32 = 1.0;

/// `X` tristimulus of the `CIE` `D65` reference white, normalized to `Y = 1`.
const XN: f32 = 0.950_489;

/// `Y` (luminance) of the `CIE` `D65` reference white, unity by construction.
const YN: f32 = 1.0;

/// `Z` tristimulus of the `CIE` `D65` reference white, normalizing the blue axis.
const ZN: f32 = 0.888_840;

/// The `CIELUV` chromaticity denominator `X + 15Y + 3Z` evaluated at the `D65`
/// reference white, reused to derive the white-point chromaticity coordinates.
const D65_DENOM: f32 = XN + 15.0 * YN + 3.0 * ZN;

/// The reference-white chromaticity `u'n = 4Xn / (Xn + 15Yn + 3Zn)`, the origin
/// the `u*` axis is measured from.
const UN_PRIME: f32 = 4.0 * XN / D65_DENOM;

/// The reference-white chromaticity `v'n = 9Yn / (Xn + 15Yn + 3Zn)`, the origin
/// the `v*` axis is measured from.
const VN_PRIME: f32 = 9.0 * YN / D65_DENOM;

/// A `CIE` `XYZ` tristimulus triple, owned by this module so the `CIELUV`
/// contract stays self-contained and shares no type with
/// [`crate::particle::cielab`] or [`crate::particle::cie_xyz`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xyz {
    /// The `X` tristimulus component.
    pub x: f32,
    /// The `Y` tristimulus component (luminance when the triple is normalized).
    pub y: f32,
    /// The `Z` tristimulus component.
    pub z: f32,
}

impl Xyz {
    /// Builds an [`Xyz`] triple from its three components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// A `CIE` 1976 `L*u*v*` color.
///
/// `l` is perceptual lightness in `[0, 100]` (`0` black, `100` reference white);
/// `u` and `v` are the chromaticity-difference coordinates, unbounded in
/// principle but small for in-gamut colors and zero on the neutral axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Luv {
    /// Perceptual lightness `L*`.
    pub l: f32,
    /// The `u*` chromaticity-difference coordinate (roughly red-green weighted).
    pub u: f32,
    /// The `v*` chromaticity-difference coordinate (roughly blue-yellow weighted).
    pub v: f32,
}

impl Luv {
    /// Builds a [`Luv`] color from its lightness and two chromaticity axes.
    #[must_use]
    pub const fn new(l: f32, u: f32, v: f32) -> Self {
        Self { l, u, v }
    }
}

/// A `CIELUV` chromaticity pair `(u', v')`, the affine projection of a `CIE`
/// `XYZ` triple onto the `1976` uniform-chromaticity-scale diagram.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvPrime {
    /// The `u'` chromaticity coordinate `4X / (X + 15Y + 3Z)`.
    pub u_prime: f32,
    /// The `v'` chromaticity coordinate `9Y / (X + 15Y + 3Z)`.
    pub v_prime: f32,
}

impl UvPrime {
    /// Builds a [`UvPrime`] chromaticity pair from its two coordinates.
    #[must_use]
    pub const fn new(u_prime: f32, v_prime: f32) -> Self {
        Self { u_prime, v_prime }
    }
}

/// Returns the reference-white chromaticity `(u'n, v'n)` used as the origin of
/// the `u*` / `v*` axes.
#[must_use]
pub fn reference_white_uv() -> UvPrime {
    UvPrime::new(UN_PRIME, VN_PRIME)
}

/// Returns the `CIE` `D65` reference white as an [`Xyz`] triple normalized to
/// `Y = 1`.
#[must_use]
pub fn d65_white() -> Xyz {
    Xyz::new(XN, YN, ZN)
}

/// Returns the real cube root of `x` using a self-contained Newton iteration,
/// avoiding the disallowed [`f32::cbrt`] intrinsic.
///
/// Negative inputs reflect through the origin (`cbrt(-x) = -cbrt(x)`) and a zero
/// input returns zero, guarding the division in the iteration. The magnitude is
/// range-reduced into `[1, 8)` by factors of eight — each division by eight
/// halves the cube root and each multiplication doubles it — so a fixed
/// [`CBRT_ITERATIONS`] budget of Newton steps `y ← y − (y³ − v) / (3y²)` reaches
/// `f32` precision regardless of the input's magnitude.
#[must_use]
pub fn cbrt_newton(x: f32) -> f32 {
    if x < 0.0 {
        return -cbrt_newton(-x);
    }
    if x <= 0.0 {
        return 0.0;
    }

    let mut v = x;
    let mut scale = 1.0_f32;
    while v > CBRT_WINDOW_HI {
        v /= 8.0;
        scale *= 2.0;
    }
    while v < CBRT_WINDOW_LO {
        v *= 8.0;
        scale /= 2.0;
    }

    let mut y = 1.5_f32;
    for _ in 0..CBRT_ITERATIONS {
        let denom = 3.0 * y * y;
        if denom < CMP_EPS {
            break;
        }
        y -= (y * y * y - v) / denom;
    }
    y * scale
}

/// The `CIE` lightness forward nonlinearity `f(t)`.
///
/// Above the knot `δ³` it is the cube root; at or below it the linear toe
/// `t / (3δ²) + 4/29` takes over. The two legs meet at `f(δ³) = δ`, so the
/// function is continuous across the knot.
#[must_use]
pub fn luv_f(t: f32) -> f32 {
    if t > DELTA_CUBED {
        cbrt_newton(t)
    } else {
        t / THREE_DELTA_SQ + TOE_OFFSET
    }
}

/// The exact inverse of [`luv_f`].
///
/// Above the knot `δ` it cubes its argument with an integer product `t·t·t`
/// (avoiding the disallowed power intrinsics); at or below it the inverse linear
/// toe `3δ² · (t − 4/29)` takes over. The two legs meet at `δ`, matching the
/// forward knot exactly.
#[must_use]
pub fn luv_f_inv(t: f32) -> f32 {
    if t > DELTA {
        t * t * t
    } else {
        THREE_DELTA_SQ * (t - TOE_OFFSET)
    }
}

/// Returns the `CIELUV` chromaticity `(u', v')` of a `CIE` `XYZ` triple.
///
/// The shared denominator `X + 15Y + 3Z` is guarded: when its magnitude falls
/// below [`CMP_EPS`] (a fully degenerate, black-or-negative triple) both
/// coordinates collapse to zero rather than dividing by a near-zero value.
#[must_use]
pub fn uv_prime(c: &Xyz) -> UvPrime {
    let denom = c.x + 15.0 * c.y + 3.0 * c.z;
    if denom.abs() < CMP_EPS {
        return UvPrime::new(0.0, 0.0);
    }
    UvPrime::new(4.0 * c.x / denom, 9.0 * c.y / denom)
}

/// Converts a `CIE` `XYZ` triple (measured relative to the `D65` white) into
/// `CIE` 1976 `L*u*v*`.
///
/// The lightness is `L* = 116 f(Y/Yn) − 16`; the chromaticity coordinates are
/// referenced to the white point through `u* = 13 L* (u' − u'n)` and
/// `v* = 13 L* (v' − v'n)`, so a neutral color (whose chromaticity equals the
/// white point's) yields `u* = v* = 0`.
#[must_use]
pub fn xyz_to_luv(c: &Xyz) -> Luv {
    let l = 116.0 * luv_f(c.y / YN) - 16.0;
    let uv = uv_prime(c);
    let scale = 13.0 * l;
    let u = scale * (uv.u_prime - UN_PRIME);
    let v = scale * (uv.v_prime - VN_PRIME);
    Luv::new(l, u, v)
}

/// Converts a `CIE` 1976 `L*u*v*` color back into a `CIE` `XYZ` triple relative
/// to the `D65` white, inverting [`xyz_to_luv`] exactly.
///
/// It recovers `Y = Yn · f⁻¹((L* + 16) / 116)`, then the chromaticity
/// `u' = u* / (13 L*) + u'n` and `v' = v* / (13 L*) + v'n`, and finally
/// `X = Y · 9u' / (4v')` and `Z = Y · (12 − 3u' − 20v') / (4v')`. A vanishing
/// `L*` short-circuits to pure black, and a vanishing `v'` yields the neutral
/// gray of that luminance, guarding both divisions.
#[must_use]
pub fn luv_to_xyz(c: &Luv) -> Xyz {
    let fy = (c.l + 16.0) / 116.0;
    let y = YN * luv_f_inv(fy);
    if c.l.abs() < CMP_EPS {
        return Xyz::new(0.0, 0.0, 0.0);
    }
    let inv = 1.0 / (13.0 * c.l);
    let u_prime = c.u * inv + UN_PRIME;
    let v_prime = c.v * inv + VN_PRIME;
    if v_prime.abs() < CMP_EPS {
        return Xyz::new(0.0, y, 0.0);
    }
    let x = y * (9.0 * u_prime) / (4.0 * v_prime);
    let z = y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * v_prime);
    Xyz::new(x, y, z)
}

/// The `CIELUV` chroma `C*uv = sqrt(u*² + v*²)`, the radial distance of a color
/// from the neutral axis. It is zero exactly on the neutral axis and never
/// negative.
#[must_use]
pub fn chroma(c: &Luv) -> f32 {
    (c.u * c.u + c.v * c.v).sqrt()
}

/// The `CIELUV` saturation `s_uv = C*uv / L*`, the chroma normalized by
/// lightness. A vanishing `L*` (black) has no defined hue, so the saturation is
/// reported as zero rather than dividing by a near-zero value.
#[must_use]
pub fn saturation(c: &Luv) -> f32 {
    if c.l.abs() < CMP_EPS {
        return 0.0;
    }
    chroma(c) / c.l
}

/// The `CIELUV` color difference `ΔE*uv`: the plain Euclidean distance between
/// two `CIELUV` colors. It is symmetric and zero exactly when the colors match.
#[must_use]
pub fn delta_e_uv(a: &Luv, b: &Luv) -> f32 {
    let dl = a.l - b.l;
    let du = a.u - b.u;
    let dv = a.v - b.v;
    (dl * dl + du * du + dv * dv).sqrt()
}

/// Linearly interpolates between two `CIELUV` colors component by component.
///
/// `t = 0` returns `a` and `t = 1` returns `b`; interpolating in `CIELUV` keeps
/// same-lightness additive blends perceptually even, which is why glow-mixing
/// tooling stores keyframes here rather than in linear `sRGB`.
#[must_use]
pub fn luv_lerp(a: &Luv, b: &Luv, t: f32) -> Luv {
    Luv::new(
        a.l + (b.l - a.l) * t,
        a.u + (b.u - a.u) * t,
        a.v + (b.v - a.v) * t,
    )
}

/// Packs a [`Luv`] triple into its `std430` byte layout: three little-endian
/// `f32` scalars (`L*`, `u*`, `v*`) followed by a zeroed one-scalar padding tail
/// filling the aligned `vec4` slot.
#[must_use]
pub fn to_std430(c: &Luv) -> [u8; CIELUV_STD430_SIZE] {
    let fields = [c.l, c.u, c.v];
    let mut bytes = [0_u8; CIELUV_STD430_SIZE];
    for (slot, value) in bytes.chunks_exact_mut(4).zip(fields.iter()) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Total `std430` storage-buffer byte size for `count` [`Luv`] elements, reusing
/// the shared clamp-to-one-element rule so an empty pool still reserves one
/// aligned `vec4` slot.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(CIELUV_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for direct value assertions.
    const EPS: f32 = 1.0e-4;

    /// Looser tolerance for `f32` round-trip chains through the cube root.
    const ROUND_EPS: f32 = 1.0e-3;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    fn approx_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn approx_xyz(a: &Xyz, b: &Xyz, eps: f32) -> bool {
        approx_eps(a.x, b.x, eps) && approx_eps(a.y, b.y, eps) && approx_eps(a.z, b.z, eps)
    }

    #[test]
    fn cbrt_newton_matches_perfect_cubes() {
        assert!(approx(cbrt_newton(8.0), 2.0));
        assert!(approx(cbrt_newton(27.0), 3.0));
        assert!(approx(cbrt_newton(64.0), 4.0));
        assert!(approx(cbrt_newton(125.0), 5.0));
        assert!(approx(cbrt_newton(1000.0), 10.0));
    }

    #[test]
    fn cbrt_newton_handles_fractions_and_large() {
        assert!(approx(cbrt_newton(0.125), 0.5));
        assert!(approx_eps(cbrt_newton(0.001), 0.1, ROUND_EPS));
        assert!(approx_eps(cbrt_newton(2.0), 1.259_921, ROUND_EPS));
        assert!(approx_eps(cbrt_newton(1_000_000.0), 100.0, 0.1));
    }

    #[test]
    fn cbrt_newton_zero_and_negative() {
        assert!(approx(cbrt_newton(0.0), 0.0));
        assert!(approx(cbrt_newton(-8.0), -2.0));
        assert!(approx(cbrt_newton(-27.0), -3.0));
    }

    #[test]
    fn cbrt_newton_cube_roundtrips() {
        for &v in &[0.3_f32, 1.7, 5.5, 42.0, 900.0] {
            let r = cbrt_newton(v);
            assert!(approx_eps(r * r * r, v, v * ROUND_EPS + ROUND_EPS));
        }
    }

    #[test]
    fn luv_f_is_continuous_at_the_knot() {
        let below = luv_f(DELTA_CUBED - 1.0e-7);
        let at = luv_f(DELTA_CUBED);
        let above = luv_f(DELTA_CUBED + 1.0e-7);
        assert!(approx_eps(below, DELTA, ROUND_EPS));
        assert!(approx_eps(at, DELTA, ROUND_EPS));
        assert!(approx_eps(above, DELTA, ROUND_EPS));
    }

    #[test]
    fn luv_f_knot_value_equals_delta() {
        assert!(approx(luv_f(DELTA_CUBED), DELTA));
    }

    #[test]
    fn luv_f_inv_inverts_luv_f() {
        for &t in &[0.0_f32, 0.005, 0.02, DELTA_CUBED, 0.1, 0.5, 1.0, 2.0] {
            let round = luv_f_inv(luv_f(t));
            assert!(approx_eps(round, t, ROUND_EPS), "t = {t}");
        }
    }

    #[test]
    fn luv_f_inv_knot_value_equals_delta_cubed() {
        assert!(approx(luv_f_inv(DELTA), DELTA_CUBED));
    }

    #[test]
    fn white_maps_to_lightness_hundred_and_neutral_chroma() {
        let luv = xyz_to_luv(&d65_white());
        assert!(approx_eps(luv.l, 100.0, ROUND_EPS));
        assert!(approx_eps(luv.u, 0.0, ROUND_EPS));
        assert!(approx_eps(luv.v, 0.0, ROUND_EPS));
    }

    #[test]
    fn black_maps_to_zero_lightness_and_chroma() {
        let luv = xyz_to_luv(&Xyz::new(0.0, 0.0, 0.0));
        assert!(approx(luv.l, 0.0));
        assert!(approx(luv.u, 0.0));
        assert!(approx(luv.v, 0.0));
    }

    #[test]
    fn white_point_uv_prime_matches_reference() {
        let uv = uv_prime(&d65_white());
        let reference = reference_white_uv();
        assert!(approx(uv.u_prime, reference.u_prime));
        assert!(approx(uv.v_prime, reference.v_prime));
    }

    #[test]
    fn reference_white_uv_is_in_expected_range() {
        let r = reference_white_uv();
        // Derived from this module's D65 tristimulus constants (Xn, Yn, Zn):
        // u'n = 4Xn / (Xn + 15Yn + 3Zn), v'n = 9Yn / (Xn + 15Yn + 3Zn).
        assert!(approx_eps(r.u_prime, 0.204_219, 1.0e-4));
        assert!(approx_eps(r.v_prime, 0.483_429, 1.0e-4));
    }

    #[test]
    fn uv_prime_degenerate_denominator_is_zero() {
        let uv = uv_prime(&Xyz::new(0.0, 0.0, 0.0));
        assert!(approx(uv.u_prime, 0.0));
        assert!(approx(uv.v_prime, 0.0));
    }

    #[test]
    fn xyz_to_luv_to_xyz_roundtrips() {
        let samples = [
            Xyz::new(0.4, 0.35, 0.30),
            Xyz::new(0.1, 0.2, 0.5),
            Xyz::new(0.5, 0.5, 0.1),
            Xyz::new(0.2, 0.2, 0.2),
            Xyz::new(0.8, 0.75, 0.9),
            d65_white(),
        ];
        for c in &samples {
            let back = luv_to_xyz(&xyz_to_luv(c));
            assert!(approx_xyz(&back, c, ROUND_EPS), "{c:?} != {back:?}");
        }
    }

    #[test]
    fn white_roundtrips_through_luv() {
        let white = d65_white();
        let back = luv_to_xyz(&xyz_to_luv(&white));
        assert!(
            approx_xyz(&back, &white, ROUND_EPS),
            "{white:?} != {back:?}"
        );
    }

    #[test]
    fn near_black_linear_toe_roundtrips() {
        // Small Y stays on the linear toe of the lightness curve.
        let c = Xyz::new(0.004, 0.005, 0.006);
        let back = luv_to_xyz(&xyz_to_luv(&c));
        assert!(approx_xyz(&back, &c, ROUND_EPS), "{c:?} != {back:?}");
    }

    #[test]
    fn luv_to_xyz_of_black_is_origin() {
        let back = luv_to_xyz(&Luv::new(0.0, 0.0, 0.0));
        assert!(approx_xyz(&back, &Xyz::new(0.0, 0.0, 0.0), EPS));
    }

    #[test]
    fn lightness_is_monotonic_in_luminance() {
        let mut prev = f32::NEG_INFINITY;
        let mut y = 0.0_f32;
        while y <= 1.0 {
            let luv = xyz_to_luv(&Xyz::new(XN * y, YN * y, ZN * y));
            assert!(luv.l >= prev - EPS, "L* dropped at Y = {y}");
            prev = luv.l;
            y += 0.05;
        }
    }

    #[test]
    fn delta_e_uv_is_zero_for_identical_colors() {
        let a = Luv::new(42.0, -7.0, 13.0);
        assert!(approx(delta_e_uv(&a, &a), 0.0));
    }

    #[test]
    fn delta_e_uv_is_symmetric() {
        let a = Luv::new(30.0, 10.0, -5.0);
        let b = Luv::new(45.0, -8.0, 12.0);
        assert!(approx(delta_e_uv(&a, &b), delta_e_uv(&b, &a)));
    }

    #[test]
    fn delta_e_uv_known_distance() {
        let a = Luv::new(50.0, 2.0, 3.0);
        let b = Luv::new(60.0, 5.0, 7.0);
        // sqrt(10^2 + 3^2 + 4^2) = sqrt(125).
        assert!(approx_eps(delta_e_uv(&a, &b), 125.0_f32.sqrt(), EPS));
    }

    #[test]
    fn delta_e_uv_single_axis() {
        let a = Luv::new(50.0, 0.0, 0.0);
        let b = Luv::new(50.0, 10.0, 0.0);
        assert!(approx(delta_e_uv(&a, &b), 10.0));
    }

    #[test]
    fn delta_e_uv_is_never_negative() {
        let a = Luv::new(12.0, -30.0, 5.0);
        let b = Luv::new(88.0, 40.0, -60.0);
        assert!(delta_e_uv(&a, &b) >= 0.0);
    }

    #[test]
    fn chroma_of_neutral_is_zero() {
        assert!(approx(chroma(&Luv::new(50.0, 0.0, 0.0)), 0.0));
        assert!(approx(chroma(&Luv::new(0.0, 0.0, 0.0)), 0.0));
    }

    #[test]
    fn chroma_known_value() {
        // sqrt(3^2 + 4^2) = 5.
        assert!(approx(chroma(&Luv::new(50.0, 3.0, 4.0)), 5.0));
    }

    #[test]
    fn saturation_is_chroma_over_lightness() {
        let c = Luv::new(40.0, 6.0, 8.0);
        // chroma = 10, L* = 40 -> 0.25.
        assert!(approx(saturation(&c), 0.25));
        assert!(approx(saturation(&c), chroma(&c) / c.l));
    }

    #[test]
    fn saturation_of_black_is_zero() {
        assert!(approx(saturation(&Luv::new(0.0, 5.0, 5.0)), 0.0));
    }

    #[test]
    fn saturation_of_neutral_is_zero() {
        assert!(approx(saturation(&Luv::new(60.0, 0.0, 0.0)), 0.0));
    }

    #[test]
    fn luv_lerp_returns_endpoints() {
        let a = Luv::new(10.0, -20.0, 30.0);
        let b = Luv::new(90.0, 40.0, -50.0);
        let start = luv_lerp(&a, &b, 0.0);
        let end = luv_lerp(&a, &b, 1.0);
        assert!(approx(start.l, a.l) && approx(start.u, a.u) && approx(start.v, a.v));
        assert!(approx(end.l, b.l) && approx(end.u, b.u) && approx(end.v, b.v));
    }

    #[test]
    fn luv_lerp_midpoint_is_the_average() {
        let a = Luv::new(20.0, 10.0, -10.0);
        let b = Luv::new(60.0, -30.0, 50.0);
        let mid = luv_lerp(&a, &b, 0.5);
        assert!(approx(mid.l, 40.0));
        assert!(approx(mid.u, -10.0));
        assert!(approx(mid.v, 20.0));
    }

    #[test]
    fn constructors_store_fields() {
        let luv = Luv::new(1.0, 2.0, 3.0);
        assert!(approx(luv.l, 1.0) && approx(luv.u, 2.0) && approx(luv.v, 3.0));
        let xyz = Xyz::new(4.0, 5.0, 6.0);
        assert!(approx(xyz.x, 4.0) && approx(xyz.y, 5.0) && approx(xyz.z, 6.0));
        let uv = UvPrime::new(0.2, 0.5);
        assert!(approx(uv.u_prime, 0.2) && approx(uv.v_prime, 0.5));
    }

    #[test]
    fn u_star_sign_tracks_chromaticity_shift() {
        // A color redder than the white point (larger u') has positive u*.
        let reddish = xyz_to_luv(&Xyz::new(0.5, 0.3, 0.2));
        assert!(reddish.u > 0.0);
    }

    #[test]
    fn std430_size_is_sixteen() {
        assert_eq!(CIELUV_STD430_SIZE, 16);
    }

    #[test]
    fn std430_size_is_multiple_of_sixteen() {
        assert_eq!(CIELUV_STD430_SIZE % 16, 0);
    }

    #[test]
    fn std430_byte_layout_roundtrips() {
        let c = Luv::new(53.5, -12.25, 41.75);
        let bytes = to_std430(&c);
        let fields = [c.l, c.u, c.v];
        for (slot, value) in bytes.chunks_exact(4).zip(fields.iter()) {
            let mut word = [0_u8; 4];
            word.copy_from_slice(slot);
            assert!(approx(f32::from_le_bytes(word), *value));
        }
        let mut tail = [0_u8; 4];
        tail.copy_from_slice(&bytes[12..16]);
        assert_eq!(u32::from_le_bytes(tail), 0);
    }

    #[test]
    fn gpu_storage_bytes_empty_reserves_one_slot() {
        assert_eq!(gpu_storage_bytes(0), CIELUV_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(gpu_storage_bytes(4), CIELUV_STD430_SIZE * 4);
        assert_eq!(gpu_storage_bytes(100), CIELUV_STD430_SIZE * 100);
    }

    #[test]
    fn gpu_storage_bytes_is_always_multiple_of_sixteen() {
        for count in 0..40 {
            assert_eq!(gpu_storage_bytes(count) % 16, 0);
        }
    }
}
