//! `CIE` 1976 `L*a*b*` (`CIELAB`) contract for the particle color pipeline:
//! the perceptually-uniform bridge between device-independent `CIE` `XYZ` and
//! the color-difference metrics emitters use to compare, sort, and blend tints.
//!
//! Where [`crate::particle::cie_xyz`] owns the linear-`sRGB` primaries matrix,
//! the `XYZ` <-> `xyY` split, and `Bradford` chromatic adaptation, this module
//! is deliberately narrow: it converts `CIE` `XYZ` to and from `CIELAB`, and it
//! computes the `CIE76` and `CIE94` color differences (`ΔE`) that gradient and
//! keyframe tooling reads. It never re-derives `sRGB`, `xyY`, or `Bradford`; a
//! caller adapts to the `D65` reference white with that module first, then hands
//! the adapted `XYZ` triple here.
//!
//! The perceptual lightness transfer is the standard `CIELAB` cube-root curve
//! with a linear toe near black. It is expressed through a single nonlinearity
//! [`lab_f`] and its exact inverse, both pinned to the `δ = 6/29` knot so the
//! forward and inverse legs meet continuously.
//!
//! Determinism rules (design §29) hold: the arithmetic is pure `+ - * /` plus
//! `f32::sqrt`, `f32::floor`, `f32::abs`, `f32::min`, `f32::max`, and
//! `f32::clamp`. There are no transcendental calls (`sin` / `cos` / `exp` /
//! `ln` / `pow` / `cbrt`); the cube root is a self-contained Newton iteration in
//! [`cbrt_newton`], and every threshold is a named constant so the result never
//! depends on a runtime-built basis. The `std430` packing mirrors an aligned
//! `vec4` slot so the future `GPU` color kernel binds against a stable `ABI` and
//! reproduces this `CPU` reference bit for bit.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Byte size of the `std430` packing of a [`Lab`] triple: three `f32` scalars
/// promoted to one aligned `vec4` slot (16 bytes), leaving a one-scalar padding
/// tail so the block honors the `std430` 16-byte base alignment.
pub const CIELAB_STD430_SIZE: usize = VEC4_STRIDE;

/// Absolute tolerance for the guarded `f32` comparisons; direct `==` / `!=` on
/// floating point is intentionally avoided in both production and test code.
const CMP_EPS: f32 = 1.0e-6;

/// The `CIELAB` knot `δ = 6/29`; the lightness curve switches from its linear
/// toe to the cube-root leg here. Written as a division of named integers so it
/// is never mistaken for a bare approximation of a mathematical constant.
const DELTA: f32 = 6.0 / 29.0;

/// The knot in domain space, `δ³`; the forward nonlinearity uses the cube root
/// above this input and the linear toe at or below it.
const DELTA_CUBED: f32 = DELTA * DELTA * DELTA;

/// The linear-toe slope denominator `3δ²`. The forward toe divides by it and
/// the inverse toe multiplies by it, keeping the two legs exact inverses.
const THREE_DELTA_SQ: f32 = 3.0 * DELTA * DELTA;

/// The linear-toe offset `4/29`, the value both legs take at `t = 0`.
const TOE_OFFSET: f32 = 4.0 / 29.0;

/// Number of Newton refinement steps [`cbrt_newton`] runs after range
/// reduction; the reduced input lies in `[1, 8)` where the quadratically
/// convergent iteration reaches `f32` precision well within this budget.
const CBRT_ITERATIONS: usize = 16;

/// Upper bound of the [`cbrt_newton`] range-reduction window: inputs above it
/// are divided by eight (halving the cube root) until they fall inside `[1, 8)`.
const CBRT_WINDOW_HI: f32 = 8.0;

/// Lower bound of the [`cbrt_newton`] range-reduction window: inputs below it
/// are multiplied by eight (doubling the cube root) until they reach `[1, 8)`.
const CBRT_WINDOW_LO: f32 = 1.0;

/// `x` chromaticity component of the `CIE` `D65` reference white as an `XYZ`
/// tristimulus value normalized to `Y = 1`.
const XN: f32 = 0.950_489;

/// `Y` (luminance) component of the `CIE` `D65` reference white, normalized to
/// unity by construction.
const YN: f32 = 1.0;

/// `Z` component of the `CIE` `D65` reference white used to normalize the blue
/// axis of the `CIELAB` transform.
const ZN: f32 = 0.888_840;

/// `CIE94` weighting `K1` for the graphic-arts application (chroma term).
const K1_GRAPHIC: f32 = 0.045;

/// `CIE94` weighting `K2` for the graphic-arts application (hue term).
const K2_GRAPHIC: f32 = 0.015;

/// `CIE94` lightness weight `kL` for the graphic-arts application.
const KL_GRAPHIC: f32 = 1.0;

/// `CIE94` weighting `K1` for the textiles application (chroma term).
const K1_TEXTILE: f32 = 0.048;

/// `CIE94` weighting `K2` for the textiles application (hue term).
const K2_TEXTILE: f32 = 0.014;

/// `CIE94` lightness weight `kL` for the textiles application.
const KL_TEXTILE: f32 = 2.0;

/// A `CIE` 1976 `L*a*b*` color.
///
/// `l` is perceptual lightness in `[0, 100]` (`0` black, `100` reference
/// white); `a` runs green-to-red and `b` runs blue-to-yellow, both unbounded in
/// principle but small for in-gamut colors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lab {
    /// Perceptual lightness `L*`.
    pub l: f32,
    /// Green-red opponent axis `a*`.
    pub a: f32,
    /// Blue-yellow opponent axis `b*`.
    pub b: f32,
}

impl Lab {
    /// Builds a [`Lab`] color from its lightness and two opponent axes.
    #[must_use]
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }
}

/// A `CIE` `XYZ` tristimulus triple, owned by this module so the `CIELAB`
/// contract stays self-contained and does not depend on
/// [`crate::particle::cie_xyz`].
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

/// Returns the real cube root of `x` using a self-contained Newton iteration,
/// avoiding the disallowed [`f32::cbrt`] intrinsic.
///
/// Negative inputs reflect through the origin (`cbrt(-x) = -cbrt(x)`) and a zero
/// input returns zero, guarding the division in the iteration. The magnitude is
/// range-reduced into `[1, 8)` by factors of eight — each division by eight
/// halves the cube root and each multiplication doubles it — so a fixed
/// [`CBRT_ITERATIONS`] budget of Newton steps `y ← y − (y³ − v) / (3y²)`
/// reaches `f32` precision regardless of the input's magnitude.
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

/// The `CIELAB` forward nonlinearity `f(t)`.
///
/// Above the knot `δ³` it is the cube root; at or below it the linear toe
/// `t / (3δ²) + 4/29` takes over. The two legs meet at `f(δ³) = δ`, so the
/// function is continuous across the knot.
#[must_use]
pub fn lab_f(t: f32) -> f32 {
    if t > DELTA_CUBED {
        cbrt_newton(t)
    } else {
        t / THREE_DELTA_SQ + TOE_OFFSET
    }
}

/// The exact inverse of [`lab_f`].
///
/// Above the knot `δ` it cubes its argument with an integer product `t·t·t`
/// (avoiding the disallowed power intrinsics); at or below it the inverse linear
/// toe `3δ² · (t − 4/29)` takes over. The two legs meet at `δ`, matching the
/// forward knot exactly.
#[must_use]
pub fn lab_f_inv(t: f32) -> f32 {
    if t > DELTA {
        t * t * t
    } else {
        THREE_DELTA_SQ * (t - TOE_OFFSET)
    }
}

/// Converts a `CIE` `XYZ` triple (measured relative to the `D65` white) into
/// `CIE` 1976 `L*a*b*`.
///
/// Each channel is normalized by the matching `D65` white component, passed
/// through [`lab_f`], and recombined: `L* = 116 fy − 16`,
/// `a* = 500 (fx − fy)`, `b* = 200 (fy − fz)`.
#[must_use]
pub fn xyz_to_lab(c: &Xyz) -> Lab {
    let fx = lab_f(c.x / XN);
    let fy = lab_f(c.y / YN);
    let fz = lab_f(c.z / ZN);
    Lab::new(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// Converts a `CIE` 1976 `L*a*b*` color back into a `CIE` `XYZ` triple relative
/// to the `D65` white, inverting [`xyz_to_lab`].
///
/// It recovers `fy = (L* + 16) / 116`, then `fx` and `fz` from the opponent
/// axes, maps each through [`lab_f_inv`], and rescales by the `D65` white.
#[must_use]
pub fn lab_to_xyz(c: &Lab) -> Xyz {
    let fy = (c.l + 16.0) / 116.0;
    let fx = fy + c.a / 500.0;
    let fz = fy - c.b / 200.0;
    Xyz::new(XN * lab_f_inv(fx), YN * lab_f_inv(fy), ZN * lab_f_inv(fz))
}

/// The `CIE76` color difference `ΔE*ab`: the plain Euclidean distance between
/// two `CIELAB` colors. It is symmetric and zero exactly when the colors match.
#[must_use]
pub fn delta_e_76(a: &Lab, b: &Lab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

/// The `CIE94` color difference `ΔE*94`, weighting the chroma and hue terms by
/// the reference color's chroma.
///
/// `reference` is the standard and `sample` the trial color; the metric is
/// intentionally asymmetric because the weights `S_C = 1 + K1 C_ref` and
/// `S_H = 1 + K2 C_ref` scale with the reference chroma. Selecting `graphic`
/// uses the graphic-arts constants (`kL = 1`, `K1 = 0.045`, `K2 = 0.015`);
/// otherwise the textiles constants (`kL = 2`, `K1 = 0.048`, `K2 = 0.014`)
/// apply. The hue term is recovered as `ΔH² = Δa² + Δb² − ΔC²`, clamped to zero
/// before its square root to absorb rounding.
#[must_use]
pub fn delta_e_94(reference: &Lab, sample: &Lab, graphic: bool) -> f32 {
    let (kl, k1, k2) = if graphic {
        (KL_GRAPHIC, K1_GRAPHIC, K2_GRAPHIC)
    } else {
        (KL_TEXTILE, K1_TEXTILE, K2_TEXTILE)
    };

    let dl = reference.l - sample.l;
    let da = reference.a - sample.a;
    let db = reference.b - sample.b;

    let c_ref = (reference.a * reference.a + reference.b * reference.b).sqrt();
    let c_sample = (sample.a * sample.a + sample.b * sample.b).sqrt();
    let dc = c_ref - c_sample;

    let dh_sq = (da * da + db * db - dc * dc).max(0.0);
    let dh = dh_sq.sqrt();

    let sl = 1.0;
    let sc = 1.0 + k1 * c_ref;
    let sh = 1.0 + k2 * c_ref;

    let term_l = dl / (kl * sl);
    let term_c = dc / sc;
    let term_h = dh / sh;

    (term_l * term_l + term_c * term_c + term_h * term_h).sqrt()
}

/// Linearly interpolates between two `CIELAB` colors component by component.
///
/// `t = 0` returns `a` and `t = 1` returns `b`; interpolating in `CIELAB` keeps
/// the blend perceptually even, which is why gradient tooling stores keyframes
/// here rather than in linear `sRGB`.
#[must_use]
pub fn lab_lerp(a: &Lab, b: &Lab, t: f32) -> Lab {
    Lab::new(
        a.l + (b.l - a.l) * t,
        a.a + (b.a - a.a) * t,
        a.b + (b.b - a.b) * t,
    )
}

/// Packs a [`Lab`] triple into its `std430` byte layout: three little-endian
/// `f32` scalars (`L*`, `a*`, `b*`) followed by a zeroed one-scalar padding tail
/// filling the aligned `vec4` slot.
#[must_use]
pub fn to_std430(c: &Lab) -> [u8; CIELAB_STD430_SIZE] {
    let fields = [c.l, c.a, c.b];
    let mut bytes = [0_u8; CIELAB_STD430_SIZE];
    for (slot, value) in bytes.chunks_exact_mut(4).zip(fields.iter()) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Total `std430` storage-buffer byte size for `count` [`Lab`] elements,
/// reusing the shared clamp-to-one-element rule so an empty pool still reserves
/// one aligned `vec4` slot.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(CIELAB_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx_eps(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_xyz(a: &Xyz, b: &Xyz) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    fn approx_lab(a: &Lab, b: &Lab) -> bool {
        approx(a.l, b.l) && approx(a.a, b.a) && approx(a.b, b.b)
    }

    fn d65_white() -> Xyz {
        Xyz::new(XN, YN, ZN)
    }

    #[test]
    fn cbrt_newton_matches_known_perfect_cubes() {
        const CUBE_EPS: f32 = 1.0e-4;
        assert!(approx_eps(cbrt_newton(8.0), 2.0, CUBE_EPS));
        assert!(approx_eps(cbrt_newton(27.0), 3.0, CUBE_EPS));
        assert!(approx_eps(cbrt_newton(64.0), 4.0, CUBE_EPS));
        assert!(approx_eps(cbrt_newton(125.0), 5.0, CUBE_EPS));
        assert!(approx_eps(cbrt_newton(1000.0), 10.0, CUBE_EPS));
    }

    #[test]
    fn cbrt_newton_matches_fractional_cubes() {
        const CUBE_EPS: f32 = 1.0e-5;
        assert!(approx_eps(cbrt_newton(0.125), 0.5, CUBE_EPS));
        assert!(approx_eps(cbrt_newton(0.001), 0.1, CUBE_EPS));
        assert!(approx_eps(cbrt_newton(1.0), 1.0, CUBE_EPS));
    }

    #[test]
    fn cbrt_newton_cubes_back_to_input() {
        const ROUND_EPS: f32 = 1.0e-4;
        for &x in &[0.02_f32, 0.2, 0.75, 1.3, 2.5, 7.9] {
            let r = cbrt_newton(x);
            assert!(
                approx_eps(r * r * r, x, ROUND_EPS),
                "cbrt({x})^3 = {}",
                r * r * r
            );
        }
    }

    #[test]
    fn cbrt_newton_handles_zero() {
        assert!(approx(cbrt_newton(0.0), 0.0));
    }

    #[test]
    fn cbrt_newton_handles_negative_by_reflection() {
        const CUBE_EPS: f32 = 1.0e-4;
        assert!(approx_eps(cbrt_newton(-8.0), -2.0, CUBE_EPS));
        assert!(approx_eps(cbrt_newton(-27.0), -3.0, CUBE_EPS));
    }

    #[test]
    fn lab_f_is_continuous_at_the_knot() {
        // Both legs meet at f(δ³) = δ.
        let below = lab_f(DELTA_CUBED - CMP_EPS);
        let above = lab_f(DELTA_CUBED + CMP_EPS);
        assert!(approx_eps(below, above, 1.0e-3));
        assert!(approx_eps(lab_f(DELTA_CUBED), DELTA, 1.0e-3));
    }

    #[test]
    fn lab_f_inv_is_continuous_at_the_knot() {
        let below = lab_f_inv(DELTA - CMP_EPS);
        let above = lab_f_inv(DELTA + CMP_EPS);
        assert!(approx_eps(below, above, 1.0e-3));
        assert!(approx_eps(lab_f_inv(DELTA), DELTA_CUBED, 1.0e-4));
    }

    #[test]
    fn lab_f_uses_linear_toe_below_the_knot() {
        // A tiny input stays on the linear leg, never touching the cube root.
        let t = DELTA_CUBED * 0.5;
        assert!(approx(lab_f(t), t / THREE_DELTA_SQ + TOE_OFFSET));
    }

    #[test]
    fn lab_f_and_inverse_roundtrip() {
        const ROUND_EPS: f32 = 1.0e-4;
        for &t in &[0.0005_f32, 0.005, 0.02, 0.1, 0.4, 0.9, 1.5] {
            let back = lab_f_inv(lab_f(t));
            assert!(approx_eps(back, t, ROUND_EPS), "f_inv(f({t})) = {back}");
        }
    }

    #[test]
    fn d65_white_maps_to_lightness_one_hundred() {
        let lab = xyz_to_lab(&d65_white());
        assert!(approx_eps(lab.l, 100.0, 1.0e-3), "L = {}", lab.l);
        assert!(approx_eps(lab.a, 0.0, 1.0e-3), "a = {}", lab.a);
        assert!(approx_eps(lab.b, 0.0, 1.0e-3), "b = {}", lab.b);
    }

    #[test]
    fn black_maps_to_zero_lightness() {
        let lab = xyz_to_lab(&Xyz::new(0.0, 0.0, 0.0));
        assert!(approx(lab.l, 0.0));
        assert!(approx(lab.a, 0.0));
        assert!(approx(lab.b, 0.0));
    }

    #[test]
    fn neutral_gray_has_zero_chroma() {
        // A neutral scaled white keeps fx = fy = fz, so both opponent axes
        // vanish while lightness sits between black and white.
        let k = 0.5_f32;
        let lab = xyz_to_lab(&Xyz::new(XN * k, YN * k, ZN * k));
        assert!(approx_eps(lab.a, 0.0, 1.0e-3));
        assert!(approx_eps(lab.b, 0.0, 1.0e-3));
        assert!(lab.l > 0.0 && lab.l < 100.0);
    }

    #[test]
    fn xyz_to_lab_to_xyz_roundtrip() {
        const ROUND_EPS: f32 = 1.0e-3;
        let samples = [
            Xyz::new(0.5, 0.4, 0.3),
            Xyz::new(0.2, 0.5, 0.9),
            Xyz::new(0.95, 1.0, 0.88),
            Xyz::new(0.01, 0.02, 0.03),
        ];
        for c in samples {
            let back = lab_to_xyz(&xyz_to_lab(&c));
            assert!(
                approx_eps(back.x, c.x, ROUND_EPS)
                    && approx_eps(back.y, c.y, ROUND_EPS)
                    && approx_eps(back.z, c.z, ROUND_EPS),
                "{c:?} != {back:?}"
            );
        }
    }

    #[test]
    fn lab_to_xyz_to_lab_roundtrip() {
        const ROUND_EPS: f32 = 1.0e-3;
        let samples = [
            Lab::new(50.0, 20.0, -30.0),
            Lab::new(75.0, -40.0, 60.0),
            Lab::new(10.0, 5.0, 5.0),
            Lab::new(100.0, 0.0, 0.0),
        ];
        for c in samples {
            let back = xyz_to_lab(&lab_to_xyz(&c));
            assert!(
                approx_eps(back.l, c.l, ROUND_EPS)
                    && approx_eps(back.a, c.a, ROUND_EPS)
                    && approx_eps(back.b, c.b, ROUND_EPS),
                "{c:?} != {back:?}"
            );
        }
    }

    #[test]
    fn white_roundtrips_through_lab() {
        let white = d65_white();
        let back = lab_to_xyz(&xyz_to_lab(&white));
        assert!(approx_xyz(&back, &white), "{white:?} != {back:?}");
    }

    #[test]
    fn delta_e_76_known_distance() {
        let a = Lab::new(50.0, 2.0, 3.0);
        let b = Lab::new(60.0, 5.0, 7.0);
        // sqrt(10^2 + 3^2 + 4^2) = sqrt(125).
        assert!(approx_eps(delta_e_76(&a, &b), 125.0_f32.sqrt(), 1.0e-4));
    }

    #[test]
    fn delta_e_76_simple_single_axis() {
        let a = Lab::new(50.0, 0.0, 0.0);
        let b = Lab::new(50.0, 10.0, 0.0);
        assert!(approx_eps(delta_e_76(&a, &b), 10.0, 1.0e-5));
    }

    #[test]
    fn delta_e_76_is_zero_for_identical_colors() {
        let a = Lab::new(42.0, -7.0, 13.0);
        assert!(approx(delta_e_76(&a, &a), 0.0));
    }

    #[test]
    fn delta_e_76_is_symmetric() {
        let a = Lab::new(30.0, 10.0, -5.0);
        let b = Lab::new(45.0, -8.0, 12.0);
        assert!(approx(delta_e_76(&a, &b), delta_e_76(&b, &a)));
    }

    #[test]
    fn delta_e_94_is_zero_for_identical_colors() {
        let a = Lab::new(60.0, 12.0, -20.0);
        assert!(approx(delta_e_94(&a, &a, true), 0.0));
        assert!(approx(delta_e_94(&a, &a, false), 0.0));
    }

    #[test]
    fn delta_e_94_zero_reference_chroma_matches_euclid() {
        // With a neutral reference the chroma/hue weights are unity, so CIE94
        // collapses to the CIE76 distance for the graphic-arts constants.
        let reference = Lab::new(50.0, 0.0, 0.0);
        let sample = Lab::new(50.0, 10.0, 0.0);
        assert!(approx_eps(
            delta_e_94(&reference, &sample, true),
            10.0,
            1.0e-4
        ));
    }

    #[test]
    fn delta_e_94_weights_reference_chroma() {
        // Swapping standard and trial changes the chroma weight, so CIE94 is
        // asymmetric: sqrt((10 / (1 + 0.045*10))^2) = 10 / 1.45.
        let reference = Lab::new(50.0, 10.0, 0.0);
        let sample = Lab::new(50.0, 0.0, 0.0);
        let expected = 10.0_f32 / 1.45_f32;
        assert!(approx_eps(
            delta_e_94(&reference, &sample, true),
            expected,
            1.0e-4
        ));
    }

    #[test]
    fn delta_e_94_graphic_and_textile_differ() {
        let reference = Lab::new(50.0, 20.0, 10.0);
        let sample = Lab::new(55.0, 30.0, 18.0);
        let graphic = delta_e_94(&reference, &sample, true);
        let textile = delta_e_94(&reference, &sample, false);
        assert!((graphic - textile).abs() > CMP_EPS);
    }

    #[test]
    fn delta_e_94_never_exceeds_delta_e_76() {
        // The CIE94 weights only ever divide the terms down, so it is never
        // larger than the plain Euclidean CIE76 distance.
        let reference = Lab::new(40.0, 25.0, -15.0);
        let sample = Lab::new(48.0, 12.0, 6.0);
        assert!(delta_e_94(&reference, &sample, true) <= delta_e_76(&reference, &sample) + CMP_EPS);
    }

    #[test]
    fn lab_lerp_returns_endpoints() {
        let a = Lab::new(10.0, -20.0, 30.0);
        let b = Lab::new(90.0, 40.0, -50.0);
        assert!(approx_lab(&lab_lerp(&a, &b, 0.0), &a));
        assert!(approx_lab(&lab_lerp(&a, &b, 1.0), &b));
    }

    #[test]
    fn lab_lerp_midpoint_is_the_average() {
        let a = Lab::new(20.0, 10.0, -10.0);
        let b = Lab::new(60.0, -30.0, 50.0);
        let mid = lab_lerp(&a, &b, 0.5);
        assert!(approx(mid.l, 40.0));
        assert!(approx(mid.a, -10.0));
        assert!(approx(mid.b, 20.0));
    }

    #[test]
    fn constructors_store_fields() {
        let lab = Lab::new(1.0, 2.0, 3.0);
        assert!(approx(lab.l, 1.0) && approx(lab.a, 2.0) && approx(lab.b, 3.0));
        let xyz = Xyz::new(4.0, 5.0, 6.0);
        assert!(approx(xyz.x, 4.0) && approx(xyz.y, 5.0) && approx(xyz.z, 6.0));
    }

    #[test]
    fn std430_size_is_sixteen() {
        assert_eq!(CIELAB_STD430_SIZE, 16);
    }

    #[test]
    fn std430_byte_layout_roundtrips() {
        let c = Lab::new(53.5, -12.25, 41.75);
        let bytes = to_std430(&c);
        let fields = [c.l, c.a, c.b];
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
        assert_eq!(gpu_storage_bytes(0), CIELAB_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(gpu_storage_bytes(4), CIELAB_STD430_SIZE * 4);
        assert_eq!(gpu_storage_bytes(100), CIELAB_STD430_SIZE * 100);
    }
}
