//! White-balance golden reference via von Kries chromatic adaptation.
//!
//! White balancing corrects a captured image so the illuminant it was shot
//! under appears neutral. The grade is driven by two artist controls:
//!
//! * **Temperature** (correlated colour temperature, in Kelvin) selects a point
//!   on the Planckian locus — the chromaticities an ideal black-body radiator
//!   emits as it heats up, running from warm amber at low Kelvin to cool blue at
//!   high Kelvin.
//! * **Tint** nudges the white point green ↔ magenta, perpendicular to the
//!   locus, correcting fluorescent / sky casts the locus alone cannot express.
//!
//! The pipeline is the classical, transcendental-free one:
//!
//! 1. [`cct_to_uv`] turns the temperature into a CIE 1960 `(u, v)` chromaticity
//!    using **Krystek's** rational approximation of the Planckian locus (valid
//!    ~1000–15000 K).
//! 2. The tint offsets `(u, v)` along the locus normal, then [`uv_to_xy`] /
//!    [`xy_to_xyz`] lift it to a tristimulus white point `XYZ` (luminance `Y =
//!    1`).
//! 3. [`von_kries_adapt`] adapts the scene from that source white to the fixed
//!    reference white in cone-response (**LMS**) space, using the **Bradford**
//!    cone matrix by default (**CAT02** is also provided).
//! 4. The grade is bracketed by the `sRGB`↔`XYZ` matrices so callers pass and
//!    receive linear `sRGB`.
//!
//! The reference white is [`REFERENCE_TEMPERATURE_K`] fed through the *same*
//! `cct_to_*` path, so the neutral setting
//! `white_balance(c, REFERENCE_TEMPERATURE_K, 0)` is the exact identity (the
//! adaptation maps a white to itself).
//!
//! # Conventions
//! * Pure deterministic `f32` maths — no RNG, IO, GPU, `unsafe` or allocation.
//! * No transcendental functions are needed: every stage is polynomial or
//!   rational, so no [`bevy_math::ops`] import is required here.
//! * Matrices are [`bevy_math::Mat3`] built row-major via [`mat3_from_rows`] so
//!   `Mat3::mul_vec3` evaluates `row · v`; colours are [`bevy_math::Vec3`].
//! * Defensive clamping throughout: temperatures are clamped to the Krystek
//!   validity range, denominators are guarded against zero, and outputs are
//!   clamped to `>= 0`, so no input can produce `NaN`.

use bevy_math::{Mat3, Vec3};

/// Lowest correlated colour temperature (Kelvin) the Krystek fit is valid for.
pub const MIN_CCT_K: f32 = 1000.0;

/// Highest correlated colour temperature (Kelvin) the Krystek fit is valid for.
pub const MAX_CCT_K: f32 = 15000.0;

/// Reference / neutral correlated colour temperature (Kelvin). The D65 standard
/// illuminant has a CCT of ~6504 K; feeding this through the same locus path as
/// the source white makes the neutral grade an exact identity.
pub const REFERENCE_TEMPERATURE_K: f32 = 6504.0;

/// Scale mapping the unitless `tint` control to an offset magnitude in CIE 1960
/// `(u, v)` space along the Planckian-locus normal. `tint = +1` / `-1` produces
/// a moderate green ↔ magenta shift.
const TINT_UV_SCALE: f32 = 0.05;

/// Smallest denominator magnitude tolerated before a reciprocal is formed.
const MIN_DENOM: f32 = 1.0e-6;

/// Build a [`Mat3`] from three row vectors such that `Mat3::mul_vec3(v)`
/// evaluates `result[i] = rows[i] · v` (row-major semantics).
///
/// `glam`'s [`Mat3`] is column-major, so the rows are transposed into columns.
#[must_use]
pub fn mat3_from_rows(r0: [f32; 3], r1: [f32; 3], r2: [f32; 3]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(r0[0], r1[0], r2[0]),
        Vec3::new(r0[1], r1[1], r2[1]),
        Vec3::new(r0[2], r1[2], r2[2]),
    )
}

// --- Colour-space matrices ------------------------------------------------

/// Linear `sRGB` (D65) → CIE `XYZ` (row-major source rows).
const SRGB_TO_XYZ_ROWS: [[f32; 3]; 3] = [
    [0.412_390_8, 0.357_584_34, 0.180_480_8],
    [0.212_639_, 0.715_168_7, 0.072_192_32],
    [0.019_330_82, 0.119_194_78, 0.950_532_14],
];

/// CIE `XYZ` → linear `sRGB` (D65) (row-major source rows).
const XYZ_TO_SRGB_ROWS: [[f32; 3]; 3] = [
    [3.240_97, -1.537_383_2, -0.498_610_76],
    [-0.969_243_64, 1.875_967_5, 0.041_555_06],
    [0.055_630_08, -0.203_976_96, 1.056_971_5],
];

/// Bradford cone-response matrix `XYZ` → `LMS` (row-major source rows).
const BRADFORD_ROWS: [[f32; 3]; 3] = [
    [0.895_1, 0.266_4, -0.161_4],
    [-0.750_2, 1.713_5, 0.036_7],
    [0.038_9, -0.068_5, 1.029_6],
];

/// Inverse Bradford cone-response matrix `LMS` → `XYZ` (row-major source rows).
const BRADFORD_INV_ROWS: [[f32; 3]; 3] = [
    [0.986_992_9, -0.147_054_3, 0.159_962_7],
    [0.432_305_3, 0.518_360_3, 0.049_291_2],
    [-0.008_528_7, 0.040_042_8, 0.968_486_7],
];

/// CAT02 cone-response matrix `XYZ` → `LMS` (row-major source rows).
const CAT02_ROWS: [[f32; 3]; 3] = [
    [0.732_8, 0.429_6, -0.162_4],
    [-0.703_6, 1.697_5, 0.006_1],
    [0.003_0, 0.013_6, 0.983_4],
];

/// Inverse CAT02 cone-response matrix `LMS` → `XYZ` (row-major source rows).
const CAT02_INV_ROWS: [[f32; 3]; 3] = [
    [1.096_124, -0.278_869, 0.182_745],
    [0.454_369, 0.473_533, 0.072_098],
    [-0.009_628, -0.005_698, 1.015_326],
];

/// Chromatic-adaptation cone basis selector for [`von_kries_adapt`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConeBasis {
    /// Bradford (sharpened) cones — the default used across the film pipeline.
    Bradford,
    /// CAT02 cones from the CIECAM02 appearance model.
    Cat02,
}

impl ConeBasis {
    /// Row-major forward `XYZ` → `LMS` matrix for this basis.
    #[must_use]
    fn forward(self) -> Mat3 {
        match self {
            ConeBasis::Bradford => mat3_from_rows(BRADFORD_ROWS[0], BRADFORD_ROWS[1], BRADFORD_ROWS[2]),
            ConeBasis::Cat02 => mat3_from_rows(CAT02_ROWS[0], CAT02_ROWS[1], CAT02_ROWS[2]),
        }
    }

    /// Row-major inverse `LMS` → `XYZ` matrix for this basis.
    #[must_use]
    fn inverse(self) -> Mat3 {
        match self {
            ConeBasis::Bradford => {
                mat3_from_rows(BRADFORD_INV_ROWS[0], BRADFORD_INV_ROWS[1], BRADFORD_INV_ROWS[2])
            }
            ConeBasis::Cat02 => mat3_from_rows(CAT02_INV_ROWS[0], CAT02_INV_ROWS[1], CAT02_INV_ROWS[2]),
        }
    }
}

/// Convert a linear `sRGB` (D65) colour to CIE `XYZ`.
#[must_use]
pub fn srgb_to_xyz(color: Vec3) -> Vec3 {
    mat3_from_rows(SRGB_TO_XYZ_ROWS[0], SRGB_TO_XYZ_ROWS[1], SRGB_TO_XYZ_ROWS[2]).mul_vec3(color)
}

/// Convert a CIE `XYZ` colour to linear `sRGB` (D65).
#[must_use]
pub fn xyz_to_srgb(xyz: Vec3) -> Vec3 {
    mat3_from_rows(XYZ_TO_SRGB_ROWS[0], XYZ_TO_SRGB_ROWS[1], XYZ_TO_SRGB_ROWS[2]).mul_vec3(xyz)
}

/// Convert CIE `XYZ` to cone-response `LMS` under the given [`ConeBasis`].
#[must_use]
pub fn xyz_to_lms(xyz: Vec3, basis: ConeBasis) -> Vec3 {
    basis.forward().mul_vec3(xyz)
}

/// Convert cone-response `LMS` back to CIE `XYZ` under the given [`ConeBasis`].
#[must_use]
pub fn lms_to_xyz(lms: Vec3, basis: ConeBasis) -> Vec3 {
    basis.inverse().mul_vec3(lms)
}

// --- Planckian locus ------------------------------------------------------

/// Krystek's rational approximation of the Planckian locus in CIE 1960 `(u, v)`
/// chromaticity for a correlated colour temperature in Kelvin.
///
/// Valid for ~1000–15000 K; the input is clamped to that range. The two
/// denominators are strictly positive across the range, but are still guarded.
#[must_use]
pub fn cct_to_uv(temperature_k: f32) -> (f32, f32) {
    let t = if temperature_k.is_finite() {
        temperature_k.clamp(MIN_CCT_K, MAX_CCT_K)
    } else {
        REFERENCE_TEMPERATURE_K
    };
    let t2 = t * t;

    let u_den = 1.0 + 8.424_203_5e-4 * t + 7.081_452e-7 * t2;
    let v_den = 1.0 - 2.897_418_2e-5 * t + 1.614_560_5e-7 * t2;

    let u_num = 0.860_117_76 + 1.541_182_5e-4 * t + 1.286_412_1e-7 * t2;
    let v_num = 0.317_398_73 + 4.228_062_4e-5 * t + 4.204_817e-8 * t2;

    let u = u_num / safe_denom(u_den);
    let v = v_num / safe_denom(v_den);
    (u, v)
}

/// Guard a denominator away from zero while preserving its sign.
#[must_use]
fn safe_denom(d: f32) -> f32 {
    if d.abs() < MIN_DENOM {
        if d < 0.0 { -MIN_DENOM } else { MIN_DENOM }
    } else {
        d
    }
}

/// Convert a CIE 1960 `(u, v)` chromaticity to CIE 1931 `(x, y)`.
#[must_use]
pub fn uv_to_xy(u: f32, v: f32) -> (f32, f32) {
    let den = safe_denom(2.0 * u - 8.0 * v + 4.0);
    let x = 3.0 * u / den;
    let y = 2.0 * v / den;
    (x, y)
}

/// Lift a CIE 1931 `(x, y)` chromaticity to a tristimulus `XYZ` with luminance
/// `Y = 1`. A degenerate `y` collapses to the equal-energy white.
#[must_use]
pub fn xy_to_xyz(x: f32, y: f32) -> Vec3 {
    if y.abs() < MIN_DENOM {
        return Vec3::new(1.0, 1.0, 1.0);
    }
    let big_x = x / y;
    let big_z = (1.0 - x - y) / y;
    Vec3::new(big_x, 1.0, big_z.max(0.0))
}

/// Compute the white-point `XYZ` for a temperature / tint pair.
///
/// The tint offsets the Planckian `(u, v)` along the locus normal (estimated by
/// a central finite difference of [`cct_to_uv`]), so the shift stays
/// perpendicular to the warm↔cool axis. `tint = 0` lands exactly on the locus.
#[must_use]
pub fn cct_to_white(temperature_k: f32, tint: f32) -> Vec3 {
    let (u, v) = cct_to_uv(temperature_k);

    // Locus tangent via central difference, then its unit normal.
    let t = if temperature_k.is_finite() {
        temperature_k.clamp(MIN_CCT_K, MAX_CCT_K)
    } else {
        REFERENCE_TEMPERATURE_K
    };
    let dt = 1.0_f32;
    let (u_hi, v_hi) = cct_to_uv((t + dt).min(MAX_CCT_K));
    let (u_lo, v_lo) = cct_to_uv((t - dt).max(MIN_CCT_K));
    let tu = u_hi - u_lo;
    let tv = v_hi - v_lo;
    let len = (tu * tu + tv * tv).sqrt();

    let tint_c = if tint.is_finite() { tint } else { 0.0 };
    let (u_t, v_t) = if len < MIN_DENOM {
        (u, v + tint_c * TINT_UV_SCALE)
    } else {
        // Normal = perpendicular to the (unit) tangent.
        let nu = -tv / len;
        let nv = tu / len;
        (u + tint_c * TINT_UV_SCALE * nu, v + tint_c * TINT_UV_SCALE * nv)
    };

    let (x, y) = uv_to_xy(u_t, v_t);
    xy_to_xyz(x, y)
}

// --- von Kries adaptation -------------------------------------------------

/// von Kries chromatic adaptation from `src_white` to `dst_white` in cone space.
///
/// Builds the diagonal cone-gain `dst_cone / src_cone` in the chosen
/// [`ConeBasis`] and sandwiches it between the forward / inverse cone matrices:
/// `M = cone⁻¹ · diag(dst / src) · cone`. When `src_white == dst_white` the
/// gains are all `1` and `M` is the identity, so the adapted colour equals the
/// input exactly. Each cone gain's denominator is guarded against zero and the
/// result is clamped to `>= 0`.
#[must_use]
pub fn von_kries_adapt(xyz: Vec3, src_white: Vec3, dst_white: Vec3, basis: ConeBasis) -> Vec3 {
    let src_cone = xyz_to_lms(src_white, basis);
    let dst_cone = xyz_to_lms(dst_white, basis);

    let gain = Vec3::new(
        dst_cone.x / safe_denom(src_cone.x),
        dst_cone.y / safe_denom(src_cone.y),
        dst_cone.z / safe_denom(src_cone.z),
    );

    let lms = xyz_to_lms(xyz, basis);
    let adapted = Vec3::new(lms.x * gain.x, lms.y * gain.y, lms.z * gain.z);
    let out = lms_to_xyz(adapted, basis);
    Vec3::new(out.x.max(0.0), out.y.max(0.0), out.z.max(0.0))
}

/// White-balance a linear `sRGB` colour for a temperature / tint pair, using the
/// Bradford cone basis.
///
/// The source white is the illuminant selected by `(temperature_k, tint)`; the
/// destination is the fixed reference white at [`REFERENCE_TEMPERATURE_K`] with
/// zero tint. The neutral setting is therefore the exact identity.
#[must_use]
pub fn white_balance(color: Vec3, temperature_k: f32, tint: f32) -> Vec3 {
    white_balance_with_basis(color, temperature_k, tint, ConeBasis::Bradford)
}

/// White-balance a linear `sRGB` colour using an explicit cone basis.
#[must_use]
pub fn white_balance_with_basis(
    color: Vec3,
    temperature_k: f32,
    tint: f32,
    basis: ConeBasis,
) -> Vec3 {
    let src_white = cct_to_white(temperature_k, tint);
    let dst_white = cct_to_white(REFERENCE_TEMPERATURE_K, 0.0);

    let xyz = srgb_to_xyz(color);
    let adapted = von_kries_adapt(xyz, src_white, dst_white, basis);
    let rgb = xyz_to_srgb(adapted);
    Vec3::new(rgb.x.max(0.0), rgb.y.max(0.0), rgb.z.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_tol(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "{a} != {b} (tol {tol})");
    }

    fn approx(a: f32, b: f32) {
        approx_tol(a, b, 1.0e-4);
    }

    fn approx3(a: Vec3, b: Vec3) {
        approx(a.x, b.x);
        approx(a.y, b.y);
        approx(a.z, b.z);
    }

    #[test]
    fn mat3_from_rows_is_row_major() {
        let m = mat3_from_rows([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let r = m.mul_vec3(v);
        // First column = (row0[0], row1[0], row2[0]).
        approx3(r, Vec3::new(1.0, 4.0, 7.0));
    }

    #[test]
    fn srgb_xyz_round_trip() {
        let c = Vec3::new(0.3, 0.6, 0.2);
        approx3(xyz_to_srgb(srgb_to_xyz(c)), c);
    }

    #[test]
    fn lms_round_trip_bradford() {
        let xyz = Vec3::new(0.4, 0.5, 0.6);
        approx3(lms_to_xyz(xyz_to_lms(xyz, ConeBasis::Bradford), ConeBasis::Bradford), xyz);
    }

    #[test]
    fn lms_round_trip_cat02() {
        let xyz = Vec3::new(0.4, 0.5, 0.6);
        approx3(lms_to_xyz(xyz_to_lms(xyz, ConeBasis::Cat02), ConeBasis::Cat02), xyz);
    }

    #[test]
    fn von_kries_same_white_is_identity() {
        let white = cct_to_white(REFERENCE_TEMPERATURE_K, 0.0);
        let c = Vec3::new(0.2, 0.7, 0.5);
        approx3(von_kries_adapt(c, white, white, ConeBasis::Bradford), c);
        approx3(von_kries_adapt(c, white, white, ConeBasis::Cat02), c);
    }

    #[test]
    fn white_balance_neutral_is_identity() {
        for &c in &[
            Vec3::new(0.1, 0.1, 0.1),
            Vec3::new(0.3, 0.6, 0.2),
            Vec3::new(0.9, 0.4, 0.7),
        ] {
            approx3(white_balance(c, REFERENCE_TEMPERATURE_K, 0.0), c);
        }
    }

    #[test]
    fn cct_to_uv_is_in_locus_range() {
        for &t in &[1500.0_f32, 3000.0, 5000.0, 6504.0, 10000.0] {
            let (u, v) = cct_to_uv(t);
            assert!(u.is_finite() && v.is_finite());
            assert!((0.0..1.0).contains(&u), "u out of range at {t}: {u}");
            assert!((0.0..1.0).contains(&v), "v out of range at {t}: {v}");
        }
    }

    #[test]
    fn cct_to_uv_is_monotonic_in_u() {
        // Along the locus the u chromaticity decreases as temperature rises.
        let mut prev = f32::INFINITY;
        for &t in &[1500.0_f32, 2500.0, 4000.0, 6500.0, 10000.0, 14000.0] {
            let (u, _) = cct_to_uv(t);
            assert!(u < prev, "u not decreasing at {t}: {u} >= {prev}");
            prev = u;
        }
    }

    #[test]
    fn cct_clamped_outside_range() {
        // Below / above the Krystek range the clamp keeps results finite.
        let (u0, v0) = cct_to_uv(10.0);
        let (umin, vmin) = cct_to_uv(MIN_CCT_K);
        approx(u0, umin);
        approx(v0, vmin);
        let (u1, _) = cct_to_uv(1.0e9);
        let (umax, _) = cct_to_uv(MAX_CCT_K);
        approx(u1, umax);
    }

    #[test]
    fn warm_temperature_pushes_toward_red() {
        // A warm (low-K) illuminant correction should boost blue / cut red to
        // neutralise the amber cast relative to the neutral grade.
        let grey = Vec3::new(0.5, 0.5, 0.5);
        let warm = white_balance(grey, 3200.0, 0.0);
        assert!(warm.is_finite());
        // Correcting a warm cast lifts the blue channel above red.
        assert!(warm.z > warm.x, "expected blue lift: {warm:?}");
    }

    #[test]
    fn cool_temperature_pushes_toward_blue() {
        let grey = Vec3::new(0.5, 0.5, 0.5);
        let cool = white_balance(grey, 10000.0, 0.0);
        // Correcting a cool cast lifts red above blue.
        assert!(cool.x > cool.z, "expected red lift: {cool:?}");
    }

    #[test]
    fn tint_shifts_green_magenta_axis() {
        let grey = Vec3::new(0.5, 0.5, 0.5);
        let pos = white_balance(grey, REFERENCE_TEMPERATURE_K, 1.0);
        let neg = white_balance(grey, REFERENCE_TEMPERATURE_K, -1.0);
        // The two tint directions move the green channel oppositely.
        assert!((pos.y - neg.y).abs() > 1.0e-3, "tint had no effect");
    }

    #[test]
    fn white_balance_never_negative_or_nan() {
        let out = white_balance(Vec3::new(-1.0, 2.0, 0.0), 2000.0, 2.0);
        assert!(out.x >= 0.0 && out.y >= 0.0 && out.z >= 0.0);
        assert!(out.is_finite());
    }

    #[test]
    fn uv_to_xy_round_trips_d65_ish() {
        // D65 ~ (x, y) = (0.3127, 0.3290); feed through uv and back.
        let (u, v) = cct_to_uv(REFERENCE_TEMPERATURE_K);
        let (x, y) = uv_to_xy(u, v);
        // Reference white should be in the neutral chromaticity neighbourhood.
        approx_tol(x, 0.313, 0.01);
        approx_tol(y, 0.329, 0.01);
    }
}
