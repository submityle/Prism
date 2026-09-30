//! `CIE` 1931 color-space contract for the particle color pipeline: linear
//! `sRGB` (`Rec.709`) tristimulus conversion, chromaticity coordinates, and
//! `Bradford` chromatic adaptation between reference white points.
//!
//! Particle emitters author colors in linear `sRGB`, but physically meaningful
//! operations — white balancing, cross-illuminant blending, spectral-style
//! tinting — live in device-independent `CIE` `XYZ`. This module is the
//! deterministic `CPU` reference the future `GPU` tint kernel reproduces bit for
//! bit. It owns three small pieces:
//!
//! 1. the fixed `sRGB`/`Rec.709` `D65` primaries matrix and its inverse, mapping
//!    linear `sRGB` to `CIE` `XYZ` and back;
//! 2. the `XYZ` <-> `xyY` split that separates chromaticity `(x, y)` from
//!    luminance `Y`, with divide-by-zero guards that fall back to the `D65`
//!    white chromaticity; and
//! 3. `Bradford` chromatic adaptation, which transports a color measured under
//!    one reference white to another by diagonally scaling it in the `Bradford`
//!    `LMS` cone-response basis.
//!
//! Determinism rules (design §29) hold: the arithmetic is pure linear algebra
//! plus division, with no transcendental calls (`sin` / `cos` / `exp` / `ln` /
//! `pow` / `cbrt`); every matrix coefficient is written out as a literal so the
//! result never depends on a runtime-built basis. The `std430` packing mirrors
//! an aligned `vec4` slot so the `GPU` color buffer binds against a stable
//! `ABI`.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Byte size of the `std430` packing of an [`Xyz`] triple: three `f32` scalars
/// promoted to one aligned `vec4` slot (16 bytes), leaving a one-scalar padding
/// tail so the block honors the `std430` 16-byte base alignment.
pub const CIE_XYZ_STD430_SIZE: usize = VEC4_STRIDE;

/// Absolute tolerance for the `f32` comparisons used by the tests; direct `==`
/// on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1.0e-6;

/// `x` chromaticity of the `CIE` `D65` reference white, used as the divide-by-
/// zero fallback for [`xyz_to_xyy`].
const D65_CHROMA_X: f32 = 0.312_7;

/// `y` chromaticity of the `CIE` `D65` reference white, used as the divide-by-
/// zero fallback for [`xyz_to_xyy`].
const D65_CHROMA_Y: f32 = 0.329_0;

/// A `CIE` `XYZ` tristimulus triple.
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

/// A `CIE` `xyY` color: chromaticity coordinates `(x, y)` plus luminance `Y`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xyy {
    /// The `x` chromaticity coordinate.
    pub x: f32,
    /// The `y` chromaticity coordinate.
    pub y: f32,
    /// The luminance `Y`.
    pub big_y: f32,
}

impl Xyy {
    /// Builds an [`Xyy`] color from chromaticity `(x, y)` and luminance `big_y`.
    #[must_use]
    pub const fn new(x: f32, y: f32, big_y: f32) -> Self {
        Self { x, y, big_y }
    }
}

/// A linear (non-gamma-encoded) `sRGB`/`Rec.709` color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearSrgb {
    /// The linear red channel.
    pub r: f32,
    /// The linear green channel.
    pub g: f32,
    /// The linear blue channel.
    pub b: f32,
}

impl LinearSrgb {
    /// Builds a [`LinearSrgb`] color from its three linear channels.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

/// `CIE` `D65` reference white as an [`Xyz`] triple normalized to `Y = 1`
/// (chromaticity `x = 0.3127`, `y = 0.3290`).
pub const D65_XYZ: Xyz = Xyz::new(0.950_455_9, 1.0, 1.089_057_8);

/// `CIE` `D50` reference white as an [`Xyz`] triple normalized to `Y = 1`
/// (chromaticity `x = 0.34567`, `y = 0.35850`).
pub const D50_XYZ: Xyz = Xyz::new(0.964_219_9, 1.0, 0.825_188_3);

/// Linear `sRGB`/`Rec.709` `D65` primaries matrix mapping linear `sRGB` to
/// `CIE` `XYZ` (rows are the `X`, `Y`, `Z` outputs).
const SRGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.412_390_8, 0.357_584_33, 0.180_480_8],
    [0.212_639, 0.715_168_65, 0.072_192_32],
    [0.019_330_818, 0.119_194_78, 0.950_532_15],
];

/// Inverse of [`SRGB_TO_XYZ`], mapping `CIE` `XYZ` back to linear `sRGB`.
const XYZ_TO_SRGB: [[f32; 3]; 3] = [
    [3.240_97, -1.537_383_2, -0.498_610_76],
    [-0.969_243_65, 1.875_967_5, 0.041_555_06],
    [0.055_630_08, -0.203_976_96, 1.056_971_5],
];

/// `Bradford` `LMS` cone-response matrix mapping `CIE` `XYZ` to the `Bradford`
/// `LMS` basis used for chromatic adaptation.
const BRADFORD: [[f32; 3]; 3] = [
    [0.895_1, 0.266_4, -0.161_4],
    [-0.750_2, 1.713_5, 0.036_7],
    [0.038_9, -0.068_5, 1.029_6],
];

/// Inverse of [`BRADFORD`], mapping the `Bradford` `LMS` basis back to `CIE`
/// `XYZ`.
const BRADFORD_INV: [[f32; 3]; 3] = [
    [0.986_992_9, -0.147_054_3, 0.159_962_7],
    [0.432_305_3, 0.518_360_3, 0.049_291_2],
    [-0.008_528_7, 0.040_042_8, 0.968_486_7],
];

/// Multiplies a 3x3 matrix by a length-3 column vector.
#[must_use]
fn mat3_mul_vec(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for (slot, row) in out.iter_mut().zip(m.iter()) {
        *slot = row[0] * v[0] + row[1] * v[1] + row[2] * v[2];
    }
    out
}

/// Converts a linear `sRGB`/`Rec.709` color to `CIE` `XYZ` using the fixed `D65`
/// primaries matrix.
#[must_use]
pub fn linear_srgb_to_xyz(c: &LinearSrgb) -> Xyz {
    let [x, y, z] = mat3_mul_vec(&SRGB_TO_XYZ, [c.r, c.g, c.b]);
    Xyz::new(x, y, z)
}

/// Converts a `CIE` `XYZ` color to linear `sRGB`/`Rec.709` using the inverse
/// `D65` primaries matrix.
#[must_use]
pub fn xyz_to_linear_srgb(c: &Xyz) -> LinearSrgb {
    let [r, g, b] = mat3_mul_vec(&XYZ_TO_SRGB, [c.x, c.y, c.z]);
    LinearSrgb::new(r, g, b)
}

/// Splits a `CIE` `XYZ` color into chromaticity `(x, y)` and luminance `Y`.
///
/// When the tristimulus sum `X + Y + Z` is non-positive (a fully black or
/// degenerate color has no defined chromaticity) the result falls back to the
/// `D65` white chromaticity while preserving the luminance `Y`.
#[must_use]
pub fn xyz_to_xyy(c: &Xyz) -> Xyy {
    let sum = c.x + c.y + c.z;
    if sum <= 0.0 {
        return Xyy::new(D65_CHROMA_X, D65_CHROMA_Y, c.y);
    }
    Xyy::new(c.x / sum, c.y / sum, c.y)
}

/// Reconstructs a `CIE` `XYZ` color from chromaticity `(x, y)` and luminance
/// `Y`.
///
/// When the chromaticity `y` is non-positive the color is treated as black and
/// maps to the `XYZ` origin, guarding the division by `y`.
#[must_use]
pub fn xyy_to_xyz(c: &Xyy) -> Xyz {
    if c.y <= 0.0 {
        return Xyz::new(0.0, 0.0, 0.0);
    }
    let ratio = c.big_y / c.y;
    let x = c.x * ratio;
    let z = (1.0 - c.x - c.y) * ratio;
    Xyz::new(x, c.big_y, z)
}

/// Transports a `CIE` `XYZ` color measured under `src_white` to the equivalent
/// color under `dst_white` using `Bradford` chromatic adaptation.
///
/// The source color and both reference whites are projected into the `Bradford`
/// `LMS` cone-response basis; the source is then diagonally scaled by the
/// component-wise ratio of destination to source white cone responses and
/// projected back to `CIE` `XYZ`. A non-positive source cone response leaves
/// that channel unscaled, guarding the division.
#[must_use]
pub fn bradford_adapt(src: &Xyz, src_white: &Xyz, dst_white: &Xyz) -> Xyz {
    let src_lms = mat3_mul_vec(&BRADFORD, [src.x, src.y, src.z]);
    let src_white_lms = mat3_mul_vec(&BRADFORD, [src_white.x, src_white.y, src_white.z]);
    let dst_white_lms = mat3_mul_vec(&BRADFORD, [dst_white.x, dst_white.y, dst_white.z]);

    let mut scaled = [0.0_f32; 3];
    for ((slot, &s), (&sw, &dw)) in scaled
        .iter_mut()
        .zip(src_lms.iter())
        .zip(src_white_lms.iter().zip(dst_white_lms.iter()))
    {
        *slot = if sw <= 0.0 { s } else { s * (dw / sw) };
    }

    let [x, y, z] = mat3_mul_vec(&BRADFORD_INV, scaled);
    Xyz::new(x, y, z)
}

/// Packs an [`Xyz`] triple into its `std430` byte layout: three little-endian
/// `f32` scalars followed by a zeroed one-scalar padding tail filling the
/// aligned `vec4` slot.
#[must_use]
pub fn to_std430(c: &Xyz) -> [u8; CIE_XYZ_STD430_SIZE] {
    let fields = [c.x, c.y, c.z];
    let mut bytes = [0_u8; CIE_XYZ_STD430_SIZE];
    for (slot, value) in bytes.chunks_exact_mut(4).zip(fields.iter()) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Total `std430` storage-buffer byte size for `count` [`Xyz`] elements,
/// reusing the shared clamp-to-one-element rule so an empty pool still reserves
/// one aligned `vec4` slot.
#[must_use]
pub fn gpu_storage_bytes(count: usize) -> usize {
    storage_bytes(CIE_XYZ_STD430_SIZE, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx_xyz(a: &Xyz, b: &Xyz) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    fn approx_srgb(a: &LinearSrgb, b: &LinearSrgb) -> bool {
        approx(a.r, b.r) && approx(a.g, b.g) && approx(a.b, b.b)
    }

    fn sample_colors() -> [LinearSrgb; 6] {
        [
            LinearSrgb::new(1.0, 0.0, 0.0),
            LinearSrgb::new(0.0, 1.0, 0.0),
            LinearSrgb::new(0.0, 0.0, 1.0),
            LinearSrgb::new(1.0, 1.0, 1.0),
            LinearSrgb::new(0.25, 0.5, 0.75),
            LinearSrgb::new(0.1, 0.9, 0.3),
        ]
    }

    #[test]
    fn srgb_to_xyz_to_srgb_roundtrip() {
        for c in sample_colors() {
            let back = xyz_to_linear_srgb(&linear_srgb_to_xyz(&c));
            assert!(approx_srgb(&c, &back), "{c:?} != {back:?}");
        }
    }

    #[test]
    fn xyz_to_srgb_to_xyz_roundtrip() {
        let samples = [
            Xyz::new(0.5, 0.4, 0.3),
            Xyz::new(0.2, 0.2, 0.2),
            Xyz::new(0.95, 1.0, 1.09),
        ];
        for c in samples {
            let back = linear_srgb_to_xyz(&xyz_to_linear_srgb(&c));
            assert!(approx_xyz(&c, &back), "{c:?} != {back:?}");
        }
    }

    #[test]
    fn white_maps_to_d65() {
        let white = LinearSrgb::new(1.0, 1.0, 1.0);
        let xyz = linear_srgb_to_xyz(&white);
        assert!(approx_xyz(&xyz, &D65_XYZ), "{xyz:?} != {:?}", D65_XYZ);
    }

    #[test]
    fn black_maps_to_origin() {
        let xyz = linear_srgb_to_xyz(&LinearSrgb::new(0.0, 0.0, 0.0));
        assert!(approx_xyz(&xyz, &Xyz::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn white_luminance_is_one() {
        let xyz = linear_srgb_to_xyz(&LinearSrgb::new(1.0, 1.0, 1.0));
        assert!(approx(xyz.y, 1.0));
    }

    #[test]
    fn xyy_roundtrip() {
        let samples = [
            Xyz::new(0.5, 0.4, 0.3),
            Xyz::new(0.2, 0.5, 0.9),
            Xyz::new(0.95, 1.0, 1.09),
        ];
        for c in samples {
            let back = xyy_to_xyz(&xyz_to_xyy(&c));
            assert!(approx_xyz(&c, &back), "{c:?} != {back:?}");
        }
    }

    #[test]
    fn xyy_big_y_equals_xyz_y() {
        let c = Xyz::new(0.5, 0.4, 0.3);
        let xyy = xyz_to_xyy(&c);
        assert!(approx(xyy.big_y, c.y));
    }

    #[test]
    fn xyy_chromaticity_sums_below_one() {
        let c = Xyz::new(0.5, 0.4, 0.3);
        let xyy = xyz_to_xyy(&c);
        assert!(xyy.x + xyy.y < 1.0);
        assert!(xyy.x > 0.0 && xyy.y > 0.0);
    }

    #[test]
    fn xyz_to_xyy_zero_falls_back_to_d65_chromaticity() {
        let xyy = xyz_to_xyy(&Xyz::new(0.0, 0.0, 0.0));
        assert!(approx(xyy.x, D65_CHROMA_X));
        assert!(approx(xyy.y, D65_CHROMA_Y));
        assert!(approx(xyy.big_y, 0.0));
    }

    #[test]
    fn xyy_to_xyz_guards_zero_luminance_chromaticity() {
        let xyz = xyy_to_xyz(&Xyy::new(0.3, 0.0, 0.5));
        assert!(approx_xyz(&xyz, &Xyz::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn d65_white_roundtrips_through_xyy() {
        let back = xyy_to_xyz(&xyz_to_xyy(&D65_XYZ));
        assert!(approx_xyz(&back, &D65_XYZ));
    }

    #[test]
    fn bradford_same_white_is_identity() {
        let c = Xyz::new(0.4, 0.5, 0.6);
        let out = bradford_adapt(&c, &D65_XYZ, &D65_XYZ);
        assert!(approx_xyz(&out, &c), "{out:?} != {c:?}");
    }

    #[test]
    fn bradford_maps_src_white_to_dst_white() {
        let out = bradford_adapt(&D65_XYZ, &D65_XYZ, &D50_XYZ);
        assert!(approx_xyz(&out, &D50_XYZ), "{out:?} != {:?}", D50_XYZ);
    }

    #[test]
    fn bradford_maps_dst_white_to_src_white_reversed() {
        let out = bradford_adapt(&D50_XYZ, &D50_XYZ, &D65_XYZ);
        assert!(approx_xyz(&out, &D65_XYZ), "{out:?} != {:?}", D65_XYZ);
    }

    #[test]
    fn bradford_changes_color_across_whites() {
        let c = Xyz::new(0.4, 0.5, 0.6);
        let out = bradford_adapt(&c, &D65_XYZ, &D50_XYZ);
        assert!(!approx_xyz(&out, &c));
    }

    #[test]
    fn bradford_roundtrip_d65_d50_d65() {
        let c = Xyz::new(0.4, 0.5, 0.6);
        let to_d50 = bradford_adapt(&c, &D65_XYZ, &D50_XYZ);
        let back = bradford_adapt(&to_d50, &D50_XYZ, &D65_XYZ);
        assert!(approx_xyz(&back, &c), "{back:?} != {c:?}");
    }

    #[test]
    fn d65_constant_is_normalized() {
        assert!(approx(D65_XYZ.y, 1.0));
        let recovered = xyz_to_xyy(&D65_XYZ);
        assert!(approx(recovered.x, D65_CHROMA_X));
        assert!(approx(recovered.y, D65_CHROMA_Y));
    }

    #[test]
    fn d50_constant_is_normalized() {
        assert!(approx(D50_XYZ.y, 1.0));
        // The white-point constant is stored to seven significant figures, so
        // its recovered chromaticity is checked against a coarser tolerance
        // than the tight round-trip epsilon.
        const CHROMA_EPS: f32 = 1.0e-4;
        let recovered = xyz_to_xyy(&D50_XYZ);
        assert!((recovered.x - 0.345_67).abs() < CHROMA_EPS);
        assert!((recovered.y - 0.358_5).abs() < CHROMA_EPS);
    }

    #[test]
    fn primary_channels_roundtrip_independently() {
        let red = LinearSrgb::new(1.0, 0.0, 0.0);
        let back = xyz_to_linear_srgb(&linear_srgb_to_xyz(&red));
        assert!(approx(back.r, 1.0));
        assert!(approx(back.g, 0.0));
        assert!(approx(back.b, 0.0));
    }

    #[test]
    fn constructors_store_fields() {
        let xyz = Xyz::new(1.0, 2.0, 3.0);
        assert!(approx(xyz.x, 1.0) && approx(xyz.y, 2.0) && approx(xyz.z, 3.0));
        let xyy = Xyy::new(0.3, 0.4, 0.5);
        assert!(approx(xyy.x, 0.3) && approx(xyy.y, 0.4) && approx(xyy.big_y, 0.5));
        let rgb = LinearSrgb::new(0.1, 0.2, 0.3);
        assert!(approx(rgb.r, 0.1) && approx(rgb.g, 0.2) && approx(rgb.b, 0.3));
    }

    #[test]
    fn std430_size_is_sixteen() {
        assert_eq!(CIE_XYZ_STD430_SIZE, 16);
    }

    #[test]
    fn std430_byte_layout_roundtrips() {
        let c = Xyz::new(0.25, 0.5, 0.75);
        let bytes = to_std430(&c);
        let fields = [c.x, c.y, c.z];
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
        assert_eq!(gpu_storage_bytes(0), CIE_XYZ_STD430_SIZE);
    }

    #[test]
    fn gpu_storage_bytes_scales_with_count() {
        assert_eq!(gpu_storage_bytes(4), CIE_XYZ_STD430_SIZE * 4);
        assert_eq!(gpu_storage_bytes(100), CIE_XYZ_STD430_SIZE * 100);
    }
}
