//! **`YCbCr`** luma / chroma transforms (ITU-R BT.601 / BT.709).
//!
//! `YCbCr` is the luma / blue-difference / red-difference basis used by the
//! overwhelming majority of image and video codecs (JPEG, H.264/HEVC, the
//! `YUV` planes that back streamed and movie textures). Material tooling that
//! ingests those sources, or that drives a chroma-subsampling compressor front
//! end, needs an exact RGB ↔ `YCbCr` map. Unlike the fixed-matrix
//! [`super::ycocg`] transform the coefficients depend on the primaries, so this
//! module is parameterised by the luma weighting [`YCbCrMatrix`] (BT.601 for
//! standard-definition / `JPEG`, BT.709 for high-definition).
//!
//! The float form operates on `[0,1]` channels: luma `Y` lands on `[0,1]` and
//! the chroma pair `Cb` / `Cr` on `[-0.5,0.5]` (full-swing, zero chroma for an
//! achromatic colour). The forward uses `Y = Kr·R + Kg·G + Kb·B`,
//! `Cb = (B - Y) / (2(1-Kb))`, `Cr = (R - Y) / (2(1-Kr))`; the inverse is the
//! exact analytic `R = Y + 2(1-Kr)·Cr`, `B = Y + 2(1-Kb)·Cb`,
//! `G = (Y - Kr·R - Kb·B) / Kg`. Round-trip to floating-point tolerance is the
//! primary anti-fake oracle.
//!
//! Pure matrix arithmetic (no transcendentals, no AI/ML), so a CPU golden
//! matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * ITU-R BT.601-7, "Studio encoding parameters of digital television".
//! * ITU-R BT.709-6, "Parameter values for the HDTV standards".

/// Luma-coefficient set selecting the `YCbCr` primaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YCbCrMatrix {
    /// ITU-R BT.601 (standard definition, also used by baseline `JPEG`):
    /// `Kr = 0.299`, `Kb = 0.114`.
    Bt601,
    /// ITU-R BT.709 (high definition): `Kr = 0.2126`, `Kb = 0.0722`.
    Bt709,
}

impl YCbCrMatrix {
    /// Return the `(Kr, Kb)` luma coefficients; `Kg = 1 - Kr - Kb`.
    #[inline]
    #[must_use]
    const fn kr_kb(self) -> (f32, f32) {
        match self {
            Self::Bt601 => (0.299, 0.114),
            Self::Bt709 => (0.2126, 0.0722),
        }
    }
}

/// Convert a linear RGB triple on `[0,1]` to full-swing `YCbCr`.
///
/// Returns `[Y, Cb, Cr]` with luma on `[0,1]` and the chroma pair on
/// `[-0.5,0.5]` (zero chroma for an achromatic input).
#[inline]
#[must_use]
pub fn rgb_to_ycbcr(rgb: [f32; 3], matrix: YCbCrMatrix) -> [f32; 3] {
    let [r, g, b] = rgb;
    let (kr, kb) = matrix.kr_kb();
    let kg = 1.0 - kr - kb;
    let y = kr * r + kg * g + kb * b;
    let cb = (b - y) / (2.0 * (1.0 - kb));
    let cr = (r - y) / (2.0 * (1.0 - kr));
    [y, cb, cr]
}

/// Inverse of [`rgb_to_ycbcr`]: reconstruct the RGB triple from `[Y, Cb, Cr]`.
///
/// Uses the exact analytic inverse `R = Y + 2(1-Kr)·Cr`,
/// `B = Y + 2(1-Kb)·Cb`, `G = (Y - Kr·R - Kb·B) / Kg`.
#[inline]
#[must_use]
pub fn ycbcr_to_rgb(ycbcr: [f32; 3], matrix: YCbCrMatrix) -> [f32; 3] {
    let [y, cb, cr] = ycbcr;
    let (kr, kb) = matrix.kr_kb();
    let kg = 1.0 - kr - kb;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = (y - kr * r - kb * b) / kg;
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::{rgb_to_ycbcr, ycbcr_to_rgb, YCbCrMatrix};

    const MATRICES: [YCbCrMatrix; 2] = [YCbCrMatrix::Bt601, YCbCrMatrix::Bt709];

    /// `YCbCr` round-trips to floating-point tolerance for both matrices.
    #[test]
    fn ycbcr_round_trips() {
        let steps = [0.0f32, 0.1, 0.25, 0.5, 0.73, 0.9, 1.0];
        for &m in &MATRICES {
            for &r in &steps {
                for &g in &steps {
                    for &b in &steps {
                        let rt = ycbcr_to_rgb(rgb_to_ycbcr([r, g, b], m), m);
                        for k in 0..3 {
                            let diff = (rt[k] - [r, g, b][k]).abs();
                            assert!(diff < 1e-5, "drift {diff} at {:?} {m:?}", [r, g, b]);
                        }
                    }
                }
            }
        }
    }

    /// Achromatic inputs zero the chroma and pass luma straight through.
    #[test]
    fn ycbcr_achromatic_is_neutral() {
        for &m in &MATRICES {
            for i in 0..=10 {
                let v = i as f32 / 10.0;
                let [y, cb, cr] = rgb_to_ycbcr([v, v, v], m);
                assert!((y - v).abs() < 1e-6, "grey luma {m:?}");
                assert!(cb.abs() < 1e-6 && cr.abs() < 1e-6, "grey chroma {m:?}");
            }
        }
    }

    /// Luma matches the matrix weights and the chroma axes carry the right
    /// signs (Cr tracks red, Cb tracks blue).
    #[test]
    fn ycbcr_known_axes() {
        for &m in &MATRICES {
            // White projects to unit luma, zero chroma.
            let [y, cb, cr] = rgb_to_ycbcr([1.0, 1.0, 1.0], m);
            assert!((y - 1.0).abs() < 1e-6, "white luma {m:?}");
            assert!(cb.abs() < 1e-6 && cr.abs() < 1e-6, "white chroma {m:?}");

            // Pure red: positive Cr, and Cb negative (B below its luma).
            let red = rgb_to_ycbcr([1.0, 0.0, 0.0], m);
            assert!(red[2] > 0.0, "red Cr positive {m:?}");
            assert!(red[1] < 0.0, "red Cb negative {m:?}");

            // Pure blue: positive Cb, and Cr negative.
            let blue = rgb_to_ycbcr([0.0, 0.0, 1.0], m);
            assert!(blue[1] > 0.0, "blue Cb positive {m:?}");
            assert!(blue[2] < 0.0, "blue Cr negative {m:?}");
        }
    }

    /// BT.601 and BT.709 genuinely differ (guards against a stubbed matrix).
    #[test]
    fn matrices_differ() {
        let green = [0.0f32, 1.0, 0.0];
        let a = rgb_to_ycbcr(green, YCbCrMatrix::Bt601);
        let b = rgb_to_ycbcr(green, YCbCrMatrix::Bt709);
        // Green luma differs between the two weightings (0.587 vs 0.7152).
        assert!((a[0] - b[0]).abs() > 0.1, "matrices must differ");
    }
}
