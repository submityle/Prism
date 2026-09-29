//! Transfer-function, color-gamut, and luminance metadata per display output.
//!
//! Each [`DisplayOutput`] maps to a [`TransferMetadata`] record describing its
//! opto-electronic transfer function, the color gamut its primaries span, and
//! its reference-white and peak luminance in `nits` (cd/m^2).
//!
//! Color primaries and `nits` values are real, standards-derived constants.
//! Transfer curves that require transcendental evaluation (the `sRGB` power
//! segment and the `PQ` curve) are described as metadata and tagged with a
//! [`CurveStatus`]; this module intentionally performs no transcendental math.

use super::DisplayOutput;

/// A CIE 1931 xy chromaticity coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chromaticity {
    /// CIE x coordinate.
    pub x: f32,
    /// CIE y coordinate.
    pub y: f32,
}

impl Chromaticity {
    /// Constructs a chromaticity coordinate.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// The RGB primaries and white point defining a color gamut.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorGamut {
    /// Red primary chromaticity.
    pub red: Chromaticity,
    /// Green primary chromaticity.
    pub green: Chromaticity,
    /// Blue primary chromaticity.
    pub blue: Chromaticity,
    /// White point chromaticity.
    pub white: Chromaticity,
}

impl ColorGamut {
    /// ITU-R BT.709 primaries with a `D65` white point (`sRGB`/`scRGB`).
    pub const REC709: Self = Self {
        red: Chromaticity::new(0.640, 0.330),
        green: Chromaticity::new(0.300, 0.600),
        blue: Chromaticity::new(0.150, 0.060),
        white: Chromaticity::new(0.3127, 0.3290),
    };

    /// ITU-R BT.2020 primaries with a `D65` white point (`HDR10`/`PQ`).
    pub const REC2020: Self = Self {
        red: Chromaticity::new(0.708, 0.292),
        green: Chromaticity::new(0.170, 0.797),
        blue: Chromaticity::new(0.131, 0.046),
        white: Chromaticity::new(0.3127, 0.3290),
    };
}

/// Opto-electronic transfer function associated with a display output.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TransferFunction {
    /// `sRGB` piecewise curve (linear toe plus a power segment).
    Srgb,
    /// `scRGB` extended-range linear encoding (identity on scene-linear).
    ScRgbLinear,
    /// `PQ` (SMPTE ST 2084) perceptual quantizer curve.
    Pq,
}

/// Whether a transfer curve is evaluated here or deferred.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CurveStatus {
    /// The transfer relationship is representable with allowed arithmetic.
    Implemented,
    /// Evaluation is deferred because it needs transcendental math.
    Pending,
}

/// Full display metadata for one [`DisplayOutput`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferMetadata {
    /// Transfer function used to encode the signal.
    pub transfer: TransferFunction,
    /// Color gamut spanned by the output's primaries.
    pub gamut: ColorGamut,
    /// Reference (diffuse) white luminance in `nits`.
    pub reference_white_nits: f32,
    /// Peak luminance the output can represent in `nits`.
    pub peak_nits: f32,
    /// Whether the transfer curve is evaluated in this crate.
    pub curve_status: CurveStatus,
}

impl DisplayOutput {
    /// Returns the standards-derived metadata for this output.
    ///
    /// Luminance follows common convention: `SDR` diffuse white at 80 `nits`
    /// (`sRGB` reference), `scRGB` sharing the 80-`nit` diffuse white with a
    /// 1000-`nit` extended-range peak, and `HDR10` using the BT.2408 reference
    /// white of 203 `nits` with the `PQ` system peak of 10000 `nits`.
    #[must_use]
    pub const fn metadata(self) -> TransferMetadata {
        match self {
            Self::SdrSrgb => TransferMetadata {
                transfer: TransferFunction::Srgb,
                gamut: ColorGamut::REC709,
                reference_white_nits: 80.0,
                peak_nits: 80.0,
                // The linear toe is representable, but the power segment is not.
                curve_status: CurveStatus::Pending,
            },
            Self::ScRgb => TransferMetadata {
                transfer: TransferFunction::ScRgbLinear,
                gamut: ColorGamut::REC709,
                reference_white_nits: 80.0,
                peak_nits: 1000.0,
                // Linear encoding is the identity: nothing transcendental needed.
                curve_status: CurveStatus::Implemented,
            },
            Self::Hdr10Pq => TransferMetadata {
                transfer: TransferFunction::Pq,
                gamut: ColorGamut::REC2020,
                reference_white_nits: 203.0,
                peak_nits: 10_000.0,
                curve_status: CurveStatus::Pending,
            },
        }
    }

    /// Peak luminance this output can represent, in `nits`.
    #[must_use]
    pub const fn peak_nits(self) -> f32 {
        self.metadata().peak_nits
    }

    /// Reports whether this output is an `HDR` signal path.
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        matches!(self, Self::ScRgb | Self::Hdr10Pq)
    }

    /// Reports whether this output uses a wide (BT.2020) gamut.
    #[must_use]
    pub const fn is_wide_gamut(self) -> bool {
        matches!(self, Self::Hdr10Pq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing chromaticity/`nits` floats.
    const EPS: f32 = 1e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    #[test]
    fn sdr_uses_rec709_and_srgb_curve() {
        let meta = DisplayOutput::SdrSrgb.metadata();
        assert_eq!(meta.transfer, TransferFunction::Srgb);
        assert_eq!(meta.gamut, ColorGamut::REC709);
        assert!(close(meta.reference_white_nits, 80.0));
        assert!(close(meta.peak_nits, 80.0));
        assert!(!DisplayOutput::SdrSrgb.is_hdr());
    }

    #[test]
    fn scrgb_is_linear_and_implemented() {
        let meta = DisplayOutput::ScRgb.metadata();
        assert_eq!(meta.transfer, TransferFunction::ScRgbLinear);
        assert_eq!(meta.curve_status, CurveStatus::Implemented);
        assert_eq!(meta.gamut, ColorGamut::REC709);
        assert!(DisplayOutput::ScRgb.is_hdr());
        assert!(!DisplayOutput::ScRgb.is_wide_gamut());
    }

    #[test]
    fn hdr10_uses_rec2020_and_pq_pending() {
        let meta = DisplayOutput::Hdr10Pq.metadata();
        assert_eq!(meta.transfer, TransferFunction::Pq);
        assert_eq!(meta.curve_status, CurveStatus::Pending);
        assert_eq!(meta.gamut, ColorGamut::REC2020);
        assert!(close(meta.reference_white_nits, 203.0));
        assert!(close(meta.peak_nits, 10_000.0));
        assert!(DisplayOutput::Hdr10Pq.is_hdr());
        assert!(DisplayOutput::Hdr10Pq.is_wide_gamut());
    }

    #[test]
    fn rec2020_is_wider_than_rec709() {
        // The green primary is the clearest gamut-width differentiator.
        // `black_box` keeps the comparison out of const evaluation.
        let wide_green = core::hint::black_box(ColorGamut::REC2020.green.y);
        let narrow_green = core::hint::black_box(ColorGamut::REC709.green.y);
        assert!(wide_green > narrow_green);
        let wide_red = core::hint::black_box(ColorGamut::REC2020.red.x);
        let narrow_red = core::hint::black_box(ColorGamut::REC709.red.x);
        assert!(wide_red > narrow_red);
    }

    #[test]
    fn peak_nits_accessor_matches_metadata() {
        for output in [
            DisplayOutput::SdrSrgb,
            DisplayOutput::ScRgb,
            DisplayOutput::Hdr10Pq,
        ] {
            assert!(close(output.peak_nits(), output.metadata().peak_nits));
        }
    }
}
