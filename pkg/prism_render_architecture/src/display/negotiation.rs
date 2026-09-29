//! `HDR`/`SDR` capability negotiation between the renderer and the display.
//!
//! The renderer requests a preferred [`DisplayOutput`], but the actual sink may
//! not support it. [`DisplayCapabilities`] captures what a sink advertises, and
//! [`DisplayCapabilities::negotiate`] resolves the preference to a supported
//! output, downgrading to `SDR` `sRGB` (always available) when an `HDR` path is
//! unsupported or under-powered.

use super::DisplayOutput;

/// Minimum peak luminance, in `nits`, we require before treating a sink as a
/// usable `HDR` target.
const MIN_HDR_PEAK_NITS: f32 = 400.0;

/// Capabilities advertised by a display sink.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayCapabilities {
    /// Whether the sink accepts an `HDR` signal at all.
    pub supports_hdr: bool,
    /// Whether the sink covers a wide (BT.2020) gamut.
    pub supports_wide_gamut: bool,
    /// Peak luminance the sink can present, in `nits`.
    pub max_peak_nits: f32,
}

impl DisplayCapabilities {
    /// A baseline `SDR`-only sink: no `HDR`, no wide gamut, 80-`nit` peak.
    pub const SDR_ONLY: Self = Self {
        supports_hdr: false,
        supports_wide_gamut: false,
        max_peak_nits: 80.0,
    };

    /// Reports whether this sink can drive `output` directly.
    #[must_use]
    pub fn supports(self, output: DisplayOutput) -> bool {
        match output {
            // SDR sRGB is the universal fallback and always supported.
            DisplayOutput::SdrSrgb => true,
            // scRGB needs an HDR-capable sink with enough headroom.
            DisplayOutput::ScRgb => self.supports_hdr && self.has_hdr_headroom(),
            // HDR10 PQ additionally needs wide-gamut support.
            DisplayOutput::Hdr10Pq => {
                self.supports_hdr && self.supports_wide_gamut && self.has_hdr_headroom()
            }
        }
    }

    /// Resolves `desired` to a supported output, downgrading when necessary.
    ///
    /// `HDR10` `PQ` falls back to `scRGB` when the sink lacks wide gamut but is
    /// otherwise `HDR`-capable; any unsupported `HDR` request ultimately falls
    /// back to `SDR` `sRGB`.
    #[must_use]
    pub fn negotiate(self, desired: DisplayOutput) -> NegotiationResult {
        if self.supports(desired) {
            return NegotiationResult {
                output: desired,
                downgraded: false,
            };
        }
        // Try a wide-gamut HDR request as plain HDR scRGB before dropping to SDR.
        if matches!(desired, DisplayOutput::Hdr10Pq) && self.supports(DisplayOutput::ScRgb) {
            return NegotiationResult {
                output: DisplayOutput::ScRgb,
                downgraded: true,
            };
        }
        NegotiationResult {
            output: DisplayOutput::SdrSrgb,
            downgraded: true,
        }
    }

    /// Whether the advertised peak clears the `HDR` usability threshold.
    fn has_hdr_headroom(self) -> bool {
        // NaN is not >=, so a NaN peak correctly fails the HDR headroom check.
        self.max_peak_nits >= MIN_HDR_PEAK_NITS
    }
}

/// Outcome of negotiating a desired output against a sink's capabilities.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NegotiationResult {
    /// The output the renderer should actually target.
    pub output: DisplayOutput,
    /// Whether the result differs from the requested output.
    pub downgraded: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    const HDR_WIDE: DisplayCapabilities = DisplayCapabilities {
        supports_hdr: true,
        supports_wide_gamut: true,
        max_peak_nits: 1000.0,
    };

    const HDR_NARROW: DisplayCapabilities = DisplayCapabilities {
        supports_hdr: true,
        supports_wide_gamut: false,
        max_peak_nits: 1000.0,
    };

    #[test]
    fn sdr_is_always_supported() {
        assert!(DisplayCapabilities::SDR_ONLY.supports(DisplayOutput::SdrSrgb));
        let result = DisplayCapabilities::SDR_ONLY.negotiate(DisplayOutput::SdrSrgb);
        assert_eq!(
            result,
            NegotiationResult {
                output: DisplayOutput::SdrSrgb,
                downgraded: false,
            }
        );
    }

    #[test]
    fn hdr_request_on_sdr_sink_downgrades_to_sdr() {
        let result = DisplayCapabilities::SDR_ONLY.negotiate(DisplayOutput::Hdr10Pq);
        assert_eq!(
            result,
            NegotiationResult {
                output: DisplayOutput::SdrSrgb,
                downgraded: true,
            }
        );
    }

    #[test]
    fn full_hdr_sink_honors_hdr10() {
        let result = HDR_WIDE.negotiate(DisplayOutput::Hdr10Pq);
        assert_eq!(
            result,
            NegotiationResult {
                output: DisplayOutput::Hdr10Pq,
                downgraded: false,
            }
        );
    }

    #[test]
    fn narrow_gamut_hdr_downgrades_hdr10_to_scrgb() {
        let result = HDR_NARROW.negotiate(DisplayOutput::Hdr10Pq);
        assert_eq!(
            result,
            NegotiationResult {
                output: DisplayOutput::ScRgb,
                downgraded: true,
            }
        );
    }

    #[test]
    fn insufficient_peak_fails_hdr_headroom() {
        let dim = DisplayCapabilities {
            supports_hdr: true,
            supports_wide_gamut: true,
            max_peak_nits: 200.0,
        };
        assert!(!dim.supports(DisplayOutput::ScRgb));
        let result = dim.negotiate(DisplayOutput::ScRgb);
        assert_eq!(result.output, DisplayOutput::SdrSrgb);
        assert!(result.downgraded);
    }

    #[test]
    fn nan_peak_is_not_hdr_capable() {
        let broken = DisplayCapabilities {
            supports_hdr: true,
            supports_wide_gamut: true,
            max_peak_nits: f32::NAN,
        };
        assert!(!broken.supports(DisplayOutput::Hdr10Pq));
    }
}
