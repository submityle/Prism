//! Tone-mapping luminance targets with `NaN`-safe clamping.
//!
//! A [`ToneMapTarget`] carries the two luminance knobs the present pass needs:
//! the diffuse *paper-white* level and the *peak* level, both in `nits`. Values
//! supplied by configuration or scripting are untrusted, so [`ToneMapTarget`]
//! clamps them into the physically meaningful range of the selected
//! [`DisplayOutput`] and guarantees `paper_white <= peak`.
//!
//! Every clamp guards `NaN` explicitly before calling [`f32::clamp`], since
//! `NaN` has no defined ordering and would otherwise produce an unspecified
//! result.

use super::DisplayOutput;

/// Smallest luminance we allow a target to be clamped to, in `nits`.
const MIN_NITS: f32 = 1.0;

/// Clamps `value` into `[lo, hi]`, mapping `NaN` to `lo`.
fn sanitize(value: f32, lo: f32, hi: f32) -> f32 {
    if value.is_nan() {
        lo
    } else {
        value.clamp(lo, hi)
    }
}

/// Diffuse-white and peak luminance targets for tone mapping, in `nits`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneMapTarget {
    /// Diffuse (paper) white luminance in `nits`.
    pub paper_white_nits: f32,
    /// Peak luminance in `nits`.
    pub peak_nits: f32,
}

impl ToneMapTarget {
    /// Builds a target from raw paper-white and peak values.
    #[must_use]
    pub const fn new(paper_white_nits: f32, peak_nits: f32) -> Self {
        Self {
            paper_white_nits,
            peak_nits,
        }
    }

    /// Returns a copy clamped to the capabilities of `output`.
    ///
    /// Peak is clamped to `[MIN_NITS, output.peak_nits()]`; paper-white is then
    /// clamped to `[MIN_NITS, peak]`, so the invariant `paper_white <= peak`
    /// always holds. `NaN` inputs collapse to the lower bound.
    #[must_use]
    pub fn clamped_to(self, output: DisplayOutput) -> Self {
        let ceiling = output.peak_nits();
        let peak = sanitize(self.peak_nits, MIN_NITS, ceiling);
        let paper_white = sanitize(self.paper_white_nits, MIN_NITS, peak);
        Self {
            paper_white_nits: paper_white,
            peak_nits: peak,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    #[test]
    fn in_range_values_pass_through() {
        let target = ToneMapTarget::new(200.0, 800.0).clamped_to(DisplayOutput::ScRgb);
        assert!(close(target.paper_white_nits, 200.0));
        assert!(close(target.peak_nits, 800.0));
    }

    #[test]
    fn peak_is_clamped_to_output_ceiling() {
        // scRGB peaks at 1000 nits; request 5000.
        let target = ToneMapTarget::new(200.0, 5000.0).clamped_to(DisplayOutput::ScRgb);
        assert!(close(target.peak_nits, 1000.0));
    }

    #[test]
    fn paper_white_never_exceeds_peak() {
        // Paper-white above the SDR ceiling collapses to the clamped peak.
        let target = ToneMapTarget::new(500.0, 60.0).clamped_to(DisplayOutput::SdrSrgb);
        assert!(target.paper_white_nits <= target.peak_nits);
        assert!(close(target.peak_nits, 60.0));
        assert!(close(target.paper_white_nits, 60.0));
    }

    #[test]
    fn nan_collapses_to_lower_bound() {
        let target = ToneMapTarget::new(f32::NAN, f32::NAN).clamped_to(DisplayOutput::Hdr10Pq);
        assert!(close(target.peak_nits, MIN_NITS));
        assert!(close(target.paper_white_nits, MIN_NITS));
    }

    #[test]
    fn negative_values_clamp_to_minimum() {
        let target = ToneMapTarget::new(-10.0, -5.0).clamped_to(DisplayOutput::Hdr10Pq);
        assert!(close(target.peak_nits, MIN_NITS));
        assert!(close(target.paper_white_nits, MIN_NITS));
    }

    #[test]
    fn hdr_allows_high_peak() {
        let target = ToneMapTarget::new(203.0, 4000.0).clamped_to(DisplayOutput::Hdr10Pq);
        assert!(close(target.peak_nits, 4000.0));
        assert!(close(target.paper_white_nits, 203.0));
    }
}
