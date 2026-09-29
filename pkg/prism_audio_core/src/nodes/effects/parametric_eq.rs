//! Multi-band parametric equaliser built by cascading reusable [`Biquad`]
//! sections.
//!
//! A parametric EQ is the workhorse of mixing: several independently tunable
//! filter bands (peaking bells, shelves, high-/low-pass) applied in series.
//! Rather than re-deriving the filter math, this node cascades the shared
//! [`Biquad`] DSP core from [`crate::nodes::biquad`], so each band is an exact
//! RBJ-cookbook second-order section.
//!
//! All state (every band's per-channel history) is pre-allocated at
//! construction, so [`ParametricEqNode::process`] is real-time safe.

use alloc::vec::Vec;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::nodes::biquad::{Biquad, BiquadCoeffs, BiquadKind};

/// The tunable description of a single EQ band.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EqBand {
    /// The response shape of this band.
    pub kind: BiquadKind,
    /// Centre / corner frequency in Hertz.
    pub freq_hz: Sample,
    /// Quality factor (bandwidth). Higher is narrower.
    pub q: Sample,
    /// Boost/cut in decibels (used by peaking and shelving kinds).
    pub gain_db: Sample,
}

impl EqBand {
    /// Convenience constructor for a peaking (bell) band.
    #[must_use]
    pub fn peaking(freq_hz: Sample, q: Sample, gain_db: Sample) -> Self {
        Self {
            kind: BiquadKind::Peaking,
            freq_hz,
            q,
            gain_db,
        }
    }

    /// Convenience constructor for a low-shelf band.
    #[must_use]
    pub fn low_shelf(freq_hz: Sample, q: Sample, gain_db: Sample) -> Self {
        Self {
            kind: BiquadKind::LowShelf,
            freq_hz,
            q,
            gain_db,
        }
    }

    /// Convenience constructor for a high-shelf band.
    #[must_use]
    pub fn high_shelf(freq_hz: Sample, q: Sample, gain_db: Sample) -> Self {
        Self {
            kind: BiquadKind::HighShelf,
            freq_hz,
            q,
            gain_db,
        }
    }
}

/// A cascade of [`Biquad`] sections forming a multi-band parametric EQ
/// (input port 0 -> output port 0).
///
/// An empty cascade is a valid unity pass-through. Bands are applied in the
/// order supplied at construction.
#[derive(Debug, Clone)]
pub struct ParametricEqNode {
    sample_rate: u32,
    /// The per-band filter descriptions, kept alongside their DSP sections so a
    /// band can be redesigned in place.
    bands: Vec<EqBand>,
    /// One [`Biquad`] section per band, sharing the channel width.
    sections: Vec<Biquad>,
}

impl ParametricEqNode {
    /// Builds an EQ for a `channels`-wide signal running at `sample_rate` Hz
    /// from an ordered list of `bands`.
    #[must_use]
    pub fn new(sample_rate: u32, channels: usize, bands: &[EqBand]) -> Self {
        let mut sections = Vec::with_capacity(bands.len());
        for band in bands {
            sections.push(Biquad::new(
                BiquadCoeffs::design(band.kind, sample_rate, band.freq_hz, band.q, band.gain_db),
                channels,
            ));
        }
        Self {
            sample_rate,
            bands: bands.to_vec(),
            sections,
        }
    }

    /// Returns the number of bands in the cascade.
    #[inline]
    #[must_use]
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Returns the current description of band `index`, if it exists.
    #[inline]
    #[must_use]
    pub fn band(&self, index: usize) -> Option<EqBand> {
        self.bands.get(index).copied()
    }

    /// Redesigns band `index` with new settings, preserving its filter state so
    /// the change is click-free. Out-of-range indices are ignored.
    pub fn set_band(&mut self, index: usize, band: EqBand) {
        if let (Some(slot), Some(section)) =
            (self.bands.get_mut(index), self.sections.get_mut(index))
        {
            *slot = band;
            section.set_coeffs(BiquadCoeffs::design(
                band.kind,
                self.sample_rate,
                band.freq_hz,
                band.q,
                band.gain_db,
            ));
        }
    }
}

impl AudioNode for ParametricEqNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        output.copy_from(input);
        // Cascade: each section filters the running signal in place.
        for section in &mut self.sections {
            section.process_inplace(output);
        }
    }

    fn reset(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use crate::nodes::biquad::BiquadNode;

    fn stereo(frames: usize) -> AudioBuffer {
        AudioBuffer::new(ChannelLayout::Stereo, frames)
    }

    #[test]
    fn empty_cascade_is_unity() {
        let mut eq = ParametricEqNode::new(48_000, 2, &[]);
        let mut input = stereo(8);
        for (i, s) in input.channel_mut(0).iter_mut().enumerate() {
            *s = i as Sample - 3.0;
        }
        let inputs = [input.clone()];
        let mut outputs = [stereo(8)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        let ctx = RenderContext {
            sample_rate: 48_000,
            frames: 8,
            playhead: 0,
        };
        eq.process(&ctx, &mut io);
        assert_eq!(outputs[0].channel(0), inputs[0].channel(0));
    }

    #[test]
    fn single_band_matches_biquad_node() {
        // A one-band cascade must be sample-identical to the equivalent
        // standalone BiquadNode fed the same signal.
        let band = EqBand::peaking(1_000.0, 1.2, 6.0);
        let mut eq = ParametricEqNode::new(48_000, 1, &[band]);
        let mut node = BiquadNode::new(BiquadKind::Peaking, 48_000, 1_000.0, 1.2, 6.0, 1);

        let mut src = AudioBuffer::new(ChannelLayout::Mono, 64);
        src.channel_mut(0)[0] = 1.0; // impulse
        let ctx = RenderContext {
            sample_rate: 48_000,
            frames: 64,
            playhead: 0,
        };

        let inputs_a = [src.clone()];
        let mut out_a = [AudioBuffer::new(ChannelLayout::Mono, 64)];
        let mut io_a = ProcessIo::new(&inputs_a, &mut out_a);
        eq.process(&ctx, &mut io_a);

        let inputs_b = [src.clone()];
        let mut out_b = [AudioBuffer::new(ChannelLayout::Mono, 64)];
        let mut io_b = ProcessIo::new(&inputs_b, &mut out_b);
        node.process(&ctx, &mut io_b);

        for (a, b) in out_a[0].channel(0).iter().zip(out_b[0].channel(0)) {
            assert!((a - b).abs() < 1.0e-6, "cascade {a} vs node {b}");
        }
    }

    #[test]
    fn cascade_is_stable_and_finite() {
        let bands = [
            EqBand::low_shelf(120.0, 0.7, 4.0),
            EqBand::peaking(1_000.0, 2.0, -6.0),
            EqBand::high_shelf(8_000.0, 0.7, 3.0),
        ];
        let mut eq = ParametricEqNode::new(48_000, 2, &bands);
        assert_eq!(eq.band_count(), 3);

        let mut src = stereo(512);
        src.channel_mut(0)[0] = 1.0;
        src.channel_mut(1)[0] = 1.0;
        let ctx = RenderContext {
            sample_rate: 48_000,
            frames: 512,
            playhead: 0,
        };
        let inputs = [src];
        let mut outputs = [stereo(512)];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        eq.process(&ctx, &mut io);

        let mut energy_tail = 0.0f32;
        for &s in &outputs[0].channel(0)[400..] {
            assert!(s.is_finite());
            energy_tail += s * s;
        }
        // A stable IIR impulse response has decayed to near silence by 400
        // samples in.
        assert!(energy_tail < 1.0e-4, "impulse response did not decay: {energy_tail}");
    }

    #[test]
    fn set_band_updates_response() {
        let mut eq = ParametricEqNode::new(48_000, 1, &[EqBand::peaking(1_000.0, 1.0, 0.0)]);
        eq.set_band(0, EqBand::peaking(2_000.0, 1.5, -12.0));
        assert_eq!(eq.band(0), Some(EqBand::peaking(2_000.0, 1.5, -12.0)));
        // Out-of-range set is a no-op.
        eq.set_band(9, EqBand::peaking(500.0, 1.0, 3.0));
        assert_eq!(eq.band_count(), 1);
    }
}
