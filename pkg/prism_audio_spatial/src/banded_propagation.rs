//! Real-time three-band propagation shaper: the voice-side consumer of a
//! geometric-acoustics [`BandGains`] spectrum.
//!
//! The geometry and propagation layers of this crate colour every arrival with
//! a compact three-band spectrum (see [`band_spectrum`]): a reflection off
//! carpet is dull, air absorption rolls off highs with distance, and an edge
//! diffraction behaves like a low-pass. That spectrum is produced at control
//! rate -- once per path update -- and travels to the audio thread as a
//! [`BandGains`]. Until now nothing on the real-time side actually *applied*
//! it: a voice could carry the colour but not hear it. This module closes that
//! gap.
//!
//! # What it does
//!
//! [`BandedPropagationShaper`] splits a voice's dry signal into the three
//! contiguous bands defined by [`PROPAGATION_BAND_EDGES`] with a fourth-order
//! Linkwitz-Riley filterbank ([`LinkwitzRileyCrossover`]), scales each band by
//! its smoothed per-band gain, and sums the bands back into a single output.
//! Because the Linkwitz-Riley bank is magnitude-flat under summation (the
//! all-pass phase compensation keeps the reconstructed response flat), a
//! [`BandGains::UNITY`] spectrum reproduces the input unchanged, and any other
//! spectrum applies exactly the per-band attenuation the geometry backend
//! asked for.
//!
//! # Real-time contract
//!
//! Construction pre-allocates the crossover, the per-band scratch buffers, and
//! the per-band gain smoothers. [`BandedPropagationShaper::process_block`] and
//! the [`AudioNode`] implementation are then allocation free, lock free, and
//! panic free, so the shaper can run on a device callback thread. Band gains
//! are updated with [`BandedPropagationShaper::set_bands`], which retargets the
//! smoothers so a changed spectrum glides in without a click.
//!
//! # Relationship
//!
//! The control-rate source is a [`BandGains`] produced by the geometry
//! backends (`prism_audio_geometry` / `prism_audio_geometry_gpu`) and carried
//! on a [`PropagationPath`]. The authoring-time per-band material lives in
//! [`BandedAcousticMaterial`]. This module is the final, real-time consumer of
//! that chain. It reuses the backend-neutral [`LinkwitzRileyCrossover`] and
//! parameter smoothers from [`prism_audio_core`]; it adds no new DSP primitive,
//! only the wiring that turns a [`BandGains`] into audible per-band shaping.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, Dolby, or Google Resonance Audio source or
//! derived code**, and no AI- or ML-derived code. The three-band split, the
//! Linkwitz-Riley crossover, and the per-band gain application are implemented
//! from standard, publicly documented signal-processing knowledge.
//!
//! [`band_spectrum`]: crate::band_spectrum
//! [`BandedAcousticMaterial`]: crate::material_spectrum::BandedAcousticMaterial
//! [`PropagationPath`]: crate::propagation::PropagationPath

use alloc::vec::Vec;

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::math::Sample;
use prism_audio_core::nodes::LinkwitzRileyCrossover;
use prism_audio_core::param::{Ramp, Smoothed};

use crate::band_spectrum::{BandGains, PROPAGATION_BAND_COUNT, PROPAGATION_BAND_EDGES};

/// A real-time node that applies a three-band [`BandGains`] propagation
/// spectrum to a voice's signal.
///
/// The shaper owns a fourth-order Linkwitz-Riley filterbank split at
/// [`PROPAGATION_BAND_EDGES`], one scratch [`AudioBuffer`] per band, and one
/// [`Smoothed`] gain per band. Each processed block splits the input into
/// bands, scales band `b` by `gains[b]` (advanced per sample for click-free
/// changes), and sums the scaled bands into the output.
///
/// All buffers and smoothers are allocated once in [`Self::new`]; processing is
/// real-time safe. The shaper is configured for a fixed [`ChannelLayout`] and a
/// maximum block size; blocks longer than that maximum are processed up to the
/// pre-allocated capacity.
pub struct BandedPropagationShaper {
    crossover: LinkwitzRileyCrossover,
    bands: Vec<AudioBuffer>,
    gains: [Smoothed; PROPAGATION_BAND_COUNT],
}

impl BandedPropagationShaper {
    /// Builds a shaper for `layout` handling blocks up to `max_frames` frames.
    ///
    /// The crossover is split at [`PROPAGATION_BAND_EDGES`], yielding
    /// [`PROPAGATION_BAND_COUNT`] bands. All per-band gains start settled at
    /// unity, so a freshly built shaper is transparent until
    /// [`Self::set_bands`] colours it. `max_frames` is clamped up to at least
    /// one frame so the internal buffers are always non-empty.
    #[must_use]
    pub fn new(sample_rate: u32, layout: ChannelLayout, max_frames: usize) -> Self {
        let capacity = max_frames.max(1);
        let crossover =
            LinkwitzRileyCrossover::new(sample_rate, layout, &PROPAGATION_BAND_EDGES, capacity);
        let mut bands = Vec::with_capacity(PROPAGATION_BAND_COUNT);
        for _ in 0..PROPAGATION_BAND_COUNT {
            bands.push(AudioBuffer::new(layout, capacity));
        }
        let gains = core::array::from_fn(|_| Smoothed::new(1.0));
        Self {
            crossover,
            bands,
            gains,
        }
    }

    /// Number of frequency bands the shaper applies (always
    /// [`PROPAGATION_BAND_COUNT`]).
    #[inline]
    #[must_use]
    pub fn num_bands(&self) -> usize {
        PROPAGATION_BAND_COUNT
    }

    /// Channel count the shaper's filterbank was configured for.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.crossover.channels()
    }

    /// Retargets the per-band gains toward `target` using the given ramp shape.
    ///
    /// Each band's [`Smoothed`] gain glides toward the matching value in
    /// `target.bands()` so a changed spectrum does not click. Pass
    /// [`Ramp::Immediate`] (or use [`Self::set_bands_immediate`]) to snap.
    pub fn set_bands(&mut self, target: BandGains, ramp: Ramp) {
        let targets = target.bands();
        for (gain, &value) in self.gains.iter_mut().zip(targets.iter()) {
            gain.set_target(value, ramp);
        }
    }

    /// Snaps the per-band gains to `target` with no glide.
    ///
    /// Convenience for the common case of seeding the shaper when a voice first
    /// acquires a path, where a ramp would be an audible artefact rather than a
    /// smoothing.
    pub fn set_bands_immediate(&mut self, target: BandGains) {
        self.set_bands(target, Ramp::Immediate);
    }

    /// Returns the per-band gains as they are *right now* (mid-glide).
    #[inline]
    #[must_use]
    pub fn current_bands(&self) -> [Sample; PROPAGATION_BAND_COUNT] {
        core::array::from_fn(|i| self.gains[i].current())
    }

    /// Returns the per-band gains the shaper is gliding *toward*.
    #[inline]
    #[must_use]
    pub fn target_bands(&self) -> [Sample; PROPAGATION_BAND_COUNT] {
        core::array::from_fn(|i| self.gains[i].target())
    }

    /// Splits `input` into bands, applies the smoothed per-band gains, and sums
    /// the result into `output`.
    ///
    /// The output's active frame count is set to match the processable block
    /// length (the input length, clamped to the pre-allocated capacity and to
    /// the output's own capacity). The output's active region is overwritten in
    /// full -- the shaper does not add to pre-existing output content. Channels
    /// beyond the smaller of the input and output channel counts are left
    /// silent.
    ///
    /// Each band's gain smoother advances exactly one step per frame, so a
    /// single call consumes `frames` of every band's glide and all bands stay
    /// aligned in time.
    pub fn process_block(&mut self, input: &AudioBuffer, output: &mut AudioBuffer) {
        let band_capacity = self.bands.first().map_or(0, AudioBuffer::capacity_frames);
        let request = input.active_frames().min(band_capacity);
        output.set_active_frames(request);
        let frames = output.active_frames();

        let out_channels = output.channels();
        for ch in 0..out_channels {
            for sample in &mut output.channel_mut(ch)[..frames] {
                *sample = 0.0;
            }
        }
        if frames == 0 {
            // Still advance the crossover so its filter memory tracks the
            // silent block, matching how it would see a zero-length input.
            self.crossover.process_block(input, &mut self.bands);
            return;
        }

        self.crossover.process_block(input, &mut self.bands);
        let consumed = self.crossover.num_bands().min(PROPAGATION_BAND_COUNT);

        let bands = &self.bands;
        let gains = &mut self.gains;
        for (b, gain) in gains.iter_mut().enumerate() {
            if b < consumed {
                let band = &bands[b];
                let channels = out_channels.min(band.channels());
                for f in 0..frames {
                    let g = gain.next_sample();
                    for ch in 0..channels {
                        output.channel_mut(ch)[f] += band.channel(ch)[f] * g;
                    }
                }
            } else {
                // A band the crossover did not produce: advance its smoother
                // anyway so every band's glide stays phase-aligned in time.
                for _ in 0..frames {
                    let _ = gain.next_sample();
                }
            }
        }
    }

    /// Clears the filterbank memory and scratch buffers and settles every band
    /// gain at its current target.
    ///
    /// Called when a voice is recycled: the smoothed gains keep their targets
    /// (the spectrum a reused voice should resume at) but their glide state is
    /// collapsed so the next block starts from the settled value rather than
    /// mid-ramp.
    pub fn reset(&mut self) {
        self.crossover.reset();
        for band in &mut self.bands {
            band.clear();
        }
        for gain in &mut self.gains {
            let target = gain.target();
            gain.set_target(target, Ramp::Immediate);
        }
    }
}

impl AudioNode for BandedPropagationShaper {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        self.process_block(input, output);
    }

    fn reset(&mut self) {
        BandedPropagationShaper::reset(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    const SR: u32 = 48_000;
    const FRAMES: usize = 4096;

    fn fill_sine(buf: &mut AudioBuffer, freq: Sample) {
        buf.set_active_frames(buf.capacity_frames());
        let step = 2.0 * core::f32::consts::PI * freq / SR as Sample;
        let channels = buf.channels();
        for ch in 0..channels {
            let data = buf.channel_mut(ch);
            let mut phase = 0.0;
            for d in data.iter_mut() {
                *d = ops::sin(phase);
                phase += step;
            }
        }
    }

    fn tail_rms(buf: &AudioBuffer, channel: usize, skip: usize) -> Sample {
        let data = buf.channel(channel);
        let start = skip.min(data.len());
        let slice = &data[start..];
        if slice.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0;
        for &s in slice {
            acc += s * s;
        }
        ops::sqrt(acc / slice.len() as Sample)
    }

    fn mono(freq: Sample) -> AudioBuffer {
        let mut input = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        fill_sine(&mut input, freq);
        input
    }

    #[test]
    fn fresh_shaper_is_transparent_in_magnitude() {
        // With the default unity spectrum the band sum is magnitude flat, so a
        // steady tone comes out at the same level it went in, at every band.
        for &freq in &[80.0, 400.0, 1_000.0, 4_000.0, 12_000.0] {
            let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
            let input = mono(freq);
            let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
            shaper.process_block(&input, &mut output);
            let skip = FRAMES / 2;
            let ratio = tail_rms(&output, 0, skip) / tail_rms(&input, 0, skip);
            assert!(
                (ratio - 1.0).abs() < 0.1,
                "unity shaper should preserve magnitude at {freq} Hz (ratio {ratio})"
            );
        }
    }

    #[test]
    fn low_pass_colour_keeps_lows_and_rejects_highs() {
        // A [1, 0, 0] spectrum passes the low band and silences mid and high.
        let colour = BandGains::new([1.0, 0.0, 0.0]);
        let skip = FRAMES / 2;

        let mut low = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        low.set_bands_immediate(colour);
        let low_in = mono(100.0);
        let mut low_out = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        low.process_block(&low_in, &mut low_out);
        let low_ratio = tail_rms(&low_out, 0, skip) / tail_rms(&low_in, 0, skip);
        assert!(
            low_ratio > 0.9,
            "low band tone should pass almost untouched (ratio {low_ratio})"
        );

        let mut high = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        high.set_bands_immediate(colour);
        let high_in = mono(12_000.0);
        let mut high_out = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        high.process_block(&high_in, &mut high_out);
        let high_ratio = tail_rms(&high_out, 0, skip) / tail_rms(&high_in, 0, skip);
        assert!(
            high_ratio < 0.05,
            "high band tone should be rejected by a low-pass colour (ratio {high_ratio})"
        );
    }

    #[test]
    fn mid_band_isolation_matches_gains() {
        // A [0, 1, 0] spectrum keeps only the mid band.
        let colour = BandGains::new([0.0, 1.0, 0.0]);
        let skip = FRAMES / 2;
        let probe = |freq: Sample| -> Sample {
            let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
            shaper.set_bands_immediate(colour);
            let input = mono(freq);
            let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
            shaper.process_block(&input, &mut output);
            tail_rms(&output, 0, skip) / tail_rms(&input, 0, skip)
        };
        assert!(probe(100.0) < 0.1, "low tone rejected by mid-only colour");
        assert!(probe(2_000.0) > 0.85, "mid tone kept by mid-only colour");
        assert!(
            probe(14_000.0) < 0.1,
            "high tone rejected by mid-only colour"
        );
    }

    #[test]
    fn silent_spectrum_silences_output() {
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        shaper.set_bands_immediate(BandGains::SILENT);
        let input = mono(1_000.0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        shaper.process_block(&input, &mut output);
        assert!(
            tail_rms(&output, 0, 0) < 1.0e-6,
            "a silent spectrum must mute every band"
        );
    }

    #[test]
    fn uniform_gain_scales_level() {
        // A flat 0.5 spectrum halves the magnitude everywhere.
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        shaper.set_bands_immediate(BandGains::uniform(0.5));
        let input = mono(1_000.0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        shaper.process_block(&input, &mut output);
        let skip = FRAMES / 2;
        let ratio = tail_rms(&output, 0, skip) / tail_rms(&input, 0, skip);
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "uniform 0.5 spectrum should halve the level (ratio {ratio})"
        );
    }

    #[test]
    fn set_bands_immediate_settles_gains() {
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        let colour = BandGains::new([0.25, 0.5, 0.75]);
        shaper.set_bands_immediate(colour);
        assert_eq!(shaper.current_bands(), colour.bands());
        assert_eq!(shaper.target_bands(), colour.bands());
    }

    #[test]
    fn ramp_glides_current_toward_target() {
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        shaper.set_bands(
            BandGains::SILENT,
            Ramp::Linear {
                samples: FRAMES as u32,
            },
        );
        // Before processing the current value is still the settled unity start.
        assert!((shaper.current_bands()[0] - 1.0).abs() < 1.0e-6);
        let input = mono(1_000.0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        shaper.process_block(&input, &mut output);
        // After one full-length block the glide has reached (or passed into)
        // the silent target.
        for g in shaper.current_bands() {
            assert!(g.abs() < 1.0e-3, "ramp should have reached the target");
        }
        // The output starts loud (near unity) and ends quiet (near silent): the
        // tail is far weaker than the head, proving a smooth glide.
        let head = tail_rms_window(&output, 0, 0, FRAMES / 8);
        let tail = tail_rms_window(&output, 0, 7 * FRAMES / 8, FRAMES);
        assert!(head > tail * 4.0, "glide should fade the signal out");
    }

    fn tail_rms_window(buf: &AudioBuffer, channel: usize, start: usize, end: usize) -> Sample {
        let data = buf.channel(channel);
        let lo = start.min(data.len());
        let hi = end.min(data.len());
        if hi <= lo {
            return 0.0;
        }
        let slice = &data[lo..hi];
        let mut acc = 0.0;
        for &s in slice {
            acc += s * s;
        }
        ops::sqrt(acc / slice.len() as Sample)
    }

    #[test]
    fn output_active_region_is_overwritten_not_summed() {
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        shaper.set_bands_immediate(BandGains::SILENT);
        let input = mono(1_000.0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        output.set_active_frames(FRAMES);
        for s in output.channel_mut(0) {
            *s = 7.0;
        }
        shaper.process_block(&input, &mut output);
        // A silent spectrum over pre-filled garbage must leave silence, proving
        // the active region was overwritten rather than accumulated onto.
        assert!(tail_rms(&output, 0, 0) < 1.0e-6);
    }

    #[test]
    fn reset_clears_filter_tail() {
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        let input = mono(1_000.0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        shaper.process_block(&input, &mut output);
        shaper.reset();
        let silence = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        let mut after = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        shaper.process_block(&silence, &mut after);
        for &s in after.channel(0) {
            assert!(s.abs() < 1.0e-6, "reset must clear the filter memory");
        }
    }

    #[test]
    fn reset_preserves_targets_and_settles() {
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        let colour = BandGains::new([0.2, 0.4, 0.6]);
        shaper.set_bands(colour, Ramp::Linear { samples: 1_000 });
        shaper.reset();
        assert_eq!(shaper.target_bands(), colour.bands());
        assert_eq!(shaper.current_bands(), colour.bands());
    }

    #[test]
    fn stereo_channels_are_shaped_independently() {
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Stereo, FRAMES);
        shaper.set_bands_immediate(BandGains::new([1.0, 0.0, 0.0]));
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, FRAMES);
        input.set_active_frames(FRAMES);
        let step_lo = 2.0 * core::f32::consts::PI * 100.0 / SR as Sample;
        let step_hi = 2.0 * core::f32::consts::PI * 12_000.0 / SR as Sample;
        let mut p0 = 0.0;
        let mut p1 = 0.0;
        for i in 0..FRAMES {
            input.channel_mut(0)[i] = ops::sin(p0);
            p0 += step_lo;
        }
        for i in 0..FRAMES {
            input.channel_mut(1)[i] = ops::sin(p1);
            p1 += step_hi;
        }
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, FRAMES);
        shaper.process_block(&input, &mut output);
        let skip = FRAMES / 2;
        let low_ratio = tail_rms(&output, 0, skip) / tail_rms(&input, 0, skip);
        let high_ratio = tail_rms(&output, 1, skip) / tail_rms(&input, 1, skip);
        assert!(low_ratio > 0.9, "left low tone kept (ratio {low_ratio})");
        assert!(
            high_ratio < 0.05,
            "right high tone rejected (ratio {high_ratio})"
        );
    }

    #[test]
    fn audionode_matches_direct_process_block() {
        let colour = BandGains::new([0.3, 0.7, 0.1]);
        let input = mono(1_000.0);

        let mut direct = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        direct.set_bands_immediate(colour);
        let mut direct_out = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        direct.process_block(&input, &mut direct_out);

        let mut node = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        node.set_bands_immediate(colour);
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, FRAMES)];
        let ctx = RenderContext {
            sample_rate: SR,
            frames: FRAMES,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);

        for (a, b) in direct_out.channel(0).iter().zip(outputs[0].channel(0)) {
            assert!(
                (a - b).abs() < 1.0e-9,
                "AudioNode path must match process_block"
            );
        }
    }

    #[test]
    fn zero_frames_is_a_no_op() {
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        input.set_active_frames(0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        output.set_active_frames(FRAMES);
        shaper.process_block(&input, &mut output);
        assert_eq!(output.active_frames(), 0);
    }

    #[test]
    fn block_longer_than_capacity_is_clamped() {
        let cap = 256;
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, cap);
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 1_024);
        fill_sine(&mut input, 1_000.0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 1_024);
        shaper.process_block(&input, &mut output);
        assert_eq!(output.active_frames(), cap, "processing clamps to capacity");
    }

    #[test]
    fn getters_report_configuration() {
        let shaper = BandedPropagationShaper::new(SR, ChannelLayout::Stereo, FRAMES);
        assert_eq!(shaper.num_bands(), PROPAGATION_BAND_COUNT);
        assert_eq!(shaper.channels(), 2);
    }

    #[test]
    fn output_is_always_finite() {
        let mut shaper = BandedPropagationShaper::new(SR, ChannelLayout::Mono, FRAMES);
        shaper.set_bands_immediate(BandGains::new([0.9, 0.3, 0.6]));
        let input = mono(1_000.0);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, FRAMES);
        shaper.process_block(&input, &mut output);
        for &s in output.channel(0) {
            assert!(s.is_finite(), "outputs must stay finite");
        }
    }
}
