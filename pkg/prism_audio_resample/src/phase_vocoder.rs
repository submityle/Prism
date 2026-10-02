//! Phase-vocoder frequency-domain time-scale modification (music grade).
//!
//! A phase vocoder changes a signal's duration by resynthesizing its
//! short-time Fourier transform (STFT) at a synthesis hop that differs from the
//! analysis hop. For every analysis frame the magnitude spectrum is kept intact
//! while the per-bin phase is advanced by the bin's measured instantaneous
//! frequency times the synthesis hop; this preserves horizontal (across-time)
//! phase coherence so steady tones stretch cleanly without the granular warble
//! of plain overlap-add. It is the high-fidelity counterpart to the
//! time-domain [`crate::wsola::WsolaStretcher`]: more expensive (an FFT per
//! frame) but smoother on tonal and musical material.
//!
//! Pitch shifting is decoupled from time scaling by composition exactly as in
//! WSOLA: the vocoder core stretches by `time_stretch * pitch` and an embedded
//! windowed-sinc resampler then rescales by `1 / pitch`, restoring the
//! requested duration while transposing the pitch. With a unity pitch ratio the
//! resampler is bypassed.
//!
//! # Provenance
//!
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code, and no AI/ML.
//! Classic DSP only. The phase vocoder is the publicly documented
//! Flanagan-Golden / Portnoff STFT time-scaling method; the Hann window,
//! instantaneous-frequency phase unwrapping, and weighted overlap-add
//! reconstruction are textbook (for example Dolson's "The Phase Vocoder: A
//! Tutorial" and Zoelzer's DAFX). Principal-argument phase wrapping is computed
//! as `atan2(sin(d), cos(d))` and every transcendental routes through
//! [`bevy_math::ops`], so the output is bit-reproducible across platforms.
//!
//! # Relationship
//!
//! Downstream of [`prism_audio_core`] (reuses its [`Fft`] and [`Sample`]),
//! [`crate::fractional_delay`], and [`crate::polyphase_sinc`] (embedded for the
//! pitch stage). Implements [`crate::time_stretcher::TimeStretcher`] as the
//! music-grade counterpart to [`crate::wsola::WsolaStretcher`]; both share the
//! [`crate::time_stretcher::SampleFifo`] streaming plumbing and the same
//! decoupled-pitch composition.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::TAU;

use prism_audio_core::fft::Fft;
use prism_audio_core::math::Sample;

use crate::fractional_delay::{EPS, close, sanitize};
use crate::polyphase_sinc::PolyphaseSincResampler;
use crate::resampler::Resampler;
use crate::time_stretcher::{
    SampleFifo, StretchProgress, TimeStretcher, clamp_pitch, clamp_stretch,
};

/// STFT analysis / synthesis frame length in samples (a power of two).
const FRAME: usize = 1024;
/// Synthesis hop in samples (output advance per frame): 75% overlap.
const HOP: usize = 256;
/// Number of non-redundant spectral bins (`FRAME / 2 + 1`).
const NBINS: usize = FRAME / 2 + 1;

/// Capacity of the input FIFO (covers the worst-case analysis reach).
const IN_CAP: usize = 16_384;
/// Capacity of the stretched staging FIFO (pre-pitch).
const STAGE_CAP: usize = 8192;
/// Capacity of the final output FIFO.
const OUT_CAP: usize = 8192;
/// Scratch length for the embedded pitch resampler.
const RS_SCRATCH: usize = 512;

/// Phase-vocoder frequency-domain time/pitch scaler.
#[derive(Clone)]
pub struct PhaseVocoderStretcher {
    /// Time-stretch factor (`output / input` duration).
    stretch: Sample,
    /// Pitch-shift ratio (`output / input` frequency).
    pitch: Sample,
    /// Shared FFT plan of length [`FRAME`].
    fft: Fft,
    /// Hann analysis / synthesis window, length [`FRAME`].
    win: Vec<Sample>,
    /// Real transform scratch, length [`FRAME`].
    re: Vec<Sample>,
    /// Imaginary transform scratch, length [`FRAME`].
    im: Vec<Sample>,
    /// Analysis phase of the previous frame per bin, length [`NBINS`].
    last_phase: Vec<Sample>,
    /// Accumulated synthesis phase per bin, length [`NBINS`].
    sum_phase: Vec<Sample>,
    /// Overlap-add accumulator, length [`FRAME`].
    acc: Vec<Sample>,
    /// Window-weight accumulator for per-sample normalization, length [`FRAME`].
    wsum: Vec<Sample>,
    /// Input FIFO addressed by absolute stream index.
    infifo: SampleFifo,
    /// Stretched (pre-pitch) staging FIFO.
    stage: SampleFifo,
    /// Final output FIFO.
    outfifo: SampleFifo,
    /// Absolute input index of the previously analyzed frame.
    grab_prev: usize,
    /// Whether the first frame has been emitted.
    started: bool,
    /// Embedded resampler for the decoupled pitch stage.
    pitch_rs: PolyphaseSincResampler,
    /// Input scratch for the pitch resampler.
    rs_in: Vec<Sample>,
    /// Output scratch for the pitch resampler.
    rs_out: Vec<Sample>,
}

impl PhaseVocoderStretcher {
    /// Builds a phase vocoder at unity time and pitch.
    #[must_use]
    pub fn new() -> Self {
        let fft = Fft::new(FRAME);
        let win: Vec<Sample> = (0..FRAME)
            .map(|k| 0.5 - 0.5 * ops::cos(TAU * k as Sample / FRAME as Sample))
            .collect();
        let mut pitch_rs = PolyphaseSincResampler::new();
        pitch_rs.set_ratio(1.0);
        Self {
            stretch: 1.0,
            pitch: 1.0,
            fft,
            win,
            re: vec![0.0; FRAME],
            im: vec![0.0; FRAME],
            last_phase: vec![0.0; NBINS],
            sum_phase: vec![0.0; NBINS],
            acc: vec![0.0; FRAME],
            wsum: vec![0.0; FRAME],
            infifo: SampleFifo::with_capacity(IN_CAP),
            stage: SampleFifo::with_capacity(STAGE_CAP),
            outfifo: SampleFifo::with_capacity(OUT_CAP),
            grab_prev: 0,
            started: false,
            pitch_rs,
            rs_in: vec![0.0; RS_SCRATCH],
            rs_out: vec![0.0; RS_SCRATCH],
        }
    }

    /// The combined internal stretch factor (`time_stretch * pitch`).
    #[inline]
    #[must_use]
    fn alpha(&self) -> Sample {
        (self.stretch * self.pitch).clamp(0.0625, 16.0)
    }

    /// Analysis hop derived from the internal stretch factor (at least `1`).
    #[inline]
    #[must_use]
    fn analysis_hop(&self) -> usize {
        let ha = (HOP as Sample / self.alpha()) + 0.5;
        (ha as usize).max(1)
    }

    /// Principal argument of `x`, wrapped into `[-pi, pi]` deterministically.
    #[inline]
    #[must_use]
    fn princ(x: Sample) -> Sample {
        ops::atan2(ops::sin(x), ops::cos(x))
    }

    /// Analyzes the windowed frame at absolute index `grab`, advances the
    /// per-bin synthesis phase by the measured instantaneous frequency, and
    /// overlap-adds the resynthesized frame into the accumulator. `ha` is the
    /// analysis hop that separated this frame from the previous one; `first`
    /// requests an identity phase initialization for the opening frame.
    fn analyze_synthesize(&mut self, grab: usize, ha: usize, first: bool) {
        // Window the input frame into the real buffer; clear the imaginary part.
        let mut k = 0;
        while k < FRAME {
            self.re[k] = self.win[k] * self.infifo.at(grab + k);
            self.im[k] = 0.0;
            k += 1;
        }
        self.fft.forward(&mut self.re, &mut self.im);

        let ha_f = ha as Sample;
        let mut bin = 0;
        while bin < NBINS {
            let real = self.re[bin];
            let imag = self.im[bin];
            let mag = ops::hypot(real, imag);
            let phase = ops::atan2(imag, real);
            // Expected (idealized) phase advance over the analysis hop.
            let omega = TAU * bin as Sample / FRAME as Sample;
            if first {
                // Identity initialization: reproduce the opening frame exactly.
                self.sum_phase[bin] = phase;
            } else {
                let expected = omega * ha_f;
                let delta = Self::princ(phase - self.last_phase[bin] - expected);
                // Instantaneous angular frequency per sample for this bin.
                let inst = omega + delta / ha_f;
                self.sum_phase[bin] = Self::princ(self.sum_phase[bin] + inst * HOP as Sample);
            }
            self.last_phase[bin] = phase;
            let out_phase = self.sum_phase[bin];
            self.re[bin] = mag * ops::cos(out_phase);
            self.im[bin] = mag * ops::sin(out_phase);
            bin += 1;
        }

        // Enforce Hermitian symmetry so the inverse transform is purely real.
        let half = FRAME / 2;
        self.im[0] = 0.0;
        self.im[half] = 0.0;
        let mut k = 1;
        while k < half {
            self.re[FRAME - k] = self.re[k];
            self.im[FRAME - k] = -self.im[k];
            k += 1;
        }

        self.fft.inverse(&mut self.re, &mut self.im);

        // Weighted overlap-add with the synthesis window.
        let mut k = 0;
        while k < FRAME {
            let w = self.win[k];
            self.acc[k] += w * self.re[k];
            self.wsum[k] += w * w;
            k += 1;
        }
    }

    /// Emits the leading [`HOP`] completed samples into the staging FIFO and
    /// slides the accumulators left by [`HOP`].
    fn emit_hop(&mut self) {
        let mut k = 0;
        while k < HOP {
            let w = self.wsum[k];
            let y = if w > EPS { self.acc[k] / w } else { 0.0 };
            let _ = self.stage.push(y);
            k += 1;
        }
        self.acc.copy_within(HOP.., 0);
        self.wsum.copy_within(HOP.., 0);
        let tail = FRAME - HOP;
        let mut k = tail;
        while k < FRAME {
            self.acc[k] = 0.0;
            self.wsum[k] = 0.0;
            k += 1;
        }
    }

    /// Produces as many resynthesized frames as the buffered input and staging
    /// room allow.
    fn generate(&mut self) {
        loop {
            if self.stage.free() < HOP {
                break;
            }
            let ha = self.analysis_hop();
            let grab;
            let first;
            if !self.started {
                if self.infifo.end() < self.infifo.base() + FRAME {
                    break;
                }
                grab = self.infifo.base();
                first = true;
                self.started = true;
            } else {
                grab = self.grab_prev + ha;
                if self.infifo.end() < grab + FRAME {
                    break;
                }
                first = false;
            }
            let step = grab - self.grab_prev;
            let ha_used = if first { ha } else { step };
            self.analyze_synthesize(grab, ha_used, first);
            self.emit_hop();
            self.grab_prev = grab;
            self.infifo.drop_until(grab);
        }
    }

    /// Moves staged samples into the output FIFO, applying the pitch resampler
    /// when the pitch ratio is not unity.
    fn pitch_stage(&mut self) {
        if close(self.pitch, 1.0) {
            while self.outfifo.free() > 0 {
                match self.stage.pop() {
                    Some(s) => {
                        let _ = self.outfifo.push(s);
                    }
                    None => break,
                }
            }
            return;
        }
        loop {
            if self.outfifo.free() == 0 || self.stage.len() == 0 {
                break;
            }
            let take = core::cmp::min(self.stage.len(), RS_SCRATCH);
            let mut i = 0;
            while i < take {
                self.rs_in[i] = self.stage.at(self.stage.base() + i);
                i += 1;
            }
            let out_room = core::cmp::min(self.outfifo.free(), RS_SCRATCH);
            let prog = self
                .pitch_rs
                .process(&self.rs_in[..take], &mut self.rs_out[..out_room]);
            let mut j = 0;
            while j < prog.produced {
                let _ = self.outfifo.push(self.rs_out[j]);
                j += 1;
            }
            let mut c = 0;
            while c < prog.consumed {
                let _ = self.stage.pop();
                c += 1;
            }
            if prog.produced == 0 && prog.consumed == 0 {
                break;
            }
        }
    }
}

impl Default for PhaseVocoderStretcher {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeStretcher for PhaseVocoderStretcher {
    fn time_stretch(&self) -> Sample {
        self.stretch
    }

    fn pitch_shift(&self) -> Sample {
        self.pitch
    }

    fn set_time_stretch(&mut self, factor: Sample) {
        self.stretch = clamp_stretch(factor);
    }

    fn set_pitch_shift(&mut self, ratio: Sample) {
        self.pitch = clamp_pitch(ratio);
        self.pitch_rs.set_ratio(1.0 / self.pitch);
    }

    fn reset(&mut self) {
        for v in &mut self.acc {
            *v = 0.0;
        }
        for v in &mut self.wsum {
            *v = 0.0;
        }
        for v in &mut self.last_phase {
            *v = 0.0;
        }
        for v in &mut self.sum_phase {
            *v = 0.0;
        }
        self.infifo.clear();
        self.stage.clear();
        self.outfifo.clear();
        self.grab_prev = 0;
        self.started = false;
        self.pitch_rs.reset();
    }

    fn process(&mut self, input: &[Sample], output: &mut [Sample]) -> StretchProgress {
        let mut consumed = 0;
        while consumed < input.len() && self.infifo.free() > 0 {
            let _ = self.infifo.push(sanitize(input[consumed]));
            consumed += 1;
        }
        self.generate();
        self.pitch_stage();
        let mut produced = 0;
        while produced < output.len() {
            match self.outfifo.pop() {
                Some(s) => {
                    output[produced] = s;
                    produced += 1;
                }
                None => break,
            }
        }
        StretchProgress { consumed, produced }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractional_delay::close;

    fn drive(st: &mut PhaseVocoderStretcher, input: &[Sample]) -> Vec<Sample> {
        let mut out = Vec::new();
        let mut scratch = vec![0.0 as Sample; 1024];
        let mut offset = 0;
        let mut idle = 0;
        loop {
            let prog = st.process(&input[offset..], &mut scratch);
            out.extend_from_slice(&scratch[..prog.produced]);
            offset += prog.consumed;
            if prog.consumed == 0 && prog.produced == 0 {
                idle += 1;
                if offset >= input.len() && idle > 4 {
                    break;
                }
                if idle > 64 {
                    break;
                }
            } else {
                idle = 0;
            }
        }
        out
    }

    fn sine(freq: Sample, rate: Sample, n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| ops::sin(TAU * freq * i as Sample / rate))
            .collect()
    }

    /// Estimates the dominant period (in samples) by autocorrelation.
    fn dominant_period(x: &[Sample], min_lag: usize, max_lag: usize) -> usize {
        let mut best_lag = min_lag;
        let mut best = Sample::NEG_INFINITY;
        let mut lag = min_lag;
        while lag <= max_lag {
            let mut acc = 0.0;
            let mut i = 0;
            while i + lag < x.len() {
                acc += x[i] * x[i + lag];
                i += 1;
            }
            if acc > best {
                best = acc;
                best_lag = lag;
            }
            lag += 1;
        }
        best_lag
    }

    #[test]
    fn stretch_changes_length_keeps_pitch() {
        let rate = 48_000.0;
        let input = sine(440.0, rate, 24_000);
        let mut st = PhaseVocoderStretcher::new();
        st.set_time_stretch(1.5);
        let out = drive(&mut st, &input);
        let ratio = out.len() as Sample / input.len() as Sample;
        assert!(ratio > 1.3 && ratio < 1.7, "ratio = {ratio}");
        let p_in = dominant_period(&input, 50, 200);
        let p_out = dominant_period(&out[4096..out.len() - 4096], 50, 200);
        assert!(
            (p_in as isize - p_out as isize).abs() <= 2,
            "p_in={p_in} p_out={p_out}"
        );
    }

    #[test]
    fn compression_shortens_length() {
        let rate = 48_000.0;
        let input = sine(300.0, rate, 24_000);
        let mut st = PhaseVocoderStretcher::new();
        st.set_time_stretch(0.5);
        let out = drive(&mut st, &input);
        let ratio = out.len() as Sample / input.len() as Sample;
        assert!(ratio > 0.35 && ratio < 0.65, "ratio = {ratio}");
    }

    #[test]
    fn determinism_two_instances_match() {
        let input = sine(440.0, 48_000.0, 12_000);
        let mut a = PhaseVocoderStretcher::new();
        let mut b = PhaseVocoderStretcher::new();
        a.set_time_stretch(1.5);
        b.set_time_stretch(1.5);
        let oa = drive(&mut a, &input);
        let ob = drive(&mut b, &input);
        assert_eq!(oa.len(), ob.len());
        for (x, y) in oa.iter().zip(ob.iter()) {
            assert!(close(*x, *y));
        }
    }

    #[test]
    fn reset_reproduces_output() {
        let input = sine(440.0, 48_000.0, 12_000);
        let mut st = PhaseVocoderStretcher::new();
        st.set_time_stretch(1.25);
        let first = drive(&mut st, &input);
        st.reset();
        st.set_time_stretch(1.25);
        let second = drive(&mut st, &input);
        assert_eq!(first.len(), second.len());
        for (x, y) in first.iter().zip(second.iter()) {
            assert!(close(*x, *y));
        }
    }

    #[test]
    fn pitch_shift_changes_period_keeps_length() {
        let rate = 48_000.0;
        let input = sine(300.0, rate, 24_000);
        let mut st = PhaseVocoderStretcher::new();
        st.set_pitch_shift(2.0);
        let out = drive(&mut st, &input);
        // Duration is roughly preserved (pitch is decoupled from time).
        let ratio = out.len() as Sample / input.len() as Sample;
        assert!(ratio > 0.7 && ratio < 1.3, "ratio = {ratio}");
        // Pitch up an octave halves the period.
        let p_in = dominant_period(&input, 40, 320);
        let p_out = dominant_period(&out[4096..out.len() - 4096], 40, 320);
        assert!(p_out < p_in, "p_in={p_in} p_out={p_out}");
    }

    #[test]
    fn silence_stays_silent() {
        let input = vec![0.0 as Sample; 8192];
        let mut st = PhaseVocoderStretcher::new();
        st.set_time_stretch(1.5);
        let out = drive(&mut st, &input);
        for s in &out {
            assert!(close(*s, 0.0));
        }
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let mut input = sine(440.0, 48_000.0, 8192);
        input[100] = Sample::NAN;
        input[200] = Sample::INFINITY;
        let mut st = PhaseVocoderStretcher::new();
        st.set_time_stretch(1.2);
        let out = drive(&mut st, &input);
        for s in &out {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn unity_is_near_passthrough_length() {
        let input = sine(440.0, 48_000.0, 12_000);
        let mut st = PhaseVocoderStretcher::new();
        let out = drive(&mut st, &input);
        let ratio = out.len() as Sample / input.len() as Sample;
        assert!(ratio > 0.9 && ratio < 1.1, "ratio = {ratio}");
    }
}
