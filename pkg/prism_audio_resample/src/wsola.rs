//! WSOLA / SOLA time-domain time-scale modification (voice / SFX grade).
//!
//! Waveform-Similarity Overlap-Add (WSOLA) changes a signal's duration by
//! copying overlapping analysis frames to a different synthesis rate and
//! overlap-adding them with a Hann window. The "waveform similarity" step is
//! what distinguishes it from plain SOLA: before copying each frame, a
//! normalized cross-correlation search within a small tolerance window aligns
//! the new frame to the natural continuation of the previously emitted audio,
//! which keeps the waveform phase-coherent and avoids the warble that pure
//! overlap-add produces. It is cheap, robust on speech and sound effects, and
//! preserves pitch exactly while changing length.
//!
//! Pitch shifting is decoupled from time scaling by composition: the WSOLA core
//! stretches by `time_stretch * pitch` and an embedded windowed-sinc resampler
//! then rescales by `1 / pitch`, which restores the requested duration while
//! transposing the pitch. With a unity pitch ratio the resampler is bypassed.
//!
//! # Provenance
//!
//! Original work; contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code, and no AI/ML.
//! Classic DSP only. WSOLA is the publicly documented Verhelst-Roelands
//! overlap-add time-scale algorithm; the Hann window and normalized
//! cross-correlation are textbook. All math routes through [`bevy_math::ops`].
//!
//! # Relationship
//!
//! Downstream of [`prism_audio_core`], [`crate::fractional_delay`], and
//! [`crate::polyphase_sinc`] (embedded for the pitch stage). Implements
//! [`crate::time_stretcher::TimeStretcher`] as the low-overhead counterpart to
//! the music-grade [`crate::phase_vocoder::PhaseVocoderStretcher`].

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::TAU;

use prism_audio_core::math::Sample;

use crate::fractional_delay::{EPS, close, sanitize};
use crate::polyphase_sinc::PolyphaseSincResampler;
use crate::resampler::Resampler;
use crate::time_stretcher::{
    SampleFifo, StretchProgress, TimeStretcher, clamp_pitch, clamp_stretch,
};

/// Analysis / synthesis frame length in samples.
const FRAME: usize = 1024;
/// Synthesis hop in samples (output advance per frame).
const HOP: usize = 256;
/// Cross-correlation search tolerance (samples) each side of the ideal grab.
const DELTA: usize = 96;
/// Overlap region length used for OLA and correlation (`FRAME - HOP`).
const OVERLAP: usize = FRAME - HOP;

/// Capacity of the input FIFO (covers the worst-case analysis reach).
const IN_CAP: usize = 8192;
/// Capacity of the stretched staging FIFO (pre-pitch).
const STAGE_CAP: usize = 4096;
/// Capacity of the final output FIFO.
const OUT_CAP: usize = 8192;
/// Scratch length for the embedded pitch resampler.
const RS_SCRATCH: usize = 512;

/// WSOLA/SOLA time-domain time/pitch scaler.
#[derive(Clone, Debug)]
pub struct WsolaStretcher {
    /// Time-stretch factor (`output / input` duration).
    stretch: Sample,
    /// Pitch-shift ratio (`output / input` frequency).
    pitch: Sample,
    /// Hann analysis window, length [`FRAME`].
    win: Vec<Sample>,
    /// Overlap-add accumulator, length [`FRAME`].
    acc: Vec<Sample>,
    /// Window-weight accumulator for per-sample normalization, length [`FRAME`].
    wsum: Vec<Sample>,
    /// Precomputed correlation template, length [`OVERLAP`].
    tmpl: Vec<Sample>,
    /// Input FIFO addressed by absolute stream index.
    infifo: SampleFifo,
    /// Stretched (pre-pitch) staging FIFO.
    stage: SampleFifo,
    /// Final output FIFO.
    outfifo: SampleFifo,
    /// Absolute input index of the previously grabbed frame.
    grab_prev: usize,
    /// Drift-free ideal analysis position (fractional absolute input index).
    ideal: Sample,
    /// Whether the first frame has been emitted.
    started: bool,
    /// Embedded resampler for the decoupled pitch stage.
    pitch_rs: PolyphaseSincResampler,
    /// Input scratch for the pitch resampler.
    rs_in: Vec<Sample>,
    /// Output scratch for the pitch resampler.
    rs_out: Vec<Sample>,
}

impl WsolaStretcher {
    /// Builds a WSOLA stretcher at unity time and pitch.
    #[must_use]
    pub fn new() -> Self {
        let win: Vec<Sample> = (0..FRAME)
            .map(|k| 0.5 - 0.5 * ops::cos(TAU * k as Sample / FRAME as Sample))
            .collect();
        let mut pitch_rs = PolyphaseSincResampler::new();
        pitch_rs.set_ratio(1.0);
        Self {
            stretch: 1.0,
            pitch: 1.0,
            win,
            acc: vec![0.0; FRAME],
            wsum: vec![0.0; FRAME],
            tmpl: vec![0.0; OVERLAP],
            infifo: SampleFifo::with_capacity(IN_CAP),
            stage: SampleFifo::with_capacity(STAGE_CAP),
            outfifo: SampleFifo::with_capacity(OUT_CAP),
            grab_prev: 0,
            ideal: 0.0,
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

    /// Overlap-adds the Hann-windowed frame starting at absolute index `grab`.
    fn overlap_add(&mut self, grab: usize) {
        let mut k = 0;
        while k < FRAME {
            let s = self.infifo.at(grab + k);
            self.acc[k] += self.win[k] * s;
            self.wsum[k] += self.win[k];
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

    /// Finds the absolute grab index in `[lo, hi]` whose frame onset best
    /// correlates with the current template.
    fn best_match(&self, lo: usize, hi: usize) -> usize {
        // Precompute the template energy.
        let mut tnorm = 0.0;
        let mut k = 0;
        while k < OVERLAP {
            tnorm += self.tmpl[k] * self.tmpl[k];
            k += 1;
        }
        let mut best_g = lo;
        let mut best_cc = Sample::NEG_INFINITY;
        let mut g = lo;
        while g <= hi {
            let mut dot = 0.0;
            let mut cnorm = 0.0;
            let mut k = 0;
            while k < OVERLAP {
                let c = self.infifo.at(g + k);
                dot += c * self.tmpl[k];
                cnorm += c * c;
                k += 1;
            }
            let denom = ops::sqrt(cnorm * tnorm) + EPS;
            let cc = dot / denom;
            if cc > best_cc {
                best_cc = cc;
                best_g = g;
            }
            g += 1;
        }
        best_g
    }

    /// Produces as many stretched frames as the buffered input and staging room
    /// allow.
    ///
    /// The ideal analysis position advances by a drift-free fractional hop
    /// `HOP / alpha` every synthesis frame; the normalized cross-correlation
    /// search only nudges each grab by up to [`DELTA`] samples to keep the
    /// overlap-add waveform-coherent. Advancing the ideal pointer independently
    /// of the matched grab is what keeps the realized stretch ratio equal to
    /// the requested factor instead of letting correlation offsets accumulate.
    fn generate(&mut self) {
        loop {
            if self.stage.free() < HOP {
                break;
            }
            let step = HOP as Sample / self.alpha();
            let grab;
            if !self.started {
                if self.infifo.end() < FRAME {
                    break;
                }
                grab = self.infifo.base();
                self.started = true;
                self.ideal = grab as Sample + step;
            } else {
                let expected = (self.ideal + 0.5) as usize;
                let hi = expected + DELTA;
                let tmpl_end = self.grab_prev + HOP + OVERLAP;
                let required_end = (hi + FRAME).max(tmpl_end);
                if self.infifo.end() < required_end {
                    break;
                }
                // Fill the correlation template from the natural continuation of
                // the previously grabbed frame (its content shifted by one hop).
                let mut k = 0;
                while k < OVERLAP {
                    self.tmpl[k] = self.infifo.at(self.grab_prev + HOP + k);
                    k += 1;
                }
                let lo = expected.saturating_sub(DELTA).max(self.infifo.base());
                grab = self.best_match(lo, hi);
                self.ideal += step;
            }
            self.overlap_add(grab);
            self.emit_hop();
            self.grab_prev = grab;
            // Retain everything the next template and search can still reference:
            // the next search starts at `round(ideal) - DELTA`, and the next
            // template reads from `grab + HOP`.
            let next_lo = ((self.ideal + 0.5) as isize) - DELTA as isize;
            let keep_i = core::cmp::min(next_lo, grab as isize);
            let keep = if keep_i > 0 { keep_i as usize } else { 0 };
            self.infifo.drop_until(keep);
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

impl Default for WsolaStretcher {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeStretcher for WsolaStretcher {
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
        self.infifo.clear();
        self.stage.clear();
        self.outfifo.clear();
        self.grab_prev = 0;
        self.ideal = 0.0;
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

    fn drive(st: &mut WsolaStretcher, input: &[Sample]) -> Vec<Sample> {
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
                // Allow the internal pipeline to flush with empty input.
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
        let mut st = WsolaStretcher::new();
        st.set_time_stretch(1.5);
        let out = drive(&mut st, &input);
        // Length grows by ~1.5x (generous slack for frame edges / warm-up).
        let ratio = out.len() as Sample / input.len() as Sample;
        assert!(ratio > 1.3 && ratio < 1.7, "ratio = {ratio}");
        // Pitch (period) is preserved.
        let p_in = dominant_period(&input, 50, 200);
        let p_out = dominant_period(&out[2048..out.len() - 2048], 50, 200);
        assert!(
            (p_in as isize - p_out as isize).abs() <= 2,
            "p_in={p_in} p_out={p_out}"
        );
    }

    #[test]
    fn compression_shortens_length() {
        let rate = 48_000.0;
        let input = sine(300.0, rate, 24_000);
        let mut st = WsolaStretcher::new();
        st.set_time_stretch(0.5);
        let out = drive(&mut st, &input);
        let ratio = out.len() as Sample / input.len() as Sample;
        assert!(ratio > 0.35 && ratio < 0.65, "ratio = {ratio}");
    }

    #[test]
    fn determinism_two_instances_match() {
        let input = sine(440.0, 48_000.0, 12_000);
        let mut a = WsolaStretcher::new();
        let mut b = WsolaStretcher::new();
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
        let mut st = WsolaStretcher::new();
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
        let mut st = WsolaStretcher::new();
        st.set_time_stretch(1.0);
        st.set_pitch_shift(2.0); // up an octave: period halves, length ~ equal.
        let out = drive(&mut st, &input);
        let ratio = out.len() as Sample / input.len() as Sample;
        assert!(ratio > 0.8 && ratio < 1.2, "len ratio = {ratio}");
        let p_in = dominant_period(&input, 40, 400);
        let p_out = dominant_period(&out[4096..out.len() - 4096], 40, 400);
        // Period should roughly halve.
        let expect = p_in as Sample / 2.0;
        assert!(
            ops::abs(p_out as Sample - expect) < 12.0,
            "p_in={p_in} p_out={p_out}"
        );
    }

    #[test]
    fn non_finite_input_stays_finite() {
        let rate = 48_000.0;
        let mut input = sine(440.0, rate, 8_000);
        input[1000] = Sample::NAN;
        input[2000] = Sample::INFINITY;
        let mut st = WsolaStretcher::new();
        st.set_time_stretch(1.5);
        let out = drive(&mut st, &input);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
