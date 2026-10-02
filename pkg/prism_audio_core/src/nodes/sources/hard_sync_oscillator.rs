//! Hard-sync sawtooth oscillator source node.
//!
//! [`HardSyncOscillatorNode`] synthesizes the classic "oscillator sync" timbre:
//! a fast *slave* sawtooth whose phase is forcibly reset to zero every time a
//! slower *master* oscillator completes a cycle. Because the slave is restarted
//! on the master's period, the perceived pitch locks to the master frequency
//! while the slave's own (higher) frequency carves a formant-like resonance into
//! the spectrum. Sweeping the master/slave ratio glides that formant up and down
//! the harmonic series, producing the aggressive, vocal "tearing" sweep heard in
//! hard-sync synth leads.
//!
//! # Model
//!
//! The slave is a band-limited sawtooth. Two distinct discontinuities are
//! corrected each sample:
//!
//! * The slave's **own wrap**: when its normalized phase crosses `1.0` it falls
//!   back toward `0.0`, a downward jump of `-2`. This is rounded with the shared
//!   two-sided [`poly_blep`](super::oscillator::poly_blep) primitive exactly as
//!   [`super::oscillator::OscillatorNode`]'s sawtooth does.
//! * The **forced reset**: when the master phase wraps, the slave phase is
//!   snapped to `0.0`. The reset lands at a fractional sample position `frac`
//!   *before* the current output sample, so the active trajectory at "now" is
//!   already the post-reset ramp. The naive value for the reset sample is
//!   therefore recomputed from the post-reset phase, and the reset step is
//!   rounded with a one-sided (two-point) `PolyBLEP` residual evaluated from the
//!   wrapped master phase. The step height is `-2 * ps_reset`, where `ps_reset`
//!   is the slave phase the old trajectory would have reached at the reset
//!   instant, so a quiet reset (small `ps_reset`) injects a correspondingly
//!   small correction.
//!
//! The slave is fixed to a sawtooth (the richest, most canonical sync shape) so
//! the node has a single, well-defined responsibility. The master is a bare
//! phase accumulator; it is never emitted, only used to trigger the reset.
//!
//! # Determinism
//!
//! The entire node is a per-sample state machine over two phase accumulators and
//! two [`Smoothed`] controls. Given the same construction parameters, sample
//! rate, and buffer sizes it reproduces its output bit-for-bit, and [`reset`]
//! returns it to its exact initial state.
//!
//! [`reset`]: HardSyncOscillatorNode::reset
//!
//! # Relationship
//!
//! Unlike the steady single [`super::oscillator::OscillatorNode`] sawtooth (no
//! moving formant), the detuned, *de-synchronized* stack of
//! [`super::supersaw::SupersawNode`] (whose voices drift apart rather than lock),
//! or the variable-duty [`super::pwm_oscillator::PwmOscillatorNode`] (which moves
//! a second edge within a single cycle), this node keeps a single slave edge but
//! *re-locks* it to the master every cycle; sweeping the ratio moves a sync
//! formant through the spectrum, a timbre none of the others can express. It
//! reuses the shared `PolyBLEP` edge-correction primitive rather than
//! reimplementing it.
//!
//! # Real-time contract
//!
//! All state is pre-computed at construction, so
//! [`HardSyncOscillatorNode::process`] performs no allocation, no locking, and
//! no panicking: it is a pure per-sample state machine. The sync ratio and
//! amplitude are driven through [`Smoothed`] values so a ratio sweep (the
//! signature sync gesture) and gain automation never produce zipper clicks.
//!
//! # Provenance
//!
//! Implemented from first principles from the public, long-documented virtual-
//! analog techniques of oscillator hard sync and `PolyBLEP` band-limited step
//! correction. It contains no code, data, or derivative of Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, Google Resonance Audio, the Web Audio API,
//! the Synthesis Toolkit (STK), or any other audio engine or toolkit; only the
//! shared mathematical ideas are used. There is no AI or machine learning of any
//! kind.

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::Sample;
use crate::nodes::sources::oscillator::poly_blep;
use crate::param::{Ramp, Smoothed};

/// Minimum master (perceived) frequency in hertz.
pub const MIN_MASTER_HZ: Sample = 20.0;

/// Default master (perceived) frequency in hertz.
pub const DEFAULT_MASTER_HZ: Sample = 110.0;

/// Maximum master (perceived) frequency in hertz.
pub const MAX_MASTER_HZ: Sample = 12_000.0;

/// Minimum sync ratio. A ratio of `1.0` means the slave runs at the master
/// frequency, which keeps the hard-sync convention `slave >= master`.
pub const MIN_SYNC_RATIO: Sample = 1.0;

/// Default sync ratio (slave an octave-and-a-fifth above the master).
pub const DEFAULT_SYNC_RATIO: Sample = 1.5;

/// Maximum sync ratio.
pub const MAX_SYNC_RATIO: Sample = 32.0;

/// Default linear output amplitude. Kept below unity so the band-limited reset
/// transient keeps a little headroom.
pub const DEFAULT_AMPLITUDE: Sample = 0.8;

/// Fraction of the sample rate above which a frequency is clamped to stay below
/// Nyquist and keep each phase accumulator to a single wrap per sample.
pub const NYQUIST_GUARD: Sample = 0.49;

/// Replaces a non-finite value with `fallback`, otherwise returns the input.
#[inline]
fn finite_or(value: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Construction parameters for a [`HardSyncOscillatorNode`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HardSyncOscillatorParams {
    /// Master (perceived) frequency in hertz. Clamped to
    /// `[MIN_MASTER_HZ, MAX_MASTER_HZ]`.
    pub master_hz: Sample,
    /// Slave/master frequency ratio. Clamped to `[MIN_SYNC_RATIO, MAX_SYNC_RATIO]`.
    pub sync_ratio: Sample,
    /// Linear output amplitude (a gain multiplier, not decibels).
    pub amplitude: Sample,
}

impl Default for HardSyncOscillatorParams {
    fn default() -> Self {
        Self {
            master_hz: DEFAULT_MASTER_HZ,
            sync_ratio: DEFAULT_SYNC_RATIO,
            amplitude: DEFAULT_AMPLITUDE,
        }
    }
}

impl HardSyncOscillatorParams {
    /// Returns a copy with every field sanitised into its valid domain.
    #[must_use]
    pub fn sanitised(self) -> Self {
        Self {
            master_hz: finite_or(self.master_hz, DEFAULT_MASTER_HZ)
                .clamp(MIN_MASTER_HZ, MAX_MASTER_HZ),
            sync_ratio: finite_or(self.sync_ratio, DEFAULT_SYNC_RATIO)
                .clamp(MIN_SYNC_RATIO, MAX_SYNC_RATIO),
            amplitude: finite_or(self.amplitude, DEFAULT_AMPLITUDE),
        }
    }
}

/// A hard-sync sawtooth oscillator source node (0 inputs, 1 output).
///
/// Every output channel receives the same mono waveform so downstream
/// stereo/surround nodes see a coherent source.
///
/// # Examples
///
/// ```
/// use prism_audio_core::nodes::sources::HardSyncOscillatorNode;
/// use prism_audio_core::param::Ramp;
///
/// let mut node = HardSyncOscillatorNode::new(110.0, 1.5, 0.8);
/// // Glide the sync formant upward for the classic sync sweep.
/// node.set_sync_ratio(4.0, Ramp::Immediate);
/// assert_eq!(node.master_hz(), 110.0);
/// ```
#[derive(Debug, Clone)]
pub struct HardSyncOscillatorNode {
    /// Master (perceived) frequency in hertz. Stored as a plain scalar because
    /// the phase accumulator is continuous, so a frequency change is click-free
    /// without smoothing.
    master_hz: Sample,
    /// Smoothed slave/master frequency ratio.
    sync_ratio: Smoothed,
    /// Smoothed linear output amplitude.
    amplitude: Smoothed,
    /// Normalized master phase accumulator in `[0, 1)`.
    master_phase: Sample,
    /// Normalized slave phase accumulator in `[0, 1)`.
    slave_phase: Sample,
}

impl HardSyncOscillatorNode {
    /// Creates a hard-sync oscillator at `master_hz` with slave/master
    /// `sync_ratio` and master `amplitude`.
    ///
    /// Non-finite inputs fall back to defaults; `master_hz` is clamped to
    /// `[MIN_MASTER_HZ, MAX_MASTER_HZ]` and `sync_ratio` to
    /// `[MIN_SYNC_RATIO, MAX_SYNC_RATIO]`.
    #[must_use]
    pub fn new(master_hz: Sample, sync_ratio: Sample, amplitude: Sample) -> Self {
        Self {
            master_hz: finite_or(master_hz, DEFAULT_MASTER_HZ)
                .clamp(MIN_MASTER_HZ, MAX_MASTER_HZ),
            sync_ratio: Smoothed::new(
                finite_or(sync_ratio, DEFAULT_SYNC_RATIO).clamp(MIN_SYNC_RATIO, MAX_SYNC_RATIO),
            ),
            amplitude: Smoothed::new(finite_or(amplitude, DEFAULT_AMPLITUDE)),
            master_phase: 0.0,
            slave_phase: 0.0,
        }
    }

    /// Builds a hard-sync oscillator from a [`HardSyncOscillatorParams`] bundle.
    #[must_use]
    pub fn from_params(params: HardSyncOscillatorParams) -> Self {
        let p = params.sanitised();
        Self::new(p.master_hz, p.sync_ratio, p.amplitude)
    }

    /// Sets the master frequency in hertz, clamped to
    /// `[MIN_MASTER_HZ, MAX_MASTER_HZ]`.
    ///
    /// Click-free without smoothing because the phase accumulator is
    /// continuous.
    #[inline]
    pub fn set_master_hz(&mut self, hz: Sample) {
        self.master_hz = finite_or(hz, self.master_hz).clamp(MIN_MASTER_HZ, MAX_MASTER_HZ);
    }

    /// Sets a new target sync ratio, gliding with `ramp` so the sync sweep is
    /// click-free. The target is clamped to `[MIN_SYNC_RATIO, MAX_SYNC_RATIO]`.
    #[inline]
    pub fn set_sync_ratio(&mut self, ratio: Sample, ramp: Ramp) {
        let target =
            finite_or(ratio, self.sync_ratio.target()).clamp(MIN_SYNC_RATIO, MAX_SYNC_RATIO);
        self.sync_ratio.set_target(target, ramp);
    }

    /// Sets a new target master amplitude (linear), gliding with `ramp`.
    #[inline]
    pub fn set_amplitude(&mut self, linear: Sample, ramp: Ramp) {
        self.amplitude
            .set_target(finite_or(linear, self.amplitude.target()), ramp);
    }

    /// Returns the current master frequency in hertz.
    #[inline]
    #[must_use]
    pub fn master_hz(&self) -> Sample {
        self.master_hz
    }

    /// Returns the sync ratio the node is gliding toward.
    #[inline]
    #[must_use]
    pub fn sync_ratio(&self) -> Sample {
        self.sync_ratio.target()
    }

    /// Returns the target amplitude the node is gliding toward (linear).
    #[inline]
    #[must_use]
    pub fn amplitude(&self) -> Sample {
        self.amplitude.target()
    }

    /// Produces one output sample given the master and slave per-sample phase
    /// increments, advancing both phase accumulators and the smoothed controls.
    #[inline]
    fn render_sample(&mut self, dms: Sample, slave_hz: Sample, sr: Sample) -> Sample {
        let amp = self.amplitude.next_sample();
        let dss = slave_hz / sr;

        // Naive slave sawtooth plus the correction for its own wrap, evaluated
        // on the pre-advance phase (shared two-sided PolyBLEP convention).
        let mut value = (2.0 * self.slave_phase - 1.0) - poly_blep(self.slave_phase, dss);

        // Advance the slave; let it wrap naturally.
        self.slave_phase += dss;
        if self.slave_phase >= 1.0 {
            self.slave_phase -= (self.slave_phase as u32) as Sample;
        }

        // Advance the master; a wrap forces a slave reset.
        self.master_phase += dms;
        if self.master_phase >= 1.0 {
            // Overflow in [0, dms): how far past the wrap "now" sits.
            self.master_phase -= (self.master_phase as u32) as Sample;
            let frac = if dms > 0.0 {
                self.master_phase / dms
            } else {
                0.0
            };
            // Slave phase the old trajectory held at the reset instant, which
            // happened `frac` samples before "now".
            let mut ps_reset = self.slave_phase - frac * dss;
            if ps_reset < 0.0 {
                ps_reset += 1.0;
            }
            // Post-reset trajectory: snapped to 0 then advanced `frac` samples.
            let new_phase = frac * dss;
            // Signed jump of the reset discontinuity: (2*0 - 1) - (2*ps - 1).
            let step = -2.0 * ps_reset;
            self.slave_phase = new_phase;
            // The active trajectory at "now" is already the post-reset ramp, so
            // recompute the naive value from `new_phase` and round the reset step
            // with a one-sided (two-point) PolyBLEP residual.
            value = (2.0 * new_phase - 1.0) + (step * 0.5) * poly_blep(self.master_phase, dms);
        }

        value * amp
    }
}

impl AudioNode for HardSyncOscillatorNode {
    fn process(&mut self, ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let channels = io.output(0).channels();
        if channels == 0 {
            return;
        }

        // `sample_rate` is validated non-zero by the graph; guard defensively.
        let sr = ctx.sample_rate.max(1) as Sample;
        let guard_hz = sr * NYQUIST_GUARD;
        let master_hz = self.master_hz.min(guard_hz);
        let dms = master_hz / sr;

        {
            let buf = io.output(0).channel_mut(0);
            for s in buf.iter_mut() {
                let ratio = self.sync_ratio.next_sample();
                let slave_hz = (self.master_hz * ratio).min(guard_hz);
                *s = self.render_sample(dms, slave_hz, sr);
            }
        }

        for ch in 1..channels {
            let (src, dst) = io.output(0).channel_pair_mut(0, ch);
            dst.copy_from_slice(src);
        }
    }

    fn reset(&mut self) {
        self.master_phase = 0.0;
        self.slave_phase = 0.0;
        self.sync_ratio = Smoothed::new(self.sync_ratio.target());
        self.amplitude = Smoothed::new(self.amplitude.target());
    }

    fn latency_frames(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};

    const SR: u32 = 48_000;

    fn ctx(sample_rate: u32, frames: usize) -> RenderContext {
        RenderContext {
            sample_rate,
            frames,
            playhead: 0,
        }
    }

    fn render(node: &mut HardSyncOscillatorNode, sample_rate: u32, frames: usize) -> AudioBuffer {
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn render_layout(
        node: &mut HardSyncOscillatorNode,
        layout: ChannelLayout,
        sample_rate: u32,
        frames: usize,
    ) -> AudioBuffer {
        let mut out = AudioBuffer::new(layout, frames);
        out.set_active_frames(frames);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(sample_rate, frames), &mut io);
        outputs.into_iter().next().unwrap()
    }

    fn energy(buf: &AudioBuffer) -> Sample {
        buf.channel(0).iter().map(|s| s * s).sum()
    }

    /// Energy of a single Goertzel bin at `freq` over the channel.
    fn goertzel(buf: &AudioBuffer, sample_rate: u32, freq: Sample) -> Sample {
        let samples = buf.channel(0);
        let n = samples.len();
        if n == 0 {
            return 0.0;
        }
        let omega = core::f32::consts::TAU * freq / sample_rate as Sample;
        let coeff = 2.0 * bevy_math::ops::cos(omega);
        let mut s_prev = 0.0;
        let mut s_prev2 = 0.0;
        for &x in samples {
            let s = x + coeff * s_prev - s_prev2;
            s_prev2 = s_prev;
            s_prev = s;
        }
        s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2
    }

    /// Spectral centroid proxy: ratio of high-band to low-band Goertzel energy.
    fn brightness(node_ratio: Sample) -> Sample {
        let mut node = HardSyncOscillatorNode::new(110.0, node_ratio, 0.8);
        let out = render(&mut node, SR, 4_096);
        let mut hi = 0.0;
        let mut lo = 0.0;
        // Harmonics of the 110 Hz master.
        for h in 1..=40 {
            let f = 110.0 * h as Sample;
            if f >= SR as Sample * 0.49 {
                break;
            }
            let e = goertzel(&out, SR, f);
            if h <= 6 {
                lo += e;
            } else {
                hi += e;
            }
        }
        hi / (lo + 1.0)
    }

    #[test]
    fn renders_bounded_finite() {
        let mut node = HardSyncOscillatorNode::new(110.0, 3.7, 1.0);
        let out = render(&mut node, SR, 8_192);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.5, "s={s}");
        }
    }

    #[test]
    fn extreme_ratio_stays_bounded_finite() {
        let mut node = HardSyncOscillatorNode::new(90.0, MAX_SYNC_RATIO, 1.0);
        let out = render(&mut node, SR, 8_192);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.5, "s={s}");
        }
    }

    #[test]
    fn fundamental_locks_to_master() {
        // Even for a non-integer ratio the perceived pitch is the master, so the
        // master bin should dominate neighbouring non-harmonic bins.
        let mut node = HardSyncOscillatorNode::new(110.0, 2.6, 0.8);
        let out = render(&mut node, SR, 8_192);
        let at_master = goertzel(&out, SR, 110.0);
        let off = goertzel(&out, SR, 110.0 * 2.6);
        assert!(
            at_master > off,
            "master={at_master} slave_bin={off}"
        );
    }

    #[test]
    fn ratio_raises_brightness() {
        let low = brightness(1.2);
        let high = brightness(8.0);
        assert!(high > low, "low={low} high={high}");
    }

    #[test]
    fn ratio_one_is_near_plain_saw() {
        // At ratio 1 the slave runs at the master frequency and the reset lands
        // on the slave's own wrap, so the fundamental dominates strongly.
        let mut node = HardSyncOscillatorNode::new(220.0, 1.0, 0.8);
        let out = render(&mut node, SR, 8_192);
        let fund = goertzel(&out, SR, 220.0);
        let h2 = goertzel(&out, SR, 440.0);
        assert!(fund > h2, "fund={fund} h2={h2}");
    }

    #[test]
    fn sweep_ratio_is_click_free() {
        let frames: usize = 4_096;
        let mut node = HardSyncOscillatorNode::new(110.0, 1.5, 0.8);
        node.set_sync_ratio(
            12.0,
            Ramp::Linear {
                samples: frames as u32,
            },
        );
        let out = render(&mut node, SR, frames);
        for w in out.channel(0).windows(2) {
            // A full reset step is at most ~2 units; a smoothed sweep must not
            // exceed that bound sample-to-sample.
            assert!((w[1] - w[0]).abs() <= 2.5, "step {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn deterministic() {
        let mut a = HardSyncOscillatorNode::new(130.0, 3.3, 0.8);
        let mut b = HardSyncOscillatorNode::new(130.0, 3.3, 0.8);
        let ra = render(&mut a, SR, 2_048);
        let rb = render(&mut b, SR, 2_048);
        for (x, y) in ra.channel(0).iter().zip(rb.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn reset_replays_output() {
        let mut node = HardSyncOscillatorNode::new(130.0, 3.3, 0.8);
        let a = render(&mut node, SR, 1_024);
        node.reset();
        let b = render(&mut node, SR, 1_024);
        for (x, y) in a.channel(0).iter().zip(b.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn reset_restores_phases() {
        let mut node = HardSyncOscillatorNode::new(130.0, 3.3, 0.8);
        let _ = render(&mut node, SR, 97);
        node.reset();
        assert_eq!(node.master_phase, 0.0);
        assert_eq!(node.slave_phase, 0.0);
    }

    #[test]
    fn silent_when_amplitude_zero() {
        let mut node = HardSyncOscillatorNode::new(110.0, 3.0, 0.0);
        let out = render(&mut node, SR, 1_024);
        for &s in out.channel(0) {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn amplitude_scales_energy_quadratically() {
        let mut quiet = HardSyncOscillatorNode::new(110.0, 3.0, 0.25);
        let mut loud = HardSyncOscillatorNode::new(110.0, 3.0, 0.5);
        let eq = energy(&render(&mut quiet, SR, 4_096));
        let el = energy(&render(&mut loud, SR, 4_096));
        // Doubling the amplitude quadruples the energy.
        assert!((el / eq - 4.0).abs() < 0.1, "ratio={}", el / eq);
    }

    #[test]
    fn mono_core_copied_to_stereo() {
        let mut node = HardSyncOscillatorNode::new(110.0, 3.0, 0.8);
        let out = render_layout(&mut node, ChannelLayout::Stereo, SR, 512);
        assert_eq!(out.channel(0), out.channel(1));
    }

    #[test]
    fn mono_core_copied_to_quad() {
        let mut node = HardSyncOscillatorNode::new(110.0, 3.0, 0.8);
        let out = render_layout(&mut node, ChannelLayout::Quad, SR, 512);
        for ch in 1..4 {
            assert_eq!(out.channel(0), out.channel(ch));
        }
    }

    #[test]
    fn not_silent() {
        let mut node = HardSyncOscillatorNode::new(110.0, 3.0, 0.8);
        let out = render(&mut node, SR, 1_024);
        assert!(energy(&out) > 1.0, "energy={}", energy(&out));
    }

    #[test]
    fn zero_frames_is_noop() {
        let mut node = HardSyncOscillatorNode::new(110.0, 3.0, 0.8);
        let mut out = AudioBuffer::new(ChannelLayout::Mono, 64);
        out.set_active_frames(0);
        let inputs: [AudioBuffer; 0] = [];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(SR, 0), &mut io);
        assert_eq!(node.master_phase, 0.0);
        assert_eq!(node.slave_phase, 0.0);
    }

    #[test]
    fn latency_is_zero() {
        let node = HardSyncOscillatorNode::new(110.0, 3.0, 0.8);
        assert_eq!(node.latency_frames(), 0);
    }

    #[test]
    fn getters_report_targets() {
        let mut node = HardSyncOscillatorNode::new(110.0, 1.5, 0.5);
        node.set_master_hz(321.0);
        node.set_sync_ratio(6.0, Ramp::Immediate);
        node.set_amplitude(0.9, Ramp::Immediate);
        assert_eq!(node.master_hz(), 321.0);
        assert_eq!(node.sync_ratio(), 6.0);
        assert_eq!(node.amplitude(), 0.9);
    }

    #[test]
    fn default_params_in_domain() {
        let p = HardSyncOscillatorParams::default();
        assert_eq!(p.master_hz, DEFAULT_MASTER_HZ);
        assert_eq!(p.sync_ratio, DEFAULT_SYNC_RATIO);
        assert_eq!(p.amplitude, DEFAULT_AMPLITUDE);
        // Sanitising a valid bundle is idempotent.
        let s = p.sanitised();
        assert_eq!(s.master_hz, p.master_hz);
        assert_eq!(s.sync_ratio, p.sync_ratio);
        assert_eq!(s.amplitude, p.amplitude);
    }

    #[test]
    fn from_params_matches_new() {
        let params = HardSyncOscillatorParams {
            master_hz: 123.0,
            sync_ratio: 4.5,
            amplitude: 0.7,
        };
        let mut a = HardSyncOscillatorNode::from_params(params);
        let mut b = HardSyncOscillatorNode::new(123.0, 4.5, 0.7);
        let ra = render(&mut a, SR, 1_024);
        let rb = render(&mut b, SR, 1_024);
        for (x, y) in ra.channel(0).iter().zip(rb.channel(0)) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn constructor_clamps_and_sanitises() {
        let low = HardSyncOscillatorNode::new(1.0, 0.1, 0.8);
        let high = HardSyncOscillatorNode::new(99_999.0, 999.0, 0.8);
        assert_eq!(low.master_hz(), MIN_MASTER_HZ);
        assert_eq!(low.sync_ratio(), MIN_SYNC_RATIO);
        assert_eq!(high.master_hz(), MAX_MASTER_HZ);
        assert_eq!(high.sync_ratio(), MAX_SYNC_RATIO);
    }

    #[test]
    fn non_finite_inputs_fall_back() {
        let node =
            HardSyncOscillatorNode::new(Sample::NAN, Sample::INFINITY, Sample::NAN);
        assert_eq!(node.master_hz(), DEFAULT_MASTER_HZ);
        assert_eq!(node.sync_ratio(), DEFAULT_SYNC_RATIO);
        assert_eq!(node.amplitude(), DEFAULT_AMPLITUDE);
    }

    #[test]
    fn setters_reject_non_finite_and_clamp() {
        let mut node = HardSyncOscillatorNode::new(110.0, 3.0, 0.8);
        node.set_master_hz(Sample::NAN);
        assert_eq!(node.master_hz(), 110.0);
        node.set_sync_ratio(Sample::INFINITY, Ramp::Immediate);
        assert_eq!(node.sync_ratio(), 3.0);
        node.set_amplitude(Sample::NAN, Ramp::Immediate);
        assert_eq!(node.amplitude(), 0.8);
        node.set_master_hz(1.0);
        assert_eq!(node.master_hz(), MIN_MASTER_HZ);
        node.set_sync_ratio(999.0, Ramp::Immediate);
        assert_eq!(node.sync_ratio(), MAX_SYNC_RATIO);
    }

    #[test]
    fn nyquist_guard_keeps_super_nyquist_bounded() {
        // An absurd master and ratio must still produce bounded, finite output
        // because both frequencies are clamped below Nyquist.
        let mut node = HardSyncOscillatorNode::new(MAX_MASTER_HZ, MAX_SYNC_RATIO, 1.0);
        let out = render(&mut node, SR, 2_048);
        for &s in out.channel(0) {
            assert!(s.is_finite() && s.abs() <= 1.5, "s={s}");
        }
    }

    #[test]
    fn master_hz_change_shifts_fundamental() {
        let mut low = HardSyncOscillatorNode::new(110.0, 3.0, 0.8);
        let mut high = HardSyncOscillatorNode::new(220.0, 3.0, 0.8);
        let lo = render(&mut low, SR, 8_192);
        let hi = render(&mut high, SR, 8_192);
        // The 220 Hz instance has more energy at 220 Hz than the 110 Hz one.
        let lo_at_220 = goertzel(&lo, SR, 220.0);
        let hi_at_220 = goertzel(&hi, SR, 220.0);
        assert!(hi_at_220 > lo_at_220, "lo={lo_at_220} hi={hi_at_220}");
    }
}
