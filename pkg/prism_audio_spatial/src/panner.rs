//! Multi-channel amplitude panning for horizontal speaker layouts.
//!
//! This module places a mono point source onto the physical speakers of a
//! target [`ChannelLayout`] using *pairwise, constant-power amplitude panning*
//! -- the horizontal special case of Vector Base Amplitude Panning (VBAP).
//!
//! # Algorithm
//!
//! Every speaker of a layout is assigned a fixed azimuth on the horizontal
//! ring (the LFE has no direction and is never fed by the panner). For a given
//! source azimuth the panner finds the two adjacent speakers that bracket the
//! source and splits the signal between them with an equal-power law:
//!
//! ```text
//! t  = (azimuth - a0) / (a1 - a0)      (clamped to [0, 1])
//! g0 = cos(t * pi/2)                   (gain of the "left" speaker of the pair)
//! g1 = sin(t * pi/2)                   (gain of the "right" speaker of the pair)
//! ```
//!
//! Because `cos^2 + sin^2 == 1`, the summed acoustic power is constant as the
//! source pans across the field, which avoids the loudness dip a linear
//! (equal-gain) pan would produce. All other speakers receive a gain of zero.
//! The pair search wraps around `+/-pi` so a source directly behind the
//! listener is shared between the two rear-most speakers.
//!
//! # Azimuth convention
//!
//! Azimuth is in radians with `0` straight ahead and **positive angles toward
//! the right**, in the half-open range `(-pi, pi]`. This matches
//! [`LocalSource::azimuth`](crate::geometry::LocalSource::azimuth).
//!
//! # Real-time contract
//!
//! Gain computation (the trigonometric part) happens in non-real-time setters
//! ([`PannerNode::set_azimuth`] and friends). The audio hot path
//! ([`PannerNode`]'s [`AudioNode::process`]) only multiplies by per-channel
//! [`Smoothed`] gains, so it is **allocation free, lock free, and panic free**
//! and uses only fixed-size stack arrays (at most eight channels).
//!
//! # Determinism
//!
//! All trigonometry routes through [`bevy_math::ops`] (libm-backed) rather than
//! `f32` intrinsics, keeping the result bit-reproducible across targets.
//!
//! # Provenance
//!
//! The technique is standard, publicly documented spatial-audio knowledge:
//! Ville Pulkki, "Virtual Sound Source Positioning Using Vector Base Amplitude
//! Panning", *Journal of the Audio Engineering Society*, 45(6), 1997, together
//! with the classic constant-power (sine/cosine) pan law. This file contains
//! **no Unreal Engine, Unity, Godot, Wwise, or FMOD source or derived code**.

use bevy_math::ops;
use core::f32::consts::{FRAC_PI_2, PI, TAU};

use prism_audio_core::buffer::ChannelLayout;
use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::math::Sample;
use prism_audio_core::param::{Ramp, Smoothed};

/// Maximum number of speakers (and thus ring entries) any supported layout can
/// have. Sized for 7.1, the widest layout; used to keep every buffer on the
/// stack so the hot path never allocates.
const MAX_SPEAKERS: usize = 8;

/// Converts a whole-degree azimuth into radians using only the compile-time
/// constant `pi`, so no `f32` transcendental intrinsic is involved.
#[inline]
const fn deg(degrees: Sample) -> Sample {
    degrees * (PI / 180.0)
}

/// Wraps an arbitrary azimuth into the half-open range `(-pi, pi]`.
///
/// Uses floating-point remainder (not a transcendental function), so it is
/// deterministic and safe on the real-time thread.
#[inline]
fn wrap_pi(azimuth: Sample) -> Sample {
    // `%` leaves the value in (-2pi, 2pi); a single conditional shift then
    // lands it in (-pi, pi].
    let mut x = azimuth % TAU;
    if x <= -PI {
        x += TAU;
    } else if x > PI {
        x -= TAU;
    }
    x
}

/// Sorts the first entries of a speaker ring ascending by azimuth.
///
/// A branch-light insertion sort is used because the ring is tiny (at most
/// [`MAX_SPEAKERS`] entries) and this runs only at construction time.
fn sort_ring(ring: &mut [(Sample, usize)]) {
    let n = ring.len();
    let mut i = 1;
    while i < n {
        let key = ring[i];
        let mut j = i;
        while j > 0 && ring[j - 1].0 > key.0 {
            ring[j] = ring[j - 1];
            j -= 1;
        }
        ring[j] = key;
        i += 1;
    }
}

/// Maps a source azimuth onto per-channel linear gains for a fixed layout.
pub trait Panner {
    /// Returns the channel layout this panner targets.
    #[must_use]
    fn layout(&self) -> ChannelLayout;

    /// Writes the linear gain for every channel of [`Panner::layout`] into
    /// `out`.
    ///
    /// `out.len()` must be at least [`ChannelLayout::channel_count`]; channels
    /// beyond that (and any that `out` cannot hold) are simply skipped. The
    /// LFE channel is always left at zero. This never panics and never
    /// allocates.
    fn compute_gains(&self, azimuth: Sample, out: &mut [Sample]);
}

/// A horizontal pairwise / VBAP panner for a fixed [`ChannelLayout`].
///
/// Construction pre-computes the sorted speaker ring; [`Panner::compute_gains`]
/// then only performs a small search plus one `sin_cos` evaluation.
#[derive(Debug, Clone)]
pub struct VbapPanner {
    layout: ChannelLayout,
    /// `(azimuth, channel_index)` for each directional speaker, sorted
    /// ascending by azimuth. The LFE is excluded.
    ring: [(Sample, usize); MAX_SPEAKERS],
    /// Number of valid entries at the front of `ring`.
    ring_len: usize,
}

impl VbapPanner {
    /// Builds a panner for `layout`, pre-computing its speaker ring.
    ///
    /// Speaker azimuths follow common ITU/production placements (0 = front,
    /// positive toward the right):
    ///
    /// * Stereo: L = -30 deg, R = +30 deg.
    /// * Quad: FL = -45 deg, FR = +45 deg, SL = -135 deg, SR = +135 deg.
    /// * 5.1: FL = -30 deg, FR = +30 deg, C = 0 deg, SL = -110 deg,
    ///   SR = +110 deg (LFE has no direction).
    /// * 7.1: FL = -30 deg, FR = +30 deg, C = 0 deg, SL = -90 deg,
    ///   SR = +90 deg, RL = -150 deg, RR = +150 deg (LFE has no direction).
    /// * Mono: the single channel always receives unity gain.
    /// * [`ChannelLayout::AmbisonicFoa`]: not handled here (Ambisonic encoding
    ///   is the ambisonics module's job); all gains are zero.
    #[must_use]
    pub fn new(layout: ChannelLayout) -> Self {
        let mut ring = [(0.0, 0usize); MAX_SPEAKERS];
        let mut len = 0usize;
        match layout {
            ChannelLayout::Stereo => {
                ring[0] = (deg(-30.0), 0);
                ring[1] = (deg(30.0), 1);
                len = 2;
            }
            ChannelLayout::Quad => {
                // Channel order: FL, FR, SL, SR.
                ring[0] = (deg(-45.0), 0);
                ring[1] = (deg(45.0), 1);
                ring[2] = (deg(-135.0), 2);
                ring[3] = (deg(135.0), 3);
                len = 4;
            }
            ChannelLayout::Surround5_1 => {
                // Channel order: FL, FR, C, LFE, SL, SR (LFE = index 3 skipped).
                ring[0] = (deg(-30.0), 0);
                ring[1] = (deg(30.0), 1);
                ring[2] = (deg(0.0), 2);
                ring[3] = (deg(-110.0), 4);
                ring[4] = (deg(110.0), 5);
                len = 5;
            }
            ChannelLayout::Surround7_1 => {
                // Channel order: FL, FR, C, LFE, SL, SR, RL, RR (LFE skipped).
                ring[0] = (deg(-30.0), 0);
                ring[1] = (deg(30.0), 1);
                ring[2] = (deg(0.0), 2);
                ring[3] = (deg(-90.0), 4);
                ring[4] = (deg(90.0), 5);
                ring[5] = (deg(-150.0), 6);
                ring[6] = (deg(150.0), 7);
                len = 7;
            }
            // Mono has a single (directionless) channel and FOA is encoded
            // elsewhere, so both leave the ring empty. `ChannelLayout` is also
            // `#[non_exhaustive]`, so any future layout falls back to an empty
            // ring (silence) here until it is explicitly supported.
            _ => {}
        }
        sort_ring(&mut ring[..len]);
        Self {
            layout,
            ring,
            ring_len: len,
        }
    }
}

impl Panner for VbapPanner {
    #[inline]
    fn layout(&self) -> ChannelLayout {
        self.layout
    }

    fn compute_gains(&self, azimuth: Sample, out: &mut [Sample]) {
        let count = self.layout.channel_count();

        // Start from silence on every channel we are responsible for.
        for c in 0..count {
            if let Some(slot) = out.get_mut(c) {
                *slot = 0.0;
            }
        }

        match self.layout {
            ChannelLayout::Mono => {
                if let Some(slot) = out.get_mut(0) {
                    *slot = 1.0;
                }
                return;
            }
            // Ambisonic encoding is handled by the ambisonics module; leave the
            // W/X/Y/Z channels silent here.
            ChannelLayout::AmbisonicFoa => return,
            _ => {}
        }

        let n = self.ring_len;
        if n == 0 {
            return;
        }
        if n == 1 {
            let (_, ch) = self.ring[0];
            if let Some(slot) = out.get_mut(ch) {
                *slot = 1.0;
            }
            return;
        }

        let az = wrap_pi(azimuth);

        // Find the adjacent speaker pair (i, j) bracketing the source, together
        // with the (possibly +2pi-unwrapped) endpoints and effective azimuth.
        let mut bracket: Option<(usize, usize, Sample, Sample, Sample)> = None;
        let mut i = 0;
        while i + 1 < n {
            let a0 = self.ring[i].0;
            let a1 = self.ring[i + 1].0;
            if az >= a0 && az <= a1 {
                bracket = Some((i, i + 1, a0, a1, az));
                break;
            }
            i += 1;
        }
        let (idx0, idx1, a0, a1, az_eff) = match bracket {
            Some(v) => v,
            None => {
                // Source falls in the wrap-around gap between the last and the
                // first speaker (crossing +/-pi).
                let a0 = self.ring[n - 1].0;
                let a1 = self.ring[0].0 + TAU;
                let az_eff = if az < self.ring[0].0 { az + TAU } else { az };
                (n - 1, 0, a0, a1, az_eff)
            }
        };

        let span = a1 - a0;
        let t = if span.abs() > Sample::EPSILON {
            ((az_eff - a0) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let (sin, cos) = ops::sin_cos(t * FRAC_PI_2);
        let ch0 = self.ring[idx0].1;
        let ch1 = self.ring[idx1].1;
        if let Some(slot) = out.get_mut(ch0) {
            *slot = cos;
        }
        if let Some(slot) = out.get_mut(ch1) {
            *slot = sin;
        }
    }
}

/// A real-time [`AudioNode`] that pans a mono input onto a target layout.
///
/// The node keeps one [`Smoothed`] gain per output channel so that changing the
/// source direction glides click-free. Direction changes are applied through
/// [`PannerNode::set_azimuth`] / [`PannerNode::set_azimuth_immediate`] off the
/// audio thread; [`AudioNode::process`] only reads the smoothed gains.
///
/// Input port 0 is expected to be [`ChannelLayout::Mono`]; output port 0 is the
/// layout passed to [`PannerNode::new`].
#[derive(Debug, Clone)]
pub struct PannerNode {
    panner: VbapPanner,
    gains: [Smoothed; MAX_SPEAKERS],
    active: usize,
}

impl PannerNode {
    /// Creates a panner node for `layout` with every channel gain settled at
    /// zero. Call [`PannerNode::set_azimuth_immediate`] once after construction
    /// to establish the initial direction without a ramp.
    #[must_use]
    pub fn new(layout: ChannelLayout) -> Self {
        Self {
            panner: VbapPanner::new(layout),
            gains: [Smoothed::new(0.0); MAX_SPEAKERS],
            active: layout.channel_count().min(MAX_SPEAKERS),
        }
    }

    /// Returns the output channel layout.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.panner.layout()
    }

    /// Retargets every channel gain toward the source `azimuth` using `ramp`.
    ///
    /// This performs the trigonometric gain computation and is **not**
    /// real-time safe; call it from a control thread, not from `process`.
    pub fn set_azimuth(&mut self, azimuth: Sample, ramp: Ramp) {
        let mut target = [0.0; MAX_SPEAKERS];
        self.panner.compute_gains(azimuth, &mut target);
        for (slot, &t) in self.gains.iter_mut().zip(target.iter()).take(self.active) {
            slot.set_target(t, ramp);
        }
    }

    /// Retargets every channel gain toward `azimuth` and snaps immediately,
    /// with no glide. Intended for establishing the initial position.
    #[inline]
    pub fn set_azimuth_immediate(&mut self, azimuth: Sample) {
        self.set_azimuth(azimuth, Ramp::Immediate);
    }
}

impl AudioNode for PannerNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);

        // A panner needs a mono source; bail out safely if there is none.
        if input.channels() == 0 {
            return;
        }
        let mono = input.channel(0);
        let frames = output.active_frames().min(mono.len());
        let active = self.active.min(output.channels());

        for (i, &sample) in mono.iter().enumerate().take(frames) {
            // Advance all channel gains for this frame first so they stay in
            // lock-step, then write the scaled sample to each channel.
            let mut frame_gain = [0.0; MAX_SPEAKERS];
            for (slot, gain) in frame_gain.iter_mut().zip(self.gains.iter_mut()).take(active) {
                *slot = gain.next_sample();
            }
            for (c, &g) in frame_gain.iter().enumerate().take(active) {
                output.channel_mut(c)[i] = sample * g;
            }
        }
    }

    fn reset(&mut self) {
        for c in 0..MAX_SPEAKERS {
            self.gains[c] = Smoothed::new(self.gains[c].target());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_6, PI, TAU};

    use prism_audio_core::buffer::AudioBuffer;

    const EPS: Sample = 1.0e-4;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    fn power(gains: &[Sample]) -> Sample {
        gains.iter().map(|g| g * g).sum()
    }

    #[test]
    fn stereo_front_is_centered_and_equal_power() {
        let p = VbapPanner::new(ChannelLayout::Stereo);
        let mut g = [0.0; MAX_SPEAKERS];
        p.compute_gains(0.0, &mut g);
        // Front-center splits equally between L and R...
        assert!(approx(g[0], g[1], EPS), "L={} R={}", g[0], g[1]);
        // ...with unity summed power.
        assert!(approx(power(&g[..2]), 1.0, EPS), "power={}", power(&g[..2]));
    }

    #[test]
    fn stereo_hard_left_biases_left() {
        let p = VbapPanner::new(ChannelLayout::Stereo);

        // Exactly at the left speaker: all energy in L.
        let mut g = [0.0; MAX_SPEAKERS];
        p.compute_gains(-FRAC_PI_6, &mut g);
        assert!(approx(g[0], 1.0, EPS), "L={}", g[0]);
        assert!(approx(g[1], 0.0, EPS), "R={}", g[1]);

        // Further left than the speaker (wrap-around arc): still L-dominant.
        let mut g2 = [0.0; MAX_SPEAKERS];
        p.compute_gains(-FRAC_PI_2, &mut g2);
        assert!(g2[0] > g2[1], "L={} R={}", g2[0], g2[1]);
        assert!(approx(power(&g2[..2]), 1.0, EPS));
    }

    #[test]
    fn quad_behind_biases_rears_symmetrically() {
        let p = VbapPanner::new(ChannelLayout::Quad);
        let mut g = [0.0; MAX_SPEAKERS];
        p.compute_gains(PI, &mut g); // directly behind
        // SL (ch2) and SR (ch3) dominate the fronts FL (ch0) / FR (ch1).
        assert!(g[2] > g[0] && g[2] > g[1], "gains={:?}", g);
        assert!(g[3] > g[0] && g[3] > g[1], "gains={:?}", g);
        // Symmetric behind the listener.
        assert!(approx(g[2], g[3], EPS), "SL={} SR={}", g[2], g[3]);
        assert!(approx(power(&g[..4]), 1.0, EPS));
    }

    #[test]
    fn surround_lfe_is_always_silent() {
        for layout in [ChannelLayout::Surround5_1, ChannelLayout::Surround7_1] {
            let p = VbapPanner::new(layout);
            let count = layout.channel_count();
            for k in 0..180 {
                let az = -PI + (k as Sample) * (TAU / 180.0);
                let mut g = [0.0; MAX_SPEAKERS];
                p.compute_gains(az, &mut g);
                // LFE is channel index 3 in both 5.1 and 7.1.
                assert_eq!(g[3], 0.0, "layout={:?} az={}", layout, az);
                assert!(approx(power(&g[..count]), 1.0, EPS));
            }
        }
    }

    #[test]
    fn all_layouts_are_constant_power() {
        for layout in [
            ChannelLayout::Stereo,
            ChannelLayout::Quad,
            ChannelLayout::Surround5_1,
            ChannelLayout::Surround7_1,
        ] {
            let p = VbapPanner::new(layout);
            let count = layout.channel_count();
            for k in 0..360 {
                let az = -PI + (k as Sample) * (TAU / 360.0);
                let mut g = [0.0; MAX_SPEAKERS];
                p.compute_gains(az, &mut g);
                let pw = power(&g[..count]);
                assert!(approx(pw, 1.0, EPS), "layout={:?} az={} power={}", layout, az, pw);
            }
        }
    }

    #[test]
    fn mono_is_unity_gain() {
        let p = VbapPanner::new(ChannelLayout::Mono);
        let mut g = [0.0; MAX_SPEAKERS];
        // Azimuth is irrelevant for a single, directionless channel.
        p.compute_gains(1.234, &mut g);
        assert!(approx(g[0], 1.0, EPS));
    }

    #[test]
    fn foa_is_silent() {
        let p = VbapPanner::new(ChannelLayout::AmbisonicFoa);
        let mut g = [0.0; MAX_SPEAKERS];
        p.compute_gains(0.5, &mut g);
        assert!(g[..4].iter().all(|&x| x == 0.0), "gains={:?}", g);
    }

    #[test]
    fn wrap_pi_normalizes_range() {
        // The contract is: the result lies in (-pi, pi] and denotes the same
        // physical direction as the input. Note that `3.0 * PI` and
        // `-3.0 * PI` are the *same* angle as +/-pi, and +pi/-pi are the same
        // direction. Because `-3.0 * PI` does not land exactly on -pi after the
        // floating-point remainder, it legitimately stays near -pi rather than
        // snapping to +pi; both are correct, so we assert direction + range
        // instead of a single signed value.
        for &x in &[3.0 * PI, -3.0 * PI] {
            let w = wrap_pi(x);
            assert!(w > -PI - 1e-4 && w <= PI + 1e-4, "out of range: {w}");
            assert!(approx(w.abs(), PI, 1e-4), "not +/-pi: {w}");
        }
        // An exact -pi input must map to +pi (half-open upper bound).
        assert!(approx(wrap_pi(-PI), PI, 1e-6));
        assert!(approx(wrap_pi(0.0), 0.0, 1e-6));
        let w = wrap_pi(2.0 * PI + 0.1);
        assert!(w > -PI && w <= PI);
    }

    #[test]
    fn node_mono_passthrough_preserves_frames() {
        let mut node = PannerNode::new(ChannelLayout::Mono);
        node.set_azimuth_immediate(0.0);

        let data = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut input = AudioBuffer::new(ChannelLayout::Mono, data.len());
        input.channel_mut(0).copy_from_slice(&data);
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, data.len())];

        let ctx = RenderContext {
            sample_rate: 48_000,
            frames: data.len(),
            playhead: 0,
        };
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }

        assert_eq!(outputs[0].active_frames(), data.len());
        for (o, d) in outputs[0].channel(0).iter().zip(data.iter()) {
            assert!(approx(*o, *d, EPS), "out={} in={}", o, d);
        }
    }

    #[test]
    fn node_smoothing_has_no_click() {
        let frames = 64;
        let mut node = PannerNode::new(ChannelLayout::Stereo);
        // Start fully left, then glide to fully right over the whole block.
        node.set_azimuth_immediate(-FRAC_PI_6);
        node.set_azimuth(FRAC_PI_6, Ramp::Linear { samples: frames as u32 });

        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        for s in input.channel_mut(0) {
            *s = 1.0;
        }
        let inputs = [input];
        let mut outputs = [AudioBuffer::new(ChannelLayout::Stereo, frames)];

        let ctx = RenderContext {
            sample_rate: 48_000,
            frames,
            playhead: 0,
        };
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }

        assert_eq!(outputs[0].active_frames(), frames);
        let right = outputs[0].channel(1);
        // Right channel gain ramps up gradually rather than jumping.
        assert!(right[0] < 0.5, "first R gain={}", right[0]);
        assert!(right[frames - 1] > 0.5, "last R gain={}", right[frames - 1]);
        assert!(right[0] < right[frames - 1]);
        // No single-sample jump exceeds a small step (click-free).
        for w in right.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.1, "jump {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn node_reset_settles_at_target() {
        let mut node = PannerNode::new(ChannelLayout::Stereo);
        node.set_azimuth(FRAC_PI_2, Ramp::Linear { samples: 128 });
        node.reset();
        // After reset the gains sit exactly on their targets (settled).
        for c in 0..node.active {
            assert!(node.gains[c].is_settled());
            assert!(approx(node.gains[c].current(), node.gains[c].target(), 1e-9));
        }
    }
}
