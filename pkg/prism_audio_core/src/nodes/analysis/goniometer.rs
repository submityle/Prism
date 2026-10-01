//! Goniometer (vectorscope) coordinate generator for stereo-field display.
//!
//! A goniometer is the oscilloscope-style display an engineer watches to *see*
//! a stereo image: every sample pair `(L, R)` becomes a point on a Lissajous
//! plot, and the resulting cloud shows width, balance, and mono compatibility
//! at a glance. This module is the real-time-safe front end that turns a stereo
//! stream into the stream of display coordinates such a scope draws; it does
//! not render pixels, it produces the `(x, y)` points a renderer plots.
//!
//! # Orientation
//!
//! The classic broadcast goniometer is rotated `45` degrees from the raw
//! `L`/`R` axes so that a mono signal (`L == R`) traces a single vertical line
//! and a perfectly anti-phase signal (`L == -R`) traces a horizontal line. This
//! is exactly the energy-preserving mid/side rotation: with
//! `FRAC_1_SQRT_2 = 1 / sqrt(2)`,
//!
//! - `x = (L - R) * FRAC_1_SQRT_2` is the side (difference) component, plotted
//!   horizontally, and
//! - `y = (L + R) * FRAC_1_SQRT_2` is the mid (sum) component, plotted
//!   vertically.
//!
//! A centred mono source therefore stands straight up, widening sources fan out
//! horizontally, and an off-centre image leans to one side.
//!
//! # Decimation and the point ring
//!
//! An audio stream produces far more sample pairs per second than a display can
//! usefully plot, so the generator keeps only every `decimation`-th pair in a
//! fixed-capacity ring buffer sized once at construction. [`Goniometer::feed_sample`]
//! overwrites the oldest point when the ring is full, so the buffer always holds
//! the most recent window of display points without ever allocating. A peak-radius
//! accumulator tracks the largest vector magnitude seen since the last
//! [`Goniometer::reset`] so a renderer can auto-scale the plot.
//!
//! # Real-time contract
//!
//! The ring buffer and every accumulator are allocated once in
//! [`Goniometer::new`] / [`GoniometerNode::new`]. The per-sample hot path
//! ([`Goniometer::feed_sample`], [`GoniometerNode::process`]) performs no
//! allocation, takes no locks, and cannot panic: non-finite inputs are treated
//! as silence so the ring and the peak accumulator can never be poisoned by a
//! `NaN`. Reading the points back ([`Goniometer::points_ordered`]) copies into a
//! caller-provided slice and allocates nothing. All transcendental math routes
//! through [`bevy_math::ops`].
//!
//! # Provenance
//!
//! A goniometer is the standard `L`/`R` versus mid/side Lissajous display
//! described in every broadcast and mastering metering reference; the `45`
//! degree rotation is the elementary energy-preserving mid/side transform. This
//! module reuses only this crate's own [`Sample`] scalar. It is pure classic
//! DSP with no AI or ML and contains **no Unreal Engine, Unity, Godot, Wwise,
//! FMOD, Steam Audio, or Google Resonance Audio source or derived code**; it is
//! implemented purely from that publicly documented display convention.
//!
//! # Relationship
//!
//! This generator is the *visual* complement to the scalar
//! [`CorrelationMeter`](crate::nodes::analysis::correlation::CorrelationMeter):
//! the correlation meter reports how wide and mono-safe a bus is as running
//! numbers, while this module emits the point cloud a renderer draws for the
//! same decision. Both share the energy-preserving mid/side rotation
//! `M = (L + R) / sqrt(2)`, `S = (L - R) / sqrt(2)`; neither reuses the other's
//! code. The rotation is also the dual of the
//! [`StereoWidthNode`](crate::nodes::effects::StereoWidthNode) processing
//! effect.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;
use core::f32::consts::FRAC_1_SQRT_2;

use crate::graph::{AudioNode, ProcessIo, RenderContext};
use crate::math::{Sample, flush_denormal};

/// Smallest permitted display-point ring capacity.
pub const MIN_POINT_CAPACITY: usize = 1;

/// Default display-point ring capacity (a comfortable scope persistence).
pub const DEFAULT_POINT_CAPACITY: usize = 1024;

/// Default decimation factor (keep every sample pair).
pub const DEFAULT_DECIMATION: u32 = 1;

/// A single goniometer display coordinate in rotated mid/side space.
///
/// `x` is the side (horizontal) component and `y` is the mid (vertical)
/// component; both are already scaled by `FRAC_1_SQRT_2`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GoniometerPoint {
    /// Side (difference) component, plotted horizontally.
    pub x: Sample,
    /// Mid (sum) component, plotted vertically.
    pub y: Sample,
}

/// Auto-scaling statistics describing the current point cloud.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GoniometerStats {
    /// Largest vector magnitude seen since the last reset (for plot scaling).
    pub peak_radius: Sample,
    /// Root-mean-square vector magnitude since the last reset.
    pub rms_radius: Sample,
    /// Number of fed sample pairs since the last reset.
    pub sample_count: u64,
}

/// Converts a stereo stream into the stream of goniometer display coordinates,
/// buffering the most recent decimated points in a fixed-capacity ring.
#[derive(Clone, Debug)]
pub struct Goniometer {
    ring: Vec<GoniometerPoint>,
    write: usize,
    len: usize,
    decimation: u32,
    counter: u32,
    peak_radius: Sample,
    sum_radius_sq: f64,
    sample_count: u64,
}

impl Goniometer {
    /// Creates a goniometer with a `capacity`-point display ring (clamped to at
    /// least [`MIN_POINT_CAPACITY`]) that keeps every `decimation`-th sample
    /// pair (clamped to at least `1`).
    #[must_use]
    pub fn new(capacity: usize, decimation: u32) -> Self {
        let capacity = capacity.max(MIN_POINT_CAPACITY);
        Self {
            ring: vec![GoniometerPoint::default(); capacity],
            write: 0,
            len: 0,
            decimation: decimation.max(1),
            counter: 0,
            peak_radius: 0.0,
            sum_radius_sq: 0.0,
            sample_count: 0,
        }
    }

    /// The display ring capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.ring.len()
    }

    /// The current decimation factor.
    #[must_use]
    pub fn decimation(&self) -> u32 {
        self.decimation
    }

    /// Retunes the decimation factor (clamped to at least `1`). The phase
    /// counter is reset so the next retained point lands on a fresh boundary.
    pub fn set_decimation(&mut self, decimation: u32) {
        self.decimation = decimation.max(1);
        self.counter = 0;
    }

    /// Number of valid points currently buffered (at most [`capacity`]).
    ///
    /// [`capacity`]: Self::capacity
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the display ring currently holds no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Largest vector magnitude observed since the last [`reset`](Self::reset).
    #[must_use]
    pub fn peak_radius(&self) -> Sample {
        self.peak_radius
    }

    /// Current auto-scaling statistics.
    #[must_use]
    pub fn stats(&self) -> GoniometerStats {
        let rms_radius = if self.sample_count == 0 {
            0.0
        } else {
            let mean = self.sum_radius_sq / self.sample_count as f64;
            ops::sqrt(mean as Sample)
        };
        GoniometerStats {
            peak_radius: self.peak_radius,
            rms_radius,
            sample_count: self.sample_count,
        }
    }

    /// Feeds one stereo sample pair, rotating it into display space and, every
    /// `decimation` pairs, pushing the resulting point into the ring. Non-finite
    /// inputs are treated as silence.
    pub fn feed_sample(&mut self, left: Sample, right: Sample) {
        let l = if left.is_finite() { left } else { 0.0 };
        let r = if right.is_finite() { right } else { 0.0 };
        let x = (l - r) * FRAC_1_SQRT_2;
        let y = (l + r) * FRAC_1_SQRT_2;

        let radius = ops::sqrt(x * x + y * y);
        if radius > self.peak_radius {
            self.peak_radius = radius;
        }
        self.sum_radius_sq += f64::from(radius) * f64::from(radius);
        self.sample_count = self.sample_count.saturating_add(1);

        self.counter += 1;
        if self.counter >= self.decimation {
            self.counter = 0;
            self.ring[self.write] = GoniometerPoint {
                x: flush_denormal(x),
                y: flush_denormal(y),
            };
            self.write = (self.write + 1) % self.ring.len();
            if self.len < self.ring.len() {
                self.len += 1;
            }
        }
    }

    /// Copies the buffered points, oldest first, into `out` and returns the
    /// number written (the minimum of [`len`](Self::len) and `out.len()`).
    /// Allocation-free.
    pub fn points_ordered(&self, out: &mut [GoniometerPoint]) -> usize {
        let count = self.len.min(out.len());
        if count == 0 {
            return 0;
        }
        // The oldest valid point sits `len` slots behind the write cursor.
        let start = (self.write + self.ring.len() - self.len) % self.ring.len();
        for (i, slot) in out[..count].iter_mut().enumerate() {
            *slot = self.ring[(start + i) % self.ring.len()];
        }
        count
    }

    /// Clears the display ring and all accumulators.
    pub fn reset(&mut self) {
        for p in &mut self.ring {
            *p = GoniometerPoint::default();
        }
        self.write = 0;
        self.len = 0;
        self.counter = 0;
        self.peak_radius = 0.0;
        self.sum_radius_sq = 0.0;
        self.sample_count = 0;
    }
}

/// An [`AudioNode`] goniometer tap: it passes its stereo input through
/// unchanged while feeding every sample pair to an embedded [`Goniometer`].
///
/// A mono input is read as a centred signal (`L == R`), tracing the vertical
/// axis; inputs with more than two channels are read on their first two.
#[derive(Clone, Debug)]
pub struct GoniometerNode {
    scope: Goniometer,
}

impl GoniometerNode {
    /// Creates a goniometer tap with the given display ring capacity and
    /// decimation factor.
    #[must_use]
    pub fn new(capacity: usize, decimation: u32) -> Self {
        Self {
            scope: Goniometer::new(capacity, decimation),
        }
    }

    /// Borrows the underlying coordinate generator (to read points or stats).
    #[must_use]
    pub fn scope(&self) -> &Goniometer {
        &self.scope
    }

    /// Mutably borrows the underlying coordinate generator (to retune or reset).
    pub fn scope_mut(&mut self) -> &mut Goniometer {
        &mut self.scope
    }
}

impl AudioNode for GoniometerNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        let out_channels = output.channels();
        let frames = output.active_frames();

        // Pass the signal through unchanged.
        let copy_channels = out_channels.min(input.channels());
        for ch in 0..copy_channels {
            let src = input.channel(ch);
            let dst = output.channel_mut(ch);
            dst[..frames].copy_from_slice(&src[..frames]);
        }

        // Feed the stereo pair. A mono input is read as a centred signal.
        if input.channels() >= 2 {
            let left = input.channel(0);
            let right = input.channel(1);
            for n in 0..frames {
                self.scope.feed_sample(left[n], right[n]);
            }
        } else if input.channels() == 1 {
            let mono = input.channel(0);
            for &x in &mono[..frames] {
                self.scope.feed_sample(x, x);
            }
        }
    }

    fn reset(&mut self) {
        self.scope.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{AudioBuffer, ChannelLayout};
    use alloc::vec::Vec;
    use core::f32::consts::FRAC_1_SQRT_2;

    const SR: u32 = 48_000;

    fn ctx(frames: usize) -> RenderContext {
        RenderContext {
            sample_rate: SR,
            frames,
            playhead: 0,
        }
    }

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn mono_signal_maps_to_vertical_axis() {
        let mut g = Goniometer::new(16, 1);
        g.feed_sample(0.5, 0.5);
        let mut pts = [GoniometerPoint::default(); 4];
        let n = g.points_ordered(&mut pts);
        assert_eq!(n, 1);
        assert!(approx(pts[0].x, 0.0, 1e-7), "x {}", pts[0].x);
        assert!(approx(pts[0].y, 0.5 * 2.0 * FRAC_1_SQRT_2, 1e-6), "y {}", pts[0].y);
    }

    #[test]
    fn antiphase_signal_maps_to_horizontal_axis() {
        let mut g = Goniometer::new(16, 1);
        g.feed_sample(0.5, -0.5);
        let mut pts = [GoniometerPoint::default(); 4];
        let n = g.points_ordered(&mut pts);
        assert_eq!(n, 1);
        assert!(approx(pts[0].y, 0.0, 1e-7), "y {}", pts[0].y);
        assert!(approx(pts[0].x, 0.5 * 2.0 * FRAC_1_SQRT_2, 1e-6), "x {}", pts[0].x);
    }

    #[test]
    fn rotation_preserves_energy() {
        // x^2 + y^2 should equal L^2 + R^2 for the orthonormal rotation.
        let mut g = Goniometer::new(4, 1);
        let (l, r) = (0.3, -0.7);
        g.feed_sample(l, r);
        let mut pts = [GoniometerPoint::default(); 1];
        g.points_ordered(&mut pts);
        let rotated = pts[0].x * pts[0].x + pts[0].y * pts[0].y;
        assert!(approx(rotated, l * l + r * r, 1e-6), "energy {rotated}");
    }

    #[test]
    fn decimation_keeps_every_nth_pair() {
        let mut g = Goniometer::new(32, 4);
        for i in 0..16 {
            g.feed_sample(i as Sample, i as Sample);
        }
        // 16 pairs, keep every 4th -> 4 points retained.
        assert_eq!(g.len(), 4);
    }

    #[test]
    fn ring_overwrites_oldest_when_full() {
        let mut g = Goniometer::new(3, 1);
        for i in 0..5 {
            g.feed_sample(i as Sample, i as Sample);
        }
        assert_eq!(g.len(), 3);
        let mut pts = [GoniometerPoint::default(); 3];
        let n = g.points_ordered(&mut pts);
        assert_eq!(n, 3);
        // Oldest retained pair is i=2, then 3, then 4; y = 2*i*FRAC_1_SQRT_2.
        for (k, p) in pts.iter().enumerate() {
            let i = (k + 2) as Sample;
            assert!(approx(p.y, 2.0 * i * FRAC_1_SQRT_2, 1e-5), "point {k} y {}", p.y);
        }
    }

    #[test]
    fn points_ordered_truncates_to_output_slice() {
        let mut g = Goniometer::new(8, 1);
        for i in 0..8 {
            g.feed_sample(i as Sample, 0.0);
        }
        let mut small = [GoniometerPoint::default(); 3];
        let n = g.points_ordered(&mut small);
        assert_eq!(n, 3);
    }

    #[test]
    fn peak_radius_tracks_largest_magnitude() {
        let mut g = Goniometer::new(8, 1);
        g.feed_sample(0.2, 0.2);
        g.feed_sample(1.0, 1.0);
        g.feed_sample(0.1, 0.1);
        // Largest pair is (1,1): radius = sqrt(2).
        assert!(approx(g.peak_radius(), ops::sqrt(2.0), 1e-6), "peak {}", g.peak_radius());
    }

    #[test]
    fn stats_report_rms_and_count() {
        let mut g = Goniometer::new(8, 1);
        g.feed_sample(1.0, 1.0); // radius sqrt(2)
        g.feed_sample(0.0, 0.0); // radius 0
        let s = g.stats();
        assert_eq!(s.sample_count, 2);
        // rms = sqrt((2 + 0)/2) = 1.
        assert!(approx(s.rms_radius, 1.0, 1e-6), "rms {}", s.rms_radius);
        assert!(approx(s.peak_radius, ops::sqrt(2.0), 1e-6));
    }

    #[test]
    fn non_finite_input_is_treated_as_silence() {
        let mut g = Goniometer::new(8, 1);
        g.feed_sample(Sample::NAN, Sample::INFINITY);
        let mut pts = [GoniometerPoint::default(); 1];
        g.points_ordered(&mut pts);
        assert_eq!(pts[0], GoniometerPoint { x: 0.0, y: 0.0 });
        assert_eq!(g.peak_radius(), 0.0);
        assert!(g.stats().rms_radius.is_finite());
    }

    #[test]
    fn reset_clears_ring_and_accumulators() {
        let mut g = Goniometer::new(8, 1);
        for i in 0..8 {
            g.feed_sample(i as Sample, i as Sample);
        }
        g.reset();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert_eq!(g.peak_radius(), 0.0);
        assert_eq!(g.stats().sample_count, 0);
    }

    #[test]
    fn capacity_and_decimation_are_clamped() {
        let g = Goniometer::new(0, 0);
        assert_eq!(g.capacity(), MIN_POINT_CAPACITY);
        assert_eq!(g.decimation(), 1);
    }

    #[test]
    fn set_decimation_resets_phase() {
        let mut g = Goniometer::new(16, 1);
        g.feed_sample(1.0, 1.0);
        g.set_decimation(2);
        assert_eq!(g.decimation(), 2);
        g.feed_sample(1.0, 1.0); // counter 1, not kept
        assert_eq!(g.len(), 1); // only the first point retained
        g.feed_sample(1.0, 1.0); // counter 2 -> kept
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn empty_output_slice_writes_nothing() {
        let mut g = Goniometer::new(4, 1);
        g.feed_sample(0.5, 0.5);
        let mut empty: [GoniometerPoint; 0] = [];
        assert_eq!(g.points_ordered(&mut empty), 0);
    }

    #[test]
    fn node_passes_signal_through_unchanged() {
        let mut node = GoniometerNode::new(64, 1);
        let frames = 32;
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, frames);
        input.set_active_frames(frames);
        for n in 0..frames {
            input.channel_mut(0)[n] = 0.3 * n as Sample;
            input.channel_mut(1)[n] = -0.2 * n as Sample;
        }
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, frames);
        out.set_active_frames(frames);
        let inputs = [input.clone()];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let [o] = outputs;
        for ch in 0..2 {
            assert_eq!(o.channel(ch), input.channel(ch));
        }
        assert_eq!(node.scope().len(), frames);
    }

    #[test]
    fn node_reads_mono_as_centred() {
        let mut node = GoniometerNode::new(16, 1);
        let frames = 4;
        let mut input = AudioBuffer::new(ChannelLayout::Mono, frames);
        input.set_active_frames(frames);
        for n in 0..frames {
            input.channel_mut(0)[n] = 0.5;
        }
        let mut out = AudioBuffer::new(ChannelLayout::Mono, frames);
        out.set_active_frames(frames);
        let inputs = [input];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(frames), &mut io);
        let mut pts = [GoniometerPoint::default(); 4];
        let n = node.scope().points_ordered(&mut pts);
        assert_eq!(n, 4);
        for p in &pts {
            assert!(approx(p.x, 0.0, 1e-7), "mono should be vertical, x {}", p.x);
        }
    }

    #[test]
    fn node_reset_clears_scope() {
        let mut node = GoniometerNode::new(16, 1);
        node.scope_mut().feed_sample(1.0, 1.0);
        node.reset();
        assert!(node.scope().is_empty());
    }

    #[test]
    fn node_zero_frames_is_safe() {
        let mut node = GoniometerNode::new(16, 1);
        let mut input = AudioBuffer::new(ChannelLayout::Stereo, 8);
        input.set_active_frames(0);
        let mut out = AudioBuffer::new(ChannelLayout::Stereo, 8);
        out.set_active_frames(0);
        let inputs = [input];
        let mut outputs = [out];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx(0), &mut io);
        let [o] = outputs;
        assert_eq!(o.active_frames(), 0);
        assert!(node.scope().is_empty());
    }
}
