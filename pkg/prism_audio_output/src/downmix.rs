//! Standard multichannel downmix matrices (`ITU-R` `BS.775`).
//!
//! A [`DownmixMatrix`] is a small, dense coefficient table that folds one
//! [`ChannelLayout`] into another by a per-output-channel weighted sum of the
//! input channels. The coefficients follow the public `ITU-R` `BS.775`
//! recommendation: the centre and surround feeds are mixed into the front
//! stereo pair at the canonical `-3 dB` level (`1/sqrt(2) ~= 0.707`), while the
//! `LFE` channel is dropped by default (its contribution is `-inf dB`) and only
//! folded in when the caller explicitly opts in.
//!
//! The matrix is built once (allocating a `channels * channels` table) in
//! [`DownmixMatrix::new`]; [`DownmixMatrix::apply`] and the graph-facing
//! [`DownmixNode::process`] are allocation-, lock-, and panic-free.
//!
//! # Supported conversions
//!
//! | Input          | Output         |
//! |----------------|----------------|
//! | `Surround5_1`  | `Stereo`       |
//! | `Surround5_1`  | `Mono`         |
//! | `Surround7_1`  | `Surround5_1`  |
//! | `Surround7_1`  | `Stereo`       |
//! | `Stereo`       | `Mono`         |
//! | `Quad`         | `Stereo`       |
//! | any layout     | itself (unity) |
//!
//! Any other pair is rejected by [`DownmixMatrix::new`] (returns `None`).
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The downmix
//! coefficients are the publicly documented `ITU-R` `BS.775` / `ATSC` `A/52`
//! equations; the matrix bookkeeping here is an original implementation on top
//! of this crate's own buffers. It is pure classic DSP with no AI/ML.
//!
//! # Relationship
//!
//! - Reuses [`AudioBuffer`] / [`ChannelLayout`] from
//!   [`prism_audio_core`](prism_audio_core::buffer) as the sample container and
//!   speaker-map descriptor; it defines no buffer type of its own.
//!   `LFE`-inclusion gain is converted with
//!   [`db_to_linear`](prism_audio_core::db_to_linear).
//! - Implements the [`AudioNode`] contract from
//!   [`prism_audio_core::graph`] via [`DownmixNode`] for in-graph use.
//! - Consumed by [`crate::profiles`], whose delivery profiles name a downmix
//!   target layout.

use alloc::vec;
use alloc::vec::Vec;

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::db_to_linear;
use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::math::Sample;

/// Canonical `-3 dB` downmix coefficient (`1/sqrt(2)`), used for the centre and
/// surround feeds per `ITU-R` `BS.775`.
pub const MINUS_3DB: Sample = core::f32::consts::FRAC_1_SQRT_2;

/// Canonical `-6 dB` downmix coefficient (`0.5`), used when a feed is averaged
/// into two output channels (e.g. the front pair collapsing to mono).
pub const MINUS_6DB: Sample = 0.5;

/// Canonical `-4.5 dB` surround coefficient (`0.5 * 1/sqrt(2) ~= 0.3536`), the
/// level a single surround receives when the stereo downmix is itself averaged
/// to mono.
pub const SURROUND_TO_MONO: Sample = 0.5 * core::f32::consts::FRAC_1_SQRT_2;

/// Options controlling how the optional `LFE` channel is treated during a
/// downmix to a layout that has **no** `LFE` channel of its own.
///
/// `ITU-R` `BS.775` drops the `LFE` by default (its level is effectively
/// `-inf dB`); some delivery targets instead fold a reduced-level copy of the
/// `LFE` into the front channels, which [`include_lfe`](Self::include_lfe)
/// enables.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DownmixOptions {
    /// When `true`, the `LFE` channel is mixed into the surviving front
    /// channels at [`lfe_gain_db`](Self::lfe_gain_db). When `false` (the
    /// default), the `LFE` is dropped entirely.
    pub include_lfe: bool,
    /// Gain in decibels applied to the folded-in `LFE` channel. Ignored unless
    /// [`include_lfe`](Self::include_lfe) is `true`. `0 dB` folds the `LFE` in
    /// at unity.
    pub lfe_gain_db: Sample,
}

impl Default for DownmixOptions {
    #[inline]
    fn default() -> Self {
        Self {
            include_lfe: false,
            lfe_gain_db: 0.0,
        }
    }
}

impl DownmixOptions {
    /// Returns the options with every field clamped into a sane range and any
    /// non-finite field replaced by its default.
    #[must_use]
    pub fn sanitised(self) -> Self {
        let gain = if self.lfe_gain_db.is_finite() {
            self.lfe_gain_db.clamp(-60.0, 12.0)
        } else {
            0.0
        };
        Self {
            include_lfe: self.include_lfe,
            lfe_gain_db: gain,
        }
    }

    /// Returns the linear `LFE` fold-in gain, or `0.0` when the `LFE` is
    /// dropped.
    #[inline]
    #[must_use]
    fn lfe_linear(self) -> Sample {
        if self.include_lfe {
            db_to_linear(self.sanitised().lfe_gain_db)
        } else {
            0.0
        }
    }
}

/// A dense downmix coefficient matrix mapping an input [`ChannelLayout`] to an
/// output [`ChannelLayout`].
///
/// The coefficients are stored row-major (`out_channels` rows, `in_channels`
/// columns): output channel `o` is `sum_i coeffs[o * in_channels + i] *
/// input[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DownmixMatrix {
    input: ChannelLayout,
    output: ChannelLayout,
    in_channels: usize,
    out_channels: usize,
    /// Row-major `out_channels * in_channels` coefficient table.
    coeffs: Vec<Sample>,
}

impl DownmixMatrix {
    /// Builds the standard downmix matrix converting `input` into `output`.
    ///
    /// Returns `None` when the pair is not one of the supported conversions
    /// (see the module table). An `input == output` request yields the identity
    /// matrix. `opts` only affects conversions that discard an `LFE` channel.
    #[must_use]
    pub fn new(input: ChannelLayout, output: ChannelLayout, opts: DownmixOptions) -> Option<Self> {
        let in_channels = input.channel_count();
        let out_channels = output.channel_count();
        let mut coeffs = vec![0.0; out_channels * in_channels];
        let lfe = opts.lfe_linear();

        let set = |coeffs: &mut [Sample], o: usize, i: usize, c: Sample| {
            coeffs[o * in_channels + i] = c;
        };

        if input == output {
            // Identity: pass every channel through at unity.
            for ch in 0..in_channels {
                set(&mut coeffs, ch, ch, 1.0);
            }
            return Some(Self {
                input,
                output,
                in_channels,
                out_channels,
                coeffs,
            });
        }

        match (input, output) {
            // 5.1 (FL FR C LFE SL SR) -> stereo (L R).
            (ChannelLayout::Surround5_1, ChannelLayout::Stereo) => {
                set(&mut coeffs, 0, 0, 1.0); // L <- FL
                set(&mut coeffs, 0, 2, MINUS_3DB); // L <- C
                set(&mut coeffs, 0, 4, MINUS_3DB); // L <- SL
                set(&mut coeffs, 1, 1, 1.0); // R <- FR
                set(&mut coeffs, 1, 2, MINUS_3DB); // R <- C
                set(&mut coeffs, 1, 5, MINUS_3DB); // R <- SR
                set(&mut coeffs, 0, 3, lfe); // L <- LFE
                set(&mut coeffs, 1, 3, lfe); // R <- LFE
            }
            // 5.1 (FL FR C LFE SL SR) -> mono: average of the stereo downmix.
            (ChannelLayout::Surround5_1, ChannelLayout::Mono) => {
                set(&mut coeffs, 0, 0, MINUS_6DB); // M <- FL
                set(&mut coeffs, 0, 1, MINUS_6DB); // M <- FR
                set(&mut coeffs, 0, 2, MINUS_3DB); // M <- C
                set(&mut coeffs, 0, 4, SURROUND_TO_MONO); // M <- SL
                set(&mut coeffs, 0, 5, SURROUND_TO_MONO); // M <- SR
                set(&mut coeffs, 0, 3, lfe); // M <- LFE
            }
            // 7.1 (FL FR C LFE SL SR RL RR) -> 5.1: fold rears into sides.
            (ChannelLayout::Surround7_1, ChannelLayout::Surround5_1) => {
                set(&mut coeffs, 0, 0, 1.0); // FL
                set(&mut coeffs, 1, 1, 1.0); // FR
                set(&mut coeffs, 2, 2, 1.0); // C
                set(&mut coeffs, 3, 3, 1.0); // LFE passes through structurally
                set(&mut coeffs, 4, 4, MINUS_3DB); // SL <- SL
                set(&mut coeffs, 4, 6, MINUS_3DB); // SL <- RL
                set(&mut coeffs, 5, 5, MINUS_3DB); // SR <- SR
                set(&mut coeffs, 5, 7, MINUS_3DB); // SR <- RR
            }
            // 7.1 -> stereo: every centre/surround feed at -3 dB into the pair.
            (ChannelLayout::Surround7_1, ChannelLayout::Stereo) => {
                set(&mut coeffs, 0, 0, 1.0); // L <- FL
                set(&mut coeffs, 0, 2, MINUS_3DB); // L <- C
                set(&mut coeffs, 0, 4, MINUS_3DB); // L <- SL
                set(&mut coeffs, 0, 6, MINUS_3DB); // L <- RL
                set(&mut coeffs, 1, 1, 1.0); // R <- FR
                set(&mut coeffs, 1, 2, MINUS_3DB); // R <- C
                set(&mut coeffs, 1, 5, MINUS_3DB); // R <- SR
                set(&mut coeffs, 1, 7, MINUS_3DB); // R <- RR
                set(&mut coeffs, 0, 3, lfe); // L <- LFE
                set(&mut coeffs, 1, 3, lfe); // R <- LFE
            }
            // Stereo (L R) -> mono: level-preserving average.
            (ChannelLayout::Stereo, ChannelLayout::Mono) => {
                set(&mut coeffs, 0, 0, MINUS_6DB); // M <- L
                set(&mut coeffs, 0, 1, MINUS_6DB); // M <- R
            }
            // Quad (FL FR SL SR) -> stereo: surrounds at -3 dB into the pair.
            (ChannelLayout::Quad, ChannelLayout::Stereo) => {
                set(&mut coeffs, 0, 0, 1.0); // L <- FL
                set(&mut coeffs, 0, 2, MINUS_3DB); // L <- SL
                set(&mut coeffs, 1, 1, 1.0); // R <- FR
                set(&mut coeffs, 1, 3, MINUS_3DB); // R <- SR
            }
            _ => return None,
        }

        Some(Self {
            input,
            output,
            in_channels,
            out_channels,
            coeffs,
        })
    }

    /// The input layout this matrix consumes.
    #[inline]
    #[must_use]
    pub fn input_layout(&self) -> ChannelLayout {
        self.input
    }

    /// The output layout this matrix produces.
    #[inline]
    #[must_use]
    pub fn output_layout(&self) -> ChannelLayout {
        self.output
    }

    /// Returns the coefficient weighting input channel `i` into output channel
    /// `o`, or `0.0` if either index is out of range.
    #[inline]
    #[must_use]
    pub fn coeff(&self, o: usize, i: usize) -> Sample {
        if o < self.out_channels && i < self.in_channels {
            self.coeffs[o * self.in_channels + i]
        } else {
            0.0
        }
    }

    /// Applies the matrix, writing the downmixed signal from `input` into
    /// `output`.
    ///
    /// Real-time safe: no allocation, no locks, cannot panic. Channel counts
    /// that disagree with the configured layouts are handled by processing only
    /// the overlapping channels, and the shorter of the two active frame counts
    /// is used.
    pub fn apply(&self, input: &AudioBuffer, output: &mut AudioBuffer) {
        let in_ch = self.in_channels.min(input.channels());
        let out_ch = self.out_channels.min(output.channels());
        let frames = input.active_frames().min(output.active_frames());

        // Clear only the output channels we are about to write.
        for o in 0..out_ch {
            for s in output.channel_mut(o)[..frames].iter_mut() {
                *s = 0.0;
            }
        }

        // Accumulate every input contribution into its output channels.
        for i in 0..in_ch {
            for o in 0..out_ch {
                let c = self.coeffs[o * self.in_channels + i];
                // Downmix coefficients are fixed finite constants; a zero
                // coefficient contributes nothing, so skip the inner loop.
                if c == 0.0 {
                    continue;
                }
                let (src, dst) = (input.channel(i), &mut output.channel_mut(o)[..frames]);
                for (d, s) in dst.iter_mut().zip(&src[..frames]) {
                    *d += c * *s;
                }
            }
        }
    }
}

/// A graph node wrapping a [`DownmixMatrix`] for in-graph delivery routing.
///
/// Input port `0` carries the source layout; output port `0` carries the
/// downmix target layout. The node is stateless beyond its matrix, so
/// [`AudioNode::reset`] is a no-op and latency is zero.
#[derive(Debug, Clone)]
pub struct DownmixNode {
    matrix: DownmixMatrix,
}

impl DownmixNode {
    /// Builds a downmix node from a prepared [`DownmixMatrix`].
    #[inline]
    #[must_use]
    pub fn new(matrix: DownmixMatrix) -> Self {
        Self { matrix }
    }

    /// Builds a downmix node directly from a layout pair, returning `None` for
    /// unsupported conversions (see [`DownmixMatrix::new`]).
    #[must_use]
    pub fn from_layouts(
        input: ChannelLayout,
        output: ChannelLayout,
        opts: DownmixOptions,
    ) -> Option<Self> {
        DownmixMatrix::new(input, output, opts).map(Self::new)
    }

    /// Immutable access to the underlying matrix.
    #[inline]
    #[must_use]
    pub fn matrix(&self) -> &DownmixMatrix {
        &self.matrix
    }
}

impl AudioNode for DownmixNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);
        self.matrix.apply(input, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-6;

    fn mono_const(layout: ChannelLayout, values: &[Sample], frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(layout, frames);
        for (ch, &v) in values.iter().enumerate() {
            for s in buf.channel_mut(ch) {
                *s = v;
            }
        }
        buf
    }

    #[test]
    fn unsupported_pair_is_rejected() {
        assert!(DownmixMatrix::new(
            ChannelLayout::Mono,
            ChannelLayout::Surround5_1,
            DownmixOptions::default()
        )
        .is_none());
    }

    #[test]
    fn identity_passes_through() {
        let m = DownmixMatrix::new(
            ChannelLayout::Stereo,
            ChannelLayout::Stereo,
            DownmixOptions::default(),
        )
        .unwrap();
        let input = mono_const(ChannelLayout::Stereo, &[0.3, -0.7], 16);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 16);
        m.apply(&input, &mut output);
        assert!((output.channel(0)[0] - 0.3).abs() < EPS);
        assert!((output.channel(1)[0] + 0.7).abs() < EPS);
    }

    #[test]
    fn surround_5_1_to_stereo_uses_minus_3db() {
        let m = DownmixMatrix::new(
            ChannelLayout::Surround5_1,
            ChannelLayout::Stereo,
            DownmixOptions::default(),
        )
        .unwrap();
        // FL FR C LFE SL SR
        let input = mono_const(ChannelLayout::Surround5_1, &[1.0, 2.0, 1.0, 9.0, 1.0, 2.0], 8);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 8);
        m.apply(&input, &mut output);
        // L = FL + 0.707*C + 0.707*SL ; LFE dropped (coeff 9.0 ignored).
        let expect_l = 1.0 + MINUS_3DB * 1.0 + MINUS_3DB * 1.0;
        let expect_r = 2.0 + MINUS_3DB * 1.0 + MINUS_3DB * 2.0;
        assert!((output.channel(0)[0] - expect_l).abs() < EPS);
        assert!((output.channel(1)[0] - expect_r).abs() < EPS);
    }

    #[test]
    fn lfe_dropped_by_default_but_included_on_request() {
        let dropped = DownmixMatrix::new(
            ChannelLayout::Surround5_1,
            ChannelLayout::Stereo,
            DownmixOptions::default(),
        )
        .unwrap();
        assert!(dropped.coeff(0, 3).abs() < EPS, "LFE dropped by default");

        let included = DownmixMatrix::new(
            ChannelLayout::Surround5_1,
            ChannelLayout::Stereo,
            DownmixOptions {
                include_lfe: true,
                lfe_gain_db: 0.0,
            },
        )
        .unwrap();
        assert!(
            (included.coeff(0, 3) - 1.0).abs() < EPS,
            "0 dB LFE folds in at unity"
        );
        assert!((included.coeff(1, 3) - 1.0).abs() < EPS);
    }

    #[test]
    fn surround_5_1_to_mono_center_is_minus_3db() {
        let m = DownmixMatrix::new(
            ChannelLayout::Surround5_1,
            ChannelLayout::Mono,
            DownmixOptions::default(),
        )
        .unwrap();
        // Centre weight must be the canonical -3 dB, fronts -6 dB.
        assert!((m.coeff(0, 2) - MINUS_3DB).abs() < EPS);
        assert!((m.coeff(0, 0) - MINUS_6DB).abs() < EPS);
        assert!((m.coeff(0, 1) - MINUS_6DB).abs() < EPS);
        assert!((m.coeff(0, 4) - SURROUND_TO_MONO).abs() < EPS);
    }

    #[test]
    fn seven_one_to_five_one_folds_rears() {
        let m = DownmixMatrix::new(
            ChannelLayout::Surround7_1,
            ChannelLayout::Surround5_1,
            DownmixOptions::default(),
        )
        .unwrap();
        // Fronts/centre/LFE pass through at unity.
        assert!((m.coeff(0, 0) - 1.0).abs() < EPS);
        assert!((m.coeff(2, 2) - 1.0).abs() < EPS);
        assert!((m.coeff(3, 3) - 1.0).abs() < EPS);
        // SL combines SL + RL at -3 dB each.
        assert!((m.coeff(4, 4) - MINUS_3DB).abs() < EPS);
        assert!((m.coeff(4, 6) - MINUS_3DB).abs() < EPS);
    }

    #[test]
    fn stereo_to_mono_preserves_correlated_level() {
        let m = DownmixMatrix::new(
            ChannelLayout::Stereo,
            ChannelLayout::Mono,
            DownmixOptions::default(),
        )
        .unwrap();
        let input = mono_const(ChannelLayout::Stereo, &[0.8, 0.8], 8);
        let mut output = AudioBuffer::new(ChannelLayout::Mono, 8);
        m.apply(&input, &mut output);
        // Correlated L == R collapses to the same level (0.5 + 0.5).
        assert!((output.channel(0)[0] - 0.8).abs() < EPS);
    }

    #[test]
    fn quad_to_stereo_surround_level() {
        let m = DownmixMatrix::new(
            ChannelLayout::Quad,
            ChannelLayout::Stereo,
            DownmixOptions::default(),
        )
        .unwrap();
        // FL FR SL SR
        let input = mono_const(ChannelLayout::Quad, &[1.0, 1.0, 1.0, 1.0], 8);
        let mut output = AudioBuffer::new(ChannelLayout::Stereo, 8);
        m.apply(&input, &mut output);
        let expect = 1.0 + MINUS_3DB;
        assert!((output.channel(0)[0] - expect).abs() < EPS);
        assert!((output.channel(1)[0] - expect).abs() < EPS);
    }

    #[test]
    fn apply_is_deterministic() {
        let m = DownmixMatrix::new(
            ChannelLayout::Surround7_1,
            ChannelLayout::Stereo,
            DownmixOptions::default(),
        )
        .unwrap();
        let input = mono_const(
            ChannelLayout::Surround7_1,
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            32,
        );
        let mut a = AudioBuffer::new(ChannelLayout::Stereo, 32);
        let mut b = AudioBuffer::new(ChannelLayout::Stereo, 32);
        m.apply(&input, &mut a);
        m.apply(&input, &mut b);
        for ch in 0..2 {
            for (x, y) in a.channel(ch).iter().zip(b.channel(ch)) {
                assert!((x - y).abs() < EPS, "repeated apply must match bit-for-bit");
            }
        }
    }

    #[test]
    fn node_matches_matrix() {
        let matrix = DownmixMatrix::new(
            ChannelLayout::Surround5_1,
            ChannelLayout::Stereo,
            DownmixOptions::default(),
        )
        .unwrap();
        let mut node = DownmixNode::new(matrix.clone());
        let input = mono_const(ChannelLayout::Surround5_1, &[1.0, 2.0, 0.5, 0.0, 0.3, 0.4], 8);
        let mut via_matrix = AudioBuffer::new(ChannelLayout::Stereo, 8);
        matrix.apply(&input, &mut via_matrix);

        let mut via_node = AudioBuffer::new(ChannelLayout::Stereo, 8);
        let inputs = [input];
        let outputs_slot = AudioBuffer::new(ChannelLayout::Stereo, 8);
        let mut outputs = [outputs_slot];
        let ctx = RenderContext {
            sample_rate: 48_000,
            frames: 8,
            playhead: 0,
        };
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        via_node.copy_from(io.output(0));
        for ch in 0..2 {
            for (x, y) in via_matrix.channel(ch).iter().zip(via_node.channel(ch)) {
                assert!((x - y).abs() < EPS);
            }
        }
    }

    #[test]
    fn options_sanitised_clamps_nonfinite() {
        let o = DownmixOptions {
            include_lfe: true,
            lfe_gain_db: Sample::NAN,
        }
        .sanitised();
        assert!(o.lfe_gain_db.abs() < EPS);
        let hi = DownmixOptions {
            include_lfe: true,
            lfe_gain_db: 999.0,
        }
        .sanitised();
        assert!(hi.lfe_gain_db <= 12.0 + EPS);
    }
}
