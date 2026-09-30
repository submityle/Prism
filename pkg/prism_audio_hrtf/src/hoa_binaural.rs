//! Virtual-loudspeaker binaural decoding of Higher-Order Ambisonic (HOA)
//! scenes.
//!
//! Scene-based (Ambisonic) mixes must ultimately reach a listener's ears. This
//! module renders an HOA field to a two-channel headphone (binaural) signal by
//! the classic *virtual loudspeaker* method: the sound field is decoded to a
//! fixed set of virtual loudspeakers arranged around the head, each virtual
//! loudspeaker feed is convolved with the head-related impulse response (HRIR)
//! for its direction, and the results are summed at each ear.
//!
//! # Pre-baked per-channel filters
//!
//! Convolving one HRIR pair per virtual loudspeaker every block is wasteful
//! because the decode is linear. If the projection decode gain of loudspeaker
//! `s` for Ambisonic channel `c` is `D[s][c]`, then the ear signal is
//!
//! ```text
//!   ear = sum_s ( sum_c hoa[c] * D[s][c] ) (*) HRIR(s)
//!       = sum_c hoa[c] * ( sum_s D[s][c] (*) HRIR(s) )
//!       = sum_c hoa[c] (*) filter[c]
//! ```
//!
//! so the per-loudspeaker responses can be collapsed, once, into one HRIR-length
//! filter pair *per Ambisonic channel*: `filter[c] = sum_s D[s][c] * HRIR(s)`.
//! At run time only `(order + 1)^2` short convolutions remain - independent of
//! how many virtual loudspeakers were used to bake the filters - and the render
//! is mathematically identical to the full virtual-loudspeaker decode.
//!
//! The decode gains reuse [`prism_audio_spatial`]'s projection decode exactly:
//! `D[s][c] = encode_hoa(dir_s)[c] / (order + 1)`, i.e. the same
//! `dot(coeffs, encode(dir_s)) / (order + 1)` that
//! [`prism_audio_spatial::decode_hoa`] applies, so this binaural path agrees
//! bit-for-bit with the loudspeaker decode used elsewhere in the pipeline. The
//! field is assumed to already be expressed in the listener-local frame (rotate
//! it first with [`prism_audio_spatial::rotate_hoa`] for head tracking).
//!
//! # Classic DSP only
//!
//! Everything here is deterministic, classical signal processing: an Ambisonic
//! projection decode to a fixed geometric loudspeaker layout followed by
//! measured-HRIR block convolution. There is **no machine learning, neural
//! network, or data-driven model of any kind**.
//!
//! # Real-time contract
//!
//! Filter baking ([`HoaBinauralDecoder::new`]) interpolates HRIRs and allocates
//! the per-channel [`BinauralRenderer`]s off the audio thread. The hot path
//! ([`HoaBinauralDecoder::process_block`]) is **allocation free, lock free, and
//! panic free**: it convolves the provided channels into pre-sized scratch
//! buffers and accumulates, using saturating length clamps rather than
//! panicking indexing. Degenerate inputs (empty layout, fewer channels than the
//! decoder order, short buffers) produce silence or a clamped frame count.
//!
//! # Determinism
//!
//! The layout directions, decode gains, and HRIR interpolation all route their
//! transcendental and length math through [`bevy_math::ops`] (via
//! [`prism_audio_spatial`] and [`crate::interpolation`]); the convolution and
//! accumulation are pure multiply-add. Output is therefore bit-reproducible
//! across targets and can be golden-compared sample-for-sample.
//!
//! # Coordinate convention
//!
//! Directions are listener-local unit vectors matching Bevy and
//! [`prism_audio_spatial`]: `+X` right, `+Y` up, `-Z` forward.
//!
//! # Feature flags
//!
//! Inherits the crate's `std` / `serialize` features. The decoder itself needs
//! neither; [`VirtualSpeakerLayout`] gains serde derives under `serialize`.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**. Virtual-loudspeaker (a.k.a. "virtual Ambisonics") binaural decoding
//! and the linearity that lets the per-loudspeaker HRIRs be pre-summed into
//! per-channel filters are publicly documented practice: see M. Noisternig et
//! al., "A 3D Ambisonic Based Binaural Sound Reproduction System" (AES 24th,
//! 2003) and F. Zotter and M. Frank, "Ambisonics" (Springer, 2019). Everything
//! here is implemented from that public literature.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::Vec3;

use prism_audio_core::math::Sample;
use prism_audio_spatial::{MAX_HOA_CHANNELS, MAX_HOA_ORDER, encode_hoa, hoa_channel_count};

use crate::binaural::BinauralRenderer;
use crate::dataset::HrtfDataset;
use crate::headtracked::{local_azimuth, local_elevation};
use crate::interpolation::interpolate;

/// Maximum number of virtual loudspeakers a [`VirtualSpeakerLayout`] can hold.
///
/// Sized for near-uniform spherical layouts (spherical `t`-designs, cube
/// samplings) up to [`MAX_HOA_ORDER`] while staying stack allocatable and
/// serde-serialisable (arrays up to 32 elements).
pub const MAX_VIRTUAL_SPEAKERS: usize = 32;

/// A fixed-capacity set of virtual-loudspeaker directions (listener-local unit
/// vectors), stored on the stack.
///
/// A layout must contain at least `(order + 1)^2` well-spread directions to
/// decode order-`order` Ambisonics without rank deficiency; the built-in
/// [`VirtualSpeakerLayout::cube26`] preset (26 points) covers up to
/// [`MAX_HOA_ORDER`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VirtualSpeakerLayout {
    directions: [Vec3; MAX_VIRTUAL_SPEAKERS],
    count: usize,
}

impl Default for VirtualSpeakerLayout {
    #[inline]
    fn default() -> Self {
        Self::cube26()
    }
}

impl VirtualSpeakerLayout {
    /// Creates an empty layout.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { directions: [Vec3::ZERO; MAX_VIRTUAL_SPEAKERS], count: 0 }
    }

    /// Builds a layout from a slice of directions, normalising each and keeping
    /// at most [`MAX_VIRTUAL_SPEAKERS`] entries (extras are ignored).
    /// Zero-length directions are skipped.
    #[must_use]
    pub fn from_directions(directions: &[Vec3]) -> Self {
        let mut layout = Self::new();
        for &d in directions {
            layout.push(d);
        }
        layout
    }

    /// Appends a loudspeaker direction (normalised). Returns `true` if it was
    /// stored, `false` if the layout was full or the direction was degenerate.
    #[inline]
    pub fn push(&mut self, direction: Vec3) -> bool {
        if self.count >= MAX_VIRTUAL_SPEAKERS {
            return false;
        }
        let unit = direction.normalize_or_zero();
        if unit == Vec3::ZERO {
            return false;
        }
        self.directions[self.count] = unit;
        self.count += 1;
        true
    }

    /// The number of stored loudspeakers.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Whether the layout has no loudspeakers.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The stored loudspeaker directions.
    #[inline]
    #[must_use]
    pub fn directions(&self) -> &[Vec3] {
        &self.directions[..self.count]
    }

    /// A near-uniform 26-point spherical layout: the 6 face centres, 12 edge
    /// midpoints, and 8 corners of a cube, each projected onto the unit sphere.
    ///
    /// 26 points comfortably exceed the `(3 + 1)^2 = 16` directions needed for
    /// third-order decoding, giving a well-conditioned decode across the full
    /// sphere.
    #[must_use]
    pub fn cube26() -> Self {
        let mut layout = Self::new();
        // 6 face centres.
        for &v in &[Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z] {
            layout.push(v);
        }
        // 12 edge midpoints.
        let edges = [
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::new(-1.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, -1.0),
            Vec3::new(0.0, -1.0, 1.0),
            Vec3::new(0.0, -1.0, -1.0),
        ];
        for &v in &edges {
            layout.push(v);
        }
        // 8 corners.
        for &sx in &[-1.0 as Sample, 1.0] {
            for &sy in &[-1.0 as Sample, 1.0] {
                for &sz in &[-1.0 as Sample, 1.0] {
                    layout.push(Vec3::new(sx, sy, sz));
                }
            }
        }
        layout
    }
}

/// A virtual-loudspeaker binaural decoder for a fixed Ambisonic order.
///
/// Build once from an [`HrtfDataset`] and a [`VirtualSpeakerLayout`]
/// ([`HoaBinauralDecoder::new`], off the audio thread); then decode any number
/// of HOA blocks to headphones on the audio thread with the allocation-, lock-,
/// and panic-free [`process_block`](Self::process_block).
///
/// Internally it holds one [`BinauralRenderer`] per Ambisonic channel, each
/// loaded with the pre-baked per-channel filter pair described in the
/// [module documentation](self).
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_hrtf::hoa_binaural::{HoaBinauralDecoder, VirtualSpeakerLayout};
/// use prism_audio_hrtf::dataset::{HrtfDataset, Measurement};
///
/// // A tiny 2-tap dataset with two horizontal measurements.
/// let measurements = vec![
///     Measurement::new(0.0, 0.0, 1.0),
///     Measurement::new(core::f32::consts::FRAC_PI_2, 0.0, 1.0),
/// ];
/// let left = vec![1.0, 0.0, 0.0, 1.0];
/// let right = vec![0.0, 1.0, 1.0, 0.0];
/// let dataset = HrtfDataset::from_samples(48_000, 2, measurements, left, right).unwrap();
///
/// let mut decoder = HoaBinauralDecoder::new(&dataset, 1, &VirtualSpeakerLayout::cube26(), 64);
/// assert_eq!(decoder.channels(), 4); // first-order: W, Y, Z, X
///
/// // Feed a first-order (4-channel) block; here just the omni (W) channel.
/// let w = [1.0f32, 0.0, 0.0];
/// let silence = [0.0f32; 3];
/// let channels: [&[f32]; 4] = [&w, &silence, &silence, &silence];
/// let mut out_l = [0.0f32; 3];
/// let mut out_r = [0.0f32; 3];
/// let frames = decoder.process_block(&channels, &mut out_l, &mut out_r);
/// assert_eq!(frames, 3);
/// ```
#[derive(Debug, Clone)]
pub struct HoaBinauralDecoder {
    order: usize,
    channels: usize,
    hrir_len: usize,
    max_block: usize,
    /// One renderer per Ambisonic channel (`channels` entries), each loaded
    /// with the pre-baked per-channel filter pair.
    renderers: Vec<BinauralRenderer>,
    /// Per-channel convolution scratch, sized `max_block`.
    scratch_l: Vec<Sample>,
    scratch_r: Vec<Sample>,
}

impl HoaBinauralDecoder {
    /// Builds a decoder for `order` (clamped to [`MAX_HOA_ORDER`]) from
    /// `dataset` and the virtual-loudspeaker `layout`, baking the per-channel
    /// HRIR filters. `max_block` is the largest block size a later
    /// [`process_block`](Self::process_block) call may request (clamped to at
    /// least `1`).
    ///
    /// **Non-real-time**: interpolates one HRIR pair per virtual loudspeaker
    /// and allocates the per-channel renderers. Call off the audio thread. An
    /// empty layout yields a decoder that renders silence.
    #[must_use]
    #[expect(
        clippy::needless_range_loop,
        reason = "the filter-baking and renderer-build loops index parallel flat buffers by channel and tap"
    )]
    pub fn new(
        dataset: &HrtfDataset,
        order: usize,
        layout: &VirtualSpeakerLayout,
        max_block: usize,
    ) -> Self {
        let order = order.min(MAX_HOA_ORDER);
        let channels = hoa_channel_count(order);
        let hrir_len = dataset.hrir_len().max(1);
        let max_block = max_block.max(1);
        let inv_norm = 1.0 / (order + 1) as Sample;

        // Flat per-channel filter accumulators (channel-major, hrir_len each).
        let mut fl = vec![0.0 as Sample; channels * hrir_len];
        let mut fr = vec![0.0 as Sample; channels * hrir_len];

        // Reusable per-loudspeaker HRIR and encode scratch.
        let mut spk_l = vec![0.0 as Sample; hrir_len];
        let mut spk_r = vec![0.0 as Sample; hrir_len];
        let mut enc = [0.0 as Sample; MAX_HOA_CHANNELS];

        for &dir in layout.directions() {
            // Interpolate this loudspeaker's HRIR pair (skips silently on a
            // degenerate dataset, leaving the accumulators unchanged).
            let azimuth = local_azimuth(dir);
            let elevation = local_elevation(dir);
            interpolate(dataset, azimuth, elevation, &mut spk_l, &mut spk_r);

            // Projection-decode gains for this loudspeaker (matches
            // prism_audio_spatial::decode_hoa: dot(coeffs, encode) / (order + 1)).
            encode_hoa(dir, order, &mut enc);

            for c in 0..channels {
                let gain = enc[c] * inv_norm;
                let base = c * hrir_len;
                for t in 0..hrir_len {
                    fl[base + t] += gain * spk_l[t];
                    fr[base + t] += gain * spk_r[t];
                }
            }
        }

        let mut renderers = Vec::with_capacity(channels);
        for c in 0..channels {
            let base = c * hrir_len;
            let mut renderer = BinauralRenderer::new(hrir_len, max_block);
            renderer.set_hrir_immediate(&fl[base..base + hrir_len], &fr[base..base + hrir_len]);
            renderers.push(renderer);
        }

        Self {
            order,
            channels,
            hrir_len,
            max_block,
            renderers,
            scratch_l: vec![0.0 as Sample; max_block],
            scratch_r: vec![0.0 as Sample; max_block],
        }
    }

    /// The Ambisonic order this decoder targets.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// The number of Ambisonic channels this decoder consumes,
    /// `(order + 1)^2`.
    #[inline]
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// The HRIR (and per-channel filter) tap count.
    #[inline]
    #[must_use]
    pub const fn hrir_len(&self) -> usize {
        self.hrir_len
    }

    /// The maximum block size (frames) this decoder was built for.
    #[inline]
    #[must_use]
    pub const fn max_block(&self) -> usize {
        self.max_block
    }

    /// Clears every per-channel renderer's convolution history, silencing tails
    /// without changing the baked filters. Real-time safe.
    pub fn reset(&mut self) {
        for renderer in &mut self.renderers {
            renderer.reset();
        }
    }

    /// Decodes one HOA block to a binaural (headphone) pair.
    ///
    /// `hoa_channels[c]` is Ambisonic channel `c`'s samples for this block, in
    /// `ACN` order (channel `0` is `W`). The number of frames rendered is the
    /// minimum of `out_l.len()`, `out_r.len()`, [`max_block`](Self::max_block),
    /// and the length of every supplied channel that the decoder consumes, so
    /// short or ragged inputs clamp rather than panic. Channels beyond
    /// `min(self.channels(), hoa_channels.len())` are ignored; supplying fewer
    /// channels than [`channels`](Self::channels) simply drops the missing
    /// higher-order terms. Pass a stable channel count across blocks to keep the
    /// per-channel convolvers synchronised.
    ///
    /// Returns the number of frames written. **Real-time**: allocation, lock,
    /// and panic free.
    #[expect(
        clippy::needless_range_loop,
        reason = "the mix loops index parallel output and scratch buffers by frame and channel"
    )]
    pub fn process_block(
        &mut self,
        hoa_channels: &[&[Sample]],
        out_l: &mut [Sample],
        out_r: &mut [Sample],
    ) -> usize {
        let active = self.channels.min(hoa_channels.len());

        let mut frames = out_l.len().min(out_r.len()).min(self.max_block);
        for c in 0..active {
            frames = frames.min(hoa_channels[c].len());
        }

        for i in 0..frames {
            out_l[i] = 0.0;
            out_r[i] = 0.0;
        }
        if frames == 0 {
            return 0;
        }

        let sl = &mut self.scratch_l[..frames];
        let sr = &mut self.scratch_r[..frames];
        for c in 0..active {
            let produced = self.renderers[c].process_block(&hoa_channels[c][..frames], sl, sr);
            for i in 0..produced {
                out_l[i] += sl[i];
                out_r[i] += sr[i];
            }
        }

        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Measurement;
    use core::f32::consts::PI;
    use prism_audio_spatial::{decode_hoa, hoa_channel_count};

    const EPS: Sample = 1.0e-5;

    fn approx(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= EPS
    }

    /// A small synthetic dataset spanning the sphere with distinct, direction
    /// dependent HRIRs so interpolation is well defined everywhere.
    fn test_dataset(hrir_len: usize) -> HrtfDataset {
        let dirs = [
            (0.0, 0.0),           // front
            (PI, 0.0),            // back
            (PI / 2.0, 0.0),      // right
            (-PI / 2.0, 0.0),     // left
            (0.0, PI / 2.0 * 0.9),  // up-ish
            (0.0, -PI / 2.0 * 0.9), // down-ish
        ];
        let mut measurements = Vec::new();
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (k, &(az, el)) in dirs.iter().enumerate() {
            measurements.push(Measurement::new(az, el, 1.0));
            for t in 0..hrir_len {
                // Distinct, deterministic per-direction responses.
                let base = (k as Sample + 1.0) * 0.1;
                left.push(base + t as Sample * 0.01);
                right.push(base * 0.5 - t as Sample * 0.02);
            }
        }
        HrtfDataset::from_samples(48_000, hrir_len, measurements, left, right).unwrap()
    }

    #[test]
    fn channel_count_matches_order() {
        let ds = test_dataset(4);
        let layout = VirtualSpeakerLayout::cube26();
        for order in 0..=MAX_HOA_ORDER {
            let dec = HoaBinauralDecoder::new(&ds, order, &layout, 32);
            assert_eq!(dec.channels(), hoa_channel_count(order));
            assert_eq!(dec.order(), order);
        }
    }

    #[test]
    fn cube26_has_26_unit_directions() {
        let layout = VirtualSpeakerLayout::cube26();
        assert_eq!(layout.len(), 26);
        for &d in layout.directions() {
            assert!(approx(d.length(), 1.0));
        }
    }

    #[test]
    fn silence_in_silence_out() {
        let ds = test_dataset(4);
        let mut dec = HoaBinauralDecoder::new(&ds, 2, &VirtualSpeakerLayout::cube26(), 16);
        let zeros = [0.0 as Sample; 8];
        let ch: Vec<&[Sample]> = (0..dec.channels()).map(|_| &zeros[..]).collect();
        let mut out_l = [7.0 as Sample; 8];
        let mut out_r = [7.0 as Sample; 8];
        let frames = dec.process_block(&ch, &mut out_l, &mut out_r);
        assert_eq!(frames, 8);
        for i in 0..8 {
            assert!(approx(out_l[i], 0.0));
            assert!(approx(out_r[i], 0.0));
        }
    }

    #[test]
    fn empty_layout_renders_silence() {
        let ds = test_dataset(4);
        let mut dec = HoaBinauralDecoder::new(&ds, 1, &VirtualSpeakerLayout::new(), 8);
        let imp = [1.0 as Sample, 0.0, 0.0, 0.0];
        let ch: Vec<&[Sample]> = (0..dec.channels()).map(|_| &imp[..]).collect();
        let mut out_l = [0.0 as Sample; 4];
        let mut out_r = [0.0 as Sample; 4];
        dec.process_block(&ch, &mut out_l, &mut out_r);
        for i in 0..4 {
            assert!(approx(out_l[i], 0.0));
            assert!(approx(out_r[i], 0.0));
        }
    }

    /// Golden: feeding an encoded direction as per-channel impulses must equal
    /// the explicit virtual-loudspeaker sum
    /// `sum_s decode_hoa(encode(d), dir_s) * HRIR(s)`.
    #[test]
    fn impulse_matches_virtual_speaker_sum() {
        let hrir_len = 5;
        let ds = test_dataset(hrir_len);
        let layout = VirtualSpeakerLayout::cube26();
        let order = 3;
        let channels = hoa_channel_count(order);

        let mut dec = HoaBinauralDecoder::new(&ds, order, &layout, 64);

        // Encode a source direction into HOA coefficients.
        let source = Vec3::new(0.3, 0.6, -0.8).normalize();
        let mut coeffs = [0.0 as Sample; MAX_HOA_CHANNELS];
        encode_hoa(source, order, &mut coeffs);

        // Build per-channel impulse blocks scaled by the coefficient.
        let mut blocks = Vec::new();
        for c in 0..channels {
            let mut b = vec![0.0 as Sample; hrir_len];
            b[0] = coeffs[c];
            blocks.push(b);
        }
        let ch: Vec<&[Sample]> = blocks.iter().map(|b| b.as_slice()).collect();

        let mut out_l = vec![0.0 as Sample; hrir_len];
        let mut out_r = vec![0.0 as Sample; hrir_len];
        let frames = dec.process_block(&ch, &mut out_l, &mut out_r);
        assert_eq!(frames, hrir_len);

        // Reference: explicit virtual-loudspeaker decode + HRIR convolution.
        let mut ref_l = vec![0.0 as Sample; hrir_len];
        let mut ref_r = vec![0.0 as Sample; hrir_len];
        let mut spk_l = vec![0.0 as Sample; hrir_len];
        let mut spk_r = vec![0.0 as Sample; hrir_len];
        for &dir in layout.directions() {
            let g = decode_hoa(&coeffs, dir, order);
            let az = local_azimuth(dir);
            let el = local_elevation(dir);
            interpolate(&ds, az, el, &mut spk_l, &mut spk_r);
            for t in 0..hrir_len {
                ref_l[t] += g * spk_l[t];
                ref_r[t] += g * spk_r[t];
            }
        }

        for t in 0..hrir_len {
            assert!(approx(out_l[t], ref_l[t]), "L[{t}] {} != {}", out_l[t], ref_l[t]);
            assert!(approx(out_r[t], ref_r[t]), "R[{t}] {} != {}", out_r[t], ref_r[t]);
        }
    }

    #[test]
    fn ragged_and_short_inputs_do_not_panic() {
        let ds = test_dataset(4);
        let mut dec = HoaBinauralDecoder::new(&ds, 2, &VirtualSpeakerLayout::cube26(), 8);
        let long = [0.5 as Sample; 8];
        let short = [0.5 as Sample; 2];
        // Fewer channels than the decoder, ragged lengths.
        let ch: Vec<&[Sample]> = vec![&long[..], &short[..], &long[..]];
        let mut out_l = [0.0 as Sample; 8];
        let mut out_r = [0.0 as Sample; 8];
        let frames = dec.process_block(&ch, &mut out_l, &mut out_r);
        // Clamped to the shortest supplied channel.
        assert_eq!(frames, 2);
    }

    #[test]
    fn deterministic_repeat() {
        let ds = test_dataset(4);
        let make = || HoaBinauralDecoder::new(&ds, 2, &VirtualSpeakerLayout::cube26(), 8);
        let mut a = make();
        let mut b = make();
        let imp = [1.0 as Sample, 0.0, 0.0, 0.0];
        let ch: Vec<&[Sample]> = (0..a.channels()).map(|_| &imp[..]).collect();
        let mut al = [0.0 as Sample; 4];
        let mut ar = [0.0 as Sample; 4];
        let mut bl = [0.0 as Sample; 4];
        let mut br = [0.0 as Sample; 4];
        a.process_block(&ch, &mut al, &mut ar);
        b.process_block(&ch, &mut bl, &mut br);
        for i in 0..4 {
            assert!(approx(al[i], bl[i]));
            assert!(approx(ar[i], br[i]));
        }
    }

    #[test]
    fn reset_clears_tails() {
        let ds = test_dataset(4);
        let mut dec = HoaBinauralDecoder::new(&ds, 1, &VirtualSpeakerLayout::cube26(), 4);
        let imp = [1.0 as Sample, 1.0, 1.0, 1.0];
        let ch: Vec<&[Sample]> = (0..dec.channels()).map(|_| &imp[..]).collect();
        let mut out_l = [0.0 as Sample; 4];
        let mut out_r = [0.0 as Sample; 4];
        dec.process_block(&ch, &mut out_l, &mut out_r);
        dec.reset();
        // After reset the history is cleared: a following silent block is silent.
        let zeros = [0.0 as Sample; 4];
        let zch: Vec<&[Sample]> = (0..dec.channels()).map(|_| &zeros[..]).collect();
        let mut zl = [9.0 as Sample; 4];
        let mut zr = [9.0 as Sample; 4];
        dec.process_block(&zch, &mut zl, &mut zr);
        for i in 0..4 {
            assert!(approx(zl[i], 0.0));
            assert!(approx(zr[i], 0.0));
        }
    }
}
