//! Seamless loop-seam crossfade rendering.
//!
//! [`crate::loop_point`] *detects* a loop region and records how many frames of
//! crossfade the seam wants, but it does not touch the samples. This module is
//! the offline render step that bakes that crossfade into the PCM so a runtime
//! sampler can loop `start..end` click-free with no per-sample work.
//!
//! The classic sample-editor technique hides the splice discontinuity at the
//! loop return by blending the `cf` frames ending at `end` with the `cf` frames
//! ending at `start`:
//!
//! ```text
//! out[end-cf + k] = in[end-cf + k] * fade_out[k] + in[start-cf + k] * fade_in[k]
//! ```
//!
//! where `fade_out` runs `1 -> 0` and `fade_in` runs `0 -> 1` across the window.
//! As the window reaches the seam the loop tail has morphed into the material
//! that naturally leads into `start`, so the `end -> start` jump is inaudible.
//! An equal-power law (`cos`/`sin`) keeps perceived loudness steady through the
//! blend; a linear law is offered for correlated material where a constant-gain
//! sum is preferred.
//!
//! Only [`LoopMode::Forward`] seams are blended: a [`LoopMode::PingPong`] loop
//! reflects at its turning points and its seam is already continuous, so this
//! stage leaves ping-pong assets untouched. All index math is exact and clamped
//! so an over-long requested crossfade, a loop that starts too close to the
//! buffer head, or an out-of-range region degrades to a no-op clone rather than
//! panicking.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. Seam
//! crossfading for seamless sample loops and equal-power (`cos`/`sin`) panning
//! laws are textbook classic DSP; only the ideas are used.
//!
//! # Relationship
//!
//! Consumes [`crate::loop_point::LoopPoints`] (and its `crossfade_frames`) and
//! produces a baked [`ConditionedPcm`] for the seamless-loop contract of design
//! sections 10, 20, and 51. Pairs with [`crate::delay_trim`], which rebases
//! loop points onto the trimmed program before this stage bakes the seam.

use bevy_math::ops;

use core::f32::consts::FRAC_PI_2;

use prism_audio_core::math::Sample;

use crate::loop_point::{LoopMode, LoopPoints};
use crate::pcm::ConditionedPcm;

/// The gain law applied across the crossfade window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum CrossfadeShape {
    /// Equal-power law: `fade_out = cos(theta)`, `fade_in = sin(theta)` with
    /// `theta` sweeping `0 -> pi/2`. The gains satisfy `out^2 + in^2 == 1`, so
    /// the perceived loudness of uncorrelated material stays constant. This is
    /// the default and the right choice for most loops.
    #[default]
    EqualPower,
    /// Linear law: `fade_out = 1 - t`, `fade_in = t`. The gains satisfy
    /// `out + in == 1`, which preserves the level of strongly correlated
    /// material (where the two windows are nearly identical) without the
    /// mid-window bump an equal-power law would add.
    Linear,
}

/// Resolves how many frames of crossfade can actually be applied to `pcm` for
/// `points`, after clamping to everything the geometry allows.
///
/// Returns `0` (meaning "leave the buffer untouched") when the loop is a
/// [`LoopMode::PingPong`] loop, when the region is empty or runs past the
/// buffer, or when there is no lead-in material before `start`. Otherwise the
/// requested `crossfade_frames` is clamped to both the loop length and the
/// available lead-in (`start`).
#[must_use]
pub fn effective_crossfade(points: &LoopPoints, frames: usize) -> usize {
    if points.mode != LoopMode::Forward {
        return 0;
    }
    if points.start >= points.end || points.end > frames {
        return 0;
    }
    let loop_len = points.end - points.start;
    (points.crossfade_frames as usize)
        .min(loop_len)
        .min(points.start)
}

/// Returns the `(fade_out, fade_in)` gain pair at normalized position
/// `t in [0, 1]` for `shape`.
#[must_use]
#[inline]
fn gains(shape: CrossfadeShape, t: Sample) -> (Sample, Sample) {
    match shape {
        CrossfadeShape::EqualPower => {
            let theta = t * FRAC_PI_2;
            (ops::cos(theta), ops::sin(theta))
        }
        CrossfadeShape::Linear => (1.0 - t, t),
    }
}

/// Bakes the loop-seam crossfade for `points` into a copy of `pcm`, returning
/// the seamless-looping asset.
///
/// Every channel is blended with identical index math. When
/// [`effective_crossfade`] resolves to zero the input is returned unchanged
/// (ping-pong loops, out-of-range regions, or a missing lead-in), so this
/// function is always safe to call on any detected loop.
#[must_use]
pub fn apply_loop_crossfade(
    pcm: &ConditionedPcm,
    points: &LoopPoints,
    shape: CrossfadeShape,
) -> ConditionedPcm {
    let frames = pcm.frames();
    let cf = effective_crossfade(points, frames);
    let mut out = pcm.clone();
    if cf == 0 {
        return out;
    }
    let tail_base = points.end - cf;
    let lead_base = points.start - cf;
    let inv = 1.0 / cf as Sample;
    for ch in 0..out.channel_count() {
        // The lead window [start-cf, start) and the tail window [end-cf, end)
        // are disjoint (end-cf >= start because cf <= loop length), so reading
        // the lead while writing the tail never observes a modified sample.
        let buf = out.channel_mut(ch).expect("channel index in range");
        for k in 0..cf {
            let t = (k as Sample + 0.5) * inv;
            let (g_out, g_in) = gains(shape, t);
            let tail = buf[tail_base + k];
            let lead = buf[lead_base + k];
            buf[tail_base + k] = tail * g_out + lead * g_in;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_core::buffer::ChannelLayout;
    use Vec;

    #[cfg(feature = "std")]
    fn wrap(ch: Vec<Sample>) -> Vec<Vec<Sample>> {
        vec![ch]
    }

    #[cfg(not(feature = "std"))]
    fn wrap(ch: Vec<Sample>) -> Vec<Vec<Sample>> {
        alloc::vec![ch]
    }

    fn mono(samples: Vec<Sample>) -> ConditionedPcm {
        ConditionedPcm::new(48_000, ChannelLayout::Mono, wrap(samples)).unwrap()
    }

    fn fwd(start: usize, end: usize, cf: u32) -> LoopPoints {
        LoopPoints {
            start,
            end,
            crossfade_frames: cf,
            mode: LoopMode::Forward,
        }
    }

    #[test]
    fn effective_crossfade_clamps_to_lead_in_and_length() {
        // Requested 100 but only start=8 lead-in and loop length 20 available.
        let p = fwd(8, 28, 100);
        assert_eq!(effective_crossfade(&p, 40), 8);
        // Requested 5 fits in both -> honored verbatim.
        assert_eq!(effective_crossfade(&fwd(8, 28, 5), 40), 5);
    }

    #[test]
    fn effective_crossfade_rejects_pingpong_and_out_of_range() {
        let mut p = fwd(8, 28, 4);
        p.mode = LoopMode::PingPong;
        assert_eq!(effective_crossfade(&p, 40), 0);
        // end past the buffer.
        assert_eq!(effective_crossfade(&fwd(8, 50, 4), 40), 0);
        // empty region.
        assert_eq!(effective_crossfade(&fwd(10, 10, 4), 40), 0);
        // no lead-in (start == 0).
        assert_eq!(effective_crossfade(&fwd(0, 20, 4), 40), 0);
    }

    #[test]
    fn zero_crossfade_is_identity() {
        let src = mono((0..40).map(|k| k as Sample).collect());
        let out = apply_loop_crossfade(&src, &fwd(8, 28, 0), CrossfadeShape::EqualPower);
        assert_eq!(out, src);
    }

    #[test]
    fn pingpong_is_untouched() {
        let src = mono((0..40).map(|k| k as Sample).collect());
        let mut p = fwd(8, 28, 6);
        p.mode = LoopMode::PingPong;
        let out = apply_loop_crossfade(&src, &p, CrossfadeShape::EqualPower);
        assert_eq!(out, src);
    }

    #[test]
    fn only_tail_window_is_modified() {
        let src = mono((0..40).map(|k| k as Sample).collect());
        let p = fwd(8, 28, 6); // tail window [22, 28)
        let out = apply_loop_crossfade(&src, &p, CrossfadeShape::Linear);
        let a = src.channel(0).unwrap();
        let b = out.channel(0).unwrap();
        for i in 0..40 {
            if (22..28).contains(&i) {
                // Blended region generally differs from the pure ramp.
            } else {
                assert_eq!(a[i], b[i], "frame {i} outside tail must be unchanged");
            }
        }
    }

    #[test]
    fn seam_morphs_tail_into_pre_start_material() {
        // A constant-per-region signal makes the blend endpoints exact.
        // Lead-in [start-cf, start) holds 100.0, tail [end-cf, end) holds 0.0.
        let mut data = alloc::vec![0.0 as Sample; 40];
        let (start, end, cf) = (12usize, 30usize, 6u32);
        for v in data.iter_mut().take(start).skip(start - cf as usize) {
            *v = 100.0;
        }
        let src = mono(data);
        let out = apply_loop_crossfade(&src, &fwd(start, end, cf), CrossfadeShape::EqualPower);
        let b = out.channel(0).unwrap();
        // First blended frame is mostly the (zero) tail; last is mostly lead.
        assert!(b[end - cf as usize] < 20.0, "seam start should stay near tail");
        assert!(b[end - 1] > 80.0, "seam end should morph into lead-in");
    }

    #[test]
    fn equal_power_preserves_energy_of_uncorrelated_windows() {
        // Tail = 1.0 everywhere, lead-in = 1.0 everywhere but treated as the two
        // legs of a cos/sin pair: out^2 + in^2 == 1, so a unit tail blended with
        // a unit lead keeps |value| within [cos+sin peak]. Verify the law.
        for k in 0..8u32 {
            let t = (k as Sample + 0.5) / 8.0;
            let (go, gi) = gains(CrossfadeShape::EqualPower, t);
            assert!((go * go + gi * gi - 1.0).abs() < 1.0e-5, "equal-power law");
        }
    }

    #[test]
    fn linear_law_is_constant_gain() {
        for k in 0..8u32 {
            let t = (k as Sample + 0.5) / 8.0;
            let (go, gi) = gains(CrossfadeShape::Linear, t);
            assert!((go + gi - 1.0).abs() < 1.0e-6, "linear law sums to one");
        }
    }

    #[test]
    fn multichannel_applies_to_every_channel() {
        let left: Vec<Sample> = (0..40).map(|k| k as Sample).collect();
        let right: Vec<Sample> = (0..40).map(|k| -(k as Sample)).collect();
        let src = ConditionedPcm::new(
            48_000,
            ChannelLayout::Stereo,
            {
                #[cfg(feature = "std")]
                {
                    vec![left, right]
                }
                #[cfg(not(feature = "std"))]
                {
                    alloc::vec![left, right]
                }
            },
        )
        .unwrap();
        let out = apply_loop_crossfade(&src, &fwd(8, 28, 6), CrossfadeShape::EqualPower);
        // Both channels changed somewhere inside the tail window.
        let changed = |c: usize| {
            let a = src.channel(c).unwrap();
            let b = out.channel(c).unwrap();
            (22..28).any(|i| (a[i] - b[i]).abs() > 1.0e-6)
        };
        assert!(changed(0) && changed(1));
    }

    #[test]
    fn default_shape_is_equal_power() {
        assert_eq!(CrossfadeShape::default(), CrossfadeShape::EqualPower);
    }

    #[test]
    fn deterministic_and_reproducible() {
        let src = mono((0..64).map(|k| ops::sin(k as Sample * 0.1)).collect());
        let p = fwd(10, 50, 12);
        let a = apply_loop_crossfade(&src, &p, CrossfadeShape::EqualPower);
        let b = apply_loop_crossfade(&src, &p, CrossfadeShape::EqualPower);
        assert_eq!(a, b);
    }
}
