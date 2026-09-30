//! First-order Ambisonics (FOA): encoding, rotation, and decoding.
//!
//! This module places a listener-relative point source into a first-order
//! Ambisonic soundfield, rotates that soundfield with the listener's head, and
//! decodes it back to an arbitrary speaker (or virtual sampling) direction. It
//! is the spatialisation representation that decouples *where a sound is* from
//! *how it is finally reproduced* (headphones, stereo, quad, 5.1, ...), exactly
//! as classic scene-based spatial audio prescribes.
//!
//! # Provenance
//!
//! Ambisonics is a publicly documented, decades-old body of acoustics
//! knowledge. The soundfield decomposition into spherical-harmonic components
//! (for first order: the omnidirectional pressure `W` plus the three
//! figure-of-eight velocity components `X`, `Y`, `Z`) originates with Michael
//! Gerzon's work in the early 1970s ("Periphony: With-Height Sound
//! Reproduction", *JAES*, 1973). The channel ordering and normalisation used
//! here follow the modern, openly specified **`AmbiX`** convention: **`ACN`**
//! (Ambisonic Channel Number) component ordering together with **`SN3D`**
//! (Schmidt semi-normalised) gains, under which the first-order encoding gains
//! reduce to `[W, X, Y, Z] = [1, x, y, z]` for a unit source direction
//! `(x, y, z)` on the acoustic axes.
//!
//! Everything here is implemented from that public literature. It contains
//! **no Unreal Engine, Unity, Godot, Wwise, or FMOD source or derived code**.
//!
//! # Channel order
//!
//! The engine's [`ChannelLayout::AmbisonicFoa`](prism_audio_core::buffer::ChannelLayout::AmbisonicFoa)
//! buffer stores its four channels as `W, X, Y, Z` at indices `0..=3`. `ACN`
//! numbers the first-order components `W=0, Y=1, Z=2, X=3`; this module adapts
//! that spec to the engine's fixed `W, X, Y, Z` slot layout while keeping the
//! `SN3D` gains. All public functions and the [`FoaEncoderNode`] speak in the
//! engine's `[W, X, Y, Z]` order.
//!
//! # Coordinate / axis convention
//!
//! Directions are expressed in the **listener-local** frame, which matches
//! Bevy: `-Z` is forward, `+X` is to the right, `+Y` is up (see
//! [`crate::geometry`]). The Ambisonic *acoustic* axes are the classic
//! `front / left / up` triple, so this module maps the two frames with:
//!
//! ```text
//! front = -dir.z     (Bevy forward is -Z)
//! left  = -dir.x     (Bevy right is +X, so left is -X)
//! up    =  dir.y
//! ```
//!
//! and the encoded velocity components are `X = front`, `Y = left`, `Z = up`.
//! This mapping is self-consistent: encoding a direction and immediately
//! decoding a speaker at that same direction yields the maximum response.
//!
//! # Real-time contract
//!
//! The free functions and [`FoaEncoderNode::process`] are **allocation free,
//! lock free, and panic free**. Gain (re)computation from a direction happens
//! only in the non-real-time setters ([`FoaEncoderNode::set_direction`] /
//! [`FoaEncoderNode::set_direction_immediate`]); the hot path merely advances
//! the per-channel [`Smoothed`] gains and multiplies.
//!
//! # Determinism
//!
//! All length/normalisation math flows through [`bevy_math`] (libm-backed via
//! the crate's `nostd-libm` feature) rather than `f32` intrinsics, so encoding,
//! rotation, and decoding are bit-reproducible across targets.

use bevy_math::{Quat, Vec3};

use prism_audio_core::graph::{AudioNode, ProcessIo, RenderContext};
use prism_audio_core::math::Sample;
use prism_audio_core::param::{Ramp, Smoothed};

/// Number of channels in a first-order Ambisonic buffer (`W, X, Y, Z`).
pub const FOA_CHANNELS: usize = 4;

/// Buffer index of the omnidirectional pressure component `W`.
pub const IDX_W: usize = 0;
/// Buffer index of the front-facing velocity component `X`.
pub const IDX_X: usize = 1;
/// Buffer index of the left-facing velocity component `Y`.
pub const IDX_Y: usize = 2;
/// Buffer index of the up-facing velocity component `Z`.
pub const IDX_Z: usize = 3;

/// The `SN3D`-normalised `W` (pressure) gain. Under `SN3D` the zeroth-order
/// component is unit-weighted, unlike the historical `1/sqrt(2)` `FuMa` weight.
const W_GAIN: Sample = 1.0;

/// Projection-decode weight for a first-order component sum.
///
/// A "basic"/projection decoder for order `N` scales the reconstructed
/// pressure by `1 / (N + 1)`; for first order (`N = 1`) that is `0.5`. This
/// keeps a front-encoded source from overshooting when sampled by a
/// coincident virtual speaker.
const DECODE_WEIGHT: Sample = 0.5;

/// Converts a listener-local Bevy direction into Ambisonic acoustic axes.
///
/// Returns `(front, left, up)` as described in the module docs
/// (`front = -z`, `left = -x`, `up = y`). The input is expected to be
/// normalised by the caller.
#[inline]
#[must_use]
fn acoustic_axes(direction: Vec3) -> (Sample, Sample, Sample) {
    (-direction.z, -direction.x, direction.y)
}

/// Rebuilds a listener-local Bevy vector from Ambisonic velocity components.
///
/// This is the exact inverse of [`acoustic_axes`] applied to the encoded
/// `(X, Y, Z)` triple, where `X = front`, `Y = left`, `Z = up`:
///
/// ```text
/// front = -z_bevy  =>  z_bevy = -X
/// left  = -x_bevy  =>  x_bevy = -Y
/// up    =  y_bevy  =>  y_bevy =  Z
/// ```
///
/// hence `Vec3::new(-Y, Z, -X)`.
#[inline]
#[must_use]
fn velocity_to_bevy(x: Sample, y: Sample, z: Sample) -> Vec3 {
    Vec3::new(-y, z, -x)
}

/// Computes the first-order `SN3D` encoding gains for a source `direction`.
///
/// The direction is first normalised (a zero-length direction collapses to all
/// velocity gains being zero, leaving only the omnidirectional `W`). The
/// returned array is in the engine's `[W, X, Y, Z]` channel order, with
/// `X = front`, `Y = left`, `Z = up` and `W = 1` (`SN3D`).
///
/// Multiplying a mono sample by these gains distributes it into the four FOA
/// channels; see [`encode_foa_sample`].
#[inline]
#[must_use]
pub fn encode_foa_gains(direction: Vec3) -> [Sample; FOA_CHANNELS] {
    let dir = direction.normalize_or_zero();
    let (front, left, up) = acoustic_axes(dir);
    [W_GAIN, front, left, up]
}

/// Encodes a single mono `sample` arriving from `direction` into the four FOA
/// channels, writing `out[k] = sample * gains[k]`.
///
/// `out` is in the engine's `[W, X, Y, Z]` order. This is a convenience for
/// one-shot / offline encoding; the real-time path uses [`FoaEncoderNode`]
/// so that direction changes are click-free.
#[inline]
pub fn encode_foa_sample(sample: Sample, direction: Vec3, out: &mut [Sample; FOA_CHANNELS]) {
    let gains = encode_foa_gains(direction);
    out[IDX_W] = sample * gains[IDX_W];
    out[IDX_X] = sample * gains[IDX_X];
    out[IDX_Y] = sample * gains[IDX_Y];
    out[IDX_Z] = sample * gains[IDX_Z];
}

/// Rotates a first-order soundfield in place by `rotation`.
///
/// The omnidirectional pressure `W` is rotation-invariant. The velocity
/// components `(X, Y, Z)` form a vector on the acoustic axes; to rotate the
/// field with the listener's head we map that vector back to the Bevy frame
/// (`Vec3::new(-Y, Z, -X)`), apply `rotation * v`, then re-express the rotated
/// vector on the acoustic axes (`X = -z'`, `Y = -x'`, `Z = y'`) and store it
/// back. An identity `rotation` therefore leaves `wxyz` exactly unchanged, and
/// because quaternion rotation is orthonormal the velocity energy
/// `X^2 + Y^2 + Z^2` is preserved.
///
/// `wxyz` is in the engine's `[W, X, Y, Z]` order.
#[inline]
pub fn rotate_foa(wxyz: &mut [Sample; FOA_CHANNELS], rotation: Quat) {
    let v = velocity_to_bevy(wxyz[IDX_X], wxyz[IDX_Y], wxyz[IDX_Z]);
    let r = rotation * v;
    let (front, left, up) = acoustic_axes(r);
    wxyz[IDX_X] = front;
    wxyz[IDX_Y] = left;
    wxyz[IDX_Z] = up;
    // wxyz[IDX_W] is intentionally left unchanged.
}

/// Decodes a first-order soundfield toward a single `speaker_direction`,
/// returning the mono signal that speaker should emit.
///
/// This is the classic first-order projection ("basic") decode: the speaker
/// direction is normalised and mapped to the acoustic axes
/// (`sf = -z`, `sl = -x`, `su = y`), and the output is
/// `DECODE_WEIGHT * (W + X*sf + Y*sl + Z*su)` with `DECODE_WEIGHT = 0.5`. A
/// speaker aligned with the encoded direction receives the strongest signal;
/// one facing the opposite way receives the weakest.
///
/// `wxyz` is in the engine's `[W, X, Y, Z]` order.
#[inline]
#[must_use]
pub fn decode_foa(wxyz: &[Sample; FOA_CHANNELS], speaker_direction: Vec3) -> Sample {
    let dir = speaker_direction.normalize_or_zero();
    let (sf, sl, su) = acoustic_axes(dir);
    DECODE_WEIGHT * (wxyz[IDX_W] + wxyz[IDX_X] * sf + wxyz[IDX_Y] * sl + wxyz[IDX_Z] * su)
}

/// A real-time node that encodes a mono source into first-order Ambisonics with
/// click-free direction changes.
///
/// * Input port `0`: mono ([`ChannelLayout::Mono`]).
/// * Output port `0`: first-order Ambisonics ([`ChannelLayout::AmbisonicFoa`],
///   channels `W, X, Y, Z`).
///
/// The four `SN3D` encoding gains are held as [`Smoothed`] parameters so that
/// moving the source is a per-sample glide rather than a step. Direction
/// changes are applied off the audio thread through
/// [`set_direction`](Self::set_direction) /
/// [`set_direction_immediate`](Self::set_direction_immediate); the hot path
/// only advances and applies the smoothers and never allocates or panics.
#[derive(Debug, Clone)]
pub struct FoaEncoderNode {
    gains: [Smoothed; FOA_CHANNELS],
}

impl FoaEncoderNode {
    /// Creates an encoder settled at "pressure only": the `W` gain starts at
    /// `1.0` (`SN3D`) and the three velocity gains at `0.0`, i.e. an
    /// undirected/omnidirectional source until a direction is set.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            gains: [
                Smoothed::new(W_GAIN),
                Smoothed::new(0.0),
                Smoothed::new(0.0),
                Smoothed::new(0.0),
            ],
        }
    }

    /// Retargets the encoding gains toward a new source `direction`, gliding
    /// over `ramp`. **Non-real-time**: this recomputes the `SN3D` gains and must
    /// be called off the audio thread (e.g. from the control/update tick).
    pub fn set_direction(&mut self, direction: Vec3, ramp: Ramp) {
        let gains = encode_foa_gains(direction);
        for (slot, &g) in self.gains.iter_mut().zip(gains.iter()) {
            slot.set_target(g, ramp);
        }
    }

    /// Snaps the encoding gains to a new source `direction` immediately (no
    /// glide). Intended for initial placement or teleports; using it on a
    /// moving source may click. **Non-real-time.**
    #[inline]
    pub fn set_direction_immediate(&mut self, direction: Vec3) {
        self.set_direction(direction, Ramp::Immediate);
    }

    /// Returns the current (instantaneous) encoding gains in `[W, X, Y, Z]`
    /// order without advancing the smoothers. Useful for tests and metering.
    #[inline]
    #[must_use]
    pub fn current_gains(&self) -> [Sample; FOA_CHANNELS] {
        [
            self.gains[IDX_W].current(),
            self.gains[IDX_X].current(),
            self.gains[IDX_Y].current(),
            self.gains[IDX_Z].current(),
        ]
    }
}

impl Default for FoaEncoderNode {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AudioNode for FoaEncoderNode {
    fn process(&mut self, _ctx: &RenderContext, io: &mut ProcessIo<'_>) {
        let (input, output) = io.io(0, 0);

        // Degenerate wiring guard: never index a missing channel, never panic.
        if input.channels() == 0 || output.channels() == 0 {
            return;
        }

        let mono = input.channel(0);
        let frames = mono.len().min(output.active_frames());
        let out_channels = output.channels().min(FOA_CHANNELS);

        for k in 0..FOA_CHANNELS {
            if k < out_channels {
                let dst = output.channel_mut(k);
                let gain = &mut self.gains[k];
                for i in 0..frames {
                    dst[i] = mono[i] * gain.next_sample();
                }
            } else {
                // Keep every smoother phase-aligned even when the output has
                // fewer channels than a full FOA bus.
                let gain = &mut self.gains[k];
                for _ in 0..frames {
                    gain.next_sample();
                }
            }
        }
    }

    fn reset(&mut self) {
        for k in 0..FOA_CHANNELS {
            let settled = self.gains[k].target();
            self.gains[k] = Smoothed::new(settled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_2;
    use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};

    const EPS: Sample = 1.0e-5;

    fn approx(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= EPS
    }

    #[test]
    fn encode_front_excites_x_only() {
        // Bevy forward is -Z.
        let g = encode_foa_gains(Vec3::new(0.0, 0.0, -1.0));
        assert!(approx(g[IDX_W], 1.0));
        assert!(g[IDX_X] > 0.5, "front should give positive X, got {}", g[IDX_X]);
        assert!(approx(g[IDX_Y], 0.0));
        assert!(approx(g[IDX_Z], 0.0));
    }

    #[test]
    fn encode_left_excites_y() {
        // Bevy right is +X, so "left" is -X.
        let g = encode_foa_gains(Vec3::new(-1.0, 0.0, 0.0));
        assert!(g[IDX_Y] > 0.5, "left should give positive Y, got {}", g[IDX_Y]);
        assert!(approx(g[IDX_W], 1.0));
        assert!(approx(g[IDX_X], 0.0));
        assert!(approx(g[IDX_Z], 0.0));
    }

    #[test]
    fn encode_up_excites_z() {
        let g = encode_foa_gains(Vec3::new(0.0, 1.0, 0.0));
        assert!(g[IDX_Z] > 0.5, "up should give positive Z, got {}", g[IDX_Z]);
        assert!(approx(g[IDX_W], 1.0));
        assert!(approx(g[IDX_X], 0.0));
        assert!(approx(g[IDX_Y], 0.0));
    }

    #[test]
    fn w_gain_is_always_unity() {
        for dir in [
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(3.0, -2.0, 5.0),
            Vec3::ZERO,
        ] {
            assert!(approx(encode_foa_gains(dir)[IDX_W], 1.0));
        }
    }

    #[test]
    fn zero_direction_leaves_only_pressure() {
        let g = encode_foa_gains(Vec3::ZERO);
        assert!(approx(g[IDX_W], 1.0));
        assert!(approx(g[IDX_X], 0.0));
        assert!(approx(g[IDX_Y], 0.0));
        assert!(approx(g[IDX_Z], 0.0));
    }

    #[test]
    fn encode_sample_scales_gains() {
        let mut out = [0.0; FOA_CHANNELS];
        encode_foa_sample(2.0, Vec3::new(0.0, 0.0, -1.0), &mut out);
        assert!(approx(out[IDX_W], 2.0));
        assert!(out[IDX_X] > 1.0);
        assert!(approx(out[IDX_Y], 0.0));
        assert!(approx(out[IDX_Z], 0.0));
    }

    #[test]
    fn rotate_identity_is_a_no_op() {
        let mut wxyz = [1.0, 0.3, -0.7, 0.5];
        let before = wxyz;
        rotate_foa(&mut wxyz, Quat::IDENTITY);
        for k in 0..FOA_CHANNELS {
            assert!(approx(wxyz[k], before[k]));
        }
    }

    #[test]
    fn rotate_y_90_moves_front_to_side_and_conserves_energy() {
        // Encode a front source, then yaw the field by +90 degrees.
        let mut wxyz = encode_foa_gains(Vec3::new(0.0, 0.0, -1.0));
        let energy_before =
            wxyz[IDX_X] * wxyz[IDX_X] + wxyz[IDX_Y] * wxyz[IDX_Y] + wxyz[IDX_Z] * wxyz[IDX_Z];

        rotate_foa(&mut wxyz, Quat::from_rotation_y(FRAC_PI_2));

        // W untouched, front energy has moved onto the Y (left/right) axis.
        assert!(approx(wxyz[IDX_W], 1.0));
        assert!(wxyz[IDX_X].abs() < 1.0e-3, "X should vanish, got {}", wxyz[IDX_X]);
        assert!(wxyz[IDX_Y].abs() > 0.5, "energy should move to Y, got {}", wxyz[IDX_Y]);

        let energy_after =
            wxyz[IDX_X] * wxyz[IDX_X] + wxyz[IDX_Y] * wxyz[IDX_Y] + wxyz[IDX_Z] * wxyz[IDX_Z];
        assert!(
            (energy_after - energy_before).abs() < 1.0e-4,
            "velocity energy must be conserved: {energy_before} vs {energy_after}"
        );
    }

    #[test]
    fn decode_favours_the_encoded_direction() {
        let front = Vec3::new(0.0, 0.0, -1.0);
        let back = Vec3::new(0.0, 0.0, 1.0);
        let field = encode_foa_gains(front);

        let front_speaker = decode_foa(&field, front);
        let back_speaker = decode_foa(&field, back);

        assert!(
            front_speaker > back_speaker,
            "front speaker ({front_speaker}) should be louder than back ({back_speaker})"
        );
        assert!(front_speaker > 0.0);
    }

    #[test]
    fn node_preserves_frame_count_and_encodes_front() {
        let mut node = FoaEncoderNode::new();
        node.set_direction_immediate(Vec3::new(0.0, 0.0, -1.0));

        let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
        input.channel_mut(0).copy_from_slice(&[1.0; 8]);
        let mut output = AudioBuffer::new(ChannelLayout::AmbisonicFoa, 8);

        let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
        let inputs = [input];
        let mut outputs = [output];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }
        output = outputs.into_iter().next().unwrap();

        assert_eq!(output.active_frames(), 8);
        // W tracks the unit mono input; X is excited (front); Y, Z stay silent.
        assert!(output.channel(IDX_W).iter().all(|&s| approx(s, 1.0)));
        assert!(output.channel(IDX_X).iter().all(|&s| s > 0.5));
        assert!(output.channel(IDX_Y).iter().all(|&s| approx(s, 0.0)));
        assert!(output.channel(IDX_Z).iter().all(|&s| approx(s, 0.0)));
    }

    #[test]
    fn node_glides_smoothly_between_directions() {
        let mut node = FoaEncoderNode::new();
        node.set_direction_immediate(Vec3::new(0.0, 0.0, -1.0)); // front: X = 1
        // Move to the left over 8 samples: X -> 0, Y -> 1.
        node.set_direction(Vec3::new(-1.0, 0.0, 0.0), Ramp::Linear { samples: 8 });

        let mut input = AudioBuffer::new(ChannelLayout::Mono, 8);
        input.channel_mut(0).copy_from_slice(&[1.0; 8]);
        let mut outputs = [AudioBuffer::new(ChannelLayout::AmbisonicFoa, 8)];
        let ctx = RenderContext { sample_rate: 48_000, frames: 8, playhead: 0 };
        let inputs = [input];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }
        let out = &outputs[0];

        // X should be monotonically (weakly) decreasing, Y increasing: a glide.
        let x = out.channel(IDX_X);
        let y = out.channel(IDX_Y);
        assert!(x[0] > x[7], "X should fall across the block: {} -> {}", x[0], x[7]);
        assert!(y[7] > y[0], "Y should rise across the block: {} -> {}", y[0], y[7]);
        // No zipper jumps larger than a single linear step (~1/8 here).
        for w in x.windows(2) {
            assert!((w[1] - w[0]).abs() < 0.2);
        }
    }

    #[test]
    fn reset_settles_gains_at_their_targets() {
        let mut node = FoaEncoderNode::new();
        // Start a long glide, advance partway, then reset.
        node.set_direction(Vec3::new(-1.0, 0.0, 0.0), Ramp::Linear { samples: 64 });

        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0; 4]);
        let mut outputs = [AudioBuffer::new(ChannelLayout::AmbisonicFoa, 4)];
        let ctx = RenderContext { sample_rate: 48_000, frames: 4, playhead: 0 };
        let inputs = [input];
        {
            let mut io = ProcessIo::new(&inputs, &mut outputs);
            node.process(&ctx, &mut io);
        }

        node.reset();
        // After reset each smoother sits exactly on its target (the left field).
        let g = node.current_gains();
        let target = encode_foa_gains(Vec3::new(-1.0, 0.0, 0.0));
        for k in 0..FOA_CHANNELS {
            assert!(approx(g[k], target[k]), "channel {k}: {} vs {}", g[k], target[k]);
        }
    }

    #[test]
    fn process_does_not_panic_on_short_output_buffer() {
        // A too-narrow (mono) output must not panic; extra components are simply
        // dropped while the smoothers still advance.
        let mut node = FoaEncoderNode::new();
        node.set_direction_immediate(Vec3::new(0.0, 0.0, -1.0));
        let mut input = AudioBuffer::new(ChannelLayout::Mono, 4);
        input.channel_mut(0).copy_from_slice(&[1.0; 4]);
        let mut outputs = [AudioBuffer::new(ChannelLayout::Mono, 4)];
        let ctx = RenderContext { sample_rate: 48_000, frames: 4, playhead: 0 };
        let inputs = [input];
        let mut io = ProcessIo::new(&inputs, &mut outputs);
        node.process(&ctx, &mut io);
        assert!(outputs[0].channel(0).iter().all(|&s| approx(s, 1.0)));
    }
}
