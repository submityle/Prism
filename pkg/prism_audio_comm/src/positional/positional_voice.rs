//! Positional voice: turning a remote speaker into a spatial source parameter
//! set.
//!
//! Design section 45.4 treats decoded remote voice as a first-class sound
//! source that the existing spatial stack (HRTF/Ambisonic panning, distance
//! attenuation, occlusion) renders in 3D. This module does not re-implement any
//! of that rendering; it only computes the per-speaker parameters the spatial
//! layer consumes: listener-local azimuth and elevation, distance, a distance
//! attenuation gain with a proximity-chat fade, and a spatial blend that routes
//! a 2D non-spatial team channel straight through while world voice goes to the
//! full 3D path. A side-tone gain is provided for optionally monitoring the
//! local processed voice in the speaker's own ears.
//!
//! The coordinate convention matches the engine's spatial crate: world space is
//! right-handed (`+X` right, `+Y` up, `-Z` forward) and the listener
//! orientation maps listener-local space into world space, so a world direction
//! is resolved into the listener frame by the inverse rotation. Azimuth is
//! `atan2(x, -z)` (zero straight ahead, positive toward the right ear) and
//! elevation is `atan2(y, hypot(x, z))`. Every transcendental routes through
//! [`bevy_math::ops`], so the parameters are bit-reproducible across targets.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 45.4. Produces parameters for an external spatial
//! renderer (the engine's spatial/HRTF crates) from a decoded remote speaker;
//! consumed by [`crate::pipeline`] once the downlink has produced playable PCM.

use bevy_math::{ops, Quat, Vec3};
use prism_audio_core::math::Sample;

/// Distances below this (in metres) are treated as coincident with the
/// listener: the direction collapses to local forward to avoid dividing by a
/// near-zero length.
const COINCIDENT_EPSILON: Sample = 1.0e-6;

/// How a remote speaker is routed into the mix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum VoiceSpatialMode {
    /// World voice: full 3D spatialisation with distance attenuation and the
    /// proximity-chat fade.
    World3d,
    /// Team channel: a 2D, non-spatialised direct path at a steady gain,
    /// independent of distance.
    Team2d,
}

/// Tuning for [`PositionalVoice`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PositionalVoiceConfig {
    /// Distance in metres within which the attenuation gain stays at unity.
    pub reference_distance: Sample,
    /// Distance in metres at and beyond which world voice is fully silent.
    pub max_distance: Sample,
    /// Rolloff factor for the inverse-distance curve past the reference
    /// distance; larger values attenuate faster.
    pub rolloff: Sample,
    /// Lower bound applied to the raw attenuation curve before the proximity
    /// fade, so distant-but-audible speakers do not drop to inaudibility too
    /// abruptly.
    pub min_gain: Sample,
    /// Width in metres of the linear fade-out ending at `max_distance`, giving
    /// proximity chat a smooth fade in and out instead of a hard cutoff.
    pub proximity_fade: Sample,
    /// Gain applied to the local processed voice when it is monitored in the
    /// speaker's own ears (side-tone); `0` disables side-tone.
    pub side_tone_gain: Sample,
}

impl Default for PositionalVoiceConfig {
    fn default() -> Self {
        Self {
            reference_distance: 1.0,
            max_distance: 40.0,
            rolloff: 1.0,
            min_gain: 0.0,
            proximity_fade: 6.0,
            side_tone_gain: 0.0,
        }
    }
}

/// Parameters describing how a remote speaker should be spatialised.
///
/// These are inputs for the engine's spatial renderer, not rendered audio.
/// `spatial_blend` is `1` for the full 3D path and `0` for a non-spatial direct
/// path; a renderer may crossfade between its 3D and direct busses by this
/// value.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PositionalVoiceParams {
    /// Listener-local azimuth in radians: `0` straight ahead, positive toward
    /// the right ear, in `(-pi, pi]`.
    pub azimuth: Sample,
    /// Listener-local elevation in radians above the horizontal plane, in
    /// `[-pi/2, pi/2]`.
    pub elevation: Sample,
    /// Distance from listener to speaker in metres.
    pub distance: Sample,
    /// Linear playback gain in `[0, 1]` after distance attenuation, the
    /// proximity fade, and the caller's base gain.
    pub gain: Sample,
    /// Blend between the non-spatial direct path (`0`) and the full 3D path
    /// (`1`).
    pub spatial_blend: Sample,
}

/// Computes [`PositionalVoiceParams`] for remote speakers from a listener pose.
///
/// The type is a thin, stateless wrapper around a [`PositionalVoiceConfig`]; it
/// holds no per-speaker state so it can resolve any number of speakers against
/// the same listener each frame.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PositionalVoice {
    config: PositionalVoiceConfig,
}

impl PositionalVoice {
    /// Creates a resolver from a configuration, sanitising degenerate values so
    /// later arithmetic is well defined.
    #[must_use]
    pub fn new(config: PositionalVoiceConfig) -> Self {
        let reference_distance = if config.reference_distance > COINCIDENT_EPSILON {
            config.reference_distance
        } else {
            COINCIDENT_EPSILON
        };
        let max_distance = if config.max_distance > reference_distance {
            config.max_distance
        } else {
            reference_distance + 1.0
        };
        let rolloff = if config.rolloff >= 0.0 { config.rolloff } else { 0.0 };
        let min_gain = config.min_gain.clamp(0.0, 1.0);
        let proximity_fade = config.proximity_fade.clamp(0.0, max_distance - reference_distance);
        let side_tone_gain = if config.side_tone_gain >= 0.0 {
            config.side_tone_gain
        } else {
            0.0
        };
        Self {
            config: PositionalVoiceConfig {
                reference_distance,
                max_distance,
                rolloff,
                min_gain,
                proximity_fade,
                side_tone_gain,
            },
        }
    }

    /// Returns the active configuration (after sanitisation).
    #[must_use]
    pub fn config(&self) -> &PositionalVoiceConfig {
        &self.config
    }

    /// Returns the configured side-tone gain for local monitoring.
    #[must_use]
    pub fn side_tone_gain(&self) -> Sample {
        self.config.side_tone_gain
    }

    /// Computes the inverse-distance attenuation with the proximity fade for a
    /// distance in metres, in `[0, 1]`.
    #[must_use]
    pub fn distance_gain(&self, distance: Sample) -> Sample {
        let d = if distance > 0.0 { distance } else { 0.0 };
        if d <= self.config.reference_distance {
            return 1.0;
        }
        if d >= self.config.max_distance {
            return 0.0;
        }
        // Classic inverse-distance rolloff past the reference distance.
        let denom =
            self.config.reference_distance + self.config.rolloff * (d - self.config.reference_distance);
        let raw = if denom > COINCIDENT_EPSILON {
            self.config.reference_distance / denom
        } else {
            1.0
        };
        let bounded = if raw > self.config.min_gain {
            raw
        } else {
            self.config.min_gain
        };
        // Linear proximity fade over the last `proximity_fade` metres.
        let fade = if self.config.proximity_fade > COINCIDENT_EPSILON {
            let fade_start = self.config.max_distance - self.config.proximity_fade;
            if d <= fade_start {
                1.0
            } else {
                ((self.config.max_distance - d) / self.config.proximity_fade).clamp(0.0, 1.0)
            }
        } else {
            1.0
        };
        (bounded * fade).clamp(0.0, 1.0)
    }

    /// Resolves a remote speaker at `speaker_pos` against a listener at
    /// `listener_pos` with orientation `listener_orientation`.
    ///
    /// `orientation` maps listener-local space into world space (its inverse is
    /// applied here). `base_gain` is the caller's own level for this speaker
    /// (talker volume, mute ducking, etc.) and multiplies the computed gain.
    /// In [`VoiceSpatialMode::Team2d`] the result is a non-spatial direct path:
    /// azimuth and elevation are zero, distance is still reported, and the gain
    /// ignores distance attenuation.
    #[must_use]
    pub fn resolve(
        &self,
        listener_pos: Vec3,
        listener_orientation: Quat,
        speaker_pos: Vec3,
        base_gain: Sample,
        mode: VoiceSpatialMode,
    ) -> PositionalVoiceParams {
        let base = if base_gain > 0.0 { base_gain } else { 0.0 };
        let to_source = speaker_pos - listener_pos;
        let distance = ops::sqrt(to_source.dot(to_source));

        match mode {
            VoiceSpatialMode::Team2d => PositionalVoiceParams {
                azimuth: 0.0,
                elevation: 0.0,
                distance,
                gain: base.clamp(0.0, 1.0),
                spatial_blend: 0.0,
            },
            VoiceSpatialMode::World3d => {
                if distance <= COINCIDENT_EPSILON {
                    return PositionalVoiceParams {
                        azimuth: 0.0,
                        elevation: 0.0,
                        distance: 0.0,
                        gain: base.clamp(0.0, 1.0),
                        spatial_blend: 1.0,
                    };
                }
                let world_dir = to_source / distance;
                let local = listener_orientation.inverse() * world_dir;
                let azimuth = ops::atan2(local.x, -local.z);
                let horizontal = ops::sqrt(local.x * local.x + local.z * local.z);
                let elevation = ops::atan2(local.y, horizontal);
                let gain = (self.distance_gain(distance) * base).clamp(0.0, 1.0);
                PositionalVoiceParams {
                    azimuth,
                    elevation,
                    distance,
                    gain,
                    spatial_blend: 1.0,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;
    use core::f32::consts::{FRAC_PI_2, PI};

    fn close(a: Sample, b: Sample, eps: Sample) -> bool {
        ops::abs(a - b) <= eps
    }

    #[test]
    fn source_ahead_is_centered() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        let params = pv.resolve(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, -3.0),
            1.0,
            VoiceSpatialMode::World3d,
        );
        assert!(close(params.azimuth, 0.0, 1.0e-4));
        assert!(close(params.elevation, 0.0, 1.0e-4));
        assert!(close(params.distance, 3.0, 1.0e-4));
    }

    #[test]
    fn source_on_right_has_positive_azimuth() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        let params = pv.resolve(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.0),
            1.0,
            VoiceSpatialMode::World3d,
        );
        assert!(close(params.azimuth, FRAC_PI_2, 1.0e-4));
    }

    #[test]
    fn source_above_has_positive_elevation() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        let params = pv.resolve(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(0.0, 5.0, 0.0),
            1.0,
            VoiceSpatialMode::World3d,
        );
        assert!(params.elevation > 0.0);
        assert!(close(params.elevation, FRAC_PI_2, 1.0e-4));
    }

    #[test]
    fn gain_is_unity_within_reference_distance() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        assert!(close(pv.distance_gain(0.5), 1.0, 1.0e-6));
        assert!(close(pv.distance_gain(1.0), 1.0, 1.0e-6));
    }

    #[test]
    fn gain_decreases_with_distance() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        let near = pv.distance_gain(2.0);
        let far = pv.distance_gain(10.0);
        assert!(near > far);
        assert!((0.0..=1.0).contains(&near));
        assert!((0.0..=1.0).contains(&far));
    }

    #[test]
    fn gain_is_zero_at_and_beyond_max_distance() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        assert!(close(pv.distance_gain(40.0), 0.0, 1.0e-6));
        assert!(close(pv.distance_gain(100.0), 0.0, 1.0e-6));
    }

    #[test]
    fn proximity_fade_reaches_zero_smoothly() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        // Within the fade window the gain must be between zero and the value at
        // the fade start.
        let at_start = pv.distance_gain(34.0);
        let mid_fade = pv.distance_gain(37.0);
        assert!(mid_fade < at_start);
        assert!(mid_fade > 0.0);
    }

    #[test]
    fn team_mode_is_non_spatial() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        let params = pv.resolve(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(100.0, 0.0, 0.0),
            0.8,
            VoiceSpatialMode::Team2d,
        );
        assert!(close(params.azimuth, 0.0, 1.0e-6));
        assert!(close(params.elevation, 0.0, 1.0e-6));
        assert!(close(params.spatial_blend, 0.0, 1.0e-6));
        // Team gain ignores distance attenuation.
        assert!(close(params.gain, 0.8, 1.0e-6));
    }

    #[test]
    fn orientation_rotates_direction_into_local_frame() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        // Listener turned 90 degrees to the right (yaw about +Y). A source due
        // world -Z should now appear to the left ear (negative azimuth).
        let yaw = Quat::from_rotation_y(-FRAC_PI_2);
        let params = pv.resolve(
            Vec3::ZERO,
            yaw,
            Vec3::new(0.0, 0.0, -3.0),
            1.0,
            VoiceSpatialMode::World3d,
        );
        assert!(params.azimuth < 0.0);
        assert!(close(ops::abs(params.azimuth), FRAC_PI_2, 1.0e-4));
    }

    #[test]
    fn coincident_source_is_centered_and_full_gain() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        let params = pv.resolve(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::IDENTITY,
            Vec3::new(1.0, 2.0, 3.0),
            0.5,
            VoiceSpatialMode::World3d,
        );
        assert!(close(params.azimuth, 0.0, 1.0e-6));
        assert!(close(params.distance, 0.0, 1.0e-6));
        assert!(close(params.gain, 0.5, 1.0e-6));
    }

    #[test]
    fn azimuth_stays_within_pi() {
        let pv = PositionalVoice::new(PositionalVoiceConfig::default());
        let params = pv.resolve(
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::new(0.0, 0.0, 3.0),
            1.0,
            VoiceSpatialMode::World3d,
        );
        // Directly behind: azimuth magnitude is pi.
        assert!(close(ops::abs(params.azimuth), PI, 1.0e-4));
    }
}
