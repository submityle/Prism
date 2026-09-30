//! Per-source spatialisation orchestration: the control-rate layer that
//! combines every leaf model in this crate (distance attenuation, directional
//! cone, Doppler, occlusion, and air absorption) into a single, coherent set
//! of DSP targets for one sound source.
//!
//! # Why an orchestration layer?
//!
//! The leaf modules ([`attenuation`](crate::attenuation), [`cone`](crate::cone),
//! [`doppler`](crate::doppler), [`occlusion`](crate::occlusion), and
//! [`air`](crate::air)) each answer one narrow question. A game or renderer,
//! however, needs a *single* answer per source per control block: "given where
//! the listener and emitter are right now, what gain, pitch, pan angles,
//! low-pass corner, and reverb-send scale should the voice use?" This module is
//! that composition point. It owns the *policy* for how the independent models
//! combine (which gains multiply, which cut-offs take the tighter of two
//! corners, how occlusion splits between the direct and wet paths) so that the
//! leaves stay orthogonal and individually testable.
//!
//! # Control rate, not audio rate
//!
//! [`resolve`] is a pure function evaluated once per control block (typically
//! once per rendered buffer), *not* per sample. It produces a [`SpatialParams`]
//! snapshot; the caller feeds those targets into the real-time nodes that
//! actually touch samples ([`PannerNode`](crate::panner::PannerNode),
//! [`OcclusionNode`](crate::occlusion::OcclusionNode),
//! [`AirAbsorptionNode`](crate::air::AirAbsorptionNode), and a pitch/resampler)
//! and lets each node's own smoothing glide toward them. Because it allocates
//! nothing, locks nothing, and cannot panic, it is safe to call from a device
//! callback, but it is deliberately parameter-only: it holds no audio state.
//!
//! # Combination policy
//!
//! * **Direct gain** multiplies three independent linear factors: distance
//!   attenuation, directional-cone gain, and the occlusion direct gain. Each is
//!   already in `[0, 1]`, so their product is too.
//! * **Pitch** comes solely from the Doppler model applied to the radial
//!   velocity reported by [`Listener::localize`].
//! * **Pan angles** ([`azimuth`](SpatialParams::azimuth) /
//!   [`elevation`](SpatialParams::elevation)) are the listener-local angles of
//!   the source, ready for [`PannerNode`](crate::panner::PannerNode).
//! * **Direct low-pass corner** is the *tighter* (lower) of the air-absorption
//!   corner (a function of distance and atmosphere) and the occlusion corner
//!   (a function of how blocked the path is). Two independent low-pass effects
//!   in series are dominated by whichever cuts lower, so taking the minimum is
//!   a faithful, cheap approximation that avoids cascading two filters.
//! * **Wet (reverb/aux) send scale** is taken straight from the occlusion
//!   model, which only lowers it for true occlusion (an obstruction leaves the
//!   reverberant field untouched).
//!
//! # Determinism
//!
//! Every angle, gain, and corner routes through the leaf models, which in turn
//! use [`bevy_math::ops`] rather than `f32` intrinsics, so a given world
//! configuration resolves to bit-identical parameters across targets and can be
//! golden-compared.
//!
//! # Provenance
//!
//! This module and every model it composes are engine-agnostic and contain
//! **no Unreal Engine, Unity, Godot, Wwise, or FMOD source or derived code**.
//! The combination policy (multiplying independent gains, taking the tighter of
//! two low-pass corners, splitting occlusion across direct and wet paths) is a
//! standard, publicly documented spatial-audio pipeline design.

use prism_audio_core::math::Sample;

use crate::air::{AirAbsorption, AtmosphericConditions};
use crate::attenuation::Attenuation;
use crate::cone::Cone;
use crate::doppler::Doppler;
use crate::geometry::{Emitter, Listener};
use crate::occlusion::{Occlusion, OcclusionFactors};
use crate::spread::{Spread, SpreadParams};

/// Authoring-time description of how a single source spatialises.
///
/// This is plain configuration data (no audio state): cheap to copy and, with
/// the `serialize` feature, (de)serializable. Bundle one per emitter and feed
/// it to [`resolve`] every control block together with the live listener and
/// emitter poses.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceDescriptor {
    /// Distance roll-off curve.
    pub attenuation: Attenuation,
    /// Directional-cone shaping applied around the emitter's facing direction.
    pub cone: Cone,
    /// Doppler pitch model driven by the listener-relative radial velocity.
    pub doppler: Doppler,
    /// Occlusion/obstruction model splitting energy between the direct and wet
    /// paths.
    pub occlusion: Occlusion,
    /// Atmospheric conditions feeding the distance-dependent air-absorption
    /// low-pass corner.
    pub conditions: AtmosphericConditions,
    /// Angular spread/focus shaping (distance-driven image width).
    pub spread: Spread,
}

impl Default for SourceDescriptor {
    /// A neutral, fully-audible source: OpenAL-style inverse-distance
    /// attenuation, an omnidirectional cone, a physically-accurate Doppler, the
    /// default occlusion model (which is inert until real factors arrive), and
    /// the standard reference atmosphere.
    #[inline]
    fn default() -> Self {
        Self {
            attenuation: Attenuation::default(),
            cone: Cone::default(),
            doppler: Doppler::default(),
            occlusion: Occlusion::default(),
            conditions: AtmosphericConditions::default(),
            spread: Spread::default(),
        }
    }
}

/// The resolved, control-rate spatialisation targets for one source.
///
/// Every field is a *target* value; the real-time nodes downstream are expected
/// to smooth toward it rather than jump. All gains are linear and in `[0, 1]`;
/// angles are in radians in the listener-local frame; the corner is in Hz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialParams {
    /// Linear gain for the direct (dry) path: the product of distance
    /// attenuation, cone gain, and occlusion direct gain. In `[0, 1]`.
    pub direct_gain: Sample,
    /// Playback-rate multiplier from the Doppler model (`1.0` is no shift).
    pub pitch_ratio: f32,
    /// Listener-local horizontal azimuth in radians: `0` straight ahead,
    /// positive toward the right ear, in `(-pi, pi]`.
    pub azimuth: Sample,
    /// Listener-local elevation in radians above the horizontal plane, in
    /// `[-pi/2, pi/2]`.
    pub elevation: Sample,
    /// Low-pass corner for the direct path in Hz: the tighter of the
    /// air-absorption and occlusion corners.
    pub direct_cutoff_hz: Sample,
    /// Multiplicative scale for the source's reverb/aux (wet) send, in
    /// `[0, 1]`. Driven by the occlusion factor alone.
    pub wet_gain: Sample,
    /// Angular spread/focus for the pan stage (feed to
    /// [`compute_spread_gains`](crate::spread::compute_spread_gains)).
    pub spread: SpreadParams,
}

/// Resolves the full spatialisation for one source in a single control-rate
/// pass.
///
/// `factors` are the occlusion/obstruction blocking factors for the current
/// listener-emitter pair (query them from an
/// [`OcclusionQuery`](crate::occlusion::OcclusionQuery) or pass
/// [`OcclusionFactors::OPEN`] for a clear line of sight). `sample_rate` is the
/// device rate, used only to clamp the air-absorption corner to the Nyquist
/// frequency.
///
/// The returned [`SpatialParams`] is a pure function of the inputs (no audio
/// state, no allocation, no locks, no panics), so it is safe to call from a
/// real-time thread.
///
/// # Examples
///
/// ```
/// # use bevy_math::Vec3;
/// # use prism_audio_spatial::geometry::{Emitter, Listener};
/// # use prism_audio_spatial::occlusion::OcclusionFactors;
/// # use prism_audio_spatial::spatializer::{resolve, SourceDescriptor};
/// let listener = Listener::default();
/// let emitter = Emitter::point(Vec3::new(0.0, 0.0, -4.0), Vec3::ZERO);
/// let descriptor = SourceDescriptor::default();
/// let params = resolve(&listener, &emitter, &descriptor, OcclusionFactors::OPEN, 48_000);
/// // A source dead ahead sits at zero azimuth and attenuates with distance.
/// assert!(params.azimuth.abs() < 1e-4);
/// assert!(params.direct_gain > 0.0 && params.direct_gain <= 1.0);
/// ```
#[must_use]
pub fn resolve(
    listener: &Listener,
    emitter: &Emitter,
    descriptor: &SourceDescriptor,
    factors: OcclusionFactors,
    sample_rate: u32,
) -> SpatialParams {
    // 1. Resolve the emitter into the listener's frame once; every downstream
    //    model consumes this rather than re-deriving geometry.
    let local = listener.localize(emitter);

    // 2. Occlusion is resolved once and reused for the direct gain, direct
    //    corner, and wet send.
    let occ = descriptor.occlusion.resolve(factors);

    // 3. Direct gain: three independent linear factors, each already in [0, 1].
    //    The cone needs the world-space emitter->listener vector (its own
    //    normalisation handles the coincident case).
    let emitter_to_listener = listener.position - emitter.position;
    let distance_gain = descriptor.attenuation.gain(local.distance);
    let cone_gain = descriptor.cone.gain(emitter.forward, emitter_to_listener);
    let direct_gain = distance_gain * cone_gain * occ.direct_gain;

    // 4. Pitch from the radial velocity reported by localisation.
    let pitch_ratio = descriptor.doppler.pitch_ratio(local.radial_velocity);

    // 5. Direct low-pass corner: the tighter of the air-absorption corner
    //    (distance + atmosphere) and the occlusion corner (blocking factor).
    let air_cutoff =
        AirAbsorption::new(descriptor.conditions).cutoff_hz(local.distance, sample_rate);
    let direct_cutoff_hz = air_cutoff.min(occ.direct_cutoff_hz);

    // 6. Angular spread widens the source image as it approaches the listener.
    let spread = descriptor.spread.resolve(local.distance);

    SpatialParams {
        direct_gain,
        pitch_ratio,
        azimuth: local.azimuth(),
        elevation: local.elevation(),
        direct_cutoff_hz,
        wet_gain: occ.wet_gain,
        spread,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attenuation::DistanceModel;
    use bevy_math::Vec3;
    use core::f32::consts::FRAC_PI_2;

    const SR: u32 = 48_000;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn front_source_has_zero_azimuth_and_elevation() {
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -6.0), Vec3::ZERO);
        let params = resolve(
            &listener,
            &emitter,
            &SourceDescriptor::default(),
            OcclusionFactors::OPEN,
            SR,
        );
        assert!(approx(params.azimuth, 0.0, 1e-4));
        assert!(approx(params.elevation, 0.0, 1e-4));
    }

    #[test]
    fn right_source_has_positive_azimuth() {
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO);
        let params = resolve(
            &listener,
            &emitter,
            &SourceDescriptor::default(),
            OcclusionFactors::OPEN,
            SR,
        );
        // +X (right ear) => +pi/2 azimuth.
        assert!(approx(params.azimuth, FRAC_PI_2, 1e-4));
    }

    #[test]
    fn overhead_source_has_positive_elevation() {
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO);
        let params = resolve(
            &listener,
            &emitter,
            &SourceDescriptor::default(),
            OcclusionFactors::OPEN,
            SR,
        );
        assert!(approx(params.elevation, FRAC_PI_2, 1e-4));
    }

    #[test]
    fn distance_attenuates_direct_gain() {
        let listener = Listener::default();
        let descriptor = SourceDescriptor::default();
        let near = resolve(
            &listener,
            &Emitter::point(Vec3::new(0.0, 0.0, -2.0), Vec3::ZERO),
            &descriptor,
            OcclusionFactors::OPEN,
            SR,
        );
        let far = resolve(
            &listener,
            &Emitter::point(Vec3::new(0.0, 0.0, -20.0), Vec3::ZERO),
            &descriptor,
            OcclusionFactors::OPEN,
            SR,
        );
        assert!(far.direct_gain < near.direct_gain);
        assert!(near.direct_gain <= 1.0);
    }

    #[test]
    fn open_path_leaves_direct_gain_from_geometry_only() {
        // With an open path and an omnidirectional cone at the reference
        // distance, the only factor is distance, which is unity at reference.
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO);
        let params = resolve(
            &listener,
            &emitter,
            &SourceDescriptor::default(),
            OcclusionFactors::OPEN,
            SR,
        );
        assert!(approx(params.direct_gain, 1.0, 1e-4));
        assert!(approx(params.wet_gain, 1.0, 1e-4));
    }

    #[test]
    fn occlusion_lowers_both_direct_and_wet_gain() {
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO);
        let descriptor = SourceDescriptor::default();
        let open = resolve(&listener, &emitter, &descriptor, OcclusionFactors::OPEN, SR);
        let blocked = resolve(
            &listener,
            &emitter,
            &descriptor,
            OcclusionFactors::new(1.0, 1.0),
            SR,
        );
        assert!(blocked.direct_gain < open.direct_gain);
        assert!(blocked.wet_gain < open.wet_gain);
        // A fully blocked path clamps the direct corner to the occlusion floor.
        assert!(blocked.direct_cutoff_hz <= open.direct_cutoff_hz);
    }

    #[test]
    fn obstruction_spares_the_wet_path() {
        // Pure obstruction (no occlusion) darkens/attenuates the direct path
        // but must not scale the reverb send.
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO);
        let descriptor = SourceDescriptor::default();
        let params = resolve(
            &listener,
            &emitter,
            &descriptor,
            OcclusionFactors::new(1.0, 0.0),
            SR,
        );
        assert!(params.direct_gain < 1.0);
        assert!(approx(params.wet_gain, 1.0, 1e-4));
    }

    #[test]
    fn approaching_source_raises_pitch() {
        // Emitter moving toward a stationary listener (negative radial velocity)
        // should raise the perceived pitch above unity.
        let listener = Listener::default();
        let emitter = Emitter::new(
            Vec3::new(0.0, 0.0, -10.0),
            Vec3::new(0.0, 0.0, 30.0), // moving toward the listener (+Z)
            Vec3::NEG_Z,
        );
        let params = resolve(
            &listener,
            &emitter,
            &SourceDescriptor::default(),
            OcclusionFactors::OPEN,
            SR,
        );
        assert!(params.pitch_ratio > 1.0);
    }

    #[test]
    fn cone_facing_away_attenuates_direct_gain() {
        // Emitter at reference distance ahead of the listener but facing away
        // from it: a narrow cone should pull the direct gain below unity while
        // the wet path (open) stays at unity.
        let listener = Listener::default();
        let emitter = Emitter::new(
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::ZERO,
            Vec3::NEG_Z, // facing further away from the listener
        );
        let descriptor = SourceDescriptor {
            cone: Cone::new(0.2, 0.6, 0.1),
            ..SourceDescriptor::default()
        };
        let params = resolve(&listener, &emitter, &descriptor, OcclusionFactors::OPEN, SR);
        assert!(params.direct_gain < 1.0);
        assert!(approx(params.wet_gain, 1.0, 1e-4));
    }

    #[test]
    fn direct_cutoff_is_the_tighter_of_air_and_occlusion() {
        // At a large distance the air-absorption corner is well below Nyquist;
        // resolve must never exceed it even with an open (20 kHz) occlusion
        // corner, and must never exceed Nyquist.
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -500.0), Vec3::ZERO);
        let descriptor = SourceDescriptor::default();
        let air_only = AirAbsorption::new(descriptor.conditions).cutoff_hz(500.0, SR);
        let params = resolve(&listener, &emitter, &descriptor, OcclusionFactors::OPEN, SR);
        assert!(approx(params.direct_cutoff_hz, air_only, 1e-3));
        assert!(params.direct_cutoff_hz <= (SR as f32) * 0.5 + 1e-3);
    }

    #[test]
    fn coincident_source_is_safe_and_centred() {
        // Listener and emitter at the same point: distance collapses to zero,
        // direction to forward, and every parameter stays finite.
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::ZERO, Vec3::ZERO);
        let params = resolve(
            &listener,
            &emitter,
            &SourceDescriptor::default(),
            OcclusionFactors::OPEN,
            SR,
        );
        assert!(params.direct_gain.is_finite());
        assert!(params.pitch_ratio.is_finite());
        assert!(approx(params.azimuth, 0.0, 1e-4));
        assert!(approx(params.elevation, 0.0, 1e-4));
        assert!(approx(params.pitch_ratio, 1.0, 1e-4));
    }

    #[test]
    fn spread_narrows_with_distance() {
        // A near source is enveloping; a distant one collapses toward a point.
        let listener = Listener::default();
        let descriptor = SourceDescriptor::default();
        let near = resolve(
            &listener,
            &Emitter::point(Vec3::new(0.0, 0.0, -0.5), Vec3::ZERO),
            &descriptor,
            OcclusionFactors::OPEN,
            SR,
        );
        let far = resolve(
            &listener,
            &Emitter::point(Vec3::new(0.0, 0.0, -100.0), Vec3::ZERO),
            &descriptor,
            OcclusionFactors::OPEN,
            SR,
        );
        assert!(near.spread.spread > far.spread.spread);
        assert!(far.spread.spread <= 1e-4);
    }

    #[test]
    fn linear_model_respects_max_distance() {
        // Sanity check that a non-default attenuation flows through resolve.
        let listener = Listener::default();
        let descriptor = SourceDescriptor {
            attenuation: Attenuation::new(DistanceModel::Linear, 1.0, 10.0, 1.0),
            ..SourceDescriptor::default()
        };
        let at_max = resolve(
            &listener,
            &Emitter::point(Vec3::new(0.0, 0.0, -10.0), Vec3::ZERO),
            &descriptor,
            OcclusionFactors::OPEN,
            SR,
        );
        // Linear model reaches zero at max_distance.
        assert!(approx(at_max.direct_gain, 0.0, 1e-4));
    }
}
