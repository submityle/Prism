//! Perceptual encoding: distilling a time-domain impulse response into the
//! compact, quantisable parameter set stored per probe.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Implements the "Perceptual Encoding" stage of design section 43: rather
//! than storing bulky impulse responses, each probe keeps a small set of
//! perceptual parameters derived from its response. The sub-modules compute
//! the individual measures ([`energy`], [`decay`], [`direction`],
//! [`wet_dry`]); [`encode_perceptual`] assembles them into a
//! [`PerceptualParams`], and [`PerceptualParams::to_spatial`] projects that
//! onto the shared [`prism_audio_spatial::SpatialParams`] bus.

pub mod decay;
pub mod direction;
pub mod energy;
pub mod wet_dry;

pub use direction::DirectionalProbe;

use bevy_math::ops;
use prism_audio_spatial::{SpatialParams, SpreadParams};

use crate::solver::ImpulseResponse;

/// Default direct-arrival window in milliseconds (direct sound and the first
/// wavefront).
pub const DEFAULT_DIRECT_WINDOW_MS: f32 = 5.0;
/// Default early window in milliseconds (direct sound plus early reflections).
pub const DEFAULT_EARLY_WINDOW_MS: f32 = 50.0;
/// Default onset detection threshold, as a fraction of the response peak.
pub const DEFAULT_ONSET_THRESHOLD: f32 = 0.1;
/// Default lower bound of the diffraction low-pass corner, in Hz.
pub const DEFAULT_MIN_CUTOFF_HZ: f32 = 250.0;
/// Default upper bound of the diffraction low-pass corner, in Hz.
pub const DEFAULT_MAX_CUTOFF_HZ: f32 = 20_000.0;
/// Default decibel range over which the `DRR` maps to the wet send.
pub const DEFAULT_DRR_RANGE_DB: f32 = 20.0;

/// Tunable windows and ranges for [`encode_perceptual`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EncodeConfig {
    /// Direct-arrival window in milliseconds.
    pub direct_window_ms: f32,
    /// Early (direct plus early reflections) window in milliseconds.
    pub early_window_ms: f32,
    /// Onset threshold as a fraction of the response peak magnitude.
    pub onset_threshold: f32,
    /// Lower bound of the low-pass corner in Hz (fully diffracted path).
    pub min_cutoff_hz: f32,
    /// Upper bound of the low-pass corner in Hz (fully open path).
    pub max_cutoff_hz: f32,
    /// Decibel range mapping the `DRR` onto the wet send gain.
    pub drr_range_db: f32,
}

impl Default for EncodeConfig {
    #[inline]
    fn default() -> Self {
        Self {
            direct_window_ms: DEFAULT_DIRECT_WINDOW_MS,
            early_window_ms: DEFAULT_EARLY_WINDOW_MS,
            onset_threshold: DEFAULT_ONSET_THRESHOLD,
            min_cutoff_hz: DEFAULT_MIN_CUTOFF_HZ,
            max_cutoff_hz: DEFAULT_MAX_CUTOFF_HZ,
            drr_range_db: DEFAULT_DRR_RANGE_DB,
        }
    }
}

/// The compact perceptual parameter set stored at each probe.
///
/// Every field is a scalar; a whole field is a lattice of these. The runtime
/// lookup interpolates them and [`PerceptualParams::to_spatial`] projects the
/// result onto [`prism_audio_spatial::SpatialParams`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PerceptualParams {
    /// Occlusion direct-path gain in `[0, 1]`.
    pub direct_gain: f32,
    /// Direct-path low-pass corner in Hz.
    pub direct_cutoff_hz: f32,
    /// Reverberation time `RT60` in seconds.
    pub rt60_s: f32,
    /// Reverb (wet) send gain in `[0, 1]`.
    pub wet_gain: f32,
    /// Listener-local arrival azimuth in radians.
    pub azimuth: f32,
    /// Listener-local arrival elevation in radians.
    pub elevation: f32,
    /// Direct-to-reverberant ratio in decibels.
    pub drr_db: f32,
}

impl PerceptualParams {
    /// A fully open path: unity direct gain, wide-open corner, no reverb send,
    /// straight-ahead arrival.
    pub const OPEN: Self = Self {
        direct_gain: 1.0,
        direct_cutoff_hz: DEFAULT_MAX_CUTOFF_HZ,
        rt60_s: 0.0,
        wet_gain: 0.0,
        azimuth: 0.0,
        elevation: 0.0,
        drr_db: wet_dry::MAX_DRR_DB,
    };

    /// A fully occluded path: no direct energy, tight corner, maximum reverb
    /// send.
    pub const OCCLUDED: Self = Self {
        direct_gain: 0.0,
        direct_cutoff_hz: DEFAULT_MIN_CUTOFF_HZ,
        rt60_s: 0.0,
        wet_gain: 1.0,
        azimuth: 0.0,
        elevation: 0.0,
        drr_db: wet_dry::MIN_DRR_DB,
    };

    /// Linearly interpolates toward `other` by `t` in `[0, 1]`.
    ///
    /// Scalar fields blend linearly; the azimuth and elevation angles blend on
    /// the unit circle to avoid wraparound artefacts.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let s = 1.0 - t;
        Self {
            direct_gain: self.direct_gain * s + other.direct_gain * t,
            direct_cutoff_hz: self.direct_cutoff_hz * s + other.direct_cutoff_hz * t,
            rt60_s: self.rt60_s * s + other.rt60_s * t,
            wet_gain: self.wet_gain * s + other.wet_gain * t,
            azimuth: lerp_angle(self.azimuth, other.azimuth, t),
            elevation: lerp_angle(self.elevation, other.elevation, t),
            drr_db: self.drr_db * s + other.drr_db * t,
        }
    }

    /// Projects the parameters onto the shared spatial parameter bus.
    ///
    /// The wave backend contributes no Doppler (unity pitch) and a point image
    /// (spread is the geometric backend's responsibility), so those fields take
    /// neutral values.
    #[must_use]
    pub fn to_spatial(&self) -> SpatialParams {
        SpatialParams {
            direct_gain: self.direct_gain,
            pitch_ratio: 1.0,
            azimuth: self.azimuth,
            elevation: self.elevation,
            direct_cutoff_hz: self.direct_cutoff_hz,
            wet_gain: self.wet_gain,
            spread: SpreadParams::POINT,
        }
    }
}

impl Default for PerceptualParams {
    #[inline]
    fn default() -> Self {
        Self::OPEN
    }
}

/// Interpolates between two angles on the unit circle by `t` in `[0, 1]`.
///
/// Blending `(cos, sin)` and taking the argument avoids the discontinuity that
/// a raw linear blend hits near the `+pi` / `-pi` wrap.
#[must_use]
pub fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    let s = 1.0 - t;
    let x = s * ops::cos(a) + t * ops::cos(b);
    let y = s * ops::sin(a) + t * ops::sin(b);
    if x * x + y * y <= 1.0e-20 {
        a
    } else {
        ops::atan2(y, x)
    }
}

/// Converts a time in milliseconds to a sample count at `sample_rate` Hz,
/// never returning zero.
#[must_use]
fn ms_to_samples(ms: f32, sample_rate: f32) -> usize {
    let n = ops::round(ms.max(0.0) * 1.0e-3 * sample_rate);
    if n < 1.0 {
        1
    } else {
        n as usize
    }
}

/// Encodes one impulse response into [`PerceptualParams`].
///
/// `reference_energy` is the free-field direct energy at the same probe (used
/// to separate occlusion from distance); pass a non-positive value when no
/// reference is available, in which case the direct gain is treated as open.
/// `direction` supplies the six neighbour responses for arrival-direction
/// estimation; pass `None` to leave the arrival straight ahead.
///
/// # Examples
///
/// ```
/// # use prism_audio_wave::encoding::{encode_perceptual, EncodeConfig};
/// # use prism_audio_wave::solver::ImpulseResponse;
/// // A lone direct spike with no tail: full direct gain, no reverb send.
/// let mut p = alloc_pulse();
/// let ir = ImpulseResponse::new(8_000.0, p);
/// let params = encode_perceptual(&ir, None, ir.total_energy(), &EncodeConfig::default());
/// assert!(params.direct_gain > 0.9);
/// assert!(params.wet_gain < 0.1);
/// # fn alloc_pulse() -> Vec<f32> { let mut v = vec![0.0; 64]; v[0] = 1.0; v }
/// ```
#[must_use]
pub fn encode_perceptual(
    ir: &ImpulseResponse,
    direction: Option<&DirectionalProbe>,
    reference_energy: f32,
    config: &EncodeConfig,
) -> PerceptualParams {
    let sr = ir.sample_rate();
    let onset = ir.onset_index(config.onset_threshold);
    let direct_win = ms_to_samples(config.direct_window_ms, sr);
    let early_win = ms_to_samples(config.early_window_ms, sr);

    let direct = energy::direct_energy(ir, onset, direct_win);
    let reverberant = energy::reverberant_energy(ir, onset, early_win);

    let direct_gain = energy::occlusion_gain(direct, reference_energy);
    let direct_cutoff_hz =
        energy::direct_cutoff(ir, onset, direct_win, config.min_cutoff_hz, config.max_cutoff_hz);
    let rt60_s = decay::rt60(ir, onset);
    let drr = wet_dry::drr_db(direct, reverberant);
    let wet_gain = wet_dry::wet_gain_from_drr(drr, config.drr_range_db);

    let (azimuth, elevation) = match direction {
        Some(probe) => {
            let dir = probe.arrival_direction(onset, early_win);
            direction::direction_to_angles(dir)
        }
        None => (0.0, 0.0),
    };

    PerceptualParams {
        direct_gain,
        direct_cutoff_hz,
        rt60_s,
        wet_gain,
        azimuth,
        elevation,
        drr_db: drr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::f32::consts::PI;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn dry_spike_is_open_and_dry() {
        let mut p = vec![0.0_f32; 128];
        p[0] = 1.0;
        let ir = ImpulseResponse::new(8_000.0, p);
        let params = encode_perceptual(&ir, None, ir.total_energy(), &EncodeConfig::default());
        assert!(params.direct_gain > 0.9, "direct_gain {}", params.direct_gain);
        assert!(params.wet_gain < 0.1, "wet_gain {}", params.wet_gain);
    }

    #[test]
    fn reverberant_tail_sends_more_wet() {
        // Short direct spike plus a long decaying tail -> more reverberant.
        let mut p = vec![0.0_f32; 4096];
        p[0] = 1.0;
        for (n, v) in p.iter_mut().enumerate().skip(1) {
            *v = 0.5 * ops::exp(-0.002 * n as f32);
        }
        let ir = ImpulseResponse::new(8_000.0, p);
        let params = encode_perceptual(&ir, None, 1.0, &EncodeConfig::default());
        assert!(params.wet_gain > 0.1, "wet_gain {}", params.wet_gain);
        assert!(params.rt60_s > 0.0, "rt60 {}", params.rt60_s);
    }

    #[test]
    fn to_spatial_is_neutral_in_pitch_and_spread() {
        let params = PerceptualParams::OPEN;
        let sp = params.to_spatial();
        assert!(approx(sp.pitch_ratio, 1.0, 1e-6));
        assert!(approx(sp.spread.half_width, 0.0, 1e-6));
        assert!(approx(sp.direct_gain, 1.0, 1e-6));
    }

    #[test]
    fn lerp_blends_endpoints() {
        let a = PerceptualParams::OPEN;
        let b = PerceptualParams::OCCLUDED;
        let mid = a.lerp(&b, 0.5);
        assert!(approx(mid.direct_gain, 0.5, 1e-6));
        assert!(approx(a.lerp(&b, 0.0).direct_gain, 1.0, 1e-6));
        assert!(approx(a.lerp(&b, 1.0).direct_gain, 0.0, 1e-6));
    }

    #[test]
    fn lerp_angle_takes_short_way_around_the_wrap() {
        // Halfway between +170 and -170 degrees is +/-180, not 0.
        let a = 170.0_f32.to_radians();
        let b = (-170.0_f32).to_radians();
        let mid = lerp_angle(a, b, 0.5);
        assert!(mid.abs() > PI - 0.1, "expected near +/-pi, got {mid}");
    }
}
