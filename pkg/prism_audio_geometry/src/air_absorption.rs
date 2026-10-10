//! Per-path frequency-dependent atmospheric air absorption (ISO 9613-1).
//!
//! Geometric arrivals already encode geometric spreading (in the scalar
//! [`gain`](prism_audio_spatial::propagation::PropagationPath::gain)) and
//! surface/edge colouration (in the per-band
//! [`bands`](prism_audio_spatial::propagation::PropagationPath::bands)
//! [`BandGains`]), but nothing yet rolls off the highs as a wave travels the
//! metres between an emitter and a listener. Real air absorbs high frequencies
//! far faster than low ones, so a distant source sounds progressively duller
//! even along a clear line of sight. This module supplies that missing term: it
//! turns the atmosphere and the path's travelled distance into a three-band
//! attenuation spectrum and folds it into the arrival's existing colour.
//!
//! The travelled distance is read straight from the arrival's
//! [`delay_seconds`](prism_audio_spatial::propagation::PropagationPath::delay_seconds),
//! which the path builders already set to the full folded route length (direct,
//! reflected, or diffracted) divided by the speed of sound; multiplying it back
//! by [`SPEED_OF_SOUND_MPS`] recovers the exact metres the wave covered.
//!
//! # Gain convention
//!
//! Air absorption is a purely *spectral* loss: it tilts the arrival darker
//! without changing its broadband level the way geometric spreading does.
//! Accordingly it only ever enters the per-band
//! [`BandGains`] through [`BandGains::combine`] (which multiplies band by band
//! and keeps the product in `[0, 1]`), and it only lowers the scalar
//! [`cutoff_hz`](prism_audio_spatial::propagation::PropagationPath::cutoff_hz)
//! toward the atmospheric corner with a `min`. The scalar
//! [`gain`](prism_audio_spatial::propagation::PropagationPath::gain) is left
//! untouched so the geometric spreading law it carries is preserved, and so the
//! loudest-first ordering and audibility floor the backend applies to secondary
//! arrivals are unaffected by whether this module runs.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. The
//! atmospheric coefficient is the published ISO 9613-1:1993 pure-tone model,
//! evaluated through the spatial crate's existing
//! [`absorption_db_per_metre`] helper.
//!
//! # Relationship
//!
//! Reuses [`prism_audio_spatial::air`] for the ISO 9613-1 coefficient and the
//! distance-dependent low-pass corner, and
//! [`prism_audio_spatial::band_spectrum`] for the three propagation bands it
//! attenuates. It is applied by [`crate::backend::GeometricBackend`] after the
//! direct and secondary arrivals are resolved, and is enabled through
//! [`crate::config::GeometricConfig::with_air_absorption`]. It complements, and
//! never duplicates, the geometric spreading and surface/edge colour the path
//! builders already compute.

use prism_audio_core::math::{db_to_linear, Sample};
use prism_audio_spatial::air::{absorption_db_per_metre, AirAbsorption, AtmosphericConditions};
use prism_audio_spatial::band_spectrum::{
    BandGains, PROPAGATION_BAND_CENTERS, PROPAGATION_BAND_COUNT,
};
use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
use prism_audio_spatial::propagation::PropagationPath;

use crate::config::GeometricConfig;

/// A reusable, distance-independent view of one atmosphere, precomputed so that
/// applying air absorption to a batch of arrivals costs only a multiply and an
/// exponentiation per band per path (never a fresh ISO 9613-1 evaluation).
///
/// Build it once from the [`AtmosphericConditions`] and render `sample_rate`,
/// then call [`AtmosphericFilter::apply`] on each arrival (or
/// [`apply_air_absorption`] on a whole slice).
#[derive(Debug, Clone)]
pub struct AtmosphericFilter {
    /// ISO 9613-1 absorption in decibels per metre at each propagation-band
    /// centre, floored at zero so a (physically impossible) negative coefficient
    /// can never turn into gain.
    per_metre_db: [Sample; PROPAGATION_BAND_COUNT],
    /// The distance-dependent low-pass corner source, shared with the spatial
    /// crate's real-time [`AirAbsorptionNode`](prism_audio_spatial::air::AirAbsorptionNode).
    corner: AirAbsorption,
    /// Render sample rate (Hz), used to clamp the derived corner into a valid
    /// low-pass range.
    sample_rate: u32,
}

impl AtmosphericFilter {
    /// Precomputes the per-band absorption for `conditions` at `sample_rate`.
    #[must_use]
    pub fn new(conditions: AtmosphericConditions, sample_rate: u32) -> Self {
        let per_metre_db = core::array::from_fn(|i| {
            absorption_db_per_metre(&conditions, PROPAGATION_BAND_CENTERS[i]).max(0.0)
        });
        Self {
            per_metre_db,
            corner: AirAbsorption::new(conditions),
            sample_rate,
        }
    }

    /// The three-band attenuation spectrum for a wave that travelled
    /// `distance_m` metres through this atmosphere.
    ///
    /// Each band gain is `10^(-alpha_i * distance / 20)`, i.e. the linear form
    /// of the accumulated decibel loss, so it is always in `(0, 1]` and shrinks
    /// monotonically with distance. The high band (fastest absorption) ends up
    /// the quietest, giving the characteristic distant-source roll-off.
    #[must_use]
    pub fn band_gains(&self, distance_m: Sample) -> BandGains {
        let distance = distance_m.max(0.0);
        BandGains::new(core::array::from_fn(|i| {
            db_to_linear(-self.per_metre_db[i] * distance)
        }))
    }

    /// The atmospheric low-pass corner (Hz) for a wave that travelled
    /// `distance_m` metres, monotonically non-increasing in distance.
    #[must_use]
    pub fn cutoff_hz(&self, distance_m: Sample) -> Sample {
        self.corner.cutoff_hz(distance_m, self.sample_rate)
    }

    /// Folds this atmosphere's distance-dependent roll-off into `path`.
    ///
    /// The travelled distance is recovered from the arrival's own propagation
    /// delay (`delay_seconds * `[`SPEED_OF_SOUND_MPS`]), so a reflected or
    /// diffracted route is attenuated over its full folded length, not the
    /// straight-line emitter-to-listener distance. Per the module's gain
    /// convention this multiplies the per-band spectrum and lowers the scalar
    /// corner, and leaves the scalar broadband gain untouched.
    pub fn apply(&self, path: &mut PropagationPath) {
        let distance = path.delay_seconds.max(0.0) * SPEED_OF_SOUND_MPS;
        path.bands = path.bands.combine(self.band_gains(distance));
        path.cutoff_hz = path.cutoff_hz.min(self.cutoff_hz(distance));
    }
}

/// Applies frequency-dependent atmospheric air absorption to every arrival in
/// `paths`, using the atmosphere and render sample rate in `config`.
///
/// This is the slice-level entry point the backend calls once per query after
/// all arrivals are resolved. It builds a single [`AtmosphericFilter`] (one ISO
/// 9613-1 evaluation per band) and reuses it across every path.
pub fn apply_air_absorption(paths: &mut [PropagationPath], config: &GeometricConfig) {
    let filter = AtmosphericFilter::new(config.atmosphere, config.sample_rate);
    for path in paths.iter_mut() {
        filter.apply(path);
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_air_absorption, AtmosphericFilter};
    use crate::config::GeometricConfig;
    use bevy_math::Vec3;
    use prism_audio_spatial::air::AtmosphericConditions;
    use prism_audio_spatial::band_spectrum::BandGains;
    use prism_audio_spatial::doppler::SPEED_OF_SOUND_MPS;
    use prism_audio_spatial::propagation::{PathKind, PropagationPath, FULL_BAND_CUTOFF_HZ};

    const SR: u32 = 48_000;

    fn path_at_distance(distance_m: f32) -> PropagationPath {
        PropagationPath {
            kind: PathKind::Direct,
            delay_seconds: distance_m / SPEED_OF_SOUND_MPS,
            gain: 1.0,
            cutoff_hz: FULL_BAND_CUTOFF_HZ,
            bands: BandGains::UNITY,
            direction: Vec3::X,
        }
    }

    #[test]
    fn zero_distance_is_transparent() {
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        let gains = filter.band_gains(0.0).bands();
        for g in gains {
            assert!((g - 1.0).abs() < 1e-6, "zero distance must not attenuate");
        }
    }

    #[test]
    fn farther_is_duller_in_the_high_band() {
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        let near = filter.band_gains(10.0).high();
        let far = filter.band_gains(100.0).high();
        assert!(far < near, "the high band must keep dropping with distance");
        assert!(far > 0.0, "a finite distance never fully silences a band");
    }

    #[test]
    fn high_frequencies_absorb_faster_than_low() {
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        let gains = filter.band_gains(100.0);
        assert!(
            gains.high() < gains.mid(),
            "high band must attenuate more than mid"
        );
        assert!(
            gains.mid() < gains.low(),
            "mid band must attenuate more than low"
        );
    }

    #[test]
    fn band_gains_stay_in_unit_interval() {
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        for &d in &[0.0_f32, 1.0, 50.0, 1_000.0, 100_000.0] {
            for g in filter.band_gains(d).bands() {
                assert!(g.is_finite(), "band gain must be finite at distance {d}");
                assert!(
                    (0.0..=1.0).contains(&g),
                    "band gain {g} out of range at {d}"
                );
            }
        }
    }

    #[test]
    fn apply_combines_without_erasing_existing_colour() {
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        let mut path = path_at_distance(80.0);
        // Give the arrival a pre-existing surface colour.
        path.bands = BandGains::new([0.9, 0.8, 0.7]);
        let before = path.bands.bands();
        filter.apply(&mut path);
        let after = path.bands.bands();
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(a <= b, "air absorption only ever attenuates a band");
        }
        // The existing colour is still visible: each band is the original times
        // the (sub-unity) air term, so none collapses to zero at this distance.
        for a in after {
            assert!(a > 0.0, "a finite distance leaves every band audible");
        }
    }

    #[test]
    fn apply_leaves_scalar_gain_untouched() {
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        let mut path = path_at_distance(80.0);
        path.gain = 0.42;
        filter.apply(&mut path);
        assert!(
            (path.gain - 0.42).abs() < 1e-6,
            "air absorption must not change the broadband gain"
        );
    }

    #[test]
    fn apply_lowers_cutoff_from_full_band() {
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        let mut path = path_at_distance(200.0);
        filter.apply(&mut path);
        assert!(
            path.cutoff_hz < FULL_BAND_CUTOFF_HZ,
            "a far arrival must pull the corner below full band"
        );
        assert!(path.cutoff_hz > 0.0, "the corner stays a valid frequency");
    }

    #[test]
    fn apply_uses_the_path_delay_for_distance() {
        // Two arrivals with different delays must attenuate differently even
        // though nothing else differs: the distance comes from the delay.
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        let mut near = path_at_distance(10.0);
        let mut far = path_at_distance(300.0);
        filter.apply(&mut near);
        filter.apply(&mut far);
        assert!(
            far.bands.high() < near.bands.high(),
            "the longer-delay arrival must be duller"
        );
    }

    #[test]
    fn cutoff_is_monotonically_non_increasing() {
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        let mut last = f32::INFINITY;
        for &d in &[0.0_f32, 5.0, 25.0, 100.0, 500.0, 2_000.0] {
            let c = filter.cutoff_hz(d);
            assert!(c <= last + 1e-3, "corner must not rise with distance");
            last = c;
        }
    }

    #[test]
    fn different_atmospheres_give_different_results() {
        let humid = AtmosphericFilter::new(AtmosphericConditions::new(20.0, 80.0, 101.325), SR);
        let dry = AtmosphericFilter::new(AtmosphericConditions::new(20.0, 10.0, 101.325), SR);
        let humid_high = humid.band_gains(100.0).high();
        let dry_high = dry.band_gains(100.0).high();
        assert!(
            (humid_high - dry_high).abs() > 1e-4,
            "humidity must change the high-band absorption"
        );
    }

    #[test]
    fn slice_entry_point_applies_to_every_path() {
        let config = GeometricConfig::new(SR);
        let mut paths = [
            path_at_distance(20.0),
            path_at_distance(120.0),
            path_at_distance(400.0),
        ];
        apply_air_absorption(&mut paths, &config);
        // Every arrival has been darkened in the high band relative to the
        // unity spectrum it started with, and increasingly so with distance.
        assert!(paths[0].bands.high() < 1.0);
        assert!(paths[1].bands.high() < paths[0].bands.high());
        assert!(paths[2].bands.high() < paths[1].bands.high());
    }

    #[test]
    fn extreme_distance_stays_finite_and_bounded() {
        let filter = AtmosphericFilter::new(AtmosphericConditions::default(), SR);
        let mut path = path_at_distance(1.0e9);
        filter.apply(&mut path);
        for g in path.bands.bands() {
            assert!(g.is_finite());
            assert!((0.0..=1.0).contains(&g));
        }
        assert!(path.cutoff_hz.is_finite() && path.cutoff_hz > 0.0);
    }
}
