//! Control-rate configuration for the geometric propagation backend.
//!
//! A [`GeometricConfig`] bounds how much geometry the backend traces per query
//! and which propagation mechanisms are enabled. It is plain, cheap-to-copy
//! description data (not touched on the real-time audio thread): the backend
//! reads it once per control-rate query to decide how many reflectors and edges
//! to consider and when an arrival is too quiet to be worth a voice slot.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Consumed by [`crate::backend::GeometricBackend`] and the per-mechanism path
//! builders in [`crate::direct_path`], [`crate::reflection_path`], and
//! [`crate::diffraction_path`]. Caps the output at
//! [`MAX_PROPAGATION_PATHS`](prism_audio_spatial::propagation::MAX_PROPAGATION_PATHS).

use prism_audio_core::math::Sample;

/// Default audibility floor (linear gain). Arrivals quieter than this are
/// dropped rather than consuming a bounded path slot: -60 dB is the classic
/// "effectively inaudible against a full-scale direct path" threshold.
pub const DEFAULT_MIN_GAIN: Sample = 0.001;

/// Default maximum number of first-order specular reflectors the backend keeps
/// per query. The strongest reflections win when more candidates qualify.
pub const DEFAULT_MAX_REFLECTIONS: usize = 4;

/// Default maximum number of diffraction edges the backend keeps per query.
pub const DEFAULT_MAX_DIFFRACTIONS: usize = 2;

/// Default surface offset (metres) used to lift ray origins off the geometry
/// they just touched, so a reflected or transmitted ray does not immediately
/// re-hit its own surface through floating-point coincidence.
pub const DEFAULT_SURFACE_EPSILON_M: Sample = 1.0e-3;

/// Reference frequency (Hz) at which broadband diffraction and the shadow
/// low-pass corner are evaluated. 1 kHz is the standard single-number acoustic
/// reference used across the spatial crate's helpers.
pub const DEFAULT_DIFFRACTION_FREQ_HZ: Sample = 1_000.0;

/// Control-rate budget and feature switches for a geometric propagation query.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct GeometricConfig {
    /// Whether to trace first-order specular reflections (image-source method).
    pub reflections_enabled: bool,
    /// Whether to trace edge diffraction when the direct path is shadowed.
    pub diffraction_enabled: bool,
    /// Whether a shadowed direct path still contributes a transmitted arrival
    /// (sound leaking through the blocking partitions). When `false`, a fully
    /// blocked direct path is reported silent.
    pub transmission_enabled: bool,
    /// Maximum first-order reflectors retained per query (strongest kept).
    pub max_reflections: usize,
    /// Maximum diffraction edges retained per query (least-detour kept).
    pub max_diffractions: usize,
    /// Linear-gain floor below which an arrival is dropped.
    pub min_gain: Sample,
    /// Surface offset (metres) applied when re-launching rays off geometry.
    pub surface_epsilon_m: Sample,
    /// Sample rate (Hz) used to derive diffraction low-pass corners, matching
    /// the voice that renders the resolved paths.
    pub sample_rate: u32,
    /// Reference frequency (Hz) for broadband diffraction attenuation.
    pub diffraction_freq_hz: Sample,
}

impl GeometricConfig {
    /// Builds a configuration for a given render sample rate, enabling all
    /// mechanisms with the default budgets.
    #[inline]
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            reflections_enabled: true,
            diffraction_enabled: true,
            transmission_enabled: true,
            max_reflections: DEFAULT_MAX_REFLECTIONS,
            max_diffractions: DEFAULT_MAX_DIFFRACTIONS,
            min_gain: DEFAULT_MIN_GAIN,
            surface_epsilon_m: DEFAULT_SURFACE_EPSILON_M,
            sample_rate,
            diffraction_freq_hz: DEFAULT_DIFFRACTION_FREQ_HZ,
        }
    }

    /// Returns a copy with specular reflections disabled.
    #[inline]
    #[must_use]
    pub fn without_reflections(mut self) -> Self {
        self.reflections_enabled = false;
        self
    }

    /// Returns a copy with edge diffraction disabled.
    #[inline]
    #[must_use]
    pub fn without_diffraction(mut self) -> Self {
        self.diffraction_enabled = false;
        self
    }

    /// Returns a copy with the reflector budget set to `max` (clamped so the
    /// query always considers at least nothing-is-forced: `0` disables them).
    #[inline]
    #[must_use]
    pub fn with_max_reflections(mut self, max: usize) -> Self {
        self.max_reflections = max;
        self
    }

    /// Returns a copy with the diffraction-edge budget set to `max`.
    #[inline]
    #[must_use]
    pub fn with_max_diffractions(mut self, max: usize) -> Self {
        self.max_diffractions = max;
        self
    }

    /// Returns a copy with the audibility floor set to `gain` (clamped to be
    /// non-negative).
    #[inline]
    #[must_use]
    pub fn with_min_gain(mut self, gain: Sample) -> Self {
        self.min_gain = gain.max(0.0);
        self
    }
}

impl Default for GeometricConfig {
    /// A 48 kHz configuration with every mechanism enabled at default budgets.
    #[inline]
    fn default() -> Self {
        Self::new(48_000)
    }
}

#[cfg(test)]
mod tests {
    use super::GeometricConfig;

    #[test]
    fn new_enables_all_mechanisms() {
        let cfg = GeometricConfig::new(44_100);
        assert!(cfg.reflections_enabled);
        assert!(cfg.diffraction_enabled);
        assert!(cfg.transmission_enabled);
        assert_eq!(cfg.sample_rate, 44_100);
    }

    #[test]
    fn builders_toggle_fields() {
        let cfg = GeometricConfig::new(48_000)
            .without_reflections()
            .without_diffraction()
            .with_max_reflections(7)
            .with_max_diffractions(3)
            .with_min_gain(0.01);
        assert!(!cfg.reflections_enabled);
        assert!(!cfg.diffraction_enabled);
        assert_eq!(cfg.max_reflections, 7);
        assert_eq!(cfg.max_diffractions, 3);
        assert!((cfg.min_gain - 0.01).abs() < 1e-9);
    }

    #[test]
    fn min_gain_floor_is_non_negative() {
        let cfg = GeometricConfig::new(48_000).with_min_gain(-5.0);
        assert!(cfg.min_gain >= 0.0);
    }

    #[test]
    fn default_is_48k() {
        assert_eq!(GeometricConfig::default().sample_rate, 48_000);
    }
}
