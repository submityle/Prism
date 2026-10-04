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

/// Hard ceiling on the specular reflection order the backend will trace. Beyond
/// a few bounces the specular image-source contribution is both vanishingly
/// quiet and combinatorially expensive, so the recursive resolver in
/// [`crate::higher_order_reflection`] is clamped to this depth regardless of the
/// configured [`GeometricConfig::max_reflection_order`]. Steam Audio caps its
/// real-time image-source method at a comparable low single-digit order.
pub const MAX_SUPPORTED_REFLECTION_ORDER: usize = 4;

/// Default specular reflection order: first-order only. Higher orders are opt-in
/// through [`GeometricConfig::with_max_reflection_order`] so existing callers
/// keep the original single-bounce behaviour.
pub const DEFAULT_MAX_REFLECTION_ORDER: usize = 1;

/// Hard ceiling on the diffraction order the backend will trace. Each extra
/// edge in a bent route multiplies the number of candidate edge sequences and
/// adds another attenuating wedge, so the multi-edge resolver in
/// [`crate::higher_order_diffraction`] is clamped to this depth regardless of
/// the configured [`GeometricConfig::max_diffraction_order`]. Steam Audio's
/// path tracer likewise keeps real-time diffraction to a low single-digit order.
pub const MAX_SUPPORTED_DIFFRACTION_ORDER: usize = 3;

/// Default diffraction order: single-edge only. Higher orders (sequential
/// bends around two or more edges, as in an L-shaped corridor) are opt-in
/// through [`GeometricConfig::with_max_diffraction_order`] so existing callers
/// keep the original single-edge behaviour.
pub const DEFAULT_MAX_DIFFRACTION_ORDER: usize = 1;

/// Default budget for coupled reflection-and-diffraction arrivals retained per
/// query (strongest kept). Each coupled arrival pairs one specular bounce with
/// one shadow-edge bend, so the combinatorial cost is bounded independently of
/// the pure-reflection and pure-diffraction budgets.
pub const DEFAULT_MAX_COUPLED_PATHS: usize = 8;

/// Which edge-diffraction model the backend evaluates for shadowed arrivals.
///
/// Both models consume the same resolved detour geometry (the least-detour
/// corner on a diffracting edge and the clear two-leg bent route); they differ
/// only in how the shadow attenuation and its spectral colour are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DiffractionModel {
    /// Maekawa's semi-empirical barrier model: one broadband gain from the
    /// Fresnel number of the detour plus a detour-dependent single-pole
    /// low-pass corner. Cheap, robust, and the backward-compatible default.
    #[default]
    Maekawa,
    /// The Kouyoumjian-Pathak Uniform Theory of Diffraction for a sound-hard
    /// wedge: a frequency-dependent complex coefficient sampled per propagation
    /// band, capturing the wedge opening angle and incidence geometry that the
    /// barrier model ignores. Physically grounded at the cost of more work,
    /// evaluated through [`UtdWedge`](prism_audio_spatial::UtdWedge).
    Utd,
}

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
    /// Maximum specular reflection order traced (`1` = first-order only; `2` or
    /// more adds the recursive higher-order image-source bounces, clamped to
    /// [`MAX_SUPPORTED_REFLECTION_ORDER`]).
    pub max_reflection_order: usize,
    /// Maximum diffraction edges retained per query (least-detour kept).
    pub max_diffractions: usize,
    /// Maximum diffraction order traced (`1` = single-edge only; `2` or more
    /// enables the sequential multi-edge bends from
    /// [`crate::higher_order_diffraction`], clamped to
    /// [`MAX_SUPPORTED_DIFFRACTION_ORDER`]).
    pub max_diffraction_order: usize,
    /// Linear-gain floor below which an arrival is dropped.
    pub min_gain: Sample,
    /// Surface offset (metres) applied when re-launching rays off geometry.
    pub surface_epsilon_m: Sample,
    /// Sample rate (Hz) used to derive diffraction low-pass corners, matching
    /// the voice that renders the resolved paths.
    pub sample_rate: u32,
    /// Reference frequency (Hz) for broadband diffraction attenuation.
    pub diffraction_freq_hz: Sample,
    /// Which diffraction model to evaluate for shadowed arrivals.
    pub diffraction_model: DiffractionModel,
    /// Whether to trace coupled reflection-and-diffraction arrivals (a specular
    /// bounce followed by an edge bend, or an edge bend followed by a bounce).
    /// Opt-in through [`GeometricConfig::with_coupled_paths`]; defaults to
    /// `false` so existing callers keep the original reflection/diffraction set.
    pub coupled_enabled: bool,
    /// Maximum coupled reflection-and-diffraction arrivals retained per query
    /// (strongest kept), evaluated only when [`Self::coupled_enabled`] is set.
    pub max_coupled_paths: usize,
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
            max_reflection_order: DEFAULT_MAX_REFLECTION_ORDER,
            max_diffractions: DEFAULT_MAX_DIFFRACTIONS,
            max_diffraction_order: DEFAULT_MAX_DIFFRACTION_ORDER,
            min_gain: DEFAULT_MIN_GAIN,
            surface_epsilon_m: DEFAULT_SURFACE_EPSILON_M,
            sample_rate,
            diffraction_freq_hz: DEFAULT_DIFFRACTION_FREQ_HZ,
            diffraction_model: DiffractionModel::Maekawa,
            coupled_enabled: false,
            max_coupled_paths: DEFAULT_MAX_COUPLED_PATHS,
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

    /// Returns a copy with the maximum specular reflection order set to `order`
    /// (clamped at query time to [`MAX_SUPPORTED_REFLECTION_ORDER`]). An order of
    /// `1` keeps only first-order bounces; `2` or more enables the recursive
    /// higher-order image-source arrivals from [`crate::higher_order_reflection`].
    #[inline]
    #[must_use]
    pub fn with_max_reflection_order(mut self, order: usize) -> Self {
        self.max_reflection_order = order;
        self
    }

    /// Returns a copy with the diffraction-edge budget set to `max`.
    #[inline]
    #[must_use]
    pub fn with_max_diffractions(mut self, max: usize) -> Self {
        self.max_diffractions = max;
        self
    }

    /// Returns a copy with the maximum diffraction order set to `order`
    /// (clamped at query time to [`MAX_SUPPORTED_DIFFRACTION_ORDER`]). An order
    /// of `1` keeps only single-edge bends; `2` or more enables the sequential
    /// multi-edge diffractions from [`crate::higher_order_diffraction`].
    #[inline]
    #[must_use]
    pub fn with_max_diffraction_order(mut self, order: usize) -> Self {
        self.max_diffraction_order = order;
        self
    }

    /// Returns a copy that evaluates edge diffraction with `model` (the default
    /// is [`DiffractionModel::Maekawa`]).
    #[inline]
    #[must_use]
    pub fn with_diffraction_model(mut self, model: DiffractionModel) -> Self {
        self.diffraction_model = model;
        self
    }

    /// Returns a copy with coupled reflection-and-diffraction tracing enabled.
    ///
    /// A coupled arrival reflects off one face and bends over one diffracting
    /// edge (in either order), the next-order route beyond a lone bounce or a
    /// lone bend. Enabling this complements, and never duplicates, the pure
    /// reflections from [`crate::reflection_path`] and the pure diffractions
    /// from [`crate::diffraction_path`].
    #[inline]
    #[must_use]
    pub fn with_coupled_paths(mut self) -> Self {
        self.coupled_enabled = true;
        self
    }

    /// Returns a copy with coupled reflection-and-diffraction tracing disabled.
    #[inline]
    #[must_use]
    pub fn without_coupled_paths(mut self) -> Self {
        self.coupled_enabled = false;
        self
    }

    /// Returns a copy with the coupled-arrival budget set to `max` (`0` disables
    /// coupled arrivals even when [`Self::coupled_enabled`] is set).
    #[inline]
    #[must_use]
    pub fn with_max_coupled_paths(mut self, max: usize) -> Self {
        self.max_coupled_paths = max;
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
    use super::{DiffractionModel, GeometricConfig};

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

    #[test]
    fn diffraction_model_defaults_to_maekawa_and_is_selectable() {
        assert_eq!(
            GeometricConfig::new(48_000).diffraction_model,
            DiffractionModel::Maekawa
        );
        let utd = GeometricConfig::new(48_000).with_diffraction_model(DiffractionModel::Utd);
        assert_eq!(utd.diffraction_model, DiffractionModel::Utd);
    }

    #[test]
    fn reflection_order_defaults_to_first_order_and_is_selectable() {
        assert_eq!(
            GeometricConfig::new(48_000).max_reflection_order,
            super::DEFAULT_MAX_REFLECTION_ORDER
        );
        let cfg = GeometricConfig::new(48_000).with_max_reflection_order(3);
        assert_eq!(cfg.max_reflection_order, 3);
    }

    #[test]
    fn diffraction_order_defaults_to_single_edge_and_is_selectable() {
        assert_eq!(
            GeometricConfig::new(48_000).max_diffraction_order,
            super::DEFAULT_MAX_DIFFRACTION_ORDER
        );
        let cfg = GeometricConfig::new(48_000).with_max_diffraction_order(2);
        assert_eq!(cfg.max_diffraction_order, 2);
    }

    #[test]
    fn coupled_paths_default_off_and_are_opt_in() {
        let cfg = GeometricConfig::new(48_000);
        assert!(!cfg.coupled_enabled);
        assert_eq!(cfg.max_coupled_paths, super::DEFAULT_MAX_COUPLED_PATHS);
        let on = cfg.with_coupled_paths();
        assert!(on.coupled_enabled);
        assert!(!on.without_coupled_paths().coupled_enabled);
    }

    #[test]
    fn coupled_budget_is_settable() {
        let cfg = GeometricConfig::new(48_000).with_max_coupled_paths(3);
        assert_eq!(cfg.max_coupled_paths, 3);
    }
}
