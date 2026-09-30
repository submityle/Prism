//! Sound propagation beyond the direct line of sight: transmission through
//! materials, edge diffraction into acoustic shadows, and specular reflection
//! off surfaces.
//!
//! The leaf models in this crate ([`attenuation`](crate::attenuation),
//! [`cone`](crate::cone), [`air`](crate::air), ...) all describe the *direct*
//! path. Real rooms are richer: sound leaks through walls, bends around
//! corners, and bounces off surfaces to arrive from several directions with
//! different delays and colours. This module supplies the classic,
//! publicly documented acoustics that turn that geometry into audible
//! parameters, plus the plumbing that keeps the geometry itself out of this
//! crate.
//!
//! # Layering: a pluggable backend
//!
//! Just as [`occlusion`](crate::occlusion) delegates ray casts to an
//! [`OcclusionQuery`](crate::occlusion::OcclusionQuery), the richer geometry
//! here is delegated to a [`PropagationBackend`]. A backend traces the scene
//! (edges, portals, reflectors) and reports a set of [`PropagationPath`]s plus
//! a [`PropagationSummary`]; this crate never depends on `prism_physics`. The
//! built-in [`FreeFieldBackend`] models an open field: a single direct path and
//! no secondary arrivals, which is the correct default for headless use and
//! tests.
//!
//! # The physics (all classic DSP)
//!
//! * **Transmission.** A wall attenuates sound by its transmission loss (dB);
//!   [`transmission_gain`] converts that to a linear factor.
//! * **Diffraction.** Sound bends around an edge into the geometric shadow, but
//!   is attenuated by an amount that grows with frequency and with the detour
//!   the wave must take. This is modelled with the **Maekawa (1968)** barrier
//!   chart via the [Fresnel number](fresnel_number): see
//!   [`maekawa_attenuation_db`]. [`diffraction_gain`] gives the broadband
//!   factor at a frequency and [`diffraction_cutoff_hz`] the equivalent
//!   low-pass corner (shadowed sound is duller).
//! * **Reflection.** A surface returns a fraction of the incident energy set by
//!   its reflection coefficient; the delayed, attenuated image arrives from the
//!   mirror direction. The geometry of the bounce is the backend's job; this
//!   module only carries the per-path gain/delay/colour.
//!
//! # Control rate, not audio rate
//!
//! [`PropagationBackend::query`] runs at control rate (it may trace geometry
//! and is **not** required to be real-time safe). Its output — a small fixed
//! array of [`PropagationPath`]s — then drives the real-time voice: each path
//! feeds a delay + gain + low-pass, exactly the primitives already implemented
//! in [`prism_audio_core`]. The pure helper functions in this module
//! ([`fresnel_number`], [`maekawa_attenuation_db`], [`diffraction_gain`],
//! [`diffraction_cutoff_hz`], [`transmission_gain`], [`edge_path_difference`])
//! allocate nothing, lock nothing, and cannot panic.
//!
//! # Determinism
//!
//! Every transcendental (`sqrt`, `exp`, `ln`) routes through
//! [`bevy_math::ops`] (libm-backed) rather than an `f32` intrinsic, so the
//! attenuation and the derived corner are bit-reproducible across targets and
//! can be golden-compared. `tanh` is built from `exp` and `log10` from `ln`
//! (neither is exposed by `bevy_math::ops`), keeping the whole path libm-backed.
//!
//! # Provenance
//!
//! The barrier-diffraction relation is the publicly documented **Maekawa,
//! "Noise reduction by screens", Applied Acoustics 1(3), 1968**, expressed
//! through the Fresnel number `N = 2 delta / lambda`, as reproduced in standard
//! acoustics references. Transmission loss and reflection coefficients are
//! textbook definitions. This module is engine-agnostic and contains **no
//! Unreal Engine, Unity, Godot, Wwise, or FMOD source or derived code**; it is
//! implemented purely from that publicly documented acoustics knowledge.

use bevy_math::{Vec3, ops};
use prism_audio_core::math::{Sample, db_to_linear};

use crate::doppler::SPEED_OF_SOUND_MPS;
use crate::geometry::{Emitter, Listener};
use crate::occlusion::OcclusionFactors;

use core::f32::consts::{LN_10, PI};

/// Largest number of propagation paths a [`PropagationBackend`] may report for
/// one source. Backends fill up to this many entries in the caller's buffer;
/// extra geometry is folded or dropped so the real-time voice stays bounded.
pub const MAX_PROPAGATION_PATHS: usize = 8;

/// Upper bound (dB) on the diffraction attenuation returned by
/// [`maekawa_attenuation_db`]. Maekawa's chart keeps rising slowly with the
/// Fresnel number; real barriers level off near this figure, so the model is
/// clamped here to stay physically plausible and to keep derived gains away
/// from silence.
pub const MAX_DIFFRACTION_DB: Sample = 24.0;

/// Absolute lower bound (Hz) for any derived low-pass corner, matching
/// [`crate::air`].
const MIN_CUTOFF_HZ: Sample = 20.0;

/// Total shadow attenuation (dB) that marks a frequency as "rolled off" in
/// [`diffraction_cutoff_hz`].
///
/// A source on the shadow boundary already loses ~5 dB broadband (Maekawa at
/// `N = 0`); the low-pass corner is where the *frequency-dependent* excess on
/// top of that floor becomes clearly audible. `8 dB` total ≈ 5 dB floor + 3 dB
/// of high-frequency roll-off.
const DIFFRACTION_CUTOFF_THRESHOLD_DB: Sample = 8.0;

/// Corner (Hz) reported for a full-band, unfiltered path. Consumers clamp this
/// to the running Nyquist frequency before designing a filter; it is a large
/// sentinel meaning "do not low-pass".
pub const FULL_BAND_CUTOFF_HZ: Sample = 1.0e6;

/// Fixed logarithmic frequency grid (Hz) scanned by
/// [`diffraction_cutoff_hz`], mirroring [`crate::air`]'s grid so both corners
/// share a resolution. Storing it as a `const` keeps the search on the stack.
const FREQUENCY_GRID: [Sample; 32] = [
    40.0, 49.1675, 60.436, 74.2871, 91.3127, 112.2403, 137.9643, 169.5838, 208.4502, 256.2241,
    314.9472, 387.1289, 475.8536, 584.9128, 718.967, 883.7445, 1086.2867, 1335.2491, 1641.2701,
    2017.4271, 2479.7942, 3048.1294, 3746.7197, 4605.4175, 5660.9165, 6958.3223, 8553.076,
    10513.325, 12922.838, 15884.578, 19525.11, 24000.0,
];

/// The acoustic behaviour of a surface or partition.
///
/// Plain, authoring-time description data (cheap to copy, `serialize`-able).
/// Values come from the caller's material library; this crate ships only the
/// [`OPEN`](Self::OPEN) free-air default rather than a fabricated table.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AcousticMaterial {
    /// Sound transmission loss through the material, in decibels. Higher means
    /// more is blocked; `0` passes the signal untouched. Negative inputs are
    /// treated as `0` by [`Self::transmission_gain`].
    pub transmission_loss_db: Sample,
    /// Fraction of incident energy the surface reflects, in `[0, 1]`. `0` is
    /// perfectly absorptive, `1` a perfect mirror. Clamped by
    /// [`Self::reflection_gain`].
    pub reflection: Sample,
}

impl AcousticMaterial {
    /// Free air: nothing is blocked and nothing is reflected.
    pub const OPEN: Self = Self {
        transmission_loss_db: 0.0,
        reflection: 0.0,
    };

    /// Builds a material from its transmission loss and reflection coefficient.
    #[inline]
    #[must_use]
    pub const fn new(transmission_loss_db: Sample, reflection: Sample) -> Self {
        Self {
            transmission_loss_db,
            reflection,
        }
    }

    /// The linear gain applied to sound transmitted *through* the material,
    /// in `(0, 1]`. See [`transmission_gain`].
    #[inline]
    #[must_use]
    pub fn transmission_gain(&self) -> Sample {
        transmission_gain(self.transmission_loss_db)
    }

    /// The linear gain applied to a specular reflection off the surface, in
    /// `[0, 1]` (the clamped reflection coefficient).
    #[inline]
    #[must_use]
    pub fn reflection_gain(&self) -> Sample {
        self.reflection.clamp(0.0, 1.0)
    }
}

impl Default for AcousticMaterial {
    #[inline]
    fn default() -> Self {
        Self::OPEN
    }
}

/// How a [`PropagationPath`] reached the listener.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum PathKind {
    /// Straight line of sight from source to listener.
    Direct,
    /// Passed through one or more partitions (attenuated by transmission loss).
    Transmission,
    /// Bent around an edge into the geometric shadow (Maekawa-attenuated).
    Diffraction,
    /// Bounced off a surface (delayed and scaled by the reflection coefficient).
    Reflection,
}

/// One arrival of a source at the listener, with the delay, gain, colour, and
/// direction the real-time voice needs to render it.
///
/// A backend fills a slice of these; the voice turns each into a delay line +
/// gain + low-pass. `direction` is a unit vector in the **listener-local**
/// frame (`-Z` forward, `+X` right, `+Y` up), matching
/// [`LocalSource`](crate::geometry::LocalSource), so it can drive a panner
/// directly.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PropagationPath {
    /// Which mechanism produced this arrival.
    pub kind: PathKind,
    /// Propagation delay from source to listener along this path, in seconds
    /// (always non-negative).
    pub delay_seconds: Sample,
    /// Linear broadband gain of this arrival, in `[0, 1]`.
    pub gain: Sample,
    /// Low-pass corner (Hz) colouring this arrival. [`FULL_BAND_CUTOFF_HZ`]
    /// means "unfiltered"; consumers clamp to Nyquist.
    pub cutoff_hz: Sample,
    /// Listener-local unit direction the arrival comes from.
    pub direction: Vec3,
}

impl PropagationPath {
    /// A silent placeholder used to pre-fill a backend's output buffer.
    pub const SILENT: Self = Self {
        kind: PathKind::Direct,
        delay_seconds: 0.0,
        gain: 0.0,
        cutoff_hz: FULL_BAND_CUTOFF_HZ,
        direction: Vec3::NEG_Z,
    };

    /// The propagation delay expressed in samples at `sample_rate`.
    #[inline]
    #[must_use]
    pub fn delay_samples(&self, sample_rate: u32) -> Sample {
        self.delay_seconds.max(0.0) * (sample_rate as Sample)
    }
}

impl Default for PropagationPath {
    #[inline]
    fn default() -> Self {
        Self::SILENT
    }
}

/// A summary of a propagation query alongside the per-path detail.
///
/// `direct` reports how blocked the *direct* line of sight is, in the same
/// [`OcclusionFactors`] vocabulary the [`occlusion`](crate::occlusion) model
/// consumes, so a backend can feed both systems from one trace. `path_count`
/// is how many entries of the caller's buffer were populated.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PropagationSummary {
    /// Blocking of the direct path (feeds [`crate::occlusion`]).
    pub direct: OcclusionFactors,
    /// Number of [`PropagationPath`]s written into the output buffer.
    pub path_count: usize,
}

/// A pluggable geometry backend that resolves the full set of arrivals for a
/// source, including secondary (transmitted / diffracted / reflected) paths.
///
/// This is the richer sibling of
/// [`OcclusionQuery`](crate::occlusion::OcclusionQuery): where that reports a
/// single pair of blocking factors, a `PropagationBackend` enumerates the
/// actual paths a wave takes through the scene. Implementors (in a physics-
/// aware layer) trace edges, portals, and reflectors; keeping this a trait
/// lets the spatial crate stay free of any physics dependency.
///
/// The backend runs at **control rate** and is *not* required to be real-time
/// safe — it may allocate scratch and trace geometry off the audio thread. Its
/// bounded output then drives the real-time voice.
pub trait PropagationBackend {
    /// Resolves the arrivals of `emitter` at `listener`, writing up to
    /// `paths.len()` entries into `paths` and returning a summary. Only the
    /// first `PropagationSummary::path_count` entries are meaningful.
    fn query(
        &self,
        listener: &Listener,
        emitter: &Emitter,
        paths: &mut [PropagationPath],
    ) -> PropagationSummary;
}

/// A [`PropagationBackend`] modelling an open free field: one direct,
/// unoccluded, full-band path and no secondary arrivals.
///
/// This is the correct default for headless rendering, tests, and scenes with
/// no acoustically significant geometry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FreeFieldBackend;

impl PropagationBackend for FreeFieldBackend {
    fn query(
        &self,
        listener: &Listener,
        emitter: &Emitter,
        paths: &mut [PropagationPath],
    ) -> PropagationSummary {
        if paths.is_empty() {
            return PropagationSummary {
                direct: OcclusionFactors::OPEN,
                path_count: 0,
            };
        }

        let local = listener.localize(emitter);
        paths[0] = PropagationPath {
            kind: PathKind::Direct,
            delay_seconds: local.distance / SPEED_OF_SOUND_MPS,
            gain: 1.0,
            cutoff_hz: FULL_BAND_CUTOFF_HZ,
            direction: local.direction,
        };

        PropagationSummary {
            direct: OcclusionFactors::OPEN,
            path_count: 1,
        }
    }
}

/// The **Fresnel number** `N = 2 * delta / lambda = 2 * delta * f / c` for a
/// diffracting edge, where `delta` is the path-length difference (the extra
/// distance the wave travels going *over the edge* versus the straight line)
/// and `lambda = c / f` is the wavelength.
///
/// `N` is positive in the geometric shadow (the detour is real), zero on the
/// shadow boundary, and negative in the illuminated zone. It grows with both
/// the detour and the frequency, which is why high frequencies are shadowed
/// more strongly.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::propagation::fresnel_number;
/// // Doubling the frequency doubles the Fresnel number.
/// let n1 = fresnel_number(0.5, 1_000.0);
/// let n2 = fresnel_number(0.5, 2_000.0);
/// assert!((n2 - 2.0 * n1).abs() < 1e-4);
/// ```
#[inline]
#[must_use]
pub fn fresnel_number(path_difference_m: Sample, freq_hz: Sample) -> Sample {
    2.0 * path_difference_m * freq_hz / SPEED_OF_SOUND_MPS
}

/// Hyperbolic tangent built from [`bevy_math::ops::exp`], since
/// `bevy_math::ops` exposes no `tanh`. `tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)`.
#[inline]
#[must_use]
fn tanh(x: Sample) -> Sample {
    let e2x = ops::exp(2.0 * x);
    (e2x - 1.0) / (e2x + 1.0)
}

/// Base-10 logarithm built from [`bevy_math::ops::ln`], since `bevy_math::ops`
/// exposes no `log10`.
#[inline]
#[must_use]
fn log10(x: Sample) -> Sample {
    ops::ln(x) / LN_10
}

/// The **Maekawa (1968)** barrier-diffraction attenuation, in decibels, for a
/// given [Fresnel number](fresnel_number).
///
/// The classic chart is captured piecewise:
///
/// * `N < -0.2` — fully in the illuminated zone: no attenuation (`0 dB`).
/// * `-0.2 <= N < 0` — the transition just outside the shadow: a linear ramp
///   from `0` to `5 dB`.
/// * `N >= 0` — the shadow zone:
///   `5 + 20 * log10(x / tanh(x))` with `x = sqrt(2 * pi * N)`, clamped to
///   [`MAX_DIFFRACTION_DB`]. As `x -> 0` the ratio `x / tanh(x) -> 1`, so the
///   shadow boundary (`N = 0`) gives exactly `5 dB`.
///
/// The result rises monotonically with `N`.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::propagation::maekawa_attenuation_db;
/// // On the shadow boundary the barrier already costs ~5 dB.
/// assert!((maekawa_attenuation_db(0.0) - 5.0).abs() < 1e-3);
/// // Deep in the shadow it is much stronger.
/// assert!(maekawa_attenuation_db(10.0) > 20.0);
/// // In the lit zone there is no attenuation.
/// assert_eq!(maekawa_attenuation_db(-1.0), 0.0);
/// ```
#[must_use]
pub fn maekawa_attenuation_db(fresnel_number: Sample) -> Sample {
    const RAMP_START: Sample = -0.2;
    let n = fresnel_number;

    if n < RAMP_START {
        0.0
    } else if n < 0.0 {
        // Linear ramp 0 -> 5 dB across [-0.2, 0).
        5.0 * (n - RAMP_START) / (0.0 - RAMP_START)
    } else {
        let x = ops::sqrt(2.0 * PI * n);
        // x / tanh(x) -> 1 as x -> 0 (avoid the 0/0 at the boundary).
        let ratio = if x < 1.0e-4 { 1.0 } else { x / tanh(x) };
        let db = 5.0 + 20.0 * log10(ratio);
        db.min(MAX_DIFFRACTION_DB)
    }
}

/// The broadband linear gain of a diffracted arrival at `freq_hz` for an edge
/// with path-length difference `path_difference_m`, in `(0, 1]`.
///
/// This is simply `db_to_linear(-maekawa_attenuation_db(fresnel_number(..)))`:
/// deeper shadows and higher frequencies attenuate more.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::propagation::diffraction_gain;
/// // A high frequency is shadowed more than a low one for the same edge.
/// let low = diffraction_gain(0.5, 200.0);
/// let high = diffraction_gain(0.5, 8_000.0);
/// assert!(high < low);
/// assert!(high > 0.0 && low <= 1.0);
/// ```
#[inline]
#[must_use]
pub fn diffraction_gain(path_difference_m: Sample, freq_hz: Sample) -> Sample {
    db_to_linear(-maekawa_attenuation_db(fresnel_number(path_difference_m, freq_hz)))
}

/// The low-pass corner (Hz) equivalent to diffraction around an edge with
/// path-length difference `path_difference_m`, rendered at `sample_rate`.
///
/// The search walks [`FREQUENCY_GRID`] from low to high and returns the first
/// frequency whose Maekawa attenuation reaches
/// [`DIFFRACTION_CUTOFF_THRESHOLD_DB`]. Because the attenuation rises with both
/// frequency and the detour, the corner is **monotonically non-increasing** in
/// `path_difference_m`: deeper shadows are duller. A source at or inside the
/// shadow boundary that never reaches the threshold (a shallow detour, or the
/// lit zone) keeps the full band. The result is clamped to
/// `[MIN_CUTOFF_HZ, sample_rate * 0.499]`.
#[must_use]
pub fn diffraction_cutoff_hz(path_difference_m: Sample, sample_rate: u32) -> Sample {
    let max_cutoff = ((sample_rate as Sample) * 0.499).max(MIN_CUTOFF_HZ);

    // Default to the grid's upper bound: nothing has rolled off yet.
    let mut cutoff = FREQUENCY_GRID[FREQUENCY_GRID.len() - 1];
    for &f in &FREQUENCY_GRID {
        let att = maekawa_attenuation_db(fresnel_number(path_difference_m, f));
        if att >= DIFFRACTION_CUTOFF_THRESHOLD_DB {
            cutoff = f;
            break;
        }
    }

    cutoff.clamp(MIN_CUTOFF_HZ, max_cutoff)
}

/// The linear gain of sound transmitted through a partition with the given
/// transmission `loss_db`, in `(0, 1]`.
///
/// `transmission_gain(loss) = db_to_linear(-max(loss, 0))`: `0 dB` passes the
/// signal untouched and larger losses attenuate it. Negative losses are treated
/// as `0` so a material can never amplify.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::propagation::transmission_gain;
/// assert!((transmission_gain(0.0) - 1.0).abs() < 1e-6);
/// // ~6 dB of loss halves the amplitude.
/// assert!((transmission_gain(6.0206) - 0.5).abs() < 1e-3);
/// ```
#[inline]
#[must_use]
pub fn transmission_gain(loss_db: Sample) -> Sample {
    db_to_linear(-loss_db.max(0.0))
}

/// The diffraction path-length difference (metres) for a wave that must bend
/// over `edge` to get from `source` to `listener`.
///
/// `delta = |listener - edge| + |edge - source| - |source - listener|`, i.e.
/// how much longer the over-the-edge detour is than the straight line. It feeds
/// [`fresnel_number`]. The result is floored at `0` (the detour can never be
/// shorter than the straight line, up to rounding).
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_spatial::propagation::edge_path_difference;
/// // An edge directly on the line of sight adds no detour.
/// let d = edge_path_difference(
///     Vec3::new(-1.0, 0.0, 0.0),
///     Vec3::ZERO,
///     Vec3::new(1.0, 0.0, 0.0),
/// );
/// assert!(d.abs() < 1e-5);
/// ```
#[inline]
#[must_use]
pub fn edge_path_difference(listener: Vec3, edge: Vec3, source: Vec3) -> Sample {
    let over = distance(listener, edge) + distance(edge, source);
    let direct = distance(source, listener);
    (over - direct).max(0.0)
}

/// Deterministic Euclidean distance between two points (routes the length
/// through [`bevy_math::ops::sqrt`] rather than an `f32` intrinsic).
#[inline]
#[must_use]
fn distance(a: Vec3, b: Vec3) -> Sample {
    let d = a - b;
    ops::sqrt(d.dot(d))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn fresnel_number_matches_formula() {
        // N = 2 * delta * f / c.
        let delta = 0.75;
        let f = 1_500.0;
        let expected = 2.0 * delta * f / SPEED_OF_SOUND_MPS;
        assert!(approx(fresnel_number(delta, f), expected, 1e-4));
        // Odd in the sign of the path difference.
        assert!(approx(fresnel_number(-delta, f), -expected, 1e-4));
    }

    #[test]
    fn maekawa_known_points() {
        // Shadow boundary: exactly 5 dB.
        assert!(approx(maekawa_attenuation_db(0.0), 5.0, 1e-3));
        // N = 1 -> ~13.1 dB.
        assert!(approx(maekawa_attenuation_db(1.0), 13.06, 0.2));
        // N = 10 -> ~23 dB (below the 24 dB cap).
        assert!(approx(maekawa_attenuation_db(10.0), 23.0, 0.5));
        // Transition ramp.
        assert!(approx(maekawa_attenuation_db(-0.1), 2.5, 1e-3));
        assert!(approx(maekawa_attenuation_db(-0.2), 0.0, 1e-6));
        // Lit zone: nothing.
        assert_eq!(maekawa_attenuation_db(-0.5), 0.0);
        assert_eq!(maekawa_attenuation_db(-100.0), 0.0);
    }

    #[test]
    fn maekawa_is_monotonic_non_decreasing() {
        let mut previous = -1.0;
        let mut n = -0.5;
        while n <= 30.0 {
            let db = maekawa_attenuation_db(n);
            assert!(
                db + 1e-4 >= previous,
                "attenuation must not fall: {db} < {previous} at N={n}",
            );
            assert!(db.is_finite() && (0.0..=MAX_DIFFRACTION_DB).contains(&db));
            previous = db;
            n += 0.1;
        }
    }

    #[test]
    fn maekawa_is_capped() {
        // A huge Fresnel number saturates at the ceiling.
        assert!(approx(maekawa_attenuation_db(1.0e6), MAX_DIFFRACTION_DB, 1e-3));
    }

    #[test]
    fn diffraction_gain_falls_with_frequency_and_shadow_depth() {
        // Higher frequency -> more shadowing -> lower gain.
        let low = diffraction_gain(0.5, 250.0);
        let high = diffraction_gain(0.5, 6_000.0);
        assert!(high < low);
        assert!((0.0..=1.0).contains(&high) && (0.0..=1.0).contains(&low));

        // Deeper detour -> lower gain at a fixed frequency.
        let shallow = diffraction_gain(0.1, 2_000.0);
        let deep = diffraction_gain(2.0, 2_000.0);
        assert!(deep < shallow);

        // Well inside the lit zone -> unity gain.
        assert!(approx(diffraction_gain(-1.0, 4_000.0), 1.0, 1e-6));
    }

    #[test]
    fn diffraction_cutoff_is_non_increasing_with_shadow_depth() {
        let sr = 48_000;
        let deltas = [-1.0, 0.0, 0.05, 0.2, 1.0, 5.0, 50.0];
        let mut previous = Sample::INFINITY;
        for d in deltas {
            let cutoff = diffraction_cutoff_hz(d, sr);
            assert!(
                cutoff <= previous + 1e-3,
                "cutoff must not rise with detour: {cutoff} > {previous} at {d} m",
            );
            assert!((MIN_CUTOFF_HZ..=(sr as Sample) * 0.499).contains(&cutoff));
            previous = cutoff;
        }
    }

    #[test]
    fn lit_zone_keeps_full_band() {
        // No shadow: the corner sits at the top of the band.
        let sr = 48_000;
        let cutoff = diffraction_cutoff_hz(-1.0, sr);
        assert!(cutoff >= (sr as Sample) * 0.499 - 1.0);
    }

    #[test]
    fn transmission_gain_behaviour() {
        assert!(approx(transmission_gain(0.0), 1.0, 1e-6));
        assert!(approx(transmission_gain(6.0206), 0.5, 1e-3));
        assert!(transmission_gain(40.0) < transmission_gain(20.0));
        // Negative loss cannot amplify.
        assert!(approx(transmission_gain(-10.0), 1.0, 1e-6));
    }

    #[test]
    fn acoustic_material_gains() {
        let m = AcousticMaterial::new(12.0, 1.5);
        assert!(m.transmission_gain() < 1.0);
        // Reflection is clamped into [0, 1].
        assert!(approx(m.reflection_gain(), 1.0, 1e-6));
        assert!(approx(AcousticMaterial::OPEN.transmission_gain(), 1.0, 1e-6));
        assert!(approx(AcousticMaterial::OPEN.reflection_gain(), 0.0, 1e-6));
    }

    #[test]
    fn edge_path_difference_geometry() {
        // Edge on the line of sight -> no detour.
        let d = edge_path_difference(
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::new(2.0, 0.0, 0.0),
        );
        assert!(approx(d, 0.0, 1e-5));

        // Edge lifted off the line -> a real detour, and never negative.
        let d2 = edge_path_difference(
            Vec3::new(-2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        );
        // |L->E| = sqrt(4+9) = 3.6056 twice = 7.2111; direct = 4 -> delta 3.2111.
        assert!(approx(d2, 3.2111, 1e-3));
        assert!(d2 >= 0.0);
    }

    #[test]
    fn free_field_backend_direct_only() {
        let listener = Listener::default();
        let emitter = Emitter::point(Vec3::new(0.0, 0.0, -343.0), Vec3::ZERO);
        let mut paths = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        let summary = FreeFieldBackend.query(&listener, &emitter, &mut paths);

        assert_eq!(summary.path_count, 1);
        assert_eq!(summary.direct, OcclusionFactors::OPEN);

        let direct = paths[0];
        assert_eq!(direct.kind, PathKind::Direct);
        assert!(approx(direct.gain, 1.0, 1e-6));
        // 343 m at 343 m/s == 1 second of delay.
        assert!(approx(direct.delay_seconds, 1.0, 1e-4));
        assert_eq!(direct.cutoff_hz, FULL_BAND_CUTOFF_HZ);
        // Source dead ahead -> local forward (-Z).
        assert!(approx(direct.direction.z, -1.0, 1e-5));
    }

    #[test]
    fn free_field_backend_handles_empty_buffer() {
        let mut paths: [PropagationPath; 0] = [];
        let summary =
            FreeFieldBackend.query(&Listener::default(), &Emitter::default(), &mut paths);
        assert_eq!(summary.path_count, 0);
    }

    #[test]
    fn delay_samples_scales_with_rate() {
        let path = PropagationPath {
            delay_seconds: 0.5,
            ..PropagationPath::SILENT
        };
        assert!(approx(path.delay_samples(48_000), 24_000.0, 1e-3));
        // Negative delay is floored at zero.
        let bad = PropagationPath {
            delay_seconds: -1.0,
            ..PropagationPath::SILENT
        };
        assert!(approx(bad.delay_samples(48_000), 0.0, 1e-6));
    }

    #[test]
    fn helpers_are_deterministic() {
        // Repeated evaluation is bit-identical (libm-backed math).
        assert_eq!(maekawa_attenuation_db(2.5), maekawa_attenuation_db(2.5));
        assert_eq!(
            diffraction_gain(0.7, 3_000.0),
            diffraction_gain(0.7, 3_000.0),
        );
        assert_eq!(
            diffraction_cutoff_hz(0.7, 44_100),
            diffraction_cutoff_hz(0.7, 44_100),
        );
    }
}
