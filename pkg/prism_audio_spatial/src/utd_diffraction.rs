//! Wedge diffraction via the Uniform Theory of Diffraction (UTD).
//!
//! Where [`crate::diffraction`] gives the empirical Maekawa barrier insertion
//! loss, this module implements the physically rigorous **Uniform Theory of
//! Diffraction** (Kouyoumjian and Pathak, 1974) for a straight wedge. UTD
//! yields a frequency-dependent *complex* diffraction coefficient `D(f)` that
//! stays finite and continuous across the shadow and reflection boundaries,
//! which is why it is the model of choice for interactive acoustic edge
//! diffraction. The coefficient is the sum of four cotangent terms, each
//! regularised by the transition function `F(X)` from
//! [`crate::fresnel_transition`].
//!
//! # Geometry
//!
//! A diffracting edge is a straight line with unit direction `e`. The source
//! sits a distance `s'` from the diffraction point along a ray that makes an
//! angle `beta0` with the edge (the *incidence cone* half-angle); the receiver
//! sits a distance `s` from the diffraction point on the same cone. Measured in
//! the plane perpendicular to the edge, the source lies at azimuth `phi'` and
//! the receiver at azimuth `phi`, both relative to the illuminated reference
//! face (face 0). The wedge is described by its *wedge index*
//!
//! `n = 2 - WA / pi`,
//!
//! where `WA` is the interior wedge angle: a thin half-plane screen has
//! `WA = 0` so `n = 2`, and a right-angle exterior corner has `WA = pi/2` so
//! `n = 1.5`. The exterior angular region spans `[0, n*pi]`.
//!
//! # Coefficient
//!
//! For a sound-hard (acoustically rigid, Neumann) wedge the reflection
//! coefficients on both faces are `+1`, so all four terms add:
//!
//! `D = -exp(-j pi/4) / (2 n sqrt(2 pi k) sin(beta0)) * (T1 + T2 + T3 + T4)`,
//!
//! `k = 2 pi f / c`,  `L = s s' sin^2(beta0) / (s + s')`,
//!
//! with `T` terms built from `cot((pi +/- beta)/(2n)) * F(k L a^+/-(beta))` over
//! the angle differences `beta = phi - phi'` and `beta = phi + phi'`. Near a
//! boundary the cotangent diverges while `F -> 0`; their product has a finite
//! limit which this module evaluates in closed form so the coefficient stays
//! bounded everywhere.
//!
//! # Relative gain and bands
//!
//! [`UtdWedge::relative_gain`] normalises `|D(f)|` against the free-field direct
//! ray over the same total distance `s + s'`:
//!
//! `G(f) = |D(f)| * sqrt((s + s') / (s' s))`,
//!
//! which, because `|D| ~ 1/sqrt(k)`, falls with frequency inside the shadow:
//! high frequencies are shadowed more strongly than low ones.
//! [`UtdWedge::band_gains`] samples `G` at the three propagation band centres
//! and clamps to `[0, 1]`, producing a [`BandGains`] that plugs straight into
//! the banded propagation pipeline.
//!
//! # Determinism and real-time safety
//!
//! Every query runs on stack scalars and fixed-size arrays with no allocation,
//! no locking and no panicking, and all transcendental math routes through
//! [`bevy_math::ops`]. Degenerate geometry (grazing incidence, coincident
//! points, non-finite inputs) is clamped to safe finite values.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or
//! derived code**. It is implemented from the publicly documented UTD wedge
//! formulation (Kouyoumjian and Pathak 1974; the textbook presentation in
//! `McNamara`, Pistorius and Malherbe 1990 and Balanis).

use bevy_math::ops;
use bevy_math::Vec3;

use core::f32::consts::{FRAC_1_SQRT_2, PI};

use prism_audio_core::math::Sample;

use crate::band_spectrum::{BandGains, PROPAGATION_BAND_CENTERS, PROPAGATION_BAND_COUNT};
use crate::early_reflections::DEFAULT_SOUND_SPEED;
use crate::fresnel_transition::{transition, TransitionValue};

/// Two pi, the full-turn angle used throughout the wedge formulation.
const TWO_PI: Sample = 2.0 * PI;

/// Smallest positive distance / sine used to keep ratios finite.
const MIN_POSITIVE: Sample = 1e-6;

/// Angular half-width (in radians of the cotangent numerator) within which the
/// `cot * F` product is evaluated by its closed-form boundary limit instead of
/// the direct quotient. Chosen small enough that the small-argument expansion
/// of the transition function is accurate, yet large enough to avoid the
/// cotangent overflowing in single precision.
const BOUNDARY_EPS: Sample = 1e-2;

/// A minimal complex number used internally to accumulate the four UTD terms.
#[derive(Clone, Copy, Debug)]
struct Cplx {
    re: Sample,
    im: Sample,
}

impl Cplx {
    /// Complex addition.
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Multiplication by a real scalar.
    fn scale(self, factor: Sample) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }

    /// Complex multiplication.
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// A straight-wedge diffraction model parameterised by its geometry.
///
/// Build one with [`UtdWedge::new`] from explicit angles and distances, or with
/// [`UtdWedge::from_geometry`] from world-space positions, then query the
/// complex coefficient, the relative gain at a frequency, or the per-band gains.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct UtdWedge {
    /// Wedge index `n = 2 - WA/pi`, clamped to `[1, 2]`.
    n: Sample,
    /// Incidence cone half-angle `beta0` (radians), clamped away from `0`/`pi`.
    beta0: Sample,
    /// Source azimuth `phi'` (radians) relative to the reference face, in `[0, 2pi)`.
    phi_i: Sample,
    /// Receiver azimuth `phi` (radians) relative to the reference face, in `[0, 2pi)`.
    phi_d: Sample,
    /// Source-to-edge distance `s'` (metres), clamped positive.
    s_src: Sample,
    /// Edge-to-receiver distance `s` (metres), clamped positive.
    s_rcv: Sample,
    /// Speed of sound (metres per second), positive.
    sound_speed: Sample,
}

impl UtdWedge {
    /// Builds a wedge from explicit angles and distances.
    ///
    /// `n` is clamped to `[1, 2]`, `beta0` to the open interval `(0, pi)`,
    /// azimuths are wrapped to `[0, 2pi)`, distances are clamped to a small
    /// positive floor, and a non-finite or non-positive `sound_speed` falls
    /// back to [`DEFAULT_SOUND_SPEED`].
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::utd_diffraction::UtdWedge;
    /// use core::f32::consts::PI;
    ///
    /// // A thin screen (n = 2) with the receiver deep in the shadow.
    /// let wedge = UtdWedge::new(2.0, PI / 2.0, PI / 4.0, 7.0 * PI / 4.0, 3.0, 3.0, 343.0);
    /// let gain = wedge.relative_gain(1_000.0);
    /// assert!(gain.is_finite() && gain >= 0.0);
    /// ```
    #[must_use]
    pub fn new(
        n: Sample,
        beta0: Sample,
        phi_i: Sample,
        phi_d: Sample,
        s_src: Sample,
        s_rcv: Sample,
        sound_speed: Sample,
    ) -> Self {
        let n = clamp_finite(n, 1.0, 2.0, 2.0);
        let beta0 = clamp_finite(beta0, MIN_POSITIVE, PI - MIN_POSITIVE, PI / 2.0);
        let phi_i = wrap_two_pi(phi_i);
        let phi_d = wrap_two_pi(phi_d);
        let s_src = floor_positive(s_src);
        let s_rcv = floor_positive(s_rcv);
        let sound_speed = if sound_speed.is_finite() && sound_speed > 0.0 {
            sound_speed
        } else {
            DEFAULT_SOUND_SPEED
        };
        Self {
            n,
            beta0,
            phi_i,
            phi_d,
            s_src,
            s_rcv,
            sound_speed,
        }
    }

    /// Builds a wedge from world-space geometry.
    ///
    /// `edge_point` is any point on the diffracting edge, `edge_dir` the edge
    /// direction, and `face_ref` a direction lying on the illuminated reference
    /// face, perpendicular to the edge and pointing into the exterior region.
    /// `wedge_index` is `n = 2 - WA/pi`. Azimuths are measured in the plane
    /// perpendicular to the edge using `face_ref` and `edge_dir x face_ref` as
    /// the in-plane axes.
    ///
    /// Degenerate directions (zero-length edge or reference face) fall back to
    /// safe defaults so the result stays finite.
    #[must_use]
    pub fn from_geometry(
        source: Vec3,
        edge_point: Vec3,
        edge_dir: Vec3,
        receiver: Vec3,
        face_ref: Vec3,
        wedge_index: Sample,
        sound_speed: Sample,
    ) -> Self {
        let edge = edge_dir.normalize_or_zero();
        let edge = if edge == Vec3::ZERO { Vec3::X } else { edge };

        // Reference face axis, re-orthogonalised against the edge.
        let face_raw = face_ref - edge * face_ref.dot(edge);
        let face0 = face_raw.normalize_or_zero();
        let face0 = if face0 == Vec3::ZERO {
            any_perpendicular(edge)
        } else {
            face0
        };
        let face_tangent = edge.cross(face0);

        let to_src = source - edge_point;
        let to_rcv = receiver - edge_point;

        let s_src = floor_positive(to_src.length());
        let s_rcv = floor_positive(to_rcv.length());

        // Perpendicular (in-plane) components relative to the edge.
        let src_perp = to_src - edge * to_src.dot(edge);
        let rcv_perp = to_rcv - edge * to_rcv.dot(edge);

        // Incidence cone half-angle from the source leg.
        let beta0 = ops::atan2(src_perp.length(), to_src.dot(edge).abs().max(MIN_POSITIVE))
            .clamp(MIN_POSITIVE, PI - MIN_POSITIVE);

        let phi_i = azimuth(src_perp, face0, face_tangent);
        let phi_d = azimuth(rcv_perp, face0, face_tangent);

        Self::new(wedge_index, beta0, phi_i, phi_d, s_src, s_rcv, sound_speed)
    }

    /// The wavenumber `k = 2 pi f / c` at `freq_hz`.
    fn wavenumber(&self, freq_hz: Sample) -> Sample {
        TWO_PI * freq_hz / self.sound_speed
    }

    /// The spherical-wave distance parameter `L = s s' sin^2(beta0) / (s + s')`.
    fn distance_parameter(&self) -> Sample {
        let sin_b = ops::sin(self.beta0);
        self.s_src * self.s_rcv * sin_b * sin_b / (self.s_src + self.s_rcv)
    }

    /// The complex UTD diffraction coefficient `D(f)` for a sound-hard wedge.
    ///
    /// Returns [`TransitionValue::ZERO`] for non-finite or non-positive
    /// frequencies.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::utd_diffraction::UtdWedge;
    /// use core::f32::consts::PI;
    ///
    /// let wedge = UtdWedge::new(2.0, PI / 2.0, PI / 4.0, 7.0 * PI / 4.0, 2.0, 2.0, 343.0);
    /// let d = wedge.coefficient(2_000.0);
    /// assert!(d.re.is_finite() && d.im.is_finite());
    /// ```
    #[must_use]
    pub fn coefficient(&self, freq_hz: Sample) -> TransitionValue {
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            return TransitionValue::ZERO;
        }

        let k = self.wavenumber(freq_hz);
        let big_l = self.distance_parameter();
        let n = self.n;
        let sin_b = ops::sin(self.beta0).max(MIN_POSITIVE);

        let beta_minus = self.phi_d - self.phi_i;
        let beta_plus = self.phi_d + self.phi_i;

        let t1 = cot_transition(beta_minus, n, k, big_l, true);
        let t2 = cot_transition(beta_minus, n, k, big_l, false);
        let t3 = cot_transition(beta_plus, n, k, big_l, true);
        let t4 = cot_transition(beta_plus, n, k, big_l, false);

        let sum = t1.add(t2).add(t3).add(t4);

        // Prefactor -exp(-j pi/4) = (-sqrt2/2, +sqrt2/2).
        let prefactor = Cplx {
            re: -FRAC_1_SQRT_2,
            im: FRAC_1_SQRT_2,
        };
        let scalar = 1.0 / (2.0 * n * ops::sqrt(TWO_PI * k) * sin_b);

        let d = prefactor.scale(scalar).mul(sum);
        TransitionValue { re: d.re, im: d.im }
    }

    /// The diffracted gain at `freq_hz`, normalised against the free-field
    /// direct ray over the total distance `s + s'` (unclamped, non-negative).
    ///
    /// Because `|D|` scales as `1/sqrt(k)`, this gain decreases with frequency
    /// inside the geometric shadow.
    #[must_use]
    pub fn relative_gain(&self, freq_hz: Sample) -> Sample {
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            return 0.0;
        }
        let d = self.coefficient(freq_hz);
        let magnitude = ops::hypot(d.re, d.im);
        let total = self.s_src + self.s_rcv;
        let spreading = ops::sqrt(total / (self.s_src * self.s_rcv));
        magnitude * spreading
    }

    /// Samples [`UtdWedge::relative_gain`] at the three propagation band centres
    /// and clamps each to `[0, 1]`, producing a [`BandGains`] ready for the
    /// banded propagation pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::utd_diffraction::UtdWedge;
    /// use core::f32::consts::PI;
    ///
    /// let wedge = UtdWedge::new(2.0, PI / 2.0, PI / 4.0, 7.0 * PI / 4.0, 3.0, 3.0, 343.0);
    /// let bands = wedge.band_gains();
    /// // Deep shadow shadows the highs more than the lows.
    /// assert!(bands.high() <= bands.low() + 1e-3);
    /// ```
    #[must_use]
    pub fn band_gains(&self) -> BandGains {
        let mut bands = [0.0; PROPAGATION_BAND_COUNT];
        for (band, &centre) in bands.iter_mut().zip(PROPAGATION_BAND_CENTERS.iter()) {
            *band = self.relative_gain(centre).clamp(0.0, 1.0);
        }
        BandGains::new(bands)
    }
}

/// Evaluates one UTD cotangent term `cot((pi +/- beta)/(2n)) * F(k L a(beta))`.
///
/// `is_plus` selects the `+beta` branch (using `N^+` and `a^+`); otherwise the
/// `-beta` branch (`N^-`, `a^-`). Near a boundary, where the cotangent diverges
/// and `F -> 0`, the product is replaced by its closed-form finite limit.
fn cot_transition(beta: Sample, n: Sample, k: Sample, big_l: Sample, is_plus: bool) -> Cplx {
    let numerator = if is_plus { PI + beta } else { PI - beta };
    let big_n = ops::round(numerator / (TWO_PI * n));
    let eps = numerator - TWO_PI * n * big_n;

    // Angle argument of the transition function: a = 2 cos^2((2 pi n N - beta)/2),
    // where N^+ = big_n for the + branch and N^- = -big_n for the - branch.
    let big_n_angle = if is_plus { big_n } else { -big_n };
    let half = (TWO_PI * n * big_n_angle - beta) * 0.5;
    let cos_half = ops::cos(half);
    let a = 2.0 * cos_half * cos_half;
    let x = k * big_l * a;

    if ops::abs(eps) < BOUNDARY_EPS {
        // Boundary limit: cot * F -> exp(j pi/4) * n *
        //   ( sqrt(2 pi k L) * sign(eps) - 2 k L eps * exp(j pi/4) ).
        let sign = if eps >= 0.0 { 1.0 } else { -1.0 };
        let kl = k * big_l;
        let lead = Cplx {
            re: ops::sqrt(TWO_PI * kl) * sign,
            im: 0.0,
        };
        let correction = Cplx {
            re: -2.0 * kl * eps,
            im: 0.0,
        }
        .mul(exp_j_pi_4());
        lead.add(correction).mul(exp_j_pi_4()).scale(n)
    } else {
        let cot = cotangent(numerator / (2.0 * n));
        let f = transition(x);
        Cplx { re: f.re, im: f.im }.scale(cot)
    }
}

/// `exp(j pi/4) = (sqrt2/2, sqrt2/2)`.
fn exp_j_pi_4() -> Cplx {
    Cplx {
        re: FRAC_1_SQRT_2,
        im: FRAC_1_SQRT_2,
    }
}

/// Cotangent `cos/sin`, guarded against a vanishing sine.
fn cotangent(angle: Sample) -> Sample {
    let s = ops::sin(angle);
    let s = if ops::abs(s) < MIN_POSITIVE {
        if s >= 0.0 {
            MIN_POSITIVE
        } else {
            -MIN_POSITIVE
        }
    } else {
        s
    };
    ops::cos(angle) / s
}

/// Wraps an angle into `[0, 2 pi)`.
fn wrap_two_pi(angle: Sample) -> Sample {
    if !angle.is_finite() {
        return 0.0;
    }
    let wrapped = angle - TWO_PI * ops::floor(angle / TWO_PI);
    // Guard against the rare case where rounding pushes the result to 2 pi.
    if wrapped >= TWO_PI {
        wrapped - TWO_PI
    } else if wrapped < 0.0 {
        wrapped + TWO_PI
    } else {
        wrapped
    }
}

/// Clamps a value to `[lo, hi]`, substituting `fallback` when non-finite.
fn clamp_finite(value: Sample, lo: Sample, hi: Sample, fallback: Sample) -> Sample {
    if value.is_finite() {
        value.clamp(lo, hi)
    } else {
        fallback
    }
}

/// Clamps a distance to a small positive floor, substituting the floor when
/// non-finite.
fn floor_positive(value: Sample) -> Sample {
    if value.is_finite() && value > MIN_POSITIVE {
        value
    } else {
        MIN_POSITIVE
    }
}

/// Azimuth of an in-plane vector in `[0, 2 pi)` from its reference-face and
/// tangent components.
fn azimuth(perp: Vec3, face0: Vec3, face_tangent: Vec3) -> Sample {
    let x = perp.dot(face0);
    let y = perp.dot(face_tangent);
    if x == 0.0 && y == 0.0 {
        0.0
    } else {
        wrap_two_pi(ops::atan2(y, x))
    }
}

/// Returns an arbitrary unit vector perpendicular to `axis`.
fn any_perpendicular(axis: Vec3) -> Vec3 {
    let candidate = if ops::abs(axis.x) < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let perp = (candidate - axis * candidate.dot(axis)).normalize_or_zero();
    if perp == Vec3::ZERO {
        Vec3::Y
    } else {
        perp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fresnel_transition::transition as _transition_check;

    const C: Sample = 343.0;

    fn thin_screen(phi_d: Sample) -> UtdWedge {
        // Thin half-plane: n = 2, source illuminating from phi' = pi/4.
        UtdWedge::new(2.0, PI / 2.0, PI / 4.0, phi_d, 4.0, 4.0, C)
    }

    #[test]
    fn coefficient_is_finite_everywhere() {
        let mut phi = 0.01;
        while phi < TWO_PI {
            let wedge = thin_screen(phi);
            for &f in &[50.0, 500.0, 5_000.0, 15_000.0] {
                let d = wedge.coefficient(f);
                assert!(d.re.is_finite() && d.im.is_finite(), "phi={phi} f={f}");
            }
            phi += 0.05;
        }
    }

    #[test]
    fn non_positive_frequency_is_zero() {
        let wedge = thin_screen(PI);
        assert_eq!(wedge.coefficient(0.0), TransitionValue::ZERO);
        assert_eq!(wedge.coefficient(-10.0), TransitionValue::ZERO);
        assert_eq!(wedge.relative_gain(0.0), 0.0);
    }

    #[test]
    fn deep_shadow_attenuates_highs_more() {
        // Receiver well into the shadow region behind the screen.
        let wedge = thin_screen(7.0 * PI / 4.0);
        let low = wedge.relative_gain(PROPAGATION_BAND_CENTERS[0]);
        let mid = wedge.relative_gain(PROPAGATION_BAND_CENTERS[1]);
        let high = wedge.relative_gain(PROPAGATION_BAND_CENTERS[2]);
        assert!(low > mid, "low {low} should exceed mid {mid}");
        assert!(mid > high, "mid {mid} should exceed high {high}");
    }

    #[test]
    fn gain_scales_as_inverse_sqrt_frequency() {
        let wedge = thin_screen(7.0 * PI / 4.0);
        let g1 = wedge.relative_gain(1_000.0);
        let g4 = wedge.relative_gain(4_000.0);
        // |D| ~ 1/sqrt(f): quadrupling f should roughly halve the gain.
        let ratio = g1 / g4;
        assert!((ratio - 2.0).abs() < 0.3, "ratio {ratio} should be near 2");
    }

    #[test]
    fn shadow_boundary_is_about_half() {
        // At the shadow boundary phi = phi' + pi the diffracted field bridges
        // the geometric-optics step, so the relative gain is close to 1/2.
        let phi_i = PI / 4.0;
        let wedge = UtdWedge::new(2.0, PI / 2.0, phi_i, phi_i + PI + 0.02, 5.0, 5.0, C);
        let g = wedge.relative_gain(1_000.0);
        assert!(
            (g - 0.5).abs() < 0.2,
            "boundary gain {g} should be near 0.5"
        );
    }

    #[test]
    fn boundary_limit_is_continuous() {
        // Approach the shadow boundary and confirm the coefficient stays finite
        // and varies smoothly (no divergence as cot -> infinity, F -> 0).
        let phi_i = PI / 4.0;
        let boundary = phi_i + PI;
        let mut prev: Option<Sample> = None;
        for step in 1..=8 {
            let delta = 0.08 / (step as Sample);
            let wedge = UtdWedge::new(2.0, PI / 2.0, phi_i, boundary + delta, 5.0, 5.0, C);
            let mag = wedge.coefficient(1_000.0).magnitude();
            assert!(mag.is_finite(), "mag not finite at delta {delta}");
            if let Some(p) = prev {
                assert!(
                    (mag - p).abs() < 0.5,
                    "jump {} at delta {delta}",
                    (mag - p).abs()
                );
            }
            prev = Some(mag);
        }
    }

    #[test]
    fn reciprocity_source_receiver_swap() {
        // Swapping source and receiver (s <-> s', phi <-> phi') leaves |D| and
        // the relative gain invariant.
        let a = UtdWedge::new(1.5, PI / 2.0, 0.6, 3.4, 2.0, 5.0, C);
        let b = UtdWedge::new(1.5, PI / 2.0, 3.4, 0.6, 5.0, 2.0, C);
        for &f in &[200.0, 2_000.0, 10_000.0] {
            let ga = a.relative_gain(f);
            let gb = b.relative_gain(f);
            assert!(
                (ga - gb).abs() < 1e-4 * (1.0 + ga.abs()),
                "f={f} {ga} vs {gb}"
            );
        }
    }

    #[test]
    fn band_gains_are_clamped_and_ordered() {
        let wedge = thin_screen(7.0 * PI / 4.0);
        let bands = wedge.band_gains();
        for g in bands.bands() {
            assert!((0.0..=1.0).contains(&g), "band gain {g} out of range");
        }
        assert!(bands.high() <= bands.low() + 1e-3);
    }

    #[test]
    fn from_geometry_thin_screen_smoke() {
        // Edge along +Y; reference face along +X. Source on +X side, receiver
        // bent around into the shadow (-X side).
        let wedge = UtdWedge::from_geometry(
            Vec3::new(2.0, 0.0, 1.0),
            Vec3::ZERO,
            Vec3::Y,
            Vec3::new(-2.0, 0.0, 1.0),
            Vec3::X,
            2.0,
            C,
        );
        let g = wedge.relative_gain(1_000.0);
        assert!(g.is_finite() && g > 0.0, "gain {g}");
        let bands = wedge.band_gains();
        for b in bands.bands() {
            assert!((0.0..=1.0).contains(&b));
        }
    }

    #[test]
    fn new_clamps_degenerate_inputs() {
        let wedge = UtdWedge::new(Sample::NAN, -1.0, 10.0 * PI, -3.0 * PI, -1.0, 0.0, -5.0);
        // Everything resolved to a finite, usable configuration.
        let d = wedge.coefficient(1_000.0);
        assert!(d.re.is_finite() && d.im.is_finite());
        // The transition helper is still reachable (keeps the import meaningful).
        assert_eq!(_transition_check(-1.0), TransitionValue::ZERO);
    }
}
