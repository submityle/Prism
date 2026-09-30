//! Statistical room-acoustics estimation: classic reverberation-time and
//! room-constant formulas derived from geometry and surface absorption.
//!
//! [`crate::early_reflections`] renders the *early* part of a room response
//! (the first discrete image-source arrivals). The *late* part -- the dense,
//! diffuse reverberant tail -- is not traced ray by ray; it is described
//! statistically by a handful of century-old closed-form results. This module
//! turns a [`ShoeboxRoom`] or [`Room`] into those late-field parameters
//! (reverberation time, Schroeder crossover frequency, mean free path, critical
//! distance) so an upstream feedback-delay-network reverb can be tuned directly
//! from scene geometry, bridging the gap between the early reflections and the
//! statistical late tail.
//!
//! # The models (all classic statistical acoustics)
//!
//! * **Mean free path.** For a diffuse field in a convex room the average
//!   distance between successive wall reflections is `4V / S` (volume over
//!   surface area) -- the Kosten / Sabine mean free path.
//! * **Sabine reverberation time.** `RT60 = k * V / A`, where `A = sum_i(S_i *
//!   alpha_i)` is the total absorption in metric sabins and `k = 0.161 s/m` is
//!   the metric Sabine constant (`24 ln(10) / c` at `c = 343 m/s`). Sabine
//!   assumes low, evenly distributed absorption.
//! * **Eyring reverberation time.** `RT60 = k * V / (-S * ln(1 - alpha_bar))`
//!   with `alpha_bar = A / S` the mean absorption. Eyring corrects Sabine for
//!   high absorption: as `alpha_bar -> 1` the tail collapses toward zero. It is
//!   always less than or equal to the Sabine estimate for a uniform room.
//! * **Millington-Sette reverberation time.** `RT60 = k * V / (-sum_i(S_i *
//!   ln(1 - alpha_i)))`, applying the logarithmic absorption per face rather
//!   than to the average, appropriate for very non-uniform absorption.
//! * **Schroeder frequency.** `f_c = 2000 * sqrt(RT60 / V)`, the crossover above
//!   which modal overlap makes the field statistically diffuse (below it,
//!   individual room modes dominate).
//! * **Critical distance.** `r_c ~= 0.057 * sqrt(V / RT60)`, the distance from a
//!   source at which the direct sound and the reverberant field are equal in
//!   level. More absorption shortens `RT60` and therefore pushes `r_c` outward.
//!
//! All absorption coefficients are energy absorption in `[0, 1]`. For a
//! [`Room`], whose single [`AcousticMaterial`] carries an energy *reflection*
//! coefficient, the per-surface absorption is `1 - reflection`.
//!
//! # Control rate, not audio rate
//!
//! Every function here is a control-rate estimate: pure scalar arithmetic that
//! allocates nothing, locks nothing, and cannot panic. Degenerate geometry
//! (zero volume, zero surface area, fully absorptive or fully reflective walls)
//! returns a safe finite value rather than a `NaN` or an infinity: divisors are
//! floored and the mean absorption is clamped just below unity so the logarithm
//! stays finite. All square roots, logarithms, and exponentials route through
//! [`bevy_math::ops`] for deterministic, golden-comparable results.
//!
//! # Provenance
//!
//! These are the textbook statistical-acoustics results: W. C. Sabine's
//! reverberation equation (*Collected Papers on Acoustics*, ca. 1900), C. F.
//! Eyring's reverberation formula (*J. Acoust. Soc. Am.*, 1930), G. Millington's
//! per-surface refinement (1932), and M. R. Schroeder's modal-crossover
//! frequency (1962); the critical-distance approximation follows the standard
//! Sabine room-constant relation. This module is engine-agnostic and contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google
//! Resonance Audio source or derived code**; it is implemented purely from that
//! publicly documented acoustics knowledge.

use bevy_math::ops;
use prism_audio_core::math::Sample;

use crate::early_reflections::ShoeboxRoom;
use crate::propagation::AcousticMaterial;
use crate::rooms::Room;

/// The metric Sabine constant `k = 24 ln(10) / c` in seconds per metre, taken
/// at `c = 343 m/s`. This is the `0.161` that multiplies `V / A`.
pub const SABINE_CONSTANT: Sample = 0.161;

/// The critical-distance coefficient in the Sabine approximation
/// `r_c = 0.057 * sqrt(V / RT60)` (metres, with `V` in cubic metres and `RT60`
/// in seconds).
pub const CRITICAL_DISTANCE_CONSTANT: Sample = 0.057;

/// The Schroeder-frequency coefficient in `f_c = 2000 * sqrt(RT60 / V)`.
pub const SCHROEDER_CONSTANT: Sample = 2000.0;

/// Smallest divisor tolerated before a quantity is treated as degenerate; keeps
/// reciprocals finite instead of producing an infinity.
const MIN_DIVISOR: Sample = 1.0e-9;

/// Mean absorption is clamped to at most this value so `ln(1 - alpha_bar)` stays
/// finite (a perfectly absorptive room would otherwise drive it to negative
/// infinity). At this clamp the Eyring / Millington tail is already effectively
/// zero.
const MAX_MEAN_ABSORPTION: Sample = 0.9999;

/// The Kosten / Sabine **mean free path** `4V / S` (metres): the average
/// straight-line distance between successive wall reflections in a diffuse
/// field.
///
/// Returns `0` when the surface area is non-positive.
///
/// # Examples
///
/// ```
/// use prism_audio_spatial::room_acoustics::mean_free_path;
/// // A 2 m cube: V = 8, S = 24, so 4V/S = 32/24 = 4/3.
/// assert!((mean_free_path(8.0, 24.0) - 4.0 / 3.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn mean_free_path(volume: Sample, surface_area: Sample) -> Sample {
    if surface_area <= MIN_DIVISOR || volume <= 0.0 {
        return 0.0;
    }
    4.0 * volume / surface_area
}

/// The **Sabine** reverberation time `RT60 = k * V / A`, where `A` is the total
/// absorption in metric sabins (`sum_i(S_i * alpha_i)`).
///
/// Returns `0` for a non-positive volume; the absorption is floored at a tiny
/// positive value so a fully reflective room yields a large but finite value
/// rather than an infinity.
#[inline]
#[must_use]
pub fn sabine_rt60(volume: Sample, total_absorption: Sample) -> Sample {
    if volume <= 0.0 {
        return 0.0;
    }
    let a = total_absorption.max(MIN_DIVISOR);
    SABINE_CONSTANT * volume / a
}

/// The **Eyring** reverberation time
/// `RT60 = k * V / (-S * ln(1 - alpha_bar))`.
///
/// `mean_absorption` (`alpha_bar`) is clamped to `[0, MAX_MEAN_ABSORPTION]` so
/// the logarithm stays finite; a fully absorptive room therefore returns a value
/// near zero. Returns `0` for a non-positive volume or surface area. For a
/// uniform room this is always less than or equal to [`sabine_rt60`].
#[inline]
#[must_use]
pub fn eyring_rt60(volume: Sample, surface_area: Sample, mean_absorption: Sample) -> Sample {
    if volume <= 0.0 || surface_area <= MIN_DIVISOR {
        return 0.0;
    }
    let alpha = mean_absorption.clamp(0.0, MAX_MEAN_ABSORPTION);
    // -ln(1 - alpha) >= 0, and >= alpha, so the denominator dominates Sabine's.
    let denom = -surface_area * ops::ln(1.0 - alpha);
    let denom = denom.max(MIN_DIVISOR);
    SABINE_CONSTANT * volume / denom
}

/// The **Millington-Sette** reverberation time
/// `RT60 = k * V / (-sum_i(S_i * ln(1 - alpha_i)))`, applying the logarithm per
/// face.
///
/// Each `(area, alpha)` face contributes `-area * ln(1 - alpha)` with `alpha`
/// clamped to `[0, MAX_MEAN_ABSORPTION]`. Returns `0` for a non-positive volume
/// or when the summed absorption term is non-positive.
#[must_use]
pub fn millington_sette_rt60(volume: Sample, faces: &[(Sample, Sample)]) -> Sample {
    if volume <= 0.0 {
        return 0.0;
    }
    let mut denom: Sample = 0.0;
    for &(area, alpha) in faces {
        if area <= 0.0 {
            continue;
        }
        let a = alpha.clamp(0.0, MAX_MEAN_ABSORPTION);
        denom += -area * ops::ln(1.0 - a);
    }
    let denom = denom.max(MIN_DIVISOR);
    SABINE_CONSTANT * volume / denom
}

/// The **Schroeder frequency** `f_c = 2000 * sqrt(RT60 / V)` in hertz: the
/// crossover above which room modes overlap into a statistically diffuse field.
///
/// Returns `0` for a non-positive volume or reverberation time.
#[inline]
#[must_use]
pub fn schroeder_frequency(rt60: Sample, volume: Sample) -> Sample {
    if volume <= MIN_DIVISOR || rt60 <= 0.0 {
        return 0.0;
    }
    SCHROEDER_CONSTANT * ops::sqrt(rt60 / volume)
}

/// The **critical distance** `r_c ~= 0.057 * sqrt(V / RT60)` in metres: the
/// distance from a source where the direct and reverberant fields are equal.
///
/// Returns `0` for a non-positive volume or reverberation time. More absorption
/// lowers `RT60` and therefore increases `r_c`.
#[inline]
#[must_use]
pub fn critical_distance(volume: Sample, rt60: Sample) -> Sample {
    if volume <= 0.0 || rt60 <= MIN_DIVISOR {
        return 0.0;
    }
    CRITICAL_DISTANCE_CONSTANT * ops::sqrt(volume / rt60)
}

/// Statistical late-field acoustics derived from a room's geometry and surface
/// absorption.
///
/// Constructed from a [`ShoeboxRoom`] (six independent face absorptions) or a
/// [`Room`] (a single broadband wall material). It caches the geometric
/// aggregates -- volume, total surface area, and total absorption in metric
/// sabins -- then serves the classic reverberation estimates on demand. All
/// accessors are control-rate, allocation free, and panic free.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RoomAcoustics {
    /// Room volume in cubic metres.
    volume: Sample,
    /// Total interior surface area in square metres.
    surface_area: Sample,
    /// Total absorption in metric sabins (`sum_i(S_i * alpha_i)`).
    total_absorption: Sample,
}

impl RoomAcoustics {
    /// Builds the acoustics from raw aggregates (volume, surface area, total
    /// absorption in sabins). Negative inputs are clamped to zero.
    #[must_use]
    #[inline]
    pub fn from_aggregates(volume: Sample, surface_area: Sample, total_absorption: Sample) -> Self {
        Self {
            volume: volume.max(0.0),
            surface_area: surface_area.max(0.0),
            total_absorption: total_absorption.max(0.0),
        }
    }

    /// Builds the acoustics from a [`ShoeboxRoom`], using its per-face energy
    /// absorption coefficients (`[x_low, x_high, y_low, y_high, z_low, z_high]`)
    /// and box dimensions.
    #[must_use]
    pub fn from_shoebox(room: &ShoeboxRoom) -> Self {
        let size = room.size();
        let lx = size.x.max(0.0);
        let ly = size.y.max(0.0);
        let lz = size.z.max(0.0);
        let volume = lx * ly * lz;

        // Face areas by axis: the two x-faces span (ly * lz), and so on.
        let area_x = ly * lz;
        let area_y = lx * lz;
        let area_z = lx * ly;
        let a = &room.wall_absorption;
        let surface_area = 2.0 * (area_x + area_y + area_z);
        let total_absorption = area_x * (a[0] + a[1])
            + area_y * (a[2] + a[3])
            + area_z * (a[4] + a[5]);
        Self {
            volume: volume.max(0.0),
            surface_area: surface_area.max(0.0),
            total_absorption: total_absorption.max(0.0),
        }
    }

    /// Builds the acoustics from a [`Room`], using its half-extents for geometry
    /// and its single wall material for a uniform absorption
    /// `alpha = 1 - reflection` across all six faces.
    #[must_use]
    pub fn from_room(room: &Room) -> Self {
        let lx = (2.0 * room.half_extents.x).max(0.0);
        let ly = (2.0 * room.half_extents.y).max(0.0);
        let lz = (2.0 * room.half_extents.z).max(0.0);
        let volume = lx * ly * lz;
        let surface_area = 2.0 * (lx * ly + ly * lz + lz * lx);
        let alpha = wall_absorption(&room.wall);
        let total_absorption = surface_area * alpha;
        Self {
            volume: volume.max(0.0),
            surface_area: surface_area.max(0.0),
            total_absorption: total_absorption.max(0.0),
        }
    }

    /// Room volume in cubic metres.
    #[inline]
    #[must_use]
    pub fn volume(&self) -> Sample {
        self.volume
    }

    /// Total interior surface area in square metres.
    #[inline]
    #[must_use]
    pub fn surface_area(&self) -> Sample {
        self.surface_area
    }

    /// Total absorption in metric sabins (`sum_i(S_i * alpha_i)`).
    #[inline]
    #[must_use]
    pub fn total_absorption(&self) -> Sample {
        self.total_absorption
    }

    /// Mean absorption `alpha_bar = A / S`, or `0` when there is no surface.
    #[inline]
    #[must_use]
    pub fn mean_absorption(&self) -> Sample {
        if self.surface_area <= MIN_DIVISOR {
            return 0.0;
        }
        (self.total_absorption / self.surface_area).clamp(0.0, 1.0)
    }

    /// The Kosten / Sabine mean free path `4V / S` (metres).
    #[inline]
    #[must_use]
    pub fn mean_free_path(&self) -> Sample {
        mean_free_path(self.volume, self.surface_area)
    }

    /// The Sabine reverberation time (seconds).
    #[inline]
    #[must_use]
    pub fn rt60_sabine(&self) -> Sample {
        sabine_rt60(self.volume, self.total_absorption)
    }

    /// The Eyring reverberation time (seconds).
    #[inline]
    #[must_use]
    pub fn rt60_eyring(&self) -> Sample {
        eyring_rt60(self.volume, self.surface_area, self.mean_absorption())
    }

    /// The Schroeder crossover frequency (hertz), using the Sabine `RT60`.
    #[inline]
    #[must_use]
    pub fn schroeder_freq(&self) -> Sample {
        schroeder_frequency(self.rt60_sabine(), self.volume)
    }

    /// The critical distance (metres), using the Sabine `RT60`.
    #[inline]
    #[must_use]
    pub fn critical_distance(&self) -> Sample {
        critical_distance(self.volume, self.rt60_sabine())
    }
}

/// The energy absorption coefficient of a wall material, `1 - reflection`,
/// clamped to `[0, 1]`.
#[inline]
#[must_use]
fn wall_absorption(material: &AcousticMaterial) -> Sample {
    (1.0 - material.reflection_gain()).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
        (a - b).abs() <= tol
    }

    /// A cube of side `a` with uniform per-face absorption `alpha`.
    fn cube_shoebox(a: Sample, alpha: Sample) -> ShoeboxRoom {
        ShoeboxRoom::new(Vec3::ZERO, Vec3::splat(a), [alpha; 6])
    }

    #[test]
    fn mean_free_path_is_exact() {
        // 10 m cube: V = 1000, S = 600, 4V/S = 4000/600 = 6.6667.
        let acoustics = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.2));
        assert!(approx(acoustics.mean_free_path(), 4000.0 / 600.0, 1e-4));
        assert!(approx(mean_free_path(1000.0, 600.0), 4000.0 / 600.0, 1e-4));
    }

    #[test]
    fn sabine_matches_analytic_cube() {
        // 10 m cube, alpha = 0.1: A = 600 * 0.1 = 60, RT60 = 0.161*1000/60.
        let acoustics = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.1));
        let expected = SABINE_CONSTANT * 1000.0 / 60.0;
        assert!(approx(acoustics.rt60_sabine(), expected, 1e-4));
        assert!(approx(acoustics.volume(), 1000.0, 1e-3));
        assert!(approx(acoustics.surface_area(), 600.0, 1e-3));
        assert!(approx(acoustics.mean_absorption(), 0.1, 1e-6));
    }

    #[test]
    fn more_absorption_shortens_rt60() {
        let quiet = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.05));
        let dead = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.5));
        assert!(dead.rt60_sabine() < quiet.rt60_sabine());
        assert!(dead.rt60_eyring() < quiet.rt60_eyring());
    }

    #[test]
    fn eyring_never_exceeds_sabine() {
        for &alpha in &[0.02_f32, 0.1, 0.3, 0.6, 0.9] {
            let acoustics = RoomAcoustics::from_shoebox(&cube_shoebox(8.0, alpha));
            let sab = acoustics.rt60_sabine();
            let eyr = acoustics.rt60_eyring();
            assert!(eyr <= sab + 1e-4, "alpha={alpha}: eyring {eyr} > sabine {sab}");
        }
    }

    #[test]
    fn eyring_approaches_sabine_at_low_absorption() {
        // At very low absorption -ln(1-alpha) ~= alpha, so the two nearly agree.
        let acoustics = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.02));
        let sab = acoustics.rt60_sabine();
        let eyr = acoustics.rt60_eyring();
        assert!(approx(eyr, sab, 0.05 * sab));
    }

    #[test]
    fn eyring_much_shorter_at_high_absorption() {
        let acoustics = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.8));
        let sab = acoustics.rt60_sabine();
        let eyr = acoustics.rt60_eyring();
        assert!(eyr < 0.6 * sab);
    }

    #[test]
    fn full_absorption_is_near_zero_and_finite() {
        let acoustics = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 1.0));
        let eyr = acoustics.rt60_eyring();
        assert!(eyr.is_finite());
        assert!(eyr >= 0.0 && eyr < 0.1);
        // Sabine with full absorption stays finite too (A = S).
        assert!(acoustics.rt60_sabine().is_finite());
    }

    #[test]
    fn schroeder_rises_with_rt60_and_falls_with_volume() {
        let live = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.05));
        let dead = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.5));
        // Longer RT60 (same volume) -> higher Schroeder frequency.
        assert!(live.schroeder_freq() > dead.schroeder_freq());

        // Larger volume (same RT60) -> lower Schroeder frequency.
        let small = schroeder_frequency(1.0, 100.0);
        let big = schroeder_frequency(1.0, 1000.0);
        assert!(big < small);
    }

    #[test]
    fn critical_distance_increases_with_absorption() {
        let live = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.05));
        let dead = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.5));
        // More absorption -> shorter RT60 -> larger critical distance.
        assert!(dead.critical_distance() > live.critical_distance());
    }

    #[test]
    fn millington_matches_uniform_room_ballpark() {
        // For uniform absorption Millington equals Eyring (same per-face log).
        let alpha = 0.3;
        let acoustics = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, alpha));
        let faces = [
            (100.0_f32, alpha),
            (100.0, alpha),
            (100.0, alpha),
            (100.0, alpha),
            (100.0, alpha),
            (100.0, alpha),
        ];
        let mill = millington_sette_rt60(1000.0, &faces);
        assert!(approx(mill, acoustics.rt60_eyring(), 1e-3));
    }

    #[test]
    fn from_room_uses_reflection_complement() {
        // reflection 0.8 -> absorption 0.2.
        let material = AcousticMaterial::new(40.0, 0.8);
        let room = Room::new(
            crate::rooms::RoomId(0),
            Vec3::ZERO,
            Vec3::splat(5.0),
            material,
        );
        let acoustics = RoomAcoustics::from_room(&room);
        // 10 m cube (half_extents 5).
        assert!(approx(acoustics.volume(), 1000.0, 1e-3));
        assert!(approx(acoustics.surface_area(), 600.0, 1e-3));
        assert!(approx(acoustics.mean_absorption(), 0.2, 1e-6));
    }

    #[test]
    fn degenerate_geometry_is_safe() {
        // Zero-size room.
        let flat = ShoeboxRoom::new(Vec3::ZERO, Vec3::ZERO, [0.5; 6]);
        let acoustics = RoomAcoustics::from_shoebox(&flat);
        assert_eq!(acoustics.volume(), 0.0);
        assert_eq!(acoustics.rt60_sabine(), 0.0);
        assert_eq!(acoustics.rt60_eyring(), 0.0);
        assert_eq!(acoustics.schroeder_freq(), 0.0);
        assert_eq!(acoustics.critical_distance(), 0.0);
        assert_eq!(acoustics.mean_free_path(), 0.0);
        assert_eq!(acoustics.mean_absorption(), 0.0);

        // Free functions with degenerate inputs.
        assert_eq!(mean_free_path(0.0, 0.0), 0.0);
        assert_eq!(sabine_rt60(-1.0, 10.0), 0.0);
        assert_eq!(eyring_rt60(1000.0, 0.0, 0.3), 0.0);
        assert_eq!(schroeder_frequency(-1.0, 100.0), 0.0);
        assert_eq!(critical_distance(1000.0, 0.0), 0.0);
        assert!(millington_sette_rt60(1000.0, &[]).is_finite());
    }

    #[test]
    fn zero_absorption_room_is_finite() {
        // Perfectly reflective walls: RT60 large but finite, never infinite.
        let acoustics = RoomAcoustics::from_shoebox(&cube_shoebox(10.0, 0.0));
        assert!(acoustics.rt60_sabine().is_finite());
        assert!(acoustics.rt60_sabine() > 0.0);
    }
}
