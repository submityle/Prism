//! Rectangular-room eigenmodes: the discrete low-frequency standing waves
//! (room modes) of a shoebox enclosure.
//!
//! At low frequencies a rectangular room does not behave as a diffuse field but
//! as a set of discrete resonances. For a room with rigid walls and interior
//! dimensions `(Lx, Ly, Lz)` the modal frequencies are the classic
//! Rayleigh solution
//!
//! `f(nx, ny, nz) = (c / 2) * sqrt((nx/Lx)^2 + (ny/Ly)^2 + (nz/Lz)^2)`
//!
//! where `(nx, ny, nz)` are non-negative integers, not all zero, and `c` is the
//! speed of sound. Each triple is one eigenmode. Modes are classified by how
//! many indices are non-zero:
//!
//! - **axial** (one non-zero index): a wave bouncing between one opposing wall
//!   pair; the strongest resonances;
//! - **tangential** (two non-zero): a wave grazing four walls; weaker;
//! - **oblique** (three non-zero): a wave touching all six walls; weakest.
//!
//! The classic relative modal energy weighting is `4 : 2 : 1`
//! (axial : tangential : oblique), which this module normalises to
//! `1.0 : 0.5 : 0.25` so an axial mode has unit weight.
//!
//! This module is a control-rate estimator: it enumerates the modes below a
//! frequency ceiling, sorts them ascending, and exposes derived quantities
//! (modal density, Schroeder crossover, per-octave-band coloration) for tuning
//! a low-frequency equaliser or modal reverberator. It performs no per-sample
//! DSP.
//!
//! # Control rate, not audio rate
//!
//! Enumeration writes into a fixed-capacity stack array
//! ([`MAX_ROOM_MODES`]); there is no heap allocation, no locking, and no
//! panicking. Degenerate geometry (a non-positive dimension), a non-finite
//! ceiling, or an empty mode set all return safe finite values. All roots go
//! through [`bevy_math::ops`], never through `f32` intrinsics.
//!
//! # Provenance
//!
//! This is the textbook wave-acoustics description of rectangular-room modes:
//! the Rayleigh eigenfrequency formula and axial/tangential/oblique
//! classification as presented in H. Kuttruff's *Room Acoustics*, the modal
//! density estimate `dN/df = 4 * PI * V * f^2 / c^3` (Maa / Kuttruff), and the
//! Schroeder crossover frequency. This module is engine-agnostic and contains
//! **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or Google
//! Resonance Audio source or derived code**; it is implemented purely from that
//! publicly documented acoustics knowledge.

use bevy_math::ops;
use core::f32::consts::{PI, SQRT_2};

use prism_audio_core::math::Sample;

use crate::early_reflections::ShoeboxRoom;
use crate::material_library::{OCTAVE_BAND_CENTERS, OCTAVE_BAND_COUNT};
use crate::room_acoustics::schroeder_frequency;

/// Maximum number of modes retained by [`RoomModes`]. When more modes fall
/// below the ceiling, the lowest-frequency ones are kept.
pub const MAX_ROOM_MODES: usize = 64;

/// Upper bound on a single axis mode index during enumeration, guarding against
/// pathological (very large or very small) dimensions.
const MAX_AXIS_MODE_INDEX: usize = 32;

/// The speed of sound in dry air at room temperature, in metres per second.
///
/// Defined locally rather than re-exported to avoid clashing with the constant
/// of the same name in [`crate::early_reflections`].
const DEFAULT_SOUND_SPEED: Sample = 343.0;

/// Smallest divisor used to keep ratios finite for degenerate geometry.
const MIN_DIVISOR: Sample = 1e-9;

/// The classification of a rectangular-room eigenmode by how many of its three
/// indices are non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ModeKind {
    /// One non-zero index: a wave between one opposing wall pair (strongest).
    Axial,
    /// Two non-zero indices: a wave grazing four walls.
    Tangential,
    /// Three non-zero indices: a wave touching all six walls (weakest).
    Oblique,
}

impl ModeKind {
    /// The classic relative modal energy weight, normalised so an axial mode is
    /// `1.0` (axial `1.0`, tangential `0.5`, oblique `0.25`).
    #[must_use]
    pub fn weight(self) -> Sample {
        match self {
            ModeKind::Axial => 1.0,
            ModeKind::Tangential => 0.5,
            ModeKind::Oblique => 0.25,
        }
    }
}

/// A single rectangular-room eigenmode.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RoomMode {
    /// The modal resonance frequency in hertz.
    pub frequency_hz: Sample,
    /// The axial/tangential/oblique classification.
    pub kind: ModeKind,
    /// The relative modal energy weight (see [`ModeKind::weight`]).
    pub weight: Sample,
}

impl RoomMode {
    /// A zeroed placeholder mode used to initialise the fixed-capacity buffer.
    const EMPTY: Self = Self {
        frequency_hz: 0.0,
        kind: ModeKind::Axial,
        weight: 0.0,
    };
}

/// A control-rate enumeration of a shoebox room's low-frequency eigenmodes.
///
/// Built with [`RoomModes::from_shoebox`], which enumerates every mode below a
/// frequency ceiling, sorts them ascending, and keeps the lowest-frequency
/// [`MAX_ROOM_MODES`].
#[derive(Clone, Copy, Debug)]
pub struct RoomModes {
    modes: [RoomMode; MAX_ROOM_MODES],
    count: usize,
    volume: Sample,
    sound_speed: Sample,
}

impl RoomModes {
    /// Enumerates the eigenmodes of `room` up to `max_frequency_hz`.
    ///
    /// `sound_speed` falls back to a room-temperature default when non-finite
    /// or non-positive. Any dimension that is not strictly positive contributes
    /// no modes on its axis. Modes are returned ascending by frequency; when
    /// more than [`MAX_ROOM_MODES`] fall below the ceiling, the lowest are kept.
    ///
    /// # Examples
    ///
    /// ```
    /// use bevy_math::Vec3;
    /// use prism_audio_spatial::early_reflections::ShoeboxRoom;
    /// use prism_audio_spatial::room_modes::RoomModes;
    ///
    /// // A 5 m cube: the first axial mode is c / (2 * L) = 343 / 10 = 34.3 Hz.
    /// let room = ShoeboxRoom::new(Vec3::ZERO, Vec3::splat(5.0), [0.0; 6]);
    /// let modes = RoomModes::from_shoebox(&room, 200.0, 343.0);
    /// assert!(modes.count() > 0);
    /// assert!((modes.fundamental_hz() - 34.3).abs() < 0.1);
    /// ```
    #[must_use]
    pub fn from_shoebox(room: &ShoeboxRoom, max_frequency_hz: Sample, sound_speed: Sample) -> Self {
        let c = if sound_speed.is_finite() && sound_speed > 0.0 {
            sound_speed
        } else {
            DEFAULT_SOUND_SPEED
        };
        let size = room.size();
        let lx = size.x.max(0.0);
        let ly = size.y.max(0.0);
        let lz = size.z.max(0.0);
        let volume = (lx * ly * lz).max(0.0);

        let mut modes = [RoomMode::EMPTY; MAX_ROOM_MODES];
        let mut count = 0usize;

        if !max_frequency_hz.is_finite() || max_frequency_hz <= 0.0 {
            return Self { modes, count, volume, sound_speed: c };
        }

        let nx_max = axis_index_limit(lx, max_frequency_hz, c);
        let ny_max = axis_index_limit(ly, max_frequency_hz, c);
        let nz_max = axis_index_limit(lz, max_frequency_hz, c);

        let mut nx = 0usize;
        while nx <= nx_max {
            let rx = if lx > 0.0 { nx as Sample / lx } else { 0.0 };
            let mut ny = 0usize;
            while ny <= ny_max {
                let ry = if ly > 0.0 { ny as Sample / ly } else { 0.0 };
                let mut nz = 0usize;
                while nz <= nz_max {
                    if nx == 0 && ny == 0 && nz == 0 {
                        nz += 1;
                        continue;
                    }
                    let rz = if lz > 0.0 { nz as Sample / lz } else { 0.0 };
                    let sum = rx * rx + ry * ry + rz * rz;
                    let freq = 0.5 * c * ops::sqrt(sum);
                    if freq.is_finite() && freq > 0.0 && freq <= max_frequency_hz {
                        let nonzero = usize::from(nx > 0)
                            + usize::from(ny > 0)
                            + usize::from(nz > 0);
                        let kind = match nonzero {
                            1 => ModeKind::Axial,
                            2 => ModeKind::Tangential,
                            _ => ModeKind::Oblique,
                        };
                        let mode = RoomMode {
                            frequency_hz: freq,
                            kind,
                            weight: kind.weight(),
                        };
                        insert_mode(&mut modes, &mut count, mode);
                    }
                    nz += 1;
                }
                ny += 1;
            }
            nx += 1;
        }

        Self { modes, count, volume, sound_speed: c }
    }

    /// The enumerated modes, ascending by frequency.
    #[must_use]
    pub fn modes(&self) -> &[RoomMode] {
        &self.modes[..self.count]
    }

    /// The number of enumerated modes.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// The room volume in cubic metres.
    #[must_use]
    pub fn volume(&self) -> Sample {
        self.volume
    }

    /// The lowest modal frequency in hertz, or `0` when no modes exist.
    #[must_use]
    pub fn fundamental_hz(&self) -> Sample {
        if self.count == 0 {
            0.0
        } else {
            self.modes[0].frequency_hz
        }
    }

    /// The Schroeder crossover frequency in hertz for a given `rt60`.
    ///
    /// Above this frequency modes overlap into a statistically diffuse field;
    /// below it the discrete modes in [`RoomModes::modes`] dominate. Delegates
    /// to [`crate::room_acoustics::schroeder_frequency`].
    #[must_use]
    pub fn schroeder_frequency(&self, rt60: Sample) -> Sample {
        schroeder_frequency(rt60, self.volume)
    }

    /// The asymptotic modal density `dN/df = 4 * PI * V * f^2 / c^3` in modes
    /// per hertz at frequency `freq_hz`.
    ///
    /// This is the smooth large-room approximation; it rises with the square of
    /// frequency. Returns `0` for a degenerate room or a non-finite/non-positive
    /// frequency.
    #[must_use]
    pub fn modal_density(&self, freq_hz: Sample) -> Sample {
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            return 0.0;
        }
        if self.volume <= MIN_DIVISOR || self.sound_speed <= MIN_DIVISOR {
            return 0.0;
        }
        let c = self.sound_speed;
        let density = 4.0 * PI * self.volume * freq_hz * freq_hz / (c * c * c);
        if density.is_finite() {
            density
        } else {
            0.0
        }
    }

    /// Per-octave-band modal coloration: the summed modal weight falling into
    /// each of the eight [`OCTAVE_BAND_CENTERS`] bands.
    ///
    /// A band with a large value has many (or strong) resonances and is a
    /// candidate for low-frequency equalisation. Frequencies below the lowest
    /// band centre fall into band 0; frequencies above the highest fall into
    /// the last band. All values are non-negative.
    #[must_use]
    pub fn band_coloration(&self) -> [Sample; OCTAVE_BAND_COUNT] {
        let mut out = [0.0; OCTAVE_BAND_COUNT];
        for mode in &self.modes[..self.count] {
            let idx = octave_band_index(mode.frequency_hz);
            out[idx] += mode.weight;
        }
        out
    }
}

/// The largest axis mode index whose pure axial frequency stays at or below
/// `max_freq`, capped at [`MAX_AXIS_MODE_INDEX`]. Zero when `length <= 0`.
fn axis_index_limit(length: Sample, max_freq: Sample, c: Sample) -> usize {
    if length <= 0.0 {
        return 0;
    }
    let mut n = 0usize;
    while n < MAX_AXIS_MODE_INDEX {
        let freq = 0.5 * c * ((n + 1) as Sample) / length;
        if freq > max_freq {
            break;
        }
        n += 1;
    }
    n
}

/// Inserts `mode` into `modes` keeping ascending frequency order and at most
/// [`MAX_ROOM_MODES`] entries (dropping the highest frequency when full).
fn insert_mode(modes: &mut [RoomMode; MAX_ROOM_MODES], count: &mut usize, mode: RoomMode) {
    if *count == MAX_ROOM_MODES && mode.frequency_hz >= modes[MAX_ROOM_MODES - 1].frequency_hz {
        return;
    }
    let mut pos = 0usize;
    while pos < *count && modes[pos].frequency_hz <= mode.frequency_hz {
        pos += 1;
    }
    let mut j = if *count < MAX_ROOM_MODES { *count } else { MAX_ROOM_MODES - 1 };
    while j > pos {
        modes[j] = modes[j - 1];
        j -= 1;
    }
    modes[pos] = mode;
    if *count < MAX_ROOM_MODES {
        *count += 1;
    }
}

/// The octave-band index for `freq_hz`, using geometric (`* sqrt(2)`) upper
/// edges around each [`OCTAVE_BAND_CENTERS`] centre.
fn octave_band_index(freq_hz: Sample) -> usize {
    for (i, &center) in OCTAVE_BAND_CENTERS.iter().enumerate() {
        if freq_hz <= center * SQRT_2 {
            return i;
        }
    }
    OCTAVE_BAND_COUNT - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec3;

    fn cube(side: Sample) -> ShoeboxRoom {
        ShoeboxRoom::new(Vec3::ZERO, Vec3::splat(side), [0.0; 6])
    }

    fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
        (a - b).abs() <= eps
    }

    fn first_of(modes: &RoomModes, kind: ModeKind) -> Option<RoomMode> {
        modes.modes().iter().copied().find(|m| m.kind == kind)
    }

    #[test]
    fn cube_axial_frequency_matches_closed_form() {
        let side = 5.0;
        let c = 343.0;
        let modes = RoomModes::from_shoebox(&cube(side), 200.0, c);
        let expected = c / (2.0 * side);
        assert!(approx(modes.fundamental_hz(), expected, 1e-3), "f0 {}", modes.fundamental_hz());
        let axial = first_of(&modes, ModeKind::Axial).expect("axial mode");
        assert!(approx(axial.frequency_hz, expected, 1e-3));
    }

    #[test]
    fn modes_are_sorted_ascending() {
        let modes = RoomModes::from_shoebox(&cube(6.0), 300.0, 343.0);
        assert!(modes.count() > 3);
        let slice = modes.modes();
        for w in slice.windows(2) {
            assert!(w[0].frequency_hz <= w[1].frequency_hz, "not ascending");
        }
    }

    #[test]
    fn classification_covers_all_kinds() {
        let modes = RoomModes::from_shoebox(&cube(5.0), 200.0, 343.0);
        assert!(first_of(&modes, ModeKind::Axial).is_some());
        assert!(first_of(&modes, ModeKind::Tangential).is_some());
        assert!(first_of(&modes, ModeKind::Oblique).is_some());
        // For a cube the tangential fundamental is sqrt(2) times the axial one,
        // and the oblique fundamental is sqrt(3) times it.
        let axial = first_of(&modes, ModeKind::Axial).unwrap().frequency_hz;
        let tang = first_of(&modes, ModeKind::Tangential).unwrap().frequency_hz;
        let obl = first_of(&modes, ModeKind::Oblique).unwrap().frequency_hz;
        assert!(approx(tang, axial * ops::sqrt(2.0), 1e-2));
        assert!(approx(obl, axial * ops::sqrt(3.0), 1e-2));
    }

    #[test]
    fn axial_outweighs_tangential_outweighs_oblique() {
        assert!(ModeKind::Axial.weight() > ModeKind::Tangential.weight());
        assert!(ModeKind::Tangential.weight() > ModeKind::Oblique.weight());
        assert!(approx(ModeKind::Axial.weight(), 1.0, 1e-9));
    }

    #[test]
    fn degenerate_zero_dimension_is_safe() {
        // A flat room (zero height): no y-axis modes, but x/z modes remain.
        let flat = ShoeboxRoom::new(Vec3::ZERO, Vec3::new(5.0, 0.0, 7.0), [0.0; 6]);
        let modes = RoomModes::from_shoebox(&flat, 200.0, 343.0);
        for mode in modes.modes() {
            assert!(mode.frequency_hz.is_finite() && mode.frequency_hz > 0.0);
        }
        // A fully collapsed room yields no modes at all.
        let point = ShoeboxRoom::new(Vec3::ZERO, Vec3::ZERO, [0.0; 6]);
        assert_eq!(RoomModes::from_shoebox(&point, 200.0, 343.0).count(), 0);
    }

    #[test]
    fn modal_density_rises_with_frequency_squared() {
        let modes = RoomModes::from_shoebox(&cube(8.0), 100.0, 343.0);
        let d1 = modes.modal_density(50.0);
        let d2 = modes.modal_density(100.0);
        assert!(d2 > d1);
        assert!(approx(d2 / d1, 4.0, 1e-3), "ratio {}", d2 / d1);
        // Larger room -> denser modes at the same frequency.
        let big = RoomModes::from_shoebox(&cube(16.0), 100.0, 343.0);
        assert!(big.modal_density(50.0) > modes.modal_density(50.0));
    }

    #[test]
    fn top_k_truncation_keeps_lowest() {
        // A large room with a high ceiling produces far more than MAX modes.
        let modes = RoomModes::from_shoebox(&cube(20.0), 400.0, 343.0);
        assert_eq!(modes.count(), MAX_ROOM_MODES);
        // Retained set is still ascending and starts at the fundamental.
        assert!(approx(modes.fundamental_hz(), 343.0 / 40.0, 1e-2));
        let slice = modes.modes();
        for w in slice.windows(2) {
            assert!(w[0].frequency_hz <= w[1].frequency_hz);
        }
    }

    #[test]
    fn fundamental_is_the_lowest_mode() {
        let modes = RoomModes::from_shoebox(&cube(4.0), 300.0, 343.0);
        let f0 = modes.fundamental_hz();
        for mode in modes.modes() {
            assert!(mode.frequency_hz >= f0 - 1e-4);
        }
    }

    #[test]
    fn schroeder_transition_is_reasonable() {
        let modes = RoomModes::from_shoebox(&cube(6.0), 300.0, 343.0);
        // A live cube (long RT60) has a higher Schroeder frequency than a dead
        // one (short RT60).
        let live = modes.schroeder_frequency(1.5);
        let dead = modes.schroeder_frequency(0.3);
        assert!(live > dead);
        assert!(live.is_finite() && live > 0.0);
        // Degenerate reverberation time is safe.
        assert!(approx(modes.schroeder_frequency(0.0), 0.0, 1e-9));
    }

    #[test]
    fn non_finite_inputs_are_safe() {
        let room = cube(5.0);
        assert_eq!(RoomModes::from_shoebox(&room, Sample::NAN, 343.0).count(), 0);
        assert_eq!(RoomModes::from_shoebox(&room, -10.0, 343.0).count(), 0);
        // Bad sound speed falls back to the default and still enumerates.
        let fallback = RoomModes::from_shoebox(&room, 200.0, -1.0);
        assert!(fallback.count() > 0);
        assert!(approx(fallback.fundamental_hz(), 343.0 / 10.0, 1e-2));
        assert!(approx(modes_density_zero(&room), 0.0, 1e-9));
    }

    fn modes_density_zero(room: &ShoeboxRoom) -> Sample {
        let modes = RoomModes::from_shoebox(room, 200.0, 343.0);
        modes.modal_density(Sample::NAN) + modes.modal_density(-5.0)
    }

    #[test]
    fn band_coloration_is_non_negative_and_conserves_weight() {
        let modes = RoomModes::from_shoebox(&cube(6.0), 300.0, 343.0);
        let bands = modes.band_coloration();
        let mut band_sum = 0.0;
        for b in bands {
            assert!(b.is_finite() && b >= 0.0, "bad band {b}");
            band_sum += b;
        }
        let mut weight_sum = 0.0;
        for mode in modes.modes() {
            weight_sum += mode.weight;
        }
        assert!(approx(band_sum, weight_sum, 1e-3), "band {band_sum} weight {weight_sum}");
    }
}
