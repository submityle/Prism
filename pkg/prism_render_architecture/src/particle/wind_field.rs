//! Art-directable deterministic wind model: directional wind, gust envelope,
//! height falloff, and relative-velocity drag (design §8.2).
//!
//! This module is a parametric *wind model* an artist can dial in, not a
//! general solver. A constant base wind blows along a chosen direction, a
//! deterministic gust envelope modulates its strength over space and time, a
//! polynomial height falloff weakens the wind with altitude, and a drag helper
//! turns the relative wind into an acceleration. It is the self-contained
//! contract behind the "wind" module found in production `VFX` stacks (Unreal
//! `Niagara` wind, Unity `VFX Graph` force fields, `Houdini` wind), rebuilt
//! from scratch.
//!
//! It is deliberately orthogonal to its three siblings and imports none of
//! them:
//!
//! * [`super::forces`] owns the *generic* force integration and the physics
//!   field library; this module only produces a wind velocity and a single
//!   drag acceleration, leaving accumulation and integration to the caller.
//! * [`super::vector_field`] samples a *stored* velocity grid; this module has
//!   no storage and evaluates its field analytically from a handful of scalar
//!   parameters.
//! * [`super::curl_noise`] synthesises divergence-free turbulence; this module
//!   makes no incompressibility claim and instead shapes an artist-controlled
//!   directional gust.
//!
//! Determinism matches the sibling particle modules (design §29): the only
//! floating-point primitives beyond ordinary arithmetic are `f32::floor`
//! (integer lattice location) and `sqrt` (through [`Vec3::length`]). There are
//! no transcendental calls (`sin` / `cos` / `exp` / `ln` / `pow`); the height
//! falloff is the rational polynomial `1 / (1 + h * falloff)`, the gust fade is
//! the multiply-only smoothstep `t * t * (3 - 2 t)`, interpolation is linear,
//! and all randomness flows through an integer hash. The result is
//! bit-reproducible against a future `GPU` kernel that hashes the same lattice
//! cells. A stateless hash also stands in for a per-particle `RNG`.

/// Scale that turns a 24-bit hash mantissa into the half-open range `[0, 1)`.
const INV_2POW24: f32 = 1.0 / 16_777_216.0;

/// Odd-integer salt folded into the gust seed so the gust realization is
/// decorrelated from any other hash-seeded field that reuses the same seed.
const GUST_SALT: u32 = 0x9E37_79B1;

/// Below this squared length a direction is treated as degenerate and
/// normalization returns the zero vector instead of dividing by ~zero.
const LEN_EPS_SQ: f32 = 1.0e-12;

/// Smallest denominator allowed in the height falloff, so the rational falloff
/// never divides by (near) zero even for adversarial parameters.
const MIN_FALLOFF_DENOM: f32 = 1.0e-3;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1.0e-6;

/// A hand-rolled three-component vector, kept local so the module is a
/// zero-dependency contract and its vector math is auditable in one place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API is specified with named add/sub methods for call-site uniformity, matching the sibling particle contracts; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses a named sub for call-site uniformity, not the operator trait."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Euclidean length (the one and only `sqrt` in the module).
    #[must_use]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Unit-length copy, or the zero vector when the input is (near) zero.
    ///
    /// The zero-length guard keeps the wind direction well defined even when an
    /// artist leaves the direction unset, avoiding a divide-by-zero `NaN`.
    #[must_use]
    pub fn normalized(self) -> Self {
        let len_sq = self.dot(self);
        if len_sq < LEN_EPS_SQ {
            Self::ZERO
        } else {
            self.scale(1.0 / len_sq.sqrt())
        }
    }
}

/// One folding step of the hash: xor-in a multiplied input word, then rotate
/// and multiply to spread the bits before the next word is folded.
#[must_use]
fn mix(mut h: u32, v: u32) -> u32 {
    h ^= v.wrapping_mul(0x9E37_79B1);
    h = h.rotate_left(15).wrapping_mul(0x85EB_CA6B);
    h
}

/// Final avalanche (an integer bit-mixer) applied once after all inputs are
/// folded, giving a well-distributed 32-bit result.
#[must_use]
fn finalize(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}

/// Stateless integer hash of a 4D lattice cell and seed (the pseudo-noise
/// `RNG`).
///
/// The cell coordinates (three spatial plus one temporal) are folded into the
/// seed with pure integer multiply / xor / rotate steps, so the hash has no
/// state, never calls a transcendental function, and agrees bit for bit
/// between the `CPU` reference and a future `GPU` kernel. Negative coordinates
/// address the whole signed lattice through a two's-complement cast.
#[must_use]
fn hash_cell(i: i32, j: i32, k: i32, l: i32, seed: u32) -> u32 {
    let mut h = seed ^ 0x811C_9DC5;
    h = mix(h, i as u32);
    h = mix(h, j as u32);
    h = mix(h, k as u32);
    h = mix(h, l as u32);
    finalize(h)
}

/// The reproducible scalar value assigned to a lattice cell, in `[-1, 1)`.
#[must_use]
fn cell_value(i: i32, j: i32, k: i32, l: i32, seed: u32) -> f32 {
    let h = hash_cell(i, j, k, l, seed);
    let unit = ((h >> 8) as f32) * INV_2POW24;
    unit * 2.0 - 1.0
}

/// The smoothstep fade `t * t * (3 - 2 t)`: a multiply-only Hermite polynomial
/// with zero first derivative at `0` and `1`, so interpolation across cells is
/// `C1`-continuous without any transcendental call.
#[must_use]
fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation `a + (b - a) * t`.
#[must_use]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Splits a coordinate into its floor cell index and the fractional offset in
/// `[0, 1)`.
#[must_use]
fn floor_split(x: f32) -> (i32, f32) {
    let f = x.floor();
    (f as i32, x - f)
}

/// Trilinearly interpolated value of one temporal slice `l` of the lattice.
#[must_use]
fn spatial_slice(xi: i32, yi: i32, zi: i32, l: i32, seed: u32, u: f32, v: f32, w: f32) -> f32 {
    let c000 = cell_value(xi, yi, zi, l, seed);
    let c100 = cell_value(xi + 1, yi, zi, l, seed);
    let c010 = cell_value(xi, yi + 1, zi, l, seed);
    let c110 = cell_value(xi + 1, yi + 1, zi, l, seed);
    let c001 = cell_value(xi, yi, zi + 1, l, seed);
    let c101 = cell_value(xi + 1, yi, zi + 1, l, seed);
    let c011 = cell_value(xi, yi + 1, zi + 1, l, seed);
    let c111 = cell_value(xi + 1, yi + 1, zi + 1, l, seed);

    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);

    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);

    lerp(y0, y1, w)
}

/// Smoothstep-faded value noise over space and time, in `[-1, 1]`.
///
/// Two spatial slices at the floor and ceiling time cells are blended with the
/// faded time fraction, so the gust is `C1`-continuous in both space and time
/// and is a pure function of the position, time, and seed.
#[must_use]
fn value_noise_4d(p: Vec3, t: f32, seed: u32) -> f32 {
    let (xi, xf) = floor_split(p.x);
    let (yi, yf) = floor_split(p.y);
    let (zi, zf) = floor_split(p.z);
    let (ti, tf) = floor_split(t);

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);
    let s = fade(tf);

    let slice0 = spatial_slice(xi, yi, zi, ti, seed, u, v, w);
    let slice1 = spatial_slice(xi, yi, zi, ti + 1, seed, u, v, w);
    lerp(slice0, slice1, s)
}

/// An art-directable, storage-free wind field: a base directional wind shaped
/// by a deterministic gust envelope and weakened with altitude.
///
/// The field owns a unit wind `direction`, a `base_speed` (the calm-air wind
/// magnitude), a `gust_amplitude` (peak extra speed the gusts add), a
/// `gust_frequency` (how quickly gusts vary in space and time), a
/// `height_falloff` (how fast the wind weakens with altitude), and a `seed`
/// selecting the gust realization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindField {
    /// Unit wind direction; [`WindField::new`] normalizes it on construction.
    pub direction: Vec3,
    /// Calm-air wind magnitude along [`WindField::direction`].
    pub base_speed: f32,
    /// Peak extra speed the gust envelope adds along the wind direction.
    pub gust_amplitude: f32,
    /// Spatial/temporal frequency of the gusts: larger packs more, faster
    /// gusts into the same space and time.
    pub gust_frequency: f32,
    /// Rate at which the wind weakens with altitude in the rational falloff
    /// `1 / (1 + h * falloff)`; `0` disables the falloff.
    pub height_falloff: f32,
    /// Seed selecting the pseudo-random gust realization; equal seeds reproduce
    /// equal fields on every backend.
    pub seed: u32,
}

impl WindField {
    /// Builds a wind field, normalizing `direction` to a unit vector.
    ///
    /// A zero `direction` is preserved as the zero vector (see
    /// [`Vec3::normalized`]), which yields a zero wind everywhere.
    #[must_use]
    pub fn new(
        direction: Vec3,
        base_speed: f32,
        gust_amplitude: f32,
        gust_frequency: f32,
        height_falloff: f32,
        seed: u32,
    ) -> Self {
        Self {
            direction: direction.normalized(),
            base_speed,
            gust_amplitude,
            gust_frequency,
            height_falloff,
            seed,
        }
    }

    /// The gust envelope at a position and time, in `[0, 1]`.
    ///
    /// The value is the space-and-time value noise remapped from `[-1, 1]` to
    /// `[0, 1]` and clamped, giving a smooth, deterministic gust factor that
    /// swells and lulls without any transcendental call. Equal `seed`, `time`,
    /// and `position` always return the same value.
    #[must_use]
    pub fn gust_envelope(&self, position: Vec3, time: f32) -> f32 {
        let q = position.scale(self.gust_frequency);
        let t = time * self.gust_frequency;
        let n = value_noise_4d(q, t, self.seed ^ GUST_SALT);
        let unit = (n + 1.0) * 0.5;
        unit.clamp(0.0, 1.0)
    }

    /// The rational height falloff factor at altitude `height`, in `(0, 1]`.
    ///
    /// Uses the polynomial `1 / (1 + h * falloff)` on the clamped altitude
    /// `max(height, 0)`, so the wind is at full strength at and below the
    /// ground plane and monotonically weakens as altitude rises. The
    /// denominator is floored at [`MIN_FALLOFF_DENOM`] so the division is
    /// always well defined.
    #[must_use]
    pub fn height_attenuation(&self, height: f32) -> f32 {
        let h = height.max(0.0);
        let denom = (1.0 + h * self.height_falloff).max(MIN_FALLOFF_DENOM);
        1.0 / denom
    }

    /// Samples the wind velocity at a position and time.
    ///
    /// The velocity is the base wind `direction * base_speed` plus the gust
    /// `direction * gust_amplitude * gust_envelope`, with the whole sum weakened
    /// by [`WindField::height_attenuation`] at the position's altitude
    /// (`position.y`). With `gust_amplitude == 0` the result is exactly the
    /// attenuated base wind.
    #[must_use]
    pub fn sample_velocity(&self, position: Vec3, time: f32) -> Vec3 {
        let gust = self.gust_amplitude * self.gust_envelope(position, time);
        let speed = self.base_speed + gust;
        let atten = self.height_attenuation(position.y);
        self.direction.scale(speed * atten)
    }

    /// The drag acceleration a particle feels from the wind.
    ///
    /// This is the linear relative-velocity drag `(wind - particle) * drag`, so
    /// a particle is accelerated toward the local wind velocity and a particle
    /// already moving with the wind feels no drag.
    #[must_use]
    pub fn drag_acceleration(wind_vel: Vec3, particle_vel: Vec3, drag_coeff: f32) -> Vec3 {
        wind_vel.sub(particle_vel).scale(drag_coeff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    #[test]
    fn new_normalizes_direction() {
        let w = WindField::new(Vec3::new(0.0, 0.0, 5.0), 3.0, 0.0, 1.0, 0.0, 7);
        assert!(approx(w.direction.length(), 1.0));
        assert!(approx_vec(w.direction, Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn zero_direction_guard_yields_zero_wind() {
        let w = WindField::new(Vec3::ZERO, 10.0, 4.0, 1.0, 0.5, 3);
        assert_eq!(w.direction, Vec3::ZERO);
        let v = w.sample_velocity(Vec3::new(1.0, 2.0, 3.0), 0.25);
        assert!(approx_vec(v, Vec3::ZERO));
    }

    #[test]
    fn base_speed_scales_wind_linearly() {
        let a = WindField::new(Vec3::new(1.0, 0.0, 0.0), 2.0, 0.0, 1.0, 0.0, 1);
        let b = WindField::new(Vec3::new(1.0, 0.0, 0.0), 4.0, 0.0, 1.0, 0.0, 1);
        let pos = Vec3::new(0.3, 0.0, -0.7);
        let va = a.sample_velocity(pos, 0.5);
        let vb = b.sample_velocity(pos, 0.5);
        assert!(approx_vec(vb, va.scale(2.0)));
    }

    #[test]
    fn zero_gust_amplitude_is_pure_base_wind() {
        let w = WindField::new(Vec3::new(0.0, 0.0, 1.0), 6.0, 0.0, 2.0, 0.0, 42);
        // Altitude zero so the falloff factor is exactly one.
        let v = w.sample_velocity(Vec3::new(1.5, 0.0, -2.0), 1.25);
        assert!(approx_vec(v, Vec3::new(0.0, 0.0, 6.0)));
    }

    #[test]
    fn zero_base_and_gust_is_zero_velocity() {
        let w = WindField::new(Vec3::new(1.0, 1.0, 0.0), 0.0, 0.0, 1.0, 0.5, 9);
        let v = w.sample_velocity(Vec3::new(2.0, 3.0, 4.0), 0.75);
        assert!(approx_vec(v, Vec3::ZERO));
    }

    #[test]
    fn gust_envelope_is_deterministic_and_in_unit_range() {
        let w = WindField::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0, 1.5, 0.0, 123);
        let pos = Vec3::new(0.4, -1.2, 3.3);
        let g0 = w.gust_envelope(pos, 2.5);
        let g1 = w.gust_envelope(pos, 2.5);
        assert!(approx(g0, g1));
        assert!((0.0..=1.0).contains(&g0));
    }

    #[test]
    fn gust_changes_over_time_and_space() {
        let w = WindField::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0, 1.0, 0.0, 55);
        let base = Vec3::new(0.0, 0.0, 0.0);
        let g_here = w.gust_envelope(base, 0.0);
        let g_later = w.gust_envelope(base, 4.7);
        let g_there = w.gust_envelope(Vec3::new(5.3, 2.1, -3.9), 0.0);
        assert!(!approx(g_here, g_later));
        assert!(!approx(g_here, g_there));
    }

    #[test]
    fn gust_adds_along_wind_direction() {
        let w = WindField::new(Vec3::new(1.0, 0.0, 0.0), 3.0, 5.0, 1.3, 0.0, 8);
        let pos = Vec3::new(0.6, 0.0, 0.2);
        let g = w.gust_envelope(pos, 1.1);
        let v = w.sample_velocity(pos, 1.1);
        assert!(approx_vec(v, Vec3::new(3.0 + 5.0 * g, 0.0, 0.0)));
    }

    #[test]
    fn height_falloff_is_monotonically_weaker_with_altitude() {
        let w = WindField::new(Vec3::new(1.0, 0.0, 0.0), 10.0, 0.0, 1.0, 0.5, 4);
        let low = w.sample_velocity(Vec3::new(0.0, 1.0, 0.0), 0.0).length();
        let mid = w.sample_velocity(Vec3::new(0.0, 5.0, 0.0), 0.0).length();
        let high = w.sample_velocity(Vec3::new(0.0, 20.0, 0.0), 0.0).length();
        assert!(low > mid);
        assert!(mid > high);
        assert!(high > 0.0);
    }

    #[test]
    fn height_attenuation_is_one_at_and_below_ground() {
        let w = WindField::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 0.0, 1.0, 2.0, 0);
        assert!(approx(w.height_attenuation(0.0), 1.0));
        assert!(approx(w.height_attenuation(-9.0), 1.0));
    }

    #[test]
    fn drag_is_relative_velocity_times_coefficient() {
        let wind = Vec3::new(4.0, 0.0, -2.0);
        let particle = Vec3::new(1.0, 1.0, 0.0);
        let a = WindField::drag_acceleration(wind, particle, 0.5);
        assert!(approx_vec(a, Vec3::new(1.5, -0.5, -1.0)));
    }

    #[test]
    fn drag_is_zero_when_particle_matches_wind() {
        let wind = Vec3::new(2.0, -3.0, 1.0);
        let a = WindField::drag_acceleration(wind, wind, 0.9);
        assert!(approx_vec(a, Vec3::ZERO));
    }
}
