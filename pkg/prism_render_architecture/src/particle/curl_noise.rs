//! Analytic divergence-free curl-noise velocity field for particle advection
//! (design §8, §10).
//!
//! This module synthesises an incompressible turbulence velocity field from a
//! hash-seeded scalar potential, with no simulation state and no stored grid.
//! It is the self-contained contract behind the "curl noise" advection force
//! used by production `VFX` stacks (Unreal `Niagara`, Unity `VFX Graph`,
//! `Houdini` flow noise): a smooth vector potential `Ψ` is built from three
//! decorrelated pseudo-noise channels, and the velocity is its analytic curl
//! `∇ × Ψ`, which is divergence-free by construction.
//!
//! The key numerical property is exploited deliberately: the velocity is taken
//! as a *central-difference* curl of the potential, and [`CurlNoiseField::divergence`]
//! estimates the divergence with the *same* central-difference stencil and
//! step. Because discrete central-difference operators commute
//! (`D_x D_y = D_y D_x` as an exact algebraic identity on the sample lattice),
//! the discrete divergence of the discrete curl cancels to floating-point
//! rounding regardless of the step, so the field is verifiably incompressible
//! on the `CPU`.
//!
//! Determinism matches the sibling particle modules: the only floating-point
//! primitives beyond ordinary arithmetic are `f32::floor` (integer lattice
//! location) and `sqrt` (through [`Vec3::length`]). There are no transcendental
//! calls (`sin` / `cos` / `exp` / `ln` / `pow`); the fade is the multiply-only
//! smoothstep polynomial `t * t * (3 - 2 t)`, interpolation is linear, and all
//! randomness flows through an integer hash. The result is bit-reproducible
//! against a future `GPU` kernel that hashes the same lattice cells.
//!
//! Scope: this module is orthogonal to [`super::noise`] (general noise
//! primitives and `fBm`) and to [`super::vector_field`] (sampling of a *stored*
//! velocity field). It owns only the analytic, storage-free curl-noise
//! generator and its hand-rolled vector math, and it imports no sibling
//! particle module.

/// Central-difference half-step used both to take the analytic curl of the
/// potential and to re-estimate the divergence. Small enough that the
/// derivative truncation stays well under the field scale, large enough that
/// `f32` cancellation is negligible for potentials of order one.
pub const CURL_EPS: f32 = 1.0e-2;

/// Scale that turns a 24-bit hash mantissa into the half-open range `[0, 1)`.
const INV_2POW24: f32 = 1.0 / 16_777_216.0;

/// Odd-integer salt selecting the seed of the first potential channel.
const POT_SEED_X: u32 = 0x68E3_1DA4;

/// Odd-integer salt selecting the seed of the second potential channel, so the
/// channel is statistically independent of the first.
const POT_SEED_Y: u32 = 0xB543_9C13;

/// Odd-integer salt selecting the seed of the third potential channel (distinct
/// from the other two).
const POT_SEED_Z: u32 = 0x2545_F491;

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

    /// Cross product `self × rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only a
    /// comparison is needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length (the one and only `sqrt` in the module).
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
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

/// Stateless integer hash of a lattice cell and seed (the pseudo-noise `RNG`).
///
/// The cell coordinates are folded into the seed with pure integer
/// multiply / xor / rotate steps, so the hash has no state, never calls a
/// transcendental function, and agrees bit for bit between the `CPU` reference
/// and a future `GPU` kernel. Negative coordinates address the whole signed
/// lattice through a two's-complement cast.
#[must_use]
fn hash_cell(i: i32, j: i32, k: i32, seed: u32) -> u32 {
    let mut h = seed ^ 0x811C_9DC5;
    h = mix(h, i as u32);
    h = mix(h, j as u32);
    h = mix(h, k as u32);
    finalize(h)
}

/// The reproducible scalar value assigned to a lattice cell, in `[-1, 1)`.
#[must_use]
fn cell_value(i: i32, j: i32, k: i32, seed: u32) -> f32 {
    let h = hash_cell(i, j, k, seed);
    let unit = ((h >> 8) as f32) * INV_2POW24;
    unit * 2.0 - 1.0
}

/// The smoothstep fade `t * t * (3 - 2 t)`: a multiply-only Hermite polynomial
/// with zero first derivative at `0` and `1`, so trilinear interpolation across
/// cells is `C1`-continuous without any transcendental call.
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

/// Trilinearly interpolated, smoothstep-faded value noise in `[-1, 1]`.
///
/// The eight surrounding lattice-cell values are blended with the fade weights,
/// yielding a smooth pseudo-random scalar field that is a pure function of the
/// position and seed.
#[must_use]
fn value_noise(p: Vec3, seed: u32) -> f32 {
    let (xi, xf) = floor_split(p.x);
    let (yi, yf) = floor_split(p.y);
    let (zi, zf) = floor_split(p.z);

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let c000 = cell_value(xi, yi, zi, seed);
    let c100 = cell_value(xi + 1, yi, zi, seed);
    let c010 = cell_value(xi, yi + 1, zi, seed);
    let c110 = cell_value(xi + 1, yi + 1, zi, seed);
    let c001 = cell_value(xi, yi, zi + 1, seed);
    let c101 = cell_value(xi + 1, yi, zi + 1, seed);
    let c011 = cell_value(xi, yi + 1, zi + 1, seed);
    let c111 = cell_value(xi + 1, yi + 1, zi + 1, seed);

    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);

    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);

    lerp(y0, y1, w)
}

/// An analytic, storage-free curl-noise velocity field for particle advection.
///
/// The field owns a domain `frequency` (how quickly the noise varies in space),
/// an output `amplitude` (the peak velocity scale), and a `seed` selecting the
/// pseudo-random realization. Its velocity is the analytic curl of a
/// hash-seeded vector potential and is divergence-free by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurlNoiseField {
    /// Spatial frequency: positions are scaled by this before the potential is
    /// sampled, so a larger value packs more swirls into the same space. A
    /// value of `0` collapses the potential to a constant (a zero field).
    pub frequency: f32,
    /// Peak velocity scale multiplying the raw curl. A value of `0` yields the
    /// zero field.
    pub amplitude: f32,
    /// Seed selecting the pseudo-random realization; equal seeds reproduce
    /// equal fields on every backend.
    pub seed: u32,
}

impl CurlNoiseField {
    /// Builds a field from its frequency, amplitude, and seed.
    #[must_use]
    pub const fn new(frequency: f32, amplitude: f32, seed: u32) -> Self {
        Self {
            frequency,
            amplitude,
            seed,
        }
    }

    /// Evaluates the vector potential `Ψ(p)` whose curl is the velocity field.
    ///
    /// The three components are decorrelated value-noise channels sampled at the
    /// frequency-scaled position, each in `[-1, 1]`.
    #[must_use]
    pub fn potential(&self, p: Vec3) -> Vec3 {
        let q = p.scale(self.frequency);
        Vec3::new(
            value_noise(q, self.seed ^ POT_SEED_X),
            value_noise(q, self.seed ^ POT_SEED_Y),
            value_noise(q, self.seed ^ POT_SEED_Z),
        )
    }

    /// Samples the divergence-free advection velocity at `p`.
    ///
    /// The velocity is the analytic curl `∇ × Ψ` of the potential, taken with a
    /// central difference of half-step [`CURL_EPS`] and scaled by
    /// [`CurlNoiseField::amplitude`]:
    ///
    /// - `vx = ∂Ψz/∂y − ∂Ψy/∂z`
    /// - `vy = ∂Ψx/∂z − ∂Ψz/∂x`
    /// - `vz = ∂Ψy/∂x − ∂Ψx/∂y`
    #[must_use]
    pub fn sample_velocity(&self, p: Vec3) -> Vec3 {
        let inv = 1.0 / (2.0 * CURL_EPS);

        let px_p = self.potential(p.add(Vec3::new(CURL_EPS, 0.0, 0.0)));
        let px_m = self.potential(p.sub(Vec3::new(CURL_EPS, 0.0, 0.0)));
        let py_p = self.potential(p.add(Vec3::new(0.0, CURL_EPS, 0.0)));
        let py_m = self.potential(p.sub(Vec3::new(0.0, CURL_EPS, 0.0)));
        let pz_p = self.potential(p.add(Vec3::new(0.0, 0.0, CURL_EPS)));
        let pz_m = self.potential(p.sub(Vec3::new(0.0, 0.0, CURL_EPS)));

        let dpz_dy = (py_p.z - py_m.z) * inv;
        let dpy_dz = (pz_p.y - pz_m.y) * inv;
        let dpx_dz = (pz_p.x - pz_m.x) * inv;
        let dpz_dx = (px_p.z - px_m.z) * inv;
        let dpy_dx = (px_p.y - px_m.y) * inv;
        let dpx_dy = (py_p.x - py_m.x) * inv;

        let vx = dpz_dy - dpy_dz;
        let vy = dpx_dz - dpz_dx;
        let vz = dpy_dx - dpx_dy;

        Vec3::new(vx, vy, vz).scale(self.amplitude)
    }

    /// Estimates the divergence `∇ · u` of the velocity field at `p` with a
    /// central difference of half-step [`CURL_EPS`].
    ///
    /// Because [`CurlNoiseField::sample_velocity`] is a central-difference curl
    /// taken with the same step, and discrete central-difference operators
    /// commute, this cancels to `f32` rounding for a smooth potential. Tests use
    /// it to confirm the field is incompressible.
    #[must_use]
    pub fn divergence(&self, p: Vec3) -> f32 {
        let inv = 1.0 / (2.0 * CURL_EPS);

        let dux_dx = (self.sample_velocity(p.add(Vec3::new(CURL_EPS, 0.0, 0.0))).x
            - self.sample_velocity(p.sub(Vec3::new(CURL_EPS, 0.0, 0.0))).x)
            * inv;
        let duy_dy = (self.sample_velocity(p.add(Vec3::new(0.0, CURL_EPS, 0.0))).y
            - self.sample_velocity(p.sub(Vec3::new(0.0, CURL_EPS, 0.0))).y)
            * inv;
        let duz_dz = (self.sample_velocity(p.add(Vec3::new(0.0, 0.0, CURL_EPS))).z
            - self.sample_velocity(p.sub(Vec3::new(0.0, 0.0, CURL_EPS))).z)
            * inv;

        dux_dx + duy_dy + duz_dz
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handful of irregular sample points spread across the signed lattice.
    const SAMPLES: [Vec3; 6] = [
        Vec3::new(0.37, -1.21, 2.05),
        Vec3::new(-3.05, 0.92, -0.48),
        Vec3::new(5.5, 5.5, 5.5),
        Vec3::new(-7.3, -2.1, 4.9),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(12.75, -8.4, 3.33),
    ];

    #[test]
    fn velocity_field_is_divergence_free() {
        let field = CurlNoiseField::new(0.7, 2.5, 0x1234_5678);
        for p in SAMPLES {
            // The discrete curl/divergence stencils commute, so this cancels to
            // f32 rounding regardless of the frequency and amplitude.
            assert!(field.divergence(p).abs() < 1.0e-2);
        }
    }

    #[test]
    fn divergence_free_across_seeds_and_scales() {
        let fields = [
            CurlNoiseField::new(1.3, 1.0, 1),
            CurlNoiseField::new(0.25, 9.0, 99),
            CurlNoiseField::new(3.0, 0.5, 0xDEAD_BEEF),
        ];
        for field in fields {
            for p in SAMPLES {
                assert!(field.divergence(p).abs() < 1.0e-2);
            }
        }
    }

    #[test]
    fn sampling_is_deterministic() {
        let field = CurlNoiseField::new(0.9, 1.75, 42);
        for p in SAMPLES {
            let a = field.sample_velocity(p);
            let b = field.sample_velocity(p);
            assert!((a.x - b.x).abs() < CMP_EPS);
            assert!((a.y - b.y).abs() < CMP_EPS);
            assert!((a.z - b.z).abs() < CMP_EPS);
        }
    }

    #[test]
    fn distinct_seeds_produce_distinct_fields() {
        let a = CurlNoiseField::new(0.9, 1.0, 1);
        let b = CurlNoiseField::new(0.9, 1.0, 2);
        // At least one sample must differ; otherwise the seed is inert.
        let mut differs = false;
        for p in SAMPLES {
            let va = a.sample_velocity(p);
            let vb = b.sample_velocity(p);
            if (va.x - vb.x).abs() > CMP_EPS
                || (va.y - vb.y).abs() > CMP_EPS
                || (va.z - vb.z).abs() > CMP_EPS
            {
                differs = true;
            }
        }
        assert!(differs);
    }

    #[test]
    fn zero_frequency_is_a_constant_potential_and_zero_field() {
        let field = CurlNoiseField::new(0.0, 3.0, 7);
        for p in SAMPLES {
            let v = field.sample_velocity(p);
            assert!(v.length() < CMP_EPS);
            assert!(field.divergence(p).abs() < CMP_EPS);
        }
    }

    #[test]
    fn zero_amplitude_is_the_zero_field() {
        let field = CurlNoiseField::new(1.5, 0.0, 7);
        for p in SAMPLES {
            let v = field.sample_velocity(p);
            assert!(v.length() < CMP_EPS);
            assert!(field.divergence(p).abs() < CMP_EPS);
        }
    }

    #[test]
    fn vector_math_matches_hand_computation() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(b);
        assert!((c.x - 0.0).abs() < CMP_EPS);
        assert!((c.y - 0.0).abs() < CMP_EPS);
        assert!((c.z - 1.0).abs() < CMP_EPS);
        assert!((a.dot(b) - 0.0).abs() < CMP_EPS);
        assert!((Vec3::new(3.0, 4.0, 0.0).length() - 5.0).abs() < CMP_EPS);
        let s = a.add(b).sub(b).scale(2.0);
        assert!((s.x - 2.0).abs() < CMP_EPS);
    }
}
