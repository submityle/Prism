//! Procedural noise fields for divergence-free turbulence forces (design §8,
//! §10).
//!
//! This is the analytic, simulation-free companion to [`super::fluid`]. Where
//! `fluid.rs` reconstructs vorticity from a *simulated* velocity grid with
//! finite differences (its [`super::fluid::NeighborVelocities::curl`] and
//! [`super::fluid::vorticity_confinement_force`] read stored neighbour samples
//! of a stable-fluids solve), this module synthesises a velocity field
//! *procedurally* from a hash-seeded lattice, with no grid and no simulation
//! state. The two are orthogonal: pyro effects run the grid solver, while
//! cheap ambient turbulence, dust swirls, and stylised flow use this analytic
//! layer without any solver dependency.
//!
//! It is the `CPU`-verifiable contract behind the curl-noise turbulence force
//! shipped by Unreal `Niagara` (its "Curl Noise Force" module), Unity's
//! `VFX Graph` ("Turbulence"), and `Houdini`'s noise-driven flow fields. The
//! stack is the classic one:
//!
//! 1. **Hash `RNG`** — a stateless integer avalanche (a `PCG`-style /
//!    Wang-hash mix of pure integer multiply/xor/rotate) maps a lattice cell
//!    `(i, j, k)` and a seed to a reproducible gradient direction; see
//!    [`hash_lattice`] and [`lattice_gradient`].
//! 2. **Value and gradient noise** — [`value_noise_3d`] trilinearly blends
//!    hashed cell values, while [`gradient_noise_3d`] is `Perlin`-style
//!    gradient noise, both smoothed by the quintic fade `6t^5 - 15t^4 + 10t^3`.
//! 3. **`fBm`** — [`fbm`] sums octaves of gradient noise with configurable
//!    lacunarity and gain for fractal detail.
//! 4. **Analytic curl noise** — [`curl_noise_3d`] and [`curl_noise_fbm`] take
//!    the analytic curl `∇ × Ψ` of a vector potential `Ψ` built from three
//!    decorrelated noise fields, yielding an (approximately) divergence-free
//!    velocity field: the incompressible swirl at the heart of the turbulence
//!    force. [`turbulence_force`] wraps it with frequency / amplitude controls.
//!
//! Determinism matches the sibling particle modules: the only floating-point
//! primitives beyond ordinary arithmetic are `f32::floor` (integer lattice
//! location) and `sqrt` (through [`Vec3`]); there are no transcendental calls
//! (`sin`/`cos`/`exp`/`ln`/`pow`), the fade polynomial and interpolation are
//! multiply-only, and all randomness flows through the integer hash. The result
//! is bit-reproducible against a future `GPU` kernel that hashes the same cells.

use super::Vec3;

/// Small positive guard against division by a (near) zero denominator, used for
/// the `fBm` amplitude-normalisation sum.
const EPS: f32 = 1.0e-9;

/// Central-difference half-step used to take the analytic curl of the vector
/// potential. Small enough that the truncation error of the derivative stays
/// well under the field scale, large enough that `f32` cancellation is
/// negligible for potentials of order one.
const CURL_EPS: f32 = 1.0e-2;

/// Odd-integer salt mixed into the seed for the second component of the vector
/// potential, so the three potential channels are statistically independent.
const SEED_SALT_Y: u32 = 0x9E37_79B9;

/// Odd-integer salt mixed into the seed for the third component of the vector
/// potential (distinct from [`SEED_SALT_Y`]).
const SEED_SALT_Z: u32 = 0x85EB_CA6B;

/// Per-octave seed increment so successive `fBm` octaves draw from independent
/// gradient fields rather than a rescaling of the same one.
const SEED_STEP: u32 = 0x1656_67B1;

/// Scale that turns a 24-bit hash mantissa into the half-open range `[0, 1)`.
const INV_2POW24: f32 = 1.0 / 16_777_216.0;

/// Stateless integer hash of a lattice cell and seed (the noise `RNG`).
///
/// This is a pure `PCG`-style avalanche: the cell coordinates are folded into
/// the seed with integer multiply / xor / rotate steps and then finalised so
/// that flipping any input bit scrambles roughly half the output bits. It has
/// no state, so the `CPU` reference and a future `GPU` kernel agree bit for bit,
/// and it never calls a transcendental function.
///
/// Negative coordinates are reinterpreted through a two's-complement cast, so
/// the whole signed lattice is addressable.
#[must_use]
pub fn hash_lattice(i: i32, j: i32, k: i32, seed: u32) -> u32 {
    let mut h = seed ^ 0x811C_9DC5;
    h = mix(h, i as u32);
    h = mix(h, j as u32);
    h = mix(h, k as u32);
    finalize(h)
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

/// The reproducible gradient direction assigned to a lattice cell.
///
/// The hash selects one of the twelve cube-edge directions of `Perlin`'s
/// improved-noise gradient set (each a unit-ish vector with two non-zero,
/// signed components). Because the selection is a pure function of the cell and
/// seed, the gradient is stable across runs and backends, and it is never the
/// zero vector.
#[must_use]
pub fn lattice_gradient(i: i32, j: i32, k: i32, seed: u32) -> Vec3 {
    grad_select(hash_lattice(i, j, k, seed))
}

/// Maps the low bits of a hash to one of the twelve edge gradients.
///
/// This reproduces `Perlin`'s improved-noise `grad` selection as an explicit
/// vector: `u` runs along X or Y, `v` along Y, X, or Z, each independently
/// signed, so `grad_select(h).dot(offset)` equals the reference `grad` dot
/// product while exposing the gradient itself.
#[must_use]
fn grad_select(h: u32) -> Vec3 {
    let hh = h & 15;
    // The "u" axis is X for the low half of the range, Y for the high half.
    let (ux, uy) = if hh < 8 { (1.0, 0.0) } else { (0.0, 1.0) };
    // The "v" axis is Y, X, or Z depending on the sub-range.
    let (vx, vy, vz) = if hh < 4 {
        (0.0, 1.0, 0.0)
    } else if hh == 12 || hh == 14 {
        (1.0, 0.0, 0.0)
    } else {
        (0.0, 0.0, 1.0)
    };
    let su = if hh & 1 == 0 { 1.0 } else { -1.0 };
    let sv = if hh & 2 == 0 { 1.0 } else { -1.0 };
    Vec3::new(su * ux + sv * vx, su * uy + sv * vy, sv * vz)
}

/// The quintic fade `6t^5 - 15t^4 + 10t^3` (`Perlin`'s improved-noise easing).
///
/// Evaluated in `Horner` form (multiply / add only), it has zero first *and*
/// second derivatives at `t = 0` and `t = 1`, so tiled noise cells join with a
/// continuous gradient and no visible lattice seams.
#[must_use]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation `a + t * (b - a)`.
#[must_use]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Signed cell value in `[-1, 1)` hashed from a lattice cell, for value noise.
#[must_use]
fn cell_value(i: i32, j: i32, k: i32, seed: u32) -> f32 {
    let h = hash_lattice(i, j, k, seed);
    // Top 24 bits give a uniform mantissa in [0, 1); remap to [-1, 1).
    let unit = ((h >> 8) as f32) * INV_2POW24;
    unit * 2.0 - 1.0
}

/// Splits a coordinate into its floor cell index and the fractional offset
/// within the cell (`0.0..1.0`).
#[must_use]
fn floor_split(x: f32) -> (i32, f32) {
    let f = x.floor();
    (f as i32, x - f)
}

/// Three-dimensional value noise sampled at `pos` with the given `seed`.
///
/// Eight hashed cell corner values are blended with the quintic fade and
/// trilinear interpolation. The output is smooth, tiles on the integer lattice,
/// and stays within `[-1, 1]`; it is the cheapest noise primitive here and is a
/// good driver for scalar fields (density, size-over-life jitter).
#[must_use]
pub fn value_noise_3d(pos: Vec3, seed: u32) -> f32 {
    let (xi, xf) = floor_split(pos.x);
    let (yi, yf) = floor_split(pos.y);
    let (zi, zf) = floor_split(pos.z);

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

/// The gradient contribution of one lattice corner: its hashed gradient dotted
/// with the displacement from the corner to the sample point.
#[must_use]
fn corner_grad(i: i32, j: i32, k: i32, dx: f32, dy: f32, dz: f32, seed: u32) -> f32 {
    grad_select(hash_lattice(i, j, k, seed)).dot(Vec3::new(dx, dy, dz))
}

/// Three-dimensional `Perlin`-style gradient noise sampled at `pos`.
///
/// Each of the eight surrounding lattice cells contributes its hashed
/// [`lattice_gradient`] dotted with the displacement to `pos`; the eight
/// contributions are blended with the quintic fade. The field is exactly
/// zero at integer lattice points, is smoother than [`value_noise_3d`] (no
/// value discontinuity in the derivative), and is bounded within `[-1, 1]`
/// (the theoretical 3D bound is `sqrt(3) / 2`). It is the building block for
/// [`fbm`] and the vector potential of the curl-noise field.
#[must_use]
pub fn gradient_noise_3d(pos: Vec3, seed: u32) -> f32 {
    let (xi, xf) = floor_split(pos.x);
    let (yi, yf) = floor_split(pos.y);
    let (zi, zf) = floor_split(pos.z);

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let g000 = corner_grad(xi, yi, zi, xf, yf, zf, seed);
    let g100 = corner_grad(xi + 1, yi, zi, xf - 1.0, yf, zf, seed);
    let g010 = corner_grad(xi, yi + 1, zi, xf, yf - 1.0, zf, seed);
    let g110 = corner_grad(xi + 1, yi + 1, zi, xf - 1.0, yf - 1.0, zf, seed);
    let g001 = corner_grad(xi, yi, zi + 1, xf, yf, zf - 1.0, seed);
    let g101 = corner_grad(xi + 1, yi, zi + 1, xf - 1.0, yf, zf - 1.0, seed);
    let g011 = corner_grad(xi, yi + 1, zi + 1, xf, yf - 1.0, zf - 1.0, seed);
    let g111 = corner_grad(xi + 1, yi + 1, zi + 1, xf - 1.0, yf - 1.0, zf - 1.0, seed);

    let x00 = lerp(g000, g100, u);
    let x10 = lerp(g010, g110, u);
    let x01 = lerp(g001, g101, u);
    let x11 = lerp(g011, g111, u);
    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);
    lerp(y0, y1, w)
}

/// Fractional-Brownian-motion parameters: how many octaves of noise are summed
/// and how frequency and amplitude evolve between them.
///
/// `fBm` layers successively higher-frequency, lower-amplitude octaves of
/// [`gradient_noise_3d`] to build fractal detail. `lacunarity` is the
/// per-octave frequency multiplier (canonically `2.0`, "octaves") and `gain`
/// is the per-octave amplitude multiplier (canonically `0.5`, so the result is
/// self-similar). The struct carries an `f32`, so it derives `PartialEq` only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FbmParams {
    /// Number of noise octaves summed. Zero yields a flat (zero) field.
    pub octaves: u32,
    /// Per-octave frequency multiplier (`> 1` adds finer detail each octave).
    pub lacunarity: f32,
    /// Per-octave amplitude multiplier (`< 1` fades finer octaves out).
    pub gain: f32,
}

impl FbmParams {
    /// The cinematic default: four octaves, `lacunarity` `2.0`, `gain` `0.5`.
    pub const DEFAULT: Self = Self {
        octaves: 4,
        lacunarity: 2.0,
        gain: 0.5,
    };

    /// A single octave: `fbm` then reduces to plain [`gradient_noise_3d`].
    pub const SINGLE_OCTAVE: Self = Self {
        octaves: 1,
        lacunarity: 2.0,
        gain: 0.5,
    };

    /// Builds parameters from explicit octave count, lacunarity, and gain.
    #[must_use]
    pub const fn new(octaves: u32, lacunarity: f32, gain: f32) -> Self {
        Self {
            octaves,
            lacunarity,
            gain,
        }
    }
}

impl Default for FbmParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Fractional Brownian motion: amplitude-normalised sum of gradient-noise
/// octaves sampled at `pos`.
///
/// Successive octaves scale the sample position by `lacunarity`, scale the
/// contribution by `gain`, and advance the seed so the octaves are
/// decorrelated. The sum is divided by the total amplitude, so the output
/// stays within the `[-1, 1]` range of a single octave regardless of octave
/// count. Zero octaves (or a fully collapsed amplitude) yield `0.0` rather
/// than a `NaN`.
#[must_use]
pub fn fbm(pos: Vec3, params: FbmParams, seed: u32) -> f32 {
    let mut freq = 1.0_f32;
    let mut amp = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    let mut octave_seed = seed;

    for _ in 0..params.octaves {
        sum += gradient_noise_3d(pos.scale(freq), octave_seed) * amp;
        norm += amp;
        freq *= params.lacunarity;
        amp *= params.gain;
        octave_seed = octave_seed.wrapping_add(SEED_STEP);
    }

    if norm.abs() > EPS {
        sum / norm
    } else {
        0.0
    }
}

/// The three-component vector potential `Ψ` whose curl becomes the noise flow.
///
/// Each component is an independent `fBm` field: the seed is salted per channel
/// so the channels are statistically independent, which keeps the resulting
/// curl field isotropic rather than biased along an axis.
#[must_use]
fn potential(pos: Vec3, params: FbmParams, seed: u32) -> Vec3 {
    Vec3::new(
        fbm(pos, params, seed),
        fbm(pos, params, seed ^ SEED_SALT_Y),
        fbm(pos, params, seed ^ SEED_SALT_Z),
    )
}

/// Analytic curl `∇ × Ψ` of the vector potential, evaluated by central
/// differences with half-step [`CURL_EPS`].
///
/// Because the field is the curl of a potential, its divergence is analytically
/// zero, so the result is (to the order of the difference stencil) an
/// incompressible velocity field: particles advected by it swirl without
/// bunching up or thinning out.
#[must_use]
fn curl_of_potential(pos: Vec3, params: FbmParams, seed: u32) -> Vec3 {
    let e = CURL_EPS;
    let inv = 1.0 / (2.0 * e);

    let px = potential(pos.add(Vec3::new(e, 0.0, 0.0)), params, seed);
    let mx = potential(pos.sub(Vec3::new(e, 0.0, 0.0)), params, seed);
    let py = potential(pos.add(Vec3::new(0.0, e, 0.0)), params, seed);
    let my = potential(pos.sub(Vec3::new(0.0, e, 0.0)), params, seed);
    let pz = potential(pos.add(Vec3::new(0.0, 0.0, e)), params, seed);
    let mz = potential(pos.sub(Vec3::new(0.0, 0.0, e)), params, seed);

    // Partial derivatives of the potential along each axis.
    let dpsi_dx = px.sub(mx).scale(inv);
    let dpsi_dy = py.sub(my).scale(inv);
    let dpsi_dz = pz.sub(mz).scale(inv);

    // curl = (∂Ψz/∂y - ∂Ψy/∂z, ∂Ψx/∂z - ∂Ψz/∂x, ∂Ψy/∂x - ∂Ψx/∂y).
    Vec3::new(
        dpsi_dy.z - dpsi_dz.y,
        dpsi_dz.x - dpsi_dx.z,
        dpsi_dx.y - dpsi_dy.x,
    )
}

/// Single-octave analytic curl noise sampled at `pos`: a divergence-free
/// velocity field.
///
/// This is the canonical curl-noise flow used for cheap ambient turbulence.
/// For multi-scale detail use [`curl_noise_fbm`]; for a ready-to-apply force
/// with frequency and amplitude controls use [`turbulence_force`].
#[must_use]
pub fn curl_noise_3d(pos: Vec3, seed: u32) -> Vec3 {
    curl_of_potential(pos, FbmParams::SINGLE_OCTAVE, seed)
}

/// Multi-octave analytic curl noise: the curl of an `fBm` vector potential.
///
/// Layering octaves into the potential before taking the curl produces
/// large-scale swirls carrying smaller eddies, the visual signature of the
/// turbulence force in production `VFX` tools, while remaining divergence-free.
#[must_use]
pub fn curl_noise_fbm(pos: Vec3, params: FbmParams, seed: u32) -> Vec3 {
    curl_of_potential(pos, params, seed)
}

/// Controls for turning curl noise into an applied turbulence force (design
/// §8).
///
/// `frequency` scales the sample position (the spatial size of the swirls),
/// `amplitude` scales the resulting force, `fbm` selects the octave structure
/// of the potential, and `seed` reproducibly re-rolls the field. The struct
/// carries `f32` fields, so it derives `PartialEq` only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurbulenceParams {
    /// Spatial frequency: larger values shrink the swirls.
    pub frequency: f32,
    /// Force magnitude scale applied to the divergence-free field.
    pub amplitude: f32,
    /// Octave structure of the vector potential.
    pub fbm: FbmParams,
    /// Hash seed selecting the field instance.
    pub seed: u32,
}

impl TurbulenceParams {
    /// A reasonable cinematic default: unit frequency and amplitude with the
    /// default `fBm` octave structure.
    pub const DEFAULT: Self = Self {
        frequency: 1.0,
        amplitude: 1.0,
        fbm: FbmParams::DEFAULT,
        seed: 0x00C0_FFEE,
    };
}

impl Default for TurbulenceParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Evaluates the turbulence force at `pos` for the given parameters.
///
/// The world position is scaled by `frequency`, the divergence-free curl-noise
/// velocity is sampled there, and the result is scaled by `amplitude` into a
/// force suitable for accumulation in the simulation force loop (design §8). An
/// `amplitude` of zero yields exactly [`Vec3::ZERO`], never a `NaN`.
#[must_use]
pub fn turbulence_force(pos: Vec3, params: TurbulenceParams) -> Vec3 {
    let sample = pos.scale(params.frequency);
    curl_noise_fbm(sample, params.fbm, params.seed).scale(params.amplitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for approximate `f32` comparisons in the tests.
    const TOL: f32 = 1.0e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TOL
    }

    #[test]
    fn hash_is_deterministic_and_seed_sensitive() {
        assert_eq!(hash_lattice(3, -7, 11, 42), hash_lattice(3, -7, 11, 42));
        assert_ne!(hash_lattice(3, -7, 11, 42), hash_lattice(3, -7, 11, 43));
        assert_ne!(hash_lattice(3, -7, 11, 42), hash_lattice(4, -7, 11, 42));
        assert_ne!(hash_lattice(3, -7, 11, 42), hash_lattice(3, -6, 11, 42));
        assert_ne!(hash_lattice(3, -7, 11, 42), hash_lattice(3, -7, 12, 42));
    }

    #[test]
    fn hash_avalanche_spreads_bits() {
        // A single-bit change in a coordinate should flip many output bits.
        let a = hash_lattice(0, 0, 0, 1);
        let b = hash_lattice(1, 0, 0, 1);
        let flipped = (a ^ b).count_ones();
        assert!(
            (8..=24).contains(&flipped),
            "avalanche flipped {flipped} bits"
        );
    }

    #[test]
    fn lattice_gradient_is_reproducible_and_nonzero() {
        for i in -2..3 {
            for j in -2..3 {
                let g = lattice_gradient(i, j, 5, 7);
                assert_eq!(g, lattice_gradient(i, j, 5, 7));
                assert!(g.length_squared() > 0.5, "gradient must be non-zero");
            }
        }
    }

    #[test]
    fn fade_matches_quintic_at_key_points() {
        assert!(approx(fade(0.0), 0.0));
        assert!(approx(fade(1.0), 1.0));
        assert!(approx(fade(0.5), 0.5));
        // Monotonically increasing on [0, 1].
        assert!(fade(0.25) < fade(0.75));
    }

    #[test]
    fn lerp_endpoints_and_midpoint() {
        assert!(approx(lerp(2.0, 6.0, 0.0), 2.0));
        assert!(approx(lerp(2.0, 6.0, 1.0), 6.0));
        assert!(approx(lerp(2.0, 6.0, 0.5), 4.0));
    }

    #[test]
    fn value_noise_is_reproducible() {
        let p = Vec3::new(1.5, -2.25, 3.75);
        assert!(approx(value_noise_3d(p, 9), value_noise_3d(p, 9)));
        // A different seed generally gives a different value here.
        assert!(!approx(value_noise_3d(p, 9), value_noise_3d(p, 10)));
    }

    #[test]
    fn value_noise_stays_in_unit_range() {
        let mut idx = 0.0_f32;
        for _ in 0..600 {
            idx += 1.0;
            let p = Vec3::new(idx * 0.13, idx * -0.27, idx * 0.41);
            let n = value_noise_3d(p, 123);
            assert!((-1.0..=1.0).contains(&n), "value noise out of range: {n}");
        }
    }

    #[test]
    fn value_noise_matches_hashed_corner_at_integer_cells() {
        // At an integer lattice point trilinear weights collapse to the corner.
        let expected = cell_value(4, -1, 2, 55);
        let sampled = value_noise_3d(Vec3::new(4.0, -1.0, 2.0), 55);
        assert!(approx(sampled, expected));
    }

    #[test]
    fn gradient_noise_is_zero_at_integer_lattice() {
        for i in -2..3 {
            for k in -2..3 {
                let n = gradient_noise_3d(Vec3::new(i as f32, 1.0, k as f32), 3);
                assert!(approx(n, 0.0), "Perlin noise must vanish on the lattice");
            }
        }
    }

    #[test]
    fn gradient_noise_stays_in_unit_range() {
        let mut idx = 0.0_f32;
        for _ in 0..800 {
            idx += 1.0;
            let p = Vec3::new(idx * 0.077, idx * 0.131, idx * -0.219);
            let n = gradient_noise_3d(p, 77);
            assert!(
                (-1.0..=1.0).contains(&n),
                "gradient noise out of range: {n}"
            );
        }
    }

    #[test]
    fn gradient_noise_is_reproducible_and_continuous() {
        let p = Vec3::new(0.37, 1.11, -0.62);
        assert!(approx(gradient_noise_3d(p, 5), gradient_noise_3d(p, 5)));
        // A small spatial step changes the value only a little (continuity).
        let q = Vec3::new(p.x + 1.0e-3, p.y, p.z);
        assert!((gradient_noise_3d(p, 5) - gradient_noise_3d(q, 5)).abs() < 1.0e-2);
    }

    #[test]
    fn fbm_single_octave_equals_gradient_noise() {
        let p = Vec3::new(0.4, -1.7, 2.9);
        let single = fbm(p, FbmParams::SINGLE_OCTAVE, 13);
        assert!(approx(single, gradient_noise_3d(p, 13)));
    }

    #[test]
    fn fbm_zero_gain_collapses_to_first_octave() {
        let p = Vec3::new(0.4, -1.7, 2.9);
        let params = FbmParams::new(4, 2.0, 0.0);
        // With gain 0 the later octaves contribute nothing.
        assert!(approx(fbm(p, params, 13), gradient_noise_3d(p, 13)));
    }

    #[test]
    fn fbm_zero_octaves_is_zero_not_nan() {
        let p = Vec3::new(0.4, -1.7, 2.9);
        let n = fbm(p, FbmParams::new(0, 2.0, 0.5), 13);
        assert!(approx(n, 0.0));
        assert!(n.is_finite());
    }

    #[test]
    fn fbm_stays_in_unit_range_and_adds_detail() {
        let p = Vec3::new(0.63, 1.29, -0.44);
        let single = fbm(p, FbmParams::SINGLE_OCTAVE, 21);
        let multi = fbm(p, FbmParams::DEFAULT, 21);
        assert!((-1.0..=1.0).contains(&multi));
        // More octaves generally change the value versus a single octave.
        assert!(!approx(single, multi));
    }

    #[test]
    fn fbm_is_reproducible() {
        let p = Vec3::new(2.1, 0.5, -3.3);
        assert!(approx(
            fbm(p, FbmParams::DEFAULT, 99),
            fbm(p, FbmParams::DEFAULT, 99)
        ));
    }

    #[test]
    fn curl_noise_is_reproducible_and_nonzero() {
        let p = Vec3::new(0.33, 1.42, -0.71);
        let a = curl_noise_3d(p, 8);
        let b = curl_noise_3d(p, 8);
        assert_eq!(a, b);
        // The field should actually move particles somewhere off the lattice.
        assert!(a.length() > 1.0e-4, "curl noise should be non-trivial");
    }

    #[test]
    fn curl_noise_is_approximately_divergence_free() {
        // Numerically estimate div(curl) at several off-lattice points; it must
        // be tiny compared with the field magnitude (the analytic value is 0).
        let step = CURL_EPS;
        let inv = 1.0 / (2.0 * step);
        let samples = [
            Vec3::new(0.31, 0.62, 0.93),
            Vec3::new(-1.27, 0.48, 2.15),
            Vec3::new(3.04, -2.11, 0.57),
            Vec3::new(-0.86, -1.53, -2.42),
        ];
        for p in samples {
            let dx = curl_noise_3d(p.add(Vec3::new(step, 0.0, 0.0)), 4)
                .sub(curl_noise_3d(p.sub(Vec3::new(step, 0.0, 0.0)), 4))
                .scale(inv);
            let dy = curl_noise_3d(p.add(Vec3::new(0.0, step, 0.0)), 4)
                .sub(curl_noise_3d(p.sub(Vec3::new(0.0, step, 0.0)), 4))
                .scale(inv);
            let dz = curl_noise_3d(p.add(Vec3::new(0.0, 0.0, step)), 4)
                .sub(curl_noise_3d(p.sub(Vec3::new(0.0, 0.0, step)), 4))
                .scale(inv);
            let divergence = dx.x + dy.y + dz.z;
            assert!(
                divergence.abs() < 1.0e-2,
                "divergence {divergence} should be ~0 for curl noise"
            );
        }
    }

    #[test]
    fn turbulence_force_scales_linearly_with_amplitude() {
        let p = Vec3::new(0.9, -0.4, 1.8);
        let base = TurbulenceParams {
            frequency: 1.0,
            amplitude: 1.0,
            fbm: FbmParams::DEFAULT,
            seed: 17,
        };
        let doubled = TurbulenceParams {
            amplitude: 2.0,
            ..base
        };
        let f1 = turbulence_force(p, base);
        let f2 = turbulence_force(p, doubled);
        assert!(approx(f2.x, f1.x * 2.0));
        assert!(approx(f2.y, f1.y * 2.0));
        assert!(approx(f2.z, f1.z * 2.0));
    }

    #[test]
    fn turbulence_force_zero_amplitude_is_zero() {
        let p = Vec3::new(0.9, -0.4, 1.8);
        let params = TurbulenceParams {
            frequency: 2.0,
            amplitude: 0.0,
            fbm: FbmParams::DEFAULT,
            seed: 17,
        };
        assert_eq!(turbulence_force(p, params), Vec3::ZERO);
    }

    #[test]
    fn turbulence_force_is_reproducible() {
        let p = Vec3::new(1.3, 2.6, -0.9);
        let params = TurbulenceParams::DEFAULT;
        assert_eq!(turbulence_force(p, params), turbulence_force(p, params));
    }

    #[test]
    fn default_params_are_canonical() {
        assert_eq!(FbmParams::default(), FbmParams::DEFAULT);
        assert_eq!(FbmParams::DEFAULT.octaves, 4);
        assert!(approx(FbmParams::DEFAULT.lacunarity, 2.0));
        assert!(approx(FbmParams::DEFAULT.gain, 0.5));
        assert_eq!(TurbulenceParams::default(), TurbulenceParams::DEFAULT);
    }
}
