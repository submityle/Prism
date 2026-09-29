//! Deterministic `Perlin` / `Worley` / `curl` noise primitives for the
//! volumetric cloud subsystem (design section 4, `Nubis`-style procedural
//! modelling).
//!
//! The cloud density field is built from a low-frequency `Perlin-Worley` base
//! shape (`Perlin` continuity plus `Worley` clumping), high-frequency `Worley`
//! detail for edge erosion, and a divergence-free `curl` field for wispy
//! advection. Every primitive here is a pure function of its arguments: a
//! fixed integer `hash` (`FNV-1a` word mixing plus an avalanche finalizer)
//! seeds all gradients and feature points, so the same input always yields the
//! same output, nothing depends on wall-clock time or a random generator, and
//! out-of-range coordinates never panic.
//!
//! The determinism policy of this crate allows only `sqrt` among the float
//! intrinsics, so this module uses no `exp` / `pow` / `ln` / `sin` / `cos`:
//! the noise math is add / multiply / `sqrt` / integer-bit work plus the
//! shared helpers ([`lerp`], [`remap`], [`saturate`]) re-exported from the
//! sibling `math` module. The `GPU` `WESL` kernels evaluate the identical
//! algorithm with native intrinsics; this `CPU` reference exists so the
//! numeric properties (`[0, 1]` value range, `Worley` non-negativity, spatial
//! continuity, `curl` divergence-freedom) can be unit-tested in the sandbox
//! where there is no `GPU`.

use super::math::{lerp, remap, saturate, Vec3, EPS};

/// Offset basis of the 32-bit `FNV-1a` hash (the standard constant).
const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;

/// Prime multiplier of the 32-bit `FNV-1a` hash (the standard constant).
const FNV_PRIME: u32 = 0x0100_0193;

/// Decorrelation seed for the `Worley` feature point's `y` offset.
const WORLEY_SEED_Y: u32 = 0x68e3_1da4;

/// Decorrelation seed for the `Worley` feature point's `z` offset.
const WORLEY_SEED_Z: u32 = 0xb527_9a45;

/// Per-octave seed stride so stacked octaves of `fBm` decorrelate.
const FBM_SEED_STRIDE: u32 = 0x9e37_79b9;

/// Default octave count for the `Perlin` `fBm` used by the base cloud shape.
pub const DEFAULT_FBM_OCTAVES: u32 = 5;

/// Default octave count for the `Worley` `fBm` used by the clump billow.
pub const DEFAULT_WORLEY_OCTAVES: u32 = 3;

/// Default frequency multiplier between successive `fBm` octaves.
pub const DEFAULT_LACUNARITY: f32 = 2.0;

/// Default amplitude decay between successive `fBm` octaves.
pub const DEFAULT_GAIN: f32 = 0.5;

/// Relative `Worley` frequency inside [`perlin_worley`] so the clump billow is
/// finer than the `Perlin` base it modulates.
const PERLIN_WORLEY_WORLEY_FREQ: f32 = 2.0;

/// Seed decorrelation between the `Perlin` and `Worley` fields of the
/// `Perlin-Worley` blend.
const PERLIN_WORLEY_SEED_MIX: u32 = 0x27d4_eb2f;

/// Central-difference step used to take the `curl` of the vector potential.
///
/// The same step must be reused when numerically estimating the divergence of
/// the resulting field: with matched central differences the mixed second
/// derivatives cancel identically, so the field is divergence-free up to
/// floating-point rounding only (asserted in the tests).
pub const CURL_EPS: f32 = 0.02;

/// Spatial offset decorrelating the second potential component from the first.
const CURL_OFFSET_G: Vec3 = Vec3::new(31.416, 47.853, 12.793);

/// Spatial offset decorrelating the third potential component from the first.
const CURL_OFFSET_B: Vec3 = Vec3::new(-53.217, 19.271, 83.155);

/// Seed decorrelating the second potential component from the first.
const CURL_SEED_G: u32 = 0x1b56_c4e9;

/// Seed decorrelating the third potential component from the first.
const CURL_SEED_B: u32 = 0x7a4f_3c1d;

/// Improved-`Perlin` gradient set: the twelve cube-edge midpoints padded to
/// sixteen entries (four repeats) so a gradient is selected with a cheap
/// four-bit mask. Every gradient has squared length `2`, keeping the raw noise
/// bounded to roughly `[-1, 1]`.
const GRAD3: [Vec3; 16] = [
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(-1.0, 1.0, 0.0),
    Vec3::new(1.0, -1.0, 0.0),
    Vec3::new(-1.0, -1.0, 0.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(-1.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, -1.0),
    Vec3::new(-1.0, 0.0, -1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(0.0, -1.0, 1.0),
    Vec3::new(0.0, 1.0, -1.0),
    Vec3::new(0.0, -1.0, -1.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(0.0, -1.0, 1.0),
    Vec3::new(-1.0, 1.0, 0.0),
    Vec3::new(0.0, -1.0, -1.0),
];

/// Mixes one 32-bit `word` into the running `FNV-1a` `hash`, byte by byte.
#[must_use]
fn fnv_word(mut hash: u32, word: u32) -> u32 {
    let mut shift = 0u32;
    while shift < 32 {
        hash ^= (word >> shift) & 0xFF;
        hash = hash.wrapping_mul(FNV_PRIME);
        shift += 8;
    }
    hash
}

/// Deterministic integer `hash` of an integer lattice point and a `seed`.
///
/// Mixes `seed` and the three coordinates through `FNV-1a`, then applies an
/// `xorshift`-multiply avalanche so neighbouring cells map to well-dispersed
/// bit patterns. Pure integer work, so it is exactly reproducible.
#[must_use]
fn hash_cell(ix: i32, iy: i32, iz: i32, seed: u32) -> u32 {
    let mut h = FNV_OFFSET_BASIS;
    h = fnv_word(h, seed);
    h = fnv_word(h, ix as u32);
    h = fnv_word(h, iy as u32);
    h = fnv_word(h, iz as u32);
    // Avalanche finalizer for good bit dispersion across adjacent cells.
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

/// Maps a 32-bit `hash` to a value in `[0, 1)` using its top twenty-four bits.
#[must_use]
fn hash_to_unit(h: u32) -> f32 {
    // 2^24 == 16_777_216; the top bits carry the best avalanche quality.
    ((h >> 8) as f32) * (1.0 / 16_777_216.0)
}

/// Selects a lattice gradient from [`GRAD3`] using the low four bits of `h`.
#[must_use]
fn gradient(h: u32) -> Vec3 {
    GRAD3[(h & 15) as usize]
}

/// Dot product of the gradient at lattice point `(ix, iy, iz)` with the
/// distance vector `(dx, dy, dz)` from that point to the sample.
#[must_use]
fn grad_dot(ix: i32, iy: i32, iz: i32, dx: f32, dy: f32, dz: f32, seed: u32) -> f32 {
    let g = gradient(hash_cell(ix, iy, iz, seed));
    g.x * dx + g.y * dy + g.z * dz
}

/// Quintic fade `6t^5 - 15t^4 + 10t^3`, the `C2`-continuous interpolant
/// `Perlin` uses in place of a linear or `smoothstep` blend so the noise (and
/// its first and second derivatives) stays continuous across cell boundaries.
#[must_use]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Raw signed gradient noise in roughly `[-1, 1]`; the un-normalized core of
/// [`perlin_3d`], reused directly by the `curl` potential.
#[must_use]
fn perlin_raw(p: Vec3, seed: u32) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let x0 = xi as i32;
    let y0 = yi as i32;
    let z0 = zi as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let z1 = z0 + 1;
    let fx = p.x - xi;
    let fy = p.y - yi;
    let fz = p.z - zi;
    let u = fade(fx);
    let v = fade(fy);
    let w = fade(fz);
    let n000 = grad_dot(x0, y0, z0, fx, fy, fz, seed);
    let n100 = grad_dot(x1, y0, z0, fx - 1.0, fy, fz, seed);
    let n010 = grad_dot(x0, y1, z0, fx, fy - 1.0, fz, seed);
    let n110 = grad_dot(x1, y1, z0, fx - 1.0, fy - 1.0, fz, seed);
    let n001 = grad_dot(x0, y0, z1, fx, fy, fz - 1.0, seed);
    let n101 = grad_dot(x1, y0, z1, fx - 1.0, fy, fz - 1.0, seed);
    let n011 = grad_dot(x0, y1, z1, fx, fy - 1.0, fz - 1.0, seed);
    let n111 = grad_dot(x1, y1, z1, fx - 1.0, fy - 1.0, fz - 1.0, seed);
    let x00 = lerp(n000, n100, u);
    let x10 = lerp(n010, n110, u);
    let x01 = lerp(n001, n101, u);
    let x11 = lerp(n011, n111, u);
    let y00 = lerp(x00, x10, v);
    let y11 = lerp(x01, x11, v);
    lerp(y00, y11, w)
}

/// `Perlin` gradient noise normalized to `[0, 1]`.
///
/// Uses a quintic fade for `C2` spatial continuity; the raw signed noise is
/// mapped by `0.5 + 0.5 * raw` and saturated as a numeric guard (the raw range
/// stays inside `[-1, 1]`, so saturation never clips in practice). Identical
/// inputs always produce identical output.
#[must_use]
pub fn perlin_3d(p: Vec3, seed: u32) -> f32 {
    saturate(0.5 + 0.5 * perlin_raw(p, seed))
}

/// `Worley` (cellular) noise: the Euclidean distance to the nearest feature
/// point, mapped to `[0, 1]`.
///
/// Each lattice cell holds exactly one feature point jittered inside its unit
/// cube by the deterministic `hash`. The `3x3x3` neighbourhood around the
/// sample cell is searched, which is sufficient to find the true nearest
/// point. The result is a distance and therefore always non-negative; it is
/// saturated so the returned value stays in `[0, 1]`.
#[must_use]
pub fn worley_3d(p: Vec3, seed: u32) -> f32 {
    let xi = p.x.floor();
    let yi = p.y.floor();
    let zi = p.z.floor();
    let x0 = xi as i32;
    let y0 = yi as i32;
    let z0 = zi as i32;
    let fx = p.x - xi;
    let fy = p.y - yi;
    let fz = p.z - zi;
    let mut min_sq = f32::INFINITY;
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let cx = x0 + dx;
                let cy = y0 + dy;
                let cz = z0 + dz;
                let h = hash_cell(cx, cy, cz, seed);
                let ox = hash_to_unit(h);
                let oy = hash_to_unit(hash_cell(cx, cy, cz, seed ^ WORLEY_SEED_Y));
                let oz = hash_to_unit(hash_cell(cx, cy, cz, seed ^ WORLEY_SEED_Z));
                let fpx = dx as f32 + ox - fx;
                let fpy = dy as f32 + oy - fy;
                let fpz = dz as f32 + oz - fz;
                let d2 = fpx * fpx + fpy * fpy + fpz * fpz;
                if d2 < min_sq {
                    min_sq = d2;
                }
            }
        }
    }
    saturate(min_sq.sqrt())
}

/// Inverted `Worley` noise `1 - worley_3d`, which is high (near `1`) inside a
/// cell's feature-point core and low at cell edges: the billowy clump form the
/// cloud modelling and `detail erosion` want.
#[must_use]
pub fn worley_inverted(p: Vec3, seed: u32) -> f32 {
    saturate(1.0 - worley_3d(p, seed))
}

/// Fractal `Perlin` noise (`fBm`): `octaves` of [`perlin_3d`] summed with
/// geometric frequency growth ([`DEFAULT_LACUNARITY`]) and amplitude decay
/// ([`DEFAULT_GAIN`]).
///
/// Each octave lies in `[0, 1]` and the amplitudes are positive, so the
/// amplitude-normalized sum is itself in `[0, 1]`. Passing `octaves == 0`
/// yields `0` rather than dividing by zero.
#[must_use]
pub fn fbm(p: Vec3, seed: u32, octaves: u32) -> f32 {
    let mut freq = 1.0;
    let mut amp = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut i = 0u32;
    while i < octaves {
        let octave_seed = seed.wrapping_add(i.wrapping_mul(FBM_SEED_STRIDE));
        sum += amp * perlin_3d(p.scale(freq), octave_seed);
        norm += amp;
        freq *= DEFAULT_LACUNARITY;
        amp *= DEFAULT_GAIN;
        i += 1;
    }
    if norm > EPS {
        saturate(sum / norm)
    } else {
        0.0
    }
}

/// Fractal inverted-`Worley` noise (billowy `fBm`) in `[0, 1]`.
///
/// Stacks `octaves` of [`worley_inverted`] with the shared lacunarity/gain so
/// the clump form gains detail while staying amplitude-normalized to `[0, 1]`.
/// Passing `octaves == 0` yields `0`.
#[must_use]
pub fn worley_fbm(p: Vec3, seed: u32, octaves: u32) -> f32 {
    let mut freq = 1.0;
    let mut amp = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut i = 0u32;
    while i < octaves {
        let octave_seed = seed.wrapping_add(i.wrapping_mul(FBM_SEED_STRIDE));
        sum += amp * worley_inverted(p.scale(freq), octave_seed);
        norm += amp;
        freq *= DEFAULT_LACUNARITY;
        amp *= DEFAULT_GAIN;
        i += 1;
    }
    if norm > EPS {
        saturate(sum / norm)
    } else {
        0.0
    }
}

/// `Perlin-Worley` base cloud noise in `[0, 1]` (`Nubis`-style).
///
/// Combines a continuous `Perlin` `fBm` base with a billowy `Worley` `fBm`
/// clump via an energy-preserving `remap` rather than an additive blend: where
/// the `Worley` billow is strong the `Perlin` field is boosted (clump cores),
/// and where it is weak the base is preserved. The result is saturated to keep
/// the value in `[0, 1]`, and is fully deterministic in `p` and `seed`.
#[must_use]
pub fn perlin_worley(p: Vec3, seed: u32) -> f32 {
    let perlin = fbm(p, seed, DEFAULT_FBM_OCTAVES);
    let billow = worley_fbm(
        p.scale(PERLIN_WORLEY_WORLEY_FREQ),
        seed ^ PERLIN_WORLEY_SEED_MIX,
        DEFAULT_WORLEY_OCTAVES,
    );
    saturate(remap(perlin, billow - 1.0, 1.0, 0.0, 1.0))
}

/// Three-component vector potential whose `curl` yields the noise flow field.
///
/// Each component is an independent signed `Perlin` field, decorrelated by a
/// spatial offset and a seed so the resulting `curl` does not collapse to a
/// degenerate direction.
#[must_use]
fn curl_potential(p: Vec3, seed: u32) -> Vec3 {
    Vec3::new(
        perlin_raw(p, seed),
        perlin_raw(p.add(CURL_OFFSET_G), seed ^ CURL_SEED_G),
        perlin_raw(p.add(CURL_OFFSET_B), seed ^ CURL_SEED_B),
    )
}

/// Divergence-free `curl` noise for wispy / vortical advection (cirrus streaks,
/// wind disturbance, contrail curl).
///
/// Takes the analytic `curl` of [`curl_potential`] with second-order central
/// differences at the fixed step [`CURL_EPS`]. The `curl` of any vector field
/// is divergence-free; with matched central differences the discrete field is
/// divergence-free up to floating-point rounding, which the tests assert.
#[must_use]
pub fn curl_noise_3d(p: Vec3, seed: u32) -> Vec3 {
    let e = CURL_EPS;
    let inv = 1.0 / (2.0 * e);
    let px1 = curl_potential(p.add(Vec3::new(e, 0.0, 0.0)), seed);
    let px0 = curl_potential(p.sub(Vec3::new(e, 0.0, 0.0)), seed);
    let py1 = curl_potential(p.add(Vec3::new(0.0, e, 0.0)), seed);
    let py0 = curl_potential(p.sub(Vec3::new(0.0, e, 0.0)), seed);
    let pz1 = curl_potential(p.add(Vec3::new(0.0, 0.0, e)), seed);
    let pz0 = curl_potential(p.sub(Vec3::new(0.0, 0.0, e)), seed);
    let dpz_dy = (py1.z - py0.z) * inv;
    let dpy_dz = (pz1.y - pz0.y) * inv;
    let dpx_dz = (pz1.x - pz0.x) * inv;
    let dpz_dx = (px1.z - px0.z) * inv;
    let dpy_dx = (px1.y - px0.y) * inv;
    let dpx_dy = (py1.x - py0.x) * inv;
    Vec3::new(dpz_dy - dpy_dz, dpx_dz - dpz_dx, dpy_dx - dpx_dy)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed off-lattice sample points used across the property tests; chosen
    /// away from integer coordinates so gradient noise is non-degenerate.
    const SAMPLES: [Vec3; 6] = [
        Vec3::new(0.37, 1.21, -2.13),
        Vec3::new(-4.51, 3.09, 0.77),
        Vec3::new(12.34, -5.68, 9.01),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-100.25, 42.75, 7.33),
        Vec3::new(3.123, -2.654, 1.789),
    ];

    /// `true` when `a` and `b` agree within `tol` absolute error.
    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn hash_is_deterministic_and_disperses() {
        // Same input -> identical bits; neighbouring cells differ.
        assert_eq!(hash_cell(1, 2, 3, 7), hash_cell(1, 2, 3, 7));
        assert_ne!(hash_cell(1, 2, 3, 7), hash_cell(2, 2, 3, 7));
        assert_ne!(hash_cell(1, 2, 3, 7), hash_cell(1, 2, 3, 8));
        // Unit mapping stays in [0, 1).
        for &s in &SAMPLES {
            let u = hash_to_unit(hash_cell(s.x as i32, s.y as i32, s.z as i32, 1));
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn noise_is_deterministic() {
        for &p in &SAMPLES {
            assert_eq!(perlin_3d(p, 5).to_bits(), perlin_3d(p, 5).to_bits());
            assert_eq!(worley_3d(p, 5).to_bits(), worley_3d(p, 5).to_bits());
            assert_eq!(perlin_worley(p, 5).to_bits(), perlin_worley(p, 5).to_bits());
            assert_eq!(
                fbm(p, 5, DEFAULT_FBM_OCTAVES).to_bits(),
                fbm(p, 5, DEFAULT_FBM_OCTAVES).to_bits()
            );
            let a = curl_noise_3d(p, 5);
            let b = curl_noise_3d(p, 5);
            assert_eq!(a.x.to_bits(), b.x.to_bits());
            assert_eq!(a.y.to_bits(), b.y.to_bits());
            assert_eq!(a.z.to_bits(), b.z.to_bits());
        }
    }

    #[test]
    fn values_stay_in_unit_range() {
        for &p in &SAMPLES {
            let perlin = perlin_3d(p, 11);
            let worley = worley_3d(p, 11);
            let pw = perlin_worley(p, 11);
            let f = fbm(p, 11, DEFAULT_FBM_OCTAVES);
            let wf = worley_fbm(p, 11, DEFAULT_WORLEY_OCTAVES);
            assert!(
                (0.0..=1.0).contains(&perlin),
                "perlin out of range: {perlin}"
            );
            assert!(
                (0.0..=1.0).contains(&worley),
                "worley out of range: {worley}"
            );
            assert!(
                (0.0..=1.0).contains(&pw),
                "perlin_worley out of range: {pw}"
            );
            assert!((0.0..=1.0).contains(&f), "fbm out of range: {f}");
            assert!((0.0..=1.0).contains(&wf), "worley_fbm out of range: {wf}");
        }
    }

    #[test]
    fn worley_is_non_negative() {
        // Distance-based, so never negative regardless of seed or position.
        for seed in 0u32..8 {
            for &p in &SAMPLES {
                assert!(worley_3d(p, seed) >= 0.0);
                assert!(worley_inverted(p, seed) >= 0.0);
            }
        }
    }

    #[test]
    fn zero_octaves_do_not_divide_by_zero() {
        for &p in &SAMPLES {
            assert_eq!(fbm(p, 3, 0), 0.0);
            assert_eq!(worley_fbm(p, 3, 0), 0.0);
        }
    }

    #[test]
    fn perlin_is_spatially_continuous() {
        // A small positional step must produce a small change in value.
        let step = 0.001;
        for &p in &SAMPLES {
            let base = perlin_3d(p, 21);
            for axis in [
                Vec3::new(step, 0.0, 0.0),
                Vec3::new(0.0, step, 0.0),
                Vec3::new(0.0, 0.0, step),
            ] {
                let shifted = perlin_3d(p.add(axis), 21);
                assert!(
                    close(base, shifted, 0.02),
                    "perlin discontinuous: {base} vs {shifted}"
                );
            }
        }
    }

    #[test]
    fn different_seeds_produce_different_values() {
        // At least one sample must differ meaningfully between two seeds.
        let mut differed = false;
        for &p in &SAMPLES {
            if !close(perlin_3d(p, 1), perlin_3d(p, 2), 1e-4) {
                differed = true;
            }
        }
        assert!(differed, "seed change had no effect on perlin_3d");

        let mut worley_differed = false;
        for &p in &SAMPLES {
            if !close(worley_3d(p, 1), worley_3d(p, 2), 1e-4) {
                worley_differed = true;
            }
        }
        assert!(worley_differed, "seed change had no effect on worley_3d");
    }

    #[test]
    fn curl_field_is_approximately_divergence_free() {
        // Estimate div(curl) with the SAME step as curl_noise_3d uses: the
        // mixed second differences cancel analytically, so only rounding
        // remains.
        let e = CURL_EPS;
        let inv = 1.0 / (2.0 * e);
        for &p in &SAMPLES {
            let cx1 = curl_noise_3d(p.add(Vec3::new(e, 0.0, 0.0)), 33);
            let cx0 = curl_noise_3d(p.sub(Vec3::new(e, 0.0, 0.0)), 33);
            let cy1 = curl_noise_3d(p.add(Vec3::new(0.0, e, 0.0)), 33);
            let cy0 = curl_noise_3d(p.sub(Vec3::new(0.0, e, 0.0)), 33);
            let cz1 = curl_noise_3d(p.add(Vec3::new(0.0, 0.0, e)), 33);
            let cz0 = curl_noise_3d(p.sub(Vec3::new(0.0, 0.0, e)), 33);
            let divergence = ((cx1.x - cx0.x) + (cy1.y - cy0.y) + (cz1.z - cz0.z)) * inv;
            assert!(
                divergence.abs() < 1e-2,
                "curl divergence too large: {divergence}"
            );
        }
    }

    #[test]
    fn curl_is_finite_and_nonzero_somewhere() {
        let mut any_flow = false;
        for &p in &SAMPLES {
            let c = curl_noise_3d(p, 44);
            assert!(c.x.is_finite() && c.y.is_finite() && c.z.is_finite());
            if c.length_squared() > 1e-6 {
                any_flow = true;
            }
        }
        assert!(any_flow, "curl field was degenerate everywhere");
    }
}
