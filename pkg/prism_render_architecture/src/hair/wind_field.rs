//! Curl-noise divergence-free wind field for groom motion (Bridson 2007).
//!
//! A believable groom under wind must gust and swirl without the hair ever
//! appearing to inflate or collapse: visually, strands should be pushed around
//! by a *turbulent but volume-preserving* flow. The physical statement of
//! "volume-preserving" for a velocity field is that it be **divergence-free**
//! (`div w = 0`), i.e. incompressible. Solving an actual incompressible fluid
//! for a groom is wasteful, so film and realtime grooms instead borrow Bridson's
//! `curl-noise` trick: take the **curl** of a smooth vector potential field `P`,
//! `w = curl P`. Because the divergence of any curl is identically zero
//! (`div(curl P) = 0`), the resulting field is divergence-free *by
//! construction*, with no solve and no grid -- just evaluate the potential's
//! derivatives at a point. This is the approach popularised by `Bridson`,
//! `Hoffman` and Houdini-style turbulence, and it is what grooms in `UE5` and
//! offline pipelines use for cheap, art-directable wind.
//!
//! The potential field `P = (Px, Py, Pz)` is built from a deterministic integer
//! **value noise**: a `u32` hash of the integer lattice cell seeds a
//! pseudo-random value at each corner, and a cubic `smoothstep`
//! (`t*t*(3 - 2t)`) fades a trilinear interpolation between corners. Three
//! independent noise channels (seeded with different offsets) give the three
//! potential components, and the wind velocity is the curl of that potential,
//! approximated with central **finite differences** of step [`EPS`]. Time is
//! injected as a fourth coordinate offset so the field evolves smoothly rather
//! than jumping between frames.
//!
//! Like the rest of this architecture crate the module is pure and
//! deterministic (fixed seed in, fixed field out; array in, array out;
//! `golden`-comparable) and panic-free on degenerate input (negative /
//! non-finite `frequency` or `amplitude` are sanitised to safe values). It uses
//! **no** transcendental math -- value noise is polynomial `smoothstep` over a
//! hashed lattice and the curl is finite differences -- so it needs no `libm`
//! determinism shim. The shading / simulation side consumes the returned
//! velocity as a per-point external force on each strand vertex.

use alloc::vec::Vec;

/// Finite-difference step used to approximate the potential's partial
/// derivatives when forming the curl. Small enough that the central difference
/// closely tracks the analytic derivative of the smooth value noise, large
/// enough to stay well clear of `f32` cancellation.
const EPS: f32 = 1.0e-3;

/// Controls for the [`curl_wind`] field.
///
/// `frequency` scales world position into noise space (higher = finer, busier
/// gusts), `amplitude` scales the output velocity magnitude, and `seed` makes
/// the whole field deterministic and reproducible. Negative or non-finite
/// `frequency` / `amplitude` are unphysical and are repaired by
/// [`WindFieldParams::sanitized`] before use; they never panic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindFieldParams {
    /// World-to-noise spatial scale; sanitised to a finite positive value.
    pub frequency: f32,
    /// Output velocity scale; sanitised to a finite non-negative value.
    pub amplitude: f32,
    /// Deterministic field seed.
    pub seed: u32,
}

impl Default for WindFieldParams {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            amplitude: 1.0,
            seed: 0,
        }
    }
}

impl WindFieldParams {
    /// Builds params from explicit fields.
    #[must_use]
    pub const fn new(frequency: f32, amplitude: f32, seed: u32) -> Self {
        Self {
            frequency,
            amplitude,
            seed,
        }
    }

    /// This parameter set with `frequency` / `amplitude` repaired to safe
    /// values. A non-finite or non-positive `frequency` falls back to `1.0` (a
    /// zero frequency would collapse the field to a constant); a non-finite or
    /// negative `amplitude` falls back to `0.0` (no wind). The result is always
    /// finite.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let frequency = if self.frequency.is_finite() && self.frequency > 0.0 {
            self.frequency
        } else {
            1.0
        };
        let amplitude = if self.amplitude.is_finite() && self.amplitude >= 0.0 {
            self.amplitude
        } else {
            0.0
        };
        Self {
            frequency,
            amplitude,
            seed: self.seed,
        }
    }
}

/// Mixes a `u32` into a well-distributed pseudo-random `u32` using only integer
/// bit operations (an `xorshift`-style finaliser). Deterministic and
/// branch-free; this is the only randomness source in the module, so the field
/// never depends on floating-point `rng`.
#[must_use]
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Hashes an integer lattice cell plus a seed into a pseudo-random value in
/// `[-1, 1]`. The three coordinates are folded together with large odd
/// multipliers so neighbouring cells decorrelate, then normalised from the full
/// `u32` range.
#[must_use]
fn lattice_value(seed: u32, ix: i32, iy: i32, iz: i32) -> f32 {
    let mut h = seed;
    h = h.wrapping_add((ix as u32).wrapping_mul(0x9e37_79b1));
    h = hash_u32(h);
    h = h.wrapping_add((iy as u32).wrapping_mul(0x8521_2bff));
    h = hash_u32(h);
    h = h.wrapping_add((iz as u32).wrapping_mul(0x6b43_a9b5));
    h = hash_u32(h);
    // Map the full u32 range to [0, 1), then to [-1, 1).
    let unit = (h as f32) / (u32::MAX as f32);
    unit * 2.0 - 1.0
}

/// Cubic `smoothstep` fade `t*t*(3 - 2t)` on an already-`0..=1` fraction. Gives
/// the value noise a continuous first derivative, which is what keeps the
/// finite-difference curl smooth. Written as plain multiplies (no `powi`).
#[must_use]
fn smooth_fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation `a + (b - a) * t`.
#[must_use]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Deterministic trilinear value noise sampled at `p`, returning a value in
/// `[-1, 1]`.
///
/// The point is split into an integer lattice cell and a fractional offset; the
/// eight corner values are hashed from the cell indices and the seed, then
/// trilinearly blended using the cubic [`smooth_fade`] of the fractional offset.
/// This is the per-channel scalar potential sampled by [`curl_wind`]; it is also
/// exposed so tests (and goldens) can pin individual values.
#[must_use]
pub fn value_noise(seed: u32, p: [f32; 3]) -> f32 {
    let xi = p[0].floor();
    let yi = p[1].floor();
    let zi = p[2].floor();

    let ix = xi as i32;
    let iy = yi as i32;
    let iz = zi as i32;

    let fx = p[0] - xi;
    let fy = p[1] - yi;
    let fz = p[2] - zi;

    let ux = smooth_fade(fx);
    let uy = smooth_fade(fy);
    let uz = smooth_fade(fz);

    let c000 = lattice_value(seed, ix, iy, iz);
    let c100 = lattice_value(seed, ix + 1, iy, iz);
    let c010 = lattice_value(seed, ix, iy + 1, iz);
    let c110 = lattice_value(seed, ix + 1, iy + 1, iz);
    let c001 = lattice_value(seed, ix, iy, iz + 1);
    let c101 = lattice_value(seed, ix + 1, iy, iz + 1);
    let c011 = lattice_value(seed, ix, iy + 1, iz + 1);
    let c111 = lattice_value(seed, ix + 1, iy + 1, iz + 1);

    let x00 = lerp(c000, c100, ux);
    let x10 = lerp(c010, c110, ux);
    let x01 = lerp(c001, c101, ux);
    let x11 = lerp(c011, c111, ux);

    let y0 = lerp(x00, x10, uy);
    let y1 = lerp(x01, x11, uy);

    lerp(y0, y1, uz)
}

/// Evaluates the scalar potential channel `channel` (`0`, `1`, or `2`) at a
/// noise-space point. Each channel uses a different seed offset so the three
/// potential components are independent, and `time` shifts the sample so the
/// field animates smoothly.
#[must_use]
fn potential_channel(seed: u32, channel: u32, p: [f32; 3], time: f32) -> f32 {
    // Distinct large offsets per channel keep the three potential components
    // decorrelated; time is folded into each coordinate so motion is smooth.
    let channel_seed = seed.wrapping_add(channel.wrapping_mul(0x1000_0001));
    let t = time + (channel as f32) * 13.0;
    value_noise(channel_seed, [p[0] + t, p[1] - t, p[2] + t])
}

/// Divergence-free wind velocity at a world position and time.
///
/// Scales `pos` by `frequency` into noise space, samples the vector potential
/// `P` there, and returns `amplitude * curl(P)`, where the curl is approximated
/// by central finite differences of step [`EPS`]. Because the divergence of a
/// curl is identically zero, the field is divergence-free by construction. The
/// result is finite for every input: non-finite / negative params are sanitised
/// first, so the function never panics.
#[must_use]
pub fn curl_wind(params: WindFieldParams, pos: [f32; 3], time: f32) -> [f32; 3] {
    let clean = params.sanitized();
    let freq = clean.frequency;

    // World position in noise space.
    let p = [pos[0] * freq, pos[1] * freq, pos[2] * freq];

    let inv_two_eps = 1.0 / (2.0 * EPS);

    // Central differences of each potential component along each axis.
    let px_py_hi = potential_channel(clean.seed, 0, [p[0], p[1] + EPS, p[2]], time);
    let px_py_lo = potential_channel(clean.seed, 0, [p[0], p[1] - EPS, p[2]], time);
    let px_pz_hi = potential_channel(clean.seed, 0, [p[0], p[1], p[2] + EPS], time);
    let px_pz_lo = potential_channel(clean.seed, 0, [p[0], p[1], p[2] - EPS], time);

    let py_px_hi = potential_channel(clean.seed, 1, [p[0] + EPS, p[1], p[2]], time);
    let py_px_lo = potential_channel(clean.seed, 1, [p[0] - EPS, p[1], p[2]], time);
    let py_pz_hi = potential_channel(clean.seed, 1, [p[0], p[1], p[2] + EPS], time);
    let py_pz_lo = potential_channel(clean.seed, 1, [p[0], p[1], p[2] - EPS], time);

    let pz_px_hi = potential_channel(clean.seed, 2, [p[0] + EPS, p[1], p[2]], time);
    let pz_px_lo = potential_channel(clean.seed, 2, [p[0] - EPS, p[1], p[2]], time);
    let pz_py_hi = potential_channel(clean.seed, 2, [p[0], p[1] + EPS, p[2]], time);
    let pz_py_lo = potential_channel(clean.seed, 2, [p[0], p[1] - EPS, p[2]], time);

    let dpz_dy = (pz_py_hi - pz_py_lo) * inv_two_eps;
    let dpy_dz = (py_pz_hi - py_pz_lo) * inv_two_eps;
    let dpx_dz = (px_pz_hi - px_pz_lo) * inv_two_eps;
    let dpz_dx = (pz_px_hi - pz_px_lo) * inv_two_eps;
    let dpy_dx = (py_px_hi - py_px_lo) * inv_two_eps;
    let dpx_dy = (px_py_hi - px_py_lo) * inv_two_eps;

    // curl P = (dPz/dy - dPy/dz, dPx/dz - dPz/dx, dPy/dx - dPx/dy).
    let wx = dpz_dy - dpy_dz;
    let wy = dpx_dz - dpz_dx;
    let wz = dpy_dx - dpx_dy;

    [
        wx * clean.amplitude,
        wy * clean.amplitude,
        wz * clean.amplitude,
    ]
}

/// Per-point wind for a whole groom: maps each position through [`curl_wind`]
/// with the same params and time, preserving input order. An empty slice returns
/// an empty [`Vec`] (no panic); this is the array-in / array-out form used to
/// drive per-strand-vertex external forces.
#[must_use]
pub fn curl_wind_map(params: WindFieldParams, positions: &[[f32; 3]], time: f32) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(positions.len());
    for &pos in positions {
        out.push(curl_wind(params, pos, time));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_T: f32 = 1.0e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS_T
    }

    fn close_vec(a: [f32; 3], b: [f32; 3]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
    }

    #[test]
    fn deterministic_same_input_same_output() {
        let params = WindFieldParams::new(1.5, 2.0, 7);
        let pos = [0.3, -1.2, 4.7];
        let a = curl_wind(params, pos, 0.25);
        let b = curl_wind(params, pos, 0.25);
        assert!(close_vec(a, b));
    }

    #[test]
    fn value_noise_is_in_unit_range_and_deterministic() {
        let samples = [
            [0.0, 0.0, 0.0],
            [0.37, 1.9, -2.4],
            [10.5, -3.3, 0.1],
            [-5.2, 7.7, 3.3],
        ];
        for p in samples {
            let v = value_noise(42, p);
            assert!(v.is_finite());
            assert!((-1.0..=1.0).contains(&v));
            assert!(close(v, value_noise(42, p)));
        }
    }

    #[test]
    fn approximately_divergence_free() {
        // Estimate div w = dwx/dx + dwy/dy + dwz/dz by central differences and
        // check it is near zero at several points. The curl is divergence-free
        // analytically; the residual here is pure finite-difference error.
        let params = WindFieldParams::new(0.8, 1.0, 123);
        let h = 2.0e-2_f32;
        let inv_two_h = 1.0 / (2.0 * h);
        let points = [
            [0.1, 0.2, 0.3],
            [1.7, -0.5, 2.2],
            [-3.1, 4.4, -1.1],
            [5.5, 5.5, 5.5],
        ];
        for p in points {
            let wx_hi = curl_wind(params, [p[0] + h, p[1], p[2]], 0.0)[0];
            let wx_lo = curl_wind(params, [p[0] - h, p[1], p[2]], 0.0)[0];
            let wy_hi = curl_wind(params, [p[0], p[1] + h, p[2]], 0.0)[1];
            let wy_lo = curl_wind(params, [p[0], p[1] - h, p[2]], 0.0)[1];
            let wz_hi = curl_wind(params, [p[0], p[1], p[2] + h], 0.0)[2];
            let wz_lo = curl_wind(params, [p[0], p[1], p[2] - h], 0.0)[2];
            let div = (wx_hi - wx_lo) * inv_two_h
                + (wy_hi - wy_lo) * inv_two_h
                + (wz_hi - wz_lo) * inv_two_h;
            assert!(div.abs() < 0.1, "divergence {div} too large at {p:?}");
        }
    }

    #[test]
    fn amplitude_scales_linearly() {
        let pos = [1.1, -2.2, 0.6];
        let time = 0.4;
        let base = WindFieldParams::new(1.0, 1.0, 55);
        let doubled = WindFieldParams::new(1.0, 2.0, 55);
        let w1 = curl_wind(base, pos, time);
        let w2 = curl_wind(doubled, pos, time);
        assert!(close(w2[0], w1[0] * 2.0));
        assert!(close(w2[1], w1[1] * 2.0));
        assert!(close(w2[2], w1[2] * 2.0));
    }

    #[test]
    fn zero_amplitude_is_zero_wind() {
        let params = WindFieldParams::new(1.0, 0.0, 9);
        let w = curl_wind(params, [3.0, 4.0, 5.0], 1.0);
        assert!(close_vec(w, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn non_finite_and_negative_params_do_not_panic_and_stay_finite() {
        let bad_freq = WindFieldParams::new(-4.0, 1.0, 1);
        let w1 = curl_wind(bad_freq, [1.0, 1.0, 1.0], 0.0);
        assert!(w1.iter().all(|c| c.is_finite()));

        let nan_amp = WindFieldParams::new(1.0, f32::NAN, 2);
        let w2 = curl_wind(nan_amp, [1.0, 1.0, 1.0], 0.0);
        assert!(w2.iter().all(|c| c.is_finite()));
        // NaN amplitude sanitises to 0 -> no wind.
        assert!(close_vec(w2, [0.0, 0.0, 0.0]));

        let inf_freq = WindFieldParams::new(f32::INFINITY, 1.0, 3);
        let w3 = curl_wind(inf_freq, [0.5, 0.5, 0.5], 0.0);
        assert!(w3.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn sanitized_repairs_bad_values() {
        let s = WindFieldParams::new(-1.0, -2.0, 4).sanitized();
        assert!(close(s.frequency, 1.0));
        assert!(close(s.amplitude, 0.0));
        assert_eq!(s.seed, 4);

        let s2 = WindFieldParams::new(f32::NAN, f32::INFINITY, 5).sanitized();
        assert!(close(s2.frequency, 1.0));
        assert!(close(s2.amplitude, 0.0));
    }

    #[test]
    fn empty_map_is_empty_without_panic() {
        let out = curl_wind_map(WindFieldParams::default(), &[], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn map_matches_scalar_and_preserves_order() {
        let params = WindFieldParams::new(1.2, 1.5, 77);
        let positions = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-4.0, 0.5, 2.5]];
        let time = 0.6;
        let mapped = curl_wind_map(params, &positions, time);
        assert_eq!(mapped.len(), positions.len());
        for (pos, got) in positions.iter().zip(mapped.iter()) {
            assert!(close_vec(*got, curl_wind(params, *pos, time)));
        }
    }

    #[test]
    fn field_evolves_with_time() {
        // Different times should generally give different wind at a fixed point;
        // this guards against time being dropped from the noise coordinates.
        let params = WindFieldParams::new(1.0, 1.0, 11);
        let pos = [2.0, 3.0, 4.0];
        let a = curl_wind(params, pos, 0.0);
        let b = curl_wind(params, pos, 5.0);
        let differs = !close(a[0], b[0]) || !close(a[1], b[1]) || !close(a[2], b[2]);
        assert!(differs);
    }

    #[test]
    fn output_is_finite_for_varied_inputs() {
        let params = WindFieldParams::new(2.5, 1.0, 321);
        for i in 0..16_i32 {
            let f = i as f32;
            let w = curl_wind(params, [f * 0.7, -f * 1.3, f * 0.2], f * 0.05);
            assert!(w.iter().all(|c| c.is_finite()));
        }
    }
}
