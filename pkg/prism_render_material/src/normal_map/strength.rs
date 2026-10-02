//! Adjusting the perceived strength of a tangent-space normal (detail intensity).
//!
//! AAA material graphs expose a "normal strength" / "bump intensity" control
//! that scales how pronounced a normal map looks without re-authoring the
//! texture. The physically meaningful place to do this is **slope space**: a
//! unit tangent-space normal `n = (nx, ny, nz)` (z-up hemisphere, `nz > 0`) is
//! the surface normal of a height field whose gradient (slope) is
//! `g = (-nx / nz, -ny / nz)`. Scaling the strength by `s` scales that slope,
//! `g' = s * g`, which steepens (`s > 1`) or flattens (`s < 1`) the relief while
//! leaving the flat direction fixed.
//!
//! Multiplying the slope by `s` and converting back is algebraically the same
//! as scaling the tangent `xy` and renormalising, so [`scale_strength`] needs no
//! division and stays robust as `nz -> 0`:
//!
//! ```text
//! n' = normalize(s * nx, s * ny, nz)
//! ```
//!
//! This gives exactly the properties a strength control must have:
//! * `s = 1` is the identity (direction unchanged);
//! * `s = 0` flattens to `(0, 0, 1)`;
//! * a flat input stays flat for any `s`;
//! * strengths compose multiplicatively: `scale(scale(n, a), b) == scale(n, a * b)`.
//!
//! [`normal_to_slope`] / [`slope_to_normal`] expose the slope-space round trip
//! directly for height-field / derivative workflows. Pure analytic math, no
//! AI/ML, so a CPU golden matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Sections 6.7
//!   (normal mapping) and 16.3 (height-field gradient / slope).
//! * Mikkelsen, "Bump Mapping Unparametrized Surfaces on the GPU"
//!   (slope-space perturbation).

/// Flat tangent-space normal (z up). Returned when a result would be degenerate.
const UP: [f32; 3] = [0.0, 0.0, 1.0];

/// Smallest `nz` magnitude used as a divisor in [`normal_to_slope`], so a normal
/// grazing the tangent plane yields a large but finite slope instead of `inf`.
const NZ_FLOOR: f32 = 1.0e-6;

#[inline]
fn normalize_up(v: [f32; 3]) -> [f32; 3] {
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len2 > 1.0e-12 {
        let inv = 1.0 / len2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        UP
    }
}

/// Scale the strength (detail intensity) of a tangent-space normal.
///
/// `strength > 1` steepens the relief, `strength < 1` flattens it, `strength`
/// of `1` is the identity and `0` returns the flat normal `(0, 0, 1)`. Negative
/// strengths mirror the slope (invert the bump), which is occasionally used to
/// flip a detail layer. Inputs are expected to be unit normals in the upper
/// (`z >= 0`) hemisphere; the result is renormalised.
#[must_use]
pub fn scale_strength(normal: [f32; 3], strength: f32) -> [f32; 3] {
    normalize_up([strength * normal[0], strength * normal[1], normal[2]])
}

/// Convert a tangent-space unit normal to its height-field slope (gradient)
/// `(-nx / nz, -ny / nz)`.
///
/// The `nz` divisor is floored at [`NZ_FLOOR`] (keeping its sign) so a normal
/// lying in the tangent plane maps to a large finite slope rather than `inf`.
#[must_use]
pub fn normal_to_slope(normal: [f32; 3]) -> [f32; 2] {
    let nz = if normal[2].abs() < NZ_FLOOR {
        if normal[2] < 0.0 {
            -NZ_FLOOR
        } else {
            NZ_FLOOR
        }
    } else {
        normal[2]
    };
    [-normal[0] / nz, -normal[1] / nz]
}

/// Convert a height-field slope (gradient) `(sx, sy)` back to a unit
/// tangent-space normal `normalize(-sx, -sy, 1)`.
#[must_use]
pub fn slope_to_normal(slope: [f32; 2]) -> [f32; 3] {
    normalize_up([-slope[0], -slope[1], 1.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: [f32; 3]) -> [f32; 3] {
        let inv = 1.0 / (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }

    fn close3(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1.0e-5, "a={a:?} b={b:?}");
        }
    }

    fn is_unit(n: [f32; 3]) {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1.0e-5, "len={len} n={n:?}");
    }

    const SAMPLES: [[f32; 3]; 4] = [
        [0.0, 0.0, 1.0],
        [0.3, -0.2, 0.9327],
        [-0.5, 0.5, 0.72],
        [0.1, 0.7, 0.7],
    ];

    #[test]
    fn strength_one_is_identity() {
        for &n in &SAMPLES {
            let n = unit(n);
            close3(scale_strength(n, 1.0), n);
        }
    }

    #[test]
    fn strength_zero_flattens() {
        for &n in &SAMPLES {
            close3(scale_strength(unit(n), 0.0), UP);
        }
    }

    #[test]
    fn flat_stays_flat() {
        for &s in &[0.0, 0.5, 1.0, 2.0, 5.0] {
            close3(scale_strength(UP, s), UP);
        }
    }

    #[test]
    fn strengths_compose_multiplicatively() {
        for &n in &SAMPLES {
            let n = unit(n);
            for &(a, b) in &[(2.0_f32, 3.0_f32), (0.5, 0.25), (4.0, 0.5), (1.5, 2.0)] {
                let stepwise = scale_strength(scale_strength(n, a), b);
                let combined = scale_strength(n, a * b);
                close3(stepwise, combined);
            }
        }
    }

    #[test]
    fn output_is_always_unit() {
        for &n in &SAMPLES {
            let n = unit(n);
            for &s in &[0.0, 0.25, 1.0, 3.0, 8.0] {
                is_unit(scale_strength(n, s));
            }
        }
    }

    #[test]
    fn slope_round_trip() {
        for &n in &SAMPLES {
            let n = unit(n);
            close3(slope_to_normal(normal_to_slope(n)), n);
        }
    }

    #[test]
    fn scale_matches_slope_space() {
        for &n in &SAMPLES {
            let n = unit(n);
            for &s in &[0.25_f32, 0.5, 2.0, 4.0] {
                let g = normal_to_slope(n);
                let via_slope = slope_to_normal([s * g[0], s * g[1]]);
                close3(scale_strength(n, s), via_slope);
            }
        }
    }

    #[test]
    fn greater_strength_steepens_lower_flattens() {
        // A tilted normal: strength > 1 lowers z (steeper), strength < 1 raises
        // z (flatter), monotonically.
        let n = unit([0.3, -0.2, 0.9327]);
        let steep = scale_strength(n, 3.0);
        let flat = scale_strength(n, 0.3);
        assert!(steep[2] < n[2], "steep z {} !< {}", steep[2], n[2]);
        assert!(flat[2] > n[2], "flat z {} !> {}", flat[2], n[2]);
    }

    #[test]
    fn negative_strength_mirrors_slope() {
        let n = unit([0.3, -0.2, 0.9327]);
        let flipped = scale_strength(n, -1.0);
        // xy slope is mirrored, z (and thus unit length) preserved.
        close3(flipped, unit([-n[0], -n[1], n[2]]));
    }
}
