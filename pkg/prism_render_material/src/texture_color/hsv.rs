//! **HSV / HSL** cylindrical colour-space conversions.
//!
//! Recolour, tint and colour-grading tools expose hue / saturation / value (or
//! lightness) controls, so the material pipeline needs an exact, invertible map
//! between an RGB triple and these cylindrical models. Both are the standard
//! hexagonal-projection models (Smith 1978 / Joblove-Greenberg 1978): the hue
//! `H` is the angle around the colour hexagon, `S` the normalised distance from
//! the achromatic axis, and `V = max(R,G,B)` (HSV) or `L = (max+min)/2` (HSL)
//! the vertical position. They differ only in that vertical coordinate and the
//! saturation normalisation, and share the identical hue.
//!
//! The conversions operate on the numeric RGB triple as given (the caller
//! chooses whether that is display-encoded or scene-linear); they are exact
//! inverses on the `[0,1]^3` cube, which is the primary anti-fake oracle. Hue is
//! reported in degrees on `[0, 360)`; `S`, `V`, `L` on `[0, 1]`.
//!
//! Pure analytic `f32` arithmetic (min / max / divide / branches, no
//! transcendentals, no AI/ML), so a CPU golden matches a GPU twin to
//! floating-point tolerance.
//!
//! # References
//! * A. R. Smith, "Color Gamut Transform Pairs", SIGGRAPH 1978 (HSV).
//! * Joblove & Greenberg, "Color spaces for computer graphics", SIGGRAPH 1978
//!   (HSL).

/// Hue angle (degrees `[0,360)`) and chroma `max-min` shared by HSV and HSL.
///
/// Returns `(hue, max, min, chroma)`. Hue is `0` for an achromatic colour.
#[inline]
#[must_use]
fn hue_max_min(rgb: [f32; 3]) -> (f32, f32, f32, f32) {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let c = max - min;
    let mut h = if c <= 0.0 {
        0.0
    } else if max == r {
        60.0 * ((g - b) / c)
    } else if max == g {
        60.0 * ((b - r) / c + 2.0)
    } else {
        60.0 * ((r - g) / c + 4.0)
    };
    if h < 0.0 {
        h += 360.0;
    }
    (h, max, min, c)
}

/// Reconstruct an RGB triple from a hue (degrees), chroma `c` and per-channel
/// offset `m` using the hexagonal sextant rule (shared by HSV and HSL).
#[inline]
#[must_use]
fn hue_chroma_to_rgb(hue: f32, c: f32, m: f32) -> [f32; 3] {
    // Normalise the hue into [0,360) then into the [0,6) sextant coordinate.
    let mut h = hue % 360.0;
    if h < 0.0 {
        h += 360.0;
    }
    let hp = h / 60.0;
    let sextant = (hp as i32).clamp(0, 5);
    let f = hp - sextant as f32;
    let x = c * (1.0 - (if sextant & 1 == 0 { 1.0 - f } else { f }));
    let (r, g, b) = match sextant {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

/// Convert an RGB triple on `[0,1]` to `HSV` `[hue(deg), saturation, value]`.
#[must_use]
pub fn rgb_to_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let (h, max, _min, c) = hue_max_min(rgb);
    let s = if max <= 0.0 { 0.0 } else { c / max };
    [h, s, max]
}

/// Convert `HSV` `[hue(deg), saturation, value]` back to an RGB triple.
///
/// Saturation and value are clamped to `[0,1]`; hue wraps modulo `360`.
#[must_use]
pub fn hsv_to_rgb(hsv: [f32; 3]) -> [f32; 3] {
    let s = hsv[1].clamp(0.0, 1.0);
    let v = hsv[2].clamp(0.0, 1.0);
    let c = v * s;
    hue_chroma_to_rgb(hsv[0], c, v - c)
}

/// Convert an RGB triple on `[0,1]` to `HSL` `[hue(deg), saturation, lightness]`.
#[must_use]
pub fn rgb_to_hsl(rgb: [f32; 3]) -> [f32; 3] {
    let (h, max, min, c) = hue_max_min(rgb);
    let l = 0.5 * (max + min);
    // Denominator 1 - |2L - 1| == chroma range at this lightness.
    let denom = 1.0 - (2.0 * l - 1.0).abs();
    let s = if denom <= 0.0 { 0.0 } else { c / denom };
    [h, s, l]
}

/// Convert `HSL` `[hue(deg), saturation, lightness]` back to an RGB triple.
///
/// Saturation and lightness are clamped to `[0,1]`; hue wraps modulo `360`.
#[must_use]
pub fn hsl_to_rgb(hsl: [f32; 3]) -> [f32; 3] {
    let s = hsl[1].clamp(0.0, 1.0);
    let l = hsl[2].clamp(0.0, 1.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    hue_chroma_to_rgb(hsl[0], c, l - 0.5 * c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const EPS: f32 = 1.0e-5;

    fn close3(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < EPS, "a={a:?} b={b:?}");
        }
    }

    fn samples() -> Vec<[f32; 3]> {
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.2, 0.6, 0.1],
            [0.9, 0.3, 0.4],
            [0.05, 0.5, 0.95],
            [0.7, 0.7, 0.2],
            [0.33, 0.33, 0.33],
            [0.8, 0.1, 0.6],
        ]
        .to_vec()
    }

    #[test]
    fn hsv_round_trip_is_identity() {
        for c in samples() {
            close3(hsv_to_rgb(rgb_to_hsv(c)), c);
        }
    }

    #[test]
    fn hsl_round_trip_is_identity() {
        for c in samples() {
            close3(hsl_to_rgb(rgb_to_hsl(c)), c);
        }
    }

    #[test]
    fn value_is_max_channel_and_lightness_is_midrange() {
        for c in samples() {
            let hsv = rgb_to_hsv(c);
            let hsl = rgb_to_hsl(c);
            let max = c[0].max(c[1]).max(c[2]);
            let min = c[0].min(c[1]).min(c[2]);
            assert!((hsv[2] - max).abs() < EPS, "V {hsv:?} c={c:?}");
            assert!(
                (hsl[2] - 0.5 * (max + min)).abs() < EPS,
                "L {hsl:?} c={c:?}"
            );
        }
    }

    #[test]
    fn primaries_and_secondaries_map_to_known_hues() {
        let cases = [
            ([1.0, 0.0, 0.0], 0.0),
            ([1.0, 1.0, 0.0], 60.0),
            ([0.0, 1.0, 0.0], 120.0),
            ([0.0, 1.0, 1.0], 180.0),
            ([0.0, 0.0, 1.0], 240.0),
            ([1.0, 0.0, 1.0], 300.0),
        ];
        for (rgb, hue) in cases {
            assert!((rgb_to_hsv(rgb)[0] - hue).abs() < 1.0e-3, "rgb={rgb:?}");
            // HSV and HSL share the identical hue.
            assert!((rgb_to_hsl(rgb)[0] - hue).abs() < 1.0e-3, "rgb={rgb:?}");
        }
    }

    #[test]
    fn achromatic_has_zero_saturation() {
        for g in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            assert!(rgb_to_hsv([g, g, g])[1].abs() < EPS, "hsv s g={g}");
            assert!(rgb_to_hsl([g, g, g])[1].abs() < EPS, "hsl s g={g}");
        }
    }

    #[test]
    fn zero_saturation_reconstructs_grey() {
        for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            close3(hsv_to_rgb([123.0, 0.0, v]), [v, v, v]);
            close3(hsl_to_rgb([47.0, 0.0, v]), [v, v, v]);
        }
    }

    #[test]
    fn black_and_white_endpoints() {
        close3(rgb_to_hsv([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
        close3(rgb_to_hsv([1.0, 1.0, 1.0]), [0.0, 0.0, 1.0]);
        let bl = rgb_to_hsl([0.0, 0.0, 0.0]);
        let wh = rgb_to_hsl([1.0, 1.0, 1.0]);
        assert!(bl[2].abs() < EPS && bl[1].abs() < EPS, "black {bl:?}");
        assert!(
            (wh[2] - 1.0).abs() < EPS && wh[1].abs() < EPS,
            "white {wh:?}"
        );
    }

    #[test]
    fn hue_wraps_modulo_360() {
        // Feeding a hue outside [0,360) must match its wrapped representative.
        for &(h, hw) in &[(-60.0f32, 300.0f32), (420.0, 60.0), (720.0, 0.0)] {
            close3(hsv_to_rgb([h, 0.8, 0.7]), hsv_to_rgb([hw, 0.8, 0.7]));
            close3(hsl_to_rgb([h, 0.8, 0.5]), hsl_to_rgb([hw, 0.8, 0.5]));
        }
    }

    #[test]
    fn outputs_stay_in_range() {
        for c in samples() {
            let hsv = rgb_to_hsv(c);
            let hsl = rgb_to_hsl(c);
            for v in [hsv[1], hsv[2], hsl[1], hsl[2]] {
                assert!((0.0..=1.0).contains(&v), "range {hsv:?} {hsl:?} c={c:?}");
            }
            assert!((0.0..360.0).contains(&hsv[0]), "hue {hsv:?}");
            // Reconstructed RGB stays in gamut.
            for ch in hsv_to_rgb(hsv) {
                assert!((-EPS..=1.0 + EPS).contains(&ch), "rgb oob {c:?}");
            }
        }
    }

    #[test]
    fn hsv_to_rgb_round_trips_from_hsv() {
        // Chromatic HSV triples recover themselves (achromatic hue is arbitrary
        // so those are covered by zero_saturation_reconstructs_grey instead).
        for &h in &[10.0f32, 95.0, 200.0, 315.0] {
            for &s in &[0.3f32, 0.7, 1.0] {
                for &v in &[0.4f32, 0.8, 1.0] {
                    let got = rgb_to_hsv(hsv_to_rgb([h, s, v]));
                    assert!((got[0] - h).abs() < 1.0e-3, "h {got:?} want {h}");
                    assert!((got[1] - s).abs() < EPS, "s {got:?} want {s}");
                    assert!((got[2] - v).abs() < EPS, "v {got:?} want {v}");
                }
            }
        }
    }
}
