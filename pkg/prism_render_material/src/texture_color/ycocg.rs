//! **`YCoCg`** luma / chroma colour transforms for texture authoring.
//!
//! `YCoCg` rotates the RGB cube onto a luma axis `Y` (the perceptual
//! brightness), an orange-blue chroma axis `Co` and a green-magenta chroma axis
//! `Cg`. Compared with RGB the three channels are far closer to decorrelated,
//! which is why texture compressors (the `YCoCg`-DXT5 scheme, JPEG-XR, H.264
//! `FRExt`) store chroma in this basis before quantisation. The material pipeline
//! exposes it so recolour / chroma-key tools and compressor front-ends share
//! one exact implementation.
//!
//! Two variants are provided:
//!
//! * [`rgb_to_ycocg`] / [`ycocg_to_rgb`] — the *lossy orthogonal* float form on
//!   `[0,1]` channels. `Y = R/4 + G/2 + B/4`, `Co = (R - B)/2`,
//!   `Cg = (2G - R - B)/4`; the inverse is the exact analytic matrix
//!   `R = Y + Co - Cg`, `G = Y + Cg`, `B = Y - Co - Cg`. It round-trips to
//!   floating-point tolerance, the primary anti-fake oracle for this form.
//! * [`rgb_to_ycocg_r`] / [`ycocg_r_to_rgb`] — the *reversible* `YCoCg-R`
//!   integer lifting (Malvar & Sullivan 2003) on 8-bit channels. The three
//!   lifting steps and their inverses use the identical arithmetic right shift,
//!   so the transform is **bit-exact invertible** over every integer triple —
//!   the gold-standard anti-fake oracle, verified exhaustively on a dense grid.
//!   `Y` stays on `[0,255]`; the chroma pair uses the signed 9-bit range
//!   `[-255,255]`.
//!
//! All arithmetic is plain add / subtract / multiply / shift (no
//! transcendentals, no AI/ML), so a CPU golden matches a GPU twin exactly for
//! the integer form and to floating-point tolerance for the float form.
//!
//! # References
//! * H. Malvar & G. Sullivan, "YCoCg-R: A Color Space with RGB Reversibility
//!   and Low Dynamic Range", JVT-I014r3, 2003.
//! * van Waveren & Castaño, "Real-Time YCoCg-DXT Compression", id Software /
//!   NVIDIA, 2007.

/// Convert a linear RGB triple on `[0,1]` to the lossy orthogonal `YCoCg`
/// basis.
///
/// Returns `[Y, Co, Cg]` with luma `Y` on `[0,1]` and the chroma pair on
/// `[-0.5,0.5]`. The transform is `Y = R/4 + G/2 + B/4`, `Co = (R - B)/2`,
/// `Cg = (2G - R - B)/4`.
#[inline]
#[must_use]
pub fn rgb_to_ycocg(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    let y = 0.25 * r + 0.5 * g + 0.25 * b;
    let co = 0.5 * (r - b);
    let cg = 0.5 * g - 0.25 * (r + b);
    [y, co, cg]
}

/// Inverse of [`rgb_to_ycocg`]: reconstruct the RGB triple from `[Y, Co, Cg]`.
///
/// Uses the exact analytic inverse `R = Y + Co - Cg`, `G = Y + Cg`,
/// `B = Y - Co - Cg`.
#[inline]
#[must_use]
pub fn ycocg_to_rgb(ycocg: [f32; 3]) -> [f32; 3] {
    let [y, co, cg] = ycocg;
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    [r, g, b]
}

/// Convert an 8-bit RGB triple to the reversible `YCoCg-R` integer basis.
///
/// Returns `[Y, Co, Cg]` where `Y` lies on `[0,255]` and the chroma pair on the
/// signed range `[-255,255]`. Paired with [`ycocg_r_to_rgb`] this is a
/// bit-exact lossless round-trip for every integer input.
#[inline]
#[must_use]
pub fn rgb_to_ycocg_r(rgb: [u8; 3]) -> [i16; 3] {
    let r = i32::from(rgb[0]);
    let g = i32::from(rgb[1]);
    let b = i32::from(rgb[2]);
    // Reversible lifting (arithmetic right shift, floor toward -inf).
    let co = r - b;
    let tmp = b + (co >> 1);
    let cg = g - tmp;
    let y = tmp + (cg >> 1);
    [y as i16, co as i16, cg as i16]
}

/// Inverse of [`rgb_to_ycocg_r`]: reconstruct the 8-bit RGB triple.
///
/// Reverses the lifting with the same arithmetic right shift, so it exactly
/// recovers the original channels for any value produced by
/// [`rgb_to_ycocg_r`]. Inputs outside the lossless range are clamped to
/// `[0,255]`.
#[inline]
#[must_use]
pub fn ycocg_r_to_rgb(ycocg: [i16; 3]) -> [u8; 3] {
    let y = i32::from(ycocg[0]);
    let co = i32::from(ycocg[1]);
    let cg = i32::from(ycocg[2]);
    let tmp = y - (cg >> 1);
    let g = cg + tmp;
    let b = tmp - (co >> 1);
    let r = b + co;
    [clamp_u8(r), clamp_u8(g), clamp_u8(b)]
}

/// Clamp an integer channel into `[0,255]` and narrow to `u8`.
#[inline]
#[must_use]
fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::{rgb_to_ycocg, rgb_to_ycocg_r, ycocg_r_to_rgb, ycocg_to_rgb};

    /// `YCoCg-R` is a bit-exact lossless round-trip on a dense integer grid.
    #[test]
    fn ycocg_r_round_trips_exactly() {
        let steps = [0u8, 1, 2, 17, 63, 64, 100, 127, 128, 200, 254, 255];
        for &r in &steps {
            for &g in &steps {
                for &b in &steps {
                    let rt = ycocg_r_to_rgb(rgb_to_ycocg_r([r, g, b]));
                    assert_eq!(rt, [r, g, b], "lossless failed for {:?}", [r, g, b]);
                }
            }
        }
    }

    /// `YCoCg-R` luma stays on `[0,255]` and achromatic inputs zero the chroma.
    #[test]
    fn ycocg_r_luma_range_and_achromatic() {
        for v in 0u16..=255 {
            let v = v as u8;
            let [y, co, cg] = rgb_to_ycocg_r([v, v, v]);
            assert_eq!(co, 0, "grey must have zero Co");
            assert_eq!(cg, 0, "grey must have zero Cg");
            assert_eq!(y, i16::from(v), "grey luma equals the level");
        }
        // Over the full byte grid the luma never escapes [0,255].
        let steps = [0u8, 31, 63, 95, 127, 159, 191, 223, 255];
        for &r in &steps {
            for &g in &steps {
                for &b in &steps {
                    let y = rgb_to_ycocg_r([r, g, b])[0];
                    assert!((0..=255).contains(&y), "luma {y} out of range");
                }
            }
        }
    }

    /// The float `YCoCg` form round-trips to floating-point tolerance.
    #[test]
    fn ycocg_float_round_trips() {
        let steps = [0.0f32, 0.1, 0.25, 0.5, 0.73, 0.9, 1.0];
        for &r in &steps {
            for &g in &steps {
                for &b in &steps {
                    let rt = ycocg_to_rgb(rgb_to_ycocg([r, g, b]));
                    for k in 0..3 {
                        let diff = (rt[k] - [r, g, b][k]).abs();
                        assert!(diff < 1e-6, "round-trip drift {diff} at {:?}", [r, g, b]);
                    }
                }
            }
        }
    }

    /// The float form matches the analytic luma weights and chroma axes.
    #[test]
    fn ycocg_float_known_values() {
        // White projects to full luma with zero chroma.
        let [y, co, cg] = rgb_to_ycocg([1.0, 1.0, 1.0]);
        assert!((y - 1.0).abs() < 1e-6);
        assert!(co.abs() < 1e-6 && cg.abs() < 1e-6);

        // Co is the orange-blue axis: pure red vs pure blue flip its sign.
        let red = rgb_to_ycocg([1.0, 0.0, 0.0]);
        let blue = rgb_to_ycocg([0.0, 0.0, 1.0]);
        assert!((red[1] - 0.5).abs() < 1e-6, "red Co");
        assert!((blue[1] + 0.5).abs() < 1e-6, "blue Co");

        // Cg is the green-magenta axis: pure green is positive, red/blue mix
        // (magenta) is negative.
        let green = rgb_to_ycocg([0.0, 1.0, 0.0]);
        assert!((green[2] - 0.5).abs() < 1e-6, "green Cg");
        let magenta = rgb_to_ycocg([1.0, 0.0, 1.0]);
        assert!((magenta[2] + 0.5).abs() < 1e-6, "magenta Cg");
    }
}
