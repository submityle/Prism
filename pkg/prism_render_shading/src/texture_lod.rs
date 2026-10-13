//! Analytic per-pixel texture LOD for the compute resolve — the CPU golden
//! (§7.1 distance LOD / §10 "远处高光不闪烁", roadmap P1-a).
//!
//! # Why a compute pass needs this at all
//!
//! A fragment shader asks the hardware for `textureSample` and the rasteriser
//! hands it `dpdx`/`dpdy` for free. The material resolve is a **compute**
//! dispatch: there are no derivatives, so the shader used to sample every
//! material map at an explicit LOD 0 — full bandwidth and full aliasing at every
//! distance, which is the opposite of §7.1's "bandwidth falls with distance".
//!
//! The footprint is still available analytically, though: the resolve has the
//! triangle's three vertices (position *and* UV) and the frame's
//! `clip_from_world`, so projecting the vertices gives the screen-space triangle
//! and hence the exact per-pixel `dUV/dx` / `dUV/dy`. Barycentric coordinates are
//! affine over a triangle, so their gradients are constant across it:
//!
//! ```text
//! dλ_b/dx =  e2.y / det      dλ_b/dy = -e2.x / det
//! dλ_c/dx = -e1.y / det      dλ_c/dy =  e1.x / det
//! det     = e1.x·e2.y − e1.y·e2.x,   e1 = p_b − p_a,  e2 = p_c − p_a
//! dUV/dx  = (uv_b − uv_a)·dλ_b/dx + (uv_c − uv_a)·dλ_c/dx
//! ```
//!
//! The isotropic mip level is then the standard GPU rule — the length of the
//! texel-space gradient, the larger axis winning:
//!
//! ```text
//! lod = log2(max(‖dUV/dx · dims‖, ‖dUV/dy · dims‖))
//! ```
//!
//! # Guarantees
//!
//! * **Deterministic**: a pure function of the triangle's vertices; no history,
//!   no jitter, no platform-specific derivative hardware.
//! * **Conservative at degeneracy**: a sliver or back-facing triangle (|det|
//!   at or below [`MIN_SCREEN_AREA`]) returns a zero gradient, which maps to LOD
//!   0 — exactly the resolve's previous behaviour, so a pathological triangle
//!   can never be blurred into a different material.
//! * **Monotone**: shrinking a triangle's screen area by `k` raises the LOD by
//!   `log2(k)`, so the mip chain is walked continuously with distance instead of
//!   snapping (the §7.3 "档位过渡" requirement).

use bevy_math::ops;

/// A screen-space triangle (or parallelogram) at or below this area is treated
/// as degenerate, in pixels².
///
/// Chosen well below one pixel so a genuine sub-pixel triangle still computes a
/// (large) LOD rather than falling back to the base level.
pub const MIN_SCREEN_AREA: f32 = 1.0e-6;

/// Inputs outside `±FINITE_RANGE` are treated as non-finite and keep the base
/// level.
///
/// The check is a pair of range comparisons rather than `is_finite()` because
/// the GPU twin has to be able to express the same thing: a `(x - x) == 0`
/// finiteness idiom is *folded away* by Metal's fast-math (measured — it made an
/// infinite UV sample at an infinite LOD), while `x > -LIMIT && x < LIMIT` is a
/// real comparison that stays. The limit sits far above any plausible
/// UV·texel product (a 16k map at 1000 texels/pixel is ~1.6e7) and far below
/// `f32::MAX`, so nothing legitimate is ever caught by it.
pub const FINITE_RANGE: f32 = 1.0e37;

/// Whether `value` is inside the finite range both implementations agree on.
#[must_use]
pub fn in_finite_range(value: f32) -> bool {
    value > -FINITE_RANGE && value < FINITE_RANGE
}

/// The per-pixel UV gradients `[dUV/dx, dUV/dy]`, in UV units per pixel.
///
/// `uv[i]` / `screen[i]` are the three vertices' texture coordinates and
/// screen-space positions in pixels, in the same winding order. Returns
/// `[[0, 0], [0, 0]]` for a degenerate triangle, which [`texture_lod`] maps to
/// the base level.
#[must_use]
pub fn uv_gradients(uv: [[f32; 2]; 3], screen: [[f32; 2]; 3]) -> [[f32; 2]; 2] {
    // A non-finite or absurd vertex keeps the base level before any arithmetic
    // can turn it into an infinite level.
    for vertex in 0..3 {
        if !in_finite_range(uv[vertex][0])
            || !in_finite_range(uv[vertex][1])
            || !in_finite_range(screen[vertex][0])
            || !in_finite_range(screen[vertex][1])
        {
            return [[0.0, 0.0], [0.0, 0.0]];
        }
    }
    let e1 = [screen[1][0] - screen[0][0], screen[1][1] - screen[0][1]];
    let e2 = [screen[2][0] - screen[0][0], screen[2][1] - screen[0][1]];
    let det = e1[0] * e2[1] - e1[1] * e2[0];
    let area = ops::abs(det);
    if area <= MIN_SCREEN_AREA {
        // Degenerate triangle: keep the base level rather than dividing by ~0
        // and blurring the surface to the smallest mip.
        return [[0.0, 0.0], [0.0, 0.0]];
    }
    let inv_det = 1.0 / det;
    let d_lambda_b_dx = e2[1] * inv_det;
    let d_lambda_c_dx = -e1[1] * inv_det;
    let d_lambda_b_dy = -e2[0] * inv_det;
    let d_lambda_c_dy = e1[0] * inv_det;
    let d_uv_b = [uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]];
    let d_uv_c = [uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]];
    [
        [
            d_uv_b[0] * d_lambda_b_dx + d_uv_c[0] * d_lambda_c_dx,
            d_uv_b[1] * d_lambda_b_dx + d_uv_c[1] * d_lambda_c_dx,
        ],
        [
            d_uv_b[0] * d_lambda_b_dy + d_uv_c[0] * d_lambda_c_dy,
            d_uv_b[1] * d_lambda_b_dy + d_uv_c[1] * d_lambda_c_dy,
        ],
    ]
}

/// The isotropic mip level for a `dimensions`-texel texture sampled with
/// `gradients` (see [`uv_gradients`]).
///
/// A magnified or exactly 1:1 footprint returns `0.0` — never a negative level,
/// so the result is directly usable as an explicit `textureSampleLevel` LOD.
#[must_use]
pub fn texture_lod(gradients: [[f32; 2]; 2], dimensions: [f32; 2]) -> f32 {
    if !in_finite_range(dimensions[0])
        || !in_finite_range(dimensions[1])
        || !in_finite_range(gradients[0][0])
        || !in_finite_range(gradients[0][1])
        || !in_finite_range(gradients[1][0])
        || !in_finite_range(gradients[1][1])
    {
        return 0.0;
    }
    let d_x = [
        gradients[0][0] * dimensions[0],
        gradients[0][1] * dimensions[1],
    ];
    let d_y = [
        gradients[1][0] * dimensions[0],
        gradients[1][1] * dimensions[1],
    ];
    let len_x = ops::sqrt(d_x[0] * d_x[0] + d_x[1] * d_x[1]);
    let len_y = ops::sqrt(d_y[0] * d_y[0] + d_y[1] * d_y[1]);
    let extent = len_x.max(len_y);
    // `extent <= 1` is the magnified / 1:1-or-clearer case: the base level is
    // already correct (and log2 of a sub-unit extent would go negative), so the
    // requested level can never leave the mip chain.
    if extent <= 1.0 {
        return 0.0;
    }
    ops::log2(extent)
}

/// Convenience: the level a `dimensions`-texel map should be sampled at for the
/// triangle `uv`/`screen`, clamped into `[0, max_level]`.
#[must_use]
pub fn triangle_lod(
    uv: [[f32; 2]; 3],
    screen: [[f32; 2]; 3],
    dimensions: [f32; 2],
    max_level: f32,
) -> f32 {
    texture_lod(uv_gradients(uv, screen), dimensions).clamp(0.0, max_level.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A right triangle spanning one UV unit over `size` screen pixels in both
    /// axes, i.e. `1/size` UV per pixel.
    fn scaled(size: f32) -> ([[f32; 2]; 3], [[f32; 2]; 3]) {
        (
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [size, 0.0], [0.0, size]],
        )
    }

    #[test]
    fn gradients_are_uv_units_per_pixel() {
        let (uv, screen) = scaled(4.0);
        let gradients = uv_gradients(uv, screen);
        // 1 UV unit over 4 pixels along each screen axis.
        assert!((gradients[0][0] - 0.25).abs() < 1.0e-6, "{gradients:?}");
        assert!(gradients[0][1].abs() < 1.0e-6, "{gradients:?}");
        assert!(gradients[1][0].abs() < 1.0e-6, "{gradients:?}");
        assert!((gradients[1][1] - 0.25).abs() < 1.0e-6, "{gradients:?}");
    }

    #[test]
    fn one_texel_per_pixel_lands_on_the_base_level() {
        let (uv, screen) = scaled(1.0);
        // 1 UV unit per pixel over a 1-texel map: exactly 1:1, base level.
        let lod = texture_lod(uv_gradients(uv, screen), [1.0, 1.0]);
        assert_eq!(lod, 0.0);
        // 0.5 UV per pixel over a 2-texel map is also exactly 1:1.
        let (uv, screen) = scaled(2.0);
        let lod = texture_lod(uv_gradients(uv, screen), [2.0, 2.0]);
        assert_eq!(lod, 0.0);
    }

    #[test]
    fn one_uv_unit_per_pixel_of_a_1024_map_lands_on_level_ten() {
        let (uv, screen) = scaled(1.0);
        let lod = texture_lod(uv_gradients(uv, screen), [1024.0, 1024.0]);
        assert!((lod - 10.0).abs() < 1.0e-4, "{lod}");
    }

    #[test]
    fn shrinking_the_triangle_raises_the_lod_by_log2_of_the_ratio() {
        let (uv, screen) = scaled(1.0);
        let base = texture_lod(uv_gradients(uv, screen), [256.0, 256.0]);
        for ratio in [2.0_f32, 4.0, 8.0] {
            let (uv, screen) = scaled(ratio);
            let lod = texture_lod(uv_gradients(uv, screen), [256.0, 256.0]);
            assert!(
                (lod - (base - ops::log2(ratio))).abs() < 1.0e-3,
                "ratio {ratio}: {lod} vs {base}"
            );
        }
    }

    #[test]
    fn a_degenerate_triangle_keeps_the_base_level() {
        // Collinear vertices: zero screen area.
        let uv = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let screen = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let gradients = uv_gradients(uv, screen);
        assert_eq!(gradients, [[0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(texture_lod(gradients, [4096.0, 4096.0]), 0.0);
        // A zero-area triangle made of coincident vertices too.
        let gradients = uv_gradients(uv, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
        assert_eq!(texture_lod(gradients, [4096.0, 4096.0]), 0.0);
    }

    #[test]
    fn a_magnified_footprint_never_asks_for_a_negative_level() {
        // 0.25 UV units per pixel over a 2-texel map: 0.5 texels per pixel, i.e.
        // magnified. `log2(0.5) = -1` would be an invalid LOD, so it clamps to 0.
        let uv = [[0.0, 0.0], [0.25, 0.0], [0.0, 0.25]];
        let screen = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_eq!(texture_lod(uv_gradients(uv, screen), [2.0, 2.0]), 0.0);
    }

    #[test]
    fn non_finite_inputs_stay_on_the_base_level() {
        // A NaN screen vertex makes the determinant NaN: the guard must keep the
        // base level rather than propagate a NaN LOD into `textureSampleLevel`.
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let screen = [[0.0, 0.0], [f32::NAN, 0.0], [0.0, 1.0]];
        assert_eq!(uv_gradients(uv, screen), [[0.0, 0.0], [0.0, 0.0]]);
        // A NaN gradient (e.g. an infinite texture size) also clamps to 0.
        assert_eq!(
            texture_lod([[f32::NAN, 0.0], [0.0, 1.0]], [1024.0, 1024.0]),
            0.0
        );
    }

    #[test]
    fn the_lod_is_monotone_in_screen_area() {
        let mut previous = f32::NEG_INFINITY;
        for size in [64.0_f32, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5] {
            let (uv, screen) = scaled(size);
            let lod = texture_lod(uv_gradients(uv, screen), [512.0, 512.0]);
            assert!(lod > previous, "size {size}: {lod} after {previous}");
            previous = lod;
        }
    }

    #[test]
    fn triangle_lod_clamps_into_the_mip_chain() {
        let (uv, screen) = scaled(1.0);
        // A 1024 map's chain ends at level 10; the raw level is 10 here.
        assert_eq!(triangle_lod(uv, screen, [1024.0, 1024.0], 10.0), 10.0);
        // A coarse chain clamps rather than sampling past the smallest mip.
        assert_eq!(triangle_lod(uv, screen, [1024.0, 1024.0], 4.0), 4.0);
    }
}
