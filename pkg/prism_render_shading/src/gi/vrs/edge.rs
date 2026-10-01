//! Sobel-gradient contrast VRS classifier.
//!
//! This is the CPU golden reference for the *structural* VRS signal.  Where the
//! [`luma`](super::luma) classifier asks "how much does this tile vary on
//! average?", the edge classifier asks the sharper question "does this tile
//! straddle a strong luminance *edge*, and if so, along which axis?".  Edges —
//! silhouettes, high-frequency texture boundaries, specular highlights — are
//! exactly where coarse shading is most visible, so a tile that contains a
//! strong edge must be forbidden from coarsening *across* that edge.
//!
//! Edge strength is estimated with the classic **Sobel** operator, a
//! 3×3 separable gradient filter:
//!
//! ```text
//!        | -1  0 +1 |            | -1 -2 -1 |
//!  Gx =  | -2  0 +2 |     Gy =   |  0  0  0 |
//!        | -1  0 +1 |            | +1 +2 +1 |
//! ```
//!
//! `Gx` responds to *vertical* edges (a left/right luminance step), `Gy` to
//! *horizontal* edges.  Averaging `|Gx|` and `|Gy|` over a tile's interior
//! gives two per-axis edge energies whose geometric combination is the overall
//! edge magnitude.  The *ratio* of the two energies drives an **anisotropic**
//! decision:
//!
//! * A tile dominated by `Gx` (vertical edges) carries detail that must be
//!   resolved horizontally but is smooth vertically, so it may coarsen only in
//!   `y`: [`ShadingRate::X1x2`].
//! * A tile dominated by `Gy` (horizontal edges) is the transpose:
//!   [`ShadingRate::X2x1`].
//! * A tile with comparable energies (corners, texture) coarsens isotropically
//!   or, if strong enough, not at all.
//!
//! # Conventions
//! * Input is the tile's row-major luminance slice plus its `width`/`height`;
//!   unlike the variance classifier, the 2-D layout is essential here.
//! * Pure, deterministic, `no_std`-friendly: no RNG, IO, GPU, or `unsafe`, no
//!   allocation (the Sobel pass streams over the borrowed slice).
//! * Only `x.sqrt()` is needed (for the gradient magnitude); no
//!   [`bevy_math::ops`] transcendental is required.
//! * Defensive clamping is pervasive.  A tile that is too small to run a 3×3
//!   Sobel (`width < 3 || height < 3`), is mis-sized (`len != width * height`),
//!   or contains any non-finite sample falls back to the finest rate
//!   [`ShadingRate::X1x1`].

use super::ShadingRate;

/// Thresholds for the Sobel-magnitude → [`ShadingRate`] mapping.
///
/// `flat_gradient` and `strong_gradient` are expressed in the same units as the
/// averaged Sobel response, i.e. luminance per pixel scaled by the Sobel kernel
/// (whose positive weights sum to `4`).  They must satisfy `0 <= flat_gradient
/// <= strong_gradient`; [`EdgeThresholds::sanitized`] restores that ordering.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct EdgeThresholds {
    /// At or below this averaged gradient magnitude the tile is treated as
    /// smooth and may coarsen.  Defaults to `0.03` (a few percent luminance
    /// step across the tile).
    pub flat_gradient: f32,
    /// At or above this averaged gradient magnitude the tile contains a strong
    /// edge and is pinned to full rate (`X1x1`).  Defaults to `0.20`.
    pub strong_gradient: f32,
    /// How much one axis's edge energy must dominate the other to be treated as
    /// a directional (anisotropic) edge rather than an isotropic one.  A ratio
    /// of `2.0` means "at least twice as much energy on one axis".  Clamped to
    /// `>= 1.0`.
    pub anisotropy_ratio: f32,
}

impl Default for EdgeThresholds {
    #[inline]
    fn default() -> Self {
        Self {
            flat_gradient: 0.03,
            strong_gradient: 0.20,
            anisotropy_ratio: 2.0,
        }
    }
}

impl EdgeThresholds {
    /// Returns a copy with non-finite / negative fields repaired and
    /// `flat_gradient <= strong_gradient` re-established.
    #[inline]
    pub fn sanitized(self) -> Self {
        let flat = sanitize_nonneg(self.flat_gradient);
        let strong = sanitize_nonneg(self.strong_gradient);
        let ratio = {
            let r = sanitize_nonneg(self.anisotropy_ratio);
            if r >= 1.0 { r } else { 1.0 }
        };
        Self {
            flat_gradient: flat.min(strong),
            strong_gradient: flat.max(strong),
            anisotropy_ratio: ratio,
        }
    }
}

/// Averaged absolute Sobel gradient energies of a tile.
///
/// Produced by [`sobel_energy`]; both components are non-negative and finite for
/// a valid tile.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GradientEnergy {
    /// Mean `|Gx|` over the tile interior (responds to vertical edges).
    pub gx: f32,
    /// Mean `|Gy|` over the tile interior (responds to horizontal edges).
    pub gy: f32,
}

impl GradientEnergy {
    /// Overall edge magnitude, `sqrt(gx^2 + gy^2)`.
    #[inline]
    pub fn magnitude(&self) -> f32 {
        (self.gx * self.gx + self.gy * self.gy).sqrt()
    }
}

/// Replaces a non-finite scalar with `0.0`, otherwise returns it unchanged.
#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

/// Sanitizes a threshold: non-finite becomes `0.0`, negatives clamp up to
/// `0.0`.
#[inline]
fn sanitize_nonneg(x: f32) -> f32 {
    finite_or_zero(x).max(0.0)
}

/// Runs a 3×3 Sobel over the tile interior and returns the mean `|Gx|` / `|Gy|`.
///
/// Returns `None` for any degenerate tile: a slice whose length does not equal
/// `width * height`, a tile smaller than `3×3` (no interior pixel exists), or a
/// tile containing a non-finite sample.  The averaging denominator is the
/// interior pixel count `(width - 2) * (height - 2)`, so the result is
/// resolution-independent and directly comparable to the fixed thresholds.
pub fn sobel_energy(samples: &[f32], width: usize, height: usize) -> Option<GradientEnergy> {
    if width < 3 || height < 3 {
        return None;
    }
    if samples.len() != width.checked_mul(height)? {
        return None;
    }
    for &s in samples {
        if !s.is_finite() {
            return None;
        }
    }

    let mut sum_gx = 0.0_f32;
    let mut sum_gy = 0.0_f32;
    // Sample the 3×3 neighbourhood around each interior pixel (y, x).
    for y in 1..height - 1 {
        let row_above = (y - 1) * width;
        let row_mid = y * width;
        let row_below = (y + 1) * width;
        for x in 1..width - 1 {
            let tl = samples[row_above + x - 1];
            let tc = samples[row_above + x];
            let tr = samples[row_above + x + 1];
            let ml = samples[row_mid + x - 1];
            let mr = samples[row_mid + x + 1];
            let bl = samples[row_below + x - 1];
            let bc = samples[row_below + x];
            let br = samples[row_below + x + 1];

            let gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
            let gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
            sum_gx += gx.abs();
            sum_gy += gy.abs();
        }
    }

    let interior = ((width - 2) * (height - 2)) as f32;
    Some(GradientEnergy {
        gx: sum_gx / interior,
        gy: sum_gy / interior,
    })
}

/// Maps a pre-computed [`GradientEnergy`] onto the shading-rate lattice.
///
/// Exposed so callers that already hold gradient energies (or tests) can
/// exercise the decision logic without re-running Sobel.  Thresholds are
/// sanitized before use.
///
/// Decision ladder (coarsest to finest):
/// * magnitude at/above `strong_gradient` → [`ShadingRate::X1x1`].
/// * magnitude at/below `flat_gradient`:
///   * essentially flat (≤ ¼ of `flat_gradient`) → [`ShadingRate::X4x4`];
///   * otherwise gently textured → [`ShadingRate::X2x2`].
/// * in between (a moderate edge) the decision becomes **directional**:
///   * `gx` dominates `gy` by `anisotropy_ratio` → [`ShadingRate::X1x2`];
///   * `gy` dominates `gx` by `anisotropy_ratio` → [`ShadingRate::X2x1`];
///   * neither dominates → [`ShadingRate::X2x2`].
#[inline]
pub fn rate_from_energy(energy: GradientEnergy, thresholds: &EdgeThresholds) -> ShadingRate {
    let t = thresholds.sanitized();
    let gx = sanitize_nonneg(energy.gx);
    let gy = sanitize_nonneg(energy.gy);
    let mag = (gx * gx + gy * gy).sqrt();

    if mag >= t.strong_gradient {
        return ShadingRate::X1x1;
    }

    if mag <= t.flat_gradient {
        // Smooth region. Fully flat tiles earn the coarsest rate; faintly
        // textured ones stay a notch finer so sub-threshold structure is not
        // smeared across a 4-pixel block.
        return if mag <= 0.25 * t.flat_gradient {
            ShadingRate::X4x4
        } else {
            ShadingRate::X2x2
        };
    }

    // Moderate edge: coarsen only *along* the edge, never across it.
    if gx >= t.anisotropy_ratio * gy {
        // Vertical edges: keep horizontal detail, coarsen vertically.
        ShadingRate::X1x2
    } else if gy >= t.anisotropy_ratio * gx {
        // Horizontal edges: keep vertical detail, coarsen horizontally.
        ShadingRate::X2x1
    } else {
        // Mixed / corner structure: coarsen isotropically but modestly.
        ShadingRate::X2x2
    }
}

/// Classifies a tile from its row-major luminance samples using Sobel edge
/// energy.
///
/// Flat regions coarsen; strong edges pin the tile to full rate; moderate
/// directional edges coarsen only along the edge.  Any degenerate tile (see
/// [`sobel_energy`]) falls back to the finest rate [`ShadingRate::X1x1`].
pub fn classify_edge(
    samples: &[f32],
    width: usize,
    height: usize,
    thresholds: &EdgeThresholds,
) -> ShadingRate {
    match sobel_energy(samples, width, height) {
        Some(energy) => rate_from_energy(energy, thresholds),
        None => ShadingRate::X1x1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    /// A 4×4 tile that is uniform (no gradient anywhere).
    fn flat_tile() -> [f32; 16] {
        [0.5_f32; 16]
    }

    /// A 4×4 tile with a sharp left/right luminance step (vertical edge → Gx).
    fn vertical_edge_tile() -> [f32; 16] {
        [
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ]
    }

    /// A 4×4 tile with a sharp top/bottom step (horizontal edge → Gy).
    fn horizontal_edge_tile() -> [f32; 16] {
        [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]
    }

    #[test]
    fn degenerate_tiles_fall_back_to_full_rate() {
        let t = EdgeThresholds::default();
        // Too small for a 3×3 Sobel.
        assert_eq!(classify_edge(&[0.0; 4], 2, 2, &t), ShadingRate::X1x1);
        // Mis-sized slice.
        assert_eq!(classify_edge(&[0.0; 8], 3, 3, &t), ShadingRate::X1x1);
        // Non-finite sample.
        let mut bad = flat_tile();
        bad[5] = f32::NAN;
        assert_eq!(classify_edge(&bad, 4, 4, &t), ShadingRate::X1x1);
        assert!(sobel_energy(&bad, 4, 4).is_none());
    }

    #[test]
    fn flat_tile_coarsens_fully() {
        let t = EdgeThresholds::default();
        let flat = flat_tile();
        let e = sobel_energy(&flat, 4, 4).expect("valid");
        assert!(e.gx.abs() < EPS && e.gy.abs() < EPS);
        assert_eq!(classify_edge(&flat, 4, 4, &t), ShadingRate::X4x4);
    }

    #[test]
    fn strong_edge_pins_to_full_rate() {
        let t = EdgeThresholds::default();
        // A full 0->1 step is a very strong edge; magnitude far exceeds the
        // strong threshold regardless of its orientation.
        assert_eq!(classify_edge(&vertical_edge_tile(), 4, 4, &t), ShadingRate::X1x1);
        assert_eq!(classify_edge(&horizontal_edge_tile(), 4, 4, &t), ShadingRate::X1x1);
    }

    #[test]
    fn moderate_vertical_edge_coarsens_vertically() {
        // Scale the step down so the magnitude lands in the moderate band
        // (flat .. strong), making the directional branch fire.
        let t = EdgeThresholds::default();
        let base = vertical_edge_tile();
        let mut tile = [0.0_f32; 16];
        for (dst, src) in tile.iter_mut().zip(base.iter()) {
            *dst = src * 0.03; // mean|Gx| = 4*0.03 = 0.12, in (0.03, 0.20)
        }
        let e = sobel_energy(&tile, 4, 4).expect("valid");
        assert!(e.gx > e.gy, "vertical edge must have gx > gy");
        let m = e.magnitude();
        assert!(m > t.flat_gradient && m < t.strong_gradient, "mag {m}");
        assert_eq!(classify_edge(&tile, 4, 4, &t), ShadingRate::X1x2);
    }

    #[test]
    fn moderate_horizontal_edge_coarsens_horizontally() {
        let t = EdgeThresholds::default();
        let base = horizontal_edge_tile();
        let mut tile = [0.0_f32; 16];
        for (dst, src) in tile.iter_mut().zip(base.iter()) {
            *dst = src * 0.03;
        }
        let e = sobel_energy(&tile, 4, 4).expect("valid");
        assert!(e.gy > e.gx, "horizontal edge must have gy > gx");
        assert_eq!(classify_edge(&tile, 4, 4, &t), ShadingRate::X2x1);
    }

    #[test]
    fn magnitude_is_monotonic_in_edge_strength() {
        // As a vertical edge is scaled up, the chosen rate must never get
        // coarser (rank non-increasing): stronger edges demand equal-or-more
        // detail.
        let t = EdgeThresholds::default();
        let base = vertical_edge_tile();
        let mut prev_rank = ShadingRate::X4x4.rank();
        let mut scale = 0.0_f32;
        while scale <= 1.0 {
            let mut tile = [0.0_f32; 16];
            for (dst, src) in tile.iter_mut().zip(base.iter()) {
                *dst = src * scale;
            }
            let rank = classify_edge(&tile, 4, 4, &t).rank();
            assert!(rank <= prev_rank, "coarsened as edge strengthened at {scale}");
            prev_rank = rank;
            scale += 0.02;
        }
    }

    #[test]
    fn gradient_energy_magnitude_is_pythagorean() {
        let e = GradientEnergy { gx: 3.0, gy: 4.0 };
        assert!((e.magnitude() - 5.0).abs() < EPS);
    }

    #[test]
    fn diagonal_gradient_is_isotropic_medium() {
        // A gentle diagonal ramp f(x, y) = s * (x + y) is symmetric in x and y,
        // so Sobel gives gx == gy exactly; at a moderate magnitude neither axis
        // dominates and the isotropic 2x2 branch is chosen.
        let t = EdgeThresholds::default();
        let s = 0.01_f32; // Gx = Gy = 8 * s = 0.08; magnitude ~ 0.113 (moderate)
        let mut tile = [0.0_f32; 16];
        for y in 0..4usize {
            for x in 0..4usize {
                tile[y * 4 + x] = s * (x + y) as f32;
            }
        }
        let e = sobel_energy(&tile, 4, 4).expect("valid");
        assert!((e.gx - e.gy).abs() < 1.0e-5, "expected gx == gy by symmetry");
        let m = e.magnitude();
        assert!(m > t.flat_gradient && m < t.strong_gradient, "mag {m}");
        assert_eq!(classify_edge(&tile, 4, 4, &t), ShadingRate::X2x2);
    }

    #[test]
    fn thresholds_sanitize_reorders_and_clamps() {
        let bad = EdgeThresholds {
            flat_gradient: 0.5,
            strong_gradient: 0.1,
            anisotropy_ratio: 0.2,
        };
        let s = bad.sanitized();
        assert!(s.flat_gradient <= s.strong_gradient);
        assert!(s.anisotropy_ratio >= 1.0);
    }
}
