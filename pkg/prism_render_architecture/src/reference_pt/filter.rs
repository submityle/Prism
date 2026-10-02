//! Pixel reconstruction filters applied by importance-sampling the sub-pixel
//! sample position.
//!
//! A naive renderer gives every sub-pixel sample the same weight (a box
//! filter), which lets high-frequency detail alias badly across pixel edges.
//! Production renderers instead reconstruct each pixel with a smooth, wider
//! kernel so neighbouring detail is blended gracefully. Rather than splatting
//! every sample across several pixels (which would break the film's independent
//! per-pixel sampling), this module reconstructs the kernel by *importance
//! sampling* it: the uniform `[0, 1)` jitter from the sampler is warped through
//! the filter's inverse cumulative distribution (`CDF`) so sample positions are
//! already distributed with the filter's shape. Averaging the radiance of those
//! samples is then exactly a filtered estimate, and the per-pixel streaming and
//! determinism of the film are preserved.
//!
//! Every warp uses only comparisons and `sqrt`, honouring the crate's
//! no-transcendentals determinism policy.

use super::sampler::Sample2;

/// A separable pixel reconstruction filter, selected per render.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PixelFilter {
    /// Uniform weighting over the pixel cell: samples stay in `[0, 1)` of the
    /// pixel and are averaged with equal weight. Cheapest, but prone to
    /// aliasing on sharp edges.
    Box,
    /// A radius-one triangular (tent) kernel centred on the pixel. Samples are
    /// pulled toward the pixel centre and spill one pixel into each neighbour,
    /// giving smoother edge reconstruction than the box at no extra cost. This
    /// is the default.
    #[default]
    Tent,
}

/// Warps a uniform `u` in `[0, 1)` into a radius-one tent offset in `[-1, 1]`
/// via the tent's inverse `CDF`: the triangular density `1 - |x|` integrates to
/// a pair of parabolas whose inverse is a shifted square root on each half.
#[must_use]
fn tent_offset(u: f32) -> f32 {
    // Split at the tent's midpoint. Each half inverts a parabolic `CDF`.
    if u < 0.5 {
        (2.0 * u).sqrt() - 1.0
    } else {
        1.0 - (2.0 - 2.0 * u).sqrt()
    }
}

impl PixelFilter {
    /// Warps a uniform `[0, 1)^2` jitter sample into a sub-pixel position whose
    /// density matches this filter, expressed in pixel units relative to the
    /// pixel's lower corner.
    ///
    /// For [`PixelFilter::Box`] the sample passes through unchanged, keeping the
    /// position in `[0, 1)`. For [`PixelFilter::Tent`] the result is centred on
    /// the pixel midpoint `0.5` and spans `[-0.5, 1.5]`, so the kernel reaches
    /// one pixel into each neighbour.
    #[must_use]
    pub fn warp(self, jitter: Sample2) -> Sample2 {
        match self {
            PixelFilter::Box => jitter,
            PixelFilter::Tent => Sample2 {
                x: 0.5 + tent_offset(jitter.x),
                y: 0.5 + tent_offset(jitter.y),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_filter_is_the_identity() {
        let j = Sample2 { x: 0.3, y: 0.8 };
        let w = PixelFilter::Box.warp(j);
        assert_eq!(w.x.to_bits(), j.x.to_bits());
        assert_eq!(w.y.to_bits(), j.y.to_bits());
    }

    #[test]
    fn tent_warp_maps_endpoints_and_centre() {
        // u = 0 -> left edge of support, u = 0.5 -> centre, u -> 1 -> right.
        assert!((tent_offset(0.0) - (-1.0)).abs() < 1e-6);
        assert!(tent_offset(0.5).abs() < 1e-6);
        assert!((tent_offset(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tent_warp_is_monotonic_and_bounded() {
        let mut prev = f32::NEG_INFINITY;
        for i in 0..=1000 {
            let u = (i as f32) / 1000.0;
            let x = tent_offset(u);
            assert!(
                (-1.0 - 1e-6..=1.0 + 1e-6).contains(&x),
                "offset out of range: {x}"
            );
            assert!(x >= prev - 1e-6, "offset must be non-decreasing");
            prev = x;
        }
    }

    #[test]
    fn tent_warp_recovers_the_triangular_density() {
        // Bin the warped offsets of a uniform grid and confirm the histogram
        // follows the tent profile 1 - |x| (peak at 0, zero at the edges).
        let bins = 20usize;
        let mut counts = [0u32; 20];
        let samples = 200_000u32;
        for i in 0..samples {
            let u = ((i as f32) + 0.5) / (samples as f32);
            let x = tent_offset(u); // in [-1, 1]
            let b = (((x + 1.0) * 0.5) * bins as f32) as usize;
            counts[b.min(bins - 1)] += 1;
        }
        // The centre bins must hold markedly more mass than the edge bins.
        let edge = counts[0] + counts[bins - 1];
        let centre = counts[bins / 2 - 1] + counts[bins / 2];
        assert!(
            centre > edge * 3,
            "tent centre density {centre} should dwarf the edges {edge}"
        );
    }

    #[test]
    fn tent_is_symmetric_about_the_pixel_centre() {
        // Warped positions are mirror images about 0.5 for mirrored inputs.
        let a = PixelFilter::Tent.warp(Sample2 { x: 0.2, y: 0.7 });
        let b = PixelFilter::Tent.warp(Sample2 { x: 0.8, y: 0.3 });
        assert!(((a.x - 0.5) + (b.x - 0.5)).abs() < 1e-6);
        assert!(((a.y - 0.5) + (b.y - 0.5)).abs() < 1e-6);
    }

    #[test]
    fn default_filter_is_tent() {
        assert_eq!(PixelFilter::default(), PixelFilter::Tent);
    }
}
