//! **Chroma subsampling** (4:4:4 / 4:2:2 / 4:2:0) for the chroma planes of a
//! [`YCbCr`](super::ycbcr) texture.
//!
//! The eye resolves luminance detail far better than colour detail, so almost
//! every image / video codec a texture pipeline ingests (JPEG, H.264/HEVC, the
//! `YUV` planes behind streamed and cinematic textures) keeps the luma plane at
//! full resolution but stores the two chroma planes at reduced resolution:
//!
//! * **4:4:4** -- no subsampling (chroma at full resolution),
//! * **4:2:2** -- chroma halved horizontally,
//! * **4:2:0** -- chroma halved in both axes.
//!
//! This module resamples a single chroma plane (one of `Cb` / `Cr`, after the
//! [`rgb_to_ycbcr`](super::rgb_to_ycbcr) transform) between full and reduced
//! resolution. [`downsample`] is a **co-sited box average** of each 1x2 / 2x2
//! source footprint (edges clamp to the valid taps for odd sizes), and
//! [`upsample`] is the matching **nearest / replication** reconstruction, so the
//! two are exact inverses on data that is constant within each subsample block.
//! The luma plane is never touched.
//!
//! Subsampling is lossy in general, but the structure gives strong anti-fake
//! oracles: 4:4:4 is the identity both ways; a plane that is already
//! block-constant (for instance one produced by [`upsample`]) round-trips
//! through [`downsample`] back to the exact low-resolution plane; and the box
//! average equals an independent mean of the covered taps. Everything is
//! deterministic analytic `f32` arithmetic (no transcendentals, no AI/ML), so a
//! CPU golden matches a GPU twin to floating-point tolerance.
//!
//! # References
//! * ITU-R BT.601 / BT.709 (chroma sampling structures).
//! * Poynton, *Digital Video and HD*, 2nd ed., chroma subsampling.

use alloc::vec;
use alloc::vec::Vec;

/// How the chroma planes are subsampled relative to the full-resolution luma.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:4:4 -- chroma kept at full resolution (no subsampling).
    Yuv444,
    /// 4:2:2 -- chroma halved horizontally, full vertical resolution.
    Yuv422,
    /// 4:2:0 -- chroma halved in both axes.
    Yuv420,
}

impl ChromaSubsampling {
    /// The horizontal / vertical decimation factors `(x, y)` the mode applies to
    /// a chroma plane.
    #[inline]
    #[must_use]
    pub const fn factors(self) -> (usize, usize) {
        match self {
            Self::Yuv444 => (1, 1),
            Self::Yuv422 => (2, 1),
            Self::Yuv420 => (2, 2),
        }
    }

    /// The dimensions a full-resolution chroma plane of size `(width, height)`
    /// decimates to under this mode (ceiling division, so odd sizes keep the
    /// trailing sample).
    #[inline]
    #[must_use]
    pub const fn reduced_size(self, width: usize, height: usize) -> (usize, usize) {
        let (fx, fy) = self.factors();
        (width.div_ceil(fx), height.div_ceil(fy))
    }
}

/// A single-channel chroma plane in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChromaPlane {
    /// Row-major samples, `width * height` of them.
    pub data: Vec<f32>,
    /// Plane width in samples.
    pub width: usize,
    /// Plane height in samples.
    pub height: usize,
}

impl ChromaPlane {
    /// Wrap row-major samples as a plane.
    ///
    /// # Panics
    /// Panics unless `data.len() == width * height`.
    #[must_use]
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height, "plane size mismatch");
        Self {
            data,
            width,
            height,
        }
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
}

/// Box-average a full-resolution chroma plane down to its subsampled size.
///
/// Each output sample is the mean of the `fx x fy` source footprint it covers;
/// footprints clamp to the valid taps at the right / bottom edge when the input
/// dimension is odd, so no out-of-range samples are read.
#[must_use]
pub fn downsample(plane: &ChromaPlane, mode: ChromaSubsampling) -> ChromaPlane {
    let (fx, fy) = mode.factors();
    let (ow, oh) = mode.reduced_size(plane.width, plane.height);
    let mut out = vec![0.0f32; ow * oh];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            for dy in 0..fy {
                let sy = oy * fy + dy;
                if sy >= plane.height {
                    break;
                }
                for dx in 0..fx {
                    let sx = ox * fx + dx;
                    if sx >= plane.width {
                        break;
                    }
                    sum += plane.get(sx, sy);
                    count += 1.0;
                }
            }
            out[oy * ow + ox] = sum / count;
        }
    }
    ChromaPlane::new(out, ow, oh)
}

/// Reconstruct a full-resolution chroma plane from its subsampled form by
/// nearest / replication upsampling.
///
/// Each reduced sample is replicated across the `fx x fy` block it represents,
/// up to the requested `(full_width, full_height)`. This is the exact inverse of
/// [`downsample`] for data that is constant within each subsample block.
#[must_use]
pub fn upsample(
    plane: &ChromaPlane,
    mode: ChromaSubsampling,
    full_width: usize,
    full_height: usize,
) -> ChromaPlane {
    let (fx, fy) = mode.factors();
    let mut out = vec![0.0f32; full_width * full_height];
    for y in 0..full_height {
        let sy = (y / fy).min(plane.height.saturating_sub(1));
        for x in 0..full_width {
            let sx = (x / fx).min(plane.width.saturating_sub(1));
            out[y * full_width + x] = plane.get(sx, sy);
        }
    }
    ChromaPlane::new(out, full_width, full_height)
}

#[cfg(test)]
mod tests {
    use super::{downsample, upsample, ChromaPlane, ChromaSubsampling};
    use alloc::vec::Vec;

    fn plane(width: usize, height: usize) -> ChromaPlane {
        let data: Vec<f32> = (0..width * height)
            .map(|i| (i as f32) * 0.013 - 0.2)
            .collect();
        ChromaPlane::new(data, width, height)
    }

    /// 4:4:4 is a no-op in both directions.
    #[test]
    fn full_resolution_is_identity() {
        let p = plane(4, 3);
        let d = downsample(&p, ChromaSubsampling::Yuv444);
        assert_eq!(d, p, "downsample identity");
        let u = upsample(&d, ChromaSubsampling::Yuv444, 4, 3);
        assert_eq!(u, p, "upsample identity");
    }

    /// Reduced dimensions follow ceiling division for the three modes.
    #[test]
    fn reduced_dimensions_are_ceiling() {
        assert_eq!(ChromaSubsampling::Yuv444.reduced_size(5, 3), (5, 3));
        assert_eq!(ChromaSubsampling::Yuv422.reduced_size(5, 3), (3, 3));
        assert_eq!(ChromaSubsampling::Yuv420.reduced_size(5, 3), (3, 2));
        assert_eq!(ChromaSubsampling::Yuv420.reduced_size(4, 4), (2, 2));
    }

    /// A constant chroma plane survives downsample then upsample exactly, for
    /// every mode (constant blocks are the exact-inverse case).
    #[test]
    fn constant_plane_round_trips() {
        for mode in [
            ChromaSubsampling::Yuv444,
            ChromaSubsampling::Yuv422,
            ChromaSubsampling::Yuv420,
        ] {
            let p = ChromaPlane::new(alloc::vec![0.375f32; 6 * 4], 6, 4);
            let d = downsample(&p, mode);
            let u = upsample(&d, mode, 6, 4);
            for (a, b) in u.data.iter().zip(p.data.iter()) {
                assert!((a - b).abs() < 1e-7, "constant {a} vs {b} ({mode:?})");
            }
        }
    }

    /// A block-constant plane (built by upsampling an arbitrary reduced plane)
    /// downsamples back to that exact reduced plane -- the inverse-recovery
    /// anti-fake oracle.
    #[test]
    fn block_constant_recovers_reduced_plane() {
        for (mode, rw, rh) in [
            (ChromaSubsampling::Yuv422, 3, 4),
            (ChromaSubsampling::Yuv420, 3, 2),
        ] {
            let reduced = plane(rw, rh);
            let (fx, fy) = mode.factors();
            let full = upsample(&reduced, mode, rw * fx, rh * fy);
            let back = downsample(&full, mode);
            assert_eq!(back.width, rw);
            assert_eq!(back.height, rh);
            for (a, b) in back.data.iter().zip(reduced.data.iter()) {
                assert!((a - b).abs() < 1e-6, "recovered {a} vs {b} ({mode:?})");
            }
        }
    }

    /// The 4:2:0 box average equals an independent mean of the covered 2x2 taps.
    #[test]
    fn box_average_matches_manual_mean() {
        let p = plane(4, 4);
        let d = downsample(&p, ChromaSubsampling::Yuv420);
        for oy in 0..2 {
            for ox in 0..2 {
                let mean = 0.25
                    * (p.get(ox * 2, oy * 2)
                        + p.get(ox * 2 + 1, oy * 2)
                        + p.get(ox * 2, oy * 2 + 1)
                        + p.get(ox * 2 + 1, oy * 2 + 1));
                assert!((d.get(ox, oy) - mean).abs() < 1e-6, "mean mismatch");
            }
        }
    }

    /// An odd-width 4:2:2 column clamps to the single trailing tap instead of
    /// reading past the edge.
    #[test]
    fn odd_edge_clamps_to_valid_taps() {
        // width 3 -> reduced width 2; the last output column averages only the
        // single remaining source column (index 2).
        let p = plane(3, 2);
        let d = downsample(&p, ChromaSubsampling::Yuv422);
        assert_eq!((d.width, d.height), (2, 2));
        for y in 0..2 {
            let edge = p.get(2, y);
            assert!((d.get(1, y) - edge).abs() < 1e-6, "edge tap {y}");
        }
    }
}
