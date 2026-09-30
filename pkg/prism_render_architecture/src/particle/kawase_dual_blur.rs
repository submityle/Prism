//! Dual `Kawase` blur: the iterative down/up filter chain from Marius
//! Bjorge's "Bandwidth-Efficient Rendering" (`SIGGRAPH` 2015).
//!
//! A dual `Kawase` blur builds a wide, smooth gaussian-like blur out of a
//! handful of very cheap bilinear taps. Instead of one large separable kernel
//! it walks the image *down* a resolution pyramid with a 5-tap "center plus
//! four diagonals" filter, then walks it back *up* with an 8-tap `tent`, each
//! pass sampling at a fixed sub-`texel` diagonal offset (classically half a
//! `texel`, `0.5px`). The genius of the scheme is that hardware bilinear
//! filtering does most of the averaging, so a small number of taps produces a
//! blur whose spatial support doubles with every down pass.
//!
//! # Distinct from [`super::bloom_upsample`]
//!
//! This module is deliberately *not* the bloom progressive-upsample composite:
//!
//! * Bloom upsamples a pre-built `mip` pyramid with a separable `[1, 2, 1]`
//!   `tent` (the `tent_filter_9tap`) and *adds* each level with a geometric
//!   per-`mip` weight to synthesize a halo. It never re-derives the pyramid.
//! * Dual `Kawase` *generates* its own pyramid on the fly with the fixed
//!   diagonal-offset down/up kernels and does **not** additively composite
//!   levels; each up pass fully *replaces* the running image at the next-finer
//!   resolution. The down kernel (center times four plus four diagonal taps,
//!   normalized by eight) and the up kernel (four edge taps plus four
//!   double-weighted diagonal taps, normalized by twelve) are the signature of
//!   the `Kawase` scheme and appear nowhere in the bloom path.
//!
//! To keep the two contracts independent this file defines its own
//! [`KawaseImage`] and never touches the bloom `MipImage`.
//!
//! # Determinism
//!
//! The determinism-locked contract layer forbids transcendental functions.
//! Every value here is produced with `+ - * /`, `f32::floor` (for the bilinear
//! integer split), and integer `div_ceil` (for the halved pyramid extents). The
//! bilinear sampler is hand-rolled with clamp-to-edge addressing so a future
//! `GPU` kernel reproduces the `CPU` result, and there is no `LUT` or table.

use crate::particle::gpu_layout::VEC4_STRIDE;
use alloc::vec::Vec;

/// Normalization divisor for the down kernel: its taps carry weights
/// `4 + 1 + 1 + 1 + 1 = 8`, so dividing by this makes a constant image survive
/// the downsample unchanged.
const DOWN_NORM: f32 = 8.0;

/// Weight of the center tap in the down kernel.
const DOWN_CENTER_WEIGHT: f32 = 4.0;

/// Normalization divisor for the up kernel: four edge taps of weight `1` plus
/// four diagonal taps of weight `2` sum to `4 + 8 = 12`, so a constant image
/// survives the upsample unchanged.
const UP_NORM: f32 = 12.0;

/// Weight of each diagonal tap in the up kernel; the four axis-aligned edge
/// taps carry an implicit weight of `1`.
const UP_DIAGONAL_WEIGHT: f32 = 2.0;

/// Number of scalar fields packed into the [`KawaseBlurParams`] `std430` block:
/// `offset` and `passes`.
const KAWASE_FIELD_COUNT: usize = 2;

/// Byte size of the `std430` packing of [`KawaseBlurParams`]: two scalars
/// rounded up to a whole `vec4` slot to honor the 16-byte `std430` base
/// alignment.
pub const KAWASE_BLUR_STD430_SIZE: usize = KAWASE_FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// Adds two `RGB` triples component-wise.
#[must_use]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales an `RGB` triple by a scalar.
#[must_use]
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Linear interpolation of two `RGB` triples: `a + (b - a) * t`.
#[must_use]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Clamps a (possibly negative) integer sample coordinate into `[0, extent)`,
/// the clamp-to-edge addressing mode the hardware sampler would use. Returns
/// `0` for a zero-extent axis so no caller can index an empty image.
#[must_use]
fn clamp_index(coord: i32, extent: usize) -> usize {
    if extent == 0 {
        return 0;
    }
    if coord < 0 {
        0
    } else {
        (coord as usize).min(extent - 1)
    }
}

/// A single linear `HDR` `RGB` image: a row-major grid of pixels.
///
/// This is the self-contained image type the dual `Kawase` chain operates on;
/// it is intentionally independent of the bloom `MipImage` so the two
/// post-process contracts do not entangle.
#[derive(Clone, Debug, PartialEq)]
pub struct KawaseImage {
    /// Width in `texels`.
    pub width: usize,
    /// Height in `texels`.
    pub height: usize,
    /// Row-major `width * height` linear `HDR` `RGB` pixels.
    pub pixels: Vec<[f32; 3]>,
}

impl KawaseImage {
    /// Builds an image from its dimensions and row-major pixels.
    ///
    /// The pixel count must equal `width * height`; a mismatch is a programming
    /// error and panics with a clear message rather than silently truncating.
    #[must_use]
    pub fn new(width: usize, height: usize, pixels: Vec<[f32; 3]>) -> Self {
        assert_eq!(
            pixels.len(),
            width.saturating_mul(height),
            "pixel count must equal width * height"
        );
        Self {
            width,
            height,
            pixels,
        }
    }

    /// A black image of the given dimensions (all pixels zero).
    #[must_use]
    pub fn black(width: usize, height: usize) -> Self {
        let pixels = alloc::vec![[0.0f32; 3]; width.saturating_mul(height)];
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Whether the image holds no `texels`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Reads the pixel at `(x, y)`, returning black for out-of-range
    /// coordinates so no caller can index past the grid.
    #[must_use]
    pub fn pixel(&self, x: usize, y: usize) -> [f32; 3] {
        if x >= self.width || y >= self.height {
            return [0.0; 3];
        }
        self.pixels[y * self.width + x]
    }

    /// Samples the image at normalized coordinates `(u, v)` in `[0, 1]` with a
    /// hand-rolled bilinear filter and clamp-to-edge addressing.
    ///
    /// `texel` centers sit at `(i + 0.5) / extent`, matching the hardware
    /// convention, so `u = (i + 0.5) / width` reads `texel` `i` exactly. Taps
    /// that fall outside the grid clamp to the nearest edge `texel`. An empty
    /// image samples to black.
    #[must_use]
    pub fn sample_bilinear(&self, u: f32, v: f32) -> [f32; 3] {
        if self.is_empty() {
            return [0.0; 3];
        }
        let fx = u * (self.width as f32) - 0.5;
        let fy = v * (self.height as f32) - 0.5;
        let fx0 = fx.floor();
        let fy0 = fy.floor();
        let tx = fx - fx0;
        let ty = fy - fy0;
        let base_x = fx0 as i32;
        let base_y = fy0 as i32;
        let x0 = clamp_index(base_x, self.width);
        let x1 = clamp_index(base_x + 1, self.width);
        let y0 = clamp_index(base_y, self.height);
        let y1 = clamp_index(base_y + 1, self.height);
        let p00 = self.pixel(x0, y0);
        let p10 = self.pixel(x1, y0);
        let p01 = self.pixel(x0, y1);
        let p11 = self.pixel(x1, y1);
        let top = lerp3(p00, p10, tx);
        let bot = lerp3(p01, p11, tx);
        lerp3(top, bot, ty)
    }

    /// Downsamples to half resolution (each axis `div_ceil` by two) with the
    /// `Kawase` down kernel: the center `texel` weighted by four plus four
    /// diagonal taps at `+/- offset` `texels`, all normalized by
    /// [`DOWN_NORM`].
    ///
    /// `offset` is the diagonal tap distance in source `texels`; the classic
    /// scheme uses `0.5`. An `offset` of `0` collapses every tap onto the
    /// center, so a constant image is reproduced exactly. An empty image
    /// downsamples to an empty image.
    #[must_use]
    pub fn downsample(&self, offset: f32) -> KawaseImage {
        let dst_w = self.width.div_ceil(2);
        let dst_h = self.height.div_ceil(2);
        if self.is_empty() {
            return KawaseImage::black(dst_w, dst_h);
        }
        let hp_u = offset / (self.width as f32);
        let hp_v = offset / (self.height as f32);
        let mut pixels = Vec::with_capacity(dst_w.saturating_mul(dst_h));
        for oy in 0..dst_h {
            let cv = (oy as f32 + 0.5) / (dst_h as f32);
            for ox in 0..dst_w {
                let cu = (ox as f32 + 0.5) / (dst_w as f32);
                let mut acc = scale3(self.sample_bilinear(cu, cv), DOWN_CENTER_WEIGHT);
                acc = add3(acc, self.sample_bilinear(cu - hp_u, cv - hp_v));
                acc = add3(acc, self.sample_bilinear(cu + hp_u, cv + hp_v));
                acc = add3(acc, self.sample_bilinear(cu + hp_u, cv - hp_v));
                acc = add3(acc, self.sample_bilinear(cu - hp_u, cv + hp_v));
                pixels.push(scale3(acc, 1.0 / DOWN_NORM));
            }
        }
        KawaseImage::new(dst_w, dst_h, pixels)
    }

    /// Upsamples to `(target_w, target_h)` with the `Kawase` up kernel: an
    /// 8-tap `tent` of four axis-aligned edge taps at `+/- 2 * offset` and four
    /// diagonal taps at `+/- offset` weighted by [`UP_DIAGONAL_WEIGHT`], all
    /// normalized by [`UP_NORM`].
    ///
    /// `offset` is the diagonal tap distance in source `texels`. An `offset`
    /// of `0` collapses every tap onto the sample point, so a constant image is
    /// reproduced exactly. An empty source or a zero-sized target yields a
    /// black target image.
    #[must_use]
    pub fn upsample(&self, target_w: usize, target_h: usize, offset: f32) -> KawaseImage {
        if self.is_empty() || target_w == 0 || target_h == 0 {
            return KawaseImage::black(target_w, target_h);
        }
        let hp_u = offset / (self.width as f32);
        let hp_v = offset / (self.height as f32);
        let mut pixels = Vec::with_capacity(target_w.saturating_mul(target_h));
        for oy in 0..target_h {
            let cv = (oy as f32 + 0.5) / (target_h as f32);
            for ox in 0..target_w {
                let cu = (ox as f32 + 0.5) / (target_w as f32);
                let mut acc = self.sample_bilinear(cu - 2.0 * hp_u, cv);
                acc = add3(acc, self.sample_bilinear(cu + 2.0 * hp_u, cv));
                acc = add3(acc, self.sample_bilinear(cu, cv - 2.0 * hp_v));
                acc = add3(acc, self.sample_bilinear(cu, cv + 2.0 * hp_v));
                acc = add3(
                    acc,
                    scale3(
                        self.sample_bilinear(cu - hp_u, cv - hp_v),
                        UP_DIAGONAL_WEIGHT,
                    ),
                );
                acc = add3(
                    acc,
                    scale3(
                        self.sample_bilinear(cu + hp_u, cv - hp_v),
                        UP_DIAGONAL_WEIGHT,
                    ),
                );
                acc = add3(
                    acc,
                    scale3(
                        self.sample_bilinear(cu - hp_u, cv + hp_v),
                        UP_DIAGONAL_WEIGHT,
                    ),
                );
                acc = add3(
                    acc,
                    scale3(
                        self.sample_bilinear(cu + hp_u, cv + hp_v),
                        UP_DIAGONAL_WEIGHT,
                    ),
                );
                pixels.push(scale3(acc, 1.0 / UP_NORM));
            }
        }
        KawaseImage::new(target_w, target_h, pixels)
    }

    /// Runs the full dual `Kawase` chain: `passes` successive [`downsample`]s
    /// followed by [`upsample`]s that retrace the recorded resolutions back to
    /// the original dimensions.
    ///
    /// The extent of every level before each down pass is recorded so the up
    /// chain can reproduce them exactly (a `div_ceil` halving is not reversible
    /// by another `div_ceil`, so the target extents must be remembered). With
    /// `passes == 0`, or an empty image, the input is returned unchanged.
    ///
    /// [`downsample`]: KawaseImage::downsample
    /// [`upsample`]: KawaseImage::upsample
    #[must_use]
    pub fn dual_blur(&self, passes: usize, offset: f32) -> KawaseImage {
        if passes == 0 || self.is_empty() {
            return self.clone();
        }
        let mut sizes: Vec<(usize, usize)> = Vec::with_capacity(passes);
        let mut current = self.clone();
        for _ in 0..passes {
            sizes.push((current.width, current.height));
            current = current.downsample(offset);
        }
        for &(tw, th) in sizes.iter().rev() {
            current = current.upsample(tw, th, offset);
        }
        current
    }
}

/// The `std430`-packable parameter block a future `GPU` dual `Kawase` kernel
/// binds: the sub-`texel` diagonal `offset` and the pyramid `passes` count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KawaseBlurParams {
    /// Diagonal tap distance in source `texels` (classically `0.5`), clamped to
    /// be non-negative.
    pub offset: f32,
    /// Number of down/up pyramid passes.
    pub passes: u32,
}

impl KawaseBlurParams {
    /// Builds a parameter block, clamping `offset` to be non-negative so a
    /// negative diagonal distance can never invert the kernel.
    #[must_use]
    pub fn new(offset: f32, passes: u32) -> Self {
        Self {
            offset: offset.max(0.0),
            passes,
        }
    }

    /// Packs the parameters into their `std430` byte block: `offset` as a
    /// little-endian `f32` followed by `passes` as a little-endian `u32`, with
    /// the remaining `vec4` tail left zero.
    #[must_use]
    pub fn to_std430(&self) -> Vec<u8> {
        let mut bytes = alloc::vec![0u8; KAWASE_BLUR_STD430_SIZE];
        bytes[0..4].copy_from_slice(&self.offset.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.passes.to_le_bytes());
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::gpu_layout::storage_bytes;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    fn uniform(width: usize, height: usize, value: f32) -> KawaseImage {
        KawaseImage::new(
            width,
            height,
            alloc::vec![[value, value, value]; width * height],
        )
    }

    #[test]
    fn new_holds_dimensions_and_pixels() {
        let img = uniform(3, 2, 0.5);
        assert_eq!(img.width, 3);
        assert_eq!(img.height, 2);
        assert_eq!(img.pixels.len(), 6);
    }

    #[test]
    fn black_is_all_zero() {
        let img = KawaseImage::black(4, 4);
        assert_eq!(img.pixels.len(), 16);
        for p in &img.pixels {
            assert!(approx3(*p, [0.0, 0.0, 0.0]));
        }
    }

    #[test]
    fn is_empty_detects_zero_dimensions() {
        assert!(KawaseImage::black(0, 4).is_empty());
        assert!(KawaseImage::black(4, 0).is_empty());
        assert!(!KawaseImage::black(1, 1).is_empty());
    }

    #[test]
    fn pixel_out_of_range_is_black() {
        let img = uniform(2, 2, 1.0);
        assert!(approx3(img.pixel(2, 0), [0.0, 0.0, 0.0]));
        assert!(approx3(img.pixel(0, 2), [0.0, 0.0, 0.0]));
    }

    #[test]
    fn pixel_reads_row_major() {
        let img = KawaseImage::new(
            2,
            2,
            alloc::vec![
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
        );
        assert!(approx3(img.pixel(0, 0), [1.0, 0.0, 0.0]));
        assert!(approx3(img.pixel(1, 0), [2.0, 0.0, 0.0]));
        assert!(approx3(img.pixel(0, 1), [3.0, 0.0, 0.0]));
        assert!(approx3(img.pixel(1, 1), [4.0, 0.0, 0.0]));
    }

    #[test]
    fn sample_bilinear_hits_texel_centers_exactly() {
        let img = KawaseImage::new(2, 1, alloc::vec![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]);
        // texel centers at u = 0.25 and u = 0.75.
        assert!(approx3(img.sample_bilinear(0.25, 0.5), [0.0, 0.0, 0.0]));
        assert!(approx3(img.sample_bilinear(0.75, 0.5), [4.0, 0.0, 0.0]));
    }

    #[test]
    fn sample_bilinear_interpolates_midpoint() {
        let img = KawaseImage::new(2, 1, alloc::vec![[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]);
        // u = 0.5 sits halfway between the two texel centers.
        assert!(approx3(img.sample_bilinear(0.5, 0.5), [2.0, 0.0, 0.0]));
    }

    #[test]
    fn sample_bilinear_averages_a_2x2_at_the_center() {
        let img = KawaseImage::new(
            2,
            2,
            alloc::vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
        );
        // The shared corner of all four texels averages them: (0+2+4+6)/4 = 3.
        assert!(approx3(img.sample_bilinear(0.5, 0.5), [3.0, 0.0, 0.0]));
    }

    #[test]
    fn sample_bilinear_clamps_out_of_range_coordinates() {
        let img = KawaseImage::new(2, 1, alloc::vec![[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
        // Far left clamps to texel 0, far right clamps to texel 1.
        assert!(approx3(img.sample_bilinear(-1.0, 0.5), [1.0, 0.0, 0.0]));
        assert!(approx3(img.sample_bilinear(2.0, 0.5), [5.0, 0.0, 0.0]));
    }

    #[test]
    fn sample_bilinear_on_empty_is_black() {
        let img = KawaseImage::black(0, 0);
        assert!(approx3(img.sample_bilinear(0.5, 0.5), [0.0, 0.0, 0.0]));
    }

    #[test]
    fn downsample_halves_even_dimensions() {
        let img = uniform(8, 6, 1.0);
        let down = img.downsample(0.5);
        assert_eq!(down.width, 4);
        assert_eq!(down.height, 3);
    }

    #[test]
    fn downsample_uses_div_ceil_on_odd_dimensions() {
        let img = uniform(5, 3, 1.0);
        let down = img.downsample(0.5);
        assert_eq!(down.width, 5usize.div_ceil(2));
        assert_eq!(down.height, 3usize.div_ceil(2));
        assert_eq!(down.width, 3);
        assert_eq!(down.height, 2);
    }

    #[test]
    fn downsample_preserves_a_constant_image() {
        let img = uniform(8, 8, 0.7);
        let down = img.downsample(0.5);
        for p in &down.pixels {
            assert!(approx3(*p, [0.7, 0.7, 0.7]));
        }
    }

    #[test]
    fn downsample_zero_offset_preserves_constant() {
        // With offset 0 every tap collapses onto the center; a constant image
        // stays constant because the tap weights sum to DOWN_NORM.
        let img = uniform(8, 8, 2.5);
        let down = img.downsample(0.0);
        for p in &down.pixels {
            assert!(approx3(*p, [2.5, 2.5, 2.5]));
        }
    }

    #[test]
    fn downsample_empty_is_empty() {
        let img = KawaseImage::black(0, 4);
        let down = img.downsample(0.5);
        assert!(down.is_empty());
    }

    #[test]
    fn downsample_one_by_one_stays_one_by_one() {
        let img = uniform(1, 1, 3.0);
        let down = img.downsample(0.5);
        assert_eq!(down.width, 1);
        assert_eq!(down.height, 1);
        assert!(approx3(down.pixel(0, 0), [3.0, 3.0, 3.0]));
    }

    #[test]
    fn upsample_returns_target_resolution() {
        let img = uniform(4, 3, 1.0);
        let up = img.upsample(8, 6, 0.5);
        assert_eq!(up.width, 8);
        assert_eq!(up.height, 6);
    }

    #[test]
    fn upsample_preserves_a_constant_image() {
        let img = uniform(4, 4, 1.25);
        let up = img.upsample(8, 8, 0.5);
        for p in &up.pixels {
            assert!(approx3(*p, [1.25, 1.25, 1.25]));
        }
    }

    #[test]
    fn upsample_zero_offset_preserves_constant() {
        let img = uniform(4, 4, 0.9);
        let up = img.upsample(8, 8, 0.0);
        for p in &up.pixels {
            assert!(approx3(*p, [0.9, 0.9, 0.9]));
        }
    }

    #[test]
    fn upsample_empty_source_is_black_target() {
        let img = KawaseImage::black(0, 0);
        let up = img.upsample(4, 4, 0.5);
        assert_eq!(up.width, 4);
        assert_eq!(up.height, 4);
        for p in &up.pixels {
            assert!(approx3(*p, [0.0, 0.0, 0.0]));
        }
    }

    #[test]
    fn upsample_zero_target_is_empty() {
        let img = uniform(4, 4, 1.0);
        let up = img.upsample(0, 0, 0.5);
        assert!(up.is_empty());
    }

    #[test]
    fn dual_blur_passes_zero_is_identity() {
        let img = uniform(8, 8, 0.42);
        let out = img.dual_blur(0, 0.5);
        assert_eq!(out, img);
    }

    #[test]
    fn dual_blur_returns_original_resolution() {
        let img = uniform(9, 7, 1.0);
        let out = img.dual_blur(2, 0.5);
        assert_eq!(out.width, 9);
        assert_eq!(out.height, 7);
    }

    #[test]
    fn dual_blur_preserves_a_constant_image() {
        let img = uniform(16, 16, 0.6);
        let out = img.dual_blur(3, 0.5);
        for p in &out.pixels {
            assert!(approx3(*p, [0.6, 0.6, 0.6]));
        }
    }

    #[test]
    fn dual_blur_empty_is_empty() {
        let img = KawaseImage::black(0, 5);
        let out = img.dual_blur(3, 0.5);
        assert!(out.is_empty());
    }

    #[test]
    fn dual_blur_spreads_a_central_impulse() {
        // A single bright texel must diffuse into its neighborhood: the center
        // darkens as its energy leaks outward and at least one neighbor lights
        // up, while the total stays finite and positive.
        let width = 8;
        let height = 8;
        let mut pixels = alloc::vec![[0.0f32; 3]; width * height];
        let cx = 4;
        let cy = 4;
        pixels[cy * width + cx] = [1.0, 1.0, 1.0];
        let img = KawaseImage::new(width, height, pixels);
        let out = img.dual_blur(2, 0.5);

        let center = out.pixel(cx, cy);
        assert!(center[0] < 1.0 - CMP_EPS);

        let neighbor = out.pixel(cx + 1, cy);
        assert!(neighbor[0] > CMP_EPS);

        let mut total = 0.0f32;
        for p in &out.pixels {
            assert!(p[0].is_finite());
            total += p[0];
        }
        assert!(total > CMP_EPS);
    }

    #[test]
    fn params_new_clamps_negative_offset() {
        let p = KawaseBlurParams::new(-2.0, 3);
        assert!(approx(p.offset, 0.0));
        assert_eq!(p.passes, 3);
    }

    #[test]
    fn params_to_std430_round_trips_the_fields() {
        let p = KawaseBlurParams::new(0.5, 4);
        let bytes = p.to_std430();
        assert_eq!(bytes.len(), KAWASE_BLUR_STD430_SIZE);
        let mut off = [0u8; 4];
        off.copy_from_slice(&bytes[0..4]);
        assert!(approx(f32::from_le_bytes(off), 0.5));
        let mut passes = [0u8; 4];
        passes.copy_from_slice(&bytes[4..8]);
        assert_eq!(u32::from_le_bytes(passes), 4);
    }

    #[test]
    fn std430_size_is_a_whole_vec4_block() {
        assert_eq!(KAWASE_BLUR_STD430_SIZE, storage_bytes(VEC4_STRIDE, 1));
        assert_eq!(KAWASE_BLUR_STD430_SIZE % VEC4_STRIDE, 0);
    }
}
