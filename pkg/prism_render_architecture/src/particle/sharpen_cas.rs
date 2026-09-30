//! `AMD` `FidelityFX` Contrast-Adaptive Sharpening (`CAS`) — the `CPU`
//! gold-standard contract for the particle stylization stack (design §16-§21).
//!
//! After a particle pass has been upscaled, temporally reprojected, or simply
//! blurred by its own alpha coverage, the composited result often reads a touch
//! soft. `CAS` is the production answer to that: a single-tap, locally adaptive
//! sharpen that raises apparent detail without the ringing halos of an
//! unsharp-mask and without over-sharpening already high-contrast regions.
//! `AMD` ships it as `FidelityFX CAS`; the same math appears in engine post
//! stacks as the "sharpen" knob that pairs with an upscaler. This module owns
//! the `CPU`-verifiable maths of that contract and packs its one parameter into
//! the `std430` block a future `GPU` sharpen kernel binds.
//!
//! # The algorithm
//!
//! For each output texel `CAS` reads the center and its four cross neighbors
//! (up / down / left / right — the diagonals are deliberately ignored, which is
//! what keeps it single-pass and cheap). Per color channel it forms the local
//! `min`/`max`, derives an adaptive amplitude that *shrinks* near black and
//! near white (so the sharpen never pushes values past the displayable range),
//! scales that by a sharpness-controlled negative peak weight, and blends the
//! cross neighbors against the center with that weight. The blend is
//! energy-normalized by its own denominator so a flat neighborhood is returned
//! unchanged.
//!
//! # Strict scope
//!
//! This file is *only* the `CAS` sharpen kernel. It is **not** a bright-pass or
//! blur pyramid (that is [`super::bloom_threshold`] and the bloom upsample
//! passes), it is **not** a `2x2` reduction / `mip`-chain builder (that is
//! [`super::depth_downsample`]), and it is **not** an edge-detection /
//! `Sobel` magnitude contract (that is [`super::edge_detect`]). It neither
//! downsamples nor upsamples: input and output share the same resolution.
//!
//! # Determinism
//!
//! The determinism-locked contract layer forbids transcendental functions
//! (`sin`/`cos`/`exp`/`ln`/`powf`). `CAS` needs only a reciprocal (division), a
//! single `sqrt` for the amplitude, hand-rolled `min`/`max`/`mix`, and
//! `f32::clamp`, every division guarded against a zero denominator, so a future
//! `GPU` kernel reproduces the `CPU` result bit for bit. Only
//! [`super::gpu_layout`] is imported.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Number of scalar fields packed into the [`CasParams`] `std430` block: just
/// `sharpness`.
const SHARPEN_FIELD_COUNT: usize = 1;

/// Byte size of the `std430` packing of [`CasParams`]: the single scalar rounded
/// up to a whole `vec4` slot so the block honors the 16-byte `std430` base
/// alignment.
pub const SHARPEN_STD430_SIZE: usize = SHARPEN_FIELD_COUNT.div_ceil(4) * VEC4_STRIDE;

/// Denominators with magnitude below this are treated as (near) zero so
/// evaluation falls back to a defined result instead of dividing by zero or
/// propagating `NaN`. It doubles as the near-black guard on the local maximum.
const MIN_DENOM: f32 = 1e-6;

/// Low endpoint of the `CAS` peak-weight interpolation (softest sharpen). The
/// reciprocal of this is the smallest-magnitude negative peak weight.
const PEAK_SOFT: f32 = 8.0;

/// High endpoint of the `CAS` peak-weight interpolation (sharpest). The
/// reciprocal of this is the largest-magnitude negative peak weight.
const PEAK_HARD: f32 = 5.0;

/// Absolute tolerance for the `f32` equality comparisons used by the tests;
/// direct `==` on floating point is intentionally avoided.
#[cfg(test)]
const CMP_EPS: f32 = 1e-5;

/// The smaller of two `f32` values, written as an explicit branch so no
/// intrinsic `min` is relied upon.
#[must_use]
fn fmin(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

/// The larger of two `f32` values, written as an explicit branch so no
/// intrinsic `max` is relied upon.
#[must_use]
fn fmax(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// Linear interpolation `a + (b - a) * t`, the hand-rolled `mix` the `CAS` peak
/// weight uses. No clamping of `t` is performed; callers pass an already-clamped
/// sharpness.
#[must_use]
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Contrast-adaptive-sharpening parameters (design §16-§21).
///
/// `sharpness` runs `0..=1`: `0` is the softest sharpen (peak weight `-1/8`),
/// `1` is the sharpest (peak weight `-1/5`). Values outside the range are
/// clamped by [`CasParams::new`]; [`CasParams::sharpen_taps`] clamps again so a
/// directly-constructed out-of-range field is still safe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CasParams {
    /// Sharpen strength in `0..=1`.
    pub sharpness: f32,
}

impl CasParams {
    /// Builds a parameter set from a raw sharpness, clamping it into `0..=1`.
    #[must_use]
    pub fn new(sharpness: f32) -> Self {
        Self {
            sharpness: sharpness.clamp(0.0, 1.0),
        }
    }

    /// The negative peak blend weight for the clamped sharpness:
    /// `-1 / mix(8, 5, sharpness)`. More sharpness yields a larger-magnitude
    /// (more negative) weight, which subtracts more of the cross-neighbor
    /// average from the center.
    #[must_use]
    fn peak_weight(self) -> f32 {
        let sharp = self.sharpness.clamp(0.0, 1.0);
        -1.0 / mix(PEAK_SOFT, PEAK_HARD, sharp)
    }

    /// Sharpens one `RGB` texel from its `3x3` row-major neighborhood.
    ///
    /// `taps` is the neighborhood in row-major order with `taps[4]` the center;
    /// only the center and the four cross neighbors (`taps[1]`, `taps[3]`,
    /// `taps[5]`, `taps[7]`) participate — the diagonals are ignored, as in
    /// `AMD` `CAS`. Each channel is processed independently, the near-black and
    /// near-white amplitude protection is applied, and the result is clamped to
    /// `0..` (never negative). A flat neighborhood is returned unchanged.
    #[must_use]
    pub fn sharpen_taps(&self, taps: &[[f32; 3]; 9]) -> [f32; 3] {
        let up = taps[1];
        let down = taps[7];
        let left = taps[3];
        let right = taps[5];
        let center = taps[4];
        let peak = self.peak_weight();
        let mut out = [0.0_f32; 3];
        for (channel, slot) in out.iter_mut().enumerate() {
            *slot = sharpen_channel(
                peak,
                up[channel],
                down[channel],
                left[channel],
                right[channel],
                center[channel],
            );
        }
        out
    }

    /// Sharpens a whole `width * height` `RGB` image, gathering each texel's
    /// `3x3` neighborhood with clamp-to-edge boundary handling (the border
    /// replicates its nearest interior texel).
    ///
    /// Returns an empty vector when either dimension is zero or when `img` holds
    /// fewer than `width * height` texels, so a malformed call never panics.
    #[must_use]
    pub fn apply(&self, img: &[[f32; 3]], width: usize, height: usize) -> Vec<[f32; 3]> {
        let count = width.saturating_mul(height);
        let mut out = Vec::new();
        if width == 0 || height == 0 || img.len() < count {
            return out;
        }
        out.reserve(count);
        for y in 0..height {
            for x in 0..width {
                let taps = gather_taps(img, x, y, width, height);
                out.push(self.sharpen_taps(&taps));
            }
        }
        out
    }

    /// Packs the parameters into their `std430` byte block for a future `GPU`
    /// sharpen kernel. The single `sharpness` scalar fills the first slot; the
    /// padding tail stays zero so the block is a whole number of `vec4` slots.
    #[must_use]
    pub fn to_std430(&self) -> [u8; SHARPEN_STD430_SIZE] {
        let fields = [self.sharpness];
        let mut bytes = [0u8; SHARPEN_STD430_SIZE];
        for (slot, value) in bytes.chunks_exact_mut(4).zip(fields.iter()) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Total storage-buffer byte size the parameter block reserves on the `GPU`
    /// (a single `std430` element, clamped up to one element by the shared rule).
    #[must_use]
    pub fn gpu_storage_bytes(&self) -> usize {
        storage_bytes(SHARPEN_STD430_SIZE, 1)
    }
}

/// The per-channel `CAS` blend for a single color channel.
///
/// Forms the local `min`/`max` over the center and four cross neighbors,
/// derives the adaptive amplitude `sqrt(clamp(min(mn, 2 - mx) / mx, 0, 1))`
/// (which shrinks near black through the `mx` guard and near white through the
/// `2 - mx` term), scales it by the negative `peak` weight, and blends. The
/// energy-normalizing denominator keeps a flat neighborhood unchanged, and the
/// output is clamped to be non-negative.
#[must_use]
fn sharpen_channel(peak: f32, up: f32, down: f32, left: f32, right: f32, center: f32) -> f32 {
    let mn = fmin(center, fmin(fmin(up, down), fmin(left, right)));
    let mx = fmax(center, fmax(fmax(up, down), fmax(left, right)));
    let amp = if mx < MIN_DENOM {
        0.0
    } else {
        let ratio = fmin(mn, 2.0 - mx) / mx;
        ratio.clamp(0.0, 1.0).sqrt()
    };
    let w = amp * peak;
    let denom = 1.0 + 4.0 * w;
    let out = if denom.abs() < MIN_DENOM {
        center
    } else {
        (w * (up + down + left + right) + center) / denom
    };
    fmax(out, 0.0)
}

/// Clamps a coordinate stepped by `delta` (only `-1`, `0`, or `+1`) into the
/// valid `0..dim` range, replicating the edge texel. `dim` is guaranteed
/// non-zero by the [`CasParams::apply`] guard.
#[must_use]
fn clamp_coord(coord: usize, delta: i8, dim: usize) -> usize {
    let last = dim - 1;
    let shifted = if delta < 0 {
        coord.saturating_sub(1)
    } else if delta > 0 {
        coord + 1
    } else {
        coord
    };
    shifted.min(last)
}

/// Gathers the `3x3` row-major neighborhood of texel `(x, y)` with clamp-to-edge
/// boundary handling, so a border texel replicates its nearest interior sample.
#[must_use]
fn gather_taps(img: &[[f32; 3]], x: usize, y: usize, width: usize, height: usize) -> [[f32; 3]; 9] {
    let mut taps = [[0.0_f32; 3]; 9];
    for (row, &dy) in [-1_i8, 0, 1].iter().enumerate() {
        for (col, &dx) in [-1_i8, 0, 1].iter().enumerate() {
            let sx = clamp_coord(x, dx, width);
            let sy = clamp_coord(y, dy, height);
            taps[row * 3 + col] = img[sy * width + sx];
        }
    }
    taps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    /// Builds a `3x3` neighborhood where every channel of every tap equals the
    /// scalar it is given (center vs. the four cross neighbors); diagonals are
    /// filled with the center value since they are ignored.
    fn cross_window(center: f32, neighbor: f32) -> [[f32; 3]; 9] {
        let c = [center; 3];
        let n = [neighbor; 3];
        [c, n, c, n, c, n, c, n, c]
    }

    #[test]
    fn constant_neighborhood_is_identity() {
        let params = CasParams::new(0.5);
        let win = cross_window(0.42, 0.42);
        assert!(approx3(params.sharpen_taps(&win), [0.42; 3]));
    }

    #[test]
    fn constant_neighborhood_identity_at_zero_sharpness() {
        let params = CasParams::new(0.0);
        let win = cross_window(0.7, 0.7);
        assert!(approx3(params.sharpen_taps(&win), [0.7; 3]));
    }

    #[test]
    fn constant_neighborhood_identity_at_max_sharpness() {
        let params = CasParams::new(1.0);
        let win = cross_window(0.3, 0.3);
        assert!(approx3(params.sharpen_taps(&win), [0.3; 3]));
    }

    #[test]
    fn bright_center_is_enhanced() {
        // A local maximum: the center is pushed brighter still.
        let params = CasParams::new(0.5);
        let win = cross_window(0.6, 0.4);
        let out = params.sharpen_taps(&win);
        assert!(out[0] > 0.6);
    }

    #[test]
    fn dark_center_is_suppressed() {
        // A local minimum: the center is pushed darker.
        let params = CasParams::new(0.5);
        let win = cross_window(0.4, 0.6);
        let out = params.sharpen_taps(&win);
        assert!(out[0] < 0.4);
    }

    #[test]
    fn stronger_sharpness_sharpens_more() {
        let soft = CasParams::new(0.2);
        let hard = CasParams::new(1.0);
        let win = cross_window(0.6, 0.4);
        let out_soft = soft.sharpen_taps(&win)[0];
        let out_hard = hard.sharpen_taps(&win)[0];
        // Both enhance a bright center, but the harder setting pushes further.
        assert!(out_hard > out_soft);
        assert!(out_soft > 0.6);
    }

    #[test]
    fn near_black_protection_disables_sharpen() {
        // The whole neighborhood sits below the near-black guard, so the local
        // maximum is treated as zero: amplitude is zero and the center passes
        // through unchanged instead of being sharpened by the ratio.
        let params = CasParams::new(1.0);
        let center = 5e-7_f32;
        let win = cross_window(center, 1e-7);
        let out = params.sharpen_taps(&win);
        assert!(approx(out[0], center));
    }

    #[test]
    fn near_white_protection_limits_sharpen() {
        // Two windows with the *same* local min/max ratio, one pushed into the
        // near-white (`HDR` > 1) range where the `2 - mx` term shrinks the
        // amplitude. The relative sharpen of the near-white case is smaller.
        let params = CasParams::new(1.0);

        let center_hi = 1.8_f32;
        let hi = params.sharpen_taps(&cross_window(center_hi, 1.0))[0];
        let rel_hi = (hi - center_hi) / center_hi;

        let center_mid = 0.9_f32;
        let mid = params.sharpen_taps(&cross_window(center_mid, 0.5))[0];
        let rel_mid = (mid - center_mid) / center_mid;

        assert!(rel_hi < rel_mid);
        assert!(rel_hi > 0.0);
    }

    #[test]
    fn near_white_matches_hand_computed_value() {
        // sharpness = 1 -> peak = -1/5. mn = 1.0, mx = 1.8, 2 - mx = 0.2 picked,
        // ratio = 0.2/1.8, amp = sqrt(ratio), w = amp * -0.2.
        let params = CasParams::new(1.0);
        let out = params.sharpen_taps(&cross_window(1.8, 1.0))[0];

        let ratio = 0.2_f32 / 1.8_f32;
        let amp = ratio.sqrt();
        let w = amp * (-1.0_f32 / 5.0_f32);
        let expected = (w * 4.0 + 1.8) / (1.0 + 4.0 * w);
        assert!(approx(out, expected));
    }

    #[test]
    fn output_is_never_negative() {
        // A dark center among bright neighbors drives the blend negative before
        // the clamp; the contract clamps it back to zero.
        let params = CasParams::new(1.0);
        let win = cross_window(0.1, 1.0);
        let out = params.sharpen_taps(&win);
        assert!(out[0] >= 0.0);
        assert!(approx(out[0], 0.0));
    }

    #[test]
    fn channels_are_independent() {
        let params = CasParams::new(0.5);
        // Red has a bright-center edge; green/blue are flat.
        let c = [0.6_f32, 0.5, 0.5];
        let n = [0.4_f32, 0.5, 0.5];
        let win = [c, n, c, n, c, n, c, n, c];
        let out = params.sharpen_taps(&win);
        assert!(out[0] > 0.6);
        assert!(approx(out[1], 0.5));
        assert!(approx(out[2], 0.5));
    }

    #[test]
    fn apply_output_length_matches_dimensions() {
        let params = CasParams::new(0.5);
        let img = [[0.5_f32; 3]; 12];
        let out = params.apply(&img, 4, 3);
        assert_eq!(out.len(), 12);
    }

    #[test]
    fn apply_zero_width_returns_empty() {
        let params = CasParams::new(0.5);
        let img = [[0.5_f32; 3]; 4];
        assert!(params.apply(&img, 0, 4).is_empty());
    }

    #[test]
    fn apply_zero_height_returns_empty() {
        let params = CasParams::new(0.5);
        let img = [[0.5_f32; 3]; 4];
        assert!(params.apply(&img, 4, 0).is_empty());
    }

    #[test]
    fn apply_short_buffer_returns_empty() {
        let params = CasParams::new(0.5);
        let img = [[0.5_f32; 3]; 3];
        // Declares 4x4 = 16 texels but only 3 are supplied.
        assert!(params.apply(&img, 4, 4).is_empty());
    }

    #[test]
    fn apply_constant_image_is_identity() {
        // A flat image stays flat everywhere, including the clamped borders and
        // corners: boundary replication feeds each edge texel its own value.
        let params = CasParams::new(1.0);
        let img = [[0.33_f32, 0.66, 0.99]; 9];
        let out = params.apply(&img, 3, 3);
        for texel in &out {
            assert!(approx3(*texel, [0.33, 0.66, 0.99]));
        }
    }

    #[test]
    fn single_pixel_image_is_identity() {
        // Every neighbor clamps back to the lone pixel, so the neighborhood is
        // flat and the pixel is returned unchanged.
        let params = CasParams::new(1.0);
        let img = [[0.25_f32, 0.5, 0.75]];
        let out = params.apply(&img, 1, 1);
        assert_eq!(out.len(), 1);
        assert!(approx3(out[0], [0.25, 0.5, 0.75]));
    }

    #[test]
    fn apply_interior_matches_sharpen_taps() {
        // For a 3x3 image the center texel's clamped neighborhood is exactly the
        // image in row-major order, so `apply` at index 4 must equal
        // `sharpen_taps` on the whole image.
        let params = CasParams::new(0.7);
        let img = [
            [0.2_f32; 3],
            [0.3; 3],
            [0.2; 3],
            [0.3; 3],
            [0.6; 3],
            [0.3; 3],
            [0.2; 3],
            [0.3; 3],
            [0.2; 3],
        ];
        let out = params.apply(&img, 3, 3);
        let taps: [[f32; 3]; 9] = img;
        assert!(approx3(out[4], params.sharpen_taps(&taps)));
    }

    #[test]
    fn corner_uses_clamped_neighbors() {
        // Top-left corner of a 2x2 image: up/left clamp to the corner itself, so
        // the effective neighborhood only sees the right and down samples. With a
        // brighter corner than its visible neighbors it is enhanced upward.
        let params = CasParams::new(1.0);
        let img = [[0.8_f32; 3], [0.4; 3], [0.4; 3], [0.4; 3]];
        let out = params.apply(&img, 2, 2);
        assert!(out[0][0] > 0.8);
    }

    #[test]
    fn to_std430_has_vec4_block_size() {
        let params = CasParams::new(0.5);
        let bytes = params.to_std430();
        assert_eq!(bytes.len(), SHARPEN_STD430_SIZE);
        assert_eq!(SHARPEN_STD430_SIZE, VEC4_STRIDE);
        assert_eq!(SHARPEN_STD430_SIZE % VEC4_STRIDE, 0);
    }

    #[test]
    fn to_std430_encodes_sharpness_first() {
        let params = CasParams::new(0.75);
        let bytes = params.to_std430();
        let mut head = [0u8; 4];
        head.copy_from_slice(&bytes[0..4]);
        assert!(approx(f32::from_le_bytes(head), 0.75));
    }

    #[test]
    fn gpu_storage_bytes_reserves_one_block() {
        let params = CasParams::new(0.5);
        assert_eq!(params.gpu_storage_bytes(), SHARPEN_STD430_SIZE);
    }

    #[test]
    fn new_clamps_sharpness_into_unit_range() {
        assert!(approx(CasParams::new(-1.0).sharpness, 0.0));
        assert!(approx(CasParams::new(2.5).sharpness, 1.0));
        assert!(approx(CasParams::new(0.3).sharpness, 0.3));
    }

    #[test]
    fn out_of_range_field_is_reclamped_by_sharpen() {
        // A hand-built out-of-range field is still safe: the blend clamps
        // sharpness internally, so it behaves like the clamped endpoint.
        let raw = CasParams { sharpness: 5.0 };
        let clamped = CasParams::new(1.0);
        let win = cross_window(0.6, 0.4);
        assert!(approx3(raw.sharpen_taps(&win), clamped.sharpen_taps(&win)));
    }
}
