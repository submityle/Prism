//! Programmable 3D colour-grading look-up cube (`LUT`) sampling — the
//! `CPU` gold reference (design §16, §30).
//!
//! Production colour pipelines bake a colourist's grade into a cubic
//! `LUT`: a `size`×`size`×`size` lattice of graded `RGB` samples that a
//! `GPU` reads as a 3D texture. At runtime the shader maps an input
//! `RGB` colour into cube space (`channel * (size - 1)`), fetches the
//! surrounding lattice points, and interpolates. This module owns the
//! device-free maths of that fetch: [`ColorGradeLut::sample_trilinear`]
//! (the 8-corner box filter) and [`ColorGradeLut::sample_tetrahedral`]
//! (the 6-tetrahedron filter that most hardware 3D-`LUT` samplers use),
//! plus identity-cube baking and the `std430` byte layouts.
//!
//! # Distinction from the sibling colour modules
//!
//! - [`super::color_gradient`] is a *1D* `RGBA` ramp sampled over a
//!   particle's normalized life (colour *over life*, authored as
//!   position/colour stops). This module is a *3D* grading cube keyed on
//!   the input colour itself, not on age, and stores `RGB` lattice
//!   points rather than life-keyed stops.
//! - [`super::tonemap`] evaluates an *analytic* tone curve (`Reinhard`,
//!   `ACES`) with a closed-form formula. This module is a *programmable
//!   table look-up*: any grade a colourist can bake into a cube is
//!   reproduced by lattice interpolation, with no fixed curve.
//!
//! # Determinism
//!
//! Every routine uses only ordinary `f32` arithmetic plus `floor` (to
//! locate a lattice cell) and `clamp`. No transcendental function
//! (`sin`/`cos`/`exp`/`ln`/`pow`) is called, so this `CPU` reference
//! stays bit-reproducible against a future `GPU` sampler, matching the
//! determinism contract of the sibling [`super::simulation`] module.

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};
use alloc::vec::Vec;

/// Byte stride of one `RGBA8` lattice texel (four unsigned bytes).
pub const RGBA8_STRIDE: usize = 4;

/// A linear `RGB` colour. Channel values are unbounded so the cube can
/// carry `HDR` grades, though sampling clamps the *input* lookup key into
/// the `0..=1` unit range before mapping it into cube space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb {
    /// Red channel (linear).
    pub r: f32,
    /// Green channel (linear).
    pub g: f32,
    /// Blue channel (linear).
    pub b: f32,
}

impl Rgb {
    /// Linear black `(0, 0, 0)`, returned as the empty-cube guard.
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };

    /// Builds a colour from its three linear channels.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// The three channels as an array in `[r, g, b]` order.
    #[must_use]
    pub const fn to_array(self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// Component-wise linear interpolation towards `other` by `t`
    /// (unclamped), blending every channel independently.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: lerp_scalar(self.r, other.r, t),
            g: lerp_scalar(self.g, other.g, t),
            b: lerp_scalar(self.b, other.b, t),
        }
    }
}

/// Linear interpolation between `a` and `b` by `t` (unclamped).
#[must_use]
fn lerp_scalar(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Convex/affine combination of four colours with four weights. Kept as a
/// free helper (rather than an operator-style `add`) so the tetrahedral
/// formula reads as a single weighted sum.
#[must_use]
fn blend4(v0: Rgb, w0: f32, v1: Rgb, w1: f32, v2: Rgb, w2: f32, v3: Rgb, w3: f32) -> Rgb {
    Rgb {
        r: v0.r * w0 + v1.r * w1 + v2.r * w2 + v3.r * w3,
        g: v0.g * w0 + v1.g * w1 + v2.g * w2 + v3.g * w3,
        b: v0.b * w0 + v1.b * w1 + v2.b * w2 + v3.b * w3,
    }
}

/// Maps one clamped input channel into cube space and splits it into the
/// base lattice index and the in-cell fraction.
///
/// The channel is clamped into `0..=1`, scaled by the last index
/// (`last = size - 1`), and floored. The base is capped at `last - 1` so a
/// query exactly at the far face selects the final cell with a fraction of
/// `1.0` instead of reading past the lattice.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "channel is clamped into 0..=1 then scaled by `last`, so the floored value is a small non-negative integer that maps to `usize` exactly"
)]
fn axis_coord(channel: f32, last: usize) -> (usize, f32) {
    let scaled = channel.clamp(0.0, 1.0) * last as f32;
    let floored = scaled.floor();
    let max_base = last.saturating_sub(1);
    let base = (floored as usize).min(max_base);
    let frac = scaled - base as f32;
    (base, frac)
}

/// Quantizes a linear unit-range channel to an `RGBA8` byte.
///
/// The channel is clamped into `0..=1`, scaled to `0..=255`, biased by a
/// half and floored (round-to-nearest without the banned `round`), then
/// clamped again so the cast into `u8` is exact and never negative.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is clamped into 0..=255 and floored before the cast, so it fits `u8` and is non-negative"
)]
fn quantize_unit(value: f32) -> u8 {
    let scaled = (value.clamp(0.0, 1.0) * 255.0 + 0.5).floor();
    scaled.clamp(0.0, 255.0) as u8
}

/// Maps a lattice index onto its identity unit coordinate `index / last`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "lattice indices are small; the ratio is the nearest representable identity coordinate and stays exact for practical cube sizes"
)]
fn unit_step(index: usize, last: usize) -> f32 {
    if last == 0 {
        0.0
    } else {
        index as f32 / last as f32
    }
}

/// A baked 3D colour-grading cube: a `size`×`size`×`size` lattice of graded
/// `RGB` samples read at runtime by trilinear or tetrahedral interpolation.
///
/// Texels are stored `RED`-fastest: the sample at lattice coordinate
/// `(r, g, b)` lives at flat index `r + size * (g + size * b)`. This is the
/// `CPU` mirror of the 3D texture a `GPU` sampler would read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColorGradeLut {
    size: usize,
    texels: Vec<Rgb>,
}

impl ColorGradeLut {
    /// Bakes the identity cube of the requested `size`: the sample at
    /// lattice coordinate `(r, g, b)` is exactly
    /// `(r / (size - 1), g / (size - 1), b / (size - 1))`, so sampling it
    /// returns the (clamped) input colour unchanged.
    ///
    /// A `size` of `0` yields an empty cube; a `size` of `1` yields the
    /// single black texel.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let last = size.saturating_sub(1);
        let count = size.saturating_mul(size).saturating_mul(size);
        let mut texels = Vec::with_capacity(count);
        for b in 0..size {
            let bv = unit_step(b, last);
            for g in 0..size {
                let gv = unit_step(g, last);
                for r in 0..size {
                    texels.push(Rgb::new(unit_step(r, last), gv, bv));
                }
            }
        }
        Self { size, texels }
    }

    /// Wraps a pre-baked texel sequence, returning [`None`] unless `size`
    /// is non-zero and `texels.len()` equals `size * size * size`.
    #[must_use]
    pub fn from_texels(size: usize, texels: Vec<Rgb>) -> Option<Self> {
        let expected = size.saturating_mul(size).saturating_mul(size);
        if size == 0 || texels.len() != expected {
            return None;
        }
        Some(Self { size, texels })
    }

    /// The number of lattice points per axis.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The total number of stored texels (`size * size * size`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.texels.len()
    }

    /// Whether the cube has no texels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.texels.is_empty()
    }

    /// The stored texels, `RED`-fastest.
    #[must_use]
    pub fn texels(&self) -> &[Rgb] {
        &self.texels
    }

    /// Fetches the lattice point at coordinate `(r, g, b)`, clamping each
    /// index into `0..=size-1`. Returns [`Rgb::BLACK`] for an empty cube.
    #[must_use]
    pub fn texel(&self, r: usize, g: usize, b: usize) -> Rgb {
        if self.texels.is_empty() {
            return Rgb::BLACK;
        }
        let last = self.size - 1;
        let ri = r.min(last);
        let gi = g.min(last);
        let bi = b.min(last);
        self.texels[ri + self.size * (gi + self.size * bi)]
    }

    /// Samples the graded colour for `input` with 8-corner trilinear
    /// interpolation. The input is clamped into the `0..=1` unit cube
    /// before it is mapped into lattice space.
    #[must_use]
    pub fn sample_trilinear(&self, input: Rgb) -> Rgb {
        if self.size <= 1 {
            return self.texels.first().copied().unwrap_or(Rgb::BLACK);
        }
        let last = self.size - 1;
        let (br, fr) = axis_coord(input.r, last);
        let (bg, fg) = axis_coord(input.g, last);
        let (bb, fb) = axis_coord(input.b, last);

        let c000 = self.texel(br, bg, bb);
        let c100 = self.texel(br + 1, bg, bb);
        let c010 = self.texel(br, bg + 1, bb);
        let c110 = self.texel(br + 1, bg + 1, bb);
        let c001 = self.texel(br, bg, bb + 1);
        let c101 = self.texel(br + 1, bg, bb + 1);
        let c011 = self.texel(br, bg + 1, bb + 1);
        let c111 = self.texel(br + 1, bg + 1, bb + 1);

        // Interpolate along red, then green, then blue.
        let c00 = c000.lerp(c100, fr);
        let c01 = c001.lerp(c101, fr);
        let c10 = c010.lerp(c110, fr);
        let c11 = c011.lerp(c111, fr);
        let c0 = c00.lerp(c10, fg);
        let c1 = c01.lerp(c11, fg);
        c0.lerp(c1, fb)
    }

    /// Samples the graded colour for `input` with 6-tetrahedron
    /// interpolation.
    ///
    /// The unit cell is split into six tetrahedra sharing the
    /// `(0,0,0)`→`(1,1,1)` diagonal; the tetrahedron enclosing the query is
    /// chosen by the ordering of the three in-cell fractions, and the four
    /// enclosing lattice points are blended with barycentric weights. This
    /// matches the filter most hardware 3D-`LUT` samplers implement and, on
    /// an affine (identity) cube, reproduces the input exactly like
    /// [`ColorGradeLut::sample_trilinear`].
    #[must_use]
    pub fn sample_tetrahedral(&self, input: Rgb) -> Rgb {
        if self.size <= 1 {
            return self.texels.first().copied().unwrap_or(Rgb::BLACK);
        }
        let last = self.size - 1;
        let (br, fr) = axis_coord(input.r, last);
        let (bg, fg) = axis_coord(input.g, last);
        let (bb, fb) = axis_coord(input.b, last);

        let c000 = self.texel(br, bg, bb);
        let c100 = self.texel(br + 1, bg, bb);
        let c010 = self.texel(br, bg + 1, bb);
        let c110 = self.texel(br + 1, bg + 1, bb);
        let c001 = self.texel(br, bg, bb + 1);
        let c101 = self.texel(br + 1, bg, bb + 1);
        let c011 = self.texel(br, bg + 1, bb + 1);
        let c111 = self.texel(br + 1, bg + 1, bb + 1);

        if fr > fg {
            if fg > fb {
                // fr >= fg >= fb
                blend4(c000, 1.0 - fr, c100, fr - fg, c110, fg - fb, c111, fb)
            } else if fr > fb {
                // fr >= fb >= fg
                blend4(c000, 1.0 - fr, c100, fr - fb, c101, fb - fg, c111, fg)
            } else {
                // fb >= fr >= fg
                blend4(c000, 1.0 - fb, c001, fb - fr, c101, fr - fg, c111, fg)
            }
        } else if fb > fg {
            // fb >= fg >= fr
            blend4(c000, 1.0 - fb, c001, fb - fg, c011, fg - fr, c111, fr)
        } else if fb > fr {
            // fg >= fb >= fr
            blend4(c000, 1.0 - fg, c010, fg - fb, c011, fb - fr, c111, fr)
        } else {
            // fg >= fr >= fb
            blend4(c000, 1.0 - fg, c010, fg - fr, c110, fr - fb, c111, fb)
        }
    }

    /// Grades a batch of colours with trilinear interpolation, preserving
    /// order (one output per input).
    #[must_use]
    pub fn grade_batch_trilinear(&self, inputs: &[Rgb]) -> Vec<Rgb> {
        inputs.iter().map(|&c| self.sample_trilinear(c)).collect()
    }

    /// Grades a batch of colours with tetrahedral interpolation, preserving
    /// order (one output per input).
    #[must_use]
    pub fn grade_batch_tetrahedral(&self, inputs: &[Rgb]) -> Vec<Rgb> {
        inputs.iter().map(|&c| self.sample_tetrahedral(c)).collect()
    }

    /// Packs the cube into its `std430` byte layout as one `vec4<f32>` per
    /// texel: the three channels followed by a zero padding lane, giving a
    /// [`VEC4_STRIDE`]-byte stride the aligned `GPU` binding expects.
    #[must_use]
    pub fn to_std430_f32(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.texels.len() * VEC4_STRIDE);
        for texel in &self.texels {
            bytes.extend_from_slice(&texel.r.to_le_bytes());
            bytes.extend_from_slice(&texel.g.to_le_bytes());
            bytes.extend_from_slice(&texel.b.to_le_bytes());
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }
        bytes
    }

    /// The `std430` byte footprint of [`ColorGradeLut::to_std430_f32`]:
    /// [`ColorGradeLut::len`] texels each occupying [`VEC4_STRIDE`] bytes,
    /// clamped up to a single element so an empty cube still yields a valid
    /// non-zero binding size.
    #[must_use]
    pub fn std430_bytes(&self) -> usize {
        storage_bytes(VEC4_STRIDE, self.texels.len())
    }

    /// Packs the cube as tightly-stored `RGBA8` texels: each channel is
    /// quantized to a byte and the alpha lane is set to `255`, giving a
    /// [`RGBA8_STRIDE`]-byte stride.
    #[must_use]
    pub fn to_rgba8(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.texels.len() * RGBA8_STRIDE);
        for texel in &self.texels {
            bytes.push(quantize_unit(texel.r));
            bytes.push(quantize_unit(texel.g));
            bytes.push(quantize_unit(texel.b));
            bytes.push(255);
        }
        bytes
    }

    /// The `std430` byte footprint of [`ColorGradeLut::to_rgba8`]:
    /// [`ColorGradeLut::len`] texels each occupying [`RGBA8_STRIDE`] bytes,
    /// clamped up to a single element.
    #[must_use]
    pub fn rgba8_bytes(&self) -> usize {
        storage_bytes(RGBA8_STRIDE, self.texels.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Absolute tolerance for `f32` equality decisions in tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn rgb_close(a: Rgb, b: Rgb) -> bool {
        approx(a.r, b.r) && approx(a.g, b.g) && approx(a.b, b.b)
    }

    /// Bakes a per-axis remapped cube: channel `x` becomes `f(x)`, applied
    /// independently on each axis. `f` must be monotonic on `0..=1` for the
    /// order-preservation test.
    fn baked_cube(size: usize, f: impl Fn(f32) -> f32) -> ColorGradeLut {
        let last = size - 1;
        let mut texels = Vec::new();
        for b in 0..size {
            let bv = f(b as f32 / last as f32);
            for g in 0..size {
                let gv = f(g as f32 / last as f32);
                for r in 0..size {
                    texels.push(Rgb::new(f(r as f32 / last as f32), gv, bv));
                }
            }
        }
        ColorGradeLut::from_texels(size, texels).unwrap()
    }

    #[test]
    fn rgb_lerp_is_component_wise() {
        let lo = Rgb::new(0.0, 0.0, 0.0);
        let hi = Rgb::new(2.0, 4.0, 6.0);
        assert!(rgb_close(lo.lerp(hi, 0.5), Rgb::new(1.0, 2.0, 3.0)));
        assert!(rgb_close(lo.lerp(hi, 0.0), lo));
        assert!(rgb_close(lo.lerp(hi, 1.0), hi));
        assert_eq!(hi.to_array(), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn identity_dimensions_and_layout() {
        let lut = ColorGradeLut::identity(8);
        assert_eq!(lut.size(), 8);
        assert_eq!(lut.len(), 8 * 8 * 8);
        assert!(!lut.is_empty());
    }

    #[test]
    fn identity_grid_points_are_normalized_coordinates() {
        let size = 5;
        let last = (size - 1) as f32;
        let lut = ColorGradeLut::identity(size);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let expected = Rgb::new(r as f32 / last, g as f32 / last, b as f32 / last);
                    assert!(rgb_close(lut.texel(r, g, b), expected));
                }
            }
        }
    }

    #[test]
    fn identity_trilinear_is_identity() {
        let lut = ColorGradeLut::identity(9);
        for &c in &[
            Rgb::new(0.0, 0.0, 0.0),
            Rgb::new(0.13, 0.42, 0.77),
            Rgb::new(0.5, 0.25, 0.9),
            Rgb::new(1.0, 1.0, 1.0),
        ] {
            assert!(rgb_close(lut.sample_trilinear(c), c));
        }
    }

    #[test]
    fn identity_tetrahedral_is_identity() {
        let lut = ColorGradeLut::identity(9);
        for &c in &[
            Rgb::new(0.0, 0.0, 0.0),
            Rgb::new(0.13, 0.42, 0.77),
            Rgb::new(0.66, 0.2, 0.31),
            Rgb::new(1.0, 1.0, 1.0),
        ] {
            assert!(rgb_close(lut.sample_tetrahedral(c), c));
        }
    }

    #[test]
    fn trilinear_hits_grid_points_exactly() {
        let size = 6;
        let last = (size - 1) as f32;
        let lut = baked_cube(size, |x| x * x);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let key = Rgb::new(r as f32 / last, g as f32 / last, b as f32 / last);
                    assert!(rgb_close(lut.sample_trilinear(key), lut.texel(r, g, b)));
                }
            }
        }
    }

    #[test]
    fn tetrahedral_hits_grid_points_exactly() {
        let size = 6;
        let last = (size - 1) as f32;
        let lut = baked_cube(size, |x| x * x);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let key = Rgb::new(r as f32 / last, g as f32 / last, b as f32 / last);
                    assert!(rgb_close(lut.sample_tetrahedral(key), lut.texel(r, g, b)));
                }
            }
        }
    }

    #[test]
    fn trilinear_centre_is_eight_corner_average() {
        let texels = vec![
            Rgb::new(0.0, 0.0, 0.0), // (0,0,0)
            Rgb::new(0.8, 0.1, 0.2), // (1,0,0)
            Rgb::new(0.1, 0.7, 0.3), // (0,1,0)
            Rgb::new(0.4, 0.4, 0.4), // (1,1,0)
            Rgb::new(0.2, 0.2, 0.9), // (0,0,1)
            Rgb::new(0.6, 0.3, 0.5), // (1,0,1)
            Rgb::new(0.3, 0.6, 0.6), // (0,1,1)
            Rgb::new(0.9, 0.9, 0.1), // (1,1,1)
        ];
        let mut sum = Rgb::BLACK;
        for t in &texels {
            sum = Rgb::new(sum.r + t.r, sum.g + t.g, sum.b + t.b);
        }
        let mean = Rgb::new(sum.r / 8.0, sum.g / 8.0, sum.b / 8.0);
        let lut = ColorGradeLut::from_texels(2, texels).unwrap();
        let centre = lut.sample_trilinear(Rgb::new(0.5, 0.5, 0.5));
        assert!(rgb_close(centre, mean));
    }

    #[test]
    fn tetrahedral_matches_trilinear_at_corners() {
        let lut = baked_cube(4, |x| x * (2.0 - x));
        for &c in &[
            Rgb::new(0.0, 0.0, 0.0),
            Rgb::new(1.0, 0.0, 0.0),
            Rgb::new(0.0, 1.0, 0.0),
            Rgb::new(0.0, 0.0, 1.0),
            Rgb::new(1.0, 1.0, 0.0),
            Rgb::new(1.0, 0.0, 1.0),
            Rgb::new(0.0, 1.0, 1.0),
            Rgb::new(1.0, 1.0, 1.0),
        ] {
            assert!(rgb_close(
                lut.sample_tetrahedral(c),
                lut.sample_trilinear(c)
            ));
        }
    }

    #[test]
    fn tetrahedral_matches_trilinear_on_affine_cube() {
        // The identity cube is affine, so both filters reproduce it exactly
        // and therefore agree at every interior query.
        let lut = ColorGradeLut::identity(7);
        for &c in &[
            Rgb::new(0.05, 0.95, 0.5),
            Rgb::new(0.33, 0.66, 0.99),
            Rgb::new(0.71, 0.14, 0.28),
            Rgb::new(0.5, 0.5, 0.5),
        ] {
            assert!(rgb_close(
                lut.sample_tetrahedral(c),
                lut.sample_trilinear(c)
            ));
        }
    }

    #[test]
    fn below_zero_input_clamps_to_first_texel() {
        let lut = baked_cube(5, |x| x * x);
        let expected = lut.texel(0, 0, 0);
        assert!(rgb_close(
            lut.sample_trilinear(Rgb::new(-1.0, -0.5, -3.0)),
            expected
        ));
        assert!(rgb_close(
            lut.sample_tetrahedral(Rgb::new(-1.0, -0.5, -3.0)),
            expected
        ));
    }

    #[test]
    fn above_one_input_clamps_to_last_texel() {
        let size = 5;
        let lut = baked_cube(size, |x| x * x);
        let last = size - 1;
        let expected = lut.texel(last, last, last);
        assert!(rgb_close(
            lut.sample_trilinear(Rgb::new(2.0, 5.0, 1.5)),
            expected
        ));
        assert!(rgb_close(
            lut.sample_tetrahedral(Rgb::new(2.0, 5.0, 1.5)),
            expected
        ));
    }

    #[test]
    fn monotonic_grade_preserves_channel_order() {
        // A monotone per-axis remap must keep sampled outputs monotone.
        let lut = baked_cube(9, |x| x * x);
        let keys = [0.1f32, 0.3, 0.55, 0.8, 0.95];
        let mut prev = -1.0f32;
        for &k in &keys {
            let out = lut.sample_trilinear(Rgb::new(k, k, k));
            assert!(out.r > prev);
            assert!(out.r > prev - CMP_EPS);
            assert!(rgb_close(out, Rgb::new(out.r, out.r, out.r)));
            prev = out.r;
        }
    }

    #[test]
    fn tetrahedral_result_lies_in_cell_bounding_box() {
        let lut = baked_cube(4, |x| x * (2.0 - x));
        let queries = [
            Rgb::new(0.17, 0.51, 0.83),
            Rgb::new(0.62, 0.29, 0.44),
            Rgb::new(0.4, 0.4, 0.9),
        ];
        let last = lut.size() - 1;
        for &q in &queries {
            // Recover the enclosing cell to bound the result.
            let scaled = |c: f32| c.clamp(0.0, 1.0) * last as f32;
            let cell = |c: f32| {
                let s = scaled(c);
                let base = (s.floor() as usize).min(last - 1);
                (base, base + 1)
            };
            let (r0, r1) = cell(q.r);
            let (g0, g1) = cell(q.g);
            let (b0, b1) = cell(q.b);
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &r in &[r0, r1] {
                for &g in &[g0, g1] {
                    for &bch in &[b0, b1] {
                        for v in lut.texel(r, g, bch).to_array() {
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                }
            }
            let out = lut.sample_tetrahedral(q);
            for v in out.to_array() {
                assert!((lo - CMP_EPS..=hi + CMP_EPS).contains(&v));
            }
        }
    }

    #[test]
    fn degenerate_size_two_cube_interpolates() {
        let lut = ColorGradeLut::identity(2);
        assert_eq!(lut.size(), 2);
        assert_eq!(lut.len(), 8);
        let mid = lut.sample_trilinear(Rgb::new(0.5, 0.5, 0.5));
        assert!(rgb_close(mid, Rgb::new(0.5, 0.5, 0.5)));
        let mid_t = lut.sample_tetrahedral(Rgb::new(0.5, 0.5, 0.5));
        assert!(rgb_close(mid_t, Rgb::new(0.5, 0.5, 0.5)));
    }

    #[test]
    fn size_one_and_empty_cubes_are_guarded() {
        let one = ColorGradeLut::identity(1);
        assert_eq!(one.len(), 1);
        assert!(rgb_close(
            one.sample_trilinear(Rgb::new(0.3, 0.7, 0.1)),
            Rgb::BLACK
        ));
        let empty = ColorGradeLut::identity(0);
        assert!(empty.is_empty());
        assert!(rgb_close(
            empty.sample_tetrahedral(Rgb::new(0.5, 0.5, 0.5)),
            Rgb::BLACK
        ));
        assert!(rgb_close(empty.texel(0, 0, 0), Rgb::BLACK));
    }

    #[test]
    fn from_texels_validates_length() {
        assert!(ColorGradeLut::from_texels(2, vec![Rgb::BLACK; 8]).is_some());
        assert!(ColorGradeLut::from_texels(2, vec![Rgb::BLACK; 7]).is_none());
        assert!(ColorGradeLut::from_texels(0, Vec::new()).is_none());
    }

    #[test]
    fn std430_layout_size_and_padding() {
        let lut = ColorGradeLut::identity(3);
        let texel_count = 3 * 3 * 3;
        assert_eq!(lut.std430_bytes(), texel_count * VEC4_STRIDE);
        let bytes = lut.to_std430_f32();
        assert_eq!(bytes.len(), texel_count * VEC4_STRIDE);
        // The first texel is black with a zero padding lane.
        assert_eq!(&bytes[0..4], &0.0f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0.0f32.to_le_bytes());
        // The final texel is white in its RGB lanes.
        let last_off = (texel_count - 1) * VEC4_STRIDE;
        assert_eq!(&bytes[last_off..last_off + 4], &1.0f32.to_le_bytes());
    }

    #[test]
    fn rgba8_layout_size_and_quantization() {
        let lut = ColorGradeLut::identity(2);
        assert_eq!(lut.rgba8_bytes(), 8 * RGBA8_STRIDE);
        let bytes = lut.to_rgba8();
        assert_eq!(bytes.len(), 8 * RGBA8_STRIDE);
        // Texel (0,0,0) is black, alpha saturated.
        assert_eq!(&bytes[0..4], &[0, 0, 0, 255]);
        // Texel (1,1,1) is white, alpha saturated.
        assert_eq!(&bytes[28..32], &[255, 255, 255, 255]);
    }

    #[test]
    fn quantize_unit_rounds_and_clamps() {
        assert_eq!(quantize_unit(0.0), 0);
        assert_eq!(quantize_unit(1.0), 255);
        assert_eq!(quantize_unit(-4.0), 0);
        assert_eq!(quantize_unit(9.0), 255);
        assert_eq!(quantize_unit(0.5), 128);
    }

    #[test]
    fn grade_batches_preserve_order_and_match_scalar() {
        let lut = baked_cube(5, |x| x * x);
        let inputs = [
            Rgb::new(0.1, 0.2, 0.3),
            Rgb::new(0.7, 0.4, 0.9),
            Rgb::new(0.55, 0.55, 0.55),
        ];
        let tri = lut.grade_batch_trilinear(&inputs);
        let tet = lut.grade_batch_tetrahedral(&inputs);
        assert_eq!(tri.len(), inputs.len());
        assert_eq!(tet.len(), inputs.len());
        for (i, &c) in inputs.iter().enumerate() {
            assert!(rgb_close(tri[i], lut.sample_trilinear(c)));
            assert!(rgb_close(tet[i], lut.sample_tetrahedral(c)));
        }
    }
}
