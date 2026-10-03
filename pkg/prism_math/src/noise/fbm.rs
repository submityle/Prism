//! Fractal noise helpers: fBm, turbulence, and ridged multifractal.
//!
//! These stack several octaves of an underlying gradient-noise source at
//! geometrically increasing frequency (`lacunarity`) and decreasing amplitude
//! (`gain`). They are generic over any [`Noise2`]/[`Noise3`] source, so they
//! work uniformly with [`crate::noise::Perlin`] and [`crate::noise::Simplex`].

use crate::noise::{Perlin, Simplex};

/// A 2D scalar noise source sampled at `(x, y)`.
pub trait Noise2 {
    /// Sample the field at `(x, y)`.
    fn sample2(&self, x: f32, y: f32) -> f32;
}

/// A 3D scalar noise source sampled at `(x, y, z)`.
pub trait Noise3 {
    /// Sample the field at `(x, y, z)`.
    fn sample3(&self, x: f32, y: f32, z: f32) -> f32;
}

impl Noise2 for Perlin {
    #[inline]
    fn sample2(&self, x: f32, y: f32) -> f32 {
        self.get2(x, y)
    }
}
impl Noise3 for Perlin {
    #[inline]
    fn sample3(&self, x: f32, y: f32, z: f32) -> f32 {
        self.get3(x, y, z)
    }
}
impl Noise2 for Simplex {
    #[inline]
    fn sample2(&self, x: f32, y: f32) -> f32 {
        self.get2(x, y)
    }
}
impl Noise3 for Simplex {
    #[inline]
    fn sample3(&self, x: f32, y: f32, z: f32) -> f32 {
        self.get3(x, y, z)
    }
}

/// Parameters controlling a fractal (multi-octave) noise sum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fractal {
    /// Number of octaves (layers) summed. Must be `>= 1`.
    pub octaves: u32,
    /// Frequency multiplier between successive octaves (typically `~2.0`).
    pub lacunarity: f32,
    /// Amplitude multiplier between successive octaves (typically `~0.5`).
    pub gain: f32,
    /// Frequency of the first octave.
    pub frequency: f32,
}

impl Default for Fractal {
    #[inline]
    fn default() -> Self {
        Self { octaves: 4, lacunarity: 2.0, gain: 0.5, frequency: 1.0 }
    }
}

impl Fractal {
    /// Fractional Brownian motion in 2D: a straight amplitude-weighted sum of
    /// octaves, normalized so the result stays within the source's range.
    #[inline]
    pub fn fbm2<N: Noise2>(&self, src: &N, x: f32, y: f32) -> f32 {
        let mut freq = self.frequency;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..self.octaves.max(1) {
            sum += amp * src.sample2(x * freq, y * freq);
            norm += amp;
            freq *= self.lacunarity;
            amp *= self.gain;
        }
        sum / norm
    }

    /// Fractional Brownian motion in 3D.
    #[inline]
    pub fn fbm3<N: Noise3>(&self, src: &N, x: f32, y: f32, z: f32) -> f32 {
        let mut freq = self.frequency;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..self.octaves.max(1) {
            sum += amp * src.sample3(x * freq, y * freq, z * freq);
            norm += amp;
            freq *= self.lacunarity;
            amp *= self.gain;
        }
        sum / norm
    }

    /// Turbulence in 2D: like [`Fractal::fbm2`] but summing `|noise|`, giving a
    /// billowy field in `[0, 1]`.
    #[inline]
    pub fn turbulence2<N: Noise2>(&self, src: &N, x: f32, y: f32) -> f32 {
        let mut freq = self.frequency;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..self.octaves.max(1) {
            sum += amp * src.sample2(x * freq, y * freq).abs();
            norm += amp;
            freq *= self.lacunarity;
            amp *= self.gain;
        }
        sum / norm
    }

    /// Ridged multifractal in 2D: inverts and sharpens turbulence to produce
    /// crisp ridge lines, in `[0, 1]`.
    #[inline]
    pub fn ridged2<N: Noise2>(&self, src: &N, x: f32, y: f32) -> f32 {
        let mut freq = self.frequency;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for _ in 0..self.octaves.max(1) {
            let n = 1.0 - src.sample2(x * freq, y * freq).abs();
            sum += amp * n * n;
            norm += amp;
            freq *= self.lacunarity;
            amp *= self.gain;
        }
        sum / norm
    }
}
