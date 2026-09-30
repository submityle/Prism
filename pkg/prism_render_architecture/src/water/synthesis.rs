//! `CPU` ocean-surface synthesis: evolve a spectral field and inverse-transform
//! it into a spatial height field with the butterfly [`fft`](super::fft).
//!
//! [`initial_spectrum`](super::initial_spectrum) builds the time-zero complex
//! amplitudes `h0(k)` / `h0(-k)`; [`spectrum::advance_amplitude`] evolves them
//! to `h(k, t)`; and the butterfly `FFT` turns that frequency-domain grid into
//! the tiled spatial displacement an author actually sees. Before this module
//! the only inverse transform in the crate was the compute shader's `O(N^4)`
//! direct sum, so nothing on the `CPU` could synthesise a surface for tests or
//! tooling. This closes that seam with the same separable `O(N^2 log N)`
//! transform shipping oceans (`Tessendorf`, `WaveWorks`, `Crest`, UE5 Water)
//! use, giving a deterministic golden surface the `GPU` path can be checked
//! against.
//!
//! # Frequency layout and the checkerboard sign
//!
//! [`initial_spectrum`](super::initial_spectrum) stores the grid *centred*: cell
//! `(i, j)` carries wave vector `k = ((j - N/2), (i - N/2)) * 2*PI / L`, so the
//! zero-frequency (`DC`) term sits at the middle of the grid. A standard inverse
//! `FFT` instead expects frequency bin `p` at index `p`. Shifting the spectrum
//! by `-N/2` per axis is, by the `FFT` shift theorem, exactly a `(-1)^(x+z)`
//! checkerboard modulation of the spatial output — so after the transform each
//! texel is multiplied by `+1` on even `(row + col)` and `-1` on odd, which
//! recentres the field without an explicit `fftshift` copy.
//!
//! # Choppiness (horizontal displacement)
//!
//! A pure height field only pushes the surface up and down; real swell also
//! drags water horizontally, sharpening wave crests. `Tessendorf` adds that by
//! inverse-transforming `-i * (k / |k|) * h(k, t)` per horizontal axis. The
//! `choppiness` scale in [`synthesize_surface`] blends that displacement in
//! (`0` = pure height, `1` = full `Tessendorf` choppiness).

use alloc::vec::Vec;
use core::f32::consts::TAU;

use super::initial_spectrum::OceanSpectrumField;
use super::spectrum::{advance_amplitude, dispersion, Complex};
use super::{fft, Vec2};

/// Smallest wave-number magnitude treated as non-zero when forming the
/// horizontal-displacement direction `k / |k|`; below it the `DC` term
/// contributes no choppiness.
const K_EPS: f32 = 1.0e-6;

/// A synthesised ocean surface patch, row-major over the `resolution x
/// resolution` grid (row `i` walks `+z`, column `j` walks `+x`), tiling a
/// `patch_size` square.
///
/// `height` is the vertical displacement; `displacement_x` / `displacement_z`
/// are the horizontal `Tessendorf` choppiness offsets (zero when synthesised
/// with `choppiness == 0`). A degenerate or non-power-of-two field yields an
/// empty surface — an honest no-op rather than a fabricated one, since the
/// radix-2 butterfly only transforms power-of-two grids.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OceanSurface {
    /// Grid resolution `N` (the surface holds `N * N` samples).
    pub resolution: u32,
    /// Spatial patch size `L` (m) the grid tiles.
    pub patch_size: f32,
    /// Vertical displacement per texel, row-major.
    pub height: Vec<f32>,
    /// Horizontal `+x` displacement per texel (choppiness), row-major.
    pub displacement_x: Vec<f32>,
    /// Horizontal `+z` displacement per texel (choppiness), row-major.
    pub displacement_z: Vec<f32>,
}

impl OceanSurface {
    /// Number of samples in the surface (`resolution * resolution`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.height.len()
    }

    /// Whether the surface carries no samples (a degenerate synthesis).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.height.is_empty()
    }

    /// Peak-to-trough vertical extent of the surface, a cheap witness that a
    /// rougher sea produces a taller field. Zero for an empty surface.
    #[must_use]
    pub fn vertical_extent(&self) -> f32 {
        if self.height.is_empty() {
            return 0.0;
        }
        let mut lo = self.height[0];
        let mut hi = self.height[0];
        for &h in &self.height {
            if h < lo {
                lo = h;
            }
            if h > hi {
                hi = h;
            }
        }
        hi - lo
    }
}

/// Evolves `field` to `time` and returns the frequency-domain height grid
/// `h(k, t)`, row-major in the same centred layout the field uses.
///
/// Kept separate so the transform inputs can be asserted directly (Hermitian
/// symmetry, `DC` behaviour) without going through the inverse `FFT`.
fn advanced_grid(field: &OceanSpectrumField, time: f32) -> Vec<Complex> {
    let n = field.resolution as usize;
    let half = (n as f32) * 0.5;
    let k_scale = TAU / field.patch_size;

    let mut grid = Vec::with_capacity(n * n);
    let mut i = 0usize;
    while i < n {
        let kz = (i as f32 - half) * k_scale;
        let mut j = 0usize;
        while j < n {
            let kx = (j as f32 - half) * k_scale;
            let idx = i * n + j;
            let k_mag = Vec2::new(kx, kz).length();
            let omega = dispersion(k_mag);
            grid.push(advance_amplitude(
                field.h0[idx],
                field.h0_neg[idx],
                omega,
                time,
            ));
            j += 1;
        }
        i += 1;
    }
    grid
}

/// Applies the `(-1)^(row + col)` recentring sign to the real part of an inverse
/// `FFT` output grid, writing the spatial field into `out`.
fn extract_real_field(spatial: &[Complex], n: usize, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(n * n);
    let mut i = 0usize;
    while i < n {
        let mut j = 0usize;
        while j < n {
            let idx = i * n + j;
            let sign = if (i + j) & 1 == 0 { 1.0 } else { -1.0 };
            out.push(spatial[idx].re * sign);
            j += 1;
        }
        i += 1;
    }
}

/// Synthesises an ocean surface from a spectral `field` at `time`, with
/// `choppiness` scaling the horizontal `Tessendorf` displacement (`0` = pure
/// height, `1` = full choppiness).
///
/// Returns an empty surface when the field is empty or its resolution is not a
/// power of two (the radix-2 butterfly's contract), so callers get a
/// deterministic no-op instead of a panic.
#[must_use]
pub fn synthesize_surface(field: &OceanSpectrumField, time: f32, choppiness: f32) -> OceanSurface {
    let n = field.resolution as usize;
    if field.is_empty() || !fft::is_power_of_two(n) {
        return OceanSurface::default();
    }

    let spectrum = advanced_grid(field, time);

    // Vertical height: inverse-transform the evolved spectrum directly.
    let height_spatial = fft::ifft2(&spectrum, n);
    let mut height = Vec::new();
    extract_real_field(&height_spatial, n, &mut height);

    // Horizontal choppiness: displace by -i * (k / |k|) * h(k, t) per axis.
    let half = (n as f32) * 0.5;
    let k_scale = TAU / field.patch_size;
    let mut disp_x_freq = Vec::with_capacity(n * n);
    let mut disp_z_freq = Vec::with_capacity(n * n);
    let mut i = 0usize;
    while i < n {
        let kz = (i as f32 - half) * k_scale;
        let mut j = 0usize;
        while j < n {
            let kx = (j as f32 - half) * k_scale;
            let idx = i * n + j;
            let k_mag = Vec2::new(kx, kz).length();
            if k_mag > K_EPS {
                let inv = choppiness / k_mag;
                // Multiply by -i * (component / |k|): -i maps (re, im) -> (im, -re).
                let dir_x = Complex::new(0.0, -kx * inv);
                let dir_z = Complex::new(0.0, -kz * inv);
                disp_x_freq.push(dir_x.mul(spectrum[idx]));
                disp_z_freq.push(dir_z.mul(spectrum[idx]));
            } else {
                disp_x_freq.push(Complex::ZERO);
                disp_z_freq.push(Complex::ZERO);
            }
            j += 1;
        }
        i += 1;
    }

    let disp_x_spatial = fft::ifft2(&disp_x_freq, n);
    let disp_z_spatial = fft::ifft2(&disp_z_freq, n);
    let mut displacement_x = Vec::new();
    let mut displacement_z = Vec::new();
    extract_real_field(&disp_x_spatial, n, &mut displacement_x);
    extract_real_field(&disp_z_spatial, n, &mut displacement_z);

    OceanSurface {
        resolution: field.resolution,
        patch_size: field.patch_size,
        height,
        displacement_x,
        displacement_z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::initial_spectrum::build_initial_spectrum;
    use crate::water::spectrum::{SpectrumKind, SpectrumParams};

    fn sea(wind_speed: f32) -> SpectrumParams {
        SpectrumParams {
            kind: SpectrumKind::Phillips,
            wind: Vec2::new(wind_speed, 0.0),
            amplitude: 0.5,
            peak_enhancement: 1.0,
            min_wavelength: 0.2,
            directional_exponent: 2,
        }
    }

    #[test]
    fn surface_has_n_squared_samples() {
        let field = build_initial_spectrum(32, 128.0, sea(12.0), 7);
        let surface = synthesize_surface(&field, 0.0, 1.0);
        assert_eq!(surface.resolution, 32);
        assert_eq!(surface.len(), 32 * 32);
        assert_eq!(surface.displacement_x.len(), 32 * 32);
        assert_eq!(surface.displacement_z.len(), 32 * 32);
        assert!(!surface.is_empty());
    }

    #[test]
    fn non_power_of_two_resolution_is_empty() {
        let field = build_initial_spectrum(24, 128.0, sea(12.0), 3);
        let surface = synthesize_surface(&field, 1.0, 1.0);
        assert!(surface.is_empty());
    }

    #[test]
    fn calm_field_yields_flat_surface() {
        // A zero-wind sea carries no spectral energy, so the synthesised
        // surface is flat: every height and displacement stays at zero.
        let field = build_initial_spectrum(32, 128.0, sea(0.0), 1);
        let surface = synthesize_surface(&field, 2.0, 1.0);
        assert!(surface.vertical_extent() < 1.0e-6);
        for &h in &surface.height {
            assert!(h.abs() < 1.0e-6);
        }
    }

    #[test]
    fn synthesis_is_deterministic() {
        let field = build_initial_spectrum(16, 64.0, sea(9.0), 42);
        let a = synthesize_surface(&field, 3.5, 0.8);
        let b = synthesize_surface(&field, 3.5, 0.8);
        assert_eq!(a, b);
    }

    #[test]
    fn evolved_height_field_is_real() {
        // Hermitian h(k, t) must inverse-transform to a (near) real field: the
        // imaginary part of the spatial grid stays at the noise floor.
        let field = build_initial_spectrum(32, 128.0, sea(11.0), 5);
        let spectrum = advanced_grid(&field, 4.0);
        let spatial = fft::ifft2(&spectrum, 32);
        let mut max_imag = 0.0f32;
        for c in &spatial {
            let a = c.im.abs();
            if a > max_imag {
                max_imag = a;
            }
        }
        assert!(max_imag < 1.0e-3, "imaginary leakage {max_imag}");
    }

    #[test]
    fn zero_choppiness_has_no_horizontal_displacement() {
        let field = build_initial_spectrum(16, 64.0, sea(10.0), 8);
        let surface = synthesize_surface(&field, 1.0, 0.0);
        for (&dx, &dz) in surface
            .displacement_x
            .iter()
            .zip(surface.displacement_z.iter())
        {
            assert!(dx.abs() < 1.0e-6 && dz.abs() < 1.0e-6);
        }
    }

    #[test]
    fn rougher_sea_makes_a_taller_surface() {
        let calm = build_initial_spectrum(32, 128.0, sea(6.0), 11);
        let rough = build_initial_spectrum(32, 128.0, sea(16.0), 11);
        let calm_surface = synthesize_surface(&calm, 0.0, 1.0);
        let rough_surface = synthesize_surface(&rough, 0.0, 1.0);
        assert!(rough_surface.vertical_extent() > calm_surface.vertical_extent());
    }

    #[test]
    fn height_field_is_zero_mean() {
        // No energy at DC (Phillips vanishes at k = 0), so the surface averages
        // to zero to the transform's numerical floor.
        let field = build_initial_spectrum(32, 128.0, sea(12.0), 2);
        let surface = synthesize_surface(&field, 5.0, 1.0);
        let mut sum = 0.0f32;
        for &h in &surface.height {
            sum += h;
        }
        let mean = sum / surface.len() as f32;
        assert!(mean.abs() < 1.0e-4, "mean height {mean}");
    }

    #[test]
    fn surface_evolves_over_time() {
        // Distinct times must give distinct surfaces (the sea is moving).
        let field = build_initial_spectrum(16, 64.0, sea(10.0), 9);
        let t0 = synthesize_surface(&field, 0.0, 1.0);
        let t1 = synthesize_surface(&field, 2.0, 1.0);
        assert!(t0 != t1);
    }
}
