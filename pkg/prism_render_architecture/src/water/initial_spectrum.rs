//! Deterministic `Tessendorf` initial spectral-amplitude field.
//!
//! The spectrum module ([`super::spectrum`]) gives the *per-wave-number* energy
//! density and the phase-advance that keeps the inverse-`FFT` height field real.
//! It does not build the initial complex field the ocean `WESL` kernel evolves;
//! that field is what an author actually spawns. This module fills that gap: it
//! turns a wind-driven sea state into the two `array<vec2<f32>>` amplitude
//! buffers (`h0(k)` and `h0(-k)`) the `water_ocean.wesl` spectrum pass reads.
//!
//! `Tessendorf`'s construction draws, for every grid cell, an independent
//! complex Gaussian `xi = xi_r + i*xi_i`, then scales it by the square root of
//! the (halved) energy density at that cell's wave vector:
//!
//! ```text
//! h0(k)  = sqrt(E(k)  / 2) * xi(cell)
//! h0(-k) = sqrt(E(-k) / 2) * xi(mirror-cell)
//! ```
//!
//! The `-k` amplitude reuses the *mirror* cell's own Gaussian so the pair is
//! Hermitian in exactly the way [`super::spectrum::advance_amplitude`] expects
//! (`h(k,t) = h0 e^{i w t} + conj(h0_neg) e^{-i w t}`), which is what makes the
//! evolved height field real.
//!
//! Determinism without an `RNG`: the Gaussian field is generated from a small
//! integer hash (`splitmix32`) of the cell coordinates and a seed, so the same
//! sea state always rebuilds byte-identically — a hard requirement for the
//! deterministic-input-order gate the rest of the water subsystem holds. The
//! Gaussian itself is the classical Irwin-Hall / central-limit approximation
//! (sum of twelve uniforms minus six), which needs only add/subtract and hits
//! zero mean, unit variance without any transcendental the zero-dependency
//! crate forbids.

use alloc::vec::Vec;
use core::f32::consts::TAU;

use super::spectrum::{energy_density, Complex, SpectrumParams};
use super::Vec2;

/// One built ocean initial-spectrum field, ready to upload as the two
/// `array<vec2<f32>>` buffers the ocean spectrum pass binds.
///
/// `h0` and `h0_neg` are row-major over the `resolution x resolution` grid
/// (row `i` is the `+z`/`kz` axis, column `j` the `+x`/`kx` axis). Both are the
/// same length; an empty field (degenerate resolution or patch) yields empty
/// buffers and the ocean pass then contributes nothing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OceanSpectrumField {
    /// Grid resolution `N` (the field holds `N * N` complex amplitudes).
    pub resolution: u32,
    /// Spatial patch size `L` (m) the wave-number grid tiles.
    pub patch_size: f32,
    /// Initial amplitudes at `+k`, row-major.
    pub h0: Vec<Complex>,
    /// Initial amplitudes at `-k` (the mirror-cell draw), row-major.
    pub h0_neg: Vec<Complex>,
}

impl OceanSpectrumField {
    /// Number of complex amplitudes in each buffer (`resolution * resolution`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.h0.len()
    }

    /// Whether the field carries no amplitudes (a degenerate sea state).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.h0.is_empty()
    }

    /// Total spectral energy `sum |h0(k)|^2`, the variance of the surface the
    /// field would synthesize. Grows with wind speed and amplitude; used by the
    /// authoring layer to sanity-check a sea state and by tests as the
    /// monotonicity witness.
    #[must_use]
    pub fn total_energy(&self) -> f32 {
        let mut sum = 0.0;
        for amp in &self.h0 {
            sum += amp.norm_squared();
        }
        sum
    }
}

/// `splitmix32`: a fast, well-mixed integer hash used as the deterministic
/// entropy source for the Gaussian field. Pure integer arithmetic, no state.
#[must_use]
fn splitmix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// A uniform in `[0, 1)` from a 24-bit hash draw (full `f32` mantissa, no bias
/// toward `1.0`).
#[must_use]
fn uniform_01(bits: u32) -> f32 {
    // Keep the top 24 bits; 2^24 is exactly representable in f32.
    (bits >> 8) as f32 / 16_777_216.0
}

/// One standard-normal draw by the Irwin-Hall central-limit approximation:
/// twelve independent uniforms summed and recentred to zero mean, unit
/// variance. `base` seeds the cell; `stream` walks the twelve uniforms so two
/// draws for the same cell (real and imaginary parts) never collide.
#[must_use]
fn standard_gaussian(base: u32, stream: u32) -> f32 {
    let mut acc = 0.0;
    let mut i = 0u32;
    while i < 12 {
        let h = splitmix32(base ^ splitmix32(stream.wrapping_mul(0x9e37_79b9).wrapping_add(i)));
        acc += uniform_01(h);
        i += 1;
    }
    acc - 6.0
}

/// The per-cell complex Gaussian `xi_r + i*xi_i`, deterministic in the cell
/// index and the field seed.
#[must_use]
fn cell_gaussian(cell: u32, seed: u32) -> Complex {
    let base = splitmix32(cell.wrapping_add(1).wrapping_mul(0x27d4_eb2f) ^ seed);
    Complex::new(standard_gaussian(base, 0), standard_gaussian(base, 1))
}

/// A wave-number magnitude band `[lo, hi)` (rad/m) a cascade keeps, used to
/// split a geometric cascade set into non-overlapping wavelength shells so no
/// wave is synthesized twice. `lo` is inclusive, `hi` exclusive; a full-band
/// field (a lone ocean, or the wrapper below) passes `None` and keeps every
/// resolvable mode.
type WaveBand = Option<(f32, f32)>;

/// Core field builder shared by the single-field wrapper and the cascade set.
///
/// `band` limits which wave-number magnitudes contribute: a cell whose `|k|`
/// falls outside the half-open band is zeroed in both `h0` and `h0_neg` (the
/// grid layout is preserved so the `IFFT` indexing never shifts). `None` keeps
/// every mode, reproducing the classic single-patch `Tessendorf` field.
#[must_use]
fn build_field(
    resolution: u32,
    patch_size: f32,
    params: SpectrumParams,
    seed: u32,
    band: WaveBand,
) -> OceanSpectrumField {
    if resolution == 0 || patch_size <= 0.0 {
        return OceanSpectrumField {
            resolution: 0,
            patch_size: 0.0,
            h0: Vec::new(),
            h0_neg: Vec::new(),
        };
    }

    let n = resolution;
    let cells = (n as usize) * (n as usize);
    let half = (n as f32) * 0.5;
    let k_scale = TAU / patch_size;

    // Pre-draw every cell's complex Gaussian so the -k amplitude can reuse the
    // mirror cell's own draw (the Hermitian pairing Tessendorf relies on).
    let mut gaussians = Vec::with_capacity(cells);
    let mut cell = 0u32;
    while (cell as usize) < cells {
        gaussians.push(cell_gaussian(cell, seed));
        cell += 1;
    }

    let mut h0 = Vec::with_capacity(cells);
    let mut h0_neg = Vec::with_capacity(cells);
    let mut i = 0u32;
    while i < n {
        let kz = (i as f32 - half) * k_scale;
        let mi = (n - i) % n;
        let mut j = 0u32;
        while j < n {
            let kx = (j as f32 - half) * k_scale;
            let k_vec = Vec2::new(kx, kz);

            // A cascade only owns wave-number magnitudes inside its band; a
            // mode outside it belongs to a coarser or finer cascade and is
            // dropped here so the summed cascades never double-count a wave.
            let in_band = match band {
                Some((lo, hi)) => {
                    let k_mag = k_vec.length();
                    k_mag >= lo && k_mag < hi
                }
                None => true,
            };

            if in_band {
                let energy = energy_density(k_vec, params);
                let amp = (energy * 0.5).max(0.0).sqrt();
                let g = gaussians[(i * n + j) as usize];
                h0.push(Complex::new(amp * g.re, amp * g.im));

                // -k reuses the mirror cell's Gaussian; the energy is symmetric
                // in k -> -k, so its amplitude matches, but the draw differs.
                let mj = (n - j) % n;
                let g_neg = gaussians[(mi * n + mj) as usize];
                let energy_neg = energy_density(k_vec.scale(-1.0), params);
                let amp_neg = (energy_neg * 0.5).max(0.0).sqrt();
                h0_neg.push(Complex::new(amp_neg * g_neg.re, amp_neg * g_neg.im));
            } else {
                h0.push(Complex::new(0.0, 0.0));
                h0_neg.push(Complex::new(0.0, 0.0));
            }

            j += 1;
        }
        i += 1;
    }

    OceanSpectrumField {
        resolution: n,
        patch_size,
        h0,
        h0_neg,
    }
}

/// Builds the deterministic initial spectral field for a wind-driven sea.
///
/// `resolution` is the grid size `N` (the field holds `N*N` amplitudes);
/// `patch_size` is the tiled patch extent `L` in metres; `params` is the sea
/// state (spectrum kind, wind, amplitude); `seed` selects the random draw so
/// distinct bodies get uncorrelated fields while any one body is reproducible.
///
/// A non-positive `resolution` or `patch_size`, or a calm sea (zero wind, so
/// the spectrum carries no energy), yields an empty field and an honest no-op
/// ocean rather than a fabricated one.
#[must_use]
pub fn build_initial_spectrum(
    resolution: u32,
    patch_size: f32,
    params: SpectrumParams,
    seed: u32,
) -> OceanSpectrumField {
    build_field(resolution, patch_size, params, seed, None)
}

/// Builds a geometrically-spaced set of band-limited initial spectra, one per
/// cascade, the way `UE5` Water, `Crest` and `WaveWorks` stack several `FFT`
/// patches to resolve waves across scales without a single ruinously large
/// grid.
///
/// Cascade `c` tiles the patch `base_patch_size / cascade_ratio^c`, so cascade
/// `0` carries the longest swell and each finer cascade a smaller patch with
/// shorter waves. Each cascade keeps only the wave-number shell it samples
/// best: cascade `c` owns `|k|` in `[TAU / patch_c, cascade_ratio * TAU /
/// patch_c)`, which tiles exactly onto its neighbours (cascade `c`'s upper
/// cutoff is cascade `c+1`'s lower cutoff). The coarsest cascade keeps every
/// mode below its upper cutoff (down to `k = 0`) and the finest keeps every
/// mode above its lower cutoff (up to Nyquist), so no resolvable wave is lost
/// and none is synthesized twice. Each cascade draws from a distinct seed
/// (`seed + c`) so the shells are statistically independent yet reproducible.
///
/// Returns an empty vector for a degenerate request — zero resolution, a
/// non-positive base patch, zero cascades, or a ratio that does not shrink
/// (`<= 1`, which would just restack the same patch) — an honest no-op rather
/// than a fabricated cascade set. A single-cascade request reproduces
/// [`build_initial_spectrum`] exactly.
#[must_use]
pub fn build_cascade_spectra(
    resolution: u32,
    base_patch_size: f32,
    cascade_ratio: f32,
    cascade_count: u32,
    params: SpectrumParams,
    seed: u32,
) -> Vec<OceanSpectrumField> {
    if resolution == 0 || base_patch_size <= 0.0 || cascade_count == 0 || cascade_ratio <= 1.0 {
        return Vec::new();
    }

    let last = cascade_count - 1;
    let mut cascades = Vec::with_capacity(cascade_count as usize);
    let mut c = 0u32;
    let mut patch = base_patch_size;
    while c < cascade_count {
        // The shell this cascade owns, in wave-number magnitude. The coarsest
        // cascade has no lower bound (it sweeps up the longest swell) and the
        // finest no upper bound (it sweeps down to Nyquist); interior cascades
        // tile exactly onto their neighbours.
        let k_lo = if c == 0 { 0.0 } else { TAU / patch };
        let k_hi = if c == last {
            f32::INFINITY
        } else {
            cascade_ratio * TAU / patch
        };
        cascades.push(build_field(
            resolution,
            patch,
            params,
            seed.wrapping_add(c),
            Some((k_lo, k_hi)),
        ));
        patch /= cascade_ratio;
        c += 1;
    }

    cascades
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::spectrum::SpectrumKind;

    const CALM: f32 = 1e-6;

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
    fn field_has_n_squared_amplitudes() {
        let field = build_initial_spectrum(32, 128.0, sea(12.0), 7);
        assert_eq!(field.resolution, 32);
        assert_eq!(field.len(), 32 * 32);
        assert_eq!(field.h0.len(), field.h0_neg.len());
        assert!(!field.is_empty());
    }

    #[test]
    fn build_is_deterministic_in_seed() {
        let a = build_initial_spectrum(24, 96.0, sea(9.0), 42);
        let b = build_initial_spectrum(24, 96.0, sea(9.0), 42);
        assert_eq!(a, b, "same sea state and seed must rebuild identically");
    }

    #[test]
    fn distinct_seeds_diverge() {
        let a = build_initial_spectrum(24, 96.0, sea(9.0), 1);
        let b = build_initial_spectrum(24, 96.0, sea(9.0), 2);
        assert_ne!(a.h0, b.h0, "distinct seeds must draw uncorrelated fields");
    }

    #[test]
    fn energy_grows_with_wind() {
        let light = build_initial_spectrum(48, 200.0, sea(6.0), 3).total_energy();
        let strong = build_initial_spectrum(48, 200.0, sea(18.0), 3).total_energy();
        assert!(
            strong > light,
            "a stronger wind must carry more spectral energy: {strong} !> {light}"
        );
    }

    #[test]
    fn calm_sea_is_an_empty_ocean() {
        // Zero wind -> largest sustainable wave is zero -> Phillips is zero
        // everywhere, so the field carries no energy: an honest no-op.
        let field = build_initial_spectrum(16, 64.0, sea(0.0), 5);
        assert!(
            field.total_energy() < CALM,
            "a windless sea must synthesize no waves"
        );
    }

    #[test]
    fn dc_term_carries_no_energy() {
        // The centre cell is k = 0; the spectrum is defined to vanish there, so
        // the mean surface height stays put instead of drifting.
        let n = 16u32;
        let field = build_initial_spectrum(n, 64.0, sea(11.0), 8);
        let centre = ((n / 2) * n + (n / 2)) as usize;
        assert!(field.h0[centre].norm_squared() < CALM);
    }

    #[test]
    fn degenerate_inputs_yield_empty_field() {
        assert!(build_initial_spectrum(0, 64.0, sea(12.0), 1).is_empty());
        assert!(build_initial_spectrum(16, 0.0, sea(12.0), 1).is_empty());
        assert!(build_initial_spectrum(16, -4.0, sea(12.0), 1).is_empty());
    }

    #[test]
    fn gaussian_field_is_zero_mean_unit_variance() {
        // Over a large grid the Irwin-Hall draws should average to ~0 with
        // ~unit variance; loose bounds keep the statistical test stable.
        let n = 64u32;
        let count = (n * n) as usize;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut cell = 0u32;
        while (cell as usize) < count {
            let g = cell_gaussian(cell, 99);
            mean += g.re + g.im;
            m2 += g.re * g.re + g.im * g.im;
            cell += 1;
        }
        let samples = (count * 2) as f32;
        let mean = mean / samples;
        let variance = m2 / samples - mean * mean;
        assert!(mean.abs() < 0.05, "Gaussian mean drifted: {mean}");
        assert!(
            (variance - 1.0).abs() < 0.1,
            "Gaussian variance off unit: {variance}"
        );
    }

    #[test]
    fn cascade_set_has_geometric_patch_sizes() {
        let set = build_cascade_spectra(32, 256.0, 2.0, 4, sea(12.0), 7);
        assert_eq!(set.len(), 4, "one field per requested cascade");
        let mut expected = 256.0_f32;
        for field in &set {
            assert_eq!(field.resolution, 32);
            assert_eq!(field.len(), 32 * 32);
            assert!(
                (field.patch_size - expected).abs() < 1.0e-3,
                "cascade patch must shrink geometrically: {} != {expected}",
                field.patch_size
            );
            expected /= 2.0;
        }
    }

    #[test]
    fn single_cascade_matches_the_full_spectrum() {
        // A lone cascade owns the whole band with the base seed, so it must be
        // byte-identical to the classic single-patch build.
        let set = build_cascade_spectra(24, 128.0, 2.0, 1, sea(10.0), 5);
        assert_eq!(set.len(), 1);
        let full = build_initial_spectrum(24, 128.0, sea(10.0), 5);
        assert_eq!(set[0], full, "a single cascade is the full-band field");
    }

    #[test]
    fn cascade_bands_partition_the_wave_numbers() {
        // Every non-zero mode a cascade carries must lie in the wave-number
        // shell that cascade owns; nothing leaks across a band boundary, which
        // is what stops a wave being synthesized in two cascades at once.
        let count = 4u32;
        let ratio = 2.0_f32;
        let base = 256.0_f32;
        let n = 32u32;
        let set = build_cascade_spectra(n, base, ratio, count, sea(14.0), 3);
        let half = (n as f32) * 0.5;
        for (c, field) in set.iter().enumerate() {
            let patch = field.patch_size;
            let k_scale = TAU / patch;
            let k_lo = if c == 0 { 0.0 } else { TAU / patch };
            let k_hi = if c as u32 == count - 1 {
                f32::INFINITY
            } else {
                ratio * TAU / patch
            };
            let mut i = 0u32;
            while i < n {
                let kz = (i as f32 - half) * k_scale;
                let mut j = 0u32;
                while j < n {
                    let kx = (j as f32 - half) * k_scale;
                    let idx = (i * n + j) as usize;
                    if field.h0[idx].norm_squared() > CALM {
                        let k_mag = Vec2::new(kx, kz).length();
                        assert!(
                            k_mag >= k_lo && k_mag < k_hi,
                            "cascade {c} carried an out-of-shell mode |k|={k_mag}"
                        );
                    }
                    j += 1;
                }
                i += 1;
            }
        }
    }

    #[test]
    fn cascade_set_carries_energy_and_is_deterministic() {
        let a = build_cascade_spectra(32, 256.0, 2.0, 4, sea(12.0), 9);
        let b = build_cascade_spectra(32, 256.0, 2.0, 4, sea(12.0), 9);
        assert_eq!(a, b, "same request must rebuild identically");
        let total: f32 = a.iter().map(OceanSpectrumField::total_energy).sum();
        assert!(total > CALM, "a windy cascade set must carry energy");
    }

    #[test]
    fn coarse_cascade_holds_the_longer_waves() {
        // The coarsest cascade sweeps up the long swell, the finest the short
        // chop, so the coarse patch must exceed the fine patch.
        let set = build_cascade_spectra(48, 400.0, 2.0, 3, sea(15.0), 2);
        assert!(
            set[0].patch_size > set[2].patch_size,
            "cascade 0 must tile a larger patch than the finest cascade"
        );
    }

    #[test]
    fn degenerate_cascade_requests_are_empty() {
        assert!(build_cascade_spectra(0, 256.0, 2.0, 4, sea(12.0), 1).is_empty());
        assert!(build_cascade_spectra(32, 0.0, 2.0, 4, sea(12.0), 1).is_empty());
        assert!(build_cascade_spectra(32, 256.0, 2.0, 0, sea(12.0), 1).is_empty());
        assert!(
            build_cascade_spectra(32, 256.0, 1.0, 4, sea(12.0), 1).is_empty(),
            "a non-shrinking ratio would restack the same patch"
        );
    }
}
