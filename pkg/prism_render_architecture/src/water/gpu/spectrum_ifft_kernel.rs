//! Ocean spectral inverse-transform compute kernel: the `WESL` shader plus its
//! `CPU` twin, the deterministic golden the production butterfly inverse `FFT`
//! is checked against.
//!
//! The shipping ocean path factors the inverse transform into the separable
//! radix-2 butterfly (`fft_bitrev` / `fft_stage` / `fft_normalize`, scheduled by
//! [`super::spectral_plan`]), exactly as `Tessendorf` / `WaveWorks` / `Crest` /
//! `UE5` Water do. This kernel is the slow but assumption-free reference the
//! butterfly reproduces: an `O(N^4)` direct-sum inverse transform that owns one
//! spatial texel per invocation and sums every frequency cell, so it is exact
//! for any grid edge `N`, not just the power-of-two sizes the butterfly
//! requires.
//!
//! Per texel it evolves every amplitude
//! `h(k, t) = h0(+k) e^{+i w t} + conj(h0(-k)) e^{-i w t}` with the shared
//! [`super::super::spectrum::advance_amplitude`] and accumulates eight real
//! fields through the physical phasor `e^{+i k . x}` (wave vector centred at
//! `k = ((j - N/2), (i - N/2)) * 2*PI / L`, spatial position `x = texel * L/N`):
//! the surface height, the two `Tessendorf` choppiness displacements, the two
//! surface slopes (for the normal), and the three horizontal displacement
//! gradients (for the folding Jacobian). It writes a displacement texture
//! `(Dx, height, Dz, Jacobian)` and a normal texture `(nx, ny, nz, foam)`.
//!
//! [`WATER_SPECTRUM_IFFT_WESL`] is the shader (entry point
//! `water_spectrum_ifft`); [`dispatch_spectrum_ifft`] is its `CPU` twin. Because
//! the sandbox has no `GPU`, the twin is the correctness proof: it consumes the
//! identical buffer `ABI`
//! ([`WaterKernel::SpectrumIfft`](super::super::kernels::WaterKernel) — two read
//! storage buffers for the `h0(+k)` / `h0(-k)` complex spectra, one uniform
//! param block, two `rgba32float` storage-texture outputs, an 8x8 texel tile and
//! the `Grid2d` dispatch domain) and reproduces the shader texel-for-texel,
//! reusing the crate's sanctioned spectral math so the twin is the same
//! numerics the shader inlines. The twin is cross-checked against the fast
//! `CPU` butterfly path [`super::super::synthesis::synthesize_surface`]: same
//! centred spectrum, same dispersion, same Hermitian phase advance must yield
//! the same height and choppiness within the floating-point floor. Pure
//! classical numerics — no AI/ML; the only float intrinsic is `sqrt`
//! (dispersion plus the normal's normalize), with `sin` / `cos` via the crate's
//! libm-free [`super::super::sin_approx`] / [`super::super::cos_approx`].

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::TAU;

use super::super::spectrum::{advance_amplitude, dispersion, Complex};
use super::super::{cos_approx, sin_approx, Vec3, EPS_LEN_SQ};

/// `WESL` source of the ocean spectral inverse-transform compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// `ABI` test; the standalone `naga` / `wesl` compile check runs out of tree,
/// because `prism_render_architecture` is a zero-dependency crate.
pub const WATER_SPECTRUM_IFFT_WESL: &str = include_str!("water_spectrum_ifft.wesl");

/// Number of `f32` lanes per output texel: displacement `(Dx, height, Dz,
/// Jacobian)` and normal `(nx, ny, nz, foam)` are each four lanes.
pub const IFFT_TEXEL_FLOATS: usize = 4;

/// Smallest `|k|` treated as non-zero when forming the choppiness direction
/// `k / |k|`; mirrors the shader `K_EPS` and [`super::super::synthesis`]'s
/// `K_EPS`.
const K_EPS: f32 = 1.0e-6;

/// Per-dispatch spectral inverse-transform scalars, mirroring the shader
/// `IfftParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IfftParams {
    /// Grid edge `N`; the dispatch covers an `N x N` spatial grid and sums over
    /// the `N x N` centred frequency grid.
    pub resolution: u32,
    /// Frame time (s) the amplitudes are advanced to.
    pub time: f32,
    /// Spatial patch size `L` (m) the grid tiles.
    pub patch_size: f32,
    /// `Tessendorf` choppiness scale (`0` = pure height, `1` = full horizontal
    /// drag).
    pub choppiness: f32,
}

/// Output of one inverse-transform dispatch: the per-texel displacement field
/// and the derived normals, each `resolution * resolution` texels of
/// [`IFFT_TEXEL_FLOATS`] lanes in row-major `(gy, gx)` order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpectrumIfftFields {
    /// Displacement texels `(Dx, height, Dz, Jacobian)`.
    pub displacement: Vec<f32>,
    /// Derived normal texels `(nx, ny, nz, foam)`.
    pub normal: Vec<f32>,
}

/// Runs the direct-sum inverse transform for every texel of an `N x N` grid,
/// returning the displacement and normal fields.
///
/// `h0` / `h0_neg` are the centred complex spectra `h0(+k)` / `h0(-k)` in
/// row-major `(i, j)` order (`N * N` entries). A frequency cell whose linear
/// index falls outside either spectrum contributes zero — the shader's
/// `arrayLength` guard — so a short or empty spectrum degrades to a flat
/// surface instead of panicking. `resolution == 0` returns empty fields.
#[must_use]
pub fn dispatch_spectrum_ifft(
    h0: &[Complex],
    h0_neg: &[Complex],
    params: IfftParams,
) -> SpectrumIfftFields {
    let n = params.resolution as usize;
    let texels = n * n;
    let mut displacement = vec![0.0f32; texels * IFFT_TEXEL_FLOATS];
    let mut normal = vec![0.0f32; texels * IFFT_TEXEL_FLOATS];
    if n == 0 {
        return SpectrumIfftFields {
            displacement,
            normal,
        };
    }

    let nf = n as f32;
    let half = nf * 0.5;
    let k_scale = TAU / params.patch_size;
    let cell = params.patch_size / nf;
    let norm = 1.0 / (nf * nf);
    let len0 = h0.len();
    let len1 = h0_neg.len();

    for gy in 0..n {
        let pz = gy as f32 * cell;
        for gx in 0..n {
            let px = gx as f32 * cell;

            let mut acc_h = Complex::ZERO;
            let mut acc_dx = Complex::ZERO;
            let mut acc_dz = Complex::ZERO;
            let mut acc_sx = Complex::ZERO;
            let mut acc_sz = Complex::ZERO;
            let mut acc_gxx = Complex::ZERO;
            let mut acc_gzz = Complex::ZERO;
            let mut acc_gxz = Complex::ZERO;

            for i in 0..n {
                let kz = (i as f32 - half) * k_scale;
                for j in 0..n {
                    let idx = i * n + j;
                    if idx >= len0 || idx >= len1 {
                        continue;
                    }
                    let kx = (j as f32 - half) * k_scale;
                    let k_mag = (kx * kx + kz * kz).sqrt();
                    let omega = dispersion(k_mag);
                    let h = advance_amplitude(h0[idx], h0_neg[idx], omega, params.time);

                    // Physical phasor e^{+i k . x}.
                    let theta = kx * px + kz * pz;
                    let phasor = Complex::new(cos_approx(theta), sin_approx(theta));
                    acc_h = acc_h.add(h.mul(phasor));

                    // Surface slopes i*kx*h and i*kz*h, then phasor.
                    acc_sx = acc_sx.add(Complex::new(0.0, kx).mul(h).mul(phasor));
                    acc_sz = acc_sz.add(Complex::new(0.0, kz).mul(h).mul(phasor));

                    if k_mag > K_EPS {
                        let inv = params.choppiness / k_mag;
                        // Tessendorf horizontal displacement -i*(component/|k|)*h.
                        let dxh = Complex::new(0.0, -kx * inv).mul(h);
                        let dzh = Complex::new(0.0, -kz * inv).mul(h);
                        acc_dx = acc_dx.add(dxh.mul(phasor));
                        acc_dz = acc_dz.add(dzh.mul(phasor));
                        // Displacement gradients for the folding Jacobian.
                        acc_gxx = acc_gxx.add(Complex::new(0.0, kx).mul(dxh).mul(phasor));
                        acc_gzz = acc_gzz.add(Complex::new(0.0, kz).mul(dzh).mul(phasor));
                        acc_gxz = acc_gxz.add(Complex::new(0.0, kz).mul(dxh).mul(phasor));
                    }
                }
            }

            let height = acc_h.re * norm;
            let disp_x = acc_dx.re * norm;
            let disp_z = acc_dz.re * norm;
            let slope_x = acc_sx.re * norm;
            let slope_z = acc_sz.re * norm;
            let d_dx_dx = acc_gxx.re * norm;
            let d_dz_dz = acc_gzz.re * norm;
            let d_dx_dz = acc_gxz.re * norm;

            // Folding Jacobian and a monotonic foam proxy.
            let jacobian = (1.0 + d_dx_dx) * (1.0 + d_dz_dz) - d_dx_dz * d_dx_dz;
            let foam = (1.0 - jacobian).max(0.0);

            // Surface normal from slopes: normalize((-slope_x, 1, -slope_z)).
            let slope = Vec3::new(-slope_x, 1.0, -slope_z);
            let len_sq = slope.dot(slope);
            let nrm = if len_sq > EPS_LEN_SQ {
                slope.scale(1.0 / len_sq.sqrt())
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };

            let out = (gy * n + gx) * IFFT_TEXEL_FLOATS;
            displacement[out] = disp_x;
            displacement[out + 1] = height;
            displacement[out + 2] = disp_z;
            displacement[out + 3] = jacobian;
            normal[out] = nrm.x;
            normal[out + 1] = nrm.y;
            normal[out + 2] = nrm.z;
            normal[out + 3] = foam;
        }
    }

    SpectrumIfftFields {
        displacement,
        normal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::initial_spectrum::build_initial_spectrum;
    use crate::water::spectrum::{SpectrumKind, SpectrumParams};
    use crate::water::synthesis::synthesize_surface;

    fn sea(wind_speed: f32) -> SpectrumParams {
        SpectrumParams {
            kind: SpectrumKind::Phillips,
            wind: crate::water::Vec2::new(wind_speed, 0.0),
            amplitude: 0.5,
            peak_enhancement: 1.0,
            min_wavelength: 0.2,
            directional_exponent: 2,
        }
    }

    /// The shipped `WESL` honors the `SpectrumIfft` descriptor's `ABI`:
    /// entry-point name, 8x8 tile, two read storage buffers, the uniform block,
    /// the two `rgba32float` storage-texture outputs, plus the `textureStore` /
    /// `arrayLength` the twin's zero-contribution guard mirrors.
    #[test]
    fn wesl_matches_descriptor_abi() {
        let src = WATER_SPECTRUM_IFFT_WESL;
        assert!(src.contains("fn water_spectrum_ifft("));
        assert!(src.contains("@workgroup_size(8, 8, 1)"));
        assert!(src.contains("var<uniform> params: IfftParams"));
        assert_eq!(src.matches("var<storage, read>").count(), 2);
        assert_eq!(
            src.matches("texture_storage_2d<rgba32float, write>")
                .count(),
            2
        );
        assert_eq!(src.matches("arrayLength").count(), 2);
        assert!(src.contains("textureStore(out_displacement"));
        assert!(src.contains("textureStore(out_normal"));

        use super::super::super::kernels::{DispatchDomain, WaterKernel};
        let desc = WaterKernel::SpectrumIfft.descriptor();
        assert_eq!(desc.layout.storage_buffers, 2);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 2);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(desc.workgroup.x, 8);
        assert_eq!(desc.workgroup.y, 8);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Grid2d);
        assert_eq!(
            WaterKernel::SpectrumIfft.wesl_entry_point(),
            "water_spectrum_ifft"
        );
    }

    /// Core anti-tautology: the direct-sum twin reproduces the independent fast
    /// butterfly path. For the same centred spectrum, height (and the two
    /// choppiness displacements) must match [`synthesize_surface`] texel-for-texel
    /// within the floating-point floor — two different algorithms, one answer.
    #[test]
    fn matches_fast_butterfly_synthesis() {
        let n = 8u32;
        let patch = 64.0f32;
        let time = 3.0f32;
        let chop = 1.0f32;
        let field = build_initial_spectrum(n, patch, sea(11.0), 7);
        let surface = synthesize_surface(&field, time, chop);
        assert_eq!(surface.len(), (n * n) as usize);

        let fields = dispatch_spectrum_ifft(
            &field.h0,
            &field.h0_neg,
            IfftParams {
                resolution: n,
                time,
                patch_size: patch,
                choppiness: chop,
            },
        );

        let mut max_err = 0.0f32;
        for t in 0..(n * n) as usize {
            let i = t * IFFT_TEXEL_FLOATS;
            let dh = (fields.displacement[i + 1] - surface.height[t]).abs();
            let ddx = (fields.displacement[i] - surface.displacement_x[t]).abs();
            let ddz = (fields.displacement[i + 2] - surface.displacement_z[t]).abs();
            max_err = max_err.max(dh).max(ddx).max(ddz);
        }
        // The two paths are mathematically identical (centred spectrum, same
        // dispersion, same Hermitian phase advance), so they agree only up to
        // the floating-point floor. The direct sum evaluates `N*N` phasors per
        // texel through the crate's polynomial `sin_approx`/`cos_approx`, while
        // the butterfly composes `log2(N)` radix-2 stages, so their rounding
        // trails diverge; the gap grows with the phasor count and lands near
        // 2.1e-3 at `N = 8`. 3e-3 keeps a tight anti-drift guard while
        // tolerating that cross-algorithm trig accumulation (the sibling
        // inline-recompute test pins the twin's own numerics bit-for-bit).
        assert!(
            max_err < 3.0e-3,
            "direct-sum vs butterfly disagree by {max_err}"
        );
    }

    /// A zero-wind (calm) sea carries no spectral energy, so every output texel
    /// is a flat surface: zero displacement, unit Jacobian, zero foam, upright
    /// normal.
    #[test]
    fn calm_sea_is_flat() {
        let n = 8u32;
        let field = build_initial_spectrum(n, 64.0, sea(0.0), 1);
        let fields = dispatch_spectrum_ifft(
            &field.h0,
            &field.h0_neg,
            IfftParams {
                resolution: n,
                time: 2.0,
                patch_size: 64.0,
                choppiness: 1.0,
            },
        );
        for t in 0..(n * n) as usize {
            let i = t * IFFT_TEXEL_FLOATS;
            assert!(fields.displacement[i].abs() < 1.0e-6);
            assert!(fields.displacement[i + 1].abs() < 1.0e-6);
            assert!(fields.displacement[i + 2].abs() < 1.0e-6);
            // Jacobian of a flat surface is 1 -> foam 0.
            assert!((fields.displacement[i + 3] - 1.0).abs() < 1.0e-6);
            assert!(fields.normal[i].abs() < 1.0e-6);
            assert!((fields.normal[i + 1] - 1.0).abs() < 1.0e-6);
            assert!(fields.normal[i + 2].abs() < 1.0e-6);
            assert!(fields.normal[i + 3].abs() < 1.0e-6);
        }
    }

    /// Independent inline recomputation of the full field, lane-for-lane, binds
    /// the twin to the documented direct-sum algorithm (anti-vacuous: at least
    /// one height must be non-zero so the sum is exercised).
    #[test]
    fn matches_independent_inline_recompute() {
        let n = 6usize;
        let patch = 40.0f32;
        let time = 1.5f32;
        let chop = 0.7f32;
        let field = build_initial_spectrum(n as u32, patch, sea(9.0), 5);
        let fields = dispatch_spectrum_ifft(
            &field.h0,
            &field.h0_neg,
            IfftParams {
                resolution: n as u32,
                time,
                patch_size: patch,
                choppiness: chop,
            },
        );

        let nf = n as f32;
        let half = nf * 0.5;
        let k_scale = TAU / patch;
        let cell = patch / nf;
        let norm = 1.0 / (nf * nf);
        let mut saw_nonzero = false;

        for gy in 0..n {
            let pz = gy as f32 * cell;
            for gx in 0..n {
                let px = gx as f32 * cell;
                let mut acc_h = Complex::ZERO;
                let mut acc_dx = Complex::ZERO;
                let mut acc_dz = Complex::ZERO;
                let mut acc_sx = Complex::ZERO;
                let mut acc_sz = Complex::ZERO;
                let mut acc_gxx = Complex::ZERO;
                let mut acc_gzz = Complex::ZERO;
                let mut acc_gxz = Complex::ZERO;
                for i in 0..n {
                    let kz = (i as f32 - half) * k_scale;
                    for j in 0..n {
                        let idx = i * n + j;
                        let kx = (j as f32 - half) * k_scale;
                        let k_mag = (kx * kx + kz * kz).sqrt();
                        let omega = dispersion(k_mag);
                        let h = advance_amplitude(field.h0[idx], field.h0_neg[idx], omega, time);
                        let theta = kx * px + kz * pz;
                        let phasor = Complex::new(cos_approx(theta), sin_approx(theta));
                        acc_h = acc_h.add(h.mul(phasor));
                        acc_sx = acc_sx.add(Complex::new(0.0, kx).mul(h).mul(phasor));
                        acc_sz = acc_sz.add(Complex::new(0.0, kz).mul(h).mul(phasor));
                        if k_mag > K_EPS {
                            let inv = chop / k_mag;
                            let dxh = Complex::new(0.0, -kx * inv).mul(h);
                            let dzh = Complex::new(0.0, -kz * inv).mul(h);
                            acc_dx = acc_dx.add(dxh.mul(phasor));
                            acc_dz = acc_dz.add(dzh.mul(phasor));
                            acc_gxx = acc_gxx.add(Complex::new(0.0, kx).mul(dxh).mul(phasor));
                            acc_gzz = acc_gzz.add(Complex::new(0.0, kz).mul(dzh).mul(phasor));
                            acc_gxz = acc_gxz.add(Complex::new(0.0, kz).mul(dxh).mul(phasor));
                        }
                    }
                }
                let height = acc_h.re * norm;
                let disp_x = acc_dx.re * norm;
                let disp_z = acc_dz.re * norm;
                let slope_x = acc_sx.re * norm;
                let slope_z = acc_sz.re * norm;
                let d_dx_dx = acc_gxx.re * norm;
                let d_dz_dz = acc_gzz.re * norm;
                let d_dx_dz = acc_gxz.re * norm;
                let jacobian = (1.0 + d_dx_dx) * (1.0 + d_dz_dz) - d_dx_dz * d_dx_dz;
                let foam = (1.0 - jacobian).max(0.0);
                let len_sq = slope_x * slope_x + 1.0 + slope_z * slope_z;
                let (nx, ny, nz) = if len_sq > EPS_LEN_SQ {
                    let r = 1.0 / len_sq.sqrt();
                    (-slope_x * r, 1.0 * r, -slope_z * r)
                } else {
                    (0.0, 1.0, 0.0)
                };

                let o = (gy * n + gx) * IFFT_TEXEL_FLOATS;
                assert_eq!(fields.displacement[o].to_bits(), disp_x.to_bits());
                assert_eq!(fields.displacement[o + 1].to_bits(), height.to_bits());
                assert_eq!(fields.displacement[o + 2].to_bits(), disp_z.to_bits());
                assert_eq!(fields.displacement[o + 3].to_bits(), jacobian.to_bits());
                assert_eq!(fields.normal[o].to_bits(), nx.to_bits());
                assert_eq!(fields.normal[o + 1].to_bits(), ny.to_bits());
                assert_eq!(fields.normal[o + 2].to_bits(), nz.to_bits());
                assert_eq!(fields.normal[o + 3].to_bits(), foam.to_bits());
                if height.abs() > 1.0e-6 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "a non-calm sea must produce non-zero height");
    }

    /// Every emitted normal is unit length (within the `f32` floor) on a
    /// non-trivial sea, confirming the normalize path is exercised and correct.
    #[test]
    fn normals_are_unit_length() {
        let n = 8u32;
        let field = build_initial_spectrum(n, 64.0, sea(13.0), 3);
        let fields = dispatch_spectrum_ifft(
            &field.h0,
            &field.h0_neg,
            IfftParams {
                resolution: n,
                time: 4.0,
                patch_size: 64.0,
                choppiness: 1.0,
            },
        );
        for t in 0..(n * n) as usize {
            let i = t * IFFT_TEXEL_FLOATS;
            let nx = fields.normal[i];
            let ny = fields.normal[i + 1];
            let nz = fields.normal[i + 2];
            let len_sq = nx * nx + ny * ny + nz * nz;
            assert!((len_sq - 1.0).abs() < 1.0e-5, "normal not unit: {len_sq}");
        }
    }

    /// A short or empty spectrum skips the out-of-range cells instead of
    /// panicking, and `resolution == 0` returns empty fields.
    #[test]
    fn short_and_empty_spectra_do_not_panic() {
        let n = 4u32;
        // One lonely cell: every other frequency index is skipped.
        let h0 = alloc::vec![Complex::new(0.3, -0.1)];
        let h0_neg = alloc::vec![Complex::new(0.2, 0.4)];
        let fields = dispatch_spectrum_ifft(
            &h0,
            &h0_neg,
            IfftParams {
                resolution: n,
                time: 1.0,
                patch_size: 32.0,
                choppiness: 1.0,
            },
        );
        assert_eq!(
            fields.displacement.len(),
            (n * n) as usize * IFFT_TEXEL_FLOATS
        );
        assert_eq!(fields.normal.len(), (n * n) as usize * IFFT_TEXEL_FLOATS);

        // Entirely empty spectra -> flat surface, no panic.
        let flat = dispatch_spectrum_ifft(
            &[],
            &[],
            IfftParams {
                resolution: n,
                time: 1.0,
                patch_size: 32.0,
                choppiness: 1.0,
            },
        );
        for t in 0..(n * n) as usize {
            let i = t * IFFT_TEXEL_FLOATS;
            assert!(flat.displacement[i + 1].abs() < 1.0e-9);
            assert!((flat.normal[i + 1] - 1.0).abs() < 1.0e-9);
        }

        // resolution == 0 -> empty fields.
        let empty = dispatch_spectrum_ifft(
            &h0,
            &h0_neg,
            IfftParams {
                resolution: 0,
                time: 1.0,
                patch_size: 32.0,
                choppiness: 1.0,
            },
        );
        assert!(empty.displacement.is_empty());
        assert!(empty.normal.is_empty());
    }

    /// The dispatch is deterministic: identical inputs produce bit-identical
    /// fields on repeated calls.
    #[test]
    fn deterministic_across_calls() {
        let n = 8u32;
        let field = build_initial_spectrum(n, 64.0, sea(10.0), 42);
        let params = IfftParams {
            resolution: n,
            time: 2.5,
            patch_size: 64.0,
            choppiness: 0.8,
        };
        let a = dispatch_spectrum_ifft(&field.h0, &field.h0_neg, params);
        let b = dispatch_spectrum_ifft(&field.h0, &field.h0_neg, params);
        assert_eq!(a, b);
    }
}
