//! Ocean spectral evolve compute kernel: the `WESL` shader plus its `CPU` twin.
//!
//! This is stage 1 of the separable `Tessendorf` ocean's three-stage spectral
//! schedule (`evolve` -> butterfly -> assemble, see [`super::spectral_plan`]),
//! exactly how `WaveWorks` / `Crest` / `UE5` Water factor the runtime inverse
//! transform instead of direct-summing the `O(N^4)` reference
//! ([`super::spectrum_ifft_kernel`] is that golden). The evolve stage owns one
//! frequency cell per invocation. It advances the two initial amplitudes
//! `h0(+k)` / `h0(-k)` to the frame time with the shared
//! [`super::super::spectrum::advance_amplitude`], forms the eight real
//! output-field spectra per cell, and packs them into four complex buffers the
//! radix-2 butterfly (`fft_bitrev` / `fft_stage` / `fft_normalize`) consumes.
//!
//! The packing amortises the transform: for two real spatial fields `a`, `b`,
//! the complex spectrum `A(k) + i*B(k)` inverse-transforms to `a(x) + i*b(x)`
//! (both fields have `Hermitian` spectra, so the real part recovers `a` and the
//! imaginary part recovers `b`). The eight real fields therefore need only four
//! complex butterflies. The fixed field ordering mirrors
//! [`super::spectral_plan`]'s `SpectralRealField::ALL`
//! (`height`, `dispX`, `dispZ`, `slopeX`, `slopeZ`, `gradXx`, `gradZz`,
//! `gradXz`), two consecutive fields sharing one buffer:
//!   `buf0 = height + i*dispX`, `buf1 = dispZ + i*slopeX`,
//!   `buf2 = slopeZ + i*gradXx`, `buf3 = gradZz + i*gradXz`.
//!
//! The per-cell amplitudes are the [`super::spectrum_ifft_kernel`] accumulands
//! before its phasor sum (the phasor sum is the butterfly's job): `height = h`,
//! `slopeX = i*kx*h`, `slopeZ = i*kz*h`,
//! `dispX = -i*(kx/|k|)*chop*h`, `dispZ` likewise, `gradXx = i*kx*dispX`,
//! `gradZz = i*kz*dispZ`, `gradXz = i*kz*dispX`.
//!
//! [`WATER_SPECTRUM_EVOLVE_WESL`] is the shader (entry point
//! `water_spectrum_evolve`); [`dispatch_spectrum_evolve`] is its `CPU` twin.
//! Because the sandbox has no `GPU`, the twin is the correctness proof: it
//! consumes the identical buffer `ABI`
//! ([`WaterKernel::SpectrumEvolve`](super::super::kernels::WaterKernel) — six
//! storage buffers, two read for the `h0(+k)` / `h0(-k)` spectra and four
//! written packed complex buffers, one uniform param block, no textures, an 8x8
//! tile, the `Grid2d` dispatch domain) and reproduces the shader cell-for-cell,
//! reusing the crate's sanctioned spectral math so the twin inlines the same
//! numerics the shader does. The twin is cross-checked against the independent
//! direct-sum [`super::spectrum_ifft_kernel::dispatch_spectrum_ifft`]: a
//! reference inverse transform of the four packed buffers, unpacked back into
//! the eight fields, must reconstruct the same displacement, `Jacobian`, and
//! normals the direct-sum ifft produces. Pure classical numerics — no `AI` /
//! `ML`; the only float intrinsic is `sqrt` (dispersion), with `sin` / `cos`
//! via the crate's libm-free [`super::super::sin_approx`] /
//! [`super::super::cos_approx`].

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::TAU;

use super::super::spectrum::{advance_amplitude, dispersion, Complex};

/// `WESL` source of the ocean spectral evolve compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// `ABI` test; the standalone `naga` / `wesl` compile check runs out of tree,
/// because `prism_render_architecture` is a zero-dependency crate.
pub const WATER_SPECTRUM_EVOLVE_WESL: &str = include_str!("water_spectrum_evolve.wesl");

/// Number of packed complex output buffers: the eight real fields map onto four
/// `Hermitian` complex transforms (two real fields per complex buffer).
pub const EVOLVE_PACKED_FIELDS: usize = 4;

/// `f32` lanes per complex entry in each packed buffer: real then imaginary.
pub const EVOLVE_COMPLEX_FLOATS: usize = 2;

/// Smallest `|k|` treated as non-zero when forming the choppiness direction
/// `k / |k|`; mirrors the shader `K_EPS` and [`super::spectrum_ifft_kernel`].
const K_EPS: f32 = 1.0e-6;

/// Per-dispatch spectral evolve scalars, mirroring the shader `EvolveParams`
/// (an identical layout to [`super::spectrum_ifft_kernel::IfftParams`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvolveParams {
    /// Grid edge `N`; the dispatch covers the `N x N` centred frequency grid.
    pub resolution: u32,
    /// Frame time (s) the amplitudes are advanced to.
    pub time: f32,
    /// Spatial patch size `L` (m) the grid tiles.
    pub patch_size: f32,
    /// `Tessendorf` choppiness scale (`0` = pure height, `1` = full horizontal
    /// drag).
    pub choppiness: f32,
}

/// Output of one evolve dispatch: the four packed complex spectra the butterfly
/// consumes, each `resolution * resolution` complex entries of
/// [`EVOLVE_COMPLEX_FLOATS`] lanes in row-major `(gy, gx)` order.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumEvolveBuffers {
    /// The [`EVOLVE_PACKED_FIELDS`] packed complex buffers, in the fixed
    /// `spectral_plan` field-pair order.
    pub packed: [Vec<f32>; EVOLVE_PACKED_FIELDS],
}

/// Packs two complex spectra `a`, `b` into one: `a + i*b`. With `a = (ar, ai)`
/// and `b = (br, bi)`, `i*b = (-bi, br)`, so `a + i*b = (ar - bi, ai + br)`.
/// Inverse-transforming this buffer yields `a(x) + i*b(x)` for the `Hermitian`
/// field pair, which the assemble stage unpacks as real -> `a`, imag -> `b`.
#[inline]
fn cpack(a: Complex, b: Complex) -> [f32; EVOLVE_COMPLEX_FLOATS] {
    [a.re - b.im, a.im + b.re]
}

/// Evolves and packs every frequency cell of an `N x N` grid, returning the
/// four packed complex spectra.
///
/// `h0` / `h0_neg` are the centred complex spectra `h0(+k)` / `h0(-k)` in
/// row-major `(i, j)` order (`N * N` entries). A cell whose linear index falls
/// outside either spectrum contributes zero — the shader's `arrayLength` guard
/// — so a short or empty spectrum degrades to all-zero buffers instead of
/// panicking. `resolution == 0` returns empty buffers.
#[must_use]
pub fn dispatch_spectrum_evolve(
    h0: &[Complex],
    h0_neg: &[Complex],
    params: EvolveParams,
) -> SpectrumEvolveBuffers {
    let n = params.resolution as usize;
    let texels = n * n;
    let mut packed: [Vec<f32>; EVOLVE_PACKED_FIELDS] =
        core::array::from_fn(|_| vec![0.0f32; texels * EVOLVE_COMPLEX_FLOATS]);
    if n == 0 {
        return SpectrumEvolveBuffers { packed };
    }

    let nf = n as f32;
    let half = nf * 0.5;
    let k_scale = TAU / params.patch_size;
    let len0 = h0.len();
    let len1 = h0_neg.len();

    for gy in 0..n {
        let kz = (gy as f32 - half) * k_scale;
        for gx in 0..n {
            let idx = gy * n + gx;
            // Short/empty spectra: a cell with no backing amplitude stays zero
            // in every packed buffer (cpack(0, 0) == 0, the pre-initialised
            // value), a deterministic skip rather than a garbled read.
            if idx >= len0 || idx >= len1 {
                continue;
            }

            let kx = (gx as f32 - half) * k_scale;
            let k_mag = (kx * kx + kz * kz).sqrt();
            let omega = dispersion(k_mag);
            let h = advance_amplitude(h0[idx], h0_neg[idx], omega, params.time);

            let height = h;
            // Surface slopes i*kx*h and i*kz*h (zero at k=0, so unconditional).
            let slope_x = Complex::new(0.0, kx).mul(h);
            let slope_z = Complex::new(0.0, kz).mul(h);

            let (disp_x, disp_z, grad_xx, grad_zz, grad_xz) = if k_mag > K_EPS {
                let inv = params.choppiness / k_mag;
                // Tessendorf horizontal displacement -i*(component/|k|)*h.
                let dxh = Complex::new(0.0, -kx * inv).mul(h);
                let dzh = Complex::new(0.0, -kz * inv).mul(h);
                // Displacement gradients for the folding Jacobian; the trailing
                // unit multiply mirrors the shader's structural phasor slot
                // (identity here, the phasor sum is the butterfly's job).
                let gxx = Complex::new(0.0, kx).mul(dxh).mul(Complex::new(1.0, 0.0));
                let gzz = Complex::new(0.0, kz).mul(dzh).mul(Complex::new(1.0, 0.0));
                let gxz = Complex::new(0.0, kz).mul(dxh).mul(Complex::new(1.0, 0.0));
                (dxh, dzh, gxx, gzz, gxz)
            } else {
                (
                    Complex::ZERO,
                    Complex::ZERO,
                    Complex::ZERO,
                    Complex::ZERO,
                    Complex::ZERO,
                )
            };

            // Pack the eight fields into the four complex buffers.
            let b0 = cpack(height, disp_x);
            let b1 = cpack(disp_z, slope_x);
            let b2 = cpack(slope_z, grad_xx);
            let b3 = cpack(grad_zz, grad_xz);

            let o = idx * EVOLVE_COMPLEX_FLOATS;
            packed[0][o] = b0[0];
            packed[0][o + 1] = b0[1];
            packed[1][o] = b1[0];
            packed[1][o + 1] = b1[1];
            packed[2][o] = b2[0];
            packed[2][o + 1] = b2[1];
            packed[3][o] = b3[0];
            packed[3][o + 1] = b3[1];
        }
    }

    SpectrumEvolveBuffers { packed }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::initial_spectrum::build_initial_spectrum;
    use crate::water::spectrum::{SpectrumKind, SpectrumParams};
    use crate::water::{cos_approx, sin_approx, Vec3, EPS_LEN_SQ};
    use core::f32::consts::TAU;

    use super::super::spectrum_ifft_kernel::{
        dispatch_spectrum_ifft, IfftParams, IFFT_TEXEL_FLOATS,
    };

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

    /// The shipped `WESL` honors the `SpectrumEvolve` descriptor's `ABI`:
    /// entry-point name, 8x8 tile, six storage buffers (two read spectra plus
    /// four read-write packed buffers), the uniform block, no textures, plus
    /// the `arrayLength` guards the twin's zero-contribution path mirrors.
    #[test]
    fn wesl_matches_descriptor_abi() {
        let src = WATER_SPECTRUM_EVOLVE_WESL;
        assert!(src.contains("fn water_spectrum_evolve("));
        assert!(src.contains("@workgroup_size(8, 8, 1)"));
        assert!(src.contains("var<uniform> params: EvolveParams"));
        assert_eq!(src.matches("var<storage, read>").count(), 2);
        assert_eq!(src.matches("var<storage, read_write>").count(), 4);
        assert_eq!(src.matches("texture_storage").count(), 0);

        use super::super::super::kernels::{DispatchDomain, WaterKernel};
        let desc = WaterKernel::SpectrumEvolve.descriptor();
        assert_eq!(desc.layout.storage_buffers, 6);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 0);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(desc.workgroup.x, 8);
        assert_eq!(desc.workgroup.y, 8);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Grid2d);
        assert_eq!(
            WaterKernel::SpectrumEvolve.wesl_entry_point(),
            "water_spectrum_evolve"
        );
    }

    /// Independent inline recompute of every cell's four packed buffers, pinned
    /// bit-for-bit against the dispatch, plus an anti-vacuous guard that a
    /// non-calm sea produces at least one non-zero lane.
    #[test]
    fn matches_independent_inline_recompute() {
        let n = 8u32;
        let patch = 64.0f32;
        let field = build_initial_spectrum(n, patch, sea(12.0), 5);
        let params = EvolveParams {
            resolution: n,
            time: 2.0,
            patch_size: patch,
            choppiness: 0.9,
        };
        let out = dispatch_spectrum_evolve(&field.h0, &field.h0_neg, params);

        let nn = n as usize;
        let nf = nn as f32;
        let half = nf * 0.5;
        let k_scale = TAU / patch;
        let mut saw_nonzero = false;

        for gy in 0..nn {
            let kz = (gy as f32 - half) * k_scale;
            for gx in 0..nn {
                let idx = gy * nn + gx;
                let kx = (gx as f32 - half) * k_scale;
                let k_mag = (kx * kx + kz * kz).sqrt();
                let omega = dispersion(k_mag);
                let h = advance_amplitude(field.h0[idx], field.h0_neg[idx], omega, params.time);
                let slope_x = Complex::new(0.0, kx).mul(h);
                let slope_z = Complex::new(0.0, kz).mul(h);
                let (disp_x, disp_z, grad_xx, grad_zz, grad_xz) = if k_mag > K_EPS {
                    let inv = params.choppiness / k_mag;
                    let dxh = Complex::new(0.0, -kx * inv).mul(h);
                    let dzh = Complex::new(0.0, -kz * inv).mul(h);
                    let gxx = Complex::new(0.0, kx).mul(dxh).mul(Complex::new(1.0, 0.0));
                    let gzz = Complex::new(0.0, kz).mul(dzh).mul(Complex::new(1.0, 0.0));
                    let gxz = Complex::new(0.0, kz).mul(dxh).mul(Complex::new(1.0, 0.0));
                    (dxh, dzh, gxx, gzz, gxz)
                } else {
                    (
                        Complex::ZERO,
                        Complex::ZERO,
                        Complex::ZERO,
                        Complex::ZERO,
                        Complex::ZERO,
                    )
                };
                let want = [
                    cpack(h, disp_x),
                    cpack(disp_z, slope_x),
                    cpack(slope_z, grad_xx),
                    cpack(grad_zz, grad_xz),
                ];
                let o = idx * EVOLVE_COMPLEX_FLOATS;
                for (c, w) in want.iter().enumerate() {
                    assert_eq!(out.packed[c][o].to_bits(), w[0].to_bits());
                    assert_eq!(out.packed[c][o + 1].to_bits(), w[1].to_bits());
                    if w[0].abs() > 1.0e-6 || w[1].abs() > 1.0e-6 {
                        saw_nonzero = true;
                    }
                }
            }
        }
        assert!(saw_nonzero, "a non-calm sea must produce non-zero spectra");
    }

    /// Core anti-tautology: inverse-transform the four packed buffers (an
    /// independent direct-sum reference) and unpack them back into the eight
    /// real fields, then rebuild displacement, `Jacobian`, and normals exactly
    /// as the sibling [`dispatch_spectrum_ifft`] does — and require the two
    /// paths to agree. The evolve kernel separates per-cell amplitude from the
    /// transform and packs two fields per complex buffer; the ifft direct-sums
    /// each field through its own phasor. If the amplitude formulas or the
    /// pack/unpack were wrong, the reconstruction would diverge.
    #[test]
    fn inverse_transform_matches_reference_ifft() {
        let n = 8u32;
        let patch = 64.0f32;
        let time = 3.0f32;
        let chop = 1.0f32;
        let mut field = build_initial_spectrum(n, patch, sea(11.0), 7);
        // The two-fields-per-complex-buffer packing is exact only when every
        // field spectrum is Hermitian. On an even grid the Nyquist row/column
        // (i == 0 or j == 0) has no in-grid conjugate partner, so production
        // oceans force those bins to zero. Mirror that here so the packed
        // inverse-split reconstruction is exact and the cross-check is tight.
        let nn0 = n as usize;
        for i in 0..nn0 {
            for j in 0..nn0 {
                if i == 0 || j == 0 {
                    let idx = i * nn0 + j;
                    field.h0[idx] = Complex::ZERO;
                    field.h0_neg[idx] = Complex::ZERO;
                }
            }
        }

        let buffers = dispatch_spectrum_evolve(
            &field.h0,
            &field.h0_neg,
            EvolveParams {
                resolution: n,
                time,
                patch_size: patch,
                choppiness: chop,
            },
        );
        let reference = dispatch_spectrum_ifft(
            &field.h0,
            &field.h0_neg,
            IfftParams {
                resolution: n,
                time,
                patch_size: patch,
                choppiness: chop,
            },
        );

        let nn = n as usize;
        let nf = nn as f32;
        let half = nf * 0.5;
        let k_scale = TAU / patch;
        let cell = patch / nf;
        let norm = 1.0 / (nf * nf);

        // Read a packed buffer entry as a complex spectrum coefficient.
        let coeff = |c: usize, idx: usize| -> Complex {
            let o = idx * EVOLVE_COMPLEX_FLOATS;
            Complex::new(buffers.packed[c][o], buffers.packed[c][o + 1])
        };

        let mut max_err = 0.0f32;
        for gy in 0..nn {
            let pz = gy as f32 * cell;
            for gx in 0..nn {
                let px = gx as f32 * cell;

                // Direct-sum inverse transform of each packed buffer through
                // the physical phasor e^{+i k . x}.
                let mut acc = [Complex::ZERO; EVOLVE_PACKED_FIELDS];
                for i in 0..nn {
                    let kz = (i as f32 - half) * k_scale;
                    for j in 0..nn {
                        let idx = i * nn + j;
                        let kx = (j as f32 - half) * k_scale;
                        let theta = kx * px + kz * pz;
                        let phasor = Complex::new(cos_approx(theta), sin_approx(theta));
                        for (c, a) in acc.iter_mut().enumerate() {
                            *a = a.add(coeff(c, idx).mul(phasor));
                        }
                    }
                }

                // Unpack: real part -> field a, imaginary part -> field b.
                // (Re(p) = a - Im(spectrum_b); the Hermitian imaginary leakage
                // is near zero, so a 3e-3 floor mirrors the ifft butterfly
                // cross-check.)
                let height = acc[0].re * norm;
                let disp_x = acc[0].im * norm;
                let disp_z = acc[1].re * norm;
                let slope_x = acc[1].im * norm;
                let slope_z = acc[2].re * norm;
                let grad_xx = acc[2].im * norm;
                let grad_zz = acc[3].re * norm;
                let grad_xz = acc[3].im * norm;

                let jacobian = (1.0 + grad_xx) * (1.0 + grad_zz) - grad_xz * grad_xz;
                let foam = (1.0 - jacobian).max(0.0);
                let slope = Vec3::new(-slope_x, 1.0, -slope_z);
                let len_sq = slope.dot(slope);
                let nrm = if len_sq > EPS_LEN_SQ {
                    slope.scale(1.0 / len_sq.sqrt())
                } else {
                    Vec3::new(0.0, 1.0, 0.0)
                };

                let o = (gy * nn + gx) * IFFT_TEXEL_FLOATS;
                max_err = max_err
                    .max((disp_x - reference.displacement[o]).abs())
                    .max((height - reference.displacement[o + 1]).abs())
                    .max((disp_z - reference.displacement[o + 2]).abs())
                    .max((jacobian - reference.displacement[o + 3]).abs())
                    .max((nrm.x - reference.normal[o]).abs())
                    .max((nrm.y - reference.normal[o + 1]).abs())
                    .max((nrm.z - reference.normal[o + 2]).abs())
                    .max((foam - reference.normal[o + 3]).abs());
            }
        }
        assert!(
            max_err < 3.0e-3,
            "evolve-then-inverse-transform disagrees with direct-sum ifft by {max_err}"
        );
    }

    /// A calm (zero-wind) sea carries no spectral energy, so every packed lane
    /// is zero.
    #[test]
    fn calm_sea_all_zero() {
        let n = 8u32;
        let patch = 64.0f32;
        let field = build_initial_spectrum(n, patch, sea(0.0), 1);
        let out = dispatch_spectrum_evolve(
            &field.h0,
            &field.h0_neg,
            EvolveParams {
                resolution: n,
                time: 2.0,
                patch_size: patch,
                choppiness: 1.0,
            },
        );
        for buf in &out.packed {
            for &lane in buf {
                assert!(
                    lane.abs() < 1.0e-6,
                    "calm sea produced non-zero lane {lane}"
                );
            }
        }
    }

    /// A short or empty spectrum skips the out-of-range cells instead of
    /// panicking, and `resolution == 0` returns empty buffers.
    #[test]
    fn short_and_empty_inputs_do_not_panic() {
        let n = 4u32;
        let h0 = alloc::vec![Complex::new(0.3, -0.1)];
        let h0_neg = alloc::vec![Complex::new(0.2, 0.4)];
        let out = dispatch_spectrum_evolve(
            &h0,
            &h0_neg,
            EvolveParams {
                resolution: n,
                time: 1.0,
                patch_size: 32.0,
                choppiness: 1.0,
            },
        );
        for buf in &out.packed {
            assert_eq!(buf.len(), (n * n) as usize * EVOLVE_COMPLEX_FLOATS);
        }

        let empty = dispatch_spectrum_evolve(
            &[],
            &[],
            EvolveParams {
                resolution: n,
                time: 1.0,
                patch_size: 32.0,
                choppiness: 1.0,
            },
        );
        for buf in &empty.packed {
            for &lane in buf {
                assert!(lane.abs() < 1.0e-9);
            }
        }

        let zero = dispatch_spectrum_evolve(
            &h0,
            &h0_neg,
            EvolveParams {
                resolution: 0,
                time: 1.0,
                patch_size: 32.0,
                choppiness: 1.0,
            },
        );
        for buf in &zero.packed {
            assert!(buf.is_empty());
        }
    }

    /// The dispatch is deterministic: identical inputs produce bit-identical
    /// buffers on repeated calls.
    #[test]
    fn deterministic_across_calls() {
        let n = 8u32;
        let patch = 64.0f32;
        let field = build_initial_spectrum(n, patch, sea(10.0), 42);
        let params = EvolveParams {
            resolution: n,
            time: 2.5,
            patch_size: patch,
            choppiness: 0.8,
        };
        let a = dispatch_spectrum_evolve(&field.h0, &field.h0_neg, params);
        let b = dispatch_spectrum_evolve(&field.h0, &field.h0_neg, params);
        assert_eq!(a, b);
    }
}
