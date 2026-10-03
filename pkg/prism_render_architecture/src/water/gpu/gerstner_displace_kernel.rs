//! Analytic `Gerstner` surface-displacement compute kernel: the `WESL` shader
//! plus its bit-exact `CPU` twin.
//!
//! This is the near-field / interactive ocean tier that complements the
//! spectral inverse-`FFT` path. It superposes `wave_count` trochoidal
//! `Gerstner` wave trains into a per-texel world-space displacement field and
//! the closed-form analytic surface normal (GPU Gems, "Effective Water
//! Simulation from Physical Models"). For a wave with unit direction `D`, wave
//! number `k = 2*PI / L`, angular frequency `omega = speed * sqrt(g*k)`,
//! steepness `Q` in `[0, 1]`, crest amplitude `A`, and phase offset, at rest
//! position `p = (x, z)` and `theta = k (D.p) - omega t + phase`:
//!   `horizontal += Q * A * D * cos(theta)`   (trochoidal crest bunching)
//!   `vertical   += A * sin(theta)`
//! and the analytic normal accumulates from the flat upright surface
//! `(0, 1, 0)` with `WA = k * A`: `n.xz -= D * WA * cos(theta)`,
//! `n.y -= Q * WA * sin(theta)`, normalized (falling back to `(0, 1, 0)` for a
//! degenerate accumulator).
//!
//! [`WATER_GERSTNER_DISPLACE_WESL`] is the shader (entry point
//! `water_gerstner_displace`); [`dispatch_gerstner_displace`] is its bit-exact
//! `CPU` twin. Because the sandbox has no `GPU`, the twin is the correctness
//! proof: it consumes the identical buffer `ABI`
//! ([`WaterKernel::GerstnerDisplace`](super::super::kernels::WaterKernel) — one
//! storage buffer of wave trains, one uniform param block, two `rgba32float`
//! storage-texture outputs, 8x8 tile, `Grid2d` domain) and reproduces the
//! shader texel-for-texel. The trigonometric phasor goes through the crate's
//! hand-rolled [`sin_approx`](super::super::sin_approx) /
//! [`cos_approx`](super::super::cos_approx) (the determinism policy forbids
//! [`f32::sin`]/[`f32::cos`]), and the per-wave accumulation mirrors the
//! shader's float order lane-for-lane, so the twin's output is bit-exact with
//! an independent inline recomputation by construction. Pure classical
//! numerics — no AI/ML.

use alloc::vec;
use alloc::vec::Vec;

use super::super::{cos_approx, sin_approx, Vec3, EPS_LEN_SQ, GRAVITY, TWO_PI};

/// `WESL` source of the analytic `Gerstner` displacement compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_GERSTNER_DISPLACE_WESL: &str = include_str!("water_gerstner_displace.wesl");

/// Number of `f32` lanes in one packed `GerstnerWave`
/// (`dir_x, dir_z, amplitude, wavelength, steepness, speed, phase, pad`).
pub const GERSTNER_WAVE_FLOATS: usize = 8;

/// Number of `f32` lanes per output texel (`xyz` + one padding lane), matching
/// the shader's `rgba32float` stores.
pub const GERSTNER_OUT_FLOATS: usize = 4;

/// Per-dispatch `Gerstner` scalars, mirroring the shader `GerstnerParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GerstnerParams {
    /// Grid resolution `N`; output textures are `N*N` texels.
    pub grid_size: u32,
    /// Number of active `GerstnerWave` entries to sum.
    pub wave_count: u32,
    /// Physical patch size `L` (m) mapping texel `(px, py)` to world xz.
    pub patch_size: f32,
    /// Simulation time `t` (s) advancing every wave's phase.
    pub time: f32,
    /// Rest water level (world y, m) added to the summed vertical displacement.
    pub base_level: f32,
}

/// Output of one `Gerstner` dispatch: the per-texel world-space displacement
/// field and the matching analytic normals, each `grid_size * grid_size`
/// texels of [`GERSTNER_OUT_FLOATS`] lanes in row-major `(y, x)` order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GerstnerFields {
    /// Displacement texels `(disp.x, base_level + disp.y, disp.z, 0)`.
    pub displacement: Vec<f32>,
    /// Analytic normal texels `(n.x, n.y, n.z, 0)`.
    pub normal: Vec<f32>,
}

/// Deep-water dispersion `omega(k) = sqrt(g*k)`; non-positive `k` yields `0`.
/// Mirrors the shader's `water_dispersion`.
#[inline]
fn dispersion(k: f32) -> f32 {
    if k <= 0.0 {
        return 0.0;
    }
    (GRAVITY * k).sqrt()
}

/// Bit-exact `CPU` twin of the `water_gerstner_displace` shader.
///
/// `waves` is the packed storage buffer ([`GERSTNER_WAVE_FLOATS`] lanes per
/// wave). Only the first `params.wave_count` waves are read, and a wave whose
/// lanes run past the end of `waves` is skipped (the shader's `array` bound
/// guard), so a short buffer degrades gracefully instead of panicking. Returns
/// the displacement and normal fields; an empty grid (`grid_size == 0`) returns
/// empty fields.
#[must_use]
pub fn dispatch_gerstner_displace(waves: &[f32], params: GerstnerParams) -> GerstnerFields {
    let n = params.grid_size as usize;
    let texels = n * n;
    let mut displacement = vec![0.0f32; texels * GERSTNER_OUT_FLOATS];
    let mut normal = vec![0.0f32; texels * GERSTNER_OUT_FLOATS];
    if n == 0 {
        return GerstnerFields {
            displacement,
            normal,
        };
    }

    let inv_n = 1.0 / (n as f32).max(1.0);
    let cell = params.patch_size * inv_n;
    let count = params.wave_count as usize;

    for gy in 0..n {
        let rest_z = gy as f32 * cell;
        for gx in 0..n {
            let rest_x = gx as f32 * cell;

            let mut disp = Vec3::new(0.0, 0.0, 0.0);
            let mut nrm = Vec3::new(0.0, 1.0, 0.0);

            for i in 0..count {
                let base = i * GERSTNER_WAVE_FLOATS;
                if base + GERSTNER_WAVE_FLOATS > waves.len() {
                    break;
                }
                let dir_x = waves[base];
                let dir_z = waves[base + 1];
                let amplitude = waves[base + 2];
                let wavelength = waves[base + 3];
                let steepness = waves[base + 4];
                let speed = waves[base + 5];
                let phase = waves[base + 6];
                // waves[base + 7] is the padding lane; ignored.

                if wavelength <= 0.0 {
                    continue;
                }
                let dir_len_sq = dir_x * dir_x + dir_z * dir_z;
                if dir_len_sq <= EPS_LEN_SQ {
                    continue;
                }
                let inv_dir_len = 1.0 / dir_len_sq.sqrt();
                let dx = dir_x * inv_dir_len;
                let dz = dir_z * inv_dir_len;

                let k = TWO_PI / wavelength;
                let omega = speed * dispersion(k);
                let theta = k * (dx * rest_x + dz * rest_z) - omega * params.time + phase;
                let c = cos_approx(theta);
                let s = sin_approx(theta);

                let qa = steepness * amplitude;
                disp.x += qa * dx * c;
                disp.z += qa * dz * c;
                disp.y += amplitude * s;

                let wa = k * amplitude;
                nrm.x -= dx * wa * c;
                nrm.z -= dz * wa * c;
                nrm.y -= steepness * wa * s;
            }

            let normal_v = normalize_or_up(nrm);

            let idx = (gy * n + gx) * GERSTNER_OUT_FLOATS;
            displacement[idx] = disp.x;
            displacement[idx + 1] = params.base_level + disp.y;
            displacement[idx + 2] = disp.z;
            displacement[idx + 3] = 0.0;
            normal[idx] = normal_v.x;
            normal[idx + 1] = normal_v.y;
            normal[idx + 2] = normal_v.z;
            normal[idx + 3] = 0.0;
        }
    }

    GerstnerFields {
        displacement,
        normal,
    }
}

/// Normalizes the accumulated normal, falling back to the upright `(0, 1, 0)`
/// surface normal for a degenerate accumulator. Mirrors the shader's
/// `if dot(nrm, nrm) > EPS { normalize(nrm) } else { (0, 1, 0) }`; the fallback
/// is `(0, 1, 0)` rather than the zero vector of
/// [`Vec3::normalize_or_zero`](super::super::Vec3::normalize_or_zero), so the
/// division order (`v * (1 / sqrt(dot))`) matches the shader lane-for-lane.
#[inline]
fn normalize_or_up(v: Vec3) -> Vec3 {
    let len_sq = v.dot(v);
    if len_sq > EPS_LEN_SQ {
        v.scale(1.0 / len_sq.sqrt())
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a packed wave buffer from `(dir_x, dir_z, amplitude, wavelength,
    /// steepness, speed, phase)` tuples (the padding lane is appended as `0`).
    fn pack(waves: &[[f32; 7]]) -> Vec<f32> {
        let mut out = Vec::with_capacity(waves.len() * GERSTNER_WAVE_FLOATS);
        for w in waves {
            out.extend_from_slice(w);
            out.push(0.0);
        }
        out
    }

    /// The shipped `WESL` honors the `GerstnerDisplace` descriptor's `ABI`:
    /// entry-point name, 8x8 tile, the storage wave buffer, the uniform block,
    /// and the two `rgba32float` storage-texture outputs.
    #[test]
    fn wesl_matches_descriptor_abi() {
        let src = WATER_GERSTNER_DISPLACE_WESL;
        assert!(src.contains("fn water_gerstner_displace("));
        assert!(src.contains("@workgroup_size(8, 8, 1)"));
        assert!(src.contains("var<storage, read> waves: array<GerstnerWave>"));
        assert!(src.contains("var<uniform> params: GerstnerParams"));
        assert_eq!(
            src.matches("texture_storage_2d<rgba32float, write>")
                .count(),
            2
        );

        use super::super::super::kernels::{DispatchDomain, WaterKernel};
        let desc = WaterKernel::GerstnerDisplace.descriptor();
        assert_eq!(desc.layout.storage_buffers, 1);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 2);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(desc.workgroup.x, 8);
        assert_eq!(desc.workgroup.y, 8);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Grid2d);
        assert_eq!(
            WaterKernel::GerstnerDisplace.wesl_entry_point(),
            "water_gerstner_displace"
        );
    }

    /// With no waves the surface is flat: every displacement texel is
    /// `(0, base_level, 0)` and every normal is the upright `(0, 1, 0)`.
    #[test]
    fn zero_waves_is_flat_rest_plane() {
        let params = GerstnerParams {
            grid_size: 4,
            wave_count: 0,
            patch_size: 10.0,
            time: 1.5,
            base_level: 2.0,
        };
        let fields = dispatch_gerstner_displace(&[], params);
        let texels = 4 * 4;
        assert_eq!(fields.displacement.len(), texels * GERSTNER_OUT_FLOATS);
        assert_eq!(fields.normal.len(), texels * GERSTNER_OUT_FLOATS);
        for t in 0..texels {
            let i = t * GERSTNER_OUT_FLOATS;
            assert_eq!(fields.displacement[i].to_bits(), 0.0f32.to_bits());
            assert_eq!(fields.displacement[i + 1].to_bits(), 2.0f32.to_bits());
            assert_eq!(fields.displacement[i + 2].to_bits(), 0.0f32.to_bits());
            assert_eq!(fields.normal[i].to_bits(), 0.0f32.to_bits());
            assert_eq!(fields.normal[i + 1].to_bits(), 1.0f32.to_bits());
            assert_eq!(fields.normal[i + 2].to_bits(), 0.0f32.to_bits());
        }
    }

    /// A single wave sampled at the origin texel with `time == 0` and
    /// `phase == 0` gives `theta == 0`, so `cos == 1`, `sin == 0`: the
    /// horizontal displacement is `Q*A*D`, the vertical is `base_level`, and the
    /// normal tilts purely in the `xz` propagation direction.
    #[test]
    fn single_wave_zero_theta_known_value() {
        let dir_x = 1.0f32;
        let dir_z = 0.0f32;
        let amplitude = 0.5f32;
        let wavelength = 8.0f32;
        let steepness = 0.4f32;
        let waves = pack(&[[dir_x, dir_z, amplitude, wavelength, steepness, 1.0, 0.0]]);
        let params = GerstnerParams {
            grid_size: 2,
            wave_count: 1,
            patch_size: 4.0,
            time: 0.0,
            base_level: 3.0,
        };
        let fields = dispatch_gerstner_displace(&waves, params);
        // Origin texel (gx == gy == 0): rest position is (0, 0), so theta == 0.
        // The hand-rolled trig is used throughout, so `cos_approx(0)` is ~1 (not
        // exactly 1) and `sin_approx(0)` is exactly 0; the expectation uses the
        // same approximations rather than the mathematical limits.
        let c0 = cos_approx(0.0);
        let s0 = sin_approx(0.0);
        let qa = steepness * amplitude;
        assert_eq!(
            fields.displacement[0].to_bits(),
            (qa * dir_x * c0).to_bits()
        );
        assert_eq!(
            fields.displacement[1].to_bits(),
            (3.0 + amplitude * s0).to_bits()
        );
        assert_eq!(
            fields.displacement[2].to_bits(),
            (qa * dir_z * c0).to_bits()
        );
        // The normal accumulates from (0, 1, 0) exactly as the kernel does, so
        // signed zeros in the degenerate z lane match bit-for-bit.
        let k = TWO_PI / wavelength;
        let wa = k * amplitude;
        let inv_dir = 1.0 / (dir_x * dir_x + dir_z * dir_z).sqrt();
        let ux = dir_x * inv_dir;
        let uz = dir_z * inv_dir;
        let raw = Vec3::new(
            0.0 - ux * wa * c0,
            1.0 - steepness * wa * s0,
            0.0 - uz * wa * c0,
        );
        let expect = raw.scale(1.0 / raw.dot(raw).sqrt());
        assert_eq!(fields.normal[0].to_bits(), expect.x.to_bits());
        assert_eq!(fields.normal[1].to_bits(), expect.y.to_bits());
        assert_eq!(fields.normal[2].to_bits(), expect.z.to_bits());
    }

    /// Independent inline recomputation of the full field, lane-for-lane, binds
    /// the twin to the documented algorithm (not a tautology: the anti-vacuous
    /// check confirms the field is non-trivial, i.e. at least one texel differs
    /// from the flat rest plane).
    #[test]
    fn matches_independent_inline_recompute() {
        let waves = pack(&[
            [0.7, 0.3, 0.6, 9.0, 0.5, 1.0, 0.2],
            [-0.4, 0.9, 0.25, 5.0, 0.3, 1.2, 1.1],
            [0.1, -0.8, 0.4, 13.0, 0.6, 0.9, -0.7],
        ]);
        let params = GerstnerParams {
            grid_size: 6,
            wave_count: 3,
            patch_size: 24.0,
            time: 2.25,
            base_level: 1.0,
        };
        let fields = dispatch_gerstner_displace(&waves, params);

        let n = params.grid_size as usize;
        let inv_n = 1.0 / (n as f32).max(1.0);
        let cell = params.patch_size * inv_n;
        let mut changed = 0usize;
        for gy in 0..n {
            let rest_z = gy as f32 * cell;
            for gx in 0..n {
                let rest_x = gx as f32 * cell;
                let mut dxp = 0.0f32;
                let mut dyp = 0.0f32;
                let mut dzp = 0.0f32;
                let mut nx = 0.0f32;
                let mut ny = 1.0f32;
                let mut nz = 0.0f32;
                for w in waves.chunks_exact(GERSTNER_WAVE_FLOATS) {
                    let (ddx, ddz, amp, wl, steep, spd, ph) =
                        (w[0], w[1], w[2], w[3], w[4], w[5], w[6]);
                    if wl <= 0.0 {
                        continue;
                    }
                    let dls = ddx * ddx + ddz * ddz;
                    if dls <= EPS_LEN_SQ {
                        continue;
                    }
                    let inv = 1.0 / dls.sqrt();
                    let ux = ddx * inv;
                    let uz = ddz * inv;
                    let k = TWO_PI / wl;
                    let kk = if k <= 0.0 { 0.0 } else { (GRAVITY * k).sqrt() };
                    let omega = spd * kk;
                    let theta = k * (ux * rest_x + uz * rest_z) - omega * params.time + ph;
                    let c = cos_approx(theta);
                    let s = sin_approx(theta);
                    let qa = steep * amp;
                    dxp += qa * ux * c;
                    dzp += qa * uz * c;
                    dyp += amp * s;
                    let wa = k * amp;
                    nx -= ux * wa * c;
                    nz -= uz * wa * c;
                    ny -= steep * wa * s;
                }
                let len_sq = nx * nx + ny * ny + nz * nz;
                let (enx, eny, enz) = if len_sq > EPS_LEN_SQ {
                    let r = 1.0 / len_sq.sqrt();
                    (nx * r, ny * r, nz * r)
                } else {
                    (0.0, 1.0, 0.0)
                };
                let idx = (gy * n + gx) * GERSTNER_OUT_FLOATS;
                assert_eq!(fields.displacement[idx].to_bits(), dxp.to_bits());
                assert_eq!(
                    fields.displacement[idx + 1].to_bits(),
                    (params.base_level + dyp).to_bits()
                );
                assert_eq!(fields.displacement[idx + 2].to_bits(), dzp.to_bits());
                assert_eq!(fields.normal[idx].to_bits(), enx.to_bits());
                assert_eq!(fields.normal[idx + 1].to_bits(), eny.to_bits());
                assert_eq!(fields.normal[idx + 2].to_bits(), enz.to_bits());
                if dxp.to_bits() != 0.0f32.to_bits() || dyp.to_bits() != 0.0f32.to_bits() {
                    changed += 1;
                }
            }
        }
        assert!(changed > 0, "field must be non-trivial");
    }

    /// A non-unit stored direction is normalized in-kernel, so it produces the
    /// identical field to its unit-length counterpart.
    #[test]
    fn direction_is_normalized_in_kernel() {
        let base = [0.6f32, 0.8, 0.5, 7.0, 0.45, 1.1, 0.3];
        let scaled = [
            base[0] * 5.0,
            base[1] * 5.0,
            base[2],
            base[3],
            base[4],
            base[5],
            base[6],
        ];
        let params = GerstnerParams {
            grid_size: 5,
            wave_count: 1,
            patch_size: 15.0,
            time: 0.75,
            base_level: 0.0,
        };
        let unit = dispatch_gerstner_displace(&pack(&[base]), params);
        let non_unit = dispatch_gerstner_displace(&pack(&[scaled]), params);
        assert_eq!(unit, non_unit);
    }

    /// Degenerate waves — non-positive wavelength or a near-zero direction — are
    /// skipped, so a field built only from them equals the flat rest plane.
    #[test]
    fn degenerate_waves_are_skipped() {
        let waves = pack(&[
            [1.0, 0.0, 0.5, 0.0, 0.4, 1.0, 0.0],  // wavelength == 0 -> skip
            [1.0, 0.0, 0.5, -3.0, 0.4, 1.0, 0.0], // wavelength < 0 -> skip
            [0.0, 0.0, 0.5, 6.0, 0.4, 1.0, 0.0],  // zero direction -> skip
        ]);
        let params = GerstnerParams {
            grid_size: 3,
            wave_count: 3,
            patch_size: 9.0,
            time: 1.0,
            base_level: 0.5,
        };
        let fields = dispatch_gerstner_displace(&waves, params);
        for t in 0..9 {
            let i = t * GERSTNER_OUT_FLOATS;
            assert_eq!(fields.displacement[i].to_bits(), 0.0f32.to_bits());
            assert_eq!(fields.displacement[i + 1].to_bits(), 0.5f32.to_bits());
            assert_eq!(fields.displacement[i + 2].to_bits(), 0.0f32.to_bits());
            assert_eq!(fields.normal[i + 1].to_bits(), 1.0f32.to_bits());
        }
    }

    /// A wave whose lanes run past the end of the buffer is skipped rather than
    /// panicking (the shader's `array` bound guard), and `grid_size == 0`
    /// returns empty fields.
    #[test]
    fn short_buffer_and_empty_grid_do_not_panic() {
        // wave_count claims 2 waves but only one full wave is present.
        let mut waves = pack(&[[1.0, 0.0, 0.5, 8.0, 0.4, 1.0, 0.0]]);
        waves.truncate(GERSTNER_WAVE_FLOATS + 3); // partial second wave
        let params = GerstnerParams {
            grid_size: 2,
            wave_count: 2,
            patch_size: 4.0,
            time: 0.0,
            base_level: 0.0,
        };
        let fields = dispatch_gerstner_displace(&waves, params);
        assert_eq!(fields.displacement.len(), 4 * GERSTNER_OUT_FLOATS);

        let empty = dispatch_gerstner_displace(
            &waves,
            GerstnerParams {
                grid_size: 0,
                ..params
            },
        );
        assert!(empty.displacement.is_empty());
        assert!(empty.normal.is_empty());
    }

    /// The dispatch is deterministic: identical inputs produce bit-identical
    /// fields on repeated calls.
    #[test]
    fn deterministic_across_calls() {
        let waves = pack(&[
            [0.5, 0.5, 0.4, 10.0, 0.5, 1.0, 0.1],
            [-0.7, 0.2, 0.3, 6.0, 0.3, 1.1, 0.9],
        ]);
        let params = GerstnerParams {
            grid_size: 4,
            wave_count: 2,
            patch_size: 16.0,
            time: 3.5,
            base_level: 1.25,
        };
        let a = dispatch_gerstner_displace(&waves, params);
        let b = dispatch_gerstner_displace(&waves, params);
        assert_eq!(a, b);
    }
}
