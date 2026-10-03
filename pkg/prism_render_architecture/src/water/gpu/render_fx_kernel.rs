//! Water render-FX compute kernel: the `WESL` foam-advection shader plus its
//! bit-exact `CPU` twin.
//!
//! This is the first real water-subsystem compute shader to land. The sibling
//! `ray_scene` backend already ships hundreds of `WESL` kernels, each paired
//! with a `CPU` twin that proves it against the golden reference; this module
//! brings the same discipline to water, starting from the foam field.
//!
//! [`WATER_RENDER_FX_WESL`] is the shader (entry point `water_foam_advect`);
//! [`dispatch_foam_advect`] is its bit-exact `CPU` twin. Because the sandbox
//! has no `GPU`, the twin is the correctness proof: it consumes the identical
//! buffer `ABI` ([`WaterKernel::FoamAdvect`](super::super::kernels::WaterKernel)
//! — one storage drive buffer, one uniform param block, one sampled previous
//! texture, one storage next texture, dispatched over an 8x8 texel tile) and
//! reconstructs the shader arithmetic cell-for-cell, so the parity test can
//! diff it against the golden [`foam::step_foam`](super::super::foam::step_foam)
//! and the structural test keeps the `WESL` entry point and `ABI` strides in
//! sync with the `Rust` constants.
//!
//! The `WESL` numerics are a transcription of
//! [`foam::step_foam`](super::super::foam::step_foam): semi-Lagrangian
//! backtrace + manual bilinear resample of the previous foam field, flow-aware
//! exponential decay via the shared [`exp_approx`](super::super::exp_approx),
//! additive reactive sources, and a final clamp to `0..=1` coverage. Only
//! `+ - * /`, `sqrt`, and `exp_approx` appear, matching the workspace
//! determinism policy that forbids every float intrinsic but `sqrt`.

use alloc::vec;
use alloc::vec::Vec;

use super::super::foam::FoamConfig;
use super::super::wetness::WetnessParams;
use super::super::{exp_approx, EPS};

/// `WESL` source of the water render-FX compute kernels.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_RENDER_FX_WESL: &str = include_str!("water_render_fx.wesl");

/// Byte stride of one entry in the per-cell foam drive buffer.
///
/// Each cell is a `vec4<f32>` of `(u, v, source, pad)`: the surface flow
/// velocity, the reactive foam source already scaled by `dt`, and one padding
/// lane so the storage buffer honours the 16-byte `vec4` alignment the shader
/// reads. Mirrors the `@binding(2) drive: array<vec4<f32>>` declaration.
pub const FOAM_DRIVE_STRIDE: u32 = 16;

/// Number of `f32` lanes in one foam drive entry (`u`, `v`, `source`, `pad`).
pub const FOAM_DRIVE_FLOATS: usize = (FOAM_DRIVE_STRIDE as usize) / size_of::<f32>();

/// Flow-aware decay rate (per second), reconstructing the shader's
/// `foam_decay_rate` from the drive/param `ABI`.
///
/// Interpolates from `base_decay * persistence_floor` in still water up to the
/// full `base_decay` at or beyond `reference_speed`, guarding the division by
/// [`EPS`] exactly as the shader does. Kept independent of
/// [`foam::foam_decay_rate`](super::super::foam::foam_decay_rate) so the parity
/// test proves the transcription rather than asserting a tautology.
fn foam_decay_rate_twin(flow_speed: f32, cfg: FoamConfig) -> f32 {
    let floor = cfg.persistence_floor.clamp(0.0, 1.0);
    let reference = if cfg.reference_speed <= EPS {
        EPS
    } else {
        cfg.reference_speed
    };
    let t = (flow_speed / reference).clamp(0.0, 1.0);
    cfg.base_decay * (floor + (1.0 - floor) * t)
}

/// Bilinearly resamples the previous foam field at fractional cell
/// coordinates, reconstructing the shader's hand-written `textureLoad`
/// bilinear tap-for-tap: coordinates clamp into `[0, nx-1] x [0, nz-1]`, the
/// four corner taps are read with the same edge clamp, and the convex
/// combination keeps a non-negative field non-negative.
fn sample_bilinear_twin(prev: &[f32], nx: usize, nz: usize, px: f32, pz: f32) -> f32 {
    let max_x = (nx - 1) as f32;
    let max_z = (nz - 1) as f32;
    let cx = px.clamp(0.0, max_x);
    let cz = pz.clamp(0.0, max_z);
    let x0 = cx as usize;
    let z0 = cz as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let z1 = (z0 + 1).min(nz - 1);
    let fx = cx - (x0 as f32);
    let fz = cz - (z0 as f32);
    let s00 = prev[z0 * nx + x0];
    let s10 = prev[z0 * nx + x1];
    let s01 = prev[z1 * nx + x0];
    let s11 = prev[z1 * nx + x1];
    let top = s00 + (s10 - s00) * fx;
    let bottom = s01 + (s11 - s01) * fx;
    top + (bottom - top) * fz
}

/// Runs the whole `water_foam_advect` dispatch on the `CPU`: one invocation per
/// grid cell, exactly as the shader maps one `global_invocation_id` to one
/// cell.
///
/// `prev` is the previous foam coverage field, one `f32` per cell in row-major
/// `nx * nz` order (the sampled `foam_prev` texture). `drive` is the flat
/// `array<vec4<f32>>` of per-cell `(u, v, source, pad)` entries,
/// [`FOAM_DRIVE_FLOATS`] lanes each (the `drive` storage buffer). The result is
/// the next foam coverage field, matching the `foam_next` storage texture the
/// shader writes. Short buffers yield a zero field rather than panicking,
/// mirroring the shader's bounds-guarded early-out.
#[must_use]
pub fn dispatch_foam_advect(prev: &[f32], drive: &[f32], cfg: FoamConfig, dt: f32) -> Vec<f32> {
    let nx = cfg.nx as usize;
    let nz = cfg.nz as usize;
    let n = nx.saturating_mul(nz);
    let mut out = vec![0.0_f32; n];
    if n == 0 || prev.len() < n || drive.len() < n * FOAM_DRIVE_FLOATS {
        return out;
    }
    let inv_dx = 1.0 / cfg.dx;
    let mut z = 0;
    while z < nz {
        let mut x = 0;
        while x < nx {
            let idx = z * nx + x;
            let base = idx * FOAM_DRIVE_FLOATS;
            let u = drive[base];
            let v = drive[base + 1];
            let source = drive[base + 2].max(0.0);

            // Semi-Lagrangian backtrace, then resample the previous field.
            let px = (x as f32) - dt * u * inv_dx;
            let pz = (z as f32) - dt * v * inv_dx;
            let advected = sample_bilinear_twin(prev, nx, nz, px, pz);

            // Flow-aware exponential decay, additive sources, unit clamp.
            let flow_speed = (u * u + v * v).sqrt();
            let rate = foam_decay_rate_twin(flow_speed, cfg);
            let decayed = advected.max(0.0) * exp_approx(-rate * dt);
            out[idx] = (decayed + source).clamp(0.0, 1.0);

            x += 1;
        }
        z += 1;
    }
    out
}

/// Byte stride of one entry in the per-cell wetness drive buffer.
///
/// Each cell occupies two `vec4<f32>` lanes (32 bytes): `(wetness, puddle,
/// contact, rain)` then `(drain, pad, pad, pad)`. `contact` is a boolean
/// encoded as a float, which the shader treats as submerged when `> 0.5`.
/// Mirrors the `@binding(0) wet_drive: array<vec4<f32>>` declaration, read two
/// lanes at a time.
pub const WETNESS_DRIVE_STRIDE: u32 = 32;

/// Number of `f32` lanes in one wetness drive entry (two `vec4<f32>`).
pub const WETNESS_DRIVE_FLOATS: usize = (WETNESS_DRIVE_STRIDE as usize) / size_of::<f32>();

/// Number of `f32` lanes the twin writes per cell: `(wetness, puddle)`,
/// matching the `rg32float` texel the shader stores.
pub const WETNESS_OUT_FLOATS: usize = 2;

/// Absorbs moisture over `dt`, reconstructing the shader's `wet_absorb`
/// tap-for-tap (golden [`wetness::absorb`](super::super::wetness::absorb)): the
/// remaining dry fraction decays exponentially via the shared
/// [`exp_approx`](super::super::exp_approx), result clamped to `0..=1`. Kept
/// independent of the golden so the parity test proves the transcription rather
/// than asserting a tautology.
fn wet_absorb_twin(wetness: f32, rate: f32, dt: f32) -> f32 {
    let w = wetness.clamp(0.0, 1.0);
    let dry_fraction = (1.0 - w) * exp_approx(-rate.max(0.0) * dt.max(0.0));
    (1.0 - dry_fraction).clamp(0.0, 1.0)
}

/// Dries moisture over `dt`, reconstructing the shader's `wet_dry` (golden
/// [`wetness::dry`](super::super::wetness::dry)): `w * exp(-rate * dt)` via the
/// shared [`exp_approx`](super::super::exp_approx).
fn wet_dry_twin(wetness: f32, rate: f32, dt: f32) -> f32 {
    let w = wetness.clamp(0.0, 1.0);
    w * exp_approx(-rate.max(0.0) * dt.max(0.0))
}

/// One wetness-film step, reconstructing the shader's `update_wetness` (golden
/// [`wetness::update_wetness`](super::super::wetness::update_wetness)): direct
/// contact soaks at the full absorb rate, otherwise rain soaks at a rain-scaled
/// rate and a rain-free exposed surface dries. The rain-vs-dry branch keys on
/// [`EPS`] exactly as the shader keys on its `wet_params.eps` uniform.
fn update_wetness_twin(
    current: f32,
    params: WetnessParams,
    water_contact: bool,
    rain_rate: f32,
    dt: f32,
) -> f32 {
    if water_contact {
        return wet_absorb_twin(current, params.absorb_rate, dt);
    }
    let rain = rain_rate.max(0.0);
    if rain > EPS {
        wet_absorb_twin(current, params.absorb_rate * rain.min(1.0), dt)
    } else {
        wet_dry_twin(current, params.dry_rate, dt)
    }
}

/// One puddle-depth step, reconstructing the shader's `step_puddle` (golden
/// [`wetness::puddle_depth`](super::super::wetness::puddle_depth)): rain fills,
/// drainage empties, clamped non-negative.
fn step_puddle_twin(accumulated: f32, rain_rate: f32, drain_rate: f32, dt: f32) -> f32 {
    let base = accumulated.max(0.0);
    (base + (rain_rate.max(0.0) - drain_rate.max(0.0)) * dt.max(0.0)).max(0.0)
}

/// Runs the whole `water_wetness_step` dispatch on the `CPU`: one invocation
/// per surface cell, exactly as the shader maps one `global_invocation_id` to
/// one cell.
///
/// `drive` is the flat `array<vec4<f32>>` of per-cell entries,
/// [`WETNESS_DRIVE_FLOATS`] lanes each: `(wetness, puddle, contact, rain)` then
/// `(drain, pad, pad, pad)`, where `contact > 0.5` marks a submerged cell.
/// `nx`/`nz` are the grid dimensions; unlike the foam config the shader's
/// `wet_params` carries them in its own uniform lanes, so the twin takes them
/// explicitly. The result is the interleaved `(wetness, puddle)` pair per cell
/// in row-major order, [`WETNESS_OUT_FLOATS`] lanes each. A drive buffer shorter
/// than the grid is handled without panicking, mirroring the shader's bounds
/// guard that returns early.
#[must_use]
pub fn dispatch_wetness_step(
    drive: &[f32],
    params: WetnessParams,
    nx: usize,
    nz: usize,
    dt: f32,
) -> Vec<f32> {
    let n = nx * nz;
    let mut out = vec![0.0_f32; n * WETNESS_OUT_FLOATS];
    if n == 0 || drive.len() < n * WETNESS_DRIVE_FLOATS {
        return out;
    }
    let mut z = 0;
    while z < nz {
        let mut x = 0;
        while x < nx {
            let idx = z * nx + x;
            let base = idx * WETNESS_DRIVE_FLOATS;
            let wetness_prev = drive[base];
            let puddle_prev = drive[base + 1];
            let water_contact = drive[base + 2] > 0.5;
            let rain = drive[base + 3];
            let drain = drive[base + 4];

            let wetness = update_wetness_twin(wetness_prev, params, water_contact, rain, dt);
            let puddle = step_puddle_twin(puddle_prev, rain, drain, dt);

            out[idx * WETNESS_OUT_FLOATS] = wetness;
            out[idx * WETNESS_OUT_FLOATS + 1] = puddle;

            x += 1;
        }
        z += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::foam::step_foam;
    use super::super::super::kernels::WaterKernel;
    use super::super::super::wetness::{step_moisture, SurfaceMoisture};
    use super::*;

    const CFG: FoamConfig = FoamConfig {
        nx: 8,
        nz: 8,
        dx: 0.5,
        base_decay: 1.0,
        persistence_floor: 0.1,
        reference_speed: 2.0,
    };

    /// Deterministic pseudo-random field in a bounded range, no `std`/`rand`.
    fn fill(seed: u32, lo: f32, hi: f32, len: usize) -> Vec<f32> {
        let mut state = seed | 1;
        let mut out = vec![0.0_f32; len];
        let mut i = 0;
        while i < len {
            // xorshift32, mapped into [lo, hi).
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = (state as f32) / (u32::MAX as f32);
            out[i] = lo + (hi - lo) * unit;
            i += 1;
        }
        out
    }

    /// Packs separate `u`, `v`, `source` fields into the flat drive `ABI`.
    fn pack_drive(u: &[f32], v: &[f32], sources: &[f32]) -> Vec<f32> {
        let n = u.len();
        let mut drive = vec![0.0_f32; n * FOAM_DRIVE_FLOATS];
        let mut i = 0;
        while i < n {
            let base = i * FOAM_DRIVE_FLOATS;
            drive[base] = u[i];
            drive[base + 1] = v[i];
            drive[base + 2] = sources[i];
            drive[base + 3] = 0.0;
            i += 1;
        }
        drive
    }

    #[test]
    fn twin_matches_cpu_golden_bit_for_bit() {
        let n = CFG.cell_count();
        // Several independent random fields exercise advection, decay, and
        // sources together across the whole grid.
        for seed in [0x1234_5678_u32, 0x9e37_79b9, 0x0514_2024, 0xdead_beef] {
            let prev = fill(seed, 0.0, 1.0, n);
            let u = fill(seed ^ 0x00f0_0f00, -3.0, 3.0, n);
            let v = fill(seed ^ 0x0f00_00f0, -3.0, 3.0, n);
            let sources = fill(seed ^ 0x0000_ffff, 0.0, 0.4, n);
            let drive = pack_drive(&u, &v, &sources);

            let twin = dispatch_foam_advect(&prev, &drive, CFG, 0.05);
            let golden = step_foam(&prev, &u, &v, &sources, CFG, 0.05);
            assert_eq!(twin, golden, "twin diverged from golden for seed {seed:#x}");
        }
    }

    #[test]
    fn twin_keeps_coverage_in_unit_range() {
        let n = CFG.cell_count();
        let prev = fill(0x55aa_55aa, 0.0, 1.0, n);
        let u = fill(0x1111_2222, -5.0, 5.0, n);
        let v = fill(0x3333_4444, -5.0, 5.0, n);
        let sources = fill(0x5555_6666, 0.0, 2.0, n);
        let drive = pack_drive(&u, &v, &sources);
        let out = dispatch_foam_advect(&prev, &drive, CFG, 0.1);
        for &c in &out {
            assert!((0.0..=1.0).contains(&c), "coverage out of range: {c}");
        }
    }

    #[test]
    fn short_buffers_do_not_panic() {
        // Fewer cells than the grid declares, and a truncated drive buffer.
        let out = dispatch_foam_advect(&[0.5; 4], &[0.0; 4], CFG, 0.05);
        assert_eq!(out.len(), CFG.cell_count());
        assert!(out.iter().all(|&c| c.abs() < EPS));
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let s = WATER_RENDER_FX_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains(&format!(
            "fn {}",
            WaterKernel::FoamAdvect.wesl_entry_point()
        )));
        assert!(s.contains("@workgroup_size(8, 8, 1)"));
        assert!(s.contains("foam_prev"));
        assert!(s.contains("foam_next"));
        assert!(s.contains("drive"));
        assert!(s.contains("params"));
        // The shader's exp is the shared non-negative approximation.
        assert!(s.contains("fn exp_approx"));
    }

    #[test]
    fn drive_abi_strides_are_consistent() {
        assert_eq!(FOAM_DRIVE_STRIDE, 16);
        assert_eq!(FOAM_DRIVE_FLOATS, 4);
        assert_eq!(
            FOAM_DRIVE_STRIDE as usize,
            FOAM_DRIVE_FLOATS * size_of::<f32>()
        );
    }

    const WET_PARAMS: WetnessParams = WetnessParams {
        max_capillary_height: 0.5,
        absorb_rate: 2.0,
        dry_rate: 0.5,
        darkening_strength: 0.4,
        puddle_threshold: 0.02,
    };

    /// Packs separate per-cell fields into the flat wetness drive `ABI`.
    fn pack_wet_drive(
        wet: &[f32],
        pud: &[f32],
        contact: &[f32],
        rain: &[f32],
        drain: &[f32],
    ) -> Vec<f32> {
        let n = wet.len();
        let mut drive = vec![0.0_f32; n * WETNESS_DRIVE_FLOATS];
        let mut i = 0;
        while i < n {
            let base = i * WETNESS_DRIVE_FLOATS;
            drive[base] = wet[i];
            drive[base + 1] = pud[i];
            drive[base + 2] = contact[i];
            drive[base + 3] = rain[i];
            drive[base + 4] = drain[i];
            // Lanes 5..8 stay padding zeros, matching the shader's second vec4.
            i += 1;
        }
        drive
    }

    #[test]
    fn wetness_twin_matches_cpu_golden_bit_for_bit() {
        let nx = 8usize;
        let nz = 8usize;
        let n = nx * nz;
        for seed in [0x1357_9bdf_u32, 0x2468_ace0, 0xcafe_babe, 0x0bad_f00d] {
            let wet = fill(seed, 0.0, 1.0, n);
            let pud = fill(seed ^ 0x00ff_00ff, 0.0, 0.1, n);
            // Spread across 0..1 so roughly half the cells read as submerged.
            let contact = fill(seed ^ 0x1234_0000, 0.0, 1.0, n);
            // Include negative rain to exercise the dry branch and the clamp.
            let rain = fill(seed ^ 0x0000_abcd, -0.2, 1.2, n);
            let drain = fill(seed ^ 0x9999_1111, 0.0, 0.3, n);
            let drive = pack_wet_drive(&wet, &pud, &contact, &rain, &drain);
            let twin = dispatch_wetness_step(&drive, WET_PARAMS, nx, nz, 0.05);
            let mut i = 0;
            while i < n {
                let water_contact = contact[i] > 0.5;
                let golden = step_moisture(
                    SurfaceMoisture {
                        wetness: wet[i],
                        puddle_depth: pud[i],
                    },
                    WET_PARAMS,
                    water_contact,
                    rain[i],
                    drain[i],
                    0.05,
                );
                assert_eq!(
                    twin[i * WETNESS_OUT_FLOATS],
                    golden.wetness,
                    "wetness diverged cell {i} seed {seed:#x}"
                );
                assert_eq!(
                    twin[i * WETNESS_OUT_FLOATS + 1],
                    golden.puddle_depth,
                    "puddle diverged cell {i} seed {seed:#x}"
                );
                i += 1;
            }
        }
    }

    #[test]
    fn wetness_twin_keeps_outputs_valid() {
        let nx = 8usize;
        let nz = 8usize;
        let n = nx * nz;
        // Deliberately out-of-range inputs: the twin must still clamp wetness to
        // `0..=1` and keep puddle depth non-negative.
        let wet = fill(0x55aa_1234, -0.5, 1.5, n);
        let pud = fill(0x1234_55aa, -0.1, 0.2, n);
        let contact = fill(0x0f0f_f0f0, 0.0, 1.0, n);
        let rain = fill(0xfeed_0001, -0.5, 2.0, n);
        let drain = fill(0x0001_feed, 0.0, 1.0, n);
        let drive = pack_wet_drive(&wet, &pud, &contact, &rain, &drain);
        let out = dispatch_wetness_step(&drive, WET_PARAMS, nx, nz, 0.1);
        let mut i = 0;
        while i < n {
            let w = out[i * WETNESS_OUT_FLOATS];
            let p = out[i * WETNESS_OUT_FLOATS + 1];
            assert!((0.0..=1.0).contains(&w), "wetness out of range: {w}");
            assert!(p >= 0.0, "puddle depth negative: {p}");
            i += 1;
        }
    }

    #[test]
    fn wetness_short_buffers_do_not_panic() {
        // One cell's worth of drive for an 8x8 grid: everything past it is
        // skipped, and the output stays the full zero-filled grid.
        let out = dispatch_wetness_step(&[0.1; 8], WET_PARAMS, 8, 8, 0.05);
        assert_eq!(out.len(), 8 * 8 * WETNESS_OUT_FLOATS);
        assert!(out.iter().all(|&v| v.abs() < EPS));
    }

    #[test]
    fn wesl_wetness_kernel_declares_expected_abi() {
        let s = WATER_RENDER_FX_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains(&format!(
            "fn {}",
            WaterKernel::WetnessStep.wesl_entry_point()
        )));
        assert!(s.contains("@workgroup_size(8, 8, 1)"));
        assert!(s.contains("moisture_next"));
        assert!(s.contains("wet_drive"));
        assert!(s.contains("wet_params"));
    }

    #[test]
    fn wetness_drive_abi_strides_are_consistent() {
        assert_eq!(WETNESS_DRIVE_STRIDE, 32);
        assert_eq!(WETNESS_DRIVE_FLOATS, 8);
        assert_eq!(WETNESS_OUT_FLOATS, 2);
        assert_eq!(
            WETNESS_DRIVE_STRIDE as usize,
            WETNESS_DRIVE_FLOATS * size_of::<f32>()
        );
    }
}
