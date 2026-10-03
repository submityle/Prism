//! Shallow-water step compute kernel: the `WESL` `water_swe_step` shader plus
//! its bit-exact `CPU` twin.
//!
//! This brings the surface shallow-water solver onto the `GPU` with the same
//! discipline the `ray_scene` backend established and the sibling
//! [`render_fx_kernel`](super::render_fx_kernel) started for water: a real
//! shader paired with a `CPU` twin that reproduces it cell-for-cell, so the
//! parity test can diff the twin against the golden [`swe::step`](super::super::swe::step).
//!
//! [`WATER_SWE_WESL`] is the shader (entry point `water_swe_step`);
//! [`dispatch_swe_step`] is its bit-exact `CPU` twin. The shader honours the
//! [`WaterKernel::SweStep`](super::super::kernels::WaterKernel) descriptor — two
//! storage buffers, one uniform block, one storage texture, one sampled texture,
//! dispatched over an 8x8 cell tile — via the classic shallow-water ping-pong:
//! water depth lives in a texture pair (sample the current frame, store the
//! next) while the velocity field lives in a storage-buffer pair.
//!
//! The numerics transcribe [`swe::step`](super::super::swe::step): conservative
//! flux-form continuity with reflective (no-flux) walls, a central pressure
//! gradient, first-order upwind self-advection, and linear velocity damping.
//! Only `+ - * /` appear, matching the workspace determinism policy.

use alloc::vec;
use alloc::vec::Vec;

use super::super::swe::SweConfig;

/// `WESL` source of the shallow-water step compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_SWE_WESL: &str = include_str!("water_swe.wesl");

/// Byte stride of one entry in the per-cell velocity storage buffer.
///
/// Each cell is a `vec4<f32>` of `(u, v, pad, pad)`: the depth-averaged flow
/// velocity plus two padding lanes so the storage buffer honours the 16-byte
/// `vec4` alignment the shader reads. Mirrors the `vel_in`/`vel_out`
/// `array<vec4<f32>>` declarations.
pub const SWE_VEL_STRIDE: u32 = 16;

/// Number of `f32` lanes in one velocity entry (`u`, `v`, `pad`, `pad`).
pub const SWE_VEL_FLOATS: usize = (SWE_VEL_STRIDE as usize) / size_of::<f32>();

/// Result of one [`dispatch_swe_step`] dispatch.
///
/// `height` is the next water depth, one `f32` per cell in row-major order (the
/// `r32float` `height_next` texture the shader stores). `velocity` is the next
/// depth-averaged velocity, [`SWE_VEL_FLOATS`] lanes per cell matching the
/// `vel_out` storage buffer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SweStepOutput {
    /// Next water depth per cell (one `f32` each).
    pub height: Vec<f32>,
    /// Next velocity per cell (`u`, `v`, `pad`, `pad`).
    pub velocity: Vec<f32>,
}

/// First-order upwind advective derivative `u * dF/dx + v * dF/dz`, taking the
/// gradient from the upwind neighbour. Reconstructs the shader's `upwind`
/// branch-for-branch (golden [`swe::upwind`](super::super::swe)); kept
/// independent of the golden so the parity test proves the transcription.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit stencil neighbours keep the twin allocation-free, mirroring the shader"
)]
fn upwind_twin(
    flow_u: f32,
    flow_v: f32,
    center: f32,
    xm: f32,
    xp: f32,
    zm: f32,
    zp: f32,
    inv_dx: f32,
) -> f32 {
    let ddx = if flow_u > 0.0 {
        (center - xm) * inv_dx
    } else {
        (xp - center) * inv_dx
    };
    let ddz = if flow_v > 0.0 {
        (center - zm) * inv_dx
    } else {
        (zp - center) * inv_dx
    };
    flow_u * ddx + flow_v * ddz
}

/// Runs the whole `water_swe_step` dispatch on the `CPU`: one invocation per
/// grid cell, exactly as the shader maps one `global_invocation_id` to one cell.
///
/// `height_curr` is the current water depth, one `f32` per cell in row-major
/// `nx * nz` order (the sampled `height_curr` texture). `vel_in` is the flat
/// `array<vec4<f32>>` of per-cell `(u, v, pad, pad)` entries, [`SWE_VEL_FLOATS`]
/// lanes each (the `vel_in` storage buffer). The result pairs the next depth
/// field with the next velocity buffer, matching the shader's `height_next`
/// texture and `vel_out` buffer. A buffer shorter than the grid is handled
/// without panicking, mirroring the shader's bounds guard and the golden
/// [`swe::step`](super::super::swe::step) short-circuit.
#[must_use]
pub fn dispatch_swe_step(
    height_curr: &[f32],
    vel_in: &[f32],
    cfg: SweConfig,
    dt: f32,
) -> SweStepOutput {
    let nx = cfg.nx as usize;
    let nz = cfg.nz as usize;
    let n = nx * nz;
    let mut height = vec![0.0_f32; n];
    let mut velocity = vec![0.0_f32; n * SWE_VEL_FLOATS];
    if n == 0 || height_curr.len() < n || vel_in.len() < n * SWE_VEL_FLOATS {
        // Degenerate drive: hand back a still, zeroed grid (never panic).
        return SweStepOutput { height, velocity };
    }

    let inv_dx = 1.0 / cfg.dx;
    let damp = (1.0 - cfg.damping * dt).max(0.0);

    // Depth and velocity accessors over the flat drive buffers.
    let h = |x: usize, z: usize| height_curr[z * nx + x];
    let u = |x: usize, z: usize| vel_in[(z * nx + x) * SWE_VEL_FLOATS];
    let v = |x: usize, z: usize| vel_in[(z * nx + x) * SWE_VEL_FLOATS + 1];

    let mut z = 0;
    while z < nz {
        let mut x = 0;
        while x < nx {
            let idx = z * nx + x;
            let h_c = h(x, z);
            let u_c = u(x, z);
            let v_c = v(x, z);
            let hu_c = h_c * u_c;
            let hv_c = h_c * v_c;

            // Conservative continuity: shared interior face flux cancels.
            let fx_right = if x + 1 < nx {
                0.5 * (hu_c + h(x + 1, z) * u(x + 1, z))
            } else {
                0.0
            };
            let fx_left = if x >= 1 {
                0.5 * (h(x - 1, z) * u(x - 1, z) + hu_c)
            } else {
                0.0
            };
            let fz_up = if z + 1 < nz {
                0.5 * (hv_c + h(x, z + 1) * v(x, z + 1))
            } else {
                0.0
            };
            let fz_down = if z >= 1 {
                0.5 * (h(x, z - 1) * v(x, z - 1) + hv_c)
            } else {
                0.0
            };
            let new_h = h_c - dt * inv_dx * ((fx_right - fx_left) + (fz_up - fz_down));

            // Reflective neighbour sampling for the momentum gradients.
            let h_xp = if x + 1 < nx { h(x + 1, z) } else { h_c };
            let h_xm = if x >= 1 { h(x - 1, z) } else { h_c };
            let h_zp = if z + 1 < nz { h(x, z + 1) } else { h_c };
            let h_zm = if z >= 1 { h(x, z - 1) } else { h_c };
            let dhdx = (h_xp - h_xm) * 0.5 * inv_dx;
            let dhdz = (h_zp - h_zm) * 0.5 * inv_dx;

            let u_xp = if x + 1 < nx { u(x + 1, z) } else { u_c };
            let u_xm = if x >= 1 { u(x - 1, z) } else { u_c };
            let u_zp = if z + 1 < nz { u(x, z + 1) } else { u_c };
            let u_zm = if z >= 1 { u(x, z - 1) } else { u_c };
            let v_xp = if x + 1 < nx { v(x + 1, z) } else { v_c };
            let v_xm = if x >= 1 { v(x - 1, z) } else { v_c };
            let v_zp = if z + 1 < nz { v(x, z + 1) } else { v_c };
            let v_zm = if z >= 1 { v(x, z - 1) } else { v_c };

            let adv_u = upwind_twin(u_c, v_c, u_c, u_xm, u_xp, u_zm, u_zp, inv_dx);
            let adv_v = upwind_twin(u_c, v_c, v_c, v_xm, v_xp, v_zm, v_zp, inv_dx);

            let new_u = (u_c - dt * (adv_u + cfg.gravity * dhdx)) * damp;
            let new_v = (v_c - dt * (adv_v + cfg.gravity * dhdz)) * damp;

            height[idx] = new_h;
            velocity[idx * SWE_VEL_FLOATS] = new_u;
            velocity[idx * SWE_VEL_FLOATS + 1] = new_v;

            x += 1;
        }
        z += 1;
    }

    SweStepOutput { height, velocity }
}

#[cfg(test)]
mod tests {
    use super::super::super::kernels::WaterKernel;
    use super::super::super::swe::{step, SweState};
    use super::*;

    const CFG: SweConfig = SweConfig {
        nx: 8,
        nz: 8,
        dx: 0.5,
        gravity: 9.81,
        damping: 0.1,
    };

    /// Deterministic pseudo-random field in a bounded range, no `std`/`rand`.
    fn fill(seed: u32, lo: f32, hi: f32, len: usize) -> Vec<f32> {
        let mut state = seed | 1;
        let mut out = vec![0.0_f32; len];
        let mut i = 0;
        while i < len {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = (state as f32) / (u32::MAX as f32);
            out[i] = lo + (hi - lo) * unit;
            i += 1;
        }
        out
    }

    /// Packs separate `u`/`v` fields into the flat velocity drive `ABI`.
    fn pack_vel(u: &[f32], v: &[f32]) -> Vec<f32> {
        let n = u.len();
        let mut vel = vec![0.0_f32; n * SWE_VEL_FLOATS];
        let mut i = 0;
        while i < n {
            vel[i * SWE_VEL_FLOATS] = u[i];
            vel[i * SWE_VEL_FLOATS + 1] = v[i];
            i += 1;
        }
        vel
    }

    #[test]
    fn twin_matches_cpu_golden_bit_for_bit() {
        let n = CFG.cell_count();
        for seed in [0x1234_5678_u32, 0x9e37_79b9, 0x0514_2024, 0xdead_beef] {
            // Positive depths plus signed velocities exercise both upwind
            // branches, the pressure gradient, continuity, and damping.
            let h = fill(seed, 0.2, 3.0, n);
            let u = fill(seed ^ 0x00f0_0f00, -2.0, 2.0, n);
            let v = fill(seed ^ 0x0f00_00f0, -2.0, 2.0, n);
            let vel = pack_vel(&u, &v);

            let out = dispatch_swe_step(&h, &vel, CFG, 0.01);
            let golden = step(
                &SweState {
                    h: h.clone(),
                    u: u.clone(),
                    v: v.clone(),
                },
                CFG,
                0.01,
            );
            let mut i = 0;
            while i < n {
                assert_eq!(
                    out.height[i], golden.h[i],
                    "depth diverged cell {i} seed {seed:#x}"
                );
                assert_eq!(
                    out.velocity[i * SWE_VEL_FLOATS],
                    golden.u[i],
                    "u diverged cell {i} seed {seed:#x}"
                );
                assert_eq!(
                    out.velocity[i * SWE_VEL_FLOATS + 1],
                    golden.v[i],
                    "v diverged cell {i} seed {seed:#x}"
                );
                i += 1;
            }
        }
    }

    #[test]
    fn still_body_stays_at_rest() {
        let n = CFG.cell_count();
        let h = vec![1.5_f32; n];
        let vel = vec![0.0_f32; n * SWE_VEL_FLOATS];
        let out = dispatch_swe_step(&h, &vel, CFG, 0.02);
        let mut i = 0;
        while i < n {
            assert_eq!(out.height[i], 1.5, "still depth drifted cell {i}");
            assert_eq!(
                out.velocity[i * SWE_VEL_FLOATS],
                0.0,
                "still u drifted cell {i}"
            );
            assert_eq!(
                out.velocity[i * SWE_VEL_FLOATS + 1],
                0.0,
                "still v drifted cell {i}"
            );
            i += 1;
        }
    }

    #[test]
    fn conserves_total_volume() {
        let n = CFG.cell_count();
        let h = fill(0x55aa_55aa, 0.5, 2.5, n);
        let u = fill(0x1111_2222, -1.0, 1.0, n);
        let v = fill(0x3333_4444, -1.0, 1.0, n);
        let vel = pack_vel(&u, &v);
        let before: f32 = h.iter().sum();
        let out = dispatch_swe_step(&h, &vel, CFG, 0.005);
        let after: f32 = out.height.iter().sum();
        // Reflective walls + telescoping face flux conserve volume to rounding.
        assert!(
            (after - before).abs() < 1e-3,
            "volume drift {}",
            after - before
        );
    }

    #[test]
    fn short_buffers_do_not_panic() {
        let out = dispatch_swe_step(&[1.0; 4], &[0.0; 4], CFG, 0.01);
        assert_eq!(out.height.len(), CFG.cell_count());
        assert_eq!(out.velocity.len(), CFG.cell_count() * SWE_VEL_FLOATS);
        assert!(out.height.iter().all(|&x| x.abs() < f32::EPSILON));
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let s = WATER_SWE_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains(&format!("fn {}", WaterKernel::SweStep.wesl_entry_point())));
        assert!(s.contains("@workgroup_size(8, 8, 1)"));
        assert!(s.contains("vel_in"));
        assert!(s.contains("vel_out"));
        assert!(s.contains("swe_params"));
        assert!(s.contains("height_next"));
        assert!(s.contains("height_curr"));
    }

    #[test]
    fn vel_abi_strides_are_consistent() {
        assert_eq!(SWE_VEL_STRIDE, 16);
        assert_eq!(SWE_VEL_FLOATS, 4);
        assert_eq!(SWE_VEL_STRIDE as usize, SWE_VEL_FLOATS * size_of::<f32>());
    }
}
