//! `FLIP`/`APIC` `MAC` pressure compute kernel: the `WESL` `mac_pressure`
//! shader plus its bit-exact `CPU` twin.
//!
//! Stage 2 of the staggered `MAC` pressure projection. The incompressible
//! projection solves the negative-Laplacian pressure Poisson system `A p = b`,
//! whose right-hand side `b` is the per-cell divergence produced by stage 1
//! ([`flip_mac_divergence_kernel`](super::flip_mac_divergence_kernel)). This
//! kernel performs a single out-of-place damped-`Jacobi` relaxation sweep,
//! reproducing the crate's golden
//! [`pressure_multigrid::smooth`](super::super::pressure_multigrid::smooth)
//! with one iteration, so the parity test diffs the twin against an independent
//! inline stencil and the result is bit-exact by construction.
//!
//! [`WATER_FLIP_MAC_PRESSURE_WESL`] is the shader (entry point `mac_pressure`);
//! [`dispatch_mac_pressure`] is its twin. The shader honours the
//! [`WaterKernel::FlipMacPressure`](super::super::kernels::WaterKernel)
//! descriptor — four storage buffers, one uniform block, dispatched over a
//! 4x4x4 voxel brick (`Grid3d`). The divergence and `Jacobi` pressure passes
//! share the bind group: `faces` (unused here), `divergence` (read as `b`),
//! the `pressure_in`/`pressure_out` ping-pong, plus `mac_params`.
//!
//! The pressure field is an `n^3` node grid in x-fastest order
//! `idx = (z*n + y)*n + x`. Boundary nodes are a fixed `Dirichlet` layer copied
//! through unchanged; only interior nodes relax. Only `+ - * /` appear.

use alloc::vec::Vec;

use super::super::pressure_multigrid::smooth;

/// `WESL` source of the `MAC` pressure `Jacobi` compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_MAC_PRESSURE_WESL: &str = include_str!("water_flip_mac_pressure.wesl");

/// Host mirror of the shader's `MacPressureParams` uniform block.
///
/// `grid_n` is the cubic node-grid resolution, `h` the cell spacing, and
/// `omega` the `Jacobi` damping factor. The shader pads the block to 32 bytes
/// for `std140`/`std430` alignment; the twin only needs these three fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacPressureParams {
    /// Cubic node-grid resolution `n` (field is `n^3`).
    pub grid_n: u32,
    /// Cell spacing `h`.
    pub h: f32,
    /// Damped-`Jacobi` relaxation factor `omega`.
    pub omega: f32,
}

impl MacPressureParams {
    /// Node count `n^3`, saturating so an overflow cannot wrap to a small
    /// bogus length.
    #[must_use]
    pub fn node_count(&self) -> usize {
        let n = self.grid_n as usize;
        n.saturating_mul(n).saturating_mul(n)
    }
}

/// Bit-exact `CPU` twin of the `mac_pressure` shader.
///
/// `divergence` is the stage-1 right-hand side `b` and `pressure_in` is the
/// current pressure estimate, both `n^3` node fields in x-fastest order.
/// Returns the next pressure estimate after one damped-`Jacobi` sweep. A
/// degenerate grid (`n < 3`), an empty grid, or a mis-sized buffer passes
/// through as the untouched `pressure_in` (or an empty vector when there are no
/// nodes), never a panic — mirroring the shader's copy-through guards and the
/// golden smoother's early return.
#[must_use]
pub fn dispatch_mac_pressure(
    divergence: &[f32],
    pressure_in: &[f32],
    params: MacPressureParams,
) -> Vec<f32> {
    let nodes = params.node_count();
    if nodes == 0 {
        return Vec::new();
    }
    // Mis-sized buffers: refuse to index out of bounds, copy through.
    if pressure_in.len() != nodes || divergence.len() != nodes {
        return pressure_in.to_vec();
    }

    let mut p = pressure_in.to_vec();
    // One damped-Jacobi sweep. `smooth` is a true Jacobi update (residual read
    // from the current field, interior-only, Dirichlet boundary fixed), so a
    // single iteration equals the shader's out-of-place sweep bit-for-bit.
    smooth(
        &mut p,
        divergence,
        params.grid_n as usize,
        params.h,
        params.omega,
        1,
    );
    p
}

#[cfg(test)]
mod tests {
    use super::super::super::kernels::{DispatchDomain, WaterKernel};
    use super::*;
    use alloc::vec;

    fn params(n: u32, h: f32, omega: f32) -> MacPressureParams {
        MacPressureParams {
            grid_n: n,
            h,
            omega,
        }
    }

    fn node_index(x: usize, y: usize, z: usize, n: usize) -> usize {
        (z * n + y) * n + x
    }

    /// Deterministic non-trivial pressure and right-hand side fields.
    fn seed_fields(n: usize) -> (Vec<f32>, Vec<f32>) {
        let count = n * n * n;
        let mut pressure = vec![0.0f32; count];
        let mut rhs = vec![0.0f32; count];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = node_index(x, y, z, n);
                    pressure[c] = ((x * 7 + y * 3 + z) % 11) as f32 * 0.25 - 1.0;
                    rhs[c] = ((x + y * 2 + z * 5) % 13) as f32 * 0.5 - 2.0;
                }
            }
        }
        (pressure, rhs)
    }

    /// Independent single-sweep damped-Jacobi reference computed inline from the
    /// documented stencil, not via `smooth`, so the parity test genuinely pins
    /// the arithmetic and the Dirichlet boundary handling.
    fn reference_sweep(div: &[f32], p_in: &[f32], n: usize, h: f32, omega: f32) -> Vec<f32> {
        let mut out = p_in.to_vec();
        if n < 3 {
            return out;
        }
        let inv_h2 = 1.0 / (h * h);
        let factor = omega * h * h / 6.0;
        for z in 1..n - 1 {
            for y in 1..n - 1 {
                for x in 1..n - 1 {
                    let c = node_index(x, y, z, n);
                    let s = 6.0 * p_in[c]
                        - p_in[node_index(x - 1, y, z, n)]
                        - p_in[node_index(x + 1, y, z, n)]
                        - p_in[node_index(x, y - 1, z, n)]
                        - p_in[node_index(x, y + 1, z, n)]
                        - p_in[node_index(x, y, z - 1, n)]
                        - p_in[node_index(x, y, z + 1, n)];
                    let ap = s * inv_h2;
                    let r = div[c] - ap;
                    out[c] = p_in[c] + factor * r;
                }
            }
        }
        out
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let s = WATER_FLIP_MAC_PRESSURE_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains(&format!(
            "fn {}",
            WaterKernel::FlipMacPressure.wesl_entry_point()
        )));
        assert!(s.contains("@workgroup_size(4, 4, 4)"));
        assert!(s.contains("divergence"));
        assert!(s.contains("pressure_in"));
        assert!(s.contains("pressure_out"));
        assert!(s.contains("mac_params"));

        let desc = WaterKernel::FlipMacPressure.descriptor();
        assert_eq!(desc.layout.storage_buffers, 4);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 0);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(desc.workgroup.x, 4);
        assert_eq!(desc.workgroup.y, 4);
        assert_eq!(desc.workgroup.z, 4);
        assert_eq!(desc.domain, DispatchDomain::Grid3d);
    }

    #[test]
    fn matches_independent_jacobi_sweep() {
        let n = 5usize;
        let h = 0.5f32;
        let omega = 0.8f32;
        let (p_in, div) = seed_fields(n);

        let twin = dispatch_mac_pressure(&div, &p_in, params(n as u32, h, omega));
        let reference = reference_sweep(&div, &p_in, n, h, omega);
        assert_eq!(twin.len(), reference.len());

        let mut interior_changed = false;
        for (c, (&t, &r)) in twin.iter().zip(reference.iter()).enumerate() {
            assert_eq!(t.to_bits(), r.to_bits(), "pressure mismatch at node {c}");
        }
        for z in 1..n - 1 {
            for y in 1..n - 1 {
                for x in 1..n - 1 {
                    let c = node_index(x, y, z, n);
                    if twin[c].to_bits() != p_in[c].to_bits() {
                        interior_changed = true;
                    }
                }
            }
        }
        // Guard against a vacuous all-unchanged comparison.
        assert!(
            interior_changed,
            "sweep must relax at least one interior node"
        );
    }

    #[test]
    fn boundary_is_dirichlet_fixed() {
        let n = 5usize;
        let (p_in, div) = seed_fields(n);
        let twin = dispatch_mac_pressure(&div, &p_in, params(n as u32, 0.5, 1.0));
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let is_boundary =
                        x == 0 || y == 0 || z == 0 || x == n - 1 || y == n - 1 || z == n - 1;
                    if is_boundary {
                        let c = node_index(x, y, z, n);
                        assert_eq!(
                            twin[c].to_bits(),
                            p_in[c].to_bits(),
                            "boundary node ({x},{y},{z}) must stay fixed"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn zero_rhs_and_zero_pressure_stay_zero() {
        let n = 4usize;
        let count = n * n * n;
        let div = vec![0.0f32; count];
        let p_in = vec![0.0f32; count];
        let twin = dispatch_mac_pressure(&div, &p_in, params(n as u32, 0.25, 0.9));
        assert_eq!(twin.len(), count);
        assert!(twin.iter().all(|v| v.to_bits() == 0.0f32.to_bits()));
    }

    #[test]
    fn degenerate_requests_pass_through() {
        // n < 3 copies the field through unchanged.
        let small = vec![1.5f32, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let twin = dispatch_mac_pressure(&small, &small, params(2, 0.5, 1.0));
        assert_eq!(twin.len(), small.len());
        for (t, s) in twin.iter().zip(small.iter()) {
            assert_eq!(t.to_bits(), s.to_bits());
        }

        // Mis-sized divergence copies pressure through, no panic.
        let n = 4u32;
        let count = (n * n * n) as usize;
        let p_in = vec![0.5f32; count];
        let bad_div = vec![1.0f32; count - 1];
        let twin2 = dispatch_mac_pressure(&bad_div, &p_in, params(n, 0.5, 1.0));
        assert_eq!(twin2.len(), count);
        for (t, s) in twin2.iter().zip(p_in.iter()) {
            assert_eq!(t.to_bits(), s.to_bits());
        }

        // Empty grid -> empty output.
        assert!(dispatch_mac_pressure(&[], &[], params(0, 0.5, 1.0)).is_empty());
    }
}
