//! `FLIP`/`APIC` `MAC` divergence compute kernel: the `WESL`
//! `mac_divergence` shader plus its bit-exact `CPU` twin.
//!
//! Stage 1 of the staggered `MAC` pressure projection. The incompressible
//! projection drives the discrete divergence of the face-centred velocity
//! field to zero; this kernel evaluates that right-hand side, one scalar per
//! fluid cell. The twin reproduces the shader cell-for-cell by reusing the
//! crate's golden [`flip::cell_divergence`](super::super::flip::cell_divergence),
//! so the parity tests diff the twin against an independent gather path and the
//! result is bit-exact by construction.
//!
//! [`WATER_FLIP_MAC_DIVERGENCE_WESL`] is the shader (entry point
//! `mac_divergence`); [`dispatch_mac_divergence`] is its twin. The shader
//! honours the [`WaterKernel::FlipMacDivergence`](super::super::kernels::WaterKernel)
//! descriptor — four storage buffers, one uniform block, dispatched over a
//! 4x4x4 voxel brick (`Grid3d`). The `MAC` divergence and `Jacobi` pressure
//! passes share the bind group: `faces` (read), `divergence` (written here),
//! and the ping-pong pressure pair (bound but untouched by this stage) plus
//! `mac_params`.
//!
//! Packed face layout: one `f32` buffer holds the three staggered face
//! families back to back, `u` then `v` then `w`, each i-fastest row-major over
//! its own face lattice. Only `+ - /` appear, matching the determinism policy.

use alloc::vec;
use alloc::vec::Vec;

use super::super::flip::{cell_divergence, FaceVelocities};

/// `WESL` source of the `MAC` divergence compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_MAC_DIVERGENCE_WESL: &str = include_str!("water_flip_mac_divergence.wesl");

/// Host mirror of the shader's `MacParams` uniform block.
///
/// `grid_nx`/`grid_ny`/`grid_nz` are the fluid cell counts along each axis and
/// `dx` is the uniform cell spacing. The shader pads the block to 32 bytes for
/// `std140`/`std430` alignment; the twin only needs these four fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacDivergenceParams {
    /// Cell count along x.
    pub grid_nx: u32,
    /// Cell count along y.
    pub grid_ny: u32,
    /// Cell count along z.
    pub grid_nz: u32,
    /// Uniform cell spacing `dx`.
    pub dx: f32,
}

impl MacDivergenceParams {
    /// Number of fluid cells, `nx * ny * nz`, saturating so an overflow cannot
    /// wrap into a small bogus length.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        (self.grid_nx as usize)
            .saturating_mul(self.grid_ny as usize)
            .saturating_mul(self.grid_nz as usize)
    }

    /// Number of `+x` face samples, `(nx + 1) * ny * nz`.
    #[must_use]
    fn u_count(&self) -> usize {
        (self.grid_nx as usize + 1)
            .saturating_mul(self.grid_ny as usize)
            .saturating_mul(self.grid_nz as usize)
    }

    /// Number of `+y` face samples, `nx * (ny + 1) * nz`.
    #[must_use]
    fn v_count(&self) -> usize {
        (self.grid_nx as usize)
            .saturating_mul(self.grid_ny as usize + 1)
            .saturating_mul(self.grid_nz as usize)
    }

    /// Number of `+z` face samples, `nx * ny * (nz + 1)`.
    #[must_use]
    fn w_count(&self) -> usize {
        (self.grid_nx as usize)
            .saturating_mul(self.grid_ny as usize)
            .saturating_mul(self.grid_nz as usize + 1)
    }

    /// Total packed-face buffer length, `u_count + v_count + w_count`.
    #[must_use]
    fn faces_len(&self) -> usize {
        self.u_count()
            .saturating_add(self.v_count())
            .saturating_add(self.w_count())
    }
}

/// `u` (x-face) flat index: `i + (nx + 1) * (j + ny * k)`.
fn u_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + (nx + 1) * (j + ny * k)
}

/// `v` (y-face) flat index: `i + nx * (j + (ny + 1) * k)`.
fn v_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + nx * (j + (ny + 1) * k)
}

/// `w` (z-face) flat index: `i + nx * (j + ny * k)`.
fn w_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + nx * (j + ny * k)
}

/// Cell flat index: `i + nx * (j + ny * k)`.
fn cell_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + nx * (j + ny * k)
}

/// Bit-exact `CPU` twin of the `mac_divergence` shader.
///
/// `faces` is the packed `u|v|w` staggered-velocity buffer. Returns one
/// divergence scalar per cell in i-fastest row-major order. Degenerate or
/// mis-sized requests pass through as an all-zero field (or an empty vector
/// when the grid has no cells), never a panic — mirroring the shader's
/// bounds guard and the golden `cell_divergence` non-positive-`dx` guard.
#[must_use]
pub fn dispatch_mac_divergence(faces: &[f32], params: MacDivergenceParams) -> Vec<f32> {
    let cells = params.cell_count();
    if cells == 0 {
        return Vec::new();
    }
    // Mis-sized face buffer: refuse to index out of bounds, emit zeros.
    if faces.len() != params.faces_len() {
        return vec![0.0; cells];
    }

    let nx = params.grid_nx as usize;
    let ny = params.grid_ny as usize;
    let nz = params.grid_nz as usize;

    let u_off = 0usize;
    let v_off = params.u_count();
    let w_off = v_off + params.v_count();

    let mut out = vec![0.0f32; cells];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let vel = FaceVelocities {
                    x_neg: faces[u_off + u_index(i, j, k, nx, ny)],
                    x_pos: faces[u_off + u_index(i + 1, j, k, nx, ny)],
                    y_neg: faces[v_off + v_index(i, j, k, nx, ny)],
                    y_pos: faces[v_off + v_index(i, j + 1, k, nx, ny)],
                    z_neg: faces[w_off + w_index(i, j, k, nx, ny)],
                    z_pos: faces[w_off + w_index(i, j, k + 1, nx, ny)],
                };
                out[cell_index(i, j, k, nx, ny)] = cell_divergence(vel, params.dx);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::kernels::{DispatchDomain, WaterKernel};
    use super::*;

    fn params(nx: u32, ny: u32, nz: u32, dx: f32) -> MacDivergenceParams {
        MacDivergenceParams {
            grid_nx: nx,
            grid_ny: ny,
            grid_nz: nz,
            dx,
        }
    }

    /// Independent reference for the three packed-face offsets, derived with a
    /// different grouping than the twin so an offset bug cannot hide.
    fn offsets(p: &MacDivergenceParams) -> (usize, usize, usize) {
        let nx = p.grid_nx as usize;
        let ny = p.grid_ny as usize;
        let nz = p.grid_nz as usize;
        let uc = (nx + 1) * ny * nz;
        let vc = nx * (ny + 1) * nz;
        (0, uc, uc + vc)
    }

    /// Build a packed face buffer whose every sample equals its own global flat
    /// index (as `f32`). Any mistake in the kernel's family offset or per-face
    /// indexing then yields the wrong divergence, so the parity test genuinely
    /// pins the staggered addressing rather than only the arithmetic.
    fn index_sentinel_faces(p: &MacDivergenceParams) -> Vec<f32> {
        let n = p.faces_len();
        (0..n).map(|idx| idx as f32).collect()
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let s = WATER_FLIP_MAC_DIVERGENCE_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains(&format!(
            "fn {}",
            WaterKernel::FlipMacDivergence.wesl_entry_point()
        )));
        assert!(s.contains("@workgroup_size(4, 4, 4)"));
        assert!(s.contains("faces"));
        assert!(s.contains("divergence"));
        assert!(s.contains("mac_params"));

        let desc = WaterKernel::FlipMacDivergence.descriptor();
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
    fn uniform_flow_is_divergence_free() {
        let p = params(3, 2, 2, 0.25);
        // Every face carries the same velocity: opposing faces cancel exactly.
        let faces = vec![1.5f32; p.faces_len()];
        let div = dispatch_mac_divergence(&faces, p);
        assert_eq!(div.len(), p.cell_count());
        for d in &div {
            assert_eq!(d.to_bits(), 0.0f32.to_bits());
        }
    }

    #[test]
    fn matches_independent_gather_on_index_sentinels() {
        let p = params(3, 3, 2, 0.5);
        let faces = index_sentinel_faces(&p);
        let twin = dispatch_mac_divergence(&faces, p);
        assert_eq!(twin.len(), p.cell_count());

        let (u_off, v_off, w_off) = offsets(&p);
        let nx = p.grid_nx as usize;
        let ny = p.grid_ny as usize;
        let nz = p.grid_nz as usize;

        // Independent gather: recompute each face's flat index inline (not via
        // the twin's helpers) and the divergence by the raw formula, then diff
        // bit-for-bit.
        let mut non_trivial = false;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x_neg = faces[u_off + (i + (nx + 1) * (j + ny * k))];
                    let x_pos = faces[u_off + ((i + 1) + (nx + 1) * (j + ny * k))];
                    let y_neg = faces[v_off + (i + nx * (j + (ny + 1) * k))];
                    let y_pos = faces[v_off + (i + nx * ((j + 1) + (ny + 1) * k))];
                    let z_neg = faces[w_off + (i + nx * (j + ny * k))];
                    let z_pos = faces[w_off + (i + nx * (j + ny * (k + 1)))];
                    let reference = ((x_pos - x_neg) + (y_pos - y_neg) + (z_pos - z_neg)) / p.dx;
                    let cell = i + nx * (j + ny * k);
                    assert_eq!(
                        twin[cell].to_bits(),
                        reference.to_bits(),
                        "divergence mismatch at cell ({i},{j},{k})"
                    );
                    if reference.to_bits() != 0.0f32.to_bits() {
                        non_trivial = true;
                    }
                }
            }
        }
        // Guard against a vacuous all-zero comparison.
        assert!(
            non_trivial,
            "sentinel field must produce non-zero divergence"
        );
    }

    #[test]
    fn constant_outflow_has_known_sign_and_magnitude() {
        // One cell, dx = 0.5. +x face = 2, -x face = 0, other opposing faces
        // equal. Net divergence = (2 - 0) / 0.5 = 4.
        let p = params(1, 1, 1, 0.5);
        let mut faces = vec![0.0f32; p.faces_len()];
        // u lattice is 2x1x1: index 0 = -x face, index 1 = +x face.
        let (u_off, _, _) = offsets(&p);
        faces[u_off + 1] = 2.0;
        let div = dispatch_mac_divergence(&faces, p);
        assert_eq!(div.len(), 1);
        assert_eq!(div[0].to_bits(), 4.0f32.to_bits());
    }

    #[test]
    fn degenerate_requests_pass_through() {
        // Non-positive dx -> zero field (golden guard).
        let p = params(2, 2, 2, 0.0);
        let faces = vec![3.0f32; p.faces_len()];
        let div = dispatch_mac_divergence(&faces, p);
        assert_eq!(div.len(), p.cell_count());
        assert!(div.iter().all(|d| d.to_bits() == 0.0f32.to_bits()));

        // Mis-sized face buffer -> zero field, no panic.
        let p2 = params(2, 2, 2, 0.5);
        let bad = vec![1.0f32; p2.faces_len() - 1];
        let div2 = dispatch_mac_divergence(&bad, p2);
        assert_eq!(div2.len(), p2.cell_count());
        assert!(div2.iter().all(|d| d.to_bits() == 0.0f32.to_bits()));

        // Empty grid -> empty output.
        let p3 = params(0, 4, 4, 0.5);
        assert!(dispatch_mac_divergence(&[], p3).is_empty());
    }
}
