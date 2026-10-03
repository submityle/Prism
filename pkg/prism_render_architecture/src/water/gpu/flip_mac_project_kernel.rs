//! `FLIP`/`APIC` `MAC` projection compute kernel: the `WESL` `mac_project`
//! shader plus its `CPU` twin.
//!
//! Stage 3 of the staggered `MAC` pressure projection. Stage 1
//! ([`flip_mac_divergence_kernel`](super::flip_mac_divergence_kernel)) builds
//! the divergence right-hand side `b`; stage 2
//! ([`flip_mac_pressure_kernel`](super::flip_mac_pressure_kernel)) relaxes the
//! Poisson system `A p = b`. This stage applies the compact pressure gradient
//! to the face velocities so the field becomes discretely divergence-free:
//!
//! ```text
//! u'[face i] = u[face i] + (p[cell i] - p[cell i-1]) / dx
//! ```
//!
//! per staggered family. With the crate's golden operator
//! `A = (6 p_c - neighbours)/dx^2`
//! ([`pressure_multigrid::apply_operator`](super::super::pressure_multigrid::apply_operator))
//! and `b` the raw `+divergence`
//! ([`flip::cell_divergence`](super::super::flip::cell_divergence)), adding the
//! gradient drives the interior divergence to the solver residual
//! `b - A p ~ 0`. Only interior faces (separating two fluid cells) are updated;
//! the domain-boundary faces are the fixed solid/free-surface wall copied
//! through unchanged, matching the `Dirichlet` `p = 0` boundary of the solve.
//!
//! [`WATER_FLIP_MAC_PROJECT_WESL`] is the shader (entry point `mac_project`);
//! [`dispatch_mac_project`] is its twin. The shader honours the
//! [`WaterKernel::FlipMacProject`](super::super::kernels::WaterKernel)
//! descriptor — four storage buffers, one uniform block, dispatched linearly
//! over the packed face family (`Faces` domain, `@workgroup_size(64, 1, 1)`).
//! The `MAC` passes share the bind group: `faces` (read-write, updated here),
//! `divergence` (unused), the pressure pair (`pressure_a` read as the solved
//! field), plus `proj_params`.
//!
//! Packed face layout is identical to the divergence stage: a single `f32`
//! buffer holds the three staggered families back to back, `u` then `v` then
//! `w`, each i-fastest row-major over its own lattice. Cells and the co-located
//! pressure field are i-fastest `cell = i + nx*(j + ny*k)`. Only `+ - * /`
//! appear.

use alloc::vec::Vec;

use super::super::EPS;

/// `WESL` source of the `MAC` projection compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_MAC_PROJECT_WESL: &str = include_str!("water_flip_mac_project.wesl");

/// Host mirror of the shader's `MacProjectParams` uniform block.
///
/// `grid_nx`/`grid_ny`/`grid_nz` are the fluid cell counts along each axis and
/// `dx` is the uniform cell spacing. The shader pads the block to 32 bytes for
/// `std140`/`std430` alignment; the twin only needs these four fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacProjectParams {
    /// Cell count along x.
    pub grid_nx: u32,
    /// Cell count along y.
    pub grid_ny: u32,
    /// Cell count along z.
    pub grid_nz: u32,
    /// Uniform cell spacing `dx`.
    pub dx: f32,
}

impl MacProjectParams {
    /// Number of fluid cells, `nx * ny * nz`, saturating so an overflow cannot
    /// wrap into a small bogus length.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        (self.grid_nx as usize)
            .saturating_mul(self.grid_ny as usize)
            .saturating_mul(self.grid_nz as usize)
    }

    /// Number of `x`-face samples, `(nx + 1) * ny * nz`.
    #[must_use]
    fn u_count(&self) -> usize {
        (self.grid_nx as usize + 1)
            .saturating_mul(self.grid_ny as usize)
            .saturating_mul(self.grid_nz as usize)
    }

    /// Number of `y`-face samples, `nx * (ny + 1) * nz`.
    #[must_use]
    fn v_count(&self) -> usize {
        (self.grid_nx as usize)
            .saturating_mul(self.grid_ny as usize + 1)
            .saturating_mul(self.grid_nz as usize)
    }

    /// Number of `z`-face samples, `nx * ny * (nz + 1)`.
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

/// Cell (and co-located pressure) flat index: `i + nx*(j + ny*k)`.
fn cell_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    i + nx * (j + ny * k)
}

/// Bit-exact `CPU` twin of the `mac_project` shader.
///
/// `faces` is the packed `u|v|w` staggered-velocity buffer and `pressure` is
/// the solved pressure field, one scalar per cell in i-fastest order. Returns
/// the projected face buffer `u' = u + grad(p)` with interior faces updated and
/// domain-boundary faces copied through. A degenerate grid, a non-positive
/// `dx`, an empty grid, or a mis-sized `faces`/`pressure` buffer passes through
/// as the untouched `faces` (or an empty vector), never a panic — mirroring the
/// shader's bounds and spacing guards.
#[must_use]
pub fn dispatch_mac_project(faces: &[f32], pressure: &[f32], params: MacProjectParams) -> Vec<f32> {
    let cells = params.cell_count();
    let faces_len = params.faces_len();
    if faces_len == 0 {
        return Vec::new();
    }
    // Mis-sized buffers or degenerate spacing: refuse to touch, copy through.
    if faces.len() != faces_len || pressure.len() != cells || params.dx <= EPS {
        return faces.to_vec();
    }

    let nx = params.grid_nx as usize;
    let ny = params.grid_ny as usize;
    let nz = params.grid_nz as usize;
    let inv_dx = 1.0 / params.dx;

    let uc = params.u_count();
    let vc = params.v_count();

    let mut out = faces.to_vec();

    // u (x) faces: interior i in 1..=nx-1 separate cell (i-1) and cell (i).
    if nx >= 2 {
        let stride_j = nx + 1;
        for k in 0..nz {
            for j in 0..ny {
                for i in 1..nx {
                    let fid = i + stride_j * (j + ny * k);
                    let p_hi = pressure[cell_index(i, j, k, nx, ny)];
                    let p_lo = pressure[cell_index(i - 1, j, k, nx, ny)];
                    out[fid] += (p_hi - p_lo) * inv_dx;
                }
            }
        }
    }

    // v (y) faces: interior j in 1..=ny-1.
    if ny >= 2 {
        let stride_k = nx * (ny + 1);
        for k in 0..nz {
            for j in 1..ny {
                for i in 0..nx {
                    let fid = uc + i + nx * j + stride_k * k;
                    let p_hi = pressure[cell_index(i, j, k, nx, ny)];
                    let p_lo = pressure[cell_index(i, j - 1, k, nx, ny)];
                    out[fid] += (p_hi - p_lo) * inv_dx;
                }
            }
        }
    }

    // w (z) faces: interior k in 1..=nz-1.
    if nz >= 2 {
        for k in 1..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let fid = uc + vc + i + nx * (j + ny * k);
                    let p_hi = pressure[cell_index(i, j, k, nx, ny)];
                    let p_lo = pressure[cell_index(i, j, k - 1, nx, ny)];
                    out[fid] += (p_hi - p_lo) * inv_dx;
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::super::super::flip::{cell_divergence, FaceVelocities};
    use super::super::super::kernels::{DispatchDomain, WaterKernel};
    use super::super::super::pressure_multigrid::{solve, MultigridConfig};
    use super::*;
    use alloc::vec;

    fn params(nx: u32, ny: u32, nz: u32, dx: f32) -> MacProjectParams {
        MacProjectParams {
            grid_nx: nx,
            grid_ny: ny,
            grid_nz: nz,
            dx,
        }
    }

    fn u_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
        i + (nx + 1) * (j + ny * k)
    }

    fn v_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
        i + nx * (j + (ny + 1) * k)
    }

    fn w_index(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
        i + nx * (j + ny * k)
    }

    fn offsets(nx: usize, ny: usize, nz: usize) -> (usize, usize) {
        let uc = (nx + 1) * ny * nz;
        let vc = nx * (ny + 1) * nz;
        (uc, uc + vc)
    }

    fn faces_len(nx: usize, ny: usize, nz: usize) -> usize {
        (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1)
    }

    /// Deterministic divergent packed face field (varies across faces so the
    /// per-cell divergence is clearly non-zero).
    fn seed_faces(nx: usize, ny: usize, nz: usize) -> Vec<f32> {
        let mut faces = vec![0.0f32; faces_len(nx, ny, nz)];
        let (voff, woff) = offsets(nx, ny, nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    faces[u_index(i, j, k, nx, ny)] = ((i * 5 + j * 2 + k) % 7) as f32 * 0.3 - 0.9;
                }
            }
        }
        for k in 0..nz {
            for j in 0..=ny {
                for i in 0..nx {
                    faces[voff + v_index(i, j, k, nx, ny)] =
                        ((i + j * 4 + k * 2) % 9) as f32 * 0.2 - 0.8;
                }
            }
        }
        for k in 0..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    faces[woff + w_index(i, j, k, nx, ny)] =
                        ((i * 2 + j + k * 3) % 5) as f32 * 0.4 - 0.8;
                }
            }
        }
        faces
    }

    /// Per-cell divergence field via the golden `cell_divergence`, independent
    /// of the twin's gradient arithmetic.
    fn divergence_field(faces: &[f32], nx: usize, ny: usize, nz: usize, dx: f32) -> Vec<f32> {
        let (voff, woff) = offsets(nx, ny, nz);
        let mut div = vec![0.0f32; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let fv = FaceVelocities {
                        x_pos: faces[u_index(i + 1, j, k, nx, ny)],
                        x_neg: faces[u_index(i, j, k, nx, ny)],
                        y_pos: faces[voff + v_index(i, j + 1, k, nx, ny)],
                        y_neg: faces[voff + v_index(i, j, k, nx, ny)],
                        z_pos: faces[woff + w_index(i, j, k + 1, nx, ny)],
                        z_neg: faces[woff + w_index(i, j, k, nx, ny)],
                    };
                    div[i + nx * (j + ny * k)] = cell_divergence(fv, dx);
                }
            }
        }
        div
    }

    /// L2 norm over the interior cells `1..n-1` on each axis.
    fn interior_l2(div: &[f32], n: usize) -> f32 {
        let mut acc = 0.0f32;
        for k in 1..n - 1 {
            for j in 1..n - 1 {
                for i in 1..n - 1 {
                    let v = div[i + n * (j + n * k)];
                    acc += v * v;
                }
            }
        }
        acc.sqrt()
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let s = WATER_FLIP_MAC_PROJECT_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains(&format!(
            "fn {}",
            WaterKernel::FlipMacProject.wesl_entry_point()
        )));
        assert!(s.contains("@workgroup_size(64, 1, 1)"));
        assert!(s.contains("faces"));
        assert!(s.contains("pressure_a"));
        assert!(s.contains("proj_params"));

        let desc = WaterKernel::FlipMacProject.descriptor();
        assert_eq!(desc.layout.storage_buffers, 4);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 0);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(desc.workgroup.x, 64);
        assert_eq!(desc.workgroup.y, 1);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Faces);
    }

    #[test]
    fn matches_independent_gradient_update() {
        let (nx, ny, nz) = (4usize, 3usize, 3usize);
        let dx = 0.5f32;
        let faces = seed_faces(nx, ny, nz);
        let cells = nx * ny * nz;
        let mut pressure = vec![0.0f32; cells];
        for (c, slot) in pressure.iter_mut().enumerate() {
            *slot = (c % 13) as f32 * 0.25 - 1.5;
        }

        let twin = dispatch_mac_project(
            &faces,
            &pressure,
            params(nx as u32, ny as u32, nz as u32, dx),
        );

        // Independent reference: copy, then add the compact gradient per family.
        let (voff, woff) = offsets(nx, ny, nz);
        let inv_dx = 1.0 / dx;
        let mut reference = faces.clone();
        for k in 0..nz {
            for j in 0..ny {
                for i in 1..nx {
                    let hi = pressure[i + nx * (j + ny * k)];
                    let lo = pressure[(i - 1) + nx * (j + ny * k)];
                    reference[u_index(i, j, k, nx, ny)] += (hi - lo) * inv_dx;
                }
            }
        }
        for k in 0..nz {
            for j in 1..ny {
                for i in 0..nx {
                    let hi = pressure[i + nx * (j + ny * k)];
                    let lo = pressure[i + nx * ((j - 1) + ny * k)];
                    reference[voff + v_index(i, j, k, nx, ny)] += (hi - lo) * inv_dx;
                }
            }
        }
        for k in 1..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let hi = pressure[i + nx * (j + ny * k)];
                    let lo = pressure[i + nx * (j + ny * (k - 1))];
                    reference[woff + w_index(i, j, k, nx, ny)] += (hi - lo) * inv_dx;
                }
            }
        }

        assert_eq!(twin.len(), reference.len());
        let mut any_changed = false;
        for (c, (&t, &r)) in twin.iter().zip(reference.iter()).enumerate() {
            assert_eq!(t.to_bits(), r.to_bits(), "face mismatch at {c}");
        }
        for (&t, &f) in twin.iter().zip(faces.iter()) {
            if t.to_bits() != f.to_bits() {
                any_changed = true;
            }
        }
        assert!(any_changed, "projection must modify at least one face");
    }

    #[test]
    fn boundary_faces_are_fixed() {
        let (nx, ny, nz) = (4usize, 4usize, 4usize);
        let dx = 0.5f32;
        let faces = seed_faces(nx, ny, nz);
        let cells = nx * ny * nz;
        let pressure: Vec<f32> = (0..cells).map(|c| (c % 11) as f32 * 0.3 - 1.0).collect();
        let twin = dispatch_mac_project(
            &faces,
            &pressure,
            params(nx as u32, ny as u32, nz as u32, dx),
        );

        // The two extreme x-face planes i = 0 and i = nx are domain walls.
        for k in 0..nz {
            for j in 0..ny {
                for &i in &[0usize, nx] {
                    let f = u_index(i, j, k, nx, ny);
                    assert_eq!(
                        twin[f].to_bits(),
                        faces[f].to_bits(),
                        "x wall face ({i},{j},{k}) must stay fixed"
                    );
                }
            }
        }
    }

    #[test]
    fn projection_reduces_interior_divergence() {
        // Cubic grid at a valid vertex-centred level size so the golden
        // multigrid solve applies and cells alias pressure nodes 1:1.
        let n = 5usize;
        let dx = 0.5f32;
        let faces0 = seed_faces(n, n, n);

        // Right-hand side b and solved pressure via golden paths only.
        let b = divergence_field(&faces0, n, n, n, dx);
        let report = solve(&b, n, dx, MultigridConfig::balanced());
        assert_eq!(report.pressure.len(), n * n * n);

        let faces1 = dispatch_mac_project(
            &faces0,
            &report.pressure,
            params(n as u32, n as u32, n as u32, dx),
        );
        let b1 = divergence_field(&faces1, n, n, n, dx);

        let before = interior_l2(&b, n);
        let after = interior_l2(&b1, n);

        // The seed must be genuinely divergent (non-vacuous).
        assert!(before > 1.0e-3, "seed divergence too small: {before}");
        // One full multigrid solve + projection must cut the interior
        // divergence by at least an order of magnitude.
        assert!(
            after * 10.0 < before,
            "projection failed to reduce interior divergence: before={before} after={after}"
        );
    }

    #[test]
    fn degenerate_requests_pass_through() {
        // Non-positive dx: untouched.
        let faces = seed_faces(3, 3, 3);
        let cells = 27;
        let pressure = vec![0.5f32; cells];
        let twin = dispatch_mac_project(&faces, &pressure, params(3, 3, 3, 0.0));
        assert_eq!(twin.len(), faces.len());
        for (t, f) in twin.iter().zip(faces.iter()) {
            assert_eq!(t.to_bits(), f.to_bits());
        }

        // Mis-sized pressure: copy faces through, no panic.
        let bad_pressure = vec![0.1f32; cells - 1];
        let twin2 = dispatch_mac_project(&faces, &bad_pressure, params(3, 3, 3, 0.5));
        assert_eq!(twin2.len(), faces.len());
        for (t, f) in twin2.iter().zip(faces.iter()) {
            assert_eq!(t.to_bits(), f.to_bits());
        }

        // Mis-sized faces: copy through.
        let short_faces = vec![1.0f32; faces.len() - 1];
        let twin3 = dispatch_mac_project(&short_faces, &pressure, params(3, 3, 3, 0.5));
        assert_eq!(twin3.len(), short_faces.len());

        // Empty grid -> empty output.
        assert!(dispatch_mac_project(&[], &[], params(0, 0, 0, 0.5)).is_empty());
    }
}
