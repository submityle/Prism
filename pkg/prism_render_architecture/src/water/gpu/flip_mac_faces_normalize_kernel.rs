//! `FLIP`/`APIC` `MAC` face-normalize compute kernel: the `WESL`
//! `water_flip_mac_faces_normalize` shader plus its `CPU` twin.
//!
//! Face-normalize is the second half of the staggered `MAC` transfer. The
//! scatter pass ([`flip_mac_p2g_kernel`](super::flip_mac_p2g_kernel)) deposits
//! a momentum/mass pair per face:
//!
//! ```text
//! face_scatter[face * 2 + 0] = sum(w * v_comp)   (weighted momentum)
//! face_scatter[face * 2 + 1] = sum(w)            (weighted mass)
//! ```
//!
//! This pass divides momentum by mass to recover the grid face velocity
//! `v = sum(w * v) / sum(w)` per face. Because the trilinear weights are a
//! partition of unity ([`flip::trilinear_weights`](super::super::flip)), a lone
//! interior particle recovers its own velocity exactly and a uniform velocity
//! field is reproduced exactly. A face no particle reached has zero mass; its
//! velocity is defined to be zero, since an empty face carries no momentum.
//!
//! On-device the scatter accumulator is signed fixed-point packed into
//! `atomic<u32>`; the shader reads it back as a plain signed-integer array and
//! decodes each slot as `f32(fixed) / SCATTER_SCALE` before the divide. The
//! scale cancels in `momentum / mass`, so it only matters for comparing the
//! accumulated mass against the threshold. This twin consumes the un-quantized
//! `f32` momentum/mass the scatter twin ([`dispatch_mac_p2g`]) produces, so it
//! divides those directly and is the golden reference for the recovered
//! velocity rather than a bit-for-bit mirror of the device's fixed-point
//! rounding.
//!
//! [`WATER_FLIP_MAC_FACES_NORMALIZE_WESL`] is the shader (entry point
//! `water_flip_mac_faces_normalize`); [`dispatch_mac_faces_normalize`] is its
//! twin. The shader honours the
//! [`WaterKernel::FlipMacFacesNormalize`](super::super::kernels::WaterKernel)
//! descriptor — three storage buffers, one uniform block, dispatched over the
//! packed face family (`Faces` domain, `@workgroup_size(64, 1, 1)`). It shares
//! the `P2G` bind group layout: binding 1 is the scatter accumulator (read),
//! binding 2 the normalized face-velocity output (`read_write`), binding 3 the
//! uniform block; binding 0 (`particles`) is unused here and left undeclared.
//!
//! Packed face layout matches the scatter/divergence/projection stages: a
//! single buffer holds the three families back to back, `u` then `v` then `w`,
//! each i-fastest row-major over its own lattice. Normalize is family-agnostic:
//! every face is divided the same way, so the twin sweeps the flat face index.
//! Only `+ - * /` and comparisons appear.

use alloc::vec::Vec;

use super::super::EPS;

/// `WESL` source of the `MAC` face-normalize compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_MAC_FACES_NORMALIZE_WESL: &str =
    include_str!("water_flip_mac_faces_normalize.wesl");

/// Scatter slots per face: a `[momentum, mass]` pair.
const SCATTER_SLOTS: usize = 2;

/// Host mirror of the shader's `FacesNormalizeParams` uniform block.
///
/// Only the grid dimensions are needed: the packed face count (and therefore
/// the scatter length) is derived from them exactly as the scatter stage does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FacesNormalizeParams {
    /// Cell count along x.
    pub grid_nx: u32,
    /// Cell count along y.
    pub grid_ny: u32,
    /// Cell count along z.
    pub grid_nz: u32,
}

impl FacesNormalizeParams {
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

    /// Total packed-face count, `u_count + v_count + w_count`.
    #[must_use]
    pub fn faces_count(&self) -> usize {
        self.u_count()
            .saturating_add(self.v_count())
            .saturating_add(self.w_count())
    }

    /// Expected length of the `[momentum, mass]` scatter accumulator,
    /// `faces * 2`.
    #[must_use]
    pub fn scatter_len(&self) -> usize {
        self.faces_count().saturating_mul(SCATTER_SLOTS)
    }
}

/// `CPU` twin of the `water_flip_mac_faces_normalize` shader.
///
/// `scatter` is the un-quantized `[momentum, mass]` accumulator produced by
/// [`dispatch_mac_p2g`](super::flip_mac_p2g_kernel::dispatch_mac_p2g), two
/// `f32` per packed face in `u | v | w` block order. Returns one recovered
/// velocity per face, `faces_count` long. A face whose accumulated mass does
/// not exceed [`EPS`] recovers zero velocity. An empty grid yields an empty
/// buffer; a mis-sized `scatter` buffer yields an all-zero velocity field of
/// the correct length (no divide, no panic).
#[must_use]
pub fn dispatch_mac_faces_normalize(scatter: &[f32], params: FacesNormalizeParams) -> Vec<f32> {
    let faces = params.faces_count();
    if faces == 0 {
        return Vec::new();
    }
    let mut faces_out = alloc::vec![0.0f32; faces];

    if scatter.len() != params.scatter_len() {
        return faces_out;
    }

    for (f, slot) in faces_out.iter_mut().enumerate() {
        let mom = scatter[f * SCATTER_SLOTS];
        let mass = scatter[f * SCATTER_SLOTS + 1];
        if mass > EPS {
            *slot = mom / mass;
        }
    }

    faces_out
}

#[cfg(test)]
mod tests {
    use super::super::super::kernels::{DispatchDomain, WaterKernel};
    use super::super::flip_mac_p2g_kernel::{dispatch_mac_p2g, FlipP2GParams};
    use super::*;
    use alloc::vec;

    const ZERO_ROWS: [[f32; 3]; 3] = [[0.0; 3]; 3];

    fn fn_params(nx: u32, ny: u32, nz: u32) -> FacesNormalizeParams {
        FacesNormalizeParams {
            grid_nx: nx,
            grid_ny: ny,
            grid_nz: nz,
        }
    }

    fn p2g_params(nx: u32, ny: u32, nz: u32, pc: u32, dx: f32, affine: bool) -> FlipP2GParams {
        FlipP2GParams {
            grid_nx: nx,
            grid_ny: ny,
            grid_nz: nz,
            particle_count: pc,
            dx,
            use_affine: affine,
        }
    }

    /// One packed `FLIP`/`APIC` particle record (active).
    fn particle(pos: [f32; 3], vel: [f32; 3], rows: [[f32; 3]; 3]) -> [f32; 20] {
        [
            pos[0], pos[1], pos[2], 1.0, vel[0], vel[1], vel[2], 0.0, rows[0][0], rows[0][1],
            rows[0][2], 0.0, rows[1][0], rows[1][1], rows[1][2], 0.0, rows[2][0], rows[2][1],
            rows[2][2], 0.0,
        ]
    }

    /// Independent inline reference: divides momentum by mass per face with no
    /// shared helper, so a bit-exact match proves the twin's arithmetic rather
    /// than re-using its code. Mirrors the twin's `mass > EPS` guard and the
    /// zero fallback for empty faces.
    fn reference_normalize(scatter: &[f32], p: FacesNormalizeParams) -> Vec<f32> {
        let faces = (p.grid_nx as usize + 1) * p.grid_ny as usize * p.grid_nz as usize
            + p.grid_nx as usize * (p.grid_ny as usize + 1) * p.grid_nz as usize
            + p.grid_nx as usize * p.grid_ny as usize * (p.grid_nz as usize + 1);
        let mut out = vec![0.0f32; faces];
        if scatter.len() != faces * 2 {
            return out;
        }
        for (f, o) in out.iter_mut().enumerate() {
            let mom = scatter[f * 2];
            let mass = scatter[f * 2 + 1];
            if mass > 1.0e-6 {
                *o = mom / mass;
            }
        }
        out
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let s = WATER_FLIP_MAC_FACES_NORMALIZE_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains(&format!(
            "fn {}",
            WaterKernel::FlipMacFacesNormalize.wesl_entry_point()
        )));
        assert!(s.contains("@workgroup_size(64, 1, 1)"));
        assert!(s.contains("face_scatter"));
        assert!(s.contains("faces"));
        assert!(s.contains("sim_params"));

        let desc = WaterKernel::FlipMacFacesNormalize.descriptor();
        assert_eq!(desc.layout.storage_buffers, 3);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 0);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(desc.workgroup.x, 64);
        assert_eq!(desc.workgroup.y, 1);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Faces);
    }

    #[test]
    fn single_interior_particle_recovers_its_velocity() {
        let (nx, ny, nz) = (5u32, 5u32, 5u32);
        let dx = 0.3f32;
        // Deep interior so every family's eight nodes are in range and each
        // family's total weight is exactly one (partition of unity).
        let vel = [0.7f32, -0.4, 0.2];
        let buf = particle([2.4 * dx, 2.6 * dx, 2.5 * dx], vel, ZERO_ROWS);
        let scatter = dispatch_mac_p2g(&buf, p2g_params(nx, ny, nz, 1, dx, false));
        let faces = dispatch_mac_faces_normalize(&scatter, fn_params(nx, ny, nz));

        let uc = (nx as usize + 1) * ny as usize * nz as usize;
        let vc = nx as usize * (ny as usize + 1) * nz as usize;
        let mut touched = [0usize; 3];
        for (f, &v) in faces.iter().enumerate() {
            let mass = scatter[f * 2 + 1];
            if mass > EPS {
                let (comp, fam) = if f < uc {
                    (vel[0], 0usize)
                } else if f < uc + vc {
                    (vel[1], 1usize)
                } else {
                    (vel[2], 2usize)
                };
                assert!(
                    (v - comp).abs() < 1.0e-4,
                    "face {f} recovered {v}, expected {comp}"
                );
                touched[fam] += 1;
            }
        }
        assert!(
            touched[0] > 0 && touched[1] > 0 && touched[2] > 0,
            "each family must recover velocity on some face: {touched:?}"
        );
    }

    #[test]
    fn uniform_velocity_field_is_reproduced() {
        let (nx, ny, nz) = (6u32, 6u32, 6u32);
        let dx = 0.25f32;
        let vel = [0.3f32, 0.9, -0.5];
        // A cloud of particles all carrying the same velocity: every face they
        // collectively cover must recover exactly that uniform velocity.
        let seeds = [
            [2.0f32, 2.0, 2.0],
            [2.5, 3.0, 2.5],
            [3.0, 2.5, 3.0],
            [2.2, 3.3, 2.8],
            [3.4, 2.1, 2.6],
        ];
        let mut buf: Vec<f32> = Vec::new();
        for c in seeds {
            buf.extend_from_slice(&particle([c[0] * dx, c[1] * dx, c[2] * dx], vel, ZERO_ROWS));
        }
        let scatter = dispatch_mac_p2g(&buf, p2g_params(nx, ny, nz, seeds.len() as u32, dx, false));
        let faces = dispatch_mac_faces_normalize(&scatter, fn_params(nx, ny, nz));

        let uc = (nx as usize + 1) * ny as usize * nz as usize;
        let vc = nx as usize * (ny as usize + 1) * nz as usize;
        let mut recovered = 0usize;
        for (f, &v) in faces.iter().enumerate() {
            if scatter[f * 2 + 1] > EPS {
                let comp = if f < uc {
                    vel[0]
                } else if f < uc + vc {
                    vel[1]
                } else {
                    vel[2]
                };
                assert!(
                    (v - comp).abs() < 1.0e-4,
                    "face {f} recovered {v}, expected uniform {comp}"
                );
                recovered += 1;
            }
        }
        assert!(recovered > 0, "a uniform field must touch some faces");
    }

    #[test]
    fn matches_independent_inline_normalize() {
        let (nx, ny, nz) = (4u32, 3u32, 3u32);
        let dx = 0.5f32;
        // Several interior particles with varied velocities and affine rows so
        // the scatter is non-trivial and includes affine momentum.
        let seeds = [
            (
                [1.6f32, 1.4, 1.5],
                [0.2f32, -0.6, 0.4],
                [[0.5, -0.2, 0.1], [0.0, 0.3, -0.4], [-0.3, 0.2, 0.6]],
            ),
            ([2.3, 1.7, 1.8], [-0.5, 0.3, 0.7], ZERO_ROWS),
            (
                [1.9, 2.1, 1.3],
                [0.8, 0.1, -0.2],
                [[0.1, 0.0, 0.2], [-0.5, 0.4, 0.0], [0.3, -0.1, 0.2]],
            ),
        ];
        let mut buf: Vec<f32> = Vec::new();
        for (pos, vel, rows) in seeds {
            buf.extend_from_slice(&particle(
                [pos[0] * dx, pos[1] * dx, pos[2] * dx],
                vel,
                rows,
            ));
        }
        let scatter = dispatch_mac_p2g(&buf, p2g_params(nx, ny, nz, seeds.len() as u32, dx, true));

        let twin = dispatch_mac_faces_normalize(&scatter, fn_params(nx, ny, nz));
        let reference = reference_normalize(&scatter, fn_params(nx, ny, nz));
        assert_eq!(twin.len(), reference.len());
        let mut nonzero = 0usize;
        for (a, b) in twin.iter().zip(reference.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
            if a.to_bits() != 0.0f32.to_bits() {
                nonzero += 1;
            }
        }
        assert!(
            nonzero > 0,
            "normalize must recover non-zero velocities (guard against a vacuous all-zero match)"
        );
    }

    #[test]
    fn empty_faces_recover_zero_velocity() {
        let (nx, ny, nz) = (5u32, 5u32, 5u32);
        let dx = 0.3f32;
        // One localised particle leaves most faces with zero mass.
        let buf = particle([2.5 * dx, 2.5 * dx, 2.5 * dx], [0.6, -0.3, 0.2], ZERO_ROWS);
        let scatter = dispatch_mac_p2g(&buf, p2g_params(nx, ny, nz, 1, dx, false));
        let faces = dispatch_mac_faces_normalize(&scatter, fn_params(nx, ny, nz));

        let mut empty_seen = false;
        for (f, &v) in faces.iter().enumerate() {
            if scatter[f * 2 + 1] <= EPS {
                assert_eq!(
                    v.to_bits(),
                    0.0f32.to_bits(),
                    "face {f} has no mass yet recovered {v}"
                );
                empty_seen = true;
            }
        }
        assert!(empty_seen, "a localised particle must leave empty faces");
    }

    #[test]
    fn degenerate_requests_pass_through() {
        // Mis-sized scatter buffer: zeroed velocity field of the right length,
        // no divide, no panic.
        let p = fn_params(3, 3, 3);
        let short = vec![1.0f32; p.scatter_len() - 1];
        let z = dispatch_mac_faces_normalize(&short, p);
        assert_eq!(z.len(), p.faces_count());
        for v in &z {
            assert_eq!(v.to_bits(), 0.0f32.to_bits());
        }

        // Over-long scatter buffer: likewise zeroed, no panic.
        let long = vec![1.0f32; p.scatter_len() + 4];
        let z2 = dispatch_mac_faces_normalize(&long, p);
        assert_eq!(z2.len(), p.faces_count());
        for v in &z2 {
            assert_eq!(v.to_bits(), 0.0f32.to_bits());
        }

        // Empty grid: empty velocity field.
        assert!(dispatch_mac_faces_normalize(&[], fn_params(0, 0, 0)).is_empty());
    }
}
