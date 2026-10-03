//! `FLIP`/`APIC` `MAC` grid-to-particle (`G2P`) gather compute kernel: the
//! `WESL` `water_flip_mac_g2p` shader plus its bit-exact `CPU` twin.
//!
//! `G2P` is the half that closes the `FLIP`/`APIC` loop against the staggered
//! `MAC` pressure projection. The scatter pass
//! ([`flip_mac_p2g_kernel`](super::flip_mac_p2g_kernel)) splats each particle's
//! velocity onto the face it is normal to; the normalize pass
//! ([`flip_mac_faces_normalize_kernel`](super::flip_mac_faces_normalize_kernel))
//! turns the accumulated momentum/mass into a face velocity; the projection pass
//! removes the divergent part. This kernel gathers the projected face field back
//! to each particle and rebuilds its velocity (and `APIC` affine matrix).
//!
//! `FLIP`/`PIC` blend and the `FLIP` delta
//! ---------------------------------------
//! Classic `FLIP` carries the change the grid applied this step, not the grid
//! velocity itself. On the staggered grid that change is read directly as
//! `faces_projected - faces_preprojection` per sampled face, so no pressure
//! gradient is recomputed on the particle side and the delta is exact even where
//! the boundary stencil clamps the gradient. `PIC` gathers the projected face
//! velocity. They are mixed as `(1 - alpha) * PIC + alpha * FLIP` with `alpha`
//! clamped to `0..=1`: `alpha = 0` is pure dissipative `PIC`, `alpha = 1` is pure
//! `FLIP`. When `faces_projected == faces_preprojection` the delta is zero and
//! `FLIP` keeps the particle's own incoming velocity.
//!
//! `APIC` affine reconstruction
//! ----------------------------
//! When the affine field is carried, each affine row is rebuilt from the
//! projected face velocities of its own component family:
//!
//! ```text
//! C_row_a = (3 / dx^2) * sum(w * v_proj * (x_face - x_p))
//! ```
//!
//! using the scalar multilinear inertia inverse `D^-1 = 3 / dx^2`, the inverse of
//! the `P2G` forward affine map. A zero affine request writes zero rows (plain
//! `PIC`/`FLIP`). The trilinear `D` is not exactly constant, so `3 / dx^2` is the
//! conventional multilinear approximation rather than an exact gradient recovery;
//! the twin mirrors the shader's arithmetic exactly but does not claim exact
//! affine-gradient round-tripping.
//!
//! [`WATER_FLIP_MAC_G2P_WESL`] is the shader (entry point `water_flip_mac_g2p`);
//! [`dispatch_mac_g2p`] is its twin. The shader honours the
//! [`WaterKernel::FlipMacG2P`](super::super::kernels::WaterKernel) descriptor —
//! three storage buffers, one uniform block, dispatched over the particle pool
//! (`Particle` domain, `@workgroup_size(64, 1, 1)`). Binding 0 is the live
//! particle pool (`read_write`), binding 1 the projected faces (read), binding 2
//! the pre-projection faces (read), binding 3 the uniform block. Particles share
//! the `P2G` layout: five `vec4` lanes (20 `f32`) each — position+active,
//! velocity, three `APIC` affine rows.
//!
//! Packed face layout matches the scatter/normalize/divergence/projection
//! stages: a single buffer holds the three families back to back, `u` then `v`
//! then `w`, each i-fastest row-major over its own lattice. The grid origin is
//! the lattice origin (`g = pos / dx`), matching the `P2G` twin. Only `+ - * /`
//! and comparisons appear; the floor is open-coded in [`floor_i32`].

use alloc::vec::Vec;

use super::super::{Vec3, EPS};
use super::flip_mac_p2g_kernel::FLIP_P2G_PARTICLE_FLOATS;

/// `WESL` source of the `MAC` `G2P` gather compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_MAC_G2P_WESL: &str = include_str!("water_flip_mac_g2p.wesl");

/// Multilinear `APIC` inertia inverse numerator: `D^-1 = APIC_INV_D_NUM / dx^2`.
const APIC_INV_D_NUM: f32 = 3.0;

/// Host mirror of the shader's `FlipG2PParams` uniform block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipG2PParams {
    /// Cell count along x.
    pub grid_nx: u32,
    /// Cell count along y.
    pub grid_ny: u32,
    /// Cell count along z.
    pub grid_nz: u32,
    /// Number of particles in the pool.
    pub particle_count: u32,
    /// Uniform cell spacing `dx`.
    pub dx: f32,
    /// `FLIP`/`PIC` blend factor, clamped to `0..=1` before use.
    pub flip_blend: f32,
    /// Whether the affine (`APIC`) rows are reconstructed.
    pub use_affine: bool,
}

impl FlipG2PParams {
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
}

/// Floor of an `f32` as an `i32`, without the forbidden `floor` intrinsic.
///
/// Truncation toward zero already floors non-negative values; for a negative
/// non-integer the truncated value is strictly greater than the input, so one
/// is subtracted. Only `<`/`>` comparisons and casts are used.
fn floor_i32(x: f32) -> i32 {
    let t = x as i32;
    let tf = t as f32;
    if tf > x {
        t - 1
    } else {
        t
    }
}

/// Clamps `x` into the closed unit interval `0.0..=1.0` without calling the
/// disallowed `f32::clamp`. Only `<`/`>` comparisons are used, so the
/// sqrt-only float policy of this crate is preserved and the result matches
/// the WESL `clamp(blend, 0.0, 1.0)` applied device-side.
#[expect(
    clippy::manual_clamp,
    reason = "f32::clamp is a disallowed float method in this crate; the manual branch keeps the sqrt-only float policy"
)]
fn clamp_unit(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// Trilinear weight along one axis for corner `c` (`0` = low node, `1` = high
/// node).
fn w1(f: f32, c: i32) -> f32 {
    if c == 0 {
        1.0 - f
    } else {
        f
    }
}

/// One component family's gather result: the `PIC` component, the `FLIP` delta
/// component, and the `APIC` affine row before the `D^-1` scale.
struct AxisGather {
    pic: f32,
    delta: f32,
    row: Vec3,
}

/// Gathers one velocity component from its staggered face family.
///
/// `g` is the particle position in grid units (`pos / dx`); `shift` the family's
/// half-cell node offset; `max_*` the inclusive max index of the family lattice;
/// `stride_j`/`stride_k`/`off` the packed-block layout. The returned affine row
/// is the raw moment sum; the caller applies `D^-1`.
#[expect(
    clippy::too_many_arguments,
    reason = "The staggered G2P gather is parameterised per face family by its node half-offset, inclusive lattice limits, and packed-block strides/offset; bundling these inherent per-family quantities into a struct would only rename them."
)]
fn gather_axis(
    projected: &[f32],
    preprojection: &[f32],
    g: Vec3,
    shift: Vec3,
    dx: f32,
    max_i: i32,
    max_j: i32,
    max_k: i32,
    stride_j: usize,
    stride_k: usize,
    off: usize,
) -> AxisGather {
    let lattice = g.sub(shift);
    let bi = floor_i32(lattice.x);
    let bj = floor_i32(lattice.y);
    let bk = floor_i32(lattice.z);
    let fx = lattice.x - bi as f32;
    let fy = lattice.y - bj as f32;
    let fz = lattice.z - bk as f32;

    let mut pic = 0.0f32;
    let mut delta = 0.0f32;
    let mut row = Vec3::ZERO;

    let mut cz = 0i32;
    while cz < 2 {
        let gk = bk + cz;
        let mut cy = 0i32;
        while cy < 2 {
            let gj = bj + cy;
            let mut cx = 0i32;
            while cx < 2 {
                let gi = bi + cx;
                if gi >= 0 && gi <= max_i && gj >= 0 && gj <= max_j && gk >= 0 && gk <= max_k {
                    let weight = w1(fx, cx) * w1(fy, cy) * w1(fz, cz);
                    let flat = off + gi as usize + stride_j * gj as usize + stride_k * gk as usize;
                    let v_proj = projected[flat];
                    let v_pre = preprojection[flat];
                    pic += v_proj * weight;
                    delta += (v_proj - v_pre) * weight;
                    let node = Vec3::new(
                        gi as f32 + shift.x,
                        gj as f32 + shift.y,
                        gk as f32 + shift.z,
                    );
                    let offset = node.sub(g).scale(dx);
                    row = row.add(offset.scale(v_proj * weight));
                }
                cx += 1;
            }
            cy += 1;
        }
        cz += 1;
    }

    AxisGather { pic, delta, row }
}

/// `CPU` twin of the `water_flip_mac_g2p` shader.
///
/// `particles` is the packed `FLIP`/`APIC` pool ([`FLIP_P2G_PARTICLE_FLOATS`]
/// per particle); `faces_projected` and `faces_preprojection` are the two packed
/// face-velocity fields (one `f32` per packed face). Returns the pool with each
/// active particle's velocity (and, when `use_affine`, its three affine rows)
/// rebuilt from the grid; inactive particles (lane 3 `< 0.5`) are untouched.
///
/// A degenerate spacing, an empty pool, a mis-sized particle buffer, a
/// zero-extent grid, or a face buffer of the wrong length yields the input
/// unchanged (pass-through, no gather, no panic).
#[must_use]
pub fn dispatch_mac_g2p(
    particles: &[f32],
    faces_projected: &[f32],
    faces_preprojection: &[f32],
    params: FlipG2PParams,
) -> Vec<f32> {
    let mut out = particles.to_vec();

    let pc = params.particle_count as usize;
    let expected = pc.saturating_mul(FLIP_P2G_PARTICLE_FLOATS);
    let faces = params.faces_count();
    if params.dx <= EPS
        || pc == 0
        || params.grid_nx == 0
        || params.grid_ny == 0
        || params.grid_nz == 0
        || particles.len() != expected
        || faces_projected.len() != faces
        || faces_preprojection.len() != faces
    {
        return out;
    }

    let nx = params.grid_nx as usize;
    let ny = params.grid_ny as usize;
    let dx = params.dx;
    let uc = params.u_count();
    let vc = params.v_count();

    let inx = params.grid_nx as i32;
    let iny = params.grid_ny as i32;
    let inz = params.grid_nz as i32;

    let alpha = clamp_unit(params.flip_blend);

    for p in 0..pc {
        let base = p * FLIP_P2G_PARTICLE_FLOATS;
        if particles[base + 3] < 0.5 {
            continue;
        }
        let pos = Vec3::new(particles[base], particles[base + 1], particles[base + 2]);
        let vel = Vec3::new(
            particles[base + 4],
            particles[base + 5],
            particles[base + 6],
        );
        let g = Vec3::new(pos.x / dx, pos.y / dx, pos.z / dx);

        // x-velocity <- u-faces (nodes at (i, j + 0.5, k + 0.5), i in 0..=nx).
        let gu = gather_axis(
            faces_projected,
            faces_preprojection,
            g,
            Vec3::new(0.0, 0.5, 0.5),
            dx,
            inx,
            iny - 1,
            inz - 1,
            nx + 1,
            (nx + 1) * ny,
            0,
        );
        // y-velocity <- v-faces (nodes at (i + 0.5, j, k + 0.5), j in 0..=ny).
        let gv = gather_axis(
            faces_projected,
            faces_preprojection,
            g,
            Vec3::new(0.5, 0.0, 0.5),
            dx,
            inx - 1,
            iny,
            inz - 1,
            nx,
            nx * (ny + 1),
            uc,
        );
        // z-velocity <- w-faces (nodes at (i + 0.5, j + 0.5, k), k in 0..=nz).
        let gw = gather_axis(
            faces_projected,
            faces_preprojection,
            g,
            Vec3::new(0.5, 0.5, 0.0),
            dx,
            inx - 1,
            iny - 1,
            inz,
            nx,
            nx * ny,
            uc + vc,
        );

        let pic = Vec3::new(gu.pic, gv.pic, gw.pic);
        let delta = Vec3::new(gu.delta, gv.delta, gw.delta);
        let flip_vel = vel.add(delta);
        let new_vel = pic.scale(1.0 - alpha).add(flip_vel.scale(alpha));

        out[base + 4] = new_vel.x;
        out[base + 5] = new_vel.y;
        out[base + 6] = new_vel.z;

        if params.use_affine {
            let inv_d = APIC_INV_D_NUM / (dx * dx);
            let c0 = gu.row.scale(inv_d);
            let c1 = gv.row.scale(inv_d);
            let c2 = gw.row.scale(inv_d);
            out[base + 8] = c0.x;
            out[base + 9] = c0.y;
            out[base + 10] = c0.z;
            out[base + 12] = c1.x;
            out[base + 13] = c1.y;
            out[base + 14] = c1.z;
            out[base + 16] = c2.x;
            out[base + 17] = c2.y;
            out[base + 18] = c2.z;
        } else {
            out[base + 8] = 0.0;
            out[base + 9] = 0.0;
            out[base + 10] = 0.0;
            out[base + 12] = 0.0;
            out[base + 13] = 0.0;
            out[base + 14] = 0.0;
            out[base + 16] = 0.0;
            out[base + 17] = 0.0;
            out[base + 18] = 0.0;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::super::super::kernels::{DispatchDomain, WaterKernel};
    use super::super::flip_mac_faces_normalize_kernel::{
        dispatch_mac_faces_normalize, FacesNormalizeParams,
    };
    use super::super::flip_mac_p2g_kernel::{dispatch_mac_p2g, FlipP2GParams};
    use super::*;
    use alloc::vec;

    const ZERO_ROWS: [[f32; 3]; 3] = [[0.0; 3]; 3];

    fn params(
        nx: u32,
        ny: u32,
        nz: u32,
        pc: u32,
        dx: f32,
        blend: f32,
        affine: bool,
    ) -> FlipG2PParams {
        FlipG2PParams {
            grid_nx: nx,
            grid_ny: ny,
            grid_nz: nz,
            particle_count: pc,
            dx,
            flip_blend: blend,
            use_affine: affine,
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

    fn fn_params(nx: u32, ny: u32, nz: u32) -> FacesNormalizeParams {
        FacesNormalizeParams {
            grid_nx: nx,
            grid_ny: ny,
            grid_nz: nz,
        }
    }

    /// Builds one packed particle record (active) from position, velocity, and
    /// affine rows.
    fn particle(pos: [f32; 3], vel: [f32; 3], rows: [[f32; 3]; 3]) -> [f32; 20] {
        [
            pos[0], pos[1], pos[2], 1.0, vel[0], vel[1], vel[2], 0.0, rows[0][0], rows[0][1],
            rows[0][2], 0.0, rows[1][0], rows[1][1], rows[1][2], 0.0, rows[2][0], rows[2][1],
            rows[2][2], 0.0,
        ]
    }

    /// Deterministic, bounded packed-face field (integer arithmetic only, so no
    /// forbidden float floor); `k` shifts every entry by a constant.
    fn ramp_faces(n: usize, k: f32) -> Vec<f32> {
        let mut v: Vec<f32> = Vec::with_capacity(n);
        let mut i = 0usize;
        while i < n {
            let a = (i % 11) as f32;
            let b = (i % 5) as f32;
            v.push(0.08 * a - 0.04 * b + 0.25 + k);
            i += 1;
        }
        v
    }

    /// Uniform packed-face field: every `u`/`v`/`w` face carries its family's
    /// component of `uvw`.
    fn uniform_faces(nx: u32, ny: u32, nz: u32, uvw: [f32; 3]) -> Vec<f32> {
        let p = params(nx, ny, nz, 0, 1.0, 0.0, false);
        let uc = (nx as usize + 1) * ny as usize * nz as usize;
        let vc = nx as usize * (ny as usize + 1) * nz as usize;
        let faces = p.faces_count();
        let mut v: Vec<f32> = Vec::with_capacity(faces);
        let mut f = 0usize;
        while f < faces {
            let comp = if f < uc {
                uvw[0]
            } else if f < uc + vc {
                uvw[1]
            } else {
                uvw[2]
            };
            v.push(comp);
            f += 1;
        }
        v
    }

    /// Independent inline `G2P`: recomputes floor, fractional, trilinear weights,
    /// the `PIC`/`FLIP` blend, and the `APIC` rows with no shared helper, in the
    /// same corner order, so a match proves the twin's arithmetic rather than
    /// re-using its code.
    fn reference_g2p(
        particles: &[f32],
        projected: &[f32],
        preprojection: &[f32],
        p: FlipG2PParams,
    ) -> Vec<f32> {
        let mut out = particles.to_vec();
        let nx = p.grid_nx as usize;
        let ny = p.grid_ny as usize;
        let dx = p.dx;
        let uc = (nx + 1) * ny * p.grid_nz as usize;
        let vc = nx * (ny + 1) * p.grid_nz as usize;
        let inx = p.grid_nx as i32;
        let iny = p.grid_ny as i32;
        let inz = p.grid_nz as i32;

        let floor_ref = |x: f32| -> i32 {
            let t = x as i32;
            if (t as f32) > x {
                t - 1
            } else {
                t
            }
        };

        let alpha = clamp_unit(p.flip_blend);

        // (shift_x, shift_y, shift_z, max_i, max_j, max_k, stride_j, stride_k, off)
        let families: [(f32, f32, f32, i32, i32, i32, usize, usize, usize); 3] = [
            (
                0.0,
                0.5,
                0.5,
                inx,
                iny - 1,
                inz - 1,
                nx + 1,
                (nx + 1) * ny,
                0,
            ),
            (0.5, 0.0, 0.5, inx - 1, iny, inz - 1, nx, nx * (ny + 1), uc),
            (0.5, 0.5, 0.0, inx - 1, iny - 1, inz, nx, nx * ny, uc + vc),
        ];

        for pi in 0..p.particle_count as usize {
            let b = pi * 20;
            if particles[b + 3] < 0.5 {
                continue;
            }
            let g = [
                particles[b] / dx,
                particles[b + 1] / dx,
                particles[b + 2] / dx,
            ];
            let vel = [particles[b + 4], particles[b + 5], particles[b + 6]];

            let mut pic = [0.0f32; 3];
            let mut delta = [0.0f32; 3];
            let mut rows = [[0.0f32; 3]; 3];

            for (fam, &(hx, hy, hz, maxi, maxj, maxk, sj, sk, off)) in families.iter().enumerate() {
                let lat = [g[0] - hx, g[1] - hy, g[2] - hz];
                let bi = floor_ref(lat[0]);
                let bj = floor_ref(lat[1]);
                let bk = floor_ref(lat[2]);
                let fx = lat[0] - bi as f32;
                let fy = lat[1] - bj as f32;
                let fz = lat[2] - bk as f32;
                let wx = [1.0 - fx, fx];
                let wy = [1.0 - fy, fy];
                let wz = [1.0 - fz, fz];
                let mut cz = 0i32;
                while cz < 2 {
                    let gk = bk + cz;
                    let mut cy = 0i32;
                    while cy < 2 {
                        let gj = bj + cy;
                        let mut cx = 0i32;
                        while cx < 2 {
                            let gi = bi + cx;
                            if gi >= 0
                                && gi <= maxi
                                && gj >= 0
                                && gj <= maxj
                                && gk >= 0
                                && gk <= maxk
                            {
                                let weight = wx[cx as usize] * wy[cy as usize] * wz[cz as usize];
                                let flat = off + gi as usize + sj * gj as usize + sk * gk as usize;
                                let vp = projected[flat];
                                let vq = preprojection[flat];
                                pic[fam] += vp * weight;
                                delta[fam] += (vp - vq) * weight;
                                let node = [gi as f32 + hx, gj as f32 + hy, gk as f32 + hz];
                                let offv = [
                                    (node[0] - g[0]) * dx,
                                    (node[1] - g[1]) * dx,
                                    (node[2] - g[2]) * dx,
                                ];
                                rows[fam][0] += offv[0] * (vp * weight);
                                rows[fam][1] += offv[1] * (vp * weight);
                                rows[fam][2] += offv[2] * (vp * weight);
                            }
                            cx += 1;
                        }
                        cy += 1;
                    }
                    cz += 1;
                }
            }

            let flip = [vel[0] + delta[0], vel[1] + delta[1], vel[2] + delta[2]];
            out[b + 4] = pic[0] * (1.0 - alpha) + flip[0] * alpha;
            out[b + 5] = pic[1] * (1.0 - alpha) + flip[1] * alpha;
            out[b + 6] = pic[2] * (1.0 - alpha) + flip[2] * alpha;

            if p.use_affine {
                let inv_d = 3.0 / (dx * dx);
                out[b + 8] = rows[0][0] * inv_d;
                out[b + 9] = rows[0][1] * inv_d;
                out[b + 10] = rows[0][2] * inv_d;
                out[b + 12] = rows[1][0] * inv_d;
                out[b + 13] = rows[1][1] * inv_d;
                out[b + 14] = rows[1][2] * inv_d;
                out[b + 16] = rows[2][0] * inv_d;
                out[b + 17] = rows[2][1] * inv_d;
                out[b + 18] = rows[2][2] * inv_d;
            } else {
                out[b + 8] = 0.0;
                out[b + 9] = 0.0;
                out[b + 10] = 0.0;
                out[b + 12] = 0.0;
                out[b + 13] = 0.0;
                out[b + 14] = 0.0;
                out[b + 16] = 0.0;
                out[b + 17] = 0.0;
                out[b + 18] = 0.0;
            }
        }

        out
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let s = WATER_FLIP_MAC_G2P_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains(&alloc::format!(
            "fn {}",
            WaterKernel::FlipMacG2P.wesl_entry_point()
        )));
        assert!(s.contains("@workgroup_size(64, 1, 1)"));
        assert!(s.contains("particles"));
        assert!(s.contains("faces_projected"));
        assert!(s.contains("faces_preprojection"));
        assert!(s.contains("sim_params"));

        let desc = WaterKernel::FlipMacG2P.descriptor();
        assert_eq!(desc.layout.storage_buffers, 3);
        assert_eq!(desc.layout.uniform_buffers, 1);
        assert_eq!(desc.layout.storage_textures, 0);
        assert_eq!(desc.layout.sampled_textures, 0);
        assert_eq!(desc.workgroup.x, 64);
        assert_eq!(desc.workgroup.y, 1);
        assert_eq!(desc.workgroup.z, 1);
        assert_eq!(desc.domain, DispatchDomain::Particle);
    }

    #[test]
    fn single_interior_particle_recovers_velocity_any_blend() {
        // A lone deep-interior particle splatted by P2G then normalized leaves a
        // face field that is a partition-of-unity reconstruction of its own
        // velocity; with projected == preprojection (no projection) G2P recovers
        // that velocity for any blend, because PIC gathers it exactly and the
        // FLIP delta is zero.
        let (nx, ny, nz) = (5u32, 5u32, 5u32);
        let dx = 0.4f32;
        let vel = [0.37f32, -0.22, 0.61];
        let pos = [2.0 * dx, 2.5 * dx, 3.0 * dx];
        let buf = particle(pos, vel, ZERO_ROWS);
        let scatter = dispatch_mac_p2g(&buf, p2g_params(nx, ny, nz, 1, dx, false));
        let faces = dispatch_mac_faces_normalize(&scatter, fn_params(nx, ny, nz));

        for &blend in &[0.0f32, 0.5, 1.0] {
            let out = dispatch_mac_g2p(
                &buf,
                &faces,
                &faces,
                params(nx, ny, nz, 1, dx, blend, false),
            );
            let got = [out[4], out[5], out[6]];
            for (axis, (&g, &e)) in got.iter().zip(vel.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 1.0e-4,
                    "blend {blend} axis {axis}: recovered {g}, expected {e}"
                );
            }
        }
    }

    #[test]
    fn uniform_field_is_gathered_exactly() {
        // A uniform grid field must gather back to the same uniform velocity at
        // any interior particle (trilinear partition of unity).
        let (nx, ny, nz) = (4u32, 4u32, 4u32);
        let dx = 0.5f32;
        let uvw = [0.8f32, -0.3, 0.45];
        let faces = uniform_faces(nx, ny, nz, uvw);
        let pos = [2.0 * dx, 2.0 * dx, 2.0 * dx];
        let buf = particle(pos, [0.0, 0.0, 0.0], ZERO_ROWS);
        // PIC (blend 0) with projected == preprojection: pure gather, no delta.
        let out = dispatch_mac_g2p(&buf, &faces, &faces, params(nx, ny, nz, 1, dx, 0.0, false));
        for (axis, &e) in uvw.iter().enumerate() {
            assert!(
                (out[4 + axis] - e).abs() < 1.0e-4,
                "axis {axis}: gathered {}, expected uniform {e}",
                out[4 + axis]
            );
        }
    }

    #[test]
    fn matches_independent_inline_g2p() {
        let (nx, ny, nz) = (5u32, 4u32, 4u32);
        let dx = 0.3f32;
        let p = params(nx, ny, nz, 3, dx, 0.37, true);
        let faces = p.faces_count();
        let projected = ramp_faces(faces, 0.0);
        let preprojection = ramp_faces(faces, 0.19);

        let seeds = [
            ([1.6f32, 1.4, 1.5], [0.2f32, -0.6, 0.4]),
            ([2.3, 1.7, 1.8], [-0.5, 0.3, 0.7]),
            ([1.9, 2.1, 2.3], [0.8, 0.1, -0.2]),
        ];
        let mut buf: Vec<f32> = Vec::new();
        for (pos, vel) in seeds {
            buf.extend_from_slice(&particle(
                [pos[0] * dx, pos[1] * dx, pos[2] * dx],
                vel,
                ZERO_ROWS,
            ));
        }

        let twin = dispatch_mac_g2p(&buf, &projected, &preprojection, p);
        let reference = reference_g2p(&buf, &projected, &preprojection, p);
        assert_eq!(twin.len(), reference.len());
        let mut changed = 0usize;
        for (i, (a, r)) in twin.iter().zip(reference.iter()).enumerate() {
            assert_eq!(a.to_bits(), r.to_bits(), "lane {i} diverged from inline");
            if a.to_bits() != buf[i].to_bits() {
                changed += 1;
            }
        }
        assert!(
            changed > 0,
            "G2P must rewrite particle lanes (guard against a vacuous no-op match)"
        );
    }

    #[test]
    fn flip_blend_endpoints_separate_pic_and_particle_velocity() {
        // With projected != preprojection, PIC (blend 0) ignores the particle's
        // stored velocity, while FLIP (blend 1) adds the same grid delta to each
        // particle's own velocity. Two particles at the same spot with different
        // velocities expose both facts without re-deriving the gather.
        let (nx, ny, nz) = (5u32, 5u32, 5u32);
        let dx = 0.3f32;
        let faces = params(nx, ny, nz, 1, dx, 0.0, false).faces_count();
        let projected = ramp_faces(faces, 0.0);
        let preprojection = ramp_faces(faces, 0.13);
        let pos = [2.0 * dx, 2.0 * dx, 2.0 * dx];
        let v1 = [0.5f32, -0.4, 0.2];
        let v2 = [-0.3f32, 0.6, 0.9];
        let b1 = particle(pos, v1, ZERO_ROWS);
        let b2 = particle(pos, v2, ZERO_ROWS);

        let pic1 = dispatch_mac_g2p(
            &b1,
            &projected,
            &preprojection,
            params(nx, ny, nz, 1, dx, 0.0, false),
        );
        let pic2 = dispatch_mac_g2p(
            &b2,
            &projected,
            &preprojection,
            params(nx, ny, nz, 1, dx, 0.0, false),
        );
        for axis in 0..3 {
            assert_eq!(
                pic1[4 + axis].to_bits(),
                pic2[4 + axis].to_bits(),
                "PIC gather must be independent of the stored velocity (axis {axis})"
            );
        }

        let flip1 = dispatch_mac_g2p(
            &b1,
            &projected,
            &preprojection,
            params(nx, ny, nz, 1, dx, 1.0, false),
        );
        let flip2 = dispatch_mac_g2p(
            &b2,
            &projected,
            &preprojection,
            params(nx, ny, nz, 1, dx, 1.0, false),
        );
        let mut delta_nonzero = false;
        for axis in 0..3 {
            let d1 = flip1[4 + axis] - v1[axis];
            let d2 = flip2[4 + axis] - v2[axis];
            assert!(
                (d1 - d2).abs() < 1.0e-5,
                "FLIP delta must be particle-independent (axis {axis}): {d1} vs {d2}"
            );
            if d1.abs() > 1.0e-6 {
                delta_nonzero = true;
            }
        }
        assert!(
            delta_nonzero,
            "projected != preprojection must yield a non-zero FLIP delta"
        );
    }

    #[test]
    fn affine_flag_controls_affine_rows() {
        let (nx, ny, nz) = (5u32, 5u32, 5u32);
        let dx = 0.4f32;
        let faces = params(nx, ny, nz, 1, dx, 0.5, true).faces_count();
        let projected = ramp_faces(faces, 0.0);
        let preprojection = ramp_faces(faces, 0.0);
        let pos = [2.3 * dx, 1.8 * dx, 2.6 * dx];
        let buf = particle(pos, [0.1, 0.2, 0.3], ZERO_ROWS);
        let affine_lanes = [8usize, 9, 10, 12, 13, 14, 16, 17, 18];

        let off = dispatch_mac_g2p(
            &buf,
            &projected,
            &preprojection,
            params(nx, ny, nz, 1, dx, 0.5, false),
        );
        for &idx in &affine_lanes {
            assert_eq!(
                off[idx].to_bits(),
                0.0f32.to_bits(),
                "affine-off must leave affine lane {idx} zero"
            );
        }

        let on = dispatch_mac_g2p(
            &buf,
            &projected,
            &preprojection,
            params(nx, ny, nz, 1, dx, 0.5, true),
        );
        let mut any = false;
        for &idx in &affine_lanes {
            if on[idx].to_bits() != 0.0f32.to_bits() {
                any = true;
            }
        }
        assert!(
            any,
            "affine reconstruction must populate at least one affine row"
        );
    }

    #[test]
    fn inactive_and_degenerate_requests_pass_through() {
        let (nx, ny, nz) = (4u32, 4u32, 4u32);
        let dx = 0.5f32;
        let faces = params(nx, ny, nz, 1, dx, 0.5, true).faces_count();
        let projected = ramp_faces(faces, 0.0);
        let preprojection = ramp_faces(faces, 0.07);

        // Inactive particle: untouched.
        let mut dead = particle(
            [2.0 * dx, 2.0 * dx, 2.0 * dx],
            [0.9, 0.9, 0.9],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        );
        dead[3] = 0.0;
        let out = dispatch_mac_g2p(
            &dead,
            &projected,
            &preprojection,
            params(nx, ny, nz, 1, dx, 0.5, true),
        );
        for (o, d) in out.iter().zip(dead.iter()) {
            assert_eq!(o.to_bits(), d.to_bits());
        }

        let live = particle([2.0 * dx, 2.0 * dx, 2.0 * dx], [0.9, 0.9, 0.9], ZERO_ROWS);

        // Non-positive dx: pass-through.
        let z = dispatch_mac_g2p(
            &live,
            &projected,
            &preprojection,
            params(nx, ny, nz, 1, 0.0, 0.5, true),
        );
        for (o, d) in z.iter().zip(live.iter()) {
            assert_eq!(o.to_bits(), d.to_bits());
        }

        // Mis-sized particle buffer: pass-through, no panic.
        let short = &live[..19];
        let z2 = dispatch_mac_g2p(
            short,
            &projected,
            &preprojection,
            params(nx, ny, nz, 1, dx, 0.5, true),
        );
        assert_eq!(z2.len(), 19);
        for (o, d) in z2.iter().zip(short.iter()) {
            assert_eq!(o.to_bits(), d.to_bits());
        }

        // Wrong face-buffer length: pass-through.
        let bad = vec![0.0f32; faces - 1];
        let z3 = dispatch_mac_g2p(
            &live,
            &bad,
            &preprojection,
            params(nx, ny, nz, 1, dx, 0.5, true),
        );
        for (o, d) in z3.iter().zip(live.iter()) {
            assert_eq!(o.to_bits(), d.to_bits());
        }

        // Zero-extent grid: pass-through (clone of the input pool).
        let z4 = dispatch_mac_g2p(&live, &[], &[], params(0, 0, 0, 1, dx, 0.5, true));
        for (o, d) in z4.iter().zip(live.iter()) {
            assert_eq!(o.to_bits(), d.to_bits());
        }
    }
}
