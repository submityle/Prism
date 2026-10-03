//! `FLIP`/`APIC` `MAC` particle-to-grid (`P2G`) scatter compute kernel: the
//! `WESL` `water_flip_mac_p2g` shader plus its `CPU` twin.
//!
//! `P2G` is the first half of the staggered `MAC` transfer. Every particle
//! splats its (optionally affine, `APIC`) velocity onto the eight surrounding
//! face nodes of each staggered family with trilinear weights, accumulating a
//! momentum/mass pair per face:
//!
//! ```text
//! face_scatter[face * 2 + 0] += w * v_comp   (weighted momentum)
//! face_scatter[face * 2 + 1] += w            (weighted mass)
//! ```
//!
//! The companion normalize pass
//! ([`flip_mac_faces_normalize_kernel`](super)) later divides momentum by mass
//! to recover `v = sum(w * v) / sum(w)`. Because the trilinear weights are a
//! partition of unity ([`flip::trilinear_weights`](super::super::flip)), a lone
//! interior particle recovers its own velocity exactly and a uniform velocity
//! field is reproduced exactly. The `x` faces deposit the particle `x`-velocity
//! (and `APIC` affine row 0 for the affine term), the `y` faces the `y`-velocity
//! (row 1), the `z` faces the `z`-velocity (row 2). The affine velocity at a
//! node is `v_p + C_p (x_node - x_p)`, matching
//! [`flip::apic_velocity`](super::super::flip); a zero affine matrix collapses
//! to plain `PIC` splatting.
//!
//! Scatter is a many-writers-one-slot operation, so the real `GPU` accumulators
//! are atomics. Metal has no float atomics, so the shader keeps momentum and
//! mass as signed fixed-point packed into `atomic<u32>`
//! (`value_fixed = i32(value * SCATTER_SCALE)` added with `atomicAdd`; two's
//! complement addition is identical for signed and unsigned). This twin models
//! the un-quantized `f32` momentum/mass that fixed-point sum approximates, so
//! it is the golden reference for the deposited quantity rather than a
//! bit-for-bit mirror of the device's fixed-point rounding.
//!
//! [`WATER_FLIP_MAC_P2G_WESL`] is the shader (entry point
//! `water_flip_mac_p2g`); [`dispatch_mac_p2g`] is its twin. The shader honours
//! the [`WaterKernel::FlipMacP2G`](super::super::kernels::WaterKernel)
//! descriptor — three storage buffers, one uniform block, dispatched over the
//! particle pool (`Particle` domain, `@workgroup_size(64, 1, 1)`). Binding 2,
//! the normalized face buffer, is reserved for the shared face-normalize pass
//! and is untouched here, so the shader declares only the two storage buffers it
//! uses (`particles`, `face_scatter`) plus `sim_params`.
//!
//! Packed face layout matches the divergence/projection stages: a single buffer
//! holds the three families back to back, `u` then `v` then `w`, each i-fastest
//! row-major over its own lattice. Particles are a flat `f32` array, five
//! `vec4` lanes (20 `f32`) each: position+active, velocity, and the three
//! `APIC` affine rows. Only `+ - * /` and comparisons appear.

use alloc::vec::Vec;

use super::super::flip::trilinear_weights;
use super::super::{Vec3, EPS};

/// `WESL` source of the `MAC` `P2G` scatter compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree, because
/// `prism_render_architecture` is a zero-dependency crate.
pub const WATER_FLIP_MAC_P2G_WESL: &str = include_str!("water_flip_mac_p2g.wesl");

/// Number of `f32` lanes in one packed `FLIP`/`APIC` particle record: five
/// `vec4` lanes (position+active, velocity, three affine rows), `5 * 4 = 20`.
pub const FLIP_P2G_PARTICLE_FLOATS: usize = 20;

/// Scatter slots per face: a `[momentum, mass]` pair.
const SCATTER_SLOTS: usize = 2;

/// Host mirror of the shader's `FlipP2GParams` uniform block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipP2GParams {
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
    /// Whether the affine (`APIC`) velocity field is splatted.
    pub use_affine: bool,
}

impl FlipP2GParams {
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
    fn faces_count(&self) -> usize {
        self.u_count()
            .saturating_add(self.v_count())
            .saturating_add(self.w_count())
    }

    /// Length of the `[momentum, mass]` scatter accumulator, `faces * 2`.
    #[must_use]
    pub fn scatter_len(&self) -> usize {
        self.faces_count().saturating_mul(SCATTER_SLOTS)
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

/// Scatter one particle onto the eight surrounding nodes of a single staggered
/// face family.
#[expect(
    clippy::too_many_arguments,
    reason = "The staggered P2G scatter is parameterised per face family by its lattice position, node half-offset, deposited velocity component, affine row, and packed-block strides; bundling these inherent per-family quantities into a struct would only rename them."
)]
fn scatter_family(
    scatter: &mut [f32],
    lattice: Vec3,
    node_half: Vec3,
    g: Vec3,
    base_comp: f32,
    affine_row: Vec3,
    use_affine: bool,
    dx: f32,
    max_i: i32,
    max_j: i32,
    max_k: i32,
    stride_j: usize,
    stride_k: usize,
    off: usize,
) {
    let bi = floor_i32(lattice.x);
    let bj = floor_i32(lattice.y);
    let bk = floor_i32(lattice.z);
    let fx = lattice.x - bi as f32;
    let fy = lattice.y - bj as f32;
    let fz = lattice.z - bk as f32;
    let w = trilinear_weights(fx, fy, fz);

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
                    let weight = w[(cx + 2 * cy + 4 * cz) as usize];
                    let node = Vec3::new(
                        gi as f32 + node_half.x,
                        gj as f32 + node_half.y,
                        gk as f32 + node_half.z,
                    );
                    let offset = node.sub(g).scale(dx);
                    let vel_comp = if use_affine {
                        base_comp + affine_row.dot(offset)
                    } else {
                        base_comp
                    };
                    let face_flat =
                        off + gi as usize + stride_j * gj as usize + stride_k * gk as usize;
                    scatter[face_flat * SCATTER_SLOTS] += weight * vel_comp;
                    scatter[face_flat * SCATTER_SLOTS + 1] += weight;
                }
                cx += 1;
            }
            cy += 1;
        }
        cz += 1;
    }
}

/// `CPU` twin of the `water_flip_mac_p2g` shader.
///
/// `particles` is the packed `FLIP`/`APIC` pool ([`FLIP_P2G_PARTICLE_FLOATS`]
/// per particle). Returns the `[momentum, mass]` scatter accumulator, two `f32`
/// per packed face, in `u | v | w` block order. Inactive particles (lane 3
/// `< 0.5`) are skipped. A degenerate spacing, an empty pool, or a mis-sized
/// particle buffer yields an all-zero accumulator (no deposit); an empty grid
/// yields an empty buffer.
#[must_use]
pub fn dispatch_mac_p2g(particles: &[f32], params: FlipP2GParams) -> Vec<f32> {
    let scatter_len = params.scatter_len();
    if scatter_len == 0 {
        return Vec::new();
    }
    let mut scatter = alloc::vec![0.0f32; scatter_len];

    let pc = params.particle_count as usize;
    let expected = pc.saturating_mul(FLIP_P2G_PARTICLE_FLOATS);
    if params.dx <= EPS || pc == 0 || particles.len() != expected {
        return scatter;
    }

    let nx = params.grid_nx as usize;
    let ny = params.grid_ny as usize;
    let dx = params.dx;
    let uc = params.u_count();
    let vc = params.v_count();

    let inx = params.grid_nx as i32;
    let iny = params.grid_ny as i32;
    let inz = params.grid_nz as i32;

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
        let c0 = Vec3::new(
            particles[base + 8],
            particles[base + 9],
            particles[base + 10],
        );
        let c1 = Vec3::new(
            particles[base + 12],
            particles[base + 13],
            particles[base + 14],
        );
        let c2 = Vec3::new(
            particles[base + 16],
            particles[base + 17],
            particles[base + 18],
        );
        let g = Vec3::new(pos.x / dx, pos.y / dx, pos.z / dx);

        // u (x) faces: node at (i, j + 0.5, k + 0.5); deposit x-velocity.
        scatter_family(
            &mut scatter,
            Vec3::new(g.x, g.y - 0.5, g.z - 0.5),
            Vec3::new(0.0, 0.5, 0.5),
            g,
            vel.x,
            c0,
            params.use_affine,
            dx,
            inx,
            iny - 1,
            inz - 1,
            nx + 1,
            (nx + 1) * ny,
            0,
        );

        // v (y) faces: node at (i + 0.5, j, k + 0.5); deposit y-velocity.
        scatter_family(
            &mut scatter,
            Vec3::new(g.x - 0.5, g.y, g.z - 0.5),
            Vec3::new(0.5, 0.0, 0.5),
            g,
            vel.y,
            c1,
            params.use_affine,
            dx,
            inx - 1,
            iny,
            inz - 1,
            nx,
            nx * (ny + 1),
            uc,
        );

        // w (z) faces: node at (i + 0.5, j + 0.5, k); deposit z-velocity.
        scatter_family(
            &mut scatter,
            Vec3::new(g.x - 0.5, g.y - 0.5, g.z),
            Vec3::new(0.5, 0.5, 0.0),
            g,
            vel.z,
            c2,
            params.use_affine,
            dx,
            inx - 1,
            iny - 1,
            inz,
            nx,
            nx * ny,
            uc + vc,
        );
    }

    scatter
}

#[cfg(test)]
mod tests {
    use super::super::super::kernels::{DispatchDomain, WaterKernel};
    use super::*;
    use alloc::vec;

    fn params(nx: u32, ny: u32, nz: u32, pc: u32, dx: f32, affine: bool) -> FlipP2GParams {
        FlipP2GParams {
            grid_nx: nx,
            grid_ny: ny,
            grid_nz: nz,
            particle_count: pc,
            dx,
            use_affine: affine,
        }
    }

    /// Builds one packed particle record from position, velocity, and affine
    /// rows (active).
    fn particle(pos: [f32; 3], vel: [f32; 3], rows: [[f32; 3]; 3]) -> [f32; 20] {
        [
            pos[0], pos[1], pos[2], 1.0, vel[0], vel[1], vel[2], 0.0, rows[0][0], rows[0][1],
            rows[0][2], 0.0, rows[1][0], rows[1][1], rows[1][2], 0.0, rows[2][0], rows[2][1],
            rows[2][2], 0.0,
        ]
    }

    const ZERO_ROWS: [[f32; 3]; 3] = [[0.0; 3]; 3];

    /// Independent inline reference: recomputes floor, fractional, trilinear
    /// weights, and the APIC component with no shared golden helper, in the
    /// same corner order, so a match proves the twin's arithmetic rather than
    /// re-using its code.
    fn reference_scatter(particles: &[f32], p: FlipP2GParams) -> Vec<f32> {
        let faces = (p.grid_nx as usize + 1) * p.grid_ny as usize * p.grid_nz as usize
            + p.grid_nx as usize * (p.grid_ny as usize + 1) * p.grid_nz as usize
            + p.grid_nx as usize * p.grid_ny as usize * (p.grid_nz as usize + 1);
        let mut out = vec![0.0f32; faces * 2];
        let nx = p.grid_nx as usize;
        let ny = p.grid_ny as usize;
        let uc = (nx + 1) * ny * p.grid_nz as usize;
        let vc = nx * (ny + 1) * p.grid_nz as usize;
        let dx = p.dx;

        let floor_ref = |x: f32| -> i32 {
            let t = x as i32;
            if (t as f32) > x {
                t - 1
            } else {
                t
            }
        };

        let families: [(f32, f32, f32, [f32; 3], usize); 3] = [
            (0.0, 0.5, 0.5, [0.0, 0.0, 0.0], 0),
            (0.5, 0.0, 0.5, [0.0, 0.0, 0.0], uc),
            (0.5, 0.5, 0.0, [0.0, 0.0, 0.0], uc + vc),
        ];
        // Only the node_half offsets and block offset are reused above; the
        // per-family index limits/strides and deposited component are inlined.
        let limits: [(i32, i32, i32, usize, usize); 3] = [
            (
                p.grid_nx as i32,
                p.grid_ny as i32 - 1,
                p.grid_nz as i32 - 1,
                nx + 1,
                (nx + 1) * ny,
            ),
            (
                p.grid_nx as i32 - 1,
                p.grid_ny as i32,
                p.grid_nz as i32 - 1,
                nx,
                nx * (ny + 1),
            ),
            (
                p.grid_nx as i32 - 1,
                p.grid_ny as i32 - 1,
                p.grid_nz as i32,
                nx,
                nx * ny,
            ),
        ];

        for pi in 0..p.particle_count as usize {
            let b = pi * 20;
            if particles[b + 3] < 0.5 {
                continue;
            }
            let pos = [particles[b], particles[b + 1], particles[b + 2]];
            let vel = [particles[b + 4], particles[b + 5], particles[b + 6]];
            let rows = [
                [particles[b + 8], particles[b + 9], particles[b + 10]],
                [particles[b + 12], particles[b + 13], particles[b + 14]],
                [particles[b + 16], particles[b + 17], particles[b + 18]],
            ];
            let g = [pos[0] / dx, pos[1] / dx, pos[2] / dx];

            for fam in 0..3 {
                let (hx, hy, hz, _unused, off) = families[fam];
                let (max_i, max_j, max_k, stride_j, stride_k) = limits[fam];
                let lat = [g[0] - hx, g[1] - hy, g[2] - hz];
                // lat recovers the staggered coordinate: subtract the family's
                // node half-offset on each axis so integer lattice coordinates
                // land exactly on the staggered face nodes (lattice = g -
                // node_half), mirroring the shader and twin exactly.
                let bi = floor_ref(lat[0]);
                let bj = floor_ref(lat[1]);
                let bk = floor_ref(lat[2]);
                let fx = lat[0] - bi as f32;
                let fy = lat[1] - bj as f32;
                let fz = lat[2] - bk as f32;
                let wx = [1.0 - fx, fx];
                let wy = [1.0 - fy, fy];
                let wz = [1.0 - fz, fz];
                let base_comp = vel[fam];
                let row = rows[fam];
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
                                && gi <= max_i
                                && gj >= 0
                                && gj <= max_j
                                && gk >= 0
                                && gk <= max_k
                            {
                                let weight = wx[cx as usize] * wy[cy as usize] * wz[cz as usize];
                                let node = [gi as f32 + hx, gj as f32 + hy, gk as f32 + hz];
                                let offv = [
                                    (node[0] - g[0]) * dx,
                                    (node[1] - g[1]) * dx,
                                    (node[2] - g[2]) * dx,
                                ];
                                let vel_comp = if p.use_affine {
                                    // Group the affine dot product before adding
                                    // the base component, matching the shader's
                                    // `base_comp + dot(affine_row, offset)` so
                                    // the summation rounds identically.
                                    base_comp
                                        + (row[0] * offv[0] + row[1] * offv[1] + row[2] * offv[2])
                                } else {
                                    base_comp
                                };
                                let flat = off
                                    + gi as usize
                                    + stride_j * gj as usize
                                    + stride_k * gk as usize;
                                out[flat * 2] += weight * vel_comp;
                                out[flat * 2 + 1] += weight;
                            }
                            cx += 1;
                        }
                        cy += 1;
                    }
                    cz += 1;
                }
            }
        }
        out
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let s = WATER_FLIP_MAC_P2G_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains(&format!(
            "fn {}",
            WaterKernel::FlipMacP2G.wesl_entry_point()
        )));
        assert!(s.contains("@workgroup_size(64, 1, 1)"));
        assert!(s.contains("particles"));
        assert!(s.contains("face_scatter"));
        assert!(s.contains("sim_params"));

        let desc = WaterKernel::FlipMacP2G.descriptor();
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
    fn matches_independent_inline_scatter() {
        let (nx, ny, nz) = (4u32, 3u32, 3u32);
        // A handful of interior particles with varied velocities and affine.
        let mut buf: Vec<f32> = Vec::new();
        let seeds = [
            ([1.3f32, 1.1, 1.7], [0.5f32, -0.3, 0.2]),
            ([2.6, 0.8, 1.2], [-0.4, 0.6, -0.1]),
            ([0.9, 1.9, 0.6], [0.1, 0.2, 0.9]),
            ([3.1, 1.4, 2.2], [-0.7, -0.2, 0.4]),
        ];
        for (pos, vel) in seeds {
            let rows = [[0.2, -0.1, 0.3], [0.0, 0.4, -0.2], [-0.3, 0.1, 0.5]];
            buf.extend_from_slice(&particle(pos, vel, rows));
        }
        let p = params(nx, ny, nz, seeds.len() as u32, 0.5, true);
        let twin = dispatch_mac_p2g(&buf, p);
        let reference = reference_scatter(&buf, p);
        assert_eq!(twin.len(), reference.len());
        let mut any = false;
        for (c, (&t, &r)) in twin.iter().zip(reference.iter()).enumerate() {
            assert_eq!(t.to_bits(), r.to_bits(), "scatter mismatch at slot {c}");
            if t.to_bits() != 0.0f32.to_bits() {
                any = true;
            }
        }
        assert!(any, "scatter must deposit something (non-vacuous)");
    }

    #[test]
    fn single_interior_particle_recovers_its_velocity() {
        let (nx, ny, nz) = (4u32, 4u32, 4u32);
        let dx = 0.5f32;
        let vel = [0.7f32, -0.4, 0.25];
        // Place the particle at a cell centre well inside the domain so every
        // family's eight nodes are valid.
        let pos = [2.0 * dx, 2.0 * dx, 2.0 * dx];
        let buf = particle(pos, vel, ZERO_ROWS);
        let p = params(nx, ny, nz, 1, dx, false);
        let scatter = dispatch_mac_p2g(&buf, p);

        // Every face that received mass must read back exactly the matching
        // velocity component (momentum / mass), since one particle's weighted
        // average collapses to its own velocity.
        let uc = (nx as usize + 1) * ny as usize * nz as usize;
        let vc = nx as usize * (ny as usize + 1) * nz as usize;
        let faces = scatter.len() / 2;
        let mut touched = [0u32; 3];
        for f in 0..faces {
            let mass = scatter[f * 2 + 1];
            if mass > 1.0e-4 {
                let v = scatter[f * 2] / mass;
                let comp = if f < uc {
                    vel[0]
                } else if f < uc + vc {
                    vel[1]
                } else {
                    vel[2]
                };
                assert!(
                    (v - comp).abs() < 1.0e-4,
                    "face {f} recovered {v}, expected {comp}"
                );
                let fam = if f < uc {
                    0
                } else if f < uc + vc {
                    1
                } else {
                    2
                };
                touched[fam] += 1;
            }
        }
        assert!(
            touched[0] > 0 && touched[1] > 0 && touched[2] > 0,
            "each family must receive a deposit: {touched:?}"
        );
    }

    #[test]
    fn interior_particles_conserve_mass_per_family() {
        let (nx, ny, nz) = (5u32, 5u32, 5u32);
        let dx = 0.4f32;
        // Three particles placed so all eight nodes of every family land in
        // range (deep interior), making each family's total weight exactly 1.
        let seeds = [[2.0f32, 2.0, 2.0], [2.5, 2.5, 2.5], [1.5, 3.0, 2.0]];
        let mut buf: Vec<f32> = Vec::new();
        for c in seeds {
            buf.extend_from_slice(&particle(
                [c[0] * dx, c[1] * dx, c[2] * dx],
                [0.3, -0.2, 0.1],
                ZERO_ROWS,
            ));
        }
        let p = params(nx, ny, nz, seeds.len() as u32, dx, false);
        let scatter = dispatch_mac_p2g(&buf, p);

        let uc = (nx as usize + 1) * ny as usize * nz as usize;
        let vc = nx as usize * (ny as usize + 1) * nz as usize;
        let faces = scatter.len() / 2;
        let (mut mu, mut mv, mut mw) = (0.0f32, 0.0f32, 0.0f32);
        for f in 0..faces {
            let mass = scatter[f * 2 + 1];
            if f < uc {
                mu += mass;
            } else if f < uc + vc {
                mv += mass;
            } else {
                mw += mass;
            }
        }
        let n = seeds.len() as f32;
        assert!((mu - n).abs() < 1.0e-4, "u mass {mu} != {n}");
        assert!((mv - n).abs() < 1.0e-4, "v mass {mv} != {n}");
        assert!((mw - n).abs() < 1.0e-4, "w mass {mw} != {n}");
    }

    #[test]
    fn affine_term_changes_the_deposit() {
        let (nx, ny, nz) = (4u32, 4u32, 4u32);
        let dx = 0.5f32;
        let pos = [1.7f32 * dx, 2.3 * dx, 1.4 * dx];
        let vel = [0.2f32, 0.5, -0.3];
        let rows = [[0.9, -0.4, 0.2], [0.1, 0.7, -0.5], [-0.6, 0.3, 0.8]];
        let buf = particle(pos, vel, rows);
        let plain = dispatch_mac_p2g(&buf, params(nx, ny, nz, 1, dx, false));
        let affine = dispatch_mac_p2g(&buf, params(nx, ny, nz, 1, dx, true));
        assert_eq!(plain.len(), affine.len());

        // Mass is identical (weights unchanged); momentum must differ.
        let mut momentum_differs = false;
        let faces = plain.len() / 2;
        for f in 0..faces {
            assert_eq!(
                plain[f * 2 + 1].to_bits(),
                affine[f * 2 + 1].to_bits(),
                "affine must not change mass at face {f}"
            );
            if plain[f * 2].to_bits() != affine[f * 2].to_bits() {
                momentum_differs = true;
            }
        }
        assert!(momentum_differs, "affine term must alter the momentum");
    }

    #[test]
    fn inactive_particles_deposit_nothing() {
        let (nx, ny, nz) = (4u32, 4u32, 4u32);
        let dx = 0.5f32;
        let mut dead = particle([2.0 * dx, 2.0 * dx, 2.0 * dx], [1.0, 1.0, 1.0], ZERO_ROWS);
        dead[3] = 0.0; // inactive
        let scatter = dispatch_mac_p2g(&dead, params(nx, ny, nz, 1, dx, false));
        for s in scatter {
            assert_eq!(s.to_bits(), 0.0f32.to_bits());
        }
    }

    #[test]
    fn degenerate_requests_pass_through() {
        let (nx, ny, nz) = (3u32, 3u32, 3u32);
        let buf = particle([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], ZERO_ROWS);

        // Non-positive dx: zeroed accumulator of the right length.
        let z = dispatch_mac_p2g(&buf, params(nx, ny, nz, 1, 0.0, false));
        assert_eq!(z.len(), params(nx, ny, nz, 1, 0.0, false).scatter_len());
        for s in &z {
            assert_eq!(s.to_bits(), 0.0f32.to_bits());
        }

        // Mis-sized particle buffer: zeroed accumulator, no panic.
        let short = &buf[..19];
        let z2 = dispatch_mac_p2g(short, params(nx, ny, nz, 1, 0.5, false));
        assert_eq!(z2.len(), params(nx, ny, nz, 1, 0.5, false).scatter_len());
        for s in &z2 {
            assert_eq!(s.to_bits(), 0.0f32.to_bits());
        }

        // Zero particle count: zeroed accumulator.
        let z3 = dispatch_mac_p2g(&[], params(nx, ny, nz, 0, 0.5, false));
        for s in &z3 {
            assert_eq!(s.to_bits(), 0.0f32.to_bits());
        }

        // Empty grid: empty buffer.
        assert!(dispatch_mac_p2g(&[], params(0, 0, 0, 0, 0.5, false)).is_empty());
    }
}
