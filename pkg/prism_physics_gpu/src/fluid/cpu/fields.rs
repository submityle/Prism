//! The `CPU` golden staggered-field twin of the `GPU` transfer buffers.
//!
//! [`GoldenGrid`] holds the same three concatenated `[u | v | w]` arrays the
//! device kernel writes: a fixed-point momentum accumulator, a fixed-point
//! weight accumulator, and the normalised face-velocity field. The particle
//! scatter mirrors the `WGSL` kernel *exactly* — the same trilinear weights are
//! quantised with [`quantise`](crate::fluid::quantize::quantise) and summed as
//! `i32`, so the two engines accumulate bit-identical integers regardless of
//! particle order. Only the final `momentum / weight` division after
//! de-quantisation is floating point, which the parity test bounds with a tight
//! tolerance.
//!
//! # Provenance
//!
//! The staggered `MAC` field layout and trilinear splat/gather follow Harlow
//! and Welch 1965 and Bridson; the fixed-point atomic accumulation is a
//! standard `GPU` scatter technique. This module contains no Unreal Engine
//! source or derived code.

use glam::{Mat3, Vec3};

use super::extrapolate::{extrapolate_axis, AxisDims};
use super::stencil::{axis_stencil, trilinear_nodes};
use crate::fluid::grid::GridDims;
use crate::fluid::quantize::{dequantise, quantise, MOMENTUM_SCALE, WEIGHT_SCALE};

/// Per-axis geometry of one staggered face field inside the concatenated array.
#[derive(Clone, Copy)]
struct AxisGeom {
    /// Cell-space offset of the face samples from the cell corner.
    off: Vec3,
    /// Node counts of the field along each axis.
    dims: (usize, usize, usize),
    /// Flat offset of this field inside the concatenated `[u | v | w]` array.
    base: usize,
}

impl AxisGeom {
    #[inline]
    fn stride_y(&self) -> usize {
        self.dims.0
    }
    #[inline]
    fn stride_z(&self) -> usize {
        self.dims.0 * self.dims.1
    }
}

/// The `CPU` golden staggered `MAC` velocity field.
#[derive(Clone, Debug, PartialEq)]
pub struct GoldenGrid {
    dims: GridDims,
    /// Fixed-point momentum accumulator, concatenated `[u | v | w]`.
    momentum: Vec<i32>,
    /// Fixed-point weight accumulator, concatenated `[u | v | w]`.
    weight: Vec<i32>,
    /// Normalised face velocities, concatenated `[u | v | w]`.
    velocity: Vec<f32>,
    /// Saved pre-projection velocities used for the `FLIP` increment.
    saved: Vec<f32>,
}

impl GoldenGrid {
    /// Creates a zeroed field for the given grid geometry.
    #[must_use]
    pub fn new(dims: GridDims) -> GoldenGrid {
        let n = dims.face_total();
        GoldenGrid {
            dims,
            momentum: vec![0; n],
            weight: vec![0; n],
            velocity: vec![0.0; n],
            saved: vec![0.0; n],
        }
    }

    /// The grid geometry.
    #[inline]
    #[must_use]
    pub fn dims(&self) -> GridDims {
        self.dims
    }

    /// The normalised face-velocity field (concatenated `[u | v | w]`).
    #[inline]
    #[must_use]
    pub fn velocity(&self) -> &[f32] {
        &self.velocity
    }

    /// The per-axis face-field descriptors in `[u, v, w]` order.
    fn axes(&self) -> [AxisGeom; 3] {
        let d = self.dims;
        let (nx, ny, nz) = (d.nx as usize, d.ny as usize, d.nz as usize);
        [
            AxisGeom {
                off: Vec3::new(0.0, 0.5, 0.5),
                dims: (nx + 1, ny, nz),
                base: d.u_offset(),
            },
            AxisGeom {
                off: Vec3::new(0.5, 0.0, 0.5),
                dims: (nx, ny + 1, nz),
                base: d.v_offset(),
            },
            AxisGeom {
                off: Vec3::new(0.5, 0.5, 0.0),
                dims: (nx, ny, nz + 1),
                base: d.w_offset(),
            },
        ]
    }

    /// Clears the momentum and weight accumulators before a fresh scatter.
    pub fn begin_transfer(&mut self) {
        for m in &mut self.momentum {
            *m = 0;
        }
        for w in &mut self.weight {
            *w = 0;
        }
    }

    /// Splats one marker particle's velocity onto the faces with trilinear
    /// weights, quantising each contribution and summing it as `i32` exactly as
    /// the device atomics do.
    pub fn scatter_velocity(&mut self, position: Vec3, vel: Vec3) {
        let c = self.dims.cell_space(position);
        let axes = self.axes();
        let components = [vel.x, vel.y, vel.z];
        for (axis, &value) in axes.iter().zip(components.iter()) {
            let sx = axis_stencil(c.x - axis.off.x, axis.dims.0);
            let sy = axis_stencil(c.y - axis.off.y, axis.dims.1);
            let sz = axis_stencil(c.z - axis.off.z, axis.dims.2);
            let nodes = trilinear_nodes(sx, sy, sz, axis.stride_y(), axis.stride_z());
            for &(local, w) in &nodes {
                if w <= 0.0 {
                    continue;
                }
                let idx = axis.base + local;
                self.momentum[idx] += quantise(w * value, MOMENTUM_SCALE);
                self.weight[idx] += quantise(w, WEIGHT_SCALE);
            }
        }
    }

    /// Splats one marker particle's velocity with the `APIC` affine correction
    /// `C·(x_face − x_p)` added to each face, preserving local angular momentum
    /// (Jiang et al. 2015). Each contribution is quantised and summed as `i32`
    /// exactly as the device atomics do, mirroring [`scatter_velocity`] plus the
    /// affine term so the `CPU` twin matches the device `p2g_scatter_affine`
    /// kernel bit-for-bit on the accumulators.
    ///
    /// [`scatter_velocity`]: GoldenGrid::scatter_velocity
    pub fn scatter_velocity_affine(&mut self, position: Vec3, vel: Vec3, affine: Mat3) {
        let c = self.dims.cell_space(position);
        let dx = self.dims.dx;
        let axes = self.axes();
        // Rows of the affine matrix (glam `Mat3` is column-major).
        let rows = [
            Vec3::new(affine.x_axis.x, affine.y_axis.x, affine.z_axis.x),
            Vec3::new(affine.x_axis.y, affine.y_axis.y, affine.z_axis.y),
            Vec3::new(affine.x_axis.z, affine.y_axis.z, affine.z_axis.z),
        ];
        let base_vels = [vel.x, vel.y, vel.z];
        for ((axis, &base_vel), &row) in axes.iter().zip(base_vels.iter()).zip(rows.iter()) {
            let sx = axis_stencil(c.x - axis.off.x, axis.dims.0);
            let sy = axis_stencil(c.y - axis.off.y, axis.dims.1);
            let sz = axis_stencil(c.z - axis.off.z, axis.dims.2);
            let xs = [(sx.lo, 1.0 - sx.frac), (sx.hi, sx.frac)];
            let ys = [(sy.lo, 1.0 - sy.frac), (sy.hi, sy.frac)];
            let zs = [(sz.lo, 1.0 - sz.frac), (sz.hi, sz.frac)];
            for &(ix, wx) in &xs {
                for &(iy, wy) in &ys {
                    for &(iz, wz) in &zs {
                        let w = wx * wy * wz;
                        if w <= 0.0 {
                            continue;
                        }
                        let node = Vec3::new(
                            ix as f32 + axis.off.x,
                            iy as f32 + axis.off.y,
                            iz as f32 + axis.off.z,
                        );
                        let dpos = (node - c) * dx;
                        let value = base_vel + row.dot(dpos);
                        let local = ix + axis.stride_y() * iy + axis.stride_z() * iz;
                        let idx = axis.base + local;
                        self.momentum[idx] += quantise(w * value, MOMENTUM_SCALE);
                        self.weight[idx] += quantise(w, WEIGHT_SCALE);
                    }
                }
            }
        }
    }

    /// Divides accumulated momentum by accumulated weight on each face,
    /// producing the mass-weighted average velocity (zero where untouched).
    pub fn normalize_velocity(&mut self) {
        for idx in 0..self.velocity.len() {
            let w = self.weight[idx];
            self.velocity[idx] = if w > 0 {
                dequantise(self.momentum[idx], MOMENTUM_SCALE) / dequantise(w, WEIGHT_SCALE)
            } else {
                0.0
            };
        }
    }

    /// Snapshots the current velocity field as the saved (pre-projection) field
    /// for the `FLIP` increment.
    pub fn save_velocity(&mut self) {
        self.saved.copy_from_slice(&self.velocity);
    }

    /// Mutable access to the normalised face-velocity field (concatenated
    /// `[u | v | w]`).
    ///
    /// The in-place grid operators of a full solver step — `add_gravity`,
    /// `enforce_solid_faces`, and the pressure projection — write the field
    /// through this handle, so a full step can run without reallocating.
    #[inline]
    pub fn velocity_mut(&mut self) -> &mut [f32] {
        &mut self.velocity
    }

    /// Extrapolates the velocity field into its unknown faces, one staggered
    /// axis at a time, using the transfer weight accumulator as the known-band
    /// mask (a face is known when its accumulated weight is positive).
    ///
    /// This is the concatenated-field twin of the core
    /// `MacGrid::extrapolate_velocity`: it runs `iterations` Jacobi sweeps per
    /// axis through [`extrapolate_axis`], growing the known band outward one
    /// cell per sweep so advection near the free surface stays stable.
    pub fn extrapolate_velocity(&mut self, iterations: u32) {
        let d = self.dims;
        let axes = [
            (
                d.u_offset(),
                d.u_count(),
                AxisDims::new(d.nx + 1, d.ny, d.nz),
            ),
            (
                d.v_offset(),
                d.v_count(),
                AxisDims::new(d.nx, d.ny + 1, d.nz),
            ),
            (
                d.w_offset(),
                d.w_count(),
                AxisDims::new(d.nx, d.ny, d.nz + 1),
            ),
        ];
        for (base, count, adims) in axes {
            let weights: Vec<f32> = self.weight[base..base + count]
                .iter()
                .map(|&w| dequantise(w, WEIGHT_SCALE))
                .collect();
            extrapolate_axis(
                &mut self.velocity[base..base + count],
                &weights,
                adims,
                iterations,
            );
        }
    }

    /// Samples the normalised velocity field trilinearly at world `position`.
    #[must_use]
    pub fn sample_velocity(&self, position: Vec3) -> Vec3 {
        self.sample_from(position, &self.velocity)
    }

    /// Samples the saved (pre-projection) velocity field trilinearly.
    #[must_use]
    pub fn sample_saved_velocity(&self, position: Vec3) -> Vec3 {
        self.sample_from(position, &self.saved)
    }

    /// Samples the velocity and reconstructs the `APIC` affine matrix `C` at
    /// world `position` via a per-component least-squares affine fit
    /// (`C_row = B·D⁻¹`, Jiang et al. 2015). `D` is regularised so the solve
    /// stays well-defined even for degenerate stencils. This mirrors the core
    /// reference and the device `g2p_affine` kernel; the only floating-point
    /// difference from the device is the reassociated `3×3` solve, bounded by
    /// the parity tolerance.
    #[must_use]
    pub fn sample_velocity_affine(&self, position: Vec3) -> (Vec3, Mat3) {
        let c = self.dims.cell_space(position);
        let dx = self.dims.dx;
        let axes = self.axes();
        let eps = 1.0e-9 + 1.0e-6 * dx * dx;
        let reg = Mat3::from_diagonal(Vec3::splat(eps));
        let mut vals = [0.0f32; 3];
        let mut rows = [Vec3::ZERO; 3];
        for (n, axis) in axes.iter().enumerate() {
            let sx = axis_stencil(c.x - axis.off.x, axis.dims.0);
            let sy = axis_stencil(c.y - axis.off.y, axis.dims.1);
            let sz = axis_stencil(c.z - axis.off.z, axis.dims.2);
            let xs = [(sx.lo, 1.0 - sx.frac), (sx.hi, sx.frac)];
            let ys = [(sy.lo, 1.0 - sy.frac), (sy.hi, sy.frac)];
            let zs = [(sz.lo, 1.0 - sz.frac), (sz.hi, sz.frac)];
            let mut val = 0.0;
            let mut b = Vec3::ZERO;
            let mut d = Mat3::ZERO;
            for &(ix, wx) in &xs {
                for &(iy, wy) in &ys {
                    for &(iz, wz) in &zs {
                        let w = wx * wy * wz;
                        let local = ix + axis.stride_y() * iy + axis.stride_z() * iz;
                        let vf = self.velocity[axis.base + local];
                        val += w * vf;
                        let node = Vec3::new(
                            ix as f32 + axis.off.x,
                            iy as f32 + axis.off.y,
                            iz as f32 + axis.off.z,
                        );
                        let dpos = (node - c) * dx;
                        b += (w * vf) * dpos;
                        d += Mat3::from_cols(
                            dpos * (w * dpos.x),
                            dpos * (w * dpos.y),
                            dpos * (w * dpos.z),
                        );
                    }
                }
            }
            vals[n] = val;
            rows[n] = (d + reg).inverse() * b;
        }
        // Assemble `C` (column-major) from its three rows.
        let cmat = Mat3::from_cols(
            Vec3::new(rows[0].x, rows[1].x, rows[2].x),
            Vec3::new(rows[0].y, rows[1].y, rows[2].y),
            Vec3::new(rows[0].z, rows[1].z, rows[2].z),
        );
        (Vec3::new(vals[0], vals[1], vals[2]), cmat)
    }

    fn sample_from(&self, position: Vec3, field: &[f32]) -> Vec3 {
        let c = self.dims.cell_space(position);
        let axes = self.axes();
        let mut out = [0.0f32; 3];
        for (component, axis) in out.iter_mut().zip(axes.iter()) {
            let sx = axis_stencil(c.x - axis.off.x, axis.dims.0);
            let sy = axis_stencil(c.y - axis.off.y, axis.dims.1);
            let sz = axis_stencil(c.z - axis.off.z, axis.dims.2);
            let nodes = trilinear_nodes(sx, sy, sz, axis.stride_y(), axis.stride_z());
            let mut acc = 0.0;
            for &(local, w) in &nodes {
                acc += w * field[axis.base + local];
            }
            *component = acc;
        }
        Vec3::new(out[0], out[1], out[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_field_scatter_then_sample() {
        let dims = GridDims::new(8, 8, 8, 0.1, Vec3::ZERO);
        let mut g = GoldenGrid::new(dims);
        g.begin_transfer();
        let vel = Vec3::new(0.5, -0.3, 0.2);
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    let p = Vec3::new(
                        0.25 + a as f32 * 0.03,
                        0.25 + b as f32 * 0.03,
                        0.25 + c as f32 * 0.03,
                    );
                    g.scatter_velocity(p, vel);
                }
            }
        }
        g.normalize_velocity();
        let s = g.sample_velocity(Vec3::new(0.3, 0.3, 0.3));
        assert!((s - vel).length() < 1e-3, "sampled {s:?}");
    }

    #[test]
    fn untouched_faces_are_zero() {
        let dims = GridDims::new(4, 4, 4, 0.1, Vec3::ZERO);
        let mut g = GoldenGrid::new(dims);
        g.begin_transfer();
        g.normalize_velocity();
        assert!(g.velocity().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn integer_accumulation_is_order_independent() {
        let dims = GridDims::new(6, 6, 6, 0.1, Vec3::ZERO);
        let vel = Vec3::new(0.4, 0.1, -0.2);
        let ps = [
            Vec3::new(0.31, 0.33, 0.35),
            Vec3::new(0.34, 0.32, 0.31),
            Vec3::new(0.30, 0.36, 0.34),
        ];
        let mut a = GoldenGrid::new(dims);
        a.begin_transfer();
        for p in &ps {
            a.scatter_velocity(*p, vel);
        }
        let mut b = GoldenGrid::new(dims);
        b.begin_transfer();
        for p in ps.iter().rev() {
            b.scatter_velocity(*p, vel);
        }
        a.normalize_velocity();
        b.normalize_velocity();
        assert_eq!(a.velocity(), b.velocity());
    }
}
