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

use glam::Vec3;

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
