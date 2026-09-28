//! Staggered Marker-And-Cell (MAC) grid for the fluid solver.
//!
//! Velocities live on cell faces: the x-component `u` on the `x`-normal faces,
//! `v` on the `y`-normal faces, and `w` on the `z`-normal faces. Pressure and
//! cell classification (fluid / air / solid) live at cell centres. Storing
//! each velocity component on its own faces makes the discrete divergence and
//! pressure-gradient operators compact and keeps the projection stable.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The MAC
//! staggered layout and trilinear face interpolation are standard,  publicly
//! documented CFD constructs (Harlow & Welch 1965; Bridson).

use glam::{Mat3, Vec3};

use crate::math::scalar::Real;

/// Classification of a MAC cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum CellType {
    /// A cell containing liquid.
    Fluid,
    /// An empty (gas) cell; pressure is pinned to zero here.
    Air,
    /// A solid obstacle / wall cell; velocity normal to it is constrained.
    Solid,
}

/// Per-axis trilinear stencil: the two node indices and the interpolation
/// fraction along that axis, already clamped to the valid node range.
#[inline]
#[must_use]
fn axis_stencil(coord: Real, nodes: usize) -> (usize, usize, Real) {
    let fi = coord.floor();
    let i0 = fi as i32;
    let frac = coord - fi;
    let max = nodes as i32 - 1;
    let c0 = i0.clamp(0, max);
    let c1 = (i0 + 1).clamp(0, max);
    (c0 as usize, c1 as usize, frac.clamp(0.0, 1.0))
}

/// A staggered MAC grid of `nx × ny × nz` cells.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MacGrid {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: Real,
    origin: Vec3,
    /// x-face velocities, dims `(nx+1, ny, nz)`.
    u: Vec<Real>,
    /// y-face velocities, dims `(nx, ny+1, nz)`.
    v: Vec<Real>,
    /// z-face velocities, dims `(nx, ny, nz+1)`.
    w: Vec<Real>,
    /// Saved copies used to compute the FLIP velocity increment.
    u_saved: Vec<Real>,
    v_saved: Vec<Real>,
    w_saved: Vec<Real>,
    /// Accumulated scatter weights (parallel to `u`/`v`/`w`).
    u_weight: Vec<Real>,
    v_weight: Vec<Real>,
    w_weight: Vec<Real>,
    /// Per-cell classification, dims `(nx, ny, nz)`.
    cell: Vec<CellType>,
}

impl MacGrid {
    /// Creates a grid whose cells are all [`CellType::Air`] and whose
    /// velocities are zero.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: Real, origin: Vec3) -> MacGrid {
        let nu = (nx + 1) * ny * nz;
        let nv = nx * (ny + 1) * nz;
        let nw = nx * ny * (nz + 1);
        let nc = nx * ny * nz;
        MacGrid {
            nx,
            ny,
            nz,
            dx,
            origin,
            u: vec![0.0; nu],
            v: vec![0.0; nv],
            w: vec![0.0; nw],
            u_saved: vec![0.0; nu],
            v_saved: vec![0.0; nv],
            w_saved: vec![0.0; nw],
            u_weight: vec![0.0; nu],
            v_weight: vec![0.0; nv],
            w_weight: vec![0.0; nw],
            cell: vec![CellType::Air; nc],
        }
    }

    /// Number of cells along x.
    #[inline]
    #[must_use]
    pub fn nx(&self) -> usize {
        self.nx
    }
    /// Number of cells along y.
    #[inline]
    #[must_use]
    pub fn ny(&self) -> usize {
        self.ny
    }
    /// Number of cells along z.
    #[inline]
    #[must_use]
    pub fn nz(&self) -> usize {
        self.nz
    }
    /// Cell spacing.
    #[inline]
    #[must_use]
    pub fn dx(&self) -> Real {
        self.dx
    }
    /// World-space position of the corner of cell `(0, 0, 0)`.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    #[inline]
    fn u_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.nx + 1) * (j + self.ny * k)
    }
    #[inline]
    fn v_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + (self.ny + 1) * k)
    }
    #[inline]
    fn w_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }
    /// Flat index of cell `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn cell_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// The classification of cell `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn cell_type(&self, i: usize, j: usize, k: usize) -> CellType {
        self.cell[self.cell_idx(i, j, k)]
    }
    /// Sets the classification of cell `(i, j, k)`.
    #[inline]
    pub fn set_cell_type(&mut self, i: usize, j: usize, k: usize, t: CellType) {
        let idx = self.cell_idx(i, j, k);
        self.cell[idx] = t;
    }

    /// The `u` (x-face) velocity at face `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn u_at(&self, i: usize, j: usize, k: usize) -> Real {
        self.u[self.u_idx(i, j, k)]
    }
    /// The `v` (y-face) velocity at face `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn v_at(&self, i: usize, j: usize, k: usize) -> Real {
        self.v[self.v_idx(i, j, k)]
    }
    /// The `w` (z-face) velocity at face `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn w_at(&self, i: usize, j: usize, k: usize) -> Real {
        self.w[self.w_idx(i, j, k)]
    }
    /// Sets the `u` velocity at face `(i, j, k)`.
    #[inline]
    pub fn set_u(&mut self, i: usize, j: usize, k: usize, val: Real) {
        let idx = self.u_idx(i, j, k);
        self.u[idx] = val;
    }
    /// Sets the `v` velocity at face `(i, j, k)`.
    #[inline]
    pub fn set_v(&mut self, i: usize, j: usize, k: usize, val: Real) {
        let idx = self.v_idx(i, j, k);
        self.v[idx] = val;
    }
    /// Sets the `w` velocity at face `(i, j, k)`.
    #[inline]
    pub fn set_w(&mut self, i: usize, j: usize, k: usize, val: Real) {
        let idx = self.w_idx(i, j, k);
        self.w[idx] = val;
    }

    /// Zeroes velocities, weights, and marks every non-solid cell as air.
    pub fn begin_transfer(&mut self) {
        for x in &mut self.u {
            *x = 0.0;
        }
        for x in &mut self.v {
            *x = 0.0;
        }
        for x in &mut self.w {
            *x = 0.0;
        }
        for x in &mut self.u_weight {
            *x = 0.0;
        }
        for x in &mut self.v_weight {
            *x = 0.0;
        }
        for x in &mut self.w_weight {
            *x = 0.0;
        }
        for c in &mut self.cell {
            if *c != CellType::Solid {
                *c = CellType::Air;
            }
        }
    }

    /// Marks the cell containing world position `p` as fluid (unless it is a
    /// solid cell).
    pub fn mark_fluid_at(&mut self, p: Vec3) {
        if let Some((i, j, k)) = self.cell_of(p) {
            let idx = self.cell_idx(i, j, k);
            if self.cell[idx] != CellType::Solid {
                self.cell[idx] = CellType::Fluid;
            }
        }
    }

    /// Returns the cell indices containing `p`, if inside the grid.
    #[must_use]
    pub fn cell_of(&self, p: Vec3) -> Option<(usize, usize, usize)> {
        let c = (p - self.origin) / self.dx;
        if c.x < 0.0 || c.y < 0.0 || c.z < 0.0 {
            return None;
        }
        let (i, j, k) = (c.x as usize, c.y as usize, c.z as usize);
        if i >= self.nx || j >= self.ny || k >= self.nz {
            return None;
        }
        Some((i, j, k))
    }

    /// Marks a `thickness`-cell layer on every face as [`CellType::Solid`].
    pub fn set_solid_walls(&mut self, thickness: usize) {
        let t = thickness;
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if i < t
                        || j < t
                        || k < t
                        || i >= self.nx - t
                        || j >= self.ny - t
                        || k >= self.nz - t
                    {
                        let idx = self.cell_idx(i, j, k);
                        self.cell[idx] = CellType::Solid;
                    }
                }
            }
        }
    }

    /// Cell-space coordinate of `p` (in units of `dx`, relative to `origin`).
    #[inline]
    fn cell_space(&self, p: Vec3) -> Vec3 {
        (p - self.origin) / self.dx
    }

    /// Scatters a scalar `value` weighted trilinearly into a face field.
    #[expect(
        clippy::too_many_arguments,
        reason = "generic staggered scatter needs the field, weights, and node dims"
    )]
    fn scatter_axis(
        field: &mut [Real],
        weights: &mut [Real],
        c: Vec3,
        off: Vec3,
        dims: (usize, usize, usize),
        stride_y: usize,
        stride_z: usize,
        value: Real,
    ) {
        let (i0, i1, fx) = axis_stencil(c.x - off.x, dims.0);
        let (j0, j1, fy) = axis_stencil(c.y - off.y, dims.1);
        let (k0, k1, fz) = axis_stencil(c.z - off.z, dims.2);
        let xs = [(i0, 1.0 - fx), (i1, fx)];
        let ys = [(j0, 1.0 - fy), (j1, fy)];
        let zs = [(k0, 1.0 - fz), (k1, fz)];
        for &(ix, wx) in &xs {
            for &(iy, wy) in &ys {
                for &(iz, wz) in &zs {
                    let wgt = wx * wy * wz;
                    if wgt <= 0.0 {
                        continue;
                    }
                    let idx = ix + stride_y * iy + stride_z * iz;
                    field[idx] += wgt * value;
                    weights[idx] += wgt;
                }
            }
        }
    }

    /// Samples a face field trilinearly at cell-space coordinate `c`.
    #[must_use]
    fn sample_axis(
        field: &[Real],
        c: Vec3,
        off: Vec3,
        dims: (usize, usize, usize),
        stride_y: usize,
        stride_z: usize,
    ) -> Real {
        let (i0, i1, fx) = axis_stencil(c.x - off.x, dims.0);
        let (j0, j1, fy) = axis_stencil(c.y - off.y, dims.1);
        let (k0, k1, fz) = axis_stencil(c.z - off.z, dims.2);
        let xs = [(i0, 1.0 - fx), (i1, fx)];
        let ys = [(j0, 1.0 - fy), (j1, fy)];
        let zs = [(k0, 1.0 - fz), (k1, fz)];
        let mut acc = 0.0;
        for &(ix, wx) in &xs {
            for &(iy, wy) in &ys {
                for &(iz, wz) in &zs {
                    let idx = ix + stride_y * iy + stride_z * iz;
                    acc += wx * wy * wz * field[idx];
                }
            }
        }
        acc
    }

    /// Scatters a particle velocity onto the faces with trilinear weights.
    pub fn scatter_velocity(&mut self, p: Vec3, vel: Vec3) {
        let c = self.cell_space(p);
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        MacGrid::scatter_axis(
            &mut self.u,
            &mut self.u_weight,
            c,
            Vec3::new(0.0, 0.5, 0.5),
            (nx + 1, ny, nz),
            nx + 1,
            (nx + 1) * ny,
            vel.x,
        );
        MacGrid::scatter_axis(
            &mut self.v,
            &mut self.v_weight,
            c,
            Vec3::new(0.5, 0.0, 0.5),
            (nx, ny + 1, nz),
            nx,
            nx * (ny + 1),
            vel.y,
        );
        MacGrid::scatter_axis(
            &mut self.w,
            &mut self.w_weight,
            c,
            Vec3::new(0.5, 0.5, 0.0),
            (nx, ny, nz + 1),
            nx,
            nx * ny,
            vel.z,
        );
    }

    /// Samples the full velocity vector at world position `p`.
    #[must_use]
    pub fn sample_velocity(&self, p: Vec3) -> Vec3 {
        self.sample_from(p, &self.u, &self.v, &self.w)
    }

    /// Samples the velocity and reconstructs the APIC affine matrix `C` at
    /// world position `p` via a per-component least-squares affine fit
    /// (`C_row = B·D⁻¹`, Jiang et al. 2015). `D` is regularised so the solve
    /// stays well-defined even for degenerate stencils.
    #[must_use]
    pub fn sample_velocity_affine(&self, p: Vec3) -> (Vec3, Mat3) {
        let c = self.cell_space(p);
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let dx = self.dx;
        let (ux, bx, dmx) = MacGrid::gather_axis_affine(
            &self.u,
            c,
            Vec3::new(0.0, 0.5, 0.5),
            (nx + 1, ny, nz),
            nx + 1,
            (nx + 1) * ny,
            dx,
        );
        let (vy, by, dmy) = MacGrid::gather_axis_affine(
            &self.v,
            c,
            Vec3::new(0.5, 0.0, 0.5),
            (nx, ny + 1, nz),
            nx,
            nx * (ny + 1),
            dx,
        );
        let (wz, bz, dmz) = MacGrid::gather_axis_affine(
            &self.w,
            c,
            Vec3::new(0.5, 0.5, 0.0),
            (nx, ny, nz + 1),
            nx,
            nx * ny,
            dx,
        );
        let eps = 1.0e-9 + 1.0e-6 * dx * dx;
        let reg = Mat3::from_diagonal(Vec3::splat(eps));
        let row_x = (dmx + reg).inverse() * bx;
        let row_y = (dmy + reg).inverse() * by;
        let row_z = (dmz + reg).inverse() * bz;
        // Assemble `C` (column-major) from its three rows.
        let cmat = Mat3::from_cols(
            Vec3::new(row_x.x, row_y.x, row_z.x),
            Vec3::new(row_x.y, row_y.y, row_z.y),
            Vec3::new(row_x.z, row_y.z, row_z.z),
        );
        (Vec3::new(ux, vy, wz), cmat)
    }

    /// Per-axis affine gather helper. Returns the interpolated component value,
    /// the moment vector `B = Σ w·v·dpos`, and the moment matrix
    /// `D = Σ w·dpos⊗dpos`, where `dpos = x_face − x_p`.
    fn gather_axis_affine(
        field: &[Real],
        c: Vec3,
        off: Vec3,
        dims: (usize, usize, usize),
        stride_y: usize,
        stride_z: usize,
        dx: Real,
    ) -> (Real, Vec3, Mat3) {
        let (i0, i1, fx) = axis_stencil(c.x - off.x, dims.0);
        let (j0, j1, fy) = axis_stencil(c.y - off.y, dims.1);
        let (k0, k1, fz) = axis_stencil(c.z - off.z, dims.2);
        let xs = [(i0, 1.0 - fx), (i1, fx)];
        let ys = [(j0, 1.0 - fy), (j1, fy)];
        let zs = [(k0, 1.0 - fz), (k1, fz)];
        let mut val = 0.0;
        let mut b = Vec3::ZERO;
        let mut d = Mat3::ZERO;
        for &(ix, wx) in &xs {
            for &(iy, wy) in &ys {
                for &(iz, wz) in &zs {
                    let wgt = wx * wy * wz;
                    let idx = ix + stride_y * iy + stride_z * iz;
                    let vf = field[idx];
                    val += wgt * vf;
                    let node =
                        Vec3::new(ix as Real + off.x, iy as Real + off.y, iz as Real + off.z);
                    let dpos = (node - c) * dx;
                    b += (wgt * vf) * dpos;
                    d += Mat3::from_cols(
                        dpos * (wgt * dpos.x),
                        dpos * (wgt * dpos.y),
                        dpos * (wgt * dpos.z),
                    );
                }
            }
        }
        (val, b, d)
    }

    /// Samples the *saved* velocity field (used by FLIP) at world position `p`.
    #[must_use]
    pub fn sample_saved_velocity(&self, p: Vec3) -> Vec3 {
        self.sample_from(p, &self.u_saved, &self.v_saved, &self.w_saved)
    }

    fn sample_from(&self, p: Vec3, u: &[Real], v: &[Real], w: &[Real]) -> Vec3 {
        let c = self.cell_space(p);
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let ux = MacGrid::sample_axis(
            u,
            c,
            Vec3::new(0.0, 0.5, 0.5),
            (nx + 1, ny, nz),
            nx + 1,
            (nx + 1) * ny,
        );
        let vy = MacGrid::sample_axis(
            v,
            c,
            Vec3::new(0.5, 0.0, 0.5),
            (nx, ny + 1, nz),
            nx,
            nx * (ny + 1),
        );
        let wz = MacGrid::sample_axis(
            w,
            c,
            Vec3::new(0.5, 0.5, 0.0),
            (nx, ny, nz + 1),
            nx,
            nx * ny,
        );
        Vec3::new(ux, vy, wz)
    }

    /// Divides accumulated face momentum by accumulated weight, producing the
    /// mass-weighted average velocity on each face.
    pub fn normalize_velocity(&mut self) {
        for idx in 0..self.u.len() {
            if self.u_weight[idx] > 0.0 {
                self.u[idx] /= self.u_weight[idx];
            }
        }
        for idx in 0..self.v.len() {
            if self.v_weight[idx] > 0.0 {
                self.v[idx] /= self.v_weight[idx];
            }
        }
        for idx in 0..self.w.len() {
            if self.w_weight[idx] > 0.0 {
                self.w[idx] /= self.w_weight[idx];
            }
        }
    }

    /// Returns `true` for a face whose accumulated weight is positive (i.e. a
    /// face touched by at least one particle).
    #[inline]
    #[must_use]
    pub fn u_known(&self, i: usize, j: usize, k: usize) -> bool {
        self.u_weight[self.u_idx(i, j, k)] > 0.0
    }
    /// See [`MacGrid::u_known`].
    #[inline]
    #[must_use]
    pub fn v_known(&self, i: usize, j: usize, k: usize) -> bool {
        self.v_weight[self.v_idx(i, j, k)] > 0.0
    }
    /// See [`MacGrid::u_known`].
    #[inline]
    #[must_use]
    pub fn w_known(&self, i: usize, j: usize, k: usize) -> bool {
        self.w_weight[self.w_idx(i, j, k)] > 0.0
    }

    /// Copies the current velocity field into the saved field for FLIP.
    pub fn save_velocity(&mut self) {
        self.u_saved.copy_from_slice(&self.u);
        self.v_saved.copy_from_slice(&self.v);
        self.w_saved.copy_from_slice(&self.w);
    }

    /// Adds a constant acceleration `g·dt` to every face velocity.
    pub fn add_gravity(&mut self, g: Vec3, dt: Real) {
        let dv = g * dt;
        if dv.x != 0.0 {
            for x in &mut self.u {
                *x += dv.x;
            }
        }
        if dv.y != 0.0 {
            for x in &mut self.v {
                *x += dv.y;
            }
        }
        if dv.z != 0.0 {
            for x in &mut self.w {
                *x += dv.z;
            }
        }
    }

    /// Zeroes the velocity on faces adjacent to solid cells (no-through-flow).
    pub fn enforce_solid_faces(&mut self) {
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if self.cell_type(i, j, k) != CellType::Solid {
                        continue;
                    }
                    // Faces bordering this solid cell get zero normal velocity.
                    self.set_u(i, j, k, 0.0);
                    self.set_u(i + 1, j, k, 0.0);
                    self.set_v(i, j, k, 0.0);
                    self.set_v(i, j + 1, k, 0.0);
                    self.set_w(i, j, k, 0.0);
                    self.set_w(i, j, k + 1, 0.0);
                }
            }
        }
    }

    /// The discrete velocity divergence of fluid cell `(i, j, k)`, scaled by
    /// `1/dx` (so it has units of inverse time).
    #[must_use]
    pub fn divergence(&self, i: usize, j: usize, k: usize) -> Real {
        let du = self.u_at(i + 1, j, k) - self.u_at(i, j, k);
        let dv = self.v_at(i, j + 1, k) - self.v_at(i, j, k);
        let dw = self.w_at(i, j, k + 1) - self.w_at(i, j, k);
        (du + dv + dw) / self.dx
    }

    /// Extrapolates each velocity component from *known* faces (those touched
    /// by particles during the scatter, i.e. positive weight) into the
    /// surrounding unknown faces using several sweeps of nearest-neighbour
    /// averaging. This provides plausible velocities in air cells so that
    /// particle advection near the free surface stays stable.
    pub fn extrapolate_velocity(&mut self, iterations: usize) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        MacGrid::extrapolate_axis(&mut self.u, &self.u_weight, (nx + 1, ny, nz), iterations);
        MacGrid::extrapolate_axis(&mut self.v, &self.v_weight, (nx, ny + 1, nz), iterations);
        MacGrid::extrapolate_axis(&mut self.w, &self.w_weight, (nx, ny, nz + 1), iterations);
    }

    /// One-component extrapolation helper: fills unknown entries from the
    /// average of their known 6-neighbours, growing the known set each sweep.
    fn extrapolate_axis(
        field: &mut [Real],
        weights: &[Real],
        dims: (usize, usize, usize),
        iterations: usize,
    ) {
        let (dx, dy, dz) = dims;
        let flat = |i: usize, j: usize, k: usize| i + dx * (j + dy * k);
        let mut known: Vec<bool> = weights.iter().map(|&w| w > 0.0).collect();
        for _ in 0..iterations {
            let prev = known.clone();
            let src = field.to_vec();
            for k in 0..dz {
                for j in 0..dy {
                    for i in 0..dx {
                        let id = flat(i, j, k);
                        if prev[id] {
                            continue;
                        }
                        let mut acc = 0.0;
                        let mut cnt = 0.0;
                        if i > 0 && prev[flat(i - 1, j, k)] {
                            acc += src[flat(i - 1, j, k)];
                            cnt += 1.0;
                        }
                        if i + 1 < dx && prev[flat(i + 1, j, k)] {
                            acc += src[flat(i + 1, j, k)];
                            cnt += 1.0;
                        }
                        if j > 0 && prev[flat(i, j - 1, k)] {
                            acc += src[flat(i, j - 1, k)];
                            cnt += 1.0;
                        }
                        if j + 1 < dy && prev[flat(i, j + 1, k)] {
                            acc += src[flat(i, j + 1, k)];
                            cnt += 1.0;
                        }
                        if k > 0 && prev[flat(i, j, k - 1)] {
                            acc += src[flat(i, j, k - 1)];
                            cnt += 1.0;
                        }
                        if k + 1 < dz && prev[flat(i, j, k + 1)] {
                            acc += src[flat(i, j, k + 1)];
                            cnt += 1.0;
                        }
                        if cnt > 0.0 {
                            field[id] = acc / cnt;
                            known[id] = true;
                        }
                    }
                }
            }
        }
    }

    /// APIC scatter: like [`MacGrid::scatter_velocity`] but adds the affine
    /// velocity correction `C·(x_face − x_p)` to each face, preserving local
    /// angular momentum (Jiang et al. 2015).
    pub fn scatter_velocity_affine(&mut self, p: Vec3, vel: Vec3, affine: Mat3) {
        let c = self.cell_space(p);
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let dx = self.dx;
        // Rows of the affine matrix (glam `Mat3` is column-major).
        let row_x = Vec3::new(affine.x_axis.x, affine.y_axis.x, affine.z_axis.x);
        let row_y = Vec3::new(affine.x_axis.y, affine.y_axis.y, affine.z_axis.y);
        let row_z = Vec3::new(affine.x_axis.z, affine.y_axis.z, affine.z_axis.z);
        MacGrid::scatter_axis_affine(
            &mut self.u,
            &mut self.u_weight,
            c,
            Vec3::new(0.0, 0.5, 0.5),
            (nx + 1, ny, nz),
            nx + 1,
            (nx + 1) * ny,
            dx,
            vel.x,
            row_x,
        );
        MacGrid::scatter_axis_affine(
            &mut self.v,
            &mut self.v_weight,
            c,
            Vec3::new(0.5, 0.0, 0.5),
            (nx, ny + 1, nz),
            nx,
            nx * (ny + 1),
            dx,
            vel.y,
            row_y,
        );
        MacGrid::scatter_axis_affine(
            &mut self.w,
            &mut self.w_weight,
            c,
            Vec3::new(0.5, 0.5, 0.0),
            (nx, ny, nz + 1),
            nx,
            nx * ny,
            dx,
            vel.z,
            row_z,
        );
    }

    /// Affine-corrected one-component scatter helper.
    #[expect(
        clippy::too_many_arguments,
        reason = "affine staggered scatter needs field, weights, geometry, and the affine row"
    )]
    fn scatter_axis_affine(
        field: &mut [Real],
        weights: &mut [Real],
        c: Vec3,
        off: Vec3,
        dims: (usize, usize, usize),
        stride_y: usize,
        stride_z: usize,
        dx: Real,
        base_vel: Real,
        row: Vec3,
    ) {
        let (i0, i1, fx) = axis_stencil(c.x - off.x, dims.0);
        let (j0, j1, fy) = axis_stencil(c.y - off.y, dims.1);
        let (k0, k1, fz) = axis_stencil(c.z - off.z, dims.2);
        let xs = [(i0, 1.0 - fx), (i1, fx)];
        let ys = [(j0, 1.0 - fy), (j1, fy)];
        let zs = [(k0, 1.0 - fz), (k1, fz)];
        for &(ix, wx) in &xs {
            for &(iy, wy) in &ys {
                for &(iz, wz) in &zs {
                    let wgt = wx * wy * wz;
                    if wgt <= 0.0 {
                        continue;
                    }
                    // Node cell-space position and offset from the particle.
                    let node =
                        Vec3::new(ix as Real + off.x, iy as Real + off.y, iz as Real + off.z);
                    let dpos = (node - c) * dx;
                    let value = base_vel + row.dot(dpos);
                    let idx = ix + stride_y * iy + stride_z * iz;
                    field[idx] += wgt * value;
                    weights[idx] += wgt;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_then_sample_constant_field() {
        let mut g = MacGrid::new(8, 8, 8, 0.1, Vec3::ZERO);
        g.begin_transfer();
        // Fill a block of particles moving at a constant velocity.
        let vel = Vec3::new(0.5, -0.3, 0.2);
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    let p = Vec3::new(
                        0.25 + a as Real * 0.03,
                        0.25 + b as Real * 0.03,
                        0.25 + c as Real * 0.03,
                    );
                    g.scatter_velocity(p, vel);
                }
            }
        }
        g.normalize_velocity();
        // Sampling well inside the filled region recovers the constant field.
        let s = g.sample_velocity(Vec3::new(0.3, 0.3, 0.3));
        assert!((s - vel).length() < 1.0e-3, "sampled {s:?}");
    }

    #[test]
    fn divergence_of_constant_field_is_zero() {
        let mut g = MacGrid::new(6, 6, 6, 0.1, Vec3::ZERO);
        for x in 0..g.u.len() {
            g.u[x] = 1.7;
        }
        assert!(g.divergence(2, 2, 2).abs() < 1.0e-5);
    }

    #[test]
    fn cell_of_maps_positions() {
        let g = MacGrid::new(10, 10, 10, 0.1, Vec3::ZERO);
        assert_eq!(g.cell_of(Vec3::new(0.05, 0.05, 0.05)), Some((0, 0, 0)));
        assert_eq!(g.cell_of(Vec3::new(0.95, 0.35, 0.15)), Some((9, 3, 1)));
        assert_eq!(g.cell_of(Vec3::new(-0.1, 0.0, 0.0)), None);
        assert_eq!(g.cell_of(Vec3::new(2.0, 0.0, 0.0)), None);
    }
}
