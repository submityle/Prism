//! Fedkiw vorticity-confinement turbulence force over a sampled velocity grid.
//!
//! Large-scale numerical advection dissipates the small rotational structure
//! that reads as *turbulent detail* in fire, smoke, and dust. Fedkiw's
//! vorticity-confinement force reinjects that detail by finding where the
//! vorticity magnitude is locally strongest and pushing the flow back toward
//! those cores. Given a discrete 3D velocity field on a regular grid, the
//! recipe is:
//!
//! 1. vorticity `omega = curl(v)` via a central-difference `curl`,
//! 2. the scalar magnitude field `|omega|`,
//! 3. its gradient `eta = grad(|omega|)`, normalized to `N = eta / |eta|`,
//! 4. the confinement force `f = epsilon * dx * (N x omega)`.
//!
//! This module operates on a *real, sampled* velocity grid: every derivative
//! is a finite difference of the caller-supplied [`VelocityGrid`]. It is
//! deliberately distinct from [`super::curl_noise`], whose `CurlNoiseField`
//! synthesizes a divergence-free field procedurally from noise derivatives and
//! never touches a stored velocity field. Nothing here generates procedural
//! noise; it only differentiates the grid it is handed.
//!
//! The math is hand-rolled and `no_std`-friendly: the only non-integer
//! primitive used is `f32::sqrt` (for vector length and normalization). No
//! transcendental functions appear. A `GPU` compute port of this contract
//! would bind the velocity field as a `std430` storage buffer; the byte size of
//! that buffer is exposed through [`VelocityGrid::velocity_buffer_bytes`] so the
//! render graph and this `CPU` reference agree on the layout.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC4_STRIDE};

/// Vectors shorter than this are treated as zero-length, so normalization and
/// finite-difference denominators never divide by a vanishing quantity.
const MIN_LENGTH: f32 = 1.0e-12;

/// A hand-rolled 3-component vector, private to this contract so the module
/// stays self-contained.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3 {
    /// The additive identity.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum (named `plus` to avoid the `core::ops::Add` trait).
    #[must_use]
    pub fn plus(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference (named `minus` to avoid `core::ops::Sub`).
    #[must_use]
    pub fn minus(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale (named `scaled` to avoid `core::ops::Mul`).
    #[must_use]
    pub fn scaled(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Euclidean dot product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Right-handed cross product `self x rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Euclidean length; the only place `f32::sqrt` is used.
    #[must_use]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Returns the unit vector, or [`Vec3::ZERO`] for a (near-)zero vector so
    /// the result is never `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len > MIN_LENGTH {
            self.scaled(1.0 / len)
        } else {
            Self::ZERO
        }
    }
}

/// A regular 3D grid of sampled velocities, stored `x`-fastest.
///
/// `data` holds one [`Vec3`] per cell; `dims` is `[nx, ny, nz]`; `cell_size` is
/// the uniform spacing `dx` between neighboring cell centers.
pub struct VelocityGrid {
    /// Cell counts along each axis, `[nx, ny, nz]`.
    pub dims: [usize; 3],
    /// Uniform grid spacing `dx` shared by all three axes.
    pub cell_size: f32,
    /// Row-major (`x`-fastest) velocity samples, `dims` product entries long.
    pub data: Vec<Vec3>,
}

/// Clamps a coordinate into `0..dim`, guarding an empty axis.
fn clamp_index(v: usize, dim: usize) -> usize {
    if dim == 0 {
        0
    } else {
        v.min(dim - 1)
    }
}

impl VelocityGrid {
    /// Wraps caller-owned grid parameters and velocity samples.
    #[must_use]
    pub fn new(dims: [usize; 3], cell_size: f32, data: Vec<Vec3>) -> Self {
        Self {
            dims,
            cell_size,
            data,
        }
    }

    /// Total number of cells (`nx * ny * nz`).
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Row-major flat index of cell `(i, j, k)`.
    #[must_use]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    /// Samples the velocity at `(i, j, k)`, clamping to the grid boundary and
    /// returning [`Vec3::ZERO`] for an empty grid.
    #[must_use]
    pub fn sample(&self, i: usize, j: usize, k: usize) -> Vec3 {
        let ci = clamp_index(i, self.dims[0]);
        let cj = clamp_index(j, self.dims[1]);
        let ck = clamp_index(k, self.dims[2]);
        let idx = self.index(ci, cj, ck);
        self.data.get(idx).copied().unwrap_or(Vec3::ZERO)
    }

    /// The partial derivative `d v / d(axis)` of the velocity vector at
    /// `coord`, by central difference with boundary clamping. Axis `0`, `1`,
    /// `2` mean `x`, `y`, `z`. A degenerate axis (span zero) yields
    /// [`Vec3::ZERO`].
    fn velocity_partial(&self, coord: [usize; 3], axis: usize) -> Vec3 {
        let dim = self.dims[axis];
        let mut hi = coord;
        let mut lo = coord;
        hi[axis] = (coord[axis] + 1).min(dim.saturating_sub(1));
        lo[axis] = coord[axis].saturating_sub(1);
        let span = hi[axis].saturating_sub(lo[axis]);
        let span_scalar = f32::from(u8::try_from(span).unwrap_or(0));
        let denom = span_scalar * self.cell_size;
        let diff = self
            .sample(hi[0], hi[1], hi[2])
            .minus(self.sample(lo[0], lo[1], lo[2]));
        if denom > MIN_LENGTH {
            diff.scaled(1.0 / denom)
        } else {
            Vec3::ZERO
        }
    }

    /// Vorticity `omega = curl(v)` at `(i, j, k)` via central differences.
    #[must_use]
    pub fn curl_at(&self, i: usize, j: usize, k: usize) -> Vec3 {
        let coord = [i, j, k];
        let dvdx = self.velocity_partial(coord, 0);
        let dvdy = self.velocity_partial(coord, 1);
        let dvdz = self.velocity_partial(coord, 2);
        Vec3::new(dvdy.z - dvdz.y, dvdz.x - dvdx.z, dvdx.y - dvdy.x)
    }

    /// The partial derivative `d|omega| / d(axis)` at `coord`, matching the
    /// clamped central-difference stencil used for the velocity.
    fn magnitude_partial(&self, coord: [usize; 3], axis: usize) -> f32 {
        let dim = self.dims[axis];
        let mut hi = coord;
        let mut lo = coord;
        hi[axis] = (coord[axis] + 1).min(dim.saturating_sub(1));
        lo[axis] = coord[axis].saturating_sub(1);
        let span = hi[axis].saturating_sub(lo[axis]);
        let span_scalar = f32::from(u8::try_from(span).unwrap_or(0));
        let denom = span_scalar * self.cell_size;
        let mag_hi = self.curl_at(hi[0], hi[1], hi[2]).length();
        let mag_lo = self.curl_at(lo[0], lo[1], lo[2]).length();
        if denom > MIN_LENGTH {
            (mag_hi - mag_lo) / denom
        } else {
            0.0
        }
    }

    /// Gradient `eta = grad(|omega|)` of the vorticity-magnitude field at
    /// `(i, j, k)`.
    #[must_use]
    pub fn vorticity_gradient(&self, i: usize, j: usize, k: usize) -> Vec3 {
        let coord = [i, j, k];
        Vec3::new(
            self.magnitude_partial(coord, 0),
            self.magnitude_partial(coord, 1),
            self.magnitude_partial(coord, 2),
        )
    }

    /// Byte size of the `std430` `GPU` storage buffer that would hold this
    /// grid's velocity field, one padded `vec4` element per cell (clamped to a
    /// single element for an empty grid).
    #[must_use]
    pub fn velocity_buffer_bytes(&self) -> usize {
        storage_bytes(VEC4_STRIDE, self.cell_count())
    }
}

/// The vorticity-magnitude field `|omega|` for every cell, in row-major order
/// matching [`VelocityGrid::index`]. Each entry is non-negative.
#[must_use]
pub fn vorticity_magnitude_field(grid: &VelocityGrid) -> Vec<f32> {
    let mut out = Vec::with_capacity(grid.cell_count());
    for k in 0..grid.dims[2] {
        for j in 0..grid.dims[1] {
            for i in 0..grid.dims[0] {
                out.push(grid.curl_at(i, j, k).length());
            }
        }
    }
    out
}

/// The confinement force `f = epsilon * dx * (N x omega)` at cell `(i, j, k)`,
/// where `omega` is the local vorticity and `N` is the unit gradient of the
/// vorticity magnitude. Returns [`Vec3::ZERO`] where the magnitude field is
/// locally flat (nothing to confine toward).
#[must_use]
pub fn confinement_force(grid: &VelocityGrid, i: usize, j: usize, k: usize, epsilon: f32) -> Vec3 {
    let omega = grid.curl_at(i, j, k);
    let normal = grid.vorticity_gradient(i, j, k).normalize_or_zero();
    normal.cross(omega).scaled(epsilon * grid.cell_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CMP_EPS: f32 = 1.0e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= CMP_EPS
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    /// Builds `v = (-y, x, 0)` (rigid rotation about `z`) on a grid whose cell
    /// spacing is `cell_size`. Analytic curl is the constant `(0, 0, 2)`.
    fn rotation_grid(nx: usize, ny: usize, nz: usize, cell_size: f32) -> VelocityGrid {
        let mut data = Vec::with_capacity(nx * ny * nz);
        for k in 0..nz {
            let _ = k;
            for j in 0..ny {
                for i in 0..nx {
                    let x = f32::from(u8::try_from(i).unwrap_or(0)) * cell_size;
                    let y = f32::from(u8::try_from(j).unwrap_or(0)) * cell_size;
                    data.push(Vec3::new(-y, x, 0.0));
                }
            }
        }
        VelocityGrid::new([nx, ny, nz], cell_size, data)
    }

    /// A grid whose every cell holds the same velocity (curl is zero).
    fn uniform_grid(nx: usize, ny: usize, nz: usize, cell_size: f32, v: Vec3) -> VelocityGrid {
        let mut data = Vec::with_capacity(nx * ny * nz);
        for _ in 0..(nx * ny * nz) {
            data.push(v);
        }
        VelocityGrid::new([nx, ny, nz], cell_size, data)
    }

    #[test]
    fn vec3_new_and_zero_components() {
        let v = Vec3::new(1.0, -2.0, 3.5);
        assert!(approx(v.x, 1.0) && approx(v.y, -2.0) && approx(v.z, 3.5));
        assert!(approx_vec(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn vec3_plus_and_minus_are_inverse() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -5.0, 6.0);
        assert!(approx_vec(a.plus(b), Vec3::new(5.0, -3.0, 9.0)));
        assert!(approx_vec(a.plus(b).minus(b), a));
    }

    #[test]
    fn vec3_scaled_is_linear() {
        let a = Vec3::new(1.0, -2.0, 4.0);
        assert!(approx_vec(a.scaled(2.0), Vec3::new(2.0, -4.0, 8.0)));
        assert!(approx_vec(a.scaled(0.0), Vec3::ZERO));
    }

    #[test]
    fn vec3_dot_matches_definition() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!(approx(a.dot(b), 32.0));
        assert!(approx(a.dot(Vec3::ZERO), 0.0));
    }

    #[test]
    fn vec3_cross_is_right_handed() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert!(approx_vec(x.cross(y), Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn vec3_cross_anticommutes_and_is_orthogonal() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-2.0, 0.5, 4.0);
        let c = a.cross(b);
        assert!(approx_vec(c, b.cross(a).scaled(-1.0)));
        assert!(approx(c.dot(a), 0.0));
        assert!(approx(c.dot(b), 0.0));
    }

    #[test]
    fn vec3_length_of_three_four_five() {
        assert!(approx(Vec3::new(3.0, 4.0, 0.0).length(), 5.0));
        assert!(approx(Vec3::ZERO.length(), 0.0));
    }

    #[test]
    fn normalize_zero_vector_is_zero() {
        assert!(approx_vec(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO));
    }

    #[test]
    fn normalize_unit_and_general_vector() {
        assert!(approx_vec(
            Vec3::new(0.0, 5.0, 0.0).normalize_or_zero(),
            Vec3::new(0.0, 1.0, 0.0)
        ));
        let n = Vec3::new(1.0, 2.0, 2.0).normalize_or_zero();
        assert!(approx(n.length(), 1.0));
    }

    #[test]
    fn grid_index_round_trips_and_stays_in_range() {
        let grid = uniform_grid(3, 4, 5, 1.0, Vec3::ZERO);
        let mut seen = 0usize;
        for k in 0..grid.dims[2] {
            for j in 0..grid.dims[1] {
                for i in 0..grid.dims[0] {
                    let idx = grid.index(i, j, k);
                    assert!(idx < grid.cell_count());
                    assert_eq!(idx, seen);
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, grid.cell_count());
    }

    #[test]
    fn grid_cell_count_is_product_of_dims() {
        let grid = uniform_grid(3, 4, 5, 1.0, Vec3::ZERO);
        assert_eq!(grid.cell_count(), 60);
    }

    #[test]
    fn grid_sample_clamps_out_of_range_coordinates() {
        let grid = rotation_grid(2, 2, 1, 1.0);
        let corner = grid.sample(1, 1, 0);
        let clamped = grid.sample(9, 9, 9);
        assert!(approx_vec(clamped, corner));
    }

    #[test]
    fn empty_grid_sample_is_zero() {
        let grid = VelocityGrid::new([0, 0, 0], 1.0, Vec::new());
        assert!(approx_vec(grid.sample(0, 0, 0), Vec3::ZERO));
        assert_eq!(grid.cell_count(), 0);
    }

    #[test]
    fn velocity_buffer_bytes_follow_std430_vec4() {
        let grid = uniform_grid(2, 2, 2, 1.0, Vec3::ZERO);
        assert_eq!(grid.velocity_buffer_bytes(), VEC4_STRIDE * 8);
        let empty = VelocityGrid::new([0, 0, 0], 1.0, Vec::new());
        assert_eq!(empty.velocity_buffer_bytes(), VEC4_STRIDE);
    }

    #[test]
    fn uniform_field_has_zero_curl_and_zero_force() {
        let grid = uniform_grid(4, 4, 4, 0.5, Vec3::new(3.0, -1.0, 2.0));
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    assert!(approx_vec(grid.curl_at(i, j, k), Vec3::ZERO));
                    assert!(approx_vec(
                        confinement_force(&grid, i, j, k, 5.0),
                        Vec3::ZERO
                    ));
                }
            }
        }
    }

    #[test]
    fn rotation_field_curl_is_constant_two_on_z() {
        let grid = rotation_grid(5, 5, 3, 0.25);
        // Interior cell where the central-difference stencil is symmetric.
        let curl = grid.curl_at(2, 2, 1);
        assert!(approx_vec(curl, Vec3::new(0.0, 0.0, 2.0)));
    }

    #[test]
    fn curl_is_antisymmetric_in_velocity_sign() {
        let grid = rotation_grid(5, 5, 3, 0.5);
        let neg: Vec<Vec3> = grid.data.iter().map(|v| v.scaled(-1.0)).collect();
        let neg_grid = VelocityGrid::new(grid.dims, grid.cell_size, neg);
        for &(i, j, k) in &[(2usize, 2usize, 1usize), (1, 3, 0), (4, 0, 2)] {
            let a = grid.curl_at(i, j, k);
            let b = neg_grid.curl_at(i, j, k);
            assert!(approx_vec(b, a.scaled(-1.0)));
        }
    }

    #[test]
    fn vorticity_magnitude_field_is_non_negative_and_row_major() {
        let grid = rotation_grid(4, 4, 2, 0.75);
        let field = vorticity_magnitude_field(&grid);
        assert_eq!(field.len(), grid.cell_count());
        for k in 0..grid.dims[2] {
            for j in 0..grid.dims[1] {
                for i in 0..grid.dims[0] {
                    let m = field[grid.index(i, j, k)];
                    assert!(m >= 0.0);
                    assert!(approx(m, grid.curl_at(i, j, k).length()));
                }
            }
        }
    }

    #[test]
    fn confinement_force_is_orthogonal_to_gradient_and_vorticity() {
        // A sheared field whose vorticity magnitude varies in space, giving a
        // non-zero gradient and a genuine confinement direction.
        let (nx, ny, nz) = (6usize, 6usize, 3usize);
        let cell = 0.5f32;
        let mut data = Vec::with_capacity(nx * ny * nz);
        for k in 0..nz {
            let _ = k;
            for j in 0..ny {
                for i in 0..nx {
                    let x = f32::from(u8::try_from(i).unwrap_or(0)) * cell;
                    let y = f32::from(u8::try_from(j).unwrap_or(0)) * cell;
                    // v = (-y*x, x, 0): vorticity magnitude depends on x.
                    data.push(Vec3::new(-y * x, x, 0.0));
                }
            }
        }
        let grid = VelocityGrid::new([nx, ny, nz], cell, data);
        let force = confinement_force(&grid, 3, 3, 1, 2.0);
        let omega = grid.curl_at(3, 3, 1);
        let normal = grid.vorticity_gradient(3, 3, 1).normalize_or_zero();
        assert!(force.length() > CMP_EPS);
        assert!(approx(force.dot(omega), 0.0));
        assert!(approx(force.dot(normal), 0.0));
    }

    #[test]
    fn confinement_force_scales_linearly_with_epsilon() {
        let (nx, ny, nz) = (6usize, 6usize, 3usize);
        let cell = 0.5f32;
        let mut data = Vec::with_capacity(nx * ny * nz);
        for k in 0..nz {
            let _ = k;
            for j in 0..ny {
                for i in 0..nx {
                    let x = f32::from(u8::try_from(i).unwrap_or(0)) * cell;
                    let y = f32::from(u8::try_from(j).unwrap_or(0)) * cell;
                    data.push(Vec3::new(-y * x, x, 0.0));
                }
            }
        }
        let grid = VelocityGrid::new([nx, ny, nz], cell, data);
        let base = confinement_force(&grid, 3, 3, 1, 1.0);
        let scaled = confinement_force(&grid, 3, 3, 1, 3.0);
        assert!(approx_vec(scaled, base.scaled(3.0)));
    }

    #[test]
    fn single_cell_grid_is_degenerate_guarded() {
        let grid = uniform_grid(1, 1, 1, 1.0, Vec3::new(1.0, 2.0, 3.0));
        assert!(approx_vec(grid.curl_at(0, 0, 0), Vec3::ZERO));
        assert!(approx_vec(grid.vorticity_gradient(0, 0, 0), Vec3::ZERO));
        assert!(approx_vec(
            confinement_force(&grid, 0, 0, 0, 10.0),
            Vec3::ZERO
        ));
    }

    #[test]
    fn boundary_cell_curl_is_finite() {
        let grid = rotation_grid(4, 4, 2, 0.5);
        let curl = grid.curl_at(0, 0, 0);
        assert!(curl.length().is_finite());
        // One-sided stencil still recovers the rigid-rotation curl on z.
        assert!(approx(curl.z, 2.0));
    }
}
