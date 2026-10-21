//! Sampling a smooth scalar field from a particle cloud on a regular grid.
//!
//! Each particle contributes a smooth, compactly supported kernel; the field
//! value at a grid node is the sum of the kernels of all nearby particles. The
//! resulting "blobby" density field is positive inside dense particle regions
//! and falls to zero away from them, so an iso-contour of the field encloses
//! the particles. The field also supports trilinear sampling and finite-
//! difference gradients used to orient the reconstructed surface normals.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! summed compactly-supported density kernel follows the metaball / blobby
//! surface construction (Blinn 1982) and the particle-skinning approach of
//! Zhu & Bridson 2005.

use glam::Vec3;

use crate::math::scalar::Real;

/// A scalar field sampled at the nodes of a regular grid.
///
/// The field has `nx × ny × nz` nodes; node `(i, j, k)` sits at world position
/// `origin + dx·(i, j, k)`. An empty field (zero nodes) reconstructs to an
/// empty mesh.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ScalarField {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: Real,
    origin: Vec3,
    values: Vec<Real>,
}

/// The compactly-supported blobby kernel `(1 − q²)³` for `q ∈ [0, 1]`, and `0`
/// beyond. `q` is the distance normalised by the support radius. Uses only
/// multiplication (no disallowed transcendental calls).
#[inline]
#[must_use]
fn kernel(q: Real) -> Real {
    if q >= 1.0 {
        return 0.0;
    }
    let t = 1.0 - q * q;
    t * t * t
}

impl ScalarField {
    /// Creates a zero-valued field with the given node dimensions.
    #[must_use]
    pub fn zeros(nx: usize, ny: usize, nz: usize, dx: Real, origin: Vec3) -> ScalarField {
        let Some(count) = nx.checked_mul(ny).and_then(|n| n.checked_mul(nz)) else {
            return ScalarField::default();
        };
        if count == 0 || !dx.is_finite() || dx <= 0.0 || !origin.is_finite() {
            return ScalarField::default();
        }
        ScalarField {
            nx,
            ny,
            nz,
            dx,
            origin,
            values: vec![0.0; count],
        }
    }

    /// Number of nodes along x.
    #[inline]
    #[must_use]
    pub fn nx(&self) -> usize {
        self.nx
    }
    /// Number of nodes along y.
    #[inline]
    #[must_use]
    pub fn ny(&self) -> usize {
        self.ny
    }
    /// Number of nodes along z.
    #[inline]
    #[must_use]
    pub fn nz(&self) -> usize {
        self.nz
    }
    /// Node spacing.
    #[inline]
    #[must_use]
    pub fn dx(&self) -> Real {
        self.dx
    }
    /// World position of node `(0, 0, 0)`.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> Vec3 {
        self.origin
    }
    /// Whether the field has no nodes.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Flat index of node `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn node_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// The field value at node `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn value(&self, i: usize, j: usize, k: usize) -> Real {
        self.values[self.node_idx(i, j, k)]
    }

    /// World position of node `(i, j, k)`.
    #[inline]
    #[must_use]
    pub fn node_position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin + Vec3::new(i as Real, j as Real, k as Real) * self.dx
    }

    /// Adds `value` to node `(i, j, k)`.
    #[inline]
    pub fn add_value(&mut self, i: usize, j: usize, k: usize, value: Real) {
        let idx = self.node_idx(i, j, k);
        self.values[idx] += value;
    }

    /// Builds a summed-kernel density field enclosing `positions`.
    ///
    /// The field grid covers the particle bounding box padded by
    /// `radius + padding_cells·dx` on every side. Each particle adds the
    /// [`kernel`] weighted by distance / `radius` to nearby nodes. Returns an
    /// empty field when `positions` is empty.
    #[must_use]
    pub fn from_particles(
        positions: &[Vec3],
        radius: Real,
        dx: Real,
        padding_cells: usize,
    ) -> ScalarField {
        if positions.is_empty()
            || !radius.is_finite()
            || radius <= 0.0
            || !dx.is_finite()
            || dx <= 0.0
            || positions.iter().any(|position| !position.is_finite())
        {
            return ScalarField::default();
        }
        let mut lo = positions[0];
        let mut hi = positions[0];
        for &p in positions {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let pad = radius + padding_cells as Real * dx;
        if !pad.is_finite() {
            return ScalarField::default();
        }
        let lo = lo - Vec3::splat(pad);
        let hi = hi + Vec3::splat(pad);
        let span = hi - lo;
        if !span.is_finite() || span.min_element() < 0.0 {
            return ScalarField::default();
        }
        let Some(nx) = axis_nodes(span.x, dx) else {
            return ScalarField::default();
        };
        let Some(ny) = axis_nodes(span.y, dx) else {
            return ScalarField::default();
        };
        let Some(nz) = axis_nodes(span.z, dx) else {
            return ScalarField::default();
        };
        let mut field = ScalarField::zeros(nx.max(2), ny.max(2), nz.max(2), dx, lo);
        if field.is_empty() {
            return field;
        }
        let inv_r = 1.0 / radius;
        let reach = (radius / dx).ceil() as i64 + 1;
        for &p in positions {
            let rel = (p - lo) / dx;
            let ci = rel.x.floor() as i64;
            let cj = rel.y.floor() as i64;
            let ck = rel.z.floor() as i64;
            for k in (ck - reach)..=(ck + reach) {
                if k < 0 || k as usize >= field.nz {
                    continue;
                }
                for j in (cj - reach)..=(cj + reach) {
                    if j < 0 || j as usize >= field.ny {
                        continue;
                    }
                    for i in (ci - reach)..=(ci + reach) {
                        if i < 0 || i as usize >= field.nx {
                            continue;
                        }
                        let node = field.node_position(i as usize, j as usize, k as usize);
                        let q = (node - p).length() * inv_r;
                        let w = kernel(q);
                        if w > 0.0 {
                            field.add_value(i as usize, j as usize, k as usize, w);
                        }
                    }
                }
            }
        }
        field
    }

    /// Trilinearly samples the field at world position `p` (clamped to the
    /// node range).
    #[must_use]
    pub fn sample(&self, p: Vec3) -> Real {
        if self.is_empty() {
            return 0.0;
        }
        let c = (p - self.origin) / self.dx;
        let (i0, i1, fx) = Self::stencil(c.x, self.nx);
        let (j0, j1, fy) = Self::stencil(c.y, self.ny);
        let (k0, k1, fz) = Self::stencil(c.z, self.nz);
        let v = |i: usize, j: usize, k: usize| self.values[self.node_idx(i, j, k)];
        let c00 = v(i0, j0, k0) * (1.0 - fx) + v(i1, j0, k0) * fx;
        let c10 = v(i0, j1, k0) * (1.0 - fx) + v(i1, j1, k0) * fx;
        let c01 = v(i0, j0, k1) * (1.0 - fx) + v(i1, j0, k1) * fx;
        let c11 = v(i0, j1, k1) * (1.0 - fx) + v(i1, j1, k1) * fx;
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }

    /// The field gradient at world position `p`, via central finite
    /// differences of [`ScalarField::sample`].
    #[must_use]
    pub fn gradient(&self, p: Vec3) -> Vec3 {
        if self.is_empty() {
            return Vec3::ZERO;
        }
        let h = 0.5 * self.dx;
        let gx = self.sample(p + Vec3::new(h, 0.0, 0.0)) - self.sample(p - Vec3::new(h, 0.0, 0.0));
        let gy = self.sample(p + Vec3::new(0.0, h, 0.0)) - self.sample(p - Vec3::new(0.0, h, 0.0));
        let gz = self.sample(p + Vec3::new(0.0, 0.0, h)) - self.sample(p - Vec3::new(0.0, 0.0, h));
        Vec3::new(gx, gy, gz) / (2.0 * h)
    }

    #[inline]
    fn stencil(coord: Real, nodes: usize) -> (usize, usize, Real) {
        let fi = coord.floor();
        let i0 = fi as i64;
        let frac = coord - fi;
        let max = nodes as i64 - 1;
        let c0 = i0.clamp(0, max);
        let c1 = (i0 + 1).clamp(0, max);
        (c0 as usize, c1 as usize, frac.clamp(0.0, 1.0))
    }
}

#[inline]
fn axis_nodes(extent: Real, dx: Real) -> Option<usize> {
    let cells = (extent / dx).ceil();
    if !cells.is_finite() || cells < 0.0 {
        return None;
    }
    let nodes = cells as usize;
    nodes.checked_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_particles_gives_empty_field() {
        let f = ScalarField::from_particles(&[], 0.2, 0.1, 2);
        assert!(f.is_empty());
        assert_eq!(f.sample(Vec3::ZERO), 0.0);
    }

    #[test]
    fn field_peaks_near_particles() {
        let pts = [Vec3::splat(0.5)];
        let f = ScalarField::from_particles(&pts, 0.25, 0.05, 2);
        assert!(!f.is_empty());
        let at_particle = f.sample(Vec3::splat(0.5));
        let far = f.sample(Vec3::splat(0.5) + Vec3::new(0.24, 0.0, 0.0));
        assert!(at_particle > far);
        assert!(at_particle > 0.0);
    }

    #[test]
    fn gradient_points_up_the_density() {
        // Two particles along x; gradient between them and the left one should
        // have a positive x-component when moving toward the cluster centre.
        let pts = [Vec3::new(0.4, 0.5, 0.5), Vec3::new(0.6, 0.5, 0.5)];
        let f = ScalarField::from_particles(&pts, 0.3, 0.05, 2);
        let g = f.gradient(Vec3::new(0.35, 0.5, 0.5));
        assert!(g.x > 0.0, "gradient toward cluster should be +x, got {g:?}");
    }

    #[test]
    fn invalid_inputs_produce_empty_field_without_nan_gradient() {
        let invalid = ScalarField::from_particles(&[Vec3::ZERO], 0.0, 0.05, 1);
        assert!(invalid.is_empty());
        assert_eq!(invalid.gradient(Vec3::ZERO), Vec3::ZERO);

        let invalid_positions =
            ScalarField::from_particles(&[Vec3::new(Real::NAN, 0.0, 0.0)], 0.2, 0.05, 1);
        assert!(invalid_positions.is_empty());

        let invalid_grid = ScalarField::zeros(usize::MAX, 2, 2, 0.1, Vec3::ZERO);
        assert!(invalid_grid.is_empty());
    }
}
