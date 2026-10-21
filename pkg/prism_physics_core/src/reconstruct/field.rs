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

/// Allocation and sampling-work ceilings for a particle density field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldBudget {
    /// Maximum allocated scalar nodes.
    pub max_nodes: usize,
    /// Maximum cells available to the subsequent extraction pass.
    pub max_cells: usize,
    /// Maximum particle/node kernel evaluations, including zero-weight visits.
    pub max_sample_visits: usize,
}

/// A rejected field build never allocates a partial scalar grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldBuildError {
    /// Non-finite particles, invalid spacing/support, or unrepresentable bounds.
    InvalidInput,
    /// Grid dimensions or their products cannot be represented.
    GridOverflow,
    /// The padded grid exceeds the node ceiling.
    NodeBudgetExceeded,
    /// The padded grid exceeds the cell ceiling.
    CellBudgetExceeded,
    /// The clipped particle stencils exceed the sampling-work ceiling.
    SampleBudgetExceeded,
    /// The allocator refused the bounded grid allocation.
    AllocationFailed,
}

struct ParticleLayout {
    nodes: [usize; 3],
    origin: Vec3,
    count: usize,
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
        Self::from_particles_bounded(
            positions,
            radius,
            dx,
            padding_cells,
            FieldBudget {
                max_nodes: usize::MAX,
                max_cells: usize::MAX,
                max_sample_visits: usize::MAX,
            },
        )
        .unwrap_or_default()
    }

    /// Builds a density field after checking its exact padded dimensions and
    /// clipped sampling work, before allocating any grid storage. Empty inputs
    /// produce an empty field even with zero budgets. An error publishes no
    /// partial field. Budgets count logical elements, not allocator metadata.
    pub fn from_particles_bounded(
        positions: &[Vec3],
        radius: Real,
        dx: Real,
        padding_cells: usize,
        budget: FieldBudget,
    ) -> Result<ScalarField, FieldBuildError> {
        if !radius.is_finite()
            || radius <= 0.0
            || !radius.recip().is_finite()
            || !dx.is_finite()
            || dx <= 0.0
            || dx * 0.5 == 0.0
            || positions.iter().any(|position| !position.is_finite())
        {
            return Err(FieldBuildError::InvalidInput);
        }
        if positions.is_empty() {
            return Ok(ScalarField::default());
        }
        let mut lo = positions[0];
        let mut hi = positions[0];
        for &p in positions {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let pad = radius + padding_cells as Real * dx;
        if !pad.is_finite() {
            return Err(FieldBuildError::InvalidInput);
        }
        let lo = lo - Vec3::splat(pad);
        let hi = hi + Vec3::splat(pad);
        let span = hi - lo;
        if !lo.is_finite() || !hi.is_finite() || !span.is_finite() || span.min_element() < 0.0 {
            return Err(FieldBuildError::InvalidInput);
        }
        let mut nodes = [0; 3];
        for (axis, extent) in [span.x, span.y, span.z].into_iter().enumerate() {
            nodes[axis] = axis_nodes(extent, dx)
                .ok_or(FieldBuildError::GridOverflow)?
                .max(2);
        }
        let count = nodes[0]
            .checked_mul(nodes[1])
            .and_then(|n| n.checked_mul(nodes[2]))
            .ok_or(FieldBuildError::GridOverflow)?;
        let cells = (nodes[0] - 1)
            .checked_mul(nodes[1] - 1)
            .and_then(|n| n.checked_mul(nodes[2] - 1))
            .ok_or(FieldBuildError::GridOverflow)?;
        if count > budget.max_nodes {
            return Err(FieldBuildError::NodeBudgetExceeded);
        }
        if cells > budget.max_cells {
            return Err(FieldBuildError::CellBudgetExceeded);
        }
        let layout = ParticleLayout {
            nodes,
            origin: lo,
            count,
        };
        let mut visits = 0usize;
        for &p in positions {
            let ranges = particle_ranges(p, &layout, dx, radius)?;
            let work = ranges
                .iter()
                .try_fold(1usize, |work, &(start, end)| {
                    work.checked_mul(end - start + 1)
                })
                .ok_or(FieldBuildError::SampleBudgetExceeded)?;
            visits = visits
                .checked_add(work)
                .ok_or(FieldBuildError::SampleBudgetExceeded)?;
            if visits > budget.max_sample_visits {
                return Err(FieldBuildError::SampleBudgetExceeded);
            }
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(layout.count)
            .map_err(|_| FieldBuildError::AllocationFailed)?;
        values.resize(layout.count, 0.0);
        let mut field = ScalarField {
            nx: nodes[0],
            ny: nodes[1],
            nz: nodes[2],
            dx,
            origin: lo,
            values,
        };
        let inv_r = radius.recip();
        for &p in positions {
            let [xr, yr, zr] = particle_ranges(p, &layout, dx, radius)?;
            for k in zr.0..=zr.1 {
                for j in yr.0..=yr.1 {
                    for i in xr.0..=xr.1 {
                        let node = field.node_position(i, j, k);
                        let q = (node - p).length() * inv_r;
                        let w = kernel(q);
                        if w > 0.0 {
                            field.add_value(i, j, k, w);
                        }
                    }
                }
            }
        }
        Ok(field)
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

fn particle_ranges(
    point: Vec3,
    layout: &ParticleLayout,
    dx: Real,
    radius: Real,
) -> Result<[(usize, usize); 3], FieldBuildError> {
    let relative = (point - layout.origin) / dx;
    let reach = (radius / dx).ceil() as f64 + 1.0;
    if !relative.is_finite() || !reach.is_finite() {
        return Err(FieldBuildError::InvalidInput);
    }
    let mut ranges = [(0, 0); 3];
    for (axis, coordinate) in [relative.x, relative.y, relative.z].into_iter().enumerate() {
        let center = coordinate.floor() as f64;
        let last = (layout.nodes[axis] - 1) as f64;
        ranges[axis] = (
            (center - reach).clamp(0.0, last) as usize,
            (center + reach).clamp(0.0, last) as usize,
        );
    }
    Ok(ranges)
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
    fn bounded_grid_checks_exact_counts_and_preserves_reference_values() {
        let points = [Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0)];
        let reference = ScalarField::from_particles(&points, 0.2, 0.1, 1);
        let nodes = reference.nx() * reference.ny() * reference.nz();
        let cells = (reference.nx() - 1) * (reference.ny() - 1) * (reference.nz() - 1);
        let layout = ParticleLayout {
            nodes: [reference.nx(), reference.ny(), reference.nz()],
            origin: reference.origin(),
            count: nodes,
        };
        let visits: usize = points
            .iter()
            .map(|&point| {
                particle_ranges(point, &layout, 0.1, 0.2)
                    .unwrap()
                    .iter()
                    .map(|&(start, end)| end - start + 1)
                    .product::<usize>()
            })
            .sum();
        let budget = FieldBudget {
            max_nodes: nodes,
            max_cells: cells,
            max_sample_visits: visits,
        };
        assert_eq!(
            ScalarField::from_particles_bounded(&points, 0.2, 0.1, 1, budget),
            Ok(reference)
        );
        assert_eq!(
            ScalarField::from_particles_bounded(
                &points,
                0.2,
                0.1,
                1,
                FieldBudget {
                    max_nodes: nodes - 1,
                    ..budget
                }
            ),
            Err(FieldBuildError::NodeBudgetExceeded),
        );
        assert_eq!(
            ScalarField::from_particles_bounded(
                &points,
                0.2,
                0.1,
                1,
                FieldBudget {
                    max_cells: cells - 1,
                    ..budget
                }
            ),
            Err(FieldBuildError::CellBudgetExceeded),
        );
        assert_eq!(
            ScalarField::from_particles_bounded(
                &points,
                0.2,
                0.1,
                1,
                FieldBudget {
                    max_sample_visits: visits - 1,
                    ..budget
                }
            ),
            Err(FieldBuildError::SampleBudgetExceeded),
        );
    }

    #[test]
    fn duplicate_particles_are_charged_for_each_sampling_stencil() {
        let budget = FieldBudget {
            max_nodes: 10_000,
            max_cells: 10_000,
            max_sample_visits: 1_000,
        };
        assert!(ScalarField::from_particles_bounded(&[Vec3::ZERO], 0.2, 0.1, 1, budget).is_ok());
        assert_eq!(
            ScalarField::from_particles_bounded(&[Vec3::ZERO; 10], 0.2, 0.1, 1, budget),
            Err(FieldBuildError::SampleBudgetExceeded),
        );
    }

    #[test]
    fn bounded_field_rejects_extreme_inputs_and_accepts_empty_zero_budget() {
        let budget = FieldBudget {
            max_nodes: 0,
            max_cells: 0,
            max_sample_visits: 0,
        };
        assert!(
            ScalarField::from_particles_bounded(&[], 0.2, 0.1, 1, budget)
                .unwrap()
                .is_empty()
        );
        for (point, radius, dx) in [
            (Vec3::splat(Real::MAX), Real::MAX, 0.1),
            (Vec3::ZERO, 0.2, Real::from_bits(1)),
            (Vec3::NAN, 0.2, 0.1),
        ] {
            assert!(ScalarField::from_particles_bounded(&[point], radius, dx, 1, budget).is_err());
        }
    }

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
