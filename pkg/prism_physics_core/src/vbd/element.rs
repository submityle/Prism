//! Elastic spring elements and their variational force / Hessian.
//!
//! A [`SpringElement`] couples two particles with a Hookean stretch energy
//!
//! ```text
//! Psi = 0.5 * k * (|x_a - x_b| - rest_length)^2
//! ```
//!
//! Vertex Block Descent needs, per incident element, the negative energy
//! gradient (the elastic *force*) and the `3x3` energy Hessian block with
//! respect to a single endpoint. The raw spring Hessian
//!
//! ```text
//! H = k * n n^T + k * (C / l) * (I - n n^T)
//! ```
//!
//! (with `C = l - rest_length`, `n` the unit separation) becomes indefinite
//! under compression (`C < 0`). Following the VBD / projective-dynamics
//! practice we project it to the nearest positive-semidefinite matrix by
//! clamping the transverse coefficient `k * C / l` to be non-negative, which
//! keeps every vertex system solvable and the descent monotone.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! Hookean spring energy, its gradient/Hessian, and the positive-semidefinite
//! Hessian projection are standard, publicly documented results (Chen et al.,
//! "Vertex Block Descent", 2024; Liu et al., "Fast Simulation of Mass-Spring
//! Systems", 2013).

use glam::{Mat3, Vec3};

use crate::math::scalar::{Real, EPSILON};
use crate::soft::constraint::ConstraintSet;
use crate::soft::particle::ParticleHandle;

/// Returns the outer product `a b^T` as a `3x3` matrix.
#[must_use]
pub fn outer(a: Vec3, b: Vec3) -> Mat3 {
    // `Mat3::from_cols` takes column vectors; column `j` of `a b^T` is `a * b[j]`.
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// A Hookean stretch spring between two particles.
///
/// Springs are the structural, shear, and bending edges of cloth, the links of
/// rope, and the lattice edges of a volumetric soft body. A higher
/// [`stiffness`](Self::stiffness) makes the spring resist stretching more; VBD
/// stays stable for arbitrarily large values.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpringElement {
    /// First coupled particle.
    pub a: ParticleHandle,
    /// Second coupled particle.
    pub b: ParticleHandle,
    /// Rest (unstressed) separation between the two particles, in metres.
    pub rest_length: Real,
    /// Hookean stiffness (energy per squared metre of stretch).
    pub stiffness: Real,
}

/// The elastic force and positive-semidefinite Hessian contributed by one
/// spring to one of its endpoints.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SpringContribution {
    /// Negative energy gradient with respect to the endpoint (the force).
    pub force: Vec3,
    /// Positive-semidefinite `3x3` energy Hessian block for the endpoint.
    pub hessian: Mat3,
}

impl SpringElement {
    /// Creates a spring holding `a` and `b` at `rest_length` with `stiffness`.
    #[must_use]
    pub fn new(a: ParticleHandle, b: ParticleHandle, rest_length: Real, stiffness: Real) -> Self {
        SpringElement {
            a,
            b,
            rest_length,
            stiffness,
        }
    }

    /// Returns the current length of the spring given the position slice.
    #[must_use]
    pub fn current_length(&self, positions: &[Vec3]) -> Real {
        (positions[self.a.index()] - positions[self.b.index()]).length()
    }

    /// Returns the stored elastic energy given the position slice.
    #[must_use]
    pub fn energy(&self, positions: &[Vec3]) -> Real {
        let c = self.current_length(positions) - self.rest_length;
        0.5 * self.stiffness * c * c
    }

    /// Returns the force and PSD Hessian this spring contributes to the
    /// endpoint identified by `vertex` (which must be `a` or `b`).
    ///
    /// The force points to reduce the spring's energy; the Hessian block is the
    /// same for either endpoint. When the two particles coincide (degenerate
    /// zero length) the separation direction is undefined and a zero
    /// contribution is returned.
    #[must_use]
    pub fn contribution(&self, vertex: ParticleHandle, positions: &[Vec3]) -> SpringContribution {
        let pa = positions[self.a.index()];
        let pb = positions[self.b.index()];
        let d = pa - pb;
        let l = d.length();
        if l <= EPSILON {
            return SpringContribution {
                force: Vec3::ZERO,
                hessian: Mat3::ZERO,
            };
        }
        let n = d / l;
        let c = l - self.rest_length;
        let k = self.stiffness;

        // Gradient of the energy w.r.t. `a` is `k * C * n`; w.r.t. `b` it is the
        // negation. The force is the negative gradient.
        let grad_a = n * (k * c);
        let force = if vertex.index() == self.a.index() {
            -grad_a
        } else {
            grad_a
        };

        // Raw Hessian: k n n^T + k (C/l) (I - n n^T). Project to PSD by clamping
        // the transverse coefficient to be non-negative.
        let nnt = outer(n, n);
        let transverse = (k * c / l).max(0.0);
        let hessian = nnt * k + (Mat3::IDENTITY - nnt) * transverse;

        SpringContribution { force, hessian }
    }
}

/// A flat collection of spring elements defining a deformable body's topology.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SpringSet {
    /// The springs, projected in index order for determinism.
    pub springs: Vec<SpringElement>,
}

impl SpringSet {
    /// Creates an empty spring set.
    #[must_use]
    pub const fn new() -> SpringSet {
        SpringSet {
            springs: Vec::new(),
        }
    }

    /// Returns the number of springs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.springs.len()
    }

    /// Returns `true` when the set holds no springs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.springs.is_empty()
    }

    /// Appends a spring.
    pub fn push(&mut self, spring: SpringElement) {
        self.springs.push(spring);
    }

    /// Builds a spring set from the distance constraints of a soft-body
    /// [`ConstraintSet`], converting each constraint's XPBD compliance into an
    /// equivalent Hookean stiffness `k = 1 / compliance`.
    ///
    /// A perfectly rigid distance constraint (compliance `0`) has no finite
    /// stiffness, so it is mapped to `default_stiffness`, which callers pick to
    /// be as stiff as their scene needs; VBD remains stable regardless.
    #[must_use]
    pub fn from_constraint_set(constraints: &ConstraintSet, default_stiffness: Real) -> SpringSet {
        let mut set = SpringSet::new();
        for c in &constraints.distance {
            let stiffness = if c.compliance > 0.0 {
                1.0 / c.compliance
            } else {
                default_stiffness
            };
            set.push(SpringElement::new(c.a, c.b, c.rest_length, stiffness));
        }
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn h(i: u32) -> ParticleHandle {
        ParticleHandle::from_index(i)
    }

    #[test]
    fn outer_product_matches_manual() {
        let m = outer(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        // Row 0 should be [4, 5, 6]; glam is column-major so check elements.
        assert_eq!(m.col(0), Vec3::new(4.0, 8.0, 12.0));
        assert_eq!(m.col(1), Vec3::new(5.0, 10.0, 15.0));
        assert_eq!(m.col(2), Vec3::new(6.0, 12.0, 18.0));
    }

    #[test]
    fn energy_is_zero_at_rest_and_positive_when_stretched() {
        let positions = [Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0)];
        let rest = SpringElement::new(h(0), h(1), 1.0, 100.0);
        assert!(rest.energy(&positions).abs() < 1e-6);
        let stretched = [Vec3::ZERO, Vec3::new(0.0, -2.0, 0.0)];
        assert!(rest.energy(&stretched) > 0.0);
    }

    #[test]
    fn stretched_spring_pulls_endpoints_together() {
        // a above b, stretched beyond rest: force on a points down (-y toward b),
        // force on b points up (+y toward a).
        let positions = [Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0)];
        let s = SpringElement::new(h(0), h(1), 1.0, 10.0);
        let fa = s.contribution(h(0), &positions).force;
        let fb = s.contribution(h(1), &positions).force;
        assert!(fa.y < 0.0);
        assert!(fb.y > 0.0);
        // Newton's third law: the two forces are equal and opposite.
        assert!((fa + fb).length() < 1e-5);
    }

    #[test]
    fn hessian_is_symmetric_and_psd_under_compression() {
        // Compressed spring (C < 0): raw transverse term is negative but the
        // projected Hessian must stay symmetric positive-semidefinite.
        let positions = [Vec3::ZERO, Vec3::new(0.0, -0.5, 0.0)];
        let s = SpringElement::new(h(0), h(1), 1.0, 10.0);
        let hh = s.contribution(h(0), &positions).hessian;
        // Symmetry.
        assert!((hh.col(0).y - hh.col(1).x).abs() < 1e-6);
        assert!((hh.col(0).z - hh.col(2).x).abs() < 1e-6);
        assert!((hh.col(1).z - hh.col(2).y).abs() < 1e-6);
        // PSD: quadratic form non-negative along a few probe directions.
        for probe in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::ONE.normalize()] {
            let q = probe.dot(hh * probe);
            assert!(q >= -1e-6, "quadratic form {q} negative");
        }
    }

    #[test]
    fn from_constraint_set_maps_compliance_to_stiffness() {
        use crate::soft::constraint::DistanceConstraint;
        let mut cs = ConstraintSet::new();
        cs.distance
            .push(DistanceConstraint::new(h(0), h(1), 1.0, 0.01));
        cs.distance
            .push(DistanceConstraint::new(h(1), h(2), 2.0, 0.0));
        let set = SpringSet::from_constraint_set(&cs, 5000.0);
        assert_eq!(set.len(), 2);
        assert!((set.springs[0].stiffness - 100.0).abs() < 1e-3);
        assert_eq!(set.springs[1].stiffness, 5000.0);
    }
}
