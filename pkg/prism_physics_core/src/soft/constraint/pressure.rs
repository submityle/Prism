//! Pressure constraint: closed-mesh volume preservation for inflatable cloth.
//!
//! A [`PressureConstraint`] keeps a *closed* triangle mesh at a target enclosed
//! volume by pushing every particle along the accumulated outward triangle
//! normals. It is the global, mesh-wide analogue of the per-tetrahedron
//! [`super::volume::TetraVolumeConstraint`]: instead of coupling four corners of
//! one tetra, it couples every vertex of a shell through the divergence-theorem
//! volume functional. This is what turns a garment shell into a balloon, an
//! inflated down jacket, an air bladder, or a billowing membrane — the same
//! pressure model shipped by Houdini `Vellum` and NVIDIA `NvCloth`, and
//! conceptually by the UE5 `Chaos` cloth aerodynamics slot, expressed here from
//! first principles without reusing their code.
//!
//! # Math
//!
//! The signed enclosed volume of a closed, outward-wound triangle mesh follows
//! from the divergence theorem as a sum of tetrahedra spanned from the origin:
//!
//! ```text
//! V = (1/6) * sum_tri  p0 . (p1 x p2)
//! ```
//!
//! The pressure constraint drives that volume toward a target `k * V_rest`,
//! where `k` (the overpressure) inflates (`k > 1`) or deflates (`k < 1`) the
//! shell:
//!
//! ```text
//! C = V - k * V_rest
//! ```
//!
//! Each triangle contributes a gradient to each of its three vertices, the
//! partial derivative of that triangle's tetra volume:
//!
//! ```text
//! dV/dp0 = (1/6)(p1 x p2)   dV/dp1 = (1/6)(p2 x p0)   dV/dp2 = (1/6)(p0 x p1)
//! ```
//!
//! Vertices shared by many triangles accumulate the sum of those contributions.
//! The compliant XPBD projection then mirrors every other constraint in this
//! module: with `alpha_tilde = compliance / dt^2`,
//!
//! ```text
//! delta_lambda = (-C - alpha_tilde * lambda) / (sum_i w_i |grad_i|^2 + alpha_tilde)
//! x_i         += w_i * delta_lambda * grad_i
//! ```
//!
//! Unlike a one-shot projection, the accumulated Lagrange multiplier `lambda`
//! makes the response stiffness-correct and iteration/step independent, exactly
//! like the tetra-volume constraint.
//!
//! Pinned particles (`inverse_mass <= 0`, weight `w = 0`) never move,
//! out-of-range triangle indices are skipped, a (near) zero denominator is a
//! no-op, and the whole routine is deterministic array-in / array-out math; no
//! transcendental functions are called and no `NaN` is produced from well-formed
//! input.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The signed
//! volume via the divergence theorem and the compliant XPBD volume projection
//! are standard, publicly documented position-based-dynamics techniques.

use alloc::vec;
use alloc::vec::Vec;

use glam::Vec3;

use crate::math::scalar::Real;

use super::{ParticleConstraint, SoftConstraintKind};

/// One sixth, the constant factor in the tetra-volume and gradient formulas.
const INV_SIX: Real = 1.0 / 6.0;

/// Minimum squared denominator below which a projection is skipped as
/// degenerate (a collapsed or near-zero-gradient mesh), avoiding division by a
/// vanishing value.
const EPS_LEN_SQ: Real = 1.0e-12;

/// Computes the signed enclosed volume of a closed triangle mesh.
///
/// `positions` is the vertex array and `triangles` indexes it with outward
/// winding (counter-clockwise seen from outside), so a well-formed shell yields
/// a positive volume. Triangles whose indices fall outside `positions` are
/// skipped rather than panicking, which keeps the function total for partial or
/// malformed meshes. The result is `0.0` for an empty mesh.
#[must_use]
pub fn mesh_volume(positions: &[Vec3], triangles: &[[u32; 3]]) -> Real {
    let mut sum = 0.0;
    for tri in triangles {
        let Some((p0, p1, p2)) = fetch_triangle_positions(positions, *tri) else {
            continue;
        };
        sum += p0.dot(p1.cross(p2));
    }
    sum * INV_SIX
}

/// A compliant XPBD constraint holding a closed triangle mesh at a target
/// enclosed volume.
///
/// The target volume is `overpressure * rest_volume`; an overpressure above one
/// inflates the shell, below one deflates it. `compliance` softens the
/// constraint exactly like every other constraint in this module: zero is
/// perfectly rigid, larger values yield a springier, slower volume response.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PressureConstraint {
    /// The outward-wound triangles of the closed shell, indexing the particle
    /// position array.
    pub triangles: Vec<[u32; 3]>,
    /// Reference (rest) enclosed volume the shell relaxes toward at unit
    /// overpressure, in cubic metres.
    pub rest_volume: Real,
    /// Multiplier applied to `rest_volume` to obtain the target volume; `> 1`
    /// inflates, `< 1` deflates.
    pub overpressure: Real,
    /// Compliance (inverse stiffness); `0` is perfectly rigid.
    pub compliance: Real,
    /// Accumulated Lagrange multiplier for the current substep.
    lambda: Real,
}

impl PressureConstraint {
    /// Creates a pressure constraint from a triangle list, rest volume,
    /// overpressure, and compliance.
    ///
    /// Inputs are defensively sanitised: the rest volume uses its magnitude (so
    /// a signed volume from [`mesh_volume`] may be passed straight in) and falls
    /// back to zero on `NaN`; the overpressure is clamped strictly positive so
    /// the target never collapses to or below zero and falls back to one on
    /// `NaN`; the compliance is clamped non-negative.
    #[must_use]
    pub fn new(
        triangles: Vec<[u32; 3]>,
        rest_volume: Real,
        overpressure: Real,
        compliance: Real,
    ) -> Self {
        PressureConstraint {
            triangles,
            rest_volume: sanitize_rest_volume(rest_volume),
            overpressure: sanitize_overpressure(overpressure),
            compliance: compliance.max(0.0),
            lambda: 0.0,
        }
    }

    /// Creates a pressure constraint whose rest volume is sampled from the
    /// current geometry in `positions`.
    #[must_use]
    pub fn from_positions(
        triangles: Vec<[u32; 3]>,
        positions: &[Vec3],
        overpressure: Real,
        compliance: Real,
    ) -> Self {
        let rest_volume = mesh_volume(positions, &triangles);
        PressureConstraint::new(triangles, rest_volume, overpressure, compliance)
    }

    /// The target enclosed volume `overpressure * rest_volume`.
    #[must_use]
    pub fn target_volume(&self) -> Real {
        self.overpressure * self.rest_volume
    }

    /// Returns the current accumulated Lagrange multiplier (for diagnostics and
    /// tests).
    #[must_use]
    pub fn lambda(&self) -> Real {
        self.lambda
    }
}

impl ParticleConstraint for PressureConstraint {
    fn kind(&self) -> SoftConstraintKind {
        SoftConstraintKind::Pressure
    }

    fn compliance(&self) -> Real {
        self.compliance
    }

    fn reset(&mut self) {
        self.lambda = 0.0;
    }

    fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real) {
        let count = positions.len();
        if dt <= 0.0 || count == 0 || self.triangles.is_empty() {
            return;
        }

        // Accumulate the signed volume and per-vertex gradient in one pass.
        let mut gradients: Vec<Vec3> = vec![Vec3::ZERO; count];
        let mut volume = 0.0;
        for tri in &self.triangles {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= count || i1 >= count || i2 >= count {
                continue;
            }
            let p0 = positions[i0];
            let p1 = positions[i1];
            let p2 = positions[i2];
            volume += p0.dot(p1.cross(p2));
            gradients[i0] += p1.cross(p2) * INV_SIX;
            gradients[i1] += p2.cross(p0) * INV_SIX;
            gradients[i2] += p0.cross(p1) * INV_SIX;
        }
        volume *= INV_SIX;

        let c = volume - self.target_volume();

        // Denominator: sum_i w_i |grad_i|^2 + alpha_tilde.
        let mut denom = 0.0;
        for (i, grad) in gradients.iter().enumerate() {
            let w = inverse_masses.get(i).copied().unwrap_or(0.0);
            if w <= 0.0 {
                continue;
            }
            denom += w * grad.length_squared();
        }
        let alpha_tilde = self.compliance / (dt * dt);
        denom += alpha_tilde;
        if denom < EPS_LEN_SQ {
            return;
        }

        let delta_lambda = (-c - alpha_tilde * self.lambda) / denom;
        self.lambda += delta_lambda;
        for (i, grad) in gradients.iter().enumerate() {
            let w = inverse_masses.get(i).copied().unwrap_or(0.0);
            if w <= 0.0 {
                continue;
            }
            positions[i] += *grad * (w * delta_lambda);
        }
    }
}

/// Clamps a rest volume to a safe, non-negative magnitude; `NaN` becomes zero.
#[must_use]
fn sanitize_rest_volume(rest_volume: Real) -> Real {
    if rest_volume.is_nan() {
        0.0
    } else {
        rest_volume.abs()
    }
}

/// Clamps an overpressure strictly positive; `NaN` becomes the neutral `1.0`.
#[must_use]
fn sanitize_overpressure(overpressure: Real) -> Real {
    if overpressure.is_nan() {
        1.0
    } else {
        overpressure.max(Real::MIN_POSITIVE)
    }
}

/// Fetches the three vertex positions of a triangle, or `None` when any index
/// is out of range.
#[must_use]
fn fetch_triangle_positions(positions: &[Vec3], tri: [u32; 3]) -> Option<(Vec3, Vec3, Vec3)> {
    let p0 = positions.get(tri[0] as usize)?;
    let p1 = positions.get(tri[1] as usize)?;
    let p2 = positions.get(tri[2] as usize)?;
    Some((*p0, *p1, *p2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for volume comparisons.
    const EPS: Real = 1.0e-4;

    /// The eight corners of the axis-aligned unit cube `[0,1]^3`.
    fn unit_cube_positions() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0), // 0
            Vec3::new(1.0, 0.0, 0.0), // 1
            Vec3::new(1.0, 1.0, 0.0), // 2
            Vec3::new(0.0, 1.0, 0.0), // 3
            Vec3::new(0.0, 0.0, 1.0), // 4
            Vec3::new(1.0, 0.0, 1.0), // 5
            Vec3::new(1.0, 1.0, 1.0), // 6
            Vec3::new(0.0, 1.0, 1.0), // 7
        ]
    }

    /// The twelve outward-wound triangles of the unit cube.
    fn unit_cube_triangles() -> Vec<[u32; 3]> {
        vec![
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 6, 2],
            [3, 7, 6], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ]
    }

    #[test]
    fn unit_cube_volume_is_one() {
        let volume = mesh_volume(&unit_cube_positions(), &unit_cube_triangles());
        assert!((volume - 1.0).abs() < EPS, "unit cube volume: {volume}");
    }

    #[test]
    fn empty_mesh_volume_is_zero() {
        assert_eq!(mesh_volume(&[], &[]), 0.0);
        let positions = unit_cube_positions();
        assert_eq!(mesh_volume(&positions, &[]), 0.0);
    }

    #[test]
    fn out_of_range_triangles_are_skipped_in_volume() {
        let positions = unit_cube_positions();
        let triangles = vec![[0, 2, 1], [0, 99, 2]];
        // Only the first (valid) triangle contributes.
        let expected = mesh_volume(&positions, &[[0, 2, 1]]);
        assert_eq!(mesh_volume(&positions, &triangles), expected);
    }

    #[test]
    fn new_sanitizes_inputs() {
        let c = PressureConstraint::new(vec![[0, 1, 2]], -3.0, -1.0, -5.0);
        assert_eq!(c.rest_volume, 3.0);
        assert_eq!(c.overpressure, Real::MIN_POSITIVE);
        assert_eq!(c.compliance, 0.0);

        let n = PressureConstraint::new(vec![[0, 1, 2]], Real::NAN, Real::NAN, 0.0);
        assert_eq!(n.rest_volume, 0.0);
        assert_eq!(n.overpressure, 1.0);
    }

    #[test]
    fn from_positions_captures_rest_volume() {
        let positions = unit_cube_positions();
        let c = PressureConstraint::from_positions(unit_cube_triangles(), &positions, 1.0, 0.0);
        assert!((c.rest_volume - 1.0).abs() < EPS, "rest {}", c.rest_volume);
        assert!((c.target_volume() - 1.0).abs() < EPS);
    }

    #[test]
    fn overpressure_inflates_the_shell() {
        let mut positions = unit_cube_positions();
        let triangles = unit_cube_triangles();
        let rest = mesh_volume(&positions, &triangles);
        let mut c = PressureConstraint::new(triangles.clone(), rest, 2.0, 0.0);
        let inv = vec![1.0; positions.len()];
        for _ in 0..32 {
            c.reset();
            c.project(&mut positions, &inv, 1.0 / 60.0);
        }
        let inflated = mesh_volume(&positions, &triangles);
        assert!(inflated > rest + 0.1, "inflated {inflated} vs rest {rest}");
    }

    #[test]
    fn underpressure_deflates_the_shell() {
        let mut positions = unit_cube_positions();
        let triangles = unit_cube_triangles();
        let rest = mesh_volume(&positions, &triangles);
        let mut c = PressureConstraint::new(triangles.clone(), rest, 0.5, 0.0);
        let inv = vec![1.0; positions.len()];
        for _ in 0..32 {
            c.reset();
            c.project(&mut positions, &inv, 1.0 / 60.0);
        }
        let deflated = mesh_volume(&positions, &triangles);
        assert!(deflated < rest - 0.1, "deflated {deflated} vs rest {rest}");
    }

    #[test]
    fn pinned_particles_do_not_move() {
        let mut positions = unit_cube_positions();
        let triangles = unit_cube_triangles();
        let rest = mesh_volume(&positions, &triangles);
        let snapshot = positions.clone();
        let mut c = PressureConstraint::new(triangles, rest, 4.0, 0.0);
        let inv = vec![0.0; positions.len()];
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions, snapshot);
    }

    #[test]
    fn empty_triangles_is_inert() {
        let mut positions = unit_cube_positions();
        let snapshot = positions.clone();
        let mut c = PressureConstraint::new(Vec::new(), 1.0, 2.0, 0.0);
        let inv = vec![1.0; positions.len()];
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions, snapshot);
    }

    #[test]
    fn non_positive_dt_is_inert() {
        let mut positions = unit_cube_positions();
        let triangles = unit_cube_triangles();
        let snapshot = positions.clone();
        let mut c = PressureConstraint::new(triangles, 1.0, 2.0, 0.0);
        let inv = vec![1.0; positions.len()];
        c.project(&mut positions, &inv, 0.0);
        assert_eq!(positions, snapshot);
    }

    #[test]
    fn out_of_range_triangles_are_skipped_in_projection() {
        // All triangles reference missing vertices: zero gradient everywhere,
        // so the projection is a no-op even with a volume error.
        let mut positions = unit_cube_positions();
        let snapshot = positions.clone();
        let mut c = PressureConstraint::new(vec![[0, 1, 99]], 1.0, 2.0, 0.0);
        let inv = vec![1.0; positions.len()];
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions, snapshot);
    }

    #[test]
    fn compliant_is_softer_than_rigid() {
        let triangles = unit_cube_triangles();
        let rest = mesh_volume(&unit_cube_positions(), &triangles);
        let inv = vec![1.0; 8];

        let mut rigid_pos = unit_cube_positions();
        let mut rigid = PressureConstraint::new(triangles.clone(), rest, 2.0, 0.0);
        rigid.reset();
        rigid.project(&mut rigid_pos, &inv, 1.0 / 60.0);
        let rigid_vol = mesh_volume(&rigid_pos, &triangles);

        let mut soft_pos = unit_cube_positions();
        let mut soft = PressureConstraint::new(triangles.clone(), rest, 2.0, 1.0e-2);
        soft.reset();
        soft.project(&mut soft_pos, &inv, 1.0 / 60.0);
        let soft_vol = mesh_volume(&soft_pos, &triangles);

        // Both inflate toward 2*rest; the compliant one moves less in one step.
        assert!(
            (rigid_vol - rest) >= (soft_vol - rest),
            "rigid {rigid_vol} soft {soft_vol} rest {rest}"
        );
    }

    #[test]
    fn kind_and_compliance_are_reported() {
        let c = PressureConstraint::new(vec![[0, 1, 2]], 1.0, 1.0, 0.25);
        assert_eq!(c.kind(), SoftConstraintKind::Pressure);
        assert_eq!(c.compliance(), 0.25);
    }

    #[test]
    fn reset_clears_accumulated_lambda() {
        let mut positions = unit_cube_positions();
        let triangles = unit_cube_triangles();
        let rest = mesh_volume(&positions, &triangles);
        let mut c = PressureConstraint::new(triangles, rest, 2.0, 0.0);
        let inv = vec![1.0; positions.len()];
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert!(c.lambda() != 0.0);
        c.reset();
        assert_eq!(c.lambda(), 0.0);
    }
}
