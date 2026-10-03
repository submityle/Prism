//! Position-based volume (incompressibility) projection for tetrahedral meshes.
//!
//! Many soft materials — flesh, rubber, muscle — are nearly incompressible, yet
//! a displacement-based FEM solved to a loose tolerance, or a position-based
//! solver using soft constraints, lets element volumes drift. This module is a
//! geometric post-projection that drives each tetrahedron's signed volume back
//! toward a target multiple of its rest volume, independent of the constitutive
//! model or the integrator.
//!
//! For a tetrahedron with vertices `p0..p3` the signed volume is
//! `V = (1/6)·(p1−p0)·((p2−p0)×(p3−p0))`. We enforce the scalar constraint
//! `C = V − ratio·V₀` with the standard position-based-dynamics gradient
//! projection (Müller et al.):
//!
//! ```text
//! ∇₁C = (1/6)(p2−p0)×(p3−p0)   ∇₂C = (1/6)(p3−p0)×(p1−p0)
//! ∇₃C = (1/6)(p1−p0)×(p2−p0)   ∇₀C = −(∇₁C + ∇₂C + ∇₃C)
//! λ   = C / Σ wⱼ |∇ⱼC|²        Δpᵢ = −stiffness·λ·wᵢ·∇ᵢC
//! ```
//!
//! Because `Σᵢ ∇ᵢC = 0`, the inverse-mass-weighted correction satisfies
//! `Σᵢ (1/wᵢ)Δpᵢ = 0`, so the mass-weighted centre of mass — and hence linear
//! momentum — is conserved exactly for the free vertices. Vertices shared by
//! several elements accumulate their corrections in a Jacobi sweep that can be
//! iterated; pinned vertices (inverse mass `0`) never move.
//!
//! The module owns no simulation state and performs no time integration; it is
//! a pure geometric projection usable before or after any step. `stiffness`
//! plays the role of a relaxation factor (full PBD at `1.0`); XPBD compliance
//! is intentionally left to the stepping layer, which owns `dt` and the
//! per-constraint multipliers.
//!
//! # Attribution
//!
//! Clean-room implementation of the PBD tetrahedral volume constraint. No
//! Unreal Engine source or derived code.

use super::tet_fem_basis::TetFemBasis;
use glam::Vec3;

/// Parameters for [`project_volume`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeProjectionParams {
    /// Target volume as a multiple of each element's signed rest volume.
    /// `1.0` restores the rest volume; `>1` inflates, `<1` deflates. Must be
    /// finite and `> 0`.
    pub target_ratio: f32,
    /// Relaxation factor in `(0, 1]`: the fraction of the full PBD correction
    /// applied each sweep.
    pub stiffness: f32,
    /// Number of Jacobi sweeps; must be `>= 1`.
    pub iterations: u32,
}

impl VolumeProjectionParams {
    /// Builds validated parameters.
    ///
    /// Returns `None` unless `target_ratio` is finite and `> 0`, `stiffness` is
    /// in `(0, 1]`, and `iterations >= 1`.
    #[must_use]
    pub fn new(target_ratio: f32, stiffness: f32, iterations: u32) -> Option<Self> {
        if !target_ratio.is_finite()
            || target_ratio <= 0.0
            || !stiffness.is_finite()
            || stiffness <= 0.0
            || stiffness > 1.0
            || iterations == 0
        {
            return None;
        }
        Some(Self {
            target_ratio,
            stiffness,
            iterations,
        })
    }

    /// Full-stiffness restoration to the exact rest volume.
    #[must_use]
    pub fn incompressible(iterations: u32) -> Option<Self> {
        Self::new(1.0, 1.0, iterations)
    }
}

/// Outcome of a volume projection, measured on the first sweep before any
/// correction is applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeProjectionReport {
    /// Largest relative volume error `|V/V₀ − target_ratio|` across elements on
    /// the first sweep. `0.0` means every element already matched the target.
    pub max_volume_error: f32,
    /// Number of elements whose first-sweep volume error exceeded `tol`.
    pub projected_elements: usize,
}

/// Relative volume error below which an element is treated as already on
/// target (and excluded from the projected-element count).
const VOLUME_TOL: f32 = 1.0e-6;

/// Projects every element's signed volume toward `ratio·V₀`, mutating
/// `positions` in place.
///
/// `tets` maps each basis element to its four vertex indices into `positions`.
/// `inv_mass`, when supplied, gives a per-vertex inverse mass used both for the
/// PBD weighting and to pin vertices (`0` = pinned). `pinned`, when supplied,
/// additionally forces the listed vertices to stay fixed.
///
/// Returns `None` on any dimension mismatch (`tets.len()` differs from the
/// element count, an out-of-range tet index, or an `inv_mass`/`pinned` slice
/// whose length differs from `positions`). Returns the first-sweep
/// [`VolumeProjectionReport`] otherwise.
#[must_use]
pub fn project_volume(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &VolumeProjectionParams,
) -> Option<VolumeProjectionReport> {
    if tets.len() != basis.elements.len() {
        return None;
    }
    let n = positions.len();
    if let Some(w) = inv_mass
        && w.len() != n
    {
        return None;
    }
    if let Some(p) = pinned
        && p.len() != n
    {
        return None;
    }
    for tet in tets {
        if tet.iter().any(|&vi| vi as usize >= n) {
            return None;
        }
    }

    let is_pinned = |v: usize| -> bool {
        pinned.is_some_and(|p| p[v]) || inv_mass.is_some_and(|w| w[v] <= 0.0)
    };
    // Inverse mass used for the PBD weighting; pinned vertices contribute 0.
    let weight = |v: usize| -> f32 {
        if is_pinned(v) {
            0.0
        } else {
            inv_mass.map_or(1.0, |w| w[v])
        }
    };

    let mut report = VolumeProjectionReport {
        max_volume_error: 0.0,
        projected_elements: 0,
    };
    let mut accum = vec![[0.0f32; 3]; n];
    let mut counts = vec![0u32; n];

    for sweep in 0..params.iterations {
        for a in accum.iter_mut() {
            *a = [0.0; 3];
        }
        for c in counts.iter_mut() {
            *c = 0;
        }

        for (element, tet) in basis.elements.iter().zip(tets.iter()) {
            let [i0, i1, i2, i3] = tet.map(|v| v as usize);
            let p0 = positions[i0];
            let p1 = positions[i1];
            let p2 = positions[i2];
            let p3 = positions[i3];

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let e3 = p3 - p0;
            let volume = e1.dot(e2.cross(e3)) / 6.0;
            let rest = element.rest_volume;
            let target = params.target_ratio * rest;
            let c = volume - target;

            if sweep == 0 && rest.abs() > f32::MIN_POSITIVE {
                let error = ((volume / rest) - params.target_ratio).abs();
                if error > VOLUME_TOL {
                    report.projected_elements += 1;
                    report.max_volume_error = report.max_volume_error.max(error);
                }
            }

            // PBD gradients of the signed-volume constraint.
            let g1 = e2.cross(e3) / 6.0;
            let g2 = e3.cross(e1) / 6.0;
            let g3 = e1.cross(e2) / 6.0;
            let g0 = -(g1 + g2 + g3);

            let w0 = weight(i0);
            let w1 = weight(i1);
            let w2 = weight(i2);
            let w3 = weight(i3);
            let denom = w0 * g0.length_squared()
                + w1 * g1.length_squared()
                + w2 * g2.length_squared()
                + w3 * g3.length_squared();
            if denom <= f32::MIN_POSITIVE {
                continue;
            }
            let lambda = c / denom;
            let scale = -params.stiffness * lambda;

            let verts = [i0, i1, i2, i3];
            let ws = [w0, w1, w2, w3];
            let grads = [g0, g1, g2, g3];
            for k in 0..4 {
                if ws[k] == 0.0 {
                    continue;
                }
                let d = scale * ws[k] * grads[k];
                let v = verts[k];
                accum[v][0] += d.x;
                accum[v][1] += d.y;
                accum[v][2] += d.z;
                counts[v] += 1;
            }
        }

        for v in 0..n {
            if counts[v] == 0 {
                continue;
            }
            let inv = 1.0 / counts[v] as f32;
            positions[v] += Vec3::new(accum[v][0] * inv, accum[v][1] * inv, accum[v][2] * inv);
        }
    }

    Some(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};

    fn single_tet() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let tets = vec![[0u32, 1, 2, 3]];
        (verts, tets)
    }

    fn basis_of(verts: &[Vec3], tets: &[[u32; 4]]) -> TetFemBasis {
        build_tet_fem_basis(verts, tets, &TetFemBasisParams::default()).unwrap()
    }

    fn signed_volume(pos: &[Vec3], tet: [u32; 4]) -> f32 {
        let [i0, i1, i2, i3] = tet.map(|v| v as usize);
        let e1 = pos[i1] - pos[i0];
        let e2 = pos[i2] - pos[i0];
        let e3 = pos[i3] - pos[i0];
        e1.dot(e2.cross(e3)) / 6.0
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(VolumeProjectionParams::new(0.0, 1.0, 1).is_none());
        assert!(VolumeProjectionParams::new(1.0, 0.0, 1).is_none());
        assert!(VolumeProjectionParams::new(1.0, 1.5, 1).is_none());
        assert!(VolumeProjectionParams::new(1.0, 1.0, 0).is_none());
        assert!(VolumeProjectionParams::new(f32::INFINITY, 1.0, 1).is_none());
        assert!(VolumeProjectionParams::incompressible(1).is_some());
    }

    #[test]
    fn rest_pose_is_left_untouched() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        let params = VolumeProjectionParams::incompressible(3).unwrap();
        let report = project_volume(&basis, &tets, &mut pos, None, None, &params).unwrap();
        assert_eq!(report.projected_elements, 0);
        assert_eq!(report.max_volume_error, 0.0);
        for (a, b) in pos.iter().zip(verts.iter()) {
            assert!((*a - *b).length() < 1e-6);
        }
    }

    #[test]
    fn inflated_element_is_restored_to_rest_volume() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let v0 = signed_volume(&verts, tets[0]);
        // Scale the whole tet by 1.3 about the origin -> volume x 1.3^3.
        let mut pos: Vec<Vec3> = verts.iter().map(|&p| p * 1.3).collect();
        let params = VolumeProjectionParams::incompressible(40).unwrap();
        let report = project_volume(&basis, &tets, &mut pos, None, None, &params).unwrap();
        assert_eq!(report.projected_elements, 1);
        assert!(
            report.max_volume_error > 1.0,
            "error = {}",
            report.max_volume_error
        );
        let after = signed_volume(&pos, tets[0]);
        assert!(
            (after - v0).abs() < 1e-3 * v0.abs(),
            "after = {after}, rest = {v0}"
        );
    }

    #[test]
    fn deflated_element_is_restored_to_rest_volume() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let v0 = signed_volume(&verts, tets[0]);
        let mut pos: Vec<Vec3> = verts.iter().map(|&p| p * 0.6).collect();
        let params = VolumeProjectionParams::incompressible(60).unwrap();
        project_volume(&basis, &tets, &mut pos, None, None, &params).unwrap();
        let after = signed_volume(&pos, tets[0]);
        assert!(
            (after - v0).abs() < 1e-3 * v0.abs(),
            "after = {after}, rest = {v0}"
        );
    }

    #[test]
    fn target_ratio_inflates_to_requested_multiple() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let v0 = signed_volume(&verts, tets[0]);
        let mut pos = verts.clone();
        let params = VolumeProjectionParams::new(1.5, 1.0, 60).unwrap();
        project_volume(&basis, &tets, &mut pos, None, None, &params).unwrap();
        let after = signed_volume(&pos, tets[0]);
        assert!(
            (after - 1.5 * v0).abs() < 1e-3 * v0.abs(),
            "after = {after}, want {}",
            1.5 * v0
        );
    }

    #[test]
    fn uniform_mass_projection_preserves_centroid() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos: Vec<Vec3> = verts.iter().map(|&p| p * 1.3).collect();
        let before: Vec3 = pos.iter().copied().sum::<Vec3>() / pos.len() as f32;
        let params = VolumeProjectionParams::incompressible(10).unwrap();
        project_volume(&basis, &tets, &mut pos, None, None, &params).unwrap();
        let after: Vec3 = pos.iter().copied().sum::<Vec3>() / pos.len() as f32;
        assert!(
            (after - before).length() < 1e-5,
            "centroid drift = {}",
            (after - before).length()
        );
    }

    #[test]
    fn pinned_vertex_does_not_move() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let v0 = signed_volume(&verts, tets[0]);
        let mut pos: Vec<Vec3> = verts.iter().map(|&p| p * 1.3).collect();
        let pinned = vec![true, false, false, false];
        let params = VolumeProjectionParams::incompressible(60).unwrap();
        project_volume(&basis, &tets, &mut pos, None, Some(&pinned), &params).unwrap();
        assert!(
            (pos[0] - verts[0] * 1.3).length() < 1e-6,
            "pinned moved to {:?}",
            pos[0]
        );
        let after = signed_volume(&pos, tets[0]);
        assert!(
            (after - v0).abs() < 5e-3 * v0.abs(),
            "after = {after}, rest = {v0}"
        );
    }

    #[test]
    fn dimension_mismatch_returns_none() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        let params = VolumeProjectionParams::incompressible(1).unwrap();
        let bad_inv = vec![1.0; 3];
        assert!(project_volume(&basis, &tets, &mut pos, Some(&bad_inv), None, &params).is_none());
        let bad_pin = vec![false; 3];
        assert!(project_volume(&basis, &tets, &mut pos, None, Some(&bad_pin), &params).is_none());
        let bad_tets = vec![[0u32, 1, 2, 3], [0, 1, 2, 3]];
        assert!(project_volume(&basis, &bad_tets, &mut pos, None, None, &params).is_none());
    }
}
