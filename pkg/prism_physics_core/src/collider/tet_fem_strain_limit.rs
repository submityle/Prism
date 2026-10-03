//! Position-based strain limiting for tetrahedral FEM meshes.
//!
//! Explicit and even implicit integrators can let individual elements stretch
//! or compress far beyond what a material should physically allow, especially
//! under stiff constraints solved to a loose tolerance or under large time
//! steps. Strain limiting is a geometric post-projection that clamps every
//! element's principal stretches into a user chosen band `[min_stretch,
//! max_stretch]` without changing the constitutive model or the integrator.
//!
//! For each element we form the deformation gradient `F = Ds · Dm⁻¹`, take its
//! signed SVD `F = U Σ Vᵀ`, clamp the singular-value magnitudes into the band
//! (keeping their sign so an inverted element is not force-flipped), and
//! rebuild a target gradient `F* = U Σ' Vᵀ`. The target edge matrix
//! `Ds* = F* · Dm` fixes the four vertices up to a rigid translation; we pick
//! that translation so the element's mass-weighted centre of mass is preserved,
//! which conserves linear momentum for the free (unpinned) vertices.
//!
//! Because a vertex is shared by several elements, the per-element corrections
//! are accumulated and averaged in a Jacobi sweep and the whole projection can
//! be iterated. Pinned vertices (inverse mass `0`) never move and bias each
//! element's anchor so neighbouring free vertices absorb the correction.
//!
//! The module owns no simulation state and performs no time integration; it is
//! a pure geometric projection that can run before or after any step. The SVD
//! is the crate's own [`svd3`]; nothing here is derived from Unreal Engine
//! source.
//!
//! # Attribution
//!
//! Clean-room implementation of SVD-clamp strain limiting (Müller et al.,
//! "Strain Based Dynamics"). No Unreal Engine source or derived code.

use super::tet_fem_basis::TetFemBasis;
use crate::mpm::svd3;
use glam::{Mat3, Vec3};

/// The admissible band of principal stretches enforced by
/// [`project_strain_limits`].
///
/// A stretch of `1.0` is the rest length; `min_stretch` caps compression and
/// `max_stretch` caps extension. `min_stretch == max_stretch == 1.0` makes each
/// element rigid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrainLimitParams {
    /// Lower bound on the principal-stretch magnitude. Must satisfy
    /// `0 < min_stretch <= max_stretch`.
    pub min_stretch: f32,
    /// Upper bound on the principal-stretch magnitude.
    pub max_stretch: f32,
    /// Number of Jacobi sweeps. More sweeps propagate corrections through
    /// shared vertices; one sweep is often enough for mild violations.
    pub iterations: u32,
}

impl StrainLimitParams {
    /// Builds a validated band.
    ///
    /// Returns `None` unless `min` and `max` are finite with
    /// `0 < min <= max` and `iterations >= 1`.
    #[must_use]
    pub fn new(min: f32, max: f32, iterations: u32) -> Option<Self> {
        if !min.is_finite() || !max.is_finite() || min <= 0.0 || max < min || iterations == 0 {
            return None;
        }
        Some(Self {
            min_stretch: min,
            max_stretch: max,
            iterations,
        })
    }

    /// A band symmetric about the rest length: `[1 - pct, 1 + pct]`.
    ///
    /// Returns `None` unless `0 < pct < 1` and `iterations >= 1`.
    #[must_use]
    pub fn symmetric(pct: f32, iterations: u32) -> Option<Self> {
        if !pct.is_finite() || pct <= 0.0 || pct >= 1.0 {
            return None;
        }
        Self::new(1.0 - pct, 1.0 + pct, iterations)
    }
}

/// Outcome of a strain-limiting projection, measured on the first sweep before
/// any correction is applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrainLimitReport {
    /// Largest band violation across all elements, i.e. the maximum over
    /// elements of `max(σ_max − max_stretch, min_stretch − σ_min, 0)` using
    /// singular-value magnitudes. `0.0` means every element was already inside
    /// the band.
    pub max_violation: f32,
    /// Number of elements that were outside the band on the first sweep.
    pub projected_elements: usize,
}

/// Clamps `|value|` into `[min, max]` while preserving its sign.
#[inline]
fn clamp_magnitude(value: f32, min: f32, max: f32) -> f32 {
    let mag = value.abs().clamp(min, max);
    if value < 0.0 {
        -mag
    } else {
        mag
    }
}

/// Projects every element's principal stretches into the band in `params`,
/// mutating `positions` in place.
///
/// `tets` maps each basis element to its four vertex indices into `positions`.
/// `inv_mass`, when supplied, gives a per-vertex inverse mass used to weight the
/// momentum-preserving anchor; `None` treats every vertex as unit mass. A zero
/// inverse mass marks a pinned vertex that never moves. `pinned`, when
/// supplied, additionally forces the listed vertices to stay fixed regardless
/// of their inverse mass.
///
/// Returns `None` on any dimension mismatch (`tets.len()` differs from the
/// element count, an out-of-range tet index, or an `inv_mass`/`pinned` slice
/// whose length differs from `positions`). Returns the first-sweep
/// [`StrainLimitReport`] otherwise.
#[must_use]
pub fn project_strain_limits(
    basis: &TetFemBasis,
    tets: &[[u32; 4]],
    positions: &mut [Vec3],
    inv_mass: Option<&[f32]>,
    pinned: Option<&[bool]>,
    params: &StrainLimitParams,
) -> Option<StrainLimitReport> {
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

    // Effective per-vertex mass for the anchor weighting; pinned vertices get a
    // large but finite anchor weight so neighbouring free vertices move
    // instead. A free vertex with inverse mass `w` has mass `1/w`.
    let is_pinned = |v: usize| -> bool {
        let pin_flag = pinned.is_some_and(|p| p[v]);
        let zero_inv = inv_mass.is_some_and(|w| w[v] <= 0.0);
        pin_flag || zero_inv
    };
    let anchor_mass = |v: usize| -> f32 {
        if is_pinned(v) {
            PIN_ANCHOR_MASS
        } else {
            match inv_mass {
                Some(w) => 1.0 / w[v],
                None => 1.0,
            }
        }
    };

    let mut report = StrainLimitReport {
        max_violation: 0.0,
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
            let p = [positions[i0], positions[i1], positions[i2], positions[i3]];
            let f = element.deformation_gradient(p[0], p[1], p[2], p[3]);
            let svd = svd3(f);
            let sigma = svd.sigma;

            if sweep == 0 {
                let mags = [sigma.x.abs(), sigma.y.abs(), sigma.z.abs()];
                let max_sigma = mags[0].max(mags[1]).max(mags[2]);
                let min_sigma = mags[0].min(mags[1]).min(mags[2]);
                let over = (max_sigma - params.max_stretch).max(0.0);
                let under = (params.min_stretch - min_sigma).max(0.0);
                let violation = over.max(under);
                if violation > 0.0 {
                    report.projected_elements += 1;
                    report.max_violation = report.max_violation.max(violation);
                }
            }

            let clamped = Vec3::new(
                clamp_magnitude(sigma.x, params.min_stretch, params.max_stretch),
                clamp_magnitude(sigma.y, params.min_stretch, params.max_stretch),
                clamp_magnitude(sigma.z, params.min_stretch, params.max_stretch),
            );
            // Skip untouched elements to avoid injecting round-off drift.
            if (clamped - sigma).abs().max_element() <= f32::EPSILON {
                continue;
            }

            // Target gradient and target edge matrix Ds* = F* · Dm.
            let f_star = svd.u * Mat3::from_diagonal(clamped) * svd.v.transpose();
            let dm = element.dm_inverse.inverse();
            let ds_star = f_star * dm;

            // Target positions anchored (temporarily) at vertex 0.
            let targets = [
                p[0],
                p[0] + ds_star.x_axis,
                p[0] + ds_star.y_axis,
                p[0] + ds_star.z_axis,
            ];

            // Momentum-preserving rigid translation: choose t so the
            // mass-weighted centre of mass of the element is unchanged.
            let masses = [
                anchor_mass(i0),
                anchor_mass(i1),
                anchor_mass(i2),
                anchor_mass(i3),
            ];
            let mut m_sum = 0.0f64;
            let mut weighted = [0.0f64; 3];
            for k in 0..4 {
                let d = targets[k] - p[k];
                let m = f64::from(masses[k]);
                m_sum += m;
                weighted[0] += m * f64::from(d.x);
                weighted[1] += m * f64::from(d.y);
                weighted[2] += m * f64::from(d.z);
            }
            let t = if m_sum > 0.0 {
                Vec3::new(
                    -(weighted[0] / m_sum) as f32,
                    -(weighted[1] / m_sum) as f32,
                    -(weighted[2] / m_sum) as f32,
                )
            } else {
                Vec3::ZERO
            };

            let verts = [i0, i1, i2, i3];
            for k in 0..4 {
                let v = verts[k];
                if is_pinned(v) {
                    continue;
                }
                let d = targets[k] - p[k] + t;
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

/// Anchor mass assigned to pinned vertices: large enough to pin the element's
/// centre of mass to them while remaining finite for the weighted average.
const PIN_ANCHOR_MASS: f32 = 1.0e12;

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

    fn max_stretch_of(basis: &TetFemBasis, tets: &[[u32; 4]], pos: &[Vec3]) -> f32 {
        let mut m = 0.0f32;
        for (e, t) in basis.elements.iter().zip(tets) {
            let [i0, i1, i2, i3] = t.map(|v| v as usize);
            let f = e.deformation_gradient(pos[i0], pos[i1], pos[i2], pos[i3]);
            let s = svd3(f).sigma;
            m = m.max(s.x.abs()).max(s.y.abs()).max(s.z.abs());
        }
        m
    }

    fn min_stretch_of(basis: &TetFemBasis, tets: &[[u32; 4]], pos: &[Vec3]) -> f32 {
        let mut m = f32::INFINITY;
        for (e, t) in basis.elements.iter().zip(tets) {
            let [i0, i1, i2, i3] = t.map(|v| v as usize);
            let f = e.deformation_gradient(pos[i0], pos[i1], pos[i2], pos[i3]);
            let s = svd3(f).sigma;
            m = m.min(s.x.abs()).min(s.y.abs()).min(s.z.abs());
        }
        m
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(StrainLimitParams::new(0.0, 1.0, 1).is_none());
        assert!(StrainLimitParams::new(1.2, 1.0, 1).is_none());
        assert!(StrainLimitParams::new(0.9, 1.1, 0).is_none());
        assert!(StrainLimitParams::new(f32::NAN, 1.0, 1).is_none());
        assert!(StrainLimitParams::symmetric(0.0, 1).is_none());
        assert!(StrainLimitParams::symmetric(1.0, 1).is_none());
        assert!(StrainLimitParams::symmetric(0.1, 1).is_some());
    }

    #[test]
    fn rest_pose_is_left_untouched() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        let params = StrainLimitParams::symmetric(0.1, 2).unwrap();
        let report = project_strain_limits(&basis, &tets, &mut pos, None, None, &params).unwrap();
        assert_eq!(report.projected_elements, 0);
        assert!(report.max_violation == 0.0);
        for (a, b) in pos.iter().zip(verts.iter()) {
            assert!((*a - *b).length() < 1e-6);
        }
    }

    #[test]
    fn over_stretched_element_is_pulled_back_into_band() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        // Stretch 2x along x: principal stretch 2.0, band caps at 1.1.
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.0, 0.0, 0.0);
        let params = StrainLimitParams::symmetric(0.1, 20).unwrap();
        let report = project_strain_limits(&basis, &tets, &mut pos, None, None, &params).unwrap();
        assert_eq!(report.projected_elements, 1);
        assert!(
            report.max_violation > 0.8,
            "violation = {}",
            report.max_violation
        );
        let after = max_stretch_of(&basis, &tets, &pos);
        assert!(after <= 1.1 + 1e-2, "max stretch after = {after}");
    }

    #[test]
    fn compressed_element_is_pushed_back_into_band() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        pos[1] = Vec3::new(0.4, 0.0, 0.0);
        let params = StrainLimitParams::symmetric(0.1, 20).unwrap();
        let report = project_strain_limits(&basis, &tets, &mut pos, None, None, &params).unwrap();
        assert_eq!(report.projected_elements, 1);
        let after = min_stretch_of(&basis, &tets, &pos);
        assert!(after >= 0.9 - 1e-2, "min stretch after = {after}");
    }

    #[test]
    fn uniform_mass_projection_preserves_centroid() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.2, 0.1, -0.1);
        let before: Vec3 = pos.iter().copied().sum::<Vec3>() / pos.len() as f32;
        let params = StrainLimitParams::symmetric(0.1, 1).unwrap();
        project_strain_limits(&basis, &tets, &mut pos, None, None, &params).unwrap();
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
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.0, 0.0, 0.0);
        let pinned = vec![true, false, false, false];
        let params = StrainLimitParams::symmetric(0.1, 10).unwrap();
        project_strain_limits(&basis, &tets, &mut pos, None, Some(&pinned), &params).unwrap();
        assert!(
            (pos[0] - verts[0]).length() < 1e-6,
            "pinned vertex moved to {:?}",
            pos[0]
        );
        let after = max_stretch_of(&basis, &tets, &pos);
        assert!(after <= 1.1 + 2e-2, "max stretch after = {after}");
    }

    #[test]
    fn zero_inverse_mass_pins_vertex() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        pos[1] = Vec3::new(2.0, 0.0, 0.0);
        let inv_mass = vec![0.0, 1.0, 1.0, 1.0];
        let params = StrainLimitParams::symmetric(0.1, 10).unwrap();
        project_strain_limits(&basis, &tets, &mut pos, Some(&inv_mass), None, &params).unwrap();
        assert!(
            (pos[0] - verts[0]).length() < 1e-6,
            "pinned vertex moved to {:?}",
            pos[0]
        );
    }

    #[test]
    fn dimension_mismatch_returns_none() {
        let (verts, tets) = single_tet();
        let basis = basis_of(&verts, &tets);
        let mut pos = verts.clone();
        let params = StrainLimitParams::symmetric(0.1, 1).unwrap();
        let bad_inv = vec![1.0; 3];
        assert!(
            project_strain_limits(&basis, &tets, &mut pos, Some(&bad_inv), None, &params).is_none()
        );
        let bad_pin = vec![false; 3];
        assert!(
            project_strain_limits(&basis, &tets, &mut pos, None, Some(&bad_pin), &params).is_none()
        );
        let bad_tets = vec![[0u32, 1, 2, 3], [0, 1, 2, 3]];
        assert!(project_strain_limits(&basis, &bad_tets, &mut pos, None, None, &params).is_none());
    }
}
