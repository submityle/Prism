//! Ergonomic tetrahedral FEM body: an owned bundle of the immutable simulation
//! data and the mutable kinematic state that drives the implicit corotational
//! integrator.
//!
//! The standalone FEM building blocks ([`step_implicit_corotational`] and the
//! assemblies it calls) take their mesh, material, mass, rest/current/velocity
//! state, external load and boundary mask as a long list of borrowed slices.
//! That is the right shape for the numerical kernel, but it pushes a lot of
//! bookkeeping onto callers: they must keep the rest pose, the current pose, the
//! velocities and the pin mask in lockstep, re-pass all of them every frame, and
//! copy the integrator's result back into their own buffers.
//!
//! [`TetFemBody`] owns that state once and exposes a small, hard-to-misuse API:
//! construct it from a mesh and material, then call [`TetFemBody::step`] (or
//! [`TetFemBody::substep`] for sub-stepping within a frame) with just the
//! per-vertex external force and the solver parameters. The body advances its
//! own positions and velocities in place and hands back the linear-solve
//! diagnostics. Dirichlet pins, per-vertex kinematic edits, and the usual
//! conserved quantities (total mass, kinetic energy, linear momentum, centre of
//! mass) are all available without reaching into the raw buffers.
//!
//! This is a thin ergonomic wrapper over standard finite-element constructions;
//! it holds no new numerics of its own. This file contains no Unreal Engine
//! source or derived code.

use super::tet_fem_basis::TetFemBasis;
use super::tet_fem_cg::CgReport;
use super::tet_fem_integrator::{step_implicit_corotational, ImplicitStepParams};
use super::tet_fem_stiffness::IsotropicElasticity;
use super::tet_lumped_mass::LumpedMass;
use glam::Vec3;

/// An owned tetrahedral finite-element body ready to be stepped in time.
///
/// Construction validates that the mesh, mass and material are mutually
/// consistent and invertible, so every subsequent [`TetFemBody::step`] call only
/// has to validate the per-frame external force. The current pose starts at the
/// rest pose with zero velocity and no pinned vertices.
#[derive(Clone, Debug)]
pub struct TetFemBody {
    /// Rest-pose finite-element basis, one element per tetrahedron.
    basis: TetFemBasis,
    /// Tetrahedron connectivity (vertex indices), parallel to `basis.elements`.
    tets: Vec<[u32; 4]>,
    /// Isotropic material shared by every element.
    material: IsotropicElasticity,
    /// Lumped (diagonal) mass matrix over `3 N` degrees of freedom.
    mass: LumpedMass,
    /// Rest (reference) positions; the corotational force vanishes here.
    rest: Vec<Vec3>,
    /// Current positions `x_n`.
    positions: Vec<Vec3>,
    /// Current velocities `v_n`.
    velocities: Vec<Vec3>,
    /// Dirichlet mask; a `true` entry freezes that vertex (`v = 0`, `x` held).
    pinned: Vec<bool>,
    /// Whether any entry of `pinned` is currently `true` (cached to avoid a
    /// scan every step).
    any_pinned: bool,
}

impl TetFemBody {
    /// Builds a body from a rest mesh, material and lumped mass.
    ///
    /// Returns `None` unless the inputs are mutually consistent:
    /// - the mesh is non-empty (`N ≥ 1`),
    /// - `basis` has exactly one element per tetrahedron,
    /// - every tetrahedron references in-range vertices,
    /// - `mass` has exactly `3 N` degrees of freedom, and
    /// - `mass` is invertible (every lumped mass strictly positive), which the
    ///   implicit solve's preconditioner requires.
    #[must_use]
    pub fn new(
        basis: TetFemBasis,
        tets: Vec<[u32; 4]>,
        material: IsotropicElasticity,
        mass: LumpedMass,
        rest: Vec<Vec3>,
    ) -> Option<Self> {
        let n = rest.len();
        if n == 0
            || basis.element_count() != tets.len()
            || mass.n_dofs() != 3 * n
            || !mass.is_invertible()
        {
            return None;
        }
        let n_u32 = u32::try_from(n).ok()?;
        if tets.iter().flatten().any(|&v| v >= n_u32) {
            return None;
        }
        let positions = rest.clone();
        Some(Self {
            basis,
            tets,
            material,
            mass,
            rest,
            positions,
            velocities: vec![Vec3::ZERO; n],
            pinned: vec![false; n],
            any_pinned: false,
        })
    }

    /// Number of vertices `N`.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.rest.len()
    }

    /// Number of tetrahedra.
    #[must_use]
    pub fn tet_count(&self) -> usize {
        self.tets.len()
    }

    /// Current positions `x_n`.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Current velocities `v_n`.
    #[must_use]
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Rest (reference) positions.
    #[must_use]
    pub fn rest(&self) -> &[Vec3] {
        &self.rest
    }

    /// Overwrites the position of vertex `i`. Returns `false` (and does nothing)
    /// if `i` is out of range or `p` is not finite.
    pub fn set_position(&mut self, i: usize, p: Vec3) -> bool {
        if i >= self.positions.len() || !p.is_finite() {
            return false;
        }
        self.positions[i] = p;
        true
    }

    /// Overwrites the velocity of vertex `i`. Returns `false` (and does nothing)
    /// if `i` is out of range or `v` is not finite.
    pub fn set_velocity(&mut self, i: usize, v: Vec3) -> bool {
        if i >= self.velocities.len() || !v.is_finite() {
            return false;
        }
        self.velocities[i] = v;
        true
    }

    /// Whether vertex `i` is pinned (Dirichlet boundary).
    #[must_use]
    pub fn is_pinned(&self, i: usize) -> bool {
        self.pinned.get(i).copied().unwrap_or(false)
    }

    /// Pins vertex `i` and zeroes its velocity. Returns `false` if `i` is out of
    /// range.
    pub fn pin(&mut self, i: usize) -> bool {
        if i >= self.pinned.len() {
            return false;
        }
        self.pinned[i] = true;
        self.velocities[i] = Vec3::ZERO;
        self.any_pinned = true;
        true
    }

    /// Unpins vertex `i`. Returns `false` if `i` is out of range.
    pub fn unpin(&mut self, i: usize) -> bool {
        if i >= self.pinned.len() {
            return false;
        }
        self.pinned[i] = false;
        self.any_pinned = self.pinned.iter().any(|&p| p);
        true
    }

    /// Replaces the whole Dirichlet mask, zeroing the velocity of every newly
    /// pinned vertex. Returns `false` (and does nothing) if `mask` does not have
    /// exactly [`TetFemBody::vertex_count`] entries.
    pub fn set_pins(&mut self, mask: &[bool]) -> bool {
        if mask.len() != self.pinned.len() {
            return false;
        }
        for (i, &p) in mask.iter().enumerate() {
            if p {
                self.velocities[i] = Vec3::ZERO;
            }
        }
        self.pinned.copy_from_slice(mask);
        self.any_pinned = mask.iter().any(|&p| p);
        true
    }

    /// Unpins every vertex.
    pub fn clear_pins(&mut self) {
        self.pinned.iter_mut().for_each(|p| *p = false);
        self.any_pinned = false;
    }

    /// Total physical mass of the body.
    #[must_use]
    pub fn total_mass(&self) -> f32 {
        self.mass.total_mass()
    }

    /// Lumped mass of vertex `i`. Returns `0.0` if `i` is out of range.
    #[must_use]
    pub fn vertex_mass(&self, i: usize) -> f32 {
        if i >= self.vertex_count() {
            return 0.0;
        }
        self.mass.get(3 * i)
    }

    /// Kinetic energy `½ Σ mᵢ |vᵢ|²` of the current state.
    #[must_use]
    pub fn kinetic_energy(&self) -> f32 {
        let mut sum = 0.0_f64;
        for (i, v) in self.velocities.iter().enumerate() {
            let m = f64::from(self.mass.get(3 * i));
            sum += m * f64::from(v.length_squared());
        }
        (0.5 * sum) as f32
    }

    /// Linear momentum `Σ mᵢ vᵢ` of the current state.
    #[must_use]
    pub fn linear_momentum(&self) -> Vec3 {
        let (mut px, mut py, mut pz) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (i, v) in self.velocities.iter().enumerate() {
            let m = f64::from(self.mass.get(3 * i));
            px += m * f64::from(v.x);
            py += m * f64::from(v.y);
            pz += m * f64::from(v.z);
        }
        Vec3::new(px as f32, py as f32, pz as f32)
    }

    /// Centre of mass `(Σ mᵢ xᵢ) / (Σ mᵢ)` of the current state. Construction
    /// guarantees a strictly positive total mass, so this is always defined.
    #[must_use]
    pub fn center_of_mass(&self) -> Vec3 {
        let (mut cx, mut cy, mut cz) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut total = 0.0_f64;
        for (i, x) in self.positions.iter().enumerate() {
            let m = f64::from(self.mass.get(3 * i));
            cx += m * f64::from(x.x);
            cy += m * f64::from(x.y);
            cz += m * f64::from(x.z);
            total += m;
        }
        Vec3::new(
            (cx / total) as f32,
            (cy / total) as f32,
            (cz / total) as f32,
        )
    }

    /// The pin mask to hand the integrator: `None` when nothing is pinned (so
    /// the solver skips the filter entirely), otherwise the full mask.
    fn pin_mask(&self) -> Option<&[bool]> {
        if self.any_pinned {
            Some(&self.pinned)
        } else {
            None
        }
    }

    /// Runs exactly one implicit corotational step with the given parameters,
    /// advancing `positions`/`velocities` in place. Validates only the external
    /// force length (everything else is guaranteed by construction). Leaves the
    /// state untouched and returns `None` on invalid input or solver breakdown.
    fn advance(
        &mut self,
        external_forces: &[Vec3],
        params: &ImplicitStepParams,
    ) -> Option<CgReport> {
        if external_forces.len() != self.vertex_count() {
            return None;
        }
        let result = step_implicit_corotational(
            &self.basis,
            &self.tets,
            &self.material,
            &self.mass,
            &self.rest,
            &self.positions,
            &self.velocities,
            external_forces,
            self.pin_mask(),
            params,
        )?;
        self.positions = result.positions;
        self.velocities = result.velocities;
        Some(result.solver)
    }

    /// Advances the body by a single implicit step of `params.dt`.
    ///
    /// `external_forces` is the per-vertex external load in Newtons (gravity,
    /// contacts, …) and must have exactly [`TetFemBody::vertex_count`] entries.
    /// On success the body's positions and velocities are updated in place and
    /// the linear-solve [`CgReport`] is returned; on failure the body is left
    /// unchanged and `None` is returned.
    #[must_use = "the solver report indicates whether the step converged"]
    pub fn step(
        &mut self,
        external_forces: &[Vec3],
        params: &ImplicitStepParams,
    ) -> Option<CgReport> {
        self.advance(external_forces, params)
    }

    /// Advances the body by `substeps` implicit steps of `params.dt / substeps`
    /// each, applying the same `external_forces` on every substep.
    ///
    /// Sub-stepping shrinks the per-step time and so the linearisation error of
    /// backward Euler, trading cost for accuracy/stability within one frame.
    /// Returns the per-substep [`CgReport`]s on success. If any substep fails
    /// the body is restored to the state it had on entry and `None` is returned,
    /// so the call is all-or-nothing. Returns `None` when `substeps == 0` or the
    /// external-force length is wrong.
    #[must_use = "the solver reports indicate whether each substep converged"]
    pub fn substep(
        &mut self,
        external_forces: &[Vec3],
        params: &ImplicitStepParams,
        substeps: u32,
    ) -> Option<Vec<CgReport>> {
        if substeps == 0 || external_forces.len() != self.vertex_count() {
            return None;
        }
        let mut sub = *params;
        sub.dt = params.dt / substeps as f32;
        if !sub.dt.is_finite() || sub.dt <= 0.0 {
            return None;
        }
        // Snapshot so a mid-sequence failure leaves the body exactly as it was.
        let saved_positions = self.positions.clone();
        let saved_velocities = self.velocities.clone();
        let mut reports = Vec::with_capacity(substeps as usize);
        for _ in 0..substeps {
            match self.advance(external_forces, &sub) {
                Some(report) => reports.push(report),
                None => {
                    self.positions = saved_positions;
                    self.velocities = saved_velocities;
                    return None;
                }
            }
        }
        Some(reports)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::tet_fem_basis::{build_tet_fem_basis, TetFemBasisParams};
    use crate::collider::tet_lumped_mass::build_lumped_mass_from_mesh;
    use crate::collider::tet_mass::TetMassParams;

    /// A small two-tet mesh sharing a face (same fixture as the integrator).
    fn mesh() -> (Vec<Vec3>, Vec<[u32; 4]>) {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.7, 0.7, 0.7),
        ];
        let tets = vec![[0u32, 1, 2, 3], [1, 2, 3, 4]];
        (verts, tets)
    }

    fn material() -> IsotropicElasticity {
        IsotropicElasticity::new(2.0e4, 0.3).unwrap()
    }

    fn body() -> TetFemBody {
        let (verts, tets) = mesh();
        let basis = build_tet_fem_basis(&verts, &tets, &TetFemBasisParams::default()).unwrap();
        let mass =
            build_lumped_mass_from_mesh(&verts, &tets, &TetMassParams { density: 1200.0 }).unwrap();
        TetFemBody::new(basis, tets, material(), mass, verts).unwrap()
    }

    /// Per-vertex gravity force `fᵢ = mᵢ g`.
    fn gravity(b: &TetFemBody) -> Vec<Vec3> {
        (0..b.vertex_count())
            .map(|i| Vec3::new(0.0, -9.81, 0.0) * b.vertex_mass(i))
            .collect()
    }

    #[test]
    fn new_rejects_inconsistent_inputs() {
        let (verts, tets) = mesh();
        let basis = build_tet_fem_basis(&verts, &tets, &TetFemBasisParams::default()).unwrap();
        let mass =
            build_lumped_mass_from_mesh(&verts, &tets, &TetMassParams { density: 1200.0 }).unwrap();

        // Empty mesh.
        assert!(TetFemBody::new(
            TetFemBasis { elements: vec![] },
            vec![],
            material(),
            LumpedMass { diagonal: vec![] },
            vec![],
        )
        .is_none());

        // Mass with wrong DOF count.
        let short_mass = LumpedMass {
            diagonal: vec![1.0; 3 * verts.len() - 3],
        };
        assert!(TetFemBody::new(
            basis.clone(),
            tets.clone(),
            material(),
            short_mass,
            verts.clone()
        )
        .is_none());

        // Non-invertible mass (a zero on the diagonal).
        let mut zero_mass = mass.clone();
        zero_mass.diagonal[0] = 0.0;
        assert!(TetFemBody::new(
            basis.clone(),
            tets.clone(),
            material(),
            zero_mass,
            verts.clone()
        )
        .is_none());

        // Tetrahedron referencing an out-of-range vertex.
        let bad_tets = vec![[0u32, 1, 2, 99], [1, 2, 3, 4]];
        let bad_basis = TetFemBasis {
            elements: basis.elements.clone(),
        };
        assert!(TetFemBody::new(bad_basis, bad_tets, material(), mass, verts).is_none());
    }

    #[test]
    fn new_initialises_rest_state() {
        let b = body();
        assert_eq!(b.vertex_count(), 5);
        assert_eq!(b.tet_count(), 2);
        assert_eq!(b.positions(), b.rest());
        assert!(b.velocities().iter().all(|v| *v == Vec3::ZERO));
        assert!((0..b.vertex_count()).all(|i| !b.is_pinned(i)));
    }

    #[test]
    fn mass_diagnostics_match_lumped_mass() {
        let b = body();
        let per_vertex: f32 = (0..b.vertex_count()).map(|i| b.vertex_mass(i)).sum();
        assert!((per_vertex - b.total_mass()).abs() < 1e-3 * b.total_mass());
        // At rest with zero velocity: no kinetic energy, no momentum.
        assert_eq!(b.kinetic_energy(), 0.0);
        assert!(b.linear_momentum().length() < 1e-6);
    }

    #[test]
    fn kinetic_energy_and_momentum_track_velocity() {
        let mut b = body();
        let v = Vec3::new(0.0, 2.0, 0.0);
        for i in 0..b.vertex_count() {
            assert!(b.set_velocity(i, v));
        }
        let m = b.total_mass();
        // Uniform velocity: p = M v, KE = ½ M |v|².
        assert!((b.linear_momentum() - m * v).length() < 1e-2);
        assert!((b.kinetic_energy() - 0.5 * m * v.length_squared()).abs() < 1e-1);
    }

    #[test]
    fn center_of_mass_is_mass_weighted() {
        let b = body();
        let mut weighted = Vec3::ZERO;
        let mut total = 0.0;
        for (i, x) in b.positions().iter().enumerate() {
            let m = b.vertex_mass(i);
            weighted += m * *x;
            total += m;
        }
        let expected = weighted / total;
        assert!((b.center_of_mass() - expected).length() < 1e-4);
    }

    #[test]
    fn set_position_and_velocity_reject_out_of_range_and_nonfinite() {
        let mut b = body();
        assert!(b.set_position(0, Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(b.positions()[0], Vec3::new(1.0, 2.0, 3.0));
        assert!(!b.set_position(99, Vec3::ONE));
        assert!(!b.set_position(0, Vec3::new(f32::NAN, 0.0, 0.0)));
        assert!(!b.set_velocity(99, Vec3::ONE));
        assert!(!b.set_velocity(0, Vec3::new(f32::INFINITY, 0.0, 0.0)));
    }

    #[test]
    fn step_under_gravity_accelerates_downward() {
        let mut b = body();
        let f = gravity(&b);
        let params = ImplicitStepParams::new(1.0 / 60.0);
        let com_before = b.center_of_mass();
        let report = b.step(&f, &params).expect("step should converge");
        assert!(report.converged);
        // Every vertex gains downward velocity close to g*h (the body is soft,
        // so the mean is slightly damped by internal forces).
        let mean_vy = b.velocities().iter().map(|v| v.y).sum::<f32>() / b.vertex_count() as f32;
        assert!(
            mean_vy < -0.1,
            "mean vertical velocity {mean_vy} not falling"
        );
        // Centre of mass drops.
        assert!(b.center_of_mass().y < com_before.y);
    }

    #[test]
    fn step_rejects_wrong_force_length() {
        let mut b = body();
        let params = ImplicitStepParams::new(1.0 / 60.0);
        let short = vec![Vec3::ZERO; b.vertex_count() - 1];
        assert!(b.step(&short, &params).is_none());
    }

    #[test]
    fn pinned_vertex_stays_fixed() {
        let mut b = body();
        assert!(b.pin(0));
        assert!(b.is_pinned(0));
        let x0 = b.positions()[0];
        let f = gravity(&b);
        let params = ImplicitStepParams::new(1.0 / 60.0);
        let _ = b.step(&f, &params).expect("step should converge");
        assert!(
            (b.positions()[0] - x0).length() < 1e-5,
            "pinned vertex moved"
        );
        assert!(
            b.velocities()[0].length() < 1e-5,
            "pinned vertex gained velocity"
        );
    }

    #[test]
    fn set_pins_and_clear_pins_update_mask() {
        let mut b = body();
        let n = b.vertex_count();
        let mut mask = vec![false; n];
        mask[1] = true;
        mask[3] = true;
        assert!(b.set_pins(&mask));
        assert!(b.is_pinned(1) && b.is_pinned(3) && !b.is_pinned(0));
        assert!(!b.set_pins(&vec![false; n + 1]));
        b.clear_pins();
        assert!((0..n).all(|i| !b.is_pinned(i)));
    }

    #[test]
    fn substep_matches_vertex_count_and_restores_on_bad_input() {
        let mut b = body();
        let f = gravity(&b);
        let params = ImplicitStepParams::new(1.0 / 60.0);
        let reports = b.substep(&f, &params, 4).expect("substeps should converge");
        assert_eq!(reports.len(), 4);
        assert!(reports.iter().all(|r| r.converged));
        // Zero substeps and wrong force length are rejected without mutation.
        let saved = b.positions().to_vec();
        assert!(b.substep(&f, &params, 0).is_none());
        let short = vec![Vec3::ZERO; b.vertex_count() - 1];
        assert!(b.substep(&short, &params, 4).is_none());
        assert_eq!(b.positions(), saved.as_slice());
    }

    #[test]
    fn substepping_tracks_single_step_fall_direction() {
        // One step of h versus four substeps of h/4 under gravity should both
        // move the centre of mass downward; substepping should not blow up.
        let mut single = body();
        let mut subbed = body();
        let f = gravity(&single);
        let params = ImplicitStepParams::new(1.0 / 60.0);
        let com0 = single.center_of_mass();
        single.step(&f, &params).unwrap();
        subbed.substep(&f, &params, 4).unwrap();
        assert!(single.center_of_mass().y < com0.y);
        assert!(subbed.center_of_mass().y < com0.y);
        // Both remain finite and close (same frame time, finer resolution).
        let gap = (single.center_of_mass() - subbed.center_of_mass()).length();
        assert!(gap < 0.05, "single vs substepped COM gap too large: {gap}");
    }
}
