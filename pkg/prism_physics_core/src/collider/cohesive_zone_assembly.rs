//! Global assembly of mixed-mode cohesive-zone forces over shared interface
//! facets.
//!
//! This is the surface-law counterpart of the volumetric damage assembly in
//! [`tet_fem_damage_force_assembly`](super::tet_fem_damage_force_assembly): it
//! takes the per-interface bilinear traction–separation kernel from
//! [`cohesive_zone`](super::cohesive_zone) and scatters it across a mesh of
//! shared interface facets into both-sided nodal forces, carrying one
//! irreversible [`CohesiveState`] per facet.
//!
//! # Interface model
//!
//! Each [`CohesiveInterface`] couples two coincident triangular faces — a
//! `side_a` triple and a `side_b` triple of vertex indices — that start
//! together at rest and may pull apart. A crack inside a tetrahedral mesh is
//! represented by duplicating the vertices along the crack surface so that the
//! two sides share geometry at rest but can displace independently afterwards.
//!
//! For each facet the assembly:
//!
//! 1. Builds the reference triangle from the rest positions of `side_a`, giving
//!    an outward unit normal `n` and reference area `A`.
//! 2. Forms the relative displacement jump `δ = ū_b − ū_a`, where `ū_s` is the
//!    mean nodal displacement (`current − rest`) of side `s`. Using
//!    displacements rather than raw positions keeps the law exact even when the
//!    two sides are not perfectly coincident at rest.
//! 3. Evaluates [`cohesive_traction`] to obtain the traction `t` (force per unit
//!    area) and advances the facet's irreversible history once.
//! 4. Scatters the resultant interface force `F = A · t` to the two sides so
//!    that side `a` is pulled toward side `b` and vice versa: each of the three
//!    `side_a` nodes receives `+F/3` and each of the three `side_b` nodes
//!    receives `−F/3`. The six nodal contributions sum to zero, so an assembled
//!    pass conserves linear momentum to within rounding.
//!
//! The returned [`CohesiveAssembly`] keeps the per-facet [`CohesiveFacetStep`]
//! alongside the global force vector so a caller can report how many facets are
//! cracking and recover the dissipated fracture energy via
//! [`total_dissipated_energy`] without re-advancing the state. This module holds
//! no solver state of its own and performs no time integration.

use crate::collider::cohesive_zone::{
    cohesive_traction, dissipated_energy, CohesiveModel, CohesiveState, CohesiveStep,
};
use glam::Vec3;

/// One shared cohesive interface facet: two triangular faces whose vertices
/// start coincident at rest and may separate.
///
/// `side_a[i]` and `side_b[i]` are the paired vertices of local corner `i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CohesiveInterface {
    /// Vertex indices of the first ("negative") face of the interface.
    pub side_a: [u32; 3],
    /// Vertex indices of the second ("positive") face of the interface.
    pub side_b: [u32; 3],
}

impl CohesiveInterface {
    /// Builds an interface from the two triangular faces.
    #[must_use]
    pub fn new(side_a: [u32; 3], side_b: [u32; 3]) -> Self {
        Self { side_a, side_b }
    }

    /// Highest vertex index referenced by either side, used for bounds checks.
    #[must_use]
    fn max_index(&self) -> u32 {
        self.side_a
            .iter()
            .chain(self.side_b.iter())
            .copied()
            .max()
            .unwrap_or(0)
    }
}

/// Per-facet outcome of one cohesive assembly pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CohesiveFacetStep {
    /// The underlying traction–separation evaluation for this facet.
    pub step: CohesiveStep,
    /// Reference (rest) area `A` of the facet.
    pub area: f32,
    /// Resultant interface force magnitude `|A · t|` developed across the facet.
    pub force_magnitude: f32,
}

/// Aggregate result of one global cohesive-force assembly pass.
#[derive(Clone, Debug, PartialEq)]
pub struct CohesiveAssembly {
    /// Per-vertex cohesive force, one entry per input position.
    pub forces: Vec<Vec3>,
    /// Per-facet outcome, in interface order.
    pub steps: Vec<CohesiveFacetStep>,
}

impl CohesiveAssembly {
    /// Number of facets carrying any decohesion damage (`d > 0`).
    #[must_use]
    pub fn damaged_facet_count(&self) -> usize {
        self.steps.iter().filter(|s| s.step.damage > 0.0).count()
    }

    /// Number of facets that are fully decohered (`d = 1`).
    #[must_use]
    pub fn decohered_facet_count(&self) -> usize {
        self.steps.iter().filter(|s| s.step.damage >= 1.0).count()
    }

    /// Number of facets whose irreversible history advanced this pass.
    #[must_use]
    pub fn advanced_facet_count(&self) -> usize {
        self.steps.iter().filter(|s| s.step.advanced).count()
    }

    /// Mean decohesion damage `d` across all facets, or `0.0` for an empty mesh.
    #[must_use]
    pub fn mean_damage(&self) -> f32 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.steps.iter().map(|s| s.step.damage).sum();
        sum / self.steps.len() as f32
    }
}

/// Builds `count` fresh rest cohesive states, one per interface facet.
#[must_use]
pub fn rest_states(count: usize) -> Vec<CohesiveState> {
    vec![CohesiveState::rest(); count]
}

/// Gathers the three vertices of a face triple from `positions`.
fn gather(positions: &[Vec3], face: [u32; 3]) -> Option<[Vec3; 3]> {
    let n = positions.len();
    if face.iter().any(|&vi| vi as usize >= n) {
        return None;
    }
    Some([
        positions[face[0] as usize],
        positions[face[1] as usize],
        positions[face[2] as usize],
    ])
}

/// Mean of a triangle's three corners.
fn centroid(tri: [Vec3; 3]) -> Vec3 {
    (tri[0] + tri[1] + tri[2]) / 3.0
}

/// Outward normal and reference area of a reference triangle. Returns `None`
/// for a degenerate (zero-area) triangle whose normal is undefined.
fn normal_and_area(tri: [Vec3; 3]) -> Option<(Vec3, f32)> {
    let cross = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
    let twice_area = cross.length();
    if twice_area <= f32::EPSILON {
        return None;
    }
    Some((cross / twice_area, 0.5 * twice_area))
}

/// Validates that the interfaces, position arrays, and state slice are mutually
/// consistent: one state per interface, equal-length rest / current positions,
/// and every referenced vertex addressable.
fn is_consistent(
    interfaces: &[CohesiveInterface],
    rest_positions: &[Vec3],
    positions: &[Vec3],
    states: &[CohesiveState],
) -> bool {
    if states.len() != interfaces.len() || rest_positions.len() != positions.len() {
        return false;
    }
    let n = positions.len() as u32;
    interfaces.iter().all(|f| f.max_index() < n)
}

/// Assembles the per-vertex cohesive force vector of the interface mesh at the
/// supplied rest and current positions, advancing each facet's state once.
///
/// `states` must hold exactly one [`CohesiveState`] per interface (see
/// [`rest_states`]); each is updated in place whenever its facet's effective
/// separation exceeds the stored history. Returns `None` when the lengths
/// disagree, when any vertex index is out of range, or when a facet's rest
/// triangle is degenerate (zero area); in that case no state is mutated.
///
/// A facet that has never been loaded past its onset separation contributes an
/// elastic penalty force only and leaves its state at rest.
#[must_use]
pub fn assemble_cohesive_forces(
    interfaces: &[CohesiveInterface],
    rest_positions: &[Vec3],
    positions: &[Vec3],
    model: &CohesiveModel,
    states: &mut [CohesiveState],
) -> Option<CohesiveAssembly> {
    if !is_consistent(interfaces, rest_positions, positions, states) {
        return None;
    }
    // Validate every facet's geometry up front so a late failure cannot leave
    // the cohesive states partially advanced.
    for f in interfaces {
        let rest_a = gather(rest_positions, f.side_a)?;
        gather(rest_positions, f.side_b)?;
        gather(positions, f.side_a)?;
        gather(positions, f.side_b)?;
        normal_and_area(rest_a)?;
    }

    let mut forces = vec![Vec3::ZERO; positions.len()];
    let mut steps = Vec::with_capacity(interfaces.len());
    for (f, state) in interfaces.iter().zip(states.iter_mut()) {
        let rest_a = gather(rest_positions, f.side_a)?;
        let rest_b = gather(rest_positions, f.side_b)?;
        let cur_a = gather(positions, f.side_a)?;
        let cur_b = gather(positions, f.side_b)?;
        let (normal, area) = normal_and_area(rest_a)?;

        // Relative displacement jump of side b with respect to side a.
        let disp_a = centroid(cur_a) - centroid(rest_a);
        let disp_b = centroid(cur_b) - centroid(rest_b);
        let delta = disp_b - disp_a;

        let step = cohesive_traction(delta, normal, model, state);
        // Resultant interface force; +F pulls side a toward side b.
        let resultant = area * step.traction;
        let per_node = resultant / 3.0;
        for &vi in &f.side_a {
            forces[vi as usize] += per_node;
        }
        for &vi in &f.side_b {
            forces[vi as usize] -= per_node;
        }

        steps.push(CohesiveFacetStep {
            step,
            area,
            force_magnitude: resultant.length(),
        });
    }
    Some(CohesiveAssembly { forces, steps })
}

/// Total fracture energy dissipated by decohesion across all facets, i.e.
/// `Σ_f A_f · Φ(κ_f)` using the per-facet areas and history variables produced
/// by a prior [`assemble_cohesive_forces`] pass.
///
/// This is a pure read of the dissipated energy; it does not advance any state.
/// Returns `None` when `states` and `steps` disagree in length.
#[must_use]
pub fn total_dissipated_energy(
    model: &CohesiveModel,
    states: &[CohesiveState],
    steps: &[CohesiveFacetStep],
) -> Option<f32> {
    if states.len() != steps.len() {
        return None;
    }
    let mut total = 0.0;
    for (state, step) in states.iter().zip(steps.iter()) {
        total += step.area * dissipated_energy(model, state.kappa());
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    // K = 1e6, σ_c = 1e3 → δ₀ = 1e-3. G_c = 1.0 → δ_f = 2e-3 (valid).
    fn model() -> CohesiveModel {
        CohesiveModel::new(1.0e6, 1.0e3, 1.0, 1.0).unwrap()
    }

    // A single interface: side a is a unit triangle in the z = 0 plane, side b
    // is its coincident twin (vertices 3..6). side_a winding gives +z normal.
    fn single_interface() -> (Vec<CohesiveInterface>, Vec<Vec3>) {
        let rest = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![CohesiveInterface::new([0, 1, 2], [3, 4, 5])];
        (faces, rest)
    }

    #[test]
    fn rest_pose_has_zero_force_and_no_advance() {
        let (faces, rest) = single_interface();
        let mut states = rest_states(faces.len());
        let out = assemble_cohesive_forces(&faces, &rest, &rest, &model(), &mut states).unwrap();
        for f in &out.forces {
            assert!(f.length() < 1e-6, "rest force must be zero, got {f:?}");
        }
        assert_eq!(out.advanced_facet_count(), 0);
        assert_eq!(out.damaged_facet_count(), 0);
        assert_eq!(states[0].kappa(), 0.0);
    }

    #[test]
    fn area_and_normal_are_correct() {
        let (faces, rest) = single_interface();
        let mut states = rest_states(faces.len());
        // Open side b by a small tensile amount along +z.
        let mut cur = rest.clone();
        for v in cur.iter_mut().skip(3) {
            v.z += 5.0e-4;
        }
        let out = assemble_cohesive_forces(&faces, &rest, &cur, &model(), &mut states).unwrap();
        assert!(
            (out.steps[0].area - 0.5).abs() < 1e-6,
            "unit triangle area 0.5"
        );
        // Elastic (pre-onset) traction t = K·δ = 1e6 · 5e-4 = 500 along +z.
        assert!((out.steps[0].step.normal_traction - 500.0).abs() < 1e-1);
        assert_eq!(out.steps[0].step.damage, 0.0, "below onset, no damage");
    }

    #[test]
    fn forces_sum_to_zero_and_pull_sides_together() {
        let (faces, rest) = single_interface();
        let mut states = rest_states(faces.len());
        let mut cur = rest.clone();
        for v in cur.iter_mut().skip(3) {
            v.z += 1.5e-3; // past onset (1e-3), into softening
        }
        let out = assemble_cohesive_forces(&faces, &rest, &cur, &model(), &mut states).unwrap();
        let total: Vec3 = out.forces.iter().copied().sum();
        assert!(
            total.length() < 1e-3,
            "net force must vanish, got {total:?}"
        );
        // side a (z = 0) is pulled toward +z (toward side b); side b toward −z.
        assert!(out.forces[0].z > 0.0, "side a pulled +z");
        assert!(out.forces[3].z < 0.0, "side b pulled -z");
        assert!(out.steps[0].step.damage > 0.0, "softening ⇒ damage");
        assert!(states[0].kappa() > model().onset_separation());
    }

    #[test]
    fn compression_develops_penalty_without_damage() {
        let (faces, rest) = single_interface();
        let mut states = rest_states(faces.len());
        let mut cur = rest.clone();
        for v in cur.iter_mut().skip(3) {
            v.z -= 2.0e-3; // interpenetration
        }
        let out = assemble_cohesive_forces(&faces, &rest, &cur, &model(), &mut states).unwrap();
        assert!(
            out.steps[0].step.normal_traction < 0.0,
            "compressive penalty"
        );
        assert_eq!(
            out.steps[0].step.damage, 0.0,
            "compression carries no damage"
        );
        assert_eq!(states[0].kappa(), 0.0);
    }

    #[test]
    fn history_is_irreversible() {
        let (faces, rest) = single_interface();
        let mut states = rest_states(faces.len());
        let mut cur = rest.clone();
        for v in cur.iter_mut().skip(3) {
            v.z += 1.5e-3;
        }
        let _ = assemble_cohesive_forces(&faces, &rest, &cur, &model(), &mut states).unwrap();
        let kappa_loaded = states[0].kappa();
        let damage_loaded = model().damage_at(kappa_loaded);
        assert!(damage_loaded > 0.0);

        // Unload back toward rest: kappa must not shrink and damage persists.
        let out2 = assemble_cohesive_forces(&faces, &rest, &rest, &model(), &mut states).unwrap();
        assert_eq!(states[0].kappa(), kappa_loaded, "kappa is monotone");
        assert!((out2.steps[0].step.damage - damage_loaded).abs() < 1e-6);
        assert!(
            !out2.steps[0].step.advanced,
            "reload below kappa does not advance"
        );
    }

    #[test]
    fn dissipated_energy_scales_with_area_and_reaches_gc() {
        let (faces, rest) = single_interface();
        let mut states = rest_states(faces.len());
        let mut cur = rest.clone();
        for v in cur.iter_mut().skip(3) {
            v.z += 5.0e-3; // past final separation (2e-3) ⇒ fully decohered
        }
        let out = assemble_cohesive_forces(&faces, &rest, &cur, &model(), &mut states).unwrap();
        assert_eq!(out.decohered_facet_count(), 1);
        let energy = total_dissipated_energy(&model(), &states, &out.steps).unwrap();
        // Full G_c (=1.0) over a 0.5-area facet ⇒ 0.5.
        assert!((energy - 0.5).abs() < 1e-5, "got {energy}");
    }

    #[test]
    fn rejects_inconsistent_inputs() {
        let (faces, rest) = single_interface();
        let mut states = rest_states(faces.len());
        // Mismatched rest / current length.
        let short = vec![Vec3::ZERO; 3];
        assert!(assemble_cohesive_forces(&faces, &rest, &short, &model(), &mut states).is_none());
        // Too few states.
        let mut no_states: Vec<CohesiveState> = Vec::new();
        assert!(assemble_cohesive_forces(&faces, &rest, &rest, &model(), &mut no_states).is_none());
    }

    #[test]
    fn rejects_out_of_range_index() {
        let rest = vec![Vec3::ZERO; 6];
        let faces = vec![CohesiveInterface::new([0, 1, 2], [3, 4, 99])];
        let mut states = rest_states(faces.len());
        assert!(assemble_cohesive_forces(&faces, &rest, &rest, &model(), &mut states).is_none());
    }

    #[test]
    fn rejects_degenerate_rest_triangle() {
        // Collinear side-a rest triangle ⇒ undefined normal.
        let rest = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let faces = vec![CohesiveInterface::new([0, 1, 2], [3, 4, 5])];
        let mut states = rest_states(faces.len());
        assert!(assemble_cohesive_forces(&faces, &rest, &rest, &model(), &mut states).is_none());
    }

    #[test]
    fn multi_facet_counts_and_momentum() {
        // Two independent interfaces (vertices 0..6 and 6..12).
        let base = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut rest = Vec::new();
        for _ in 0..2 {
            rest.extend_from_slice(&base); // side a
            rest.extend_from_slice(&base); // side b
        }
        let faces = vec![
            CohesiveInterface::new([0, 1, 2], [3, 4, 5]),
            CohesiveInterface::new([6, 7, 8], [9, 10, 11]),
        ];
        let mut states = rest_states(faces.len());
        let mut cur = rest.clone();
        // Crack only the first interface (side b vertices 3,4,5).
        for v in cur.iter_mut().take(6).skip(3) {
            v.z += 1.5e-3;
        }
        let out = assemble_cohesive_forces(&faces, &rest, &cur, &model(), &mut states).unwrap();
        assert_eq!(out.damaged_facet_count(), 1);
        assert_eq!(out.steps.len(), 2);
        let total: Vec3 = out.forces.iter().copied().sum();
        assert!(total.length() < 1e-3, "global momentum conserved");
        assert!(out.mean_damage() > 0.0 && out.mean_damage() < 1.0);
    }
}
