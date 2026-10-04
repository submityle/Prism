//! Persistent wet-bridge topology with formation/rupture hysteresis.
//!
//! Wet granular media differ from dry ones in that a liquid bridge between two
//! grains does not appear and vanish at a single separation. Instead a bridge
//! *forms* when the grains come (nearly) into contact and then *persists* as the
//! grains separate, only *rupturing* once the gap exceeds a larger critical
//! distance. That gap between the formation and rupture thresholds is the
//! hysteresis that gives wet piles their cohesion and history dependence.
//!
//! This module tracks the active bridge set across frames using exactly that
//! rule. It owns a reusable [`UniformGridBroadphase`](crate::collider::uniform_grid_broadphase::UniformGridBroadphase)
//! to find near pairs cheaply and keeps a sorted list of currently bonded
//! grain pairs. It is deliberately *model-agnostic*: it decides only *which*
//! pairs are bonded, using surface-to-surface separation thresholds supplied by
//! the caller. The resulting topology is the natural input to a force
//! evaluator such as
//! [`FiniteLiquidCapillary`](crate::collider::finite_liquid_capillary_forces::FiniteLiquidCapillary),
//! keeping topology and force concerns in separate, composable pieces.
//!
//! Surface gap between two spheres `a` and `b` is
//! `gap = |p_b - p_a| - r_a - r_b` (negative while overlapping). A pair bonds
//! when `gap <= formation_gap` and stays bonded until `gap > rupture_gap`, with
//! `rupture_gap > formation_gap`.

use glam::Vec3;

use crate::collider::uniform_grid_broadphase::UniformGridBroadphase;

/// Tracks a persistent set of bonded grain pairs with formation/rupture
/// hysteresis.
///
/// Construct once with the two separation thresholds and call
/// [`update`](Self::update) each frame with the current grain positions and
/// radii. The internal broad phase and bond list are reused across calls, so
/// the per-frame allocation is amortised.
#[derive(Clone, Debug)]
pub struct WetBridgeHysteresis {
    broadphase: UniformGridBroadphase,
    formation_gap: f32,
    rupture_gap: f32,
    /// Currently bonded pairs, sorted ascending as `(a, b)` with `a < b`.
    active: Vec<(u32, u32)>,
}

/// Outcome of a single [`WetBridgeHysteresis::update`] call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WetBridgeReport {
    /// Number of bonds that newly formed this frame.
    pub formed: u32,
    /// Number of bonds that ruptured this frame.
    pub ruptured: u32,
    /// Number of bonds active after the update.
    pub active: u32,
}

impl WetBridgeHysteresis {
    /// Create a tracker with the given hysteresis thresholds.
    ///
    /// `formation_gap` is the surface separation at or below which a new bond
    /// forms; it must be finite and may be zero (pure contact) or a small
    /// positive value. `rupture_gap` is the separation above which an existing
    /// bond breaks; it must be finite and strictly greater than
    /// `formation_gap`. Returns `None` if either requirement is violated.
    pub fn new(formation_gap: f32, rupture_gap: f32) -> Option<Self> {
        if !formation_gap.is_finite() || !rupture_gap.is_finite() {
            return None;
        }
        if formation_gap < 0.0 {
            return None;
        }
        if rupture_gap <= formation_gap {
            return None;
        }
        Some(Self {
            broadphase: UniformGridBroadphase::new(),
            formation_gap,
            rupture_gap,
            active: Vec::new(),
        })
    }

    /// Surface separation at or below which a bond forms.
    pub fn formation_gap(&self) -> f32 {
        self.formation_gap
    }

    /// Surface separation above which a bond ruptures.
    pub fn rupture_gap(&self) -> f32 {
        self.rupture_gap
    }

    /// The currently bonded pairs, sorted ascending as `(a, b)` with `a < b`.
    pub fn active_bridges(&self) -> &[(u32, u32)] {
        &self.active
    }

    /// Number of currently bonded pairs.
    pub fn bridge_count(&self) -> usize {
        self.active.len()
    }

    /// Returns `true` if the (ordered) pair `(a, b)` with `a < b` is bonded.
    pub fn is_bonded(&self, a: u32, b: u32) -> bool {
        let key = if a < b { (a, b) } else { (b, a) };
        self.active.binary_search(&key).is_ok()
    }

    /// Clear all bonds, dropping the entire topology history.
    pub fn reset(&mut self) {
        self.active.clear();
    }

    /// Advance the bonded-pair topology by one frame.
    ///
    /// `positions` and `radii` must have equal length; every position must be
    /// finite and every radius finite and strictly positive. Returns `None` if
    /// any input is invalid (the stored topology is left unchanged in that
    /// case).
    ///
    /// A pair that is not yet bonded forms a bond when its surface gap is at or
    /// below `formation_gap`. A bonded pair ruptures when its surface gap
    /// exceeds `rupture_gap`, or when the pair has drifted far enough apart that
    /// the broad phase no longer reports it as a candidate. The active list is
    /// rebuilt sorted ascending each call.
    pub fn update(&mut self, positions: &[Vec3], radii: &[f32]) -> Option<WetBridgeReport> {
        let n = positions.len();
        if radii.len() != n {
            return None;
        }
        for (pos, &r) in positions.iter().zip(radii.iter()) {
            if !pos.is_finite() || !r.is_finite() || r <= 0.0 {
                return None;
            }
        }

        // Candidate pairs within the rupture reach. Using the rupture gap as the
        // broad-phase margin guarantees every pair that could still be bonded
        // is enumerated; pairs beyond it are implicitly ruptured.
        let candidates = self
            .broadphase
            .candidate_pairs(positions, radii, self.rupture_gap)?;

        let mut next: Vec<(u32, u32)> = Vec::with_capacity(self.active.len());
        let mut formed: u32 = 0;
        for &(a, b) in candidates.iter() {
            let gap = surface_gap(positions, radii, a, b);
            let was_bonded = self.active.binary_search(&(a, b)).is_ok();
            if was_bonded {
                // Persist unless the pair has pulled past the rupture gap.
                if gap <= self.rupture_gap {
                    next.push((a, b));
                }
            } else if gap <= self.formation_gap {
                next.push((a, b));
                formed += 1;
            }
        }

        // `candidates` is sorted ascending and we iterate it in order, so `next`
        // is already sorted; no extra sort is required.
        let ruptured = (self.active.len() + formed as usize - next.len()) as u32;
        self.active = next;
        Some(WetBridgeReport {
            formed,
            ruptured,
            active: self.active.len() as u32,
        })
    }
}

/// Surface-to-surface separation between spheres `a` and `b` (negative while
/// overlapping). Callers guarantee the indices are in range.
fn surface_gap(positions: &[Vec3], radii: &[f32], a: u32, b: u32) -> f32 {
    let ai = a as usize;
    let bi = b as usize;
    let centre_distance = (positions[bi] - positions[ai]).length();
    centre_distance - radii[ai] - radii[bi]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_grains(gap: f32) -> (Vec<Vec3>, Vec<f32>) {
        // Two unit-radius grains separated along x by the requested surface gap.
        let r = 1.0_f32;
        let positions = vec![Vec3::ZERO, Vec3::new(2.0 * r + gap, 0.0, 0.0)];
        let radii = vec![r, r];
        (positions, radii)
    }

    #[test]
    fn new_validates_parameters() {
        assert!(WetBridgeHysteresis::new(0.0, 0.2).is_some());
        assert!(WetBridgeHysteresis::new(0.05, 0.2).is_some());
        // Negative formation gap rejected.
        assert!(WetBridgeHysteresis::new(-0.01, 0.2).is_none());
        // Rupture must strictly exceed formation.
        assert!(WetBridgeHysteresis::new(0.2, 0.2).is_none());
        assert!(WetBridgeHysteresis::new(0.3, 0.2).is_none());
        // Non-finite rejected.
        assert!(WetBridgeHysteresis::new(f32::NAN, 0.2).is_none());
        assert!(WetBridgeHysteresis::new(0.0, f32::INFINITY).is_none());
    }

    #[test]
    fn rejects_invalid_update_input() {
        let mut h = WetBridgeHysteresis::new(0.0, 0.2).unwrap();
        let positions = vec![Vec3::ZERO, Vec3::X];
        // Mismatched lengths.
        assert!(h.update(&positions, &[1.0]).is_none());
        // Non-positive radius.
        assert!(h.update(&positions, &[1.0, 0.0]).is_none());
        // Non-finite position.
        let bad = vec![Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0)];
        assert!(h.update(&bad, &[1.0, 1.0]).is_none());
        // Nothing was recorded.
        assert_eq!(h.bridge_count(), 0);
    }

    #[test]
    fn bond_forms_on_contact() {
        let mut h = WetBridgeHysteresis::new(0.0, 0.3).unwrap();
        let (positions, radii) = two_grains(-0.001); // slight overlap => contact
        let report = h.update(&positions, &radii).unwrap();
        assert_eq!(report.formed, 1);
        assert_eq!(report.active, 1);
        assert!(h.is_bonded(0, 1));
        assert_eq!(h.active_bridges(), &[(0, 1)]);
    }

    #[test]
    fn separated_pair_does_not_form() {
        let mut h = WetBridgeHysteresis::new(0.0, 0.3).unwrap();
        // Gap above the formation threshold but within rupture reach: a pristine
        // (never-bonded) pair must NOT form a bond here.
        let (positions, radii) = two_grains(0.15);
        let report = h.update(&positions, &radii).unwrap();
        assert_eq!(report.formed, 0);
        assert_eq!(report.active, 0);
        assert!(!h.is_bonded(0, 1));
    }

    #[test]
    fn bond_persists_across_hysteresis_band() {
        let mut h = WetBridgeHysteresis::new(0.0, 0.3).unwrap();
        // Form on contact.
        let (p0, r0) = two_grains(-0.001);
        assert_eq!(h.update(&p0, &r0).unwrap().formed, 1);
        // Separate into the hysteresis band (formation < gap < rupture). A fresh
        // pair would not form here, but the existing bond must persist.
        let (p1, r1) = two_grains(0.15);
        let report = h.update(&p1, &r1).unwrap();
        assert_eq!(report.formed, 0);
        assert_eq!(report.ruptured, 0);
        assert_eq!(report.active, 1);
        assert!(h.is_bonded(0, 1));
    }

    #[test]
    fn bond_ruptures_beyond_rupture_gap() {
        let mut h = WetBridgeHysteresis::new(0.0, 0.3).unwrap();
        let (p0, r0) = two_grains(-0.001);
        h.update(&p0, &r0).unwrap();
        // Pull apart past the rupture gap.
        let (p1, r1) = two_grains(0.5);
        let report = h.update(&p1, &r1).unwrap();
        assert_eq!(report.ruptured, 1);
        assert_eq!(report.active, 0);
        assert!(!h.is_bonded(0, 1));
    }

    #[test]
    fn reformation_requires_recontact() {
        let mut h = WetBridgeHysteresis::new(0.0, 0.3).unwrap();
        let (p0, r0) = two_grains(-0.001);
        h.update(&p0, &r0).unwrap();
        // Rupture it.
        let (p1, r1) = two_grains(0.5);
        h.update(&p1, &r1).unwrap();
        // Return into the hysteresis band but not into contact: stays unbonded.
        let (p2, r2) = two_grains(0.15);
        let report = h.update(&p2, &r2).unwrap();
        assert_eq!(report.formed, 0);
        assert_eq!(report.active, 0);
        // Re-contact reforms the bond.
        let (p3, r3) = two_grains(-0.001);
        assert_eq!(h.update(&p3, &r3).unwrap().formed, 1);
        assert!(h.is_bonded(0, 1));
    }

    #[test]
    fn reset_clears_topology() {
        let mut h = WetBridgeHysteresis::new(0.0, 0.3).unwrap();
        let (p0, r0) = two_grains(-0.001);
        h.update(&p0, &r0).unwrap();
        assert_eq!(h.bridge_count(), 1);
        h.reset();
        assert_eq!(h.bridge_count(), 0);
        assert!(!h.is_bonded(0, 1));
    }

    #[test]
    fn tracks_multiple_bonds_sorted() {
        let mut h = WetBridgeHysteresis::new(0.02, 0.3).unwrap();
        // A short chain of four grains all in contact along x.
        let r = 1.0_f32;
        let step = 2.0 * r; // touching
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(step, 0.0, 0.0),
            Vec3::new(2.0 * step, 0.0, 0.0),
            Vec3::new(3.0 * step, 0.0, 0.0),
        ];
        let radii = vec![r; 4];
        let report = h.update(&positions, &radii).unwrap();
        assert_eq!(report.formed, 3);
        assert_eq!(h.active_bridges(), &[(0, 1), (1, 2), (2, 3)]);
        // The list is sorted ascending.
        let mut sorted = h.active_bridges().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, h.active_bridges());
    }
}
