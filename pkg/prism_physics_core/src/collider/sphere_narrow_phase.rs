//! Broad-phase-accelerated sphere-sphere narrow phase.
//!
//! Granular and DEM-style solvers need, each frame, the set of actually
//! touching (or near-touching) grain pairs together with the contact geometry:
//! a unit normal, a penetration depth, and a contact point. This module turns
//! a sphere cloud into exactly that set.
//!
//! It reuses [`UniformGridBroadphase`](crate::collider::uniform_grid_broadphase::UniformGridBroadphase)
//! to cheaply enumerate nearby candidate pairs, then performs the exact
//! sphere-sphere test on each candidate. Keeping the broad phase and this
//! narrow phase in separate modules means either can be reused or replaced
//! independently; this one adds only the exact geometric test and the reusable
//! contact buffer.
//!
//! A contact is emitted when the surface gap `|p_b - p_a| - r_a - r_b` is at or
//! below the detection `margin`. With `margin = 0` only overlapping pairs are
//! reported; a positive margin additionally reports near pairs (useful for
//! predictive/speculative contacts), whose penetration is then negative.

use glam::Vec3;

use crate::collider::uniform_grid_broadphase::UniformGridBroadphase;

/// Minimum centre distance below which two grains are treated as coincident and
/// a fallback normal is used.
const COINCIDENT_EPSILON: f32 = 1.0e-9;

/// A single resolved sphere-sphere contact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereContact {
    /// Lower grain index (`a < b`).
    pub a: u32,
    /// Upper grain index (`a < b`).
    pub b: u32,
    /// Unit contact normal pointing from grain `a` toward grain `b`.
    pub normal: Vec3,
    /// Overlap depth `r_a + r_b - |p_b - p_a|`. Positive while overlapping;
    /// negative for near pairs reported because of a positive detection margin.
    pub penetration: f32,
    /// Point on the contact plane, on the `a -> b` axis midway through the
    /// overlap region.
    pub contact_point: Vec3,
}

/// Reusable sphere-sphere narrow phase.
///
/// Construct once and call [`detect`](Self::detect) each frame; both the
/// internal broad phase and the contact buffer are reused, so the per-frame
/// allocation is amortised.
#[derive(Clone, Debug, Default)]
pub struct SphereNarrowPhase {
    broadphase: UniformGridBroadphase,
    contacts: Vec<SphereContact>,
}

impl SphereNarrowPhase {
    /// Create an empty narrow phase.
    pub fn new() -> Self {
        Self::default()
    }

    /// The contacts produced by the most recent [`detect`](Self::detect) call.
    pub fn contacts(&self) -> &[SphereContact] {
        &self.contacts
    }

    /// Number of contacts from the most recent [`detect`](Self::detect) call.
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// Largest penetration depth among the current contacts, or `0.0` if there
    /// are none.
    pub fn max_penetration(&self) -> f32 {
        self.contacts
            .iter()
            .fold(0.0_f32, |acc, c| acc.max(c.penetration))
    }

    /// Detect sphere-sphere contacts for the given cloud.
    ///
    /// `positions` and `radii` must have equal length; every position must be
    /// finite and every radius finite and strictly positive. `margin` is the
    /// surface gap at or below which a pair is reported (finite, non-negative;
    /// `0.0` means contact/overlap only). Returns `None` on invalid input.
    ///
    /// On success the returned slice is sorted ascending by `(a, b)` with
    /// `a < b`. The contact buffer is cleared and refilled in place.
    pub fn detect(
        &mut self,
        positions: &[Vec3],
        radii: &[f32],
        margin: f32,
    ) -> Option<&[SphereContact]> {
        let n = positions.len();
        if radii.len() != n {
            return None;
        }
        if !margin.is_finite() || margin < 0.0 {
            return None;
        }
        for (pos, &r) in positions.iter().zip(radii.iter()) {
            if !pos.is_finite() || !r.is_finite() || r <= 0.0 {
                return None;
            }
        }

        let candidates = self.broadphase.candidate_pairs(positions, radii, margin)?;

        self.contacts.clear();
        for &(a, b) in candidates.iter() {
            let ai = a as usize;
            let bi = b as usize;
            let delta = positions[bi] - positions[ai];
            let distance = delta.length();
            let sum_radii = radii[ai] + radii[bi];
            let penetration = sum_radii - distance;
            // The broad phase is conservative, so re-test the exact gap here.
            if distance > sum_radii + margin {
                continue;
            }

            let normal = if distance > COINCIDENT_EPSILON {
                delta / distance
            } else {
                // Coincident centres: pick a deterministic fallback axis.
                Vec3::X
            };
            // Midplane of the overlap on the a -> b axis.
            let contact_point = positions[ai] + normal * (radii[ai] - 0.5 * penetration);
            self.contacts.push(SphereContact {
                a,
                b,
                normal,
                penetration,
                contact_point,
            });
        }

        Some(&self.contacts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference O(n^2) narrow phase used to validate the accelerated one.
    fn brute_force(positions: &[Vec3], radii: &[f32], margin: f32) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                let gap = (positions[j] - positions[i]).length() - radii[i] - radii[j];
                if gap <= margin {
                    out.push((i as u32, j as u32));
                }
            }
        }
        out
    }

    #[test]
    fn new_is_empty() {
        let np = SphereNarrowPhase::new();
        assert_eq!(np.contact_count(), 0);
        assert_eq!(np.max_penetration(), 0.0);
    }

    #[test]
    fn rejects_invalid_input() {
        let mut np = SphereNarrowPhase::new();
        let positions = vec![Vec3::ZERO, Vec3::X];
        assert!(np.detect(&positions, &[1.0], 0.0).is_none());
        assert!(np.detect(&positions, &[1.0, 0.0], 0.0).is_none());
        assert!(np.detect(&positions, &[1.0, 1.0], -0.1).is_none());
        let bad = vec![Vec3::ZERO, Vec3::new(f32::NAN, 0.0, 0.0)];
        assert!(np.detect(&bad, &[1.0, 1.0], 0.0).is_none());
    }

    #[test]
    fn overlapping_pair_reports_contact() {
        let mut np = SphereNarrowPhase::new();
        // Unit spheres whose centres are 1.5 apart overlap by 0.5.
        let positions = vec![Vec3::ZERO, Vec3::new(1.5, 0.0, 0.0)];
        let radii = vec![1.0, 1.0];
        let contacts = np.detect(&positions, &radii, 0.0).unwrap();
        assert_eq!(contacts.len(), 1);
        let c = contacts[0];
        assert_eq!((c.a, c.b), (0, 1));
        assert!((c.normal - Vec3::X).length() < 1.0e-6);
        assert!((c.penetration - 0.5).abs() < 1.0e-6);
        // Contact midplane sits 0.75 from grain 0 along +x.
        assert!((c.contact_point - Vec3::new(0.75, 0.0, 0.0)).length() < 1.0e-6);
        assert!((np.max_penetration() - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn separated_pair_has_no_contact() {
        let mut np = SphereNarrowPhase::new();
        let positions = vec![Vec3::ZERO, Vec3::new(2.5, 0.0, 0.0)];
        let radii = vec![1.0, 1.0];
        assert_eq!(np.detect(&positions, &radii, 0.0).unwrap().len(), 0);
    }

    #[test]
    fn margin_reports_near_pair_with_negative_penetration() {
        let mut np = SphereNarrowPhase::new();
        // Gap of 0.2; a 0.3 margin should catch it.
        let positions = vec![Vec3::ZERO, Vec3::new(2.2, 0.0, 0.0)];
        let radii = vec![1.0, 1.0];
        let contacts = np.detect(&positions, &radii, 0.3).unwrap();
        assert_eq!(contacts.len(), 1);
        assert!((contacts[0].penetration + 0.2).abs() < 1.0e-6);
        // The same pair is excluded when the margin is too small.
        assert_eq!(np.detect(&positions, &radii, 0.1).unwrap().len(), 0);
    }

    #[test]
    fn coincident_centres_use_fallback_normal() {
        let mut np = SphereNarrowPhase::new();
        let positions = vec![Vec3::ZERO, Vec3::ZERO];
        let radii = vec![1.0, 1.0];
        let contacts = np.detect(&positions, &radii, 0.0).unwrap();
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].normal, Vec3::X);
        assert!((contacts[0].penetration - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn normal_points_from_a_to_b() {
        let mut np = SphereNarrowPhase::new();
        // Grain 1 is above grain 0 along +z.
        let positions = vec![Vec3::ZERO, Vec3::new(0.0, 0.0, 1.5)];
        let radii = vec![1.0, 1.0];
        let contacts = np.detect(&positions, &radii, 0.0).unwrap();
        assert_eq!(contacts.len(), 1);
        assert!((contacts[0].normal - Vec3::Z).length() < 1.0e-6);
    }

    #[test]
    fn matches_brute_force_on_cluster() {
        let mut np = SphereNarrowPhase::new();
        // A small 3x3 grid of overlapping unit spheres plus one far outlier.
        let mut positions = Vec::new();
        let spacing = 1.7_f32; // < 2r so neighbours overlap
        for ix in 0..3 {
            for iy in 0..3 {
                positions.push(Vec3::new(ix as f32 * spacing, iy as f32 * spacing, 0.0));
            }
        }
        positions.push(Vec3::new(100.0, 100.0, 100.0));
        let radii = vec![1.0_f32; positions.len()];

        let margin = 0.0;
        let mut accel: Vec<(u32, u32)> = np
            .detect(&positions, &radii, margin)
            .unwrap()
            .iter()
            .map(|c| (c.a, c.b))
            .collect();
        accel.sort_unstable();
        let reference = brute_force(&positions, &radii, margin);
        assert_eq!(accel, reference);
        assert!(!accel.is_empty());
    }
}
