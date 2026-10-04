//! Persistent pendular-bridge bookkeeping across a cloud of wet grains.
//!
//! [`capillary_bridge`](super::capillary_bridge) gives the *force law* of a
//! single liquid bridge. A scene, though, needs to decide *which* grain pairs
//! are actually bridged, and that decision is history-dependent. Wet granular
//! media show a pronounced **formation/rupture hysteresis**: a bridge only
//! nucleates once two grains have come into contact (the liquid films must
//! touch), yet once formed it stretches and keeps pulling as the grains draw
//! apart, snapping only when the surface separation exceeds the rupture
//! distance `H_rupture`. Two grains sitting a hair's breadth apart that have
//! *never* touched feel nothing; the identical geometry reached by pulling a
//! freshly bonded pair apart feels the full capillary attraction.
//!
//! [`CapillaryBridgeResolver`] captures exactly that history. It remembers the
//! set of currently live bridges between calls and, on each [`resolve`] pass:
//!
//! * forms a new bridge for any pair whose cores touch (`gap ≤ 0`);
//! * keeps an existing bridge alive while `gap ≤ H_rupture`;
//! * ruptures (forgets) a bridge once `gap > H_rupture`.
//!
//! Every live bridge then contributes the central, momentum-conserving pull of
//! the force law to both grains. The resolver owns no integration state beyond
//! the live-bridge set, so it composes cleanly with any grain integrator.
//!
//! [`resolve`]: CapillaryBridgeResolver::resolve

use crate::collider::capillary_bridge::CapillaryBridgeModel;
use glam::Vec3;
use std::collections::HashSet;

/// A single live bridge reported by the resolver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CapillaryPairBridge {
    /// Lower grain index of the bonded pair.
    pub grain_a: u32,
    /// Higher grain index of the bonded pair.
    pub grain_b: u32,
    /// Attraction magnitude of the bridge (non-negative).
    pub magnitude: f32,
    /// Surface separation of the pair (negative when the cores overlap).
    pub gap: f32,
}

/// Outcome of a single resolve pass over the grain cloud.
#[derive(Clone, Debug, PartialEq)]
pub struct CapillaryResolution {
    forces: Vec<Vec3>,
    bridges: Vec<CapillaryPairBridge>,
}

impl CapillaryResolution {
    /// Net capillary force accumulated on each grain, indexed by grain.
    pub fn forces(&self) -> &[Vec3] {
        &self.forces
    }

    /// Every live bridge resolved this pass, in ascending `(grain_a, grain_b)`
    /// order.
    pub fn bridges(&self) -> &[CapillaryPairBridge] {
        &self.bridges
    }

    /// Number of live bridges.
    pub fn bridge_count(&self) -> usize {
        self.bridges.len()
    }

    /// Largest single-bridge attraction magnitude, or `0.0` when no bridge is
    /// live.
    pub fn max_force(&self) -> f32 {
        self.bridges
            .iter()
            .map(|b| b.magnitude)
            .fold(0.0_f32, f32::max)
    }

    /// Vector sum of every grain force. Capillary bridges are internal pair
    /// forces, so this is zero up to round-off — a conservation diagnostic.
    pub fn net_force(&self) -> Vec3 {
        self.forces.iter().copied().fold(Vec3::ZERO, |a, f| a + f)
    }
}

/// Tracks the live pendular bridges of a grain cloud across resolve passes.
#[derive(Clone, Debug, Default)]
pub struct CapillaryBridgeResolver {
    active: HashSet<(u32, u32)>,
}

impl CapillaryBridgeResolver {
    /// Creates a resolver with no live bridges.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolves the capillary forces of the grain cloud, updating the live-bridge
    /// set with formation and rupture hysteresis.
    ///
    /// Returns `None` when `positions` and `radii` disagree in length, a
    /// position is non-finite, or a radius is non-positive or non-finite.
    pub fn resolve(
        &mut self,
        positions: &[Vec3],
        radii: &[f32],
        model: &CapillaryBridgeModel,
    ) -> Option<CapillaryResolution> {
        let n = positions.len();
        if radii.len() != n {
            return None;
        }
        for i in 0..n {
            if !positions[i].is_finite() || !radii[i].is_finite() || radii[i] <= 0.0 {
                return None;
            }
        }
        // Drop any live bridge that references a grain the cloud no longer has.
        self.active
            .retain(|&(a, b)| (a as usize) < n && (b as usize) < n);

        let rupture = model.rupture_distance();
        let mut forces = vec![Vec3::ZERO; n];
        let mut bridges = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let key = (i as u32, j as u32);
                let offset = positions[j] - positions[i];
                let distance = offset.length();
                if !distance.is_finite() || distance <= 0.0 {
                    // Coincident grains: no well-defined bridge; drop any stale one.
                    self.active.remove(&key);
                    continue;
                }
                let gap = distance - (radii[i] + radii[j]);
                let was_active = self.active.contains(&key);
                let live = if was_active {
                    // Persist until the bridge stretches past rupture.
                    gap <= rupture
                } else {
                    // Nucleate only once the cores actually touch.
                    gap <= 0.0
                };
                if !live {
                    if was_active {
                        self.active.remove(&key);
                    }
                    continue;
                }
                let Some(bridge) = model.bridge(positions[i], radii[i], positions[j], radii[j])
                else {
                    self.active.remove(&key);
                    continue;
                };
                self.active.insert(key);
                forces[i] += bridge.force;
                forces[j] -= bridge.force;
                bridges.push(CapillaryPairBridge {
                    grain_a: key.0,
                    grain_b: key.1,
                    magnitude: bridge.magnitude,
                    gap,
                });
            }
        }

        Some(CapillaryResolution { forces, bridges })
    }

    /// Number of live bridges currently tracked.
    pub fn active_bridges(&self) -> usize {
        self.active.len()
    }

    /// Whether a specific grain pair is currently bridged (order-independent).
    pub fn is_bridged(&self, grain_a: u32, grain_b: u32) -> bool {
        let key = if grain_a <= grain_b {
            (grain_a, grain_b)
        } else {
            (grain_b, grain_a)
        };
        self.active.contains(&key)
    }

    /// Forgets every live bridge.
    pub fn clear(&mut self) {
        self.active.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CapillaryBridgeModel {
        CapillaryBridgeModel::new(0.072, 0.0, 1.0e-9).unwrap()
    }

    #[test]
    fn resolve_validates_inputs() {
        let mut r = CapillaryBridgeResolver::new();
        // Length mismatch.
        assert!(r.resolve(&[Vec3::ZERO], &[0.5, 0.5], &model()).is_none());
        // Non-positive radius.
        assert!(r.resolve(&[Vec3::ZERO], &[0.0], &model()).is_none());
        // Non-finite position.
        assert!(r
            .resolve(&[Vec3::new(f32::NAN, 0.0, 0.0)], &[0.5], &model())
            .is_none());
    }

    #[test]
    fn untouched_grains_feel_nothing() {
        let mut r = CapillaryBridgeResolver::new();
        // Two grains a small gap apart that have never touched.
        let positions = vec![Vec3::ZERO, Vec3::new(1.0 + 1.0e-3, 0.0, 0.0)];
        let radii = vec![0.5, 0.5];
        let res = r.resolve(&positions, &radii, &model()).unwrap();
        assert_eq!(res.bridge_count(), 0);
        assert!(res.forces()[0].length() < 1.0e-9);
        assert!(res.forces()[1].length() < 1.0e-9);
        assert_eq!(r.active_bridges(), 0);
    }

    #[test]
    fn bridge_forms_on_contact_and_pulls() {
        let mut r = CapillaryBridgeResolver::new();
        // Exactly touching.
        let positions = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let radii = vec![0.5, 0.5];
        let res = r.resolve(&positions, &radii, &model()).unwrap();
        assert_eq!(res.bridge_count(), 1);
        assert!(r.is_bridged(0, 1));
        assert!(r.is_bridged(1, 0));
        // Grain 0 is pulled toward grain 1 (+x), grain 1 toward 0 (-x).
        assert!(res.forces()[0].x > 0.0);
        assert!(res.forces()[1].x < 0.0);
        // Internal pair force conserves momentum.
        assert!(res.net_force().length() < 1.0e-5);
    }

    #[test]
    fn bridge_persists_while_stretching_then_ruptures() {
        let mut r = CapillaryBridgeResolver::new();
        let radii = vec![0.5, 0.5];
        let rupture = model().rupture_distance();
        // Form the bridge on contact.
        let _ = r
            .resolve(&[Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)], &radii, &model())
            .unwrap();
        assert!(r.is_bridged(0, 1));
        // Pull apart to a gap within the rupture distance: the bridge persists.
        let stretched = vec![Vec3::ZERO, Vec3::new(1.0 + rupture * 0.5, 0.0, 0.0)];
        let res = r.resolve(&stretched, &radii, &model()).unwrap();
        assert_eq!(res.bridge_count(), 1);
        assert!(r.is_bridged(0, 1));
        // Pull apart beyond rupture: the bridge snaps.
        let snapped = vec![Vec3::ZERO, Vec3::new(1.0 + rupture * 1.5, 0.0, 0.0)];
        let res = r.resolve(&snapped, &radii, &model()).unwrap();
        assert_eq!(res.bridge_count(), 0);
        assert!(!r.is_bridged(0, 1));
    }

    #[test]
    fn ruptured_bridge_does_not_reform_without_contact() {
        let mut r = CapillaryBridgeResolver::new();
        let radii = vec![0.5, 0.5];
        let rupture = model().rupture_distance();
        // Form, then rupture.
        let _ = r
            .resolve(&[Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)], &radii, &model())
            .unwrap();
        let _ = r
            .resolve(
                &[Vec3::ZERO, Vec3::new(1.0 + rupture * 1.5, 0.0, 0.0)],
                &radii,
                &model(),
            )
            .unwrap();
        assert!(!r.is_bridged(0, 1));
        // Approach again to a small positive gap (no contact): no reformation.
        let near = vec![Vec3::ZERO, Vec3::new(1.0 + rupture * 0.5, 0.0, 0.0)];
        let res = r.resolve(&near, &radii, &model()).unwrap();
        assert_eq!(res.bridge_count(), 0);
        assert!(!r.is_bridged(0, 1));
    }

    #[test]
    fn bridges_reported_in_ascending_order() {
        let mut r = CapillaryBridgeResolver::new();
        // Three grains in a touching row along x.
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let radii = vec![0.5, 0.5, 0.5];
        let res = r.resolve(&positions, &radii, &model()).unwrap();
        // Neighbours (0,1) and (1,2) touch; (0,2) are two diameters apart.
        assert_eq!(res.bridge_count(), 2);
        let b = res.bridges();
        assert_eq!((b[0].grain_a, b[0].grain_b), (0, 1));
        assert_eq!((b[1].grain_a, b[1].grain_b), (1, 2));
        assert!(res.max_force() > 0.0);
    }

    #[test]
    fn clear_forgets_every_bridge() {
        let mut r = CapillaryBridgeResolver::new();
        let positions = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let radii = vec![0.5, 0.5];
        let _ = r.resolve(&positions, &radii, &model()).unwrap();
        assert_eq!(r.active_bridges(), 1);
        r.clear();
        assert_eq!(r.active_bridges(), 0);
        assert!(!r.is_bridged(0, 1));
    }

    #[test]
    fn stale_bridges_pruned_when_cloud_shrinks() {
        let mut r = CapillaryBridgeResolver::new();
        let radii3 = vec![0.5, 0.5, 0.5];
        let positions3 = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let _ = r.resolve(&positions3, &radii3, &model()).unwrap();
        assert_eq!(r.active_bridges(), 2);
        // Resolve a smaller cloud: the (1,2) bridge references a dropped grain.
        let radii2 = vec![0.5, 0.5];
        let positions2 = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let res = r.resolve(&positions2, &radii2, &model()).unwrap();
        assert_eq!(res.bridge_count(), 1);
        assert_eq!(r.active_bridges(), 1);
        assert!(r.is_bridged(0, 1));
    }
}
