//! Finite per-grain liquid budgets and their distribution across capillary
//! bridges.
//!
//! The existing [`CapillaryBridgeModel`](crate::collider::capillary_bridge::CapillaryBridgeModel)
//! assigns every pendular bridge the same fixed liquid volume. Real wet
//! granular media instead carry a *finite* amount of liquid that is shared out
//! among the bridges a grain participates in: a grain wetting many neighbours
//! spreads its liquid thinner, so each of its bridges is smaller (and weaker)
//! than it would be in isolation. This module provides the bookkeeping for
//! that finite-liquid picture as a standalone, verifiable building block.
//!
//! # Distribution rule
//!
//! Each grain holds a liquid volume. Given the set of currently live bridges,
//! a grain splits its liquid *equally* among the bridges that touch it (its
//! degree). A bridge between grains `a` and `b` then receives
//! `V_a / deg(a) + V_b / deg(b)`. This conserves liquid: the total volume
//! handed to bridges equals the summed liquid of every grain that has at least
//! one bridge, while grains with no bridges simply retain their liquid.
//!
//! A downstream driver can feed each bridge's volume into a per-bridge
//! [`CapillaryBridgeModel`](crate::collider::capillary_bridge::CapillaryBridgeModel)
//! to obtain a cohesion force that reflects the real, shared liquid budget
//! rather than a global constant. This module performs only the volume
//! accounting; it deliberately holds no force or geometry state.

/// Per-grain liquid volumes with a bridge-distribution rule.
#[derive(Clone, Debug, PartialEq)]
pub struct LiquidDistribution {
    /// Liquid volume held by each grain. Always finite and non-negative.
    volumes: Vec<f32>,
}

impl LiquidDistribution {
    /// Build from explicit per-grain volumes.
    ///
    /// Returns `None` unless every volume is finite and non-negative.
    pub fn new(volumes: Vec<f32>) -> Option<Self> {
        if volumes.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return None;
        }
        Some(Self { volumes })
    }

    /// Build a distribution where every grain starts with the same volume.
    ///
    /// Returns `None` if `per_grain` is not finite and non-negative.
    pub fn uniform(grain_count: usize, per_grain: f32) -> Option<Self> {
        if !per_grain.is_finite() || per_grain < 0.0 {
            return None;
        }
        Some(Self {
            volumes: vec![per_grain; grain_count],
        })
    }

    /// Number of grains tracked.
    pub fn grain_count(&self) -> usize {
        self.volumes.len()
    }

    /// Per-grain liquid volumes.
    pub fn volumes(&self) -> &[f32] {
        &self.volumes
    }

    /// Total liquid summed over all grains.
    pub fn total_liquid(&self) -> f32 {
        self.volumes.iter().sum()
    }

    /// Overwrite grain `i`'s liquid volume.
    ///
    /// Returns `false` (leaving the state unchanged) if `i` is out of range or
    /// `volume` is not finite and non-negative.
    pub fn set_volume(&mut self, i: usize, volume: f32) -> bool {
        if i >= self.volumes.len() || !volume.is_finite() || volume < 0.0 {
            return false;
        }
        self.volumes[i] = volume;
        true
    }

    /// Degree (number of incident bridges) of every grain under `bridges`.
    ///
    /// Returns `None` if any bridge references a grain out of range or is a
    /// self-bridge (`a == b`).
    pub fn degrees(&self, bridges: &[(u32, u32)]) -> Option<Vec<u32>> {
        let n = self.volumes.len();
        let mut degree = vec![0u32; n];
        for &(a, b) in bridges {
            let (au, bu) = (a as usize, b as usize);
            if au >= n || bu >= n || au == bu {
                return None;
            }
            degree[au] += 1;
            degree[bu] += 1;
        }
        Some(degree)
    }

    /// Liquid volume assigned to each bridge, in the same order as `bridges`.
    ///
    /// Each grain's liquid is split equally among its incident bridges; a
    /// bridge receives the sum of the shares from both of its grains. Returns
    /// `None` if any bridge references a grain out of range or is a
    /// self-bridge.
    pub fn bridge_volumes(&self, bridges: &[(u32, u32)]) -> Option<Vec<f32>> {
        let degree = self.degrees(bridges)?;
        let mut volumes = Vec::with_capacity(bridges.len());
        for &(a, b) in bridges {
            let (au, bu) = (a as usize, b as usize);
            // Both grains have degree >= 1 here (they are in this bridge).
            let share_a = self.volumes[au] / degree[au] as f32;
            let share_b = self.volumes[bu] / degree[bu] as f32;
            volumes.push(share_a + share_b);
        }
        Some(volumes)
    }

    /// Total liquid currently committed to bridges under `bridges`.
    ///
    /// Equals the summed liquid of every grain with at least one bridge.
    /// Returns `None` on an invalid bridge set.
    pub fn committed_liquid(&self, bridges: &[(u32, u32)]) -> Option<f32> {
        Some(self.bridge_volumes(bridges)?.iter().sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_validates_volumes() {
        assert!(LiquidDistribution::new(vec![0.0, 1.0, 2.5]).is_some());
        assert!(LiquidDistribution::new(vec![-0.1]).is_none());
        assert!(LiquidDistribution::new(vec![f32::NAN]).is_none());
        assert!(LiquidDistribution::new(vec![f32::INFINITY]).is_none());
    }

    #[test]
    fn uniform_builder() {
        let d = LiquidDistribution::uniform(4, 2.0).unwrap();
        assert_eq!(d.grain_count(), 4);
        assert_eq!(d.volumes(), &[2.0, 2.0, 2.0, 2.0]);
        assert_eq!(d.total_liquid(), 8.0);
        assert!(LiquidDistribution::uniform(2, -1.0).is_none());
    }

    #[test]
    fn set_volume_guards_inputs() {
        let mut d = LiquidDistribution::uniform(2, 1.0).unwrap();
        assert!(d.set_volume(0, 5.0));
        assert_eq!(d.volumes()[0], 5.0);
        assert!(!d.set_volume(2, 1.0)); // out of range
        assert!(!d.set_volume(1, -1.0)); // negative
        assert!(!d.set_volume(1, f32::NAN)); // non-finite
                                             // Unchanged after rejected writes.
        assert_eq!(d.volumes()[1], 1.0);
    }

    #[test]
    fn degrees_count_incident_bridges() {
        let d = LiquidDistribution::uniform(3, 1.0).unwrap();
        // Chain 0-1-2.
        let deg = d.degrees(&[(0, 1), (1, 2)]).unwrap();
        assert_eq!(deg, vec![1, 2, 1]);
    }

    #[test]
    fn single_bridge_gets_both_grain_volumes() {
        let d = LiquidDistribution::new(vec![1.0, 3.0]).unwrap();
        let vols = d.bridge_volumes(&[(0, 1)]).unwrap();
        // Each grain has degree 1, so the bridge receives all of both.
        assert_eq!(vols, vec![4.0]);
    }

    #[test]
    fn shared_grain_splits_its_liquid() {
        let d = LiquidDistribution::uniform(3, 1.0).unwrap();
        // Chain 0-1-2: grain 1 has degree 2 and splits its unit of liquid.
        let vols = d.bridge_volumes(&[(0, 1), (1, 2)]).unwrap();
        // bridge(0,1) = 1/1 + 1/2 = 1.5; bridge(1,2) = 1/2 + 1/1 = 1.5.
        assert_eq!(vols, vec![1.5, 1.5]);
    }

    #[test]
    fn liquid_is_conserved_over_bridged_grains() {
        // Grains 0..4 with assorted volumes; grain 4 has no bridge.
        let d = LiquidDistribution::new(vec![2.0, 1.0, 3.0, 0.5, 9.0]).unwrap();
        let bridges = [(0, 1), (1, 2), (2, 3), (0, 2)];
        let committed = d.committed_liquid(&bridges).unwrap();
        // All of grains 0..=3 is committed; grain 4 (no bridge) retains its 9.
        let expected = 2.0 + 1.0 + 3.0 + 0.5;
        assert!(
            (committed - expected).abs() < 1e-5,
            "committed {committed} != expected {expected}"
        );
        assert!(committed <= d.total_liquid());
    }

    #[test]
    fn empty_bridges_commit_nothing() {
        let d = LiquidDistribution::uniform(3, 1.0).unwrap();
        assert_eq!(d.bridge_volumes(&[]).unwrap(), Vec::<f32>::new());
        assert_eq!(d.committed_liquid(&[]).unwrap(), 0.0);
    }

    #[test]
    fn invalid_bridges_are_rejected() {
        let d = LiquidDistribution::uniform(2, 1.0).unwrap();
        // Out-of-range grain index.
        assert!(d.bridge_volumes(&[(0, 2)]).is_none());
        assert!(d.degrees(&[(0, 2)]).is_none());
        // Self-bridge.
        assert!(d.bridge_volumes(&[(1, 1)]).is_none());
        assert!(d.degrees(&[(1, 1)]).is_none());
    }

    #[test]
    fn dry_grain_contributes_zero() {
        // Grain 0 is dry (0 liquid), grain 1 wet.
        let d = LiquidDistribution::new(vec![0.0, 2.0]).unwrap();
        let vols = d.bridge_volumes(&[(0, 1)]).unwrap();
        assert_eq!(vols, vec![2.0]);
    }
}
