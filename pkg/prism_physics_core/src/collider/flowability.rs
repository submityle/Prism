//! Bulk-powder flowability indices (Carr index and Hausner ratio).
//!
//! Granular flowability is routinely graded from two bulk measurements: the
//! *poured* (aerated) bulk density `ρ_b` and the *tapped* density `ρ_t`
//! obtained after settling the same mass by repeated tapping. The two densities
//! define the two classic compressibility descriptors
//!
//! ```text
//! Carr index   C  = 100 · (ρ_t − ρ_b) / ρ_t      (percent, 0 → incompressible)
//! Hausner ratio H  = ρ_t / ρ_b                     (≥ 1, 1 → free flowing)
//! ```
//!
//! which are algebraically linked by `H = 100 / (100 − C)`. Lower values mean a
//! looser packing collapses little under tapping, i.e. the powder flows freely;
//! larger values signal cohesion and poor flow. The thresholds used here follow
//! the widely tabulated USP flow-character scale. This module is a pure
//! constitutive correlation and does not couple to the simulation step.

/// Qualitative flow character inferred from the Carr index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowCharacter {
    /// Carr index `≤ 10`.
    Excellent,
    /// Carr index `11..=15`.
    Good,
    /// Carr index `16..=20`.
    Fair,
    /// Carr index `21..=25`.
    Passable,
    /// Carr index `26..=31`.
    Poor,
    /// Carr index `32..=37`.
    VeryPoor,
    /// Carr index `≥ 38`.
    ExtremelyPoor,
}

/// Bulk flowability descriptors derived from poured and tapped densities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowderFlowability {
    bulk_density: f32,
    tapped_density: f32,
    carr_index: f32,
    hausner_ratio: f32,
}

impl PowderFlowability {
    /// Builds the descriptors directly from the two bulk densities.
    ///
    /// * `bulk_density` — poured/aerated density `ρ_b` (`> 0`).
    /// * `tapped_density` — tapped density `ρ_t` (`≥ ρ_b`).
    ///
    /// Returns `None` for non-finite inputs, non-positive densities, or a
    /// tapped density below the poured density (physically the tapped state can
    /// only be denser or equal).
    pub fn from_densities(bulk_density: f32, tapped_density: f32) -> Option<Self> {
        if !bulk_density.is_finite() || bulk_density <= 0.0 {
            return None;
        }
        if !tapped_density.is_finite() || tapped_density <= 0.0 {
            return None;
        }
        if tapped_density < bulk_density {
            return None;
        }

        let carr_index = 100.0 * (tapped_density - bulk_density) / tapped_density;
        let hausner_ratio = tapped_density / bulk_density;
        if !carr_index.is_finite() || !hausner_ratio.is_finite() {
            return None;
        }

        Some(Self {
            bulk_density,
            tapped_density,
            carr_index,
            hausner_ratio,
        })
    }

    /// Builds the descriptors from a shared sample mass and the two measured
    /// volumes (poured `V_b` and tapped `V_t`).
    ///
    /// Densities are `ρ = m / V`, so the mass cancels in both indices; it only
    /// needs to be positive. Returns `None` for non-finite inputs, non-positive
    /// values, or a tapped volume exceeding the poured volume (tapping can only
    /// compact the bed).
    pub fn from_mass_volumes(mass: f32, poured_volume: f32, tapped_volume: f32) -> Option<Self> {
        if !mass.is_finite() || mass <= 0.0 {
            return None;
        }
        if !poured_volume.is_finite() || poured_volume <= 0.0 {
            return None;
        }
        if !tapped_volume.is_finite() || tapped_volume <= 0.0 {
            return None;
        }
        if tapped_volume > poured_volume {
            return None;
        }

        let bulk_density = mass / poured_volume;
        let tapped_density = mass / tapped_volume;
        Self::from_densities(bulk_density, tapped_density)
    }

    /// Poured/aerated bulk density `ρ_b`.
    pub fn bulk_density(&self) -> f32 {
        self.bulk_density
    }

    /// Tapped density `ρ_t`.
    pub fn tapped_density(&self) -> f32 {
        self.tapped_density
    }

    /// Carr compressibility index `C = 100 · (ρ_t − ρ_b)/ρ_t` (percent).
    pub fn carr_index(&self) -> f32 {
        self.carr_index
    }

    /// Hausner ratio `H = ρ_t / ρ_b` (`≥ 1`).
    pub fn hausner_ratio(&self) -> f32 {
        self.hausner_ratio
    }

    /// Qualitative flow character from the Carr index on the USP scale.
    pub fn flow_character(&self) -> FlowCharacter {
        let c = self.carr_index;
        if c <= 10.0 {
            FlowCharacter::Excellent
        } else if c <= 15.0 {
            FlowCharacter::Good
        } else if c <= 20.0 {
            FlowCharacter::Fair
        } else if c <= 25.0 {
            FlowCharacter::Passable
        } else if c <= 31.0 {
            FlowCharacter::Poor
        } else if c <= 37.0 {
            FlowCharacter::VeryPoor
        } else {
            FlowCharacter::ExtremelyPoor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_input() {
        assert!(PowderFlowability::from_densities(0.0, 1.0).is_none());
        assert!(PowderFlowability::from_densities(1.0, 0.0).is_none());
        assert!(PowderFlowability::from_densities(1.2, 1.0).is_none()); // tapped < bulk
        assert!(PowderFlowability::from_densities(f32::NAN, 1.0).is_none());
        assert!(PowderFlowability::from_densities(1.0, f32::INFINITY).is_none());
    }

    #[test]
    fn carr_and_hausner_known_values() {
        // ρ_b = 0.5, ρ_t = 0.6 → C = 100·0.1/0.6 = 16.666…, H = 1.2.
        let f = PowderFlowability::from_densities(0.5, 0.6).unwrap();
        assert!((f.carr_index() - 100.0 / 6.0).abs() < 1e-3);
        assert!((f.hausner_ratio() - 1.2).abs() < 1e-5);
    }

    #[test]
    fn incompressible_powder_is_excellent() {
        // Equal densities → C = 0, H = 1, best flow.
        let f = PowderFlowability::from_densities(0.7, 0.7).unwrap();
        assert_eq!(f.carr_index(), 0.0);
        assert_eq!(f.hausner_ratio(), 1.0);
        assert_eq!(f.flow_character(), FlowCharacter::Excellent);
    }

    #[test]
    fn carr_hausner_algebraic_link() {
        // H = 100 / (100 − C).
        let f = PowderFlowability::from_densities(0.4, 0.55).unwrap();
        let predicted = 100.0 / (100.0 - f.carr_index());
        assert!((f.hausner_ratio() - predicted).abs() < 1e-4);
    }

    #[test]
    fn mass_volume_matches_density_route() {
        // m = 2, V_b = 4 → ρ_b = 0.5; V_t = 10/3 → ρ_t = 0.6.
        let f = PowderFlowability::from_mass_volumes(2.0, 4.0, 2.0 / 0.6).unwrap();
        assert!((f.bulk_density() - 0.5).abs() < 1e-4);
        assert!((f.tapped_density() - 0.6).abs() < 1e-4);
        assert!((f.carr_index() - 100.0 / 6.0).abs() < 1e-2);
    }

    #[test]
    fn mass_volume_rejects_expansion() {
        // Tapped volume cannot exceed poured volume.
        assert!(PowderFlowability::from_mass_volumes(1.0, 2.0, 3.0).is_none());
        assert!(PowderFlowability::from_mass_volumes(0.0, 2.0, 1.0).is_none());
    }

    #[test]
    fn flow_character_boundaries() {
        let at = |c: f32| {
            // Pick ρ_t = 1, solve ρ_b from C = 100(1 − ρ_b).
            let bulk = 1.0 - c / 100.0;
            PowderFlowability::from_densities(bulk, 1.0)
                .unwrap()
                .flow_character()
        };
        assert_eq!(at(8.0), FlowCharacter::Excellent);
        assert_eq!(at(13.0), FlowCharacter::Good);
        assert_eq!(at(18.0), FlowCharacter::Fair);
        assert_eq!(at(23.0), FlowCharacter::Passable);
        assert_eq!(at(29.0), FlowCharacter::Poor);
        assert_eq!(at(35.0), FlowCharacter::VeryPoor);
        assert_eq!(at(45.0), FlowCharacter::ExtremelyPoor);
    }

    #[test]
    fn cohesive_powder_flows_worse_than_free_flowing() {
        let free = PowderFlowability::from_densities(0.90, 0.95).unwrap();
        let cohesive = PowderFlowability::from_densities(0.50, 0.80).unwrap();
        assert!(cohesive.carr_index() > free.carr_index());
        assert!(cohesive.hausner_ratio() > free.hausner_ratio());
    }
}
