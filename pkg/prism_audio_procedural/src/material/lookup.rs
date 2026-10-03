//! Material-pair to synthesis-parameter lookup with category fallback.
//!
//! A contact's timbre is decided by the pair of materials that collide. This
//! module turns a [`MaterialPairId`] into a [`MaterialPairProfile`] (a modal
//! table plus a friction template) with no holes: a specific authored pair can
//! be registered to override the default, and any unregistered pair falls back
//! to the acoustic category of each material and, ultimately, to the generic
//! category. Mode tables are synthesised deterministically from compact
//! per-category acoustic descriptors (fundamental, decay, inharmonicity,
//! brightness), so the same pair always yields the same table and the whole
//! material space is covered.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The inharmonic
//! mode-series construction and the material-combination heuristics are
//! original classic-DSP parameter mappings.
//!
//! # Relationship
//! Implements the acoustic material coupling of design section 47.5; produces
//! [`crate::modal::bank::Mode`] tables for [`crate::modal`] and friction
//! templates for [`crate::continuous::friction`]. Standalone lookup layer so the
//! crate does not hard-depend on `prism_material_pipeline`.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::ops;

use crate::material::pair::{MaterialCategory, MaterialId, MaterialPairId};
use crate::modal::bank::{Mode, MAX_MODES};
use prism_audio_core::math::Sample;

/// Compact acoustic descriptor for a material category.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CategoryAcoustics {
    /// Fundamental (lowest mode) frequency in hertz.
    pub fundamental_hz: Sample,
    /// Amplitude half-life of the fundamental in seconds.
    pub half_life_s: Sample,
    /// Inharmonicity: how far upper modes stretch above a harmonic series.
    pub inharmonicity: Sample,
    /// Default brightness in `[0, 1]`; biases high-mode gains.
    pub brightness: Sample,
    /// Friction noise spectral centroid in hertz at unit speed.
    pub friction_centroid_hz: Sample,
    /// Rolling pulse density (grains per metre of travel).
    pub rolling_density: Sample,
    /// Overall linear level for the category.
    pub level: Sample,
}

/// Friction/rolling template produced for a contact.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct FrictionTemplate {
    /// Base linear gain of the friction noise at unit tangential speed.
    pub base_gain: Sample,
    /// Noise spectral centroid in hertz at unit tangential speed.
    pub centroid_hz: Sample,
    /// Noise bandwidth in hertz at full roughness.
    pub bandwidth_hz: Sample,
    /// Rolling pulse density (grains per metre of travel).
    pub rolling_density: Sample,
}

/// The resolved synthesis parameters for a material pair.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialPairProfile {
    /// The modal table for the colliding bodies.
    pub modes: Vec<Mode>,
    /// The friction/rolling template.
    pub friction: FrictionTemplate,
    /// The combined fundamental frequency, exposed for pitch-tracking rolls.
    pub fundamental_hz: Sample,
}

/// Returns the built-in acoustic descriptor for a category.
#[must_use]
pub fn category_acoustics(category: MaterialCategory) -> CategoryAcoustics {
    match category {
        MaterialCategory::Metal => CategoryAcoustics {
            fundamental_hz: 320.0,
            half_life_s: 1.6,
            inharmonicity: 0.08,
            brightness: 0.85,
            friction_centroid_hz: 4_200.0,
            rolling_density: 90.0,
            level: 1.0,
        },
        MaterialCategory::Wood => CategoryAcoustics {
            fundamental_hz: 180.0,
            half_life_s: 0.18,
            inharmonicity: 0.03,
            brightness: 0.45,
            friction_centroid_hz: 1_600.0,
            rolling_density: 60.0,
            level: 0.8,
        },
        MaterialCategory::Stone => CategoryAcoustics {
            fundamental_hz: 140.0,
            half_life_s: 0.08,
            inharmonicity: 0.12,
            brightness: 0.35,
            friction_centroid_hz: 2_000.0,
            rolling_density: 110.0,
            level: 0.9,
        },
        MaterialCategory::Glass => CategoryAcoustics {
            fundamental_hz: 620.0,
            half_life_s: 0.9,
            inharmonicity: 0.15,
            brightness: 0.95,
            friction_centroid_hz: 6_000.0,
            rolling_density: 70.0,
            level: 0.85,
        },
        MaterialCategory::Plastic => CategoryAcoustics {
            fundamental_hz: 240.0,
            half_life_s: 0.12,
            inharmonicity: 0.05,
            brightness: 0.4,
            friction_centroid_hz: 1_800.0,
            rolling_density: 50.0,
            level: 0.6,
        },
        MaterialCategory::Ceramic => CategoryAcoustics {
            fundamental_hz: 540.0,
            half_life_s: 0.5,
            inharmonicity: 0.1,
            brightness: 0.8,
            friction_centroid_hz: 5_000.0,
            rolling_density: 80.0,
            level: 0.8,
        },
        MaterialCategory::Fabric => CategoryAcoustics {
            fundamental_hz: 90.0,
            half_life_s: 0.04,
            inharmonicity: 0.02,
            brightness: 0.2,
            friction_centroid_hz: 900.0,
            rolling_density: 30.0,
            level: 0.4,
        },
        MaterialCategory::Liquid => CategoryAcoustics {
            fundamental_hz: 110.0,
            half_life_s: 0.03,
            inharmonicity: 0.2,
            brightness: 0.5,
            friction_centroid_hz: 2_400.0,
            rolling_density: 140.0,
            level: 0.5,
        },
        MaterialCategory::Generic => CategoryAcoustics {
            fundamental_hz: 260.0,
            half_life_s: 0.25,
            inharmonicity: 0.06,
            brightness: 0.5,
            friction_centroid_hz: 2_600.0,
            rolling_density: 70.0,
            level: 0.7,
        },
    }
}

/// Combines two category descriptors into one effective descriptor.
///
/// The fundamental is the geometric mean (perceptually central), the half-life
/// takes the shorter of the two (the softer material damps the ring), and the
/// remaining fields average.
#[must_use]
fn combine(a: CategoryAcoustics, b: CategoryAcoustics) -> CategoryAcoustics {
    CategoryAcoustics {
        fundamental_hz: ops::sqrt(a.fundamental_hz * b.fundamental_hz),
        half_life_s: a.half_life_s.min(b.half_life_s),
        inharmonicity: 0.5 * (a.inharmonicity + b.inharmonicity),
        brightness: 0.5 * (a.brightness + b.brightness),
        friction_centroid_hz: ops::sqrt(a.friction_centroid_hz * b.friction_centroid_hz),
        rolling_density: 0.5 * (a.rolling_density + b.rolling_density),
        level: 0.5 * (a.level + b.level),
    }
}

/// Builds a modal table of `mode_count` modes from an effective descriptor.
///
/// Mode `k` sits at `fundamental * (k + 1) * (1 + inharmonicity * k)`, giving a
/// mildly stretched inharmonic series; its gain rolls off with `k` (tilted by
/// brightness) and its half-life shortens with `k` (high modes die first), both
/// standard for struck resonant bodies.
#[must_use]
pub fn build_modes(acoustics: CategoryAcoustics, mode_count: usize) -> Vec<Mode> {
    let count = mode_count.clamp(1, MAX_MODES);
    let mut modes = Vec::with_capacity(count);
    for k in 0..count {
        let kf = k as Sample;
        let ratio = (kf + 1.0) * (1.0 + acoustics.inharmonicity * kf);
        let freq = acoustics.fundamental_hz * ratio;
        // Brightness lifts the high-mode gain floor; darkness steepens roll-off.
        let rolloff = ops::powf(0.72, kf);
        let bright_gain = 1.0 + acoustics.brightness * kf * 0.12;
        let gain = acoustics.level * rolloff * bright_gain;
        // High modes decay faster: half-life shrinks geometrically.
        let half_life = (acoustics.half_life_s * ops::powf(0.82, kf)).max(0.01);
        modes.push(Mode::new(freq, half_life, gain));
    }
    modes
}

/// Builds a friction template from an effective descriptor.
#[must_use]
fn build_friction(acoustics: CategoryAcoustics) -> FrictionTemplate {
    FrictionTemplate {
        base_gain: acoustics.level * 0.5,
        centroid_hz: acoustics.friction_centroid_hz,
        bandwidth_hz: 0.6 * acoustics.friction_centroid_hz,
        rolling_density: acoustics.rolling_density,
    }
}

/// A specific authored override for one material pair.
#[derive(Clone, Debug)]
struct PairOverride {
    pair: MaterialPairId,
    profile: MaterialPairProfile,
}

/// The material lookup library: category assignments, optional pair overrides,
/// and the default mode count.
#[derive(Clone, Debug)]
pub struct MaterialLibrary {
    categories: Vec<(u16, MaterialCategory)>,
    overrides: Vec<PairOverride>,
    mode_count: usize,
}

impl MaterialLibrary {
    /// Creates an empty library producing `mode_count` modes per table.
    #[must_use]
    pub fn new(mode_count: usize) -> Self {
        Self {
            categories: Vec::new(),
            overrides: Vec::new(),
            mode_count: mode_count.clamp(1, MAX_MODES),
        }
    }

    /// Assigns `category` to the material `id` (last assignment wins).
    pub fn register_material(&mut self, id: MaterialId, category: MaterialCategory) {
        if let Some(slot) = self.categories.iter_mut().find(|(m, _)| *m == id.0) {
            slot.1 = category;
        } else {
            self.categories.push((id.0, category));
        }
    }

    /// Registers a fully authored profile for a specific pair, overriding the
    /// category-derived default.
    pub fn register_pair(&mut self, pair: MaterialPairId, profile: MaterialPairProfile) {
        if let Some(slot) = self.overrides.iter_mut().find(|o| o.pair == pair) {
            slot.profile = profile;
        } else {
            self.overrides.push(PairOverride { pair, profile });
        }
    }

    /// Returns the acoustic category assigned to a material, or
    /// [`MaterialCategory::Generic`] when none was registered.
    #[must_use]
    pub fn category_of(&self, id: u16) -> MaterialCategory {
        self.categories
            .iter()
            .find(|(m, _)| *m == id)
            .map(|(_, c)| *c)
            .unwrap_or(MaterialCategory::Generic)
    }

    /// Resolves the synthesis profile for a material pair.
    ///
    /// Registered overrides win; otherwise the profile is derived from the two
    /// materials' categories, so there is never a missing entry.
    #[must_use]
    pub fn profile(&self, pair: MaterialPairId) -> MaterialPairProfile {
        if let Some(found) = self.overrides.iter().find(|o| o.pair == pair) {
            return found.profile.clone();
        }
        let a = category_acoustics(self.category_of(pair.lo()));
        let b = category_acoustics(self.category_of(pair.hi()));
        let effective = combine(a, b);
        MaterialPairProfile {
            modes: build_modes(effective, self.mode_count),
            friction: build_friction(effective),
            fundamental_hz: effective.fundamental_hz,
        }
    }
}

impl Default for MaterialLibrary {
    #[inline]
    fn default() -> Self {
        Self::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unregistered_pair_has_profile() {
        let lib = MaterialLibrary::default();
        let p = lib.profile(MaterialPairId::new(1, 2));
        assert_eq!(p.modes.len(), 8);
        assert!(p.fundamental_hz > 0.0);
    }

    #[test]
    fn modes_ascend_in_frequency() {
        let lib = MaterialLibrary::default();
        let p = lib.profile(MaterialPairId::new(5, 9));
        for w in p.modes.windows(2) {
            assert!(w[1].freq_hz > w[0].freq_hz);
        }
    }

    #[test]
    fn categories_change_timbre() {
        let mut lib = MaterialLibrary::default();
        lib.register_material(MaterialId(1), MaterialCategory::Metal);
        lib.register_material(MaterialId(2), MaterialCategory::Fabric);
        let metal = lib.profile(MaterialPairId::new(1, 1));
        let fabric = lib.profile(MaterialPairId::new(2, 2));
        // Metal rings far longer than fabric.
        assert!(metal.modes[0].half_life_s > fabric.modes[0].half_life_s);
    }

    #[test]
    fn softer_material_damps_pair() {
        let mut lib = MaterialLibrary::default();
        lib.register_material(MaterialId(1), MaterialCategory::Metal);
        lib.register_material(MaterialId(2), MaterialCategory::Fabric);
        let mixed = lib.profile(MaterialPairId::new(1, 2));
        let metal = lib.profile(MaterialPairId::new(1, 1));
        assert!(mixed.modes[0].half_life_s <= metal.modes[0].half_life_s);
    }

    #[test]
    fn override_wins() {
        let mut lib = MaterialLibrary::default();
        let custom = MaterialPairProfile {
            modes: alloc::vec![Mode::new(1234.0, 2.0, 1.0)],
            friction: FrictionTemplate {
                base_gain: 1.0,
                centroid_hz: 3000.0,
                bandwidth_hz: 1000.0,
                rolling_density: 50.0,
            },
            fundamental_hz: 1234.0,
        };
        lib.register_pair(MaterialPairId::new(3, 4), custom);
        let p = lib.profile(MaterialPairId::new(4, 3));
        assert_eq!(p.modes.len(), 1);
        assert!((p.fundamental_hz - 1234.0).abs() < 1e-3);
    }

    #[test]
    fn profile_is_deterministic() {
        let lib = MaterialLibrary::default();
        assert_eq!(
            lib.profile(MaterialPairId::new(7, 11)),
            lib.profile(MaterialPairId::new(11, 7))
        );
    }
}
