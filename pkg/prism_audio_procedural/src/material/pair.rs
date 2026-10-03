//! Material identity and acoustic category for contact synthesis.
//!
//! A sounding contact is defined by the two materials that collide. This module
//! holds the lightweight identifiers used throughout the crate: a [`MaterialId`]
//! naming a specific authored material, a [`MaterialCategory`] giving the broad
//! acoustic family a material falls back to when it has no bespoke table, and a
//! symmetric [`MaterialPairId`] that keys the per-pair synthesis parameters.
//! The pair id is order-independent (steel-on-stone equals stone-on-steel), so
//! the lookup never depends on which body the solver listed first.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Backs the acoustic material coupling of design section 47.5; the categories
//! mirror the field bus shared with spatial-acoustic materials
//! (`prism_material_pipeline`) without depending on its API, and key the
//! lookups in [`crate::material::lookup`].

/// Identifier of a specific authored material.
///
/// Wraps a 16-bit index into the host's material registry. The crate never
/// interprets the number beyond equality and pairing; its acoustic meaning is
/// resolved through [`crate::material::lookup`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MaterialId(pub u16);

/// Broad acoustic family a material belongs to.
///
/// When a specific material pair has no bespoke table, synthesis falls back to
/// the category pair so there is never a silent "hole" in the material space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum MaterialCategory {
    /// Hard, bright, long-ringing (steel, iron, aluminium).
    Metal,
    /// Mid-bright, moderate decay (oak, pine, plywood).
    Wood,
    /// Dense, dull, short decay (granite, concrete, brick).
    Stone,
    /// Very bright, high modes, medium decay (window glass, bottles).
    Glass,
    /// Soft, damped, dark (ABS, polythene, rubber).
    Plastic,
    /// Bright but quick-damped (porcelain, tile).
    Ceramic,
    /// Highly damped, almost tuneless (cloth, carpet, foam).
    Fabric,
    /// Non-resonant granular splatter (water, mud).
    Liquid,
    /// Catch-all used when a material declares no category.
    Generic,
}

impl MaterialCategory {
    /// The number of distinct categories, for building fallback tables.
    pub const COUNT: usize = 9;

    /// Returns a stable `0..COUNT` index for this category.
    #[inline]
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            MaterialCategory::Metal => 0,
            MaterialCategory::Wood => 1,
            MaterialCategory::Stone => 2,
            MaterialCategory::Glass => 3,
            MaterialCategory::Plastic => 4,
            MaterialCategory::Ceramic => 5,
            MaterialCategory::Fabric => 6,
            MaterialCategory::Liquid => 7,
            MaterialCategory::Generic => 8,
        }
    }
}

impl Default for MaterialCategory {
    #[inline]
    fn default() -> Self {
        MaterialCategory::Generic
    }
}

/// Order-independent identifier of a colliding material pair.
///
/// Construction normalises the two ids so `MaterialPairId::new(a, b)` equals
/// `MaterialPairId::new(b, a)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MaterialPairId {
    lo: u16,
    hi: u16,
}

impl MaterialPairId {
    /// Builds a symmetric pair id from two raw material indices.
    #[inline]
    #[must_use]
    pub fn new(a: u16, b: u16) -> Self {
        if a <= b {
            Self { lo: a, hi: b }
        } else {
            Self { lo: b, hi: a }
        }
    }

    /// Builds a pair id from two [`MaterialId`]s.
    #[inline]
    #[must_use]
    pub fn from_ids(a: MaterialId, b: MaterialId) -> Self {
        Self::new(a.0, b.0)
    }

    /// Returns the lower raw id of the pair.
    #[inline]
    #[must_use]
    pub fn lo(self) -> u16 {
        self.lo
    }

    /// Returns the higher raw id of the pair.
    #[inline]
    #[must_use]
    pub fn hi(self) -> u16 {
        self.hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_is_symmetric() {
        assert_eq!(MaterialPairId::new(3, 7), MaterialPairId::new(7, 3));
    }

    #[test]
    fn pair_keeps_both_ids() {
        let p = MaterialPairId::new(7, 3);
        assert_eq!(p.lo(), 3);
        assert_eq!(p.hi(), 7);
    }

    #[test]
    fn category_indices_are_unique() {
        let cats = [
            MaterialCategory::Metal,
            MaterialCategory::Wood,
            MaterialCategory::Stone,
            MaterialCategory::Glass,
            MaterialCategory::Plastic,
            MaterialCategory::Ceramic,
            MaterialCategory::Fabric,
            MaterialCategory::Liquid,
            MaterialCategory::Generic,
        ];
        let mut seen = [false; MaterialCategory::COUNT];
        for c in cats {
            assert!(!seen[c.index()]);
            seen[c.index()] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
