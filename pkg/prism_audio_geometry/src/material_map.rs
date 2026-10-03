//! Per-triangle acoustic-material assignment.
//!
//! A [`MaterialTable`] maps each triangle of a scene mesh to an
//! [`AcousticMaterial`](prism_audio_spatial::propagation::AcousticMaterial)
//! through a small palette: a list of distinct materials plus one palette index
//! per triangle. This keeps a large mesh that reuses a handful of surface types
//! compact while still letting every triangle carry its own acoustics. Any
//! triangle without an explicit assignment, and any out-of-range index, resolve
//! to a caller-chosen default, so the table never leaves an acoustic hole.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Held by [`crate::scene::AcousticScene`] and read by every path builder to
//! fetch the [`AcousticMaterial`](prism_audio_spatial::propagation::AcousticMaterial)
//! governing a ray hit.

use alloc::vec;
use alloc::vec::Vec;

use prism_audio_spatial::propagation::AcousticMaterial;

/// Maps triangle indices to acoustic materials via a shared palette.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MaterialTable {
    palette: Vec<AcousticMaterial>,
    /// One palette index per triangle. Entries past `assignments.len()` or past
    /// the palette resolve to `default`.
    assignments: Vec<u32>,
    default: AcousticMaterial,
}

impl MaterialTable {
    /// Builds a table where every triangle resolves to `default` until an
    /// explicit assignment is added. The default is seeded as palette slot 0.
    #[inline]
    #[must_use]
    pub fn uniform(default: AcousticMaterial) -> Self {
        Self {
            palette: vec![default],
            assignments: Vec::new(),
            default,
        }
    }

    /// Builds a table from an explicit palette, a per-triangle index list, and
    /// a fallback material used for unassigned or out-of-range triangles.
    #[inline]
    #[must_use]
    pub fn from_palette(
        palette: Vec<AcousticMaterial>,
        assignments: Vec<u32>,
        default: AcousticMaterial,
    ) -> Self {
        Self {
            palette,
            assignments,
            default,
        }
    }

    /// Interns `material` into the palette (reusing an identical existing slot)
    /// and returns its palette index.
    #[must_use]
    pub fn intern(&mut self, material: AcousticMaterial) -> u32 {
        for (slot, existing) in self.palette.iter().enumerate() {
            if *existing == material {
                return slot as u32;
            }
        }
        let slot = self.palette.len() as u32;
        self.palette.push(material);
        slot
    }

    /// Assigns `material` to `triangle`, growing the assignment list with the
    /// default palette slot (0) for any triangles skipped before it.
    pub fn assign(&mut self, triangle: usize, material: AcousticMaterial) {
        let slot = self.intern(material);
        if triangle >= self.assignments.len() {
            self.assignments.resize(triangle + 1, 0);
        }
        self.assignments[triangle] = slot;
    }

    /// Returns the material governing `triangle`, falling back to the table's
    /// default when the triangle is unassigned or its palette index is stale.
    #[inline]
    #[must_use]
    pub fn material(&self, triangle: usize) -> AcousticMaterial {
        match self.assignments.get(triangle) {
            Some(&slot) => self
                .palette
                .get(slot as usize)
                .copied()
                .unwrap_or(self.default),
            None => self.default,
        }
    }

    /// The fallback material used for unassigned or out-of-range triangles.
    #[inline]
    #[must_use]
    pub fn default_material(&self) -> AcousticMaterial {
        self.default
    }

    /// Number of distinct materials currently in the palette.
    #[inline]
    #[must_use]
    pub fn palette_len(&self) -> usize {
        self.palette.len()
    }
}

impl Default for MaterialTable {
    /// A table whose every triangle resolves to
    /// [`AcousticMaterial::OPEN`](prism_audio_spatial::propagation::AcousticMaterial::OPEN).
    #[inline]
    fn default() -> Self {
        Self::uniform(AcousticMaterial::OPEN)
    }
}

#[cfg(test)]
mod tests {
    use super::MaterialTable;
    use alloc::vec;
    use prism_audio_spatial::propagation::AcousticMaterial;

    #[test]
    fn uniform_resolves_everywhere() {
        let m = AcousticMaterial::new(20.0, 0.5);
        let table = MaterialTable::uniform(m);
        assert_eq!(table.material(0), m);
        assert_eq!(table.material(10_000), m);
    }

    #[test]
    fn assign_overrides_single_triangle() {
        let base = AcousticMaterial::OPEN;
        let wall = AcousticMaterial::new(30.0, 0.8);
        let mut table = MaterialTable::uniform(base);
        table.assign(5, wall);
        assert_eq!(table.material(5), wall);
        // Triangles skipped before index 5 keep palette slot 0 (the default).
        assert_eq!(table.material(0), base);
        // Out-of-range still falls back.
        assert_eq!(table.material(99), base);
    }

    #[test]
    fn intern_deduplicates() {
        let mut table = MaterialTable::uniform(AcousticMaterial::OPEN);
        let a = table.intern(AcousticMaterial::new(12.0, 0.3));
        let b = table.intern(AcousticMaterial::new(12.0, 0.3));
        assert_eq!(a, b);
        // OPEN(slot 0) + the one distinct material.
        assert_eq!(table.palette_len(), 2);
    }

    #[test]
    fn from_palette_resolves_and_falls_back() {
        let open = AcousticMaterial::OPEN;
        let wall = AcousticMaterial::new(25.0, 0.6);
        let table = MaterialTable::from_palette(vec![open, wall], vec![1, 0, 9], open);
        assert_eq!(table.material(0), wall);
        assert_eq!(table.material(1), open);
        // Stale palette index 9 -> default.
        assert_eq!(table.material(2), open);
        // Beyond the assignment list -> default.
        assert_eq!(table.material(3), open);
    }
}
