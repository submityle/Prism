//! The authoritative Encoder Input Format (EIF) scene model.
//!
//! An [`EifScene`] is the declarative, renderer-independent description of a
//! six-degree-of-freedom acoustic world: a table of acoustic materials, the
//! geometry primitives that reference them, the sources that radiate into the
//! scene, the regions the listener may walk through, the portals that couple
//! those regions, and the reverberation zones that give each region its decay
//! character. It is the authoring-time truth that the importer folds onto the
//! engine's shared geometry and parameter buses.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The scene composition (geometry + materials + sources + walkable regions +
//! portals + reverb zones) is the publicly documented EIF scene structure
//! (MPEG-I Immersive Audio, ISO/IEC 23090-4, section 49.1), modelled here as
//! plain owned data with validation.
//!
//! # Relationship
//!
//! Aggregates [`crate::eif::material::AcousticMaterial`],
//! [`crate::eif::geometry::GeometryPrimitive`], and
//! [`crate::eif::source::EifSource`]. The portal and reverb-zone types mirror
//! the engine's section 14 room coupling and reverberation parameters. The
//! scene is consumed by [`crate::eif::import`] (scene to runtime nodes) and
//! produced by [`crate::eif::export`] (engine scene to EIF).

use alloc::vec::Vec;

use prism_audio_core::math::Sample;
use prism_audio_spatial::material_library::OCTAVE_BAND_COUNT;

use crate::eif::geometry::{Aabb, GeometryPrimitive, MaterialRef};
use crate::eif::material::AcousticMaterial;
use crate::eif::source::EifSource;

/// A portal (door, window, opening) coupling two walkable regions.
///
/// The portal references the two regions it joins by their index in
/// [`EifScene::walkable`] and carries the aperture rectangle through which
/// sound passes, plus a broadband transmission scalar in `[0, 1]` for a closed
/// or partially open portal.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EifPortal {
    /// Index of the first coupled region in [`EifScene::walkable`].
    pub region_a: usize,
    /// Index of the second coupled region in [`EifScene::walkable`].
    pub region_b: usize,
    /// The world-space aperture through which sound couples.
    pub aperture: Aabb,
    /// Broadband open fraction in `[0, 1]`: `1` is a fully open portal, `0` a
    /// sealed one.
    pub openness: Sample,
}

impl EifPortal {
    /// Builds a fully-open portal between two regions.
    #[must_use]
    pub fn new(region_a: usize, region_b: usize, aperture: Aabb) -> Self {
        Self {
            region_a,
            region_b,
            aperture,
            openness: 1.0,
        }
    }

    /// The sanitised open fraction, clamped into `[0, 1]`.
    #[must_use]
    pub fn clamped_openness(&self) -> Sample {
        if self.openness.is_finite() {
            self.openness.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// A reverberation zone: an axis-aligned region with a per-octave-band
/// reverberation time (RT60, in seconds).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EifReverbZone {
    /// The world-space extent of the zone.
    pub bounds: Aabb,
    /// Per-octave-band RT60 in seconds, aligned with the engine's octave grid.
    pub rt60: [Sample; OCTAVE_BAND_COUNT],
}

impl EifReverbZone {
    /// Builds a reverb zone from an extent and a per-band RT60, clamping each
    /// RT60 to be non-negative and finite.
    #[must_use]
    pub fn new(bounds: Aabb, rt60: [Sample; OCTAVE_BAND_COUNT]) -> Self {
        let mut clamped = [0.0; OCTAVE_BAND_COUNT];
        for (out, &v) in clamped.iter_mut().zip(rt60.iter()) {
            *out = if v.is_finite() { v.max(0.0) } else { 0.0 };
        }
        Self {
            bounds,
            rt60: clamped,
        }
    }

    /// Builds a reverb zone with the same RT60 in every band.
    #[must_use]
    pub fn broadband(bounds: Aabb, rt60_seconds: Sample) -> Self {
        Self::new(bounds, [rt60_seconds; OCTAVE_BAND_COUNT])
    }

    /// The mean RT60 across all bands.
    #[must_use]
    pub fn mean_rt60(&self) -> Sample {
        let mut sum = 0.0;
        for &v in &self.rt60 {
            sum += v;
        }
        sum / (OCTAVE_BAND_COUNT as Sample)
    }
}

/// An error describing why an [`EifScene`] failed validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SceneError {
    /// A geometry primitive referenced a material index outside the table.
    MaterialOutOfRange {
        /// The offending primitive index in [`EifScene::geometry`].
        primitive: usize,
        /// The out-of-range material index that was referenced.
        material: usize,
    },
    /// A portal referenced a walkable-region index outside the region list.
    PortalRegionOutOfRange {
        /// The offending portal index in [`EifScene::portals`].
        portal: usize,
        /// The out-of-range region index that was referenced.
        region: usize,
    },
}

/// The complete declarative EIF scene.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EifScene {
    /// The acoustic material table; geometry references entries by index.
    pub materials: Vec<AcousticMaterial>,
    /// The geometry primitives of the scene.
    pub geometry: Vec<GeometryPrimitive>,
    /// The sources radiating into the scene.
    pub sources: Vec<EifSource>,
    /// Axis-aligned regions the listener may occupy.
    pub walkable: Vec<Aabb>,
    /// Portals coupling pairs of walkable regions.
    pub portals: Vec<EifPortal>,
    /// Reverberation zones.
    pub reverb_zones: Vec<EifReverbZone>,
}

impl EifScene {
    /// An empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a material to the table, returning the [`MaterialRef`] that now
    /// addresses it.
    pub fn add_material(&mut self, material: AcousticMaterial) -> MaterialRef {
        let index = self.materials.len();
        self.materials.push(material);
        MaterialRef(index)
    }

    /// Adds a geometry primitive, returning its index.
    pub fn add_geometry(&mut self, primitive: GeometryPrimitive) -> usize {
        let index = self.geometry.len();
        self.geometry.push(primitive);
        index
    }

    /// Adds a source, returning its index.
    pub fn add_source(&mut self, source: EifSource) -> usize {
        let index = self.sources.len();
        self.sources.push(source);
        index
    }

    /// Adds a walkable region, returning its index.
    pub fn add_walkable(&mut self, region: Aabb) -> usize {
        let index = self.walkable.len();
        self.walkable.push(region);
        index
    }

    /// Adds a portal, returning its index.
    pub fn add_portal(&mut self, portal: EifPortal) -> usize {
        let index = self.portals.len();
        self.portals.push(portal);
        index
    }

    /// Adds a reverb zone, returning its index.
    pub fn add_reverb_zone(&mut self, zone: EifReverbZone) -> usize {
        let index = self.reverb_zones.len();
        self.reverb_zones.push(zone);
        index
    }

    /// Resolves a [`MaterialRef`] against the table.
    #[must_use]
    pub fn material(&self, reference: MaterialRef) -> Option<&AcousticMaterial> {
        self.materials.get(reference.index())
    }

    /// The axis-aligned bounds enclosing every geometry primitive and walkable
    /// region in the scene (an empty box when the scene has no geometry).
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        let mut bounds = Aabb::empty();
        for primitive in &self.geometry {
            bounds = bounds.merge(&primitive.bounds());
        }
        for region in &self.walkable {
            bounds = bounds.merge(region);
        }
        bounds
    }

    /// Finds the index of the first walkable region containing `point`.
    #[must_use]
    pub fn region_at(&self, point: bevy_math::Vec3) -> Option<usize> {
        self.walkable.iter().position(|r| r.contains(point))
    }

    /// Validates the scene's cross-references: every geometry primitive must
    /// reference a material inside the table, and every portal must reference
    /// walkable regions that exist.
    ///
    /// # Errors
    ///
    /// Returns the first [`SceneError`] encountered, or `Ok(())` when every
    /// reference is in range.
    pub fn validate(&self) -> Result<(), SceneError> {
        let material_count = self.materials.len();
        for (primitive_index, primitive) in self.geometry.iter().enumerate() {
            let material = primitive.material().index();
            if material >= material_count {
                return Err(SceneError::MaterialOutOfRange {
                    primitive: primitive_index,
                    material,
                });
            }
        }
        let region_count = self.walkable.len();
        for (portal_index, portal) in self.portals.iter().enumerate() {
            if portal.region_a >= region_count {
                return Err(SceneError::PortalRegionOutOfRange {
                    portal: portal_index,
                    region: portal.region_a,
                });
            }
            if portal.region_b >= region_count {
                return Err(SceneError::PortalRegionOutOfRange {
                    portal: portal_index,
                    region: portal.region_b,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eif::geometry::AcousticBox;
    use crate::eif::source::EifSourceId;
    use bevy_math::{ops, Vec3};

    const EPS: Sample = 1e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn room() -> EifScene {
        let mut scene = EifScene::new();
        let mat = scene.add_material(AcousticMaterial::broadband(0.2, 0.1, 0.02));
        scene.add_geometry(GeometryPrimitive::Box(AcousticBox::new(
            Aabb::from_corners(Vec3::ZERO, Vec3::splat(4.0)),
            mat,
        )));
        scene.add_source(EifSource::object(EifSourceId(1), Vec3::splat(2.0)));
        scene.add_walkable(Aabb::from_corners(Vec3::ZERO, Vec3::splat(4.0)));
        scene
    }

    #[test]
    fn valid_scene_passes_validation() {
        let scene = room();
        assert_eq!(scene.validate(), Ok(()));
    }

    #[test]
    fn dangling_material_is_rejected() {
        let mut scene = EifScene::new();
        scene.add_geometry(GeometryPrimitive::Box(AcousticBox::new(
            Aabb::from_corners(Vec3::ZERO, Vec3::ONE),
            MaterialRef(3),
        )));
        assert_eq!(
            scene.validate(),
            Err(SceneError::MaterialOutOfRange {
                primitive: 0,
                material: 3,
            })
        );
    }

    #[test]
    fn dangling_portal_region_is_rejected() {
        let mut scene = room();
        scene.add_portal(EifPortal::new(0, 5, Aabb::from_corners(Vec3::ZERO, Vec3::ONE)));
        assert_eq!(
            scene.validate(),
            Err(SceneError::PortalRegionOutOfRange {
                portal: 0,
                region: 5,
            })
        );
    }

    #[test]
    fn bounds_enclose_geometry_and_regions() {
        let scene = room();
        let b = scene.bounds();
        assert!(close(b.min.x, 0.0));
        assert!(close(b.max.x, 4.0));
    }

    #[test]
    fn region_lookup_finds_point() {
        let scene = room();
        assert_eq!(scene.region_at(Vec3::splat(1.0)), Some(0));
        assert_eq!(scene.region_at(Vec3::splat(-1.0)), None);
    }

    #[test]
    fn reverb_zone_clamps_and_averages() {
        let zone = EifReverbZone::new(
            Aabb::from_corners(Vec3::ZERO, Vec3::ONE),
            [0.5, -1.0, Sample::NAN, 0.5, 0.5, 0.5, 0.5, 0.5],
        );
        assert!(close(zone.rt60[1], 0.0));
        assert!(close(zone.rt60[2], 0.0));
        assert!(zone.mean_rt60() > 0.0);
    }

    #[test]
    fn portal_openness_clamps() {
        let p = EifPortal {
            region_a: 0,
            region_b: 1,
            aperture: Aabb::from_corners(Vec3::ZERO, Vec3::ONE),
            openness: 4.0,
        };
        assert!(close(p.clamped_openness(), 1.0));
    }
}
