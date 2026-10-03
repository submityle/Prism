//! Importing an EIF scene into the engine's runtime source and parameter model.
//!
//! [`import_scene`] is the asset-time bridge from a declarative
//! [`crate::eif::scene::EifScene`] to the engine's runtime truth: each source
//! becomes a [`prism_audio_spatial::geometry::Emitter`] plus a
//! [`prism_audio_spatial::spatializer::SourceDescriptor`], resolved against the
//! current listener into the shared [`SpatialParams`] bus; each geometry
//! primitive becomes a `(bounds, absorption)` record ready to be inserted into
//! the engine's shared bounding-volume hierarchy with its section 14 material.
//!
//! This path allocates and is intended for the asset / task-thread stage, not
//! the real-time audio callback (per section 49.1: EIF parsing and geometry
//! preprocessing happen at asset time; the RT thread only does table lookups,
//! smoothing, and sample DSP).
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The import rules are a straightforward mapping from the public EIF scene
//! structure (MPEG-I Immersive Audio, ISO/IEC 23090-4, section 49.1) onto the
//! engine's existing source/parameter buses.
//!
//! # Relationship
//!
//! Reads [`crate::eif::scene::EifScene`] and reuses
//! [`prism_audio_spatial::spatializer::resolve`] with the standard
//! [`prism_audio_spatial::occlusion::OcclusionFactors::OPEN`] clear-path
//! factors to produce control-rate [`SpatialParams`]. The geometry records use
//! [`prism_audio_spatial::material_library::MaterialAbsorption`], the same
//! section 14 absorption view the rest of the engine consumes.

use alloc::vec::Vec;

use prism_audio_core::math::Sample;
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::material_library::MaterialAbsorption;
use prism_audio_spatial::occlusion::OcclusionFactors;
use prism_audio_spatial::spatializer::{resolve, SourceDescriptor, SpatialParams};

use crate::eif::geometry::Aabb;
use crate::eif::scene::EifScene;
use crate::eif::source::{EifSource, EifSourceId, EifSourceKind};

/// One source after import: its identity and class, the geometry/descriptor the
/// engine needs to spatialise it, and the control-rate targets resolved against
/// the supplied listener.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedSource {
    /// The scene source id.
    pub id: EifSourceId,
    /// The source class (object / channel / HOA).
    pub kind: EifSourceKind,
    /// The emitter geometry (position + facing; zero velocity at import time).
    pub emitter: Emitter,
    /// The resolved spatialisation descriptor (distance curve from the EIF
    /// source; neutral cone/doppler/occlusion/atmosphere until the runtime
    /// supplies live factors).
    pub descriptor: SourceDescriptor,
    /// The control-rate spatial targets for the import-time listener pose.
    pub params: SpatialParams,
    /// The source's overall linear gain, folded in separately from the
    /// distance-driven [`SpatialParams::direct_gain`].
    pub gain: Sample,
}

/// One geometry primitive after import: its world bounds and the engine-shared
/// absorption spectrum of its material, ready for BVH insertion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImportedGeometry {
    /// The world-space bounds used to place the primitive in the shared BVH.
    pub bounds: Aabb,
    /// The section 14 absorption spectrum of the primitive's material.
    pub absorption: MaterialAbsorption,
}

/// The result of importing a whole scene.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportedScene {
    /// The imported sources, in scene order.
    pub sources: Vec<ImportedSource>,
    /// The imported geometry records, in scene order.
    pub geometry: Vec<ImportedGeometry>,
}

/// Builds the spatialisation descriptor for one EIF source: its distance
/// attenuation is carried over verbatim; the remaining models stay at their
/// neutral defaults until the runtime feeds live occlusion/doppler factors.
fn descriptor_for(source: &EifSource) -> SourceDescriptor {
    SourceDescriptor {
        attenuation: source.attenuation,
        ..SourceDescriptor::default()
    }
}

/// Imports a single source against `listener` at `sample_rate`.
#[must_use]
pub fn import_source(source: &EifSource, listener: &Listener, sample_rate: u32) -> ImportedSource {
    let emitter = Emitter::new(source.position, bevy_math::Vec3::ZERO, source.facing());
    let descriptor = descriptor_for(source);
    let params = resolve(
        listener,
        &emitter,
        &descriptor,
        OcclusionFactors::OPEN,
        sample_rate,
    );
    ImportedSource {
        id: source.id,
        kind: source.kind.clone(),
        emitter,
        descriptor,
        params,
        gain: source.sanitized_gain(),
    }
}

/// Imports every source and geometry primitive of `scene`, resolving sources
/// against `listener` at `sample_rate`.
///
/// Geometry primitives whose material reference is out of range are skipped
/// (the scene should be validated with
/// [`EifScene::validate`](crate::eif::scene::EifScene::validate) first); this
/// keeps the import total rather than panicking on a malformed scene.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_immersive::eif::import::import_scene;
/// use prism_audio_immersive::eif::scene::EifScene;
/// use prism_audio_immersive::eif::source::{EifSource, EifSourceId};
/// use prism_audio_spatial::geometry::Listener;
///
/// let mut scene = EifScene::new();
/// scene.add_source(EifSource::object(EifSourceId(1), Vec3::new(0.0, 0.0, -3.0)));
/// let imported = import_scene(&scene, &Listener::default(), 48_000);
/// assert_eq!(imported.sources.len(), 1);
/// // A source dead ahead resolves to roughly zero azimuth.
/// assert!(imported.sources[0].params.azimuth.abs() < 1e-3);
/// ```
#[must_use]
pub fn import_scene(scene: &EifScene, listener: &Listener, sample_rate: u32) -> ImportedScene {
    let mut sources = Vec::with_capacity(scene.sources.len());
    for source in &scene.sources {
        sources.push(import_source(source, listener, sample_rate));
    }

    let mut geometry = Vec::with_capacity(scene.geometry.len());
    for primitive in &scene.geometry {
        if let Some(material) = scene.material(primitive.material()) {
            geometry.push(ImportedGeometry {
                bounds: primitive.bounds(),
                absorption: material.as_absorption(),
            });
        }
    }

    ImportedScene { sources, geometry }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eif::geometry::{AcousticBox, GeometryPrimitive, MaterialRef};
    use crate::eif::material::AcousticMaterial;
    use bevy_math::{ops, Vec3};

    const EPS: Sample = 1e-3;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn source_ahead_imports_to_zero_azimuth() {
        let mut scene = EifScene::new();
        scene.add_source(EifSource::object(EifSourceId(1), Vec3::new(0.0, 0.0, -4.0)));
        let imported = import_scene(&scene, &Listener::default(), 48_000);
        assert_eq!(imported.sources.len(), 1);
        assert!(close(imported.sources[0].params.azimuth, 0.0));
        assert!(imported.sources[0].params.direct_gain > 0.0);
    }

    #[test]
    fn geometry_carries_material_absorption() {
        let mut scene = EifScene::new();
        let mat = scene.add_material(AcousticMaterial::broadband(0.4, 0.1, 0.0));
        scene.add_geometry(GeometryPrimitive::Box(AcousticBox::new(
            Aabb::from_corners(Vec3::ZERO, Vec3::splat(2.0)),
            mat,
        )));
        let imported = import_scene(&scene, &Listener::default(), 48_000);
        assert_eq!(imported.geometry.len(), 1);
        assert!(close(imported.geometry[0].absorption.bands()[0], 0.4));
    }

    #[test]
    fn dangling_geometry_is_skipped_not_panicked() {
        let mut scene = EifScene::new();
        scene.add_geometry(GeometryPrimitive::Box(AcousticBox::new(
            Aabb::from_corners(Vec3::ZERO, Vec3::ONE),
            MaterialRef(9),
        )));
        let imported = import_scene(&scene, &Listener::default(), 48_000);
        assert!(imported.geometry.is_empty());
    }

    #[test]
    fn source_gain_is_preserved() {
        let mut scene = EifScene::new();
        scene.add_source(
            EifSource::object(EifSourceId(2), Vec3::new(0.0, 0.0, -2.0)).with_gain(0.5),
        );
        let imported = import_scene(&scene, &Listener::default(), 48_000);
        assert!(close(imported.sources[0].gain, 0.5));
    }
}
