//! Exporting an engine object/bed scene to the declarative EIF model.
//!
//! [`export_object_scene`] turns a [`prism_audio_object::scene::ObjectScene`]
//! (the engine's bed-plus-objects runtime model) into a
//! [`crate::eif::scene::EifScene`] that a third party can validate for standard
//! compliance or load into another renderer. Bed channels become channel
//! sources placed along their canonical loudspeaker directions, and each audio
//! object becomes an object source at its world position.
//!
//! Export is the inverse direction of [`crate::eif::import`]: it does not try
//! to reconstruct geometry or materials (the engine's object scene carries
//! none), only the source layer that EIF and the object model share.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The mapping follows the public EIF source taxonomy (MPEG-I Immersive Audio,
//! ISO/IEC 23090-4, section 49.1) and the engine's own bed/object model.
//!
//! # Relationship
//!
//! Reads [`prism_audio_object::scene::ObjectScene`] and
//! [`prism_audio_object::bed::BedLayout`], producing a
//! [`crate::eif::scene::EifScene`] of [`crate::eif::source::EifSource`] entries
//! only. Complements [`crate::eif::import`].

use alloc::vec::Vec;

use prism_audio_core::math::Sample;
use prism_audio_object::bed::BedLayout;
use prism_audio_object::object::AudioObject;
use prism_audio_object::scene::ObjectScene;

use crate::eif::scene::EifScene;
use crate::eif::source::{EifSource, EifSourceId};

/// The reference distance (metres) at which bed channel sources are placed
/// along their loudspeaker directions.
const BED_RADIUS_M: Sample = 2.0;

/// Exports a single [`AudioObject`] as an object [`EifSource`].
#[must_use]
pub fn export_object(object: &AudioObject) -> EifSource {
    EifSource::object(EifSourceId(object.id.get()), object.position).with_gain(object.gain)
}

/// Exports a bed layout as a set of channel [`EifSource`] entries, one per
/// directional loudspeaker (the LFE is skipped: it carries no direction).
///
/// Ids are assigned sequentially starting at `first_id`.
#[must_use]
pub fn export_bed(bed: BedLayout, first_id: u32) -> Vec<EifSource> {
    let mut sources = Vec::new();
    let mut next_id = first_id;
    for channel in bed.channels() {
        if channel.is_lfe {
            continue;
        }
        let position = channel.direction * BED_RADIUS_M;
        let source = EifSource::channel(EifSourceId(next_id), channel.label, position)
            .with_forward(-channel.direction);
        sources.push(source);
        next_id += 1;
    }
    sources
}

/// Exports a whole [`ObjectScene`] to an [`EifScene`].
///
/// Bed channels are exported first (ids `0..n`), then the scene's objects at
/// their current positions. The resulting scene carries sources only; its
/// geometry, material, portal, and reverb tables are left empty.
///
/// # Examples
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_immersive::eif::export::export_object_scene;
/// use prism_audio_object::bed::BedLayout;
/// use prism_audio_object::object::{AudioObject, ObjectId};
/// use prism_audio_object::scene::ObjectScene;
///
/// let mut scene = ObjectScene::new(BedLayout::Stereo);
/// scene.add_object(AudioObject::new(ObjectId(7), Vec3::new(1.0, 0.0, -2.0), 0.8));
/// let eif = export_object_scene(&scene);
/// // Two stereo bed channels plus one object.
/// assert_eq!(eif.sources.len(), 3);
/// ```
#[must_use]
pub fn export_object_scene(scene: &ObjectScene) -> EifScene {
    let mut eif = EifScene::new();

    let bed_sources = export_bed(scene.bed(), 0);
    let bed_count = bed_sources.len() as u32;
    for source in bed_sources {
        eif.add_source(source);
    }

    for object in scene.current_objects() {
        let mut source = export_object(&object);
        // Keep object ids disjoint from the bed channel id block.
        source.id = EifSourceId(bed_count + object.id.get());
        eif.add_source(source);
    }

    eif
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eif::source::EifSourceKind;
    use bevy_math::{ops, Vec3};
    use prism_audio_object::object::ObjectId;

    const EPS: Sample = 1e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    #[test]
    fn bed_export_skips_lfe_and_counts_directional() {
        let sources = export_bed(BedLayout::Surround5_1_4, 0);
        assert_eq!(sources.len(), BedLayout::Surround5_1_4.directional_count());
        assert!(
            sources
                .iter()
                .all(|s| matches!(s.kind, EifSourceKind::Channel { .. }))
        );
    }

    #[test]
    fn object_export_preserves_position_and_gain() {
        let obj = AudioObject::new(ObjectId(3), Vec3::new(1.0, 2.0, -3.0), 0.7);
        let src = export_object(&obj);
        assert!(close(src.position.x, 1.0));
        assert!(close(src.position.z, -3.0));
        assert!(close(src.gain, 0.7));
    }

    #[test]
    fn scene_export_counts_bed_plus_objects() {
        let mut scene = ObjectScene::new(BedLayout::Stereo);
        scene.add_object(AudioObject::new(ObjectId(1), Vec3::new(1.0, 0.0, -2.0), 0.8));
        scene.add_object(AudioObject::new(ObjectId(2), Vec3::new(-1.0, 0.0, -2.0), 0.6));
        let eif = export_object_scene(&scene);
        assert_eq!(eif.sources.len(), 2 + 2);
    }

    #[test]
    fn bed_channel_positions_lie_on_reference_radius() {
        let sources = export_bed(BedLayout::Surround7_1_4, 0);
        for s in &sources {
            let r = ops::sqrt(
                s.position.x * s.position.x
                    + s.position.y * s.position.y
                    + s.position.z * s.position.z,
            );
            assert!(close(r, BED_RADIUS_M));
        }
    }
}
