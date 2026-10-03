//! Concrete adapters that read `prism_physics_core` contact facts.
//!
//! The translator is engine-agnostic; this module is the one place that knows
//! the concrete [`prism_physics_core`] types. It converts a native
//! [`prism_physics_core::collide::ContactManifold`] into a
//! [`crate::contact_input::ContactManifoldView`], maps a
//! [`prism_physics_core::events::PhysicsEvent`] into a contact key and lifecycle
//! phase (collisions sound; triggers are silent sensors), and packs a
//! generational [`prism_physics_core::state::handle::BodyHandle`] into a stable
//! [`crate::body::BodyAudioId`]. It is compiled only under the `physics-core`
//! feature so the core crate stays free of a hard physics dependency.
//!
//! The physics world and `bevy_math` share the same `glam` vector type, so the
//! witness points are copied component-wise into [`bevy_math::Vec3`] with no
//! lossy conversion.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the engine-binding edge of design section 47.1: it feeds
//! [`crate::translator::ContactAudioTranslator`] from the rigid-body truth
//! exposed by `prism_physics_core`.

use bevy_math::Vec3;

use prism_physics_core::collide::ContactManifold;
use prism_physics_core::events::PhysicsEvent;
use prism_physics_core::state::handle::BodyHandle;

use crate::body::BodyAudioId;
use crate::contact_id::ContactKey;
use crate::contact_input::{ContactManifoldView, ContactPhase, ContactPointView};

/// Packs a generational body handle into a stable audio-body id.
///
/// The slot index occupies the high 32 bits and the generation the low 32 bits,
/// so distinct handles map to distinct ids and a reused slot (bumped
/// generation) yields a fresh id rather than aliasing the old body.
#[inline]
#[must_use]
pub fn body_id(h: BodyHandle) -> BodyAudioId {
    let packed = ((h.index() as u64) << 32) | (h.generation() as u64);
    BodyAudioId(packed)
}

/// Builds an engine-agnostic manifold view from a physics-core manifold.
///
/// The manifold normal and each contact point's `point_a` witness and
/// penetration are copied component-wise into the view; the body ids are
/// supplied by the caller (typically via [`body_id`]) so this adapter stays
/// free of storage lookups.
#[inline]
#[must_use]
pub fn view_from_manifold(
    m: &ContactManifold,
    body_a: BodyAudioId,
    body_b: BodyAudioId,
) -> ContactManifoldView {
    let mut points = Vec::with_capacity(m.points().len());
    for p in m.points() {
        points.push(ContactPointView::new(
            Vec3::new(p.point_a.x, p.point_a.y, p.point_a.z),
            p.penetration,
        ));
    }
    let normal = Vec3::new(m.normal.x, m.normal.y, m.normal.z);
    ContactManifoldView::new(body_a, body_b, normal, points)
}

/// Maps a physics event into a contact key and audio lifecycle phase.
///
/// Collision start/end become [`ContactPhase::Started`] / [`ContactPhase::Ended`];
/// trigger enter/exit return `None` because sensor overlaps resolve no force
/// and make no sound. The contact key is normalised so it matches the
/// translator's persistent map regardless of body order.
#[inline]
#[must_use]
pub fn phase_from_event(ev: &PhysicsEvent) -> Option<(ContactKey, ContactPhase)> {
    match ev {
        PhysicsEvent::CollisionStarted(pair) => {
            let key = ContactKey::new(body_id(pair.first()), body_id(pair.second()));
            Some((key, ContactPhase::Started))
        }
        PhysicsEvent::CollisionEnded(pair) => {
            let key = ContactKey::new(body_id(pair.first()), body_id(pair.second()));
            Some((key, ContactPhase::Ended))
        }
        PhysicsEvent::TriggerEntered { .. } | PhysicsEvent::TriggerExited { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_physics_core::collide::contact::ContactPoint as PcContactPoint;
    use prism_physics_core::events::ContactPair;
    use prism_physics_core::state::body::BodyDesc;
    use prism_physics_core::state::storage::BodyStorage;

    #[test]
    fn view_maps_normal_and_points() {
        let mut m = ContactManifold::new(BodyHandle::INVALID, BodyHandle::INVALID, Vec3::Y);
        m.push(PcContactPoint::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 1.0, 3.0),
            0.5,
        ));
        let view = view_from_manifold(&m, BodyAudioId(10), BodyAudioId(20));
        assert_eq!(view.body_a, BodyAudioId(10));
        assert_eq!(view.body_b, BodyAudioId(20));
        assert_eq!(view.normal, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(view.points.len(), 1);
        assert_eq!(view.points[0].world_point, Vec3::new(1.0, 2.0, 3.0));
        assert!((view.points[0].penetration - 0.5).abs() < 1e-6);
    }

    #[test]
    fn body_id_packs_index_and_generation() {
        let mut storage = BodyStorage::new();
        let h0 = storage.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let h1 = storage.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        assert_ne!(body_id(h0), body_id(h1));
        // Packing is stable for the same handle.
        assert_eq!(body_id(h0), body_id(h0));
    }

    #[test]
    fn collision_events_map_to_phases() {
        let mut storage = BodyStorage::new();
        let a = storage.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let b = storage.insert(BodyDesc::dynamic_at(Vec3::X));
        let pair = ContactPair::new(a, b);

        let (key_start, phase_start) =
            phase_from_event(&PhysicsEvent::CollisionStarted(pair)).unwrap();
        assert_eq!(phase_start, ContactPhase::Started);

        let (key_end, phase_end) = phase_from_event(&PhysicsEvent::CollisionEnded(pair)).unwrap();
        assert_eq!(phase_end, ContactPhase::Ended);
        // Same manifold maps to the same key for start and end.
        assert_eq!(key_start, key_end);
    }

    #[test]
    fn triggers_are_silent() {
        let mut storage = BodyStorage::new();
        let sensor = storage.insert(BodyDesc::dynamic_at(Vec3::ZERO));
        let other = storage.insert(BodyDesc::dynamic_at(Vec3::X));
        assert!(phase_from_event(&PhysicsEvent::TriggerEntered { sensor, other }).is_none());
        assert!(phase_from_event(&PhysicsEvent::TriggerExited { sensor, other }).is_none());
    }
}
