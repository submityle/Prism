//! The pluggable platform spatial backend extension point.
//!
//! [`PlatformSpatialBackend`] is the trait a platform/device integration
//! implements to describe its spatial renderer and turn an engine object set
//! into a [`DeliveryPlan`]. The default [`PlatformSpatialBackend::deliver`] and
//! [`PlatformSpatialBackend::deliver_scene`] route through
//! [`negotiate_delivery`], so an implementation normally only supplies its
//! [`PlatformCapability`]. [`ProfileBackend`] is the ready-made implementation
//! that wraps a named [`PlatformProfile`].
//!
//! The trait produces only the control-rate delivery description; the actual
//! operating-system handoff lives in the platform/device layer, which consumes
//! the returned [`DeliveryPlan`].
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Realises the platform-backend extension point of design section 16 and the
//! engine's extension-point registry (`PlatformSpatialBackend`). Built on
//! [`crate::platform::delivery`] and [`crate::platform::profile`].

use crate::object::AudioObject;
use crate::platform::capability::PlatformCapability;
use crate::platform::delivery::{negotiate_delivery, DeliveryPlan};
use crate::platform::profile::PlatformProfile;
use crate::scene::ObjectScene;

/// A platform spatial renderer the engine can deliver objects to.
pub trait PlatformSpatialBackend {
    /// Returns the rendering contract this backend exposes.
    fn capability(&self) -> PlatformCapability;

    /// Negotiates `objects` into a [`DeliveryPlan`] for this backend.
    #[must_use]
    fn deliver(&self, objects: &[AudioObject]) -> DeliveryPlan {
        negotiate_delivery(objects, self.capability())
    }

    /// Negotiates the current objects of `scene` into a [`DeliveryPlan`].
    ///
    /// Callers should [`ObjectScene::advance_to`] the desired time first.
    #[must_use]
    fn deliver_scene(&self, scene: &ObjectScene) -> DeliveryPlan {
        negotiate_delivery(&scene.current_objects(), self.capability())
    }
}

/// A backend defined by a named [`PlatformProfile`] with its default
/// capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ProfileBackend {
    /// The profile whose default capability this backend reports.
    profile: PlatformProfile,
}

impl ProfileBackend {
    /// Creates a backend for the given platform `profile`.
    #[must_use]
    pub const fn new(profile: PlatformProfile) -> Self {
        Self { profile }
    }

    /// Returns the profile this backend wraps.
    #[must_use]
    pub const fn profile(self) -> PlatformProfile {
        self.profile
    }
}

impl PlatformSpatialBackend for ProfileBackend {
    fn capability(&self) -> PlatformCapability {
        self.profile.capability()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bed::{direction_from_angles, BedLayout};
    use crate::object::ObjectId;
    use crate::platform::capability::ObjectRenderMode;

    fn obj(id: u32, az: f32) -> AudioObject {
        AudioObject::new(ObjectId(id), direction_from_angles(az, 0.0), 1.0)
    }

    #[test]
    fn profile_backend_reports_profile_capability() {
        let backend = ProfileBackend::new(PlatformProfile::AtmosRenderer);
        assert_eq!(backend.profile(), PlatformProfile::AtmosRenderer);
        assert_eq!(
            backend.capability().object_mode,
            ObjectRenderMode::DiscreteObjects
        );
    }

    #[test]
    fn deliver_routes_through_negotiation() {
        let backend = ProfileBackend::new(PlatformProfile::StereoHeadphones);
        let objects = [obj(0, 0.0), obj(1, 90.0)];
        let plan = backend.deliver(&objects);
        assert_eq!(plan.object_count, 2);
        assert!(plan.channel_order.is_empty());
    }

    #[test]
    fn deliver_scene_uses_current_objects() {
        let backend = ProfileBackend::new(PlatformProfile::ChannelBed(BedLayout::Surround5_1_4));
        let mut scene = ObjectScene::new(BedLayout::Stereo);
        scene.add_object(obj(0, 0.0));
        scene.add_object(obj(1, 45.0));
        scene.advance_to(0.0);

        let plan = backend.deliver_scene(&scene);
        assert_eq!(plan.object_count, 2);
        // The platform bed (from the capability) overrides the scene bed.
        assert_eq!(plan.capability.bed, BedLayout::Surround5_1_4);
    }
}
